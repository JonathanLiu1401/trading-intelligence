"""Trivial-baseline comparison — is the 17-feature MLP worth its complexity
*out of sample*, or would a one-line rule do just as well?

This is a **read-only diagnostic**. It never trains, never touches
`decision_scorer.pkl`, `decision_outcomes.jsonl`, `build_features`,
`N_FEATURES`, or any trade path — same operational discipline as
`paper_trader/ml/calibration.py` / `gate_audit.py` / `regime_audit.py`.

**Why this is not any existing tool.** The ML/backtest domain already has:

- `calibration.py` — *statistical*: is pred monotone with realized (deciles).
- `gate_audit.py` — *economic*: do the 5 fixed multipliers buy realized edge.
- `skill_trend.py` — *error-trend*: is `oos_rmse` better than a **constant**
  mean-predictor, cycle over cycle.
- `feature_importance.py` — *attribution*: which of the 17 inputs the trained
  MLP leans on.
- `regime_audit.py` — *regime-conditional*: is the (near-zero) OOS skill a
  regime-mix artifact.

`skill_trend` compares the MLP to the only-trivial-it-knows: a **constant**
predictor (σ(target) RMSE floor). **Nothing compares it to a non-constant
one-line rule** — raw `ml_score`, momentum carry, RSI mean-reversion. That is
the decisive question a skeptical quant asks once `gate_audit=GATE_INEFFECTIVE`
and `regime_audit=REGIME_UNIFORM_NULL` are on record: *is the neural net
extracting signal a single feature already carries, or is the 17-dim model
genuinely additive out of sample?* If a one-liner matches or beats the MLP's
OOS rank skill, the gate's complexity (and its sizing variance, invariant #5)
is unjustified by anything the MLP uniquely contributes.

**Method.** On the **temporal-OOS slice** (`validation.split_outcomes_temporal`
— the EXACT split `_train_decision_scorer` uses for `oos_rmse`/`oos_ic`, so
this and the ledger describe the *same* holdout), score every record with the
deployed MLP and with each trivial baseline, then compare on two
**scale-invariant** primitives (RMSE is unusable here — `mom20` predicts in a
different unit than the MLP's % forward return, so an RMSE race is decided by
scale, not skill):

- `rank_ic` — tie-aware Spearman, reusing `calibration._spearman` (single
  source of truth; tie-awareness load-bearing because the MLP clamps to
  ±`PRED_CLAMP_PCT` and naïve `argsort(argsort)` fabricates rank skill on the
  ±50 ties — a constant predictor would read 1.0).
- `dir_acc` — fraction of decisions where `sign(pred) == sign(realized)`.

The codebase-universal SELL sign-flip (`-forward_return_5d`) is applied to the
realized target for **every** predictor (exactly as `train_scorer` /
`calibration` / `gate_audit` / `_oos_rank_metrics`), so the MLP's `rank_ic`
here equals the value `calibration --oos` / the skill ledger report — a
built-in cross-check. Each trivial baseline's prediction is **also** flipped on
SELL so it is measured in the same "goodness of THIS action" space as the
target the (training-aligned) MLP already lives in; without that symmetry a
feature baseline would look spuriously anti-correlated on the SELL subset and
fabricate a false `MLP_ADDS_SKILL`.

| Verdict | Meaning |
|---------|---------|
| `INSUFFICIENT_DATA` | scorer untrained, or < `MIN_PAIRS` OOS pairs |
| `MLP_WORSE_THAN_TRIVIAL` | best non-degenerate baseline IC > MLP IC + `IC_MARGIN` — a one-liner beats the net |
| `MLP_NO_BETTER_THAN_TRIVIAL` | MLP does not clear best baseline + `IC_MARGIN` (or its own `MLP_IC_MIN` floor) — the 17-dim model is not additive OOS |
| `MLP_ADDS_SKILL` | MLP IC > best baseline IC + `IC_MARGIN` AND MLP IC > `MLP_IC_MIN` — the net genuinely contributes beyond every one-liner |

```bash
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.baseline_compare
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.baseline_compare --all
```
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .calibration import _spearman
from .decision_scorer import _to_float

# Verdict thresholds — module-level so tests assert exact verdicts and a
# tuning change is a single reviewable edit (mirrors calibration.py /
# gate_audit.py).
MIN_PAIRS = 30      # need a real OOS sample before any verdict (≈ calibration.MIN_PAIRS)
IC_MARGIN = 0.05    # rank-IC margin the MLP must clear / lose by to be decisive
MLP_IC_MIN = 0.10   # the MLP must itself have real rank skill to "add" anything
                    # (matches calibration.SPEARMAN_MIN — a model that only
                    # beats trivial because *everything* is noise is not skill)


def _aligned_pred_target(value: float, forward_return_5d: float,
                          is_sell: bool) -> tuple[float, float]:
    """Return ``(pred, target)`` in the codebase-universal action-aligned
    space. The realized target is ``-forward_return_5d`` on a SELL (a drop
    after a SELL was the *right* call — exactly `train_scorer` /
    `calibration.scorer_calibration`). A trivial baseline's raw feature value
    is flipped the same way so it is measured in the SAME goodness space as
    the (training-aligned) MLP prediction, which is NOT flipped (the MLP
    learned the flip during training — mirroring `calibration.scorer_calibration`
    which pairs the raw `scorer.predict` with the flipped target)."""
    tgt = -forward_return_5d if is_sell else forward_return_5d
    pv = -value if is_sell else value
    return pv, tgt


# Trivial one-line baselines. Each maps a decision_outcomes.jsonl row to a
# scalar "predicted goodness of this BUY" (pre SELL-flip — the flip is applied
# uniformly by `_skill`). They are deliberately the cheapest things a quant
# would try before reaching for a neural net:
#   constant_zero  — sanity floor (must read degenerate / no rank skill)
#   ml_score       — the raw signal that drove the decision (feature slot 0)
#   mom20          — 20-day momentum carry
#   mom5           — 5-day momentum carry
#   rsi_meanrev    — classic RSI mean-reversion: -(rsi - 50)
#   neg_bb         — Bollinger mean-reversion: -bb_position
BASELINES: dict[str, callable] = {
    "constant_zero": lambda r: 0.0,
    "ml_score": lambda r: _to_float(r.get("ml_score"), 0.0),
    "mom20": lambda r: _to_float(r.get("mom20"), 0.0),
    "mom5": lambda r: _to_float(r.get("mom5"), 0.0),
    "rsi_meanrev": lambda r: -(_to_float(r.get("rsi"), 50.0) - 50.0),
    "neg_bb": lambda r: -_to_float(r.get("bb_position"), 0.0),
}


def _skill(preds: list[float], targets: list[float]) -> dict:
    """Scale-invariant skill of ``preds`` vs action-aligned ``targets``.

    Returns ``{n, rank_ic, dir_acc, degenerate}``. ``degenerate`` is True when
    the prediction vector has zero variance on the slice (a constant baseline
    — `_spearman` already returns 0.0 there, but the explicit flag stops a
    reader concluding "matched the MLP"; same honesty pattern as
    `feature_importance`'s degenerate flag). ``rank_ic``/``dir_acc`` are None
    on a degenerate or too-small vector. Never raises."""
    out = {"n": len(preds), "rank_ic": None, "dir_acc": None,
           "degenerate": False}
    try:
        if len(preds) < 2:
            return out
        p = np.asarray(preds, dtype=np.float64)
        t = np.asarray(targets, dtype=np.float64)
        if float(np.std(p)) == 0.0:
            out["degenerate"] = True
            return out
        ic = _spearman(p, t)
        if ic == ic:  # not NaN
            out["rank_ic"] = round(float(ic), 4)
        dir_pairs = [(pp, tt) for pp, tt in zip(preds, targets)
                     if pp != 0.0 and tt != 0.0]
        if dir_pairs:
            hits = sum(1 for pp, tt in dir_pairs if (pp > 0) == (tt > 0))
            out["dir_acc"] = round(hits / len(dir_pairs), 4)
    except Exception:
        return {"n": len(preds), "rank_ic": None, "dir_acc": None,
                "degenerate": False}
    return out


def baseline_compare_report(mlp_preds, baseline_preds, targets) -> dict:
    """Compare one MLP prediction vector against a ``{name: vector}`` of
    trivial baselines, all against the SAME action-aligned ``targets``.

    All three inputs are parallel sequences already in action-aligned space
    (the caller applied the SELL flip). Non-finite/length faults degrade to
    ``INSUFFICIENT_DATA``; never raises.

    Returns a JSON-safe dict ``{status, verdict, n, mlp:{...},
    baselines:[{name, rank_ic, dir_acc, degenerate, n}], best_baseline,
    best_baseline_ic, ic_gap, hint}``.
    """
    base = {
        "status": "ok", "verdict": "INSUFFICIENT_DATA",
        "n": len(targets) if targets is not None else 0,
        "mlp": {"rank_ic": None, "dir_acc": None, "n": 0},
        "baselines": [], "best_baseline": None, "best_baseline_ic": None,
        "ic_gap": None, "hint": "",
    }
    try:
        tgt = list(targets or [])
        mp = list(mlp_preds or [])
        n = len(tgt)
        if n < MIN_PAIRS or len(mp) != n:
            base["hint"] = (f"need ≥{MIN_PAIRS} aligned OOS pairs, "
                            f"have n={n} (mlp={len(mp)})")
            return base

        mlp_skill = _skill(mp, tgt)
        base["mlp"] = {"rank_ic": mlp_skill["rank_ic"],
                       "dir_acc": mlp_skill["dir_acc"], "n": mlp_skill["n"]}

        rows = []
        for name, vec in baseline_preds.items():
            v = list(vec or [])
            if len(v) != n:
                rows.append({"name": name, "rank_ic": None, "dir_acc": None,
                             "degenerate": True, "n": len(v)})
                continue
            s = _skill(v, tgt)
            rows.append({"name": name, "rank_ic": s["rank_ic"],
                         "dir_acc": s["dir_acc"],
                         "degenerate": s["degenerate"], "n": s["n"]})
        base["baselines"] = rows

        # Best baseline = highest rank_ic among NON-degenerate baselines with a
        # finite IC. A degenerate (constant) baseline can never be "the
        # one-liner that beats the net" — that would be a measurement artifact.
        finite = [b for b in rows
                  if not b["degenerate"] and b["rank_ic"] is not None]
        mlp_ic = mlp_skill["rank_ic"]
        if not finite or mlp_ic is None:
            base["hint"] = ("no non-degenerate baseline IC or MLP IC could be "
                            "computed on the slice")
            base["verdict"] = "INSUFFICIENT_DATA"
            return base

        best = max(finite, key=lambda b: b["rank_ic"])
        base["best_baseline"] = best["name"]
        base["best_baseline_ic"] = best["rank_ic"]
        gap = round(mlp_ic - best["rank_ic"], 4)
        base["ic_gap"] = gap

        if best["rank_ic"] > mlp_ic + IC_MARGIN:
            base["verdict"] = "MLP_WORSE_THAN_TRIVIAL"
            base["hint"] = (
                f"one-liner '{best['name']}' rank_ic {best['rank_ic']:+.3f} "
                f"beats the 17-feature MLP {mlp_ic:+.3f} by "
                f"{best['rank_ic'] - mlp_ic:.3f} (> {IC_MARGIN}) — the net is "
                f"net-negative complexity out of sample")
        elif mlp_ic > best["rank_ic"] + IC_MARGIN and mlp_ic > MLP_IC_MIN:
            base["verdict"] = "MLP_ADDS_SKILL"
            base["hint"] = (
                f"MLP rank_ic {mlp_ic:+.3f} clears best baseline "
                f"'{best['name']}' {best['rank_ic']:+.3f} by {gap:+.3f} "
                f"(> {IC_MARGIN}) and its own {MLP_IC_MIN} skill floor — the "
                f"17-dim model is genuinely additive OOS")
        else:
            base["verdict"] = "MLP_NO_BETTER_THAN_TRIVIAL"
            base["hint"] = (
                f"MLP rank_ic {mlp_ic:+.3f} vs best one-liner "
                f"'{best['name']}' {best['rank_ic']:+.3f} (gap {gap:+.3f}, "
                f"within ±{IC_MARGIN} or below the {MLP_IC_MIN} MLP skill "
                f"floor) — the neural net's complexity buys no OOS edge a "
                f"single feature doesn't already carry")
        return base
    except Exception as e:  # never raises into the unattended loop's vicinity
        base["status"] = "error"
        base["verdict"] = "INSUFFICIENT_DATA"
        base["hint"] = f"compare failed: {type(e).__name__}: {e}"
        return base


def scorer_baseline_compare(scorer, records, oos_only: bool = True) -> dict:
    """Run ``baseline_compare_report`` for a DecisionScorer over outcome
    records (the ``data/decision_outcomes.jsonl`` row shape).

    The 11-kwarg ``predict`` signature mirrors
    ``validation.evaluate_scorer_oos`` / ``gate_audit.scorer_gate_audit`` /
    ``_oos_rank_metrics`` so the MLP column describes the SAME prediction path
    the live gate uses (and its `rank_ic` equals `calibration --oos`'s — a
    built-in contradiction check).

    ``oos_only`` (default True) restricts to the temporal-OOS slice via
    ``validation.split_outcomes_temporal`` — the trustworthy,
    generalization-relevant view. The chosen slice is reported in ``slice``.

    Never raises — any fault degrades to ``INSUFFICIENT_DATA``.
    """
    try:
        recs = list(records or [])
    except Exception:
        recs = []

    slice_name = "all"
    if oos_only:
        try:
            from paper_trader.validation import split_outcomes_temporal
            _, oos = split_outcomes_temporal(recs, oos_fraction=0.2)
            if oos:
                recs = oos
                slice_name = "oos"
        except Exception:
            slice_name = "all"

    mlp_preds: list[float] = []
    base_preds: dict[str, list[float]] = {k: [] for k in BASELINES}
    targets: list[float] = []
    for r in recs:
        y = r.get("forward_return_5d")
        if y is None:
            continue
        try:
            yf = float(y)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(yf):
            continue
        try:
            pred = scorer.predict(
                ml_score=r.get("ml_score", 0.0),
                rsi=r.get("rsi"),
                macd=r.get("macd"),
                mom5=r.get("mom5"),
                mom20=r.get("mom20"),
                regime_mult=r.get("regime_mult", 1.0),
                ticker=r.get("ticker", ""),
                vol_ratio=r.get("vol_ratio"),
                bb_pos=r.get("bb_position"),
                news_urgency=r.get("news_urgency"),
                news_article_count=r.get("news_article_count"),
            )
            predf = float(pred)
        except Exception:
            continue
        if not np.isfinite(predf):
            continue
        is_sell = str(r.get("action") or "BUY").upper() == "SELL"
        # MLP is training-aligned: its raw prediction is paired with the
        # flipped target (NOT flipped itself) — exactly calibration.py.
        _, tgt = _aligned_pred_target(predf, yf, is_sell)
        mlp_preds.append(predf)
        targets.append(tgt)
        for name, fn in BASELINES.items():
            try:
                raw = float(fn(r))
            except Exception:
                raw = 0.0
            if not np.isfinite(raw):
                raw = 0.0
            bp, _ = _aligned_pred_target(raw, yf, is_sell)
            base_preds[name].append(bp)

    rep = baseline_compare_report(mlp_preds, base_preds, targets)
    rep["slice"] = slice_name
    rep["n_records_considered"] = len(recs)
    return rep


def analyze(outcomes_path: Path | str, oos_only: bool = True) -> dict:
    """Load the live pickled scorer + outcomes file and return the comparison.
    Read-only; never raises."""
    from .decision_scorer import DecisionScorer

    out: dict = {"status": "error", "verdict": "INSUFFICIENT_DATA",
                 "n": 0, "baselines": [], "hint": ""}
    try:
        p = Path(outcomes_path)
        if not p.exists():
            out["hint"] = f"no outcomes file at {p}"
            return out
        records: list[dict] = []
        for ln in p.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                records.append(json.loads(ln))
            except Exception:
                continue
        scorer = DecisionScorer()
        if not getattr(scorer, "is_trained", False):
            out["hint"] = "scorer not trained — nothing to compare"
            return out
        rep = scorer_baseline_compare(scorer, records, oos_only=oos_only)
        rep.setdefault("n_train", getattr(scorer, "n_train", None))
        return rep
    except Exception as e:
        out["hint"] = f"compare failed: {type(e).__name__}: {e}"
        return out


def _cli(argv: list[str] | None = None) -> int:
    """`python3 -m paper_trader.ml.baseline_compare [--all]` — compare the
    live pickled MLP against trivial one-line baselines on the temporal-OOS
    slice (default) or the full in-sample set (``--all``). Read-only.

    Exit code: 0 on ``MLP_ADDS_SKILL`` / ``INSUFFICIENT_DATA``, 2 on
    ``MLP_NO_BETTER_THAN_TRIVIAL`` / ``MLP_WORSE_THAN_TRIVIAL`` — so an
    operator/cron can branch on "the net earns its complexity" exactly like
    `gate_audit` / `feature_importance` / `persona_skill`.
    """
    import sys

    args = sys.argv[1:] if argv is None else argv
    oos_only = "--all" not in args

    root = Path(__file__).resolve().parent.parent.parent
    rep = analyze(root / "data" / "decision_outcomes.jsonl", oos_only=oos_only)
    print(f"VERDICT: {rep['verdict']}  ({rep.get('hint', '')})")
    mlp = rep.get("mlp", {})
    print(f"  slice={rep.get('slice')}  n={rep.get('n')}  "
          f"n_train={rep.get('n_train')}  "
          f"MLP rank_ic={mlp.get('rank_ic')} dir_acc={mlp.get('dir_acc')}")
    print(f"  best_baseline={rep.get('best_baseline')} "
          f"ic={rep.get('best_baseline_ic')}  "
          f"ic_gap(MLP−best)={rep.get('ic_gap')}")
    for b in rep.get("baselines", []):
        ic = b["rank_ic"]
        ic_s = f"{ic:+.4f}" if ic is not None else "   n/a"
        da = b["dir_acc"]
        da_s = f"{da:.4f}" if da is not None else "  n/a"
        deg = " [degenerate]" if b["degenerate"] else ""
        print(f"  {b['name']:<14} rank_ic={ic_s}  dir_acc={da_s}  "
              f"n={b['n']}{deg}")
    verdict = rep.get("verdict")
    if verdict in ("MLP_NO_BETTER_THAN_TRIVIAL", "MLP_WORSE_THAN_TRIVIAL"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())

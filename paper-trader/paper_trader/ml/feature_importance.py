"""Permutation feature-importance diagnostic for the DecisionScorer — read-only.

The natural quant question after `gate_audit` = GATE_INEFFECTIVE and
`skill_trend` = NEGATIVE_OOS_SKILL: the conviction gate is economically inert
and has no demonstrated out-of-sample skill — but is that because the model
learned **nothing**, or because it overfits to the 7-way sector one-hot
(memorizing the DFEN reverse-split / FAS-COVID extreme-label tail that
`label_audit` flags) instead of the actual quant signals?

`calibration` says "monotone in-sample", `gate_audit` says "the multipliers
buy ≈0 realized edge", `skill_trend` says "no OOS edge vs a mean predictor" —
**none of them say WHICH feature carries (or fails to carry) the prediction.**
That is the gap this module fills.

Permutation importance answers it directly: for each logical feature, shuffle
that feature's values across the held-out records (every other feature left
intact), re-predict through the **same** ``scorer.predict`` path the live
``_ml_decide`` gate uses, and measure how much skill the model loses. A
feature the model genuinely relies on loses skill when its values are
scrambled; a feature it ignores does not move the metrics at all (exactly
``0.0`` importance — a deterministic, testable signature).

Three skill metrics are reported per feature, not just RMSE, because the gate
acts on the prediction's **sign / bucket**, not its magnitude (CLAUDE.md §6):
``rmse_increase`` (magnitude), ``rank_ic_drop`` and ``dir_acc_drop``
(direction). A feature whose permutation tanks ``dir_acc`` but not ``rmse`` is
gate-relevant; the reverse is magnitude-only.

The 7-way sector one-hot is permuted **jointly** by shuffling the ``ticker``
field and letting ``build_features``' ``SECTOR_MAP`` reassign the block —
permuting a single one-hot slot would emit all-zero / double-one vectors that
fabricate importance. So the logical feature list is the 10 numeric record
fields plus one ``sector`` group keyed on ``ticker``.

Same operational discipline as ``paper_trader/ml/calibration.py``: read-only,
no train, no pickle / ``build_features`` / ``N_FEATURES`` / trade-path touch —
safe to run against the live unattended loop. Never raises on bad input.
Reuses ``calibration._spearman`` for the rank-IC delta (single source of
truth — the ``_oos_rank_metrics`` precedent; the tie-awareness is load-bearing
because the scorer clamps off-distribution predictions to ±50) and
``validation.split_outcomes_temporal`` for the trustworthy OOS slice (the
``gate_audit`` / ``skill_trend`` default). The baseline metrics are
constructed identically to the ledger's ``oos_ic`` / ``oos_dir_acc`` so they
read on the same scale.

NOTE: this is a CLI / ``ml/`` reader, not wired into
``run_continuous_backtests.py::main()`` — it has zero deploy-stale impact and
needs no loop restart to take effect (unlike a ``main()`` wiring change).

```bash
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.feature_importance
```
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

# Thresholds are module-level so tests assert exact verdicts and a tuning
# change is a single reviewable edit (the codebase's constants-at-module-scope
# convention — PRED_CLAMP_PCT, calibration.SPEARMAN_MIN, …).
MIN_RECORDS = 30          # ≥ calibration.MIN_PAIRS — need a stable baseline
N_REPEATS = 5             # average permutation importance over this many shuffles
# A feature is "material" if scrambling it raises RMSE by ≥ this many points
# of 5-day return, OR drops the rank-IC by ≥ this much. Deliberately modest:
# the point is to separate "moves the model at all" from "model is blind to
# it" (which is exactly 0.0 for a feature predict() ignores).
MIN_RMSE_INCREASE = 0.10
MIN_IC_DROP = 0.02

# The 10 numeric record fields + the sector group. Each entry is
# (logical_name, predict_kwarg, record_key). The sector block is permuted via
# the ``ticker`` field — build_features' SECTOR_MAP is the single source of
# truth for the one-hot, so this keeps the joint-permutation correct without
# this module ever importing build_features / N_FEATURES.
FEATURES: list[tuple[str, str, str]] = [
    ("ml_score", "ml_score", "ml_score"),
    ("rsi", "rsi", "rsi"),
    ("macd", "macd", "macd"),
    ("mom5", "mom5", "mom5"),
    ("mom20", "mom20", "mom20"),
    ("regime_mult", "regime_mult", "regime_mult"),
    ("vol_ratio", "vol_ratio", "vol_ratio"),
    ("bb_position", "bb_pos", "bb_position"),
    ("news_urgency", "news_urgency", "news_urgency"),
    ("news_article_count", "news_article_count", "news_article_count"),
    # Enhanced MACD/EMA200 features — permute-importance must cover the
    # same 14 inputs the live ``_ml_decide`` and ``_compute_decision_outcomes``
    # plumb (pass #36 OOS parity fix). Omitting them silently understated
    # the importance of features the deployed model actively learned on
    # (mean|w|≈0.42 / 0.30 / 0.27 — non-zero by audit).
    ("ema200_above", "ema200_above", "ema200_above"),
    ("hist_cross_up", "hist_cross_up", "hist_cross_up"),
    ("macd_below_zero_cross", "macd_below_zero_cross", "macd_below_zero_cross"),
    ("sector", "ticker", "ticker"),
]
QUANT_FEATURES = frozenset(n for n, _, _ in FEATURES if n != "sector")


def _kwargs(r: dict) -> dict:
    """Build the 14-kwarg ``predict`` call for one record, constructed
    identically to ``_oos_rank_metrics`` (the ledger's ``oos_ic`` path) so
    feature-importance baselines read on the same scale as the ledger.

    Forwards the 3 enhanced MACD features (ema200_above / hist_cross_up /
    macd_below_zero_cross) that the live ``_ml_decide`` and
    ``_compute_decision_outcomes`` plumb — pass #36 OOS parity fix."""
    from paper_trader.ml.decision_scorer import _to_float
    return dict(
        ml_score=_to_float(r.get("ml_score"), 0.0),
        rsi=r.get("rsi"),
        macd=r.get("macd"),
        mom5=r.get("mom5"),
        mom20=r.get("mom20"),
        regime_mult=_to_float(r.get("regime_mult"), 1.0),
        ticker=str(r.get("ticker") or ""),
        vol_ratio=r.get("vol_ratio"),
        bb_pos=r.get("bb_position"),
        news_urgency=r.get("news_urgency"),
        news_article_count=r.get("news_article_count"),
        ema200_above=r.get("ema200_above"),
        hist_cross_up=r.get("hist_cross_up"),
        macd_below_zero_cross=r.get("macd_below_zero_cross"),
    )


def _override(kw: dict, predict_kwarg: str, raw_value) -> dict:
    """Return a copy of ``kw`` with one feature replaced by a permuted raw
    value, applying the SAME normalization the baseline ``_kwargs`` applied to
    that slot (so the only thing that changes is the permutation, not the
    encoding)."""
    from paper_trader.ml.decision_scorer import _to_float
    out = dict(kw)
    if predict_kwarg == "ml_score":
        out["ml_score"] = _to_float(raw_value, 0.0)
    elif predict_kwarg == "regime_mult":
        out["regime_mult"] = _to_float(raw_value, 1.0)
    elif predict_kwarg == "ticker":
        out["ticker"] = str(raw_value or "")
    else:
        out[predict_kwarg] = raw_value
    return out


def _realized(r: dict) -> float:
    """Action-aligned realized 5d return — the universal SELL sign-flip
    (`train_scorer` / `evaluate_scorer_oos` / `calibration` / `gate_audit`):
    a drop after a SELL was the *right* call, so "good" has one meaning."""
    from paper_trader.ml.decision_scorer import _to_float
    a = _to_float(r.get("forward_return_5d"), 0.0)
    if str(r.get("action") or "BUY").upper() == "SELL":
        a = -a
    return a


def _metrics(preds: list[float], actuals: list[float]) -> dict:
    """{rmse, rank_ic, dir_acc, n} over aligned (pred, realized) pairs.

    Non-finite pairs are dropped (the calibration/_to_float hardening class).
    ``rank_ic`` reuses ``calibration._spearman`` so it is the SAME tie-aware
    statistic the in-sample calibration diagnostic and the ledger's
    ``oos_ic`` use (single source of truth). ``dir_acc`` excludes a zero on
    either side (no directional truth) exactly like ``_oos_rank_metrics``."""
    from paper_trader.ml.calibration import _spearman
    p: list[float] = []
    a: list[float] = []
    for pp, aa in zip(preds, actuals):
        try:
            pf = float(pp)
            af = float(aa)
        except (TypeError, ValueError):
            continue
        if pf == pf and af == af and np.isfinite(pf) and np.isfinite(af):
            p.append(pf)
            a.append(af)
    n = len(p)
    out = {"rmse": None, "rank_ic": None, "dir_acc": None, "n": n}
    if n < 2:
        return out
    pa = np.asarray(p, dtype=np.float64)
    aa = np.asarray(a, dtype=np.float64)
    out["rmse"] = float(np.sqrt(np.mean((pa - aa) ** 2)))
    ic = _spearman(pa, aa)
    if ic == ic:
        out["rank_ic"] = float(ic)
    dir_pairs = [(x, y) for x, y in zip(p, a) if x != 0.0 and y != 0.0]
    if dir_pairs:
        hits = sum(1 for x, y in dir_pairs if (x > 0) == (y > 0))
        out["dir_acc"] = hits / len(dir_pairs)
    return out


def feature_importance(scorer, records, oos_only: bool = True,
                       n_repeats: int = N_REPEATS) -> dict:
    """Permutation feature-importance of ``scorer`` over outcome records
    (the ``data/decision_outcomes.jsonl`` row shape).

    For each logical feature, shuffle that feature's column across the
    records ``n_repeats`` times (deterministic seeds 0..n_repeats-1, reused
    across features so the comparison is paired/low-variance), re-predict, and
    average the skill loss vs the unpermuted baseline.

    ``oos_only`` (default True) restricts to the temporal-OOS slice via
    ``validation.split_outcomes_temporal`` — the trustworthy generalization
    view (in-sample is optimistic; the scorer trained on most of it).

    Verdicts (exact-value test-locked in tests/test_feature_importance.py):
      * ``UNTRAINED``         — scorer reports ``is_trained`` False
      * ``INSUFFICIENT_DATA`` — < MIN_RECORDS finite baseline pairs
      * ``FLAT``              — no feature is material: permuting nothing moves
                                the model ⇒ it is ≈constant noise on this
                                slice (consistent with NEGATIVE_OOS_SKILL)
      * ``SECTOR_DOMINATED``  — ``sector`` is the single most important
                                feature AND no quant feature is material: the
                                only "skill" is sector memorization (dangerous
                                given label_audit's DFEN/FAS extreme-label
                                concentration)
      * ``SECTOR_LEANING``    — ``sector`` is the single most important feature
                                but ≥1 quant feature is also material
      * ``SIGNAL_GROUNDED``   — the most important feature is a quant signal
                                and ≥1 quant feature is material

    Honesty guard: a feature whose column has < 2 distinct non-null values on
    the evaluated slice has NOTHING to permute, so its 0.0 importance is a
    data-sparsity artifact, NOT evidence the model weights it ≈0. Such a
    feature is flagged ``degenerate: True`` and can never be ``material``.
    (Live: ``news_urgency`` / ``news_article_count`` are null for 100% of the
    OOS slice — see the AGENTS.md finding — so the tool would otherwise
    mislead a reader into "the model ignores news"; it actually never sees
    varying news features there.)

    Never raises — any fault degrades to a status/verdict dict.
    """
    try:
        recs = [r for r in (records or []) if isinstance(r, dict)]
    except Exception:
        recs = []

    base: dict = {
        "status": "error",
        "verdict": "INSUFFICIENT_DATA",
        "slice": "all",
        "n": 0,
        "n_records_considered": 0,
        "n_repeats": n_repeats,
        "baseline_rmse": None,
        "baseline_rank_ic": None,
        "baseline_dir_acc": None,
        "top_feature": None,
        "features": [],
        "hint": "",
    }

    if not getattr(scorer, "is_trained", False):
        base["verdict"] = "UNTRAINED"
        base["status"] = "untrained"
        base["hint"] = "scorer not trained — nothing to attribute"
        return base

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
    base["slice"] = slice_name
    base["n_records_considered"] = len(recs)

    # Baseline: predict each record once, pair with its action-aligned target.
    # Keep the record alongside its kwargs so the permuted feature columns are
    # built from EXACTLY the records that produced a valid baseline prediction
    # (a single pass — never predict the same record twice).
    base_kw: list[dict] = []
    actuals: list[float] = []
    base_preds: list[float] = []
    aligned_recs: list[dict] = []
    for r in recs:
        try:
            kw = _kwargs(r)
            p = float(scorer.predict(**kw))
        except Exception:
            continue
        base_kw.append(kw)
        actuals.append(_realized(r))
        base_preds.append(p)
        aligned_recs.append(r)

    bm = _metrics(base_preds, actuals)
    base["n"] = bm["n"]
    if bm["n"] < MIN_RECORDS or bm["rmse"] is None:
        base["hint"] = (f"need ≥{MIN_RECORDS} finite baseline pairs, "
                        f"have {bm['n']}")
        return base
    base["baseline_rmse"] = round(bm["rmse"], 4)
    base["baseline_rank_ic"] = (round(bm["rank_ic"], 4)
                                if bm["rank_ic"] is not None else None)
    base["baseline_dir_acc"] = (round(bm["dir_acc"], 4)
                                if bm["dir_acc"] is not None else None)

    n = len(base_kw)
    # Precompute the shuffle permutations once and reuse for every feature so
    # the per-feature comparison is paired (lower variance) and deterministic.
    perms: list[list[int]] = []
    for s in range(max(1, n_repeats)):
        idx = list(range(n))
        random.Random(s).shuffle(idx)
        perms.append(idx)

    # Raw feature columns aligned to base_kw (only the feature column is ever
    # permuted below — targets and every other feature stay put).
    aligned_raw = {rk: [rr.get(rk) for rr in aligned_recs]
                   for _, _, rk in FEATURES}

    def _n_distinct_nonnull(col) -> int:
        seen = set()
        for v in col:
            if v is None:
                continue
            try:
                seen.add(v)
            except TypeError:  # unhashable — count as present-but-opaque
                seen.add(repr(v))
        return len(seen)

    feats_out: list[dict] = []
    for name, kwarg, rk in FEATURES:
        col = aligned_raw[rk]
        rmse_incs: list[float] = []
        ic_drops: list[float] = []
        dir_drops: list[float] = []
        for perm in perms:
            shuffled = [col[perm[i]] for i in range(n)]
            preds: list[float] = []
            for i in range(n):
                try:
                    kw = _override(base_kw[i], kwarg, shuffled[i])
                    preds.append(float(scorer.predict(**kw)))
                except Exception:
                    preds.append(float("nan"))
            pm = _metrics(preds, actuals)
            if pm["rmse"] is not None:
                rmse_incs.append(pm["rmse"] - bm["rmse"])
            if pm["rank_ic"] is not None and bm["rank_ic"] is not None:
                ic_drops.append(bm["rank_ic"] - pm["rank_ic"])
            if pm["dir_acc"] is not None and bm["dir_acc"] is not None:
                dir_drops.append(bm["dir_acc"] - pm["dir_acc"])

        def _avg(xs):
            return float(np.mean(xs)) if xs else 0.0

        rmse_inc = _avg(rmse_incs)
        ic_drop = _avg(ic_drops)
        dir_drop = _avg(dir_drops)
        # A column with <2 distinct non-null values has NOTHING to permute, so
        # its 0.0 importance is a data-sparsity artifact, NOT evidence the
        # model ignores the feature. Surface it explicitly (`degenerate`) so a
        # reader never mistakes "this slice has no variance here" for "the
        # scorer weights this near zero". A degenerate column can never be
        # material regardless of the metric thresholds.
        n_distinct = _n_distinct_nonnull(col)
        degenerate = n_distinct < 2
        material = (not degenerate
                    and (rmse_inc >= MIN_RMSE_INCREASE
                         or ic_drop >= MIN_IC_DROP))
        feats_out.append({
            "feature": name,
            "is_quant": name in QUANT_FEATURES,
            "rmse_increase": round(rmse_inc, 4),
            "rank_ic_drop": round(ic_drop, 4),
            "dir_acc_drop": round(dir_drop, 4),
            "n_distinct": n_distinct,
            "degenerate": bool(degenerate),
            "material": bool(material),
        })

    feats_out.sort(key=lambda f: f["rmse_increase"], reverse=True)
    base["features"] = feats_out
    base["status"] = "ok"
    degenerate_feats = [f["feature"] for f in feats_out if f["degenerate"]]
    base["n_degenerate_features"] = len(degenerate_feats)
    base["degenerate_features"] = degenerate_feats

    materials = [f for f in feats_out if f["material"]]
    quant_materials = [f for f in materials if f["is_quant"]]
    top = feats_out[0]
    base["top_feature"] = top["feature"]

    if not materials:
        base["verdict"] = "FLAT"
        deg_note = (f" ({len(degenerate_feats)} feature(s) are "
                    f"degenerate/constant on this slice and could not be "
                    f"permuted: {', '.join(degenerate_feats)})"
                    if degenerate_feats else "")
        base["hint"] = ("no non-degenerate feature changes the model when "
                        "scrambled — the scorer is ≈constant noise on this "
                        "slice (consistent with a NEGATIVE_OOS_SKILL / inert "
                        "gate)" + deg_note)
    elif top["feature"] == "sector":
        if not quant_materials:
            base["verdict"] = "SECTOR_DOMINATED"
            base["hint"] = ("the 7-way sector one-hot is the ONLY thing the "
                            "model uses — no quant signal is material; this "
                            "is sector memorization, not signal skill "
                            "(see label_audit's per-ticker extreme-label "
                            "concentration)")
        else:
            base["verdict"] = "SECTOR_LEANING"
            base["hint"] = ("sector is the single most important feature but "
                            "≥1 quant signal is also material — the model "
                            "leans on sector identity")
    else:
        base["verdict"] = "SIGNAL_GROUNDED"
        base["hint"] = (f"most important feature is the quant signal "
                        f"'{top['feature']}' (rmse_increase "
                        f"{top['rmse_increase']:+.3f}); ≥1 quant feature is "
                        f"material")
    return base


def analyze(outcomes_path: Path | str, oos_only: bool = True,
            n_repeats: int = N_REPEATS) -> dict:
    """Load the live pickled scorer + outcomes file and return the report.
    Read-only; never raises."""
    from .decision_scorer import DecisionScorer

    out: dict = {"status": "error", "verdict": "INSUFFICIENT_DATA",
                 "n": 0, "features": [], "hint": ""}
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
                obj = json.loads(ln)
                if isinstance(obj, dict):
                    records.append(obj)
            except Exception:
                continue
        scorer = DecisionScorer()
        rep = feature_importance(scorer, records, oos_only=oos_only,
                                 n_repeats=n_repeats)
        rep.setdefault("n_train", getattr(scorer, "n_train", None))
        return rep
    except Exception as e:
        out["hint"] = f"feature-importance failed: {type(e).__name__}: {e}"
        return out


def _cli() -> int:
    """`python3 -m paper_trader.ml.feature_importance` — permutation
    feature-importance of the live pickled scorer against the accumulated
    outcomes tail. Read-only. Exit 2 on SECTOR_DOMINATED / FLAT (the
    "model is not reading quant signal" verdicts — operator/cron branchable,
    like label_audit / persona_skill); 0 otherwise."""
    root = Path(__file__).resolve().parent.parent.parent
    rep = analyze(root / "data" / "decision_outcomes.jsonl", oos_only=True)
    print(f"VERDICT: {rep['verdict']}  ({rep.get('hint', '')})")
    print(f"  slice={rep.get('slice')}  n={rep.get('n')}  "
          f"n_train={rep.get('n_train')}  "
          f"top_feature={rep.get('top_feature')}")
    print(f"  baseline  rmse={rep.get('baseline_rmse')}  "
          f"rank_ic={rep.get('baseline_rank_ic')}  "
          f"dir_acc={rep.get('baseline_dir_acc')}")
    for f in rep.get("features", []):
        flag = "★" if f["material"] else ("∅" if f.get("degenerate") else " ")
        kind = "quant" if f["is_quant"] else "secto"
        print(f"  {flag} {f['feature']:<20} [{kind}] "
              f"rmse+={f['rmse_increase']:+7.3f}  "
              f"ic_drop={f['rank_ic_drop']:+6.3f}  "
              f"diracc_drop={f['dir_acc_drop']:+6.3f}  "
              f"distinct={f.get('n_distinct')}")
    if rep.get("degenerate_features"):
        print(f"  ∅ degenerate (no variance to permute on this slice): "
              f"{', '.join(rep['degenerate_features'])}")
    return 2 if rep.get("verdict") in ("SECTOR_DOMINATED", "FLAT") else 0


if __name__ == "__main__":
    raise SystemExit(_cli())

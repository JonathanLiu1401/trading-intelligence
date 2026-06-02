"""Per-action *scorer* skill diagnostic — does the DecisionScorer's OOS
edge come from BUY predictions, SELL predictions, or both?

This is the **action-conditional** sibling of
`paper_trader/ml/persona_skill.py` (per-persona) and
`paper_trader/ml/regime_audit.py` (per-regime). The aggregate
`_oos_rank_metrics` (one number across BUY+SELL) hides a load-bearing
asymmetry: the conviction gate (CLAUDE.md §6, invariant #5) acts on BUY
predictions only, so SELL skill **does not gate trades** — but the model
is trained on a *mix* of BUYs and SELLs (with the universal codebase
SELL convention: target sign flipped). If the BUY-only OOS rank-IC is
materially worse than the aggregate (because the BUY half is weak and
the SELL half carries the score), the deployed gate is sizing on a
slice of the model that has less skill than the headline number
suggests.

`action_skill` answers exactly that, per action: tie-aware Spearman
between the **action-aligned** scorer prediction and the
**action-aligned** realized 5-trading-day forward return — the SAME
SELL-flip convention used by `train_scorer`,
`validation.evaluate_scorer_oos`, `ml.calibration`,
`run_continuous_backtests._oos_rank_metrics`, and
`ml.persona_skill._aligned`. `_spearman` is **imported from
`ml.calibration`** (single source of truth — the AGENTS.md invariant-10
spirit — so this and every sibling rank metric can never drift; tie-
awareness is load-bearing because clamped ±50 predictions tie at the
empirical label support).

Operational discipline is identical to its siblings: **read-only** — no
train, no `decision_scorer.pkl` / `build_features` / `N_FEATURES` /
trade path touch, never raises on bad input — so it is safe to run
against the live unattended continuous loop and cannot break pickle
compatibility. It does **not** change the gate; that is a separate,
explicit decision. This tool exists only to *inform* it.

Verdict per action (crisp, threshold-driven so it is exactly testable):

| Verdict | Meaning |
|---------|---------|
| `INSUFFICIENT` | < ``MIN_OUTCOMES_PER_ACTION`` aligned outcomes — no stable IC |
| `INVERTED` | ``rank_ic ≤ -IC_GOOD`` — predictions are anti-correlated with realized; the model is actively wrong on this slice |
| `EDGE` | ``rank_ic ≥ IC_GOOD`` — predictions genuinely rank-predict realized goodness |
| `WEAK_EDGE` | ``IC_MIN ≤ rank_ic < IC_GOOD`` — usable as a tie-breaker, not a primary signal |
| `NO_EDGE` | ``-IC_GOOD < rank_ic < IC_MIN`` — no demonstrated rank skill |

Overall verdict: ``INSUFFICIENT_DATA`` (< ``MIN_RECORDS`` aligned rows),
``ASYMMETRIC_BUY_EDGE`` (BUY=EDGE, SELL!=EDGE — the gate-relevant
healthy case: the gate fires on BUY and BUY is where the skill is),
``ASYMMETRIC_SELL_EDGE`` (BUY!=EDGE, SELL=EDGE — the gate-relevant
*concerning* case: the model's edge is on a slice the gate doesn't
use), ``BOTH_SKILLED`` (BUY=EDGE and SELL=EDGE — healthiest),
``NEITHER_SKILLED`` (neither side reaches EDGE), or
``HAS_INVERTED_ACTION`` (≥1 INVERTED — actionable red flag).

```bash
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.action_skill
cd /home/zeph/paper-trader && python3 -m pytest tests/test_action_skill.py -v
```
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Single source of truth for the tie-aware rank correlation — reused so
# this per-action IC and the in-sample calibration / persona_skill /
# regime_audit metrics can never drift (the AGENTS.md invariant-10
# spirit). Tie-awareness is load-bearing: the scorer clamps to
# ±PRED_CLAMP_PCT, so off-distribution predictions tie at exactly ±50
# and a naïve argsort would fabricate rank skill there.
from paper_trader.ml.calibration import _spearman
from paper_trader.ml.decision_scorer import _to_float

# Thresholds are module-level so tests assert exact verdicts and a
# tuning change is a single reviewable edit (mirrors calibration.py /
# persona_skill.py and the codebase's constants-at-module-scope rule).
MIN_RECORDS = 30              # minimum aligned outcomes overall for any verdict
MIN_OUTCOMES_PER_ACTION = 20  # below this an action's Spearman is not stable
IC_MIN = 0.05                 # rank-skill bar — mirrors skill_trend.IC_MIN
IC_GOOD = 0.10                # stronger rank-skill bar — mirrors persona_skill.IC_GOOD

# The only actions the gate / scorer ever see — HOLD is excluded
# upstream by `_compute_decision_outcomes`'s `action IN ('BUY','SELL')`
# filter, so it never appears in `decision_outcomes.jsonl` rows.
_ACTIONS = ("BUY", "SELL")


def _aligned_pred(scorer, record: dict) -> tuple[float, float] | None:
    """Return (action-aligned prediction, action-aligned realized return)
    for one outcome record, or ``None`` when the record is unusable.

    Mirrors ``persona_skill._aligned`` exactly:
      * NaN sentinel for both legs so a missing/None target is *dropped*
        (the `_oos_rank_metrics` Phase-1 fix discipline).
      * Universal SELL convention — realized goodness of a SELL is
        ``-forward_return_5d``. The PREDICTION is NOT flipped: the scorer
        was trained on flipped targets, so its output for a SELL feature
        vector already encodes action-aligned goodness directly. (This
        matches every other consumer of the predict path.)

    Never raises — a scorer that raises on this record yields ``None``
    so the caller can simply skip; the wider report stays healthy.
    """
    fr = record.get("forward_return_5d")
    if fr is None:
        return None
    t = _to_float(fr, float("nan"))
    if t != t:                          # NaN ⇒ unparseable / non-finite ⇒ drop
        return None
    try:
        p = scorer.predict(
            ml_score=_to_float(record.get("ml_score"), 0.0),
            rsi=record.get("rsi"), macd=record.get("macd"),
            mom5=record.get("mom5"), mom20=record.get("mom20"),
            regime_mult=_to_float(record.get("regime_mult"), 1.0),
            ticker=str(record.get("ticker") or ""),
            vol_ratio=record.get("vol_ratio"),
            bb_pos=record.get("bb_position"),
            news_urgency=record.get("news_urgency"),
            news_article_count=record.get("news_article_count"),
            ema200_above=record.get("ema200_above"),
            hist_cross_up=record.get("hist_cross_up"),
            macd_below_zero_cross=record.get("macd_below_zero_cross"),
        )
    except Exception:
        return None
    try:
        p = float(p)
    except (TypeError, ValueError):
        return None
    if p != p:                          # NaN predict result (shouldn't happen, defensive)
        return None
    if str(record.get("action") or "BUY").upper() == "SELL":
        t = -t
    return p, t


def _verdict_for(ic: float, n: int) -> str:
    """Per-action verdict for a single (ic, n) pair. Pure, deterministic,
    threshold-driven so tests can assert exact strings."""
    if n < MIN_OUTCOMES_PER_ACTION:
        return "INSUFFICIENT"
    if ic <= -IC_GOOD:
        return "INVERTED"
    if ic >= IC_GOOD:
        return "EDGE"
    if ic >= IC_MIN:
        return "WEAK_EDGE"
    return "NO_EDGE"


def action_skill(scorer, records) -> dict:
    """Per-action OOS rank skill of a deployed scorer over outcome records.

    ``records`` is any iterable of dicts with at least ``action``,
    ``ml_score``, ``forward_return_5d``, and the quant feature columns
    (the ``decision_outcomes.jsonl`` row shape). Rows missing/with a
    non-finite ``forward_return_5d``, an untrained scorer, or a predict
    fault are dropped.

    Returns a JSON-safe dict:
    ``{status, verdict, n_records, by_action:{BUY:{n,rank_ic,dir_acc,
       mean_aligned_return,verdict}, SELL:{...}}, hint}``.
    """
    out_skel = {
        "status": "ok",
        "verdict": "INSUFFICIENT_DATA",
        "n_records": 0,
        "by_action": {a: {"n": 0, "rank_ic": None, "dir_acc": None,
                          "mean_aligned_return": None,
                          "verdict": "INSUFFICIENT"} for a in _ACTIONS},
        "hint": "",
    }

    if not getattr(scorer, "is_trained", False):
        out_skel["status"] = "untrained"
        out_skel["hint"] = ("scorer is not trained — no predictions to "
                            "evaluate. Train the scorer first "
                            "(`run_continuous_backtests` cycle ≥ a "
                            "deduped-record threshold).")
        return out_skel

    buckets: dict[str, dict] = {a: {"sig": [], "tgt": []} for a in _ACTIONS}
    n_aligned = 0
    for r in records:
        action = str(r.get("action") or "BUY").upper()
        if action not in _ACTIONS:      # HOLD or garbage — skip
            continue
        pair = _aligned_pred(scorer, r)
        if pair is None:
            continue
        p, t = pair
        n_aligned += 1
        buckets[action]["sig"].append(p)
        buckets[action]["tgt"].append(t)

    if n_aligned < MIN_RECORDS:
        out_skel["n_records"] = n_aligned
        out_skel["status"] = "insufficient_data"
        out_skel["hint"] = (f"need ≥{MIN_RECORDS} aligned outcomes with a "
                            f"finite forward_return_5d, have {n_aligned}")
        return out_skel

    by_action: dict[str, dict] = {}
    for action, b in buckets.items():
        n = len(b["sig"])
        if n == 0:
            by_action[action] = {"n": 0, "rank_ic": None, "dir_acc": None,
                                 "mean_aligned_return": None,
                                 "verdict": "INSUFFICIENT"}
            continue
        sig = np.asarray(b["sig"], dtype=np.float64)
        tgt = np.asarray(b["tgt"], dtype=np.float64)
        if n >= 2:
            ic_raw = float(_spearman(sig, tgt))
            ic = round(ic_raw, 4) if ic_raw == ic_raw else None
        else:
            ic = None
        # dir_acc = sign(pred) == sign(realized) excluding zero pairs that
        # carry no directional truth (the `_oos_rank_metrics` convention,
        # single source of truth).
        dir_pairs = [(p, t) for p, t in zip(sig.tolist(), tgt.tolist())
                     if p != 0.0 and t != 0.0]
        if dir_pairs:
            hits = sum(1 for p, t in dir_pairs if (p > 0) == (t > 0))
            dir_acc: float | None = round(hits / len(dir_pairs), 4)
        else:
            dir_acc = None
        mean_ret = round(float(tgt.mean()), 4) if n else None
        verdict = _verdict_for(ic if ic is not None else 0.0, n)
        by_action[action] = {
            "n": n, "rank_ic": ic, "dir_acc": dir_acc,
            "mean_aligned_return": mean_ret, "verdict": verdict,
        }

    buy_v = by_action["BUY"]["verdict"]
    sell_v = by_action["SELL"]["verdict"]
    has_inverted = any(by_action[a]["verdict"] == "INVERTED" for a in _ACTIONS)

    if has_inverted:
        # An anti-predictive slice is a red flag regardless of the
        # other action's skill — surface it FIRST so an operator can't
        # miss it under a healthy aggregate.
        verdict = "HAS_INVERTED_ACTION"
        inv = [a for a in _ACTIONS if by_action[a]["verdict"] == "INVERTED"]
        hint = (f"{','.join(inv)} predictions are anti-correlated with "
                f"realized goodness (rank_ic ≤ -{IC_GOOD}) — the model is "
                f"actively wrong on this slice. Read the per-action "
                f"rank_ic before sizing on the aggregate.")
    elif buy_v == "EDGE" and sell_v == "EDGE":
        verdict = "BOTH_SKILLED"
        hint = ("scorer carries rank-skill on BOTH actions — the gate is "
                "acting on a slice that has demonstrated edge.")
    elif buy_v == "EDGE" and sell_v != "EDGE":
        verdict = "ASYMMETRIC_BUY_EDGE"
        hint = ("scorer's rank-skill is concentrated in BUY predictions, "
                "which is the gate-relevant slice (the conviction gate "
                "acts BUY-only — invariant #5). The aggregate ic is a "
                "fair summary of what the gate uses.")
    elif buy_v != "EDGE" and sell_v == "EDGE":
        verdict = "ASYMMETRIC_SELL_EDGE"
        hint = ("scorer's rank-skill is concentrated in SELL predictions, "
                "but the conviction gate acts BUY-only (invariant #5). "
                "The aggregate ic OVERSTATES what the gate actually uses; "
                "do not size BUYs on the aggregate.")
    else:
        verdict = "NEITHER_SKILLED"
        hint = (f"neither action reaches rank_ic ≥ {IC_GOOD} on a stable "
                f"sample — gate sizing is variance, not edge "
                f"(the documented MLP_NO_BETTER_THAN_TRIVIAL state at the "
                f"action level).")

    return {
        "status": "ok",
        "verdict": verdict,
        "n_records": n_aligned,
        "by_action": by_action,
        "hint": hint,
    }


def _load_outcomes(path: Path) -> list[dict]:
    """Robust JSONL load of ``decision_outcomes.jsonl``. Skips unparseable
    lines; never raises (a missing/corrupt file yields ``[]`` so the CLI
    degrades to ``INSUFFICIENT_DATA`` rather than crashing — the same
    best-effort discipline as ``persona_skill._load_outcomes``)."""
    rows: list[dict] = []
    try:
        if not path.exists():
            return rows
        for ln in path.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception:
                continue
    except Exception:
        return rows
    return rows


def analyze(outcomes_path: "Path | str | None" = None,
            oos_only: bool = True) -> dict:
    """End-to-end CLI/import entrypoint: load outcomes, load the deployed
    scorer, run ``action_skill``.

    ``oos_only`` (default True) limits the analysis to the most recent
    20% of records by ``sim_date`` via ``validation.split_outcomes_temporal``
    — the SAME holdout `_train_decision_scorer` reports `oos_rmse`/`oos_ic`
    on, so this metric and the per-cycle ledger describe the same slice.
    The split is best-effort: a split failure degrades to "use all
    records" rather than a crash (the
    ``_train_decision_scorer.split_failure`` precedent).
    """
    root = Path(__file__).resolve().parent.parent.parent
    if outcomes_path is None:
        outcomes_path = root / "data" / "decision_outcomes.jsonl"
    records = _load_outcomes(Path(outcomes_path))

    slice_label = "all"
    if oos_only and records:
        try:
            from paper_trader.validation import split_outcomes_temporal
            _, oos = split_outcomes_temporal(records, oos_fraction=0.2)
            if oos:
                records = oos
                slice_label = "oos"
        except Exception:
            # Honest degrade — same discipline as
            # _train_decision_scorer's split fallback.
            pass

    try:
        from paper_trader.ml.decision_scorer import DecisionScorer
        scorer = DecisionScorer()
    except Exception as e:
        return {
            "status": "error",
            "verdict": "INSUFFICIENT_DATA",
            "n_records": 0,
            "by_action": {a: {"n": 0, "rank_ic": None, "dir_acc": None,
                              "mean_aligned_return": None,
                              "verdict": "INSUFFICIENT"} for a in _ACTIONS},
            "hint": f"scorer load failed: {type(e).__name__}",
            "slice": slice_label,
        }

    rep = action_skill(scorer, records)
    rep["slice"] = slice_label
    return rep


def _cli() -> int:
    """`python3 -m paper_trader.ml.action_skill` — per-action OOS rank skill
    over the live ``decision_outcomes.jsonl``. Read-only; never writes
    anything. Exit 0 healthy / insufficient, 2 if any action is INVERTED
    (so an operator/cron can branch on it, exactly like calibration._cli
    / persona_skill._cli)."""
    rep = analyze()
    print(f"slice={rep.get('slice', 'all')}  "
          f"aligned_outcomes={rep['n_records']}")
    print(f"VERDICT: {rep['verdict']}  ({rep['hint']})")
    print(f"  {'action':<6} {'n':>6} {'rank_ic':>9} {'dir_acc':>8} "
          f"{'mean_ret':>9}  verdict")
    for a in _ACTIONS:
        e = rep["by_action"][a]
        ic_s = f"{e['rank_ic']:>+9.3f}" if e["rank_ic"] is not None else f"{'n/a':>9}"
        da_s = f"{e['dir_acc']:>8.3f}" if e["dir_acc"] is not None else f"{'n/a':>8}"
        mr_s = f"{e['mean_aligned_return']:>+9.2f}" if e["mean_aligned_return"] is not None else f"{'n/a':>9}"
        print(f"  {a:<6} {e['n']:>6} {ic_s} {da_s} {mr_s}  {e['verdict']}")
    return 0 if rep["verdict"] != "HAS_INVERTED_ACTION" else 2


if __name__ == "__main__":
    raise SystemExit(_cli())

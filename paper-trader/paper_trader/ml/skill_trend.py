"""Scorer-skill trend diagnostic — read-only.

`run_continuous_backtests.py::_append_scorer_skill_log` writes one structured
row per cycle to `data/scorer_skill_log.jsonl` (status, train_n, val_rmse,
oos_n, oos_rmse, oos_dir_acc, oos_ic, gate_active). AGENTS.md calls that
ledger *"the canonical instrument for the negative-OOS-skill question"* — but
there was no reader: a skeptical quant had to `tail -f` JSONL and eyeball it.

This module answers the question the ledger exists for, with an exact verdict:
**is the DecisionScorer's out-of-sample skill better than the trivial
mean-predictor baseline, holding at the documented negative-skill plateau, or
degrading?**

The baseline is computed *fresh* from the current `decision_outcomes.jsonl`
temporal-OOS slice — NOT hardcoded to the AGENTS.md σ≈11.7 figure, which is
explicitly regime-dependent. The RMSE a constant predictor of the OOS-target
mean achieves equals the population σ of those targets (with the same SELL
sign-flip `evaluate_scorer_oos`/`train_scorer` apply), so it is the exact,
regime-current comparator for the ledger's `oos_rmse`.

Same operational discipline as `paper_trader/ml/calibration.py`: read-only,
no train, no pickle / `build_features` / `N_FEATURES` touch, no trade path —
safe to run against the live unattended loop. Never raises on bad input.

```bash
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.skill_trend
```
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Need this many `status=="ok"` ledger rows carrying a numeric oos_rmse
# before a trend verdict is meaningful (mirrors calibration.MIN_PAIRS intent).
MIN_CYCLES = 5
# Rolling window for the "recent" aggregate vs the older tail.
RECENT_CYCLES = 10
# ±band around the fresh baseline RMSE that counts as "indistinguishable from
# the mean predictor". 10% mirrors calibration's tolerance feel.
RMSE_TOL = 0.10
# Median OOS rank-IC above this is real directional skill. The _ml_decide gate
# acts ONLY on the prediction's sign/bucket (CLAUDE.md §6), so a model with
# poor RMSE but positive IC can still carry gate edge — exactly the nuance
# `_oos_rank_metrics` / calibration's DIRECTIONAL verdict capture.
IC_MIN = 0.05


def _median(xs: list[float]) -> float | None:
    vals = [float(x) for x in xs if x is not None and float(x) == float(x)]
    if not vals:
        return None
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def load_skill_ledger(path: Path | str) -> list[dict]:
    """Robust JSONL load of the scorer-skill ledger. Skips unparseable lines.

    Never raises — a missing/corrupt ledger yields ``[]`` so callers degrade
    to ``INSUFFICIENT_DATA`` rather than crashing (the ledger is best-effort
    by construction; a reader of it must be too)."""
    p = Path(path)
    rows: list[dict] = []
    try:
        if not p.exists():
            return rows
        for ln in p.read_text().splitlines():
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


def mean_predictor_baseline_rmse(
    outcome_records: list[dict], oos_fraction: float = 0.2
) -> float | None:
    """RMSE the trivial *predict-the-mean* model scores on the temporal-OOS
    slice — the exact comparator for the ledger's ``oos_rmse``.

    Reuses `validation.split_outcomes_temporal` (the SAME split the continuous
    loop uses) so the baseline describes the same held-out slice the logged
    `oos_rmse` was measured on. Applies the SELL `-forward_return_5d` sign-flip
    `train_scorer`/`evaluate_scorer_oos` apply so "good" has one meaning.

    RMSE of a constant predictor c against targets a is minimized at
    c=mean(a), where it equals the population σ of a — so that σ *is* the
    mean-predictor RMSE. Never raises → ``None`` on any fault."""
    try:
        if not outcome_records:
            return None
        from paper_trader.validation import split_outcomes_temporal
        from paper_trader.ml.decision_scorer import _to_float

        _, oos = split_outcomes_temporal(outcome_records, oos_fraction=oos_fraction)
        targets: list[float] = []
        for r in oos:
            a = _to_float(r.get("forward_return_5d"), 0.0)
            if str(r.get("action") or "BUY").upper() == "SELL":
                a = -a
            if a == a:  # drop non-finite defensively
                targets.append(a)
        if len(targets) < 2:
            return None
        arr = np.asarray(targets, dtype=np.float64)
        # Population std (ddof=0) == RMSE of predicting the sample mean.
        return float(np.sqrt(np.mean((arr - arr.mean()) ** 2)))
    except Exception:
        return None


def skill_trend_report(
    ledger_rows: list[dict],
    baseline_rmse: float | None,
    recent_n: int = RECENT_CYCLES,
) -> dict:
    """Aggregate the ledger into a verdict vs the fresh mean-predictor baseline.

    Verdicts (exact-value test-locked in tests/test_skill_trend.py):
      * ``INSUFFICIENT_DATA``        — < MIN_CYCLES usable rows, or no baseline
      * ``BEATS_MEAN_PREDICTOR``     — recent median oos_rmse ≤ baseline·(1−TOL)
      * ``NEGATIVE_OOS_SKILL``       — recent median oos_rmse ≥ baseline·(1+TOL)
                                       AND recent median oos_ic ≤ IC_MIN
      * ``DIRECTIONAL_BUT_HIGH_ERROR``— RMSE ≥ baseline·(1+TOL) but median
                                        oos_ic > IC_MIN (gate may still carry
                                        edge — it acts on sign, not magnitude)
      * ``BORDERLINE``               — RMSE within the ±TOL band of baseline
    """
    ok = [r for r in ledger_rows
          if str(r.get("status")) == "ok"
          and r.get("oos_rmse") is not None
          and float(r.get("oos_rmse")) == float(r.get("oos_rmse"))]
    n = len(ok)
    out: dict = {
        "verdict": "INSUFFICIENT_DATA",
        "n_cycles_total": len(ledger_rows),
        "n_cycles_usable": n,
        "baseline_rmse": (round(baseline_rmse, 4)
                          if baseline_rmse is not None else None),
        "recent_n": recent_n,
        "recent_median_oos_rmse": None,
        "older_median_oos_rmse": None,
        "overall_median_oos_rmse": None,
        "recent_median_oos_ic": None,
        "recent_median_oos_dir_acc": None,
        "recent_median_val_rmse": None,
        "gate_active_fraction": None,
        "trend": "UNKNOWN",
        "hint": "",
    }
    if ledger_rows:
        ga = [1.0 if r.get("gate_active") else 0.0 for r in ledger_rows]
        out["gate_active_fraction"] = round(sum(ga) / len(ga), 4)

    if n < MIN_CYCLES or baseline_rmse is None or baseline_rmse <= 0:
        out["hint"] = (f"need ≥{MIN_CYCLES} ok cycles + a baseline; "
                       f"have {n} usable, baseline="
                       f"{out['baseline_rmse']}")
        return out

    recent = ok[-recent_n:]
    older = ok[:-recent_n] if len(ok) > recent_n else []

    rec_rmse = _median([r["oos_rmse"] for r in recent])
    old_rmse = _median([r["oos_rmse"] for r in older]) if older else None
    out["recent_median_oos_rmse"] = round(rec_rmse, 4) if rec_rmse is not None else None
    out["older_median_oos_rmse"] = round(old_rmse, 4) if old_rmse is not None else None
    out["overall_median_oos_rmse"] = round(_median([r["oos_rmse"] for r in ok]), 4)
    out["recent_median_oos_ic"] = (
        round(_median([r.get("oos_ic") for r in recent]), 4)
        if _median([r.get("oos_ic") for r in recent]) is not None else None)
    out["recent_median_oos_dir_acc"] = (
        round(_median([r.get("oos_dir_acc") for r in recent]), 4)
        if _median([r.get("oos_dir_acc") for r in recent]) is not None else None)
    out["recent_median_val_rmse"] = (
        round(_median([r.get("val_rmse") for r in recent]), 4)
        if _median([r.get("val_rmse") for r in recent]) is not None else None)

    # Trend: lower RMSE is better, so recent < older ⇒ IMPROVING.
    if old_rmse is not None and rec_rmse is not None:
        if rec_rmse <= old_rmse * (1.0 - RMSE_TOL):
            out["trend"] = "IMPROVING"
        elif rec_rmse >= old_rmse * (1.0 + RMSE_TOL):
            out["trend"] = "DEGRADING"
        else:
            out["trend"] = "STABLE"

    hi = baseline_rmse * (1.0 + RMSE_TOL)
    lo = baseline_rmse * (1.0 - RMSE_TOL)
    ic = out["recent_median_oos_ic"]
    if rec_rmse <= lo:
        out["verdict"] = "BEATS_MEAN_PREDICTOR"
        out["hint"] = (f"recent oos_rmse {rec_rmse:.2f} < mean-predictor "
                       f"baseline {baseline_rmse:.2f}")
    elif rec_rmse >= hi:
        if ic is not None and ic > IC_MIN:
            out["verdict"] = "DIRECTIONAL_BUT_HIGH_ERROR"
            out["hint"] = (f"oos_rmse {rec_rmse:.2f} ≥ baseline "
                           f"{baseline_rmse:.2f} (worse than mean) but median "
                           f"oos_ic {ic:+.2f} > {IC_MIN} — gate acts on sign")
        else:
            out["verdict"] = "NEGATIVE_OOS_SKILL"
            out["hint"] = (f"oos_rmse {rec_rmse:.2f} ≥ baseline "
                           f"{baseline_rmse:.2f} AND no directional skill "
                           f"(median oos_ic {ic}) — worse than predicting the "
                           f"mean")
    else:
        out["verdict"] = "BORDERLINE"
        out["hint"] = (f"recent oos_rmse {rec_rmse:.2f} within ±{RMSE_TOL:.0%} "
                       f"of mean-predictor baseline {baseline_rmse:.2f}")
    return out


def analyze(ledger_path: Path | str, outcomes_path: Path | str) -> dict:
    """Load the ledger + compute the fresh baseline + return the full report."""
    rows = load_skill_ledger(ledger_path)
    outcomes: list[dict] = []
    try:
        p = Path(outcomes_path)
        if p.exists():
            for ln in p.read_text().splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    outcomes.append(json.loads(ln))
                except Exception:
                    continue
    except Exception:
        outcomes = []
    baseline = mean_predictor_baseline_rmse(outcomes)
    return skill_trend_report(rows, baseline)


def _cli() -> int:
    """`python3 -m paper_trader.ml.skill_trend` — read-only scorer-skill trend
    of the live ledger vs the current mean-predictor baseline."""
    root = Path(__file__).resolve().parent.parent.parent
    ledger = root / "data" / "scorer_skill_log.jsonl"
    outcomes = root / "data" / "decision_outcomes.jsonl"
    rep = analyze(ledger, outcomes)
    print(f"VERDICT: {rep['verdict']}  ({rep['hint']})")
    print(f"  cycles: {rep['n_cycles_usable']} usable / "
          f"{rep['n_cycles_total']} total   "
          f"gate_active={rep['gate_active_fraction']}")
    print(f"  mean-predictor baseline RMSE = {rep['baseline_rmse']}")
    print(f"  oos_rmse  recent={rep['recent_median_oos_rmse']}  "
          f"older={rep['older_median_oos_rmse']}  "
          f"overall={rep['overall_median_oos_rmse']}  "
          f"trend={rep['trend']}")
    print(f"  oos_ic recent={rep['recent_median_oos_ic']}  "
          f"oos_dir_acc recent={rep['recent_median_oos_dir_acc']}  "
          f"val_rmse recent={rep['recent_median_val_rmse']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())

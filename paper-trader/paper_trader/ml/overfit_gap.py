"""Generalization-gap (val vs OOS) trend diagnostic — read-only.

`run_continuous_backtests.py::_append_scorer_skill_log` writes one row per
cycle to `data/scorer_skill_log.jsonl` carrying BOTH `val_rmse` (the
random-split in-sample-ish error `train_scorer` reports) and `oos_rmse` (the
temporal-holdout error `_train_decision_scorer` measures). AGENTS.md cites the
`val_rmse ≪ oos_rmse` divergence as *"textbook overfit"* over and over, and
HEAD commit `5a0af2d` ("regularize DecisionScorer MLP — (32,16)+L2+early-stop
kills the val≪oos overfit") exists for the sole purpose of closing that gap.

Yet **nothing trends the gap itself**:

  * `skill_trend.py`     — verdicts `oos_rmse` vs a fresh mean-predictor
    baseline (is the scorer better than predicting the mean?). It only
    *reports* `recent_median_val_rmse` as a side metric; no gap verdict.
  * `baseline_trend.py`  — verdicts `ic_gap` (MLP rank-IC − best one-liner
    rank-IC). A different axis entirely.

So a skeptical quant could not durably answer the one question commit
`5a0af2d` is supposed to settle: **once the loop redeploys on the regularized
net, does the val/oos gap actually shrink — or is the network still
memorizing its training fold?** This module answers exactly that, with an
exact verdict, off the same ledger.

Design choices for a trustworthy, non-flaky verdict:

* The verdict is driven by the **ratio** `oos_rmse / val_rmse`, not the
  absolute `oos − val` pp. The continuous loop draws random 1–10yr windows
  whose target σ varies several-fold (a 3×-ETF bull window vs a flat one), so
  an absolute-pp gap conflates regime σ with overfit. The ratio is
  scale-free: ratio≈1 ⇒ generalizes; ratio≫1 ⇒ memorized the fold
  regardless of the window's σ. (Documented anchors: the prior memorizing
  (64,32,16) net posted oos≈16.7 / val≈10.7 ⇒ ratio≈1.56; the regularized
  target is ratio→~1.0; the live pre-`5a0af2d` ledger shows 1.16–1.56.)
* Aggregates the **median** ratio over a recent window (per-cycle ratio is
  noisy because each cycle is one random window) — the `skill_trend` /
  `baseline_trend` precedent.
* Reuses `skill_trend.load_skill_ledger`, `_median`, `MIN_CYCLES`,
  `RECENT_CYCLES` verbatim (single source of truth — the same DRY rule
  `baseline_trend` follows importing `IC_MARGIN` from `baseline_compare`, and
  `_oos_rank_metrics` reusing `calibration._spearman`). If the ledger format
  or the recent-window size ever changes, this trender and the sibling stay
  aligned by construction, never drift.

Same operational discipline as `skill_trend` / `calibration`: read-only, no
train, no pickle / `build_features` / `N_FEATURES` touch, no trade path —
safe to run against the live unattended loop. Never raises on bad input.

```bash
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.overfit_gap
```
"""
from __future__ import annotations

from pathlib import Path

# Single source of truth — reuse the ledger loader, median helper and the
# usable-cycle / recent-window sizing the sibling scorer-skill trender already
# defines. Importing (not re-implementing) means a ledger-schema change can
# never make this verdict and `skill_trend`'s disagree about which rows count.
from paper_trader.ml.skill_trend import (
    MIN_CYCLES,
    RECENT_CYCLES,
    _median,
    load_skill_ledger,
)

# `oos_rmse / val_rmse` at or above this ⇒ the OOS error is ≥40% larger than
# the in-sample error: the documented memorizing-net regime (oos≈16.7 /
# val≈10.7 ≈ 1.56 sat here; the live pre-regularization ledger's worst
# cycles — 16.68/10.74≈1.55, 14.04/9.01≈1.56 — are SEVERE by this bar).
SEVERE_RATIO = 1.40
# At/above this but below SEVERE ⇒ a real but moderate gap (the live ledger's
# better cycles — 9.63/7.98≈1.21, 10.63/8.91≈1.19 — land here).
MILD_RATIO = 1.15
# Relative band for the recent-vs-older trend axis (mirrors
# skill_trend.RMSE_TOL feel — 10%). Lower ratio is better, so a recent median
# ≤ older·(1−TOL) is the gap shrinking ⇒ IMPROVING.
RATIO_TOL = 0.10


def _usable_rows(ledger_rows: list[dict]) -> list[dict]:
    """Rows where the val/oos ratio is well-defined.

    A row qualifies iff ``status=="ok"`` AND both ``val_rmse`` and
    ``oos_rmse`` are present, finite, and ``val_rmse > 0`` (so the ratio is
    real and not a divide-by-zero). The numpy-lstsq fallback path writes
    ``val_rmse=NaN`` (no holdout on a sklearn-less host) — those rows carry
    no gap signal and are correctly excluded here, mirroring
    ``skill_trend``'s ``oos_rmse``-finite filter."""
    out: list[dict] = []
    for r in ledger_rows:
        if str(r.get("status")) != "ok":
            continue
        v = r.get("val_rmse")
        o = r.get("oos_rmse")
        if v is None or o is None:
            continue
        try:
            vf = float(v)
            of = float(o)
        except (TypeError, ValueError):
            continue
        if vf != vf or of != of:  # NaN
            continue
        if vf <= 0.0:
            continue
        out.append(r)
    return out


def overfit_gap_report(
    ledger_rows: list[dict],
    recent_n: int = RECENT_CYCLES,
) -> dict:
    """Aggregate the ledger into a generalization-gap verdict.

    Verdicts (exact-value test-locked in tests/test_overfit_gap.py):
      * ``INSUFFICIENT_DATA`` — < ``MIN_CYCLES`` usable rows (status==ok with
        a finite val_rmse>0 and finite oos_rmse).
      * ``SEVERE_OVERFIT``    — recent median ratio ≥ ``SEVERE_RATIO``: the
        OOS error is ≥40% above in-sample — the memorizing-net signature.
      * ``MILD_OVERFIT``      — ``MILD_RATIO`` ≤ recent median ratio
        < ``SEVERE_RATIO``: a real but moderate gap.
      * ``WELL_GENERALIZED``  — recent median ratio < ``MILD_RATIO``: the
        scorer's held-out error tracks its in-sample error (commit
        ``5a0af2d``'s stated goal, once it deploys).

    Boundaries are inclusive at the lower edge (``>=``), matching
    ``skill_trend``'s threshold style: a ratio of exactly 1.40 is
    ``SEVERE_OVERFIT``; exactly 1.15 is ``MILD_OVERFIT``; 1.149… is
    ``WELL_GENERALIZED``.

    ``trend`` is recent-vs-older median ratio (lower = better): recent ≤
    older·(1−TOL) ⇒ ``IMPROVING``; recent ≥ older·(1+TOL) ⇒ ``DEGRADING``;
    within the band ⇒ ``STABLE``; no older tail ⇒ ``UNKNOWN``.

    Never raises — any fault degrades to ``INSUFFICIENT_DATA``.
    """
    out: dict = {
        "verdict": "INSUFFICIENT_DATA",
        "n_cycles_total": len(ledger_rows) if ledger_rows else 0,
        "n_cycles_usable": 0,
        "recent_n": recent_n,
        "recent_median_ratio": None,
        "older_median_ratio": None,
        "overall_median_ratio": None,
        "recent_median_abs_gap": None,
        "recent_median_val_rmse": None,
        "recent_median_oos_rmse": None,
        # Decisive cross-signal: SEVERE_OVERFIT *while* the gate is sizing
        # trades (gate_active ⇔ deployed n_train ≥ 500, invariant #5) is the
        # "underwriting variance on a demonstrably memorized net" state.
        "gate_active_fraction": None,
        "trend": "UNKNOWN",
        "hint": "",
    }
    try:
        if ledger_rows:
            ga = [1.0 if r.get("gate_active") else 0.0 for r in ledger_rows]
            out["gate_active_fraction"] = round(sum(ga) / len(ga), 4)

        usable = _usable_rows(ledger_rows or [])
        n = len(usable)
        out["n_cycles_usable"] = n
        if n < MIN_CYCLES:
            out["hint"] = (f"need ≥{MIN_CYCLES} ok cycles with a finite "
                           f"val_rmse>0 and oos_rmse; have {n} usable")
            return out

        def _ratio(r: dict) -> float:
            return float(r["oos_rmse"]) / float(r["val_rmse"])

        def _abs_gap(r: dict) -> float:
            return float(r["oos_rmse"]) - float(r["val_rmse"])

        recent = usable[-recent_n:]
        older = usable[:-recent_n] if n > recent_n else []

        rec_ratio = _median([_ratio(r) for r in recent])
        old_ratio = _median([_ratio(r) for r in older]) if older else None
        out["recent_median_ratio"] = (
            round(rec_ratio, 4) if rec_ratio is not None else None)
        out["older_median_ratio"] = (
            round(old_ratio, 4) if old_ratio is not None else None)
        out["overall_median_ratio"] = round(
            _median([_ratio(r) for r in usable]), 4)
        out["recent_median_abs_gap"] = round(
            _median([_abs_gap(r) for r in recent]), 4)
        out["recent_median_val_rmse"] = round(
            _median([float(r["val_rmse"]) for r in recent]), 4)
        out["recent_median_oos_rmse"] = round(
            _median([float(r["oos_rmse"]) for r in recent]), 4)

        # Trend: lower ratio is better, so recent < older ⇒ IMPROVING.
        if old_ratio is not None and rec_ratio is not None:
            if rec_ratio <= old_ratio * (1.0 - RATIO_TOL):
                out["trend"] = "IMPROVING"
            elif rec_ratio >= old_ratio * (1.0 + RATIO_TOL):
                out["trend"] = "DEGRADING"
            else:
                out["trend"] = "STABLE"

        if rec_ratio is None:
            out["hint"] = "no finite recent ratio"
            return out
        if rec_ratio >= SEVERE_RATIO:
            out["verdict"] = "SEVERE_OVERFIT"
            out["hint"] = (
                f"recent median oos/val ratio {rec_ratio:.2f} ≥ "
                f"{SEVERE_RATIO:.2f} — OOS error ≥"
                f"{(SEVERE_RATIO - 1.0) * 100:.0f}% above in-sample; the "
                f"memorizing-net signature (gate_active="
                f"{out['gate_active_fraction']})")
        elif rec_ratio >= MILD_RATIO:
            out["verdict"] = "MILD_OVERFIT"
            out["hint"] = (
                f"recent median oos/val ratio {rec_ratio:.2f} in "
                f"[{MILD_RATIO:.2f}, {SEVERE_RATIO:.2f}) — a real but "
                f"moderate generalization gap")
        else:
            out["verdict"] = "WELL_GENERALIZED"
            out["hint"] = (
                f"recent median oos/val ratio {rec_ratio:.2f} < "
                f"{MILD_RATIO:.2f} — held-out error tracks in-sample error")
        return out
    except Exception:
        return {
            "verdict": "INSUFFICIENT_DATA",
            "n_cycles_total": len(ledger_rows) if ledger_rows else 0,
            "n_cycles_usable": 0,
            "recent_n": recent_n,
            "recent_median_ratio": None,
            "older_median_ratio": None,
            "overall_median_ratio": None,
            "recent_median_abs_gap": None,
            "recent_median_val_rmse": None,
            "recent_median_oos_rmse": None,
            "gate_active_fraction": None,
            "trend": "UNKNOWN",
            "hint": "error",
        }


def analyze(ledger_path: Path | str) -> dict:
    """Load the scorer-skill ledger and return the full gap report.

    Only needs the ledger (val_rmse + oos_rmse are both already persisted per
    cycle) — unlike `skill_trend.analyze`, no `decision_outcomes.jsonl` read
    is required. Never raises (`load_skill_ledger` degrades to `[]`)."""
    return overfit_gap_report(load_skill_ledger(ledger_path))


def _cli() -> int:
    """`python3 -m paper_trader.ml.overfit_gap` — read-only val/oos
    generalization-gap trend of the live scorer-skill ledger.

    Exit code mirrors the sibling trenders so a cron can branch on "the net
    is *persistently* memorizing its training fold": 0 on WELL_GENERALIZED /
    INSUFFICIENT_DATA, 2 on MILD_OVERFIT / SEVERE_OVERFIT."""
    root = Path(__file__).resolve().parent.parent.parent
    ledger = root / "data" / "scorer_skill_log.jsonl"
    rep = analyze(ledger)
    print(f"VERDICT: {rep['verdict']}  ({rep['hint']})")
    print(f"  cycles: {rep['n_cycles_usable']} usable / "
          f"{rep['n_cycles_total']} total   "
          f"gate_active={rep['gate_active_fraction']}")
    print(f"  oos/val ratio  recent={rep['recent_median_ratio']}  "
          f"older={rep['older_median_ratio']}  "
          f"overall={rep['overall_median_ratio']}  "
          f"trend={rep['trend']}")
    print(f"  recent medians:  val_rmse={rep['recent_median_val_rmse']}  "
          f"oos_rmse={rep['recent_median_oos_rmse']}  "
          f"abs_gap={rep['recent_median_abs_gap']}")
    return 2 if rep["verdict"] in ("MILD_OVERFIT", "SEVERE_OVERFIT") else 0


if __name__ == "__main__":
    raise SystemExit(_cli())

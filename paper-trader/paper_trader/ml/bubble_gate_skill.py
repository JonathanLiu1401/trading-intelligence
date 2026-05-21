"""Bubble-gate REALIZED skill — does ``wk52_pos > 0.80`` suppression
actually improve returns, or is it dumping legitimate breakouts?

This is a **read-only diagnostic**. It never trains, never touches
``decision_scorer.pkl``, ``decision_outcomes.jsonl``, ``build_features``,
``N_FEATURES``, or any trade path — same operational discipline as
``gate_audit`` / ``gate_pnl`` / ``gate_realized`` / ``calibration`` /
``skill_trend``. Safe to run against the live unattended loop.

**The gate this module audits.** ``paper_trader/backtest.py::_ml_decide``
applies a peak-penalty / suppression to a BUY whose 52-week position
exceeds 0.80:

    if buy_ticker:
        _w52 = quant.get(buy_ticker, {}).get("wk52_pos")
        if isinstance(_w52, (int, float)) and _w52 > 0.80:
            _peak_penalty = (_w52 - 0.80) * 20.0
            if best_score - _peak_penalty < buy_threshold:
                buy_ticker = None

The implicit hypothesis: ``wk52_pos > 0.80`` predicts negative forward
return — buying near the high is a bubble trap. **As implemented** the
gate only sees BUYs that *passed* the suppression (the rejected ones
left no outcome row), so the corpus is truncated by the gate itself.
But every BUY with ``wk52_pos`` near 0.80 is still informative — does
the realized 5d return CONTINUE to weaken as ``wk52_pos`` climbs into
the 0.7–0.8 band the gate considers "safe"?

**What this measures.** Bucket every FILLED BUY outcome row by its
``wk52_pos`` (the field captured by ``_compute_decision_outcomes`` since
the ``wk52_pos`` outcomes feature shipped) and report the mean realized
5d return per bucket. If the at-high (0.7–0.8) bucket has the SAME or
HIGHER mean realized than the mid bucket, the bubble-gate hypothesis
that ``wk52_pos`` is a bearish signal is falsified by the data.

**Why this is not** ``gate_audit`` / ``gate_pnl`` / ``gate_realized``.
Those instruments audit the DecisionScorer conviction gate (the ±10/±5/0
prediction buckets). This audits a SEPARATE structural gate — the
position-relative 52-week-high penalty in ``_ml_decide`` — that fires
BEFORE the scorer gate and is invariant to the scorer's retrain state.
A reading quant needs both signals to know which gate is justified.

**Honest limitations.**

* **Survivorship bias inside the corpus.** The corpus contains only BUYs
  that the bubble gate ALLOWED through. The pre-gate counterfactual
  (what would the realized return have been for the suppressed rows?)
  is unobservable from this file — that question is the domain of a
  perm-test run on a code path with the gate disabled. This diagnostic
  measures only the WITHIN-CORPUS gradient, which is what the gate's
  threshold tuning actually trades off.
* **wk52_pos comes from the captured field**, which depends on
  ``_compute_technical_indicators`` having ≥60 closes for the ticker at
  ``sim_date``. Older outcomes pre-date the wk52_pos capture and read
  None — those rows are simply dropped from the audit (the
  ``horizon_audit`` / ``gate_realized`` "populates as the loop runs"
  precedent).

```bash
cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.bubble_gate_skill
cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.bubble_gate_skill --all
```
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Verdict thresholds — module-level so tests assert exact verdicts and a
# tuning change is a single reviewable edit (== gate_audit / gate_realized).
MIN_TOTAL = 30      # need a real BUY sample before any verdict (== gate_realized.MIN_TOTAL)
MIN_BUCKET_N = 5    # min trades in each compared bucket
EDGE_TOL_PP = 1.0   # |at_high − mid| band that reads as noise (== gate_realized.EDGE_TOL_PP)

# Bucket boundaries on wk52_pos (0.0 = 52-week low, 1.0 = 52-week high).
# `near_high` is the band immediately under the gate threshold; `at_high`
# is the band that ONLY exists because the gate's peak-penalty is soft (not
# a hard cutoff — see `_ml_decide`'s `_peak_penalty` logic: a strong score
# can still survive the suppression). If the at_high bucket is well-
# populated, the gate is letting through high-conviction breakouts; if
# nearly empty, the gate's soft penalty is in practice a hard cutoff.
_BUCKETS = [
    ("deep_discount", 0.0, 0.50),
    ("mid",           0.50, 0.70),
    ("near_high",     0.70, 0.80),
    ("at_high",       0.80, 1.0001),  # inclusive of 1.0 (at the high)
]


def _f(v):
    """Finite float or None — the ``gate_realized._f`` discipline."""
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _bucket_for(wk52: float) -> str | None:
    """Return the bucket name for a given ``wk52_pos`` value, or None if
    out of the [0, 1] range. Boundaries: lo ≤ x < hi (right-open) except
    the last bucket which is right-CLOSED on 1.0 to capture a 52-week-high
    pure peak."""
    if wk52 < 0.0 or wk52 > 1.0:
        return None
    for name, lo, hi in _BUCKETS:
        if lo <= wk52 < hi:
            return name
    return None


def bubble_gate_skill_report(rows) -> dict:
    """Bucket FILLED-BUY outcome rows by ``wk52_pos`` and report each
    bucket's mean realized 5d return.

    ``rows`` — any iterable of ``decision_outcomes.jsonl``-shaped dicts.
    For each row this reads, and never re-predicts or re-fetches:

      * ``action == "BUY"`` (SELL/HOLD rows are excluded — the bubble
        gate is BUY-only by construction)
      * ``wk52_pos`` — the row's captured 52-week position (None / out of
        [0,1] / non-finite drops that row)
      * ``forward_return_5d`` — realized return (None drops that row)

    The verdict is driven solely by the 5d-realized spread between the
    ``at_high`` and ``mid`` buckets:

    | Verdict | Meaning |
    |---------|---------|
    | ``WK52_NOT_YET_POPULATED`` | 0 BUYs carry a non-null ``wk52_pos`` — the loop predates the wk52_pos capture (commit message references the outcomes-side wk52_pos field) |
    | ``INSUFFICIENT_DATA`` | some captured rows exist but < ``MIN_TOTAL`` BUYs, or either compared bucket < ``MIN_BUCKET_N`` |
    | ``BUBBLE_GATE_JUSTIFIED`` | ``at_high − mid < −EDGE_TOL_PP`` — buying at the 52-week high realized *less* than buying mid-range; the gate's hypothesis (that wk52_pos is bearish) is supported |
    | ``BUBBLE_GATE_NEUTRAL`` | \\|spread\\| ≤ ``EDGE_TOL_PP`` — at-high and mid realize within noise; the gate adds variance without realized edge |
    | ``BUBBLE_GATE_HARMFUL`` | ``at_high − mid > +EDGE_TOL_PP`` — buying at the 52-week high realized MORE; suppressing those buys was leaving alpha on the table (the documented "near 52-week high IS a breakout signal" outcome) |

    ``bucket_monotone_fraction`` (adjacent buckets in wk52_pos order,
    realized-mean non-decreasing) is reported but **NOT** in the verdict
    — the gate_audit "monotone is informational" precedent. The verdict
    stays crisply exact-value testable on the two-bucket spread alone.

    Returns a JSON-safe dict. Never raises.
    """
    by_bucket: dict[str, list[float]] = {name: [] for name, _, _ in _BUCKETS}
    n_buys = 0           # BUY rows total (post action filter)
    n_with_wk52 = 0      # BUY rows with finite, in-range wk52_pos
    n_with_fwd = 0       # BUY rows with both wk52_pos AND forward_return_5d

    try:
        it = list(rows or [])
    except Exception:
        it = []

    for r in it:
        if not isinstance(r, dict):
            continue
        action = str(r.get("action") or "").upper()
        if action != "BUY":
            continue
        n_buys += 1
        wk52 = _f(r.get("wk52_pos"))
        if wk52 is None:
            continue
        bucket = _bucket_for(wk52)
        if bucket is None:
            # Out-of-range value — not "no wk52_pos captured", but
            # corrupt / mis-encoded. Don't count it as captured either
            # (the n_with_wk52 metric is a coverage report, not a
            # nan-input tally).
            continue
        n_with_wk52 += 1
        fwd = _f(r.get("forward_return_5d"))
        if fwd is None:
            continue
        n_with_fwd += 1
        by_bucket[bucket].append(fwd)

    def _stats(vals: list[float]) -> dict:
        if not vals:
            return {"n": 0, "mean": None, "lo": None, "hi": None}
        a = np.asarray(vals, dtype=np.float64)
        return {
            "n": int(a.size),
            "mean": round(float(a.mean()), 4),
            "lo": round(float(a.min()), 4),
            "hi": round(float(a.max()), 4),
        }

    buckets_out = []
    for name, lo, hi in _BUCKETS:
        s = _stats(by_bucket[name])
        buckets_out.append({
            "bucket": name,
            "wk52_lo": lo,
            "wk52_hi": (1.0 if hi > 1.0 else hi),  # display-clean upper edge
            "n": s["n"],
            "mean_realized_5d": s["mean"],
            "lo_5d": s["lo"],
            "hi_5d": s["hi"],
        })

    base = {
        "status": "ok",
        "verdict": "WK52_NOT_YET_POPULATED",
        "measurement": "wk52_pos_realized_no_reprediction",
        "n_buys": n_buys,
        "n_with_wk52": n_with_wk52,
        "n_with_fwd": n_with_fwd,
        "buckets": buckets_out,
        "at_high_minus_mid_pp": None,
        "bucket_monotone_fraction": None,
        "hint": "",
    }

    if n_with_wk52 == 0:
        base["hint"] = (
            f"0 of {n_buys} BUY rows carry a non-null wk52_pos — the "
            f"continuous loop predates the wk52_pos outcome capture or "
            f"hasn't accumulated rows since; populates as the loop runs"
        )
        return base

    # Monotonicity across buckets with ≥1 sample, in wk52_pos order.
    present = [b for b in buckets_out if b["n"] > 0]
    if len(present) >= 2:
        steps = len(present) - 1
        nondec = sum(
            1 for j in range(steps)
            if present[j + 1]["mean_realized_5d"]
            >= present[j]["mean_realized_5d"]
        )
        base["bucket_monotone_fraction"] = round(nondec / steps, 4)

    mid = by_bucket["mid"]
    at_high = by_bucket["at_high"]
    if (n_with_fwd < MIN_TOTAL or len(mid) < MIN_BUCKET_N
            or len(at_high) < MIN_BUCKET_N):
        base["verdict"] = "INSUFFICIENT_DATA"
        base["hint"] = (
            f"need ≥{MIN_TOTAL} BUY rows with wk52_pos AND forward_return_5d, "
            f"AND ≥{MIN_BUCKET_N} in EACH of mid + at_high buckets; have "
            f"n_with_fwd={n_with_fwd}, mid={len(mid)}, at_high={len(at_high)}"
        )
        return base

    mid_mean = float(np.mean(mid))
    at_high_mean = float(np.mean(at_high))
    spread = at_high_mean - mid_mean
    base["at_high_minus_mid_pp"] = round(spread, 4)

    if spread < -EDGE_TOL_PP:
        base["verdict"] = "BUBBLE_GATE_JUSTIFIED"
        base["hint"] = (
            f"at_high BUYs realized {at_high_mean:+.2f}% < mid {mid_mean:+.2f}% "
            f"(spread {spread:+.2f}pp) — buying near the 52-week high "
            f"underperformed; the gate's wk52_pos>0.80 suppression is "
            f"supported by within-corpus data"
        )
    elif abs(spread) <= EDGE_TOL_PP:
        base["verdict"] = "BUBBLE_GATE_NEUTRAL"
        base["hint"] = (
            f"at_high {at_high_mean:+.2f}% vs mid {mid_mean:+.2f}% "
            f"(spread {spread:+.2f}pp, within ±{EDGE_TOL_PP:.1f}pp) — "
            f"no within-corpus edge for either direction; the gate adds "
            f"sizing variance without a realized return justification"
        )
    else:
        base["verdict"] = "BUBBLE_GATE_HARMFUL"
        base["hint"] = (
            f"at_high BUYs realized {at_high_mean:+.2f}% > mid {mid_mean:+.2f}% "
            f"(spread {spread:+.2f}pp) — buying at the 52-week high "
            f"OUTPERFORMED; the gate's wk52_pos>0.80 suppression is "
            f"leaving alpha on the table (52-week-high breakouts as a "
            f"momentum signal, not a bubble trap)"
        )
    return base


def analyze(outcomes_path: Path | str, oos_only: bool = True) -> dict:
    """Load the outcomes file, take the temporal-OOS slice (default) and
    run the bubble-gate skill report. **No scorer / pickle is loaded** —
    this reads only the captured fields. Read-only; never raises."""
    out: dict = {
        "status": "error",
        "verdict": "WK52_NOT_YET_POPULATED",
        "measurement": "wk52_pos_realized_no_reprediction",
        "n_buys": 0, "n_with_wk52": 0, "n_with_fwd": 0,
        "buckets": [], "slice": "all", "hint": "",
    }
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

        slice_name = "all"
        recs = records
        if oos_only:
            try:
                from paper_trader.validation import split_outcomes_temporal
                _, oos = split_outcomes_temporal(records, oos_fraction=0.2)
                if oos:
                    recs = oos
                    slice_name = "oos"
            except Exception:
                slice_name = "all"

        rep = bubble_gate_skill_report(recs)
        rep["slice"] = slice_name
        rep["n_records_total"] = len(records)
        return rep
    except Exception as e:  # pragma: no cover - defensive, never raises
        out["hint"] = f"audit failed: {type(e).__name__}: {e}"
        return out


def _cli(argv: list[str] | None = None) -> int:
    """``python3 -m paper_trader.ml.bubble_gate_skill [--all]`` — the
    wk52_pos>0.80 suppression's REALIZED skill from captured outcomes
    (no re-prediction). Read-only. Exits 2 on ``BUBBLE_GATE_HARMFUL``
    (cron-branchable — the actionable "the gate is suppressing winners"
    signal, mirroring ``gate_realized``'s exit-2-on-harmful)."""
    import sys
    argv = sys.argv[1:] if argv is None else argv
    oos_only = "--all" not in argv
    root = Path(__file__).resolve().parent.parent.parent
    rep = analyze(root / "data" / "decision_outcomes.jsonl", oos_only=oos_only)
    print(f"VERDICT: {rep['verdict']}  ({rep.get('hint', '')})")
    print(f"  slice={rep.get('slice')}  "
          f"n_buys={rep.get('n_buys')}  "
          f"n_with_wk52={rep.get('n_with_wk52')}  "
          f"at_high−mid={rep.get('at_high_minus_mid_pp')}pp  "
          f"bucket_monotone={rep.get('bucket_monotone_fraction')}")
    for b in rep.get("buckets", []):
        mr = b["mean_realized_5d"]
        mr_s = f"{mr:+7.2f}%" if mr is not None else "    n/a"
        print(f"  {b['bucket']:<14} [{b['wk52_lo']:.2f}..{b['wk52_hi']:.2f})  "
              f"n={b['n']:<5} mean_realized_5d={mr_s}")
    return 2 if rep.get("verdict") == "BUBBLE_GATE_HARMFUL" else 0


if __name__ == "__main__":
    raise SystemExit(_cli())

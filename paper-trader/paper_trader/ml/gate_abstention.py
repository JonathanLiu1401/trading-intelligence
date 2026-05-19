"""Off-distribution gate-abstention diagnostic — read-only.

The 2026-05-17 ``84d8234`` commit added an off-distribution abstention to
``_ml_decide``'s conviction gate: when the scorer's raw output exceeds
``PRED_CLAMP_PCT`` (±50), the prediction is treated as untrustworthy
extrapolation and the gate **leaves conviction untouched** — the multiplier
arm is skipped entirely. ``60b20d9`` then made that decision durable: every
captured BUY outcome row carries a ``gate_off_dist`` boolean recording
whether the live gate abstained on that decision.

**Nothing reports how often the guard actually fires.** ``gate_realized.py``
consumes the field (to route abstained rows to a separate bucket) but its
verdict grades only ACTED rows; the *rate* of abstention itself, the per-arm
distribution it abstained from, and whether it is rising or falling over
time are unread. That is the precise gap this module closes — the
``_append_scorer_skill_log``-wired-but-unread pattern of pass #15 / the
``baseline_skill_log``-wired-but-unread pattern of ``baseline_trend.py``,
applied to a per-row field instead of a per-cycle ledger column.

This matters operationally. Two failure modes look identical from a
distance:

* **Guard inactive** — the rate is effectively 0%. Either the model is
  always in-distribution (good — but worth knowing) OR the abstention
  threshold is too lax to ever fire (the documented "the off-dist note has
  never appeared in any live reasoning" precondition). The guard exists to
  catch the −89→+32 same-LITE-vector extrapolation AGENTS.md documents; if
  it never fires, that protection is dead code.
* **Guard rampant** — the rate is high. The model is regularly emitting
  clamped ±50 outputs the gate then refuses to act on — so the gate is
  mostly *neutral* despite being "active". The 1.30/0.60 multiplier
  spread the ``gate_audit`` / ``gate_realized`` verdicts grade is being
  applied to a shrinking fraction of decisions.

Same operational discipline as the rest of the ``paper_trader/ml/`` family:
read-only, no train, no pickle / ``build_features`` / ``N_FEATURES`` touch,
no trade path — safe to run against the live unattended loop. Never raises
on bad input.

Imports ``gate_arm`` from ``gate_audit`` (single source of truth — the
"would-have-been-arm" distribution must use the live ``_ml_decide`` arm
boundaries verbatim, the ``gate_realized``-imports-``gate_arm`` precedent)
and ``split_outcomes_temporal`` from ``validation`` for the OOS slice (the
same temporal split every sibling OOS tool uses).

```bash
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.gate_abstention
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.gate_abstention --all
```
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

# Reuse the single source of truth for the gate's arm boundaries — the
# "would-have-been arm" report below must match ``_ml_decide`` to the bit
# (the ``gate_realized``-imports-``gate_arm`` precedent).
from .gate_audit import gate_arm

# Verdict thresholds — module-level so tests assert exact verdicts and a
# tuning change is a single reviewable edit (mirrors the sibling tools).
MIN_TOTAL = 30        # need a real captured sample before any verdict
INACTIVE_MAX = 0.005  # < 0.5% rate ⇒ guard effectively dead
RAMPANT_MIN = 0.15    # ≥ 15% rate ⇒ model frequently extrapolates

# Recent vs older trend window over the captured rows (rows ordered by
# ``sim_date`` so "recent" = newest sim_dates, the temporal direction the
# scorer ledger trends use).
RECENT_FRACTION = 0.5
# Need this many usable rows on EACH side of the temporal split before a
# trend axis is meaningful — mirrors ``baseline_trend.MIN_CYCLES`` intent.
MIN_PER_SIDE = 15
# ±band on the recent-minus-older rate that reads as STABLE (in percentage
# points of rate, e.g. 0.02 = ±2pp).
TREND_TOL_PP = 0.02

_ARM_ORDER = ["strong_headwind", "mild_headwind", "neutral",
              "mild_tailwind", "strong_tailwind"]


def _f(v):
    """Finite float or None — the ``_to_float`` hardening class, local so
    this module imports nothing from ``decision_scorer`` (read-only, no
    pickle path; the ``gate_realized`` precedent)."""
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def abstention_report(rows) -> dict:
    """Compute abstention statistics over a captured-outcomes row set.

    A row counts as **captured** iff ``gate_scorer_pred`` is a finite float
    (the gate emitted a real decision — see ``_parse_gate_decision``).
    Rows with a None ``gate_scorer_pred`` (SELL / pre-``60b20d9`` /
    untrained-cycle) are not counted at all — they have no gate decision
    to characterize.

    Among captured rows:
      * ``n_abstained`` = number where ``gate_off_dist`` is True (the live
        ``_ml_decide`` left conviction untouched on that decision).
      * ``n_acted`` = captured AND not abstained.
      * ``rate`` = ``n_abstained / n_captured``.

    Additionally:
      * ``arm_dist`` — the distribution of "would-have-been arms" for
        abstained rows (what arm the captured prediction WOULD map to via
        ``gate_arm``). Tells whether abstentions cluster at the extreme
        clamp (the expected pattern — the guard fires on ±50 outputs).
      * ``top_tickers`` — top 5 tickers by abstention count (informational
        only — surfaces names whose features push the model
        off-distribution most often).
      * ``trend_rate_diff_pp`` — abstention-rate(recent half by sim_date)
        − abstention-rate(older half), in percentage points (rate units).

    Verdicts (driven by the OVERALL rate; arm/ticker/trend are
    informational descriptors — the ``gate_realized.arm_monotone_fraction``
    honesty pattern of "informational, not verdict-input"):

    | Verdict | Meaning |
    |---------|---------|
    | ``INSUFFICIENT_DATA`` | ``n_captured < MIN_TOTAL`` — no claim either way |
    | ``GUARD_INACTIVE``    | rate < ``INACTIVE_MAX`` — guard effectively dead; the −89→+32 extrapolation protection it was added for is never engaging |
    | ``GUARD_HEALTHY``     | rate ∈ [``INACTIVE_MAX``, ``RAMPANT_MIN``) — fires occasionally on real extrapolation |
    | ``GUARD_RAMPANT``     | rate ≥ ``RAMPANT_MIN`` — model frequently extrapolates; the live gate is mostly neutral (multipliers apply to a shrinking minority) |

    Trend axis (independent of the verdict axis, the ``baseline_trend``
    precedent):
      * ``IMPROVING`` — recent rate < older by > ``TREND_TOL_PP``
        (fewer extrapolations than before — model getting more
        in-distribution)
      * ``DEGRADING`` — recent > older + ``TREND_TOL_PP`` (more
        extrapolations — model drifting OOD or seeing wilder inputs)
      * ``STABLE``   — within ±``TREND_TOL_PP``
      * ``UNKNOWN``  — fewer than ``MIN_PER_SIDE`` usable rows on either
        side (the "INSUFFICIENT_LONG_HORIZON" honesty precedent)

    Returns a JSON-safe dict. Never raises.
    """
    try:
        it = list(rows or [])
    except Exception:
        it = []

    captured: list[tuple[str, float, bool]] = []  # (sim_date, pred, abstained)
    abst_ticker_counts: Counter = Counter()
    arm_counts: Counter = Counter()

    for r in it:
        if not isinstance(r, dict):
            continue
        gp = _f(r.get("gate_scorer_pred"))
        if gp is None:
            continue  # no gate decision on this row
        abstained = bool(r.get("gate_off_dist"))
        sim_date = str(r.get("sim_date") or "")
        captured.append((sim_date, gp, abstained))
        if abstained:
            tkr = str(r.get("ticker") or "")
            if tkr:
                abst_ticker_counts[tkr] += 1
            arm, _ = gate_arm(gp)
            arm_counts[arm] += 1

    n_captured = len(captured)
    n_abstained = sum(1 for _, _, a in captured if a)
    n_acted = n_captured - n_abstained
    rate = (n_abstained / n_captured) if n_captured else 0.0

    arm_dist = [
        {"arm": a, "n_abstained": int(arm_counts.get(a, 0))}
        for a in _ARM_ORDER
    ]
    top_tickers = [
        {"ticker": t, "n_abstained": int(c)}
        for t, c in abst_ticker_counts.most_common(5)
    ]

    # Trend: split captured rows by sim_date into older/recent halves.
    # Ordering on the raw sim_date STRING is correct for ISO-8601 dates
    # (YYYY-MM-DD lex-sorts == chrono-sorts; mirrors the
    # ``split_outcomes_temporal`` precedent of sim_date-based ordering).
    trend = "UNKNOWN"
    trend_diff: float | None = None
    older_rate: float | None = None
    recent_rate: float | None = None
    if n_captured >= 2 * MIN_PER_SIDE:
        ordered = sorted(captured, key=lambda x: x[0])
        split = int(len(ordered) * (1.0 - RECENT_FRACTION))
        older = ordered[:split]
        recent = ordered[split:]
        if len(older) >= MIN_PER_SIDE and len(recent) >= MIN_PER_SIDE:
            older_rate = round(
                sum(1 for _, _, a in older if a) / len(older), 4)
            recent_rate = round(
                sum(1 for _, _, a in recent if a) / len(recent), 4)
            diff = recent_rate - older_rate
            trend_diff = round(diff, 4)
            if diff > TREND_TOL_PP:
                trend = "DEGRADING"
            elif diff < -TREND_TOL_PP:
                trend = "IMPROVING"
            else:
                trend = "STABLE"

    base: dict = {
        "status": "ok",
        "verdict": "INSUFFICIENT_DATA",
        "n_captured": n_captured,
        "n_acted": n_acted,
        "n_abstained": n_abstained,
        "rate": round(rate, 4),
        "rate_older": older_rate,
        "rate_recent": recent_rate,
        "trend": trend,
        "trend_rate_diff_pp": trend_diff,
        "arm_dist": arm_dist,
        "top_tickers": top_tickers,
        "hint": "",
    }

    if n_captured < MIN_TOTAL:
        base["hint"] = (
            f"only {n_captured} captured rows — need ≥{MIN_TOTAL} before any "
            f"rate verdict; populates as the loop accumulates `gate_scorer_pred`"
        )
        return base

    if rate < INACTIVE_MAX:
        base["verdict"] = "GUARD_INACTIVE"
        base["hint"] = (
            f"abstention rate {rate * 100:.2f}% < {INACTIVE_MAX * 100:.1f}% — "
            f"the off-distribution guard (commit 84d8234) effectively never "
            f"fires; the ±PRED_CLAMP_PCT extrapolation protection it was "
            f"added for is not engaging"
        )
    elif rate >= RAMPANT_MIN:
        base["verdict"] = "GUARD_RAMPANT"
        base["hint"] = (
            f"abstention rate {rate * 100:.2f}% ≥ {RAMPANT_MIN * 100:.0f}% — "
            f"the live gate is leaving conviction untouched on a large "
            f"fraction of decisions; the 1.30/0.60 multiplier spread is "
            f"applied to a shrinking minority"
        )
    else:
        base["verdict"] = "GUARD_HEALTHY"
        base["hint"] = (
            f"abstention rate {rate * 100:.2f}% in [{INACTIVE_MAX * 100:.1f}%, "
            f"{RAMPANT_MIN * 100:.0f}%) — guard fires occasionally on real "
            f"extrapolation; check arm_dist to confirm abstentions cluster at "
            f"the extreme clamp (the expected pattern)"
        )
    return base


def analyze(outcomes_path: Path | str, oos_only: bool = True) -> dict:
    """Load the outcomes file, take the temporal-OOS slice (default) and run
    the abstention report. **No scorer / pickle is loaded** — this reads
    only the captured ``gate_scorer_pred`` / ``gate_off_dist`` fields.
    Read-only; never raises."""
    out: dict = {
        "status": "error", "verdict": "INSUFFICIENT_DATA",
        "n_captured": 0, "n_acted": 0, "n_abstained": 0,
        "rate": 0.0, "arm_dist": [], "top_tickers": [],
        "slice": "all", "hint": "",
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

        rep = abstention_report(recs)
        rep["slice"] = slice_name
        rep["n_records_total"] = len(records)
        return rep
    except Exception as e:  # pragma: no cover - defensive
        out["hint"] = f"audit failed: {type(e).__name__}: {e}"
        return out


def _cli(argv: list[str] | None = None) -> int:
    """``python3 -m paper_trader.ml.gate_abstention [--all]`` — read-only
    rate of the gate's off-distribution abstention. Exits 2 on
    ``GUARD_RAMPANT`` (operator-actionable: the gate is mostly neutral),
    0 on every other verdict (mirrors the sibling diagnostics' cron-branch
    convention)."""
    import sys
    argv = sys.argv[1:] if argv is None else argv
    oos_only = "--all" not in argv
    root = Path(__file__).resolve().parent.parent.parent
    rep = analyze(root / "data" / "decision_outcomes.jsonl", oos_only=oos_only)
    print(f"VERDICT: {rep['verdict']}  ({rep.get('hint', '')})")
    print(f"  slice={rep.get('slice')}  "
          f"n_captured={rep.get('n_captured')}  "
          f"n_acted={rep.get('n_acted')}  "
          f"n_abstained={rep.get('n_abstained')}  "
          f"rate={rep.get('rate', 0) * 100:.2f}%")
    if rep.get("rate_older") is not None:
        print(f"  trend={rep.get('trend')}  "
              f"older_rate={rep.get('rate_older', 0) * 100:.2f}%  "
              f"recent_rate={rep.get('rate_recent', 0) * 100:.2f}%  "
              f"diff_pp={rep.get('trend_rate_diff_pp', 0) * 100:+.2f}")
    arm_dist = rep.get("arm_dist") or []
    if arm_dist and rep.get("n_abstained", 0) > 0:
        print("  abstention by would-have-been arm:")
        for ad in arm_dist:
            print(f"    {ad['arm']:<16} n_abstained={ad['n_abstained']}")
    tops = rep.get("top_tickers") or []
    if tops:
        print("  top abstaining tickers: " +
              ", ".join(f"{t['ticker']}({t['n_abstained']})" for t in tops))
    return 2 if rep.get("verdict") == "GUARD_RAMPANT" else 0


if __name__ == "__main__":
    raise SystemExit(_cli())

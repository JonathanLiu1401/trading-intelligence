"""Gate kill-switch health and trend audit.

Read-only research signal (2026-05-30 feature, Agent 2 ML+backtests).
Sibling to ``scorer_offdist_rate`` / ``gate_realized`` — mirrors their
operational discipline exactly (never trains, never writes the scorer
pickle / outcomes JSONL / any trade path; safe to run against the
live unattended loop; never raises in analyzer functions).

**The gap this closes.** ``backtest._should_gate_modulate_conviction``
short-circuits the conviction gate when the trailing
``_GATE_SKILL_MIN_CYCLES`` (=20) median ``oos_buy_ic`` is below
``_GATE_SKILL_IC_TOLERANCE`` (+0.03). The kill-switch state is
recomputed every cycle and persisted to ``scorer_skill_log.jsonl`` as
``gate_effectively_active`` plus the underlying ``oos_buy_ic`` /
``gate_killswitch_reason`` fields — yet no diagnostic asks the
quant-researcher-level question:

  * *Is the gate ever going to come back on, and if so when?*
  * Is the trailing median **flat at noise**, **deteriorating
    (anti-skill)**, or **recovering (trending up)**?
  * How many cycles since the gate last actually fired?

The textbook answer is a slope-aware reading of the persisted IC
time-series. This analyzer surfaces it as a single verdict.

**Metrics reported**:

  * ``n``                  — number of skill-log rows with a parseable
                              ``oos_buy_ic`` (the analyzer's input
                              after filtering corrupt / blank rows).
  * ``trailing_median_20`` — median ``oos_buy_ic`` of the last 20
                              cycles. THIS is the value the live
                              ``_should_gate_modulate_conviction``
                              compares against ``+gate_threshold``.
  * ``trailing_median_50`` — median of the last 50 cycles (smoother
                              read — distinguishes a brief positive
                              run from a sustained one).
  * ``slope_20``           — least-squares slope (IC per cycle) over
                              the last 20 cycles; positive => rising,
                              negative => falling.
  * ``cycles_to_threshold`` — at the current ``slope_20``, the number
                              of additional cycles before the
                              trailing-20 median crosses
                              ``gate_threshold``. None when slope is
                              non-positive or the median is already
                              above threshold.
  * ``n_active``           — count of rows where
                              ``gate_effectively_active=True``.
  * ``n_active_in_last_20`` — count over the trailing-20 window — the
                              honest "is the gate firing recently"
                              read (an active gate 200 cycles ago
                              doesn't help today).
  * ``cycles_since_active`` — index distance from the LAST
                              ``gate_effectively_active=True`` row to
                              the end of the log. None if the gate
                              has NEVER fired in the recorded
                              history (the deployed state today).
  * ``ic_min`` / ``ic_max`` / ``ic_mean`` / ``ic_p5`` / ``ic_p95`` —
                              empirical distribution of ``oos_buy_ic``
                              across the WHOLE filtered series.

**Verdict ladder** (driven by ``trailing_median_20`` vs
``gate_threshold`` and the sign of ``slope_20``):

| Verdict | Meaning |
|---------|---------|
| ``INSUFFICIENT_DATA`` | < ``MIN_CYCLES_FOR_TREND`` cycles with a parseable ``oos_buy_ic``, or the skill log is missing. |
| ``GATE_ACTIVE_STABLE`` | ``trailing_median_20 >= gate_threshold`` AND slope is not significantly negative — the gate is firing and the trend is holding or improving. |
| ``GATE_ACTIVE_DETERIORATING`` | ``trailing_median_20 >= gate_threshold`` BUT slope is significantly negative — the gate is firing but the median is trending toward the kill threshold; budget for a return to dark soon. |
| ``GATE_DARK_RECOVERING`` | ``trailing_median_20 < gate_threshold`` AND slope is significantly positive — gate is killed but the median is climbing; the projection in ``cycles_to_threshold`` estimates time-to-active. |
| ``GATE_DARK_DETERIORATING`` | ``trailing_median_20 < 0`` AND slope is significantly negative — strictly anti-predictive AND getting worse; the kill-switch is doing real work (gate's per-arm sizing is inverted under negative IC; see ``_should_gate_modulate_conviction`` rationale). |
| ``GATE_DARK_FLAT`` | ``trailing_median_20 < gate_threshold`` AND ``|slope|`` is at noise — stuck near zero with no directional trend; no recovery in sight. |

Independent of the verdict, a separate ``has_never_been_active`` flag
fires when the entire log shows zero ``gate_effectively_active=True``
rows — operationally distinct from "gate was active and is now off",
because it means the deployed pickle has never demonstrated the
positive rank skill the gate was designed to exploit.

**Pure / module-level constants for testability**. CLI:

```bash
cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.gate_health_trend
cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.gate_health_trend --json
cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.gate_health_trend --log path/to/skill_log.jsonl
```

Exit code is 0 on ``GATE_ACTIVE_*`` verdicts and 1 on ``GATE_DARK_*`` /
``INSUFFICIENT_DATA`` so shell callers can gate ``$?`` like
``host_guard``'s CLI.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# Module-level constants (testable, single point of tuning).
MIN_CYCLES_FOR_TREND = 20      # min IC rows before any non-INSUFFICIENT verdict
TREND_WINDOW = 20              # cycles for trailing median + slope fit
LONG_WINDOW = 50               # cycles for the smoother trailing_median_50
# Must match backtest._GATE_SKILL_IC_TOLERANCE — the live kill-switch
# threshold. Kept as a module-level const so a future tuning of the gate
# tolerance is centralized and visible to this analyzer's tests.
GATE_THRESHOLD = 0.03
# IC-per-cycle slope magnitude below this is treated as "at noise" — neither
# rising nor falling. Empirically the cycle-to-cycle IC swing is ~0.01-0.05
# (see scorer_skill_log distribution), so a slope of ±0.001/cycle is
# meaningfully above measurement noise without being trigger-happy.
SLOPE_SIGNIFICANCE = 0.001


SCORER_SKILL_LOG = (Path(__file__).resolve().parent.parent.parent
                    / "data" / "scorer_skill_log.jsonl")


def _maybe_float(v):
    """Coerce to finite float or None — mirrors sibling analyzers."""
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    return x


def gate_health_trend_report(rows: list[dict]) -> dict:
    """Compute kill-switch state and trend from skill-log rows.

    ``rows`` MUST be in chronological order (append order as written to
    ``scorer_skill_log.jsonl``). The analyzer treats the LAST entries as
    the most recent cycles.

    Returns a JSON-safe dict. Never raises (the analyzer must not break
    the unattended loop)."""
    try:
        it = list(rows or [])
    except Exception:
        it = []

    ics: list[float] = []
    gate_flags: list[bool | None] = []
    for r in it:
        if not isinstance(r, dict):
            continue
        v = _maybe_float(r.get("oos_buy_ic"))
        if v is None:
            continue
        ics.append(v)
        # gate_effectively_active is a recent field — None on legacy
        # rows. Tracked separately from ICs so a partial log (recent
        # rows have the flag, older don't) still gives a usable trend
        # while honestly reporting "we don't know" for cycles_since.
        gflag = r.get("gate_effectively_active")
        if isinstance(gflag, bool):
            gate_flags.append(gflag)
        else:
            gate_flags.append(None)

    n = len(ics)
    if n < MIN_CYCLES_FOR_TREND:
        return {
            "verdict": "INSUFFICIENT_DATA",
            "n": n,
            "trailing_median_5": None,
            "trailing_median_20": None,
            "trailing_median_50": None,
            "slope_20": None,
            "cycles_to_threshold": None,
            "n_active": None,
            "n_active_in_last_20": None,
            "cycles_since_active": None,
            "has_never_been_active": None,
            "ic_min": None, "ic_max": None, "ic_mean": None,
            "ic_p5": None, "ic_p95": None,
            "min_cycles_for_trend": MIN_CYCLES_FOR_TREND,
            "gate_threshold": GATE_THRESHOLD,
            "slope_significance": SLOPE_SIGNIFICANCE,
            "hint": (f"only {n} rows with a parseable oos_buy_ic; need "
                     f">={MIN_CYCLES_FOR_TREND} for a trend reading."),
        }

    last5 = ics[-min(5, n):]
    last20 = ics[-TREND_WINDOW:]
    last50 = ics[-LONG_WINDOW:] if n >= LONG_WINDOW else None

    trailing_median_5 = round(float(np.median(last5)), 4)
    trailing_median_20 = round(float(np.median(last20)), 4)
    trailing_median_50 = (round(float(np.median(last50)), 4)
                         if last50 is not None else None)

    # least-squares slope (IC per cycle) over the last 20. polyfit returns
    # (slope, intercept) for deg=1. The x-axis is a positional index, NOT a
    # timestamp — cycles are evenly spaced by design (one row per backtest
    # cycle).
    xs = np.arange(len(last20), dtype=np.float64)
    ys = np.asarray(last20, dtype=np.float64)
    try:
        slope, _intercept = np.polyfit(xs, ys, 1)
        slope = float(slope)
    except Exception:
        slope = 0.0
    slope_20 = round(slope, 6)

    # cycles-to-threshold projection: only meaningful when the gate is
    # currently dark AND the slope is significantly positive. Linear
    # extrapolation is a rough estimate (real IC dynamics are nonlinear)
    # so we cap at a sentinel max — a slope of +0.0001/cycle would
    # otherwise project decades.
    cycles_to_threshold: int | None = None
    if (trailing_median_20 < GATE_THRESHOLD
            and slope > SLOPE_SIGNIFICANCE):
        gap = GATE_THRESHOLD - trailing_median_20
        # gap > 0 and slope > 0 by construction here.
        est = int(np.ceil(gap / slope))
        # Cap at 9999 — anything past that is "not in any forecastable
        # horizon" and the integer cap is the honest "unbounded" signal
        # without choosing a misleading specific number.
        cycles_to_threshold = est if est < 9999 else 9999

    # Gate-active counts. Treat None as "unknown" — count only definite
    # True rows in n_active, but use the full positional log for
    # cycles_since_active (so a "True earlier than the legacy-None tail"
    # is still found).
    n_active = sum(1 for g in gate_flags if g is True)
    last20_flags = gate_flags[-TREND_WINDOW:]
    n_active_in_last_20 = sum(1 for g in last20_flags if g is True)
    cycles_since_active: int | None = None
    for i in range(len(gate_flags) - 1, -1, -1):
        if gate_flags[i] is True:
            cycles_since_active = len(gate_flags) - 1 - i
            break
    # has_never_been_active applies when the log is non-trivial AND
    # contains zero True flags in EVERY row (None and False both count
    # as "not active"). The flag's purpose is to surface the case where
    # the deployed scorer's gate has never demonstrated realized
    # positive skill on the persisted record, distinct from "was active
    # and is now dark".
    has_never_been_active = n_active == 0

    ic_arr = np.asarray(ics, dtype=np.float64)
    ic_min = round(float(ic_arr.min()), 4)
    ic_max = round(float(ic_arr.max()), 4)
    ic_mean = round(float(ic_arr.mean()), 4)
    ic_p5 = round(float(np.percentile(ic_arr, 5)), 4)
    ic_p95 = round(float(np.percentile(ic_arr, 95)), 4)

    # Verdict ladder. The four GATE_DARK / GATE_ACTIVE combinations
    # are derived from two orthogonal axes: median-above-threshold and
    # slope-sign. ``GATE_DARK_DETERIORATING`` collapses two stricter
    # conditions (median < 0 AND slope < -significance) into one verdict
    # because that's the genuinely-dangerous state the kill-switch was
    # designed to detect (strictly anti-predictive AND getting worse).
    if trailing_median_20 >= GATE_THRESHOLD:
        if slope_20 < -SLOPE_SIGNIFICANCE:
            verdict = "GATE_ACTIVE_DETERIORATING"
            hint = (f"trailing-20 median oos_buy_ic={trailing_median_20:+.3f} "
                    f"is above gate_threshold (+{GATE_THRESHOLD}) but slope "
                    f"is significantly negative ({slope_20:+.4f}/cycle). "
                    "Budget for the gate to return to dark within "
                    f"~{int(abs((trailing_median_20 - GATE_THRESHOLD) / slope_20))} "
                    "cycles at this slope.")
        else:
            verdict = "GATE_ACTIVE_STABLE"
            hint = (f"trailing-20 median oos_buy_ic={trailing_median_20:+.3f} "
                    f"is above gate_threshold (+{GATE_THRESHOLD}); slope "
                    f"{slope_20:+.4f}/cycle is at noise or positive. The "
                    "gate is firing.")
    else:
        if slope_20 > SLOPE_SIGNIFICANCE:
            verdict = "GATE_DARK_RECOVERING"
            ctf_str = (f"~{cycles_to_threshold} cycles"
                       if cycles_to_threshold is not None else "unbounded")
            hint = (f"trailing-20 median oos_buy_ic={trailing_median_20:+.3f} "
                    f"is below gate_threshold (+{GATE_THRESHOLD}) but slope "
                    f"is significantly positive ({slope_20:+.4f}/cycle). At "
                    f"this trajectory the gate will reactivate in "
                    f"{ctf_str}.")
        elif (trailing_median_20 < 0.0
              and slope_20 < -SLOPE_SIGNIFICANCE):
            verdict = "GATE_DARK_DETERIORATING"
            hint = (f"trailing-20 median oos_buy_ic={trailing_median_20:+.3f} "
                    f"is BELOW zero (anti-predictive) AND slope is "
                    f"significantly negative ({slope_20:+.4f}/cycle). The "
                    "kill-switch is doing real work — the gate's per-arm "
                    "sizing assumes positive rank-IC; an active gate "
                    "under negative IC would invert the modulation "
                    "direction relative to realized returns.")
        else:
            verdict = "GATE_DARK_FLAT"
            hint = (f"trailing-20 median oos_buy_ic={trailing_median_20:+.3f} "
                    f"is below gate_threshold (+{GATE_THRESHOLD}) and slope "
                    f"({slope_20:+.4f}/cycle) is at noise. The gate has "
                    "been dark and is not trending toward recovery.")

    return {
        "verdict": verdict,
        "n": n,
        "trailing_median_5": trailing_median_5,
        "trailing_median_20": trailing_median_20,
        "trailing_median_50": trailing_median_50,
        "slope_20": slope_20,
        "cycles_to_threshold": cycles_to_threshold,
        "n_active": n_active,
        "n_active_in_last_20": n_active_in_last_20,
        "cycles_since_active": cycles_since_active,
        "has_never_been_active": has_never_been_active,
        "ic_min": ic_min,
        "ic_max": ic_max,
        "ic_mean": ic_mean,
        "ic_p5": ic_p5,
        "ic_p95": ic_p95,
        "min_cycles_for_trend": MIN_CYCLES_FOR_TREND,
        "gate_threshold": GATE_THRESHOLD,
        "slope_significance": SLOPE_SIGNIFICANCE,
        "hint": hint,
    }


def analyze(log_path: Path | str | None = None) -> dict:
    """Read skill-log JSONL and run the gate-health-trend report.

    Returns ``{status, ...report fields}``. ``status='error'`` on read
    failure; ``status='ok'`` otherwise. Never raises."""
    if log_path is None:
        log_path = SCORER_SKILL_LOG
    p = Path(log_path)
    if not p.exists():
        report = gate_health_trend_report([])
        report["status"] = "error"
        report["error"] = f"missing {p}"
        return report
    rows: list[dict] = []
    try:
        with p.open("r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception as exc:
        report = gate_health_trend_report([])
        report["status"] = "error"
        report["error"] = f"read failed: {exc}"
        return report

    report = gate_health_trend_report(rows)
    report["status"] = "ok"
    return report


def _cli(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.gate_health_trend",
        description="Gate kill-switch health & trend audit on the "
                    "persisted scorer_skill_log.jsonl.",
    )
    parser.add_argument("--log", type=Path, default=None,
                        help=("Path to the skill log JSONL (default: "
                              "data/scorer_skill_log.jsonl relative to repo)."))
    parser.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)

    report = analyze(args.log)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        verdict = report.get("verdict") or ""
        return 0 if verdict.startswith("GATE_ACTIVE") else 1

    verdict = report.get("verdict")
    print(f"[gate_health_trend] verdict={verdict}  n={report.get('n')}")
    if verdict == "INSUFFICIENT_DATA":
        print(f"  hint: {report.get('hint')}")
        return 1
    tm5 = report.get("trailing_median_5")
    tm20 = report.get("trailing_median_20")
    tm50 = report.get("trailing_median_50")
    print(f"  trailing median: 5={tm5:+.3f}  20={tm20:+.3f}  "
          + (f"50={tm50:+.3f}" if tm50 is not None else "50=n/a"))
    slope = report.get("slope_20")
    if slope is not None:
        print(f"  slope_20: {slope:+.4f}/cycle  "
              f"(threshold ±{report.get('slope_significance')})")
    ctf = report.get("cycles_to_threshold")
    if ctf is not None:
        print(f"  cycles_to_threshold: {ctf}  (linear projection at current slope)")
    n_act = report.get("n_active")
    n_act20 = report.get("n_active_in_last_20")
    csa = report.get("cycles_since_active")
    print(f"  gate fires: total={n_act}  last_20={n_act20}  "
          f"cycles_since_active="
          + (str(csa) if csa is not None else "never"))
    if report.get("has_never_been_active"):
        print("  ⚠ has_never_been_active=True — the deployed pickle has "
              "NEVER fired the gate in the recorded skill log.")
    if report.get("ic_min") is not None:
        print(f"  ic dist: min={report['ic_min']:+.3f} "
              f"p5={report['ic_p5']:+.3f} mean={report['ic_mean']:+.3f} "
              f"p95={report['ic_p95']:+.3f} max={report['ic_max']:+.3f}")
    print(f"  hint: {report.get('hint')}")
    # Shell-gateable exit: GATE_ACTIVE_* → 0, anything else → 1.
    return 0 if (verdict or "").startswith("GATE_ACTIVE") else 1


if __name__ == "__main__":
    raise SystemExit(_cli())

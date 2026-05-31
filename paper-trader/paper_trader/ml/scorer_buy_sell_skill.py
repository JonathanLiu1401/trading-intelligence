"""Scorer BUY-vs-SELL rank-IC asymmetry audit.

Read-only research signal (2026-05-30 feature, Agent 2 ML+backtests).
Sibling to ``gate_health_trend`` — same canonical pattern (reads
``scorer_skill_log.jsonl``, emits a verdict ladder, never trains /
writes pickle / touches trade path; safe to run against the live
unattended loop; ``analyze`` never raises).

**The gap this closes.** The kill-switch in
``backtest._should_gate_modulate_conviction`` reads ``oos_buy_ic``
ALONE — the gate is BUY-only by construction, so a positive trailing
median there enables the conviction modulation. But ``scorer_skill_log``
captures **both** ``oos_buy_ic`` and ``oos_sell_ic`` per cycle, and the
AGENTS.md 2026-05-30 pass-#5 finding observed an asymmetry: trailing-20
``oos_buy_ic`` median ≈ -0.010 while ``oos_sell_ic`` median ≈ +0.010.
That state is operationally meaningful: the deployed model has slightly
NEGATIVE BUY skill but slightly POSITIVE SELL skill — yet the gate
sits on the WRONG side of that asymmetry and is structurally locked
away from the side carrying skill.

``gate_health_trend`` tracks the gate's own state (BUY-side) but cannot
diagnose this asymmetry because it never reads ``oos_sell_ic``. Every
other existing diagnostic (``gate_pnl``, ``gate_audit``, ``gate_realized``,
``baseline_compare``) is gate-action-effect or model-vs-trivial — none
cross-checks per-direction OOS rank skill against where the gate
modulates.

**Metrics reported**:

  * ``n``                  — number of skill-log rows with BOTH
                              ``oos_buy_ic`` AND ``oos_sell_ic``
                              parseable.
  * ``buy_trailing_median_20`` — trailing-20 median of BUY rank-IC.
  * ``sell_trailing_median_20`` — trailing-20 median of SELL rank-IC.
  * ``buy_trailing_median_50`` / ``sell_trailing_median_50`` — smoother
    50-cycle reads (None when fewer than 50 rows).
  * ``asymmetry_20`` — ``sell_trailing_median_20 -
    buy_trailing_median_20``. Positive ⇒ SELL side has more skill;
    negative ⇒ BUY side has more skill.
  * ``buy_p5`` / ``buy_p95`` / ``sell_p5`` / ``sell_p95`` —
    empirical 5%/95% percentiles across the WHOLE filtered series.

**Verdict ladder** (driven by trailing-20 medians and the asymmetry):

| Verdict | Trigger |
|---------|---------|
| ``INSUFFICIENT_DATA`` | < ``MIN_CYCLES_FOR_TREND`` rows with BOTH ICs parseable, OR the skill log is missing. |
| ``BUY_SKILLED_ONLY`` | ``buy_median >= +SKILL_TOL`` AND ``sell_median < +SKILL_TOL`` — gate is aligned with the skill direction. |
| ``SELL_SKILLED_ONLY`` | ``sell_median >= +SKILL_TOL`` AND ``buy_median < +SKILL_TOL`` — operationally notable: the gate (BUY-only) sees no skill but the SELL side does. |
| ``BOTH_SKILLED`` | both medians ``>= +SKILL_TOL`` — model has rank skill on either direction. |
| ``NEITHER_SKILLED`` | both medians ``< +SKILL_TOL`` — at noise on both sides; kill-switch is doing real work. |

Independent of the skill verdict, a separate ``gate_misaligned`` flag
fires when ``asymmetry_20 > MISALIGN_TOL`` — the SELL side has
meaningfully MORE skill than the BUY side, regardless of whether
either reaches the absolute skill threshold. This is the actionable
signal: the gate, which is BUY-only, is structurally on the WRONG
side of the model's rank skill. Pairs ``SELL_SKILLED_ONLY`` /
``BOTH_SKILLED`` cleanly: ``SELL_SKILLED_ONLY`` AND ``gate_misaligned``
is the maximum-confidence "gate is on the wrong side" state.

Pure / module-level constants for testability. CLI:

```bash
cd /home/zeph/trading-intelligence/paper-trader && \
    python3 -m paper_trader.ml.scorer_buy_sell_skill
cd /home/zeph/trading-intelligence/paper-trader && \
    python3 -m paper_trader.ml.scorer_buy_sell_skill --json
cd /home/zeph/trading-intelligence/paper-trader && \
    python3 -m paper_trader.ml.scorer_buy_sell_skill --log path/to/skill_log.jsonl
```

Exit code mirrors ``gate_health_trend``: 0 on benign verdicts
(``BUY_SKILLED_ONLY`` / ``BOTH_SKILLED`` / ``INSUFFICIENT_DATA``),
2 on actionable verdicts (``SELL_SKILLED_ONLY`` /
``NEITHER_SKILLED`` with ``gate_misaligned=True``).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# Module-level constants (testable, single point of tuning).
MIN_CYCLES_FOR_TREND = 20  # need at least 20 rows with BOTH ICs
TREND_WINDOW = 20
LONG_WINDOW = 50
# Must match backtest._GATE_SKILL_IC_TOLERANCE for cross-tool consistency —
# the live kill-switch fires below +0.03 BUY-IC, so the "skill threshold"
# the analyzer compares against is the same value. Centralizing here so a
# future tuning of the gate is reflected in one place.
SKILL_TOL = 0.03
# Asymmetry tolerance: SELL median - BUY median above this is a meaningful
# directional asymmetry. Set deliberately above noise (cycle-to-cycle IC
# swings are ~0.01-0.05). 0.02 is 2× the noise floor and below the
# SKILL_TOL itself — captures asymmetry even when neither side passes
# absolute skill.
MISALIGN_TOL = 0.02

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


def buy_sell_skill_report(rows: list[dict]) -> dict:
    """Compute BUY-vs-SELL OOS rank-IC asymmetry from skill-log rows.

    ``rows`` MUST be in chronological order (append order as written to
    ``scorer_skill_log.jsonl``). Only rows where BOTH ``oos_buy_ic`` AND
    ``oos_sell_ic`` are parseable are kept — pairing them per-cycle
    avoids the "BUY has 30 valid rows / SELL has 15" misalignment that
    would silently bias the comparison.

    Returns a JSON-safe dict. Never raises (the analyzer must not break
    the unattended loop)."""
    try:
        it = list(rows or [])
    except Exception:
        it = []

    buy_ics: list[float] = []
    sell_ics: list[float] = []
    for r in it:
        if not isinstance(r, dict):
            continue
        b = _maybe_float(r.get("oos_buy_ic"))
        s = _maybe_float(r.get("oos_sell_ic"))
        # PAIRED: skip rows where either side is missing so the trailing
        # medians and the asymmetry difference are computed on the SAME
        # underlying cycles. A "BUY has more rows than SELL" scenario
        # would silently bias the comparison toward whatever cycles only
        # one side recorded.
        if b is None or s is None:
            continue
        buy_ics.append(b)
        sell_ics.append(s)

    n = len(buy_ics)
    if n < MIN_CYCLES_FOR_TREND:
        return {
            "verdict": "INSUFFICIENT_DATA",
            "n": n,
            "buy_trailing_median_20": None,
            "sell_trailing_median_20": None,
            "buy_trailing_median_50": None,
            "sell_trailing_median_50": None,
            "asymmetry_20": None,
            "buy_p5": None, "buy_p95": None,
            "sell_p5": None, "sell_p95": None,
            "gate_misaligned": False,
            "min_cycles_for_trend": MIN_CYCLES_FOR_TREND,
            "skill_tol": SKILL_TOL,
            "misalign_tol": MISALIGN_TOL,
            "hint": (f"only {n} rows with BOTH oos_buy_ic AND oos_sell_ic "
                     f"parseable; need >={MIN_CYCLES_FOR_TREND}."),
        }

    last20_buy = buy_ics[-TREND_WINDOW:]
    last20_sell = sell_ics[-TREND_WINDOW:]
    last50_buy = buy_ics[-LONG_WINDOW:] if n >= LONG_WINDOW else None
    last50_sell = sell_ics[-LONG_WINDOW:] if n >= LONG_WINDOW else None

    buy_t20 = round(float(np.median(last20_buy)), 4)
    sell_t20 = round(float(np.median(last20_sell)), 4)
    buy_t50 = (round(float(np.median(last50_buy)), 4)
               if last50_buy is not None else None)
    sell_t50 = (round(float(np.median(last50_sell)), 4)
                if last50_sell is not None else None)

    asymmetry_20 = round(sell_t20 - buy_t20, 4)

    buy_arr = np.asarray(buy_ics, dtype=np.float64)
    sell_arr = np.asarray(sell_ics, dtype=np.float64)
    buy_p5 = round(float(np.percentile(buy_arr, 5)), 4)
    buy_p95 = round(float(np.percentile(buy_arr, 95)), 4)
    sell_p5 = round(float(np.percentile(sell_arr, 5)), 4)
    sell_p95 = round(float(np.percentile(sell_arr, 95)), 4)

    # Verdict ladder — orthogonal axes: buy >= SKILL_TOL, sell >= SKILL_TOL.
    buy_skilled = buy_t20 >= SKILL_TOL
    sell_skilled = sell_t20 >= SKILL_TOL
    if buy_skilled and sell_skilled:
        verdict = "BOTH_SKILLED"
        hint = (f"buy_t20={buy_t20:+.3f}  sell_t20={sell_t20:+.3f}  "
                f"both >= +{SKILL_TOL} — model has rank skill on either "
                "direction.")
    elif buy_skilled and not sell_skilled:
        verdict = "BUY_SKILLED_ONLY"
        hint = (f"buy_t20={buy_t20:+.3f}  sell_t20={sell_t20:+.3f}  "
                f"BUY-only skill (sell side below +{SKILL_TOL}) — gate "
                "is aligned with the skill direction.")
    elif sell_skilled and not buy_skilled:
        verdict = "SELL_SKILLED_ONLY"
        hint = (f"buy_t20={buy_t20:+.3f}  sell_t20={sell_t20:+.3f}  "
                f"SELL-only skill (buy side below +{SKILL_TOL}) — gate "
                "is BUY-only and structurally LOCKED AWAY from the side "
                "carrying skill.")
    else:
        verdict = "NEITHER_SKILLED"
        hint = (f"buy_t20={buy_t20:+.3f}  sell_t20={sell_t20:+.3f}  "
                f"both below +{SKILL_TOL} — model is at noise on both "
                "sides; kill-switch is doing real work.")

    # Misalignment is independent — it can fire on any verdict where
    # SELL > BUY by a meaningful margin (the skewed-asymmetry case).
    # Most operationally relevant when verdict == SELL_SKILLED_ONLY
    # (which always implies misalignment, by construction) but also
    # fires under NEITHER_SKILLED when the SELL side is BEATING the
    # BUY side at sub-threshold magnitudes — early warning that the
    # gate's structural BUY-only bias may be a growing liability.
    gate_misaligned = asymmetry_20 > MISALIGN_TOL

    return {
        "verdict": verdict,
        "n": n,
        "buy_trailing_median_20": buy_t20,
        "sell_trailing_median_20": sell_t20,
        "buy_trailing_median_50": buy_t50,
        "sell_trailing_median_50": sell_t50,
        "asymmetry_20": asymmetry_20,
        "buy_p5": buy_p5, "buy_p95": buy_p95,
        "sell_p5": sell_p5, "sell_p95": sell_p95,
        "gate_misaligned": gate_misaligned,
        "min_cycles_for_trend": MIN_CYCLES_FOR_TREND,
        "skill_tol": SKILL_TOL,
        "misalign_tol": MISALIGN_TOL,
        "hint": hint,
    }


def analyze(log_path: "Path | str | None" = None) -> dict:
    """Read skill-log JSONL and run the BUY-vs-SELL asymmetry report.

    Returns ``{status, ...report fields}``. ``status='error'`` on read
    failure; ``status='ok'`` otherwise. Never raises."""
    if log_path is None:
        log_path = SCORER_SKILL_LOG
    p = Path(log_path)
    if not p.exists():
        report = buy_sell_skill_report([])
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
        report = buy_sell_skill_report([])
        report["status"] = "error"
        report["error"] = f"read failed: {exc}"
        return report

    report = buy_sell_skill_report(rows)
    report["status"] = "ok"
    return report


def _cli(argv: "list[str] | None" = None) -> int:
    """CLI entrypoint. Exit codes mirror sibling diagnostics."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.scorer_buy_sell_skill",
        description="BUY-vs-SELL OOS rank-IC asymmetry audit on the "
                    "persisted scorer_skill_log.jsonl.",
    )
    parser.add_argument("--log", type=Path, default=None,
                        help=("Path to the skill log JSONL (default: "
                              "data/scorer_skill_log.jsonl relative to "
                              "repo)."))
    parser.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)

    report = analyze(args.log)
    verdict = report.get("verdict") or ""

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        # Actionable verdicts → exit 2; benign → 0.
        if verdict == "SELL_SKILLED_ONLY":
            return 2
        if verdict == "NEITHER_SKILLED" and report.get("gate_misaligned"):
            return 2
        return 0

    print(f"[scorer_buy_sell_skill] verdict={verdict}  n={report.get('n')}")
    if verdict == "INSUFFICIENT_DATA":
        print(f"  hint: {report.get('hint')}")
        return 0
    b20 = report.get("buy_trailing_median_20")
    s20 = report.get("sell_trailing_median_20")
    b50 = report.get("buy_trailing_median_50")
    s50 = report.get("sell_trailing_median_50")
    print(f"  buy  trailing median: 20={b20:+.3f}  "
          + (f"50={b50:+.3f}" if b50 is not None else "50=n/a"))
    print(f"  sell trailing median: 20={s20:+.3f}  "
          + (f"50={s50:+.3f}" if s50 is not None else "50=n/a"))
    asym = report.get("asymmetry_20")
    if asym is not None:
        sign = "SELL>BUY" if asym > 0 else ("BUY>SELL" if asym < 0 else "even")
        print(f"  asymmetry_20: {asym:+.4f}  ({sign})")
    if report.get("gate_misaligned"):
        print(f"  ⚠ gate_misaligned=True — asymmetry exceeds +"
              f"{report.get('misalign_tol')}; the BUY-only gate is "
              "structurally on the wrong side of the model's rank skill.")
    if report.get("buy_p5") is not None:
        print(f"  buy  dist: p5={report['buy_p5']:+.3f}  "
              f"p95={report['buy_p95']:+.3f}")
        print(f"  sell dist: p5={report['sell_p5']:+.3f}  "
              f"p95={report['sell_p95']:+.3f}")
    print(f"  hint: {report.get('hint')}")
    if verdict == "SELL_SKILLED_ONLY":
        return 2
    if verdict == "NEITHER_SKILLED" and report.get("gate_misaligned"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())

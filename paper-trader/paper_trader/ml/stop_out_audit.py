"""Stop-out audit — would the documented -8% stop_loss have helped or hurt?

This is a **read-only diagnostic**. It never trains, never touches
``decision_scorer.pkl`` or ``decision_outcomes.jsonl`` (only reads it),
and never enters a trade path — the same operational discipline as
``paper_trader/ml/baseline_compare.py`` / ``calibration.py`` /
``gate_audit.py``.

**Why this is not any existing tool.** The ML/backtest domain has many
analyzers that ask "does the SCORER carry rank skill" (calibration,
baseline_compare, regime_audit, …) and several that ask "does the GATE
weight sizing right" (conviction_calibration, gate_pnl, …) — but **none
quantify the realized economic effect of the documented -8% stop_loss
band** that ``backtest._buy`` writes onto every BUY position
(``stop_loss=round(price * 0.92, 2)``). The intraperiod-min/max fields
``_compute_decision_outcomes`` started persisting on 2026-05-23
(``forward_intraperiod_min_5d`` / ``forward_intraperiod_max_5d``) are
captured-but-unused data; this analyzer is the FIRST consumer.

**Method.** For each BUY outcome row with a finite
``forward_intraperiod_min_5d``:

  * If ``intra_min <= -STOP_PCT`` (the stop would have triggered intra-
    window), the realized return *with stop* is ``-STOP_PCT`` (the stop
    fired at or near the band; the actual fill price is unknowable from
    closes alone, so use the band itself — conservative because a real
    fill on a gap-down would be WORSE than the band).
  * Else: realized return *with stop* equals ``forward_return_5d`` (the
    stop never fired, so the endpoint reading is what would have been
    captured).

The aggregate ``stop_benefit_pct`` is the mean realized return *with*
the stop minus the mean realized return *without* — positive when the
stop saves more from limited-loss trades than it costs in
prematurely-exited recoveries. This answers a question a skeptical
quant asks once: is the inherited 0.92 floor a real defensive arm, or
is it variance-only chop the gate would do better without?

| Verdict | Meaning |
|---------|---------|
| ``INSUFFICIENT_DATA`` | < ``MIN_BUYS`` BUY rows carry a finite ``forward_intraperiod_min_5d``; the gate behaviour is not measurable. The historical 8.7k-row corpus pre-dates the intraperiod feature, so cycles that retrained AFTER the feature shipped are what populate this row. |
| ``STOP_HELPS`` | ``stop_benefit_pct`` >= ``BENEFIT_MARGIN`` (e.g. +0.30pp realized) — the stop measurably improves aggregate realized return. |
| ``STOP_HURTS`` | ``stop_benefit_pct`` <= -``BENEFIT_MARGIN`` — the stop cuts winners more than it saves losers, dragging realized return down. |
| ``STOP_NEUTRAL`` | benefit within ``±BENEFIT_MARGIN`` — the stop is in the noise; sizing variance with no measurable economic effect. |

```bash
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.stop_out_audit
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.stop_out_audit --json
```
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable


# Stop-loss percentage — the actual band ``backtest._buy`` writes is
# ``price * 0.92`` (an 8% drop from entry triggers). Module-level so a
# tuning change is a single reviewable edit.
STOP_PCT = 8.0

# Minimum BUYs with finite intraperiod data before any verdict. Below this
# the report is honestly ``INSUFFICIENT_DATA``.
MIN_BUYS = 30

# Realized-return margin (percentage points) the stop must clear / lose by
# before a non-neutral verdict is declared. 0.30pp is roughly one
# standard error on a 1000-trade aggregate at σ(target) ≈ 12pp:
# se(mean) = σ/√n = 12/√1000 ≈ 0.38pp. Anything tighter is sampling noise.
BENEFIT_MARGIN = 0.30


def _to_finite_float(v) -> float | None:
    """Return ``float(v)`` if finite, else None. Mirrors decision_scorer
    ``_to_float`` semantics but returns None on missing/invalid so the
    caller can DROP a row rather than coerce to a default value (which would
    fabricate a "no stop trigger" reading on rows with missing data)."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _stop_protected_return(forward_return_5d: float,
                            intra_min: float,
                            stop_pct: float = STOP_PCT) -> float:
    """Realized return for ONE BUY with the documented -stop_pct% stop.

    ``intra_min`` is the worst intraperiod return reached (signed %).
    If it drops at or below ``-stop_pct``, the stop fires — model the
    fill at exactly ``-stop_pct`` (a real fill on a gap-down would be
    worse; this is the optimistic-stop assumption a quant would treat
    as an UPPER bound on the stop's benefit).

    If ``intra_min > -stop_pct`` (no trigger), the position rides to the
    5d endpoint and ``forward_return_5d`` is captured.

    Pure, total, never raises."""
    if intra_min <= -stop_pct:
        return -stop_pct
    return forward_return_5d


def _iter_rows(path: Path) -> Iterable[dict]:
    """Stream one JSON record per line, silently dropping unparseable rows.

    Mirrors the line-tolerant loader used throughout the codebase
    (``_compute_decision_outcomes`` / ``_inject_and_train`` / etc.) —
    a single corrupt line must not abort an audit run.
    """
    with path.open("r") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                yield json.loads(ln)
            except json.JSONDecodeError:
                continue


def analyze(outcomes_path: "Path | str | None" = None,
            stop_pct: float = STOP_PCT,
            benefit_margin: float = BENEFIT_MARGIN,
            min_buys: int = MIN_BUYS) -> dict:
    """Compute the stop-out audit report.

    Reads ``data/decision_outcomes.jsonl`` (or the passed path),
    filters to BUYs with a finite ``forward_intraperiod_min_5d`` AND a
    finite ``forward_return_5d``, then computes the with-stop vs
    without-stop aggregate realized return.

    Returns a JSON-safe dict — never raises. On any fault returns an
    INSUFFICIENT_DATA envelope so a ledger consumer can persist the
    failure mode rather than swallow it.
    """
    if outcomes_path is None:
        outcomes_path = Path(__file__).resolve().parent.parent.parent / "data" / "decision_outcomes.jsonl"
    else:
        outcomes_path = Path(outcomes_path)

    empty = {
        "status": "insufficient_data",
        "verdict": "INSUFFICIENT_DATA",
        "stop_pct": stop_pct,
        "n_buys": 0,
        "n_with_intraperiod": 0,
        "n_stop_triggered": 0,
        "pct_stop_triggered": None,
        "mean_realized_return_pct": None,
        "mean_stop_protected_return_pct": None,
        "stop_benefit_pct": None,
        "median_realized_return_pct": None,
        "median_stop_protected_return_pct": None,
        "hint": None,
    }

    if not outcomes_path.exists():
        empty["hint"] = f"outcomes file not found: {outcomes_path}"
        return empty

    n_buys = 0
    n_with_intra = 0
    n_stop_triggered = 0
    realized: list[float] = []
    stop_protected: list[float] = []

    try:
        for row in _iter_rows(outcomes_path):
            if not isinstance(row, dict):
                continue
            action = str(row.get("action") or "").upper()
            if action != "BUY":
                continue
            n_buys += 1
            fwd = _to_finite_float(row.get("forward_return_5d"))
            intra = _to_finite_float(row.get("forward_intraperiod_min_5d"))
            if fwd is None or intra is None:
                continue
            n_with_intra += 1
            sp = _stop_protected_return(fwd, intra, stop_pct=stop_pct)
            if intra <= -stop_pct:
                n_stop_triggered += 1
            realized.append(fwd)
            stop_protected.append(sp)
    except Exception as exc:
        empty["hint"] = f"row scan failed: {type(exc).__name__}: {exc}"
        return empty

    if n_with_intra < min_buys:
        empty["n_buys"] = n_buys
        empty["n_with_intraperiod"] = n_with_intra
        empty["hint"] = (
            f"only {n_with_intra} BUYs with intraperiod data (< {min_buys}); "
            f"older outcome rows predate the 2026-05-23 forward_intraperiod_* "
            f"feature — re-run until the rolling 5000-record tail is dominated "
            f"by post-feature rows."
        )
        return empty

    mean_real = sum(realized) / len(realized)
    mean_stop = sum(stop_protected) / len(stop_protected)
    benefit = mean_stop - mean_real

    # Sorted lists for medians — small overhead on n ~ thousands.
    realized_sorted = sorted(realized)
    stop_sorted = sorted(stop_protected)

    def _median(s: list[float]) -> float:
        n = len(s)
        mid = n // 2
        if n % 2 == 1:
            return s[mid]
        return (s[mid - 1] + s[mid]) / 2.0

    median_real = _median(realized_sorted)
    median_stop = _median(stop_sorted)

    if benefit >= benefit_margin:
        verdict = "STOP_HELPS"
    elif benefit <= -benefit_margin:
        verdict = "STOP_HURTS"
    else:
        verdict = "STOP_NEUTRAL"

    return {
        "status": "ok",
        "verdict": verdict,
        "stop_pct": stop_pct,
        "benefit_margin_pp": benefit_margin,
        "n_buys": n_buys,
        "n_with_intraperiod": n_with_intra,
        "n_stop_triggered": n_stop_triggered,
        "pct_stop_triggered": round(n_stop_triggered / n_with_intra * 100.0, 2),
        "mean_realized_return_pct": round(mean_real, 4),
        "mean_stop_protected_return_pct": round(mean_stop, 4),
        "stop_benefit_pct": round(benefit, 4),
        "median_realized_return_pct": round(median_real, 4),
        "median_stop_protected_return_pct": round(median_stop, 4),
        "hint": None,
    }


# ---------------------------------------------------------------------------
# CLI: `python3 -m paper_trader.ml.stop_out_audit [--json] [--stop 5.0]`
#
# Pattern mirrors the existing read-only diagnostic CLIs
# (decision_scorer --explain, baseline_compare, conviction_calibration).
# Returns 0 only when a non-neutral verdict was produced — `$?` lets a
# shell caller gate on "the stop is decisively helping/hurting".
# ---------------------------------------------------------------------------


def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.stop_out_audit",
        description="Stop-out audit: would the documented -8%% stop_loss "
                    "have improved or harmed realized return? Read-only — "
                    "reads decision_outcomes.jsonl, never trains, never "
                    "modifies the pickle or the gate.",
    )
    p.add_argument("--outcomes", default=None,
                   help="Path to decision_outcomes.jsonl (default: the "
                        "repo data/ path).")
    p.add_argument("--stop", type=float, default=STOP_PCT, dest="stop_pct",
                   help=f"Stop-loss percentage (default {STOP_PCT}). The "
                        f"deployed band is _buy()'s 0.92 = 8.0%%.")
    p.add_argument("--margin", type=float, default=BENEFIT_MARGIN,
                   dest="benefit_margin",
                   help=f"Realized-return margin (pp) the stop must clear "
                        f"/ lose by for a non-NEUTRAL verdict "
                        f"(default {BENEFIT_MARGIN}pp).")
    p.add_argument("--min-buys", type=int, default=MIN_BUYS, dest="min_buys",
                   help=f"Minimum BUYs with intraperiod data before any "
                        f"verdict (default {MIN_BUYS}).")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of a table.")
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Returns 0 on a decisive non-neutral verdict, 1 on
    INSUFFICIENT_DATA or NEUTRAL — so shell callers can gate on `$?`."""
    import sys

    args = _build_arg_parser().parse_args(
        sys.argv[1:] if argv is None else argv)
    rep = analyze(
        outcomes_path=args.outcomes,
        stop_pct=args.stop_pct,
        benefit_margin=args.benefit_margin,
        min_buys=args.min_buys,
    )

    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
        return 0 if rep["verdict"] in {"STOP_HELPS", "STOP_HURTS"} else 1

    verdict = rep["verdict"]
    print(f"[stop_out_audit] verdict={verdict}  stop={rep['stop_pct']}%")
    print(f"  n_buys={rep['n_buys']}  n_with_intraperiod={rep['n_with_intraperiod']}")
    if rep["n_with_intraperiod"] > 0:
        print(f"  n_stop_triggered={rep['n_stop_triggered']} "
              f"({rep['pct_stop_triggered']}% of with-intraperiod)")
    if rep.get("mean_realized_return_pct") is not None:
        print(f"  realized:    mean={rep['mean_realized_return_pct']:+.3f}pp  "
              f"median={rep['median_realized_return_pct']:+.3f}pp")
        print(f"  with-stop:   mean={rep['mean_stop_protected_return_pct']:+.3f}pp  "
              f"median={rep['median_stop_protected_return_pct']:+.3f}pp")
        print(f"  benefit:     {rep['stop_benefit_pct']:+.3f}pp  "
              f"(margin: ±{rep['benefit_margin_pp']}pp)")
    if rep.get("hint"):
        print(f"  hint: {rep['hint']}")
    return 0 if verdict in {"STOP_HELPS", "STOP_HURTS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

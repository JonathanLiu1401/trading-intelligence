"""MFE-conversion audit — would the documented +15% take_profit have helped or hurt?

This is a **read-only diagnostic**. It never trains, never touches
``decision_scorer.pkl`` or ``decision_outcomes.jsonl`` (only reads it),
and never enters a trade path — the same operational discipline as
``paper_trader/ml/stop_out_audit.py`` / ``baseline_compare.py`` /
``calibration.py`` / ``gate_audit.py``.

**Why this is not any existing tool.** ``stop_out_audit`` audits the
``backtest._buy`` ``stop_loss = price * 0.92`` band (the -8% downside
arm). The deployed code writes BOTH ``stop_loss`` AND
``take_profit = price * 1.15`` (the +15% upside arm) on EVERY BUY, yet
NO read-only analyzer measures whether the take_profit band actually
realizes economic benefit — its realized effect was invisible until
now. The intraperiod-max field ``_compute_decision_outcomes`` started
persisting on 2026-05-23 (``forward_intraperiod_max_5d``) is the
captured-but-unused data this analyzer turns into a verdict; it is the
FIRST consumer of that column. ``stop_out_audit`` consumes the
matching ``forward_intraperiod_min_5d``.

The deeper question this answers is the classic quant **MFE conversion**
question: across BUYs that *reached* a positive intraperiod peak (MFE,
Maximum Favorable Excursion), how much of that peak did the 5d
endpoint actually retain? A low conversion ratio = the bot's positions
peak then crater — a take-profit band would have captured the peak.

**Method.** For each BUY outcome row with a finite
``forward_intraperiod_max_5d`` AND a finite ``forward_return_5d``:

  * If ``intra_max >= TP_PCT`` (the take-profit would have triggered
    intra-window), the realized return *with TP* is ``TP_PCT`` (the TP
    fired at or near the band; the actual fill price is unknowable from
    closes alone, so use the band itself — symmetric with
    ``stop_out_audit``'s conservative-band assumption).
  * Else: realized return *with TP* equals ``forward_return_5d`` (the
    TP never fired, so the endpoint reading is what would have been
    captured).

The aggregate ``tp_benefit_pct`` is the mean realized return *with*
the TP minus the mean realized return *without* — positive when the
TP captures more upside than it forfeits in trades that would have
recovered further. Together with ``stop_out_audit``'s verdict this
gives a skeptical quant the FULL realized economic effect of the
inherited ``_buy`` band pair — neither arm was measurable before.

Conversion ratio: across BUYs with positive MFE, the mean ratio
``endpoint / mfe`` — bounded above by 1.0 (perfect hold), 0 means the
position reverted exactly to entry, negative means it closed below
entry despite a positive peak. Reported alongside the TP verdict
because the take-profit's economic logic only fires when conversion
is poor; HIGH conversion + neutral TP verdict is the "the gate is
already capturing the peak naturally" shape, LOW conversion + TP_HELPS
is the "positions peak then revert and the band rescues realized
return" shape.

| Verdict | Meaning |
|---------|---------|
| ``INSUFFICIENT_DATA`` | < ``MIN_BUYS`` BUY rows carry a finite ``forward_intraperiod_max_5d``; the band behaviour is not measurable. The historical 8.7k-row corpus pre-dates the intraperiod feature, so cycles that retrained AFTER the feature shipped are what populate this row. |
| ``TP_HELPS`` | ``tp_benefit_pct`` >= ``BENEFIT_MARGIN`` (e.g. +0.30pp realized) — the take-profit measurably captures peak that the 5d endpoint forfeits. |
| ``TP_HURTS`` | ``tp_benefit_pct`` <= -``BENEFIT_MARGIN`` — the take-profit exits trades that would have run further, dragging realized return down. |
| ``TP_NEUTRAL`` | benefit within ``±BENEFIT_MARGIN`` — the TP is in the noise; sizing variance with no measurable economic effect. |

```bash
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.mfe_conversion
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.mfe_conversion --json
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.mfe_conversion --tp 10.0
```
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable


# Take-profit percentage — the deployed band ``backtest._buy`` writes is
# ``price * 1.15`` (a 15% gain from entry triggers). Module-level so a
# tuning change is a single reviewable edit, symmetric with
# ``stop_out_audit.STOP_PCT``.
TP_PCT = 15.0

# Minimum BUYs with finite intraperiod data before any verdict. Below this
# the report is honestly ``INSUFFICIENT_DATA``. Same threshold as
# ``stop_out_audit.MIN_BUYS`` so the two audits report comparable n_buys
# minima.
MIN_BUYS = 30

# Realized-return margin (percentage points) the TP must clear / lose by
# before a non-neutral verdict is declared. Symmetric with
# ``stop_out_audit.BENEFIT_MARGIN`` — 0.30pp is roughly one standard error
# on a 1000-trade aggregate at σ(target) ≈ 12pp. Anything tighter is
# sampling noise.
BENEFIT_MARGIN = 0.30


def _to_finite_float(v) -> float | None:
    """Return ``float(v)`` if finite, else None. Same contract as
    ``stop_out_audit._to_finite_float`` — kept locally (not imported)
    because tooling that copies one diagnostic into a fresh environment
    should not depend on a sibling module's helpers."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _tp_protected_return(forward_return_5d: float,
                          intra_max: float,
                          tp_pct: float = TP_PCT) -> float:
    """Realized return for ONE BUY with the documented +tp_pct% take-profit.

    ``intra_max`` is the best intraperiod return reached (signed %).
    If it rises at or above ``+tp_pct``, the TP fires — model the fill
    at exactly ``+tp_pct`` (a real fill on a gap-up would be better;
    this is the conservative-fill assumption a quant would treat as a
    LOWER bound on the TP's benefit, mirroring
    ``stop_out_audit._stop_protected_return``'s pessimistic stop-fill
    assumption on the symmetric arm).

    If ``intra_max < +tp_pct`` (no trigger), the position rides to the
    5d endpoint and ``forward_return_5d`` is captured.

    Pure, total, never raises."""
    if intra_max >= tp_pct:
        return tp_pct
    return forward_return_5d


def _iter_rows(path: Path) -> Iterable[dict]:
    """Stream one JSON record per line, silently dropping unparseable rows.

    Mirrors the line-tolerant loader used throughout the codebase
    (``_compute_decision_outcomes`` / ``_inject_and_train`` /
    ``stop_out_audit._iter_rows``) — a single corrupt line must not
    abort an audit run.
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


def _median(s: list[float]) -> float:
    """Median of a non-empty pre-sorted list."""
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def analyze(outcomes_path: "Path | str | None" = None,
            tp_pct: float = TP_PCT,
            benefit_margin: float = BENEFIT_MARGIN,
            min_buys: int = MIN_BUYS) -> dict:
    """Compute the MFE-conversion / take-profit audit report.

    Reads ``data/decision_outcomes.jsonl`` (or the passed path),
    filters to BUYs with a finite ``forward_intraperiod_max_5d`` AND a
    finite ``forward_return_5d``, then computes the with-TP vs
    without-TP aggregate realized return AND the MFE conversion ratio
    (endpoint / mfe) across BUYs whose intraperiod peak was positive.

    Returns a JSON-safe dict — never raises. On any fault returns an
    INSUFFICIENT_DATA envelope so a ledger consumer can persist the
    failure mode rather than swallow it. Output shape mirrors
    ``stop_out_audit.analyze`` for the band fields (n_*, mean_*,
    benefit_*) so a ledger / dashboard reader can table both audits
    side-by-side with no per-key remapping.
    """
    if outcomes_path is None:
        outcomes_path = Path(__file__).resolve().parent.parent.parent / "data" / "decision_outcomes.jsonl"
    else:
        outcomes_path = Path(outcomes_path)

    empty = {
        "status": "insufficient_data",
        "verdict": "INSUFFICIENT_DATA",
        "tp_pct": tp_pct,
        "n_buys": 0,
        "n_with_intraperiod": 0,
        "n_tp_triggered": 0,
        "pct_tp_triggered": None,
        "n_positive_mfe": 0,
        "n_reverted": 0,
        "pct_reverted": None,
        "mean_realized_return_pct": None,
        "mean_tp_protected_return_pct": None,
        "tp_benefit_pct": None,
        "median_realized_return_pct": None,
        "median_tp_protected_return_pct": None,
        "mean_mfe_pct": None,
        "median_mfe_pct": None,
        "mean_conversion_ratio": None,
        "median_conversion_ratio": None,
        "hint": None,
    }

    if not outcomes_path.exists():
        empty["hint"] = f"outcomes file not found: {outcomes_path}"
        return empty

    n_buys = 0
    n_with_intra = 0
    n_tp_triggered = 0
    n_positive_mfe = 0
    n_reverted = 0
    realized: list[float] = []
    tp_protected: list[float] = []
    mfe_vals: list[float] = []
    conversion_ratios: list[float] = []

    try:
        for row in _iter_rows(outcomes_path):
            if not isinstance(row, dict):
                continue
            action = str(row.get("action") or "").upper()
            if action != "BUY":
                continue
            n_buys += 1
            fwd = _to_finite_float(row.get("forward_return_5d"))
            intra = _to_finite_float(row.get("forward_intraperiod_max_5d"))
            if fwd is None or intra is None:
                continue
            n_with_intra += 1
            tp = _tp_protected_return(fwd, intra, tp_pct=tp_pct)
            if intra >= tp_pct:
                n_tp_triggered += 1
            realized.append(fwd)
            tp_protected.append(tp)
            mfe_vals.append(intra)
            # Conversion ratio (endpoint / mfe) only defined for positive
            # MFE — a non-positive peak means the position never rallied,
            # so "fraction of peak captured" is meaningless. The reverted
            # counter is a separate signal: how often a BUY that DID
            # peak positive nonetheless reverted to a non-positive
            # endpoint (the "peak then crater" pattern a TP captures).
            if intra > 0.0:
                n_positive_mfe += 1
                # Floor ratio at -10.0 — a tiny intra peak (e.g. +0.1%)
                # divided into a strongly negative endpoint (e.g. -8%)
                # would produce a meaningless -80 ratio that dominates
                # the mean. Cap symmetric with the inference-side
                # ±PRED_CLAMP_PCT discipline (a defensible ratio bound
                # for a sane quant report). -10 keeps real peak-then-
                # crater shape (a +5% peak that ends at -3% gives -0.6)
                # while neutralising pathologically thin peaks.
                ratio = fwd / intra
                ratio = max(-10.0, min(1.0, ratio))
                conversion_ratios.append(ratio)
                if fwd <= 0.0:
                    n_reverted += 1
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
    mean_tp = sum(tp_protected) / len(tp_protected)
    benefit = mean_tp - mean_real
    mean_mfe = sum(mfe_vals) / len(mfe_vals)

    realized_sorted = sorted(realized)
    tp_sorted = sorted(tp_protected)
    mfe_sorted = sorted(mfe_vals)
    median_real = _median(realized_sorted)
    median_tp = _median(tp_sorted)
    median_mfe = _median(mfe_sorted)

    mean_conv: float | None = None
    median_conv: float | None = None
    if conversion_ratios:
        mean_conv = sum(conversion_ratios) / len(conversion_ratios)
        median_conv = _median(sorted(conversion_ratios))

    if benefit >= benefit_margin:
        verdict = "TP_HELPS"
    elif benefit <= -benefit_margin:
        verdict = "TP_HURTS"
    else:
        verdict = "TP_NEUTRAL"

    return {
        "status": "ok",
        "verdict": verdict,
        "tp_pct": tp_pct,
        "benefit_margin_pp": benefit_margin,
        "n_buys": n_buys,
        "n_with_intraperiod": n_with_intra,
        "n_tp_triggered": n_tp_triggered,
        "pct_tp_triggered": round(n_tp_triggered / n_with_intra * 100.0, 2),
        "n_positive_mfe": n_positive_mfe,
        "n_reverted": n_reverted,
        "pct_reverted": (round(n_reverted / n_positive_mfe * 100.0, 2)
                         if n_positive_mfe else None),
        "mean_realized_return_pct": round(mean_real, 4),
        "mean_tp_protected_return_pct": round(mean_tp, 4),
        "tp_benefit_pct": round(benefit, 4),
        "median_realized_return_pct": round(median_real, 4),
        "median_tp_protected_return_pct": round(median_tp, 4),
        "mean_mfe_pct": round(mean_mfe, 4),
        "median_mfe_pct": round(median_mfe, 4),
        "mean_conversion_ratio": (round(mean_conv, 4)
                                  if mean_conv is not None else None),
        "median_conversion_ratio": (round(median_conv, 4)
                                    if median_conv is not None else None),
        "hint": None,
    }


# ---------------------------------------------------------------------------
# CLI: `python3 -m paper_trader.ml.mfe_conversion [--json] [--tp 10.0]`
#
# Pattern mirrors the sibling read-only diagnostic CLIs
# (stop_out_audit, baseline_compare, conviction_calibration). Returns 0
# only when a decisive non-neutral verdict was produced — `$?` lets a
# shell caller gate on "the take-profit is decisively helping/hurting".
# ---------------------------------------------------------------------------


def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.mfe_conversion",
        description="MFE-conversion / take-profit audit: would the "
                    "documented +15%% take_profit band have improved or "
                    "harmed realized return? Read-only — reads "
                    "decision_outcomes.jsonl, never trains, never modifies "
                    "the pickle or the gate. Sibling to stop_out_audit on "
                    "the matching downside arm.",
    )
    p.add_argument("--outcomes", default=None,
                   help="Path to decision_outcomes.jsonl (default: the "
                        "repo data/ path).")
    p.add_argument("--tp", type=float, default=TP_PCT, dest="tp_pct",
                   help=f"Take-profit percentage (default {TP_PCT}). The "
                        f"deployed band is _buy()'s 1.15 = 15.0%%.")
    p.add_argument("--margin", type=float, default=BENEFIT_MARGIN,
                   dest="benefit_margin",
                   help=f"Realized-return margin (pp) the TP must clear "
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
        tp_pct=args.tp_pct,
        benefit_margin=args.benefit_margin,
        min_buys=args.min_buys,
    )

    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
        return 0 if rep["verdict"] in {"TP_HELPS", "TP_HURTS"} else 1

    verdict = rep["verdict"]
    print(f"[mfe_conversion] verdict={verdict}  tp={rep['tp_pct']}%")
    print(f"  n_buys={rep['n_buys']}  n_with_intraperiod={rep['n_with_intraperiod']}")
    if rep["n_with_intraperiod"] > 0:
        print(f"  n_tp_triggered={rep['n_tp_triggered']} "
              f"({rep['pct_tp_triggered']}% of with-intraperiod)")
        print(f"  n_positive_mfe={rep['n_positive_mfe']}  "
              f"n_reverted={rep['n_reverted']}"
              + (f" ({rep['pct_reverted']}% of positive-MFE)"
                 if rep['pct_reverted'] is not None else ""))
    if rep.get("mean_realized_return_pct") is not None:
        print(f"  realized:    mean={rep['mean_realized_return_pct']:+.3f}pp  "
              f"median={rep['median_realized_return_pct']:+.3f}pp")
        print(f"  with-TP:     mean={rep['mean_tp_protected_return_pct']:+.3f}pp  "
              f"median={rep['median_tp_protected_return_pct']:+.3f}pp")
        print(f"  benefit:     {rep['tp_benefit_pct']:+.3f}pp  "
              f"(margin: ±{rep['benefit_margin_pp']}pp)")
        print(f"  MFE:         mean={rep['mean_mfe_pct']:+.3f}pp  "
              f"median={rep['median_mfe_pct']:+.3f}pp")
        if rep.get("mean_conversion_ratio") is not None:
            print(f"  conversion:  mean={rep['mean_conversion_ratio']:+.3f}  "
                  f"median={rep['median_conversion_ratio']:+.3f}  "
                  f"(endpoint / MFE across positive-MFE BUYs; "
                  f"1.0 = held peak, 0 = back to entry, <0 = below entry)")
    if rep.get("hint"):
        print(f"  hint: {rep['hint']}")
    return 0 if verdict in {"TP_HELPS", "TP_HURTS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Risk-adjusted performance metrics for backtest runs.

The continuous loop ranks runs by ``total_return_pct`` — a headline metric
that conflates skill with risk. A +200% return with -90% intraperiod
drawdown carries the same surface reading as a +200% return with -10%
drawdown, but only the latter is investable. A skeptical quant judges
runs by **risk-adjusted** performance: annualized Sharpe ratio, max
drawdown (MDD), and Calmar ratio (return / |MDD|).

This module exposes those readings derived from each run's persisted
``equity_curve_json`` blob, without retraining or modifying any row.
The existing return-ranking remains the training target by design (the
DecisionScorer learns realized return), so this is a pure read-only
diagnostic layered on top.

Pure / module-level constants for testability. CLI:

    python3 -m paper_trader.ml.run_risk_metrics
        → top 20 runs ranked by Calmar (and by Sharpe in a second table)
    python3 -m paper_trader.ml.run_risk_metrics --run-id 42
        → detailed risk metrics + verdict for one run
    python3 -m paper_trader.ml.run_risk_metrics --json
    python3 -m paper_trader.ml.run_risk_metrics --rank-by sharpe
    python3 -m paper_trader.ml.run_risk_metrics --top 50

Exit code mirrors the rest of the OOS suite: 0 on a successful report,
1 on data unavailability.
"""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_BACKTEST_DB = ROOT / "backtest.db"

# Trading days per year — standard quant convention. SQRT(252) is the
# canonical Sharpe annualization factor for daily-frequency returns.
TRADING_DAYS_PER_YEAR = 252

# Below this many equity points (post-deduplication of consecutive equal
# values from non-trading days) no meaningful Sharpe / MDD can be computed.
# Matches the documented "5d window" minimum: we need at least a week's
# worth of data points so daily-return std isn't dominated by sampling noise.
MIN_POINTS = 5

# Calmar verdict ladder — operator-readable summary of the
# return/|drawdown| ratio. Above 1.0 means realized return exceeded peak
# drawdown — investment-grade by the textbook Calmar bar. Below 0 means
# the run lost money overall. None means insufficient data.
CALMAR_BANDS: tuple[tuple[str, float, float], ...] = (
    ("INVESTMENT_GRADE", 1.0, float("inf")),    # Calmar >= 1.0
    ("ACCEPTABLE",       0.5, 1.0),             # Calmar 0.5..1.0
    ("MARGINAL",         0.0, 0.5),             # Calmar 0..0.5
    ("LOSS_MAKING",      float("-inf"), 0.0),   # Calmar < 0
)


def _calmar_verdict(calmar: float | None) -> str:
    """Map a Calmar ratio to a human-readable verdict.

    None → INSUFFICIENT_DATA. NaN/Inf are tolerated and routed by sign:
    +Inf (return positive with zero drawdown — impossible in real data but
    possible in a synthetic test) → INVESTMENT_GRADE. -Inf → LOSS_MAKING.
    """
    if calmar is None:
        return "INSUFFICIENT_DATA"
    if not math.isfinite(calmar):
        return "INVESTMENT_GRADE" if calmar > 0 else "LOSS_MAKING"
    for name, lo, hi in CALMAR_BANDS:
        if lo <= calmar < hi:
            return name
    return "INSUFFICIENT_DATA"  # unreachable given the bands cover all reals


def _daily_returns(values: list[float]) -> list[float]:
    """Simple percent returns r_t = (v_t / v_{t-1}) - 1 between consecutive
    points. A non-finite or non-positive prior value yields a 0 return
    (the textbook handling for a stale / missing point — see PriceCache
    walk-back collision discipline in backtest.py).
    """
    out: list[float] = []
    for prev, cur in zip(values, values[1:]):
        if (not math.isfinite(prev) or not math.isfinite(cur)
                or prev <= 0):
            out.append(0.0)
            continue
        out.append(cur / prev - 1.0)
    return out


def sharpe_ratio(equity: list[float],
                 risk_free_rate_annual: float = 0.0) -> float | None:
    """Annualized Sharpe ratio from a daily equity series.

    SR = sqrt(252) * (mean_excess_daily / std_daily). Excess return uses
    the annual risk-free rate converted to a daily geometric equivalent.
    Returns None when fewer than MIN_POINTS, or when daily-return std is
    zero (a flat-cash series — the metric is undefined). Never raises.
    """
    if equity is None or len(equity) < MIN_POINTS:
        return None
    rets = _daily_returns(equity)
    if not rets:
        return None
    rf_daily = (1.0 + risk_free_rate_annual) ** (
        1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    excess = [r - rf_daily for r in rets]
    n = len(excess)
    if n < 2:
        return None
    mean_e = sum(excess) / n
    var = sum((e - mean_e) ** 2 for e in excess) / (n - 1)
    sd = math.sqrt(var)
    if sd == 0.0 or not math.isfinite(sd):
        return None
    sr = math.sqrt(TRADING_DAYS_PER_YEAR) * (mean_e / sd)
    if not math.isfinite(sr):
        return None
    return round(sr, 4)


def max_drawdown_pct(equity: list[float]) -> float | None:
    """Max drawdown as a NEGATIVE percent (e.g. -25.0 for a 25% peak-to-trough
    drawdown). Returns None when fewer than MIN_POINTS, or when the running
    peak is non-positive (a corrupted series). 0.0 for a strictly monotonic
    rising series (no drawdown). Never raises.
    """
    if equity is None or len(equity) < MIN_POINTS:
        return None
    peak = float("-inf")
    worst = 0.0  # smallest (most-negative) dd seen
    for v in equity:
        if not math.isfinite(v):
            continue
        if v > peak:
            peak = v
        if peak > 0:
            dd_pct = (v - peak) / peak * 100.0
            if dd_pct < worst:
                worst = dd_pct
    if peak <= 0:
        return None
    return round(worst, 4)


def annualized_return_pct(equity: list[float], days: int | None) -> float | None:
    """CAGR-style annualized return from start-value to end-value over
    ``days`` calendar days. Returns None on insufficient data or non-positive
    start value. ``days`` is calendar-days (365.25/year basis) — that
    matches BacktestStore.all_runs's ``duration_days`` computation so the
    two views agree.
    """
    if equity is None or len(equity) < 2 or not days or days <= 0:
        return None
    start = equity[0]
    end = equity[-1]
    if not (math.isfinite(start) and math.isfinite(end)) or start <= 0:
        return None
    years = days / 365.25
    if years <= 0:
        return None
    growth = end / start
    if growth <= 0:
        # End value <= 0 — investor wiped out. Return -100% annualized.
        return round(-100.0, 4)
    cagr = growth ** (1.0 / years) - 1.0
    return round(cagr * 100.0, 4)


def calmar_ratio(annualized_ret_pct: float | None,
                 mdd_pct: float | None) -> float | None:
    """Calmar = annualized_return / |max_drawdown|. Returns None when either
    input is missing. When mdd_pct is 0 (no drawdown — a strictly rising
    series) and annualized return is positive, returns +inf (the test
    layer uses ``math.isfinite`` to detect this corner). Negative annualized
    return with zero drawdown is investable in theory but never observed —
    returns -inf so the verdict ladder catches it.
    """
    if annualized_ret_pct is None or mdd_pct is None:
        return None
    if mdd_pct == 0.0:
        # No drawdown — degenerate case. A real backtest always has some.
        if annualized_ret_pct > 0:
            return float("inf")
        if annualized_ret_pct < 0:
            return float("-inf")
        return 0.0
    return round(annualized_ret_pct / abs(mdd_pct), 4)


def compute_run_risk(equity_curve: list[dict],
                     duration_days: int | None = None,
                     risk_free_rate_annual: float = 0.0) -> dict:
    """Top-level: compute risk metrics from a run's equity_curve_json.

    ``equity_curve`` is the persisted list of ``{date, value, cash}`` dicts.
    ``duration_days`` (calendar-days from start_date to end_date) is used
    for the CAGR; when None we fall back to len(curve)-1 days (a rough
    proxy that assumes one point per day — true for the deployed engine).

    Returns a JSON-safe dict::

        {
          "n_points": int,
          "start_value": float | None,
          "end_value": float | None,
          "duration_days": int | None,
          "annualized_return_pct": float | None,
          "max_drawdown_pct": float | None,       # negative (or 0)
          "sharpe_ratio": float | None,           # annualized
          "calmar_ratio": float | None,
          "verdict": str,                         # CALMAR_BANDS name
        }

    Pure and total — never raises. Missing / corrupt curve degrades to
    None metrics + INSUFFICIENT_DATA verdict.
    """
    out: dict[str, Any] = {
        "n_points": 0,
        "start_value": None,
        "end_value": None,
        "duration_days": duration_days,
        "annualized_return_pct": None,
        "max_drawdown_pct": None,
        "sharpe_ratio": None,
        "calmar_ratio": None,
        "verdict": "INSUFFICIENT_DATA",
    }
    if not isinstance(equity_curve, list) or not equity_curve:
        return out
    values: list[float] = []
    for p in equity_curve:
        if not isinstance(p, dict):
            continue
        v = p.get("value")
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fv):
            values.append(fv)
    if not values:
        return out
    out["n_points"] = len(values)
    out["start_value"] = round(values[0], 4)
    out["end_value"] = round(values[-1], 4)

    # Default duration when caller didn't pass one.
    if duration_days is None:
        # Try to derive from the first/last date strings.
        try:
            d0 = date.fromisoformat(str(equity_curve[0].get("date") or "")[:10])
            d1 = date.fromisoformat(str(equity_curve[-1].get("date") or "")[:10])
            duration_days = (d1 - d0).days
            out["duration_days"] = duration_days
        except (TypeError, ValueError):
            duration_days = max(1, len(values) - 1)
            out["duration_days"] = duration_days

    out["annualized_return_pct"] = annualized_return_pct(values, duration_days)
    out["max_drawdown_pct"] = max_drawdown_pct(values)
    out["sharpe_ratio"] = sharpe_ratio(values, risk_free_rate_annual)
    out["calmar_ratio"] = calmar_ratio(out["annualized_return_pct"],
                                       out["max_drawdown_pct"])
    out["verdict"] = _calmar_verdict(out["calmar_ratio"])
    return out


# ---------------------------------------------------------------------------
# DB readers
# ---------------------------------------------------------------------------

def _load_runs_with_curves(db_path: Path,
                           run_ids: list[int] | None = None,
                           limit: int | None = None) -> list[dict]:
    """Read backtest_runs (with equity_curve_json) from the persisted DB.

    Read-only — never writes. Honors a per-run-id filter or a global LIMIT
    for the leaderboard view. Returns [] on DB-missing / unreadable.
    """
    if not Path(db_path).exists():
        return []
    conn = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
        conn.row_factory = sqlite3.Row
        where = ""
        params: tuple = ()
        if run_ids:
            placeholders = ",".join("?" * len(run_ids))
            where = f" WHERE run_id IN ({placeholders})"
            params = tuple(int(i) for i in run_ids)
        sql = (
            f"SELECT run_id, seed, start_date, end_date, start_value, "
            f"final_value, total_return_pct, status, equity_curve_json "
            f"FROM backtest_runs{where} "
            f"ORDER BY run_id DESC"
        )
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        rows = conn.execute(sql, params).fetchall()
        out: list[dict] = []
        for row in rows:
            d = dict(row)
            try:
                d["equity_curve"] = json.loads(d.pop("equity_curve_json") or "[]")
            except Exception:
                d["equity_curve"] = []
            try:
                s = date.fromisoformat(d["start_date"])
                e = date.fromisoformat(d["end_date"])
                d["duration_days"] = (e - s).days
            except Exception:
                d["duration_days"] = None
            out.append(d)
        return out
    except Exception:
        return []
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def leaderboard(db_path: Path = DEFAULT_BACKTEST_DB,
                rank_by: str = "calmar",
                top: int = 20,
                limit_scan: int = 500) -> dict:
    """Compute risk metrics for the most recent ``limit_scan`` runs and
    return the top ``top`` by ``rank_by`` ('calmar' | 'sharpe' | 'mdd' |
    'return'). MDD ranks LEAST-negative-first (smallest drawdown).

    Pure read — uses the read-only DB URI. ``rank_by='return'`` exists for
    the cross-check view (it should agree with the continuous-loop ranking).
    """
    rank_by = rank_by.lower()
    if rank_by not in {"calmar", "sharpe", "mdd", "return"}:
        raise ValueError(f"rank_by must be calmar|sharpe|mdd|return, got {rank_by!r}")
    runs = _load_runs_with_curves(db_path, limit=limit_scan)
    enriched: list[dict] = []
    for r in runs:
        if r.get("status") != "complete":
            continue
        m = compute_run_risk(r.get("equity_curve") or [],
                             duration_days=r.get("duration_days"))
        m["run_id"] = r.get("run_id")
        m["total_return_pct"] = r.get("total_return_pct")
        m["start_date"] = r.get("start_date")
        m["end_date"] = r.get("end_date")
        enriched.append(m)
    if not enriched:
        return {"status": "no_runs", "rank_by": rank_by, "top": []}

    def _key(d: dict):
        if rank_by == "calmar":
            v = d.get("calmar_ratio")
        elif rank_by == "sharpe":
            v = d.get("sharpe_ratio")
        elif rank_by == "mdd":
            v = d.get("max_drawdown_pct")  # less-negative = better
        else:
            v = d.get("total_return_pct")
        # None sorts last regardless of order
        if v is None:
            return (1, 0.0)
        if isinstance(v, float):
            if math.isnan(v):
                return (1, 0.0)
            if math.isinf(v):
                # +inf ranks first (best — degenerate "no drawdown" case),
                # -inf ranks last (loss with no drawdown — also degenerate).
                # Live data never produces inf; this guard exists so a
                # synthetic test or a one-point curve cannot displace real
                # runs from the top.
                return (0, -math.inf) if v > 0 else (1, math.inf)
        return (0, -float(v))  # higher value first

    enriched.sort(key=_key)
    return {
        "status": "ok",
        "rank_by": rank_by,
        "n_scanned": len(enriched),
        "top": enriched[:top],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.run_risk_metrics",
        description=(
            "Risk-adjusted backtest metrics: annualized Sharpe ratio, max "
            "drawdown, Calmar ratio. The continuous loop ranks by raw "
            "return_pct; this is the quant-grade companion view."
        ),
    )
    p.add_argument("--db", default=str(DEFAULT_BACKTEST_DB),
                   help="Path to backtest.db (default: project backtest.db)")
    p.add_argument("--run-id", type=int, default=None, dest="run_id",
                   help="Show detailed risk view for one run instead of the leaderboard")
    p.add_argument("--rank-by", default="calmar", dest="rank_by",
                   choices=("calmar", "sharpe", "mdd", "return"),
                   help="Leaderboard ranking metric")
    p.add_argument("--top", type=int, default=20,
                   help="Top N rows to print (default 20)")
    p.add_argument("--limit-scan", type=int, default=500, dest="limit_scan",
                   help="Scan the most recent N runs (default 500)")
    p.add_argument("--rf", type=float, default=0.0, dest="rf",
                   help="Annual risk-free rate for Sharpe (default 0)")
    p.add_argument("--json", action="store_true",
                   help="Machine-readable JSON output")
    return p


def main(argv: list[str] | None = None) -> int:
    import sys
    args = _build_arg_parser().parse_args(
        sys.argv[1:] if argv is None else argv)
    db = Path(args.db)

    if args.run_id is not None:
        runs = _load_runs_with_curves(db, run_ids=[args.run_id])
        if not runs:
            msg = {"status": "not_found", "run_id": args.run_id, "db": str(db)}
            print(json.dumps(msg, indent=2) if args.json
                  else f"[run_risk] run_id={args.run_id} not found in {db}")
            return 1
        r = runs[0]
        m = compute_run_risk(r.get("equity_curve") or [],
                             duration_days=r.get("duration_days"),
                             risk_free_rate_annual=args.rf)
        m["run_id"] = r.get("run_id")
        m["total_return_pct"] = r.get("total_return_pct")
        m["start_date"] = r.get("start_date")
        m["end_date"] = r.get("end_date")
        m["status"] = r.get("status")
        if args.json:
            print(json.dumps(m, indent=2, sort_keys=True))
            return 0
        print(f"[run_risk] run_id={m['run_id']}  "
              f"({m['start_date']} → {m['end_date']}, "
              f"{m['duration_days']}d, status={m['status']})")
        print(f"  start ${m['start_value']:.2f}  end ${m['end_value']:.2f}  "
              f"n_points={m['n_points']}")
        if m.get("annualized_return_pct") is not None:
            print(f"  CAGR (annualized return): {m['annualized_return_pct']:+.2f}%")
        else:
            print("  CAGR: n/a")
        if m.get("max_drawdown_pct") is not None:
            print(f"  Max drawdown: {m['max_drawdown_pct']:+.2f}%")
        else:
            print("  Max drawdown: n/a")
        if m.get("sharpe_ratio") is not None:
            print(f"  Sharpe ratio (annualized, rf={args.rf:.1%}): "
                  f"{m['sharpe_ratio']:+.2f}")
        else:
            print("  Sharpe: n/a")
        if m.get("calmar_ratio") is not None:
            c = m["calmar_ratio"]
            if math.isfinite(c):
                print(f"  Calmar ratio: {c:+.2f}")
            else:
                sign = "+" if c > 0 else "-"
                print(f"  Calmar ratio: {sign}inf  (degenerate — no drawdown)")
        else:
            print("  Calmar: n/a")
        print(f"  verdict: {m['verdict']}")
        return 0

    out = leaderboard(db, rank_by=args.rank_by, top=args.top,
                      limit_scan=args.limit_scan)
    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0 if out.get("status") == "ok" else 1
    if out["status"] != "ok":
        print(f"[run_risk] no runs at {db}")
        return 1
    label = {"calmar": "Calmar", "sharpe": "Sharpe", "mdd": "MaxDD",
             "return": "Total Return"}[args.rank_by]
    print(f"[run_risk] top {min(args.top, len(out['top']))} of "
          f"{out['n_scanned']} complete runs ranked by {label}")
    print(f"  {'run_id':>7} {'CAGR%':>9} {'MDD%':>9} {'Sharpe':>8} "
          f"{'Calmar':>8} verdict")
    for r in out["top"]:
        cagr = ("n/a" if r["annualized_return_pct"] is None
                else f"{r['annualized_return_pct']:+.2f}")
        mdd = ("n/a" if r["max_drawdown_pct"] is None
               else f"{r['max_drawdown_pct']:+.2f}")
        sr = ("n/a" if r["sharpe_ratio"] is None
              else f"{r['sharpe_ratio']:+.2f}")
        c = r["calmar_ratio"]
        if c is None:
            cs = "n/a"
        elif not math.isfinite(c):
            cs = "+inf" if c > 0 else "-inf"
        else:
            cs = f"{c:+.2f}"
        print(f"  {r['run_id']:>7} {cagr:>9} {mdd:>9} {sr:>8} {cs:>8} "
              f"{r['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Transaction-cost drag audit — how much of the backtest's headline
out-performance survives a realistic per-trade cost?

This is a **read-only diagnostic**. It never trains, never touches
`decision_scorer.pkl`, `decision_outcomes.jsonl`, `build_features`, or any
trade path — it reads `backtest.db` only (same operational discipline as
`paper_trader/ml/regime_audit.py` / `calibration.py` / `gate_audit.py`), so
it is safe to run against the live unattended continuous loop.

**Why this is not any existing module.** `model_rankings` /
`persona_leaderboard` rank runs by raw `vs_spy_pct`; `permutation_test`
(validation) asks whether signal *time-ordering* carries value;
`gate_audit` / `gate_pnl` measure the conviction gate's economic effect.
**None of them models transaction costs.** The continuous backtest engine
(`backtest.py::_execute_decision` / `_enforce_risk_exits`) fills every trade
at the daily close with **zero commission and zero slippage**, yet a typical
completed run makes ~1,200 trades. A skeptical quant deciding whether a
strategy is worth real capital must know how much of the headline
`vs_spy_pct` is an artifact of that frictionless assumption — that is the
exact question this module answers, and no other module can.

**Cost model — stated honestly, not hidden behind an approximation.** Each
trade of notional `value` (the `backtest_trades.value` column, `qty*price`,
positive for BUY and SELL alike) incurs `value * bps/10000` in cost. Per run
three numbers are reported side by side so the reader judges, not the module:

  * `total_cost_usd`        — `Σ(trade.value) * bps/10000`
  * `turnover_annualized`   — `Σ(trade.value) / mean(equity_curve) / years`,
    the standard finance turnover metric, so the number is comparable to
    strategies outside this repo.
  * `cost_adjusted_vs_spy_pct` — `vs_spy_pct - total_cost_usd/start_value*100`.
    **First-order, non-compounding**: it does NOT model that cost paid early
    would also have forgone its own compounding, so it is a *conservative
    upper bound* on surviving alpha (the true drag is slightly worse). This
    limitation is surfaced, never hidden.

**Verdict is corpus-derived, not a hardcoded threshold.** The module reports,
for each bps level, the corpus median `cost_adjusted_vs_spy_pct` and the
fraction of completed runs whose alpha flips negative. The verdict compares
the median at the *highest* bps level against zero — a fact about the
corpus, which cannot be wrong:

  * `COST_ROBUST`   — median cost-adjusted alpha at the max bps level is
    still positive: headline out-performance is not a frictionless artifact.
  * `COST_FRAGILE`  — raw median alpha is positive but flips <= 0 once the
    max bps cost is applied: the edge is cost-sensitive, treat headline
    `vs_spy_pct` with suspicion.
  * `COST_NEGATIVE` — raw median alpha is already <= 0 before any cost.
  * `INSUFFICIENT_DATA` — fewer than `MIN_RUNS` costable completed runs.

```bash
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.cost_drag
cd /home/zeph/paper-trader && python3 -m pytest tests/test_cost_drag.py -v
```
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

# Below this many costable completed runs the corpus medians are noise.
MIN_RUNS = 10
# Default per-trade cost levels (basis points of traded notional) swept by
# the diagnostic. 2 bps ≈ a liquid US large-cap retail round-trip leg;
# 10 bps covers leveraged-ETF spreads + slippage on a churny strategy.
DEFAULT_BPS_LEVELS: tuple[float, ...] = (2.0, 5.0, 10.0)


def _median(xs: list[float]) -> float | None:
    """Plain median; None on an empty list. Pure — no numpy dependency so the
    module stays importable on a sklearn-absent host (the `train_scorer`
    numpy-fallback discipline)."""
    s = sorted(x for x in xs if x is not None)
    n = len(s)
    if n == 0:
        return None
    mid = n // 2
    if n % 2:
        return float(s[mid])
    return float((s[mid - 1] + s[mid]) / 2.0)


def _years_between(start_iso: str | None, end_iso: str | None) -> float | None:
    """Calendar-year span between two ISO dates, or None if unparseable / a
    non-positive span (a degenerate start==end run we never want to annualize)."""
    try:
        s = date.fromisoformat(str(start_iso)[:10])
        e = date.fromisoformat(str(end_iso)[:10])
    except (TypeError, ValueError):
        return None
    days = (e - s).days
    if days <= 0:
        return None
    return days / 365.25


def _run_costs(run: dict, traded_notional: float,
               bps_levels: tuple[float, ...]) -> dict | None:
    """Build the per-run cost record, or None when the run is not costable
    (empty equity curve, unparseable window, non-positive denominators).

    Skipping rather than degrading silently is deliberate — a failed /
    orphaned run or a manual smoke-test run (e.g. run_id 99001/90001, which
    carry 0 trades and an empty curve) must not poison the corpus medians
    with a fabricated turnover or a divide-by-zero (the advisor's explicit
    empty-curve guard)."""
    start_value = float(run.get("start_value") or 0.0)
    if start_value <= 0:
        return None
    years = _years_between(run.get("start_date"), run.get("end_date"))
    if years is None:
        return None
    # Mean equity over the run's own equity curve — the standard turnover
    # denominator (average capital at risk), not start or final value.
    try:
        curve = json.loads(run.get("equity_curve_json") or "[]")
    except Exception:
        curve = []
    eq_vals = [float(p.get("value")) for p in curve
               if isinstance(p, dict) and p.get("value") is not None]
    mean_equity = (sum(eq_vals) / len(eq_vals)) if eq_vals else 0.0
    if mean_equity <= 0:
        return None

    turnover = traded_notional / mean_equity / years
    vs_spy = float(run.get("vs_spy_pct") or 0.0)

    per_bps: dict[str, dict] = {}
    for bps in bps_levels:
        cost_usd = traded_notional * bps / 10000.0
        # First-order, non-compounding drag expressed in return points
        # relative to the run's starting capital (see module docstring).
        cost_pct = cost_usd / start_value * 100.0
        per_bps[f"{bps:g}"] = {
            "bps": bps,
            "total_cost_usd": round(cost_usd, 2),
            "cost_pct": round(cost_pct, 4),
            "cost_adjusted_vs_spy_pct": round(vs_spy - cost_pct, 4),
        }

    return {
        "run_id": run.get("run_id"),
        "traded_notional_usd": round(traded_notional, 2),
        "mean_equity_usd": round(mean_equity, 2),
        "years": round(years, 3),
        "turnover_annualized": round(turnover, 3),
        "vs_spy_pct": round(vs_spy, 4),
        "total_return_pct": round(float(run.get("total_return_pct") or 0.0), 4),
        "n_trades": int(run.get("n_trades") or 0),
        "cost": per_bps,
    }


def analyze(db_path: "Path | str | None" = None,
            bps_levels: tuple[float, ...] = DEFAULT_BPS_LEVELS) -> dict:
    """Audit transaction-cost drag across all completed backtest runs.

    Reads `backtest.db` read-only. `db_path` defaults to the live
    `backtest.BACKTEST_DB`, resolved at *call* time so a test redirect is
    honoured (the AGENTS.md call-time-resolution rule). Never raises — every
    fault degrades to a `status='error'` dict.

    Returns ``{status, verdict, n_runs, bps_levels, corpus, runs, hint}``
    where ``corpus`` carries the median raw `vs_spy_pct` plus, per bps level,
    the median `cost_adjusted_vs_spy_pct` and the fraction of runs whose
    alpha flips negative."""
    out: dict = {"status": "error", "verdict": "INSUFFICIENT_DATA",
                 "n_runs": 0, "bps_levels": list(bps_levels),
                 "corpus": {}, "runs": [], "hint": ""}
    try:
        if db_path is None:
            from paper_trader.backtest import BACKTEST_DB
            db_path = BACKTEST_DB
        db_path = Path(db_path)
        if not db_path.exists():
            out["hint"] = f"no backtest db at {db_path}"
            return out

        conn = None
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True,
                                   timeout=10)
            conn.row_factory = sqlite3.Row
            runs = conn.execute(
                "SELECT run_id, start_value, total_return_pct, vs_spy_pct, "
                "n_trades, start_date, end_date, equity_curve_json "
                "FROM backtest_runs WHERE status='complete' AND n_trades > 0"
            ).fetchall()
            # Σ traded notional per run — value is qty*price, positive for
            # BUY and SELL legs alike, so this is total two-sided turnover.
            notional_rows = conn.execute(
                "SELECT run_id, COALESCE(SUM(value), 0.0) AS notional "
                "FROM backtest_trades GROUP BY run_id"
            ).fetchall()
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

        notional_by_run = {r["run_id"]: float(r["notional"] or 0.0)
                           for r in notional_rows}

        costed: list[dict] = []
        for row in runs:
            run = dict(row)
            traded = notional_by_run.get(run["run_id"], 0.0)
            if traded <= 0:
                # n_trades>0 but no trade rows / zero notional — not costable.
                continue
            rec = _run_costs(run, traded, bps_levels)
            if rec is not None:
                costed.append(rec)

        out["n_runs"] = len(costed)
        if len(costed) < MIN_RUNS:
            out["hint"] = (f"only {len(costed)} costable completed runs "
                           f"(need {MIN_RUNS})")
            out["runs"] = sorted(costed,
                                 key=lambda r: -r["turnover_annualized"])
            return out

        raw_alpha = [r["vs_spy_pct"] for r in costed]
        median_raw = _median(raw_alpha)
        corpus: dict = {
            "median_vs_spy_pct": (round(median_raw, 4)
                                  if median_raw is not None else None),
            "median_turnover_annualized": _round_or_none(
                _median([r["turnover_annualized"] for r in costed])),
            "median_n_trades": _round_or_none(
                _median([float(r["n_trades"]) for r in costed]), 0),
            "per_bps": {},
        }
        for bps in bps_levels:
            key = f"{bps:g}"
            adj = [r["cost"][key]["cost_adjusted_vs_spy_pct"] for r in costed]
            median_adj = _median(adj)
            n_below = sum(1 for a in adj if a < 0.0)
            corpus["per_bps"][key] = {
                "bps": bps,
                "median_cost_adjusted_vs_spy_pct": (
                    round(median_adj, 4) if median_adj is not None else None),
                "frac_runs_below_spy": round(n_below / len(adj), 4),
                "median_cost_pct": _round_or_none(
                    _median([r["cost"][key]["cost_pct"] for r in costed])),
            }
        out["corpus"] = corpus

        # Verdict — a fact about the corpus median at the HIGHEST bps level.
        max_key = f"{max(bps_levels):g}"
        median_adj_max = corpus["per_bps"][max_key][
            "median_cost_adjusted_vs_spy_pct"]
        if median_raw is None:
            out["verdict"] = "INSUFFICIENT_DATA"
        elif median_raw <= 0.0:
            out["verdict"] = "COST_NEGATIVE"
            out["hint"] = ("raw median vs_spy is already <= 0 — no edge to "
                           "erode")
        elif median_adj_max is not None and median_adj_max > 0.0:
            out["verdict"] = "COST_ROBUST"
            out["hint"] = (f"median alpha survives {max(bps_levels):g}bps: "
                           f"{median_raw:+.1f}% → {median_adj_max:+.1f}%")
        else:
            out["verdict"] = "COST_FRAGILE"
            out["hint"] = (f"median alpha flips negative at "
                           f"{max(bps_levels):g}bps: {median_raw:+.1f}% → "
                           f"{median_adj_max:+.1f}%")

        out["status"] = "ok"
        # Surface the churniest runs first — those bleed the most to costs.
        out["runs"] = sorted(costed,
                             key=lambda r: -r["turnover_annualized"])
        return out
    except Exception as e:
        out["hint"] = f"audit failed: {type(e).__name__}: {e}"
        return out


def _round_or_none(v: float | None, ndigits: int = 3):
    """round() that tolerates None (a corpus median over an empty list)."""
    return round(v, ndigits) if v is not None else None


def _cli() -> int:
    """`python3 -m paper_trader.ml.cost_drag [--json] [--bps 2,5,10]` —
    transaction-cost drag audit of every completed backtest run. Read-only.

    Exit code 2 on `COST_FRAGILE` (the operator-actionable branch — headline
    out-performance does not survive realistic costs), mirroring the
    cron-branchable exit-2 convention of `regime_audit` / `label_audit`.
    """
    import sys

    args = sys.argv[1:]
    bps_levels = DEFAULT_BPS_LEVELS
    if "--bps" in args:
        try:
            raw = args[args.index("--bps") + 1]
            parsed = tuple(sorted(float(x) for x in raw.split(",") if x))
            if parsed:
                bps_levels = parsed
        except (IndexError, ValueError):
            print("[cost_drag] bad --bps value — using defaults")

    rep = analyze(bps_levels=bps_levels)

    if "--json" in args:
        print(json.dumps(rep, indent=2, sort_keys=True))
        return 2 if rep.get("verdict") == "COST_FRAGILE" else 0

    print(f"VERDICT: {rep['verdict']}  ({rep.get('hint', '')})")
    print(f"  costable completed runs: {rep.get('n_runs', 0)}  "
          f"bps swept: {','.join(f'{b:g}' for b in bps_levels)}")
    corpus = rep.get("corpus") or {}
    if corpus:
        print(f"  median raw vs_spy: {corpus.get('median_vs_spy_pct')}%  "
              f"median turnover: {corpus.get('median_turnover_annualized')}x/yr  "
              f"median trades: {corpus.get('median_n_trades')}")
        for key, cell in (corpus.get("per_bps") or {}).items():
            print(f"  @ {cell['bps']:g}bps: median cost-adj alpha="
                  f"{cell['median_cost_adjusted_vs_spy_pct']}%  "
                  f"runs below SPY={cell['frac_runs_below_spy']*100:.0f}%  "
                  f"median cost={cell['median_cost_pct']}pp")
    runs = rep.get("runs") or []
    if runs:
        print(f"  churniest {min(5, len(runs))} runs (by annualized turnover):")
        for r in runs[:5]:
            print(f"    run {r['run_id']:<7} turnover={r['turnover_annualized']:>7.2f}x/yr "
                  f"trades={r['n_trades']:<5} vs_spy={r['vs_spy_pct']:+.1f}%")
    return 2 if rep.get("verdict") == "COST_FRAGILE" else 0


if __name__ == "__main__":
    raise SystemExit(_cli())

"""Run the 10k-monkey benchmark and rank the AI's backtest runs against it.

Picks the most data-rich cached price window as the simulation window, then
restricts AI ranking to backtest runs whose ``(start_date, end_date)``
exactly matches that window — comparing AI runs from 2004→2014 against
monkeys simulated on 2020→2025 would be meaningless.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

from paper_trader.analytics.monkey_benchmark import (
    default_window,
    list_available_windows,
    run_and_cache,
)
from paper_trader.backtest import WATCHLIST


def _ai_returns_for_window(db_path: Path, start: str, end: str) -> list[float]:
    """Total-return-pct values for completed AI runs on the exact window."""
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT total_return_pct FROM backtest_runs "
            "WHERE status='complete' AND total_return_pct IS NOT NULL "
            "AND start_date = ? AND end_date = ? "
            "ORDER BY run_id DESC",
            (start, end),
        ).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows if r[0] is not None]


def main() -> None:
    base = Path(__file__).resolve().parent
    db_path = base / "backtest.db"

    # Allow optional window override: `python3 run_monkey_benchmark.py START END`
    if len(sys.argv) == 3:
        start, end = sys.argv[1], sys.argv[2]
    else:
        start, end = default_window()
        print(f"[monkey] auto-selected window {start} → {end}")
        windows = list_available_windows()
        if windows:
            print(f"[monkey] {len(windows)} cached windows available; using the "
                  f"largest ({windows[0][2] // 1024} KB)")

    ai_returns = _ai_returns_for_window(db_path, start, end)
    print(f"[monkey] {len(ai_returns)} AI backtest runs match window "
          f"{start} → {end}")

    result = run_and_cache(
        tickers=WATCHLIST,
        start_date=start,
        end_date=end,
        ai_returns=ai_returns,
    )

    rp = result["random_portfolio"]
    ar = result["active_random"]
    print("\n=== MONKEY BENCHMARK RESULTS ===")
    print(f"Window: {start} → {end}  ({result['n_tickers']} tickers, "
          f"{result['n_monkeys']:,} monkeys)")
    print()
    print("Random Portfolio (buy-and-hold K=3 random tickers):")
    print(f"  median={rp['median_pct']}%  mean={rp['mean_pct']}%  "
          f"p10={rp['p10_pct']}%  p90={rp['p90_pct']}%  p99={rp['p99_pct']}%")
    print()
    print("Active Random Trader:")
    print(f"  median={ar['median_pct']}%  mean={ar['mean_pct']}%  "
          f"p10={ar['p10_pct']}%  p90={ar['p90_pct']}%  p99={ar['p99_pct']}%")

    rankings = result.get("ai_rankings") or []
    if rankings:
        ai_rets = [x["ai_return_pct"] for x in rankings]
        rp_med = rp["median_pct"]
        ar_med = ar["median_pct"]
        beats_rp = sum(1 for r in ai_rets if r > rp_med) / len(ai_rets) * 100
        beats_ar = sum(1 for r in ai_rets if r > ar_med) / len(ai_rets) * 100
        best_ai = max(ai_rets)
        best_idx = ai_rets.index(best_ai)
        best_rp_rank = rankings[best_idx]["vs_random_portfolio"]["beats_pct_of_monkeys"]
        best_ar_rank = rankings[best_idx]["vs_active_random"]["beats_pct_of_monkeys"]
        print(f"\nAI vs Monkeys ({len(ai_rets)} matching runs):")
        print(f"  {beats_rp:.1f}% of AI runs beat random-portfolio median")
        print(f"  {beats_ar:.1f}% of AI runs beat active-random median")
        print(f"  Best AI run: {best_ai:.1f}%  →  beats {best_rp_rank:.1f}% "
              f"(rp) / {best_ar_rank:.1f}% (active) of monkeys")
    else:
        print(f"\nNo AI backtest runs match window {start} → {end} — "
              "rankings will populate once continuous backtests draw this window.")


if __name__ == "__main__":
    main()

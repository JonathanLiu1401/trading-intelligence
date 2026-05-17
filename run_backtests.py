#!/usr/bin/env python3
"""Run 10 year-long backtests. Stores results in backtest.db.

Usage:
  python3 run_backtests.py            # 10 runs
  python3 run_backtests.py N          # N runs
"""
import sys

from paper_trader.backtest import BacktestEngine


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    engine = BacktestEngine()
    engine.run_all(n)


if __name__ == "__main__":
    main()

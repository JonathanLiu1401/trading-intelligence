"""Signal directional accuracy: verify if sentiment momentum calls led to correct price moves.

For each ticker with a bullish/bearish call in the last 6h window of
ticker_sentiment_momentum.json, fetch the day's price change via yfinance
and check if the direction was correct.  Writes a rolling accuracy report
to /home/zeph/logs/signal_accuracy.json.

Standalone: python3 -m analytics.signal_accuracy
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

MOMENTUM_PATH = Path("/home/zeph/logs/ticker_sentiment_momentum.json")
OUT_PATH = Path("/home/zeph/logs/signal_accuracy.json")
MIN_MOVE_PCT = 0.3   # ignore flat moves <0.3% — noise floor
MAX_TICKERS = 15     # cap yfinance calls per run to avoid rate-limits


def _fetch_price_change(ticker: str) -> float | None:
    """Return today's pct_change for ticker, or None on failure."""
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="2d")
        if len(hist) < 2:
            return None
        last = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2])
        if prev == 0:
            return None
        return round((last - prev) / prev * 100.0, 3)
    except Exception:
        return None


def compute() -> dict:
    now = datetime.now(timezone.utc)

    if not MOMENTUM_PATH.exists():
        return {
            "generated_at": now.isoformat(),
            "error": "ticker_sentiment_momentum.json not found — run that first",
        }

    data = json.loads(MOMENTUM_PATH.read_text())
    calls = data.get("tickers", [])[:MAX_TICKERS]

    if not calls:
        return {
            "generated_at": now.isoformat(),
            "momentum_generated_at": data.get("generated_at"),
            "evaluated": 0,
            "correct": 0,
            "accuracy": None,
            "tickers": [],
        }

    results: list[dict] = []
    correct = 0
    flat = 0
    failed = 0

    for entry in calls:
        ticker = entry["ticker"]
        direction = entry["direction"]  # "bullish" or "bearish"
        pct = _fetch_price_change(ticker)
        if pct is None:
            failed += 1
            results.append({
                "ticker": ticker,
                "direction": direction,
                "pct_change": None,
                "verdict": "no_data",
            })
            continue

        if abs(pct) < MIN_MOVE_PCT:
            flat += 1
            verdict = "flat"
        elif (direction == "bullish" and pct > 0) or (direction == "bearish" and pct < 0):
            correct += 1
            verdict = "hit"
        else:
            verdict = "miss"

        results.append({
            "ticker": ticker,
            "direction": direction,
            "pct_change": pct,
            "delta": entry["delta"],
            "verdict": verdict,
        })

    evaluated = len(calls) - failed
    decisive = evaluated - flat
    accuracy = round(correct / decisive, 4) if decisive > 0 else None

    return {
        "generated_at": now.isoformat(),
        "momentum_generated_at": data.get("generated_at"),
        "min_move_pct": MIN_MOVE_PCT,
        "evaluated": evaluated,
        "decisive": decisive,
        "correct": correct,
        "flat": flat,
        "failed": failed,
        "accuracy": accuracy,
        "tickers": results,
    }


def main() -> int:
    report = compute()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2))

    if "error" in report:
        print(f"signal_accuracy: ERROR — {report['error']}", file=sys.stderr)
        return 1

    acc = report["accuracy"]
    acc_str = f"{acc:.1%}" if acc is not None else "n/a"
    print(
        f"signal_accuracy: {report['correct']}/{report['decisive']} decisive hits "
        f"({acc_str}) | flat={report['flat']} no_data={report['failed']} "
        f"evaluated={report['evaluated']}"
    )
    hits = [t for t in report["tickers"] if t["verdict"] == "hit"]
    misses = [t for t in report["tickers"] if t["verdict"] == "miss"]
    for r in (hits + misses)[:5]:
        pct = r["pct_change"]
        pct_str = f"{pct:+.2f}%" if pct is not None else "n/a"
        print(
            f"  {r['verdict']:4s}  {r['ticker']:8s}  signal={r['direction']:7s}  "
            f"move={pct_str}  delta={r['delta']:+.3f}"
        )
    print(f"output={OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

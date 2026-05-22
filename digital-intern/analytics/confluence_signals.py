"""Multi-signal confluence detector.

Synthesises three independent analytics outputs into a unified ticker
confidence score.  A ticker that appears strongly in velocity, sentiment
momentum, AND source convergence is a "triple-confirmed" signal — far more
actionable than any single dimension alone.

Scoring (max 5 pts):
  Velocity  (+1 in top movers, +1 if ratio>=2.0)
  Sentiment (+1 bullish momentum, +1 if delta>=1.5)
  Convergence (+1 if 3+ distinct sources)

Output: /home/zeph/logs/confluence_signals.json
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path("/home/zeph/logs")
TV_PATH  = LOG_DIR / "trend_velocity.json"
SM_PATH  = LOG_DIR / "ticker_sentiment_momentum.json"
SC_PATH  = LOG_DIR / "source_convergence.json"
OUT_PATH = LOG_DIR / "confluence_signals.json"

VELOCITY_RATIO_BONUS = 2.0   # ratio >= this earns the second velocity point
SENTIMENT_DELTA_BONUS = 1.5  # delta >= this earns the second sentiment point


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def run() -> list[dict]:
    tv_data = _load(TV_PATH)
    sm_data = _load(SM_PATH)
    sc_data = _load(SC_PATH)

    scores: dict[str, dict] = {}

    def _entry(ticker: str) -> dict:
        if ticker not in scores:
            scores[ticker] = {"ticker": ticker, "score": 0, "signals": []}
        return scores[ticker]

    # --- velocity ---
    if tv_data:
        for item in tv_data.get("top", []):
            t = item["ticker"]
            e = _entry(t)
            e["score"] += 1
            e["signals"].append(f"velocity(delta={item['delta']:+d})")
            if item.get("ratio", 0) >= VELOCITY_RATIO_BONUS:
                e["score"] += 1
                e["signals"].append(f"velocity_ratio({item['ratio']:.1f}x)")
            e["velocity"] = item

    # --- sentiment momentum ---
    if sm_data:
        for item in sm_data.get("tickers", []):
            if item["direction"] != "bullish":
                continue
            t = item["ticker"]
            e = _entry(t)
            e["score"] += 1
            e["signals"].append(f"sentiment_bullish(delta={item['delta']:+.2f})")
            if item["delta"] >= SENTIMENT_DELTA_BONUS:
                e["score"] += 1
                e["signals"].append(f"sentiment_strong(delta={item['delta']:.2f})")
            e["sentiment"] = item

    # --- source convergence ---
    if sc_data:
        for item in sc_data.get("events", []):
            t = item["ticker"]
            e = _entry(t)
            e["score"] += 1
            e["signals"].append(
                f"convergence({item['distinct_sources']}src,ai={item['avg_ai_score']:.1f})"
            )
            e["convergence"] = item

    ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    ranked = [r for r in ranked if r["score"] >= 2]

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_ages": {
            "trend_velocity": tv_data.get("generated_at") if tv_data else None,
            "ticker_sentiment": sm_data.get("generated_at") if sm_data else None,
            "source_convergence": sc_data.get("generated_at") if sc_data else None,
        },
        "confluence_count": len(ranked),
        "events": ranked,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    return ranked


if __name__ == "__main__":
    results = run()
    print(f"confluence_signals: {len(results)} tickers with multi-signal confluence")
    for r in results:
        sigs = ", ".join(r["signals"])
        print(f"  {r['ticker']:<8} score={r['score']}/5  [{sigs}]")
    if not results:
        print("  (no tickers with >= 2 confluent signals this window)")

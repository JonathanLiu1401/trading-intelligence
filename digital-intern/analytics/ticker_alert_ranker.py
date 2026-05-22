"""Ticker alert ranker: composite confidence score across multiple signal sources.

Reads from existing analytics outputs and assigns each ticker a composite score
based on how many independent signals it appears in and how strongly:

  - trend_velocity.json     → mention velocity (ratio of now/prev window counts)
  - ticker_sentiment_momentum.json → ai_score delta between windows
  - source_convergence.json → multi-source coverage (distinct sources)
  - breaking_news.jsonl     → rapid burst articles (last 4h)

Each signal contributes 0–1 normalized points. Total composite score is 0–4.
Tickers scoring >= COMPOSITE_THRESHOLD with >= 2 signals are flagged as alerts.

Output: /home/zeph/logs/ticker_alert_rank.json
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

LOGS = Path("/home/zeph/logs")
OUT_PATH = LOGS / "ticker_alert_rank.json"

VELOCITY_PATH = LOGS / "trend_velocity.json"
MOMENTUM_PATH = LOGS / "ticker_sentiment_momentum.json"
CONVERGENCE_PATH = LOGS / "source_convergence.json"
BREAKING_PATH = LOGS / "breaking_news.jsonl"

COMPOSITE_THRESHOLD = 0.8  # minimum composite score to be included in alerts
MIN_SIGNALS = 2            # must appear in at least this many signal types
BREAKING_LOOKBACK_H = 4    # how many hours back to scan breaking_news.jsonl


def _load_json(path: Path) -> dict | list | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _velocity_scores() -> dict[str, float]:
    """Return per-ticker normalized velocity score (0–1) from trend_velocity.json."""
    data = _load_json(VELOCITY_PATH)
    if not data or not isinstance(data, dict):
        return {}
    top = data.get("top", [])
    if not top:
        return {}
    # ratio = now/prev articles; cap at 20x for normalization
    max_ratio = max((t.get("ratio", 1) for t in top), default=1)
    clamp = max(max_ratio, 5.0)  # ensure denominator >= 5
    scores: dict[str, float] = {}
    for entry in top:
        ticker = entry.get("ticker", "")
        if not ticker:
            continue
        ratio = entry.get("ratio", 1.0)
        # log-normalize: score = log(ratio) / log(clamp)
        score = min(1.0, math.log(max(ratio, 1)) / math.log(clamp))
        scores[ticker] = round(score, 4)
    return scores


def _momentum_scores() -> dict[str, float]:
    """Return per-ticker normalized sentiment momentum score (0–1)."""
    data = _load_json(MOMENTUM_PATH)
    if not data or not isinstance(data, dict):
        return {}
    tickers = data.get("tickers", [])
    if not tickers:
        return {}
    max_delta = max((abs(t.get("delta", 0)) for t in tickers), default=1)
    clamp = max(max_delta, 0.5)
    scores: dict[str, float] = {}
    for entry in tickers:
        ticker = entry.get("ticker", "")
        if not ticker:
            continue
        delta = abs(entry.get("delta", 0))
        scores[ticker] = round(min(1.0, delta / clamp), 4)
    return scores


def _convergence_scores() -> dict[str, float]:
    """Return per-ticker normalized multi-source convergence score (0–1)."""
    data = _load_json(CONVERGENCE_PATH)
    if not data or not isinstance(data, dict):
        return {}
    events = data.get("events", [])
    if not events:
        return {}
    max_sources = max((e.get("distinct_sources", 1) for e in events), default=1)
    clamp = max(max_sources, 3.0)
    scores: dict[str, float] = {}
    for entry in events:
        ticker = entry.get("ticker", "")
        if not ticker:
            continue
        n = entry.get("distinct_sources", 1)
        scores[ticker] = round(min(1.0, n / clamp), 4)
    return scores


def _breaking_scores() -> dict[str, float]:
    """Return per-ticker score from breaking_news.jsonl (0 or 1)."""
    if not BREAKING_PATH.exists():
        return {}
    cutoff = datetime.now(timezone.utc) - timedelta(hours=BREAKING_LOOKBACK_H)
    seen: set[str] = set()
    try:
        lines = BREAKING_PATH.read_text().splitlines()
    except Exception:
        return {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        raw_ts = obj.get("run_at", "")
        try:
            ts = datetime.fromisoformat(raw_ts)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
        if ts < cutoff:
            continue
        ticker = obj.get("ticker", "")
        if ticker:
            seen.add(ticker)
    return {t: 1.0 for t in seen}


def main() -> int:
    now = datetime.now(timezone.utc)

    vel = _velocity_scores()
    mom = _momentum_scores()
    conv = _convergence_scores()
    brk = _breaking_scores()

    all_tickers: set[str] = set(vel) | set(mom) | set(conv) | set(brk)

    results: list[dict] = []
    for ticker in sorted(all_tickers):
        v_score = vel.get(ticker, 0.0)
        m_score = mom.get(ticker, 0.0)
        c_score = conv.get(ticker, 0.0)
        b_score = brk.get(ticker, 0.0)

        signals_hit = sum(1 for s in (v_score, m_score, c_score, b_score) if s > 0)
        composite = round(v_score + m_score + c_score + b_score, 4)

        if composite < COMPOSITE_THRESHOLD or signals_hit < MIN_SIGNALS:
            continue

        results.append({
            "ticker": ticker,
            "composite": composite,
            "signals": signals_hit,
            "velocity": v_score,
            "momentum": m_score,
            "convergence": c_score,
            "breaking": b_score,
        })

    results.sort(key=lambda x: x["composite"], reverse=True)

    output = {
        "generated_at": now.isoformat(),
        "composite_threshold": COMPOSITE_THRESHOLD,
        "min_signals": MIN_SIGNALS,
        "total_alerts": len(results),
        "signal_counts": {
            "velocity_tickers": len(vel),
            "momentum_tickers": len(mom),
            "convergence_tickers": len(conv),
            "breaking_tickers": len(brk),
        },
        "alerts": results,
    }

    OUT_PATH.write_text(json.dumps(output, indent=2))

    print(f"ticker_alert_ranker: {len(results)} composite alerts")
    for r in results[:5]:
        sigs = []
        if r["velocity"] > 0:
            sigs.append(f"vel={r['velocity']:.2f}")
        if r["momentum"] > 0:
            sigs.append(f"mom={r['momentum']:.2f}")
        if r["convergence"] > 0:
            sigs.append(f"conv={r['convergence']:.2f}")
        if r["breaking"] > 0:
            sigs.append("BREAKING")
        print(f"  {ticker:<8}  composite={r['composite']:.3f}  signals={r['signals']}  [{', '.join(sigs)}]".replace(
            ticker, r["ticker"]
        ))

    return 0


if __name__ == "__main__":
    sys.exit(main())

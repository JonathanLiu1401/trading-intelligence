"""Multi-signal confluence consolidator.

Reads the JSON outputs of six independent detectors and ranks tickers by how
many distinct signal types they appear in.  A ticker that fires in 3+ signals
simultaneously has much stronger conviction than one triggering a single
detector in isolation.

Signal sources (all read-only, no DB access):
  * trend_velocity.json      — tickers with rising mention velocity (ratio >= 2)
  * ticker_anomaly.json      — z-score spikes vs 7-day baseline
  * strong_consensus.jsonl   — 5+ high-ai_score articles from 3+ sources
  * breaking_news.jsonl      — 3+ articles on same ticker in 5 min
  * sentiment_divergence.json — bull/bear conflict across sources
  * market_closed_watchlist.json — high-score articles since last market close

Each signal type carries a weight.  The final confluence_score is the sum of
weights for all signals the ticker appears in.  Ties broken by total raw
signal count.

Output: /home/zeph/logs/momentum_brief.json
Standalone: python3 -m analytics.signal_consolidator
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

LOG = Path("/home/zeph/logs")

# Weight per signal type — higher = stronger conviction signal
WEIGHTS = {
    "breaking_news":        3.0,   # real-time cluster
    "strong_consensus":     2.5,   # multi-source sustained high-score
    "ticker_anomaly":       2.0,   # statistically abnormal z-score spike
    "trend_velocity":       1.5,   # rising velocity vs prior 2h window
    "sentiment_divergence": 1.0,   # market disagreement (watch both sides)
    "market_watchlist":     0.5,   # accumulated since market close
}

# Only include signals from this many hours back to avoid stale data
MAX_AGE_HOURS: dict[str, int] = {
    "breaking_news":        2,
    "strong_consensus":     6,
    "ticker_anomaly":       2,
    "trend_velocity":       1,
    "sentiment_divergence": 2,
    "market_watchlist":     24,
}

OUT_PATH = LOG / "momentum_brief.json"
TOP_N = 10


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(s: str) -> datetime | None:
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S.%f+00:00", "%Y-%m-%dT%H:%M:%S+00:00"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _is_fresh(ts_str: str, max_hours: int) -> bool:
    ts = _parse_ts(ts_str)
    if ts is None:
        return True  # if unparseable, include it
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (_now() - ts) <= timedelta(hours=max_hours)


def _load_json(path: Path) -> dict | list | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _load_jsonl(path: Path) -> list[dict]:
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except FileNotFoundError:
        return []


def collect_signals() -> dict[str, dict]:
    """Return {ticker: {signal_name: True, ..., 'details': [...]}}."""
    now = _now()
    hits: dict[str, dict] = defaultdict(lambda: {"signals": {}, "details": []})

    # 1. trend_velocity.json
    tv = _load_json(LOG / "trend_velocity.json")
    if tv and _is_fresh(tv.get("generated_at", ""), MAX_AGE_HOURS["trend_velocity"]):
        for row in tv.get("top", []):
            tk = row.get("ticker", "")
            ratio = row.get("ratio", 0)
            if ratio >= 2.0 and tk:
                hits[tk]["signals"]["trend_velocity"] = True
                hits[tk]["details"].append(f"velocity ratio={ratio:.1f}x")

    # 2. ticker_anomaly.json
    ta = _load_json(LOG / "ticker_anomaly.json")
    if ta and _is_fresh(ta.get("generated_at", ""), MAX_AGE_HOURS["ticker_anomaly"]):
        for row in ta.get("top", []):
            tk = row.get("ticker", "")
            z = row.get("z_score", 0)
            if tk and z > 0:
                hits[tk]["signals"]["ticker_anomaly"] = True
                hits[tk]["details"].append(f"z={z:.1f} spike")

    # 3. strong_consensus.jsonl — last MAX_AGE_HOURS events
    cutoff_cs = now - timedelta(hours=MAX_AGE_HOURS["strong_consensus"])
    for row in _load_jsonl(LOG / "strong_consensus.jsonl"):
        ts = _parse_ts(row.get("detected_at", ""))
        if ts and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts and ts >= cutoff_cs:
            tk = row.get("ticker", "")
            cnt = row.get("count", 0)
            if tk:
                hits[tk]["signals"]["strong_consensus"] = True
                hits[tk]["details"].append(f"consensus {cnt} articles")

    # 4. breaking_news.jsonl
    cutoff_bn = now - timedelta(hours=MAX_AGE_HOURS["breaking_news"])
    for row in _load_jsonl(LOG / "breaking_news.jsonl"):
        ts = _parse_ts(row.get("run_at", ""))
        if ts and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts and ts >= cutoff_bn:
            tk = row.get("ticker", "")
            cnt = row.get("count", 0)
            if tk:
                hits[tk]["signals"]["breaking_news"] = True
                hits[tk]["details"].append(f"breaking {cnt}-art cluster")

    # 5. sentiment_divergence.json
    sd = _load_json(LOG / "sentiment_divergence.json")
    if sd and _is_fresh(sd.get("generated_at", ""), MAX_AGE_HOURS["sentiment_divergence"]):
        for row in sd.get("top", []):
            tk = row.get("ticker", "")
            score = row.get("divergence_score", 0)
            if tk:
                hits[tk]["signals"]["sentiment_divergence"] = True
                hits[tk]["details"].append(f"divergence_score={score:.1f}")

    # 6. market_closed_watchlist.json
    mw = _load_json(LOG / "market_closed_watchlist.json")
    if mw and _is_fresh(mw.get("generated_at", ""), MAX_AGE_HOURS["market_watchlist"]):
        for row in mw.get("tickers", []):
            tk = row.get("ticker", "")
            avg = row.get("avg_score", 0)
            if tk and avg >= 6.5:
                hits[tk]["signals"]["market_watchlist"] = True
                hits[tk]["details"].append(f"watchlist avg_score={avg:.1f}")

    return hits


def rank(hits: dict[str, dict]) -> list[dict]:
    results = []
    for tk, data in hits.items():
        sigs = data["signals"]
        score = sum(WEIGHTS[s] for s in sigs if s in WEIGHTS)
        results.append({
            "ticker": tk,
            "confluence_score": round(score, 2),
            "signal_count": len(sigs),
            "signals": sorted(sigs.keys()),
            "details": data["details"],
        })
    results.sort(key=lambda r: (-r["confluence_score"], -r["signal_count"]))
    return results[:TOP_N]


def main() -> int:
    hits = collect_signals()
    ranked = rank(hits)

    out = {
        "generated_at": _now().isoformat(),
        "signal_weights": WEIGHTS,
        "top": ranked,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))

    if ranked:
        print(f"Signal consolidator: {len(ranked)} tickers with multi-signal confluence")
        for r in ranked[:5]:
            sigs_str = "+".join(r["signals"])
            print(f"  {r['ticker']:6s}  score={r['confluence_score']:4.1f}  [{sigs_str}]  {r['details'][0] if r['details'] else ''}")
    else:
        print("Signal consolidator: no multi-signal tickers detected in current windows")
    print(f"Output: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

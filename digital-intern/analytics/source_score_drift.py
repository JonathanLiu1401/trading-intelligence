#!/usr/bin/env python3
"""Per-source ML score drift detector.

Extends the global score_drift_detector to show WHICH sources are dragging
scores up or down. Useful for diagnosing DRIFT-DOWN alerts from score_drift_detector.

Design:
  * One bounded idx_first_seen scan (SCAN_LIMIT rows)
  * Groups by normalized source, filters backtest/annotation rows
  * Maintains rolling 7d per-source hourly baselines in JSON state
  * Flags sources whose current-window avg deviates > SIGMA std from their baseline
  * Only considers sources with MIN_ARTICLES rows in the current window

Output:
  * /home/zeph/logs/source_score_drift.json   (machine-readable summary)
  * /home/zeph/logs/source_score_drift.log    (append-only trail)

Standalone: python3 -m analytics.source_score_drift
"""
from __future__ import annotations

import json
import os
import sqlite3
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

DB_PATH = str(Path(__file__).resolve().parents[1] / "data" / "articles.db")
LOG_DIR = Path("/home/zeph/logs")
STATE_PATH = LOG_DIR / ".source_score_drift_state.json"
OUT_PATH = LOG_DIR / "source_score_drift.json"
LOG_PATH = LOG_DIR / "source_score_drift.log"

SCAN_LIMIT = 8000
WINDOW_MINUTES = 60
MAX_HISTORY = 168          # 7 days of hourly points per source
MIN_BASELINE_POINTS = 6
MIN_ARTICLES = 3           # min articles from source in current window to evaluate
SIGMA = 1.5
ABS_FLOOR = 0.5

LIVE_ONLY = (
    "url NOT LIKE 'backtest://%' "
    "AND source NOT LIKE 'backtest_%' "
    "AND source NOT LIKE 'opus_annotation%'"
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(raw: str):
    if not raw:
        return None
    s = str(raw).replace("T", " ").split("+")[0].strip()[:26]
    try:
        return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def fetch_recent() -> dict[str, list[float]]:
    """Return {source: [ml_scores]} for live rows in the current window."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        rows = conn.execute(
            f"SELECT source, ml_score, first_seen FROM articles "
            f"WHERE {LIVE_ONLY} AND ml_score IS NOT NULL "
            f"ORDER BY first_seen DESC LIMIT ?",
            (SCAN_LIMIT,),
        ).fetchall()
    finally:
        conn.close()

    cutoff = _utcnow() - timedelta(minutes=WINDOW_MINUTES)
    per_source: dict[str, list[float]] = defaultdict(list)
    for source, ml_score, first_seen in rows:
        ts = _parse_ts(first_seen)
        if ts is not None and ts >= cutoff:
            per_source[source].append(float(ml_score))
    return dict(per_source)


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"sources": {}}


def save_state(state: dict) -> None:
    tmp = str(STATE_PATH) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh)
    os.replace(tmp, str(STATE_PATH))


def main() -> int:
    now = _utcnow()
    per_source = fetch_recent()
    state = load_state()
    sources_state = state.setdefault("sources", {})

    results = []
    for source, scores in per_source.items():
        if len(scores) < MIN_ARTICLES:
            continue
        cur_avg = statistics.fmean(scores)
        history = sources_state.get(source, {}).get("history", [])
        baseline_avgs = [p["avg"] for p in history]

        status = "BASELINE-BUILD"
        deviation = 0.0
        b_mean = b_std = band = 0.0

        if len(baseline_avgs) >= MIN_BASELINE_POINTS:
            b_mean = statistics.fmean(baseline_avgs)
            b_std = statistics.pstdev(baseline_avgs)
            band = max(SIGMA * b_std, ABS_FLOOR)
            deviation = cur_avg - b_mean
            if abs(deviation) > band:
                status = "DRIFT-UP" if deviation > 0 else "DRIFT-DOWN"
            else:
                status = "OK"

        history.append({"ts": now.isoformat(), "avg": round(cur_avg, 6), "n": len(scores)})
        sources_state[source] = {"history": history[-MAX_HISTORY:]}

        results.append({
            "source": source,
            "status": status,
            "cur_avg": round(cur_avg, 4),
            "n": len(scores),
            "b_mean": round(b_mean, 4),
            "deviation": round(deviation, 4),
            "band": round(band, 4),
        })

    save_state(state)

    # Sort: drifters first, then by absolute deviation descending
    results.sort(key=lambda r: (r["status"] == "OK", r["status"] == "BASELINE-BUILD", -abs(r["deviation"])))

    drifters_down = [r for r in results if r["status"] == "DRIFT-DOWN"]
    drifters_up = [r for r in results if r["status"] == "DRIFT-UP"]

    summary = {
        "ts": now.isoformat(),
        "sources_evaluated": len(results),
        "drift_down": len(drifters_down),
        "drift_up": len(drifters_up),
        "top_drifters": results[:10],
    }
    OUT_PATH.write_text(json.dumps(summary, indent=2))

    log_line = (
        f"{now.isoformat()} | evaluated={len(results)} "
        f"drift_down={len(drifters_down)} drift_up={len(drifters_up)}"
    )
    with open(LOG_PATH, "a") as fh:
        fh.write(log_line + "\n")

    total = len(results)
    print(f"source_score_drift: evaluated={total} drift_down={len(drifters_down)} drift_up={len(drifters_up)}")
    for r in results[:8]:
        tag = f"[{r['status']}]" if r["status"] != "OK" else ""
        print(f"  {r['source'][:40]:<40} avg={r['cur_avg']:.3f} n={r['n']:>3} {tag}")
    if drifters_down:
        print(f"\nDRIFT-DOWN sources: {', '.join(r['source'] for r in drifters_down)}")
    if drifters_up:
        print(f"DRIFT-UP sources: {', '.join(r['source'] for r in drifters_up)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""ML model confidence distribution tracker.

Measures how *confident* the ArticleNet scoring model is right now vs its
rolling 24-hour baseline.  Confidence is defined by where ml_score falls:

  confident zone  : ml_score < 0.2  OR  ml_score > 0.8   (clear signal)
  uncertain zone  : 0.3 <= ml_score <= 0.7               (model unsure)

confidence_ratio = confident_count / (confident_count + uncertain_count)

A healthy system operating on in-distribution financial news typically
scores >60% of articles confidently.  A ratio drop below WARN_THRESHOLD
(default 0.40) suggests the model is seeing an unusual content mix — newly
viral topics, a surge of short noisy articles, or an OOV vocabulary spike.
This fires *before* score_drift_detector, which only triggers when the mean
shifts; a bimodal distribution collapsing to 0.5 leaves the mean unchanged
but tanks confidence_ratio.

Design constraints:
  * No full-table COUNT(*) — single bounded idx_first_seen scan.
  * Read-only sqlite URI, busy_timeout 10 000 ms (USB-backed DB).
  * _LIVE_ONLY_CLAUSE applied so backtest rows never pollute the signal.
  * State file persists 24 hourly snapshots for trend comparison.

Artifacts:
  * /home/zeph/logs/ml_confidence.json     — latest snapshot + trend
  * /home/zeph/logs/.ml_confidence_state.json — rolling 24h history

Standalone:  python3 -m analytics.ml_confidence_tracker
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, stdev

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

try:
    from storage.article_store import _LIVE_ONLY_CLAUSE, _get_db_path
    DB_PATH = _get_db_path()
except Exception:
    _LIVE_ONLY_CLAUSE = "source NOT LIKE 'backtest_run_%'"
    DB_PATH = BASE / "data" / "articles.db"

OUT_PATH     = Path("/home/zeph/logs/ml_confidence.json")
STATE_PATH   = Path("/home/zeph/logs/.ml_confidence_state.json")

SCAN_LIMIT      = 6000    # covers ~12h of live rows via idx_first_seen
WINDOW_MINUTES  = 60      # "current" 1h window
MAX_HISTORY     = 24      # keep 24 hourly points
WARN_THRESHOLD  = 0.40    # confidence_ratio below this → warn
ALERT_THRESHOLD = 0.25    # confidence_ratio below this → alert

# Score boundary constants
CONFIDENT_HIGH  = 0.80
CONFIDENT_LOW   = 0.20
UNCERTAIN_LOW   = 0.30
UNCERTAIN_HIGH  = 0.70


def _parse_ts(raw: str) -> datetime | None:
    if not raw:
        return None
    s = str(raw).replace("T", " ").split("+")[0].strip()[:19]
    try:
        dt = datetime.fromisoformat(s)
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _confidence_bucket(score: float) -> str:
    if score < CONFIDENT_LOW or score > CONFIDENT_HIGH:
        return "confident"
    if UNCERTAIN_LOW <= score <= UNCERTAIN_HIGH:
        return "uncertain"
    return "transition"


def _load_state() -> list[dict]:
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:
        return []


def _save_state(history: list[dict]) -> None:
    # prune to MAX_HISTORY most recent points
    history = history[-MAX_HISTORY:]
    STATE_PATH.write_text(json.dumps(history, indent=2))


def main() -> int:
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(minutes=WINDOW_MINUTES)

    try:
        conn = sqlite3.connect(
            f"file:{DB_PATH}?mode=ro", uri=True, timeout=10,
            check_same_thread=False,
        )
        conn.execute("PRAGMA busy_timeout=10000")
        rows = conn.execute(
            f"""
            SELECT first_seen, ml_score
            FROM   articles
            WHERE  ml_score IS NOT NULL
              AND  {_LIVE_ONLY_CLAUSE}
            ORDER BY first_seen DESC
            LIMIT  {SCAN_LIMIT}
            """,
        ).fetchall()
        conn.close()
    except Exception as exc:
        print(f"ml_confidence_tracker: db error — {exc}", file=sys.stderr)
        return 1

    # bucket by hour within the scan window
    hour_buckets: dict[str, dict] = defaultdict(lambda: {
        "confident": 0, "uncertain": 0, "transition": 0, "total": 0
    })
    current: dict = {"confident": 0, "uncertain": 0, "transition": 0, "total": 0}

    for raw_ts, score in rows:
        ts = _parse_ts(str(raw_ts))
        if ts is None:
            continue
        if ts < now - timedelta(hours=MAX_HISTORY):
            continue
        bucket = _confidence_bucket(float(score))
        hour_key = ts.strftime("%Y-%m-%dT%H:00Z")
        hour_buckets[hour_key][bucket] += 1
        hour_buckets[hour_key]["total"] += 1
        if ts >= window_start:
            current[bucket] += 1
            current["total"] += 1

    # current confidence ratio
    signal_count = current["confident"] + current["uncertain"]
    current_ratio = (
        round(current["confident"] / signal_count, 4) if signal_count >= 5 else None
    )

    # per-hour confidence ratios for trend
    hourly_ratios = []
    for h_key in sorted(hour_buckets):
        hb = hour_buckets[h_key]
        s = hb["confident"] + hb["uncertain"]
        ratio = round(hb["confident"] / s, 4) if s >= 3 else None
        hourly_ratios.append({"hour": h_key, "ratio": ratio,
                               "confident": hb["confident"],
                               "uncertain": hb["uncertain"],
                               "total": hb["total"]})

    # load history + append current point for trend
    history = _load_state()
    cur_point = {
        "ts": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ratio": current_ratio,
        "confident": current["confident"],
        "uncertain": current["uncertain"],
        "total": current["total"],
    }
    history.append(cur_point)
    _save_state(history)

    # 24h trend (skip None entries)
    valid_ratios = [p["ratio"] for p in history if p["ratio"] is not None]
    trend_mean   = round(mean(valid_ratios), 4) if len(valid_ratios) >= 2 else None
    trend_std    = round(stdev(valid_ratios), 4) if len(valid_ratios) >= 3 else None
    trend_delta  = (
        round(current_ratio - trend_mean, 4)
        if (current_ratio is not None and trend_mean is not None)
        else None
    )

    # status
    if current_ratio is None:
        status = "insufficient_data"
    elif current_ratio < ALERT_THRESHOLD:
        status = "alert"
    elif current_ratio < WARN_THRESHOLD:
        status = "warn"
    else:
        status = "ok"

    snapshot = {
        "generated_at": now.isoformat(),
        "window_minutes": WINDOW_MINUTES,
        "current": {
            "confident": current["confident"],
            "uncertain": current["uncertain"],
            "transition": current["transition"],
            "total": current["total"],
            "confidence_ratio": current_ratio,
            "status": status,
        },
        "trend_24h": {
            "mean_ratio": trend_mean,
            "std_ratio": trend_std,
            "delta_from_mean": trend_delta,
            "n_points": len(valid_ratios),
        },
        "thresholds": {
            "warn": WARN_THRESHOLD,
            "alert": ALERT_THRESHOLD,
        },
        "hourly_breakdown": hourly_ratios[-24:],
    }

    OUT_PATH.write_text(json.dumps(snapshot, indent=2))

    ratio_str = f"{current_ratio:.2%}" if current_ratio is not None else "n/a"
    delta_str = (
        f"  Δ from 24h mean: {trend_delta:+.4f}" if trend_delta is not None else ""
    )
    print(
        f"ml_confidence_tracker: status={status}  "
        f"ratio={ratio_str}  "
        f"confident={current['confident']}  "
        f"uncertain={current['uncertain']}  "
        f"total={current['total']}"
        f"{delta_str}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

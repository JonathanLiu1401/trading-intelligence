"""Source burst detector: alert when a collector publishes articles at >3x its normal rate.

For every live collector family, compares the article count in the last 1 hour
against the trailing 24h baseline (24h_count / 24). A collector with count_1h >
BURST_MULTIPLIER × baseline_hourly_rate AND count_1h >= MIN_BURST_COUNT is
flagged as bursting — likely covering breaking news or an exclusive story.

This is the inverse of ``stale_source_alerter``: where that flags silence,
this flags unexpected volume surges that often precede urgency spikes.

Output: /home/zeph/logs/source_burst.json

Standalone: python3 -m analytics.source_burst_detector
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.logger import get_logger
from storage.article_store import _LIVE_ONLY_CLAUSE, _get_db_path

log = get_logger("source_burst_detector")

BURST_MULTIPLIER = 3.0   # burst if count_1h > 3× baseline hourly rate
MIN_BURST_COUNT  = 5     # ignore bursts below this (sparse sources are noisy)
MIN_BASELINE     = 2.0   # skip sources averaging <2 articles/h (too sparse for ratio)

SNAPSHOT_PATH = Path("/home/zeph/logs/source_burst.json")

_TS_FMT = "%Y-%m-%d %H:%M:%S"


def _collector_of(source: str) -> str:
    return source.split("/", 1)[0] if "/" in source else source


def compute_bursts(now: datetime | None = None) -> dict:
    """Return the full burst report.

    Shape::

        {
          "generated_at": "<iso utc>",
          "burst_multiplier": 3.0,
          "min_burst_count": 5,
          "burst_count": <int>,
          "burst_collectors": ["coll", ...],
          "collectors": {
            "coll": {
              "count_1h": <int>,
              "count_24h": <int>,
              "baseline_hourly": <float>,
              "burst_ratio": <float>,
              "is_burst": <bool>
            }, ...
          }
        }
    """
    now = now or datetime.now(timezone.utc)
    cutoff_1h  = (now - timedelta(hours=1)).strftime(_TS_FMT)
    cutoff_24h = (now - timedelta(hours=24)).strftime(_TS_FMT)

    db_path = _get_db_path()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            f"""
            SELECT source,
                   SUM(CASE WHEN substr(replace(first_seen,'T',' '),1,19) >= ?
                            THEN 1 ELSE 0 END) AS count_1h,
                   SUM(CASE WHEN substr(replace(first_seen,'T',' '),1,19) >= ?
                            THEN 1 ELSE 0 END) AS count_24h
            FROM articles
            WHERE first_seen >= datetime('now', '-1 days')
              AND {_LIVE_ONLY_CLAUSE}
              AND source IS NOT NULL
              AND source != ''
            GROUP BY source
            """,
            (cutoff_1h, cutoff_24h),
        ).fetchall()
    finally:
        conn.close()

    # Roll up to collector family
    coll_acc: dict[str, dict] = {}
    for source, count_1h, count_24h in rows:
        key = _collector_of(source)
        acc = coll_acc.setdefault(key, {"count_1h": 0, "count_24h": 0})
        acc["count_1h"]  += int(count_1h  or 0)
        acc["count_24h"] += int(count_24h or 0)

    collectors: dict[str, dict] = {}
    burst_collectors: list[str] = []

    for key, acc in coll_acc.items():
        c1h  = acc["count_1h"]
        c24h = acc["count_24h"]
        # Baseline is the 23h trailing average (exclude the current burst hour
        # to avoid circular comparison). Use max(c24h - c1h, 0) for trailing
        # 23h count, divide by 23 to get the average hourly rate when quiet.
        trailing_23h = max(c24h - c1h, 0)
        baseline_h   = trailing_23h / 23.0
        burst_ratio  = (c1h / baseline_h) if baseline_h >= MIN_BASELINE else 0.0
        is_burst     = (
            burst_ratio >= BURST_MULTIPLIER
            and c1h >= MIN_BURST_COUNT
            and baseline_h >= MIN_BASELINE
        )
        collectors[key] = {
            "count_1h":       c1h,
            "count_24h":      c24h,
            "baseline_hourly": round(baseline_h, 2),
            "burst_ratio":     round(burst_ratio, 2),
            "is_burst":        is_burst,
        }
        if is_burst:
            burst_collectors.append(key)

    burst_collectors.sort(
        key=lambda k: collectors[k]["burst_ratio"], reverse=True
    )

    return {
        "generated_at":     now.isoformat(),
        "burst_multiplier": BURST_MULTIPLIER,
        "min_burst_count":  MIN_BURST_COUNT,
        "burst_count":      len(burst_collectors),
        "burst_collectors": burst_collectors,
        "collectors":       collectors,
    }


def write_snapshot(report: dict, path: Path = SNAPSHOT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path


def alert_bursts(report: dict) -> None:
    for coll in report["burst_collectors"]:
        info = report["collectors"][coll]
        log.warning(
            "SOURCE BURST %s: count_1h=%d vs baseline_hourly=%.1f (ratio=%.1fx)",
            coll, info["count_1h"], info["baseline_hourly"], info["burst_ratio"],
        )


def main() -> int:
    report = compute_bursts()
    snap_path = write_snapshot(report)

    print(
        f"source-burst detector: {report['burst_count']} bursting collectors "
        f"(>{BURST_MULTIPLIER}× baseline, >={MIN_BURST_COUNT}/h); "
        f"snapshot={snap_path}"
    )
    for coll in report["burst_collectors"]:
        info = report["collectors"][coll]
        print(
            f"BURST {coll}: {info['count_1h']}/h vs baseline "
            f"{info['baseline_hourly']:.1f}/h (ratio={info['burst_ratio']:.1f}×)"
        )
    if not report["burst_collectors"]:
        # Show top-3 by count_1h for context
        top3 = sorted(
            report["collectors"].items(),
            key=lambda kv: kv[1]["count_1h"], reverse=True
        )[:3]
        for coll, info in top3:
            print(
                f"  {coll}: {info['count_1h']}/h (ratio={info['burst_ratio']:.1f}×, "
                f"baseline={info['baseline_hourly']:.1f}/h)"
            )

    log.info(
        "burst snapshot written: %d bursting collectors -> %s",
        report["burst_count"], snap_path,
    )
    alert_bursts(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())

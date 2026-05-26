"""Timestamp anomaly auditor: catches future-dated articles that slip past publish_lag_audit.

``publish_lag_audit`` discards any row where ``published > first_seen + 1h`` (MIN_LAG_S=-3600),
silently hiding a pervasive data-quality bug: Google News collectors store articles with
``published`` timestamps *ahead* of ``first_seen``, which corrupts recency-decay scoring
and urgency calibration.

This module surfaces:
  * ``future_dated_pct``  — % of scanned live articles where published > first_seen
  * Per-source breakdown: count, median forward bias (minutes), max forward bias (hours)
  * ``worst_offenders``   — top sources ranked by median forward bias descending
  * Whether the issue is structural (>5% of source's articles) or marginal noise

Distinct from existing analytics:
  * ``publish_lag_audit``  — measures staleness lag; silently discards future-dated rows
  * ``alert_freshness``    — looks at urgency>=2 alerts only, not raw ingest
  * ``source_quality``     — measures score quality, not timestamp integrity

Output: /home/zeph/logs/timestamp_anomaly.json
Standalone: python3 -m analytics.timestamp_anomaly_auditor
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from statistics import median

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from storage.article_store import _LIVE_ONLY_CLAUSE, _get_db_path  # noqa: E402

OUT_PATH = Path("/home/zeph/logs/timestamp_anomaly.json")

# Bounded scan — never full-table
SCAN_LIMIT = 12_000

# Forward-bias threshold: articles with published > first_seen + GRACE_S are "anomalous"
# 5 minutes of grace for minor clock skew / timezone rounding
GRACE_S = 5 * 60

# A source is a "structural offender" if >=STRUCTURAL_PCT_THRESHOLD% of its articles
# have forward-biased timestamps
STRUCTURAL_PCT_THRESHOLD = 10.0


def _parse_ts(value: str | None) -> datetime | None:
    """Parse RFC822 or ISO timestamp to aware UTC. Returns None on failure."""
    if not value:
        return None
    # RFC822 (used by RSS/Atom feeds and Google News)
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            return None
        return dt.astimezone(timezone.utc)
    except Exception:
        pass
    # ISO-8601 fallback
    try:
        s = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            return None
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _parse_first_seen(value: str | None) -> datetime | None:
    """Parse first_seen (stored as 'YYYY-MM-DD HH:MM:SS' or ISO) to aware UTC."""
    if not value:
        return None
    try:
        s = value.replace("T", " ").rstrip("Z")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def compute(scan_limit: int = SCAN_LIMIT) -> dict:
    """Scan recent live articles and report timestamp anomaly stats."""
    db_path = _get_db_path()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=15)
    conn.execute("PRAGMA busy_timeout=8000")

    try:
        rows = conn.execute(
            f"""
            SELECT source, published, first_seen
              FROM articles
             WHERE {_LIVE_ONLY_CLAUSE}
               AND published IS NOT NULL AND published != ''
               AND first_seen IS NOT NULL
             ORDER BY first_seen DESC
             LIMIT {scan_limit}
            """
        ).fetchall()
    finally:
        conn.close()

    # Per-source accumulators
    source_total: dict[str, int] = defaultdict(int)
    source_anomalous: dict[str, int] = defaultdict(int)
    # Forward biases in seconds (positive = published is ahead of first_seen)
    source_forward_bias_s: dict[str, list[float]] = defaultdict(list)

    total_parseable = 0
    total_anomalous = 0

    for source, published, first_seen in rows:
        if not source:
            continue
        pub = _parse_ts(published)
        seen = _parse_first_seen(first_seen)
        if pub is None or seen is None:
            continue

        total_parseable += 1
        source_total[source] += 1

        # forward_bias_s > 0 means published is AHEAD of first_seen (anomalous)
        forward_bias_s = (pub - seen).total_seconds()

        if forward_bias_s > GRACE_S:
            total_anomalous += 1
            source_anomalous[source] += 1
            source_forward_bias_s[source].append(forward_bias_s)

    # Build per-source stats
    source_stats: list[dict] = []
    for src in source_anomalous:
        biases = source_forward_bias_s[src]
        total_src = source_total[src]
        anomalous_src = source_anomalous[src]
        pct = round(100.0 * anomalous_src / total_src, 1) if total_src else 0.0
        med_bias_min = round(median(biases) / 60.0, 1)
        max_bias_hr = round(max(biases) / 3600.0, 2)
        source_stats.append({
            "source": src,
            "total": total_src,
            "anomalous": anomalous_src,
            "anomaly_pct": pct,
            "median_forward_bias_min": med_bias_min,
            "max_forward_bias_hr": max_bias_hr,
            "structural": pct >= STRUCTURAL_PCT_THRESHOLD,
        })

    source_stats.sort(key=lambda x: x["median_forward_bias_min"], reverse=True)

    structural_offenders = [s for s in source_stats if s["structural"]]
    future_pct = round(100.0 * total_anomalous / total_parseable, 1) if total_parseable else 0.0

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scanned": total_parseable,
        "future_dated_count": total_anomalous,
        "future_dated_pct": future_pct,
        "structural_offenders_count": len(structural_offenders),
        "grace_s": GRACE_S,
        "structural_threshold_pct": STRUCTURAL_PCT_THRESHOLD,
        "worst_offenders": source_stats[:15],
        "structural_offenders": structural_offenders,
    }

    return report


def write_snapshot(report: dict, path: Path = OUT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path


def main() -> None:
    report = compute()
    out = write_snapshot(report)
    pct = report["future_dated_pct"]
    structural = report["structural_offenders_count"]
    count = report["future_dated_count"]
    print(
        f"timestamp-anomaly: scanned={report['scanned']} "
        f"future_dated={count} ({pct}%) "
        f"structural_offenders={structural} -> {out}"
    )
    if report["worst_offenders"]:
        for s in report["worst_offenders"][:5]:
            flag = " [STRUCTURAL]" if s["structural"] else ""
            print(
                f"  {s['source']}: {s['anomalous']}/{s['total']} "
                f"({s['anomaly_pct']}%) "
                f"median_bias={s['median_forward_bias_min']}m "
                f"max={s['max_forward_bias_hr']}h{flag}"
            )


if __name__ == "__main__":
    main()

"""Publish-lag audit: per-collector latency from ``published`` to ``first_seen``.

``stale_source_alerter`` answers "is this collector still ingesting at all?".
This module answers a different, complementary question: *when* a collector
ingests, how soon after publication does it see the article?

A collector with a healthy ingest rate may still be feeding ArticleNet stale
news — e.g. a 30-min RSS poll seeing publisher-dated items from 6 hours ago.
Stale items are still scored, still considered for urgency, and still appear
in the briefing's recency-decay ranker, where ``time_sensitivity`` weights
them against the wall clock. Knowing per-collector publication latency lets
the operator tell "first-mover" feeds (Nitter, SEC EDGAR, Finnhub) from
"digest" feeds (rolled-up wires, low-frequency RSS) without staring at the
DB.

Output shape — JSON snapshot at ``SNAPSHOT_PATH`` and the ``compute()``
return value::

    {
      "generated_at": "<iso utc>",
      "scan_limit": 5000,
      "scanned": <int>,                    # rows pulled
      "rows_with_parseable_lag": <int>,    # rows that contributed to stats
      "collectors": {
         "<collector>": {
            "n": <int>,
            "median_lag_min": <float|null>,
            "p90_lag_min": <float|null>,
            "mean_lag_min": <float>,
            "fresh_5m_pct": <0..100>,      # share of items with lag<5m
            "stale_60m_pct": <0..100>,     # share with lag>60m
         },
         ...
      },
      "ranked_freshest": [...],   # collectors sorted by median_lag_min asc
      "ranked_stalest":  [...]    # collectors sorted by median_lag_min desc
    }

Read-only sqlite (``mode=ro``) — never takes a write lock on the production
DB, never blocks the daemon's writers. The scan is bounded by
``SCAN_LIMIT`` (a recent-id slice, not a full-table COUNT) so it stays fast
against the ~1.4GB USB-backed DB.

Standalone:   ``python3 -m analytics.publish_lag_audit``
Importable:   ``from analytics.publish_lag_audit import compute``
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Optional

from storage.article_store import _LIVE_ONLY_CLAUSE, _get_db_path

# Scan the most recent N rows by id. Identical bound to scorer_skew /
# attribution audits, sized for sub-second reads even on the slow USB DB.
SCAN_LIMIT = 5000

# A collector needs at least this many parseable-lag samples to be reported.
# Below this, medians are dominated by noise (and a single oddly-stamped row
# could rank a collector at the top or bottom).
MIN_PER_COLLECTOR = 5

# Snapshot lives alongside the operator's other advisory artifacts. Matches
# stale_source_alerter.SNAPSHOT_PATH's containing directory deliberately.
SNAPSHOT_PATH = Path("/home/zeph/logs/publish_lag.json")

# Lag-bucket thresholds (seconds).
_FRESH_BUCKET_S = 5 * 60
_STALE_BUCKET_S = 60 * 60

# Reject samples with absurd lag values. Clock skew of a few hours is real,
# but a `published` parsed as e.g. 1970-01-01 or 2099 is publisher garbage
# and would dominate any mean. Caps chosen wide enough to keep all realistic
# delays (a once-a-day digest is ~24h) but tight enough to discard junk.
_MIN_LAG_S = -3600          # tolerate up to 1h of clock skew into the future
_MAX_LAG_S = 30 * 86400      # 30 days; anything older is a stale rewrite / parse error


def _parse_published(value: str) -> Optional[datetime]:
    """Parse a ``published`` column to aware UTC. Returns None on failure.

    Mirrors ``storage.article_store._published_older_than``'s dual-form
    parsing (RFC822 first, ISO fallback) so the lag measurement uses the
    same notion of publication time the briefing's staleness filter uses.
    """
    if not value:
        return None
    dt = None
    try:
        dt = parsedate_to_datetime(value)
    except Exception:
        dt = None
    if dt is None:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_first_seen(value: str) -> Optional[datetime]:
    """Parse a ``first_seen`` value. These are written by the daemon as
    ISO8601-with-tz, but defensively fall back to RFC822 if a producer ever
    deviates."""
    return _parse_published(value)


def _collector_of(source: str) -> str:
    """Collapse a granular ``source`` tag to its collector family.

    Matches ``stale_source_alerter._collector_of`` exactly so the two
    audits aggregate at the same granularity and can be cross-read.
    """
    return source.split("/", 1)[0] if "/" in source else source


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0,1]. Caller guarantees
    non-empty input."""
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _bucket(rows: Iterable[tuple[str, str, str]]) -> tuple[dict[str, list[float]], int]:
    """Group rows into ``{collector: [lag_seconds, ...]}`` and return the
    parseable-sample count. Rejects rows with unparseable ``published`` or
    out-of-range lags."""
    buckets: dict[str, list[float]] = {}
    parsed = 0
    for source, published, first_seen in rows:
        if not source:
            continue
        p = _parse_published(published)
        if p is None:
            continue
        f = _parse_first_seen(first_seen)
        if f is None:
            continue
        lag_s = (f - p).total_seconds()
        if lag_s < _MIN_LAG_S or lag_s > _MAX_LAG_S:
            continue
        buckets.setdefault(_collector_of(source), []).append(lag_s)
        parsed += 1
    return buckets, parsed


def _summarise(lags_s: list[float]) -> dict:
    n = len(lags_s)
    fresh = sum(1 for x in lags_s if x < _FRESH_BUCKET_S)
    stale = sum(1 for x in lags_s if x > _STALE_BUCKET_S)
    return {
        "n": n,
        "median_lag_min": round(median(lags_s) / 60.0, 2),
        "p90_lag_min": round(_percentile(lags_s, 0.90) / 60.0, 2),
        "mean_lag_min": round(mean(lags_s) / 60.0, 2),
        "fresh_5m_pct": round(100.0 * fresh / n, 1),
        "stale_60m_pct": round(100.0 * stale / n, 1),
    }


def compute(now: Optional[datetime] = None, scan_limit: int = SCAN_LIMIT) -> dict:
    """Build the publish-lag report. ``now`` is recorded as ``generated_at``;
    it does not affect any computation (lag is published→first_seen, both
    intrinsic to the row)."""
    now = now or datetime.now(timezone.utc)
    db_path = _get_db_path()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=15)
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        rows = conn.execute(
            f"""
            SELECT source, published, first_seen
              FROM articles
             WHERE id IN (SELECT id FROM articles ORDER BY id DESC LIMIT ?)
               AND {_LIVE_ONLY_CLAUSE}
               AND published IS NOT NULL AND published != ''
            """,
            (scan_limit,),
        ).fetchall()
    finally:
        conn.close()

    buckets, parsed = _bucket(rows)

    collectors: dict[str, dict] = {}
    for coll, lags in buckets.items():
        if len(lags) < MIN_PER_COLLECTOR:
            continue
        collectors[coll] = _summarise(lags)

    ranked = sorted(collectors.items(), key=lambda kv: kv[1]["median_lag_min"])
    ranked_freshest = [{"collector": k, **v} for k, v in ranked]
    ranked_stalest = list(reversed(ranked_freshest))

    return {
        "generated_at": now.isoformat(),
        "scan_limit": scan_limit,
        "scanned": len(rows),
        "rows_with_parseable_lag": parsed,
        "collectors": collectors,
        "ranked_freshest": ranked_freshest,
        "ranked_stalest": ranked_stalest,
    }


def write_snapshot(report: dict, path: Path = SNAPSHOT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path


def main() -> int:
    report = compute()
    snap = write_snapshot(report)
    print(
        f"publish-lag audit: {report['rows_with_parseable_lag']}/"
        f"{report['scanned']} parseable, "
        f"{len(report['collectors'])} collectors reported; snapshot={snap}"
    )
    for entry in report["ranked_freshest"][:5]:
        print(
            f"  FRESH {entry['collector'][:32]:<32} "
            f"n={entry['n']:>4}  median={entry['median_lag_min']:>7.1f}m  "
            f"p90={entry['p90_lag_min']:>7.1f}m  "
            f"<5m={entry['fresh_5m_pct']:>5.1f}%"
        )
    for entry in report["ranked_stalest"][:5]:
        print(
            f"  STALE {entry['collector'][:32]:<32} "
            f"n={entry['n']:>4}  median={entry['median_lag_min']:>7.1f}m  "
            f"p90={entry['p90_lag_min']:>7.1f}m  "
            f">60m={entry['stale_60m_pct']:>5.1f}%"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

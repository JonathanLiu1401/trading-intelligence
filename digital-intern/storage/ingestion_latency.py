"""storage/ingestion_latency.py — read-only per-source ingestion latency monitor.

Why this exists (news-analyst lens): a collector worker can be *alive*
(``[source_health]`` happy, ``db_health.ingestion_by_source`` ticking) yet
still be *slow* — surfacing items long after their upstream ``published``
timestamp. A chronically slow source has already lost the alpha window by
the time ``alert_agent`` sees the row, even though nothing in the daemon
flags it.

This module measures, per source over a recency window, the distribution of
``first_seen − published`` (seconds). It is the dual of ``db_health``:

  * ``db_health`` — *did news flow*, *was the WAL contended*, *was a batch
    silently dropped*. Volume + contention monitor.
  * ``ingestion_latency`` — *how stale was the news when it arrived*. Quality
    monitor.

Pure function ``compute_latency_stats((src, published, first_seen)…)`` is the
unit-tested contract — all clock parsing, clamping, and percentile maths live
there, so the DB shell is trivial and untestable parts are confined to I/O.

Load-bearing invariants respected:

  * Backtest isolation: the SQL pull carries the canonical ``_LIVE_ONLY_CLAUSE``
    verbatim (mirror of ``storage/article_store.py``; pinned by test so a drift
    in the SSOT fails CI). Synthetic ``backtest://`` rows and
    ``backtest_*`` / ``opus_annotation*`` sources can never colour a real
    collector's latency stats.
  * Read-only: the DB is opened in ``mode=ro`` and no file is written. Cannot
    add to writer contention; cannot affect ``ai_score`` / ``ml_score`` /
    ``score_source`` semantics.

CLI: ``python3 -m storage.ingestion_latency [hours]`` prints a JSON report.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Iterable

# ── Canonical backtest-isolation clause ─────────────────────────────────────
# Copied verbatim from storage/article_store.py::_LIVE_ONLY_CLAUSE. Duplicated
# (not imported) on purpose: this module must keep working even while
# article_store.py is being rewritten by a concurrent process. The test suite
# pins this string so a drift from the canonical fragment fails CI loudly —
# same discipline as storage/db_health.py.
LIVE_ONLY_CLAUSE = (
    "url NOT LIKE 'backtest://%' "
    "AND source NOT LIKE 'backtest_%' "
    "AND source NOT LIKE 'opus_annotation%'"
)

_USB_PATH = Path(os.environ.get("DIGITAL_INTERN_USB", "/media/zeph/projects/digital-intern/db"))
_LOCAL_PATH = Path(__file__).resolve().parent.parent / "data"

# A latency that exceeds this is implausible (the article was almost certainly
# back-filled from an archive, not freshly published). Such rows are counted
# under ``skipped_implausible`` rather than dragging the p90/max upwards.
# 7 days is generous: SEC EDGAR full-text reposts of older filings, Wikipedia
# revision sweeps of stale pages, etc., still fall outside.
_MAX_PLAUSIBLE_SEC = 7 * 24 * 3600.0


def _now(now: datetime | None) -> datetime:
    return (now or datetime.now(timezone.utc)).astimezone(timezone.utc)


def resolve_db_path() -> Path:
    """Resolve live ``articles.db`` (USB-preferred), no side effects.

    Mirrors ``storage.db_health.resolve_db_path`` exactly so this monitor
    reads the same DB the daemon writes, but never calls ``mkdir`` — a
    read-only observer must not materialise an empty fallback directory.
    """
    usb_db = _USB_PATH / "articles.db"
    if _USB_PATH.exists() and (usb_db.exists() or _USB_PATH.is_mount()):
        return usb_db
    return _LOCAL_PATH / "articles.db"


def open_ro(path: Path) -> sqlite3.Connection:
    """Open ``path`` strictly read-only. Raises sqlite3.OperationalError if absent."""
    uri = f"file:{path}?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=5.0)


def parse_published(value: str | None) -> datetime | None:
    """Best-effort parse of an article ``published`` field.

    The ``published`` column is heterogeneous: RSS collectors emit RFC 2822
    (``Mon, 18 May 2026 11:30:00 GMT``); GDELT/SEC/Polygon emit ISO-8601
    variants; some collectors leave it blank or copy a non-timestamp slug.
    Returns ``None`` for anything that doesn't yield an aware UTC datetime —
    those rows are surfaced under ``skipped_no_published`` in the report so
    the operator can see which sources have weak metadata coverage.
    """
    if not value:
        return None
    s = value.strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        try:
            dt = parsedate_to_datetime(s)
        except (TypeError, ValueError):
            return None
        if dt is None:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_first_seen(value: str | None) -> datetime | None:
    """Parse ``first_seen`` — written by ``article_store`` so always ISO-8601."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _percentile(sorted_xs: list[float], q: float) -> float:
    """Linear-interp percentile, no numpy. ``q`` in [0, 1]. ``sorted_xs`` non-empty."""
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    idx = q * (len(sorted_xs) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_xs) - 1)
    frac = idx - lo
    return sorted_xs[lo] + frac * (sorted_xs[hi] - sorted_xs[lo])


def compute_latency_stats(
    rows: Iterable[tuple[str | None, str | None, str | None]],
) -> dict[str, dict[str, object]]:
    """Per-source latency distribution. Pure function — the unit-tested contract.

    ``rows`` is an iterable of ``(source, published, first_seen)`` raw column
    values straight out of SQLite (any may be NULL/empty). Output is keyed by
    source string (NULL → ``"?"``); the per-source dict carries:

      * ``n`` — rows that contributed a usable latency sample.
      * ``median_sec``, ``p90_sec``, ``mean_sec``, ``max_sec`` — seconds.
      * ``skipped_no_published`` — rows whose ``published`` was unparseable
        (operator visibility into metadata coverage by source).
      * ``skipped_implausible`` — rows whose computed latency exceeded
        ``_MAX_PLAUSIBLE_SEC`` (almost certainly an archive backfill, not a
        fresh ingestion — kept out of the percentiles so one stale row does
        not dominate the source's p90/max).

    Negative latencies (``published`` in the future relative to ``first_seen``
    — clock skew on the upstream feed) are clamped to 0 rather than dropped:
    they are real ingestions and silently dropping them would bias the stats
    toward fresher-than-reality. Sources with zero usable samples are still
    present in the output, carrying only the ``skipped_*`` counters — that
    keeps "we saw articles from this source but could not measure latency"
    distinct from "we saw nothing from this source".
    """
    buckets: dict[str, list[float]] = {}
    no_pub: dict[str, int] = {}
    implausible: dict[str, int] = {}
    for src, published, first_seen in rows:
        key = src if src else "?"
        pub = parse_published(published)
        seen = parse_first_seen(first_seen)
        if pub is None or seen is None:
            no_pub[key] = no_pub.get(key, 0) + 1
            continue
        delta = (seen - pub).total_seconds()
        if delta < 0:
            delta = 0.0
        if delta > _MAX_PLAUSIBLE_SEC:
            implausible[key] = implausible.get(key, 0) + 1
            continue
        buckets.setdefault(key, []).append(delta)

    out: dict[str, dict[str, object]] = {}
    all_keys = set(buckets) | set(no_pub) | set(implausible)
    for key in sorted(all_keys):
        samples = sorted(buckets.get(key, []))
        n = len(samples)
        if n == 0:
            entry: dict[str, object] = {
                "n": 0,
                "median_sec": None,
                "p90_sec": None,
                "mean_sec": None,
                "max_sec": None,
            }
        else:
            entry = {
                "n": n,
                "median_sec": round(_percentile(samples, 0.5), 1),
                "p90_sec": round(_percentile(samples, 0.9), 1),
                "mean_sec": round(sum(samples) / n, 1),
                "max_sec": round(samples[-1], 1),
            }
        if no_pub.get(key):
            entry["skipped_no_published"] = no_pub[key]
        if implausible.get(key):
            entry["skipped_implausible"] = implausible[key]
        out[key] = entry
    return out


def latency_rows(
    conn: sqlite3.Connection, hours: float = 24.0, now: datetime | None = None
) -> list[tuple[str | None, str | None, str | None]]:
    """Fetch ``(source, published, first_seen)`` for live rows in the window.

    Uses ``first_seen`` (not ``published``) for the window because:
      * ``first_seen`` is index-backed by ``idx_first_seen``.
      * It directly answers "how stale was the news the daemon *actually
        ingested* in the last N hours" — the operator's question.

    Backtest / opus rows excluded via the canonical clause.
    """
    cutoff = (_now(now) - timedelta(hours=hours)).isoformat()
    return list(
        conn.execute(
            "SELECT source, published, first_seen FROM articles "
            f"WHERE first_seen >= ? AND {LIVE_ONLY_CLAUSE}",
            (cutoff,),
        )
    )


def latency_report(
    db_path: Path | str | None = None,
    hours: float = 24.0,
    now: datetime | None = None,
) -> dict[str, object]:
    """Assemble the full, JSON-serialisable read-only latency report."""
    path = Path(db_path) if db_path else resolve_db_path()
    report: dict[str, object] = {
        "generated_at": _now(now).isoformat(),
        "db_path": str(path),
        "window_hours": hours,
    }
    try:
        conn = open_ro(path)
    except sqlite3.OperationalError as exc:
        report["error"] = f"cannot open db read-only: {exc}"
        report["per_source"] = {}
        return report
    try:
        rows = latency_rows(conn, hours=hours, now=now)
    finally:
        conn.close()
    report["sample_size"] = len(rows)
    report["per_source"] = compute_latency_stats(rows)
    return report


if __name__ == "__main__":
    hrs = float(sys.argv[1]) if len(sys.argv) > 1 else 24.0
    print(json.dumps(latency_report(hours=hrs), indent=2, default=str))

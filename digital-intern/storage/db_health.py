"""storage/db_health.py — read-only ingestion & SQLite write-contention monitor.

Why this exists (news-analyst lens): the daemon periodically logs
``insert_batch: lock retry exhausted after 5 attempts — raising``. Every one
of those lines is a *silently dropped batch of collected articles* — news the
analyst never sees. Nothing in the system surfaces that data loss as a number,
and nothing gives a fast read on whether collection is actually flowing or a
source has quietly gone dark.

This module answers those questions **read-only** and **without importing
``ArticleStore``**: it opens ``articles.db`` directly in SQLite ``mode=ro``.
That has two deliberate properties:

  1. It is immune to concurrent rewrites of ``storage/article_store.py`` — no
     dependency on that module's evolving API.
  2. A read-only connection physically cannot add to the writer contention it
     is reporting on.

Load-bearing invariant respected verbatim: every article query carries the
canonical live-only clause, so ``backtest://`` rows and ``backtest_*`` /
``opus_annotation*`` sources are never counted as live ingestion. This module
only ever reads, so it cannot affect ``ai_score`` / ``ml_score`` /
``score_source`` semantics.

CLI: ``python3 -m storage.db_health`` (or ``python3 storage/db_health.py``)
prints a JSON health report for the operator.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── Canonical backtest-isolation clause ─────────────────────────────────────
# Copied verbatim from storage/article_store.py::_LIVE_ONLY_CLAUSE. Duplicated
# (not imported) on purpose: this module must keep working even while
# article_store.py is being rewritten by a concurrent process. test_db_health
# pins this string so a drift from the canonical fragment fails CI loudly.
LIVE_ONLY_CLAUSE = (
    "url NOT LIKE 'backtest://%' "
    "AND source NOT LIKE 'backtest_%' "
    "AND source NOT LIKE 'opus_annotation%'"
)

# Mirror of article_store's path resolution, minus any mkdir side-effects
# (this module is strictly read-only and must never create directories).
_USB_PATH = Path(os.environ.get("DIGITAL_INTERN_USB", "/media/zeph/projects/digital-intern/db"))
_LOCAL_PATH = Path(__file__).resolve().parent.parent / "data"

_LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "daemon.log"

# Lines look like:
#   2026-05-18T11:11:15Z [ERROR] article_store: [article_store] insert_batch: lock retry exhausted after 5 attempts — raising
# Anchored on the leading ISO-Z timestamp so the ANSI-coloured tty echo lines
# (which start with a short HH:MM:SS) are not double-counted.
_DROP_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\s.*?\b"
    r"(?P<op>\w+): lock retry exhausted after \d+ attempts"
)


def resolve_db_path() -> Path:
    """Resolve the live ``articles.db`` path (USB-preferred), no side effects.

    Matches ``storage.article_store``'s selection logic so this monitor reads
    exactly the DB the daemon writes, but never calls ``mkdir`` — a read-only
    observer must not materialise an empty fallback directory.
    """
    usb_db = _USB_PATH / "articles.db"
    if _USB_PATH.exists() and (usb_db.exists() or _USB_PATH.is_mount()):
        return usb_db
    return _LOCAL_PATH / "articles.db"


def open_ro(path: Path) -> sqlite3.Connection:
    """Open ``path`` strictly read-only. Raises sqlite3.OperationalError if absent."""
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    return conn


def _parse_ts(value: str) -> datetime | None:
    """Parse an ISO-8601 ``first_seen`` value into an aware UTC datetime."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _now(now: datetime | None) -> datetime:
    return (now or datetime.now(timezone.utc)).astimezone(timezone.utc)


def ingestion_by_source(
    conn: sqlite3.Connection, hours: float = 1.0, now: datetime | None = None
) -> dict[str, int]:
    """Live articles inserted in the last ``hours``, counted per source.

    Backtest / opus-annotation rows are excluded via the canonical clause, so
    these counts are true *news* ingestion only. Uses a string comparison on
    ``first_seen`` (ISO-8601 UTC), the same index-friendly pattern the store
    itself uses for its windowed reads.

    Keep ``hours`` small (≤ ~2 in steady state): the ``idx_first_seen`` range
    is used, but the un-indexable ``NOT LIKE`` clause is still evaluated per
    row in the range, so a wide window over a backtest-injection burst gets
    expensive (~3s at 6h on the live ~2M-row DB; sub-second at ≤1h).
    """
    cutoff = (_now(now) - timedelta(hours=hours)).isoformat()
    rows = conn.execute(
        "SELECT COALESCE(source, '?') AS src, COUNT(*) "
        "FROM articles "
        f"WHERE first_seen >= ? AND {LIVE_ONLY_CLAUSE} "
        "GROUP BY src ORDER BY 2 DESC",
        (cutoff,),
    ).fetchall()
    return {src: int(n) for src, n in rows}


def newest_live_age_seconds(
    conn: sqlite3.Connection, now: datetime | None = None
) -> float | None:
    """Seconds since the most recent *live* article was first seen.

    ``None`` if there are no live rows at all. A large value here is the
    single clearest "collection has stalled" signal.
    """
    row = conn.execute(
        f"SELECT MAX(first_seen) FROM articles WHERE {LIVE_ONLY_CLAUSE}"
    ).fetchone()
    if not row or not row[0]:
        return None
    newest = _parse_ts(row[0])
    if newest is None:
        return None
    return max(0.0, (_now(now) - newest).total_seconds())


def stale_sources(
    conn: sqlite3.Connection,
    window_hours: float = 2.0,
    baseline_hours: float = 48.0,
    now: datetime | None = None,
) -> list[str]:
    """Sources that were producing in the recent baseline but have gone silent.

    OPT-IN ONLY — intentionally not part of ``health_report``. The running
    daemon already computes this via its ``[source_health]`` worker (grep
    ``logs/daemon.log``) without a ``DISTINCT`` scan. This implementation is
    kept for ad-hoc operator use against a *small* window; on the
    production-scale DB a backtest-injection burst can flood any wide
    ``first_seen`` range with rows the un-indexable ``NOT LIKE`` clause must
    still evaluate, so keep both windows small (default 2h vs 48h is fine on a
    quiet DB; do not widen ``baseline_hours`` for routine polling).

    "Baseline" = within the last ``baseline_hours``. "Recently" = within
    ``window_hours``. The set difference is the gone-dark list.
    """
    n = _now(now)
    recent_cut = (n - timedelta(hours=window_hours)).isoformat()
    hist_cut = (n - timedelta(hours=baseline_hours)).isoformat()

    hist = {
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT COALESCE(source, '?') FROM articles "
            f"WHERE first_seen >= ? AND {LIVE_ONLY_CLAUSE}",
            (hist_cut,),
        )
    }
    recent = {
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT COALESCE(source, '?') FROM articles "
            f"WHERE first_seen >= ? AND {LIVE_ONLY_CLAUSE}",
            (recent_cut,),
        )
    }
    return sorted(hist - recent)


def count_dropped_batches(
    log_path: Path | str = _LOG_PATH,
    hours: float = 1.0,
    now: datetime | None = None,
) -> dict[str, int]:
    """Count ``lock retry exhausted`` (dropped-batch) events in the log window.

    Returns ``{operation: count}`` plus a ``"_total"`` key, e.g.
    ``{"insert_batch": 6, "update_ml_scores_batch": 2, "_total": 8}``. Each
    counted event is one batch of collected articles that was *silently lost*
    to writer contention — this is the data-loss number nothing else exposes.
    """
    path = Path(log_path)
    counts: dict[str, int] = {}
    if not path.exists():
        return {"_total": 0}
    cutoff = _now(now) - timedelta(hours=hours)
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return {"_total": 0}
    total = 0
    for line in text.splitlines():
        m = _DROP_RE.match(line)
        if not m:
            continue
        try:
            ts = datetime.strptime(m.group("ts"), "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue
        if ts < cutoff:
            continue
        op = m.group("op")
        counts[op] = counts.get(op, 0) + 1
        total += 1
    counts["_total"] = total
    return counts


def wal_status(db_path: Path) -> dict[str, object]:
    """Size of the main DB and its ``-wal`` sidecar, in MB.

    A WAL that keeps growing without checkpointing is the textbook precursor
    to the writer-lock contention this module exists to surface.
    """
    out: dict[str, object] = {}
    try:
        out["db_mb"] = round(db_path.stat().st_size / 1e6, 1)
    except OSError:
        out["db_mb"] = None
    wal = db_path.with_name(db_path.name + "-wal")
    out["wal_mb"] = round(wal.stat().st_size / 1e6, 1) if wal.exists() else 0.0
    return out


def health_report(
    db_path: Path | str | None = None,
    log_path: Path | str | None = None,
    hours: float = 1.0,
    now: datetime | None = None,
) -> dict[str, object]:
    """Assemble the full, JSON-serialisable read-only health report."""
    path = Path(db_path) if db_path else resolve_db_path()
    report: dict[str, object] = {
        "generated_at": _now(now).isoformat(),
        "db_path": str(path),
        "window_hours": hours,
    }
    report.update(wal_status(path))
    report["dropped_batches"] = count_dropped_batches(
        log_path if log_path is not None else _LOG_PATH, hours=hours, now=now
    )
    try:
        conn = open_ro(path)
    except sqlite3.OperationalError as exc:
        report["error"] = f"cannot open db read-only: {exc}"
        return report
    try:
        by_src = ingestion_by_source(conn, hours=hours, now=now)
        report["ingestion_by_source"] = by_src
        report["live_articles_in_window"] = sum(by_src.values())
        report["newest_live_age_sec"] = newest_live_age_seconds(conn, now=now)
        # stale_sources() is deliberately NOT called here: the daemon already
        # runs a [source_health] worker (see logs/daemon.log) that computes the
        # gone-dark set without an expensive DISTINCT scan, and on the
        # production-scale DB a backtest-injection burst can flood any wide
        # first_seen window with rows the NOT LIKE clause must still evaluate.
        # The four signals above are all sub-second on the live ~2M-row DB.
    finally:
        conn.close()
    return report


if __name__ == "__main__":
    hrs = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    print(json.dumps(health_report(hours=hrs), indent=2, default=str))

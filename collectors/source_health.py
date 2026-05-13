"""Source health monitor — tracks scrape pass results per source, auto-disables silent sources."""
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

USB_PATH = Path(os.environ.get("DIGITAL_INTERN_USB", "/media/zeph/projects/digital-intern/db"))
LOCAL_PATH = Path(__file__).resolve().parent.parent / "data"

FAILURE_THRESHOLD = 3  # consecutive zero-article passes before auto-disable

SCHEMA = """
CREATE TABLE IF NOT EXISTS source_health (
    source_name           TEXT PRIMARY KEY,
    last_success_ts       TEXT,
    last_pass_ts          TEXT,
    consecutive_failures  INTEGER DEFAULT 0,
    total_passes          INTEGER DEFAULT 0,
    total_articles        INTEGER DEFAULT 0,
    disabled              INTEGER DEFAULT 0
);
"""

_lock = threading.Lock()


def _db_path() -> Path:
    if USB_PATH.parent.exists():
        try:
            USB_PATH.mkdir(parents=True, exist_ok=True)
            return USB_PATH / "source_health.db"
        except PermissionError:
            pass
    LOCAL_PATH.mkdir(parents=True, exist_ok=True)
    return LOCAL_PATH / "source_health.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db_path()), timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record_result(source_name: str, article_count: int) -> None:
    """Record a scrape pass. Increments consecutive_failures on zero, resets on success.
    Auto-disables sources that hit FAILURE_THRESHOLD consecutive zero passes."""
    now = _now()
    with _lock:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT consecutive_failures, disabled FROM source_health WHERE source_name = ?",
                (source_name,),
            ).fetchone()

            if row is None:
                cons_fail = 0 if article_count > 0 else 1
                disabled = 0
                last_success = now if article_count > 0 else None
                conn.execute(
                    """INSERT INTO source_health
                       (source_name, last_success_ts, last_pass_ts,
                        consecutive_failures, total_passes, total_articles, disabled)
                       VALUES (?, ?, ?, ?, 1, ?, ?)""",
                    (source_name, last_success, now, cons_fail, max(article_count, 0), disabled),
                )
            else:
                cur_fail, cur_disabled = row
                if article_count > 0:
                    new_fail = 0
                    new_disabled = 0  # re-enable on any success
                    conn.execute(
                        """UPDATE source_health
                           SET last_success_ts = ?, last_pass_ts = ?,
                               consecutive_failures = ?,
                               total_passes = total_passes + 1,
                               total_articles = total_articles + ?,
                               disabled = ?
                           WHERE source_name = ?""",
                        (now, now, new_fail, article_count, new_disabled, source_name),
                    )
                else:
                    new_fail = cur_fail + 1
                    new_disabled = 1 if new_fail >= FAILURE_THRESHOLD else cur_disabled
                    conn.execute(
                        """UPDATE source_health
                           SET last_pass_ts = ?,
                               consecutive_failures = ?,
                               total_passes = total_passes + 1,
                               disabled = ?
                           WHERE source_name = ?""",
                        (now, new_fail, new_disabled, source_name),
                    )
            conn.commit()
        finally:
            conn.close()


def get_disabled_sources() -> list[str]:
    """Return list of source names that are currently auto-disabled (>= FAILURE_THRESHOLD)."""
    with _lock:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT source_name FROM source_health "
                "WHERE disabled = 1 OR consecutive_failures >= ?",
                (FAILURE_THRESHOLD,),
            ).fetchall()
        finally:
            conn.close()
    return [r[0] for r in rows]


def get_health_report() -> dict:
    """Return health status for every tracked source."""
    with _lock:
        conn = _connect()
        try:
            rows = conn.execute(
                """SELECT source_name, last_success_ts, last_pass_ts,
                          consecutive_failures, total_passes, total_articles, disabled
                   FROM source_health
                   ORDER BY source_name"""
            ).fetchall()
        finally:
            conn.close()

    report = {}
    for r in rows:
        name = r[0]
        report[name] = {
            "last_success_ts": r[1],
            "last_pass_ts": r[2],
            "consecutive_failures": r[3],
            "total_passes": r[4],
            "total_articles": r[5],
            "disabled": bool(r[6]),
        }
    return report


def reset_source(source_name: str) -> None:
    """Manually re-enable a source and clear its failure counter."""
    with _lock:
        conn = _connect()
        try:
            conn.execute(
                "UPDATE source_health SET disabled = 0, consecutive_failures = 0 WHERE source_name = ?",
                (source_name,),
            )
            conn.commit()
        finally:
            conn.close()


if __name__ == "__main__":
    import json
    print(json.dumps(get_health_report(), indent=2))
    print("Disabled:", get_disabled_sources())

"""Source health monitor — tracks consecutive failures per data source.

Uses its own SQLite DB (source_health.db) placed alongside articles.db.
A "failure" is a pass that returned 0 articles. After FAILURE_THRESHOLD
consecutive failures, the source is marked disabled (callers can consult
is_disabled() to skip work).
"""
import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("source_health")

FAILURE_THRESHOLD = 3

# Suppress duplicate warning spam: same message logged at most once per window.
_WARN_DEDUP_WINDOW_SEC = 300
_warn_last_emitted: dict[str, float] = {}
_warn_lock = threading.Lock()


def _warn_dedup(msg: str) -> None:
    """Log a warning, but suppress identical messages within the dedup window."""
    now = time.monotonic()
    with _warn_lock:
        last = _warn_last_emitted.get(msg, 0.0)
        if now - last < _WARN_DEDUP_WINDOW_SEC:
            return
        _warn_last_emitted[msg] = now
    log.warning(msg)

SCHEMA = """
CREATE TABLE IF NOT EXISTS source_health (
    source TEXT PRIMARY KEY,
    last_seen TEXT,
    consecutive_failures INTEGER DEFAULT 0,
    total_articles INTEGER DEFAULT 0,
    disabled INTEGER DEFAULT 0
);
"""

_lock = threading.Lock()
_db_path_cache: Path | None = None


def _resolve_db_path() -> Path:
    """Place source_health.db alongside articles.db."""
    global _db_path_cache
    if _db_path_cache is not None:
        return _db_path_cache
    try:
        from storage.article_store import _get_db_path  # type: ignore
        articles_db = _get_db_path()
        _db_path_cache = Path(articles_db).parent / "source_health.db"
    except Exception:
        # Fallback: local data dir
        local = Path(__file__).resolve().parent.parent / "data"
        local.mkdir(parents=True, exist_ok=True)
        _db_path_cache = local / "source_health.db"
    return _db_path_cache


def _connect() -> sqlite3.Connection:
    db = _resolve_db_path()
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db), timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    # Migrate older schemas (column `source_name` -> `source`) by dropping table
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(source_health)").fetchall()]
        if cols and "source" not in cols:
            conn.execute("DROP TABLE IF EXISTS source_health")
            conn.commit()
    except Exception:
        pass
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record_result(source: str, article_count: int) -> None:
    """Record one pass of a source.

    - Resets consecutive_failures to 0 (and re-enables) when article_count > 0
    - Increments on zero; disables when consecutive_failures >= FAILURE_THRESHOLD
    """
    if not source:
        return
    now = _now()
    article_count = max(int(article_count or 0), 0)

    with _lock:
        try:
            conn = _connect()
        except Exception as e:
            _warn_dedup(f"source_health DB open failed: {e}")
            return
        try:
            row = conn.execute(
                "SELECT consecutive_failures, disabled FROM source_health WHERE source = ?",
                (source,),
            ).fetchone()

            if row is None:
                if article_count > 0:
                    cons_fail = 0
                    disabled = 0
                else:
                    cons_fail = 1
                    disabled = 0
                conn.execute(
                    """INSERT INTO source_health
                       (source, last_seen, consecutive_failures, total_articles, disabled)
                       VALUES (?, ?, ?, ?, ?)""",
                    (source, now, cons_fail, article_count, disabled),
                )
            else:
                cur_fail, cur_disabled = row[0], row[1]
                if article_count > 0:
                    new_fail = 0
                    new_disabled = 0  # re-enable on any success
                else:
                    new_fail = cur_fail + 1
                    new_disabled = 1 if new_fail >= FAILURE_THRESHOLD else cur_disabled
                conn.execute(
                    """UPDATE source_health
                       SET last_seen = ?,
                           consecutive_failures = ?,
                           total_articles = total_articles + ?,
                           disabled = ?
                       WHERE source = ?""",
                    (now, new_fail, article_count, new_disabled, source),
                )
            conn.commit()
        except Exception as e:
            _warn_dedup(f"source_health record_result failed: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass


def get_disabled_sources() -> list[str]:
    """Return source names currently disabled."""
    with _lock:
        try:
            conn = _connect()
        except Exception:
            return []
        try:
            rows = conn.execute(
                "SELECT source FROM source_health WHERE disabled = 1"
            ).fetchall()
        except Exception:
            rows = []
        finally:
            try:
                conn.close()
            except Exception:
                pass
    return [r[0] for r in rows]


def get_health_report() -> dict:
    """Return {source: {...status...}} for every tracked source."""
    with _lock:
        try:
            conn = _connect()
        except Exception:
            return {}
        try:
            rows = conn.execute(
                """SELECT source, last_seen, consecutive_failures,
                          total_articles, disabled
                   FROM source_health
                   ORDER BY source"""
            ).fetchall()
        except Exception:
            rows = []
        finally:
            try:
                conn.close()
            except Exception:
                pass
    report: dict[str, dict] = {}
    for r in rows:
        report[r[0]] = {
            "last_seen": r[1],
            "consecutive_failures": r[2],
            "total_articles": r[3],
            "disabled": bool(r[4]),
        }
    return report


def is_disabled(source: str) -> bool:
    if not source:
        return False
    with _lock:
        try:
            conn = _connect()
        except Exception:
            return False
        try:
            row = conn.execute(
                "SELECT disabled FROM source_health WHERE source = ?",
                (source,),
            ).fetchone()
        except Exception:
            row = None
        finally:
            try:
                conn.close()
            except Exception:
                pass
    return bool(row and row[0])


def reset_source(source: str) -> None:
    """Manually re-enable a source and clear its failure counter."""
    with _lock:
        try:
            conn = _connect()
        except Exception:
            return
        try:
            conn.execute(
                "UPDATE source_health SET disabled = 0, consecutive_failures = 0 WHERE source = ?",
                (source,),
            )
            conn.commit()
        finally:
            try:
                conn.close()
            except Exception:
                pass


if __name__ == "__main__":
    import json
    print(json.dumps(get_health_report(), indent=2))
    print("Disabled:", get_disabled_sources())

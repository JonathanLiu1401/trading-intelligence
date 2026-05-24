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
    disabled INTEGER DEFAULT 0,
    last_success TEXT,
    fetch_attempts INTEGER DEFAULT 0,
    fetch_successes INTEGER DEFAULT 0
);
"""

_lock = threading.Lock()
_db_path_cache: Path | None = None

# Path whose schema + legacy-column migration has already been ensured this
# process. _connect() runs on the hot path (every collector pass, ~20 worker
# threads, continuously); the executescript + PRAGMA introspection only need
# to happen once per DB. Keyed by path (not a bool) so tests that monkeypatch
# `_db_path_cache` to a fresh tmp DB still get their schema created.
_schema_ready_path: Path | None = None


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
    global _schema_ready_path
    db = _resolve_db_path()
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db), timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    if _schema_ready_path != db:
        # One-time-per-DB schema creation + legacy-column migration
        # (`source_name` -> `source`, by dropping the table). CREATE TABLE
        # IF NOT EXISTS is idempotent and the schema never changes within a
        # process, so this is skipped on every subsequent connect on the
        # hot path. Only set the guard after a successful commit so a
        # failed init is retried next call.
        try:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(source_health)").fetchall()]
            if cols and "source" not in cols:
                conn.execute("DROP TABLE IF EXISTS source_health")
                conn.commit()
        except Exception:
            pass
        conn.executescript(SCHEMA)
        # Additive migration: pre-existing DBs created before the
        # last_success column existed need an ALTER (CREATE TABLE IF NOT
        # EXISTS won't add new columns to an existing table). Idempotent —
        # caught silently if the column is already present.
        try:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(source_health)").fetchall()]
            if cols and "last_success" not in cols:
                conn.execute("ALTER TABLE source_health ADD COLUMN last_success TEXT")
            if cols and "fetch_attempts" not in cols:
                conn.execute("ALTER TABLE source_health ADD COLUMN fetch_attempts INTEGER DEFAULT 0")
            if cols and "fetch_successes" not in cols:
                conn.execute("ALTER TABLE source_health ADD COLUMN fetch_successes INTEGER DEFAULT 0")
        except Exception:
            pass
        conn.commit()
        _schema_ready_path = db
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

            success_inc = 1 if article_count > 0 else 0

            if row is None:
                if article_count > 0:
                    cons_fail = 0
                    disabled = 0
                    last_success = now
                else:
                    cons_fail = 1
                    disabled = 0
                    last_success = None
                conn.execute(
                    """INSERT INTO source_health
                       (source, last_seen, consecutive_failures, total_articles, disabled, last_success,
                        fetch_attempts, fetch_successes)
                       VALUES (?, ?, ?, ?, ?, ?, 1, ?)""",
                    (source, now, cons_fail, article_count, disabled, last_success, success_inc),
                )
            else:
                cur_fail, cur_disabled = row[0], row[1]
                if article_count > 0:
                    new_fail = 0
                    new_disabled = 0  # re-enable on any success
                    # Stamp last_success only on a productive pass; a zero
                    # pass MUST leave last_success untouched so get_dark_sources
                    # measures from the last genuine article.
                    conn.execute(
                        """UPDATE source_health
                           SET last_seen = ?,
                               consecutive_failures = ?,
                               total_articles = total_articles + ?,
                               disabled = ?,
                               last_success = ?,
                               fetch_attempts = fetch_attempts + 1,
                               fetch_successes = fetch_successes + 1
                           WHERE source = ?""",
                        (now, new_fail, article_count, new_disabled, now, source),
                    )
                else:
                    new_fail = cur_fail + 1
                    new_disabled = 1 if new_fail >= FAILURE_THRESHOLD else cur_disabled
                    conn.execute(
                        """UPDATE source_health
                           SET last_seen = ?,
                               consecutive_failures = ?,
                               total_articles = total_articles + ?,
                               disabled = ?,
                               fetch_attempts = fetch_attempts + 1
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


# A source whose last_seen is older than this is considered "stale": its
# worker is no longer calling record_result at all (crashed, or the
# collector raises before recording). This is distinct from `disabled`,
# which means the source IS being polled but produced 0 articles for
# FAILURE_THRESHOLD consecutive passes. Together they cover both failure
# modes: silently-dead workers (stale) and silently-empty sources (disabled).
DEFAULT_STALE_SECS = 3 * 3600  # 3h without a single poll


def get_stale_sources(max_age_secs: int = DEFAULT_STALE_SECS) -> list[str]:
    """Return tracked sources not polled within max_age_secs.

    A missing or unparseable last_seen counts as stale (we cannot prove the
    source is healthy). Returns a sorted list so callers/log lines are stable.
    """
    cutoff = datetime.now(timezone.utc).timestamp() - max(int(max_age_secs), 0)
    with _lock:
        try:
            conn = _connect()
        except Exception:
            return []
        try:
            rows = conn.execute(
                "SELECT source, last_seen FROM source_health"
            ).fetchall()
        except Exception:
            rows = []
        finally:
            try:
                conn.close()
            except Exception:
                pass
    stale: list[str] = []
    for source, last_seen in rows:
        if not source:
            continue
        if not last_seen:
            stale.append(source)
            continue
        try:
            ts = datetime.fromisoformat(last_seen).timestamp()
        except (ValueError, TypeError):
            stale.append(source)
            continue
        if ts < cutoff:
            stale.append(source)
    return sorted(stale)


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
                          total_articles, disabled, last_success,
                          fetch_attempts, fetch_successes
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
        attempts = r[6] or 0
        successes = r[7] or 0
        report[r[0]] = {
            "last_seen": r[1],
            "consecutive_failures": r[2],
            "total_articles": r[3],
            "disabled": bool(r[4]),
            "last_success": r[5],
            "fetch_attempts": attempts,
            "fetch_successes": successes,
            "reliability": round(successes / attempts, 3) if attempts > 0 else None,
        }
    return report


def get_reliability_report(min_attempts: int = 5) -> list[dict]:
    """Return sources sorted by fetch success rate (ascending — worst first).

    Only sources with at least ``min_attempts`` recorded passes are included so
    brand-new or barely-polled sources don't pollute the ranking.

    Each entry: {source, reliability, fetch_attempts, fetch_successes, disabled}
    """
    with _lock:
        try:
            conn = _connect()
        except Exception:
            return []
        try:
            rows = conn.execute(
                """SELECT source, fetch_attempts, fetch_successes, disabled
                   FROM source_health
                   WHERE fetch_attempts >= ?
                   ORDER BY source""",
                (min_attempts,),
            ).fetchall()
        except Exception:
            rows = []
        finally:
            try:
                conn.close()
            except Exception:
                pass

    out: list[dict] = []
    for source, attempts, successes, disabled in rows:
        attempts = attempts or 0
        successes = successes or 0
        reliability = round(successes / attempts, 3) if attempts > 0 else 0.0
        out.append({
            "source": source,
            "reliability": reliability,
            "fetch_attempts": attempts,
            "fetch_successes": successes,
            "disabled": bool(disabled),
        })
    out.sort(key=lambda r: (r["reliability"], r["source"]))
    return out


# A source whose last_seen is fresh (still being polled) but whose
# last_success is older than this threshold is "dark": the worker is
# alive and recording results, but every result has been zero articles
# for a long stretch. This is the chronic state of sec_edgar / polygon /
# newsapi / nitter — never disabled long enough to count as stale, never
# producing actual news, and impossible to tell from the disabled bit
# alone (disabled flips off on the first article and back on after 3
# zero passes — it doesn't carry "how long has this been going on").
DEFAULT_DARK_SECS = 24 * 3600  # 24h without a single article


def get_dark_sources(min_dark_secs: int = DEFAULT_DARK_SECS) -> list[tuple[str, int]]:
    """Return (source, dark_secs) for every source actively polling but
    productively dark.

    Two distinct populations:
      * ``last_success`` set but old: dark_secs = now - last_success, included
        when dark_secs >= ``min_dark_secs``.
      * ``last_success`` NULL (never produced): always included with
        dark_secs = ``-1`` as a sentinel — "we have observed this source poll
        but it has never returned an article since last_success tracking
        was introduced." This is the worst case for an analyst: a source
        that looks alive (it's polling) but has yielded literally nothing.

    Stale sources (last_seen older than DEFAULT_STALE_SECS) are excluded —
    ``get_stale_sources`` already covers a worker that stopped polling.

    Sorted with NEVER-succeeded first, then by dark_secs DESC, then alpha:
    the worst offenders surface first, and the order is stable for log lines.
    """
    if min_dark_secs < 0:
        min_dark_secs = 0
    now = datetime.now(timezone.utc).timestamp()
    stale_cutoff = now - DEFAULT_STALE_SECS

    with _lock:
        try:
            conn = _connect()
        except Exception:
            return []
        try:
            rows = conn.execute(
                "SELECT source, last_seen, last_success FROM source_health"
            ).fetchall()
        except Exception:
            rows = []
        finally:
            try:
                conn.close()
            except Exception:
                pass

    out: list[tuple[str, int]] = []
    for source, last_seen, last_success in rows:
        if not source or not last_seen:
            continue
        try:
            seen_ts = datetime.fromisoformat(last_seen).timestamp()
        except (ValueError, TypeError):
            continue
        # Exclude stale: get_stale_sources is the right signal for those.
        if seen_ts < stale_cutoff:
            continue
        if not last_success:
            out.append((source, -1))
            continue
        try:
            success_ts = datetime.fromisoformat(last_success).timestamp()
        except (ValueError, TypeError):
            # Unparseable last_success: treat as never-succeeded.
            out.append((source, -1))
            continue
        dark_secs = int(now - success_ts)
        if dark_secs >= min_dark_secs:
            out.append((source, dark_secs))
    # -1 sentinel sorts last numerically; flip its sort key so NEVER comes
    # first. Then by descending dark_secs for actual aged-out rows.
    out.sort(key=lambda r: (0 if r[1] == -1 else 1, -r[1], r[0]))
    return out


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


def delete_sources(prefix: str) -> int:
    """Delete every source_health row whose name starts with ``prefix``.

    Used by purge_worker to sweep away legacy high-cardinality keys (e.g.
    per-query ``gdelt:<query>`` rows). Those were recorded under cross-query
    dedup, so virtually every key tripped the disable threshold while the
    source itself was healthy — drowning the genuinely-down sources in the
    hourly [source_health] alert. An empty prefix is a no-op (refuses to
    wipe the whole table). Returns the number of rows removed.
    """
    if not prefix:
        return 0
    with _lock:
        try:
            conn = _connect()
        except Exception as e:
            _warn_dedup(f"source_health delete_sources open failed: {e}")
            return 0
        try:
            cur = conn.execute(
                "DELETE FROM source_health WHERE source LIKE ? ESCAPE '\\'",
                (prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%",),
            )
            conn.commit()
            return cur.rowcount or 0
        except Exception as e:
            _warn_dedup(f"source_health delete_sources failed: {e}")
            return 0
        finally:
            try:
                conn.close()
            except Exception:
                pass


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
    print("=== Reliability Report (worst first, min 5 attempts) ===")
    rel = get_reliability_report(min_attempts=5)
    for r in rel[:20]:
        pct = f"{r['reliability']*100:.1f}%"
        dis = " [DISABLED]" if r["disabled"] else ""
        print(f"  {r['source']:<35} {pct:>7}  ({r['fetch_successes']}/{r['fetch_attempts']}){dis}")
    print(f"\n  ...{len(rel)} sources total with >=5 attempts")
    print("\n=== Disabled:", get_disabled_sources())

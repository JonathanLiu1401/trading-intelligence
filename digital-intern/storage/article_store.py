"""
Compressed SQLite article store — prefers USB drive, falls back to local data/.
Stores article metadata + compressed full text. Auto-purges articles older than RETENTION_DAYS.
"""
import functools
import hashlib
import logging
import os
import random
import re
import sqlite3
import threading
import time
import zlib
from collections import deque
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path

# Near-duplicate detection at insert time — prevents syndicated copies of the
# same story from inflating the DB. Pure-Python, no network, no LLM.
try:
    from ml.dedup import dedupe_articles as _dedup_articles
    from ml.dedup import jaccard_similarity as _jaccard_sim
    from ml.dedup import title_tokens as _title_tokens
    _DEDUP_AVAILABLE = True
except ImportError:
    _DEDUP_AVAILABLE = False

try:
    from collectors.url_canonicalizer import canonical_article_id as _canonical_article_id
    _CANON_AVAILABLE = True
except ImportError:
    _CANON_AVAILABLE = False

# Module logger — uses central logger if available, falls back to stdlib.
try:
    from core.logger import get_logger
    _log = get_logger("article_store")
except Exception:
    _log = logging.getLogger("article_store")


# ── DB lock / cursor-collision retry helper ─────────────────────────────────
# Three distinct transient DB errors are retried here; all are recoverable and
# every method this decorates is idempotent — the writers
# (``UPDATE … WHERE id=?``, ``INSERT OR IGNORE``, ``DELETE … WHERE``) AND the
# pure-SELECT readers (``get_unscored``, ``get_unalerted_urgent``,
# ``get_top_for_briefing``, ``count_unscored``, ``stats``, ``stats_since``,
# ``get_briefings_for_training``) — so a clean re-run is always safe:
#
#  1. ``OperationalError("database is locked")`` — even with
#     PRAGMA busy_timeout=60000, sustained writer contention from many threads
#     (e.g. during a long PRAGMA wal_checkpoint(TRUNCATE)) still surfaces it.
#  2. ``DatabaseError("another row available"/"another row pending")`` — the
#     shared ``self.conn`` is opened ``check_same_thread=False`` and used by
#     ~30 daemon threads. ``_write_lock`` serialises *writers*, but the many
#     LOCKLESS readers (get_unscored, get_unalerted_urgent,
#     get_top_for_briefing, count_unscored, stats, …) iterate a cursor on the
#     SAME connection; a reader still mid-fetch corrupts the connection's
#     statement state when a writer's ``executemany`` runs, raising this. It
#     was observed 48x in one production ``daemon.log`` window — each one
#     dropped a whole collected/Sonnet-labeled batch (urgent items then never
#     got urgency=1 → missed alerts; articles re-queued to the LLM forever).
#     ``ef7fbe4`` decorated the writers but left these named reader victims
#     undecorated, so a collision on ``get_unalerted_urgent`` still bubbled
#     to ``alert_worker``'s broad except (urgent items unfetched that 20s
#     cycle → delayed alerts) and ``stats`` still 500'd ``/api/stats``. The
#     readers are now decorated too — the colliding reader's ``.fetchall()``
#     completes well within the first backoff tick, so the retried SELECT
#     succeeds. (The remaining uncovered path is the trainer's *direct*
#     ``store.conn.execute`` in train_continuous; it is exception-swallowed
#     and retried next cycle. A future full fix is per-call connection
#     isolation, mirroring dashboard ``_ro_query``.)
#  3. ``DatabaseError("no more rows available")`` — the SAME shared-connection
#     cursor-collision class as (2): a writer's ``executemany`` resets the
#     connection's statement state while a lockless reader (``get_unscored``,
#     ``recursive_labeler``'s scan) is mid-iteration, and the corrupted
#     cursor surfaces this variant instead of "another row …". Live evidence
#     (2026-05-18 daemon.log): ``[scorer_worker] error: no more rows
#     available`` recurred ~hourly (06:05, 08:43) plus ``[recursive_labeler]``
#     08:01 — each leaked to the worker's broad ``except`` (the substring was
#     absent from this tuple) and dropped that cycle's scored batch, so
#     urgent items went un-scored → delayed BREAKING alerts: exactly the (2)
#     failure mode, on the scoring path. Same idempotent-retry remedy. It is
#     never a legitimate end-of-results signal inside these methods —
#     ``fetchall()`` returns ``[]`` on an empty result, so this string only
#     ever means the cursor-state corruption above.
#  4. ``DatabaseError("not an error")`` — the SAME shared-connection
#     cursor-collision class as (2)/(3), but it can corrupt a *writer's*
#     ``executemany`` too, not only a lockless reader. ``pysqlite`` returns
#     "not an error" as the ``SQLITE_OK`` (errno 0) default message when the
#     connection's statement state was reset out from under the in-flight
#     call by a concurrent writer on the SAME shared ``self.conn`` — there is
#     no other sqlite message that contains this exact substring. Live
#     evidence (2026-05-18 daemon.log): ``[recursive_labeler] error: not an
#     error`` at 12:09:20Z landed exactly at the onset of a ``database is
#     locked`` writer-contention storm (insert_batch/update_ml_scores_batch
#     exhausting at 12:09:24-32Z). It surfaced from the
#     ``@_retry_on_lock``-decorated ``update_ai_scores_batch.executemany``
#     inside the recursive-labeler's round 1 (the ``round=1 candidates=500``
#     line preceded it; ``round=1 labeled=…`` was never logged) — the
#     substring was absent from THIS tuple so the decorator re-raised instead
#     of retrying the idempotent ``UPDATE … WHERE id=?``, bubbling to the
#     worker's broad ``except`` and aborting the entire 4h Sonnet/Opus
#     gold-label cycle (every remaining batch's labels discarded). The
#     recursive_labeler had ZERO successful runs since the 07:29Z daemon
#     start (08:01 "no more rows available" pre-(3)-fix on a stale daemon,
#     12:09 "not an error" — a HEAD bug until this entry). Same
#     idempotent-retry remedy; safe by construction (every decorated op is
#     idempotent and only the errno-0 default carries this string).
#
# Retry with exponential backoff + jitter to avoid thundering-herd retries
# that would just collide again at the same instant. Bubble up after the
# budget is spent. The substring filter keeps this tight — other
# ``DatabaseError`` flavors (e.g. ``IntegrityError`` "UNIQUE constraint
# failed") must still propagate, never be silently swallowed.
#  6. ``OperationalError("cannot commit transaction - SQL statements in
#     progress")`` — the SAME shared-connection cursor-collision class as
#     (2)/(3)/(4)/(5), but it surfaces at the ``self.conn.commit()`` step
#     AFTER the decorated writer's ``executemany`` completes: a sibling
#     lockless reader cursor on the same ``self.conn``
#     (``check_same_thread=False``, ~30 daemon threads) is still mid-fetch,
#     so SQLite refuses the COMMIT because the in-flight prepared statements
#     would be torn down by the transaction boundary. Live evidence
#     (2026-05-23 daemon.log): 4 occurrences in one day across
#     scorer_worker, hackernews_worker and the alert path's
#     ``mark_alerted_batch`` — the latter's traceback bubbled all the way
#     to ``[alert] failed to mark stale rows alerted``, leaving the
#     stale-but-not-marked urgent rows re-fetched every 20s cycle until the
#     24h reaper. Every decorated method is fully idempotent
#     (``UPDATE … WHERE id=?`` / ``INSERT OR IGNORE``) so re-executing the
#     write on retry is safe by the same argument as the other five
#     flavours; without this string the OperationalError bubbles to the
#     worker's broad ``except`` and a whole cycle's work is lost.
_RETRYABLE_DB_ERRORS = (
    "database is locked",
    "another row available",
    "another row pending",
    "no more rows available",
    "not an error",
    "cannot commit transaction",
)
_LOCK_RETRY_ATTEMPTS = 5
_LOCK_RETRY_BASE_S = 0.25
_LOCK_RETRY_CAP_S = 4.0

# Process-lifetime lock-contention counters. Surfaced via ArticleStore.stats()
# so the dashboard / hourly healthcheck can quantify "database is locked"
# pressure instead of only seeing scattered WARNING lines in the logs.
# ``lock_retries`` counts individual retry sleeps; ``lock_failures`` counts
# calls that exhausted the retry budget and raised.
_lock_metrics_lock = threading.Lock()
_lock_retries = 0
_lock_failures = 0


def lock_metrics() -> dict:
    """Snapshot of process-lifetime DB lock-contention counters."""
    with _lock_metrics_lock:
        return {"lock_retries": _lock_retries, "lock_failures": _lock_failures}


# The CPython sqlite3 C bindings sometimes propagate a cursor-collision
# failure as ``SystemError: error return without exception set`` instead of a
# proper ``sqlite3.DatabaseError`` — most often at ``self.conn.commit()`` when
# a sibling lockless reader still holds prepared statements on the shared
# ``self.conn``. The C code returned an error code but failed to set the
# Python-level exception state, so CPython raises a bare ``SystemError`` with
# this canonical message. Live evidence (2026-05-23 daemon.log): 3
# occurrences from rss_worker / scorer_worker stats during the same
# writer-contention storm that produced the (6) ``cannot commit transaction``
# variant. Same idempotent-retry remedy; we only retry the EXACT canonical
# message to keep the discrimination tight (a real Python internal SystemError
# is not retryable and must propagate).
_SYSTEMERROR_RETRYABLE_SUBSTR = "error return without exception set"


def _retry_on_lock(func):
    """Decorator: retry transient DB errors (``database is locked`` /
    shared-connection ``another row available`` / cursor-collision
    ``SystemError: error return without exception set``) with exp backoff +
    jitter. Non-retryable ``DatabaseError`` flavors (IntegrityError, etc.)
    propagate. See ``_RETRYABLE_DB_ERRORS`` and
    ``_SYSTEMERROR_RETRYABLE_SUBSTR`` for the full rationale."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        last = None
        for attempt in range(_LOCK_RETRY_ATTEMPTS):
            try:
                return func(*args, **kwargs)
            except sqlite3.DatabaseError as e:
                # DatabaseError is the base of OperationalError AND
                # IntegrityError; discriminate on the message so only the
                # known-transient classes retry and everything else raises.
                if not any(s in str(e).lower() for s in _RETRYABLE_DB_ERRORS):
                    raise
                last = e
            except SystemError as e:
                # Narrow discrimination: only the canonical C-binding cursor-
                # collision message (see _SYSTEMERROR_RETRYABLE_SUBSTR). Any
                # other SystemError is a genuine Python internal bug and
                # MUST propagate unmolested — broadly catching SystemError
                # would mask serious failures.
                if _SYSTEMERROR_RETRYABLE_SUBSTR not in str(e).lower():
                    raise
                last = e
            if attempt + 1 < _LOCK_RETRY_ATTEMPTS:
                global _lock_retries
                with _lock_metrics_lock:
                    _lock_retries += 1
                # Exponential backoff: 0.25, 0.5, 1.0, 2.0, ... capped at 4s.
                # Add jitter in [0.5x, 1.5x) so concurrent retriers desync.
                delay = min(_LOCK_RETRY_BASE_S * (2 ** attempt), _LOCK_RETRY_CAP_S)
                delay *= 0.5 + random.random()
                _log.warning(
                    f"[article_store] {func.__name__}: transient DB error "
                    f"{str(last)[:60]!r} "
                    f"(attempt {attempt + 1}/{_LOCK_RETRY_ATTEMPTS}); "
                    f"retrying in {delay:.2f}s"
                )
                time.sleep(delay)
        assert last is not None
        global _lock_failures
        with _lock_metrics_lock:
            _lock_failures += 1
        _log.error(
            f"[article_store] {func.__name__}: lock retry exhausted after "
            f"{_LOCK_RETRY_ATTEMPTS} attempts — raising"
        )
        raise last
    return wrapper

def _expect_row(cur):
    """``cur.fetchone()`` for an aggregate (``MAX(...)`` / ``COUNT(*)``) that
    SQL guarantees yields EXACTLY one row — so a ``None`` here is never a
    legitimate empty result.

    It is the SAME shared-``self.conn`` cursor-collision documented at length
    on ``_retry_on_lock`` (a concurrent writer's ``executemany`` resets the
    connection's statement state while this lockless reader is mid-fetch),
    just surfacing as a corrupted ``fetchone()`` that returns ``None`` instead
    of raising the ``another row available`` / ``no more rows available``
    ``DatabaseError`` variant. The caller does ``.fetchone()[0]``, so a ``None``
    became ``TypeError: 'NoneType' object is not subscriptable`` — which is NOT
    a ``sqlite3.DatabaseError``, so ``_retry_on_lock`` never caught it and it
    bubbled to the worker's broad ``except`` EVERY contended cycle. Live
    evidence (2026-05-18 daemon.log): ``[stats_worker] error: 'NoneType'
    object is not subscriptable`` recurred 12+×/h, exactly correlated with the
    concurrent ``database is locked`` writer-contention storm.

    Re-raise it as the same retryable signal the decorator already handles for
    the ``DatabaseError`` flavour of this identical collision, so the
    ``@_retry_on_lock``-decorated idempotent reader simply retries and the
    next attempt — past the writer's ``executemany`` — succeeds. Safe by
    construction: every call site is a ``MAX``/``COUNT`` aggregate, which
    SQLite ALWAYS returns one row for, so this can never mask a real empty
    result (mirrors the ``_retry_on_lock`` rationale: this only ever means
    cursor-state corruption).

    Empty-tuple guard: cursor-state corruption can ALSO surface as
    ``fetchone()`` returning ``()`` instead of ``None`` (or the documented
    ``DatabaseError`` variants). The caller's ``[0]`` then raises
    ``IndexError: tuple index out of range`` — also NOT a
    ``sqlite3.DatabaseError``, so ``_retry_on_lock`` never catches it and it
    bubbles to the worker's broad ``except`` exactly like the ``None`` case
    did before the 2026-05-18 fix. Live evidence (2026-05-19/20 daemon.log):
    ``[stats_worker] error: tuple index out of range`` recurred under the
    same ``database is locked`` writer-contention storm pattern. Treating an
    empty tuple identically to ``None`` is safe: every aggregate call site
    yields a 1-column row, so ``len(row) == 0`` is also a corruption signal
    that can never be a legitimate result."""
    row = cur.fetchone()
    if row is None or len(row) == 0:
        raise sqlite3.OperationalError("another row available")
    return row


USB_PATH = Path(os.environ.get("DIGITAL_INTERN_USB", "/media/zeph/projects/digital-intern/db"))
LOCAL_PATH = Path(__file__).resolve().parent.parent / "data"
RETENTION_DAYS = 90

# SQL fragment used to exclude synthetic / historical training data from the
# live news pipeline. Backtest replays and Opus annotation runs insert rows
# with ``backtest://`` URLs and ``backtest_*`` / ``opus_annotation*`` source
# tags; these are valid for training but must never be re-scored, re-alerted,
# or surfaced in heartbeat briefings as breaking news.
_LIVE_ONLY_CLAUSE = (
    "url NOT LIKE 'backtest://%' "
    "AND source NOT LIKE 'backtest_%' "
    "AND source NOT LIKE 'opus_annotation%'"
)

# ── stats() backlog-count cache ─────────────────────────────────────────────
# ``stats()`` is called by the /api/stats dashboard endpoint. Its ``total`` and
# ``urgent`` counts are O(log N) (MAX(rowid) and the idx_urgency lookup), but
# ``unscored``/``below_threshold`` filter on ``ai_score=0 AND ml_score IS NULL
# AND <kw_score cmp> AND <live LIKE clauses>`` for which no index is selective:
# SQLite full-scans the ~1.9M-row table, reading every row's compressed
# ``full_text`` BLOB pages off the slow USB drive. Measured ~115 s for
# ``unscored`` alone — a ``LIMIT`` does NOT bound this because matches
# (~34k) are far fewer than any sane cap, so the scan never short-circuits.
# That single query made /api/stats time out (>30 s) so the dashboard rendered
# "0 Total in DB". These two counts are a slowly-changing backlog gauge, not a
# realtime figure, so they are served from a short-TTL cache that is refreshed
# OFF the request path by one background thread on its own transient
# connection (never ``self.conn`` — see the cursor-collision hazard documented
# on the ``_retry_on_lock`` decorator above). A cold cache returns 0 for one
# poll cycle and is filled within a couple of minutes; ``stats()`` itself never
# blocks on the scan again.
_STATS_BACKLOG_TTL_SECS = 300
_STATS_BACKLOG_LOCK = threading.Lock()
_STATS_BACKLOG_CACHE: dict = {"ts": 0.0, "unscored": 0,
                              "below_threshold": 0, "refreshing": False}


def _refresh_backlog_counts(score_min_kw: float) -> None:
    """Recompute the expensive unscored/below_threshold counts on a private,
    short-lived connection and update the module cache. Runs in a daemon
    background thread; exceptions are swallowed (next cycle retries)."""
    conn = None
    try:
        conn = sqlite3.connect(str(_get_db_path()), timeout=60,
                               check_same_thread=False)
        conn.execute("PRAGMA busy_timeout=60000")
        unscored = conn.execute(
            "SELECT COUNT(*) FROM articles "
            f"WHERE ai_score=0 AND ml_score IS NULL AND kw_score>=? "
            f"AND {_LIVE_ONLY_CLAUSE}",
            (score_min_kw,),
        ).fetchone()[0]
        below = conn.execute(
            "SELECT COUNT(*) FROM articles "
            f"WHERE ai_score=0 AND ml_score IS NULL AND kw_score<? "
            f"AND {_LIVE_ONLY_CLAUSE}",
            (score_min_kw,),
        ).fetchone()[0]
        with _STATS_BACKLOG_LOCK:
            _STATS_BACKLOG_CACHE.update(ts=time.time(), unscored=unscored,
                                        below_threshold=below)
    except Exception as e:  # pragma: no cover - best-effort cache refresh
        _log.warning("backlog count refresh failed: %s", e)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        with _STATS_BACKLOG_LOCK:
            _STATS_BACKLOG_CACHE["refreshing"] = False

# Max rows any single resolved publisher domain may occupy in one heartbeat
# briefing. Live evidence (2026-05-18): one scrape channel
# (``scraped/finance.yahoo.com`` price-quote widget pages, ML-scored ~9.9)
# took 10 of the top-50 briefing slots, several near-identical, crowding out
# diverse real headlines — the consuming analyst's top noise complaint. The
# cap re-prioritises the digest for source diversity; it NEVER shrinks it
# (a low-diversity window backfills from score-ordered overflow). Tuned so a
# single publisher can hold at most ~⅓ of a 20-slot digest yet a genuinely
# active wire is still well represented.
BRIEFING_MAX_PER_DOMAIN = 6

# Global non-reentrant inference lock — prevents two callers running
# score_pending() concurrently. Non-blocking acquire so concurrent
# callers no-op rather than queue.
_INFER_LOCK = threading.Lock()
SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id          TEXT PRIMARY KEY,
    url         TEXT NOT NULL,
    title       TEXT NOT NULL,
    source      TEXT,
    published   TEXT,
    kw_score    REAL DEFAULT 0,
    ai_score    REAL DEFAULT 0,
    urgency     INTEGER DEFAULT 0,   -- 0=normal 1=urgent 2=alerted
    full_text   BLOB,               -- zlib-compressed
    first_seen  TEXT NOT NULL,
    cycle       INTEGER DEFAULT 0,  -- collection cycle number
    time_sensitivity REAL DEFAULT NULL, -- 0..1, ML-predicted recency decay rate (NULL until scored)
    ml_score    REAL DEFAULT NULL,  -- model's own prediction (separate from ai_score to avoid training-feedback contamination)
    score_source TEXT DEFAULT NULL, -- 'llm' (Sonnet/Opus ground truth), 'ml' (model), 'briefing_boost' (Opus curation nudge); NULL=unscored
    breaking_news INTEGER DEFAULT 0 -- 1 when analytics.breaking_news_detector finds a same-ticker burst
);
CREATE INDEX IF NOT EXISTS idx_urgency   ON articles(urgency);
CREATE INDEX IF NOT EXISTS idx_ai_score  ON articles(ai_score DESC);
CREATE INDEX IF NOT EXISTS idx_first_seen ON articles(first_seen);
CREATE INDEX IF NOT EXISTS idx_cycle     ON articles(cycle);

CREATE TABLE IF NOT EXISTS briefings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL,
    text            TEXT NOT NULL,
    article_count   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_briefings_ts ON briefings(ts);
"""


def _get_db_path() -> Path:
    # Prefer USB drive when mounted — it has far more capacity than the root NVMe.
    # Falls back to local data/ if USB is not available.
    usb_db = USB_PATH / "articles.db"
    if USB_PATH.exists() and (usb_db.exists() or USB_PATH.is_mount()):
        USB_PATH.mkdir(parents=True, exist_ok=True)
        return usb_db
    LOCAL_PATH.mkdir(parents=True, exist_ok=True)
    return LOCAL_PATH / "articles.db"


# Path for which the schema has already been ensured in *this* process.
# `executescript(SCHEMA) + commit()` is idempotent but not free: it is parsed
# on every connect and the commit forces a transaction boundary. The daemon,
# the dashboard, monitor.py and — notably — every spawned ml/trainer child
# process open their own connection, so this ran far more than necessary.
# Mirrors the proven guard in collectors/source_health.py.
_schema_ready_path: str | None = None


def _schema_present(conn: sqlite3.Connection) -> bool:
    """Read-only check (no write lock) that the core tables already exist."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name IN ('articles', 'briefings')"
    ).fetchall()
    return len({r[0] for r in rows}) == 2


def _connect() -> sqlite3.Connection:
    global _schema_ready_path
    db = _get_db_path()
    db_key = str(db)
    conn = sqlite3.connect(db_key, timeout=60, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")   # 64MB cache
    conn.execute("PRAGMA busy_timeout=60000")  # wait up to 60s on lock
    # Skip the executescript + commit on the hot path. The cheap sqlite_master
    # read still runs on a fresh process so a brand-new / empty DB is always
    # bootstrapped correctly; only the redundant write-transaction boundary on
    # an already-initialised DB is elided. NOTE: like source_health.py this is
    # create-only — a future column addition needs an explicit migration here,
    # it will NOT be picked up by the IF NOT EXISTS schema alone.
    if _schema_ready_path != db_key and not _schema_present(conn):
        conn.executescript(SCHEMA)
        conn.commit()
    _schema_ready_path = db_key
    return conn


def article_id(url: str, title: str) -> str:
    return hashlib.sha256(f"{url}||{title}".encode()).hexdigest()


def _published_older_than(published: str, cutoff: datetime) -> bool:
    """True only when ``published`` parses successfully AND is older than
    ``cutoff``. Empty or unparseable values return False (keep the article).

    The SQL pre-filter in get_top_for_briefing compares ``published`` as a raw
    string, which is meaningless for RFC822-formatted dates ("Wed, 14 May
    2026 ...") — their leading letter lex-sorts after any ISO cutoff, so every
    RSS article silently bypasses the 24h staleness check. This parses both
    RFC822 and ISO forms so the recency filter actually works. Dropping on a
    parse failure is deliberately avoided — that risks emptying the briefing.
    """
    if not published:
        return False
    dt = None
    try:
        dt = parsedate_to_datetime(published)
    except Exception:
        dt = None
    if dt is None:
        try:
            dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
        except Exception:
            return False
    if dt is None:
        return False
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt < cutoff


def _briefing_domain_key(source: str) -> str:
    """Resolve a ``source`` tag to a diversity-grouping key for the briefing cap.

    Aggregator-prefixed tags carry the real publisher as an embedded host:
    ``scraped/finance.yahoo.com`` → ``finance.yahoo.com``,
    ``GDELT/techtimes.com`` → ``techtimes.com``. When a dotted host is present
    that host (verbatim, not eTLD+1 — avoids the ``co.uk`` public-suffix trap)
    is the key, so the live failure mode (the same scrape host repeated 10×)
    collapses to one bucket. Tags with no dotted host (``GoogleNews/MSN``,
    ``AlphaVantage/Seeking Alpha``, ``Reuters Markets GN``) fall back to the
    full normalised tag, so distinct publishers under one aggregator stay
    distinct and only exact-duplicate tags are capped — deliberately
    conservative.

    Intentionally a small local helper rather than reusing
    ``ml.features._domain_candidates``: the storage layer must not pull the
    ml/numpy import graph (and risk an import cycle) for a pure string key,
    and the diversity key does not need to match credibility resolution
    byte-for-byte — it only needs to group obvious same-publisher repetition.
    """
    if not source:
        return ""
    s = source.strip().lower()
    for part in re.split(r"[\s/:|]+", s):
        part = part.strip().strip(".")
        if part.startswith("www."):
            part = part[4:]
        if "." in part and len(part) >= 4 and not part.replace(".", "").isdigit():
            return part
    return s


def compress(text: str) -> bytes:
    return zlib.compress(text.encode("utf-8", errors="replace"), level=6)


def decompress(data) -> str:
    """Decompress a zlib-compressed BLOB back to text.

    Defensive against a string slipping into the ``full_text`` column —
    SQLite has dynamic typing so a collector that forgets to ``compress()``
    its summary leaves a TEXT-affinity value behind a BLOB-declared
    column. Live regression (2026-05-27): ``collectors.source_quality_scorer``
    wrote ``summary`` (a str) directly into ``articles.full_text``, leaving
    a single row that crashed ``decompress(r[4])`` in ``get_unscored``
    with ``a bytes-like object is required, not 'str'`` — and because that
    one row sits at ai_score=0 / ml_score=NULL it is re-fetched on every
    scorer cycle, taking out the *entire* batch every 30s (300+ WARNING
    lines/day in daemon.log, scorer effectively dark for that batch).
    Treating an already-str payload as already-decoded is the only
    interpretation a future caller could want — it was clearly inserted
    pre-decompressed. Bytes path is byte-unchanged."""
    if isinstance(data, str):
        return data
    return zlib.decompress(data).decode("utf-8", errors="replace")


class ArticleStore:
    def __init__(self):
        self.conn = _connect()
        # Serializes write operations across worker threads. The connection is shared
        # with check_same_thread=False; concurrent execute+commit pairs would otherwise
        # race on implicit transaction starts ("cannot start a transaction within a transaction").
        self._write_lock = threading.Lock()
        # Rolling cache of title-token sets for cross-batch near-dedup.
        # maxlen=500 keeps ~last 10 collection cycles in memory; deque append
        # is thread-safe for single-item ops, but we only touch it inside _write_lock.
        self._recent_title_fps: deque = deque(maxlen=500)
        self._dedup_skipped = 0  # lifetime counter for logging
        self._migrate()
        db = _get_db_path()
        print(f"[store] Using DB at {db} ({db.stat().st_size // 1024}KB)" if db.exists() else f"[store] New DB at {db}")

    def _migrate(self) -> None:
        """Apply additive schema migrations to pre-existing DBs.
        SQLite's CREATE TABLE IF NOT EXISTS won't add new columns to an existing
        table, so we ALTER TABLE on-the-fly for each added column."""
        try:
            cols = {row[1] for row in self.conn.execute("PRAGMA table_info(articles)").fetchall()}
        except sqlite3.OperationalError:
            return
        with self._write_lock:
            if "time_sensitivity" not in cols:
                try:
                    self.conn.execute(
                        "ALTER TABLE articles ADD COLUMN time_sensitivity REAL DEFAULT NULL"
                    )
                    self.conn.commit()
                    _log.info("[article_store] migration: added articles.time_sensitivity column")
                except sqlite3.OperationalError as e:
                    # Race with another process — column may already exist.
                    if "duplicate column" not in str(e).lower():
                        raise
            if "ml_score" not in cols:
                try:
                    self.conn.execute(
                        "ALTER TABLE articles ADD COLUMN ml_score REAL DEFAULT NULL"
                    )
                    self.conn.commit()
                    _log.info("[article_store] migration: added articles.ml_score column")
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
            if "score_source" not in cols:
                try:
                    self.conn.execute(
                        "ALTER TABLE articles ADD COLUMN score_source TEXT DEFAULT NULL"
                    )
                    self.conn.commit()
                    _log.info("[article_store] migration: added articles.score_source column")
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
            if "breaking_news" not in cols:
                try:
                    self.conn.execute(
                        "ALTER TABLE articles ADD COLUMN breaking_news INTEGER DEFAULT 0"
                    )
                    self.conn.commit()
                    _log.info("[article_store] migration: added articles.breaking_news column")
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
            # One-time cleanup of training-label contamination: model predictions
            # that were written into ai_score (the column the trainer reads as
            # ground truth) created a feedback loop where the model trained on
            # its own outputs. Heuristic: real LLM labels are integer-valued
            # (Sonnet returns int "score"; recursive_labeler does int*2.0 → still
            # integer). Anything else in ai_score is a model output. We move it
            # to ml_score, tag score_source='ml', and zero ai_score so those
            # articles get re-routed through Sonnet via get_unscored().
            #
            # CRITICAL: the integer heuristic only holds for *live news* rows.
            # Synthetic backtest / opus-annotation rows legitimately carry
            # fractional labels (backtest SELL=0.5, opus NEUTRAL=2.5, BAD=0.5)
            # and are inserted by paper-trader with score_source NULL. Applying
            # the heuristic to them destroys training data, so every clause
            # below is scoped with _LIVE_ONLY_CLAUSE.
            try:
                # Recovery: an earlier revision of the cleanup ran the integer
                # heuristic against ALL rows, zeroing ai_score on synthetic rows
                # and stashing the real label in ml_score. Synthetic rows never
                # go through ML scoring (get_unscored excludes them), so a
                # non-NULL ml_score on a synthetic row can only be that misfire.
                # Restore the label and clear the bogus 'ml' tag. Idempotent.
                rec = self.conn.execute(
                    "UPDATE articles SET ai_score = ml_score, ml_score = NULL, "
                    "score_source = NULL "
                    "WHERE score_source = 'ml' AND ml_score IS NOT NULL "
                    f"AND NOT ({_LIVE_ONLY_CLAUSE})"
                )
                n_restored = rec.rowcount
                self.conn.commit()
                if n_restored:
                    _log.info(
                        f"[article_store] migration: restored {n_restored} synthetic "
                        f"training labels wrongly moved to ml_score"
                    )
            except sqlite3.OperationalError as e:
                _log.warning(f"[article_store] synthetic-label recovery skipped: {e}")
            try:
                # Run only when live-news rows still need the split — guarded by
                # checking for any score_source=NULL live row with a non-integer
                # ai_score. Scoping to live-only also makes this idempotent: a
                # past version re-fired on every restart because freshly
                # injected synthetic rows kept matching.
                needs_cleanup = self.conn.execute(
                    "SELECT EXISTS(SELECT 1 FROM articles "
                    "WHERE score_source IS NULL AND ai_score > 0 "
                    f"AND ai_score != CAST(ai_score AS INTEGER) AND {_LIVE_ONLY_CLAUSE})"
                ).fetchone()[0]
                if needs_cleanup:
                    # Move suspected-model values into ml_score for visibility.
                    self.conn.execute(
                        "UPDATE articles SET ml_score = ai_score, "
                        "ai_score = 0, score_source = 'ml' "
                        "WHERE score_source IS NULL AND ai_score > 0 "
                        f"AND ai_score != CAST(ai_score AS INTEGER) AND {_LIVE_ONLY_CLAUSE}"
                    )
                    n_cleaned = self.conn.execute("SELECT changes()").fetchone()[0]
                    # Mark surviving integer-valued ai_score rows as LLM source.
                    self.conn.execute(
                        "UPDATE articles SET score_source = 'llm' "
                        "WHERE score_source IS NULL AND ai_score > 0 "
                        f"AND ai_score = CAST(ai_score AS INTEGER) AND {_LIVE_ONLY_CLAUSE}"
                    )
                    self.conn.commit()
                    _log.info(
                        f"[article_store] migration: cleaned {n_cleaned} contaminated "
                        f"ai_score rows (moved to ml_score, ai_score=0 for re-labeling)"
                    )
            except sqlite3.OperationalError as e:
                _log.warning(f"[article_store] label-cleanup migration skipped: {e}")

    @_retry_on_lock
    def insert_batch(self, articles: list, cycle: int = 0) -> int:
        """Insert new articles; skip duplicates. Returns count of new insertions."""
        if not articles:
            return 0
        now = datetime.now(timezone.utc).isoformat()
        inserted = 0
        skipped_dedup = 0
        with self._write_lock:
            # ── Near-duplicate collapse ────────────────────────────────────────
            # 1. Within-batch: dedupe_articles collapses syndicated copies of the
            #    same story (e.g. same headline from GDELT + RSS + Finnhub arriving
            #    in the same cycle). Keeps the highest-scored representative.
            # 2. Cross-batch: check each surviving article's title tokens against
            #    _recent_title_fps (rolling 500-entry deque of the last few cycles).
            #    Jaccard >= 0.6 = same story already in DB → skip.
            if _DEDUP_AVAILABLE:
                pre = len(articles)
                articles = _dedup_articles(articles, score_key="_relevance_score")
                skipped_dedup += pre - len(articles)

                unique: list = []
                for art in articles:
                    toks = _title_tokens(art.get("title"))
                    if toks and any(
                        _jaccard_sim(toks, cached) >= 0.6
                        for cached in self._recent_title_fps
                    ):
                        skipped_dedup += 1
                        continue
                    unique.append(art)
                    if toks:
                        self._recent_title_fps.append(frozenset(toks))
                articles = unique

            # ── DB insertion ───────────────────────────────────────────────────
            # Bulk URL pre-check: fetch which URLs in this batch already exist.
            # INSERT OR IGNORE only dedupes by id (PRIMARY KEY); the same URL can
            # arrive under a different id from different collectors and slip through.
            candidate_urls = [a.get("link", "") for a in articles if a.get("link")]
            existing_urls: set[str] = set()
            if candidate_urls:
                placeholders = ",".join("?" * len(candidate_urls))
                rows = self.conn.execute(
                    f"SELECT url FROM articles WHERE url IN ({placeholders})",
                    candidate_urls,
                ).fetchall()
                existing_urls = {r[0] for r in rows}

            skipped_url_dup = 0
            for art in articles:
                url = art.get("link", "")
                title = art.get("title", "")
                if not url or not title:
                    continue
                if url in existing_urls:
                    skipped_url_dup += 1
                    continue
                aid = _canonical_article_id(url, title) if _CANON_AVAILABLE else article_id(url, title)
                summary = art.get("summary", "")
                self.conn.execute(
                    "INSERT OR IGNORE INTO articles "
                    "(id, url, title, source, published, kw_score, first_seen, cycle, full_text) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (aid, url, title, art.get("source", ""), art.get("published", ""),
                     art.get("_relevance_score", 0), now, cycle,
                     compress(summary) if summary else None),
                )
                if self.conn.execute("SELECT changes()").fetchone()[0]:
                    inserted += 1
                    existing_urls.add(url)  # prevent within-batch dups after first insert
            self.conn.commit()
            if skipped_url_dup:
                _log.debug("[insert_batch] url-dedup skipped %d articles", skipped_url_dup)

        if skipped_dedup:
            self._dedup_skipped += skipped_dedup
            _log.info(
                f"[insert_batch] near-dedup skipped {skipped_dedup} articles "
                f"(lifetime total: {self._dedup_skipped})"
            )
        return inserted

    @_retry_on_lock
    def update_ai_score(self, aid: str, score: float, urgency: int = 0):
        """Write an LLM (Sonnet/Opus) relevance label. Tags score_source='llm'
        so the trainer treats it as ground truth."""
        with self._write_lock:
            self.conn.execute(
                "UPDATE articles SET ai_score=?, urgency=MAX(urgency,?), "
                "score_source='llm' WHERE id=?",
                (score, urgency, aid),
            )
            self.conn.commit()

    @_retry_on_lock
    def update_ai_scores_batch(self, updates: list[tuple[str, float, int]]):
        """Bulk update for LLM labels: updates is list of (aid, score, urgency).
        Single transaction. Tags score_source='llm' so the trainer treats these
        as ground-truth labels (not model self-predictions)."""
        with self._write_lock:
            self.conn.executemany(
                "UPDATE articles SET ai_score=?, urgency=MAX(urgency,?), "
                "score_source='llm' WHERE id=?",
                [(score, urgency, aid) for aid, score, urgency in updates],
            )
            self.conn.commit()

    @_retry_on_lock
    def update_ml_scores_batch(self, updates: list[tuple[str, float, int]]):
        """Bulk-write the *model's* own predictions to ml_score (NOT ai_score).
        ``updates`` is a list of (aid, relevance_score, urgency_flag).

        Keeping model output out of ai_score is what prevents the training-label
        feedback loop — the trainer reads ai_score (LLM labels only). Urgent
        items still bump urgency so the alerting path keeps working. Readers
        that want a unified score should COALESCE(NULLIF(ai_score,0), ml_score)."""
        if not updates:
            return
        with self._write_lock:
            self.conn.executemany(
                "UPDATE articles SET ml_score=?, urgency=MAX(urgency,?), "
                "score_source=COALESCE(score_source, 'ml') WHERE id=?",
                [(score, urgency, aid) for aid, score, urgency in updates],
            )
            self.conn.commit()

    @_retry_on_lock
    def update_time_sensitivity_batch(self, updates: list[tuple[str, float]]):
        """Bulk-store ML-predicted time_sensitivity (0..1).
        ``updates`` is a list of (aid, time_sensitivity)."""
        if not updates:
            return
        with self._write_lock:
            self.conn.executemany(
                "UPDATE articles SET time_sensitivity=? WHERE id=?",
                [(float(ts), aid) for aid, ts in updates],
            )
            self.conn.commit()

    @_retry_on_lock
    def mark_alerted(self, aid: str):
        with self._write_lock:
            self.conn.execute("UPDATE articles SET urgency=2 WHERE id=?", (aid,))
            self.conn.commit()

    @_retry_on_lock
    def mark_alerted_batch(self, aids: list[str]) -> int:
        """Mark multiple articles alerted in one transaction. Returns rowcount."""
        if not aids:
            return 0
        with self._write_lock:
            cur = self.conn.executemany(
                "UPDATE articles SET urgency=2 WHERE id=?",
                [(aid,) for aid in aids],
            )
            n = cur.rowcount
            self.conn.commit()
        return n

    @_retry_on_lock
    def update_scores_from_labels(self, labels: list[dict]) -> int:
        """Label articles flagged in_briefing by Opus as mid-tier positive (4.5).

        Briefing inclusion is one of the strongest signals available (Opus
        curated this article into the heartbeat). The previous
        ``MIN(5.0, ai_score + 0.3)`` formula misbehaved at both ends: it
        *downgraded* high-LLM-scored articles (8 → capped at 5.0) and
        *under-labeled* unscored briefing mentions (0 → 0.3, which the trainer
        then sees as "3% relevance"). Use ``MAX(ai_score, 4.5)`` so existing
        higher-quality LLM labels are preserved and unscored articles enter
        the training pool with the same magnitude that
        ``trainer._fetch_briefing_samples`` already uses.
        """
        urls = [l.get("url") for l in labels if l.get("in_briefing") and l.get("url")]
        if not urls:
            return 0
        placeholders = ",".join("?" * len(urls))
        with self._write_lock:
            # Defense-in-depth: this is a *write* of strong ground-truth labels
            # (briefing_boost is read by the trainer with the same weight as
            # 'llm'). The label list is derived from get_top_for_briefing()
            # which is already live-only, so a backtest:// URL cannot reach
            # here today — but if it ever did, MAX(ai_score, 4.5) would rewrite
            # a synthetic row's fractional outcome label (a SELL-loser's 0.5 →
            # 4.5) and flip its source tag, silently poisoning the training
            # pool. Apply _LIVE_ONLY_CLAUSE so this write keeps the same
            # backtest-isolation discipline as every other live path.
            cur = self.conn.execute(
                f"UPDATE articles SET ai_score = MAX(ai_score, 4.5), "
                f"score_source = CASE WHEN score_source='llm' THEN 'llm' "
                f"ELSE 'briefing_boost' END "
                f"WHERE url IN ({placeholders}) AND {_LIVE_ONLY_CLAUSE}",
                urls,
            )
            n = cur.rowcount
            self.conn.commit()
        return n

    @_retry_on_lock
    def save_briefing(self, ts: str, text: str, article_count: int) -> int:
        with self._write_lock:
            cur = self.conn.execute(
                "INSERT INTO briefings (ts, text, article_count) VALUES (?, ?, ?)",
                (ts, text, article_count),
            )
            self.conn.commit()
            return cur.lastrowid

    @_retry_on_lock
    def get_briefings_for_training(self, limit: int = 100) -> list[dict]:
        cur = self.conn.execute(
            "SELECT id, ts, text, article_count FROM briefings "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [
            {"id": r[0], "ts": r[1], "text": r[2], "article_count": r[3]}
            for r in cur.fetchall()
        ]

    @_retry_on_lock
    def briefing_health(self, window_h: int = 24) -> dict:
        """Heartbeat-briefing pipeline health snapshot.

        The 5h Opus briefing is the analyst's primary synthesised intelligence
        product. When the briefing path goes dark — Opus quota exhausted,
        Claude CLI auth lapsed, persistent network failures, or the heartbeat
        worker itself wedged — the dashboard's ``/api/briefings`` simply
        returns an older briefing and the analyst doesn't realise the digest
        is stale. Existing siblings:

          * ``source_freshness`` — per-collector liveness ("which feeds went
            dark?"); cannot detect a Claude-side outage.
          * ``urgent_queue_health`` — what's queued for the alert path;
            unrelated to the briefing path.
          * ``urgency_label_split*`` — alert calibration; doesn't track
            briefing health.

        This is the missing primitive: one query that turns the briefings
        table into an analyst-actionable "is the briefing path healthy?"
        verdict. Built purely from the ``briefings`` table (no article
        cross-join needed; the table only carries Opus-generated text and is
        never touched by backtest paths — synthetic rows live in
        ``articles`` only).

        ``HEARTBEAT_INTERVAL = 5h`` (see ``daemon.py``) is the production
        cadence, so ``expected_in_window`` is ``window_h / 5``. The verdict
        ladder is deliberately conservative — a single skipped briefing
        (Claude transient empty response, see ``heartbeat_worker``'s
        ``startswith("[analyst]")`` skip) is normal and must not flag DEAD.

        Returns::

            {
              "window_h":           int,
              "last_briefing_age_h": float | None,    # None when no briefings ever
              "count_in_window":    int,
              "expected_in_window": float,             # window_h / 5h cadence
              "verdict":            "HEALTHY" | "STALE" | "DEAD" | "NO_DATA",
            }

        Verdict semantics:
          * ``NO_DATA`` — the briefings table is empty (fresh DB, first cycle
            hasn't fired yet). Distinct from DEAD: the analyst should NOT
            interpret an empty table as an outage on a just-started daemon.
          * ``DEAD`` — last briefing age > 12h. Two full 5h cadences have
            elapsed; the path is materially down and the analyst is
            operating on a stale digest.
          * ``STALE`` — last briefing age between 6h and 12h, OR count is
            below 60% of expected (e.g. <3 in 24h). One cadence skipped or
            persistent under-production — early-warning, not yet DEAD.
          * ``HEALTHY`` — everything else. Last < 6h ago AND count meets the
            60%-of-expected floor.

        Read-only (single MAX + COUNT SELECT). NO DB write — no
        ai_score / ml_score / score_source / urgency mutation. All four
        load-bearing invariants intact by construction.
        """
        window_h = max(int(window_h), 1)
        now = datetime.now(timezone.utc)
        since = (now - timedelta(hours=window_h)).isoformat()

        # MAX(ts) + COUNT(*) bounded by the window in one round trip.
        # Aggregates always yield exactly one row, so ``_expect_row`` is
        # the right guard against shared-connection cursor-state corruption
        # (same discipline as ``stats`` / ``stats_since``).
        row = _expect_row(self.conn.execute(
            "SELECT MAX(ts), COUNT(*) FROM briefings WHERE ts >= ?",
            (since,),
        ))
        newest_in_window, count_in_window = row[0], int(row[1] or 0)

        # When the window is empty, look beyond it for the overall most-
        # recent briefing — needed to distinguish DEAD (one stale briefing
        # >12h ago) from NO_DATA (no briefings ever recorded).
        if newest_in_window is None:
            overall = _expect_row(self.conn.execute(
                "SELECT MAX(ts) FROM briefings"
            ))[0]
        else:
            overall = newest_in_window

        last_age_h: float | None = None
        if overall:
            try:
                ts = datetime.fromisoformat(overall)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                last_age_h = round(
                    max(0.0, (now - ts).total_seconds() / 3600.0), 2
                )
            except (ValueError, TypeError):
                # Unparseable timestamp in the briefings table — treat as
                # NO_DATA rather than crashing. The save path always writes
                # ``datetime.now(timezone.utc).isoformat()`` so a corrupt
                # value is the only way this branch fires.
                last_age_h = None

        # 5h cadence is the daemon's HEARTBEAT_INTERVAL — duplicated here
        # rather than imported (the storage layer must not pull the daemon /
        # collectors graph; mirrors ``_briefing_domain_key`` duplicating
        # ml.features). 60% floor matches the analyst-meaningful "we lost
        # ~2 of 5 expected briefings in 24h is concerning" threshold.
        expected_in_window = round(window_h / 5.0, 2)
        min_count_healthy = max(1, int(0.6 * expected_in_window))

        if last_age_h is None:
            verdict = "NO_DATA"
        elif last_age_h > 12.0:
            verdict = "DEAD"
        elif last_age_h > 6.0 or count_in_window < min_count_healthy:
            verdict = "STALE"
        else:
            verdict = "HEALTHY"

        return {
            "window_h": window_h,
            "last_briefing_age_h": last_age_h,
            "count_in_window": count_in_window,
            "expected_in_window": expected_in_window,
            "verdict": verdict,
        }

    @_retry_on_lock
    def briefing_cadence_trend(
        self, last_n: int = 10, expected_cadence_h: float = 5.0
    ) -> dict:
        """Per-interval briefing cadence trend — the *trend* sibling to
        ``briefing_health``.

        ``briefing_health`` answers "is the MOST RECENT briefing fresh?" in
        verdict form. It is point-in-time: a path that just produced a
        briefing 5h ago reads HEALTHY even if the prior 10 briefings averaged
        9h gaps (Opus quota throttling, Claude CLI auth lapsing intermittently,
        the heartbeat_worker getting blocked on a slow DB read). This is
        early-warning that has so far been invisible to every operator-facing
        surface — the analyst would only learn the path was degrading once it
        flipped to STALE/DEAD, by which point briefings have been missing
        for hours.

        Live evidence (2026-05-24 pull, last 11 intervals in the briefings
        table)::

            5.21, 5.26, 6.26, 10.23, 7.08, 10.26, 5.09, 27.64, 5.21, 5.43, 8.61

        mean ≈ 8.75h (75% slower than the 5h ``HEARTBEAT_INTERVAL``), max
        27.64h (a 5x miss). ``briefing_health`` on this same pull returned
        verdict=HEALTHY (most recent 5.15h ago, count_in_window 3 >= 60%
        floor of 4.8 expected). Both are correct on what they measure; the
        trend is the missing axis.

        Sibling to ``briefing_health`` exactly as ``source_throughput`` is
        sibling to ``source_freshness`` — point-in-time freshness vs the
        first-derivative degradation rate. Same disciplines: pure SELECT,
        ``_LIVE_ONLY_CLAUSE`` not needed (briefings table is Opus-write only;
        never touched by backtest paths) but documented intent below.

        Returns::

            {
              "expected_cadence_h":  float,
              "last_n":              int,
              "n_intervals":         int,         # actual intervals computed
              "intervals_h":         [float,...], # chronological, newest LAST
              "mean_interval_h":     float | None,
              "max_interval_h":      float | None,
              "p50_interval_h":      float | None,
              "drift_pct":           float | None,
                  # (mean - expected) / expected * 100 — positive = slipping
              "verdict": "ON_CADENCE" | "SLIPPING" | "DRIFTING" | "NO_DATA",
            }

        Verdict semantics (chosen to be a strict pre-warning of
        ``briefing_health.STALE``/``DEAD``, never a post-hoc duplicate):

          * ``NO_DATA`` — fewer than 2 intervals (need at least 2 to draw a
            trend; a single gap could be a transient and is already the
            ``briefing_health.STALE`` regime).
          * ``DRIFTING`` — ``mean > 1.5 * expected`` (50%+ slower than
            expected on average — the path is materially slipping, even if
            the most recent briefing happened to fire).
          * ``SLIPPING`` — ``mean > 1.2 * expected`` OR
            ``max > 2.0 * expected`` (early warning: 20% slower on average,
            or a single gap >= 2 cadences — exactly what flipped to DEAD
            once in the live evidence above).
          * ``ON_CADENCE`` — everything else.

        Read-only (single SELECT). NO DB write — no
        ai_score / ml_score / score_source / urgency mutation. All four
        load-bearing invariants intact by construction.
        """
        last_n = max(int(last_n), 1)
        # 0.01h ≈ 36s — well below any plausible production cadence and above
        # the 2-decimal rounding floor used in the return field, so the clamp
        # always yields a strictly-positive returned value (a divide-by-zero
        # in drift_pct cannot occur on a misconfigured caller).
        expected_cadence_h = max(float(expected_cadence_h), 0.01)

        # last_n+1 rows yield last_n intervals (a row pair per gap).
        # Briefings are write-once / append-only by ``save_briefing``, so
        # ORDER BY id DESC is identical to ORDER BY ts DESC except in the
        # exotic case of a clock skew — id-ordering is the authoritative
        # write-order signal (same discipline as ``get_briefings_for_training``).
        rows = self.conn.execute(
            "SELECT ts FROM briefings ORDER BY id DESC LIMIT ?",
            (last_n + 1,),
        ).fetchall()

        # Parse + filter to UTC-aware datetimes. Defensive against corrupt
        # rows (same discipline as ``briefing_health``'s
        # ``fromisoformat`` guard) — bad rows are silently dropped, not
        # raised, so an analytics caller never crashes on a malformed
        # briefings table.
        parsed: list[datetime] = []
        for (ts_str,) in rows:
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(ts_str)
            except (ValueError, TypeError):
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            parsed.append(ts)

        # Reverse to chronological order (oldest first) for natural interval
        # computation. Each interval is ``parsed[i+1] - parsed[i]`` — the
        # gap an analyst would draw on a timeline.
        parsed.reverse()
        intervals_h: list[float] = []
        for i in range(len(parsed) - 1):
            delta_s = (parsed[i + 1] - parsed[i]).total_seconds()
            # Defensive non-negative clamp — a future clock-skew + id-order
            # mismatch must never produce a negative interval that breaks
            # the mean / drift_pct math.
            intervals_h.append(round(max(0.0, delta_s / 3600.0), 2))

        n_intervals = len(intervals_h)
        if n_intervals < 2:
            return {
                "expected_cadence_h": round(expected_cadence_h, 2),
                "last_n": last_n,
                "n_intervals": n_intervals,
                "intervals_h": intervals_h,
                "mean_interval_h": None,
                "max_interval_h": None,
                "p50_interval_h": None,
                "drift_pct": None,
                "verdict": "NO_DATA",
            }

        mean_h = sum(intervals_h) / n_intervals
        max_h = max(intervals_h)
        # p50 = median. Sorted list; ``//2`` is exactly right for both even
        # and odd n_intervals since we just want the middle ordered value
        # (no interpolation — operator-readable round-trip to a real bucket).
        sorted_h = sorted(intervals_h)
        p50_h = sorted_h[n_intervals // 2]
        drift_pct = round(
            (mean_h - expected_cadence_h) / expected_cadence_h * 100.0, 1
        )

        # Verdict ladder: most-severe first (same discipline as
        # ``ml_training_health``'s NO_DATA > DEAD > STALE > … precedence).
        if mean_h > 1.5 * expected_cadence_h:
            verdict = "DRIFTING"
        elif (mean_h > 1.2 * expected_cadence_h
              or max_h > 2.0 * expected_cadence_h):
            verdict = "SLIPPING"
        else:
            verdict = "ON_CADENCE"

        return {
            "expected_cadence_h": round(expected_cadence_h, 2),
            "last_n": last_n,
            "n_intervals": n_intervals,
            "intervals_h": intervals_h,
            "mean_interval_h": round(mean_h, 2),
            "max_interval_h": round(max_h, 2),
            "p50_interval_h": round(p50_h, 2),
            "drift_pct": drift_pct,
            "verdict": verdict,
        }

    @_retry_on_lock
    def briefing_text_overlap_trend(self, last_n: int = 6) -> dict:
        """Per-pair token Jaccard across the ``last_n`` most recent briefings —
        the *content-staleness* sibling to ``briefing_cadence_trend``.

        ``briefing_cadence_trend`` asks "are briefings firing on schedule?";
        this asks "if a briefing fires, is it carrying fresh content or
        recapping the prior one?". A 5h Opus briefing technically ON_CADENCE
        can still be functionally useless to the analyst if it recaps the
        same handful of events as the previous one — they have already been
        told everything. This surfaces that pattern as a single verdict.

        Sibling discipline to ``briefing_cadence_trend`` / ``briefing_health``:
        pure SELECT, ``_LIVE_ONLY_CLAUSE`` not needed (briefings table is
        Opus-write only; never touched by backtest paths). Read-only, no
        ai_score / ml_score / score_source / urgency mutation.

        Tokenisation: lowercased alphanumeric runs of length >= 5. Length
        floor strips common stop words and the heavy noise tail of 1-4 char
        tokens (tickers, "the", "and", numbers, separators) so the Jaccard
        actually reflects whether the same companies / events are being
        re-discussed, not whether both briefings say "the".

        Returns::

            {
              "last_n":         int,
              "n_pairs":        int,
              "pair_jaccards":  [float, ...],  # chronological, newest LAST
              "mean_jaccard":   float | None,
              "max_jaccard":    float | None,
              "verdict": "FRESH" | "WARMING" | "REPETITIVE" | "NO_DATA",
            }

        Verdict semantics (mirrors ``briefing_cadence_trend``'s
        most-severe-first ladder):
          * ``NO_DATA``    — fewer than 2 briefings (need a pair).
          * ``REPETITIVE`` — mean > 0.60 OR max > 0.75. Briefings are
            substantially recycling content.
          * ``WARMING``    — mean > 0.45. Early-warning: notable overlap
            but not yet a recap problem.
          * ``FRESH``      — everything else.
        """
        last_n = max(int(last_n), 2)

        rows = self.conn.execute(
            "SELECT text FROM briefings ORDER BY id DESC LIMIT ?",
            (last_n,),
        ).fetchall()

        # Reverse to chronological order (oldest first) — pairs are
        # then (older, newer), so ``pair_jaccards[-1]`` is the freshest
        # comparison. Defensive against corrupt rows: empty / None text
        # silently yields an empty token set (no crash on a malformed
        # briefings row, same discipline as ``briefing_cadence_trend``).
        texts: list[str] = [(r[0] or "") for r in rows]
        texts.reverse()

        token_re = re.compile(r"[a-z0-9]{5,}")
        token_sets: list[frozenset] = [
            frozenset(token_re.findall(t.lower())) for t in texts
        ]

        pair_jaccards: list[float] = []
        for i in range(len(token_sets) - 1):
            a, b = token_sets[i], token_sets[i + 1]
            if not a and not b:
                pair_jaccards.append(0.0)
                continue
            inter = len(a & b)
            union = len(a | b)
            pair_jaccards.append(
                round(inter / union, 3) if union else 0.0
            )

        n_pairs = len(pair_jaccards)
        if n_pairs < 1:
            return {
                "last_n": last_n,
                "n_pairs": 0,
                "pair_jaccards": [],
                "mean_jaccard": None,
                "max_jaccard": None,
                "verdict": "NO_DATA",
            }

        mean_j = sum(pair_jaccards) / n_pairs
        max_j = max(pair_jaccards)

        if mean_j > 0.60 or max_j > 0.75:
            verdict = "REPETITIVE"
        elif mean_j > 0.45:
            verdict = "WARMING"
        else:
            verdict = "FRESH"

        return {
            "last_n": last_n,
            "n_pairs": n_pairs,
            "pair_jaccards": pair_jaccards,
            "mean_jaccard": round(mean_j, 3),
            "max_jaccard": round(max_j, 3),
            "verdict": verdict,
        }

    @_retry_on_lock
    def briefing_length_trend(self, last_n: int = 10) -> dict:
        """Per-briefing text-length trend — the *output-density* sibling to
        ``briefing_cadence_trend`` and ``briefing_text_overlap_trend``.

        ``briefing_cadence_trend`` asks "are briefings firing on schedule?";
        ``briefing_text_overlap_trend`` asks "if a briefing fires, is it
        fresh or recapping prior content?"; this asks "if a briefing fires
        with fresh content, is it as DETAILED as it used to be, or is Opus
        producing materially shorter output per cycle?". A briefing that
        comes in 30% shorter than the recent baseline is a real signal of
        Opus quota throttling, prompt-context truncation, or response
        cutoff — all conditions an analyst should know about because the
        digest stops covering the news exhaustively even when it fires
        ON_CADENCE with FRESH content.

        Sibling discipline: pure SELECT, ``_LIVE_ONLY_CLAUSE`` not needed
        (briefings table is Opus-write only; never touched by backtest
        paths). Read-only, no ai_score / ml_score / score_source / urgency
        mutation. All four load-bearing invariants intact by construction.

        Returns::

            {
              "last_n":         int,
              "n_briefings":    int,
              "lengths":        [int, ...],     # chronological, newest LAST
              "median_length":  int | None,     # over the full window
              "min_length":     int | None,
              "max_length":     int | None,
              "recent_median":  int | None,     # newest half
              "older_median":   int | None,     # older half
              "shrink_ratio":   float | None,   # recent_median / older_median
              "verdict": "STABLE" | "SHRINKING" | "GROWING" | "NO_DATA",
            }

        Verdict semantics:
          * ``NO_DATA`` — fewer than 4 briefings (need a meaningful split
            into "newer half" vs "older half"; below this any verdict would
            be noise).
          * ``SHRINKING`` — ``recent_median <= 0.7 * older_median`` (30%+
            shrink). The Opus output density has dropped materially —
            quota throttling, truncation, or prompt-context loss.
          * ``GROWING`` — ``recent_median >= 1.3 * older_median`` (30%+
            growth). Either Opus prompt is producing more content, or
            article coverage has gotten broader. Surfaced for symmetry —
            an analyst surprised by suddenly-longer briefings can use this
            to spot prompt drift in the other direction.
          * ``STABLE`` — everything in between.
        """
        last_n = max(int(last_n), 1)

        # Briefings are append-only (``save_briefing``), so ORDER BY id DESC
        # is the authoritative write-order signal — same discipline as
        # ``briefing_cadence_trend`` / ``briefing_text_overlap_trend``.
        # LENGTH(text) computed in SQL so we don't decompress / fetch the
        # full body for what is purely a size statistic.
        rows = self.conn.execute(
            "SELECT LENGTH(text) FROM briefings "
            "ORDER BY id DESC LIMIT ?",
            (last_n,),
        ).fetchall()

        # Reverse to chronological order (oldest first) so the split into
        # older-half / newer-half is unambiguous.
        lengths: list[int] = []
        for (n,) in rows:
            if n is None:
                continue
            try:
                lengths.append(int(n))
            except (TypeError, ValueError):
                continue
        lengths.reverse()

        n_briefings = len(lengths)
        if n_briefings < 4:
            return {
                "last_n": last_n,
                "n_briefings": n_briefings,
                "lengths": lengths,
                "median_length": None,
                "min_length": None,
                "max_length": None,
                "recent_median": None,
                "older_median": None,
                "shrink_ratio": None,
                "verdict": "NO_DATA",
            }

        sorted_all = sorted(lengths)
        median_full = sorted_all[n_briefings // 2]

        # Split point: even halves (or older half is 1 larger on odd counts).
        # The newer half is what the analyst cares about — the recent state.
        split = n_briefings // 2
        older = lengths[:split]
        newer = lengths[split:]

        def _median(xs: list[int]) -> int:
            return sorted(xs)[len(xs) // 2]

        older_median = _median(older)
        recent_median = _median(newer)

        # Guard divide-by-zero (briefings.text is NOT NULL in the schema, so
        # LENGTH > 0 in practice — but a future malformed row could surface
        # zero. Treat that as NO_DATA rather than crashing the analytics).
        if older_median <= 0:
            return {
                "last_n": last_n,
                "n_briefings": n_briefings,
                "lengths": lengths,
                "median_length": int(median_full),
                "min_length": min(lengths),
                "max_length": max(lengths),
                "recent_median": int(recent_median),
                "older_median": int(older_median),
                "shrink_ratio": None,
                "verdict": "NO_DATA",
            }

        shrink_ratio = round(recent_median / older_median, 3)

        # Verdict ladder — most-severe first (same discipline as the sibling
        # trend methods). SHRINKING is the analyst-actionable verdict here:
        # GROWING is informational only (the digest got bigger).
        if shrink_ratio <= 0.70:
            verdict = "SHRINKING"
        elif shrink_ratio >= 1.30:
            verdict = "GROWING"
        else:
            verdict = "STABLE"

        return {
            "last_n": last_n,
            "n_briefings": n_briefings,
            "lengths": lengths,
            "median_length": int(median_full),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "recent_median": int(recent_median),
            "older_median": int(older_median),
            "shrink_ratio": shrink_ratio,
            "verdict": verdict,
        }

    @_retry_on_lock
    def briefing_article_count_trend(self, last_n: int = 10) -> dict:
        """Per-briefing INPUT-pool-size trend — distinct from the existing
        trend siblings.

        ``briefing_cadence_trend``       asks "are briefings firing on schedule?"
        ``briefing_text_overlap_trend``  asks "is the OUTPUT fresh or recapping?"
        ``briefing_length_trend``        asks "is the OUTPUT text getting shorter?"

        None answer "is the INPUT POOL feeding Opus shrinking?" — the
        ``briefings.article_count`` column records how many candidate articles
        each briefing was built from. A shrinking input pool while cadence and
        output length look HEALTHY is a distinct failure mode: the briefing
        pipeline is healthy and Opus is writing the same length of digest, but
        from materially fewer articles per cycle — the analyst gets a normal-
        looking briefing that covered far less of the news than it usually does.

        Possible causes the input-pool trend catches that the other three miss:
          * collectors / scoring pipeline producing fewer high-score articles
            in the briefing-input window (e.g. a 24h cycle with most workers
            silent — the cadence-still-firing-but-feeding-thin case);
          * stricter pre-filters in ``get_top_for_briefing`` dropping more
            rows (e.g. an over-tight quote-widget / recap regex that catches
            a wider net than intended);
          * the heartbeat worker timing out on the pre-briefing fetch and
            falling back to a smaller candidate list.

        Sibling discipline (mirrors ``briefing_length_trend`` verbatim): pure
        SELECT, ``_LIVE_ONLY_CLAUSE`` not needed (the ``briefings`` table is
        Opus-write only; never touched by backtest paths). Read-only; NO DB
        write — no ai_score / ml_score / score_source / urgency mutation. All
        four load-bearing invariants intact by construction.

        Returns::

            {
              "last_n":         int,
              "n_briefings":    int,
              "counts":         [int, ...],     # chronological, newest LAST
              "median_count":   int | None,     # over the full window
              "min_count":      int | None,
              "max_count":      int | None,
              "recent_median":  int | None,     # newer half
              "older_median":   int | None,     # older half
              "shrink_ratio":   float | None,   # recent_median / older_median
              "verdict": "STABLE" | "SHRINKING" | "GROWING" | "NO_DATA",
            }

        Verdict semantics (same conservative ladder as ``briefing_length_trend``;
        chosen to avoid noisy flags on a 2-3 briefing sample):
          * ``NO_DATA``   — fewer than 4 briefings, OR older_median == 0
                            (cannot compute a ratio).
          * ``SHRINKING`` — ``recent_median <= 0.7 * older_median`` (30%+
                            shrink). The candidate pool is materially smaller
                            in the newer half.
          * ``GROWING``   — ``recent_median >= 1.3 * older_median``. Pool got
                            broader — informational, surfaced for symmetry.
          * ``STABLE``    — everything in between.
        """
        last_n = max(int(last_n), 1)

        rows = self.conn.execute(
            "SELECT article_count FROM briefings "
            "ORDER BY id DESC LIMIT ?",
            (last_n,),
        ).fetchall()

        # Defense against malformed rows: drop NULL / non-int values silently
        # (mirrors briefing_length_trend's discipline). The schema has
        # ``article_count INTEGER NOT NULL DEFAULT 0`` so the live path
        # always emits a clean int; this guards a future malformed write.
        counts: list[int] = []
        for (c,) in rows:
            try:
                if c is None:
                    continue
                counts.append(int(c))
            except (TypeError, ValueError):
                continue
        counts.reverse()  # chronological order — oldest first, newest LAST
        n_briefings = len(counts)

        if n_briefings < 4:
            return {
                "last_n": last_n,
                "n_briefings": n_briefings,
                "counts": counts,
                "median_count": None,
                "min_count": min(counts) if counts else None,
                "max_count": max(counts) if counts else None,
                "recent_median": None,
                "older_median": None,
                "shrink_ratio": None,
                "verdict": "NO_DATA",
            }

        sorted_all = sorted(counts)
        median_full = sorted_all[n_briefings // 2]

        split = n_briefings // 2
        older = counts[:split]
        newer = counts[split:]

        def _median(xs: list[int]) -> int:
            return sorted(xs)[len(xs) // 2]

        older_median = _median(older)
        recent_median = _median(newer)

        # NO_DATA fallback when older_median is zero so shrink_ratio cannot
        # divide-by-zero (the article_count column defaults to 0 in the
        # schema so a freshly-deployed DB with placeholder rows can produce
        # this case — same defensive branch as briefing_length_trend).
        if older_median <= 0:
            return {
                "last_n": last_n,
                "n_briefings": n_briefings,
                "counts": counts,
                "median_count": int(median_full),
                "min_count": min(counts),
                "max_count": max(counts),
                "recent_median": int(recent_median),
                "older_median": int(older_median),
                "shrink_ratio": None,
                "verdict": "NO_DATA",
            }

        shrink_ratio = round(recent_median / older_median, 3)

        if shrink_ratio <= 0.70:
            verdict = "SHRINKING"
        elif shrink_ratio >= 1.30:
            verdict = "GROWING"
        else:
            verdict = "STABLE"

        return {
            "last_n": last_n,
            "n_briefings": n_briefings,
            "counts": counts,
            "median_count": int(median_full),
            "min_count": min(counts),
            "max_count": max(counts),
            "recent_median": int(recent_median),
            "older_median": int(older_median),
            "shrink_ratio": shrink_ratio,
            "verdict": verdict,
        }

    @_retry_on_lock
    def urgent_count_per_briefing_window_trend(self, last_n: int = 10) -> dict:
        """Per-briefing-window URGENT-row-count trend — the urgent-flow sibling
        to the four existing briefing-trend primitives.

        ``briefing_cadence_trend``          asks "are briefings firing on schedule?"
        ``briefing_text_overlap_trend``     asks "is the OUTPUT fresh or recapping?"
        ``briefing_length_trend``           asks "is the OUTPUT text getting shorter?"
        ``briefing_article_count_trend``    asks "is the INPUT POOL feeding Opus shrinking?"

        None of them answers "is the URGENT-flow per briefing window surging or
        quieting?". An analyst tracking N briefings worth of urgent rate
        cycle-over-cycle has no single primitive — they have to query
        ``stats_since`` per window manually and compose the trend by eye. A
        surge from 5→15 urgent rows per 5h window is exactly the operator-
        actionable "the wire heated up" signal that should fire without
        needing the analyst to spot it in the Discord channel.

        For each consecutive pair of briefings (older, newer) in the last N+1
        records, counts ``urgency>=1`` rows with ``first_seen`` in
        ``[older.ts, newer.ts)``. Each yielded count corresponds to ONE
        completed briefing-cadence window. Returns the trend in the same
        shape as ``briefing_*_trend`` siblings.

        Sibling discipline (mirrors ``briefing_article_count_trend`` verbatim):
        single SELECT against ``briefings`` for the cadence anchors plus one
        aggregate SELECT against ``articles`` with ``_LIVE_ONLY_CLAUSE`` so
        synthetic backtest/opus rows can never inflate the per-window urgent
        count. Read-only; NO DB write — no ai_score / ml_score / score_source
        / urgency mutation. All four load-bearing invariants intact by
        construction.

        Returns::

            {
              "last_n":         int,
              "n_windows":      int,
              "windows": [
                {
                  "older_ts":    str,
                  "newer_ts":    str,
                  "duration_h":  float,
                  "urgent_count": int,
                },
                ...                              # chronological, newest LAST
              ],
              "counts":         [int, ...],     # urgent_count per window
              "median_count":   int | None,
              "min_count":      int | None,
              "max_count":      int | None,
              "recent_median":  int | None,     # newer half
              "older_median":   int | None,     # older half
              "surge_ratio":    float | None,   # recent_median / older_median
              "verdict": "STABLE" | "SURGING" | "QUIETING" | "NO_DATA",
            }

        Verdict semantics (same conservative ladder as
        ``briefing_article_count_trend``, but oriented around the urgent-flow
        analyst signal — SURGING is the actionable verdict, QUIETING is
        informational symmetry):

          * ``NO_DATA``   — fewer than 4 windows, OR older_median == 0
                            (cannot compute a ratio).
          * ``SURGING``   — ``recent_median >= 1.3 * older_median``. The urgent
                            flow per window has materially picked up; the wire
                            is hotter than the older baseline.
          * ``QUIETING``  — ``recent_median <= 0.7 * older_median``. Urgent
                            flow has dropped — pipeline issue, or a genuine
                            calm wire.
          * ``STABLE``    — everything in between.
        """
        last_n = max(int(last_n), 1)

        # last_n+1 briefings yield last_n windows (a window = the gap between
        # two consecutive briefings). Briefings are append-only (save_briefing)
        # so ORDER BY id DESC is the authoritative write-order signal — same
        # discipline as briefing_cadence_trend / briefing_length_trend.
        rows = self.conn.execute(
            "SELECT ts FROM briefings ORDER BY id DESC LIMIT ?",
            (last_n + 1,),
        ).fetchall()

        # Parse + filter to UTC-aware datetimes, defensive against corrupt rows
        # (same discipline as briefing_cadence_trend).
        parsed: list[datetime] = []
        for (ts_str,) in rows:
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(ts_str)
            except (ValueError, TypeError):
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            parsed.append(ts)

        # Reverse to chronological (oldest first). Pair (parsed[i], parsed[i+1])
        # = ONE window — first_seen in [older.ts, newer.ts).
        parsed.reverse()
        n_windows = max(0, len(parsed) - 1)

        if n_windows < 4:
            # Same NO_DATA discipline as the sibling trend methods — below 4
            # windows any verdict would be noise (older_half / newer_half split
            # has < 2 samples each).
            return {
                "last_n": last_n,
                "n_windows": n_windows,
                "windows": [],
                "counts": [],
                "median_count": None,
                "min_count": None,
                "max_count": None,
                "recent_median": None,
                "older_median": None,
                "surge_ratio": None,
                "verdict": "NO_DATA",
            }

        windows: list[dict] = []
        counts: list[int] = []
        for i in range(n_windows):
            older = parsed[i]
            newer = parsed[i + 1]
            older_iso = older.isoformat()
            newer_iso = newer.isoformat()
            duration_h = max(0.0, (newer - older).total_seconds() / 3600.0)
            # ``_LIVE_ONLY_CLAUSE`` keeps synthetic backtest/opus injection rows
            # out of the per-window urgent count — exact discipline the
            # aggregate ``stats_since`` carries (CLAUDE.md §5 load-bearing
            # invariant #1).
            cur = self.conn.execute(
                "SELECT COUNT(*) FROM articles "
                "WHERE urgency>=1 AND first_seen >= ? AND first_seen < ? "
                f"AND {_LIVE_ONLY_CLAUSE}",
                (older_iso, newer_iso),
            )
            row = _expect_row(cur)
            count = int(row[0] or 0)
            windows.append({
                "older_ts": older_iso,
                "newer_ts": newer_iso,
                "duration_h": round(duration_h, 2),
                "urgent_count": count,
            })
            counts.append(count)

        sorted_all = sorted(counts)
        median_full = sorted_all[n_windows // 2]

        split = n_windows // 2
        older_half = counts[:split]
        newer_half = counts[split:]

        def _median(xs: list[int]) -> int:
            return sorted(xs)[len(xs) // 2]

        older_median = _median(older_half)
        recent_median = _median(newer_half)

        # NO_DATA when older_median is zero so surge_ratio cannot
        # divide-by-zero (the older half can legitimately be all-zero on a
        # quiet wire — same defensive branch as briefing_article_count_trend
        # / briefing_length_trend's older_median <= 0 guard).
        if older_median <= 0:
            return {
                "last_n": last_n,
                "n_windows": n_windows,
                "windows": windows,
                "counts": counts,
                "median_count": int(median_full),
                "min_count": min(counts),
                "max_count": max(counts),
                "recent_median": int(recent_median),
                "older_median": int(older_median),
                "surge_ratio": None,
                "verdict": "NO_DATA",
            }

        surge_ratio = round(recent_median / older_median, 3)

        # Verdict ladder — most-severe-first (mirrors briefing_*_trend
        # discipline). SURGING is the analyst-actionable verdict; QUIETING is
        # informational symmetry.
        if surge_ratio >= 1.30:
            verdict = "SURGING"
        elif surge_ratio <= 0.70:
            verdict = "QUIETING"
        else:
            verdict = "STABLE"

        return {
            "last_n": last_n,
            "n_windows": n_windows,
            "windows": windows,
            "counts": counts,
            "median_count": int(median_full),
            "min_count": min(counts),
            "max_count": max(counts),
            "recent_median": int(recent_median),
            "older_median": int(older_median),
            "surge_ratio": surge_ratio,
            "verdict": verdict,
        }

    @_retry_on_lock
    def count_unscored(self, min_kw: float = 0.0) -> int:
        """Count articles still pending the scorer. Mirrors ``get_unscored``'s
        filter (ai_score=0 AND ml_score IS NULL AND kw_score>=min_kw AND
        live-only) without fetching/decompressing the rows themselves —
        intended for the scorer worker's "remaining backlog" status line."""
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM articles "
            f"WHERE ai_score=0 AND ml_score IS NULL AND kw_score>=? "
            f"AND {_LIVE_ONLY_CLAUSE}",
            (min_kw,),
        )
        return _expect_row(cur)[0]

    @_retry_on_lock
    def get_unscored(self, limit: int = 500, min_kw: float = 0.5) -> list:
        """Get articles that haven't been AI-scored yet.

        Excludes backtest replays and Opus annotation rows — those are training
        artefacts, not live news, and must not enter the live scoring path.

        ``ml_score IS NULL`` excludes articles the model has already scored
        with confidence. Without this, every 30s scorer cycle re-fetched the
        full ML-scored backlog and re-ran inference on it, while the LLM
        retry path (which writes ai_score, not ml_score) still works: a
        Sonnet-failed article keeps both ai_score=0 and ml_score=NULL, so it
        gets re-routed on the next cycle.

        ``published`` and ``first_seen`` are returned (not just id/title/body)
        because two downstream consumers need the article's age:

          - ``ml/features.py::extract_features`` derives 5 temporal features
            (hour/dow cyclic encodings, days_since_published) from
            ``published``. ``_fetch_training_data`` passes the real value at
            train time; if inference omits it the parser falls back to
            ``now()`` and those 5 features become a constant — a silent
            train/serve skew on every scored article.
          - ``watchers/urgency_scorer.score_batch`` derives each article's
            ``age_hours`` via ``_article_age_hours`` (reads ``published`` /
            ``first_seen``). That feeds both the Sonnet prompt's staleness
            rule and the hard ``STALE_HOURS``/``STALE_SCORE_CAP`` clamp.
            Without these fields every article looks 0h old and the entire
            staleness system is inert on the live path.
        """
        cur = self.conn.execute(
            "SELECT id, url, title, source, full_text, published, first_seen "
            "FROM articles "
            f"WHERE ai_score=0 AND ml_score IS NULL AND kw_score>=? "
            f"AND {_LIVE_ONLY_CLAUSE} "
            "ORDER BY kw_score DESC LIMIT ?",
            (min_kw, limit),
        )
        rows = cur.fetchall()
        return [
            {"_id": r[0], "link": r[1], "title": r[2], "source": r[3],
             "summary": decompress(r[4]) if r[4] else "",
             "published": r[5] or "", "first_seen": r[6] or ""}
            for r in rows
        ]

    def prefloor_pseudo_articles(
        self, batch: list[dict]
    ) -> tuple[list[dict], int]:
        """Pre-floor quote-widget / recap-template pseudo-articles BEFORE scoring.

        Partitions ``batch`` into (real, pseudo). Pseudo-articles get
        ``ml_score=0.01``, ``urgency=0``, ``score_source='ml'`` via
        ``update_ml_scores_batch`` so they exit the unscored queue but never
        warrant urgency=1 regardless of what the urgency head would predict.
        Returns ``(real_articles, n_pre_floored)``.

        Single source of truth for the ML-path pre-floor, called by:
          * ``score_pending`` — the one-shot in-store inference driver
          * ``daemon.scorer_worker`` — the production worker loop

        Without this defense on the daemon worker path, a 48h audit
        (2026-05-26) found 10 ML-confident quote-widget / recap-template rows
        reached ``urgency=2`` (alerted state). The alert gates caught them
        before Discord push, but only after polluting urgent_queue_health and
        burning alert worker cycles. Mirrors the LLM-path pre-floor in
        ``watchers.urgency_scorer.score_batch``.

        Lazy import: ``watchers.alert_agent → ml.features``; no cycle back into
        storage.
        """
        from watchers.alert_agent import (
            _looks_like_quote_widget,
            _looks_like_recap_template,
            _looks_like_stocktwits_chatter,
        )
        # Non-English title gate — separate module so the alert-side
        # mirror can be added later without a circular import. The
        # multi-language press-release noise (Spanish/French/German/
        # Portuguese copies of the same corporate wire) is a distinct
        # surface from the four sibling gates above; the ML urgency head
        # over-scores foreign-language titles to ml_score >= 9 because
        # it has no language-ID prior. Live evidence (2026-05-30 30d
        # audit): 5 of 8 PR Newswire BREAKING alerts on the consumer's
        # Discord were non-English wire copies. See
        # ``watchers.non_english_filter`` docstring for the full
        # discriminator + sample evidence.
        from watchers.non_english_filter import looks_non_english
        # YouTube-share-card SEO-mill rows — GoogleNews/Mshale,
        # GoogleNews/Fathom Journal, etc. emit page titles carrying a
        # parenthesised opaque YouTube video ID at end-of-title. ML
        # urgency head over-scores them to ml_score>=9 (held-ticker
        # density + clickbait verbs). Live evidence (2026-05-30, 30d):
        # 293 corpus matches, 6 reached urgency=2 (alerted). See
        # ``watchers.youtube_mill_filter.looks_like_youtube_mill`` for
        # the full discriminator + evidence — same defense-in-depth
        # class as ``looks_non_english`` above.
        from watchers.youtube_mill_filter import looks_like_youtube_mill
        pre_floor: list[tuple[str, float, int]] = []
        real: list[dict] = []
        for art in batch:
            aid = art.get("_id")
            if not aid:
                continue
            if _looks_like_quote_widget(art):
                pre_floor.append((aid, 0.01, 0))
                continue
            hit, _name = _looks_like_recap_template(art)
            if hit:
                pre_floor.append((aid, 0.01, 0))
                continue
            # Raw stocktwits forum-chatter rows ("$MU lol", "$MU yum") —
            # source + title-length + no-news-keyword gate. Live evidence
            # (2026-05-27, 24h): 2444 raw stocktwits rows, 95 of the
            # ml_score>=8 set < 30 chars. See _looks_like_stocktwits_chatter
            # docstring for the full discriminator. SSOT in alert_agent so a
            # threshold tweak engages across all three pre-floor surfaces
            # (this store helper + urgency_scorer + the alert-side gate).
            if _looks_like_stocktwits_chatter(art):
                pre_floor.append((aid, 0.01, 0))
                continue
            # Non-English title — Spanish/French/German/Portuguese press-
            # release copies that the urgency head over-scores. Two-signal
            # AND (non-English stopword + diacritic) keeps precision against
            # English headlines with one accented name ("São Paulo").
            if looks_non_english(art):
                pre_floor.append((aid, 0.01, 0))
                continue
            # YouTube-share-card SEO mill — parenthesised opaque
            # 8-15 char alnum ID mixing lower+upper+digit, anchored
            # at end-of-title before an optional ``- Publisher`` tail.
            # 293 corpus matches in 30 days, 6 alerted; zero false
            # positives on the curated must-survive corpus.
            if looks_like_youtube_mill(art):
                pre_floor.append((aid, 0.01, 0))
                continue
            real.append(art)
        if pre_floor:
            self.update_ml_scores_batch(pre_floor)
        return real, len(pre_floor)

    def score_pending(self, batch_size: int = 500) -> int:
        """Run ML inference on all unscored articles. Returns count scored.
        Non-blocking: if another caller already holds the inference lock, returns 0."""
        if not _INFER_LOCK.acquire(blocking=False):
            return 0
        total = 0
        try:
            # Lazy import to avoid circular imports at module load
            from ml.inference import score_articles
            while True:
                batch = self.get_unscored(limit=batch_size, min_kw=0.0)
                if not batch:
                    break

                # Pre-floor recap-template / quote-widget pseudo-articles
                # BEFORE inference. They never warrant urgency=1 regardless
                # of what the urgency head predicts, and skipping inference
                # also saves GPU cycles on rows that would just be floored.
                real_batch, n_pre_floored = self.prefloor_pseudo_articles(batch)
                if n_pre_floored:
                    total += n_pre_floored

                if not real_batch:
                    # The whole batch was pseudo-articles; the rows are now
                    # ml_score=0.01 so get_unscored will not re-fetch them.
                    # Loop continues to drain the rest of the backlog; if the
                    # next get_unscored is empty the outer ``if not batch``
                    # breaks. Nothing to score this iteration.
                    continue

                scores = score_articles(real_batch)
                updates = []
                ts_updates = []
                for art, sc in zip(real_batch, scores):
                    aid = art.get("_id")
                    if not aid:
                        continue
                    # Persist time_sensitivity for every article the model
                    # produced a real prediction for — including ones routed
                    # to the LLM for relevance scoring. The two heads are
                    # independent, so the time_sensitivity estimate is still
                    # useful for the briefing ranker.
                    if not sc.needs_llm or sc.rel_std < 99:
                        ts_updates.append((aid, sc.time_sensitivity))
                    # Skip articles flagged for LLM — leave ai_score=0 so the
                    # scorer_worker (Sonnet path) picks them up.
                    if sc.needs_llm:
                        continue
                    final = max(sc.relevance, sc.urgency, 0.01)
                    is_urgent = 1 if sc.urgency >= 8.0 else 0
                    updates.append((aid, final, is_urgent))
                if ts_updates:
                    self.update_time_sensitivity_batch(ts_updates)
                if not updates:
                    # No batch member was scoreable (model not fitted, or
                    # the entire remaining backlog is uncertain). Stop —
                    # otherwise we'd re-fetch the same rows forever.
                    break
                # Write model predictions to ml_score (NOT ai_score) so the
                # trainer doesn't ingest its own outputs as ground-truth labels.
                self.update_ml_scores_batch(updates)
                total += len(updates)
                if total // 1000 != (total - len(updates)) // 1000:
                    print(f"[score_pending] {total} scored so far...")
        finally:
            _INFER_LOCK.release()
        return total

    @_retry_on_lock
    def get_unalerted_urgent(self, limit: int = 50) -> list:
        """Get articles scored urgent but not yet alerted.

        ai_score is LLM-only; model self-predictions live in ml_score (see
        update_ml_scores_batch). A model-flagged urgent item has ai_score=0
        and ml_score>0, so COALESCE picks the meaningful number for the alert
        prompt (otherwise it'd show [score=0] for every NN-detected alert).

        ``limit`` caps the result — the alerter only consumes ALERT_BATCH_SIZE
        (5) per cycle, so an unbounded query during a backlog surge would
        fetch (and decompress) thousands of rows uselessly.

        ``full_text`` is decompressed into a ``summary`` field (capped at 600
        chars) so the alert LLM can ground its CONTEXT line on real article
        content instead of inventing background from the headline alone.
        """
        cutoff_24h = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        cur = self.conn.execute(
            "SELECT id, url, title, source, "
            "COALESCE(NULLIF(ai_score, 0), ml_score, 0) AS score, full_text, "
            "first_seen, published, ai_score "
            "FROM articles "
            f"WHERE urgency=1 AND {_LIVE_ONLY_CLAUSE} "
            # Primary SQL guard: only articles first collected in the last 24h.
            # This is a fast indexed filter; the alerter re-checks published date
            # in Python for articles whose RSS date predates first_seen.
            "AND first_seen >= ? "
            "ORDER BY score DESC LIMIT ?",
            (cutoff_24h, limit),
        )
        rows = cur.fetchall()
        return [{"_id": r[0], "link": r[1], "title": r[2], "source": r[3],
                 "ai_score": r[4],
                 "summary": (decompress(r[5])[:600] if r[5] else ""),
                 "first_seen": r[6],
                 "published": r[7],
                 # True iff this row carries a real LLM ground-truth label (raw
                 # ai_score > 0). Model self-predictions go to ml_score and
                 # NEVER ai_score (invariant #2), so a falsy raw ai_score means
                 # the displayed COALESCEd ``ai_score`` field above came from
                 # ml_score — an UNVERIFIED local-model urgent call. The ML
                 # urgency head demonstrably over-scores forum/wiki/social /
                 # recap-template rows; the briefing already exposes this via
                 # its [model] tag (see get_top_for_briefing). The alert path
                 # is the analyst's MORE time-critical product, so it should
                 # carry the same calibration signal — alert_agent._fmt reads
                 # this key and Sonnet hedges the CONTEXT/IMPACT line for
                 # unverified urgent rows. Read-side only — does NOT change
                 # which rows are returned (urgency=1 + 24h freshness + live-
                 # only clause are unchanged), and the existing ``ai_score``
                 # field (COALESCEd score, used by every existing caller and
                 # the score= line) is byte-unchanged. Same shape as
                 # get_top_for_briefing's _llm_vetted addition (66c349f).
                 "_llm_vetted": bool(r[8])}
                for r in rows]

    @_retry_on_lock
    def get_top_for_briefing(self, hours: int = 5, limit: int = 50) -> list:
        """Get highest-scoring articles from last N hours for the heartbeat briefing.

        Articles with a known publish date older than 24 hours are excluded so
        that GDELT-indexed stale articles don't surface as breaking news.

        Returned rows include ``time_sensitivity`` (0..1, ML-predicted, may be
        None if the article hasn't been scored yet). The briefing ranker uses
        it for intelligent recency decay:

            effective_score = ai_score * (0.5 ^ (age_hours * time_sensitivity / 12))
                              -- set by ML, not hardcoded

        i.e. ts=1.0 halves the score every 12h, ts=0.0 disables decay entirely.
        We deliberately do NOT apply that decay here — we just return ai_score
        unchanged so consumers can pick their own policy (or skip decay).
        """
        now = datetime.now(timezone.utc)
        since = (now - timedelta(hours=hours)).isoformat()
        pub_cutoff_dt = now - timedelta(hours=24)
        pub_cutoff = pub_cutoff_dt.isoformat()
        # Sort by the effective score (LLM label if present, else model score).
        # Without the COALESCE, ML-flagged urgent articles (ai_score=0, ml_score=9)
        # land at the bottom of the briefing — even after being alerted. The
        # briefing should surface them prominently alongside LLM-vetted items.
        #
        # The ``published >= ?`` SQL clause is only a cheap pre-filter — it is
        # correct for ISO-formatted dates but not for RFC822 ones (RSS), so the
        # authoritative 24h staleness check is applied in Python below via
        # _published_older_than. ``limit * 3`` is fetched so that dropping
        # stale RFC822 rows still leaves enough to fill the briefing.
        cur = self.conn.execute(
            "SELECT id, url, title, source, ai_score, kw_score, full_text, first_seen, "
            "time_sensitivity, ml_score, published FROM articles "
            "WHERE first_seen >= ? "
            "AND (published IS NULL OR published = '' OR published >= ?) "
            f"AND {_LIVE_ONLY_CLAUSE} "
            "ORDER BY COALESCE(NULLIF(ai_score, 0), ml_score, 0) DESC, "
            "         urgency DESC, kw_score DESC LIMIT ?",
            (since, pub_cutoff, limit * 3),
        )
        rows = cur.fetchall()
        # Per-publisher-domain diversity cap. ``rows`` is already score-ordered
        # (SQL ORDER BY), so the FIRST up-to-cap rows of any domain are its
        # highest-scored ones; weaker same-domain rows spill to ``overflow``.
        # The cap is then backfilled from that score-ordered overflow so the
        # digest is NEVER shorter than the pre-cap behaviour — it is only
        # re-prioritised for source diversity. See BRIEFING_MAX_PER_DOMAIN.
        out = []
        overflow = []
        per_domain: dict[str, int] = {}
        for r in rows:
            if _published_older_than(r[10], pub_cutoff_dt):
                continue  # GDELT/RSS-indexed stale article — not breaking news
            item = {"_id": r[0], "link": r[1], "title": r[2], "source": r[3],
                    "ai_score": r[4] if r[4] else (r[9] or 0),
                    "_relevance_score": r[5],
                    "summary": decompress(r[6]) if r[6] else "",
                    "first_seen": r[7],
                    "time_sensitivity": r[8],
                    # True iff this row carries a real LLM ground-truth label
                    # (raw ai_score > 0). Model self-predictions go to
                    # ml_score and NEVER to ai_score (invariant #2), so a
                    # falsy raw ai_score means the displayed score above came
                    # from ml_score — an UNVERIFIED local-model estimate. The
                    # ML relevance head demonstrably over-scores
                    # forum/wiki/social rows (recurring live finding), and the
                    # COALESCE above otherwise erases the distinction, so the
                    # briefing consumer can't tell an Opus/Sonnet-vetted 9 from
                    # a raw-model 9.8. Additive key; the displayed ``ai_score``
                    # field and all ordering/diversity/decay logic are
                    # unchanged. Read-only — no DB write, no ai_score/ml_score/
                    # score_source mutation, backtest already excluded by
                    # _LIVE_ONLY_CLAUSE above: all four invariants intact.
                    "_llm_vetted": bool(r[4])}
            key = _briefing_domain_key(r[3] or "")
            if per_domain.get(key, 0) >= BRIEFING_MAX_PER_DOMAIN:
                overflow.append(item)
                continue
            per_domain[key] = per_domain.get(key, 0) + 1
            out.append(item)
            if len(out) >= limit:
                return out
        # Cap left us short of `limit` (low-diversity window): backfill from the
        # highest-scored overflow so a sparse window still yields a full digest.
        if len(out) < limit and overflow:
            out.extend(overflow[: limit - len(out)])
        return out

    @_retry_on_lock
    def reap_stale_urgent(self, max_age_hours: int = 24) -> int:
        """Demote ``urgency=1`` rows that aged out of the alert-fetch window
        back to ``urgency=0``. Returns the number of rows demoted.

        ``get_unalerted_urgent`` only ever returns rows with
        ``first_seen >= now - 24h``. So the instant a still-pending
        ``urgency=1`` row's ``first_seen`` crosses that boundary it becomes
        permanently invisible to the alert worker: it can never be alerted and
        — because it is still ``urgency=1``, not ``2`` — nothing ever clears
        it. It then lingers until the 90-day purge, the whole time inflating
        ``stats()``'s ``urgent`` tile (which counts ``urgency>=1`` with no time
        filter), so the dashboard shows phantom "urgent" items the analyst will
        never actually be pushed. Live evidence (2026-05-18): 26 rows stuck at
        ``urgency=1`` since 2026-05-13 — 5 days, never alerted.

        This is the structural counterpart to the ``alert_agent`` pass-18
        stale-drop fix (``d5918e3``), NOT a duplicate of it:

          - That fix marks *in-window* rows ``urgency=2`` (mark_alerted) — for
            a row the formatter actively *decided* not to deliver, "alerted"
            is both truthful and blocks re-fetch.
          - These rows are *aged-out* — the alert worker NEVER saw them, so
            ``urgency=2`` would be a lie (no analyst was ever pushed) AND
            would keep inflating the ``urgency>=1`` tile this method exists to
            fix. ``urgency=0`` is the only state that is both honest and
            corrective. The two fixes must NOT be "harmonized".

        Demotion loses zero delivery: a row older than the 24h window is
        provably never returned by ``get_unalerted_urgent`` again, so it could
        never have fired regardless (identical reasoning to pass-18's "a stale
        row only ages further — it can never become a valid fresh alert").

        Invariants: only ``urgency`` is written — ``ai_score`` / ``ml_score``
        / ``score_source`` are untouched (label/score-source separation
        intact). ``_LIVE_ONLY_CLAUSE`` is applied as defense-in-depth (same
        discipline as ``update_scores_from_labels``): synthetic rows are
        inserted ``urgency=0`` by construction so the clause is a no-op here,
        but it guarantees a future invariant violation elsewhere can't make
        this path mutate a training row.
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        ).isoformat()
        with self._write_lock:
            cur = self.conn.execute(
                "UPDATE articles SET urgency=0 "
                f"WHERE urgency=1 AND first_seen < ? AND {_LIVE_ONLY_CLAUSE}",
                (cutoff,),
            )
            n = cur.rowcount
            self.conn.commit()
        if n > 0:
            _log.info(
                f"[article_store] reaped {n} stale urgency=1 row(s) "
                f"(first_seen older than {max_age_hours}h, never alerted — "
                f"unreachable by the alert worker, demoted to urgency=0)"
            )
        return n

    @_retry_on_lock
    def purge_old(self):
        """Delete articles older than RETENTION_DAYS and vacuum.

        Also reaps stale ``urgency=1`` residue first (see
        ``reap_stale_urgent``) — ``purge_worker`` is the periodic maintenance
        sweep, so this is its natural home. Called before acquiring
        ``_write_lock`` because ``reap_stale_urgent`` takes that same
        non-reentrant lock itself; nesting it inside the block below would
        deadlock."""
        self.reap_stale_urgent()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)).isoformat()
        with self._write_lock:
            cur = self.conn.execute("DELETE FROM articles WHERE first_seen < ?", (cutoff,))
            deleted = cur.rowcount
            self.conn.commit()
            if deleted > 0:
                self.conn.execute("PRAGMA wal_checkpoint(PASSIVE)")  # TRUNCATE blocks readers; PASSIVE doesn't
                print(f"[store] Purged {deleted} articles older than {RETENTION_DAYS} days")
        return deleted

    @_retry_on_lock
    def stats(self, score_min_kw: float = 1.5) -> dict:
        # ``total``: MAX(rowid) is an O(log N) B-tree walk to the rightmost
        # leaf; a bare COUNT(*) over the ~1.9M-row table on the slow USB drive
        # is O(N) and was the >30 s blocker that made /api/stats time out (the
        # dashboard then showed "0 Total in DB"). rowid is monotonic here (TEXT
        # primary key, no AUTOINCREMENT, purge only deletes the OLDEST/lowest
        # rowids) so MAX(rowid) over-counts vs the live row count by the volume
        # purged out of the RETENTION_DAYS window — an acceptable, order-of-
        # magnitude-correct figure for a dashboard tile. ``fetchone()`` is
        # ``(None,)`` on an empty table, hence ``or 0``.
        total = _expect_row(self.conn.execute(
            "SELECT MAX(rowid) FROM articles"))[0] or 0
        # idx_urgency makes this O(log N); the LIMIT 10000 subquery is a belt-
        # and-braces cap so a missing/disabled index can never reintroduce a
        # full-table scan on the request path.
        urgent = _expect_row(self.conn.execute(
            "SELECT COUNT(*) FROM "
            "(SELECT 1 FROM articles WHERE urgency>=1 LIMIT 10000)"
        ))[0]
        # "unscored" = pending the scorer (kw_score above the scorer's
        # threshold); "below_threshold" = intentionally skipped, not a backlog.
        # Both mirror ``get_unscored`` exactly (ai_score=0 AND ml_score IS NULL
        # AND live-only) so the count reflects what the scorer will actually
        # re-fetch — without _LIVE_ONLY_CLAUSE this over-counted synthetic
        # backtest rows the scorer never touches. No index is selective for
        # that predicate, so each is a ~115 s full scan of BLOB pages on USB —
        # far too slow for the request path. They change slowly, so serve them
        # from a short-TTL cache refreshed by ONE background thread (private
        # connection); a cold cache reads 0 for one poll then self-fills.
        with _STATS_BACKLOG_LOCK:
            cache_age = time.time() - _STATS_BACKLOG_CACHE["ts"]
            unscored = _STATS_BACKLOG_CACHE["unscored"]
            below_threshold = _STATS_BACKLOG_CACHE["below_threshold"]
            need_refresh = (cache_age > _STATS_BACKLOG_TTL_SECS
                            and not _STATS_BACKLOG_CACHE["refreshing"])
            if need_refresh:
                _STATS_BACKLOG_CACHE["refreshing"] = True
        if need_refresh:
            threading.Thread(
                target=_refresh_backlog_counts, args=(score_min_kw,),
                name="stats-backlog-refresh", daemon=True,
            ).start()
        db = _get_db_path()
        # Include -wal and -shm sidecars so db_mb reflects true on-disk usage
        # under WAL journaling (otherwise stats undercount during active writes).
        size_bytes = 0
        for suffix in ("", "-wal", "-shm"):
            p = db.with_name(db.name + suffix)
            if p.exists():
                size_bytes += p.stat().st_size
        size_mb = size_bytes / 1024 / 1024
        return {"total": total, "urgent": urgent, "unscored": unscored,
                "below_threshold": below_threshold, "db_mb": round(size_mb, 1),
                **lock_metrics()}

    @_retry_on_lock
    def stats_since(self, hours: int) -> dict:
        since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        total = _expect_row(self.conn.execute(
            f"SELECT COUNT(*) FROM articles WHERE first_seen >= ? AND {_LIVE_ONLY_CLAUSE}",
            (since,),
        ))[0]
        urgent = _expect_row(self.conn.execute(
            f"SELECT COUNT(*) FROM articles WHERE first_seen >= ? AND urgency>=1 AND {_LIVE_ONLY_CLAUSE}",
            (since,),
        ))[0]
        return {"total": total, "urgent": urgent}

    @_retry_on_lock
    def urgency_label_split(self, hours: int = 24) -> dict:
        """Per-``score_source`` breakdown of urgent rows in the last ``hours``.

        The analyst-facing calibration metric the dashboard was missing: of all
        urgency>=1 rows the alerter saw in the window, what fraction carry a
        real LLM ground-truth label vs only a model self-prediction? Live
        evidence (2026-05-19): every single urgency>=1 row alerted/marked in
        the last 6h had ``ai_score=0`` (so ``score_source='ml'`` — model-only,
        unverified) — exactly the case the alert prompt's "[unverified —
        model-only urgent]" calibration tag exists to hedge per-row, but with
        nothing exposing the aggregate fact at a glance. A persistent
        ``llm_fraction`` near zero means the Sonnet urgency_scorer path is
        either dark, quota-throttled, or flooring everything to noise — the
        analyst's standalone-push channel is being fed by a single (over-
        confident) head and they should know.

        Returns ``{"window_h": int, "total": int, "by_source": {"llm": N,
        "ml": N, "briefing_boost": N, "null": N}, "llm_fraction": float}``.
        ``llm_fraction`` = ``(llm + briefing_boost) / total`` (the two
        ground-truth tags), 0.0 when ``total == 0``. ``null`` covers the
        legacy pre-migration rows still without an explicit tag.

        Read-only (single GROUP BY SELECT) with ``_LIVE_ONLY_CLAUSE`` so the
        synthetic backtest/opus injection rows never inflate either side.
        Decorated with ``@_retry_on_lock`` like every other reader for the
        documented shared-connection cursor-collision class. NO DB write,
        no ai_score/ml_score/score_source/urgency mutation — all four
        load-bearing invariants intact by construction.
        """
        since = (
            datetime.now(timezone.utc) - timedelta(hours=hours)
        ).isoformat()
        rows = self.conn.execute(
            "SELECT score_source, COUNT(*) FROM articles "
            f"WHERE urgency>=1 AND first_seen >= ? AND {_LIVE_ONLY_CLAUSE} "
            "GROUP BY score_source",
            (since,),
        ).fetchall()
        # Always emit the same four keys so a dashboard / health check can
        # render an empty window without conditional branches (mirrors the
        # zero-data discipline of ``ticker_mention_velocity`` returning a row
        # per requested ticker even when it has no mentions).
        by_source: dict[str, int] = {
            "llm": 0, "ml": 0, "briefing_boost": 0, "null": 0,
        }
        for src, n in rows:
            key = src if src in ("llm", "ml", "briefing_boost") else "null"
            by_source[key] += int(n or 0)
        total = sum(by_source.values())
        vetted = by_source["llm"] + by_source["briefing_boost"]
        llm_fraction = round(vetted / total, 4) if total else 0.0
        return {
            "window_h": int(hours),
            "total": total,
            "by_source": by_source,
            "llm_fraction": llm_fraction,
        }

    @_retry_on_lock
    def urgent_source_breakdown(
        self, hours: int = 24, top_n: int = 20
    ) -> dict:
        """Per-``source``-tag breakdown of alerted (urgency>=2) rows in the
        last ``hours``, partitioned LLM-vetted vs ML-only.

        The per-source decomposition that ``urgency_label_split`` is missing:
        that method answers "what fraction of recent urgents carry an LLM
        ground-truth label?" in aggregate but is silent on WHICH source tags
        are dragging the verified fraction down. A persistent
        ``llm_fraction`` near 0.3 on the top-line is operator-actionable only
        if the analyst can see "GN: Nasdaq is 0/47 LLM-vetted, scraped/cnbc
        is 18/18" — then the noisy source can be down-rated, gated, or
        de-prioritised. Without the per-source split the alert channel just
        looks generically degraded.

        Live evidence (2026-05-31 24h pull, urgency=2 rows): the recent
        alerted set was dominated by ml-only ``GN: SP500`` /
        ``GN: Nvidia`` / ``hackernews`` / ``stocktwits`` /
        ``GDELT/ibtimes.com.au`` / ``AlphaVantage/Seeking Alpha`` /
        ``scraped/www.chinadaily.com.cn`` / ``market_valuation`` — the
        analyst's standalone-push channel was being fed by single,
        unverified ml head calls. ``urgency_label_split`` shows the
        aggregate llm_fraction; this surfaces the per-source story.

        ``llm_vetted`` counts rows with ``score_source IN ('llm',
        'briefing_boost')`` — the trainer's STRONG_LABEL_WHERE ground-truth
        tier. ``ml_only`` counts ``score_source = 'ml'`` AND legacy
        ``score_source IS NULL`` rows where ``ai_score = 0`` (no LLM ever
        labeled). Everything else is uncategorised — should always be 0 for
        urgency>=2 rows in practice but kept as a defensive bucket.

        Returns::

            {
              "window_h":  int,
              "total":     int,                     # all urgency>=2 in window
              "llm_vetted_total": int,
              "ml_only_total":    int,
              "llm_fraction":     float,            # vetted / total
              "by_source": [                        # sorted by total DESC
                {
                  "source":       str,
                  "total":        int,
                  "llm_vetted":   int,
                  "ml_only":      int,
                  "llm_fraction": float,            # 0.0..1.0
                },
                ...
              ],
              "worst_offender": {                   # max ml_only with total>=2
                "source":        str,
                "ml_only":       int,
                "total":         int,
                "llm_fraction":  float,
              } | None,
            }

        ``top_n`` caps the ``by_source`` list (sorted by ``total`` desc) so
        long-tail one-offs don't dominate the dashboard tile. The
        ``worst_offender`` slot is computed across ALL sources (not just the
        top-N) but requires ``total >= 2`` so a lone ml-only outlier from an
        unknown publisher doesn't read as the "worst" by accident.

        Backtest isolation: ``_LIVE_ONLY_CLAUSE`` excludes synthetic rows.
        Read-only — no ai_score / ml_score / score_source / urgency
        mutation. All four load-bearing invariants intact by construction.
        Decorated with ``@_retry_on_lock`` for the documented
        shared-connection cursor-collision class — same as sibling readers.
        """
        hours = max(int(hours), 1)
        top_n = max(int(top_n), 1)
        since = (
            datetime.now(timezone.utc) - timedelta(hours=hours)
        ).isoformat()
        # Single GROUP BY pass: (source, score_source, ai_score==0 flag) so the
        # ml-only-via-NULL branch (legacy rows: score_source NULL AND ai_score=0)
        # is captured in the same scan. ``ai_score = 0`` is the discriminator
        # for "no LLM ever labeled" — invariant #2 says ml outputs go to
        # ml_score and never ai_score, so ai_score=0 on a urgency>=2 row means
        # the urgency tag came from a model call.
        rows = self.conn.execute(
            "SELECT source, score_source, "
            "  CASE WHEN ai_score = 0 THEN 1 ELSE 0 END AS no_llm, "
            "  COUNT(*) "
            "FROM articles "
            f"WHERE urgency>=2 AND first_seen >= ? AND {_LIVE_ONLY_CLAUSE} "
            "GROUP BY source, score_source, no_llm",
            (since,),
        ).fetchall()

        # Aggregate per-source.
        per_src: dict[str, dict[str, int]] = {}
        total = 0
        vetted_total = 0
        ml_total = 0
        for src, score_src, no_llm, n in rows:
            n = int(n or 0)
            if not n:
                continue
            src_key = src or ""
            bucket = per_src.setdefault(
                src_key, {"total": 0, "llm_vetted": 0, "ml_only": 0}
            )
            bucket["total"] += n
            total += n
            if score_src in ("llm", "briefing_boost"):
                bucket["llm_vetted"] += n
                vetted_total += n
            elif score_src == "ml" or (score_src is None and no_llm):
                # Explicit ml tag, OR legacy null tag with no LLM label
                # (ai_score=0) — both mean the urgency came from a model
                # call without LLM ground truth.
                bucket["ml_only"] += n
                ml_total += n
            # else: legacy null with non-zero ai_score (pre-migration integer
            # LLM label) — counts toward total but not toward either tier.

        # Materialise sorted output. Stable secondary sort on source name keeps
        # ties deterministic for tests / dashboards.
        by_source_full = sorted(
            (
                {
                    "source": s,
                    "total": d["total"],
                    "llm_vetted": d["llm_vetted"],
                    "ml_only": d["ml_only"],
                    "llm_fraction": (
                        round(d["llm_vetted"] / d["total"], 4)
                        if d["total"] else 0.0
                    ),
                }
                for s, d in per_src.items()
            ),
            key=lambda r: (-r["total"], r["source"]),
        )
        by_source = by_source_full[:top_n]

        # Worst offender = source with the highest ml_only count, total>=2 so a
        # single ml-only outlier from an unknown publisher does NOT read as the
        # worst by accident. Same min-sample defensive pattern other dashboard
        # tiles use against long-tail noise (briefing_health 60%-of-expected
        # floor, urgent_source_breakdown's ``ml_only`` filter mirrors that).
        candidates = [r for r in by_source_full if r["ml_only"] >= 1 and r["total"] >= 2]
        if candidates:
            worst = max(candidates, key=lambda r: (r["ml_only"], -ord(r["source"][:1] or "\x00")))
            worst_offender = {
                "source": worst["source"],
                "ml_only": worst["ml_only"],
                "total": worst["total"],
                "llm_fraction": worst["llm_fraction"],
            }
        else:
            worst_offender = None

        return {
            "window_h": hours,
            "total": total,
            "llm_vetted_total": vetted_total,
            "ml_only_total": ml_total,
            "llm_fraction": round(vetted_total / total, 4) if total else 0.0,
            "by_source": by_source,
            "worst_offender": worst_offender,
        }

    @_retry_on_lock
    def label_production_rate(self, window_min: int = 60) -> dict:
        """Per-``score_source`` label-production rate over the recent window.

        Time-derivative companion to ``urgency_label_split``: that method counts
        urgency>=1 rows by source-tag and is silent on whether Sonnet is actively
        labelling NON-urgent traffic. A live failure mode the existing surfaces
        cannot catch is "Sonnet has gone silent for the last 30 minutes" — the
        aggregate urgency-split still shows yesterday's labels, but the
        per-minute LLM rate just dropped to zero across the whole live corpus.

        Counts EVERY article ``first_seen >= now - window_min`` grouped by
        ``score_source``, including unlabeled rows (NULL bucket). Counts are
        keyed off ``first_seen`` rather than the label-write moment because
        ``score_source`` carries no timestamp of its own — the production
        pipeline always labels an article within a few cycles of insertion, so
        in any window much larger than the scoring cadence the two are
        operationally equivalent. Same proxy ``urgency_label_split`` already
        relies on; documented for the analyst-persona reader.

        Returns::

            {
              "window_min":         int,
              "total":              int,       # all live articles in window
              "by_source":          {"llm": N, "ml": N, "briefing_boost": N,
                                     "null": N},
              "rate_per_min":       float,     # total / window_min
              "llm_rate_per_min":   float,     # (llm + briefing_boost) / window_min
              "unscored_fraction":  float,     # null / total
              "verdict":            "HEALTHY" | "THROTTLED" | "DARK" | "NO_DATA",
            }

        Verdict ladder (mirrors the conservative most-severe-first discipline
        of ``briefing_health`` / ``briefing_cadence_trend``):

          * ``NO_DATA`` — no articles in the window. The collectors are dark,
            not the LLM; refuse to flag the LLM path on absent input.
          * ``DARK`` — articles exist but ``llm_rate_per_min == 0``. The Sonnet
            urgency_scorer has produced zero labels this window; the analyst's
            unverified-rate jumped to 100%.
          * ``THROTTLED`` — ``llm_rate_per_min`` < 0.05 (less than one LLM
            label per 20 min on average — Sonnet is alive but moving slowly,
            often the "quota exhausted, retry after backoff" regime).
          * ``HEALTHY`` — everything else.

        Read-only (single GROUP BY SELECT) with ``_LIVE_ONLY_CLAUSE`` so the
        synthetic backtest/opus rows never inflate either side. NO DB write,
        no ai_score/ml_score/score_source/urgency mutation — all four
        load-bearing invariants intact by construction.
        """
        window_min = max(int(window_min), 1)
        since = (
            datetime.now(timezone.utc) - timedelta(minutes=window_min)
        ).isoformat()
        rows = self.conn.execute(
            "SELECT score_source, COUNT(*) FROM articles "
            f"WHERE first_seen >= ? AND {_LIVE_ONLY_CLAUSE} "
            "GROUP BY score_source",
            (since,),
        ).fetchall()
        by_source: dict[str, int] = {
            "llm": 0, "ml": 0, "briefing_boost": 0, "null": 0,
        }
        for src, n in rows:
            key = src if src in ("llm", "ml", "briefing_boost") else "null"
            by_source[key] += int(n or 0)
        total = sum(by_source.values())
        llm_total = by_source["llm"] + by_source["briefing_boost"]
        rate_per_min = round(total / window_min, 3)
        llm_rate_per_min = round(llm_total / window_min, 3)
        unscored_fraction = (
            round(by_source["null"] / total, 4) if total else 0.0
        )
        if total == 0:
            verdict = "NO_DATA"
        elif llm_total == 0:
            verdict = "DARK"
        elif llm_rate_per_min < 0.05:
            verdict = "THROTTLED"
        else:
            verdict = "HEALTHY"
        return {
            "window_min": window_min,
            "total": total,
            "by_source": by_source,
            "rate_per_min": rate_per_min,
            "llm_rate_per_min": llm_rate_per_min,
            "unscored_fraction": unscored_fraction,
            "verdict": verdict,
        }

    @_retry_on_lock
    def urgent_score_distribution(self, hours: int = 24) -> dict:
        """Score-magnitude histogram of urgent rows over the recent window —
        the missing CALIBRATION-AXIS sibling to ``urgency_label_split``.

        ``urgency_label_split`` reports the LLM-vetted fraction (which source-
        tagged the urgency). ``urgent_score_distribution`` reports HOW HIGH
        those scores are: a flood at the 8.0 threshold means urgency calls are
        clustered at the boundary (liberal threshold use, low-confidence
        urgent), while a healthy distribution skews toward 9-10 (strong calls).
        Live evidence (2026-05-25, articles.db 24h pull): of 66 urgency=2 rows
        the score distribution is the analyst's only way to tell whether
        Sonnet/ML are scoring genuine 9-10 events or borderline 8.0 calls; the
        existing sliced-by-source metrics are silent on the magnitude axis.
        A persistent 8.0 spike under high ml-only fraction is the same
        "over-confident urgency head firing every borderline call" failure
        mode this CLAUDE.md repeatedly traces to recap-template / forum noise.

        Buckets the unified score (``COALESCE(NULLIF(ai_score,0), ml_score, 0)``
        — same convention as ``get_unalerted_urgent`` and
        ``get_top_for_briefing``'s ordering, so the histogram aligns with the
        score the alerter / briefing reader actually saw) into the five
        analyst-meaningful ranges:

          * ``[0, 5)``     — sub-threshold; should be empty by construction
                              (the urgency head only fires at >= 8.0). A
                              non-zero count here is a load-bearing-invariant
                              violation; surfaced explicitly so a regression
                              becomes visible.
          * ``[5, 7)``     — sub-urgent prose-quality; same caveat as above.
          * ``[7, 8)``     — sub-threshold borderline; should also be empty.
          * ``[8, 9)``     — borderline urgent (the Sonnet `>= URGENT_THRESHOLD`
                              boundary; the ML urgency head's `>= 8.0`).
          * ``[9, 10]``    — strong urgent.

        Returned per-bucket entries carry both the bucket boundaries and the
        score_source split inside the bucket (so the analyst can see "of the
        87 borderline-8 rows, 80 are ML-only and only 7 are LLM-vetted" — the
        most diagnostic single view of the unverified-rate problem). Aggregate
        ``borderline_fraction`` (rows in `[8, 9)` / total) gives a single
        scalar verdict the dashboard can render.

        Verdict ladder (mirrors briefing_health / label_production_rate
        most-severe-first conservative discipline):

          * ``NO_DATA``        — no urgent rows in the window. Distinct from
                                 BORDERLINE_HEAVY: the analyst should not
                                 interpret an empty window as a calibration
                                 failure.
          * ``BORDERLINE_HEAVY`` — `borderline_fraction > 0.7` (most urgent
                                   calls are at the threshold; the urgency
                                   classifier is liberal — the "many 8.0
                                   urgent" failure mode).
          * ``MIXED``          — `borderline_fraction > 0.4` (40-70% at the
                                 threshold; warning regime).
          * ``WELL_CALIBRATED`` — everything else (most urgent calls are
                                  comfortably above the threshold).

        Read-only (single SELECT) with ``_LIVE_ONLY_CLAUSE`` so synthetic
        backtest/opus rows never inflate either side. Decorated with
        ``@_retry_on_lock`` for the documented shared-connection cursor-
        collision class. NO DB write — no ai_score / ml_score / score_source /
        urgency mutation. All four load-bearing invariants intact by
        construction.

        Returns::

            {
              "window_h":             int,
              "total":                int,
              "buckets": [ {lo, hi, count, by_source: {llm, ml, briefing_boost, null}}, ... ],
              "borderline_fraction":  float,   # bucket [8,9) / total
              "strong_fraction":      float,   # bucket [9,10] / total
              "verdict":              "WELL_CALIBRATED" | "MIXED" | "BORDERLINE_HEAVY" | "NO_DATA",
            }
        """
        hours = max(int(hours), 1)
        since = (
            datetime.now(timezone.utc) - timedelta(hours=hours)
        ).isoformat()
        rows = self.conn.execute(
            "SELECT COALESCE(NULLIF(ai_score, 0), ml_score, 0) AS score, "
            "score_source "
            "FROM articles "
            f"WHERE urgency>=1 AND first_seen >= ? AND {_LIVE_ONLY_CLAUSE}",
            (since,),
        ).fetchall()

        # Five bucket boundaries (lo inclusive, hi exclusive except the last
        # which is inclusive at 10 — clamp on ai_score is [0, 10]).
        bucket_defs = (
            (0.0, 5.0), (5.0, 7.0), (7.0, 8.0), (8.0, 9.0), (9.0, 10.0),
        )
        buckets: list[dict] = []
        for lo, hi in bucket_defs:
            buckets.append({
                "lo": lo, "hi": hi, "count": 0,
                "by_source": {"llm": 0, "ml": 0,
                              "briefing_boost": 0, "null": 0},
            })

        for score, src_tag in rows:
            s = float(score or 0.0)
            # Locate bucket — the final bucket includes 10.0 (inclusive hi).
            for idx, (lo, hi) in enumerate(bucket_defs):
                last = (idx == len(bucket_defs) - 1)
                if s >= lo and (s < hi if not last else s <= hi):
                    bucket = buckets[idx]
                    bucket["count"] += 1
                    key = (src_tag if src_tag in ("llm", "ml", "briefing_boost")
                           else "null")
                    bucket["by_source"][key] += 1
                    break

        total = sum(b["count"] for b in buckets)
        # Bucket index 3 is [8, 9); index 4 is [9, 10].
        borderline = buckets[3]["count"]
        strong = buckets[4]["count"]
        borderline_fraction = round(borderline / total, 4) if total else 0.0
        strong_fraction = round(strong / total, 4) if total else 0.0

        if total == 0:
            verdict = "NO_DATA"
        elif borderline_fraction > 0.7:
            verdict = "BORDERLINE_HEAVY"
        elif borderline_fraction > 0.4:
            verdict = "MIXED"
        else:
            verdict = "WELL_CALIBRATED"

        return {
            "window_h": int(hours),
            "total": total,
            "buckets": buckets,
            "borderline_fraction": borderline_fraction,
            "strong_fraction": strong_fraction,
            "verdict": verdict,
        }

    @_retry_on_lock
    def recent_ml_only_urgent(
        self, hours: int = 24, limit: int = 50
    ) -> list[dict]:
        """Live audit list of UNVERIFIED urgent rows — ``urgency>=1`` rows
        whose ai_score is still 0 and whose only score is the model's own
        ``ml_score`` (``score_source='ml'``) — for the analyst's "what
        slipped past Sonnet verification?" review.

        Sibling to ``urgent_score_distribution`` (calibration histogram,
        aggregated counts) and ``urgency_label_split_by_source``
        (per-source LLM-vetted fraction). Those answer
        *how much* and *from where*; this returns *the actual titles*,
        newest first — a focused audit primitive the analyst can paste
        into a triage channel to ask "is this Sonnet missing something
        real, or noise the gates correctly suppressed?".

        Practical motivation (live evidence 2026-05-25 24h window): 56
        ml-only urgent rows reached ``urgency>=1`` un-LLM-verified, of
        which ~26 were the NVDA $80B buyback wire saturation. The
        aggregate metrics tell the analyst the rate is elevated but not
        which specific titles to look at; this method does.

        Returns one row per article (newest ``first_seen`` first), capped
        at ``limit``::

            [
              {
                "id": str, "title": str, "url": str, "source": str,
                "ml_score": float, "ai_score": float,    # ai_score==0 always
                "urgency": int,
                "first_seen": str,
                "age_hours": float,
              },
              ...
            ]

        Filter semantics — ``score_source='ml'`` AND ``urgency>=1`` AND
        ``_LIVE_ONLY_CLAUSE`` — match the corpus the analyst cares about:
        live news the ML head escalated to urgent that no LLM
        independently labeled. Synthetic backtest/opus rows are excluded
        by construction (load-bearing invariant #1).

        Read-only (single SELECT). NO DB write — no ai_score / ml_score
        / score_source / urgency mutation. All four load-bearing
        invariants intact by construction.
        """
        hours = max(int(hours), 1)
        limit = max(int(limit), 1)
        now = datetime.now(timezone.utc)
        since_iso = (now - timedelta(hours=hours)).isoformat()
        cur = self.conn.execute(
            "SELECT id, title, url, source, ml_score, ai_score, urgency, "
            "first_seen FROM articles "
            "WHERE urgency >= 1 AND score_source = 'ml' "
            "AND first_seen >= ? "
            f"AND {_LIVE_ONLY_CLAUSE} "
            "ORDER BY first_seen DESC LIMIT ?",
            (since_iso, limit),
        )
        rows = cur.fetchall()
        out: list[dict] = []
        for r in rows:
            aid, title, url, source, ml_score, ai_score, urgency, first_seen = r
            age_h = 0.0
            if first_seen:
                try:
                    dt = datetime.fromisoformat(first_seen)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    age_h = max(0.0, (now - dt).total_seconds() / 3600.0)
                except (ValueError, TypeError):
                    age_h = 0.0
            out.append({
                "id": aid or "",
                "title": title or "",
                "url": url or "",
                "source": source or "",
                "ml_score": float(ml_score or 0.0),
                "ai_score": float(ai_score or 0.0),
                "urgency": int(urgency or 0),
                "first_seen": first_seen or "",
                "age_hours": round(age_h, 2),
            })
        return out

    @_retry_on_lock
    def urgent_syndication_clusters(
        self, hours: int = 24, min_cluster_size: int = 2, top_n: int = 10,
    ) -> dict:
        """Same-story syndication clusters within the urgent queue —
        the *distinct-events* axis the existing urgent analytics lack.

        ``urgency_label_split`` answers "what fraction of urgent rows are
        LLM-vetted?"; ``urgent_score_distribution`` answers "are the
        scores borderline or strong?"; ``urgent_event_saturation``
        buckets by ``(held_ticker × closed-vocab event-class)``. None of
        them answers the analyst's "how many DISTINCT events does my
        urgent queue actually represent?" question — the queue can look
        full of 24 urgent rows while really being 8 events syndicated
        across 3 publishers each. Live evidence (2026-05-29, articles.db
        24h pull): "NVDA Stock Drops 10% From All-Time High" landed
        urgent×3 across GN: dividend buyback / Finnhub/Yahoo / GN:
        semiconductor; "NVDA shares fall premarket: CEO Jensen Huang
        announces" landed urgent×3 across three GN: Nvidia variants —
        the dedup gate at the alert path correctly suppresses the
        Discord push, but the analyst reading the queue sees inflated
        urgency counts and has no surface that tells them "this is 14
        rows but 5 events".

        SSOT for the signature: the same
        ``watchers.alert_dedup._signature`` the live alert path already
        uses to collapse duplicates. Re-using it here means the cluster
        membership returned by this analytics method is the SAME group
        the alert gate collapsed at push time — analyst-facing report
        and live behaviour can never silently disagree on what counts
        as "the same story". Lazy-imported to keep the storage layer
        independent of the watchers module load order.

        Returns::

            {
              "window_h":         int,
              "total_urgent":     int,
              "n_clusters":       int,   # signatures with >= min_cluster_size copies
              "n_clustered_rows": int,   # rows inside any cluster
              "n_unique_events":  int,   # distinct signatures (clusters + singletons)
              "syndication_pct":  float, # n_clustered_rows / total_urgent * 100
              "top_clusters": [
                {
                  "signature": str,
                  "size": int,                # number of urgent rows in this cluster
                  "lead_title": str,          # highest-effective-score title
                  "n_sources": int,           # distinct source tags
                  "sources": [str, ...],      # up to 5
                  "first_seen": str,          # earliest first_seen across the cluster
                  "last_seen":  str,          # latest first_seen across the cluster
                },
                ...
              ],
              "verdict": "HEAVY_SYNDICATION" | "MODERATE" | "LIGHT" | "NO_DATA",
            }

        Verdict ladder (most-severe-first, mirrors
        ``urgent_score_distribution`` discipline):

          * ``NO_DATA``           — no urgent rows in the window.
          * ``HEAVY_SYNDICATION`` — ``syndication_pct >= 40``. Most of
                                    the urgent queue is the same handful
                                    of events repeating across sources;
                                    operator-actionable as "your queue
                                    inflation is publisher syndication,
                                    not a flood of events".
          * ``MODERATE``          — ``syndication_pct >= 20``. Notable
                                    syndication but the queue still
                                    carries diverse events.
          * ``LIGHT``             — everything else (< 20% syndication).

        Read-only (single SELECT) with ``_LIVE_ONLY_CLAUSE`` so
        synthetic backtest/opus rows never enter the cluster pool.
        Decorated with ``@_retry_on_lock`` for the documented
        shared-connection cursor-collision class. NO DB write — no
        ai_score / ml_score / score_source / urgency mutation. All four
        load-bearing invariants intact by construction.
        """
        hours = max(int(hours), 1)
        min_cluster_size = max(int(min_cluster_size), 2)
        top_n = max(int(top_n), 1)
        since = (
            datetime.now(timezone.utc) - timedelta(hours=hours)
        ).isoformat()
        rows = self.conn.execute(
            "SELECT title, source, first_seen, ai_score, ml_score "
            "FROM articles "
            f"WHERE urgency>=1 AND first_seen >= ? AND {_LIVE_ONLY_CLAUSE}",
            (since,),
        ).fetchall()

        total_urgent = len(rows)
        if total_urgent == 0:
            return {
                "window_h": hours,
                "total_urgent": 0,
                "n_clusters": 0,
                "n_clustered_rows": 0,
                "n_unique_events": 0,
                "syndication_pct": 0.0,
                "top_clusters": [],
                "verdict": "NO_DATA",
            }

        # SSOT signature — same function the alert path uses to collapse
        # duplicates, so the analyst's cluster view matches what the gate
        # actually merges at push time. Lazy import (storage must not pull
        # the watchers graph at module load).
        try:
            from watchers.alert_dedup import _signature
        except Exception:  # pragma: no cover - import path defense only
            _signature = lambda t: (t or "").strip().lower()[:30]

        # Group rows by signature. Empty signatures (untitled / pre-stripped
        # to nothing) are kept distinct (one bucket per row) so they cannot
        # collapse into a single phantom cluster that inflates the verdict.
        groups: dict[str, list[dict]] = {}
        for idx, (title, source, first_seen, ai_score, ml_score) in enumerate(rows):
            sig = _signature(title)
            if not sig:
                sig = f"__no_sig__{idx}"
            try:
                ai = float(ai_score or 0.0)
            except (TypeError, ValueError):
                ai = 0.0
            try:
                ml = float(ml_score or 0.0)
            except (TypeError, ValueError):
                ml = 0.0
            effective = ai if ai > 0 else ml
            groups.setdefault(sig, []).append({
                "title": title or "",
                "source": source or "",
                "first_seen": first_seen or "",
                "effective_score": effective,
            })

        n_unique_events = len(groups)
        clusters = [
            (sig, members) for sig, members in groups.items()
            if len(members) >= min_cluster_size and not sig.startswith("__no_sig__")
        ]
        n_clusters = len(clusters)
        n_clustered_rows = sum(len(m) for _, m in clusters)
        syndication_pct = round(
            n_clustered_rows / total_urgent * 100.0, 1
        ) if total_urgent else 0.0

        # Largest clusters first; ties broken by lead score (more newsworthy
        # cluster surfaces first). The lead row is the highest effective
        # score so the displayed title is the strongest call in the cluster.
        clusters.sort(
            key=lambda kv: (-len(kv[1]),
                            -max((m["effective_score"] for m in kv[1]),
                                 default=0.0))
        )

        top_clusters: list[dict] = []
        for sig, members in clusters[:top_n]:
            lead = max(members, key=lambda m: m["effective_score"])
            distinct_sources = sorted({m["source"] for m in members if m["source"]})
            timestamps = sorted(m["first_seen"] for m in members if m["first_seen"])
            top_clusters.append({
                "signature": sig,
                "size": len(members),
                "lead_title": lead["title"][:160],
                "n_sources": len(distinct_sources),
                "sources": distinct_sources[:5],
                "first_seen": timestamps[0] if timestamps else "",
                "last_seen": timestamps[-1] if timestamps else "",
            })

        if syndication_pct >= 40.0:
            verdict = "HEAVY_SYNDICATION"
        elif syndication_pct >= 20.0:
            verdict = "MODERATE"
        else:
            verdict = "LIGHT"

        return {
            "window_h": hours,
            "total_urgent": total_urgent,
            "n_clusters": n_clusters,
            "n_clustered_rows": n_clustered_rows,
            "n_unique_events": n_unique_events,
            "syndication_pct": syndication_pct,
            "top_clusters": top_clusters,
            "verdict": verdict,
        }

    @_retry_on_lock
    def urgency_label_split_by_source(
        self, hours: int = 24, top_n: int = 15
    ) -> dict:
        """Per-source breakdown of urgent rows by score_source — answers
        the analyst's "WHICH FEEDERS are driving the unverified rate?"
        question that the aggregate ``urgency_label_split`` cannot.

        Live evidence (2026-05-19 → 2026-05-21): the aggregate metric has
        been pinned at ``mostly_unverified`` (29% LLM-vetted, 71% ML-only)
        for days; the analyst knows the *rate* is bad but not which
        collectors generate the bulk of the ML-only firings, so the action
        ("prune the worst feeders") is ungrounded. This is the natural
        complement — same data, sliced by ``source`` — so the next step
        becomes "yfinance/Motley Fool produced 80 of the 283 ML-only
        urgent rows" rather than guesswork.

        Sibling to ``source_freshness`` / ``source_throughput`` /
        ``ticker_mention_velocity`` (same one-call-instead-of-eyeballing-
        the-log ergonomics, same ``_LIVE_ONLY_CLAUSE`` discipline).
        Returns one row per source that contributed at least one urgent
        article in the window:

          * ``source``         — verbatim ``articles.source`` value
          * ``total``          — urgent rows from this source in the window
          * ``llm``            — tagged ``score_source='llm'``
          * ``ml``             — tagged ``score_source='ml'``
          * ``briefing_boost`` — tagged ``score_source='briefing_boost'``
          * ``null``           — legacy / pre-migration rows with no tag
          * ``llm_fraction``   — ``(llm + briefing_boost) / total``

        Rows are sorted most-ml-only-first (``ml`` desc) so the worst
        offenders surface at the top; alphabetical by ``source`` for ties
        (mirrors ``source_throughput``'s deterministic-tiebreak convention).
        Capped at ``top_n``; ``total_sources`` returns the full count so
        a UI can report "showing 15 of 47".

        Read-only (single GROUP BY SELECT) with ``_LIVE_ONLY_CLAUSE`` so
        the synthetic backtest/opus rows never inflate the per-source
        figure — exactly the discipline the aggregate metric carries and
        the recurring partial-filter regression class
        (``analytics/trend_velocity.py``) violates. NO DB write, no
        ai_score/ml_score/score_source/urgency mutation. All four
        load-bearing invariants intact by construction.
        """
        since = (
            datetime.now(timezone.utc) - timedelta(hours=hours)
        ).isoformat()
        rows = self.conn.execute(
            "SELECT source, score_source, COUNT(*) FROM articles "
            f"WHERE urgency>=1 AND first_seen >= ? AND {_LIVE_ONLY_CLAUSE} "
            "GROUP BY source, score_source",
            (since,),
        ).fetchall()

        per_source: dict[str, dict[str, int]] = {}
        for source, src_tag, n in rows:
            bucket = per_source.setdefault(
                source or "",
                {"llm": 0, "ml": 0, "briefing_boost": 0, "null": 0},
            )
            key = (src_tag if src_tag in ("llm", "ml", "briefing_boost")
                   else "null")
            bucket[key] += int(n or 0)

        materialised: list[dict] = []
        for source, b in per_source.items():
            total = b["llm"] + b["ml"] + b["briefing_boost"] + b["null"]
            if total == 0:
                continue  # defensive — shouldn't happen, dropped row by row
            vetted = b["llm"] + b["briefing_boost"]
            materialised.append({
                "source": source,
                "total": total,
                "llm": b["llm"],
                "ml": b["ml"],
                "briefing_boost": b["briefing_boost"],
                "null": b["null"],
                "llm_fraction": round(vetted / total, 4),
            })

        # Most-ml-only-first; alphabetical tiebreak. Worst-offender feeders
        # surface at the top so the analyst's "which sources do I prune?"
        # question has an immediate answer.
        materialised.sort(key=lambda r: (-r["ml"], r["source"]))

        return {
            "window_h": int(hours),
            "by_source": materialised[: max(int(top_n), 0)],
            "total_urgent": sum(r["total"] for r in materialised),
            "total_sources": len(materialised),
        }

    @_retry_on_lock
    def source_urgent_yield(
        self, hours: int = 24, top_n: int = 15, min_total: int = 20,
    ) -> dict:
        """Per-source urgent-yield rate — the analyst-facing "which feeds are
        net SIGNAL vs net NOISE?" metric the existing per-source siblings
        cannot answer.

        ``source_throughput`` reports volume (how many articles per source);
        ``urgency_label_split_by_source`` reports the LLM-vetted *fraction*
        within urgent rows only. Neither tells the analyst what fraction of
        a source's TOTAL output reaches urgency>=1 — the actual signal-rate.
        A 50-article/h feed where 5 are urgent (10% yield) is materially
        different from a 5,000-article/h feed where 5 are urgent (0.1%
        yield); the latter is a noise firehose burning scorer budget for
        the same 5 useful articles. This metric ranks sources by yield so
        the analyst can prune low-yield high-volume feeds with one query
        instead of cross-correlating two existing primitives by eye.

        Returns one row per source whose article count in the window is
        at least ``min_total`` (sources below the floor are too low-volume
        for a stable rate; the noise of small-N would dominate the
        ranking):

          * ``source``       — verbatim ``articles.source`` value
          * ``total``        — articles from this source in the window
          * ``urgent``       — rows with ``urgency >= 1`` (queued OR alerted)
          * ``alerted``      — rows with ``urgency = 2`` (fired-to-Discord)
          * ``urgent_pct``   — ``urgent / total * 100`` (rounded 0.01)
          * ``alerted_pct``  — ``alerted / total * 100`` (rounded 0.01)

        Rows are sorted ``urgent_pct`` desc — highest-yield (signal-rich)
        feeders surface at the top, lowest-yield (noise-rich) at the
        bottom of the truncated tail. Alphabetical tiebreak on ``source``
        for deterministic, test-pinnable ordering (mirrors
        ``urgency_label_split_by_source``).

        Read-only (single GROUP BY SELECT) with ``_LIVE_ONLY_CLAUSE`` so
        synthetic backtest/opus rows never inflate either side of the
        ratio — the exact discipline the partial-filter regression class
        (``analytics/trend_velocity.py``) violates. NO DB write, no
        ai_score/ml_score/score_source/urgency mutation. All four
        load-bearing invariants intact by construction.
        """
        # Defensive clamps — matches the convention of sibling analytics
        # (``urgency_label_split_by_source`` clamps ``top_n``).
        hours = max(int(hours), 1)
        top_n = max(int(top_n), 0)
        min_total = max(int(min_total), 1)

        since = (
            datetime.now(timezone.utc) - timedelta(hours=hours)
        ).isoformat()
        # Single aggregation pass: per-source counts at each urgency tier.
        # SUM(CASE …) is the cheapest way to fold three counts into one
        # GROUP BY (same shape as ``urgency_label_split``'s GROUP BY
        # score_source). The ``first_seen >= ?`` filter pushes the window
        # bound down so we never scan beyond the recent partition.
        rows = self.conn.execute(
            "SELECT source, "
            "  COUNT(*) AS total, "
            "  SUM(CASE WHEN urgency >= 1 THEN 1 ELSE 0 END) AS urgent, "
            "  SUM(CASE WHEN urgency = 2 THEN 1 ELSE 0 END) AS alerted "
            "FROM articles "
            f"WHERE first_seen >= ? AND {_LIVE_ONLY_CLAUSE} "
            "GROUP BY source",
            (since,),
        ).fetchall()

        materialised: list[dict] = []
        for source, total, urgent, alerted in rows:
            total = int(total or 0)
            if total < min_total:
                continue
            urgent = int(urgent or 0)
            alerted = int(alerted or 0)
            materialised.append({
                "source": source or "",
                "total": total,
                "urgent": urgent,
                "alerted": alerted,
                "urgent_pct": round(100.0 * urgent / total, 2),
                "alerted_pct": round(100.0 * alerted / total, 2),
            })

        # Highest yield first; alphabetical on ``source`` for deterministic
        # tiebreak. Mirrors the sort discipline of
        # ``urgency_label_split_by_source``.
        materialised.sort(
            key=lambda r: (-r["urgent_pct"], r["source"])
        )

        return {
            "window_h": hours,
            "min_total": min_total,
            "by_source": materialised[:top_n],
            "total_sources_qualifying": len(materialised),
        }

    @_retry_on_lock
    def urgency_label_split_by_ticker(
        self, tickers: list[str], hours: int = 24
    ) -> dict:
        """Per-held-ticker breakdown of urgent rows by ``score_source`` —
        the analyst-facing question: which of MY held names are getting
        LLM-vetted urgent alerts vs only model-only ones?

        Sibling to ``urgency_label_split`` (aggregate — the calibration
        headline) and ``urgency_label_split_by_source`` (per-collector
        slice — "which feeders to prune?"). Per-ticker is the third
        natural slice for the analyst persona "I depend on these alerts
        to react to events affecting MY positions". The aggregate metric
        answers "is the alert path LLM-vetted?"; the per-source slice
        answers "which feeders produce the unverified noise?"; this
        answers "which of my OPEN POSITIONS are getting good vs bad
        urgent vetting?". Live evidence (2026-05-21, 24h): NVDA had 89
        urgent rows at 25% LLM-vetted (67 ML-only); AXTI had 10 urgent
        rows at 60% LLM-vetted — the biggest held name has the worst
        verification rate, a per-position answer no other metric
        surfaces.

        Mirrors ``ticker_mention_velocity``'s discipline: tickers are
        passed in (single source of truth lives at the caller —
        ``ml.features.LIVE_PORTFOLIO_TICKERS`` / ``daemon.PORTFOLIO_TICKERS``),
        avoiding the storage→ml import cycle and keeping the held-book
        definition outside the storage layer. Matching is whole-word and
        ALL-CAPS so a substring like ``NVDAQ`` cannot leak a hit for
        ``NVDA``. A leading ``$`` is allowed (``$NVDA`` matches ``NVDA``).
        Tickers shorter than 2 chars are skipped (no signal, would
        over-match). Match surface is ``title + summary`` — same surface
        as ``_book_tickers`` in alert_agent.py so the alert path and this
        metric never disagree about whether a row touches a held name.

        Returns one row per requested ticker that contributed at least
        one urgent article in the window (a ticker with zero urgent
        mentions is omitted — the analyst wants signal, not zero-rows
        for the entire book):

          * ``ticker``         — preserved verbatim from the input
          * ``total``          — urgent rows mentioning this ticker
          * ``llm``            — tagged ``score_source='llm'``
          * ``ml``             — tagged ``score_source='ml'``
          * ``briefing_boost`` — tagged ``score_source='briefing_boost'``
          * ``null``           — legacy / pre-migration rows with no tag
          * ``llm_fraction``   — ``(llm + briefing_boost) / total``

        Rows are sorted most-ml-only-first (``ml`` desc) so the worst-
        vetted held name surfaces at the top; ALPHABETICAL by ticker for
        ties (matches ``urgency_label_split_by_source``'s deterministic
        tiebreak). ``total_urgent`` and ``total_tickers`` come from rows
        actually returned (matched ≥1 ticker), so a UI can render
        "showing N of M held names".

        Read-only (single SELECT) with ``_LIVE_ONLY_CLAUSE`` so the
        synthetic backtest/opus rows never inflate the per-ticker figure
        (the partial-filter regression class ``analytics/trend_velocity.py``
        violates is what this discipline exists to prevent). NO DB write,
        no ai_score/ml_score/score_source/urgency mutation. All four
        load-bearing invariants intact by construction.
        """
        if not tickers:
            return {
                "window_h": int(hours),
                "by_ticker": [],
                "total_urgent": 0,
                "total_tickers": 0,
            }

        clean: list[str] = []
        for raw in tickers:
            if not raw:
                continue
            t = str(raw).strip().upper()
            if len(t) < 2:
                continue
            if t not in clean:
                clean.append(t)
        if not clean:
            return {
                "window_h": int(hours),
                "by_ticker": [],
                "total_urgent": 0,
                "total_tickers": 0,
            }

        since = (
            datetime.now(timezone.utc) - timedelta(hours=hours)
        ).isoformat()
        # Mirror urgency_label_split's WHERE: urgency>=1 (queued OR alerted)
        # so both the actually-pushed (urgency=2) AND the suppressed-but-
        # marked-alerted formatter rows count — same surface the aggregate
        # metric describes. Summary is decompressed so the match surface
        # matches alert_agent._book_tickers exactly (title + summary).
        rows = self.conn.execute(
            "SELECT title, full_text, score_source FROM articles "
            f"WHERE urgency>=1 AND first_seen >= ? AND {_LIVE_ONLY_CLAUSE}",
            (since,),
        ).fetchall()

        # Whole-word, ALL-CAPS, optional leading $. Compiled once per ticker
        # outside the row loop so the row-scan stays O(rows * tickers).
        patterns = {
            t: re.compile(rf"\b\${{0,1}}{re.escape(t)}\b")
            for t in clean
        }
        per_ticker: dict[str, dict[str, int]] = {}
        for title, blob, src_tag in rows:
            title = title or ""
            try:
                summary = decompress(blob) if blob else ""
            except Exception:
                summary = ""
            hay = f"{title} {summary}"
            key = (src_tag if src_tag in ("llm", "ml", "briefing_boost")
                   else "null")
            for t, pat in patterns.items():
                if not pat.search(hay):
                    continue
                bucket = per_ticker.setdefault(
                    t,
                    {"llm": 0, "ml": 0, "briefing_boost": 0, "null": 0},
                )
                bucket[key] += 1

        materialised: list[dict] = []
        for t in clean:
            b = per_ticker.get(t)
            if not b:
                continue  # held name with zero urgent mentions — omit
            total = b["llm"] + b["ml"] + b["briefing_boost"] + b["null"]
            if total == 0:
                continue  # defensive — shouldn't happen, dropped row by row
            vetted = b["llm"] + b["briefing_boost"]
            materialised.append({
                "ticker": t,
                "total": total,
                "llm": b["llm"],
                "ml": b["ml"],
                "briefing_boost": b["briefing_boost"],
                "null": b["null"],
                "llm_fraction": round(vetted / total, 4),
            })

        # Worst-vetted held name first: most-ml-only-first, alphabetical
        # tiebreak. Same deterministic discipline as
        # urgency_label_split_by_source so the dashboard ordering is
        # stable cycle-to-cycle.
        materialised.sort(key=lambda r: (-r["ml"], r["ticker"]))

        return {
            "window_h": int(hours),
            "by_ticker": materialised,
            "total_urgent": sum(r["total"] for r in materialised),
            "total_tickers": len(materialised),
        }

    @_retry_on_lock
    def urgency_label_split_trend(
        self, hours: int = 24, bucket_h: int = 4
    ) -> dict:
        """Per-time-bucket breakdown of urgent rows by ``score_source`` — the
        TIME-AXIS sibling to ``urgency_label_split``.

        ``urgency_label_split`` reports a single point-in-time calibration
        figure (LLM-vetted fraction over the whole window). It cannot tell
        the analyst whether the rate is improving, degrading, or stable —
        critical when Sonnet quota throttles or recovers mid-window, when a
        ML-only spike (recap-template cluster, screener-tape burst) drives
        a transient hour of unverified pushes, or when the daemon was
        restarted partway through and the early buckets have no LLM yet.

        Returns one dict per ``bucket_h``-sized bucket in the requested
        window, oldest-first (so a chart renders left-to-right by time).
        Bucket boundaries are aligned to ``now - hours`` so a 24h window
        with bucket_h=4 yields exactly 6 buckets. An empty bucket (no
        urgent rows) is still emitted — same zero-data discipline as
        ``urgency_label_split`` / ``ticker_mention_velocity`` — so a
        consumer can iterate a fixed-length series without conditional
        branches and the dashboard never renders a gap.

        Each bucket dict has:

          * ``bucket_start`` — ISO timestamp of the bucket's start
                                (``first_seen >= bucket_start``)
          * ``bucket_end``   — ISO timestamp of the bucket's end
                                (``first_seen < bucket_end``)
          * ``total``        — urgent rows in this bucket
          * ``llm``          — score_source='llm'
          * ``ml``           — score_source='ml'
          * ``briefing_boost`` — score_source='briefing_boost'
          * ``null``         — legacy / pre-migration rows
          * ``llm_fraction`` — ``(llm + briefing_boost) / total`` (0.0 when
                                ``total == 0``)

        Top-level keys mirror ``urgency_label_split``: ``window_h``,
        ``bucket_h``, ``total`` (over all buckets), ``llm_fraction``
        (over all buckets — identical to ``urgency_label_split``'s value
        when the same window is queried).

        Read-only (single GROUP BY SELECT) with ``_LIVE_ONLY_CLAUSE`` so
        the synthetic backtest/opus rows never inflate either side. NO DB
        write, no ai_score/ml_score/score_source/urgency mutation. All
        four load-bearing invariants intact by construction.
        """
        hours = max(int(hours), 1)
        bucket_h = max(int(bucket_h), 1)
        # Round up so the window always contains complete buckets — a
        # half-bucket tail would produce an emptier-looking series than the
        # caller asked for.
        n_buckets = (hours + bucket_h - 1) // bucket_h
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(hours=n_buckets * bucket_h)
        since_iso = window_start.isoformat()

        rows = self.conn.execute(
            "SELECT first_seen, score_source FROM articles "
            f"WHERE urgency>=1 AND first_seen >= ? AND {_LIVE_ONLY_CLAUSE}",
            (since_iso,),
        ).fetchall()

        # Pre-seed every bucket with zeros so a quiet hour still emits a
        # row — same discipline as ``urgency_label_split``'s fixed-key dict.
        buckets: list[dict] = []
        for i in range(n_buckets):
            start = window_start + timedelta(hours=i * bucket_h)
            end = window_start + timedelta(hours=(i + 1) * bucket_h)
            buckets.append({
                "bucket_start": start.isoformat(),
                "bucket_end": end.isoformat(),
                "total": 0,
                "llm": 0, "ml": 0, "briefing_boost": 0, "null": 0,
                "llm_fraction": 0.0,
            })

        bucket_secs = bucket_h * 3600
        for first_seen, src_tag in rows:
            if not first_seen:
                continue
            try:
                ts = datetime.fromisoformat(first_seen)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue
            offset_s = (ts - window_start).total_seconds()
            if offset_s < 0:
                continue
            idx = int(offset_s // bucket_secs)
            if idx >= n_buckets:
                continue  # future row or off-by-one at the trailing edge
            key = (src_tag if src_tag in ("llm", "ml", "briefing_boost")
                   else "null")
            buckets[idx][key] += 1
            buckets[idx]["total"] += 1

        # Compute per-bucket llm_fraction now that all rows are bucketed.
        grand_total = 0
        grand_vetted = 0
        for b in buckets:
            total = b["total"]
            grand_total += total
            vetted = b["llm"] + b["briefing_boost"]
            grand_vetted += vetted
            b["llm_fraction"] = round(vetted / total, 4) if total else 0.0

        return {
            "window_h": int(hours),
            "bucket_h": int(bucket_h),
            "total": grand_total,
            "llm_fraction": round(grand_vetted / grand_total, 4)
                            if grand_total else 0.0,
            "buckets": buckets,
        }

    @_retry_on_lock
    def source_freshness(self) -> list[dict]:
        """Per-source liveness view: for every live source, its live-row count
        and how long ago its most recent article landed.

        Built purely from ``articles.db`` over the canonical live-only set
        (``_LIVE_ONLY_CLAUSE`` — synthetic backtest/opus rows are excluded so a
        gone-dark *collector* is never masked by backtest injections that share
        the table). Ordered most-stale-first so a caller sees dark collectors at
        the top of the list. ``newest_age_s`` is seconds since the newest
        ``first_seen`` (insert time — never the back-datable ``published``);
        it is ``None`` only when a source has no parseable timestamp.

        Turns the "which collectors went dark?" question (previously only
        answerable by eyeballing the daemon log) into one queryable call for
        the dashboard / healthcheck.
        """
        now = datetime.now(timezone.utc)
        rows = self.conn.execute(
            "SELECT source, COUNT(*) AS n, MAX(first_seen) AS newest "
            f"FROM articles WHERE {_LIVE_ONLY_CLAUSE} "
            "GROUP BY source"
        ).fetchall()
        out: list[dict] = []
        for source, n, newest in rows:
            age_s = None
            if newest:
                try:
                    ts = datetime.fromisoformat(newest)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    age_s = round((now - ts).total_seconds(), 1)
                except (ValueError, TypeError):
                    age_s = None
            out.append({"source": source or "", "count": n,
                        "newest_age_s": age_s})
        # Most-stale-first; an unparseable/unknown age sorts last (treated as
        # -1 so reverse-sort pushes it below every real age).
        out.sort(
            key=lambda r: r["newest_age_s"] if r["newest_age_s"] is not None else -1.0,
            reverse=True,
        )
        return out

    @_retry_on_lock
    def source_throughput(self, window_min: int = 60) -> list[dict]:
        """Per live source: article count in the most-recent ``window_min``
        minutes vs the immediately-preceding equal window, and the
        deceleration between them.

        ``source_freshness`` answers "how stale is each source's NEWEST
        item"; this answers the *leading-indicator* question "which
        collectors are SLOWING DOWN right now". A source can be
        decelerating sharply while its newest item is still only minutes old
        (e.g. an RSS feed that dropped from 40/h to 3/h but is not yet dark),
        so a rate delta surfaces a degrading collector well before
        ``source_freshness`` reads it as fully stale — earlier warning, same
        one-query-instead-of-eyeballing-the-log ergonomics.

        Built over the canonical live-only set (``_LIVE_ONLY_CLAUSE`` —
        synthetic backtest/opus rows are excluded so an injection burst can
        never mask, or fake, a real collector's rate, CLAUDE.md §5).
        ``first_seen`` (insert time — never the back-datable ``published``)
        is the clock, and the ``first_seen >= prior_cut`` predicate bounds
        the scan to the last two windows via ``idx_first_seen``.

        ``decel_pct`` is the percentage drop from ``prior`` to ``recent``
        (positive = slowing, negative = accelerating). It is ``None`` when
        ``prior`` is 0 — a brand-new or just-recovered source has no
        baseline, which is not a measurable deceleration. Rows are ordered
        most-decelerated-first so the worst-degrading collector is at the
        top; a ``None`` decel sorts last so it never jumps a real slowdown.
        Sources idle in BOTH windows are omitted (no signal to report).
        """
        now = datetime.now(timezone.utc)
        recent_cut = (now - timedelta(minutes=window_min)).isoformat()
        prior_cut = (now - timedelta(minutes=2 * window_min)).isoformat()
        rows = self.conn.execute(
            "SELECT source, "
            "SUM(CASE WHEN first_seen >= ? THEN 1 ELSE 0 END) AS recent, "
            "SUM(CASE WHEN first_seen >= ? AND first_seen < ? THEN 1 ELSE 0 END) AS prior "
            f"FROM articles WHERE first_seen >= ? AND {_LIVE_ONLY_CLAUSE} "
            "GROUP BY source",
            (recent_cut, prior_cut, recent_cut, prior_cut),
        ).fetchall()
        out: list[dict] = []
        for source, recent, prior in rows:
            recent = int(recent or 0)
            prior = int(prior or 0)
            if recent == 0 and prior == 0:
                continue  # idle in both windows — nothing to report
            decel_pct = (
                round((prior - recent) / prior * 100.0, 1)
                if prior > 0 else None
            )
            out.append({"source": source or "", "recent": recent,
                        "prior": prior, "delta": recent - prior,
                        "decel_pct": decel_pct})
        out.sort(
            key=lambda r: r["decel_pct"] if r["decel_pct"] is not None
            else float("-inf"),
            reverse=True,
        )
        return out

    @_retry_on_lock
    def ticker_mention_velocity(
        self, tickers: list[str], window_min: int = 120
    ) -> list[dict]:
        """Per-ticker mention velocity: live article count in the most-recent
        ``window_min`` minutes vs the immediately-preceding equal window, for
        every requested ticker.

        ``source_freshness`` / ``source_throughput`` answer "which collectors
        are slowing"; this answers the *book-level* question "which held names
        are getting unusual coverage right now". A programmatic primitive that
        external callers (paper-trader pre-trade checks, dashboard
        portfolio-signals, chat enrichment) can consume without re-running
        the regex-and-counter glue inline.

        Built over the canonical live-only set (``_LIVE_ONLY_CLAUSE`` —
        synthetic backtest/opus rows are excluded so a per-ticker count can
        never be inflated/masked by an injection burst sharing the table,
        CLAUDE.md §5). This is the bug that ``analytics/trend_velocity.py``
        carries: a partial filter (``source NOT LIKE 'backtest_run_%'``)
        which lets ``backtest://`` URLs and ``opus_annotation*`` sources leak
        through. Use this method instead of writing a new ad-hoc scan.

        Matching is whole-word and ALL-CAPS so a substring like ``NVDAQ`` or
        ``AMDOCS`` cannot leak a hit for ``NVDA`` / ``AMD``. Tickers shorter
        than 2 chars are skipped (no signal, would over-match). A leading
        ``$`` is allowed (``$NVDA`` matches ``NVDA``).

        Returns one dict per requested ticker (a missing ticker is still
        returned with zero counts so callers can iterate the full book
        without conditional branches):
          * ``ticker``         — the input symbol (preserved verbatim)
          * ``recent``         — live mentions in ``[now-window_min, now]``
          * ``prior``          — live mentions in ``[now-2*window_min, now-window_min]``
          * ``ratio``          — ``(recent + 1) / (prior + 1)`` (Laplace-smoothed
                                  so the prior=0 case yields a finite ratio)
          * ``newest_age_s``   — seconds since the newest matching mention
                                  (``None`` when no mentions at all)

        Ordered highest-ratio-first so an accelerating ticker surfaces at
        the top. A ticker with no signal at all (``ratio == 1.0`` from
        zero data) sorts below a real decelerator (whose ratio < 1.0 IS a
        signal); the secondary key (``recent`` desc) breaks ties between
        equally-rising names.
        """
        if not tickers:
            return []

        clean: list[str] = []
        for raw in tickers:
            if not raw:
                continue
            t = str(raw).strip().upper()
            if len(t) < 2:
                continue
            clean.append(t)
        if not clean:
            return []

        now = datetime.now(timezone.utc)
        recent_cut = (now - timedelta(minutes=window_min)).isoformat()
        prior_cut = (now - timedelta(minutes=2 * window_min)).isoformat()
        rows = self.conn.execute(
            "SELECT title, first_seen "
            f"FROM articles WHERE first_seen >= ? AND {_LIVE_ONLY_CLAUSE}",
            (prior_cut,),
        ).fetchall()

        # Compile one regex per ticker — whole-word, ALL-CAPS, optional
        # leading $. Avoids re-compiling inside the title-scan loop and
        # makes each match O(len(title)) regardless of ticker-set size.
        patterns = {t: re.compile(rf"\b\${{0,1}}{re.escape(t)}\b")
                    for t in clean}
        recent_counts: dict[str, int] = {t: 0 for t in clean}
        prior_counts: dict[str, int] = {t: 0 for t in clean}
        newest: dict[str, datetime | None] = {t: None for t in clean}

        for title, first_seen in rows:
            if not first_seen:
                continue
            try:
                ts = datetime.fromisoformat(first_seen)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue
            in_recent = first_seen >= recent_cut
            hay = title or ""
            for t, pat in patterns.items():
                if not pat.search(hay):
                    continue
                if in_recent:
                    recent_counts[t] += 1
                else:
                    prior_counts[t] += 1
                cur = newest[t]
                if cur is None or ts > cur:
                    newest[t] = ts

        out: list[dict] = []
        for t in clean:
            r = recent_counts[t]
            p = prior_counts[t]
            ratio = round((r + 1) / (p + 1), 3)
            age_s: float | None
            if newest[t] is None:
                age_s = None
            else:
                age_s = round((now - newest[t]).total_seconds(), 1)
            out.append({
                "ticker": t,
                "recent": r,
                "prior": p,
                "ratio": ratio,
                "newest_age_s": age_s,
            })

        # Highest-velocity first; ties broken by larger recent count so a
        # ticker with 10/5 ranks above one with 2/1 at the same ratio.
        # A never-mentioned ticker (ratio==1.0, recent==0) sinks below a
        # real decelerator (ratio<1.0) — the float ordering does this for
        # free since 1.0 > any sub-1 ratio. The unary tuple key uses the
        # numeric ratio first so the natural sort holds.
        out.sort(
            key=lambda r: (
                # No-data rows (recent==prior==0) ranked AFTER real signals
                # regardless of their ratio (which would otherwise tie at
                # 1.0). False sorts before True with reverse=True.
                r["recent"] == 0 and r["prior"] == 0,
                -r["ratio"],
                -r["recent"],
            ),
        )
        return out

    @_retry_on_lock
    def urgent_queue_health(
        self,
        tickers: list[str] | None = None,
        reap_age_hours: int = 24,
        near_reap_hours: float = 3.0,
    ) -> dict:
        """Health of the *unalerted* urgent backlog — the analyst's standing
        "am I about to silently miss an urgent item?" view.

        A ``urgency=1`` row is "ML/LLM-scored urgent, not yet pushed to
        Discord". Two facts make a growing ``urgency=1`` backlog dangerous and
        currently invisible:

          * ``get_unalerted_urgent`` only ever returns ``first_seen >= now-24h``
            rows — the instant a still-pending row crosses that boundary the
            alert worker can never see it again;
          * ``reap_stale_urgent`` then demotes it to ``urgency=0``.

        So a ``urgency=1`` row that ages past ``reap_age_hours`` is dropped
        with NO push and NO trace — exactly the "missed urgent item" the
        consuming analyst fears. ``urgency_label_split*`` count rows the
        alerter already *saw*; this is the complement — what is still WAITING,
        and how close it is to being lost.

        A row is ``overdue`` once its age >= ``reap_age_hours`` (push already
        lost, awaiting the next purge sweep); ``near_reap`` once its age is
        within ``near_reap_hours`` of that deadline but not yet overdue.

        ``tickers`` (optional) drives the per-held-name breakdown so the
        analyst can answer "is my BOOK the thing going un-alerted?". Matching
        mirrors ``urgency_label_split_by_ticker`` exactly — whole-word,
        ALL-CAPS, optional leading ``$``, ``len >= 2``, surface = title +
        summary — so the two metrics never disagree about whether an urgent
        row touches a held name.

        Returns::

            {
              "queued":          int,            # all live urgency=1 rows
              "oldest_age_h":    float | None,    # None when queued == 0
              "near_reap":       int,
              "overdue":         int,
              "reap_age_hours":  int,
              "near_reap_hours": float,
              "by_ticker": [ {ticker, queued, oldest_age_h,
                              near_reap, overdue}, ... ],   # worst-oldest-first
            }

        Read-only (single SELECT) scoped with ``_LIVE_ONLY_CLAUSE`` so the
        synthetic backtest/opus rows (inserted ``urgency=0`` by construction,
        but defense-in-depth) can never inflate the backlog. NO DB write — no
        ai_score / ml_score / score_source / urgency mutation. All four
        load-bearing invariants intact by construction.
        """
        now = datetime.now(timezone.utc)
        rows = self.conn.execute(
            "SELECT first_seen, title, full_text FROM articles "
            f"WHERE urgency=1 AND {_LIVE_ONLY_CLAUSE}"
        ).fetchall()

        reap_age_hours = max(int(reap_age_hours), 1)
        near_reap_hours = max(float(near_reap_hours), 0.0)
        near_cut = reap_age_hours - near_reap_hours  # age >= this → near-reap

        # Whole-word, ALL-CAPS, optional leading $ — identical discipline to
        # urgency_label_split_by_ticker so the two metrics never disagree.
        clean: list[str] = []
        for raw in tickers or []:
            if not raw:
                continue
            t = str(raw).strip().upper()
            if len(t) >= 2 and t not in clean:
                clean.append(t)
        patterns = {
            t: re.compile(rf"\b\${{0,1}}{re.escape(t)}\b") for t in clean
        }
        per_ticker: dict[str, list[float]] = {t: [] for t in clean}

        ages: list[float] = []
        for first_seen, title, blob in rows:
            if not first_seen:
                # Unparseable timestamp — count it as queued but it cannot be
                # aged. Treated as age 0.0 (fresh) so it never fakes an
                # overdue/near-reap warning the operator would chase.
                age_h = 0.0
            else:
                try:
                    ts = datetime.fromisoformat(first_seen)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    age_h = max(0.0, (now - ts).total_seconds() / 3600.0)
                except (ValueError, TypeError):
                    age_h = 0.0
            ages.append(age_h)
            if clean:
                try:
                    summary = decompress(blob) if blob else ""
                except Exception:
                    summary = ""
                hay = f"{title or ''} {summary}"
                for t, pat in patterns.items():
                    if pat.search(hay):
                        per_ticker[t].append(age_h)

        def _classify(age_list: list[float]) -> dict:
            n = len(age_list)
            oldest = round(max(age_list), 2) if age_list else None
            overdue = sum(1 for a in age_list if a >= reap_age_hours)
            near = sum(1 for a in age_list
                       if near_cut <= a < reap_age_hours)
            return {"queued": n, "oldest_age_h": oldest,
                    "near_reap": near, "overdue": overdue}

        top = _classify(ages)
        by_ticker: list[dict] = []
        for t in clean:
            al = per_ticker[t]
            if not al:
                continue  # held name with zero queued urgent rows — omit
            by_ticker.append({"ticker": t, **_classify(al)})
        # Worst-oldest-first so the held name closest to a silent drop is at
        # the top; alphabetical tiebreak for a stable, test-pinnable order.
        by_ticker.sort(
            key=lambda r: (-(r["oldest_age_h"] or 0.0), r["ticker"])
        )

        return {
            "queued": top["queued"],
            "oldest_age_h": top["oldest_age_h"],
            "near_reap": top["near_reap"],
            "overdue": top["overdue"],
            "reap_age_hours": reap_age_hours,
            "near_reap_hours": near_reap_hours,
            "by_ticker": by_ticker,
        }

    @_retry_on_lock
    def book_alert_coverage(
        self,
        tickers: list[str],
        hours: int = 24,
        mentions_only_min: int = 5,
    ) -> dict:
        """Per-held-ticker alert-pipeline coverage — the analyst's "is the
        alert path actually surfacing news on MY positions?" view.

        Sibling to ``urgency_label_split_by_ticker`` (per-ticker LLM-vetted
        fraction over urgency>=1 only — quality of what reached urgent) and
        ``ticker_mention_velocity`` (rate-of-change of total mentions); this
        is the third orthogonal slice — *coverage*, the ratio of urgent
        classifications to total article volume.

        The novel signal is the ``MENTIONS_ONLY`` verdict — a held name with
        substantial article volume (>= ``mentions_only_min`` in the window)
        yet ZERO ``urgency>=1`` classifications. Either the urgency scorer
        missed real signal on that ticker (a calibration miss), or the
        coverage is genuinely all low-urgency colour/recap (a coverage-mix
        problem). Either way, the analyst-facing message is the same: "you
        appear to have unprocessed news on this position." Nothing else
        surfaces this — ``urgent_queue_health`` tracks the queued-but-
        unpushed backlog (rows that DID reach urgency=1), ``held_ticker_
        news_silence`` tracks 24h DARK (zero mentions at all), and the
        per-ticker calibration split sees only urgent rows.

        Note on ``urgency=2`` semantics: this column is set by both an
        actual Discord push (``send_urgent_alert`` → ``mark_alerted_batch``
        on success) AND by every defense-in-depth gate's unconditional
        ``mark_alerted_batch`` on suppression (quote-widget / recap-
        template / low-authority lone / stale / cross-cycle / paraphrase
        duplicate). So ``alerted`` here counts "rows that exited the urgent
        queue", not "rows pushed to Discord" — the latter requires joining
        ``alert_recency.db`` (see ``analytics/alert_delivery_audit.py``).
        The verdict ladder uses ``urgent >= 1`` (urgency>=1 rows), which is
        invariant to that ambiguity — "did this ticker ever reach urgent
        classification this window?" is what matters for the coverage gap.

        Matching is whole-word, ALL-CAPS, optional leading ``$``,
        ``len >= 2`` — byte-identical to ``urgency_label_split_by_ticker``
        / ``ticker_mention_velocity`` / ``urgent_queue_health``'s discipline
        so the four per-ticker primitives never disagree about whether a
        row touches a held name. Match surface is ``title + decompressed
        summary`` — same as those siblings.

        Returns::

            {
              "window_h": int,
              "mentions_only_min": int,
              "by_ticker": [
                {
                  "ticker": "NVDA",
                  "mentions": int,            # all live rows mentioning it
                  "urgent": int,              # urgency>=1 rows
                  "alerted": int,             # urgency=2 rows (queue-exited)
                  "latest_mention_age_h": float|None,
                  "latest_urgent_age_h": float|None,
                  "verdict": "QUIET"|"LOW_VOLUME"|"MENTIONS_ONLY"|"URGENT",
                }, ...
              ],
              "n_quiet": int,
              "n_low_volume": int,
              "n_mentions_only": int,         # the actionable count
              "n_urgent": int,
            }

        Sorted worst-first: ``MENTIONS_ONLY`` then ``LOW_VOLUME`` then
        ``URGENT`` then ``QUIET``; within bucket, descending ``mentions``
        then alphabetical ticker (deterministic, test-pinnable — same
        tiebreak discipline as ``urgency_label_split_by_source``).

        Read-only (single SELECT) scoped with ``_LIVE_ONLY_CLAUSE`` so
        synthetic backtest/opus rows never inflate any per-ticker figure.
        NO DB write — no ai_score / ml_score / score_source / urgency
        mutation. All four load-bearing invariants intact by construction.
        """
        hours = max(int(hours), 1)
        mentions_only_min = max(int(mentions_only_min), 1)

        if not tickers:
            return {
                "window_h": hours,
                "mentions_only_min": mentions_only_min,
                "by_ticker": [],
                "n_quiet": 0, "n_low_volume": 0,
                "n_mentions_only": 0, "n_urgent": 0,
            }

        clean: list[str] = []
        for raw in tickers:
            if not raw:
                continue
            t = str(raw).strip().upper()
            if len(t) >= 2 and t not in clean:
                clean.append(t)
        if not clean:
            return {
                "window_h": hours,
                "mentions_only_min": mentions_only_min,
                "by_ticker": [],
                "n_quiet": 0, "n_low_volume": 0,
                "n_mentions_only": 0, "n_urgent": 0,
            }

        now = datetime.now(timezone.utc)
        since_iso = (now - timedelta(hours=hours)).isoformat()
        rows = self.conn.execute(
            "SELECT title, full_text, urgency, first_seen FROM articles "
            f"WHERE first_seen >= ? AND {_LIVE_ONLY_CLAUSE}",
            (since_iso,),
        ).fetchall()

        patterns = {
            t: re.compile(rf"\b\${{0,1}}{re.escape(t)}\b") for t in clean
        }
        # Per-ticker mutable state — counts + most-recent timestamps.
        state: dict[str, dict] = {
            t: {"mentions": 0, "urgent": 0, "alerted": 0,
                "latest_mention": None, "latest_urgent": None}
            for t in clean
        }

        for title, blob, urg, first_seen in rows:
            try:
                summary = decompress(blob) if blob else ""
            except Exception:
                summary = ""
            hay = f"{title or ''} {summary}"
            ts: datetime | None = None
            if first_seen:
                try:
                    ts = datetime.fromisoformat(first_seen)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    ts = None
            try:
                urg_i = int(urg or 0)
            except (TypeError, ValueError):
                urg_i = 0
            for t, pat in patterns.items():
                if not pat.search(hay):
                    continue
                s = state[t]
                s["mentions"] += 1
                if ts is not None and (s["latest_mention"] is None
                                        or ts > s["latest_mention"]):
                    s["latest_mention"] = ts
                if urg_i >= 1:
                    s["urgent"] += 1
                    if ts is not None and (s["latest_urgent"] is None
                                            or ts > s["latest_urgent"]):
                        s["latest_urgent"] = ts
                if urg_i >= 2:
                    s["alerted"] += 1

        def _age_h(ts: datetime | None) -> float | None:
            if ts is None:
                return None
            return round(max(0.0, (now - ts).total_seconds() / 3600.0), 2)

        def _verdict(s: dict) -> str:
            if s["mentions"] == 0:
                return "QUIET"
            if s["urgent"] >= 1:
                return "URGENT"
            if s["mentions"] >= mentions_only_min:
                return "MENTIONS_ONLY"
            return "LOW_VOLUME"

        # Worst-first verdict ladder so the actionable signal surfaces at
        # the top — same discipline as urgency_label_split_by_source's
        # worst-ml-offender-first sort.
        verdict_rank = {
            "MENTIONS_ONLY": 0, "LOW_VOLUME": 1,
            "URGENT": 2, "QUIET": 3,
        }
        by_ticker: list[dict] = []
        n_quiet = n_low = n_mo = n_urgent = 0
        for t in clean:
            s = state[t]
            v = _verdict(s)
            if v == "QUIET":
                n_quiet += 1
            elif v == "LOW_VOLUME":
                n_low += 1
            elif v == "MENTIONS_ONLY":
                n_mo += 1
            else:
                n_urgent += 1
            by_ticker.append({
                "ticker": t,
                "mentions": s["mentions"],
                "urgent": s["urgent"],
                "alerted": s["alerted"],
                "latest_mention_age_h": _age_h(s["latest_mention"]),
                "latest_urgent_age_h": _age_h(s["latest_urgent"]),
                "verdict": v,
            })
        by_ticker.sort(
            key=lambda r: (verdict_rank[r["verdict"]],
                           -r["mentions"], r["ticker"])
        )

        return {
            "window_h": hours,
            "mentions_only_min": mentions_only_min,
            "by_ticker": by_ticker,
            "n_quiet": n_quiet,
            "n_low_volume": n_low,
            "n_mentions_only": n_mo,
            "n_urgent": n_urgent,
        }

    @_retry_on_lock
    def source_recap_pollution(
        self,
        recap_matcher,
        hours: int = 24,
        min_total: int = 5,
        top_n: int = 20,
    ) -> dict:
        """Per-source recap-template pollution rate — the analyst's "which
        feeds should I prune?" view.

        ``urgency_label_split_by_source`` reports per-source LLM-vetted
        fraction over urgency>=1 rows — the *verification* angle. This
        complements it with the *content-type* angle: of a source's urgent
        rows in the window, what fraction match a recap/SEO template the
        urgency head over-scores (the same fingerprint set the alert and
        briefing layers gate against — see
        ``watchers.alert_agent._RECAP_TEMPLATE_PATTERNS`` and
        ``analysis.claude_analyst._BRIEFING_RECAP_TEMPLATE_PATTERNS``,
        kept in lockstep by ``tests/test_briefing_recap_template.py``).
        A source with high recap_rate is generating model-detectable
        noise the operator can act on (prune the collector, lower its
        cadence, deprioritise its source_cred floor).

        ``recap_matcher`` is an injected callable ``(title) -> bool`` so
        the storage layer never imports the analysis or watchers gates
        (storage is below both; an import would invert the dependency
        graph). The dashboard / CLI caller passes the SSOT matcher; tests
        pass a stub. A row whose title raises in the matcher is treated
        as non-recap (best-effort — a buggy matcher must never bring
        down the metric).

        Returns::

            {
              "window_h":      int,
              "min_total":     int,
              "by_source": [
                {
                  "source":      str,
                  "total":       int,    # urgent rows in the window
                  "recap":       int,    # of those, recap_matcher hits
                  "recap_rate":  float,  # recap / total
                  "fingerprints": {name: count, ...},
                },
                ...
              ],
              "total_urgent":   int,    # over all sources (no min_total)
              "total_recap":    int,    # over all sources (no min_total)
              "global_rate":    float,  # total_recap / total_urgent
            }

        ``fingerprints`` carries the per-template count when the matcher
        returns the ``(hit, name)`` tuple form (the canonical signature of
        ``_looks_like_recap_template`` in both the alert and briefing
        gates). A boolean-only matcher still works — its hits land under
        the synthetic name ``""`` (empty string) — but the SSOT matchers
        always emit names so the dashboard can break each source's
        pollution down by fingerprint and target the worst contributor.

        ``min_total`` bounds the per-source result list to sources that
        had at least N urgent rows in the window — a source with 1 urgent
        row of which 1 is recap reads "100% polluted" without volume to
        justify the verdict. ``top_n`` caps the response size; rows are
        worst-recap-rate-first with alphabetical-source tiebreak (the same
        deterministic discipline as ``urgency_label_split_by_source`` /
        ``ticker_mention_velocity`` so the dashboard ordering is stable
        cycle-to-cycle).

        Read-only (single SELECT) scoped with ``_LIVE_ONLY_CLAUSE`` so
        synthetic backtest/opus rows never inflate either ratio. NO DB
        write — no ai_score / ml_score / score_source / urgency mutation.
        All four load-bearing invariants intact by construction.
        """
        hours = max(int(hours), 1)
        min_total = max(int(min_total), 1)
        top_n = max(int(top_n), 0)

        since_iso = (
            datetime.now(timezone.utc) - timedelta(hours=hours)
        ).isoformat()
        rows = self.conn.execute(
            "SELECT source, title FROM articles "
            f"WHERE urgency >= 1 AND first_seen >= ? AND {_LIVE_ONLY_CLAUSE}",
            (since_iso,),
        ).fetchall()

        per_source: dict[str, dict] = {}
        total_urgent = 0
        total_recap = 0
        for source, title in rows:
            src = source or ""
            bucket = per_source.setdefault(
                src, {"total": 0, "recap": 0, "fingerprints": {}}
            )
            bucket["total"] += 1
            total_urgent += 1
            # Matcher may be bool-returning OR (bool, name)-returning — both
            # forms supported (the alert / briefing SSOT matchers all return
            # the tuple form ``(hit, name)``; a simpler boolean matcher
            # used by tests / dashboards still works). A matcher exception
            # is degraded to "no hit" — pollution metrics must never crash
            # because the upstream regex set raised.
            try:
                result = recap_matcher(title or "")
            except Exception:
                continue
            if isinstance(result, tuple):
                hit = bool(result[0])
                name = (result[1] if len(result) > 1 and result[1] else "")
            else:
                hit = bool(result)
                name = ""
            if not hit:
                continue
            bucket["recap"] += 1
            total_recap += 1
            if name:
                bucket["fingerprints"][name] = (
                    bucket["fingerprints"].get(name, 0) + 1
                )

        materialised: list[dict] = []
        for src, b in per_source.items():
            if b["total"] < min_total:
                continue
            materialised.append({
                "source": src,
                "total": b["total"],
                "recap": b["recap"],
                "recap_rate": round(b["recap"] / b["total"], 4),
                "fingerprints": dict(b["fingerprints"]),
            })

        # Worst-recap-rate first; alphabetical tiebreak — same deterministic
        # discipline as ``urgency_label_split_by_source``.
        materialised.sort(
            key=lambda r: (-r["recap_rate"], -r["recap"], r["source"])
        )

        return {
            "window_h": hours,
            "min_total": min_total,
            "by_source": materialised[: top_n] if top_n else materialised,
            "total_urgent": total_urgent,
            "total_recap": total_recap,
            "global_rate": (
                round(total_recap / total_urgent, 4) if total_urgent else 0.0
            ),
        }

    @_retry_on_lock
    def ticker_recap_pollution(
        self,
        tickers: list[str],
        recap_matcher,
        hours: int = 24,
        min_total: int = 3,
        top_n: int = 20,
    ) -> dict:
        """Per-held-ticker recap-template pollution rate — the analyst's
        "which of MY positions' urgent streams are noise?" view.

        Sibling to ``source_recap_pollution`` (per-collector — "which
        feeders to prune?"); per-ticker is the natural complement for the
        analyst persona "I depend on these alerts to react to events
        affecting MY positions". A held ticker whose urgent rows are 80%
        recap-template (post-earnings "Why X Stock Just Popped" mill
        content) is materially less actionable than one with 5% recap —
        even if the *aggregate* and per-source rates look fine, the
        per-ticker view is what answers the persona's actual question.

        Sibling row shape to ``urgency_label_split_by_ticker`` (per-held-
        ticker LLM-vetted fraction — the *verification* angle) and
        ``book_alert_coverage`` (per-held-ticker urgent-yield over total
        coverage). The four per-held-ticker primitives now answer:

          * ``urgency_label_split_by_ticker`` — "is my book's urgent stream
            LLM-vetted?" (calibration)
          * ``ticker_mention_velocity`` — "is my book's coverage rate
            accelerating?" (momentum)
          * ``book_alert_coverage`` — "did my book's volume actually reach
            urgent?" (yield)
          * ``ticker_recap_pollution`` — "is my book's urgent stream real
            news or recap mill content?" (content type)

        ``recap_matcher`` is an injected callable ``(title) -> bool`` or
        ``(title) -> (bool, name)`` — the SAME signature and SSOT discipline
        as ``source_recap_pollution`` (storage layer must not import the
        analysis or watchers gates; the caller passes the SSOT matcher from
        either layer). A row whose title raises in the matcher degrades to
        "no hit" — pollution metrics must never crash because the upstream
        regex set raised.

        Matching is whole-word, ALL-CAPS, optional leading ``$``,
        ``len >= 2`` — byte-identical to
        ``urgency_label_split_by_ticker`` / ``ticker_mention_velocity`` /
        ``urgent_queue_health`` / ``book_alert_coverage`` so the five
        per-ticker primitives never disagree about whether a row touches
        a held name. Match surface is ``title + decompressed summary`` —
        same as those siblings (ensures a ticker mentioned ONLY in body
        text still counts).

        Returns::

            {
              "window_h":      int,
              "min_total":     int,
              "by_ticker": [
                {
                  "ticker":      str,
                  "total":       int,     # urgent rows mentioning ticker
                  "recap":       int,     # of those, recap_matcher hits
                  "recap_rate":  float,   # recap / total
                  "fingerprints": {name: count, ...},
                },
                ...
              ],
              "total_urgent":   int,     # over all matched tickers (no min_total)
              "total_recap":    int,     # over all matched tickers (no min_total)
              "global_rate":    float,   # total_recap / total_urgent
            }

        ``min_total`` excludes tickers with fewer than N urgent rows from
        the verdict list — a held name with one urgent row of which one is
        recap reads "100% polluted" without volume to justify the verdict.
        ``top_n`` caps the response size; worst-recap-rate-first with
        alphabetical-ticker tiebreak — same deterministic discipline as
        ``source_recap_pollution`` / ``urgency_label_split_by_source`` so
        the dashboard ordering is stable cycle-to-cycle.

        Read-only (single SELECT) scoped with ``_LIVE_ONLY_CLAUSE`` so
        synthetic backtest/opus rows never inflate either ratio. NO DB
        write — no ai_score / ml_score / score_source / urgency mutation.
        All four load-bearing invariants intact by construction.
        """
        hours = max(int(hours), 1)
        min_total = max(int(min_total), 1)
        top_n = max(int(top_n), 0)

        if not tickers:
            return {
                "window_h": hours, "min_total": min_total,
                "by_ticker": [],
                "total_urgent": 0, "total_recap": 0, "global_rate": 0.0,
            }

        clean: list[str] = []
        for raw in tickers:
            if not raw:
                continue
            t = str(raw).strip().upper()
            if len(t) >= 2 and t not in clean:
                clean.append(t)
        if not clean:
            return {
                "window_h": hours, "min_total": min_total,
                "by_ticker": [],
                "total_urgent": 0, "total_recap": 0, "global_rate": 0.0,
            }

        since_iso = (
            datetime.now(timezone.utc) - timedelta(hours=hours)
        ).isoformat()
        rows = self.conn.execute(
            "SELECT title, full_text FROM articles "
            f"WHERE urgency >= 1 AND first_seen >= ? AND {_LIVE_ONLY_CLAUSE}",
            (since_iso,),
        ).fetchall()

        # Whole-word, ALL-CAPS, optional leading $ — byte-identical to the
        # other per-ticker primitives. Compiled once per ticker, outside
        # the row scan, so each match is O(len(hay)) regardless of ticker
        # count.
        patterns = {
            t: re.compile(rf"\b\${{0,1}}{re.escape(t)}\b") for t in clean
        }
        per_ticker: dict[str, dict] = {
            t: {"total": 0, "recap": 0, "fingerprints": {}} for t in clean
        }
        total_urgent = 0
        total_recap = 0

        for title, blob in rows:
            try:
                summary = decompress(blob) if blob else ""
            except Exception:
                summary = ""
            hay = f"{title or ''} {summary}"
            # Match recap fingerprints ONCE per row (matcher is title-only,
            # mirrors ``_looks_like_recap_template``'s signature). Then
            # increment every held-ticker bucket whose pattern matches the
            # row. A single recap row mentioning two held names counts
            # toward both — same multi-attribution discipline as
            # ``urgency_label_split_by_ticker`` and ``book_alert_coverage``.
            try:
                result = recap_matcher(title or "")
            except Exception:
                result = (False, "")
            if isinstance(result, tuple):
                hit = bool(result[0])
                name = (result[1] if len(result) > 1 and result[1] else "")
            else:
                hit = bool(result)
                name = ""

            row_matched_any_ticker = False
            for t, pat in patterns.items():
                if not pat.search(hay):
                    continue
                row_matched_any_ticker = True
                b = per_ticker[t]
                b["total"] += 1
                if hit:
                    b["recap"] += 1
                    if name:
                        b["fingerprints"][name] = (
                            b["fingerprints"].get(name, 0) + 1
                        )
            # Global totals: count the ROW once (not per matching ticker),
            # so total_urgent matches the raw urgent-row count of rows that
            # touch ANY held name in the input set — comparable to the
            # source-side ``total_urgent`` (which is also row-counted, not
            # bucket-summed).
            if row_matched_any_ticker:
                total_urgent += 1
                if hit:
                    total_recap += 1

        materialised: list[dict] = []
        for t in clean:
            b = per_ticker[t]
            if b["total"] < min_total:
                continue  # held name with too-few urgent rows — no verdict
            materialised.append({
                "ticker": t,
                "total": b["total"],
                "recap": b["recap"],
                "recap_rate": round(b["recap"] / b["total"], 4),
                "fingerprints": dict(b["fingerprints"]),
            })

        materialised.sort(
            key=lambda r: (-r["recap_rate"], -r["recap"], r["ticker"])
        )

        return {
            "window_h": hours,
            "min_total": min_total,
            "by_ticker": materialised[: top_n] if top_n else materialised,
            "total_urgent": total_urgent,
            "total_recap": total_recap,
            "global_rate": (
                round(total_recap / total_urgent, 4) if total_urgent else 0.0
            ),
        }

    @_retry_on_lock
    def cross_book_event_pulse(
        self,
        tickers: list[str],
        hours: int = 24,
        min_tickers: int = 2,
        top_n: int = 20,
    ) -> dict:
        """Articles whose title+summary mention ``>= min_tickers`` distinct
        held tickers — the analyst's "what events impact MULTIPLE of my
        positions simultaneously?" view.

        This is the missing cross-position primitive: every other
        per-ticker metric (``urgency_label_split_by_ticker``,
        ``ticker_mention_velocity``, ``urgent_queue_health``,
        ``book_alert_coverage``) slices by ONE ticker at a time, so a single
        wire like "MU, STX, WDC, SNDK stocks sink as Samsung strike ripples
        rattle red-hot AI memory chip trade" — which simultaneously affects
        three held names — splits into three independent per-ticker rows.
        The analyst-facing question "what concentrated-risk events are
        actually happening across the book today?" has no answer.

        Live evidence (2026-05-24, 24h, real production snapshot): 50 of
        4,946 live rows carried 2+ held tickers in the title alone — a
        small but meaningful basket of cross-position events buried in the
        per-ticker noise. The standout urgent row above (urgency=2,
        score_source='ml', source='GN: semiconductor') hit three of the
        analyst's memory positions in one go; nothing in the existing
        primitive set surfaces it as a single "basket" event.

        Articles are grouped by their canonical ticker basket (sorted
        tuple of held tickers mentioned). Repeated coverage of the SAME
        basket collapses into one row, so 5 syndicated copies of the
        Samsung-strike story show as ``count=5`` on the ``(MU, STX, WDC)``
        basket — preserving the recurring-coverage signal without
        flooding the digest.

        Matching is whole-word, ALL-CAPS, optional leading ``$``,
        ``len >= 2`` — byte-identical to ``urgency_label_split_by_ticker``
        / ``ticker_mention_velocity`` / ``urgent_queue_health`` /
        ``book_alert_coverage`` so the five per-ticker primitives never
        disagree about whether a row touches a held name. Match surface
        is ``title + decompressed summary``.

        Returns::

            {
              "window_h": int,
              "min_tickers": int,
              "by_basket": [
                {
                  "basket": ["MU", "STX", "WDC"],     # sorted, deterministic
                  "basket_size": int,
                  "count": int,                       # distinct articles in window
                  "urgent_count": int,                # urgency >= 1 subset
                  "alerted_count": int,               # urgency = 2 subset
                  "max_score": float,                 # max COALESCE(ai_score, ml_score)
                  "newest_age_h": float | None,
                  "sample_title": str,                # representative title
                  "sample_source": str,
                  "score_sources": {"llm": N, "ml": N, ...},
                },
                ...
              ],
              "total_baskets": int,
              "total_articles": int,                  # rows fulfilling min_tickers
            }

        Sorted strongest-event-first: descending ``urgent_count``, then
        ``basket_size``, then ``count``, then alphabetical first ticker
        (deterministic, test-pinnable — same tiebreak discipline as
        ``urgency_label_split_by_source``).

        Read-only (single SELECT) scoped with ``_LIVE_ONLY_CLAUSE`` so the
        synthetic backtest/opus rows never inflate any basket figure (a
        backtest title like "NVDA NVDA NVDA" + "MU MU MU" would otherwise
        manufacture a fake ``(MU, NVDA)`` urgent basket every cycle the
        runner injected). NO DB write — no ai_score / ml_score /
        score_source / urgency mutation. All four load-bearing invariants
        intact by construction.
        """
        hours = max(int(hours), 1)
        min_tickers = max(int(min_tickers), 2)
        top_n = max(int(top_n), 0)

        if not tickers:
            return {
                "window_h": hours,
                "min_tickers": min_tickers,
                "by_basket": [],
                "total_baskets": 0,
                "total_articles": 0,
            }

        clean: list[str] = []
        for raw in tickers:
            if not raw:
                continue
            t = str(raw).strip().upper()
            if len(t) >= 2 and t not in clean:
                clean.append(t)
        if not clean:
            return {
                "window_h": hours,
                "min_tickers": min_tickers,
                "by_basket": [],
                "total_baskets": 0,
                "total_articles": 0,
            }

        now = datetime.now(timezone.utc)
        since_iso = (now - timedelta(hours=hours)).isoformat()
        rows = self.conn.execute(
            "SELECT title, full_text, urgency, ai_score, ml_score, "
            "       score_source, source, first_seen "
            "FROM articles "
            f"WHERE first_seen >= ? AND {_LIVE_ONLY_CLAUSE}",
            (since_iso,),
        ).fetchall()

        # One regex over the held-ticker alternation — same convention as
        # ticker_mention_velocity / urgency_label_split_by_ticker etc.
        patterns = {
            t: re.compile(rf"\b\${{0,1}}{re.escape(t)}\b") for t in clean
        }

        # basket (sorted-tuple) -> aggregated state
        baskets: dict[tuple[str, ...], dict] = {}
        total_articles = 0

        for (title, blob, urg, ai_score, ml_score,
             src_tag, source, first_seen) in rows:
            try:
                summary = decompress(blob) if blob else ""
            except Exception:
                summary = ""
            hay = f"{title or ''} {summary}"
            hits = sorted(t for t, pat in patterns.items() if pat.search(hay))
            if len(hits) < min_tickers:
                continue
            basket = tuple(hits)
            total_articles += 1

            try:
                urg_i = int(urg or 0)
            except (TypeError, ValueError):
                urg_i = 0
            try:
                ai_f = float(ai_score or 0.0)
            except (TypeError, ValueError):
                ai_f = 0.0
            try:
                ml_f = float(ml_score) if ml_score is not None else 0.0
            except (TypeError, ValueError):
                ml_f = 0.0
            # Display score uses COALESCE(NULLIF(ai_score,0), ml_score, 0)
            # — the same convention get_unalerted_urgent / get_top_for_briefing
            # carry, so the basket pulse's max_score matches what the alert
            # and briefing paths would render for the same row.
            row_score = ai_f if ai_f > 0 else ml_f

            ts: datetime | None = None
            if first_seen:
                try:
                    ts = datetime.fromisoformat(first_seen)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    ts = None

            slot = baskets.get(basket)
            if slot is None:
                slot = {
                    "basket": list(basket),
                    "basket_size": len(basket),
                    "count": 0,
                    "urgent_count": 0,
                    "alerted_count": 0,
                    "max_score": 0.0,
                    "newest_ts": None,
                    "sample_title": "",
                    "sample_source": "",
                    "_sample_score": -1.0,
                    "score_sources": {
                        "llm": 0, "ml": 0, "briefing_boost": 0, "null": 0,
                    },
                }
                baskets[basket] = slot
            slot["count"] += 1
            if urg_i >= 1:
                slot["urgent_count"] += 1
            if urg_i >= 2:
                slot["alerted_count"] += 1
            if row_score > slot["max_score"]:
                slot["max_score"] = row_score
            if ts is not None and (
                slot["newest_ts"] is None or ts > slot["newest_ts"]
            ):
                slot["newest_ts"] = ts
            # Sample title is the highest-scoring representative — same
            # discipline as briefing_coverage_audit's max-urgency sample.
            # Tie keeps the FIRST seen so the digest is deterministic.
            if row_score > slot["_sample_score"]:
                slot["_sample_score"] = row_score
                slot["sample_title"] = title or ""
                slot["sample_source"] = source or ""
            key = (src_tag if src_tag in ("llm", "ml", "briefing_boost")
                   else "null")
            slot["score_sources"][key] += 1

        out: list[dict] = []
        for basket, slot in baskets.items():
            age_h: float | None
            if slot["newest_ts"] is None:
                age_h = None
            else:
                age_h = round(
                    max(0.0, (now - slot["newest_ts"]).total_seconds() / 3600.0),
                    2,
                )
            out.append({
                "basket": slot["basket"],
                "basket_size": slot["basket_size"],
                "count": slot["count"],
                "urgent_count": slot["urgent_count"],
                "alerted_count": slot["alerted_count"],
                "max_score": round(slot["max_score"], 3),
                "newest_age_h": age_h,
                "sample_title": slot["sample_title"],
                "sample_source": slot["sample_source"],
                "score_sources": slot["score_sources"],
            })

        # Strongest-event-first: urgent_count desc → basket_size desc →
        # count desc → alphabetical first ticker (deterministic tiebreak).
        out.sort(
            key=lambda r: (
                -r["urgent_count"], -r["basket_size"], -r["count"],
                r["basket"][0] if r["basket"] else "",
            )
        )
        if top_n:
            sliced = out[:top_n]
        else:
            sliced = out

        return {
            "window_h": hours,
            "min_tickers": min_tickers,
            "by_basket": sliced,
            "total_baskets": len(out),
            "total_articles": total_articles,
        }

    @_retry_on_lock
    def ticker_news_burst(
        self,
        tickers: list[str] | None = None,
        window_h: float = 1.0,
        baseline_h: float = 24.0,
    ) -> dict:
        """Per-ticker news-volume burst detector — "is the wire heating up on
        my book RIGHT NOW?".

        The 5h Opus briefing summarises news at briefing cadence; the alert
        path fires on individually-urgent articles. NEITHER answers the
        between-briefings question: which held tickers have an unusual flurry
        of mid-relevance news in the last hour, signalling that something is
        BREWING even if no single article tripped the urgency threshold? Live
        evidence (2026-05-26): SOXX (semiconductor ETF) at 18× its hourly
        baseline (9 articles in 1h vs ~0.48 avg over the prior 23h), MU
        12.67×, QBTS 12.55×, DRAM 10.31×, STX 8× — none surfaced anywhere
        else in the system.

        Per ticker, compares the volume in the last ``window_h`` hours against
        the per-hour-normalised baseline from the PRIOR ``baseline_h`` hours
        (excluding the window). A "spike" of ≥2 with ≥2 articles flags
        WARMING, ≥5 + ≥3 HOT, ≥10 + ≥5 BLAZING. Zero current activity is
        COLD. Otherwise NORMAL.

        ``tickers`` defaults to ``ml.features.LIVE_PORTFOLIO_TICKERS`` (the
        live held + watched set the model and alert ``book:`` tag already
        use). Pass an explicit list for testing or a non-default universe.

        Read-only, ``_LIVE_ONLY_CLAUSE`` applied: no DB write, no
        ai_score/ml_score/score_source/urgency mutation, backtest rows
        excluded — all four load-bearing invariants intact by construction.

        Returns::

            {
              "window_h":     float,
              "baseline_h":   float,
              "n_window":     int,        # total live articles in window
              "n_baseline":   int,        # total live articles in baseline window
              "by_ticker":    [
                {
                  "ticker":           str,
                  "count_window":     int,
                  "count_baseline":   int,
                  "baseline_per_h":   float,
                  "spike":            float | None,   # None if no baseline
                  "verdict":          "BLAZING"|"HOT"|"WARMING"|"NORMAL"|"COLD",
                },
                ...
              ],     # sorted by spike desc, then count_window desc
              "hottest":      str | None,     # top-spike ticker or None if all COLD
              "n_hot":        int,            # count with verdict in {HOT, BLAZING}
            }
        """
        window_h = max(float(window_h), 0.05)
        baseline_h = max(float(baseline_h), window_h * 1.5)

        if tickers is None:
            # Lazy import: ml.features pulls TF-IDF / numpy graph which we
            # only want imported on the analytics path (storage layer must
            # not always pull ml graph).
            from ml.features import LIVE_PORTFOLIO_TICKERS
            tickers = sorted(LIVE_PORTFOLIO_TICKERS)

        clean: list[str] = []
        for raw in tickers or []:
            if not raw:
                continue
            t = str(raw).strip().upper()
            if len(t) < 2 or len(t) > 8:
                continue
            if t not in clean:
                clean.append(t)
        if not clean:
            return {
                "window_h": round(window_h, 2),
                "baseline_h": round(baseline_h, 2),
                "n_window": 0,
                "n_baseline": 0,
                "by_ticker": [],
                "hottest": None,
                "n_hot": 0,
            }

        now = datetime.now(timezone.utc)
        window_start = (now - timedelta(hours=window_h)).isoformat()
        baseline_start = (
            now - timedelta(hours=baseline_h + window_h)
        ).isoformat()
        # baseline_end = window_start: baseline window is [now-baseline-window, now-window]
        baseline_end = window_start

        # Compile word-boundary patterns once. Same shape as the existing
        # urgent_book_breakdown helper above (re.escape + \b). $TICKER and
        # bare TICKER both match; allCAPS minimum 2 chars filters generic
        # English words.
        patterns = {
            t: re.compile(rf"\b\${{0,1}}{re.escape(t)}\b")
            for t in clean
        }

        # Single SELECT pulling only title (cheap — no full_text decompress
        # because a burst is well-evidenced by headline mentions alone; the
        # operator can read summaries off the existing /api/articles surface
        # if they want to drill in).
        window_rows = self.conn.execute(
            "SELECT title FROM articles "
            f"WHERE first_seen >= ? AND {_LIVE_ONLY_CLAUSE}",
            (window_start,),
        ).fetchall()
        baseline_rows = self.conn.execute(
            "SELECT title FROM articles "
            f"WHERE first_seen >= ? AND first_seen < ? AND {_LIVE_ONLY_CLAUSE}",
            (baseline_start, baseline_end),
        ).fetchall()

        def _count(rows) -> dict[str, int]:
            counts = {t: 0 for t in clean}
            for (title,) in rows:
                if not title:
                    continue
                for t, pat in patterns.items():
                    if pat.search(title):
                        counts[t] += 1
            return counts

        c_win = _count(window_rows)
        c_base = _count(baseline_rows)

        out: list[dict] = []
        n_hot = 0
        for t in clean:
            cw = c_win.get(t, 0)
            cb = c_base.get(t, 0)
            base_per_h = cb / baseline_h if baseline_h > 0 else 0.0
            # Spike is the per-hour-normalised ratio. A floor of 0.5 on
            # base_per_h prevents division-by-near-zero from blowing up to
            # absurd ratios on a ticker with one mention every couple days
            # (e.g. a 1h surge of 2 mentions on a 0.04/h baseline becomes
            # 50× — analyst-misleading).
            if cb == 0 and cw == 0:
                spike: float | None = None
            elif cb == 0:
                spike = float(cw) / 0.5  # treat zero baseline as ≤0.5/h
            else:
                spike = cw / max(base_per_h, 0.5)

            # Verdict ladder — most-severe first (mirrors briefing_health /
            # briefing_cadence_trend / urgent_queue_health discipline).
            if cw == 0:
                verdict = "COLD"
            elif spike is not None and spike >= 10.0 and cw >= 5:
                verdict = "BLAZING"
                n_hot += 1
            elif spike is not None and spike >= 5.0 and cw >= 3:
                verdict = "HOT"
                n_hot += 1
            elif spike is not None and spike >= 2.0 and cw >= 2:
                verdict = "WARMING"
            else:
                verdict = "NORMAL"

            out.append({
                "ticker": t,
                "count_window": cw,
                "count_baseline": cb,
                "baseline_per_h": round(base_per_h, 2),
                "spike": round(spike, 2) if spike is not None else None,
                "verdict": verdict,
            })

        # Sort: spike DESC (None last), then count_window DESC, then ticker.
        # Deterministic — the analyst always sees the hottest first.
        out.sort(
            key=lambda r: (
                -(r["spike"] if r["spike"] is not None else -1.0),
                -r["count_window"],
                r["ticker"],
            )
        )
        hottest = None
        for r in out:
            if r["verdict"] in ("BLAZING", "HOT", "WARMING"):
                hottest = r["ticker"]
                break

        return {
            "window_h": round(window_h, 2),
            "baseline_h": round(baseline_h, 2),
            "n_window": len(window_rows),
            "n_baseline": len(baseline_rows),
            "by_ticker": out,
            "hottest": hottest,
            "n_hot": n_hot,
        }

    @_retry_on_lock
    def held_ticker_latest_article(
        self,
        tickers: list[str] | None = None,
        window_h: float = 24.0,
    ) -> dict:
        """Per-held-ticker most-recent live article — the analyst's "what's the
        freshest headline I have about each open position right now?" primitive.

        Existing surfaces answer related but distinct questions:

          * ``ticker_news_burst`` — aggregate volume + spike verdict per ticker;
            does NOT name the specific most-recent article.
          * ``analytics.held_ticker_news_silence`` — CLI audit module writing
            JSON; multi-window counts + ECHO/DARK verdicts, does NOT identify
            the single most-recent article.
          * ``analysis.claude_analyst._book_silence_lines`` — 5h briefing
            silence flag scoped to the post-cap digest; doesn't carry the
            article that broke the silence.
          * ``urgent_book_breakdown`` — per-ticker URGENT counts only; ignores
            the larger relevance pool.

        This primitive returns, per ticker, the single freshest live mention
        in the last ``window_h`` hours (id, title, source, first_seen, link,
        ai_score, ml_score, age_h) plus an in-window mention count. Tickers
        with zero mentions in the window go to ``dark_tickers``. Suitable
        for a dashboard, chat enrichment, or briefing pre-render line:

            MU:   1.2h — "Micron Q3 beat estimates" (finnhub) ai=9.0
            NVDA: 3.4h — "Nvidia buyback announced" (yfinance) ml=8.5
            AXTI: dark — no coverage in 24h

        Match surface: case-insensitive whole-word against ``title`` only —
        SAME convention as ``ticker_news_burst`` / ``ticker_mention_velocity``
        / ``ml.features._LIVE_RE`` so the four held-book surfaces never
        disagree on what counts as a mention. ``$TICKER`` prefix is honored
        via the same ``\\b\\$?TICKER\\b`` discriminator used by velocity.

        Read-only: single SELECT scoped by ``first_seen >= since`` and
        ``_LIVE_ONLY_CLAUSE``, no ai_score / ml_score / score_source /
        urgency mutation, backtest excluded by the SQL clause — all four
        load-bearing invariants intact by construction.

        ``tickers`` defaults to ``ml.features.LIVE_PORTFOLIO_TICKERS``
        (config/portfolio.json positions + option underlyings +
        sector_watchlist, unioned with the hardcoded fallback). Pass an
        explicit list for testing or a non-default universe.

        Returns::

            {
              "window_h":    float,
              "now_iso":     str,                # snapshot wall-clock (UTC)
              "by_ticker":   [
                {
                  "ticker":       str,
                  "id":           str,
                  "title":        str,
                  "source":       str,
                  "first_seen":   str,
                  "link":         str,
                  "ai_score":     float,         # raw, NOT COALESCEd
                  "ml_score":     float | None,
                  "latest_age_h": float | None,  # hours since first_seen
                  "n_in_window":  int,           # total mentions in window
                },
                ...
              ],                                  # sorted freshest-first
              "dark_tickers": [str, ...],         # held tickers with 0 mentions
            }
        """
        window_h = max(float(window_h), 0.05)
        if tickers is None:
            # Lazy import — storage layer must not always pull the ml graph,
            # same discipline as ``ticker_news_burst``'s lazy import.
            from ml.features import LIVE_PORTFOLIO_TICKERS
            tickers = sorted(LIVE_PORTFOLIO_TICKERS)

        clean: list[str] = []
        seen_upper: set[str] = set()
        for raw in tickers or []:
            if not raw:
                continue
            t = str(raw).strip().upper()
            # Same symbol hygiene as ticker_news_burst: 2..8 chars, alphanum.
            # Filters falsy entries, sub-2-char ambiguous symbols, and
            # foreign / compound tickers ("005930.KS") that would otherwise
            # blow up the word-boundary regex compilation.
            if len(t) < 2 or len(t) > 8 or t in seen_upper:
                continue
            seen_upper.add(t)
            clean.append(t)
        if not clean:
            return {
                "window_h": round(window_h, 2),
                "now_iso": datetime.now(timezone.utc).isoformat(),
                "by_ticker": [],
                "dark_tickers": [],
            }

        now = datetime.now(timezone.utc)
        since = (now - timedelta(hours=window_h)).isoformat()

        # \b\$?TICKER\b discriminator — same convention as
        # ticker_news_burst / ticker_mention_velocity so $NVDA matches NVDA
        # and NVDAQ does not leak. The {0,1} on the dollar sign anchors a
        # bounded optional prefix instead of *-style backtracking.
        patterns = {
            t: re.compile(rf"\b\${{0,1}}{re.escape(t)}\b")
            for t in clean
        }

        # ORDER BY first_seen DESC means the first hit per ticker is
        # automatically the newest — no per-ticker re-scan needed. Pull only
        # the metadata fields the consumer needs (no full_text decompress —
        # the consumer can drill in via the link field if they want body
        # text; same discipline as ``ticker_news_burst``'s title-only scan).
        cur = self.conn.execute(
            "SELECT id, title, source, first_seen, url, ai_score, ml_score "
            "FROM articles "
            f"WHERE first_seen >= ? AND {_LIVE_ONLY_CLAUSE} "
            "ORDER BY first_seen DESC",
            (since,),
        )

        latest: dict[str, dict] = {}
        counts: dict[str, int] = {t: 0 for t in clean}
        for row in cur.fetchall():
            aid, title, src, fs, url, ai, ml = row
            if not title:
                continue
            for t, pat in patterns.items():
                if pat.search(title):
                    counts[t] += 1
                    if t not in latest:
                        # First hit per ticker = newest (ORDER BY first_seen DESC).
                        age_h: float | None = None
                        try:
                            if fs:
                                ts = datetime.fromisoformat(fs)
                                if ts.tzinfo is None:
                                    ts = ts.replace(tzinfo=timezone.utc)
                                age_h = round(
                                    max(0.0, (now - ts).total_seconds() / 3600.0),
                                    2,
                                )
                        except (TypeError, ValueError):
                            age_h = None
                        latest[t] = {
                            "ticker": t,
                            "id": aid,
                            "title": title,
                            "source": src or "",
                            "first_seen": fs or "",
                            "link": url or "",
                            "ai_score": float(ai) if ai is not None else 0.0,
                            "ml_score": float(ml) if ml is not None else None,
                            "latest_age_h": age_h,
                        }

        by_ticker: list[dict] = []
        dark: list[str] = []
        for t in clean:
            if t in latest:
                row_out = latest[t]
                row_out["n_in_window"] = counts[t]
                by_ticker.append(row_out)
            else:
                dark.append(t)

        # Freshest-first ordering (smallest age leads). Unknown age (rare —
        # corrupt first_seen) goes last, then ticker name as the stable
        # tiebreak. Mirrors ``ticker_news_burst``'s deterministic sort.
        by_ticker.sort(
            key=lambda r: (
                r["latest_age_h"] if r["latest_age_h"] is not None else float("inf"),
                r["ticker"],
            )
        )

        return {
            "window_h": round(window_h, 2),
            "now_iso": now.isoformat(),
            "by_ticker": by_ticker,
            "dark_tickers": dark,
        }

    # ── Title-keyword sentiment direction (the DIRECTIONAL axis) ──────────
    # ``ticker_news_burst`` measures VOLUME (is the wire heating up on this
    # ticker?). ``ticker_mention_velocity`` measures RATE-OF-CHANGE. Neither
    # answers the analyst-persona question "are my held names getting
    # BULLISH or BEARISH news right now?" — a 10-article surge on NVDA reads
    # identically in volume whether all 10 are "Nvidia beats earnings" or
    # all 10 are "Nvidia misses guidance, drops 15%". The two ml_score-
    # based analytics (``ticker_sentiment_momentum`` / ``sentiment_streak``)
    # use *score deltas*, which conflate relevance/urgency with direction —
    # a recap-template floor (ml_score=0.01) reads as "bearish swing" even
    # though the content has no directional signal at all.
    #
    # This is the missing primitive: a TITLE-KEYWORD direction count, the
    # exact heuristic the analyst applies when skimming a headline list
    # ("rises" / "beats" / "upgrades" → bullish; "plunges" / "misses" /
    # "downgrades" → bearish). Orthogonal to volume and to the score-based
    # sibling — three different axes of the same per-ticker live read.
    #
    # Keyword sets are evidence-anchored to common financial-news verbs;
    # word-boundary anchored so prose ("downgrades" matches but "downgrade
    # cycle" doesn't substring-leak into a different word). Stopwords (FELL
    # vs FELT) are not a concern because the pattern is \bword\b. Lowercased
    # title surface. Same SSOT (LIVE_PORTFOLIO_TICKERS) as ticker_news_burst
    # / held_ticker_latest_article, same `\b\$?TICKER\b` matcher so the
    # four held-book surfaces never disagree about what counts as a mention.
    #
    # Pure read-side primitive: single SELECT scoped by first_seen and
    # _LIVE_ONLY_CLAUSE, no ai_score / ml_score / score_source / urgency
    # mutation, backtest excluded by the SQL clause — all four load-bearing
    # invariants intact by construction.
    _BULLISH_KEYWORDS = (
        # Price-action verbs
        "surge", "surges", "surged", "surging",
        "soar", "soars", "soared", "soaring",
        "rally", "rallies", "rallied", "rallying",
        "rise", "rises", "rose", "rising",
        "jump", "jumps", "jumped", "jumping",
        "climb", "climbs", "climbed", "climbing",
        "spike", "spikes", "spiked", "spiking",
        "gain", "gains", "gained", "gaining",
        "advance", "advances", "advanced",
        # Earnings / guidance
        "beat", "beats", "beating", "topped", "tops",
        "smash", "smashes", "smashed",
        "exceed", "exceeds", "exceeded",
        # Analyst actions
        "upgrade", "upgrades", "upgraded",
        "raised", "raises", "boost", "boosts", "boosted",
        # Corporate actions / outlook
        "buyback", "buybacks", "dividend",
        "partnership", "contract", "wins", "won", "awarded",
        "expand", "expansion", "expansionary",
        "approval", "approved", "approves",
        "record", "outperform", "bullish",
        # Breakout / strength markers
        "breakout", "breakthrough", "strong",
    )
    _BEARISH_KEYWORDS = (
        # Price-action verbs
        "plunge", "plunges", "plunged", "plunging",
        "crash", "crashes", "crashed", "crashing",
        "tumble", "tumbles", "tumbled", "tumbling",
        "drop", "drops", "dropped", "dropping",
        "fall", "falls", "fell", "falling",
        "slide", "slides", "slid", "sliding",
        "sink", "sinks", "sunk", "sinking",
        "slump", "slumps", "slumped", "slumping",
        "decline", "declines", "declined", "declining",
        "slip", "slips", "slipped", "slipping",
        # Earnings / guidance
        "miss", "misses", "missed", "missing",
        "cut", "cuts", "lowered", "lowers", "slash", "slashed", "slashes",
        # Analyst actions
        "downgrade", "downgrades", "downgraded",
        # Corporate / regulatory pain
        "layoff", "layoffs", "fired", "terminated",
        "scandal", "fraud", "probe", "probes", "investigation",
        "suspended", "suspend", "halt", "halts", "halted",
        "ban", "banned", "bans",
        "lawsuit", "lawsuits", "sued", "sue",
        "recall", "recalls", "recalled",
        "default", "defaults", "defaulted",
        "bankruptcy", "bankrupt",
        "warning", "warns", "warned",
        "loss", "losses", "downturn",
        "weak", "bearish", "underperform",
    )

    @_retry_on_lock
    def ticker_sentiment_burst(
        self,
        tickers: list[str] | None = None,
        window_h: float = 6.0,
    ) -> dict:
        """Per-held-ticker title-keyword sentiment direction in ``window_h``.

        For each ticker mentioned in a live article title in the window,
        counts how many of those titles also carry a bullish-direction verb
        (rises, beats, upgrades, ...) vs a bearish-direction verb (plunges,
        misses, downgrades, ...). A title with NEITHER (e.g. "Nvidia CEO
        speaks at conference") counts as ``neutral``. Same title can contain
        both buckets (rare — "Stock falls despite earnings beat"); both are
        incremented honestly so the analyst sees the mixed signal.

        Verdict ladder — most-severe first (mirrors briefing_health /
        ticker_news_burst / urgent_queue_health discipline):

          * ``BULLISH`` — bull >= 2 AND bull >= 2 × bear
          * ``BEARISH`` — bear >= 2 AND bear >= 2 × bull
          * ``MIXED``   — bull + bear >= 3 and neither dominates 2×
          * ``QUIET``   — count_window >= 1 but no directional verbs above 1
          * ``DARK``    — count_window == 0

        Returns per-ticker (sorted by directional intensity desc, then
        ticker name):

            {
              "window_h":    float,
              "n_window":    int,           # total live articles in window
              "by_ticker":   [
                {
                  "ticker":         str,
                  "count_window":   int,    # total title mentions
                  "bull":           int,
                  "bear":           int,
                  "neutral":        int,
                  "intensity":      float,  # (bull - bear) / max(bull+bear,1)
                  "verdict":        "BULLISH"|"BEARISH"|"MIXED"|"QUIET"|"DARK",
                },
                ...
              ],
              "n_bullish":   int,
              "n_bearish":   int,
              "most_bullish": str | None,    # ticker with strongest +intensity ≥ BULLISH bar
              "most_bearish": str | None,    # ticker with strongest -intensity ≥ BEARISH bar
            }

        Pure read-side: ``_LIVE_ONLY_CLAUSE`` applied, no DB write, no
        ai_score / ml_score / score_source / urgency mutation, backtest
        rows excluded — all four load-bearing invariants intact by
        construction.

        ``tickers`` defaults to ``ml.features.LIVE_PORTFOLIO_TICKERS``
        (config/portfolio.json positions + option underlyings +
        sector_watchlist, unioned with hardcoded fallback) — same SSOT as
        ``ticker_news_burst`` / ``held_ticker_latest_article`` so the four
        held-book surfaces stay drift-free.
        """
        window_h = max(float(window_h), 0.05)

        if tickers is None:
            # Lazy import (storage layer must not always pull the ml graph).
            from ml.features import LIVE_PORTFOLIO_TICKERS
            tickers = sorted(LIVE_PORTFOLIO_TICKERS)

        clean: list[str] = []
        seen_upper: set[str] = set()
        for raw in tickers or []:
            if not raw:
                continue
            t = str(raw).strip().upper()
            # Same symbol hygiene as ticker_news_burst / held_ticker_latest_article:
            # 2..8 chars, dedupe by uppercase. Foreign-suffix tickers
            # ("005930.KS") would blow up the word-boundary regex compile.
            if len(t) < 2 or len(t) > 8 or t in seen_upper:
                continue
            seen_upper.add(t)
            clean.append(t)
        if not clean:
            return {
                "window_h": round(window_h, 2),
                "n_window": 0,
                "by_ticker": [],
                "n_bullish": 0,
                "n_bearish": 0,
                "most_bullish": None,
                "most_bearish": None,
            }

        now = datetime.now(timezone.utc)
        since = (now - timedelta(hours=window_h)).isoformat()

        # \b\$?TICKER\b — same matcher as ticker_news_burst /
        # held_ticker_latest_article / urgent_book_breakdown so $NVDA matches
        # NVDA and NVDAQ does not leak. Word-boundaries also keep AMD from
        # matching AMDOCS.
        patterns = {
            t: re.compile(rf"\b\${{0,1}}{re.escape(t)}\b")
            for t in clean
        }
        # Pre-compile the bullish / bearish keyword unions ONCE. Group into
        # one alternation per direction; case-insensitive; \b-anchored so
        # "fell" matches but "felt" doesn't (substring-leak class).
        bull_re = re.compile(
            r"\b(?:" + "|".join(re.escape(k) for k in self._BULLISH_KEYWORDS) + r")\b",
            re.IGNORECASE,
        )
        bear_re = re.compile(
            r"\b(?:" + "|".join(re.escape(k) for k in self._BEARISH_KEYWORDS) + r")\b",
            re.IGNORECASE,
        )

        rows = self.conn.execute(
            "SELECT title FROM articles "
            f"WHERE first_seen >= ? AND {_LIVE_ONLY_CLAUSE}",
            (since,),
        ).fetchall()

        counts: dict[str, dict[str, int]] = {
            t: {"count": 0, "bull": 0, "bear": 0, "neutral": 0} for t in clean
        }
        for (title,) in rows:
            if not title:
                continue
            has_bull = bool(bull_re.search(title))
            has_bear = bool(bear_re.search(title))
            for t, pat in patterns.items():
                if not pat.search(title):
                    continue
                bucket = counts[t]
                bucket["count"] += 1
                if has_bull:
                    bucket["bull"] += 1
                if has_bear:
                    bucket["bear"] += 1
                if not has_bull and not has_bear:
                    bucket["neutral"] += 1

        out: list[dict] = []
        n_bullish = 0
        n_bearish = 0
        for t in clean:
            b = counts[t]
            cw = b["count"]
            bull = b["bull"]
            bear = b["bear"]
            neutral = b["neutral"]
            directional = bull + bear
            # intensity in [-1, +1]: +1 = all-bull, -1 = all-bear, 0 = balanced.
            # max(directional, 1) avoids divide-by-zero on the zero-direction case.
            intensity = (bull - bear) / max(directional, 1)
            # Verdict ladder. Magnitude bar (>= 2) blocks single-mention noise
            # from flipping verdict; the 2× dominance bar separates BULLISH /
            # BEARISH from MIXED. QUIET = some mentions but no real signal.
            if cw == 0:
                verdict = "DARK"
            elif bull >= 2 and bull >= 2 * bear:
                verdict = "BULLISH"
                n_bullish += 1
            elif bear >= 2 and bear >= 2 * bull:
                verdict = "BEARISH"
                n_bearish += 1
            elif directional >= 3:
                verdict = "MIXED"
            else:
                verdict = "QUIET"
            out.append({
                "ticker": t,
                "count_window": cw,
                "bull": bull,
                "bear": bear,
                "neutral": neutral,
                "intensity": round(intensity, 3),
                "verdict": verdict,
            })

        # Sort: by absolute intensity desc (strongest direction first), then
        # by count_window desc, then alphabetical. So a 5-bull-0-bear ticker
        # surfaces above a 2-bull-0-bear ticker even though both are +1.0.
        out.sort(
            key=lambda r: (
                -abs(r["intensity"]),
                -r["count_window"],
                r["ticker"],
            )
        )

        most_bullish: str | None = None
        most_bearish: str | None = None
        for r in out:
            if most_bullish is None and r["verdict"] == "BULLISH":
                most_bullish = r["ticker"]
            if most_bearish is None and r["verdict"] == "BEARISH":
                most_bearish = r["ticker"]
            if most_bullish and most_bearish:
                break

        return {
            "window_h": round(window_h, 2),
            "n_window": len(rows),
            "by_ticker": out,
            "n_bullish": n_bullish,
            "n_bearish": n_bearish,
            "most_bullish": most_bullish,
            "most_bearish": most_bearish,
        }

    def close(self):
        self.conn.close()

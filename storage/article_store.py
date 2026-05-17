"""
Compressed SQLite article store — prefers USB drive, falls back to local data/.
Stores article metadata + compressed full text. Auto-purges articles older than RETENTION_DAYS.
"""
import functools
import hashlib
import logging
import os
import random
import sqlite3
import threading
import time
import zlib
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path

# Module logger — uses central logger if available, falls back to stdlib.
try:
    from core.logger import get_logger
    _log = get_logger("article_store")
except Exception:
    _log = logging.getLogger("article_store")


# ── DB lock retry helper ────────────────────────────────────────────────────
# Even with PRAGMA busy_timeout=60000, sustained writer contention from many
# threads occasionally surfaces ``OperationalError("database is locked")``
# (e.g., during long PRAGMA wal_checkpoint(TRUNCATE) calls). Retry with
# exponential backoff + jitter to avoid thundering-herd retries that would
# just collide again at the same instant. Bubble up after the budget is spent.
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


def _retry_on_lock(func):
    """Decorator: retry on 'database is locked' with exp backoff + jitter."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        last = None
        for attempt in range(_LOCK_RETRY_ATTEMPTS):
            try:
                return func(*args, **kwargs)
            except sqlite3.OperationalError as e:
                if "database is locked" not in str(e).lower():
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
                        f"[article_store] {func.__name__}: 'database is locked' "
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
    score_source TEXT DEFAULT NULL  -- 'llm' (Sonnet/Opus ground truth), 'ml' (model), 'briefing_boost' (Opus curation nudge); NULL=unscored
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


def compress(text: str) -> bytes:
    return zlib.compress(text.encode("utf-8", errors="replace"), level=6)


def decompress(data: bytes) -> str:
    return zlib.decompress(data).decode("utf-8", errors="replace")


class ArticleStore:
    def __init__(self):
        self.conn = _connect()
        # Serializes write operations across worker threads. The connection is shared
        # with check_same_thread=False; concurrent execute+commit pairs would otherwise
        # race on implicit transaction starts ("cannot start a transaction within a transaction").
        self._write_lock = threading.Lock()
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
        now = datetime.now(timezone.utc).isoformat()
        inserted = 0
        with self._write_lock:
            for art in articles:
                url = art.get("link", "")
                title = art.get("title", "")
                if not url or not title:
                    continue
                aid = article_id(url, title)
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
            self.conn.commit()
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
        return cur.fetchone()[0]

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
                scores = score_articles(batch)
                updates = []
                ts_updates = []
                for art, sc in zip(batch, scores):
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
            "first_seen, published "
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
                 "published": r[7]}
                for r in rows]

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
        out = []
        for r in rows:
            if _published_older_than(r[10], pub_cutoff_dt):
                continue  # GDELT/RSS-indexed stale article — not breaking news
            out.append(
                {"_id": r[0], "link": r[1], "title": r[2], "source": r[3],
                 "ai_score": r[4] if r[4] else (r[9] or 0),
                 "_relevance_score": r[5],
                 "summary": decompress(r[6]) if r[6] else "",
                 "first_seen": r[7],
                 "time_sensitivity": r[8]}
            )
            if len(out) >= limit:
                break
        return out

    @_retry_on_lock
    def purge_old(self):
        """Delete articles older than RETENTION_DAYS and vacuum."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)).isoformat()
        with self._write_lock:
            cur = self.conn.execute("DELETE FROM articles WHERE first_seen < ?", (cutoff,))
            deleted = cur.rowcount
            self.conn.commit()
            if deleted > 0:
                self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                print(f"[store] Purged {deleted} articles older than {RETENTION_DAYS} days")
        return deleted

    def stats(self, score_min_kw: float = 1.5) -> dict:
        total = self.conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        urgent = self.conn.execute("SELECT COUNT(*) FROM articles WHERE urgency>=1").fetchone()[0]
        # "unscored" = pending the scorer (kw_score above the scorer's threshold).
        # Articles below the threshold are intentionally skipped, not a backlog;
        # split them out so the metric reflects actionable work. Mirror
        # ``get_unscored`` exactly (ai_score=0 AND ml_score IS NULL AND
        # live-only) so the count reflects what the scorer will actually
        # re-fetch — without _LIVE_ONLY_CLAUSE this over-counted synthetic
        # backtest rows that the scorer never touches.
        unscored = self.conn.execute(
            "SELECT COUNT(*) FROM articles "
            f"WHERE ai_score=0 AND ml_score IS NULL AND kw_score>=? "
            f"AND {_LIVE_ONLY_CLAUSE}",
            (score_min_kw,),
        ).fetchone()[0]
        below_threshold = self.conn.execute(
            "SELECT COUNT(*) FROM articles "
            f"WHERE ai_score=0 AND ml_score IS NULL AND kw_score<? "
            f"AND {_LIVE_ONLY_CLAUSE}",
            (score_min_kw,),
        ).fetchone()[0]
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

    def stats_since(self, hours: int) -> dict:
        since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        total = self.conn.execute(
            f"SELECT COUNT(*) FROM articles WHERE first_seen >= ? AND {_LIVE_ONLY_CLAUSE}",
            (since,),
        ).fetchone()[0]
        urgent = self.conn.execute(
            f"SELECT COUNT(*) FROM articles WHERE first_seen >= ? AND urgency>=1 AND {_LIVE_ONLY_CLAUSE}",
            (since,),
        ).fetchone()[0]
        return {"total": total, "urgent": urgent}

    def close(self):
        self.conn.close()

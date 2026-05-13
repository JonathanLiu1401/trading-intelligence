"""
Compressed SQLite article store — prefers USB drive, falls back to local data/.
Stores article metadata + compressed full text. Auto-purges articles older than RETENTION_DAYS.
"""
import hashlib
import os
import sqlite3
import zlib
from datetime import datetime, timezone, timedelta
from pathlib import Path

USB_PATH = Path(os.environ.get("DIGITAL_INTERN_USB", "/media/zeph/projects/digital-intern/db"))
LOCAL_PATH = Path(__file__).resolve().parent.parent / "data"
RETENTION_DAYS = 14
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
    cycle       INTEGER DEFAULT 0   -- collection cycle number
);
CREATE INDEX IF NOT EXISTS idx_urgency   ON articles(urgency);
CREATE INDEX IF NOT EXISTS idx_ai_score  ON articles(ai_score DESC);
CREATE INDEX IF NOT EXISTS idx_first_seen ON articles(first_seen);
CREATE INDEX IF NOT EXISTS idx_cycle     ON articles(cycle);
"""


def _get_db_path() -> Path:
    if USB_PATH.parent.exists():
        try:
            USB_PATH.mkdir(parents=True, exist_ok=True)
            return USB_PATH / "articles.db"
        except PermissionError:
            pass
    LOCAL_PATH.mkdir(parents=True, exist_ok=True)
    return LOCAL_PATH / "articles.db"


def _connect() -> sqlite3.Connection:
    db = _get_db_path()
    conn = sqlite3.connect(str(db), timeout=60, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")   # 64MB cache
    conn.execute("PRAGMA busy_timeout=60000")  # wait up to 60s on lock
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def article_id(url: str, title: str) -> str:
    return hashlib.sha256(f"{url}||{title}".encode()).hexdigest()


def compress(text: str) -> bytes:
    return zlib.compress(text.encode("utf-8", errors="replace"), level=6)


def decompress(data: bytes) -> str:
    return zlib.decompress(data).decode("utf-8", errors="replace")


class ArticleStore:
    def __init__(self):
        self.conn = _connect()
        db = _get_db_path()
        print(f"[store] Using DB at {db} ({db.stat().st_size // 1024}KB)" if db.exists() else f"[store] New DB at {db}")

    def insert_batch(self, articles: list, cycle: int = 0) -> int:
        """Insert new articles; skip duplicates. Returns count of new insertions."""
        now = datetime.now(timezone.utc).isoformat()
        inserted = 0
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

    def update_ai_score(self, aid: str, score: float, urgency: int = 0):
        self.conn.execute(
            "UPDATE articles SET ai_score=?, urgency=MAX(urgency,?) WHERE id=?",
            (score, urgency, aid),
        )
        self.conn.commit()

    def mark_alerted(self, aid: str):
        self.conn.execute("UPDATE articles SET urgency=2 WHERE id=?", (aid,))
        self.conn.commit()

    def get_unscored(self, limit: int = 500, min_kw: float = 2) -> list:
        """Get articles that haven't been AI-scored yet."""
        cur = self.conn.execute(
            "SELECT id, url, title, source, full_text FROM articles "
            "WHERE ai_score=0 AND kw_score>=? ORDER BY kw_score DESC LIMIT ?",
            (min_kw, limit),
        )
        rows = cur.fetchall()
        return [
            {"_id": r[0], "link": r[1], "title": r[2], "source": r[3],
             "summary": decompress(r[4]) if r[4] else ""}
            for r in rows
        ]

    def get_unalerted_urgent(self) -> list:
        """Get articles scored urgent but not yet alerted."""
        cur = self.conn.execute(
            "SELECT id, url, title, source, ai_score FROM articles "
            "WHERE urgency=1 ORDER BY ai_score DESC"
        )
        rows = cur.fetchall()
        return [{"_id": r[0], "link": r[1], "title": r[2], "source": r[3], "ai_score": r[4]}
                for r in rows]

    def get_top_for_briefing(self, hours: int = 5, limit: int = 50) -> list:
        """Get highest-scoring articles from last N hours for the heartbeat briefing."""
        since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        cur = self.conn.execute(
            "SELECT id, url, title, source, ai_score, kw_score, full_text FROM articles "
            "WHERE first_seen >= ? ORDER BY ai_score DESC, kw_score DESC LIMIT ?",
            (since, limit),
        )
        rows = cur.fetchall()
        return [
            {"_id": r[0], "link": r[1], "title": r[2], "source": r[3],
             "ai_score": r[4], "_relevance_score": r[5],
             "summary": decompress(r[6]) if r[6] else ""}
            for r in rows
        ]

    def purge_old(self):
        """Delete articles older than RETENTION_DAYS and vacuum."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)).isoformat()
        cur = self.conn.execute("DELETE FROM articles WHERE first_seen < ?", (cutoff,))
        deleted = cur.rowcount
        self.conn.commit()
        if deleted > 0:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            print(f"[store] Purged {deleted} articles older than {RETENTION_DAYS} days")
        return deleted

    def stats(self) -> dict:
        total = self.conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        urgent = self.conn.execute("SELECT COUNT(*) FROM articles WHERE urgency>=1").fetchone()[0]
        unscored = self.conn.execute("SELECT COUNT(*) FROM articles WHERE ai_score=0").fetchone()[0]
        db = _get_db_path()
        size_mb = db.stat().st_size / 1024 / 1024 if db.exists() else 0
        return {"total": total, "urgent": urgent, "unscored": unscored, "db_mb": round(size_mb, 1)}

    def stats_since(self, hours: int) -> dict:
        since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        total = self.conn.execute(
            "SELECT COUNT(*) FROM articles WHERE first_seen >= ?", (since,)
        ).fetchone()[0]
        urgent = self.conn.execute(
            "SELECT COUNT(*) FROM articles WHERE first_seen >= ? AND urgency>=1", (since,)
        ).fetchone()[0]
        return {"total": total, "urgent": urgent}

    def close(self):
        self.conn.close()

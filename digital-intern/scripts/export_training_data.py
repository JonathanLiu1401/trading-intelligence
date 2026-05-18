"""
Export scored articles to training-data outputs co-located with the source DB:
  - training_data.json.gz : line-delimited JSON for ML training (all ai_score > 0)
  - paper_trader_signals.db : SQLite signals table for the backtester (ai_score >= 4.0)

Streams rows from the source DB to keep memory bounded.
"""
import gzip
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running this as a standalone script from /home/zeph/digital-intern.
_BASE = Path(__file__).resolve().parent.parent
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))

from storage.article_store import ArticleStore, decompress, _get_db_path  # noqa: E402


# ── Ticker extraction (copied verbatim from paper-trader/paper_trader/signals.py
#    so this module has no cross-package dependency on paper-trader) ──────────
_TICKER_RE = re.compile(r"\b([A-Z]{1,5})\b")
# common english noise that's all-caps but not tickers
_NOT_TICKERS = {
    "A", "I", "AI", "ALL", "AN", "AND", "AS", "AT", "BE", "BY", "BUT", "CEO", "CFO", "CTO",
    "DOJ", "ETF", "ETFS", "EU", "FBI", "FDA", "FOR", "FX", "GDP", "GOP", "IPO", "IT", "ITS",
    "ON", "OR", "PE", "PM", "QE", "QT", "RE", "SEC", "SO", "TBA", "THE", "TO", "UN", "UP",
    "US", "USA", "USD", "VS", "WE", "WTI", "YES", "NO", "YOY", "QOQ", "MOM", "Q1", "Q2",
    "Q3", "Q4", "FY", "OK", "EPS", "PE", "PB", "OF", "IS", "IN", "WHO", "WHAT", "WHEN",
    "WHERE", "WHY", "HOW", "NEW", "OLD", "ALL", "ANY", "ONE", "TWO", "MAY", "JUNE", "JULY",
    "AUG", "SEPT", "OCT", "NOV", "DEC", "JAN", "FEB", "MAR", "APR", "FED", "BOE", "ECB",
    "BOJ", "PBOC", "OPEC", "NATO", "WTO", "IMF", "WHO", "API", "CPI", "PPI", "GDP", "PMI",
    "ISM", "ADP", "EIA", "USDA", "BLS", "BEA", "FOMC",
}


def _extract_tickers(text: str) -> set[str]:
    """Heuristic ticker extraction — pulls $TICKER or ALLCAPS 1-5 char tokens, filters noise."""
    out = set()
    for m in re.finditer(r"\$([A-Z]{1,5})\b", text or ""):
        out.add(m.group(1))
    for m in _TICKER_RE.finditer(text or ""):
        tok = m.group(1)
        if tok in _NOT_TICKERS or len(tok) < 2:
            continue
        out.add(tok)
    return out


def export_all() -> dict:
    """Export scored articles to JSON.gz (all ai_score > 0) and SQLite (ai_score >= 4.0).

    Both outputs land in the same directory as the source articles DB.
    Streams rows so memory stays bounded.
    """
    db_path = _get_db_path()
    out_dir = db_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "training_data.json.gz"
    db_out_path = out_dir / "paper_trader_signals.db"

    # Open read-only cursor over the source DB
    src = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    src.execute("PRAGMA query_only=1")
    cur = src.execute(
        "SELECT id, title, source, ai_score, full_text, first_seen "
        "FROM articles WHERE ai_score > 0"
    )

    # paper_trader_signals.db is a fully-derived artifact rebuilt from the
    # source DB on every run (the JSON.gz already has rebuild-from-scratch
    # semantics via gzip.open(..., "wt")). Tearing the destination down first
    # makes the export both idempotent — rows whose ai_score fell below the
    # 4.0 threshold (or that were deleted at source) do NOT linger and pollute
    # the backtester's signal set — and self-healing: a corrupt/malformed
    # destination is replaced instead of raising "database disk image is
    # malformed" and crashing the whole export. Sidecars must go too or a
    # stale -wal can resurrect dropped rows.
    for _suffix in ("", "-wal", "-shm", "-journal"):
        try:
            os.unlink(f"{db_out_path}{_suffix}")
        except OSError:
            pass

    # Open destination SQLite
    dst = sqlite3.connect(str(db_out_path), timeout=30)
    dst.execute("PRAGMA journal_mode=WAL")
    dst.execute("PRAGMA synchronous=NORMAL")
    dst.execute(
        "CREATE TABLE IF NOT EXISTS signals ("
        "id TEXT PRIMARY KEY, "
        "title TEXT, "
        "source TEXT, "
        "ai_score REAL, "
        "tickers TEXT, "
        "first_seen TEXT, "
        "exported_at TEXT"
        ")"
    )

    exported_at = datetime.now(timezone.utc).isoformat()
    json_count = 0
    db_count = 0

    # Stream rows; write JSON line + (conditionally) signals row per article
    with gzip.open(json_path, "wt", encoding="utf-8") as gz:
        while True:
            rows = cur.fetchmany(500)
            if not rows:
                break
            db_inserts = []
            for r in rows:
                aid, title, source, ai_score, full_text, first_seen = r
                summary = decompress(full_text) if full_text else ""
                tickers = sorted(_extract_tickers(f"{title or ''} {summary}"))
                gz.write(json.dumps({
                    "id": aid,
                    "title": title,
                    "source": source,
                    "score": ai_score,
                    "tickers": tickers,
                    "ts": first_seen,
                }, ensure_ascii=False) + "\n")
                json_count += 1
                if ai_score is not None and ai_score >= 4.0:
                    db_inserts.append((
                        aid, title, source, float(ai_score),
                        ",".join(tickers), first_seen, exported_at,
                    ))
            if db_inserts:
                dst.executemany(
                    "INSERT OR REPLACE INTO signals "
                    "(id, title, source, ai_score, tickers, first_seen, exported_at) "
                    "VALUES (?,?,?,?,?,?,?)",
                    db_inserts,
                )
                dst.commit()
                db_count += len(db_inserts)

    src.close()
    dst.close()

    return {
        "json_count": json_count,
        "db_count": db_count,
        "json_path": str(json_path),
        "db_path": str(db_out_path),
    }


if __name__ == "__main__":
    result = export_all()
    print(result)

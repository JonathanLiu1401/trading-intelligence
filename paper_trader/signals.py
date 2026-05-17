"""Pull scored news signals + ML predictions from the digital-intern pipeline."""
import os
import re
import sys
import sqlite3
import time
import zlib
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

DIGITAL_INTERN = "/home/zeph/digital-intern"
if DIGITAL_INTERN not in sys.path:
    sys.path.insert(0, DIGITAL_INTERN)

# ── Article-DB resolution ────────────────────────────────────────────────
# Vendored from /home/zeph/paper-trader/paper_trader/signals.py — ported here
# (resolver only) so this snapshot can't reintroduce the USB-stale split-brain.
# The original returned the USB copy whenever it merely ``exists()``; once the
# daemon falls back to writing LOCAL, that USB file keeps existing while going
# stale and the live trader silently reads day-old news. Now picks the
# candidate whose newest *live* article is most recent; USB still preferred on
# a tie / when freshness is indeterminate. Data-sourcing fix, not a risk limit.
USB_DB = Path(os.environ.get("DIGITAL_INTERN_USB", "/media/zeph/projects/digital-intern/db")) / "articles.db"
LOCAL_DB = Path(DIGITAL_INTERN) / "data" / "articles.db"

_LIVE_ONLY_SQL = (
    "url NOT LIKE 'backtest://%' "
    "AND source NOT LIKE 'backtest_%' "
    "AND source NOT LIKE 'opus_annotation%'"
)

_DB_RESOLVE_TTL_S = 120.0
_STALE_FEED_WARN_HOURS = 6.0
_SPLIT_BRAIN_GAP_H = 6.0

_db_resolve_cache: tuple[tuple[Path, ...], float, Path] | None = None
_STALE_WARNED: set[str] = set()


def _candidates() -> tuple[Path, ...]:
    """Candidates in preference order (USB first). Read from module globals at
    call time so tests can monkeypatch ``USB_DB`` / ``LOCAL_DB``."""
    return (USB_DB, LOCAL_DB)


def _age_hours(first_seen: str | None) -> float | None:
    if not first_seen:
        return None
    try:
        dt = datetime.fromisoformat(first_seen.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0


def _live_newest_first_seen(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
        try:
            row = conn.execute(
                f"SELECT MAX(first_seen) FROM articles WHERE {_LIVE_ONLY_SQL}"
            ).fetchone()
        finally:
            conn.close()
        return row[0] if row and row[0] else None
    except Exception:
        return None


def _db_freshness() -> dict[Path, str | None]:
    return {p: _live_newest_first_seen(p) for p in _candidates() if p.exists()}


def _choose(freshness: dict[Path, str | None]) -> Path:
    existing = [p for p in _candidates() if p in freshness]
    if not existing:
        return LOCAL_DB
    if len(existing) == 1:
        return existing[0]
    best: Path | None = None
    best_ts: str | None = None
    for p in existing:
        ts = freshness.get(p)
        if ts is not None and (best_ts is None or ts > best_ts):
            best, best_ts = p, ts
    return best if best is not None else existing[0]


def _maybe_warn_stale(chosen: Path, freshness: dict[Path, str | None]) -> None:
    key = str(chosen)
    if key in _STALE_WARNED:
        return
    age = _age_hours(freshness.get(chosen))
    if age is None or age < _STALE_FEED_WARN_HOURS:
        return
    _STALE_WARNED.add(key)
    print(
        f"[signals] WARNING reading STALE feed {chosen} — newest live article "
        f"is {age:.1f}h old; live trader may be blind.",
        file=sys.stderr,
    )


def _db_path() -> Path:
    global _db_resolve_cache
    cands = _candidates()
    now = time.monotonic()
    if (
        _db_resolve_cache is not None
        and _db_resolve_cache[0] == cands
        and now - _db_resolve_cache[1] < _DB_RESOLVE_TTL_S
    ):
        return _db_resolve_cache[2]
    freshness = _db_freshness()
    chosen = _choose(freshness)
    _maybe_warn_stale(chosen, freshness)
    _db_resolve_cache = (cands, now, chosen)
    return chosen


def _reset_resolver_cache() -> None:
    global _db_resolve_cache
    _db_resolve_cache = None
    _STALE_WARNED.clear()


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


def _decompress(blob: bytes | None) -> str:
    if not blob:
        return ""
    try:
        return zlib.decompress(blob).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _connect_ro() -> sqlite3.Connection | None:
    path = _db_path()
    if not path.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        print(f"[signals] cannot open {path}: {e}")
        return None


def get_top_signals(n: int = 20, hours: int = 2, min_score: float = 4.0) -> list[dict]:
    """Top scored articles from the last N hours with ai_score >= min_score."""
    conn = _connect_ro()
    if not conn:
        return []
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    try:
        rows = conn.execute(
            "SELECT id, url, title, source, ai_score, urgency, first_seen, full_text "
            "FROM articles WHERE first_seen >= ? AND ai_score >= ? "
            "AND url NOT LIKE 'backtest://%' AND source NOT LIKE 'backtest_%' "
            "AND source NOT LIKE 'opus_annotation%' "
            "ORDER BY ai_score DESC, first_seen DESC LIMIT ?",
            (since, min_score, n),
        ).fetchall()
    finally:
        conn.close()
    out = []
    for r in rows:
        summary = _decompress(r["full_text"])
        out.append({
            "id": r["id"],
            "url": r["url"],
            "title": r["title"],
            "source": r["source"],
            "ai_score": r["ai_score"],
            "urgency": r["urgency"],
            "first_seen": r["first_seen"],
            "summary": summary[:400],
            "tickers": sorted(_extract_tickers(f"{r['title']} {summary}")),
        })
    return out


def get_ticker_sentiment(ticker: str, hours: int = 4) -> dict:
    """Average score + counts of articles mentioning the ticker."""
    conn = _connect_ro()
    if not conn:
        return {"ticker": ticker, "avg_score": 0.0, "n": 0, "urgent": 0}
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    try:
        rows = conn.execute(
            "SELECT title, full_text, ai_score, urgency FROM articles "
            "WHERE first_seen >= ? AND ai_score > 0 "
            "AND url NOT LIKE 'backtest://%' AND source NOT LIKE 'backtest_%' "
            "AND source NOT LIKE 'opus_annotation%'",
            (since,),
        ).fetchall()
    finally:
        conn.close()
    scores = []
    urgent = 0
    needle = ticker.upper()
    for r in rows:
        body = f"{r['title']} {_decompress(r['full_text'])}".upper()
        if re.search(rf"(?:\$|\b){needle}\b", body):
            scores.append(r["ai_score"])
            if r["urgency"] >= 1:
                urgent += 1
    if not scores:
        return {"ticker": ticker, "avg_score": 0.0, "n": 0, "urgent": urgent}
    return {
        "ticker": ticker,
        "avg_score": round(sum(scores) / len(scores), 2),
        "max_score": max(scores),
        "n": len(scores),
        "urgent": urgent,
    }


def get_urgent_articles(minutes: int = 30) -> list[dict]:
    """Articles flagged urgent (>=1) in the last N minutes."""
    conn = _connect_ro()
    if not conn:
        return []
    since = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()
    try:
        rows = conn.execute(
            "SELECT id, title, source, ai_score, urgency, first_seen, full_text "
            "FROM articles WHERE urgency >= 1 AND first_seen >= ? "
            "AND url NOT LIKE 'backtest://%' AND source NOT LIKE 'backtest_%' "
            "AND source NOT LIKE 'opus_annotation%' "
            "ORDER BY ai_score DESC LIMIT 20",
            (since,),
        ).fetchall()
    finally:
        conn.close()
    out = []
    for r in rows:
        summary = _decompress(r["full_text"])
        out.append({
            "id": r["id"],
            "title": r["title"],
            "source": r["source"],
            "ai_score": r["ai_score"],
            "urgency": r["urgency"],
            "first_seen": r["first_seen"],
            "summary": summary[:300],
            "tickers": sorted(_extract_tickers(f"{r['title']} {summary}")),
        })
    return out


def get_ml_predictions(articles: list[dict] | None = None) -> list[dict]:
    """Run digital-intern ML scoring against a candidate list of articles.

    If `articles` is omitted, scores the most recent unscored-or-low-score batch.
    Safe to return [] on failure — caller continues with rule-based signals.
    """
    try:
        from ml.inference import score_articles  # type: ignore
    except Exception as e:
        print(f"[signals] ML unavailable: {e}")
        return []

    if articles is None:
        articles = get_top_signals(30, hours=6, min_score=0.0)
    if not articles:
        return []

    try:
        scores = score_articles(articles)
    except Exception as e:
        print(f"[signals] ML inference failed: {e}")
        return []

    out = []
    for a, s in zip(articles, scores):
        out.append({
            "id": a.get("id"),
            "title": a.get("title"),
            "tickers": a.get("tickers", []),
            "relevance": s.relevance,
            "urgency": s.urgency,
            "rel_std": s.rel_std,
            "urg_std": s.urg_std,
            "needs_llm": s.needs_llm,
        })
    return out


def ticker_sentiments(tickers: list[str], hours: int = 4) -> list[dict]:
    """Bulk wrapper — one scan, scores aggregated per ticker."""
    conn = _connect_ro()
    if not conn:
        return [{"ticker": t, "avg_score": 0.0, "n": 0, "urgent": 0} for t in tickers]
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    try:
        rows = conn.execute(
            "SELECT title, full_text, ai_score, urgency FROM articles "
            "WHERE first_seen >= ? AND ai_score > 0 "
            "AND url NOT LIKE 'backtest://%' AND source NOT LIKE 'backtest_%' "
            "AND source NOT LIKE 'opus_annotation%'",
            (since,),
        ).fetchall()
    finally:
        conn.close()
    agg = defaultdict(lambda: {"scores": [], "urgent": 0})
    upper_tickers = [t.upper() for t in tickers]
    patterns = {t: re.compile(rf"(?:\$|\b){re.escape(t)}\b") for t in upper_tickers}
    for r in rows:
        body = f"{r['title']} {_decompress(r['full_text'])}".upper()
        for t, pat in patterns.items():
            if pat.search(body):
                agg[t]["scores"].append(r["ai_score"])
                if r["urgency"] >= 1:
                    agg[t]["urgent"] += 1
    out = []
    for t in upper_tickers:
        sc = agg[t]["scores"]
        out.append({
            "ticker": t,
            "avg_score": round(sum(sc) / len(sc), 2) if sc else 0.0,
            "max_score": max(sc) if sc else 0.0,
            "n": len(sc),
            "urgent": agg[t]["urgent"],
        })
    return out


if __name__ == "__main__":
    print("=== top signals ===")
    for s in get_top_signals(5):
        print(f"  [{s['ai_score']:.1f}] {s['title']!r:60} tickers={s['tickers']}")
    print("\n=== urgent ===")
    for s in get_urgent_articles():
        print(f"  [{s['urgency']}] {s['title']!r}")
    print("\n=== ticker sentiments ===")
    for r in ticker_sentiments(["NVDA", "MU", "AMD", "LITE"]):
        print(f"  {r}")

"""Pull scored news signals + ML predictions from the digital-intern pipeline."""
import gzip
import json
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
# digital-intern keeps the canonical store on the USB projects drive and
# falls back to writing the LOCAL copy when the USB mount is unavailable for
# writes. The original resolver returned the USB copy whenever it merely
# ``exists()`` — but a USB file keeps existing while going stale once the
# daemon silently switches to writing LOCAL. The live trader then reads
# day-old news while every other surface (daemon, unified dashboard) reads the
# fresh LOCAL DB — the "split brain" that was *detected* (/api/feed-health)
# but never root-fixed. We now pick the candidate whose newest *live* article
# is most recent, so the trader always reads the freshest feed regardless of
# which copy the daemon wrote. LOCAL order is preferred on a tie / when
# freshness cannot be determined: the live daemon's write path is LOCAL, so
# trying it first is the safest default (commit 6227cd5 flipped this from the
# old USB-first ordering — see _candidates()). This is a *data-sourcing* fix,
# not a risk limit: it changes which feed is read, never a trading decision
# (invariants #2/#12 untouched — same reasoning as the #13 valuation fix).
USB_DB = Path(os.environ.get("DIGITAL_INTERN_USB", "/media/zeph/projects/digital-intern/db")) / "articles.db"
LOCAL_DB = Path(DIGITAL_INTERN) / "data" / "articles.db"

# Canonical backtest-isolation filter (invariant #1 / #3). The freshness probe
# applies it too, so a fresh batch of injected synthetic rows on a stale
# mirror can never make that mirror *look* current and win the race.
_LIVE_ONLY_SQL = (
    "url NOT LIKE 'backtest://%' "
    "AND source NOT LIKE 'backtest_%' "
    "AND source NOT LIKE 'opus_annotation%'"
)

_DB_RESOLVE_TTL_S = 120.0        # re-probe at most this often (a cycle is ≥1800s)
_STALE_FEED_WARN_HOURS = 6.0     # one-shot WARN when the chosen feed is older
_SPLIT_BRAIN_GAP_H = 6.0         # "another candidate is materially fresher"

_db_resolve_cache: tuple[tuple[Path, ...], float, Path] | None = None
_STALE_WARNED: set[str] = set()


def _candidates() -> tuple[Path, ...]:
    """Article-DB candidates in *preference order* (LOCAL first — the live daemon
    writes here; USB is fallback when LOCAL is unavailable). Read from the module
    globals at call time so tests can monkeypatch ``USB_DB`` / ``LOCAL_DB``."""
    return (LOCAL_DB, USB_DB)


def _age_hours(first_seen: str | None) -> float | None:
    """Hours between ``first_seen`` (ISO-8601) and now (UTC); None if
    unparseable. ``first_seen`` is always an ISO insert timestamp (digital-
    intern schema), so a lexicographic ``>`` also compares it correctly —
    unlike ``published`` it never carries RFC822 dates."""
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
    """Newest live-article ``first_seen`` in ``path``, or None if the DB is
    unreadable / has no live rows."""
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
    """{candidate: newest_live_first_seen|None} for each candidate that exists."""
    return {p: _live_newest_first_seen(p) for p in _candidates() if p.exists()}


def _choose(freshness: dict[Path, str | None]) -> Path:
    """Pure chooser: the candidate with the newest live article.

    Iterates in preference order with a strict ``>``, so LOCAL wins a tie and is
    also the fallback when no freshness value is determinable."""
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
    """One-shot stderr WARN when the chosen feed's newest live article is older
    than ``_STALE_FEED_WARN_HOURS`` — turns a silently-blind trader into a
    visible one in the runner log. Deduped per chosen path so it never floods
    the cycle; names the fresher culprit when split-brain."""
    key = str(chosen)
    if key in _STALE_WARNED:
        return
    age = _age_hours(freshness.get(chosen))
    if age is None or age < _STALE_FEED_WARN_HOURS:
        return
    _STALE_WARNED.add(key)
    others = [
        f"{p.name}@{_age_hours(ts):.1f}h"
        for p, ts in freshness.items()
        if p != chosen and ts is not None
    ]
    extra = f" (fresher candidate(s): {', '.join(others)})" if others else ""
    print(
        f"[signals] WARNING reading STALE feed {chosen} — newest live article "
        f"is {age:.1f}h old{extra}; live trader may be blind. Diagnose with "
        f"`python3 -m paper_trader.signals --check-freshness`.",
        file=sys.stderr,
    )


def _db_path() -> Path:
    """Resolve the freshest live article DB (TTL-cached; the cache is keyed on
    the candidate tuple so a test monkeypatching the paths always re-resolves)."""
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
    """Test hook: drop the TTL cache and the one-shot WARN dedup set."""
    global _db_resolve_cache
    _db_resolve_cache = None
    _STALE_WARNED.clear()


def _legacy_choice() -> Path:
    """Model the *historical* pre-freshness-aware resolver, which was
    USB-first existence order ("return the USB copy whenever it merely
    exists()" — see the module docstring and commit 6227cd5).

    This is deliberately DECOUPLED from ``_candidates()``. ``_candidates()``
    returns the *current* resolver's tie-break order (LOCAL-first, since
    6227cd5); but split-brain detection asks a different question — "would a
    trader/dashboard process still running the pre-fix code be reading a stale
    feed?" — and that pre-fix code was USB-first. Reusing ``_candidates()``
    here made ``_legacy_choice()`` return LOCAL whenever LOCAL exists, which
    (a) defeated split-brain detection of the classic "stale USB mirror"
    failure and (b) *falsely* flagged split-brain when both feeds are merely
    stale but USB happens to be the fresher one (legacy=LOCAL≠chosen=USB)."""
    for p in (USB_DB, LOCAL_DB):
        if p.exists():
            return p
    return LOCAL_DB


def feed_status() -> dict:
    """Operator snapshot of feed resolution — consumed by the
    ``--check-freshness`` CLI and safe for any caller (pure reads).

    ``split_brain`` is the actionable signal: the freshest pick differs from
    what the legacy existence-first resolver would have picked **and** the
    legacy pick is materially staler. When True, a live trader still running
    pre-fix code (or any process that booted before the resolver landed —
    /api/build-info ``stale``) is reading day-old news and must be RESTARTED;
    the on-disk fix alone does not rescue the running process.
    ``stale`` is the orthogonal failure: the freshest copy *anywhere* is old
    (the whole digital-intern pipeline is down) — a restart would not help."""
    freshness = _db_freshness()
    chosen = _choose(freshness)
    chosen_age = _age_hours(freshness.get(chosen))
    legacy = _legacy_choice()
    legacy_age = _age_hours(freshness.get(legacy))
    candidates = []
    for p in _candidates():
        ts = freshness.get(p)
        candidates.append({
            "path": str(p),
            "exists": p.exists(),
            "newest_live_first_seen": ts,
            "age_hours": _age_hours(ts),
            "chosen": p == chosen,
        })
    split_brain = (
        legacy != chosen
        and legacy_age is not None
        and chosen_age is not None
        and legacy_age - chosen_age >= _SPLIT_BRAIN_GAP_H
    )
    return {
        "chosen": str(chosen),
        "chosen_age_hours": chosen_age,
        "legacy_choice": str(legacy),
        "legacy_age_hours": legacy_age,
        "stale": chosen_age is not None and chosen_age >= _STALE_FEED_WARN_HOURS,
        "split_brain": bool(split_brain),
        "candidates": candidates,
    }


_TICKER_RE = re.compile(r"\b([A-Z]{1,5})\b")
# common english noise that's all-caps but not tickers
_NOT_TICKERS = {
    "A", "I", "AI", "ALL", "AN", "AND", "ANY", "API", "APR", "AS", "AT",
    "AUG", "BE", "BEA", "BLS", "BOE", "BOJ", "BUT", "BY", "CEO", "CFO",
    "CPI", "CTO", "DEC", "DOJ", "ECB", "EIA", "EPS", "ETF", "ETFS", "EU",
    "FBI", "FDA", "FEB", "FED", "FOMC", "FOR", "FX", "FY", "GDP", "GOP",
    "HOW", "IMF", "IN", "IPO", "IS", "ISM", "IT", "ITS", "JAN", "JULY",
    "JUNE", "MAR", "MAY", "MOM", "NATO", "NEW", "NO", "NOV", "OCT", "OF",
    "OK", "OLD", "ON", "ONE", "OPEC", "OR", "PB", "PBOC", "PCE", "PE", "PM",
    "PMI", "PPI", "Q1", "Q2", "Q3", "Q4", "QE", "QOQ", "QT", "RE", "SEC",
    "SEPT", "SO", "TBA", "THE", "TO", "TWO", "UN", "UP", "US", "USA",
    "USD", "USDA", "VS", "WE", "WHAT", "WHEN", "WHERE", "WHO", "WHY",
    "WTI", "WTO", "YES", "YOY", "ADP",
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
        return {"ticker": ticker, "avg_score": 0.0, "max_score": 0.0, "n": 0, "urgent": 0}
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
    pattern = re.compile(rf"(?:\$|\b){re.escape(needle)}\b")
    for r in rows:
        body = f"{r['title']} {_decompress(r['full_text'])}".upper()
        if pattern.search(body):
            scores.append(r["ai_score"])
            if (r["urgency"] or 0) >= 1:
                urgent += 1
    if not scores:
        return {"ticker": ticker, "avg_score": 0.0, "max_score": 0.0, "n": 0, "urgent": urgent}
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
            # urgent rows are not score-filtered, so ai_score may be NULL —
            # coerce to 0.0 so downstream `f"{ai_score:.1f}"` formatting is safe.
            "ai_score": r["ai_score"] or 0.0,
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
        return [{"ticker": t, "avg_score": 0.0, "max_score": 0.0, "n": 0, "urgent": 0} for t in tickers]
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
        urg = (r["urgency"] or 0) >= 1
        for t, pat in patterns.items():
            if pat.search(body):
                agg[t]["scores"].append(r["ai_score"])
                if urg:
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


HISTORICAL_GZ = Path(
    os.environ.get(
        "DIGITAL_INTERN_HISTORICAL",
        "/media/zeph/projects/digital-intern/db/training_data.json.gz",
    )
)


def get_historical_signals(min_score: float = 4.0, limit: int | None = None) -> list[dict]:
    """Backtest-friendly fallback: read the gzip training-data export.

    Returns up to ``limit`` records with ``ai_score >= min_score`` (or all if
    ``limit`` is None). Returns [] and prints a short note if the file is missing.
    """
    if not HISTORICAL_GZ.exists():
        print(f"[signals] historical gzip missing at {HISTORICAL_GZ}")
        return []
    out: list[dict] = []
    try:
        with gzip.open(HISTORICAL_GZ, "rt", encoding="utf-8") as gz:
            for line in gz:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                try:
                    score = rec.get("score") or rec.get("ai_score")
                    if score is None or float(score) < min_score:
                        continue
                except (TypeError, ValueError):
                    # Non-numeric / corrupt score field — skip this record but
                    # keep reading the rest of the file.
                    continue
                out.append(rec)
                if limit is not None and len(out) >= limit:
                    break
    except Exception as e:
        print(f"[signals] historical read error: {e}")
        return []
    return out


def _print_freshness_report() -> int:
    """`--check-freshness` body. Returns a shell exit code:
    0 fresh · 2 stale-but-not-split · 3 split-brain (the actionable one)."""
    st = feed_status()
    print("=== article-DB freshness ===")
    for c in st["candidates"]:
        age = c["age_hours"]
        agestr = f"{age:.1f}h" if age is not None else "n/a"
        ex = "exists " if c["exists"] else "MISSING"
        mark = "  <- CHOSEN" if c["chosen"] else ""
        print(f"  [{ex}] {c['path']}  newest_live="
              f"{c['newest_live_first_seen'] or 'n/a'}  age={agestr}{mark}")
    ca, la = st["chosen_age_hours"], st["legacy_age_hours"]
    castr = f"{ca:.1f}h" if ca is not None else "n/a"
    lastr = f"{la:.1f}h" if la is not None else "n/a"
    print(f"\nfreshest pick : {st['chosen']}  (age {castr})")
    print(f"legacy pick   : {st['legacy_choice']}  (age {lastr})")
    if st["split_brain"]:
        print("\nSPLIT-BRAIN: the legacy existence-first resolver would read a "
              f"feed {(la - ca):.1f}h staler than the freshest copy. Any trader "
              "process that booted before this fix (/api/build-info `stale`) is "
              "STILL reading the stale feed and is effectively blind — the "
              "on-disk fix only takes effect on the NEXT start. RESTART the "
              "paper trader to apply.")
        return 3
    if st["stale"]:
        print(f"\nSTALE: the freshest copy anywhere is {castr} old "
              f"(>= {_STALE_FEED_WARN_HOURS:.0f}h). The digital-intern pipeline "
              "looks down — a trader restart will NOT help; fix the news daemon.")
        return 2
    print("\nOK: the freshest feed is current and the legacy resolver agrees.")
    return 0


if __name__ == "__main__":
    if "--check-freshness" in sys.argv:
        sys.exit(_print_freshness_report())
    print("=== top signals ===")
    for s in get_top_signals(5):
        print(f"  [{s['ai_score']:.1f}] {s['title']!r:60} tickers={s['tickers']}")
    print("\n=== urgent ===")
    for s in get_urgent_articles():
        print(f"  [{s['urgency']}] {s['title']!r}")
    print("\n=== ticker sentiments ===")
    for r in ticker_sentiments(["NVDA", "MU", "AMD", "LITE"]):
        print(f"  {r}")

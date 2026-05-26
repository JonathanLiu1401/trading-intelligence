"""NY Fed Liquidity Signals collector.

Fetches two key Federal Reserve liquidity metrics from the NY Fed's
free public API — no API key required:

  1. Overnight Reverse Repo (RRP) operations
     https://markets.newyorkfed.org/api/rp/reverserepo/propositions/search.json
     → How much cash money market funds / banks are parking at the Fed overnight.
       A declining RRP balance = liquidity flowing OUT of the Fed and into risk
       assets (bullish). Rising RRP = hoarding safety (risk-off).

  2. SOMA (System Open Market Account) holdings summary
     https://markets.newyorkfed.org/api/soma/summary.json
     → Weekly snapshot of Fed's treasury + MBS portfolio size.
       Weekly QT pace can be inferred from WoW change; faster runoff = tightening.

Both datasets are emitted as synthetic article rows so the pipeline can
score and brief on them. Dedup key: (source_tag, date) so each date emits
at most once.

Standalone usage / smoke test:
    python3 collectors/nyfed_liquidity_collector.py
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

log = logging.getLogger("nyfed_liquidity_collector")

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

SOURCE_RRP = "nyfed_rrp"
SOURCE_SOMA = "nyfed_soma"
REQUEST_TIMEOUT = 15
LOOKBACK_DAYS = 14  # how many calendar days back to fetch for RRP

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_HEADERS = {"User-Agent": _UA}


# ─── DB helpers ──────────────────────────────────────────────────────────────

def _ensure_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS seen_articles (
            id TEXT PRIMARY KEY, link TEXT, title TEXT,
            source TEXT, first_seen TEXT
        )"""
    )
    conn.commit()
    return conn


def _already_seen(conn: sqlite3.Connection, aid: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM seen_articles WHERE id=?", (aid,)
    ).fetchone() is not None


def _mark_seen(conn: sqlite3.Connection, aid: str, link: str, title: str, source: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
        "VALUES (?, ?, ?, ?, ?)",
        (aid, link, title, source, datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")),
    )
    conn.commit()


def _article_id(tag: str, date_str: str) -> str:
    key = f"{tag}|{date_str}"
    return hashlib.sha256(key.encode()).hexdigest()


# ─── RRP collector ───────────────────────────────────────────────────────────

def collect_rrp(lookback_days: int = LOOKBACK_DAYS) -> list[dict]:
    """Fetch NY Fed overnight reverse repo (RRP) operation data.

    Returns list of article-like dicts for new (unseen) operation dates.
    """
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=lookback_days)
    url = (
        "https://markets.newyorkfed.org/api/rp/reverserepo/propositions/search.json"
        f"?startDate={start_dt.strftime('%Y-%m-%d')}&endDate={end_dt.strftime('%Y-%m-%d')}"
    )

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.warning("[nyfed_rrp] fetch error: %s", e)
        return []

    operations = data.get("repo", {}).get("operations", [])
    if not operations:
        log.debug("[nyfed_rrp] no operations in response")
        return []

    conn = _ensure_db()
    articles: list[dict] = []

    # Sort oldest first so we process chronologically
    ops_sorted = sorted(operations, key=lambda x: x.get("operationDate", ""))

    # Collect last N for trend calculation
    recent_amounts = [
        float(op.get("totalAmtAccepted", 0)) / 1e9
        for op in ops_sorted
        if op.get("totalAmtAccepted") is not None
    ]

    for i, op in enumerate(ops_sorted):
        op_date = op.get("operationDate", "")
        amt_raw = op.get("totalAmtAccepted")
        if not op_date or amt_raw is None:
            continue

        aid = _article_id(SOURCE_RRP, op_date)
        if _already_seen(conn, aid):
            continue

        amt_b = float(amt_raw) / 1e9  # → $ billions

        # Compute WoW change if we have a prior point
        if i > 0 and recent_amounts[i - 1] > 0:
            prev = recent_amounts[i - 1]
            chg = amt_b - prev
            chg_str = f" (prev ${prev:.1f}B, Δ{'+' if chg>=0 else ''}{chg:.1f}B)"
        else:
            chg_str = ""

        # Signal interpretation
        if amt_b < 50:
            signal = "VERY LOW — excess liquidity fully drained; risk-on environment"
        elif amt_b < 200:
            signal = "LOW — most excess liquidity deployed"
        elif amt_b < 500:
            signal = "MODERATE — some safety hoarding"
        elif amt_b < 1000:
            signal = "ELEVATED — significant cash parked at Fed"
        else:
            signal = "HIGH — large-scale risk-off / excess liquidity overhang"

        title = (
            f"NY Fed Reverse Repo: ${amt_b:.1f}B on {op_date}{chg_str} — {signal}"
        )
        summary = (
            f"Fed overnight reverse repo operations on {op_date}: "
            f"${amt_b:.2f}B accepted. "
            f"RRP balance {signal.lower()}. "
            f"High RRP = excess cash hoarded at Fed (risk-off); "
            f"declining RRP = liquidity deploying into risk assets (bullish)."
        )
        link = "https://markets.newyorkfed.org/omo/dmm/reverserepo.do"

        article = {
            "title": title,
            "link": link,
            "summary": summary,
            "published": op_date + "T20:00:00Z",  # RRP results typically released ~3pm ET
            "source": SOURCE_RRP,
            "_tickers": ["SPY", "QQQ", "TLT", "SHY"],
        }
        articles.append(article)
        _mark_seen(conn, aid, link, title, SOURCE_RRP)

    log.info("[nyfed_rrp] %d new RRP records", len(articles))
    return articles


# ─── SOMA collector ──────────────────────────────────────────────────────────

def collect_soma(lookback_weeks: int = 4) -> list[dict]:
    """Fetch NY Fed SOMA holdings summary (weekly Fed balance sheet).

    Returns list of article-like dicts for new (unseen) weekly snapshots.
    """
    url = "https://markets.newyorkfed.org/api/soma/summary.json"

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.warning("[nyfed_soma] fetch error: %s", e)
        return []

    rows = data.get("soma", {}).get("summary", [])
    if not rows:
        log.debug("[nyfed_soma] no summary rows")
        return []

    # Only look at recent rows
    cutoff = (datetime.now(timezone.utc) - timedelta(weeks=lookback_weeks)).strftime("%Y-%m-%d")
    recent = [r for r in rows if r.get("asOfDate", "") >= cutoff]
    if not recent:
        recent = rows[-lookback_weeks:]  # fallback: last N rows

    conn = _ensure_db()
    articles: list[dict] = []

    for i, row in enumerate(recent):
        as_of = row.get("asOfDate", "")
        if not as_of:
            continue

        aid = _article_id(SOURCE_SOMA, as_of)
        if _already_seen(conn, aid):
            continue

        total = float(row.get("total") or 0) / 1e12  # → $ trillions
        mbs = float(row.get("mbs") or 0) / 1e12
        notes_bonds = float(row.get("notesbonds") or 0) / 1e12
        tips = float(row.get("tips") or 0) / 1e12
        bills = float(row.get("bills") or 0) / 1e12

        # WoW change
        if i > 0:
            prev_total = float(recent[i - 1].get("total") or 0) / 1e12
            delta = total - prev_total
            delta_str = f" (WoW Δ{'+' if delta>=0 else ''}{delta:.3f}T)"
            # Monthly QT pace
            qt_pace = delta * (52 / 1)  # annualized
        else:
            delta_str = ""
            delta = 0.0
            qt_pace = 0.0

        qt_signal = ""
        if delta < -0.010:
            qt_signal = " — QT accelerating (tightening)"
        elif delta < 0:
            qt_signal = " — QT continues (tightening)"
        elif delta > 0.005:
            qt_signal = " — balance sheet EXPANDING (easing)"
        else:
            qt_signal = " — balance sheet roughly stable"

        title = (
            f"Fed SOMA Holdings ({as_of}): ${total:.3f}T total"
            f"{delta_str}{qt_signal}"
        )
        summary = (
            f"NY Fed SOMA portfolio as of {as_of}: "
            f"Total ${total:.3f}T (Notes/Bonds ${notes_bonds:.3f}T, "
            f"MBS ${mbs:.3f}T, TIPS ${tips:.3f}T, Bills ${bills:.3f}T). "
            f"WoW change: {delta:+.3f}T.{qt_signal} "
            f"QT (quantitative tightening) shrinks Fed balance sheet, "
            f"draining bank reserves and tightening financial conditions."
        )
        link = "https://markets.newyorkfed.org/soma/sysopen_accholdings.html"

        article = {
            "title": title,
            "link": link,
            "summary": summary,
            "published": as_of + "T20:00:00Z",
            "source": SOURCE_SOMA,
            "_tickers": ["SPY", "TLT", "GLD", "IEF"],
        }
        articles.append(article)
        _mark_seen(conn, aid, link, title, SOURCE_SOMA)

    log.info("[nyfed_soma] %d new SOMA records", len(articles))
    return articles


# ─── Combined entry point ─────────────────────────────────────────────────────

def collect_nyfed_liquidity() -> list[dict]:
    """Collect all NY Fed liquidity signals."""
    articles: list[dict] = []
    articles.extend(collect_rrp())
    articles.extend(collect_soma())
    return articles


# ─── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results = collect_nyfed_liquidity()
    print(f"\n=== NY Fed Liquidity Signals: {len(results)} articles ===\n")
    for a in results:
        print(f"[{a['source']}] {a['title']}")
        print(f"  → {a['summary'][:120]}...")
        print()

"""FINRA Margin Statistics collector.

Tracks monthly margin debt and free credit balances published by FINRA at:
  https://www.finra.org/investors/learn-to-invest/advanced-investing/margin-statistics

Why this matters:
  - Margin debt is a key market leverage indicator; historical peaks precede
    major corrections (2000, 2008, 2021-22 bubble tops).
  - Rising margin debt + falling prices = forced liquidation risk (margin calls).
  - Free credit balances measure cash sitting on the sidelines (contrarian signal).
  - MoM change in debit balances is a leading risk-appetite gauge.

Emits one article per month per data point (deduped by month). Extreme
readings (debit balance MoM change ≥ |10%|) get an 'ALERT' prefix.

State: data/finra_margin_state.json tracks last seen month.
Dedup: seen_articles.db keyed by sha256('finra_margin'|month_year).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

try:
    from core.logger import get_logger
    log = get_logger("finra_margin")
except Exception:
    log = logging.getLogger("finra_margin")

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"
STATE_PATH = BASE_DIR / "data" / "finra_margin_state.json"

FETCH_URL = "https://www.finra.org/investors/learn-to-invest/advanced-investing/margin-statistics"
FETCH_TIMEOUT = 20
SOURCE = "finra/margin_statistics"

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_HEADERS = {
    "User-Agent": _UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# MoM % change threshold for elevated urgency
ALERT_THRESHOLD_PCT = 10.0

RELATED_TICKERS = ["SPY", "QQQ", "IWM", "VIX", "SQQQ", "SPXU"]


def _ensure_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS seen_articles (
            id TEXT PRIMARY KEY,
            link TEXT,
            title TEXT,
            source TEXT,
            first_seen TEXT
        )"""
    )
    conn.commit()


def _article_id(month_year: str, metric: str) -> str:
    raw = f"finra_margin|{month_year}|{metric}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _load_state() -> dict:
    try:
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return {}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def _parse_amount(s: str) -> float | None:
    """Parse a formatted dollar amount like '1,304,281' into a float."""
    cleaned = re.sub(r"[^\d.]", "", s)
    try:
        return float(cleaned)
    except ValueError:
        return None


def _mom_pct(current: float, prior: float) -> float | None:
    if prior == 0:
        return None
    return ((current - prior) / prior) * 100.0


def fetch_margin_data() -> list[dict]:
    """Scrape FINRA margin statistics table, return list of row dicts."""
    try:
        r = requests.get(FETCH_URL, headers=_HEADERS, timeout=FETCH_TIMEOUT)
        r.raise_for_status()
    except Exception as exc:
        log.warning("finra_margin: fetch failed: %s", exc)
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table")
    if not table:
        log.warning("finra_margin: no table found on page")
        return []

    rows = table.find_all("tr")
    results = []
    for row in rows[1:]:  # skip header
        cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
        if len(cells) < 4:
            continue
        month_year = cells[0]
        debit = _parse_amount(cells[1])
        free_cash = _parse_amount(cells[2])
        free_margin = _parse_amount(cells[3])
        if not month_year or debit is None:
            continue
        results.append({
            "month_year": month_year,
            "debit_balances": debit,
            "free_credit_cash": free_cash,
            "free_credit_margin": free_margin,
        })
    return results


def collect(conn: sqlite3.Connection | None = None) -> int:
    """Run one collection cycle. Returns count of new articles inserted."""
    _own_conn = conn is None
    if _own_conn:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)

    try:
        _ensure_db(conn)
        state = _load_state()
        rows = fetch_margin_data()
        if not rows:
            log.info("finra_margin: no data fetched")
            return 0

        now_iso = datetime.now(timezone.utc).isoformat()
        inserted = 0

        # rows are newest-first; compute MoM for each row where prior is available
        for i, row in enumerate(rows):
            month_year = row["month_year"]
            debit = row["debit_balances"]
            free_cash = row["free_credit_cash"]
            free_margin = row["free_credit_margin"]

            # MoM change (compare to next entry which is one month prior)
            mom_pct = None
            if i + 1 < len(rows):
                prior_debit = rows[i + 1]["debit_balances"]
                if prior_debit:
                    mom_pct = _mom_pct(debit, prior_debit)

            # Build article for debit balance
            art_id = _article_id(month_year, "debit")
            existing = conn.execute(
                "SELECT id FROM seen_articles WHERE id=?", (art_id,)
            ).fetchone()
            if not existing:
                total_credit = (free_cash or 0) + (free_margin or 0)
                mom_str = f" (MoM: {mom_pct:+.1f}%)" if mom_pct is not None else ""

                prefix = ""
                if mom_pct is not None and abs(mom_pct) >= ALERT_THRESHOLD_PCT:
                    direction = "SURGE" if mom_pct > 0 else "DROP"
                    prefix = f"ALERT: MARGIN DEBT {direction} — "

                title = (
                    f"{prefix}FINRA Margin Debt {month_year}: "
                    f"${debit/1000:.1f}B debit{mom_str}, "
                    f"${total_credit/1000:.1f}B free credit"
                )
                tickers = " ".join(RELATED_TICKERS)
                summary = (
                    f"FINRA margin statistics for {month_year}: "
                    f"Debit balances (margin debt) = ${debit:,.0f}M{mom_str}. "
                    f"Free credit (cash accounts) = ${free_cash:,.0f}M. "
                    f"Free credit (margin accounts) = ${free_margin:,.0f}M. "
                    f"Total free credit = ${total_credit:,.0f}M. "
                    f"Related: {tickers}. "
                    f"High margin debt relative to prior peaks signals elevated crash risk."
                )
                conn.execute(
                    "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) VALUES (?,?,?,?,?)",
                    (art_id, FETCH_URL, title, SOURCE, now_iso),
                )
                # Also write to main articles table if it exists
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO articles (id, url, title, source, summary, first_seen) VALUES (?,?,?,?,?,?)",
                        (art_id, FETCH_URL, title, SOURCE, summary, now_iso),
                    )
                except Exception:
                    pass
                conn.commit()
                inserted += 1
                log.info("finra_margin: new entry %s: %s", month_year, title[:80])

        state["last_run"] = now_iso
        state["latest_month"] = rows[0]["month_year"] if rows else state.get("latest_month")
        _save_state(state)
        return inserted

    finally:
        if _own_conn:
            conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    n = collect()
    print(f"finra_margin: inserted {n} new articles")

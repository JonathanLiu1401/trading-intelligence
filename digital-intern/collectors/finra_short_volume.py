"""FINRA RegSHO consolidated short volume collector.

Fetches daily short-sale volume data from FINRA's REGSHO CDN:
  https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt

Format: Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market

Computes short ratio = ShortVolume / TotalVolume for each ticker.
Inserts synthetic article rows for:
  - Portfolio/watchlist tickers (any ratio)
  - Any ticker with short ratio > 0.70 (potential squeeze setup)

Uses articles.db directly (like other collectors) with source='finra_short_volume'.
Deduplicates by date+ticker so the same day's data is only inserted once.
"""
import hashlib
import json
import sqlite3
import zlib
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
PORTFOLIO_PATH = BASE_DIR / "config" / "portfolio.json"
WATCHLIST_PATH = BASE_DIR / "config" / "watchlist.json"
DB_PATH = BASE_DIR / "data" / "articles.db"

FINRA_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"
USER_AGENT = "Digital-Intern-Daemon contact@digital-intern.local"

HIGH_SHORT_THRESHOLD = 0.70  # flag any ticker above this ratio
SOURCE = "finra_short_volume"

# Look back up to N trading days to find the latest available file
LOOKBACK_DAYS = 5


def _load_tracked_tickers() -> set[str]:
    tickers: set[str] = set()
    try:
        with open(PORTFOLIO_PATH) as f:
            pf = json.load(f)
        for pos in pf.get("positions", []):
            t = (pos.get("ticker") or "").strip().upper()
            if t:
                tickers.add(t)
        for opt in pf.get("options", []):
            t = (opt.get("underlying") or "").strip().upper()
            if t:
                tickers.add(t)
        for t in pf.get("sector_watchlist", []):
            u = t.strip().upper()
            if u:
                tickers.add(u)
    except Exception:
        pass
    try:
        with open(WATCHLIST_PATH) as f:
            wl = json.load(f)
        for t in wl if isinstance(wl, list) else wl.get("tickers", []):
            u = (t or "").strip().upper()
            if u:
                tickers.add(u)
    except Exception:
        pass
    return tickers


def _article_id(date_str: str, ticker: str) -> str:
    return hashlib.sha256(f"finra_short|{date_str}|{ticker}".encode()).hexdigest()


def _ensure_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _fetch_short_data(trade_date: str) -> list[dict] | None:
    url = FINRA_URL.format(date=trade_date)
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": USER_AGENT})
        if r.status_code == 404:
            return None
        r.raise_for_status()
    except Exception as e:
        print(f"[finra_short] fetch error for {trade_date}: {e}")
        return None

    rows = []
    for line in r.text.splitlines()[1:]:  # skip header
        parts = line.strip().split("|")
        if len(parts) < 5:
            continue
        try:
            sym = parts[1].strip().upper()
            short_vol = float(parts[2])
            total_vol = float(parts[4])
            if total_vol <= 0:
                continue
            ratio = short_vol / total_vol
            rows.append({"date": parts[0], "symbol": sym, "short_vol": short_vol,
                         "total_vol": total_vol, "ratio": ratio})
        except (ValueError, IndexError):
            continue
    return rows


def collect_finra_short_volume(max_items: int = 200) -> list[dict]:
    tracked = _load_tracked_tickers()
    conn = _ensure_db()
    now_utc = datetime.now(timezone.utc).isoformat()

    # Walk back to find latest available trading day
    rows = None
    trade_date = None
    for delta in range(LOOKBACK_DAYS):
        d = date.today() - timedelta(days=delta)
        if d.weekday() >= 5:  # skip weekends
            continue
        ds = d.strftime("%Y%m%d")
        data = _fetch_short_data(ds)
        if data is not None:
            rows = data
            trade_date = ds
            break

    if not rows:
        print("[finra_short] no data found in lookback window")
        conn.close()
        return []

    # Build lookup
    by_symbol = {r["symbol"]: r for r in rows}

    articles = []

    # 1) Portfolio/watchlist tickers — always include
    for ticker in tracked:
        row = by_symbol.get(ticker)
        if not row:
            continue
        art_id = _article_id(trade_date, ticker)
        ratio_pct = row["ratio"] * 100
        title = (
            f"FINRA Short Volume {trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}: "
            f"{ticker} short ratio {ratio_pct:.1f}% "
            f"({int(row['short_vol']):,} / {int(row['total_vol']):,})"
        )
        url = f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{trade_date}.txt#{ticker}"
        body = (
            f"FINRA RegSHO consolidated short volume for {ticker} on "
            f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}:\n"
            f"  Short volume: {int(row['short_vol']):,}\n"
            f"  Total volume: {int(row['total_vol']):,}\n"
            f"  Short ratio:  {ratio_pct:.2f}%\n"
        )
        if row["ratio"] >= HIGH_SHORT_THRESHOLD:
            body += f"  ⚠ HIGH SHORT RATIO — potential squeeze candidate\n"
        urgency = 1 if row["ratio"] >= HIGH_SHORT_THRESHOLD else 0
        articles.append({
            "id": art_id, "url": url, "title": title,
            "source": SOURCE, "published": now_utc,
            "kw_score": min(row["ratio"] * 2, 1.0),
            "urgency": urgency,
            "full_text": body,
            "first_seen": now_utc,
        })

    # 2) High short-ratio tickers NOT in portfolio (wider market signal)
    high_short = sorted(
        [r for r in rows if r["ratio"] >= HIGH_SHORT_THRESHOLD],
        key=lambda r: r["ratio"], reverse=True
    )[:50]  # cap at top-50 to avoid flooding
    for row in high_short:
        ticker = row["symbol"]
        if ticker in tracked:
            continue  # already added above
        art_id = _article_id(trade_date, ticker)
        ratio_pct = row["ratio"] * 100
        title = (
            f"FINRA High Short Volume Alert {trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}: "
            f"{ticker} {ratio_pct:.1f}% shorted ({int(row['short_vol']):,}/{int(row['total_vol']):,})"
        )
        url = f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{trade_date}.txt#{ticker}"
        body = (
            f"High short volume alert for {ticker} on "
            f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}:\n"
            f"  Short volume: {int(row['short_vol']):,}\n"
            f"  Total volume: {int(row['total_vol']):,}\n"
            f"  Short ratio:  {ratio_pct:.2f}% (threshold: {HIGH_SHORT_THRESHOLD*100:.0f}%)\n"
            f"  Potential short squeeze candidate.\n"
        )
        articles.append({
            "id": art_id, "url": url, "title": title,
            "source": SOURCE, "published": now_utc,
            "kw_score": row["ratio"],
            "urgency": 0,
            "full_text": body,
            "first_seen": now_utc,
        })

    if not articles:
        print(f"[finra_short] {trade_date}: 0 articles (no tracked tickers matched, 0 high-short)")
        conn.close()
        return []

    # Dedup and insert
    inserted = 0
    try:
        cur = conn.cursor()
        for art in articles[:max_items]:
            try:
                compressed = zlib.compress(art["full_text"].encode("utf-8"))
                cur.execute(
                    """INSERT OR IGNORE INTO articles
                       (id, url, title, source, published, kw_score, urgency, full_text, first_seen)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (art["id"], art["url"], art["title"], art["source"],
                     art["published"], art["kw_score"], art["urgency"],
                     compressed, art["first_seen"]),
                )
                if cur.rowcount:
                    inserted += 1
            except sqlite3.Error as e:
                print(f"[finra_short] db insert error for {art.get('title','?')}: {e}")
        conn.commit()
    finally:
        conn.close()

    print(f"[finra_short] {trade_date}: {len(articles)} candidates, {inserted} new inserted "
          f"({len(tracked)} tracked tickers, {len(high_short)} high-short alerts)")
    return [a for a in articles[:inserted] if a]

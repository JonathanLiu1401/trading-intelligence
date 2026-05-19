"""CoinGecko crypto market sentiment collector — synthetic 'article' rows
from the free CoinGecko public API (no API key required).

Why crypto for an equities/news pipeline:
  Crypto trades 24/7 and reacts to risk sentiment faster than equities. BTC
  dominance + total market cap shifts are a real-time read on global risk
  appetite — useful as a leading signal for the regular-session open.

Emits three article streams:
  1. ``coingecko/global`` — global market snapshot: total market cap, total
     24h volume, BTC/ETH dominance, market-cap change %. One article per
     fetch (date-keyed so we don't re-emit minute-to-minute).
  2. ``coingecko/movers`` — top gainer + top loser from the top-100 by mkt
     cap (24h % change). Two articles per fetch, deduped by symbol|date.
  3. ``coingecko/btc`` and ``coingecko/eth`` — daily price + 24h change for
     the two flagships, the canonical risk-on/risk-off pair.

Dedup matches the rest of the pipeline:
  - ``data/seen_articles.db`` (WAL, busy_timeout=30000), key = stream|date.
  - ``articles.db`` PRIMARY KEY = sha256(url||title) in insert_batch.
"""
import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

CG_GLOBAL_URL = "https://api.coingecko.com/api/v3/global"
CG_MARKETS_URL = (
    "https://api.coingecko.com/api/v3/coins/markets"
    "?vs_currency=usd&order=market_cap_desc&per_page=100&page=1"
    "&price_change_percentage=24h"
)
COINGECKO_HOME = "https://www.coingecko.com/"

FETCH_TIMEOUT = 15

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


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


def _seen_id(stream: str, key: str) -> str:
    return hashlib.sha256(f"coingecko:{stream}:{key}".encode("utf-8")).hexdigest()


def _is_seen(conn, sid: str) -> bool:
    return conn.execute("SELECT 1 FROM seen_articles WHERE id=?", (sid,)).fetchone() is not None


def _mark_seen(conn, sid: str, link: str, title: str, source: str):
    conn.execute(
        "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
        "VALUES (?, ?, ?, ?, ?)",
        (sid, link, title, source, datetime.now(timezone.utc).isoformat()),
    )


def _fmt_usd(x: float) -> str:
    if x >= 1e12:
        return f"${x/1e12:.2f}T"
    if x >= 1e9:
        return f"${x/1e9:.2f}B"
    if x >= 1e6:
        return f"${x/1e6:.2f}M"
    if x >= 1:
        return f"${x:,.2f}"
    return f"${x:.6f}"


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _fetch_json(url: str) -> dict | list:
    r = requests.get(url, timeout=FETCH_TIMEOUT, headers={"User-Agent": _UA})
    r.raise_for_status()
    return r.json()


def _global_article(conn, payload: dict) -> list[dict]:
    data = payload.get("data") or {}
    if not data:
        return []
    today = _today()
    sid = _seen_id("global", today)
    if _is_seen(conn, sid):
        return []
    total_mc = (data.get("total_market_cap") or {}).get("usd", 0.0)
    total_vol = (data.get("total_volume") or {}).get("usd", 0.0)
    dom = data.get("market_cap_percentage") or {}
    btc_dom = dom.get("btc", 0.0)
    eth_dom = dom.get("eth", 0.0)
    mc_change = data.get("market_cap_change_percentage_24h_usd", 0.0)
    n_active = data.get("active_cryptocurrencies", 0)

    title = (
        f"Crypto global {today}: total cap {_fmt_usd(total_mc)} "
        f"({mc_change:+.2f}% 24h), BTC dom {btc_dom:.1f}%, "
        f"ETH dom {eth_dom:.1f}%, 24h vol {_fmt_usd(total_vol)}"
    )
    body = (
        f"CoinGecko global crypto snapshot for {today}. "
        f"Total market cap: {_fmt_usd(total_mc)} "
        f"(24h change {mc_change:+.2f}%). "
        f"Total 24h volume: {_fmt_usd(total_vol)}. "
        f"BTC dominance: {btc_dom:.2f}%. ETH dominance: {eth_dom:.2f}%. "
        f"Active cryptocurrencies tracked: {n_active}. "
        f"Crypto is a 24/7 risk-sentiment proxy — BTC dominance rising in "
        f"a falling total-cap tape = altcoin de-risk, classic flight to BTC."
    )
    _mark_seen(conn, sid, COINGECKO_HOME, title, "coingecko/global")
    return [{
        "title": title,
        "link": COINGECKO_HOME,
        "summary": body,
        "published": today,
        "source": "coingecko/global",
    }]


def _movers_articles(conn, markets: list[dict]) -> list[dict]:
    if not markets:
        return []
    today = _today()
    valid = [m for m in markets if m.get("price_change_percentage_24h") is not None]
    if not valid:
        return []
    gainer = max(valid, key=lambda m: m["price_change_percentage_24h"])
    loser = min(valid, key=lambda m: m["price_change_percentage_24h"])
    out: list[dict] = []
    for tag, coin in (("gainer", gainer), ("loser", loser)):
        sym = (coin.get("symbol") or "?").upper()
        name = coin.get("name") or sym
        price = coin.get("current_price") or 0.0
        chg = coin.get("price_change_percentage_24h") or 0.0
        mc = coin.get("market_cap") or 0.0
        sid = _seen_id(f"movers/{tag}", f"{sym}|{today}")
        if _is_seen(conn, sid):
            continue
        link = f"https://www.coingecko.com/en/coins/{coin.get('id','')}"
        direction = "leading 24h gainer" if tag == "gainer" else "biggest 24h loser"
        title = (
            f"Crypto top-100 {direction} {today}: {sym} ({name}) "
            f"{chg:+.2f}% to {_fmt_usd(price)}"
        )
        body = (
            f"CoinGecko top-100 by market cap, {direction} for {today}: "
            f"{name} ({sym}) at {_fmt_usd(price)}, 24h change {chg:+.2f}%, "
            f"market cap {_fmt_usd(mc)}. "
            f"Outsized single-name moves in the top-100 often telegraph "
            f"narrative rotations (AI tokens, RWA, L2s) before mainstream "
            f"coverage picks them up."
        )
        _mark_seen(conn, sid, link, title, f"coingecko/movers")
        out.append({
            "title": title,
            "link": link,
            "summary": body,
            "published": today,
            "source": f"coingecko/movers",
        })
    return out


def _flagship_articles(conn, markets: list[dict]) -> list[dict]:
    if not markets:
        return []
    today = _today()
    by_id = {m.get("id"): m for m in markets}
    out: list[dict] = []
    for cid, label in (("bitcoin", "BTC"), ("ethereum", "ETH")):
        m = by_id.get(cid)
        if not m:
            continue
        chg = m.get("price_change_percentage_24h")
        if chg is None:
            continue
        sid = _seen_id(f"flagship/{label}", today)
        if _is_seen(conn, sid):
            continue
        price = m.get("current_price") or 0.0
        vol = m.get("total_volume") or 0.0
        link = f"https://www.coingecko.com/en/coins/{cid}"
        title = (
            f"{label} {today}: {_fmt_usd(price)} ({chg:+.2f}% 24h), "
            f"vol {_fmt_usd(vol)}"
        )
        body = (
            f"CoinGecko flagship snapshot for {label} on {today}. "
            f"Price: {_fmt_usd(price)}. 24h change: {chg:+.2f}%. "
            f"24h volume: {_fmt_usd(vol)}. "
            f"BTC/ETH price action is the canonical risk-on/risk-off pulse "
            f"used as a cross-asset sentiment input."
        )
        _mark_seen(conn, sid, link, title, f"coingecko/{label.lower()}")
        out.append({
            "title": title,
            "link": link,
            "summary": body,
            "published": today,
            "source": f"coingecko/{label.lower()}",
        })
    return out


def collect_coingecko() -> list[dict]:
    """Collect deduplicated synthetic crypto-sentiment articles from CoinGecko.

    Returns standard collector dicts: {title, link, summary, published, source}.
    """
    conn = _ensure_db()
    out: list[dict] = []
    try:
        g = _fetch_json(CG_GLOBAL_URL)
        out.extend(_global_article(conn, g))
    except Exception as e:
        print(f"[coingecko_collector] global fetch failed: {e}")
    try:
        m = _fetch_json(CG_MARKETS_URL)
        if isinstance(m, list):
            out.extend(_movers_articles(conn, m))
            out.extend(_flagship_articles(conn, m))
    except Exception as e:
        print(f"[coingecko_collector] markets fetch failed: {e}")
    conn.commit()
    conn.close()
    return out


collect = collect_coingecko


if __name__ == "__main__":
    print("=== CoinGecko live fetch ===")
    try:
        g = _fetch_json(CG_GLOBAL_URL)
        d = g.get("data", {})
        tmc = (d.get("total_market_cap") or {}).get("usd", 0)
        chg = d.get("market_cap_change_percentage_24h_usd", 0)
        btc_dom = (d.get("market_cap_percentage") or {}).get("btc", 0)
        print(f"  Global: total cap {_fmt_usd(tmc)} ({chg:+.2f}% 24h), BTC dom {btc_dom:.2f}%")
    except Exception as e:
        print(f"  Global fetch FAILED: {e}")

    eg_line = None
    try:
        m = _fetch_json(CG_MARKETS_URL)
        if isinstance(m, list):
            print(f"  Top-100 markets fetched: {len(m)} coins")
            valid = [c for c in m if c.get("price_change_percentage_24h") is not None]
            if valid:
                top = sorted(valid, key=lambda c: c["price_change_percentage_24h"], reverse=True)
                for c in top[:3]:
                    sym = (c.get("symbol") or "?").upper()
                    print(f"    + {sym:6s} {c['price_change_percentage_24h']:+6.2f}% @ {_fmt_usd(c['current_price'] or 0)}")
                if eg_line is None:
                    c = top[0]
                    eg_line = f"{(c.get('symbol') or '').upper()} {c['price_change_percentage_24h']:+.2f}%"
    except Exception as e:
        print(f"  Markets fetch FAILED: {e}")

    items = collect_coingecko()
    inserted = 0
    if items:
        from storage.article_store import ArticleStore
        store = ArticleStore()
        inserted = store.insert_batch(items)

    print("\n=== Summary ===")
    print(f"New synthetic articles built : {len(items)}")
    print(f"Total new items inserted into articles.db : {inserted}")
    if eg_line:
        print(f"DISCORD_EG: {eg_line}")
    for a in items[:8]:
        print(f"  + {a['title']}")

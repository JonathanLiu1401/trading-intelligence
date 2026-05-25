"""DeFiLlama TVL (Total Value Locked) collector.

DeFi TVL is a distinct signal from crypto prices: large TVL changes reveal
on-chain capital flows, protocol failures (sudden drops), chain migrations
(ETH→Solana), and risk-on/risk-off sentiment. Complements coingecko_collector.

Endpoints (all free, no API key):
  - /v2/chains          : per-chain TVL with 1d/7d change
  - /protocols          : top protocols with TVL change_1d/change_7d
  - /v2/historicalChainTvl : total DeFi TVL history (for trend context)

Emits synthetic articles when:
  1. Global DeFi TVL changes >3% in 24h    → macro signal
  2. Any major chain TVL changes >5% in 24h → chain rotation signal
  3. Top protocol TVL changes >10% in 24h  → protocol event signal

Dedup: seen_articles.db keyed by "defillama|<type>|<name>|<date_utc>"
Each signal emits at most once per calendar day.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import requests

log = logging.getLogger("defillama_tvl")

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "defillama_seen.db"

CHAINS_URL = "https://api.llama.fi/v2/chains"
PROTOCOLS_URL = "https://api.llama.fi/protocols"
HIST_TVL_URL = "https://api.llama.fi/v2/historicalChainTvl"

FETCH_TIMEOUT = 15
SOURCE_CHAINS = "defillama/chains"
SOURCE_PROTOCOLS = "defillama/protocols"
SOURCE_GLOBAL = "defillama/global"

# Thresholds for signal emission
GLOBAL_TVL_CHANGE_PCT = 3.0     # emit if total DeFi TVL moves >3% in 24h
CHAIN_TVL_CHANGE_PCT = 5.0      # emit if a major chain TVL moves >5% in 24h
PROTOCOL_TVL_CHANGE_PCT = 10.0  # emit if a top protocol TVL moves >10% in 24h

# Top N protocols to monitor (by TVL rank)
TOP_PROTOCOLS = 30
# Major chains to track for rotation signals
MAJOR_CHAINS = {
    "Ethereum", "Solana", "BSC", "Bitcoin", "Tron", "Arbitrum",
    "Base", "Polygon", "Avalanche", "Optimism", "Sui", "Aptos",
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


def _ensure_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS seen_defillama "
        "(key TEXT PRIMARY KEY, first_seen TEXT)"
    )
    conn.commit()
    return conn


def _seen(conn: sqlite3.Connection, key: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM seen_defillama WHERE key=?", (key,)
    ).fetchone()
    return row is not None


def _mark_seen(conn: sqlite3.Connection, key: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO seen_defillama (key, first_seen) VALUES (?,?)",
        (key, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def _article_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _fmt_tvl(tvl: float) -> str:
    """Format TVL as human-readable string."""
    if tvl >= 1e9:
        return f"${tvl/1e9:.2f}B"
    elif tvl >= 1e6:
        return f"${tvl/1e6:.1f}M"
    else:
        return f"${tvl:,.0f}"


def _fetch_global_tvl() -> list[dict]:
    """Fetch global DeFi TVL history and emit if >3% 24h change."""
    articles: list[dict] = []
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    try:
        resp = requests.get(HIST_TVL_URL, headers=_HEADERS, timeout=FETCH_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.warning("defillama: historicalChainTvl fetch failed: %s", e)
        return articles

    if not isinstance(data, list) or len(data) < 2:
        return articles

    latest = data[-1].get("tvl", 0)
    prev_day = data[-2].get("tvl", 0) if len(data) >= 2 else 0
    week_ago = data[-8].get("tvl", 0) if len(data) >= 8 else 0

    if prev_day > 0:
        pct_1d = (latest - prev_day) / prev_day * 100
    else:
        pct_1d = 0.0

    if week_ago > 0:
        pct_7d = (latest - week_ago) / week_ago * 100
    else:
        pct_7d = 0.0

    conn = _ensure_db()
    key = f"defillama|global|{today}"
    if not _seen(conn, key) and abs(pct_1d) >= GLOBAL_TVL_CHANGE_PCT:
        direction = "up" if pct_1d > 0 else "down"
        title = (
            f"[DeFi/TVL] Total DeFi TVL {direction} {abs(pct_1d):.1f}% in 24h "
            f"→ {_fmt_tvl(latest)} (7d: {pct_7d:+.1f}%)"
        )
        summary = (
            f"Total DeFi TVL moved {pct_1d:+.1f}% in 24 hours to {_fmt_tvl(latest)}. "
            f"7-day change: {pct_7d:+.1f}%. "
            f"Previous day: {_fmt_tvl(prev_day)}. "
            f"Signal: {'capital inflow' if pct_1d > 0 else 'capital outflow or deleverage'}."
        )
        art_key = f"defillama_global_{today}"
        articles.append({
            "title": title,
            "link": f"https://defillama.com/#{art_key}",
            "summary": summary,
            "published": datetime.now(timezone.utc).isoformat(),
            "source": SOURCE_GLOBAL,
            "article_id": _article_id(art_key),
        })
        _mark_seen(conn, key)
        log.info("defillama: global TVL signal: %s", title)
    elif not _seen(conn, key):
        # Still record that we checked today (no signal needed)
        _mark_seen(conn, key)

    conn.close()
    return articles


def _fetch_chain_signals() -> list[dict]:
    """Emit signal articles for major chains with large 24h TVL changes."""
    articles: list[dict] = []
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    try:
        resp = requests.get(CHAINS_URL, headers=_HEADERS, timeout=FETCH_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.warning("defillama: chains fetch failed: %s", e)
        return articles

    conn = _ensure_db()
    for chain in data:
        name = chain.get("name", "")
        if name not in MAJOR_CHAINS:
            continue
        tvl = chain.get("tvl", 0)
        change_1d = chain.get("change_1d", None)
        change_7d = chain.get("change_7d", None)

        if change_1d is None or tvl < 100_000_000:  # skip <$100M chains
            continue

        key = f"defillama|chain|{name}|{today}"
        if not _seen(conn, key) and abs(change_1d) >= CHAIN_TVL_CHANGE_PCT:
            direction = "surge" if change_1d > 0 else "drain"
            c7 = f"{change_7d:+.1f}%" if change_7d is not None else "N/A"
            title = (
                f"[DeFi/TVL] {name} chain TVL {direction} {change_1d:+.1f}% 24h "
                f"→ {_fmt_tvl(tvl)} (7d: {c7})"
            )
            summary = (
                f"{name} chain TVL changed {change_1d:+.1f}% in 24h to {_fmt_tvl(tvl)}. "
                f"7-day change: {c7}. "
                f"{'Capital rotating into' if change_1d > 0 else 'Capital exiting'} "
                f"{name} chain — watch bridging flows and {name}-native DeFi protocols."
            )
            art_key = f"defillama_chain_{name}_{today}"
            articles.append({
                "title": title,
                "link": f"https://defillama.com/chain/{name}",
                "summary": summary,
                "published": datetime.now(timezone.utc).isoformat(),
                "source": SOURCE_CHAINS,
                "article_id": _article_id(art_key),
            })
            log.info("defillama: chain signal: %s", title)
        _mark_seen(conn, key)

    conn.close()
    return articles


def _fetch_protocol_signals() -> list[dict]:
    """Emit signal articles for top protocols with large 24h TVL changes."""
    articles: list[dict] = []
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    try:
        resp = requests.get(PROTOCOLS_URL, headers=_HEADERS, timeout=FETCH_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.warning("defillama: protocols fetch failed: %s", e)
        return articles

    # Sort by TVL, take top N
    top = sorted(
        [p for p in data if isinstance(p.get("tvl"), (int, float)) and p["tvl"] > 0],
        key=lambda x: -x["tvl"],
    )[:TOP_PROTOCOLS]

    conn = _ensure_db()
    for proto in top:
        name = proto.get("name", "")
        tvl = proto.get("tvl", 0)
        change_1d = proto.get("change_1d", None)
        change_7d = proto.get("change_7d", None)
        category = proto.get("category", "DeFi")
        chain = proto.get("chain", "")

        if change_1d is None:
            continue

        key = f"defillama|protocol|{name}|{today}"
        if not _seen(conn, key) and abs(change_1d) >= PROTOCOL_TVL_CHANGE_PCT:
            direction = "surge" if change_1d > 0 else "drop"
            c7 = f"{change_7d:+.1f}%" if change_7d is not None else "N/A"
            chain_str = f" ({chain})" if chain else ""
            title = (
                f"[DeFi/TVL] {name}{chain_str} TVL {direction} {change_1d:+.1f}% 24h "
                f"→ {_fmt_tvl(tvl)} [{category}]"
            )
            summary = (
                f"{name} ({category}{chain_str}) TVL changed {change_1d:+.1f}% in 24h "
                f"to {_fmt_tvl(tvl)}. 7d: {c7}. "
                f"{'Large inflow may signal protocol growth or liquidation event.' if change_1d > 0 else 'Large outflow may signal protocol risk, exploit, or rotation.'}"
            )
            art_key = f"defillama_proto_{name}_{today}"
            articles.append({
                "title": title,
                "link": f"https://defillama.com/protocol/{name.lower().replace(' ', '-')}",
                "summary": summary,
                "published": datetime.now(timezone.utc).isoformat(),
                "source": SOURCE_PROTOCOLS,
                "article_id": _article_id(art_key),
            })
            log.info("defillama: protocol signal: %s", title)
        _mark_seen(conn, key)

    conn.close()
    return articles


def collect_defillama_tvl() -> list[dict]:
    """Collect DeFi TVL signals from DeFiLlama. Returns list of article dicts."""
    articles: list[dict] = []
    articles.extend(_fetch_global_tvl())
    articles.extend(_fetch_chain_signals())
    articles.extend(_fetch_protocol_signals())
    log.info("defillama: collected %d TVL signal articles", len(articles))
    return articles


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    results = collect_defillama_tvl()
    print(f"\n=== DeFiLlama TVL Collector Test ===")
    print(f"Signals found: {len(results)}")
    for r in results:
        print(f"\n  TITLE: {r['title']}")
        print(f"  LINK:  {r['link']}")
        print(f"  SUMMARY: {r['summary'][:120]}...")
    if not results:
        # Show current state even if no threshold-breaching signals today
        print("\n[No threshold signals today - showing current TVL snapshot]")
        try:
            resp = requests.get(CHAINS_URL, headers=_HEADERS, timeout=FETCH_TIMEOUT)
            chains = sorted(resp.json(), key=lambda x: -x.get("tvl", 0))[:8]
            for c in chains:
                d1 = c.get("change_1d")
                print(f"  {c['name']:15s} TVL={_fmt_tvl(c.get('tvl',0)):>10s}  24h={d1:+.1f}%" if d1 is not None else f"  {c['name']:15s} TVL={_fmt_tvl(c.get('tvl',0)):>10s}")
        except Exception as e:
            print(f"  [snapshot failed: {e}]")

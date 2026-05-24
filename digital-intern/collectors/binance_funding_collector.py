"""Crypto perpetual futures funding rate collector (OKX public API).

Fetches current funding rates from OKX USDT-margined perpetual swaps
(no API key required). Funding rates are the 8-hour interest exchanged
between longs and shorts — positive = longs pay shorts (crowded long),
negative = shorts pay longs (crowded short/potential squeeze).

Emits two article streams:
  1. ``crypto/funding_extreme`` — any tracked symbol with |funding| ≥ EXTREME_THRESHOLD,
     emitted at most once per 8h per symbol. High-signal leverage-stress events.
  2. ``crypto/funding_summary`` — 8h snapshot of BTC/ETH rates + top 3 extremes.

Why this matters: extreme funding precedes sharp deleveraging moves and is a
useful cross-asset risk-sentiment signal for equities (especially tech/AI names
correlated with crypto risk appetite).
"""
import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from time import sleep

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

OKX_FUNDING_URL = "https://www.okx.com/api/v5/public/funding-rate?instId={sym}"

FETCH_TIMEOUT = 12
EXTREME_THRESHOLD = 0.0005   # 0.05% per 8h (~22% annualised) — genuine stress
INTER_REQUEST_DELAY = 0.3    # seconds between per-symbol requests

TRACKED_SYMBOLS = [
    "BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP",
    "BNB-USDT-SWAP", "XRP-USDT-SWAP", "DOGE-USDT-SWAP",
    "LINK-USDT-SWAP", "AVAX-USDT-SWAP", "NEAR-USDT-SWAP",
    "AAVE-USDT-SWAP", "SUI-USDT-SWAP",
]

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _8h_bucket() -> str:
    now = _now_utc()
    bucket = (now.hour // 8) * 8
    return f"{now.strftime('%Y-%m-%d')}-{bucket:02d}h"


def _seen_id(stream: str, key: str) -> str:
    raw = f"{stream}|{key}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


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


def _is_seen(conn: sqlite3.Connection, sid: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM seen_articles WHERE id=?", (sid,)
    ).fetchone() is not None


def _mark_seen(
    conn: sqlite3.Connection, sid: str, link: str, title: str, source: str
) -> None:
    now = _now_utc().strftime("%Y-%m-%dT%H:%M:%SZ")
    conn.execute(
        "INSERT OR IGNORE INTO seen_articles(id,link,title,source,first_seen) "
        "VALUES(?,?,?,?,?)",
        (sid, link, title, source, now),
    )


def _fetch_funding_rate(sym: str) -> float | None:
    """Fetch current funding rate for one OKX perpetual swap symbol."""
    try:
        resp = requests.get(
            OKX_FUNDING_URL.format(sym=sym),
            headers={"User-Agent": _UA},
            timeout=FETCH_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if data:
            return float(data[0]["fundingRate"])
    except Exception:
        pass
    return None


def _fmt_pct(v: float) -> str:
    return f"{v * 100:+.4f}%"


def _sym_short(sym: str) -> str:
    return sym.replace("-USDT-SWAP", "")


# ── article builders ──────────────────────────────────────────────────────────

def _build_articles(conn, rates: dict[str, float]) -> list[dict]:
    bucket = _8h_bucket()
    out = []

    # Extreme alerts
    for sym, rate in rates.items():
        if abs(rate) < EXTREME_THRESHOLD:
            continue
        sid = _seen_id(f"funding_extreme/{sym}", bucket)
        if _is_seen(conn, sid):
            continue
        direction = "LONG-HEAVY" if rate > 0 else "SHORT-HEAVY"
        label = _sym_short(sym)
        link = f"https://www.okx.com/trade-swap/{sym.lower().replace('usdt', 'usdt')}"
        title = (
            f"Extreme OKX funding: {label} {_fmt_pct(rate)}/8h ({direction}) "
            f"[{bucket}]"
        )
        body = (
            f"OKX perpetual {sym} funding rate: {_fmt_pct(rate)} per 8h "
            f"(extreme threshold ±{_fmt_pct(EXTREME_THRESHOLD)}). "
            f"{'Longs paying shorts heavily — crowded leveraged longs, elevated reversal/liquidation risk.' if rate > 0 else 'Shorts paying longs heavily — crowded short, potential short squeeze risk.'} "
            f"Annualised rate: ~{rate * 3 * 365 * 100:.0f}%. 8h bucket: {bucket}."
        )
        _mark_seen(conn, sid, link, title, "crypto/funding_extreme")
        out.append({
            "title": title, "link": link, "summary": body,
            "published": _now_utc().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "crypto/funding_extreme",
        })

    # 8h summary
    sid = _seen_id("funding_summary", bucket)
    if not _is_seen(conn, sid) and rates:
        btc_r = rates.get("BTC-USDT-SWAP")
        eth_r = rates.get("ETH-USDT-SWAP")
        extremes = sorted(rates.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        extreme_str = ", ".join(f"{_sym_short(s)} {_fmt_pct(r)}" for s, r in extremes)
        link = "https://www.okx.com/markets/futures"
        title = (
            f"OKX funding snapshot {bucket}: "
            f"BTC {_fmt_pct(btc_r) if btc_r is not None else 'n/a'} | "
            f"ETH {_fmt_pct(eth_r) if eth_r is not None else 'n/a'} | "
            f"Extremes: {extreme_str}"
        )
        body = (
            f"OKX perpetual swaps 8h funding rate snapshot (UTC bucket {bucket}). "
            f"BTC: {_fmt_pct(btc_r) if btc_r is not None else 'unavailable'}/8h. "
            f"ETH: {_fmt_pct(eth_r) if eth_r is not None else 'unavailable'}/8h. "
            f"Top extreme positions: {extreme_str}. "
            f"Positive funding = net long bias (longs pay); negative = net short. "
            f"Rates above ±0.05%/8h indicate significant leverage imbalance and "
            f"precede elevated volatility or mean-reversion moves."
        )
        _mark_seen(conn, sid, link, title, "crypto/funding_summary")
        out.append({
            "title": title, "link": link, "summary": body,
            "published": _now_utc().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "crypto/funding_summary",
        })

    return out


# ── public entry point ────────────────────────────────────────────────────────

def collect_crypto_funding() -> list[dict]:
    """Collect OKX perpetual funding rate articles."""
    conn = _ensure_db()
    rates: dict[str, float] = {}
    for sym in TRACKED_SYMBOLS:
        r = _fetch_funding_rate(sym)
        if r is not None:
            rates[sym] = r
        sleep(INTER_REQUEST_DELAY)
    out = []
    try:
        out = _build_articles(conn, rates)
    finally:
        conn.commit()
        conn.close()
    return out


collect = collect_crypto_funding


if __name__ == "__main__":
    print("=== OKX Crypto Funding Rate live fetch ===\n")
    rates: dict[str, float] = {}
    for sym in TRACKED_SYMBOLS:
        r = _fetch_funding_rate(sym)
        if r is not None:
            rates[sym] = r
            flag = "  *** EXTREME" if abs(r) >= EXTREME_THRESHOLD else ""
            print(f"  {sym:22s}  {_fmt_pct(r)}/8h{flag}")
        else:
            print(f"  {sym:22s}  FETCH FAILED")
        sleep(INTER_REQUEST_DELAY)

    all_sorted = sorted(rates.items(), key=lambda kv: abs(kv[1]), reverse=True)
    print(f"\n  Top 3 most extreme:")
    for sym, r in all_sorted[:3]:
        print(f"    {_sym_short(sym):8s}  {_fmt_pct(r)}/8h")

    print("\n  Running collector (dedup check)...")
    conn = _ensure_db()
    items = _build_articles(conn, rates)
    conn.commit()
    conn.close()

    inserted = 0
    if items:
        import sys
        sys.path.insert(0, str(BASE_DIR))
        from storage.article_store import ArticleStore
        store = ArticleStore()
        inserted = store.insert_batch(items)

    print(f"\n=== Summary ===")
    print(f"Articles built      : {len(items)}")
    print(f"Inserted to DB      : {inserted}")
    for a in items:
        print(f"  + {a['title'][:110]}")

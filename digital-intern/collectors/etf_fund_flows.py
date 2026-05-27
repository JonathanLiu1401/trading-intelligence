"""ETF fund flow tracker — monitors shares-outstanding changes for major ETFs.

Shares outstanding on an ETF fluctuates daily as authorised participants
create/redeem baskets. The daily change multiplied by the NAV approximates
net fund flows — a key institutional-demand signal used in professional
equity research.

  flow_USD ≈ (shares_t  -  shares_{t-1}) × nav

Significant inflows (+$1B+) or outflows (-$1B+) into sector / broad-market
ETFs often precede or confirm directional moves in the underlying.

Data source: Yahoo Finance (yfinance), free and delay-free for EOD data.
Dedup key: ticker|date so one article per ETF per calendar day.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import zlib
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf

log = logging.getLogger("etf_fund_flows")

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"
STATE_PATH = BASE_DIR / "data" / "etf_fund_flows_state.json"

SOURCE_NAME = "ETF Fund Flows"
REQUEST_TIMEOUT = 15

# Major ETFs to monitor (broad market, sectors, bonds, commodities)
ETF_UNIVERSE = {
    # Broad US equity
    "SPY": "S&P 500",
    "QQQ": "Nasdaq-100",
    "IWM": "Russell 2000",
    "DIA": "Dow Jones",
    "VTI": "Total US Market",
    # Sectors (SPDR)
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
    # Fixed income
    "TLT": "20+ Year Treasury",
    "IEF": "7-10 Year Treasury",
    "SHY": "1-3 Year Treasury",
    "LQD": "IG Corporate Bonds",
    "HYG": "High Yield Bonds",
    "AGG": "US Aggregate Bond",
    # International
    "EEM": "Emerging Markets",
    "EFA": "Developed ex-US",
    "FXI": "China Large-Cap",
    # Commodities / alternatives
    "GLD": "Gold",
    "SLV": "Silver",
    "USO": "Oil",
    "UNG": "Natural Gas",
    # Volatility / inverse
    "UVXY": "2x Short-Term VIX",
    "SQQQ": "3x Inverse Nasdaq",
    "SPXU": "3x Inverse S&P",
    # Thematic / high-interest
    "ARKK": "ARK Innovation",
    "SOXX": "Semiconductor",
    "SMH": "Semiconductor (VanEck)",
    "BOTZ": "Robotics & AI",
    "CIBR": "Cybersecurity",
    "ROBO": "Robotics & Automation",
}

# Thresholds for article emission
FLOW_THRESHOLD_B = 0.5   # $0.5B single-ETF absolute flow to emit
NOTABLE_THRESHOLD_B = 2.0  # $2B = "significant" wording upgrade


def _load_state() -> dict:
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, ValueError, OSError):
        return {}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def _seen_id(ticker: str, date: str) -> str:
    return hashlib.sha256(f"{ticker}|{date}".encode()).hexdigest()


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


def _is_seen(conn: sqlite3.Connection, sid: str) -> bool:
    return bool(conn.execute("SELECT 1 FROM seen_articles WHERE id=?", (sid,)).fetchone())


def _mark_seen(conn: sqlite3.Connection, sid: str, ticker: str, title: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO seen_articles(id, link, title, source, first_seen) VALUES(?,?,?,?,?)",
        (sid, f"https://finance.yahoo.com/quote/{ticker}", title, SOURCE_NAME,
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def _fmt_flow(flow_b: float) -> str:
    """Format a flow value with sign and B suffix."""
    sign = "+" if flow_b >= 0 else ""
    return f"{sign}{flow_b:.2f}B"


def _build_article(ticker: str, name: str, flow_usd: float, shares_now: int,
                   nav: float, date_str: str) -> dict:
    flow_b = flow_usd / 1e9
    direction = "inflow" if flow_b > 0 else "outflow"
    magnitude = "significant" if abs(flow_b) >= NOTABLE_THRESHOLD_B else "notable"

    title = (
        f"ETF Fund Flows — {ticker} ({name}): "
        f"{magnitude.title()} {direction} of {_fmt_flow(abs(flow_b))} on {date_str}"
    )

    summary = (
        f"ETF fund flow alert for {ticker} ({name}) on {date_str}.\n"
        f"Estimated net {direction}: ${abs(flow_b):.2f}B "
        f"({'inflows: institutions buying baskets' if flow_b > 0 else 'outflows: redemptions / risk-off'}).\n"
        f"Shares outstanding now: {shares_now:,}  |  NAV: ${nav:.2f}\n"
        f"Signal interpretation: "
        + (
            f"Large {direction}s into broad/sector ETFs often precede sustained "
            f"directional momentum in the underlying index or sector. "
            + (f"The {_fmt_flow(flow_b)} move into {ticker} suggests institutional "
               f"{'accumulation' if flow_b > 0 else 'distribution'}.")
        )
    )

    link = f"https://finance.yahoo.com/quote/{ticker}"
    return {
        "title": title,
        "link": link,
        "summary": summary,
        "published": datetime.now(timezone.utc).isoformat(),
        "source": SOURCE_NAME,
    }


def collect_etf_flows() -> list[dict]:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    state = _load_state()
    articles: list[dict] = []

    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    _ensure_db(conn)

    for ticker, name in ETF_UNIVERSE.items():
        try:
            info = yf.Ticker(ticker).info
            shares_now = info.get("sharesOutstanding")
            nav = info.get("navPrice") or info.get("regularMarketPrice") or 0.0

            if not shares_now or not nav:
                log.debug("%s: missing sharesOutstanding or nav, skipping", ticker)
                continue

            prior_key = f"{ticker}_shares"
            prior_date_key = f"{ticker}_date"
            shares_prior = state.get(prior_key)
            prior_date = state.get(prior_date_key, "")

            # Update state with today's reading
            state[prior_key] = shares_now
            state[prior_date_key] = today

            # Skip if no prior data or same-day re-run
            if shares_prior is None:
                log.debug("%s: no prior data, seeding state", ticker)
                continue
            if prior_date == today:
                # Same-day re-run: flow already calculated, check dedup
                pass
            if shares_prior == shares_now:
                continue

            flow_usd = (shares_now - shares_prior) * nav
            flow_b = abs(flow_usd) / 1e9

            if flow_b < FLOW_THRESHOLD_B:
                continue

            sid = _seen_id(ticker, today)
            if _is_seen(conn, sid):
                continue

            article = _build_article(ticker, name, flow_usd, shares_now, nav, today)
            _mark_seen(conn, sid, ticker, article["title"])
            articles.append(article)
            log.info("%s %s flow: %s", ticker, name, _fmt_flow(flow_usd / 1e9))

        except Exception as exc:
            log.warning("%s: fetch error: %s", ticker, exc)

    conn.close()
    _save_state(state)
    log.info("etf_fund_flows: %d flow articles emitted", len(articles))
    return articles


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    print("Fetching ETF shares-outstanding data for fund flow calculation...")
    results = collect_etf_flows()

    if results:
        print(f"\nEmitted {len(results)} fund flow articles:\n")
        for a in results:
            print(f"  TITLE: {a['title']}")
            print(f"  LINK:  {a['link']}")
            print(f"  BODY:  {a['summary'][:200]}...")
            print()
    else:
        # Show raw data even if no threshold-crossing flows today
        print("\nNo threshold-crossing flows detected. Raw ETF data sample:\n")
        state = _load_state()
        for ticker, name in list(ETF_UNIVERSE.items())[:8]:
            try:
                info = yf.Ticker(ticker).info
                shares = info.get("sharesOutstanding", "N/A")
                nav = info.get("navPrice") or info.get("regularMarketPrice", "N/A")
                assets = info.get("totalAssets", 0)
                print(f"  {ticker:6s} ({name:30s}): shares={shares:>15,}  nav=${nav}  AUM=${assets/1e9:.1f}B")
            except Exception as e:
                print(f"  {ticker}: error: {e}")

"""Sector ETF momentum and rotation collector.

Tracks relative strength of the 11 SPDR sector ETFs vs SPY and emits
articles when notable rotation is detected:

  - Leadership shift: top-2 sectors change vs prior day
  - Extreme spread: top sector outperforms bottom by >= SPREAD_THRESHOLD %
  - Single-sector spike: any ETF moves >= SPIKE_THRESHOLD % in a day

Dedup key: (date, regime_key) — at most one article per rotation regime per day.
Mirrors dxy_collector / vix_term_structure patterns (direct articles.db insert,
state table for daily tripwire).
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import zlib
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "articles.db"
SOURCE = "sector_etf_momentum"

# 11 SPDR Select Sector ETFs + SPY as benchmark
SECTORS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLC": "Communication Svcs",
    "XLY": "Consumer Discret",
    "XLP": "Consumer Staples",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
}
BENCHMARK = "SPY"

# Thresholds for emit decisions
SPREAD_THRESHOLD_PCT = 2.0   # top minus bottom sector 1d return
SPIKE_THRESHOLD_PCT = 2.5    # single sector daily move
LEADERSHIP_LOOKBACK = "5d"   # yfinance period for daily returns

log = logging.getLogger("sector_etf_momentum")

_STATE_TABLE = """
CREATE TABLE IF NOT EXISTS sector_etf_state (
    key   TEXT PRIMARY KEY,
    value TEXT
)
"""


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(_STATE_TABLE)
    conn.commit()


def _get_state(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute(
        "SELECT value FROM sector_etf_state WHERE key=?", (key,)
    ).fetchone()
    return row[0] if row else None


def _set_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO sector_etf_state (key, value) VALUES (?,?)",
        (key, value),
    )
    conn.commit()


def _article_id(date_str: str, regime_key: str) -> str:
    raw = f"{SOURCE}|{date_str}|{regime_key}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _fetch_returns() -> dict[str, float] | None:
    """Fetch 1-day returns for all sectors + SPY. Returns None if SPY fails."""
    returns: dict[str, float] = {}
    tickers = list(SECTORS.keys()) + [BENCHMARK]
    for sym in tickers:
        try:
            h = yf.Ticker(sym).history(period=LEADERSHIP_LOOKBACK)
            closes = h["Close"].dropna().tolist()
            if len(closes) < 2:
                log.warning("sector_etf_momentum: insufficient data for %s", sym)
                continue
            pct = (closes[-1] - closes[-2]) / closes[-2] * 100.0
            returns[sym] = round(pct, 3)
        except Exception as exc:
            log.warning("sector_etf_momentum: failed to fetch %s: %s", sym, exc)

    if BENCHMARK not in returns:
        log.warning("sector_etf_momentum: SPY fetch failed, aborting")
        return None
    if len(returns) < 6:
        log.warning("sector_etf_momentum: too few sectors fetched (%d)", len(returns) - 1)
        return None
    return returns


def collect(conn: sqlite3.Connection | None = None) -> list[dict]:
    """Fetch sector ETF returns and emit an article if rotation signal fires."""
    close_conn = conn is None
    if conn is None:
        conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)

    try:
        _ensure_schema(conn)
        returns = _fetch_returns()
        if not returns:
            return []

        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")
        spy_ret = returns.pop(BENCHMARK, 0.0)

        # Relative returns vs SPY
        rel = {sym: round(ret - spy_ret, 3) for sym, ret in returns.items()}
        sorted_sectors = sorted(rel.items(), key=lambda x: x[1], reverse=True)
        top2 = [s for s, _ in sorted_sectors[:2]]
        bottom2 = [s for s, _ in sorted_sectors[-2:]]
        top_ret = sorted_sectors[0][1]
        bot_ret = sorted_sectors[-1][1]
        spread = round(top_ret - bot_ret, 3)

        # Detect spikes in absolute terms
        spike_sectors = [
            sym for sym, ret in returns.items()
            if abs(ret) >= SPIKE_THRESHOLD_PCT
        ]

        # Regime key for dedup: top-2 leaders + bottom-2 laggards
        regime_key = f"{'_'.join(top2)}_vs_{'_'.join(bottom2)}"

        # Leadership shift: did top-2 change since yesterday?
        prev_top2_str = _get_state(conn, f"top2_{date_str}")
        if prev_top2_str is None:
            # Check yesterday's state key
            prev_top2_str = _get_state(conn, "top2_prev")

        current_top2_str = "_".join(sorted(top2))
        leadership_shifted = prev_top2_str and prev_top2_str != current_top2_str
        _set_state(conn, "top2_prev", current_top2_str)
        _set_state(conn, f"top2_{date_str}", current_top2_str)

        should_emit = (
            spread >= SPREAD_THRESHOLD_PCT
            or bool(spike_sectors)
            or leadership_shifted
        )
        if not should_emit:
            log.debug("sector_etf_momentum: no signal today (spread=%.2f%%)", spread)
            return []

        article_id = _article_id(date_str, regime_key)
        already_exists = conn.execute(
            "SELECT 1 FROM articles WHERE id=? LIMIT 1", (article_id,)
        ).fetchone()
        if already_exists:
            log.debug("sector_etf_momentum: already emitted for this regime today")
            return []

        # Build title
        top_sym, top_rel = sorted_sectors[0]
        bot_sym, bot_rel = sorted_sectors[-1]
        top_name = SECTORS.get(top_sym, top_sym)
        bot_name = SECTORS.get(bot_sym, bot_sym)

        if spike_sectors:
            spike_parts = ", ".join(
                f"{s} {returns[s]:+.2f}%" for s in spike_sectors
            )
            title = f"Sector spike: {spike_parts} | SPY {spy_ret:+.2f}%"
        elif leadership_shifted:
            title = (
                f"Sector rotation: {top_name} ({top_sym} {top_rel:+.2f}% vs SPY) "
                f"takes lead from {prev_top2_str or 'prior leaders'}"
            )
        else:
            title = (
                f"Sector spread {spread:.1f}%: "
                f"{top_sym} leads ({top_rel:+.2f}%) | {bot_sym} lags ({bot_rel:+.2f}%)"
            )

        # Build body
        lines = [
            f"SPY: {spy_ret:+.2f}% (benchmark)",
            "",
            "Sector relative returns vs SPY (today):",
        ]
        for sym, rel_ret in sorted_sectors:
            name = SECTORS.get(sym, sym)
            bar = "▲" if rel_ret >= 0 else "▼"
            lines.append(f"  {bar} {sym:5s} {name:<22s} {rel_ret:+.2f}%")

        lines += [
            "",
            f"Spread (top - bottom): {spread:.2f}%",
        ]
        if spike_sectors:
            lines.append(f"Spikes (abs ≥{SPIKE_THRESHOLD_PCT}%): {', '.join(spike_sectors)}")
        if leadership_shifted:
            lines.append(f"Leadership shift: {prev_top2_str} → {current_top2_str}")

        full_text = "\n".join(lines)
        ts = now.strftime("%Y-%m-%d %H:%M:%S")
        url = f"internal://sector_etf_momentum/{date_str}/{regime_key}"

        # Urgency: spike or large spread or rotation gets a higher score
        kw = 3.0
        if spike_sectors:
            kw += 2.0
        if spread >= SPREAD_THRESHOLD_PCT:
            kw += min(spread / 2.0, 3.0)
        if leadership_shifted:
            kw += 1.5
        kw = round(min(kw, 9.0), 1)
        urgency = 1 if kw >= 6.0 else 0

        compressed = zlib.compress(full_text.encode("utf-8"))
        conn.execute(
            """INSERT OR IGNORE INTO articles
               (id, url, title, source, published, kw_score, urgency, full_text, first_seen)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (article_id, url, title, SOURCE, ts, kw, urgency, compressed, ts),
        )
        conn.commit()
        log.info("sector_etf_momentum: emitted — %s", title)

        row = {
            "id": article_id,
            "url": url,
            "title": title,
            "full_text": full_text,
            "source": SOURCE,
            "published": ts,
            "first_seen": ts,
            "kw_score": kw,
            "urgency": urgency,
        }
        return [row]

    finally:
        if close_conn:
            conn.close()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results = collect()
    if results:
        for r in results:
            print(f"\nTitle:  {r['title']}")
            print(f"Score:  {r['kw_score']}  urgency={r['urgency']}")
            print(r["full_text"])
    else:
        # Still show data even if no article emitted (already exists or no signal)
        log.info("No article emitted — fetching raw data for inspection...")
        ret = _fetch_returns()
        if ret:
            spy = ret.pop("SPY", 0.0)
            print(f"SPY: {spy:+.2f}%")
            rel = sorted(
                {s: round(r - spy, 3) for s, r in ret.items()}.items(),
                key=lambda x: x[1], reverse=True,
            )
            for sym, r in rel:
                print(f"  {sym}: {r:+.3f}% vs SPY  (abs: {ret[sym]:+.3f}%)")
        else:
            print("Failed to fetch data", file=sys.stderr)
            sys.exit(1)

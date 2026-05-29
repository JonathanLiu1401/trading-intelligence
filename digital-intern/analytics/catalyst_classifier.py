"""Catalyst classifier: WHY is each trending ticker spiking?

For each ticker in the top-N trend_velocity movers, fetches article titles
from the current 2h window and classifies the dominant catalyst type using
keyword matching.  Writes an attributed spike record per ticker to
/home/zeph/logs/catalyst_classifier.json.

Catalyst types (ordered by match priority):
  EARNINGS      — earnings beat/miss, EPS, revenue, guidance, results
  ANALYST       — upgrade/downgrade, price target, rating, overweight/underweight
  SHORT_SQUEEZE — short squeeze, short interest, short seller report
  M&A           — merger, acquisition, buyout, takeover, bid, deal closed
  PRODUCT       — launch, new product, partnership, contract, collaboration
  REGULATORY    — FDA, SEC, DOJ, lawsuit, fine, settlement, ban, sanction
  MACRO         — Fed, rate, inflation, tariff, CPI, PPI, GDP, jobs, recession
  EARNINGS_PRE  — pre-earnings, earnings calendar, options flow ahead of earnings
  TECHNICAL     — breakout, all-time high, 52-week, support, resistance, squeeze
  UNKNOWN       — none of the above matched

Output: /home/zeph/logs/catalyst_classifier.json
Standalone: python3 -m analytics.catalyst_classifier
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import NamedTuple

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

try:
    from storage.article_store import _LIVE_ONLY_CLAUSE, _get_db_path
    DB_PATH = _get_db_path()
except Exception:
    _LIVE_ONLY_CLAUSE = "source NOT LIKE 'backtest_run_%'"
    DB_PATH = BASE / "data" / "articles.db"

from analytics.trend_velocity import _parse_ts, extract_tickers, FETCH_LIMIT, TOP_N, WINDOW_HOURS

OUT_PATH = Path("/home/zeph/logs/catalyst_classifier.json")

# Catalyst rules: (type, pattern) — first match wins per title, majority wins per ticker
_CATALYST_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("EARNINGS", re.compile(
        r"\b(earnings?|beat|miss|eps|revenue|guid(ance)?|results?|profit|"
        r"quarter|q[1-4]\s*\d{4}|report(ed|s|ing)?|blowout|surpass|exceed)\b",
        re.IGNORECASE,
    )),
    ("ANALYST", re.compile(
        r"\b(upgrad(e|ed|es|ing)|downgrad(e|ed|es|ing)|price\s*target|"
        r"pt\s*\$?\d|rating|overweight|underweight|outperform|"
        r"neutral|initiat(e|ed|ion)|rais(e|ed|es|ing)\s+target|"
        r"cut\s+target|analyst|coverage)\b",
        re.IGNORECASE,
    )),
    ("SHORT_SQUEEZE", re.compile(
        r"\b(short\s*squeeze|short\s*interest|short\s*seller|"
        r"short\s*report|heavily\s*shorted|meme\s*stock|gamma\s*squeeze)\b",
        re.IGNORECASE,
    )),
    ("M&A", re.compile(
        r"\b(merger|acqui(sition|red|res|ring)|buyout|takeover|"
        r"bid\s+for|deal\s+(closed|complete|sign)|"
        r"acqui(re|red)\s+by|going\s+private|lbo)\b",
        re.IGNORECASE,
    )),
    ("PRODUCT", re.compile(
        r"\b(launch(ed|es|ing)?|new\s+product|partnership|contract|"
        r"collaborat(e|ion)|integrat(e|ion)|annonc(e|ed|ing)|"
        r"introduc(e|ed|ing)|debuted?|release(d|s)?)\b",
        re.IGNORECASE,
    )),
    ("REGULATORY", re.compile(
        r"\b(fda|sec\s+|doj|lawsuit|fine(d|s)?|settlement|ban(ned)?|"
        r"sanction|investigation|subpoena|injunction|approv(al|ed)|"
        r"reject(ed|ion)|recall|warning\s+letter)\b",
        re.IGNORECASE,
    )),
    ("MACRO", re.compile(
        r"\b(fed(eral\s+reserve)?|rate\s+(cut|hike|hold)|inflation|"
        r"tariff|cpi|ppi|gdp|jobs?\s+report|unemployment|recession|"
        r"fomc|powell|yellen|treasury|yield\s+curve)\b",
        re.IGNORECASE,
    )),
    ("EARNINGS_PRE", re.compile(
        r"\b(pre.?earnings|earnings\s+preview|ahead\s+of\s+earnings|"
        r"options\s+flow|unusual\s+options|call\s+sweep|put\s+sweep|"
        r"earnings\s+whisper|consensus\s+estimate)\b",
        re.IGNORECASE,
    )),
    ("GOVERNMENT", re.compile(
        r"\b(government|pentagon|department\s+of\s+defense|dod|"
        r"trump|white\s+house|congress|senate|executive\s+order|"
        r"equity\s+stake|nationali[sz]|drone\s+maker|defense\s+contract)\b",
        re.IGNORECASE,
    )),
    ("TECHNICAL", re.compile(
        r"\b(breakout|all.?time\s+high|52.?week\s+(highs?|lows?)|"
        r"hitting\s+(high|low)|support|resistance|oversold|overbought|"
        r"rsi|macd|golden\s+cross|death\s+cross|momentum|rally)\b",
        re.IGNORECASE,
    )),
]


class SpikeRecord(NamedTuple):
    ticker: str
    now: int
    prev: int
    delta: int
    ratio: float
    catalyst: str
    confidence: float
    sample_titles: list[str]


def _classify_title(title: str) -> str:
    for cat, pat in _CATALYST_RULES:
        if pat.search(title):
            return cat
    return "UNKNOWN"


def classify_spike(ticker: str, titles: list[str]) -> tuple[str, float]:
    """Return (catalyst_type, confidence) for a list of titles."""
    if not titles:
        return "UNKNOWN", 0.0
    counts: Counter[str] = Counter(_classify_title(t) for t in titles)
    top_cat, top_n = counts.most_common(1)[0]
    confidence = round(top_n / len(titles), 2)
    return top_cat, confidence


def main() -> int:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=20)
    conn.execute("PRAGMA query_only=ON")

    now = datetime.now(timezone.utc)
    cur_cut = now - timedelta(hours=WINDOW_HOURS)
    prev_cut = now - timedelta(hours=WINDOW_HOURS * 2)

    cur = conn.execute(
        "SELECT first_seen, title FROM articles "
        f"WHERE {_LIVE_ONLY_CLAUSE} "
        "ORDER BY first_seen DESC LIMIT ?",
        (FETCH_LIMIT,),
    )
    rows = cur.fetchall()
    if not rows:
        print("catalyst_classifier: no rows", file=sys.stderr)
        return 1

    cur_mentions: dict[str, list[str]] = {}
    prev_counts: Counter[str] = Counter()

    for fs, title in rows:
        ts = _parse_ts(fs)
        if ts is None or not title:
            continue
        tickers = extract_tickers(title)
        if not tickers:
            continue
        if ts >= cur_cut:
            for tk in tickers:
                cur_mentions.setdefault(tk, []).append(title)
        elif ts >= prev_cut:
            prev_counts.update(tickers)

    # Build movers list (same logic as trend_velocity)
    movers: list[tuple[str, int, int, int, float]] = []
    for tk, titles in cur_mentions.items():
        c = len(titles)
        p = prev_counts.get(tk, 0)
        movers.append((tk, c, p, c - p, (c + 1) / (p + 1)))

    movers.sort(key=lambda r: (r[3], r[4]), reverse=True)
    top = movers[:TOP_N]

    results: list[dict] = []
    for tk, c, p, delta, ratio in top:
        titles = cur_mentions[tk]
        catalyst, confidence = classify_spike(tk, titles)
        sample = titles[:3]
        results.append({
            "ticker": tk,
            "now": c,
            "prev": p,
            "delta": delta,
            "ratio": round(ratio, 2),
            "catalyst": catalyst,
            "confidence": confidence,
            "sample_titles": sample,
        })
        print(f"  {tk}: delta=+{delta} ratio={ratio:.2f}x  [{catalyst} {confidence:.0%}]")
        for t in sample[:1]:
            print(f"    \"{t[:100]}\"")

    payload = {
        "generated_at": now.isoformat(),
        "window_hours": WINDOW_HOURS,
        "scanned": len(rows),
        "spikes": results,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"catalyst_classifier: scanned={len(rows)} top_spikes={len(results)} → {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Weekend Position Brief: per-held-ticker news risk summary for closed-market periods.

For each open paper-trader position, scans the last 48 h of articles
mentioning that ticker, classifies catalysts, and computes a risk score
based on ml_score distribution (negative slant = low ml_score cluster).

Distinct from:
  * portfolio_overlap_scorer  — ranks articles by overlap count, no risk scoring
  * weekend_catalyst_brief    — not held-position-aware, broad market focus

Output:
  JSON  /home/zeph/logs/weekend_position_brief.json
  Text  /home/zeph/logs/weekend_position_brief.txt

Standalone:  python3 -m analytics.weekend_position_brief
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from storage.article_store import _LIVE_ONLY_CLAUSE

ARTICLES_DB = BASE / "data" / "articles.db"
PAPER_TRADER_DB = Path("/media/zeph/projects/paper-trader/data/paper_trader.db")
JSON_OUT = Path("/home/zeph/logs/weekend_position_brief.json")
TEXT_OUT = Path("/home/zeph/logs/weekend_position_brief.txt")

SCAN_LIMIT = 5000
WINDOW_HOURS = 48
HIGH_SCORE = 6.0   # ml_score >= this → bullish signal
LOW_SCORE  = 3.0   # ml_score <= this → bearish signal

_CATALYST_RULES: list[tuple[str, re.Pattern]] = [
    ("EARNINGS",   re.compile(r"\b(earnings?|beat|miss|eps|revenue|guidance|results?|profit|quarter)\b", re.I)),
    ("M&A",        re.compile(r"\b(merger|acquisition|acquir|buyout|takeover|bid|deal)\b", re.I)),
    ("ANALYST",    re.compile(r"\b(upgrad|downgrad|price\s*target|rating|outperform|initiat)\b", re.I)),
    ("REGULATORY", re.compile(r"\b(fda|sec\b|doj|ftc|lawsuit|fine|settlement|ban|sanction|approval|investigation)\b", re.I)),
    ("MACRO",      re.compile(r"\b(fed|rate\s+cut|rate\s+hike|inflation|tariff|cpi|ppi|gdp|recession|powell|fomc)\b", re.I)),
    ("PRODUCT",    re.compile(r"\b(launch|new\s+product|partnership|contract|announce|unveil|release)\b", re.I)),
    ("TECHNICAL",  re.compile(r"\b(breakout|all.time\s+high|52.week|short\s+squeeze|support|resistance)\b", re.I)),
]

TICKER_RE = re.compile(r"\b([A-Z]{2,5})\b")
_STOP = frozenset({
    "A", "AN", "AS", "AT", "BE", "BY", "DO", "FOR", "GO", "HE", "IF", "IN",
    "IS", "IT", "ME", "MY", "NO", "NOT", "OF", "ON", "OR", "SO", "THE", "TO",
    "UP", "US", "WE", "WHO", "WHY", "AND", "BUT", "CAN", "DID", "GET", "HOW",
    "ALL", "ANY", "ARE", "HAD", "HER", "HIM", "HIS", "ITS", "OUR", "NEW",
    "CEO", "CFO", "CTO", "ETF", "IPO", "SEC", "AI", "ML", "API", "USD", "EUR",
    "USA", "UK", "EU", "FED", "GDP", "CPI", "PPI", "Q1", "Q2", "Q3", "Q4",
    "NOW", "ONE", "TWO", "WAY", "DAY", "WEEK", "MONTH", "YEAR", "NEWS",
    "INC", "LLC", "LTD", "CORP", "PLC", "MSN", "CNN", "BBC", "WSJ", "NYT",
})


def _classify_catalyst(title: str) -> str:
    for cat, pat in _CATALYST_RULES:
        if pat.search(title or ""):
            return cat
    return "OTHER"


def _mentions_ticker(title: str, ticker: str) -> bool:
    return bool(re.search(rf"\b\$?{re.escape(ticker)}\b", title or "", re.I))


def _held_tickers() -> list[str]:
    candidates = [
        PAPER_TRADER_DB,
        Path("/home/zeph/trading-intelligence/paper-trader/data/paper_trader.db"),
        BASE / "data" / "paper_trader.db",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
            rows = conn.execute(
                "SELECT UPPER(ticker) FROM positions WHERE closed_at IS NULL"
            ).fetchall()
            conn.close()
            tickers = sorted({r[0] for r in rows if r[0]})
            if tickers:
                return tickers
        except Exception:
            continue
    return []


def _risk_score(high: int, low: int, total: int) -> float:
    """Bearish slant score in [-1, 1]. Positive = net bearish risk."""
    if total == 0:
        return 0.0
    return round((low - high) / total, 3)


def compute() -> dict:
    held = _held_tickers()
    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(hours=WINDOW_HOURS)).isoformat()

    conn = sqlite3.connect(str(ARTICLES_DB))
    conn.execute("PRAGMA busy_timeout=5000")
    rows = conn.execute(
        f"""
        SELECT title, source, ml_score, ai_score, urgency, first_seen
          FROM articles
         WHERE first_seen >= ?
           AND {_LIVE_ONLY_CLAUSE}
         ORDER BY first_seen DESC
         LIMIT ?
        """,
        (cutoff, SCAN_LIMIT),
    ).fetchall()
    conn.close()

    scanned = len(rows)
    per_ticker: dict[str, dict] = {}

    for ticker in held:
        matching = [r for r in rows if _mentions_ticker(r[0], ticker)]
        if not matching:
            per_ticker[ticker] = {
                "ticker": ticker,
                "article_count": 0,
                "avg_ml_score": None,
                "top_article": None,
                "catalyst_counts": {},
                "risk_score": 0.0,
                "note": "no articles in window",
            }
            continue

        ml_vals = [float(r[2]) for r in matching if r[2] is not None]
        avg_ml = round(sum(ml_vals) / len(ml_vals), 3) if ml_vals else None

        high_count = sum(1 for v in ml_vals if v >= HIGH_SCORE)
        low_count  = sum(1 for v in ml_vals if v <= LOW_SCORE)
        risk = _risk_score(high_count, low_count, len(ml_vals)) if ml_vals else 0.0

        catalysts = Counter(_classify_catalyst(r[0]) for r in matching)

        best = max(matching, key=lambda r: float(r[2]) if r[2] is not None else 0.0)

        per_ticker[ticker] = {
            "ticker": ticker,
            "article_count": len(matching),
            "avg_ml_score": avg_ml,
            "high_score_n": high_count,
            "low_score_n": low_count,
            "risk_score": risk,
            "top_article": {
                "title": (best[0] or "")[:120],
                "source": best[1],
                "ml_score": best[2],
                "first_seen": best[5],
            },
            "catalyst_counts": dict(catalysts.most_common()),
        }

    payload = {
        "generated_at": now.isoformat(),
        "window_hours": WINDOW_HOURS,
        "scanned": scanned,
        "held_tickers": held,
        "positions": list(per_ticker.values()),
    }
    return payload


def _render_text(p: dict) -> str:
    lines = [
        f"=== Weekend Position Brief [{p['generated_at'][:16]} UTC] ===",
        f"Window: last {p['window_hours']}h  |  articles scanned: {p['scanned']}",
        f"Held: {', '.join(p['held_tickers']) or 'none'}",
        "",
    ]
    for pos in p["positions"]:
        risk = pos["risk_score"]
        risk_label = "BEAR" if risk > 0.2 else "BULL" if risk < -0.2 else "NEUTRAL"
        lines.append(f"── {pos['ticker']} ── {risk_label} (risk={risk:+.2f})")
        lines.append(f"   Articles: {pos['article_count']}  |  avg_ml: {pos.get('avg_ml_score') or 'n/a'}  |  high={pos.get('high_score_n',0)} low={pos.get('low_score_n',0)}")
        if pos.get("catalyst_counts"):
            cats = "  ".join(f"{k}:{v}" for k, v in pos["catalyst_counts"].items())
            lines.append(f"   Catalysts: {cats}")
        if pos.get("top_article"):
            a = pos["top_article"]
            lines.append(f"   TOP: [{a.get('ml_score','?')}] {a['title']}")
            lines.append(f"        <{a['source']}> {a.get('first_seen','?')[:16]}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    payload = compute()
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(payload, indent=2))

    text = _render_text(payload)
    TEXT_OUT.write_text(text)

    print(text)
    print(f"\njson → {JSON_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

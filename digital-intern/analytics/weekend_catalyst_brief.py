"""Weekend Catalyst Brief — Monday gap-open readiness scanner.

Scans articles published from Friday 4 PM ET through the current moment and
surfaces the highest-signal catalyst events that are most likely to move
markets at Monday open.  Distinct from ``overnight_gap_scanner`` (which looks
at a single overnight window) and ``premarket_brief`` (which reads pre-built
artifacts): this script operates on the *full* Friday-close → Sunday window
so weekend analysts get a single consolidated view of everything that landed
over the two-day quiet period.

When run on a non-weekend day the window shrinks to the last 16 h (standard
overnight); the script still runs cleanly — it's just less interesting.

Design:
  * Bounded idx_first_seen scan (SCAN_LIMIT rows), read-only, USB-safe.
  * Pure builder (``build_weekend_brief``) for testability; ``main()`` owns
    the DB read + JSON write.
  * ``_LIVE_ONLY_CLAUSE`` discipline: synthetic backtest/opus rows excluded.
  * Catalyst typing reuses the same keyword rules as ``catalyst_classifier``
    so the two surfaces stay behaviourally aligned.
  * No DB writes, no ai_score / ml_score / urgency mutation.

Output: /home/zeph/logs/weekend_catalyst_brief.json
Standalone: python3 -m analytics.weekend_catalyst_brief
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import defaultdict, Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

try:
    from storage.article_store import _LIVE_ONLY_CLAUSE, _get_db_path
    DB_PATH = _get_db_path()
except Exception:
    _LIVE_ONLY_CLAUSE = "source NOT LIKE 'backtest_run_%'"
    DB_PATH = BASE / "data" / "articles.db"

OUT_PATH = Path("/home/zeph/logs/weekend_catalyst_brief.json")
SCAN_LIMIT = 500  # USB-backed DB times out on large scans
TOP_N = 15
ET = ZoneInfo("America/New_York")

# Minimum signal thresholds to include a row
MIN_ML_SCORE = 0.25
MIN_AI_SCORE = 0.30

# Catalyst keyword rules (first match wins per title)
_CATALYST_RULES: list[tuple[str, re.Pattern]] = [
    ("EARNINGS", re.compile(
        r"\b(earnings?|beat|miss|eps|revenue|guid(ance)?|results?|profit|"
        r"quarter|q[1-4]\s*\d{4}|blowout|surpass|exceed|reported)\b",
        re.IGNORECASE,
    )),
    ("M&A", re.compile(
        r"\b(merger|acquisition|acquir|buyout|takeover|bid|deal|"
        r"offer\s+to\s+buy|going\s+private)\b",
        re.IGNORECASE,
    )),
    ("ANALYST", re.compile(
        r"\b(upgrad|downgrad|price\s*target|rating|overweight|underweight|"
        r"outperform|neutral|initiat|raises\s+target|cuts?\s+target)\b",
        re.IGNORECASE,
    )),
    ("REGULATORY", re.compile(
        r"\b(fda|sec\b|doj|ftc|lawsuit|fine|settlement|ban|sanction|"
        r"approval|approved|rejected|clearance|investigation)\b",
        re.IGNORECASE,
    )),
    ("MACRO", re.compile(
        r"\b(fed|rate\s+cut|rate\s+hike|inflation|tariff|cpi|ppi|gdp|"
        r"recession|jobs\s+report|employment|powell|fomc)\b",
        re.IGNORECASE,
    )),
    ("PRODUCT", re.compile(
        r"\b(launch|new\s+product|partnership|contract|collaboration|"
        r"announce|unveil|release|debut)\b",
        re.IGNORECASE,
    )),
    ("TECHNICAL", re.compile(
        r"\b(breakout|all[-\s]time\s+high|52[-\s]week|support|resistance|"
        r"short\s+squeeze|gamma\s+squeeze)\b",
        re.IGNORECASE,
    )),
]

TICKER_RE = re.compile(r"\b([A-Z]{2,5})\b")

# Common English words that look like tickers — exclude
_STOP_TICKERS = frozenset({
    "A", "AN", "AS", "AT", "BE", "BY", "DO", "FOR", "GO", "HAS", "HE",
    "IF", "IN", "IS", "IT", "ITS", "ME", "MY", "NO", "NOT", "OF", "ON",
    "OR", "OUR", "OUT", "OWN", "RE", "SO", "THE", "TO", "UP", "US",
    "WAS", "WE", "WHO", "WHY", "WILL", "WITH", "YOU", "AND", "BUT",
    "CAN", "DID", "GET", "HIM", "HOW", "LET", "MAY", "NOW", "OFF",
    "OLD", "ONE", "SAY", "SHE", "TOO", "TWO", "USE", "WAY", "NEW",
    "ALL", "ANY", "ARE", "HAD", "HER", "HIM", "HIS", "ITS", "OUR",
    "SAID", "SAYS", "THAN", "THAT", "THEM", "THEN", "THEY", "THIS",
    "FROM", "HAVE", "BEEN", "WERE", "WHAT", "WHEN", "ALSO", "EACH",
    "BEEN", "MORE", "MUCH", "OVER", "SAME", "SOME", "SUCH", "WELL",
    "YEAR", "JUST", "MOST", "BOTH", "INTO", "HERE", "ONLY", "VERY",
    "THEIR", "THERE", "ABOUT", "AFTER", "FIRST", "COULD", "WOULD",
    "SHOULD", "STILL", "BEING", "WHILE", "WHICH", "DURING", "BEFORE",
    "AFTER", "SINCE", "EVERY", "THESE", "THOSE", "OTHER", "UNDER",
    "MAKES", "MADE", "MAKE", "TAKE", "CAME", "COME", "BACK", "LAST",
    "NEXT", "LIKE", "LOOK", "LONG", "MANY", "NEED", "PLAN", "SAID",
    "SHOW", "TELL", "THINK", "WANT", "WEEK", "WORK", "YEAR", "MOVE",
    "MUST", "NEAR", "ONCE", "OPEN", "PART", "PAST", "PLAY", "REAL",
    "RISK", "ROLE", "RULE", "SALE", "SENT", "SIGN", "SITE", "SIZE",
    "STAY", "STEP", "STOP", "SURE", "TEAM", "TERM", "TIME", "TOOK",
    "TURN", "TYPE", "UNIT", "USED", "VIEW", "VOTE", "WAIT", "WALK",
    "WARN", "WIDE", "WORD", "GREW", "HELD", "HIGH", "HOLD", "HOPE",
    "FELL", "FEEL", "FELT", "FIND", "FIRM", "FIVE", "FOUR", "FREE",
    "FULL", "FUND", "GAIN", "GIVE", "GOES", "GOOD", "GREW",
    "PRESS", "TRADE", "SHARE", "STOCK", "PRICE", "GROWTH", "MARKET",
    "SALES", "COSTS", "BOARD", "CHIEF", "COURT", "EARLY", "ENTER",
    "EQUAL", "ERROR", "EVEN", "EVENT", "EXACT", "EXIST",
})


def _classify_catalyst(title: str) -> str:
    for cat, pat in _CATALYST_RULES:
        if pat.search(title):
            return cat
    return "OTHER"


def _extract_tickers(title: str) -> list[str]:
    return [t for t in TICKER_RE.findall(title) if t not in _STOP_TICKERS]


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("T", " ").replace("Z", "+00:00"))
    except Exception:
        return None


def _weekend_window_start(now: datetime) -> datetime:
    """Return the start of the weekend analysis window (Friday 16:00 ET)."""
    now_et = now.astimezone(ET)
    dow = now_et.weekday()  # Mon=0 … Sun=6
    if dow == 5:  # Saturday → window started yesterday (Fri)
        days_back = 1
    elif dow == 6:  # Sunday → window started two days ago (Fri)
        days_back = 2
    else:
        # Weekday: use standard overnight window (16h back)
        cutoff = now - timedelta(hours=16)
        return cutoff
    fri_et = now_et.replace(hour=16, minute=0, second=0, microsecond=0) - timedelta(days=days_back)
    return fri_et.astimezone(timezone.utc)


def build_weekend_brief(
    rows: list[tuple],
    window_start: datetime,
    now: datetime,
) -> dict:
    """Pure builder — no I/O.

    ``rows`` are (title, source, first_seen, ml_score, ai_score, urgency) tuples
    already filtered to the analysis window by the caller.
    """
    # Per-ticker accumulator
    ticker_data: dict[str, dict] = defaultdict(lambda: {
        "articles": 0,
        "max_ml": 0.0,
        "max_ai": 0.0,
        "max_urgency": 0,
        "catalysts": Counter(),
        "sample_titles": [],
        "sources": set(),
    })

    total_scanned = 0
    for title, source, first_seen_raw, ml_score, ai_score, urgency in rows:
        total_scanned += 1
        ml = ml_score or 0.0
        ai = ai_score or 0.0
        urg = urgency or 0

        if ml < MIN_ML_SCORE and ai < MIN_AI_SCORE:
            continue

        tickers = _extract_tickers(title or "")
        if not tickers:
            continue

        cat = _classify_catalyst(title or "")

        for ticker in tickers:
            d = ticker_data[ticker]
            d["articles"] += 1
            if ml > d["max_ml"]:
                d["max_ml"] = ml
            if ai > d["max_ai"]:
                d["max_ai"] = ai
            if urg > d["max_urgency"]:
                d["max_urgency"] = urg
            d["catalysts"][cat] += 1
            d["sources"].add(source or "unknown")
            if len(d["sample_titles"]) < 3:
                d["sample_titles"].append((title or "")[:120])

    # Score each ticker: weight urgency heavily since it's human/LLM vetted
    results = []
    for ticker, d in ticker_data.items():
        composite = (
            0.35 * d["max_ml"]
            + 0.25 * d["max_ai"]
            + 0.20 * min(d["articles"] / 5.0, 1.0)
            + 0.20 * (1.0 if d["max_urgency"] >= 2 else 0.5 if d["max_urgency"] == 1 else 0.0)
        )
        top_catalyst = d["catalysts"].most_common(1)[0][0] if d["catalysts"] else "OTHER"
        results.append({
            "ticker": ticker,
            "composite_score": round(composite, 4),
            "articles": d["articles"],
            "max_ml_score": round(d["max_ml"], 4),
            "max_ai_score": round(d["max_ai"], 4),
            "max_urgency": d["max_urgency"],
            "top_catalyst": top_catalyst,
            "source_count": len(d["sources"]),
            "sample_titles": d["sample_titles"],
        })

    results.sort(key=lambda x: (-x["composite_score"], -x["articles"]))
    top = results[:TOP_N]

    window_hours = round((now - window_start).total_seconds() / 3600, 1)
    is_weekend = now.astimezone(ET).weekday() >= 5

    return {
        "generated_at": now.isoformat(),
        "window_start": window_start.isoformat(),
        "window_hours": window_hours,
        "is_weekend": is_weekend,
        "total_scanned": total_scanned,
        "qualifying_tickers": len(results),
        "top_movers": top,
    }


def main() -> None:
    now = datetime.now(timezone.utc)
    window_start = _weekend_window_start(now)

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        f"""
        SELECT title, source, first_seen, ml_score, ai_score, urgency
        FROM articles
        WHERE replace(first_seen,'T',' ') >= ?
          AND {_LIVE_ONLY_CLAUSE}
        ORDER BY first_seen DESC
        LIMIT {SCAN_LIMIT}
        """,
        (window_start.strftime("%Y-%m-%d %H:%M:%S"),),
    ).fetchall()
    conn.close()

    brief = build_weekend_brief(
        [(r["title"], r["source"], r["first_seen"], r["ml_score"], r["ai_score"], r["urgency"])
         for r in rows],
        window_start,
        now,
    )

    OUT_PATH.write_text(json.dumps(brief, indent=2))

    # CLI summary
    is_wknd = brief["is_weekend"]
    label = "WEEKEND" if is_wknd else "OVERNIGHT"
    print(f"[{label} CATALYST BRIEF] window={brief['window_hours']}h "
          f"scanned={brief['total_scanned']} qualifying={brief['qualifying_tickers']}")
    for item in brief["top_movers"][:5]:
        print(f"  {item['ticker']:6s}  composite={item['composite_score']:.3f}  "
              f"catalyst={item['top_catalyst']:12s}  articles={item['articles']}  "
              f"urgency={item['max_urgency']}")
    print(f"Output → {OUT_PATH}")


if __name__ == "__main__":
    main()

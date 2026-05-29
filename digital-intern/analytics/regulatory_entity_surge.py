"""Regulatory entity mention surge detector.

Tracks regulatory/enforcement body mentions (FINRA, SEC, DOJ, FTC, CFTC, FDIC,
OCC, CFPB, FBI, IRS) across articles in the last 2h vs prior 2h. Any entity
with >=5x surge is flagged. Cross-correlates with tickers that co-appear in
the same articles to identify potential enforcement targets.

Output: /home/zeph/logs/regulatory_entity_surge.json
Standalone: python3 -m analytics.regulatory_entity_surge
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from storage.article_store import _LIVE_ONLY_CLAUSE

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "articles.db"
OUT_PATH = Path("/home/zeph/logs/regulatory_entity_surge.json")
WINDOW_HOURS = 2
FETCH_LIMIT = 6000
SURGE_THRESHOLD = 5.0  # ratio for flagging

# Regulatory entities to track — full names and common abbreviations
ENTITIES: dict[str, list[str]] = {
    "FINRA":  ["FINRA", "Financial Industry Regulatory"],
    "SEC":    ["SEC", "Securities and Exchange Commission"],
    "DOJ":    ["DOJ", "Department of Justice", "Justice Department"],
    "FTC":    ["FTC", "Federal Trade Commission"],
    "CFTC":   ["CFTC", "Commodity Futures Trading Commission"],
    "FDIC":   ["FDIC", "Federal Deposit Insurance"],
    "OCC":    ["OCC", "Office of the Comptroller"],
    "CFPB":   ["CFPB", "Consumer Financial Protection"],
    "FBI":    ["FBI", "Federal Bureau of Investigation"],
    "IRS":    ["IRS", "Internal Revenue Service"],
    "OFAC":   ["OFAC", "Office of Foreign Assets Control"],
}

# Ticker extraction (same stop-list as trend_velocity)
_TICKER_RE = re.compile(r"\b\$?([A-Z]{2,5})\b")
_STOP = {
    "CEO", "CFO", "CTO", "USA", "USD", "EUR", "GBP", "EU", "UK", "US",
    "AI", "ML", "API", "IPO", "ETF", "SEC", "FOMC", "FED", "GDP", "CPI",
    "PPI", "ECB", "BOJ", "PBOC", "OPEC", "NYSE", "NASDAQ", "AMEX",
    "Q1", "Q2", "Q3", "Q4", "YTD", "YOY", "EPS", "PE", "EV", "ESG",
    "BUY", "SELL", "HOLD", "ON", "AT", "IN", "TO", "OF", "FOR", "THE",
    "AND", "OR", "BY", "AS", "IS", "WAS", "ARE", "BE", "AN", "A",
    "NEW", "OLD", "TOP", "LOW", "HIGH", "BIG", "DAY", "WEEK", "MONTH",
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
    "MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN",
    "INC", "LLC", "LTD", "CORP", "CO", "PLC",
    "MSN", "CNN", "BBC", "WSJ", "NYT", "FT", "AP", "AFP",
    "NEWS", "STOCK", "STOCKS", "MARKET", "DEAL", "DEALS",
    "JUNE", "JULY", "FINRA", "CFTC", "FDIC", "CFPB", "OFAC",
    "DOJ", "FTC", "OCC", "FBI", "IRS",
}


def _extract_entities(text: str) -> list[str]:
    found = []
    for entity, patterns in ENTITIES.items():
        for pat in patterns:
            if pat in text:
                found.append(entity)
                break
    return found


def _extract_tickers(text: str) -> list[str]:
    out = []
    for m in _TICKER_RE.findall(text or ""):
        if m not in _STOP and 2 <= len(m) <= 5:
            out.append(m)
    return out


def main() -> None:
    now = datetime.now(timezone.utc)
    cutoff_4h = now - timedelta(hours=WINDOW_HOURS * 2)
    cutoff_2h = now - timedelta(hours=WINDOW_HOURS)

    con = sqlite3.connect(str(DB_PATH), timeout=15)
    con.row_factory = sqlite3.Row

    rows = con.execute(
        f"""
        SELECT title, first_seen
        FROM articles
        WHERE {_LIVE_ONLY_CLAUSE}
          AND first_seen >= datetime('now', '-{WINDOW_HOURS * 2} hours')
        ORDER BY first_seen DESC
        LIMIT {FETCH_LIMIT}
        """,
    ).fetchall()
    con.close()

    # Split into now-window vs prior-window
    now_entity: Counter[str] = Counter()
    prev_entity: Counter[str] = Counter()
    # ticker co-mentions per entity, now-window only
    entity_tickers: dict[str, Counter[str]] = defaultdict(Counter)
    # sample titles per entity, now-window
    entity_samples: dict[str, list[str]] = defaultdict(list)

    for row in rows:
        title = row["title"] or ""
        ts_raw = row["first_seen"] or ""
        ts_str = ts_raw.replace("T", " ")[:19]
        try:
            ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        entities = _extract_entities(title)
        if not entities:
            continue

        in_now = ts >= cutoff_2h
        in_prev = ts >= cutoff_4h and ts < cutoff_2h

        for ent in entities:
            if in_now:
                now_entity[ent] += 1
                for tk in _extract_tickers(title):
                    entity_tickers[ent][tk] += 1
                if len(entity_samples[ent]) < 3:
                    entity_samples[ent].append(title[:120])
            elif in_prev:
                prev_entity[ent] += 1

    surges: list[dict] = []
    quiet: list[dict] = []

    for entity in ENTITIES:
        now_n = now_entity[entity]
        prev_n = prev_entity[entity]
        ratio = (now_n + 1) / (prev_n + 1)  # +1 Laplace smoothing
        top_tickers = [t for t, _ in entity_tickers[entity].most_common(5)]
        record = {
            "entity": entity,
            "now_2h": now_n,
            "prev_2h": prev_n,
            "ratio": round(ratio, 2),
            "surge": ratio >= SURGE_THRESHOLD and now_n >= 3,
            "top_tickers": top_tickers,
            "samples": entity_samples[entity],
        }
        if record["surge"]:
            surges.append(record)
        else:
            quiet.append(record)

    surges.sort(key=lambda x: x["ratio"], reverse=True)

    result = {
        "generated_at": now.isoformat(),
        "window_hours": WINDOW_HOURS,
        "scanned": len(rows),
        "surge_threshold": SURGE_THRESHOLD,
        "surges_detected": len(surges),
        "surges": surges,
        "all_entities": surges + sorted(quiet, key=lambda x: x["now_2h"], reverse=True),
    }
    OUT_PATH.write_text(json.dumps(result, indent=2))

    for s in surges:
        tickers_str = ",".join(s["top_tickers"]) if s["top_tickers"] else "none"
        print(
            f"SURGE {s['entity']}: {s['now_2h']} articles (2h) vs {s['prev_2h']} prior | "
            f"{s['ratio']}x | co-tickers: {tickers_str}"
        )
    if not surges:
        top = sorted(result["all_entities"], key=lambda x: x["now_2h"], reverse=True)[:3]
        for e in top:
            print(f"  {e['entity']}: now={e['now_2h']} prev={e['prev_2h']} ratio={e['ratio']}x")
    print(f"regulatory_entity_surge: {len(surges)} surge(s) | scanned={len(rows)} | wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

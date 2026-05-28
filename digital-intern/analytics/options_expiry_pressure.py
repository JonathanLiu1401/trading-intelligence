"""Options expiry news pressure monitor.

Reads open option positions from config/portfolio.json (options array).
For positions expiring within EXPIRY_DAYS, pulls the last LOOKBACK_H hours
of live articles mentioning the underlying ticker and reports:
  - days_to_expiry
  - article_count         (recent coverage depth)
  - avg_ml_score          (signal quality; None if no ml-scored articles)
  - top_headlines         (up to TOP_HEADLINES, newest first)

Distinct from analytics.opex_news_overlap — that module tracks market-wide
OpEx calendar dates vs. article velocity; this module reads the analyst's
ACTUAL option positions from the portfolio SSOT (config/portfolio.json) and
answers "what does today's news say about my expiring options?"

Output: /home/zeph/logs/options_expiry_pressure.json
Standalone: python3 -m analytics.options_expiry_pressure
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

try:
    from storage.article_store import _LIVE_ONLY_CLAUSE, _get_db_path
    DB_PATH = _get_db_path()
except Exception:
    _LIVE_ONLY_CLAUSE = (
        "url NOT LIKE 'backtest://%' "
        "AND source NOT LIKE 'backtest_%' "
        "AND source NOT LIKE 'opus_annotation%'"
    )
    DB_PATH = BASE / "data" / "articles.db"

PORTFOLIO_PATH = BASE / "config" / "portfolio.json"
OUT_PATH = Path("/home/zeph/logs/options_expiry_pressure.json")

EXPIRY_DAYS = 7      # flag options expiring within this many days
SCAN_LIMIT = 3000    # bounded idx_first_seen scan (covers ~1-3h at live ingest rate)
TOP_HEADLINES = 3


def _parse_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        s = str(raw).replace("T", " ").replace("Z", "").split("+")[0][:19]
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _mentions(title: str, ticker: str) -> bool:
    return bool(re.search(rf"\b\$?{re.escape(ticker)}\b", title or "", re.IGNORECASE))


def _load_options() -> list[dict]:
    try:
        return json.loads(PORTFOLIO_PATH.read_text()).get("options", [])
    except Exception:
        return []


def run() -> dict:
    now = datetime.now(timezone.utc)
    today = now.date()

    options = _load_options()
    results: list[dict] = []

    # One shared scan for all options — fast idx_first_seen bounded read
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=12)
    rows = con.execute(
        f"SELECT title, ml_score, ai_score, first_seen, source "
        f"FROM articles WHERE {_LIVE_ONLY_CLAUSE} "
        f"ORDER BY first_seen DESC LIMIT {SCAN_LIMIT}"
    ).fetchall()
    con.close()

    scan_oldest = _parse_ts(rows[-1][3]) if rows else None

    for opt in options:
        expiry_str = opt.get("expiry", "")
        underlying = (opt.get("underlying") or "").upper().strip()
        if not expiry_str or not underlying:
            continue
        try:
            expiry_date = date.fromisoformat(expiry_str)
        except ValueError:
            continue

        days_left = (expiry_date - today).days
        if days_left < 0 or days_left > EXPIRY_DAYS:
            continue

        matching = [
            {"title": r[0], "ml_score": r[1], "ai_score": r[2],
             "first_seen": r[3], "source": r[4]}
            for r in rows
            if _mentions(r[0], underlying)
        ]

        ml_scores = [m["ml_score"] for m in matching if m["ml_score"] is not None]
        avg_ml = round(sum(ml_scores) / len(ml_scores), 4) if ml_scores else None

        results.append({
            "symbol": opt.get("symbol", ""),
            "underlying": underlying,
            "type": opt.get("type", ""),
            "strike": opt.get("strike"),
            "qty": opt.get("qty"),
            "expiry": expiry_str,
            "days_to_expiry": days_left,
            "article_count": len(matching),
            "avg_ml_score": avg_ml,
            "ml_scored_count": len(ml_scores),
            "top_headlines": [
                {"title": h["title"], "first_seen": h["first_seen"],
                 "source": h["source"], "ml_score": h["ml_score"]}
                for h in matching[:TOP_HEADLINES]
            ],
        })

    out = {
        "generated_at": now.isoformat(),
        "scan_rows": len(rows),
        "scan_oldest": scan_oldest.isoformat() if scan_oldest else None,
        "expiry_window_days": EXPIRY_DAYS,
        "expiring_options": len(results),
        "options": results,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    result = run()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

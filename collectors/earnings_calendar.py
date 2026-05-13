"""Earnings calendar collector using yfinance."""
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent.parent
WATCHLIST_PATH = BASE_DIR / "config" / "watchlist.json"


def _load_watchlist():
    with open(WATCHLIST_PATH, "r") as f:
        return json.load(f)


def _all_tickers(watchlist):
    tickers = []
    for key in ("memory_core", "semis_equipment", "broader_semis", "korean", "japanese"):
        tickers.extend(watchlist.get(key, []))
    seen = set()
    return [t for t in tickers if not (t in seen or seen.add(t))]


def get_earnings():
    """Return earnings events within the next 48 hours for watchlist tickers."""
    watchlist = _load_watchlist()
    tickers = _all_tickers(watchlist)

    now = datetime.now(timezone.utc)
    horizon = now + timedelta(hours=48)
    upcoming = []

    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            cal = None
            try:
                cal = t.calendar
            except Exception:
                cal = None

            earnings_date = None
            if cal is not None:
                # yfinance returns either a DataFrame or a dict depending on version
                if isinstance(cal, dict):
                    earnings_date = cal.get("Earnings Date")
                else:
                    try:
                        earnings_date = cal.loc["Earnings Date"].iloc[0]
                    except Exception:
                        earnings_date = None

            if not earnings_date:
                continue

            # Normalize to list
            candidates = earnings_date if isinstance(earnings_date, list) else [earnings_date]
            for ed in candidates:
                try:
                    if hasattr(ed, "to_pydatetime"):
                        ed = ed.to_pydatetime()
                    if isinstance(ed, datetime):
                        if ed.tzinfo is None:
                            ed = ed.replace(tzinfo=timezone.utc)
                        if now <= ed <= horizon:
                            upcoming.append({"ticker": ticker, "earnings_date": ed.isoformat()})
                            break
                except Exception:
                    continue
        except Exception as e:
            print(f"[earnings_calendar] Error for {ticker}: {e}")
            continue

    return upcoming


if __name__ == "__main__":
    events = get_earnings()
    for e in events:
        print(f"{e['ticker']}: {e['earnings_date']}")

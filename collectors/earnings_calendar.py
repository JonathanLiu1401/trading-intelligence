"""Earnings calendar collector using yfinance.

Two modes:
- ``get_earnings()`` — 48h horizon, kept for backward compatibility with the
  heartbeat worker which folds the result into Opus's briefing payload.
- ``get_earnings_extended(horizon_days=14)`` — wider window used by the
  dashboard card and the Position Watchlist "EARN in Nd" badge. Also persisted
  to ``data/earnings_calendar.json`` via ``write_snapshot()`` so a single
  yfinance sweep serves both the daemon's briefing and the dashboard refresh.

Ticker universe is the union of ``config/portfolio.json`` (live holdings + the
``sector_watchlist`` block) and ``config/watchlist.json`` (the legacy file that
still drives the heartbeat briefing). Held tickers must appear so the
Position Watchlist EARN badge resolves correctly.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from datetime import date, datetime, timedelta, timezone

import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent.parent
WATCHLIST_PATH = BASE_DIR / "config" / "watchlist.json"
PORTFOLIO_PATH = BASE_DIR / "config" / "portfolio.json"
SNAPSHOT_PATH = BASE_DIR / "data" / "earnings_calendar.json"


def _load_watchlist() -> dict:
    try:
        with open(WATCHLIST_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_portfolio() -> dict:
    try:
        with open(PORTFOLIO_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _all_tickers() -> list[str]:
    """Union of watchlist groups + portfolio positions + sector_watchlist.

    Yields a stable, de-duplicated order so cached snapshots are deterministic.
    """
    out: list[str] = []
    seen: set[str] = set()

    watchlist = _load_watchlist()
    for key in ("memory_core", "semis_equipment", "broader_semis", "korean", "japanese", "portfolio"):
        for t in watchlist.get(key, []) or []:
            if t and t not in seen:
                seen.add(t)
                out.append(t)

    portfolio = _load_portfolio()
    for pos in portfolio.get("positions", []) or []:
        t = pos.get("ticker")
        # Skip closed positions (qty=0) so we don't waste yfinance calls on
        # tickers the user no longer holds; but keep them in the heartbeat
        # universe by re-adding via sector_watchlist below if relevant.
        if t and t not in seen and float(pos.get("qty") or 0) > 0:
            seen.add(t)
            out.append(t)
    for t in portfolio.get("sector_watchlist", []) or []:
        if t and t not in seen:
            seen.add(t)
            out.append(t)

    return out


def _normalize_earnings_date(cal) -> list[datetime]:
    """Pull all earnings_date candidates from a yfinance calendar object.

    yfinance returns either a dict or a DataFrame depending on the version, and
    the field may be a single Timestamp or a list. Returns a list of timezone-
    aware datetimes (UTC).
    """
    if cal is None:
        return []
    earnings_date = None
    try:
        if isinstance(cal, dict):
            earnings_date = cal.get("Earnings Date")
        else:
            try:
                earnings_date = cal.loc["Earnings Date"].iloc[0]
            except Exception:
                earnings_date = None
    except Exception:
        return []
    if not earnings_date:
        return []
    candidates = earnings_date if isinstance(earnings_date, list) else [earnings_date]
    out: list[datetime] = []
    for ed in candidates:
        try:
            if hasattr(ed, "to_pydatetime"):
                ed = ed.to_pydatetime()
            # yfinance returns a mix of datetime.date and datetime.datetime
            # depending on field/version; coerce date → midnight-UTC datetime.
            if isinstance(ed, datetime):
                if ed.tzinfo is None:
                    ed = ed.replace(tzinfo=timezone.utc)
                out.append(ed)
            elif isinstance(ed, date):
                out.append(datetime(ed.year, ed.month, ed.day, tzinfo=timezone.utc))
        except Exception:
            continue
    return out


def _scan(horizon_days: int) -> list[dict]:
    """yfinance sweep over the ticker union, return events in [now, now+horizon]."""
    now = datetime.now(timezone.utc)
    horizon = now + timedelta(days=horizon_days)
    tickers = _all_tickers()
    events: list[dict] = []
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            try:
                cal = t.calendar
            except Exception:
                cal = None
            dates = _normalize_earnings_date(cal)
            for ed in dates:
                if now <= ed <= horizon:
                    days_away = (ed - now).total_seconds() / 86400.0
                    events.append({
                        "ticker": ticker,
                        "earnings_date": ed.isoformat(),
                        "days_away": round(days_away, 2),
                    })
                    break
        except Exception as e:
            # Don't let a single bad pull poison the whole snapshot.
            print(f"[earnings_calendar] Error for {ticker}: {e}")
            continue
    events.sort(key=lambda r: r["days_away"])
    return events


def get_earnings() -> list[dict]:
    """Backward-compatible 48h-horizon call used by the heartbeat briefing."""
    return _scan(horizon_days=2)


def get_earnings_extended(horizon_days: int = 14) -> list[dict]:
    """Wider sweep for the dashboard card + watchlist EARN badge."""
    return _scan(horizon_days=horizon_days)


def write_snapshot(horizon_days: int = 14, path: Path | None = None) -> dict:
    """Run an extended scan and write the result atomically to JSON.

    Returns the snapshot dict that was written. Atomic write avoids the
    dashboard ever reading a half-written file: write to a tmp sibling, fsync,
    then ``os.replace`` onto the canonical path.
    """
    if path is None:
        path = SNAPSHOT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    events = get_earnings_extended(horizon_days=horizon_days)
    snap = {
        "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "horizon_days": horizon_days,
        "n_events": len(events),
        "events": events,
    }

    fd, tmp_name = tempfile.mkstemp(prefix=".earnings_", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(snap, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        # Best-effort cleanup if replace failed.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return snap


def read_snapshot(path: Path | None = None) -> dict | None:
    """Read the persisted snapshot. Returns None if missing or unreadable."""
    if path is None:
        path = SNAPSHOT_PATH
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


if __name__ == "__main__":
    snap = write_snapshot(horizon_days=14)
    print(f"[earnings_calendar] wrote {snap['n_events']} events to {SNAPSHOT_PATH}")
    for e in snap["events"]:
        print(f"  {e['ticker']:8} {e['earnings_date']} ({e['days_away']:.1f}d)")

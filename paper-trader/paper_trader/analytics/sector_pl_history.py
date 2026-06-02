"""Cumulative realized P&L by sector — where the desk made & lost money.

``/api/sector-exposure`` / ``/api/analytics`` show the **current** book's
sector distribution — a static snapshot of where capital sits today.
``/api/sector-signal-fit`` audits per-sector signal usage. Nothing yet
rolls up the **realized** P&L ledger by sector over time, so the obvious
trader question — *which sectors have actually paid me, and which have
bled me, over the last week / month / lifetime?* — has no panel.

This module is exactly that rollup. It takes the trades log, derives
round-trips, classifies each ticker via the
``analytics.sector_exposure.SECTOR_MAP`` (the same dict that drives
``/api/sector-exposure`` — drift-locked by tests), and aggregates total
realized P&L + win rate + trip count per sector across three time windows:
``last_7d``, ``last_30d``, ``all_time``.

Pure helper. Read-only. Never raises, never writes, never trains, never
has a path to ``_execute``. Time windows close on the closing SELL's
``closed_at`` timestamp — partial-cycle round-trips that opened inside
the window but haven't closed are correctly excluded (the realized side
is what this surface reports).

Run as a CLI::

    python3 -m paper_trader.analytics.sector_pl_history
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .round_trips_derived import derive_round_trips
from .sector_exposure import classify


_WINDOWS = (
    ("last_7d", 7),
    ("last_30d", 30),
    ("all_time", None),
)


def _parse_ts(ts):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _in_window(closed_at_dt, now, days):
    """``True`` iff ``closed_at_dt`` is within ``days`` of ``now``. ``days=None``
    means the all-time window (every row qualifies)."""
    if closed_at_dt is None:
        return False
    if days is None:
        return True
    cutoff = now - timedelta(days=days)
    return closed_at_dt >= cutoff


def _aggregate(round_trips, now, days):
    """Group qualifying round-trips by sector and emit a per-sector dict."""
    by_sector: dict[str, dict] = {}
    n_total = 0
    pl_total = 0.0
    for rt in round_trips:
        if not isinstance(rt, dict):
            continue
        closed_at = _parse_ts(rt.get("closed_at"))
        if not _in_window(closed_at, now, days):
            continue
        sec = classify(rt.get("ticker"))
        bucket = by_sector.setdefault(sec, {
            "sector": sec,
            "n_trips": 0,
            "n_wins": 0,
            "n_losses": 0,
            "total_pl_usd": 0.0,
            "total_cost_usd": 0.0,
            "tickers": set(),
        })
        try:
            pl = float(rt.get("realized_pl") or 0.0)
        except (TypeError, ValueError):
            pl = 0.0
        try:
            cost = float(rt.get("cost") or 0.0)
        except (TypeError, ValueError):
            cost = 0.0
        bucket["n_trips"] += 1
        bucket["total_pl_usd"] += pl
        bucket["total_cost_usd"] += cost
        if pl > 0:
            bucket["n_wins"] += 1
        elif pl < 0:
            bucket["n_losses"] += 1
        if rt.get("ticker"):
            bucket["tickers"].add(rt["ticker"])
        n_total += 1
        pl_total += pl

    out = []
    for sec in sorted(by_sector):
        b = by_sector[sec]
        decided = b["n_wins"] + b["n_losses"]
        win_rate = (round(100.0 * b["n_wins"] / decided, 2)
                    if decided else None)
        avg_pl_pct = (round(b["total_pl_usd"] / b["total_cost_usd"] * 100.0, 2)
                      if b["total_cost_usd"] > 1e-9 else None)
        out.append({
            "sector": sec,
            "n_trips": b["n_trips"],
            "n_wins": b["n_wins"],
            "n_losses": b["n_losses"],
            "win_rate_pct": win_rate,
            "total_pl_usd": round(b["total_pl_usd"], 2),
            "total_cost_usd": round(b["total_cost_usd"], 2),
            "avg_pl_pct": avg_pl_pct,
            "tickers": sorted(b["tickers"]),
        })
    # Descending by total_pl_usd so the leader sector is row 0 — readers can
    # peel the top/bottom without re-sorting.
    out.sort(key=lambda r: r["total_pl_usd"], reverse=True)
    return out, n_total, round(pl_total, 2)


def _verdict_and_headline(by_window):
    """Verdict ladder + one-line headline from the all-time slice.

    * ``NO_CLOSED_TRIPS`` — every window is empty.
    * ``SINGLE_SECTOR`` — every closed round-trip sat in one sector (no
      diversification edge to report).
    * ``MIXED`` — at least two sectors have closed round-trips. Headline
      points to the leader/laggard from the all-time window.
    """
    all_time = by_window["all_time"]["sectors"]
    n_all = by_window["all_time"]["n_round_trips"]
    if n_all == 0:
        return "NO_CLOSED_TRIPS", "No closed round-trips on record yet."
    if len(all_time) == 1:
        only = all_time[0]
        return "SINGLE_SECTOR", (
            f"All {n_all} closed round-trip{'s' if n_all != 1 else ''} sat "
            f"in sector ``{only['sector']}`` "
            f"(${only['total_pl_usd']:+.2f} total)."
        )
    leader = all_time[0]
    laggard = all_time[-1]
    if leader["total_pl_usd"] == laggard["total_pl_usd"]:
        # Two-or-more sectors tied at the same total — flat ledger.
        return "MIXED", (
            f"{n_all} closed round-trips across {len(all_time)} sectors; "
            f"net realized ${by_window['all_time']['total_pl_usd']:+.2f}."
        )
    return "MIXED", (
        f"{n_all} closed round-trips across {len(all_time)} sectors. "
        f"Leader: ``{leader['sector']}`` ${leader['total_pl_usd']:+.2f} "
        f"({leader['n_trips']} trip{'s' if leader['n_trips'] != 1 else ''}). "
        f"Laggard: ``{laggard['sector']}`` ${laggard['total_pl_usd']:+.2f} "
        f"({laggard['n_trips']} trip{'s' if laggard['n_trips'] != 1 else ''})."
    )


def build(trades, now=None):
    """Top-level: take raw trade rows, return the time-windowed envelope.

    Parameters
    ----------
    trades : iterable of dict
        Output of ``store.recent_trades(limit=N)``. Append-only, so the same
        rows can be re-walked safely across calls.
    now : datetime or None
        Cutoff for the rolling windows. Defaults to ``datetime.now(UTC)``
        — explicit override is for tests so windows are deterministic.

    Returns
    -------
    dict
        ``{verdict, headline, windows: {last_7d, last_30d, all_time},
           as_of}``. Each window carries ``sectors`` (descending by
        ``total_pl_usd``), ``n_round_trips``, and ``total_pl_usd``.
        Never raises.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    round_trips = derive_round_trips(trades)

    by_window: dict[str, dict] = {}
    for label, days in _WINDOWS:
        sectors, n, total = _aggregate(round_trips, now, days)
        by_window[label] = {
            "sectors": sectors,
            "n_round_trips": n,
            "total_pl_usd": total,
        }

    verdict, headline = _verdict_and_headline(by_window)
    return {
        "verdict": verdict,
        "headline": headline,
        "windows": by_window,
        "as_of": now.isoformat(),
    }


if __name__ == "__main__":
    import json
    from ..store import get_store
    store = get_store()
    trades = store.recent_trades(limit=5000)
    print(json.dumps(build(trades), indent=2, default=str))

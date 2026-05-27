"""Today's realized P/L — what did the desk actually BANK today?

The existing realized-P/L surfaces each describe a different shape:

  * ``/api/realized-vs-unrealized`` — ALL-TIME realized vs unrealized split
    since the $1000 start. The headline framing of "today's $199.52 gain" is
    misleading — it includes every realized dollar since the start, not the
    day's actual fills.
  * ``/api/today-action-tape``      — flat aggregate (n_buys, n_sells,
    notional_in/out, net_cash_flow) of *every* decision today. No P/L
    framing per closed lot, no per-name win/loss, no biggest mover.
  * ``/api/closed-positions``       — every closed lot in history with
    realized_pl per round-trip. Unbounded by date, no day filter.

None of them answer the question every live operator's first morning click
asks: **"how did yesterday close out — what did I actually lock in, and on
which names?"**.

This module walks ``store.closed_positions(N)`` (each row already carries the
matched-round-trip ``realized_pl`` from ``store.py``'s same-key trade walk),
filters to lots whose ``closed_at`` falls on TODAY in NY-session time (the
canonical trading day for daily-close-anchored UX), and rolls up an
operator-facing summary:

  - ``net_realized_usd``: sum of realized_pl across today's closures
  - ``net_realized_pct``: net_realized / total_cost_basis_today (the actual
    return on capital deployed in today's round-trips; not the % of starting
    book)
  - ``n_winners`` / ``n_losers`` / ``n_closes`` (closes are winners +
    losers + scratch)
  - ``biggest_win`` / ``biggest_loss`` (None on no-closes-today)
  - ``avg_hold_seconds`` / ``avg_hold_hours``: how long the day's lots
    were held on average
  - ``closes``: per-lot records sorted by ``realized_pl`` desc (best first)

Verdict ladder (silence-by-default per AGENTS.md invariants #2/#12 — the
hourly summary should not become its own lying green light):

  * ``NO_CLOSES_TODAY`` — no closed_at falls on today (silent in Discord;
                          the operator already sees cash unchanged).
  * ``WINNING_DAY``      — net realized > ``_BREAKEVEN_EPSILON_USD``.
  * ``LOSING_DAY``       — net realized < -``_BREAKEVEN_EPSILON_USD``.
  * ``BREAKEVEN_DAY``    — |net realized| ≤ ``_BREAKEVEN_EPSILON_USD``.

Pure builder, never raises. Observational only — never gates Opus, no
caps, no path to ``_execute()``. Single source of truth: the per-lot
``realized_pl`` field is the same one ``store.py``'s round-trip walker
emits for ``/api/closed-positions``, so this surface and that one can
never disagree on a closed lot's realized P/L.

Run as a CLI for a one-shot ops view::

    python3 -m paper_trader.analytics.today_realized_pl
"""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")

# Sub-cent absolute net = "scratch / breakeven". Float rounding from
# yfinance fills (per-share, 4dp) on small round-trips routinely produces a
# net realized $0.0009 — calling that a WINNING_DAY is misleading.
_BREAKEVEN_EPSILON_USD = 0.01

# Cap on per-close rows surfaced in the response so a very busy day stays
# bounded. The headline still aggregates over every match; only the
# per-lot `closes` array is capped.
_MAX_CLOSES_RENDERED = 20


def _ny_today(now: datetime | None = None):
    """Return today's date in NY-tz (the canonical trading day)."""
    n = now or datetime.now(timezone.utc)
    if n.tzinfo is None:
        n = n.replace(tzinfo=timezone.utc)
    return n.astimezone(NY).date()


def _parse_iso_to_ny_date(ts):
    """Parse an ISO-8601 string to its NY-tz date, or None on bad input."""
    if not ts:
        return None
    if isinstance(ts, datetime):
        dt = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    else:
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(NY).date()


def build_today_realized_pl(closed_positions, now: datetime | None = None) -> dict:
    """Builder. Pure: never raises, never reads from store/network.

    ``closed_positions`` is the shape ``store.closed_positions(N)`` returns:
    each row is a dict with ``ticker``, ``closed_at`` (ISO UTC),
    ``realized_pl`` (float or None), ``cost`` (float, total cost basis of
    the round-trip), ``hold_seconds`` (int or None), ``type``, plus other
    metadata. Rows missing ``closed_at`` or ``realized_pl`` are filtered.
    Rows whose ``closed_at`` doesn't parse cleanly are filtered.

    Returns a stable shape suitable for a JSON endpoint:
    ``{verdict, headline, ny_date, net_realized_usd, net_realized_pct,
       n_closes, n_winners, n_losers, n_scratch, biggest_win, biggest_loss,
       avg_hold_seconds, avg_hold_hours, closes}``.
    """
    today = _ny_today(now)
    today_iso = today.isoformat()

    rows = closed_positions or []
    if not isinstance(rows, list):
        return _empty(today_iso, headline="no closed positions data available")

    today_closes: list[dict] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        d = _parse_iso_to_ny_date(r.get("closed_at"))
        if d is None or d != today:
            continue
        pl = r.get("realized_pl")
        if pl is None:
            continue
        try:
            pl_f = float(pl)
        except (TypeError, ValueError):
            continue
        try:
            cost_f = float(r.get("cost") or 0.0)
        except (TypeError, ValueError):
            cost_f = 0.0
        try:
            hs_raw = r.get("hold_seconds")
            hold_seconds = int(hs_raw) if hs_raw is not None else None
        except (TypeError, ValueError):
            hold_seconds = None
        today_closes.append({
            "ticker": r.get("ticker") or "?",
            "type": r.get("type") or "stock",
            "closed_at": r.get("closed_at"),
            "opened_at": r.get("opened_at"),
            "realized_pl": round(pl_f, 4),
            "realized_pl_pct": r.get("realized_pl_pct"),
            "cost": round(cost_f, 4),
            "hold_seconds": hold_seconds,
            "n_trades": r.get("n_trades"),
        })

    n_closes = len(today_closes)
    if n_closes == 0:
        return _empty(today_iso, headline="no closed positions today")

    net_usd = round(sum(c["realized_pl"] for c in today_closes), 4)
    total_cost = round(sum(c["cost"] for c in today_closes), 4)
    net_pct: float | None
    if total_cost > 1e-9:
        net_pct = round(net_usd / total_cost * 100.0, 4)
    else:
        net_pct = None

    n_winners = sum(1 for c in today_closes if c["realized_pl"] > _BREAKEVEN_EPSILON_USD)
    n_losers = sum(1 for c in today_closes if c["realized_pl"] < -_BREAKEVEN_EPSILON_USD)
    n_scratch = n_closes - n_winners - n_losers

    sorted_by_pl = sorted(today_closes, key=lambda c: c["realized_pl"], reverse=True)
    biggest_win = sorted_by_pl[0] if sorted_by_pl[0]["realized_pl"] > _BREAKEVEN_EPSILON_USD else None
    biggest_loss = (
        sorted_by_pl[-1]
        if sorted_by_pl[-1]["realized_pl"] < -_BREAKEVEN_EPSILON_USD
        else None
    )

    valid_holds = [c["hold_seconds"] for c in today_closes if c["hold_seconds"] is not None]
    if valid_holds:
        avg_hs = int(sum(valid_holds) / len(valid_holds))
        avg_hh = round(avg_hs / 3600.0, 4)
    else:
        avg_hs = None
        avg_hh = None

    if net_usd > _BREAKEVEN_EPSILON_USD:
        verdict = "WINNING_DAY"
        sign = "+"
    elif net_usd < -_BREAKEVEN_EPSILON_USD:
        verdict = "LOSING_DAY"
        sign = ""
    else:
        verdict = "BREAKEVEN_DAY"
        sign = "+" if net_usd >= 0 else ""

    win_loss = f"{n_winners}W/{n_losers}L"
    if n_scratch:
        win_loss += f"/{n_scratch}S"
    headline = (
        f"{verdict} — {sign}${net_usd:.2f} realized over {n_closes} "
        f"close{'s' if n_closes != 1 else ''} ({win_loss})"
    )
    if biggest_win is not None:
        bw_tk = biggest_win["ticker"]
        bw_pl = biggest_win["realized_pl"]
        headline += f"; best {bw_tk} +${bw_pl:.2f}"
    if biggest_loss is not None:
        bl_tk = biggest_loss["ticker"]
        bl_pl = biggest_loss["realized_pl"]
        headline += f"; worst {bl_tk} -${abs(bl_pl):.2f}"

    return {
        "verdict": verdict,
        "headline": headline,
        "ny_date": today_iso,
        "net_realized_usd": net_usd,
        "net_realized_pct": net_pct,
        "total_cost_basis_usd": total_cost,
        "n_closes": n_closes,
        "n_winners": n_winners,
        "n_losers": n_losers,
        "n_scratch": n_scratch,
        "biggest_win": biggest_win,
        "biggest_loss": biggest_loss,
        "avg_hold_seconds": avg_hs,
        "avg_hold_hours": avg_hh,
        "closes": sorted_by_pl[:_MAX_CLOSES_RENDERED],
    }


def _empty(today_iso: str, headline: str) -> dict:
    return {
        "verdict": "NO_CLOSES_TODAY",
        "headline": headline,
        "ny_date": today_iso,
        "net_realized_usd": 0.0,
        "net_realized_pct": None,
        "total_cost_basis_usd": 0.0,
        "n_closes": 0,
        "n_winners": 0,
        "n_losers": 0,
        "n_scratch": 0,
        "biggest_win": None,
        "biggest_loss": None,
        "avg_hold_seconds": None,
        "avg_hold_hours": None,
        "closes": [],
    }


if __name__ == "__main__":
    import json
    from ..store import get_store
    store = get_store()
    closed = store.closed_positions(limit=500)
    out = build_today_realized_pl(closed)
    print(json.dumps(out, indent=2, default=str))

"""Round-trips derived directly from the ``trades`` table.

``store.closed_positions(N)`` only sees lots whose ``positions`` row currently
carries ``closed_at IS NOT NULL``. But ``upsert_position`` reactivates a closed
row in place whenever the same (ticker, type, expiry, strike) key re-opens —
``closed_at`` is set back to NULL on the next BUY. Every completed round-trip
sitting on that reactivated row vanishes from ``/api/closed-positions``,
``/api/today-realized-pl``, the per-lot win-rate calc, and the autopsy
track-record.

Live evidence (2026-05-27): MU position id=4 has been re-opened multiple times
in place. Two real MU round-trips both produced realized P/L (2026-05-26
SELL @ 889.50 → +$180; 2026-05-27 SELL @ 928.41 → +$32) but neither shows up
in ``closed_positions`` because the row is currently reactivated. ~$212 of
realized gains are invisible to closed-position analytics.

This module re-derives round-trips from the ``trades`` table directly. The
trade log is append-only — never reactivated, never overwritten — so every
completed BUY-to-flat sequence on a given key is visible. The output shape
matches ``store.closed_positions(N)`` so existing builders (e.g.
``today_realized_pl.build_today_realized_pl``) can consume it unchanged.

Pure builder, never raises. Read-only — never writes to the store, never
trains, never has a path to ``_execute``. No SSOT split with
``store.closed_positions``: every closed row that closed_positions emits has
a corresponding round-trip in this output (a regression-pinned invariant).
The reverse is NOT true — reactivated lots' historical round-trips only show
up here.

Run as a CLI::

    python3 -m paper_trader.analytics.round_trips_derived
"""
from __future__ import annotations

from datetime import datetime, timezone


def _hold_duration(opened_at, closed_at):
    """Same shape as store._hold_duration. Returns (hold_seconds:int|None,
    hold_days:float|None). Negative span (clock step-back) clamps to zero."""
    if not opened_at or not closed_at:
        return None, None
    try:
        op = datetime.fromisoformat(str(opened_at).replace("Z", "+00:00"))
        cl = datetime.fromisoformat(str(closed_at).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None, None
    secs = (cl - op).total_seconds()
    if secs < 0:
        secs = 0.0
    return int(secs), round(secs / 86400.0, 4)


def _trade_key(t):
    """Group key for a trade row. Stocks have NULL option_type/expiry/strike;
    options have all three. The IFNULL-equivalent (None → empty string / 0)
    matches store.closed_positions' SQL group-by exactly so an option round-
    trip never bleeds into a stock round-trip with the same ticker."""
    return (
        t.get("ticker") or "",
        (t.get("option_type") or None) if t.get("option_type") else None,
        t.get("expiry") or None,
        t.get("strike") if t.get("strike") is not None else None,
    )


def _ptype_from_key(key):
    """Position ``type`` field derived from a key. Mirrors the closed_positions
    row shape — stock keys map to ``stock``; option keys map to ``call`` /
    ``put``."""
    _t, opt, _e, _s = key
    if opt in ("call", "put"):
        return opt
    return "stock"


def derive_round_trips(trades, limit=None):
    """Walk every trade chronologically and emit one dict per completed
    round-trip (BUY → ... → SELL leaving held ≈ 0 on the key).

    Parameters
    ----------
    trades : iterable of dict
        Each row at minimum needs ``timestamp``, ``ticker``, ``action``,
        ``qty``, ``value``. Optional: ``price``, ``option_type``, ``expiry``,
        ``strike``. The list is grouped internally by
        (ticker, option_type, expiry, strike), then walked chronologically
        per group. Rows are tolerant of None/missing fields.
    limit : int or None
        Cap on the number of round-trips returned (newest-closed first).
        ``None`` returns every round-trip found.

    Returns
    -------
    list of dict
        Each round-trip dict matches the shape ``store.closed_positions(N)``
        emits — same keys (``ticker``, ``type``, ``qty``, ``avg_cost``,
        ``expiry``, ``strike``, ``opened_at``, ``closed_at``, ``realized_pl``,
        ``realized_pl_pct``, ``cost``, ``proceeds``, ``n_trades``,
        ``hold_seconds``, ``hold_days``) so existing builders consume the
        output unchanged. Sorted newest-closed first.
    """
    if not trades:
        return []

    # Group trades by key. Bad rows (no ticker, unparseable qty/value) drop.
    groups: dict[tuple, list[dict]] = {}
    for t in trades:
        if not isinstance(t, dict):
            continue
        if not t.get("ticker"):
            continue
        if not t.get("action"):
            continue
        key = _trade_key(t)
        groups.setdefault(key, []).append(t)

    # Sort each group chronologically. Ties on timestamp break on id (matches
    # store.closed_positions' "ORDER BY timestamp ASC, id ASC" exactly so the
    # round-trip slices align byte-for-byte with what closed_positions sees).
    round_trips: list[dict] = []
    for key, key_trades in groups.items():
        key_trades.sort(key=lambda r: (
            str(r.get("timestamp") or ""),
            int(r.get("id") or 0),
        ))
        ptype = _ptype_from_key(key)
        _, opt, expiry, strike = key
        ticker = key[0]

        held = 0.0
        start_idx = 0
        # Walk the trades; close a round-trip every time held returns to ≈0.
        for i, t in enumerate(key_trades):
            act = (t.get("action") or "").upper()
            try:
                q = float(t.get("qty") or 0.0)
            except (TypeError, ValueError):
                q = 0.0
            if act.startswith("BUY"):
                if abs(held) < 1e-6:
                    start_idx = i
                held += q
            elif act.startswith("SELL"):
                held -= q
                if abs(held) < 1e-6:
                    # Slice [start_idx..i] is one complete round-trip.
                    slice_ = key_trades[start_idx:i + 1]
                    rt = _round_trip_from_slice(
                        slice_, ticker, ptype, opt, expiry, strike,
                    )
                    if rt is not None:
                        round_trips.append(rt)

    # Newest-closed first. closed_at is the lexically-comparable ISO-8601
    # timestamp of the closing SELL — same field closed_positions sorts on.
    round_trips.sort(
        key=lambda d: str(d.get("closed_at") or ""), reverse=True,
    )
    if limit is not None:
        return round_trips[: max(0, int(limit))]
    return round_trips


def _round_trip_from_slice(slice_, ticker, ptype, opt, expiry, strike):
    """Aggregate one round-trip slice into the closed_positions row shape.

    ``slice_`` is the list of trades from the opening BUY (held was ≈0) to the
    closing SELL (held returned to ≈0) inclusive. Walks every action,
    summing realized P/L, cost, proceeds; opened_at = first BUY's timestamp,
    closed_at = last SELL's timestamp. Quantity = total BUY qty so qty/cost
    framing matches what ``store.closed_positions`` emits for the lot.
    """
    if not slice_:
        return None
    realized = 0.0
    cost = 0.0
    proceeds = 0.0
    buy_qty_total = 0.0
    buy_value_total = 0.0
    opened_at = None
    closed_at = None
    n_trades = 0

    for t in slice_:
        act = (t.get("action") or "").upper()
        try:
            val = float(t.get("value") or 0.0)
        except (TypeError, ValueError):
            val = 0.0
        try:
            q = float(t.get("qty") or 0.0)
        except (TypeError, ValueError):
            q = 0.0
        ts = t.get("timestamp") or None
        if act.startswith("BUY"):
            cost += val
            realized -= val
            buy_qty_total += q
            buy_value_total += val
            if opened_at is None:
                opened_at = ts
            n_trades += 1
        elif act.startswith("SELL"):
            proceeds += val
            realized += val
            closed_at = ts  # last SELL wins by virtue of overwriting
            n_trades += 1

    if buy_qty_total <= 0 or closed_at is None:
        # Defensive: a "round-trip" with no opening BUY (or no closing SELL)
        # shouldn't reach here given the walker's held-≈0 guard, but never
        # emit a malformed row to consumers if it somehow does.
        return None

    avg_cost = buy_value_total / buy_qty_total if buy_qty_total > 1e-9 else 0.0
    realized_pct = (round(realized / cost * 100.0, 2)
                    if cost > 1e-9 else None)
    hold_seconds, hold_days = _hold_duration(opened_at, closed_at)

    return {
        "ticker": ticker,
        "type": ptype,
        "qty": round(buy_qty_total, 6),
        "avg_cost": avg_cost,
        "expiry": expiry,
        "strike": strike,
        "opened_at": opened_at,
        "closed_at": closed_at,
        "realized_pl": round(realized, 2),
        "realized_pl_pct": realized_pct,
        "cost": round(cost, 2),
        "proceeds": round(proceeds, 2),
        "n_trades": n_trades,
        "hold_seconds": hold_seconds,
        "hold_days": hold_days,
    }


def reconcile(positions_closed, trades_derived):
    """Diagnostic: how many round-trips and how much realized $ are hidden
    from ``store.closed_positions`` because of position-row reactivation?

    The trades log is append-only so it sees every completed round-trip; the
    positions table emits only round-trips whose row currently still has
    ``closed_at IS NOT NULL``. Diff = the hidden tail.

    Parameters
    ----------
    positions_closed : list of dict
        Output of ``store.closed_positions(N)``.
    trades_derived : list of dict
        Output of ``derive_round_trips(trades)``.

    Returns
    -------
    dict
        ``{verdict, headline, n_positions_visible, n_trades_derived,
           n_hidden, hidden_realized_usd, hidden_tickers, as_of}``.

        ``verdict``:
          * ``CONSISTENT`` — every trades-derived round-trip has a matching
            closed_positions row (within float tolerance on realized_pl).
          * ``HIDDEN_REALIZED_PL`` — at least one round-trip lives only in
            the trades-derived output; the headline reports the total
            hidden $ and the affected tickers.
          * ``NO_DATA`` — both inputs are empty.
    """
    pc = positions_closed or []
    td = trades_derived or []
    n_pos = sum(1 for r in pc if isinstance(r, dict))
    n_td = sum(1 for r in td if isinstance(r, dict))

    if n_pos == 0 and n_td == 0:
        return {
            "verdict": "NO_DATA",
            "headline": "no closed round-trips observed yet",
            "n_positions_visible": 0,
            "n_trades_derived": 0,
            "n_hidden": 0,
            "hidden_realized_usd": 0.0,
            "hidden_tickers": [],
        }

    # Build a "fingerprint" for each row that's robust to the microsecond
    # gap between the SELL trade's timestamp and the position row's
    # closed_at (set ~µs later by upsert_position's UPDATE). Matching on the
    # exact timestamp would mark every visible lot as also hidden. Instead
    # we key on (ticker, type, expiry, strike, realized_pl_rounded_cents).
    # Two distinct round-trips on the same key with the exact same realized
    # P/L to the cent are vanishingly rare, and both paths compute
    # realized_pl from the same trade values so the cent-rounded equality
    # is byte-identical. ``opened_at`` is also added (truncated to seconds)
    # for the unlikely tie-break — without it, a watchlist that flat-scratch
    # round-trips a ticker twice in a row would collapse to one match.
    def fp(r):
        try:
            pl_cents = round(float(r.get("realized_pl") or 0.0), 2)
        except (TypeError, ValueError):
            pl_cents = 0.0
        opened = (str(r.get("opened_at") or ""))[:19]  # ISO YYYY-MM-DDTHH:MM:SS
        return (
            r.get("ticker") or "",
            r.get("type") or "stock",
            r.get("expiry") or None,
            r.get("strike") if r.get("strike") is not None else None,
            pl_cents,
            opened,
        )

    pos_fps = {fp(r) for r in pc if isinstance(r, dict)}
    hidden = [r for r in td if isinstance(r, dict) and fp(r) not in pos_fps]
    n_hidden = len(hidden)
    hidden_usd = round(
        sum(float(r.get("realized_pl") or 0.0) for r in hidden), 2,
    )
    hidden_tickers = sorted({(r.get("ticker") or "?") for r in hidden})

    if n_hidden == 0:
        verdict = "CONSISTENT"
        headline = (
            f"closed-position table matches trades log "
            f"({n_pos} visible / {n_td} derived)"
        )
    else:
        verdict = "HIDDEN_REALIZED_PL"
        sign = "+" if hidden_usd >= 0 else ""
        tickers_str = ", ".join(hidden_tickers[:5])
        if len(hidden_tickers) > 5:
            tickers_str += f" (+{len(hidden_tickers) - 5} more)"
        headline = (
            f"{n_hidden} round-trip{'s' if n_hidden != 1 else ''} "
            f"({sign}${hidden_usd:.2f}) hidden from /api/closed-positions "
            f"by position-row reactivation — tickers: {tickers_str}"
        )

    return {
        "verdict": verdict,
        "headline": headline,
        "n_positions_visible": n_pos,
        "n_trades_derived": n_td,
        "n_hidden": n_hidden,
        "hidden_realized_usd": hidden_usd,
        "hidden_tickers": hidden_tickers,
    }


def summarize(trades_derived):
    """Roll up the same headline-summary shape ``/api/closed-positions``
    emits, but over the trades-derived round-trips. Pure: never raises."""
    lots = [r for r in (trades_derived or []) if isinstance(r, dict)]
    n = len(lots)
    wins = sum(1 for p in lots if (p.get("realized_pl") or 0) > 0)
    losses = sum(1 for p in lots if (p.get("realized_pl") or 0) < 0)
    total_realized = round(
        sum((p.get("realized_pl") or 0.0) for p in lots), 2,
    )
    total_cost = round(sum((p.get("cost") or 0.0) for p in lots), 2)
    total_proceeds = round(
        sum((p.get("proceeds") or 0.0) for p in lots), 2,
    )
    decided = wins + losses
    win_rate = round(100.0 * wins / decided, 2) if decided else None
    avg_pl_pct = (round(total_realized / total_cost * 100.0, 2)
                  if total_cost > 1e-9 else None)
    return {
        "n": n,
        "wins": wins,
        "losses": losses,
        "flat": n - wins - losses,
        "total_realized_pl": total_realized,
        "total_cost": total_cost,
        "total_proceeds": total_proceeds,
        "win_rate_pct": win_rate,
        "avg_realized_pl_pct": avg_pl_pct,
    }


if __name__ == "__main__":
    import json
    from ..store import get_store
    store = get_store()
    trades = store.recent_trades(limit=5000)
    # recent_trades returns newest-first; derive_round_trips re-sorts per
    # group so the input order doesn't matter, but pass them as-is.
    derived = derive_round_trips(trades)
    closed = store.closed_positions(limit=500)
    recon = reconcile(closed, derived)
    summary = summarize(derived)
    print(json.dumps({
        "reconciliation": recon,
        "summary": summary,
        "n_derived": len(derived),
        "first_3_derived": derived[:3],
    }, indent=2, default=str))

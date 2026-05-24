"""Recycled-ticker P&L — does the bot's re-buy habit actually pay?

``initiation_drought`` surfaces *that* the bot is recycling (9 buys, 3
distinct names, 67% re-cycle rate is the live shape that drove this
analytic to ship). What it does NOT answer: **does the recycling
actually make money?** Is the bot buying NVDA over and over because
each NVDA round-trip prints, or is it scratching the same name 5x for
losses (the classic "trying to make the trade work" failure mode)?

The adjacent endpoints all answer a different question:

  * ``/api/round-trips``        — raw closed round-trip list, no per-name aggregate.
  * ``/api/repeat-loser``       — current losing-streak per ticker only;
                                  never sums up the cumulative P&L cost of
                                  a recycled name, and the prompt_block is
                                  only emitted when a ticker is *currently*
                                  in a loss streak.
  * ``/api/trade-asymmetry``    — book-wide win/loss expectancy; no per-name
                                  decomposition.
  * ``/api/pnl-attribution``    — OPEN-position attribution (β + idiosyncratic),
                                  not closed-round-trip realized P&L.
  * ``/api/per-ticker-skill``   — ML scorer's per-ticker rank-IC; about the
                                  model's predictive skill on the name, not
                                  the trader's realized P&L on it.
  * ``/api/round-trip-postmortem`` — per-recently-closed-trip post-exit move,
                                     not a per-name aggregate.

This module fills the gap. It walks the trade ledger (chronological)
into closed round-trips (reusing the canonical
``analytics.round_trips.build_round_trips`` helper — single source of
truth for the round-trip aggregation), then groups round-trips by
ticker:

  * For each ticker with ≥2 round-trips it's a **recycled** name.
  * For each recycled name we aggregate cost / proceeds / realized P&L /
    win count / total hold days.
  * A per-name verdict:
      - ``PROFITABLE_RECYCLE`` — cumulative realized P&L > +$ ``_PROFITABLE_USD``
                                AND win rate > 50%
      - ``DRAG_RECYCLE``       — cumulative realized P&L < -$ ``_DRAG_USD``
                                OR win rate < ``_DRAG_WIN_RATE`` AND loss
      - ``NEUTRAL_RECYCLE``    — neither
  * An overall verdict:
      - ``WORTH_THE_CHURN``    — net realized P&L across recycled names > +$ ``_PROFITABLE_USD``
      - ``CHURN_DRAG``         — net realized P&L across recycled names < -$ ``_DRAG_USD``
      - ``CHURN_NEUTRAL``      — net is in the ±$ middle band
      - ``NO_RECYCLED_NAMES``  — every ticker has ≤1 round-trip
      - ``NO_DATA``            — no trades / no closed round-trips

Pure builder, never raises. Observational only — never gates Opus, no
caps, no path to ``_execute()`` (AGENTS.md invariants #2/#12).

Run as a CLI for a one-shot ops view::

    python3 -m paper_trader.analytics.recycled_ticker_pnl
"""
from __future__ import annotations

from datetime import datetime, timezone

from .round_trips import build_round_trips

# Verdict thresholds — module-owned so tests read constants.
_PROFITABLE_USD = 1.0   # ≥ $1 net realized = small but real edge
_DRAG_USD = 1.0          # ≤ -$1 net realized = real drag
_DRAG_WIN_RATE = 0.34    # < 1-in-3 wins is a clear loser pattern
_MIN_TRIPS_TO_RECYCLE = 2  # 2 round-trips on a name ⇒ it's been recycled
_INSUFFICIENT_TRIPS_GLOBAL = 2  # below this we say NO_DATA


def _parse_ts(ts):
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        s = ts.replace("Z", "+00:00") if isinstance(ts, str) else ""
        d = datetime.fromisoformat(s)
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _ticker_verdict(net_pnl: float, win_rate: float) -> str:
    """Per-ticker verdict ladder. ``win_rate`` is wins / trips (0..1)."""
    if net_pnl >= _PROFITABLE_USD and win_rate > 0.5:
        return "PROFITABLE_RECYCLE"
    if net_pnl <= -_DRAG_USD or (win_rate < _DRAG_WIN_RATE and net_pnl < 0):
        return "DRAG_RECYCLE"
    return "NEUTRAL_RECYCLE"


def _overall_verdict(net_pnl: float, n_recycled: int) -> str:
    if n_recycled == 0:
        return "NO_RECYCLED_NAMES"
    if net_pnl >= _PROFITABLE_USD:
        return "WORTH_THE_CHURN"
    if net_pnl <= -_DRAG_USD:
        return "CHURN_DRAG"
    return "CHURN_NEUTRAL"


def build_recycled_ticker_pnl(
    trades: list[dict] | None,
    now: datetime | None = None,
) -> dict:
    """Group closed round-trips by ticker and aggregate.

    ``trades`` should be the ``store.recent_trades()`` shape but in
    OLDEST→NEWEST order (``build_round_trips`` walks chronologically).
    Callers reading from ``store.recent_trades`` (newest-first) must
    reverse before passing in — see ``realized_vs_unrealized`` for the
    same idiom. ``now`` is injectable for tests.

    Result shape (always present, never raises):

      * ``state``                   — ``"NO_DATA"`` if no round-trips; ``"OK"`` otherwise.
      * ``as_of``                   — ISO timestamp.
      * ``n_round_trips``           — total closed round-trips seen.
      * ``n_distinct_tickers``      — distinct tickers with ≥1 round-trip.
      * ``n_recycled_tickers``      — tickers with ≥ ``_MIN_TRIPS_TO_RECYCLE`` round-trips.
      * ``net_realized_pnl_usd``    — sum of realized P&L across recycled names only.
      * ``net_realized_pnl_pct``    — net_pnl / sum(cost_recycled) × 100 (None if cost=0).
      * ``recycled_tickers``        — list of per-ticker records (sorted
                                      worst-realized-first then alpha tiebreak).
      * ``one_shot_tickers``        — list of per-ticker records for tickers
                                      with exactly 1 round-trip (sorted
                                      worst-pnl-first); kept compact and capped.
      * ``verdict``                 — overall verdict (see ladder above).
      * ``verdict_detail``          — one-line explanation.
      * ``headline``                — dashboard one-liner.
      * ``thresholds``              — module constants for UI.

    Per-ticker record shape:

      * ``ticker``                  — uppercase symbol.
      * ``n_round_trips``           — count.
      * ``n_wins`` / ``n_losses`` / ``n_washes`` — outcome split
        (loss = pnl < -$0.50; win = pnl > +$0.50; wash otherwise — the
        same dollar threshold trade_asymmetry uses for its decided count).
      * ``win_rate``                — wins / n_round_trips (0..1).
      * ``total_cost_usd``          — sum of gross BUY value across trips.
      * ``total_proceeds_usd``      — sum of gross SELL value across trips.
      * ``realized_pnl_usd``        — total_proceeds - total_cost.
      * ``realized_pnl_pct``        — realized_pnl / total_cost × 100 (None if cost=0).
      * ``avg_pnl_per_trip_usd``    — mean per-trip P&L.
      * ``best_trip_usd`` / ``worst_trip_usd`` — extremes per name.
      * ``avg_hold_days``           — mean per-trip hold (excludes None values).
      * ``first_entry_ts`` / ``last_exit_ts`` — bookends of the recycling history.
      * ``verdict``                 — per-ticker verdict.
    """
    now = now or datetime.now(timezone.utc)
    out = {
        "as_of": now.isoformat(timespec="seconds"),
        "state": "NO_DATA",
        "n_round_trips": 0,
        "n_distinct_tickers": 0,
        "n_recycled_tickers": 0,
        "net_realized_pnl_usd": 0.0,
        "net_realized_pnl_pct": None,
        "recycled_tickers": [],
        "one_shot_tickers": [],
        "verdict": "NO_DATA",
        "verdict_detail": "no closed round-trips on record — cannot grade recycling",
        "headline": "no closed round-trips — cannot grade recycling cost",
        "thresholds": {
            "profitable_usd": _PROFITABLE_USD,
            "drag_usd": _DRAG_USD,
            "drag_win_rate": _DRAG_WIN_RATE,
            "min_trips_to_recycle": _MIN_TRIPS_TO_RECYCLE,
            "wash_band_usd": 0.50,
        },
    }
    rows = list(trades or [])
    if not rows:
        return out

    # ``build_round_trips`` walks chronologically; pass-through. It raises
    # on non-numeric ``qty``/``price``/``value`` cells (a real upstream
    # store would never write those, but the never-raises contract on
    # this module is load-bearing for the dashboard endpoint). Degrade to
    # NO_DATA on the rare malformed input rather than 500.
    try:
        trips = build_round_trips(rows)
    except Exception:
        out["verdict_detail"] = (
            "malformed trade rows — could not aggregate round-trips"
        )
        out["headline"] = (
            "NO_DATA — malformed trade rows could not be aggregated"
        )
        return out
    out["n_round_trips"] = len(trips)
    if len(trips) < _INSUFFICIENT_TRIPS_GLOBAL:
        out["state"] = "OK" if len(trips) > 0 else "NO_DATA"
        out["verdict"] = "NO_DATA" if len(trips) == 0 else "NO_RECYCLED_NAMES"
        if len(trips) == 0:
            out["verdict_detail"] = "no closed round-trips on record — cannot grade recycling"
            out["headline"] = "no closed round-trips — cannot grade recycling cost"
        else:
            tk = (trips[0].get("ticker") or "?").upper()
            out["n_distinct_tickers"] = 1
            out["verdict_detail"] = (
                f"only 1 closed round-trip on record ({tk}) — no ticker has been recycled yet"
            )
            out["headline"] = (
                f"NO_RECYCLED_NAMES — 1 closed round-trip ({tk}); no recycling history"
            )
        return out

    # Group round-trips by uppercase ticker. The (type, strike, expiry)
    # discriminator stays inside the trip records; for the recycling
    # question we want everything that ever touched the symbol — equity
    # AND options round-trips on NVDA both count as "NVDA recycling".
    per_ticker: dict[str, list[dict]] = {}
    for trip in trips:
        sym = (trip.get("ticker") or "").strip().upper()
        if not sym:
            continue
        per_ticker.setdefault(sym, []).append(trip)

    out["n_distinct_tickers"] = len(per_ticker)
    recycled_records: list[dict] = []
    one_shot_records: list[dict] = []

    for sym, trip_list in per_ticker.items():
        n = len(trip_list)
        total_cost = sum(float(t.get("cost") or 0.0) for t in trip_list)
        total_proceeds = sum(float(t.get("proceeds") or 0.0) for t in trip_list)
        pnl = total_proceeds - total_cost
        pnl_pct = (round(pnl / total_cost * 100.0, 4)
                   if total_cost > 1e-9 else None)
        # Wash band mirrors trade_asymmetry's "decided" rule: |pnl| ≤ $0.50.
        wins = sum(1 for t in trip_list if (t.get("pnl_usd") or 0.0) > 0.50)
        losses = sum(1 for t in trip_list if (t.get("pnl_usd") or 0.0) < -0.50)
        washes = n - wins - losses
        win_rate = wins / n if n else 0.0
        per_trip_pnls = [float(t.get("pnl_usd") or 0.0) for t in trip_list]
        best = max(per_trip_pnls) if per_trip_pnls else 0.0
        worst = min(per_trip_pnls) if per_trip_pnls else 0.0
        avg_pnl = sum(per_trip_pnls) / n if n else 0.0
        hold_days = [float(t.get("hold_days")) for t in trip_list
                     if t.get("hold_days") is not None]
        avg_hold = (round(sum(hold_days) / len(hold_days), 4)
                    if hold_days else None)
        # Bookends — first entry across all trips, last exit across all trips.
        first_entry = None
        last_exit = None
        for t in trip_list:
            ets = _parse_ts(t.get("entry_ts"))
            xts = _parse_ts(t.get("exit_ts"))
            if ets is not None and (first_entry is None or ets < first_entry[0]):
                first_entry = (ets, t.get("entry_ts"))
            if xts is not None and (last_exit is None or xts > last_exit[0]):
                last_exit = (xts, t.get("exit_ts"))

        rec = {
            "ticker": sym,
            "n_round_trips": n,
            "n_wins": wins,
            "n_losses": losses,
            "n_washes": washes,
            "win_rate": round(win_rate, 4),
            "total_cost_usd": round(total_cost, 4),
            "total_proceeds_usd": round(total_proceeds, 4),
            "realized_pnl_usd": round(pnl, 4),
            "realized_pnl_pct": pnl_pct,
            "avg_pnl_per_trip_usd": round(avg_pnl, 4),
            "best_trip_usd": round(best, 4),
            "worst_trip_usd": round(worst, 4),
            "avg_hold_days": avg_hold,
            "first_entry_ts": first_entry[1] if first_entry else None,
            "last_exit_ts": last_exit[1] if last_exit else None,
            "verdict": _ticker_verdict(pnl, win_rate) if n >= _MIN_TRIPS_TO_RECYCLE else "ONE_SHOT",
        }
        if n >= _MIN_TRIPS_TO_RECYCLE:
            recycled_records.append(rec)
        else:
            one_shot_records.append(rec)

    # Sort: recycled worst-realized-first (most drag at top), alpha tiebreak.
    recycled_records.sort(key=lambda r: (r["realized_pnl_usd"], r["ticker"]))
    # One-shots similarly — worst-pnl first; cap to keep payload bounded.
    one_shot_records.sort(key=lambda r: (r["realized_pnl_usd"], r["ticker"]))
    one_shot_records = one_shot_records[:20]

    out["recycled_tickers"] = recycled_records
    out["one_shot_tickers"] = one_shot_records
    out["n_recycled_tickers"] = len(recycled_records)

    net_recycled_pnl = sum(r["realized_pnl_usd"] for r in recycled_records)
    net_recycled_cost = sum(r["total_cost_usd"] for r in recycled_records)
    out["net_realized_pnl_usd"] = round(net_recycled_pnl, 4)
    out["net_realized_pnl_pct"] = (
        round(net_recycled_pnl / net_recycled_cost * 100.0, 4)
        if net_recycled_cost > 1e-9 else None)

    verdict = _overall_verdict(net_recycled_pnl, len(recycled_records))
    out["verdict"] = verdict
    out["state"] = "OK"

    # Headline + detail composition.
    if verdict == "NO_RECYCLED_NAMES":
        out["verdict_detail"] = (
            f"{len(trips)} closed round-trip(s) across {len(per_ticker)} "
            f"ticker(s); none has been recycled yet"
        )
        out["headline"] = (
            f"NO_RECYCLED_NAMES — {len(trips)} round-trip(s), "
            f"{len(per_ticker)} ticker(s); no recycling history"
        )
    else:
        worst = recycled_records[0]
        best = recycled_records[-1]
        if verdict == "WORTH_THE_CHURN":
            out["verdict_detail"] = (
                f"{len(recycled_records)} recycled name(s) net "
                f"${net_recycled_pnl:+.2f} realized — recycling has paid; "
                f"best {best['ticker']} ${best['realized_pnl_usd']:+.2f} "
                f"({best['n_round_trips']}x)"
            )
        elif verdict == "CHURN_DRAG":
            out["verdict_detail"] = (
                f"{len(recycled_records)} recycled name(s) net "
                f"${net_recycled_pnl:+.2f} realized — recycling is a drag; "
                f"worst {worst['ticker']} ${worst['realized_pnl_usd']:+.2f} "
                f"({worst['n_round_trips']}x)"
            )
        else:  # CHURN_NEUTRAL
            out["verdict_detail"] = (
                f"{len(recycled_records)} recycled name(s) net "
                f"${net_recycled_pnl:+.2f} realized — recycling is roughly flat"
            )
        out["headline"] = (
            f"{verdict} — {len(recycled_records)} recycled name(s), "
            f"net ${net_recycled_pnl:+.2f}"
        )

    return out


def _main() -> int:
    """One-shot CLI — read live trades and print JSON."""
    import json
    from ..store import get_store
    # Store returns newest-first; build_round_trips wants chronological.
    trades = list(reversed(get_store().recent_trades(limit=5000)))
    print(json.dumps(build_recycled_ticker_pnl(trades), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

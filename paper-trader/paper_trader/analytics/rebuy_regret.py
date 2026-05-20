"""Sell-then-rebuy **$ regret** quantifier — did closing early cost or save money?

``round_trips.build_round_trips`` groups raw trades into closed round-trips.
``reentry_velocity`` tracks the *time* gap between close and re-buy.
``churn`` measures size-weighted turnover. ``hold_discipline`` reports hold
duration. None of them answer the discretionary trader's hardest exit
question:

  **"When I sold a name and bought it back later, did I save money or lose
  money in the round-trip-to-re-entry hop?"**

The discriminating disagreement vs ``reentry_velocity``: a fast re-entry is
not inherently bad — what matters is the *price delta* over the close→re-buy
gap. Selling NVDA at $220 and re-buying 2h later at $218 *saved* money
(timing edge); selling NVDA at $220 and re-buying 2h later at $223 *cost*
money (whipsaw). Existing surfaces measure WHEN, not COST.

``build_rebuy_regret`` is pure — composes ``build_round_trips`` (single
source of truth, AGENTS.md #10) and walks each (ticker, type, strike,
expiry) key's exits to the next same-key entry, computing the price delta
against the shared quantity ``min(sell_qty, rebuy_qty)``. The endpoint
owns I/O (the documented ``round_trips`` / ``reentry_velocity`` builder
split). Observational only — never gates Opus, never injected into the
decision prompt, no caps (AGENTS.md #2 / #12 — the ``reentry_velocity``
precedent).

Sign convention: ``regret_usd > 0`` means **lost** money (sold low,
bought back higher). ``regret_usd < 0`` means **saved** money (sold high,
bought back lower). The headline uses words ``REGRET`` / ``SAVED`` instead
of leaving the operator to interpret a signed number.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .round_trips import build_round_trips


# Verdict ladder thresholds. Pinned in tests — adjust both sides together.
# A net |regret| below the noise floor is NEUTRAL (rounding noise dominates).
_NEUTRAL_USD_FLOOR = 0.50

# Per-event severity ladder so a single big regret doesn't get drowned in
# many small ones in the headline.
_MATERIAL_USD = 5.0


def _parse_ts(ts):
    """Best-effort ISO → tz-aware datetime; ``None`` on garbage."""
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _f(x, default=0.0):
    """Best-effort float coercion; degrade rather than raise."""
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def _classify(regret_usd: float) -> str:
    """One event's verdict. ``SAVED`` = negative regret (good)."""
    if regret_usd > _MATERIAL_USD:
        return "REGRET_HIGH"
    if regret_usd > _NEUTRAL_USD_FLOOR:
        return "REGRET"
    if regret_usd < -_MATERIAL_USD:
        return "SAVED_HIGH"
    if regret_usd < -_NEUTRAL_USD_FLOOR:
        return "SAVED"
    return "NEUTRAL"


def _find_next_entry(trades_sorted: list[dict], after_idx: int,
                     key: tuple) -> tuple[int | None, dict | None]:
    """First subsequent BUY on the same (ticker, type, strike, expiry) key.

    Returns ``(index, trade)`` or ``(None, None)`` if no subsequent entry.
    ``trades_sorted`` must be oldest→newest.
    """
    for j in range(after_idx + 1, len(trades_sorted)):
        t = trades_sorted[j]
        typ = t.get("option_type") or "stock"
        tkey = (t.get("ticker"), typ, t.get("strike"), t.get("expiry"))
        if tkey != key:
            continue
        action = (t.get("action") or "").upper()
        if action.startswith("BUY"):
            return j, t
    return None, None


def build_rebuy_regret(trades: list[dict], now: datetime | None = None,
                       recent_limit: int = 10) -> dict:
    """Compose per-key close→re-buy price-delta regret events.

    Inputs:
      trades — list shaped like ``Store.recent_trades()`` (``timestamp``,
        ``ticker``, ``action``, ``qty``, ``price``, ``value``, ``strike``,
        ``expiry``, ``option_type``, ``id``). Direction-tolerant — sorted
        internally to oldest→newest.
      now — injected clock (defaults to UTC now); cosmetic only (the
        ``as_of`` stamp).
      recent_limit — newest-first cap on ``recent_events`` (default 10,
        clamped 1..100).

    Returns a JSON-ready dict:
      ``as_of`` (ISO seconds), ``state`` (NO_DATA / NO_REBUYS / OK),
      ``verdict`` (SAVINGS / NET_NEUTRAL / REGRETTING),
      ``n_round_trips``, ``n_events``, ``total_regret_usd`` (positive=lost),
      ``net_regret_usd`` (= total_regret_usd, alias for headline clarity),
      ``median_regret_usd``, ``worst_regret_usd``, ``best_savings_usd``,
      ``regret_event_count``, ``saved_event_count``, ``neutral_event_count``,
      ``recent_events`` (newest first: ticker/type/strike/expiry/sold_at/
      sold_price/rebought_at/rebought_price/shared_qty/price_delta/regret_usd/
      classification/gap_hours),
      ``per_ticker`` (per-ticker summary: ticker/n_events/net_regret_usd/
      worst_regret_usd/best_savings_usd/last_classification),
      ``headline``.

    Pure — no DB, no network. ``trades`` may be empty; the function
    returns NO_DATA rather than raising. Option round-trips factor the
    ×100 multiplier (regret_usd is in **real** dollars, the
    ``round_trips`` precedent).
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    recent_limit = max(1, min(int(recent_limit or 10), 100))

    # Defensive sort to oldest→newest so the caller can pass either direction.
    def _sort_key(t):
        if not isinstance(t, dict):
            return datetime.min.replace(tzinfo=timezone.utc)
        p = _parse_ts(t.get("timestamp"))
        return p or datetime.min.replace(tzinfo=timezone.utc)

    # Defensive: build_round_trips upstream is NOT robust to garbage rows
    # (it does ``t["ticker"]`` raw, raising KeyError on a row missing the
    # key). We pre-filter to dicts with both ``ticker`` and a parseable
    # timestamp so the _safe contract holds at this boundary.
    sorted_trades = sorted(
        [t for t in (trades or [])
         if isinstance(t, dict) and t.get("ticker") is not None
         and _parse_ts(t.get("timestamp")) is not None],
        key=_sort_key,
    )
    if not sorted_trades:
        return {
            "as_of": now.isoformat(timespec="seconds"),
            "state": "NO_DATA",
            "verdict": "NO_DATA",
            "n_round_trips": 0, "n_events": 0,
            "total_regret_usd": 0.0, "net_regret_usd": 0.0,
            "median_regret_usd": None, "worst_regret_usd": None,
            "best_savings_usd": None,
            "regret_event_count": 0, "saved_event_count": 0,
            "neutral_event_count": 0,
            "recent_events": [], "per_ticker": [],
            "headline": "no trades yet — no rebuy regret to measure.",
        }

    # Use round_trips as the SSOT to enumerate closed round-trips. For each
    # round-trip's *exit* trade, find the next same-key BUY in the trade
    # stream — that's the "re-buy". The price delta against shared quantity
    # is the realized regret (positive = lost money on the round-trip).
    rts = build_round_trips(sorted_trades)
    n_round_trips = len(rts)

    # Index trades by their id for O(1) lookup of the exit trade.
    trades_by_id = {t.get("id"): (i, t) for i, t in enumerate(sorted_trades)
                    if t.get("id") is not None}

    events: list[dict] = []
    for rt in rts:
        key = (rt["ticker"], rt["type"], rt.get("strike"), rt.get("expiry"))
        # Take the *last* exit trade id (the one that closed the round-trip)
        # — that's where we measure the close price from.
        exit_ids = rt.get("exit_trade_ids") or []
        if not exit_ids:
            continue
        last_exit_id = exit_ids[-1]
        if last_exit_id not in trades_by_id:
            continue
        exit_idx, exit_trade = trades_by_id[last_exit_id]
        sell_price = _f(exit_trade.get("price"))
        sell_qty = _f(exit_trade.get("qty"))
        if sell_price <= 0 or sell_qty <= 0:
            continue

        next_idx, next_buy = _find_next_entry(sorted_trades, exit_idx, key)
        if next_buy is None:
            continue
        rebuy_price = _f(next_buy.get("price"))
        rebuy_qty = _f(next_buy.get("qty"))
        if rebuy_price <= 0 or rebuy_qty <= 0:
            continue

        # Option multiplier: ×100 (the round_trips precedent — value already
        # bakes this in but we work from price here).
        mult = 100 if (rt["type"] in ("call", "put")) else 1
        shared_qty = min(sell_qty, rebuy_qty)
        # Sign: positive regret = bought back higher than we sold = lost
        # money on the round-trip-to-re-entry hop.
        price_delta = rebuy_price - sell_price
        regret_usd = price_delta * shared_qty * mult

        closed_at = _parse_ts(rt.get("exit_ts"))
        rebought_at = _parse_ts(next_buy.get("timestamp"))
        gap_h = None
        if closed_at and rebought_at:
            dh = (rebought_at - closed_at).total_seconds() / 3600.0
            if dh >= 0:
                gap_h = round(dh, 4)

        events.append({
            "ticker": key[0],
            "type": key[1],
            "strike": key[2],
            "expiry": key[3],
            "sold_at": rt.get("exit_ts"),
            "sold_price": round(sell_price, 4),
            "sold_qty": round(sell_qty, 6),
            "rebought_at": next_buy.get("timestamp"),
            "rebought_price": round(rebuy_price, 4),
            "rebought_qty": round(rebuy_qty, 6),
            "shared_qty": round(shared_qty, 6),
            "price_delta": round(price_delta, 4),
            "regret_usd": round(regret_usd, 4),
            "classification": _classify(regret_usd),
            "gap_hours": gap_h,
            "rt_pnl_usd": rt.get("pnl_usd"),
        })

    n_events = len(events)
    if n_events == 0:
        return {
            "as_of": now.isoformat(timespec="seconds"),
            "state": "NO_REBUYS",
            "verdict": "NO_REBUYS",
            "n_round_trips": n_round_trips, "n_events": 0,
            "total_regret_usd": 0.0, "net_regret_usd": 0.0,
            "median_regret_usd": None, "worst_regret_usd": None,
            "best_savings_usd": None,
            "regret_event_count": 0, "saved_event_count": 0,
            "neutral_event_count": 0,
            "recent_events": [], "per_ticker": [],
            "headline": (f"{n_round_trips} closed round-trip(s) but no "
                         f"re-entries yet — no regret/savings to measure."),
        }

    regret_values = [e["regret_usd"] for e in events]
    total_regret = sum(regret_values)
    median_regret = _median(regret_values)
    worst_regret = max(regret_values)  # most positive = biggest loss
    best_savings = min(regret_values)  # most negative = biggest save

    regret_count = sum(1 for e in events
                       if e["classification"] in ("REGRET", "REGRET_HIGH"))
    saved_count = sum(1 for e in events
                      if e["classification"] in ("SAVED", "SAVED_HIGH"))
    neutral_count = sum(1 for e in events
                        if e["classification"] == "NEUTRAL")

    # Per-ticker rollup.
    per_ticker_acc: dict[str, list[dict]] = {}
    for e in events:
        per_ticker_acc.setdefault(e["ticker"], []).append(e)

    per_ticker: list[dict] = []
    for tkr, rows in per_ticker_acc.items():
        rows_newest = sorted(
            rows, key=lambda r: _parse_ts(r["rebought_at"])
            or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        regrets = [r["regret_usd"] for r in rows]
        per_ticker.append({
            "ticker": tkr,
            "n_events": len(rows),
            "net_regret_usd": round(sum(regrets), 4),
            "worst_regret_usd": round(max(regrets), 4),
            "best_savings_usd": round(min(regrets), 4),
            "last_classification": rows_newest[0]["classification"],
            "last_regret_usd": rows_newest[0]["regret_usd"],
        })
    # Worst net offender first so the operator's eye lands on it.
    per_ticker.sort(key=lambda p: -p["net_regret_usd"])

    # Sort events newest first for the recent_events slice.
    events.sort(
        key=lambda e: _parse_ts(e["rebought_at"])
        or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )

    # Top-level verdict: signed net vs the noise floor.
    if total_regret > _NEUTRAL_USD_FLOOR:
        verdict = "REGRETTING"
    elif total_regret < -_NEUTRAL_USD_FLOOR:
        verdict = "SAVINGS"
    else:
        verdict = "NET_NEUTRAL"

    if verdict == "REGRETTING":
        # Name the worst single event in the headline.
        worst = max(events, key=lambda e: e["regret_usd"])
        headline = (
            f"Net REGRET ${total_regret:+.2f} across {n_events} re-entry "
            f"event(s). Worst: {worst['ticker']} "
            f"${worst['regret_usd']:+.2f} (sold ${worst['sold_price']} → "
            f"re-bought ${worst['rebought_price']})."
        )
    elif verdict == "SAVINGS":
        best = min(events, key=lambda e: e["regret_usd"])
        headline = (
            f"Net SAVINGS ${-total_regret:+.2f} across {n_events} re-entry "
            f"event(s). Best: {best['ticker']} "
            f"${-best['regret_usd']:+.2f} (sold ${best['sold_price']} → "
            f"re-bought ${best['rebought_price']})."
        )
    else:
        headline = (
            f"Net ≈$0 across {n_events} re-entry event(s) "
            f"({regret_count} regret, {saved_count} saved, "
            f"{neutral_count} neutral)."
        )

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "state": "OK",
        "verdict": verdict,
        "n_round_trips": n_round_trips,
        "n_events": n_events,
        "total_regret_usd": round(total_regret, 4),
        "net_regret_usd": round(total_regret, 4),
        "median_regret_usd": round(median_regret, 4)
        if median_regret is not None else None,
        "worst_regret_usd": round(worst_regret, 4),
        "best_savings_usd": round(best_savings, 4),
        "regret_event_count": regret_count,
        "saved_event_count": saved_count,
        "neutral_event_count": neutral_count,
        "recent_events": events[:recent_limit],
        "per_ticker": per_ticker,
        "headline": headline,
    }

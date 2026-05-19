"""Options-market-implied move for held positions with imminent earnings.

``/api/earnings-shock`` reports a **historical-Gaussian σ** (n=8 prior prints
fed through a population-stdev calc). ``/api/earnings-distribution`` reports
the **empirical observed quartiles** of those same prints. Both are looking
*backward* at what the stock has done after past prints. Neither answers the
discretionary trader's most-asked question the morning of an earnings event:

    *"What is the market currently pricing as the move on this print?"*

The desk-standard answer is the **ATM straddle**. For an event-expiry
option chain, ``(call_mid + put_mid) / spot`` is the breakeven move the
market is implying — i.e. the size of the move at which a long-straddle
buyer breaks even. Empirically this is what every discretionary trader
glances at pre-earnings to size their conviction against what the options
market expects (which is itself a forward-looking, real-money consensus).

This is the **forward, market-priced complement** to ``earnings_shock``'s
backward-looking σ and ``earnings_distribution``'s backward-looking
quartiles. The three together form the desk's "what's priced vs what
historically happened" pre-earnings frame.

Composes ``build_event_calendar``'s ``events`` list **verbatim** (single
source of truth, AGENTS.md #10): the held set, ``days_away``, and
``tier`` come from the canonical event-calendar verdict so this builder
and ``/api/event-calendar`` / ``/api/earnings-shock`` /
``/api/earnings-distribution`` can never disagree on what counts as
held-imminent. For each held event the ``options_provider`` callable
returns a chain dict (the ``market.get_options_chain`` shape:
``{"expiry": "YYYY-MM-DD", "calls": [...], "puts": [...]}``); the builder
is otherwise pure: no I/O, no yfinance, never raises (the endpoint owns
the I/O — the ``earnings_shock`` / ``tail_risk`` builder/endpoint split).

State ladder mirrors the sibling builders:

* ``NO_DATA``   — empty book / unpriceable / no total_value (the
  ``earnings_shock`` "no priced book to shock yet" precedent).
* ``NO_EVENTS`` — book is fine, calendar is fine, but no held name has an
  imminent print inside the horizon (the most common branch on a typical
  desk; intentionally distinct from NO_DATA so the operator can tell
  *"calendar quiet"* from *"book empty"*).
* ``OK``        — at least one held imminent event with implied-move
  numerics emitted.

Per-event sample-size honesty (the ``earnings_shock`` precedent):

* ``OK`` — both ATM call and put have a real bid/ask (or last) so the
  straddle mid is defensible.
* ``NO_QUOTES`` — the chain came back but the ATM strikes have no bid /
  no ask / zero last-price; we surface the row (the operator never misses
  *"NVDA reports tomorrow"*) but withhold the implied figure rather than
  fabricate it from a stale lastPrice.
* ``NO_CHAIN`` — the provider returned ``None`` (yfinance miss, ticker
  delisted, no listed options). Same honesty contract.

ATM strike pick: the call with min ``|strike − spot|``; independently the
put with min ``|strike − spot|`` (in practice the same strike for a
normal listed chain, but the *call* table and the *put* table are picked
independently so a one-sided gap doesn't silently mis-pair). ``spot``
falls back ``current_price → avg_cost`` (the ``stress_scenarios``
``_position_value`` discipline).

Quote convention: ``mid = (bid + ask) / 2`` when both > 0, else ``last``
when > 0, else ``None``. Wide-spread / one-sided quotes are honest:
spread is **not** silently averaged from a stale last when the live
bid/ask is missing.

Vol convention: ``implied_move_pct = (call_mid + put_mid) / spot * 100``
— this is the **breakeven move**, the desk-standard ATM-straddle
shorthand. For a strict mathematical 1σ Black-Scholes-implied move
(``≈ 0.8 × straddle / spot``) the per-strike ``impliedVolatility`` is
also surfaced as ``iv_atm`` so the operator can reconcile both numbers
without re-querying yfinance.

Observational / advisory only — never gates Opus, never injected into
the decision prompt, no caps (invariants #2/#12 — the ``earnings_shock``
/ ``stress_scenarios`` / ``recovery`` precedent).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

DEFAULT_HORIZON_DAYS = 7.0     # only events ≤ this many days away are scored
ELEVATED_BOOK_PCT = 5.0        # |implied| book impact ≥ this ⇒ ELEVATED tier
MODERATE_BOOK_PCT = 2.0        # |implied| book impact ≥ this ⇒ MODERATE tier


def _z(v: float | None, ndigits: int = 2) -> float | None:
    """Round; fold -0.0 → 0.0 (the ``earnings_shock._z`` precedent — same
    shape, same contract). A non-numeric / None input degrades to ``None``,
    never raises."""
    if v is None:
        return None
    try:
        r = round(float(v), ndigits)
    except (TypeError, ValueError):
        return None
    return 0.0 if r == 0 else r


def _safe_float(x) -> float | None:
    """Best-effort coercion to ``float`` with ``NaN`` rejected.

    yfinance returns ``NaN`` for missing bid/ask cells in a thin chain;
    a naive ``float(NaN)`` is numeric but propagates through ``(b+a)/2``
    silently. ``x != x`` is the canonical NaN check (NaN is the only
    float not equal to itself) — reject it here so a one-sided NaN
    quote degrades the row to ``NO_QUOTES`` honestly instead of
    pricing a straddle off a half-NaN sum."""
    try:
        if x is None:
            return None
        f = float(x)
        if f != f:  # NaN
            return None
        return f
    except (TypeError, ValueError):
        return None


def _mid_or_last(row: dict) -> float | None:
    """``(bid+ask)/2`` when both > 0, else ``last`` when > 0, else ``None``.

    A wide-spread / one-sided quote (only bid or only ask) is treated as
    no quote — silently averaging against zero would halve every implied
    move on a thin chain. ``lastPrice`` is the documented secondary
    fallback (the ``market.get_option_price`` precedent — same shape)."""
    if not isinstance(row, dict):
        return None
    b = _safe_float(row.get("bid"))
    a = _safe_float(row.get("ask"))
    if b is not None and a is not None and b > 0 and a > 0:
        return (b + a) / 2.0
    last = _safe_float(row.get("lastPrice"))
    return last if (last is not None and last > 0) else None


def _atm_row(side: list, spot: float) -> dict | None:
    """The row in ``side`` (calls or puts) with the strike closest to ``spot``.

    Returns ``None`` on empty/garbage input — never raises. Equidistant
    strikes resolve to the first by ``min(..., key=)``; in a normal listed
    chain this is deterministic and matches the desk's ATM pick."""
    if not isinstance(side, list) or not side:
        return None
    if not isinstance(spot, (int, float)) or spot <= 0:
        return None
    best = None
    best_d = None
    for r in side:
        if not isinstance(r, dict):
            continue
        k = _safe_float(r.get("strike"))
        if k is None or k <= 0:
            continue
        d = abs(k - spot)
        if best_d is None or d < best_d:
            best, best_d = r, d
    return best


def _position_value(position: dict) -> float | None:
    """Match ``earnings_shock._position_value`` semantics: options ×100,
    price falls back avg_cost. ``None`` on a fully-unpriceable row so the
    caller can skip rather than dollarize against zero."""
    if not isinstance(position, dict):
        return None
    try:
        ptype = position.get("type") or "stock"
        mult = 100 if ptype in ("call", "put") else 1
        price = position.get("current_price") or position.get("avg_cost") or 0.0
        qty = float(position.get("qty") or 0)
        val = float(price) * qty * mult
    except (TypeError, ValueError):
        return None
    return val if val > 0 else None


def _position_spot(position: dict) -> float | None:
    """Per-name underlying spot for the ATM pick: ``current_price`` → ``avg_cost``.

    For an *option* position the ``current_price`` is the option premium,
    not the underlying spot — in that case we have no clean spot here so
    we return ``None`` (the row degrades to ``NO_CHAIN``/``NO_QUOTES``
    honestly; an option-on-an-earnings-name implied-move read is its own
    feature, not this one). Stocks: spot is ``current_price`` or fall
    back to ``avg_cost``."""
    if not isinstance(position, dict):
        return None
    ptype = (position.get("type") or "stock").lower()
    if ptype in ("call", "put"):
        return None
    px = _safe_float(position.get("current_price"))
    if px is not None and px > 0:
        return px
    px = _safe_float(position.get("avg_cost"))
    return px if (px is not None and px > 0) else None


def _row_verdict(book_pct: float | None) -> str:
    """ELEVATED/MODERATE/LOW classification of the |implied| book impact."""
    if book_pct is None:
        return "UNKNOWN"
    a = abs(book_pct)
    if a >= ELEVATED_BOOK_PCT:
        return "ELEVATED"
    if a >= MODERATE_BOOK_PCT:
        return "MODERATE"
    return "LOW"


def build_implied_move(
    positions: list[dict],
    total_value: float,
    event_calendar_result: dict | None,
    options_provider: Callable[[str, int], dict | None] | None,
    now: datetime | None = None,
    horizon_days: float = DEFAULT_HORIZON_DAYS,
) -> dict:
    """Pure: no I/O, never raises. ``positions`` is the
    ``store.open_positions()`` shape (or the strategy snapshot — both
    carry ``ticker``/``qty``/``current_price``/``avg_cost``/``type``).
    ``event_calendar_result`` is the dict from ``build_event_calendar``
    (its ``events`` list provides held + days_away). ``options_provider``
    is the I/O seam: given ``(ticker, target_dte_days)`` returns a chain
    dict with ``expiry``/``calls``/``puts`` keys (the
    ``market.get_options_chain`` shape). Pass ``None`` to skip chain
    lookup entirely (all rows read ``NO_CHAIN``)."""
    now = now or datetime.now(timezone.utc)
    try:
        tv = float(total_value or 0.0)
    except (TypeError, ValueError):
        tv = 0.0

    base = {
        "as_of": now.isoformat(timespec="seconds"),
        "horizon_days": float(horizon_days),
        "n_events": 0,
        "events": [],
        "total_implied_book_pct": None,
        "headline": None,
        "verdict": None,
    }

    # Value- and spot-index the held book so we can dollarize each event.
    value_by_ticker: dict[str, float] = {}
    spot_by_ticker: dict[str, float] = {}
    for p in positions or []:
        try:
            tk = (p.get("ticker") or "").upper()
        except (TypeError, AttributeError):
            continue
        if not tk:
            continue
        v = _position_value(p)
        if v is None:
            continue
        value_by_ticker[tk] = value_by_ticker.get(tk, 0.0) + v
        sp = _position_spot(p)
        if sp is not None and tk not in spot_by_ticker:
            spot_by_ticker[tk] = sp

    if not value_by_ticker or tv <= 0:
        base["state"] = "NO_DATA"
        base["headline"] = "Implied move: no priced book to score yet."
        return base

    cal_events: list[dict] = []
    if isinstance(event_calendar_result, dict):
        try:
            cal_events = list(event_calendar_result.get("events") or [])
        except (TypeError, AttributeError):
            cal_events = []

    rows: list[dict] = []
    for ev in cal_events:
        if not isinstance(ev, dict):
            continue
        try:
            tk = (ev.get("ticker") or "").upper()
        except (TypeError, AttributeError):
            continue
        if not tk or tk not in value_by_ticker:
            continue
        try:
            days_away = float(ev.get("days_away"))
        except (TypeError, ValueError):
            continue
        # Only score events inside the horizon. A "held but distant" event
        # is honest awareness (event_calendar emits it) but not an implied-
        # move candidate — a far-DTE chain dilutes implied with non-event
        # noise. The ``earnings_shock`` precedent.
        if days_away < 0 or days_away > horizon_days:
            continue

        position_value = value_by_ticker[tk]
        pct_of_book = position_value / tv * 100.0
        spot = spot_by_ticker.get(tk)

        # Pick the option chain expiry that *captures* the event: at least
        # 1 DTE so a today-AMC event isn't priced off a same-day expiry
        # that has already settled (the ``get_options_chain`` target_dte
        # is approximate and yfinance returns the nearest listed expiry).
        target_dte = max(1, int(round(days_away)))
        row: dict = {
            "ticker": tk,
            "days_to_earnings": _z(days_away),
            "earnings_date": ev.get("earnings_date"),
            "tier": ev.get("tier"),
            "current_value_usd": _z(position_value),
            "weight_pct": _z(pct_of_book),
            "spot": _z(spot),
            "expiry": None,
            "atm_call_strike": None,
            "atm_put_strike": None,
            "call_mid": None,
            "put_mid": None,
            "straddle": None,
            "implied_move_pct": None,
            "implied_one_sigma_pct": None,
            "iv_atm": None,
            "implied_dollar_move": None,
            "implied_book_pct": None,
            "row_verdict": "UNKNOWN",
        }

        chain = None
        if options_provider is not None and spot is not None:
            try:
                chain = options_provider(tk, target_dte)
            except Exception:
                chain = None

        if not isinstance(chain, dict) or spot is None:
            row["state"] = "NO_CHAIN"
            row["headline"] = (
                f"{tk}: earnings in {days_away:.1f}d — options chain unavailable, "
                "implied move withheld."
            )
            rows.append(row)
            continue

        row["expiry"] = chain.get("expiry")
        call = _atm_row(chain.get("calls") or [], spot)
        put = _atm_row(chain.get("puts") or [], spot)
        call_mid = _mid_or_last(call) if call is not None else None
        put_mid = _mid_or_last(put) if put is not None else None
        if call is not None:
            row["atm_call_strike"] = _z(_safe_float(call.get("strike")))
        if put is not None:
            row["atm_put_strike"] = _z(_safe_float(put.get("strike")))
        row["call_mid"] = _z(call_mid)
        row["put_mid"] = _z(put_mid)

        if call_mid is None or put_mid is None:
            row["state"] = "NO_QUOTES"
            row["headline"] = (
                f"{tk}: earnings in {days_away:.1f}d, ATM chain found but "
                "bid/ask/last are missing — implied move withheld."
            )
            rows.append(row)
            continue

        straddle = call_mid + put_mid
        implied_pct = straddle / spot * 100.0
        # Desk shorthand: a long-straddle's breakeven IS the implied move
        # the market is pricing. Black-Scholes 1σ ≈ 0.8 × straddle/spot is
        # surfaced separately so the operator can reconcile without
        # re-querying yfinance.
        sigma_pct = implied_pct * 0.8
        implied_dollar = position_value * (implied_pct / 100.0)
        implied_book_pct = implied_dollar / tv * 100.0
        # Per-strike IV: prefer the call's; fall back to the put's. yfinance
        # reports decimal (0.45 = 45%); convert to percent for consistency.
        iv_atm = None
        for r in (call, put):
            if not isinstance(r, dict):
                continue
            iv_dec = _safe_float(r.get("impliedVolatility"))
            if iv_dec is not None and iv_dec > 0:
                iv_atm = iv_dec * 100.0
                break

        row["state"] = "OK"
        row["straddle"] = _z(straddle)
        row["implied_move_pct"] = _z(implied_pct)
        row["implied_one_sigma_pct"] = _z(sigma_pct)
        row["iv_atm"] = _z(iv_atm)
        row["implied_dollar_move"] = _z(implied_dollar)
        row["implied_book_pct"] = _z(implied_book_pct)
        row["row_verdict"] = _row_verdict(implied_book_pct)
        row["headline"] = (
            f"{tk} in {days_away:.1f}d: market is pricing ±{implied_pct:.1f}% "
            f"(1σ ±{sigma_pct:.1f}%, straddle ${straddle:.2f} @ "
            f"spot ${spot:.2f}, exp {chain.get('expiry')}) "
            f"→ ±${implied_dollar:.2f} (book ±{implied_book_pct:.2f}%)."
        )
        rows.append(row)

    rows.sort(key=lambda r: r.get("days_to_earnings") or 1e9)
    base["events"] = rows
    base["n_events"] = len(rows)

    if not rows:
        base["state"] = "NO_EVENTS"
        base["headline"] = (
            f"Implied move: no held name reports within {horizon_days:.0f}d."
        )
        base["verdict"] = "NO_EVENTS"
        return base

    base["state"] = "OK"
    scored = [r for r in rows if r.get("implied_dollar_move") is not None]
    if scored:
        # Sum of |implied| dollar moves — the honest upper bound (same
        # convention as ``build_earnings_shock``'s ``total_sigma_book_pct``;
        # earnings prints are independent draws, a quadrature add would
        # understate the worst-case correlated-shock reality).
        total_implied_dollar = sum(abs(r["implied_dollar_move"]) for r in scored)
        total_implied_book_pct = total_implied_dollar / tv * 100.0
        base["total_implied_book_pct"] = _z(total_implied_book_pct)
        base["verdict"] = _row_verdict(total_implied_book_pct)

    worst_row = max(
        rows,
        key=lambda r: abs(r["implied_book_pct"]) if r.get("implied_book_pct") is not None else -1,
    )
    if worst_row.get("implied_book_pct") is not None:
        base["headline"] = (
            f"Options-implied move ({base['n_events']} held name"
            f"{'' if base['n_events'] == 1 else 's'} ≤{horizon_days:.0f}d): "
            f"worst is {worst_row['headline']}"
        )
    else:
        # All rows withheld their implied — surface the bare event list.
        first = rows[0]
        base["headline"] = (
            f"Options-implied move: {base['n_events']} held event"
            f"{'' if base['n_events'] == 1 else 's'} within "
            f"{horizon_days:.0f}d, implied withheld "
            f"(chain unavailable on {first['ticker']})."
        )
        base["verdict"] = "NO_QUOTES"

    return base

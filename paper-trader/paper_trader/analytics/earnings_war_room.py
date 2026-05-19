"""Pre-print game plan — every desk-relevant figure for held imminent earnings, composed into one view.

Forward earnings exposure has six siloed surfaces today, each answering a
different sub-question on the same event:

* ``/api/event-calendar``       — who reports, when
* ``/api/earnings-shock``       — historical-Gaussian σ from prior prints
* ``/api/earnings-distribution``— empirical observed quartiles
* ``/api/implied-move``         — options-priced straddle (market-implied)
* ``/api/stress-scenarios``     — generic −10 % single-name / sector gap
* ``/api/sector-exposure``      — current concentration

The operator's actual morning-of-print question composes *across* all six:
**"If NVDA gaps by the implied move tomorrow, what does my book look like
after — total value vs the $1000 start, post-shock concentration, and which
single event carries the most $-at-risk?"** Nothing on the surface answers
that without the operator manually opening six tabs and doing the arithmetic.

``build_earnings_war_room`` is the composer. It does **no new measurement**
— every numeric input is read **verbatim** (single source of truth,
AGENTS.md #10) from the sibling builders' results passed in:

* per-event ``days_away`` / ``tier`` come from ``build_event_calendar``
* per-event ``implied_move_pct`` + ``implied_dollar_at_risk`` come from
  ``build_implied_move``
* per-event ``sigma_pct`` / ``sigma_dollar_move`` / ``sigma_book_pct``
  come from ``build_earnings_shock``
* per-name single-name shock comes from
  ``build_stress_scenarios.single_name`` when the ticker matches; else a
  flat −10 % single-name reference is computed against the position value
  (matches ``stress_scenarios._SINGLE_NAME_GAP_PCT``).

Each event row carries the **post-implied projection**: if the position
moves by ``-|implied_move_pct|`` (worst-direction stress) while everything
else holds flat, what is the resulting:

* ``post_shock_total_value``  — book mark-to-market
* ``post_shock_vs_initial_pct`` — vs the $1000 start (the universal P/L
  baseline every Discord report is measured against)
* ``post_shock_weight_pct``   — this position's % of the new (smaller) book

A per-event impact tier (``HIGH`` / ``MEDIUM`` / ``LOW``) is set on
``max(|implied_book_pct|, |sigma_book_pct|)`` so a missing chain (no
implied) still tiers off the historical σ, and vice-versa for a fresh-IPO
name with no print history.

Top-level summary:

* ``total_implied_dollars_at_risk`` — sum of |implied_dollar_at_risk|
* ``worst_case_event``  — the event with the largest |implied_dollar_at_risk|
* ``verdict``           — overall tier from the worst event
* ``headline``          — one Discord-ready sentence

State ladder mirrors the sibling earnings builders:

* ``NO_DATA``   — empty / unpriceable book (no total_value)
* ``NO_EVENTS`` — book is priced, no held imminent event inside horizon
* ``OK``        — at least one held imminent event with at least one of
  (implied, σ) available; an event with neither is still emitted with
  ``state=INSUFFICIENT`` so the operator never misses *"NVDA reports
  tomorrow"* — the ``earnings_shock`` honesty precedent.

Per-event ``state`` ladder (independent of top-level):

* ``OK``                — at least one of implied / σ available
* ``INSUFFICIENT``      — both implied and σ unavailable (chain miss AND
  print history < ``MIN_HISTORY`` from ``earnings_shock``); the row still
  lists ticker / days / weight so the calendar awareness survives.

Observational / advisory only — never gates Opus, never injected into the
decision prompt, no caps (AGENTS.md #2/#12 — the ``earnings_shock`` /
``stress_scenarios`` / ``recovery`` precedent). Pure, no I/O, never raises.
"""
from __future__ import annotations

from datetime import datetime, timezone

DEFAULT_HORIZON_DAYS = 7.0
HIGH_BOOK_PCT = 5.0
MEDIUM_BOOK_PCT = 2.0
# Matches stress_scenarios._SINGLE_NAME_GAP_PCT. Verbatim copy so the
# fallback single-name shock is byte-identical to /api/stress-scenarios
# when the war_room caller has no stress result to consume (the
# stress_scenarios._LEVERAGE_BETA verbatim-copy precedent).
_DEFAULT_SINGLE_NAME_GAP_PCT = -10.0


def _z(v: float | None, ndigits: int = 2) -> float | None:
    if v is None:
        return None
    try:
        r = round(float(v), ndigits)
    except (TypeError, ValueError):
        return None
    return 0.0 if r == 0 else r


def _f(v, default: float | None = None) -> float | None:
    if v is None:
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if f != f:  # NaN
        return default
    return f


def _row_tier(implied_book_pct: float | None,
              sigma_book_pct: float | None) -> str:
    """HIGH/MEDIUM/LOW on max(|implied|, |σ|) of the two book-impact figures.

    Either may be ``None`` (missing chain or insufficient print history);
    the tier reflects whichever is available so the event still gets a
    risk label."""
    candidates = []
    if implied_book_pct is not None:
        candidates.append(abs(implied_book_pct))
    if sigma_book_pct is not None:
        candidates.append(abs(sigma_book_pct))
    if not candidates:
        return "UNKNOWN"
    worst = max(candidates)
    if worst >= HIGH_BOOK_PCT:
        return "HIGH"
    if worst >= MEDIUM_BOOK_PCT:
        return "MEDIUM"
    return "LOW"


def _position_value(position: dict) -> float | None:
    """Match ``stress_scenarios._position_betas`` value semantics (options
    ×100, price falls back avg_cost). ``None`` on unpriceable so the caller
    can skip rather than dollarize against zero."""
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


def _index_by_ticker(rows, key: str = "ticker") -> dict:
    """Build {ticker.upper(): row} from a builder's events list. Skips
    non-dict / missing-key rows silently — the ``_safe`` contract."""
    out: dict = {}
    if not isinstance(rows, list):
        return out
    for r in rows:
        if not isinstance(r, dict):
            continue
        try:
            tk = (r.get(key) or "").upper()
        except (TypeError, AttributeError):
            continue
        if tk and tk not in out:
            out[tk] = r
    return out


def _single_name_dollar_shock(
    ticker: str,
    position_value: float,
    stress_result: dict | None,
) -> tuple[float, str]:
    """Single-name idiosyncratic shock for this ticker, in dollars.

    Returns ``(pnl_usd_signed_negative, source)``. ``source`` is ``"stress"``
    when the stress_scenarios single_name row matches this ticker (SSOT
    consumption — same gap %, same arithmetic), else ``"default"`` and we
    apply ``_DEFAULT_SINGLE_NAME_GAP_PCT`` to the position value (verbatim
    copy of stress_scenarios._SINGLE_NAME_GAP_PCT).

    Never raises; a malformed stress result silently falls back to default."""
    if isinstance(stress_result, dict):
        try:
            sn = stress_result.get("single_name") or {}
            if isinstance(sn, dict):
                sn_ticker = (sn.get("ticker") or "").upper()
                if sn_ticker == ticker.upper():
                    pnl = _f(sn.get("pnl_usd"))
                    if pnl is not None:
                        return pnl, "stress"
        except (TypeError, AttributeError):
            pass
    return position_value * (_DEFAULT_SINGLE_NAME_GAP_PCT / 100.0), "default"


def build_earnings_war_room(
    positions: list[dict],
    total_value: float,
    initial_equity: float,
    event_calendar_result: dict | None,
    implied_move_result: dict | None = None,
    earnings_shock_result: dict | None = None,
    stress_scenarios_result: dict | None = None,
    now: datetime | None = None,
    horizon_days: float = DEFAULT_HORIZON_DAYS,
) -> dict:
    """Pure: no I/O, never raises. ``positions`` is the
    ``store.open_positions()`` shape. Sibling builder results are read
    verbatim — pass ``None`` to skip a source (the corresponding fields
    degrade to ``None`` honestly rather than fabricate).
    """
    now = now or datetime.now(timezone.utc)
    try:
        tv = float(total_value or 0.0)
    except (TypeError, ValueError):
        tv = 0.0
    try:
        ie = float(initial_equity or 0.0)
    except (TypeError, ValueError):
        ie = 0.0

    base = {
        "as_of": now.isoformat(timespec="seconds"),
        "horizon_days": float(horizon_days),
        "initial_equity": _z(ie),
        "current_total_value": _z(tv),
        "n_events": 0,
        "events": [],
        "total_implied_dollars_at_risk": None,
        "worst_case_event": None,
        "verdict": None,
        "headline": None,
    }

    # Value-index the held book.
    value_by_ticker: dict[str, float] = {}
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

    if not value_by_ticker or tv <= 0:
        base["state"] = "NO_DATA"
        base["headline"] = "Earnings war room: no priced book to project."
        return base

    cal_events: list[dict] = []
    if isinstance(event_calendar_result, dict):
        try:
            cal_events = list(event_calendar_result.get("events") or [])
        except (TypeError, AttributeError):
            cal_events = []

    # Index sibling builder rows by ticker for verbatim consumption.
    implied_rows = _index_by_ticker(
        (implied_move_result or {}).get("events")
        if isinstance(implied_move_result, dict) else None
    )
    shock_rows = _index_by_ticker(
        (earnings_shock_result or {}).get("events")
        if isinstance(earnings_shock_result, dict) else None
    )

    rows: list[dict] = []
    for ev in cal_events:
        if not isinstance(ev, dict):
            continue
        try:
            tk = (ev.get("ticker") or "").upper()
        except (TypeError, AttributeError):
            continue
        if not tk or tk not in value_by_ticker:
            # Skip non-held names — war room is for the held book.
            continue
        days_away = _f(ev.get("days_away"))
        if days_away is None or days_away < 0 or days_away > horizon_days:
            continue
        position_value = value_by_ticker[tk]
        weight_pct = position_value / tv * 100.0

        # Implied (forward, market-priced).
        imp = implied_rows.get(tk) or {}
        implied_pct = _f(imp.get("implied_move_pct"))
        # implied_dollar_at_risk is unsigned (a +X% gap can cost a short or
        # gain a long — operationally the desk wants $-at-risk magnitude on
        # the worst direction of the implied move).
        implied_dollar_at_risk = (
            position_value * (implied_pct / 100.0)
            if implied_pct is not None else None
        )
        implied_book_pct = (
            implied_dollar_at_risk / tv * 100.0
            if implied_dollar_at_risk is not None else None
        )

        # Historical σ (backward).
        shk = shock_rows.get(tk) or {}
        sigma_pct = _f(shk.get("sigma_pct"))
        sigma_dollar = _f(shk.get("sigma_dollar_move"))
        sigma_book_pct = _f(shk.get("sigma_book_pct"))

        # Single-name -10% shock (idiosyncratic, no beta — verbatim from
        # stress_scenarios when matched, else the verbatim-copied default).
        single_name_dollar, single_name_source = _single_name_dollar_shock(
            tk, position_value, stress_scenarios_result,
        )
        single_name_book_pct = single_name_dollar / tv * 100.0

        # Post-shock projection: this position drops by |implied|, others flat.
        # We use the downside of the implied straddle as the stress direction
        # (the desk's worst-direction frame). When implied is unavailable we
        # fall back to |σ| from earnings_shock; if both are missing the
        # projection is withheld honestly.
        shock_pct_for_projection = None
        if implied_pct is not None and implied_pct > 0:
            shock_pct_for_projection = implied_pct
        elif sigma_pct is not None and sigma_pct > 0:
            shock_pct_for_projection = sigma_pct

        if shock_pct_for_projection is not None:
            shock_dollar_loss = position_value * (shock_pct_for_projection / 100.0)
            post_total = tv - shock_dollar_loss
            post_position_value = position_value - shock_dollar_loss
            post_vs_initial_pct = (
                (post_total - ie) / ie * 100.0 if ie > 0 else None
            )
            post_weight_pct = (
                post_position_value / post_total * 100.0
                if post_total > 0 else None
            )
        else:
            shock_dollar_loss = None
            post_total = None
            post_vs_initial_pct = None
            post_weight_pct = None

        tier = _row_tier(implied_book_pct, sigma_book_pct)
        row_state = "OK" if (implied_pct is not None or sigma_pct is not None) else "INSUFFICIENT"

        row: dict = {
            "ticker": tk,
            "days_to_earnings": _z(days_away),
            "earnings_date": ev.get("earnings_date"),
            "tier_event_calendar": ev.get("tier"),
            "current_value_usd": _z(position_value),
            "weight_pct": _z(weight_pct),
            "implied_move_pct": _z(implied_pct),
            "implied_dollar_at_risk": _z(implied_dollar_at_risk),
            "implied_book_pct": _z(implied_book_pct),
            "sigma_pct": _z(sigma_pct),
            "sigma_dollar_move": _z(sigma_dollar),
            "sigma_book_pct": _z(sigma_book_pct),
            "single_name_shock_dollar": _z(single_name_dollar),
            "single_name_shock_book_pct": _z(single_name_book_pct),
            "single_name_shock_source": single_name_source,
            "post_shock_dollar_loss": _z(shock_dollar_loss),
            "post_shock_total_value": _z(post_total),
            "post_shock_vs_initial_pct": _z(post_vs_initial_pct),
            "post_shock_weight_pct": _z(post_weight_pct),
            "impact_tier": tier,
            "state": row_state,
        }
        # One-line per-event headline. Always rendered (never fabricates,
        # the INSUFFICIENT branch reads honestly).
        if row_state == "INSUFFICIENT":
            row["headline"] = (
                f"{tk} reports in {days_away:.1f}d — implied move + σ both "
                f"unavailable (chain miss + insufficient print history); "
                f"position is {weight_pct:.1f}% of book."
            )
        else:
            parts = [f"{tk} in {days_away:.1f}d"]
            if implied_pct is not None and implied_dollar_at_risk is not None:
                parts.append(
                    f"implied ±{implied_pct:.1f}% → ±${implied_dollar_at_risk:.2f} "
                    f"({(implied_book_pct or 0):+.2f}% of book)"
                )
            if sigma_pct is not None and sigma_dollar is not None:
                parts.append(
                    f"σ ±{sigma_pct:.1f}% → ±${sigma_dollar:.2f}"
                )
            if shock_dollar_loss is not None and post_vs_initial_pct is not None:
                parts.append(
                    f"down-shock leaves book at ${post_total:.2f} "
                    f"({post_vs_initial_pct:+.2f}% vs ${ie:.0f} start)"
                )
            parts.append(f"tier {tier}")
            row["headline"] = "; ".join(parts) + "."
        rows.append(row)

    if not rows:
        base["state"] = "NO_EVENTS"
        base["headline"] = (
            "Earnings war room: no held imminent print inside the horizon."
        )
        return base

    rows.sort(key=lambda r: r.get("days_to_earnings") or 1e9)

    # Top-level aggregates.
    impl_dollars = [
        abs(r["implied_dollar_at_risk"])
        for r in rows if r.get("implied_dollar_at_risk") is not None
    ]
    total_implied_at_risk = sum(impl_dollars) if impl_dollars else None

    # Worst case by |implied|, falling back to |σ| when implied is missing.
    def _worst_key(r: dict) -> float:
        imp = r.get("implied_dollar_at_risk")
        if imp is not None:
            return abs(imp)
        sig = r.get("sigma_dollar_move")
        if sig is not None:
            return abs(sig)
        sn = r.get("single_name_shock_dollar")
        if sn is not None:
            return abs(sn)
        return 0.0

    worst = max(rows, key=_worst_key)
    overall_tier = _row_tier(
        worst.get("implied_book_pct"),
        worst.get("sigma_book_pct"),
    )

    base.update({
        "state": "OK",
        "n_events": len(rows),
        "events": rows,
        "total_implied_dollars_at_risk": _z(total_implied_at_risk),
        "worst_case_event": {
            "ticker": worst["ticker"],
            "days_to_earnings": worst.get("days_to_earnings"),
            "implied_dollar_at_risk": worst.get("implied_dollar_at_risk"),
            "post_shock_vs_initial_pct": worst.get("post_shock_vs_initial_pct"),
            "impact_tier": worst.get("impact_tier"),
        },
        "verdict": overall_tier,
    })

    # Top-level headline — Discord-ready.
    headline_bits = [
        f"Earnings war room ({len(rows)} held imminent ≤{horizon_days:.0f}d)"
    ]
    if total_implied_at_risk is not None:
        headline_bits.append(
            f"implied $-at-risk ${total_implied_at_risk:.2f}"
        )
    headline_bits.append(f"worst: {worst['ticker']}")
    if worst.get("implied_dollar_at_risk") is not None:
        headline_bits.append(
            f"in {worst['days_to_earnings']:.1f}d → "
            f"±${abs(worst['implied_dollar_at_risk']):.2f} "
            f"(book {worst.get('implied_book_pct') or 0:+.2f}%)"
        )
    if worst.get("post_shock_vs_initial_pct") is not None:
        headline_bits.append(
            f"down-shock book {worst['post_shock_vs_initial_pct']:+.2f}% vs start"
        )
    headline_bits.append(f"verdict {overall_tier}")
    base["headline"] = "; ".join(headline_bits) + "."
    return base

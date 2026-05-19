"""Pre-earnings dollarized-shock estimator for held positions with imminent prints.

``event_calendar`` already tells the live trader *which* held names report
and *when* (the forward-awareness block fed into the Opus prompt). But
nothing else on the surface translates "NVDA earnings in 0.9d" into the
question every desk asks before holding into a print: *if this name gaps
by a typical 1σ on its release, what does it cost MY book today?*

A 44 %-of-book NVDA position is the live 2026-05-19 shape — and the
``stress_scenarios`` builder (which is otherwise the closest sibling) only
models market / sector / single-name *generic* −10 % gaps. Earnings prints
are bigger than that on a typical AI/semis name (historical 1σ ≈ 5–8 %, 3σ
tails routinely 15-20 %) and the trader's existing surfaces are silent
about them. This is the missing pre-earnings $-at-risk line.

``build_earnings_shock`` composes ``build_event_calendar``'s ``events``
list **verbatim** (single source of truth, AGENTS.md #10): the held set,
days_away, and tier come from the canonical event-calendar verdict so this
builder and ``/api/event-calendar`` / ``/api/earnings-risk`` can never
disagree on what counts as held-imminent. For each held event the
``history_provider`` callable returns a list of historical 1-day post-
earnings reactions in **percent**. The builder is otherwise pure: no I/O,
no yfinance, never raises (the endpoint owns the I/O — the ``tail_risk`` /
``stress_scenarios`` builder/endpoint split).

State ladder mirrors the sibling builders:

* ``NO_DATA``   — empty book / unpriceable / no total_value (the
  ``stress_scenarios`` "no priced book to shock yet" precedent).
* ``NO_EVENTS`` — book is fine, calendar is fine, but no held name has an
  imminent print inside the horizon (the most common branch on a typical
  desk; intentionally distinct from NO_DATA so the operator can tell
  *"calendar quiet"* from *"book empty"*).
* ``OK``        — at least one held imminent event with shock numerics
  emitted.

Per-event sample-size honesty (the ``build_correlation`` /
``build_news_velocity`` precedent): a name with fewer than
``MIN_HISTORY=3`` historical reactions reads ``INSUFFICIENT_HISTORY`` at
the row level — the event still surfaces (so the operator never misses
*"NVDA reports tomorrow"*) but the σ figure is **withheld** with an
honest sentence, never fabricated from one print.

Vol convention: **population stdev (÷n)**, byte-identical to
``build_tail_risk``'s ``annualized_vol_pct`` calc on the equity curve
(``Σ (x − μ)² / n``). A drift in either side would fail the no-drift
test (SSOT, invariant #10 — the same discipline the ``recovery`` builder
uses for the σ-day figure).

Observational / advisory only — never gates Opus, never injected into the
decision prompt, no caps (invariants #2/#12 — the
``stress_scenarios`` / ``recovery`` / ``event_calendar`` precedent).
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Callable, Iterable

MIN_HISTORY = 3                # below this, withhold σ at the row level
DEFAULT_HORIZON_DAYS = 7.0     # only events ≤ this many days away are scored
DEFAULT_HISTORY_DEPTH = 8      # most recent N earnings reactions per name
ELEVATED_BOOK_PCT = 5.0        # |1σ book impact| ≥ this ⇒ ELEVATED tier
MODERATE_BOOK_PCT = 2.0        # |1σ book impact| ≥ this ⇒ MODERATE tier (else LOW)


def _z(v: float | None, ndigits: int = 2) -> float | None:
    """Round; fold -0.0 → 0.0 so the JSON never carries a signed zero
    (the ``stress_scenarios._z`` precedent — same shape, same contract)."""
    if v is None:
        return None
    try:
        r = round(float(v), ndigits)
    except (TypeError, ValueError):
        return None
    return 0.0 if r == 0 else r


def _pop_stdev(xs: Iterable[float]) -> float | None:
    """Population stdev (÷n) of an iterable of floats. ``None`` on <2
    samples or non-numeric input. Matches ``build_tail_risk``'s vol
    convention (SSOT, AGENTS.md #10) — a ÷(n−1) regression would shift
    every σ figure here and fail the no-drift cross-check."""
    vals: list[float] = []
    for x in xs:
        try:
            vals.append(float(x))
        except (TypeError, ValueError):
            continue
    n = len(vals)
    if n < 2:
        return None
    mu = sum(vals) / n
    var = sum((v - mu) ** 2 for v in vals) / n
    return math.sqrt(var)


def _position_value(position: dict) -> float | None:
    """Match ``stress_scenarios._position_betas`` value semantics: options
    ×100, price falls back avg_cost. Returns ``None`` on a fully-unpriceable
    row so the caller can skip rather than dollarize against zero."""
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


def _row_verdict(sigma_book_pct: float | None) -> str:
    """ELEVATED/MODERATE/LOW classification of the |1σ| book impact."""
    if sigma_book_pct is None:
        return "UNKNOWN"
    a = abs(sigma_book_pct)
    if a >= ELEVATED_BOOK_PCT:
        return "ELEVATED"
    if a >= MODERATE_BOOK_PCT:
        return "MODERATE"
    return "LOW"


def build_earnings_shock(
    positions: list[dict],
    total_value: float,
    event_calendar_result: dict | None,
    history_provider: Callable[[str], list[float]] | None,
    now: datetime | None = None,
    horizon_days: float = DEFAULT_HORIZON_DAYS,
    history_depth: int = DEFAULT_HISTORY_DEPTH,
) -> dict:
    """Pure: no I/O, never raises. ``positions`` is the
    ``store.open_positions()`` shape (or the strategy snapshot — both carry
    ``ticker``/``qty``/``current_price``/``avg_cost``/``type``).
    ``event_calendar_result`` is the dict from ``build_event_calendar``
    (its ``events`` list provides held + days_away). ``history_provider`` is
    the I/O seam: given a ticker string, returns a list of historical
    1-day post-earnings reactions **in percent** (e.g. ``[-3.2, +6.5, ...]``).
    Pass ``None`` to skip history entirely (all rows read
    ``INSUFFICIENT_HISTORY``).
    """
    now = now or datetime.now(timezone.utc)
    try:
        tv = float(total_value or 0.0)
    except (TypeError, ValueError):
        tv = 0.0

    base = {
        "as_of": now.isoformat(timespec="seconds"),
        "horizon_days": float(horizon_days),
        "history_depth": int(history_depth),
        "n_events": 0,
        "events": [],
        "total_sigma_book_pct": None,
        "headline": None,
        "verdict": None,
    }

    # Value-index the held book so we can dollarize each event.
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
        base["headline"] = "Earnings shock: no priced book to shock yet."
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
        # Only score events inside the shock horizon. A "held but distant"
        # event is honest awareness (event_calendar emits it) but not a
        # shock candidate — emitting it here would dilute the headline
        # with a 30-days-out figure no one acts on.
        if days_away < 0 or days_away > horizon_days:
            continue
        position_value = value_by_ticker[tk]
        pct_of_book = position_value / tv * 100.0

        history: list[float] = []
        if history_provider is not None:
            try:
                raw = history_provider(tk) or []
            except Exception:
                raw = []
            for x in raw[:history_depth]:
                try:
                    history.append(float(x))
                except (TypeError, ValueError):
                    continue

        sigma_pct = _pop_stdev(history) if len(history) >= MIN_HISTORY else None
        worst_pct = min(history) if history else None
        best_pct = max(history) if history else None
        mean_pct = (sum(history) / len(history)) if history else None

        row: dict = {
            "ticker": tk,
            "days_to_earnings": _z(days_away),
            "earnings_date": ev.get("earnings_date"),
            "tier": ev.get("tier"),
            "current_value_usd": _z(position_value),
            "weight_pct": _z(pct_of_book),
            "n_history": len(history),
            "history_mean_pct": _z(mean_pct),
            "history_worst_pct": _z(worst_pct),
            "history_best_pct": _z(best_pct),
        }

        if sigma_pct is None:
            row["state"] = "INSUFFICIENT_HISTORY"
            row["sigma_pct"] = None
            row["sigma_dollar_move"] = None
            row["sigma_book_pct"] = None
            row["stress_3sigma_dollar_down"] = None
            row["stress_3sigma_book_pct_down"] = None
            row["row_verdict"] = "UNKNOWN"
            row["headline"] = (
                f"{tk}: earnings in {days_away:.1f}d — σ withheld "
                f"({len(history)} historical print"
                f"{'' if len(history) == 1 else 's'}, "
                f"need ≥{MIN_HISTORY})"
            )
        else:
            sigma_dollar = position_value * (sigma_pct / 100.0)
            sigma_book_pct = sigma_dollar / tv * 100.0
            stress_3s_dollar_down = -3.0 * sigma_dollar
            stress_3s_book_pct_down = stress_3s_dollar_down / tv * 100.0
            row["state"] = "OK"
            row["sigma_pct"] = _z(sigma_pct)
            row["sigma_dollar_move"] = _z(sigma_dollar)
            row["sigma_book_pct"] = _z(sigma_book_pct)
            row["stress_3sigma_dollar_down"] = _z(stress_3s_dollar_down)
            row["stress_3sigma_book_pct_down"] = _z(stress_3s_book_pct_down)
            row["row_verdict"] = _row_verdict(sigma_book_pct)
            row["headline"] = (
                f"{tk} in {days_away:.1f}d: σ ±{sigma_pct:.1f}% "
                f"(n={len(history)} prints, worst {worst_pct:+.1f}%, "
                f"best {best_pct:+.1f}%) → ±${sigma_dollar:.2f} "
                f"(book ±{sigma_book_pct:.2f}%); 3σ down stress "
                f"${stress_3s_dollar_down:+.2f} "
                f"({stress_3s_book_pct_down:+.2f}% of book)."
            )
        rows.append(row)

    rows.sort(key=lambda r: r.get("days_to_earnings") or 1e9)
    base["events"] = rows
    base["n_events"] = len(rows)

    if not rows:
        base["state"] = "NO_EVENTS"
        base["headline"] = (
            f"Earnings shock: no held name reports within {horizon_days:.0f}d."
        )
        base["verdict"] = "NO_EVENTS"
        return base

    base["state"] = "OK"
    # Book-wide aggregate: sum of |1σ| dollar moves over scored events (the
    # honest upper bound — earnings prints are independent draws, so a
    # quadrature add would understate the worst-case "all surprise the same
    # way" reality that drives correlated book moves).
    scored = [r for r in rows if r.get("sigma_dollar_move") is not None]
    if scored:
        total_sigma_dollar = sum(abs(r["sigma_dollar_move"]) for r in scored)
        total_sigma_book_pct = total_sigma_dollar / tv * 100.0
        base["total_sigma_book_pct"] = _z(total_sigma_book_pct)
        base["verdict"] = _row_verdict(total_sigma_book_pct)

    worst_row = max(
        rows,
        key=lambda r: abs(r["sigma_book_pct"]) if r.get("sigma_book_pct") is not None else -1,
    )
    if worst_row.get("sigma_book_pct") is not None:
        base["headline"] = (
            f"Pre-earnings shock ({base['n_events']} held name"
            f"{'' if base['n_events'] == 1 else 's'} ≤{horizon_days:.0f}d): "
            f"worst is {worst_row['headline']}"
        )
    else:
        # All rows withheld their σ — surface the bare event list honestly.
        first = rows[0]
        base["headline"] = (
            f"Pre-earnings shock: {base['n_events']} held event"
            f"{'' if base['n_events'] == 1 else 's'} within "
            f"{horizon_days:.0f}d, σ withheld "
            f"(insufficient history on {first['ticker']})."
        )
        base["verdict"] = "INSUFFICIENT_HISTORY"

    return base


if __name__ == "__main__":  # smoke test against the live snapshot
    import json as _json

    from paper_trader.analytics.event_calendar import build_event_calendar
    from paper_trader.store import get_store
    from paper_trader.strategy import WATCHLIST

    s = get_store()
    pos = s.open_positions()
    pf = s.get_portfolio()
    ec = build_event_calendar(
        pos,
        {(p.get("ticker") or "").upper() for p in pos} | set(WATCHLIST[:5]),
    )
    # No yfinance here — smoke runs with empty history (every row reads
    # INSUFFICIENT_HISTORY but the per-event surface still proves the wire-up).
    rep = build_earnings_shock(
        pos, float(pf.get("total_value") or 0.0), ec, history_provider=lambda _t: [],
    )
    print(rep["headline"])
    print("\n---\n")
    print(_json.dumps(rep, indent=2, default=str))

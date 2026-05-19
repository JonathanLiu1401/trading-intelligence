"""Empirical observed-quantile companion to ``/api/earnings-shock``.

``earnings_shock`` assumes a Gaussian shock model: it reports a single 1σ
figure and a 3σ down-stress for each held imminent-print position. That's
the conservative-desk view, but earnings moves are *fat-tailed* — the worst
historical observed reaction is routinely larger than the 1σ figure for
small-n names. The Gaussian framing hides the historical worst case in the
math.

This builder reports the **empirical distribution** instead — for each held
imminent earnings event, the worst / Q1 / median / Q3 / best observed
single-day post-earnings reaction, each dollarized against the current
position size. Operator gets a direct read of "what has this name actually
done on prior prints, and what would that mean for my book today?" alongside
the Gaussian σ from ``/api/earnings-shock``.

Per advisor feedback, the quantiles are framed as **observed** (worst,
Q1, median, Q3, best) — not as "5th percentile / 25th percentile / etc." —
because n=4-8 historical prints cannot legitimately support distributional
percentile claims. Calling Q1 "the 25th percentile" with n=4 prints implies
a confidence the data doesn't support; "observed quartile" is what we can
actually defend.

Composes ``event_calendar``'s ``events`` list verbatim (SSOT, AGENTS.md
#10) — the same held-imminent set surfaced by ``/api/earnings-shock`` and
``/api/event-calendar`` — and reuses the same ``history_provider`` I/O
seam so the dashboard wrapper passes the cached
``_earnings_history_for`` and this module stays pure.

Observational only — never gates Opus, never injected into the decision
prompt, no caps (invariants #2/#12 — the ``earnings_shock`` /
``stress_scenarios`` / ``recovery`` precedent).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Iterable

MIN_HISTORY = 3                # below this we withhold quantiles (one
                               # observation isn't a distribution).
DEFAULT_HORIZON_DAYS = 7.0     # only events ≤ this many days away are scored
DEFAULT_HISTORY_DEPTH = 8      # most recent N earnings reactions per name
# Book-impact tiers on the |worst observed| outcome — these mirror the
# earnings_shock thresholds so the two endpoints rate the same shape of risk
# consistently (5% / 2% book impact thresholds = ELEVATED/MODERATE/LOW).
ELEVATED_BOOK_PCT = 5.0
MODERATE_BOOK_PCT = 2.0


def _z(v: float | None, ndigits: int = 2) -> float | None:
    """Round; fold ``-0.0`` → ``0.0`` so the JSON never carries a signed
    zero (the ``earnings_shock._z`` / ``stress_scenarios._z`` precedent)."""
    if v is None:
        return None
    try:
        r = round(float(v), ndigits)
    except (TypeError, ValueError):
        return None
    return 0.0 if r == 0 else r


def _position_value(position: dict) -> float | None:
    """Match ``earnings_shock._position_value`` semantics: options ×100,
    price falls back to ``avg_cost``. Returns ``None`` on a fully-unpriceable
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


def _observed_quartiles(xs: list[float]) -> dict[str, float] | None:
    """Empirical worst / Q1 / median / Q3 / best of a list of floats. Uses
    linear interpolation on the sorted samples (NIST type 7, the numpy /
    pandas default). Returns ``None`` on fewer than ``MIN_HISTORY`` samples.

    Why linear interpolation: with n=8 prints, the index-1 sample isn't
    *literally* the 25th percentile but it's the most reasonable point
    estimate the data supports. Naming the field ``q1`` rather than
    ``p25`` is the honest framing (per advisor) — we expose the shape
    of the distribution without making distributional claims the small-n
    data can't carry.
    """
    if not xs or len(xs) < MIN_HISTORY:
        return None
    s = sorted(float(x) for x in xs)
    n = len(s)

    def _interp(q: float) -> float:
        """Linear-interpolated quantile, q∈[0,1]. Matches numpy default."""
        # Numpy's "linear" rule: pos = q * (n - 1).
        pos = q * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return s[lo] + frac * (s[hi] - s[lo])

    return {
        "worst": s[0],
        "q1": _interp(0.25),
        "median": _interp(0.50),
        "q3": _interp(0.75),
        "best": s[-1],
    }


def _row_verdict(worst_book_pct: float | None) -> str:
    """ELEVATED/MODERATE/LOW classification of the |worst-observed| book
    impact. Mirrors ``earnings_shock._row_verdict`` thresholds so the two
    endpoints stay consistent."""
    if worst_book_pct is None:
        return "UNKNOWN"
    a = abs(worst_book_pct)
    if a >= ELEVATED_BOOK_PCT:
        return "ELEVATED"
    if a >= MODERATE_BOOK_PCT:
        return "MODERATE"
    return "LOW"


def build_earnings_distribution(
    positions: list[dict],
    total_value: float,
    event_calendar_result: dict | None,
    history_provider: Callable[[str], list[float]] | None,
    now: datetime | None = None,
    horizon_days: float = DEFAULT_HORIZON_DAYS,
    history_depth: int = DEFAULT_HISTORY_DEPTH,
) -> dict:
    """Pure: no I/O, never raises. Shape mirrors
    ``build_earnings_shock`` (same state ladder, same ``events`` shape) so
    the dashboard can render the two side-by-side.

    Per-event payload (state == ``OK``):

    * ``ticker``, ``days_to_earnings``, ``earnings_date``, ``tier``
    * ``current_value_usd`` — current marked book exposure on the position
    * ``weight_pct`` — share of book (% of ``total_value``)
    * ``n_history`` — count of historical prints used
    * ``observed_quartiles`` — ``{worst, q1, median, q3, best}`` of historical
      single-day post-earnings reactions (in % terms, signed)
    * ``dollar_quartiles`` — same shape, scaled by ``current_value_usd``
    * ``book_pct_quartiles`` — same shape, normalised by ``total_value``
    * ``downside_worst_dollar`` / ``downside_worst_book_pct`` — the loss-
      side worst case ($\\le 0$) clipped at zero (gain-side worst case is
      not a loss); convenient for the operator's "what's the worst this
      name has cost on a print?" question
    * ``row_verdict`` — ELEVATED/MODERATE/LOW on |worst observed book impact|
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
        "min_history": int(MIN_HISTORY),
        "n_events": 0,
        "events": [],
        "total_downside_book_pct": None,
        "headline": None,
        "verdict": None,
    }

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
        base["headline"] = "Earnings distribution: no priced book to dollarize yet."
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

        q = _observed_quartiles(history)

        row: dict = {
            "ticker": tk,
            "days_to_earnings": _z(days_away),
            "earnings_date": ev.get("earnings_date"),
            "tier": ev.get("tier"),
            "current_value_usd": _z(position_value),
            "weight_pct": _z(pct_of_book),
            "n_history": len(history),
        }

        if q is None:
            row["state"] = "INSUFFICIENT_HISTORY"
            row["observed_quartiles"] = None
            row["dollar_quartiles"] = None
            row["book_pct_quartiles"] = None
            row["downside_worst_dollar"] = None
            row["downside_worst_book_pct"] = None
            row["row_verdict"] = "UNKNOWN"
            row["headline"] = (
                f"{tk}: earnings in {days_away:.1f}d — distribution withheld "
                f"({len(history)} historical print"
                f"{'' if len(history) == 1 else 's'}, "
                f"need ≥{MIN_HISTORY})"
            )
        else:
            # Dollar + book-pct quartiles (same shape as observed_quartiles).
            dollar_q = {
                k: (position_value * v / 100.0) for k, v in q.items()
            }
            book_pct_q = {
                k: (dollar_q[k] / tv * 100.0) for k in q
            }
            # Downside-only headline numbers: if the worst observed reaction
            # is *positive* (i.e., every observed print was a gain), the
            # "downside" worst case is 0 — there is no historical loss to
            # surface. Don't manufacture a negative number out of a positive
            # observation.
            downside_worst_dollar = min(dollar_q["worst"], 0.0)
            downside_worst_book_pct = downside_worst_dollar / tv * 100.0
            row["state"] = "OK"
            row["observed_quartiles"] = {k: _z(v, 2) for k, v in q.items()}
            row["dollar_quartiles"] = {k: _z(v, 2) for k, v in dollar_q.items()}
            row["book_pct_quartiles"] = {
                k: _z(v, 2) for k, v in book_pct_q.items()
            }
            row["downside_worst_dollar"] = _z(downside_worst_dollar, 2)
            row["downside_worst_book_pct"] = _z(downside_worst_book_pct, 2)
            row["row_verdict"] = _row_verdict(downside_worst_book_pct)
            row["headline"] = (
                f"{tk} in {days_away:.1f}d (n={len(history)} prints): "
                f"worst {q['worst']:+.1f}% → ${dollar_q['worst']:+.2f} "
                f"(book {book_pct_q['worst']:+.2f}%); median "
                f"{q['median']:+.1f}%; best {q['best']:+.1f}%."
            )
        rows.append(row)

    rows.sort(key=lambda r: r.get("days_to_earnings") or 1e9)
    base["events"] = rows
    base["n_events"] = len(rows)

    if not rows:
        base["state"] = "NO_EVENTS"
        base["headline"] = (
            f"Earnings distribution: no held name reports within {horizon_days:.0f}d."
        )
        base["verdict"] = "NO_EVENTS"
        return base

    base["state"] = "OK"
    # Aggregate book-wide downside: sum of |downside_worst_dollar| across
    # scored events (independent-name assumption — fine for the operator's
    # "what's the cumulative bad-case across my held prints?" question).
    # Quadrature would understate the correlated-surprise reality; this is
    # the same SSOT shape ``earnings_shock`` uses for its 1σ sum.
    scored = [r for r in rows if r.get("downside_worst_dollar") is not None]
    if scored:
        total_downside_dollar = sum(
            abs(r["downside_worst_dollar"]) for r in scored
        )
        total_downside_book_pct = -total_downside_dollar / tv * 100.0
        base["total_downside_book_pct"] = _z(total_downside_book_pct)
        base["verdict"] = _row_verdict(total_downside_book_pct)

    # Headline picks the row with the largest |downside| book impact (the
    # "worst observed historical loss" — the operator's clearest single
    # actionable number).
    def _row_downside_abs(r: dict) -> float:
        v = r.get("downside_worst_book_pct")
        return abs(v) if v is not None else -1.0

    worst_row = max(rows, key=_row_downside_abs)
    if worst_row.get("downside_worst_book_pct") is not None:
        base["headline"] = (
            f"Pre-earnings observed distribution ({base['n_events']} held "
            f"name{'' if base['n_events'] == 1 else 's'} ≤{horizon_days:.0f}d): "
            f"worst is {worst_row['headline']}"
        )
    else:
        first = rows[0]
        base["headline"] = (
            f"Pre-earnings observed distribution: {base['n_events']} held "
            f"event{'' if base['n_events'] == 1 else 's'} within "
            f"{horizon_days:.0f}d, distribution withheld "
            f"(insufficient history on {first['ticker']})."
        )
        base["verdict"] = "INSUFFICIENT_HISTORY"

    return base

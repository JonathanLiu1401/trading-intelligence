"""Left-tail / downside-shape diagnostic for the live book.

The stack measures *upside* and *peak-to-trough drawdown* exhaustively
(``/api/analytics`` Sharpe/Sortino/Calmar, ``/api/drawdown``,
``compute_drawdown``) but **nothing answers the desk-risk question a
trader asks first**: *what is a realistic bad day, and how fat / skewed
is the loss distribution?* Max-drawdown is a single worst path; it says
nothing about the frequency or shape of daily losses.

``build_tail_risk`` fills exactly that gap and nothing else:

* **Historical VaR** (95% / 99%, 1-day) — the loss the book has actually
  matched-or-exceeded at that frequency. Non-parametric on purpose: a
  Gaussian VaR understates fat tails, which is the whole point of the
  panel.
* **CVaR / expected shortfall** — the mean loss *given* we are in the
  tail (the number that actually hurts).
* **Annualised vol & downside deviation**, **return skew**, **worst /
  best day**, **max consecutive down-day streak**, **Ulcer index**.

Convention lock (single-source-of-truth discipline, AGENTS.md #10
*spirit*): the daily-return series is built **byte-identically** to
``dashboard.analytics_api`` — resample ``equity_curve`` to the *last*
``total_value`` per UTC day, simple returns between consecutive days
with a positive prior value. ``annualised_vol`` divides by ``n``
(population) to match that endpoint's Sharpe ``var = sum/len``;
``downside_deviation`` divides by total ``n`` to match its Sortino
``dvar = sum(r*r)/len``. A future refactor that changes one must change
both, or the dashboard's Sharpe/Sortino and this panel silently disagree.

Sample-size honesty mirrors ``build_correlation`` / ``build_churn``:
numeric metrics are emitted as soon as they are mathematically defined,
but the **verdict is withheld** (``state="INSUFFICIENT"``) until
``MIN_RETURNS`` daily observations exist — a five-day VaR is noise. With
no equity history at all the state is ``NO_DATA`` and every metric is
``None``.

This is a *diagnostic / advisory* panel only. It never gates Opus and
adds no caps (AGENTS.md #2 / #12) — it is deliberately **not** injected
into the live decision prompt.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

# Below this many daily returns the distribution shape is noise; emit the
# numerics but withhold the verdict (the build_correlation precedent).
MIN_RETURNS = 20

_TRADING_DAYS = 252
_ANN = math.sqrt(_TRADING_DAYS)


def _z(v: float | None, ndigits: int = 2) -> float | None:
    """Round, and fold -0.0 → 0.0 so the JSON never carries a signed zero."""
    if v is None:
        return None
    r = round(v, ndigits)
    return 0.0 if r == 0 else r


def _daily_series(equity_curve: list[dict]) -> tuple[list[float], list[float]]:
    """(daily_values, daily_returns).

    Resample to the last total_value per UTC day, then simple returns
    between consecutive days with a positive prior value — identical to
    ``dashboard.analytics_api``'s ``by_day`` loop so the two never drift.
    """
    by_day: dict[str, float] = {}
    for p in equity_curve or []:
        day = (p.get("timestamp") or "")[:10]
        if day:
            by_day[day] = p.get("total_value")
    day_keys = sorted(by_day.keys())
    values = [by_day[d] for d in day_keys]
    returns: list[float] = []
    for i in range(1, len(values)):
        prev, cur = values[i - 1], values[i]
        if prev and prev > 0 and cur is not None:
            returns.append(cur / prev - 1.0)
    return values, returns


def _percentile_loss(sorted_returns: list[float], q: float) -> float:
    """Nearest-rank historical VaR at confidence ``1-q``, as a positive
    loss magnitude. ``idx = ceil(q*n) - 1`` clamped to ``[0, n-1]`` on the
    ascending-sorted returns; a positive quantile return yields a negative
    VaR (honest "no loss at that confidence"), not a clamped 0."""
    n = len(sorted_returns)
    idx = math.ceil(q * n) - 1
    idx = max(0, min(idx, n - 1))
    return -sorted_returns[idx] * 100.0


def _cvar(sorted_returns: list[float], q: float) -> float:
    """Historical expected shortfall: mean of the worst ``ceil(q*n)``
    daily returns, as a positive loss. Deliberately **positional**
    (slice the k most-negative returns) not value-threshold — two
    returns that are equal in theory (e.g. 99/110-1 and 89.1/99-1) differ
    in the last float bits, so a ``r <= threshold`` filter would silently
    drop one tie and halve the tail. The slice is always non-empty
    (``k >= 1``)."""
    n = len(sorted_returns)
    k = max(1, min(math.ceil(q * n), n))
    tail = sorted_returns[:k]
    return -(sum(tail) / len(tail)) * 100.0


def _max_down_streak(returns: list[float]) -> int:
    best = cur = 0
    for r in returns:
        if r < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _ulcer_index_pct(values: list[float]) -> float | None:
    """Martin/Ulcer index on the daily-resampled equity: RMS of the
    percentage drawdown from the running peak."""
    if not values:
        return None
    peak = values[0]
    sq_sum = 0.0
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100.0 if peak else 0.0
        sq_sum += dd * dd
    return math.sqrt(sq_sum / len(values))


def build_tail_risk(
    equity_curve: list[dict], now: datetime | None = None
) -> dict:
    now = now or datetime.now(timezone.utc)
    values, returns = _daily_series(equity_curve)
    n_days = len(values)
    n = len(returns)

    base = {
        "as_of": now.isoformat(timespec="seconds"),
        "n_days": n_days,
        "n_returns": n,
        "min_returns": MIN_RETURNS,
        "var_95_pct": None,
        "var_99_pct": None,
        "cvar_95_pct": None,
        "annualized_vol_pct": None,
        "downside_deviation_pct": None,
        "return_skew": None,
        "worst_day_pct": None,
        "best_day_pct": None,
        "max_consecutive_down_days": None,
        "ulcer_index_pct": None,
    }

    if n == 0:
        base["state"] = "NO_DATA"
        base["headline"] = "Tail risk: no equity history yet."
        return base

    state = "OK" if n >= MIN_RETURNS else "INSUFFICIENT"
    sorted_r = sorted(returns)

    mean = sum(returns) / n
    var_pop = sum((r - mean) ** 2 for r in returns) / n
    std = math.sqrt(var_pop)

    down_sq = sum((min(r, 0.0)) ** 2 for r in returns) / n

    skew: float | None = None
    if n >= 3 and std > 0:
        m3 = sum((r - mean) ** 3 for r in returns) / n
        skew = m3 / (std ** 3)

    base.update(
        {
            "state": state,
            "var_95_pct": _z(_percentile_loss(sorted_r, 0.05)),
            "var_99_pct": _z(_percentile_loss(sorted_r, 0.01)),
            "cvar_95_pct": _z(_cvar(sorted_r, 0.05)),
            "annualized_vol_pct": _z(std * _ANN * 100.0),
            "downside_deviation_pct": _z(math.sqrt(down_sq) * _ANN * 100.0),
            "return_skew": _z(skew, 3) if skew is not None else None,
            "worst_day_pct": _z(min(returns) * 100.0),
            "best_day_pct": _z(max(returns) * 100.0),
            "max_consecutive_down_days": _max_down_streak(returns),
            "ulcer_index_pct": _z(_ulcer_index_pct(values)),
        }
    )

    if state == "INSUFFICIENT":
        base["headline"] = (
            f"Tail risk: verdict withheld — only {n}/{MIN_RETURNS} "
            f"daily observations."
        )
    else:
        base["headline"] = (
            f"Tail risk ({n} daily obs): 95% 1-day VaR "
            f"{base['var_95_pct']:.2f}% (CVaR {base['cvar_95_pct']:.2f}%), "
            f"99% VaR {base['var_99_pct']:.2f}%, ann.vol "
            f"{base['annualized_vol_pct']:.1f}%, worst day "
            f"{base['worst_day_pct']:.2f}%, "
            f"skew {base['return_skew'] if base['return_skew'] is not None else 'n/a'}, "
            f"max down-streak {base['max_consecutive_down_days']}d, "
            f"ulcer {base['ulcer_index_pct']:.2f}."
        )
    return base

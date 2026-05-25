"""Portfolio market-model regression — beta, alpha, R², and a regime-shift probe.

``/api/analytics`` already surfaces ``sp500_beta`` + ``sp500_correlation`` as
two bare numbers, but the trader's market-model question is richer than two
scalars:

* **Alpha** (the regression intercept) is the part of the book's return that
  is *not* explained by SPY — the desk's actual edge or drag versus a passive
  beta. The existing endpoint omits it entirely.
* **R²** says how much of the book's variance SPY actually explains. A beta
  of 1.2 with R²=0.05 is a fundamentally different book than beta 1.2 with
  R²=0.85 — one is idiosyncratic, the other is leveraged-index.
* **Beta standard error** lets the operator distinguish "we have an estimate"
  from "we have a wide CI". With ~20 daily returns, a printed beta of 1.4
  with SE 0.6 is not a number to hedge against.
* **Rolling 30d beta vs all-time** flags regime shifts — the book may have
  re-rated from defensive to leveraged this week and the bare scalar would
  smooth over the change.

Convention lock (SSOT, AGENTS.md #10 *spirit*): the daily-return series is
built **byte-identically** to ``dashboard.analytics_api`` — resample
``equity_curve`` to the *last* ``total_value`` per UTC day, simple returns
between consecutive days where the prior portfolio value AND the prior SPY
value are both > 0. ``cov`` and the two ``var``s are **population** (divide
by ``n``, not ``n-1``) to match ``analytics_api``'s exact computation so the
two endpoints never disagree on the rounded beta. ``alpha`` here is the
*daily* mean residual ``mp - beta * ms``, annualized by ``× 252`` (additive
on the percent scale, matching how the dashboard reports annualized vol).

Sample-size honesty mirrors ``build_tail_risk`` / ``build_correlation``:
numeric metrics are emitted as soon as they are mathematically defined, but
the **verdict is withheld** (``state="INSUFFICIENT"``) until ``MIN_RETURNS``
daily observations exist — a 5-day beta is noise. With no equity history at
all the state is ``NO_DATA`` and every metric is ``None``.

This is a *diagnostic / advisory* panel only. It never gates Opus and adds no
caps (AGENTS.md #2 / #12) — it is deliberately not injected into the live
decision prompt.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

MIN_RETURNS = 20
ROLLING_WINDOW = 30
# When |rolling - all-time| exceeds this many beta units, flag a regime
# shift. 0.30 is a meaningful structural re-rating (defensive→leveraged
# crosses ~0.3 alone) without firing on noise from a 30-day re-sample.
REGIME_SHIFT_THRESHOLD = 0.30

_TRADING_DAYS = 252


def _z(v: float | None, ndigits: int = 2) -> float | None:
    """Round, fold -0.0 → 0.0 so the JSON never carries a signed zero."""
    if v is None:
        return None
    r = round(v, ndigits)
    return 0.0 if r == 0 else r


def _paired_daily_returns(
    equity_curve: list[dict],
) -> tuple[list[float], list[float]]:
    """(port_returns, spy_returns) — paired daily-resampled returns.

    Walk the curve, take the last total_value AND last sp500_price per UTC
    day, then compute simple returns on consecutive days where both legs
    have a positive prior value. Byte-identical to ``analytics_api``'s 4d
    block so the two cannot drift on the same input.
    """
    pf_by_day: dict[str, float] = {}
    sp_by_day: dict[str, float] = {}
    for p in equity_curve or []:
        day = (p.get("timestamp") or "")[:10]
        if not day:
            continue
        tv = p.get("total_value")
        if tv is not None:
            pf_by_day[day] = tv
        sx = p.get("sp500_price")
        if sx:
            sp_by_day[day] = sx
    day_keys = sorted(pf_by_day.keys())
    port_ret: list[float] = []
    spx_ret: list[float] = []
    for i in range(1, len(day_keys)):
        d0, d1 = day_keys[i - 1], day_keys[i]
        if d0 in sp_by_day and d1 in sp_by_day:
            pv0 = pf_by_day[d0]
            sv0 = sp_by_day[d0]
            pv1 = pf_by_day[d1]
            sv1 = sp_by_day[d1]
            if pv0 and pv0 > 0 and sv0 > 0 and pv1 is not None:
                port_ret.append(pv1 / pv0 - 1.0)
                spx_ret.append(sv1 / sv0 - 1.0)
    return port_ret, spx_ret


def _regress(
    port: list[float], spx: list[float]
) -> tuple[float | None, float | None, float | None, float | None]:
    """OLS beta + daily-alpha + R² + SE(beta) on paired returns.

    Returns ``(beta, alpha_daily, r_squared, beta_stderr)`` or ``None``s
    when the regression is undefined (zero SPY variance, n < 2, etc.).
    Population variance to match the analytics_api convention.
    """
    n = len(port)
    if n < 2 or len(spx) != n:
        return None, None, None, None
    mp = sum(port) / n
    ms = sum(spx) / n
    cov = sum((port[i] - mp) * (spx[i] - ms) for i in range(n)) / n
    var_s = sum((s - ms) ** 2 for s in spx) / n
    var_p = sum((p - mp) ** 2 for p in port) / n
    if var_s <= 0:
        return None, None, None, None
    beta = cov / var_s
    alpha_daily = mp - beta * ms
    r_sq = None
    if var_p > 0:
        r_sq = max(0.0, min(1.0, (cov * cov) / (var_s * var_p)))
    # Population residual variance → SE(beta) = sqrt(var_resid / (n * var_s)).
    # Matches OLS SE under the population-variance convention chosen above;
    # with n-1 normalisation the two factors of (n-1)/n cancel anyway, but
    # we lock the SSOT to population so the math here matches /api/analytics.
    # Floor at 0 so a perfect-fit float-noise residual doesn't print a
    # negative SE; 0 is the honest "no residual dispersion left" value.
    resid_var = max(0.0, var_p - beta * cov)
    se_beta = math.sqrt(resid_var / (n * var_s))
    return beta, alpha_daily, r_sq, se_beta


def build_portfolio_beta(
    equity_curve: list[dict],
    now: datetime | None = None,
    rolling_window: int = ROLLING_WINDOW,
) -> dict:
    """Build the full portfolio market-model report."""
    now = now or datetime.now(timezone.utc)
    port, spx = _paired_daily_returns(equity_curve)
    n = len(port)

    base = {
        "as_of": now.isoformat(timespec="seconds"),
        "n_returns": n,
        "min_returns": MIN_RETURNS,
        "rolling_window": rolling_window,
        "beta": None,
        "alpha_daily_pct": None,
        "alpha_annualized_pct": None,
        "r_squared": None,
        "beta_stderr": None,
        "rolling_beta": None,
        "rolling_n": 0,
        "regime_shift": False,
        "regime_shift_delta": None,
    }

    if n == 0:
        base["state"] = "NO_DATA"
        base["headline"] = "Portfolio beta: no paired daily returns yet."
        return base

    state = "OK" if n >= MIN_RETURNS else "INSUFFICIENT"

    beta, alpha_d, r_sq, se = _regress(port, spx)
    rolling_beta = None
    rolling_n = 0
    regime_shift = False
    regime_delta: float | None = None
    if rolling_window > 1 and n >= rolling_window:
        r_port = port[-rolling_window:]
        r_spx = spx[-rolling_window:]
        rb, _, _, _ = _regress(r_port, r_spx)
        if rb is not None:
            rolling_beta = rb
            rolling_n = rolling_window
            if beta is not None:
                regime_delta = rb - beta
                regime_shift = abs(regime_delta) >= REGIME_SHIFT_THRESHOLD

    alpha_ann = alpha_d * _TRADING_DAYS if alpha_d is not None else None

    base.update(
        {
            "state": state,
            "beta": _z(beta, 3) if beta is not None else None,
            "alpha_daily_pct": _z(alpha_d * 100.0, 4)
            if alpha_d is not None else None,
            "alpha_annualized_pct": _z(alpha_ann * 100.0, 2)
            if alpha_ann is not None else None,
            "r_squared": _z(r_sq, 3) if r_sq is not None else None,
            "beta_stderr": _z(se, 3) if se is not None else None,
            "rolling_beta": _z(rolling_beta, 3)
            if rolling_beta is not None else None,
            "rolling_n": rolling_n,
            "regime_shift": regime_shift,
            "regime_shift_delta": _z(regime_delta, 3)
            if regime_delta is not None else None,
        }
    )

    if state == "INSUFFICIENT":
        base["headline"] = (
            f"Portfolio beta: verdict withheld — only {n}/{MIN_RETURNS} "
            "paired daily returns."
        )
    elif base["beta"] is None:
        # SPY series is constant ⇒ regression undefined. Honest one-liner;
        # the rest of the fields are correctly None and would format-crash.
        base["headline"] = (
            f"Portfolio beta ({n} paired obs): SPY variance is zero — "
            "beta undefined."
        )
    else:
        b = base["beta"]
        r2 = base["r_squared"]
        aa = base["alpha_annualized_pct"]
        sb = base["beta_stderr"]
        rb = base["rolling_beta"]
        parts = [
            f"Portfolio beta ({n} paired obs): β={b:.2f}",
        ]
        if sb is not None:
            parts.append(f"±{sb:.2f} SE")
        if r2 is not None:
            parts.append(f"R²={r2:.2f}")
        if aa is not None:
            parts.append(f"α(ann)={aa:+.2f}%")
        if rb is not None:
            parts.append(f"rolling β({rolling_n}d)={rb:.2f}")
        if regime_shift and regime_delta is not None:
            parts.append(f"REGIME SHIFT Δ={regime_delta:+.2f}")
        base["headline"] = ", ".join(parts) + "."
    return base


if __name__ == "__main__":  # CLI parity with drawdown / benchmark
    import json
    import sqlite3
    import sys
    from pathlib import Path

    db = Path(__file__).resolve().parents[2] / "data" / "paper_trader.db"
    try:
        c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        eq = [
            {"timestamp": r[0], "total_value": r[1], "cash": r[2],
             "sp500_price": r[3]}
            for r in c.execute(
                "SELECT timestamp,total_value,cash,sp500_price FROM "
                "equity_curve ORDER BY timestamp ASC, id ASC").fetchall()
        ]
        c.close()
    except Exception as e:
        print(f"portfolio_beta: cannot read {db}: {e}")
        sys.exit(2)
    rep = build_portfolio_beta(eq)
    if "--json" in sys.argv:
        print(json.dumps(rep, indent=2, default=str))
    else:
        print(rep["headline"])

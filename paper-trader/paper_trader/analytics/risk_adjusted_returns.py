"""Risk-adjusted returns vs the S&P 500 — Sharpe, Sortino, information ratio.

The single desk question this answers: *is the bot's return alpha worth
the volatility it took on?* — the risk-aware companion to
``/api/benchmark`` (dollar alpha, blind to path). A book that beat the
index by +5pp with double the volatility may be worse on risk-adjusted
terms; ``/api/benchmark`` cannot see that.

Distinct from every neighbour (AGENTS.md invariant #10 — do not
consolidate):

* ``/api/benchmark`` — cumulative DOLLAR alpha vs S&P at every cycle.
  Path-blind: a smooth +3pp climb and a +3pp finish reached by
  -10pp drawdown and recovery look identical.
* ``/api/analytics`` — *scalar* ``sharpe_annualized`` /
  ``sortino_annualized`` for the portfolio only. No SPY-parity
  comparison, no information ratio, no verdict, no headline.
* ``/api/drawdown`` — peak-to-trough depth + recovery%. Volatility-blind
  (a steep V and a slow drift to the same trough read identical).
* ``/api/tail-risk`` — VaR / CVaR / skew on the *daily-return distribution*.
  A left-tail diagnostic; doesn't condense to "are you ahead on
  risk-adjusted terms".

This module composes:

  * Daily-return series for the portfolio (last EOD ``total_value`` per
    UTC date, identical to ``analytics_api``'s ``by_day`` loop so the
    two never drift).
  * Daily-return series for the S&P 500 (``sp500_price`` on the same
    dates — paired observations only, so a partial S&P day drops out
    of both series, not just one).
  * Annualized Sharpe = mean(returns) / std(returns) × √252.
  * Annualized Sortino = mean(returns) / downside_std × √252.
  * Information ratio = mean(active_return) / std(active_return) × √252,
    where active_return = portfolio_return − benchmark_return per day.

Sample-size honesty mirrors ``benchmark.py`` and ``trade_asymmetry``:
numeric metrics are still emitted, but the **verdict** is withheld
until ``STABLE`` (``n_paired_days >= STABLE_MIN_DAYS``). A 3-day Sharpe
is statistically meaningless and surfacing a verdict from it would
just whipsaw on every new day's return.

Verdict ladder (STABLE only):

* ``OUTPERFORMING_RISK_ADJUSTED`` — Sharpe alpha (port − S&P) ≥
  ``ALPHA_BAND``. The book beats the index in risk-adjusted terms.
* ``LAGGING_RISK_ADJUSTED``       — Sharpe alpha ≤ -``ALPHA_BAND``.
  The book loses to the index in risk-adjusted terms.
* ``TRACKING_RISK_ADJUSTED``      — |Sharpe alpha| < ``ALPHA_BAND``.
  The book is risk-adjusted-flat to the index.

Advisory only — never gates Opus, adds no caps (AGENTS.md invariants
#2/#12). Pure & network-free: a walk of the ``equity_curve`` rows the
caller already read (the ``benchmark.py`` / ``drawdown.py`` split).
"""
from __future__ import annotations

from datetime import datetime, timezone

# Sample-size gate: a Sharpe ratio computed on fewer than this many
# paired daily returns is noise. Identical idea to ``benchmark._MIN_POINTS``
# but at a paired-daily-return granularity. The existing
# ``analytics_api`` uses ``>= 5`` to even compute Sharpe — we keep that
# as the LOWER bound for emitting numerics, then require this many
# before emitting a verdict.
EMIT_MIN_DAYS = 5
STABLE_MIN_DAYS = 7      # one full trading week

# Risk-adjusted-alpha band: |port_sharpe − sp_sharpe| ≤ this is TRACKING.
ALPHA_BAND = 0.25

# Trading days per year for annualization — the equity_curve writes
# include weekends + holidays for any cycle that fires, but the index
# arm of the comparison annualizes by the same √252 either way, so the
# RELATIVE Sharpe is what carries signal. Documented constant.
_ANNUALIZATION_DAYS = 252


def _round(x: float | None, n: int = 4) -> float | None:
    if x is None:
        return None
    try:
        return round(float(x), n)
    except (TypeError, ValueError):
        return None


def _by_day_last(equity_curve: list[dict], field: str) -> dict[str, float]:
    """Last-write-wins per UTC date for ``field`` (the ``analytics_api``
    by-day convention). Rows missing/0/non-positive value are skipped.
    Garbage rows degrade — never raise."""
    out: dict[str, float] = {}
    for p in (equity_curve or []):
        try:
            ts = p.get("timestamp")
            if not ts:
                continue
            day = ts[:10]
            if not day:
                continue
            v = p.get(field)
            if v is None:
                continue
            fv = float(v)
            if fv <= 0:
                continue
            out[day] = fv  # last write wins → EOD close
        except (TypeError, ValueError, AttributeError):
            continue
    return out


def _stddev(xs: list[float]) -> float:
    """Population stddev (matches ``analytics_api`` Sharpe convention).
    Returns 0.0 on empty/single-point input."""
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def _downside_stddev(xs: list[float]) -> float:
    """Sortino denominator — RMS of negative returns scaled by FULL
    sample size (matches ``analytics_api``'s sortino_annualized:
    ``dvar = sum(r*r for r in downside) / len(daily_returns)``).
    Returns 0.0 if no downside or input is empty."""
    if not xs:
        return 0.0
    downside = [x for x in xs if x < 0]
    if not downside:
        return 0.0
    return (sum(r * r for r in downside) / len(xs)) ** 0.5


def _annualize_sharpe(returns: list[float]) -> float | None:
    if len(returns) < EMIT_MIN_DAYS:
        return None
    m = sum(returns) / len(returns)
    s = _stddev(returns)
    if s <= 0:
        return None
    return (m / s) * (_ANNUALIZATION_DAYS ** 0.5)


def _annualize_sortino(returns: list[float]) -> float | None:
    if len(returns) < EMIT_MIN_DAYS:
        return None
    m = sum(returns) / len(returns)
    ds = _downside_stddev(returns)
    if ds <= 0:
        return None
    return (m / ds) * (_ANNUALIZATION_DAYS ** 0.5)


def build_risk_adjusted_returns(equity_curve: list[dict]) -> dict:
    """Risk-adjusted return diagnostics for the account vs the S&P 500.

    Args:
        equity_curve: chronological list of
            ``{timestamp, total_value, cash, sp500_price}`` — exactly
            ``store.equity_curve(...)``'s shape (the same input
            ``build_benchmark`` and ``compute_drawdown`` take).

    ``state`` ∈ ``NO_DATA`` → ``INSUFFICIENT`` (fewer than
    ``EMIT_MIN_DAYS`` paired daily returns; numerics None but the dict
    shape is complete) → ``EMITTING`` (numerics present, verdict
    withheld below ``STABLE_MIN_DAYS``) → ``STABLE`` (verdict emitted).

    Pure, deterministic, never raises.
    """
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    base = {
        "as_of": now,
        "state": "NO_DATA",
        "verdict": None,
        "headline": ("No equity history yet — risk-adjusted return "
                     "unscorable."),
        "n_paired_days": 0,
        "n_port_days": 0,
        "n_sp500_days": 0,
        "port_sharpe": None,
        "port_sortino": None,
        "sp500_sharpe": None,
        "sp500_sortino": None,
        "sharpe_alpha": None,        # port_sharpe − sp500_sharpe
        "sortino_alpha": None,
        "information_ratio": None,
        "port_mean_daily_pct": None,
        "port_stddev_daily_pct": None,
        "sp500_mean_daily_pct": None,
        "sp500_stddev_daily_pct": None,
        "first_day": None,
        "last_day": None,
        "thresholds": {
            "EMIT_MIN_DAYS": EMIT_MIN_DAYS,
            "STABLE_MIN_DAYS": STABLE_MIN_DAYS,
            "ALPHA_BAND": ALPHA_BAND,
        },
    }

    port_by_day = _by_day_last(equity_curve, "total_value")
    sp_by_day = _by_day_last(equity_curve, "sp500_price")
    base["n_port_days"] = len(port_by_day)
    base["n_sp500_days"] = len(sp_by_day)

    if not port_by_day:
        return base

    # Paired-day series: both port AND sp500 must exist on the day AND
    # on the prior day for a return to enter either series (so neither
    # leg drifts on a missing day, the analytics_api by_day convention).
    day_keys = sorted(set(port_by_day) & set(sp_by_day))
    port_returns: list[float] = []
    sp_returns: list[float] = []
    active_returns: list[float] = []  # port − sp per day, for info ratio
    for i in range(1, len(day_keys)):
        d0, d1 = day_keys[i - 1], day_keys[i]
        pv0, pv1 = port_by_day[d0], port_by_day[d1]
        sv0, sv1 = sp_by_day[d0], sp_by_day[d1]
        if pv0 <= 0 or sv0 <= 0:
            continue
        pr = pv1 / pv0 - 1.0
        sr = sv1 / sv0 - 1.0
        port_returns.append(pr)
        sp_returns.append(sr)
        active_returns.append(pr - sr)

    n = len(port_returns)
    base["n_paired_days"] = n
    if day_keys:
        base["first_day"] = day_keys[0]
        base["last_day"] = day_keys[-1]

    if n < EMIT_MIN_DAYS:
        base["state"] = "INSUFFICIENT"
        base["headline"] = (
            f"Risk-adjusted view maturing — {n} paired daily returns "
            f"(need ≥{EMIT_MIN_DAYS} to emit Sharpe/Sortino, "
            f"≥{STABLE_MIN_DAYS} for a verdict)."
        )
        return base

    # Numerics — emitted at INSUFFICIENT_FOR_VERDICT (state=EMITTING)
    # and STABLE alike. The verdict is the only gated artifact.
    pm = sum(port_returns) / n
    ps = _stddev(port_returns)
    sm = sum(sp_returns) / n
    ss = _stddev(sp_returns)
    p_sharpe = _annualize_sharpe(port_returns)
    p_sortino = _annualize_sortino(port_returns)
    s_sharpe = _annualize_sharpe(sp_returns)
    s_sortino = _annualize_sortino(sp_returns)

    sharpe_alpha = None
    if p_sharpe is not None and s_sharpe is not None:
        sharpe_alpha = p_sharpe - s_sharpe
    sortino_alpha = None
    if p_sortino is not None and s_sortino is not None:
        sortino_alpha = p_sortino - s_sortino

    # Information ratio = mean(active) / std(active) × √252. The
    # canonical risk-adjusted-active-return number from Grinold.
    info_ratio = None
    if active_returns:
        am = sum(active_returns) / len(active_returns)
        as_ = _stddev(active_returns)
        if as_ > 0:
            info_ratio = (am / as_) * (_ANNUALIZATION_DAYS ** 0.5)

    base.update({
        "port_sharpe": _round(p_sharpe, 4),
        "port_sortino": _round(p_sortino, 4),
        "sp500_sharpe": _round(s_sharpe, 4),
        "sp500_sortino": _round(s_sortino, 4),
        "sharpe_alpha": _round(sharpe_alpha, 4),
        "sortino_alpha": _round(sortino_alpha, 4),
        "information_ratio": _round(info_ratio, 4),
        "port_mean_daily_pct": _round(pm * 100.0, 4),
        "port_stddev_daily_pct": _round(ps * 100.0, 4),
        "sp500_mean_daily_pct": _round(sm * 100.0, 4),
        "sp500_stddev_daily_pct": _round(ss * 100.0, 4),
    })

    if n < STABLE_MIN_DAYS:
        base["state"] = "EMITTING"
        prov = ""
        if p_sharpe is not None and s_sharpe is not None:
            prov = (f" Provisional: port Sharpe {p_sharpe:+.2f} vs S&P "
                    f"{s_sharpe:+.2f} (alpha {sharpe_alpha:+.2f}).")
        base["headline"] = (
            f"Numerics emitting — {n} paired daily returns; verdict "
            f"withheld until ≥{STABLE_MIN_DAYS} days.{prov}"
        )
        return base

    base["state"] = "STABLE"
    if sharpe_alpha is None:
        # Sharpe undefined on either leg (zero variance). Rare —
        # degrades cleanly to TRACKING_RISK_ADJUSTED.
        verdict = "TRACKING_RISK_ADJUSTED"
    elif sharpe_alpha > ALPHA_BAND:
        verdict = "OUTPERFORMING_RISK_ADJUSTED"
    elif sharpe_alpha < -ALPHA_BAND:
        verdict = "LAGGING_RISK_ADJUSTED"
    else:
        verdict = "TRACKING_RISK_ADJUSTED"
    base["verdict"] = verdict

    sp_clause = (f"port Sharpe {p_sharpe:+.2f} vs S&P {s_sharpe:+.2f}"
                 if (p_sharpe is not None and s_sharpe is not None)
                 else "Sharpe undefined")
    so_clause = ""
    if p_sortino is not None and s_sortino is not None:
        so_clause = (f"; port Sortino {p_sortino:+.2f} vs S&P "
                     f"{s_sortino:+.2f}")
    ir_clause = ""
    if info_ratio is not None:
        ir_clause = f"; information ratio {info_ratio:+.2f}"

    if verdict == "OUTPERFORMING_RISK_ADJUSTED":
        headline = (
            f"Outperforming the S&P 500 on risk-adjusted terms — "
            f"{sp_clause} (alpha {sharpe_alpha:+.2f}, band ±"
            f"{ALPHA_BAND}){so_clause}{ir_clause}."
        )
    elif verdict == "LAGGING_RISK_ADJUSTED":
        headline = (
            f"Lagging the S&P 500 on risk-adjusted terms — "
            f"{sp_clause} (alpha {sharpe_alpha:+.2f}, band ±"
            f"{ALPHA_BAND}){so_clause}{ir_clause}."
        )
    else:
        headline = (
            f"Tracking the S&P 500 on risk-adjusted terms — "
            f"{sp_clause} (alpha "
            f"{(sharpe_alpha or 0):+.2f}, within ±"
            f"{ALPHA_BAND} band){so_clause}{ir_clause}."
        )
    base["headline"] = headline
    return base


if __name__ == "__main__":  # the benchmark.py / drawdown.py CLI precedent
    import json
    import sqlite3
    import sys
    from pathlib import Path

    db = Path(__file__).resolve().parents[2] / "data" / "paper_trader.db"
    try:
        c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        rows = [
            {"timestamp": r[0], "total_value": r[1], "cash": r[2],
             "sp500_price": r[3]}
            for r in c.execute(
                "SELECT timestamp,total_value,cash,sp500_price FROM "
                "equity_curve ORDER BY timestamp ASC, id ASC").fetchall()
        ]
        c.close()
    except Exception as e:
        print(f"risk_adjusted_returns: cannot read {db}: {e}")
        sys.exit(2)

    rep = build_risk_adjusted_returns(rows)
    if "--json" in sys.argv:
        print(json.dumps(rep, indent=2, default=str))
    else:
        tag = rep["state"] + (f"/{rep['verdict']}" if rep["verdict"] else "")
        print(f"RISK-ADJUSTED RETURNS  [{tag}]  {rep['headline']}")
        if rep["state"] not in ("NO_DATA",):
            print(f"  paired days     : {rep['n_paired_days']}  "
                  f"({rep['first_day']} → {rep['last_day']})")
            if rep["port_sharpe"] is not None:
                print(f"  port Sharpe     : {rep['port_sharpe']:+.2f}   "
                      f"port Sortino : "
                      f"{(rep['port_sortino'] or 0):+.2f}")
                print(f"  S&P  Sharpe     : "
                      f"{(rep['sp500_sharpe'] or 0):+.2f}   "
                      f"S&P  Sortino : "
                      f"{(rep['sp500_sortino'] or 0):+.2f}")
                print(f"  sharpe alpha    : "
                      f"{(rep['sharpe_alpha'] or 0):+.2f}   "
                      f"info ratio  : "
                      f"{(rep['information_ratio'] or 0):+.2f}")
                print(f"  daily port mean : "
                      f"{(rep['port_mean_daily_pct'] or 0):+.4f}%   "
                      f"daily port σ : "
                      f"{(rep['port_stddev_daily_pct'] or 0):.4f}%")
                print(f"  daily S&P  mean : "
                      f"{(rep['sp500_mean_daily_pct'] or 0):+.4f}%   "
                      f"daily S&P  σ : "
                      f"{(rep['sp500_stddev_daily_pct'] or 0):.4f}%")

"""Black-Scholes Greeks for open option positions.

Provides per-leg + portfolio-level delta / gamma / theta / vega aggregation.
yfinance gives us implied vol per contract, so we never invert BS for IV.
A flat 4.5% risk-free rate is used (matches 2026 ~3M T-bill range).
"""
from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

# Standard-normal density / CDF using the math library (no scipy dep).
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _norm_pdf(x: float) -> float:
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


RISK_FREE_RATE = 0.045  # flat 4.5% — close enough for sub-90d options
DEFAULT_IV = 0.45  # fallback when yfinance returns 0/None for IV (mid-cap-ish semis)


def _years_to_expiry(expiry: str) -> float:
    """Return time-to-expiry in years; clipped to a small positive floor."""
    try:
        exp = date.fromisoformat(expiry)
    except Exception:
        return 1.0 / 365.0
    today = date.today()
    days = max((exp - today).days, 0)
    # Floor at 1 day so Greeks don't divide by zero on expiration day.
    return max(days, 1) / 365.0


def bs_greeks(S: float, K: float, T: float, sigma: float,
              r: float = RISK_FREE_RATE, opt_type: str = "call") -> dict[str, float]:
    """Per-contract Greeks. Delta is per-share; multiply by 100*qty for per-position.

    Returns:
      delta (-1..1), gamma (>0), theta (per-day, negative for long), vega (per 1% IV),
      price (BS theoretical), iv (the sigma we used).
    """
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0,
                "price": 0.0, "iv": sigma}

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    pdf_d1 = _norm_pdf(d1)
    N_d1 = _norm_cdf(d1)
    N_d2 = _norm_cdf(d2)

    if opt_type == "call":
        delta = N_d1
        theta_year = (-(S * pdf_d1 * sigma) / (2 * sqrtT)
                      - r * K * math.exp(-r * T) * N_d2)
        price = S * N_d1 - K * math.exp(-r * T) * N_d2
    else:  # put
        delta = N_d1 - 1.0
        theta_year = (-(S * pdf_d1 * sigma) / (2 * sqrtT)
                      + r * K * math.exp(-r * T) * _norm_cdf(-d2))
        price = K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)

    gamma = pdf_d1 / (S * sigma * sqrtT)
    vega_per_vol = S * pdf_d1 * sqrtT  # change in price per 1.0 change in sigma
    theta_per_day = theta_year / 365.0
    vega_per_pct = vega_per_vol / 100.0  # per 1% absolute change in IV

    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta_per_day,
        "vega": vega_per_pct,
        "price": price,
        "iv": sigma,
    }


def _underlying_iv(ticker: str, expiry: str, strike: float, opt_type: str) -> float:
    """Look up implied vol from the live options chain. Falls back to DEFAULT_IV."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        chain = t.option_chain(expiry)
        df = chain.calls if opt_type == "call" else chain.puts
        row = df[df["strike"] == strike]
        if not row.empty:
            iv = float(row["impliedVolatility"].iloc[0])
            if 0.05 <= iv <= 5.0:
                return iv
    except Exception:
        pass
    return DEFAULT_IV


def compute_position_greeks(positions: list[dict], price_lookup: dict[str, float] | None = None
                            ) -> dict[str, Any]:
    """Aggregate Greeks across an open-position list.

    `positions` rows must include: ticker, type ("call" | "put" | "stock"),
    qty, strike, expiry, current_price.
    `price_lookup` maps ticker → spot. If absent, we fetch via market.get_price.

    Returns:
      {
        "positions": [ {ticker, type, qty, expiry, strike, iv, days_to_expiry,
                        delta, gamma, theta, vega, notional}, ... ],
        "totals": {delta, gamma, theta, vega,
                   notional_long, notional_short, leverage_ratio},
        "as_of": iso8601,
      }
    """
    from datetime import timezone as _tz

    # Lazy import to avoid a cycle when this module is imported by market.
    if price_lookup is None:
        try:
            from ..market import get_prices
            tickers = sorted({p["ticker"] for p in positions
                              if p.get("type") in ("call", "put", "stock")})
            price_lookup = get_prices(tickers) if tickers else {}
        except Exception:
            price_lookup = {}

    out_positions: list[dict] = []
    tot_delta = tot_gamma = tot_theta = tot_vega = 0.0
    notional_long = notional_short = 0.0

    for p in positions:
        ttype = p.get("type") or "stock"
        ticker = p["ticker"]
        qty = float(p.get("qty") or 0)
        spot = price_lookup.get(ticker) or p.get("current_price") or 0.0
        if not spot or spot <= 0:
            continue

        if ttype == "stock":
            # Stock = pure delta 1.0 per share. No gamma/theta/vega.
            d = qty  # 1 delta per share
            notional = qty * spot
            tot_delta += d
            if qty >= 0:
                notional_long += notional
            else:
                notional_short += -notional
            out_positions.append({
                "ticker": ticker, "type": "stock", "qty": qty,
                "expiry": None, "strike": None,
                "days_to_expiry": None, "iv": None,
                "delta": round(d, 4), "gamma": 0.0, "theta": 0.0, "vega": 0.0,
                "notional": round(notional, 2),
            })
            continue

        if ttype not in ("call", "put"):
            continue
        strike = float(p.get("strike") or 0)
        expiry = p.get("expiry") or ""
        if not strike or not expiry:
            continue
        T = _years_to_expiry(expiry)
        days_to_expiry = max(int(T * 365), 0)
        iv = _underlying_iv(ticker, expiry, strike, ttype)
        g = bs_greeks(spot, strike, T, iv, opt_type=ttype)

        # Each option contract = 100 shares. qty is # of contracts.
        mult = 100.0 * qty
        leg_delta = g["delta"] * mult
        leg_gamma = g["gamma"] * mult
        leg_theta = g["theta"] * mult
        leg_vega = g["vega"] * mult
        # Notional = delta * spot (delta-equivalent dollar exposure)
        notional = abs(leg_delta) * spot
        if leg_delta >= 0:
            notional_long += notional
        else:
            notional_short += notional

        tot_delta += leg_delta
        tot_gamma += leg_gamma
        tot_theta += leg_theta
        tot_vega += leg_vega

        out_positions.append({
            "ticker": ticker, "type": ttype, "qty": qty,
            "expiry": expiry, "strike": strike,
            "days_to_expiry": days_to_expiry, "iv": round(iv, 4),
            "delta": round(leg_delta, 4),
            "gamma": round(leg_gamma, 6),
            "theta": round(leg_theta, 4),
            "vega": round(leg_vega, 4),
            "notional": round(notional, 2),
        })

    gross = notional_long + notional_short
    return {
        "positions": out_positions,
        "totals": {
            "delta": round(tot_delta, 4),
            "gamma": round(tot_gamma, 6),
            "theta": round(tot_theta, 4),
            "vega": round(tot_vega, 4),
            "notional_long": round(notional_long, 2),
            "notional_short": round(notional_short, 2),
            "gross_notional": round(gross, 2),
        },
        "as_of": datetime.now(_tz.utc).isoformat(timespec="seconds"),
    }


if __name__ == "__main__":
    # smoke
    g = bs_greeks(100, 100, 30 / 365, 0.45, opt_type="call")
    print(f"ATM 30d call: delta={g['delta']:.3f} gamma={g['gamma']:.5f} "
          f"theta={g['theta']:.4f} vega={g['vega']:.4f} px={g['price']:.2f}")

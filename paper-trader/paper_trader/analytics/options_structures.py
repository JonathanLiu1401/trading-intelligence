"""Advanced multi-leg option structures for the paper trader.

Execution helpers for:
  - covered calls / cash-secured puts (short opens with collateral)
  - debit/credit vertical spreads (defined risk)
  - buy-to-cover short options

All helpers are pure validation/pricing; strategy._execute owns store writes.
"""
from __future__ import annotations

from typing import Any

MULT = 100.0

STRUCTURE_ACTIONS = {
    "BUY_CALL_SPREAD",
    "BUY_PUT_SPREAD",
    "SELL_CALL_SPREAD",
    "SELL_PUT_SPREAD",
}

OPTION_OPEN_SELL_ACTIONS = {"SELL_CALL", "SELL_PUT"}
OPTION_BUY_ACTIONS = {"BUY_CALL", "BUY_PUT"}


def _f(x: Any, default: float | None = None) -> float | None:
    try:
        if x is None:
            return default
        v = float(x)
        return v
    except (TypeError, ValueError):
        return default


def _leg_price(market, ticker: str, expiry: str, strike: float, otype: str) -> float | None:
    try:
        px = market.get_option_price(ticker, expiry, strike, otype)
    except Exception:
        return None
    if px is None:
        return None
    try:
        px_f = float(px)
    except (TypeError, ValueError):
        return None
    if px_f <= 0:
        return None
    return px_f


def stock_long_shares(positions, ticker: str) -> float:
    total = 0.0
    for p in positions or []:
        if str(p.get("ticker") or "").upper() != ticker:
            continue
        if p.get("type") != "stock":
            continue
        q = _f(p.get("qty"), 0.0) or 0.0
        if q > 0:
            total += q
    return total


def matching_option_lots(positions, ticker, otype, strike=None, expiry=None, side="long"):
    """Return open option lots matching filters.

    side: 'long' (qty>0), 'short' (qty<0), 'any'
    """
    out = []
    strike_f = _f(strike) if strike is not None else None
    for p in positions or []:
        if str(p.get("ticker") or "").upper() != ticker:
            continue
        if p.get("type") != otype:
            continue
        q = _f(p.get("qty"), 0.0) or 0.0
        if side == "long" and q <= 0:
            continue
        if side == "short" and q >= 0:
            continue
        if strike_f is not None:
            ps = _f(p.get("strike"))
            if ps is None or abs(ps - strike_f) > 1e-9:
                continue
        if expiry and str(p.get("expiry") or "")[:10] != str(expiry)[:10]:
            continue
        out.append(p)
    return out


def classify_sell_option(decision, snapshot) -> dict:
    """Decide whether SELL_CALL/SELL_PUT is close-long, covered/CSP open, or blocked."""
    action = str(decision.get("action") or "").upper()
    ticker = str(decision.get("ticker") or "").upper()
    otype = "call" if action == "SELL_CALL" else "put"
    qty = abs(_f(decision.get("qty"), 0.0) or 0.0)
    strike = _f(decision.get("strike"))
    expiry = decision.get("expiry")
    if qty <= 0:
        return {"mode": "blocked", "reason": "qty must be > 0"}
    if strike is None or not expiry:
        # Close path may omit if unique; open path requires both
        longs = matching_option_lots(snapshot.get("positions"), ticker, otype, side="long")
        if len(longs) == 1:
            return {
                "mode": "close_long",
                "otype": otype,
                "qty": qty,
                "match": longs[0],
                "strike": _f(longs[0].get("strike")),
                "expiry": longs[0].get("expiry"),
            }
        if not longs:
            return {"mode": "blocked", "reason": "open short requires strike+expiry"}
        return {"mode": "blocked", "reason": "ambiguous long close; specify strike+expiry"}

    longs = matching_option_lots(
        snapshot.get("positions"), ticker, otype, strike=strike, expiry=expiry, side="long"
    )
    if longs:
        return {
            "mode": "close_long",
            "otype": otype,
            "qty": qty,
            "match": longs[0],
            "strike": strike,
            "expiry": expiry,
        }

    # Open short
    if otype == "call":
        shares = stock_long_shares(snapshot.get("positions"), ticker)
        need = qty * MULT
        if shares + 1e-6 < need:
            return {
                "mode": "blocked",
                "reason": (
                    "naked short call blocked; need %.0f long shares for covered call "
                    "(have %.4g)" % (need, shares)
                ),
            }
        return {
            "mode": "open_covered_call",
            "otype": otype,
            "qty": qty,
            "strike": strike,
            "expiry": expiry,
            "shares_cover": shares,
        }

    # CSP
    return {
        "mode": "open_csp",
        "otype": otype,
        "qty": qty,
        "strike": strike,
        "expiry": expiry,
    }


def classify_buy_option(decision, snapshot) -> dict:
    """BUY_CALL/BUY_PUT: cover short if present, else open long."""
    action = str(decision.get("action") or "").upper()
    ticker = str(decision.get("ticker") or "").upper()
    otype = "call" if action == "BUY_CALL" else "put"
    qty = abs(_f(decision.get("qty"), 0.0) or 0.0)
    strike = _f(decision.get("strike"))
    expiry = decision.get("expiry")
    if qty <= 0:
        return {"mode": "blocked", "reason": "qty must be > 0"}
    if strike is None or not expiry:
        return {"mode": "blocked", "reason": "option trade missing strike/expiry"}
    shorts = matching_option_lots(
        snapshot.get("positions"), ticker, otype, strike=strike, expiry=expiry, side="short"
    )
    if shorts:
        held = abs(_f(shorts[0].get("qty"), 0.0) or 0.0)
        return {
            "mode": "close_short",
            "otype": otype,
            "qty": qty,
            "match": shorts[0],
            "held_short": held,
            "strike": strike,
            "expiry": expiry,
        }
    return {
        "mode": "open_long",
        "otype": otype,
        "qty": qty,
        "strike": strike,
        "expiry": expiry,
    }


def parse_vertical_spread(decision) -> dict:
    """Normalize a vertical spread decision.

    Accepted shapes:
      action=BUY_CALL_SPREAD|BUY_PUT_SPREAD|SELL_CALL_SPREAD|SELL_PUT_SPREAD
      ticker, qty, expiry,
      long_strike + short_strike  OR  strike (long) + width  OR legs[]
    """
    action = str(decision.get("action") or "").upper()
    if action not in STRUCTURE_ACTIONS:
        return {"ok": False, "reason": "not a spread action"}
    ticker = str(decision.get("ticker") or "").upper()
    qty = abs(_f(decision.get("qty"), 0.0) or 0.0)
    expiry = decision.get("expiry")
    if not ticker or qty <= 0 or not expiry:
        return {"ok": False, "reason": "spread needs ticker, qty>0, expiry"}

    is_call = "CALL" in action
    otype = "call" if is_call else "put"
    is_debit = action.startswith("BUY_")

    long_k = _f(decision.get("long_strike"))
    short_k = _f(decision.get("short_strike"))
    width = _f(decision.get("width"))
    base_k = _f(decision.get("strike"))

    legs_in = decision.get("legs")
    if isinstance(legs_in, list) and len(legs_in) >= 2:
        # Expect one buy and one sell same type/expiry
        buys = []
        sells = []
        for leg in legs_in:
            la = str(leg.get("action") or leg.get("side") or "").upper()
            lk = _f(leg.get("strike"))
            if lk is None:
                continue
            if la in {"BUY", "LONG", "BUY_CALL", "BUY_PUT"}:
                buys.append(lk)
            elif la in {"SELL", "SHORT", "SELL_CALL", "SELL_PUT"}:
                sells.append(lk)
        if len(buys) != 1 or len(sells) != 1:
            return {"ok": False, "reason": "legs must include exactly one buy and one sell"}
        long_k, short_k = buys[0], sells[0]

    if long_k is None and base_k is not None and width is not None and width > 0:
        if is_call:
            # debit call: long lower, short higher; credit call opposite
            if is_debit:
                long_k = base_k
                short_k = base_k + width
            else:
                short_k = base_k
                long_k = base_k + width
        else:
            if is_debit:
                long_k = base_k
                short_k = base_k - width
            else:
                short_k = base_k
                long_k = base_k - width

    if long_k is None or short_k is None:
        return {"ok": False, "reason": "spread needs long_strike+short_strike (or strike+width or legs)"}
    if abs(long_k - short_k) < 1e-9:
        return {"ok": False, "reason": "long and short strikes must differ"}

    # Sanity orientation
    if is_call and is_debit and long_k >= short_k:
        return {"ok": False, "reason": "debit call spread needs long_strike < short_strike"}
    if is_call and (not is_debit) and short_k >= long_k:
        return {"ok": False, "reason": "credit call spread needs short_strike < long_strike"}
    if (not is_call) and is_debit and long_k <= short_k:
        return {"ok": False, "reason": "debit put spread needs long_strike > short_strike"}
    if (not is_call) and (not is_debit) and short_k <= long_k:
        return {"ok": False, "reason": "credit put spread needs short_strike > long_strike"}

    width_abs = abs(short_k - long_k)
    return {
        "ok": True,
        "action": action,
        "ticker": ticker,
        "qty": qty,
        "expiry": str(expiry)[:10],
        "otype": otype,
        "is_debit": is_debit,
        "long_strike": long_k,
        "short_strike": short_k,
        "width": width_abs,
    }


def price_vertical_spread(market, spec: dict) -> dict:
    """Price both legs; return net debit/credit per share and cash flow."""
    if not spec.get("ok"):
        return spec
    ticker = spec["ticker"]
    expiry = spec["expiry"]
    otype = spec["otype"]
    qty = spec["qty"]
    long_px = _leg_price(market, ticker, expiry, spec["long_strike"], otype)
    short_px = _leg_price(market, ticker, expiry, spec["short_strike"], otype)
    if long_px is None or short_px is None:
        return {
            "ok": False,
            "reason": "no option price for one or both spread legs",
            "long_px": long_px,
            "short_px": short_px,
        }
    # Net premium per share from long perspective
    net_per_share = long_px - short_px  # debit if >0
    if spec["is_debit"]:
        if net_per_share <= 0:
            # market inverted / crossed — still allow small credit as blocked
            return {
                "ok": False,
                "reason": "debit spread priced at credit/zero (long_px=%.2f short_px=%.2f)" % (long_px, short_px),
            }
        cash_delta = -net_per_share * qty * MULT  # pay debit
        max_loss = net_per_share * qty * MULT
        max_gain = (spec["width"] - net_per_share) * qty * MULT
    else:
        credit = short_px - long_px
        if credit <= 0:
            return {
                "ok": False,
                "reason": "credit spread priced at debit/zero (long_px=%.2f short_px=%.2f)" % (long_px, short_px),
            }
        cash_delta = credit * qty * MULT
        max_gain = credit * qty * MULT
        max_loss = (spec["width"] - credit) * qty * MULT
        net_per_share = -credit
    return {
        "ok": True,
        **spec,
        "long_px": long_px,
        "short_px": short_px,
        "net_per_share": net_per_share,
        "cash_delta": cash_delta,
        "max_loss": max_loss,
        "max_gain": max_gain,
    }


def csp_collateral_required(strike: float, qty: float, premium: float) -> float:
    """Cash needed to secure short puts: strike*100*qty - premium credit."""
    return max(0.0, strike * MULT * qty - max(0.0, premium) * MULT * qty)


def structure_menu_lines() -> list:
    return [
        "Advanced structures (first-class actions):",
        "  - COVERED CALL: SELL_CALL while long >=100 shares/contract (opens short call)",
        "  - CASH-SECURED PUT: SELL_PUT with cash >= strike*100*qty - premium (opens short put)",
        "  - BUY_CALL_SPREAD: long lower call + short higher call; fields: long_strike, short_strike, expiry, qty",
        "  - BUY_PUT_SPREAD: long higher put + short lower put",
        "  - SELL_CALL_SPREAD: short lower call + long higher call (credit, defined risk)",
        "  - SELL_PUT_SPREAD: short higher put + long lower put (credit, defined risk)",
        "  - Alt shape: strike + width (debit call: long strike, short strike+width)",
        "  - BUY_CALL/BUY_PUT against short option lots = buy-to-cover",
        "  - Naked short calls are blocked; use covered or call credit spreads",
    ]

"""Deterministic Friday-weekly closer.

Leftover September mark is not the $800 new-debit budget.
This is the live send path for the 8/14 climb back to $10k.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
TARGET_EXPIRY = "2026-08-14"
TARGET_EQUITY = 10000.0
NEW_DEBIT_CAP = 800.0
DEFAULT_QTY = 3
WIDTH = 5.0
WRONG_ISSUER = "SK"
CLOSER_UNDERLYINGS = ("NVDA", "QQQ")
PPI_READY_HOUR = 8
PPI_READY_MINUTE = 45


def now_et(now: datetime | None = None) -> datetime:
    if now is None:
        return datetime.now(ET)
    if now.tzinfo is None:
        return now.replace(tzinfo=ET)
    return now.astimezone(ET)


def leftover_option_mark(positions: list[dict] | None) -> float:
    """Current long-option mark. Does not consume the new-debit closer cap."""
    total = 0.0
    for pos in positions or []:
        qty = _f(pos.get("qty"))
        if qty <= 0:
            continue
        typ = str(pos.get("type") or "").lower()
        if typ not in {"call", "put"}:
            continue
        mark = _f(pos.get("current_price"), _f(pos.get("avg_cost")))
        total += mark * qty * 100.0
    return total


def new_debit_budget_remaining(positions: list[dict] | None, cap: float = NEW_DEBIT_CAP) -> float:
    """Only 8/14 closer premium consumes the new-debit cap."""
    used = 0.0
    for pos in positions or []:
        qty = _f(pos.get("qty"))
        if qty <= 0:
            continue
        typ = str(pos.get("type") or "").lower()
        if typ not in {"call", "put"}:
            continue
        expiry = str(pos.get("expiry") or "")[:10]
        if expiry != TARGET_EXPIRY:
            continue
        mark = _f(pos.get("current_price"), _f(pos.get("avg_cost")))
        used += mark * qty * 100.0
    return max(0.0, cap - used)


def has_friday_weekly(positions: list[dict] | None) -> bool:
    for pos in positions or []:
        typ = str(pos.get("type") or "").lower()
        if typ not in {"call", "put"}:
            continue
        if str(pos.get("expiry") or "")[:10] != TARGET_EXPIRY:
            continue
        if abs(_f(pos.get("qty"))) > 1e-9:
            return True
    return False


def wrong_issuer_lots(positions: list[dict] | None) -> list[dict]:
    out = []
    for pos in positions or []:
        if str(pos.get("ticker") or "").upper() != WRONG_ISSUER:
            continue
        typ = str(pos.get("type") or "").lower()
        if typ not in {"call", "put"}:
            continue
        if abs(_f(pos.get("qty"))) <= 1e-9:
            continue
        out.append(pos)
    return out


def ppi_window_open(now: datetime | None = None) -> bool:
    ts = now_et(now)
    d = ts.date()
    if d < date(2026, 8, 13):
        return False
    if d > date(2026, 8, 14):
        return False
    if d == date(2026, 8, 13) and (ts.hour, ts.minute) < (PPI_READY_HOUR, PPI_READY_MINUTE):
        return False
    return True


def _f(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _round_strike(spot: float) -> float:
    if spot <= 0:
        return 0.0
    step = 5.0 if spot >= 50 else 1.0
    return round(spot / step) * step


def tape_is_hot(spot: float | None, prior_close: float | None) -> bool:
    if not spot or not prior_close or prior_close <= 0:
        return False
    return (float(spot) / float(prior_close) - 1.0) <= -0.015


def build_friday_weekly_decision(
    snapshot: dict,
    *,
    spot: float | None,
    prior_close: float | None = None,
    ticker: str = "NVDA",
    hot: bool | None = None,
) -> dict | None:
    positions = list((snapshot or {}).get("positions") or [])
    if has_friday_weekly(positions):
        return None
    remaining = new_debit_budget_remaining(positions)
    if remaining < 50:
        return None
    px = _f(spot)
    if px <= 0:
        return None
    long_k = _round_strike(px)
    if long_k <= 0:
        return None
    if hot is None:
        hot = tape_is_hot(px, prior_close)
    if hot:
        action = "BUY_PUT_SPREAD"
        short_k = max(long_k - WIDTH, 1.0)
        long_strike, short_strike = long_k, short_k
    else:
        action = "BUY_CALL_SPREAD"
        short_k = long_k + WIDTH
        long_strike, short_strike = long_k, short_k
    return {
        "action": action,
        "ticker": str(ticker).upper(),
        "qty": float(DEFAULT_QTY),
        "expiry": TARGET_EXPIRY,
        "long_strike": float(long_strike),
        "short_strike": float(short_strike),
        "reason": (
            "OPERATOR HARD SEND: one 8/14 Friday weekly to close the $10k gap. "
            "Leftover Sep mark is not the new-debit cap. Stop at $10k. "
            "Underlying is NVIDIA (NVDA) or Invesco QQQ Trust (QQQ). "
            "PPI/tape=%s." % ("hot puts" if hot else "cool/in-line calls")
        ),
    }


def flatten_wrong_issuer(store: Any, snapshot: dict, market_mod: Any) -> list[str]:
    notes: list[str] = []
    port = store.get_portfolio()
    cash = _f(port.get("cash"))
    total = _f(port.get("total_value"), cash)
    lots = wrong_issuer_lots(list((snapshot or {}).get("positions") or []))
    if not lots:
        return notes
    for pos in lots:
        ticker = str(pos.get("ticker") or "").upper()
        otype = str(pos.get("type") or "").lower()
        qty = _f(pos.get("qty"))
        expiry = str(pos.get("expiry") or "")[:10]
        strike = _f(pos.get("strike"))
        try:
            live = market_mod.get_option_price(ticker, expiry, strike, otype)
        except Exception:
            live = None
        px = _f(live, _f(pos.get("current_price"), _f(pos.get("avg_cost"))))
        if px <= 0 or abs(qty) <= 1e-9:
            notes.append("%s %s %s %g: no mark, held" % (ticker, otype, expiry, strike))
            continue
        if qty > 0:
            action = "SELL_CALL" if otype == "call" else "SELL_PUT"
        else:
            action = "BUY_CALL" if otype == "call" else "BUY_PUT"
        reason = (
            "OPERATOR HARD EXIT: SK is the corgi ETF, not Hynix. "
            "Flatten wrong-issuer sleeve before the Friday weekly."
        )
        store.record_trade(
            ticker, action, abs(qty), px, reason,
            expiry=expiry, strike=strike, option_type=otype,
        )
        store.upsert_position(
            ticker, otype, -qty, px,
            expiry=expiry, strike=strike,
        )
        cash += qty * px * 100.0
        notes.append(
            "flatten %s %s %g %s %g @ %.2f" % (ticker, action, abs(qty), expiry, strike, px)
        )
    store.update_portfolio(cash, total)
    return notes


def _spread_debit(priced: dict) -> float:
    for key in ("net_per_share", "debit", "net_debit", "debit_per_share", "net", "premium"):
        val = _f(priced.get(key))
        if val > 0:
            return val
    long_px = _f(priced.get("long_px"))
    short_px = _f(priced.get("short_px"))
    debit = long_px - short_px
    return debit if debit > 0 else 0.0


def maybe_execute_friday_closer(
    store: Any,
    snapshot: dict,
    *,
    market_mod: Any,
    execute_fn: Any,
    market_open: bool,
    now: datetime | None = None,
    watch_prices: dict | None = None,
) -> list[str]:
    notes: list[str] = []
    if not market_open:
        return notes
    ts = now_et(now)
    if ts.date() > date(2026, 8, 14):
        return notes

    flatten_notes = flatten_wrong_issuer(store, snapshot, market_mod)
    if flatten_notes:
        notes.extend(flatten_notes)
        snapshot = dict(snapshot or {})
        try:
            snapshot["positions"] = store.open_positions()
            snapshot["cash"] = store.get_portfolio().get("cash")
        except Exception:
            pass

    if not ppi_window_open(ts):
        notes.append("closer armed; waiting for Thu 08:45 ET PPI tape")
        return notes
    if has_friday_weekly(list((snapshot or {}).get("positions") or [])):
        notes.append("8/14 weekly already on; no second closer")
        return notes

    prices = watch_prices or {}
    decision = None
    last_err = None
    for ticker in CLOSER_UNDERLYINGS:
        spot = prices.get(ticker)
        if not isinstance(spot, (int, float)) or spot <= 0:
            try:
                spot = market_mod.get_price(ticker)
            except Exception as exc:
                last_err = str(exc)
                continue
        try:
            prior = _prior_close(market_mod, ticker)
        except Exception:
            prior = None
        candidate = build_friday_weekly_decision(
            snapshot, spot=spot, prior_close=prior, ticker=ticker
        )
        if not candidate:
            continue
        try:
            from .options_structures import parse_vertical_spread, price_vertical_spread
            spec = parse_vertical_spread(candidate)
            priced = price_vertical_spread(market_mod, spec)
        except Exception as exc:
            last_err = str(exc)
            continue
        if not priced.get("ok"):
            last_err = priced.get("reason") or "unpriced"
            continue
        debit = _spread_debit(priced)
        if debit <= 0:
            last_err = "%s weekly debit not positive" % ticker
            continue
        qty = int(candidate.get("qty") or DEFAULT_QTY)
        cost = debit * 100.0 * qty
        if cost > NEW_DEBIT_CAP:
            qty = int(NEW_DEBIT_CAP // max(debit * 100.0, 1.0))
            if qty < 1:
                last_err = "%s weekly debit $%.0f exceeds $%.0f" % (ticker, cost, NEW_DEBIT_CAP)
                continue
            candidate["qty"] = float(qty)
            cost = debit * 100.0 * qty
        decision = candidate
        notes.append(
            "closer %s %s %g/%g x%g debit~$%.0f"
            % (
                ticker,
                candidate["action"],
                candidate["long_strike"],
                candidate["short_strike"],
                candidate["qty"],
                cost,
            )
        )
        break

    if not decision:
        notes.append("closer held: chain/price missing (%s)" % (last_err or "no ATM weekly"))
        return notes

    status, detail = execute_fn(decision, snapshot, store)
    notes.append("execute %s: %s" % (status, detail))
    return notes


def _prior_close(market_mod: Any, ticker: str) -> float | None:
    getter = getattr(market_mod, "previous_close", None)
    if callable(getter):
        px = getter(ticker)
        if isinstance(px, (int, float)) and px > 0:
            return float(px)
    return None

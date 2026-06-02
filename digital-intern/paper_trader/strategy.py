"""Opus 4.7 trading strategy — packages context, asks Claude for a JSON decision,
executes it through paper trade plumbing, and enforces hard risk rules."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import datetime, timezone

from core.claude_cli import DEFAULT_LLM_MODEL, claude_call

from . import market, signals
from .store import Store, get_store

MODEL = DEFAULT_LLM_MODEL
DECISION_TIMEOUT_S = 120

WATCHLIST = [
    "LITE", "LNOK", "MUU", "DRAM", "SNDU",  # current real-account interests
    "NVDA", "AMD", "MU", "AMAT", "LRCX", "KLAC", "TSM", "ASML", "MRVL",  # semis
    "SMH", "SOXX", "SPY", "QQQ",  # ETFs
]

FUTURES = ["ES=F", "NQ=F", "CL=F", "GC=F"]

CASH_RESERVE = 100.0           # never go below this
MAX_POSITION_PCT = 0.40        # max 40% of portfolio in one position
MAX_OPTION_POSITIONS = 3       # at most 3 open option positions
STOP_LOSS_PCT = -0.20          # auto-sell at -20%
TAKE_PROFIT_PCT = 0.50         # trim half at +50%

SYSTEM_PROMPT = """You are an aggressive but disciplined paper-trading desk running a $1000 account.
Your goal: maximize total return by end of May 2026. Trade US stocks, options (calls/puts), and futures.

You must respond with a SINGLE JSON object — no prose, no markdown fences. Schema:

{
  "action": "BUY" | "SELL" | "BUY_CALL" | "BUY_PUT" | "SELL_CALL" | "SELL_PUT" | "HOLD" | "REBALANCE",
  "ticker": "NVDA",
  "qty": 0.5,
  "strike": 900,             // only for option actions
  "expiry": "2026-05-30",    // only for option actions, YYYY-MM-DD
  "confidence": 0.85,
  "reasoning": "1-3 sentences why",
  "stop_loss": 850.0,        // optional
  "take_profit": 950.0       // optional
}

Rules:
- HOLD when signal quality is low. Patience is alpha.
- Size by conviction. Confidence >= 0.8 → larger size. Below 0.5 → HOLD.
- Use options when a thesis is clear and you want leverage. Pick liquid strikes (near-the-money).
- Watch the existing portfolio positions; don't accidentally double-up risk.
- Never bet the whole account on one trade. Position cap is 40% of equity.
- Never go below $100 cash.
- Prioritize signals with ai_score >= 8 (urgent or near-urgent).
- If you can't find a clean trade, HOLD with reasoning "no setup".
- For SELL/SELL_CALL/SELL_PUT, ticker must match an open position (and strike/expiry for options).

Return JSON ONLY.
"""


def _claude_call(prompt: str) -> str | None:
    return claude_call(prompt, model=MODEL, timeout=DECISION_TIMEOUT_S)


def _parse_decision(raw: str) -> dict | None:
    if not raw:
        return None
    # strip ```json fences if model ignored instructions
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # find the first JSON object
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception as e:
        print(f"[strategy] JSON parse failed: {e}\nraw: {text[:300]}")
        return None


def _portfolio_snapshot(store: Store) -> dict:
    """Mark-to-market every open position, write back to DB, return summary."""
    positions = store.open_positions()
    stock_tickers = sorted({p["ticker"] for p in positions if p["type"] == "stock"})
    prices = market.get_prices(stock_tickers) if stock_tickers else {}

    marks: dict[int, tuple[float, float]] = {}
    enriched = []
    open_value = 0.0
    for p in positions:
        if p["type"] in ("call", "put"):
            cur = market.get_option_price(p["ticker"], p["expiry"], p["strike"], p["type"])
            multiplier = 100
        else:
            cur = prices.get(p["ticker"])
            multiplier = 1
        cur = cur or p["avg_cost"]
        pl = (cur - p["avg_cost"]) * p["qty"] * multiplier
        pl_pct = ((cur - p["avg_cost"]) / p["avg_cost"]) * 100 if p["avg_cost"] else 0.0
        marks[p["id"]] = (cur, pl)
        enriched.append({**p, "current_price": cur, "unrealized_pl": pl, "pl_pct": pl_pct,
                         "market_value": cur * p["qty"] * multiplier})
        open_value += cur * p["qty"] * multiplier

    if marks:
        store.update_position_marks(marks)

    pf = store.get_portfolio()
    total = pf["cash"] + open_value
    store.update_portfolio(pf["cash"], total, [
        {k: v for k, v in pos.items() if k != "opened_at"} for pos in enriched
    ])
    return {
        "cash": pf["cash"],
        "total_value": total,
        "open_value": open_value,
        "positions": enriched,
    }


def _build_payload(snapshot: dict, top_signals: list[dict], sentiments: list[dict],
                   watch_prices: dict[str, float | None],
                   futures_prices: dict[str, float | None],
                   sp500: float | None, market_open: bool) -> str:
    now = datetime.now(timezone.utc).isoformat()
    pos_lines = []
    for p in snapshot["positions"]:
        if p["type"] in ("call", "put"):
            pos_lines.append(
                f"  {p['ticker']} {p['type'].upper()} {p['strike']} {p['expiry']}: "
                f"qty={p['qty']} avg={p['avg_cost']:.2f} mark={p['current_price']:.2f} "
                f"P/L=${p['unrealized_pl']:.2f} ({p['pl_pct']:.1f}%)"
            )
        else:
            pos_lines.append(
                f"  {p['ticker']} {p['type']}: qty={p['qty']} avg={p['avg_cost']:.2f} "
                f"mark={p['current_price']:.2f} P/L=${p['unrealized_pl']:.2f} ({p['pl_pct']:.1f}%)"
            )

    sig_lines = []
    for s in top_signals[:10]:
        sig_lines.append(
            f"  [{s['ai_score']:.1f}] urg={s['urgency']} {s['title'][:140]}"
            + (f"  tickers={','.join(s['tickers'][:5])}" if s['tickers'] else "")
        )

    sent_lines = [
        f"  {r['ticker']:>6}: avg={r['avg_score']:.1f} n={r['n']} urgent={r['urgent']}"
        for r in sentiments if r["n"] > 0
    ]

    px_lines = [f"  {t}: {p:.2f}" if p else f"  {t}: N/A" for t, p in watch_prices.items()]
    fut_lines = [f"  {t}: {p:.2f}" if p else f"  {t}: N/A" for t, p in futures_prices.items()]

    sp = f"{sp500:.2f}" if sp500 else "N/A"

    return f"""TIME (UTC): {now}
MARKET_OPEN: {market_open}
S&P 500 BENCHMARK: {sp}

PORTFOLIO:
  cash: ${snapshot['cash']:.2f}
  open positions value: ${snapshot['open_value']:.2f}
  total value: ${snapshot['total_value']:.2f}
  positions:
{chr(10).join(pos_lines) if pos_lines else '  (none)'}

WATCHLIST PRICES:
{chr(10).join(px_lines)}

FUTURES:
{chr(10).join(fut_lines)}

TICKER SENTIMENT (last 4h, from scored news):
{chr(10).join(sent_lines) if sent_lines else '  (no scored mentions)'}

TOP SCORED SIGNALS (last 2h, ai_score >= 4.0):
{chr(10).join(sig_lines) if sig_lines else '  (no high-score signals)'}

RISK RULES (auto-enforced after your reply):
  - cash floor: $100
  - max single position: 40% of total value
  - max open option positions: 3

Return JSON only."""


def _enforce_risk_pre_trade(decision: dict, snapshot: dict) -> tuple[bool, str]:
    """Block trades that violate hard rules. Returns (ok, reason_if_blocked)."""
    action = decision.get("action", "HOLD")
    if action == "HOLD":
        return True, ""

    ticker = (decision.get("ticker") or "").upper()
    qty = float(decision.get("qty") or 0)
    if qty <= 0 and action != "REBALANCE":
        return False, "qty must be > 0"

    open_options = [p for p in snapshot["positions"] if p["type"] in ("call", "put")]

    if action in ("BUY_CALL", "BUY_PUT"):
        if len(open_options) >= MAX_OPTION_POSITIONS:
            return False, f"option position cap reached ({MAX_OPTION_POSITIONS})"

    if action in ("SELL", "SELL_CALL", "SELL_PUT"):
        # must match an existing position
        opt_type = "call" if action == "SELL_CALL" else "put" if action == "SELL_PUT" else "stock"
        matches = [
            p for p in snapshot["positions"]
            if p["ticker"] == ticker and p["type"] == opt_type
        ]
        if not matches:
            return False, f"no open {opt_type} position in {ticker} to close"
        # qty bound: cannot sell more than held across matching positions
        held = sum(p["qty"] for p in matches)
        if qty > held + 1e-6:
            return False, f"sell qty {qty} exceeds held {held} for {ticker} {opt_type}"
    return True, ""


def _execute(decision: dict, snapshot: dict, store: Store) -> tuple[str, str]:
    """Apply the decision against the paper book. Returns (status, detail)."""
    action = decision.get("action", "HOLD")
    if action == "HOLD":
        return "HOLD", decision.get("reasoning", "")

    if action == "REBALANCE":
        return "HOLD", "REBALANCE not yet implemented; treated as HOLD"

    ticker = (decision.get("ticker") or "").upper()
    qty = float(decision.get("qty") or 0)
    reason = decision.get("reasoning", "")

    ok, why = _enforce_risk_pre_trade(decision, snapshot)
    if not ok:
        return "BLOCKED", why

    if action in ("BUY", "SELL"):
        price = market.get_price(ticker)
        if not price:
            return "BLOCKED", f"no price for {ticker}"
        notional = price * qty
        if action == "BUY":
            if snapshot["cash"] - notional < CASH_RESERVE:
                return "BLOCKED", f"would breach $100 cash floor (need ${notional:.2f})"
            if notional > snapshot["total_value"] * MAX_POSITION_PCT:
                return "BLOCKED", f"position would exceed {int(MAX_POSITION_PCT*100)}% cap"
            store.record_trade(ticker, "BUY", qty, price, reason)
            store.upsert_position(ticker, "stock", qty, price)
            store.update_portfolio(snapshot["cash"] - notional, snapshot["total_value"], snapshot["positions"])
            return "FILLED", f"BUY {qty} {ticker} @ {price:.2f}"
        else:
            store.record_trade(ticker, "SELL", qty, price, reason)
            store.upsert_position(ticker, "stock", -qty, price)
            store.update_portfolio(snapshot["cash"] + notional, snapshot["total_value"], snapshot["positions"])
            return "FILLED", f"SELL {qty} {ticker} @ {price:.2f}"

    if action in ("BUY_CALL", "BUY_PUT"):
        otype = "call" if action == "BUY_CALL" else "put"
        strike = decision.get("strike")
        expiry = decision.get("expiry")
        if not (strike and expiry):
            return "BLOCKED", "option trade missing strike/expiry"
        opt_px = market.get_option_price(ticker, expiry, float(strike), otype)
        if not opt_px:
            return "BLOCKED", f"no option price for {ticker} {expiry} {strike} {otype}"
        notional = opt_px * qty * 100
        if snapshot["cash"] - notional < CASH_RESERVE:
            return "BLOCKED", f"would breach $100 cash floor (need ${notional:.2f})"
        if notional > snapshot["total_value"] * MAX_POSITION_PCT:
            return "BLOCKED", f"option premium would exceed {int(MAX_POSITION_PCT*100)}% cap"
        store.record_trade(ticker, action, qty, opt_px, reason, expiry=expiry,
                           strike=float(strike), option_type=otype)
        store.upsert_position(ticker, otype, qty, opt_px, expiry=expiry, strike=float(strike))
        store.update_portfolio(snapshot["cash"] - notional, snapshot["total_value"], snapshot["positions"])
        return "FILLED", f"{action} {qty} {ticker} {strike}{otype[0].upper()} {expiry} @ {opt_px:.2f}"

    if action in ("SELL_CALL", "SELL_PUT"):
        otype = "call" if action == "SELL_CALL" else "put"
        strike = decision.get("strike")
        expiry = decision.get("expiry")
        opt_px = market.get_option_price(ticker, expiry, float(strike), otype) if strike and expiry else None
        # fallback to opening cost if no live quote
        match = next((p for p in snapshot["positions"]
                      if p["ticker"] == ticker and p["type"] == otype
                      and (not strike or p["strike"] == float(strike))
                      and (not expiry or p["expiry"] == expiry)), None)
        if not match:
            return "BLOCKED", f"no matching open {otype} for {ticker}"
        opt_px = opt_px or match["avg_cost"]
        notional = opt_px * qty * 100
        store.record_trade(ticker, action, qty, opt_px, reason,
                           expiry=match["expiry"], strike=match["strike"], option_type=otype)
        store.upsert_position(ticker, otype, -qty, opt_px,
                              expiry=match["expiry"], strike=match["strike"])
        store.update_portfolio(snapshot["cash"] + notional, snapshot["total_value"], snapshot["positions"])
        return "FILLED", f"{action} {qty} {ticker} {match['strike']}{otype[0].upper()} {match['expiry']} @ {opt_px:.2f}"

    return "BLOCKED", f"unknown action {action}"


def _auto_risk_exits(snapshot: dict, store: Store) -> list[str]:
    """Auto stop-loss and take-profit before consulting Opus. Returns log lines."""
    actions = []
    for p in snapshot["positions"]:
        pct = p["pl_pct"] / 100.0
        if pct <= STOP_LOSS_PCT:
            if p["type"] == "stock":
                price = p["current_price"]
                store.record_trade(p["ticker"], "SELL", p["qty"], price,
                                   f"auto stop-loss {pct*100:.1f}%")
                store.upsert_position(p["ticker"], "stock", -p["qty"], price)
                store.update_portfolio(
                    store.get_portfolio()["cash"] + price * p["qty"],
                    snapshot["total_value"], snapshot["positions"]
                )
                actions.append(f"STOP {p['ticker']} stock @ {price:.2f} ({pct*100:.1f}%)")
            else:
                price = p["current_price"]
                action = "SELL_CALL" if p["type"] == "call" else "SELL_PUT"
                store.record_trade(p["ticker"], action, p["qty"], price,
                                   f"auto stop-loss {pct*100:.1f}%",
                                   expiry=p["expiry"], strike=p["strike"], option_type=p["type"])
                store.upsert_position(p["ticker"], p["type"], -p["qty"], price,
                                      expiry=p["expiry"], strike=p["strike"])
                store.update_portfolio(
                    store.get_portfolio()["cash"] + price * p["qty"] * 100,
                    snapshot["total_value"], snapshot["positions"]
                )
                actions.append(f"STOP {p['ticker']} {p['type']} @ {price:.2f} ({pct*100:.1f}%)")
        elif pct >= TAKE_PROFIT_PCT and p["qty"] > 0.0002:
            half = round(p["qty"] / 2, 4)
            if half <= 0:
                continue
            if p["type"] == "stock":
                price = p["current_price"]
                store.record_trade(p["ticker"], "SELL", half, price,
                                   f"auto take-profit trim {pct*100:.1f}%")
                store.upsert_position(p["ticker"], "stock", -half, price)
                store.update_portfolio(
                    store.get_portfolio()["cash"] + price * half,
                    snapshot["total_value"], snapshot["positions"]
                )
                actions.append(f"TP TRIM {p['ticker']} stock {half} @ {price:.2f} ({pct*100:.1f}%)")
            else:
                price = p["current_price"]
                action = "SELL_CALL" if p["type"] == "call" else "SELL_PUT"
                store.record_trade(p["ticker"], action, half, price,
                                   f"auto take-profit trim {pct*100:.1f}%",
                                   expiry=p["expiry"], strike=p["strike"], option_type=p["type"])
                store.upsert_position(p["ticker"], p["type"], -half, price,
                                      expiry=p["expiry"], strike=p["strike"])
                store.update_portfolio(
                    store.get_portfolio()["cash"] + price * half * 100,
                    snapshot["total_value"], snapshot["positions"]
                )
                actions.append(f"TP TRIM {p['ticker']} {p['type']} {half} @ {price:.2f} ({pct*100:.1f}%)")
    return actions


def decide() -> dict:
    """Run one decision cycle. Returns summary dict for logging."""
    store = get_store()
    market_open = market.is_market_open()

    snap = _portfolio_snapshot(store)
    auto_exits = _auto_risk_exits(snap, store)
    if auto_exits:
        # refresh snapshot after auto-exits
        snap = _portfolio_snapshot(store)

    top = signals.get_top_signals(20, hours=2, min_score=4.0)
    urgent = signals.get_urgent_articles(minutes=30)
    sents = signals.ticker_sentiments(WATCHLIST, hours=4)
    watch_px = market.get_prices(WATCHLIST)
    fut_px = {f: market.get_futures_price(f) for f in FUTURES}
    sp500 = market.benchmark_sp500()

    # include urgent items at the top
    seen_ids = {s["id"] for s in top}
    merged = [a for a in urgent if a["id"] not in seen_ids] + top

    payload = _build_payload(snap, merged, sents, watch_px, fut_px, sp500, market_open)
    prompt = f"{SYSTEM_PROMPT}\n\n---\nCONTEXT:\n{payload}"

    raw = _claude_call(prompt)
    decision = _parse_decision(raw) if raw else None

    summary = {
        "market_open": market_open,
        "signal_count": len(merged),
        "auto_exits": auto_exits,
        "decision": decision,
        "raw": raw,
        "snapshot": snap,
        "status": "NO_DECISION",
        "detail": "",
    }

    if not decision:
        store.record_decision(market_open, len(merged), "NO_DECISION",
                              "claude returned no parseable JSON",
                              snap["total_value"], snap["cash"])
        store.record_equity_point(snap["total_value"], snap["cash"], sp500)
        return summary

    status, detail = _execute(decision, snap, store)
    summary["status"] = status
    summary["detail"] = detail

    action_label = f"{decision.get('action','?')} {decision.get('ticker','')}".strip()
    store.record_decision(
        market_open,
        len(merged),
        f"{action_label} → {status}",
        json.dumps({"decision": decision, "auto_exits": auto_exits, "detail": detail}),
        snap["total_value"],
        snap["cash"],
    )
    # final mark + equity point
    final = _portfolio_snapshot(store)
    store.record_equity_point(final["total_value"], final["cash"], sp500)
    summary["snapshot"] = final
    return summary


if __name__ == "__main__":
    import pprint
    pprint.pp(decide())

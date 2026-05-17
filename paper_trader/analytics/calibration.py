"""Confidence calibration + signal attribution for the live Opus trader.

Two questions this answers:
  1. **Calibration** — does Opus's stated confidence (0.0–1.0) match its actual
     hit rate? If conf≥0.8 trades win 50%, the model is overconfident.
  2. **Signal attribution** — what kind of trade is profitable? Heuristically
     classify each BUY's reasoning text into news-driven / technical-driven /
     mixed buckets and compute per-bucket win rate + avg return.

Realized return is computed by FIFO-matching SELL trades back to BUYs (the same
approach the unified dashboard's trade journal uses). Open positions aren't
included in win rate, but get a separate "open exposure" line for context.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone

# Confidence buckets — bounded inclusive on the low end, exclusive on the high.
CONF_BINS: list[tuple[str, float, float]] = [
    ("0.0-0.5",  0.0,   0.5),
    ("0.5-0.65", 0.5,   0.65),
    ("0.65-0.8", 0.65,  0.8),
    ("0.8-1.0",  0.8,   1.01),
]

# Keyword sets for crude signal-source attribution.
NEWS_RX = re.compile(
    r"\b(news|article|headline|upgrade|downgrade|guidance|earnings|catalyst|target|"
    r"insider|congressional|reuters|bloomberg|wsj|bofa|morgan|goldman|jefferies|"
    r"score|ai_score|urgent|breaking|announce|leak|deal|merger|acquisition)\b",
    re.IGNORECASE,
)
TECH_RX = re.compile(
    r"\b(rsi|macd|bollinger|momentum|mom_5d|mom_20d|overbought|oversold|golden\s?cross|"
    r"death\s?cross|moving\s?average|vol_ratio|breakout|breakdown|support|resistance|"
    r"trend|52[- ]?week)\b",
    re.IGNORECASE,
)


def _attribute(reasoning: str) -> str:
    """Classify reasoning text into news / technical / mixed / other."""
    if not reasoning:
        return "other"
    has_news = bool(NEWS_RX.search(reasoning))
    has_tech = bool(TECH_RX.search(reasoning))
    if has_news and has_tech:
        return "mixed"
    if has_news:
        return "news"
    if has_tech:
        return "technical"
    return "other"


def _parse_decision_row(row: dict) -> dict | None:
    """Pull confidence + reasoning + action + ticker out of a decisions row."""
    raw = row.get("reasoning") or ""
    try:
        blob = json.loads(raw)
        inner = blob.get("decision") or {}
    except Exception:
        return None
    if not isinstance(inner, dict):
        return None
    action = (inner.get("action") or "").upper()
    if action not in ("BUY", "SELL", "BUY_CALL", "BUY_PUT", "SELL_CALL", "SELL_PUT"):
        return None
    return {
        "timestamp": row.get("timestamp"),
        "ticker": (inner.get("ticker") or "").upper(),
        "action": action,
        "confidence": inner.get("confidence"),
        "reasoning": inner.get("reasoning") or "",
        "portfolio_value": row.get("portfolio_value"),
    }


def _fifo_realized(trades: list[dict]) -> list[dict]:
    """FIFO-match SELLs to BUYs and emit one realized-trade dict per match.

    Each output row contains buy/sell timestamps, prices, qty, return_pct, ticker.
    Options are matched by (ticker, strike, expiry, option_type)."""
    # trades are most-recent-first from the store; we want chronological.
    chrono = list(reversed(trades))
    buys: dict[tuple, list[dict]] = defaultdict(list)
    out: list[dict] = []
    for t in chrono:
        ticker = (t.get("ticker") or "").upper()
        action = (t.get("action") or "").upper()
        opt = t.get("option_type")
        if opt:
            key = (ticker, opt, t.get("strike"), t.get("expiry"))
        else:
            key = (ticker, "stock", None, None)
        qty = float(t.get("qty") or 0.0)
        price = float(t.get("price") or 0.0)
        if qty <= 0 or price <= 0:
            continue
        if action == "BUY" or action.endswith("_CALL") and action.startswith("BUY") or \
           action.endswith("_PUT") and action.startswith("BUY"):
            buys[key].append({"qty": qty, "price": price, "ts": t.get("timestamp")})
        elif action == "SELL" or (action.startswith("SELL") and ("CALL" in action or "PUT" in action)):
            remaining = qty
            while remaining > 1e-9 and buys[key]:
                head = buys[key][0]
                take = min(head["qty"], remaining)
                ret = (price - head["price"]) / head["price"] * 100.0
                out.append({
                    "ticker": ticker,
                    "type": key[1],
                    "buy_ts": head["ts"],
                    "sell_ts": t.get("timestamp"),
                    "qty": take,
                    "buy_price": head["price"],
                    "sell_price": price,
                    "return_pct": round(ret, 2),
                })
                head["qty"] -= take
                remaining -= take
                if head["qty"] <= 1e-9:
                    buys[key].pop(0)
    return out


def _match_decision_to_trade(decisions: list[dict], trade: dict) -> dict | None:
    """Find the BUY decision closest in time to the buy_ts of a realized trade."""
    if not decisions or not trade.get("buy_ts"):
        return None
    try:
        target = datetime.fromisoformat(trade["buy_ts"].replace("Z", "+00:00"))
    except Exception:
        return None
    best = None
    best_diff = None
    for d in decisions:
        if d["action"] not in ("BUY", "BUY_CALL", "BUY_PUT"):
            continue
        if d["ticker"] != trade["ticker"]:
            continue
        try:
            dt = datetime.fromisoformat(d["timestamp"].replace("Z", "+00:00"))
        except Exception:
            continue
        diff = abs((dt - target).total_seconds())
        # Match if within 5 minutes — the trade is recorded ~immediately after the decision.
        if diff > 600:
            continue
        if best_diff is None or diff < best_diff:
            best = d
            best_diff = diff
    return best


def _bucket(conf: float | None) -> str | None:
    if conf is None:
        return None
    try:
        c = float(conf)
    except Exception:
        return None
    for label, lo, hi in CONF_BINS:
        if lo <= c < hi:
            return label
    return None


def build_calibration(decisions_raw: list[dict], trades_raw: list[dict]) -> dict:
    """Compute calibration + attribution tables.

    `decisions_raw` is the list returned by store.recent_decisions() (most recent
    first); we'll parse confidence out. `trades_raw` is store.recent_trades().
    """
    parsed: list[dict] = []
    for r in decisions_raw:
        p = _parse_decision_row(r)
        if p is not None:
            parsed.append(p)

    realized = _fifo_realized(trades_raw)

    # Confidence calibration over CLOSED trades that were matchable to a BUY decision.
    by_conf: dict[str, dict] = {lbl: {"n": 0, "wins": 0, "avg_return": 0.0, "ret_sum": 0.0,
                                       "avg_conf": 0.0, "conf_sum": 0.0}
                                 for lbl, _, _ in CONF_BINS}
    by_src: dict[str, dict] = {k: {"n": 0, "wins": 0, "ret_sum": 0.0, "avg_return": 0.0,
                                    "best": -1e9, "worst": 1e9}
                                for k in ("news", "technical", "mixed", "other")}

    enriched_realized: list[dict] = []
    for trade in realized:
        dec = _match_decision_to_trade(parsed, trade)
        conf = dec["confidence"] if dec else None
        source = _attribute(dec["reasoning"]) if dec else "other"
        ret = float(trade["return_pct"])
        bkt = _bucket(conf)
        if bkt:
            by_conf[bkt]["n"] += 1
            by_conf[bkt]["conf_sum"] += float(conf or 0.0)
            by_conf[bkt]["ret_sum"] += ret
            if ret > 0:
                by_conf[bkt]["wins"] += 1
        src = by_src[source]
        src["n"] += 1
        src["ret_sum"] += ret
        if ret > 0:
            src["wins"] += 1
        src["best"] = max(src["best"], ret)
        src["worst"] = min(src["worst"], ret)
        enriched_realized.append({
            **trade,
            "confidence": conf,
            "source": source,
            "reasoning_excerpt": (dec["reasoning"][:160] if dec else None),
        })

    conf_rows = []
    for label, _, _ in CONF_BINS:
        b = by_conf[label]
        if b["n"]:
            b["avg_return"] = round(b["ret_sum"] / b["n"], 2)
            b["avg_conf"] = round(b["conf_sum"] / b["n"], 3)
            b["win_rate"] = round(b["wins"] / b["n"] * 100, 1)
        else:
            b["avg_return"] = 0.0
            b["avg_conf"] = 0.0
            b["win_rate"] = 0.0
        conf_rows.append({"bucket": label, **{k: b[k] for k in
                          ("n", "wins", "win_rate", "avg_return", "avg_conf")}})

    src_rows = []
    for name, b in by_src.items():
        if b["n"]:
            b["avg_return"] = round(b["ret_sum"] / b["n"], 2)
            b["win_rate"] = round(b["wins"] / b["n"] * 100, 1)
            b["best"] = round(b["best"], 2)
            b["worst"] = round(b["worst"], 2)
        else:
            b["avg_return"] = 0.0
            b["win_rate"] = 0.0
            b["best"] = None
            b["worst"] = None
        src_rows.append({"source": name, **{k: b[k] for k in
                          ("n", "wins", "win_rate", "avg_return", "best", "worst")}})

    # Confidence distribution of OPEN decisions (per source) — surfaces what
    # kind of trade the bot is currently in flight on.
    open_conf = []
    seen = set()
    for d in parsed:
        if d["action"] not in ("BUY", "BUY_CALL", "BUY_PUT"):
            continue
        key = (d["ticker"], d["action"])
        if key in seen:
            continue
        seen.add(key)
        open_conf.append({
            "timestamp": d["timestamp"],
            "ticker": d["ticker"],
            "action": d["action"],
            "confidence": d["confidence"],
            "source": _attribute(d["reasoning"]),
        })
        if len(open_conf) >= 12:
            break

    return {
        "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_decisions_parsed": len(parsed),
        "n_realized_trades": len(realized),
        "confidence_buckets": conf_rows,
        "signal_sources": src_rows,
        "recent_realized": enriched_realized[-25:],
        "recent_buy_decisions": open_conf,
    }

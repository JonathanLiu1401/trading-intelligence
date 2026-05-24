"""Round-trip trade outcome analytics — win rate, expectancy, profit factor.

Pairs raw store-native trades (BUY → matching SELL, FIFO per ticker) into
round-trips, then derives the canonical 3 hedge-fund metrics every desk
reviews on its own track record:

* win_rate          — % of round-trips that closed positive
* expectancy        — win_rate · avg_win  −  loss_rate · avg_loss
* profit_factor     — gross wins / gross losses

Pure: feed it ``store.recent_trades(N)`` (any order — sorting is internal).
The MACD-strategy 1.5:1 R:R guard added with HARD_SL / HARD_TP exit reasons
is also surfaced as a count so the operator can see how often the hard-exit
discipline fires vs. discretionary closes. Used by self_review to feed Opus
its own mechanical scorecard, the same way a real desk reviews P&L
attribution before the next trade."""
from __future__ import annotations

from collections import defaultdict


def _pair_trades(trades: list[dict]) -> list[dict]:
    """Match BUY/SELL pairs per ticker. Returns list of round-trip dicts.

    Stock-only — option trades carry a strike/expiry that doesn't FIFO-pair
    cleanly with the same-ticker stock; option_type=NULL is the proxy for
    "stock trade". FIFO per ticker so a partial-size exit pairs against the
    oldest open BUY, the convention round_trips.py already uses."""
    open_buys: dict[str, list[dict]] = defaultdict(list)
    round_trips: list[dict] = []
    for t in sorted(trades, key=lambda x: x.get("timestamp") or ""):
        # Skip option legs — they don't pair as plain BUY/SELL pairs.
        if t.get("option_type"):
            continue
        ticker = t.get("ticker") or ""
        action = (t.get("action") or "").upper()
        if action == "BUY":
            open_buys[ticker].append(t)
        elif action == "SELL" and open_buys[ticker]:
            buy = open_buys[ticker].pop(0)  # FIFO
            try:
                entry = float(buy.get("price") or 0)
                exit_p = float(t.get("price") or 0)
            except (TypeError, ValueError):
                continue
            if entry <= 0:
                continue
            pnl_pct = (exit_p - entry) / entry * 100
            round_trips.append({
                "ticker": ticker,
                "entry_price": entry,
                "exit_price": exit_p,
                "pnl_pct": round(pnl_pct, 2),
                "win": pnl_pct > 0,
                "entry_ts": buy.get("timestamp") or "",
                "exit_ts": t.get("timestamp") or "",
                "exit_reason": t.get("reason") or "",
            })
    return round_trips


def _compute_stats(round_trips: list[dict]) -> dict:
    """Aggregate round-trip stats. Empty input → empty dict."""
    if not round_trips:
        return {}
    wins = [r for r in round_trips if r["win"]]
    losses = [r for r in round_trips if not r["win"]]
    n = len(round_trips)
    win_rate = len(wins) / n * 100
    avg_win = sum(r["pnl_pct"] for r in wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(r["pnl_pct"] for r in losses) / len(losses)) if losses else 0.0
    loss_rate = len(losses) / n
    expectancy = (win_rate / 100 * avg_win) - (loss_rate * avg_loss)
    if losses and avg_loss > 0:
        profit_factor = (avg_win * len(wins)) / (avg_loss * len(losses))
    else:
        profit_factor = float("inf")
    last_20 = round_trips[-20:]
    wins_20 = sum(1 for r in last_20 if r["win"])
    win_rate_20 = wins_20 / len(last_20) * 100 if last_20 else 0.0
    sl_exits = sum(1 for r in round_trips if "HARD_SL" in (r.get("exit_reason") or ""))
    tp_exits = sum(1 for r in round_trips if "HARD_TP" in (r.get("exit_reason") or ""))
    return {
        "n_trades": n,
        "win_rate_pct": round(win_rate, 1),
        "win_rate_last20_pct": round(win_rate_20, 1),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "expectancy_pct": round(expectancy, 3),
        "profit_factor": (round(profit_factor, 2)
                          if profit_factor != float("inf") else "∞"),
        "n_hard_sl": sl_exits,
        "n_hard_tp": tp_exits,
    }


def build_trade_outcomes_block(trades: list[dict]) -> str | None:
    """Compose a compact prompt block summarising round-trip outcomes.

    Returns None when there are no round-trips yet (a fresh book) — the
    self_review caller already handles "no data" with its preamble fallback,
    so a None here means "no section this cycle" (silence-when-nothing-
    actionable, the same precedent the other analytics blocks follow).
    Never raises — any failure degrades to None so the live decide() cycle
    is never aborted by a diagnostics fault."""
    try:
        round_trips = _pair_trades(trades or [])
        if not round_trips:
            return None
        stats = _compute_stats(round_trips)
        lines = [
            "TRADE OUTCOMES (all-time round trips):",
            f"  trades={stats['n_trades']}  win_rate={stats['win_rate_pct']}%"
            f"  (last20: {stats['win_rate_last20_pct']}%)",
            f"  avg_win={stats['avg_win_pct']}%  avg_loss=-{stats['avg_loss_pct']}%",
            f"  expectancy={stats['expectancy_pct']}%  profit_factor={stats['profit_factor']}",
            f"  hard_SL_exits={stats['n_hard_sl']}  hard_TP_exits={stats['n_hard_tp']}",
        ]
        return "\n".join(lines)
    except Exception:
        return None

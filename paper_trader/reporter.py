"""Discord reporter — pushes trades, hourly summaries, and daily close to the channel."""
from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone

from . import market
from .store import Store, get_store

DISCORD_CHANNEL = "channel:1496099475838603324"
_INITIAL_EQUITY = 1000.0


def _send(message: str) -> bool:
    if not shutil.which("openclaw"):
        print(f"[reporter] openclaw not installed; would send:\n{message}")
        return False
    try:
        r = subprocess.run(
            ["openclaw", "message", "send",
             "--channel", "discord",
             "--target", DISCORD_CHANNEL,
             "--message", message],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0:
            print(f"[reporter] openclaw failed: {r.stderr.strip()[:300]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("[reporter] openclaw timeout")
        return False
    except Exception as e:
        print(f"[reporter] openclaw exception: {e}")
        return False


def send_trade_alert(trade: dict) -> bool:
    """Post a single trade immediately."""
    t = trade
    extra = ""
    if t.get("option_type"):
        extra = f" {t['strike']}{t['option_type'][0].upper()} {t['expiry']}"
    body = (
        f"**TRADE** `{t['action']}` `{t['ticker']}`{extra}\n"
        f"qty `{t['qty']}` @ `${t['price']:.2f}` = `${t['value']:.2f}`\n"
        f"_{t.get('reason','')}_"
    )
    return _send(body)


def send_decision_log(summary: dict) -> bool:
    d = summary.get("decision") or {}
    action = d.get("action", "NO_DECISION")
    ticker = d.get("ticker", "")
    conf = d.get("confidence", "?")
    reasoning = d.get("reasoning", "")
    status = summary.get("status", "?")
    detail = summary.get("detail", "")
    auto = summary.get("auto_exits") or []
    pf = summary["snapshot"]
    pl = pf["total_value"] - _INITIAL_EQUITY
    pl_pct = pl / _INITIAL_EQUITY * 100

    parts = [
        f"**Δ DECISION** `{action} {ticker}` → `{status}`",
        f"conf=`{conf}` value=`${pf['total_value']:.2f}` "
        f"P/L=`${pl:+.2f}` (`{pl_pct:+.2f}%`) cash=`${pf['cash']:.2f}`",
    ]
    if auto:
        parts.append("auto: " + "; ".join(f"`{a}`" for a in auto))
    if detail:
        parts.append(f"_{detail}_")
    if reasoning:
        parts.append(f"> {reasoning[:600]}")
    return _send("\n".join(parts))


def _portfolio_lines(positions: list[dict]) -> list[str]:
    lines = []
    for p in positions:
        if p["type"] in ("call", "put"):
            lines.append(
                f"  {p['ticker']} {p['type'].upper()}{p['strike']} {p['expiry']}  "
                f"qty {p['qty']}  P/L ${p.get('unrealized_pl',0):+.2f}"
            )
        else:
            lines.append(
                f"  {p['ticker']:<6} qty {p['qty']:<8} avg ${p['avg_cost']:.2f} "
                f"now ${p.get('current_price',0):.2f}  P/L ${p.get('unrealized_pl',0):+.2f}"
            )
    return lines


def send_hourly_summary() -> bool:
    store = get_store()
    pf = store.get_portfolio()
    positions = store.open_positions()
    sp = market.benchmark_sp500()
    pl = pf["total_value"] - _INITIAL_EQUITY
    pl_pct = pl / _INITIAL_EQUITY * 100

    recent_trades = store.recent_trades(5)
    trade_lines = [
        f"  [{t['timestamp'][11:16]}] {t['action']} {t['qty']} {t['ticker']} @ ${t['price']:.2f}"
        for t in recent_trades
    ] or ["  (no trades yet)"]

    sp_line = f"S&P 500: {sp:.2f}" if sp else "S&P 500: N/A"

    body = (
        f"**HOURLY** ◈ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"```\n"
        f"Equity      ${pf['total_value']:.2f}\n"
        f"Cash        ${pf['cash']:.2f}\n"
        f"P/L         ${pl:+.2f} ({pl_pct:+.2f}%)\n"
        f"{sp_line}\n"
        f"```\n"
        f"**Positions**\n```\n"
        + ("\n".join(_portfolio_lines(positions)) or "  (none)")
        + "\n```\n**Recent trades**\n```\n"
        + "\n".join(trade_lines)
        + "\n```"
    )
    return _send(body)


def send_daily_close() -> bool:
    store = get_store()
    pf = store.get_portfolio()
    positions = store.open_positions()
    sp = market.benchmark_sp500()

    pl = pf["total_value"] - _INITIAL_EQUITY
    pl_pct = pl / _INITIAL_EQUITY * 100

    # daily slice
    today = datetime.now(timezone.utc).date().isoformat()
    todays_trades = [t for t in store.recent_trades(200) if t["timestamp"].startswith(today)]
    n_trades = len(todays_trades)
    pnl_real = sum(
        t["value"] if t["action"].startswith("SELL") else -t["value"]
        for t in todays_trades
    )

    sp_line = f"S&P 500: {sp:.2f}" if sp else "S&P 500: N/A"

    body = (
        f"**DAILY CLOSE** ◈ {today}\n"
        f"```\n"
        f"Equity         ${pf['total_value']:.2f}\n"
        f"Cash           ${pf['cash']:.2f}\n"
        f"Total P/L      ${pl:+.2f} ({pl_pct:+.2f}%)  vs $1000 start\n"
        f"Realized P/L (today, cash flow basis)  ${pnl_real:+.2f}\n"
        f"Trades today   {n_trades}\n"
        f"{sp_line}\n"
        f"```\n"
        f"**Open positions**\n```\n"
        + ("\n".join(_portfolio_lines(positions)) or "  (none)")
        + "\n```"
    )
    return _send(body)

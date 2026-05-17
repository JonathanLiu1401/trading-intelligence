"""Options chain monitor — tracks IV, OI, and unusual activity for portfolio tickers."""
import json
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent.parent
PORTFOLIO_PATH = BASE_DIR / "config" / "portfolio.json"

TICKERS = ["MU", "LITE", "MSFT", "AXTI", "ORCL", "TSEM", "QBTS", "NVDA", "AMD", "LRCX"]


def _get_options_snapshot(ticker: str) -> dict | None:
    try:
        t = yf.Ticker(ticker)
        expirations = t.options
        if not expirations:
            return None

        # use nearest expiry
        exp = expirations[0]
        chain = t.option_chain(exp)
        calls = chain.calls
        puts  = chain.puts

        if calls.empty or puts.empty:
            return None

        # put/call ratio by OI
        total_call_oi = int(calls["openInterest"].sum())
        total_put_oi  = int(puts["openInterest"].sum())
        pc_ratio = round(total_put_oi / total_call_oi, 2) if total_call_oi else None

        # ATM IV: call with strike closest to current price
        info = t.info or {}
        price = info.get("regularMarketPrice") or info.get("currentPrice")
        atm_iv = None
        if price:
            calls_sorted = calls.reindex(
                (calls["strike"] - price).abs().sort_values().index
            )
            if not calls_sorted.empty:
                atm_iv = round(float(calls_sorted.iloc[0].get("impliedVolatility", 0)) * 100, 1)

        # highest OI call/put strikes
        top_call_strike = float(calls.loc[calls["openInterest"].idxmax(), "strike"]) if not calls.empty else None
        top_put_strike  = float(puts.loc[puts["openInterest"].idxmax(), "strike"])  if not puts.empty else None

        return {
            "ticker": ticker,
            "expiry": exp,
            "pc_ratio": pc_ratio,
            "atm_iv_pct": atm_iv,
            "top_call_strike": top_call_strike,
            "top_put_strike": top_put_strike,
            "call_oi": total_call_oi,
            "put_oi": total_put_oi,
            "current_price": price,
        }
    except Exception as e:
        print(f"[options] {ticker}: {e}")
        return None


def get_options_data() -> list:
    results = []
    for ticker in TICKERS:
        snap = _get_options_snapshot(ticker)
        if snap:
            results.append(snap)
    return results


def format_options_block(data: list) -> str:
    if not data:
        return "N/A"
    lines = [
        f"{'TICKER':>6}  {'EXPIRY':>10}  {'ATM_IV':>7}  {'P/C':>5}  {'MAX_CALL':>9}  {'MAX_PUT':>9}",
        "─" * 65,
    ]
    for d in data:
        iv   = f"{d['atm_iv_pct']:>6.1f}%" if d.get("atm_iv_pct") else "   N/A "
        pc   = f"{d['pc_ratio']:>5.2f}"    if d.get("pc_ratio")   else "  N/A "
        mc   = f"${d['top_call_strike']:>8.0f}" if d.get("top_call_strike") else "      N/A"
        mp   = f"${d['top_put_strike']:>8.0f}"  if d.get("top_put_strike")  else "      N/A"
        lines.append(f"{d['ticker']:>6}  {d['expiry']:>10}  {iv}  {pc}  {mc}  {mp}")
    return "\n".join(lines)


if __name__ == "__main__":
    data = get_options_data()
    print(format_options_block(data))

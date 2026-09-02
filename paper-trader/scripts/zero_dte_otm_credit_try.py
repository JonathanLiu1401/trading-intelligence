#!/usr/bin/env python3
"""One-shot: try Lark's 0DTE far-OTM short-call experiment.

Naked short calls are engine-blocked. The legal version is a 1-lot
call credit spread (sell OTM, buy further OTM). Defined risk, same bet.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXPIRY = date.today().isoformat()


def _chain_calls(ticker: str, expiry: str):
    import yfinance as yf

    t = yf.Ticker(ticker)
    chain = t.option_chain(expiry)
    last = None
    try:
        last = float(t.fast_info.last_price)
    except Exception:
        last = None
    if last is None:
        hist = t.history(period='1d')
        if hist is None or hist.empty:
            raise RuntimeError('no spot for %s' % ticker)
        last = float(hist['Close'].iloc[-1])
    return chain.calls, last


def _pick(ticker: str):
    df, spot = _chain_calls(ticker, EXPIRY)
    if df is None or df.empty:
        return None
    rows = []
    for _, r in df.iterrows():
        k = float(r['strike'])
        if k < spot * 1.04:
            continue
        bid = float(r.get('bid') or 0)
        ask = float(r.get('ask') or 0)
        last = float(r.get('lastPrice') or 0)
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else last
        if mid <= 0.04:
            continue
        rows.append((k, mid, bid, ask))
    if len(rows) < 2:
        return None
    rows.sort(key=lambda x: x[0])
    for i, short in enumerate(rows[:-1]):
        for long in rows[i + 1:]:
            width = long[0] - short[0]
            if width <= 0:
                continue
            if width > 5.01:
                break
            credit = short[1] - long[1]
            if credit <= 0:
                continue
            return {
                'ticker': ticker,
                'spot': spot,
                'short_strike': short[0],
                'long_strike': long[0],
                'short_mid': short[1],
                'long_mid': long[1],
                'width': width,
                'est_credit': credit,
            }
    return None


def main() -> int:
    from paper_trader.store import Store
    from paper_trader import strategy

    store = Store()
    snap = strategy._portfolio_snapshot(store)
    print('BOOK', json.dumps({
        'equity': snap.get('total_value'),
        'cash': snap.get('cash'),
        'bp': snap.get('stock_buying_power'),
    }, default=str))

    pick = None
    last_err = None
    for ticker in ('QQQ', 'SPY', 'MU', 'NVDA'):
        try:
            pick = _pick(ticker)
        except Exception as e:
            last_err = '%s: %s: %s' % (ticker, type(e).__name__, e)
            print('CHAIN_FAIL', last_err)
            pick = None
        if pick:
            break
    if not pick:
        print('NO_PICK', last_err)
        return 3

    decision = {
        'action': 'SELL_CALL_SPREAD',
        'ticker': pick['ticker'],
        'qty': 1,
        'expiry': EXPIRY,
        'short_strike': pick['short_strike'],
        'long_strike': pick['long_strike'],
        'confidence': 0.4,
        'reasoning': (
            'Lark 2026-08-17 11:10 PDT: try shorting extreme 0DTE OTM calls. '
            'Naked shorts are engine-blocked; this is the 1-lot defined-risk '
            'credit version (sell OTM / buy further OTM). Experiment, not a new '
            'standing closer.'
        ),
    }
    print('PICK', json.dumps(pick, default=str))
    print('DECISION', json.dumps(decision, default=str))
    status, detail = strategy._execute(decision, snap, store)
    print('RESULT', status, detail)
    return 0 if status == 'FILLED' else 4


if __name__ == '__main__':
    raise SystemExit(main())

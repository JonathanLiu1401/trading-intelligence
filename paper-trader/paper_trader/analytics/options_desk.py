"""Options desk awareness for the live decision prompt.

Puts chain snapshots, DTE, IV, delta proxies, and strategy skills into the
Opus context so options are first-class alongside stocks. Observational only.
"""
from __future__ import annotations

from datetime import date
from typing import Any

_PREAMBLE = (
    "OPTIONS DESK (chain + strategy awareness -- facts for instrument choice "
    "only, NOT a directive; you retain complete autonomy; use BUY_CALL / "
    "BUY_PUT when options are the better expression of conviction, else stock). "
    "FRIDAY CLOSER: leftover Sep mark is NOT the $800 new-debit cap. One 8/14 "
    "NVDA/QQQ debit spread is the climb-to-10k send. Snapshot DTE~2 weeklies."
)

STRATEGY_SKILLS = [
    {
        "id": "leaps_call_proxy",
        "name": "Deep ITM LEAPS call (stock proxy)",
        "when": "High-conviction multi-month long; leveraged upside with defined premium risk",
        "prefer": "BUY_CALL delta>~0.70, DTE 180-720, strike deep ITM",
        "avoid": "Short-dated OTM lotto; earnings week unless sized tiny",
    },
    {
        "id": "swing_call",
        "name": "Swing call",
        "when": "Catalyst + technical confirmation, 2-6 week horizon",
        "prefer": "BUY_CALL DTE 21-60, delta 0.40-0.65, liquid strikes",
        "avoid": "DTE <14 unless the 8/14 Friday closer or an imminent sized flyer",
    },
    {
        "id": "protective_put",
        "name": "Protective / bearish put",
        "when": "Bearish catalyst or hedge on a long book name",
        "prefer": "BUY_PUT DTE 30-90, delta -0.35 to -0.55",
        "avoid": "Far OTM puts as default; naked short puts (unsupported open)",
    },
    {
        "id": "event_debit",
        "name": "Event debit option",
        "when": "Binary catalyst inside 14 DTE; shares too capital-heavy",
        "prefer": "BUY_CALL_SPREAD / BUY_PUT_SPREAD on the Friday weekly (8/14), not Sep leftovers",
        "avoid": "Selling premium into events; treating leftover Sep mark as the $800 cap",
    },
    {
        "id": "stock_when_better",
        "name": "Prefer shares",
        "when": "Need SHORT borrow, no expiry, or chain illiquid/wide",
        "prefer": "BUY / SHORT stock",
        "avoid": "Forcing options when bid/ask empty or IV unusable",
    },
    {
        "id": "covered_call",
        "name": "Covered call",
        "when": "Long >=100 shares, neutral/mildly bullish, want income",
        "prefer": "SELL_CALL OTM, DTE 21-45, against long stock",
        "avoid": "Naked short calls; calling away core high-conviction longs cheaply",
    },
    {
        "id": "csp",
        "name": "Cash-secured put",
        "when": "Want to buy stock lower / collect premium with cash collateral",
        "prefer": "SELL_PUT at support, DTE 21-45; cash >= strike*100 - premium",
        "avoid": "Undercollateralized puts; earnings CSP without reduced size",
    },
    {
        "id": "debit_call_spread",
        "name": "Bull call debit spread",
        "when": "Bullish but want cheaper/defined risk vs naked long call",
        "prefer": "BUY_CALL_SPREAD long_strike < short_strike same expiry",
        "avoid": "Ultra-wide spreads that recreate naked long premium risk",
    },
    {
        "id": "debit_put_spread",
        "name": "Bear put debit spread",
        "when": "Bearish with defined risk",
        "prefer": "BUY_PUT_SPREAD long_strike > short_strike same expiry",
        "avoid": "Far OTM cheap lottery put spreads as default",
    },
    {
        "id": "credit_put_spread",
        "name": "Bull put credit spread",
        "when": "Mildly bullish / range-bound; want defined-risk premium sell",
        "prefer": "SELL_PUT_SPREAD short higher put + long lower put",
        "avoid": "Selling naked puts when a spread fits; skinny credit vs width",
    },
    {
        "id": "credit_call_spread",
        "name": "Bear call credit spread",
        "when": "Mildly bearish / resistance overhead",
        "prefer": "SELL_CALL_SPREAD short lower call + long higher call",
        "avoid": "Naked short calls; credit too small vs width",
    },
]

def _f(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default

def _dte(expiry, today=None):
    if not expiry:
        return None
    try:
        exp = date.fromisoformat(str(expiry)[:10])
    except ValueError:
        return None
    base = today or date.today()
    return (exp - base).days

def _mid(row: dict):
    bid = _f(row.get('bid'), default=-1.0)
    ask = _f(row.get('ask'), default=-1.0)
    last = _f(row.get('lastPrice'), default=-1.0)
    if bid >= 0 and ask > 0:
        return (bid + ask) / 2.0
    if last > 0:
        return last
    if bid > 0:
        return bid
    if ask > 0:
        return ask
    return None

def _pick_strikes(rows, spot, side, limit=4):
    cleaned = []
    for r in rows or []:
        k = _f(r.get('strike'))
        if k <= 0:
            continue
        mid = _mid(r)
        if mid is None or mid <= 0:
            continue
        cleaned.append((k, r, mid))
    if not cleaned or spot <= 0:
        return []
    cleaned.sort(key=lambda t: t[0])
    atm_i = min(range(len(cleaned)), key=lambda i: abs(cleaned[i][0] - spot))
    idxs = {atm_i}
    if side == "call":
        if atm_i > 0:
            idxs.add(atm_i - 1)
        if atm_i + 1 < len(cleaned):
            idxs.add(atm_i + 1)
        if atm_i + 2 < len(cleaned):
            idxs.add(atm_i + 2)
    else:
        if atm_i + 1 < len(cleaned):
            idxs.add(atm_i + 1)
        if atm_i > 0:
            idxs.add(atm_i - 1)
        if atm_i > 1:
            idxs.add(atm_i - 2)
    out = []
    for i in sorted(idxs):
        k, r, mid = cleaned[i]
        iv = r.get('impliedVolatility')
        try:
            iv_f = float(iv) if iv is not None else None
        except (TypeError, ValueError):
            iv_f = None
        moneyness = (spot - k) / spot if side == "call" else (k - spot) / spot
        if side == "call":
            delta_proxy = max(0.05, min(0.95, 0.5 + moneyness * 2.5))
        else:
            delta_proxy = -max(0.05, min(0.95, 0.5 + moneyness * 2.5))
        label = "ATM"
        if side == "call":
            if k < spot * 0.98:
                label = "ITM"
            elif k > spot * 1.02:
                label = "OTM"
        else:
            if k > spot * 1.02:
                label = "ITM"
            elif k < spot * 0.98:
                label = "OTM"
        out.append({
            "strike": k,
            "mid": mid,
            "bid": _f(r.get('bid')),
            "ask": _f(r.get('ask')),
            "iv": iv_f,
            "volume": _f(r.get('volume')),
            "open_interest": _f(r.get('openInterest')),
            "delta_proxy": round(delta_proxy, 2),
            "moneyness_label": label,
            "notional_per_contract": round(mid * 100.0, 2),
        })
        if len(out) >= limit:
            break
    return out

def _chain_snapshot(ticker, target_dte, spot):
    try:
        from .. import market
    except Exception:
        from paper_trader import market  # type: ignore
    try:
        chain = market.get_options_chain(ticker, target_dte=target_dte)
    except Exception as e:
        return {"ticker": ticker, "error": f"chain_failed:{e}", "target_dte": target_dte}
    if not chain:
        return None
    exp = chain.get('expiry')
    dte = _dte(exp)
    spot_f = _f(spot) if spot else 0.0
    if spot_f <= 0:
        try:
            px = market.get_prices([ticker]).get(ticker)
            spot_f = _f(px)
        except Exception:
            spot_f = 0.0
    calls = _pick_strikes(list(chain.get('calls') or []), spot_f, "call")
    puts = _pick_strikes(list(chain.get('puts') or []), spot_f, "put")
    if not calls and not puts:
        return None
    return {
        "ticker": ticker.upper(),
        "expiry": exp,
        "dte": dte,
        "target_dte": target_dte,
        "spot": spot_f or None,
        "calls": calls,
        "puts": puts,
    }

def _open_option_facts(positions):
    facts = []
    for p in positions or []:
        if p.get('type') not in ("call", "put"):
            continue
        dte = _dte(p.get('expiry'))
        qty = _f(p.get('qty'))
        avg = _f(p.get('avg_cost'))
        mark = _f(p.get('current_price'), avg)
        mult = 100.0
        premium_at_risk = max(0.0, mark * qty * mult)
        u_pl = _f(p.get('unrealized_pl'))
        theta_pressure = "LOW"
        if dte is not None:
            if dte <= 7:
                theta_pressure = "CRITICAL"
            elif dte <= 21:
                theta_pressure = "HIGH"
            elif dte <= 45:
                theta_pressure = "MED"
        facts.append({
            "ticker": str(p.get("ticker", "")).upper(),
            "type": p.get('type'),
            "strike": _f(p.get('strike')),
            "expiry": p.get('expiry'),
            "dte": dte,
            "qty": qty,
            "avg": avg,
            "mark": mark,
            "unrealized_pl": u_pl,
            "premium_mark_value": premium_at_risk,
            "cost_basis": avg * qty * mult,
            "theta_pressure": theta_pressure,
        })
    return facts

def _candidate_tickers(positions, names_in_play, watch_prices, limit=4):
    held = []
    for p in positions or []:
        t = str(p.get("ticker", "")).upper()
        if t and t not in held:
            held.append(t)
    play = []
    for t in names_in_play or []:
        u = str(t).upper()
        if u and u not in held and u not in play:
            play.append(u)
    closer_first = ["NVDA", "QQQ"]
    out = []
    for t in closer_first + held + play:
        if t in out:
            continue
        out.append(t)
        if len(out) >= limit:
            break
    return out

def render_prompt_block(open_opts, chains, cash, strategies, buying_power=None, margin_available=None):
    lines = [_PREAMBLE]
    lines.append(
        "Instrument choice: treat options like stocks when conviction is high -- "
        "pick LEAPS / swing / covered call / CSP / vertical spreads / shares. "
        "SELL_CALL opens covered short if long shares else closes long calls; "
        "SELL_PUT opens CSP if cash secured else closes long puts; "
        "BUY_* against short lots = buy-to-cover. Naked short calls blocked."
    )
    lines.append("Strategy skills:")
    for s in strategies:
        lines.append(
            f"  - {s['name']}: WHEN {s['when']} | PREFER {s['prefer']} | AVOID {s['avoid']}"
        )
    try:
        bp = float(buying_power) if buying_power is not None else float(cash)
    except (TypeError, ValueError):
        bp = float(cash or 0.0)
    try:
        marg = float(margin_available) if margin_available is not None else max(0.0, bp - float(cash or 0.0))
    except (TypeError, ValueError):
        marg = max(0.0, bp - float(cash or 0.0))
    lines.append(
        f"Buying power for long option premium: ${bp:.2f} "
        f"(cash ${float(cash or 0.0):.2f} + margin ${marg:.2f}). "
        "One contract costs mid*100. Long premium may use the same cash+50% net-worth BP as stocks; "
        "cash can go negative within BP. CSP collateral still needs cash. "
        "Size premium budget explicitly in reasoning."
    )
    if open_opts:
        lines.append("Open option lots:")
        for o in open_opts:
            lines.append(
                f"  - {o['ticker']} {str(o['type']).upper()} {o['strike']} {o['expiry']} "
                f"dte={o['dte']} qty={o['qty']:.0f} avg={o['avg']:.2f} mark={o['mark']:.2f} "
                f"U-P/L=${o['unrealized_pl']:.2f} theta={o['theta_pressure']} "
                f"mark_value=${o['premium_mark_value']:.2f}"
            )
            if o.get('dte') is not None and o['dte'] <= 21:
                lines.append("    note: short DTE -- decide roll/close/hold deliberately")
    else:
        lines.append("Open option lots: (none)")
    if chains:
        lines.append("Chain snapshots (nearest listed expiry to target DTE):")
        for c in chains:
            if c.get('error'):
                lines.append(f"  - {c.get('ticker')} target_dte={c.get('target_dte')}: {c['error']}")
                continue
            lines.append(
                f"  - {c['ticker']} exp={c.get('expiry')} dte={c.get('dte')} "
                f"(target {c.get('target_dte')}) spot={c.get('spot')}"
            )
            if c.get('calls'):
                bits = []
                for x in c['calls'][:4]:
                    iv = f" iv={x['iv']:.2f}" if isinstance(x.get('iv'), (int, float)) else ""
                    bits.append(
                        f"{x['moneyness_label']} k={x['strike']:g} mid={x['mid']:.2f} "
                        f"d~{x['delta_proxy']} cost/ct=${x['notional_per_contract']:.0f}{iv}"
                    )
                lines.append("      calls: " + " | ".join(bits))
            if c.get('puts'):
                bits = []
                for x in c['puts'][:4]:
                    iv = f" iv={x['iv']:.2f}" if isinstance(x.get('iv'), (int, float)) else ""
                    bits.append(
                        f"{x['moneyness_label']} k={x['strike']:g} mid={x['mid']:.2f} "
                        f"d~{x['delta_proxy']} cost/ct=${x['notional_per_contract']:.0f}{iv}"
                    )
                lines.append("      puts: " + " | ".join(bits))
    else:
        lines.append("Chain snapshots: (none this cycle -- still may trade options if strike/expiry known)")
    lines.append(
        "When options beat shares: high conviction + defined risk, leverage without margin stock, "
        "asymmetric catalyst. When shares beat options: no expiry wanted, short stock needed, "
        "wide/illiquid chain. Always include strike+expiry on option actions."
    )
    return chr(10).join(lines)

def build_options_desk(snapshot, watch_prices, names_in_play=None, max_underlyings=3, fetch_chains=True):
    """Compose options desk context. Never raises."""
    try:
        snap = snapshot or {}
        positions = list(snap.get('positions') or [])
        open_opts = _open_option_facts(positions)
        cash = _f(snap.get('cash'))
        total = _f(snap.get('total_value'), cash)
        if 'stock_buying_power' in snap:
            buying_power = _f(snap.get('stock_buying_power'), cash)
        else:
            buying_power = cash + max(0.0, total) * 0.50
        if 'margin_available' in snap:
            margin_available = _f(snap.get('margin_available'), max(0.0, buying_power - cash))
        else:
            margin_available = max(0.0, total) * 0.50
        tickers = _candidate_tickers(positions, names_in_play, watch_prices or {}, limit=max_underlyings)
        chains = []
        if fetch_chains:
            for t in tickers:
                spot = (watch_prices or {}).get(t)
                for target in (2, 45, 365):
                    snap_c = _chain_snapshot(t, target, spot if isinstance(spot, (int, float)) else None)
                    if snap_c and (snap_c.get('calls') or snap_c.get('puts') or snap_c.get('error')):
                        chains.append(snap_c)
        block = render_prompt_block(
            open_opts, chains, cash, STRATEGY_SKILLS,
            buying_power=buying_power, margin_available=margin_available,
        )
        # Advanced risk + structure scanner (observational)
        try:
            from .options_risk import build_options_risk, scan_structure_opportunities
            from .options_structures import structure_menu_lines
            risk = build_options_risk(snap, watch_prices)
            opps = scan_structure_opportunities(chains, cash, positions)
            extra = []
            extra.extend(structure_menu_lines())
            if risk.get("prompt_block"):
                extra.append(risk["prompt_block"])
            if opps.get("prompt_block"):
                extra.append(opps["prompt_block"])
            if extra:
                block = (block or "") + chr(10) + chr(10).join(extra)
        except Exception as e:
            risk, opps = {"error": str(e)}, {"ideas": []}
            block = (block or "") + chr(10) + ("OPTIONS ADVANCED layer failed: %s" % e)
        state = "ACTIVE" if (open_opts or chains) else "EMPTY"
        return {
            "state": state,
            "prompt_block": block,
            "open_options": open_opts,
            "chains": chains,
            "strategies": STRATEGY_SKILLS,
            "cash": cash,
            "buying_power": buying_power,
            "margin_available": margin_available,
            "tickers": tickers,
            "risk": risk if isinstance(risk, dict) else {},
            "opportunities": (opps or {}).get("ideas") if isinstance(opps, dict) else [],
            "summary": "open_option_lots=%d chain_snapshots=%d underlyings=%s ideas=%d" % (
                len(open_opts), len(chains), chr(44).join(tickers) or '-',
                len((opps or {}).get("ideas") or []) if isinstance(opps, dict) else 0,
            ),
        }
    except Exception as e:
        return {
            "state": "ERROR",
            "prompt_block": None,
            "open_options": [],
            "chains": [],
            "strategies": STRATEGY_SKILLS,
            "error": str(e),
            "summary": f"options desk error: {e}",
        }

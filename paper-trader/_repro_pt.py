"""Standalone repro of /api/position-thesis to surface the swallowed traceback.
Mirrors dashboard.position_thesis_api() step-by-step, printing where it throws.
"""
import sys, traceback

def step(name, fn):
    try:
        r = fn()
        print(f"OK   {name}")
        return r
    except Exception as e:
        print(f"FAIL {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise

from paper_trader.analytics.position_thesis import build_thesis_cards
from paper_trader.ml.decision_scorer import DecisionScorer
from paper_trader.strategy import get_quant_signals_live
from paper_trader import signals as _sig
from paper_trader.store import get_store

store = step("get_store", lambda: get_store())
positions = step("store.open_positions", lambda: store.open_positions())
held = sorted({p["ticker"] for p in positions
               if p.get("type") == "stock" and (p.get("qty") or 0) > 0})
print("held:", held)
quant = step("get_quant_signals_live(held)", lambda: get_quant_signals_live(held) if held else {})
sent_list = step("signals.ticker_sentiments", lambda: _sig.ticker_sentiments(held, hours=4) if held else [])
spy_q = step("get_quant_signals_live(SPY)", lambda: (get_quant_signals_live(["SPY"]) or {}).get("SPY") or {})
scorer = step("DecisionScorer()", lambda: DecisionScorer())
decisions = step("store.recent_decisions(80)", lambda: store.recent_decisions(limit=80))
out = step("build_thesis_cards", lambda: build_thesis_cards(positions, decisions, [], quant))
print("DONE n_positions:", out.get("n_positions"))

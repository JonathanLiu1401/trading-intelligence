"""Options desk + trade-memory learning surfaces."""
from __future__ import annotations

import json
from pathlib import Path

from paper_trader.analytics import claude_mem_trades, options_desk


def test_render_prompt_block_includes_skills_and_cash():
    block = options_desk.render_prompt_block(
        open_opts=[],
        chains=[],
        cash=12345.0,
        strategies=options_desk.STRATEGY_SKILLS,
        buying_power=18517.5,
        margin_available=6172.5,
    )
    assert block is not None
    assert "OPTIONS DESK" in block
    assert "Strategy skills:" in block
    assert "Deep ITM LEAPS" in block
    assert "12345.00" in block
    assert "18517.50" in block
    assert "Buying power for long option premium" in block
    assert "BUY_CALL" in block or "BUY_PUT" in block


def test_open_option_facts_theta_buckets():
    facts = options_desk._open_option_facts(
        [
            {
                "ticker": "AAPL",
                "type": "call",
                "strike": 200,
                "expiry": "2099-01-01",
                "qty": 1,
                "avg_cost": 5.0,
                "current_price": 6.0,
                "unrealized_pl": 100.0,
            },
            {
                "ticker": "MSFT",
                "type": "stock",
                "qty": 10,
                "avg_cost": 400,
                "current_price": 410,
            },
        ]
    )
    assert len(facts) == 1
    assert facts[0]["ticker"] == "AAPL"
    assert facts[0]["premium_mark_value"] == 600.0
    assert facts[0]["theta_pressure"] in {"LOW", "MED", "HIGH", "CRITICAL"}


def test_build_options_desk_no_network():
    snap = {
        "cash": 5000.0,
        "positions": [
            {
                "ticker": "AAPL",
                "type": "call",
                "strike": 180,
                "expiry": "2099-06-01",
                "qty": 2,
                "avg_cost": 4.0,
                "current_price": 5.0,
                "unrealized_pl": 200.0,
            }
        ],
    }
    out = options_desk.build_options_desk(
        snap,
        {"AAPL": 190.0},
        ["AAPL"],
        max_underlyings=1,
        fetch_chains=False,
    )
    assert out["state"] in {"ACTIVE", "EMPTY", "ERROR"}
    assert out.get("prompt_block")
    assert "Open option lots:" in out["prompt_block"]
    assert "AAPL" in out["prompt_block"]
    assert out["open_options"][0]["qty"] == 2


def test_pick_strikes_ladder():
    rows = [
        {"strike": 90, "bid": 12, "ask": 13, "lastPrice": 12.5, "impliedVolatility": 0.3},
        {"strike": 100, "bid": 4, "ask": 4.5, "lastPrice": 4.2, "impliedVolatility": 0.35},
        {"strike": 110, "bid": 1, "ask": 1.2, "lastPrice": 1.1, "impliedVolatility": 0.4},
    ]
    out = options_desk._pick_strikes(rows, spot=100.0, side="call", limit=4)
    assert out
    strikes = [x["strike"] for x in out]
    assert 100 in strikes
    assert all(x["notional_per_contract"] > 0 for x in out)


def test_trade_memory_local_jsonl(tmp_path: Path):
    decision = {
        "action": "BUY_CALL",
        "ticker": "AAPL",
        "qty": 1,
        "strike": 200,
        "expiry": "2026-12-18",
        "confidence": 0.8,
        "reasoning": "unit-test option fill",
    }
    res = claude_mem_trades.record_trade_learning(
        decision=decision,
        status="FILLED",
        detail="FILLED @ 3.50",
        snapshot_before={"cash": 10000, "total_value": 10000},
        snapshot_after={"cash": 9650, "total_value": 10050},
        push_remote=False,
        memory_dir=tmp_path,
    )
    assert res.get("ok") is True
    path = Path(res["local_path"])
    assert path.exists()
    row = json.loads(path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert row["action"] == "BUY_CALL"
    assert row["is_option"] is True
    assert row["ticker"] == "AAPL"
    recent = claude_mem_trades.recent_trade_memory(limit=5, memory_dir=tmp_path)
    assert recent
    assert recent[-1]["action"] == "BUY_CALL"


def test_trade_outcomes_pairs_option_round_trips():
    from paper_trader.analytics.trade_outcomes import _pair_trades

    trades = [
        {
            "timestamp": "2026-07-01T10:00:00",
            "ticker": "AAPL",
            "action": "BUY_CALL",
            "price": 4.0,
            "option_type": "call",
            "strike": 200,
            "expiry": "2026-12-18",
            "reason": "open",
        },
        {
            "timestamp": "2026-07-10T10:00:00",
            "ticker": "AAPL",
            "action": "SELL_CALL",
            "price": 6.0,
            "option_type": "call",
            "strike": 200,
            "expiry": "2026-12-18",
            "reason": "take profit",
        },
        {
            "timestamp": "2026-07-02T10:00:00",
            "ticker": "MSFT",
            "action": "BUY",
            "price": 400,
            "reason": "stock open",
        },
        {
            "timestamp": "2026-07-05T10:00:00",
            "ticker": "MSFT",
            "action": "SELL",
            "price": 420,
            "reason": "stock close",
        },
    ]
    rts = _pair_trades(trades)
    assert len(rts) == 2
    opt = next(r for r in rts if r.get("instrument") == "option")
    stk = next(r for r in rts if r.get("instrument") == "stock")
    assert opt["ticker"] == "AAPL"
    assert opt["pnl_pct"] == 50.0
    assert opt["win"] is True
    assert stk["ticker"] == "MSFT"
    assert stk["pnl_pct"] == 5.0


def test_strategy_payload_includes_options_desk_section():
    from paper_trader import strategy

    snap = {
        "cash": 8000.0,
        "total_value": 12000.0,
        "open_value": 4000.0,
        "positions": [],
        "deployed_pct": 0.0,
    }
    block = "OPTIONS DESK TEST BLOCK\n  - LEAPS skill present"
    payload = strategy._build_payload(
        snap,
        top_signals=[],
        sentiments=[],
        watch_prices={"AAPL": 200.0},
        futures_prices={},
        sp500=5000.0,
        market_open=True,
        options_desk_block=block,
    )
    assert "OPTIONS DESK TEST BLOCK" in payload

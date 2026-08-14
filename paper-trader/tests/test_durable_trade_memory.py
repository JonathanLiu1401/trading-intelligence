"""Durable order/lesson memory for paper-trader decision cycles."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from paper_trader.analytics import durable_trade_memory as dtm
from paper_trader.analytics import claude_mem_trades as cmt


def _row(ts, action, ticker, status="FILLED", detail="", reasoning=""):
    return {
        "ts": ts,
        "kind": "paper_trade",
        "status": status,
        "action": action,
        "ticker": ticker,
        "qty": 1,
        "detail": detail,
        "reasoning": reasoning,
        "is_option": False,
    }


def test_detect_same_name_flip_lesson():
    now = datetime(2026, 7, 20, 18, 0, tzinfo=timezone.utc)
    rows = [
        _row((now - timedelta(hours=2)).isoformat(), "BUY", "AAPL", detail="BUY 5 AAPL", reasoning="chase breakout"),
        _row(now.isoformat(), "SELL", "AAPL", detail="SELL 5 AAPL", reasoning="fade rejection"),
    ]
    lessons = dtm.detect_lessons_from_orders(rows)
    assert any(l.get("code") == "same_name_flip" and l.get("ticker") == "AAPL" for l in lessons)


def test_build_durable_memory_prompt_block(tmp_path: Path):
    day = tmp_path / "trades-2026-07-20.jsonl"
    now = datetime(2026, 7, 20, 18, 0, tzinfo=timezone.utc)
    rows = [
        _row((now - timedelta(hours=3)).isoformat(), "BUY", "MSFT", detail="BUY 2 MSFT", reasoning="add leader"),
        _row((now - timedelta(hours=1)).isoformat(), "BUY", "AAPL", detail="BUY 5 AAPL", reasoning="clean setup"),
        _row(now.isoformat(), "SELL", "AAPL", detail="SELL 5 AAPL", reasoning="intraday invert"),
    ]
    day.write_text(chr(10).join(json.dumps(r) for r in rows) + chr(10), encoding="utf-8")
    dtm.refresh_lessons_log(memory_dir=tmp_path)
    block = dtm.build_durable_memory_prompt_block(memory_dir=tmp_path, names_in_play={"AAPL", "MSFT"})
    assert block is not None
    assert "DURABLE TRADE MEMORY" in block
    assert "AAPL" in block
    assert "same_name_flip" in block or "flipped" in block
    assert (tmp_path / "RECENT_ORDERS.md").exists()
    assert (tmp_path / "LESSONS.md").exists()


def test_record_trade_learning_writes_crumb_and_lessons(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(cmt, "push_claude_mem_observation", lambda *a, **k: {"ok": True})
    out = cmt.record_trade_learning(
        decision={"action": "BUY", "ticker": "NVDA", "qty": 1, "reasoning": "test", "confidence": 0.5},
        status="FILLED",
        detail="BUY 1 NVDA @ 100",
        snapshot_before={"cash": 1000, "total_value": 1000},
        snapshot_after={"cash": 900, "total_value": 1000},
        push_remote=False,
        memory_dir=tmp_path,
    )
    assert out.get("ok") is True
    assert out.get("crumb_path")
    assert list((tmp_path / "crumbs").glob("*.md"))
    assert (tmp_path / "LESSONS.md").exists()


def test_strategy_payload_includes_durable_memory():
    from paper_trader import strategy
    snap = {
        "cash": 1000.0,
        "total_value": 1000.0,
        "open_value": 0.0,
        "stock_buying_power": 1500.0,
        "margin_available": 500.0,
        "positions": [],
    }
    payload = strategy._build_payload(
        snap, [], [], {}, {}, None, False,
        durable_memory_block="DURABLE TRADE MEMORY\nRecent orders: test",
    )
    assert "DURABLE TRADE MEMORY" in payload

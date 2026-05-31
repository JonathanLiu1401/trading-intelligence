"""Tests for exit_proximity's ``prompt_block`` field — the live-decision
prompt section that surfaces mechanical SL/TP proximity to Opus.

Asserts specific rendered strings, silence on COMFORTABLE / NO_DATA /
NO_SL_TP_SET, ranking by actionability, max-row cap, threshold tokens.
Pairs with the strategy-wiring tests in ``test_exit_proximity_prompt_wiring.py``.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from paper_trader.analytics.exit_proximity import (
    PROMPT_BLOCK_MAX_ROWS,
    _PROMPT_BLOCK_VERDICTS,
    _render_prompt_block,
    build_exit_proximity,
)


FIXED_NOW = datetime(2026, 5, 25, 12, 0, 0, tzinfo=timezone.utc)


def _mk_pos(
    ticker: str = "NVDA",
    qty: float = 10.0,
    avg: float = 100.0,
    cur: float | None = 100.0,
    sl: float | None = 98.0,
    tp: float | None = 103.0,
    ptype: str = "stock",
):
    return {
        "ticker": ticker,
        "type": ptype,
        "qty": qty,
        "avg_cost": avg,
        "current_price": cur,
        "stop_loss_price": sl,
        "take_profit_price": tp,
    }


# ─────────────────────── prompt_block silence ───────────────────────

class TestPromptBlockSilence:
    def test_no_positions_returns_none(self):
        out = build_exit_proximity([], now=FIXED_NOW)
        assert out["verdict"] == "NO_DATA"
        assert out["prompt_block"] is None

    def test_no_sl_tp_set_returns_none(self):
        out = build_exit_proximity(
            [_mk_pos(sl=None, tp=None)], now=FIXED_NOW
        )
        assert out["verdict"] == "NO_SL_TP_SET"
        assert out["prompt_block"] is None

    def test_comfortable_book_returns_none(self):
        # mid-band — all positions safely centered
        out = build_exit_proximity(
            [_mk_pos(cur=100.0, sl=98.0, tp=103.0)], now=FIXED_NOW
        )
        assert out["verdict"] == "COMFORTABLE"
        assert out["prompt_block"] is None

    def test_silence_set_pins_exact_verdicts(self):
        # Regression guard: if a future verdict rename silently drops
        # AT_RISK / NEAR_THRESHOLD from the actionable set, the prompt
        # block would silently go dark on real risk. Pin it.
        assert _PROMPT_BLOCK_VERDICTS == frozenset(
            {"AT_RISK", "NEAR_THRESHOLD"}
        )


# ─────────────────────── prompt_block AT_RISK ───────────────────────

class TestPromptBlockAtRiskSL:
    def test_at_risk_sl_renders_block(self):
        # current=97 < SL=98 → AT_RISK_SL (corridor_pos = -0.2)
        out = build_exit_proximity(
            [_mk_pos(cur=97.0, sl=98.0, tp=103.0)], now=FIXED_NOW
        )
        assert out["verdict"] == "AT_RISK"
        pb = out["prompt_block"]
        assert pb is not None
        assert pb.startswith("EXIT PROXIMITY (AT_RISK):")
        # SSOT headline carried verbatim (single source of truth — same
        # text as /api/exit-proximity and the Discord hourly).
        assert out["headline"] in pb
        # Per-row line includes the ticker + band + distance + thresholds.
        assert "NVDA" in pb
        assert "[AT_RISK_SL]" in pb
        # mark < SL → dist_to_sl_pct negative; closer is SL
        assert "from SL" in pb
        assert "mark $97.00" in pb
        assert "SL $98.00" in pb
        assert "TP $103.00" in pb
        # Closing advisory pinned (Opus must see it is mechanical, not gating)
        assert "advisory" in pb


class TestPromptBlockAtRiskTP:
    def test_at_risk_tp_renders_block_with_tp_side(self):
        # current=104 > TP=103 → AT_RISK_TP (corridor_pos > 1)
        out = build_exit_proximity(
            [_mk_pos(cur=104.0, sl=98.0, tp=103.0)], now=FIXED_NOW
        )
        assert out["verdict"] == "AT_RISK"
        pb = out["prompt_block"]
        assert pb is not None
        # mark above TP → closer is TP
        assert "from TP" in pb
        # Distance to TP is signed negative (we already crossed it)
        # dist_to_tp_pct = (103-104)/104 * 100 = -0.96%
        assert "-0.96% from TP" in pb


# ─────────────────────── prompt_block NEAR_THRESHOLD ───────────────────────

class TestPromptBlockNearThreshold:
    def test_near_sl_renders_block(self):
        # corridor_pos = 0.1 (in [0, 0.25)) → NEAR_SL
        out = build_exit_proximity(
            [_mk_pos(cur=98.5, sl=98.0, tp=103.0)], now=FIXED_NOW
        )
        assert out["verdict"] == "NEAR_THRESHOLD"
        pb = out["prompt_block"]
        assert pb is not None
        assert pb.startswith("EXIT PROXIMITY (NEAR_THRESHOLD):")
        assert "[NEAR_SL]" in pb
        # closer is SL (0.5/98.5 = 0.51% vs 4.57% to TP)
        assert "from SL" in pb

    def test_near_tp_renders_block(self):
        # corridor_pos = 0.9 (in [0.75, 1.0]) → NEAR_TP
        out = build_exit_proximity(
            [_mk_pos(cur=102.5, sl=98.0, tp=103.0)], now=FIXED_NOW
        )
        assert out["verdict"] == "NEAR_THRESHOLD"
        pb = out["prompt_block"]
        assert pb is not None
        assert "[NEAR_TP]" in pb
        assert "from TP" in pb


# ─────────────────────── ranking + cap ───────────────────────

class TestPromptBlockRanking:
    def test_at_risk_row_ranks_before_near_in_block(self):
        rows = [
            _mk_pos(ticker="NEAR1", cur=98.5, sl=98.0, tp=103.0),    # NEAR_SL
            _mk_pos(ticker="ATRISK", cur=97.0, sl=98.0, tp=103.0),   # AT_RISK_SL
        ]
        out = build_exit_proximity(rows, now=FIXED_NOW)
        pb = out["prompt_block"]
        assert pb is not None
        # The AT_RISK line must appear BEFORE the NEAR_SL line.
        atrisk_pos = pb.index("ATRISK")
        near_pos = pb.index("NEAR1")
        assert atrisk_pos < near_pos, (
            f"AT_RISK row should rank above NEAR_SL row in prompt block; "
            f"got:\n{pb}"
        )

    def test_block_caps_to_max_rows(self):
        # 7 NEAR_SL positions; cap is 5
        positions = [
            _mk_pos(ticker=f"NEAR{i}", cur=98.5 + i * 0.001,
                    sl=98.0, tp=103.0)
            for i in range(7)
        ]
        out = build_exit_proximity(positions, now=FIXED_NOW)
        pb = out["prompt_block"]
        assert pb is not None
        # Count per-row bullets (start with "  • ").
        bullets = [
            ln for ln in pb.splitlines() if ln.startswith("  • ")
        ]
        assert len(bullets) == PROMPT_BLOCK_MAX_ROWS == 5

    def test_block_excludes_non_actionable_rows(self):
        # Mix: 1 AT_RISK_SL + 1 MID_BAND + 1 NO_SL_TP. Only AT_RISK should appear.
        positions = [
            _mk_pos(ticker="ATR", cur=97.0, sl=98.0, tp=103.0),
            _mk_pos(ticker="MID", cur=100.0, sl=98.0, tp=103.0),
            _mk_pos(ticker="NOSL", cur=100.0, sl=None, tp=None),
        ]
        out = build_exit_proximity(positions, now=FIXED_NOW)
        pb = out["prompt_block"]
        assert pb is not None
        assert "ATR" in pb
        assert "MID" not in pb
        assert "NOSL" not in pb


# ─────────────────────── _render_prompt_block direct ───────────────────────

class TestRenderPromptBlockDirect:
    def test_comfortable_returns_none(self):
        assert _render_prompt_block("COMFORTABLE", "hl", []) is None

    def test_no_data_returns_none(self):
        assert _render_prompt_block("NO_DATA", "hl", []) is None

    def test_no_sl_tp_set_returns_none(self):
        assert _render_prompt_block("NO_SL_TP_SET", "hl", []) is None

    def test_at_risk_with_empty_rows_returns_none(self):
        # Defensive: verdict says actionable but no rows fall in the
        # actionable bands → silent rather than emit an empty bullet list.
        assert _render_prompt_block("AT_RISK", "hl", []) is None

    def test_render_picks_closer_target(self):
        # SL=98, TP=103, current=98.5 → corridor 0.1
        # dist_sl = 0.5/98.5 = 0.51%, dist_tp = 4.5/98.5 = 4.57%
        # closer should be SL.
        rows = [{
            "ticker": "NVDA",
            "proximity_band": "NEAR_SL",
            "dist_to_sl_pct": 0.51,
            "dist_to_tp_pct": 4.57,
            "closer_target": "SL",
            "current_price": 98.5,
            "stop_loss_price": 98.0,
            "take_profit_price": 103.0,
        }]
        out = _render_prompt_block("NEAR_THRESHOLD", "near-headline", rows)
        assert out is not None
        assert "+0.51% from SL" in out

    def test_render_falls_back_when_closer_target_unset(self):
        # No closer_target → both-sides format
        rows = [{
            "ticker": "X",
            "proximity_band": "NEAR_SL",
            "dist_to_sl_pct": 1.0,
            "dist_to_tp_pct": 4.0,
            "closer_target": "",
            "current_price": 99.0,
            "stop_loss_price": 98.0,
            "take_profit_price": 103.0,
        }]
        out = _render_prompt_block("NEAR_THRESHOLD", "h", rows)
        assert out is not None
        assert "SL +1.00% / TP +4.00%" in out

    def test_render_threshold_tag_dropped_when_any_field_missing(self):
        rows = [{
            "ticker": "X",
            "proximity_band": "NEAR_SL",
            "dist_to_sl_pct": 1.0,
            "dist_to_tp_pct": 4.0,
            "closer_target": "SL",
            "current_price": None,            # missing → drop threshold tag
            "stop_loss_price": 98.0,
            "take_profit_price": 103.0,
        }]
        out = _render_prompt_block("NEAR_THRESHOLD", "h", rows)
        assert out is not None
        # No threshold parenthetical present
        assert "mark $" not in out

    def test_max_rows_param_overrides_cap(self):
        # Generate 4 rows but pass max_rows=2; only 2 bullets in output.
        rows = [
            {
                "ticker": f"T{i}",
                "proximity_band": "NEAR_SL",
                "dist_to_sl_pct": 1.0 + i * 0.1,
                "dist_to_tp_pct": 4.0,
                "closer_target": "SL",
                "current_price": 99.0,
                "stop_loss_price": 98.0,
                "take_profit_price": 103.0,
            }
            for i in range(4)
        ]
        out = _render_prompt_block("NEAR_THRESHOLD", "h", rows, max_rows=2)
        assert out is not None
        bullets = [ln for ln in out.splitlines() if ln.startswith("  • ")]
        assert len(bullets) == 2


# ─────────────────────── degrade-safe ───────────────────────

class TestPromptBlockDegradeSafe:
    def test_garbage_position_does_not_raise(self):
        out = build_exit_proximity([{}, None, "not-a-dict"],  # type: ignore[list-item]
                                   now=FIXED_NOW)
        # Either NO_DATA (no rows survived) or NO_SL_TP_SET — both silent.
        assert out["prompt_block"] is None

    def test_options_position_doesnt_render_in_block(self):
        out = build_exit_proximity(
            [_mk_pos(ptype="call", cur=2.0, sl=None, tp=None)],
            now=FIXED_NOW,
        )
        # Options have no SL/TP enforcement → NO_SL_TP_SET → silent.
        assert out["prompt_block"] is None

"""Tests for paper_trader.analytics.first_trade_after_drought.

The verdict drives operator interpretation — PANIC_FLIP ("the bot reverses
on resume"), PANIC_RECYCLE ("it grabs familiar names"), DROUGHT_EXPLORATION
("uses the pause to step back"), STEADY_RECOVERY ("nothing unusual").
A silently-broken drought→trade pairing or wrong flip-window calculation
would misdirect, so every assertion below is on a *specific* expected
verdict / count / rate.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics import first_trade_after_drought as ftad  # noqa: E402

_BASE = datetime(2026, 5, 24, 12, 0, 0, tzinfo=timezone.utc)


def _decision(action_taken: str, minutes_offset: float) -> dict:
    """One ``store.recent_decisions()`` row at ``_BASE + minutes_offset``."""
    ts = (_BASE + timedelta(minutes=minutes_offset)).isoformat()
    return {"action_taken": action_taken, "timestamp": ts, "reasoning": ""}


def _trade(action: str, ticker: str, minutes_offset: float) -> dict:
    ts = (_BASE + timedelta(minutes=minutes_offset)).isoformat()
    return {"action": action, "ticker": ticker, "timestamp": ts,
            "qty": 1.0, "price": 100.0, "value": 100.0}


def _newest_first(*chronological: dict) -> list[dict]:
    """Tests express chronological order; ``store.recent_*`` returns newest-first."""
    return list(reversed(chronological))


# ─── canonical predicate drift-lock ───────────────────────────────────

def test_is_no_decision_mirrors_canonical():
    # Must accept exactly NO_DECISION + empty + whitespace.
    assert ftad._is_no_decision("NO_DECISION") is True
    assert ftad._is_no_decision("") is True
    assert ftad._is_no_decision("   ") is True
    assert ftad._is_no_decision(None) is True
    assert ftad._is_no_decision("BUY NVDA → FILLED") is False
    assert ftad._is_no_decision("HOLD NVDA → HOLD") is False
    # BLOCKED is a *real* decision the risk gate refused — not a no-decision.
    assert ftad._is_no_decision("BLOCKED") is False


def test_is_filled_matches_action_taken_shape():
    assert ftad._is_filled("BUY NVDA → FILLED") is True
    assert ftad._is_filled("SELL TQQQ → FILLED") is True
    assert ftad._is_filled("HOLD NVDA → HOLD") is False
    assert ftad._is_filled("NO_DECISION") is False
    assert ftad._is_filled("") is False


# ─── NO_DATA / empty branches ─────────────────────────────────────────

def test_no_decisions_returns_no_data():
    out = ftad.build_first_trade_after_drought([], [])
    assert out["state"] == "NO_DATA"
    assert out["verdict"] == "NO_DATA"
    assert out["n_droughts"] == 0


def test_no_droughts_returns_no_data_with_drought_count_zero():
    # All real decisions; no NO_DECISION runs at all.
    decisions = _newest_first(
        _decision("BUY NVDA → FILLED", 0),
        _decision("HOLD NVDA → HOLD", 1),
        _decision("BUY AMD → FILLED", 2),
    )
    trades = _newest_first(
        _trade("BUY", "NVDA", 0),
        _trade("BUY", "AMD", 2),
    )
    out = ftad.build_first_trade_after_drought(decisions, trades)
    assert out["state"] == "NO_DATA"
    assert out["n_droughts"] == 0


def test_short_no_decision_run_not_counted_as_drought():
    # 4-cycle NO_DECISION run — below the default min_drought_run=5.
    decisions = _newest_first(
        _decision("BUY NVDA → FILLED", 0),
        _decision("NO_DECISION", 1),
        _decision("NO_DECISION", 2),
        _decision("NO_DECISION", 3),
        _decision("NO_DECISION", 4),
        _decision("BUY AMD → FILLED", 5),
    )
    trades = _newest_first(_trade("BUY", "AMD", 5), _trade("BUY", "NVDA", 0))
    out = ftad.build_first_trade_after_drought(decisions, trades)
    assert out["n_droughts"] == 0


# ─── drought identification ───────────────────────────────────────────

def test_completed_drought_of_min_length_counts():
    # 5 NO_DECISION cycles in the middle, sandwiched by FILLED decisions.
    decisions = _newest_first(
        _decision("BUY NVDA → FILLED", 0),
        _decision("NO_DECISION", 1),
        _decision("NO_DECISION", 2),
        _decision("NO_DECISION", 3),
        _decision("NO_DECISION", 4),
        _decision("NO_DECISION", 5),
        _decision("BUY AMD → FILLED", 6),  # post-drought trade
    )
    trades = _newest_first(
        _trade("BUY", "AMD", 6),  # the post-drought trade
        _trade("BUY", "NVDA", 0),
    )
    out = ftad.build_first_trade_after_drought(decisions, trades)
    assert out["n_droughts"] == 1
    assert out["n_post_drought_trades"] == 1
    rec = out["post_drought_trades"][0]
    assert rec["ticker"] == "AMD"
    assert rec["action"] == "BUY"


def test_open_drought_at_end_excluded():
    # Drought is *open* — the NEWEST decisions are NO_DECISION and no
    # FILLED row comes after them. Chronologically: FILLED first, then
    # the open NO_DECISION run trails to "now". Builder must NOT count
    # an open run as a drought (no post-drought trade exists yet).
    decisions = _newest_first(
        _decision("BUY NVDA → FILLED", 0),  # oldest
        _decision("NO_DECISION", 1),
        _decision("NO_DECISION", 2),
        _decision("NO_DECISION", 3),
        _decision("NO_DECISION", 4),
        _decision("NO_DECISION", 5),         # newest — open run trails here
    )
    trades = _newest_first(_trade("BUY", "NVDA", 0))
    out = ftad.build_first_trade_after_drought(decisions, trades)
    assert out["n_droughts"] == 0


# ─── new-ticker classification ────────────────────────────────────────

def test_first_buy_of_ticker_is_new_ticker_true():
    # First-ever BUY of AMD after a drought is is_new_ticker=True.
    decisions = _newest_first(
        _decision("BUY NVDA → FILLED", 0),
        _decision("NO_DECISION", 1),
        _decision("NO_DECISION", 2),
        _decision("NO_DECISION", 3),
        _decision("NO_DECISION", 4),
        _decision("NO_DECISION", 5),
        _decision("BUY AMD → FILLED", 6),
    )
    trades = _newest_first(
        _trade("BUY", "AMD", 6),
        _trade("BUY", "NVDA", 0),
    )
    out = ftad.build_first_trade_after_drought(decisions, trades)
    assert out["post_drought_trades"][0]["is_new_ticker"] is True


def test_second_buy_of_same_ticker_is_new_ticker_false():
    decisions = _newest_first(
        _decision("BUY NVDA → FILLED", 0),
        _decision("NO_DECISION", 1),
        _decision("NO_DECISION", 2),
        _decision("NO_DECISION", 3),
        _decision("NO_DECISION", 4),
        _decision("NO_DECISION", 5),
        _decision("BUY NVDA → FILLED", 6),
    )
    trades = _newest_first(
        _trade("BUY", "NVDA", 6),
        _trade("BUY", "NVDA", 0),
    )
    out = ftad.build_first_trade_after_drought(decisions, trades)
    assert out["post_drought_trades"][0]["is_new_ticker"] is False


# ─── flip detection ───────────────────────────────────────────────────

def test_buy_within_flip_window_after_sell_is_flip():
    # SELL NVDA at t=0, drought 1–5, BUY NVDA at t=6 minutes → flip=True
    # because the prior SELL is < 24h ago. The buy is NOT new (NVDA was
    # bought before — implied by the SELL, but we have an explicit BUY
    # earlier to be precise).
    decisions = _newest_first(
        _decision("BUY NVDA → FILLED", -10),
        _decision("SELL NVDA → FILLED", 0),
        _decision("NO_DECISION", 1),
        _decision("NO_DECISION", 2),
        _decision("NO_DECISION", 3),
        _decision("NO_DECISION", 4),
        _decision("NO_DECISION", 5),
        _decision("BUY NVDA → FILLED", 6),
    )
    trades = _newest_first(
        _trade("BUY", "NVDA", 6),
        _trade("SELL", "NVDA", 0),
        _trade("BUY", "NVDA", -10),
    )
    out = ftad.build_first_trade_after_drought(decisions, trades)
    rec = out["post_drought_trades"][0]
    assert rec["ticker"] == "NVDA"
    assert rec["is_flip"] is True
    assert rec["is_new_ticker"] is False


def test_buy_outside_flip_window_is_not_flip():
    # SELL 30h before the BUY → outside default _FLIP_WINDOW_HOURS=24.
    decisions = _newest_first(
        _decision("BUY NVDA → FILLED", -60 * 30),    # very old, before the SELL
        _decision("SELL NVDA → FILLED", -60 * 30),   # SELL 30h before drought end
        _decision("NO_DECISION", -60 * 25),
        _decision("NO_DECISION", -60 * 24),
        _decision("NO_DECISION", -60 * 23),
        _decision("NO_DECISION", -60 * 22),
        _decision("NO_DECISION", -60 * 21),
        _decision("BUY NVDA → FILLED", 0),  # 30h after the SELL
    )
    trades = _newest_first(
        _trade("BUY", "NVDA", 0),
        _trade("SELL", "NVDA", -60 * 30),
        _trade("BUY", "NVDA", -60 * 30),
    )
    out = ftad.build_first_trade_after_drought(decisions, trades)
    rec = out["post_drought_trades"][0]
    assert rec["is_flip"] is False


# ─── verdict ladder ───────────────────────────────────────────────────

def _build_panic_flip_ledger() -> tuple[list[dict], list[dict]]:
    """3 droughts each followed by a SELL→drought→BUY flip of the same
    ticker. Normal (non-post-drought) trades are spaced fresh-ticker buys
    so flip_rate among "other" trades is 0%."""
    decisions: list[dict] = []
    trades: list[dict] = []
    # Three (SELL X, drought, BUY X) flip patterns + a couple of normal buys
    # of new names to pad "other" rows.
    for i, t in enumerate(["NVDA", "AMD", "MU"]):
        base = i * 200  # minutes; spread events out
        decisions += [
            _decision("BUY %s → FILLED" % t, base + 0),
            _decision("SELL %s → FILLED" % t, base + 10),
            _decision("NO_DECISION", base + 11),
            _decision("NO_DECISION", base + 12),
            _decision("NO_DECISION", base + 13),
            _decision("NO_DECISION", base + 14),
            _decision("NO_DECISION", base + 15),
            _decision("BUY %s → FILLED" % t, base + 20),   # flip
        ]
        trades += [
            _trade("BUY", t, base + 0),
            _trade("SELL", t, base + 10),
            _trade("BUY", t, base + 20),  # this is the post-drought flip
        ]
    # Normal trades after all flips — non-post-drought, fresh tickers.
    normal_tickers = ["TSM", "META", "SPY"]
    for j, nt in enumerate(normal_tickers):
        decisions.append(_decision("BUY %s → FILLED" % nt, 1000 + j * 10))
        trades.append(_trade("BUY", nt, 1000 + j * 10))
    return list(reversed(decisions)), list(reversed(trades))


def test_panic_flip_verdict_when_post_drought_dominated_by_flips():
    decisions, trades = _build_panic_flip_ledger()
    out = ftad.build_first_trade_after_drought(decisions, trades)
    assert out["n_droughts"] == 3
    assert out["n_post_drought_trades"] == 3
    # All 3 post-drought trades are flips → 100%.
    assert out["post_drought_flip_rate"] == 100.0
    # Other trades: the 3 initial BUYs, 3 SELLs, 3 fresh-name BUYs = 9 rows;
    # none qualify as flips (an "other" BUY before a SELL can't be a flip).
    assert out["other_flip_rate"] == 0.0
    assert out["verdict"] == "PANIC_FLIP"


def test_drought_exploration_verdict_when_post_drought_is_mostly_new_tickers():
    # 3 droughts each followed by a *brand-new* ticker. Other trades are
    # all on a single repeated ticker, so other_new_rate is low.
    decisions: list[dict] = []
    trades: list[dict] = []
    # 6 NVDA buys at the start — establishes NVDA as not-new throughout.
    for k in range(6):
        decisions.append(_decision("BUY NVDA → FILLED", k))
        trades.append(_trade("BUY", "NVDA", k))
    # 3 droughts, each ending with a brand-new ticker.
    for i, t in enumerate(["AMD", "MU", "TSM"]):
        base = 100 + i * 100
        decisions += [
            _decision("NO_DECISION", base + 0),
            _decision("NO_DECISION", base + 1),
            _decision("NO_DECISION", base + 2),
            _decision("NO_DECISION", base + 3),
            _decision("NO_DECISION", base + 4),
            _decision("BUY %s → FILLED" % t, base + 10),
        ]
        trades.append(_trade("BUY", t, base + 10))
    out = ftad.build_first_trade_after_drought(
        list(reversed(decisions)), list(reversed(trades)))
    assert out["n_droughts"] == 3
    assert out["n_post_drought_trades"] == 3
    # All 3 post-drought trades are net-new tickers → 100%.
    assert out["post_drought_new_rate"] == 100.0
    # Other trades: 6 NVDA buys. Only the FIRST is new → 1/6 ≈ 16.67%.
    assert out["other_new_rate"] == round(100.0 * 1 / 6, 2)
    assert out["verdict"] == "DROUGHT_EXPLORATION"


def test_insufficient_history_when_n_post_drought_below_min():
    # 2 droughts → only 2 post-drought trades, below _MIN_POST_DROUGHT=3.
    decisions: list[dict] = []
    trades: list[dict] = []
    for i, t in enumerate(["AMD", "MU"]):
        base = i * 100
        decisions += [
            _decision("BUY NVDA → FILLED", base + 0),
            _decision("NO_DECISION", base + 1),
            _decision("NO_DECISION", base + 2),
            _decision("NO_DECISION", base + 3),
            _decision("NO_DECISION", base + 4),
            _decision("NO_DECISION", base + 5),
            _decision("BUY %s → FILLED" % t, base + 10),
        ]
        trades += [
            _trade("BUY", "NVDA", base + 0),
            _trade("BUY", t, base + 10),
        ]
    out = ftad.build_first_trade_after_drought(
        list(reversed(decisions)), list(reversed(trades)))
    assert out["n_droughts"] == 2
    assert out["n_post_drought_trades"] == 2
    assert out["verdict"] == "INSUFFICIENT_HISTORY"


# ─── shared-trade dedupe ──────────────────────────────────────────────

def test_back_to_back_droughts_share_first_trade_only_once():
    # Two droughts separated by a HOLD (not a real trade). The same first
    # FILLED trade follows BOTH, but we count it only once.
    decisions = _newest_first(
        _decision("BUY NVDA → FILLED", 0),
        _decision("NO_DECISION", 1),
        _decision("NO_DECISION", 2),
        _decision("NO_DECISION", 3),
        _decision("NO_DECISION", 4),
        _decision("NO_DECISION", 5),
        _decision("HOLD NVDA → HOLD", 6),     # ends drought 1
        _decision("NO_DECISION", 7),
        _decision("NO_DECISION", 8),
        _decision("NO_DECISION", 9),
        _decision("NO_DECISION", 10),
        _decision("NO_DECISION", 11),
        _decision("BUY AMD → FILLED", 12),    # first FILLED trade after BOTH droughts
    )
    trades = _newest_first(
        _trade("BUY", "AMD", 12),
        _trade("BUY", "NVDA", 0),
    )
    out = ftad.build_first_trade_after_drought(decisions, trades)
    assert out["n_droughts"] == 2
    assert out["n_post_drought_trades"] == 1  # not 2 — shared trade counted once


# ─── garbage input never raises ──────────────────────────────────────

def test_garbage_timestamps_dont_raise():
    decisions = [{"action_taken": "NO_DECISION", "timestamp": "not-a-date"}]
    trades = [{"action": "BUY", "ticker": "NVDA", "timestamp": None,
               "qty": 1, "price": 1, "value": 1}]
    out = ftad.build_first_trade_after_drought(decisions, trades)
    # Builder must not raise; with no parseable timestamps it degrades to
    # the no-pairings shape (either NO_DATA or n_droughts=0).
    assert out["state"] in ("NO_DATA", "OK")
    assert out["verdict"] in ("NO_DATA", "INSUFFICIENT_HISTORY")


# ─── headline / verdict_detail populated for every state ─────────────

def test_headline_and_detail_always_strings():
    out = ftad.build_first_trade_after_drought([], [])
    assert isinstance(out["headline"], str) and out["headline"]
    assert isinstance(out["verdict_detail"], str) and out["verdict_detail"]

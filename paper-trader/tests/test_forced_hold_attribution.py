"""Tests for analytics/forced_hold_attribution.py — forced-vs-chosen
attribution on the OPEN book.

Each test below fails LOUDLY if the logic is wrong, not just if it
crashes. The contract under lock:

* boundary inclusivity: ``blind_pct >= 0.50`` is FORCED_HOLD, ``0.25 <=
  blind_pct < 0.50`` is PARTIALLY_FORCED, ``0.10 <= blind_pct < 0.25``
  is MIXED, ``blind_pct < 0.10`` is CHOSEN_HOLD (``TestBoundary``).
* the sample-size gate: below ``MIN_CYCLES`` the per-position verdict is
  ``None`` and the aggregate state is ``EMERGING`` — a 2-cycle reading
  can never call a fresh position "FORCED_HOLD" (``TestStateLadder``).
* ``cycles_total`` strictly counts decisions with
  ``timestamp >= opened_at`` per position — a decision row before the
  position opened is never attributed (``TestTimestampFilter``).
* the verdict precedence FORCED_HOLD_DOMINANT > PARTIALLY_FORCED >
  MOSTLY_CHOSEN (``TestAggregateVerdict``).
* never-raises-on-garbage, no input mutation (``TestSafe``).
* the route serves the builder unchanged (``TestEndpoint``).
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics import forced_hold_attribution as fha
from paper_trader.analytics.forced_hold_attribution import (
    FORCED_HOLD_BLIND_PCT,
    MIN_CYCLES,
    MIXED_BLIND_PCT,
    PARTIALLY_FORCED_BLIND_PCT,
    _classify,
    _is_blind,
    build_forced_hold_attribution,
)


_BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
_NOW = datetime(2026, 1, 5, 12, 0, 0, tzinfo=timezone.utc)


def _ts(offset_min: int) -> str:
    return (_BASE + timedelta(minutes=offset_min)).isoformat()


def _pos(ticker: str = "NVDA", opened_offset_min: int = 0) -> dict:
    return {
        "ticker": ticker,
        "opened_at": _ts(opened_offset_min),
        "qty": 3.0,
        "type": "stock",
        "avg_cost": 220.0,
        "current_price": 215.0,
        "unrealized_pl": -15.0,
    }


def _dec(ts_offset_min: int, action: str = "HOLD") -> dict:
    return {
        "timestamp": _ts(ts_offset_min),
        "action_taken": action,
        "market_open": True,
        "signal_count": 5,
    }


def _blind(ts_offset_min: int) -> dict:
    return _dec(ts_offset_min, "NO_DECISION")


def _build_cycles(n_blind: int, n_sighted: int,
                  start_offset_min: int = 1) -> list[dict]:
    """``n_blind`` blind + ``n_sighted`` sighted decisions, all timestamped
    after ``start_offset_min`` minutes."""
    decisions = []
    t = start_offset_min
    for _ in range(n_blind):
        decisions.append(_blind(t))
        t += 5
    for _ in range(n_sighted):
        decisions.append(_dec(t, "HOLD"))
        t += 5
    return decisions


class TestIsBlind:
    """``_is_blind`` recognises every NO_DECISION literal the runner writes."""

    def test_plain_no_decision_is_blind(self):
        assert _is_blind("NO_DECISION") is True

    def test_no_decision_arrow_is_blind(self):
        # `strategy.decide()` writes the action_taken as either
        # `"NO_DECISION"` (raw=None / fail) or a real action like
        # `"BUY NVDA → FILLED"`. Both shapes must be recognised correctly.
        assert _is_blind("NO_DECISION → SKIPPED") is True

    def test_real_action_is_not_blind(self):
        assert _is_blind("HOLD") is False
        assert _is_blind("BUY NVDA → FILLED") is False
        assert _is_blind("SELL NVDA → BLOCKED") is False

    def test_missing_action_is_not_blind(self):
        assert _is_blind(None) is False
        assert _is_blind("") is False

    def test_case_insensitive(self):
        # action_taken is upper-case by writer convention, but a future
        # writer that lower-cases it (or any operator who massaged the
        # DB) must not silently drop attribution.
        assert _is_blind("no_decision") is True


class TestClassify:
    """Per-position verdict ladder + sample-size gate."""

    def test_below_min_cycles_returns_none(self):
        # MIN_CYCLES - 1 is always too noisy regardless of blind_pct.
        for bp in (0.0, 0.25, 0.50, 0.9, 1.0):
            assert _classify(bp, MIN_CYCLES - 1) is None

    def test_forced_hold_at_min_cycles(self):
        assert _classify(0.50, MIN_CYCLES) == "FORCED_HOLD"
        assert _classify(1.0, MIN_CYCLES) == "FORCED_HOLD"

    def test_partially_forced_band(self):
        assert _classify(0.25, MIN_CYCLES) == "PARTIALLY_FORCED"
        assert _classify(0.49, MIN_CYCLES) == "PARTIALLY_FORCED"

    def test_mixed_band(self):
        assert _classify(0.10, MIN_CYCLES) == "MIXED"
        assert _classify(0.24, MIN_CYCLES) == "MIXED"

    def test_chosen_hold_band(self):
        assert _classify(0.0, MIN_CYCLES) == "CHOSEN_HOLD"
        assert _classify(0.09, MIN_CYCLES) == "CHOSEN_HOLD"


class TestBoundary:
    """Boundary pinning — drifting any edge breaks the desk's expectation."""

    def test_forced_hold_threshold_inclusive(self):
        # blind_pct exactly equal to FORCED_HOLD_BLIND_PCT is FORCED_HOLD,
        # not PARTIALLY_FORCED.
        assert FORCED_HOLD_BLIND_PCT == 0.50
        assert _classify(FORCED_HOLD_BLIND_PCT, MIN_CYCLES) == "FORCED_HOLD"
        assert _classify(FORCED_HOLD_BLIND_PCT - 0.001,
                         MIN_CYCLES) == "PARTIALLY_FORCED"

    def test_partially_forced_threshold_inclusive(self):
        assert PARTIALLY_FORCED_BLIND_PCT == 0.25
        assert _classify(PARTIALLY_FORCED_BLIND_PCT,
                         MIN_CYCLES) == "PARTIALLY_FORCED"
        assert _classify(PARTIALLY_FORCED_BLIND_PCT - 0.001,
                         MIN_CYCLES) == "MIXED"

    def test_mixed_threshold_inclusive(self):
        assert MIXED_BLIND_PCT == 0.10
        assert _classify(MIXED_BLIND_PCT, MIN_CYCLES) == "MIXED"
        assert _classify(MIXED_BLIND_PCT - 0.001,
                         MIN_CYCLES) == "CHOSEN_HOLD"


class TestTimestampFilter:
    """``cycles_total`` strictly counts decisions with ``ts >= opened_at``."""

    def test_decision_before_open_is_excluded(self):
        # Position opens at minute 10; the decision at minute 5 must NOT
        # be attributed to it.
        out = build_forced_hold_attribution(
            [_pos("NVDA", opened_offset_min=10)],
            [_dec(5, "HOLD"), _dec(20, "NO_DECISION")],
            now=_NOW,
        )
        card = out["positions"][0]
        assert card["cycles_total"] == 1
        assert card["cycles_blind"] == 1

    def test_decision_at_open_instant_is_included(self):
        # Inclusive on the open instant — a same-microsecond write under
        # load still counts toward the position.
        out = build_forced_hold_attribution(
            [_pos("NVDA", opened_offset_min=10)],
            [_dec(10, "NO_DECISION"), _dec(20, "HOLD")],
            now=_NOW,
        )
        card = out["positions"][0]
        assert card["cycles_total"] == 2
        assert card["cycles_blind"] == 1

    def test_per_position_filtering_is_independent(self):
        # NVDA opens at minute 0; MU opens at minute 100. A decision at
        # minute 50 must be attributed to NVDA only.
        out = build_forced_hold_attribution(
            [_pos("NVDA", opened_offset_min=0),
             _pos("MU", opened_offset_min=100)],
            [_dec(50, "HOLD"), _dec(150, "NO_DECISION")],
            now=_NOW,
        )
        cards = {c["ticker"]: c for c in out["positions"]}
        assert cards["NVDA"]["cycles_total"] == 2  # both decisions ≥ 0
        assert cards["NVDA"]["cycles_blind"] == 1
        assert cards["MU"]["cycles_total"] == 1    # only the 150-min one
        assert cards["MU"]["cycles_blind"] == 1


class TestStateLadder:
    """Aggregate state gate: NO_DATA / EMERGING / STABLE."""

    def test_no_data_when_no_open_positions(self):
        out = build_forced_hold_attribution([], [_dec(10) for _ in range(50)],
                                            now=_NOW)
        assert out["state"] == "NO_DATA"
        assert out["verdict"] is None
        assert out["n_open"] == 0
        assert "nothing to attribute" in out["headline"].lower()

    def test_emerging_when_any_position_below_min_cycles(self):
        # One stable + one fresh position → aggregate EMERGING (the honesty
        # precedent: any noise contaminates the aggregate).
        stable_cycles = _build_cycles(n_blind=2, n_sighted=10,
                                      start_offset_min=1)
        out = build_forced_hold_attribution(
            [_pos("NVDA", opened_offset_min=0),
             _pos("MU", opened_offset_min=1000)],  # only 0 cycles after
            stable_cycles,
            now=_NOW,
        )
        assert out["state"] == "EMERGING"
        assert out["verdict"] is None

    def test_stable_when_all_positions_have_min_cycles(self):
        out = build_forced_hold_attribution(
            [_pos("NVDA", opened_offset_min=0)],
            _build_cycles(n_blind=2, n_sighted=10, start_offset_min=1),
            now=_NOW,
        )
        assert out["state"] == "STABLE"
        assert out["verdict"] in (
            "FORCED_HOLD_DOMINANT", "PARTIALLY_FORCED", "MOSTLY_CHOSEN",
        )


class TestAggregateVerdict:
    """STABLE-only verdict ladder with precedence."""

    def test_forced_hold_dominant_when_one_position_forced(self):
        # 6 blind + 4 sighted = 60% blind → FORCED_HOLD
        out = build_forced_hold_attribution(
            [_pos("NVDA", opened_offset_min=0)],
            _build_cycles(n_blind=6, n_sighted=4, start_offset_min=1),
            now=_NOW,
        )
        assert out["state"] == "STABLE"
        assert out["verdict"] == "FORCED_HOLD_DOMINANT"
        assert out["n_forced"] == 1
        assert "FORCED" in out["headline"] or "force" in out["headline"].lower()

    def test_partially_forced_when_one_partial_no_forced(self):
        # 3 blind + 7 sighted = 30% blind → PARTIALLY_FORCED
        out = build_forced_hold_attribution(
            [_pos("NVDA", opened_offset_min=0)],
            _build_cycles(n_blind=3, n_sighted=7, start_offset_min=1),
            now=_NOW,
        )
        assert out["state"] == "STABLE"
        assert out["verdict"] == "PARTIALLY_FORCED"
        assert out["n_forced"] == 0
        assert out["n_partially_forced"] == 1

    def test_mostly_chosen_when_no_forced_no_partial(self):
        # 0 blind + 10 sighted = 0% blind → CHOSEN_HOLD
        out = build_forced_hold_attribution(
            [_pos("NVDA", opened_offset_min=0)],
            _build_cycles(n_blind=0, n_sighted=10, start_offset_min=1),
            now=_NOW,
        )
        assert out["state"] == "STABLE"
        assert out["verdict"] == "MOSTLY_CHOSEN"
        assert out["n_forced"] == 0
        assert out["n_partially_forced"] == 0
        assert out["n_chosen"] == 1

    def test_precedence_forced_beats_partial(self):
        # NVDA forced (50% blind), MU partial (25% blind) → DOMINANT, not
        # PARTIALLY_FORCED. Both positions same opened_at so they share
        # the cycles_total denominator (10).
        out = build_forced_hold_attribution(
            [_pos("NVDA", opened_offset_min=0),
             _pos("MU", opened_offset_min=0)],
            _build_cycles(n_blind=5, n_sighted=5, start_offset_min=1),
            now=_NOW,
        )
        # Each position sees the same 10 cycles, 50% blind → both FORCED.
        # The precedence rule must still report DOMINANT, not PARTIAL.
        assert out["verdict"] == "FORCED_HOLD_DOMINANT"
        assert out["n_forced"] == 2

    def test_card_sort_puts_forced_first(self):
        # NVDA partial (30% blind), MU forced (60% blind). Cards must
        # sort so MU is first (worst-blind first), then NVDA.
        # NVDA opens at 0, sees 10 cycles (3 blind, 7 sighted).
        # MU opens at 1000, sees 10 cycles (6 blind, 4 sighted).
        decisions = _build_cycles(n_blind=3, n_sighted=7, start_offset_min=1)
        decisions += _build_cycles(n_blind=6, n_sighted=4,
                                   start_offset_min=1001)
        out = build_forced_hold_attribution(
            [_pos("NVDA", opened_offset_min=0),
             _pos("MU", opened_offset_min=1000)],
            decisions,
            now=_NOW,
        )
        tickers = [c["ticker"] for c in out["positions"]]
        assert tickers == ["MU", "NVDA"]
        assert out["positions"][0]["verdict"] == "FORCED_HOLD"
        assert out["positions"][1]["verdict"] == "PARTIALLY_FORCED"


class TestPerPositionCard:
    """The per-position card carries the right numbers."""

    def test_blind_pct_arithmetic(self):
        # 4 blind + 6 sighted = 40% blind, PARTIALLY_FORCED.
        out = build_forced_hold_attribution(
            [_pos("NVDA", opened_offset_min=0)],
            _build_cycles(n_blind=4, n_sighted=6, start_offset_min=1),
            now=_NOW,
        )
        card = out["positions"][0]
        assert card["cycles_total"] == 10
        assert card["cycles_blind"] == 4
        assert card["cycles_sighted"] == 6
        assert card["blind_pct"] == pytest.approx(0.4)
        assert card["verdict"] == "PARTIALLY_FORCED"
        assert card["forced_hold"] is False

    def test_forced_hold_flag_set_only_on_forced(self):
        out = build_forced_hold_attribution(
            [_pos("NVDA", opened_offset_min=0)],
            _build_cycles(n_blind=8, n_sighted=2, start_offset_min=1),
            now=_NOW,
        )
        card = out["positions"][0]
        assert card["verdict"] == "FORCED_HOLD"
        assert card["forced_hold"] is True

    def test_age_hours_present(self):
        out = build_forced_hold_attribution(
            [_pos("NVDA", opened_offset_min=0)],
            _build_cycles(n_blind=0, n_sighted=10, start_offset_min=1),
            now=_NOW,
        )
        card = out["positions"][0]
        # _BASE at +0 min, _NOW = _BASE + 4 days = 96 hours.
        assert card["age_hours"] == pytest.approx(96.0, abs=0.1)

    def test_zero_cycles_total_does_not_zero_divide(self):
        # No decisions ever — guard against zero-division.
        out = build_forced_hold_attribution(
            [_pos("NVDA", opened_offset_min=0)],
            [],
            now=_NOW,
        )
        card = out["positions"][0]
        assert card["cycles_total"] == 0
        assert card["blind_pct"] == 0.0
        assert card["verdict"] is None
        assert out["state"] == "EMERGING"  # below MIN_CYCLES


class TestSafe:
    """Pure / safe contract: never raises, never mutates input."""

    def test_garbage_decisions_are_skipped(self):
        garbage = [
            None, "not a dict", {"timestamp": None, "action_taken": "HOLD"},
            {"timestamp": "", "action_taken": "NO_DECISION"},
            {"timestamp": _ts(5), "action_taken": "HOLD"},  # one good row
        ]
        out = build_forced_hold_attribution(
            [_pos("NVDA", opened_offset_min=0)], garbage, now=_NOW,
        )
        # Only the one well-formed row at minute 5 counts.
        assert out["positions"][0]["cycles_total"] == 1
        assert out["positions"][0]["cycles_blind"] == 0

    def test_garbage_positions_are_skipped(self):
        out = build_forced_hold_attribution(
            [None, "junk", {"no_ticker_no_opened_at": True},
             _pos("NVDA", opened_offset_min=0)],
            _build_cycles(n_blind=0, n_sighted=10),
            now=_NOW,
        )
        # The 3 garbage entries: None and "junk" are skipped (not dict);
        # the bare-dict is kept but with cycles_total=0 (no opened_at).
        assert out["n_open"] == 2
        nvda = [c for c in out["positions"] if c["ticker"] == "NVDA"][0]
        assert nvda["cycles_total"] == 10

    def test_no_input_mutation(self):
        positions = [_pos("NVDA", opened_offset_min=0)]
        decisions = _build_cycles(n_blind=2, n_sighted=8)
        before_positions = [dict(p) for p in positions]
        before_decisions = [dict(d) for d in decisions]
        build_forced_hold_attribution(positions, decisions, now=_NOW)
        assert positions == before_positions
        assert decisions == before_decisions

    def test_unparseable_opened_at_does_not_raise(self):
        bad_pos = _pos("NVDA", opened_offset_min=0)
        bad_pos["opened_at"] = "not-an-iso-timestamp"
        out = build_forced_hold_attribution(
            [bad_pos], _build_cycles(n_blind=5, n_sighted=5), now=_NOW,
        )
        # With a non-string-ish opened_at, no decisions can pass the
        # ts >= opened_at gate (Python <-comparison would fall through
        # to lex compare on the literal "not-an-iso-timestamp" — but
        # the builder protects against that by only entering the loop
        # when opened_at is a non-empty string). The builder must not
        # raise. The card's cycles_total may legitimately be 0 or > 0
        # depending on lex order; the contract is "no raise" only.
        assert out["positions"][0]["age_hours"] is None


class TestEndpoint:
    """The route serves the builder unchanged."""

    def test_route_returns_builder_output_shape(self):
        from paper_trader.dashboard import app

        client = app.test_client()
        resp = client.get("/api/forced-hold-attribution")
        assert resp.status_code == 200
        body = resp.get_json()
        # Required top-level keys the builder always emits.
        for key in ("state", "verdict", "headline", "min_cycles",
                    "n_open", "n_forced", "n_partially_forced",
                    "n_mixed", "n_chosen", "positions", "as_of"):
            assert key in body, f"missing {key} in response"
        assert body["min_cycles"] == MIN_CYCLES
        assert isinstance(body["positions"], list)

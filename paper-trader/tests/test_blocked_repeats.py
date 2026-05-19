"""Unit tests for paper_trader.analytics.blocked_repeats + endpoint.

The builder is pure (no DB, no network); these tests pin the actual
classification mapping AND the operator-actionable contracts that matter
when the desk is trying to triage a stuck Opus loop.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from paper_trader.analytics.blocked_repeats import (
    MIN_REPEAT,
    _classify_cause,
    _parse_verb_ticker,
    build_blocked_repeats,
)


NOW = datetime(2026, 5, 19, 21, 0, 0, tzinfo=timezone.utc)


def _dec(action_taken: str, detail: str, ts: str) -> dict:
    """Build a ``store.recent_decisions()``-shaped row.

    ``reasoning`` mirrors what ``strategy.decide`` actually writes for a
    non-NO_DECISION row: a JSON dump of ``{decision, auto_exits, detail,
    fallback_used}``."""
    return {
        "timestamp": ts,
        "market_open": 1,
        "signal_count": 5,
        "action_taken": action_taken,
        "reasoning": json.dumps({
            "decision": {"action": action_taken.split()[0],
                         "ticker": action_taken.split()[1]
                         if len(action_taken.split()) > 1 else None},
            "auto_exits": [],
            "detail": detail,
            "fallback_used": False,
        }),
        "portfolio_value": 1000.0,
        "cash": 50.0,
    }


class TestClassifyCause:
    """The free-text → bucket mapping is the contract every endpoint
    consumer keys off — pin each documented blocked-detail string."""

    def test_insufficient_cash_classifies_as_cash(self):
        d = "insufficient cash (have $50.00, need $500.00)"
        assert _classify_cause(d) == "CASH"

    def test_no_price_classifies_as_data(self):
        assert _classify_cause("no price for NVDA") == "DATA"

    def test_no_option_price_classifies_as_data(self):
        d = "no option price for NVDA 2026-12-19 600.0 call"
        assert _classify_cause(d) == "DATA"

    def test_exceeds_held_classifies_as_sizing(self):
        d = "sell qty 100 exceeds held 10 for MU stock"
        assert _classify_cause(d) == "SIZING"

    def test_no_open_position_classifies_as_sizing(self):
        d = "no open call position in NVDA to close"
        assert _classify_cause(d) == "SIZING"

    def test_no_matching_open_classifies_as_sizing(self):
        d = "no matching open call for NVDA"
        assert _classify_cause(d) == "SIZING"

    def test_ambiguous_classifies_as_specification(self):
        d = "ambiguous call close for NVDA; specify strike+expiry"
        assert _classify_cause(d) == "SPECIFICATION"

    def test_missing_strike_expiry_classifies_as_specification(self):
        assert _classify_cause("option trade missing strike/expiry") \
            == "SPECIFICATION"

    def test_strike_not_numeric_classifies_as_specification(self):
        assert _classify_cause("strike not numeric: 'ATM'") \
            == "SPECIFICATION"

    def test_qty_not_numeric_classifies_as_specification(self):
        assert _classify_cause("qty not numeric: 'all'") \
            == "SPECIFICATION"

    def test_unknown_action_classifies_as_specification(self):
        assert _classify_cause("unknown action HEDGE") \
            == "SPECIFICATION"

    def test_empty_classifies_as_other(self):
        assert _classify_cause("") == "OTHER"
        assert _classify_cause(None) == "OTHER"

    def test_unknown_message_classifies_as_other(self):
        assert _classify_cause("something we never emit") == "OTHER"

    def test_case_insensitive(self):
        # Strategy might emit "Insufficient cash" with capital I.
        assert _classify_cause("Insufficient cash blah") == "CASH"


class TestParseVerbTicker:
    """Local parser mirrors dashboard._parse_action_ticker."""

    def test_buy_ticker(self):
        assert _parse_verb_ticker("BUY NVDA → BLOCKED") == ("BUY", "NVDA")

    def test_buy_call(self):
        assert _parse_verb_ticker("BUY_CALL NVDA → BLOCKED") \
            == ("BUY_CALL", "NVDA")

    def test_sell_put(self):
        assert _parse_verb_ticker("SELL_PUT MU → BLOCKED") \
            == ("SELL_PUT", "MU")

    def test_sentinel_blocked_alone(self):
        assert _parse_verb_ticker("BLOCKED") == (None, None)

    def test_sentinel_no_decision(self):
        assert _parse_verb_ticker("NO_DECISION") == (None, None)

    def test_empty(self):
        assert _parse_verb_ticker("") == (None, None)
        assert _parse_verb_ticker(None) == (None, None)

    def test_cash_pseudo_ticker_rejected(self):
        # dashboard._parse_action_ticker treats CASH/NONE as None.
        verb, tk = _parse_verb_ticker("BUY CASH → BLOCKED")
        assert tk is None

    def test_lowercase_normalised(self):
        assert _parse_verb_ticker("buy nvda → BLOCKED") == ("BUY", "NVDA")


class TestBuilderEmptyAndSingleBlocks:
    """The silence-when-nothing-actionable contract — a clean book or a
    book with only single (non-repeating) BLOCKs must not raise an alarm."""

    def test_no_decisions_returns_no_data(self):
        out = build_blocked_repeats([], now=NOW)
        assert out["state"] == "NO_DATA"
        assert out["verdict"] == "CLEAN"
        assert out["blocked_repeats"] == []
        assert out["n_blocked_total"] == 0

    def test_no_blocked_rows_returns_no_repeats(self):
        decs = [
            _dec("BUY NVDA → FILLED", "BUY 1 NVDA @ 600.00",
                 "2026-05-19T20:00:00+00:00"),
            _dec("HOLD MU → HOLD", "", "2026-05-19T19:30:00+00:00"),
        ]
        out = build_blocked_repeats(decs, now=NOW)
        assert out["state"] == "NO_REPEATS"
        assert out["n_blocked_total"] == 0
        assert out["blocked_repeats"] == []

    def test_single_block_below_min_repeat_returns_no_repeats(self):
        decs = [
            _dec("BUY NVDA → BLOCKED",
                 "insufficient cash (have $50.00, need $500.00)",
                 "2026-05-19T20:00:00+00:00"),
        ]
        out = build_blocked_repeats(decs, now=NOW)
        assert out["state"] == "NO_REPEATS"
        assert out["n_blocked_total"] == 1
        assert out["n_distinct_repeats"] == 0
        # Verdict stays CLEAN — one block is not a pattern.
        assert out["verdict"] == "CLEAN"


class TestBuilderRepeats:
    """The actionable case — Opus is being repeatedly blocked, the dashboard
    must surface the dominant cause and worst offender."""

    def test_two_cash_blocks_on_same_ticker_surfaces_as_repeating(self):
        decs = [
            _dec("BUY NVDA → BLOCKED",
                 "insufficient cash (have $50.00, need $500.00)",
                 "2026-05-19T20:00:00+00:00"),
            _dec("BUY NVDA → BLOCKED",
                 "insufficient cash (have $50.00, need $480.00)",
                 "2026-05-19T18:00:00+00:00"),
        ]
        out = build_blocked_repeats(decs, now=NOW)
        assert out["state"] == "OK"
        assert out["verdict"] == "REPEATING"
        assert out["n_blocked_total"] == 2
        assert out["n_distinct_repeats"] == 1
        assert len(out["blocked_repeats"]) == 1
        row = out["blocked_repeats"][0]
        assert row["verb"] == "BUY"
        assert row["ticker"] == "NVDA"
        assert row["count"] == 2
        assert row["dominant_cause"] == "CASH"
        assert row["latest_ts"] == "2026-05-19T20:00:00+00:00"

    def test_headline_names_worst_offender_and_count(self):
        decs = [
            _dec("BUY NVDA → BLOCKED", "insufficient cash", f"2026-05-19T{20-i:02d}:00:00+00:00")
            for i in range(5)
        ]
        out = build_blocked_repeats(decs, now=NOW)
        # Format: "BUY NVDA blocked 5x (CASH); 1 distinct repeat."
        assert "BUY NVDA" in out["headline"]
        assert "5x" in out["headline"]
        assert "CASH" in out["headline"]
        assert "1 distinct repeat" in out["headline"]

    def test_dominant_cause_picks_majority_when_mixed(self):
        # 3x CASH + 1x DATA on same key → dominant_cause = CASH.
        decs = [
            _dec("BUY NVDA → BLOCKED", "insufficient cash", "2026-05-19T20:00:00+00:00"),
            _dec("BUY NVDA → BLOCKED", "insufficient cash", "2026-05-19T19:00:00+00:00"),
            _dec("BUY NVDA → BLOCKED", "insufficient cash", "2026-05-19T18:00:00+00:00"),
            _dec("BUY NVDA → BLOCKED", "no price for NVDA",  "2026-05-19T17:00:00+00:00"),
        ]
        out = build_blocked_repeats(decs, now=NOW)
        row = out["blocked_repeats"][0]
        assert row["count"] == 4
        assert row["dominant_cause"] == "CASH"
        assert row["by_cause"] == {"CASH": 3, "DATA": 1}

    def test_two_distinct_repeats_sort_by_count_then_recency(self):
        # NVDA has 3 blocks (newer), MU has 5 blocks (older). Sort by count DESC.
        nvda = [_dec("BUY NVDA → BLOCKED", "insufficient cash",
                     f"2026-05-19T{20-i:02d}:00:00+00:00")
                for i in range(3)]
        mu = [_dec("BUY MU → BLOCKED", "insufficient cash",
                   f"2026-05-19T{10-i:02d}:00:00+00:00")
              for i in range(5)]
        out = build_blocked_repeats(nvda + mu, now=NOW)
        assert out["n_distinct_repeats"] == 2
        # MU has higher count → first.
        assert out["blocked_repeats"][0]["ticker"] == "MU"
        assert out["blocked_repeats"][0]["count"] == 5
        assert out["blocked_repeats"][1]["ticker"] == "NVDA"
        assert out["blocked_repeats"][1]["count"] == 3

    def test_tie_count_sorts_by_latest_ts_desc(self):
        # Equal counts → newest latest_ts wins.
        nvda = [_dec("BUY NVDA → BLOCKED", "x",
                     f"2026-05-19T{20-i:02d}:00:00+00:00")
                for i in range(2)]
        mu = [_dec("BUY MU → BLOCKED", "x",
                   f"2026-05-19T{14-i:02d}:00:00+00:00")
              for i in range(2)]
        out = build_blocked_repeats(nvda + mu, now=NOW)
        assert [r["ticker"] for r in out["blocked_repeats"]] == ["NVDA", "MU"]

    def test_different_verbs_on_same_ticker_are_distinct(self):
        # BUY NVDA and BUY_CALL NVDA are different (verb, ticker) pairs.
        decs = [
            _dec("BUY NVDA → BLOCKED", "insufficient cash",
                 "2026-05-19T20:00:00+00:00"),
            _dec("BUY NVDA → BLOCKED", "insufficient cash",
                 "2026-05-19T19:00:00+00:00"),
            _dec("BUY_CALL NVDA → BLOCKED", "option trade missing strike/expiry",
                 "2026-05-19T18:00:00+00:00"),
            _dec("BUY_CALL NVDA → BLOCKED", "option trade missing strike/expiry",
                 "2026-05-19T17:00:00+00:00"),
        ]
        out = build_blocked_repeats(decs, now=NOW)
        assert out["n_distinct_repeats"] == 2
        verbs = {(r["verb"], r["ticker"]) for r in out["blocked_repeats"]}
        assert verbs == {("BUY", "NVDA"), ("BUY_CALL", "NVDA")}

    def test_latest_age_hours_is_computed(self):
        decs = [
            _dec("BUY NVDA → BLOCKED", "insufficient cash",
                 "2026-05-19T20:00:00+00:00"),
            _dec("BUY NVDA → BLOCKED", "insufficient cash",
                 "2026-05-19T18:00:00+00:00"),
        ]
        out = build_blocked_repeats(decs, now=NOW)
        # NOW is 21:00 → 1.0h ago.
        assert out["blocked_repeats"][0]["latest_age_hours"] == pytest.approx(1.0)

    def test_filled_rows_are_ignored_even_if_action_contains_blocked_word(self):
        # The detection is "BLOCKED" anywhere in action_taken. A FILLED row
        # like "BUY NVDA → FILLED" must NOT count.
        decs = [
            _dec("BUY NVDA → FILLED", "ok", "2026-05-19T20:00:00+00:00"),
            _dec("BUY NVDA → FILLED", "ok", "2026-05-19T19:00:00+00:00"),
        ]
        out = build_blocked_repeats(decs, now=NOW)
        assert out["n_blocked_total"] == 0

    def test_non_dict_rows_skipped(self):
        decs = [None, "garbage", 42,
                _dec("BUY NVDA → BLOCKED", "insufficient cash",
                     "2026-05-19T20:00:00+00:00"),
                _dec("BUY NVDA → BLOCKED", "insufficient cash",
                     "2026-05-19T19:00:00+00:00")]
        out = build_blocked_repeats(decs, now=NOW)
        # The two valid rows still aggregate into one repeat.
        assert out["n_distinct_repeats"] == 1
        assert out["blocked_repeats"][0]["count"] == 2

    def test_malformed_reasoning_does_not_crash(self):
        # reasoning isn't JSON — detail extraction returns "", cause = OTHER.
        decs = [
            {"timestamp": "2026-05-19T20:00:00+00:00",
             "action_taken": "BUY NVDA → BLOCKED",
             "reasoning": "not valid json {{"},
            {"timestamp": "2026-05-19T19:00:00+00:00",
             "action_taken": "BUY NVDA → BLOCKED",
             "reasoning": None},
        ]
        out = build_blocked_repeats(decs, now=NOW)
        assert out["n_distinct_repeats"] == 1
        row = out["blocked_repeats"][0]
        assert row["dominant_cause"] == "OTHER"
        assert row["count"] == 2

    def test_min_repeat_override(self):
        # With min_repeat=3, a 2x block does NOT surface.
        decs = [
            _dec("BUY NVDA → BLOCKED", "insufficient cash",
                 "2026-05-19T20:00:00+00:00"),
            _dec("BUY NVDA → BLOCKED", "insufficient cash",
                 "2026-05-19T19:00:00+00:00"),
        ]
        out = build_blocked_repeats(decs, now=NOW, min_repeat=3)
        assert out["n_distinct_repeats"] == 0
        assert out["state"] == "NO_REPEATS"

    def test_min_repeat_default_is_2(self):
        assert MIN_REPEAT == 2


class TestEndpoint:
    """Light Flask wiring check — the endpoint must serialize the builder
    output without 500, with query-param clamping."""

    def _setup_app(self, monkeypatch):
        """Wire the Flask app to a pure in-memory stub Store.

        CRITICAL: do NOT call ``store._connect()`` here — that opens the
        LIVE ``data/paper_trader.db`` (DB_PATH is read at module level,
        not from any monkeypatched fixture) and any DDL/DML against it
        WIPES production decisions. The endpoint only needs
        ``recent_decisions`` so a pure list stub is correct and safe.
        """
        from paper_trader import dashboard as d

        class _StubStore:
            def __init__(self):
                self._decisions: list[dict] = []

            def recent_decisions(self, limit: int = 20):
                return list(self._decisions[:limit])

            def seed(self, decs):
                self._decisions = list(decs)

        store = _StubStore()
        monkeypatch.setattr(d, "get_store", lambda: store)
        return d.app, store

    def test_endpoint_returns_clean_on_empty(self, monkeypatch):
        app, _ = self._setup_app(monkeypatch)
        with app.test_client() as c:
            r = c.get("/api/blocked-repeats")
            assert r.status_code == 200
            j = r.get_json()
            assert j["state"] == "NO_DATA"
            assert j["verdict"] == "CLEAN"

    def test_endpoint_surfaces_repeats(self, monkeypatch):
        app, store = self._setup_app(monkeypatch)
        store.seed([
            _dec("BUY NVDA → BLOCKED", "insufficient cash",
                 "2026-05-19T20:00:00+00:00"),
            _dec("BUY NVDA → BLOCKED", "insufficient cash",
                 "2026-05-19T19:00:00+00:00"),
            _dec("BUY NVDA → BLOCKED", "insufficient cash",
                 "2026-05-19T18:00:00+00:00"),
        ])
        with app.test_client() as c:
            r = c.get("/api/blocked-repeats")
            assert r.status_code == 200
            j = r.get_json()
            assert j["verdict"] == "REPEATING"
            assert j["blocked_repeats"][0]["ticker"] == "NVDA"
            assert j["blocked_repeats"][0]["count"] == 3
            assert j["blocked_repeats"][0]["dominant_cause"] == "CASH"

    def test_endpoint_clamps_garbage_params(self, monkeypatch):
        app, _ = self._setup_app(monkeypatch)
        with app.test_client() as c:
            # Garbage params must not 500.
            r = c.get("/api/blocked-repeats?limit=banana&min_repeat=potato")
            assert r.status_code == 200
            assert r.get_json()["state"] in ("NO_DATA", "NO_REPEATS")

    def test_endpoint_min_repeat_override(self, monkeypatch):
        app, store = self._setup_app(monkeypatch)
        store.seed([
            _dec("BUY NVDA → BLOCKED", "insufficient cash",
                 "2026-05-19T20:00:00+00:00"),
            _dec("BUY NVDA → BLOCKED", "insufficient cash",
                 "2026-05-19T19:00:00+00:00"),
        ])
        with app.test_client() as c:
            # min_repeat=3 → these two don't qualify.
            r = c.get("/api/blocked-repeats?min_repeat=3")
            assert r.status_code == 200
            j = r.get_json()
            assert j["state"] == "NO_REPEATS"
            # min_repeat=2 → they do.
            r = c.get("/api/blocked-repeats?min_repeat=2")
            assert r.get_json()["verdict"] == "REPEATING"

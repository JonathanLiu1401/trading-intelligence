"""Tests for paper_trader.analytics.decision_vapor_skill.

Pins:
* the SPECIFIC × MIXED × VAPOR_DECISIONS × NO_DATA verdict ladder
* per-text signal detection (has_numeric / has_catalyst / has_ticker)
* watchlist-aware ticker matching (1-letter tickers when watchlist
  explicitly contains them; otherwise filtered out)
* envelope key stability across every verdict
* defensive: malformed decisions, missing reasoning, garbage JSON
  envelopes, parse_failed prefixes all degrade — never raise
* Flask route smoke
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.decision_vapor_skill import (
    DEFAULT_SPECIFIC_PCT_FLOOR,
    DEFAULT_VAPOR_PCT_CEIL,
    DEFAULT_VAPOR_PCT_FLOOR,
    DEFAULT_WINDOW_HOURS,
    MIN_FILLED_FOR_VERDICT,
    build_decision_vapor_skill,
    classify_specificity,
    detect_signals,
)


def _now():
    return datetime(2026, 5, 21, 18, 0, 0, tzinfo=timezone.utc)


def _dec(action_taken, reasoning_text, hours_ago, *, now=None, did=1):
    """Build a decision row in the canonical Opus-envelope format."""
    now = now or _now()
    ts = now - timedelta(hours=hours_ago)
    env = {
        "decision": {
            "action": "BUY",
            "ticker": "NVDA",
            "qty": 1,
            "confidence": 0.75,
            "reasoning": reasoning_text,
        },
        "auto_exits": [],
    }
    return {
        "id": did,
        "action_taken": action_taken,
        "reasoning": json.dumps(env),
        "timestamp": ts.isoformat(),
        "cash": 500.0,
        "portfolio_value": 1000.0,
        "market_open": 1,
        "signal_count": 10,
    }


_ENVELOPE_KEYS = {
    "verdict", "headline", "as_of", "window_hours",
    "stats", "thresholds", "samples",
}


class TestSignalDetection:
    def test_has_numeric_basic(self):
        s = detect_signals("Earnings beat by 5.5%, $58.3B net income")
        assert s["has_numeric"] is True

    def test_has_numeric_year_counts(self):
        # Year-only is grounding enough to qualify
        s = detect_signals("The 2025 earnings season started strong")
        assert s["has_numeric"] is True

    def test_no_numeric_in_pure_prose(self):
        s = detect_signals("Strong setup, building position, conviction high")
        assert s["has_numeric"] is False

    def test_has_catalyst_basic(self):
        assert detect_signals("Q1 earnings beat consensus")["has_catalyst"] is True
        assert detect_signals("FDA approval pending")["has_catalyst"] is True
        assert detect_signals("Fed rate hike next week")["has_catalyst"] is True
        assert detect_signals("Big merger announcement")["has_catalyst"] is True
        assert detect_signals("$80B buyback authorization")["has_catalyst"] is True

    def test_no_catalyst_in_generic_prose(self):
        s = detect_signals("Strong setup, building position")
        assert s["has_catalyst"] is False

    def test_has_ticker_watchlist(self):
        s = detect_signals("NVDA looks strong here", watchlist={"NVDA", "AAPL"})
        assert s["has_ticker"] is True
        # Cashtag also works
        s2 = detect_signals("Buying $NVDA on the dip", watchlist={"NVDA"})
        assert s2["has_ticker"] is True

    def test_has_ticker_without_watchlist(self):
        # Regex fallback picks up 2-5 cap-letter tokens
        assert detect_signals("NVDA looks strong here")["has_ticker"] is True
        # 1-letter caps are not matched without an explicit watchlist
        assert detect_signals("A B C D")["has_ticker"] is False

    def test_ticker_blacklist_filters_common_caps(self):
        # "BUY", "ML", "AI", "JSON" should not register as tickers
        # against the regex fallback
        s = detect_signals("BUY signal from ML model with JSON output")
        assert s["has_ticker"] is False

    def test_empty_text(self):
        s = detect_signals("")
        assert s == {"has_numeric": False, "has_catalyst": False, "has_ticker": False}

    def test_non_string_text(self):
        # Defensive: non-string input degrades to all-False
        s = detect_signals(None)  # type: ignore
        assert s == {"has_numeric": False, "has_catalyst": False, "has_ticker": False}


class TestClassifySpecificity:
    def test_all_three_is_specific(self):
        assert classify_specificity({
            "has_numeric": True, "has_catalyst": True, "has_ticker": True,
        }) == "SPECIFIC"

    def test_two_is_semi(self):
        assert classify_specificity({
            "has_numeric": True, "has_catalyst": True, "has_ticker": False,
        }) == "SEMI"
        assert classify_specificity({
            "has_numeric": True, "has_catalyst": False, "has_ticker": True,
        }) == "SEMI"
        assert classify_specificity({
            "has_numeric": False, "has_catalyst": True, "has_ticker": True,
        }) == "SEMI"

    def test_one_or_zero_is_vapor(self):
        assert classify_specificity({
            "has_numeric": True, "has_catalyst": False, "has_ticker": False,
        }) == "VAPOR"
        assert classify_specificity({
            "has_numeric": False, "has_catalyst": False, "has_ticker": False,
        }) == "VAPOR"


class TestEnvelopeStability:
    def test_no_data_empty(self):
        out = build_decision_vapor_skill(None, now=_now())
        assert set(out.keys()) >= _ENVELOPE_KEYS
        assert out["verdict"] == "NO_DATA"
        assert out["stats"]["n_filled"] == 0

    def test_no_data_few_filled(self):
        decisions = [
            _dec("BUY NVDA → FILLED", "Earnings beat NVDA $58B", 2.0, did=i)
            for i in range(3)  # below MIN_FILLED_FOR_VERDICT (5)
        ]
        out = build_decision_vapor_skill(decisions, now=_now())
        assert out["verdict"] == "NO_DATA"


class TestVerdictLadder:
    def test_specific_when_all_filled_are_grounded(self):
        decisions = [
            _dec(
                "BUY NVDA → FILLED",
                "NVDA earnings beat: $58.3B net income +211%, $80B buyback",
                2.0 + i, did=i,
            )
            for i in range(6)
        ]
        out = build_decision_vapor_skill(
            decisions, watchlist={"NVDA"}, now=_now(),
        )
        assert out["verdict"] == "SPECIFIC"
        assert out["stats"]["specific_pct"] == 100.0
        assert out["stats"]["vapor_pct"] == 0.0

    def test_vapor_when_most_filled_are_unanchored(self):
        # 5 filled, 4 vapor / 1 specific → 80% vapor → VAPOR_DECISIONS
        decisions = [
            _dec("BUY NVDA → FILLED", "Strong setup, building position", 2.0, did=1),
            _dec("BUY NVDA → FILLED", "High conviction, scaling in", 3.0, did=2),
            _dec("SELL NVDA → FILLED", "Trimming risk, defensive", 4.0, did=3),
            _dec("BUY NVDA → FILLED", "Looks good here", 5.0, did=4),
            _dec("BUY NVDA → FILLED", "Q1 earnings beat NVDA $58B", 6.0, did=5),
        ]
        out = build_decision_vapor_skill(
            decisions, watchlist={"NVDA"}, now=_now(),
        )
        assert out["verdict"] == "VAPOR_DECISIONS"
        assert out["stats"]["vapor_pct"] >= DEFAULT_VAPOR_PCT_FLOOR

    def test_mixed(self):
        # 5 filled: 2 specific, 3 semi → not enough to be SPECIFIC,
        # not enough vapor to be VAPOR_DECISIONS → MIXED
        decisions = [
            _dec("BUY NVDA → FILLED", "NVDA earnings beat $58B", 2.0, did=1),
            _dec("BUY NVDA → FILLED", "NVDA earnings beat $58B", 3.0, did=2),
            # Semi (numeric + catalyst, no ticker because watchlist=NVDA only)
            _dec("BUY NVDA → FILLED", "earnings beat at 5.5%", 4.0, did=3),
            _dec("BUY NVDA → FILLED", "earnings beat at 5.5%", 5.0, did=4),
            _dec("BUY NVDA → FILLED", "earnings beat at 5.5%", 6.0, did=5),
        ]
        out = build_decision_vapor_skill(
            decisions, watchlist={"NVDA"}, now=_now(),
        )
        assert out["verdict"] == "MIXED"


class TestFilledFiltering:
    def test_only_filled_counted(self):
        # 3 FILLED + 3 NO_DECISION + 3 BLOCKED — only FILLED contribute
        decisions = []
        for i in range(3):
            decisions.append(
                _dec("BUY NVDA → FILLED", "NVDA beat $58B", 2.0 + i, did=i),
            )
        # Add 6 non-FILLED (with same specific reasoning) to push below floor
        for i in range(6):
            d = _dec("NO_DECISION", "NVDA beat $58B", 3.0 + i, did=100 + i)
            d["action_taken"] = "NO_DECISION"
            decisions.append(d)
        out = build_decision_vapor_skill(
            decisions, watchlist={"NVDA"}, now=_now(),
        )
        assert out["stats"]["n_filled"] == 3
        assert out["verdict"] == "NO_DATA"  # below MIN_FILLED_FOR_VERDICT


class TestWindowEnforcement:
    def test_old_decisions_excluded(self):
        # 5 in-window FILLED specific + 10 old vapor
        decisions = []
        for i in range(5):
            decisions.append(
                _dec("BUY NVDA → FILLED", "NVDA beat $58B", 2.0 + i, did=i),
            )
        for i in range(10):
            # 200h ago is well outside default 168h window
            decisions.append(
                _dec("BUY NVDA → FILLED", "Looks good", 200.0 + i, did=100 + i),
            )
        out = build_decision_vapor_skill(
            decisions, watchlist={"NVDA"}, now=_now(),
        )
        assert out["stats"]["n_filled"] == 5
        assert out["verdict"] == "SPECIFIC"


class TestDefensiveDegradation:
    def test_malformed_decisions_never_raise(self):
        # 5 garbage rows. None should crash. The third row
        # ({reasoning: 12345, action_taken="BUY → FILLED"}) and the
        # first ({action_taken: "BUY → FILLED"} with no reasoning) DO
        # have a parseable FILLED action_taken plus a valid timestamp,
        # so they're counted as FILLED-but-vapor rows. The point is
        # defensive degradation, not zero counting — the builder
        # should never raise and the envelope must stay stable.
        garbage = [
            None,
            {},
            {"action_taken": "BUY → FILLED"},                # no reasoning, no ts → excluded by ts
            {"action_taken": "BUY → FILLED",                  # bad reasoning type
             "reasoning": 12345, "timestamp": _now().isoformat()},
            {"action_taken": None, "reasoning": "x",          # bad action
             "timestamp": _now().isoformat()},
            {"action_taken": "BUY → FILLED", "reasoning": "x",  # bad ts
             "timestamp": "not-iso"},
            "not a dict",
        ]
        out = build_decision_vapor_skill(garbage, now=_now())
        # Envelope is stable
        assert set(out.keys()) >= _ENVELOPE_KEYS
        # At most 1 row makes it through (the one with FILLED action +
        # valid ts but garbage reasoning) → vapor. Below MIN floor → NO_DATA.
        assert out["verdict"] == "NO_DATA"
        assert out["stats"]["n_filled"] <= 1

    def test_parse_failed_raw_text_handled(self):
        # When reasoning is the raw parse_failed: prefix (not JSON), we
        # still scan the body for signals.
        d = {
            "id": 1,
            "action_taken": "BUY NVDA → FILLED",
            "reasoning": "parse_failed: NVDA earnings beat $58B",
            "timestamp": _now().isoformat(),
        }
        # Add 4 more so we clear the floor
        decisions = [d]
        for i in range(4):
            decisions.append(
                _dec("BUY NVDA → FILLED", "NVDA beat $58B", 2.0 + i, did=100 + i),
            )
        out = build_decision_vapor_skill(
            decisions, watchlist={"NVDA"}, now=_now(),
        )
        # All 5 should be SPECIFIC
        assert out["stats"]["n_filled"] == 5
        assert out["stats"]["n_specific"] == 5


class TestSampleCap:
    def test_sample_limit_enforced(self):
        decisions = [
            _dec("BUY NVDA → FILLED", "NVDA beat $58B", 2.0 + i, did=i)
            for i in range(50)
        ]
        out = build_decision_vapor_skill(
            decisions, watchlist={"NVDA"}, now=_now(), sample_limit=10,
        )
        assert len(out["samples"]) == 10


class TestThresholdOverride:
    def test_lowering_vapor_floor_promotes_to_vapor(self):
        # 5 FILLED at 60% specific / 40% vapor.
        # Default: vapor_pct_floor=35 → VAPOR_DECISIONS (40>=35)
        # Override vapor_pct_floor=50 → MIXED (40<50)
        decisions = []
        for i in range(3):
            decisions.append(
                _dec("BUY NVDA → FILLED", "NVDA beat $58B", 2.0 + i, did=i),
            )
        for i in range(2):
            decisions.append(
                _dec("BUY NVDA → FILLED", "Looks good", 5.0 + i, did=100 + i),
            )
        out_default = build_decision_vapor_skill(
            decisions, watchlist={"NVDA"}, now=_now(),
        )
        # 40% vapor → triggers default VAPOR_DECISIONS
        assert out_default["verdict"] == "VAPOR_DECISIONS"
        out_lax = build_decision_vapor_skill(
            decisions, watchlist={"NVDA"}, now=_now(),
            vapor_pct_floor=50.0,
        )
        # 40% vapor < 50% floor → MIXED
        assert out_lax["verdict"] == "MIXED"


class TestFlaskRoute:
    def test_route_returns_envelope(self):
        from paper_trader.dashboard import app
        client = app.test_client()
        resp = client.get("/api/decision-vapor-skill")
        assert resp.status_code in (200, 500), resp.status_code
        body = resp.get_json()
        assert isinstance(body, dict)
        for k in ("verdict", "headline", "stats", "thresholds", "samples"):
            assert k in body, f"missing key: {k}"


class TestDefaults:
    def test_default_relationships(self):
        # The verdict ladder requires vapor_pct_ceil < vapor_pct_floor
        assert DEFAULT_VAPOR_PCT_CEIL < DEFAULT_VAPOR_PCT_FLOOR
        assert 0 < DEFAULT_SPECIFIC_PCT_FLOOR <= 100
        assert DEFAULT_WINDOW_HOURS > 0
        assert MIN_FILLED_FOR_VERDICT >= 2

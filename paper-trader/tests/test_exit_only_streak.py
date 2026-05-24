"""Tests for paper_trader.analytics.exit_only_streak.

The analyzer counts consecutive SELL/SELL_CALL/SELL_PUT fills at the END
of the trade ledger (newest-backward) and emits a verdict that surfaces
the structural "engine is liquidating, not running the strategy" pattern.

Behaviour locked here so a future refactor cannot silently:
  * mis-classify a BUY_CALL as an exit (or vice-versa);
  * count the wrong direction at the head of the ledger;
  * skip non-BUY/SELL actions when computing the run;
  * surface a verdict on a tiny sample (1-2 exits is normal turnover);
  * mis-render a future-dated trade timestamp as a negative hour age;
  * raise on a malformed / empty / unknown-action ledger.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from paper_trader.analytics.exit_only_streak import (
    DEFENSIVE_LIQUIDATION_MIN,
    DEFENSIVE_TRIM_MIN,
    ENTRY_ACTIONS,
    EXIT_ACTIONS,
    RECENT_SEQUENCE_LEN,
    _direction,
    _hours_since,
    build_exit_only_streak,
)


# ── helpers ─────────────────────────────────────────────────────────────


def _trade(action: str, ticker: str = "NVDA",
           ts: datetime | None = None) -> dict:
    """Build the minimum trade shape build_exit_only_streak reads."""
    if ts is None:
        ts = datetime(2026, 5, 23, 12, 0, 0, tzinfo=timezone.utc)
    return {
        "action": action,
        "ticker": ticker,
        "timestamp": ts.isoformat(),
        "qty": 1.0,
        "price": 100.0,
        "value": 100.0,
    }


# ── _direction ──────────────────────────────────────────────────────────


class TestDirection:

    @pytest.mark.parametrize("a", list(ENTRY_ACTIONS))
    def test_entry_actions_classified(self, a):
        assert _direction(a) == "ENTRY"

    @pytest.mark.parametrize("a", list(EXIT_ACTIONS))
    def test_exit_actions_classified(self, a):
        assert _direction(a) == "EXIT"

    @pytest.mark.parametrize("a", ["REBALANCE", "HOLD", "NO_DECISION",
                                    "BUY_FOO", "", "OPEN"])
    def test_unknown_returns_none(self, a):
        assert _direction(a) is None

    @pytest.mark.parametrize("a", [None, 42, [], {}])
    def test_non_string_returns_none(self, a):
        assert _direction(a) is None

    def test_case_insensitive(self):
        assert _direction("buy") == "ENTRY"
        assert _direction("sell_CALL") == "EXIT"

    def test_whitespace_trimmed(self):
        assert _direction("  BUY  ") == "ENTRY"


# ── _hours_since ────────────────────────────────────────────────────────


class TestHoursSince:

    def _now(self):
        return datetime(2026, 5, 23, 12, 0, 0, tzinfo=timezone.utc)

    def test_one_hour_ago(self):
        ts = (self._now() - timedelta(hours=1)).isoformat()
        assert _hours_since(ts, self._now()) == 1.0

    def test_naive_treated_as_utc(self):
        ts = (self._now() - timedelta(hours=2)).replace(tzinfo=None).isoformat()
        assert _hours_since(ts, self._now()) == 2.0

    def test_future_clamps_to_zero(self):
        ts = (self._now() + timedelta(hours=3)).isoformat()
        # A clock step-back must NOT render negative.
        assert _hours_since(ts, self._now()) == 0.0

    def test_z_suffix_parsed(self):
        ts = (self._now() - timedelta(hours=0.5)).isoformat().replace("+00:00", "Z")
        assert _hours_since(ts, self._now()) == 0.5

    @pytest.mark.parametrize("bad", [None, "", "not a date", "2026-99-99"])
    def test_unparseable_returns_none(self, bad):
        assert _hours_since(bad, self._now()) is None


# ── build_exit_only_streak — NO_DATA ────────────────────────────────────


class TestNoData:

    def test_empty_ledger(self):
        out = build_exit_only_streak([])
        assert out["state"] == "NO_DATA"
        assert out["verdict"] is None
        assert out["exit_run_length"] == 0
        assert out["n_total_fills"] == 0
        assert "no exit run" in out["headline"].lower()

    def test_only_unknown_actions(self):
        out = build_exit_only_streak([
            _trade("REBALANCE"),
            _trade("HOLD"),
            _trade("NO_DECISION"),
        ])
        # Unknown rows skipped → effectively empty.
        assert out["state"] == "NO_DATA"
        assert out["verdict"] is None
        assert out["n_total_fills"] == 0

    def test_none_input(self):
        out = build_exit_only_streak(None)
        assert out["state"] == "NO_DATA"


# ── build_exit_only_streak — MOST_RECENT_IS_ENTRY ───────────────────────


class TestMostRecentIsEntry:

    def test_single_entry(self):
        out = build_exit_only_streak([_trade("BUY", "NVDA")])
        assert out["verdict"] == "MOST_RECENT_IS_ENTRY"
        assert out["exit_run_length"] == 0
        assert out["last_entry_action"] == "BUY"
        assert out["last_entry_ticker"] == "NVDA"

    def test_mixed_ending_in_entry(self):
        out = build_exit_only_streak([
            _trade("SELL", "AMD"),
            _trade("SELL", "MU"),
            _trade("BUY", "NVDA"),
        ])
        # Most recent fill is a BUY → no exit run even though exits precede.
        assert out["verdict"] == "MOST_RECENT_IS_ENTRY"
        assert out["exit_run_length"] == 0
        assert out["last_entry_ticker"] == "NVDA"

    def test_one_trailing_exit_silent(self):
        out = build_exit_only_streak([
            _trade("BUY", "NVDA"),
            _trade("SELL", "AMD"),
        ])
        # 1 trailing exit is below the trim floor → must NOT surface a verdict.
        assert out["verdict"] == "MOST_RECENT_IS_ENTRY"
        assert out["exit_run_length"] == 1
        assert "below" in out["headline"].lower()

    def test_two_trailing_exits_silent(self):
        out = build_exit_only_streak([
            _trade("BUY", "NVDA"),
            _trade("SELL", "AMD"),
            _trade("SELL", "MU"),
        ])
        # 2 trailing exits is still below trim floor (=3).
        assert out["verdict"] == "MOST_RECENT_IS_ENTRY"
        assert out["exit_run_length"] == 2


# ── build_exit_only_streak — DEFENSIVE_TRIM ─────────────────────────────


class TestDefensiveTrim:

    def test_three_consecutive_sells(self):
        out = build_exit_only_streak([
            _trade("BUY", "NVDA"),
            _trade("SELL", "AMD"),
            _trade("SELL", "MU"),
            _trade("SELL", "TSLA"),
        ])
        assert out["verdict"] == "DEFENSIVE_TRIM"
        assert out["exit_run_length"] == 3
        assert "DEFENSIVE_TRIM" in out["headline"]
        # All three tickers surfaced (in walk order — newest exit first).
        assert out["exit_run_tickers"] == ["TSLA", "MU", "AMD"]

    def test_five_consecutive_sells_still_trim(self):
        ledger = [_trade("BUY", "FIRST")] + [
            _trade("SELL", f"T{i}") for i in range(5)
        ]
        out = build_exit_only_streak(ledger)
        # 5 < DEFENSIVE_LIQUIDATION_MIN (=6) → still TRIM, not LIQUIDATION.
        assert out["verdict"] == "DEFENSIVE_TRIM"
        assert out["exit_run_length"] == 5

    def test_sell_call_counts_as_exit(self):
        out = build_exit_only_streak([
            _trade("BUY", "NVDA"),
            _trade("SELL_CALL", "AAPL"),
            _trade("SELL_PUT", "MSFT"),
            _trade("SELL", "TSLA"),
        ])
        assert out["verdict"] == "DEFENSIVE_TRIM"
        assert out["exit_run_length"] == 3

    def test_unknown_action_between_exits_does_not_break_run(self):
        # REBALANCE is skipped entirely; the 3 SELLs remain consecutive in
        # the *filtered* sequence the analyzer reads.
        out = build_exit_only_streak([
            _trade("BUY", "NVDA"),
            _trade("SELL", "AMD"),
            _trade("REBALANCE", "X"),
            _trade("SELL", "MU"),
            _trade("SELL", "TSLA"),
        ])
        assert out["verdict"] == "DEFENSIVE_TRIM"
        assert out["exit_run_length"] == 3


# ── build_exit_only_streak — DEFENSIVE_LIQUIDATION ──────────────────────


class TestDefensiveLiquidation:

    def test_six_consecutive_sells(self):
        ledger = [_trade("BUY", "FIRST")] + [
            _trade("SELL", f"T{i}") for i in range(6)
        ]
        out = build_exit_only_streak(ledger)
        assert out["verdict"] == "DEFENSIVE_LIQUIDATION"
        assert out["exit_run_length"] == 6
        assert "liquidating" in out["headline"].lower()

    def test_only_exits_no_entries_ever(self):
        # No BUY ever in history — run length = total exits.
        ledger = [_trade("SELL", f"T{i}") for i in range(10)]
        out = build_exit_only_streak(ledger)
        assert out["verdict"] == "DEFENSIVE_LIQUIDATION"
        assert out["exit_run_length"] == 10
        assert out["last_entry_ts"] is None
        assert out["hours_since_last_entry"] is None
        # Headline must NOT crash on missing last-entry — renders "n/a".
        assert "n/a" in out["headline"]

    def test_hours_since_last_entry_computed(self):
        # The builder reads real wall-clock `now`; we cannot pin it, so
        # verify only that the field IS the float-hours delta between the
        # BUY ts and `now` and is non-negative. The exact value drifts with
        # the wall clock and is not what this test is locking.
        now = datetime.now(timezone.utc)
        ledger = [_trade("BUY", "NVDA", ts=now - timedelta(hours=14))]
        ledger += [_trade("SELL", f"T{i}",
                          ts=now - timedelta(hours=10 - i))
                   for i in range(6)]
        out = build_exit_only_streak(ledger)
        assert out["verdict"] == "DEFENSIVE_LIQUIDATION"
        assert out["hours_since_last_entry"] is not None
        # Tight tolerance against the synthetic ts spread.
        assert 13.9 < out["hours_since_last_entry"] < 14.1


# ── recent_sequence rendering ───────────────────────────────────────────


class TestRecentSequence:

    def test_renders_e_and_x(self):
        out = build_exit_only_streak([
            _trade("BUY"),
            _trade("SELL"),
            _trade("BUY_CALL"),
            _trade("SELL_PUT"),
        ])
        assert out["recent_sequence"] == ["E", "X", "E", "X"]

    def test_capped_to_window(self):
        # Build more than the cap; only the tail of length RECENT_SEQUENCE_LEN
        # should survive.
        ledger = [_trade("BUY" if i % 2 == 0 else "SELL")
                  for i in range(RECENT_SEQUENCE_LEN + 5)]
        out = build_exit_only_streak(ledger)
        assert len(out["recent_sequence"]) == RECENT_SEQUENCE_LEN

    def test_unknown_actions_excluded(self):
        out = build_exit_only_streak([
            _trade("BUY"),
            _trade("REBALANCE"),
            _trade("SELL"),
        ])
        # REBALANCE not in sequence at all.
        assert out["recent_sequence"] == ["E", "X"]


# ── degrade-safe contract ───────────────────────────────────────────────


class TestDegradeSafe:

    def test_malformed_action_skipped_not_raised(self):
        out = build_exit_only_streak([
            {"action": None, "ticker": "X"},
            _trade("BUY"),
            _trade("SELL"),
            _trade("SELL"),
            _trade("SELL"),
        ])
        # Malformed row counts as unknown → skipped; trailing 3 SELLs surface.
        assert out["verdict"] == "DEFENSIVE_TRIM"
        assert out["exit_run_length"] == 3

    def test_missing_ticker_in_exit_run(self):
        # A trade with no ticker still counts as an exit but contributes
        # nothing to the exit_run_tickers list.
        ledger = [
            _trade("BUY", "NVDA"),
            {"action": "SELL", "ticker": "", "timestamp": "2026-05-23T10:00:00+00:00"},
            _trade("SELL", "MU"),
            _trade("SELL", "TSLA"),
        ]
        out = build_exit_only_streak(ledger)
        assert out["verdict"] == "DEFENSIVE_TRIM"
        # The blank-ticker row is included in the run but not the ticker list.
        assert "" not in out["exit_run_tickers"]
        assert "TSLA" in out["exit_run_tickers"]

    def test_constants_exposed(self):
        out = build_exit_only_streak([_trade("BUY")])
        assert out["defensive_trim_min"] == DEFENSIVE_TRIM_MIN
        assert out["defensive_liquidation_min"] == DEFENSIVE_LIQUIDATION_MIN

    def test_n_entries_n_exits_summed_correctly(self):
        out = build_exit_only_streak([
            _trade("BUY"), _trade("BUY"), _trade("SELL"),
            _trade("BUY_CALL"), _trade("SELL_PUT"),
            _trade("REBALANCE"),  # ignored
        ])
        assert out["n_entries"] == 3
        assert out["n_exits"] == 2
        assert out["n_total_fills"] == 5

    def test_returns_dict_not_none(self):
        # Type contract is fundamental — callers index into the dict.
        out = build_exit_only_streak([])
        assert isinstance(out, dict)

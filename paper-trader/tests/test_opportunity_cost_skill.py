"""Pure-builder tests for ``analytics.opportunity_cost_skill``.

Every test injects deterministic callables — the builder never touches
the DB, articles store, or yfinance. Pins each verdict branch + each
sit-out classification + sample emission.
"""
from datetime import datetime, timedelta, timezone

import pytest

from paper_trader.analytics.opportunity_cost_skill import (
    DEFAULT_DEFENSIVE_PCT_CEIL,
    DEFAULT_OK_PCT_FLOOR,
    DEFAULT_RUNNER_PCT_FLOOR,
    MIN_DECISIONS_FOR_VERDICT,
    _aggregate_verdict,
    _classify_row,
    _is_sitout,
    build_opportunity_cost_skill,
    top_ticker_by_heat,
)


# --------------------------------------------------------------------------
# _is_sitout — sit-out detector
# --------------------------------------------------------------------------
class TestIsSitout:
    def test_hold_cash_branch_is_sitout(self):
        assert _is_sitout("HOLD CASH → HOLD") is True

    def test_bare_hold_cash_is_sitout(self):
        assert _is_sitout("HOLD CASH") is True

    def test_no_decision_is_sitout(self):
        assert _is_sitout("NO_DECISION") is True

    def test_no_decision_with_suffix_still_sitout(self):
        assert _is_sitout("NO_DECISION (host_saturated)") is True

    def test_filled_buy_is_not_sitout(self):
        assert _is_sitout("BUY NVDA → FILLED") is False

    def test_filled_sell_is_not_sitout(self):
        assert _is_sitout("SELL NVDA → FILLED") is False

    def test_blocked_is_not_sitout(self):
        assert _is_sitout("BUY NVDA → BLOCKED") is False

    def test_rebalance_is_not_sitout(self):
        assert _is_sitout("REBALANCE") is False

    def test_empty_string_not_sitout(self):
        assert _is_sitout("") is False

    def test_none_not_sitout(self):
        assert _is_sitout(None) is False

    def test_non_string_not_sitout(self):
        assert _is_sitout(42) is False
        assert _is_sitout({"action": "HOLD"}) is False


# --------------------------------------------------------------------------
# _classify_row — per-row verdict from fwd_3d_pct
# --------------------------------------------------------------------------
class TestClassifyRow:
    def _f(self, v):
        return _classify_row(
            v,
            runner_pct_floor=DEFAULT_RUNNER_PCT_FLOOR,
            ok_pct_floor=DEFAULT_OK_PCT_FLOOR,
            defensive_pct_ceil=DEFAULT_DEFENSIVE_PCT_CEIL,
        )

    def test_runner_at_floor(self):
        assert self._f(5.0) == "MISSED_RUNNER"

    def test_runner_above_floor(self):
        assert self._f(7.5) == "MISSED_RUNNER"

    def test_ok_at_floor(self):
        assert self._f(1.0) == "MISSED_OK"

    def test_ok_below_runner_floor(self):
        assert self._f(3.9) == "MISSED_OK"

    def test_neutral_just_below_ok(self):
        assert self._f(0.5) == "NEUTRAL_HOLD"

    def test_neutral_just_above_defensive(self):
        assert self._f(-0.9) == "NEUTRAL_HOLD"

    def test_defensive_at_ceil(self):
        assert self._f(-1.0) == "DEFENSIVE_HIT"

    def test_defensive_below_ceil(self):
        assert self._f(-3.5) == "DEFENSIVE_HIT"

    def test_none_is_no_fwd(self):
        assert self._f(None) == "NO_FWD"

    def test_nan_is_no_fwd(self):
        assert self._f(float("nan")) == "NO_FWD"

    def test_garbage_is_no_fwd(self):
        assert self._f("not-a-float") == "NO_FWD"


# --------------------------------------------------------------------------
# _aggregate_verdict — verdict ladder
# --------------------------------------------------------------------------
class TestAggregateVerdict:
    def _agg(self, **kw):
        defaults = dict(
            n_runner=0, n_ok=0, n_neutral=0, n_defensive=0,
            mean_fwd_3d_pct=0.0,
            min_decisions=5, missed_pct_floor=50.0, mean_fwd_pct_floor=2.0,
        )
        defaults.update(kw)
        return _aggregate_verdict(**defaults)

    def test_below_min_decisions_is_no_data(self):
        v, h = self._agg(n_runner=1, n_neutral=1, mean_fwd_3d_pct=10.0)
        assert v == "NO_DATA"
        assert "accumulate more" in h

    def test_missed_alpha_pure_runners(self):
        v, h = self._agg(n_runner=6, mean_fwd_3d_pct=8.5)
        assert v == "MISSED_ALPHA"
        assert "8.50" in h or "+8.50" in h

    def test_missed_alpha_mixed_runners_and_ok(self):
        v, h = self._agg(n_runner=2, n_ok=2, n_neutral=2, mean_fwd_3d_pct=3.5)
        assert v == "MISSED_ALPHA"

    def test_defensive_win(self):
        v, h = self._agg(n_defensive=5, mean_fwd_3d_pct=-4.2)
        assert v == "DEFENSIVE_WIN"

    def test_neutral_below_pct_floor(self):
        v, h = self._agg(n_runner=1, n_ok=1, n_neutral=3, mean_fwd_3d_pct=0.5)
        # missed_pct = 40% < 50%
        assert v == "NEUTRAL"

    def test_neutral_mean_too_small_for_missed_alpha(self):
        v, h = self._agg(n_runner=3, n_ok=2, n_neutral=0, mean_fwd_3d_pct=1.0)
        # missed_pct = 100% but mean < 2.0
        assert v == "NEUTRAL"

    def test_missed_alpha_at_exact_thresholds(self):
        v, h = self._agg(n_runner=3, n_ok=2, n_neutral=3, n_defensive=2,
                         mean_fwd_3d_pct=2.0)
        # missed = 50%, mean = +2.0 — both at the floors
        assert v == "MISSED_ALPHA"


# --------------------------------------------------------------------------
# build_opportunity_cost_skill — end-to-end pure builder
# --------------------------------------------------------------------------
class TestBuildOpportunityCostSkill:
    NOW = datetime(2026, 5, 24, 4, 0, 0, tzinfo=timezone.utc)

    def _dec(self, hours_ago: float, action: str = "HOLD CASH → HOLD"):
        ts = (self.NOW - timedelta(hours=hours_ago)).isoformat()
        return {"action_taken": action, "timestamp": ts, "reasoning": ""}

    def test_empty_input_is_no_data(self):
        rep = build_opportunity_cost_skill([], now=self.NOW)
        assert rep["verdict"] == "NO_DATA"
        assert rep["stats"]["n_sitout_total"] == 0
        assert rep["samples"] == []

    def test_none_input_is_no_data(self):
        rep = build_opportunity_cost_skill(None, now=self.NOW)
        assert rep["verdict"] == "NO_DATA"

    def test_filled_decisions_are_skipped(self):
        decisions = [
            self._dec(1, action="BUY NVDA → FILLED"),
            self._dec(2, action="SELL NVDA → FILLED"),
        ]
        rep = build_opportunity_cost_skill(
            decisions, now=self.NOW,
            top_ticker_at=lambda ts: ("NVDA", 10.0),
            forward_returns_for=lambda tk, ts: (1.0, 8.0),
        )
        # No sit-outs → NO_DATA
        assert rep["stats"]["n_sitout_total"] == 0
        assert rep["verdict"] == "NO_DATA"

    def test_decisions_outside_window_are_dropped(self):
        decisions = [self._dec(200) for _ in range(10)]  # > 168h
        rep = build_opportunity_cost_skill(decisions, now=self.NOW)
        assert rep["stats"]["n_sitout_total"] == 0

    def test_missed_alpha_branch(self):
        decisions = [self._dec(2.0 * i + 5) for i in range(6)]
        rep = build_opportunity_cost_skill(
            decisions, now=self.NOW,
            top_ticker_at=lambda ts: ("NVDA", 12.0),
            forward_returns_for=lambda tk, ts: (2.0, 7.0),
        )
        assert rep["verdict"] == "MISSED_ALPHA"
        assert rep["stats"]["n_missed_runner"] == 6
        assert rep["stats"]["n_classified"] == 6
        assert rep["stats"]["missed_pct"] == 100.0
        assert rep["stats"]["mean_fwd_3d_pct"] == 7.0
        # All samples carry the runner verdict
        assert all(s["verdict"] == "MISSED_RUNNER" for s in rep["samples"])

    def test_defensive_win_branch(self):
        decisions = [self._dec(2.0 * i + 5) for i in range(6)]
        rep = build_opportunity_cost_skill(
            decisions, now=self.NOW,
            top_ticker_at=lambda ts: ("NVDA", 12.0),
            forward_returns_for=lambda tk, ts: (None, -4.5),
        )
        assert rep["verdict"] == "DEFENSIVE_WIN"
        assert rep["stats"]["n_defensive"] == 6
        assert rep["stats"]["defensive_pct"] == 100.0

    def test_neutral_branch_mixed(self):
        # 2 runners, 2 neutrals, 2 defensive — neither side hits 50% missed
        # AND mean stays inside the +/-2 window
        decisions = [self._dec(2.0 * i + 5) for i in range(6)]
        returns = [7.0, 8.0, 0.5, -0.5, -1.5, -2.0]
        idx = {"i": 0}

        def fwd(tk, ts):
            i = idx["i"]
            idx["i"] += 1
            return (None, returns[i])

        rep = build_opportunity_cost_skill(
            decisions, now=self.NOW,
            top_ticker_at=lambda ts: ("NVDA", 12.0),
            forward_returns_for=fwd,
        )
        assert rep["verdict"] == "NEUTRAL"

    def test_no_candidate_does_not_classify(self):
        decisions = [self._dec(2.0 * i + 5) for i in range(6)]
        rep = build_opportunity_cost_skill(
            decisions, now=self.NOW,
            top_ticker_at=lambda ts: None,
            forward_returns_for=lambda tk, ts: (1.0, 5.0),
        )
        assert rep["stats"]["n_no_candidate"] == 6
        assert rep["stats"]["n_classified"] == 0
        assert rep["verdict"] == "NO_DATA"

    def test_no_fwd_does_not_classify(self):
        decisions = [self._dec(2.0 * i + 5) for i in range(8)]
        rep = build_opportunity_cost_skill(
            decisions, now=self.NOW,
            top_ticker_at=lambda ts: ("NVDA", 12.0),
            forward_returns_for=lambda tk, ts: (None, None),
        )
        assert rep["stats"]["n_no_fwd"] == 8
        assert rep["stats"]["n_classified"] == 0
        assert rep["verdict"] == "NO_DATA"

    def test_top_ticker_callback_raising_is_safe(self):
        decisions = [self._dec(2.0 * i + 5) for i in range(6)]

        def boom(ts):
            raise RuntimeError("articles.db unavailable")

        rep = build_opportunity_cost_skill(
            decisions, now=self.NOW,
            top_ticker_at=boom,
            forward_returns_for=lambda tk, ts: (1.0, 5.0),
        )
        # All rows counted as no-candidate; never raises
        assert rep["stats"]["n_no_candidate"] == 6
        assert rep["verdict"] == "NO_DATA"

    def test_forward_returns_callback_raising_is_safe(self):
        decisions = [self._dec(2.0 * i + 5) for i in range(6)]

        def boom(tk, ts):
            raise RuntimeError("yfinance throttled")

        rep = build_opportunity_cost_skill(
            decisions, now=self.NOW,
            top_ticker_at=lambda ts: ("NVDA", 12.0),
            forward_returns_for=boom,
        )
        assert rep["stats"]["n_no_fwd"] == 6
        assert rep["verdict"] == "NO_DATA"

    def test_sample_limit_caps_samples(self):
        decisions = [self._dec(0.1 * i + 1) for i in range(30)]
        rep = build_opportunity_cost_skill(
            decisions, now=self.NOW,
            top_ticker_at=lambda ts: ("NVDA", 12.0),
            forward_returns_for=lambda tk, ts: (1.0, 6.0),
            sample_limit=5,
        )
        assert len(rep["samples"]) == 5
        assert rep["stats"]["n_classified"] == 30

    def test_no_decision_rows_count_as_sitouts(self):
        decisions = [
            self._dec(i + 1, action="NO_DECISION") for i in range(6)
        ]
        rep = build_opportunity_cost_skill(
            decisions, now=self.NOW,
            top_ticker_at=lambda ts: ("NVDA", 9.0),
            forward_returns_for=lambda tk, ts: (1.5, 6.5),
        )
        assert rep["stats"]["n_sitout_total"] == 6
        assert rep["verdict"] == "MISSED_ALPHA"

    def test_thresholds_block_echoed(self):
        rep = build_opportunity_cost_skill([], now=self.NOW)
        th = rep["thresholds"]
        assert th["runner_pct_floor"] == DEFAULT_RUNNER_PCT_FLOOR
        assert th["min_decisions"] == MIN_DECISIONS_FOR_VERDICT


# --------------------------------------------------------------------------
# top_ticker_by_heat — pure article-heat helper
# --------------------------------------------------------------------------
class TestTopTickerByHeat:
    NOW = datetime(2026, 5, 24, 4, 0, 0, tzinfo=timezone.utc)

    def _art(self, hours_ago: float, title: str, ai_score=5.0, urgency=0):
        ts = (self.NOW - timedelta(hours=hours_ago)).isoformat()
        return {
            "title": title,
            "ai_score": ai_score,
            "urgency": urgency,
            "first_seen": ts,
        }

    def test_picks_highest_heat_ticker(self):
        arts = [
            self._art(0.5, "NVDA earnings smash records", 8.0, 2),
            self._art(0.5, "AMD modest guide", 5.0, 0),
            self._art(0.5, "MSFT routine update", 3.0, 0),
        ]
        out = top_ticker_by_heat(arts, ["NVDA", "AMD", "MSFT"], self.NOW)
        assert out is not None
        tk, heat = out
        assert tk == "NVDA"
        assert heat > 0

    def test_excludes_articles_outside_window(self):
        arts = [self._art(48.0, "NVDA blowout", 9.0, 2)]
        out = top_ticker_by_heat(
            arts, ["NVDA"], self.NOW, lookback_hours=2.0,
        )
        assert out is None

    def test_returns_none_when_no_match(self):
        arts = [self._art(0.5, "Generic market update", 5.0, 0)]
        out = top_ticker_by_heat(arts, ["NVDA"], self.NOW)
        assert out is None

    def test_returns_none_on_empty(self):
        assert top_ticker_by_heat([], ["NVDA"], self.NOW) is None
        assert top_ticker_by_heat(
            [self._art(0.5, "x")], [], self.NOW,
        ) is None

    def test_cashtag_matches(self):
        arts = [self._art(0.5, "$NVDA cashtag mention", 6.0, 1)]
        out = top_ticker_by_heat(arts, ["NVDA"], self.NOW)
        assert out is not None and out[0] == "NVDA"

    def test_urgency_amplifies_heat(self):
        arts_low = [self._art(0.5, "NVDA news", 5.0, 0)]
        arts_high = [self._art(0.5, "NVDA news", 5.0, 2)]
        low = top_ticker_by_heat(arts_low, ["NVDA"], self.NOW)
        high = top_ticker_by_heat(arts_high, ["NVDA"], self.NOW)
        assert low is not None and high is not None
        assert high[1] > low[1]

    def test_first_ticker_match_wins_per_article(self):
        # Article mentions NVDA + AMD. WATCHLIST order is NVDA then AMD.
        arts = [self._art(0.5, "NVDA and AMD both beat", 5.0, 0)]
        out = top_ticker_by_heat(arts, ["NVDA", "AMD"], self.NOW)
        assert out is not None and out[0] == "NVDA"

    def test_zero_ai_score_drops_article(self):
        arts = [self._art(0.5, "NVDA news", 0.0, 2)]
        out = top_ticker_by_heat(arts, ["NVDA"], self.NOW)
        assert out is None

    def test_garbage_first_seen_dropped(self):
        a = {"title": "NVDA", "ai_score": 5.0, "urgency": 1,
             "first_seen": "not-a-date"}
        out = top_ticker_by_heat([a], ["NVDA"], self.NOW)
        assert out is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

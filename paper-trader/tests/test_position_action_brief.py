"""Tests for analytics/position_action_brief.py — per-held-position composite.

Discriminating locks:

* **Action-precedence ladder** — TRIM_BEFORE_EVENT > HOLD_THROUGH_EVENT >
  RESTART_RUNNER > MONITOR > OK. A regression that reorders fails.
* **The wedge × held-imminent case** = TRIM_BEFORE_EVENT — the dangerous
  combination of high empty_rate + earnings <24h + held exposure.
* **Decision-status taxonomy** is stable: DECIDED / EMPTY / HOST_SKIP /
  PARSE_FAIL / NEVER. The classifier reads the action_taken/reasoning
  shape that already lives in paper_trader.db.
* **Event proximity overrides news state** — a SURGING news read does NOT
  pivot HOLD_THROUGH_EVENT to TRIM unless the bot is also wedged.
* **Option lots fold into the underlying** — a NVDA stock + NVDA call lot
  produces one brief, exposure summed.
* **n_imminent_events** counts unique tickers with hours_to_event <= 24h
  (not events — option duplication shouldn't double-count).
* **Sort order** — urgency desc, exposure desc, ticker asc. Deterministic.
* **Robustness** — empty positions, missing news/event data, malformed
  rows degrade gracefully, never raise.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.position_action_brief import (
    ACTION_HOLD_THROUGH_EVENT,
    ACTION_MONITOR,
    ACTION_OK,
    ACTION_RESTART_RUNNER,
    ACTION_TRIM_BEFORE_EVENT,
    DECISION_DECIDED,
    DECISION_EMPTY,
    DECISION_HOST_SKIP,
    DECISION_NEVER,
    DECISION_PARSE_FAIL,
    EVENT_IMMINENT_HOURS,
    NEGLECTED_AGE_MIN,
    WEDGED_EMPTY_RATE,
    _classify_decision,
    build_position_action_brief,
)

_NOW = datetime(2026, 5, 19, 12, 30, 0, tzinfo=timezone.utc)


def _position(ticker: str, qty: float = 1.0, avg: float = 100.0,
              cur: float = 110.0, side: str = "LONG") -> dict:
    return {
        "ticker": ticker,
        "quantity": qty,
        "avg_cost": avg,
        "current_price": cur,
        "side": side,
    }


def _decision(action: str, mins_ago: float, reasoning: str = "") -> dict:
    ts = _NOW - timedelta(minutes=mins_ago)
    return {
        "timestamp": ts.isoformat(),
        "action_taken": action,
        "reasoning": reasoning,
    }


def _event(ticker: str, days_away: float, verdict: str = "BLIND",
           tier: str = "HELD_IMMINENT") -> dict:
    """Synthesise an event row.

    Default emits BOTH ``verdict`` (build_event_readiness shape) AND ``tier``
    (earnings-risk shape) so test fixtures cover both producers. Pass
    ``verdict=None`` to simulate an earnings-risk-only event.
    """
    out: dict = {
        "ticker": ticker,
        "days_away": days_away,
        "earnings_date": (_NOW + timedelta(days=days_away)).isoformat(),
    }
    if verdict is not None:
        out["verdict"] = verdict
    if tier is not None:
        out["tier"] = tier
    return out


def _news_velocity(rows: list[dict]) -> dict:
    return {
        "as_of": _NOW.isoformat(),
        "state": "OK",
        "per_ticker": rows,
        "n_held": len(rows),
    }


def _news_row(ticker: str, state: str, wn: int = 0,
              z: float | None = None,
              top: str | None = None) -> dict:
    return {
        "ticker": ticker,
        "state": state,
        "window_count": wn,
        "z_score": z,
        "max_ai_score_window": 8.0 if state == "SURGING" else None,
        "top_window_title": top,
    }


def _build(positions, decisions=None, news=None, events=None,
           empty_rate=None, saturated=None, starting=1000.0):
    return build_position_action_brief(
        positions=positions,
        decisions=decisions or [],
        news_velocity=news,
        held_events=events,
        empty_rate_pct=empty_rate,
        host_saturated=saturated,
        starting_equity_usd=starting,
        now=_NOW,
    )


# ───────────────────────── Decision classifier ─────────────────────────


class TestClassifyDecision:

    def test_buy_is_decided(self):
        assert _classify_decision("BUY NVDA → FILLED", "") == DECISION_DECIDED

    def test_sell_is_decided(self):
        assert _classify_decision("SELL TQQQ → FILLED", "") == DECISION_DECIDED

    def test_hold_is_decided(self):
        assert _classify_decision("HOLD NVDA", "") == DECISION_DECIDED

    def test_blocked_is_decided(self):
        # The risk gate fired — the bot's brain produced an opinion.
        assert _classify_decision("BLOCKED — qty exceeds held",
                                  "") == DECISION_DECIDED

    def test_empty_is_empty(self):
        assert _classify_decision(
            "NO_DECISION",
            "claude returned no response (timeout)") == DECISION_EMPTY

    def test_host_skip_is_host_skip(self):
        assert _classify_decision(
            "NO_DECISION",
            "host saturated — skipped claude call") == DECISION_HOST_SKIP

    def test_parse_fail_is_parse_fail(self):
        assert _classify_decision(
            "NO_DECISION", "parse_failed: ...") == DECISION_PARSE_FAIL

    def test_unknown_no_decision_falls_to_empty(self):
        assert _classify_decision("NO_DECISION", "") == DECISION_EMPTY

    def test_none_action_is_never(self):
        assert _classify_decision(None, "") == DECISION_NEVER


# ───────────────────────── Empty / robustness ─────────────────────────


class TestRobustness:

    def test_empty_positions(self):
        r = _build([])
        assert r["n_positions"] == 0
        assert r["briefs"] == []
        assert r["overall_action"] == ACTION_OK
        assert r["overall_urgency"] == 0.0

    def test_pseudo_tickers_dropped(self):
        r = _build([
            _position("CASH"),
            _position("NONE"),
            _position("NO_DECISION"),
            _position("NVDA"),
        ])
        tickers = [b["ticker"] for b in r["briefs"]]
        assert tickers == ["NVDA"]

    def test_malformed_position_rows_skipped(self):
        r = _build([
            None,                # type: ignore[list-item]
            "not-a-dict",        # type: ignore[list-item]
            _position("NVDA"),
            {"ticker": "", "quantity": 1.0, "avg_cost": 100.0,
             "current_price": 100.0},
        ])
        tickers = [b["ticker"] for b in r["briefs"]]
        assert tickers == ["NVDA"]

    def test_malformed_news_velocity_does_not_raise(self):
        r = _build([_position("NVDA")], news={"garbage": True})
        assert r["briefs"][0]["news_state"] == "INSUFFICIENT"

    def test_malformed_events_do_not_raise(self):
        r = _build([_position("NVDA")],
                   events=[None, "bad", {"ticker": "NVDA"}])  # type: ignore[list-item]
        assert r["briefs"][0]["hours_to_event"] is None


# ───────────────────────── Aggregation (stock + option fold) ─────────────────────────


class TestAggregation:

    def test_stock_plus_option_fold_into_one_brief(self):
        r = _build([
            _position("NVDA", qty=10, avg=100.0, cur=110.0),
            _position("NVDA", qty=1, avg=5.0, cur=6.0),  # call option
        ])
        assert len(r["briefs"]) == 1
        b = r["briefs"][0]
        assert b["ticker"] == "NVDA"
        # 10*110 + 1*6 = 1106 market value
        assert b["exposure_usd"] == pytest.approx(1106.0, abs=0.01)
        # 10*100 + 1*5 = 1005 cost basis
        assert b["cost_basis_usd"] == pytest.approx(1005.0, abs=0.01)
        # unrealized = 1106 - 1005 = 101
        assert b["unrealized_pl_usd"] == pytest.approx(101.0, abs=0.01)
        assert b["n_lots"] == 2

    def test_pct_portfolio_uses_starting_equity_when_supplied(self):
        r = _build([_position("NVDA", qty=10, avg=100.0, cur=100.0)],
                   starting=2000.0)
        # 10*100 = 1000 exposure on a 2000 book = 50%
        assert r["briefs"][0]["pct_portfolio"] == 50.0

    def test_pct_portfolio_falls_back_to_open_only(self):
        r = _build([_position("NVDA", qty=10, avg=100.0, cur=100.0)],
                   starting=None)
        # 100% of open positions
        assert r["briefs"][0]["pct_portfolio"] == 100.0


# ───────────────────────── The wedge × imminent (anchor case) ─────────────────────────


class TestTrimBeforeEvent:
    """The exact pathology this builder was designed to surface."""

    def test_imminent_print_with_high_empty_rate_trims(self):
        r = _build(
            [_position("NVDA", qty=10, avg=100.0, cur=110.0)],
            decisions=[_decision("NO_DECISION", 60.0,
                                 "claude returned no response (timeout)")],
            events=[_event("NVDA", days_away=0.48)],
            empty_rate=81.4,
            saturated=True,
        )
        b = r["briefs"][0]
        assert b["recommended_action"] == ACTION_TRIM_BEFORE_EVENT
        assert b["urgency_score"] >= 0.9
        # The overall surface bubbles this up.
        assert r["overall_action"] == ACTION_TRIM_BEFORE_EVENT
        assert r["overall_urgency"] >= 0.9
        assert "URGENT" in r["headline"]

    def test_imminent_print_with_working_bot_holds_through(self):
        r = _build(
            [_position("NVDA", qty=10, avg=100.0, cur=110.0)],
            decisions=[_decision("HOLD NVDA", 5.0)],
            events=[_event("NVDA", days_away=0.48)],
            empty_rate=5.0,
        )
        b = r["briefs"][0]
        assert b["recommended_action"] == ACTION_HOLD_THROUGH_EVENT
        assert b["urgency_score"] < 0.9

    def test_imminent_print_zero_empty_rate_holds(self):
        r = _build(
            [_position("NVDA", qty=10, avg=100.0, cur=110.0)],
            decisions=[_decision("HOLD NVDA", 2.0)],
            events=[_event("NVDA", days_away=0.5)],
            empty_rate=0.0,
            saturated=False,
        )
        assert r["briefs"][0]["recommended_action"] == ACTION_HOLD_THROUGH_EVENT


# ───────────────────────── Restart-runner path (no imminent event) ─────────────────────────


class TestRestartRunner:

    def test_wedged_with_surging_news_recommends_restart(self):
        r = _build(
            [_position("NVDA")],
            decisions=[_decision("NO_DECISION", 300.0,
                                 "claude returned no response")],
            news=_news_velocity([
                _news_row("NVDA", "SURGING", wn=12, z=4.2,
                          top="NVDA breaking: HBM4 demand surge")
            ]),
            empty_rate=70.0,
        )
        b = r["briefs"][0]
        assert b["recommended_action"] == ACTION_RESTART_RUNNER
        # The news headline is folded into the reason bundle.
        assert any("SURGING" in s.upper() for s in b["reasons"])

    def test_wedged_without_news_still_restart(self):
        r = _build(
            [_position("NVDA")],
            decisions=[_decision("NO_DECISION", 300.0,
                                 "host saturated — skipped claude call")],
            empty_rate=55.0,
            saturated=True,
        )
        assert r["briefs"][0]["recommended_action"] == ACTION_RESTART_RUNNER


# ───────────────────────── MONITOR / OK paths ─────────────────────────


class TestMonitorAndOk:

    def test_surging_news_alone_is_monitor(self):
        r = _build(
            [_position("NVDA")],
            decisions=[_decision("HOLD NVDA", 10.0)],
            news=_news_velocity([
                _news_row("NVDA", "SURGING", wn=8, z=3.0,
                          top="NVDA breakout"),
            ]),
            empty_rate=5.0,
        )
        b = r["briefs"][0]
        assert b["recommended_action"] == ACTION_MONITOR

    def test_near_event_without_wedge_is_monitor(self):
        r = _build(
            [_position("NVDA")],
            decisions=[_decision("HOLD NVDA", 10.0)],
            events=[_event("NVDA", days_away=2.0)],  # 48h, not <24h
            empty_rate=5.0,
        )
        assert r["briefs"][0]["recommended_action"] == ACTION_MONITOR

    def test_stale_decision_without_wedge_is_monitor(self):
        # No-real-decision flag — bot has been silent ~2h but empty rate
        # is low so this is not a wedge.
        r = _build(
            [_position("NVDA")],
            decisions=[_decision("NO_DECISION", 120.0,
                                 "claude returned no response")],
            empty_rate=5.0,
        )
        b = r["briefs"][0]
        assert b["recommended_action"] in (ACTION_MONITOR, ACTION_OK)

    def test_clean_state_is_ok(self):
        r = _build(
            [_position("NVDA")],
            decisions=[_decision("HOLD NVDA", 5.0)],
            empty_rate=5.0,
            saturated=False,
        )
        b = r["briefs"][0]
        assert b["recommended_action"] == ACTION_OK
        assert b["urgency_score"] < 0.2


# ───────────────────────── Sorting / overall headline ─────────────────────────


class TestSortingAndHeadline:

    def test_most_urgent_first(self):
        r = _build(
            [
                _position("NVDA", qty=10, avg=100.0, cur=110.0),  # imminent print
                _position("TQQQ", qty=10, avg=50.0, cur=51.0),    # quiet
            ],
            decisions=[
                _decision("NO_DECISION", 60.0,
                          "claude returned no response"),
                _decision("HOLD TQQQ", 5.0),
            ],
            events=[_event("NVDA", days_away=0.48)],
            empty_rate=81.0,
        )
        assert r["briefs"][0]["ticker"] == "NVDA"
        assert r["briefs"][0]["urgency_score"] > r["briefs"][1]["urgency_score"]

    def test_tie_broken_by_exposure_then_ticker(self):
        # Two positions, both OK clean — sort by exposure desc, then ticker.
        r = _build(
            [
                _position("AAA", qty=10, avg=100.0, cur=100.0),  # exposure 1000
                _position("BBB", qty=20, avg=100.0, cur=100.0),  # exposure 2000
            ],
            decisions=[_decision("HOLD AAA", 5.0), _decision("HOLD BBB", 5.0)],
        )
        assert [b["ticker"] for b in r["briefs"]] == ["BBB", "AAA"]

    def test_n_imminent_counts_unique_tickers(self):
        # Same ticker present twice via stock + option fold ⇒ counted once.
        r = _build(
            [
                _position("NVDA", qty=10, avg=100.0, cur=110.0),
                _position("NVDA", qty=1, avg=5.0, cur=6.0),
            ],
            events=[_event("NVDA", days_away=0.48)],
        )
        assert r["n_imminent_events"] == 1


# ───────────────────────── Event-row schema interop ─────────────────────────


class TestEventRowSchemaInterop:
    """Regression: live route feeds events from ``build_event_readiness``
    whose rows emit ``verdict`` ∈ {BLIND, DEGRADED, READY, IMMINENT_OVERDUE}
    and have no ``tier`` key. The brief must surface the verdict semantic in
    ``event_verdict``, not the tier ladder."""

    def test_readiness_shape_only_verdict_reads_correctly(self):
        # Exactly the ``build_event_readiness`` row schema observed live.
        readiness_event = {
            "ticker": "NVDA",
            "days_away": 0.48,
            "hours_until_event": 11.5,
            "exposure_usd": 444.7,
            "expected_decisions_before_event": 1.94,
            "base_verdict": "BLIND",
            "verdict": "BLIND",
            "recommended_action": "trim",
            "earnings_date": "2026-05-20T00:00:00+00:00",
        }
        r = _build(
            [_position("NVDA", qty=10, avg=100.0, cur=110.0)],
            events=[readiness_event],
        )
        b = r["briefs"][0]
        assert b["event_verdict"] == "BLIND"
        assert b["hours_to_event"] == 11.5

    def test_earnings_risk_shape_only_tier_falls_back(self):
        # The other producer (/api/earnings-risk) emits ``tier`` not
        # ``verdict``. The fallback path must still surface it.
        er_event = {
            "ticker": "NVDA",
            "days_away": 0.48,
            "earnings_date": "2026-05-20T00:00:00+00:00",
            "tier": "HELD_IMMINENT",
            "held": True,
            "exposure_usd": 444.7,
        }
        r = _build([_position("NVDA")], events=[er_event])
        b = r["briefs"][0]
        assert b["event_verdict"] == "HELD_IMMINENT"

    def test_both_present_verdict_wins(self):
        # When both keys are populated the readiness verdict (strictly more
        # actionable than the tier) wins.
        ev = _event("NVDA", 0.48, verdict="BLIND", tier="HELD_IMMINENT")
        r = _build([_position("NVDA")], events=[ev])
        assert r["briefs"][0]["event_verdict"] == "BLIND"


# ───────────────────────── Schema stability ─────────────────────────


class TestSchemaStability:

    def test_brief_schema_keys(self):
        r = _build(
            [_position("NVDA")],
            decisions=[_decision("HOLD NVDA", 5.0)],
        )
        b = r["briefs"][0]
        for k in ("ticker", "exposure_usd", "cost_basis_usd",
                  "unrealized_pl_usd", "pct_portfolio", "n_lots", "side",
                  "hours_to_event", "event_verdict", "earnings_date",
                  "news_state", "news_window_count", "news_z_score",
                  "news_top_title", "news_max_ai_score",
                  "last_decision_status", "last_decision_action",
                  "last_decision_age_min", "last_decision_timestamp",
                  "recommended_action", "urgency_score", "reasons"):
            assert k in b, f"missing {k} in brief"

    def test_root_schema_keys(self):
        r = _build([_position("NVDA")])
        for k in ("as_of", "n_positions", "n_imminent_events",
                  "overall_action", "overall_urgency", "headline",
                  "inputs", "briefs"):
            assert k in r, f"missing {k} at root"

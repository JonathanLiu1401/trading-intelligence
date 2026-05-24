"""Tests for ``paper_trader.analytics.intent_followthrough_skill``.

Pure-helper tests — no Flask, no DB, no subprocess. Decision rows and
trade rows are hand-built to exercise every status transition and the
verdict ladder.

The intent extractor in ``decision_conditionals`` is the SSOT; we
compose its real output rather than fabricating intent dicts, so any
future change to the pattern set must keep these tests honest.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from paper_trader.analytics.intent_followthrough_skill import (
    DEFAULT_ABANDONED_MIN_N,
    DEFAULT_DISCIPLINE_FLOOR,
    DEFAULT_DRIFTING_FLOOR,
    DEFAULT_EVAL_WINDOW_HOURS,
    _infer_verb_hint,
    _verb_matches_hint,
    build_intent_followthrough,
    is_followthrough_abandoned,
)

NOW = datetime(2026, 5, 24, 12, 0, 0, tzinfo=timezone.utc)


def _decision(*, did, hours_ago, action_taken, reasoning):
    ts = NOW - timedelta(hours=hours_ago)
    return {
        "id": did,
        "timestamp": ts.isoformat(),
        "action_taken": action_taken,
        "reasoning": reasoning,
    }


def _trade(*, tid, hours_ago, ticker, action):
    ts = NOW - timedelta(hours=hours_ago)
    return {
        "id": tid,
        "timestamp": ts.isoformat(),
        "ticker": ticker,
        "action": action,
        "qty": 1.0,
        "price": 100.0,
    }


# ─── verb-hint primitives ──────────────────────────────────────────────

class TestVerbHint:
    def test_buy_keywords_detected(self):
        for snippet in (
            "ready to add on breakout",
            "going to buy the dip",
            "will scale in if it holds",
            "looking to initiate a starter position",
        ):
            assert _infer_verb_hint(snippet) == "BUY", snippet

    def test_sell_keywords_detected(self):
        for snippet in (
            "ready to trim on bounce",
            "prepared to exit if MACD flips",
            "will take profits at 230",
            "scale out into strength",
        ):
            assert _infer_verb_hint(snippet) == "SELL", snippet

    def test_sell_wins_over_buy_when_both_present(self):
        # The local action is SELL — selling here to fund a future BUY
        # elsewhere. The hint should reflect the trade the bot will make
        # FROM this decision, not a hypothetical reinvestment.
        snippet = "scale out of NVDA to add MU later"
        assert _infer_verb_hint(snippet) == "SELL"

    def test_no_keyword_returns_none(self):
        assert _infer_verb_hint("wait for the cash session") is None
        assert _infer_verb_hint("") is None
        assert _infer_verb_hint(None) is None  # type: ignore[arg-type]

    def test_verb_matches_hint_none_matches_anything(self):
        for action in ("BUY", "SELL", "REBALANCE", "BUY_CALL"):
            assert _verb_matches_hint(action, None)

    def test_verb_matches_hint_families(self):
        assert _verb_matches_hint("BUY", "BUY")
        assert _verb_matches_hint("BUY_CALL", "BUY")
        assert _verb_matches_hint("BUY_PUT", "BUY")
        assert _verb_matches_hint("SELL", "SELL")
        assert _verb_matches_hint("SELL_CALL", "SELL")
        assert not _verb_matches_hint("SELL", "BUY")
        assert not _verb_matches_hint("BUY", "SELL")

    def test_rebalance_is_verb_neutral(self):
        # A REBALANCE row satisfies either hint — it touches the book in
        # both directions by definition.
        assert _verb_matches_hint("REBALANCE", "BUY")
        assert _verb_matches_hint("REBALANCE", "SELL")

    def test_garbage_action_returns_false(self):
        assert not _verb_matches_hint(None, "BUY")
        assert not _verb_matches_hint(123, "BUY")  # type: ignore[arg-type]
        assert not _verb_matches_hint("MAYBE", "BUY")


# ─── end-to-end: empty / NO_DATA ───────────────────────────────────────

class TestEmptyInputs:
    def test_no_decisions_no_trades_yields_no_data(self):
        r = build_intent_followthrough([], [], now=NOW)
        assert r["state"] == "NO_DATA"
        assert r["verdict"] == "NO_DATA"
        assert r["n_intents"] == 0
        assert r["n_actionable"] == 0
        assert r["followthrough_rate"] is None
        assert r["abstention"] == {
            "n_preserve_deployed": 0, "n_preserve_active": 0,
            "n_preserve_dead": 0, "n_restraint_held": 0,
            "n_restraint_broken": 0,
        }

    def test_decisions_without_intent_keywords_yields_no_data(self):
        # Both decisions present, neither carries a parseable intent
        # snippet — base extractor returns no intents → NO_DATA.
        decisions = [
            _decision(did=1, hours_ago=1.0, action_taken="HOLD CASH → HOLD",
                      reasoning="market closed quiet"),
            _decision(did=2, hours_ago=2.0, action_taken="HOLD CASH → HOLD",
                      reasoning="nothing actionable"),
        ]
        r = build_intent_followthrough(decisions, [], now=NOW)
        assert r["verdict"] == "NO_DATA"
        assert r["n_intents"] == 0

    def test_garbage_rows_do_not_raise(self):
        decisions = [None, 42, {}, {"id": 1}, "string"]  # type: ignore[list-item]
        trades = [None, "x", {}, {"timestamp": None}]    # type: ignore[list-item]
        r = build_intent_followthrough(decisions, trades, now=NOW)
        assert r["state"] == "NO_DATA"
        assert r["verdict"] == "NO_DATA"


# ─── FOLLOWED: matching trade after intent ─────────────────────────────

class TestFollowedStatus:
    def test_actionable_intent_with_matching_trade_is_followed(self):
        # "Wait for cash session ... rotate into NVDA" intent in a
        # HOLD NVDA decision 2h ago → BUY NVDA trade 1h ago. Match.
        d = _decision(
            did=10, hours_ago=2.0,
            action_taken="HOLD NVDA → HOLD",
            reasoning=(
                "wait for cash session price action before deciding "
                "to add NVDA on a fresh catalyst"
            ),
        )
        # BUY NVDA happened 1h later (1h ago from NOW) — inside the
        # 12h evaluation window.
        t = _trade(tid=1, hours_ago=1.0, ticker="NVDA", action="BUY")
        r = build_intent_followthrough([d], [t], now=NOW)
        assert r["state"] == "OK"
        assert r["n_followed"] >= 1
        followed = [i for i in r["intents"] if i["status"] == "FOLLOWED"]
        assert followed, r
        # The match metadata should carry trade id + age.
        m = followed[0]["match"]
        assert m["trade_id"] == 1
        assert m["trade_ticker"] == "NVDA"
        assert m["trade_action"] == "BUY"
        assert 0.5 < m["age_after_intent_h"] < 1.5

    def test_match_must_be_after_intent_not_before(self):
        # Trade fired 3h ago, intent fired 2h ago. Trade is BEFORE
        # the intent — must NOT match.
        d = _decision(
            did=20, hours_ago=2.0,
            action_taken="HOLD NVDA → HOLD",
            reasoning="ready to add NVDA on the next confirmation",
        )
        t = _trade(tid=2, hours_ago=3.0, ticker="NVDA", action="BUY")
        r = build_intent_followthrough([d], [t], now=NOW)
        # No FOLLOWED. The intent's status depends on age vs eval_window;
        # at 2h ago with default 12h eval_window it's still PENDING.
        actionable = [i for i in r["intents"]
                      if i.get("bucket") == "ACTIONABLE"]
        assert actionable
        assert all(i["status"] != "FOLLOWED" for i in actionable)
        assert any(i["status"] == "PENDING" for i in actionable)

    def test_verb_hint_filters_out_wrong_direction_trade(self):
        # Intent says "ready to TRIM" (SELL hint). A BUY trade afterwards
        # must NOT satisfy this intent.
        d = _decision(
            did=30, hours_ago=4.0,
            action_taken="HOLD NVDA → HOLD",
            reasoning="ready to trim NVDA on bounce to 230",
        )
        t = _trade(tid=3, hours_ago=2.0, ticker="NVDA", action="BUY")
        r = build_intent_followthrough([d], [t], now=NOW)
        actionable = [i for i in r["intents"]
                      if i.get("bucket") == "ACTIONABLE"]
        # The BUY does not satisfy the SELL hint → not FOLLOWED.
        assert all(i["status"] != "FOLLOWED" for i in actionable), (
            actionable
        )

    def test_ticker_mismatch_is_not_a_match(self):
        d = _decision(
            did=40, hours_ago=4.0,
            action_taken="HOLD NVDA → HOLD",
            reasoning="ready to add NVDA on a fresh print",
        )
        # MU trade does not satisfy NVDA intent.
        t = _trade(tid=4, hours_ago=2.0, ticker="MU", action="BUY")
        r = build_intent_followthrough([d], [t], now=NOW)
        actionable = [i for i in r["intents"]
                      if i.get("bucket") == "ACTIONABLE"]
        assert actionable
        assert all(i["status"] != "FOLLOWED" for i in actionable)


# ─── PENDING vs ABANDONED ──────────────────────────────────────────────

class TestPendingAbandoned:
    def test_fresh_intent_no_match_is_pending(self):
        # Intent emitted 2h ago, no matching trade. eval_window=12h
        # default → intent is still inside its evaluation window → PENDING.
        d = _decision(
            did=50, hours_ago=2.0,
            action_taken="HOLD CASH → HOLD",
            reasoning="wait for the cash session before doing anything",
        )
        r = build_intent_followthrough([d], [], now=NOW)
        actionable = [i for i in r["intents"]
                      if i.get("bucket") == "ACTIONABLE"]
        assert actionable
        assert all(i["status"] == "PENDING" for i in actionable)
        assert r["n_pending"] >= 1
        # PENDING-only desk → NO_RESOLVED verdict (rate not yet decided).
        assert r["verdict"] == "NO_RESOLVED"

    def test_stale_intent_no_match_is_abandoned_when_outside_eval_window(self):
        # Intent 20h ago, no matching trade. eval_window=12h →
        # intent is past its window → ABANDONED.
        d = _decision(
            did=60, hours_ago=20.0,
            action_taken="HOLD NVDA → HOLD",
            reasoning="ready to add NVDA on a fresh print",
        )
        r = build_intent_followthrough(
            [d], [], now=NOW,
            window_hours=48.0,  # extend so the 20h-old decision is in scope
        )
        actionable = [i for i in r["intents"]
                      if i.get("bucket") == "ACTIONABLE"]
        assert actionable
        assert all(i["status"] == "ABANDONED" for i in actionable)
        assert r["n_abandoned"] >= 1

    def test_trade_outside_eval_window_does_not_count(self):
        # Intent 20h ago, matching trade 18h ago (so trade is 2h after
        # intent — that's inside the 12h eval window for the intent).
        # vs the same intent with a trade that's 14h after (outside).
        d1 = _decision(
            did=70, hours_ago=20.0,
            action_taken="HOLD NVDA → HOLD",
            reasoning="ready to add NVDA on the next print",
        )
        t_inside = _trade(tid=70, hours_ago=18.0, ticker="NVDA", action="BUY")
        r_inside = build_intent_followthrough(
            [d1], [t_inside], now=NOW, window_hours=48.0)
        actionable_inside = [i for i in r_inside["intents"]
                             if i.get("bucket") == "ACTIONABLE"]
        assert any(i["status"] == "FOLLOWED" for i in actionable_inside), (
            "trade 2h after intent should be FOLLOWED"
        )

        d2 = _decision(
            did=71, hours_ago=20.0,
            action_taken="HOLD NVDA → HOLD",
            reasoning="ready to add NVDA on the next print",
        )
        t_outside = _trade(tid=71, hours_ago=5.0, ticker="NVDA", action="BUY")
        # Trade is 15h after the intent — past the default 12h eval window.
        r_outside = build_intent_followthrough(
            [d2], [t_outside], now=NOW, window_hours=48.0)
        actionable_outside = [i for i in r_outside["intents"]
                              if i.get("bucket") == "ACTIONABLE"]
        assert all(i["status"] != "FOLLOWED" for i in actionable_outside), (
            "trade 15h after intent must NOT be credited (past eval_window)"
        )


# ─── verdict ladder ────────────────────────────────────────────────────

class TestVerdictLadder:
    def _build_n_intents(self, n_followed, n_abandoned):
        """Build a decision set with the requested mix.

        FOLLOWED intents pair with matching BUY trades after them; ABANDONED
        intents are old (>eval_window) with no matching trade. Each intent
        uses a unique ticker so dedup does not collapse them.
        """
        decisions = []
        trades = []
        unique_tickers = [
            "NVDA", "AMD", "MU", "AMAT", "LRCX", "KLAC", "TSM",
            "ASML", "MRVL", "QQQ", "SPY", "TQQQ",
        ]
        ti = 0
        for i in range(n_followed):
            tk = unique_tickers[ti % len(unique_tickers)]
            ti += 1
            d = _decision(
                did=100 + i, hours_ago=4.0,
                action_taken=f"HOLD {tk} → HOLD",
                reasoning=f"ready to add {tk} on the next print",
            )
            decisions.append(d)
            trades.append(_trade(tid=500 + i, hours_ago=2.0, ticker=tk, action="BUY"))
        for j in range(n_abandoned):
            tk = unique_tickers[ti % len(unique_tickers)]
            ti += 1
            # 20h ago decision, no matching trade → ABANDONED.
            d = _decision(
                did=200 + j, hours_ago=20.0,
                action_taken=f"HOLD {tk} → HOLD",
                reasoning=f"ready to add {tk} on a confirmation print",
            )
            decisions.append(d)
        return decisions, trades

    def test_disciplined_verdict_at_or_above_floor(self):
        # 4 followed, 1 abandoned → 80% rate, ≥ 66% floor → DISCIPLINED.
        decs, trs = self._build_n_intents(4, 1)
        r = build_intent_followthrough(
            decs, trs, now=NOW, window_hours=48.0)
        assert r["verdict"] == "DISCIPLINED", (
            r["headline"], r["n_followed"], r["n_abandoned"]
        )
        assert r["followthrough_rate"] >= DEFAULT_DISCIPLINE_FLOOR

    def test_drifting_verdict_in_middle_band(self):
        # 2 followed, 3 abandoned → 40% rate (between 33% and 66%) → DRIFTING.
        decs, trs = self._build_n_intents(2, 3)
        r = build_intent_followthrough(
            decs, trs, now=NOW, window_hours=48.0)
        assert r["verdict"] == "DRIFTING", (
            r["headline"], r["n_followed"], r["n_abandoned"]
        )

    def test_abandoned_verdict_requires_minimum_sample(self):
        # 0 followed, 3 abandoned → 0% rate, n_abandoned ≥ 3 → ABANDONED.
        decs, trs = self._build_n_intents(0, 3)
        r = build_intent_followthrough(
            decs, trs, now=NOW, window_hours=48.0)
        assert r["verdict"] == "ABANDONED", (
            r["headline"], r["n_followed"], r["n_abandoned"]
        )
        # 0 followed, 2 abandoned → fails sample-size guard → DRIFTING.
        decs, trs = self._build_n_intents(0, 2)
        r = build_intent_followthrough(
            decs, trs, now=NOW, window_hours=48.0,
            abandoned_min_n=3,
        )
        assert r["verdict"] == "DRIFTING", (
            r["headline"], r["n_followed"], r["n_abandoned"]
        )

    def test_is_followthrough_abandoned_helper(self):
        decs, trs = self._build_n_intents(0, 3)
        r = build_intent_followthrough(
            decs, trs, now=NOW, window_hours=48.0)
        assert is_followthrough_abandoned(r) is True

        decs, trs = self._build_n_intents(4, 1)
        r = build_intent_followthrough(
            decs, trs, now=NOW, window_hours=48.0)
        assert is_followthrough_abandoned(r) is False

        assert is_followthrough_abandoned(None) is False
        assert is_followthrough_abandoned({"verdict": "OK"}) is False


# ─── abstention bucket ─────────────────────────────────────────────────

class TestAbstentionBucket:
    def test_preserve_for_with_followup_buy_is_deployed(self):
        d = _decision(
            did=300, hours_ago=4.0,
            action_taken="HOLD CASH → HOLD",
            reasoning=(
                "preserve cash for tomorrow's open — MRVL earnings setup "
                "is the next catalyst"
            ),
        )
        t = _trade(tid=300, hours_ago=2.0, ticker="MRVL", action="BUY")
        r = build_intent_followthrough([d], [t], now=NOW)
        abstention_rows = [i for i in r["intents"]
                           if i.get("bucket") == "ABSTENTION"]
        assert abstention_rows
        deployed = [i for i in abstention_rows if i["status"] == "DEPLOYED"]
        assert deployed
        assert r["abstention"]["n_preserve_deployed"] >= 1

    def test_preserve_for_no_buy_inside_window_is_preserved(self):
        d = _decision(
            did=310, hours_ago=2.0,
            action_taken="HOLD CASH → HOLD",
            reasoning="preserve dry powder for the cash session",
        )
        r = build_intent_followthrough([d], [], now=NOW)
        assert r["abstention"]["n_preserve_active"] >= 1
        assert r["abstention"]["n_preserve_deployed"] == 0

    def test_preserve_for_past_window_no_buy_is_dead(self):
        # 20h-old preserve intent with no BUY → DEPLOYED_NEVER.
        d = _decision(
            did=320, hours_ago=20.0,
            action_taken="HOLD CASH → HOLD",
            reasoning="preserve cash for the next catalyst",
        )
        r = build_intent_followthrough(
            [d], [], now=NOW, window_hours=48.0)
        assert r["abstention"]["n_preserve_dead"] >= 1
        assert r["abstention"]["n_preserve_active"] == 0
        assert r["abstention"]["n_preserve_deployed"] == 0

    def test_too_early_to_buy_with_subsequent_buy_breaks_restraint(self):
        d = _decision(
            did=330, hours_ago=4.0,
            action_taken="HOLD NVDA → HOLD",
            reasoning="premature to add NVDA before the earnings print",
        )
        # Bot buys NVDA anyway 2h later — broke its own restraint.
        t = _trade(tid=330, hours_ago=2.0, ticker="NVDA", action="BUY")
        r = build_intent_followthrough([d], [t], now=NOW)
        abstention_rows = [i for i in r["intents"]
                           if i.get("bucket") == "ABSTENTION"]
        assert abstention_rows
        broken = [i for i in abstention_rows
                  if i["status"] == "BROKE_RESTRAINT"]
        assert broken, abstention_rows
        assert r["abstention"]["n_restraint_broken"] >= 1

    def test_too_early_to_buy_no_buy_is_restrained(self):
        d = _decision(
            did=340, hours_ago=4.0,
            action_taken="HOLD NVDA → HOLD",
            reasoning="too early to add NVDA before tomorrow's print",
        )
        r = build_intent_followthrough([d], [], now=NOW)
        abstention_rows = [i for i in r["intents"]
                           if i.get("bucket") == "ABSTENTION"]
        assert abstention_rows
        restrained = [i for i in abstention_rows
                      if i["status"] == "RESTRAINED"]
        assert restrained
        assert r["abstention"]["n_restraint_held"] >= 1


# ─── echo + invariants ─────────────────────────────────────────────────

class TestEchoAndInvariants:
    def test_returns_echo_block(self):
        r = build_intent_followthrough([], [], now=NOW)
        assert r["window_hours"] == 24.0
        assert r["stale_hours"] == 12.0
        assert r["eval_window_hours"] == DEFAULT_EVAL_WINDOW_HOURS
        assert r["discipline_floor"] == DEFAULT_DISCIPLINE_FLOOR
        assert r["drifting_floor"] == DEFAULT_DRIFTING_FLOOR
        assert r["abandoned_min_n"] == DEFAULT_ABANDONED_MIN_N
        # as_of mirrors the now we passed in.
        assert r["as_of"] == NOW.isoformat()

    def test_intent_text_is_passed_verbatim(self):
        # SSOT discipline: intent.text is the verbatim snippet from the
        # base extractor, never paraphrased by this module.
        d = _decision(
            did=400, hours_ago=2.0,
            action_taken="HOLD NVDA → HOLD",
            reasoning="ready to trim NVDA on a bounce above 225",
        )
        r = build_intent_followthrough([d], [], now=NOW)
        actionable = [i for i in r["intents"]
                      if i.get("bucket") == "ACTIONABLE"]
        assert actionable
        # The verbatim snippet from the matched intent pattern must be
        # present in the intent's text field — no paraphrase.
        assert any("trim" in (i.get("text") or "").lower()
                   for i in actionable)

    def test_outside_window_decisions_excluded(self):
        # 100h-old decision with window_hours=24 → excluded entirely.
        d = _decision(
            did=500, hours_ago=100.0,
            action_taken="HOLD NVDA → HOLD",
            reasoning="ready to add NVDA",
        )
        r = build_intent_followthrough([d], [], now=NOW, window_hours=24.0)
        assert r["verdict"] == "NO_DATA"
        assert r["n_intents"] == 0

    def test_json_round_trip(self):
        # The output must be JSON-serializable (the Flask endpoint will
        # jsonify it). All datetime values must be ISO strings, not
        # datetime objects.
        d = _decision(
            did=600, hours_ago=2.0,
            action_taken="HOLD NVDA → HOLD",
            reasoning="ready to add NVDA on the next print",
        )
        t = _trade(tid=600, hours_ago=1.0, ticker="NVDA", action="BUY")
        r = build_intent_followthrough([d], [t], now=NOW)
        # If anything is a datetime() it will raise on json.dumps.
        s = json.dumps(r)
        assert "FOLLOWED" in s or "PENDING" in s

    def test_decision_with_json_envelope_reasoning(self):
        # Live reasoning rows are JSON envelopes — the inner
        # decision.reasoning is the prose. Verify the extractor handles
        # this via the SSOT path.
        env = json.dumps({
            "decision": {
                "action": "HOLD",
                "ticker": "NVDA",
                "reasoning": "ready to add NVDA on a fresh print",
            }
        })
        d = {
            "id": 700,
            "timestamp": (NOW - timedelta(hours=2.0)).isoformat(),
            "action_taken": "HOLD NVDA → HOLD",
            "reasoning": env,
        }
        t = _trade(tid=700, hours_ago=1.0, ticker="NVDA", action="BUY")
        r = build_intent_followthrough([d], [t], now=NOW)
        followed = [i for i in r["intents"] if i["status"] == "FOLLOWED"]
        assert followed, r["intents"]

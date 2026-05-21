"""Tests for analytics/thesis_keyword_lift.py — open-vocabulary keyword lift.

The lift formula and the STABLE gate are load-bearing — a regression where
the verdict fires on a single one-sided keyword, where stopwords leak into
the ranking, where a wash counts as a win or loss, where lift uses
ratio-instead-of-pp math, or where the dominant-keyword tie-break is
non-deterministic all fail an assertion here.

Mirrors the test_winner_autopsy / test_loser_autopsy structure — same ledger
helpers, same hand-computed arithmetic, same boundary discipline.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.thesis_keyword_lift import (
    DEFAULT_MIN_KW_OCCURRENCES,
    MIN_TOKEN_LEN,
    STABLE_MIN_PER_SIDE,
    _tokenize,
    build_thesis_keyword_lift,
)

_BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _day(offset: int) -> str:
    return (_BASE + timedelta(days=offset)).isoformat()


def _rt(tid, ticker, buy_day, sell_day, qty, buy_px, sell_px,
        entry_reason="", exit_reason=""):
    """A buy+sell pair build_round_trips folds into one closed round-trip."""
    return [
        {"id": tid, "timestamp": _day(buy_day), "ticker": ticker,
         "action": "BUY", "qty": qty, "price": buy_px,
         "value": qty * buy_px, "strike": None, "expiry": None,
         "option_type": None, "reason": entry_reason},
        {"id": tid + 1, "timestamp": _day(sell_day), "ticker": ticker,
         "action": "SELL", "qty": qty, "price": sell_px,
         "value": qty * sell_px, "strike": None, "expiry": None,
         "option_type": None, "reason": exit_reason},
    ]


def _ledger(specs):
    """specs: (ticker, buy_px, sell_px, hold_days, entry_reason, exit_reason).
    Each becomes its own round-trip on a strictly increasing, disjoint
    window (qty fixed at 10) so build_round_trips closes each independently.
    """
    trades, tid, day = [], 1, 0
    for ticker, bpx, spx, hold, er, xr in specs:
        trades += _rt(tid, ticker, day, day + hold, 10, bpx, spx, er, xr)
        tid += 2
        day += hold + 1
    return trades


# ───────────────────────── _tokenize ─────────────────────────────────────

class TestTokenize:
    def test_empty_or_none_input_returns_empty_set(self):
        assert _tokenize(None) == set()
        assert _tokenize("") == set()
        assert _tokenize("   ") == set()

    def test_non_string_input_does_not_raise(self):
        # Mirrors winner_autopsy's defensive contract.
        for bad in (123, 4.5, [], {}, object()):
            assert _tokenize(bad) == set()

    def test_lowercases_and_splits_words(self):
        out = _tokenize("Earnings BEAT raised guidance")
        assert "earnings" in out
        assert "beat" in out
        assert "raised" in out
        assert "guidance" in out
        # No capitalised duplicates.
        assert "EARNINGS" not in out
        assert "BEAT" not in out

    def test_strips_punctuation(self):
        out = _tokenize("Q1 earnings beat: revenue, EPS, guidance!")
        assert "earnings" in out
        assert "beat" in out
        assert "revenue" in out
        assert "eps" in out
        assert "guidance" in out
        # Punctuation never produced a token.
        for tok in out:
            assert "," not in tok and "!" not in tok and ":" not in tok

    def test_drops_short_tokens(self):
        out = _tokenize("RSI MACD MA MV X")
        # MIN_TOKEN_LEN=3 → "rsi" / "macd" survive; "ma", "mv", "x" don't.
        assert "rsi" in out
        assert "macd" in out
        assert "ma" not in out
        assert "mv" not in out
        assert "x" not in out
        # Sanity-check the constant we depend on.
        assert MIN_TOKEN_LEN == 3

    def test_drops_stopwords(self):
        # These should all be on the stopword list (the open_reason
        # boilerplate-noise filter).
        out = _tokenize("this trade will hold the position")
        for sw in ("this", "trade", "will", "hold", "the", "position"):
            assert sw not in out, f"stopword '{sw}' leaked through"

    def test_drops_pure_numeric_tokens(self):
        out = _tokenize("rsi60 2026 guidance 80 billion buyback")
        # "rsi60" is alphanumeric and ≥ MIN_TOKEN_LEN so it survives;
        # "2026" and "80" are pure-numeric and must be dropped.
        assert "rsi60" in out
        assert "guidance" in out
        assert "buyback" in out
        assert "billion" in out
        assert "2026" not in out
        assert "80" not in out

    def test_deduplicates_within_a_single_reason(self):
        # An analyst writing 'earnings earnings earnings' shouldn't get
        # triple credit — tokenize returns a SET.
        out = _tokenize("earnings earnings earnings guidance")
        # Multiset would be {earnings: 3, guidance: 1}; set is {earnings, guidance}.
        assert out == {"earnings", "guidance"}


# ──────────────────── empty / one-sided pools ──────────────────────────

class TestEmptyAndOneSided:
    def test_empty_trades_returns_no_data(self):
        rep = build_thesis_keyword_lift([])
        assert rep["state"] == "NO_DATA"
        assert rep["verdict"] is None
        assert rep["n_round_trips"] == 0
        assert rep["n_winners"] == 0
        assert rep["n_losers"] == 0
        assert rep["top_winning_keywords"] == []
        assert rep["top_losing_keywords"] == []
        assert "no thesis keywords" in rep["headline"].lower()

    def test_non_list_input_does_not_raise(self):
        # Mirrors event_threads / winner_autopsy defensive contract.
        for bad in (None, 0, "string", {"not": "a list"}, 12.3):
            rep = build_thesis_keyword_lift(bad)  # type: ignore[arg-type]
            assert rep["state"] == "NO_DATA"
            assert rep["top_winning_keywords"] == []

    def test_only_winners_returns_no_losses(self):
        # 3 wins, 0 losses — every keyword is a "winning pattern by
        # definition" so the verdict gate refuses to fire.
        trades = _ledger([
            ("AAA", 10.0, 12.0, 1, "earnings beat raised guidance", "took profit"),
            ("BBB", 20.0, 25.0, 2, "earnings beat strong revenue", "took profit"),
            ("CCC", 30.0, 33.0, 1, "earnings beat fresh catalyst", "took profit"),
        ])
        rep = build_thesis_keyword_lift(trades)
        assert rep["state"] == "NO_LOSSES"
        assert rep["verdict"] is None
        assert rep["n_winners"] == 3
        assert rep["n_losers"] == 0
        assert "no losing round-trips" in rep["headline"].lower()

    def test_only_losers_returns_no_wins(self):
        trades = _ledger([
            ("AAA", 10.0, 8.0, 1, "oversold bounce reversal play", "stopped"),
            ("BBB", 20.0, 16.0, 2, "oversold knife catch", "stopped"),
            ("CCC", 30.0, 24.0, 1, "oversold counter-trend", "stopped"),
        ])
        rep = build_thesis_keyword_lift(trades)
        assert rep["state"] == "NO_WINS"
        assert rep["verdict"] is None
        assert rep["n_winners"] == 0
        assert rep["n_losers"] == 3
        assert "no winning round-trips" in rep["headline"].lower()

    def test_wash_round_trip_excluded_from_both_sides(self):
        # A trade closed at exactly entry price ($0 PnL) is not a win
        # nor a loss — winner_autopsy / loser_autopsy / trade_asymmetry
        # all skip washes. Verify this builder agrees.
        trades = _ledger([
            ("AAA", 10.0, 12.0, 1, "earnings beat", "took profit"),
            ("BBB", 10.0, 10.0, 1, "stale catalyst", "wash"),  # wash
            ("CCC", 10.0, 8.0, 1, "oversold reversal", "stopped"),
        ])
        rep = build_thesis_keyword_lift(trades)
        assert rep["n_round_trips"] == 3
        assert rep["n_winners"] == 1
        assert rep["n_losers"] == 1
        assert rep["n_decisive"] == 2
        # 'stale' must NOT show up in either side (the wash gave it no signal).
        all_kws = {r["keyword"] for r in rep["top_winning_keywords"]}
        all_kws |= {r["keyword"] for r in rep["top_losing_keywords"]}
        assert "stale" not in all_kws


# ───────────────────── lift formula correctness ────────────────────────

class TestLiftFormula:
    def test_baseline_win_rate_uses_decisive_only(self):
        # 2W / 2L / 1 wash → baseline 50% (washes excluded).
        trades = _ledger([
            ("AAA", 10.0, 12.0, 1, "alpha", "took profit"),
            ("BBB", 10.0, 12.0, 1, "alpha", "took profit"),
            ("CCC", 10.0, 8.0, 1, "beta", "stopped"),
            ("DDD", 10.0, 8.0, 1, "beta", "stopped"),
            ("EEE", 10.0, 10.0, 1, "wash", "flat"),  # wash
        ])
        rep = build_thesis_keyword_lift(
            trades, min_kw_occurrences=2, top_n=10,
        )
        assert rep["baseline_win_rate_pct"] == 50.0
        # 'alpha' appears in 2W / 0L → win_rate 100%, lift +50pp.
        winning = {r["keyword"]: r for r in rep["top_winning_keywords"]}
        assert "alpha" in winning
        assert winning["alpha"]["n_winners"] == 2
        assert winning["alpha"]["n_losers"] == 0
        assert winning["alpha"]["lift_pp"] == 50.0
        # 'beta' appears in 0W / 2L → win_rate 0%, lift -50pp.
        losing = {r["keyword"]: r for r in rep["top_losing_keywords"]}
        assert "beta" in losing
        assert losing["beta"]["n_winners"] == 0
        assert losing["beta"]["n_losers"] == 2
        assert losing["beta"]["lift_pp"] == -50.0

    def test_lift_is_percentage_points_not_ratio(self):
        # In a 50/50 baseline a keyword with 100% win rate gets +50pp
        # (not 2x). A ratio formula would be undefined when n_losers=0.
        trades = _ledger([
            ("AAA", 10.0, 12.0, 1, "guidance", "took profit"),
            ("BBB", 10.0, 12.0, 1, "guidance", "took profit"),
            ("CCC", 10.0, 8.0, 1, "noise", "stopped"),
            ("DDD", 10.0, 8.0, 1, "noise", "stopped"),
        ])
        rep = build_thesis_keyword_lift(trades, min_kw_occurrences=2)
        winning = {r["keyword"]: r for r in rep["top_winning_keywords"]}
        # 50% baseline; 100% win rate on 'guidance' → 50pp lift, bounded.
        assert winning["guidance"]["lift_pp"] == 50.0
        assert winning["guidance"]["lift_pp"] <= 100.0

    def test_min_kw_occurrences_threshold(self):
        # A keyword that appears only once is filtered (default is 3;
        # we explicitly override to 2 here to keep the ledger small).
        trades = _ledger([
            ("AAA", 10.0, 12.0, 1, "rare guidance",  "took profit"),
            ("BBB", 10.0, 12.0, 1, "rare guidance",  "took profit"),
            ("CCC", 10.0, 12.0, 1, "common pattern", "took profit"),
            ("DDD", 10.0, 12.0, 1, "common pattern", "took profit"),
            ("EEE", 10.0, 8.0, 1, "common pattern",  "stopped"),
        ])
        rep = build_thesis_keyword_lift(trades, min_kw_occurrences=2)
        kws_present = ({r["keyword"] for r in rep["top_winning_keywords"]} |
                       {r["keyword"] for r in rep["top_losing_keywords"]})
        # 'rare' / 'guidance' (n_total=2) survive the floor.
        assert "rare" in kws_present or "guidance" in kws_present
        # Now raise the floor; sparse ones drop out.
        rep_strict = build_thesis_keyword_lift(trades, min_kw_occurrences=3)
        strict_kws = ({r["keyword"] for r in rep_strict["top_winning_keywords"]} |
                      {r["keyword"] for r in rep_strict["top_losing_keywords"]})
        assert "rare" not in strict_kws
        assert "guidance" not in strict_kws
        # 'common' / 'pattern' (n_total=3) still survive.
        assert "common" in strict_kws
        assert "pattern" in strict_kws

    def test_default_min_kw_occurrences_is_three(self):
        # Sanity-check the documented default — a single edit that flips
        # this should be deliberate.
        assert DEFAULT_MIN_KW_OCCURRENCES == 3


# ─────────────────────── verdict / STABLE gate ──────────────────────────

class TestVerdictGate:
    def test_stable_min_per_side_constant_is_four(self):
        assert STABLE_MIN_PER_SIDE == 4

    def test_emerging_below_stable_threshold(self):
        # 3W / 4L — winners side under-floor.
        trades = _ledger([
            ("A1", 10.0, 12.0, 1, "earnings beat",     "took profit"),
            ("A2", 10.0, 12.0, 1, "earnings beat",     "took profit"),
            ("A3", 10.0, 12.0, 1, "earnings beat",     "took profit"),
            ("L1", 10.0, 8.0, 1,  "oversold reversal", "stopped"),
            ("L2", 10.0, 8.0, 1,  "oversold reversal", "stopped"),
            ("L3", 10.0, 8.0, 1,  "oversold reversal", "stopped"),
            ("L4", 10.0, 8.0, 1,  "oversold reversal", "stopped"),
        ])
        rep = build_thesis_keyword_lift(trades, min_kw_occurrences=2)
        assert rep["state"] == "EMERGING"
        assert rep["verdict"] is None
        assert "emerging" in rep["headline"].lower()

    def test_stable_emits_verdict_at_threshold(self):
        # Exactly 4W / 4L — both sides at floor, STABLE.
        trades = _ledger([
            ("A1", 10.0, 12.0, 1, "earnings beat",  "took profit"),
            ("A2", 10.0, 12.0, 1, "earnings beat",  "took profit"),
            ("A3", 10.0, 12.0, 1, "earnings beat",  "took profit"),
            ("A4", 10.0, 12.0, 1, "earnings beat",  "took profit"),
            ("L1", 10.0, 8.0, 1,  "oversold play",  "stopped"),
            ("L2", 10.0, 8.0, 1,  "oversold play",  "stopped"),
            ("L3", 10.0, 8.0, 1,  "oversold play",  "stopped"),
            ("L4", 10.0, 8.0, 1,  "oversold play",  "stopped"),
        ])
        rep = build_thesis_keyword_lift(trades)
        assert rep["state"] == "STABLE"
        assert rep["verdict"] in ("earnings", "beat")
        # The two winning keywords have identical lift; tie-break is
        # n_total then alphabetical → "beat" sorts before "earnings".
        assert rep["verdict"] == "beat"
        # Verdict surfaces a positive-lift keyword.
        wins_by_kw = {r["keyword"]: r for r in rep["top_winning_keywords"]}
        assert wins_by_kw[rep["verdict"]]["lift_pp"] > 0

    def test_stable_pool_with_no_positive_lift_returns_no_verdict(self):
        # Same keyword on both sides — lift = 0 — verdict withheld.
        trades = _ledger([
            ("A1", 10.0, 12.0, 1, "neutral catalyst", "took profit"),
            ("A2", 10.0, 12.0, 1, "neutral catalyst", "took profit"),
            ("A3", 10.0, 12.0, 1, "neutral catalyst", "took profit"),
            ("A4", 10.0, 12.0, 1, "neutral catalyst", "took profit"),
            ("L1", 10.0, 8.0, 1,  "neutral catalyst", "stopped"),
            ("L2", 10.0, 8.0, 1,  "neutral catalyst", "stopped"),
            ("L3", 10.0, 8.0, 1,  "neutral catalyst", "stopped"),
            ("L4", 10.0, 8.0, 1,  "neutral catalyst", "stopped"),
        ])
        rep = build_thesis_keyword_lift(trades)
        assert rep["state"] == "STABLE"
        # 50% baseline, 50% win rate on 'neutral'/'catalyst' → 0pp lift
        # → no winning verdict.
        assert rep["verdict"] is None
        # Headline should say no keyword cleared the baseline by a
        # positive margin.
        assert "baseline" in rep["headline"].lower()


# ────────────────────── ranking / determinism ──────────────────────────

class TestRankingDeterminism:
    def test_tied_lift_breaks_by_sample_size_then_alphabetical(self):
        # Two keywords with identical 50pp lift — one appears in 4 trips,
        # the other in 2. Sample size wins; ties beyond that go
        # alphabetical so the card order is stable across runs.
        trades = _ledger([
            ("A1", 10.0, 12.0, 1, "common earnings", "took profit"),
            ("A2", 10.0, 12.0, 1, "common earnings", "took profit"),
            ("A3", 10.0, 12.0, 1, "common rare",     "took profit"),
            ("A4", 10.0, 12.0, 1, "common rare",     "took profit"),
            ("L1", 10.0, 8.0, 1,  "broken thesis",   "stopped"),
            ("L2", 10.0, 8.0, 1,  "broken thesis",   "stopped"),
            ("L3", 10.0, 8.0, 1,  "broken thesis",   "stopped"),
            ("L4", 10.0, 8.0, 1,  "broken thesis",   "stopped"),
        ])
        rep = build_thesis_keyword_lift(trades, min_kw_occurrences=2)
        # 'common' wins on every winning trip (4W/0L, n_total=4),
        # 'earnings' / 'rare' each only in 2W/0L, n_total=2.
        # Top winning keyword should be 'common'.
        assert rep["top_winning_keywords"][0]["keyword"] == "common"
        # And the second / third position should be deterministic
        # alphabetical between 'earnings' and 'rare'.
        runners = [r["keyword"] for r in rep["top_winning_keywords"][1:3]]
        assert runners == ["earnings", "rare"]

    def test_losing_ranking_orders_by_most_negative_lift_first(self):
        trades = _ledger([
            ("A1", 10.0, 12.0, 1, "mild winner",  "took profit"),
            ("A2", 10.0, 12.0, 1, "mild winner",  "took profit"),
            ("A3", 10.0, 12.0, 1, "mild winner",  "took profit"),
            ("A4", 10.0, 12.0, 1, "mild winner",  "took profit"),
            ("L1", 10.0, 8.0, 1,  "deep loser",   "stopped"),
            ("L2", 10.0, 8.0, 1,  "deep loser",   "stopped"),
            ("L3", 10.0, 8.0, 1,  "deep loser",   "stopped"),
            ("L4", 10.0, 8.0, 1,  "deep loser",   "stopped"),
        ])
        rep = build_thesis_keyword_lift(trades, min_kw_occurrences=2)
        top_losing = rep["top_losing_keywords"][0]
        # 'deep' and 'loser' both have -50pp lift; alphabetical tiebreak
        # puts 'deep' first.
        assert top_losing["keyword"] in ("deep", "loser")
        assert top_losing["lift_pp"] == -50.0
        # Sorted in ascending lift_pp.
        lifts = [r["lift_pp"] for r in rep["top_losing_keywords"]]
        assert lifts == sorted(lifts)

    def test_top_n_caps_each_ranking_independently(self):
        # 5 distinct winning keywords; top_n=2 truncates both lists to 2.
        winning_specs = [
            ("A1", 10.0, 12.0, 1, "kwa kwa kwa", "p"),
            ("A2", 10.0, 12.0, 1, "kwb kwb kwb", "p"),
            ("A3", 10.0, 12.0, 1, "kwc kwc kwc", "p"),
            ("A4", 10.0, 12.0, 1, "kwd kwd kwd", "p"),
            ("A5", 10.0, 12.0, 1, "kwe kwe kwe", "p"),
            ("A6", 10.0, 12.0, 1, "kwa kwb kwc kwd kwe", "p"),
            ("L1", 10.0, 8.0, 1,  "lossa lossa",  "stopped"),
            ("L2", 10.0, 8.0, 1,  "lossa lossa",  "stopped"),
        ]
        rep = build_thesis_keyword_lift(
            _ledger(winning_specs), top_n=2, min_kw_occurrences=2,
        )
        assert len(rep["top_winning_keywords"]) <= 2
        assert len(rep["top_losing_keywords"]) <= 2

    def test_top_n_zero_returns_empty_lists(self):
        # 3-char-min token filter means "kw" gets dropped; use a real
        # keyword the tokenizer keeps.
        trades = _ledger([
            ("A1", 10.0, 12.0, 1, "alpha", "p"),
            ("A2", 10.0, 12.0, 1, "alpha", "p"),
            ("L1", 10.0, 8.0, 1,  "alpha", "s"),
        ])
        rep = build_thesis_keyword_lift(trades, top_n=0, min_kw_occurrences=2)
        assert rep["top_winning_keywords"] == []
        assert rep["top_losing_keywords"] == []
        # But the underlying tally still exposes n_distinct_keywords.
        assert rep["n_distinct_keywords"] >= 1


# ──────────────────── envelope / response shape ─────────────────────────

class TestResponseShape:
    def test_envelope_keys_present(self):
        # The dashboard / chat enrichment binds these field names; a
        # rename here would silently blank the panel.
        rep = build_thesis_keyword_lift([])
        for key in (
            "as_of", "state", "verdict", "headline",
            "n_round_trips", "n_winners", "n_losers", "n_decisive",
            "baseline_win_rate_pct", "min_kw_occurrences",
            "stable_min_per_side", "n_distinct_keywords",
            "top_winning_keywords", "top_losing_keywords",
        ):
            assert key in rep, f"envelope key '{key}' missing"

    def test_as_of_uses_injected_now(self):
        custom_now = datetime(2026, 5, 21, 3, 0, 0, tzinfo=timezone.utc)
        rep = build_thesis_keyword_lift([], now=custom_now)
        assert rep["as_of"].startswith("2026-05-21T03:00:00")

    def test_keyword_row_keys_present(self):
        trades = _ledger([
            ("A1", 10.0, 12.0, 1, "alpha alpha", "p"),
            ("A2", 10.0, 12.0, 1, "alpha alpha", "p"),
            ("L1", 10.0, 8.0, 1,  "beta",        "s"),
            ("L2", 10.0, 8.0, 1,  "beta",        "s"),
        ])
        rep = build_thesis_keyword_lift(trades, min_kw_occurrences=2)
        # Pick any non-empty row.
        rows = rep["top_winning_keywords"] + rep["top_losing_keywords"]
        assert rows  # there must be at least one rankable keyword
        sample = rows[0]
        for key in ("keyword", "n_winners", "n_losers", "n_total",
                    "win_rate_pct", "lift_pp"):
            assert key in sample, f"row key '{key}' missing"


# ─────────────────── entry-reason verbatim discipline ───────────────────

class TestEntryReasonVerbatim:
    def test_only_entry_reason_is_tokenised_not_exit_reason(self):
        # The exit reason ("stopped on knife catch") must NOT contribute
        # to the keyword tally — only the entry thesis does, since this
        # is a *thesis* keyword lift.
        trades = _ledger([
            ("A1", 10.0, 8.0, 1, "earnings beat",
             "stopped on knife catch"),
            ("A2", 10.0, 8.0, 1, "earnings beat",
             "stopped on knife catch"),
            ("A3", 10.0, 8.0, 1, "earnings beat",
             "stopped on knife catch"),
        ])
        rep = build_thesis_keyword_lift(trades, min_kw_occurrences=2)
        loss_kws = {r["keyword"] for r in rep["top_losing_keywords"]}
        # If exit_reason leaked in, 'knife' / 'catch' would show up as
        # 3-loss keywords.
        assert "knife" not in loss_kws
        assert "catch" not in loss_kws

    def test_blank_entry_reason_round_trip_contributes_no_keywords(self):
        # A round-trip with an empty entry reason still counts toward
        # n_winners / n_losers (the structural pool) but its missing
        # text adds nothing to either tally.
        trades = _ledger([
            ("A1", 10.0, 12.0, 1, "earnings beat", "p"),
            ("A2", 10.0, 12.0, 1, "earnings beat", "p"),
            ("A3", 10.0, 12.0, 1, "",              "p"),  # blank
            ("L1", 10.0, 8.0, 1,  "broken thesis", "s"),
            ("L2", 10.0, 8.0, 1,  "broken thesis", "s"),
        ])
        rep = build_thesis_keyword_lift(trades, min_kw_occurrences=2)
        # n_winners counts all 3 wins, including the blank-reason one.
        assert rep["n_winners"] == 3
        winning = {r["keyword"]: r for r in rep["top_winning_keywords"]}
        # 'earnings' n_winners is 2 — the blank-reason win didn't bump it.
        if "earnings" in winning:
            assert winning["earnings"]["n_winners"] == 2

    def test_first_entry_trade_carries_thesis_not_add_on(self):
        # If a round-trip has two BUY rows ("opening thesis" then "add"),
        # the FIRST BUY's reason is what builds the thesis. Add-on text
        # must not leak into the tally — same convention as
        # winner_autopsy._reason_for(pick_last=False).
        # Construct a two-buy / one-sell round trip manually.
        trades = [
            {"id": 1, "timestamp": _day(0), "ticker": "AAA",
             "action": "BUY", "qty": 10, "price": 10.0, "value": 100.0,
             "strike": None, "expiry": None, "option_type": None,
             "reason": "opening earnings beat"},
            {"id": 2, "timestamp": _day(1), "ticker": "AAA",
             "action": "BUY", "qty": 5, "price": 11.0, "value": 55.0,
             "strike": None, "expiry": None, "option_type": None,
             "reason": "addon adding more here"},
            {"id": 3, "timestamp": _day(2), "ticker": "AAA",
             "action": "SELL", "qty": 15, "price": 13.0, "value": 195.0,
             "strike": None, "expiry": None, "option_type": None,
             "reason": "p"},
            # Another winner so 'opening' clears min_kw_occurrences=2.
        ] + _rt(4, "BBB", 3, 4, 10, 10.0, 12.0, "opening earnings", "p") + (
            # Two losers so we can hit STABLE for verdict.
            _rt(6, "CCC", 5, 6, 10, 10.0, 8.0, "broken thesis", "s")
            + _rt(8, "DDD", 7, 8, 10, 10.0, 8.0, "broken thesis", "s")
        )
        rep = build_thesis_keyword_lift(trades, min_kw_occurrences=2)
        winning = {r["keyword"]: r for r in rep["top_winning_keywords"]}
        # 'opening' is on the first BUY of both winners → 2 wins.
        assert "opening" in winning
        assert winning["opening"]["n_winners"] == 2
        # 'addon' is ONLY on the add-on BUY; first-buy convention drops it.
        assert "addon" not in winning

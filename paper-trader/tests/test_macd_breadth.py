"""Tests for ``paper_trader.analytics.macd_breadth`` — verdict ladder,
share math, headline formatting, and degrade-safe envelopes.

The builder is the natural follow-up to the ``strategy._macd_live`` fix
(AGENTS.md pass #38 — the new ``"flat"`` label). It rolls per-name
MACD labels into a one-line market-structure headline Opus reads in
the prompt and an operator reads on the dashboard.

Every test asserts a specific output (verdict, count, share value, or
headline substring) — no "just runs without crash" coverage.
"""
from __future__ import annotations

import pytest

from paper_trader.analytics import macd_breadth


def _sigs(labels: list[str | None]) -> dict[str, dict]:
    """Convenience: build a quant_sigs dict where each ticker has the
    given MACD label. The other quant fields are irrelevant to the
    breadth builder and intentionally absent so a mis-key (the builder
    reading the wrong field by accident) would fail loudly."""
    return {f"T{i:02d}": {"MACD": lab} for i, lab in enumerate(labels)}


class TestVerdictLadder:
    def test_no_data_when_empty(self):
        out = macd_breadth.build_macd_breadth({})
        assert out["verdict"] == "NO_DATA"
        assert out["n"] == 0
        # The "no quant data this cycle" copy is part of the contract —
        # the dashboard / Discord renderer relies on it for the silent
        # case. Lock the substring so a future copy edit can't quietly
        # break the operator's no-data signal.
        assert "no quant data" in out["headline"]

    def test_no_data_when_none(self):
        # None is the legitimate "quant fetch errored" path — must
        # degrade to the same envelope as the empty-dict case.
        out = macd_breadth.build_macd_breadth(None)
        assert out["verdict"] == "NO_DATA"

    def test_bull_breadth_when_bullish_dominates(self):
        # 7 bullish, 2 bearish, 1 flat over 10 names. Bullish share
        # 0.70 vs bearish 0.20 → spread 0.50, well above the 0.10
        # dominance threshold → BULL_BREADTH.
        labels = (["bullish"] * 7) + (["bearish"] * 2) + ["flat"]
        out = macd_breadth.build_macd_breadth(_sigs(labels))
        assert out["verdict"] == "BULL_BREADTH"
        assert out["counts"]["bullish"] == 7
        assert out["counts"]["bearish"] == 2

    def test_bear_breadth_when_bearish_dominates(self):
        labels = (["bullish"] * 2) + (["bearish"] * 7) + ["flat"]
        out = macd_breadth.build_macd_breadth(_sigs(labels))
        assert out["verdict"] == "BEAR_BREADTH"

    def test_flat_tape_when_majority_flat(self):
        # 6/10 flat ⇒ flat share 0.60 ≥ 0.50 majority threshold ⇒
        # FLAT_TAPE wins over both BULL and BEAR even if one side is
        # slightly ahead among the non-flat names.
        labels = (["flat"] * 6) + (["bullish"] * 3) + ["bearish"]
        out = macd_breadth.build_macd_breadth(_sigs(labels))
        assert out["verdict"] == "FLAT_TAPE"

    def test_mixed_when_close_split(self):
        # 5 bullish, 4 bearish, 1 flat ⇒ spread 0.10 (not > 0.10),
        # flat share 0.10 (not majority) ⇒ MIXED.
        labels = (["bullish"] * 5) + (["bearish"] * 4) + ["flat"]
        out = macd_breadth.build_macd_breadth(_sigs(labels))
        assert out["verdict"] == "MIXED"

    def test_dominance_threshold_is_strict_gt(self):
        # Exactly at the 0.10 threshold (1-bull, 0-bear over 10 with
        # 9 flats yields share 0.10 vs 0 — that's a 0.10 spread,
        # NOT > 0.10) must NOT tip to BULL — verdict stays FLAT_TAPE
        # because the flat majority dominates anyway.
        # Build a clean ≤-threshold case using non-majority flats:
        # 1 bull, 0 bear, 9 unknown → bull share 0.10, bear share 0,
        # flat share 0 → MIXED (spread 0.10, not > 0.10).
        labels = ["bullish"] + ([None] * 9)
        out = macd_breadth.build_macd_breadth(_sigs(labels))
        assert out["verdict"] == "MIXED"


class TestCounts:
    def test_unknown_label_counted_as_unknown(self):
        # An unrecognised label string is not bullish/bearish/flat —
        # it must count toward `unknown`. Otherwise a typo'd label
        # would silently drop the row from the breadth total and the
        # operator wouldn't see it.
        out = macd_breadth.build_macd_breadth(
            _sigs(["bullish", "neutral?", "flat"])
        )
        assert out["counts"]["unknown"] == 1
        assert out["counts"]["bullish"] == 1
        assert out["counts"]["flat"] == 1

    def test_none_label_counted_as_unknown(self):
        out = macd_breadth.build_macd_breadth(_sigs([None, None, "flat"]))
        assert out["counts"]["unknown"] == 2
        assert out["counts"]["flat"] == 1

    def test_missing_MACD_key_counted_as_unknown(self):
        # A row that has no MACD key (e.g. older snapshot) must NOT
        # crash — counts as unknown.
        sigs = {"NVDA": {}}  # no MACD key at all
        out = macd_breadth.build_macd_breadth(sigs)
        assert out["counts"]["unknown"] == 1

    def test_non_dict_row_counted_as_unknown(self):
        # A malformed quant row (e.g. None passed through by a fault
        # in get_quant_signals_live) must not crash the builder.
        sigs = {"NVDA": None, "AMD": {"MACD": "bullish"}}
        out = macd_breadth.build_macd_breadth(sigs)
        assert out["counts"]["unknown"] == 1
        assert out["counts"]["bullish"] == 1

    def test_counts_sum_equals_n(self):
        # The sum of all per-label counts must equal the input
        # dictionary length. A drift here would silently misreport
        # breadth in the prompt headline.
        labels = ["bullish", "bearish", "flat", None, "bullish"]
        out = macd_breadth.build_macd_breadth(_sigs(labels))
        assert sum(out["counts"].values()) == len(labels)
        assert out["n"] == len(labels)


class TestShares:
    def test_shares_are_normalized(self):
        # Each share must equal count/n. Locks the division logic
        # (and the 3dp rounding) so a /n-1 slip would fail loudly.
        labels = (["bullish"] * 4) + (["bearish"] * 1)
        out = macd_breadth.build_macd_breadth(_sigs(labels))
        assert out["shares"]["bullish"] == pytest.approx(0.8)
        assert out["shares"]["bearish"] == pytest.approx(0.2)
        assert out["shares"]["flat"] == 0.0

    def test_shares_zero_when_no_data(self):
        out = macd_breadth.build_macd_breadth({})
        # Every share is exactly 0.0 on the no-data envelope — the
        # JSON shape stays stable so dashboard JS can read it without
        # a None-guard.
        for v in out["shares"].values():
            assert v == 0.0


class TestHeadline:
    def test_bull_headline_names_split(self):
        labels = (["bullish"] * 7) + (["bearish"] * 2) + ["flat"]
        out = macd_breadth.build_macd_breadth(_sigs(labels))
        # The headline carries the bull and bear counts verbatim —
        # locks the BULL template against a future copy edit that
        # would drop them and silently degrade the operator signal.
        assert "7/10" in out["headline"]  # 7 bullish of 10 total
        assert "2 bearish" in out["headline"]

    def test_flat_tape_headline_names_flat_count(self):
        labels = (["flat"] * 6) + (["bullish"] * 3) + ["bearish"]
        out = macd_breadth.build_macd_breadth(_sigs(labels))
        assert "FLAT TAPE" in out["headline"]
        assert "6/10" in out["headline"]

    def test_headline_under_120_chars(self):
        # Lock the headline length so it fits the prompt MARKET
        # STRUCTURE line without wrapping. A 121-char headline here
        # would silently truncate in the Discord embed.
        labels = (["bullish"] * 20) + (["bearish"] * 10) + (["flat"] * 5)
        out = macd_breadth.build_macd_breadth(_sigs(labels))
        assert len(out["headline"]) <= 120


class TestRenderPromptLine:
    def test_empty_snap_returns_empty(self):
        # Silence-when-nothing-actionable: no data → no token in the
        # prompt. Mirrors the sibling analytics' silent-on-empty
        # contract so a healthy book / first-cycle stays clean.
        assert macd_breadth.render_prompt_line(None) == ""
        assert macd_breadth.render_prompt_line({}) == ""

    def test_no_data_verdict_returns_empty(self):
        snap = macd_breadth.build_macd_breadth({})
        assert macd_breadth.render_prompt_line(snap) == ""

    def test_non_dict_snap_returns_empty(self):
        # Degrade-safe: a non-dict in the snap slot must not raise.
        assert macd_breadth.render_prompt_line("oops") == ""
        assert macd_breadth.render_prompt_line(42) == ""

    def test_returns_headline_when_data(self):
        labels = (["bullish"] * 5) + (["bearish"] * 5)
        snap = macd_breadth.build_macd_breadth(_sigs(labels))
        line = macd_breadth.render_prompt_line(snap)
        assert line  # non-empty
        assert "MACD BREADTH" in line


class TestPromptWiring:
    """Pins that ``strategy._build_payload`` actually surfaces the
    breadth line in the rendered prompt. Without this the builder
    would be live-dark — Opus would never see the headline regardless
    of how good the breadth math is.

    Uses WATCHLIST[:5] tickers (LITE/LNOK/MUU/DRAM/SNDU) so
    ``_names_in_play`` keeps them in the rendered quant block. The
    test confirms the new MARKET STRUCTURE row appears in the
    TECHNICAL SIGNALS block."""

    def test_breadth_line_in_prompt_when_quant_signals_present(self):
        from paper_trader.strategy import _build_payload

        snap = {
            "cash": 1000.0, "open_value": 0.0,
            "total_value": 1000.0, "positions": [],
        }
        quant = {
            "LITE": {"MACD": "bullish"},
            "LNOK": {"MACD": "bullish"},
            "MUU": {"MACD": "flat"},
            "DRAM": {"MACD": "bearish"},
            "SNDU": {"MACD": "flat"},
        }
        prompt = _build_payload(
            snap, [], [], {}, {}, None, True, quant_signals=quant,
        )
        assert "MACD BREADTH" in prompt, (
            "Expected the MACD BREADTH headline in the rendered prompt — "
            "the analytics wiring in _build_payload is dropping it."
        )
        # And the line must be inside the TECHNICAL SIGNALS block, not
        # somewhere arbitrary — the trader's mental model is "I read
        # the technical signals section to understand momentum, and the
        # breadth headline tops it". Locks the placement.
        tech_block_start = prompt.find("TECHNICAL SIGNALS")
        breadth_pos = prompt.find("MACD BREADTH")
        # First sentiment section comes AFTER the TECHNICAL block —
        # use it as the end-of-block sentinel.
        sentiment_start = prompt.find("TICKER SENTIMENT")
        assert tech_block_start < breadth_pos < sentiment_start, (
            "MACD BREADTH must render inside the TECHNICAL SIGNALS block"
        )

    def test_breadth_line_omitted_when_no_quant_signals(self):
        # Silence-when-nothing-actionable: empty quant ⇒ no breadth
        # line in the prompt. A "MACD BREADTH: no quant data" headline
        # would just add a row of noise to a quiet prompt — the
        # render_prompt_line contract intentionally returns "" for
        # NO_DATA so the caller emits nothing.
        from paper_trader.strategy import _build_payload

        snap = {
            "cash": 1000.0, "open_value": 0.0,
            "total_value": 1000.0, "positions": [],
        }
        prompt = _build_payload(
            snap, [], [], {}, {}, None, True, quant_signals={},
        )
        assert "MACD BREADTH" not in prompt

"""analysis/claude_analyst.py — the heartbeat briefing payload builder.

This module had zero direct coverage. It runs every 5h on live data; a
formatting regression here corrupts the only long-form Discord briefing and
the briefing-derived training labels. These pin the three real bug classes
the source comments call out explicitly:

  * `_fmt_ticker` — a present-but-None ticker/price/pct must not raise mid
    briefing (the `or` guards exist because dict.get() only defaults on a
    *missing* key).
  * `_build_payload` — the article cap is 60, NOT 50: the caller prepends up
    to 2 synthetic snapshot rows (P&L, options) to a 50-item top list, so a
    [:50] cap silently truncates the last two *real* articles.
  * `analyze` — must return the sentinel placeholder (which heartbeat_worker
    detects and retries on) when the Claude CLI yields nothing, never None.
"""
from __future__ import annotations

from unittest.mock import patch

from analysis import claude_analyst


class TestFmtTicker:
    def test_none_ticker_does_not_raise(self):
        """A row carrying ticker=None must render '?' — f'{None:>12}' raises
        TypeError and would abort the whole briefing build."""
        line = claude_analyst._fmt_ticker(
            {"ticker": None, "price": 12.5, "pct_change": 1.0, "name": None}
        )
        assert "?" in line
        assert "None" not in line  # name=None must not render literally

    def test_none_price_and_pct_render_na(self):
        line = claude_analyst._fmt_ticker(
            {"ticker": "MU", "price": None, "pct_change": None, "name": "Micron"}
        )
        assert "N/A" in line
        assert "MU" in line
        assert "Micron" in line

    def test_valid_row_formats_numbers(self):
        line = claude_analyst._fmt_ticker(
            {"ticker": "NVDA", "price": 123.456, "pct_change": 2.5,
             "name": "NVIDIA"}
        )
        # price 2dp, pct signed
        assert "123.46" in line
        assert "+2.50%" in line

    def test_missing_keys_default(self):
        """Entirely empty dict — every field absent, not just None."""
        line = claude_analyst._fmt_ticker({})
        assert "?" in line and "N/A" in line


class TestBuildPayload:
    def _articles(self, n):
        # Each title must be genuinely distinct under the briefing's
        # order-independent near-dup collapse (ml.dedup, token-set Jaccard).
        # The original `f"headline {i}"` was NOT: the bare digit is a len-1
        # token dropped by `_MIN_TOKEN_LEN=2`, so all N normalized to the
        # identical token set {headline} and collapsed to one row — a latent
        # fixture defect (the helper's whole purpose is N *distinct* rows).
        # `alpha{i}`/`topic{i}` are alphanumeric len>=2 tokens kept verbatim,
        # so row i's set differs from row j's (J~0.43 < the 0.7 threshold).
        # Assertions and the cap-60 contract they pin are unchanged.
        return [
            {"title": f"headline alpha{i} bravo desk topic{i}", "source": "rss",
             "ai_score": 7.0, "summary": "body"}
            for i in range(n)
        ]

    def test_article_cap_is_60_not_50(self):
        """Regression pin: the cap must be 60. With 65 articles, line '60.'
        must be present and '61.' absent. A [:50] cap (the documented bug)
        would drop real articles whenever synthetic snapshot rows prepend."""
        payload = claude_analyst._build_payload(
            self._articles(65), {"macro": [], "equities": []}, []
        )
        assert "\n60. " in payload, "60th article missing — cap regressed below 60"
        assert "61. " not in payload, "more than 60 articles — cap not applied"

    def test_empty_articles_placeholder(self):
        payload = claude_analyst._build_payload(
            [], {"macro": [], "equities": []}, []
        )
        assert "(no high-relevance articles this cycle)" in payload

    def test_score_falls_back_to_relevance_then_placeholder(self):
        """ai_score=0 is falsy → fall to _relevance_score; neither → '?'."""
        payload = claude_analyst._build_payload(
            [
                {"title": "A", "source": "rss", "ai_score": 0,
                 "_relevance_score": 3.0, "summary": ""},
                {"title": "B", "source": "rss", "summary": ""},
            ],
            {"macro": [], "equities": []}, [],
        )
        assert "[score=3.0]" in payload
        assert "[score=?]" in payload

    def test_non_dict_stock_data_does_not_crash(self):
        """stock_data is sometimes a non-dict (collector failure). The
        isinstance guards must yield empty macro/equity sections, not raise."""
        payload = claude_analyst._build_payload(
            self._articles(2), None, None
        )
        assert "LIVE MARKET DATA" in payload
        assert "NEWSWIRE" in payload

    def test_earnings_none_ticker_renders_placeholder(self):
        payload = claude_analyst._build_payload(
            [], {"macro": [], "equities": []},
            [{"ticker": None, "earnings_date": None}],
        )
        # `or` guard, not .get default — present-but-None must not be "None"
        assert "  ?  N/A" in payload


class TestAnalyze:
    def test_returns_placeholder_when_claude_returns_none(self):
        """heartbeat_worker keys its 5-min retry on this exact prefix; analyze
        must never propagate None or an empty string."""
        with patch.object(claude_analyst, "claude_call", return_value=None):
            out = claude_analyst.analyze([], {}, [])
        assert out == "[analyst] No response from Claude."

    def test_returns_claude_text_on_success(self):
        with patch.object(claude_analyst, "claude_call",
                          return_value="**DIGITAL INTERN** briefing body"):
            out = claude_analyst.analyze(
                [{"title": "x", "source": "rss", "ai_score": 9, "summary": ""}],
                {"macro": [], "equities": []}, [],
            )
        assert out == "**DIGITAL INTERN** briefing body"

    def test_empty_string_response_falls_back_to_placeholder(self):
        """`result or placeholder` — an empty Claude response is as useless as
        None and must trip the same retry sentinel."""
        with patch.object(claude_analyst, "claude_call", return_value=""):
            out = claude_analyst.analyze([], {}, [])
        assert out == "[analyst] No response from Claude."

"""Pure-helper tests for the /api/chat watchlist-news-silence enrichment.

``_watchlist_news_silence_chat_lines`` renders paper-trader's
``/api/watchlist-news-silence-skill`` (per-WATCHLIST-ticker live-news
coverage; how many of the ~47 names Opus may choose from are SILENT,
HOT, etc.) into compact chat-context lines.

Discriminating locks:

- **verbatim SSOT** (paper-trader invariant #10): the builder's own
  ``headline`` passes through UNCHANGED — no chat-side re-derivation.
- **healthy universe = silence**: WELL_COVERED / NO_DATA collapse to
  ``[]``, following the ``_persona_book_fit_chat_lines`` silence
  precedent — chat must not carry "coverage fine" filler.
- **actionable verdicts emit ticker lists**: silent_tickers (cap 8) and
  hot_storms (cap 3) carry through verbatim.
- **pure/total**: non-dict / missing keys / non-string entries never
  raise and degrade to the safe subset.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.web_server import _watchlist_news_silence_chat_lines


def _rep(
    verdict="BLIND_UNIVERSE",
    headline="BLIND_UNIVERSE — 30/47 silent (64%)",
    *,
    silent=None,
    storms=None,
):
    if silent is None:
        silent = ["AAPL", "AMD", "AMZN", "INTC", "META", "MU", "NVDA", "TSM"]
    if storms is None:
        storms = [
            {"ticker": "SPY", "n_in_window": 22, "max_score": 8.0},
            {"ticker": "QQQ", "n_in_window": 18, "max_score": 7.5},
            {"ticker": "TSLA", "n_in_window": 10, "max_score": 6.0},
        ]
    return {
        "verdict": verdict,
        "headline": headline,
        "silent_tickers": silent,
        "hot_storms": storms,
    }


# ─── silence on non-actionable verdicts ──────────────────────────────────


def test_non_dict_input_returns_empty():
    assert _watchlist_news_silence_chat_lines(None) == []
    assert _watchlist_news_silence_chat_lines("") == []
    assert _watchlist_news_silence_chat_lines(42) == []
    assert _watchlist_news_silence_chat_lines([]) == []


def test_well_covered_collapses_to_silence():
    assert _watchlist_news_silence_chat_lines(_rep(verdict="WELL_COVERED")) == []


def test_no_data_collapses_to_silence():
    assert _watchlist_news_silence_chat_lines(_rep(verdict="NO_DATA")) == []


def test_unknown_verdict_collapses_to_silence():
    assert _watchlist_news_silence_chat_lines(_rep(verdict="GARBAGE")) == []


# ─── BLIND_UNIVERSE / SPARSE_COVERAGE render verbatim ────────────────────


def test_blind_universe_emits_verbatim_headline_first():
    out = _watchlist_news_silence_chat_lines(_rep())
    assert len(out) >= 1
    assert out[0] == "BLIND_UNIVERSE — 30/47 silent (64%)"


def test_sparse_coverage_also_renders():
    rep = _rep(verdict="SPARSE_COVERAGE", headline="SPARSE_COVERAGE — 12/47 silent (26%)")
    out = _watchlist_news_silence_chat_lines(rep)
    assert out[0] == "SPARSE_COVERAGE — 12/47 silent (26%)"


def test_detail_line_lists_silent_then_storms():
    out = _watchlist_news_silence_chat_lines(_rep())
    detail = out[1]
    assert detail.startswith("  ")
    assert "silent: " in detail
    assert "storms: " in detail


def test_silent_capped_at_8_in_detail_line():
    # Builder caps at 10; chat slice caps at 8 for line length.
    rep = _rep(silent=[f"T{i:02d}" for i in range(10)])
    out = _watchlist_news_silence_chat_lines(rep)
    detail = out[1]
    assert "T00" in detail and "T07" in detail
    assert "T08" not in detail and "T09" not in detail


def test_storms_capped_at_3_in_detail_line():
    rep = _rep(storms=[
        {"ticker": f"S{i}", "n_in_window": 10 + i, "max_score": 5.0}
        for i in range(6)
    ])
    out = _watchlist_news_silence_chat_lines(rep)
    detail = out[1]
    assert "S0" in detail and "S2" in detail
    assert "S3" not in detail and "S5" not in detail


def test_only_silent_present():
    rep = _rep(storms=[])
    out = _watchlist_news_silence_chat_lines(rep)
    assert "silent:" in out[1]
    assert "storms:" not in out[1]


def test_only_storms_present():
    rep = _rep(silent=[])
    out = _watchlist_news_silence_chat_lines(rep)
    assert "storms:" in out[1]
    assert "silent:" not in out[1]


# ─── degradation: malformed sub-rows ─────────────────────────────────────


def test_missing_headline_only_emits_detail():
    rep = _rep(headline=None)
    out = _watchlist_news_silence_chat_lines(rep)
    # Headline dropped, detail line still present.
    assert len(out) == 1
    assert out[0].lstrip().startswith("silent:")


def test_missing_silent_and_storms_only_emits_headline():
    rep = {"verdict": "BLIND_UNIVERSE", "headline": "h"}
    out = _watchlist_news_silence_chat_lines(rep)
    assert out == ["h"]


def test_silent_with_non_string_entries():
    rep = _rep(silent=["NVDA", None, 42, "", "AMD"])
    out = _watchlist_news_silence_chat_lines(rep)
    detail = out[1]
    assert "NVDA" in detail and "AMD" in detail


def test_storms_with_garbage_rows():
    rep = _rep(storms=[
        None,
        "x",
        {"ticker": "SPY"},
        {"ticker": 42},
        {"n_in_window": 10},        # no ticker
        {"ticker": "QQQ"},
    ])
    out = _watchlist_news_silence_chat_lines(rep)
    detail = out[1]
    assert "SPY" in detail
    assert "QQQ" in detail


def test_silent_and_storms_not_lists_skipped():
    rep = {"verdict": "BLIND_UNIVERSE", "headline": "h",
           "silent_tickers": "garbage", "hot_storms": 42}
    out = _watchlist_news_silence_chat_lines(rep)
    assert out == ["h"]

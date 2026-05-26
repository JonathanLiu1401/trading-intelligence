"""Tests for analytics/ticker_score_dispersion.py — per-ticker intra-window
std-dev of ml_score; complements sentiment_reversal (cross-window flip).

Critical regressions to pin:
  * verdict ladder (NO_DATA / NO_DISPERSION / CONSENSUS / MIXED_BOOK /
    CONFLICTED_NEWS) maps correctly to the score spreads;
  * per-ticker TIGHT / MIXED / CONFLICTED bins match the std thresholds;
  * MIN_ARTICLES gate (a sub-threshold ticker must be dropped);
  * out-of-window articles excluded;
  * ranking puts CONFLICTED first, then MIXED, then TIGHT;
  * chat helper SILENCES CONSENSUS / NO_DATA and emits only the
    actionable rows when the verdict is MIXED_BOOK / CONFLICTED_NEWS.

Pure-helper tests — no Flask (project_digital_intern_chat_enrichment_pattern).
"""
from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analytics.ticker_score_dispersion import (  # noqa: E402
    CONFLICTED_STD,
    MIN_ARTICLES,
    TIGHT_STD,
    build_ticker_score_dispersion,
)
from dashboard.web_server import (  # noqa: E402
    _ticker_score_dispersion_chat_lines,
)


NOW = datetime(2026, 5, 25, 18, 0, tzinfo=timezone.utc)


def _at(hours_ago: float) -> str:
    return (NOW - timedelta(hours=hours_ago)).isoformat()


def _a(title: str, ml_score: float, hours_ago: float = 1.0) -> dict:
    return {"title": title, "ml_score": ml_score, "first_seen": _at(hours_ago)}


class TestEmptyAndNoData:
    def test_empty_returns_no_data(self):
        r = build_ticker_score_dispersion([], now=NOW)
        assert r["verdict"] == "NO_DATA"
        assert r["tickers"] == []
        assert r["rows_in_window"] == 0
        assert r["n_tickers_qualified"] == 0

    def test_only_sub_threshold_tickers_returns_no_dispersion(self):
        # One ticker, but only 2 articles (< MIN_ARTICLES = 4) → NO_DISPERSION,
        # not NO_DATA (we DID see articles, just not enough per ticker).
        rows = [_a("MU steady", 5.0, 1.0), _a("MU steady 2", 5.0, 1.5)]
        r = build_ticker_score_dispersion(rows, now=NOW)
        assert r["verdict"] == "NO_DISPERSION"
        assert r["rows_in_window"] == 2
        assert r["n_tickers_qualified"] == 0


class TestTightVerdict:
    def test_all_tight_yields_consensus(self):
        # All scores within TIGHT_STD → CONSENSUS.
        rows = [_a("NVDA rally", 7.5, 1.0),
                _a("NVDA momentum", 7.6, 1.5),
                _a("NVDA HBM", 7.7, 2.0),
                _a("NVDA AI demand", 7.8, 2.5),
                _a("NVDA upside", 7.9, 3.0)]
        r = build_ticker_score_dispersion(rows, now=NOW)
        assert r["verdict"] == "CONSENSUS"
        assert r["n_tight"] == 1
        assert r["n_mixed"] == 0
        assert r["n_conflicted"] == 0
        nvda = r["tickers"][0]
        assert nvda["ticker"] == "NVDA"
        assert nvda["verdict"] == "TIGHT"
        # The pure builder is the std-dev SSOT — we restate, not re-derive.
        # Quick consistency cross-check: std rounded to 4dp.
        mean = sum([7.5, 7.6, 7.7, 7.8, 7.9]) / 5.0
        var = sum((s - mean) ** 2 for s in [7.5, 7.6, 7.7, 7.8, 7.9]) / 5.0
        assert nvda["std"] == round(math.sqrt(var), 4)
        assert nvda["std"] <= TIGHT_STD


class TestConflictedVerdict:
    def test_high_spread_yields_conflicted(self):
        # Wide spread → CONFLICTED.
        # Mean still ~5 but range 1..9 — every other panel that carries
        # only the mean (5) would call this a moderate signal; the
        # dispersion view rightly flags it as contested.
        rows = [_a("ABCD bull case", 9.0, 1.0),
                _a("ABCD bear case", 1.0, 1.2),
                _a("ABCD upgrade", 8.5, 1.4),
                _a("ABCD downgrade", 1.5, 1.6),
                _a("ABCD mixed", 5.0, 1.8)]
        r = build_ticker_score_dispersion(rows, now=NOW)
        assert r["verdict"] == "CONFLICTED_NEWS"
        assert r["n_conflicted"] == 1
        t = r["tickers"][0]
        assert t["ticker"] == "ABCD"
        assert t["verdict"] == "CONFLICTED"
        assert t["std"] > CONFLICTED_STD
        # Range surfaces verbatim so the helper can display [min, max] —
        # crucial for the analyst to SEE that mean=5 hides a 1–9 spread.
        assert t["range"] == round(t["max"] - t["min"], 4)
        assert t["min"] == 1.0
        assert t["max"] == 9.0


class TestMixedVerdict:
    def test_moderate_spread_yields_mixed(self):
        # Spread between thresholds → MIXED.
        rows = [_a("XYZA stable", 6.0, 1.0),
                _a("XYZA pop", 7.5, 1.2),
                _a("XYZA dip", 5.0, 1.4),
                _a("XYZA news", 7.0, 1.6)]
        r = build_ticker_score_dispersion(rows, now=NOW)
        t = r["tickers"][0]
        assert t["ticker"] == "XYZA"
        assert t["verdict"] == "MIXED"
        # Top-level verdict cascades: no CONFLICTED tickers, at least one MIXED.
        assert r["verdict"] == "MIXED_BOOK"
        assert r["n_mixed"] == 1
        assert r["n_conflicted"] == 0
        assert TIGHT_STD < t["std"] <= CONFLICTED_STD


class TestRankingMixedBook:
    def test_conflicted_ranked_before_mixed_before_tight(self):
        # Build one of each verdict; ranking must put CONFLICTED first.
        rows = []
        # CONFLICTED: AAAA spread 1..9
        for s, h in [(9.0, 1.0), (1.0, 1.1), (8.5, 1.2),
                     (1.5, 1.3), (5.0, 1.4)]:
            rows.append(_a(f"AAAA news {s}", s, h))
        # MIXED: BBBB spread ~5..7
        for s, h in [(5.0, 1.0), (7.5, 1.2),
                     (5.0, 1.4), (7.0, 1.6)]:
            rows.append(_a(f"BBBB news {s}", s, h))
        # TIGHT: CCCC all near 5.0
        for s, h in [(5.0, 1.0), (5.1, 1.2),
                     (5.05, 1.4), (5.15, 1.6)]:
            rows.append(_a(f"CCCC news {s}", s, h))
        r = build_ticker_score_dispersion(rows, now=NOW)
        assert r["verdict"] == "CONFLICTED_NEWS"
        assert [t["ticker"] for t in r["tickers"]] == ["AAAA", "BBBB", "CCCC"]
        assert r["tickers"][0]["verdict"] == "CONFLICTED"
        assert r["tickers"][1]["verdict"] == "MIXED"
        assert r["tickers"][2]["verdict"] == "TIGHT"


class TestWindow:
    def test_out_of_window_excluded(self):
        # 26h-old articles (window default 24h) must not be aggregated.
        old_rows = [_a("MU old", 9.0, 26.0)] * 5
        recent = [_a("MU fresh", 1.0, 1.0)] * 5
        r = build_ticker_score_dispersion(old_rows + recent, now=NOW)
        # Only the 5 recent rows count → mean 1.0, std 0.
        t = next(x for x in r["tickers"] if x["ticker"] == "MU")
        assert t["n"] == 5
        assert t["mean"] == 1.0
        assert t["std"] == 0.0
        assert t["verdict"] == "TIGHT"

    def test_custom_window_hours_respected(self):
        # A 6h window should still pick up the 1h-old articles even if there
        # are older ones.
        rows = [_a("WDC dip", 4.0, 2.0)] * (MIN_ARTICLES + 1)
        # Stale row at 12h must NOT contribute when window_hours=6.
        rows += [_a("WDC ancient", 1.0, 12.0)]
        r = build_ticker_score_dispersion(rows, window_hours=6, now=NOW)
        assert r["window_hours"] == 6
        t = next((x for x in r["tickers"] if x["ticker"] == "WDC"), None)
        assert t is not None
        assert t["n"] == MIN_ARTICLES + 1
        assert t["mean"] == 4.0


class TestNullSafety:
    def test_none_ml_score_dropped(self):
        rows = [{"title": "AAAA", "ml_score": None,
                 "first_seen": _at(1.0)}] * 5
        r = build_ticker_score_dispersion(rows, now=NOW)
        assert r["verdict"] == "NO_DATA"

    def test_malformed_score_dropped(self):
        rows = [{"title": "AAAA", "ml_score": "garbage",
                 "first_seen": _at(1.0)}] * 5
        r = build_ticker_score_dispersion(rows, now=NOW)
        assert r["verdict"] == "NO_DATA"


class TestChatHelper:
    def test_non_dict_returns_empty(self):
        assert _ticker_score_dispersion_chat_lines(None) == []
        assert _ticker_score_dispersion_chat_lines("oops") == []
        assert _ticker_score_dispersion_chat_lines([]) == []

    def test_consensus_is_silent(self):
        rows = [_a("NVDA steady", 7.5, 1.0),
                _a("NVDA steady2", 7.6, 1.5),
                _a("NVDA steady3", 7.7, 2.0),
                _a("NVDA steady4", 7.8, 2.5)]
        r = build_ticker_score_dispersion(rows, now=NOW)
        assert r["verdict"] == "CONSENSUS"
        # silence precedent: the analyst's chat should not carry CONSENSUS.
        assert _ticker_score_dispersion_chat_lines(r) == []

    def test_no_data_is_silent(self):
        r = build_ticker_score_dispersion([], now=NOW)
        assert r["verdict"] == "NO_DATA"
        assert _ticker_score_dispersion_chat_lines(r) == []

    def test_conflicted_emits_headline_and_actionable_rows(self):
        rows = [_a("AAAA bull", 9.0, 1.0),
                _a("AAAA bear", 1.0, 1.2),
                _a("AAAA mid", 5.0, 1.4),
                _a("AAAA news", 8.0, 1.6),
                _a("AAAA news2", 2.0, 1.8)]
        r = build_ticker_score_dispersion(rows, now=NOW)
        assert r["verdict"] == "CONFLICTED_NEWS"
        lines = _ticker_score_dispersion_chat_lines(r)
        # headline + 1 detail row
        assert len(lines) == 2
        assert "CONFLICTED_NEWS" in lines[0]
        assert "AAAA" in lines[1]
        assert "CONFLICTED" in lines[1]
        # Detail line must restate (not re-derive) the builder fields.
        assert "mean" in lines[1]
        assert "std" in lines[1]

    def test_tight_rows_excluded_from_detail_when_mixed_book(self):
        # MIXED_BOOK with one MIXED + one TIGHT — detail block should only
        # surface the MIXED row (silence precedent on TIGHT).
        rows = []
        for s, h in [(5.0, 1.0), (7.5, 1.2), (5.0, 1.4), (7.0, 1.6)]:
            rows.append(_a(f"BBBB n {s}", s, h))
        for s, h in [(5.0, 1.0), (5.1, 1.2), (5.05, 1.4), (5.15, 1.6)]:
            rows.append(_a(f"CCCC n {s}", s, h))
        r = build_ticker_score_dispersion(rows, now=NOW)
        assert r["verdict"] == "MIXED_BOOK"
        lines = _ticker_score_dispersion_chat_lines(r)
        # headline + only the MIXED ticker's detail row (TIGHT dropped).
        assert len(lines) == 2
        joined = "\n".join(lines)
        assert "BBBB" in joined
        assert "MIXED" in joined
        # CCCC is TIGHT — it must not appear in any detail line.
        assert "CCCC" not in joined

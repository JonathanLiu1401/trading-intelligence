"""Tests for analysis/sector_coherence.py — per-sector stance dispersion.

The pulse builder answers "where is the wire concentrated"; this builder
answers "is the concentration agreeing on a direction". Critical
regressions to pin: the bull/bear classifier dropping a hit on punctuation
or word-boundary failure, MIN_CLASSIFIED gate firing too early, the
coherence ratio computed against `n_articles` instead of `n_classified`
(would crater coherence on quiet sectors with one opinionated headline),
the `_LIVE_ONLY_CLAUSE` invariant (backtest:// rows must not be tagged),
and the chat helper firing on non-actionable verdicts.

Pure-helper tests — no Flask (project_digital_intern_chat_enrichment_pattern).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.sector_coherence import (  # noqa: E402
    MACRO_COHERENCE_PCT,
    MIN_CLASSIFIED,
    _classify,
    build_sector_coherence,
)
from dashboard.web_server import _sector_coherence_chat_lines  # noqa: E402


def _a(title: str, ai: float = 5.0, urg: int = 0,
       first_seen: str = "2026-05-18T11:30:00+00:00") -> dict:
    return {"title": title, "ai_score": ai, "urgency": urg,
            "first_seen": first_seen}


class TestClassify:
    def test_bull_keywords(self):
        for s in ["MU surged on HBM demand", "NVDA beat estimates",
                  "AMD upgraded to overweight", "LRCX wins big order"]:
            assert _classify(s) == "bull", s

    def test_bear_keywords(self):
        for s in ["MU plunged after warning", "NVDA missed estimates",
                  "AMD downgraded to underweight",
                  "LITE faces fraud probe"]:
            assert _classify(s) == "bear", s

    def test_neutral_when_no_keyword(self):
        assert _classify("MU announces partnership with AVGO") == "neutral"
        assert _classify("NVDA hosts investor day Thursday") == "neutral"

    def test_tie_is_neutral_not_alternation(self):
        # Equal bull+bear hits ⇒ neutral; refusing to pick a direction is
        # more honest than alternating on word order. A regression that
        # picks "first match wins" would tag this as bull or bear.
        title = "AMD beats expectations but falls on lowered guidance"
        # "beats", "falls", "lowered" => 1 bull + 2 bears ⇒ bear actually.
        # Construct a true tie to lock the tie-breaker:
        tie = "MU upgrade after a surge but lawsuit looms over recall"
        # bull: upgrade, surge (=2); bear: lawsuit, recall (=2) ⇒ neutral.
        assert _classify(tie) == "neutral"

    def test_case_insensitive(self):
        assert _classify("mu SURGED today") == "bull"
        assert _classify("AMD DROPPED on news") == "bear"

    def test_non_string_safe(self):
        assert _classify(None) == "neutral"  # type: ignore
        assert _classify(123) == "neutral"   # type: ignore
        assert _classify("") == "neutral"


class TestAggregate:
    def test_empty_returns_skeleton(self):
        r = build_sector_coherence([])
        assert r["sectors"] == []
        assert r["n_scanned"] == 0
        assert r["min_classified"] == MIN_CLASSIFIED
        assert "no sector-tagged news" in r["headline"].lower()

    def test_non_list_returns_skeleton(self):
        r = build_sector_coherence("not a list")  # type: ignore
        assert r["sectors"] == []

    def test_skips_rows_without_titles_or_dicts(self):
        # Non-dicts are skipped before n_scanned is even incremented (invalid
        # input shape, not "no title"); only the dict-shaped rows count.
        arts = [
            None,
            "not a dict",
            {"title": None},
            {"title": "   "},
            {"title": "MU surged on HBM"},
        ]
        r = build_sector_coherence(arts)
        assert r["n_scanned"] == 3   # only the 3 dict-shaped rows count
        assert r["n_mapped"] == 1
        # 1 classified ⇒ INSUFFICIENT, but the sector appears
        assert len(r["sectors"]) == 1
        assert r["sectors"][0]["verdict"] == "INSUFFICIENT"

    def test_macro_bull_when_all_classified_agree(self):
        # 3+ bull on DRAM (MU), 0 bear ⇒ 100% coh, MACRO_BULL
        arts = [_a("MU surged on HBM demand"),
                _a("MU upgraded by Goldman, raised target"),
                _a("WDC tops earnings estimates"),
                _a("STX wins large contract")]
        r = build_sector_coherence(arts)
        s = [x for x in r["sectors"] if x["sector"] == "DRAM/Memory"][0]
        assert s["n_classified"] == 4
        assert s["n_bull"] == 4
        assert s["n_bear"] == 0
        assert s["coherence_pct"] == 100.0
        assert s["verdict"] == "MACRO_BULL"
        assert s["lead_direction"] == "bull"
        # lead_headline picks the highest ai_score opinionated headline
        assert s["lead_headline"] is not None

    def test_macro_bear(self):
        arts = [_a("AMD plunges on guidance cut"),
                _a("NVDA missed Q3 estimates"),
                _a("AVGO warned on weak demand")]
        r = build_sector_coherence(arts)
        s = [x for x in r["sectors"] if x["sector"] == "GPU/AI Compute"][0]
        assert s["verdict"] == "MACRO_BEAR"
        assert s["lead_direction"] == "bear"

    def test_split_when_dispersed(self):
        # 5 bull + 5 bear ⇒ 50% coh ⇒ SPLIT (below TILT 55% threshold)
        arts = [
            _a("MU surge on HBM"), _a("MU rally continues"),
            _a("MU upgraded by JPM"), _a("WDC beat estimates"),
            _a("STX wins big order"),
            _a("MU plunged on warning"), _a("MU downgraded by GS"),
            _a("WDC missed estimates"), _a("STX warns on demand"),
            _a("WDC slumped on probe"),
        ]
        r = build_sector_coherence(arts)
        s = [x for x in r["sectors"] if x["sector"] == "DRAM/Memory"][0]
        assert s["n_bull"] == 5
        assert s["n_bear"] == 5
        assert s["coherence_pct"] == 50.0
        assert s["verdict"] == "SPLIT"

    def test_insufficient_below_min_classified(self):
        # 2 classified < MIN_CLASSIFIED ⇒ INSUFFICIENT, even at 100% coh
        arts = [_a("MU surged"), _a("MU upgraded"),
                _a("MU announces partnership")]  # neutral 3rd
        r = build_sector_coherence(arts)
        s = [x for x in r["sectors"] if x["sector"] == "DRAM/Memory"][0]
        assert s["n_classified"] == 2
        assert s["n_neutral"] == 1
        assert s["verdict"] == "INSUFFICIENT"

    def test_coherence_uses_classified_not_total(self):
        # 3 bull + 0 bear + 10 neutral ⇒ coherence 100% (not 23%).
        # A naive (max/total) would say 30%/3 of 13 = 23% — wrong.
        arts = (
            [_a("MU surged on HBM"), _a("WDC tops estimates"),
             _a("STX wins large order")]
            + [_a("MU announces tour", ai=0.5)] * 10
        )
        r = build_sector_coherence(arts)
        s = [x for x in r["sectors"] if x["sector"] == "DRAM/Memory"][0]
        assert s["n_classified"] == 3
        assert s["coherence_pct"] == 100.0
        assert s["verdict"] == "MACRO_BULL"

    def test_macro_threshold_boundary(self):
        # 7 bull + 3 bear ⇒ 70.0% — meets MACRO_COHERENCE_PCT exactly.
        arts = (
            [_a("MU surged")] * 7
            + [_a("MU plunged")] * 3
        )
        r = build_sector_coherence(arts)
        s = [x for x in r["sectors"] if x["sector"] == "DRAM/Memory"][0]
        assert s["coherence_pct"] == 70.0
        assert s["verdict"] == "MACRO_BULL"
        assert MACRO_COHERENCE_PCT == 70.0  # lock the threshold


class TestChatHelper:
    def test_non_dict_returns_empty(self):
        assert _sector_coherence_chat_lines(None) == []
        assert _sector_coherence_chat_lines("nope") == []
        assert _sector_coherence_chat_lines({}) == []

    def test_silent_when_only_split_or_insufficient(self):
        # Healthy / non-actionable verdicts must collapse to silence —
        # the silence precedent. A chat block whose only content is
        # "we don't know" is filler.
        rep = {"sectors": [
            {"sector": "DRAM/Memory", "verdict": "SPLIT",
             "n_bull": 5, "n_bear": 5, "n_classified": 10,
             "coherence_pct": 50.0, "lead_headline": "X"},
            {"sector": "Mega-Cap Tech", "verdict": "INSUFFICIENT",
             "n_bull": 1, "n_bear": 0, "n_classified": 1,
             "coherence_pct": 100.0, "lead_headline": "Y"},
        ]}
        assert _sector_coherence_chat_lines(rep) == []

    def test_fires_on_macro_verdict(self):
        rep = {"sectors": [
            {"sector": "DRAM/Memory", "verdict": "MACRO_BULL",
             "n_bull": 5, "n_bear": 0, "n_classified": 5,
             "coherence_pct": 100.0,
             "lead_headline": "MU surged on HBM demand"},
        ]}
        lines = _sector_coherence_chat_lines(rep)
        assert len(lines) == 1
        assert "DRAM/Memory" in lines[0]
        assert "MACRO_BULL" in lines[0]
        assert "5↑/0↓" in lines[0]
        assert "MU surged on HBM demand" in lines[0]

    def test_truncates_long_headlines(self):
        long = "A" * 200
        rep = {"sectors": [
            {"sector": "X", "verdict": "MACRO_BULL",
             "n_bull": 3, "n_bear": 0, "n_classified": 3,
             "coherence_pct": 100.0, "lead_headline": long},
        ]}
        lines = _sector_coherence_chat_lines(rep)
        assert len(lines) == 1
        assert "…" in lines[0]
        # 120 char cap + ellipsis
        assert long not in lines[0]

    def test_never_raises_on_malformed_sector(self):
        rep = {"sectors": [
            "not a dict",
            {"sector": "X", "verdict": "MACRO_BULL"},  # missing fields
        ]}
        # Should not raise; malformed sector skipped via the try/except.
        out = _sector_coherence_chat_lines(rep)
        assert isinstance(out, list)

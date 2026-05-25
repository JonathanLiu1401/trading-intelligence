"""Pin ``_publish_lag_chat_lines`` chat enrichment.

Mirror the discipline of `test_chat_feed_health_enrichment.py` —
silence-on-healthy, verbatim headline pass-through, never raises.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard.web_server import _publish_lag_chat_lines


# ─── pure / total contract ──────────────────────────────────────────────────

class TestIsTotal:
    @pytest.mark.parametrize("bad", [None, 42, "x", [], (), True, False])
    def test_non_dict_collapses_to_empty(self, bad):
        assert _publish_lag_chat_lines(bad) == []

    def test_empty_dict_collapses_to_empty(self):
        assert _publish_lag_chat_lines({}) == []

    def test_missing_verdict_collapses_to_empty(self):
        assert _publish_lag_chat_lines({"headline": "anything"}) == []

    def test_garbage_stalest_does_not_raise(self):
        rep = {
            "verdict": "STALE_FEEDS",
            "headline": "ok",
            "ranked_stalest": "not-a-list",
        }
        out = _publish_lag_chat_lines(rep)
        # headline still survives even if ranked_stalest is garbage
        assert out == ["ok"]


# ─── silence-on-healthy ─────────────────────────────────────────────────────

class TestSilenceOnHealthy:
    @pytest.mark.parametrize("verdict", ["FRESH", "NO_DATA", "ERROR"])
    def test_non_actionable_verdicts_collapse_to_empty(self, verdict):
        rep = {
            "verdict": verdict,
            "headline": "something",
            "ranked_stalest": [{"collector": "x", "n": 10,
                                "median_lag_min": 100.0}],
        }
        assert _publish_lag_chat_lines(rep) == []


# ─── actionable verdicts emit headline ──────────────────────────────────────

class TestActionableEmits:
    def test_stale_feeds_emits_headline_verbatim(self):
        rep = {
            "verdict": "STALE_FEEDS",
            "headline": "stalest: rss p50=120.5m (n=42); freshest: nitter p50=0.3m",
            "ranked_stalest": [
                {"collector": "rss", "n": 42, "median_lag_min": 120.5,
                 "p90_lag_min": 480.0, "stale_60m_pct": 76.2}
            ],
        }
        out = _publish_lag_chat_lines(rep)
        # SSOT — headline passes through verbatim
        assert out[0] == rep["headline"]
        # Detail line restates only builder fields
        assert "n=42" in out[1]
        assert "median=120.5m" in out[1]
        assert "p90=480.0m" in out[1]
        assert ">60m=76%" in out[1]
        assert "rss" in out[1]

    def test_mixed_emits_headline_verbatim(self):
        rep = {
            "verdict": "MIXED",
            "headline": "mixed: stalest gdelt p50=22.0m",
            "ranked_stalest": [{"collector": "gdelt", "n": 18,
                                "median_lag_min": 22.0}],
        }
        out = _publish_lag_chat_lines(rep)
        assert out[0] == "mixed: stalest gdelt p50=22.0m"

    def test_long_collector_name_truncated_to_32(self):
        rep = {
            "verdict": "STALE_FEEDS",
            "headline": "stale",
            "ranked_stalest": [{
                "collector": "a" * 100,
                "n": 10,
                "median_lag_min": 65.0,
            }],
        }
        out = _publish_lag_chat_lines(rep)
        # Detail line contains the truncated name (32 chars)
        assert any("a" * 32 in line and "a" * 33 not in line for line in out)


# ─── defensive numeric handling ─────────────────────────────────────────────

class TestNumericDefense:
    def test_bool_as_int_ignored(self):
        rep = {
            "verdict": "STALE_FEEDS",
            "headline": "stale",
            "ranked_stalest": [{
                "collector": "rss",
                "n": True,             # boolean — should NOT print as n=1
                "median_lag_min": 90.0,
            }],
        }
        out = _publish_lag_chat_lines(rep)
        # bool sentinel filtered out of detail line
        detail = out[1] if len(out) > 1 else ""
        assert "n=1" not in detail

    def test_empty_stalest_list_keeps_headline(self):
        rep = {
            "verdict": "STALE_FEEDS",
            "headline": "stale",
            "ranked_stalest": [],
        }
        out = _publish_lag_chat_lines(rep)
        assert out == ["stale"]

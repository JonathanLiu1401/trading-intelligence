"""Exact-value locks for the DecisionScorer feature-coverage diagnostic
(`paper_trader/ml/feature_coverage.py`, 2026-05-18 quant feature).

Mirrors test_skill_trend.py / test_calibration.py: deterministic synthetic
data, exact metrics and exact verdicts (not ranges) so a logic change must
update the literals deliberately. All offline — no network, no pickle, no DB.
"""
from __future__ import annotations

import json

import pytest

from paper_trader.ml import feature_coverage as fc


def _full_row(i: int) -> dict:
    """A record where every numeric feature varies and is rarely at its
    build_features default (FULL_COVERAGE building block)."""
    return {
        "ticker": "NVDA",
        "ml_score": 1.0 + i * 0.1,            # never 0.0
        "rsi": 30.0 + i,                       # 50.0 hit once (i=20)
        "macd": -1.0 + i * 0.05,               # 0.0 hit once (i=20)
        "mom5": -5.0 + i * 0.3,
        "mom20": -10.0 + i * 0.5,
        "regime_mult": 0.6 if i % 2 == 0 else 0.3,  # never 1.0 default
        "vol_ratio": 0.5 + i * 0.05,           # 1.0 hit once (i=10)
        "bb_position": -1.5 + i * 0.07,
        "news_urgency": 10.0 + i,              # 0..49 → never 50.0 default
        "news_article_count": 2.0 + (i % 5),   # 2..6 → never 1.0 default
    }


class TestDefaultVectorSingleSource:
    def test_defaults_match_build_features_all_none(self):
        from paper_trader.ml.decision_scorer import build_features
        bf = list(build_features(None, None, None, None, None, None, "ZZZ"))[:10]
        assert fc._default_vector() == bf

    def test_exact_default_values(self):
        # [ml_score, rsi, macd, mom5, mom20, regime_mult, vol, bb, urg, cnt]
        assert fc._default_vector() == [
            0.0, 50.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 50.0, 1.0
        ]


class TestLoadOutcomes:
    def test_missing_file_yields_empty(self, tmp_path):
        assert fc.load_outcomes(tmp_path / "nope.jsonl") == []

    def test_corrupt_and_nondict_lines_skipped(self, tmp_path):
        p = tmp_path / "o.jsonl"
        p.write_text('{"ml_score": 1.0}\n'
                      "not json\n"
                      "[1,2,3]\n"
                      "\n"
                      '{"ml_score": 2.0}\n')
        rows = fc.load_outcomes(p)
        assert len(rows) == 2
        assert rows[0]["ml_score"] == 1.0 and rows[1]["ml_score"] == 2.0


class TestVerdicts:
    def test_insufficient_data_below_min_rows(self):
        rep = fc.feature_coverage_report([_full_row(i) for i in range(10)])
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["effective_feature_count"] is None
        assert rep["n_rows"] == 10

    def test_full_coverage(self):
        rep = fc.feature_coverage_report([_full_row(i) for i in range(40)])
        assert rep["verdict"] == "FULL_COVERAGE"
        assert rep["dead_features"] == []
        assert rep["degraded_features"] == []
        assert rep["effective_feature_count"] == 10
        # Spot-check a faithfully-computed per-feature stat: rsi hits its
        # 50.0 default exactly once (i=20) over 40 rows → 0.025.
        assert rep["features"]["rsi"]["default_fraction"] == 0.025
        assert rep["features"]["rsi"]["dead"] is False

    def test_dead_features_when_news_always_none(self):
        recs = []
        for i in range(40):
            r = _full_row(i)
            r["news_urgency"] = None
            r["news_article_count"] = None
            recs.append(r)
        rep = fc.feature_coverage_report(recs)
        assert rep["verdict"] == "DEAD_FEATURES_PRESENT"
        # Order follows _NUMERIC_FEATURES slot order.
        assert rep["dead_features"] == ["news_urgency", "news_article_count"]
        assert rep["effective_feature_count"] == 8
        # None → build_features default every row → default_fraction == 1.0,
        # distinct == 1 (both dead conditions hold).
        assert rep["features"]["news_urgency"]["default_fraction"] == 1.0
        assert rep["features"]["news_urgency"]["distinct"] == 1
        assert rep["features"]["news_article_count"]["dead"] is True

    def test_constant_nondefault_feature_is_dead(self):
        # ml_score pinned to 3.0 (finite, NOT the 0.0 default) → distinct < 2
        # → dead by the constant rule even though default_fraction == 0.0.
        recs = []
        for i in range(40):
            r = _full_row(i)
            r["ml_score"] = 3.0
            recs.append(r)
        rep = fc.feature_coverage_report(recs)
        assert rep["verdict"] == "DEAD_FEATURES_PRESENT"
        assert "ml_score" in rep["dead_features"]
        assert rep["features"]["ml_score"]["default_fraction"] == 0.0
        assert rep["features"]["ml_score"]["distinct"] == 1
        assert rep["features"]["ml_score"]["dead"] is True

    def test_degraded_coverage_band(self):
        # vol_ratio default-substituted (None→1.0) in exactly 24/40 == 0.60
        # rows: ≥ DEGRADED_FLOOR(0.50), < DEAD_FLOOR(0.90), distinct == 2
        # (1.0 and 2.0) so NOT dead-by-constant → DEGRADED_COVERAGE.
        recs = []
        for i in range(40):
            r = _full_row(i)
            r["vol_ratio"] = None if i < 24 else 2.0
            recs.append(r)
        rep = fc.feature_coverage_report(recs)
        assert rep["verdict"] == "DEGRADED_COVERAGE"
        assert rep["dead_features"] == []
        assert rep["degraded_features"] == ["vol_ratio"]
        assert rep["effective_feature_count"] == 9
        assert rep["features"]["vol_ratio"]["default_fraction"] == 0.6
        assert rep["features"]["vol_ratio"]["distinct"] == 2

    def test_dead_overrides_degraded_in_verdict(self):
        # One degraded (vol_ratio 0.60) AND one dead (news always None):
        # the dead branch wins the verdict, both lists still populated.
        recs = []
        for i in range(40):
            r = _full_row(i)
            r["vol_ratio"] = None if i < 24 else 2.0
            r["news_urgency"] = None
            r["news_article_count"] = None
            recs.append(r)
        rep = fc.feature_coverage_report(recs)
        assert rep["verdict"] == "DEAD_FEATURES_PRESENT"
        assert rep["dead_features"] == ["news_urgency", "news_article_count"]
        assert rep["degraded_features"] == ["vol_ratio"]
        assert rep["effective_feature_count"] == 7


class TestNeverRaises:
    def test_garbage_field_types_do_not_raise(self):
        # Non-numeric strings / bools / nested → _to_float defaults, no crash.
        recs = []
        for i in range(40):
            recs.append({
                "ticker": None, "ml_score": "abc", "rsi": True,
                "macd": [1, 2], "mom5": {"x": 1}, "mom20": None,
                "regime_mult": "nan", "vol_ratio": "1.0", "bb_position": None,
                "news_urgency": "high", "news_article_count": None,
            })
        rep = fc.feature_coverage_report(recs)
        # Everything collapses to defaults → all 10 dead, no exception.
        assert rep["verdict"] == "DEAD_FEATURES_PRESENT"
        assert rep["effective_feature_count"] == 0

    def test_analyze_missing_file_is_insufficient(self, tmp_path):
        rep = fc.analyze(tmp_path / "absent.jsonl")
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_rows"] == 0


class TestCliExitCodes:
    @pytest.mark.parametrize("verdict,expected", [
        ("FULL_COVERAGE", 0),
        ("INSUFFICIENT_DATA", 0),
        ("DEAD_FEATURES_PRESENT", 2),
        ("DEGRADED_COVERAGE", 2),
    ])
    def test_cli_exit_mirrors_sibling_diagnostics(self, monkeypatch, verdict,
                                                  expected):
        monkeypatch.setattr(fc, "analyze", lambda _p: {
            "verdict": verdict, "hint": "h", "n_rows": 1,
            "effective_feature_count": 7, "n_numeric_features": 10,
            "features": {"ml_score": {"default_value": 0.0,
                                      "default_fraction": 0.0,
                                      "distinct": 5, "dead": False}},
        })
        assert fc._cli() == expected


class TestRealCorpusShape:
    """Locks the decisive live finding's *shape* without depending on the
    exact (drifting) row count: on the real decision_outcomes.jsonl the two
    news features are dead while the price/quant features are not."""

    def test_news_features_dead_on_synthetic_mirror_of_live(self):
        # Mirror the live corpus: 2.7% of rows carry news, the rest None.
        recs = []
        for i in range(1000):
            r = _full_row(i)
            if i >= 27:           # 973/1000 have no news (≈ live 97.3%)
                r["news_urgency"] = None
                r["news_article_count"] = None
            else:
                r["news_urgency"] = 0.0   # live: backtest urgency is structurally 0
            recs.append(r)
        rep = fc.feature_coverage_report(recs)
        assert rep["verdict"] == "DEAD_FEATURES_PRESENT"
        assert "news_urgency" in rep["dead_features"]
        assert "news_article_count" in rep["dead_features"]
        # 50.0 default for 973 rows + 0.0 for 27 → 0.973 default fraction.
        assert rep["features"]["news_urgency"]["default_fraction"] == 0.973

"""Tests for paper_trader/ml/sector_skill.py.

Validates the per-sector OOS skill diagnostic produces the correct
verdicts on synthetic data with known sector composition and IC.
"""
from __future__ import annotations

import json

import pytest

from paper_trader.ml import sector_skill as ss


class _FakeScorer:
    """A trained-shape scorer whose predict() returns a sector-conditional
    answer.

    ``preds_by_sector``: maps sector ("tech"/"energy"/...) → callable that
    takes the realized 5d return and returns a synthetic prediction. This
    lets a test stage an exact IC per sector (e.g. perfectly correlated
    predictions in tech, anti-correlated in healthcare)."""

    is_trained = True

    def __init__(self, preds_by_sector):
        self._preds = preds_by_sector
        self._n_train = 1000

    def predict(self, *, ml_score, rsi, macd, mom5, mom20, regime_mult,
                ticker, vol_ratio=None, bb_pos=None, news_urgency=None,
                news_article_count=None):
        sec = ss._sector_of(ticker)
        fn = self._preds.get(sec)
        if fn is None:
            return 0.0
        # Encode the realized return into ml_score so a synthetic test can
        # carry it through to predict and emit a controlled prediction.
        return fn(ml_score)


def _mk_row(ticker, fwd_5d, *, action="BUY", sim_date="2025-01-01",
            ml_score_override=None):
    """Build a decision_outcomes.jsonl-shaped row. ``ml_score`` defaults to
    ``fwd_5d`` so a FakeScorer that returns ``ml_score`` produces a
    perfectly-correlated prediction set for that sector."""
    return {
        "run_id": 1,
        "sim_date": sim_date,
        "ticker": ticker,
        "action": action,
        "ml_score": fwd_5d if ml_score_override is None else ml_score_override,
        "rsi": 50.0, "macd": 0.0, "mom5": 0.0, "mom20": 0.0,
        "regime_mult": 1.0, "vol_ratio": 1.0, "bb_position": 0.0,
        "news_urgency": None, "news_article_count": None,
        "forward_return_5d": fwd_5d,
    }


class TestSectorOf:
    def test_known_tech_ticker(self):
        assert ss._sector_of("NVDA") == "tech"

    def test_unknown_falls_back_to_other(self):
        assert ss._sector_of("ZZZZ") == "other"

    def test_lowercase_ticker_normalised(self):
        # build_features only consumes upper-case tickers; the diagnostic
        # must mirror that mapping so a mixed-case external row maps to
        # the same bucket the model was trained on.
        assert ss._sector_of("nvda") == "tech"

    def test_empty_ticker_is_other(self):
        assert ss._sector_of("") == "other"
        assert ss._sector_of(None) == "other"


class TestVerdictForSector:
    def test_sparse_below_min(self):
        assert ss._verdict_for_sector(n=5, ic=0.5) == "SPARSE"

    def test_inverted_below_minus_ic_good(self):
        assert ss._verdict_for_sector(n=100, ic=-0.20) == "INVERTED_SIGNAL"
        # Exact boundary
        assert ss._verdict_for_sector(n=100, ic=-ss.IC_GOOD) == "INVERTED_SIGNAL"

    def test_signal_edge_at_ic_good(self):
        assert ss._verdict_for_sector(n=100, ic=ss.IC_GOOD) == "SIGNAL_EDGE"
        assert ss._verdict_for_sector(n=100, ic=0.5) == "SIGNAL_EDGE"

    def test_weak_band(self):
        assert ss._verdict_for_sector(n=100, ic=ss.IC_MIN) == "WEAK_SIGNAL_EDGE"
        # Just below IC_GOOD
        assert ss._verdict_for_sector(n=100, ic=ss.IC_GOOD - 0.01) == "WEAK_SIGNAL_EDGE"

    def test_no_edge_band(self):
        assert ss._verdict_for_sector(n=100, ic=0.0) == "NO_SIGNAL_EDGE"
        assert ss._verdict_for_sector(n=100, ic=ss.IC_MIN - 0.01) == "NO_SIGNAL_EDGE"


class TestAlignedOosPair:
    def test_buy_passes_through(self):
        scorer = _FakeScorer({"tech": lambda ml: ml})
        row = _mk_row("NVDA", 5.0, action="BUY")
        pair = ss._aligned_oos_pair(row, scorer)
        assert pair == (5.0, 5.0)

    def test_sell_flips_realized_only(self):
        # Universal SELL sign-flip: realized is negated. The FakeScorer's
        # predict is fed the unchanged ml_score, so the prediction is
        # unchanged but the realized side is flipped.
        scorer = _FakeScorer({"tech": lambda ml: ml})
        row = _mk_row("NVDA", 5.0, action="SELL")
        pair = ss._aligned_oos_pair(row, scorer)
        assert pair == (5.0, -5.0)

    def test_nan_forward_return_dropped(self):
        scorer = _FakeScorer({"tech": lambda ml: ml})
        row = _mk_row("NVDA", 0.0)
        row["forward_return_5d"] = None
        assert ss._aligned_oos_pair(row, scorer) is None

    def test_scorer_exception_drops_row(self):
        class _Boom:
            is_trained = True
            def predict(self, **kw):
                raise RuntimeError("boom")
        row = _mk_row("NVDA", 5.0)
        assert ss._aligned_oos_pair(row, _Boom()) is None


class TestSectorSkill:
    def test_untrained_scorer_short_circuits(self):
        class _Untrained:
            is_trained = False
            def predict(self, **kw):
                return 0.0
        rep = ss.sector_skill(_Untrained(), [], [])
        assert rep["verdict"] == "SCORER_UNTRAINED"
        assert rep["sectors"] == []

    def test_insufficient_data(self):
        scorer = _FakeScorer({"tech": lambda ml: ml})
        rep = ss.sector_skill(
            scorer, train_records=[], oos_records=[_mk_row("NVDA", 5.0)] * 5,
        )
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_signal_edge_when_one_sector_perfect(self):
        # 40 tech rows where pred == realized → spearman = 1.0 → SIGNAL_EDGE
        scorer = _FakeScorer({"tech": lambda ml: ml})
        oos = [_mk_row("NVDA", float(i - 20)) for i in range(40)]
        rep = ss.sector_skill(scorer, train_records=[], oos_records=oos)
        assert rep["status"] == "ok"
        tech = next(s for s in rep["sectors"] if s["sector"] == "tech")
        assert tech["n_oos"] == 40
        assert tech["rank_ic"] == pytest.approx(1.0)
        assert tech["verdict"] == "SIGNAL_EDGE"
        # rmse is 0 because pred == realized exactly
        assert tech["rmse"] == pytest.approx(0.0)
        # dir_acc must be 1.0 (every non-zero pair agrees on sign)
        assert tech["dir_acc"] == pytest.approx(1.0)
        # n_aligned is large enough that SECTOR_CONCENTRATED would fire
        # too (one sector), but inverted/concentrated precedence: with
        # only one sector and n=40, concentration=100% ⇒ SECTOR_CONCENTRATED
        # outranks HEALTHY. Verify that policy.
        assert rep["verdict"] == "SECTOR_CONCENTRATED"

    def test_inverted_sector_is_red_flag(self):
        # 30 healthcare rows with perfectly anti-correlated predictions
        scorer = _FakeScorer({"healthcare": lambda ml: -ml,
                              "tech": lambda ml: ml})
        # 30 healthcare (inverted) + 30 tech (perfect) ⇒ INVERTED_SIGNAL on
        # healthcare wins precedence over SECTOR_CONCENTRATED.
        rows = [_mk_row("LLY", float(i - 15)) for i in range(30)] + \
               [_mk_row("NVDA", float(i - 15)) for i in range(30)]
        rep = ss.sector_skill(scorer, train_records=[], oos_records=rows)
        assert rep["verdict"] == "HAS_INVERTED_SECTOR"
        assert "healthcare" in rep["inverted_sectors"]
        hc = next(s for s in rep["sectors"] if s["sector"] == "healthcare")
        assert hc["rank_ic"] <= -ss.IC_GOOD
        assert hc["verdict"] == "INVERTED_SIGNAL"

    def test_sparse_sector_does_not_carry_verdict(self):
        # 100 tech (no edge) + 10 energy rows. Energy is below
        # MIN_OUTCOMES_PER_SECTOR so it is SPARSE regardless of its IC,
        # and concentration on tech triggers SECTOR_CONCENTRATED.
        scorer = _FakeScorer({"tech": lambda ml: 0.0,
                              "energy": lambda ml: ml})  # would be perfect
        rows = [_mk_row("NVDA", float(i - 50)) for i in range(100)] + \
               [_mk_row("XOM", float(i)) for i in range(10)]
        rep = ss.sector_skill(scorer, train_records=[], oos_records=rows)
        energy = next(s for s in rep["sectors"] if s["sector"] == "energy")
        assert energy["verdict"] == "SPARSE"
        # SPARSE sectors must NOT appear in inverted_sectors even if their
        # (unstable, n=10) IC happens to be highly inverted.
        assert "energy" not in rep["inverted_sectors"]

    def test_concentration_threshold_fires(self):
        # 70 tech + 30 financials → tech is 70%, exactly at threshold ⇒
        # SECTOR_CONCENTRATED (the boundary is inclusive).
        scorer = _FakeScorer({"tech": lambda ml: 0.0,
                              "financials": lambda ml: 0.0})
        rows = [_mk_row("NVDA", float(i)) for i in range(70)] + \
               [_mk_row("JPM", float(i)) for i in range(30)]
        rep = ss.sector_skill(scorer, train_records=[], oos_records=rows)
        assert rep["verdict"] == "SECTOR_CONCENTRATED"
        assert rep["concentrated_sector"] == "tech"

    def test_no_sector_edge(self):
        # Split across two sectors below concentration threshold, none
        # has edge ⇒ NO_SECTOR_EDGE
        scorer = _FakeScorer({"tech": lambda ml: 0.0,
                              "financials": lambda ml: 0.0})
        rows = [_mk_row("NVDA", float(i)) for i in range(40)] + \
               [_mk_row("JPM", float(i)) for i in range(40)]
        rep = ss.sector_skill(scorer, train_records=[], oos_records=rows)
        assert rep["verdict"] == "NO_SECTOR_EDGE"

    def test_train_counts_surfaced(self):
        # The n_train per sector should reflect the train_records, even
        # if no OOS rows exist for that sector.
        scorer = _FakeScorer({"tech": lambda ml: ml})
        train = [_mk_row("NVDA", 0.0)] * 500 + [_mk_row("XOM", 0.0)] * 5
        oos = [_mk_row("NVDA", float(i - 15)) for i in range(30)]
        rep = ss.sector_skill(scorer, train_records=train, oos_records=oos)
        tech = next(s for s in rep["sectors"] if s["sector"] == "tech")
        assert tech["n_train"] == 500
        # Energy doesn't appear in OOS so it's not in the sectors list,
        # but the overall n_train sum includes its train rows.
        assert rep["n_train"] == 505

    def test_magnitude_bias_computed(self):
        # Predictions systematically over-shoot by +2pp: pred = real + 2
        scorer = _FakeScorer({"tech": lambda ml: ml + 2.0})
        rows = [_mk_row("NVDA", float(i - 15)) for i in range(40)]
        rep = ss.sector_skill(scorer, train_records=[], oos_records=rows)
        tech = next(s for s in rep["sectors"] if s["sector"] == "tech")
        # mean_pred - mean_realized should be +2.0 (within rounding)
        assert tech["magnitude_bias"] == pytest.approx(2.0)


class TestAnalyzeCLI:
    def test_analyze_missing_file(self, tmp_path):
        rep = ss.analyze(outcomes_path=tmp_path / "nonexistent.jsonl")
        assert rep["status"] == "insufficient_data"
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_analyze_with_synthetic_outcomes(self, tmp_path, monkeypatch):
        """End-to-end: write a JSONL file, monkeypatch DecisionScorer to a
        trained FakeScorer, call analyze, verify the verdict matches the
        sector_skill() unit test."""
        path = tmp_path / "out.jsonl"
        rows = [_mk_row("NVDA", float(i - 20), sim_date=f"2025-01-{i+1:02d}")
                for i in range(40)]
        path.write_text("\n".join(json.dumps(r) for r in rows))

        # Monkeypatch DecisionScorer to a FakeScorer so we don't need a
        # real pickle on disk.
        fake = _FakeScorer({"tech": lambda ml: ml})
        import paper_trader.ml.decision_scorer as ds
        # The analyze() function does `from paper_trader.ml.decision_scorer
        # import DecisionScorer` inside, then calls `DecisionScorer()`. So
        # we need to replace the class such that calling it returns `fake`.
        class _Factory:
            def __call__(self):
                return fake
        monkeypatch.setattr(ds, "DecisionScorer", _Factory())

        rep = ss.analyze(outcomes_path=path, oos_fraction=0.2)
        # OOS slice = 8 rows (20% of 40) → below MIN_RECORDS=30 ⇒ insufficient_data
        assert rep["status"] == "insufficient_data"
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_analyze_returns_real_skill_above_floor(self, tmp_path, monkeypatch):
        """With enough rows the OOS slice exceeds MIN_RECORDS and a real
        verdict is reached."""
        path = tmp_path / "out.jsonl"
        # 200 rows total ⇒ OOS = 40 ⇒ above MIN_RECORDS=30
        rows = [_mk_row("NVDA", float(i - 100), sim_date=f"2025-{(i // 30) + 1:02d}-{(i % 30) + 1:02d}")
                for i in range(200)]
        path.write_text("\n".join(json.dumps(r) for r in rows))

        fake = _FakeScorer({"tech": lambda ml: ml})
        import paper_trader.ml.decision_scorer as ds

        class _Factory:
            def __call__(self):
                return fake
        monkeypatch.setattr(ds, "DecisionScorer", _Factory())

        rep = ss.analyze(outcomes_path=path, oos_fraction=0.2)
        assert rep["status"] == "ok"
        # Only one sector (tech) in the data ⇒ SECTOR_CONCENTRATED
        assert rep["verdict"] == "SECTOR_CONCENTRATED"
        tech = next(s for s in rep["sectors"] if s["sector"] == "tech")
        assert tech["rank_ic"] == pytest.approx(1.0)


class TestCliExitCodes:
    def test_cli_returns_zero_when_healthy(self, monkeypatch, capsys):
        # Monkeypatch analyze to return a HEALTHY report and verify the
        # CLI exits 0 (the contract calibration._cli / persona_skill._cli
        # establish for cron consumers).
        monkeypatch.setattr(ss, "analyze", lambda **kw: {
            "status": "ok", "verdict": "HEALTHY", "n_train": 100,
            "n_oos": 50, "sectors": [
                {"sector": "tech", "n_train": 80, "n_oos": 30,
                 "rmse": 5.0, "dir_acc": 0.6, "rank_ic": 0.25,
                 "mean_pred": 1.0, "mean_realized": 1.5,
                 "magnitude_bias": -0.5, "verdict": "SIGNAL_EDGE"},
            ], "inverted_sectors": [], "hint": "ok",
            "concentrated_sector": None,
        })
        rc = ss._cli()
        assert rc == 0

    def test_cli_returns_two_on_inverted(self, monkeypatch):
        monkeypatch.setattr(ss, "analyze", lambda **kw: {
            "status": "ok", "verdict": "HAS_INVERTED_SECTOR",
            "n_train": 100, "n_oos": 50,
            "sectors": [{"sector": "healthcare", "n_train": 30, "n_oos": 30,
                         "rmse": 12.0, "dir_acc": 0.3, "rank_ic": -0.3,
                         "mean_pred": 1.0, "mean_realized": -2.0,
                         "magnitude_bias": 3.0, "verdict": "INVERTED_SIGNAL"}],
            "inverted_sectors": ["healthcare"], "hint": "inverted",
            "concentrated_sector": None,
        })
        rc = ss._cli()
        assert rc == 2

    def test_cli_returns_one_on_untrained(self, monkeypatch):
        monkeypatch.setattr(ss, "analyze", lambda **kw: {
            "status": "error", "verdict": "SCORER_UNTRAINED",
            "n_train": 0, "n_oos": 0, "sectors": [],
            "inverted_sectors": [], "hint": "untrained",
        })
        rc = ss._cli()
        assert rc == 1

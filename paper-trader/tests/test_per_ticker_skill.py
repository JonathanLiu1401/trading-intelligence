"""Tests for paper_trader/ml/per_ticker_skill.py.

Validates the per-ticker OOS skill diagnostic produces the correct
verdicts on synthetic data with known per-ticker IC. Mirrors the
shape and discipline of tests/test_sector_skill.py so a future change
to the diagnostic contract must update one consistent test pattern.
"""
from __future__ import annotations

import json

import pytest

from paper_trader.ml import per_ticker_skill as pts


class _FakeScorer:
    """A trained-shape scorer whose predict() returns a ticker-conditional
    answer.

    ``preds_by_ticker``: maps ticker → callable that takes the realized 5d
    return (encoded into ``ml_score``) and returns a synthetic prediction.
    This lets a test stage an exact IC per ticker (e.g. perfectly
    correlated predictions on NVDA, anti-correlated on AMD).
    """

    is_trained = True

    def __init__(self, preds_by_ticker, default_pred=0.0):
        self._preds = preds_by_ticker
        self._default = default_pred
        self._n_train = 1000

    def predict(self, *, ml_score, rsi, macd, mom5, mom20, regime_mult,
                ticker, vol_ratio=None, bb_pos=None, news_urgency=None,
                news_article_count=None):
        fn = self._preds.get(str(ticker).upper())
        if fn is None:
            return self._default
        # Encode the realized return into ml_score so a synthetic test can
        # carry it through to predict and emit a controlled prediction.
        return fn(ml_score)


def _mk_row(ticker, fwd_5d, *, action="BUY", sim_date="2025-01-01",
            ml_score_override=None):
    """Build a decision_outcomes.jsonl-shaped row. ``ml_score`` defaults to
    ``fwd_5d`` so a FakeScorer that returns ``ml_score`` produces a
    perfectly-correlated prediction set for that ticker."""
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


class TestVerdictForTicker:
    def test_sparse_below_min(self):
        assert pts._verdict_for_ticker(n=5, ic=0.5) == "SPARSE"
        # Exact boundary — n == MIN should clear SPARSE
        assert pts._verdict_for_ticker(n=pts.MIN_OUTCOMES_PER_TICKER, ic=0.5) \
            == "SIGNAL_EDGE"
        # And n one below MIN is still SPARSE regardless of IC
        assert pts._verdict_for_ticker(
            n=pts.MIN_OUTCOMES_PER_TICKER - 1, ic=0.5) == "SPARSE"

    def test_inverted_below_minus_ic_good(self):
        assert pts._verdict_for_ticker(n=100, ic=-0.20) == "INVERTED_SIGNAL"
        # Exact boundary
        assert pts._verdict_for_ticker(n=100, ic=-pts.IC_GOOD) == "INVERTED_SIGNAL"

    def test_signal_edge_at_ic_good(self):
        assert pts._verdict_for_ticker(n=100, ic=pts.IC_GOOD) == "SIGNAL_EDGE"
        assert pts._verdict_for_ticker(n=100, ic=0.5) == "SIGNAL_EDGE"

    def test_weak_band(self):
        assert pts._verdict_for_ticker(n=100, ic=pts.IC_MIN) == "WEAK_SIGNAL_EDGE"
        # Just below IC_GOOD
        assert pts._verdict_for_ticker(
            n=100, ic=pts.IC_GOOD - 0.01) == "WEAK_SIGNAL_EDGE"

    def test_no_edge_band(self):
        assert pts._verdict_for_ticker(n=100, ic=0.0) == "NO_SIGNAL_EDGE"
        assert pts._verdict_for_ticker(
            n=100, ic=pts.IC_MIN - 0.01) == "NO_SIGNAL_EDGE"


class TestAlignedOosPair:
    def test_buy_passes_through(self):
        scorer = _FakeScorer({"NVDA": lambda ml: ml})
        row = _mk_row("NVDA", 5.0, action="BUY")
        pair = pts._aligned_oos_pair(row, scorer)
        assert pair == (5.0, 5.0)

    def test_sell_flips_realized_only(self):
        # Universal SELL sign-flip: realized is negated. The FakeScorer's
        # predict is fed the unchanged ml_score, so the prediction is
        # unchanged but the realized side is flipped — matches train_scorer.
        scorer = _FakeScorer({"NVDA": lambda ml: ml})
        row = _mk_row("NVDA", 5.0, action="SELL")
        pair = pts._aligned_oos_pair(row, scorer)
        assert pair == (5.0, -5.0)

    def test_none_forward_return_dropped(self):
        scorer = _FakeScorer({"NVDA": lambda ml: ml})
        row = _mk_row("NVDA", 0.0)
        row["forward_return_5d"] = None
        assert pts._aligned_oos_pair(row, scorer) is None

    def test_string_forward_return_dropped(self):
        # _to_float rejects strings → NaN → drop. A single poisoned line
        # cannot corrupt the report.
        scorer = _FakeScorer({"NVDA": lambda ml: ml})
        row = _mk_row("NVDA", 0.0)
        row["forward_return_5d"] = "not a number"
        assert pts._aligned_oos_pair(row, scorer) is None

    def test_scorer_exception_drops_row(self):
        class _Boom:
            is_trained = True

            def predict(self, **kw):
                raise RuntimeError("boom")

        row = _mk_row("NVDA", 5.0)
        assert pts._aligned_oos_pair(row, _Boom()) is None


class TestPerTickerSkill:
    def test_untrained_scorer_short_circuits(self):
        class _Untrained:
            is_trained = False

            def predict(self, **kw):
                return 0.0

        rep = pts.per_ticker_skill(_Untrained(), [], [])
        assert rep["verdict"] == "SCORER_UNTRAINED"
        assert rep["tickers"] == []
        assert rep["inverted_tickers"] == []

    def test_insufficient_data_when_below_min_records(self):
        # n_aligned < MIN_RECORDS even with a perfectly-correlated single
        # ticker still degrades to INSUFFICIENT_DATA — Spearman is not
        # stable on so few points.
        scorer = _FakeScorer({"NVDA": lambda ml: ml})
        rep = pts.per_ticker_skill(
            scorer, train_records=[],
            oos_records=[_mk_row("NVDA", float(i)) for i in range(5)],
        )
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_oos"] == 5

    def test_perfectly_predicted_ticker_yields_signal_edge(self):
        # 40 NVDA rows where pred == realized → Spearman = 1.0 → SIGNAL_EDGE
        # And 30 AMD rows with no prediction skill (constant 0) — so the
        # overall verdict is HEALTHY (≥1 SIGNAL_EDGE, no inverted).
        scorer = _FakeScorer({
            "NVDA": lambda ml: ml,    # perfect
            "AMD": lambda ml: 0.0,    # constant predictor — no rank skill
        })
        oos = (
            [_mk_row("NVDA", float(i - 20)) for i in range(40)]
            + [_mk_row("AMD", float((i % 10) - 5)) for i in range(30)]
        )
        rep = pts.per_ticker_skill(scorer, train_records=[], oos_records=oos)
        assert rep["status"] == "ok"
        nvda = next(t for t in rep["tickers"] if t["ticker"] == "NVDA")
        assert nvda["n_oos"] == 40
        assert nvda["rank_ic"] == pytest.approx(1.0)
        assert nvda["verdict"] == "SIGNAL_EDGE"
        # rmse is 0 because pred == realized exactly
        assert nvda["rmse"] == pytest.approx(0.0)
        # dir_acc must be 1.0 (every non-zero pair agrees on sign)
        assert nvda["dir_acc"] == pytest.approx(1.0)
        amd = next(t for t in rep["tickers"] if t["ticker"] == "AMD")
        # Constant predictor → IC=0 → NO_SIGNAL_EDGE
        assert amd["rank_ic"] == pytest.approx(0.0)
        assert amd["verdict"] == "NO_SIGNAL_EDGE"
        # Overall: has edge, no inverted ⇒ HEALTHY (no SECTOR_CONCENTRATED
        # equivalent at the ticker level by design)
        assert rep["verdict"] == "HEALTHY"

    def test_inverted_ticker_is_red_flag(self):
        # 30 NVDA rows where pred = -realized → Spearman = -1.0 →
        # INVERTED_SIGNAL → HAS_INVERTED_TICKER (the scorer is actively
        # WORSE than no scorer on NVDA).
        scorer = _FakeScorer({"NVDA": lambda ml: -ml})
        oos = [_mk_row("NVDA", float(i - 15)) for i in range(30)]
        rep = pts.per_ticker_skill(scorer, train_records=[], oos_records=oos)
        assert rep["status"] == "ok"
        nvda = next(t for t in rep["tickers"] if t["ticker"] == "NVDA")
        assert nvda["rank_ic"] == pytest.approx(-1.0)
        assert nvda["verdict"] == "INVERTED_SIGNAL"
        assert nvda["dir_acc"] == pytest.approx(0.0)  # every sign wrong
        assert rep["inverted_tickers"] == ["NVDA"]
        assert rep["verdict"] == "HAS_INVERTED_TICKER"
        assert "NVDA" in rep["hint"]

    def test_inverted_outranks_no_edge_in_verdict(self):
        # Mix of an inverted ticker and a no-edge ticker. The inverted is
        # the red flag and must dominate (an inverted ticker inside an
        # otherwise no-edge population is still inverted; the gate would
        # be actively harmful on it).
        scorer = _FakeScorer({
            "NVDA": lambda ml: -ml,    # inverted
            "AMD": lambda ml: 0.0,      # no rank skill
        })
        oos = (
            [_mk_row("NVDA", float(i - 15)) for i in range(30)]
            + [_mk_row("AMD", float(i)) for i in range(30)]
        )
        rep = pts.per_ticker_skill(scorer, train_records=[], oos_records=oos)
        assert rep["verdict"] == "HAS_INVERTED_TICKER"
        assert "NVDA" in rep["inverted_tickers"]

    def test_no_ticker_edge_when_all_predictions_constant(self):
        # No ticker has rank-IC ≥ IC_GOOD on a stable sample ⇒ NO_TICKER_EDGE
        scorer = _FakeScorer({}, default_pred=0.0)
        oos = (
            [_mk_row("NVDA", float(i)) for i in range(30)]
            + [_mk_row("AMD", float(i)) for i in range(30)]
        )
        rep = pts.per_ticker_skill(scorer, train_records=[], oos_records=oos)
        assert rep["verdict"] == "NO_TICKER_EDGE"
        # All tickers landed at SPARSE-or-NO_SIGNAL_EDGE
        for t in rep["tickers"]:
            assert t["verdict"] in ("SPARSE", "NO_SIGNAL_EDGE",
                                    "WEAK_SIGNAL_EDGE")

    def test_sparse_ticker_visible_but_not_in_inverted_or_signal(self):
        # 35 NVDA rows (above MIN) with perfect IC + 15 AMD rows (below
        # MIN_OUTCOMES_PER_TICKER). NVDA is SIGNAL_EDGE, AMD is SPARSE.
        scorer = _FakeScorer({
            "NVDA": lambda ml: ml,
            "AMD": lambda ml: ml,
        })
        oos = (
            [_mk_row("NVDA", float(i - 17)) for i in range(35)]
            + [_mk_row("AMD", float(i - 7)) for i in range(15)]
        )
        rep = pts.per_ticker_skill(scorer, train_records=[], oos_records=oos)
        nvda = next(t for t in rep["tickers"] if t["ticker"] == "NVDA")
        amd = next(t for t in rep["tickers"] if t["ticker"] == "AMD")
        assert nvda["verdict"] == "SIGNAL_EDGE"
        assert amd["verdict"] == "SPARSE"
        # SPARSE sinks to the bottom of the sort regardless of its
        # (unstable, small-n) IC
        last = rep["tickers"][-1]
        assert last["verdict"] == "SPARSE"
        # SPARSE is NOT counted in inverted_tickers even though it is not
        # in SIGNAL_EDGE
        assert "AMD" not in rep["inverted_tickers"]

    def test_n_train_count_surfaced(self):
        # train_records carries the per-ticker training-row count surfaced
        # in the report. The diagnostic is read-only so this is only for
        # operator visibility.
        scorer = _FakeScorer({"NVDA": lambda ml: ml})
        train = [_mk_row("NVDA", 1.0) for _ in range(100)]
        oos = [_mk_row("NVDA", float(i - 17)) for i in range(35)]
        rep = pts.per_ticker_skill(scorer, train_records=train,
                                   oos_records=oos)
        nvda = next(t for t in rep["tickers"] if t["ticker"] == "NVDA")
        assert nvda["n_train"] == 100
        assert rep["n_train"] == 100

    def test_ticker_case_normalised_in_buckets(self):
        # Mixed-case external rows must map to the same ticker bucket
        # (upper-cased) so an importer that didn't normalise can't fragment
        # the per-ticker aggregate.
        scorer = _FakeScorer({"NVDA": lambda ml: ml})
        oos = (
            [_mk_row("NVDA", float(i - 15)) for i in range(20)]
            + [_mk_row("nvda", float(i - 5)) for i in range(15)]
        )
        rep = pts.per_ticker_skill(scorer, train_records=[], oos_records=oos)
        # Both 20 + 15 = 35 rows land in the single NVDA bucket
        nvda = next(t for t in rep["tickers"] if t["ticker"] == "NVDA")
        assert nvda["n_oos"] == 35

    def test_empty_ticker_rows_dropped(self):
        # A row with no ticker has no per-ticker meaning — dropped from
        # the aggregate. Must not crash and must not occupy a bucket
        # (which would then sort meaninglessly).
        scorer = _FakeScorer({"NVDA": lambda ml: ml})
        oos = (
            [_mk_row("", float(i)) for i in range(15)]
            + [_mk_row("NVDA", float(i - 15)) for i in range(30)]
        )
        rep = pts.per_ticker_skill(scorer, train_records=[], oos_records=oos)
        # Only NVDA bucket survives
        tk_names = [t["ticker"] for t in rep["tickers"]]
        assert "NVDA" in tk_names
        assert "" not in tk_names
        # n_aligned reflects only NVDA's 30 rows
        assert rep["n_oos"] == 30

    def test_report_capped_at_max_tickers(self):
        # When unique tickers exceed MAX_TICKERS_IN_REPORT, the report
        # caps the list but keeps inverted_tickers complete (a red-flag
        # name far down the sort is NEVER dropped from the inverted list).
        scorer = _FakeScorer({}, default_pred=0.0)
        # MAX_TICKERS_IN_REPORT + 5 buckets, each with MIN_OUTCOMES rows
        n_extra = 5
        oos = []
        for i in range(pts.MAX_TICKERS_IN_REPORT + n_extra):
            tk = f"T{i:03d}"
            for j in range(pts.MIN_OUTCOMES_PER_TICKER):
                oos.append(_mk_row(tk, float(j - 10)))
        rep = pts.per_ticker_skill(scorer, train_records=[], oos_records=oos)
        assert len(rep["tickers"]) == pts.MAX_TICKERS_IN_REPORT
        assert rep["tickers_truncated"] is True
        assert (rep["n_unique_tickers_oos"]
                == pts.MAX_TICKERS_IN_REPORT + n_extra)


class TestLoadOutcomes:
    def test_missing_file_returns_empty_list(self, tmp_path):
        # Missing file must NOT raise — a fresh checkout has no
        # decision_outcomes.jsonl. Mirrors sector_skill / persona_skill.
        out = pts._load_outcomes(tmp_path / "nope.jsonl")
        assert out == []

    def test_skips_unparseable_lines(self, tmp_path):
        p = tmp_path / "out.jsonl"
        p.write_text('\n'.join([
            '{"ticker": "NVDA", "forward_return_5d": 3.0}',
            'this is not JSON',
            '{"ticker": "AMD", "forward_return_5d": -2.0}',
            '   ',  # blank line
            '{"ticker": "TQQQ"}',  # parses, ticker present
        ]))
        rows = pts._load_outcomes(p)
        assert len(rows) == 3
        assert rows[0]["ticker"] == "NVDA"
        assert rows[1]["ticker"] == "AMD"
        assert rows[2]["ticker"] == "TQQQ"

    def test_non_dict_top_level_lines_dropped(self, tmp_path):
        # A list / string at the JSON top level isn't a record — silently
        # drop rather than crash downstream consumers that assume dict shape.
        p = tmp_path / "out.jsonl"
        p.write_text('\n'.join([
            '{"ticker": "NVDA"}',
            '["not", "a", "record"]',
            '"a bare string"',
        ]))
        rows = pts._load_outcomes(p)
        assert len(rows) == 1


class TestAnalyzeEndToEnd:
    def test_no_records_yields_insufficient_data(self, tmp_path):
        # The "fresh checkout / no outcomes yet" path. Must produce an
        # honest INSUFFICIENT_DATA report, never raise.
        rep = pts.analyze(outcomes_path=tmp_path / "nope.jsonl")
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_oos"] == 0


class TestCli:
    """CLI exit-code contract: callers (cron jobs / scripts) branch on $?."""

    def test_exit_code_for_inverted_ticker(self, tmp_path, monkeypatch, capsys):
        # Stage analyze() to return HAS_INVERTED_TICKER and verify rc=2.
        def _fake_analyze(*a, **kw):
            return {
                "status": "ok",
                "verdict": "HAS_INVERTED_TICKER",
                "n_train": 100, "n_oos": 50,
                "n_unique_tickers_oos": 2,
                "tickers": [{
                    "ticker": "NVDA", "n_train": 50, "n_oos": 30,
                    "rmse": 8.0, "dir_acc": 0.10, "rank_ic": -0.4,
                    "mean_pred": 1.5, "mean_realized": -1.0,
                    "magnitude_bias": 2.5, "verdict": "INVERTED_SIGNAL",
                }],
                "tickers_truncated": False,
                "inverted_tickers": ["NVDA"],
                "hint": "test inverted",
            }

        monkeypatch.setattr(pts, "analyze", _fake_analyze)
        rc = pts._cli()
        assert rc == 2

    def test_exit_code_for_healthy(self, monkeypatch):
        def _fake_analyze(*a, **kw):
            return {
                "status": "ok", "verdict": "HEALTHY",
                "n_train": 100, "n_oos": 50, "n_unique_tickers_oos": 2,
                "tickers": [], "tickers_truncated": False,
                "inverted_tickers": [], "hint": "ok",
            }

        monkeypatch.setattr(pts, "analyze", _fake_analyze)
        assert pts._cli() == 0

    def test_exit_code_for_untrained(self, monkeypatch):
        def _fake_analyze(*a, **kw):
            return {
                "status": "error", "verdict": "SCORER_UNTRAINED",
                "n_train": 0, "n_oos": 0, "tickers": [],
                "tickers_truncated": False,
                "inverted_tickers": [],
                "hint": "untrained",
            }

        monkeypatch.setattr(pts, "analyze", _fake_analyze)
        assert pts._cli() == 1

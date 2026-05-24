"""HYBRID pass — Agent 2 (ML+backtests), 2026-05-23.

Locks two classes of behaviour found by this review:

1. ``_raw_to_calibrated`` symmetric collapsed-label-quantiles guard. The
   sibling ``_raw_to_percentile`` already short-circuits to ``None`` when
   the persisted ``pred_quantiles`` table collapses to a single value (the
   2026-05-23 finding #1 footprint — synthetic n=39 clobber). The same
   degeneracy on ``label_quantiles`` had NO guard: ``np.interp`` over a
   constant ``fp`` array silently returned that constant for every rank,
   so a degenerate single-label corpus surfaced as e.g. ``calibrated=7.0%``
   for every prediction regardless of input rank — fake magnitude with no
   honest "I can't tell" sentinel for downstream consumers. The fix mirrors
   the pred-side guard; these tests pin both the now-correct None behaviour
   and the unchanged healthy path.

2. End-to-end ``predict_with_meta`` cascade: a collapsed-``label_quantiles``
   pickle must surface ``calibrated=None`` in the consumer-visible dict the
   dashboard / honesty panels read, exactly like the existing
   pred-quantiles-collapsed cascade test does.
"""
from __future__ import annotations

import numpy as np
import pytest

from paper_trader.ml.decision_scorer import DecisionScorer, train_scorer


@pytest.fixture
def trained_scorer():
    """Train a real scorer on a synthetic dataset whose forward returns vary
    over a meaningful range, so the resulting label_quantiles span a real
    (non-collapsed) interval — the same idiom test_scorer_calibrated uses."""
    rng = np.random.default_rng(42)
    records = []
    for i in range(120):
        # mix of tickers / actions / realistic ranges so both pred_quantiles
        # and label_quantiles are non-degenerate. The (ticker, sim_date,
        # action) dedup key needs to be unique across all 120 — use the
        # full integer index as the day fraction so no two records collide.
        records.append({
            "ticker": "NVDA" if i % 2 == 0 else "AAPL",
            "sim_date": (f"2025-{((i // 28) % 12) + 1:02d}-"
                         f"{(i % 28) + 1:02d}"),
            "action": "BUY",
            "ml_score": float(rng.uniform(-2, 4)),
            "rsi": float(rng.uniform(30, 75)),
            "macd": float(rng.uniform(-1, 1)),
            "mom5": float(rng.uniform(-10, 10)),
            "mom20": float(rng.uniform(-15, 15)),
            "regime_mult": rng.choice([0.3, 0.6, 1.0]),
            "vol_ratio": float(rng.uniform(0.5, 2.5)),
            "bb_position": float(rng.uniform(-2, 2)),
            "news_urgency": float(rng.uniform(0, 100)),
            "news_article_count": float(rng.integers(1, 10)),
            "forward_return_5d": float(rng.normal(0.5, 8)),
            "return_pct": float(rng.uniform(-50, 200)),
        })
    res = train_scorer(records)
    assert res.get("status") == "ok"
    return DecisionScorer()


class TestRawToCalibratedCollapsedLabelGuard:
    """Symmetric defense-in-depth to ``TestCollapsedQuantileGuard`` in
    test_scorer_percentile.py: the collapsed-table footprint can hit either
    side of the quantile pair, and only the pred-side previously degraded
    to None. The honest contract is the same on both sides."""

    def test_raw_to_calibrated_none_when_label_quantiles_collapsed(self):
        """A label_quantiles table with every entry equal must return None.
        With the bug, np.interp returned the constant value for every rank,
        fabricating a fixed calibrated magnitude across the entire dashboard."""
        s = DecisionScorer()
        s._trained = True
        s._n_train = 100
        s._pred_quantiles = np.asarray(np.linspace(-5.0, 5.0, 101),
                                       dtype=np.float64)
        # 101-entry collapsed LABEL table — a degenerate single-label corpus
        # or a synthetic train_scorer call whose forward_return_5d were all
        # equal (e.g., all clamped to the same edge).
        s._label_quantiles = np.asarray([7.0] * 101, dtype=np.float64)
        # Sanity: percentile is still finite (pred_quantiles non-collapsed)
        # so the regression is specifically on the calibrated cascade.
        assert s._raw_to_percentile(-3.0) is not None
        assert s._raw_to_percentile(0.0) is not None
        assert s._raw_to_percentile(3.0) is not None
        # Each raw value previously returned 7.0 silently; now None.
        for raw in (-50.0, -3.0, 0.0, 3.0, 18.934, 50.0):
            assert s._raw_to_calibrated(raw) is None, raw

    def test_raw_to_calibrated_works_with_healthy_label_quantiles(self):
        """Counterfactual: an honest non-collapsed label table continues to
        produce a finite calibrated value. Proves the guard fires ONLY on
        the degenerate case."""
        s = DecisionScorer()
        s._trained = True
        s._n_train = 100
        s._pred_quantiles = np.asarray(np.linspace(-5.0, 5.0, 101),
                                       dtype=np.float64)
        s._label_quantiles = np.asarray(np.linspace(-10.0, 10.0, 101),
                                        dtype=np.float64)
        # Median raw → median label.
        assert s._raw_to_calibrated(0.0) == pytest.approx(0.0, abs=0.5)
        # Top-rank raw → top-rank label.
        assert s._raw_to_calibrated(5.0) == pytest.approx(10.0, abs=0.5)
        # Bottom-rank raw → bottom-rank label.
        assert s._raw_to_calibrated(-5.0) == pytest.approx(-10.0, abs=0.5)

    def test_predict_with_meta_calibrated_none_on_collapsed_label_quantiles(
            self, trained_scorer):
        """End-to-end cascade: the consumer-visible `calibrated` field
        surfaces None when label_quantiles are collapsed, even though
        `pred` (the scorer's scalar output) is unaffected (the gate reads
        `pred`, not `calibrated`, so this is a diagnostic field that must
        degrade honestly — same contract as the pred-side guard).
        """
        scorer = trained_scorer
        # Clobber just the label side. predict() / percentile / failed
        # should remain correct; only calibrated should None out.
        scorer._label_quantiles = np.asarray([3.5] * 101, dtype=np.float64)
        meta = scorer.predict_with_meta(
            ml_score=2.0, rsi=55.0, macd=0.3, mom5=2.0, mom20=4.0,
            regime_mult=1.0, ticker="NVDA",
        )
        assert meta["calibrated"] is None
        # Percentile and pred contract intact.
        assert meta["percentile"] is not None
        assert isinstance(meta["pred"], float)
        assert meta["failed"] is False

    def test_predict_calibrated_helper_also_none_on_collapsed_labels(
            self, trained_scorer):
        """The convenience helper ``predict_calibrated`` must mirror the
        ``predict_with_meta['calibrated']`` cascade — calling sites that
        read the helper directly (e.g. backwards-compat dashboard panels)
        must not see a fabricated number."""
        scorer = trained_scorer
        scorer._label_quantiles = np.asarray([-2.5] * 101, dtype=np.float64)
        val = scorer.predict_calibrated(
            ml_score=2.0, rsi=55.0, macd=0.3, mom5=2.0, mom20=4.0,
            regime_mult=1.0, ticker="NVDA",
        )
        assert val is None

    def test_collapsed_both_sides_still_none(self):
        """Both pred AND label collapsed (the worst-case clobber): the
        result is still a single None — never raises, never falls back."""
        s = DecisionScorer()
        s._trained = True
        s._n_train = 100
        s._pred_quantiles = np.asarray([1.5] * 101, dtype=np.float64)
        s._label_quantiles = np.asarray([2.5] * 101, dtype=np.float64)
        for raw in (-5.0, 0.0, 1.5, 5.0):
            assert s._raw_to_calibrated(raw) is None
            assert s._raw_to_percentile(raw) is None

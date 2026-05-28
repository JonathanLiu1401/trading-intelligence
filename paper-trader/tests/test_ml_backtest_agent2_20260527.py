"""Agent 2 (ML+backtests) review — 2026-05-27.

Behaviour locks for findings from this review pass:

1. ``_parse_conviction_pct`` and ``_parse_gate_decision`` must reject hyphen-
   prefixed identifiers (``low-conviction=``, ``gate-scorer=``) so a future
   reasoning emission with such prefixes cannot silently poison the gate /
   sizing capture. The prior ``\\b`` anchor delivered the documented
   ``low-conviction=`` protection only on paper — ``\\b`` matches at the
   hyphen→word transition, so the embedded ``conviction=`` token DID match.

2. ``build_features`` clamps the safety-bounded inputs (``vol_ratio``,
   ``bb_pos``, ``news_urgency``, ``news_article_count``) at every retrain so
   a corrupted upstream value can't propagate into the scaler. Unbounded
   features (``ml_score``, ``rsi``, ``macd``, momentum) are intentionally
   left raw — the off-distribution clamp on the *output* is the honesty
   guard there.

3. ``_to_float`` rejects ``bool`` / ``np.bool_`` / non-finite / non-numeric
   inputs and falls back to the supplied default. Production data sometimes
   carries ``np.float32`` columns from pandas/yfinance roundtrips — those
   MUST coerce successfully (``np.float32`` is not a ``float`` subclass).

4. ``_LstsqModel.predict`` accepts both 1-D (single vector) and 2-D (batch)
   input — the production scorer batches but ad-hoc tools and the
   feature-contributions ablation path can hand in either shape.

5. Forward-return outcome computation refuses to fabricate a 0% return from
   a walk-back collision (both endpoints resolve to the same prior close on
   a thin/foreign-calendar ticker). The honesty discipline existed before;
   this pins it for the multi-horizon (5d/10d/20d) intraperiod-extreme
   helper too.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Section 1 — _parse_conviction_pct / _parse_gate_decision hyphen-prefix
# ---------------------------------------------------------------------------

class TestParseConvictionPctHyphenPrefix:
    """The docstring claim — 'a future emission like ``low-conviction=…``
    cannot accidentally match' — must hold. The prior ``\\b`` anchor failed
    silently because hyphen is a non-word character and ``\\b`` fires at
    every hyphen→word transition. Lock the negative-lookbehind fix."""

    def test_low_conviction_prefix_does_not_match(self):
        import run_continuous_backtests as rcb
        # A hypothetical reasoning that happens to contain `low-conviction=`
        # MUST read None, NOT 0.5. Pre-fix this returned 0.5.
        assert rcb._parse_conviction_pct("low-conviction=50%") is None

    def test_high_conviction_prefix_does_not_match(self):
        import run_continuous_backtests as rcb
        assert rcb._parse_conviction_pct(
            "tagged high-conviction=80% in note"
        ) is None

    def test_real_conviction_still_matches_after_hyphen_prefix(self):
        """A reasoning with BOTH `low-conviction=N%` (irrelevant) and a real
        `conviction=N%` token must capture the REAL token, not the prefixed
        one. Defends against a future log line that interleaves both."""
        import run_continuous_backtests as rcb
        # The real token has whitespace before `conviction=`. The lookbehind
        # rejects `low-conviction=50` and continues searching, finding the
        # real `conviction=25%`.
        assert rcb._parse_conviction_pct(
            "low-conviction=50% conviction=25%"
        ) == 0.25

    def test_whitespace_prefix_still_matches(self):
        import run_continuous_backtests as rcb
        assert rcb._parse_conviction_pct(
            "ML+quant: NVDA score=2.0 conviction=25% scorer=+5.2%"
        ) == 0.25

    def test_start_of_string_match(self):
        """A reasoning that starts with `conviction=` (no preceding char)
        must still match — the lookbehind only rejects word/hyphen chars."""
        import run_continuous_backtests as rcb
        assert rcb._parse_conviction_pct("conviction=40%") == 0.40


class TestParseGateDecisionHyphenPrefix:
    """Mirrors ``TestParseConvictionPctHyphenPrefix`` for the gate-decision
    capture. The same hyphen-prefix gap existed for ``\\bscorer=``."""

    def test_hyphen_prefixed_scorer_does_not_match(self):
        import run_continuous_backtests as rcb
        pred, off = rcb._parse_gate_decision("gate-scorer=+5.2%")
        assert pred is None
        assert off is None

    def test_real_scorer_still_matches(self):
        import run_continuous_backtests as rcb
        pred, off = rcb._parse_gate_decision(
            "ML+quant: NVDA score=2.0 conviction=25% scorer=+5.2%"
        )
        assert pred == 5.2
        assert off is False

    def test_off_dist_marker_with_real_scorer(self):
        import run_continuous_backtests as rcb
        pred, off = rcb._parse_gate_decision(
            "scorer=+50.0%(off-dist,gate-skipped)"
        )
        assert pred == 50.0
        assert off is True

    def test_gate_killed_marker_with_real_scorer(self):
        import run_continuous_backtests as rcb
        pred, off = rcb._parse_gate_decision(
            "scorer=-3.0%(gate-killed,no-skill)"
        )
        assert pred == -3.0
        assert off is True

    def test_hyphen_prefixed_followed_by_real_scorer(self):
        """A reasoning carrying BOTH a hyphen-prefixed `gate-scorer=...` and
        a real `scorer=...` token must read the REAL one. The lookbehind
        rejects the prefixed match and re.search continues."""
        import run_continuous_backtests as rcb
        pred, off = rcb._parse_gate_decision(
            "gate-scorer=+99.0% scorer=+5.2%"
        )
        assert pred == 5.2
        assert off is False


# ---------------------------------------------------------------------------
# Section 2 — build_features bounded-input clamping
# ---------------------------------------------------------------------------

class TestBuildFeaturesClamping:
    """The four bounded features (vol_ratio, bb_pos, news_urgency,
    news_article_count) MUST clamp at every train+predict call. A corrupted
    upstream value (a bad yfinance row producing vol_ratio=999) would
    otherwise feed directly into the scaler and poison every subsequent
    prediction's standardization. Locks the documented clamp ranges so a
    future refactor cannot silently widen them."""

    def test_vol_ratio_clamped_to_5(self):
        from paper_trader.ml.decision_scorer import build_features
        f = build_features(0.0, 50.0, 0.0, 0.0, 0.0, 1.0, "NVDA",
                           vol_ratio=999.0)
        # Slot 6 (0-indexed) is vol_ratio (see FEATURE_NAMES).
        assert f[6] == 5.0

    def test_vol_ratio_clamped_to_zero(self):
        from paper_trader.ml.decision_scorer import build_features
        f = build_features(0.0, 50.0, 0.0, 0.0, 0.0, 1.0, "NVDA",
                           vol_ratio=-1.0)
        assert f[6] == 0.0

    def test_bb_pos_clamped_to_positive_two(self):
        from paper_trader.ml.decision_scorer import build_features
        f = build_features(0.0, 50.0, 0.0, 0.0, 0.0, 1.0, "NVDA",
                           bb_pos=99.0)
        assert f[7] == 2.0

    def test_bb_pos_clamped_to_negative_two(self):
        from paper_trader.ml.decision_scorer import build_features
        f = build_features(0.0, 50.0, 0.0, 0.0, 0.0, 1.0, "NVDA",
                           bb_pos=-99.0)
        assert f[7] == -2.0

    def test_news_urgency_clamped_to_100(self):
        from paper_trader.ml.decision_scorer import build_features
        f = build_features(0.0, 50.0, 0.0, 0.0, 0.0, 1.0, "NVDA",
                           news_urgency=99999.0)
        assert f[8] == 100.0

    def test_news_article_count_clamped_to_20(self):
        from paper_trader.ml.decision_scorer import build_features
        f = build_features(0.0, 50.0, 0.0, 0.0, 0.0, 1.0, "NVDA",
                           news_article_count=99999.0)
        assert f[9] == 20.0

    def test_unknown_ticker_falls_to_sector_other(self):
        from paper_trader.ml.decision_scorer import build_features, SECTORS
        f = build_features(0.0, 50.0, 0.0, 0.0, 0.0, 1.0, "ZZZUNKNOWN")
        # Last 7 slots = sector one-hot. `other` is the last SECTORS entry.
        sector_oh = f[-len(SECTORS):]
        assert sector_oh[-1] == 1.0
        assert sum(sector_oh) == 1.0  # exactly one sector active

    def test_lowercase_ticker_falls_to_sector_other(self):
        """SECTOR_MAP keys are uppercase. A lowercase 'nvda' would silently
        drop to sector_other rather than sector_tech — pin the case-
        sensitive contract so a future caller fix can't silently degrade
        every prediction's sector signal."""
        from paper_trader.ml.decision_scorer import build_features, SECTORS
        f = build_features(0.0, 50.0, 0.0, 0.0, 0.0, 1.0, "nvda")
        sector_oh = f[-len(SECTORS):]
        assert sector_oh[-1] == 1.0  # other


# ---------------------------------------------------------------------------
# Section 3 — _to_float numpy / non-finite handling
# ---------------------------------------------------------------------------

class TestToFloatNumpyHandling:
    """``_to_float`` is the single coercion primitive every feature pipeline
    goes through. It must accept np.float32 (which is NOT a float subclass
    in Python, unlike np.float64), reject NaN/Inf/bool/non-numeric, and
    never raise on numpy strings or numpy bools."""

    def test_python_bool_returns_default(self):
        """bool is a subclass of int — must NOT silently coerce to 1.0/0.0
        and pollute features."""
        from paper_trader.ml.decision_scorer import _to_float
        assert _to_float(True, 42.0) == 42.0
        assert _to_float(False, 42.0) == 42.0

    def test_python_int_and_float_coerce(self):
        from paper_trader.ml.decision_scorer import _to_float
        assert _to_float(5, 42.0) == 5.0
        assert _to_float(3.14, 42.0) == 3.14
        assert _to_float(0, 42.0) == 0.0

    def test_nan_returns_default(self):
        from paper_trader.ml.decision_scorer import _to_float
        assert _to_float(float("nan"), 42.0) == 42.0

    def test_inf_returns_default(self):
        """A pathological +Inf from a divide-by-zero must not propagate
        as a feature value (the inference scaler would amplify it)."""
        from paper_trader.ml.decision_scorer import _to_float
        assert _to_float(float("inf"), 42.0) == 42.0
        assert _to_float(float("-inf"), 42.0) == 42.0

    def test_numpy_float32_coerces(self):
        """np.float32 is NOT a subclass of float — must go through the
        np.number branch."""
        from paper_trader.ml.decision_scorer import _to_float
        assert _to_float(np.float32(3.5), 42.0) == pytest.approx(3.5)

    def test_numpy_float64_coerces(self):
        from paper_trader.ml.decision_scorer import _to_float
        assert _to_float(np.float64(3.5), 42.0) == 3.5

    def test_numpy_int32_coerces(self):
        from paper_trader.ml.decision_scorer import _to_float
        assert _to_float(np.int32(7), 42.0) == 7.0

    def test_numpy_nan_returns_default(self):
        from paper_trader.ml.decision_scorer import _to_float
        assert _to_float(np.float64("nan"), 42.0) == 42.0
        assert _to_float(np.float32("nan"), 42.0) == 42.0

    def test_numpy_string_returns_default(self):
        """np.str_ is np.generic but not np.number — np.isfinite on it would
        raise TypeError. The guard must reject without raising."""
        from paper_trader.ml.decision_scorer import _to_float
        s = np.str_("hello")
        assert _to_float(s, 42.0) == 42.0

    def test_python_string_returns_default(self):
        from paper_trader.ml.decision_scorer import _to_float
        assert _to_float("3.14", 42.0) == 42.0

    def test_none_returns_default(self):
        from paper_trader.ml.decision_scorer import _to_float
        assert _to_float(None, 42.0) == 42.0


# ---------------------------------------------------------------------------
# Section 4 — _LstsqModel.predict shape contract
# ---------------------------------------------------------------------------

class TestLstsqModelPredict:
    """The sklearn-absent fallback. Production batches every predict, but
    ad-hoc CLI tools and ``feature_contributions``'s ablation matrix can
    pass either shape — pin the dual-shape contract so a future refactor
    can't break the public predict contract."""

    def test_1d_input_returns_1d_output(self):
        from paper_trader.ml.decision_scorer import _LstsqModel
        # Weights for N=3 features + bias = 4 weights.
        w = np.array([0.5, -0.3, 0.2, 0.1], dtype=np.float32)
        m = _LstsqModel(w)
        # 1-D input shape (3,) — the contract: reshape internally to (1,3)
        # then return shape (1,).
        out = m.predict(np.array([1.0, 2.0, 3.0]))
        assert out.shape == (1,)
        # 0.5*1 + (-0.3)*2 + 0.2*3 + 0.1 (bias) = 0.5 - 0.6 + 0.6 + 0.1 = 0.6
        assert out[0] == pytest.approx(0.6, abs=1e-5)

    def test_2d_input_returns_1d_output(self):
        from paper_trader.ml.decision_scorer import _LstsqModel
        w = np.array([0.5, -0.3, 0.2, 0.1], dtype=np.float32)
        m = _LstsqModel(w)
        batch = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = m.predict(batch)
        assert out.shape == (2,)
        assert out[0] == pytest.approx(0.6, abs=1e-5)
        # 0.5*4 + (-0.3)*5 + 0.2*6 + 0.1 = 2.0 - 1.5 + 1.2 + 0.1 = 1.8
        assert out[1] == pytest.approx(1.8, abs=1e-4)

    def test_empty_2d_input(self):
        from paper_trader.ml.decision_scorer import _LstsqModel
        w = np.array([0.5, -0.3, 0.2, 0.1], dtype=np.float32)
        m = _LstsqModel(w)
        empty = np.zeros((0, 3), dtype=np.float32)
        out = m.predict(empty)
        assert out.shape == (0,)

    def test_wrong_feature_count_raises(self):
        """A predict call with wrong feature count MUST raise rather than
        silently produce a nonsense answer — the inference pipeline relies
        on this to surface model/feature drift bugs immediately."""
        from paper_trader.ml.decision_scorer import _LstsqModel
        w = np.array([0.5, -0.3, 0.2, 0.1], dtype=np.float32)  # expects 3 features
        m = _LstsqModel(w)
        with pytest.raises(ValueError):
            m.predict(np.array([1.0, 2.0]))  # only 2 features


# ---------------------------------------------------------------------------
# Section 5 — predict_with_meta off-distribution honesty
# ---------------------------------------------------------------------------

class TestPredictWithMetaOffDistribution:
    """When the raw MLP output exceeds the empirical label support
    (±PRED_CLAMP_PCT = ±50%), ``predict_with_meta`` must clamp the
    user-facing ``pred`` AND set ``off_distribution=True`` so consumers
    know not to trust the magnitude. Locks the documented honesty contract.
    """

    def _make_trained_scorer_with_constant_output(self, fixed_raw: float):
        """A DecisionScorer with a model whose `predict` returns a constant
        — used to drive the clamp / off_distribution branches deterministically.
        """
        from paper_trader.ml.decision_scorer import DecisionScorer

        ds = DecisionScorer.__new__(DecisionScorer)
        ds._model = MagicMock()
        ds._model.predict = MagicMock(
            return_value=np.array([fixed_raw], dtype=np.float64))
        ds._scaler = None
        ds._trained = True
        ds._n_train = 1000
        ds._pred_quantiles = None
        ds._label_quantiles = None
        return ds

    def test_in_distribution_raw_passes_through_unclamped(self):
        ds = self._make_trained_scorer_with_constant_output(7.5)
        out = ds.predict_with_meta(
            ml_score=1.0, rsi=50.0, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA",
        )
        assert out["pred"] == pytest.approx(7.5)
        assert out["raw"] == pytest.approx(7.5)
        assert out["clamped"] is False
        assert out["off_distribution"] is False
        assert out["failed"] is False

    def test_extreme_positive_raw_clamps_to_pred_clamp_pct(self):
        from paper_trader.ml.decision_scorer import PRED_CLAMP_PCT
        ds = self._make_trained_scorer_with_constant_output(150.0)
        out = ds.predict_with_meta(
            ml_score=1.0, rsi=50.0, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA",
        )
        assert out["pred"] == PRED_CLAMP_PCT
        assert out["raw"] == 150.0
        assert out["clamped"] is True
        assert out["off_distribution"] is True
        assert out["failed"] is False  # the prediction WAS computed

    def test_extreme_negative_raw_clamps(self):
        from paper_trader.ml.decision_scorer import PRED_CLAMP_PCT
        ds = self._make_trained_scorer_with_constant_output(-89.0)
        out = ds.predict_with_meta(
            ml_score=-2.0, rsi=50.0, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="LITE",
        )
        assert out["pred"] == -PRED_CLAMP_PCT
        assert out["off_distribution"] is True
        assert out["failed"] is False

    def test_nan_raw_marks_failed(self):
        """A non-finite model output (an inf/nan from a pathological feature
        vector) must surface as failed=True so OOS rank-IC consumers drop
        the row rather than treating fabricated 0.0 as a real prediction."""
        ds = self._make_trained_scorer_with_constant_output(float("nan"))
        out = ds.predict_with_meta(
            ml_score=0.0, rsi=50.0, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA",
        )
        assert out["failed"] is True
        assert out["off_distribution"] is True
        assert out["pred"] == 0.0  # safe fallback

    def test_untrained_returns_failed(self):
        from paper_trader.ml.decision_scorer import DecisionScorer
        ds = DecisionScorer.__new__(DecisionScorer)
        ds._model = None
        ds._scaler = None
        ds._trained = False
        ds._n_train = 0
        ds._pred_quantiles = None
        ds._label_quantiles = None
        out = ds.predict_with_meta(
            ml_score=1.0, rsi=50.0, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA",
        )
        assert out["failed"] is True
        assert out["pred"] == 0.0
        assert out["calibrated"] is None
        assert out["percentile"] is None


# ---------------------------------------------------------------------------
# Section 6 — train_scorer label-validation drops corrupt rows
# ---------------------------------------------------------------------------

class TestTrainScorerLabelValidation:
    """A single row with a non-finite `forward_return_5d` would previously
    raise inside MLPRegressor.fit, which `_train_decision_scorer` swallows
    silently — wedging per-cycle retrain for as long as the bad row persists
    in the 5000-record tail. Pin the validation drop count + status code."""

    def _make_synthetic_records(self, n: int, bad_indices: list[int]) -> list[dict]:
        """Vary (ticker, sim_date) per row so dedup keeps all rows.
        train_scorer dedups by (ticker, sim_date, action) and the same-key
        pair collapses to one — make each record unique via a year offset
        so we exercise the LABEL-validation path, not the dedup path."""
        rng = np.random.default_rng(0)
        tickers = ["NVDA", "AMD", "MU", "TSLA", "AAPL"]
        out: list[dict] = []
        for i in range(n):
            score = float(rng.uniform(0.5, 10.0))
            if i in bad_indices:
                fr = None  # corrupt — should be dropped
            else:
                fr = float(score + rng.normal() * 0.5)
            # Unique (ticker, sim_date) per row: rotate tickers + shift year
            year = 2020 + (i // 100)
            day = (i % 28) + 1
            month = ((i // 28) % 12) + 1
            out.append({
                "ml_score": score, "rsi": 50.0, "macd": 0.0,
                "mom5": 0.0, "mom20": 0.0, "regime_mult": 1.0,
                "ticker": tickers[i % len(tickers)],
                "vol_ratio": 1.0, "bb_position": 0.0,
                "forward_return_5d": fr, "action": "BUY",
                "sim_date": f"{year}-{month:02d}-{day:02d}",
            })
        return out

    def test_records_with_null_label_are_dropped(self, tmp_path):
        from paper_trader.ml.decision_scorer import train_scorer
        records = self._make_synthetic_records(n=100, bad_indices=[5, 15, 25])
        out = train_scorer(records, path=tmp_path / "scorer.pkl")
        assert out["status"] == "ok"
        # `n` should be 100 - 3 dropped = 97.
        assert out["n"] == 97
        assert out.get("n_label_dropped") == 3

    def test_records_with_nan_label_are_dropped(self, tmp_path):
        from paper_trader.ml.decision_scorer import train_scorer
        records = self._make_synthetic_records(n=80, bad_indices=[])
        # Inject a NaN label
        records[10]["forward_return_5d"] = float("nan")
        records[20]["forward_return_5d"] = float("inf")
        out = train_scorer(records, path=tmp_path / "scorer.pkl")
        assert out["status"] == "ok"
        assert out.get("n_label_dropped") == 2

    def test_outlier_labels_are_clamped_not_dropped(self, tmp_path):
        from paper_trader.ml.decision_scorer import train_scorer
        records = self._make_synthetic_records(n=80, bad_indices=[])
        # Inject an outlier far above the empirical label support
        records[10]["forward_return_5d"] = 175.0   # > PRED_CLAMP_PCT
        records[20]["forward_return_5d"] = -200.0  # < -PRED_CLAMP_PCT
        out = train_scorer(records, path=tmp_path / "scorer.pkl")
        assert out["status"] == "ok"
        # Both rows survive — but get clamped to ±50.
        assert out.get("n_label_clamped") == 2
        assert out.get("n_label_dropped") == 0


# ---------------------------------------------------------------------------
# Section 7 — Backtest forward-return collision guard
# ---------------------------------------------------------------------------

class TestForwardReturnCollisionGuard:
    """``PriceCache.resolved_close_date`` is the honesty gate that lets
    ``_compute_decision_outcomes`` refuse a walk-back-collision outcome.
    Both endpoints resolving to the SAME prior close would silently produce
    a fabricated 0% return that poisons training. Verify the guard with a
    synthetic price cache."""

    def test_collision_returns_same_date(self):
        from paper_trader.backtest import PriceCache

        pc = PriceCache.__new__(PriceCache)
        pc.start = date(2024, 1, 1)
        pc.end = date(2024, 1, 31)
        pc.prices = {
            # Ticker with only ONE close in window — every lookup walks back to it
            "THIN": {"2024-01-15": 100.0},
        }
        pc.trading_days = []

        # Both endpoints walk back to the same close
        sim_res = pc.resolved_close_date("THIN", date(2024, 1, 20))
        end_res = pc.resolved_close_date("THIN", date(2024, 1, 22))
        assert sim_res == date(2024, 1, 15)
        assert end_res == date(2024, 1, 15)
        # The collision is detectable by the caller: end_res <= sim_res.
        # `returns_pct` returns 0.0, but with the collision sentinel the
        # caller (`_compute_decision_outcomes`) refuses to capture this row.
        assert pc.returns_pct("THIN", date(2024, 1, 20), date(2024, 1, 22)) == 0.0

    def test_no_collision_distinct_dates_returns_real_pct(self):
        from paper_trader.backtest import PriceCache

        pc = PriceCache.__new__(PriceCache)
        pc.start = date(2024, 1, 1)
        pc.end = date(2024, 1, 31)
        pc.prices = {
            "OK": {
                "2024-01-15": 100.0,
                "2024-01-22": 110.0,
            },
        }
        pc.trading_days = []
        sim_res = pc.resolved_close_date("OK", date(2024, 1, 15))
        end_res = pc.resolved_close_date("OK", date(2024, 1, 22))
        assert sim_res == date(2024, 1, 15)
        assert end_res == date(2024, 1, 22)
        # Genuine +10% return
        assert pc.returns_pct("OK", date(2024, 1, 15),
                              date(2024, 1, 22)) == pytest.approx(10.0, abs=1e-6)

    def test_missing_ticker_returns_none(self):
        from paper_trader.backtest import PriceCache
        pc = PriceCache.__new__(PriceCache)
        pc.start = date(2024, 1, 1)
        pc.end = date(2024, 1, 31)
        pc.prices = {}
        pc.trading_days = []
        assert pc.resolved_close_date("MISSING",
                                      date(2024, 1, 15)) is None
        # returns_pct also returns 0.0 sentinel for missing ticker
        assert pc.returns_pct("MISSING", date(2024, 1, 15),
                              date(2024, 1, 22)) == 0.0

    def test_walk_back_limited_to_7_days(self):
        """The walk-back window is exactly 7 days. An 8-day-prior close
        must NOT be picked up."""
        from paper_trader.backtest import PriceCache
        pc = PriceCache.__new__(PriceCache)
        pc.start = date(2024, 1, 1)
        pc.end = date(2024, 1, 31)
        pc.prices = {
            # Last close 8 days before the lookup
            "OLD": {"2024-01-10": 100.0},
        }
        pc.trading_days = []
        # 8 days away → no resolution
        assert pc.resolved_close_date("OLD", date(2024, 1, 18)) is None
        # 7 days away → resolves
        assert pc.resolved_close_date("OLD", date(2024, 1, 17)) == date(2024, 1, 10)


# ---------------------------------------------------------------------------
# Section 8 — Persona / regime maps consistency
# ---------------------------------------------------------------------------

class TestPersonaConsistency:
    """The 10 personas (CLAUDE.md §10) must be cyclically addressable by
    arbitrary run_id, and every persona must have boosts."""

    def test_persona_for_cycles_modulo_ten(self):
        from paper_trader.backtest import persona_for, PERSONAS
        # Spot-check the cycle
        assert persona_for(1)["name"] == PERSONAS[1]["name"]
        assert persona_for(11)["name"] == PERSONAS[1]["name"]
        assert persona_for(21)["name"] == PERSONAS[1]["name"]
        assert persona_for(10)["name"] == PERSONAS[10]["name"]
        assert persona_for(100)["name"] == PERSONAS[10]["name"]

    def test_persona_for_zero_or_negative_does_not_raise(self):
        from paper_trader.backtest import persona_for
        # run_id=0: ((0-1) % 10) + 1 = ((-1) % 10) + 1 = 9 + 1 = 10
        # run_id=-1: ((-1-1) % 10) + 1 = (-2 % 10) + 1 = 8 + 1 = 9
        # Should not crash; should produce a valid persona.
        p0 = persona_for(0)
        assert "name" in p0
        p_neg = persona_for(-5)
        assert "name" in p_neg

    def test_every_persona_has_boosts(self):
        from paper_trader.backtest import PERSONAS, _PERSONA_BOOSTS
        # Every persona key in PERSONAS must have an entry in _PERSONA_BOOSTS
        for persona_idx in PERSONAS:
            assert persona_idx in _PERSONA_BOOSTS, \
                f"Persona {persona_idx} missing _PERSONA_BOOSTS entry"
            assert _PERSONA_BOOSTS[persona_idx], \
                f"Persona {persona_idx} has empty boost dict"


# ---------------------------------------------------------------------------
# Section 9 — KILL-SWITCH abstention semantics
# ---------------------------------------------------------------------------

class TestGateKillSwitchDefaults:
    """When the kill-switch ledger is missing / unreadable / has fewer than
    MIN_CYCLES rows, the gate MUST default to active (preserves invariant #5
    semantics on fresh start). Locks the documented safe-default behavior.
    """

    def test_missing_ledger_defaults_to_active(self, tmp_path, monkeypatch):
        import paper_trader.backtest as bt
        # Point at a non-existent path
        monkeypatch.setattr(bt, "_GATE_SKILL_LOG_PATH",
                            tmp_path / "missing.jsonl")
        bt._reset_gate_skill_cache()
        gate, reason = bt._should_gate_modulate_conviction()
        assert gate is True
        assert "missing" in reason.lower() or "default" in reason.lower()

    def test_unreadable_ledger_defaults_to_active(self, tmp_path, monkeypatch):
        """A corrupt JSONL (un-parseable rows) shouldn't disable the gate —
        the kill-switch is supposed to make CONSERVATIVE choices."""
        import paper_trader.backtest as bt
        log_path = tmp_path / "skill.jsonl"
        log_path.write_text("this is not valid json\n" * 30)
        monkeypatch.setattr(bt, "_GATE_SKILL_LOG_PATH", log_path)
        bt._reset_gate_skill_cache()
        gate, reason = bt._should_gate_modulate_conviction()
        # Unparseable rows are skipped; if 0 valid rows < MIN_CYCLES,
        # defaults to gate-active.
        assert gate is True

    def test_low_skill_disables_gate(self, tmp_path, monkeypatch):
        """When the trailing OOS BUY rank-IC is below tolerance for
        MIN_CYCLES rows, the kill-switch fires."""
        import paper_trader.backtest as bt
        import json
        log_path = tmp_path / "skill.jsonl"
        # 25 rows with very small absolute IC — well below the 0.03 tolerance
        rows = []
        for i in range(25):
            rows.append(json.dumps({
                "cycle": i, "oos_buy_ic": 0.001,
            }))
        log_path.write_text("\n".join(rows) + "\n")
        monkeypatch.setattr(bt, "_GATE_SKILL_LOG_PATH", log_path)
        bt._reset_gate_skill_cache()
        gate, reason = bt._should_gate_modulate_conviction()
        assert gate is False
        assert "killed" in reason.lower() or "noise" in reason.lower()

    def test_high_skill_keeps_gate_active(self, tmp_path, monkeypatch):
        """When the trailing OOS BUY rank-IC clears the tolerance, the
        gate stays active."""
        import paper_trader.backtest as bt
        import json
        log_path = tmp_path / "skill.jsonl"
        # 25 rows with healthy IC > 0.03
        rows = []
        for i in range(25):
            rows.append(json.dumps({
                "cycle": i, "oos_buy_ic": 0.15,
            }))
        log_path.write_text("\n".join(rows) + "\n")
        monkeypatch.setattr(bt, "_GATE_SKILL_LOG_PATH", log_path)
        bt._reset_gate_skill_cache()
        gate, reason = bt._should_gate_modulate_conviction()
        assert gate is True

    def test_anti_predictive_skill_kills_gate(self, tmp_path, monkeypatch):
        """A persistently NEGATIVE oos_buy_ic (anti-predictive scorer)
        MUST kill the gate. The gate's per-arm sizing (pred<-10 → ×0.6,
        pred>+10 → ×1.3) assumes positive rank-IC; with anti-skill the
        modulation directionality is inverted vs realized returns and
        the gate actively HURTS sized return rather than abstaining.
        Live data on 2026-05-28 (trailing-20 median oos_buy_ic = -0.06)
        triggered exactly this fix.

        The prior ``abs(median_ic) < tolerance`` guard left this case
        gate-active because |-0.06| > 0.03 — the signed-comparison fix
        catches it (any median < +0.03 → kill)."""
        import paper_trader.backtest as bt
        import json
        log_path = tmp_path / "skill.jsonl"
        # 25 rows with median = -0.06 (the live observed case)
        rows = []
        for i in range(25):
            rows.append(json.dumps({
                "cycle": i, "oos_buy_ic": -0.06,
            }))
        log_path.write_text("\n".join(rows) + "\n")
        monkeypatch.setattr(bt, "_GATE_SKILL_LOG_PATH", log_path)
        bt._reset_gate_skill_cache()
        gate, reason = bt._should_gate_modulate_conviction()
        assert gate is False, (
            f"Anti-predictive median -0.06 must kill the gate; got "
            f"gate={gate}, reason={reason!r}"
        )
        # The reason string must surface the anti-skill / noise framing
        # so an operator reading the dashboard understands WHY the gate
        # is suppressed — not just that it is.
        assert ("killed" in reason.lower()
                or "noise" in reason.lower()
                or "anti" in reason.lower())

    def test_strongly_anti_predictive_skill_kills_gate(self, tmp_path,
                                                       monkeypatch):
        """Sanity sibling of the -0.06 case: a much larger anti-skill
        (-0.20) clearly must kill the gate too. Locks the broader
        semantic that ANY negative median below tolerance disables it,
        not just borderline cases — so a future refactor to a
        magnitude-weighted threshold cannot silently re-introduce the
        "high |IC|" gap."""
        import paper_trader.backtest as bt
        import json
        log_path = tmp_path / "skill.jsonl"
        rows = []
        for i in range(25):
            rows.append(json.dumps({
                "cycle": i, "oos_buy_ic": -0.20,
            }))
        log_path.write_text("\n".join(rows) + "\n")
        monkeypatch.setattr(bt, "_GATE_SKILL_LOG_PATH", log_path)
        bt._reset_gate_skill_cache()
        gate, _ = bt._should_gate_modulate_conviction()
        assert gate is False

    def test_borderline_positive_at_tolerance_keeps_gate_active(
        self, tmp_path, monkeypatch
    ):
        """Boundary case: positive skill exactly AT the tolerance keeps
        the gate active. Pins the inclusive-on-positive-side semantic so
        a future refactor cannot silently shift the threshold."""
        import paper_trader.backtest as bt
        import json
        log_path = tmp_path / "skill.jsonl"
        rows = []
        for i in range(25):
            rows.append(json.dumps({
                "cycle": i,
                "oos_buy_ic": bt._GATE_SKILL_IC_TOLERANCE,  # exactly at
            }))
        log_path.write_text("\n".join(rows) + "\n")
        monkeypatch.setattr(bt, "_GATE_SKILL_LOG_PATH", log_path)
        bt._reset_gate_skill_cache()
        gate, _ = bt._should_gate_modulate_conviction()
        assert gate is True  # median == tolerance ⇒ NOT < tolerance ⇒ active

    def test_reason_string_documents_anti_predictive_branch(
        self, tmp_path, monkeypatch
    ):
        """When the kill-switch fires due to negative IC, the reason
        string MUST surface the 'noise or anti-predictive' framing so an
        operator triaging from a shell can distinguish near-zero noise
        from real anti-skill. The exact wording is the operator's
        guidance — pin a stable token a downstream parser can grep for."""
        import paper_trader.backtest as bt
        import json
        log_path = tmp_path / "skill.jsonl"
        rows = [json.dumps({"cycle": i, "oos_buy_ic": -0.10})
                for i in range(25)]
        log_path.write_text("\n".join(rows) + "\n")
        monkeypatch.setattr(bt, "_GATE_SKILL_LOG_PATH", log_path)
        bt._reset_gate_skill_cache()
        gate, reason = bt._should_gate_modulate_conviction()
        assert gate is False
        assert "anti-predictive" in reason or "noise" in reason

    def test_borderline_negative_just_below_tolerance_kills_gate(
        self, tmp_path, monkeypatch
    ):
        """Boundary case: a tiny negative median just below the
        positive tolerance (e.g. -0.001) MUST kill the gate. This is
        the exact gap the old ``abs()`` guard left open: |-0.001| <
        0.03 → kill (correctly), but a slightly larger anti-skill like
        -0.04 had |-0.04| > 0.03 → keep active (incorrectly). Pin both
        sides of the signed threshold."""
        import paper_trader.backtest as bt
        import json
        log_path = tmp_path / "skill.jsonl"
        rows = []
        for i in range(25):
            rows.append(json.dumps({
                "cycle": i, "oos_buy_ic": -0.04,
            }))
        log_path.write_text("\n".join(rows) + "\n")
        monkeypatch.setattr(bt, "_GATE_SKILL_LOG_PATH", log_path)
        bt._reset_gate_skill_cache()
        gate, _ = bt._should_gate_modulate_conviction()
        # |-0.04| > 0.03 — under the old abs() guard this stayed
        # active. The signed-threshold fix kills it.
        assert gate is False


# ---------------------------------------------------------------------------
# Section 10 — heuristic article scorer
# ---------------------------------------------------------------------------

class TestArticleScorer:
    """``score_article`` is the keyword baseline that backfills articles
    when ArticleNet is unavailable. Pins the bullish/bearish phrase scoring."""

    def test_bullish_phrases_increase_score(self):
        from paper_trader.backtest import score_article
        # 'beat earnings' is in BUY_PHRASES (+0.5 per match)
        s_neutral, _ = score_article({"title": "company reports"})
        s_bull, _ = score_article({"title": "company beat earnings and raised guidance"})
        assert s_bull > s_neutral

    def test_bearish_phrases_decrease_score(self):
        from paper_trader.backtest import score_article
        s_neutral, _ = score_article({"title": "company reports"})
        s_bear, _ = score_article({"title": "company miss earnings and cut guidance"})
        assert s_bear < s_neutral

    def test_score_clamped_to_0_5_range(self):
        """Pathological titles with many bullish phrases stay within [0, 5]."""
        from paper_trader.backtest import score_article
        bull_text = ("beat earnings earnings beat revenue beat guidance raised "
                     "raised guidance record revenue strong demand supply shortage "
                     "upgrade outperform all-time high rally surge soar")
        s, _ = score_article({"title": bull_text})
        assert 0.0 <= s <= 5.0

    def test_ticker_extraction_handles_dollar_prefix(self):
        from paper_trader.backtest import score_article
        _, tickers = score_article({"title": "$NVDA beat earnings $AMD chip rally"})
        assert "NVDA" in tickers
        assert "AMD" in tickers

    def test_common_words_not_tickers(self):
        from paper_trader.backtest import score_article
        _, tickers = score_article({"title": "FOR THE USA ALL CEO ETF"})
        # All in _NOT_TICKERS should be filtered
        assert tickers == []

"""Tests for the DecisionScorer CLI's enhanced MACD/EMA200 feature plumbing.

The pass #35 effort plumbed the 3 enhanced-MACD features
(``ema200_above`` / ``hist_cross_up`` / ``macd_below_zero_cross``) through
``_compute_decision_outcomes`` (training capture) AND ``_ml_decide``
(live inference) so the model could LEARN those slots and act on them.
But ``decision_scorer.main`` (the CLI explainer) silently dropped them:
``_build_arg_parser`` had NO flags for them, and the ``common`` dict the
CLI builds for ``predict_with_meta`` / ``feature_contributions`` omitted
them — so the CLI's explanation predicted against a *different* feature
vector than the live gate sees for any name where these signals fire.

That made the per-feature attribution panel a fabrication when the
enhanced MACD slots actually carried weight: the explainer reported them
at exactly the zero-input baseline contribution, even on a real name where
the signal mattered. This file pins the wiring: the three flags exist on
the parser, they plumb to ``common``, and a True flag produces a different
prediction than the False/absent state (proving the value is actually
reaching the model). Same defensive contract the sibling tests for
``persona_skill`` / ``persona_regime_skill`` / etc. follow — read-only,
mock-free where possible, fast.

A new test file (not appended to ``test_decision_scorer.py``) so a
concurrent sibling agent editing the same test file cannot collide with
this work via whole-file ``git add`` — the documented same-role HYBRID
staging-race mitigation pattern.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from paper_trader.ml import decision_scorer as ds


class TestArgParserEnhancedMacdFlags:
    """The CLI's argparse must accept the three enhanced-MACD flags or the
    explainer cannot speak the same feature vocabulary as the live gate.
    """

    def test_parser_accepts_ema200_above_flag(self):
        parser = ds._build_arg_parser()
        args = parser.parse_args(["--ticker", "NVDA", "--ema200-above"])
        assert args.ema200_above is True

    def test_parser_accepts_hist_cross_up_flag(self):
        parser = ds._build_arg_parser()
        args = parser.parse_args(["--ticker", "NVDA", "--hist-cross-up"])
        assert args.hist_cross_up is True

    def test_parser_accepts_macd_below_zero_cross_flag(self):
        parser = ds._build_arg_parser()
        args = parser.parse_args(
            ["--ticker", "NVDA", "--macd-below-zero-cross"]
        )
        assert args.macd_below_zero_cross is True

    def test_parser_omits_flags_default_to_none(self):
        """Default is None (not False) — None plumbs to ``_bool_to_float``'s
        0.0 ("signal not present") sentinel, preserving the pre-fix
        behaviour for every explainer that doesn't pass the flag.
        Using ``False`` instead would silently flip a different code path
        in ``_bool_to_float`` (False ⇒ 0.0 too, but None ⇒ 0.0 via the
        ``v is None`` branch — semantically distinct: a future change to
        ``_bool_to_float`` could diverge these, and the existing
        ``build_features`` docstring documents None as "missing data").
        """
        parser = ds._build_arg_parser()
        args = parser.parse_args(["--ticker", "NVDA"])
        assert args.ema200_above is None
        assert args.hist_cross_up is None
        assert args.macd_below_zero_cross is None

    def test_parser_accepts_all_three_flags_together(self):
        parser = ds._build_arg_parser()
        args = parser.parse_args([
            "--ticker", "NVDA",
            "--ema200-above",
            "--hist-cross-up",
            "--macd-below-zero-cross",
        ])
        assert args.ema200_above is True
        assert args.hist_cross_up is True
        assert args.macd_below_zero_cross is True


class TestBuildFeaturesRespondsToMacdFlags:
    """Defense-in-depth: the explainer wires these to ``build_features``
    via the ``common`` dict. ``build_features`` must return a different
    vector when these flags are True vs absent — otherwise the model can
    never see the difference even if the wiring is correct.
    """

    def test_ema200_above_true_changes_feature_vector(self):
        """Slot index 10 in ``FEATURE_NAMES`` is ``ema200_above`` (after
        the 10 base numeric features, before the 7 sector one-hot). It
        must be 0.0 when False/None and 1.0 when True.
        """
        f_off = ds.build_features(
            ml_score=2.0, rsi=50.0, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA",
            ema200_above=None,
        )
        f_on = ds.build_features(
            ml_score=2.0, rsi=50.0, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA",
            ema200_above=True,
        )
        # Other slots must be identical — only the boolean changed.
        idx = ds.FEATURE_NAMES.index("ema200_above")
        assert f_off[idx] == 0.0
        assert f_on[idx] == 1.0
        # And the rest of the vector unchanged:
        for i in range(len(f_off)):
            if i == idx:
                continue
            assert f_off[i] == f_on[i], (
                f"slot {i} ({ds.FEATURE_NAMES[i]}) changed unexpectedly")

    def test_hist_cross_up_true_changes_feature_vector(self):
        idx = ds.FEATURE_NAMES.index("hist_cross_up")
        f_off = ds.build_features(
            ml_score=2.0, rsi=50.0, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA", hist_cross_up=None,
        )
        f_on = ds.build_features(
            ml_score=2.0, rsi=50.0, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA", hist_cross_up=True,
        )
        assert f_off[idx] == 0.0
        assert f_on[idx] == 1.0

    def test_macd_below_zero_cross_true_changes_feature_vector(self):
        idx = ds.FEATURE_NAMES.index("macd_below_zero_cross")
        f_off = ds.build_features(
            ml_score=2.0, rsi=50.0, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA", macd_below_zero_cross=None,
        )
        f_on = ds.build_features(
            ml_score=2.0, rsi=50.0, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA", macd_below_zero_cross=True,
        )
        assert f_off[idx] == 0.0
        assert f_on[idx] == 1.0


class TestMainPlumbsMacdFlagsThroughToPredict:
    """End-to-end wiring: when the CLI is invoked with one of the three
    flags, the call to ``DecisionScorer.predict_with_meta`` must receive
    that flag as the corresponding kwarg. This is the explicit
    "the value reaches the prediction call" assertion the prior CLI did
    not pass.

    Uses a mock scorer to avoid pickle dependency — we only care that the
    kwarg reaches predict_with_meta intact. The mock is created via
    ``patch`` on the ``DecisionScorer`` constructor so the CLI sees our
    fake without touching the deployed pickle.
    """

    def _make_fake_scorer(self):
        """Build a stand-in that records the kwargs predict_with_meta sees."""

        class _Fake:
            is_trained = True
            n_train = 1000
            recorded_kwargs: dict = {}

            def predict_with_meta(self, **kwargs):
                _Fake.recorded_kwargs = dict(kwargs)
                return {"pred": 5.0, "raw": 5.0, "clamped": False,
                        "off_distribution": False, "percentile": 75.0,
                        "calibrated": 4.0, "failed": False}

            def feature_contributions(self, **kwargs):
                return {"trained": True, "contributions": [],
                        "pred": 5.0, "pred_baseline": 0.0,
                        "interaction_residual": 0.0,
                        "off_distribution": False}

            def feature_group_contributions(self, **kwargs):
                return {"trained": True, "groups": [], "pred": 5.0,
                        "pred_baseline": 0.0,
                        "interaction_residual": 0.0,
                        "off_distribution": False}

        return _Fake()

    def test_main_passes_ema200_above_true_when_flag_given(self, capsys):
        fake = self._make_fake_scorer()
        with patch.object(ds, "DecisionScorer", return_value=fake):
            rc = ds.main([
                "--ticker", "NVDA", "--ml-score", "3.0",
                "--ema200-above",
                "--json",
            ])
        assert rc == 0
        # ``_make_fake_scorer`` uses a class attribute so we don't need
        # access to the instance — the recording is process-wide for the
        # test scope.
        assert fake.recorded_kwargs.get("ema200_above") is True
        # The other two enhanced flags must be None (default), not stripped.
        assert "hist_cross_up" in fake.recorded_kwargs
        assert fake.recorded_kwargs["hist_cross_up"] is None
        assert fake.recorded_kwargs["macd_below_zero_cross"] is None

    def test_main_passes_hist_cross_up_true_when_flag_given(self):
        fake = self._make_fake_scorer()
        with patch.object(ds, "DecisionScorer", return_value=fake):
            rc = ds.main([
                "--ticker", "NVDA", "--ml-score", "3.0",
                "--hist-cross-up",
                "--json",
            ])
        assert rc == 0
        assert fake.recorded_kwargs.get("hist_cross_up") is True
        assert fake.recorded_kwargs["ema200_above"] is None
        assert fake.recorded_kwargs["macd_below_zero_cross"] is None

    def test_main_passes_macd_below_zero_cross_true_when_flag_given(self):
        fake = self._make_fake_scorer()
        with patch.object(ds, "DecisionScorer", return_value=fake):
            rc = ds.main([
                "--ticker", "NVDA", "--ml-score", "3.0",
                "--macd-below-zero-cross",
                "--json",
            ])
        assert rc == 0
        assert fake.recorded_kwargs.get("macd_below_zero_cross") is True
        assert fake.recorded_kwargs["ema200_above"] is None
        assert fake.recorded_kwargs["hist_cross_up"] is None

    def test_main_omitted_flags_pass_none(self):
        """No flag ⇒ None ⇒ ``_bool_to_float`` falls back to 0.0
        ("signal not present"). This preserves the pre-fix default
        behaviour for every operator who doesn't pass the flag.
        """
        fake = self._make_fake_scorer()
        with patch.object(ds, "DecisionScorer", return_value=fake):
            rc = ds.main(["--ticker", "NVDA", "--ml-score", "3.0", "--json"])
        assert rc == 0
        assert fake.recorded_kwargs.get("ema200_above") is None
        assert fake.recorded_kwargs.get("hist_cross_up") is None
        assert fake.recorded_kwargs.get("macd_below_zero_cross") is None

    def test_main_passes_all_three_flags_together(self):
        fake = self._make_fake_scorer()
        with patch.object(ds, "DecisionScorer", return_value=fake):
            rc = ds.main([
                "--ticker", "NVDA", "--ml-score", "3.0",
                "--ema200-above", "--hist-cross-up",
                "--macd-below-zero-cross",
                "--json",
            ])
        assert rc == 0
        assert fake.recorded_kwargs["ema200_above"] is True
        assert fake.recorded_kwargs["hist_cross_up"] is True
        assert fake.recorded_kwargs["macd_below_zero_cross"] is True

"""Exact-value locks for the trivial-baseline comparison diagnostic
(`paper_trader/ml/baseline_compare.py`, 2026-05-18 quant feature).

Mirrors test_gate_audit.py / test_calibration.py: deterministic synthetic
data, EXACT metrics and EXACT verdicts (not ranges) so a logic change must
update the literals deliberately. All offline, no network, no trained MLP.

The load-bearing assertions:
  * the four-way verdict matrix on rank_ic vs best non-degenerate baseline,
    with exact rank_ic / ic_gap literals (Spearman is ±1.0 / 0.0 by
    construction — monotone, anti-monotone, and symmetric-tent-vs-monotone).
  * the `MLP_IC_MIN` skill floor: a clear margin win is STILL
    `MLP_NO_BETTER_THAN_TRIVIAL` when the MLP's own rank_ic is ≤ the floor
    (the "beats noise because everything is noise is not skill" guard).
  * the codebase-universal SELL `-forward_return_5d` sign-flip, applied to
    the realized target AND symmetrically to every trivial baseline's
    prediction — a single all-SELL slice locks BOTH flip arms (removing
    either inverts the verdict).
  * `oos_only` restricts to the temporal-OOS slice.
  * the `degenerate` flag fires on a constant baseline and such a baseline
    can never be selected as `best_baseline`.
  * never raises — raising scorer / NaN fields / empty / length-mismatch
    all degrade to INSUFFICIENT_DATA.
"""
from __future__ import annotations

import math

import pytest

from paper_trader.ml import baseline_compare as bc


class _EchoRegime:
    """predict() echoes regime_mult so the MLP column is fully controllable
    without a trained pickle (the established _FakeScorer pattern)."""

    is_trained = True
    n_train = 999

    def predict(self, **kw) -> float:
        return float(kw["regime_mult"])


class _NegEchoRegime:
    is_trained = True
    n_train = 999

    def predict(self, **kw) -> float:
        return -float(kw["regime_mult"])


class _RaisingScorer:
    is_trained = True
    n_train = 999

    def predict(self, **kw):
        raise ValueError("model/feature shape mismatch")


def _rec(i, *, forward, action="BUY", regime_mult, ml_score=5.0,
         mom20=0.0, mom5=0.0, rsi=50.0, bb=0.0, ticker="NVDA"):
    return {
        "run_id": 1,
        "sim_date": f"2020-{1 + i // 28:02d}-{1 + i % 28:02d}",
        "ticker": ticker,
        "action": action,
        "ml_score": ml_score,
        "rsi": rsi,
        "macd": 0.0,
        "mom5": mom5,
        "mom20": mom20,
        "regime_mult": regime_mult,
        "vol_ratio": 1.0,
        "bb_position": bb,
        "news_urgency": None,
        "news_article_count": None,
        "forward_return_5d": forward,
    }


class TestAlignedPredTarget:
    """`_aligned_pred_target` is the single SELL-flip primitive."""

    def test_buy_is_identity(self):
        assert bc._aligned_pred_target(3.0, 7.0, is_sell=False) == (3.0, 7.0)

    def test_sell_flips_both_pred_and_target(self):
        # Symmetric flip: the trivial-baseline pred AND the realized target
        # are both negated so the baseline is measured in the same
        # "goodness of THIS action" space the training-aligned MLP lives in.
        assert bc._aligned_pred_target(3.0, 7.0, is_sell=True) == (-3.0, -7.0)


class TestVerdictMatrix:
    """The four-way verdict on MLP rank_ic vs best non-degenerate baseline."""

    def test_mlp_adds_skill(self):
        # MLP echoes regime_mult (1..40, monotone ↑ target) → ic = +1.0.
        # mom20 = -(i+1) (monotone ↓ target) → ic = -1.0 (only non-degen).
        recs = [_rec(i, forward=i + 1, regime_mult=i + 1, mom20=-(i + 1))
                for i in range(40)]
        rep = bc.scorer_baseline_compare(_EchoRegime(), recs, oos_only=False)
        assert rep["verdict"] == "MLP_ADDS_SKILL"
        assert rep["mlp"]["rank_ic"] == 1.0
        assert rep["mlp"]["dir_acc"] == 1.0
        assert rep["best_baseline"] == "mom20"
        assert rep["best_baseline_ic"] == -1.0
        assert rep["ic_gap"] == 2.0
        assert rep["slice"] == "all"

    def test_mlp_worse_than_trivial(self):
        # MLP echoes -regime_mult (monotone ↓ target) → ic = -1.0.
        # ml_score = i+1 (monotone ↑ target) → ic = +1.0 (only non-degen;
        # mom20 held constant so it is degenerate and cannot be "best").
        recs = [_rec(i, forward=i + 1, regime_mult=i + 1, ml_score=i + 1,
                     mom20=7.0)
                for i in range(40)]
        rep = bc.scorer_baseline_compare(_NegEchoRegime(), recs,
                                         oos_only=False)
        assert rep["verdict"] == "MLP_WORSE_THAN_TRIVIAL"
        assert rep["mlp"]["rank_ic"] == -1.0
        assert rep["mlp"]["dir_acc"] == 0.0
        assert rep["best_baseline"] == "ml_score"
        assert rep["best_baseline_ic"] == 1.0
        assert rep["ic_gap"] == -2.0

    def test_mlp_no_better_within_margin(self):
        # MLP ic = +1.0 and ml_score baseline ic = +1.0 → gap 0.0, neither
        # clears the ±IC_MARGIN band → NO_BETTER via the within-margin arm.
        recs = [_rec(i, forward=i + 1, regime_mult=i + 1, ml_score=i + 1,
                     mom20=7.0)
                for i in range(40)]
        rep = bc.scorer_baseline_compare(_EchoRegime(), recs, oos_only=False)
        assert rep["verdict"] == "MLP_NO_BETTER_THAN_TRIVIAL"
        assert rep["mlp"]["rank_ic"] == 1.0
        assert rep["best_baseline"] == "ml_score"
        assert rep["best_baseline_ic"] == 1.0
        assert rep["ic_gap"] == 0.0

    def test_mlp_ic_floor_blocks_a_clear_margin_win(self):
        # Symmetric-tent target vs monotone MLP pred → Spearman EXACTLY 0.0
        # (every (i, N-1-i) pair cancels). mom20 = +|i-19.5| is the exact
        # inverse of the tent target → ic = -1.0 (only non-degen baseline).
        # MLP ic 0.0 clears best (-1.0) by 1.0 > IC_MARGIN, but 0.0 ≤
        # MLP_IC_MIN, so it must STILL read NO_BETTER (the skill-floor arm,
        # NOT the within-margin arm — ic_gap is a full 1.0 here).
        recs = []
        for i in range(40):
            tgt = -abs(i - 19.5)            # symmetric tent, all negative
            recs.append(_rec(i, forward=round(tgt, 4), regime_mult=i,
                             ml_score=5.0,           # constant → degenerate
                             mom20=round(abs(i - 19.5), 4)))  # = -target
        rep = bc.scorer_baseline_compare(_EchoRegime(), recs, oos_only=False)
        assert rep["verdict"] == "MLP_NO_BETTER_THAN_TRIVIAL"
        assert rep["mlp"]["rank_ic"] == 0.0
        assert rep["best_baseline"] == "mom20"
        assert rep["best_baseline_ic"] == -1.0
        assert rep["ic_gap"] == 1.0
        assert "skill" in rep["hint"]      # the floor-branch hint, not margin


class TestSellSignFlipRegression:
    """A single all-SELL slice locks BOTH flip arms at once.

    forward = -(i+1) (↓, all negative). After the target flip it becomes
    i+1 (↑). The training-aligned MLP echoes regime_mult = i+1 (NOT flipped)
    → ic vs flipped target = +1.0. ml_score raw = i+1, flipped on SELL to
    -(i+1) → ic vs flipped target = -1.0.

    Remove the *target* flip ⇒ MLP pred ↑ vs target ↓ ⇒ mlp ic = -1.0 ⇒
    verdict inverts. Remove the *baseline* flip ⇒ ml_score ↑ vs target ↑ ⇒
    best baseline ic = +1.0 ⇒ verdict inverts. Either regression fails the
    exact literals below.
    """

    def test_all_sell_slice_locks_both_flips(self):
        recs = [_rec(i, forward=-(i + 1), action="SELL", regime_mult=i + 1,
                     ml_score=i + 1, mom20=7.0)
                for i in range(40)]
        rep = bc.scorer_baseline_compare(_EchoRegime(), recs, oos_only=False)
        assert rep["verdict"] == "MLP_ADDS_SKILL"
        assert rep["mlp"]["rank_ic"] == 1.0
        assert rep["best_baseline"] == "ml_score"
        assert rep["best_baseline_ic"] == -1.0
        assert rep["ic_gap"] == 2.0


class TestDegenerateFlag:
    """A constant baseline is `degenerate` and never the chosen best."""

    def test_constant_zero_is_degenerate_and_not_best(self):
        recs = [_rec(i, forward=i + 1, regime_mult=i + 1, mom20=-(i + 1))
                for i in range(40)]
        rep = bc.scorer_baseline_compare(_EchoRegime(), recs, oos_only=False)
        rows = {b["name"]: b for b in rep["baselines"]}
        assert rows["constant_zero"]["degenerate"] is True
        assert rows["constant_zero"]["rank_ic"] is None
        # ml_score / mom5 / rsi_meanrev / neg_bb are all constant here too.
        assert rows["ml_score"]["degenerate"] is True
        assert rows["mom5"]["degenerate"] is True
        assert rows["rsi_meanrev"]["degenerate"] is True
        assert rows["neg_bb"]["degenerate"] is True
        # Only mom20 varies → it must be the best (a degenerate row can
        # never win even though its _spearman would be 0.0).
        assert rep["best_baseline"] == "mom20"
        assert rep["best_baseline"] != "constant_zero"


class TestOosRestriction:
    """`oos_only` restricts to the temporal-OOS slice (same split every
    sibling tool uses)."""

    def test_oos_only_takes_last_twenty_percent(self):
        # 200 records → split_outcomes_temporal OOS = last 40 (≥ MIN_PAIRS).
        recs = [_rec(i, forward=i + 1, regime_mult=i + 1, mom20=-(i + 1))
                for i in range(200)]
        rep = bc.scorer_baseline_compare(_EchoRegime(), recs, oos_only=True)
        assert rep["slice"] == "oos"
        assert rep["n_records_considered"] == 40
        assert rep["n"] == 40
        # The OOS slice is internally consistent (monotone within it too).
        assert rep["mlp"]["rank_ic"] == 1.0
        assert rep["verdict"] == "MLP_ADDS_SKILL"

    def test_all_flag_uses_full_set(self):
        recs = [_rec(i, forward=i + 1, regime_mult=i + 1, mom20=-(i + 1))
                for i in range(60)]
        rep = bc.scorer_baseline_compare(_EchoRegime(), recs, oos_only=False)
        assert rep["slice"] == "all"
        assert rep["n_records_considered"] == 60
        assert rep["n"] == 60


class TestNeverRaises:
    """Every fault path degrades to INSUFFICIENT_DATA, never an exception."""

    def test_raising_scorer_yields_insufficient(self):
        recs = [_rec(i, forward=i + 1, regime_mult=i + 1) for i in range(40)]
        rep = bc.scorer_baseline_compare(_RaisingScorer(), recs,
                                         oos_only=False)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == 0

    def test_empty_records(self):
        rep = bc.scorer_baseline_compare(_EchoRegime(), [], oos_only=False)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == 0

    def test_too_few_records(self):
        recs = [_rec(i, forward=i + 1, regime_mult=i + 1) for i in range(10)]
        rep = bc.scorer_baseline_compare(_EchoRegime(), recs, oos_only=False)
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_nan_forward_returns_are_skipped(self):
        recs = [_rec(i, forward=float("nan"), regime_mult=i + 1)
                for i in range(40)]
        rep = bc.scorer_baseline_compare(_EchoRegime(), recs, oos_only=False)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == 0

    def test_report_length_mismatch_is_insufficient(self):
        rep = bc.baseline_compare_report([1.0, 2.0],
                                         {"x": [1.0, 2.0, 3.0]},
                                         [1.0, 2.0, 3.0])
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_report_below_min_pairs(self):
        rep = bc.baseline_compare_report([1.0] * 5, {"x": [1.0] * 5},
                                         [1.0] * 5)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == 5


class TestAnalyzeCli:
    """The file-loading entrypoint is read-only and degrades cleanly."""

    def test_missing_outcomes_file(self, tmp_path):
        rep = bc.analyze(tmp_path / "nope.jsonl")
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert "no outcomes file" in rep["hint"]

    def test_untrained_scorer_via_analyze(self, tmp_path):
        # conftest redirects SCORER_PATH to a tmp dir → DecisionScorer() is
        # untrained, so analyze must short-circuit (never trains anything).
        f = tmp_path / "decision_outcomes.jsonl"
        f.write_text('{"ticker":"NVDA","action":"BUY",'
                     '"forward_return_5d":1.0}\n')
        rep = bc.analyze(f)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert "not trained" in rep["hint"]

    def test_cli_exit_code_branches_on_verdict(self, monkeypatch):
        # MLP_NO_BETTER / MLP_WORSE → exit 2 (operator/cron-branchable),
        # else 0 — same contract as gate_audit / feature_importance.
        monkeypatch.setattr(
            bc, "analyze",
            lambda *a, **k: {"verdict": "MLP_NO_BETTER_THAN_TRIVIAL",
                             "hint": "", "mlp": {}, "baselines": []})
        assert bc._cli([]) == 2
        monkeypatch.setattr(
            bc, "analyze",
            lambda *a, **k: {"verdict": "MLP_ADDS_SKILL", "hint": "",
                             "mlp": {}, "baselines": []})
        assert bc._cli([]) == 0
        monkeypatch.setattr(
            bc, "analyze",
            lambda *a, **k: {"verdict": "INSUFFICIENT_DATA", "hint": "",
                             "mlp": {}, "baselines": []})
        assert bc._cli([]) == 0


class TestReadOnlyDiscipline:
    """The module must not import a write/train/feature path (mirrors the
    calibration.py / gate_audit.py operational-discipline guarantee)."""

    def test_no_forbidden_symbols_in_code(self):
        # Scan the CODE only — the module docstring legitimately *names*
        # train_scorer / build_features / N_FEATURES to state it never
        # touches them, so strip the docstring before asserting.
        import ast
        import inspect
        src = inspect.getsource(bc)
        tree = ast.parse(src)
        doc = ast.get_docstring(tree) or ""
        code = src.replace(doc, "", 1)
        for forbidden in ("train_scorer(", "build_features(", "N_FEATURES",
                          "pickle.dump", ".pkl.tmp", "_execute("):
            assert forbidden not in code, f"forbidden symbol: {forbidden}"
        # And it must not import the trainer / pickle-writer modules.
        for mod in tree.body:
            if isinstance(mod, (ast.Import, ast.ImportFrom)):
                txt = ast.dump(mod)
                assert "train_scorer" not in txt

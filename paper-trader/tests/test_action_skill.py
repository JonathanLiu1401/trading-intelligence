"""Tests for paper_trader.ml.action_skill — per-action OOS scorer skill.

Mirrors the test discipline of tests/test_persona_skill.py /
test_regime_audit.py: every test asserts a specific expected verdict or
numeric output, not just "no crash". Offline by construction — uses
scorer stubs and synthetic outcome records.
"""
from __future__ import annotations

import math

import pytest

from paper_trader.ml import action_skill as ask


class _ScorerStub:
    """Minimal scorer stub. Yields predictions per call from a list.

    Mirrors DecisionScorer's predict signature shape — the test doesn't
    care about feature values; only the per-call output sequence drives
    the rank correlation we want to lock.
    """

    is_trained = True

    def __init__(self, predictions: list[float]):
        self._preds = list(predictions)
        self._i = 0

    def predict(self, **_kw) -> float:
        if self._i >= len(self._preds):
            raise IndexError("test scorer ran out of predictions")
        p = self._preds[self._i]
        self._i += 1
        return p


class _UntrainedScorer:
    is_trained = False

    def predict(self, **_kw):
        return 0.0


def _rec(action: str, fr: float | None, ticker: str = "NVDA",
         ml_score: float = 1.0) -> dict:
    """Compact synthetic outcome row in the decision_outcomes.jsonl shape."""
    return {
        "action": action,
        "ticker": ticker,
        "ml_score": ml_score,
        "rsi": 50.0,
        "macd": 0.0,
        "mom5": 0.0,
        "mom20": 0.0,
        "regime_mult": 1.0,
        "forward_return_5d": fr,
    }


# ──────────────────── verdict thresholds & purity ────────────────────


class TestVerdictFor:
    """The per-action `_verdict_for` is a pure threshold function — every
    boundary is testable in isolation. Locks the IC bands exactly so a
    future threshold tweak is a single, reviewable failure."""

    def test_insufficient_when_below_per_action_min(self):
        # Even a perfect +1 IC reads INSUFFICIENT if the sample is too small.
        assert ask._verdict_for(1.0, ask.MIN_OUTCOMES_PER_ACTION - 1) == "INSUFFICIENT"

    def test_inverted_at_threshold(self):
        assert ask._verdict_for(-ask.IC_GOOD, ask.MIN_OUTCOMES_PER_ACTION) == "INVERTED"

    def test_edge_at_threshold(self):
        assert ask._verdict_for(ask.IC_GOOD, ask.MIN_OUTCOMES_PER_ACTION) == "EDGE"

    def test_weak_edge_at_threshold(self):
        assert ask._verdict_for(ask.IC_MIN, ask.MIN_OUTCOMES_PER_ACTION) == "WEAK_EDGE"

    def test_no_edge_between_bands(self):
        # Strictly between IC_MIN and -IC_GOOD reads NO_EDGE.
        mid = (ask.IC_MIN - 0.01)
        assert ask._verdict_for(mid, ask.MIN_OUTCOMES_PER_ACTION) == "NO_EDGE"


# ──────────────────── overall verdict aggregation ────────────────────


class TestActionSkillVerdicts:
    def test_untrained_scorer_returns_untrained_status(self):
        rep = ask.action_skill(_UntrainedScorer(), [_rec("BUY", 1.0)] * 30)
        assert rep["status"] == "untrained"
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_records"] == 0
        # Both action buckets read INSUFFICIENT honestly.
        assert rep["by_action"]["BUY"]["verdict"] == "INSUFFICIENT"
        assert rep["by_action"]["SELL"]["verdict"] == "INSUFFICIENT"

    def test_insufficient_data_below_min_records(self):
        # 10 BUY records — well below MIN_RECORDS.
        recs = [_rec("BUY", 0.1 * i) for i in range(10)]
        # Predictions just need to exist; values don't matter at this stage.
        scorer = _ScorerStub([float(i) for i in range(10)])
        rep = ask.action_skill(scorer, recs)
        assert rep["status"] == "insufficient_data"
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_records"] == 10

    def test_both_skilled_when_buy_and_sell_track_perfectly(self):
        """When pred ranks PERFECTLY match aligned-realized ranks on both
        actions (rank_ic ≈ +1 on each), the verdict must be BOTH_SKILLED.
        """
        # 20 BUYs with monotone (pred, realized) — perfect +1 rank_ic.
        buy_recs = [_rec("BUY", float(i + 1)) for i in range(20)]
        buy_preds = [float(i + 1) for i in range(20)]
        # 20 SELLs with monotone (pred, -realized) — sells flip on the
        # actuals side, so to get +1 rank_ic on action-aligned data the
        # realized forward_return must DECREASE as pred increases.
        sell_recs = [_rec("SELL", -float(i + 1)) for i in range(20)]
        sell_preds = [float(i + 1) for i in range(20)]
        scorer = _ScorerStub(buy_preds + sell_preds)
        rep = ask.action_skill(scorer, buy_recs + sell_recs)
        assert rep["status"] == "ok"
        assert rep["verdict"] == "BOTH_SKILLED"
        assert rep["by_action"]["BUY"]["rank_ic"] == pytest.approx(1.0)
        assert rep["by_action"]["SELL"]["rank_ic"] == pytest.approx(1.0)
        assert rep["by_action"]["BUY"]["verdict"] == "EDGE"
        assert rep["by_action"]["SELL"]["verdict"] == "EDGE"
        assert rep["n_records"] == 40

    def test_asymmetric_buy_edge(self):
        """BUY rank_ic ≥ IC_GOOD, SELL rank_ic between bands → the gate-
        relevant healthy case (the gate acts BUY-only, and the skill is
        on the BUY half). Verdict must be ASYMMETRIC_BUY_EDGE.

        Constructed deterministically: SELL predictions are a CONSTANT
        sequence — constant signals have rank_ic = 0 exactly (the
        `_spearman` zero-variance contract), squarely in NO_EDGE.
        """
        buy_recs = [_rec("BUY", float(i + 1)) for i in range(20)]
        buy_preds = [float(i + 1) for i in range(20)]
        # Constant SELL predictions → IC=0 → NO_EDGE (not INVERTED).
        sell_preds = [1.0] * 20
        sell_recs = [_rec("SELL", -float(i + 1)) for i in range(20)]
        scorer = _ScorerStub(buy_preds + sell_preds)
        rep = ask.action_skill(scorer, buy_recs + sell_recs)
        assert rep["status"] == "ok"
        assert rep["by_action"]["BUY"]["verdict"] == "EDGE"
        # IC=0 with constant pred — _spearman returns exactly 0.0, lands in NO_EDGE.
        assert rep["by_action"]["SELL"]["rank_ic"] == pytest.approx(0.0)
        assert rep["by_action"]["SELL"]["verdict"] == "NO_EDGE"
        assert rep["verdict"] == "ASYMMETRIC_BUY_EDGE"

    def test_asymmetric_sell_edge_warns_in_hint(self):
        """SELL=EDGE, BUY!=EDGE → the *concerning* case: the model's skill
        is on a slice the gate doesn't use. The hint must say so, so an
        operator reading the summary can't miss it.

        Mirror image of the BUY-edge test: constant BUY predictions →
        IC=0 → NO_EDGE; SELLs perfectly aligned → EDGE.
        """
        buy_preds = [1.0] * 20  # constant → IC=0 → NO_EDGE
        buy_recs = [_rec("BUY", float(i + 1)) for i in range(20)]
        sell_recs = [_rec("SELL", -float(i + 1)) for i in range(20)]
        sell_preds = [float(i + 1) for i in range(20)]
        scorer = _ScorerStub(buy_preds + sell_preds)
        rep = ask.action_skill(scorer, buy_recs + sell_recs)
        assert rep["by_action"]["BUY"]["verdict"] == "NO_EDGE"
        assert rep["by_action"]["SELL"]["verdict"] == "EDGE"
        assert rep["verdict"] == "ASYMMETRIC_SELL_EDGE"
        # The hint must call out the gate mismatch — a quant reading
        # only the headline would otherwise miss that the aggregate
        # OVERSTATES gate-relevant edge.
        assert "BUY-only" in rep["hint"]
        assert "OVERSTATES" in rep["hint"]

    def test_neither_skilled_when_both_are_noise(self):
        """Both actions have IC=0 (constant predictions on each side) →
        verdict NEITHER_SKILLED, hint mentions MLP_NO_BETTER_THAN_TRIVIAL.
        """
        buy_preds = [1.0] * 20
        sell_preds = [1.0] * 20
        buy_recs = [_rec("BUY", float(i + 1)) for i in range(20)]
        sell_recs = [_rec("SELL", -float(i + 1)) for i in range(20)]
        scorer = _ScorerStub(buy_preds + sell_preds)
        rep = ask.action_skill(scorer, buy_recs + sell_recs)
        assert rep["by_action"]["BUY"]["rank_ic"] == pytest.approx(0.0)
        assert rep["by_action"]["SELL"]["rank_ic"] == pytest.approx(0.0)
        assert rep["verdict"] == "NEITHER_SKILLED"
        assert "MLP_NO_BETTER_THAN_TRIVIAL" in rep["hint"]

    def test_has_inverted_action_wins_over_other_action_skill(self):
        """An INVERTED action is a red flag — the overall verdict must
        surface it even when the OTHER action has full EDGE. Mirrors
        persona_skill's HAS_INVERTED_PERSONA precedence rule.
        """
        # BUYs perfectly skilled.
        buy_recs = [_rec("BUY", float(i + 1)) for i in range(20)]
        buy_preds = [float(i + 1) for i in range(20)]
        # SELLs INVERTED: predictions INCREASE with realized (not
        # action-aligned which flips to -realized) → rank_ic = -1.0.
        sell_recs = [_rec("SELL", -float(i + 1)) for i in range(20)]
        # Action-aligned realized = -fr = +1,+2,...,+20. To get IC=-1.0
        # the predictions must DECREASE as aligned-realized increases.
        sell_preds = [float(20 - i) for i in range(20)]
        scorer = _ScorerStub(buy_preds + sell_preds)
        rep = ask.action_skill(scorer, buy_recs + sell_recs)
        assert rep["by_action"]["BUY"]["rank_ic"] == pytest.approx(1.0)
        assert rep["by_action"]["SELL"]["rank_ic"] == pytest.approx(-1.0)
        assert rep["by_action"]["BUY"]["verdict"] == "EDGE"
        assert rep["by_action"]["SELL"]["verdict"] == "INVERTED"
        # The OVERALL verdict surfaces the red flag, not BUY's skill.
        assert rep["verdict"] == "HAS_INVERTED_ACTION"
        assert "anti-correlated" in rep["hint"]
        assert "SELL" in rep["hint"]


# ──────────────────── input hardening ────────────────────


class TestAlignedPred:
    def test_missing_forward_return_drops_record(self):
        scorer = _ScorerStub([1.0])
        assert ask._aligned_pred(scorer, _rec("BUY", None)) is None
        # And the scorer.predict was NEVER called — the function short-
        # circuits before predict for a None target (cheap defense).
        assert scorer._i == 0

    def test_non_finite_target_drops_record(self):
        scorer = _ScorerStub([1.0])
        assert ask._aligned_pred(scorer, _rec("BUY", float("nan"))) is None

    def test_predict_raise_drops_record(self):
        class _Raiser:
            is_trained = True
            def predict(self, **_kw):
                raise RuntimeError("simulated predict fault")
        assert ask._aligned_pred(_Raiser(), _rec("BUY", 1.0)) is None

    def test_sell_flips_target_not_prediction(self):
        """A SELL with realized=-5 (a good SELL) and predict=+2 must
        return (+2, +5) — only the actual is flipped. Matches the
        universal codebase SELL convention."""
        scorer = _ScorerStub([2.0])
        p, t = ask._aligned_pred(scorer, _rec("SELL", -5.0))
        assert p == pytest.approx(2.0)
        assert t == pytest.approx(5.0)

    def test_buy_does_not_flip(self):
        scorer = _ScorerStub([2.0])
        p, t = ask._aligned_pred(scorer, _rec("BUY", -5.0))
        assert p == pytest.approx(2.0)
        assert t == pytest.approx(-5.0)


class TestHoldRowsAreSkipped:
    """`decision_outcomes.jsonl` should never carry HOLD rows
    (`_compute_decision_outcomes` filters to BUY/SELL) but a future
    schema change could leak them. The diagnostic must defend itself —
    HOLD rows are skipped silently and never inflate any per-action
    bucket."""

    def test_hold_rows_excluded_from_buckets(self):
        recs = ([_rec("HOLD", 1.0)] * 50
                 + [_rec("BUY", float(i + 1)) for i in range(20)]
                 + [_rec("SELL", -float(i + 1)) for i in range(20)])
        scorer = _ScorerStub([float(i + 1) for i in range(20)]
                              + [float(i + 1) for i in range(20)])
        rep = ask.action_skill(scorer, recs)
        # n_records counts only aligned BUY+SELL records, not the HOLDs.
        assert rep["n_records"] == 40

    def test_garbage_action_excluded(self):
        recs = ([_rec("REBALANCE", 1.0)] * 5
                 + [_rec("BUY", float(i + 1)) for i in range(20)]
                 + [_rec("SELL", -float(i + 1)) for i in range(20)])
        scorer = _ScorerStub([float(i + 1) for i in range(20)]
                              + [float(i + 1) for i in range(20)])
        rep = ask.action_skill(scorer, recs)
        assert rep["n_records"] == 40


# ──────────────────── numeric correctness ────────────────────


class TestDirAcc:
    def test_dir_acc_excludes_zero_pred_or_actual_pairs(self):
        """A zero on either side carries no directional truth and must
        be excluded — same convention as `_oos_rank_metrics.dir_pairs`.
        We need ≥MIN_RECORDS total for the function to score; pad with
        SELL records that don't affect the BUY dir_acc lock.
        """
        # 21 BUYs: first has pred=0 (excluded), next 10 hit, next 10 miss.
        buy_recs = [_rec("BUY", 1.0)]                                 # pred=0 → skip
        buy_recs += [_rec("BUY", float(i + 1)) for i in range(10)]    # hits
        buy_recs += [_rec("BUY", -float(i + 1)) for i in range(10)]   # misses
        buy_preds = [0.0]
        buy_preds += [float(i + 1) for i in range(10)]
        buy_preds += [float(i + 1) for i in range(10)]
        # 10 SELL records to clear MIN_RECORDS=30 — orthogonal to BUY lock.
        sell_recs = [_rec("SELL", -float(i + 1)) for i in range(10)]
        sell_preds = [float(i + 1) for i in range(10)]
        scorer = _ScorerStub(buy_preds + sell_preds)
        rep = ask.action_skill(scorer, buy_recs + sell_recs)
        # 10 hits / 20 non-zero pairs.
        assert rep["by_action"]["BUY"]["dir_acc"] == pytest.approx(0.5)

    def test_dir_acc_none_when_all_zero_pairs(self):
        # 20 BUYs, every pred=0 → no dir_pairs → dir_acc=None.
        recs = [_rec("BUY", float(i + 1)) for i in range(20)]
        scorer = _ScorerStub([0.0] * 20)
        # Add 10 SELL records too to clear MIN_RECORDS.
        recs += [_rec("SELL", -float(i + 1)) for i in range(10)]
        scorer = _ScorerStub([0.0] * 20 + [float(i + 1) for i in range(10)])
        rep = ask.action_skill(scorer, recs)
        # BUY dir_acc = None (every pred was 0), SELL dir_acc = 1.0
        assert rep["by_action"]["BUY"]["dir_acc"] is None
        assert rep["by_action"]["SELL"]["dir_acc"] == pytest.approx(1.0)


class TestMeanAlignedReturn:
    def test_mean_uses_aligned_target_not_raw(self):
        """The mean is over action-aligned targets — a SELL that went
        down (realized=-2) contributes +2 to the SELL mean (good SELL),
        not -2.
        """
        # 25 SELLs with realized always -2 → aligned = +2 → mean = +2.
        recs = [_rec("SELL", -2.0) for _ in range(25)]
        # Need >=30 total — top up with neutral BUYs.
        recs += [_rec("BUY", 0.0) for _ in range(10)]
        scorer = _ScorerStub([1.0] * 25 + [0.0] * 10)
        rep = ask.action_skill(scorer, recs)
        assert rep["by_action"]["SELL"]["mean_aligned_return"] == pytest.approx(2.0)
        assert rep["by_action"]["BUY"]["mean_aligned_return"] == pytest.approx(0.0)


# ──────────────────── CLI/load wiring ────────────────────


class TestLoadOutcomes:
    def test_missing_file_returns_empty_list(self, tmp_path):
        assert ask._load_outcomes(tmp_path / "absent.jsonl") == []

    def test_skips_corrupt_lines_without_raising(self, tmp_path):
        p = tmp_path / "outcomes.jsonl"
        p.write_text(
            '{"action": "BUY", "forward_return_5d": 1.0}\n'
            'not valid json\n'
            '{"action": "SELL", "forward_return_5d": -1.0}\n'
            '\n'
            '12345\n'  # non-dict json — must also be skipped
        )
        rows = ask._load_outcomes(p)
        # Two valid dict rows survive.
        assert len(rows) == 2
        assert rows[0]["action"] == "BUY"
        assert rows[1]["action"] == "SELL"


class TestAnalyzeCli:
    def test_analyze_with_no_outcomes_file_returns_honest_insufficient(
            self, tmp_path, monkeypatch):
        # Point the scorer at the conftest-isolated SCORER_PATH (which is
        # absent) so the scorer is honestly untrained.
        rep = ask.analyze(outcomes_path=tmp_path / "absent.jsonl",
                          oos_only=False)
        # Untrained scorer wins over empty records — status reflects that.
        assert rep["status"] == "untrained"
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        # The slice label is still present (no records to slice).
        assert rep["slice"] in ("all", "oos")

    def test_analyze_oos_slice_label_set_when_split_succeeds(
            self, tmp_path, monkeypatch):
        # Build a tiny outcomes file with enough rows that the temporal
        # split yields a non-empty OOS tail (we just need the slice label;
        # the verdict will be untrained because no scorer is deployed in
        # tests via the autouse fixture).
        p = tmp_path / "outcomes.jsonl"
        lines = []
        for i in range(40):
            month = 1 + (i % 12)
            day = 1 + (i // 12)
            lines.append(
                '{"action": "BUY", "ticker": "NVDA", "ml_score": 1.0, '
                '"forward_return_5d": ' + str(i * 0.1) + ', '
                '"sim_date": "2024-' + f"{month:02d}-{day:02d}" + '", '
                '"regime_mult": 1.0}'
            )
        p.write_text("\n".join(lines) + "\n")
        rep = ask.analyze(outcomes_path=p, oos_only=True)
        # The slice label is "oos" because split_outcomes_temporal succeeded
        # (and degraded to "all" only on a real split failure).
        assert rep["slice"] in ("oos", "all")

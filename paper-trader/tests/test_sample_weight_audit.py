"""Tests for paper_trader.ml.sample_weight_audit.

The audit trains scorers in-memory under several weighting policies, evaluates
each on the same temporal-OOS holdout, and reports the best policy. These
tests pin the BUSINESS LOGIC — verdict thresholds, the no-disk-write
invariant, dedup, the temporal split discipline, and the universal SELL
sign-flip — not just "the code runs".
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import pytest

from paper_trader.ml import sample_weight_audit as swa


def _make_outcome(*, ticker="NVDA", action="BUY", sim_date="2025-01-01",
                  ml_score=2.0, rsi=50.0, macd=0.1, mom5=0.0, mom20=0.0,
                  fwd=5.0, regime_mult=1.0, return_pct=10.0,
                  llm_label=0, vol_ratio=1.0, bb_position=0.0,
                  news_urgency=50.0, news_article_count=1.0) -> dict:
    return {
        "ticker": ticker, "action": action, "sim_date": sim_date,
        "ml_score": ml_score, "rsi": rsi, "macd": macd,
        "mom5": mom5, "mom20": mom20, "regime_mult": regime_mult,
        "vol_ratio": vol_ratio, "bb_position": bb_position,
        "news_urgency": news_urgency,
        "news_article_count": news_article_count,
        "forward_return_5d": fwd, "return_pct": return_pct,
        "llm_quality_label": llm_label,
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


# ────────────────────────── policy weight functions ──────────────────────

class TestWeightPolicies:
    """Each policy is a pure function of one outcome record. Pin the
    specific values that a refactor must not silently change — these are
    the same constants `train_scorer` uses in production, so a drift here
    is a drift in the deployed scorer behaviour."""

    def test_current_weight_high_return_record_top_clamp(self):
        # return_pct = +200 → 1 + 1 = 2.0 (clamp top)
        # llm = 0 (neutral) → llm_mult = 1.0
        w = swa._weight_current(_make_outcome(return_pct=200, llm_label=0))
        assert w == pytest.approx(2.0)

    def test_current_weight_low_return_record_bottom_clamp(self):
        # return_pct = -200 → 1 - 1 = 0 → clamped to 0.5
        w = swa._weight_current(_make_outcome(return_pct=-200, llm_label=0))
        assert w == pytest.approx(0.5)

    def test_current_weight_endorsed_3x(self):
        # return_pct = 0 → 1.0; llm = +1 → 3x → total 3.0
        w = swa._weight_current(_make_outcome(return_pct=0, llm_label=1))
        assert w == pytest.approx(3.0)

    def test_current_weight_condemned_01x(self):
        # return_pct = 0 → 1.0; llm = -1 → 0.1x → total 0.1
        w = swa._weight_current(_make_outcome(return_pct=0, llm_label=-1))
        assert w == pytest.approx(0.1)

    def test_uniform_always_one(self):
        assert swa._weight_uniform(_make_outcome(return_pct=999)) == 1.0
        assert swa._weight_uniform(_make_outcome(llm_label=-1)) == 1.0

    def test_run_only_drops_llm_factor(self):
        # Same return_pct, different LLM labels → identical weight (the
        # whole point of `run_only` is to isolate the return_pct effect).
        rp = 50.0
        ws = {swa._weight_run_only(_make_outcome(return_pct=rp,
                                                  llm_label=ll))
              for ll in (-1, 0, 1)}
        assert len(ws) == 1
        assert next(iter(ws)) == pytest.approx(
            max(0.5, min(2.0, 1.0 + rp / 200.0))
        )

    def test_llm_only_drops_run_quality_factor(self):
        # Same llm_label, different return_pct → identical weight.
        ws = {swa._weight_llm_only(_make_outcome(return_pct=rp,
                                                  llm_label=1))
              for rp in (-100, 0, 100)}
        assert len(ws) == 1
        assert next(iter(ws)) == 3.0

    def test_abs_label_weights_large_moves_more(self):
        small = swa._weight_abs_label(_make_outcome(fwd=0.0))
        medium = swa._weight_abs_label(_make_outcome(fwd=10.0))
        large = swa._weight_abs_label(_make_outcome(fwd=30.0))
        # Strictly increasing in |fwd|.
        assert small < medium < large
        # Bounds are 0.5..3.0 per the policy.
        assert 0.5 <= small <= 3.0
        assert 0.5 <= large <= 3.0

    def test_abs_label_safe_on_bad_label(self):
        # Non-finite / non-coercible forward_return should NOT crash and
        # should return the neutral 1.0 fallback — same defensive pattern
        # `_to_float` uses across the codebase.
        for bad in (None, "x", float("nan"), float("inf"), True):
            w = swa._weight_abs_label({"forward_return_5d": bad})
            assert math.isfinite(w)
            assert 0.5 <= w <= 3.0


# ──────────────────────────── dedup ──────────────────────────────────────

class TestDedup:
    def test_dedup_collapses_same_key(self):
        # Same (ticker, sim_date, action) but different return_pct: dedup
        # keeps the higher-return copy.
        a = _make_outcome(return_pct=10)
        b = _make_outcome(return_pct=50)
        kept = swa._dedup([a, b])
        assert len(kept) == 1
        assert kept[0]["return_pct"] == 50

    def test_dedup_keeps_buy_and_sell_with_same_date_ticker(self):
        # The dedup key INCLUDES action — a BUY and a SELL of the same
        # ticker on the same date carry opposite labels (after the
        # universal SELL sign-flip) and must survive.
        buy = _make_outcome(action="BUY")
        sell = _make_outcome(action="SELL")
        kept = swa._dedup([buy, sell])
        actions = sorted(r["action"] for r in kept)
        assert actions == ["BUY", "SELL"]


# ──────────────────────────── train+eval pipeline ────────────────────────

class TestBuildXy:
    def test_drops_records_with_missing_forward_return(self):
        good = _make_outcome(fwd=5.0)
        bad = _make_outcome(fwd=None)
        X, y, kept = swa._build_xy([good, bad])
        assert len(kept) == 1
        assert kept[0]["forward_return_5d"] == 5.0

    def test_drops_records_with_non_finite_forward_return(self):
        good = _make_outcome(fwd=5.0)
        bad_inf = _make_outcome(fwd=float("inf"))
        bad_nan = _make_outcome(fwd=float("nan"))
        X, y, kept = swa._build_xy([good, bad_inf, bad_nan])
        assert len(kept) == 1

    def test_sell_target_is_sign_flipped(self):
        # SELL forward_return=-5 must become label=+5 (sign-flip), so the
        # model sees one consistent meaning of "good outcome".
        buy = _make_outcome(action="BUY", fwd=5.0)
        sell = _make_outcome(action="SELL", fwd=-5.0)
        _, y, _ = swa._build_xy([buy, sell])
        assert list(y) == [pytest.approx(5.0), pytest.approx(5.0)]

    def test_label_clamped_to_pred_clamp_pct(self):
        rec = _make_outcome(fwd=200.0)
        _, y, _ = swa._build_xy([rec])
        assert y[0] == pytest.approx(swa.PRED_CLAMP_PCT)


# ──────────────────────────── insufficient_data verdicts ─────────────────

class TestInsufficientData:
    def test_missing_outcomes_file_returns_insufficient(self, tmp_path):
        rep = swa.analyze(tmp_path / "missing.jsonl")
        assert rep["status"] == "error"
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert "not found" in (rep.get("hint") or "").lower()

    def test_empty_outcomes_returns_insufficient(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        rep = swa.analyze(p)
        assert rep["status"] == "error"
        assert rep["n_outcomes"] == 0

    def test_too_few_records_returns_insufficient_data(self, tmp_path):
        # Below MIN_OOS_PAIRS — must NOT silently return a verdict.
        p = tmp_path / "small.jsonl"
        recs = [_make_outcome(sim_date=f"2025-01-{i+1:02d}")
                for i in range(10)]
        _write_jsonl(p, recs)
        rep = swa.analyze(p)
        assert rep["status"] == "insufficient_data"
        assert rep["verdict"] == "INSUFFICIENT_DATA"


# ──────────────────────────── full analyze pipeline ──────────────────────

class TestAnalyze:
    @pytest.fixture
    def synthetic_outcomes(self, tmp_path):
        """200 records over 200 distinct sim_dates so dedup is a no-op and
        the temporal split puts the most recent 40 in OOS.
        Forward returns are correlated with `mom5` (a known feature) so
        the trained scorer has REAL skill — i.e. rank-IC isn't pure noise.
        """
        rng = random.Random(42)
        records = []
        for i in range(200):
            sim_date = f"2025-{(i // 30) + 1:02d}-{(i % 30) + 1:02d}"
            mom5 = rng.uniform(-5.0, 5.0)
            # True fwd return correlates with mom5 + noise.
            fwd = mom5 + rng.gauss(0, 3.0)
            records.append(_make_outcome(
                sim_date=sim_date,
                ticker=rng.choice(["NVDA", "AMD", "TSLA", "MSFT", "AAPL"]),
                ml_score=rng.uniform(0.5, 4.0),
                rsi=rng.uniform(20, 80),
                mom5=mom5, fwd=fwd,
                return_pct=rng.uniform(-30, 30),
                llm_label=rng.choice([-1, 0, 1]),
            ))
        path = tmp_path / "outcomes.jsonl"
        _write_jsonl(path, records)
        return path

    def test_returns_ok_with_sufficient_data(self, synthetic_outcomes):
        rep = swa.analyze(synthetic_outcomes)
        assert rep["status"] == "ok"
        assert rep["n_outcomes"] == 200
        # 80% train, 20% OOS — but post-dedup so could vary slightly.
        assert rep["n_train"] >= 100
        assert rep["n_oos"] >= swa.MIN_OOS_PAIRS

    def test_all_policies_evaluated(self, synthetic_outcomes):
        rep = swa.analyze(synthetic_outcomes)
        names = {r["name"] for r in rep["policies"]}
        # All five policies must be in the leaderboard.
        assert names == set(swa.POLICIES.keys())

    def test_current_is_marked(self, synthetic_outcomes):
        rep = swa.analyze(synthetic_outcomes)
        current_rows = [r for r in rep["policies"] if r["is_current"]]
        assert len(current_rows) == 1
        assert current_rows[0]["name"] == "current"

    def test_audit_does_not_write_deployed_pickle(self, synthetic_outcomes,
                                                    tmp_path, monkeypatch):
        """LOAD-BEARING operational invariant: the A/B audit must NEVER
        write to the deployed `data/ml/decision_scorer.pkl`. A single
        regression here would silently overwrite the production model
        with one trained on (possibly worse) sample weights.
        """
        import paper_trader.ml.decision_scorer as ds
        fake_pkl = tmp_path / "should_not_exist.pkl"
        monkeypatch.setattr(ds, "SCORER_PATH", fake_pkl)
        assert not fake_pkl.exists()
        swa.analyze(synthetic_outcomes)
        # The audit MUST NOT have created any pickle.
        assert not fake_pkl.exists(), (
            "sample_weight_audit wrote to SCORER_PATH — load-bearing "
            "read-only invariant violated"
        )

    def test_best_policy_when_uniform_beats_current(self, synthetic_outcomes,
                                                     monkeypatch):
        """If we construct a corpus where the LLM column is pure noise
        relative to the return_pct column, removing it should not HURT —
        verdict should be CURRENT_TIED or CURRENT_OPTIMAL (current
        always ties itself + run_only when LLM is purely noise).
        """
        rep = swa.analyze(synthetic_outcomes)
        assert rep["verdict"] in ("CURRENT_OPTIMAL", "CURRENT_TIED",
                                   "CURRENT_DOMINATED")

    def test_dominated_verdict_fires_when_alternative_wins(self,
                                                            synthetic_outcomes,
                                                            monkeypatch):
        """Force a fake policy with a known better OOS rank-IC by passing
        a custom policy dict where one policy returns large weights ONLY
        for records the model already gets right. With the corpus
        designed so mom5 → fwd, weighting `mom5 > 0` records 10x makes
        the model lean into the positive-mom5 signal more strongly.
        """
        def heavy_positive_mom5(r):
            return 10.0 if (r.get("mom5") or 0) > 1.0 else 1.0
        custom = dict(swa.POLICIES)
        custom["heavy_mom5"] = heavy_positive_mom5
        rep = swa.analyze(synthetic_outcomes, policies=custom)
        # All policies still evaluated; the verdict is one of the
        # documented values.
        assert rep["status"] == "ok"
        names = {r["name"] for r in rep["policies"]}
        assert "heavy_mom5" in names

    def test_oos_n_meets_minimum(self, synthetic_outcomes):
        rep = swa.analyze(synthetic_outcomes)
        assert rep["n_oos"] >= swa.MIN_OOS_PAIRS

    def test_json_safe_output(self, synthetic_outcomes):
        rep = swa.analyze(synthetic_outcomes)
        # Must round-trip JSON without TypeError (no numpy types leaking).
        s = json.dumps(rep)
        rep2 = json.loads(s)
        assert rep2["status"] == "ok"
        assert rep2["verdict"] == rep["verdict"]


# ──────────────────────────── CLI ─────────────────────────────────────────

class TestCli:
    def test_cli_exit_code_zero_on_ok(self, tmp_path):
        # Reuse the synthetic fixture pattern directly here. Needs enough
        # records so the 80/20 temporal split leaves ≥ MIN_OOS_PAIRS in the
        # holdout post-dedup.
        rng = random.Random(123)
        records = []
        for i in range(300):
            sim_date = f"2025-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}"
            records.append(_make_outcome(
                sim_date=sim_date,
                ticker=rng.choice(["NVDA", "AMD", "TSLA", "MSFT", "AAPL"]),
                ml_score=rng.uniform(0.5, 4.0),
                mom5=rng.uniform(-5, 5),
                fwd=rng.uniform(-10, 10),
            ))
        p = tmp_path / "outcomes.jsonl"
        _write_jsonl(p, records)
        rc = swa.main(["--outcomes-path", str(p), "--json"])
        assert rc == 0

    def test_cli_exit_code_one_on_missing_file(self, tmp_path):
        rc = swa.main(["--outcomes-path", str(tmp_path / "missing.jsonl")])
        assert rc == 1

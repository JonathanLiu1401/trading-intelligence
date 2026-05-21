"""Tests for paper_trader.ml.llm_annotation_skill — read-only LLM-annotation
realized-return diagnostic.

Pinned invariants: read-only (no disk writes), pure (NaN/garbage input
degrades to honest verdicts not exceptions), exact-string verdict ladder so
operator scripts and the future skill-ledger consumer can branch on it
deterministically.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from paper_trader.ml import llm_annotation_skill as las


# ─────────────────────────── _label_for ───────────────────────────

class TestLabelFor:
    def test_explicit_plus_one_is_endorse(self):
        assert las._label_for({"llm_quality_label": 1}) == 1

    def test_explicit_minus_one_is_condemn(self):
        assert las._label_for({"llm_quality_label": -1}) == -1

    def test_zero_is_unlabeled(self):
        assert las._label_for({"llm_quality_label": 0}) == 0

    def test_missing_key_defaults_to_zero(self):
        # Mirrors train_scorer's `int(r.get("llm_quality_label") or 0)`.
        assert las._label_for({}) == 0

    def test_explicit_null_defaults_to_zero(self):
        assert las._label_for({"llm_quality_label": None}) == 0

    def test_string_label_drops_to_none(self):
        # Out-of-range / unparseable values must not contaminate the buckets.
        assert las._label_for({"llm_quality_label": "BAD"}) is None

    def test_out_of_range_drops_to_none(self):
        # Only -1/0/+1 are part of the trainer contract.
        assert las._label_for({"llm_quality_label": 2}) is None
        assert las._label_for({"llm_quality_label": -3}) is None

    def test_non_dict_input_returns_none(self):
        # Defensive: a malformed record (e.g. a list / string) must not crash.
        assert las._label_for("not a dict") is None
        assert las._label_for(None) is None


# ─────────────────────────── _aligned_return ───────────────────────────

class TestAlignedReturn:
    def test_buy_passes_through(self):
        r = {"action": "BUY", "forward_return_5d": 3.5}
        assert las._aligned_return(r) == 3.5

    def test_sell_sign_flipped(self):
        r = {"action": "SELL", "forward_return_5d": 3.5}
        # SELL "good" = price fell — sign flipped to align with BUY space.
        assert las._aligned_return(r) == -3.5

    def test_default_action_is_buy(self):
        r = {"forward_return_5d": 2.0}
        assert las._aligned_return(r) == 2.0

    def test_missing_return_is_none(self):
        assert las._aligned_return({"action": "BUY"}) is None

    def test_null_return_is_none(self):
        assert las._aligned_return(
            {"action": "BUY", "forward_return_5d": None}) is None

    def test_nan_return_is_none(self):
        assert las._aligned_return(
            {"action": "BUY", "forward_return_5d": float("nan")}) is None

    def test_inf_return_is_none(self):
        # _to_float rejects ±inf per the documented contract.
        assert las._aligned_return(
            {"action": "BUY", "forward_return_5d": float("inf")}) is None

    def test_non_dict_returns_none(self):
        assert las._aligned_return("garbage") is None


# ─────────────────────────── llm_annotation_skill verdicts ───────────────────────────

class TestVerdictLadder:
    def test_no_labels_produced_when_all_zero(self):
        # The real-world live-state finding: 7413/7413 rows have label=0.
        records = [{"action": "BUY", "forward_return_5d": 1.0,
                    "llm_quality_label": 0} for _ in range(100)]
        rep = las.llm_annotation_skill(records)
        assert rep["verdict"] == "NO_LABELS_PRODUCED"
        assert rep["n_endorsed"] == 0
        assert rep["n_condemned"] == 0
        assert rep["n_unlabeled"] == 100
        assert rep["n_total"] == 100
        # Empty corpus still surfaces unlabeled mean (a quant cares about the
        # baseline return on the records the LLM never touched).
        assert rep["unlabeled_mean_return"] == 1.0

    def test_no_labels_produced_on_empty_corpus(self):
        rep = las.llm_annotation_skill([])
        assert rep["verdict"] == "NO_LABELS_PRODUCED"
        assert rep["n_total"] == 0

    def test_insufficient_labels_when_only_one_side(self):
        # 50 endorsed, 0 condemned → can't compare → INSUFFICIENT.
        records = [{"action": "BUY", "forward_return_5d": 1.0,
                    "llm_quality_label": 1} for _ in range(50)]
        rep = las.llm_annotation_skill(records)
        assert rep["verdict"] == "INSUFFICIENT_LABELS"
        assert rep["n_endorsed"] == 50
        assert rep["n_condemned"] == 0

    def test_insufficient_labels_when_below_min_per_group(self):
        # 5 endorsed + 5 condemned — under MIN_PER_GROUP=10 on each side.
        records = (
            [{"action": "BUY", "forward_return_5d": 5.0,
              "llm_quality_label": 1} for _ in range(5)]
            + [{"action": "BUY", "forward_return_5d": -5.0,
                "llm_quality_label": -1} for _ in range(5)]
        )
        rep = las.llm_annotation_skill(records)
        assert rep["verdict"] == "INSUFFICIENT_LABELS"

    def test_llm_directional_when_endorsed_outperforms_strongly(self):
        # 15 endorsed with +5pp returns, 15 condemned with -5pp — gap = 10pp,
        # well above RETURN_GAP_GOOD (1.0).
        records = (
            [{"action": "BUY", "forward_return_5d": 5.0,
              "llm_quality_label": 1} for _ in range(15)]
            + [{"action": "BUY", "forward_return_5d": -5.0,
                "llm_quality_label": -1} for _ in range(15)]
        )
        rep = las.llm_annotation_skill(records)
        assert rep["verdict"] == "LLM_DIRECTIONAL"
        assert rep["n_endorsed"] == 15
        assert rep["n_condemned"] == 15
        assert rep["endorsed_mean_return"] == pytest.approx(5.0)
        assert rep["condemned_mean_return"] == pytest.approx(-5.0)
        assert rep["endorsed_minus_condemned"] == pytest.approx(10.0)
        # rank_ic of a perfect sort is +1.0.
        assert rep["rank_ic"] is not None and rep["rank_ic"] > 0.9

    def test_llm_anti_predictive_when_condemned_outperforms_strongly(self):
        # Inverted: condemned-labeled trades actually do better. RED FLAG.
        records = (
            [{"action": "BUY", "forward_return_5d": -3.0,
              "llm_quality_label": 1} for _ in range(15)]
            + [{"action": "BUY", "forward_return_5d": +3.0,
                "llm_quality_label": -1} for _ in range(15)]
        )
        rep = las.llm_annotation_skill(records)
        assert rep["verdict"] == "LLM_ANTI_PREDICTIVE"
        assert rep["endorsed_minus_condemned"] == pytest.approx(-6.0)
        assert rep["rank_ic"] is not None and rep["rank_ic"] < -0.9

    def test_llm_inert_when_means_indistinguishable(self):
        # Endorsed mean ≈ condemned mean (gap < RETURN_GAP_MIN). Both
        # populations are identical noise.
        rng_seed = [0.1, -0.2, 0.05, -0.1, 0.0, 0.2, -0.05, 0.15, -0.15, 0.1,
                    -0.1, 0.05, -0.05, 0.0, 0.2]
        records = (
            [{"action": "BUY", "forward_return_5d": v,
              "llm_quality_label": 1} for v in rng_seed]
            + [{"action": "BUY", "forward_return_5d": v,
                "llm_quality_label": -1} for v in rng_seed]
        )
        rep = las.llm_annotation_skill(records)
        # Same population on both sides → gap exactly 0 → INERT.
        assert rep["verdict"] == "LLM_INERT"
        assert abs(rep["endorsed_minus_condemned"]) < las.RETURN_GAP_MIN

    def test_llm_directional_weak_when_gap_between_thresholds(self):
        # Gap = 0.7pp — in [RETURN_GAP_MIN=0.5, RETURN_GAP_GOOD=1.0).
        records = (
            [{"action": "BUY", "forward_return_5d": 0.7,
              "llm_quality_label": 1} for _ in range(15)]
            + [{"action": "BUY", "forward_return_5d": 0.0,
                "llm_quality_label": -1} for _ in range(15)]
        )
        rep = las.llm_annotation_skill(records)
        assert rep["verdict"] == "LLM_DIRECTIONAL_WEAK"

    def test_llm_anti_weak_when_negative_gap_between_thresholds(self):
        # Gap = -0.7pp — condemned slightly outperforms.
        records = (
            [{"action": "BUY", "forward_return_5d": 0.0,
              "llm_quality_label": 1} for _ in range(15)]
            + [{"action": "BUY", "forward_return_5d": 0.7,
                "llm_quality_label": -1} for _ in range(15)]
        )
        rep = las.llm_annotation_skill(records)
        assert rep["verdict"] == "LLM_ANTI_WEAK"


class TestSellSignFlip:
    def test_sell_endorse_with_negative_return_counts_as_good(self):
        # A SELL that preceded a -5pp return: that's GOOD (we sold before
        # the drop). Action-aligned, this becomes +5pp.
        records = (
            [{"action": "SELL", "forward_return_5d": -5.0,
              "llm_quality_label": 1} for _ in range(15)]
            + [{"action": "SELL", "forward_return_5d": +5.0,
                "llm_quality_label": -1} for _ in range(15)]
        )
        rep = las.llm_annotation_skill(records)
        # Endorsed SELLs preceded drops (good); condemned SELLs preceded
        # rises (bad). Aligned: endorsed = +5, condemned = -5 → +10 gap.
        assert rep["verdict"] == "LLM_DIRECTIONAL"
        assert rep["endorsed_mean_return"] == pytest.approx(5.0)
        assert rep["condemned_mean_return"] == pytest.approx(-5.0)


class TestRobustness:
    def test_malformed_record_does_not_crash(self):
        records = [
            "not-a-dict",
            None,
            {"action": "BUY", "forward_return_5d": "bad", "llm_quality_label": 1},
            {"action": "BUY", "forward_return_5d": 2.0,
             "llm_quality_label": "junk"},
            {"action": "BUY", "forward_return_5d": float("nan"),
             "llm_quality_label": 1},
        ]
        # Should not raise.
        rep = las.llm_annotation_skill(records)
        assert isinstance(rep, dict)
        assert "verdict" in rep
        # All rows dropped → NO_LABELS_PRODUCED.
        assert rep["n_total"] == 0

    def test_unparseable_label_value_drops_record(self):
        # A row with an unparseable label must NOT inflate n_unlabeled or
        # leak into either group.
        records = [
            {"action": "BUY", "forward_return_5d": 1.0,
             "llm_quality_label": "ENDORSE"},  # string is dropped
        ]
        rep = las.llm_annotation_skill(records)
        assert rep["n_total"] == 0

    def test_iterable_input_works(self):
        # Generator (not list) — function must consume any iterable.
        def gen():
            yield {"action": "BUY", "forward_return_5d": 1.0,
                   "llm_quality_label": 0}
        rep = las.llm_annotation_skill(gen())
        assert rep["n_unlabeled"] == 1


# ─────────────────────────── analyze() entrypoint ───────────────────────────

class TestAnalyze:
    def test_missing_file_returns_no_labels_produced(self, tmp_path):
        out = las.analyze(tmp_path / "does_not_exist.jsonl")
        assert out["verdict"] == "NO_LABELS_PRODUCED"
        assert out["n_total"] == 0

    def test_loads_jsonl(self, tmp_path):
        p = tmp_path / "outcomes.jsonl"
        rows = (
            [{"action": "BUY", "forward_return_5d": 4.0,
              "llm_quality_label": 1} for _ in range(15)]
            + [{"action": "BUY", "forward_return_5d": -4.0,
                "llm_quality_label": -1} for _ in range(15)]
        )
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        out = las.analyze(p)
        assert out["verdict"] == "LLM_DIRECTIONAL"
        assert out["n_endorsed"] == 15
        assert out["n_condemned"] == 15

    def test_malformed_jsonl_lines_skipped(self, tmp_path):
        p = tmp_path / "outcomes.jsonl"
        good = json.dumps({"action": "BUY", "forward_return_5d": 1.0,
                           "llm_quality_label": 1})
        p.write_text(good + "\n{bad json\n" + good + "\n")
        out = las.analyze(p)
        # Two good rows, one bad — must parse the two.
        assert out["n_endorsed"] == 2

    def test_does_not_write_anything(self, tmp_path, monkeypatch):
        # Lock the read-only invariant: analyze must NEVER write to disk.
        # We intercept Path.write_text / open(...,"w") at the file system
        # level by snapshotting mtimes before and after.
        p = tmp_path / "outcomes.jsonl"
        p.write_text(json.dumps({"action": "BUY",
                                  "forward_return_5d": 1.0,
                                  "llm_quality_label": 0}) + "\n")
        before = p.stat().st_mtime_ns
        las.analyze(p)
        after = p.stat().st_mtime_ns
        # mtime must be exactly identical (no rewrite, no truncate).
        assert before == after


# ─────────────────────────── CLI ───────────────────────────

class TestCLI:
    def _run_cli(self, *args, cwd: Path):
        return subprocess.run(
            [sys.executable, "-m", "paper_trader.ml.llm_annotation_skill",
             *args],
            cwd=str(cwd), capture_output=True, text=True, timeout=30,
        )

    def test_cli_emits_json(self, tmp_path):
        p = tmp_path / "outcomes.jsonl"
        rows = [{"action": "BUY", "forward_return_5d": 4.0,
                 "llm_quality_label": 1} for _ in range(15)]
        rows += [{"action": "BUY", "forward_return_5d": -4.0,
                  "llm_quality_label": -1} for _ in range(15)]
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        proj_root = Path(__file__).resolve().parent.parent
        res = self._run_cli("--path", str(p), "--json", cwd=proj_root)
        assert res.returncode == 0, (
            f"stderr: {res.stderr}\nstdout: {res.stdout}")
        rep = json.loads(res.stdout)
        assert rep["verdict"] == "LLM_DIRECTIONAL"

    def test_cli_exit_2_on_anti_predictive(self, tmp_path):
        # The actionable red-flag verdict should produce exit code 2 so
        # operator cron jobs can branch on it. Mirrors action_skill._cli /
        # news_volume_skill._cli precedent.
        p = tmp_path / "outcomes.jsonl"
        rows = [{"action": "BUY", "forward_return_5d": -3.0,
                 "llm_quality_label": 1} for _ in range(15)]
        rows += [{"action": "BUY", "forward_return_5d": +3.0,
                  "llm_quality_label": -1} for _ in range(15)]
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        proj_root = Path(__file__).resolve().parent.parent
        res = self._run_cli("--path", str(p), "--json", cwd=proj_root)
        assert res.returncode == 2
        rep = json.loads(res.stdout)
        assert rep["verdict"] == "LLM_ANTI_PREDICTIVE"

    def test_cli_text_table(self, tmp_path):
        p = tmp_path / "outcomes.jsonl"
        rows = [{"action": "BUY", "forward_return_5d": 1.0,
                 "llm_quality_label": 0} for _ in range(20)]
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        proj_root = Path(__file__).resolve().parent.parent
        res = self._run_cli("--path", str(p), cwd=proj_root)
        assert res.returncode == 0
        # The dark-pipeline state must be prominent in the text output —
        # that's the operationally important verdict.
        assert "NO_LABELS_PRODUCED" in res.stdout


# ─────────────────────────── thresholds ───────────────────────────

class TestThresholds:
    def test_boundary_gap_exactly_at_good_is_directional(self):
        # gap == RETURN_GAP_GOOD: the >= boundary fires (verdict ladder
        # locks the inclusive bound).
        gap = las.RETURN_GAP_GOOD
        records = (
            [{"action": "BUY", "forward_return_5d": gap,
              "llm_quality_label": 1} for _ in range(15)]
            + [{"action": "BUY", "forward_return_5d": 0.0,
                "llm_quality_label": -1} for _ in range(15)]
        )
        rep = las.llm_annotation_skill(records)
        assert rep["verdict"] == "LLM_DIRECTIONAL"

    def test_boundary_gap_just_below_min_is_inert(self):
        # gap just under RETURN_GAP_MIN — INERT not WEAK_DIRECTIONAL.
        gap = las.RETURN_GAP_MIN - 0.001
        records = (
            [{"action": "BUY", "forward_return_5d": gap,
              "llm_quality_label": 1} for _ in range(15)]
            + [{"action": "BUY", "forward_return_5d": 0.0,
                "llm_quality_label": -1} for _ in range(15)]
        )
        rep = las.llm_annotation_skill(records)
        assert rep["verdict"] == "LLM_INERT"

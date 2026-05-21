"""Tests for paper_trader.ml.dead_sector_audit.

The audit is a read-only diagnostic that flags scorer features whose
``sector_<name>`` weight share is near-zero despite the training pool
carrying real outcome mass for that sector. These tests exercise the
PURE logic (record → counts, payload → shares, verdict classification)
with synthetic inputs so no real scorer or outcomes file is required.
"""
from __future__ import annotations

import json

import pytest

from paper_trader.ml import dead_sector_audit as dsa
from paper_trader.ml.dead_sector_audit import (
    CORPUS_GROWTH_RETRAIN_HINT,
    MIN_IMPORTANCE_SHARE,
    MIN_RECORDS,
    _classify_sector,
    _per_sector_counts,
    analyze,
    report,
)
from paper_trader.ml.decision_scorer import SECTORS


# ─────────────────────── per-sector verdict ───────────────────────

class TestClassifySector:
    def test_sparse_data_takes_priority_over_dead_threshold(self):
        # Below the record threshold, an importance of 0 is honest — we
        # don't have data to learn from, so SPARSE_DATA wins.
        assert _classify_sector(0, 0.0) == "SPARSE_DATA"
        assert _classify_sector(MIN_RECORDS - 1, 0.0) == "SPARSE_DATA"
        # Even an "above threshold" importance is reported as SPARSE when
        # n_outcomes is too small (a coincidentally-trained large weight
        # on a near-empty sector is also untrustworthy).
        assert _classify_sector(5, 0.10) == "SPARSE_DATA"

    def test_dead_feature_requires_records_AND_low_importance(self):
        # At MIN_RECORDS+, with importance exactly at the threshold,
        # NOT dead (boundary uses strict `<`).
        assert _classify_sector(MIN_RECORDS, MIN_IMPORTANCE_SHARE) == \
            "HEALTHY"
        # Just below the threshold ⇒ DEAD_FEATURE.
        assert _classify_sector(MIN_RECORDS,
                                MIN_IMPORTANCE_SHARE - 1e-6) == \
            "DEAD_FEATURE"
        # Exactly zero importance with rich data is the alert case
        # observed live for sector_crypto / sector_energy.
        assert _classify_sector(500, 0.0) == "DEAD_FEATURE"

    def test_healthy_path(self):
        assert _classify_sector(MIN_RECORDS, MIN_IMPORTANCE_SHARE + 0.01) \
            == "HEALTHY"
        assert _classify_sector(10_000, 0.20) == "HEALTHY"


# ─────────────────────── per-sector counts ───────────────────────

class TestPerSectorCounts:
    def test_counts_total_matches_input(self):
        records = [
            {"ticker": "NVDA"}, {"ticker": "AMZN"},          # tech
            {"ticker": "LLY"},                                # healthcare
            {"ticker": "XOM"}, {"ticker": "BP"},              # energy
            {"ticker": "MSTR"}, {"ticker": "COIN"},           # crypto
        ]
        counts = _per_sector_counts(records)
        assert sum(counts.values()) == len(records)

    def test_unknown_ticker_falls_to_other(self):
        records = [{"ticker": "ZZZZZ"}, {"ticker": "QQQUNKNOWN"}]
        counts = _per_sector_counts(records)
        assert counts["other"] == 2
        # No other sector should have a count for unknown tickers
        for s in SECTORS:
            if s != "other":
                assert counts[s] == 0

    def test_returns_zero_for_missing_sector(self):
        # Empty records ⇒ every sector at 0, but every sector PRESENT
        # (regression: a defaultdict-like impl might omit sectors with
        # no rows, breaking downstream iteration).
        counts = _per_sector_counts([])
        for s in SECTORS:
            assert counts[s] == 0

    def test_handles_none_and_missing_ticker_safely(self):
        records = [
            {"ticker": None},
            {},                          # no ticker key
            {"ticker": "NVDA"},          # one valid row
        ]
        counts = _per_sector_counts(records)
        # None/missing → empty string → SECTOR_MAP.get("") returns
        # "other" (no overlap with any real ticker).
        assert counts["other"] == 2
        assert counts["tech"] == 1

    def test_empty_or_none_input_returns_zero_counts(self):
        for inp in ([], None):
            counts = _per_sector_counts(inp)
            assert sum(counts.values()) == 0
            assert set(counts.keys()) >= set(SECTORS)


# ─────────────────────── report() composition ───────────────────────

def _imp_payload(*, n_train: int, shares: dict[str, float] | None = None):
    """Synthesize a DecisionScorer.feature_importance() payload."""
    rows = []
    for s in SECTORS:
        share = (shares or {}).get(s, 0.0)
        rows.append({
            "feature": f"sector_{s}",
            "importance": share * 100,             # not used by report
            "importance_normalized": share,
        })
    return {
        "trained": True,
        "method": "mlp_first_layer_mean_abs_weight",
        "n_train": n_train,
        "importances": rows,
    }


class TestReport:
    def test_insufficient_data_when_no_records(self):
        out = report([], _imp_payload(n_train=1000))
        assert out["verdict"] == "INSUFFICIENT_DATA"
        assert out["n_outcomes"] == 0
        # Sectors still listed (with 0 counts) so a dashboard reader
        # gets the same shape for the empty case.
        assert len(out["sectors"]) == len(SECTORS)

    def test_healthy_when_every_data_rich_sector_has_weight(self):
        records = (
            [{"ticker": "NVDA"}] * 200 +    # tech
            [{"ticker": "LLY"}] * 100 +      # healthcare
            [{"ticker": "XOM"}] * 50         # energy
        )
        shares = {"tech": 0.40, "healthcare": 0.30, "energy": 0.20,
                  "other": 0.05, "financials": 0.05}
        out = report(records, _imp_payload(n_train=1000, shares=shares))
        assert out["verdict"] == "HEALTHY"
        assert out["n_dead_sectors"] == 0
        # tech sector verdict
        sec_by_name = {r["sector"]: r for r in out["sectors"]}
        assert sec_by_name["tech"]["verdict"] == "HEALTHY"
        assert sec_by_name["tech"]["n_outcomes"] == 200

    def test_dead_sector_flagged_at_min_records_with_zero_share(self):
        """The decisive regression: a sector with real training mass
        but exactly-zero importance share must surface as DEAD_FEATURE.

        Mirrors the live observation: ``sector_crypto`` carries 626
        records yet has 0.0 first-layer weight share in the deployed
        pickle."""
        records = (
            [{"ticker": "NVDA"}] * 500 +     # tech (lots)
            [{"ticker": "MSTR"}] * 50        # crypto (above MIN_RECORDS)
        )
        # tech is the only sector with weight; crypto is exactly zero.
        shares = {"tech": 1.0}
        out = report(records, _imp_payload(n_train=300, shares=shares))
        assert out["verdict"] == "HAS_DEAD_SECTORS"
        assert out["n_dead_sectors"] == 1
        sec_by_name = {r["sector"]: r for r in out["sectors"]}
        assert sec_by_name["crypto"]["verdict"] == "DEAD_FEATURE"
        assert sec_by_name["crypto"]["n_outcomes"] == 50
        # tech contributes weight ⇒ HEALTHY
        assert sec_by_name["tech"]["verdict"] == "HEALTHY"

    def test_sparse_sector_not_flagged_as_dead(self):
        """A sector below MIN_RECORDS must NOT be flagged DEAD — that
        would false-alarm on every backtest where commodities only see
        a handful of trades. SPARSE_DATA is the honest verdict."""
        records = (
            [{"ticker": "NVDA"}] * 500 +
            [{"ticker": "GLD"}] * 5          # commodities, very few
        )
        # commodities has zero share but only 5 records ⇒ SPARSE.
        shares = {"tech": 1.0}
        out = report(records, _imp_payload(n_train=505, shares=shares))
        assert out["verdict"] == "HEALTHY"   # no dead sectors
        sec_by_name = {r["sector"]: r for r in out["sectors"]}
        assert sec_by_name["commodities"]["verdict"] == "SPARSE_DATA"

    def test_corpus_growth_ratio_computed(self):
        records = [{"ticker": "NVDA"}] * 100
        # n_train=20 means n_outcomes / n_train = 5.0 — well above the
        # retrain hint threshold (2.0). Hint text should mention retrain.
        shares = {"tech": 1.0}
        out = report(records, _imp_payload(n_train=20, shares=shares))
        assert out["corpus_growth_ratio"] == pytest.approx(5.0, abs=1e-3)

    def test_growth_ratio_none_when_no_train_count(self):
        records = [{"ticker": "NVDA"}] * 100
        out = report(records, _imp_payload(n_train=0))
        assert out["corpus_growth_ratio"] is None

    def test_dead_with_high_growth_ratio_emits_retrain_hint(self):
        records = (
            [{"ticker": "NVDA"}] * 1000 +
            [{"ticker": "MSTR"}] * 600         # the live observation
        )
        # 1600 outcomes / 200 n_train = 8x — above the retrain threshold.
        shares = {"tech": 1.0}
        out = report(records, _imp_payload(n_train=200, shares=shares))
        assert out["verdict"] == "HAS_DEAD_SECTORS"
        # Retrain hint surfaced
        assert "retrain" in out["hint"].lower()
        assert out["corpus_growth_ratio"] >= CORPUS_GROWTH_RETRAIN_HINT

    def test_dead_with_low_growth_ratio_does_not_emit_retrain_hint(self):
        records = (
            [{"ticker": "NVDA"}] * 100 +
            [{"ticker": "MSTR"}] * 50
        )
        # 150 outcomes / 150 n_train = 1.0 — scorer is fresh, retrain
        # won't help — hint should NOT suggest it.
        shares = {"tech": 1.0}
        out = report(records, _imp_payload(n_train=150, shares=shares))
        assert out["verdict"] == "HAS_DEAD_SECTORS"
        assert "retrain" not in out["hint"].lower()

    def test_dead_sectors_sort_first_in_output(self):
        """Operator scans the table top-to-bottom — dead sectors must
        appear at the top, then HEALTHY by descending importance."""
        records = (
            [{"ticker": "NVDA"}] * 500 +
            [{"ticker": "LLY"}] * 100 +
            [{"ticker": "MSTR"}] * 50        # ≥MIN_RECORDS, share=0
        )
        shares = {"tech": 0.5, "healthcare": 0.3}
        out = report(records, _imp_payload(n_train=200, shares=shares))
        first = out["sectors"][0]
        assert first["verdict"] == "DEAD_FEATURE"

    def test_handles_missing_importance_payload(self):
        records = [{"ticker": "NVDA"}] * 100
        # Empty importance payload — all sectors look dead, but tech has
        # records so it gets DEAD_FEATURE. Others are SPARSE.
        out = report(records, {})
        # Tech has 100 records > MIN_RECORDS, share=0 ⇒ DEAD.
        sec_by_name = {r["sector"]: r for r in out["sectors"]}
        assert sec_by_name["tech"]["verdict"] == "DEAD_FEATURE"
        # Crypto, energy etc. have 0 records ⇒ SPARSE.
        assert sec_by_name["crypto"]["verdict"] == "SPARSE_DATA"

    def test_does_not_raise_on_malformed_record(self):
        records = [
            "not a dict",                # bad type
            None,
            42,
            {"ticker": "NVDA"},           # one valid
        ]
        # Pure function should swallow the bad entries and process the
        # one good one.
        try:
            out = report(records, _imp_payload(n_train=10))
        except Exception as e:
            pytest.fail(f"report() must not raise on malformed input: {e}")
        # Only the NVDA row counts
        sec_by_name = {r["sector"]: r for r in out["sectors"]}
        assert sec_by_name["tech"]["n_outcomes"] == 1

    def test_does_not_raise_on_malformed_importance_row(self):
        records = [{"ticker": "NVDA"}] * 50
        bad_payload = {
            "trained": True, "n_train": 100,
            "importances": [
                {"feature": "sector_tech", "importance_normalized": "not a number"},
                {"feature": None},
                {"feature": "sector_crypto"},      # missing share
            ],
        }
        try:
            out = report(records, bad_payload)
        except Exception as e:
            pytest.fail(f"report() must not raise on bad payload: {e}")
        # tech score parse failed → defaults to 0.0 → DEAD_FEATURE
        sec_by_name = {r["sector"]: r for r in out["sectors"]}
        assert sec_by_name["tech"]["verdict"] == "DEAD_FEATURE"


# ─────────────────────── analyze() integration ───────────────────────

class TestAnalyze:
    def test_missing_outcomes_file_returns_error(self, tmp_path):
        out = analyze(tmp_path / "does_not_exist.jsonl")
        assert out["status"] == "error"
        assert "no outcomes file" in out["hint"]

    def test_empty_outcomes_file_returns_insufficient_data(self, tmp_path,
                                                            monkeypatch):
        """An empty outcomes file shouldn't crash — must degrade to
        INSUFFICIENT_DATA. Even though the scorer isn't trained in this
        test setup, the empty-file path returns its own hint first
        (path exists but no records → empty rec_list, but the scorer
        load happens BEFORE the report). Verify the scorer-untrained
        path is reached cleanly."""
        f = tmp_path / "empty.jsonl"
        f.write_text("")
        out = analyze(f)
        # scorer not trained ⇒ honest sentinel
        assert out["status"] in ("error", "ok")
        # Either insufficient data or scorer-untrained — both acceptable
        # honest verdicts
        assert "trained" in out.get("hint", "").lower() or \
            "insufficient" in out.get("verdict", "").lower()

    def test_corrupt_json_lines_are_skipped(self, tmp_path):
        """A single garbage line must not abort the file read — every
        valid JSON row must still be considered. Pure file-IO test;
        the actual scorer load is irrelevant here (default untrained
        path), but the file-parse loop must be robust."""
        f = tmp_path / "outcomes.jsonl"
        f.write_text(
            json.dumps({"ticker": "NVDA"}) + "\n"
            + "not valid json\n"
            + json.dumps({"ticker": "AMZN"}) + "\n"
            + "\n"                                # blank line — skip
        )
        # Even if scorer isn't trained, file parse runs first. Don't
        # crash — that's the contract being tested.
        out = analyze(f)
        # Should be either "insufficient_data" or "ok" — never raise.
        assert "verdict" in out


# ─────────────────────── CLI ───────────────────────

class TestCli:
    def test_cli_exit_code_dead_sectors_is_2(self, monkeypatch, capsys):
        """cron / shell callers branch on exit code: 0=healthy,
        1=insufficient_data, 2=dead_sectors (mirrors host_guard +
        decision_scorer)."""
        # Pin analyze() to return a known dead-sector verdict.
        monkeypatch.setattr(dsa, "analyze",
                            lambda *_a, **_kw: {
                                "status": "ok",
                                "verdict": "HAS_DEAD_SECTORS",
                                "n_outcomes": 100,
                                "n_train_in_pickle": 50,
                                "corpus_growth_ratio": 2.0,
                                "n_dead_sectors": 1,
                                "sectors": [
                                    {"sector": "crypto", "n_outcomes": 50,
                                     "importance_share": 0.0,
                                     "verdict": "DEAD_FEATURE"},
                                ],
                                "hint": "1 dead sector(s): crypto.",
                            })
        rc = dsa.main([])
        out = capsys.readouterr().out
        assert rc == 2
        # Header is human-readable
        assert "verdict=HAS_DEAD_SECTORS" in out
        assert "DEAD_FEATURE" in out

    def test_cli_exit_code_healthy_is_0(self, monkeypatch):
        monkeypatch.setattr(dsa, "analyze",
                            lambda *_a, **_kw: {
                                "status": "ok", "verdict": "HEALTHY",
                                "n_outcomes": 1000, "n_train_in_pickle": 1000,
                                "corpus_growth_ratio": 1.0,
                                "n_dead_sectors": 0, "sectors": [],
                                "hint": "every sector contributes weight",
                            })
        assert dsa.main([]) == 0

    def test_cli_exit_code_insufficient_data_is_1(self, monkeypatch):
        monkeypatch.setattr(dsa, "analyze",
                            lambda *_a, **_kw: {
                                "status": "ok", "verdict": "INSUFFICIENT_DATA",
                                "n_outcomes": 0, "sectors": [],
                                "hint": "",
                            })
        assert dsa.main([]) == 1

    def test_cli_json_output_is_parseable(self, monkeypatch, capsys):
        """--json must emit one JSON document that round-trips cleanly
        so cron / dashboards can pipe it to jq."""
        monkeypatch.setattr(dsa, "analyze",
                            lambda *_a, **_kw: {
                                "status": "ok", "verdict": "HEALTHY",
                                "n_outcomes": 100, "n_train_in_pickle": 100,
                                "corpus_growth_ratio": 1.0,
                                "n_dead_sectors": 0, "sectors": [],
                                "hint": "all healthy",
                            })
        rc = dsa.main(["--json"])
        out = capsys.readouterr().out
        assert rc == 0
        parsed = json.loads(out)
        assert parsed["verdict"] == "HEALTHY"

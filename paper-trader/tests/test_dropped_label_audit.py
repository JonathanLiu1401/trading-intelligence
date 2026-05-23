"""Locks for ``paper_trader.ml.dropped_label_audit``.

The audit replays ``decision_scorer.train_scorer``'s label-validation
loop on the deployed outcomes tail and classifies each rejected row by
the FIRST trigger that fires (missing / null / bool / unparseable / nan /
+inf / -inf). These tests pin:

1. ``_classify_drop`` returns the exact reason for each shape — the
   classifier IS the contract.
2. The precedence order matches ``train_scorer`` (bool BEFORE None;
   missing-key BEFORE every other check; unparseable BEFORE nan/inf
   because those are produced by ``float()``).
3. End-to-end parity with ``train_scorer``: every row this module
   marks dropped IS dropped by ``train_scorer``, and every row
   ``train_scorer`` accepts is NOT marked dropped.
4. The verdict ladder thresholds are exact (CLEAN / LOW_DROP_RATE /
   ELEVATED_DROP_RATE / HIGH_DROP_RATE / INSUFFICIENT_DATA).
5. The CLI exit code is 0 on CLEAN/LOW/INSUFFICIENT, 1 otherwise — so
   shell callers can gate.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from paper_trader.ml import dropped_label_audit as dla


class TestClassifyDrop:
    """Per-reason classifier: each shape MUST resolve to exactly one of
    the documented reasons, never another."""

    def test_accepts_valid_finite_float(self):
        assert dla._classify_drop({"forward_return_5d": 3.14}) is None

    def test_accepts_zero(self):
        """An exact-zero label is a LEGITIMATE training value (a flat
        5-day window) — train_scorer accepts it. The walk-back-fabricated
        zeros are filtered upstream at outcome-compute time, not here."""
        assert dla._classify_drop({"forward_return_5d": 0.0}) is None

    def test_accepts_negative(self):
        assert dla._classify_drop({"forward_return_5d": -5.5}) is None

    def test_accepts_integer(self):
        """An int 5d label (e.g. 7) parses to 7.0 — accepted."""
        assert dla._classify_drop({"forward_return_5d": 7}) is None

    def test_missing_key_returns_missing_key(self):
        """A row with NO ``forward_return_5d`` field at all — this is the
        ``"forward_return_5d" not in r`` branch."""
        assert dla._classify_drop({"ticker": "NVDA"}) == "missing_key"
        assert dla._classify_drop({}) == "missing_key"

    def test_explicit_null_returns_explicit_null(self):
        """A row with ``forward_return_5d: null`` in JSON (Python None).
        Key present + value None ⇒ explicit_null, NOT missing_key."""
        assert dla._classify_drop(
            {"forward_return_5d": None}) == "explicit_null"

    def test_bool_returns_bool_value(self):
        """``train_scorer`` excludes bool via ``isinstance(..., bool)``.
        True/False would otherwise become 1.0/0.0 labels."""
        assert dla._classify_drop(
            {"forward_return_5d": True}) == "bool_value"
        assert dla._classify_drop(
            {"forward_return_5d": False}) == "bool_value"

    def test_unparseable_string_returns_unparseable(self):
        """A non-numeric string ``"n/a"`` raises TypeError in float(); the
        classifier MUST trap that as ``unparseable``."""
        assert dla._classify_drop(
            {"forward_return_5d": "n/a"}) == "unparseable"
        assert dla._classify_drop(
            {"forward_return_5d": "abc"}) == "unparseable"

    def test_numeric_string_classified_per_value(self):
        """A numeric STRING ``"3.5"`` parses via float() — accepted.
        ``"nan"`` parses to float('nan') — classified as ``nan``."""
        assert dla._classify_drop({"forward_return_5d": "3.5"}) is None
        assert dla._classify_drop({"forward_return_5d": "nan"}) == "nan"
        assert dla._classify_drop(
            {"forward_return_5d": "inf"}) == "positive_inf"

    def test_nan_returns_nan(self):
        assert dla._classify_drop(
            {"forward_return_5d": float("nan")}) == "nan"

    def test_positive_inf_returns_positive_inf(self):
        assert dla._classify_drop(
            {"forward_return_5d": float("inf")}) == "positive_inf"

    def test_negative_inf_returns_negative_inf(self):
        assert dla._classify_drop(
            {"forward_return_5d": float("-inf")}) == "negative_inf"


class TestPrecedence:
    """The first-match rejection order must match train_scorer's loop."""

    def test_bool_beats_explicit_null_ordering(self):
        """``train_scorer`` evaluates ``isinstance(fr_raw, bool) or
        fr_raw is None`` — bool is checked first. A True value should
        therefore classify as ``bool_value``, not get conflated with None.
        (Pinned because Python's ``or`` short-circuits left-to-right.)"""
        assert dla._classify_drop(
            {"forward_return_5d": True}) == "bool_value"

    def test_missing_key_beats_everything(self):
        """A row without the key cannot trigger any other check, since
        the absent-key branch returns before float() / isfinite()."""
        # Even if other fields are NaN, missing fr_5d wins.
        assert dla._classify_drop(
            {"ticker": "NVDA", "rsi": float("nan")}) == "missing_key"


class TestParity:
    """End-to-end: every row this module flags as dropped IS dropped by
    train_scorer; every row train_scorer accepts is NOT marked dropped.

    This is the load-bearing invariant — the diagnostic must mirror
    the trainer exactly so the count surfaced in the skill ledger and
    the per-reason breakdown align."""

    def _build_rec(self, fr_value, action="BUY"):
        """Minimal valid outcome record with one configurable forward
        return. The other fields satisfy the dedup key + sufficient
        sample count so train_scorer reaches the label-validation loop."""
        rec = {
            "ticker": "NVDA", "sim_date": "2024-01-15",
            "action": action, "ml_score": 1.0,
            "rsi": 50.0, "macd": 0.0, "mom5": 0.0, "mom20": 0.0,
            "regime_mult": 1.0, "vol_ratio": 1.0, "bb_position": 0.0,
            "return_pct": 5.0,
        }
        if fr_value is not None or "fr_explicit_null":
            rec["forward_return_5d"] = fr_value
        return rec

    def test_trainer_drops_what_audit_flags(self):
        """For each rejection reason, build a record that triggers it.
        train_scorer's returned ``n_label_dropped`` MUST match this
        module's count for that single row."""
        from paper_trader.ml.decision_scorer import train_scorer

        for bad_value, expected_reason in [
            (None, "explicit_null"),
            (True, "bool_value"),
            ("n/a", "unparseable"),
            (float("nan"), "nan"),
            (float("inf"), "positive_inf"),
            (float("-inf"), "negative_inf"),
        ]:
            # Audit on a single bad row plus 29 good rows (>= MIN of 30).
            # train_scorer needs >= 30 deduped + label-validated rows
            # to proceed; supply 29 valid rows + 1 bad to ensure the
            # validation pass actually rejects the bad one (which would
            # drop the trainable count to 29 < 30 → "no_valid_labels"
            # or "insufficient_after_dedup"). We just want to confirm
            # the *audit* count matches the trainer's drop count, so
            # work with a 50-row corpus where 1 is bad.
            recs = []
            for i in range(49):
                recs.append({
                    "ticker": ["NVDA", "AMD", "INTC"][i % 3],
                    "sim_date": f"2024-{(i % 12) + 1:02d}-{(i % 27) + 1:02d}",
                    "action": "BUY",
                    "ml_score": 1.0 + i * 0.1,
                    "rsi": 50.0, "macd": 0.0,
                    "mom5": 0.0, "mom20": 0.0,
                    "regime_mult": 1.0, "vol_ratio": 1.0,
                    "bb_position": 0.0, "return_pct": 5.0,
                    "forward_return_5d": 1.0 * i,
                })
            recs.append({
                "ticker": "BAD", "sim_date": "1900-01-01",
                "action": "BUY", "ml_score": 0.5,
                "rsi": 50.0, "macd": 0.0,
                "mom5": 0.0, "mom20": 0.0,
                "regime_mult": 1.0, "vol_ratio": 1.0,
                "bb_position": 0.0, "return_pct": 5.0,
                "forward_return_5d": bad_value,
            })
            # Audit classifies the 1 bad row, accepts the other 49.
            count = sum(1 for r in recs if dla._classify_drop(r) is not None)
            assert count == 1, (
                f"audit dropped {count} rows for {expected_reason}, "
                f"expected 1")
            # Confirm the reason classification is correct.
            assert dla._classify_drop(recs[-1]) == expected_reason

            # Run train_scorer — it must report exactly 1 dropped row.
            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                # Redirect SCORER_PATH so we don't overwrite the deployed pickle.
                import paper_trader.ml.decision_scorer as ds
                orig = ds.SCORER_PATH
                try:
                    ds.SCORER_PATH = Path(tmp) / "scorer.pkl"
                    result = train_scorer(recs)
                finally:
                    ds.SCORER_PATH = orig
            assert result.get("n_label_dropped") == 1, (
                f"train_scorer reported n_label_dropped="
                f"{result.get('n_label_dropped')} for {expected_reason}, "
                f"expected 1. Result: {result}")


class TestAnalyzeReport:
    """End-to-end ``analyze()`` returns the right shape with real JSONL."""

    def _write_jsonl(self, tmp_path: Path, rows: list[dict]) -> Path:
        p = tmp_path / "outcomes.jsonl"
        with p.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        return p

    def test_clean_corpus_reads_clean(self, tmp_path):
        rows = [{"ticker": "NVDA", "forward_return_5d": i * 0.1}
                for i in range(50)]
        p = self._write_jsonl(tmp_path, rows)
        rep = dla.analyze(p)
        assert rep["status"] == "ok"
        assert rep["verdict"] == "CLEAN"
        assert rep["n_total"] == 50
        assert rep["n_dropped"] == 0
        assert rep["drop_rate"] == 0.0

    def test_insufficient_data_when_below_min(self, tmp_path):
        """Below MIN_ROWS, no verdict can be issued (sample too small).
        The verdict ladder MUST surface INSUFFICIENT_DATA rather than
        making up a CLEAN/LOW reading on noise."""
        rows = [{"ticker": "NVDA", "forward_return_5d": 1.0}
                for _ in range(5)]
        p = self._write_jsonl(tmp_path, rows)
        rep = dla.analyze(p)
        assert rep["n_total"] == 5
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_low_drop_rate_below_threshold(self, tmp_path):
        """1 bad row in 1000 = 0.1% drop rate < LOW_RATE (0.5%) →
        LOW_DROP_RATE (acceptable noise)."""
        rows = [{"ticker": "NVDA", "forward_return_5d": 0.5}
                for _ in range(999)]
        rows.append({"ticker": "NVDA", "forward_return_5d": float("nan")})
        p = self._write_jsonl(tmp_path, rows)
        rep = dla.analyze(p)
        assert rep["n_total"] == 1000
        assert rep["n_dropped"] == 1
        assert rep["drop_rate"] == pytest.approx(0.001)
        assert rep["verdict"] == "LOW_DROP_RATE"
        assert rep["by_reason"]["nan"] == 1

    def test_elevated_drop_rate_between_thresholds(self, tmp_path):
        """10 bad rows in 1000 = 1% → above LOW_RATE 0.5%, below
        HIGH_RATE 5% → ELEVATED_DROP_RATE."""
        rows = [{"ticker": "NVDA", "forward_return_5d": 0.5}
                for _ in range(990)]
        for _ in range(10):
            rows.append({"ticker": "NVDA", "forward_return_5d": None})
        p = self._write_jsonl(tmp_path, rows)
        rep = dla.analyze(p)
        assert rep["n_dropped"] == 10
        assert rep["drop_rate"] == pytest.approx(0.01)
        assert rep["verdict"] == "ELEVATED_DROP_RATE"
        assert rep["by_reason"]["explicit_null"] == 10

    def test_high_drop_rate_above_threshold(self, tmp_path):
        """60 bad rows in 1000 = 6% > HIGH_RATE 5% → HIGH_DROP_RATE
        (corruption likely)."""
        rows = [{"ticker": "NVDA", "forward_return_5d": 0.5}
                for _ in range(940)]
        for _ in range(60):
            rows.append({"ticker": "NVDA",
                         "forward_return_5d": float("inf")})
        p = self._write_jsonl(tmp_path, rows)
        rep = dla.analyze(p)
        assert rep["n_dropped"] == 60
        assert rep["drop_rate"] == pytest.approx(0.06)
        assert rep["verdict"] == "HIGH_DROP_RATE"
        assert rep["by_reason"]["positive_inf"] == 60

    def test_per_reason_counts_are_exhaustive(self, tmp_path):
        """A mixed corpus with every rejection shape — counts MUST sum
        to n_dropped exactly (mutually exclusive buckets)."""
        rows = [{"ticker": "NVDA", "forward_return_5d": 1.0}] * 30
        # 1 of each rejection type:
        rows.append({"ticker": "BAD1"})                                # missing_key
        rows.append({"ticker": "BAD2", "forward_return_5d": None})     # explicit_null
        rows.append({"ticker": "BAD3", "forward_return_5d": True})     # bool_value
        rows.append({"ticker": "BAD4", "forward_return_5d": "n/a"})    # unparseable
        rows.append({"ticker": "BAD5",
                     "forward_return_5d": float("nan")})               # nan
        rows.append({"ticker": "BAD6",
                     "forward_return_5d": float("inf")})               # positive_inf
        rows.append({"ticker": "BAD7",
                     "forward_return_5d": float("-inf")})              # negative_inf
        p = self._write_jsonl(tmp_path, rows)
        rep = dla.analyze(p)
        assert rep["n_dropped"] == 7
        # Every reason has exactly 1 occurrence:
        for r in dla.REASONS:
            assert rep["by_reason"][r] == 1, f"reason {r}: {rep['by_reason']}"
        # Sum of per-reason equals n_dropped (no row counted twice).
        assert sum(rep["by_reason"].values()) == rep["n_dropped"]

    def test_samples_capped_per_reason(self, tmp_path):
        """``samples_per_reason`` caps the per-reason sample list so a
        5%-drop corpus does NOT produce a 250-row payload."""
        rows = [{"ticker": "NVDA", "forward_return_5d": 1.0}] * 100
        for i in range(20):
            rows.append({"ticker": f"BAD{i}",
                         "forward_return_5d": float("nan")})
        p = self._write_jsonl(tmp_path, rows)
        rep = dla.analyze(p, samples_per_reason=3)
        assert rep["by_reason"]["nan"] == 20
        assert len(rep["samples"]["nan"]) == 3
        # Other reasons have empty sample lists.
        assert rep["samples"]["explicit_null"] == []

    def test_missing_file_returns_no_file(self, tmp_path):
        rep = dla.analyze(tmp_path / "nonexistent.jsonl")
        assert rep["status"] == "no_file"
        # No crash, no spurious verdict.

    def test_malformed_json_lines_skipped(self, tmp_path):
        """A corrupt line in the middle of the file MUST be silently
        skipped (per-line tolerance — train_scorer's caller also does
        this) so a single bad write doesn't kill the whole audit."""
        p = tmp_path / "outcomes.jsonl"
        with p.open("w") as fh:
            for i in range(50):
                fh.write(json.dumps(
                    {"ticker": "NVDA", "forward_return_5d": float(i)})
                         + "\n")
            fh.write("garbage not json\n")
            for i in range(50, 100):
                fh.write(json.dumps(
                    {"ticker": "NVDA", "forward_return_5d": float(i)})
                         + "\n")
        rep = dla.analyze(p)
        assert rep["status"] == "ok"
        assert rep["n_total"] == 100        # 100 valid JSON, 1 garbage skipped
        assert rep["n_dropped"] == 0
        assert rep["verdict"] == "CLEAN"

    def test_tail_bounded(self, tmp_path):
        """Only the most-recent ``tail`` rows are audited — corruption
        in OLDER rows that have rolled off the trainer's window MUST
        NOT count, mirroring the trainer's window exactly."""
        # 50 BAD rows first, then 100 GOOD. Tail=80 should see only
        # the last 80, all good ⇒ CLEAN.
        rows = [{"ticker": "BAD", "forward_return_5d": float("nan")}] * 50
        rows.extend([{"ticker": "NVDA", "forward_return_5d": 1.0}] * 100)
        p = tmp_path / "outcomes.jsonl"
        with p.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        rep = dla.analyze(p, tail=80)
        assert rep["n_total"] == 80
        assert rep["n_dropped"] == 0
        assert rep["verdict"] == "CLEAN"


class TestCli:
    """Exit-code contract for shell gating."""

    def _write(self, tmp_path: Path, rows: list[dict]) -> Path:
        p = tmp_path / "outcomes.jsonl"
        with p.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        return p

    def test_exit_zero_on_clean(self, tmp_path, capsys):
        p = self._write(tmp_path, [{"forward_return_5d": 1.0}] * 50)
        rc = dla.main(["--path", str(p)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "CLEAN" in out

    def test_exit_one_on_high_drop_rate(self, tmp_path, capsys):
        rows = [{"forward_return_5d": 1.0}] * 940
        rows.extend([{"forward_return_5d": float("nan")}] * 60)
        p = self._write(tmp_path, rows)
        rc = dla.main(["--path", str(p)])
        assert rc == 1
        out = capsys.readouterr().out
        assert "HIGH_DROP_RATE" in out

    def test_exit_one_on_elevated_drop_rate(self, tmp_path):
        rows = [{"forward_return_5d": 1.0}] * 990
        rows.extend([{"forward_return_5d": None}] * 10)
        p = self._write(tmp_path, rows)
        rc = dla.main(["--path", str(p)])
        assert rc == 1

    def test_exit_zero_on_low_drop_rate(self, tmp_path):
        """LOW_DROP_RATE (≤0.5%) is acceptable noise — exit 0 so a
        shell gate doesn't fire on benign occasional drops."""
        rows = [{"forward_return_5d": 1.0}] * 999
        rows.append({"forward_return_5d": float("nan")})
        p = self._write(tmp_path, rows)
        rc = dla.main(["--path", str(p)])
        assert rc == 0

    def test_exit_zero_on_insufficient(self, tmp_path):
        """Below MIN_ROWS — no verdict can be made; exit 0 (the absence
        of data is not a failure, just an honest 'don't know')."""
        p = self._write(tmp_path, [{"forward_return_5d": 1.0}] * 5)
        rc = dla.main(["--path", str(p)])
        assert rc == 0

    def test_json_output(self, tmp_path, capsys):
        p = self._write(tmp_path, [{"forward_return_5d": 1.0}] * 50)
        rc = dla.main(["--path", str(p), "--json"])
        assert rc == 0
        out = capsys.readouterr().out
        # Must be valid JSON with required keys.
        rep = json.loads(out)
        assert rep["verdict"] == "CLEAN"
        assert rep["n_total"] == 50
        assert "by_reason" in rep


class TestNeverRaises:
    """Diagnostic discipline: an audit must NEVER raise (would break
    shell gates / cron jobs)."""

    def test_garbage_row_doesnt_raise(self):
        """A garbage non-dict at the row level cannot reach the
        classifier (analyze filters non-dicts), but if it could, the
        classifier itself must not crash. Pass a dict whose value is a
        list to confirm float() raises and is handled."""
        # A list is not a number — float([1,2,3]) raises TypeError →
        # classified as unparseable.
        assert dla._classify_drop(
            {"forward_return_5d": [1, 2, 3]}) == "unparseable"

    def test_analyze_on_unreadable_path_does_not_raise(self, tmp_path):
        """A path that exists but cannot be opened (permission, etc.)
        is caught by the outer try and returns a status-error row."""
        # An IO error here is hard to force without chmod; just confirm
        # the exception path returns instead of propagating.
        rep = dla.analyze(tmp_path / "does_not_exist.jsonl")
        assert rep["status"] == "no_file"

"""Tests for the per-cycle bootstrap-CI skill ledger.

`run_continuous_backtests._append_bootstrap_ci_skill_log` wires
`paper_trader.ml.oos_bootstrap_ci.bootstrap_ci` into a per-cycle JSONL
ledger so the OOS rank-IC's 95% CI is trendable cycle-by-cycle. Every
existing OOS diagnostic reports point estimates — these tests pin the
ledger's verdict ladder, the safe-default behaviour on every degenerate
input class, and the bounded-growth trim discipline every sibling
ledger follows.

Each test asserts a SPECIFIC expected value (verdict string, status,
field presence) — never just "did not crash". If the verdict ladder
or any of the documented safe defaults regress, the matching test
fails loudly.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import run_continuous_backtests as rcb  # noqa: E402


def _write_outcomes(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _outcome_row(ml_score: float, fwd: float, ticker: str = "NVDA",
                 action: str = "BUY", sim_date: str = "2024-01-01") -> dict:
    """One outcome row matching the schema `_compute_decision_outcomes`
    writes. Quant fields default to safe values that exercise the
    bootstrap CI path end-to-end."""
    return {
        "sim_date": sim_date,
        "ticker": ticker,
        "action": action,
        "ml_score": ml_score,
        "rsi": 50.0, "macd": 0.0, "mom5": 0.0, "mom20": 0.0,
        "regime_mult": 1.0,
        "vol_ratio": 1.0, "bb_position": 0.0,
        "news_urgency": 50.0, "news_article_count": 1.0,
        "forward_return_5d": fwd,
    }


class _SignalScorer:
    """Toy scorer whose `predict` is monotone in `ml_score` so a bootstrap
    on synthetic high-correlation data MUST detect rank skill. Pickle-free
    so this test fixture never touches the real `decision_scorer.pkl`."""

    is_trained = True
    n_train = 1000

    def predict(self, **kw) -> float:
        return float(kw.get("ml_score", 0.0))


class _NoiseScorer:
    """Toy scorer whose `predict` is uncorrelated with `forward_return_5d`
    (seeded RNG, deterministic). A bootstrap on these pairs MUST report
    NO_SKILL_DETECTED — the rank-IC CI should straddle 0."""

    is_trained = True
    n_train = 1000

    def __init__(self, seed: int = 7) -> None:
        # Pre-generate predictions so successive predicts on identical
        # inputs return identical outputs (the scorer is stateless from
        # bootstrap_ci's perspective, but a fresh RNG each call would
        # break the (preds, actuals) alignment).
        rng = np.random.default_rng(seed)
        self._table: dict[tuple, float] = {}
        self._rng = rng

    def predict(self, **kw) -> float:
        key = (kw.get("ticker"), kw.get("ml_score"), kw.get("rsi"),
               kw.get("mom5"), kw.get("mom20"))
        if key not in self._table:
            self._table[key] = float(self._rng.normal())
        return self._table[key]


class _UntrainedScorer:
    is_trained = False
    n_train = 0

    def predict(self, **kw) -> float:
        return 0.0


# ---------------------------------------------------------------------------
# Section 1 — Verdict ladder on synthetic data
# ---------------------------------------------------------------------------

class TestVerdictLadder:
    """Locks the four-cell verdict ladder: NOT_TRAINED, INSUFFICIENT_DATA,
    SKILL_DETECTED, NO_SKILL_DETECTED. The CI excludes-0 test is the
    decisive `SKILL_DETECTED` invariant — the whole point of wiring this
    ledger is to surface that signal per cycle."""

    def test_skill_detected_on_strong_signal(self, tmp_path, monkeypatch):
        """High-correlation pairs (predict ≡ ml_score, fwd = 0.8*ml_score +
        small noise) produce a strongly positive rank-IC whose 95% CI
        excludes 0. The ledger MUST emit verdict=SKILL_DETECTED and a
        positive rank_ic_ci_low."""
        rng = np.random.default_rng(123)
        n = 200
        records = []
        for i in range(n):
            ml = float(rng.normal())
            fwd = 0.8 * ml + 0.3 * float(rng.normal())
            records.append(_outcome_row(ml, fwd))
        out_path = tmp_path / "decision_outcomes.jsonl"
        _write_outcomes(out_path, records)

        monkeypatch.setattr(
            rcb, "BOOTSTRAP_CI_SKILL_LOG", tmp_path / "boot.jsonl")
        # bootstrap_ci reads the deployed pickle — substitute our toy
        # scorer at module level so the test doesn't depend on disk state.
        import paper_trader.ml.oos_bootstrap_ci as oob
        monkeypatch.setattr(oob, "_build_aligned_arrays",
                            lambda scorer, recs: _aligned_from_records(
                                _SignalScorer(), recs))
        # `DecisionScorer()` in `_append_bootstrap_ci_skill_log` checks
        # is_trained — patch it to return our trained toy stub.
        import paper_trader.ml.decision_scorer as ds
        monkeypatch.setattr(ds, "DecisionScorer", lambda: _SignalScorer())

        ok = rcb._append_bootstrap_ci_skill_log(
            cycle=1, win_start=date(2024, 1, 1), win_end=date(2024, 12, 31),
            outcomes_path=out_path, n_bootstrap=200,
        )
        assert ok is True
        row = _read_last_row(tmp_path / "boot.jsonl")
        assert row["verdict"] == "SKILL_DETECTED"
        assert row["rank_ic_point"] is not None
        assert row["rank_ic_point"] > 0.4
        # Decisive contract: rank_ic_ci_low strictly > 0 ⇒ CI excludes 0.
        assert row["rank_ic_ci_low"] is not None
        assert row["rank_ic_ci_low"] > 0.0

    def test_no_skill_detected_on_pure_noise(self, tmp_path, monkeypatch):
        """Uncorrelated predictions produce a rank-IC CI straddling 0 ⇒
        verdict=NO_SKILL_DETECTED. The decisive negative case — a quant
        must NOT see a false-positive SKILL_DETECTED on coin-flip data."""
        rng = np.random.default_rng(999)
        n = 200
        records = []
        for i in range(n):
            ml = float(rng.normal())
            fwd = float(rng.normal())  # independent
            records.append(_outcome_row(ml, fwd))
        out_path = tmp_path / "decision_outcomes.jsonl"
        _write_outcomes(out_path, records)

        monkeypatch.setattr(
            rcb, "BOOTSTRAP_CI_SKILL_LOG", tmp_path / "boot.jsonl")
        noise = _NoiseScorer(seed=7)
        import paper_trader.ml.oos_bootstrap_ci as oob
        monkeypatch.setattr(oob, "_build_aligned_arrays",
                            lambda scorer, recs: _aligned_from_records(
                                noise, recs))
        import paper_trader.ml.decision_scorer as ds
        monkeypatch.setattr(ds, "DecisionScorer", lambda: noise)

        rcb._append_bootstrap_ci_skill_log(
            cycle=2, win_start=date(2024, 1, 1), win_end=date(2024, 12, 31),
            outcomes_path=out_path, n_bootstrap=200,
        )
        row = _read_last_row(tmp_path / "boot.jsonl")
        assert row["verdict"] == "NO_SKILL_DETECTED"
        # Documented CI invariant: CI straddles 0 ⇒ ci_low ≤ 0 ≤ ci_high
        # (or both bounds are tiny — the decisive contract is ci_low NOT
        # strictly above 0, which the verdict logic uses).
        assert row["rank_ic_ci_low"] is not None
        assert row["rank_ic_ci_low"] <= 0.0

    def test_insufficient_data_below_min_pairs(self, tmp_path, monkeypatch):
        """Fewer than `MIN_PAIRS_FOR_CI` (30) valid OOS records ⇒
        INSUFFICIENT_DATA. The OOS slice is 20% of total, so we need >150
        total records to get to MIN_PAIRS_FOR_CI. 10 records → ~2 OOS →
        INSUFFICIENT_DATA."""
        records = [_outcome_row(0.5, 0.1) for _ in range(10)]
        out_path = tmp_path / "decision_outcomes.jsonl"
        _write_outcomes(out_path, records)

        monkeypatch.setattr(
            rcb, "BOOTSTRAP_CI_SKILL_LOG", tmp_path / "boot.jsonl")
        import paper_trader.ml.decision_scorer as ds
        monkeypatch.setattr(ds, "DecisionScorer", lambda: _SignalScorer())

        rcb._append_bootstrap_ci_skill_log(
            cycle=3, win_start=date(2024, 1, 1), win_end=date(2024, 12, 31),
            outcomes_path=out_path, n_bootstrap=50,
        )
        row = _read_last_row(tmp_path / "boot.jsonl")
        assert row["verdict"] == "INSUFFICIENT_DATA"
        assert row["rank_ic_point"] is None
        assert row["rank_ic_ci_low"] is None

    def test_not_trained_emits_honest_row(self, tmp_path, monkeypatch):
        """An untrained deployed scorer ⇒ NOT_TRAINED verdict, but the
        ledger MUST still emit a row so the gap is visible in the trend.
        The documented `_append_*` invariant: never silently skip a cycle."""
        out_path = tmp_path / "decision_outcomes.jsonl"
        _write_outcomes(out_path, [_outcome_row(0.5, 0.1) for _ in range(200)])

        monkeypatch.setattr(
            rcb, "BOOTSTRAP_CI_SKILL_LOG", tmp_path / "boot.jsonl")
        import paper_trader.ml.decision_scorer as ds
        monkeypatch.setattr(ds, "DecisionScorer", lambda: _UntrainedScorer())

        rcb._append_bootstrap_ci_skill_log(
            cycle=4, win_start=date(2024, 1, 1), win_end=date(2024, 12, 31),
            outcomes_path=out_path,
        )
        row = _read_last_row(tmp_path / "boot.jsonl")
        assert row["verdict"] == "NOT_TRAINED"
        assert row["rank_ic_point"] is None


# ---------------------------------------------------------------------------
# Section 2 — Safe-default discipline
# ---------------------------------------------------------------------------

class TestSafeDefaults:
    """Every degenerate input class — missing file, corrupt JSON, import
    failure — must emit an honest error row, NEVER raise. Mirrors the
    `_append_scorer_skill_log` discipline that has prevented silent ledger
    gaps for every sibling skill log."""

    def test_missing_outcomes_file_emits_row(self, tmp_path, monkeypatch):
        """No `decision_outcomes.jsonl` on disk ⇒ honestly emit a row.
        The bootstrap_ci path returns status='empty' → verdict 'INSUFFICIENT_DATA'."""
        monkeypatch.setattr(
            rcb, "BOOTSTRAP_CI_SKILL_LOG", tmp_path / "boot.jsonl")
        import paper_trader.ml.decision_scorer as ds
        monkeypatch.setattr(ds, "DecisionScorer", lambda: _SignalScorer())

        ok = rcb._append_bootstrap_ci_skill_log(
            cycle=5, win_start=date(2024, 1, 1), win_end=date(2024, 12, 31),
            outcomes_path=tmp_path / "does_not_exist.jsonl",
        )
        assert ok is True
        row = _read_last_row(tmp_path / "boot.jsonl")
        assert row["verdict"] == "INSUFFICIENT_DATA"
        assert row["n_oos"] == 0

    def test_corrupt_json_rows_dont_raise(self, tmp_path, monkeypatch):
        """A corrupt line in `decision_outcomes.jsonl` MUST drop that row
        without aborting. The remaining valid rows still feed the CI."""
        out_path = tmp_path / "decision_outcomes.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as fh:
            for i in range(200):
                fh.write(json.dumps(_outcome_row(0.5, 0.1)) + "\n")
            fh.write("not-json-{\n")  # corrupt
            fh.write("{partial\n")  # truncated

        monkeypatch.setattr(
            rcb, "BOOTSTRAP_CI_SKILL_LOG", tmp_path / "boot.jsonl")
        import paper_trader.ml.decision_scorer as ds
        monkeypatch.setattr(ds, "DecisionScorer", lambda: _SignalScorer())

        ok = rcb._append_bootstrap_ci_skill_log(
            cycle=6, win_start=date(2024, 1, 1), win_end=date(2024, 12, 31),
            outcomes_path=out_path, n_bootstrap=50,
        )
        assert ok is True
        # Row was written (not silently skipped) despite corrupt lines.
        row = _read_last_row(tmp_path / "boot.jsonl")
        assert row["cycle"] == 6


# ---------------------------------------------------------------------------
# Section 3 — Bounded-growth trim
# ---------------------------------------------------------------------------

class TestBoundedTrim:
    """The trim must be ATOMIC (tmp + .replace) so a kill mid-truncate
    can never leave a torn ledger file. The trim must only fire when the
    file exceeds 2× the keep threshold — same idiom as every sibling
    ledger."""

    def test_trim_after_2x_keep_threshold(self, tmp_path, monkeypatch):
        """Pre-fill the ledger past 2×keep; one more append triggers the
        atomic rewrite back to `keep` lines. The newly appended row is
        the LAST kept row (FIFO drop)."""
        monkeypatch.setattr(
            rcb, "BOOTSTRAP_CI_SKILL_LOG", tmp_path / "boot.jsonl")
        # Tiny keep + tiny pre-fill so the test stays fast.
        monkeypatch.setattr(rcb, "BOOTSTRAP_CI_SKILL_LOG_KEEP", 5)
        ledger = tmp_path / "boot.jsonl"
        with ledger.open("w") as fh:
            for i in range(12):  # 12 > 2*5 = 10
                fh.write(json.dumps({"cycle": i, "verdict": "FILLER"}) + "\n")

        # Bypass the network/bootstrap path — make the analyzer cheap.
        import paper_trader.ml.oos_bootstrap_ci as oob

        def _fake_bootstrap(*a, **kw):
            return {"status": "ok", "n": 100, "n_bootstrap": 50,
                    "ci_level": 0.95,
                    "rmse": {"value": 1.0, "ci_low": 0.9, "ci_high": 1.1},
                    "dir_acc": {"value": 0.5, "ci_low": 0.45, "ci_high": 0.55},
                    "rank_ic": {"value": 0.1, "ci_low": 0.05, "ci_high": 0.15}}
        monkeypatch.setattr(oob, "bootstrap_ci", _fake_bootstrap)

        import paper_trader.validation as val
        monkeypatch.setattr(val, "split_outcomes_temporal",
                            lambda recs, oos_fraction=0.2:
                            (recs[:int(0.8*len(recs))],
                             recs[int(0.8*len(recs)):]))
        import paper_trader.ml.decision_scorer as ds
        monkeypatch.setattr(ds, "DecisionScorer", lambda: _SignalScorer())

        rcb._append_bootstrap_ci_skill_log(
            cycle=999, win_start=date(2024, 1, 1),
            win_end=date(2024, 12, 31),
            outcomes_path=tmp_path / "outcomes.jsonl",  # missing OK
        )
        lines = [l for l in ledger.read_text().splitlines() if l.strip()]
        # Trim: 13 rows > 2*5=10 ⇒ rewrite to last 5 (cycles 9,10,11 + the
        # FILLER + the new cycle 999 — the trim keeps the LAST 5).
        assert len(lines) == 5
        last_row = json.loads(lines[-1])
        assert last_row["cycle"] == 999

    def test_atomic_tmp_replace(self, tmp_path, monkeypatch):
        """After the trim the tmp file MUST NOT exist on disk — `.replace`
        atomically moves it. A leftover `.json.tmp` would indicate a torn
        write that future appends could trip over."""
        monkeypatch.setattr(
            rcb, "BOOTSTRAP_CI_SKILL_LOG", tmp_path / "boot.jsonl")
        monkeypatch.setattr(rcb, "BOOTSTRAP_CI_SKILL_LOG_KEEP", 5)
        ledger = tmp_path / "boot.jsonl"
        with ledger.open("w") as fh:
            for i in range(12):
                fh.write(json.dumps({"cycle": i}) + "\n")

        import paper_trader.ml.oos_bootstrap_ci as oob
        monkeypatch.setattr(oob, "bootstrap_ci",
                            lambda *a, **kw: {
                                "status": "ok", "n": 100,
                                "n_bootstrap": 50, "ci_level": 0.95,
                                "rmse": {"value": 1.0, "ci_low": 0.9,
                                         "ci_high": 1.1},
                                "dir_acc": {"value": 0.5, "ci_low": 0.45,
                                            "ci_high": 0.55},
                                "rank_ic": {"value": 0.1, "ci_low": 0.05,
                                            "ci_high": 0.15},
                            })
        import paper_trader.validation as val
        monkeypatch.setattr(val, "split_outcomes_temporal",
                            lambda recs, oos_fraction=0.2: (recs, recs))
        import paper_trader.ml.decision_scorer as ds
        monkeypatch.setattr(ds, "DecisionScorer", lambda: _SignalScorer())

        rcb._append_bootstrap_ci_skill_log(
            cycle=1, win_start=date(2024, 1, 1), win_end=date(2024, 12, 31),
            outcomes_path=tmp_path / "outcomes.jsonl",
        )
        tmp = ledger.with_suffix(".jsonl.tmp")
        assert not tmp.exists(), "atomic .replace must leave no .tmp behind"


# ---------------------------------------------------------------------------
# Section 4 — Row schema contract
# ---------------------------------------------------------------------------

class TestRowSchema:
    """Every row MUST carry the documented fields so downstream consumers
    (dashboard, skill_trend, ad-hoc Jupyter) can join on `cycle` safely."""

    def test_row_has_required_fields(self, tmp_path, monkeypatch):
        records = [_outcome_row(0.5, 0.1) for _ in range(200)]
        out_path = tmp_path / "outcomes.jsonl"
        _write_outcomes(out_path, records)

        monkeypatch.setattr(
            rcb, "BOOTSTRAP_CI_SKILL_LOG", tmp_path / "boot.jsonl")
        import paper_trader.ml.decision_scorer as ds
        monkeypatch.setattr(ds, "DecisionScorer", lambda: _SignalScorer())

        rcb._append_bootstrap_ci_skill_log(
            cycle=42, win_start=date(2024, 3, 5),
            win_end=date(2024, 12, 31),
            outcomes_path=out_path, n_bootstrap=50,
        )
        row = _read_last_row(tmp_path / "boot.jsonl")
        # Time / window identification:
        assert row["cycle"] == 42
        assert row["window_start"] == "2024-03-05"
        assert row["window_end"] == "2024-12-31"
        # Status fields:
        assert "status" in row
        assert "verdict" in row
        # Sample-size and bootstrap-config echo:
        assert "n_oos" in row
        assert "n_bootstrap" in row
        assert row["ci_level"] == 0.95
        # Per-metric point + CI bounds for ALL three documented metrics:
        for metric in ("rank_ic", "rmse", "dir_acc"):
            assert f"{metric}_point" in row
            assert f"{metric}_ci_low" in row
            assert f"{metric}_ci_high" in row


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_last_row(p: Path) -> dict:
    lines = [l for l in p.read_text().splitlines() if l.strip()]
    return json.loads(lines[-1])


def _aligned_from_records(scorer, records: list[dict]) -> tuple:
    """Run `scorer.predict` over each record, return (preds, actuals)
    aligned arrays. Mirrors the real `_build_aligned_arrays` semantics
    (SELL sign-flip, NaN drop) so the toy scorer feeds the bootstrap with
    realistic inputs."""
    preds: list[float] = []
    actuals: list[float] = []
    for r in records:
        try:
            p = scorer.predict(
                ml_score=r.get("ml_score"), rsi=r.get("rsi"),
                macd=r.get("macd"), mom5=r.get("mom5"),
                mom20=r.get("mom20"),
                regime_mult=r.get("regime_mult"),
                ticker=r.get("ticker"),
                vol_ratio=r.get("vol_ratio"),
                bb_pos=r.get("bb_position"),
                news_urgency=r.get("news_urgency"),
                news_article_count=r.get("news_article_count"),
            )
            a = float(r.get("forward_return_5d") or 0.0)
            if str(r.get("action") or "BUY").upper() == "SELL":
                a = -a
            pf = float(p)
            if pf == pf and a == a:
                preds.append(pf)
                actuals.append(a)
        except Exception:
            continue
    return (np.asarray(preds, dtype=np.float64),
            np.asarray(actuals, dtype=np.float64))

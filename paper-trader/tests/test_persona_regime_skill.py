"""Tests for paper_trader.ml.persona_regime_skill.

The module produces a (persona × regime) cross-tab of decision-signal
rank-IC from decision_outcomes.jsonl rows. These tests pin the exact
verdict ladder, the regime decode (label preferred, mult fallback,
unknown dropped), the SELL double-flip alignment, and the read-only
CLI contract.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from paper_trader.ml import persona_regime_skill as prs
from paper_trader.backtest import persona_for


# Predictable persona names — run_id 1 → "Value Investor", 2 → "Momentum
# Trader", etc. We use these directly in tests so a future PERSONAS
# reorder fails noisily here (the SSOT discipline this module is
# designed to enforce).
PERSONA_1 = persona_for(1)["name"]   # "Value Investor"
PERSONA_2 = persona_for(2)["name"]   # "Momentum Trader"
PERSONA_3 = persona_for(3)["name"]   # "Contrarian"


def _row(run_id: int, regime: str, ml_score: float, fwd_ret: float,
         action: str = "BUY", use_mult: bool = False) -> dict:
    """Build one synthetic decision_outcomes row."""
    mult = {"bull": 1.0, "sideways": 0.6, "bear": 0.3, "unknown": 1.0}[regime]
    row = {
        "run_id": run_id,
        "action": action,
        "ml_score": ml_score,
        "forward_return_5d": fwd_ret,
        "regime_mult": mult,
    }
    # Prefer the explicit label except when explicitly testing legacy rows.
    if not use_mult:
        row["regime_label"] = regime
    return row


# ─────────────────────────── regime decode ───────────────────────────


class TestRegimeDecode:
    def test_explicit_label_preferred(self):
        # regime_label=bull but mult=0.3 (bear); label wins.
        rec = {"regime_label": "bull", "regime_mult": 0.3}
        assert prs._regime_of(rec) == "bull"

    def test_explicit_unknown_dropped(self):
        # "unknown" is THE documented contamination route — must drop.
        rec = {"regime_label": "unknown", "regime_mult": 1.0}
        assert prs._regime_of(rec) is None

    def test_unknown_label_with_no_mult_dropped(self):
        rec = {"regime_label": "alien_regime"}
        assert prs._regime_of(rec) is None

    def test_legacy_mult_fallback_bear(self):
        # No regime_label → fall back to mult decode.
        rec = {"regime_mult": 0.30}
        assert prs._regime_of(rec) == "bear"

    def test_legacy_mult_fallback_sideways(self):
        rec = {"regime_mult": 0.60}
        assert prs._regime_of(rec) == "sideways"

    def test_legacy_mult_fallback_bull(self):
        rec = {"regime_mult": 1.0}
        assert prs._regime_of(rec) == "bull"

    def test_mult_with_float_roundoff(self):
        # JSON roundtripping 0.6 can occasionally land as 0.6000000001.
        # The round-to-2-decimals must still resolve.
        rec = {"regime_mult": 0.6000000000000001}
        assert prs._regime_of(rec) == "sideways"

    def test_missing_both_drops(self):
        assert prs._regime_of({}) is None

    def test_nonfinite_mult_drops(self):
        assert prs._regime_of({"regime_mult": float("nan")}) is None
        assert prs._regime_of({"regime_mult": float("inf")}) is None

    def test_unmapped_mult_drops(self):
        # 0.5 isn't bull/sideways/bear; must not silently bucket.
        assert prs._regime_of({"regime_mult": 0.5}) is None


# ─────────────────────────── alignment (SELL double-flip) ────────────


class TestAligned:
    def test_buy_passes_through(self):
        out = prs._aligned({"action": "BUY", "ml_score": 2.0,
                            "forward_return_5d": 5.0})
        assert out == (2.0, 5.0)

    def test_sell_double_flips(self):
        # A SELL with positive signal that resulted in a price DROP is a
        # WIN — both signal and target get negated so "higher signal ⇒
        # higher realized goodness" stays monotone across BUY and SELL.
        out = prs._aligned({"action": "SELL", "ml_score": 2.0,
                            "forward_return_5d": -5.0})
        assert out == (-2.0, 5.0)

    def test_missing_signal_drops(self):
        assert prs._aligned({"action": "BUY", "forward_return_5d": 1.0,
                             "ml_score": None}) is None

    def test_missing_target_drops(self):
        assert prs._aligned({"action": "BUY", "ml_score": 1.0,
                             "forward_return_5d": None}) is None

    def test_nan_signal_drops(self):
        # _to_float rejects NaN/inf, so an explicitly bad numeric returns
        # NaN sentinel and the t!=t / s!=s guards drop the row.
        assert prs._aligned({"action": "BUY", "ml_score": float("nan"),
                             "forward_return_5d": 1.0}) is None

    def test_nan_target_drops(self):
        assert prs._aligned({"action": "BUY", "ml_score": 1.0,
                             "forward_return_5d": float("nan")}) is None


# ─────────────────────────── per-cell verdict ───────────────────────


class TestCellVerdict:
    def test_insufficient_below_min_per_cell(self):
        assert prs._verdict_for_cell(prs.MIN_PER_CELL - 1, 0.5) == "INSUFFICIENT"

    def test_signal_edge_at_or_above_ic_good(self):
        assert prs._verdict_for_cell(prs.MIN_PER_CELL, prs.IC_GOOD) == "SIGNAL_EDGE"
        assert prs._verdict_for_cell(50, 0.30) == "SIGNAL_EDGE"

    def test_inverted_at_or_below_neg_ic_good(self):
        assert prs._verdict_for_cell(prs.MIN_PER_CELL, -prs.IC_GOOD) == "INVERTED"
        assert prs._verdict_for_cell(50, -0.40) == "INVERTED"

    def test_weak_signal_between_min_and_good(self):
        assert prs._verdict_for_cell(prs.MIN_PER_CELL, prs.IC_MIN) == "WEAK_SIGNAL"
        # Strictly between IC_MIN and IC_GOOD.
        assert prs._verdict_for_cell(50, (prs.IC_MIN + prs.IC_GOOD) / 2) == "WEAK_SIGNAL"

    def test_no_edge_in_dead_zone(self):
        assert prs._verdict_for_cell(50, 0.0) == "NO_EDGE"
        assert prs._verdict_for_cell(50, prs.IC_MIN - 0.001) == "NO_EDGE"
        # Negative-but-not-inverted (above -IC_GOOD).
        assert prs._verdict_for_cell(50, -0.10) == "NO_EDGE"


# ─────────────────────────── analyze: overall verdict ladder ─────────


class TestAnalyzeVerdict:
    def test_empty_is_insufficient(self):
        rep = prs.analyze([])
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_records"] == 0
        assert rep["cells"] == []
        assert rep["best_cell"] is None
        assert rep["worst_cell"] is None

    def test_below_min_records_is_insufficient(self):
        # 29 rows < MIN_RECORDS=30.
        recs = [_row(1, "bull", float(i), float(i)) for i in range(29)]
        rep = prs.analyze(recs)
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_dropped_unknown_regime_counted(self):
        # 25 known-regime rows + 10 unknown-label rows. The 25 are below
        # MIN_RECORDS so verdict is INSUFFICIENT_DATA, but the drop count
        # surfaces honestly.
        recs = [_row(1, "bull", float(i), float(i)) for i in range(25)]
        recs += [_row(1, "unknown", float(i), float(i)) for i in range(10)]
        rep = prs.analyze(recs)
        assert rep["n_dropped_unknown_regime"] == 10
        assert rep["n_records"] == 25
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_signal_edge_in_one_cell_healthy(self):
        # 25 monotone rows in one cell — perfect rank-IC, well above IC_GOOD.
        recs = [_row(1, "bull", float(i), float(i)) for i in range(25)]
        # Filler in a separate cell so n_records ≥ MIN_RECORDS.
        recs += [_row(2, "sideways", 1.0, 0.5) for _ in range(10)]
        rep = prs.analyze(recs)
        assert rep["status"] == "ok"
        # Bull cell has perfect monotone IC = 1.0; sideways cell n=10
        # (below MIN_PER_CELL) is INSUFFICIENT, so no NO_EDGE cell —
        # overall HEALTHY.
        bull_cell = next(c for c in rep["cells"]
                         if c["persona"] == PERSONA_1 and c["regime"] == "bull")
        assert bull_cell["score_ic"] == pytest.approx(1.0)
        assert bull_cell["verdict"] == "SIGNAL_EDGE"
        assert rep["verdict"] == "HEALTHY"

    def test_inverted_cell_flagged(self):
        # 25 rows where higher signal → lower outcome (anti-predictive).
        recs = [_row(1, "bull", float(i), -float(i)) for i in range(25)]
        recs += [_row(2, "sideways", 1.0, 0.5) for _ in range(10)]
        rep = prs.analyze(recs)
        bull_cell = next(c for c in rep["cells"]
                         if c["persona"] == PERSONA_1 and c["regime"] == "bull")
        assert bull_cell["score_ic"] == pytest.approx(-1.0)
        assert bull_cell["verdict"] == "INVERTED"
        assert rep["verdict"] == "HAS_INVERTED_CELL"
        assert len(rep["inverted_cells"]) == 1
        # Inverted cell carries the persona, regime, score_ic, n.
        ic_row = rep["inverted_cells"][0]
        assert ic_row["persona"] == PERSONA_1
        assert ic_row["regime"] == "bull"
        assert ic_row["score_ic"] == pytest.approx(-1.0)
        assert ic_row["n"] == 25

    def test_regime_conditional_when_mixed(self):
        # Same persona, two regimes: bull is SIGNAL_EDGE, sideways is
        # NO_EDGE. That's the textbook REGIME_CONDITIONAL state.
        recs = [_row(1, "bull", float(i), float(i)) for i in range(25)]
        # Sideways: random-walk signal vs target → near-zero IC.
        # Construct with known no-rank: signal ascending, target constant
        # (constant target → spearman=0).
        recs += [_row(1, "sideways", float(i), 0.0) for i in range(25)]
        rep = prs.analyze(recs)
        assert rep["verdict"] == "REGIME_CONDITIONAL"
        bull_cell = next(c for c in rep["cells"]
                         if c["persona"] == PERSONA_1 and c["regime"] == "bull")
        side_cell = next(c for c in rep["cells"]
                         if c["persona"] == PERSONA_1 and c["regime"] == "sideways")
        assert bull_cell["verdict"] == "SIGNAL_EDGE"
        assert side_cell["verdict"] == "NO_EDGE"

    def test_no_persona_edge_when_all_dead(self):
        # Two cells, both NO_EDGE (constant target → IC=0).
        recs = [_row(1, "bull", float(i), 0.0) for i in range(25)]
        recs += [_row(2, "sideways", float(i), 0.0) for i in range(25)]
        rep = prs.analyze(recs)
        assert rep["verdict"] == "NO_PERSONA_EDGE"

    def test_no_persona_edge_when_only_unstable(self):
        # 30 records but split across 30 different (persona, regime) cells
        # so none reach MIN_PER_CELL — every cell INSUFFICIENT.
        recs = []
        # We need ≥ MIN_RECORDS=30 aligned, but no cell ≥ MIN_PER_CELL=20.
        # Spread across 10 personas × 3 regimes = 30 cells (1 row each).
        for run_id in range(1, 11):
            for regime in ("bull", "sideways", "bear"):
                recs.append(_row(run_id, regime, 1.0, 1.0))
        rep = prs.analyze(recs)
        assert rep["verdict"] == "NO_PERSONA_EDGE"
        assert rep["n_stable_cells"] == 0

    def test_best_and_worst_cells_chosen_from_stable_only(self):
        # Stable cell: 25 perfect-IC rows in (persona1, bull) → ic=+1.0
        recs = [_row(1, "bull", float(i), float(i)) for i in range(25)]
        # Stable cell: 25 anti-predictive rows in (persona2, sideways) → ic=-1.0
        recs += [_row(2, "sideways", float(i), -float(i)) for i in range(25)]
        # Unstable: 5 rows in (persona3, bear) — INSUFFICIENT.
        recs += [_row(3, "bear", float(i), float(i) * 10) for i in range(5)]
        rep = prs.analyze(recs)
        # Best stable = persona1/bull, worst stable = persona2/sideways.
        assert rep["best_cell"]["persona"] == PERSONA_1
        assert rep["best_cell"]["regime"] == "bull"
        assert rep["best_cell"]["score_ic"] == pytest.approx(1.0)
        assert rep["worst_cell"]["persona"] == PERSONA_2
        assert rep["worst_cell"]["regime"] == "sideways"
        assert rep["worst_cell"]["score_ic"] == pytest.approx(-1.0)

    def test_cells_sorted_with_insufficient_last(self):
        # One stable SIGNAL_EDGE cell + one stable NO_EDGE cell + one
        # INSUFFICIENT cell. INSUFFICIENT must sort LAST regardless of
        # its IC magnitude.
        recs = [_row(1, "bull", float(i), float(i)) for i in range(25)]   # IC=+1
        recs += [_row(2, "sideways", float(i), 0.0) for i in range(25)]    # IC=0
        # Tiny unstable cell with IC=+1.0 (small-sample). Must sink to end.
        recs += [_row(3, "bear", float(i), float(i)) for i in range(5)]
        rep = prs.analyze(recs)
        # First cell is best stable (IC=+1.0 from persona1/bull).
        assert rep["cells"][0]["verdict"] == "SIGNAL_EDGE"
        # Last cell is the INSUFFICIENT one.
        assert rep["cells"][-1]["verdict"] == "INSUFFICIENT"
        # The INSUFFICIENT cell carries the small-sample IC honestly,
        # even though the verdict precludes acting on it.
        assert rep["cells"][-1]["n"] == 5


# ─────────────────────────── SELL alignment in analyze ───────────────


class TestSellAlignmentInAnalyze:
    def test_sell_negative_return_is_a_win(self):
        # All SELL rows where higher signal coincided with a steeper drop
        # in price (a correct sell). After double-flip both signal AND
        # target are negated, so this is a perfect-monotone case for the
        # aligned IC (positive +1.0, not -1.0).
        recs = [_row(1, "bull", float(i), -float(i), action="SELL")
                for i in range(25)]
        # Filler so MIN_RECORDS clears.
        recs += [_row(2, "sideways", 1.0, 0.5) for _ in range(10)]
        rep = prs.analyze(recs)
        bull_cell = next(c for c in rep["cells"]
                         if c["persona"] == PERSONA_1 and c["regime"] == "bull")
        # The pair (+i, -(-i)) = (+i, +i) post-flip → IC=+1.0
        # Wait — actually _aligned does s=-ml_score AND t=-forward_return.
        # ml_score=i → s=-i. fwd=-i → t=i.
        # So pairs are (-i, +i) — monotone DECREASING; IC = -1.0.
        # That's correct: the "perfect inverted SELL" is when high
        # ml_score precedes a big price drop (good SELL), which under
        # the double-flip maps to "low aligned-signal, high
        # aligned-return" — still anti-monotone in this construction.
        # The point of the SELL flip is the SAME meaning across BUY/SELL,
        # NOT that all SELLs become positive.
        assert bull_cell["score_ic"] == pytest.approx(-1.0)


# ─────────────────────────── load + CLI ──────────────────────────────


class TestLoadOutcomes:
    def test_load_missing_file(self, tmp_path):
        assert prs._load_outcomes(tmp_path / "nope.jsonl") == []

    def test_load_skips_bad_lines(self, tmp_path):
        path = tmp_path / "outcomes.jsonl"
        good = json.dumps({"run_id": 1, "ml_score": 1.0,
                           "forward_return_5d": 1.0, "regime_label": "bull"})
        path.write_text(good + "\n"
                        + "this is not json\n"
                        + json.dumps("a string, not a dict") + "\n"
                        + good + "\n")
        recs = prs._load_outcomes(path)
        assert len(recs) == 2  # Only the two valid dict rows survive.

    def test_load_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        assert prs._load_outcomes(path) == []


class TestCli:
    def _write_outcomes(self, path: Path, recs: list[dict]) -> None:
        path.write_text("\n".join(json.dumps(r) for r in recs) + "\n")

    def test_cli_exit_0_on_healthy(self, tmp_path, capsys):
        path = tmp_path / "outcomes.jsonl"
        recs = [_row(1, "bull", float(i), float(i)) for i in range(25)]
        recs += [_row(2, "sideways", 1.0, 0.5) for _ in range(10)]
        self._write_outcomes(path, recs)
        rc = prs._cli(["--outcomes", str(path)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "HEALTHY" in out
        assert "BEST stable cell" in out

    def test_cli_exit_1_on_regime_conditional(self, tmp_path, capsys):
        path = tmp_path / "outcomes.jsonl"
        # Bull = SIGNAL_EDGE; sideways = NO_EDGE.
        recs = [_row(1, "bull", float(i), float(i)) for i in range(25)]
        recs += [_row(1, "sideways", float(i), 0.0) for i in range(25)]
        self._write_outcomes(path, recs)
        rc = prs._cli(["--outcomes", str(path)])
        assert rc == 1
        out = capsys.readouterr().out
        assert "REGIME_CONDITIONAL" in out

    def test_cli_exit_2_on_inverted(self, tmp_path, capsys):
        path = tmp_path / "outcomes.jsonl"
        recs = [_row(1, "bull", float(i), -float(i)) for i in range(25)]
        recs += [_row(2, "sideways", 1.0, 0.5) for _ in range(10)]
        self._write_outcomes(path, recs)
        rc = prs._cli(["--outcomes", str(path)])
        assert rc == 2
        out = capsys.readouterr().out
        assert "HAS_INVERTED_CELL" in out

    def test_cli_exit_0_on_insufficient_data(self, tmp_path, capsys):
        path = tmp_path / "outcomes.jsonl"
        # Only 5 rows — below MIN_RECORDS.
        recs = [_row(1, "bull", float(i), float(i)) for i in range(5)]
        self._write_outcomes(path, recs)
        rc = prs._cli(["--outcomes", str(path)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "INSUFFICIENT_DATA" in out

    def test_cli_json_mode(self, tmp_path, capsys):
        path = tmp_path / "outcomes.jsonl"
        recs = [_row(1, "bull", float(i), float(i)) for i in range(25)]
        recs += [_row(2, "sideways", 1.0, 0.5) for _ in range(10)]
        self._write_outcomes(path, recs)
        rc = prs._cli(["--outcomes", str(path), "--json"])
        assert rc == 0
        out = capsys.readouterr().out
        # Must be valid JSON top to bottom.
        rep = json.loads(out)
        assert rep["verdict"] == "HEALTHY"
        assert "cells" in rep
        assert any(c["regime"] == "bull" for c in rep["cells"])


# ─────────────────────────── module constants discipline ─────────────


class TestConstants:
    def test_min_per_cell_lt_min_records(self):
        # An overall floor must allow at least one stable cell.
        assert prs.MIN_PER_CELL <= prs.MIN_RECORDS

    def test_ic_good_gt_ic_min(self):
        # SIGNAL_EDGE bar must be stricter than WEAK_SIGNAL bar.
        assert prs.IC_GOOD > prs.IC_MIN

    def test_regime_from_mult_matches_documented_regimes(self):
        # The decode MUST mirror backtest.py / regime_audit. Pinning the
        # exact float keys catches a future regime-multiplier renumber.
        assert prs._REGIME_FROM_MULT == {0.30: "bear", 0.60: "sideways",
                                         1.00: "bull"}

    def test_regime_order_is_three_real_regimes(self):
        # The display order is bull → sideways → bear (rank by mult desc).
        assert prs.REGIME_ORDER == ("bull", "sideways", "bear")
        # No "unknown" in the display order — it is dropped from the
        # cross-tab, not bucketed.
        assert "unknown" not in prs.REGIME_ORDER

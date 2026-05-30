"""Tests for paper_trader.ml.sizing_rule_regime_breakdown.

Pins exact verdicts, exact winner names, exact regime-decode behavior,
and exact CLI exit codes. NO "no crash" assertions — every test asserts
a specific expected value so a real logic bug surfaces.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_trader.ml import sizing_rule_regime_breakdown as mod
from paper_trader.ml.sizing_rule_counterfactual import MIN_ROWS


# ──────────────────────────── helpers ────────────────────────────


def _buy(ml_score: float, conviction: float, ret: float,
         regime_label: str = "bull", regime_mult: float | None = None,
         news_urgency: float | None = None) -> dict:
    """Construct a synthetic BUY outcome row."""
    row = {
        "action": "BUY",
        "ml_score": ml_score,
        "conviction_pct": conviction,
        "forward_return_5d": ret,
    }
    if regime_label is not None:
        row["regime_label"] = regime_label
    if regime_mult is not None:
        row["regime_mult"] = regime_mult
    if news_urgency is not None:
        row["news_urgency"] = news_urgency
    return row


def _gen_regime_rows(n: int, regime_label: str, ret: float,
                    conviction: float = 0.20,
                    ml_score: float = 3.0) -> list[dict]:
    """Generate N BUY rows tagged with one regime label, all carrying the
    SAME forward return so the analyzer math is exactly predictable."""
    return [_buy(ml_score=ml_score, conviction=conviction, ret=ret,
                 regime_label=regime_label)
            for _ in range(n)]


# ──────────────────────────── _row_regime decode ────────────────────


class TestRowRegimeDecode:
    """Pin the regime-decode behavior across explicit label, legacy
    regime_mult fallback, and the unknown fall-through."""

    def test_explicit_bull_label(self):
        assert mod._row_regime({"regime_label": "bull"}) == "bull"

    def test_explicit_sideways_label(self):
        assert mod._row_regime({"regime_label": "sideways"}) == "sideways"

    def test_explicit_bear_label(self):
        assert mod._row_regime({"regime_label": "bear"}) == "bear"

    def test_legacy_regime_mult_03_decodes_bear(self):
        # Legacy row: no regime_label, regime_mult=0.3 → bear.
        assert mod._row_regime({"regime_mult": 0.3}) == "bear"

    def test_legacy_regime_mult_06_decodes_sideways(self):
        assert mod._row_regime({"regime_mult": 0.6}) == "sideways"

    def test_legacy_regime_mult_10_decodes_bull(self):
        assert mod._row_regime({"regime_mult": 1.0}) == "bull"

    def test_explicit_unknown_label_dropped(self):
        # The documented "unknown" → drop path (the unknown-regime
        # fall-through the mult-only path silently mis-bucketed as bull).
        assert mod._row_regime({"regime_label": "unknown"}) is None

    def test_explicit_label_overrides_mult(self):
        # An explicit "unknown" label drops even when regime_mult would
        # decode to bull — the documented contamination fix.
        assert mod._row_regime({"regime_label": "unknown",
                                "regime_mult": 1.0}) is None

    def test_missing_both_fields_returns_none(self):
        assert mod._row_regime({}) is None

    def test_unparseable_regime_mult_returns_none(self):
        # A non-numeric regime_mult on a legacy row drops cleanly.
        assert mod._row_regime({"regime_mult": "garbage"}) is None

    def test_unknown_regime_mult_value_returns_none(self):
        # regime_mult=2.0 doesn't decode to any regime in the map.
        assert mod._row_regime({"regime_mult": 2.0}) is None


# ──────────────────────────── _winner_name ─────────────────────────


class TestWinnerName:
    """Pin the winner extraction per-regime-verdict."""

    def test_alt_beats_actual_returns_best_alt(self):
        assert mod._winner_name({
            "verdict": "ALT_BEATS_ACTUAL",
            "best_alt_rule": "UNIFORM_25",
        }) == "UNIFORM_25"

    def test_actual_best_returns_actual(self):
        assert mod._winner_name({"verdict": "ACTUAL_BEST"}) == "ACTUAL"

    def test_tie_returns_none(self):
        # TIE is intentionally inconclusive — the aggregate verdict
        # treats this regime as not contributing a winner.
        assert mod._winner_name({"verdict": "TIE"}) is None

    def test_insufficient_data_returns_none(self):
        assert mod._winner_name({"verdict": "INSUFFICIENT_DATA"}) is None


# ──────────────────────────── build_regime_breakdown ────────────────


class TestBuildRegimeBreakdownEmpty:
    """Empty / pathological corpora — pin the honest-empty envelope."""

    def test_empty_records_returns_insufficient(self):
        rep = mod.build_regime_breakdown([])
        assert rep["status"] in ("ok", "insufficient_data")
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_total"] == 0
        assert rep["n_regime_decoded"] == 0

    def test_non_dict_rows_dropped(self):
        rep = mod.build_regime_breakdown([1, "garbage", None, {}])
        # Only the empty dict survives the isinstance filter and the
        # empty dict has no regime; n_total counts the dict only.
        assert rep["n_total"] == 1
        assert rep["n_regime_decoded"] == 0
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_all_unknown_regime_returns_insufficient(self):
        rows = [_buy(3, 0.2, 5, regime_label="unknown") for _ in range(100)]
        rep = mod.build_regime_breakdown(rows)
        assert rep["n_regime_decoded"] == 0
        assert rep["verdict"] == "INSUFFICIENT_DATA"


class TestBullDominated:
    """The most common production state: only bull has enough rows."""

    def test_only_bull_populated_returns_bull_dominated(self):
        # MIN_ROWS bull, zero bear, zero sideways.
        rows = _gen_regime_rows(MIN_ROWS, "bull", ret=5.0)
        rep = mod.build_regime_breakdown(rows)
        assert rep["verdict"] == "BULL_DOMINATED"
        assert rep["n_regime_decoded"] == MIN_ROWS

    def test_only_bear_populated_returns_insufficient(self):
        # Only bear (not bull). Since the verdict ladder distinguishes
        # BULL_DOMINATED vs INSUFFICIENT_DATA based on WHICH single regime
        # is populated, a bear-only corpus correctly degrades to
        # INSUFFICIENT_DATA (it tells us nothing about generalization
        # because bear is rare in real corpora).
        rows = _gen_regime_rows(MIN_ROWS, "bear", ret=5.0)
        rep = mod.build_regime_breakdown(rows)
        assert rep["verdict"] == "INSUFFICIENT_DATA"


class TestAllSameWinner:
    """The "deploy with confidence" verdict — same alt wins every regime."""

    def test_uniform_25_wins_both_populated_regimes(self):
        # In both bull and sideways, ACTUAL is tiny (conviction=0.05 on
        # +5% return = +0.25pp/trade) while UNIFORM_25 lands +1.25pp/trade.
        # With MIN_ROWS rows in each, UNIFORM_25 will beat ACTUAL by 5×
        # in BOTH regimes — well past the 20% threshold.
        bull_rows = _gen_regime_rows(MIN_ROWS, "bull", ret=5.0, conviction=0.05)
        side_rows = _gen_regime_rows(MIN_ROWS, "sideways", ret=5.0, conviction=0.05)
        rep = mod.build_regime_breakdown(bull_rows + side_rows)
        assert rep["verdict"] == "ALL_SAME_WINNER"
        # Both regimes' winners must equal UNIFORM_25.
        winners = rep["winners"]
        assert winners == ["UNIFORM_25", "UNIFORM_25"]

    def test_actual_wins_when_deployed_rule_already_optimal(self):
        # If ACTUAL is sizing 0.25 (the cap), UNIFORM_25 ties it exactly
        # and UNIFORM_10 loses. ACTUAL is also tied with SCORE_BASED at
        # ml_score=5 (5/20=0.25 → caps to 0.25). NEWS_DRIVEN falls back
        # to 0.10 (no news_urgency). INVERSE_SCORE for ml_score=5
        # gives 0.25 - 0.05 = 0.20.
        #
        # So per-regime:
        #   ACTUAL=UNIFORM_25=SCORE_BASED=0.25×5=1.25pp/trade
        #   INVERSE_SCORE=0.20×5=1.00pp/trade
        #   UNIFORM_10=0.10×5=0.50pp/trade
        # Best alt vs ACTUAL: UNIFORM_25 at same total → TIE.
        # Verdict per regime: TIE → no winner → aggregate ALL_TIE.
        bull_rows = _gen_regime_rows(MIN_ROWS, "bull", ret=5.0,
                                     conviction=0.25, ml_score=5.0)
        side_rows = _gen_regime_rows(MIN_ROWS, "sideways", ret=5.0,
                                     conviction=0.25, ml_score=5.0)
        rep = mod.build_regime_breakdown(bull_rows + side_rows)
        assert rep["verdict"] == "ALL_TIE"


class TestDifferentWinners:
    """The quant-decisive verdict — regime-conditional sizing is needed."""

    def test_alt_wins_bull_actual_wins_bear(self):
        # Construct: in bull, UNIFORM_25 beats ACTUAL hard (small actual
        # conviction, positive realized returns).
        # In bear, ACTUAL conviction is tiny on losses while UNIFORM_25
        # would lose much more (sizing into negative returns hurts more).
        #
        # Bull: ACTUAL=0.05, ret=+5%, so ACTUAL=+0.25pp/trade,
        #       UNIFORM_25=+1.25pp/trade → ALT_BEATS_ACTUAL (winner=UNIFORM_25).
        # Bear: ACTUAL=0.05, ret=-5%, so ACTUAL=-0.25pp/trade,
        #       UNIFORM_25=-1.25pp/trade → UNIFORM_10 at -0.50 is best alt,
        #       but vs ACTUAL=-0.25 the best alt loses → ACTUAL_BEST.
        #
        # Best alt for bear is the LEAST-negative non-ACTUAL: UNIFORM_10
        # at -0.50 or INVERSE_SCORE at -0.05*0.25=... let's compute.
        # ml_score=3.0 → INVERSE_SCORE = max(0.05, 0.25 - 0.03) = 0.22
        #   → 0.22 × -5 = -1.10
        # NEWS_DRIVEN (no news) = 0.10 → -0.50
        # SCORE_BASED (ml_score=3) = min(0.25, 3/20)=0.15 → -0.75
        # So best alt in bear is UNIFORM_10 at -0.50. ACTUAL=-0.25 beats
        # UNIFORM_10=-0.50 by ((-0.25) - (-0.50)) / 0.25 = 1.0 → 100%
        # absolute, but rel_improvement = (best_alt - actual)/|actual|
        # = (-0.50 - (-0.25))/0.25 = -1.0 → rel_improvement <= -0.20
        # → ACTUAL_BEST.
        bull_rows = _gen_regime_rows(MIN_ROWS, "bull", ret=5.0,
                                     conviction=0.05, ml_score=3.0)
        bear_rows = _gen_regime_rows(MIN_ROWS, "bear", ret=-5.0,
                                     conviction=0.05, ml_score=3.0)
        rep = mod.build_regime_breakdown(bull_rows + bear_rows)
        assert rep["verdict"] == "DIFFERENT_WINNERS"
        # Order: REGIMES iter is ("bull", "sideways", "bear"), so
        # populated and winners is [bull_winner, bear_winner] (sideways
        # is INSUFFICIENT_DATA and dropped).
        winners = rep["winners"]
        assert "UNIFORM_25" in winners  # bull's winner
        assert "ACTUAL" in winners      # bear's winner


class TestRegimeReportShape:
    """Per-regime row contains every documented field."""

    def test_per_regime_row_has_expected_keys(self):
        rows = _gen_regime_rows(MIN_ROWS, "bull", ret=5.0)
        rep = mod.build_regime_breakdown(rows)
        bull = next(r for r in rep["regimes"] if r["regime"] == "bull")
        for key in ("regime", "verdict", "status", "n", "winner",
                    "rules", "actual_total_pp", "best_alt_rule",
                    "best_alt_total_pp", "rel_improvement", "hint"):
            assert key in bull, f"missing key {key}"

    def test_unpopulated_regime_n_is_zero(self):
        rows = _gen_regime_rows(MIN_ROWS, "bull", ret=5.0)
        rep = mod.build_regime_breakdown(rows)
        bear = next(r for r in rep["regimes"] if r["regime"] == "bear")
        assert bear["n"] == 0
        assert bear["verdict"] == "INSUFFICIENT_DATA"
        assert bear["winner"] is None

    def test_n_regime_missing_counts_undecodable(self):
        # 10 bull rows + 5 rows with "unknown" label that get dropped.
        rows = _gen_regime_rows(10, "bull", ret=5.0) + [
            _buy(3, 0.2, 5.0, regime_label="unknown") for _ in range(5)
        ]
        rep = mod.build_regime_breakdown(rows)
        assert rep["n_total"] == 15
        assert rep["n_regime_decoded"] == 10
        assert rep["n_regime_missing"] == 5


class TestSizingRulesSet:
    """Spot-check the inherited rule set — the regime-breakdown analyzer
    must NOT drift the rule set from the parent module."""

    def test_inherited_six_rules(self):
        names = [name for name, _ in mod.SIZING_RULES]
        assert names == ["ACTUAL", "UNIFORM_10", "UNIFORM_25",
                         "SCORE_BASED", "INVERSE_SCORE", "NEWS_DRIVEN"]


# ──────────────────────────── analyze + I/O ────────────────────────


class TestAnalyzeMissingFile:
    """Missing path degrades to honest INSUFFICIENT_DATA, never raises."""

    def test_missing_path_returns_envelope(self, tmp_path: Path):
        rep = mod.analyze(tmp_path / "does_not_exist.jsonl")
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_total"] == 0


class TestAnalyzeRoundTrip:
    """Write outcomes to a real tmp JSONL, call analyze, assert verdict."""

    def test_round_trip_all_same_winner(self, tmp_path: Path):
        rows = (_gen_regime_rows(MIN_ROWS, "bull", ret=5.0, conviction=0.05)
                + _gen_regime_rows(MIN_ROWS, "sideways", ret=5.0,
                                   conviction=0.05))
        path = tmp_path / "outcomes.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows))
        rep = mod.analyze(path)
        assert rep["verdict"] == "ALL_SAME_WINNER"
        assert "UNIFORM_25" in rep["winners"]


# ──────────────────────────── CLI ────────────────────────────


class TestCLIExitCodes:
    """Exit 2 ONLY on DIFFERENT_WINNERS. Every other state is exit 0."""

    def test_exit_0_on_all_same_winner(self, tmp_path: Path, capsys):
        rows = (_gen_regime_rows(MIN_ROWS, "bull", ret=5.0, conviction=0.05)
                + _gen_regime_rows(MIN_ROWS, "sideways", ret=5.0,
                                   conviction=0.05))
        path = tmp_path / "outcomes.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows))
        rc = mod._cli(["--outcomes", str(path)])
        assert rc == 0

    def test_exit_0_on_insufficient_data(self, tmp_path: Path, capsys):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        rc = mod._cli(["--outcomes", str(path)])
        assert rc == 0

    def test_exit_0_on_bull_dominated(self, tmp_path: Path, capsys):
        rows = _gen_regime_rows(MIN_ROWS, "bull", ret=5.0)
        path = tmp_path / "outcomes.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows))
        rc = mod._cli(["--outcomes", str(path)])
        assert rc == 0

    def test_exit_2_on_different_winners(self, tmp_path: Path, capsys):
        bull_rows = _gen_regime_rows(MIN_ROWS, "bull", ret=5.0,
                                     conviction=0.05, ml_score=3.0)
        bear_rows = _gen_regime_rows(MIN_ROWS, "bear", ret=-5.0,
                                     conviction=0.05, ml_score=3.0)
        path = tmp_path / "outcomes.jsonl"
        path.write_text("\n".join(json.dumps(r)
                                  for r in bull_rows + bear_rows))
        rc = mod._cli(["--outcomes", str(path)])
        assert rc == 2

    def test_json_output_is_parseable(self, tmp_path: Path, capsys):
        rows = _gen_regime_rows(MIN_ROWS, "bull", ret=5.0)
        path = tmp_path / "outcomes.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows))
        rc = mod._cli(["--outcomes", str(path), "--json"])
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert "verdict" in parsed
        assert "regimes" in parsed
        assert rc == 0


# ──────────────────────────── Legacy rows ───────────────────────────


class TestLegacyRegimeMult:
    """Pre-2026-05-19 corpora carry regime_mult but no regime_label —
    the analyzer must decode them so historical analysis still works."""

    def test_legacy_mult_only_corpus_decodes(self):
        # Build MIN_ROWS bull rows with ONLY regime_mult=1.0, no label.
        # _row_regime falls back to the mult decode → "bull".
        rows = [
            {
                "action": "BUY",
                "ml_score": 3.0,
                "conviction_pct": 0.05,
                "forward_return_5d": 5.0,
                "regime_mult": 1.0,
            }
            for _ in range(MIN_ROWS)
        ]
        rep = mod.build_regime_breakdown(rows)
        assert rep["n_regime_decoded"] == MIN_ROWS
        # Single regime → BULL_DOMINATED.
        assert rep["verdict"] == "BULL_DOMINATED"

    def test_legacy_unknown_mult_dropped(self):
        # regime_mult=1.0 BUT regime_label="unknown" → still dropped
        # because the explicit label is honored first and "unknown"
        # falls through.
        rows = [
            {
                "action": "BUY", "ml_score": 3.0,
                "conviction_pct": 0.05, "forward_return_5d": 5.0,
                "regime_mult": 1.0, "regime_label": "unknown",
            }
            for _ in range(MIN_ROWS)
        ]
        rep = mod.build_regime_breakdown(rows)
        assert rep["n_regime_decoded"] == 0

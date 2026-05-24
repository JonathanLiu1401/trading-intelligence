"""Tests for paper_trader/analytics/setup_analogues.py.

Covers:
  - The 4×4×3 bucket grid edges (oversold/overbought, deep_neg/strong, bear/bull)
  - Action semantics: BUY win on up, SELL win on down (and option flips)
  - Verdict ladder precedence: STRONG_EDGE / EDGE / NEUTRAL / HEADWIND /
    STRONG_HEADWIND / INSUFFICIENT_DATA
  - Percentile shape (no numpy dep — pure-Python linear interp)
  - Missing-feature fallback (None inputs → no bucket match, honest verdict)
  - File reader robustness (missing / malformed / tail-limit)
  - The "no DB / no model / no network" anti-coupling claim
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from paper_trader.analytics.setup_analogues import (
    DEFAULT_MIN_MATCHES,
    _bucket,
    _bucketize,
    _is_win,
    _percentiles,
    _RSI_EDGES,
    _MOM20_EDGES,
    _REGIME_EDGES,
    build_setup_analogues,
    _load_outcomes,
)


# ─────────────────────────────────────────────────────────── unit primitives


class TestBucketEdges:
    """Boundary behaviour of the four-band RSI / four-band mom20 / three-band
    regime grid.  The ladder is half-open [lo, hi) with the right-most edge
    catching equality on the high side — verify that on every transition."""

    def test_rsi_oversold_under_30(self):
        assert _bucket(15.0, _RSI_EDGES) == "oversold"
        assert _bucket(29.99, _RSI_EDGES) == "oversold"

    def test_rsi_mid_low_30_to_50(self):
        assert _bucket(30.0, _RSI_EDGES) == "mid_low"
        assert _bucket(49.99, _RSI_EDGES) == "mid_low"

    def test_rsi_mid_high_50_to_70(self):
        assert _bucket(50.0, _RSI_EDGES) == "mid_high"
        assert _bucket(69.99, _RSI_EDGES) == "mid_high"

    def test_rsi_overbought_70_plus(self):
        assert _bucket(70.0, _RSI_EDGES) == "overbought"
        assert _bucket(100.0, _RSI_EDGES) == "overbought"

    def test_mom20_deep_neg(self):
        assert _bucket(-25.0, _MOM20_EDGES) == "deep_neg"
        assert _bucket(-10.01, _MOM20_EDGES) == "deep_neg"

    def test_mom20_strong_over_10(self):
        assert _bucket(10.0, _MOM20_EDGES) == "strong"
        assert _bucket(50.0, _MOM20_EDGES) == "strong"

    def test_regime_bear(self):
        assert _bucket(0.6, _REGIME_EDGES) == "bear"
        assert _bucket(0.84, _REGIME_EDGES) == "bear"

    def test_regime_sideways(self):
        assert _bucket(0.85, _REGIME_EDGES) == "sideways"
        assert _bucket(1.04, _REGIME_EDGES) == "sideways"

    def test_regime_bull(self):
        assert _bucket(1.05, _REGIME_EDGES) == "bull"
        assert _bucket(1.3, _REGIME_EDGES) == "bull"

    def test_none_value_returns_none(self):
        assert _bucket(None, _RSI_EDGES) is None
        assert _bucket("abc", _RSI_EDGES) is None
        # NaN check — float('nan') != itself
        assert _bucket(float("nan"), _RSI_EDGES) is None


class TestBucketize:
    """The three-way classifier composes the per-axis edges; all-None inputs
    must yield all-None outputs (no silent zero fallback)."""

    def test_full_bucketize_typical_buy(self):
        b = _bucketize(rsi=65.0, mom20=12.0, regime_mult=1.0)
        assert b == {"rsi": "mid_high", "mom20": "strong", "regime": "sideways"}

    def test_full_bucketize_oversold_bear(self):
        b = _bucketize(rsi=22.0, mom20=-18.0, regime_mult=0.7)
        assert b == {"rsi": "oversold", "mom20": "deep_neg", "regime": "bear"}

    def test_partial_none_inputs(self):
        b = _bucketize(rsi=None, mom20=12.0, regime_mult=1.0)
        assert b == {"rsi": None, "mom20": "strong", "regime": "sideways"}


class TestActionWinSemantics:
    """BUY wins on up moves; SELL wins on down moves.  Options track the
    underlying direction.  A zero-move is neither side's win."""

    def test_buy_wins_when_market_up(self):
        assert _is_win("BUY", 3.0) is True
        assert _is_win("BUY", -2.0) is False

    def test_sell_wins_when_market_down(self):
        assert _is_win("SELL", -3.0) is True
        assert _is_win("SELL", 2.0) is False

    def test_buy_call_tracks_buy(self):
        assert _is_win("BUY_CALL", 1.5) is True
        assert _is_win("BUY_CALL", -1.5) is False

    def test_sell_put_tracks_sell(self):
        assert _is_win("SELL_PUT", -1.5) is True

    def test_unknown_action_treated_as_buy(self):
        # An exotic action label that isn't in either sell set falls through
        # to the default BUY-style win predicate.  This is the right
        # conservative default for an unknown action.
        assert _is_win("HOLD", 1.0) is True


# ─────────────────────────────────────────────────────────── percentiles

class TestPercentiles:
    def test_empty_returns_zeros(self):
        s = _percentiles([])
        assert s["p25"] == 0.0 and s["p50"] == 0.0 and s["p75"] == 0.0
        assert s["mean"] == 0.0 and s["best"] == 0.0 and s["worst"] == 0.0

    def test_single_value(self):
        s = _percentiles([3.5])
        assert s["p25"] == s["p50"] == s["p75"] == 3.5
        assert s["best"] == s["worst"] == 3.5

    def test_known_quartiles(self):
        # 0..100 — quartiles are 25, 50, 75 by linear interp.
        s = _percentiles([float(i) for i in range(101)])
        assert s["p25"] == 25.0
        assert s["p50"] == 50.0
        assert s["p75"] == 75.0
        assert s["best"] == 100.0
        assert s["worst"] == 0.0
        assert s["mean"] == 50.0

    def test_handles_negatives(self):
        s = _percentiles([-10.0, -5.0, 0.0, 5.0, 10.0])
        assert s["worst"] == -10.0
        assert s["best"] == 10.0
        assert s["p50"] == 0.0


# ─────────────────────────────────────────────────────────── builder

def _outcome(action="BUY", rsi=60.0, mom20=12.0, regime_mult=1.0,
             fr=1.0, ticker="NVDA", **extra):
    # Defaults are chosen to land in bucket (mid_high, strong, sideways) so a
    # test that queries with rsi=60 / mom20=12 / regime_mult=1.0 (the natural
    # "typical bullish setup" case) finds them without ceremony.
    """Helper — a single ``decision_outcomes.jsonl`` row with sensible defaults."""
    row = {
        "action": action,
        "ticker": ticker,
        "rsi": rsi,
        "mom20": mom20,
        "regime_mult": regime_mult,
        "forward_return_5d": fr,
    }
    row.update(extra)
    return row


class TestBuilderShape:
    """Required-keys contract — the dashboard wrapper depends on every key
    being present even on degenerate inputs."""

    def test_required_keys_on_empty_corpus(self):
        out = build_setup_analogues([], ticker="X", action="BUY",
                                    rsi=50.0, mom20=0.0, regime_mult=1.0)
        for k in ("as_of", "ticker", "action", "current_features",
                  "current_buckets", "min_matches", "n_outcomes",
                  "n_action_only_matches", "n_matches", "stats",
                  "trader_median_pct", "win_rate", "verdict", "headline"):
            assert k in out, f"missing key: {k}"

    def test_empty_corpus_collapses_to_insufficient(self):
        out = build_setup_analogues([], ticker="X", action="BUY",
                                    rsi=50.0, mom20=0.0, regime_mult=1.0)
        assert out["verdict"] == "INSUFFICIENT_DATA"
        assert out["n_outcomes"] == 0
        assert out["n_matches"] == 0
        assert out["win_rate"] == 0.0

    def test_now_injection_threads_through(self):
        fixed = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        out = build_setup_analogues([], ticker="X", action="BUY",
                                    rsi=50.0, mom20=0.0, regime_mult=1.0,
                                    now=fixed)
        assert out["as_of"].startswith("2026-01-01T12:00:00")


class TestBuilderMatching:
    """The bucket-cell match logic — only rows in the *same* (rsi, mom20,
    regime) cell AND with the same action count as matches."""

    def test_bucket_only_matches_same_cell(self):
        # 30 BUYs in mid_high / strong / sideways, all +5%.
        # 30 BUYs in oversold / deep_neg / bear, all -8%.  Two distinct cells.
        outcomes = (
            [_outcome(action="BUY", rsi=60, mom20=12, regime_mult=1.0, fr=5.0)
             for _ in range(30)]
            + [_outcome(action="BUY", rsi=25, mom20=-15, regime_mult=0.7, fr=-8.0)
               for _ in range(30)]
        )
        # Query the first cell — should see only the +5 rows.
        out = build_setup_analogues(outcomes, ticker="X", action="BUY",
                                    rsi=60.0, mom20=12.0, regime_mult=1.0)
        assert out["n_matches"] == 30
        assert out["stats"]["p50"] == 5.0
        assert out["win_rate"] == 1.0
        # The other cell isn't pooled in.
        assert out["stats"]["worst"] == 5.0

    def test_action_filter_excludes_other_actions(self):
        outcomes = (
            [_outcome(action="BUY", rsi=60, mom20=12, regime_mult=1.0, fr=5.0)
             for _ in range(25)]
            + [_outcome(action="SELL", rsi=60, mom20=12, regime_mult=1.0, fr=5.0)
               for _ in range(25)]
        )
        out = build_setup_analogues(outcomes, ticker="X", action="BUY",
                                    rsi=60.0, mom20=12.0, regime_mult=1.0)
        # Only the 25 BUYs should match.
        assert out["n_matches"] == 25
        assert out["n_action_only_matches"] == 25

    def test_missing_feature_drops_to_action_only_zero_matches(self):
        outcomes = [_outcome(fr=2.0) for _ in range(50)]
        out = build_setup_analogues(outcomes, ticker="X", action="BUY",
                                    rsi=None, mom20=12.0, regime_mult=1.0)
        # rsi missing ⇒ no bucket cell ⇒ no full match, only action-only count.
        assert out["n_matches"] == 0
        assert out["n_action_only_matches"] == 50
        assert "missing feature" in out["headline"].lower()

    def test_malformed_rows_silently_skipped(self):
        outcomes = [
            "not-a-dict",
            {"action": "BUY", "forward_return_5d": "not-a-number",
             "rsi": 60, "mom20": 12, "regime_mult": 1.0},
            _outcome(fr=3.0),
        ]
        out = build_setup_analogues(outcomes, ticker="X", action="BUY",
                                    rsi=60.0, mom20=12.0, regime_mult=1.0)
        # Only the one well-formed BUY counts.
        assert out["n_matches"] == 1
        assert out["n_action_only_matches"] == 1


class TestVerdictLadder:
    """Closed verdict alphabet — STRONG_EDGE, EDGE, NEUTRAL, HEADWIND,
    STRONG_HEADWIND, INSUFFICIENT_DATA.  Tested with synthetic distributions
    that hit each rung."""

    def test_strong_edge_buy(self):
        # Median +5%, 70% wins, n=30 ⇒ STRONG_EDGE.
        outcomes = (
            [_outcome(fr=5.0) for _ in range(21)]
            + [_outcome(fr=-2.0) for _ in range(9)]
        )
        out = build_setup_analogues(outcomes, ticker="X", action="BUY",
                                    rsi=60.0, mom20=12.0, regime_mult=1.0)
        assert out["verdict"] == "STRONG_EDGE"
        assert out["win_rate"] >= 0.6
        assert out["stats"]["p50"] > 3.0

    def test_edge_buy(self):
        # 14 × +2%, 4 × +1.5%, 12 × -2%.  Sorted median (n=30) lands on the
        # interp of indices 14/15 = (1.5 + 2.0)/2 = +1.75% — above EDGE floor.
        # 18 wins / 30 ⇒ win_rate 0.60 — clears EDGE (>= 0.55) but not
        # STRONG_EDGE (> 0.60 strict).
        outcomes = (
            [_outcome(fr=2.0) for _ in range(14)]
            + [_outcome(fr=1.5) for _ in range(4)]
            + [_outcome(fr=-2.0) for _ in range(12)]
        )
        out = build_setup_analogues(outcomes, ticker="X", action="BUY",
                                    rsi=60.0, mom20=12.0, regime_mult=1.0)
        assert out["verdict"] == "EDGE"

    def test_neutral_buy(self):
        # Median ~0, win rate ~50% ⇒ NEUTRAL.
        outcomes = (
            [_outcome(fr=0.5) for _ in range(15)]
            + [_outcome(fr=-0.5) for _ in range(15)]
        )
        out = build_setup_analogues(outcomes, ticker="X", action="BUY",
                                    rsi=60.0, mom20=12.0, regime_mult=1.0)
        assert out["verdict"] == "NEUTRAL"

    def test_strong_headwind_buy(self):
        # Median -5%, win rate 30% ⇒ STRONG_HEADWIND.
        outcomes = (
            [_outcome(fr=-5.0) for _ in range(21)]
            + [_outcome(fr=2.0) for _ in range(9)]
        )
        out = build_setup_analogues(outcomes, ticker="X", action="BUY",
                                    rsi=60.0, mom20=12.0, regime_mult=1.0)
        assert out["verdict"] == "STRONG_HEADWIND"

    def test_headwind_buy(self):
        # Median -1.5%, win rate 40% ⇒ HEADWIND.
        outcomes = (
            [_outcome(fr=-2.0) for _ in range(18)]
            + [_outcome(fr=1.5) for _ in range(12)]
        )
        out = build_setup_analogues(outcomes, ticker="X", action="BUY",
                                    rsi=60.0, mom20=12.0, regime_mult=1.0)
        assert out["verdict"] == "HEADWIND"

    def test_thin_sample_collapses_to_insufficient(self):
        # 5 BUYs all +10% ⇒ would be STRONG_EDGE on stats, but n < min_matches.
        outcomes = [_outcome(fr=10.0) for _ in range(5)]
        out = build_setup_analogues(
            outcomes, ticker="X", action="BUY",
            rsi=60.0, mom20=12.0, regime_mult=1.0, min_matches=20)
        assert out["verdict"] == "INSUFFICIENT_DATA"
        # Stats are still emitted so the caller can show "provisional".
        assert out["stats"]["p50"] == 10.0

    def test_min_matches_threshold_honors_caller(self):
        # 10 BUYs all +5%.  With min_matches=10 we get STRONG_EDGE, with 11
        # we collapse to INSUFFICIENT_DATA.  Proves the threshold is honored.
        outcomes = [_outcome(fr=5.0) for _ in range(10)]
        out_low = build_setup_analogues(
            outcomes, ticker="X", action="BUY",
            rsi=60.0, mom20=12.0, regime_mult=1.0, min_matches=10)
        assert out_low["verdict"] == "STRONG_EDGE"
        out_high = build_setup_analogues(
            outcomes, ticker="X", action="BUY",
            rsi=60.0, mom20=12.0, regime_mult=1.0, min_matches=11)
        assert out_high["verdict"] == "INSUFFICIENT_DATA"


class TestSellSemantics:
    """SELL median sign-flips into trader space — a SELL ahead of a -5% move
    is a WIN.  trader_median_pct should be positive for a successful SELL."""

    def test_sell_ahead_of_drop_is_edge(self):
        # 25 SELLs, median forward_return_5d = -4% (market dropped, sell was right).
        # Raw p50 = -4%, trader_median = +4% ⇒ EDGE (positive trader median + win rate).
        outcomes = [_outcome(action="SELL", fr=-4.0) for _ in range(25)]
        out = build_setup_analogues(outcomes, ticker="X", action="SELL",
                                    rsi=60.0, mom20=12.0, regime_mult=1.0)
        # Raw stats are in market space.
        assert out["stats"]["p50"] == -4.0
        # Trader sees the sign-flipped median.
        assert out["trader_median_pct"] == 4.0
        # Verdict reads STRONG_EDGE (trader median +4 > 3, win rate 100%).
        assert out["verdict"] == "STRONG_EDGE"
        # Win rate counts down-moves as wins for SELL.
        assert out["win_rate"] == 1.0

    def test_sell_ahead_of_rally_is_headwind(self):
        # 25 SELLs into a rally — every one was wrong.
        outcomes = [_outcome(action="SELL", fr=5.0) for _ in range(25)]
        out = build_setup_analogues(outcomes, ticker="X", action="SELL",
                                    rsi=60.0, mom20=12.0, regime_mult=1.0)
        assert out["trader_median_pct"] == -5.0
        assert out["win_rate"] == 0.0
        assert out["verdict"] == "STRONG_HEADWIND"


# ─────────────────────────────────────────────────────────── file reader

class TestLoadOutcomes:
    def test_missing_file_returns_empty(self, tmp_path):
        # tmp_path is a pytest fixture — non-existent file path under tmp.
        p = tmp_path / "no-such-file.jsonl"
        assert _load_outcomes(p) == []

    def test_malformed_lines_skipped(self, tmp_path):
        p = tmp_path / "outcomes.jsonl"
        p.write_text("not-json\n"
                     '{"a": 1}\n'
                     "\n"
                     '{"b": 2}\n')
        rows = _load_outcomes(p)
        assert rows == [{"a": 1}, {"b": 2}]

    def test_tail_limit_honored(self, tmp_path):
        p = tmp_path / "outcomes.jsonl"
        lines = [json.dumps({"i": i}) for i in range(100)]
        p.write_text("\n".join(lines))
        rows = _load_outcomes(p, max_rows=10)
        assert len(rows) == 10
        # Tail — last 10 rows (indices 90..99).
        assert rows[0]["i"] == 90
        assert rows[-1]["i"] == 99


# ─────────────────────────────────────────────────────────── anti-coupling

class TestAntiCoupling:
    """The skill's no-DB / no-model / no-network claim.  If any of these
    imports leaked into the module, this test would fail."""

    def test_module_does_not_import_db(self):
        # Read the source and assert it doesn't reach into the engine.  Strip
        # the module docstring first — it legitimately discusses the sibling
        # modules ('DecisionScorer', 'scorer-confidence') as design context.
        src = (Path(__file__).resolve().parent.parent
               / "paper_trader" / "analytics" / "setup_analogues.py").read_text()
        # Strip the first triple-quoted block — that's the module docstring.
        if src.startswith('"""'):
            end = src.index('"""', 3) + 3
            src = src[end:]
        forbidden = ["import sqlite3", "from .. import store",
                     "from ..store", "from ..ml import",
                     "from ..backtest import",
                     "requests.get", "requests.post", "urllib.request"]
        for token in forbidden:
            assert token not in src, f"setup_analogues should not contain {token!r}"


# ─────────────────────────────────────────────────────────── live-corpus smoke

class TestLiveCorpusSmoke:
    """Pulls a tail of the real data/decision_outcomes.jsonl if present —
    proves the builder runs against production-shaped rows without surprise."""

    def test_runs_against_real_corpus_if_available(self):
        corpus = (Path(__file__).resolve().parent.parent
                  / "data" / "decision_outcomes.jsonl")
        if not corpus.exists():
            pytest.skip("decision_outcomes.jsonl not present — skip live smoke")
        rows = _load_outcomes(corpus, max_rows=1000)
        if len(rows) < 100:
            pytest.skip("corpus too thin for smoke test")
        # A common live setup — mid_high RSI, positive mom20, sideways regime.
        out = build_setup_analogues(rows, ticker="NVDA", action="BUY",
                                    rsi=60.0, mom20=5.0, regime_mult=1.0)
        # Every required key, no crashes, verdict in the closed alphabet.
        assert out["verdict"] in {"STRONG_EDGE", "EDGE", "NEUTRAL",
                                  "HEADWIND", "STRONG_HEADWIND",
                                  "INSUFFICIENT_DATA"}
        assert out["n_outcomes"] == len(rows)
        assert out["n_action_only_matches"] >= out["n_matches"]

"""Tests for paper_trader.ml.buy_ticker_quant_coverage.

The diagnostic answers the gate-relevant quant question: how many BUY rows
in the corpus had `ticker` outside the cycle's pre-fetched
QUANT_SIGNAL_TICKERS set (a sentiment-only buy_ticker), and how many of
those gap rows nevertheless carry real training-time RSI/MACD. That gap
is the training/inference feature-parity drift documented in commit
9268ee0 (the fix that made `_ml_decide` lazily fetch `_get_quant_signals`
for sentiment-only buys).

Tests assert exact counts / verdicts on synthetic JSONL fixtures so a
threshold regression or counting drift would fail loudly.
"""
from __future__ import annotations

import json

import pytest

from paper_trader.ml import buy_ticker_quant_coverage as btqc


def _write_outcomes(tmp_path, rows):
    p = tmp_path / "decision_outcomes.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


class TestVerdictThresholds:
    def test_healthy_below_5pct_gap(self, tmp_path):
        # 100 BUYs: 95 NVDA (covered), 5 BTC-USD (gap, real RSI). 5% gap.
        rows = [{"action": "BUY", "ticker": "NVDA", "rsi": 50} for _ in range(95)]
        rows += [{"action": "BUY", "ticker": "BTC-USD", "rsi": 55}
                 for _ in range(5)]
        # gap_fraction = 0.05 — right at the floor, should NOT be HEALTHY.
        rep = btqc.analyze(_write_outcomes(tmp_path, rows))
        assert rep["n_total_buys"] == 100
        assert rep["n_quant_gap"] == 5
        # Falls into the GAP_PRESENT band (>= 0.05).
        assert rep["verdict"] == "GAP_PRESENT"

    def test_healthy_below_threshold(self, tmp_path):
        # 100 BUYs: 99 NVDA (covered), 1 BTC-USD (gap). 1% gap < 5%.
        rows = [{"action": "BUY", "ticker": "NVDA", "rsi": 50} for _ in range(99)]
        rows += [{"action": "BUY", "ticker": "BTC-USD", "rsi": 55}]
        rep = btqc.analyze(_write_outcomes(tmp_path, rows))
        assert rep["verdict"] == "HEALTHY"
        assert rep["gap_fraction"] == pytest.approx(0.01)

    def test_gap_dominant_above_30pct(self, tmp_path):
        # 100 BUYs: 60 covered, 40 gap (all w/ real RSI). 40% gap.
        rows = [{"action": "BUY", "ticker": "NVDA", "rsi": 50} for _ in range(60)]
        rows += [{"action": "BUY", "ticker": "BTC-USD", "rsi": 55}
                 for _ in range(40)]
        rep = btqc.analyze(_write_outcomes(tmp_path, rows))
        assert rep["verdict"] == "GAP_DOMINANT"
        assert rep["gap_fraction"] == pytest.approx(0.40)
        assert rep["gap_rsi_real_fraction"] == pytest.approx(1.0)

    def test_gap_dominant_15pct_with_real_rsi(self, tmp_path):
        # 100 BUYs: 80 covered, 20 gap (all w/ real RSI). 20% gap, 100% real.
        # This is the "drift severity" branch — below 30% but high real-RSI
        # fraction means the training/inference mismatch is widespread.
        rows = [{"action": "BUY", "ticker": "NVDA", "rsi": 50} for _ in range(80)]
        rows += [{"action": "BUY", "ticker": "BTC-USD", "rsi": 55}
                 for _ in range(20)]
        rep = btqc.analyze(_write_outcomes(tmp_path, rows))
        assert rep["verdict"] == "GAP_DOMINANT"

    def test_gap_present_moderate(self, tmp_path):
        # 100 BUYs: 92 covered, 8 gap. 8% gap, below the dominant-15 threshold.
        rows = [{"action": "BUY", "ticker": "NVDA", "rsi": 50} for _ in range(92)]
        rows += [{"action": "BUY", "ticker": "BTC-USD", "rsi": 55}
                 for _ in range(8)]
        rep = btqc.analyze(_write_outcomes(tmp_path, rows))
        assert rep["verdict"] == "GAP_PRESENT"
        assert rep["gap_fraction"] == pytest.approx(0.08)


class TestRowFiltering:
    def test_non_buy_rows_ignored(self, tmp_path):
        # SELLs and HOLDs must not be counted as BUYs.
        rows = [{"action": "BUY", "ticker": "NVDA", "rsi": 50} for _ in range(30)]
        rows += [{"action": "SELL", "ticker": "BTC-USD", "rsi": 55}
                 for _ in range(50)]
        rows += [{"action": "HOLD", "ticker": "ZZZ"} for _ in range(20)]
        rep = btqc.analyze(_write_outcomes(tmp_path, rows))
        assert rep["n_total_buys"] == 30
        assert rep["n_quant_gap"] == 0

    def test_missing_ticker_skipped(self, tmp_path):
        # Rows with empty ticker must be skipped entirely.
        rows = [{"action": "BUY", "ticker": "NVDA", "rsi": 50} for _ in range(30)]
        rows += [{"action": "BUY", "ticker": "", "rsi": 50} for _ in range(5)]
        rep = btqc.analyze(_write_outcomes(tmp_path, rows))
        assert rep["n_total_buys"] == 30

    def test_lowercase_ticker_normalized(self, tmp_path):
        # Tickers are upper-cased for the QUANT_SIGNAL_TICKERS membership test.
        rows = [{"action": "buy", "ticker": "nvda", "rsi": 50}
                for _ in range(30)]
        rep = btqc.analyze(_write_outcomes(tmp_path, rows))
        assert rep["n_total_buys"] == 30
        assert rep["n_quant_covered"] == 30
        assert rep["n_quant_gap"] == 0


class TestTopTickers:
    def test_top_gap_tickers_sorted_by_count(self, tmp_path):
        rows: list = []
        # 10 BTC-USD, 5 XLF, 2 JPM, 30 covered.
        rows += [{"action": "BUY", "ticker": "NVDA", "rsi": 50}
                 for _ in range(30)]
        rows += [{"action": "BUY", "ticker": "BTC-USD", "rsi": 55}
                 for _ in range(10)]
        rows += [{"action": "BUY", "ticker": "XLF", "rsi": 60}
                 for _ in range(5)]
        rows += [{"action": "BUY", "ticker": "JPM", "rsi": 65}
                 for _ in range(2)]
        rep = btqc.analyze(_write_outcomes(tmp_path, rows))
        top = rep["top_gap_tickers"]
        # Sorted by count descending — Counter.most_common semantics.
        assert top[0]["ticker"] == "BTC-USD"
        assert top[0]["count"] == 10
        assert top[0]["rsi_real_count"] == 10
        assert top[1]["ticker"] == "XLF"
        assert top[2]["ticker"] == "JPM"


class TestDriftSeverity:
    def test_gap_rsi_real_counted(self, tmp_path):
        # 5 gap rows with rsi=None (no real indicator), 5 with rsi=55.
        rows = [{"action": "BUY", "ticker": "NVDA", "rsi": 50}
                for _ in range(30)]
        rows += [{"action": "BUY", "ticker": "BTC-USD", "rsi": None}
                 for _ in range(5)]
        rows += [{"action": "BUY", "ticker": "BTC-USD", "rsi": 55}
                 for _ in range(5)]
        rep = btqc.analyze(_write_outcomes(tmp_path, rows))
        assert rep["n_quant_gap"] == 10
        assert rep["gap_rsi_real_count"] == 5
        assert rep["gap_rsi_real_fraction"] == pytest.approx(0.5)


class TestInsufficientData:
    def test_below_min_rows(self, tmp_path):
        # 20 BUYs < MIN_ROWS=30 → INSUFFICIENT_DATA, no gap_fraction.
        rows = [{"action": "BUY", "ticker": "NVDA", "rsi": 50}
                for _ in range(20)]
        rep = btqc.analyze(_write_outcomes(tmp_path, rows))
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["gap_fraction"] is None
        assert rep["gap_rsi_real_fraction"] is None

    def test_missing_file_degrades_gracefully(self, tmp_path):
        # Non-existent path must NOT raise — degrades to insufficient data.
        rep = btqc.analyze(tmp_path / "does-not-exist.jsonl")
        assert rep["status"] == "ok"
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_total_buys"] == 0

    def test_corrupt_jsonl_skipped(self, tmp_path):
        # A line that fails JSON parse must NOT abort the load.
        p = tmp_path / "outcomes.jsonl"
        good = json.dumps({"action": "BUY", "ticker": "NVDA", "rsi": 50})
        p.write_text(
            ("\n".join([good] * 30 + ["{not json}"] + [good] * 5)) + "\n"
        )
        rep = btqc.analyze(p)
        assert rep["n_total_buys"] == 35
        assert rep["status"] == "ok"


class TestTailParameter:
    def test_tail_limits_analysis(self, tmp_path):
        # 100 rows total: first 90 covered (NVDA), last 10 gap (BTC-USD).
        # tail=10 should see ONLY the last 10 → 100% gap (10/10).
        rows = [{"action": "BUY", "ticker": "NVDA", "rsi": 50}
                for _ in range(90)]
        rows += [{"action": "BUY", "ticker": "BTC-USD", "rsi": 55}
                 for _ in range(10)]
        rep = btqc.analyze(_write_outcomes(tmp_path, rows), tail=10)
        # tail=10 < MIN_ROWS=30 → INSUFFICIENT_DATA but n_total still right.
        assert rep["n_total_buys"] == 10
        assert rep["n_quant_gap"] == 10

    def test_full_tail_when_below_cap(self, tmp_path):
        rows = [{"action": "BUY", "ticker": "NVDA", "rsi": 50}
                for _ in range(50)]
        rep = btqc.analyze(_write_outcomes(tmp_path, rows), tail=5000)
        assert rep["n_total_buys"] == 50


class TestCLIIntegration:
    def test_main_returns_zero_on_ok(self, tmp_path, capsys):
        rows = [{"action": "BUY", "ticker": "NVDA", "rsi": 50}
                for _ in range(30)]
        p = _write_outcomes(tmp_path, rows)
        rc = btqc.main(["--outcomes", str(p), "--json"])
        assert rc == 0
        out = capsys.readouterr().out
        obj = json.loads(out)
        assert obj["status"] == "ok"
        assert obj["n_total_buys"] == 30

    def test_main_table_output(self, tmp_path, capsys):
        rows = [{"action": "BUY", "ticker": "NVDA", "rsi": 50}
                for _ in range(30)]
        rows += [{"action": "BUY", "ticker": "BTC-USD", "rsi": 55}
                 for _ in range(5)]
        rc = btqc.main(["--outcomes", str(_write_outcomes(tmp_path, rows))])
        assert rc == 0
        out = capsys.readouterr().out
        assert "buy_ticker_quant_coverage" in out
        assert "BTC-USD" in out


class TestLiveCorpusConsistency:
    """The diagnostic must agree with the same membership decisions the
    `_ml_decide` cycle makes. A regression to a different ticker set in
    `_quant_signal_tickers` would silently report misleading gap counts.
    """

    def test_quant_signal_tickers_match_backtest_module(self):
        from paper_trader.backtest import QUANT_SIGNAL_TICKERS
        assert btqc._quant_signal_tickers() == frozenset(QUANT_SIGNAL_TICKERS)

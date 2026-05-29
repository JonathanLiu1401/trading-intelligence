"""Lock tests for `paper_trader.ml.outcome_corpus_health`.

The analyzer is a pre-model data-quality diagnostic — it must produce
deterministic, structured verdicts regardless of input shape, and must
NEVER raise (a corpus-health writer is wired into per-cycle ledger
infrastructure and a crash here would break the loop).
"""
from __future__ import annotations

import json

import pytest

from paper_trader.ml import outcome_corpus_health as och


def _write_jsonl(tmp_path, rows):
    p = tmp_path / "outcomes.jsonl"
    with p.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return p


class TestVerdictEnvelope:
    def test_empty_corpus_insufficient(self, tmp_path):
        p = _write_jsonl(tmp_path, [])
        rep = och.analyze(p)
        assert rep["status"] == "empty"
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_total"] == 0

    def test_below_min_records_insufficient(self, tmp_path):
        # 50 rows < MIN_RECORDS=100
        rows = [{"action": "BUY", "ticker": "NVDA", "sim_date": "2024-01-01",
                 "forward_return_5d": 1.0, "regime_label": "bull"}
                for _ in range(50)]
        p = _write_jsonl(tmp_path, rows)
        rep = och.analyze(p)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_total"] == 50

    def test_missing_file_returns_empty(self, tmp_path):
        # Non-existent path degrades to INSUFFICIENT_DATA — analyzer
        # MUST never raise on this (per-cycle wrapper depends on it).
        rep = och.analyze(tmp_path / "no_such_file.jsonl")
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_total"] == 0


class TestNewsDarkVerdict:
    def test_news_features_dark_below_threshold(self, tmp_path):
        # 200 rows, all without news fields → fraction_with_news=0 → DARK
        rows = []
        for i in range(200):
            rows.append({
                "action": "BUY", "ticker": "NVDA",
                "sim_date": f"2024-01-{(i % 28) + 1:02d}",
                "forward_return_5d": (i % 10) - 5,
                "regime_label": "bull",
            })
        p = _write_jsonl(tmp_path, rows)
        rep = och.analyze(p)
        assert rep["verdict"] == "NEWS_FEATURES_DARK"
        assert rep["n_with_news"] == 0
        assert rep["fraction_with_news"] == 0.0
        # Hint message must reference the actual count
        assert any("news features in only" in h for h in rep["hints"])

    def test_news_features_present_majority_not_dark(self, tmp_path):
        # 200 rows, 50% with news fields → above 10% threshold → not DARK
        rows = []
        for i in range(200):
            r = {
                "action": "BUY" if i % 3 else "SELL",
                "ticker": "NVDA",
                "sim_date": f"2024-01-{(i % 28) + 1:02d}",
                "forward_return_5d": (i % 10) - 5,
                "regime_label": "bull",
            }
            if i < 100:  # half carry news
                r["news_urgency"] = 50.0
                r["news_article_count"] = 3.0
            rows.append(r)
        p = _write_jsonl(tmp_path, rows)
        rep = och.analyze(p)
        # With 50% news, news-dark doesn't fire — but we may still hit
        # REGIME_BUCKETS_SPARSE because all bull (no sideways/bear).
        assert rep["verdict"] != "NEWS_FEATURES_DARK"
        assert rep["n_with_news"] == 100
        assert rep["fraction_with_news"] == 0.5


class TestActionImbalanceVerdict:
    def test_action_imbalanced_buy_dominant(self, tmp_path):
        # 100% BUY → > 85% → ACTION_IMBALANCED (or earlier verdict)
        rows = []
        for i in range(200):
            rows.append({
                "action": "BUY", "ticker": "NVDA",
                "sim_date": f"2024-01-{(i % 28) + 1:02d}",
                "forward_return_5d": 1.0,
                "regime_label": "bull",
                "news_urgency": 50.0, "news_article_count": 3.0,
            })
        p = _write_jsonl(tmp_path, rows)
        rep = och.analyze(p)
        # News is healthy; action imbalance should be the next verdict.
        # (Regime sparsity may still cascade since bear=0 < 50.)
        assert rep["n_buys"] == 200
        assert rep["n_sells"] == 0
        # One of these verdicts should fire — they have priority order
        # but ACTION_IMBALANCED OR REGIME_BUCKETS_SPARSE both apply here.
        assert rep["verdict"] in (
            "ACTION_IMBALANCED", "REGIME_BUCKETS_SPARSE",
        )
        assert any("BUY 200/200" in h for h in rep["hints"])

    def test_balanced_actions_not_imbalanced_hint(self, tmp_path):
        # 50/50 BUY/SELL with news + regimes → HEALTHY-ish
        rows = []
        regs = ["bull", "sideways", "bear"]
        for i in range(300):
            rows.append({
                "action": "BUY" if i % 2 == 0 else "SELL",
                "ticker": "NVDA",
                "sim_date": f"2024-01-{(i % 28) + 1:02d}",
                "forward_return_5d": (i % 10) - 5,
                "regime_label": regs[i % 3],
                "news_urgency": 50.0, "news_article_count": 3.0,
            })
        p = _write_jsonl(tmp_path, rows)
        rep = och.analyze(p)
        # Each regime has 100 rows → above MIN_REGIME_ROWS=50
        # Actions are balanced → not imbalanced
        # News present → not dark
        assert rep["verdict"] == "HEALTHY"
        assert not any("exceeds" in h for h in rep["hints"])


class TestRegimeSparsityVerdict:
    def test_sparse_bear_regime_flagged(self, tmp_path):
        # Plenty of rows, news present, balanced actions — but
        # bear regime has only 10 rows → sparse
        rows = []
        for i in range(200):
            rows.append({
                "action": "BUY" if i % 2 == 0 else "SELL",
                "ticker": "NVDA",
                "sim_date": f"2024-01-{(i % 28) + 1:02d}",
                "forward_return_5d": (i % 10) - 5,
                "regime_label": "bull" if i < 100 else "sideways",
                "news_urgency": 50.0, "news_article_count": 3.0,
            })
        # 10 bear rows — below MIN_REGIME_ROWS=50
        for i in range(10):
            rows.append({
                "action": "BUY", "ticker": "NVDA",
                "sim_date": f"2024-02-{i + 1:02d}",
                "forward_return_5d": -5.0,
                "regime_label": "bear",
                "news_urgency": 50.0, "news_article_count": 3.0,
            })
        p = _write_jsonl(tmp_path, rows)
        rep = och.analyze(p)
        assert rep["verdict"] == "REGIME_BUCKETS_SPARSE"
        assert rep["regime_counts"]["bear"] == 10
        assert any("bear=10" in h for h in rep["hints"])


class TestRegimeMultFallback:
    def test_regime_decoded_from_mult_when_label_absent(self, tmp_path):
        # Legacy rows have no regime_label — only regime_mult.
        # 0.3 → bear, 0.6 → sideways, 1.0 → bull
        rows = []
        for i in range(150):
            rows.append({
                "action": "BUY",
                "ticker": "NVDA",
                "sim_date": "2024-01-01",
                "forward_return_5d": 1.0,
                "regime_mult": 1.0,
                "news_urgency": 50.0, "news_article_count": 3.0,
            })
        p = _write_jsonl(tmp_path, rows)
        rep = och.analyze(p)
        # Should decode 1.0 → bull
        assert rep["regime_counts"].get("bull", 0) == 150

    def test_unknown_regime_label_lumped_in_unknown(self, tmp_path):
        rows = []
        for i in range(150):
            rows.append({
                "action": "BUY", "ticker": "NVDA",
                "sim_date": "2024-01-01",
                "forward_return_5d": 1.0,
                "regime_label": "unknown",
                "news_urgency": 50.0, "news_article_count": 3.0,
            })
        p = _write_jsonl(tmp_path, rows)
        rep = och.analyze(p)
        assert rep["regime_counts"].get("unknown", 0) == 150


class TestFeatureDensity:
    def test_density_reports_partial_population(self, tmp_path):
        # 100 rows, 70 with RSI populated, 30 without
        rows = []
        for i in range(100):
            r = {"action": "BUY", "ticker": "NVDA",
                 "sim_date": "2024-01-01", "forward_return_5d": 1.0,
                 "regime_label": "bull"}
            if i < 70:
                r["rsi"] = 50.0
            rows.append(r)
        p = _write_jsonl(tmp_path, rows)
        rep = och.analyze(p)
        assert rep["feature_density"]["rsi"]["n_non_null"] == 70
        assert rep["feature_density"]["rsi"]["fraction"] == 0.70

    def test_density_treats_bool_as_present(self, tmp_path):
        # bool features (ema200_above etc.) must count as present
        # when True OR False (only None is missing).
        rows = []
        for i in range(120):
            r = {"action": "BUY", "ticker": "NVDA",
                 "sim_date": "2024-01-01", "forward_return_5d": 1.0,
                 "regime_label": "bull",
                 "ema200_above": (i % 2 == 0)}  # True for even, False odd
            rows.append(r)
        p = _write_jsonl(tmp_path, rows)
        rep = och.analyze(p)
        assert rep["feature_density"]["ema200_above"]["fraction"] == 1.0

    def test_density_treats_nan_as_missing(self, tmp_path):
        # NaN is non-finite — should be treated as missing, not present.
        rows = []
        for i in range(120):
            r = {"action": "BUY", "ticker": "NVDA",
                 "sim_date": "2024-01-01", "forward_return_5d": 1.0,
                 "regime_label": "bull",
                 "rsi": float("nan") if i < 30 else 50.0}
            rows.append(r)
        p = _write_jsonl(tmp_path, rows)
        rep = och.analyze(p)
        # 30 NaN rows must be excluded; only 90 are non-null
        assert rep["feature_density"]["rsi"]["n_non_null"] == 90


class TestTargetStats:
    def test_target_stats_descriptive(self, tmp_path):
        # Known mean/std target so we can assert specific values
        targets = [1.0, 2.0, 3.0, 4.0, 5.0] * 24  # 120 rows
        rows = []
        for i, t in enumerate(targets):
            rows.append({
                "action": "BUY", "ticker": "NVDA",
                "sim_date": "2024-01-01",
                "forward_return_5d": t,
                "regime_label": "bull",
                "news_urgency": 50.0, "news_article_count": 3.0,
            })
        p = _write_jsonl(tmp_path, rows)
        rep = och.analyze(p)
        # mean of [1..5] cycled 24× = 3.0
        assert rep["target"]["mean"] == pytest.approx(3.0)
        assert rep["target"]["min"] == 1.0
        assert rep["target"]["max"] == 5.0
        assert rep["target"]["n"] == 120

    def test_target_degenerate_std_flagged(self, tmp_path):
        # All targets identical → std=0 → TARGET_DEGENERATE
        rows = []
        for _ in range(150):
            rows.append({
                "action": "BUY" if _ % 2 else "SELL",
                "ticker": "NVDA",
                "sim_date": "2024-01-01",
                "forward_return_5d": 1.0,  # constant
                "regime_label": "bull",
                "news_urgency": 50.0, "news_article_count": 3.0,
            })
        # Add enough rows in other regimes too so REGIME_BUCKETS_SPARSE
        # doesn't preempt
        for _ in range(100):
            rows.append({
                "action": "BUY", "ticker": "NVDA",
                "sim_date": "2024-02-01",
                "forward_return_5d": 1.0,
                "regime_label": "sideways",
                "news_urgency": 50.0, "news_article_count": 3.0,
            })
        for _ in range(100):
            rows.append({
                "action": "SELL", "ticker": "NVDA",
                "sim_date": "2024-03-01",
                "forward_return_5d": 1.0,
                "regime_label": "bear",
                "news_urgency": 50.0, "news_article_count": 3.0,
            })
        p = _write_jsonl(tmp_path, rows)
        rep = och.analyze(p)
        # Target std=0 < 1.0pp → TARGET_DEGENERATE verdict.
        # Action imbalance is balanced (BUY/SELL ~50/50), regimes >= 100
        # each, news present, so TARGET_DEGENERATE wins.
        assert rep["verdict"] == "TARGET_DEGENERATE"
        assert rep["target"]["std"] == pytest.approx(0.0)


class TestNeverRaises:
    def test_corrupted_lines_skipped(self, tmp_path):
        # Mix of valid + invalid JSON lines — analyzer must not raise
        p = tmp_path / "outcomes.jsonl"
        with p.open("w") as fh:
            for i in range(120):
                fh.write(json.dumps({
                    "action": "BUY", "ticker": "NVDA",
                    "sim_date": "2024-01-01",
                    "forward_return_5d": 1.0,
                    "regime_label": "bull",
                    "news_urgency": 50.0, "news_article_count": 3.0,
                }) + "\n")
            fh.write("not valid json{\n")
            fh.write("\n")
            fh.write("{another corrupt}\n")
        rep = och.analyze(p)
        # 120 good rows survive
        assert rep["n_total"] == 120

    def test_malformed_row_does_not_crash(self, tmp_path):
        # Row missing action / sim_date / forward_return_5d
        rows = [{"foo": "bar"}, {}, {"action": None}, {"action": "BUY"}]
        for _ in range(120):
            rows.append({
                "action": "BUY", "ticker": "NVDA",
                "sim_date": "2024-01-01",
                "forward_return_5d": 1.0,
                "regime_label": "bull",
                "news_urgency": 50.0, "news_article_count": 3.0,
            })
        p = _write_jsonl(tmp_path, rows)
        rep = och.analyze(p)  # must not raise
        assert rep["status"] == "ok"
        # The 4 malformed rows are counted but don't break action/regime
        # parsing — they fall to OTHER or "unknown".
        assert rep["n_total"] >= 120


class TestCLI:
    def test_cli_json_returns_0_on_healthy(self, tmp_path, capsys):
        # Synthetic healthy corpus
        rows = []
        regs = ["bull", "sideways", "bear"]
        for i in range(300):
            rows.append({
                "action": "BUY" if i % 2 == 0 else "SELL",
                "ticker": "NVDA",
                "sim_date": f"2024-01-{(i % 28) + 1:02d}",
                "forward_return_5d": (i % 10) - 5,
                "regime_label": regs[i % 3],
                "news_urgency": 50.0, "news_article_count": 3.0,
            })
        p = _write_jsonl(tmp_path, rows)
        rc = och.main(["--path", str(p), "--json"])
        assert rc == 0
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert payload["verdict"] == "HEALTHY"

    def test_cli_returns_nonzero_on_bad_corpus(self, tmp_path, capsys):
        # Empty corpus
        p = _write_jsonl(tmp_path, [])
        rc = och.main(["--path", str(p)])
        assert rc != 0

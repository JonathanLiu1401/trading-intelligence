"""``daemon._format_training_pool_composition`` is pure and deterministic —
exact verdict ladder pinned. Mirrors ``test_briefing_label_calibration``'s
shape: a series of curated audit-result snapshots driven through the helper
via the ``audit_fn`` injection point so we never touch the live DB.

The helper surfaces the ``ml.label_audit`` composition fields (synthetic-vs-
LLM-tagged strong-pool fractions) as a one-line briefing footer, silent on
healthy windows and emitting only on extreme synthetic-dominance / Claude-
label-dark conditions. Same Discord-only / silence-on-healthy discipline as
``_format_label_calibration``.
"""
from __future__ import annotations

import daemon


def _audit_dict(*, total, llm=0, briefing_boost=0, synthetic=0, heuristic=0,
                hygiene=0, ml=0):
    """Compose the exact dict shape ``ml.label_audit.audit`` returns."""
    if total > 0:
        synth_frac = round(synthetic / total, 4)
        llm_frac = round((llm + briefing_boost) / total, 4)
        heur_frac = round(heuristic / total, 4)
    else:
        synth_frac = llm_frac = heur_frac = 0.0
    return {
        "strong_pool": {
            "total": total,
            "llm": llm,
            "briefing_boost": briefing_boost,
            "synthetic_backtest_opus": synthetic,
            "heuristic_null_integer": heuristic,
            "reconciles": (llm + briefing_boost + synthetic + heuristic) == total,
        },
        "column_hygiene_violations": hygiene,
        "heuristic_trust_gap": heuristic,
        "heuristic_fraction_of_strong": heur_frac,
        "synthetic_fraction_of_strong": synth_frac,
        "llm_fraction_of_strong": llm_frac,
        "ml_predictions_total": ml,
        "ok": hygiene == 0,
    }


class TestHealthyWindowSilent:
    def test_balanced_pool_emits_nothing(self):
        # 80% LLM-tagged, 20% synthetic — well above the 15% Claude floor.
        data = _audit_dict(total=1000, llm=800, synthetic=200)
        assert daemon._format_training_pool_composition(
            audit_fn=lambda: data
        ) == ""

    def test_borderline_above_threshold_emits_nothing(self):
        # 16% Claude-tagged, 84% synthetic — JUST above the 0.85 dominance
        # threshold so the line stays silent (healthy enough mix).
        data = _audit_dict(total=1000, llm=160, synthetic=840)
        assert daemon._format_training_pool_composition(
            audit_fn=lambda: data
        ) == ""


class TestExtremeSyntheticDominance:
    def test_synthetic_dominant_line_at_8pct_llm(self):
        # 8% Claude-tagged, 92% synthetic — llm_fraction == 0.08 sits above
        # the 0.05 extreme threshold but synth_frac (0.92) exceeds the 0.85
        # dominance bar, so 'synthetic-dominant' fires (not the 'mostly
        # from backtest replay' extreme variant).
        data = _audit_dict(total=10000, llm=800, synthetic=9200)
        line = daemon._format_training_pool_composition(audit_fn=lambda: data)
        assert "🧪 Training pool" in line
        assert "synthetic-dominant" in line
        # Exact numerics carried verbatim: pct of llm, counts, magnitudes.
        assert "8% Claude-tagged" in line
        assert "800 LLM" in line
        assert "9200 synthetic" in line

    def test_live_chronic_state_emits_backtest_replay_line(self):
        # Matches the live 2026-05-20 snapshot proportionally: 3.5% LLM /
        # 96.5% synthetic. llm_fraction (0.035) is BELOW the 0.05 extreme
        # threshold, so the stronger 'mostly from backtest replay' verdict
        # wins over the synthetic-dominant one. This pins the verdict the
        # production briefing will actually carry today.
        data = _audit_dict(total=10000, llm=350, synthetic=9650)
        line = daemon._format_training_pool_composition(audit_fn=lambda: data)
        assert "🧪 Training pool" in line
        assert "backtest replay" in line
        assert "4% Claude-tagged" in line  # round(0.035*100) == 4
        assert "350 LLM" in line
        assert "9650 synthetic" in line

    def test_exactly_at_synth_threshold(self):
        # synth_frac == 0.85 should TRIGGER (>= comparison) given
        # llm_frac >= 0.05. Use llm=150 (15%), synth=850 (85%).
        data = _audit_dict(total=1000, llm=150, synthetic=850)
        line = daemon._format_training_pool_composition(audit_fn=lambda: data)
        assert "synthetic-dominant" in line


class TestClaudeLabelsEffectivelyDark:
    def test_below_5_percent_llm_emits_extreme_line(self):
        # 1% LLM / 99% synthetic — the urgency_scorer is dark / quota-floored
        # to near-zero. The 'mostly from backtest replay' verdict overrides
        # the synthetic-dominant one.
        data = _audit_dict(total=10000, llm=100, synthetic=9900)
        line = daemon._format_training_pool_composition(audit_fn=lambda: data)
        assert "Training pool: only" in line
        assert "1% Claude-tagged" in line
        assert "backtest replay" in line

    def test_zero_llm_emits_extreme_line(self):
        # No Claude labels at all — the strongest signal an analyst should
        # see. The 'mostly from backtest replay' verdict applies.
        data = _audit_dict(total=10000, synthetic=10000)
        line = daemon._format_training_pool_composition(audit_fn=lambda: data)
        assert "Training pool: only 0%" in line
        assert "backtest replay" in line


class TestSmallPoolSilent:
    def test_below_min_size_emits_nothing(self):
        # A tiny pool (< 100 rows) can swing dramatically on a single label —
        # the composition would be analyst-noise. Even at 0% LLM, the
        # helper stays silent until the pool is meaningful.
        data = _audit_dict(total=50, synthetic=50)
        assert daemon._format_training_pool_composition(
            audit_fn=lambda: data
        ) == ""

    def test_empty_pool_emits_nothing(self):
        data = _audit_dict(total=0)
        assert daemon._format_training_pool_composition(
            audit_fn=lambda: data
        ) == ""


class TestBestEffortSilenceOnFailure:
    def test_audit_raises_returns_empty(self):
        def _boom():
            raise RuntimeError("audit DB locked")
        assert daemon._format_training_pool_composition(audit_fn=_boom) == ""

    def test_non_dict_audit_returns_empty(self):
        assert daemon._format_training_pool_composition(audit_fn=lambda: None) == ""
        assert daemon._format_training_pool_composition(audit_fn=lambda: 42) == ""

    def test_malformed_audit_dict_returns_empty(self):
        # Missing strong_pool key
        assert daemon._format_training_pool_composition(
            audit_fn=lambda: {"ok": True}
        ) == ""

    def test_non_numeric_fractions_return_empty(self):
        bad = _audit_dict(total=10000, synthetic=10000)
        bad["synthetic_fraction_of_strong"] = "not a number"
        assert daemon._format_training_pool_composition(
            audit_fn=lambda: bad
        ) == ""


class TestMaxCharsTruncation:
    def test_long_line_truncated_with_ellipsis(self):
        # Force a verdict line then crank max_chars below it; the helper
        # truncates with the ellipsis suffix.
        data = _audit_dict(total=10000, llm=100, synthetic=9900)
        out = daemon._format_training_pool_composition(
            audit_fn=lambda: data, max_chars=40
        )
        assert len(out) <= 40
        assert out.endswith("…")

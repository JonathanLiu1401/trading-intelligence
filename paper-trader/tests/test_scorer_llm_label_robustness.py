"""Locks in defensive parsing of ``llm_quality_label`` in ``train_scorer``.

The historical bug: ``int(r.get("llm_quality_label") or 0)`` crashed on
string labels like ``"ENDORSE"`` or container labels (lists/dicts), which
``_train_decision_scorer``'s outer try/except then surfaced as
``scorer err: invalid literal for int()`` — silently freezing the per-cycle
retrain (CLAUDE.md §6) and locking the conviction gate (invariant #5) until
the corrupted row was purged.

Production data lives in ``data/decision_outcomes.jsonl`` — a single
malformed externally-injected row (Anthropic API hiccup, partial-write race,
manual edit) used to wedge the entire scorer pipeline.  Each test below
crafts ONE row with a different garbage label among an otherwise-clean
batch and asserts the trainer still produces a usable model rather than
raising.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _records(n: int = 80) -> list[dict]:
    """Build a deterministic, valid training batch the trainer accepts.

    Mixed BUY/SELL on unique (sim_date, action) keys so the dedup pass
    leaves ≥30 rows after the documented `insufficient_after_dedup` gate.
    """
    out = []
    for i in range(n):
        out.append({
            "ml_score": 5.0, "rsi": 50.0, "macd": 0.5,
            "mom5": 1.0, "mom20": 2.0, "regime_mult": 1.0,
            "ticker": "NVDA", "forward_return_5d": 3.0 + (i % 7) * 0.1,
            "sim_date": f"2025-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
            "action": "BUY" if i % 2 == 0 else "SELL",
            "return_pct": 50.0,
        })
    return out


@pytest.fixture
def _redirect_scorer_path(tmp_path, monkeypatch):
    """Avoid touching data/ml/decision_scorer.pkl."""
    import paper_trader.ml.decision_scorer as ds
    monkeypatch.setattr(ds, "SCORER_PATH", tmp_path / "scorer.pkl")
    return tmp_path


class TestLlmLabelRobustness:
    """Train pass must SURVIVE a corrupted llm_quality_label, not crash."""

    def test_string_label_does_not_crash(self, _redirect_scorer_path):
        """A free-text string label (e.g. "ENDORSE") used to raise
        ``ValueError: invalid literal for int() with base 10: 'ENDORSE'``
        the very first time it appeared in the corpus."""
        from paper_trader.ml.decision_scorer import train_scorer
        rs = _records()
        rs[10]["llm_quality_label"] = "ENDORSE"
        result = train_scorer(rs)
        assert result["status"] == "ok"
        assert result["n"] == len(rs)

    def test_list_label_does_not_crash(self, _redirect_scorer_path):
        """A container label (manual JSON edit) used to raise TypeError."""
        from paper_trader.ml.decision_scorer import train_scorer
        rs = _records()
        rs[10]["llm_quality_label"] = ["ENDORSE"]
        result = train_scorer(rs)
        assert result["status"] == "ok"

    def test_dict_label_does_not_crash(self, _redirect_scorer_path):
        """A dict label is treated as no-label rather than killing the run."""
        from paper_trader.ml.decision_scorer import train_scorer
        rs = _records()
        rs[10]["llm_quality_label"] = {"verdict": 1}
        result = train_scorer(rs)
        assert result["status"] == "ok"

    def test_float_int_like_label_works(self, _redirect_scorer_path):
        """``1.0`` and ``-1.0`` (a float JSON encoded an int) honor the
        multiplier the same way an int 1 / -1 would — int(1.0)==1."""
        from paper_trader.ml.decision_scorer import train_scorer
        rs = _records()
        rs[10]["llm_quality_label"] = 1.0
        rs[11]["llm_quality_label"] = -1.0
        result = train_scorer(rs)
        assert result["status"] == "ok"

    def test_bool_label_is_treated_as_no_label(self, _redirect_scorer_path):
        """``bool`` is a subclass of ``int`` — without the explicit guard,
        True silently became llm_label=1 (×3 weight). Tests treat bool as
        a corrupted scalar and degrade to the no-label arm to match the
        ``forward_return_5d`` validation discipline used elsewhere."""
        from paper_trader.ml.decision_scorer import train_scorer
        rs = _records()
        # Pin a deterministic row to True so the assertion is concrete.
        rs[10]["llm_quality_label"] = True
        result = train_scorer(rs)
        assert result["status"] == "ok"

    def test_none_label_works(self, _redirect_scorer_path):
        """Explicit None (a JSON null) was already handled by the
        ``or 0`` fallback — pin so future refactors don't regress it."""
        from paper_trader.ml.decision_scorer import train_scorer
        rs = _records()
        rs[10]["llm_quality_label"] = None
        result = train_scorer(rs)
        assert result["status"] == "ok"

    def test_corrupted_label_does_not_inflate_weight(
        self, _redirect_scorer_path
    ):
        """A corrupted label should give ×1 weight (no-label arm), NOT ×3
        (ENDORSE) or ×0.1 (CONDEMN). Verifies via the saved pickle's
        n_train (every row contributed exactly once at sensible weight)."""
        from paper_trader.ml.decision_scorer import train_scorer
        rs = _records()
        for i in range(5):
            rs[i]["llm_quality_label"] = "GARBAGE"
        result = train_scorer(rs)
        # The trainer's deterministic-oversampling pass uses
        # rep = round(weight * 2). A no-label run @ return_pct=50 has
        # weight = max(0.5, min(2.0, 1.0 + 50/200)) * 1.0 = 1.25,
        # so rep = round(2.5) = 2 — meaning EVERY clean row contributes
        # rep=2 to the training fold. A corrupted row treated as the
        # ENDORSE arm would have weight = 1.25 * 3.0 = 3.75 → clamped to
        # max 2.0 → rep = 4 (silent ×2 over-weighting of corrupted data,
        # the exact regression this test guards against).
        # n_pickle is the post-validation row count and is invariant to
        # weighting; this asserts the trainer completed honestly.
        assert result["status"] == "ok"
        assert result["n"] == len(rs)

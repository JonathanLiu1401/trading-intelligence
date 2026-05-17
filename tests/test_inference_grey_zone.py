"""ml.inference grey-zone / needs_llm routing.

This pins a behavior that is a known code-vs-doc trap: the CLAUDE.md glossary
and the ml/model.py docstring describe the "grey zone" as a *relevance*
prediction band, but the actual router keys ``in_grey`` on the **urgency head**
(sigmoid probability scaled to 0..10). That is deliberate — the LLM on this
path does urgent/not-urgent disambiguation, and ``LLM_ZONE_MID_LO..HI`` (7.0,
8.5) straddles the 8.0 urgent threshold. A previous reviewer "correcting" the
code to use relevance would silently change which articles burn a Sonnet call.
These tests fail loudly if that happens.

``score_articles`` is exercised with stubbed embedder/model so the routing
decision is deterministic and independent of any trained checkpoint.
"""
from __future__ import annotations

import numpy as np
import pytest

from ml import inference
from ml.trainer import (
    LLM_ZONE_CLEAR_NOISE,
    LLM_ZONE_MID_HI,
    LLM_ZONE_MID_LO,
    UNCERTAINTY_REL,
)


class _StubEmb:
    fitted = True

    def transform(self, texts):
        # Width is arbitrary — the stub model ignores X content and only uses
        # the row count. inference concatenates the real 15 extra features.
        return np.zeros((len(texts), 8), dtype=np.float32)


class _StubModel:
    """predict() returns constant (rel, rel_std, urg, urg_std, ts) for every row."""

    fitted = True

    def __init__(self, rel, urg, rel_std=0.1, urg_std=0.1, ts=0.5):
        self._rel, self._urg = rel, urg
        self._rstd, self._ustd, self._ts = rel_std, urg_std, ts

    def predict(self, X):
        n = X.shape[0]
        return (
            np.full(n, self._rel, dtype=np.float64),
            np.full(n, self._rstd, dtype=np.float64),
            np.full(n, self._urg, dtype=np.float64),
            np.full(n, self._ustd, dtype=np.float64),
            np.full(n, self._ts, dtype=np.float64),
        )


@pytest.fixture
def patch_stack(monkeypatch):
    def _apply(model):
        monkeypatch.setattr(inference, "get_embedder", lambda: _StubEmb())
        monkeypatch.setattr(inference, "get_model", lambda: model)

    return _apply


_ARTS = [{"_id": "x", "title": "Micron guides DRAM ASP higher", "summary": "body"}]


def test_unfitted_model_routes_to_llm(monkeypatch):
    """Sentinel path: embedder/model not fitted → every article needs_llm with
    the rel_std==99 marker (the value scorer_worker keys on to skip ts writes)."""
    class _Unfit:
        fitted = False

    monkeypatch.setattr(inference, "get_embedder", lambda: _Unfit())
    monkeypatch.setattr(inference, "get_model", lambda: _Unfit())
    out = inference.score_articles(_ARTS)
    assert len(out) == 1
    assert out[0].needs_llm is True
    assert out[0].rel_std == 99


def test_urgency_in_grey_band_routes_to_llm(patch_stack):
    """Urgency-head estimate inside [LLM_ZONE_MID_LO, LLM_ZONE_MID_HI] with low
    spread → escalate to Sonnet even though variance is tiny."""
    mid_urg = (LLM_ZONE_MID_LO + LLM_ZONE_MID_HI) / 2.0  # 7.75
    patch_stack(_StubModel(rel=5.0, urg=mid_urg, rel_std=0.1, urg_std=0.1))
    out = inference.score_articles(_ARTS)[0]
    assert out.needs_llm is True
    assert out.confident_noise is False


def test_relevance_in_band_but_low_urgency_does_NOT_route(patch_stack):
    """The decisive assertion: a *relevance* value sitting inside the same
    7.0..8.5 numeric band must NOT trigger LLM routing when urgency is low and
    variance is small. If someone re-points in_grey at relevance, this flips."""
    rel_in_band = (LLM_ZONE_MID_LO + LLM_ZONE_MID_HI) / 2.0  # 7.75
    patch_stack(_StubModel(rel=rel_in_band, urg=2.0, rel_std=0.1, urg_std=0.1))
    out = inference.score_articles(_ARTS)[0]
    assert out.needs_llm is False, (
        "grey-zone routing must key on the urgency head, not relevance"
    )
    assert out.relevance == pytest.approx(rel_in_band, abs=0.01)


def test_high_relevance_variance_forces_llm(patch_stack):
    """Wide ensemble spread on relevance overrides a confident-looking mean."""
    patch_stack(_StubModel(rel=5.0, urg=1.0,
                            rel_std=UNCERTAINTY_REL + 0.5, urg_std=0.1))
    out = inference.score_articles(_ARTS)[0]
    assert out.needs_llm is True


def test_confident_noise_is_skipped_not_routed(patch_stack):
    """rel < LLM_ZONE_CLEAR_NOISE with tight spread → confident_noise, and
    confident_noise must suppress needs_llm (don't pay Sonnet for clear junk)."""
    patch_stack(_StubModel(rel=LLM_ZONE_CLEAR_NOISE - 1.0, urg=0.5,
                            rel_std=0.1, urg_std=0.1))
    out = inference.score_articles(_ARTS)[0]
    assert out.confident_noise is True
    assert out.needs_llm is False

"""ArticleNet output contract — relevance in [0,10], urgency in [0,1], no NaN."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from ml.model import ArticleNetModule, INPUT_DIM


def test_relevance_in_zero_to_ten():
    """relevance_head outputs sigmoid * 10. Used as ai_score 0..10 throughout
    the system — out-of-range outputs would break the urgency threshold."""
    net = ArticleNetModule(input_dim=64)
    net.eval()
    x = torch.randn(16, 64)
    with torch.no_grad():
        rel, urg, unc, tsens = net(x)
    rel_np = rel.cpu().numpy().flatten()
    assert rel_np.min() >= 0.0
    assert rel_np.max() <= 10.0
    assert np.isfinite(rel_np).all()


def test_urgency_in_zero_to_one():
    net = ArticleNetModule(input_dim=64)
    net.eval()
    x = torch.randn(16, 64)
    with torch.no_grad():
        _, urg, _, _ = net(x)
    urg_np = urg.cpu().numpy().flatten()
    assert urg_np.min() >= 0.0
    assert urg_np.max() <= 1.0


def test_time_sensitivity_in_zero_to_one():
    net = ArticleNetModule(input_dim=64)
    net.eval()
    x = torch.randn(8, 64)
    with torch.no_grad():
        _, _, _, ts = net(x)
    ts_np = ts.cpu().numpy().flatten()
    assert ts_np.min() >= 0.0
    assert ts_np.max() <= 1.0


def test_uncertainty_in_zero_to_one():
    net = ArticleNetModule(input_dim=64)
    net.eval()
    x = torch.randn(8, 64)
    with torch.no_grad():
        _, _, unc, _ = net(x)
    unc_np = unc.cpu().numpy().flatten()
    assert unc_np.min() >= 0.0
    assert unc_np.max() <= 1.0


def test_zero_input_does_not_nan():
    """Zero input through 4 LayerNorm+GELU layers must produce finite output.
    NaN here would propagate to ai_score and silently corrupt the alert path."""
    net = ArticleNetModule(input_dim=128)
    net.eval()
    x = torch.zeros(4, 128)
    with torch.no_grad():
        rel, urg, unc, ts = net(x)
    for out in (rel, urg, unc, ts):
        assert torch.isfinite(out).all(), "non-finite output from zero input"


def test_forward_batch_shape():
    net = ArticleNetModule(input_dim=32)
    net.eval()
    x = torch.randn(7, 32)
    with torch.no_grad():
        rel, urg, unc, ts = net(x)
    assert rel.shape == (7, 1)
    assert urg.shape == (7, 1)
    assert unc.shape == (7, 1)
    assert ts.shape == (7, 1)

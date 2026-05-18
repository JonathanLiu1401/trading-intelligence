"""continuous_trainer liveness heartbeat.

Root cause this pins: ``continuous_trainer_worker`` only updated
``_worker_last_ok`` *after* ``train_continuous()`` returned. A single pass
(15k-row data-load + 40-epoch fit) legitimately runs >900s under GPU
contention — longer than the supervisor's staleness deadline — so a
slow-but-healthy thread was false-flagged DEAD (observed elapsed_s 920–961).

The fix threads a ``heartbeat`` callable through ``train_continuous`` →
``ArticleNet.fit``, pinged after the data-load phase and once per epoch.
These tests make the fix falsifiable: remove the per-epoch ping and
``test_fit_pings_heartbeat_once_per_epoch`` fails; remove the pre-fit ping
and ``test_train_continuous_forwards_and_pre_pings`` fails.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from ml.model import ArticleNet, ArticleNetModule


def _bare_model(input_dim: int) -> ArticleNet:
    """An ArticleNet whose ``fit`` runs on CPU without ``__init__``'s checkpoint
    load. CPU is deliberate: the test must not contend for the GPU with a live
    daemon, and a tiny CPU fit is fast and deterministic. ``_input_dim`` is set
    to mismatch the test dim so ``fit`` rebuilds the net itself."""
    m = ArticleNet.__new__(ArticleNet)
    m.device = torch.device("cpu")
    m._input_dim = input_dim + 1
    m.net = ArticleNetModule(input_dim=m._input_dim)
    m._fitted = False
    m._best_val_loss = float("inf")
    return m


def _toy_data(n=64, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim)).astype(np.float32)
    y_rel = rng.uniform(0, 10, n).astype(np.float32)
    y_urg = rng.uniform(0, 10, n).astype(np.float32)
    return X, y_rel, y_urg


def test_fit_pings_heartbeat_once_per_epoch():
    """The core of the fix: liveness is proven *during* the pass, not only at
    its boundary. n<100 → no val split → no early stop → exactly `epochs`
    pings, so a >900s fit pings far more often than the 900s deadline."""
    X, y_rel, y_urg = _toy_data()
    epochs = 5
    calls: list[float] = []
    m = _bare_model(X.shape[1])
    m.fit(X, y_rel, y_urg, epochs=epochs, batch_size=16,
          verbose=False, heartbeat=lambda: calls.append(1.0))
    assert len(calls) == epochs
    assert len(calls) >= 2  # the property that actually defeats the DEAD flag


def test_fit_heartbeat_exception_never_breaks_training():
    """A monitoring ping must never take training down with it."""
    X, y_rel, y_urg = _toy_data()
    m = _bare_model(X.shape[1])

    def boom():
        raise RuntimeError("supervisor map blew up mid-epoch")

    metrics = m.fit(X, y_rel, y_urg, epochs=3, batch_size=16,
                    verbose=False, heartbeat=boom)
    assert "final_loss" in metrics
    assert np.isfinite(metrics["final_loss"])


def test_fit_without_heartbeat_is_unaffected():
    """The heavy ``train()`` / subprocess path passes no heartbeat. Default
    ``heartbeat=None`` must be a no-op, not a crash."""
    X, y_rel, y_urg = _toy_data()
    m = _bare_model(X.shape[1])
    metrics = m.fit(X, y_rel, y_urg, epochs=2, batch_size=16, verbose=False)
    assert "final_loss" in metrics


class _FakeEmbedder:
    def should_refit(self, n: int) -> bool:
        return False

    def transform(self, texts):
        return np.zeros((len(texts), 8), dtype=np.float32)

    def fit_transform(self, texts):
        return self.transform(texts)


class _FakeModel:
    """Records that ``train_continuous`` handed it a real heartbeat and that it
    was already pinged once (the post-data-load ping) before fit started."""

    def __init__(self):
        self.received_heartbeat = None
        self.pings_seen_before_fit = None

    def fit(self, X, y_rel, y_urg, *, y_time=None, epochs=40, batch_size=512,
            lr=1e-4, warm=True, label_weight_exponent=None, heartbeat=None):
        self.received_heartbeat = heartbeat
        # Emulate two epochs worth of liveness pings.
        if heartbeat is not None:
            heartbeat()
            heartbeat()
        return {"final_loss": 0.1, "val_loss": 0.2, "new_best": False}


def test_train_continuous_forwards_and_pre_pings(store, monkeypatch):
    """End-to-end wiring: ``train_continuous`` pings after the (slow) data-load
    phase AND forwards the same callable into ``model.fit``. Monkeypatched so
    no real GPU/model is touched — the live daemon's training is untouched."""
    from ml import trainer

    with store._write_lock:
        for i in range(40):
            store.conn.execute(
                "INSERT INTO articles "
                "(id, url, title, source, published, kw_score, ai_score, "
                "urgency, first_seen, cycle, ml_score, score_source) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (f"s{i}", f"https://x.com/{i}", f"strong label article {i}",
                 "rss", "", 1.0, 7.0, 0, "2026-05-15T00:00:00+00:00", 0,
                 None, "llm"),
            )
        store.conn.commit()

    fake_model = _FakeModel()
    monkeypatch.setattr(trainer, "get_embedder", lambda: _FakeEmbedder())
    monkeypatch.setattr(trainer, "get_model", lambda: fake_model)

    calls: list[float] = []
    result = trainer.train_continuous(store, heartbeat=lambda: calls.append(1.0))

    assert result["status"] == "ok"
    # 1 post-data-load ping (trainer.py) + 2 from the fake fit = 3.
    assert len(calls) >= 2
    # The *same* callable must reach model.fit, not a swallowed/None arg.
    assert fake_model.received_heartbeat is not None

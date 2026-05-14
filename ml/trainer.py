"""
Online trainer — pulls labeled articles from the store, trains ArticleNet on GPU.

Phase 1: bootstraps on heuristic kw_score (weak labels, available immediately).
Phase 2: retrains as Sonnet ai_score labels accumulate (stronger labels).

Aggressive GPU training: 50 epochs per cycle, batch_size=256, Adam + cosine LR.
Schedule lives in daemon.py (RETRAIN_INTERVAL drives the cadence).
"""
import threading
import time
import numpy as np

from ml.embedder import get_embedder
from ml.model import get_model

# Serialize all model.fit() calls. ml_trainer_worker and continuous_trainer_worker
# both mutate the same global ArticleNet on GPU; without this lock, overlapping
# fits raise "variable needed for gradient computation has been modified by an
# inplace operation" because autograd buffers from one pass get overwritten by
# the other before backward() finishes.
_TRAIN_LOCK = threading.Lock()

# Cadence — daemon reads this if it wants, but daemon.ML_TRAIN_INTERVAL is the
# source of truth. Set to 30 min to keep the GPU busy.
RETRAIN_INTERVAL = 1800     # retrain at most once every 30 minutes
MIN_NEW_LABELS   = 200      # retrain if this many new LLM labels since last train

# Training hyperparameters
EPOCHS_PER_CYCLE = 50
BATCH_SIZE       = 256
LEARNING_RATE    = 1e-3
MIN_TRAIN_SAMPLES = 100      # below this we bootstrap from kw_score weak labels

# Uncertainty thresholds — articles with uncertainty above this go to LLM
UNCERTAINTY_REL  = 1.5      # std in relevance score
UNCERTAINTY_URG  = 1.2      # std in urgency score
# Score zones where we still defer to LLM regardless of confidence
LLM_ZONE_MID_LO  = 7.0
LLM_ZONE_MID_HI  = 8.5
LLM_ZONE_CLEAR_NOISE  = 3.0  # below this → clearly noise, skip LLM


def _fetch_training_data(store) -> tuple[list[str], np.ndarray, np.ndarray, str]:
    """
    Pull articles with scores from the store.
    Returns (texts, y_rel, y_urg, source) where source ∈ {"strong","weak","mixed"}.
    """
    from storage.article_store import decompress

    texts, rels, urgs = [], [], []
    source = "strong"

    # Strong labels: Sonnet ai_score
    cur = store.conn.execute(
        "SELECT title, full_text, ai_score FROM articles "
        "WHERE ai_score > 0 ORDER BY first_seen DESC LIMIT 15000"
    )
    for title, blob, ai in cur.fetchall():
        summary = decompress(blob) if blob else ""
        texts.append(f"{title} {summary}")
        rels.append(float(ai))
        urgs.append(float(ai) if ai >= 8.0 else 0.0)

    n_strong = len(texts)

    # Bootstrap with weak labels when strong labels are sparse
    if n_strong < MIN_TRAIN_SAMPLES:
        cur2 = store.conn.execute(
            "SELECT title, full_text, kw_score FROM articles "
            "WHERE ai_score = 0 AND kw_score > 0 ORDER BY kw_score DESC LIMIT 8000"
        )
        for title, blob, kw in cur2.fetchall():
            summary = decompress(blob) if blob else ""
            texts.append(f"{title} {summary}")
            rels.append(float(kw))
            urgs.append(0.0)
        source = "weak" if n_strong == 0 else "mixed"

    y_rel = np.clip(np.array(rels, dtype=np.float32), 0, 10)
    y_urg = np.clip(np.array(urgs, dtype=np.float32), 0, 10)
    return texts, y_rel, y_urg, source


def train(store, force: bool = False) -> dict:
    """Train/retrain ArticleNet on available labeled data. Returns metrics dict."""
    t0 = time.time()
    texts, y_rel, y_urg, source = _fetch_training_data(store)

    if len(texts) < 50:
        return {"status": "skipped", "reason": "too_few_samples", "n": len(texts)}

    print(f"[ml:trainer] Training on {len(texts)} samples "
          f"({int((y_rel > 0).sum())} labeled, {int((y_urg >= 8).sum())} urgent) "
          f"source={source}")

    emb = get_embedder()
    if not emb.fitted:
        X = emb.fit_transform(texts)
    else:
        X = emb.transform(texts)

    model = get_model()
    with _TRAIN_LOCK:
        metrics = model.fit(
            X, y_rel, y_urg,
            epochs=EPOCHS_PER_CYCLE,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
        )

    elapsed = time.time() - t0
    return {
        "status": "ok",
        "n": len(texts),
        "n_llm_labeled": int((y_rel > 0).sum()),
        "n_urgent": int((y_urg >= 8).sum()),
        "label_source": source,
        "final_loss": metrics.get("final_loss"),
        "epochs": metrics.get("epochs"),
        "device": metrics.get("device"),
        "elapsed_s": round(elapsed, 1),
    }


_last_continuous_loss = float('inf')


def train_continuous(store) -> dict:
    """Lightweight continuous GPU training pass — 20 epochs on all scored articles."""
    global _last_continuous_loss
    from storage.article_store import decompress

    t0 = time.time()

    texts, rels, urgs = [], [], []
    cur = store.conn.execute(
        "SELECT title, full_text, ai_score FROM articles WHERE ai_score > 0"
    )
    for title, blob, ai in cur.fetchall():
        summary = decompress(blob) if blob else ""
        texts.append(f"{title} {summary}")
        rels.append(float(ai))
        urgs.append(float(ai) if ai >= 8.0 else 0.0)

    n = len(texts)
    if n < 50:
        return {"status": "skipped", "reason": "too_few_samples", "n": n}

    y_rel = np.clip(np.array(rels, dtype=np.float32), 0, 10)
    y_urg = np.clip(np.array(urgs, dtype=np.float32), 0, 10)

    emb = get_embedder()
    if not emb.fitted:
        X = emb.fit_transform(texts)
    else:
        X = emb.transform(texts)

    model = get_model()
    # Skip rather than block: if the heavy trainer is mid-fit, just wait for the
    # next 60s tick. Avoids piling up continuous passes behind a long retrain.
    if not _TRAIN_LOCK.acquire(blocking=False):
        return {"status": "skipped", "reason": "trainer_busy", "n": n}
    try:
        metrics = model.fit(X, y_rel, y_urg, epochs=20, batch_size=512, lr=1e-3)
    finally:
        _TRAIN_LOCK.release()

    final_loss = metrics.get("final_loss")
    if final_loss is not None and final_loss < _last_continuous_loss:
        if hasattr(model, "save"):
            try:
                model.save()
            except Exception:
                pass
        _last_continuous_loss = final_loss

    try:
        import torch
        gpu_mem_mb = round(torch.cuda.memory_allocated() / 1024 / 1024, 1) if torch.cuda.is_available() else 0.0
    except Exception:
        gpu_mem_mb = 0.0

    elapsed = round(time.time() - t0, 1)
    return {
        "status": "ok",
        "n": n,
        "loss": final_loss,
        "gpu_mem_mb": gpu_mem_mb,
        "elapsed_s": elapsed,
    }

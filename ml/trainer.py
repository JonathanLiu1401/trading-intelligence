"""
Online trainer — pulls labeled articles from the store, trains ArticleNet.

Phase 1: bootstraps on heuristic kw_score (weak labels, available immediately).
Phase 2: retrains as Sonnet ai_score labels accumulate (stronger labels).

Training schedule (managed by daemon's ml_trainer_worker):
  - Bootstrap run on startup if no checkpoint exists
  - Retrain every RETRAIN_INTERVAL seconds or when MIN_NEW_LABELS new LLM labels arrive
"""
import time
import numpy as np

from ml.embedder import get_embedder
from ml.model import get_model

RETRAIN_INTERVAL = 3600     # retrain at most once per hour
MIN_NEW_LABELS   = 200      # retrain if this many new LLM labels since last train

# Uncertainty thresholds — articles with uncertainty above this go to LLM
UNCERTAINTY_REL  = 1.5      # std in relevance score
UNCERTAINTY_URG  = 1.2      # std in urgency score
# Score zones where we still defer to LLM regardless of confidence
LLM_ZONE_MID_LO  = 7.0
LLM_ZONE_MID_HI  = 8.5
LLM_ZONE_CLEAR_NOISE  = 3.0  # below this → clearly noise, skip LLM


def _fetch_training_data(store) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Pull articles with scores from the store."""
    from storage.article_store import decompress

    texts, rels, urgs = [], [], []

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

    # Weak labels: heuristic kw_score (bootstrap when LLM labels are sparse)
    if len(texts) < 500:
        cur2 = store.conn.execute(
            "SELECT title, full_text, kw_score FROM articles "
            "WHERE ai_score = 0 AND kw_score > 0 ORDER BY kw_score DESC LIMIT 8000"
        )
        for title, blob, kw in cur2.fetchall():
            summary = decompress(blob) if blob else ""
            texts.append(f"{title} {summary}")
            rels.append(float(kw))
            urgs.append(0.0)

    y_rel = np.clip(np.array(rels, dtype=np.float32), 0, 10)
    y_urg = np.clip(np.array(urgs, dtype=np.float32), 0, 10)
    return texts, y_rel, y_urg


def train(store, force: bool = False) -> dict:
    """Train/retrain ArticleNet on available labeled data. Returns metrics dict."""
    t0 = time.time()
    texts, y_rel, y_urg = _fetch_training_data(store)

    if len(texts) < 50:
        return {"status": "skipped", "reason": "too_few_samples", "n": len(texts)}

    print(f"[ml:trainer] Training on {len(texts)} samples "
          f"({int((y_rel > 0).sum())} labeled, {int((y_urg >= 8).sum())} urgent)")

    emb = get_embedder()
    if not emb.fitted:
        X = emb.fit_transform(texts)
    else:
        X = emb.transform(texts)

    model = get_model()
    model.fit(X, y_rel, y_urg)

    elapsed = time.time() - t0
    return {
        "status": "ok",
        "n": len(texts),
        "n_llm_labeled": int((y_rel > 0).sum()),
        "n_urgent": int((y_urg >= 8).sum()),
        "elapsed_s": round(elapsed, 1),
    }

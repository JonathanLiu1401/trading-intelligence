"""
Online trainer — pulls LLM-labeled articles from the store, trains ArticleNet.

Phase 1: bootstraps on heuristic kw_score (weak labels, available immediately).
Phase 2: retrains as Sonnet ai_score labels accumulate (stronger labels).

Training schedule (managed by daemon's ml_trainer_worker):
  - Bootstrap run on startup if no checkpoint exists
  - Retrain every RETRAIN_INTERVAL seconds or every MIN_NEW_LABELS new LLM labels
  - Saves checkpoint to MODEL_DIR/articlenet.pt
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from ml.model import ArticleNet
from ml.embedder import get_embedder

MODEL_DIR = Path(os.environ.get("DIGITAL_INTERN_ML_DIR",
                                Path(__file__).resolve().parent.parent / "data" / "ml"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT   = MODEL_DIR / "articlenet.pt"
RETRAIN_INTERVAL  = 3600        # retrain at most once per hour
MIN_NEW_LABELS    = 200         # retrain if this many new LLM labels since last train
BATCH_SIZE        = 64
EPOCHS            = 10
LR                = 3e-4
WEIGHT_DECAY      = 1e-4

# Uncertainty thresholds — articles with uncertainty above this go to LLM
UNCERTAINTY_REL   = 1.5         # std in relevance score
UNCERTAINTY_URG   = 1.2         # std in urgency score
# Score zones where we still defer to LLM regardless of confidence
LLM_ZONE_LOW      = 6.0         # below this is clearly noise → skip LLM
LLM_ZONE_HIGH     = 9.0         # above this is clearly urgent → skip LLM
LLM_ZONE_MID_LO   = 7.0         # 7-8.5 is the "maybe urgent" grey zone
LLM_ZONE_MID_HI   = 8.5


_model: ArticleNet | None = None
_model_loaded_at: float = 0.0


def get_model(input_dim: int | None = None) -> ArticleNet:
    global _model, _model_loaded_at
    if _model is not None:
        return _model

    emb = get_embedder()
    dim = input_dim or (emb.dim if emb.fitted else 15_000)
    _model = ArticleNet(input_dim=dim)

    if CHECKPOINT.exists():
        state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        # handle dim mismatch (vectorizer was refitted with different vocab)
        try:
            _model.load_state_dict(state)
            print(f"[ml] Loaded checkpoint from {CHECKPOINT}")
        except RuntimeError:
            print("[ml] Checkpoint dim mismatch — starting fresh")
            _model = ArticleNet(input_dim=dim)
    else:
        print("[ml] No checkpoint — model initialized randomly")

    _model.eval()
    _model_loaded_at = time.time()
    return _model


def _fetch_training_data(store) -> tuple[list[str], list[float], list[float]]:
    """Pull articles with scores from the store.

    Priority: ai_score > 0 (LLM label) → use directly.
    Fallback: kw_score > 0 (heuristic) → map 0-10 as weak relevance label, urgency=0.
    Returns (texts, rel_labels, urg_labels).
    """
    # LLM-labeled articles (strong)
    cur = store.conn.execute(
        "SELECT title, full_text, kw_score, ai_score FROM articles "
        "WHERE ai_score > 0 ORDER BY first_seen DESC LIMIT 10000"
    )
    rows = cur.fetchall()

    texts, rels, urgs = [], [], []
    from storage.article_store import decompress

    for title, blob, kw, ai in rows:
        summary = decompress(blob) if blob else ""
        texts.append(f"{title} {summary}")
        rels.append(float(ai))
        # urgency: if ai_score >= 8 treat as urgent; else 0
        urgs.append(float(ai) if ai >= 8.0 else 0.0)

    # Bootstrap with heuristic scores if not enough LLM labels
    if len(texts) < 500:
        cur2 = store.conn.execute(
            "SELECT title, full_text, kw_score FROM articles "
            "WHERE ai_score = 0 AND kw_score > 0 ORDER BY kw_score DESC LIMIT 5000"
        )
        for title, blob, kw in cur2.fetchall():
            summary = decompress(blob) if blob else ""
            texts.append(f"{title} {summary}")
            rels.append(float(kw))    # weak label
            urgs.append(0.0)          # no urgency info

    return texts, rels, urgs


def train(store, force: bool = False) -> dict:
    """
    Train (or retrain) ArticleNet on all available labeled data.
    Returns metrics dict.
    """
    t0 = time.time()
    texts, rels, urgs = _fetch_training_data(store)
    if len(texts) < 50:
        return {"status": "skipped", "reason": "too_few_samples", "n": len(texts)}

    print(f"[ml:trainer] Training on {len(texts)} samples...")

    emb = get_embedder()
    if not emb.fitted:
        emb.fit(texts)
        X = emb.transform(texts)
    else:
        # partial refit if vocab has grown significantly
        X = emb.transform(texts)

    # Normalise labels to [0, 1] for sigmoid output
    y_rel = np.clip(np.array(rels, dtype=np.float32) / 10.0, 0, 1)
    y_urg = np.clip(np.array(urgs, dtype=np.float32) / 10.0, 0, 1)

    X_t  = torch.from_numpy(X)
    yr_t = torch.from_numpy(y_rel)
    yu_t = torch.from_numpy(y_urg)

    dataset = TensorDataset(X_t, yr_t, yu_t)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = get_model(input_dim=X.shape[1])
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for xb, yr_b, yu_b in loader:
            optimizer.zero_grad()
            # Forward: scale sigmoid back
            rel_pred = model.relevance_head(model.trunk(xb)).squeeze(-1) * 10.0
            urg_pred = model.urgency_head(model.trunk(xb)).squeeze(-1) * 10.0
            # Loss on [0,10] scale
            loss = criterion(rel_pred, yr_b * 10.0) + criterion(urg_pred, yu_b * 10.0)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        if (epoch + 1) % 5 == 0:
            print(f"[ml:trainer] Epoch {epoch+1}/{EPOCHS} loss={avg:.4f}")

    model.eval()
    torch.save(model.state_dict(), CHECKPOINT)
    elapsed = time.time() - t0
    print(f"[ml:trainer] Done in {elapsed:.1f}s — checkpoint saved")
    return {"status": "ok", "n": len(texts), "epochs": EPOCHS,
            "final_loss": avg, "elapsed_s": elapsed}

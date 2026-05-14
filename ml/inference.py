"""
Fast inference engine — scores articles locally without any LLM call.

Returns per-article: relevance, urgency, and a `needs_llm` flag.
`needs_llm=True` means the ensemble is uncertain → route to Sonnet.

Uncertainty is measured via bootstrap ensemble spread (5 models).
The LLM is only consulted for articles in the grey zone or high variance
— typically <15% of total articles once the model is trained.

Active learning: every uncertain article gets appended to
``data/active_learning_queue.jsonl`` so we can audit which items the model
flags for priority LLM review.
"""
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from threading import Lock

import numpy as np

from ml.embedder import get_embedder
from ml.features import extract_features_batch, EXTRA_FEATURE_DIM
from ml.model import get_model
from ml.trainer import (
    UNCERTAINTY_REL, UNCERTAINTY_URG,
    LLM_ZONE_MID_LO, LLM_ZONE_MID_HI, LLM_ZONE_CLEAR_NOISE,
)


# Active learning queue — capped at this many lines to prevent unbounded growth.
ACTIVE_LEARN_PATH = Path(__file__).resolve().parent.parent / "data" / "active_learning_queue.jsonl"
ACTIVE_LEARN_MAX_LINES = 5000
_AL_LOCK = Lock()


@dataclass
class ArticleScore:
    relevance: float      # 0–10
    urgency: float        # 0–10
    rel_std: float        # ensemble spread in relevance (uncertainty)
    urg_std: float        # ensemble spread in urgency
    needs_llm: bool       # True → uncertain, send to Sonnet
    confident_noise: bool # True → clearly irrelevant, skip everything
    priority: float = 0.0  # active-learning priority: higher = more important to label
    # 0..1 — ML-predicted recency decay rate. 1.0 = decays fast (earnings, price
    # moves); 0.0 = timeless (macro thesis). Consumed by article_store's
    # briefing ranker to apply per-article decay instead of blanket decay.
    time_sensitivity: float = 0.5


def _log_active_learning(records: list[dict]) -> None:
    """Append uncertain articles to the active-learning queue. Best-effort."""
    if not records:
        return
    try:
        ACTIVE_LEARN_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _AL_LOCK:
            with ACTIVE_LEARN_PATH.open("a") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")
            # Trim if file has grown beyond the cap.
            try:
                with ACTIVE_LEARN_PATH.open("rb") as f:
                    lines = f.readlines()
                if len(lines) > ACTIVE_LEARN_MAX_LINES:
                    keep = lines[-ACTIVE_LEARN_MAX_LINES:]
                    with ACTIVE_LEARN_PATH.open("wb") as f:
                        f.writelines(keep)
            except Exception:
                pass
    except Exception:
        pass


def score_articles(articles: list[dict]) -> list[ArticleScore]:
    """
    Score a batch of articles using the local ensemble.
    Returns ArticleScore per article in input order.
    """
    _uncertain = [ArticleScore(0, 0, 99, 99, needs_llm=True, confident_noise=False,
                               priority=1.0, time_sensitivity=0.5)
                  for _ in articles]

    emb = get_embedder()
    model = get_model()

    if not emb.fitted or not model.fitted:
        return _uncertain

    texts = [f"{a.get('title', '')} {a.get('summary', '')}" for a in articles]

    try:
        X_text = emb.transform(texts)
    except Exception:
        return _uncertain

    X_extra = extract_features_batch(articles)
    X = np.concatenate([X_text, X_extra], axis=1).astype(np.float32)

    rel_mean, rel_std, urg_mean, urg_std, tsens_mean = model.predict(X)

    results = []
    al_records: list[dict] = []
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    for i in range(len(articles)):
        rel   = float(np.clip(rel_mean[i], 0, 10))
        urg   = float(np.clip(urg_mean[i], 0, 10))
        r_std = float(rel_std[i])
        u_std = float(urg_std[i])
        tsens = float(np.clip(tsens_mean[i], 0.0, 1.0))

        confident_noise = (rel < LLM_ZONE_CLEAR_NOISE and r_std < UNCERTAINTY_REL)
        in_grey   = LLM_ZONE_MID_LO <= urg <= LLM_ZONE_MID_HI
        uncertain = r_std > UNCERTAINTY_REL or u_std > UNCERTAINTY_URG
        needs_llm = (in_grey or uncertain) and not confident_noise

        # Active-learning priority: combines uncertainty with potential urgency.
        # Higher value → label this one first.
        priority = (r_std + u_std) * 0.5 + max(0.0, urg - 5.0) * 0.5

        results.append(ArticleScore(
            relevance=round(rel, 2),
            urgency=round(urg, 2),
            rel_std=round(r_std, 3),
            urg_std=round(u_std, 3),
            needs_llm=needs_llm,
            confident_noise=confident_noise,
            priority=round(priority, 3),
            time_sensitivity=round(tsens, 3),
        ))

        if needs_llm:
            art = articles[i]
            al_records.append({
                "ts": now_iso,
                "id": art.get("_id") or art.get("id") or "",
                "title": art.get("title", "")[:200],
                "source": art.get("source", ""),
                "url": art.get("link") or art.get("url", ""),
                "rel": round(rel, 2), "urg": round(urg, 2),
                "rel_std": round(r_std, 3), "urg_std": round(u_std, 3),
                "priority": round(priority, 3),
                "reason": "grey_zone" if in_grey else "high_variance",
            })

    if al_records:
        _log_active_learning(al_records)

    return results


def triage_articles(articles: list[dict]) -> dict:
    """
    Split a batch into three buckets:
      noise      → NN confident irrelevant, drop
      confident  → NN confident, apply score directly
      uncertain  → NN unsure, route to Sonnet — sorted by priority desc

    Returns dict with keys 'noise', 'confident', 'uncertain',
    each a list of (article, ArticleScore) tuples.
    """
    scores = score_articles(articles)
    noise, confident, uncertain = [], [], []

    for art, sc in zip(articles, scores):
        if sc.confident_noise:
            noise.append((art, sc))
        elif sc.needs_llm:
            uncertain.append((art, sc))
        else:
            confident.append((art, sc))

    # Prioritise uncertain articles by active-learning score (high uncertainty +
    # high potential urgency → label these first).
    uncertain.sort(key=lambda pair: pair[1].priority, reverse=True)

    llm_pct = 100 * len(uncertain) / max(len(articles), 1)
    print(f"[ml:inference] {len(articles)} articles → "
          f"noise={len(noise)} confident={len(confident)} "
          f"llm_needed={len(uncertain)} ({llm_pct:.0f}%)")

    return {"noise": noise, "confident": confident, "uncertain": uncertain}

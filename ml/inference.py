"""
Fast inference engine — scores articles locally without any LLM call.

Returns per-article: relevance, urgency, and a `needs_llm` flag.
`needs_llm=True` means the ensemble is uncertain → route to Sonnet.

Uncertainty is measured via bootstrap ensemble spread (5 models).
The LLM is only consulted for articles in the grey zone or high variance
— typically <15% of total articles once the model is trained.
"""
import numpy as np
from dataclasses import dataclass

from ml.embedder import get_embedder
from ml.model import get_model
from ml.trainer import (
    UNCERTAINTY_REL, UNCERTAINTY_URG,
    LLM_ZONE_MID_LO, LLM_ZONE_MID_HI, LLM_ZONE_CLEAR_NOISE,
)


@dataclass
class ArticleScore:
    relevance: float      # 0–10
    urgency: float        # 0–10
    rel_std: float        # ensemble spread in relevance (uncertainty)
    urg_std: float        # ensemble spread in urgency
    needs_llm: bool       # True → uncertain, send to Sonnet
    confident_noise: bool # True → clearly irrelevant, skip everything


def score_articles(articles: list[dict]) -> list[ArticleScore]:
    """
    Score a batch of articles using the local ensemble.
    Returns ArticleScore per article in input order.
    """
    _uncertain = [ArticleScore(0, 0, 99, 99, needs_llm=True, confident_noise=False)
                  for _ in articles]

    emb = get_embedder()
    model = get_model()

    if not emb.fitted or not model.fitted:
        return _uncertain

    texts = [f"{a.get('title', '')} {a.get('summary', '')}" for a in articles]

    try:
        X = emb.transform(texts)
    except Exception:
        return _uncertain

    rel_mean, rel_std, urg_mean, urg_std = model.predict(X)

    results = []
    for i in range(len(articles)):
        rel   = float(np.clip(rel_mean[i], 0, 10))
        urg   = float(np.clip(urg_mean[i], 0, 10))
        r_std = float(rel_std[i])
        u_std = float(urg_std[i])

        confident_noise = (rel < LLM_ZONE_CLEAR_NOISE and r_std < UNCERTAINTY_REL)
        in_grey   = LLM_ZONE_MID_LO <= urg <= LLM_ZONE_MID_HI
        uncertain = r_std > UNCERTAINTY_REL or u_std > UNCERTAINTY_URG
        needs_llm = (in_grey or uncertain) and not confident_noise

        results.append(ArticleScore(
            relevance=round(rel, 2),
            urgency=round(urg, 2),
            rel_std=round(r_std, 3),
            urg_std=round(u_std, 3),
            needs_llm=needs_llm,
            confident_noise=confident_noise,
        ))

    return results


def triage_articles(articles: list[dict]) -> dict:
    """
    Split a batch into three buckets:
      noise      → NN confident irrelevant, drop
      confident  → NN confident, apply score directly
      uncertain  → NN unsure, route to Sonnet

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

    llm_pct = 100 * len(uncertain) / max(len(articles), 1)
    print(f"[ml:inference] {len(articles)} articles → "
          f"noise={len(noise)} confident={len(confident)} "
          f"llm_needed={len(uncertain)} ({llm_pct:.0f}%)")

    return {"noise": noise, "confident": confident, "uncertain": uncertain}

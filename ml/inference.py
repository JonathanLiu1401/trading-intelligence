"""
Fast inference engine — scores articles locally without any LLM call.

Returns per-article: relevance, urgency, and a `needs_llm` flag.
`needs_llm=True` means the NN is uncertain → route to Sonnet for verification.

Uncertainty is measured via Monte Carlo Dropout (15 forward passes).
The LLM is only consulted for articles in the grey zone (score 7-8.5) or
where dropout variance is high — typically <15% of articles.
"""
import numpy as np
from dataclasses import dataclass

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

from ml.embedder import get_embedder
from ml.trainer import (
    get_model,
    UNCERTAINTY_REL, UNCERTAINTY_URG,
    LLM_ZONE_MID_LO, LLM_ZONE_MID_HI,
    LLM_ZONE_LOW, LLM_ZONE_HIGH,
)


@dataclass
class ArticleScore:
    relevance: float    # 0–10
    urgency: float      # 0–10
    rel_std: float      # uncertainty in relevance
    urg_std: float      # uncertainty in urgency
    needs_llm: bool     # True → uncertain, send to LLM
    confident_noise: bool  # True → clearly irrelevant, skip everything


def score_articles(articles: list[dict], mc_passes: int = 15) -> list[ArticleScore]:
    """
    Score a batch of articles using the local NN.

    articles: list of dicts with 'title' and optionally 'summary'.
    Returns a list of ArticleScore in the same order.
    """
    _uncertain = [ArticleScore(0, 0, 99, 99, needs_llm=True, confident_noise=False)
                  for _ in articles]

    if not _TORCH:
        return _uncertain

    emb = get_embedder()
    if not emb.fitted:
        return _uncertain

    texts = [f"{a.get('title', '')} {a.get('summary', '')}" for a in articles]

    try:
        X = emb.transform(texts)
    except Exception:
        return _uncertain

    model = get_model(input_dim=X.shape[1])
    x_t = torch.from_numpy(X)

    with torch.no_grad():
        rel_mean, rel_std, urg_mean, urg_std = model.predict_with_uncertainty(x_t, mc_passes)

    rel_mean = rel_mean.numpy()
    rel_std  = rel_std.numpy()
    urg_mean = urg_mean.numpy()
    urg_std  = urg_std.numpy()

    results = []
    for i in range(len(articles)):
        rel  = float(np.clip(rel_mean[i], 0, 10))
        urg  = float(np.clip(urg_mean[i], 0, 10))
        r_std = float(rel_std[i])
        u_std = float(urg_std[i])

        # Clearly irrelevant — high confidence low score
        confident_noise = (rel < LLM_ZONE_LOW and r_std < UNCERTAINTY_REL)

        # Grey zone: uncertain enough OR score is in the hard-to-call range
        in_grey = LLM_ZONE_MID_LO <= urg <= LLM_ZONE_MID_HI
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


def triage_articles(articles: list[dict], mc_passes: int = 15) -> dict:
    """
    Split a batch into three buckets:
      - 'noise':      NN confident they're irrelevant → drop
      - 'confident':  NN confident, no LLM needed → use NN score
      - 'uncertain':  NN unsure → route to Sonnet

    Returns dict with keys 'noise', 'confident', 'uncertain',
    each a list of (article, ArticleScore) tuples.
    Prints a summary line.
    """
    scores = score_articles(articles, mc_passes)
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

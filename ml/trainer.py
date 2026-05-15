"""
Online trainer — pulls labeled articles from the store, trains ArticleNet on GPU.

Phase 1: bootstraps on heuristic kw_score (weak labels, available immediately).
Phase 2: retrains as Sonnet ai_score labels accumulate (stronger labels).

Aggressive GPU training: 100 epochs per cycle, batch_size=256, Adam + cosine LR.
Schedule lives in daemon.py (RETRAIN_INTERVAL drives the cadence).
"""
import json
import os
import re
import threading
import time
from pathlib import Path

import numpy as np

from ml.embedder import get_embedder
from ml.features import extract_features_batch
from ml.model import get_model

# Per-cycle training metrics get appended here so the val_loss trend is visible
# without grepping the daemon log. Each line is a JSON dict — `tail -n 50 |
# jq .val_loss` shows whether the model is improving over time. Lives under the
# ml dir to share the same lifetime as the checkpoints.
_ML_DIR = Path(os.environ.get(
    "DIGITAL_INTERN_ML_DIR",
    Path(__file__).resolve().parent.parent / "data" / "ml",
))
TRAINING_METRICS_LOG = _ML_DIR / "training_metrics.jsonl"
_METRICS_LOCK = threading.Lock()


def _log_metrics(record: dict) -> None:
    """Append one training-cycle record to TRAINING_METRICS_LOG. Best-effort."""
    try:
        _ML_DIR.mkdir(parents=True, exist_ok=True)
        payload = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **record}
        with _METRICS_LOCK, TRAINING_METRICS_LOG.open("a") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass


# ── time_sensitivity bootstrap labels ─────────────────────────────────────────
# Heuristic 0..1 label fed to the time_sensitivity head. The model is expected
# to refine these from text features over time; this is only a warm start.
#
#   1.0 = highly time-sensitive (earnings beats, breaking price moves, "today")
#   0.5 = neutral / unknown
#   0.0 = timeless (long-term thesis, secular trends, technology analysis)

_HIGH_TIME_PATTERNS = re.compile(
    r"\b(?:today|now|breaking|just|moments?\s+ago|minutes?\s+ago|hours?\s+ago|"
    r"surges?|surged|surging|crashes?|crashed|crashing|plunges?|plunged|"
    r"soar(?:s|ed|ing)?|tumbl(?:e|es|ed|ing)|jumps?|jumped|plummet(?:s|ed|ing)?|"
    r"rallies|rallied|rally|spike[ds]?|nosediv(?:e|es|ed|ing)|halt(?:s|ed)?|"
    r"earnings\s+(?:beat|miss|topped|missed)|"
    r"beats?\s+(?:estimates?|expectations?)|misses?\s+(?:estimates?|expectations?)|"
    r"price\s+target\s+(?:raised|cut|lowered|increased|reiterated)|"
    r"upgrades?|downgrades?|guidance|pre[- ]?market|after[- ]?hours|"
    r"trading\s+(?:up|down|higher|lower)|Q[1-4]\s*\d{2,4}?|"
    r"\$\d+(?:\.\d+)?\s*(?:million|billion|m\b|b\b)?)\b",
    re.IGNORECASE,
)
# Percent moves like "+5%", "down 12%", "rose 3.2%" — strong recency signal.
_PCT_RE = re.compile(r"\b\d+(?:\.\d+)?\s*%")
# Dollar figures with magnitude — earnings/deals are time-sensitive.
_DOLLAR_MAG_RE = re.compile(
    r"\$\d+(?:\.\d+)?\s*(?:k|m|b|million|billion|thousand)\b", re.IGNORECASE
)

_LOW_TIME_PATTERNS = re.compile(
    r"\b(?:thesis|long[- ]?term|outlook|secular|trend(?:s|ing)?|technology|"
    r"analysis|deep\s+dive|why|how\s+(?:to|the|i)|primer|guide|explained|"
    r"explainer|fundamentals?|valuation\s+(?:case|model)|moat|tam|sam|"
    r"the\s+case\s+for|what\s+is|introduction\s+to|overview\s+of|history\s+of|"
    r"future\s+of|landscape|state\s+of\s+the)\b",
    re.IGNORECASE,
)


def _time_sensitivity_label(title: str, summary: str = "") -> float:
    """Heuristic 0..1 label for time-sensitivity. Bootstrap-only — model learns
    from text features once trained on these weak labels.

    Priority: high-signal recency tokens trump low-signal timeless tokens
    (a piece titled "Why NVDA surged today" is still time-sensitive)."""
    title_l = (title or "").strip()
    if not title_l:
        return 0.5

    high_hits = len(_HIGH_TIME_PATTERNS.findall(title_l))
    high_hits += len(_PCT_RE.findall(title_l))
    high_hits += len(_DOLLAR_MAG_RE.findall(title_l))
    low_hits  = len(_LOW_TIME_PATTERNS.findall(title_l))

    # Summary contributes at half weight — title is the strongest signal.
    if summary:
        sum_l = summary[:500]  # cap to keep label computation cheap
        high_hits += 0.5 * (
            len(_HIGH_TIME_PATTERNS.findall(sum_l))
            + len(_PCT_RE.findall(sum_l))
            + len(_DOLLAR_MAG_RE.findall(sum_l))
        )
        low_hits  += 0.5 * len(_LOW_TIME_PATTERNS.findall(sum_l))

    if high_hits >= 2:
        return 0.95
    if high_hits >= 1:
        return 0.85
    if low_hits >= 2:
        return 0.15
    if low_hits >= 1:
        return 0.25
    return 0.5


def _time_sensitivity_batch(articles: list[dict]) -> np.ndarray:
    """Compute heuristic time_sensitivity labels for a batch of articles."""
    return np.array(
        [_time_sensitivity_label(a.get("title", ""), a.get("summary", ""))
         for a in articles],
        dtype=np.float32,
    )

# Serialize all model.fit() calls. ml_trainer_worker and continuous_trainer_worker
# both mutate the same global ArticleNet on GPU; without this lock, overlapping
# fits raise "variable needed for gradient computation has been modified by an
# inplace operation" because autograd buffers from one pass get overwritten by
# the other before backward() finishes.
_TRAIN_LOCK = threading.Lock()

# Cadence — daemon.ML_TRAIN_INTERVAL is the source of truth (now 3 min).
RETRAIN_INTERVAL = 180      # retrain at most once every 3 minutes
MIN_NEW_LABELS   = 50       # retrain if this many new LLM labels since last train

# Training hyperparameters — aggressive GPU usage
# 100 epochs per cycle for the deep multi-task net (RTX 3060 trains in <5s).
EPOCHS_PER_CYCLE = 100
BATCH_SIZE       = 256

# Sample-weighting for label magnitude. Higher relevance scores train harder so
# strong-signal articles (9-10 / "200% profit") dominate gradient updates over
# borderline 5-6 noise. Convex exponent on (y_rel / 10):
#   exp=2 → score 10 = w 1.0, score 5 = w 0.25, score 2 = w 0.04 (pre-normalize).
# Weights are normalized to mean=1 inside ArticleNet.fit so the overall loss
# scale (and therefore optimal LR) stays roughly invariant to label distribution.
LABEL_WEIGHT_EXPONENT = 2.0
# Cold-start LR for a fresh model. ArticleNet.fit auto-drops to 1e-4 once the
# model is already fitted (warm fine-tune) — pumping LR back to 1e-3 every cycle
# was kicking weights off the basin and producing the "no upward trend" symptom.
LEARNING_RATE       = 1e-3
LEARNING_RATE_WARM  = 1e-4
MIN_TRAIN_SAMPLES = 50       # below this we bootstrap from kw_score weak labels

# Uncertainty thresholds — articles with uncertainty above this go to LLM
UNCERTAINTY_REL  = 1.5      # std in relevance score
UNCERTAINTY_URG  = 1.2      # std in urgency score
# Score zones where we still defer to LLM regardless of confidence
LLM_ZONE_MID_LO  = 7.0
LLM_ZONE_MID_HI  = 8.5
LLM_ZONE_CLEAR_NOISE  = 3.0  # below this → clearly noise, skip LLM


def _fetch_briefing_samples(
    store, limit: int = 100
) -> tuple[list[str], list[dict], list[float], list[float]]:
    """Pull articles mentioned in recent Opus briefings as high-quality positive labels.

    Articles whose title prefix appears in any recent briefing get a strong
    positive label (score=4.5). Briefings are generated by Opus 4.7 every 5h
    and serve as a curation signal independent of Sonnet scoring."""
    from storage.article_store import decompress

    texts: list[str] = []
    articles: list[dict] = []
    rels: list[float] = []
    urgs: list[float] = []
    try:
        briefings = store.get_briefings_for_training(limit=limit)
    except Exception:
        return texts, articles, rels, urgs
    if not briefings:
        return texts, articles, rels, urgs

    combined = " ||| ".join((b.get("text") or "").lower() for b in briefings)

    cur = store.conn.execute(
        "SELECT title, full_text, source, published FROM articles "
        "ORDER BY first_seen DESC LIMIT 5000"
    )
    for title, blob, src, published in cur.fetchall():
        # 12-char minimum matches heartbeat_worker._extract_briefing_labels —
        # short generic titles ("Stocks", "Markets") otherwise false-match
        # common briefing prose and inject noisy positive labels.
        if not title or len(title) < 12:
            continue
        if title[:40].lower() not in combined:
            continue
        summary = decompress(blob) if blob else ""
        texts.append(f"{title} {summary}")
        articles.append({
            "title": title, "summary": summary,
            "source": src or "", "published": published or "",
        })
        rels.append(4.5)
        urgs.append(0.0)
    return texts, articles, rels, urgs


def _fetch_training_data(
    store,
) -> tuple[list[str], list[dict], np.ndarray, np.ndarray, str]:
    """
    Pull articles with scores from the store.
    Returns (texts, articles, y_rel, y_urg, source). ``articles`` is a list of
    light dicts (title/summary/source/published) parallel to ``texts``, used by
    ml/features.py to produce the extra numeric feature columns.
    ``source`` ∈ {"strong","weak","mixed","briefing_only"}.

    Only LLM-sourced labels are pulled as "strong" — we explicitly exclude rows
    whose score_source is 'ml' (model self-predictions) to prevent the trainer
    from learning its own outputs. Rows predating the score_source migration are
    distinguished by ai_score being integer-valued (Sonnet returns int score;
    recursive_labeler does int*2.0 → still integer; briefing_boost legacy rows
    are at most a few hundred and have negligible noise impact).

    Synthetic backtest / opus-annotation rows are intentionally included (see
    CLAUDE.md §5 — training reads include backtest). They carry score_source
    NULL and may have fractional ai_score (SELL=0.5, opus NEUTRAL=2.5), so the
    integer heuristic alone would wrongly drop them — the explicit synthetic
    clause keeps them in the pool.
    """
    from storage.article_store import decompress

    texts: list[str] = []
    articles: list[dict] = []
    rels, urgs = [], []
    source = "strong"

    # Strong labels: ground-truth LLM-sourced ai_score, plus synthetic
    # backtest/opus rows (legitimate fractional labels, score_source NULL).
    cur = store.conn.execute(
        "SELECT title, full_text, ai_score, source, published "
        "FROM articles "
        "WHERE ai_score > 0 "
        "  AND (score_source IN ('llm','briefing_boost') "
        "       OR (score_source IS NULL AND ai_score = CAST(ai_score AS INTEGER)) "
        "       OR (score_source IS NULL AND (url LIKE 'backtest://%' "
        "            OR source LIKE 'backtest_%' OR source LIKE 'opus_annotation%'))) "
        "ORDER BY first_seen DESC LIMIT 15000"
    )
    for title, blob, ai, src, published in cur.fetchall():
        summary = decompress(blob) if blob else ""
        texts.append(f"{title} {summary}")
        articles.append({
            "title": title or "", "summary": summary,
            "source": src or "", "published": published or "",
        })
        rels.append(float(ai))
        urgs.append(float(ai) if ai >= 8.0 else 0.0)

    n_strong = len(texts)

    # Always mix in kw_score-labeled articles that lack an LLM label. The LLM
    # corpus is heavily clustered (most rows score 3-5); kw_score has much
    # higher variance (std≈3.3) and provides the signal the model needs to
    # learn discriminative features. We previously only mixed when strong
    # labels were < MIN_TRAIN_SAMPLES, which meant the trainer saw a near-flat
    # label distribution once Sonnet had labeled enough rows. Cap the kw pool
    # so it doesn't overwhelm the high-quality LLM signal.
    KW_MAX_FRACTION = 0.5
    kw_cap = max(2000, int(n_strong * KW_MAX_FRACTION))
    cur2 = store.conn.execute(
        "SELECT title, full_text, kw_score, source, published "
        "FROM articles WHERE ai_score = 0 AND kw_score > 0 "
        "  AND (score_source IS NULL OR score_source = 'ml') "
        "ORDER BY kw_score DESC LIMIT ?",
        (kw_cap,),
    )
    n_kw_added = 0
    for title, blob, kw, src, published in cur2.fetchall():
        summary = decompress(blob) if blob else ""
        texts.append(f"{title} {summary}")
        articles.append({
            "title": title or "", "summary": summary,
            "source": src or "", "published": published or "",
        })
        rels.append(float(kw))
        urgs.append(0.0)
        n_kw_added += 1
    if n_strong == 0 and n_kw_added > 0:
        source = "weak"
    elif n_kw_added > 0:
        source = "mixed"

    # Opus briefing-derived positive labels — high signal quality.
    b_texts, b_articles, b_rels, b_urgs = _fetch_briefing_samples(store)
    if b_texts:
        texts.extend(b_texts)
        articles.extend(b_articles)
        rels.extend(b_rels)
        urgs.extend(b_urgs)
        if n_strong == 0 and source == "strong":
            source = "briefing_only"

    y_rel = np.clip(np.array(rels, dtype=np.float32), 0, 10)
    y_urg = np.clip(np.array(urgs, dtype=np.float32), 0, 10)
    return texts, articles, y_rel, y_urg, source


def train(store, force: bool = False) -> dict:
    """Train/retrain ArticleNet on available labeled data. Returns metrics dict."""
    t0 = time.time()
    texts, articles, y_rel, y_urg, source = _fetch_training_data(store)

    if len(texts) < 30:
        return {"status": "skipped", "reason": "too_few_samples", "n": len(texts)}

    # Label-variance sanity check — if std is near zero, the model literally
    # cannot beat predicting the mean. Surface this in the log so flat val_loss
    # gets attributed to the right cause (label collapse, not training bug).
    rel_std = float(np.std(y_rel)) if len(y_rel) else 0.0
    rel_mean = float(np.mean(y_rel)) if len(y_rel) else 0.0
    mse_baseline = float(np.var(y_rel)) if len(y_rel) else 0.0
    print(f"[ml:trainer] Training on {len(texts)} samples "
          f"({int((y_rel > 0).sum())} labeled, {int((y_urg >= 8).sum())} urgent) "
          f"source={source}")
    print(f"[ml:trainer] label stats: y_rel mean={rel_mean:.3f} std={rel_std:.3f} "
          f"mse_floor(predict-mean)={mse_baseline:.3f}")
    if rel_std < 0.5:
        print("[ml:trainer] WARNING: label std < 0.5 — model has almost no "
              "signal to learn; expect flat val_loss until labels diversify.")

    # Preview the convex sample-weights (w = (y_rel / 10) ** EXP, normalized to
    # mean=1 inside fit). Logging here makes label-imbalance visible per cycle.
    _w_preview = np.power(np.clip(y_rel, 0, 10) / 10.0, LABEL_WEIGHT_EXPONENT)
    _w_mean = float(_w_preview.mean()) if len(_w_preview) else 0.0
    if _w_mean > 0:
        _w_preview = _w_preview / _w_mean
    print(f"[ml:trainer] sample weights (exp={LABEL_WEIGHT_EXPONENT}): "
          f"min={float(_w_preview.min()):.3f} "
          f"max={float(_w_preview.max()):.3f} "
          f"mean={float(_w_preview.mean()):.3f} "
          f"(higher-scoring articles train harder)")

    emb = get_embedder()
    if emb.should_refit(len(texts)):
        X_text = emb.fit_transform(texts)
    else:
        X_text = emb.transform(texts)

    X_extra = extract_features_batch(articles)
    X = np.concatenate([X_text, X_extra], axis=1).astype(np.float32)

    # Bootstrap time_sensitivity labels from title/summary heuristics. The model
    # will refine these from the underlying text features after a few cycles.
    y_time = _time_sensitivity_batch(articles)

    model = get_model()
    with _TRAIN_LOCK:
        metrics = model.fit(
            X, y_rel, y_urg, y_time=y_time,
            epochs=EPOCHS_PER_CYCLE,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            label_weight_exponent=LABEL_WEIGHT_EXPONENT,
        )

    elapsed = time.time() - t0
    result = {
        "status": "ok",
        "n": len(texts),
        "n_llm_labeled": int((y_rel > 0).sum()),
        "n_urgent": int((y_urg >= 8).sum()),
        "label_source": source,
        "final_loss": metrics.get("final_loss"),
        "val_loss": metrics.get("val_loss"),
        "best_in_run": metrics.get("best_in_run"),
        "new_best": metrics.get("new_best", False),
        "epochs": metrics.get("epochs"),
        "device": metrics.get("device"),
        "elapsed_s": round(elapsed, 1),
    }
    _log_metrics({"phase": "train", **result})
    return result


class Trainer:
    """Thin class wrapper around ``train()`` for ad-hoc and scripted use.

    Opens its own ``ArticleStore`` so callers can drive a one-off training
    cycle without instantiating the daemon. Safe to call while the daemon
    is also running — SQLite WAL mode permits concurrent readers, and
    ArticleNet/Embedder use process-local singletons (a second Python
    interpreter gets its own model state)."""

    def __init__(self):
        from storage.article_store import ArticleStore
        self.store = ArticleStore()

    def run_once(self, force: bool = False) -> dict:
        return train(self.store, force=force)

    def run_continuous_once(self) -> dict:
        return train_continuous(self.store)

    def health(self) -> dict:
        """Snapshot the current ML stack — useful when the user wants to
        confirm the model has actually learned text features."""
        emb = get_embedder()
        model = get_model()
        try:
            vocab = len(emb._vec.vocabulary_) if emb._vec is not None else 0
        except Exception:
            vocab = 0
        return {
            "embedder_fitted": emb.fitted,
            "embedder_vocab_size": vocab,
            "embedder_healthy": not emb.should_refit(self.store.conn.execute(
                "SELECT COUNT(*) FROM articles WHERE ai_score > 0"
            ).fetchone()[0]),
            "model_fitted": model.fitted,
            "model_input_dim": getattr(model, "_input_dim", None),
            "model_best_val_loss": getattr(model, "_best_val_loss", None),
        }


_last_continuous_loss = float('inf')


def train_continuous(store) -> dict:
    """Lightweight continuous GPU training pass — 40 epochs on all scored articles."""
    global _last_continuous_loss
    from storage.article_store import decompress

    t0 = time.time()

    texts: list[str] = []
    articles: list[dict] = []
    rels, urgs = [], []
    # Mirror _fetch_training_data: LLM-sourced labels, legacy integer-valued
    # ai_score from before the score_source split, and synthetic backtest/opus
    # rows (fractional labels, score_source NULL). Never ingest
    # score_source='ml' — that would reopen the label-feedback loop.
    cur = store.conn.execute(
        "SELECT title, full_text, ai_score, source, published "
        "FROM articles "
        "WHERE ai_score > 0 "
        "  AND (score_source IN ('llm','briefing_boost') "
        "       OR (score_source IS NULL AND ai_score = CAST(ai_score AS INTEGER)) "
        "       OR (score_source IS NULL AND (url LIKE 'backtest://%' "
        "            OR source LIKE 'backtest_%' OR source LIKE 'opus_annotation%')))"
    )
    for title, blob, ai, src, published in cur.fetchall():
        summary = decompress(blob) if blob else ""
        texts.append(f"{title} {summary}")
        articles.append({
            "title": title or "", "summary": summary,
            "source": src or "", "published": published or "",
        })
        rels.append(float(ai))
        urgs.append(float(ai) if ai >= 8.0 else 0.0)

    n = len(texts)
    if n < 30:
        return {"status": "skipped", "reason": "too_few_samples", "n": n}

    y_rel = np.clip(np.array(rels, dtype=np.float32), 0, 10)
    y_urg = np.clip(np.array(urgs, dtype=np.float32), 0, 10)

    emb = get_embedder()
    if emb.should_refit(len(texts)):
        X_text = emb.fit_transform(texts)
    else:
        X_text = emb.transform(texts)
    X_extra = extract_features_batch(articles)
    X = np.concatenate([X_text, X_extra], axis=1).astype(np.float32)

    # Bootstrap time_sensitivity labels alongside relevance/urgency targets.
    y_time = _time_sensitivity_batch(articles)

    model = get_model()
    # Skip rather than block: if the heavy trainer is mid-fit, just wait for the
    # next 60s tick. Avoids piling up continuous passes behind a long retrain.
    if not _TRAIN_LOCK.acquire(blocking=False):
        return {"status": "skipped", "reason": "trainer_busy", "n": n}
    try:
        # Continuous trainer is always a fine-tune (the heavy trainer warm-started
        # the model on cycle 1). Use a small LR so we don't undo progress.
        metrics = model.fit(X, y_rel, y_urg, y_time=y_time,
                            epochs=40, batch_size=512, lr=LEARNING_RATE_WARM,
                            warm=True,
                            label_weight_exponent=LABEL_WEIGHT_EXPONENT)
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
    result = {
        "status": "ok",
        "n": n,
        "loss": final_loss,
        "val_loss": metrics.get("val_loss"),
        "best_in_run": metrics.get("best_in_run"),
        "new_best": metrics.get("new_best", False),
        "gpu_mem_mb": gpu_mem_mb,
        "elapsed_s": elapsed,
    }
    _log_metrics({"phase": "continuous", **result})
    return result

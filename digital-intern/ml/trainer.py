"""
Online trainer — pulls labeled articles from the store, trains ArticleNet on GPU.

Phase 1: bootstraps on heuristic kw_score (weak labels, available immediately).
Phase 2: retrains as Sonnet ai_score labels accumulate (stronger labels).

Aggressive GPU training: 100 epochs per cycle, batch_size=256, Adam + cosine LR.
Schedule lives in daemon.py (ML_TRAIN_INTERVAL drives the cadence).
"""
import json
import multiprocessing
import os
import queue
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

# SSOT for the trainer's "strong label" row predicate (the WHERE body, no
# leading "WHERE"). Both _fetch_training_data and train_continuous select on
# this exact clause, and ml/label_audit.py imports it so the training-pool
# integrity audit can never drift from what the model actually trains on.
# Semantics (see _fetch_training_data docstring): ground-truth LLM-sourced
# ai_score, plus legacy pre-migration integer ai_score (score_source NULL),
# plus synthetic backtest/opus rows (score_source NULL, CLAUDE.md §5).
# score_source='ml' (the model's own predictions) is deliberately excluded —
# ingesting it would reopen the label-feedback loop.
STRONG_LABEL_WHERE = (
    "ai_score > 0 "
    "AND (score_source IN ('llm','briefing_boost') "
    "     OR (score_source IS NULL AND ai_score = CAST(ai_score AS INTEGER)) "
    "     OR (score_source IS NULL AND (url LIKE 'backtest://%' "
    "          OR source LIKE 'backtest_%' OR source LIKE 'opus_annotation%')))"
)


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
        f"WHERE {STRONG_LABEL_WHERE} "
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


_DATASET_CACHE = _ML_DIR / "dataset_cache.npz"
_DATASET_META  = _ML_DIR / "dataset_cache_meta.json"
# Rebuild feature cache when labeled count drifts >5% from the cached value.
_CACHE_DRIFT_THRESHOLD = 0.05


def _load_dataset_cache(n_labeled: int) -> dict | None:
    """Return cached {X, y_rel, y_urg, y_time, source, n} if still fresh, else None."""
    if not _DATASET_CACHE.exists() or not _DATASET_META.exists():
        return None
    try:
        meta = json.loads(_DATASET_META.read_text())
        cached_n = meta.get("n_labeled", 0)
        drift = abs(n_labeled - cached_n) / max(cached_n, 1)
        if drift > _CACHE_DRIFT_THRESHOLD:
            return None  # dataset has grown/shrunk enough to warrant a rebuild
        arrays = np.load(str(_DATASET_CACHE), allow_pickle=False)
        return {
            "X": arrays["X"],
            "y_rel": arrays["y_rel"],
            "y_urg": arrays["y_urg"],
            "y_time": arrays["y_time"],
            "source": meta.get("source", "cached"),
            "n": int(meta.get("n", 0)),
        }
    except Exception as e:
        print(f"[ml:trainer] cache load failed ({e}), rebuilding")
        return None


def _save_dataset_cache(X: np.ndarray, y_rel: np.ndarray, y_urg: np.ndarray,
                         y_time: np.ndarray, source: str, n_labeled: int, n: int) -> None:
    """Persist feature matrix and labels to disk so the next training cycle
    avoids re-decompressing 20k articles and re-running TF-IDF."""
    _ML_DIR.mkdir(parents=True, exist_ok=True)
    try:
        np.savez_compressed(
            str(_DATASET_CACHE),
            X=X.astype(np.float32),
            y_rel=y_rel.astype(np.float32),
            y_urg=y_urg.astype(np.float32),
            y_time=y_time.astype(np.float32),
        )
        _DATASET_META.write_text(json.dumps({
            "n_labeled": n_labeled,
            "source": source,
            "n": n,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }))
        size_mb = _DATASET_CACHE.stat().st_size / 1e6
        print(f"[ml:trainer] dataset cached to disk ({size_mb:.1f} MB, n={n})")
    except Exception as e:
        print(f"[ml:trainer] cache save failed: {e}")


def _train_impl(store, force: bool = False) -> dict:
    """Full training cycle (original body of ``train()``): fetch labeled data,
    build/refit the TF-IDF + extra-feature matrix, fit ArticleNet, persist
    checkpoints + tfidf pickle to disk, return a metrics dict.

    This now runs inside a spawned child process (see ``train()``) so its large
    transient allocations — the dense ~1.35GB float32 feature matrix, the
    pin_memory DataLoader pages, and the CUDA context — are returned to the OS
    when the child exits. glibc never releases those large arenas back within a
    long-lived daemon, which is what was driving the 9-21GB creep + OOM-kill.

    Serialization against ``train_continuous`` is enforced by the *parent*
    holding ``_TRAIN_LOCK`` for the whole subprocess lifetime — acquiring the
    lock here would be a useless process-local copy in the child."""
    t0 = time.time()

    # Count current labeled articles to decide whether to use disk cache.
    try:
        n_labeled = store.conn.execute(
            "SELECT COUNT(*) FROM articles WHERE ai_score > 0 "
            "AND score_source IN ('llm','briefing_boost')"
        ).fetchone()[0]
    except Exception:
        n_labeled = 0

    cached = None if force else _load_dataset_cache(n_labeled)
    if cached:
        X      = cached["X"]
        y_rel  = cached["y_rel"]
        y_urg  = cached["y_urg"]
        y_time = cached["y_time"]
        source = cached["source"]
        n      = cached["n"]
        print(f"[ml:trainer] loaded dataset from disk cache (n={n}, n_labeled≈{n_labeled})")
    else:
        texts, articles, y_rel, y_urg, source = _fetch_training_data(store)
        n = len(texts)

        if n < 30:
            return {"status": "skipped", "reason": "too_few_samples", "n": n}

        emb = get_embedder()
        if emb.should_refit(n):
            X_text = emb.fit_transform(texts)
        else:
            X_text = emb.transform(texts)

        X_extra = extract_features_batch(articles)
        X = np.concatenate([X_text, X_extra], axis=1).astype(np.float32)
        y_time = _time_sensitivity_batch(articles)

        # Persist to disk immediately — free the raw text lists from RAM.
        _save_dataset_cache(X, y_rel, y_urg, y_time, source, n_labeled, n)
        del texts, articles, X_text, X_extra  # release before GPU training

    if len(y_rel) < 30:
        return {"status": "skipped", "reason": "too_few_samples", "n": len(y_rel)}

    # Label-variance sanity check — if std is near zero, the model literally
    # cannot beat predicting the mean. Surface this in the log so flat val_loss
    # gets attributed to the right cause (label collapse, not training bug).
    rel_std = float(np.std(y_rel)) if len(y_rel) else 0.0
    rel_mean = float(np.mean(y_rel)) if len(y_rel) else 0.0
    mse_baseline = float(np.var(y_rel)) if len(y_rel) else 0.0
    print(f"[ml:trainer] Training on {n} samples "
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

    # X / y_rel / y_urg / y_time were produced above by exactly one of the two
    # branches (disk cache hit, or fresh _fetch_training_data + embed + cache
    # write). The fresh branch ``del``s ``texts``/``articles`` to free RAM
    # before GPU training, and the cache branch never builds them at all — so
    # re-embedding here (the pre-cache code path) raised NameError every cycle
    # and ArticleNet silently never retrained. Train directly on the prepared
    # matrices.
    # No _TRAIN_LOCK here: in the production path this runs in a child process
    # (its own single training thread), and the parent holds _TRAIN_LOCK around
    # the subprocess so train_continuous in the daemon still serializes. In the
    # in-process test/stub fallback, train() holds _TRAIN_LOCK around this call.
    model = get_model()
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
        "n": n,
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


def _run_training_child(db_path: str, force: bool, result_queue) -> None:
    """Child-process entrypoint. Spawned (not forked) so it does NOT inherit the
    parent daemon's CUDA context. Opens its own ArticleStore, runs the full
    training cycle, then exits — at which point the OS reclaims every page the
    cycle allocated. Puts a metrics dict on ``result_queue`` for the parent."""
    try:
        import sys
        # spawn re-execs a bare interpreter; multiprocessing normally forwards
        # the parent's sys.path, but make the repo importable defensively so a
        # stripped-down launch environment can't break the child's imports.
        repo_root = str(Path(__file__).resolve().parent.parent)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        from storage.article_store import ArticleStore
        # ArticleStore() takes no args and resolves the canonical local DB
        # itself via _get_db_path(); db_path is passed for logging/traceability
        # only (there is no constructor path argument or self.db_path attr).
        print(f"[ml:trainer:child] pid={os.getpid()} opening store (db={db_path})")
        store = ArticleStore()

        metrics = _train_impl(store, force=force)

        try:
            result_queue.put(metrics)
        finally:
            # Best-effort: drop CUDA caching-allocator blocks before the
            # interpreter tears down (the process exit frees them anyway).
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        print(f"[ml:trainer:child] pid={os.getpid()} done status={metrics.get('status')}")
    except Exception as e:
        import traceback
        try:
            result_queue.put({
                "status": "error",
                "reason": f"child_exception: {e}",
                "traceback": traceback.format_exc(),
            })
        except Exception:
            pass


# Production always isolates training in a spawned subprocess. The in-process
# fallback below exists only so the TestTrainOrchestration regression guard
# (which monkeypatches get_model/get_embedder with stubs a spawned interpreter
# would never see) keeps exercising the real cache/embed/fit orchestration.
import ml.embedder as _ml_embedder_mod  # noqa: E402
import ml.model as _ml_model_mod        # noqa: E402

_TRAIN_TIMEOUT_S = 600


def _collaborators_stubbed() -> bool:
    """True when get_model/get_embedder have been swapped out (test doubles).
    Production never patches these, so this is False in the daemon."""
    return (get_model is not _ml_model_mod.get_model
            or get_embedder is not _ml_embedder_mod.get_embedder)


def train(store, force: bool = False) -> dict:
    """Train/retrain ArticleNet on available labeled data. Returns metrics dict.

    Training runs in a spawned subprocess so the ~1.35GB dense feature matrix,
    the pin_memory DataLoader pages and the CUDA context are fully released to
    the OS when the child exits. The long-lived daemon was accumulating 9-21GB
    of unreclaimed glibc heap per cycle and getting OOM-killed (then restarting
    and repeating); an isolated process is the standard fix for large transient
    allocations inside a persistent service. Signature is unchanged."""
    if _collaborators_stubbed():
        # Test/stub path: a spawned interpreter would re-import a clean
        # ml.trainer and never see the monkeypatched stubs (and would hit the
        # real on-disk DB instead of the test fixture). Run in-process,
        # holding _TRAIN_LOCK to preserve the original serialization semantics.
        with _TRAIN_LOCK:
            return _train_impl(store, force=force)

    try:
        from storage.article_store import _get_db_path
        db_path = str(_get_db_path())
    except Exception:
        db_path = "<unknown>"

    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_run_training_child,
        args=(db_path, force, result_queue),
        name="ml-train-child",
    )

    t0 = time.time()
    # Hold _TRAIN_LOCK for the entire subprocess lifetime. fit() no longer runs
    # in this process, so this lock is now process-local — without holding it
    # here, train_continuous (which does _TRAIN_LOCK.acquire(blocking=False) in
    # this same daemon process) would fine-tune the in-memory net concurrently
    # with the child and both would race on the shared model_gpu.pt /
    # best_model.pt checkpoint writes.
    with _TRAIN_LOCK:
        print(f"[ml:trainer] starting training subprocess "
              f"(db={db_path}, force={force}, timeout={_TRAIN_TIMEOUT_S}s)")
        proc.start()

        metrics = None
        try:
            # Drain the queue BEFORE join — join-first can deadlock if the
            # payload ever exceeds the OS pipe buffer (tiny today, cheap
            # insurance). This get() is itself the bounded wait for the child.
            metrics = result_queue.get(timeout=_TRAIN_TIMEOUT_S)
        except queue.Empty:
            metrics = None

        # The child queues its result then exits almost immediately; give it a
        # short grace period to tear down before deciding it's hung.
        proc.join(timeout=30)

        if proc.is_alive():
            print("[ml:trainer] subprocess still alive after result/timeout — "
                  "terminating")
            proc.terminate()
            proc.join(timeout=10)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)
            elapsed = round(time.time() - t0, 1)
            print(f"[ml:trainer] training subprocess timed out after {elapsed}s")
            return {"status": "error", "reason": "subprocess_timeout",
                    "elapsed_s": elapsed}

        elapsed = round(time.time() - t0, 1)

        if metrics is None:
            print(f"[ml:trainer] subprocess produced no result "
                  f"(exitcode={proc.exitcode}) after {elapsed}s")
            return {"status": "error", "reason": "no_result",
                    "exitcode": proc.exitcode, "elapsed_s": elapsed}

        if metrics.get("status") == "ok":
            # The child wrote fresh model checkpoints + tfidf pickle to disk.
            # get_model()/get_embedder() are cache-once singletons with no
            # reload, and fit() no longer mutates THIS process's net — so drop
            # the daemon's cached singletons. The next get_model()/
            # get_embedder() (scorer, continuous trainer) then reconstructs
            # from the freshly-trained on-disk artifacts. Without this the
            # daemon would score with the stale startup model forever.
            try:
                _ml_model_mod._net = None
                _ml_embedder_mod._embedder = None
                print("[ml:trainer] reset in-memory model/embedder singletons "
                      "→ next inference reloads fresh checkpoint")
            except Exception as e:
                print(f"[ml:trainer] singleton reset failed (non-fatal): {e}")

        print(f"[ml:trainer] training subprocess finished in {elapsed}s "
              f"status={metrics.get('status')} "
              f"(memory released back to OS on child exit)")
        return metrics


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


def train_continuous(store, heartbeat=None) -> dict:
    """Lightweight continuous GPU training pass — 40 epochs on all scored articles.

    ``heartbeat``: optional zero-arg callable. It is invoked once after the
    (potentially slow) data-load phase and once per training epoch so the
    continuous_trainer worker can prove liveness during a pass that legitimately
    runs longer than the supervisor's staleness deadline. Exceptions raised by
    the callback are swallowed — a monitoring ping must never break training.
    """
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
        f"WHERE {STRONG_LABEL_WHERE} "
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

    del texts, articles, X_text, X_extra  # release before GPU training

    model = get_model()
    # Data-load phase (SELECT 15k + decompress + TF-IDF) is now done; ping
    # before the fit so the worst-case ping gap is max(load, per-epoch) rather
    # than load + first-epoch.
    if heartbeat is not None:
        try:
            heartbeat()
        except Exception:
            pass
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
                            label_weight_exponent=LABEL_WEIGHT_EXPONENT,
                            heartbeat=heartbeat)
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

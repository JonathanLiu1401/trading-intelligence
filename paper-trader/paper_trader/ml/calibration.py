"""Scorer calibration diagnostic — does a high predicted 5d return actually
precede a high realized one?

This is a **read-only diagnostic**. It never trains, never touches
`decision_scorer.pkl`, `decision_outcomes.jsonl`, `build_features`,
`N_FEATURES`, or any trade path — so it cannot perturb the unattended
continuous loop or break pickle compatibility (AGENTS.md "When to bump
model versions" / "Common pitfalls").

A quant researcher's first question about any scorer is *"is it
calibrated, or just confident?"* A model can rank outcomes correctly
(useful) while being systematically over-confident in the tails
(dangerous if the magnitude is trusted). `calibration_report` separates
those two failure modes:

- **rank skill** — Spearman(pred, realized). Does ordering by prediction
  order the realized outcomes?
- **magnitude bias** — per-decile ``mean_pred`` vs ``mean_realized``. A
  monotone curve that sits far off the 45° line is rank-skilled but
  magnitude-biased (the real DecisionScorer is exactly this: tails
  over-predict by ~30%).

Verdict (crisp, threshold-driven so it is exactly testable):

| Verdict | Meaning |
|---------|---------|
| `INSUFFICIENT_DATA` | < ``MIN_PAIRS`` finite (pred, realized) pairs |
| `MISCALIBRATED` | no rank skill (spearman < ``SPEARMAN_MIN``) or the decile curve is not mostly monotone |
| `DIRECTIONAL_BUT_BIASED` | rank-skilled + mostly monotone, but mean abs decile error > ``BIAS_TOL_PCT`` (ordering is trustworthy, the *number* is not) |
| `WELL_CALIBRATED` | strong rank skill, monotone, and the decile curve tracks the 45° line within ``BIAS_TOL_PCT`` |
| `WEAK_SIGNAL` | some rank skill but below the strong bar, magnitude OK |
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Thresholds are module-level so tests assert exact verdicts and a tuning
# change is a single, reviewable edit (mirrors the codebase's
# constants-at-module-scope convention, e.g. PRED_CLAMP_PCT).
MIN_PAIRS = 30           # need ≥3 samples/decile at the default 10 buckets
SPEARMAN_MIN = 0.10      # below this there is essentially no rank skill
SPEARMAN_GOOD = 0.30     # "strong" rank skill bar for WELL_CALIBRATED
MONOTONE_MIN = 0.60      # ≥60% of adjacent decile steps must be non-decreasing
MONOTONE_GOOD = 0.80     # stricter monotonicity bar for WELL_CALIBRATED
BIAS_TOL_PCT = 3.0       # mean |decile mean_pred − mean_realized| tolerance (pp)


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (tie-aware), scipy-free. Tied values share the mean of
    the ranks they span. This is load-bearing here, not a nicety: the real
    DecisionScorer clamps to ±PRED_CLAMP_PCT, so a batch of off-distribution
    predictions ties at exactly ±50. Plain ``argsort(argsort)`` assigns those
    ties distinct ordinal ranks by input order, fabricating rank skill — a
    constant predictor would score spearman 1.0 against any target."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    sx = x[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sx[j] == sx[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation via tie-aware average ranks. 0.0 when
    either side has zero variance (a constant predictor has *no* rank
    skill — that must read as 0.0, never NaN and never a tie-ordering
    artifact)."""
    if len(a) < 2:
        return 0.0
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0
    ar = _rankdata(a)
    br = _rankdata(b)
    if ar.std() == 0.0 or br.std() == 0.0:
        return 0.0
    return float(np.corrcoef(ar, br)[0, 1])


def calibration_report(pairs, n_buckets: int = 10) -> dict:
    """Bucket (predicted, realized) pairs into ``n_buckets`` predicted-value
    quantiles and report rank skill + per-bucket magnitude bias.

    ``pairs`` is any iterable of ``(predicted, realized)`` 2-tuples. Non-finite
    entries are dropped (a single inf/nan must not poison the report — same
    hardening class as ``_to_float`` in decision_scorer.py).

    Returns a JSON-safe dict:
    ``{status, verdict, n, spearman, pearson, monotone_fraction,
       mean_abs_decile_error, buckets:[{idx,n,mean_pred,mean_realized,
       pred_lo,pred_hi}], hint}``.
    """
    clean = []
    for p, y in pairs:
        try:
            pf = float(p)
            yf = float(y)
        except (TypeError, ValueError):
            continue
        if np.isfinite(pf) and np.isfinite(yf):
            clean.append((pf, yf))

    n = len(clean)
    if n < MIN_PAIRS:
        return {
            "status": "insufficient_data",
            "verdict": "INSUFFICIENT_DATA",
            "n": n,
            "spearman": None,
            "pearson": None,
            "monotone_fraction": None,
            "mean_abs_decile_error": None,
            "buckets": [],
            "hint": f"need ≥{MIN_PAIRS} finite pairs, have {n}",
        }

    clean.sort(key=lambda t: t[0])           # sort by predicted, ascending
    P = np.array([c[0] for c in clean], dtype=np.float64)
    Y = np.array([c[1] for c in clean], dtype=np.float64)

    k = max(2, min(n_buckets, n // 3))       # never fewer than 3 per bucket
    buckets = []
    realized_means = []
    abs_errs = []
    for i in range(k):
        lo = i * n // k
        hi = (i + 1) * n // k
        if hi <= lo:
            continue
        seg_p = P[lo:hi]
        seg_y = Y[lo:hi]
        mp = float(seg_p.mean())
        my = float(seg_y.mean())
        realized_means.append(my)
        abs_errs.append(abs(mp - my))
        buckets.append({
            "idx": i + 1,
            "n": int(hi - lo),
            "mean_pred": round(mp, 4),
            "mean_realized": round(my, 4),
            "pred_lo": round(float(seg_p.min()), 4),
            "pred_hi": round(float(seg_p.max()), 4),
        })

    # Rank skill over ALL pairs (not bucket means — bucket means hide
    # within-bucket noise and inflate the correlation).
    spearman = _spearman(P, Y)
    pearson = (0.0 if P.std() == 0.0 or Y.std() == 0.0
               else float(np.corrcoef(P, Y)[0, 1]))

    # Monotonicity of the decile realized curve: fraction of adjacent steps
    # that do not go DOWN. A perfectly calibrated scorer's decile
    # mean_realized rises with the decile.
    steps = len(realized_means) - 1
    if steps <= 0:
        monotone_fraction = 1.0
    else:
        nondec = sum(1 for j in range(steps)
                     if realized_means[j + 1] >= realized_means[j])
        monotone_fraction = nondec / steps

    mean_abs_decile_error = float(np.mean(abs_errs)) if abs_errs else 0.0

    if spearman < SPEARMAN_MIN or monotone_fraction < MONOTONE_MIN:
        verdict = "MISCALIBRATED"
        hint = ("predicted order does not track realized outcomes — the "
                "scorer is noise on this sample; do not size on it")
    elif mean_abs_decile_error > BIAS_TOL_PCT:
        verdict = "DIRECTIONAL_BUT_BIASED"
        hint = ("ranking is trustworthy but the magnitude is not — trust "
                "the sign/ordering, discount the predicted % "
                f"(decile error {mean_abs_decile_error:.1f}pp > "
                f"{BIAS_TOL_PCT:.1f}pp)")
    elif spearman >= SPEARMAN_GOOD and monotone_fraction >= MONOTONE_GOOD:
        verdict = "WELL_CALIBRATED"
        hint = "predicted % tracks realized % within tolerance"
    else:
        verdict = "WEAK_SIGNAL"
        hint = ("some rank skill but below the strong bar — usable as a "
                "tie-breaker, not a primary signal")

    return {
        "status": "ok",
        "verdict": verdict,
        "n": n,
        "spearman": round(spearman, 4),
        "pearson": round(pearson, 4),
        "monotone_fraction": round(monotone_fraction, 4),
        "mean_abs_decile_error": round(mean_abs_decile_error, 4),
        "buckets": buckets,
        "hint": hint,
    }


def scorer_calibration(scorer, records, n_buckets: int = 10) -> dict:
    """Run ``calibration_report`` for a DecisionScorer over outcome records
    (the ``data/decision_outcomes.jsonl`` row shape).

    The target is **action-aligned** exactly like ``train_scorer``: a SELL's
    realized goodness is ``-forward_return_5d`` (a drop after a SELL was the
    *right* call). Without this flip a rank-skilled SELL model would look
    anti-correlated and the verdict would be a false MISCALIBRATED. Records
    missing/with a non-finite ``forward_return_5d`` are skipped by
    ``calibration_report``'s own finite filter.
    """
    pairs = []
    for r in records:
        try:
            pred = scorer.predict(
                ml_score=r.get("ml_score", 0.0),
                rsi=r.get("rsi"),
                macd=r.get("macd"),
                mom5=r.get("mom5"),
                mom20=r.get("mom20"),
                regime_mult=r.get("regime_mult", 1.0),
                ticker=r.get("ticker", ""),
                vol_ratio=r.get("vol_ratio"),
                bb_pos=r.get("bb_position"),
                news_urgency=r.get("news_urgency"),
                news_article_count=r.get("news_article_count"),
            )
        except Exception:
            continue
        y = r.get("forward_return_5d")
        if y is None:
            continue
        action = str(r.get("action") or "BUY").upper()
        pairs.append((pred, -y if action == "SELL" else y))
    return calibration_report(pairs, n_buckets=n_buckets)


def _cli() -> int:
    """`python3 -m paper_trader.ml.calibration` — calibration of the live
    pickled scorer against the accumulated outcomes tail. Read-only."""
    from .decision_scorer import DecisionScorer

    root = Path(__file__).resolve().parent.parent.parent
    out_path = root / "data" / "decision_outcomes.jsonl"
    if not out_path.exists():
        print(f"[calibration] no outcomes file at {out_path}")
        return 1
    records = []
    for ln in out_path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            records.append(json.loads(ln))
        except Exception:
            pass
    scorer = DecisionScorer()
    if not scorer.is_trained:
        print("[calibration] scorer not trained — nothing to calibrate")
        return 1
    rep = scorer_calibration(scorer, records)
    print(f"scorer n_train={scorer.n_train}  outcomes={len(records)}")
    print(f"VERDICT: {rep['verdict']}  ({rep['hint']})")
    print(f"  n={rep['n']} spearman={rep['spearman']} "
          f"pearson={rep['pearson']} monotone={rep['monotone_fraction']} "
          f"mean_abs_decile_err={rep['mean_abs_decile_error']}pp")
    for b in rep["buckets"]:
        print(f"  d{b['idx']:>2} pred[{b['pred_lo']:+8.2f},{b['pred_hi']:+8.2f}] "
              f"mean_pred={b['mean_pred']:+7.2f}  "
              f"mean_realized={b['mean_realized']:+7.2f}  n={b['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())

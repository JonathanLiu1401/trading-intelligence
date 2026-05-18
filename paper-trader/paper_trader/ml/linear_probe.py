"""Linear-probe diagnostic — is the DecisionScorer's near-zero OOS skill an
**MLP-architecture** failure or a **feature-set** ceiling? Read-only.

This is a **read-only diagnostic**. It never trains the deployed model, never
touches `decision_scorer.pkl`, `decision_outcomes.jsonl`, `build_features`,
`N_FEATURES`, or any trade path — same operational discipline as
`paper_trader/ml/baseline_compare.py` / `calibration.py` / `gate_audit.py` /
`skill_trend.py`. Safe to run against the live unattended loop.

**Why this is not `baseline_compare.py` (the decisive distinction).**
`baseline_compare` settles "does the 17-feature MLP beat a *single*-feature
one-liner OOS?" — the standing answer is `MLP_NO_BETTER_THAN_TRIVIAL`
(MLP rank-IC ≈ best one-liner). But that verdict is structurally ambiguous to
a quant deciding what to *do* about it. It cannot tell apart two very
different worlds, each of which the data is equally consistent with:

  1. **MLP-architecture failure.** The numeric features *do* carry a
     combinable linear signal, but the regularized/clamped MLP (or its
     7-way sector one-hot memorization, which `feature_importance` flags)
     fails to extract it. ⇒ *Action: a linear head would beat the net.*
  2. **Feature-set ceiling.** No linear combination of the numeric features
     beats the single best feature either — the ceiling is the *inputs*, not
     the model class. ⇒ *Action: stop tuning the model; the gate (invariant
     #5) is structurally capped regardless of model family.*

`baseline_compare` (single-feature one-liners) and `overfit_gap` (the MLP's
own val/oos ratio) and `feature_importance` (which input the MLP leans on)
**none of them fit a multi-feature model of a different class** and ask
whether *that* recovers signal. This module is exactly that missing
discriminator: a small numpy ridge on the **same 10 numeric features the MLP
sees** (the sector one-hot is deliberately excluded — it is the documented
memorization vector, not quant signal), fit on the temporal-**train** slice
and scored on the temporal-**OOS** slice.

**Methodological care (load-bearing).** A multi-feature linear combination
must be *fit*, so — unlike the parameter-free one-liners in
`baseline_compare` — it can only be evaluated honestly on data it never saw.
The probe is fit on the **train** slice of
`validation.split_outcomes_temporal` (the EXACT split
`_train_decision_scorer` / `baseline_compare` use) with standardization
statistics taken from **train only** (no leakage — the same "split before
scale" discipline `decision_scorer.train_scorer` follows), then scored on the
held-out OOS slice. The deployed MLP is *data-advantaged* in this comparison:
it was trained in production on the full accumulated 5000-outcome tail, while
the probe sees only this file's train slice — so a **probe win is
conservative** (it beat a model that saw strictly more data). That asymmetry
is the honest, no-leakage choice and is stated here so a skeptical quant
reads a probe win as a floor, not a ceiling.

The codebase-universal SELL sign-flip is applied to the realized target for
**every** predictor via `baseline_compare._aligned_pred_target` (reused
verbatim — single source of truth). The probe is fit on the action-aligned
target, so — exactly like the training-aligned MLP — it learns the flip and
its raw prediction is paired with the aligned target (NOT flipped again).
Skill is `baseline_compare._skill` (tie-aware Spearman rank-IC + directional
accuracy), reused verbatim so the probe's, the MLP's, and the one-liners'
numbers are on the identical scale `calibration --oos` reports.

| Verdict | Meaning | Action |
|---------|---------|--------|
| `INSUFFICIENT_DATA` | scorer untrained, probe could not fit, or < `MIN_PAIRS` aligned OOS pairs | — |
| `LINEAR_PROBE_RECOVERS_SIGNAL` | probe OOS rank-IC clears BOTH the MLP and the best one-liner by `IC_MARGIN`, and clears the `MLP_IC_MIN` real-skill floor | the features carry combinable signal the MLP wastes — a linear head would beat the net |
| `NO_COMBINABLE_SIGNAL` | probe OOS rank-IC does NOT clear the best one-liner by `IC_MARGIN` | the ceiling is the feature set, not the model class — the gate is structurally capped |
| `LINEAR_MATCHES_MLP` | probe ≈ MLP (neither clearly additive beyond one-liners) | model class is not the lever; consistent with the standing `MLP_NO_BETTER_THAN_TRIVIAL` |

```bash
cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.linear_probe
cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.linear_probe --all
```
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .baseline_compare import (
    BASELINES,
    IC_MARGIN,
    MIN_PAIRS,
    MLP_IC_MIN,
    _aligned_pred_target,
    _skill,
)
from .decision_scorer import _to_float, build_features

# Ridge L2 strength on the standardized design matrix. Small but non-zero:
# enough to keep the closed-form solve well-conditioned when two numeric
# features are near-collinear (e.g. mom5/mom20), not so large it shrinks a
# genuine combinable signal away (which would fabricate a false
# NO_COMBINABLE_SIGNAL). 1.0 on standardized columns ≈ the same regime as
# decision_scorer.MLP_CONFIG's alpha=1e-2 once the differing scales are
# accounted for; the verdict is robust across 0.1–10 (locked by a test).
RIDGE_ALPHA = 1.0

# How many leading slots of build_features() are the numeric (non-sector)
# features. build_features returns [10 numeric] + [7 sector one-hot]; the
# probe deliberately uses ONLY the numeric block (the sector one-hot is the
# documented memorization vector feature_importance flags, not quant signal).
# Sourced from build_features() output, NOT re-listed, so a feature reorder
# can never silently desync the probe from the model it is judging.
_N_NUMERIC = 10


def _numeric_features(rec: dict) -> list[float]:
    """The 10 numeric features for one outcome row, IN build_features() order.

    Reuses `decision_scorer.build_features` and slices its numeric prefix so
    the probe sees byte-identical feature engineering (defaults, clamps,
    `_to_float` coercion) to the MLP it is judged against — single source of
    truth, zero drift. The decision_outcomes.jsonl row uses `bb_position`
    where build_features' kwarg is `bb_pos`; map it explicitly."""
    feats = build_features(
        ml_score=_to_float(rec.get("ml_score"), 0.0),
        rsi=rec.get("rsi"),
        macd=rec.get("macd"),
        mom5=rec.get("mom5"),
        mom20=rec.get("mom20"),
        regime_mult=_to_float(rec.get("regime_mult"), 1.0),
        ticker=str(rec.get("ticker") or ""),
        vol_ratio=rec.get("vol_ratio"),
        bb_pos=rec.get("bb_position"),
        news_urgency=rec.get("news_urgency"),
        news_article_count=rec.get("news_article_count"),
    )
    return [float(x) for x in feats[:_N_NUMERIC]]


class _RidgeProbe:
    """Closed-form L2 ridge on standardized features, numpy-only.

    Pickle-irrelevant (this diagnostic never persists) but kept as a small
    class so the fitted transform + weights travel together and prediction is
    a pure, deterministic, side-effect-free call — the same shape as
    `decision_scorer._LstsqModel`. The bias term is NOT regularized (a
    regularized intercept biases predictions toward zero for no benefit)."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, weights: np.ndarray) -> None:
        self._mean = np.asarray(mean, dtype=np.float64)
        self._std = np.asarray(std, dtype=np.float64)
        self._w = np.asarray(weights, dtype=np.float64)

    def predict(self, X) -> np.ndarray:
        Z = (np.asarray(X, dtype=np.float64) - self._mean) / self._std
        Za = np.hstack([Z, np.ones((len(Z), 1), dtype=np.float64)])
        return Za @ self._w


def _fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float = RIDGE_ALPHA):
    """Fit a `_RidgeProbe` on `X` (n×d) → `y` (n,). Standardize columns by
    *this X's* mean/std (the caller passes ONLY the train fold — no OOS
    leakage), augment a bias column, solve the regularized normal equations
    `(ZᵀZ + αI')⁻¹ Zᵀy` with the bias diagonal left at 0 (unregularized
    intercept). A zero-variance column gets std←1 so it contributes a
    constant the bias absorbs rather than dividing by zero. Returns None on
    any degenerate input (< 2 rows, non-finite, singular) so callers degrade
    to INSUFFICIENT_DATA rather than raising. Deterministic."""
    try:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < 2 or X.shape[0] != y.shape[0]:
            return None
        if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
            return None
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.where(std < 1e-12, 1.0, std)
        Z = (X - mean) / std
        Za = np.hstack([Z, np.ones((len(Z), 1), dtype=np.float64)])
        d = Za.shape[1]
        reg = np.eye(d, dtype=np.float64) * float(alpha)
        reg[-1, -1] = 0.0  # do not regularize the bias term
        A = Za.T @ Za + reg
        b = Za.T @ y
        w = np.linalg.solve(A, b)
        if not np.all(np.isfinite(w)):
            return None
        return _RidgeProbe(mean, std, w)
    except Exception:
        return None


def linear_probe_report(probe_preds, mlp_preds, baseline_preds, targets) -> dict:
    """Pure verdict function. All inputs are parallel sequences already in
    action-aligned space (caller applied the SELL flip / fit on aligned y).

    `probe_preds` — the fitted ridge's OOS predictions.
    `mlp_preds`   — the deployed scorer's OOS predictions (same rows).
    `baseline_preds` — `{name: vector}` of one-liners (same rows).
    `targets`     — action-aligned realized 5d returns (same rows).

    Returns a JSON-safe dict; never raises (faults → INSUFFICIENT_DATA)."""
    base = {
        "status": "ok", "verdict": "INSUFFICIENT_DATA",
        "n": len(targets) if targets is not None else 0,
        "probe": {"rank_ic": None, "dir_acc": None, "n": 0},
        "mlp": {"rank_ic": None, "dir_acc": None, "n": 0},
        "baselines": [], "best_baseline": None, "best_baseline_ic": None,
        "probe_minus_mlp": None, "probe_minus_best_baseline": None, "hint": "",
    }
    try:
        tgt = list(targets or [])
        pp = list(probe_preds or [])
        mp = list(mlp_preds or [])
        n = len(tgt)
        if n < MIN_PAIRS or len(pp) != n or len(mp) != n:
            base["hint"] = (f"need ≥{MIN_PAIRS} aligned OOS pairs, have "
                            f"n={n} (probe={len(pp)} mlp={len(mp)})")
            return base

        probe_skill = _skill(pp, tgt)
        mlp_skill = _skill(mp, tgt)
        base["probe"] = {"rank_ic": probe_skill["rank_ic"],
                         "dir_acc": probe_skill["dir_acc"],
                         "n": probe_skill["n"]}
        base["mlp"] = {"rank_ic": mlp_skill["rank_ic"],
                       "dir_acc": mlp_skill["dir_acc"], "n": mlp_skill["n"]}

        rows = []
        for name, vec in baseline_preds.items():
            v = list(vec or [])
            if len(v) != n:
                rows.append({"name": name, "rank_ic": None, "dir_acc": None,
                             "degenerate": True, "n": len(v)})
                continue
            s = _skill(v, tgt)
            rows.append({"name": name, "rank_ic": s["rank_ic"],
                         "dir_acc": s["dir_acc"],
                         "degenerate": s["degenerate"], "n": s["n"]})
        base["baselines"] = rows

        finite = [b for b in rows
                  if not b["degenerate"] and b["rank_ic"] is not None]
        probe_ic = probe_skill["rank_ic"]
        mlp_ic = mlp_skill["rank_ic"]
        if not finite or probe_ic is None or mlp_ic is None:
            base["hint"] = ("could not compute probe / MLP / any "
                            "non-degenerate baseline IC on the slice")
            return base

        best = max(finite, key=lambda b: b["rank_ic"])
        base["best_baseline"] = best["name"]
        base["best_baseline_ic"] = best["rank_ic"]
        best_ic = best["rank_ic"]
        base["probe_minus_mlp"] = round(probe_ic - mlp_ic, 4)
        base["probe_minus_best_baseline"] = round(probe_ic - best_ic, 4)

        beats_mlp = probe_ic > mlp_ic + IC_MARGIN
        beats_oneliner = probe_ic > best_ic + IC_MARGIN
        has_real_skill = probe_ic > MLP_IC_MIN

        if beats_mlp and beats_oneliner and has_real_skill:
            base["verdict"] = "LINEAR_PROBE_RECOVERS_SIGNAL"
            base["hint"] = (
                f"ridge rank_ic {probe_ic:+.3f} clears the deployed MLP "
                f"{mlp_ic:+.3f} AND best one-liner '{best['name']}' "
                f"{best_ic:+.3f} (both by > {IC_MARGIN}) and the "
                f"{MLP_IC_MIN} skill floor — the numeric features carry a "
                f"combinable linear signal the MLP architecture wastes; a "
                f"linear head would beat the net")
        elif probe_ic <= best_ic + IC_MARGIN:
            base["verdict"] = "NO_COMBINABLE_SIGNAL"
            base["hint"] = (
                f"ridge rank_ic {probe_ic:+.3f} does not clear best "
                f"one-liner '{best['name']}' {best_ic:+.3f} by {IC_MARGIN} "
                f"— no linear combination of the numeric features beats the "
                f"single best feature; the gate's ceiling is the feature "
                f"set, not the model class")
        else:
            base["verdict"] = "LINEAR_MATCHES_MLP"
            base["hint"] = (
                f"ridge rank_ic {probe_ic:+.3f} ≈ deployed MLP {mlp_ic:+.3f} "
                f"(Δ {probe_ic - mlp_ic:+.3f}); neither is clearly additive "
                f"beyond one-liner '{best['name']}' {best_ic:+.3f} — model "
                f"class is not the lever, consistent with the standing "
                f"MLP_NO_BETTER_THAN_TRIVIAL")
        return base
    except Exception as e:  # never raises near the unattended loop
        base["status"] = "error"
        base["verdict"] = "INSUFFICIENT_DATA"
        base["hint"] = f"probe report failed: {type(e).__name__}: {e}"
        return base


def scorer_linear_probe(scorer, records, oos_only: bool = True) -> dict:
    """Fit the ridge probe on the temporal-train slice, evaluate probe + the
    deployed `scorer` + every `baseline_compare` one-liner on the IDENTICAL
    temporal-OOS slice, and return `linear_probe_report`.

    `oos_only` (default True) is the trustworthy generalization view. With
    `oos_only=False` the probe is still fit on the temporal-train slice but
    evaluated in-sample on ALL records (diagnostic only — an in-sample probe
    trivially fits and the verdict is not generalization-meaningful; the
    `slice` field records which was used). Never raises."""
    try:
        recs = list(records or [])
    except Exception:
        recs = []

    slice_name = "all"
    train_recs: list[dict] = recs
    eval_recs: list[dict] = recs
    try:
        from paper_trader.validation import split_outcomes_temporal
        tr, oos = split_outcomes_temporal(recs, oos_fraction=0.2)
        if tr and oos:
            train_recs = tr
            eval_recs = oos if oos_only else recs
            slice_name = "oos" if oos_only else "all"
    except Exception:
        slice_name = "all"

    def _aligned(rec):
        """(numeric_features, raw_aligned_target, is_sell) for one row, or
        None when the realized 5d return is missing / non-finite."""
        y = rec.get("forward_return_5d")
        if y is None:
            return None
        try:
            yf = float(y)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(yf):
            return None
        is_sell = str(rec.get("action") or "BUY").upper() == "SELL"
        _, tgt = _aligned_pred_target(0.0, yf, is_sell)
        try:
            feats = _numeric_features(rec)
        except Exception:
            return None
        if not all(np.isfinite(f) for f in feats):
            return None
        return feats, tgt, is_sell

    # ── Fit the ridge on TRAIN ONLY (no leakage) ──────────────────────────
    Xtr, ytr = [], []
    for rec in train_recs:
        a = _aligned(rec)
        if a is None:
            continue
        feats, tgt, _ = a
        Xtr.append(feats)
        ytr.append(tgt)
    probe = None
    if len(Xtr) >= 2:
        probe = _fit_ridge(np.asarray(Xtr, dtype=np.float64),
                           np.asarray(ytr, dtype=np.float64))

    # ── Evaluate probe + MLP + one-liners on the SAME eval rows ───────────
    probe_preds: list[float] = []
    mlp_preds: list[float] = []
    base_preds: dict[str, list[float]] = {k: [] for k in BASELINES}
    targets: list[float] = []
    for rec in eval_recs:
        a = _aligned(rec)
        if a is None:
            continue
        feats, tgt, is_sell = a
        if probe is None:
            continue
        try:
            pv = float(probe.predict([feats])[0])
        except Exception:
            continue
        if not np.isfinite(pv):
            continue
        try:
            mp = float(scorer.predict(
                ml_score=rec.get("ml_score", 0.0),
                rsi=rec.get("rsi"), macd=rec.get("macd"),
                mom5=rec.get("mom5"), mom20=rec.get("mom20"),
                regime_mult=rec.get("regime_mult", 1.0),
                ticker=rec.get("ticker", ""),
                vol_ratio=rec.get("vol_ratio"), bb_pos=rec.get("bb_position"),
                news_urgency=rec.get("news_urgency"),
                news_article_count=rec.get("news_article_count"),
            ))
        except Exception:
            continue
        if not np.isfinite(mp):
            continue
        probe_preds.append(pv)
        mlp_preds.append(mp)
        targets.append(tgt)
        for name, fn in BASELINES.items():
            try:
                raw = float(fn(rec))
            except Exception:
                raw = 0.0
            if not np.isfinite(raw):
                raw = 0.0
            bp, _ = _aligned_pred_target(raw, tgt if not is_sell else -tgt,
                                         is_sell)
            base_preds[name].append(bp)

    rep = linear_probe_report(probe_preds, mlp_preds, base_preds, targets)
    rep["slice"] = slice_name
    rep["n_train_fit"] = len(Xtr)
    rep["n_records_considered"] = len(eval_recs)
    rep["ridge_alpha"] = RIDGE_ALPHA
    if probe is None:
        rep["verdict"] = "INSUFFICIENT_DATA"
        rep["status"] = "ok"
        if not rep.get("hint"):
            rep["hint"] = (f"ridge could not fit (n_train_usable="
                           f"{len(Xtr)}) — need ≥2 finite train rows")
    return rep


def analyze(outcomes_path: "Path | str", oos_only: bool = True) -> dict:
    """Load the live pickled scorer + outcomes file and run the probe.
    Read-only; never raises."""
    from .decision_scorer import DecisionScorer

    out: dict = {"status": "error", "verdict": "INSUFFICIENT_DATA",
                 "n": 0, "baselines": [], "hint": ""}
    try:
        p = Path(outcomes_path)
        if not p.exists():
            out["hint"] = f"no outcomes file at {p}"
            return out
        records: list[dict] = []
        for ln in p.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                records.append(json.loads(ln))
            except Exception:
                continue
        scorer = DecisionScorer()
        if not getattr(scorer, "is_trained", False):
            out["hint"] = "scorer not trained — nothing to compare"
            return out
        rep = scorer_linear_probe(scorer, records, oos_only=oos_only)
        rep.setdefault("n_train", getattr(scorer, "n_train", None))
        return rep
    except Exception as e:
        out["hint"] = f"probe failed: {type(e).__name__}: {e}"
        return out


def _cli(argv: list[str] | None = None) -> int:
    """`python3 -m paper_trader.ml.linear_probe [--all]` — fit a ridge on the
    10 numeric features (temporal-train slice) and compare its OOS rank-IC to
    the deployed MLP and every one-liner. Read-only.

    Exit code: 2 on `LINEAR_PROBE_RECOVERS_SIGNAL` (a linear head would beat
    the net — act) or `NO_COMBINABLE_SIGNAL` (the feature set is the ceiling —
    stop tuning the model), 0 on `LINEAR_MATCHES_MLP` / `INSUFFICIENT_DATA`
    (no actionable signal) — so an operator/cron can branch on "is the model
    class the lever?" exactly like `baseline_compare` / `gate_audit`."""
    import sys

    args = sys.argv[1:] if argv is None else argv
    oos_only = "--all" not in args

    root = Path(__file__).resolve().parent.parent.parent
    rep = analyze(root / "data" / "decision_outcomes.jsonl", oos_only=oos_only)
    print(f"VERDICT: {rep['verdict']}  ({rep.get('hint', '')})")
    pr = rep.get("probe", {})
    mlp = rep.get("mlp", {})
    print(f"  slice={rep.get('slice')}  n={rep.get('n')}  "
          f"n_train={rep.get('n_train')}  n_train_fit={rep.get('n_train_fit')}")
    print(f"  ridge  rank_ic={pr.get('rank_ic')} dir_acc={pr.get('dir_acc')}")
    print(f"  MLP    rank_ic={mlp.get('rank_ic')} dir_acc={mlp.get('dir_acc')}")
    print(f"  best_baseline={rep.get('best_baseline')} "
          f"ic={rep.get('best_baseline_ic')}  "
          f"probe−MLP={rep.get('probe_minus_mlp')}  "
          f"probe−best={rep.get('probe_minus_best_baseline')}")
    for b in rep.get("baselines", []):
        ic = b["rank_ic"]
        ic_s = f"{ic:+.4f}" if ic is not None else "   n/a"
        da = b["dir_acc"]
        da_s = f"{da:.4f}" if da is not None else "  n/a"
        deg = " [degenerate]" if b["degenerate"] else ""
        print(f"  {b['name']:<14} rank_ic={ic_s}  dir_acc={da_s}  "
              f"n={b['n']}{deg}")
    verdict = rep.get("verdict")
    if verdict in ("LINEAR_PROBE_RECOVERS_SIGNAL", "NO_COMBINABLE_SIGNAL"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())

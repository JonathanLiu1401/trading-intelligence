"""Conviction-gate ARM stability under bootstrap retraining — is the gate's
bucket decision a stable property of the features, or an artifact of which
80% of outcomes the scorer happened to train on?

This is a **read-only diagnostic**. It never writes `decision_scorer.pkl`
(it fits throwaway in-memory models — it does NOT call `train_scorer`, which
pickles to `SCORER_PATH`), never touches `decision_outcomes.jsonl`, never
mutates `build_features` / `N_FEATURES` / any trade path — same operational
discipline as `paper_trader/ml/gate_audit.py` / `gate_pnl.py` /
`calibration.py` / `skill_trend.py`. Safe to run against the live unattended
loop. Reads only the fast `decision_outcomes.jsonl` (no `backtest.db` scan).

**Why this is not any existing tool.** The ML/backtest domain has a
saturated *point-estimate* diagnostic suite — every one of `calibration`,
`gate_audit`, `gate_pnl`, `gate_realized`, `skill_trend`,
`baseline_compare`, `regime_audit`, `feature_importance`, `horizon_audit`
takes **one** model (the single deployed pickle, or one freshly trained
model) and asks a question about *its* predictions. `overfit_gap` trends
the val/oos RMSE *ratio* from the per-cycle ledger (memorization), and
`deploy_audit` compares the deployed pickle's hyper-params to source. None
of them can answer the question AGENTS.md keeps surfacing as the smoking
gun behind the gate's near-zero OOS skill:

  AGENTS.md, `decision_scorer.py`: *"observed −89% then +32% for the same
  LITE vector across two retrain cycles — the unbounded head is volatile."*

That is **prediction instability across retrains** — the scorer is
non-stationary, so the gate applies `×0.6` to a setup one cycle and `×1.3`
to the *same* setup the next. The point-estimate tools structurally cannot
see it (they each hold the model fixed). This module measures it directly:
it bootstrap-resamples the training slice, fits K independent scorers with
the **exact** `decision_scorer.MLP_CONFIG` + `build_features` pipeline
`train_scorer` uses, predicts a fixed evaluation slice with each, and
reports — at the **economic actuator, the five real gate arms** — how often
the SAME feature row's arm assignment is *not unanimous* across the K
bootstraps. A high flip rate means the gate's capital-sizing decision is
determined by training-resample luck, not signal: the documented LITE
volatility, quantified, and the strongest skeptical verdict on whether the
conviction gate underwrites noise.

**Single source of truth.** The five arms / boundary operators come from
`gate_audit.gate_arm` (imported, never re-declared — the same DRY rule
`gate_pnl` follows). The MLP hyper-parameters come from
`decision_scorer.MLP_CONFIG`; the feature vector from
`decision_scorer.build_features`; the ±output clamp from
`decision_scorer.PRED_CLAMP_PCT`; the temporal holdout from
`validation.split_outcomes_temporal` — the exact split `skill_trend` /
`gate_audit` / `_train_decision_scorer`'s `oos_rmse` use, so this describes
the same generalization-relevant slice. The dedup + sample-weight +
oversample steps mirror `train_scorer`'s sklearn branch line-for-line
(faithfully reconstructed here the way `gate_pnl._reconstruct_base_conviction`
mirrors `_ml_decide` — those steps are not separately importable).

```bash
cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.gate_stability
cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.gate_stability --all --bootstraps 16
```
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Single source of truth for the five gate arms / boundary operators —
# importing (not re-declaring) guarantees this diagnostic and gate_audit can
# never disagree about what the live `_ml_decide` gate does.
from .gate_audit import gate_arm
from .decision_scorer import (
    MLP_CONFIG,
    PRED_CLAMP_PCT,
    build_features,
    _to_float,
)

# Verdict thresholds — module-level so tests assert exact verdicts and a
# tuning change is a single reviewable edit (mirrors gate_audit.py /
# gate_pnl.py / calibration.py).
MIN_EVAL = 30          # need a real eval sample before any verdict
MIN_BOOTSTRAP = 4      # below this the flip estimate is itself too noisy
MIN_TRAIN = 60         # train_scorer's own floor is 30 post-dedup; require more
                       # so each bootstrap resample still has real diversity
STABLE_TOL = 0.15      # arm-flip rate ≤ this ⇒ the gate decision survives
                       # resampling — a stable property of the features
UNSTABLE_TOL = 0.40    # arm-flip rate ≥ this ⇒ most sizing decisions are
                       # resample luck; the gate underwrites training noise
DEFAULT_BOOTSTRAP = 10
DEFAULT_SEED = 42      # fixed ⇒ analyze() is reproducible (tests lock this)


def _load_outcomes(path: Path | str) -> list[dict]:
    """Read decision_outcomes.jsonl into a list of dicts. Never raises —
    a missing/corrupt file yields ``[]`` (the `gate_audit._load` contract)."""
    out: list[dict] = []
    try:
        p = Path(path)
        if not p.exists():
            return out
        with p.open("r") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    out.append(json.loads(ln))
                except Exception:
                    continue
    except Exception:
        return []
    return out


def _dedup_train(records: list[dict]) -> list[dict]:
    """Mirror `train_scorer`'s dedup EXACTLY: key on
    ``(ticker, sim_date, action.upper())`` (action is load-bearing — a BUY
    and SELL of the same name/day share features but carry opposite labels),
    keep the highest-``return_pct`` copy. Faithfully reconstructed, not
    importable (it lives inside `train_scorer`)."""
    seen: dict[tuple, dict] = {}
    for r in records:
        key = (
            str(r.get("ticker") or ""),
            str(r.get("sim_date") or ""),
            str(r.get("action") or "BUY").upper(),
        )
        rp = _to_float(r.get("return_pct"), 0.0)
        if key not in seen or rp > _to_float(seen[key].get("return_pct"), 0.0):
            seen[key] = r
    return list(seen.values())


def _features_targets_weights(records: list[dict]):
    """Mirror `train_scorer`'s sklearn-branch feature/target/weight build
    line-for-line (SELL sign-flip; weight = run-quality × llm multiplier).
    Returns ``(X, y, w)`` float32 arrays."""
    X_raw, y, weights = [], [], []
    for r in records:
        X_raw.append(build_features(
            _to_float(r.get("ml_score"), 0.0),
            r.get("rsi"), r.get("macd"), r.get("mom5"), r.get("mom20"),
            _to_float(r.get("regime_mult"), 1.0),
            str(r.get("ticker") or ""),
            vol_ratio=r.get("vol_ratio"),
            bb_pos=r.get("bb_position"),
            news_urgency=r.get("news_urgency"),
            news_article_count=r.get("news_article_count"),
        ))
        fr = _to_float(r.get("forward_return_5d"), 0.0)
        action = str(r.get("action") or "BUY").upper()
        y.append(-fr if action == "SELL" else fr)
        rp = _to_float(r.get("return_pct"), 0.0)
        llm_label = int(r.get("llm_quality_label") or 0)
        llm_mult = {1: 3.0, -1: 0.1, 0: 1.0}.get(llm_label, 1.0)
        weights.append(max(0.5, min(2.0, 1.0 + rp / 200.0)) * llm_mult)
    return (
        np.array(X_raw, dtype=np.float32),
        np.array(y, dtype=np.float32),
        np.array(weights, dtype=np.float32),
    )


def _eval_features(records: list[dict]):
    """Build the fixed evaluation matrix + per-row action (for the SELL
    sign convention) WITHOUT dedup — `evaluate_scorer_oos` does not dedup
    the holdout, and the gate acts on every decision, so neither do we."""
    X_raw, actions = [], []
    for r in records:
        X_raw.append(build_features(
            _to_float(r.get("ml_score"), 0.0),
            r.get("rsi"), r.get("macd"), r.get("mom5"), r.get("mom20"),
            _to_float(r.get("regime_mult"), 1.0),
            str(r.get("ticker") or ""),
            vol_ratio=r.get("vol_ratio"),
            bb_pos=r.get("bb_position"),
            news_urgency=r.get("news_urgency"),
            news_article_count=r.get("news_article_count"),
        ))
        actions.append(str(r.get("action") or "BUY").upper())
    return np.array(X_raw, dtype=np.float32), actions


def analyze(
    outcomes_path: Path | str,
    oos_only: bool = True,
    n_bootstrap: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Bootstrap-retrain stability of the conviction gate's ARM decision.

    | Verdict | Meaning |
    |---------|---------|
    | ``INSUFFICIENT_DATA`` | sklearn absent, or < ``MIN_TRAIN`` train / < ``MIN_EVAL`` eval rows / < ``MIN_BOOTSTRAP`` bootstraps |
    | ``GATE_ARM_STABLE`` | arm-flip rate ≤ ``STABLE_TOL`` — the gate's bucket decision survives resampling; it is a property of the features, not training luck |
    | ``GATE_ARM_BORDERLINE`` | between the two tolerances — partial instability |
    | ``GATE_ARM_UNSTABLE`` | arm-flip rate ≥ ``UNSTABLE_TOL`` — most sizing decisions flip arm under resampling; the gate underwrites training noise (the documented LITE volatility, measured) |

    Pure / total — never raises. Every fault degrades to a populated dict
    with ``status='error'`` / ``INSUFFICIENT_DATA`` (the diagnostic-feeding
    discipline of every sibling tool).

    `seed` is fixed by default so the bootstrap resamples — and therefore
    every metric — are reproducible (the determinism a quant needs to trend
    this and a test needs to lock it).
    """
    base: dict = {
        "status": "error",
        "verdict": "INSUFFICIENT_DATA",
        "slice": "oos" if oos_only else "all",
        "n_eval": 0,
        "n_train": 0,
        "n_bootstrap": 0,
        "gate_arm_flip_rate": None,
        "mean_pred_std": None,
        "median_pred_std": None,
        "mean_modal_agreement": None,
        "arm_distribution": {},
        "hint": "",
    }
    try:
        records = _load_outcomes(outcomes_path)
        if len(records) < (MIN_TRAIN + MIN_EVAL):
            base["hint"] = (
                f"need ≥{MIN_TRAIN + MIN_EVAL} outcome rows; "
                f"have {len(records)}"
            )
            return base

        try:
            from sklearn.neural_network import MLPRegressor
            from sklearn.preprocessing import StandardScaler
        except Exception:
            base["hint"] = "sklearn unavailable — MLP stability undefined"
            return base

        # Temporal split — bootstrap the history, measure stability on the
        # generalization-relevant tail (the exact slice skill_trend /
        # gate_audit / oos_rmse use).
        try:
            from paper_trader.validation import split_outcomes_temporal
            train_recs, oos_recs = split_outcomes_temporal(
                records, oos_fraction=0.2
            )
        except Exception:
            train_recs, oos_recs = records, []

        eval_recs = oos_recs if oos_only else records
        train_recs = _dedup_train(train_recs)

        if len(train_recs) < MIN_TRAIN:
            base["hint"] = (
                f"post-dedup train slice {len(train_recs)} < {MIN_TRAIN}"
            )
            return base
        if len(eval_recs) < MIN_EVAL:
            base["hint"] = f"eval slice {len(eval_recs)} < {MIN_EVAL}"
            return base
        if n_bootstrap < MIN_BOOTSTRAP:
            base["hint"] = f"n_bootstrap {n_bootstrap} < {MIN_BOOTSTRAP}"
            return base

        X, y, w = _features_targets_weights(train_recs)
        X_eval, eval_actions = _eval_features(eval_recs)
        n_train = len(X)
        n_eval = len(X_eval)

        rng = np.random.default_rng(seed)
        # preds[k] = clamped predictions of bootstrap-k on the FIXED eval set.
        preds = np.empty((n_bootstrap, n_eval), dtype=np.float64)
        fitted = 0
        for k in range(n_bootstrap):
            idx = rng.integers(0, n_train, size=n_train)  # resample w/ repl.
            Xb, yb, wb = X[idx], y[idx], w[idx]
            try:
                scaler = StandardScaler()
                Xb_s = scaler.fit_transform(Xb)
                # Mirror train_scorer's deterministic weight-oversampling
                # (0.5→1×, 1.0→2×, 1.5→3×, 2.0→4×) exactly.
                rep = np.maximum(1, np.round(wb * 2).astype(int))
                Xb_w = np.repeat(Xb_s, rep, axis=0)
                yb_w = np.repeat(yb, rep, axis=0)
                model = MLPRegressor(**MLP_CONFIG)
                model.fit(Xb_w, yb_w)
                raw = model.predict(scaler.transform(X_eval))
                raw = np.where(np.isfinite(raw), raw, 0.0)
                preds[k] = np.clip(raw, -PRED_CLAMP_PCT, PRED_CLAMP_PCT)
                fitted += 1
            except Exception:
                # A single bootstrap fit failing must not void the audit —
                # fill with the no-op (neutral-arm) prediction so it neither
                # fabricates a flip nor an agreement.
                preds[k] = 0.0

        if fitted < MIN_BOOTSTRAP:
            base["hint"] = (
                f"only {fitted}/{n_bootstrap} bootstrap fits succeeded "
                f"(< {MIN_BOOTSTRAP})"
            )
            return base

        # Per eval row: cross-bootstrap prediction σ, and the set of gate
        # arms the K predictions map to. flip ⇔ not unanimous.
        col_std = preds.std(axis=0)                       # (n_eval,)
        flips = 0
        agreements = []
        arm_of_mean: dict[str, int] = {}
        for j in range(n_eval):
            arms = [gate_arm(float(preds[k, j]))[0] for k in range(n_bootstrap)]
            distinct = set(arms)
            if len(distinct) > 1:
                flips += 1
            # Modal-arm agreement: how concentrated the K votes are.
            counts: dict[str, int] = {}
            for a in arms:
                counts[a] = counts.get(a, 0) + 1
            agreements.append(max(counts.values()) / float(n_bootstrap))
            # Informational: the arm the MEAN prediction lands in.
            ma = gate_arm(float(preds[:, j].mean()))[0]
            arm_of_mean[ma] = arm_of_mean.get(ma, 0) + 1

        flip_rate = flips / float(n_eval)
        base.update({
            "status": "ok",
            "n_eval": int(n_eval),
            "n_train": int(n_train),
            "n_bootstrap": int(fitted),
            "gate_arm_flip_rate": round(flip_rate, 4),
            "mean_pred_std": round(float(col_std.mean()), 4),
            "median_pred_std": round(float(np.median(col_std)), 4),
            "mean_modal_agreement": round(float(np.mean(agreements)), 4),
            "arm_distribution": {k: int(v) for k, v in
                                 sorted(arm_of_mean.items())},
        })

        if flip_rate <= STABLE_TOL:
            base["verdict"] = "GATE_ARM_STABLE"
            base["hint"] = (
                f"{flip_rate:.1%} of eval decisions flip gate arm across "
                f"{fitted} bootstrap retrains (≤ {STABLE_TOL:.0%}) — the "
                f"sizing decision is a stable property of the features"
            )
        elif flip_rate >= UNSTABLE_TOL:
            base["verdict"] = "GATE_ARM_UNSTABLE"
            base["hint"] = (
                f"{flip_rate:.1%} of eval decisions flip gate arm across "
                f"{fitted} bootstrap retrains (≥ {UNSTABLE_TOL:.0%}); mean "
                f"cross-bootstrap pred σ {base['mean_pred_std']:.1f}pp — the "
                f"gate's capital sizing is resample luck, not signal "
                f"(the documented LITE volatility, measured)"
            )
        else:
            base["verdict"] = "GATE_ARM_BORDERLINE"
            base["hint"] = (
                f"{flip_rate:.1%} of eval decisions flip gate arm "
                f"({STABLE_TOL:.0%}–{UNSTABLE_TOL:.0%}) — partial sizing "
                f"instability under resampling"
            )
        return base
    except Exception as e:  # never raises — total, like every sibling
        base["status"] = "error"
        base["verdict"] = "INSUFFICIENT_DATA"
        base["hint"] = f"{type(e).__name__}: {e}"
        return base


def main(argv: list[str] | None = None) -> int:
    """`python3 -m paper_trader.ml.gate_stability [--all] [--bootstraps N]`.

    Exit code 2 on ``GATE_ARM_UNSTABLE`` (cron-branchable, mirroring
    `gate_pnl`'s exit-2-on-the-bad-verdict convention); 0 otherwise.
    """
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    oos_only = "--all" not in args
    n_boot = DEFAULT_BOOTSTRAP
    if "--bootstraps" in args:
        try:
            n_boot = max(MIN_BOOTSTRAP, int(args[args.index("--bootstraps") + 1]))
        except (ValueError, IndexError):
            n_boot = DEFAULT_BOOTSTRAP

    root = Path(__file__).resolve().parent.parent.parent
    rep = analyze(
        root / "data" / "decision_outcomes.jsonl",
        oos_only=oos_only,
        n_bootstrap=n_boot,
    )
    print("── conviction-gate ARM stability (bootstrap retrain) ──")
    print(f"  slice           : {rep['slice']}")
    print(f"  status          : {rep['status']}")
    print(f"  verdict         : {rep['verdict']}")
    print(f"  n_train / n_eval: {rep['n_train']} / {rep['n_eval']}")
    print(f"  n_bootstrap     : {rep['n_bootstrap']}")
    print(f"  gate_arm_flip   : {rep['gate_arm_flip_rate']}")
    print(f"  mean_pred_std   : {rep['mean_pred_std']} pp")
    print(f"  median_pred_std : {rep['median_pred_std']} pp")
    print(f"  modal_agreement : {rep['mean_modal_agreement']}")
    print(f"  arm(mean) dist  : {rep['arm_distribution']}")
    print(f"  hint            : {rep['hint']}")
    return 2 if rep.get("verdict") == "GATE_ARM_UNSTABLE" else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Dead-feature audit for the deployed DecisionScorer pickle.

Catches the specific class of bug that pass #35 fixed: a feature added to
``DecisionScorer.build_features`` whose values are never plumbed into either
the training-data capture (``_compute_decision_outcomes``) or the inference
path (``_ml_decide``). Such a feature trains on constant zero (the
build_features default), and after StandardScaler centering ⇒ identical
zero input ⇒ the MLP's first-layer weights for that input neuron converge
to *exactly* 0.0. The model wastes one feature slot; the diagnostic-only
``feature_importance`` permutation reading reports it as "zero importance"
(plausibly confused with "model genuinely ignores the feature").

This audit looks at the **model's own weights**, not at downstream metrics:

* For ``MLPRegressor``: per-input neuron, ``mean(|W[i]|)`` where ``W`` is
  ``coefs_[0]`` (the input-layer matrix). A feature with mean |w| ≤ ``EPS``
  was almost certainly trained on a constant input — the StandardScaler
  divides by ``std + 1e-8``, so a constant-zero feature scales to
  effectively zero and L2 ``alpha`` then drives its weights to zero.
* For ``_LstsqModel`` (numpy fallback): per-feature, ``|w[i]|`` of the
  fitted weights. Same interpretation.

The check is structurally **complementary** to permutation importance:
permutation answers "does shuffling this feature change the prediction?"
(data-level). This answers "did the model ever LEARN anything from this
input?" (model-level). A feature that's dead at the weight level is the
specific signature of a plumbing bug — the feature *could* carry signal,
the model just never saw any variance in it.

Same operational discipline as ``feature_importance`` /
``calibration``: read-only, no train, no pickle write, no
``build_features`` / ``N_FEATURES`` touch — safe to run against the live
unattended loop. Never raises; every failure path yields an honest
sentinel envelope so a wired ledger consumer can persist the fault
without breaking the cycle.

CLI:

```bash
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.dead_feature_audit
# or --json for machine-readable
```
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from paper_trader.ml.decision_scorer import (
    DecisionScorer,
    FEATURE_NAMES,
    N_FEATURES,
)

# A trained sklearn MLPRegressor's input weights for a *real* (variance-
# carrying) feature land in the ~0.1–0.6 mean|w| range under the deployed
# (32, 16) net + L2 alpha=1e-2 configuration (verified directly against the
# live data/ml/decision_scorer.pkl: 0.27–0.49 across the 10 numeric + 3
# active sector slots, exactly 0.000000 for the dead enhanced-MACD slots
# the pass-#35 plumbing fix targeted). 1e-6 is loose enough that no real
# trained feature lands below it (the smallest live coefficient was 0.257
# — five orders of magnitude above the bound) AND tight enough that a
# genuinely dead feature (exactly 0.0 due to StandardScaler ÷ near-zero
# std) is unambiguously flagged. Cf. the sibling ``feature_importance``
# diagnostic's reliance on the deterministic 0.0 signature for the same
# class of dead feature.
DEAD_EPS = 1e-6


def _feature_weight_magnitudes(model) -> tuple[str, np.ndarray] | None:
    """Return ``(method, raw_per_feature_magnitudes)`` or None when no
    fitted weight matrix can be located.

    Mirrors ``DecisionScorer.feature_importance``'s model-type dispatch
    so the audit honours both the deployed sklearn MLP and the numpy
    lstsq fallback. Never raises — a degenerate model returns None
    and the caller emits an honest sentinel envelope."""
    try:
        if hasattr(model, "coefs_") and model.coefs_:
            W = np.asarray(model.coefs_[0], dtype=np.float64)
            return "mlp_first_layer_mean_abs_weight", np.abs(W).mean(axis=1)
        if hasattr(model, "w_"):
            w = np.asarray(model.w_, dtype=np.float64)
            # _LstsqModel: w_ has shape (n_features + 1,) — last entry is bias.
            if w.shape[0] < N_FEATURES:
                return None
            return "lstsq_abs_weight", np.abs(w[:N_FEATURES])
    except Exception:
        return None
    return None


def audit_dead_features(scorer: DecisionScorer | None = None,
                        eps: float = DEAD_EPS) -> dict[str, Any]:
    """Audit the deployed DecisionScorer's first-layer weights for features
    that the model never learned from (constant-input plumbing bugs).

    Verdict ladder:

    * ``NOT_TRAINED`` — no pickle on disk / load failed; cannot audit.
    * ``UNKNOWN_MODEL`` — model has no recognisable weight matrix
      (future model type that doesn't expose ``coefs_`` / ``w_``).
    * ``SHAPE_MISMATCH`` — weights' input dimension disagrees with
      ``N_FEATURES`` (a build_features change without a retrain — the
      pickle is stale; the gate-relevant ``deploy_audit`` already
      surfaces this from a different angle).
    * ``OK`` — zero dead features under the ``eps`` bound.
    * ``HAS_DEAD`` — one or more features have mean |w| ≤ ``eps``;
      the diagnostic operator must check whether the data pipeline
      plumbs them in OR remove the input slot from ``build_features``.

    Returns a JSON-safe dict ready to persist into a per-cycle ledger.
    Never raises (a wired ledger consumer must never be able to break
    the loop — same discipline every sibling audit follows)."""
    if scorer is None:
        scorer = DecisionScorer()
    try:
        if not scorer.is_trained or scorer._model is None:
            return {
                "verdict": "NOT_TRAINED",
                "method": None,
                "n_train": int(getattr(scorer, "_n_train", 0) or 0),
                "n_features_total": N_FEATURES,
                "n_features_dead": 0,
                "dead_features": [],
                "eps": float(eps),
            }
        out = _feature_weight_magnitudes(scorer._model)
        if out is None:
            return {
                "verdict": "UNKNOWN_MODEL",
                "method": None,
                "n_train": int(getattr(scorer, "_n_train", 0) or 0),
                "n_features_total": N_FEATURES,
                "n_features_dead": 0,
                "dead_features": [],
                "eps": float(eps),
            }
        method, raw = out
        if raw.size != N_FEATURES:
            return {
                "verdict": "SHAPE_MISMATCH",
                "method": method,
                "n_train": int(getattr(scorer, "_n_train", 0) or 0),
                "n_features_total": N_FEATURES,
                "n_features_in_pickle": int(raw.size),
                "n_features_dead": 0,
                "dead_features": [],
                "eps": float(eps),
            }
        # Non-finite weights would otherwise compare unpredictably against
        # `eps` (NaN ≤ x is False), so a poisoned weight could silently slip
        # past the gate. Treat non-finite as dead — that's the safer side:
        # NaN/Inf in coefs_[0] is itself a bug an operator wants surfaced.
        # Mirrors the `feature_importance.feature_importance` non-finite
        # replacement discipline.
        raw = np.where(np.isfinite(raw), raw, 0.0)
        dead_idx = np.where(raw <= eps)[0]
        dead = [
            {"feature": FEATURE_NAMES[int(i)],
             "mean_abs_weight": round(float(raw[int(i)]), 9)}
            for i in dead_idx
        ]
        verdict = "HAS_DEAD" if dead else "OK"
        return {
            "verdict": verdict,
            "method": method,
            "n_train": int(getattr(scorer, "_n_train", 0) or 0),
            "n_features_total": int(N_FEATURES),
            "n_features_dead": int(len(dead)),
            "dead_features": dead,
            "eps": float(eps),
        }
    except Exception as e:
        return {
            "verdict": "ERROR",
            "method": None,
            "n_train": int(getattr(scorer, "_n_train", 0) or 0),
            "n_features_total": N_FEATURES,
            "n_features_dead": 0,
            "dead_features": [],
            "eps": float(eps),
            "error": str(e),
        }


def main(argv: list[str] | None = None) -> int:
    """CLI: report the audit verdict in human-readable or JSON form.

    Exit code: 0 when verdict ∈ {OK, NOT_TRAINED} (no actionable bug);
    1 when HAS_DEAD / SHAPE_MISMATCH / UNKNOWN_MODEL / ERROR — so shell
    callers can gate on ``$?`` like sibling ml/ CLIs (host_guard,
    decision_scorer CLI)."""
    import argparse
    import sys

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.dead_feature_audit",
        description="Audit the deployed DecisionScorer pickle's first-"
                    "layer weights for features the model never learned "
                    "from (mean |w| ≤ eps ⇒ constant-input plumbing bug "
                    "OR untouched build_features slot).",
    )
    p.add_argument("--eps", type=float, default=DEAD_EPS,
                   help=f"Threshold for declaring a feature dead "
                        f"(default {DEAD_EPS}).")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of a table.")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    out = audit_dead_features(eps=args.eps)
    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0 if out["verdict"] in ("OK", "NOT_TRAINED") else 1
    print(f"[dead_feature_audit] verdict={out['verdict']}  "
          f"n_train={out['n_train']}  "
          f"n_dead={out['n_features_dead']}/{out['n_features_total']}  "
          f"method={out.get('method')}  eps={out['eps']}")
    if out.get("dead_features"):
        print("  dead features (mean |w| ≤ eps):")
        for r in out["dead_features"]:
            print(f"    {r['feature']:<28}  mean|w|={r['mean_abs_weight']:.6e}")
        print("  → check `_compute_decision_outcomes` and `_ml_decide` "
              "actually pass each listed feature to the scorer; if so the "
              "build_features slot is stale and should be removed.")
    if out.get("error"):
        print(f"  error: {out['error']}")
    return 0 if out["verdict"] in ("OK", "NOT_TRAINED") else 1


if __name__ == "__main__":
    raise SystemExit(main())

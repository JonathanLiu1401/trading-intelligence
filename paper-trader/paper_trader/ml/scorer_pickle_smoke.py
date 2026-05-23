"""Deployed-scorer pickle SMOKE diagnostic — read-only.

Sibling to ``deploy_audit`` (architecture drift) and ``scorer_freshness``
(loop heartbeat). Those answer "is the deployed config right?" and "is the
loop still retraining?" — but neither catches the third class of pathology
the 2026-05-23 finding #1 exposed: **the pickle exists, the architecture
matches, the loop is alive, yet the pickle's internals are bogus** because
a smoke-test or synthetic retrain wrote a degenerate model. That episode's
smoking gun was 101 ``pred_quantiles`` collapsed to ``18.934`` and
``label_quantiles`` = ``[1, 2, …, 39]`` (sequential integers, not real
forward returns). Every existing diagnostic looked healthy, the gate kept
gating, and the consumer-visible ``predict_percentile`` / ``predict_calibrated``
fields silently returned 0 or 100 for every input.

This module asks the question those don't: *does the on-disk pickle pass a
basic battery of internal sanity probes — does it predict, do its
quantiles span a real range, does it vary across inputs?* The companion
``decision_scorer._raw_to_percentile`` guard (the same-pass Phase-1 fix)
makes the rank-derived consumer fields degrade truthfully on a collapsed
table; this module surfaces the underlying condition so an operator's
shell triage gets one verdict instead of a "why is every prediction's
percentile None now" mystery.

Same operational discipline as every sibling diagnostic (``deploy_audit``,
``scorer_freshness``, ``scorer_health``): read-only, no train, no pickle
write, no ``build_features`` / ``N_FEATURES`` touch, no trade path — safe
to run against the live unattended loop. Never raises — every fault
degrades to an honest non-FAILURE verdict.

```bash
cd /home/zeph/trading-intelligence/paper-trader && \
    python3 -m paper_trader.ml.scorer_pickle_smoke
# Exit 2 on COLLAPSED_PRED_QUANTILES / COLLAPSED_LABEL_QUANTILES /
# DEGENERATE_PREDICTIONS / INSUFFICIENT_N_TRAIN (operator-actionable
# states); 0 on HEALTHY / INSUFFICIENT_DATA / UNREADABLE_PICKLE /
# LSTSQ_FALLBACK (none of which prove a regression).
```
"""
from __future__ import annotations

import math
import pickle
from pathlib import Path

# Below this threshold the conviction gate (CLAUDE.md invariant #5) never
# engages, but a deployed pickle with `n_train` orders of magnitude below
# the gate floor is more likely a synthetic-clobber than a legitimate
# warm-up state. The 2026-05-23 finding #1 footprint was `n_train=39`.
# 100 is the same "meaningful prediction floor" `stack_liveness.SCORER_PKL_MIN_N_TRAIN`
# already uses — keeping it consistent so the two surfaces agree.
MIN_N_TRAIN = 100

# Reference grid for the predict-variance probe. Vary `ml_score` across a
# wide bullish→bearish range with one ticker so we have an honest expectation:
# a healthy model trained on real outcomes MUST emit varying predictions
# across this grid (the scorer skill table has rank-IC ≈ 0.36 OOS, so
# ml_score is correlated with predicted return — a constant predictor is
# definitionally broken). The grid is intentionally NOT MLPRegressor-coupled
# (no need to import sklearn): we just measure the spread of predictions
# the deployed model emits.
_PROBE_ML_SCORES = (-10.0, -5.0, 0.0, 5.0, 10.0)
_PROBE_TICKER = "NVDA"  # in WATCHLIST + SECTOR_MAP, so build_features always shapes correctly

# A model emitting predictions within this band across the bullish→bearish
# probe grid is effectively a constant predictor. Realistic real-corpus
# spreads are typically 5–15pp across this range; 0.5pp is a conservative
# "this is essentially flat" floor that catches the documented finding
# without false-flagging a quiet model.
DEGENERATE_PRED_VAR_THRESHOLD = 0.5  # percentage points


def _is_lstsq_fallback(model) -> bool:
    """Mirror `deploy_audit._is_lstsq_fallback` — class-name match without
    importing the class (no import cycle risk)."""
    cls = type(model)
    return (cls.__name__ == "_LstsqModel"
            and "decision_scorer" in (getattr(cls, "__module__", "") or ""))


def _is_quantile_collapsed(q) -> bool | None:
    """True if a 101-point quantile table has zero spread (max == min) —
    the documented synthetic-clobber footprint. Returns None for a legacy
    pickle where the field is absent (cannot tell ⇒ not a failure).
    """
    if q is None:
        return None
    try:
        import numpy as _np
        arr = _np.asarray(q, dtype=_np.float64)
        if arr.size < 2:
            return None
        # Strictly: max <= min covers the constant case. `<` would never
        # be True for a sorted ascending table.
        return float(arr.max()) <= float(arr.min())
    except Exception:
        return None


def _probe_prediction_variance(scorer_path: Path) -> dict:
    """Run the predict-variance probe through ``DecisionScorer``.

    Returns ``{spread, n_predicted, n_failed, predictions}``. ``spread`` is
    ``max - min`` over successfully-produced predictions across the
    bullish→bearish probe grid; ``n_failed`` counts probes whose
    ``predict_with_meta`` returned ``failed=True`` (model load issue,
    shape mismatch). Always finite numerics or None — never raises.
    """
    out: dict = {"spread": None, "n_predicted": 0, "n_failed": 0,
                 "predictions": []}
    try:
        from .decision_scorer import DecisionScorer
        # Import-time decision_scorer rebinds SCORER_PATH internally — we
        # construct DecisionScorer to honor the same singleton/load cache
        # path tests redirect. The probe is read-only: predict_with_meta
        # never trains, never writes.
        s = DecisionScorer()
        if not s.is_trained:
            return out
        preds: list[float] = []
        for score in _PROBE_ML_SCORES:
            meta = s.predict_with_meta(
                ml_score=score, rsi=50.0, macd=0.0, mom5=0.0, mom20=0.0,
                regime_mult=1.0, ticker=_PROBE_TICKER,
                vol_ratio=1.0, bb_pos=0.0,
                news_urgency=50.0, news_article_count=1.0,
            )
            if meta.get("failed"):
                out["n_failed"] += 1
                continue
            pred = float(meta.get("pred", 0.0))
            if not math.isfinite(pred):
                out["n_failed"] += 1
                continue
            preds.append(pred)
        out["n_predicted"] = len(preds)
        out["predictions"] = [round(p, 4) for p in preds]
        if len(preds) >= 2:
            out["spread"] = round(max(preds) - min(preds), 4)
        return out
    except Exception:
        return out


def analyze(scorer_path: Path | str | None = None) -> dict:
    """Smoke-test the deployed scorer pickle. Pure, total, never raises.

    Verdict ladder (precedence-ordered, first matching wins — adverse
    states take precedence so a CLI / cron caller's exit code reflects
    the WORST observed condition):

    1. ``INSUFFICIENT_DATA``         — no pickle on disk
    2. ``UNREADABLE_PICKLE``         — load failed (torn write, sklearn-absent host)
    3. ``LSTSQ_FALLBACK``            — numpy lstsq deployed; MLP probes N/A
    4. ``INSUFFICIENT_N_TRAIN``      — n_train < MIN_N_TRAIN
    5. ``COLLAPSED_PRED_QUANTILES``  — pred_quantiles max == min
       (synthetic-clobber footprint; rank/calibrated fields degrade to None
       per the same-pass _raw_to_percentile guard)
    6. ``COLLAPSED_LABEL_QUANTILES`` — label_quantiles max == min
       (every realized label is the same value — corpus poisoned or
       trainer wrote sequential integers)
    7. ``DEGENERATE_PREDICTIONS``    — predict spread across the
       bullish→bearish probe grid is below DEGENERATE_PRED_VAR_THRESHOLD
       (the model is essentially a constant predictor, gate-relevant)
    8. ``HEALTHY``                   — every probe passes

    The dict echoes the per-check evidence so a reading quant or
    dashboard sees the exact numbers, not just a label.
    """
    if scorer_path is None:
        try:
            from .decision_scorer import SCORER_PATH
            scorer_path = SCORER_PATH
        except Exception as exc:
            return {
                "verdict": "INSUFFICIENT_DATA",
                "hint": f"decision_scorer import failed ({type(exc).__name__})",
                "n_train": None, "pred_quantiles_collapsed": None,
                "label_quantiles_collapsed": None,
                "prediction_spread": None, "predictions": [],
                "n_predicted": 0, "n_failed": 0,
            }

    out: dict = {
        "verdict": "INSUFFICIENT_DATA",
        "hint": "",
        "n_train": None,
        "pred_quantiles_collapsed": None,
        "label_quantiles_collapsed": None,
        "prediction_spread": None,
        "predictions": [],
        "n_predicted": 0,
        "n_failed": 0,
    }

    try:
        p = Path(scorer_path)
        if not p.exists():
            out["hint"] = "no deployed pickle (untrained loop / fresh checkout)"
            return out

        try:
            with p.open("rb") as fh:
                state = pickle.load(fh)
        except Exception as exc:
            out["verdict"] = "UNREADABLE_PICKLE"
            out["hint"] = (f"pickle unreadable ({type(exc).__name__}) — "
                           "torn write or sklearn-absent host; can't prove "
                           "a regression")
            return out

        if not isinstance(state, dict):
            out["verdict"] = "UNREADABLE_PICKLE"
            out["hint"] = ("pickle deserialized to non-dict — unsupported "
                           "scorer pickle layout")
            return out

        model = state.get("model")
        if model is None:
            out["verdict"] = "UNREADABLE_PICKLE"
            out["hint"] = "pickle missing 'model' key"
            return out

        if _is_lstsq_fallback(model):
            out["verdict"] = "LSTSQ_FALLBACK"
            out["hint"] = ("numpy lstsq fallback deployed (sklearn-absent "
                           "host) — MLP smoke probes do not apply; "
                           "deploy_audit handles the config check")
            # Still surface n_train so the dashboard can render it.
            try:
                out["n_train"] = int(state.get("n_train", 0)) or None
            except (TypeError, ValueError):
                pass
            return out

        # n_train check — fast, no probe call.
        try:
            n_train = int(state.get("n_train", 0))
        except (TypeError, ValueError):
            n_train = 0
        out["n_train"] = n_train

        # Quantile collapse checks — direct read of the pickled tables.
        out["pred_quantiles_collapsed"] = _is_quantile_collapsed(
            state.get("pred_quantiles"))
        out["label_quantiles_collapsed"] = _is_quantile_collapsed(
            state.get("label_quantiles"))

        # Predict-variance probe — runs through DecisionScorer so we cover
        # the same load + scaler + predict path the conviction gate uses.
        probe = _probe_prediction_variance(p)
        out["prediction_spread"] = probe["spread"]
        out["n_predicted"] = probe["n_predicted"]
        out["n_failed"] = probe["n_failed"]
        out["predictions"] = probe["predictions"]

        # Verdict — adverse precedence: insufficient n_train > collapsed
        # quantiles > degenerate predictions. A pickle with very few rows
        # is the most actionable case (delete + wait for the loop) and a
        # collapsed quantile table is the second (a re-retrain on real
        # outcomes corrects it).
        if n_train < MIN_N_TRAIN:
            out["verdict"] = "INSUFFICIENT_N_TRAIN"
            out["hint"] = (f"n_train={n_train} < MIN_N_TRAIN={MIN_N_TRAIN}; "
                           "smoke-test footprint (the 2026-05-23 finding #1 "
                           "synthetic n=39 clobber landed here). Restore "
                           "the pickle from a real retrain.")
            return out

        if out["pred_quantiles_collapsed"]:
            out["verdict"] = "COLLAPSED_PRED_QUANTILES"
            out["hint"] = ("pred_quantiles collapsed to a single value — "
                           "consumer-visible percentile/calibrated fields "
                           "return None (per the same-pass _raw_to_percentile "
                           "guard). Retrain from the real outcomes corpus.")
            return out

        if out["label_quantiles_collapsed"]:
            out["verdict"] = "COLLAPSED_LABEL_QUANTILES"
            out["hint"] = ("label_quantiles collapsed to a single value — "
                           "the training corpus has zero variance in "
                           "forward_return_5d, or the trainer wrote "
                           "sequential integers (the 2026-05-23 finding #1 "
                           "synthetic-clobber footprint).")
            return out

        # Degenerate-predictions check: if every probe ran successfully
        # AND the spread across the bullish→bearish grid is below the
        # threshold, the model is effectively constant. Skip the check
        # when too few probes succeeded — a partial probe set isn't
        # enough evidence either way.
        spread = out["prediction_spread"]
        if (out["n_predicted"] >= 2 and spread is not None
                and spread < DEGENERATE_PRED_VAR_THRESHOLD):
            out["verdict"] = "DEGENERATE_PREDICTIONS"
            out["hint"] = (
                f"predict spread {spread:.4f}pp across "
                f"ml_score∈{list(_PROBE_ML_SCORES)} < "
                f"{DEGENERATE_PRED_VAR_THRESHOLD}pp — the model emits "
                "essentially the same prediction for every input. The "
                "gate would modulate every BUY by the same multiplier; "
                "useless work. Retrain from the real outcomes corpus."
            )
            return out

        if out["n_predicted"] == 0:
            out["verdict"] = "UNREADABLE_PICKLE"
            out["hint"] = ("every probe predict failed — model load or "
                           "feature-shape mismatch; can't smoke-test")
            return out

        out["verdict"] = "HEALTHY"
        out["hint"] = (f"n_train={n_train}, predict spread "
                       f"{spread:.2f}pp across the probe grid, both "
                       "quantile tables span a real range")
        return out

    except Exception as exc:  # pragma: no cover - belt & braces
        return {
            "verdict": "INSUFFICIENT_DATA",
            "hint": f"analyze error ({type(exc).__name__})",
            "n_train": None, "pred_quantiles_collapsed": None,
            "label_quantiles_collapsed": None,
            "prediction_spread": None, "predictions": [],
            "n_predicted": 0, "n_failed": 0,
        }


_ADVERSE_VERDICTS = frozenset({
    "INSUFFICIENT_N_TRAIN", "COLLAPSED_PRED_QUANTILES",
    "COLLAPSED_LABEL_QUANTILES", "DEGENERATE_PREDICTIONS",
})


def is_pickle_smoke_failed(scorer_path: Path | str | None = None) -> bool | None:
    """Convenience boolean for a future per-cycle skill ledger or
    dashboard panel. ``True`` on a proven adverse verdict (operator
    action needed), ``False`` on HEALTHY, ``None`` on can't-tell
    (INSUFFICIENT_DATA / UNREADABLE_PICKLE / LSTSQ_FALLBACK). Mirrors
    ``deploy_audit.is_deploy_stale`` exactly.
    """
    try:
        rep = analyze(scorer_path)
        v = rep.get("verdict")
        if v in _ADVERSE_VERDICTS:
            return True
        if v == "HEALTHY":
            return False
        return None
    except Exception:
        return None


def _cli(argv: list[str] | None = None) -> int:
    """`python3 -m paper_trader.ml.scorer_pickle_smoke`.

    Exit code 2 on an adverse (operator-actionable) verdict, 0 otherwise —
    same shell-gateable contract as ``deploy_audit._cli`` / ``host_guard``.
    Accepts ``--json`` for machine-readable output and ``--scorer-path``
    for testing against a non-default pickle.
    """
    import argparse
    import json
    import sys

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.scorer_pickle_smoke",
        description="Smoke-test the deployed DecisionScorer pickle. "
                    "Catches the synthetic-clobber footprint (collapsed "
                    "quantiles, n_train≪floor, constant predictions) that "
                    "deploy_audit / scorer_freshness structurally cannot.",
    )
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of a table.")
    p.add_argument("--scorer-path", default=None, dest="scorer_path",
                   help="Audit a non-default pickle path (test helper).")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    rep = analyze(args.scorer_path)
    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
        return 2 if rep["verdict"] in _ADVERSE_VERDICTS else 0

    print(f"VERDICT: {rep['verdict']}")
    print(f"  {rep['hint']}")
    if rep["n_train"] is not None:
        print(f"  n_train={rep['n_train']}")
    if rep["pred_quantiles_collapsed"] is not None:
        print(f"  pred_quantiles_collapsed={rep['pred_quantiles_collapsed']}")
    if rep["label_quantiles_collapsed"] is not None:
        print(f"  label_quantiles_collapsed={rep['label_quantiles_collapsed']}")
    if rep["prediction_spread"] is not None:
        print(f"  prediction_spread={rep['prediction_spread']:.4f}pp  "
              f"({rep['n_predicted']} predicted, {rep['n_failed']} failed)")
        print(f"  probe predictions: {rep['predictions']}")
    return 2 if rep["verdict"] in _ADVERSE_VERDICTS else 0


if __name__ == "__main__":
    raise SystemExit(_cli())

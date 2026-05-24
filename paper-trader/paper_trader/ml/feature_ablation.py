"""Feature ablation audit — read-only counterpart to ``feature_importance``.

``feature_importance`` reports which features the trained MLP *leans on*
(mean-|first-layer-weight|). ``feature_correlation_audit`` reports
``SEVERE_COLLINEARITY`` (rsi↔bb_position |spearman|=0.91, etc) — the input
space has fewer than 10 effective degrees of freedom. But neither answers
the natural follow-up question a quant cares about:

  *If I drop feature X, does the scorer's OOS rank-IC go UP (X was
  redundant noise the MLP overweights), DOWN (X was load-bearing), or
  stay roughly the same?*

That is the question this analyzer answers head-on. For each feature
group it produces a counterfactual "what would the scorer's OOS rank-IC
be if we set this feature to its training-mean baseline?" reading, then
reports the delta vs the baseline (un-ablated) rank-IC.

Implementation choice: **inference-time ablation, no retrain.** Zero out
the relevant standardized column(s) of the OOS feature matrix and run
``model.predict`` on the modified inputs. Setting a standardized feature
to 0 is exactly equivalent to setting the raw feature to its training
mean — i.e. "what does the model output if this feature carries no
information for these rows?". No new training cost (a full ablation
sweep over 11 groups runs in seconds vs minutes-to-hours for
retrain-from-scratch ablation), and the resulting deltas describe the
*deployed pickle's* sensitivity — exactly what an operator needs to know
right now, not a counterfactual model.

| Verdict | Trigger |
|---------|---------|
| ``SCORER_UNTRAINED`` | no deployed pickle / lstsq fallback / model unavailable |
| ``INSUFFICIENT_DATA`` | fewer than ``MIN_OBS`` OOS rows survive feature/label validation |
| ``BASELINE_DEGENERATE`` | baseline rank-IC ties at 0 (constant-prediction model — every ablation is no-op by construction) |
| ``LOAD_BEARING_DETECTED`` | one or more features have ``delta <= -EDGE_TOL`` (removing them HURTS) but none REDUNDANT — the gate's skill is concentrated in those features |
| ``REDUNDANT_DETECTED`` | one or more features have ``delta >= EDGE_TOL`` (removing them HELPS) but none LOAD_BEARING — the MLP is over-weighting noise |
| ``MIXED`` | both kinds present — some features carry signal, some are noise |
| ``NO_SIGNIFICANT_EFFECT`` | every ``|delta| < EDGE_TOL`` — the OOS rank ordering is invariant to which feature the model sees |

This is a **read-only diagnostic**. It never trains, never touches
``decision_scorer.pkl``, ``decision_outcomes.jsonl``, ``build_features``,
``N_FEATURES``, or any trade path — the same operational discipline as
``feature_correlation_audit`` / ``calibration`` / ``baseline_compare``.

```bash
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.feature_ablation
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.feature_ablation --all   # full corpus
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.feature_ablation --json
```
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


# Feature groups to ablate. Each entry maps a human-readable group name
# to the list of standardized-column indices that get zeroed for that
# ablation. Mirrors ``decision_scorer.FEATURE_NAMES``: the first 10 are
# numeric, the last 7 are sector one-hots which are ablated together
# (zeroing them individually is meaningless — a row with no sector dummy
# is impossible in build_features).
NUMERIC_FEATURES = [
    "ml_score", "rsi", "macd", "mom5", "mom20", "regime_mult",
    "vol_ratio", "bb_pos", "news_urgency", "news_article_count",
]
# Index ranges built dynamically from FEATURE_NAMES so a future feature
# reorder in decision_scorer.py is caught by the analyze() check that
# pulls FEATURE_NAMES at call time.

# Minimum OOS row count before any ablation verdict is produced. Below
# this floor rank-IC is too noisy to read deltas off (mirrors the floor
# `baseline_compare` and `feature_correlation_audit` use).
MIN_OBS = 50

# Delta tolerance (rank-IC absolute points). A feature whose removal
# moves OOS rank-IC by less than this is treated as "no effect" — the
# null hypothesis the verdict ladder starts from. 0.02 is small enough
# to catch real economic signal (in a noisy n~1000 OOS slice, a +0.05
# rank-IC move is meaningful) but large enough to ignore numerical
# precision noise. Mirrors the AGENTS.md gate_audit `EDGE_TOL_PP` discipline.
EDGE_TOL = 0.02


def _finite_float(v) -> float | None:
    """Mirrors ``feature_correlation_audit._finite_float`` — returns None on
    missing/non-finite so callers can drop the row rather than fabricate
    a zero (which would inflate the rank metric with fake ties)."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _iter_rows(path: Path) -> Iterable[dict]:
    """Yield parsed dicts from a JSONL file; skip malformed lines."""
    try:
        with path.open() as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    yield json.loads(ln)
                except Exception:
                    continue
    except (OSError, ValueError):
        return


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank tie-aware rank transform (self-contained — same impl as
    feature_correlation_audit)."""
    a = np.asarray(a, dtype=np.float64)
    order = a.argsort()
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    s = a[order]
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s[j] == s[i]:
            j += 1
        if j - i > 1:
            mean_rank = (i + j + 1) / 2.0
            ranks[order[i:j]] = mean_rank
        i = j
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    """Tie-aware Spearman; returns None when either side has zero variance
    or n<2 (degenerate — no rank skill expressible). Mirrors
    feature_correlation_audit._spearman EXCEPT we return None on
    degeneracy instead of 0.0 so the caller can mark BASELINE_DEGENERATE
    honestly rather than fabricate a flat-zero baseline that every
    ablation looks 'no-op' against."""
    if len(a) < 2:
        return None
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return None
    ar = _rankdata(a)
    br = _rankdata(b)
    if ar.std() == 0.0 or br.std() == 0.0:
        return None
    rho = float(np.corrcoef(ar, br)[0, 1])
    return rho if math.isfinite(rho) else None


def _build_feature_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build (raw_features, realized_returns, action) for OOS rows that have
    every key the scorer feature builder needs AND a finite
    ``forward_return_5d``. Drops rows missing any required field — mirrors
    the ``evaluate_scorer_oos`` / ``_oos_rank_metrics`` drop discipline so
    the resulting matrix is exactly the set of trustworthy rows the gate's
    OOS metric is computed on.

    Returns ``(X_raw, y_realized_sell_flipped, actions)``. ``X_raw`` is
    shape ``(n, N_FEATURES)`` with the ticker→sector one-hot expanded;
    ``y_realized_sell_flipped`` mirrors train_scorer's SELL sign-flip so
    'good' has one meaning across actions. The aggregate rank-IC is then
    just ``_spearman(predictions, y_realized_sell_flipped)``.
    """
    from paper_trader.ml.decision_scorer import (
        _to_float, build_features, PRED_CLAMP_PCT,
    )

    X_raw: list[list[float]] = []
    y: list[float] = []
    actions: list[str] = []
    for r in rows:
        # Skip rows whose realized 5d return is missing/non-finite — the
        # rank-IC target requires a real number. Mirrors evaluate_scorer_oos
        # and _oos_rank_metrics which both drop these silently.
        fr = _finite_float(r.get("forward_return_5d"))
        if fr is None:
            continue
        # SELL sign-flip matches train_scorer / evaluate_scorer_oos /
        # _oos_rank_metrics so "good" has one meaning regardless of action.
        action = str(r.get("action") or "BUY").upper()
        if action == "SELL":
            fr = -fr
        # Symmetric label clamp — same constant the scorer trained on,
        # so the metric describes the same target space.
        fr = max(-PRED_CLAMP_PCT, min(PRED_CLAMP_PCT, fr))

        # build_features tolerates None (uses build_features defaults) so
        # we don't need to drop rows missing optional fields — the same
        # behaviour the gate sees in production.
        feat = build_features(
            _to_float(r.get("ml_score"), 0.0),
            r.get("rsi"), r.get("macd"),
            r.get("mom5"), r.get("mom20"),
            _to_float(r.get("regime_mult"), 1.0),
            str(r.get("ticker") or ""),
            vol_ratio=r.get("vol_ratio"),
            bb_pos=r.get("bb_position"),
            news_urgency=r.get("news_urgency"),
            news_article_count=r.get("news_article_count"),
        )
        X_raw.append([float(v) for v in feat])
        y.append(fr)
        actions.append(action)
    return (np.asarray(X_raw, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            actions)


def _predict_with_optional_ablation(
    model, scaler, X_raw: np.ndarray, zero_cols: tuple[int, ...] = ()
) -> np.ndarray | None:
    """Run ``model.predict`` on standardized features, optionally zeroing
    one or more standardized columns to ablate those features.

    Setting a standardized column to 0 is exactly equivalent to setting
    the raw feature to its training mean — the textbook "ablate to
    baseline" interpretation. The model architecture is unchanged; only
    its input for THIS prediction pass is modified.

    Returns the prediction vector (shape ``(n,)``) or None on any fault
    (the analyzer treats fault as "no prediction available" and skips
    that ablation — never crashes the loop).
    """
    try:
        if scaler is not None:
            # sklearn's StandardScaler.transform always returns a fresh
            # array, but test stubs / a future scaler that returns its
            # input untouched (the identity case) would let our zero-out
            # mutate the caller's matrix. Always copy on ablation to
            # guarantee purity (no behaviour change in production where
            # sklearn's copy already costs us nothing extra).
            X_s = np.array(scaler.transform(X_raw), dtype=np.float64, copy=True)
        else:
            X_s = X_raw.copy()
        for c in zero_cols:
            X_s[:, c] = 0.0
        preds = np.asarray(model.predict(X_s), dtype=np.float64)
        return preds
    except Exception:
        return None


def analyze(outcomes_path: "Path | str | None" = None,
            oos_only: bool = True) -> dict:
    """Per-feature inference-time ablation report.

    ``oos_only=True`` (default) runs against the temporal holdout via
    ``validation.split_outcomes_temporal`` — the trustworthy
    generalization slice. ``False`` uses the full accumulated corpus
    (useful when the OOS slice is too thin).

    Returns a JSON-safe dict:
    ``{status, verdict, n, slice, baseline_rank_ic, ablations:[
        {feature, rank_ic, delta, n}
      ], top_redundant:[...], top_load_bearing:[...], hint}``

    Never raises — every fault degrades to a verdict ladder row.
    """
    if outcomes_path is None:
        outcomes_path = (Path(__file__).resolve().parent.parent.parent
                         / "data" / "decision_outcomes.jsonl")
    path = Path(outcomes_path)

    rows = list(_iter_rows(path))
    slice_label = "full"
    if oos_only and rows:
        try:
            from paper_trader.validation import split_outcomes_temporal
            _, rows = split_outcomes_temporal(rows, oos_fraction=0.2)
            slice_label = "temporal_oos"
        except Exception as exc:
            return {
                "status": "error", "verdict": "INSUFFICIENT_DATA",
                "n": 0, "slice": "temporal_oos",
                "baseline_rank_ic": None,
                "ablations": [], "top_redundant": [], "top_load_bearing": [],
                "hint": f"temporal split unavailable: {type(exc).__name__}",
            }

    # Build the OOS feature matrix + realized vector. A row missing
    # forward_return_5d is dropped (no rank-IC target available).
    X_raw, y, _actions = _build_feature_matrix(rows)
    n = int(X_raw.shape[0])
    if n < MIN_OBS:
        return {
            "status": "insufficient_data",
            "verdict": "INSUFFICIENT_DATA",
            "n": n, "slice": slice_label,
            "baseline_rank_ic": None,
            "ablations": [], "top_redundant": [], "top_load_bearing": [],
            "hint": (f"only {n} OOS rows with finite forward_return_5d; "
                     f"need ≥{MIN_OBS}. Increase corpus or use --all."),
        }

    # Load the deployed scorer. We use its private _model / _scaler
    # because the ablation is a non-standard inference (zero a column in
    # standardized space). DecisionScorer's public predict() is a
    # convenience wrapper that doesn't expose the ablation hook.
    try:
        from paper_trader.ml.decision_scorer import DecisionScorer
        ds = DecisionScorer()
    except Exception as exc:
        return {
            "status": "scorer_unavailable",
            "verdict": "SCORER_UNTRAINED",
            "n": n, "slice": slice_label,
            "baseline_rank_ic": None,
            "ablations": [], "top_redundant": [], "top_load_bearing": [],
            "hint": f"DecisionScorer import failed: {type(exc).__name__}",
        }
    if not ds.is_trained or ds._model is None:
        return {
            "status": "scorer_untrained",
            "verdict": "SCORER_UNTRAINED",
            "n": n, "slice": slice_label,
            "baseline_rank_ic": None,
            "ablations": [], "top_redundant": [], "top_load_bearing": [],
            "hint": ("no deployed scorer pickle (or scorer failed to "
                     "load). Run the continuous loop to produce one."),
        }

    # Resolve feature indices at call time so a future FEATURE_NAMES
    # reorder in decision_scorer.py is caught — the canonical no-drift
    # discipline (mirrors MLP_CONFIG / deploy_audit).
    from paper_trader.ml.decision_scorer import FEATURE_NAMES, SECTORS

    # Build group → column indices map. Numeric features are 1:1 with
    # their FEATURE_NAMES index; the 7 sector one-hots collapse to a
    # single "sector" group (ablating one one-hot in isolation is
    # ill-defined when one row is one-hot to exactly one sector).
    groups: list[tuple[str, tuple[int, ...]]] = []
    for name in NUMERIC_FEATURES:
        # FEATURE_NAMES uses "bb_pos" exactly as we list above.
        try:
            idx = FEATURE_NAMES.index(name)
        except ValueError:
            continue
        groups.append((name, (idx,)))
    sector_idxs = tuple(
        i for i, fname in enumerate(FEATURE_NAMES)
        if fname.startswith("sector_")
    )
    if sector_idxs and len(sector_idxs) == len(SECTORS):
        groups.append(("sector", sector_idxs))

    # Baseline prediction — no ablation.
    base_preds = _predict_with_optional_ablation(ds._model, ds._scaler, X_raw)
    if base_preds is None:
        return {
            "status": "error",
            "verdict": "SCORER_UNTRAINED",
            "n": n, "slice": slice_label,
            "baseline_rank_ic": None,
            "ablations": [], "top_redundant": [], "top_load_bearing": [],
            "hint": "deployed scorer model.predict crashed on OOS features",
        }
    baseline_ic = _spearman(base_preds, y)
    if baseline_ic is None:
        return {
            "status": "baseline_degenerate",
            "verdict": "BASELINE_DEGENERATE",
            "n": n, "slice": slice_label,
            "baseline_rank_ic": None,
            "ablations": [], "top_redundant": [], "top_load_bearing": [],
            "hint": ("scorer baseline predictions tie (constant output) — "
                     "every ablation is identically no-op. Likely a "
                     "collapsed-quantile pickle or a single-class model."),
        }

    # Per-group ablation sweep.
    ablations: list[dict] = []
    for name, cols in groups:
        preds = _predict_with_optional_ablation(
            ds._model, ds._scaler, X_raw, zero_cols=cols)
        if preds is None:
            ablations.append({
                "feature": name, "rank_ic": None,
                "delta": None, "n": n,
            })
            continue
        ic = _spearman(preds, y)
        if ic is None:
            ablations.append({
                "feature": name, "rank_ic": None,
                "delta": None, "n": n,
            })
            continue
        ablations.append({
            "feature": name,
            "rank_ic": round(ic, 4),
            "delta": round(ic - baseline_ic, 4),
            "n": n,
        })

    # Partition by sign of delta. Positive delta = removal HELPS = the
    # feature was redundant / noisy. Negative delta = removal HURTS =
    # the feature was load-bearing.
    redundant = [a for a in ablations
                 if a["delta"] is not None and a["delta"] >= EDGE_TOL]
    load_bearing = [a for a in ablations
                    if a["delta"] is not None and a["delta"] <= -EDGE_TOL]
    redundant.sort(key=lambda a: -a["delta"])     # biggest help first
    load_bearing.sort(key=lambda a: a["delta"])   # biggest hurt first

    if redundant and load_bearing:
        verdict = "MIXED"
        hint = (
            f"baseline OOS rank-IC = {baseline_ic:+.4f}. "
            f"{len(redundant)} features are net-noise "
            f"(top: {redundant[0]['feature']} delta {redundant[0]['delta']:+.4f}) "
            f"and {len(load_bearing)} are load-bearing "
            f"(top: {load_bearing[0]['feature']} delta "
            f"{load_bearing[0]['delta']:+.4f}). The MLP is over-weighting "
            "some inputs and correctly relying on others — net effect is "
            "the documented MLP_NO_BETTER_THAN_TRIVIAL drift, but the "
            "actionable cut is to drop the redundant-flagged features and "
            "retrain to see if rank-IC consolidates."
        )
    elif redundant:
        verdict = "REDUNDANT_DETECTED"
        hint = (
            f"baseline OOS rank-IC = {baseline_ic:+.4f}. "
            f"{len(redundant)} features can be REMOVED without losing "
            f"rank skill (top: {redundant[0]['feature']} delta "
            f"{redundant[0]['delta']:+.4f}). The MLP is over-weighting "
            "noise — a smaller feature set should generalize equally or "
            "better. Concrete next step: drop the top-1 redundant feature "
            "from build_features and re-run baseline_compare."
        )
    elif load_bearing:
        verdict = "LOAD_BEARING_DETECTED"
        hint = (
            f"baseline OOS rank-IC = {baseline_ic:+.4f}. "
            f"{len(load_bearing)} features are LOAD-BEARING "
            f"(top: {load_bearing[0]['feature']} delta "
            f"{load_bearing[0]['delta']:+.4f}). Removing any one of "
            "them measurably hurts rank-IC — the gate's skill is "
            "concentrated there. NO redundancy: the feature set is at "
            "or near its minimum useful size."
        )
    else:
        verdict = "NO_SIGNIFICANT_EFFECT"
        hint = (
            f"baseline OOS rank-IC = {baseline_ic:+.4f}. Every ablation "
            f"delta is within ±{EDGE_TOL:.2f} of baseline — the OOS "
            "rank ordering is INVARIANT to which feature the model sees. "
            "This is the textbook fingerprint of a near-constant predictor "
            "or noise-only signal: the scorer is not using any single "
            "feature decisively. Pairs with the baseline_compare 'MLP "
            "carries no real edge' verdict."
        )

    return {
        "status": "ok",
        "verdict": verdict,
        "n": n,
        "slice": slice_label,
        "baseline_rank_ic": round(baseline_ic, 4),
        "ablations": ablations,
        "top_redundant": redundant[:3],
        "top_load_bearing": load_bearing[:3],
        "hint": hint,
    }


def _print_report(rep: dict) -> None:
    """Operator-readable table — per-feature rank-IC + delta vs baseline."""
    print(f"[feature_ablation] status={rep.get('status')} "
          f"verdict={rep.get('verdict')} n={rep.get('n')} "
          f"slice={rep.get('slice')}")
    b = rep.get("baseline_rank_ic")
    if b is not None:
        print(f"  baseline rank-IC: {b:+.4f}")
    abl = rep.get("ablations") or []
    if abl:
        print(f"  {'feature':<22}{'rank-IC':>10}{'delta':>10}")
        for a in abl:
            ic = a.get("rank_ic")
            d = a.get("delta")
            ic_s = "n/a" if ic is None else f"{ic:+.4f}"
            d_s = "n/a" if d is None else f"{d:+.4f}"
            print(f"  {a['feature']:<22}{ic_s:>10}{d_s:>10}")
    tr = rep.get("top_redundant") or []
    if tr:
        print("  Top redundant (removing helps):")
        for a in tr:
            print(f"    {a['feature']:<22} delta {a['delta']:+.4f}")
    tlb = rep.get("top_load_bearing") or []
    if tlb:
        print("  Top load-bearing (removing hurts):")
        for a in tlb:
            print(f"    {a['feature']:<22} delta {a['delta']:+.4f}")
    if rep.get("hint"):
        print(f"  hint: {rep['hint']}")


def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.feature_ablation",
        description="Per-feature inference-time ablation of the deployed "
                    "DecisionScorer — reports OOS rank-IC delta when each "
                    "feature is set to its training-mean baseline. "
                    "Read-only — never trains or writes.",
    )
    p.add_argument("--all", action="store_true",
                   help="Use the full accumulated corpus instead of the "
                        "temporal OOS slice. Useful when the OOS slice is "
                        "too thin (default OOS holdout is the last 20%).")
    p.add_argument("--path", default=None,
                   help="Override the decision_outcomes.jsonl path.")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of a table.")
    return p


def main(argv: list[str] | None = None) -> int:
    import sys

    args = _build_arg_parser().parse_args(
        sys.argv[1:] if argv is None else argv)

    rep = analyze(outcomes_path=args.path, oos_only=not args.all)

    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
    else:
        _print_report(rep)

    # Exit code: 2 on adverse verdicts so a shell pipeline can gate on
    # `$?` — same convention as host_guard / scorer_pickle_smoke /
    # failed_run_audit. The two adverse verdicts are REDUNDANT_DETECTED
    # (the MLP is fitting noise) and MIXED (some features are wasted);
    # LOAD_BEARING_DETECTED is the *desirable* state (every feature
    # carries signal) so exit 0 there. NO_SIGNIFICANT_EFFECT is also
    # adverse (the scorer has no skill ANYWHERE to ablate).
    bad = {"REDUNDANT_DETECTED", "MIXED", "NO_SIGNIFICANT_EFFECT",
           "BASELINE_DEGENERATE", "SCORER_UNTRAINED"}
    return 2 if rep.get("verdict") in bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

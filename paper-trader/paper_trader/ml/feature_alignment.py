"""Feature-alignment diagnostic — is the model weighting the right features?

The natural quant question every existing diagnostic structurally cannot
answer:

  * ``baseline_compare``  — raw ``ml_score`` has higher OOS rank-IC than
    the trained MLP (``MLP_WORSE_THAN_TRIVIAL`` verdict).
  * ``feature_importance``— answers WHICH features the MODEL relies on
    (permutation importance through ``scorer.predict``).
  * ``linear_probe``      — answers whether a LINEAR baseline beats the
    MLP and which features the linear baseline relies on.

  None of them ASK the orthogonal question: *for each input feature, what
  is the raw value's UNIVARIATE rank-IC against the realized 5d return,
  and does the model's mean-|first-layer-weight| importance MATCH that
  signal strength?*

That mismatch is the smoking gun a quant interpreting
``MLP_WORSE_THAN_TRIVIAL`` actually needs. Two distinct economic verdicts
hide inside the same "model loses to a constant":

  * WEIGHTED_NOISE   — the model places its largest weights on features
    with near-zero univariate IC. The architecture is *learning noise*;
    the feature is structurally non-predictive but the optimizer found
    spurious in-sample correlation. Action: reduce capacity / increase L2 /
    drop the feature.
  * IGNORED_SIGNAL   — the model places near-zero weights on features
    with non-trivial univariate IC. Real signal exists in the input but
    the architecture (StandardScaler + ReLU MLP + L2 alpha) is structurally
    failing to extract it. Action: revisit the inference pipeline (scaler
    centering for non-Gaussian inputs, MLP depth, alpha regularization).

  Both readings are actionable. Neither is detectable today.

Universally a single read on whether the WORSE-THAN-TRIVIAL state is a
data problem or a model problem — the most decisive question a quant
can ask before deciding to spend another retrain cycle. Verdict ladder
sorted by decreasing operational urgency:

  * WEIGHTED_NOISE     — at least one feature is top-N by model weight
    AND bottom-N by |univariate IC| (the model is learning noise).
  * IGNORED_SIGNAL     — at least one feature is top-N by |univariate IC|
    AND bottom-N by model weight (the model is ignoring signal).
  * ALIGNED            — the top-weighted feature is also the
    top-univariate-IC feature (model and data agree).
  * DEGENERATE         — all univariate |IC|s are < ``MIN_IC`` (no edge
    available in raw features for ANY model architecture).
  * INSUFFICIENT_DATA  — fewer than ``MIN_RECORDS`` OOS rows or scorer
    untrained.

Same operational discipline as every sibling ``ml/`` analyzer: read-only,
no train, no pickle write, never raises on bad input. Reuses
``calibration._spearman`` for tie-aware rank-IC (the
``_oos_rank_metrics`` precedent — load-bearing because the realized
target distribution has long tails) and ``validation.split_outcomes_temporal``
for the trustworthy OOS slice (the ``gate_audit`` / ``feature_importance``
default).

Wired into ``run_continuous_backtests.py::main()`` via
``_append_feature_alignment_log`` so a researcher reading
``MLP_WORSE_THAN_TRIVIAL`` cycle after cycle can see the architecture /
data verdict next to it.

CLI usage:

```bash
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.feature_alignment
```
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


MIN_RECORDS = 30           # mirrors calibration.MIN_PAIRS — need a stable baseline
MIN_IC = 0.03              # |univariate IC| < this counts as "no edge"
TOP_N = 3                  # top-N / bottom-N bucketing for the verdict ladder

# Module-level path so tests can redirect (mirrors every sibling analyzer).
DEFAULT_OUTCOMES_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "decision_outcomes.jsonl"
)

# The 10 base numeric + 3 enhanced MACD record fields. Each entry is
# (record_key_for_target_alignment, display_name). The 7-way sector
# one-hot is excluded — univariate IC of a per-sector binary is
# ill-defined (every row is 1 for exactly one sector, so 6 of 7 columns
# are constant-0 with one constant-1 in a fixed minority — Spearman tied
# at 0 for the constants). Aligns with ``feature_importance.FEATURES``
# excluding "sector".
NUMERIC_FEATURES: tuple[tuple[str, str], ...] = (
    ("ml_score", "ml_score"),
    ("rsi", "rsi"),
    ("macd", "macd"),
    ("mom5", "mom5"),
    ("mom20", "mom20"),
    ("regime_mult", "regime_mult"),
    ("vol_ratio", "vol_ratio"),
    ("bb_position", "bb_position"),
    ("news_urgency", "news_urgency"),
    ("news_article_count", "news_article_count"),
    ("ema200_above", "ema200_above"),
    ("hist_cross_up", "hist_cross_up"),
    ("macd_below_zero_cross", "macd_below_zero_cross"),
)


def _to_float(v) -> float | None:
    """Coerce v to a finite float or None. Mirrors `decision_scorer._to_float`
    discipline but returns None on missing/non-finite so the caller can
    drop a row rather than fabricate a zero (which would inflate the rank
    metric with fake ties)."""
    if v is None or isinstance(v, bool) and v is None:
        # bool branch above is dead but explicit; the canonical check is below.
        return None
    if isinstance(v, bool):
        # Treat booleans as numeric (0.0/1.0) — required for the 3 enhanced
        # MACD signals that come in as Python bools. Spearman on a binary
        # column is the textbook "rank-biserial" correlation; valid.
        return float(v)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank tie-aware rank transform. Mirrors
    ``feature_correlation_audit._rankdata`` so cross-analyzer rank-IC
    values are byte-identical for the same input — important when a
    researcher is comparing this analyzer's numbers to
    ``feature_importance`` / ``baseline_compare``."""
    a = np.asarray(a, dtype=np.float64)
    order = a.argsort()
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(len(a))
    # Ties → average rank.
    sorted_a = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i + 1
        while j < n and sorted_a[j] == sorted_a[i]:
            j += 1
        if j - i > 1:
            avg = ranks[order[i:j]].mean()
            ranks[order[i:j]] = avg
        i = j
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    """Tie-aware Spearman; None on degenerate input (zero variance or
    n<2). Returning None lets the caller mark the per-feature row
    DEGENERATE rather than fabricate a flat-zero IC that would mislead
    the verdict ladder."""
    if a.shape[0] != b.shape[0] or a.shape[0] < 2:
        return None
    ra = _rankdata(a)
    rb = _rankdata(b)
    # Pearson on ranks = Spearman.
    da = ra - ra.mean()
    db = rb - rb.mean()
    denom = float(np.sqrt((da * da).sum() * (db * db).sum()))
    if denom <= 0:
        return None
    return float((da * db).sum() / denom)


def _action_aligned_target(records: list[dict]) -> tuple[
        np.ndarray, list[int]]:
    """Return ``(targets, kept_indices)`` — the action-aligned realized
    5d return per surviving record. A SELL outcome has its target
    sign-flipped (mirrors ``_oos_rank_metrics`` / ``train_scorer``); a
    record with no finite ``forward_return_5d`` is dropped from
    ``kept_indices``."""
    targets: list[float] = []
    kept: list[int] = []
    for i, r in enumerate(records):
        raw = _to_float(r.get("forward_return_5d"))
        if raw is None:
            continue
        action = str(r.get("action") or "BUY").upper()
        targets.append(-raw if action == "SELL" else raw)
        kept.append(i)
    return np.asarray(targets, dtype=np.float64), kept


def _univariate_ic(records: list[dict], targets: np.ndarray,
                   kept: list[int], feature_key: str
                   ) -> tuple[float | None, int]:
    """Per-feature univariate Spearman IC against the action-aligned
    realized 5d return. Returns ``(ic, n)`` — ``ic`` is None on
    degenerate input (all-null column or zero variance), ``n`` is the
    count of rows that contributed a finite (feature, target) pair."""
    paired_x: list[float] = []
    paired_y: list[float] = []
    for kept_pos, src_idx in enumerate(kept):
        v = _to_float(records[src_idx].get(feature_key))
        if v is None:
            continue
        paired_x.append(v)
        paired_y.append(float(targets[kept_pos]))
    n = len(paired_x)
    if n < MIN_RECORDS:
        return None, n
    ic = _spearman(np.asarray(paired_x), np.asarray(paired_y))
    return ic, n


def _iter_rows(path: Path):
    """Yield parsed dicts from a JSONL file; skip malformed lines."""
    try:
        with path.open() as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    yield json.loads(ln)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return


def _model_importance(scorer) -> dict[str, float]:
    """Map feature name → mean-|first-layer-weight| from the deployed
    scorer's ``feature_importance()`` output. Returns an empty dict on
    any fault — the caller degrades that branch to None for every
    feature so the alignment columns honestly report "no importance
    available" rather than fabricating zeros."""
    try:
        out = scorer.feature_importance()
    except Exception:
        return {}
    if not isinstance(out, dict) or not out.get("trained"):
        return {}
    rows = out.get("importances") or []
    result: dict[str, float] = {}
    for r in rows:
        try:
            result[str(r["feature"])] = float(r["importance"])
        except (TypeError, ValueError, KeyError):
            continue
    return result


def _rank_bucket(name: str, ranks: dict[str, int], n_total: int,
                 top_n: int) -> str:
    """Classify a feature's rank position as 'top' / 'mid' / 'bottom'
    given a `name → rank-index` map (0 = highest score). 'top' = top_n
    by score, 'bottom' = bottom_n. n_total is the count of features
    that produced a finite score."""
    idx = ranks.get(name)
    if idx is None:
        return "mid"
    if idx < top_n:
        return "top"
    if idx >= n_total - top_n:
        return "bottom"
    return "mid"


def analyze(outcomes_path: "Path | str | None" = None,
            oos_only: bool = True) -> dict:
    """Per-feature univariate-IC vs model-importance alignment diagnostic.

    ``oos_only=True`` (default) runs against the temporal holdout via
    ``validation.split_outcomes_temporal``; ``False`` uses the full
    accumulated corpus.

    Returns a JSON-safe dict::

        {
          "status": "ok" | "insufficient_data" | "error",
          "verdict": "WEIGHTED_NOISE" | "IGNORED_SIGNAL" | "ALIGNED"
                     | "DEGENERATE" | "INSUFFICIENT_DATA",
          "n": int,
          "slice": "temporal_oos" | "full",
          "features": [
            {"feature", "univariate_ic", "model_importance",
             "univariate_rank", "importance_rank",
             "alignment_bucket": "WEIGHTED_NOISE" | "IGNORED_SIGNAL"
                                  | "ALIGNED" | "MID" | "DEGENERATE",
             "n"},
            ...
          ],
          "n_features_with_signal": int,    # |IC| ≥ MIN_IC
          "top_weighted_noise": [str, ...], # feature names
          "top_ignored_signal": [str, ...],
          "hint": str,
        }

    Never raises — every fault degrades to a verdict ladder row.
    """
    if outcomes_path is None:
        outcomes_path = DEFAULT_OUTCOMES_PATH
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
                "features": [], "n_features_with_signal": 0,
                "top_weighted_noise": [], "top_ignored_signal": [],
                "hint": f"temporal split unavailable: {type(exc).__name__}",
            }

    if len(rows) < MIN_RECORDS:
        return {
            "status": "insufficient_data",
            "verdict": "INSUFFICIENT_DATA",
            "n": len(rows), "slice": slice_label,
            "features": [], "n_features_with_signal": 0,
            "top_weighted_noise": [], "top_ignored_signal": [],
            "hint": (f"only {len(rows)} rows; need ≥{MIN_RECORDS}. "
                     "accumulate more decision_outcomes."),
        }

    targets, kept = _action_aligned_target(rows)
    if targets.shape[0] < MIN_RECORDS:
        return {
            "status": "insufficient_data",
            "verdict": "INSUFFICIENT_DATA",
            "n": int(targets.shape[0]), "slice": slice_label,
            "features": [], "n_features_with_signal": 0,
            "top_weighted_noise": [], "top_ignored_signal": [],
            "hint": (f"only {targets.shape[0]} rows have finite "
                     f"forward_return_5d; need ≥{MIN_RECORDS}."),
        }

    # Load the deployed scorer for model-importance lookups. A missing
    # pickle degrades model_importance to {} so the analyzer still
    # produces the univariate side (which doesn't depend on the model).
    try:
        from paper_trader.ml.decision_scorer import DecisionScorer
        scorer = DecisionScorer()
        importance_map = _model_importance(scorer)
    except Exception:
        importance_map = {}

    # 1) Compute per-feature univariate IC + n.
    per_feature: list[dict] = []
    for rkey, display in NUMERIC_FEATURES:
        ic, n = _univariate_ic(rows, targets, kept, rkey)
        # Model importance lives in FEATURE_NAMES space. The display
        # name matches: NUMERIC_FEATURES uses the same names
        # decision_scorer.FEATURE_NAMES exposes for the numeric block
        # (no "sector_*" entries here). ``bb_position`` ↔ ``bb_pos``
        # is the one renaming — handle it explicitly.
        importance_key = "bb_pos" if display == "bb_position" else display
        weight = importance_map.get(importance_key)
        per_feature.append({
            "feature": display,
            "univariate_ic": (round(ic, 4)
                              if ic is not None else None),
            "model_importance": (round(weight, 6)
                                 if weight is not None else None),
            "n": int(n),
        })

    # 2) Rank by |univariate IC| desc (None → bottom) and by importance
    # desc (None → bottom). The bucket classification compares the
    # feature's rank position in each.
    finite_ic = [r for r in per_feature
                 if r["univariate_ic"] is not None]
    finite_imp = [r for r in per_feature
                  if r["model_importance"] is not None]
    ic_order = sorted(finite_ic,
                      key=lambda r: -abs(r["univariate_ic"]))
    imp_order = sorted(finite_imp,
                       key=lambda r: -(r["model_importance"] or 0.0))
    ic_rank = {r["feature"]: i for i, r in enumerate(ic_order)}
    imp_rank = {r["feature"]: i for i, r in enumerate(imp_order)}

    n_ic = len(ic_order)
    n_imp = len(imp_order)
    for r in per_feature:
        r["univariate_rank"] = ic_rank.get(r["feature"])
        r["importance_rank"] = imp_rank.get(r["feature"])
        # 3) Classify alignment per feature. A feature must have BOTH
        # an IC and an importance to be classified — otherwise its
        # alignment is structurally undefined (degenerate column).
        if (r["univariate_ic"] is None
                or r["model_importance"] is None):
            r["alignment_bucket"] = "DEGENERATE"
            continue
        ic_pos = _rank_bucket(r["feature"], ic_rank, n_ic, TOP_N)
        imp_pos = _rank_bucket(r["feature"], imp_rank, n_imp, TOP_N)
        if imp_pos == "top" and ic_pos == "bottom":
            r["alignment_bucket"] = "WEIGHTED_NOISE"
        elif ic_pos == "top" and imp_pos == "bottom":
            r["alignment_bucket"] = "IGNORED_SIGNAL"
        elif ic_pos == "top" and imp_pos == "top":
            r["alignment_bucket"] = "ALIGNED"
        else:
            r["alignment_bucket"] = "MID"

    # 4) Compute the overall verdict + actionable hint.
    n_signal = sum(1 for r in per_feature
                   if r["univariate_ic"] is not None
                   and abs(r["univariate_ic"]) >= MIN_IC)
    top_weighted_noise = [r["feature"] for r in per_feature
                          if r["alignment_bucket"] == "WEIGHTED_NOISE"]
    top_ignored_signal = [r["feature"] for r in per_feature
                          if r["alignment_bucket"] == "IGNORED_SIGNAL"]

    if n_signal == 0:
        verdict = "DEGENERATE"
        hint = (f"no feature has |univariate IC| ≥ {MIN_IC}; raw "
                "inputs carry essentially no edge against realized 5d "
                "return — no model architecture can extract signal "
                "that isn't there.")
    elif top_weighted_noise:
        verdict = "WEIGHTED_NOISE"
        hint = (f"top-weighted features {top_weighted_noise} carry "
                f"bottom-quartile univariate IC — the MLP is learning "
                "spurious in-sample correlation. Consider raising L2 "
                "`alpha`, reducing capacity, or dropping the feature.")
    elif top_ignored_signal:
        verdict = "IGNORED_SIGNAL"
        hint = (f"features {top_ignored_signal} carry top-quartile "
                f"univariate IC but bottom-quartile model weight — "
                "the MLP architecture (StandardScaler + ReLU + L2) is "
                "failing to extract real signal. Inspect the scaler / "
                "depth / alpha tradeoff.")
    else:
        # No strong mismatches. Decide ALIGNED vs MID by checking the
        # top-IC feature's importance rank.
        if ic_order and ic_order[0]["feature"] in {
                r["feature"] for r in imp_order[:TOP_N]}:
            verdict = "ALIGNED"
            hint = (f"top-IC feature {ic_order[0]['feature']!r} is also "
                    "top-weighted — model and data agree on what carries "
                    "signal.")
        else:
            verdict = "ALIGNED"  # no actionable mismatch
            hint = ("no strong feature-alignment mismatches detected; "
                    f"{n_signal}/{len(per_feature)} features carry "
                    f"|univariate IC| ≥ {MIN_IC}.")

    return {
        "status": "ok",
        "verdict": verdict,
        "n": int(targets.shape[0]),
        "slice": slice_label,
        "features": per_feature,
        "n_features_with_signal": n_signal,
        "top_weighted_noise": top_weighted_noise,
        "top_ignored_signal": top_ignored_signal,
        "hint": hint,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry — prints a table and exits 0 on ALIGNED, 1 otherwise.

    Run with --all to use the full corpus instead of the temporal OOS
    slice (helpful when the OOS slice is too thin).
    """
    import argparse
    import sys

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.feature_alignment",
        description="Per-feature univariate-IC vs model-importance "
                    "alignment — is the scorer weighting the right "
                    "features?",
    )
    p.add_argument("--all", action="store_true",
                   help="Use full corpus instead of the temporal OOS slice.")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON.")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    rep = analyze(oos_only=not args.all)

    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
    else:
        print(f"[feature_alignment] verdict={rep['verdict']}  "
              f"n={rep['n']}  slice={rep['slice']}")
        print(f"  features with |IC|≥{MIN_IC}: "
              f"{rep['n_features_with_signal']}/"
              f"{len(rep['features'])}")
        if rep["features"]:
            print(f"  {'feature':<24}{'univariate_ic':>14}"
                  f"{'model_weight':>14}{'ic_rank':>10}"
                  f"{'w_rank':>10}{'bucket':>20}")
            for r in rep["features"]:
                ic = (f"{r['univariate_ic']:+.3f}"
                      if r["univariate_ic"] is not None else "n/a")
                w = (f"{r['model_importance']:.4f}"
                     if r["model_importance"] is not None else "n/a")
                ir = (str(r["univariate_rank"])
                      if r["univariate_rank"] is not None else "n/a")
                wr = (str(r["importance_rank"])
                      if r["importance_rank"] is not None else "n/a")
                print(f"  {r['feature']:<24}{ic:>14}{w:>14}"
                      f"{ir:>10}{wr:>10}{r['alignment_bucket']:>20}")
        print(f"  hint: {rep['hint']}")
    return 0 if rep["verdict"] == "ALIGNED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Per-feature-value *scorer* skill diagnostic — does the DecisionScorer's
OOS rank skill vary across quantile buckets of a feature the model trains on?

This answers the one quant-decisive question the existing 35-module diagnostic
suite structurally cannot:

  *Is the scorer's near-zero OOS rank-IC uniform across (e.g.) momentum levels,
  or is the model essentially a one-feature gate with extra steps?*

Existing siblings slice OOS skill by news volume, sector, regime (SPY
bull/bear/sideways), persona, leveraged-vs-not, action, and ticker. None
of them buckets by a value of a feature **the scorer actually trains on**.
``baseline_compare`` already shows that the MLP either does or doesn't beat
single-feature rules in aggregate — but it can't tell us *where* the model's
edge lives. If rank-IC is concentrated only in the extreme-momentum bucket,
the 17-feature MLP is acting like a momentum gate (with extra steps). If
rank-IC is uniform across momentum buckets, the model genuinely combines
features. Either answer is operationally decisive.

The feature is configurable: ``mom5``, ``vol_ratio``, ``bb_position``, or
``ml_score``. Each is a feature the model trains on, with empirically
meaningful and continuously distributed values across the corpus. Defaults
to ``mom5`` (the strongest single-feature predictor that ``baseline_compare``
evaluates).

Bucketing uses **empirical quintiles** of the chosen feature over the slice
under analysis — that way each bucket gets ~equal mass, and a feature whose
distribution is heavily zero-inflated (vol_ratio, ml_score) still produces a
useful per-bucket comparison instead of one bucket dominating.

Operational discipline is identical to ``news_volume_skill`` /
``sector_skill`` siblings: **read-only**, never trains, never raises on bad
input, never touches the trade path. Reuses ``calibration._spearman`` (the
single source of truth for tie-aware rank correlation) and
``decision_scorer._to_float`` so this metric can never drift from
``_oos_rank_metrics`` / per-cycle skill ledger.

Verdict per bucket (crisp, threshold-driven so it is exactly testable):

| Verdict | Meaning |
|---------|---------|
| `INSUFFICIENT` | < ``MIN_OUTCOMES_PER_BUCKET`` aligned outcomes — no stable IC |
| `INVERTED` | ``rank_ic ≤ -IC_GOOD`` — predictions anti-correlated with realized |
| `EDGE` | ``rank_ic ≥ IC_GOOD`` — predictions genuinely rank-predict realized |
| `WEAK_EDGE` | ``IC_MIN ≤ rank_ic < IC_GOOD`` — usable as a tie-breaker, not primary |
| `NO_EDGE` | ``-IC_GOOD < rank_ic < IC_MIN`` — no demonstrated rank skill |

Overall verdict (across all sufficient quintile buckets):

| Verdict | Meaning |
|---------|---------|
| `HAS_INVERTED_BUCKET` | ≥1 bucket INVERTED — actionable red flag |
| `LOCALIZED_EDGE` | only 1 bucket has EDGE — scorer is essentially a gate on this feature |
| `UNIFORM_EDGE` | 3+ buckets have EDGE, spread < BUCKET_SPREAD_TOL — generalizes |
| `MIXED_EDGE` | 2+ buckets have EDGE, but with significant spread — partial skill |
| `NO_EDGE_ANY` | no bucket clears IC_MIN — the slice carries no rank skill |
| `INSUFFICIENT_DATA` | < ``MIN_RECORDS`` aligned outcomes or <2 sufficient buckets |

```bash
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.feature_value_skill
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.feature_value_skill --feature vol_ratio
cd /home/zeph/paper-trader && python3 -m pytest tests/test_feature_value_skill.py -v
```
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from paper_trader.ml.calibration import _spearman
from paper_trader.ml.decision_scorer import _to_float

# Thresholds — module-level so tests assert exact verdicts and a tuning
# change is one reviewable edit (mirrors news_volume_skill / sector_skill,
# the codebase constants-at-module-scope convention).
MIN_RECORDS = 30
MIN_OUTCOMES_PER_BUCKET = 15
IC_MIN = 0.05
IC_GOOD = 0.10
# A bucket-to-bucket rank-IC spread under this magnitude is "uniform";
# above it the buckets are visibly differentiated. 0.05 = one full
# rank-skill bar, mirroring news_volume_skill.BUCKET_SPREAD_TOL.
BUCKET_SPREAD_TOL = 0.05

# The features supported. Each must:
#   - Appear in build_features's input list (already a scorer feature)
#   - Be a numeric column in decision_outcomes.jsonl rows
#   - Carry a meaningful continuous distribution worth bucketing
#
# Maps the user-facing flag to the decision_outcomes.jsonl column name.
# `bb_position` is the JSONL column; `bb_pos` is the build_features kwarg
# name (kept consistent with the sibling skills that read the JSONL).
FEATURES: dict[str, str] = {
    "mom5":        "mom5",         # 5-day momentum % (most predictive single feature)
    "vol_ratio":   "vol_ratio",    # today's vol / 20d avg
    "bb_position": "bb_position",  # Bollinger band position [-2, 2]
    "ml_score":    "ml_score",     # ArticleNet ai_score for the name
}

# Bucket labels in canonical order — printed in this order and walked by
# `_overall_verdict`. Quintile-based so each gets ~20% mass on the slice.
BUCKET_NAMES: tuple[str, ...] = ("q1_low", "q2", "q3_mid", "q4", "q5_high")


def _safe_float(v) -> float | None:
    """Coerce v to a finite float or return None — single source for the
    feature-value parse so quintile breakpoints can't drift from per-row
    bucketing."""
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x != x:                       # NaN
        return None
    if x == float("inf") or x == float("-inf"):
        return None
    return x


def _quintile_breakpoints(values: list[float]) -> list[float]:
    """4 internal breakpoints partitioning ``values`` into 5 quantile
    buckets (q1..q5). Uses linear interpolation via ``np.quantile`` so
    the result is bit-deterministic for a given input array.

    Returns 4 floats in ascending order. Caller pairs these with the
    ``BUCKET_NAMES`` tuple to assign a bucket to each new value.
    """
    if len(values) < 5:
        # Degenerate input — return 4 copies of the median so every value
        # lands in q3_mid and the per-bucket pass falls through to
        # INSUFFICIENT for all other buckets. Better than raising.
        med = float(np.median(values)) if values else 0.0
        return [med, med, med, med]
    arr = np.asarray(values, dtype=np.float64)
    qs = np.quantile(arr, [0.2, 0.4, 0.6, 0.8])
    return [float(q) for q in qs]


def _bucket_for(value: float, breakpoints: list[float]) -> str:
    """Return the bucket name for one value given the 4 quintile
    breakpoints. Pure, total, never raises. A value equal to a breakpoint
    falls into the LOWER bucket (canonical `np.quantile`-aligned convention)."""
    if value <= breakpoints[0]:
        return BUCKET_NAMES[0]
    if value <= breakpoints[1]:
        return BUCKET_NAMES[1]
    if value <= breakpoints[2]:
        return BUCKET_NAMES[2]
    if value <= breakpoints[3]:
        return BUCKET_NAMES[3]
    return BUCKET_NAMES[4]


def _aligned_pred(scorer, record: dict) -> tuple[float, float] | None:
    """Return (prediction, action-aligned realized return) for one outcome
    record, or ``None`` when the record is unusable.

    Mirrors ``news_volume_skill._aligned_pred`` / ``action_skill._aligned_pred``
    EXACTLY (same SELL sign-flip convention, NaN-sentinel target drop,
    scorer-exception → drop). The single source of truth for "scorer
    prediction aligned to outcome" across this diagnostic suite.
    """
    fr = record.get("forward_return_5d")
    if fr is None:
        return None
    t = _to_float(fr, float("nan"))
    if t != t:
        return None
    try:
        p = scorer.predict(
            ml_score=_to_float(record.get("ml_score"), 0.0),
            rsi=record.get("rsi"), macd=record.get("macd"),
            mom5=record.get("mom5"), mom20=record.get("mom20"),
            regime_mult=_to_float(record.get("regime_mult"), 1.0),
            ticker=str(record.get("ticker") or ""),
            vol_ratio=record.get("vol_ratio"),
            bb_pos=record.get("bb_position"),
            news_urgency=record.get("news_urgency"),
            news_article_count=record.get("news_article_count"),
            ema200_above=record.get("ema200_above"),
            hist_cross_up=record.get("hist_cross_up"),
            macd_below_zero_cross=record.get("macd_below_zero_cross"),
        )
    except Exception:
        return None
    try:
        p = float(p)
    except (TypeError, ValueError):
        return None
    if p != p:
        return None
    if str(record.get("action") or "BUY").upper() == "SELL":
        t = -t
    return p, t


def _verdict_for(ic: float | None, n: int) -> str:
    """Per-bucket verdict for one (ic, n) pair. Pure, deterministic,
    threshold-driven so tests can assert exact strings (mirrors the
    sibling skills' contract)."""
    if n < MIN_OUTCOMES_PER_BUCKET:
        return "INSUFFICIENT"
    if ic is None or ic != ic:
        return "INSUFFICIENT"
    if ic <= -IC_GOOD:
        return "INVERTED"
    if ic >= IC_GOOD:
        return "EDGE"
    if ic >= IC_MIN:
        return "WEAK_EDGE"
    return "NO_EDGE"


def _overall_verdict(by_bucket: dict[str, dict], feature: str) -> tuple[str, str]:
    """Combine per-bucket verdicts into one ``(verdict, hint)`` pair.

    The actionable signals (HAS_INVERTED_BUCKET first, then LOCALIZED_EDGE)
    are surfaced ahead of "uniform" verdicts so they cannot be missed
    under a healthy-looking aggregate.
    """
    # Same actionable-red-flag-first discipline as news_volume_skill.
    inverted = [b for b in BUCKET_NAMES
                if by_bucket[b]["verdict"] == "INVERTED"]
    if inverted:
        return ("HAS_INVERTED_BUCKET",
                f"{','.join(inverted)} quintile(s) of '{feature}' have "
                f"rank_ic ≤ -{IC_GOOD} — the scorer is anti-predictive in "
                f"this slice. Treat as actionable, not a tuning detail.")

    sufficient = [b for b in BUCKET_NAMES
                  if by_bucket[b]["verdict"] != "INSUFFICIENT"
                  and by_bucket[b]["rank_ic"] is not None]
    if len(sufficient) < 2:
        return ("INSUFFICIENT_DATA",
                f"< 2 quintiles of '{feature}' reach the per-bucket "
                f"minimum ({MIN_OUTCOMES_PER_BUCKET}) — cannot assess "
                f"per-value skill.")

    edges = [b for b in sufficient
             if by_bucket[b]["verdict"] == "EDGE"]
    n_edges = len(edges)

    if n_edges == 0:
        return ("NO_EDGE_ANY",
                f"no quintile of '{feature}' clears rank_ic ≥ {IC_MIN}. "
                f"The slice carries no demonstrated rank skill — the "
                f"scorer's aggregate IC may be noise on this slice.")

    if n_edges == 1:
        edge_bucket = edges[0]
        edge_ic = by_bucket[edge_bucket]["rank_ic"]
        return ("LOCALIZED_EDGE",
                f"only quintile '{edge_bucket}' has rank_ic ≥ {IC_GOOD} "
                f"({edge_ic:+.3f}). The scorer is essentially a gate on "
                f"'{feature}' — its edge lives in ONE bucket, suggesting "
                f"a single-feature rule could replicate the skill. Run "
                f"`python3 -m paper_trader.ml.baseline_compare` to confirm.")

    ics = [by_bucket[b]["rank_ic"] for b in sufficient]
    spread = max(ics) - min(ics)
    if n_edges >= 3 and spread < BUCKET_SPREAD_TOL:
        return ("UNIFORM_EDGE",
                f"{n_edges} of {len(sufficient)} sufficient quintiles of "
                f"'{feature}' have rank_ic ≥ {IC_GOOD}, with a spread of "
                f"only {spread:.3f}. The scorer genuinely generalizes "
                f"across '{feature}' values — multi-feature combination "
                f"is doing real work.")
    return ("MIXED_EDGE",
            f"{n_edges} of {len(sufficient)} sufficient quintiles have "
            f"EDGE but with spread {spread:.3f} across all buckets — the "
            f"scorer has edge in part of '{feature}'s range but not "
            f"uniformly. The per-bucket table is the honest read.")


def feature_value_skill(scorer, records, feature: str = "mom5") -> dict:
    """Per-feature-value OOS rank skill of a deployed scorer over outcomes.

    ``records`` is any iterable of dicts with at least ``action``,
    ``ml_score``, ``forward_return_5d``, plus the chosen ``feature``
    column. Rows with missing/non-finite feature value or forward return,
    an untrained scorer, or a predict fault are dropped.

    Returns a JSON-safe dict::

        {
          "status": "ok" | "untrained" | "insufficient_data" | "error",
          "feature": <feature name>,
          "verdict": <overall verdict string>,
          "n_records": int,
          "breakpoints": [q0.2, q0.4, q0.6, q0.8],
          "by_bucket": {bucket_name: {n, rank_ic, dir_acc,
                                       mean_aligned_return, verdict}},
          "hint": str,
        }
    """
    col = FEATURES.get(feature)
    out_skel: dict = {
        "status": "ok",
        "feature": feature,
        "verdict": "INSUFFICIENT_DATA",
        "n_records": 0,
        "breakpoints": [0.0, 0.0, 0.0, 0.0],
        "by_bucket": {b: {"n": 0, "rank_ic": None, "dir_acc": None,
                          "mean_aligned_return": None,
                          "verdict": "INSUFFICIENT"} for b in BUCKET_NAMES},
        "hint": "",
    }

    if col is None:
        out_skel["status"] = "error"
        out_skel["hint"] = (f"unknown feature {feature!r}. Valid: "
                            f"{sorted(FEATURES.keys())}")
        return out_skel

    if not getattr(scorer, "is_trained", False):
        out_skel["status"] = "untrained"
        out_skel["hint"] = ("scorer is not trained — no predictions to "
                            "evaluate. Train the scorer first.")
        return out_skel

    # First pass: collect feature values + aligned (pred, target) — drop
    # rows that don't have BOTH a feature value AND a scorer prediction.
    pre: list[tuple[float, float, float]] = []
    for r in records:
        fv = _safe_float(r.get(col))
        if fv is None:
            continue
        pair = _aligned_pred(scorer, r)
        if pair is None:
            continue
        p, t = pair
        pre.append((fv, p, t))

    n_aligned = len(pre)
    if n_aligned < MIN_RECORDS:
        out_skel["n_records"] = n_aligned
        out_skel["status"] = "insufficient_data"
        out_skel["hint"] = (f"need ≥{MIN_RECORDS} aligned outcomes with a "
                            f"finite '{feature}' and forward_return_5d, "
                            f"have {n_aligned}")
        return out_skel

    # Compute quintile breakpoints on the SLICE itself — that way each
    # bucket gets ~equal mass even when the feature distribution is
    # zero-inflated or one-sided.
    breakpoints = _quintile_breakpoints([fv for fv, _, _ in pre])

    buckets: dict[str, dict] = {b: {"sig": [], "tgt": []}
                                for b in BUCKET_NAMES}
    for fv, p, t in pre:
        bname = _bucket_for(fv, breakpoints)
        buckets[bname]["sig"].append(p)
        buckets[bname]["tgt"].append(t)

    by_bucket: dict[str, dict] = {}
    for bname in BUCKET_NAMES:
        b = buckets[bname]
        n = len(b["sig"])
        if n == 0:
            by_bucket[bname] = {"n": 0, "rank_ic": None, "dir_acc": None,
                                "mean_aligned_return": None,
                                "verdict": "INSUFFICIENT"}
            continue
        sig = np.asarray(b["sig"], dtype=np.float64)
        tgt = np.asarray(b["tgt"], dtype=np.float64)
        ic: float | None
        if n >= 2:
            ic_raw = float(_spearman(sig, tgt))
            ic = round(ic_raw, 4) if ic_raw == ic_raw else None
        else:
            ic = None
        dir_pairs = [(p, t) for p, t in zip(sig.tolist(), tgt.tolist())
                     if p != 0.0 and t != 0.0]
        if dir_pairs:
            hits = sum(1 for p, t in dir_pairs if (p > 0) == (t > 0))
            dir_acc: float | None = round(hits / len(dir_pairs), 4)
        else:
            dir_acc = None
        mean_ret = round(float(tgt.mean()), 4) if n else None
        verdict = _verdict_for(ic, n)
        by_bucket[bname] = {
            "n": n, "rank_ic": ic, "dir_acc": dir_acc,
            "mean_aligned_return": mean_ret, "verdict": verdict,
        }

    verdict, hint = _overall_verdict(by_bucket, feature)
    return {
        "status": "ok",
        "feature": feature,
        "verdict": verdict,
        "n_records": n_aligned,
        "breakpoints": [round(b, 4) for b in breakpoints],
        "by_bucket": by_bucket,
        "hint": hint,
    }


def _load_outcomes(path: Path) -> list[dict]:
    """Robust JSONL load — same best-effort discipline as
    ``news_volume_skill._load_outcomes``."""
    rows: list[dict] = []
    try:
        if not path.exists():
            return rows
        for ln in path.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception:
                continue
    except Exception:
        return rows
    return rows


def analyze(outcomes_path: "Path | str | None" = None,
            feature: str = "mom5",
            oos_only: bool = True) -> dict:
    """End-to-end CLI/import entrypoint: load outcomes, load the deployed
    scorer, run ``feature_value_skill``.

    ``oos_only`` (default True) limits the analysis to the most recent
    20% of records via ``validation.split_outcomes_temporal`` — the SAME
    holdout `_train_decision_scorer` reports `oos_rmse`/`oos_ic` on.
    """
    root = Path(__file__).resolve().parent.parent.parent
    if outcomes_path is None:
        outcomes_path = root / "data" / "decision_outcomes.jsonl"
    records = _load_outcomes(Path(outcomes_path))

    slice_label = "all"
    if oos_only and records:
        try:
            from paper_trader.validation import split_outcomes_temporal
            _, oos = split_outcomes_temporal(records, oos_fraction=0.2)
            if oos:
                records = oos
                slice_label = "oos"
        except Exception:
            pass

    try:
        from paper_trader.ml.decision_scorer import DecisionScorer
        scorer = DecisionScorer()
    except Exception as e:
        out = {
            "status": "error",
            "feature": feature,
            "verdict": "INSUFFICIENT_DATA",
            "n_records": 0,
            "breakpoints": [0.0, 0.0, 0.0, 0.0],
            "by_bucket": {b: {"n": 0, "rank_ic": None, "dir_acc": None,
                              "mean_aligned_return": None,
                              "verdict": "INSUFFICIENT"}
                          for b in BUCKET_NAMES},
            "hint": f"scorer load failed: {type(e).__name__}",
            "slice": slice_label,
        }
        return out

    rep = feature_value_skill(scorer, records, feature=feature)
    rep["slice"] = slice_label
    return rep


def _cli(argv: list[str] | None = None) -> int:
    """`python3 -m paper_trader.ml.feature_value_skill [--feature mom5]`.

    Read-only; never writes. Exit 0 healthy / insufficient, 2 if any
    bucket is INVERTED (so an operator/cron can branch on it).
    """
    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.feature_value_skill",
        description="Per-feature-value OOS rank skill of the deployed "
                    "DecisionScorer over decision_outcomes.jsonl.",
    )
    p.add_argument("--feature", choices=sorted(FEATURES.keys()),
                   default="mom5",
                   help="Feature to bucket on (default: mom5).")
    p.add_argument("--no-oos-only", action="store_true",
                   help="Use all records, not just the OOS holdout.")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of a table.")
    args = p.parse_args(argv)

    rep = analyze(feature=args.feature, oos_only=not args.no_oos_only)
    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
        return 0 if rep["verdict"] != "HAS_INVERTED_BUCKET" else 2

    print(f"slice={rep.get('slice', 'all')}  feature={rep['feature']}  "
          f"aligned_outcomes={rep['n_records']}")
    bps = rep.get("breakpoints") or []
    if bps:
        print(f"  quintile breakpoints (q20/q40/q60/q80): "
              f"{bps[0]:+.3f} | {bps[1]:+.3f} | "
              f"{bps[2]:+.3f} | {bps[3]:+.3f}")
    print(f"VERDICT: {rep['verdict']}  ({rep['hint']})")
    print(f"  {'bucket':<8} {'n':>6} {'rank_ic':>9} {'dir_acc':>8} "
          f"{'mean_ret':>9}  verdict")
    for b in BUCKET_NAMES:
        e = rep["by_bucket"][b]
        ic_s = (f"{e['rank_ic']:>+9.3f}" if e["rank_ic"] is not None
                else f"{'n/a':>9}")
        da_s = (f"{e['dir_acc']:>8.3f}" if e["dir_acc"] is not None
                else f"{'n/a':>8}")
        mr_s = (f"{e['mean_aligned_return']:>+9.2f}"
                if e["mean_aligned_return"] is not None else f"{'n/a':>9}")
        print(f"  {b:<8} {e['n']:>6} {ic_s} {da_s} {mr_s}  {e['verdict']}")
    return 0 if rep["verdict"] != "HAS_INVERTED_BUCKET" else 2


if __name__ == "__main__":
    raise SystemExit(_cli())

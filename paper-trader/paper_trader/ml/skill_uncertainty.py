"""Bootstrap confidence intervals on the DecisionScorer's OOS skill metrics.

This is a **read-only diagnostic**. It never trains, never touches
`decision_scorer.pkl`, `decision_outcomes.jsonl`, `build_features`,
`N_FEATURES`, or any trade path — same operational discipline as
`paper_trader/ml/calibration.py` / `gate_audit.py` / `skill_trend.py`.

**Why this is not any existing tool.** Every existing OOS diagnostic
(`_oos_rank_metrics`, `_oos_multi_horizon_metrics`, `evaluate_scorer_oos`,
`calibration --oos`, `skill_trend`) reports the OOS rank-IC / RMSE /
dir_acc as a **single point estimate** per cycle. With cycle-to-cycle
OOS sample sizes in the hundreds-to-low-thousands and a near-zero
underlying skill, a single number cannot tell a skeptical quant whether
the reported 0.04 rank-IC is:

  (a) a real-but-weak signal — useful information, or
  (b) statistical noise around 0.0 — gating capital on a coin flip.

`skill_trend` answers the related question "is the rolling MEDIAN drifting
across cycles" but does not address per-cycle uncertainty: a single 0.10
cycle in a noisy sequence is ambiguous without a confidence band. The
documented finding `MLP_WORSE_THAN_TRIVIAL` (the MLP's OOS rank-IC trails
raw `ml_score`'s baseline) is itself a *point estimate* — knowing whether
that gap is significant or noise is a quant-decisive question, not a
cosmetic refinement.

**Method.** Standard nonparametric percentile bootstrap (Efron):

1. Predict every OOS record once with the deployed scorer (single pass —
   the slow part, not in the bootstrap loop).
2. Resample the (pred, realized) pairs *with replacement* B times.
3. Recompute rank-IC / RMSE / dir_acc on each resample.
4. Report the 2.5th / 97.5th percentiles as a 95% CI.
5. Verdict: ``SKILL_DETECTED`` when the rank-IC CI excludes 0,
   ``NO_SKILL_DETECTED`` when it straddles 0, ``INSUFFICIENT_DATA`` below
   ``MIN_OOS``.

The single source of truth for the predict signature, the SELL sign-flip,
and the tie-aware Spearman is the exact same path `_oos_rank_metrics` /
`evaluate_scorer_oos` use — `validation.split_outcomes_temporal` for the
holdout, `_to_float` for the target parse, `calibration._spearman` for the
rank-IC — so this module and the ledger's scalar metrics can never drift.

```bash
cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.skill_uncertainty
cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.skill_uncertainty --bootstraps 2000 --json
```
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Need at least this many OOS (pred, realized) pairs before a bootstrap
# CI is meaningful — mirrors calibration.MIN_PAIRS / _oos_rank_metrics'
# `n >= 2` floor but with a stricter quant-research bar. Below 30, the
# resamples are so small that the percentile CI is dominated by the
# discreteness of the resampling itself, not the underlying skill.
MIN_OOS = 30

# Number of bootstrap resamples. 1000 is the textbook lower bound for a
# stable 95% percentile CI (the 2.5th and 97.5th percentiles need enough
# samples that their estimate is itself stable). Higher is cleaner but
# slower. Each resample is O(n log n) (the Spearman sort dominates), so
# 1000 × n=5000 ≈ 0.4s on this box.
DEFAULT_BOOTSTRAPS = 1000

# CI level — 95% is the textbook default (matches the "is this skill
# distinguishable from noise" decision the verdict is reading).
DEFAULT_ALPHA = 0.05

# Seed so the resamples — and therefore every metric — are reproducible
# (the determinism a quant needs to trend the verdict across cycles).
DEFAULT_SEED = 17


def _percentile_ci(values: np.ndarray, alpha: float) -> tuple[float, float]:
    """Two-tailed percentile CI from a bootstrap distribution.

    Drops non-finite resamples (a Spearman of a constant resample is 0.0
    by `calibration._spearman`'s std-guard, never NaN — but RMSE could
    pick up an inf if the model returns a non-finite prediction that
    slipped through `predict_with_meta`'s finite check). Returns
    (nan, nan) when fewer than 2 finite values survive — honest empty,
    never a fabricated zero-width band.
    """
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        return float("nan"), float("nan")
    lo = float(np.percentile(finite, 100.0 * alpha / 2.0))
    hi = float(np.percentile(finite, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def _rmse(preds: np.ndarray, actuals: np.ndarray) -> float:
    if len(preds) == 0:
        return float("nan")
    diff = preds - actuals
    return float(np.sqrt(np.mean(diff * diff)))


def _dir_acc(preds: np.ndarray, actuals: np.ndarray) -> float:
    """Directional accuracy ignoring zero on either side (no truth there).

    Returns NaN when no informative pairs survive — never a fabricated 0.5.
    """
    mask = (preds != 0.0) & (actuals != 0.0)
    if not mask.any():
        return float("nan")
    hits = ((preds[mask] > 0) == (actuals[mask] > 0)).sum()
    return float(hits) / float(mask.sum())


def _predict_oos_pairs(scorer, records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Single-pass predict of every OOS record. Returns (preds, actuals)
    aligned arrays with SELL sign-flip applied (mirroring
    `_oos_rank_metrics` / `evaluate_scorer_oos`) and non-finite rows
    dropped.

    The expensive scorer.predict path runs ONCE here, outside the
    bootstrap loop — the resamples are pure array indexing.
    """
    from .decision_scorer import _to_float

    preds: list[float] = []
    actuals: list[float] = []
    for r in records:
        try:
            p = scorer.predict(
                ml_score=_to_float(r.get("ml_score"), 0.0),
                rsi=r.get("rsi"), macd=r.get("macd"),
                mom5=r.get("mom5"), mom20=r.get("mom20"),
                regime_mult=_to_float(r.get("regime_mult"), 1.0),
                ticker=str(r.get("ticker") or ""),
                vol_ratio=r.get("vol_ratio"), bb_pos=r.get("bb_position"),
                news_urgency=r.get("news_urgency"),
                news_article_count=r.get("news_article_count"),
            )
            a = _to_float(r.get("forward_return_5d"), float("nan"))
            if str(r.get("action") or "BUY").upper() == "SELL":
                a = -a
            pf = float(p)
            af = float(a)
            if pf == pf and af == af:  # drop NaN defensively
                preds.append(pf)
                actuals.append(af)
        except Exception:
            continue
    return (np.asarray(preds, dtype=np.float64),
            np.asarray(actuals, dtype=np.float64))


def bootstrap_skill_ci(
    scorer,
    oos_records: list[dict],
    n_bootstraps: int = DEFAULT_BOOTSTRAPS,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Bootstrap percentile CIs on the OOS rank-IC, RMSE, and directional
    accuracy of a deployed scorer.

    Returns a JSON-safe dict::

      {
        "status": "ok" | "insufficient_data" | "not_trained",
        "verdict": "SKILL_DETECTED" | "NO_SKILL_DETECTED" |
                   "INSUFFICIENT_DATA" | "NOT_TRAINED",
        "n_oos": int,
        "n_bootstraps": int,
        "alpha": float,
        "rank_ic":  {"point": float, "ci_low": float, "ci_high": float},
        "rmse":     {"point": float, "ci_low": float, "ci_high": float},
        "dir_acc":  {"point": float, "ci_low": float, "ci_high": float},
        "hint": str,
      }

    Verdict thresholds (testable):
      - ``NOT_TRAINED`` — scorer.is_trained is False
      - ``INSUFFICIENT_DATA`` — fewer than ``MIN_OOS`` finite OOS pairs
      - ``SKILL_DETECTED`` — rank-IC CI is strictly above 0
      - ``NO_SKILL_DETECTED`` — rank-IC CI straddles or sits below 0

    Never raises on bad input — every fault degrades to a ``status='error'``
    dict so a continuous-loop wrapper can never be blocked by this
    diagnostic (mirrors the AGENTS.md "scorer-train status must stay
    truthful" discipline observed elsewhere).
    """
    empty = {
        "status": "not_trained",
        "verdict": "NOT_TRAINED",
        "n_oos": 0,
        "n_bootstraps": int(n_bootstraps),
        "alpha": float(alpha),
        "rank_ic": {"point": None, "ci_low": None, "ci_high": None},
        "rmse": {"point": None, "ci_low": None, "ci_high": None},
        "dir_acc": {"point": None, "ci_low": None, "ci_high": None},
        "hint": "scorer is not trained — nothing to evaluate",
    }
    try:
        if not getattr(scorer, "is_trained", False):
            return empty
        from .calibration import _spearman

        preds, actuals = _predict_oos_pairs(scorer, oos_records or [])
        n = len(preds)
        if n < MIN_OOS:
            return {
                "status": "insufficient_data",
                "verdict": "INSUFFICIENT_DATA",
                "n_oos": n,
                "n_bootstraps": int(n_bootstraps),
                "alpha": float(alpha),
                "rank_ic": {"point": None, "ci_low": None, "ci_high": None},
                "rmse": {"point": None, "ci_low": None, "ci_high": None},
                "dir_acc": {"point": None, "ci_low": None, "ci_high": None},
                "hint": f"need ≥{MIN_OOS} OOS pairs, have {n}",
            }

        # Point estimates on the full OOS sample (no resampling).
        ic_point = _spearman(preds, actuals)
        rmse_point = _rmse(preds, actuals)
        dir_point = _dir_acc(preds, actuals)

        rng = np.random.default_rng(seed)
        idx_pool = np.arange(n, dtype=np.int64)
        ic_samples = np.empty(n_bootstraps, dtype=np.float64)
        rmse_samples = np.empty(n_bootstraps, dtype=np.float64)
        dir_samples = np.empty(n_bootstraps, dtype=np.float64)
        for b in range(n_bootstraps):
            # Resample with replacement — standard Efron bootstrap. Size
            # equal to the original sample is the textbook choice; smaller
            # would understate uncertainty, larger would understate it the
            # other way (overconfident CIs).
            idx = rng.choice(idx_pool, size=n, replace=True)
            bp = preds[idx]
            ba = actuals[idx]
            ic_samples[b] = _spearman(bp, ba)
            rmse_samples[b] = _rmse(bp, ba)
            dir_samples[b] = _dir_acc(bp, ba)

        ic_lo, ic_hi = _percentile_ci(ic_samples, alpha)
        rmse_lo, rmse_hi = _percentile_ci(rmse_samples, alpha)
        dir_lo, dir_hi = _percentile_ci(dir_samples, alpha)

        # Verdict on rank-IC: the gate acts on prediction sign/bucket
        # (CLAUDE.md §6), so rank skill is what economically matters. A CI
        # strictly above 0 ⇒ statistically distinguishable from a zero-skill
        # baseline at the chosen alpha. CI straddling 0 ⇒ indistinguishable.
        # Use math.isfinite to reject the (nan,nan) honest-empty CI from
        # `_percentile_ci` — a degenerate resample would otherwise be read
        # as "CI excludes 0" which would be a silent false-positive verdict.
        if not (np.isfinite(ic_lo) and np.isfinite(ic_hi)):
            verdict = "NO_SKILL_DETECTED"
            hint = ("rank-IC bootstrap CI is degenerate (constant resamples) — "
                    "treat as no detectable skill")
        elif ic_lo > 0.0:
            verdict = "SKILL_DETECTED"
            hint = (f"rank-IC 95% CI [{ic_lo:+.3f}, {ic_hi:+.3f}] excludes 0 — "
                    f"the OOS skill is statistically distinguishable from noise")
        else:
            verdict = "NO_SKILL_DETECTED"
            hint = (f"rank-IC 95% CI [{ic_lo:+.3f}, {ic_hi:+.3f}] straddles 0 — "
                    f"OOS skill is indistinguishable from noise at this n_oos")

        def _cell(point, lo, hi):
            return {
                "point": (round(float(point), 4)
                          if (point is not None and point == point) else None),
                "ci_low": (round(float(lo), 4)
                           if np.isfinite(lo) else None),
                "ci_high": (round(float(hi), 4)
                            if np.isfinite(hi) else None),
            }

        return {
            "status": "ok",
            "verdict": verdict,
            "n_oos": n,
            "n_bootstraps": int(n_bootstraps),
            "alpha": float(alpha),
            "rank_ic": _cell(ic_point, ic_lo, ic_hi),
            "rmse": _cell(rmse_point, rmse_lo, rmse_hi),
            "dir_acc": _cell(dir_point, dir_lo, dir_hi),
            "hint": hint,
        }
    except Exception as e:
        return {
            "status": "error",
            "verdict": "ERROR",
            "n_oos": 0,
            "n_bootstraps": int(n_bootstraps),
            "alpha": float(alpha),
            "rank_ic": {"point": None, "ci_low": None, "ci_high": None},
            "rmse": {"point": None, "ci_low": None, "ci_high": None},
            "dir_acc": {"point": None, "ci_low": None, "ci_high": None},
            "hint": f"bootstrap failed: {type(e).__name__}: {e}",
        }


def _cli(argv: list[str] | None = None) -> int:
    """`python3 -m paper_trader.ml.skill_uncertainty [--bootstraps N] [--json]`

    Runs the bootstrap CI over the deployed scorer + the temporal OOS
    holdout from `data/decision_outcomes.jsonl`. Read-only.

    Exit code 0 when status is ok AND verdict is SKILL_DETECTED, 1
    otherwise — so a shell caller can `if !; then` gate dashboards or
    pages on real distinguishable skill, not noisy point estimates.
    """
    import argparse
    import sys

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.skill_uncertainty",
        description="Bootstrap 95% CIs on the DecisionScorer's OOS "
                    "rank-IC / RMSE / dir-acc. Exit 0 only when the "
                    "rank-IC CI excludes 0 (distinguishable skill).",
    )
    p.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS,
                   help=f"resamples (default {DEFAULT_BOOTSTRAPS})")
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                   help=f"two-tailed alpha (default {DEFAULT_ALPHA})")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"bootstrap RNG seed (default {DEFAULT_SEED})")
    p.add_argument("--json", action="store_true",
                   help="Machine-readable JSON instead of a text report.")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    from .decision_scorer import DecisionScorer
    from paper_trader.validation import split_outcomes_temporal

    root = Path(__file__).resolve().parent.parent.parent
    out_path = root / "data" / "decision_outcomes.jsonl"
    if not out_path.exists():
        msg = f"[skill_uncertainty] no outcomes file at {out_path}"
        if args.json:
            print(json.dumps({"status": "error", "hint": msg}))
        else:
            print(msg)
        return 1
    records: list[dict] = []
    for ln in out_path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            records.append(json.loads(ln))
        except Exception:
            pass
    try:
        _, oos = split_outcomes_temporal(records, oos_fraction=0.2)
    except Exception:
        oos = []
    scorer = DecisionScorer()
    rep = bootstrap_skill_ci(
        scorer, oos,
        n_bootstraps=args.bootstraps,
        alpha=args.alpha,
        seed=args.seed,
    )

    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
    else:
        print(f"[skill_uncertainty] VERDICT: {rep['verdict']}  ({rep['hint']})")
        print(f"  n_oos={rep['n_oos']} n_bootstraps={rep['n_bootstraps']} "
              f"alpha={rep['alpha']}")
        for k in ("rank_ic", "rmse", "dir_acc"):
            c = rep[k]
            pt = c["point"]
            lo = c["ci_low"]
            hi = c["ci_high"]
            pt_s = f"{pt:+.4f}" if pt is not None else "n/a"
            ci_s = (f"[{lo:+.4f}, {hi:+.4f}]"
                    if lo is not None and hi is not None else "[n/a, n/a]")
            print(f"  {k:<10} point={pt_s}  ci={ci_s}")
    return 0 if rep.get("verdict") == "SKILL_DETECTED" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())

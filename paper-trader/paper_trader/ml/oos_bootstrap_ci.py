"""Bootstrap confidence intervals for out-of-sample scorer skill metrics.

Every other OOS diagnostic in this package reports POINT ESTIMATES:
``skill_trend`` / ``baseline_compare`` / ``calibration`` / ``persona_skill`` /
``sector_skill`` all answer "what is the OOS RMSE / rank-IC / dir-acc" with a
single number. None of them answers the question a skeptical quant **actually**
needs to act on the gate:

  *Is ``oos_rmse=11.83`` actually distinguishable from the σ(target)≈11.7
   mean-predictor baseline, or is it within sampling noise?*

  *Is the "+0.11 OOS rank-IC" the skill ledger reports this cycle a real
   signal, or a coin flip on a ~1000-row OOS slice?*

This module answers both via a non-parametric bootstrap over the SAME
temporal-OOS slice ``_train_decision_scorer`` evaluates with: sample N
records with replacement, recompute the headline metrics, repeat
``n_bootstrap`` times, then report the empirical 2.5%/97.5% percentiles as
a 95% CI. The output is purely additive — no existing call sites change, no
training behaviour shifts, no pickle compatibility risk.

A quant can now read a single CLI line that previously took 10 minutes of
ad-hoc analysis to derive:

    python3 -m paper_trader.ml.oos_bootstrap_ci

  oos_rmse        =  11.83  [11.41, 12.27]   <-- straddles σ=11.7 → no skill
  oos_dir_acc     =  0.531  [0.504, 0.560]   <-- 50% inside CI → coin flip
  oos_rank_ic     =  0.089  [0.027, 0.151]   <-- excludes 0 → real but tiny
  n               = 1000    n_bootstrap=1000

The CI for ``rank_ic`` is the decisive one: a CI that excludes 0 means the
ordering edge is statistically real (however small in magnitude); a CI that
straddles 0 means the recent +0.11 reads are within sampling noise of zero.

This is a **read-only diagnostic**. It NEVER trains, NEVER touches
``decision_scorer.pkl`` / ``build_features`` / ``N_FEATURES`` / any trade
path. Safe to run against the unattended continuous loop — like the other
ml/* diagnostics, it loads the deployed pickle + outcomes file fresh and
exits.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Module-level constants — matches the codebase convention. Threshold
# choices are documented inline with each.
DEFAULT_N_BOOTSTRAP = 1000  # ~1k samples is the sweet spot: stable to ±2pp on
# the CI bounds, completes in <2s on the live 1000-row slice.
DEFAULT_CI_LEVEL = 0.95     # 95% CI is the industry-standard frequentist bar.
MIN_PAIRS_FOR_CI = 30       # below this, the bootstrap CI is not meaningful —
# matches calibration.MIN_PAIRS so cross-diagnostic verdict floors agree.


def _percentile_ci(values: np.ndarray, ci_level: float) -> tuple[float, float]:
    """Empirical percentile CI from a bootstrap distribution. Tolerates an
    empty or near-degenerate sample (returns ``(NaN, NaN)`` so the headline
    metric reports clean None without an exception).
    """
    if values.size == 0:
        return float("nan"), float("nan")
    alpha = (1.0 - ci_level) / 2.0
    lo = float(np.percentile(values, 100.0 * alpha))
    hi = float(np.percentile(values, 100.0 * (1.0 - alpha)))
    return lo, hi


def _build_aligned_arrays(
    scorer,
    oos_records: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-compute the (pred, realized) arrays once. Each bootstrap iteration
    samples indices, so the expensive scorer.predict() call only happens N
    times instead of N * n_bootstrap.

    Mirrors ``evaluate_scorer_oos`` / ``_oos_rank_metrics``'s exact predict
    signature, the universal SELL sign-flip, the ``±PRED_CLAMP_PCT`` label
    clamp (apples-to-apples with val_rmse and the rest of the OOS suite),
    and the NaN-sentinel drop discipline — single source of truth across
    the OOS suite means a future hardening in any of those three sibling
    paths must be mirrored here. Records that fail to predict (shape/dtype
    mismatch from a build_features change without retrain) are silently
    dropped so a single bad row never poisons the CI.
    """
    if not oos_records or not getattr(scorer, "is_trained", False):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    # Local imports keep module-import cost zero, matching the ``ml/`` package
    # convention.
    from paper_trader.ml.decision_scorer import _to_float, PRED_CLAMP_PCT

    preds: list[float] = []
    actuals: list[float] = []
    for r in oos_records:
        try:
            p = scorer.predict(
                ml_score=_to_float(r.get("ml_score"), 0.0),
                rsi=r.get("rsi"),
                macd=r.get("macd"),
                mom5=r.get("mom5"),
                mom20=r.get("mom20"),
                regime_mult=_to_float(r.get("regime_mult"), 1.0),
                ticker=str(r.get("ticker") or ""),
                vol_ratio=r.get("vol_ratio"),
                bb_pos=r.get("bb_position"),
                news_urgency=r.get("news_urgency"),
                news_article_count=r.get("news_article_count"),
            )
            a = _to_float(r.get("forward_return_5d"), float("nan"))
            if str(r.get("action") or "BUY").upper() == "SELL":
                a = -a
            pf = float(p)
            af = float(a)
            if pf == pf and af == af:
                # Mirror train_scorer's symmetric clamp so this CI describes
                # the same target space the model was trained against.
                af = max(-PRED_CLAMP_PCT, min(PRED_CLAMP_PCT, af))
                preds.append(pf)
                actuals.append(af)
        except Exception:
            continue
    return (np.asarray(preds, dtype=np.float64),
            np.asarray(actuals, dtype=np.float64))


def _rmse(p: np.ndarray, a: np.ndarray) -> float:
    return float(np.sqrt(np.mean((p - a) ** 2)))


def _dir_acc(p: np.ndarray, a: np.ndarray) -> float | None:
    """Fraction of pairs where sign(p) == sign(a), excluding ties at 0."""
    mask = (p != 0.0) & (a != 0.0)
    if not mask.any():
        return None
    return float(np.mean((p[mask] > 0) == (a[mask] > 0)))


def _rank_ic(p: np.ndarray, a: np.ndarray) -> float | None:
    """Tie-aware Spearman — same _spearman primitive ``calibration``,
    ``_oos_rank_metrics`` and ``baseline_compare`` use, so this CI's point
    estimate equals theirs by construction (single source of truth across
    the OOS suite)."""
    if len(p) < 2:
        return None
    # Local import: avoids a circular import at module load.
    from paper_trader.ml.calibration import _spearman
    ic = _spearman(p, a)
    if ic != ic:  # NaN guard — constant predictor produces 0.0 not NaN, but
        return None  # defensively report None on any non-finite outcome
    return float(ic)


def bootstrap_ci(
    scorer,
    oos_records: list[dict],
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci_level: float = DEFAULT_CI_LEVEL,
    seed: int = 42,
) -> dict:
    """Compute 95% bootstrap CIs for OOS RMSE / dir_acc / rank_ic.

    The CIs answer the operator-decisive "is +0.11 OOS rank-IC really above
    zero or sampling noise?" / "is oos_rmse=11.83 below σ(target)=11.7 or
    indistinguishable from the mean-predictor baseline?" questions — neither
    of which any existing OOS diagnostic answers (they all report point
    estimates).

    Returns a JSON-safe dict::

        {
          "n": int,                        # number of valid (pred, realized) pairs
          "n_bootstrap": int,              # number of resamples actually run
          "ci_level": float,               # echoed for cross-validation
          "rmse": {"value": float, "ci_low": float, "ci_high": float},
          "dir_acc": {"value": float, "ci_low": float, "ci_high": float},
          "rank_ic": {"value": float, "ci_low": float, "ci_high": float},
          "status": "ok" | "insufficient_data" | "scorer_not_trained" | "empty",
        }

    A scorer that's not trained, an empty OOS set, or fewer than
    ``MIN_PAIRS_FOR_CI`` valid pairs degrades to a verdict-keyed
    insufficient-data dict (the calibration / baseline_compare honest-empty
    precedent). Never raises — every failure path returns a well-formed
    dict so a dashboard/CLI never sees an exception.
    """
    out_empty = {
        "n": 0, "n_bootstrap": 0, "ci_level": float(ci_level),
        "rmse": {"value": None, "ci_low": None, "ci_high": None},
        "dir_acc": {"value": None, "ci_low": None, "ci_high": None},
        "rank_ic": {"value": None, "ci_low": None, "ci_high": None},
        "status": "empty",
    }

    if not oos_records:
        return out_empty
    if not getattr(scorer, "is_trained", False):
        out_empty["status"] = "scorer_not_trained"
        return out_empty

    preds, actuals = _build_aligned_arrays(scorer, oos_records)
    n = int(preds.size)
    if n < MIN_PAIRS_FOR_CI:
        return {
            "n": n, "n_bootstrap": 0, "ci_level": float(ci_level),
            "rmse": {"value": None, "ci_low": None, "ci_high": None},
            "dir_acc": {"value": None, "ci_low": None, "ci_high": None},
            "rank_ic": {"value": None, "ci_low": None, "ci_high": None},
            "status": "insufficient_data",
        }

    # Point estimates (the headline values every CI is centred around).
    point_rmse = _rmse(preds, actuals)
    point_dir_acc = _dir_acc(preds, actuals)
    point_rank_ic = _rank_ic(preds, actuals)

    # Bootstrap distributions. Seeded RNG so the CI is deterministic — a
    # cycle-over-cycle CI drift then reflects real data shifts, not RNG
    # noise. The ``np.random.default_rng`` API is the modern bit-generator
    # path; ``choice`` with replacement is the canonical bootstrap resample.
    rng = np.random.default_rng(int(seed))
    rmse_dist = np.empty(n_bootstrap, dtype=np.float64)
    dir_dist: list[float] = []
    ic_dist: list[float] = []

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        p_b = preds[idx]
        a_b = actuals[idx]
        rmse_dist[i] = _rmse(p_b, a_b)
        d = _dir_acc(p_b, a_b)
        if d is not None:
            dir_dist.append(d)
        ic = _rank_ic(p_b, a_b)
        if ic is not None:
            ic_dist.append(ic)

    rmse_lo, rmse_hi = _percentile_ci(rmse_dist, ci_level)
    dir_arr = np.asarray(dir_dist, dtype=np.float64)
    ic_arr = np.asarray(ic_dist, dtype=np.float64)
    dir_lo, dir_hi = _percentile_ci(dir_arr, ci_level)
    ic_lo, ic_hi = _percentile_ci(ic_arr, ci_level)

    def _round_or_none(v):
        if v is None:
            return None
        if isinstance(v, float) and v != v:  # NaN
            return None
        return round(float(v), 4)

    return {
        "n": n,
        "n_bootstrap": int(n_bootstrap),
        "ci_level": float(ci_level),
        "rmse": {
            "value": _round_or_none(point_rmse),
            "ci_low": _round_or_none(rmse_lo),
            "ci_high": _round_or_none(rmse_hi),
        },
        "dir_acc": {
            "value": _round_or_none(point_dir_acc),
            "ci_low": _round_or_none(dir_lo),
            "ci_high": _round_or_none(dir_hi),
        },
        "rank_ic": {
            "value": _round_or_none(point_rank_ic),
            "ci_low": _round_or_none(ic_lo),
            "ci_high": _round_or_none(ic_hi),
        },
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# CLI: `python3 -m paper_trader.ml.oos_bootstrap_ci`
#
# Mirrors the existing CLI pattern in `decision_scorer.py::main` and
# `host_guard.py` (int return + --json + SystemExit) so an operator gets
# one muscle memory. Loads `data/decision_outcomes.jsonl` + the deployed
# pickle, runs the temporal OOS split (same one `_train_decision_scorer`
# uses for `oos_rmse`), then prints CIs. Read-only by construction.
# ---------------------------------------------------------------------------


def _default_outcomes_path() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "data" / "decision_outcomes.jsonl"


def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.oos_bootstrap_ci",
        description=(
            "Bootstrap 95% CIs for out-of-sample DecisionScorer skill "
            "(RMSE, dir_acc, rank_ic). Answers the operator-decisive "
            "question every other OOS diagnostic dodges: is the headline "
            "metric STATISTICALLY distinguishable from sampling noise?"
        ),
    )
    p.add_argument(
        "--outcomes", default=str(_default_outcomes_path()),
        help="Path to decision_outcomes.jsonl (default: data/decision_outcomes.jsonl)",
    )
    p.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP,
                   dest="n_bootstrap", help="Bootstrap resample count.")
    p.add_argument("--ci-level", type=float, default=DEFAULT_CI_LEVEL,
                   dest="ci_level", help="CI confidence level (0..1).")
    p.add_argument("--oos-fraction", type=float, default=0.2,
                   dest="oos_fraction",
                   help="Most-recent fraction held out as OOS slice.")
    p.add_argument("--all-records", action="store_true", dest="all_records",
                   help="Evaluate against ALL records (in-sample + OOS) "
                        "instead of the temporal-OOS holdout. Useful for "
                        "comparison only — the OOS slice is the trustworthy "
                        "generalization view.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of a table.")
    return p


def _load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
    return out


def main(argv: list[str] | None = None) -> int:
    import sys

    args = _build_arg_parser().parse_args(
        sys.argv[1:] if argv is None else argv)

    from paper_trader.ml.decision_scorer import DecisionScorer
    from paper_trader.validation import split_outcomes_temporal

    records = _load_records(Path(args.outcomes))
    if not records:
        msg = {"status": "no_outcomes", "path": args.outcomes}
        print(json.dumps(msg, indent=2) if args.json
              else f"[oos_bootstrap_ci] no records at {args.outcomes}")
        return 1

    if args.all_records:
        slice_records = records
        slice_label = "all"
    else:
        _, oos = split_outcomes_temporal(records, oos_fraction=args.oos_fraction)
        slice_records = oos
        slice_label = "oos"

    scorer = DecisionScorer()
    if not scorer.is_trained:
        msg = {"status": "scorer_not_trained",
               "hint": "no pickle at data/ml/decision_scorer.pkl"}
        print(json.dumps(msg, indent=2) if args.json
              else "[oos_bootstrap_ci] scorer NOT trained — no pickle on disk")
        return 1

    result = bootstrap_ci(
        scorer, slice_records,
        n_bootstrap=args.n_bootstrap, ci_level=args.ci_level,
        seed=args.seed,
    )
    result["slice"] = slice_label
    result["n_train"] = scorer.n_train

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("status") == "ok" else 1

    status = result.get("status", "?")
    if status != "ok":
        print(f"[oos_bootstrap_ci] {status}  n={result.get('n', 0)}  "
              f"slice={slice_label}")
        return 1

    pct = int(args.ci_level * 100)
    n = result["n"]
    nb = result["n_bootstrap"]
    print(f"[oos_bootstrap_ci] slice={slice_label}  n={n}  n_bootstrap={nb}  "
          f"n_train={scorer.n_train}  ({pct}% CI)")

    def _fmt(metric: str, sign_prefix: bool = False) -> None:
        cell = result[metric]
        v, lo, hi = cell["value"], cell["ci_low"], cell["ci_high"]
        if v is None:
            print(f"  {metric:<15} = n/a")
            return
        fmt_v = f"{v:+.4f}" if sign_prefix else f"{v:.4f}"
        fmt_lo = f"{lo:+.4f}" if sign_prefix else f"{lo:.4f}"
        fmt_hi = f"{hi:+.4f}" if sign_prefix else f"{hi:.4f}"
        print(f"  {metric:<15} = {fmt_v}  [{fmt_lo}, {fmt_hi}]")

    _fmt("rmse")
    _fmt("dir_acc")
    _fmt("rank_ic", sign_prefix=True)
    # The decisive verdict: a rank_ic CI that EXCLUDES 0 means the ordering
    # edge is statistically real (however small). A CI that straddles 0
    # means the recent "+0.11 rank_ic" reads are within sampling noise of
    # zero — the gate is modulating sizing on an unverified signal.
    ic = result["rank_ic"]
    if (ic["ci_low"] is not None and ic["ci_high"] is not None):
        if ic["ci_low"] > 0:
            print("  → rank_ic CI EXCLUDES 0 — directional edge is real")
        elif ic["ci_high"] < 0:
            print("  → rank_ic CI EXCLUDES 0 (negative) — ANTI-skill detected")
        else:
            print("  → rank_ic CI STRADDLES 0 — point estimate within "
                  "sampling noise of zero")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

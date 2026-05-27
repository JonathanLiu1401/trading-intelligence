"""Permutation null-hypothesis test for out-of-sample DecisionScorer rank skill.

The OOS suite already answers two questions:

  * ``oos_bootstrap_ci``      — "is +0.11 rank-IC inside the sampling-error
                                 band of zero?" (point-estimate variability)
  * ``oos_parity_audit``      — "how big was the OOS-inference feature-parity
                                 bias the gate had been silently absorbing?"

Neither answers the orthogonal frequentist question every skeptical quant
**also** asks before trusting a positive rank-IC:

  *Under H0 (the scorer has no skill — predictions are uncorrelated with
  realized returns), how often would I see a rank-IC at least this extreme
  by pure chance from random label shuffling?*

That's a one-sided permutation p-value. A bootstrap CI tells you the
variability of your **estimate**; a permutation null tells you whether
your estimate is **distinguishable from chance**. They overlap in their
conclusion on well-behaved estimators but are not redundant — a single
small OOS slice can read a "tight CI excluding 0" (high estimate
precision) while still landing inside the null distribution's bulk
(no real signal). Quant practitioners use both because they answer
different questions.

The verdict ladder is crisp and operator-actionable:

  * ``STATISTICALLY_SIGNIFICANT`` — p_value < 0.01  (real edge, very likely)
  * ``WEAKLY_SIGNIFICANT``        — p_value < 0.05  (real edge, marginal)
  * ``AT_NOISE``                  — p_value >= 0.05 (not distinguishable
                                                    from H0; current
                                                    documented state)
  * ``INSUFFICIENT_DATA``         — < ``MIN_PAIRS`` valid (pred, realized)
                                    pairs
  * ``SCORER_NOT_TRAINED``        — no pickle on disk
  * ``EMPTY``                     — no outcome records at all

Per-action breakdown — the gate is BUY-only (CLAUDE.md §6), so the BUY
p-value is what matters for the keep-or-kill conviction-gate decision.
SELL p-value is reported as a sanity check because the model is trained
on SELL-sign-flipped labels so a positive SELL rank-IC is informative
about whether the trained skill generalises to inverse selection.

**Read-only diagnostic.** Never trains, never touches
``decision_scorer.pkl`` / ``build_features`` / ``N_FEATURES``, never
modifies ``decision_outcomes.jsonl``. Safe to run against the unattended
loop — like every other ``ml/*`` audit. Mirrors the
``oos_bootstrap_ci`` predict signature exactly (single source of truth
across the OOS suite) so the **point estimate** equals what the skill
ledger already reports by construction. A divergence between this
module's reported ``rank_ic`` and the ledger's ``oos_ic`` for the same
cycle would mean either an analyzer drifted from the trainer's predict
signature, or the OOS slice fraction differs — both are caller-fixable.

CLI::

    python3 -m paper_trader.ml.oos_permutation_test
    python3 -m paper_trader.ml.oos_permutation_test --json
    python3 -m paper_trader.ml.oos_permutation_test --n-permutations 5000
    python3 -m paper_trader.ml.oos_permutation_test --action buy

Exit code mirrors ``host_guard`` / ``decision_scorer``'s CLI: 0 when the
verdict is ``STATISTICALLY_SIGNIFICANT`` / ``WEAKLY_SIGNIFICANT`` (a real
edge was detected), 1 otherwise — so shell callers can gate on ``$?``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTCOMES_PATH = ROOT / "data" / "decision_outcomes.jsonl"

# Default permutation count. With the smoothed p-value formula
# ``p = (k + 1) / (n + 1)`` the minimum achievable p is ``1 / (n + 1)``,
# so a ``< 0.01`` verdict requires at least ~100 permutations. 1000 gives a
# comfortable floor (``p_min ≈ 0.001``) and completes in <3s on the typical
# 1000-row OOS slice. Operators wanting tighter resolution can override via
# ``--n-permutations``.
DEFAULT_N_PERMUTATIONS = 1000

# Default OOS fraction — matches ``_train_decision_scorer`` /
# ``calibration.scorer_calibration_oos`` so the slice this module reads is
# the same one every other OOS diagnostic evaluates against. Single
# source of truth across the OOS suite.
DEFAULT_OOS_FRACTION = 0.2

# Significance thresholds. ``ALPHA_STRICT`` is the canonical "real edge"
# bar (the 0.01 frequentist threshold) and ``ALPHA_WEAK`` is the
# marginal-but-suggestive band. A p-value above ALPHA_WEAK lands in
# AT_NOISE — the gate is modulating on noise.
ALPHA_STRICT = 0.01
ALPHA_WEAK = 0.05

# Below this many valid (pred, realized) pairs neither the headline IC
# nor the permutation null is meaningful. Matches the sibling
# ``oos_bootstrap_ci.MIN_PAIRS_FOR_CI`` so the two diagnostics agree on
# when the slice is too small to interpret.
MIN_PAIRS = 30


def _spearman_local(p: np.ndarray, a: np.ndarray) -> float | None:
    """Tie-aware Spearman via the single SSOT primitive every other OOS
    diagnostic uses. Local import to avoid module-load circular imports —
    matches ``oos_bootstrap_ci._rank_ic``'s convention.

    Zero-variance guard: a constant predictions array (e.g. an
    untrained scorer that always returns 0.0) has *no* rank ordering
    to shuffle against — every permutation produces the same IC, so
    the permutation null distribution is degenerate (p ≈ 1.0 by
    construction) rather than informative. The SSOT ``_spearman``
    deliberately returns ``0.0`` in that case (the right call for
    calibration's aggregation use); for a permutation test the more
    honest signal is ``None`` so the bucket reports
    ``INSUFFICIENT_DATA`` rather than fake AT_NOISE."""
    if p.size < 2:
        return None
    # ddof=0 to match ``_spearman``'s np.std default. A length-1 or
    # constant input fails this check; .size < 2 above already covered
    # length-1 so this is the constant-array branch.
    if float(np.std(p)) == 0.0 or float(np.std(a)) == 0.0:
        return None
    from paper_trader.ml.calibration import _spearman
    ic = _spearman(p, a)
    if ic != ic:  # NaN guard
        return None
    return float(ic)


def _build_aligned_arrays(
    scorer,
    oos_records: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-compute (preds, actuals, is_sell) once. Each permutation iteration
    only shuffles labels, so the scorer.predict() cost is paid exactly once
    rather than N * n_permutations.

    Mirrors ``oos_bootstrap_ci._build_aligned_arrays`` byte-for-byte on the
    11-kwarg ``predict`` signature, SELL sign-flip, ``±PRED_CLAMP_PCT``
    label clamp, and NaN-sentinel drop — so a permutation-test point
    estimate cannot diverge from the bootstrap-CI point estimate or the
    per-cycle skill ledger's ``oos_ic``. The enhanced MACD kwargs are
    forwarded so this audit hits the same predict path the gate does
    (the pass #36 OOS-inference parity fix). Records that fail to
    predict are silently dropped — a single bad row never poisons
    the null distribution.

    The third array carries the per-row SELL flag so the per-action
    breakdown can index without re-walking the records.
    """
    from paper_trader.ml.decision_scorer import _to_float, PRED_CLAMP_PCT

    preds: list[float] = []
    actuals: list[float] = []
    sells: list[bool] = []
    _pwm = getattr(scorer, "predict_with_meta", None)
    _use_meta = callable(_pwm)

    for r in oos_records:
        try:
            if _use_meta:
                meta = _pwm(
                    ml_score=_to_float(r.get("ml_score"), 0.0),
                    rsi=r.get("rsi"), macd=r.get("macd"),
                    mom5=r.get("mom5"), mom20=r.get("mom20"),
                    regime_mult=_to_float(r.get("regime_mult"), 1.0),
                    ticker=str(r.get("ticker") or ""),
                    vol_ratio=r.get("vol_ratio"), bb_pos=r.get("bb_position"),
                    news_urgency=r.get("news_urgency"),
                    news_article_count=r.get("news_article_count"),
                    ema200_above=r.get("ema200_above"),
                    hist_cross_up=r.get("hist_cross_up"),
                    macd_below_zero_cross=r.get("macd_below_zero_cross"),
                )
                # `failed=True` means the 0.0 in `pred` is a sentinel, NOT
                # a real prediction — drop so it cannot contribute a fake
                # rank-tie at zero to either the headline IC or any
                # shuffled null IC.
                if meta.get("failed"):
                    continue
                p = float(meta.get("pred", 0.0))
            else:
                p = float(scorer.predict(
                    ml_score=_to_float(r.get("ml_score"), 0.0),
                    rsi=r.get("rsi"), macd=r.get("macd"),
                    mom5=r.get("mom5"), mom20=r.get("mom20"),
                    regime_mult=_to_float(r.get("regime_mult"), 1.0),
                    ticker=str(r.get("ticker") or ""),
                    vol_ratio=r.get("vol_ratio"), bb_pos=r.get("bb_position"),
                    news_urgency=r.get("news_urgency"),
                    news_article_count=r.get("news_article_count"),
                    ema200_above=r.get("ema200_above"),
                    hist_cross_up=r.get("hist_cross_up"),
                    macd_below_zero_cross=r.get("macd_below_zero_cross"),
                ))
            a = _to_float(r.get("forward_return_5d"), float("nan"))
            is_sell = str(r.get("action") or "BUY").upper() == "SELL"
            if is_sell:
                a = -a
            if p != p or a != a:  # NaN guard
                continue
            a = max(-PRED_CLAMP_PCT, min(PRED_CLAMP_PCT, a))
            preds.append(p)
            actuals.append(a)
            sells.append(is_sell)
        except Exception:
            continue
    return (np.asarray(preds, dtype=np.float64),
            np.asarray(actuals, dtype=np.float64),
            np.asarray(sells, dtype=bool))


def _verdict_from_p(p_value: float | None) -> str:
    """Map a one-sided p-value to the verdict ladder. None ⇒ INSUFFICIENT_DATA."""
    if p_value is None:
        return "INSUFFICIENT_DATA"
    if p_value < ALPHA_STRICT:
        return "STATISTICALLY_SIGNIFICANT"
    if p_value < ALPHA_WEAK:
        return "WEAKLY_SIGNIFICANT"
    return "AT_NOISE"


def _permutation_p_value(
    preds: np.ndarray,
    actuals: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[float | None, float | None, list[float]]:
    """Two-sided permutation test for rank-IC. Returns (p_value, point_ic,
    null_distribution).

    Why two-sided: the gate uses sign(pred). Both a strongly positive AND
    a strongly negative rank-IC are operationally meaningful (the latter
    is ANTI-skill — flip the sign and you have a profitable signal). H0
    is "the predictions are uncorrelated with realized returns"; H1 is
    "they correlate, either direction". The p-value counts shuffled
    rank-IC values whose ABSOLUTE magnitude is at least the observed
    |IC| — so an IC of -0.20 reaches significance the same way +0.20 does,
    matching how the operator interprets the gate's signal.

    Smoothed estimator ``p = (k + 1) / (n + 1)`` (Davison & Hinkley §4.2
    convention) — the classical unbiased correction that ensures
    ``p > 0`` even when no shuffled IC matches the observed magnitude
    (the alternative ``k/n`` floors at 0 and falsely claims arbitrarily
    strong significance from a finite sample).
    """
    if preds.size < MIN_PAIRS:
        return None, None, []
    point_ic = _spearman_local(preds, actuals)
    if point_ic is None:
        return None, None, []

    obs_abs = abs(point_ic)
    null_distribution: list[float] = []
    k_ge = 0
    n_done = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(actuals)
        ic = _spearman_local(preds, shuffled)
        if ic is None:
            continue
        null_distribution.append(float(ic))
        if abs(ic) >= obs_abs:
            k_ge += 1
        n_done += 1
    if n_done == 0:
        return None, point_ic, []
    p_value = (k_ge + 1.0) / (n_done + 1.0)
    return p_value, point_ic, null_distribution


def permutation_test(
    scorer,
    oos_records: list[dict],
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    seed: int = 42,
) -> dict:
    """Compute permutation-null p-values for aggregate / BUY / SELL rank-IC
    on the temporal-OOS slice.

    Returns a JSON-safe dict::

        {
          "status": "ok" | "scorer_not_trained" | "empty"
                    | "insufficient_data",
          "n": int,                           # total (pred, realized) pairs
          "n_permutations": int,              # echoed (per-bucket count may
                                              #   differ if MIN_PAIRS not met)
          "aggregate": {
            "n": int, "rank_ic": float | None,
            "p_value": float | None,
            "verdict": "STATISTICALLY_SIGNIFICANT" | ... | "INSUFFICIENT_DATA",
            "null_p10": float | None,         # 10/50/90th pct of null dist
            "null_p50": float | None,         #   (operator-readable shape
            "null_p90": float | None,         #   summary so the verdict is
                                              #   visually verifiable)
          },
          "buy":  { …same shape… },
          "sell": { …same shape… },
          "alpha_strict": ALPHA_STRICT,
          "alpha_weak":   ALPHA_WEAK,
          "min_pairs":    MIN_PAIRS,
        }

    Never raises — every failure degrades to a well-formed dict with
    ``status`` reporting the cause and ``verdict`` set on each bucket
    so the diagnostic is honestly empty rather than crashing the caller.
    """
    out_empty = {
        "status": "empty", "n": 0,
        "n_permutations": int(n_permutations),
        "aggregate": _bucket_empty(),
        "buy":       _bucket_empty(),
        "sell":      _bucket_empty(),
        "alpha_strict": ALPHA_STRICT,
        "alpha_weak":   ALPHA_WEAK,
        "min_pairs":    MIN_PAIRS,
    }
    if not oos_records:
        return out_empty
    if not getattr(scorer, "is_trained", False):
        out_empty["status"] = "scorer_not_trained"
        return out_empty

    preds, actuals, sells = _build_aligned_arrays(scorer, oos_records)
    n = int(preds.size)
    if n < MIN_PAIRS:
        out_empty["status"] = "insufficient_data"
        out_empty["n"] = n
        return out_empty

    rng = np.random.default_rng(int(seed))

    def _bucket(p_arr: np.ndarray, a_arr: np.ndarray) -> dict:
        if p_arr.size < MIN_PAIRS:
            return _bucket_empty(n=int(p_arr.size))
        p_val, ic, nullv = _permutation_p_value(
            p_arr, a_arr, n_permutations, rng)
        if ic is None or p_val is None:
            return _bucket_empty(n=int(p_arr.size))
        # Operator-readable null-dist shape. p10/p50/p90 is enough to see at
        # a glance whether the null is centred on zero (the textbook
        # H0 expectation) and how wide its support is.
        null_arr = np.asarray(nullv, dtype=np.float64)
        p10 = float(np.percentile(null_arr, 10)) if null_arr.size else None
        p50 = float(np.percentile(null_arr, 50)) if null_arr.size else None
        p90 = float(np.percentile(null_arr, 90)) if null_arr.size else None
        return {
            "n": int(p_arr.size),
            "rank_ic": round(float(ic), 4),
            "p_value": round(float(p_val), 4),
            "verdict": _verdict_from_p(p_val),
            "null_p10": round(p10, 4) if p10 is not None else None,
            "null_p50": round(p50, 4) if p50 is not None else None,
            "null_p90": round(p90, 4) if p90 is not None else None,
        }

    aggregate = _bucket(preds, actuals)
    buy_mask = ~sells
    sell_mask = sells
    buy = _bucket(preds[buy_mask], actuals[buy_mask])
    sell = _bucket(preds[sell_mask], actuals[sell_mask])

    return {
        "status": "ok",
        "n": n,
        "n_permutations": int(n_permutations),
        "aggregate": aggregate,
        "buy": buy,
        "sell": sell,
        "alpha_strict": ALPHA_STRICT,
        "alpha_weak":   ALPHA_WEAK,
        "min_pairs":    MIN_PAIRS,
    }


def _bucket_empty(n: int = 0) -> dict:
    return {
        "n": int(n), "rank_ic": None, "p_value": None,
        "verdict": "INSUFFICIENT_DATA",
        "null_p10": None, "null_p50": None, "null_p90": None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.oos_permutation_test",
        description=(
            "Permutation null-hypothesis test for the deployed "
            "DecisionScorer's OOS rank-IC. Answers: under H0 (no skill), "
            "how often would shuffled labels produce a rank-IC at least "
            "as extreme as the observed one? Complement to "
            "oos_bootstrap_ci (sampling variability) — together they "
            "form the canonical 'is this signal real' check."
        ),
    )
    p.add_argument(
        "--outcomes", default=str(DEFAULT_OUTCOMES_PATH),
        help=("Path to decision_outcomes.jsonl "
              "(default: data/decision_outcomes.jsonl)"),
    )
    p.add_argument("--n-permutations", type=int,
                   default=DEFAULT_N_PERMUTATIONS, dest="n_permutations",
                   help=("Permutation count (>=100 for 0.01 floor; "
                         "1000 default)."))
    p.add_argument("--oos-fraction", type=float, default=DEFAULT_OOS_FRACTION,
                   dest="oos_fraction",
                   help="Most-recent fraction held out as OOS slice.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--action", choices=("aggregate", "buy", "sell"),
                   default="aggregate",
                   help=("Print only this bucket's verdict in the table "
                         "view. JSON output always includes all three."))
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
              else f"[oos_permutation_test] no records at {args.outcomes}")
        return 1

    _, oos = split_outcomes_temporal(records, oos_fraction=args.oos_fraction)
    scorer = DecisionScorer()
    if not scorer.is_trained:
        msg = {"status": "scorer_not_trained",
               "hint": "no pickle at data/ml/decision_scorer.pkl"}
        print(json.dumps(msg, indent=2) if args.json
              else "[oos_permutation_test] scorer NOT trained")
        return 1

    result = permutation_test(
        scorer, oos,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )
    result["n_train"] = scorer.n_train

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return _exit_code_from_result(result)

    status = result.get("status", "?")
    if status != "ok":
        print(f"[oos_permutation_test] {status}  n={result.get('n', 0)}")
        return 1

    print(f"[oos_permutation_test] n={result['n']}  "
          f"n_perm={result['n_permutations']}  "
          f"n_train={scorer.n_train}")
    print(f"  α_strict={ALPHA_STRICT}  α_weak={ALPHA_WEAK}")

    def _fmt(bucket_name: str) -> None:
        b = result[bucket_name]
        if b["rank_ic"] is None:
            print(f"  {bucket_name:<11} n={b['n']}  (insufficient data)")
            return
        marker = ""
        if b["verdict"] == "STATISTICALLY_SIGNIFICANT":
            marker = "  ✓ real edge"
        elif b["verdict"] == "WEAKLY_SIGNIFICANT":
            marker = "  ~ marginal"
        elif b["verdict"] == "AT_NOISE":
            marker = "  ✗ at noise"
        print(f"  {bucket_name:<11} n={b['n']:<5} "
              f"ic={b['rank_ic']:+.4f}  p={b['p_value']:.4f}  "
              f"null=[{b['null_p10']:+.3f},{b['null_p50']:+.3f},"
              f"{b['null_p90']:+.3f}]  {b['verdict']}{marker}")

    _fmt("aggregate")
    _fmt("buy")
    _fmt("sell")

    return _exit_code_from_result(result)


def _exit_code_from_result(result: dict) -> int:
    """Exit 0 when AT LEAST the gate-relevant BUY bucket reaches significance
    (the keep-or-kill criterion); 1 otherwise. The aggregate bucket can
    read significant while BUY is at noise — the gate would then be
    riding a SELL-driven edge it never acts on. Operator-decisive
    semantics: this CLI's exit code answers 'should I keep the gate on?',
    not 'is there any signal anywhere'."""
    if result.get("status") != "ok":
        return 1
    buy = result.get("buy", {})
    verdict = buy.get("verdict")
    if verdict in ("STATISTICALLY_SIGNIFICANT", "WEAKLY_SIGNIFICANT"):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

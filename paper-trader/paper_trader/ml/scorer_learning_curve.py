"""DecisionScorer learning-curve analyzer — does adding more outcomes actually
improve the model's out-of-sample rank skill, or is the corpus already past
the point where more data helps?

This is the *quant-decisive* follow-up to ``baseline_compare``'s
``MLP_WORSE_THAN_TRIVIAL`` / ``MLP_NO_BETTER_THAN_TRIVIAL`` verdict: the
baseline analyzer answers "is the MLP worth its complexity TODAY at
n_train=4987?" — but a skeptical operator immediately asks "if I had 10x
more data, would the gap close, hold, or widen?" That's a different
question entirely, and no existing tool answers it.

Method (mirrors ``baseline_compare`` / ``feature_ablation`` discipline):

1. Pick a fixed temporal-OOS holdout via
   ``validation.split_outcomes_temporal`` (the exact split every
   sibling analyzer uses; identical sim_date sort order, identical
   semantics). The holdout is the SAME slice the deployed pickle's
   ``oos_ic`` was reported against — so a learning-curve point at
   ``n_train=4987`` should approximately equal the deployed ledger
   reading, a built-in no-drift cross-check.
2. From the in-sample portion, train DecisionScorer at a ladder of
   sizes: ``LADDER = (250, 500, 1000, 2500, 5000, ALL)`` (filtered to
   what's actually available). Each training point uses the most
   recent ``n`` rows of the in-sample portion — the same trailing-tail
   selection the continuous loop uses (``MAX_OUTCOMES_FOR_TRAINING``).
3. For each ``n``, train ``--seeds`` (default 3) models on the same
   slice with different ``MLP_CONFIG`` ``random_state`` seeds, score
   each against the holdout, report mean ± std of OOS rank-IC and
   directional accuracy. Multi-seed because at small ``n`` (250) a
   single-seed rank-IC is dominated by initialization variance — the
   verdict ladder reads the *mean* curve so a noisy ``n=250 > n=500``
   inversion at one seed cannot fabricate a DEGRADING verdict.
4. Read the **shape** of the mean rank-IC curve to a verdict:

   - ``MONOTONE_LEARNING`` — strictly improving across every step AND
     the largest ``n`` exceeds the smallest by > ``LEARNING_DELTA``.
     Implies "more data would help."
   - ``SATURATED`` — the largest ``n`` is within ``DELTA_TOL`` of the
     mid ``n``; the curve has plateaued. Implies "more data does not
     close the gap; architecture is the bottleneck."
   - ``U_SHAPED`` — the curve peaks at a mid ``n`` and falls at the
     largest. Implies the trailing 5000-row window has drifted from
     the regime that earlier outcomes describe (the model is being
     trained on stale data). Quant-actionable: shrink
     ``MAX_OUTCOMES_FOR_TRAINING``.
   - ``DEGRADING`` — the largest ``n`` is strictly worse than the
     smallest by > ``LEARNING_DELTA`` (no peak in middle).
   - ``NO_SKILL`` — every mean rank-IC is below ``MIN_SKILL_IC``; the
     model has no signal at any data size (architecture / feature
     ceiling).
   - ``INSUFFICIENT_DATA`` — fewer than ``MIN_TOTAL`` rows in the
     corpus (cannot honestly carve a holdout and run the smallest
     ladder point).

The ladder + verdict is **scoring-pipeline isolated**: training is
redirected to a per-run temp pickle path via the ``train_scorer(path=...)``
override added in the same change; the deployed pickle is never touched
(``feature_ablation``'s read-only discipline, generalized — we *do*
train, but never trample the live artifact). The OOS rank-IC reuses the
same ``calibration._spearman`` and SELL sign-flip every sibling analyzer
shares, so the numbers can never drift from ``baseline_compare`` /
``calibration --oos``.

Slow by construction — training six MLPs × ``--seeds`` × the
``MLP_CONFIG`` ``max_iter`` cap is a one-time diagnostic, NOT a per-cycle
ledger. CLI-only.

```bash
cd /home/zeph/trading-intelligence/paper-trader
python3 -m paper_trader.ml.scorer_learning_curve              # default 3 seeds, table out
python3 -m paper_trader.ml.scorer_learning_curve --seeds 5    # tighter variance, slower
python3 -m paper_trader.ml.scorer_learning_curve --json       # machine-readable
```
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from .calibration import _spearman
from .decision_scorer import (
    MLP_CONFIG,
    DecisionScorer,
    PRED_CLAMP_PCT,
    _to_float,
    train_scorer,
)

# Verdict thresholds — module-level so a tuning change is one reviewable
# edit (mirrors baseline_compare / calibration thresholds).
MIN_TOTAL = 800          # need enough corpus to carve a holdout + run the smallest ladder
HOLDOUT_FRACTION = 0.15  # most-recent slice held out for every ladder point
LADDER = (250, 500, 1000, 2500, 5000)  # training sizes to sweep; ALL is added when distinct
DEFAULT_SEEDS = 3        # per-n training repeats for variance control
LEARNING_DELTA = 0.04    # min rank-IC gain end-vs-start to call LEARNING / DEGRADING
DELTA_TOL = 0.02         # tolerance for "same as mid n" → SATURATED
MIN_SKILL_IC = 0.05      # below this everywhere ⇒ NO_SKILL


def _load_records(path: "Path | str") -> list[dict]:
    """Load JSONL outcomes — best-effort, per-line parse (a single corrupt
    line never blocks the rest; same defensive discipline as
    ``run_continuous_backtests.py``'s outcomes reader).
    """
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict] = []
    try:
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


def _holdout_split(records: list[dict],
                   oos_fraction: float = HOLDOUT_FRACTION
                   ) -> tuple[list[dict], list[dict]]:
    """Reuse ``validation.split_outcomes_temporal`` for the sim_date-sorted
    chronological holdout — single source of truth across analyzers.

    Falls back to a trailing-slice split when the validation module is
    unavailable (same defensive degrade ``_train_decision_scorer`` already
    accepts), so this analyzer never silently no-ops on an unrelated import
    failure.
    """
    try:
        from paper_trader.validation import split_outcomes_temporal
        return split_outcomes_temporal(records, oos_fraction=oos_fraction)
    except Exception:
        if not records:
            return [], []
        n_oos = max(1, int(len(records) * oos_fraction))
        if n_oos >= len(records):
            return list(records), []
        return records[:-n_oos], records[-n_oos:]


def _predict_rank_ic(scorer, oos: list[dict]) -> tuple[float | None, float | None, int]:
    """Return (rank_ic, dir_acc, n_pairs) for ``scorer`` on ``oos``.

    Mirrors the EXACT predict path / SELL sign-flip / failed-row drop /
    label-clamp ``_oos_rank_metrics`` uses in run_continuous_backtests.py,
    so this analyzer's numbers and the per-cycle ledger's must agree by
    construction (the "no-drift across siblings" cross-check). Never
    raises — a per-row predict failure drops just that row.
    """
    if not oos or not getattr(scorer, "is_trained", False):
        return None, None, 0
    preds: list[float] = []
    acts: list[float] = []
    pwm = getattr(scorer, "predict_with_meta", None)
    use_meta = callable(pwm)
    for r in oos:
        try:
            if use_meta:
                meta = pwm(
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
            if p != p:  # NaN
                continue
            a_raw = _to_float(r.get("forward_return_5d"), float("nan"))
            is_sell = str(r.get("action") or "BUY").upper() == "SELL"
            a = -a_raw if is_sell else a_raw
            if a != a:  # NaN
                continue
            a = max(-PRED_CLAMP_PCT, min(PRED_CLAMP_PCT, float(a)))
            preds.append(p)
            acts.append(a)
        except Exception:
            continue
    n = len(preds)
    if n < 2:
        return None, None, n
    ic = _spearman(np.asarray(preds, dtype=float), np.asarray(acts, dtype=float))
    rank_ic = round(float(ic), 4) if ic == ic else None
    dir_pairs = [(p, a) for p, a in zip(preds, acts) if p != 0.0 and a != 0.0]
    dir_acc = None
    if dir_pairs:
        hits = sum(1 for p, a in dir_pairs if (p > 0) == (a > 0))
        dir_acc = round(hits / len(dir_pairs), 4)
    return rank_ic, dir_acc, n


def _train_and_score_once(train_records: list[dict], oos: list[dict],
                          seed: int) -> tuple[float | None, float | None, int, int]:
    """Train one DecisionScorer at ``seed``, evaluate on ``oos``.

    Trains into a unique tmp pickle path so the deployed
    ``decision_scorer.pkl`` is never trampled. Loads that pickle via
    ``DecisionScorer()`` — but the constructor reads the MODULE-LEVEL
    ``SCORER_PATH``, so we monkey-patch it for the load only (atomic
    swap inside a ``try/finally``; even on a train-or-load exception the
    deployed path is restored).

    Returns ``(rank_ic, dir_acc, n_pairs, n_train_actual)``. The fourth
    value is the post-validation count train_scorer ACTUALLY trained on
    (deduped, label-validated) — usually slightly below the input
    ``len(train_records)`` because of the in-trainer dedup. On any
    training or evaluation failure, returns ``(None, None, 0, 0)`` —
    a per-rung crash MUST NOT abort the whole ``analyze`` sweep AND
    must restore both module globals (SCORER_PATH, MLP_CONFIG seed) so
    a downstream caller never sees a leaked test/temp path.
    """
    import paper_trader.ml.decision_scorer as ds_mod

    # Per-call temp pickle so concurrent learning-curve runs can never
    # race on the same file. (CLI is single-threaded so concurrency
    # isn't expected, but the isolation makes the function reentrant
    # and matches the test_pickle_rewrite_is_picked_up tmp_path idiom.)
    with tempfile.TemporaryDirectory(prefix="lc_train_") as tmpdir:
        tmp_pkl = Path(tmpdir) / "scorer.pkl"
        # Patch MLP_CONFIG's random_state for THIS train only — restored
        # in finally so the module-global is byte-identical on return.
        original_seed = ds_mod.MLP_CONFIG.get("random_state")
        ds_mod.MLP_CONFIG["random_state"] = int(seed)
        # Save and swap SCORER_PATH so the freshly trained pickle loads
        # into a DecisionScorer instance without hitting the deployed
        # path (the constructor reads SCORER_PATH at import-time, so a
        # kwarg there would need a deeper refactor).
        original_path = ds_mod.SCORER_PATH
        try:
            try:
                result = train_scorer(train_records, path=tmp_pkl)
            except Exception:
                # Training crash on this rung — degrade to "no data
                # point" rather than aborting the entire sweep, AND
                # ensure the outer finally restores SCORER_PATH /
                # MLP_CONFIG['random_state'] so subsequent rungs are
                # unaffected. The verdict reader excludes None rows.
                return None, None, 0, 0
            if result.get("status") != "ok":
                return None, None, 0, int(result.get("n", 0) or 0)
            ds_mod.SCORER_PATH = tmp_pkl
            scorer = DecisionScorer()
            ic, da, n = _predict_rank_ic(scorer, oos)
            return ic, da, n, int(result.get("n", 0) or 0)
        finally:
            ds_mod.SCORER_PATH = original_path
            if original_seed is not None:
                ds_mod.MLP_CONFIG["random_state"] = original_seed


def _mean_std(vals: list[float | None]) -> tuple[float | None, float | None, int]:
    """Mean and std of the non-None values; returns (mean, std, n_used)."""
    clean = [v for v in vals if v is not None and v == v]
    if not clean:
        return None, None, 0
    arr = np.asarray(clean, dtype=float)
    return round(float(arr.mean()), 4), round(float(arr.std()), 4), len(clean)


def _verdict(curve: list[dict]) -> str:
    """Read the shape of the mean rank-IC curve.

    Order of checks matters — INSUFFICIENT_DATA wins over NO_SKILL, and
    U_SHAPED is checked BEFORE DEGRADING so a curve that peaks in the
    middle isn't misclassified as a strict end-vs-start regression.
    """
    points = [p for p in curve if p.get("mean_rank_ic") is not None]
    if len(points) < 2:
        return "INSUFFICIENT_DATA"
    ics = [p["mean_rank_ic"] for p in points]
    # NO_SKILL — NO rung ever achieves predictive skill above the threshold.
    # The prior `all(abs(v) < MIN_SKILL_IC)` rule misclassified pure-noise
    # corpora as DEGRADING: an MLP trained on noise produces ICs that
    # wander around zero with seed-dependent variance, and one rung's
    # |IC|≥MIN_SKILL_IC from that wander defeated the absolute-value AND.
    # The operator-meaningful semantic is asymmetric: a NEGATIVE IC is
    # still "no useful skill" (anti-predictive at noise level), so the
    # check is on the maximum (positive) IC reached — if no rung ever
    # exceeded the threshold, the model never learned anything, period.
    # This complements (does NOT replace) DEGRADING below, which now
    # additionally requires the START to have had skill (otherwise a
    # decline from sub-threshold to anti-predictive is more honestly
    # NO_SKILL than "degrading skill the model never had").
    if max(ics) < MIN_SKILL_IC:
        return "NO_SKILL"
    first = ics[0]
    last = ics[-1]
    delta = last - first
    # U_SHAPED — the peak is INTERIOR (not at first or last) AND the last
    # is at least DELTA_TOL below the peak. Catches the
    # "stale-data tail drags the curve down" failure mode that a pure
    # end-vs-start rule misses.
    if len(points) >= 3:
        peak_idx = int(np.argmax(ics))
        if 0 < peak_idx < len(points) - 1:
            peak = ics[peak_idx]
            if (peak - last) > DELTA_TOL and (peak - first) > DELTA_TOL:
                return "U_SHAPED"
    # MONOTONE_LEARNING — strictly increasing AND end clears start by margin.
    strictly_up = all(ics[i] >= ics[i - 1] - DELTA_TOL for i in range(1, len(ics)))
    if strictly_up and delta >= LEARNING_DELTA:
        return "MONOTONE_LEARNING"
    # DEGRADING — end is materially worse than start AND start had real
    # skill. The `first >= MIN_SKILL_IC` precondition prevents calling a
    # "decline" from sub-threshold to anti-predictive a degradation —
    # that pattern is more honestly NO_SKILL (the model never had skill
    # to degrade from). The NO_SKILL clause above already catches the
    # max(ics) < MIN_SKILL_IC case; this guard handles the asymmetric
    # case where start was barely above threshold and end is well below.
    if delta <= -LEARNING_DELTA and first >= MIN_SKILL_IC:
        return "DEGRADING"
    # SATURATED — endpoints close, mid close to last (curve plateaued).
    if abs(delta) < LEARNING_DELTA and len(points) >= 3:
        mid = ics[len(ics) // 2]
        if abs(last - mid) < DELTA_TOL:
            return "SATURATED"
    # Fall-through: small change, no clear shape.
    return "SATURATED"


def analyze(records: "list[dict] | Path | str",
            seeds: int = DEFAULT_SEEDS,
            ladder: tuple[int, ...] = LADDER,
            holdout_fraction: float = HOLDOUT_FRACTION) -> dict:
    """Run the learning-curve sweep on ``records`` (or a JSONL path).

    Returns ``{"verdict", "curve", "n_total", "n_holdout", "ladder", "seeds",
    "holdout_fraction", "hint"}``. Each ``curve`` row carries:
    ``{"train_n_requested", "train_n_actual", "mean_rank_ic", "std_rank_ic",
    "mean_dir_acc", "n_oos_pairs", "n_successful_seeds"}``. None values are
    honest signals — the caller can distinguish "not enough data at this
    rung" (mean None) from "successful but flat" (mean ≈ 0).

    Never raises — any unexpected fault returns
    ``{"verdict": "INSUFFICIENT_DATA", "hint": "<error class name>"}`` so
    a downstream consumer (CLI exit code, future ledger wiring) sees a
    structured failure not a crash.
    """
    try:
        if isinstance(records, (str, Path)):
            recs = _load_records(records)
        else:
            recs = list(records)
        n_total = len(recs)
        if n_total < MIN_TOTAL:
            return {
                "verdict": "INSUFFICIENT_DATA",
                "curve": [],
                "n_total": n_total,
                "n_holdout": 0,
                "ladder": list(ladder),
                "seeds": seeds,
                "holdout_fraction": holdout_fraction,
                "hint": f"need >= {MIN_TOTAL} outcomes, have {n_total}",
            }

        in_sample, holdout = _holdout_split(recs, oos_fraction=holdout_fraction)
        if len(holdout) < 30 or len(in_sample) < min(ladder):
            return {
                "verdict": "INSUFFICIENT_DATA",
                "curve": [],
                "n_total": n_total,
                "n_holdout": len(holdout),
                "ladder": list(ladder),
                "seeds": seeds,
                "holdout_fraction": holdout_fraction,
                "hint": (f"holdout={len(holdout)} (need >= 30) and "
                         f"in_sample={len(in_sample)} (need >= "
                         f"{min(ladder)})"),
            }

        # Build the actual ladder: requested sizes that fit in the
        # in_sample portion, plus ALL_AVAILABLE if distinct from the
        # largest requested. De-dup keeps each rung at most once.
        rungs: list[int] = sorted(
            {n for n in ladder if n <= len(in_sample)} | {len(in_sample)})
        if not rungs:
            return {
                "verdict": "INSUFFICIENT_DATA",
                "curve": [],
                "n_total": n_total,
                "n_holdout": len(holdout),
                "ladder": list(ladder),
                "seeds": seeds,
                "holdout_fraction": holdout_fraction,
                "hint": "no ladder rung fits in_sample",
            }

        # Use trailing slice of in_sample at each rung — same selection
        # strategy as the continuous loop's MAX_OUTCOMES_FOR_TRAINING tail
        # (the deployed pickle is trained the same way).
        curve: list[dict] = []
        for n in rungs:
            train_slice = in_sample[-n:]
            ics: list[float | None] = []
            das: list[float | None] = []
            actual_train_n = 0
            n_oos_used = 0
            n_succ = 0
            for s in range(seeds):
                ic, da, n_pairs, n_train = _train_and_score_once(
                    train_slice, holdout, seed=42 + s)
                if ic is not None:
                    n_succ += 1
                ics.append(ic)
                das.append(da)
                actual_train_n = max(actual_train_n, n_train)
                n_oos_used = max(n_oos_used, n_pairs)
            mean_ic, std_ic, _n_ic = _mean_std(ics)
            mean_da, _std_da, _ = _mean_std(das)
            curve.append({
                "train_n_requested": n,
                "train_n_actual": actual_train_n,
                "mean_rank_ic": mean_ic,
                "std_rank_ic": std_ic,
                "mean_dir_acc": mean_da,
                "n_oos_pairs": n_oos_used,
                "n_successful_seeds": n_succ,
            })
        verdict = _verdict(curve)
        return {
            "verdict": verdict,
            "curve": curve,
            "n_total": n_total,
            "n_holdout": len(holdout),
            "ladder": list(ladder),
            "seeds": seeds,
            "holdout_fraction": holdout_fraction,
        }
    except Exception as exc:
        return {
            "verdict": "INSUFFICIENT_DATA",
            "curve": [],
            "n_total": 0,
            "n_holdout": 0,
            "ladder": list(ladder),
            "seeds": seeds,
            "holdout_fraction": holdout_fraction,
            "hint": f"{type(exc).__name__}: {exc}",
        }


def _default_outcomes_path() -> Path:
    return (Path(__file__).resolve().parent.parent.parent
            / "data" / "decision_outcomes.jsonl")


def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.scorer_learning_curve",
        description="Sweep DecisionScorer training-set size; report OOS "
                    "rank-IC vs n_train. Trains into a temp pickle — the "
                    "deployed scorer.pkl is NEVER touched.",
    )
    p.add_argument("--outcomes", default=str(_default_outcomes_path()),
                   help="Path to decision_outcomes.jsonl")
    p.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                   help="Per-rung MLP_CONFIG.random_state repeats "
                        "(default 3; raise for tighter variance, slower).")
    p.add_argument("--holdout-fraction", type=float, default=HOLDOUT_FRACTION,
                   dest="holdout_fraction",
                   help=f"Temporal-OOS slice size (default "
                        f"{HOLDOUT_FRACTION}).")
    p.add_argument("--json", action="store_true",
                   help="Emit JSON instead of a table.")
    return p


def main(argv: "list[str] | None" = None) -> int:
    import sys

    args = _build_arg_parser().parse_args(
        sys.argv[1:] if argv is None else argv)

    rep = analyze(args.outcomes, seeds=args.seeds,
                  holdout_fraction=args.holdout_fraction)
    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
    else:
        print(f"[scorer_learning_curve] verdict = {rep.get('verdict')}  "
              f"n_total={rep.get('n_total')}  "
              f"n_holdout={rep.get('n_holdout')}  "
              f"seeds={rep.get('seeds')}")
        if rep.get("hint"):
            print(f"  hint: {rep['hint']}")
        if rep.get("curve"):
            print(f"  {'train_n':>10}{'actual':>10}{'mean_IC':>12}"
                  f"{'std_IC':>10}{'mean_DA':>10}{'n_pairs':>10}"
                  f"{'seeds':>10}")
            for row in rep["curve"]:
                ic = row.get("mean_rank_ic")
                std = row.get("std_rank_ic")
                da = row.get("mean_dir_acc")
                print(f"  {row.get('train_n_requested', 0):>10}"
                      f"{row.get('train_n_actual', 0):>10}"
                      f"{'n/a' if ic is None else f'{ic:+.4f}':>12}"
                      f"{'n/a' if std is None else f'{std:.4f}':>10}"
                      f"{'n/a' if da is None else f'{da:.4f}':>10}"
                      f"{row.get('n_oos_pairs', 0):>10}"
                      f"{row.get('n_successful_seeds', 0):>10}")
    # Exit codes: 0 = useful diagnostic (any decisive verdict — even
    # DEGRADING is useful information), 1 = degenerate (no curve to read).
    bad = {"INSUFFICIENT_DATA"}
    return 1 if rep.get("verdict") in bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

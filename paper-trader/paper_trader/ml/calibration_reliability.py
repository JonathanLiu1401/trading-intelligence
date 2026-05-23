"""Reliability analyzer for ``DecisionScorer.predict_calibrated``.

``predict_calibrated`` (decision_scorer.py pass #10) quantile-maps a raw
prediction onto the *training-set realized-return distribution*. The CLAIM
is that the calibrated value is the honest 5-day return magnitude — a raw
``+8`` calibrated to, say, ``+5.2`` means "this rank position historically
corresponded to a +5.2% realized return". Until this module, that claim was
untested OOS: the existing ``calibration`` module bins by ``predict()`` (the
clamped raw output), not by the calibrated value, and the existing skill
ledger only trends scalar ``oos_rmse`` / ``oos_ic``.

This module answers, OOS:

  * **Does calibrated_pred ≈ realized?** Per-decile ``mean_calibrated_pred``
    vs ``mean_realized`` reliability curve, on the SAME temporal holdout
    used by ``_train_decision_scorer``. A WELL_CALIBRATED verdict means the
    calibrated reading actually delivers on its honest-magnitude promise.
  * **Is the calibration step a real improvement, or cosmetic?**
    ``vs_raw_bias_reduction`` reports the mean |decile error| of
    ``predict_calibrated`` MINUS the same metric for raw ``predict()`` on the
    identical OOS pairs. Positive ⇒ calibration narrowed the magnitude bias;
    ~0 ⇒ no measurable improvement; negative ⇒ the quantile mapping
    OVERshoots and made magnitudes WORSE OOS (a genuine regression signal
    a quant would want to know before trusting the calibrated value).

Same operational discipline as the sibling ``calibration`` module:
read-only, no train / pickle / ``build_features`` / ``N_FEATURES`` /
trade-path touch, and it never raises in the analyzer functions — the
unattended continuous loop can wire this into a per-cycle ledger entry
with the same best-effort discipline as the scorer/baseline/llm-annotation
ledgers without risk of breaking a cycle.

Verdict thresholds mirror ``calibration.py``'s crisp bands so the two
modules are directly comparable (``calibration`` reports the raw-predict
view, ``calibration_reliability`` reports the calibrated-predict view; both
use the same OOS pairs and the same decile binning math).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Re-use the same thresholds as the raw-predict reliability module so the
# two reports are directly comparable. A single source of truth means a
# threshold tune in ``calibration.py`` automatically propagates here.
from .calibration import (
    BIAS_TOL_PCT,
    MIN_PAIRS,
    MONOTONE_GOOD,
    MONOTONE_MIN,
    SPEARMAN_GOOD,
    SPEARMAN_MIN,
    _spearman,
)


def _empty_report(n: int, hint: str) -> dict:
    return {
        "status": "insufficient_data",
        "verdict": "INSUFFICIENT_DATA",
        "n": n,
        "spearman": None,
        "monotone_fraction": None,
        "mean_abs_decile_error": None,
        "raw_mean_abs_decile_error": None,
        "vs_raw_bias_reduction": None,
        "buckets": [],
        "hint": hint,
    }


def calibrated_reliability_report(pairs, n_buckets: int = 10) -> dict:
    """Bin (calibrated_pred, raw_pred, realized) triples by *calibrated*
    prediction quantile and report per-decile reliability + a side-by-side
    bias-reduction metric vs the raw predict() reading.

    ``pairs`` is any iterable of 3-tuples ``(calibrated_pred, raw_pred,
    realized)``. The two prediction columns must come from the same model on
    the same input vectors so the per-row comparison is apples-to-apples
    (the analyzer takes them as already-paired triples rather than recomputing
    to keep this module model-agnostic — ``scorer_calibrated_reliability``
    below does the per-row predict pass and feeds this).

    Non-finite entries are dropped (a single inf/nan must not poison the
    report — the same hardening class as ``calibration_report``).

    Returns a JSON-safe dict with ``calibration_report``'s shape plus two
    extra keys:

      * ``raw_mean_abs_decile_error`` — the same metric computed for the raw
        ``predict()`` output on the IDENTICAL decile bins (so a quant can
        eyeball whether calibration helped, hurt, or did nothing).
      * ``vs_raw_bias_reduction`` — ``raw_mean_abs_decile_error -
        mean_abs_decile_error`` (pp). Positive ⇒ calibration narrowed the
        magnitude bias; ~0 ⇒ no measurable improvement; negative ⇒ the
        quantile mapping made magnitudes WORSE OOS.
    """
    clean: list[tuple[float, float, float]] = []
    for cp, rp, y in pairs:
        try:
            cpf = float(cp)
            rpf = float(rp)
            yf = float(y)
        except (TypeError, ValueError):
            continue
        if np.isfinite(cpf) and np.isfinite(rpf) and np.isfinite(yf):
            clean.append((cpf, rpf, yf))

    n = len(clean)
    if n < MIN_PAIRS:
        return _empty_report(n, f"need ≥{MIN_PAIRS} finite pairs, have {n}")

    # Sort by calibrated prediction so deciles cut the calibrated axis (the
    # value the operator would read). Raw predictions ride along on the same
    # row so the bias-reduction comparison uses the EXACT same decile bins.
    clean.sort(key=lambda t: t[0])
    CP = np.array([c[0] for c in clean], dtype=np.float64)
    RP = np.array([c[1] for c in clean], dtype=np.float64)
    Y = np.array([c[2] for c in clean], dtype=np.float64)

    k = max(2, min(n_buckets, n // 3))   # never fewer than 3 per bucket
    buckets: list[dict] = []
    realized_means: list[float] = []
    calibrated_abs_errs: list[float] = []
    raw_abs_errs: list[float] = []
    for i in range(k):
        lo = i * n // k
        hi = (i + 1) * n // k
        if hi <= lo:
            continue
        seg_cp = CP[lo:hi]
        seg_rp = RP[lo:hi]
        seg_y = Y[lo:hi]
        mcp = float(seg_cp.mean())
        mrp = float(seg_rp.mean())
        my = float(seg_y.mean())
        realized_means.append(my)
        calibrated_abs_errs.append(abs(mcp - my))
        raw_abs_errs.append(abs(mrp - my))
        buckets.append({
            "idx": i + 1,
            "n": int(hi - lo),
            "mean_calibrated_pred": round(mcp, 4),
            "mean_raw_pred": round(mrp, 4),
            "mean_realized": round(my, 4),
            "pred_lo": round(float(seg_cp.min()), 4),
            "pred_hi": round(float(seg_cp.max()), 4),
        })

    # Rank skill over ALL pairs of the CALIBRATED predictions (not bucket
    # means — bucket means hide within-bucket noise and inflate the
    # correlation). ``predict_calibrated`` is MONOTONIC in ``predict()`` by
    # construction (see ``_raw_to_calibrated``), so this Spearman is identical
    # to the raw-predict Spearman on the same rows — surfacing it here is
    # informational, not a second independent metric.
    spearman = _spearman(CP, Y)

    # Monotonicity of the decile realized curve: fraction of adjacent steps
    # that do not go DOWN. A perfectly calibrated scorer's decile realized
    # mean rises with the decile.
    steps = len(realized_means) - 1
    if steps <= 0:
        monotone_fraction = 1.0
    else:
        nondec = sum(1 for j in range(steps)
                     if realized_means[j + 1] >= realized_means[j])
        monotone_fraction = nondec / steps

    mean_abs_decile_error = (float(np.mean(calibrated_abs_errs))
                             if calibrated_abs_errs else 0.0)
    raw_mean_abs_decile_error = (float(np.mean(raw_abs_errs))
                                 if raw_abs_errs else 0.0)
    bias_reduction = raw_mean_abs_decile_error - mean_abs_decile_error

    if spearman < SPEARMAN_MIN or monotone_fraction < MONOTONE_MIN:
        verdict = "MISCALIBRATED"
        hint = ("predicted order does not track realized outcomes — "
                "calibrated value carries no usable magnitude signal")
    elif mean_abs_decile_error > BIAS_TOL_PCT:
        verdict = "DIRECTIONAL_BUT_BIASED"
        hint = ("calibrated magnitude is biased — trust the sign/ordering, "
                f"discount the predicted % (decile error "
                f"{mean_abs_decile_error:.1f}pp > {BIAS_TOL_PCT:.1f}pp)")
    elif spearman >= SPEARMAN_GOOD and monotone_fraction >= MONOTONE_GOOD:
        verdict = "WELL_CALIBRATED"
        hint = ("calibrated % tracks realized % within tolerance — the "
                "quantile-mapped reading is the honest magnitude")
    else:
        verdict = "WEAK_SIGNAL"
        hint = ("some rank skill but below the strong bar — calibrated "
                "value is a tie-breaker, not a primary signal")

    return {
        "status": "ok",
        "verdict": verdict,
        "n": n,
        "spearman": round(spearman, 4),
        "monotone_fraction": round(monotone_fraction, 4),
        "mean_abs_decile_error": round(mean_abs_decile_error, 4),
        "raw_mean_abs_decile_error": round(raw_mean_abs_decile_error, 4),
        "vs_raw_bias_reduction": round(bias_reduction, 4),
        "buckets": buckets,
        "hint": hint,
    }


def scorer_calibrated_reliability(scorer, records, n_buckets: int = 10) -> dict:
    """Run ``calibrated_reliability_report`` for a DecisionScorer over outcome
    records (the ``data/decision_outcomes.jsonl`` row shape).

    Per-row pass: for each record produce ``(calibrated_pred, raw_pred,
    realized)`` then feed the bunch to the report. Action-aligned exactly
    like ``train_scorer`` / ``scorer_calibration``: a SELL's realized
    goodness is ``-forward_return_5d`` (a drop after a SELL was the *right*
    call). Records missing/with a non-finite ``forward_return_5d`` are
    skipped by the report's own finite filter.

    Records whose scorer carries no ``label_quantiles`` table (legacy
    pickle written before pass #10) are skipped at the row level —
    ``predict_calibrated`` returns None for those, and a triple containing
    None can't enter the calibrated-axis comparison. The report's
    INSUFFICIENT_DATA verdict then makes the unavailability visible.
    """
    triples: list[tuple] = []
    pwm = getattr(scorer, "predict_with_meta", None)
    if not callable(pwm):
        return _empty_report(0, "scorer has no predict_with_meta")
    for r in records:
        try:
            meta = pwm(
                ml_score=r.get("ml_score", 0.0),
                rsi=r.get("rsi"),
                macd=r.get("macd"),
                mom5=r.get("mom5"),
                mom20=r.get("mom20"),
                regime_mult=r.get("regime_mult", 1.0),
                ticker=r.get("ticker", ""),
                vol_ratio=r.get("vol_ratio"),
                bb_pos=r.get("bb_position"),
                news_urgency=r.get("news_urgency"),
                news_article_count=r.get("news_article_count"),
            )
        except Exception:
            continue
        if not isinstance(meta, dict):
            continue
        cal = meta.get("calibrated")
        raw = meta.get("pred")
        if cal is None or raw is None:
            continue
        y = r.get("forward_return_5d")
        if y is None:
            continue
        action = str(r.get("action") or "BUY").upper()
        target = -y if action == "SELL" else y
        triples.append((cal, raw, target))
    return calibrated_reliability_report(triples, n_buckets=n_buckets)


def scorer_calibrated_reliability_oos(scorer, records,
                                      oos_fraction: float = 0.2,
                                      n_buckets: int = 10) -> dict:
    """Reliability on the *temporal out-of-sample holdout*.

    ``predict_calibrated`` is fit on the training pickle's label distribution
    — so the in-sample report below would always look perfect by
    construction. The OOS view answers the only honest question: does the
    calibrated value still track realized magnitude on data the calibration
    table never saw?

    Reuses ``paper_trader.validation.split_outcomes_temporal`` — the EXACT
    split ``run_continuous_backtests._train_decision_scorer`` uses for
    ``oos_rmse``/``oos_ic`` — so this decile view describes the same
    holdout as the ledger's scalar OOS metrics (single source of truth,
    no split mismatch).

    Returns ``calibrated_reliability_report``'s dict with three extra keys:
    ``oos_n``, ``train_n``, ``oos_fraction``. Best-effort — a split failure
    degrades to "no holdout" (INSUFFICIENT_DATA) rather than raising.
    """
    try:
        from paper_trader.validation import split_outcomes_temporal
        train_recs, oos_recs = split_outcomes_temporal(
            list(records or []), oos_fraction=oos_fraction
        )
    except Exception:
        train_recs, oos_recs = list(records or []), []
    rep = scorer_calibrated_reliability(scorer, oos_recs, n_buckets=n_buckets)
    rep["oos_n"] = len(oos_recs)
    rep["train_n"] = len(train_recs)
    rep["oos_fraction"] = oos_fraction
    return rep


def analyze(outcomes_path: "Path | str | None" = None,
            oos_only: bool = True,
            n_buckets: int = 10) -> dict:
    """Convenience: load outcomes, load deployed scorer, run the OOS report.

    Mirrors the ``baseline_compare.analyze`` / ``llm_annotation_skill.analyze``
    entry-point shape so the per-cycle ledger wiring is consistent across
    sibling diagnostics. Default ``oos_only=True`` returns the honest holdout
    reading (set to False to get the in-sample report instead — useful only
    as a sanity check on the calibration table, not as a quality verdict).

    Never raises: every fault degrades to an INSUFFICIENT_DATA report with
    an explanatory hint, matching the discipline of
    ``calibration._cli`` / ``baseline_compare.analyze`` / etc.
    """
    try:
        if outcomes_path is None:
            outcomes_path = (Path(__file__).resolve().parent.parent.parent
                             / "data" / "decision_outcomes.jsonl")
        p = Path(outcomes_path)
        if not p.exists():
            return _empty_report(0, f"no outcomes file at {p}")
        records: list[dict] = []
        for ln in p.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                records.append(json.loads(ln))
            except Exception:
                pass
        try:
            from .decision_scorer import DecisionScorer
            scorer = DecisionScorer()
        except Exception as e:
            return _empty_report(0, f"scorer load failed: {type(e).__name__}")
        if not scorer.is_trained:
            return _empty_report(0, "scorer not trained — nothing to calibrate")
        if oos_only:
            rep = scorer_calibrated_reliability_oos(
                scorer, records, n_buckets=n_buckets
            )
            rep["slice"] = "oos"
        else:
            rep = scorer_calibrated_reliability(
                scorer, records, n_buckets=n_buckets
            )
            rep["slice"] = "in_sample"
        rep["scorer_n_train"] = scorer.n_train
        rep["outcomes_n"] = len(records)
        return rep
    except Exception as e:
        return _empty_report(0, f"analyze failed: {type(e).__name__}: {e}")


def _print_report(tag: str, rep: dict) -> None:
    print(f"[{tag}] VERDICT: {rep['verdict']}  ({rep['hint']})")
    print(f"  n={rep['n']} spearman={rep['spearman']} "
          f"monotone={rep['monotone_fraction']} "
          f"calibrated_decile_err={rep['mean_abs_decile_error']}pp "
          f"raw_decile_err={rep['raw_mean_abs_decile_error']}pp "
          f"bias_reduction={rep['vs_raw_bias_reduction']:+.2f}pp"
          if rep["vs_raw_bias_reduction"] is not None
          else f"  n={rep['n']} (insufficient)")
    for b in rep.get("buckets") or []:
        print(f"  d{b['idx']:>2} cal[{b['pred_lo']:+8.2f},{b['pred_hi']:+8.2f}] "
              f"cal={b['mean_calibrated_pred']:+7.2f}  "
              f"raw={b['mean_raw_pred']:+7.2f}  "
              f"realized={b['mean_realized']:+7.2f}  n={b['n']}")


def _cli(argv: list[str] | None = None) -> int:
    """``python3 -m paper_trader.ml.calibration_reliability [--in-sample]``
    — calibrated-prediction reliability of the live pickled scorer against
    the accumulated outcomes tail. Read-only.

    Default mode is the temporal-OOS holdout report (the only honest view).
    ``--in-sample`` adds a side-by-side in-sample read so the optimism gap
    is visible in one invocation.

    Exit code mirrors the sibling ``decision_scorer`` CLI: 0 when a
    WELL_CALIBRATED or WEAK_SIGNAL verdict (some usable signal), 1 when
    INSUFFICIENT_DATA / MISCALIBRATED / DIRECTIONAL_BUT_BIASED. Shell
    callers can gate on ``$?``.
    """
    import sys

    args = sys.argv[1:] if argv is None else argv
    want_in_sample = "--in-sample" in args

    rep = analyze(oos_only=True)
    print(f"scorer n_train={rep.get('scorer_n_train', '?')} "
          f"outcomes={rep.get('outcomes_n', '?')}")
    _print_report("oos", rep)
    if want_in_sample:
        in_s = analyze(oos_only=False)
        print()
        _print_report("in-sample", in_s)
        if (rep["verdict"] == "WELL_CALIBRATED"
                and in_s["verdict"] != "WELL_CALIBRATED"):
            # Reverse of the raw-predict case — predict_calibrated is by
            # construction near-perfect in-sample (it's a quantile lookup),
            # so a degraded OOS view is the more usual surprise direction.
            print("  ⚠ OOS WELL_CALIBRATED but in-sample is "
                  f"{in_s['verdict']} — investigate (should be impossible "
                  "by construction).")
    return 0 if rep["verdict"] in ("WELL_CALIBRATED", "WEAK_SIGNAL") else 1


if __name__ == "__main__":
    raise SystemExit(_cli())

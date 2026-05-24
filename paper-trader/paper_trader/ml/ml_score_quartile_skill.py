"""ml_score-quartile-conditioned scorer rank-skill diagnostic.

Read-only diagnostic. Mirrors ``calibration.py`` / ``conviction_calibration.py``
discipline exactly: never trains, never writes a pickle, never touches
``build_features`` / ``N_FEATURES``, never modifies trade-path state —
safe to run against the unattended continuous loop (AGENTS.md
"Operational quirks a quant should know"). A fault degrades to
``status='error'`` + a verdict; this module never raises (the AGENTS.md
"ledger / diagnostic must not break the cycle" discipline).

The question this answers, which no existing analyzer does:

* ``calibration`` reports the AGGREGATE rank skill of the scorer across
  all outcome rows — a single Spearman over the full corpus.
* ``conviction_calibration`` buckets by the gate's then-applied SIZING,
  asking whether higher conviction predicts higher realized return.
* ``baseline_compare`` compares the MLP's aggregate rank-IC against a
  one-line raw ``ml_score`` baseline.

None of those answer the most directly operational gate-relevance
question: *the conviction gate fires only on BUYs whose ``ml_score``
clears the per-persona threshold (usually 1.0). Within that
above-threshold space, does the scorer's prediction-rank skill stay
constant, concentrate in high-conviction trades (where it matters
economically), or evaporate at the boundary?*

A scorer with rank-IC ≈ 0 aggregate could still be gate-useful IF its
skill is concentrated in the top ml_score quartile — the bucket the
gate actually acts on. Conversely, a positive-aggregate rank-IC that
comes entirely from the BOTTOM ml_score quartile (low-conviction probes
the gate doesn't size up) is NOT a gate-relevant edge.

Quartile cut on ``ml_score`` is the right axis because:

  * It is the same quantity ``_ml_decide`` thresholds against
    (``buy_threshold`` = 0.85 / 1.0 / 1.15 by persona).
  * It is the dominant scorer feature (`feature_importance` consistently
    ranks ``ml_score`` near the top across retrains).
  * It is the only feature an operator can read directly from the live
    decision pipeline without re-running the scorer.

Verdict ladder (crisp, threshold-driven, test-locked):

| Verdict | Trigger |
|---------|---------|
| ``INSUFFICIENT_DATA`` | < ``MIN_PAIRS_PER_BUCKET`` × 4 valid rows |
| ``INVERTED`` | aggregate spearman ≤ -``SPEARMAN_FLAT`` (anti-predictive) |
| ``NO_SKILL`` | every bucket's ``|rank_ic|`` < ``SPEARMAN_FLAT`` |
| ``CONCENTRATED_LOW`` | top quartile rank-IC < ``SPEARMAN_FLAT`` AND bottom quartile rank-IC ≥ ``SPEARMAN_FLAT`` (skill only in the bucket the gate ignores) |
| ``CONCENTRATED_HIGH`` | top quartile rank-IC ≥ ``SPEARMAN_GOOD`` AND bottom quartile rank-IC < ``SPEARMAN_FLAT`` (the gate-favorable shape) |
| ``UNIFORM`` | every bucket's rank-IC ≥ ``SPEARMAN_FLAT`` AND |top-bot rank-IC| < ``SPREAD_TOL`` (skill is broad, not concentrated) |
| ``DIRECTIONAL`` | aggregate spearman ≥ ``SPEARMAN_FLAT`` but does not fit any of the above shape patterns |
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Single source of truth for the tie-aware Spearman that the existing
# scorer-calibration diagnostic uses.
from .calibration import _spearman

# Thresholds at module scope so tests pin exact verdicts AND a tuning
# change is a single, reviewable edit (mirrors `calibration.py`'s
# constants-at-module-scope convention).
MIN_PAIRS_PER_BUCKET = 10   # minimum samples per quartile for a stable Spearman
N_BUCKETS = 4               # quartile cut
SPEARMAN_FLAT = 0.05        # below |this| → no rank skill
SPEARMAN_GOOD = 0.15        # strong rank skill bar
SPREAD_TOL = 0.10           # |top_q − bot_q rank_ic| below this → uniform


def _to_finite_float(v):
    """Coerce to a finite float or None. Mirrors decision_scorer._to_float
    sentinel handling."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


def _empty(reason: str) -> dict:
    return {
        "status": "insufficient_data",
        "verdict": "INSUFFICIENT_DATA",
        "n": 0,
        "buckets": [],
        "aggregate_rank_ic": None,
        "top_quartile_rank_ic": None,
        "bottom_quartile_rank_ic": None,
        "spread": None,
        "n_dropped_no_pred": 0,
        "n_dropped_no_return": 0,
        "n_dropped_no_ml_score": 0,
        "hint": reason,
    }


def _predict_one(scorer, r: dict) -> float | None:
    """Best-effort scorer prediction for one outcome row. Returns None on
    any failure (including the `failed=True` sentinel from
    `predict_with_meta`). Mirrors `_oos_rank_metrics`'s prediction discipline."""
    from .decision_scorer import _to_float
    try:
        pwm = getattr(scorer, "predict_with_meta", None)
        if callable(pwm):
            meta = pwm(
                ml_score=_to_float(r.get("ml_score"), 0.0),
                rsi=r.get("rsi"), macd=r.get("macd"),
                mom5=r.get("mom5"), mom20=r.get("mom20"),
                regime_mult=_to_float(r.get("regime_mult"), 1.0),
                ticker=str(r.get("ticker") or ""),
                vol_ratio=r.get("vol_ratio"),
                bb_pos=r.get("bb_position"),
                news_urgency=r.get("news_urgency"),
                news_article_count=r.get("news_article_count"),
            )
            if meta.get("failed"):
                return None
            v = float(meta.get("pred", 0.0))
        else:
            v = float(scorer.predict(
                ml_score=_to_float(r.get("ml_score"), 0.0),
                rsi=r.get("rsi"), macd=r.get("macd"),
                mom5=r.get("mom5"), mom20=r.get("mom20"),
                regime_mult=_to_float(r.get("regime_mult"), 1.0),
                ticker=str(r.get("ticker") or ""),
                vol_ratio=r.get("vol_ratio"),
                bb_pos=r.get("bb_position"),
                news_urgency=r.get("news_urgency"),
                news_article_count=r.get("news_article_count"),
            ))
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


def build_quartile_skill_report(scorer, records,
                                n_buckets: int = N_BUCKETS) -> dict:
    """Bucket outcome rows into ``n_buckets`` ml_score quantiles and report
    the scorer's per-bucket rank-IC.

    Selection criteria:

      * ``ml_score`` finite — quartile axis must be real
      * ``forward_return_5d`` finite — target must be comparable
      * scorer predicts successfully (no `failed=True` sentinel)
      * ``action`` in {BUY, SELL} — applies the standard SELL sign-flip
        (same convention as ``_oos_rank_metrics`` /
        ``validation.evaluate_scorer_oos``) so "good" has one meaning.
    """
    if not getattr(scorer, "is_trained", False):
        return _empty("scorer is not trained — feature_importance has no model "
                      "to interrogate")
    if not records:
        return _empty("no records supplied")

    n_skip_pred = 0
    n_skip_ret = 0
    n_skip_ms = 0
    triples: list[tuple[float, float, float]] = []  # (ml_score, pred, realized)
    for r in records:
        if not isinstance(r, dict):
            n_skip_ms += 1
            continue
        ms = _to_finite_float(r.get("ml_score"))
        if ms is None:
            n_skip_ms += 1
            continue
        y_raw = _to_finite_float(r.get("forward_return_5d"))
        if y_raw is None:
            n_skip_ret += 1
            continue
        p = _predict_one(scorer, r)
        if p is None:
            n_skip_pred += 1
            continue
        is_sell = str(r.get("action") or "BUY").upper() == "SELL"
        y = -y_raw if is_sell else y_raw
        triples.append((ms, p, y))

    n = len(triples)
    if n < MIN_PAIRS_PER_BUCKET * n_buckets:
        out = _empty(f"need ≥{MIN_PAIRS_PER_BUCKET * n_buckets} valid rows "
                     f"for {n_buckets}-quantile cut, have {n}")
        out["n"] = n
        out["n_dropped_no_pred"] = n_skip_pred
        out["n_dropped_no_return"] = n_skip_ret
        out["n_dropped_no_ml_score"] = n_skip_ms
        return out

    # Stable-sort by ml_score so identical-ml_score rows carry deterministic
    # ordering through the quantile cut. Tied boundary values land on the
    # lower bucket via the integer-index cut below — consistent with
    # `conviction_calibration`'s tie discipline.
    triples.sort(key=lambda t: t[0])
    M = np.array([t[0] for t in triples], dtype=np.float64)
    P = np.array([t[1] for t in triples], dtype=np.float64)
    Y = np.array([t[2] for t in triples], dtype=np.float64)

    buckets: list[dict] = []
    bucket_ics: list[float] = []
    for i in range(n_buckets):
        lo = i * n // n_buckets
        hi = (i + 1) * n // n_buckets
        if hi <= lo:
            continue
        seg_p = P[lo:hi]
        seg_y = Y[lo:hi]
        seg_m = M[lo:hi]
        n_seg = hi - lo
        if n_seg < 2:
            ic = None
        else:
            ic_v = _spearman(seg_p, seg_y)
            ic = round(float(ic_v), 4) if ic_v == ic_v else None
            if ic is not None:
                bucket_ics.append(ic)
        buckets.append({
            "idx": i + 1,
            "n": int(n_seg),
            "ml_score_lo": round(float(seg_m.min()), 4),
            "ml_score_hi": round(float(seg_m.max()), 4),
            "mean_pred": round(float(seg_p.mean()), 4),
            "mean_realized": round(float(seg_y.mean()), 4),
            "rank_ic": ic,
        })

    aggregate_ic = _spearman(P, Y)
    aggregate_ic_r = round(float(aggregate_ic), 4) \
        if aggregate_ic == aggregate_ic else None

    top_ic = buckets[-1]["rank_ic"] if buckets else None
    bot_ic = buckets[0]["rank_ic"] if buckets else None
    spread = None
    if top_ic is not None and bot_ic is not None:
        spread = round(top_ic - bot_ic, 4)

    # Verdict ladder — applied in order so a stronger shape wins. The
    # "inverted aggregate" check is the safety net (the gate is firing on
    # negative-IC predictions overall — alarm worthy).
    verdict: str
    if aggregate_ic_r is not None and aggregate_ic_r <= -SPEARMAN_FLAT:
        verdict = "INVERTED"
        hint = (f"aggregate rank skill is anti-predictive "
                f"(rank_ic={aggregate_ic_r:+.3f}). The scorer is "
                f"ranking outcomes BACKWARDS on this slice — every "
                f"gate firing on its predictions is sized in the "
                f"wrong direction.")
    elif (top_ic is not None and bot_ic is not None
          and top_ic < SPEARMAN_FLAT and bot_ic >= SPEARMAN_FLAT):
        verdict = "CONCENTRATED_LOW"
        hint = (f"rank skill concentrated in the BOTTOM ml_score "
                f"quartile (bot={bot_ic:+.3f}, top={top_ic:+.3f}). "
                f"The buckets the gate ignores carry the signal; "
                f"buckets the gate sizes up have none.")
    elif (top_ic is not None and bot_ic is not None
          and top_ic >= SPEARMAN_GOOD and bot_ic < SPEARMAN_FLAT):
        verdict = "CONCENTRATED_HIGH"
        hint = (f"rank skill concentrated in the TOP ml_score quartile "
                f"(top={top_ic:+.3f}, bot={bot_ic:+.3f}). The gate-"
                f"favorable shape — the scorer's edge IS in the "
                f"high-conviction bucket the gate sizes up.")
    elif (bucket_ics
          and all(abs(ic) < SPEARMAN_FLAT for ic in bucket_ics)):
        verdict = "NO_SKILL"
        hint = (f"every quartile's |rank_ic| < {SPEARMAN_FLAT}. The "
                f"scorer carries no rank skill in any ml_score band — "
                f"the gate is sizing on pure noise.")
    elif (bucket_ics and len(bucket_ics) >= 2
          and all(ic >= SPEARMAN_FLAT for ic in bucket_ics)
          and spread is not None and abs(spread) < SPREAD_TOL):
        verdict = "UNIFORM"
        hint = (f"rank skill is broad — every quartile has "
                f"rank_ic ≥ {SPEARMAN_FLAT} and the top-bottom "
                f"spread ({spread:+.3f}) is within ±{SPREAD_TOL}. The "
                f"scorer's edge is real but not concentrated; the gate's "
                f"high-conviction bias does not exploit it differentially.")
    else:
        verdict = "DIRECTIONAL"
        hint = (f"some aggregate rank skill (rank_ic={aggregate_ic_r}) "
                f"but the per-quartile shape does not match a strong "
                f"verdict. Inspect the bucket breakdown manually.")

    return {
        "status": "ok",
        "verdict": verdict,
        "n": n,
        "buckets": buckets,
        "aggregate_rank_ic": aggregate_ic_r,
        "top_quartile_rank_ic": top_ic,
        "bottom_quartile_rank_ic": bot_ic,
        "spread": spread,
        "n_dropped_no_pred": n_skip_pred,
        "n_dropped_no_return": n_skip_ret,
        "n_dropped_no_ml_score": n_skip_ms,
        "hint": hint,
    }


def load_outcomes(path: Path | str) -> list[dict]:
    """Stream-load outcome records from a JSONL file. Returns ``[]`` on a
    missing file / unparseable line — never raises (mirrors the sibling
    ``conviction_calibration.load_outcomes``)."""
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict] = []
    try:
        with p.open("r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    return out


def analyze(outcomes_path: Path | str | None = None,
            n_buckets: int = N_BUCKETS,
            oos_only: bool = True) -> dict:
    """Convenience: load + build. The CLI uses this. Never raises — a
    fault in the loader degrades to ``insufficient_data``; a fault in
    the analyzer is caught and returned as a ``status='error'`` row.

    ``oos_only=True`` (default) keeps only the most recent 20% of records
    by ``sim_date``, matching ``validation.split_outcomes_temporal``'s
    convention. ``oos_only=False`` analyses the full corpus — useful for
    quick exploration but mixes train + holdout, so a positive verdict
    there does NOT generalize.
    """
    try:
        from .decision_scorer import DecisionScorer
        if outcomes_path is None:
            outcomes_path = (
                Path(__file__).resolve().parent.parent.parent
                / "data" / "decision_outcomes.jsonl")
        recs = load_outcomes(outcomes_path)
        if oos_only and recs:
            try:
                from ..validation import split_outcomes_temporal
                _train, oos = split_outcomes_temporal(recs, oos_fraction=0.2)
                recs = oos
            except Exception:
                # Fall through to full-corpus analysis — sub-optimal but
                # still honest if validation module is unavailable.
                pass
        scorer = DecisionScorer()
        return build_quartile_skill_report(scorer, recs, n_buckets=n_buckets)
    except Exception as exc:
        out = _empty(f"analyze error: {type(exc).__name__}: {exc}")
        out["status"] = "error"
        return out


def _print_report(rep: dict) -> None:
    print(f"VERDICT: {rep['verdict']}  ({rep['hint']})")
    if rep.get("status") != "ok":
        return
    agg = rep.get("aggregate_rank_ic")
    spread = rep.get("spread")
    agg_s = f"{agg:+.3f}" if agg is not None else "n/a"
    spread_s = f"{spread:+.3f}" if spread is not None else "n/a"
    print(f"  n={rep['n']}  aggregate_rank_ic={agg_s}  "
          f"top-bot spread={spread_s}")
    print(f"  dropped: no_pred={rep['n_dropped_no_pred']}  "
          f"no_return={rep['n_dropped_no_return']}  "
          f"no_ml_score={rep['n_dropped_no_ml_score']}")
    print(f"  {'idx':>3}  {'n':>5}  {'ml_score lo':>12}  "
          f"{'ml_score hi':>12}  {'mean_pred':>10}  "
          f"{'mean_real':>10}  {'rank_ic':>8}")
    for b in rep.get("buckets") or []:
        ic_s = f"{b['rank_ic']:+.3f}" if b['rank_ic'] is not None else "  n/a"
        print(f"  {b['idx']:>3}  {b['n']:>5}  "
              f"{b['ml_score_lo']:>12.4f}  {b['ml_score_hi']:>12.4f}  "
              f"{b['mean_pred']:>+10.4f}  {b['mean_realized']:>+10.4f}  "
              f"{ic_s:>8}")


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.ml_score_quartile_skill",
        description="Bucket OOS outcome rows by ml_score quartile and "
                    "report the scorer's per-bucket rank-IC. Answers "
                    "WHERE in the conviction space the scorer's edge sits.",
    )
    p.add_argument("--outcomes", default=None,
                   help="Path to decision_outcomes.jsonl (default: "
                        "data/decision_outcomes.jsonl).")
    p.add_argument("--all", action="store_true",
                   help="Use the full corpus instead of just the OOS tail "
                        "(20% temporal holdout). Default is OOS-only.")
    p.add_argument("--buckets", type=int, default=N_BUCKETS,
                   help=f"Number of quantile buckets (default {N_BUCKETS}).")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of a table.")
    args = p.parse_args(argv)
    rep = analyze(args.outcomes, n_buckets=args.buckets,
                  oos_only=not args.all)
    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
    else:
        _print_report(rep)
    # Exit code mirrors the verdict ladder: 0 if the scorer is gate-useful
    # on this slice, 1 if NO_SKILL / INSUFFICIENT_DATA, 2 if INVERTED /
    # CONCENTRATED_LOW (the alarm shapes — the gate is sizing on absent
    # or anti-predictive signal in the bucket that matters).
    verdict = rep.get("verdict") or "INSUFFICIENT_DATA"
    if verdict in ("INVERTED", "CONCENTRATED_LOW"):
        return 2
    if verdict in ("NO_SKILL", "INSUFFICIENT_DATA"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())

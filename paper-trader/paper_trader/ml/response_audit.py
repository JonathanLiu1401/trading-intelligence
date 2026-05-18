"""Scorer response-shape / monotonicity diagnostic — read-only.

The documented state of the DecisionScorer is *inert*: `skill_trend` =
near-zero OOS skill, `gate_audit` = GATE_INEFFECTIVE, `gate_pnl` =
≈+0.02pp economic contribution, `calibration --oos` = MISCALIBRATED. Every
one of those is a **statistical summary** (RMSE, rank-IC, decile error,
realized-pp). A quant deciding whether a near-zero-skill gate is *safe to
keep running* (vs. a bug masquerading as bad luck) wants the **geometric**
picture none of them give: *as each input feature moves across its real
range, which way — and how hard — does the model bend its predicted
return?*

`feature_importance` answers "how much skill is lost if I scramble feature
X" (permutation importance). It can report a feature is **material** while
the model bends it the economically *backwards* way (momentum-on-RSI,
say) — permutation importance is sign-blind. Only a response curve reveals
that. This module is the missing complement: importance vs. response-shape.

Method — **ICE-then-average** (Individual Conditional Expectation):
sweeping one feature while pinning the others at their medians fabricates
feature combinations the training set never contained, so the curve would
partly measure the MLP head's known off-distribution extrapolation (the
−89→+32 LITE pathology `PRED_CLAMP_PCT` exists for) rather than learned
structure. Instead, for each record we hold that record's *real* other
features and only override the swept one across the slice's empirical
p5..p95 grid, predict through the **same** `scorer.predict` path the live
`_ml_decide` gate uses, then average the per-record curves. Every point is
an in-distribution-ish combination.

The curve is the **BUY-context** response (the gate only ever queries
`predict` for BUY candidates), so the per-feature economic-sign priors
below are coherent with the gate's actual usage despite the train-time
SELL `-forward_return_5d` flip.

Primary verdict is **sign-agnostic** (shape only) so it cannot be wrong
about economics:

| Verdict | Meaning |
|---------|---------|
| `INSUFFICIENT_DATA` | scorer untrained or < `MIN_RECORDS` OOS rows / no sweepable feature has ≥2 distinct grid points |
| `FLAT_NO_RESPONSE` | the averaged prediction moves < `FLAT_TOL_PCT` across **every** feature's full empirical range — the model is response-surface-inert (concrete confirmation of the documented near-zero-skill state, expressed geometrically) |
| `RESPONSIVE_MONOTONE` | the model moves materially AND a majority of materially-responsive features have a monotone curve (`|spearman| ≥ MONO_MIN`) — it has a coherent learned structure (whether that structure is *profitable* is calibration/skill_trend's job, not this tool's) |
| `RESPONSIVE_JAGGED` | the model moves materially but the responsive curves are mostly non-monotone — the overfit-noise signature |

The economic-sign match (does rsi bend down, does momentum bend up, …) is
reported **per feature and as an informational tally only — it never
drives the verdict** (the `gate_audit` arm-monotone honesty pattern: the
SELL flip makes the training target a BUY/SELL blend, so a "wrong" sign is
not provably a defect).

Same operational discipline as `paper_trader/ml/calibration.py`:
read-only, no train, no pickle / `build_features` / `N_FEATURES` /
trade-path touch — safe against the live unattended loop. Never raises on
bad input. Reuses `calibration._spearman` for the monotonicity statistic
(single source of truth — the `_oos_rank_metrics` / `feature_importance`
precedent; tie-awareness is load-bearing because the scorer clamps
off-distribution predictions to ±`PRED_CLAMP_PCT`) and
`validation.split_outcomes_temporal` for the trustworthy OOS slice (the
`gate_audit` / `skill_trend` / `feature_importance` default — so this
curve and every other OOS diagnostic describe the SAME holdout).

NOTE: CLI / `ml/` reader, NOT wired into
`run_continuous_backtests.py::main()` — zero deploy-stale impact, no loop
restart needed.

```bash
cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.response_audit
cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.response_audit --all
```
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Module-level thresholds so tests assert exact verdicts and a tuning
# change is one reviewable edit (PRED_CLAMP_PCT / calibration.SPEARMAN_MIN
# / feature_importance.MIN_RMSE_INCREASE convention).
MIN_RECORDS = 30          # ≥ calibration.MIN_PAIRS — need a stable curve
GRID_POINTS = 9           # sweep resolution across p5..p95
# A feature's averaged curve "moves materially" if its peak-to-trough
# response exceeds this many points of predicted 5-day return. Modest by
# design: it separates "the model bends at all for this input" from
# "blind to it". The FLAT_NO_RESPONSE verdict triggers when NOT ONE
# feature clears this across its whole empirical range.
FEATURE_RESPONSE_TOL = 0.50
# Below this max-over-all-features response the whole model is inert.
FLAT_TOL_PCT = FEATURE_RESPONSE_TOL
# |spearman(grid, curve)| at/above this is a monotone (coherent-shape)
# response. Mirrors calibration.SPEARMAN_GOOD's "real structure" feel.
MONO_MIN = 0.80

# (logical_name, predict_kwarg, record_key, slot_default). Numeric features
# only — a monotone response curve over the 7-way sector one-hot is
# meaningless (no ordering), so `sector` is deliberately excluded here
# (feature_importance handles it via joint ticker permutation instead).
# slot_default mirrors build_features' per-slot neutral default so the
# sweep grid is built in the feature's natural units.
SWEEP_FEATURES: list[tuple[str, str, str, float]] = [
    ("ml_score", "ml_score", "ml_score", 0.0),
    ("rsi", "rsi", "rsi", 50.0),
    ("macd", "macd", "macd", 0.0),
    ("mom5", "mom5", "mom5", 0.0),
    ("mom20", "mom20", "mom20", 0.0),
    ("regime_mult", "regime_mult", "regime_mult", 1.0),
    ("vol_ratio", "vol_ratio", "vol_ratio", 1.0),
    ("bb_position", "bb_pos", "bb_position", 0.0),
    ("news_urgency", "news_urgency", "news_urgency", 50.0),
    ("news_article_count", "news_article_count", "news_article_count", 1.0),
]

# Economically-expected slope sign in the BUY context (informational ONLY —
# never feeds the verdict). +1: higher feature → higher expected return.
# -1: mean-reversion. None: no defensible prior (ambiguous), excluded from
# the consistency tally.
EXPECTED_SIGN: dict[str, int | None] = {
    "ml_score": +1,        # stronger picked signal → higher expected return
    "rsi": -1,             # overbought → mean-reversion down
    "macd": +1,            # bullish MACD → continuation up
    "mom5": +1,            # short-trend continuation
    "mom20": +1,           # medium-trend continuation
    "regime_mult": +1,     # bull regime → higher returns
    "bb_position": -1,     # above upper band → mean-reversion down
    "vol_ratio": None,     # breakout vs capitulation — ambiguous
    "news_urgency": None,  # ambiguous (also ~98% degenerate on corpus)
    "news_article_count": None,
}


def _base_kwargs(r: dict) -> dict:
    """The 11-kwarg ``predict`` call for one record, constructed identically
    to ``feature_importance._kwargs`` / ``_oos_rank_metrics`` so this curve
    reads on the SAME scale as every other OOS diagnostic."""
    from paper_trader.ml.decision_scorer import _to_float
    return dict(
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


def _grid(records: list[dict], record_key: str, slot_default: float) -> list[float]:
    """``GRID_POINTS`` evenly-spaced values across the slice's empirical
    p5..p95 for ``record_key``. Empirical range ⇒ in-distribution sweep
    (the advisor's ICE pitfall guard). Returns [] when < 2 distinct finite
    values exist (degenerate feature on this slice — e.g. the documented
    ~98%-NULL news features)."""
    from paper_trader.ml.decision_scorer import _to_float
    vals: list[float] = []
    for r in records:
        v = _to_float(r.get(record_key), slot_default)
        if np.isfinite(v):
            vals.append(float(v))
    if len(vals) < 2:
        return []
    arr = np.asarray(vals, dtype=np.float64)
    lo = float(np.percentile(arr, 5))
    hi = float(np.percentile(arr, 95))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= 1e-12:
        return []
    return [float(x) for x in np.linspace(lo, hi, GRID_POINTS)]


def _override(kw: dict, predict_kwarg: str, value: float) -> dict:
    """Copy of ``kw`` with one feature replaced by a swept value, applying
    the SAME normalization ``_base_kwargs`` applied to that slot (only the
    swept value changes, not the encoding)."""
    from paper_trader.ml.decision_scorer import _to_float
    out = dict(kw)
    if predict_kwarg == "ml_score":
        out["ml_score"] = _to_float(value, 0.0)
    elif predict_kwarg == "regime_mult":
        out["regime_mult"] = _to_float(value, 1.0)
    else:
        out[predict_kwarg] = value
    return out


def _curve(scorer, records: list[dict], predict_kwarg: str,
           grid: list[float]) -> list[float] | None:
    """ICE-then-average: for each grid value, predict every record with ONLY
    ``predict_kwarg`` overridden, average across records. Returns the
    length-``len(grid)`` averaged response, or None if nothing predicted."""
    bases = [_base_kwargs(r) for r in records]
    curve: list[float] = []
    for gv in grid:
        preds: list[float] = []
        for base in bases:
            try:
                p = float(scorer.predict(**_override(base, predict_kwarg, gv)))
            except Exception:
                continue
            if np.isfinite(p):
                preds.append(p)
        if not preds:
            return None
        curve.append(float(np.mean(preds)))
    return curve


def response_report(scorer, records: list[dict]) -> dict:
    """Build the per-feature averaged response curves + the sign-agnostic
    verdict. ``records`` is the ``decision_outcomes.jsonl`` row shape.

    Returns a JSON-safe dict:
    ``{status, verdict, n, n_train, max_response_range, n_responsive,
       n_monotone, n_sign_consistent, n_with_prior, features:[…], hint}``.
    Never raises — a fault degrades to INSUFFICIENT_DATA, never a crash."""
    from paper_trader.ml.calibration import _spearman

    out: dict = {
        "status": "error", "verdict": "INSUFFICIENT_DATA", "n": 0,
        "n_train": getattr(scorer, "n_train", None),
        "max_response_range": None, "n_responsive": 0, "n_monotone": 0,
        "n_sign_consistent": 0, "n_with_prior": 0, "features": [], "hint": "",
    }
    try:
        recs = [r for r in (records or []) if isinstance(r, dict)]
        if not getattr(scorer, "is_trained", False):
            out["hint"] = "scorer not trained — no response surface to audit"
            return out
        if len(recs) < MIN_RECORDS:
            out["hint"] = f"need ≥{MIN_RECORDS} records, have {len(recs)}"
            return out

        feats: list[dict] = []
        any_grid = False
        n_curves = 0  # features for which predict produced a usable curve
        for name, kwarg, key, dflt in SWEEP_FEATURES:
            grid = _grid(recs, key, dflt)
            if not grid:
                feats.append({
                    "feature": name, "degenerate": True,
                    "response_range": None, "spearman": None,
                    "responsive": False, "monotone": False,
                    "expected_sign": EXPECTED_SIGN.get(name),
                    "sign_consistent": None, "grid_lo": None,
                    "grid_hi": None, "curve": [],
                })
                continue
            any_grid = True
            curve = _curve(scorer, recs, kwarg, grid)
            if curve is None:
                feats.append({
                    "feature": name, "degenerate": True,
                    "response_range": None, "spearman": None,
                    "responsive": False, "monotone": False,
                    "expected_sign": EXPECTED_SIGN.get(name),
                    "sign_consistent": None, "grid_lo": round(grid[0], 4),
                    "grid_hi": round(grid[-1], 4), "curve": [],
                })
                continue
            n_curves += 1
            rng = float(max(curve) - min(curve))
            sp = _spearman(np.asarray(grid, dtype=np.float64),
                           np.asarray(curve, dtype=np.float64))
            responsive = rng >= FEATURE_RESPONSE_TOL
            monotone = responsive and abs(sp) >= MONO_MIN
            exp = EXPECTED_SIGN.get(name)
            # Observed slope sign from the monotone curve; only meaningful
            # when the feature actually moves the model.
            obs_sign = 0
            if responsive and abs(sp) >= 1e-9:
                obs_sign = 1 if sp > 0 else -1
            sign_consistent: bool | None = None
            if exp is not None and obs_sign != 0:
                sign_consistent = (obs_sign == exp)
            feats.append({
                "feature": name, "degenerate": False,
                "response_range": round(rng, 4),
                "spearman": round(float(sp), 4),
                "responsive": responsive, "monotone": monotone,
                "expected_sign": exp, "observed_sign": obs_sign,
                "sign_consistent": sign_consistent,
                "grid_lo": round(grid[0], 4), "grid_hi": round(grid[-1], 4),
                "curve": [round(c, 4) for c in curve],
            })

        if not any_grid:
            out["features"] = feats
            out["hint"] = ("no sweepable feature has ≥2 distinct values on "
                           "this slice (corpus is degenerate)")
            return out
        if n_curves == 0:
            # Grids exist but ``scorer.predict`` produced nothing usable for
            # ANY feature (a broken / always-raising / all-NaN model). That
            # is "cannot audit", NOT "model is flat" — surface it honestly
            # rather than emitting a misleading FLAT_NO_RESPONSE.
            out["features"] = feats
            out["n"] = len(recs)
            out["hint"] = ("scorer.predict produced no finite output on any "
                           "feature sweep — model unusable, cannot audit")
            return out

        ranges = [f["response_range"] for f in feats
                  if f["response_range"] is not None]
        max_rng = max(ranges) if ranges else 0.0
        responsive = [f for f in feats if f["responsive"]]
        monotone = [f for f in responsive if f["monotone"]]
        sign_priored = [f for f in responsive
                        if f["sign_consistent"] is not None]
        sign_ok = [f for f in sign_priored if f["sign_consistent"]]

        out.update({
            "status": "ok",
            "n": len(recs),
            "max_response_range": round(float(max_rng), 4),
            "n_responsive": len(responsive),
            "n_monotone": len(monotone),
            "n_sign_consistent": len(sign_ok),
            "n_with_prior": len(sign_priored),
            "features": feats,
        })

        if max_rng < FLAT_TOL_PCT:
            out["verdict"] = "FLAT_NO_RESPONSE"
            out["hint"] = (
                f"the averaged prediction moves < {FLAT_TOL_PCT:.2f}pp across "
                f"EVERY feature's full empirical range (max {max_rng:.3f}pp) — "
                f"the model is response-surface-inert; this is the documented "
                f"near-zero-skill state seen geometrically, not new alpha")
        elif responsive and len(monotone) * 2 >= len(responsive):
            out["verdict"] = "RESPONSIVE_MONOTONE"
            out["hint"] = (
                f"{len(monotone)}/{len(responsive)} materially-responsive "
                f"features are monotone — the model has a coherent learned "
                f"shape (whether it is PROFITABLE is calibration/skill_trend's "
                f"question, not this tool's); economic-sign-consistent "
                f"{len(sign_ok)}/{len(sign_priored)} (informational only)")
        else:
            out["verdict"] = "RESPONSIVE_JAGGED"
            out["hint"] = (
                f"the model moves materially (max {max_rng:.2f}pp) but only "
                f"{len(monotone)}/{len(responsive)} responsive curves are "
                f"monotone — the overfit-noise signature; do not read the "
                f"point estimates as a learned economic relationship")
        return out
    except Exception as e:  # never raise near the unattended loop
        out["hint"] = f"response-audit failed: {type(e).__name__}: {e}"
        return out


def analyze(outcomes_path: Path | str, oos_only: bool = True) -> dict:
    """Load the live pickled scorer + outcomes tail and return the report.
    Read-only; never raises. ``oos_only`` restricts to the temporal holdout
    via ``validation.split_outcomes_temporal`` (the SSOT split every other
    OOS diagnostic uses) so this curve and the ledger describe one slice."""
    from .decision_scorer import DecisionScorer

    out: dict = {"status": "error", "verdict": "INSUFFICIENT_DATA", "n": 0,
                 "features": [], "hint": "", "slice": None}
    try:
        p = Path(outcomes_path)
        if not p.exists():
            out["hint"] = f"no outcomes file at {p}"
            return out
        records: list[dict] = []
        for ln in p.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                if isinstance(obj, dict):
                    records.append(obj)
            except Exception:
                continue
        if oos_only:
            try:
                from paper_trader.validation import split_outcomes_temporal
                _train, records = split_outcomes_temporal(
                    records, oos_fraction=0.2)
                slice_tag = "oos"
            except Exception:
                slice_tag = "all(split-failed)"
        else:
            slice_tag = "all"
        scorer = DecisionScorer()
        rep = response_report(scorer, records)
        rep["slice"] = slice_tag
        rep.setdefault("n_train", getattr(scorer, "n_train", None))
        return rep
    except Exception as e:
        out["hint"] = f"response-audit failed: {type(e).__name__}: {e}"
        return out


def _cli() -> int:
    """`python3 -m paper_trader.ml.response_audit [--all]` — averaged
    response-curve audit of the live pickled scorer. Read-only.

    Default: temporal-OOS slice (the trustworthy view). `--all`: full
    accumulated tail (in-sample). Exit 2 on FLAT_NO_RESPONSE /
    RESPONSIVE_JAGGED (the "no coherent learned structure" verdicts —
    operator/cron branchable, like feature_importance's exit-2 on
    FLAT / SECTOR_DOMINATED); 1 if the scorer is untrained / no file;
    0 otherwise."""
    import sys
    args = sys.argv[1:]
    oos_only = "--all" not in args
    root = Path(__file__).resolve().parent.parent.parent
    rep = analyze(root / "data" / "decision_outcomes.jsonl",
                  oos_only=oos_only)
    print(f"VERDICT: {rep['verdict']}  ({rep.get('hint', '')})")
    print(f"  slice={rep.get('slice')}  n={rep.get('n')}  "
          f"n_train={rep.get('n_train')}  "
          f"max_response_range={rep.get('max_response_range')}pp")
    print(f"  responsive={rep.get('n_responsive')}  "
          f"monotone={rep.get('n_monotone')}  "
          f"sign_consistent={rep.get('n_sign_consistent')}"
          f"/{rep.get('n_with_prior')} (informational)")
    for f in rep.get("features", []):
        if f.get("degenerate"):
            print(f"  ∅ {f['feature']:<20} degenerate (no spread on slice)")
            continue
        flag = "▲" if f["monotone"] else ("~" if f["responsive"] else " ")
        exp = f.get("expected_sign")
        exp_s = "+" if exp == 1 else ("-" if exp == -1 else "·")
        sc = f.get("sign_consistent")
        sc_s = "✓" if sc is True else ("✗" if sc is False else " ")
        print(f"  {flag} {f['feature']:<20} "
              f"range={f['response_range']:+7.3f}pp  "
              f"spearman={f['spearman']:+.3f}  "
              f"exp={exp_s} obs={f.get('observed_sign', 0):+d} {sc_s}  "
              f"[{f['grid_lo']:+.2f}→{f['grid_hi']:+.2f}]")
    if rep.get("status") == "error" and "not trained" in rep.get("hint", ""):
        return 1
    if rep.get("status") == "error" and "no outcomes file" in rep.get("hint", ""):
        return 1
    return 2 if rep.get("verdict") in ("FLAT_NO_RESPONSE",
                                       "RESPONSIVE_JAGGED") else 0


if __name__ == "__main__":
    raise SystemExit(_cli())

"""Scorer off-distribution / clamp / failure rate audit.

Read-only research signal (2026-05-30 feature, Agent 2 ML+backtests).
Sibling to ``oos_parity_audit`` / ``gate_realized`` — mirrors their
operational discipline exactly (never trains, never touches
``decision_scorer.pkl`` / ``decision_outcomes.jsonl`` / ``build_features``
/ any trade path; safe to run against the live unattended loop; never
raises in analyzer functions).

**The gap this closes.** ``predict_with_meta`` exposes three trust flags
for every prediction — ``failed`` (predict raised / non-finite raw),
``clamped`` (raw exceeded ±``PRED_CLAMP_PCT``), and ``off_distribution``
(same condition, used by the live gate to ABSTAIN — see the
``_ml_decide`` ``not scorer_off_dist`` guard). The live conviction gate
SKIPS arm modulation entirely whenever ``off_distribution=True``, so
the gate's economic effect depends critically on the off-distribution
rate against the realized OOS slice. Yet no existing diagnostic
reports that rate:

  * ``oos_parity_audit`` measures bias between gate-aligned and biased
    prediction paths — orthogonal to "how often is the prediction
    even trustworthy?"
  * ``gate_realized`` / ``gate_arm_kelly`` filter OOD rows out (into
    the ``abstained`` bucket); they don't expose what fraction.
  * ``feature_importance`` / ``feature_alignment`` describe what the
    model relies on — not how often its output is usable.

The textbook quant question — *what fraction of OOS predictions does
the deployed scorer trust, and how extreme are the rest?* — is invisible
today. A scorer that abstains on 80% of OOS rows is structurally
half-functional even when its in-distribution rank-IC is healthy: the
gate's per-arm sizing only fires on the 20% in-distribution tail, so
the realized gate effect (``gate_pnl``) is computed on a shrinking
slice. This analyzer surfaces that rate directly.

**Metrics reported**:

  * ``n``                 — total OOS rows where a prediction was attempted.
  * ``n_failed``          — predict raised / non-finite raw (``failed=True``).
  * ``n_clamped``         — raw exceeded ±``PRED_CLAMP_PCT`` (``clamped=True``;
                             includes the failed-path's True-clamped value
                             so an honest "saw a clamp" count is reported,
                             but the failed-rate is separated below).
  * ``n_off_distribution`` — ``off_distribution=True`` (the flag the live
                             gate uses to abstain). Equal to
                             ``n_clamped`` by construction in the current
                             ``predict_with_meta`` implementation, but
                             surfaced separately so a future code change
                             that decouples the two is observable here.
  * ``n_in_distribution`` — neither failed nor off-distribution.
  * ``failed_rate``       — ``n_failed / n``.
  * ``clamped_rate``      — ``n_clamped / n``.
  * ``off_dist_rate``     — ``n_off_distribution / n``.
  * ``in_dist_rate``      — ``n_in_distribution / n``.
  * ``raw_min`` / ``raw_max`` / ``raw_p5`` / ``raw_p95`` — empirical
    distribution of the RAW (pre-clamp) prediction; lets a reading quant
    see how far the model extrapolates past the empirical label support.
  * ``raw_mean``          — mean of raw predictions (informational —
                             biases toward extrapolated extremes when
                             many predictions land outside ±50).

**Verdict ladder** (driven by the off-distribution rate — the
gate-relevant quantity):

| Verdict | Meaning |
|---------|---------|
| ``INSUFFICIENT_DATA`` | < ``MIN_TOTAL`` OOS rows attempted, or scorer untrained / pickle absent. |
| ``GATE_DARK`` | ``off_dist_rate > GATE_DARK_THRESHOLD`` (>50% default) — the gate abstains on more than half of OOS predictions; arm-level analytics describe a shrunk and unrepresentative slice. |
| ``SEVERE_OOD`` | ``off_dist_rate > SEVERE_THRESHOLD`` (>25%) — substantial extrapolation; the deployed model is operating well outside its training support. |
| ``MILD_OOD`` | ``off_dist_rate > MILD_THRESHOLD`` (>5%) — occasional extrapolation; the model is mostly trusted but some inputs land outside support. |
| ``HEALTHY`` | ``off_dist_rate ≤ MILD_THRESHOLD`` — the scorer is essentially never extrapolating; predictions are trustworthy in magnitude. |

Independently, a non-zero ``failed_rate`` always carries a separate
``has_failures=True`` flag in the report — a `failed` prediction means
the predict CALL itself broke (shape mismatch, exception), which is a
distinct class of fault from clamping. A reading quant should treat
ANY failure rate > 0 as a code/data shape investigation cue, even if
the headline verdict is HEALTHY.

**Pure / module-level constants for testability**. CLI:

```bash
cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.scorer_offdist_rate
cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.scorer_offdist_rate --all
cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.scorer_offdist_rate --json
```

The default OOS-only slice mirrors every sibling analyzer — uses
``validation.split_outcomes_temporal`` 20% tail so the metric describes
held-out data the deployed scorer has not seen. ``--all`` runs the
analyzer on the full corpus.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


# Module-level constants (testable, single point of tuning).
MIN_TOTAL = 30                  # min OOS rows before any non-INSUFFICIENT verdict
MILD_THRESHOLD = 0.05           # >5% OOD → MILD_OOD
SEVERE_THRESHOLD = 0.25         # >25% OOD → SEVERE_OOD
GATE_DARK_THRESHOLD = 0.50      # >50% OOD → GATE_DARK


def _maybe_float(v):
    """Coerce to finite float or None — mirrors gate_arm_kelly._f."""
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def offdist_rate_report(scorer, rows) -> dict:
    """Compute off-distribution / clamp / failure rates on `rows`.

    `scorer` MUST expose ``predict_with_meta`` (the contract). A scorer
    that only carries the scalar ``predict`` (test fakes / the
    init-failure ``_Dummy`` stub) returns an ``INSUFFICIENT_DATA``
    verdict with a clear ``no_predict_with_meta`` hint — the flags this
    analyzer reports literally don't exist on that interface.

    `rows` is any iterable of decision_outcomes-style dicts. Rows that
    don't carry a usable feature vector (no ticker, or every input
    coerces to None and trips ``failed`` inside predict_with_meta) are
    counted in the per-row totals — the analyzer's purpose is precisely
    to surface that rate.

    Returns a JSON-safe dict. Never raises (the analyzer must not break
    the unattended loop).
    """
    if not getattr(scorer, "is_trained", False):
        return {
            "verdict": "INSUFFICIENT_DATA",
            "n": 0, "n_failed": 0, "n_clamped": 0,
            "n_off_distribution": 0, "n_in_distribution": 0,
            "failed_rate": None, "clamped_rate": None,
            "off_dist_rate": None, "in_dist_rate": None,
            "raw_min": None, "raw_max": None, "raw_mean": None,
            "raw_p5": None, "raw_p95": None,
            "has_failures": False,
            "hint": "scorer not trained — no predictions to evaluate.",
        }
    _pwm = getattr(scorer, "predict_with_meta", None)
    if not callable(_pwm):
        return {
            "verdict": "INSUFFICIENT_DATA",
            "n": 0, "n_failed": 0, "n_clamped": 0,
            "n_off_distribution": 0, "n_in_distribution": 0,
            "failed_rate": None, "clamped_rate": None,
            "off_dist_rate": None, "in_dist_rate": None,
            "raw_min": None, "raw_max": None, "raw_mean": None,
            "raw_p5": None, "raw_p95": None,
            "has_failures": False,
            "hint": "scorer has no predict_with_meta — analyzer needs the "
                    "11-flag meta envelope to classify each prediction.",
        }

    n = 0
    n_failed = 0
    n_clamped = 0
    n_off_distribution = 0
    n_in_distribution = 0
    raws: list[float] = []

    try:
        it = list(rows or [])
    except Exception:
        it = []

    for r in it:
        if not isinstance(r, dict):
            continue
        try:
            meta = _pwm(
                ml_score=_maybe_float(r.get("ml_score")) or 0.0,
                rsi=r.get("rsi"),
                macd=r.get("macd"),
                mom5=r.get("mom5"),
                mom20=r.get("mom20"),
                regime_mult=_maybe_float(r.get("regime_mult")) or 1.0,
                ticker=str(r.get("ticker") or ""),
                vol_ratio=r.get("vol_ratio"),
                bb_pos=r.get("bb_position"),
                news_urgency=r.get("news_urgency"),
                news_article_count=r.get("news_article_count"),
                ema200_above=r.get("ema200_above"),
                hist_cross_up=r.get("hist_cross_up"),
                macd_below_zero_cross=r.get("macd_below_zero_cross"),
            )
        except Exception:
            # A predict_with_meta that itself raises (should never happen
            # — the contract is "never raises") counts as failed AND
            # ineligible for raw-distribution stats (no raw to capture).
            n += 1
            n_failed += 1
            continue
        if not isinstance(meta, dict):
            n += 1
            n_failed += 1
            continue
        n += 1
        # `failed=True` envelopes also carry `off_distribution=True` and
        # `clamped=True` (predict_with_meta's documented contract — the
        # failed path is the maximally untrustworthy result), so a single
        # row contributes to MULTIPLE counters by design. Counting
        # `in_distribution` directly (NEITHER failed NOR off-distribution)
        # is the only honest way to handle the overlap — anything derived
        # from `n - n_failed - n_off_distribution` double-counts the
        # failed-AND-off-dist overlap row.
        failed_f = bool(meta.get("failed"))
        off_dist_f = bool(meta.get("off_distribution"))
        if failed_f:
            n_failed += 1
        if meta.get("clamped"):
            n_clamped += 1
        if off_dist_f:
            n_off_distribution += 1
        if not failed_f and not off_dist_f:
            n_in_distribution += 1
        raw = _maybe_float(meta.get("raw"))
        if raw is not None:
            raws.append(raw)

    def _rate(num):
        return round(num / n, 4) if n > 0 else None

    raw_arr = np.asarray(raws, dtype=np.float64) if raws else None
    if raw_arr is not None and raw_arr.size >= 1:
        raw_min = round(float(raw_arr.min()), 4)
        raw_max = round(float(raw_arr.max()), 4)
        raw_mean = round(float(raw_arr.mean()), 4)
        # Percentiles need at least 2 points to differ meaningfully;
        # numpy handles n=1 by returning the single value at any q.
        raw_p5 = round(float(np.percentile(raw_arr, 5)), 4)
        raw_p95 = round(float(np.percentile(raw_arr, 95)), 4)
    else:
        raw_min = raw_max = raw_mean = raw_p5 = raw_p95 = None

    off_dist_rate = _rate(n_off_distribution)
    failed_rate = _rate(n_failed)

    if n < MIN_TOTAL:
        verdict = "INSUFFICIENT_DATA"
        hint = (f"only {n} OOS rows attempted; need ≥{MIN_TOTAL} for a "
                "trustworthy rate.")
    elif off_dist_rate is None:
        verdict = "INSUFFICIENT_DATA"
        hint = "off_dist_rate undefined (n=0)."
    elif off_dist_rate > GATE_DARK_THRESHOLD:
        verdict = "GATE_DARK"
        hint = (f"{n_off_distribution}/{n} ({off_dist_rate*100:.1f}%) OOS "
                "predictions are off-distribution. The live gate "
                "abstains on each — arm-level analytics describe a "
                "shrunk, unrepresentative slice. Investigate scaler / "
                "label distribution / training-input range.")
    elif off_dist_rate > SEVERE_THRESHOLD:
        verdict = "SEVERE_OOD"
        hint = (f"{n_off_distribution}/{n} ({off_dist_rate*100:.1f}%) OOS "
                "predictions are off-distribution. Substantial "
                "extrapolation; the deployed model is operating well "
                "outside its training support.")
    elif off_dist_rate > MILD_THRESHOLD:
        verdict = "MILD_OOD"
        hint = (f"{n_off_distribution}/{n} ({off_dist_rate*100:.1f}%) OOS "
                "predictions are off-distribution. Mostly trusted but "
                "watch for further drift.")
    else:
        verdict = "HEALTHY"
        hint = (f"{n_off_distribution}/{n} ({off_dist_rate*100:.1f}%) OOS "
                "predictions off-distribution — well within tolerance.")

    return {
        "verdict": verdict,
        "n": n,
        "n_failed": n_failed,
        "n_clamped": n_clamped,
        "n_off_distribution": n_off_distribution,
        "n_in_distribution": n_in_distribution,
        "failed_rate": failed_rate,
        "clamped_rate": _rate(n_clamped),
        "off_dist_rate": off_dist_rate,
        "in_dist_rate": _rate(n_in_distribution),
        "raw_min": raw_min,
        "raw_max": raw_max,
        "raw_mean": raw_mean,
        "raw_p5": raw_p5,
        "raw_p95": raw_p95,
        "has_failures": n_failed > 0,
        "min_total": MIN_TOTAL,
        "mild_threshold": MILD_THRESHOLD,
        "severe_threshold": SEVERE_THRESHOLD,
        "gate_dark_threshold": GATE_DARK_THRESHOLD,
        "hint": hint,
    }


def analyze(outcomes_path: Path | str | None = None,
            oos_only: bool = True) -> dict:
    """Read outcomes JSONL and run the off-distribution rate report.

    Mirrors every sibling analyzer's analyze() semantics:
      * ``oos_only=True`` (default) applies validation temporal 20% tail.
      * ``oos_only=False`` runs on the full corpus.

    Returns ``{status, ...report fields}``. ``status='error'`` on read /
    parse failure; ``status='ok'`` otherwise. Never raises.
    """
    if outcomes_path is None:
        outcomes_path = (Path(__file__).resolve().parent.parent.parent
                         / "data" / "decision_outcomes.jsonl")
    p = Path(outcomes_path)
    if not p.exists():
        out = offdist_rate_report(_DummyUntrained(), [])
        out["status"] = "error"
        out["error"] = f"missing {p}"
        return out
    rows: list[dict] = []
    try:
        with p.open("r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception as exc:
        out = offdist_rate_report(_DummyUntrained(), [])
        out["status"] = "error"
        out["error"] = f"read failed: {exc}"
        return out

    if oos_only:
        try:
            from paper_trader.validation import split_outcomes_temporal
            _, rows = split_outcomes_temporal(rows, oos_fraction=0.2)
        except Exception as exc:
            print(f"[scorer_offdist_rate] temporal split unavailable "
                  f"({exc}) — reporting on full corpus")

    try:
        from paper_trader.ml.decision_scorer import DecisionScorer
        scorer = DecisionScorer()
    except Exception as exc:
        out = offdist_rate_report(_DummyUntrained(), [])
        out["status"] = "error"
        out["error"] = f"scorer load failed: {type(exc).__name__}: {exc}"
        return out

    report = offdist_rate_report(scorer, rows)
    report["status"] = "ok"
    report["oos_only"] = bool(oos_only)
    return report


class _DummyUntrained:
    """Stand-in for the deployed scorer when the pickle is unavailable.
    Mirrors the contract just enough that ``offdist_rate_report`` returns
    a clean INSUFFICIENT_DATA envelope rather than crashing."""
    is_trained = False

    def predict_with_meta(self, **kw):
        return {"pred": 0.0, "raw": 0.0, "clamped": False,
                "off_distribution": False, "failed": True}


def _cli(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.scorer_offdist_rate",
        description="Scorer off-distribution / clamp / failure rate "
                    "audit on the OOS slice.",
    )
    parser.add_argument("--all", action="store_true",
                        help="Run on the full corpus (default: OOS 20%% tail).")
    parser.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON.")
    parser.add_argument("--outcomes", type=Path, default=None,
                        help="Path to the outcomes JSONL (default: "
                             "data/decision_outcomes.jsonl relative to repo).")
    args = parser.parse_args(argv)

    report = analyze(args.outcomes, oos_only=not args.all)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report.get("status") == "ok" else 1

    print(f"[scorer_offdist_rate] verdict={report.get('verdict')}  "
          f"n={report.get('n')}  oos_only={report.get('oos_only')}")
    print(f"  failed={report.get('n_failed')} "
          f"clamped={report.get('n_clamped')} "
          f"off_distribution={report.get('n_off_distribution')} "
          f"in_distribution={report.get('n_in_distribution')}")
    fr = report.get("failed_rate")
    cr = report.get("clamped_rate")
    odr = report.get("off_dist_rate")
    idr = report.get("in_dist_rate")
    print(f"  rates: failed={fr*100:.1f}%  clamped={cr*100:.1f}%  "
          f"off_dist={odr*100:.1f}%  in_dist={idr*100:.1f}%"
          if (fr is not None and cr is not None and odr is not None
              and idr is not None) else
          "  rates: n/a (insufficient_data)")
    if report.get("raw_min") is not None:
        print(f"  raw pred dist: min={report['raw_min']:+.2f} "
              f"p5={report['raw_p5']:+.2f} mean={report['raw_mean']:+.2f} "
              f"p95={report['raw_p95']:+.2f} max={report['raw_max']:+.2f}")
    if report.get("has_failures"):
        print(f"  ⚠ has_failures=True ({report['n_failed']} rows could not "
              "be predicted — investigate predict-with-meta exception path)")
    print(f"  hint: {report.get('hint')}")
    return 0 if report.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())

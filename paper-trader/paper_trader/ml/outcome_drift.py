"""Decision-outcome distribution drift report.

The DecisionScorer is retrained every cycle on the last
``MAX_OUTCOMES_FOR_TRAINING`` (5000) records of
``data/decision_outcomes.jsonl``. With ~9k accumulated rows the trainer
already drops the older half, but a more insidious failure mode is
**concept drift** *inside* the trainer's tail: when the recent 25%
of outcomes describes a regime materially different from the older 75%,
the scorer learns a target whose label dynamics have already shifted.
``regime_audit`` answers "skill conditional on bull/bear/sideways", but
nothing in the existing surface (skill_trend, baseline_compare,
calibration, feature_coverage, feature_importance) answers the prior
question: *has the input/output distribution itself shifted across the
training tail?*

Concretely, for every numeric feature this module reports the **drift
score** ``(mean_recent − mean_older) / σ_older`` — a z-style shift of
recent population mean off the older one, expressed in older-population
σ units. The same statistic applies to ``forward_return_5d`` so a
regime that drifted from net-positive to net-negative realized returns
is flagged independently from feature drift.

State ladder (sample-size honesty mirrors ``tail_risk`` /
``build_correlation``):

- ``NO_DATA``        — no records / file missing
- ``INSUFFICIENT``   — fewer than ``MIN_PER_BUCKET`` rows in either bucket
- ``STABLE``         — every feature within ``DRIFT_MILD`` σ of older mean
- ``MILD_DRIFT``     — at least one feature in [``DRIFT_MILD``, ``DRIFT_SEVERE``] σ
- ``SEVERE_DRIFT``   — at least one feature beyond ``DRIFT_SEVERE`` σ

Sort: ``|drift_score|`` descending. Pure / no I/O / never raises (the
``_safe`` contract — None inputs, missing keys, non-numeric values
all degrade to a NO_DATA row rather than an exception). The CLI is
``python3 -m paper_trader.ml.outcome_drift``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

# 17 base feature names mirrored from `build_features()` order, plus the
# label (`forward_return_5d`) and the macro `regime_mult` slot. Kept
# tight: only the features the scorer actually trains on AND the
# realized target — not every additive enrichment field that may or
# may not be present on older rows.
TRACKED_FEATURES = [
    "ml_score", "rsi", "macd", "mom5", "mom20", "regime_mult",
    "vol_ratio", "bb_position", "news_urgency", "news_article_count",
    "forward_return_5d",
]

# Bucket sizing — recent share of the temporal tail. 0.25 (last 25%)
# balances responsiveness vs. statistical power on the 5000-record
# trainer tail (the documented `MAX_OUTCOMES_FOR_TRAINING`): 1250 recent
# vs 3750 older keeps both well above any per-feature population guard.
RECENT_FRACTION = 0.25
MIN_PER_BUCKET = 20

# Drift bucket thresholds. 0.5σ is a conservative "noticeable shift" line
# (~38% of a normal's mass within a ½σ band of the mean); 1.0σ marks a
# material regime change. Bands chosen to keep STABLE the common case
# under a market that's drifting gradually and only escalate when the
# shift is loud enough that a calibration audit is warranted.
DRIFT_MILD = 0.5
DRIFT_SEVERE = 1.0


def _safe_float(v) -> float | None:
    """Coerce ``v`` to a finite float or ``None``.

    Mirrors the ``decision_scorer._to_float`` discipline: rejects
    NaN/inf, np.bool_, and non-numeric types so a single garbage row
    can't poison the population statistics. Returns ``None`` rather
    than a fallback default — drift requires *present* numeric values;
    a missing one must be dropped, not imputed (imputing to a neutral
    value silently shrinks the apparent drift on a feature whose
    coverage *itself* is the drifting signal).
    """
    if isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _population_stats(values: list[float]) -> tuple[float, float, int]:
    """Population mean, std, n. Used over a sample stat because the
    "older" bucket is the model's view of training history, not a
    sample inferring some larger super-population — we describe it
    exactly. ``n=0 → (0.0, 0.0, 0)``, ``n=1 → (mean, 0.0, 1)``.
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0, 1
    var = sum((v - mean) ** 2 for v in values) / n  # population variance
    return mean, math.sqrt(var), n


def _drift_score(recent: list[float], older: list[float]) -> dict:
    """Per-feature drift summary.

    Returns ``{n_recent, n_older, mean_recent, mean_older, std_older,
    drift_score, status}`` where ``drift_score = (μ_recent − μ_older)
    / σ_older``. When ``σ_older == 0`` (all older values identical) the
    score is sentinel-coded: 0.0 if recent mean also equals it (no
    drift), else ±inf is reported as a SEVERE flag with magnitude=∞ so
    a constant feature that suddenly varies in the recent bucket is
    not silently treated as STABLE.

    ``status`` is one of ``OK`` / ``INSUFFICIENT`` (sample-size honesty —
    can't characterize drift below ``MIN_PER_BUCKET`` rows in either
    bucket).
    """
    n_recent = len(recent)
    n_older = len(older)
    if n_recent < MIN_PER_BUCKET or n_older < MIN_PER_BUCKET:
        return {
            "n_recent": n_recent, "n_older": n_older,
            "mean_recent": None, "mean_older": None,
            "std_older": None, "drift_score": None,
            "status": "INSUFFICIENT",
        }
    mean_r, _, _ = _population_stats(recent)
    mean_o, std_o, _ = _population_stats(older)
    if std_o == 0.0:
        if mean_r == mean_o:
            drift = 0.0
        else:
            # The older bucket is constant; any recent variation is
            # unbounded in σ_older units. Treat as severe by convention
            # (a constant feature that suddenly drifts IS a regime change).
            drift = math.inf if mean_r > mean_o else -math.inf
    else:
        drift = (mean_r - mean_o) / std_o
    return {
        "n_recent": n_recent, "n_older": n_older,
        "mean_recent": round(mean_r, 6),
        "mean_older": round(mean_o, 6),
        "std_older": round(std_o, 6),
        "drift_score": (drift if math.isinf(drift) else round(drift, 4)),
        "status": "OK",
    }


def _classify_feature(drift_score) -> str:
    """``STABLE`` / ``MILD_DRIFT`` / ``SEVERE_DRIFT`` per the thresholds
    above. ``None`` (the INSUFFICIENT case) classifies as ``UNKNOWN`` so
    the overall verdict can ignore it cleanly."""
    if drift_score is None:
        return "UNKNOWN"
    if math.isinf(drift_score):
        return "SEVERE_DRIFT"
    a = abs(drift_score)
    if a < DRIFT_MILD:
        return "STABLE"
    if a < DRIFT_SEVERE:
        return "MILD_DRIFT"
    return "SEVERE_DRIFT"


def build_outcome_drift(
    records: list[dict] | None,
    recent_fraction: float = RECENT_FRACTION,
) -> dict:
    """Pure builder — no I/O. Returns the full drift report dict.

    ``records`` should already be ordered (the trainer's input is
    append-only by sim_date in practice). When ``sim_date`` is present
    we sort defensively so a caller passing an unordered list still
    gets a chronological split; a missing/garbage ``sim_date`` sorts
    last (treated as "we don't know when this row landed").
    """
    if not records:
        return {
            "state": "NO_DATA",
            "n_total": 0,
            "n_recent": 0, "n_older": 0,
            "recent_fraction": recent_fraction,
            "verdict": "UNKNOWN",
            "headline": "no_outcome_records",
            "features": [],
        }
    try:
        rec_frac = float(recent_fraction)
    except (TypeError, ValueError):
        rec_frac = RECENT_FRACTION
    rec_frac = max(0.05, min(0.5, rec_frac))

    # Sort by sim_date defensively. A row with no sim_date sorts last —
    # we DO NOT silently drop it (the trainer would have trained on it
    # too), so its values still feed the population stats; only its
    # bucket placement is deterministic-suffix.
    def _sort_key(r):
        sd = r.get("sim_date") if isinstance(r, dict) else None
        # tuple: (1 if sortable else 0, value) so missing dates land last
        if isinstance(sd, str) and sd:
            return (0, sd)
        return (1, "")

    rows = sorted(
        (r for r in records if isinstance(r, dict)),
        key=_sort_key,
    )
    n_total = len(rows)
    if n_total == 0:
        return {
            "state": "NO_DATA",
            "n_total": 0,
            "n_recent": 0, "n_older": 0,
            "recent_fraction": rec_frac,
            "verdict": "UNKNOWN",
            "headline": "no_outcome_records",
            "features": [],
        }
    n_recent_target = max(1, int(round(n_total * rec_frac)))
    if n_recent_target >= n_total:
        n_recent_target = n_total - 1 if n_total >= 2 else n_total
    n_older_target = n_total - n_recent_target

    older_rows = rows[:n_older_target]
    recent_rows = rows[n_older_target:]

    if (len(older_rows) < MIN_PER_BUCKET or
            len(recent_rows) < MIN_PER_BUCKET):
        return {
            "state": "INSUFFICIENT",
            "n_total": n_total,
            "n_recent": len(recent_rows), "n_older": len(older_rows),
            "recent_fraction": rec_frac,
            "verdict": "UNKNOWN",
            "headline": (f"insufficient sample — need >= "
                         f"{MIN_PER_BUCKET} rows per bucket"),
            "features": [],
        }

    features: list[dict] = []
    overall_status = "STABLE"
    worst_feat = None
    worst_abs = -1.0
    for feat in TRACKED_FEATURES:
        recent_vals = [
            v for v in (_safe_float(r.get(feat)) for r in recent_rows)
            if v is not None
        ]
        older_vals = [
            v for v in (_safe_float(r.get(feat)) for r in older_rows)
            if v is not None
        ]
        summary = _drift_score(recent_vals, older_vals)
        ds = summary.get("drift_score")
        cls = _classify_feature(ds)
        summary["feature"] = feat
        summary["classification"] = cls
        features.append(summary)
        if cls == "SEVERE_DRIFT":
            overall_status = "SEVERE_DRIFT"
        elif cls == "MILD_DRIFT" and overall_status == "STABLE":
            overall_status = "MILD_DRIFT"
        if ds is not None and not math.isinf(ds):
            a = abs(ds)
            if a > worst_abs:
                worst_abs = a
                worst_feat = feat
        elif ds is not None and math.isinf(ds):
            if worst_feat is None or not math.isinf(worst_abs):
                worst_feat = feat
                worst_abs = math.inf

    # Sort by |drift_score| desc, INSUFFICIENT rows last.
    def _sort_drift(row):
        ds = row.get("drift_score")
        if ds is None:
            return (1, 0.0)
        if math.isinf(ds):
            return (0, -1e18)  # always at top
        return (0, -abs(ds))

    features.sort(key=_sort_drift)

    # Realized-return direction shift — a quick reader summary line.
    label_row = next(
        (f for f in features if f["feature"] == "forward_return_5d"), None
    )
    label_shift_pct = None
    if label_row and label_row.get("status") == "OK":
        try:
            label_shift_pct = (
                float(label_row["mean_recent"]) - float(label_row["mean_older"])
            )
        except (TypeError, ValueError, KeyError):
            label_shift_pct = None

    if overall_status == "SEVERE_DRIFT" and worst_feat:
        headline = (f"SEVERE drift in {worst_feat} "
                    f"({worst_abs:+.2f}σ vs older bucket)")
    elif overall_status == "MILD_DRIFT" and worst_feat:
        headline = (f"mild drift in {worst_feat} "
                    f"({worst_abs:+.2f}σ vs older bucket)")
    else:
        headline = f"stable — max drift {worst_abs:+.2f}σ"

    return {
        "state": "OK",
        "n_total": n_total,
        "n_recent": len(recent_rows), "n_older": len(older_rows),
        "recent_fraction": rec_frac,
        "verdict": overall_status,
        "headline": headline,
        "worst_feature": worst_feat,
        "worst_abs_drift": (worst_abs if not math.isinf(worst_abs)
                            else float("inf")),
        "label_shift_pct": (round(label_shift_pct, 4)
                            if label_shift_pct is not None else None),
        "features": features,
    }


def load_outcomes(path: Path | str) -> list[dict]:
    """Stream-load outcome records from a JSONL file. Returns ``[]`` on
    a missing file / unparseable line — never raises (mirrors
    ``feature_coverage.load_outcomes``).
    """
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


def analyze(outcomes_path: Path | str,
            recent_fraction: float = RECENT_FRACTION) -> dict:
    """Convenience: load + build. The CLI uses this."""
    return build_outcome_drift(load_outcomes(outcomes_path),
                               recent_fraction=recent_fraction)


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.outcome_drift",
        description=(
            "Decision-outcome distribution drift report. Splits "
            "decision_outcomes.jsonl into a recent vs older bucket "
            "and surfaces per-feature mean-shift in σ-of-older units, "
            "plus a realized-return directional shift."
        ),
    )
    p.add_argument("--path", default="data/decision_outcomes.jsonl",
                   help="Path to decision_outcomes.jsonl")
    p.add_argument("--recent-fraction", type=float, default=RECENT_FRACTION,
                   dest="recent_fraction",
                   help=(f"Share of the tail to treat as 'recent' "
                         f"(default {RECENT_FRACTION})"))
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of a table")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    rep = analyze(args.path, recent_fraction=args.recent_fraction)
    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
        return 0 if rep.get("state") == "OK" else 1

    print(f"[outcome_drift] state={rep.get('state')} "
          f"verdict={rep.get('verdict')}  "
          f"n_total={rep.get('n_total')}  "
          f"recent/older={rep.get('n_recent')}/{rep.get('n_older')}")
    print(f"  {rep.get('headline', '')}")
    if rep.get("label_shift_pct") is not None:
        print(f"  realized 5d-return shift: "
              f"{rep['label_shift_pct']:+.3f}% (recent − older)")
    if rep.get("state") != "OK":
        return 1
    print(f"  {'feature':<22}{'n_rec':>8}{'n_old':>8}"
          f"{'μ_recent':>12}{'μ_older':>12}{'σ_older':>10}"
          f"{'drift_σ':>10}  class")
    for r in rep["features"]:
        ds = r.get("drift_score")
        if ds is None:
            ds_s = "n/a"
        elif math.isinf(ds):
            ds_s = "+inf" if ds > 0 else "-inf"
        else:
            ds_s = f"{ds:+.3f}"
        mr = r.get("mean_recent")
        mo = r.get("mean_older")
        so = r.get("std_older")
        print(f"  {r['feature']:<22}"
              f"{r['n_recent']:>8}{r['n_older']:>8}"
              f"{(mr if mr is not None else 0):>12.4f}"
              f"{(mo if mo is not None else 0):>12.4f}"
              f"{(so if so is not None else 0):>10.4f}"
              f"{ds_s:>10}  {r['classification']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())

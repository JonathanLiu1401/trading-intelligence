"""Training-corpus data-quality auditor for ``decision_outcomes.jsonl``.

This is a **read-only diagnostic**. It never trains, never touches
``decision_scorer.pkl``, ``decision_outcomes.jsonl``, ``build_features``,
``N_FEATURES``, or any trade path — same operational discipline as the
existing ``paper_trader.ml.corpus_audit`` / ``calibration`` / ``skill_trend``
modules.

**Why this is not any existing tool.** ``corpus_audit`` validates the
*structural* honesty of the temporal-OOS split (does the holdout share
``run_id``s with training?). ``label_audit`` measures hindsight-contamination
of news labels in ``articles.db``. Neither inspects the OUTCOME records
themselves for *data quality* issues — silent fabricated zeros, NaN/inf in
feature columns, or duplicate (run_id, sim_date, ticker, action) rows whose
``forward_return_5d`` disagrees across runs (a clear sign of price-cache
drift between cycles). A skeptical quant treating ``oos_rmse`` /
``oos_ic`` as evidence needs to know that the underlying labels and
features survive a basic sanity bar.

The concrete motivation: ``run_continuous_backtests._compute_decision_outcomes``
silently produced ``forward_return_5d=0.0`` rows whenever both endpoints of
the 5/10/20-day window walked back via ``PriceCache.price_on`` to the *same*
prior close (a thin/foreign-calendar ticker like AAPL/MSFT in 1998, MSTR,
LLY). 23 such rows live in the current corpus — bias-correlated flat labels
that look identical to a real flat window. The walk-back collision is now
refused at write time (a separate fix), but the historical rows persist in
the trainer's 5000-record tail until they roll off.

This module surfaces:

| Field | Meaning |
|---|---|
| ``n_rows`` | total non-empty JSONL rows scanned |
| ``n_parsed`` | rows that parsed as JSON dicts |
| ``n_null_target_5d`` | rows where ``forward_return_5d`` is None (would be dropped by ``train_scorer``) |
| ``n_exact_zero_5d`` | rows where ``forward_return_5d`` is exactly 0.0 — the walk-back collision signature |
| ``n_extreme_5d`` | rows where ``|forward_return_5d| > EXTREME_PCT`` (sanity check on outliers) |
| ``n_nonfinite_feature`` | rows where any scorer-input feature is NaN/inf (would crash ``MLPRegressor.fit`` before the ``_to_float`` hardening landed) |
| ``n_conflict_dup_5d`` | (run_id, sim_date, ticker, action) groups with multiple rows whose ``forward_return_5d`` disagrees by ``> CONFLICT_TOL_PCT`` |
| ``target_stats`` | mean / std / p1 / p99 / min / max of the 5d target |
| ``feature_nonfinite_by_field`` | per-feature count of non-finite values |

Crisp threshold-driven verdict (exactly testable):

| Verdict | Trigger |
|---|---|
| ``INSUFFICIENT_DATA`` | < ``MIN_ROWS`` parsed rows |
| ``ZERO_LABEL_CONTAMINATION`` | exact-zero rate > ``ZERO_TARGET_RATE_MAX`` (walk-back fabrication footprint) |
| ``CONFLICTING_DUPLICATES`` | conflict count > ``CONFLICT_COUNT_MAX`` (price-cache drift between runs) |
| ``NONFINITE_FEATURES`` | non-finite-feature row count > ``NONFINITE_COUNT_MAX`` |
| ``CLEAN`` | every check passes — corpus is safe to train on |

Each verdict is the *first* trigger that fires (precedence above), so the
report names the most decisive issue rather than aggregating them into a
single mushy "looks bad" string. Multiple issues are still visible in the
counts — only the verdict label is precedence-ordered.

```bash
cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.outcome_data_quality
cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.outcome_data_quality --json
cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.outcome_data_quality --path /custom/path/to/outcomes.jsonl
```

Exit code mirrors ``host_guard`` / ``decision_scorer``'s CLI: 0 on
``CLEAN``, 1 on any non-clean verdict (so shell callers can gate on ``$?``).
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTCOMES_PATH = ROOT / "data" / "decision_outcomes.jsonl"

# Need at least this many parsed rows before any quality metric is
# meaningful. Mirrors the conventions used by sibling diagnostics — well
# below the trainer's 30-record dedup floor so that even a barely-populated
# corpus gets a real audit instead of an INSUFFICIENT_DATA early-out
# masking the actual issue.
MIN_ROWS = 50

# What "extreme" means for a 5-trading-day forward return. From the live
# data: p99 ≈ +32%, p1 ≈ -25%; |fr_5d| > 50% is essentially always a
# 3x-leveraged ETF crash/rip week (real but extreme). Anything larger is
# almost certainly a yfinance data glitch or a price-cache bug — surface
# but do NOT fail on. Matches `decision_scorer.PRED_CLAMP_PCT` so the
# extreme-label rate visible to a quant is the same boundary the scorer
# clamps to.
EXTREME_PCT = 50.0

# Threshold on the exact-zero rate. The walk-back collision rate observed
# live before the fix was 23/7413 = 0.31%. Set the bar at 0.5% so a
# uniform contamination level (e.g., legacy rows that pre-date the fix)
# trips the verdict but a single "genuine flat week" row in a small
# corpus doesn't.
ZERO_TARGET_RATE_MAX = 0.005

# Conflict-detection tolerance — duplicate (run_id, sim_date, ticker,
# action) rows whose `forward_return_5d` differs by MORE than this many
# percentage points count as a conflict. Rounding-noise differences below
# this floor are normal (the 4-decimal `round` in `_compute_decision_outcomes`
# can shift the last digit between cycles when float math reorders).
CONFLICT_TOL_PCT = 0.01

# Maximum tolerated conflict count before the corpus is marked as
# price-cache-drifted. 1 conflict is plausible noise; >5 is a real
# regression in cache reproducibility.
CONFLICT_COUNT_MAX = 5

# Maximum tolerated non-finite-feature row count. The `_to_float`
# hardening in `decision_scorer` already coerces non-finite inputs to a
# safe default at predict/train time, but a non-finite VALUE in the
# outcome record itself is a sign of upstream bug (e.g., a regex parse
# leaking inf, or a yfinance NaN propagating). Keep the bar low.
NONFINITE_COUNT_MAX = 5

# Per-feature numeric columns scanned for NaN/inf. Matches the keys
# `_compute_decision_outcomes` emits per record (NOT the 17-feature
# `build_features` slot list — this audits the raw on-disk record before
# build_features has a chance to coerce). Keeping this list in lockstep
# with `_compute_decision_outcomes` is a maintenance contract.
NUMERIC_FEATURE_KEYS = (
    "ml_score", "rsi", "macd", "mom5", "mom20",
    "regime_mult", "vol_ratio", "bb_position",
    "news_urgency", "news_article_count",
    "forward_return_5d", "forward_return_10d", "forward_return_20d",
)


def _is_finite_or_none(v: Any) -> bool:
    """True if v is None or a finite number. False for NaN / inf / non-numeric.

    None is allowed because optional features (e.g. `forward_return_10d`,
    `news_urgency`) are legitimately absent for older rows. A non-finite
    NUMERIC value is the actual bug we want to flag.
    """
    if v is None:
        return True
    if isinstance(v, bool):
        # bool is a numeric subclass in Python; an explicit True/False in a
        # numeric slot is a type error, not a finite value.
        return False
    if isinstance(v, (int, float)):
        return math.isfinite(v)
    return False


def _safe_float(v: Any) -> float | None:
    """Coerce to float when v is a finite number, else None.

    Bools are rejected (same rule as `_is_finite_or_none`) so a True/False
    in a numeric slot does not silently become 1.0/0.0 in the stats.
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)) and math.isfinite(v):
        return float(v)
    return None


def _percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile, scipy-free. `pct` in [0, 100]. None on empty."""
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    if pct <= 0:
        return s[0]
    if pct >= 100:
        return s[-1]
    k = (len(s) - 1) * pct / 100.0
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return s[lo]
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def audit_outcomes(path: Path | str | None = None) -> dict:
    """Scan `decision_outcomes.jsonl` for training-corpus data-quality issues.

    Pure read; opens the file once, streams it line by line so peak memory is
    bounded at the size of the per-key conflict-detection map. Every fault
    degrades gracefully (a corrupt line is counted but doesn't abort the
    audit) — mirroring the `_inject_and_train`/winner-trim discipline.

    Returns a JSON-safe dict suitable for both the CLI table renderer below
    and (future) dashboard wiring. Never raises — the outer try/except
    captures an unexpected fault and emits a `status='error'` row so a
    skeptical quant can see the audit itself broke rather than silently
    missing the verdict.
    """
    p = Path(path) if path is not None else DEFAULT_OUTCOMES_PATH
    out: dict[str, Any] = {
        "status": "ok",
        "verdict": "INSUFFICIENT_DATA",
        "path": str(p),
        "n_rows": 0,
        "n_parsed": 0,
        "n_corrupt": 0,
        "n_null_target_5d": 0,
        "n_exact_zero_5d": 0,
        "n_extreme_5d": 0,
        "n_nonfinite_feature": 0,
        "n_conflict_dup_5d": 0,
        "feature_nonfinite_by_field": {k: 0 for k in NUMERIC_FEATURE_KEYS},
        "target_stats": None,
    }
    if not p.exists():
        out["status"] = "error"
        out["error"] = f"outcomes file not found: {p}"
        return out

    target_5d: list[float] = []
    # (run_id, sim_date, ticker, action) -> list of forward_return_5d seen
    dup_map: dict[tuple, list[float]] = {}

    try:
        with p.open("r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                out["n_rows"] += 1
                try:
                    r = json.loads(line)
                except Exception:
                    out["n_corrupt"] += 1
                    continue
                if not isinstance(r, dict):
                    out["n_corrupt"] += 1
                    continue
                out["n_parsed"] += 1

                # Target sanity
                fr5 = r.get("forward_return_5d")
                if fr5 is None:
                    out["n_null_target_5d"] += 1
                else:
                    fr5_f = _safe_float(fr5)
                    if fr5_f is None:
                        # Numeric but non-finite — counted under
                        # nonfinite_feature below as well.
                        pass
                    else:
                        target_5d.append(fr5_f)
                        if fr5_f == 0.0:
                            out["n_exact_zero_5d"] += 1
                        if abs(fr5_f) > EXTREME_PCT:
                            out["n_extreme_5d"] += 1
                        # Dup-conflict tracking — key on stable identifiers.
                        key = (
                            r.get("run_id"),
                            r.get("sim_date"),
                            str(r.get("ticker") or "").upper(),
                            str(r.get("action") or "").upper(),
                        )
                        dup_map.setdefault(key, []).append(fr5_f)

                # Per-feature non-finite detection
                row_dirty = False
                for k in NUMERIC_FEATURE_KEYS:
                    v = r.get(k)
                    if not _is_finite_or_none(v):
                        out["feature_nonfinite_by_field"][k] += 1
                        row_dirty = True
                if row_dirty:
                    out["n_nonfinite_feature"] += 1

        # Conflict detection: groups with > 1 value whose spread exceeds
        # the rounding-noise tolerance.
        for key, vals in dup_map.items():
            if len(vals) < 2:
                continue
            if (max(vals) - min(vals)) > CONFLICT_TOL_PCT:
                out["n_conflict_dup_5d"] += 1
    except Exception as exc:
        out["status"] = "error"
        out["error"] = f"audit faulted: {type(exc).__name__}: {exc}"
        return out

    # Target distribution stats (mean/std/p1/p99/min/max) — read-only,
    # never used to alter behaviour, just exposed so a quant can sanity-
    # check the corpus against the documented p1=-25 / p99=+32 baseline.
    if target_5d:
        out["target_stats"] = {
            "n": len(target_5d),
            "mean": round(statistics.fmean(target_5d), 4),
            "std": round(statistics.pstdev(target_5d), 4)
                   if len(target_5d) > 1 else 0.0,
            "min": round(min(target_5d), 4),
            "max": round(max(target_5d), 4),
            "p1": round(_percentile(target_5d, 1.0) or 0.0, 4),
            "p99": round(_percentile(target_5d, 99.0) or 0.0, 4),
        }

    # Verdict (precedence-ordered — first trigger wins).
    if out["n_parsed"] < MIN_ROWS:
        out["verdict"] = "INSUFFICIENT_DATA"
    elif out["n_parsed"] > 0 and (
        out["n_exact_zero_5d"] / out["n_parsed"] > ZERO_TARGET_RATE_MAX
    ):
        out["verdict"] = "ZERO_LABEL_CONTAMINATION"
    elif out["n_conflict_dup_5d"] > CONFLICT_COUNT_MAX:
        out["verdict"] = "CONFLICTING_DUPLICATES"
    elif out["n_nonfinite_feature"] > NONFINITE_COUNT_MAX:
        out["verdict"] = "NONFINITE_FEATURES"
    else:
        out["verdict"] = "CLEAN"

    return out


# ─────────────────────────── CLI ───────────────────────────

def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.outcome_data_quality",
        description="Audit decision_outcomes.jsonl for training-corpus data-"
                    "quality issues (exact-zero contamination, non-finite "
                    "features, conflicting duplicates). Read-only — never "
                    "writes or retrains. Exits 0 on CLEAN, 1 otherwise.",
    )
    p.add_argument("--path", default=str(DEFAULT_OUTCOMES_PATH),
                   help="Path to decision_outcomes.jsonl "
                        f"(default: {DEFAULT_OUTCOMES_PATH}).")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of a table.")
    return p


def main(argv: list[str] | None = None) -> int:
    import sys

    args = _build_arg_parser().parse_args(
        sys.argv[1:] if argv is None else argv)
    rep = audit_outcomes(args.path)

    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
        return 0 if rep.get("verdict") == "CLEAN" else 1

    print(f"[outcome_data_quality] {rep['path']}")
    print(f"  verdict: {rep['verdict']}")
    print(f"  rows: {rep['n_rows']} parsed={rep['n_parsed']} "
          f"corrupt={rep['n_corrupt']}")
    print(f"  target_5d: null={rep['n_null_target_5d']} "
          f"exact_zero={rep['n_exact_zero_5d']} "
          f"extreme(|>{EXTREME_PCT:.0f}%|)={rep['n_extreme_5d']}")
    print(f"  conflicts (dup keys, |Δfr5d| > "
          f"{CONFLICT_TOL_PCT:.2f}pp): {rep['n_conflict_dup_5d']}")
    print(f"  nonfinite-feature rows: {rep['n_nonfinite_feature']}")
    if rep.get("target_stats"):
        ts = rep["target_stats"]
        print(f"  fr_5d distribution n={ts['n']}  mean={ts['mean']:+.2f}%  "
              f"std={ts['std']:.2f}  p1={ts['p1']:+.2f}%  "
              f"p99={ts['p99']:+.2f}%  min={ts['min']:+.2f}%  "
              f"max={ts['max']:+.2f}%")
    nonfinite_by_field = {k: v for k, v
                          in rep["feature_nonfinite_by_field"].items()
                          if v > 0}
    if nonfinite_by_field:
        print("  per-feature non-finite counts:")
        for k, v in sorted(nonfinite_by_field.items(),
                           key=lambda kv: -kv[1]):
            print(f"    {k}: {v}")
    return 0 if rep.get("verdict") == "CLEAN" else 1


if __name__ == "__main__":
    raise SystemExit(main())

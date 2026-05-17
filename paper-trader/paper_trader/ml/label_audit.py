"""Training-label hygiene audit — how many DecisionScorer training labels
are split/corporate-action artifacts rather than real 5-day moves?

This is a **read-only diagnostic**, the exact sibling of
`paper_trader/ml/calibration.py`. It never trains, never touches
`decision_scorer.pkl`, `decision_outcomes.jsonl`, `build_features`,
`N_FEATURES`, or any trade path — so it cannot perturb the unattended
continuous loop or break pickle compatibility (AGENTS.md "When to bump
model versions" / "Common pitfalls"). It does **not** filter, winsorize,
or rewrite the label set: re-shaping the `y` vector in `train_scorer` is
the documented out-of-scope training-dynamics change. This tool exists to
tell the operator *when* to invoke the documented remediation (delete the
pickle and let the loop retrain) — it is the missing measurement, not a
silent fix.

Why this matters to a quant: `PriceCache` fetches with
`yf.history(auto_adjust=False)`, so a reverse split (e.g. DFEN's 2024-06
1:5) injects a step discontinuity into the raw close series. A backtest
BUY straddling that step records a `forward_return_5d` of +180% that is
**pure corporate-action noise, not signal**. `MLPRegressor` has no output
bound, so a handful of such labels — *up-weighted 2–4× by the run-quality
oversampling in `train_scorer`* — teach the unbounded head to extrapolate,
which is precisely the off-distribution behaviour the inference-side
`PRED_CLAMP_PCT` clamp was added to suppress. The clamp protects
*predictions*; nothing protects the *labels*. This audit measures that gap.

The boundary is `PRED_CLAMP_PCT` itself (imported — single source of
truth): across the real `decision_outcomes.jsonl` only ~0.4–0.5% of 5d
outcomes exceed |50%| (p1≈−25%, p99≈+31%), and the inference head is
already clamped there. A label past that bound is, by the system's own
established definition, outside the empirical support.

Verdict (crisp, threshold-driven so it is exactly testable):

| Verdict | Meaning |
|---------|---------|
| `INSUFFICIENT_DATA` | < ``MIN_RECORDS`` finite outcome rows |
| `CLEAN` | extreme-label rate ≤ ``CLEAN_MAX_RATE`` (≈ the documented real baseline) |
| `ELEVATED` | ``CLEAN_MAX_RATE`` < rate ≤ ``ELEVATED_MAX_RATE`` — watch it |
| `CONTAMINATED` | rate > ``ELEVATED_MAX_RATE`` — retrain off a cleaned tail |
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .decision_scorer import PRED_CLAMP_PCT

# Thresholds are module-level so tests assert exact verdicts and a tuning
# change is a single reviewable edit (mirrors calibration.py's convention
# and the codebase's constants-at-module-scope rule, e.g. PRED_CLAMP_PCT).
MIN_RECORDS = 30

# A label is "extreme" past the same bound the inference head is clamped to.
# Imported, not redefined, so the audit can never drift from the clamp it
# is auditing against (single source of truth — AGENTS.md invariant #10
# spirit, mirrored from _oos_rank_metrics reusing calibration._spearman).
EXTREME_RETURN_PCT = PRED_CLAMP_PCT  # = 50.0

CLEAN_MAX_RATE = 0.006      # ≈ the documented ~0.4–0.5% real |fwd|>50 baseline
ELEVATED_MAX_RATE = 0.015   # 2–3× the baseline → actively watch

# Directional-anomaly heuristic (INFORMATIONAL ONLY, never drives the
# verdict): a large forward move whose sign opposes recent momentum is a
# split-signature *candidate* — but a genuine crash-rebound (2020-03 COVID
# on a 3× ETF) has the same shape, so this is reported as context, not used
# to classify.
DIRECTIONAL_PCT = 40.0
DIRECTIONAL_MOM_MIN = 5.0


def _finite(v):
    """Parse to a finite float or return None. Mirrors calibration.py's
    drop-non-finite hardening (a single inf/nan must not poison the audit —
    same class as ``decision_scorer._to_float``)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def audit_outcome_labels(records, top_n: int = 10) -> dict:
    """Audit a list of ``decision_outcomes.jsonl`` rows for extreme /
    split-artifact-suspect ``forward_return_5d`` training labels.

    ``records`` is any iterable of dicts with at least ``forward_return_5d``;
    ``ticker``, ``sim_date``, ``mom5``, ``action`` are used when present.
    Magnitude analysis is action-agnostic on purpose: a split discontinuity
    lives in the raw price series regardless of whether the decision was a
    BUY or a SELL, and ``train_scorer``'s SELL sign-flip does not change
    ``|forward_return_5d|``.

    Returns a JSON-safe dict:
    ``{status, verdict, n, extreme_count, extreme_rate, extreme_pct_bound,
       directional_anomaly_count, directional_anomaly_rate,
       worst_offenders:[{ticker,extreme_count,n,rate}],
       worst_labels:[{ticker,sim_date,forward_return_5d,mom5,action}],
       hint}``.
    """
    n = 0
    extreme = 0
    directional = 0
    per_ticker: dict[str, dict] = {}
    worst: list[dict] = []

    for r in records:
        f = _finite(r.get("forward_return_5d"))
        if f is None:
            continue
        n += 1
        ticker = str(r.get("ticker") or "") or "?"
        slot = per_ticker.setdefault(ticker, {"n": 0, "extreme": 0})
        slot["n"] += 1
        is_extreme = abs(f) > EXTREME_RETURN_PCT
        if is_extreme:
            extreme += 1
            slot["extreme"] += 1
            worst.append({
                "ticker": ticker,
                "sim_date": str(r.get("sim_date") or ""),
                "forward_return_5d": round(f, 4),
                "mom5": (round(m, 4) if (m := _finite(r.get("mom5"))) is not None
                         else None),
                "action": str(r.get("action") or "BUY").upper(),
            })
        m5 = _finite(r.get("mom5"))
        if (m5 is not None and abs(f) > DIRECTIONAL_PCT
                and abs(m5) > DIRECTIONAL_MOM_MIN
                and (f > 0) != (m5 > 0)):
            directional += 1

    if n < MIN_RECORDS:
        return {
            "status": "insufficient_data",
            "verdict": "INSUFFICIENT_DATA",
            "n": n,
            "extreme_count": extreme,
            "extreme_rate": None,
            "extreme_pct_bound": EXTREME_RETURN_PCT,
            "directional_anomaly_count": directional,
            "directional_anomaly_rate": None,
            "worst_offenders": [],
            "worst_labels": [],
            "hint": f"need ≥{MIN_RECORDS} finite outcome rows, have {n}",
        }

    rate = extreme / n
    dir_rate = directional / n

    # Per-ticker offenders: only tickers that actually contributed an extreme
    # label, sorted by extreme count desc then rate desc (deterministic).
    offenders = sorted(
        ({"ticker": t, "extreme_count": s["extreme"], "n": s["n"],
          "rate": round(s["extreme"] / s["n"], 4)}
         for t, s in per_ticker.items() if s["extreme"] > 0),
        key=lambda d: (-d["extreme_count"], -d["rate"], d["ticker"]),
    )[:top_n]

    worst_sorted = sorted(
        worst, key=lambda d: abs(d["forward_return_5d"]), reverse=True
    )[:top_n]

    if rate <= CLEAN_MAX_RATE:
        verdict = "CLEAN"
        hint = ("extreme-label rate is within the documented real-outcome "
                "baseline — training labels are not split-contaminated")
    elif rate <= ELEVATED_MAX_RATE:
        verdict = "ELEVATED"
        hint = ("extreme-label rate is 2–3× the real baseline — watch the "
                "per-ticker offenders (known reverse-split names); the "
                "scorer's tail predictions will over-extrapolate")
    else:
        verdict = "CONTAMINATED"
        hint = ("extreme-label rate far exceeds the real baseline — the "
                "unbounded MLP head is being trained toward "
                "corporate-action noise. Remediation is the documented "
                "protocol: delete data/ml/decision_scorer.pkl and let the "
                "next continuous cycle retrain (do NOT winsorize y in "
                "train_scorer — that is the out-of-scope training-dynamics "
                "change)")

    return {
        "status": "ok",
        "verdict": verdict,
        "n": n,
        "extreme_count": extreme,
        "extreme_rate": round(rate, 6),
        "extreme_pct_bound": EXTREME_RETURN_PCT,
        "directional_anomaly_count": directional,
        "directional_anomaly_rate": round(dir_rate, 6),
        "worst_offenders": offenders,
        "worst_labels": worst_sorted,
        "hint": hint,
    }


def _load_outcomes(path: Path) -> list[dict]:
    """Read a ``decision_outcomes.jsonl`` file, skipping blank/corrupt lines
    (a single bad line must not abort the audit — mirrors the loop's own
    per-line parse hardening)."""
    out: list[dict] = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            pass
    return out


def _cli() -> int:
    """`python3 -m paper_trader.ml.label_audit` — label hygiene of the
    accumulated outcomes tail. Read-only; never writes anything."""
    root = Path(__file__).resolve().parent.parent.parent
    out_path = root / "data" / "decision_outcomes.jsonl"
    if not out_path.exists():
        print(f"[label_audit] no outcomes file at {out_path}")
        return 1
    records = _load_outcomes(out_path)
    rep = audit_outcome_labels(records)
    print(f"outcomes={len(records)}  bound=±{rep['extreme_pct_bound']:.0f}% "
          f"(= PRED_CLAMP_PCT)")
    print(f"VERDICT: {rep['verdict']}  ({rep['hint']})")
    er = rep["extreme_rate"]
    dr = rep["directional_anomaly_rate"]
    print(f"  n={rep['n']} extreme={rep['extreme_count']} "
          f"({'n/a' if er is None else f'{er*100:.3f}%'})  "
          f"directional_anomaly={rep['directional_anomaly_count']} "
          f"({'n/a' if dr is None else f'{dr*100:.3f}%'})")
    if rep["worst_offenders"]:
        print("  worst offenders (ticker: extreme/n):")
        for o in rep["worst_offenders"]:
            print(f"    {o['ticker']:<7} {o['extreme_count']}/{o['n']} "
                  f"({o['rate']*100:.2f}%)")
    if rep["worst_labels"]:
        print("  most extreme labels:")
        for w in rep["worst_labels"]:
            print(f"    {w['ticker']:<7} {w['sim_date']} "
                  f"{w['action']:<4} fwd={w['forward_return_5d']:+8.2f}% "
                  f"mom5={w['mom5']}")
    # Verdict-driven exit code so an operator/cron can branch on it, exactly
    # like calibration._cli's contract (0 healthy, non-zero needs attention).
    return 0 if rep["verdict"] in ("CLEAN", "INSUFFICIENT_DATA") else 2


if __name__ == "__main__":
    raise SystemExit(_cli())

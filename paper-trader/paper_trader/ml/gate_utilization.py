"""Conviction-gate **utilization** audit — does the scorer's actual OOS
prediction distribution populate the five gate arms in a balanced way, or
is the gate effectively a no-op for most decisions?

This is a **read-only diagnostic**. Same operational discipline as
``paper_trader/ml/gate_audit.py`` / ``calibration.py`` / ``skill_trend.py``:
never trains, never touches ``decision_scorer.pkl`` /
``decision_outcomes.jsonl`` / ``build_features`` / ``N_FEATURES`` / any
trade path. Safe under the live unattended continuous loop, cannot break
pickle compatibility.

**Why this is not ``gate_audit``.** ``gate_audit`` answers the *economic*
question: given the trades that fell into each arm, did the strong_tailwind
arm realize a higher mean return than the strong_headwind arm? It compares
the **realized spread** to the **multiplier spread** and emits
``GATE_HARMFUL`` / ``GATE_INEFFECTIVE`` / ``GATE_EFFECTIVE``.

That tells you whether the gate's bet on the predictions was directionally
right. It does NOT tell you whether the gate was **even reachable** —
specifically:

  * If 95% of OOS predictions fall in the neutral arm, the four
    conviction multipliers are a no-op on essentially every decision.
    The gate is irrelevant by construction; ``gate_audit``'s verdict
    is computed on a thin slice of the book and likely reads
    ``INSUFFICIENT_DATA``.
  * If the prediction distribution is wholly contained in one half
    (e.g. all predictions in ``[+5, +∞)``, never the headwind arms),
    the multiplier asymmetry built into the gate has nothing to bite on —
    real headwind capital allocation never happens.
  * If a single extreme arm is empty (e.g. zero ``strong_headwind``
    predictions), the gate is structurally one-sided regardless of any
    multiplier-vs-realized comparison.

This diagnostic answers exactly that **distributional reachability**
question, complementary to (and never overlapping with) ``gate_audit``'s
economic verdict. The two together close the audit loop on the gate:

    gate_utilization    — does the gate get used in a balanced way?
    gate_audit          — when it does get used, are the bets right?

**Method.** Reuses ``gate_arm()`` from ``gate_audit`` (single source of
truth — AGENTS.md invariant #10 spirit) so this module and that one can
never disagree on which arm a prediction belongs to. The gate-arm
membership semantics — every boundary operator, the if/elif order, the
non-finite → neutral fallback — are byte-identical between the two.

Reports:

| Field | Meaning |
|---|---|
| ``n`` | total OOS predictions analysed |
| ``arms`` | per-arm ``{arm, multiplier, n, pct_of_total}`` |
| ``pred_distribution`` | ``min/p10/p25/median/p75/p90/max`` of OOS predictions |
| ``empty_arms`` | list of arm names with zero population (the ``[]`` case is healthy) |
| ``lopsided_arm`` | arm carrying ≥ ``LOPSIDED_PCT`` of predictions, else ``None`` |
| ``neutral_dominated`` | True iff ``neutral`` arm carries ≥ ``LOPSIDED_PCT`` |
| ``side_balance`` | (n_tailwind + n_neutral_positive_half) / n — fraction of predictions on the tailwind side |
| ``verdict`` | crisp threshold-driven verdict, see below |

Verdict ladder (ordered: alarming first — same precedent as
``feature_value_skill.HAS_INVERTED_BUCKET`` surfacing first):

| Verdict | Trigger |
|---|---|
| ``INSUFFICIENT_DATA`` | < ``MIN_TOTAL`` predictions |
| ``EMPTY_EXTREME_ARM`` | ``strong_headwind`` or ``strong_tailwind`` has zero population — the gate cannot exercise its full multiplier range |
| ``NEUTRAL_DOMINATED`` | ≥ ``LOPSIDED_PCT`` of predictions in neutral — the four multipliers are inactive for most decisions |
| ``LOPSIDED`` | ≥ ``LOPSIDED_PCT`` of predictions in a single non-neutral arm — the gate is acting like a constant multiplier |
| ``BALANCED`` | every arm has ≥ ``MIN_ARM_PCT`` of predictions AND no arm exceeds ``LOPSIDED_PCT`` |
| ``WEAK_BALANCE`` | every arm populated but ≥1 arm under ``MIN_ARM_PCT`` — the gate is reachable everywhere but thinly in some arms |

Verdict precedence ensures the most actionable signal is surfaced first:
an empty extreme arm dominates a lopsided one, which dominates a balanced
verdict. Tests pin the precedence at the threshold boundaries.

```bash
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.gate_utilization
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.gate_utilization --json
```
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from paper_trader.ml.gate_audit import gate_arm, GATE_ARMS, _ARM_ORDER, _ARM_MULT

# Thresholds — module-level so tests assert exact verdicts at boundaries
# and a tuning change is one reviewable edit (mirrors gate_audit /
# calibration / news_volume_skill — the codebase constants-at-module-scope
# convention). Percentages are expressed as fractions (0.60 = 60%).
MIN_TOTAL = 30        # need a real sample before any verdict (mirrors gate_audit.MIN_TOTAL)
MIN_ARM_PCT = 0.05    # an arm under this share reads as thinly populated
LOPSIDED_PCT = 0.60   # one arm carrying ≥60% of predictions dominates the verdict


def _percentiles(values: list[float]) -> dict:
    """Empirical quantiles of a non-empty 1D float list.

    Uses ``np.percentile`` with the default linear interpolation — the
    operational standard for descriptive distribution reporting (and what
    every existing diagnostic that quantile-buckets the OOS slice uses).
    Returns rounded floats so the JSON payload is stable across NumPy
    floating-point round-trips.
    """
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": round(float(arr.min()), 4),
        "p10": round(float(np.percentile(arr, 10)), 4),
        "p25": round(float(np.percentile(arr, 25)), 4),
        "median": round(float(np.percentile(arr, 50)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "max": round(float(arr.max()), 4),
    }


def gate_utilization_report(predictions) -> dict:
    """Distributional gate-arm utilization report.

    ``predictions`` is any iterable of scalar prediction values (one per
    OOS decision). Realized returns are NOT needed here — the question is
    about prediction-distribution reachability, not realized skill. (Use
    ``gate_audit.gate_effectiveness_report`` for the realized-spread view.)

    A non-finite / unparseable value reports to ``("neutral", 1.0)`` via
    ``gate_arm()`` — the same off-distribution-treatment-as-no-op rule
    the live gate uses, so this metric never artificially populates an
    extreme arm. Pure / never raises (the AGENTS.md _safe contract every
    sibling diagnostic respects).

    Returns a JSON-safe dict. Never raises on bad input — a non-iterable
    or completely empty input yields ``INSUFFICIENT_DATA`` with ``n=0``.
    """
    # ── 1. Clean + bucket ────────────────────────────────────────────────
    cleaned: list[float] = []
    per_arm: dict[str, list[float]] = {a: [] for a in _ARM_ORDER}
    try:
        for p in predictions:
            try:
                pf = float(p)
            except (TypeError, ValueError):
                # Mirror gate_arm()'s non-finite-as-neutral: a raw garbage
                # row goes to the neutral arm rather than dropping silently
                # (would otherwise under-count the neutral population a
                # quant cares about when verdict reads NEUTRAL_DOMINATED).
                arm, _ = gate_arm(p)
                per_arm[arm].append(0.0)
                cleaned.append(0.0)
                continue
            if not np.isfinite(pf):
                arm, _ = gate_arm(pf)
                per_arm[arm].append(pf)
                cleaned.append(pf)
                continue
            arm, _ = gate_arm(pf)
            per_arm[arm].append(pf)
            cleaned.append(pf)
    except TypeError:
        # `predictions` was not iterable at all — the documented degrade
        # path mirrors gate_audit's "INSUFFICIENT_DATA + n=0".
        cleaned = []
        per_arm = {a: [] for a in _ARM_ORDER}

    n = len(cleaned)
    base = {
        "status": "ok",
        "verdict": "INSUFFICIENT_DATA",
        "n": n,
        "arms": [],
        "pred_distribution": None,
        "empty_arms": [],
        "lopsided_arm": None,
        "neutral_dominated": False,
        "side_balance": None,
        "hint": "",
    }

    # ── 2. Per-arm counts + percentages (always emitted, even sub-MIN) ───
    arms_out: list[dict] = []
    empty_arms: list[str] = []
    for a in _ARM_ORDER:
        cnt = len(per_arm[a])
        pct = (cnt / n) if n > 0 else 0.0
        if cnt == 0:
            empty_arms.append(a)
        arms_out.append({
            "arm": a,
            "multiplier": _ARM_MULT[a],
            "n": cnt,
            "pct_of_total": round(pct, 4),
        })
    base["arms"] = arms_out
    base["empty_arms"] = empty_arms

    # ── 3. Distribution + side balance (need at least one prediction) ────
    if n > 0:
        # NaN/Inf values can poison np.percentile silently; filter for the
        # distribution stats while keeping the arm counts honest (the gate
        # already maps non-finite → neutral, so it IS counted there).
        finite = [v for v in cleaned if np.isfinite(v)]
        if finite:
            base["pred_distribution"] = _percentiles(finite)
        # side_balance: fraction of predictions on the tailwind side.
        # `neutral` arm spans [0, 5] inclusive — the live `_ml_decide`
        # treats `p == 0` as neutral (the else branch), so split neutrals
        # by their actual sign for an honest tailwind/headwind tally that
        # ignores arm semantics. This is purely informational.
        n_tail = sum(1 for v in finite if v > 0.0)
        base["side_balance"] = round(n_tail / n, 4) if n else None

    # ── 4. Verdict — INSUFFICIENT_DATA short-circuit first ───────────────
    if n < MIN_TOTAL:
        base["hint"] = (
            f"need ≥{MIN_TOTAL} predictions for a utilization verdict; "
            f"have n={n}"
        )
        return base

    # EMPTY_EXTREME_ARM: the gate's full multiplier range (×0.6 ↔ ×1.3) is
    # structurally unreachable when either extreme is unpopulated. This
    # is the strongest single-signal finding and surfaces first regardless
    # of LOPSIDED / NEUTRAL state. Equal-share gate_audit / EMPTY_EXTREME
    # discriminator: gate_audit's INSUFFICIENT_DATA fires under <MIN_ARM_N=5
    # in either extreme; this verdict fires only at EXACTLY zero — a
    # stricter signal still consistent with that gate_audit cell already
    # reading INSUFFICIENT_DATA.
    if "strong_headwind" in empty_arms or "strong_tailwind" in empty_arms:
        base["verdict"] = "EMPTY_EXTREME_ARM"
        base["hint"] = (
            f"empty arm(s): {', '.join(empty_arms)} — the gate's "
            f"×0.60/×1.30 multipliers are structurally unreachable in "
            f"this prediction distribution"
        )
        return base

    # NEUTRAL_DOMINATED / LOPSIDED: a single arm holds ≥ LOPSIDED_PCT of
    # the predictions. Neutral dominance is reported as its own verdict
    # since the operational consequence — "the gate is a no-op for most
    # decisions" — is qualitatively different from "the gate fires its
    # extreme multipliers most of the time".
    arms_sorted = sorted(arms_out, key=lambda o: -o["n"])
    top_arm = arms_sorted[0]
    if top_arm["pct_of_total"] >= LOPSIDED_PCT:
        if top_arm["arm"] == "neutral":
            base["verdict"] = "NEUTRAL_DOMINATED"
            base["neutral_dominated"] = True
            base["hint"] = (
                f"neutral arm carries {top_arm['pct_of_total']*100:.1f}% "
                f"of predictions (≥{LOPSIDED_PCT*100:.0f}%) — the four "
                f"non-neutral multipliers are inactive for most decisions"
            )
        else:
            base["verdict"] = "LOPSIDED"
            base["lopsided_arm"] = top_arm["arm"]
            base["hint"] = (
                f"{top_arm['arm']} arm carries "
                f"{top_arm['pct_of_total']*100:.1f}% of predictions — the "
                f"gate is behaving like a constant ×{top_arm['multiplier']:.2f} "
                f"multiplier for the bulk of decisions"
            )
        return base

    # BALANCED / WEAK_BALANCE: every arm has at least SOME population
    # (already proved by the empty-arm check above). Distinguish full
    # balance (every arm ≥ MIN_ARM_PCT) from weak balance (≥1 arm < that).
    weak_arms = [o["arm"] for o in arms_out if o["pct_of_total"] < MIN_ARM_PCT]
    if weak_arms:
        base["verdict"] = "WEAK_BALANCE"
        base["hint"] = (
            f"every arm reachable, but thinly populated arm(s): "
            f"{', '.join(weak_arms)} (each <{MIN_ARM_PCT*100:.0f}%)"
        )
    else:
        base["verdict"] = "BALANCED"
        base["hint"] = (
            f"all five arms populated ≥{MIN_ARM_PCT*100:.0f}%, no single arm "
            f"≥{LOPSIDED_PCT*100:.0f}% — the gate is being exercised across "
            f"its full multiplier range"
        )
    return base


# ─────────────────────────────────────────────────────────────────────────
# Top-level analyzer (mirrors gate_audit.analyze / baseline_compare.analyze
# operational shape so the CLI is the same one-liner discipline operators
# already use across the diagnostic suite).
# ─────────────────────────────────────────────────────────────────────────

def analyze(outcomes_path: Path | str, oos_only: bool = True) -> dict:
    """Read the outcomes JSONL, optionally restrict to the temporal-OOS
    holdout (the slice the scorer-skill ledger trusts), predict with the
    deployed scorer, and emit the utilization report.

    Returns a JSON-safe dict. Never raises — any IO/fault degrades to
    ``status='error'`` with an ``INSUFFICIENT_DATA`` verdict (mirrors the
    sibling analyzers).
    """
    out: dict = {"status": "error", "verdict": "INSUFFICIENT_DATA",
                 "n": 0, "arms": [], "pred_distribution": None,
                 "empty_arms": [], "lopsided_arm": None,
                 "neutral_dominated": False, "side_balance": None,
                 "hint": ""}
    try:
        path = Path(outcomes_path)
        if not path.exists():
            out["hint"] = f"outcomes file missing: {path}"
            return out
        records: list[dict] = []
        with path.open("r") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    records.append(json.loads(ln))
                except Exception:
                    continue
        if oos_only:
            try:
                from paper_trader.validation import split_outcomes_temporal
                _, records = split_outcomes_temporal(records, oos_fraction=0.2)
            except Exception:
                # Mirror _train_decision_scorer's degrade path — fall back
                # to the full corpus rather than failing the whole report
                # on a validation-module issue.
                pass

        from paper_trader.ml.decision_scorer import DecisionScorer, _to_float
        scorer = DecisionScorer()
        if not scorer.is_trained:
            out["hint"] = "scorer not trained — no on-disk pickle"
            return out

        preds: list[float] = []
        for r in records:
            try:
                p = scorer.predict(
                    ml_score=_to_float(r.get("ml_score"), 0.0),
                    rsi=r.get("rsi"), macd=r.get("macd"),
                    mom5=r.get("mom5"), mom20=r.get("mom20"),
                    regime_mult=_to_float(r.get("regime_mult"), 1.0),
                    ticker=str(r.get("ticker") or ""),
                    vol_ratio=r.get("vol_ratio"),
                    bb_pos=r.get("bb_position"),
                    news_urgency=r.get("news_urgency"),
                    news_article_count=r.get("news_article_count"),
                ema200_above=r.get("ema200_above"),
                hist_cross_up=r.get("hist_cross_up"),
                macd_below_zero_cross=r.get("macd_below_zero_cross"),
                )
                preds.append(float(p))
            except Exception:
                continue

        rep = gate_utilization_report(preds)
        rep["slice"] = "oos" if oos_only else "all"
        rep["n_train"] = scorer.n_train
        return rep
    except Exception as exc:
        out["hint"] = f"analyze fault: {type(exc).__name__}: {exc}"
        return out


def _cli() -> int:
    """`python3 -m paper_trader.ml.gate_utilization [--json] [--all]`.

    Exit code mirrors the verdict severity so shell callers can gate on
    `$?`: 0 = BALANCED / WEAK_BALANCE; 1 = LOPSIDED / NEUTRAL_DOMINATED;
    2 = EMPTY_EXTREME_ARM; 3 = INSUFFICIENT_DATA / error.
    """
    import argparse

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.gate_utilization",
        description="Conviction-gate utilization audit — does the scorer's "
                    "OOS prediction distribution populate the five gate arms "
                    "in a balanced way?",
    )
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of a table.")
    p.add_argument("--all", action="store_true",
                   help="Analyze the full outcomes corpus instead of the "
                        "temporal-OOS holdout.")
    p.add_argument("--outcomes",
                   default="data/decision_outcomes.jsonl",
                   help="Path to the outcomes JSONL (default: "
                        "data/decision_outcomes.jsonl).")
    args = p.parse_args()

    rep = analyze(args.outcomes, oos_only=not args.all)

    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
    else:
        verdict = rep.get("verdict", "INSUFFICIENT_DATA")
        n = rep.get("n", 0)
        slice_ = rep.get("slice", "?")
        print(f"[gate_utilization]  n={n}  slice={slice_}  verdict={verdict}")
        if rep.get("hint"):
            print(f"  {rep['hint']}")
        print("  per-arm utilization:")
        print(f"    {'arm':<18}{'mult':>8}{'n':>10}{'pct':>10}")
        for o in rep.get("arms") or []:
            print(f"    {o['arm']:<18}{o['multiplier']:>8.2f}"
                  f"{o['n']:>10}{o['pct_of_total']*100:>9.2f}%")
        dist = rep.get("pred_distribution") or {}
        if dist:
            print("  prediction distribution:")
            print(f"    min={dist['min']:+.2f}  p10={dist['p10']:+.2f}  "
                  f"p25={dist['p25']:+.2f}  median={dist['median']:+.2f}  "
                  f"p75={dist['p75']:+.2f}  p90={dist['p90']:+.2f}  "
                  f"max={dist['max']:+.2f}")
        sb = rep.get("side_balance")
        if sb is not None:
            print(f"  side_balance (frac on tailwind side): {sb:.3f}")

    verdict = rep.get("verdict", "INSUFFICIENT_DATA")
    if verdict in ("BALANCED", "WEAK_BALANCE"):
        return 0
    if verdict in ("LOPSIDED", "NEUTRAL_DOMINATED"):
        return 1
    if verdict == "EMPTY_EXTREME_ARM":
        return 2
    return 3


if __name__ == "__main__":
    raise SystemExit(_cli())

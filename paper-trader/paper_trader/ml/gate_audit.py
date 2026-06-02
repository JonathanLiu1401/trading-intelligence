"""Conviction-gate effectiveness audit — is each gate arm's *multiplier*
economically justified by the realized return of the trades it sized?

This is a **read-only diagnostic**. It never trains, never touches
`decision_scorer.pkl`, `decision_outcomes.jsonl`, `build_features`,
`N_FEATURES`, or any trade path — so it cannot perturb the unattended
continuous loop or break pickle compatibility (same operational
discipline as `paper_trader/ml/calibration.py` / `skill_trend.py`).

**Why this is not `calibration.py`.** `calibration.py` answers a
*statistical* question — does prediction order track realized order
(rank skill), and is the magnitude biased? It buckets by 10 quantile
deciles. That can read `WELL_CALIBRATED` while the thing that actually
moves capital — the five FIXED conviction multipliers `_ml_decide`
applies at FIXED prediction thresholds — is still mis-sized. A monotone
decile curve does not tell a quant whether the `×1.3` arm's realized
edge over the `×0.6` arm justifies betting >2× as much on it. This
module answers exactly that *economic* question, bucketing by the five
real gate arms (not deciles) and reporting the realized spread that the
multiplier spread is implicitly underwriting.

**Why it matters now.** `data/scorer_skill_log.jsonl` shows the scorer
runs with `gate_active: true` every cycle (`_n_train ≥ 500`, invariant
#5) while its out-of-sample directional skill hovers at coin-flip
(`oos_dir_acc ≈ 0.5`, `oos_ic ≈ 0`). If the gate has no OOS skill, the
five multipliers are adding sizing variance with no compensating edge —
or, worse, inverting it (sizing UP the losers). This audit quantifies
that directly on the **temporal-OOS slice** (the trustworthy,
generalization-relevant view, matching `skill_trend.py` / the AGENTS.md
"read it next to oos_rmse" guidance), turning "oos_ic ≈ 0" into a
concrete "the ×1.3 arm realized X%, the ×0.6 arm realized Y%, spread
Z pp".

The gate arms below mirror `paper_trader/backtest.py::_ml_decide`
*exactly* (the same `if/elif` order and boundary operators). They are
duplicated here as module constants — `calibration.py` / `skill_trend.py`
deliberately import nothing from `backtest.py` to avoid the circular
dependency, and this module follows that precedent. Any change to the
gate in `_ml_decide` must update `GATE_ARMS` here (locked by
`tests/test_gate_audit.py::TestGateArm`).

```bash
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.gate_audit
```
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# ── Gate arms — MUST mirror backtest.py::_ml_decide's scorer block ──────────
# _ml_decide applies, in this exact if/elif order:
#   if   p < -10.0 : conviction *= 0.6
#   elif p <   0.0 : conviction *= 0.85
#   elif p >  10.0 : conviction  = min(conviction*1.3, 0.95)
#   elif p >   5.0 : conviction  = min(conviction*1.15, 0.95)
#   else (0 ≤ p ≤ 5): unchanged (×1.0)
# Ordered by multiplier so the "ascending arm" monotonicity check is
# unambiguous. `gate_arm()` reproduces the if/elif chain, not these bounds,
# so the boundary semantics (p == -10 → mild_headwind, p == 10 → mild_tailwind,
# p == 5 / p == 0 → neutral) are byte-identical to the live gate.
GATE_ARMS = [
    ("strong_headwind", 0.60),
    ("mild_headwind", 0.85),
    ("neutral", 1.00),
    ("mild_tailwind", 1.15),
    ("strong_tailwind", 1.30),
]
_ARM_ORDER = [a for a, _ in GATE_ARMS]
_ARM_MULT = dict(GATE_ARMS)

# Verdict thresholds — module-level so tests assert exact verdicts and a
# tuning change is a single reviewable edit (mirrors calibration.py).
MIN_TOTAL = 30      # need a real sample before any verdict (≈ calibration.MIN_PAIRS)
MIN_ARM_N = 5       # min trades in EACH extreme arm to compare them honestly
EDGE_TOL_PP = 1.0   # |strong_tailwind − strong_headwind| band that reads as noise


def gate_arm(pred: float) -> tuple[str, float]:
    """Return ``(arm_name, conviction_multiplier)`` for a scorer prediction,
    reproducing ``_ml_decide``'s exact if/elif chain (NOT a bounds table) so
    the boundary operators match the live gate to the bit.

    Non-finite input → ``("neutral", 1.0)``: `_ml_decide` only ever reaches
    the gate with a finite clamped `predict()` output, and the off-distribution
    abstention skips the arms entirely — treating an unusable value as the
    no-op arm is the honest, side-effect-free analogue.
    """
    try:
        p = float(pred)
    except (TypeError, ValueError):
        return ("neutral", 1.00)
    if not np.isfinite(p):
        return ("neutral", 1.00)
    if p < -10.0:
        return ("strong_headwind", 0.60)
    if p < 0.0:
        return ("mild_headwind", 0.85)
    if p > 10.0:
        return ("strong_tailwind", 1.30)
    if p > 5.0:
        return ("mild_tailwind", 1.15)
    return ("neutral", 1.00)


def gate_effectiveness_report(pairs) -> dict:
    """Bucket ``(predicted, realized)`` pairs by the gate arm the prediction
    triggers and report the per-arm realized mean.

    ``pairs`` is any iterable of ``(predicted, realized_action_aligned)``
    2-tuples — the realized value MUST already carry the SELL sign-flip the
    caller applies (``scorer_gate_audit`` does this, exactly like
    ``train_scorer`` / ``calibration.scorer_calibration``). Non-finite entries
    are dropped (one nan must not poison the report — the ``_to_float``
    hardening class).

    The verdict is driven solely by the realized spread between the two
    EXTREME arms — ``strong_tailwind`` (the gate's biggest bet, ×1.30) minus
    ``strong_headwind`` (its smallest, ×0.60). That spread is precisely what
    the 1.30/0.60 multiplier ratio is implicitly underwriting:

    | Verdict | Meaning |
    |---------|---------|
    | ``INSUFFICIENT_DATA`` | < ``MIN_TOTAL`` pairs, or either extreme arm < ``MIN_ARM_N`` |
    | ``GATE_HARMFUL`` | tailwind − headwind < −``EDGE_TOL_PP`` — the gate sizes UP the losers (inverted edge); turning it off would *improve* realized return |
    | ``GATE_INEFFECTIVE`` | \\|spread\\| ≤ ``EDGE_TOL_PP`` — multipliers are sizing noise; the gate adds variance with no compensating edge |
    | ``GATE_EFFECTIVE`` | spread > +``EDGE_TOL_PP`` — the bigger bets really did realize more; the multiplier ordering is economically justified |

    ``arm_monotone_fraction`` (fraction of adjacent arms, ordered by
    multiplier, whose realized mean is non-decreasing) is reported as an
    informational descriptor — NOT folded into the verdict, so the verdict
    stays crisply exact-value testable on the two-arm spread alone.

    Returns a JSON-safe dict. Never raises.
    """
    clean: list[tuple[str, float, float]] = []
    for p, y in pairs:
        try:
            pf = float(p)
            yf = float(y)
        except (TypeError, ValueError):
            continue
        if np.isfinite(pf) and np.isfinite(yf):
            arm, _ = gate_arm(pf)
            clean.append((arm, pf, yf))

    n = len(clean)
    base = {
        "status": "ok",
        "verdict": "INSUFFICIENT_DATA",
        "n": n,
        "arms": [],
        "strong_tailwind_minus_headwind_pp": None,
        "arm_monotone_fraction": None,
        "hint": "",
    }

    per_arm: dict[str, list[float]] = {a: [] for a in _ARM_ORDER}
    for arm, _pf, yf in clean:
        per_arm[arm].append(yf)

    arms_out = []
    for a in _ARM_ORDER:
        ys = per_arm[a]
        if ys:
            arr = np.asarray(ys, dtype=np.float64)
            arms_out.append({
                "arm": a,
                "multiplier": _ARM_MULT[a],
                "n": int(len(ys)),
                "mean_realized": round(float(arr.mean()), 4),
                "lo": round(float(arr.min()), 4),
                "hi": round(float(arr.max()), 4),
            })
        else:
            arms_out.append({
                "arm": a, "multiplier": _ARM_MULT[a], "n": 0,
                "mean_realized": None, "lo": None, "hi": None,
            })
    base["arms"] = arms_out

    # Monotonicity across arms with ≥1 sample, in multiplier order.
    present = [o for o in arms_out if o["n"] > 0]
    if len(present) >= 2:
        steps = len(present) - 1
        nondec = sum(1 for j in range(steps)
                     if present[j + 1]["mean_realized"] >= present[j]["mean_realized"])
        base["arm_monotone_fraction"] = round(nondec / steps, 4)

    head = per_arm["strong_headwind"]
    tail = per_arm["strong_tailwind"]
    if n < MIN_TOTAL or len(head) < MIN_ARM_N or len(tail) < MIN_ARM_N:
        base["hint"] = (
            f"need ≥{MIN_TOTAL} pairs and ≥{MIN_ARM_N} in BOTH extreme arms; "
            f"have n={n}, strong_headwind={len(head)}, "
            f"strong_tailwind={len(tail)}"
        )
        return base

    head_mean = float(np.mean(head))
    tail_mean = float(np.mean(tail))
    spread = tail_mean - head_mean
    base["strong_tailwind_minus_headwind_pp"] = round(spread, 4)

    if spread < -EDGE_TOL_PP:
        base["verdict"] = "GATE_HARMFUL"
        base["hint"] = (
            f"strong_tailwind realized {tail_mean:+.2f}% < strong_headwind "
            f"{head_mean:+.2f}% (spread {spread:+.2f}pp) — the ×1.30 arm "
            f"sizes UP trades that underperform the ×0.60 arm; the gate is "
            f"inverting capital allocation"
        )
    elif abs(spread) <= EDGE_TOL_PP:
        base["verdict"] = "GATE_INEFFECTIVE"
        base["hint"] = (
            f"strong_tailwind {tail_mean:+.2f}% vs strong_headwind "
            f"{head_mean:+.2f}% (spread {spread:+.2f}pp, within ±"
            f"{EDGE_TOL_PP:.1f}pp) — the 1.30/0.60 multiplier ratio is "
            f"underwriting essentially no realized edge"
        )
    else:
        base["verdict"] = "GATE_EFFECTIVE"
        base["hint"] = (
            f"strong_tailwind realized {tail_mean:+.2f}% > strong_headwind "
            f"{head_mean:+.2f}% (spread {spread:+.2f}pp) — the bigger bets "
            f"did earn the multiplier; gate ordering is economically justified"
        )
    return base


def scorer_gate_audit(scorer, records, oos_only: bool = True) -> dict:
    """Run ``gate_effectiveness_report`` for a DecisionScorer over outcome
    records (the ``data/decision_outcomes.jsonl`` row shape).

    SELL realized goodness is ``-forward_return_5d`` exactly like
    ``train_scorer`` / ``calibration.scorer_calibration`` (a drop after a SELL
    was the *right* call). The 11-kwarg ``predict`` signature mirrors
    ``validation.evaluate_scorer_oos`` / ``_oos_rank_metrics`` so this audit
    describes the SAME prediction path the live gate uses.

    ``oos_only`` (default True) restricts the audit to the temporal-OOS slice
    via ``validation.split_outcomes_temporal`` — the trustworthy,
    generalization-relevant view (the in-sample slice is optimistic because the
    scorer trained on most of it; AGENTS.md: "always read it next to
    oos_rmse"). Falls back to the full set only if the split is unavailable.
    The chosen slice is reported in ``slice`` (``"oos"`` / ``"all"``).

    Never raises — any fault degrades to ``INSUFFICIENT_DATA``.
    """
    try:
        recs = list(records or [])
    except Exception:
        recs = []

    slice_name = "all"
    if oos_only:
        try:
            from paper_trader.validation import split_outcomes_temporal
            _, oos = split_outcomes_temporal(recs, oos_fraction=0.2)
            if oos:
                recs = oos
                slice_name = "oos"
        except Exception:
            slice_name = "all"

    pairs: list[tuple[float, float]] = []
    for r in recs:
        try:
            # Forward the 3 enhanced MACD features (pass #36 OOS feature
            # parity). Without them, this analyzer predicts on a degraded
            # vector vs the live ``_ml_decide`` gate — ``oos_parity_audit``
            # measures BIAS_LARGE (delta_rank_ic=+0.11) on the deployed
            # pickle, biasing the per-arm verdict.
            pred = scorer.predict(
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
                ema200_above=r.get("ema200_above"),
                hist_cross_up=r.get("hist_cross_up"),
                macd_below_zero_cross=r.get("macd_below_zero_cross"),
            )
        except Exception:
            continue
        y = r.get("forward_return_5d")
        if y is None:
            continue
        action = str(r.get("action") or "BUY").upper()
        pairs.append((pred, -y if action == "SELL" else y))

    rep = gate_effectiveness_report(pairs)
    rep["slice"] = slice_name
    rep["n_records_considered"] = len(recs)
    return rep


def analyze(outcomes_path: Path | str, oos_only: bool = True) -> dict:
    """Load the live pickled scorer + outcomes file and return the audit.
    Read-only; never raises."""
    from .decision_scorer import DecisionScorer

    out: dict = {"status": "error", "verdict": "INSUFFICIENT_DATA",
                 "n": 0, "arms": [], "hint": ""}
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
                records.append(json.loads(ln))
            except Exception:
                continue
        scorer = DecisionScorer()
        if not getattr(scorer, "is_trained", False):
            out["hint"] = "scorer not trained — nothing to audit"
            return out
        rep = scorer_gate_audit(scorer, records, oos_only=oos_only)
        rep.setdefault("n_train", getattr(scorer, "n_train", None))
        return rep
    except Exception as e:
        out["hint"] = f"audit failed: {type(e).__name__}: {e}"
        return out


def _cli() -> int:
    """`python3 -m paper_trader.ml.gate_audit` — gate-effectiveness audit of
    the live pickled scorer against the accumulated outcomes tail. Read-only."""
    root = Path(__file__).resolve().parent.parent.parent
    rep = analyze(root / "data" / "decision_outcomes.jsonl", oos_only=True)
    print(f"VERDICT: {rep['verdict']}  ({rep.get('hint', '')})")
    print(f"  slice={rep.get('slice')}  n={rep.get('n')}  "
          f"n_train={rep.get('n_train')}  "
          f"tail−head={rep.get('strong_tailwind_minus_headwind_pp')}pp  "
          f"arm_monotone={rep.get('arm_monotone_fraction')}")
    for a in rep.get("arms", []):
        mr = a["mean_realized"]
        mr_s = f"{mr:+7.2f}%" if mr is not None else "    n/a"
        print(f"  {a['arm']:<16} ×{a['multiplier']:.2f}  "
              f"n={a['n']:<5} mean_realized={mr_s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())

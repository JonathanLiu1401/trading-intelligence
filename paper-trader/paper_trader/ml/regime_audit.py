"""Regime-conditional scorer-skill audit — is the documented near-zero
out-of-sample skill UNIFORM across market regimes, or does the scorer (and
therefore the conviction gate) carry real edge in one specific regime that
the aggregate verdict is hiding?

This is a **read-only diagnostic**. It never trains, never touches
`decision_scorer.pkl`, `decision_outcomes.jsonl`, `build_features`,
`N_FEATURES`, or any trade path — so it cannot perturb the unattended
continuous loop or break pickle compatibility (same operational discipline
as `paper_trader/ml/calibration.py` / `gate_audit.py` / `skill_trend.py`).

**Why this is not `calibration.py` / `gate_audit.py` / `skill_trend.py` /
`feature_importance.py`.** Those answer, respectively: *is pred monotone
with realized* (statistical, 10 deciles), *do the 5 fixed multipliers buy
realized edge* (economic, 5 gate arms), *is oos_rmse better than a mean
predictor over cycles* (error-trend), and *which of the 17 features carries
the prediction* (attribution). **None of them conditions on the market
regime.** That is the natural next question once
`calibration`=WELL_CALIBRATED-in-sample, `gate_audit`=GATE_INEFFECTIVE,
`skill_trend`=NEGATIVE_OOS_SKILL, `feature_importance`=SIGNAL_GROUNDED have
established the aggregate picture: a scorer that reads real RSI/Bollinger
mean-reversion signal but shows ≈0 OOS rank skill *on average* may still be
skilled in (say) bull and inverted in sideways — in which case the
aggregate "no edge" verdict is a regime-mix artifact and the actionable
finding is regime-conditional, not "the model is blind".

The regime label is decoded from the `regime_mult` field every
`decision_outcomes.jsonl` row already carries (written by
`run_continuous_backtests._compute_decision_outcomes`, mirroring
`backtest.py::_ml_decide`): `0.3 → bear`, `0.6 → sideways`,
`1.0 → bull_or_unknown`. The `bull_or_unknown` label is deliberate and
honest: `backtest.py::_market_regime` collapses a true bull AND an
insufficient-SPY-history "unknown" to the SAME `1.0` multiplier, so the
two are *not separable from the stored feature alone* — claiming a clean
"bull" bucket would be a fabricated distinction (the same honesty
discipline as `feature_importance`'s `degenerate` flag).

```bash
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.regime_audit
cd /home/zeph/paper-trader && python3 -m pytest tests/test_regime_audit.py -v
```
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# regime_mult → regime label. MUST mirror backtest.py::_ml_decide /
# run_continuous_backtests._compute_decision_outcomes (bull=1.0, sideways=0.6,
# bear=0.3, unknown=1.0). Keyed by the rounded float so a JSON-serialized
# 0.6000000001 still maps. `1.0` is honestly labeled bull_or_unknown — the
# stored feature cannot separate a real bull from an unknown (both 1.0).
REGIME_FROM_MULT: dict[float, str] = {
    0.30: "bear",
    0.60: "sideways",
    1.00: "bull_or_unknown",
}
_REGIME_ORDER = ["bear", "sideways", "bull_or_unknown"]

# Verdict thresholds — module-level so tests assert exact verdicts and a
# tuning change is a single reviewable edit (mirrors calibration.py /
# gate_audit.py / skill_trend.py constants-at-module-scope convention).
MIN_TOTAL = 30       # need a real sample before any verdict (≈ calibration.MIN_PAIRS)
MIN_REGIME_N = 20    # min pairs in a regime before its skill metric is trustworthy
IC_MIN = 0.05        # rank-skill bar — mirrors skill_trend.IC_MIN exactly
                     # (the gate acts on sign; skill_trend uses this same bar
                     #  for "gate may still carry value")


def _regime_of(regime_mult) -> str | None:
    """Decode a stored ``regime_mult`` into a regime label, or ``None`` when
    it is missing / non-finite / not one of the three known multipliers.

    Rounds to 2 decimals before the dict lookup so a JSON float like
    ``0.6000000000000001`` still resolves. An unmapped value is NOT
    silently bucketed — it is dropped and counted in ``dropped_unmapped``
    so a future fourth multiplier can never masquerade as one of these
    three (the ``_to_float`` / ``degenerate`` honesty class)."""
    try:
        v = float(regime_mult)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return REGIME_FROM_MULT.get(round(v, 2))


def _spearman_safe(a: np.ndarray, b: np.ndarray) -> float:
    """Tie-aware Spearman, reusing ``calibration._spearman`` so this OOS
    metric and the in-sample calibration diagnostic can never drift (single
    source of truth — the AGENTS.md invariant-#10 spirit). Tie-awareness is
    load-bearing: the scorer clamps to ±PRED_CLAMP_PCT, so off-distribution
    predictions tie at exactly ±50 and a naïve argsort would fabricate rank
    skill there. 0.0 on any fault / <2 points."""
    try:
        from paper_trader.ml.calibration import _spearman
        if len(a) < 2:
            return 0.0
        ic = _spearman(np.asarray(a, dtype=np.float64),
                       np.asarray(b, dtype=np.float64))
        return float(ic) if ic == ic else 0.0  # NaN → 0.0
    except Exception:
        return 0.0


def regime_skill_report(triples) -> dict:
    """Bucket ``(regime, predicted, realized_action_aligned)`` triples by
    regime and report per-regime rank skill + directional accuracy + the
    gate's extreme-arm realized spread.

    ``triples`` is any iterable of 3-tuples; the realized value MUST already
    carry the SELL sign-flip the caller applies (``scorer_regime_audit`` does
    this, exactly like ``train_scorer`` / ``calibration.scorer_calibration`` /
    ``gate_audit.scorer_gate_audit``). Non-finite pred/realized and
    unmappable regimes are dropped (one nan must not poison the report — the
    ``_to_float`` hardening class); the unmapped count is surfaced honestly.

    Per regime with ≥ ``MIN_REGIME_N`` pairs the regime is classified
    **skilled** iff ``rank_ic ≥ IC_MIN`` (the gate acts on sign, so rank
    skill — not RMSE — is the gate-relevant criterion; the bar mirrors
    ``skill_trend.IC_MIN`` exactly). The verdict compares regimes:

    | Verdict | Meaning |
    |---------|---------|
    | ``INSUFFICIENT_DATA`` | < ``MIN_TOTAL`` clean pairs total |
    | ``SINGLE_REGIME_ONLY`` | only one regime clears ``MIN_REGIME_N`` — the OOS slice is regime-degenerate, regimes cannot be compared (honest limitation, not a skill claim) |
    | ``REGIME_UNIFORM_NULL`` | ≥2 measurable regimes, NONE skilled — the documented near-zero OOS skill holds across every regime; the aggregate verdict is not a regime-mix artifact |
    | ``REGIME_DEPENDENT_EDGE`` | ≥1 measurable regime skilled AND ≥1 not — the aggregate "no edge" verdict HIDES regime structure; the gate may carry real edge in a specific regime (the actionable finding) |
    | ``REGIME_UNIFORM_EDGE`` | every measurable regime skilled — the scorer generalizes across regimes |

    Returns a JSON-safe dict. Never raises.
    """
    clean: list[tuple[str, float, float]] = []
    dropped_unmapped = 0
    for rec in triples:
        try:
            reg, p, y = rec
        except Exception:
            continue
        label = _regime_of(reg)
        if label is None:
            dropped_unmapped += 1
            continue
        try:
            pf = float(p)
            yf = float(y)
        except (TypeError, ValueError):
            continue
        if np.isfinite(pf) and np.isfinite(yf):
            clean.append((label, pf, yf))

    n = len(clean)
    base: dict = {
        "status": "ok",
        "verdict": "INSUFFICIENT_DATA",
        "n": n,
        "dropped_unmapped": dropped_unmapped,
        "regimes": [],
        "n_measurable_regimes": 0,
        "n_skilled_regimes": 0,
        "hint": "",
    }

    per: dict[str, list[tuple[float, float]]] = {r: [] for r in _REGIME_ORDER}
    for label, pf, yf in clean:
        per[label].append((pf, yf))

    # Lazy import — same circular-dependency-avoidance precedent the sibling
    # ml/ modules follow (gate_arm lives in gate_audit, not backtest.py).
    try:
        from paper_trader.ml.gate_audit import gate_arm
    except Exception:
        gate_arm = None  # gate-spread degrades to None, rest of report stands

    regimes_out = []
    measurable = []
    for r in _REGIME_ORDER:
        pairs = per[r]
        nr = len(pairs)
        if nr == 0:
            regimes_out.append({
                "regime": r, "n": 0, "mean_pred": None,
                "mean_realized": None, "rank_ic": None, "dir_acc": None,
                "gate_tail_minus_head_pp": None, "measurable": False,
                "skilled": None,
            })
            continue
        preds = np.array([p for p, _ in pairs], dtype=np.float64)
        reals = np.array([y for _, y in pairs], dtype=np.float64)
        rank_ic = round(_spearman_safe(preds, reals), 4)

        # Directional accuracy — exclude exact-zero pred/realized (no
        # directional truth), mirroring _oos_rank_metrics in the loop.
        dir_pairs = [(p, y) for p, y in pairs if p != 0.0 and y != 0.0]
        dir_acc = (round(sum(1 for p, y in dir_pairs
                             if (p > 0) == (y > 0)) / len(dir_pairs), 4)
                   if dir_pairs else None)

        # Gate extreme-arm realized spread within this regime (the same
        # economic quantity gate_audit reports, conditioned on regime).
        gate_spread = None
        if gate_arm is not None:
            head = [y for p, y in pairs if gate_arm(p)[0] == "strong_headwind"]
            tail = [y for p, y in pairs if gate_arm(p)[0] == "strong_tailwind"]
            if len(head) >= 5 and len(tail) >= 5:
                gate_spread = round(float(np.mean(tail) - np.mean(head)), 4)

        is_measurable = nr >= MIN_REGIME_N
        skilled = (rank_ic >= IC_MIN) if is_measurable else None
        if is_measurable:
            measurable.append((r, bool(skilled)))
        regimes_out.append({
            "regime": r,
            "n": nr,
            "mean_pred": round(float(preds.mean()), 4),
            "mean_realized": round(float(reals.mean()), 4),
            "rank_ic": rank_ic,
            "dir_acc": dir_acc,
            "gate_tail_minus_head_pp": gate_spread,
            "measurable": is_measurable,
            "skilled": skilled,
        })

    base["regimes"] = regimes_out
    base["n_measurable_regimes"] = len(measurable)
    base["n_skilled_regimes"] = sum(1 for _, s in measurable if s)

    if n < MIN_TOTAL:
        base["hint"] = (f"need ≥{MIN_TOTAL} clean pairs, have {n} "
                        f"({dropped_unmapped} dropped as unmapped regime_mult)")
        return base

    n_meas = len(measurable)
    n_skill = base["n_skilled_regimes"]
    if n_meas <= 1:
        base["verdict"] = "SINGLE_REGIME_ONLY"
        only = measurable[0][0] if measurable else "none"
        base["hint"] = (
            f"only {n_meas} regime clears MIN_REGIME_N={MIN_REGIME_N} "
            f"(measurable: {only}) — the OOS slice is regime-degenerate; "
            f"regimes cannot be compared (not a skill claim)"
        )
    elif n_skill == 0:
        base["verdict"] = "REGIME_UNIFORM_NULL"
        base["hint"] = (
            f"{n_meas} measurable regimes, none with rank_ic ≥ {IC_MIN} — "
            f"the near-zero OOS skill holds across EVERY regime; the "
            f"aggregate verdict is not a regime-mix artifact"
        )
    elif n_skill == n_meas:
        base["verdict"] = "REGIME_UNIFORM_EDGE"
        base["hint"] = (
            f"all {n_meas} measurable regimes have rank_ic ≥ {IC_MIN} — "
            f"the scorer's rank skill generalizes across regimes"
        )
    else:
        skilled_names = [r for r, s in measurable if s]
        null_names = [r for r, s in measurable if not s]
        base["verdict"] = "REGIME_DEPENDENT_EDGE"
        base["hint"] = (
            f"rank skill is regime-dependent: skilled in {skilled_names}, "
            f"null in {null_names} — the aggregate 'no edge' verdict HIDES "
            f"regime structure; the gate may carry edge in {skilled_names}"
        )
    return base


def scorer_regime_audit(scorer, records, oos_only: bool = True) -> dict:
    """Run ``regime_skill_report`` for a DecisionScorer over outcome records
    (the ``data/decision_outcomes.jsonl`` row shape).

    SELL realized goodness is ``-forward_return_5d`` exactly like
    ``train_scorer`` / ``calibration.scorer_calibration`` /
    ``gate_audit.scorer_gate_audit`` (a drop after a SELL was the *right*
    call). The 11-kwarg ``predict`` signature mirrors
    ``validation.evaluate_scorer_oos`` / ``_oos_rank_metrics`` so this audit
    describes the SAME prediction path the live gate uses.

    ``oos_only`` (default True) restricts to the temporal-OOS slice via
    ``validation.split_outcomes_temporal`` — the trustworthy,
    generalization-relevant view, and the EXACT split
    ``_train_decision_scorer`` uses for ``oos_rmse``/``oos_ic`` so this
    regime breakdown and the ledger's scalar OOS metrics describe the *same*
    holdout. Falls back to the full set only if the split is unavailable.
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

    triples: list[tuple[str, float, float]] = []
    for r in recs:
        try:
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
        triples.append((r.get("regime_mult", 1.0), pred,
                        -y if action == "SELL" else y))

    rep = regime_skill_report(triples)
    rep["slice"] = slice_name
    rep["n_records_considered"] = len(recs)
    return rep


def analyze(outcomes_path: Path | str, oos_only: bool = True) -> dict:
    """Load the live pickled scorer + outcomes file and return the audit.
    Read-only; never raises."""
    from .decision_scorer import DecisionScorer

    out: dict = {"status": "error", "verdict": "INSUFFICIENT_DATA",
                 "n": 0, "regimes": [], "hint": ""}
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
        rep = scorer_regime_audit(scorer, records, oos_only=oos_only)
        rep.setdefault("n_train", getattr(scorer, "n_train", None))
        return rep
    except Exception as e:
        out["hint"] = f"audit failed: {type(e).__name__}: {e}"
        return out


def _cli() -> int:
    """`python3 -m paper_trader.ml.regime_audit [--all]` — regime-conditional
    scorer-skill audit of the live pickled scorer against the accumulated
    outcomes tail. Read-only.

    Exit code 2 on ``REGIME_DEPENDENT_EDGE`` (the operator-actionable
    branch — the aggregate verdict is hiding regime structure worth a
    model-dynamics look), mirroring the cron-branchable exit-2 convention of
    ``label_audit`` / ``persona_skill`` / ``feature_importance``.
    """
    import sys
    args = sys.argv[1:]
    oos_only = "--all" not in args

    root = Path(__file__).resolve().parent.parent.parent
    rep = analyze(root / "data" / "decision_outcomes.jsonl", oos_only=oos_only)
    print(f"VERDICT: {rep['verdict']}  ({rep.get('hint', '')})")
    print(f"  slice={rep.get('slice')}  n={rep.get('n')}  "
          f"n_train={rep.get('n_train')}  "
          f"measurable={rep.get('n_measurable_regimes')}  "
          f"skilled={rep.get('n_skilled_regimes')}  "
          f"dropped_unmapped={rep.get('dropped_unmapped')}")
    for r in rep.get("regimes", []):
        ic = r["rank_ic"]
        da = r["dir_acc"]
        mr = r["mean_realized"]
        gs = r["gate_tail_minus_head_pp"]
        ic_s = f"{ic:+.3f}" if ic is not None else "  n/a"
        da_s = f"{da:.3f}" if da is not None else " n/a"
        mr_s = f"{mr:+7.2f}%" if mr is not None else "    n/a"
        gs_s = f"{gs:+.2f}pp" if gs is not None else "  n/a"
        flag = ("skilled" if r["skilled"] else "null") if r["measurable"] \
            else ("thin" if r["n"] else "absent")
        print(f"  {r['regime']:<16} n={r['n']:<5} rank_ic={ic_s} "
              f"dir_acc={da_s} mean_realized={mr_s} "
              f"gate_tail−head={gs_s}  [{flag}]")
    return 2 if rep.get("verdict") == "REGIME_DEPENDENT_EDGE" else 0


if __name__ == "__main__":
    raise SystemExit(_cli())

"""OOS-inference feature-parity audit for the deployed DecisionScorer pickle.

Quantifies the realized bias the pass #36 fix removed: how much did the
per-cycle skill ledger's OOS metrics diverge from what the live
``_ml_decide`` gate would actually predict on the same rows? The 3
enhanced MACD features (``ema200_above`` / ``hist_cross_up`` /
``macd_below_zero_cross``) are captured by ``_compute_decision_outcomes``
and forwarded by the live gate, but until pass #36 every OOS-side
``predict_with_meta`` / ``predict`` call stripped those 3 kwargs and let
``build_features`` default them to ``None → 0.0``. The deployed pickle's
first-layer ``mean|w|`` for these slots is non-zero (≈0.45 / 0.26 / 0.24,
the 3rd / 15th / 16th most important features), so the OOS rank-IC and
RMSE the skill ledger reported were systematically biased away from the
true production behaviour.

This audit answers the natural follow-up question a quant would ask:
*how big was the bias — 0.05pp on OOS RMSE, or 2pp?* It does so by
running each outcome row through the deployed scorer TWICE: once with
the 3 enhanced MACD kwargs forwarded (the corrected, gate-aligned path)
and once with them defaulted to ``None`` (the pre-fix, biased path). The
delta between the two paths' aggregate metrics IS the realized bias.

Operationally identical to ``feature_importance`` / ``dead_feature_audit``:
read-only, no train, no pickle write, safe to run against the live
unattended loop. Never raises; every failure path yields an honest
sentinel envelope so a wired ledger consumer can persist the fault
without breaking the cycle.

CLI:

```bash
cd /home/zeph/trading-intelligence/paper-trader && \\
    python3 -m paper_trader.ml.oos_parity_audit            # human-readable
cd /home/zeph/trading-intelligence/paper-trader && \\
    python3 -m paper_trader.ml.oos_parity_audit --json     # machine-readable
```

Exit code ``0`` on ``OK / BIAS_SMALL / NO_PARITY_FEATURES / NOT_TRAINED``,
``1`` otherwise (``BIAS_LARGE`` / ``BIAS_MODERATE`` / error) — so the
audit is shell-gateable like the sibling diagnostics.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from paper_trader.ml.decision_scorer import (
    DecisionScorer,
    PRED_CLAMP_PCT,
    _to_float,
)

# Default outcome corpus the continuous loop accumulates. The audit reads
# from disk (read-only) and never writes — same operational discipline
# the sibling ``feature_importance`` CLI follows.
DEFAULT_OUTCOMES_PATH = Path(__file__).resolve().parent.parent.parent / (
    "data/decision_outcomes.jsonl"
)

# Bias-magnitude verdict bands. The threshold has to compare against
# something economically meaningful — OOS RMSE on the live corpus sits
# around 8–10pp (verified live: scorer_skill_log oos_rmse=8.83) and a
# 5d return target has σ ≈ 11.7pp, so a 0.10pp shift is well below noise,
# 0.50pp is "operator should look", and 1.00pp+ means the skill ledger
# was being read meaningfully wrong.
BIAS_RMSE_PP_SMALL = 0.10
BIAS_RMSE_PP_MODERATE = 0.50
# rank-IC sits in the [-0.1, +0.1] band on the live corpus, so the bias
# magnitudes that matter here are much smaller — 0.005 is "look",
# 0.02 is "real".
BIAS_RANK_IC_SMALL = 0.005
BIAS_RANK_IC_MODERATE = 0.02


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    """Tie-aware Spearman on two equal-length 1-D arrays. None on n<2,
    constant input, or any non-finite result. Mirrors
    ``ml.calibration._spearman``'s contract so audits cross-compare with
    the per-cycle skill ledger. Never raises."""
    try:
        if x.size != y.size or x.size < 2:
            return None
        # A constant input produces a NaN spearman + a ConstantInputWarning.
        # We already degrade to None on a non-finite result; suppress the
        # warning so test output stays clean (same idiom the wider audit
        # surface uses around scipy's stricter inputs).
        import warnings
        from scipy import stats as _stats

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho, _p = _stats.spearmanr(x, y)
        if rho is None or not np.isfinite(rho):
            return None
        return float(rho)
    except ImportError:
        # numpy-only fallback. Rank with argsort(argsort) then Pearson.
        try:
            rx = np.argsort(np.argsort(x))
            ry = np.argsort(np.argsort(y))
            if rx.std() == 0 or ry.std() == 0:
                return None
            r = float(np.corrcoef(rx, ry)[0, 1])
            return r if np.isfinite(r) else None
        except Exception:
            return None
    except Exception:
        return None


def _rmse(preds: np.ndarray, actuals: np.ndarray) -> float | None:
    """RMSE with the same clamp+drop discipline ``evaluate_scorer_oos``
    uses (clamp realized labels to ±PRED_CLAMP_PCT, drop NaN). None on
    empty input or all-NaN."""
    try:
        if preds.size == 0:
            return None
        mask = np.isfinite(preds) & np.isfinite(actuals)
        if not mask.any():
            return None
        p = preds[mask]
        a = np.clip(actuals[mask], -PRED_CLAMP_PCT, PRED_CLAMP_PCT)
        return float(np.sqrt(np.mean((p - a) ** 2)))
    except Exception:
        return None


def _row_has_enhanced_macd(r: dict) -> bool:
    """True iff the row carries at least one non-None enhanced MACD field.
    Rows where every enhanced MACD value is None / missing produce the
    SAME prediction on both paths (the corrected path forwards None which
    build_features defaults to 0.0 — same as the degraded path's implicit
    default). The audit's bias signal lives entirely in the rows where at
    least one of the 3 values is True/False.
    """
    for k in ("ema200_above", "hist_cross_up", "macd_below_zero_cross"):
        v = r.get(k)
        if v is not None:
            return True
    return False


def _predict_pair(scorer: DecisionScorer, r: dict) -> tuple[float | None,
                                                            float | None]:
    """Run one outcome row through the scorer TWICE: with enhanced MACD
    forwarded (parity path, gate-aligned) and with them defaulted to None
    (degraded path, pre-pass-#36 OOS state). Returns ``(p_parity,
    p_degraded)`` — each is a clamped prediction in % or None if the
    scorer returned a failed/non-finite result. Mirrors
    ``predict_with_meta``'s ``failed=True`` drop discipline."""
    base = dict(
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
    parity_kwargs = dict(
        base,
        ema200_above=r.get("ema200_above"),
        hist_cross_up=r.get("hist_cross_up"),
        macd_below_zero_cross=r.get("macd_below_zero_cross"),
    )
    try:
        _pwm = getattr(scorer, "predict_with_meta", None)
        if callable(_pwm):
            mp = _pwm(**parity_kwargs)
            md = _pwm(**base)
            p_parity = (None if mp.get("failed")
                        else float(mp.get("pred", 0.0)))
            p_degraded = (None if md.get("failed")
                          else float(md.get("pred", 0.0)))
        else:
            p_parity = float(scorer.predict(**parity_kwargs))
            p_degraded = float(scorer.predict(**base))
    except Exception:
        return None, None
    # Defensive — predict_with_meta should never emit non-finite past its
    # own internal guard, but if a test fake / future scorer does, drop.
    if p_parity is not None and not math.isfinite(p_parity):
        p_parity = None
    if p_degraded is not None and not math.isfinite(p_degraded):
        p_degraded = None
    return p_parity, p_degraded


def audit_oos_parity(
    records: list[dict],
    scorer: DecisionScorer | None = None,
) -> dict[str, Any]:
    """Quantify the realized bias the pass #36 OOS-inference parity fix
    removes by comparing the corrected path vs the pre-fix degraded path
    on the same scorer + outcome corpus.

    Returns a JSON-safe dict:

    * ``verdict`` — ladder:
      - ``NOT_TRAINED`` — pickle absent or not loaded
      - ``NO_DATA`` — no input records / every row had a failed predict
      - ``NO_PARITY_FEATURES`` — no row carried a non-None enhanced MACD
        field, so both paths produce identical predictions by
        construction (the corrected path forwards None which
        build_features defaults to 0.0 — same as the degraded path)
      - ``BIAS_LARGE`` — |delta_rmse| ≥ MODERATE band OR |delta_rank_ic|
        ≥ MODERATE
      - ``BIAS_MODERATE`` — delta is between SMALL and MODERATE
      - ``BIAS_SMALL`` — delta is below the SMALL band (effectively
        negligible, the corrected path matches the degraded path on
        the corpus presented)
      - ``OK`` — alias for BIAS_SMALL preserved for shell-script
        compatibility with sibling audits

    * ``n_records`` — total rows examined
    * ``n_with_any_enhanced`` — rows where at least one enhanced MACD
      value was non-None (the rows that actually drive divergence)
    * ``rmse_parity`` / ``rmse_degraded`` — aggregate RMSE on the
      corpus under each prediction path
    * ``delta_rmse_pp`` — ``rmse_parity - rmse_degraded`` (signed pp).
      A negative value means the corrected path has LOWER error (the
      live gate's predictions are closer to realized outcomes than the
      pre-fix OOS evaluation would have suggested) — the expected
      direction if the enhanced MACD weights carry real signal.
    * ``rank_ic_parity`` / ``rank_ic_degraded`` — aggregate Spearman
      rank-IC of prediction vs realized 5d return (action-aligned)
    * ``delta_rank_ic`` — ``rank_ic_parity - rank_ic_degraded``
    * ``mean_abs_pred_diff_pp`` / ``max_abs_pred_diff_pp`` — per-row
      prediction divergence, in % return units
    * ``n_predict_failures`` — rows dropped because either path
      returned a failed prediction (the same drop discipline
      ``_oos_rank_metrics`` uses)
    * ``eps_rmse`` / ``eps_rank_ic`` — the SMALL-band thresholds the
      verdict was computed against, surfaced for a reading quant to
      verify which side of the band the deployed model lands on

    Pure, total, never raises — degrades to an honest envelope on any
    fault (the same discipline ``dead_feature_audit`` /
    ``feature_importance`` follow)."""
    out: dict[str, Any] = {
        "verdict": "NO_DATA",
        "method": "predict_with_meta_pair",
        "n_records": 0,
        "n_with_any_enhanced": 0,
        "n_predict_failures": 0,
        "rmse_parity": None,
        "rmse_degraded": None,
        "delta_rmse_pp": None,
        "rank_ic_parity": None,
        "rank_ic_degraded": None,
        "delta_rank_ic": None,
        "mean_abs_pred_diff_pp": None,
        "max_abs_pred_diff_pp": None,
        "eps_rmse": BIAS_RMSE_PP_SMALL,
        "eps_rank_ic": BIAS_RANK_IC_SMALL,
    }
    try:
        sc = scorer if scorer is not None else DecisionScorer()
        if not getattr(sc, "is_trained", False):
            out["verdict"] = "NOT_TRAINED"
            out["n_train"] = int(getattr(sc, "n_train", 0))
            return out
        out["n_train"] = int(getattr(sc, "n_train", 0))

        if not records:
            out["verdict"] = "NO_DATA"
            return out

        out["n_records"] = len(records)

        preds_par: list[float] = []
        preds_deg: list[float] = []
        actuals: list[float] = []
        diffs: list[float] = []
        n_with_any_enhanced = 0
        n_predict_failures = 0
        for r in records:
            if _row_has_enhanced_macd(r):
                n_with_any_enhanced += 1
            actual_raw = _to_float(r.get("forward_return_5d"), float("nan"))
            if not (isinstance(actual_raw, float) and math.isfinite(actual_raw)):
                # Skip rows with no realized target — same drop discipline
                # _oos_rank_metrics uses. They contribute neither to
                # RMSE nor to rank-IC, on either path.
                continue
            is_sell = str(r.get("action") or "BUY").upper() == "SELL"
            a = -actual_raw if is_sell else actual_raw
            p_par, p_deg = _predict_pair(sc, r)
            if p_par is None or p_deg is None:
                n_predict_failures += 1
                continue
            preds_par.append(p_par)
            preds_deg.append(p_deg)
            actuals.append(a)
            diffs.append(abs(p_par - p_deg))

        out["n_with_any_enhanced"] = n_with_any_enhanced
        out["n_predict_failures"] = n_predict_failures

        if not preds_par:
            # Every row's prediction failed — no signal to compare
            out["verdict"] = "NO_DATA"
            return out

        # If no row in the corpus carried an enhanced MACD value, both
        # paths receive identical inputs and must produce identical
        # outputs — surface that as a distinct verdict so a reading
        # operator can tell "audit ran cleanly but corpus pre-dates the
        # pass #35 capture" from "audit ran and bias is genuinely zero".
        if n_with_any_enhanced == 0:
            out["verdict"] = "NO_PARITY_FEATURES"
            return out

        p_par_arr = np.asarray(preds_par, dtype=np.float64)
        p_deg_arr = np.asarray(preds_deg, dtype=np.float64)
        a_arr = np.asarray(actuals, dtype=np.float64)
        d_arr = np.asarray(diffs, dtype=np.float64)

        rmse_par = _rmse(p_par_arr, a_arr)
        rmse_deg = _rmse(p_deg_arr, a_arr)
        ic_par = _spearman(p_par_arr, a_arr)
        ic_deg = _spearman(p_deg_arr, a_arr)

        out["rmse_parity"] = round(rmse_par, 4) if rmse_par is not None else None
        out["rmse_degraded"] = (round(rmse_deg, 4)
                                if rmse_deg is not None else None)
        if rmse_par is not None and rmse_deg is not None:
            out["delta_rmse_pp"] = round(rmse_par - rmse_deg, 4)
        out["rank_ic_parity"] = (round(ic_par, 4)
                                 if ic_par is not None else None)
        out["rank_ic_degraded"] = (round(ic_deg, 4)
                                   if ic_deg is not None else None)
        if ic_par is not None and ic_deg is not None:
            out["delta_rank_ic"] = round(ic_par - ic_deg, 4)
        out["mean_abs_pred_diff_pp"] = round(float(d_arr.mean()), 4)
        out["max_abs_pred_diff_pp"] = round(float(d_arr.max()), 4)

        # Verdict from the larger-magnitude of the two delta signals.
        # |delta_rmse_pp| and |delta_rank_ic| are checked against their
        # own bands; we promote to the tier of the more extreme one
        # rather than averaging — a small RMSE shift but a big rank
        # shift (or vice versa) is still a meaningful state.
        verdict = "BIAS_SMALL"
        d_rmse = out["delta_rmse_pp"]
        d_ic = out["delta_rank_ic"]
        if d_rmse is not None:
            if abs(d_rmse) >= BIAS_RMSE_PP_MODERATE:
                verdict = "BIAS_LARGE"
            elif abs(d_rmse) >= BIAS_RMSE_PP_SMALL:
                verdict = "BIAS_MODERATE"
        if d_ic is not None:
            if abs(d_ic) >= BIAS_RANK_IC_MODERATE:
                verdict = "BIAS_LARGE"
            elif (abs(d_ic) >= BIAS_RANK_IC_SMALL
                  and verdict == "BIAS_SMALL"):
                verdict = "BIAS_MODERATE"
        out["verdict"] = verdict
        return out
    except Exception as e:
        out["verdict"] = "ERROR"
        out["error"] = str(e)
        return out


def _load_records(path: Path, tail: int | None = None) -> list[dict]:
    """Load outcome JSONL from ``path``, keeping the last ``tail`` rows
    when set. Mirrors ``run_continuous_backtests``'s tail-trim idiom so
    the audit reads exactly the same corpus the continuous loop's
    per-cycle retrain consumes. Never raises — a malformed line is
    dropped silently (the same idiom the loop uses)."""
    if not path.exists():
        return []
    out: list[dict] = []
    try:
        with path.open("r") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    out.append(json.loads(ln))
                except Exception:
                    continue
    except Exception:
        return []
    if tail is not None and tail > 0:
        out = out[-tail:]
    return out


def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.oos_parity_audit",
        description=(
            "Quantify the bias the pass #36 OOS-feature-parity fix removes "
            "by predicting on every outcome row TWICE: once with the 3 "
            "enhanced MACD kwargs forwarded (gate-aligned), once with them "
            "defaulted to None (pre-fix OOS state). Reports aggregate "
            "delta_rmse, delta_rank_ic, and per-row max divergence. "
            "Read-only — never trains or writes."
        ),
    )
    p.add_argument(
        "--outcomes", default=str(DEFAULT_OUTCOMES_PATH),
        help=("Path to a decision_outcomes JSONL. Default: live "
              "data/decision_outcomes.jsonl."),
    )
    p.add_argument(
        "--tail", type=int, default=5000,
        help=("Read the last N lines only (default 5000 — matches "
              "MAX_OUTCOMES_FOR_TRAINING). Set <=0 to read the entire file."),
    )
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of a table.")
    return p


def main(argv: list[str] | None = None) -> int:
    """Run the audit against the deployed pickle + outcome corpus. Returns
    0 when verdict is ``OK`` / ``BIAS_SMALL`` / ``NO_PARITY_FEATURES`` /
    ``NOT_TRAINED``, 1 otherwise (BIAS_MODERATE / BIAS_LARGE / ERROR /
    NO_DATA) — shell-gateable like ``dead_feature_audit``'s CLI."""
    import sys

    args = _build_arg_parser().parse_args(
        sys.argv[1:] if argv is None else argv)

    tail = args.tail if args.tail and args.tail > 0 else None
    records = _load_records(Path(args.outcomes), tail=tail)
    audit = audit_oos_parity(records)

    if args.json:
        print(json.dumps(audit, indent=2, sort_keys=True))
    else:
        v = audit.get("verdict")
        print(f"[oos_parity_audit] verdict={v}  "
              f"n_records={audit.get('n_records', 0)}  "
              f"n_with_enhanced={audit.get('n_with_any_enhanced', 0)}  "
              f"n_train={audit.get('n_train', 0)}")
        d_rmse = audit.get("delta_rmse_pp")
        d_ic = audit.get("delta_rank_ic")
        rmse_par = audit.get("rmse_parity")
        rmse_deg = audit.get("rmse_degraded")
        ic_par = audit.get("rank_ic_parity")
        ic_deg = audit.get("rank_ic_degraded")
        if rmse_par is not None and rmse_deg is not None:
            print(f"  RMSE        parity={rmse_par:7.4f}pp  "
                  f"degraded={rmse_deg:7.4f}pp  "
                  f"delta={d_rmse:+.4f}pp")
        if ic_par is not None and ic_deg is not None:
            print(f"  rank-IC     parity={ic_par:+7.4f}  "
                  f"degraded={ic_deg:+7.4f}  "
                  f"delta={d_ic:+.4f}")
        m_abs = audit.get("mean_abs_pred_diff_pp")
        mx_abs = audit.get("max_abs_pred_diff_pp")
        if m_abs is not None:
            print(f"  pred diff   mean|Δ|={m_abs:.4f}pp  "
                  f"max|Δ|={mx_abs:.4f}pp")
        nf = audit.get("n_predict_failures") or 0
        if nf:
            print(f"  predict-failed rows dropped: {nf}")
        if audit.get("error"):
            print(f"  error: {audit['error']}")
        print(f"  thresholds: SMALL_RMSE={BIAS_RMSE_PP_SMALL}pp "
              f"MODERATE_RMSE={BIAS_RMSE_PP_MODERATE}pp "
              f"SMALL_IC={BIAS_RANK_IC_SMALL} "
              f"MODERATE_IC={BIAS_RANK_IC_MODERATE}")

    v = audit.get("verdict")
    if v in ("BIAS_SMALL", "OK", "NO_PARITY_FEATURES", "NOT_TRAINED"):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Empirical confidence intervals for DecisionScorer predictions.

The DecisionScorer MLP emits a *point* estimate of 5-day forward return. A point
estimate with no error bar invites false precision — a bare "+2.3%" verdict
reads as authoritative even when the model's typical miss in that prediction
band is ±6%. The dashboard's scorer card and the EXIT/TRIM/HOLD verdicts derived
from it are only trustworthy if the trader can see how wide the model's error
actually is.

This module computes *empirical* prediction intervals by replaying the trained
model over its own accumulated outcomes (``data/decision_outcomes.jsonl``). Every
historical row carries the features the model sees plus the realized signed
return it *should* have predicted. The residual ``pred - target`` distribution,
bucketed by predicted value, gives a non-parametric error band — P10/P90 of
residuals — plus a directional hit rate (did the sign of the prediction match
the realized outcome?).

Residuals are **not** assumed Gaussian: trading returns are fat-tailed and
skewed, so we report empirical quantiles, never ±σ.

Both public functions are pure: callers pass in the outcome rows and a trained
scorer, so the module is trivially unit-testable and never touches disk itself.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

# A SELL decision's target is the *negated* forward return — the model is
# trained to predict "goodness of taking THIS action", so a SELL ahead of a
# -8% move scored +8. Mirror that flip here or the residuals are nonsense.
_SELL_ACTIONS = {"SELL", "SELL_CALL", "SELL_PUT"}


def _signed_target(record: dict) -> float | None:
    """Realized return in the model's output space (SELL sign-flipped)."""
    fr = record.get("forward_return_5d")
    if not isinstance(fr, (int, float)) or fr != fr:  # reject None / NaN
        return None
    action = str(record.get("action") or "BUY").upper()
    return -float(fr) if action in _SELL_ACTIONS else float(fr)


def compute_residuals(outcomes: list[dict], scorer) -> list[dict]:
    """Replay ``scorer`` over ``outcomes`` → list of {pred, target, residual}.

    Rows the scorer can't score (untrained model, malformed features) or rows
    with no realized return are silently dropped.
    """
    if not getattr(scorer, "is_trained", False):
        return []
    rows: list[dict] = []
    for r in outcomes:
        target = _signed_target(r)
        if target is None:
            continue
        try:
            pred = scorer.predict(
                ml_score=r.get("ml_score", 0.0),
                rsi=r.get("rsi"),
                macd=r.get("macd"),
                mom5=r.get("mom5"),
                mom20=r.get("mom20"),
                regime_mult=r.get("regime_mult", 1.0),
                ticker=str(r.get("ticker") or ""),
                vol_ratio=r.get("vol_ratio"),
                bb_pos=r.get("bb_position"),
                news_urgency=r.get("news_urgency"),
                news_article_count=r.get("news_article_count"),
            )
        except Exception:
            continue
        pred = float(pred)
        rows.append({
            "pred": pred,
            "target": target,
            "residual": pred - target,           # >0 ⇒ model over-predicted
            "dir_hit": (pred > 0) == (target > 0) and pred != 0 and target != 0,
        })
    return rows


def _bucket_stats(rows: list[dict]) -> dict:
    """P10/P50/P90 residual, MAE and directional hit-rate for one bucket."""
    resid = np.array([x["residual"] for x in rows], dtype=np.float64)
    target = np.array([x["target"] for x in rows], dtype=np.float64)
    pred = np.array([x["pred"] for x in rows], dtype=np.float64)
    hits = sum(1 for x in rows if x["dir_hit"])
    return {
        "n": len(rows),
        "pred_lo": round(float(pred.min()), 2),
        "pred_hi": round(float(pred.max()), 2),
        "mae": round(float(np.mean(np.abs(resid))), 2),
        "bias": round(float(np.mean(resid)), 2),
        "resid_p10": round(float(np.percentile(resid, 10)), 2),
        "resid_p50": round(float(np.percentile(resid, 50)), 2),
        "resid_p90": round(float(np.percentile(resid, 90)), 2),
        "mean_actual": round(float(np.mean(target)), 2),
        "directional_accuracy_pct": round(hits / len(rows) * 100, 1),
    }


def build_scorer_confidence(outcomes: list[dict], scorer,
                            n_buckets: int = 5) -> dict:
    """Calibration table: residual quantiles + hit-rate per prediction band.

    Buckets are equal-frequency (quantile) over predicted value, so each band
    holds a comparable sample count rather than empty tails.
    """
    residuals = compute_residuals(outcomes, scorer)
    out: dict = {
        "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "is_trained": bool(getattr(scorer, "is_trained", False)),
        "n_train": int(getattr(scorer, "n_train", 0)),
        "n_samples": len(residuals),
        "buckets": [],
        "overall": None,
    }
    if len(residuals) < n_buckets * 4:
        # Too few replayed rows to bucket meaningfully — caller greys it out.
        return out

    resid_all = np.array([x["residual"] for x in residuals])
    hits = sum(1 for x in residuals if x["dir_hit"])
    out["overall"] = {
        "directional_accuracy_pct": round(hits / len(residuals) * 100, 1),
        "mae": round(float(np.mean(np.abs(resid_all))), 2),
        "bias": round(float(np.mean(resid_all)), 2),
        "resid_p10": round(float(np.percentile(resid_all, 10)), 2),
        "resid_p90": round(float(np.percentile(resid_all, 90)), 2),
    }

    # Equal-frequency split on predicted value.
    preds = np.array([x["pred"] for x in residuals])
    edges = np.quantile(preds, np.linspace(0, 1, n_buckets + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    for i in range(n_buckets):
        lo, hi = edges[i], edges[i + 1]
        members = [x for x in residuals
                   if lo <= x["pred"] < hi or (i == n_buckets - 1 and x["pred"] == hi)]
        if not members:
            continue
        stats = _bucket_stats(members)
        stats["bucket"] = i
        out["buckets"].append(stats)
    return out


def reliability(n: int, mae: float) -> str:
    """Coarse trust label for an interval, from sample count + error width."""
    if n >= 150 and mae <= 6.0:
        return "high"
    if n >= 40:
        return "medium"
    return "low"


def interval_for(pred: float, confidence: dict) -> dict:
    """Map a fresh prediction to an empirical [low, high] band + hit rate.

    ``residual = pred - target`` ⇒ ``target ≈ pred - residual``. The band is
    therefore ``[pred - resid_p90, pred - resid_p10]`` using the residual
    quantiles of whichever calibration bucket the prediction falls into.
    """
    buckets = (confidence or {}).get("buckets") or []
    if not buckets:
        return {"low": None, "high": None, "directional_accuracy_pct": None,
                "reliability": "none", "bucket": None}
    chosen = buckets[0]
    for b in buckets:
        if b["pred_lo"] <= pred <= b["pred_hi"]:
            chosen = b
            break
    else:
        # Outside every observed band — clamp to the nearest extreme bucket.
        chosen = buckets[-1] if pred > buckets[-1]["pred_hi"] else buckets[0]
    return {
        "low": round(pred - chosen["resid_p90"], 2),
        "high": round(pred - chosen["resid_p10"], 2),
        "directional_accuracy_pct": chosen["directional_accuracy_pct"],
        "reliability": reliability(chosen["n"], chosen["mae"]),
        "bucket": chosen["bucket"],
    }


if __name__ == "__main__":  # smoke test
    import json
    from pathlib import Path
    from paper_trader.ml.decision_scorer import DecisionScorer

    path = Path(__file__).resolve().parent.parent.parent / "data" / "decision_outcomes.jsonl"
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()][-4000:]
    conf = build_scorer_confidence(rows, DecisionScorer())
    print(json.dumps(conf, indent=2))
    if conf["overall"]:
        print("interval for +5%:", interval_for(5.0, conf))
        print("interval for -3%:", interval_for(-3.0, conf))

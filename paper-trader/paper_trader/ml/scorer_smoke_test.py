"""DecisionScorer smoke test — fast, read-only end-to-end sanity check.

The natural quant question after deploying a freshly retrained pickle:
**does the model actually produce sensible predictions, RIGHT NOW, before
the next trade cycle reads it?** Every sibling diagnostic answers a deeper
quality question (calibration / gate skill / OOS rank-IC) but each one runs
a multi-second sweep over `decision_outcomes.jsonl` — useless when an
operator just wants a sub-second "is this pickle even alive" check before a
restart.

This is a **read-only diagnostic**. It never trains, never touches
`decision_scorer.pkl`, `decision_outcomes.jsonl`, `scorer_skill_log.jsonl`,
`build_features`, `N_FEATURES`, or any trade path — same operational
discipline as `paper_trader/ml/calibration.py` / `gate_audit.py` /
`scorer_freshness.py` / `deploy_audit.py`. Safe to run against the live
unattended loop.

**Why this is not any existing tool.** `scorer_freshness` answers *is the
loop still re-pickling*; `deploy_audit` answers *does the pickled config
match source*. Neither asks the basic question: *does
`DecisionScorer().predict_with_meta(...)` return finite, non-degenerate
values for a sweep of realistic inputs?* A model that loaded successfully,
matches source config, but predicts the same constant for every input — a
*degenerate predictor* — passes both existing diagnostics and silently
disables the conviction gate at the predict level (the gate's bucket
arms ±10/±5/0 collapse to one bucket forever). This module catches that.

Verdicts (exit code mirrors the sibling diagnostics for cron):
  HEALTHY                  all probes finite, predictions span ≥ 2 distinct conviction-gate buckets → 0
  UNTRAINED                no pickle / DecisionScorer.is_trained is False                          → 0
  DEGENERATE_CONSTANT      every probe returned the same prediction (±tolerance)                   → 2
  GATE_BUCKETS_DEGENERATE  predictions nominally distinct but all in ONE conviction-gate arm       → 2
  BROKEN_PREDICT           one or more probes raised / returned non-finite                         → 2

The GATE_BUCKETS_DEGENERATE verdict is strictly weaker than DEGENERATE_CONSTANT
and catches a real failure mode the existing check misses: a model whose
predictions vary by 0.0001%-1% (passing the constant-tolerance check) but
NEVER cross the gate's ±10 / ±5 / 0 thresholds. In that state the
conviction gate's ×0.6/×0.85/×1.0/×1.15/×1.3 arms collapse to one
multiplier across all real inputs — every BUY ends up with the same gate
adjustment regardless of the model's "prediction". See AGENTS.md review
pass #2 for the n_train=400 clobber that produced exactly this pattern
while passing every other smoke check.
"""
from __future__ import annotations

from typing import Any


# Eight probes spanning the sector axis + a couple of edge-case features.
# Picked to be IN-distribution so the `off_distribution` flag should be
# False on every probe of a healthy model — that flag firing on a normal
# probe is itself a signal something is wrong with the deployed pickle.
# Tickers are real watchlist names so SECTOR_MAP routes them to the
# documented sector bucket; this is INTENTIONALLY not a randomised draw so
# the report is deterministic and reproducible across runs and operators.
_PROBES: tuple[dict[str, Any], ...] = (
    {"label": "tech_neutral",       "ticker": "NVDA",  "ml_score": 1.0,
     "rsi": 55.0, "macd": 0.0, "mom5": 0.0, "mom20": 0.0, "regime_mult": 1.0,
     "vol_ratio": 1.0, "bb_pos": 0.0, "news_urgency": 50.0,
     "news_article_count": 1.0},
    {"label": "tech_oversold",      "ticker": "AMD",   "ml_score": 2.0,
     "rsi": 28.0, "macd": -0.5, "mom5": -3.0, "mom20": -5.0, "regime_mult": 0.6,
     "vol_ratio": 1.5, "bb_pos": -1.5, "news_urgency": 70.0,
     "news_article_count": 3.0},
    {"label": "tech_overbought",    "ticker": "AAPL",  "ml_score": 1.5,
     "rsi": 78.0, "macd": 1.2, "mom5": 6.0, "mom20": 12.0, "regime_mult": 1.0,
     "vol_ratio": 1.1, "bb_pos": 1.5, "news_urgency": 40.0,
     "news_article_count": 2.0},
    {"label": "energy_neutral",     "ticker": "XOM",   "ml_score": 0.5,
     "rsi": 50.0, "macd": 0.0, "mom5": 1.0, "mom20": 2.0, "regime_mult": 1.0,
     "vol_ratio": 1.0, "bb_pos": 0.0, "news_urgency": 50.0,
     "news_article_count": 1.0},
    {"label": "financials_strong",  "ticker": "JPM",   "ml_score": 2.5,
     "rsi": 62.0, "macd": 0.8, "mom5": 2.5, "mom20": 4.0, "regime_mult": 1.0,
     "vol_ratio": 1.2, "bb_pos": 0.8, "news_urgency": 60.0,
     "news_article_count": 2.0},
    {"label": "healthcare_weak",    "ticker": "UNH",   "ml_score": -0.5,
     "rsi": 42.0, "macd": -0.3, "mom5": -1.5, "mom20": -3.0, "regime_mult": 0.6,
     "vol_ratio": 1.3, "bb_pos": -0.8, "news_urgency": 65.0,
     "news_article_count": 2.0},
    {"label": "commodities_neutral", "ticker": "GLD",  "ml_score": 0.0,
     "rsi": 51.0, "macd": 0.0, "mom5": 0.5, "mom20": 1.0, "regime_mult": 1.0,
     "vol_ratio": 1.0, "bb_pos": 0.0, "news_urgency": 50.0,
     "news_article_count": 1.0},
    {"label": "crypto_strong",      "ticker": "COIN",  "ml_score": 2.0,
     "rsi": 60.0, "macd": 0.5, "mom5": 4.0, "mom20": 8.0, "regime_mult": 1.0,
     "vol_ratio": 1.4, "bb_pos": 1.0, "news_urgency": 55.0,
     "news_article_count": 2.0},
)

# Two probes that should genuinely trigger off-distribution (extreme RSI /
# extreme momentum past the empirical label support). A trained scorer
# whose `off_distribution` flag NEVER fires on these — or fires on every
# in-distribution probe above — is also a degenerate signature, but the
# core HEALTHY verdict does not depend on this (the flag is informational).
_EDGE_PROBES: tuple[dict[str, Any], ...] = (
    {"label": "edge_extreme_oversold", "ticker": "NVDA", "ml_score": 5.0,
     "rsi": 10.0, "macd": -3.0, "mom5": -25.0, "mom20": -50.0,
     "regime_mult": 0.3, "vol_ratio": 3.0, "bb_pos": -2.0,
     "news_urgency": 95.0, "news_article_count": 10.0},
    {"label": "edge_extreme_overbought", "ticker": "NVDA", "ml_score": 5.0,
     "rsi": 90.0, "macd": 3.0, "mom5": 25.0, "mom20": 50.0,
     "regime_mult": 1.0, "vol_ratio": 3.0, "bb_pos": 2.0,
     "news_urgency": 95.0, "news_article_count": 10.0},
)

# Tolerance for "two predictions are the same constant" — a healthy MLP
# differs on at least this many basis points across the 8 in-distribution
# probes above. 1e-4 (0.0001%) is well below any realistic prediction
# variance from a trained net but generous enough that float-precision
# wobble in the lstsq fallback does not falsely flag DEGENERATE.
_CONSTANT_TOLERANCE_PCT = 1e-4


# Conviction-gate bucket boundaries — kept in lockstep with `_ml_decide`'s
# four-arm scorer gate (CLAUDE.md §6, `paper_trader/backtest.py::_ml_decide`):
#   p < -10  → ×0.6  (strong headwind)
#   p < 0    → ×0.85 (mild headwind)
#   0 ≤ p ≤ 5 → ×1.0 (neutral / no change)
#   p > 5    → ×1.15 (mild tailwind)
#   p > 10   → ×1.3  (strong tailwind, capped at 0.95)
# `_gate_bucket()` returns the string label _ml_decide acts on, so a scorer
# whose predictions all map to the SAME bucket across the in-distribution
# probes has a dormant gate — every conviction adjustment is the same number.
# That is a *strictly weaker* failure mode than DEGENERATE_CONSTANT (which
# checks raw equality at 1e-4 tolerance): two predictions of 0.5% and 2.0%
# are "distinct" at that tolerance but BOTH fall in the neutral bucket, so
# the gate's ±10/±5/0 arms never engage. AGENTS.md (review pass 2) records
# the live-system failure pattern this catches: deployed pkl with
# `n_train=400` posted near-zero predictive value despite "passing" every
# existing diagnostic.
_GATE_BUCKETS: tuple[str, ...] = (
    "strong_headwind", "mild_headwind", "neutral",
    "mild_tailwind", "strong_tailwind",
)


def _gate_bucket(pred: float) -> str:
    """Map a scorer prediction (% 5d forward return) to the conviction-gate
    arm label `_ml_decide` would apply. Pure, total, never raises — a
    non-finite input falls through to ``neutral`` (the no-op arm) so a
    diagnostic crash can never propagate from this helper to the verdict.

    Lockstep mirror of `paper_trader/backtest.py::_ml_decide`'s four-arm
    threshold ladder (CLAUDE.md §6). The neutral arm is INCLUSIVE on both
    ends of [0, 5] to match `_ml_decide`'s `elif scorer_pred > 5.0` /
    `elif scorer_pred < 0.0` chain exactly: a prediction of exactly 0.0
    hits neither the negative branch (`< 0`) nor the positive branch
    (`> 5`), and a prediction of exactly 5.0 hits neither (`> 5` is False)
    — so both are neutral, just like the gate.
    """
    try:
        p = float(pred)
        if p != p:  # NaN
            return "neutral"
    except (TypeError, ValueError):
        return "neutral"
    if p < -10.0:
        return "strong_headwind"
    if p < 0.0:
        return "mild_headwind"
    if p > 10.0:
        return "strong_tailwind"
    if p > 5.0:
        return "mild_tailwind"
    return "neutral"


VERDICTS: tuple[str, ...] = (
    "HEALTHY",
    "UNTRAINED",
    "DEGENERATE_CONSTANT",
    "GATE_BUCKETS_DEGENERATE",
    "BROKEN_PREDICT",
)


def _safe_pred_with_meta(scorer, probe: dict) -> dict:
    """Call `scorer.predict_with_meta` for one probe. Never raises — a
    failure returns a record with `error` set so the caller can verdict
    BROKEN_PREDICT rather than crashing. Mirrors the rest of the diagnostic
    family: degrade, never propagate."""
    kw = {k: probe[k] for k in (
        "ml_score", "rsi", "macd", "mom5", "mom20", "regime_mult", "ticker",
        "vol_ratio", "bb_pos", "news_urgency", "news_article_count",
    )}
    try:
        meta = scorer.predict_with_meta(**kw)
        # Defensive: `predict_with_meta` is contracted to always return a
        # dict with finite `pred`. A scorer whose internal predict raised
        # already returns a dict with `clamped=True / off_distribution=True
        # / pred=0.0`. We treat that as a healthy degrade, not BROKEN,
        # because the public scalar contract held. Real BROKEN is only when
        # this CALL itself raises (caught below) or `pred` is non-finite.
        pred = meta.get("pred")
        try:
            pf = float(pred)
        except (TypeError, ValueError):
            return {"label": probe["label"], "ticker": probe["ticker"],
                    "error": f"non-numeric pred: {pred!r}"}
        if pf != pf or pf in (float("inf"), float("-inf")):
            return {"label": probe["label"], "ticker": probe["ticker"],
                    "error": f"non-finite pred: {pred!r}"}
        return {
            "label": probe["label"],
            "ticker": probe["ticker"],
            "pred": round(pf, 4),
            "raw": round(float(meta.get("raw", pf)), 4),
            "clamped": bool(meta.get("clamped", False)),
            "off_distribution": bool(meta.get("off_distribution", False)),
        }
    except Exception as e:
        return {"label": probe["label"], "ticker": probe["ticker"],
                "error": f"{type(e).__name__}: {e}"}


def scorer_smoke_report(scorer=None) -> dict:
    """End-to-end smoke check of the deployed DecisionScorer.

    ``scorer`` defaults to a fresh ``DecisionScorer()`` — i.e. the exact
    pickle the live ``_ml_decide`` gate loads. Injecting a scorer is for
    tests: every existing diagnostic in this folder follows the same
    inject-for-test, default-to-production pattern.

    Returns a JSON-safe dict. Never raises — a scorer construction fault
    degrades to ``BROKEN_LOAD``-shaped output via the import guard, the
    same hardening class as ``deploy_audit.is_deploy_stale``.
    """
    out: dict = {
        "verdict": "UNTRAINED",
        "is_trained": False,
        "n_train": 0,
        "n_probes": len(_PROBES),
        "n_edge_probes": len(_EDGE_PROBES),
        "probes": [],
        "edge_probes": [],
        "distinct_predictions": 0,
        # Per-bucket count across the in-distribution probes — pre-populated
        # with every documented bucket name (zeros) so downstream consumers
        # never KeyError on an absent arm and a dashboard / Discord template
        # can render the histogram unconditionally. Keys are stable; ORDER
        # is `_GATE_BUCKETS` (headwind → neutral → tailwind) so a sorted
        # render reads left-to-right as it does in `_ml_decide`'s ladder.
        "gate_bucket_counts": {b: 0 for b in _GATE_BUCKETS},
        "distinct_gate_buckets": 0,
        "off_distribution_in_distribution": 0,
        "off_distribution_edge": 0,
        "broken_probe_count": 0,
        "hint": "",
    }

    if scorer is None:
        try:
            from paper_trader.ml.decision_scorer import DecisionScorer
            scorer = DecisionScorer()
        except Exception as e:
            out["verdict"] = "BROKEN_PREDICT"
            out["hint"] = (
                f"DecisionScorer() construction raised "
                f"{type(e).__name__}: {e}"
            )
            return out

    is_trained = bool(getattr(scorer, "is_trained", False))
    out["is_trained"] = is_trained
    out["n_train"] = int(getattr(scorer, "n_train", 0) or 0)

    if not is_trained:
        out["hint"] = (
            "DecisionScorer.is_trained is False — no pkl on disk or load "
            "failed. predict() is a no-op 0.0%; accumulate ≥30 deduped "
            "outcomes then retrain."
        )
        return out

    probe_results = [_safe_pred_with_meta(scorer, p) for p in _PROBES]
    edge_results = [_safe_pred_with_meta(scorer, p) for p in _EDGE_PROBES]
    out["probes"] = probe_results
    out["edge_probes"] = edge_results

    broken = [r for r in probe_results + edge_results if "error" in r]
    out["broken_probe_count"] = len(broken)
    if broken:
        # Even one BROKEN probe is critical — the live gate would hit this
        # exact same exception on its next call. Verdict it CONSPICUOUSLY,
        # do not paper over with a HEALTHY-derived count of good probes.
        out["verdict"] = "BROKEN_PREDICT"
        first = broken[0]
        out["hint"] = (
            f"{len(broken)} of {len(probe_results) + len(edge_results)} "
            f"probes failed; first: {first['label']} ({first['ticker']}) "
            f"→ {first['error']}"
        )
        return out

    preds = [r["pred"] for r in probe_results]
    # Distinct predictions, modulo the constant-tolerance bucketing.
    # A genuinely healthy MLP differs across these 8 probes by far more
    # than 1e-4 — anything pinning to one value is degenerate.
    distinct = sorted(set(round(p / _CONSTANT_TOLERANCE_PCT) for p in preds))
    out["distinct_predictions"] = len(distinct)

    # Conviction-gate bucket histogram across the in-distribution probes.
    # `_gate_bucket` is the lockstep mirror of `_ml_decide`'s four-arm
    # ladder, so this counts the *actual gate decisions* the deployed
    # pickle would make on these probes — not just raw prediction
    # variance. See the `_GATE_BUCKETS` comment for why this is a
    # strictly different failure mode than DEGENERATE_CONSTANT.
    for p in preds:
        out["gate_bucket_counts"][_gate_bucket(p)] += 1
    out["distinct_gate_buckets"] = sum(
        1 for c in out["gate_bucket_counts"].values() if c > 0
    )

    # In-distribution probes should ideally NOT fire off_distribution.
    # If most do, the scaler / model trained on a wildly different feature
    # distribution than the inputs we just fed it — a silent feature drift.
    out["off_distribution_in_distribution"] = sum(
        1 for r in probe_results if r.get("off_distribution"))
    out["off_distribution_edge"] = sum(
        1 for r in edge_results if r.get("off_distribution"))

    if out["distinct_predictions"] < 2:
        out["verdict"] = "DEGENERATE_CONSTANT"
        out["hint"] = (
            f"all {len(preds)} in-distribution probes returned the same "
            f"prediction (within ±{_CONSTANT_TOLERANCE_PCT}% tolerance) — "
            f"the scorer is a constant predictor; the gate's ±10/±5/0 "
            f"buckets collapse to one bucket forever, disabling the "
            f"conviction gate at the predict level"
        )
        return out

    # Gate-bucket diversity check — STRICTLY WEAKER than the constant
    # check above, so it runs only after DEGENERATE_CONSTANT has cleared.
    # A model whose predictions are nominally distinct (passes the 1e-4
    # tolerance) but all map to the same gate arm is in the "the gate is
    # dormant" failure mode AGENTS.md review pass #2 documented for the
    # n_train=400 clobber: every BUY gets the same conviction multiplier
    # regardless of the model's "prediction".
    if out["distinct_gate_buckets"] < 2:
        out["verdict"] = "GATE_BUCKETS_DEGENERATE"
        only_bucket = next(
            (b for b, c in out["gate_bucket_counts"].items() if c > 0),
            "neutral",
        )
        out["hint"] = (
            f"all {len(preds)} in-distribution probes mapped to the SAME "
            f"conviction-gate arm '{only_bucket}' — predictions vary "
            f"({out['distinct_predictions']} distinct values) but none "
            f"cross the ±10/±5/0 thresholds, so the gate's "
            f"×0.6/×0.85/×1.0/×1.15/×1.3 arms collapse to one multiplier "
            f"on these inputs; the conviction gate is operationally "
            f"dormant"
        )
        return out

    out["verdict"] = "HEALTHY"
    out["hint"] = (
        f"n_train={out['n_train']} | {out['distinct_predictions']} distinct "
        f"predictions across {len(preds)} in-distribution probes | "
        f"{out['distinct_gate_buckets']}/{len(_GATE_BUCKETS)} gate buckets | "
        f"off-dist: {out['off_distribution_in_distribution']}/{len(preds)} "
        f"in-distribution, {out['off_distribution_edge']}/"
        f"{len(_EDGE_PROBES)} edge"
    )
    return out


def analyze() -> dict:
    """Public entry point — full smoke report (read-only)."""
    return scorer_smoke_report()


def _cli() -> int:
    """`python3 -m paper_trader.ml.scorer_smoke_test [--json]` — fast
    read-only sanity check.

    Exit mirrors the sibling diagnostics so a cron can branch on "the
    deployed pickle is degenerate or broken right now": 0 on HEALTHY /
    UNTRAINED, 2 on DEGENERATE_CONSTANT / BROKEN_PREDICT."""
    import argparse
    import json
    import sys

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.scorer_smoke_test",
        description="Fast read-only smoke check of the deployed DecisionScorer.",
    )
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of a table.")
    args = p.parse_args(sys.argv[1:])

    rep = scorer_smoke_report()
    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
    else:
        print(f"VERDICT: {rep['verdict']}  ({rep['hint']})")
        print(f"  is_trained={rep['is_trained']}  n_train={rep['n_train']}")
        print(f"  distinct_predictions={rep['distinct_predictions']}/"
              f"{rep['n_probes']}  "
              f"distinct_gate_buckets={rep.get('distinct_gate_buckets', 0)}/"
              f"{len(_GATE_BUCKETS)}  broken={rep['broken_probe_count']}")
        buckets = rep.get("gate_bucket_counts") or {}
        if buckets:
            # Render in the documented headwind→neutral→tailwind order so
            # the histogram reads left-to-right as it does in `_ml_decide`.
            ordered = "  ".join(
                f"{b}={buckets.get(b, 0)}" for b in _GATE_BUCKETS
            )
            print(f"  gate buckets: {ordered}")
        for r in rep["probes"]:
            if "error" in r:
                print(f"    {r['label']:<26} {r['ticker']:<6} "
                      f"ERROR {r['error']}")
                continue
            flag = " [off-dist]" if r.get("off_distribution") else ""
            bucket = _gate_bucket(r["pred"])
            print(f"    {r['label']:<26} {r['ticker']:<6} "
                  f"pred={r['pred']:+.3f}%  raw={r['raw']:+.3f}%  "
                  f"gate={bucket}{flag}")
    return 0 if rep["verdict"] in ("HEALTHY", "UNTRAINED") else 2


if __name__ == "__main__":
    raise SystemExit(_cli())

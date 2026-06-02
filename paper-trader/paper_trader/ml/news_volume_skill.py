"""Per-news-volume *scorer* skill diagnostic — does the DecisionScorer's
OOS edge IMPROVE, STAY FLAT, or DEGRADE as more news articles support the
decision?

This is the **news-availability-conditional** sibling of
``paper_trader/ml/action_skill.py`` (per-action),
``paper_trader/ml/regime_audit.py`` (per-regime), and
``paper_trader/ml/sector_skill.py`` (per-sector). The aggregate
``_oos_rank_metrics`` (one number across all news-volume slices) hides a
load-bearing question every existing diagnostic structurally misses:

  *Is the scorer's near-zero OOS skill the same in news-poor and news-rich
  conditions, or does adding news context move it?*

`feature_importance` answers the model's view ("does the model use the news
columns?"). `attribution_audit` answers the model's per-record view ("what
contribution does the model assign each feature?"). **Neither slices the
realized OOS outcomes by news availability** and asks whether the model's
predictions are more accurate when more news exists. That is the exact gap
this module fills, and is genuinely orthogonal to both:

- `feature_importance`  — "if I scramble news columns, how much skill is lost?"
- `attribution_audit`   — "how strongly does the model lean on news features?"
- `news_volume_skill`   — "does the model's PREDICTION SKILL vary with news count?"

A monotone-positive verdict means the scorer is genuinely extracting signal
from news context — more news → better predictions. A monotone-negative
verdict (skill degrades with more news) is the strongest concerning signal:
adding news context is making predictions WORSE, suggesting the news
features are encoding noise. A flat verdict says news availability doesn't
differentiate skill — the model is leaning on quant signals, news is
decorative.

The buckets are inclusive lower-bound / exclusive upper-bound on
``news_article_count``, matching the ``build_features`` ``cnt_v`` clamp
(0..20). ``None`` (the "no-news" sentinel ``_compute_decision_outcomes``
emits when ``news_article_count <= 0``) collapses into bucket ``no_news``
so a quant can see the "decisions made without supporting news" slice
directly.

Operational discipline is identical to its siblings: **read-only** — no
train, no ``decision_scorer.pkl`` / ``build_features`` / ``N_FEATURES`` /
trade path touch, never raises on bad input — so it is safe to run against
the live unattended continuous loop and cannot break pickle compatibility.

Verdict per bucket (crisp, threshold-driven so it is exactly testable):

| Verdict | Meaning |
|---------|---------|
| `INSUFFICIENT` | < ``MIN_OUTCOMES_PER_BUCKET`` aligned outcomes — no stable IC |
| `INVERTED` | ``rank_ic ≤ -IC_GOOD`` — predictions anti-correlated with realized in this slice |
| `EDGE` | ``rank_ic ≥ IC_GOOD`` — predictions genuinely rank-predict realized in this slice |
| `WEAK_EDGE` | ``IC_MIN ≤ rank_ic < IC_GOOD`` — usable as a tie-breaker, not primary |
| `NO_EDGE` | ``-IC_GOOD < rank_ic < IC_MIN`` — no demonstrated rank skill |

Overall verdict (across all sufficient buckets, ordered no_news < sparse <
moderate < dense): ``INSUFFICIENT_DATA`` (< ``MIN_RECORDS`` total or <2
sufficient buckets), ``NEWS_VALUE_MONOTONIC_POSITIVE`` (rank_ic strictly
non-decreasing across sufficient buckets, with a measurable spread),
``NEWS_VALUE_MONOTONIC_NEGATIVE`` (strictly non-increasing with a spread),
``NEWS_VALUE_INVARIANT`` (spread < ``BUCKET_SPREAD_TOL``), ``MIXED`` (non-
monotone), or ``HAS_INVERTED_BUCKET`` (≥1 bucket INVERTED — actionable red
flag, surfaced first so it is unmissable under a healthy-looking aggregate).

```bash
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.news_volume_skill
cd /home/zeph/paper-trader && python3 -m pytest tests/test_news_volume_skill.py -v
```
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Single source of truth for the tie-aware rank correlation — reused so
# this per-bucket IC and the in-sample calibration / persona_skill /
# action_skill / regime_audit metrics can never drift (the AGENTS.md
# invariant-10 spirit). Tie-awareness is load-bearing: the scorer clamps
# to ±PRED_CLAMP_PCT, so off-distribution predictions tie at exactly ±50
# and a naïve argsort would fabricate rank skill there.
from paper_trader.ml.calibration import _spearman
from paper_trader.ml.decision_scorer import _to_float

# Thresholds — module-level so tests assert exact verdicts and a tuning
# change is one reviewable edit (mirrors action_skill / calibration /
# persona_skill, the codebase constants-at-module-scope convention).
MIN_RECORDS = 30              # minimum aligned outcomes overall for any verdict
MIN_OUTCOMES_PER_BUCKET = 15  # below this a bucket's Spearman is too noisy
IC_MIN = 0.05                 # rank-skill bar — mirrors skill_trend.IC_MIN
IC_GOOD = 0.10                # stronger rank-skill bar — mirrors action_skill.IC_GOOD
# Below this rank-IC range across sufficient buckets we call it INVARIANT.
# 0.05 = one full "rank-skill bar" of spread — anything smaller is well
# within sampling noise at n≈15 per bucket.
BUCKET_SPREAD_TOL = 0.05

# Buckets — keyed in the order they will print and the order the
# monotonicity check walks. Ranges are [lo, hi): a record with
# news_article_count == hi belongs to the NEXT bucket. None / 0
# collapse to ``no_news`` (matching `_compute_decision_outcomes`'s
# "<=0 → None" sentinel: a decision with no supporting news).
#
# The empirical distribution of news_article_count in
# data/decision_outcomes.jsonl is heavily zero-inflated (most decisions
# have no specific news, just quant signals); these breakpoints were
# chosen so each bucket carries enough mass to compute a stable IC
# without artificially inflating MIN_OUTCOMES_PER_BUCKET. The 20 cap
# matches `build_features.cnt_v`'s `min(20, ...)` clamp.
NEWS_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("no_news",  0.0,  1.0),   # None or 0 articles
    ("sparse",   1.0,  3.0),   # 1-2 articles
    ("moderate", 3.0, 10.0),   # 3-9 articles
    ("dense",   10.0, 100.0),  # 10+ articles (cnt_v clamps at 20)
)
_BUCKET_NAMES = tuple(b[0] for b in NEWS_BUCKETS)


def _bucket_for(news_article_count) -> str | None:
    """Map a record's ``news_article_count`` to one of the 4 bucket names,
    or ``None`` when the value is unparseable as a non-negative number.

    Pure, total, never raises. ``None`` (the no-news sentinel) and 0.0
    both map to ``no_news`` — the bucket that captures "decisions made
    without supporting news". Negative values are unparseable
    (`_compute_decision_outcomes` already coerces <=0 to None — this is
    belt-and-braces).
    """
    if news_article_count is None:
        return "no_news"
    try:
        v = float(news_article_count)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    if v < 0:
        return None
    # Clamp >100 into the dense bucket — defensive, matches build_features.
    if v >= NEWS_BUCKETS[-1][2]:
        return NEWS_BUCKETS[-1][0]
    for name, lo, hi in NEWS_BUCKETS:
        if lo <= v < hi:
            return name
    # Unreachable given the contiguous coverage above (no_news starts at 0,
    # dense ends at 100, no gaps) — return None defensively.
    return None


def _aligned_pred(scorer, record: dict) -> tuple[float, float] | None:
    """Return (scorer prediction, action-aligned realized return) for one
    outcome record, or ``None`` when the record is unusable.

    Mirrors ``action_skill._aligned_pred`` and ``persona_skill._aligned``
    EXACTLY (single SELL-sign-flip convention, NaN-sentinel target drop,
    scorer-exception → drop). The prediction is NOT flipped on SELL: the
    scorer was trained on already-flipped SELL targets, so its output for
    a SELL feature vector encodes action-aligned goodness directly. The
    REALIZED target IS flipped for SELL so "good" carries one consistent
    meaning across the codebase.
    """
    fr = record.get("forward_return_5d")
    if fr is None:
        return None
    t = _to_float(fr, float("nan"))
    if t != t:                          # NaN → unparseable → drop
        return None
    try:
        p = scorer.predict(
            ml_score=_to_float(record.get("ml_score"), 0.0),
            rsi=record.get("rsi"), macd=record.get("macd"),
            mom5=record.get("mom5"), mom20=record.get("mom20"),
            regime_mult=_to_float(record.get("regime_mult"), 1.0),
            ticker=str(record.get("ticker") or ""),
            vol_ratio=record.get("vol_ratio"),
            bb_pos=record.get("bb_position"),
            news_urgency=record.get("news_urgency"),
            news_article_count=record.get("news_article_count"),
            ema200_above=record.get("ema200_above"),
            hist_cross_up=record.get("hist_cross_up"),
            macd_below_zero_cross=record.get("macd_below_zero_cross"),
        )
    except Exception:
        return None
    try:
        p = float(p)
    except (TypeError, ValueError):
        return None
    if p != p:                          # NaN predict result — defensive
        return None
    if str(record.get("action") or "BUY").upper() == "SELL":
        t = -t
    return p, t


def _verdict_for(ic: float | None, n: int) -> str:
    """Per-bucket verdict for a single (ic, n) pair. Pure, deterministic,
    threshold-driven so tests can assert exact strings."""
    if n < MIN_OUTCOMES_PER_BUCKET:
        return "INSUFFICIENT"
    if ic is None or ic != ic:
        return "INSUFFICIENT"
    if ic <= -IC_GOOD:
        return "INVERTED"
    if ic >= IC_GOOD:
        return "EDGE"
    if ic >= IC_MIN:
        return "WEAK_EDGE"
    return "NO_EDGE"


def _overall_verdict(by_bucket: dict[str, dict]) -> tuple[str, str]:
    """Combine per-bucket verdicts into one ``(verdict, hint)`` pair.

    Sufficient buckets are those whose verdict ≠ "INSUFFICIENT". The
    overall verdict is driven by the rank-IC trajectory across them, in
    bucket order (no_news → sparse → moderate → dense).
    """
    # An inverted bucket is the actionable red flag — surface first so
    # it cannot be missed under a healthy-looking aggregate (same
    # discipline as action_skill's HAS_INVERTED_ACTION).
    inverted = [b for b in _BUCKET_NAMES
                if by_bucket[b]["verdict"] == "INVERTED"]
    if inverted:
        return ("HAS_INVERTED_BUCKET",
                f"{','.join(inverted)} bucket(s) have rank_ic ≤ -{IC_GOOD} — "
                f"the scorer is anti-predictive on the news slice(s). Treat "
                f"as an actionable concerning state, not a tuning detail.")

    sufficient = [b for b in _BUCKET_NAMES
                  if by_bucket[b]["verdict"] != "INSUFFICIENT"
                  and by_bucket[b]["rank_ic"] is not None]
    if len(sufficient) < 2:
        return ("INSUFFICIENT_DATA",
                f"< 2 buckets reach the per-bucket minimum "
                f"({MIN_OUTCOMES_PER_BUCKET}) — cannot assess "
                f"news-volume effect.")

    ics = [by_bucket[b]["rank_ic"] for b in sufficient]
    spread = max(ics) - min(ics)
    if spread < BUCKET_SPREAD_TOL:
        return ("NEWS_VALUE_INVARIANT",
                f"rank-IC varies by only {spread:.3f} across the "
                f"{len(sufficient)} sufficient bucket(s) — news volume does "
                f"NOT differentiate scorer skill. The model is leaning on "
                f"quant signals; news is decorative.")

    # Monotone non-decreasing across sufficient buckets in canonical
    # order (no_news < sparse < moderate < dense). Strict equality is
    # allowed (>=) but we already required spread >= BUCKET_SPREAD_TOL.
    non_decreasing = all(ics[i] <= ics[i + 1] for i in range(len(ics) - 1))
    non_increasing = all(ics[i] >= ics[i + 1] for i in range(len(ics) - 1))

    if non_decreasing and not non_increasing:
        return ("NEWS_VALUE_MONOTONIC_POSITIVE",
                f"rank-IC rises monotonically with news volume "
                f"({ics[0]:+.3f} → {ics[-1]:+.3f}, spread {spread:.3f}) "
                f"— the scorer extracts more signal from news-rich "
                f"decisions. Lifting the news features in training would "
                f"likely help.")
    if non_increasing and not non_decreasing:
        return ("NEWS_VALUE_MONOTONIC_NEGATIVE",
                f"rank-IC FALLS monotonically with news volume "
                f"({ics[0]:+.3f} → {ics[-1]:+.3f}, spread {spread:.3f}) "
                f"— adding news context degrades predictions. Suggests "
                f"the news columns are encoding noise; auditing the "
                f"upstream news scoring pipeline is the lever.")
    return ("MIXED",
            f"rank-IC varies non-monotonically across sufficient buckets "
            f"(spread {spread:.3f}). No clean news-volume effect; the "
            f"per-bucket table is the honest read.")


def news_volume_skill(scorer, records) -> dict:
    """Per-news-volume OOS rank skill of a deployed scorer over outcomes.

    ``records`` is any iterable of dicts with at least ``action``,
    ``ml_score``, ``forward_return_5d``, ``news_article_count``, and the
    other quant feature columns (the ``decision_outcomes.jsonl`` row
    shape). Rows missing/with a non-finite ``forward_return_5d``, an
    untrained scorer, or a predict fault are dropped.

    Returns a JSON-safe dict::

        {
          "status": "ok" | "untrained" | "insufficient_data" | "error",
          "verdict": <overall verdict string>,
          "n_records": int,
          "by_bucket": {bucket_name: {n, rank_ic, dir_acc,
                                       mean_aligned_return, verdict}},
          "hint": str,
        }
    """
    out_skel = {
        "status": "ok",
        "verdict": "INSUFFICIENT_DATA",
        "n_records": 0,
        "by_bucket": {b: {"n": 0, "rank_ic": None, "dir_acc": None,
                          "mean_aligned_return": None,
                          "verdict": "INSUFFICIENT"} for b in _BUCKET_NAMES},
        "hint": "",
    }

    if not getattr(scorer, "is_trained", False):
        out_skel["status"] = "untrained"
        out_skel["hint"] = ("scorer is not trained — no predictions to "
                            "evaluate. Train the scorer first.")
        return out_skel

    buckets: dict[str, dict] = {b: {"sig": [], "tgt": []}
                                for b in _BUCKET_NAMES}
    n_aligned = 0
    for r in records:
        bname = _bucket_for(r.get("news_article_count"))
        if bname is None:
            continue
        pair = _aligned_pred(scorer, r)
        if pair is None:
            continue
        p, t = pair
        n_aligned += 1
        buckets[bname]["sig"].append(p)
        buckets[bname]["tgt"].append(t)

    if n_aligned < MIN_RECORDS:
        out_skel["n_records"] = n_aligned
        out_skel["status"] = "insufficient_data"
        out_skel["hint"] = (f"need ≥{MIN_RECORDS} aligned outcomes with a "
                            f"finite forward_return_5d, have {n_aligned}")
        return out_skel

    by_bucket: dict[str, dict] = {}
    for bname in _BUCKET_NAMES:
        b = buckets[bname]
        n = len(b["sig"])
        if n == 0:
            by_bucket[bname] = {"n": 0, "rank_ic": None, "dir_acc": None,
                                "mean_aligned_return": None,
                                "verdict": "INSUFFICIENT"}
            continue
        sig = np.asarray(b["sig"], dtype=np.float64)
        tgt = np.asarray(b["tgt"], dtype=np.float64)
        ic: float | None
        if n >= 2:
            ic_raw = float(_spearman(sig, tgt))
            ic = round(ic_raw, 4) if ic_raw == ic_raw else None
        else:
            ic = None
        # dir_acc = sign(pred) == sign(realized) excluding zero pairs that
        # carry no directional truth (the `_oos_rank_metrics` convention).
        dir_pairs = [(p, t) for p, t in zip(sig.tolist(), tgt.tolist())
                     if p != 0.0 and t != 0.0]
        if dir_pairs:
            hits = sum(1 for p, t in dir_pairs if (p > 0) == (t > 0))
            dir_acc: float | None = round(hits / len(dir_pairs), 4)
        else:
            dir_acc = None
        mean_ret = round(float(tgt.mean()), 4) if n else None
        verdict = _verdict_for(ic, n)
        by_bucket[bname] = {
            "n": n, "rank_ic": ic, "dir_acc": dir_acc,
            "mean_aligned_return": mean_ret, "verdict": verdict,
        }

    verdict, hint = _overall_verdict(by_bucket)
    return {
        "status": "ok",
        "verdict": verdict,
        "n_records": n_aligned,
        "by_bucket": by_bucket,
        "hint": hint,
    }


def _load_outcomes(path: Path) -> list[dict]:
    """Robust JSONL load — same best-effort discipline as
    ``action_skill._load_outcomes`` / ``persona_skill._load_outcomes``."""
    rows: list[dict] = []
    try:
        if not path.exists():
            return rows
        for ln in path.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception:
                continue
    except Exception:
        return rows
    return rows


def analyze(outcomes_path: "Path | str | None" = None,
            oos_only: bool = True) -> dict:
    """End-to-end CLI/import entrypoint: load outcomes, load the deployed
    scorer, run ``news_volume_skill``.

    ``oos_only`` (default True) limits the analysis to the most recent
    20% of records by ``sim_date`` via ``validation.split_outcomes_temporal``
    — the SAME holdout `_train_decision_scorer` reports `oos_rmse`/`oos_ic`
    on, so this metric and the per-cycle ledger describe the same slice.
    A split failure degrades to "use all records" rather than a crash —
    same operational discipline as ``action_skill.analyze``.
    """
    root = Path(__file__).resolve().parent.parent.parent
    if outcomes_path is None:
        outcomes_path = root / "data" / "decision_outcomes.jsonl"
    records = _load_outcomes(Path(outcomes_path))

    slice_label = "all"
    if oos_only and records:
        try:
            from paper_trader.validation import split_outcomes_temporal
            _, oos = split_outcomes_temporal(records, oos_fraction=0.2)
            if oos:
                records = oos
                slice_label = "oos"
        except Exception:
            pass

    try:
        from paper_trader.ml.decision_scorer import DecisionScorer
        scorer = DecisionScorer()
    except Exception as e:
        out = {
            "status": "error",
            "verdict": "INSUFFICIENT_DATA",
            "n_records": 0,
            "by_bucket": {b: {"n": 0, "rank_ic": None, "dir_acc": None,
                              "mean_aligned_return": None,
                              "verdict": "INSUFFICIENT"}
                          for b in _BUCKET_NAMES},
            "hint": f"scorer load failed: {type(e).__name__}",
            "slice": slice_label,
        }
        return out

    rep = news_volume_skill(scorer, records)
    rep["slice"] = slice_label
    return rep


def _cli() -> int:
    """`python3 -m paper_trader.ml.news_volume_skill` — per-bucket OOS rank
    skill over the live ``decision_outcomes.jsonl``. Read-only; never writes
    anything. Exit 0 healthy / insufficient, 2 if any bucket is INVERTED
    (so an operator/cron can branch on it, exactly like action_skill._cli
    / calibration._cli)."""
    rep = analyze()
    print(f"slice={rep.get('slice', 'all')}  "
          f"aligned_outcomes={rep['n_records']}")
    print(f"VERDICT: {rep['verdict']}  ({rep['hint']})")
    print(f"  {'bucket':<10} {'n':>6} {'rank_ic':>9} {'dir_acc':>8} "
          f"{'mean_ret':>9}  verdict")
    for b in _BUCKET_NAMES:
        e = rep["by_bucket"][b]
        ic_s = (f"{e['rank_ic']:>+9.3f}" if e["rank_ic"] is not None
                else f"{'n/a':>9}")
        da_s = (f"{e['dir_acc']:>8.3f}" if e["dir_acc"] is not None
                else f"{'n/a':>8}")
        mr_s = (f"{e['mean_aligned_return']:>+9.2f}"
                if e["mean_aligned_return"] is not None else f"{'n/a':>9}")
        print(f"  {b:<10} {e['n']:>6} {ic_s} {da_s} {mr_s}  {e['verdict']}")
    return 0 if rep["verdict"] != "HAS_INVERTED_BUCKET" else 2


if __name__ == "__main__":
    raise SystemExit(_cli())

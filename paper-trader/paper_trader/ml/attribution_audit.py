"""Attribution-based feature audit for the DecisionScorer — read-only.

The natural skeptical-quant question that ``feature_importance`` (permutation)
answers but with a *destructive* method: which features carry the prediction?
Permutation scrambles a column and re-measures skill — it tells you how much
skill the model would LOSE without each feature. That is one valid answer.

This module asks a *different* question with a non-destructive method: across
the actual outcomes corpus, what does the model's **own per-feature
attribution** ( ``DecisionScorer.feature_contributions`` ) say is driving its
predictions? A feature with a near-zero mean |contribution| across thousands
of records is one the model already isn't using, even if permutation tanks
skill because of some narrow regime — so the two diagnostics are complementary,
not redundant. Concretely:

- ``feature_importance``     — "what would the model lose if I removed feature X?"
- ``attribution_audit``      — "what does the model say it's actually using right now?"
- ``feature_coverage``       — "is feature X varying in the input data at all?"

The combination disambiguates "dead model dimension" (low coverage), "model
ignores it" (high coverage, low attribution), and "carries real skill" (high
coverage, high attribution, high permutation drop) — three actionable states
that no single existing diagnostic separates.

Same operational discipline as ``paper_trader/ml/calibration.py`` /
``feature_importance.py`` / ``feature_coverage.py``: **read-only** — no train,
no ``decision_scorer.pkl`` / ``build_features`` / ``N_FEATURES`` / trade-path
touch, never raises on bad input. Safe to run against the live unattended
continuous loop. Reuses ``decision_scorer.feature_contributions`` (single
source of truth — every numerical claim here is the same Shapley-style
ablation the dashboard's ``/api/scorer-attribution`` already renders, so the
two can never drift).

Output per feature:

  - ``mean_abs_contribution`` — how strongly the model leans on this feature
    on average. The headline rank: features with the largest values are the
    ones actually moving predictions.
  - ``mean_contribution``      — signed mean. A large positive value means the
    feature is systematically *raising* predictions across the corpus (e.g.
    a sector one-hot that always boosts); a near-zero mean with large abs
    contribution means the feature pushes both ways.
  - ``top3_share``             — fraction of records where this feature is in
    the top-3 drivers (by ``|contribution|``). A feature with high top3_share
    is decisive on a meaningful slice — not just a tail driver.
  - ``stdev_contribution``     — variability. A feature with non-trivial mean
    abs but ~0 stdev is essentially constant (a bias), not a discriminator.

Verdict (crisp, threshold-driven so it is exactly testable):

| Verdict | Meaning |
|---------|---------|
| ``INSUFFICIENT_DATA`` | < ``MIN_RECORDS`` analyzable rows |
| ``UNTRAINED``         | scorer pickle missing or untrained — nothing to attribute |
| ``MODEL_INERT``       | every feature's ``mean_abs_contribution`` < ``INERT_MAX_ABS`` (the model is essentially a constant predictor — the gate has no leverage) |
| ``CONCENTRATED``      | a single feature accounts for > ``CONCENTRATED_TOP1_SHARE`` of total |contribution| (the "17-feature MLP" is effectively a 1-feature rule — the documented ``MLP_WORSE_THAN_TRIVIAL`` shape) |
| ``DIVERSIFIED``       | no single feature exceeds the concentration bar — the model spreads its attribution |

```bash
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.attribution_audit
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.attribution_audit --json
```
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

# Thresholds at module scope so tests assert exact verdicts and a tuning
# change is a single, reviewable edit (mirrors the codebase's
# constants-at-module-scope convention used by ``calibration``,
# ``feature_coverage``, ``label_audit`` etc.).
MIN_RECORDS = 30                # below this, per-feature means are noise
INERT_MAX_ABS = 0.10            # a feature contributing <0.1pp on average is effectively dead
CONCENTRATED_TOP1_SHARE = 0.50  # one feature > 50% of total |contribution| → effectively a 1-feature model

OUTCOMES_PATH_DEFAULT = (Path(__file__).resolve().parent.parent.parent
                         / "data" / "decision_outcomes.jsonl")


def _iter_records(path: Path) -> Iterable[dict]:
    """Stream-parse a JSONL outcomes file. Corrupt lines are skipped — a
    single bad line must never break the audit (the ``_inject_and_train``
    per-line-tolerant precedent)."""
    if not path.exists():
        return
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _attribution_for(scorer, rec: dict) -> dict | None:
    """Compute one record's per-feature attribution using the EXACT
    ``feature_contributions`` API the dashboard ``/api/scorer-attribution``
    consumer uses. Returns None on any failure (mirrors the discipline of
    every sibling diagnostic — a per-record fault must never break the
    aggregate)."""
    from paper_trader.ml.decision_scorer import _to_float

    try:
        out = scorer.feature_contributions(
            ml_score=_to_float(rec.get("ml_score"), 0.0),
            rsi=rec.get("rsi"), macd=rec.get("macd"),
            mom5=rec.get("mom5"), mom20=rec.get("mom20"),
            regime_mult=_to_float(rec.get("regime_mult"), 1.0),
            ticker=str(rec.get("ticker") or ""),
            vol_ratio=rec.get("vol_ratio"), bb_pos=rec.get("bb_position"),
            news_urgency=rec.get("news_urgency"),
            news_article_count=rec.get("news_article_count"),
        )
    except Exception:
        return None
    if not out.get("trained") or not out.get("contributions"):
        return None
    return out


def analyze(records: list[dict], scorer=None) -> dict:
    """Aggregate attribution across ``records``. Returns a verdict dict.

    The default ``scorer=None`` constructs a fresh ``DecisionScorer()`` —
    the same lazy-load path every dashboard endpoint uses (single source
    of truth, picks up retrain-rewritten pickles via the cache key).
    Pass an explicit scorer for testing.
    """
    from paper_trader.ml.decision_scorer import (
        DecisionScorer, FEATURE_NAMES, N_FEATURES,
    )

    if scorer is None:
        scorer = DecisionScorer()
    if not getattr(scorer, "is_trained", False):
        return {
            "status": "ok",
            "verdict": "UNTRAINED",
            "n_records": 0,
            "n_analyzed": 0,
            "scorer_n_train": getattr(scorer, "n_train", 0),
            "features": [],
        }

    if not records:
        return {
            "status": "ok",
            "verdict": "INSUFFICIENT_DATA",
            "n_records": 0,
            "n_analyzed": 0,
            "scorer_n_train": int(getattr(scorer, "n_train", 0)),
            "features": [],
        }

    # contributions[i] gathers all per-record contribution values for slot i.
    contribs: list[list[float]] = [[] for _ in range(N_FEATURES)]
    # top3_hits[i] counts records where feature i was in the top-3 drivers
    # by |contribution|.
    top3_hits = [0] * N_FEATURES
    analyzed = 0

    for rec in records:
        out = _attribution_for(scorer, rec)
        if out is None:
            continue
        rows = out["contributions"]
        # Re-key by FEATURE_NAMES so slot indices stay stable even though
        # feature_contributions emits rows sorted by |impact|.
        by_name = {r["feature"]: r for r in rows}
        try:
            ordered_vals = [float(by_name[FEATURE_NAMES[i]]["contribution"])
                            for i in range(N_FEATURES)]
        except KeyError:
            # A FEATURE_NAMES drift would surface here — skip the record
            # rather than fabricate slot alignment.
            continue
        # Skip records whose contributions contain non-finite values — the
        # attribution dict already flags this as off_distribution, but the
        # numeric defensive guard mirrors the _to_float discipline.
        if not all(np.isfinite(v) for v in ordered_vals):
            continue
        for i, v in enumerate(ordered_vals):
            contribs[i].append(v)
        # Top-3 by |contribution|. argsort returns ascending; reverse and
        # take first 3.
        order = np.argsort(np.abs(np.asarray(ordered_vals)))[::-1][:3]
        for i in order:
            top3_hits[int(i)] += 1
        analyzed += 1

    if analyzed < MIN_RECORDS:
        return {
            "status": "ok",
            "verdict": "INSUFFICIENT_DATA",
            "n_records": len(records),
            "n_analyzed": analyzed,
            "scorer_n_train": int(getattr(scorer, "n_train", 0)),
            "features": [],
        }

    feature_rows: list[dict] = []
    for i, name in enumerate(FEATURE_NAMES):
        arr = np.asarray(contribs[i], dtype=np.float64)
        if arr.size == 0:
            mean_abs = 0.0
            mean_sgn = 0.0
            std = 0.0
        else:
            mean_abs = float(np.mean(np.abs(arr)))
            mean_sgn = float(np.mean(arr))
            std = float(np.std(arr))
        feature_rows.append({
            "feature": name,
            "mean_abs_contribution": round(mean_abs, 4),
            "mean_contribution": round(mean_sgn, 4),
            "stdev_contribution": round(std, 4),
            "top3_share": round(top3_hits[i] / analyzed, 4),
        })
    # Sort by mean_abs_contribution descending — the natural rank for a
    # quant scanning "what is the model leaning on".
    feature_rows.sort(key=lambda r: -r["mean_abs_contribution"])

    total_abs = sum(r["mean_abs_contribution"] for r in feature_rows)
    if total_abs <= 0:
        verdict = "MODEL_INERT"
        top1_share = 0.0
    else:
        top1_share = feature_rows[0]["mean_abs_contribution"] / total_abs
        if all(r["mean_abs_contribution"] < INERT_MAX_ABS
               for r in feature_rows):
            verdict = "MODEL_INERT"
        elif top1_share > CONCENTRATED_TOP1_SHARE:
            verdict = "CONCENTRATED"
        else:
            verdict = "DIVERSIFIED"

    return {
        "status": "ok",
        "verdict": verdict,
        "n_records": len(records),
        "n_analyzed": analyzed,
        "scorer_n_train": int(getattr(scorer, "n_train", 0)),
        "total_mean_abs_contribution": round(total_abs, 4),
        "top1_share": round(top1_share, 4),
        "features": feature_rows,
    }


def analyze_path(outcomes_path: Path | str | None = None,
                 oos_only: bool = True) -> dict:
    """Convenience wrapper: load records from disk, optionally restrict to
    the temporal-OOS slice (mirroring ``baseline_compare.analyze`` /
    ``calibration --oos`` — every cross-diagnostic OOS metric reuses
    ``validation.split_outcomes_temporal``, so the slice this audits is
    EXACTLY the slice every sibling read of the same file audits)."""
    if outcomes_path is None:
        outcomes_path = OUTCOMES_PATH_DEFAULT
    p = Path(outcomes_path)
    records = list(_iter_records(p))
    slice_tag = "all"
    if oos_only and records:
        try:
            from paper_trader.validation import split_outcomes_temporal
            _train, records = split_outcomes_temporal(records, oos_fraction=0.2)
            slice_tag = "oos"
        except Exception:
            # Fall back to all-records if validation module unavailable —
            # explicit slice_tag tells the operator what they got.
            slice_tag = "all"
    rep = analyze(records)
    rep["outcomes_path"] = str(p)
    rep["slice"] = slice_tag
    return rep


def _print_report(rep: dict) -> None:
    print(f"verdict: {rep['verdict']}")
    print(f"  outcomes_path: {rep.get('outcomes_path', '(in-memory)')}")
    print(f"  slice: {rep.get('slice', 'in-memory')}")
    print(f"  n_records: {rep['n_records']}  "
          f"n_analyzed: {rep['n_analyzed']}  "
          f"scorer_n_train: {rep['scorer_n_train']}")
    if rep["verdict"] in ("INSUFFICIENT_DATA", "UNTRAINED"):
        return
    print(f"  total mean |contribution|: "
          f"{rep.get('total_mean_abs_contribution', 0):.3f}pp  "
          f"top1 share: {rep.get('top1_share', 0):.1%}")
    print()
    print(f"  {'feature':<22}{'mean_abs':>10}{'mean_sgn':>10}"
          f"{'stdev':>10}{'top3_share':>14}")
    for r in rep["features"]:
        print(f"  {r['feature']:<22}{r['mean_abs_contribution']:>10.3f}"
              f"{r['mean_contribution']:>+10.3f}"
              f"{r['stdev_contribution']:>10.3f}"
              f"{r['top3_share']:>14.1%}")


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.attribution_audit",
        description="Aggregate per-feature attribution across the outcomes "
                    "corpus. Read-only — never trains or writes.",
    )
    p.add_argument("--outcomes",
                   help="Path to a decision_outcomes.jsonl. Defaults to "
                        "data/decision_outcomes.jsonl.")
    p.add_argument("--all", action="store_true",
                   help="Audit the full corpus instead of the temporal-OOS "
                        "slice (default: OOS).")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON.")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    rep = analyze_path(args.outcomes, oos_only=not args.all)

    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
    else:
        _print_report(rep)

    # Exit code convention mirrors sibling diagnostics: 0 = OK,
    # 2 = actionable problem (MODEL_INERT / UNTRAINED), 1 = INSUFFICIENT_DATA.
    v = rep["verdict"]
    if v in ("MODEL_INERT", "UNTRAINED"):
        return 2
    if v == "INSUFFICIENT_DATA":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())

"""LLM-annotation skill diagnostic — do the ENDORSE/CONDEMN labels written by
``run_continuous_backtests._llm_annotate_outcomes`` actually correspond to
better realized 5d forward returns, or is the LLM annotator inert (or worse,
anti-predictive)?

The LLM annotator drives one of the two weighting columns in
``train_scorer``'s sample-weight policy (`{1: 3.0, -1: 0.1, 0: 1.0}` —
``decision_scorer.train_scorer`` line ~658), making it a load-bearing input
to the gate-relevant DecisionScorer. ``sample_weight_audit`` already answers
"does TURNING OFF the LLM column change the trained model's OOS rank-IC", but
that is the **train-time** view of the column's *gradient effect*. It cannot
answer the more primitive **data-quality** question every existing diagnostic
structurally misses: *do the labels themselves have rank-skill on the
realized 5d forward returns the scorer is trained on?*

A `LLM_DIRECTIONAL` verdict means the ENDORSE labels genuinely precede
larger action-aligned returns than the CONDEMN labels — the column is
carrying real signal, the 3.0×/0.1× weight policy is well-grounded.

A `LLM_INERT` verdict means the labels exist but ENDORSE and CONDEMN
realized returns are statistically indistinguishable — the column is
overhead that drives no measurable training advantage (the documented
``sample_weight_audit`` `CURRENT_TIED` finding, but observed in the *data*
rather than only in the trained model's OOS skill).

A `LLM_ANTI_PREDICTIVE` verdict is the actionable red flag — CONDEMN
returns exceed ENDORSE returns, so the 3.0×/0.1× weighting is *inverted*
relative to true signal direction. This is gate-relevant: the deployed
scorer is being told to up-weight noise and down-weight signal.

The most operationally important verdict for the live deployment is
`NO_LABELS_PRODUCED` — the LLM annotation pipeline (``_llm_annotate_outcomes``
in ``run_continuous_backtests.py``) is fully *dark*: ZERO rows in
``decision_outcomes.jsonl`` carry a non-zero ``llm_quality_label``. The
column reaches the trainer as a constant `0` ⇒ `llm_mult` is constant `1.0`
on every record ⇒ the entire LLM annotation subsystem is overhead with no
trainer-side effect. The continuous loop's silent ``except Exception``
swallows the (e.g.) missing-API-key failure — without this diagnostic the
darkness is invisible until an operator manually greps the outcomes file.

Operational discipline mirrors ``news_volume_skill`` / ``action_skill``:
**read-only** (no train, no ``decision_scorer.pkl`` / ``build_features`` /
``N_FEATURES`` / trade path touch, never raises on bad input), so safe to
run against the live unattended continuous loop and cannot break pickle
compatibility.

Verdict ladder (crisp, threshold-driven so it is exactly testable):

| Verdict                | Meaning                                                |
|------------------------|--------------------------------------------------------|
| `NO_LABELS_PRODUCED`   | `n_endorsed + n_condemned == 0` — pipeline dark        |
| `INSUFFICIENT_LABELS`  | min(n_endorsed, n_condemned) < `MIN_PER_GROUP` (=10)   |
| `LLM_ANTI_PREDICTIVE`  | endorsed mean ≤ condemned mean - `RETURN_GAP_GOOD`     |
| `LLM_DIRECTIONAL`      | endorsed mean ≥ condemned mean + `RETURN_GAP_GOOD`     |
| `LLM_INERT`            | |endorsed - condemned| < `RETURN_GAP_GOOD` (both sides exist) |
| `LLM_DIRECTIONAL_WEAK` | gap in `[RETURN_GAP_MIN, RETURN_GAP_GOOD)`             |
| `LLM_ANTI_WEAK`        | gap in `[-RETURN_GAP_GOOD, -RETURN_GAP_MIN)`           |

CLI: ``python3 -m paper_trader.ml.llm_annotation_skill [--json] [--path PATH]``

```bash
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.llm_annotation_skill
cd /home/zeph/paper-trader && python3 -m pytest tests/test_llm_annotation_skill.py -v
```
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from paper_trader.ml.calibration import _spearman
from paper_trader.ml.decision_scorer import _to_float

# Thresholds at module scope (mirrors news_volume_skill / action_skill /
# calibration / persona_skill). A tuning change is one reviewable edit.
MIN_PER_GROUP = 10              # need ≥10 endorsed AND ≥10 condemned for any verdict
MIN_TOTAL_LABELED = 30          # need ≥30 endorsed+condemned overall for IC
# Mean-return gap (percentage points, action-aligned) above which the
# direction is considered measurable. Aligned with `news_volume_skill`'s
# IC_MIN / IC_GOOD pattern but in return-space: 0.5pp ≈ one trading week's
# noise on a leveraged ETF; 1.0pp is a more confident effect.
RETURN_GAP_MIN = 0.5            # weak directional bar (pp action-aligned)
RETURN_GAP_GOOD = 1.0           # strong directional bar (pp action-aligned)


def _aligned_return(record: dict) -> float | None:
    """Return the action-aligned realized 5d forward return for a record,
    or ``None`` when the record is unusable.

    Mirrors ``train_scorer``'s SELL sign-flip convention exactly: a SELL's
    "good" outcome is a NEGATIVE forward return (the price dropped after
    we sold), so flipping the sign aligns the label space across BUY/SELL.
    ``persona_skill`` / ``news_volume_skill`` / ``_oos_rank_metrics`` all
    use this same convention — single source of truth for "goodness of
    the action".

    Pure, total, never raises (the ``_to_float`` NaN-sentinel discipline).
    """
    if not isinstance(record, dict):
        return None
    fr = record.get("forward_return_5d")
    if fr is None:
        return None
    t = _to_float(fr, float("nan"))
    if t != t:                          # NaN → drop
        return None
    if str(record.get("action") or "BUY").upper() == "SELL":
        t = -t
    return float(t)


def _label_for(record: dict) -> int | None:
    """Return the LLM quality label for a record (∈ {-1, 0, +1}), or
    ``None`` when the value is unparseable / out-of-range.

    ``train_scorer`` reads this as ``int(r.get("llm_quality_label") or 0)``;
    we mirror that coercion so this diagnostic and the trainer see the same
    label space. The trainer's permissive ``or 0`` collapses None/null to 0
    — we keep that exact behaviour so the count of "unlabeled" rows here
    matches the count of rows that received the 1.0× ``llm_mult`` at train.

    Pure, total, never raises (`int(False)` → 0, `int(True)` → 1; bools
    pass through the same way the trainer's coercion does).
    """
    if not isinstance(record, dict):
        return None
    raw = record.get("llm_quality_label")
    if raw is None:
        return 0
    try:
        # Bool → int silently produces 0/1; the trainer accepts that, and
        # we mirror it. Anything else that int() can't coerce drops to None.
        v = int(raw)
    except (TypeError, ValueError):
        return None
    if v not in (-1, 0, 1):
        # Out-of-range labels are not part of the trainer's `{-1, 0, 1}`
        # contract — surface them as `None` (drop) rather than letting them
        # contaminate the per-group bucketing.
        return None
    return v


def llm_annotation_skill(records) -> dict:
    """Per-LLM-label realized-return skill over a corpus of outcome rows.

    ``records`` is any iterable of dicts shaped like ``decision_outcomes.jsonl``
    rows (must carry ``action``, ``forward_return_5d``, ``llm_quality_label``).
    Rows with a non-finite forward return, an unparseable label, or that fail
    the action-aligned-return helper are dropped. Pure read; never raises on
    malformed input — degrades to an honest ``status='error'`` or empty
    ``status='insufficient_data'`` envelope.

    Returns a JSON-safe dict::

        {
          "status": "ok" | "insufficient_data" | "error",
          "verdict": <verdict string from the ladder>,
          "n_total": int,            # rows that produced a usable (label, return)
          "n_endorsed": int,         # subset with label == +1
          "n_condemned": int,        # subset with label == -1
          "n_unlabeled": int,        # subset with label == 0
          "endorsed_mean_return": float | None,    # action-aligned, percentage points
          "condemned_mean_return": float | None,
          "unlabeled_mean_return": float | None,
          "endorsed_minus_condemned": float | None,
          "rank_ic": float | None,   # tie-aware Spearman(label, aligned return)
          "hint": str,
        }
    """
    out = {
        "status": "ok",
        "verdict": "NO_LABELS_PRODUCED",
        "n_total": 0,
        "n_endorsed": 0,
        "n_condemned": 0,
        "n_unlabeled": 0,
        "endorsed_mean_return": None,
        "condemned_mean_return": None,
        "unlabeled_mean_return": None,
        "endorsed_minus_condemned": None,
        "rank_ic": None,
        "hint": "",
    }

    try:
        endorsed: list[float] = []
        condemned: list[float] = []
        unlabeled: list[float] = []
        for r in records:
            lbl = _label_for(r)
            if lbl is None:
                continue
            ret = _aligned_return(r)
            if ret is None:
                continue
            if lbl == 1:
                endorsed.append(ret)
            elif lbl == -1:
                condemned.append(ret)
            else:
                unlabeled.append(ret)

        out["n_endorsed"] = len(endorsed)
        out["n_condemned"] = len(condemned)
        out["n_unlabeled"] = len(unlabeled)
        out["n_total"] = len(endorsed) + len(condemned) + len(unlabeled)

        if endorsed:
            out["endorsed_mean_return"] = round(
                float(np.mean(endorsed)), 4)
        if condemned:
            out["condemned_mean_return"] = round(
                float(np.mean(condemned)), 4)
        if unlabeled:
            out["unlabeled_mean_return"] = round(
                float(np.mean(unlabeled)), 4)

        n_lab = len(endorsed) + len(condemned)
        # 1. Pipeline-dark check — the strongest finding for the live state.
        if n_lab == 0:
            out["verdict"] = "NO_LABELS_PRODUCED"
            out["hint"] = (
                "Zero rows carry a non-zero llm_quality_label. The "
                "_llm_annotate_outcomes pipeline (run_continuous_backtests.py) "
                "is dark — likely a missing ANTHROPIC_API_KEY, an unreachable "
                "API, or the regex failing every cycle. The 3.0x/0.1x "
                "llm_mult weight is constant 1.0 in train_scorer; the LLM "
                "annotation subsystem is overhead with no trainer effect."
            )
            return out

        # 2. Bilateral sufficiency — need BOTH groups present.
        if (len(endorsed) < MIN_PER_GROUP or
                len(condemned) < MIN_PER_GROUP or
                n_lab < MIN_TOTAL_LABELED):
            out["status"] = "insufficient_data"
            out["verdict"] = "INSUFFICIENT_LABELS"
            out["hint"] = (
                f"Need ≥{MIN_PER_GROUP} endorsed AND ≥{MIN_PER_GROUP} "
                f"condemned (total ≥{MIN_TOTAL_LABELED}); have "
                f"{len(endorsed)} / {len(condemned)} (total {n_lab}). "
                f"Accumulate more cycles."
            )
            return out

        # 3. Rank-IC on the labeled subset (label ∈ {-1, +1}, aligned return).
        labels_arr = np.array([1.0] * len(endorsed) + [-1.0] * len(condemned),
                              dtype=np.float64)
        rets_arr = np.array(endorsed + condemned, dtype=np.float64)
        ic_raw = float(_spearman(labels_arr, rets_arr))
        rank_ic = round(ic_raw, 4) if ic_raw == ic_raw else None
        out["rank_ic"] = rank_ic

        # 4. Direction verdict from the mean-return gap. Gap = endorsed -
        # condemned (action-aligned). Positive ⇒ LLM is right that endorsed
        # trades outperform condemned ones.
        gap = (out["endorsed_mean_return"] or 0.0) - (
            out["condemned_mean_return"] or 0.0)
        out["endorsed_minus_condemned"] = round(gap, 4)

        if gap >= RETURN_GAP_GOOD:
            out["verdict"] = "LLM_DIRECTIONAL"
            out["hint"] = (
                f"Endorsed trades realize +{gap:.2f}pp higher action-aligned "
                f"return than condemned ones — the LLM annotator carries real "
                f"signal. The 3.0×/0.1× train_scorer weighting is well-grounded."
            )
        elif gap >= RETURN_GAP_MIN:
            out["verdict"] = "LLM_DIRECTIONAL_WEAK"
            out["hint"] = (
                f"Endorsed > condemned by {gap:.2f}pp (between weak and "
                f"strong thresholds). Some signal but not decisive; treat as "
                f"a tie-breaker, not primary."
            )
        elif gap <= -RETURN_GAP_GOOD:
            out["verdict"] = "LLM_ANTI_PREDICTIVE"
            out["hint"] = (
                f"Condemned trades realize +{-gap:.2f}pp HIGHER action-aligned "
                f"return than endorsed — the LLM is inverted. The 3.0×/0.1× "
                f"weighting is upweighting noise and downweighting signal. "
                f"Switch to llm_mult=1.0 in train_scorer immediately."
            )
        elif gap <= -RETURN_GAP_MIN:
            out["verdict"] = "LLM_ANTI_WEAK"
            out["hint"] = (
                f"Condemned > endorsed by {-gap:.2f}pp (weak anti-signal). "
                f"Marginal evidence the LLM column is mis-directing the "
                f"trainer; not yet decisive."
            )
        else:
            out["verdict"] = "LLM_INERT"
            out["hint"] = (
                f"Endorsed and condemned returns differ by only "
                f"{abs(gap):.2f}pp (< {RETURN_GAP_MIN}). The LLM annotation "
                f"carries no measurable directional signal — overhead with "
                f"no trainer-side effect. Mirrors sample_weight_audit's "
                f"CURRENT_TIED finding."
            )

        return out
    except Exception as exc:
        return {
            "status": "error",
            "verdict": "NO_LABELS_PRODUCED",
            "n_total": 0,
            "n_endorsed": 0,
            "n_condemned": 0,
            "n_unlabeled": 0,
            "endorsed_mean_return": None,
            "condemned_mean_return": None,
            "unlabeled_mean_return": None,
            "endorsed_minus_condemned": None,
            "rank_ic": None,
            "hint": f"analysis failed: {type(exc).__name__}: {exc}",
        }


def _load_outcomes(path: Path) -> list[dict]:
    """Robust JSONL load — mirrors ``news_volume_skill._load_outcomes`` /
    ``action_skill._load_outcomes`` exactly. Never raises."""
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


def analyze(outcomes_path: "Path | str | None" = None) -> dict:
    """End-to-end CLI/import entrypoint: load outcomes, run
    ``llm_annotation_skill``. No scorer load required — the analysis is
    pure data, not predict-based (unlike its siblings ``news_volume_skill``
    / ``persona_skill``).

    Returns the same envelope ``llm_annotation_skill`` returns, with an
    extra ``"slice"`` key surfacing the data slice used ("all" — the LLM
    annotation lives in the data, not the model, so the temporal-OOS split
    used by `_train_decision_scorer` does not apply here).
    """
    root = Path(__file__).resolve().parent.parent.parent
    if outcomes_path is None:
        outcomes_path = root / "data" / "decision_outcomes.jsonl"
    records = _load_outcomes(Path(outcomes_path))
    rep = llm_annotation_skill(records)
    rep["slice"] = "all"
    return rep


def _cli() -> int:
    """`python3 -m paper_trader.ml.llm_annotation_skill` — labeled-vs-unlabeled
    realized-return diagnostic over the live ``decision_outcomes.jsonl``.
    Read-only; never writes anything. Exit 0 healthy / insufficient / inert /
    directional, 2 if `LLM_ANTI_PREDICTIVE` (so an operator/cron can branch
    on it, exactly like ``action_skill._cli`` / ``news_volume_skill._cli``).
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.llm_annotation_skill",
        description="Read-only diagnostic: do _llm_annotate_outcomes ENDORSE "
                    "/ CONDEMN labels predict realized 5d forward returns?",
    )
    parser.add_argument("--path", default=None,
                        help="Outcomes JSONL path (default: "
                             "data/decision_outcomes.jsonl).")
    parser.add_argument("--json", action="store_true",
                        help="Emit JSON instead of a text table.")
    args = parser.parse_args()

    rep = analyze(args.path)

    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
        return 2 if rep["verdict"] == "LLM_ANTI_PREDICTIVE" else 0

    print(f"slice={rep.get('slice', 'all')}  "
          f"n_total={rep['n_total']}  "
          f"n_endorsed={rep['n_endorsed']}  "
          f"n_condemned={rep['n_condemned']}  "
          f"n_unlabeled={rep['n_unlabeled']}")
    print(f"VERDICT: {rep['verdict']}")
    if rep["hint"]:
        print(f"  {rep['hint']}")

    if rep["endorsed_mean_return"] is not None or \
            rep["condemned_mean_return"] is not None:
        print()
        print(f"  {'group':<12} {'n':>6} {'mean_aligned_return_%':>22}")
        em = (f"{rep['endorsed_mean_return']:>+22.4f}"
              if rep["endorsed_mean_return"] is not None
              else f"{'n/a':>22}")
        cm = (f"{rep['condemned_mean_return']:>+22.4f}"
              if rep["condemned_mean_return"] is not None
              else f"{'n/a':>22}")
        um = (f"{rep['unlabeled_mean_return']:>+22.4f}"
              if rep["unlabeled_mean_return"] is not None
              else f"{'n/a':>22}")
        print(f"  {'endorsed':<12} {rep['n_endorsed']:>6} {em}")
        print(f"  {'condemned':<12} {rep['n_condemned']:>6} {cm}")
        print(f"  {'unlabeled':<12} {rep['n_unlabeled']:>6} {um}")
        if rep["endorsed_minus_condemned"] is not None:
            print()
            print(f"  endorsed − condemned = "
                  f"{rep['endorsed_minus_condemned']:+.4f}pp")
        if rep["rank_ic"] is not None:
            print(f"  rank_ic(label, return) = "
                  f"{rep['rank_ic']:+.4f}")

    return 2 if rep["verdict"] == "LLM_ANTI_PREDICTIVE" else 0


if __name__ == "__main__":
    raise SystemExit(_cli())

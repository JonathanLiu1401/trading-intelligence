"""Training-corpus composition & OOS-construction audit — read-only.

The natural quant question once `calibration`=WELL_CALIBRATED-in-sample,
`gate_audit`=GATE_INEFFECTIVE, `skill_trend`=NEGATIVE_OOS_SKILL,
`feature_importance`=SIGNAL_GROUNDED, `regime_audit`=REGIME_UNIFORM_NULL,
`baseline_compare`=MLP_NO_BETTER_THAN_TRIVIAL are all on record:

  *Every one of those verdicts trusts the temporal-OOS slice
  `validation.split_outcomes_temporal` carves out of
  `decision_outcomes.jsonl` as a generalization holdout. Is it actually one?*

This is a **read-only diagnostic**. It never trains, never touches
`decision_scorer.pkl`, `decision_outcomes.jsonl`, `build_features`,
`N_FEATURES`, or any trade path — same operational discipline as
`paper_trader/ml/calibration.py` / `gate_audit.py` / `skill_trend.py` /
`regime_audit.py` / `baseline_compare.py`.

**Why this is not any existing tool.** Every sibling diagnostic takes the
corpus *as given* and measures the scorer's skill on it (deciles, gate arms,
ledger trend, feature attribution, regime buckets, trivial baselines).
**None of them characterize the corpus itself, nor validate that the OOS
holdout the loop reports `oos_rmse`/`oos_ic` on is a genuine held-out
draw.** That gap matters here because of how the corpus is produced:

- `MAX_OUTCOMES_FOR_TRAINING = 5000` caps `decision_outcomes.jsonl`.
- Each cycle runs `RUNS_PER_CYCLE = 5` backtests over **one random
  multi-year window**; a multi-year run emits ~1000 BUY/SELL outcomes, so a
  single cycle produces ≈5000 rows — i.e. the cap ≈ one cycle's one window.
- Each backtest run emits decisions across the **whole** window, so when
  `split_outcomes_temporal` sorts the corpus by `sim_date` and holds out the
  latest fraction, every run that contributes to OOS (its late `sim_date`
  decisions) **also contributed to train** (its early `sim_date` decisions).

The consequence: the loop's "temporal OOS holdout" is, given a single-cycle
corpus, the late slice of the *same* backtest runs over the *same* window —
the same personas, the same price-cache, one contiguous regime path. It is a
within-window front/back split, **not** a generalization test against an
unseen window/regime. Every OOS verdict elsewhere in this domain inherits
that construction. This module detects and names it.

Verdict (crisp, threshold-driven so it is exactly testable):

| Verdict | Meaning |
|---------|---------|
| `INSUFFICIENT_DATA` | < `MIN_RECORDS` rows, or the split yields no OOS slice |
| `OOS_NOT_HELD_OUT` | every OOS run_id also appears in train **and** the corpus spans ≤ `NARROW_MAX_RUNS` distinct runs — the loop's `oos_rmse`/`oos_ic`/`calibration --oos`/`regime_audit`/`baseline_compare` OOS metrics are a within-window late-slice of the same ≤~2-cycle draw, not a generalization test |
| `OOS_OVERLAPS_TRAIN` | every OOS run_id also appears in train but the corpus spans many distinct runs (> `NARROW_MAX_RUNS`) — the holdout still shares all its runs with train, milder than a single-draw corpus |
| `OOS_HELD_OUT` | ≥1 OOS run_id is absent from train — there is genuine run-level separation between train and the holdout |

`corpus_breadth` (`SINGLE_DRAW` / `NARROW` / `DIVERSE`) and `regime_mix`
are reported as informational descriptors and are **not** folded into the
verdict (the gate_audit `arm_monotone_fraction` honesty pattern), so the
verdict stays crisply exact-value testable on the run-set relationship
alone.

```bash
cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.corpus_audit
cd /home/zeph/trading-intelligence/paper-trader && python3 -m pytest tests/test_corpus_audit.py -v
```
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# regime_mult → regime label. MUST mirror backtest.py::_ml_decide /
# run_continuous_backtests._compute_decision_outcomes / regime_audit.py
# (bull=1.0, sideways=0.6, bear=0.3, unknown=1.0). `1.0` is honestly
# labeled bull_or_unknown — the stored feature cannot separate a real bull
# from an unknown (both 1.0). Informational here (not in the verdict), so a
# future fourth multiplier degrades to the "unmapped" bucket, never a
# fabricated regime.
REGIME_FROM_MULT: dict[float, str] = {
    0.30: "bear",
    0.60: "sideways",
    1.00: "bull_or_unknown",
}

# Verdict thresholds — module-level so tests assert exact verdicts and a
# tuning change is a single reviewable edit (mirrors calibration.py /
# gate_audit.py / skill_trend.py / regime_audit.py constants-at-module-scope
# convention).
MIN_RECORDS = 30          # need a real sample before any verdict (≈ calibration.MIN_PAIRS)
# RUNS_PER_CYCLE in run_continuous_backtests.py is 5. A corpus of ≤ that many
# distinct runs is at most one cycle's single random window. Kept as a local
# constant (importing run_continuous_backtests pulls heavy deps); breadth is
# informational only, so a drift here cannot corrupt the verdict.
RUNS_PER_CYCLE_HINT = 5
# ≤ this many distinct runs ≈ ≤ 2 cycles ≈ ≤ 2 random windows: a corpus this
# narrow makes a run-subset OOS holdout a within-window split, not a
# generalization test. Drives the OOS_NOT_HELD_OUT vs OOS_OVERLAPS_TRAIN split.
NARROW_MAX_RUNS = 10


def load_outcomes(path: Path | str) -> list[dict]:
    """Robust JSONL load of ``decision_outcomes.jsonl``. Skips unparseable /
    non-dict lines and never raises — a missing/corrupt file yields ``[]`` so
    callers degrade to ``INSUFFICIENT_DATA`` rather than crashing (the file is
    best-effort by construction; a reader of it must be too — the
    `skill_trend.load_skill_ledger` discipline)."""
    p = Path(path)
    rows: list[dict] = []
    try:
        if not p.exists():
            return rows
        for ln in p.read_text().splitlines():
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


def _regime_label(regime_mult) -> str:
    """Decode a stored ``regime_mult`` into a regime label, or ``"unmapped"``
    when it is missing / non-finite / not one of the three known multipliers.
    Rounds to 2 decimals so a JSON float like ``0.6000000001`` resolves."""
    try:
        v = float(regime_mult)
    except (TypeError, ValueError):
        return "unmapped"
    if not np.isfinite(v):
        return "unmapped"
    return REGIME_FROM_MULT.get(round(v, 2), "unmapped")


def _run_id(rec: dict):
    """Extract a hashable run identifier, or ``None`` when absent.

    A run_id of ``0`` is a legitimate id and must NOT be coerced to None, so
    test explicit presence rather than truthiness."""
    rid = rec.get("run_id")
    if rid is None:
        return None
    return rid


def corpus_audit_report(records: list[dict], oos_fraction: float = 0.2) -> dict:
    """Characterize the training corpus and validate the OOS construction.

    ``records`` is the ``decision_outcomes.jsonl`` row shape. ``oos_fraction``
    is forwarded to ``validation.split_outcomes_temporal`` — the EXACT split
    ``run_continuous_backtests._train_decision_scorer`` uses for
    ``oos_rmse``/``oos_ic`` — so this audit describes the *same* holdout the
    ledger's scalar OOS metrics were measured on (single source of truth: a
    split mismatch here would make the verdict describe a different slice than
    the one every other OOS tool reports on).

    Returns a JSON-safe dict. Never raises — a split fault degrades to
    ``INSUFFICIENT_DATA`` rather than crashing near the unattended loop.
    """
    recs = list(records or [])
    n = len(recs)
    out: dict = {
        "status": "ok",
        "verdict": "INSUFFICIENT_DATA",
        "n": n,
        "n_distinct_run_ids": 0,
        "run_id_counts": [],
        "sim_date_min": None,
        "sim_date_max": None,
        "n_distinct_sim_dates": 0,
        "regime_mix": {},
        "dominant_regime_fraction": None,
        "corpus_breadth": None,
        "likely_single_cycle": None,
        "train_n": 0,
        "oos_n": 0,
        "n_train_run_ids": 0,
        "n_oos_run_ids": 0,
        "oos_run_ids_in_train": 0,
        "oos_run_ids_not_in_train": 0,
        "oos_shares_all_runs_with_train": None,
        "train_sim_date_max": None,
        "oos_sim_date_min": None,
        "hint": "",
    }

    # ── Corpus composition (computed even when too thin to verdict, so the
    #    descriptor fields are always populated for the operator) ───────────
    run_counts: dict = {}
    for r in recs:
        rid = _run_id(r)
        run_counts[rid] = run_counts.get(rid, 0) + 1
    out["n_distinct_run_ids"] = len(run_counts)
    out["run_id_counts"] = sorted(
        ([rid, c] for rid, c in run_counts.items()),
        key=lambda kc: (kc[0] is None, kc[0]),
    )

    sim_dates = [str(r.get("sim_date")) for r in recs if r.get("sim_date")]
    if sim_dates:
        out["sim_date_min"] = min(sim_dates)
        out["sim_date_max"] = max(sim_dates)
        out["n_distinct_sim_dates"] = len(set(sim_dates))

    mix: dict = {}
    for r in recs:
        lbl = _regime_label(r.get("regime_mult"))
        mix[lbl] = mix.get(lbl, 0) + 1
    out["regime_mix"] = dict(sorted(mix.items()))
    if n:
        out["dominant_regime_fraction"] = round(max(mix.values()) / n, 4)

    n_runs = out["n_distinct_run_ids"]
    if n_runs <= RUNS_PER_CYCLE_HINT:
        out["corpus_breadth"] = "SINGLE_DRAW"
    elif n_runs <= NARROW_MAX_RUNS:
        out["corpus_breadth"] = "NARROW"
    else:
        out["corpus_breadth"] = "DIVERSE"
    out["likely_single_cycle"] = n_runs <= RUNS_PER_CYCLE_HINT

    if n < MIN_RECORDS:
        out["hint"] = f"need ≥{MIN_RECORDS} records, have {n}"
        return out

    # ── Apply the EXACT loop split and validate the holdout construction ──
    try:
        from paper_trader.validation import split_outcomes_temporal
        train, oos = split_outcomes_temporal(recs, oos_fraction=oos_fraction)
    except Exception as exc:
        out["hint"] = f"split unavailable ({type(exc).__name__}) — cannot assess OOS"
        return out

    out["train_n"] = len(train)
    out["oos_n"] = len(oos)
    if not oos:
        out["hint"] = (f"split produced no OOS slice (train_n={len(train)}) — "
                       f"need oos_fraction>0 and ≥5 records")
        return out

    train_runs = {_run_id(r) for r in train}
    oos_runs = {_run_id(r) for r in oos}
    in_train = oos_runs & train_runs
    not_in_train = oos_runs - train_runs
    out["n_train_run_ids"] = len(train_runs)
    out["n_oos_run_ids"] = len(oos_runs)
    out["oos_run_ids_in_train"] = len(in_train)
    out["oos_run_ids_not_in_train"] = len(not_in_train)
    out["oos_shares_all_runs_with_train"] = not not_in_train

    tr_sd = sorted(str(r.get("sim_date")) for r in train if r.get("sim_date"))
    oo_sd = sorted(str(r.get("sim_date")) for r in oos if r.get("sim_date"))
    out["train_sim_date_max"] = tr_sd[-1] if tr_sd else None
    out["oos_sim_date_min"] = oo_sd[0] if oo_sd else None

    # Verdict — solely the run-set relationship between train and the OOS
    # holdout, gated by corpus breadth so a many-window corpus that merely
    # overlaps reads milder than a single-draw corpus.
    if not not_in_train:
        if n_runs <= NARROW_MAX_RUNS:
            out["verdict"] = "OOS_NOT_HELD_OUT"
            out["hint"] = (
                f"all {len(oos_runs)} OOS run_ids appear in train and the "
                f"corpus spans only {n_runs} distinct run(s) "
                f"(≤{NARROW_MAX_RUNS}, ≈≤2 cycles/windows) — the loop's "
                f"oos_rmse/oos_ic and calibration --oos / regime_audit / "
                f"baseline_compare OOS verdicts are a within-window late "
                f"slice of the SAME backtest runs (train sim_date ≤ "
                f"{out['train_sim_date_max']}, OOS ≥ "
                f"{out['oos_sim_date_min']}), NOT a generalization test"
            )
        else:
            out["verdict"] = "OOS_OVERLAPS_TRAIN"
            out["hint"] = (
                f"all {len(oos_runs)} OOS run_ids appear in train but the "
                f"corpus spans {n_runs} distinct runs (>{NARROW_MAX_RUNS}) — "
                f"the holdout shares every run with train (no run-level "
                f"separation) yet draws on many windows"
            )
    else:
        out["verdict"] = "OOS_HELD_OUT"
        out["hint"] = (
            f"{len(not_in_train)} of {len(oos_runs)} OOS run_ids are absent "
            f"from train — there is genuine run-level separation between "
            f"train and the holdout"
        )
    return out


def analyze(outcomes_path: Path | str, oos_fraction: float = 0.2) -> dict:
    """Load ``decision_outcomes.jsonl`` and return the full audit report."""
    return corpus_audit_report(load_outcomes(outcomes_path),
                               oos_fraction=oos_fraction)


def _cli() -> int:
    """``python3 -m paper_trader.ml.corpus_audit`` — read-only audit of the
    live ``decision_outcomes.jsonl`` corpus + OOS-holdout construction.

    Exits 2 on ``OOS_NOT_HELD_OUT`` so a cron/CI guard can branch on it
    (the regime_audit / baseline_compare CLI-exit-code convention)."""
    root = Path(__file__).resolve().parent.parent.parent
    outcomes = root / "data" / "decision_outcomes.jsonl"
    rep = analyze(outcomes)
    print(f"VERDICT: {rep['verdict']}  ({rep['hint']})")
    print(f"  n={rep['n']}  distinct_run_ids={rep['n_distinct_run_ids']}  "
          f"breadth={rep['corpus_breadth']}  "
          f"likely_single_cycle={rep['likely_single_cycle']}")
    print(f"  sim_date {rep['sim_date_min']} → {rep['sim_date_max']}  "
          f"({rep['n_distinct_sim_dates']} distinct)")
    print(f"  regime_mix={rep['regime_mix']}  "
          f"dominant={rep['dominant_regime_fraction']}")
    print(f"  split: train_n={rep['train_n']} ({rep['n_train_run_ids']} runs)  "
          f"oos_n={rep['oos_n']} ({rep['n_oos_run_ids']} runs)")
    print(f"  oos_run_ids in_train={rep['oos_run_ids_in_train']}  "
          f"not_in_train={rep['oos_run_ids_not_in_train']}  "
          f"shares_all={rep['oos_shares_all_runs_with_train']}")
    print(f"  train sim_date ≤ {rep['train_sim_date_max']}  "
          f"oos sim_date ≥ {rep['oos_sim_date_min']}")
    return 2 if rep["verdict"] == "OOS_NOT_HELD_OUT" else 0


if __name__ == "__main__":
    raise SystemExit(_cli())

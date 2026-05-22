"""Training-corpus *diversity* & gate-eligibility audit — read-only.

The natural quant question once `corpus_audit`=OOS_NOT_HELD_OUT,
`baseline_compare`=MLP_NO_BETTER_THAN_TRIVIAL, `skill_trend`=NEGATIVE_OOS_SKILL
are all on record:

  *The conviction gate (`_ml_decide`, invariant #5) only engages once the
  deployed scorer's `n_train >= 500`. `decision_outcomes.jsonl` holds
  thousands of rows. So why does the deployed pickle keep reporting
  `n_train` in the low hundreds — i.e. why is the gate asleep?*

Two effects can drive the gate-relevant `n_train` far below the raw row
count, and this module measures both:

1. **Dedup collapse.** `train_scorer` collapses every
   `(ticker, sim_date, action)` triple to a single record (keeping the
   highest-`return_pct` copy), because the same ticker bought repeatedly on
   the same `sim_date` across runs — or twice in one run's intraday loop —
   carries identical features and an identical 5-day label. When
   `MAX_DECISIONS_PER_DAY=10` keeps re-picking the same momentum names the
   raw-to-distinct ratio can run many-fold; `dedup_ratio` quantifies it.
2. **Deployment lag.** The deployed pickle's `n_train` only advances when
   the loop actually retrains. A dormant or retrain-failing loop leaves a
   stale low-`n_train` pickle even while `decision_outcomes.jsonl` grows
   well past 500 distinct samples. Comparing this module's `trainable_n`
   against the deployed pickle's `n_train` exposes that gap directly.

Either way the conviction gate (`_ml_decide`, invariant #5 — engages only at
`n_train >= 500`) can sit asleep while the operator believes "thousands of
outcomes" should have woken it. This module reports the *true* gate-relevant
count `train_scorer` would pickle from the current corpus.

**Why this is not any existing tool.** `corpus_audit` validates the
*structural* honesty of the temporal-OOS split (does the holdout share
`run_id`s with train?). `outcome_data_quality` counts conflicting duplicates
keyed by `(run_id, sim_date, ticker, action)` — a **run_id-inclusive** key,
so it cannot see the cross-run collapse `train_scorer`'s **run_id-free** key
performs. `sample_weight_audit` / `gate_audit` / `skill_trend` measure skill
on the corpus as given. **None of them answer "after `train_scorer`'s own
dedup, how many distinct samples will the gate see, and is that ≥ 500?"**
That is exactly the number that decides whether the conviction gate is alive
or dead, and this module is the only place it is surfaced.

This is a **read-only diagnostic**. It never trains, never touches
`decision_scorer.pkl`, `decision_outcomes.jsonl`, `build_features`,
`N_FEATURES`, or any trade path — the same operational discipline as
`corpus_audit.py` / `outcome_data_quality.py` / `baseline_compare.py`.

`analyze()` faithfully reproduces the loop's pipeline — last
`MAX_OUTCOMES_FOR_TRAINING` rows → `split_outcomes_temporal` 80/20 → the
train slice handed to `train_scorer` — so the reported `trainable_n` equals
the deployed pickle's `n_train` by construction (a no-drift cross-check, not
a re-derivation).

Verdict (crisp, threshold-driven so it is exactly testable):

| Verdict | Meaning |
|---------|---------|
| `INSUFFICIENT_DATA` | < `MIN_RECORDS` distinct samples — too thin to characterize |
| `GATE_STARVED` | `trainable_n < GATE_MIN_N` — `train_scorer` will pickle an `n_train` below the gate floor; the conviction gate stays asleep |
| `GATE_ELIGIBLE` | `trainable_n >= GATE_MIN_N` — the next retrain produces a gate-active scorer |

```bash
cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.corpus_diversity
cd /home/zeph/trading-intelligence/paper-trader && python3 -m pytest tests/test_corpus_diversity.py -v
```
"""
from __future__ import annotations

import json
import math
from pathlib import Path

# The `_ml_decide` conviction gate (CLAUDE.md invariant #5) engages only once
# the deployed scorer's `n_train >= 500`. This is the single number the whole
# module exists to predict. Module-level so a tests can assert exact verdicts
# and a threshold change is one reviewable edit (the corpus_audit convention).
GATE_MIN_N = 500
# `train_scorer` returns `insufficient_after_dedup` below 30 distinct records.
# Mirror it so a too-thin corpus reads INSUFFICIENT_DATA, not a fake verdict.
MIN_RECORDS = 30
# `run_continuous_backtests.MAX_OUTCOMES_FOR_TRAINING`. Kept as a local hint
# (importing run_continuous_backtests pulls heavy deps — the corpus_audit
# `RUNS_PER_CYCLE_HINT` precedent). Only used by `analyze()` to reproduce the
# loop's trainer-tail cap; the pure core never reads it.
MAX_OUTCOMES_FOR_TRAINING_HINT = 5000
# `_train_decision_scorer` holds out the latest 20% as OOS and trains on the
# remaining 80%. The deployed `n_train` is the dedup of THAT 80% slice.
OOS_FRACTION = 0.2


def load_outcomes(path: Path | str) -> list[dict]:
    """Robust JSONL load of ``decision_outcomes.jsonl``. Skips unparseable /
    non-dict lines and never raises — a missing/corrupt file yields ``[]`` so
    callers degrade to ``INSUFFICIENT_DATA`` rather than crashing (the file is
    best-effort by construction; a reader of it must be too — the
    ``corpus_audit.load_outcomes`` discipline)."""
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


def _dedup_key(rec: dict) -> tuple:
    """The EXACT key ``train_scorer`` dedups on:
    ``(ticker, sim_date, action.upper())`` — run_id-free, so two runs that
    bought the same ticker on the same calendar day collapse to one sample.
    Mirrors ``decision_scorer.train_scorer`` byte-for-byte (single source of
    truth: a drift here would make ``trainable_n`` describe a different
    dedup than the one the gate actually sees)."""
    return (
        str(rec.get("ticker") or ""),
        str(rec.get("sim_date") or ""),
        str(rec.get("action") or "BUY").upper(),
    )


def _return_pct(rec: dict) -> float:
    """``return_pct`` as a float (0.0 on missing/non-numeric) — the tie-break
    ``train_scorer`` uses when two records share a dedup key (keep the
    highest-return-run copy)."""
    try:
        v = float(rec.get("return_pct"))
    except (TypeError, ValueError):
        return 0.0
    return v if math.isfinite(v) else 0.0


def _has_valid_label(rec: dict) -> bool:
    """True iff ``forward_return_5d`` survives ``train_scorer``'s label
    validation — i.e. it is a finite float. ``bool`` / ``None`` /
    non-castable / NaN / inf are all dropped by the trainer before fitting,
    so they do NOT count toward the gate-relevant ``n_train``. Mirrors the
    ``train_scorer`` per-record drop logic exactly."""
    fr = rec.get("forward_return_5d")
    if isinstance(fr, bool) or fr is None:
        return False
    try:
        fr = float(fr)
    except (TypeError, ValueError):
        return False
    return math.isfinite(fr)


def _dedup(records: list[dict]) -> list[dict]:
    """Apply ``train_scorer``'s dedup: one record per ``_dedup_key``, keeping
    the highest-``return_pct`` copy. Returns the surviving records."""
    seen: dict[tuple, dict] = {}
    for r in records:
        k = _dedup_key(r)
        if k not in seen or _return_pct(r) > _return_pct(seen[k]):
            seen[k] = r
    return list(seen.values())


def corpus_diversity_report(records: list[dict]) -> dict:
    """Characterize the diversity of a record set and predict the gate-relevant
    ``n_train`` ``train_scorer`` would pickle from it.

    ``records`` is the exact set ``train_scorer`` receives (for the live loop
    that is the 80% temporal-train slice of the trainer tail — ``analyze()``
    feeds it that). The pure core makes no assumption about tail/split; it
    only replicates ``train_scorer``'s dedup + label validation.

    Returns a JSON-safe dict. Never raises.
    """
    recs = [r for r in (records or []) if isinstance(r, dict)]
    raw_n = len(recs)
    out: dict = {
        "status": "ok",
        "verdict": "INSUFFICIENT_DATA",
        "raw_n": raw_n,
        "distinct_n": 0,
        "trainable_n": 0,
        "n_dropped_bad_label": 0,
        "dedup_ratio": None,
        "distinct_tickers": 0,
        "distinct_sim_dates": 0,
        "distinct_run_ids": 0,
        "action_mix": {},
        "top_collisions": [],
        "gate_min_n": GATE_MIN_N,
        "gate_eligible": False,
        "gate_shortfall": GATE_MIN_N,
        "hint": "",
    }
    if raw_n == 0:
        out["hint"] = "no records — empty or missing corpus"
        return out

    # Per-key collision counts on the RAW set (before dedup) — this is the
    # direct evidence of WHY dedup collapses the corpus.
    key_counts: dict[tuple, int] = {}
    for r in recs:
        k = _dedup_key(r)
        key_counts[k] = key_counts.get(k, 0) + 1

    deduped = _dedup(recs)
    distinct_n = len(deduped)
    trainable = [r for r in deduped if _has_valid_label(r)]
    trainable_n = len(trainable)

    out["distinct_n"] = distinct_n
    out["trainable_n"] = trainable_n
    out["n_dropped_bad_label"] = distinct_n - trainable_n
    out["dedup_ratio"] = round(raw_n / distinct_n, 3) if distinct_n else None

    # Diversity descriptors computed on the deduped set — the distinct samples
    # the model actually fits, not the collision-inflated raw count.
    out["distinct_tickers"] = len({_dedup_key(r)[0] for r in deduped})
    out["distinct_sim_dates"] = len({_dedup_key(r)[1] for r in deduped})
    run_ids = set()
    for r in deduped:
        rid = r.get("run_id")
        if rid is not None:
            run_ids.add(rid)
    out["distinct_run_ids"] = len(run_ids)

    action_mix: dict[str, int] = {}
    for r in deduped:
        a = _dedup_key(r)[2]
        action_mix[a] = action_mix.get(a, 0) + 1
    out["action_mix"] = dict(sorted(action_mix.items()))

    # Worst collision keys (descending count) — the operator sees exactly
    # which ticker/date triples the corpus is wasting rows on.
    out["top_collisions"] = [
        {"ticker": k[0], "sim_date": k[1], "action": k[2], "count": c}
        for k, c in sorted(key_counts.items(), key=lambda kc: -kc[1])
        if c > 1
    ][:10]

    out["gate_eligible"] = trainable_n >= GATE_MIN_N
    out["gate_shortfall"] = max(0, GATE_MIN_N - trainable_n)

    if distinct_n < MIN_RECORDS:
        out["verdict"] = "INSUFFICIENT_DATA"
        out["hint"] = (
            f"only {distinct_n} distinct (ticker,sim_date,action) samples "
            f"from {raw_n} raw rows — need ≥{MIN_RECORDS} to characterize"
        )
        return out

    if trainable_n < GATE_MIN_N:
        out["verdict"] = "GATE_STARVED"
        out["hint"] = (
            f"{raw_n} raw rows dedup to {distinct_n} distinct samples "
            f"({out['dedup_ratio']}× collision); {trainable_n} carry a valid "
            f"5d label — train_scorer will pickle n_train={trainable_n}, "
            f"{out['gate_shortfall']} short of the {GATE_MIN_N} gate floor. "
            f"The conviction gate stays asleep. Raise RUNS_PER_CYCLE or "
            f"diversify personas/windows so distinct (ticker,sim_date) "
            f"coverage grows."
        )
    else:
        out["verdict"] = "GATE_ELIGIBLE"
        out["hint"] = (
            f"{raw_n} raw rows dedup to {distinct_n} distinct samples; "
            f"{trainable_n} carry a valid 5d label — train_scorer will "
            f"pickle n_train={trainable_n} ≥ {GATE_MIN_N}, so the next "
            f"retrain produces a gate-active scorer"
        )
    return out


def analyze(outcomes_path: Path | str,
            max_outcomes: int = MAX_OUTCOMES_FOR_TRAINING_HINT,
            oos_fraction: float = OOS_FRACTION) -> dict:
    """Load ``decision_outcomes.jsonl`` and reproduce the loop's trainer
    pipeline so the report's ``trainable_n`` equals the deployed pickle's
    ``n_train``.

    Pipeline (mirrors ``run_continuous_backtests._train_decision_scorer``):
    last ``max_outcomes`` rows → ``split_outcomes_temporal`` 80/20 → the
    train slice handed to ``train_scorer``. The diversity verdict is computed
    on that train slice. ``file_raw_n`` / ``tail_n`` / ``train_slice_n`` are
    added as context so the operator sees the whole funnel.

    Never raises — a split-module fault degrades to a report computed on the
    untrimmed tail (honest, just not split-faithful), with the fault noted.
    """
    rows = load_outcomes(outcomes_path)
    file_raw_n = len(rows)
    tail = rows[-max_outcomes:] if max_outcomes and max_outcomes > 0 else rows
    train_slice = tail
    split_note = ""
    try:
        from paper_trader.validation import split_outcomes_temporal
        train_slice, _oos = split_outcomes_temporal(tail, oos_fraction=oos_fraction)
    except Exception as exc:  # pragma: no cover - defensive
        split_note = (f" (temporal split unavailable: {type(exc).__name__} — "
                      f"report computed on the untrimmed trainer tail)")

    rep = corpus_diversity_report(train_slice)
    rep["file_raw_n"] = file_raw_n
    rep["tail_n"] = len(tail)
    rep["train_slice_n"] = len(train_slice)
    if split_note:
        rep["hint"] = (rep.get("hint") or "") + split_note
    return rep


def _cli() -> int:
    """``python3 -m paper_trader.ml.corpus_diversity`` — read-only audit of
    whether the live ``decision_outcomes.jsonl`` corpus will produce a
    gate-active scorer.

    Exits 2 on ``GATE_STARVED`` so a cron/CI guard can branch on it (the
    ``corpus_audit`` / ``baseline_compare`` CLI-exit-code convention)."""
    import sys

    root = Path(__file__).resolve().parent.parent.parent
    outcomes = root / "data" / "decision_outcomes.jsonl"
    rep = analyze(outcomes)

    if "--json" in sys.argv[1:]:
        print(json.dumps(rep, indent=2, sort_keys=True))
        return 2 if rep["verdict"] == "GATE_STARVED" else 0

    print(f"VERDICT: {rep['verdict']}  ({rep['hint']})")
    print(f"  file_raw_n={rep.get('file_raw_n')}  tail_n={rep.get('tail_n')}  "
          f"train_slice_n={rep.get('train_slice_n')}")
    print(f"  raw_n={rep['raw_n']}  distinct_n={rep['distinct_n']}  "
          f"trainable_n={rep['trainable_n']}  "
          f"dedup_ratio={rep['dedup_ratio']}×")
    print(f"  gate: n_train≈{rep['trainable_n']}  floor={rep['gate_min_n']}  "
          f"eligible={rep['gate_eligible']}  shortfall={rep['gate_shortfall']}")
    print(f"  distinct_tickers={rep['distinct_tickers']}  "
          f"distinct_sim_dates={rep['distinct_sim_dates']}  "
          f"distinct_run_ids={rep['distinct_run_ids']}")
    print(f"  action_mix={rep['action_mix']}  "
          f"dropped_bad_label={rep['n_dropped_bad_label']}")
    if rep["top_collisions"]:
        print("  worst collisions (raw rows per dedup key):")
        for c in rep["top_collisions"][:5]:
            print(f"    {c['ticker']:<8} {c['sim_date']} {c['action']:<4} "
                  f"×{c['count']}")
    return 2 if rep["verdict"] == "GATE_STARVED" else 0


if __name__ == "__main__":
    raise SystemExit(_cli())

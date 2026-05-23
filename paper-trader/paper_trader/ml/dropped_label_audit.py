"""Audit of WHY training rows are dropped from the DecisionScorer training set.

This is a **read-only diagnostic**. It never trains, never touches
``decision_scorer.pkl``, ``decision_outcomes.jsonl``, ``build_features``,
``N_FEATURES``, or any trade path — same operational discipline as the
sibling ``paper_trader.ml.outcome_data_quality`` / ``corpus_audit`` /
``skill_trend`` modules.

**Why this is not ``outcome_data_quality``.** ``outcome_data_quality``
inspects the JSONL file as data: it counts ``null`` / ``exact_zero`` /
``extreme`` target values and conflicts between duplicate keys. It does
NOT replay the trainer's actual rejection logic. As a result, an operator
seeing ``n_label_dropped=N`` in the cycle skill ledger (the pass #16
wiring) has no way to find WHICH rows the trainer dropped or WHY — only
the count.

This module replays ``decision_scorer.train_scorer``'s *exact* label-
validation loop (the bool / None / float-cast / ``math.isfinite`` checks
in the same order) and classifies each rejected row by the FIRST trigger
that fires:

| Reason | Trigger (first match wins) |
|---|---|
| ``missing_key`` | ``"forward_return_5d"`` not present in the row dict |
| ``explicit_null`` | value is JSON ``null`` (Python ``None``) |
| ``bool_value`` | value is Python ``bool`` (``train_scorer`` excludes bool explicitly because ``True``/``False`` are int-subclasses that would otherwise become ``1.0``/``0.0`` labels) |
| ``unparseable`` | ``float(value)`` raises ``TypeError`` / ``ValueError`` (e.g. a string ``"n/a"``) |
| ``nan`` | parses to ``float('nan')`` |
| ``positive_inf`` | parses to ``+float('inf')`` |
| ``negative_inf`` | parses to ``-float('inf')`` |

Crisp threshold-driven verdict:

| Verdict | Trigger |
|---|---|
| ``INSUFFICIENT_DATA`` | < ``MIN_ROWS`` parsed rows in the trainer's tail |
| ``CLEAN`` | zero rows would be dropped |
| ``LOW_DROP_RATE`` | drop rate ≤ ``LOW_RATE`` (default 0.5%) — corpus is fine |
| ``ELEVATED_DROP_RATE`` | drop rate > ``LOW_RATE`` — investigate |
| ``HIGH_DROP_RATE`` | drop rate > ``HIGH_RATE`` (default 5%) — corruption likely |

The CLI prints sample dropped rows for each reason (capped via
``--samples``) so an operator can immediately see the actual content of
the offending records and grep them out of ``decision_outcomes.jsonl``
if needed.

The single-source-of-truth discipline: this module imports the SAME
``MAX_OUTCOMES_FOR_TRAINING`` cap from ``run_continuous_backtests`` so
the audited tail matches what the trainer ACTUALLY sees, never drifts
from the trainer's window.

```bash
cd /home/zeph/trading-intelligence/paper-trader
python3 -m paper_trader.ml.dropped_label_audit
python3 -m paper_trader.ml.dropped_label_audit --json
python3 -m paper_trader.ml.dropped_label_audit --samples 5
```

Exit code: 0 when ``CLEAN`` or ``LOW_DROP_RATE``, 1 otherwise — so shell
callers can gate on ``$?`` the same way ``host_guard`` / ``scorer_health``
CLIs do.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

# Single source of truth for the tail size the trainer ACTUALLY sees.
# A divergence between this module's audited window and the trainer's
# window would make the report unrepresentative — e.g., a corruption
# that only sits in the older 4000 rows would be invisible to the
# trainer (it only sees the most-recent 5000) but ALSO invisible to a
# diagnostic that audits the whole file. Importing keeps the two paths
# in lockstep across any future cap change.
try:
    from run_continuous_backtests import MAX_OUTCOMES_FOR_TRAINING
except Exception:
    # Fallback to the documented default. Best-effort: if the import
    # fails (test isolation, packaging change), the audit still runs
    # against the same observed-tail size used by the deployed loop.
    MAX_OUTCOMES_FOR_TRAINING = 5000

DEFAULT_OUTCOMES_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "decision_outcomes.jsonl"

# Verdict thresholds. Tuned so a single bad row in a 5000-row tail
# (0.02%) reads CLEAN, a handful of bad rows (≤25 / 0.5%) reads
# LOW_DROP_RATE (acceptable noise), more than that signals corruption
# worth investigating.
MIN_ROWS = 30
LOW_RATE = 0.005   # 0.5%
HIGH_RATE = 0.05   # 5%

# All rejection reasons in the precedence order ``train_scorer`` checks
# them — same order as the audit loop below so the per-reason counts
# are exhaustive and mutually exclusive (no row is counted twice).
REASONS = (
    "missing_key",
    "explicit_null",
    "bool_value",
    "unparseable",
    "nan",
    "positive_inf",
    "negative_inf",
)


def _classify_drop(row: dict) -> str | None:
    """Return the rejection-reason for this row, or None if the trainer
    would accept it.

    Mirrors ``train_scorer``'s validation order EXACTLY:

    1. ``r.get("forward_return_5d")`` — None if key is missing OR if the
       JSON value is explicitly null. Distinguish via ``"forward_return_5d"
       in r``.
    2. ``isinstance(fr_raw, bool)`` — trainer rejects bools before the
       None check (the source uses ``if isinstance(fr_raw, bool) or
       fr_raw is None:``; ``isinstance`` is evaluated first by `or`).
    3. ``float(fr_raw)`` — ``TypeError`` / ``ValueError`` on a non-numeric
       value.
    4. ``math.isfinite(fr)`` — partition non-finite into NaN / +inf / -inf
       so an operator can tell ``math.nan`` corruption (e.g. a divide-by-
       zero outcome) from ``math.inf`` overflow (e.g. a 0-price denom).

    Pure, total, never raises. The classifier and the trainer's loop are
    defined against the SAME 4 checks in the SAME order, so any row this
    function flags would ALSO be dropped by ``train_scorer`` (and vice
    versa). Verified by ``test_dropped_label_audit::TestParity``.
    """
    if "forward_return_5d" not in row:
        return "missing_key"
    fr_raw = row.get("forward_return_5d")
    # bool BEFORE None — see step 2 above. ``isinstance(True, bool)`` is
    # True; ``isinstance(None, bool)`` is False, so this branch is
    # mutually exclusive with the None branch.
    if isinstance(fr_raw, bool):
        return "bool_value"
    if fr_raw is None:
        return "explicit_null"
    try:
        fr = float(fr_raw)
    except (TypeError, ValueError):
        return "unparseable"
    if math.isfinite(fr):
        return None  # would be accepted by the trainer
    if math.isnan(fr):
        return "nan"
    return "positive_inf" if fr > 0 else "negative_inf"


def analyze(
    outcomes_path: "Path | str | None" = None,
    *,
    samples_per_reason: int = 3,
    tail: int = MAX_OUTCOMES_FOR_TRAINING,
) -> dict:
    """Read the most-recent ``tail`` rows from ``outcomes_path`` and
    classify each according to whether ``train_scorer`` would drop it.

    Returns a JSON-safe dict ``{ status, verdict, n_total, n_dropped,
    drop_rate, by_reason: {...}, samples: {reason: [row, ...]}, slice,
    path }``. Best-effort and total: every fault path returns a status
    string instead of raising (a diagnostic CLI must never crash a
    shell-script gate; same discipline as ``baseline_compare`` /
    ``calibration``).

    ``samples_per_reason`` caps the number of full row dicts kept per
    reason (so a 5%-drop corpus doesn't produce a 250-row sample
    payload). The first ``samples_per_reason`` rows encountered are
    kept — they reflect the most recent corruption (the audited tail
    is the *most recent* outcomes; we iterate it in order)."""
    out = {
        "status": "ok",
        "verdict": "INSUFFICIENT_DATA",
        "path": "",
        "slice": f"last {tail}",
        "n_total": 0,
        "n_dropped": 0,
        "drop_rate": 0.0,
        "by_reason": {r: 0 for r in REASONS},
        "samples": {r: [] for r in REASONS},
    }
    try:
        p = Path(outcomes_path) if outcomes_path is not None else DEFAULT_OUTCOMES_PATH
        out["path"] = str(p)
        if not p.exists():
            out["status"] = "no_file"
            return out
        # Bounded-memory read: only keep the most-recent ``tail`` rows
        # so a multi-MB JSONL doesn't load fully. ``deque`` cap mirrors
        # the trainer's tail-only consumption.
        from collections import deque

        lines = deque(maxlen=tail)
        with p.open("r") as fh:
            for ln in fh:
                if ln.strip():
                    lines.append(ln)
        rows: list[dict] = []
        for ln in lines:
            try:
                rec = json.loads(ln)
                if isinstance(rec, dict):
                    rows.append(rec)
            except Exception:
                pass  # malformed JSON: silently skipped (the trainer
                # paths also tolerate per-line corruption — see
                # `main()` in run_continuous_backtests). Counted as
                # "not a parsed row" so it doesn't pollute the
                # rejection-reason buckets.

        n_total = len(rows)
        out["n_total"] = n_total
        if n_total < MIN_ROWS:
            return out

        for row in rows:
            reason = _classify_drop(row)
            if reason is None:
                continue
            out["by_reason"][reason] += 1
            out["n_dropped"] += 1
            buf = out["samples"][reason]
            if len(buf) < samples_per_reason:
                # Persist the FULL row (small dict). A reader needs
                # the (run_id, sim_date, ticker) to triage; the value
                # itself (``forward_return_5d``) reveals the
                # corruption shape.
                buf.append(row)

        out["drop_rate"] = (out["n_dropped"] / n_total) if n_total else 0.0
        # Verdict ladder — precedence-ordered, first-match wins.
        if out["n_dropped"] == 0:
            out["verdict"] = "CLEAN"
        elif out["drop_rate"] <= LOW_RATE:
            out["verdict"] = "LOW_DROP_RATE"
        elif out["drop_rate"] > HIGH_RATE:
            out["verdict"] = "HIGH_DROP_RATE"
        else:
            out["verdict"] = "ELEVATED_DROP_RATE"
        return out
    except Exception as exc:
        out["status"] = f"error: {type(exc).__name__}: {exc}"
        return out


def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.dropped_label_audit",
        description="Audit which decision_outcomes.jsonl rows would be "
                    "dropped by train_scorer's label-validation pass, "
                    "classified by the first-match rejection reason "
                    "(missing/null/bool/unparseable/nan/+inf/-inf). "
                    "Read-only — never trains or writes.",
    )
    p.add_argument("--path", default=str(DEFAULT_OUTCOMES_PATH),
                   help="Path to decision_outcomes.jsonl (default: "
                        "the deployed file).")
    p.add_argument("--samples", type=int, default=3, dest="samples",
                   help="Max sample rows kept per reason in the report.")
    p.add_argument("--tail", type=int, default=MAX_OUTCOMES_FOR_TRAINING,
                   help="How many of the most-recent rows to audit "
                        "(matches the trainer's tail cap).")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of a table.")
    return p


def _format_row(row: dict, max_len: int = 80) -> str:
    """One-line summary of a dropped row for human-readable output.

    Picks the fields a triage operator needs: ``(run_id, sim_date,
    ticker, action)`` plus the offending ``forward_return_5d`` value.
    The bool, NaN, +inf, -inf shapes all stringify legibly via repr."""
    rid = row.get("run_id", "?")
    sd = row.get("sim_date", "?")
    tk = row.get("ticker", "?")
    act = row.get("action", "?")
    val = repr(row.get("forward_return_5d"))
    if len(val) > max_len:
        val = val[:max_len] + "…"
    return f"run={rid} {sd} {act} {tk}  forward_return_5d={val}"


def main(argv: "list[str] | None" = None) -> int:
    """Run the audit. Returns 0 on CLEAN / LOW_DROP_RATE / INSUFFICIENT_DATA,
    1 otherwise — shell callers can gate on ``$?`` like ``host_guard``."""
    import sys

    args = _build_arg_parser().parse_args(
        sys.argv[1:] if argv is None else argv)
    rep = analyze(args.path, samples_per_reason=args.samples, tail=args.tail)

    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
        return 0 if rep.get("verdict") in ("CLEAN", "LOW_DROP_RATE",
                                           "INSUFFICIENT_DATA") else 1

    print(f"[dropped_label_audit] path={rep.get('path')}")
    print(f"  slice={rep.get('slice')}  status={rep.get('status')}  "
          f"verdict={rep.get('verdict')}")
    print(f"  n_total={rep['n_total']}  n_dropped={rep['n_dropped']}  "
          f"drop_rate={rep['drop_rate']*100:.3f}%")
    print(f"  thresholds: LOW≤{LOW_RATE*100:.2f}%  "
          f"HIGH>{HIGH_RATE*100:.2f}%")
    if rep["n_dropped"] == 0:
        return 0
    print("  per-reason counts (precedence-ordered):")
    print(f"    {'reason':<16}{'n':>10}{'share_of_dropped':>22}")
    for reason in REASONS:
        n = rep["by_reason"][reason]
        if n == 0:
            continue
        share = n / max(1, rep["n_dropped"])
        print(f"    {reason:<16}{n:>10}{share*100:>20.1f}%")
    print("  sample dropped rows (run / sim_date / action / ticker / value):")
    for reason in REASONS:
        samples = rep["samples"][reason]
        if not samples:
            continue
        print(f"    [{reason}]")
        for row in samples:
            print(f"      {_format_row(row)}")
    return 0 if rep.get("verdict") in ("CLEAN", "LOW_DROP_RATE",
                                       "INSUFFICIENT_DATA") else 1


if __name__ == "__main__":
    raise SystemExit(main())

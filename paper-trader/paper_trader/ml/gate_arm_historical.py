"""Historical gate-arm audit — what the conviction gate ACTUALLY did at
decision time, using the persisted ``gate_scorer_pred`` field rather than
re-predicting with today's deployed model.

This is the truth-aware sibling of ``gate_audit.scorer_gate_audit``.

**Why this is not ``gate_audit``.** ``gate_audit`` re-predicts every stored
feature vector with the **currently deployed** ``decision_scorer.pkl`` and
buckets by the reconstructed prediction. The bucketing is a *counterfactual*:
"what would today's model say about each historical trade." The gate's
actual then-deployed prediction can differ — the model has been retrained
hundreds of times between decision time and the audit (`_DECISION_SCORER`
reset every cycle, per CLAUDE.md §5), and a decision the historical gate
sized down (×0.6) might be sized up (×1.3) by today's model on the same
features. The reconstruction has a documented residual ``gate_pnl`` itself
calls out: *NOT in its verdict*.

**This module reads ``gate_scorer_pred`` directly** (the 2026-05-18
``_parse_gate_decision`` capture, AGENTS.md "additive gate-decision
capture"). Each row's gate arm is decoded from the value the gate
historically saw, not from a fresh predict. The realized 5d return per
bucket is then the gate's literal economic effect on the trades it sized.

**Drop semantics — kept honest.**
* Rows with ``gate_scorer_pred=None`` (sub-gate cycle: scorer untrained or
  ``n_train<500``; SELL row; ``HOLD``; or a backtest run that pre-dates
  the 2026-05-18 capture) are dropped — they were never gate-acted-upon,
  so the gate's realized effect on them is undefined.
* Rows with ``gate_off_dist=True`` are dropped — the gate explicitly
  abstained on those (the ×1.0 no-op arm), so they don't measure any
  arm's edge. ``n_dropped_off_dist`` reports the count for transparency.
* Rows with ``forward_return_5d=None`` are dropped — no realized truth
  to bucket. ``n_dropped_no_return`` reports the count.

**Verdict identical to** ``gate_effectiveness_report``. Reuses the exact
``gate_arm`` decoder (single source of truth — both gate diagnostics must
agree on what each multiplier means), the same ``EDGE_TOL_PP``, the same
``MIN_TOTAL`` / ``MIN_ARM_N``. So a discrepancy between this report's
verdict and ``gate_audit``'s on the same outcomes pinpoints *training
drift* — the realized historical gate behaviour vs the counterfactual a
deployed-model re-predict produces — exactly the quantity ``gate_pnl``
documents as "outside its verdict scope."

This is a **read-only diagnostic**. It never trains, never touches
``decision_scorer.pkl``, ``decision_outcomes.jsonl``, ``build_features``,
``N_FEATURES``, or any trade path — same operational discipline as every
sibling diagnostic.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Single source of truth for arm decoding — must NEVER drift from
# `gate_audit.gate_arm` / the live `_ml_decide` if/elif chain.
from .gate_audit import gate_arm, _ARM_ORDER, _ARM_MULT, MIN_TOTAL, MIN_ARM_N, EDGE_TOL_PP


def gate_arm_historical_report(outcome_records) -> dict:
    """Bucket ``decision_outcomes.jsonl`` rows by the gate arm that ACTUALLY
    fired at decision time (decoded from the persisted ``gate_scorer_pred``)
    and report each arm's realized 5d return.

    ``outcome_records`` is any iterable of dict rows (the row shape
    ``_compute_decision_outcomes`` emits). The SELL sign-flip is applied
    here exactly like ``train_scorer`` / ``gate_audit`` / ``calibration``
    so realized direction has one consistent meaning across all diagnostics.

    Returns a JSON-safe dict with the same verdict semantics as
    ``gate_effectiveness_report`` (``INSUFFICIENT_DATA`` / ``GATE_HARMFUL`` /
    ``GATE_INEFFECTIVE`` / ``GATE_EFFECTIVE``) plus three additional fields
    not in the re-predict sibling:

    * ``n_dropped_no_gate_pred`` — rows with ``gate_scorer_pred=None``
      (sub-gate / SELL / HOLD / legacy-corpus).
    * ``n_dropped_off_dist`` — rows where the historical gate abstained.
    * ``n_dropped_no_return`` — rows missing ``forward_return_5d``.

    Never raises — any fault degrades to ``INSUFFICIENT_DATA``.
    """
    base = {
        "status": "ok",
        "verdict": "INSUFFICIENT_DATA",
        "n": 0,
        "arms": [],
        "strong_tailwind_minus_headwind_pp": None,
        "arm_monotone_fraction": None,
        "hint": "",
        "n_dropped_no_gate_pred": 0,
        "n_dropped_off_dist": 0,
        "n_dropped_no_return": 0,
    }
    try:
        recs = list(outcome_records or [])
    except Exception:
        recs = []

    if not recs:
        base["hint"] = "no outcome records"
        return base

    per_arm: dict[str, list[float]] = {a: [] for a in _ARM_ORDER}
    n_no_pred = 0
    n_off_dist = 0
    n_no_ret = 0
    for r in recs:
        try:
            pred = r.get("gate_scorer_pred")
            if pred is None:
                n_no_pred += 1
                continue
            # Off-distribution abstention: gate explicitly skipped sizing.
            # Defaults to False on legacy rows that pre-date the capture —
            # ``_parse_gate_decision`` returns ``(pred, off_dist)`` and we
            # only get here when pred is non-None, so off_dist is the
            # truthful then-recorded value.
            if r.get("gate_off_dist") is True:
                n_off_dist += 1
                continue
            y = r.get("forward_return_5d")
            if y is None:
                n_no_ret += 1
                continue
            pf = float(pred)
            yf = float(y)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(pf) and np.isfinite(yf)):
            continue
        action = str(r.get("action") or "BUY").upper()
        # SELL sign-flip mirrors train_scorer / gate_audit / calibration —
        # a drop after a SELL was the right call.
        y_aligned = -yf if action == "SELL" else yf
        arm, _ = gate_arm(pf)
        per_arm[arm].append(y_aligned)

    base["n_dropped_no_gate_pred"] = n_no_pred
    base["n_dropped_off_dist"] = n_off_dist
    base["n_dropped_no_return"] = n_no_ret

    n = sum(len(v) for v in per_arm.values())
    base["n"] = n

    arms_out = []
    for a in _ARM_ORDER:
        ys = per_arm[a]
        if ys:
            arr = np.asarray(ys, dtype=np.float64)
            arms_out.append({
                "arm": a,
                "multiplier": _ARM_MULT[a],
                "n": int(len(ys)),
                "mean_realized": round(float(arr.mean()), 4),
                "lo": round(float(arr.min()), 4),
                "hi": round(float(arr.max()), 4),
            })
        else:
            arms_out.append({
                "arm": a, "multiplier": _ARM_MULT[a], "n": 0,
                "mean_realized": None, "lo": None, "hi": None,
            })
    base["arms"] = arms_out

    # Monotonicity across arms with ≥1 sample, in multiplier order.
    present = [o for o in arms_out if o["n"] > 0]
    if len(present) >= 2:
        steps = len(present) - 1
        nondec = sum(1 for j in range(steps)
                     if present[j + 1]["mean_realized"] >= present[j]["mean_realized"])
        base["arm_monotone_fraction"] = round(nondec / steps, 4)

    head = per_arm["strong_headwind"]
    tail = per_arm["strong_tailwind"]
    if n < MIN_TOTAL or len(head) < MIN_ARM_N or len(tail) < MIN_ARM_N:
        base["hint"] = (
            f"need ≥{MIN_TOTAL} historical-gate pairs and ≥{MIN_ARM_N} in "
            f"BOTH extreme arms; have n={n}, "
            f"strong_headwind={len(head)}, strong_tailwind={len(tail)}"
        )
        return base

    head_mean = float(np.mean(head))
    tail_mean = float(np.mean(tail))
    spread = tail_mean - head_mean
    base["strong_tailwind_minus_headwind_pp"] = round(spread, 4)

    if spread < -EDGE_TOL_PP:
        base["verdict"] = "GATE_HARMFUL"
        base["hint"] = (
            f"historical gate ×1.30 arm realized {tail_mean:+.2f}% < ×0.60 arm "
            f"{head_mean:+.2f}% (spread {spread:+.2f}pp) — the gate has been "
            f"inverting capital allocation in production"
        )
    elif abs(spread) <= EDGE_TOL_PP:
        base["verdict"] = "GATE_INEFFECTIVE"
        base["hint"] = (
            f"historical gate ×1.30 arm {tail_mean:+.2f}% vs ×0.60 arm "
            f"{head_mean:+.2f}% (spread {spread:+.2f}pp, within ±"
            f"{EDGE_TOL_PP:.1f}pp) — the production gate's bigger bets "
            f"have not realized higher returns"
        )
    else:
        base["verdict"] = "GATE_EFFECTIVE"
        base["hint"] = (
            f"historical gate ×1.30 arm realized {tail_mean:+.2f}% > ×0.60 arm "
            f"{head_mean:+.2f}% (spread {spread:+.2f}pp) — the gate's "
            f"production sizing has been economically justified"
        )
    return base


def analyze(outcomes_path: Path | str, oos_only: bool = True) -> dict:
    """Load ``outcomes_path`` and return the historical gate-arm report.

    ``oos_only`` (default True) restricts the audit to the temporal-OOS
    slice via ``validation.split_outcomes_temporal``. The chosen slice is
    reported in ``slice`` (``"oos"`` / ``"all"``). Read-only; never raises.
    """
    out: dict = {
        "status": "error", "verdict": "INSUFFICIENT_DATA",
        "n": 0, "arms": [], "hint": "",
        "n_dropped_no_gate_pred": 0,
        "n_dropped_off_dist": 0,
        "n_dropped_no_return": 0,
    }
    try:
        path = Path(outcomes_path)
        if not path.exists():
            out["hint"] = f"no file at {path}"
            return out
        records: list[dict] = []
        with path.open("r") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    records.append(json.loads(ln))
                except Exception:
                    continue
        slice_name = "all"
        if oos_only:
            try:
                from paper_trader.validation import split_outcomes_temporal
                _, oos = split_outcomes_temporal(records, oos_fraction=0.2)
                if oos:
                    records = oos
                    slice_name = "oos"
            except Exception:
                slice_name = "all"
        rep = gate_arm_historical_report(records)
        rep["slice"] = slice_name
        rep["outcomes_n"] = len(records)
        return rep
    except Exception as e:
        out["hint"] = f"analyze failed: {type(e).__name__}: {e}"
        return out


def _cli() -> int:
    """Read-only CLI: `python3 -m paper_trader.ml.gate_arm_historical`."""
    import argparse
    import sys

    ROOT = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.gate_arm_historical",
        description="Audit the conviction gate's historical (then-deployed) "
                    "sizing effect using persisted gate_scorer_pred, not "
                    "today's model re-prediction. Read-only.",
    )
    parser.add_argument(
        "--outcomes",
        default=str(ROOT / "data" / "decision_outcomes.jsonl"),
        help="Path to decision_outcomes.jsonl",
    )
    parser.add_argument("--all", action="store_true",
                        help="Audit full corpus (default: OOS slice only)")
    parser.add_argument("--json", action="store_true",
                        help="Emit JSON instead of a table")
    args = parser.parse_args()

    rep = analyze(args.outcomes, oos_only=not args.all)
    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
        return 0 if rep.get("verdict") not in (
            "INSUFFICIENT_DATA", None) else 1

    slice_s = rep.get("slice") or "?"
    n_drop = (rep.get("n_dropped_no_gate_pred", 0)
              + rep.get("n_dropped_off_dist", 0)
              + rep.get("n_dropped_no_return", 0))
    print(f"[gate_arm_historical] slice={slice_s} "
          f"outcomes_n={rep.get('outcomes_n', 0)} "
          f"gate_n={rep.get('n', 0)} dropped={n_drop} "
          f"(no_pred={rep.get('n_dropped_no_gate_pred', 0)}, "
          f"off_dist={rep.get('n_dropped_off_dist', 0)}, "
          f"no_return={rep.get('n_dropped_no_return', 0)})")
    print(f"  verdict: {rep.get('verdict')}")
    if rep.get("hint"):
        print(f"  {rep['hint']}")
    spread = rep.get("strong_tailwind_minus_headwind_pp")
    if spread is not None:
        print(f"  strong_tailwind − strong_headwind: {spread:+.2f}pp")
    mono = rep.get("arm_monotone_fraction")
    if mono is not None:
        print(f"  arm_monotone_fraction: {mono:.2f} "
              f"(1.0 = perfect, 0.0 = inverted)")
    print(f"  {'arm':<18}{'mult':>6}{'n':>8}{'mean_realized':>16}"
          f"{'lo':>10}{'hi':>10}")
    for a in (rep.get("arms") or []):
        mean_s = (f"{a['mean_realized']:+.2f}%" if a.get('mean_realized') is not None
                  else "(none)")
        lo_s = (f"{a['lo']:+.2f}%" if a.get('lo') is not None else "(none)")
        hi_s = (f"{a['hi']:+.2f}%" if a.get('hi') is not None else "(none)")
        print(f"  {a['arm']:<18}{a['multiplier']:>6.2f}{a['n']:>8}"
              f"{mean_s:>16}{lo_s:>10}{hi_s:>10}")
    return 0 if rep.get("verdict") not in ("INSUFFICIENT_DATA", None) else 1


if __name__ == "__main__":
    raise SystemExit(_cli())

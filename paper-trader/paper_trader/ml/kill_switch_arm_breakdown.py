"""Kill-switch realized effect — per-would-have-been-arm breakdown.

Read-only research signal (2026-05-31 feature, Agent 2 ML+backtests pass #8).
Sibling to ``kill_switch_realized_effect`` (which already aggregates the
realized rank-IC inside the killswitch bucket) and to ``gate_arm_historical``
(which already does per-arm realized means but FILTERS to ``gate_off_dist
is not True`` — line 119 — so in a production where 100% of BUYs are
abstained, that analyzer's output is structurally empty). This module is the
intersection neither covers:

  *For the BUYs the kill-switch DECIDED NOT TO MODULATE, decompose by the
  would-have-been gate arm and report per-arm realized return.*

**The gap this closes.** ``kill_switch_realized_effect`` delivers an
aggregate verdict (KILLSWITCH_HELPS / NEUTRAL / HURTS) driven by rank-IC
across the whole killswitch bucket. Rank-IC has a known blind spot to
LEVEL effects: a configuration where every arm carries the same +5%
realized mean produces rank-IC ≈ 0 (NEUTRAL verdict), but firing the gate
in that state would systematically REDUCE conviction on profitable headwind
arms (×0.6 / ×0.85) — a real economic cost the aggregate cannot see.
Conversely, the gate's ×1.30 ``strong_tailwind`` arm dominates the
counterfactual P&L; an analyzer that doesn't separate it from the noisy
``neutral`` bulk understates the cost of suppressing that arm specifically.

**The live state this is designed to debug.** Current production (snapshot
2026-05-31, ``data/decision_outcomes.jsonl``): 0 ``acted``, 21 ``clamp``,
4094 ``killswitch`` BUYs — the entire DecisionScorer→gate apparatus is
OFF. ``kill_switch_realized_effect`` already verdicts ``KILLSWITCH_HURTS``
(rank-IC +0.094, n=4094) — predictions still carry signal. The natural
follow-up — *which arms drive the cost?* — was unobservable until now.

**Metrics per would-have-been arm** (decoded via ``gate_audit.gate_arm``
from each row's ``gate_scorer_pred``; arm names + multipliers match
``_ml_decide``'s if/elif chain to the bit, single source of truth):

  * ``n``                  — count of killswitch BUYs in this arm bucket
  * ``mean_pred``          — mean of ``gate_scorer_pred`` in this bucket
                              (sanity check that decoded arm aligns with
                              the value range)
  * ``mean_realized``      — mean of ``forward_return_5d`` (%)
  * ``median_realized``    — median ``forward_return_5d`` (%)
  * ``multiplier``         — the conviction multiplier the gate WOULD have
                              applied (0.60 / 0.85 / 1.00 / 1.15 / 1.30)
  * ``per_trade_tilt_pp``  — ``(multiplier - 1.0) * mean_realized`` —
                              the per-dollar-IN-THIS-TRADE tilt the gate
                              would have applied. NOT a portfolio P&L
                              (matches the discipline ``gate_pnl`` documents
                              for its own reconstruction residual: the
                              cash-budget coupling and the cross-trade
                              re-sizing are NOT in this number; it is the
                              honest *per-trade* counterfactual, useful for
                              sign and rank).

**Aggregate** (across all kill-switched rows with a decoded arm):

  * ``aggregate_per_trade_tilt_pp`` — ``Σ_i n_i × (mult_i - 1) × mean_realized_i / Σ_i n_i``.
    Same per-trade unit as the arm-level metric (NOT portfolio P&L).
    Positive ⇒ firing the gate would have added per-trade return on
    average; negative ⇒ firing would have subtracted.

**Verdict ladder** (driven by the EXTREME arms because they have the
biggest multiplier deltas — the gate's economic effect is concentrated
where the multiplier is furthest from 1.0):

| Verdict                       | Trigger |
|-------------------------------|---------|
| ``INSUFFICIENT_DATA``         | total killswitch n < ``MIN_TOTAL_N`` |
| ``STRONG_TAILWIND_COSTLY``    | ``strong_tailwind`` arm n ≥ ``MIN_ARM_N`` AND mean_realized ≥ +``ARM_TOL_PCT`` — the gate would have boosted these trades by ×1.30 into a realized win; abstaining suppressed an opportunity. |
| ``STRONG_HEADWIND_COSTLY``    | ``strong_headwind`` arm n ≥ ``MIN_ARM_N`` AND mean_realized ≤ -``ARM_TOL_PCT`` — the gate would have shrunk these trades by ×0.60 ahead of a realized loss; abstaining left full-sized losers in. |
| ``BOTH_TAILS_COSTLY``         | both of the above are true — maximum opportunity cost; the killswitch is missing on BOTH directions. |
| ``ARMS_NEUTRAL``              | neither extreme arm reaches ARM_TOL_PCT; gate firing wouldn't materially shift per-trade realized in expectation. |

INSUFFICIENT_DATA / ARMS_NEUTRAL are benign; the three TAIL_COSTLY
verdicts are operationally actionable (the kill-switch is concealing a
modulation that would have improved per-trade return). Pairs cleanly
with ``kill_switch_realized_effect``: this module answers ``WHERE`` /
``WHICH`` while the sibling answers ``WHETHER``.

Pure / module-level constants for testability. CLI:

```bash
cd /home/zeph/trading-intelligence/paper-trader && \\
    python3 -m paper_trader.ml.kill_switch_arm_breakdown
cd /home/zeph/trading-intelligence/paper-trader && \\
    python3 -m paper_trader.ml.kill_switch_arm_breakdown --json
cd /home/zeph/trading-intelligence/paper-trader && \\
    python3 -m paper_trader.ml.kill_switch_arm_breakdown \\
        --outcomes path/to/decision_outcomes.jsonl
```

Exit code mirrors sibling diagnostics: 0 on benign verdicts
(``INSUFFICIENT_DATA`` / ``ARMS_NEUTRAL``), 2 on actionable verdicts
(``STRONG_TAILWIND_COSTLY`` / ``STRONG_HEADWIND_COSTLY`` /
``BOTH_TAILS_COSTLY``).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from paper_trader.ml.gate_audit import gate_arm as _gate_arm_decode


# Module-level constants (testable, single point of tuning).

# Minimum total killswitch n below which no verdict is emitted. 200 mirrors
# ``kill_switch_realized_effect.MIN_PAIRS`` for cross-tool consistency.
MIN_TOTAL_N = 200

# Minimum n inside a single arm bucket below which that arm's mean_realized
# is too noisy to ground a verdict on. The two extreme arms (strong_tailwind
# / strong_headwind) drive the verdict, and a 5% mean on n=10 is well within
# noise; n>=50 puts the SE on a typical realized 5d return distribution
# (σ ≈ 8-12% per AGENTS.md) at roughly ±1.4pp — comfortably below the
# 1.0pp ARM_TOL_PCT band so a verdict is statistically meaningful.
MIN_ARM_N = 50

# Per-arm realized-mean tolerance band. A |mean_realized| inside ±1.0pp is
# treated as "no material level effect" for THAT arm regardless of n. The
# 1.0pp value matches ``gate_audit.EDGE_TOL_PP`` / ``gate_pnl.EDGE_TOL_PP``
# (single source of truth across every gate-economic diagnostic) so the
# verdict thresholds are coherent across the whole gate analyzer family.
ARM_TOL_PCT = 1.0

# Canonical arm order — matches ``gate_audit.gate_arm`` if/elif emission
# and the conviction multipliers ``_ml_decide`` applies. Holding this here
# (vs deriving it from the gate_audit decoder) makes the report
# deterministic across Python versions.
_ARMS: tuple[tuple[str, float], ...] = (
    ("strong_headwind", 0.60),
    ("mild_headwind", 0.85),
    ("neutral", 1.00),
    ("mild_tailwind", 1.15),
    ("strong_tailwind", 1.30),
)

DECISION_OUTCOMES = (Path(__file__).resolve().parent.parent.parent
                     / "data" / "decision_outcomes.jsonl")


def _maybe_float(v):
    """Coerce to finite float or None — mirrors sibling analyzers."""
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    return x


def _is_killswitched(kind) -> bool:
    """A killswitch abstention is the only bucket this analyzer reports on.

    "clamp" / None / anything else is dropped — the per-bucket comparison
    (acted / clamp / killswitch) is already handled by the sibling
    ``kill_switch_realized_effect``. This module is the per-arm DEEP-DIVE
    inside the killswitch bucket.
    """
    if not isinstance(kind, str):
        return False
    return kind.strip().lower() == "killswitch"


def _arm_for(pred: float) -> tuple[str, float]:
    """Decode the would-have-been arm via the canonical ``gate_audit.gate_arm``
    helper so the boundary operators match ``_ml_decide`` to the bit.

    Defensive against the import-time circular surprise: a future move of
    ``gate_audit`` to a package that imports back into ml/ should not break
    this analyzer's report path. The decoder is total (a non-finite input
    returns ``("neutral", 1.00)`` by contract), so this never raises.
    """
    return _gate_arm_decode(pred)


def kill_switch_arm_report(rows: list[dict]) -> dict:
    """Compute the per-arm killswitch breakdown from outcome rows.

    Filters to BUYs with ``gate_abstention_kind == "killswitch"`` AND a
    finite ``gate_scorer_pred`` AND a finite ``forward_return_5d``. Decodes
    each surviving row to its would-have-been arm and reports per-arm
    n / mean_pred / mean_realized / median_realized / multiplier /
    per_trade_tilt_pp, plus the aggregate ``aggregate_per_trade_tilt_pp``
    and a verdict driven by the two extreme arms (where the gate's
    multiplier is furthest from 1.0 — the place its economic effect
    concentrates).

    Returns a JSON-safe dict. Never raises (the analyzer must not break
    the unattended loop)."""
    try:
        it = list(rows or [])
    except Exception:
        it = []

    # Bucket per arm name. Initialize all arms so the report always has
    # a fixed shape — a missing arm renders n=0 / metrics=None instead of
    # vanishing from the output.
    arm_preds: dict[str, list[float]] = {n: [] for n, _ in _ARMS}
    arm_acts: dict[str, list[float]] = {n: [] for n, _ in _ARMS}
    arm_mult: dict[str, float] = {n: m for n, m in _ARMS}

    n_buys = 0
    n_killswitched = 0
    for r in it:
        if not isinstance(r, dict):
            continue
        if str(r.get("action") or "").upper() != "BUY":
            continue
        n_buys += 1
        if not _is_killswitched(r.get("gate_abstention_kind")):
            continue
        pred = _maybe_float(r.get("gate_scorer_pred"))
        ret = _maybe_float(r.get("forward_return_5d"))
        if pred is None or ret is None:
            continue
        n_killswitched += 1
        arm, _mult = _arm_for(pred)
        if arm not in arm_preds:
            # Defensive: an arm name from a future gate_audit refactor we
            # don't track would silently land NOWHERE. Drop the row honestly
            # rather than fabricate a bucket — same drift-guard discipline
            # as ``kill_switch_realized_effect._bucket_for``.
            continue
        arm_preds[arm].append(pred)
        arm_acts[arm].append(ret)

    # Per-arm summary stats.
    arms_out: list[dict] = []
    weighted_tilt_num = 0.0
    weighted_tilt_den = 0
    for arm_name, mult in _ARMS:
        ps = arm_preds[arm_name]
        rs = arm_acts[arm_name]
        n = len(ps)
        mean_pred: float | None
        mean_realized: float | None
        median_realized: float | None
        per_trade_tilt_pp: float | None
        if n == 0:
            mean_pred = mean_realized = median_realized = None
            per_trade_tilt_pp = None
        else:
            mean_pred = round(float(np.mean(np.asarray(ps, dtype=np.float64))),
                              4)
            mean_realized = round(
                float(np.mean(np.asarray(rs, dtype=np.float64))), 4)
            median_realized = round(
                float(np.median(np.asarray(rs, dtype=np.float64))), 4)
            per_trade_tilt_pp = round((mult - 1.0) * mean_realized, 4)
            weighted_tilt_num += n * (mult - 1.0) * mean_realized
            weighted_tilt_den += n
        arms_out.append({
            "arm": arm_name,
            "multiplier": mult,
            "n": n,
            "mean_pred": mean_pred,
            "mean_realized": mean_realized,
            "median_realized": median_realized,
            "per_trade_tilt_pp": per_trade_tilt_pp,
        })

    aggregate_tilt: float | None
    if weighted_tilt_den > 0:
        aggregate_tilt = round(weighted_tilt_num / weighted_tilt_den, 4)
    else:
        aggregate_tilt = None

    # Verdict — driven by the two extreme arms (largest multiplier delta
    # from 1.0). Each arm must independently clear MIN_ARM_N to contribute
    # to a tail-costly verdict (a 3-sample +50% arm should not flip the
    # ladder). ARM_TOL_PCT centralizes the level tolerance — see the
    # module docstring for the rationale on its value.
    by_name = {a["arm"]: a for a in arms_out}
    st = by_name["strong_tailwind"]
    sh = by_name["strong_headwind"]
    st_costly = (
        st["n"] >= MIN_ARM_N
        and st["mean_realized"] is not None
        and st["mean_realized"] >= ARM_TOL_PCT
    )
    sh_costly = (
        sh["n"] >= MIN_ARM_N
        and sh["mean_realized"] is not None
        and sh["mean_realized"] <= -ARM_TOL_PCT
    )

    if n_killswitched < MIN_TOTAL_N:
        verdict = "INSUFFICIENT_DATA"
        hint = (
            f"killswitch n={n_killswitched} (need >={MIN_TOTAL_N}); per-arm "
            "breakdown undefined — accumulate more outcomes."
        )
    elif st_costly and sh_costly:
        verdict = "BOTH_TAILS_COSTLY"
        hint = (
            f"strong_tailwind n={st['n']} mean={st['mean_realized']:+.2f}% "
            f">= +{ARM_TOL_PCT} AND strong_headwind n={sh['n']} "
            f"mean={sh['mean_realized']:+.2f}% <= -{ARM_TOL_PCT} — the "
            "kill-switch is concealing modulation on BOTH directions. The "
            "×1.30 arm would have boosted realized winners; the ×0.60 arm "
            "would have trimmed realized losers."
        )
    elif st_costly:
        verdict = "STRONG_TAILWIND_COSTLY"
        hint = (
            f"strong_tailwind n={st['n']} mean={st['mean_realized']:+.2f}% "
            f">= +{ARM_TOL_PCT} — the kill-switch is preventing a ×1.30 "
            "boost into trades that DID realize a meaningful win on average."
        )
    elif sh_costly:
        verdict = "STRONG_HEADWIND_COSTLY"
        hint = (
            f"strong_headwind n={sh['n']} mean={sh['mean_realized']:+.2f}% "
            f"<= -{ARM_TOL_PCT} — the kill-switch is preventing a ×0.60 "
            "trim ahead of trades that DID realize a meaningful loss on "
            "average."
        )
    else:
        verdict = "ARMS_NEUTRAL"
        hint = (
            f"strong_tailwind mean={st.get('mean_realized')}, "
            f"strong_headwind mean={sh.get('mean_realized')} — neither "
            f"extreme arm clears ±{ARM_TOL_PCT}pp; the kill-switch's "
            "abstention has no obvious tail-cost per arm."
        )

    return {
        "verdict": verdict,
        "n_buys": n_buys,
        "n_killswitched": n_killswitched,
        "arms": arms_out,
        "aggregate_per_trade_tilt_pp": aggregate_tilt,
        "min_total_n": MIN_TOTAL_N,
        "min_arm_n": MIN_ARM_N,
        "arm_tol_pct": ARM_TOL_PCT,
        "hint": hint,
        # Honest unit reminder so a reader doesn't mistake the tilt for a
        # portfolio P&L (the documented kill_switch_realized_effect /
        # gate_pnl reconstruction-residual discipline).
        "unit_note": (
            "per_trade_tilt_pp and aggregate_per_trade_tilt_pp are "
            "per-dollar-in-this-trade tilts, NOT portfolio P&L. They "
            "ignore the cash-budget coupling and cross-trade re-sizing "
            "the live gate would have produced. Use for sign / arm-rank, "
            "not absolute portfolio return reconstruction."
        ),
    }


def analyze(outcomes_path: "Path | str | None" = None) -> dict:
    """Read ``decision_outcomes.jsonl`` and run the per-arm killswitch report.

    Returns ``{status, ...report fields}``. ``status='error'`` on read
    failure; ``status='ok'`` otherwise. Never raises."""
    if outcomes_path is None:
        outcomes_path = DECISION_OUTCOMES
    p = Path(outcomes_path)
    if not p.exists():
        report = kill_switch_arm_report([])
        report["status"] = "error"
        report["error"] = f"missing {p}"
        return report
    rows: list[dict] = []
    try:
        with p.open("r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception as exc:
        report = kill_switch_arm_report([])
        report["status"] = "error"
        report["error"] = f"read failed: {exc}"
        return report

    report = kill_switch_arm_report(rows)
    report["status"] = "ok"
    return report


def _cli(argv: "list[str] | None" = None) -> int:
    """CLI entrypoint. Exit codes mirror sibling diagnostics."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.kill_switch_arm_breakdown",
        description=(
            "Per-would-have-been-arm breakdown of kill-switched BUYs in "
            "decision_outcomes.jsonl. Decodes the would-have-been gate arm "
            "for each killswitch-abstained BUY and reports realized mean / "
            "median + per-trade tilt the gate would have applied. "
            "Complements kill_switch_realized_effect (aggregate verdict) "
            "with the per-arm level lens that rank-IC structurally cannot "
            "see."
        ),
    )
    parser.add_argument("--outcomes", type=Path, default=None,
                        help=("Path to decision_outcomes.jsonl (default: "
                              "data/decision_outcomes.jsonl relative to "
                              "repo)."))
    parser.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)

    report = analyze(args.outcomes)
    verdict = report.get("verdict") or ""

    actionable = verdict in (
        "STRONG_TAILWIND_COSTLY",
        "STRONG_HEADWIND_COSTLY",
        "BOTH_TAILS_COSTLY",
    )

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2 if actionable else 0

    print(f"[kill_switch_arm_breakdown] verdict={verdict}  "
          f"n_buys={report.get('n_buys')}  "
          f"n_killswitched={report.get('n_killswitched')}")
    if verdict == "INSUFFICIENT_DATA":
        print(f"  hint: {report.get('hint')}")
        return 0
    print(f"  {'arm':<18}{'mult':>6}{'n':>8}"
          f"{'mean_pred':>14}{'mean_realized':>18}"
          f"{'median_realized':>18}{'per_trade_tilt_pp':>20}")
    for a in report.get("arms") or []:
        n = a.get("n") or 0
        mp = a.get("mean_pred")
        mr = a.get("mean_realized")
        med = a.get("median_realized")
        tilt = a.get("per_trade_tilt_pp")
        mp_s = f"{mp:+.3f}" if isinstance(mp, (int, float)) else "n/a"
        mr_s = f"{mr:+.3f}%" if isinstance(mr, (int, float)) else "n/a"
        med_s = f"{med:+.3f}%" if isinstance(med, (int, float)) else "n/a"
        tilt_s = (f"{tilt:+.4f}pp" if isinstance(tilt, (int, float))
                  else "n/a")
        print(f"  {a['arm']:<18}{a['multiplier']:>6.2f}{n:>8}"
              f"{mp_s:>14}{mr_s:>18}{med_s:>18}{tilt_s:>20}")
    agg = report.get("aggregate_per_trade_tilt_pp")
    if agg is not None:
        print(f"  aggregate per-trade tilt (mult-weighted): {agg:+.4f}pp")
    print(f"  hint: {report.get('hint')}")
    print(f"  unit: {report.get('unit_note')}")
    return 2 if actionable else 0


if __name__ == "__main__":
    raise SystemExit(_cli())

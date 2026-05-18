"""Conviction-gate REALIZED effect — measured from the gate's *captured
then-deployed* decision, with **no re-prediction**.

This is a **read-only diagnostic**. It never trains, never touches
`decision_scorer.pkl`, `decision_outcomes.jsonl`, `build_features`,
`N_FEATURES`, or any trade path — same operational discipline as
`paper_trader/ml/gate_audit.py` / `gate_pnl.py` / `calibration.py` /
`skill_trend.py`. Safe to run against the live unattended loop.

**Why this is not `gate_audit.py` or `gate_pnl.py`.** Both of those take
the *current* pickled scorer and call `scorer.predict(...)` on each stored
feature row — i.e. they RE-PREDICT with **today's** model. Their own
docstrings disclaim the consequence: that is a counterfactual ("what would
the model say *now*"), provably **not** what the gate did at decision time
with that cycle's own pickle (`gate_pnl` explicitly states its
reconstruction residual "is *NOT in its verdict*"). A scorer retrains every
cycle; the model that sized a trade six cycles ago is gone.

Commit `60b20d9` ("capture gate's actual then-deployed decision in
decision_outcomes") closed exactly that observability gap: every BUY
outcome row now additively carries

  * ``gate_scorer_pred``  — the float % the gate **actually** modulated
    conviction on at decision time (``None`` on SELL / untrained-cycle
    rows: the gate is BUY-only and inert below ``_n_train >= 500``), and
  * ``gate_off_dist``     — ``True`` when the off-distribution guard fired
    so the gate **abstained** (conviction left untouched — NOT the arm
    ``gate_scorer_pred`` would otherwise map to).

`gate_audit`/`gate_pnl` predate that capture and still re-predict.
**Nothing consumes the captured field.** This module is its payoff: it
buckets realized forward returns by the gate's *true historical* arm —
zero `predict()` calls, zero pickle load — answering the question the
re-predicting tools structurally cannot: *as the gate was actually
deployed, did its bigger bets realize more?*

**The off-distribution honesty re-prediction cannot replicate.** When
`gate_off_dist` is True the live gate left conviction **untouched** — it
applied no multiplier at all, regardless of where the (clamped ±50)
prediction falls. A re-predicting tool has no way to know the gate
abstained on that row and would mis-assign it to the ``strong_headwind``
arm, fabricating an effect the gate never applied. Here those rows go to a
separate ``abstained`` bucket and are **excluded from the verdict** — the
verdict grades only arms the gate genuinely applied.

`gate_arm` is imported from `gate_audit` (single source of truth — the
five arms must never drift between the gate diagnostics, the codebase's
invariant-#10 spirit, exactly as `gate_pnl` does) and the temporal-OOS
slice from `validation.split_outcomes_temporal` (the same split
`gate_audit`/`gate_pnl`/`skill_trend`/`_train_decision_scorer`'s
`oos_rmse` use, so this describes the same holdout).

**Expected state today (a documented limitation, like every sibling).**
The running continuous loop predates `60b20d9`, so its
`decision_outcomes.jsonl` tail carries no `gate_scorer_pred` yet — this
tool returns ``GATE_CAPTURE_NOT_YET_POPULATED`` until the loop redeploys
and accumulates rows under the new capture (the `horizon_audit`
``INSUFFICIENT_LONG_HORIZON`` "populates as the loop runs" precedent). The
verdict logic is fully exercised offline with synthetic rows so it stands
independent of that backlog.

```bash
cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.gate_realized
cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.gate_realized --all
```
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Single source of truth for the five gate arms / multipliers — importing
# (not re-declaring) guarantees this diagnostic and gate_audit / gate_pnl can
# never disagree about what the live `_ml_decide` gate does (the
# gate_pnl-imports-gate_arm precedent).
from .gate_audit import gate_arm

# Verdict thresholds — module-level so tests assert exact verdicts and a
# tuning change is a single reviewable edit (== gate_audit / gate_pnl).
MIN_TOTAL = 30      # need a real acted sample before any verdict (== gate_audit.MIN_TOTAL)
MIN_ARM_N = 5       # min trades in EACH extreme arm to compare them (== gate_audit.MIN_ARM_N)
EDGE_TOL_PP = 1.0   # |tailwind − headwind| band that reads as noise (== gate_audit.EDGE_TOL_PP)

# Additive research horizons (commit ccc4d31). 5d is the gate's training
# target and the verdict anchor (every sibling locks to it); 10/20 are
# reported per-arm when resolvable but never drive the verdict.
HORIZONS = (5, 10, 20)

_ARM_ORDER = ["strong_headwind", "mild_headwind", "neutral",
              "mild_tailwind", "strong_tailwind"]


def _f(v):
    """Finite float or None — the `_to_float` hardening class, local so this
    module imports nothing from decision_scorer (read-only, no pickle path)."""
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def gate_realized_report(rows) -> dict:
    """Bucket outcome rows by the gate's **captured then-deployed** arm and
    report each arm's realized forward return.

    ``rows`` — any iterable of ``decision_outcomes.jsonl``-shaped dicts. For
    each row this reads, and never re-predicts:

      * ``gate_scorer_pred`` — the gate's true decision-time prediction.
        ``None`` ⇒ no gate decision on this row (SELL / sub-`n_train` /
        pre-`60b20d9` deploy-stale): **excluded entirely**.
      * ``gate_off_dist``    — ``True`` ⇒ the gate abstained (no multiplier
        applied): routed to the separate ``abstained`` bucket, **not** an
        arm, and **excluded from the verdict**.
      * ``forward_return_5d`` (+ 10d/20d when present) — realized return,
        SELL-sign-flipped (`-fwd`) exactly like ``train_scorer`` /
        ``gate_audit``. ``gate_scorer_pred`` is BUY-only by construction so
        the flip is a defensive consistency guard, not a common path.

    The verdict is driven solely by the realized 5d spread between the two
    EXTREME *acted* arms — ``strong_tailwind`` (the gate's biggest bet,
    ×1.30) minus ``strong_headwind`` (its smallest, ×0.60) — exactly the
    spread the 1.30/0.60 multiplier ratio is implicitly underwriting, but
    measured from what the gate **actually did**, not a re-prediction:

    | Verdict | Meaning |
    |---------|---------|
    | ``GATE_CAPTURE_NOT_YET_POPULATED`` | 0 rows carry a non-null ``gate_scorer_pred`` — the loop predates `60b20d9` / hasn't accumulated captured rows yet (the expected deploy-stale state) |
    | ``INSUFFICIENT_DATA`` | some captured rows exist but < ``MIN_TOTAL`` acted, or either extreme arm < ``MIN_ARM_N`` |
    | ``GATE_HARMFUL`` | tailwind − headwind < −``EDGE_TOL_PP`` — as actually deployed the ×1.30 arm sized UP the losers; turning the gate off would have realized more |
    | ``GATE_INEFFECTIVE`` | \\|spread\\| ≤ ``EDGE_TOL_PP`` — the multipliers reallocated with no realized edge (pure added sizing variance) |
    | ``GATE_EFFECTIVE`` | spread > +``EDGE_TOL_PP`` — the bigger bets really did realize more; the gate's ordering earned its keep |

    ``arm_monotone_fraction`` (adjacent acted arms, multiplier order,
    realized-mean non-decreasing) and the ``abstained`` bucket are
    informational descriptors — **NOT** folded into the verdict (the
    `gate_audit` arm-monotone honesty pattern), so the verdict stays
    crisply exact-value testable on the two-arm 5d spread alone.

    Returns a JSON-safe dict. Never raises.
    """
    # Per-arm realized lists, per horizon: arm -> {h: [realized,...]}.
    acted: dict[str, dict[int, list[float]]] = {
        a: {h: [] for h in HORIZONS} for a in _ARM_ORDER
    }
    abstained: dict[int, list[float]] = {h: [] for h in HORIZONS}
    n_captured = 0       # rows with a non-null gate_scorer_pred
    n_acted = 0          # captured AND not off-distribution (an arm applied)
    n_abstained = 0      # captured AND off-distribution (gate left it alone)

    try:
        it = list(rows or [])
    except Exception:
        it = []

    for r in it:
        if not isinstance(r, dict):
            continue
        gp = _f(r.get("gate_scorer_pred"))
        if gp is None:
            continue  # no gate decision on this row — never counted
        n_captured += 1

        is_sell = str(r.get("action") or "BUY").upper() == "SELL"
        # Resolve every horizon this row supports (5d is the anchor; a row
        # may carry only 5d on older captured rows — that is fine, 10/20
        # simply have a smaller n, the horizon_audit per-cell-n discipline).
        realized: dict[int, float] = {}
        for h in HORIZONS:
            fv = _f(r.get(f"forward_return_{h}d"))
            if fv is None:
                continue
            realized[h] = -fv if is_sell else fv
        if 5 not in realized:
            # The verdict anchor is unusable for this row — it can carry no
            # 5d signal. (10/20-only is impossible: _fwd_ret_h gates longer
            # horizons strictly tighter than 5d, but guard anyway.)
            continue

        if bool(r.get("gate_off_dist")):
            n_abstained += 1
            for h, v in realized.items():
                abstained[h].append(v)
            continue

        n_acted += 1
        arm, _mult = gate_arm(gp)
        for h, v in realized.items():
            acted[arm][h].append(v)

    def _stats(vals: list[float]) -> dict:
        if not vals:
            return {"n": 0, "mean": None, "lo": None, "hi": None}
        a = np.asarray(vals, dtype=np.float64)
        return {
            "n": int(a.size),
            "mean": round(float(a.mean()), 4),
            "lo": round(float(a.min()), 4),
            "hi": round(float(a.max()), 4),
        }

    arms_out = []
    for a in _ARM_ORDER:
        _, mult = gate_arm(
            {"strong_headwind": -50.0, "mild_headwind": -5.0,
             "neutral": 0.0, "mild_tailwind": 7.5,
             "strong_tailwind": 50.0}[a]
        )
        s5 = _stats(acted[a][5])
        arms_out.append({
            "arm": a,
            "multiplier": mult,
            "n": s5["n"],
            "mean_realized_5d": s5["mean"],
            "lo_5d": s5["lo"],
            "hi_5d": s5["hi"],
            "mean_realized_10d": _stats(acted[a][10])["mean"],
            "mean_realized_20d": _stats(acted[a][20])["mean"],
            "n_10d": _stats(acted[a][10])["n"],
            "n_20d": _stats(acted[a][20])["n"],
        })

    abst5 = _stats(abstained[5])
    base = {
        "status": "ok",
        "verdict": "GATE_CAPTURE_NOT_YET_POPULATED",
        "measurement": "captured_then_deployed_no_reprediction",
        "n_captured": n_captured,
        "n_acted": n_acted,
        "n_abstained": n_abstained,
        "arms": arms_out,
        "abstained_mean_realized_5d": abst5["mean"],
        "strong_tailwind_minus_headwind_pp": None,
        "arm_monotone_fraction": None,
        "hint": "",
    }

    if n_captured == 0:
        base["hint"] = (
            "no row carries a non-null gate_scorer_pred — the continuous "
            "loop predates commit 60b20d9 (gate-decision capture) or has "
            "not accumulated captured rows since; populates on redeploy"
        )
        return base

    # Monotonicity across acted arms with ≥1 sample, in multiplier order.
    present = [o for o in arms_out if o["n"] > 0]
    if len(present) >= 2:
        steps = len(present) - 1
        nondec = sum(
            1 for j in range(steps)
            if present[j + 1]["mean_realized_5d"] >= present[j]["mean_realized_5d"]
        )
        base["arm_monotone_fraction"] = round(nondec / steps, 4)

    head = acted["strong_headwind"][5]
    tail = acted["strong_tailwind"][5]
    if n_acted < MIN_TOTAL or len(head) < MIN_ARM_N or len(tail) < MIN_ARM_N:
        base["verdict"] = "INSUFFICIENT_DATA"
        base["hint"] = (
            f"captured rows exist (n_captured={n_captured}) but need "
            f"≥{MIN_TOTAL} ACTED and ≥{MIN_ARM_N} in BOTH extreme arms; "
            f"have n_acted={n_acted}, strong_headwind={len(head)}, "
            f"strong_tailwind={len(tail)}, abstained={n_abstained}"
        )
        return base

    head_mean = float(np.mean(head))
    tail_mean = float(np.mean(tail))
    spread = tail_mean - head_mean
    base["strong_tailwind_minus_headwind_pp"] = round(spread, 4)

    if spread < -EDGE_TOL_PP:
        base["verdict"] = "GATE_HARMFUL"
        base["hint"] = (
            f"as deployed: strong_tailwind realized {tail_mean:+.2f}% < "
            f"strong_headwind {head_mean:+.2f}% (spread {spread:+.2f}pp) — "
            f"the gate's actual ×1.30 calls underperformed its ×0.60 calls; "
            f"the deployed gate inverted capital allocation"
        )
    elif abs(spread) <= EDGE_TOL_PP:
        base["verdict"] = "GATE_INEFFECTIVE"
        base["hint"] = (
            f"as deployed: strong_tailwind {tail_mean:+.2f}% vs "
            f"strong_headwind {head_mean:+.2f}% (spread {spread:+.2f}pp, "
            f"within ±{EDGE_TOL_PP:.1f}pp) — the multipliers reallocated "
            f"with no realized edge: pure added sizing variance"
        )
    else:
        base["verdict"] = "GATE_EFFECTIVE"
        base["hint"] = (
            f"as deployed: strong_tailwind realized {tail_mean:+.2f}% > "
            f"strong_headwind {head_mean:+.2f}% (spread {spread:+.2f}pp) — "
            f"the gate's actual bigger bets earned the multiplier; its "
            f"ordering was economically justified"
        )
    return base


def analyze(outcomes_path: Path | str, oos_only: bool = True) -> dict:
    """Load the outcomes file, take the temporal-OOS slice (default) and run
    the realized-gate report. **No scorer / pickle is loaded** — this reads
    only the captured fields. Read-only; never raises."""
    out: dict = {
        "status": "error", "verdict": "GATE_CAPTURE_NOT_YET_POPULATED",
        "measurement": "captured_then_deployed_no_reprediction",
        "n_captured": 0, "n_acted": 0, "arms": [], "slice": "all", "hint": "",
    }
    try:
        p = Path(outcomes_path)
        if not p.exists():
            out["hint"] = f"no outcomes file at {p}"
            return out
        records: list[dict] = []
        for ln in p.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                if isinstance(obj, dict):
                    records.append(obj)
            except Exception:
                continue

        slice_name = "all"
        recs = records
        if oos_only:
            try:
                from paper_trader.validation import split_outcomes_temporal
                _, oos = split_outcomes_temporal(records, oos_fraction=0.2)
                if oos:
                    recs = oos
                    slice_name = "oos"
            except Exception:
                slice_name = "all"

        rep = gate_realized_report(recs)
        rep["slice"] = slice_name
        rep["n_records_total"] = len(records)
        return rep
    except Exception as e:  # pragma: no cover - defensive, never raises
        out["hint"] = f"audit failed: {type(e).__name__}: {e}"
        return out


def _cli(argv: list[str] | None = None) -> int:
    """`python3 -m paper_trader.ml.gate_realized [--all]` — the gate's
    REALIZED arm-effect from its captured then-deployed decision (no
    re-prediction). Read-only. Exits 2 on ``GATE_HARMFUL`` (cron-branchable —
    the actionable "the gate as actually deployed inverted allocation"
    signal, mirroring `gate_pnl`'s exit-2-on-subtracts)."""
    import sys
    argv = sys.argv[1:] if argv is None else argv
    oos_only = "--all" not in argv
    root = Path(__file__).resolve().parent.parent.parent
    rep = analyze(root / "data" / "decision_outcomes.jsonl", oos_only=oos_only)
    print(f"VERDICT: {rep['verdict']}  ({rep.get('hint', '')})")
    print(f"  slice={rep.get('slice')}  "
          f"n_captured={rep.get('n_captured')}  "
          f"n_acted={rep.get('n_acted')}  "
          f"n_abstained={rep.get('n_abstained')}  "
          f"tail−head={rep.get('strong_tailwind_minus_headwind_pp')}pp  "
          f"arm_monotone={rep.get('arm_monotone_fraction')}")
    for a in rep.get("arms", []):
        mr = a["mean_realized_5d"]
        mr_s = f"{mr:+7.2f}%" if mr is not None else "    n/a"
        print(f"  {a['arm']:<16} ×{a['multiplier']:.2f}  "
              f"n={a['n']:<5} mean_realized_5d={mr_s}")
    return 2 if rep.get("verdict") == "GATE_HARMFUL" else 0


if __name__ == "__main__":
    raise SystemExit(_cli())

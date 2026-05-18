"""Forward-return horizon audit — is the DecisionScorer's ~0 OOS skill a
*target-horizon* problem, or do the signals carry no edge at any horizon?

This is a **read-only diagnostic**. It never trains, never touches
`decision_scorer.pkl`, `decision_outcomes.jsonl`, `build_features`,
`N_FEATURES`, or any trade path — same operational discipline as
`paper_trader/ml/calibration.py` / `gate_audit.py` / `skill_trend.py` /
`baseline_compare.py`. Safe to run against the live unattended loop.

**Why this is not any existing tool.** The ML/backtest domain already has
a saturated diagnostic suite — calibration (decile rank skill), gate_audit
(economic spread of the 5 gate arms), skill_trend (oos_rmse vs the constant
mean-predictor), baseline_compare (MLP vs one-line rules), regime_audit
(regime-conditional skill). **Every one of them can only ever measure skill
against `forward_return_5d`** — that is the *only* realized label
`_compute_decision_outcomes` historically captured. None of them can answer
the decisive skeptical-quant question that follows from their shared
"`oos_ic ≈ 0`, `gate_active=true`" finding:

  *Is the scorer near-blind because the 17 features carry no signal — or
  because the 5-trading-day target is just too noisy a thing to predict?*

AGENTS.md states it explicitly: leveraged ETFs (a large slice of the
watchlist and persona boosts) "have noisy 5d windows but strong 3-12 month
returns" — the documented reason an earlier HOLD-blocking gate oscillated.
If the same primary signals rank-predict the 10d/20d realized return
materially better than the 5d one, the scorer's near-zero OOS skill is a
*target-horizon artifact*, not a dead feature set — a concrete, actionable
finding that points future work at the target, not the model. If no horizon
has edge, that is the strongest skeptical conclusion of all.

**Method.** `_compute_decision_outcomes` now additively records
`forward_return_10d` / `forward_return_20d` alongside the unchanged
`forward_return_5d` (the scorer still trains ONLY on 5d). On the
**temporal-OOS slice** (`validation.split_outcomes_temporal` — the EXACT
split `_train_decision_scorer` uses for `oos_rmse`/`oos_ic`, so this and the
ledger describe the same holdout), probe each horizon with the two signals
that actually drive `_ml_decide`:

  * `ml_score` — feature slot 0, the dominant signal the decision was made on
  * `mom20`    — 20-day momentum carry (the classic horizon-sensitive signal:
                 short windows mean-revert, long windows trend — if any
                 horizon difference exists at all, momentum surfaces it)

Skill is **tie-aware Spearman rank-IC** (`calibration._spearman`, single
source of truth — tie-awareness is load-bearing across the codebase). The
codebase-universal SELL sign-flip (`-forward_return`) is applied to the
realized target *and* the probe (mirroring `baseline_compare._aligned` so a
feature is not spuriously anti-correlated on the SELL subset). Each horizon
is scored on its own resolvable subset (10d/20d are None near a window's
tail) with the per-cell `n` disclosed — the honest "how predictable is
horizon h" question, not a forced common window.

| Verdict | Meaning |
|---------|---------|
| `INSUFFICIENT_DATA` | < `MIN_PAIRS` resolvable OOS pairs even at 5d |
| `INSUFFICIENT_LONG_HORIZON` | 5d resolvable but no longer horizon has ≥ `MIN_PAIRS` yet — the expected state until the loop accumulates rows under the new multi-horizon capture |
| `NO_HORIZON_HAS_EDGE` | best probe \\|rank-IC\\| < `EDGE_FLOOR` at *every* horizon — the signals carry no edge at any horizon (strongest skeptical conclusion) |
| `LONGER_HORIZON_MORE_PREDICTABLE` | best longer-horizon IC > 5d IC + `IC_MARGIN` AND > `EDGE_FLOOR` — the 5d target is handicapping the scorer; the signal lives at a longer horizon |
| `5D_ADEQUATE` | 5d IC is within `IC_MARGIN` of (or beats) the best longer horizon — the target horizon is not the bottleneck |

```bash
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.horizon_audit
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.horizon_audit --all
```
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .calibration import _spearman
from .decision_scorer import _to_float

# Horizons probed. 5 is the scorer's training/gate target (the anchor every
# other diagnostic is locked to); 10/20 are the additive research labels.
HORIZONS = (5, 10, 20)

# Probe signals — the two inputs that actually move `_ml_decide`. Each maps a
# decision_outcomes row to a scalar "predicted goodness of this BUY"
# (pre SELL-flip; the flip is applied uniformly by `_aligned`).
PROBES: dict[str, callable] = {
    "ml_score": lambda r: _to_float(r.get("ml_score"), 0.0),
    "mom20": lambda r: _to_float(r.get("mom20"), 0.0),
}

# Verdict thresholds — module-level so tests assert exact verdicts and a
# tuning change is a single reviewable edit (mirrors calibration.py /
# gate_audit.py / baseline_compare.py).
MIN_PAIRS = 30        # need a real per-horizon OOS sample before grading it
IC_MARGIN = 0.05      # rank-IC margin a longer horizon must clear to be decisive
EDGE_FLOOR = 0.10     # real-rank-skill floor (== calibration.SPEARMAN_MIN)


def _aligned(value: float, fwd: float, is_sell: bool) -> tuple[float, float]:
    """`(probe, target)` in the codebase-universal action-aligned space — a
    drop after a SELL was the *right* call, so both the realized return and
    the raw probe are negated on a SELL (exactly `baseline_compare._aligned`
    / `train_scorer` / `calibration.scorer_calibration`)."""
    return (-value if is_sell else value), (-fwd if is_sell else fwd)


def _horizon_skill(records: list[dict], probe, horizon: int) -> dict:
    """Tie-aware rank-IC of `probe` vs the action-aligned realized return at
    `horizon` trading days, over the records that resolve that horizon.

    Returns ``{n, rank_ic}`` (rank_ic None when degenerate / < 2 pairs).
    Never raises."""
    key = f"forward_return_{horizon}d"
    preds: list[float] = []
    tgts: list[float] = []
    try:
        for r in records:
            raw = r.get(key)
            if raw is None:
                continue
            # nan default → a non-finite / unparseable / absent label is
            # DROPPED, never coerced to a fake 0.0 outcome (that would bias
            # the IC toward the constant predictor).
            fwd = _to_float(raw, float("nan"))
            if fwd != fwd:
                continue
            try:
                pv = probe(r)
            except Exception:
                continue
            if pv is None or pv != pv:
                continue
            is_sell = str(r.get("action") or "BUY").upper() == "SELL"
            ap, at = _aligned(float(pv), float(fwd), is_sell)
            preds.append(ap)
            tgts.append(at)
        n = len(preds)
        out = {"n": n, "rank_ic": None}
        if n < 2:
            return out
        p = np.asarray(preds, dtype=np.float64)
        t = np.asarray(tgts, dtype=np.float64)
        if float(np.std(p)) == 0.0 or float(np.std(t)) == 0.0:
            return out  # degenerate — _spearman would (correctly) give 0.0
        ic = _spearman(p, t)
        if ic == ic:  # not NaN
            out["rank_ic"] = round(float(ic), 4)
        return out
    except Exception:
        return {"n": len(preds), "rank_ic": None}


def horizon_audit_report(records: list[dict]) -> dict:
    """Grade every (probe, horizon) cell and emit one crisp verdict.

    `records` are decision_outcomes rows (already restricted to the slice the
    caller wants — `analyze` passes the temporal-OOS slice). Never raises;
    any fault degrades to ``INSUFFICIENT_DATA``.
    """
    base: dict = {
        "status": "ok",
        "verdict": "INSUFFICIENT_DATA",
        "n_records": 0,
        "cells": [],            # [{probe, horizon, n, rank_ic}]
        "best_ic_by_horizon": {},  # {horizon: best probe rank_ic (or None)}
        "best_5d_ic": None,
        "best_long_ic": None,
        "best_long_horizon": None,
        "hint": "",
    }
    try:
        recs = list(records or [])
        base["n_records"] = len(recs)

        cells = []
        best_by_h: dict[int, float | None] = {}
        n_by_h: dict[int, int] = {}
        for h in HORIZONS:
            h_best: float | None = None
            h_n = 0
            for pname, pfn in PROBES.items():
                s = _horizon_skill(recs, pfn, h)
                cells.append({
                    "probe": pname, "horizon": h,
                    "n": s["n"], "rank_ic": s["rank_ic"],
                })
                h_n = max(h_n, s["n"])
                if (s["rank_ic"] is not None
                        and (h_best is None or s["rank_ic"] > h_best)):
                    h_best = s["rank_ic"]
            best_by_h[h] = h_best
            n_by_h[h] = h_n
        base["cells"] = cells
        base["best_ic_by_horizon"] = {str(h): best_by_h[h] for h in HORIZONS}

        # 5d is the anchor. Need it well-sampled to make ANY comparison.
        if n_by_h.get(5, 0) < MIN_PAIRS:
            base["hint"] = (
                f"need ≥{MIN_PAIRS} resolvable OOS pairs at 5d; "
                f"have n_5d={n_by_h.get(5, 0)}"
            )
            return base

        five_ic = best_by_h.get(5)
        base["best_5d_ic"] = five_ic

        # Longer horizons that are themselves well-sampled.
        long_candidates = [
            (h, best_by_h[h]) for h in HORIZONS
            if h > 5 and n_by_h.get(h, 0) >= MIN_PAIRS
            and best_by_h[h] is not None
        ]
        if not long_candidates:
            base["verdict"] = "INSUFFICIENT_LONG_HORIZON"
            base["hint"] = (
                f"5d sampled (n={n_by_h.get(5, 0)}) but no longer horizon has "
                f"≥{MIN_PAIRS} resolvable pairs yet "
                f"(n_10d={n_by_h.get(10, 0)}, n_20d={n_by_h.get(20, 0)}) — "
                f"the multi-horizon capture populates as the loop runs"
            )
            return base

        best_long_h, best_long_ic = max(long_candidates, key=lambda x: x[1])
        base["best_long_ic"] = best_long_ic
        base["best_long_horizon"] = best_long_h

        # absolute edge — the scorer/gate acts on sign, so |IC| is the
        # relevant magnitude (a strongly negative IC is also exploitable,
        # just inverted; near-zero is the dead case).
        all_best = [v for v in best_by_h.values() if v is not None]
        max_abs = max((abs(v) for v in all_best), default=0.0)
        if max_abs < EDGE_FLOOR:
            base["verdict"] = "NO_HORIZON_HAS_EDGE"
            base["hint"] = (
                f"best |rank-IC| across all horizons is {max_abs:.3f} "
                f"< {EDGE_FLOOR} — the probe signals carry no rank skill at "
                f"5d, 10d OR 20d (a dead feature set, not a horizon problem)"
            )
            return base

        five_ref = five_ic if five_ic is not None else 0.0
        if best_long_ic > five_ref + IC_MARGIN and best_long_ic > EDGE_FLOOR:
            base["verdict"] = "LONGER_HORIZON_MORE_PREDICTABLE"
            base["hint"] = (
                f"best {best_long_h}d rank-IC {best_long_ic:+.3f} clears 5d "
                f"{five_ref:+.3f} by >{IC_MARGIN} — the scorer's ~0 OOS skill "
                f"is a 5d-target-noise artifact; the signal lives at "
                f"{best_long_h}d"
            )
        else:
            base["verdict"] = "5D_ADEQUATE"
            base["hint"] = (
                f"5d rank-IC {five_ref:+.3f} is within {IC_MARGIN} of the "
                f"best longer horizon ({best_long_h}d {best_long_ic:+.3f}) — "
                f"the target horizon is not the bottleneck"
            )
        return base
    except Exception as e:  # pragma: no cover - defensive, never raises
        base["verdict"] = "INSUFFICIENT_DATA"
        base["hint"] = f"audit failed: {type(e).__name__}: {e}"
        return base


def analyze(outcomes_path: Path | str, oos_only: bool = True) -> dict:
    """Load the outcomes file, take the temporal-OOS slice (default) and run
    the horizon audit. Read-only; never raises."""
    out: dict = {"status": "ok", "verdict": "INSUFFICIENT_DATA",
                 "n_records": 0, "cells": [], "slice": "all", "hint": ""}
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
                records.append(json.loads(ln))
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

        rep = horizon_audit_report(recs)
        rep["slice"] = slice_name
        rep["n_records_total"] = len(records)
        return rep
    except Exception as e:  # pragma: no cover - defensive
        out["hint"] = f"audit failed: {type(e).__name__}: {e}"
        return out


def _cli(argv: list[str] | None = None) -> int:
    """`python3 -m paper_trader.ml.horizon_audit [--all]` — read-only
    forward-return-horizon predictability audit of the accumulated outcomes."""
    argv = sys.argv[1:] if argv is None else argv
    oos_only = "--all" not in argv
    root = Path(__file__).resolve().parent.parent.parent
    rep = analyze(root / "data" / "decision_outcomes.jsonl", oos_only=oos_only)
    print(f"VERDICT: {rep['verdict']}  ({rep.get('hint', '')})")
    blh = rep.get("best_long_horizon")
    blh_s = f"@{blh}d" if blh is not None else "@n/a"
    print(f"  slice={rep.get('slice')}  n_records={rep.get('n_records')}  "
          f"best_5d_ic={rep.get('best_5d_ic')}  "
          f"best_long_ic={rep.get('best_long_ic')} ({blh_s})")
    print(f"  {'probe':<10} {'horizon':>7} {'n':>6} {'rank_ic':>9}")
    for c in rep.get("cells", []):
        ic = c["rank_ic"]
        ic_s = f"{ic:+.4f}" if ic is not None else "    n/a"
        print(f"  {c['probe']:<10} {c['horizon']:>6}d {c['n']:>6} {ic_s:>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())

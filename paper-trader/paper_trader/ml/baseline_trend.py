"""Baseline-trend diagnostic — read-only.

`run_continuous_backtests.py::_append_baseline_skill_log` writes one
structured row per cycle to `data/baseline_skill_log.jsonl` (status, verdict,
`mlp_rank_ic`, `best_baseline`, `best_baseline_ic`, `ic_gap`, `gate_active`,
`n_train`). That ledger exists to make the single most economically-decisive
recurring ML/backtest finding durable and trendable:

    the 17-feature MLP carries *less* out-of-sample rank skill than a one-line
    rule (raw `ml_score`, feature slot 0) — `MLP_WORSE_THAN_TRIVIAL` — yet the
    `_ml_decide` conviction gate (invariant #5, active every cycle once
    `n_train ≥ 500`) sizes real positions on the MLP's prediction.

`skill_trend.py` already trends the *scorer-skill* ledger (`oos_rmse` vs a
constant mean-predictor). **Nothing trended the baseline ledger** — the
`ic_gap = MLP_rank_ic − best_one_liner_rank_ic` column was written every cycle
but never read, so a skeptical quant still had to run
`python3 -m paper_trader.ml.baseline_compare` by hand for a point-in-time
answer and could not see whether the net stays net-negative-complexity,
recovers, or worsens as `decision_outcomes.jsonl` accumulates. This is the
exact gap `skill_trend` filled for the sibling ledger; this module is its
counterpart for the baseline ledger.

Same operational discipline as `paper_trader/ml/calibration.py` /
`skill_trend.py`: read-only, no train, no pickle / `build_features` /
`N_FEATURES` touch, no trade path — safe to run against the live unattended
loop. Never raises on bad input.

`IC_MARGIN` / `MLP_IC_MIN` are imported from `baseline_compare` (single
source of truth — this trends *that* tool's per-cycle verdict, so the
margins must match by construction, the `_oos_rank_metrics`-reuses-`_spearman`
precedent).

```bash
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.baseline_trend
```
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from .baseline_compare import IC_MARGIN, MLP_IC_MIN

# Need this many usable ledger rows (status=="ok" with a finite ic_gap)
# before a trend verdict is meaningful (mirrors skill_trend.MIN_CYCLES).
MIN_CYCLES = 5
# Rolling window for the "recent" aggregate vs the older tail.
RECENT_CYCLES = 10


def _median(xs: list) -> float | None:
    vals = [float(x) for x in xs if x is not None and float(x) == float(x)]
    if not vals:
        return None
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def load_baseline_ledger(path: Path | str) -> list[dict]:
    """Robust JSONL load of the baseline-skill ledger. Skips unparseable lines.

    Never raises — a missing/corrupt ledger yields ``[]`` so callers degrade
    to ``INSUFFICIENT_DATA`` rather than crashing (the ledger is best-effort
    by construction; a reader of it must be too — the `load_skill_ledger`
    precedent)."""
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


def baseline_trend_report(ledger_rows: list[dict],
                          recent_n: int = RECENT_CYCLES) -> dict:
    """Aggregate the baseline ledger into a trended verdict on whether the
    MLP earns its complexity out of sample.

    A row is *usable* iff ``status == "ok"`` AND ``ic_gap`` is a finite
    number — a ``verdict == "INSUFFICIENT_DATA"`` cycle (scorer untrained or
    too few OOS pairs) carries ``ic_gap = None`` and is excluded, exactly as
    ``skill_trend`` excludes non-``ok`` / null-``oos_rmse`` rows.

    ``ic_gap = MLP_rank_ic − best_one_liner_rank_ic``. Higher is better:
    ``ic_gap ≥ 0`` means the net matches/beats every cheap baseline; a
    persistently negative gap is the documented net-negative-complexity state.

    Verdicts (exact-value test-locked in tests/test_baseline_trend.py),
    decided on the **recent** median ``ic_gap`` so the call reflects the
    current regime, not stale history:

      * ``INSUFFICIENT_DATA``         — < MIN_CYCLES usable rows
      * ``MLP_WORSE_THAN_TRIVIAL``    — recent median ic_gap ≤ −IC_MARGIN
                                        (a one-liner persistently beats the net)
      * ``MLP_ADDS_SKILL``            — recent median ic_gap ≥ +IC_MARGIN AND
                                        recent median mlp_rank_ic > MLP_IC_MIN
                                        (genuinely additive AND itself skilled)
      * ``MLP_NO_BETTER_THAN_TRIVIAL``— otherwise (within ±IC_MARGIN, or a
                                        positive gap the MLP's own sub-floor
                                        rank skill doesn't justify)

    ``trend`` compares recent vs older median ic_gap (higher = better, so
    recent > older ⇒ IMPROVING). The window-specific noise in ``ic_gap`` is
    large, so the verdict is intentionally taken on the *median*, not the mean.
    """
    out: dict = {
        "verdict": "INSUFFICIENT_DATA",
        "n_cycles_total": len(ledger_rows),
        "n_cycles_usable": 0,
        "recent_n": recent_n,
        "recent_median_ic_gap": None,
        "older_median_ic_gap": None,
        "overall_median_ic_gap": None,
        "recent_median_mlp_rank_ic": None,
        "recent_median_best_baseline_ic": None,
        "recent_median_n_train": None,
        "most_common_best_baseline": None,
        "gate_active_fraction": None,
        "trend": "UNKNOWN",
        "hint": "",
    }
    if ledger_rows:
        ga = [1.0 if r.get("gate_active") else 0.0 for r in ledger_rows]
        out["gate_active_fraction"] = round(sum(ga) / len(ga), 4)

    def _finite(v) -> bool:
        try:
            return v is not None and float(v) == float(v)
        except (TypeError, ValueError):
            return False

    ok = [r for r in ledger_rows
          if str(r.get("status")) == "ok" and _finite(r.get("ic_gap"))]
    n = len(ok)
    out["n_cycles_usable"] = n

    if n < MIN_CYCLES:
        out["hint"] = (f"need ≥{MIN_CYCLES} usable cycles "
                       f"(status==ok with a finite ic_gap); have {n}")
        return out

    recent = ok[-recent_n:]
    older = ok[:-recent_n] if len(ok) > recent_n else []

    rec_gap = _median([r.get("ic_gap") for r in recent])
    old_gap = _median([r.get("ic_gap") for r in older]) if older else None
    out["recent_median_ic_gap"] = (round(rec_gap, 4)
                                   if rec_gap is not None else None)
    out["older_median_ic_gap"] = (round(old_gap, 4)
                                  if old_gap is not None else None)
    out["overall_median_ic_gap"] = round(
        _median([r.get("ic_gap") for r in ok]), 4)

    rec_mlp_ic = _median([r.get("mlp_rank_ic") for r in recent])
    out["recent_median_mlp_rank_ic"] = (round(rec_mlp_ic, 4)
                                        if rec_mlp_ic is not None else None)
    rec_base_ic = _median([r.get("best_baseline_ic") for r in recent])
    out["recent_median_best_baseline_ic"] = (
        round(rec_base_ic, 4) if rec_base_ic is not None else None)
    rec_ntrain = _median([r.get("n_train") for r in recent])
    out["recent_median_n_train"] = (int(rec_ntrain)
                                    if rec_ntrain is not None else None)

    # Which one-liner keeps beating the net? On the live corpus this is
    # ``ml_score`` (feature slot 0) — the decisive detail: the signal the
    # MLP is fed is the signal it destroys.
    names = [r.get("best_baseline") for r in recent if r.get("best_baseline")]
    if names:
        out["most_common_best_baseline"] = Counter(names).most_common(1)[0][0]

    # Trend: higher ic_gap is better, so recent > older ⇒ IMPROVING.
    if old_gap is not None and rec_gap is not None:
        if rec_gap >= old_gap + IC_MARGIN:
            out["trend"] = "IMPROVING"
        elif rec_gap <= old_gap - IC_MARGIN:
            out["trend"] = "DEGRADING"
        else:
            out["trend"] = "STABLE"

    if rec_gap <= -IC_MARGIN:
        out["verdict"] = "MLP_WORSE_THAN_TRIVIAL"
        out["hint"] = (
            f"recent median ic_gap {rec_gap:+.3f} ≤ -{IC_MARGIN} — the best "
            f"one-liner ('{out['most_common_best_baseline']}') persistently "
            f"beats the 17-feature MLP out of sample; the conviction gate "
            f"(invariant #5) underwrites sizing variance with no compensating "
            f"edge")
    elif (rec_gap >= IC_MARGIN and rec_mlp_ic is not None
          and rec_mlp_ic > MLP_IC_MIN):
        out["verdict"] = "MLP_ADDS_SKILL"
        out["hint"] = (
            f"recent median ic_gap {rec_gap:+.3f} ≥ {IC_MARGIN} and MLP "
            f"rank_ic {rec_mlp_ic:+.3f} > {MLP_IC_MIN} floor — the net is "
            f"genuinely additive beyond every cheap baseline")
    else:
        out["verdict"] = "MLP_NO_BETTER_THAN_TRIVIAL"
        mlp_s = ("n/a" if rec_mlp_ic is None else f"{rec_mlp_ic:+.3f}")
        out["hint"] = (
            f"recent median ic_gap {rec_gap:+.3f} within ±{IC_MARGIN} (or a "
            f"positive gap the MLP's own rank_ic {mlp_s} ≤ {MLP_IC_MIN} floor "
            f"does not justify) — the net's complexity buys no durable OOS "
            f"edge a single feature doesn't already carry")
    return out


def analyze(ledger_path: Path | str) -> dict:
    """Load the baseline ledger + return the full trend report. Read-only;
    never raises."""
    rows = load_baseline_ledger(ledger_path)
    return baseline_trend_report(rows)


def _cli(argv: list[str] | None = None) -> int:
    """`python3 -m paper_trader.ml.baseline_trend` — read-only trend of the
    per-cycle baseline ledger: is the MLP *still* net-negative complexity
    out of sample, and is that improving or worsening?

    Exit code mirrors the sibling whose verdict this trends
    (`baseline_compare`): 0 on ``MLP_ADDS_SKILL`` / ``INSUFFICIENT_DATA``,
    2 on ``MLP_WORSE_THAN_TRIVIAL`` / ``MLP_NO_BETTER_THAN_TRIVIAL`` — so an
    operator/cron can branch on "the net persistently fails to earn its
    complexity" exactly like `baseline_compare` / `gate_audit`.
    """
    root = Path(__file__).resolve().parent.parent.parent
    ledger = root / "data" / "baseline_skill_log.jsonl"
    rep = analyze(ledger)
    print(f"VERDICT: {rep['verdict']}  ({rep['hint']})")
    print(f"  cycles: {rep['n_cycles_usable']} usable / "
          f"{rep['n_cycles_total']} total   "
          f"gate_active={rep['gate_active_fraction']}")
    print(f"  ic_gap(MLP−best)  recent={rep['recent_median_ic_gap']}  "
          f"older={rep['older_median_ic_gap']}  "
          f"overall={rep['overall_median_ic_gap']}  "
          f"trend={rep['trend']}")
    print(f"  recent MLP rank_ic={rep['recent_median_mlp_rank_ic']}  "
          f"best one-liner ic={rep['recent_median_best_baseline_ic']} "
          f"({rep['most_common_best_baseline']})  "
          f"n_train={rep['recent_median_n_train']}")
    if rep["verdict"] in ("MLP_WORSE_THAN_TRIVIAL",
                          "MLP_NO_BETTER_THAN_TRIVIAL"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())

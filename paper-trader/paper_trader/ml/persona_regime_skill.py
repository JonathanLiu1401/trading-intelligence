"""Per-persona × per-regime decision-signal skill cross-tabulation.

This is the missing intersection of two existing reads:

* ``paper_trader/ml/persona_skill.py`` aggregates each persona's signal
  skill across every regime — it can hide a persona that's
  +0.25 IC in bear but -0.10 IC in bull (the bull and bear cells
  cancel each other in the aggregate).
* ``paper_trader/ml/regime_audit.py`` aggregates each regime's signal
  skill across every persona — it can hide a regime where one
  persona is strong and another is anti-predictive (the per-persona
  ICs average out within the regime).

Neither view answers the actionable question: **does a particular
persona carry real signal in a particular regime?** That cross-tab
is what this module produces, reading the same
``decision_outcomes.jsonl`` rows the sibling diagnostics already use
(both ``persona`` and ``regime_label`` are persisted alongside the
signed ``ml_score`` and ``forward_return_5d``).

Operational discipline mirrors ``persona_skill`` exactly:

* **Read-only.** Never trains, never touches ``decision_scorer.pkl``,
  ``build_features``, ``N_FEATURES``, or any trade path. Safe to run
  against the unattended continuous loop.
* **SSOT cross-checks.** ``persona_for`` is imported from
  ``backtest.py``; ``_spearman`` is imported from
  ``ml.calibration``; the regime decode mirrors
  ``ml.regime_audit.REGIME_FROM_MULT`` exactly. A future PERSONAS
  reorder or a regime-multiplier change cannot silently shift this
  cross-tab.
* **Action-aligned signal + target.** Same double-flip convention
  every sibling uses: a SELL's ``forward_return_5d`` and
  ``ml_score`` are both negated so "higher signal → higher
  realized goodness" has one consistent meaning across BUY and
  SELL.

Verdict per cell:

| Verdict           | Meaning                                                    |
|-------------------|------------------------------------------------------------|
| ``INSUFFICIENT``  | < ``MIN_PER_CELL`` aligned outcomes — no stable IC.        |
| ``INVERTED``      | ``score_ic ≤ -IC_GOOD`` — signal is anti-predictive in this regime; the more confident the persona is, the WORSE it does. |
| ``SIGNAL_EDGE``   | ``score_ic ≥ IC_GOOD`` — signal genuinely rank-predicts realized goodness in this regime. |
| ``WEAK_SIGNAL``   | ``IC_MIN ≤ score_ic < IC_GOOD`` — usable as a tie-breaker, not a primary signal. |
| ``NO_EDGE``       | ``-IC_GOOD < score_ic < IC_MIN`` — no demonstrated rank skill in this regime. |

Overall verdict:

| Verdict                  | Meaning                                              |
|--------------------------|------------------------------------------------------|
| ``INSUFFICIENT_DATA``    | < ``MIN_RECORDS`` aligned rows overall.              |
| ``HAS_INVERTED_CELL``    | ≥1 cell is ``INVERTED`` — actionable red flag (a specific persona-in-a-specific-regime is harmful). |
| ``REGIME_CONDITIONAL``   | ≥1 ``SIGNAL_EDGE`` cell AND ≥1 ``NO_EDGE`` cell — the persona's aggregate skill is regime-conditional, not uniform. |
| ``NO_PERSONA_EDGE``      | No cell reaches ``SIGNAL_EDGE`` — every persona-regime pair is at noise or worse. |
| ``HEALTHY``              | At least one ``SIGNAL_EDGE`` cell and no ``INVERTED`` cell. |

CLI:

```bash
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.persona_regime_skill
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.persona_regime_skill --json
```
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Single source of truth for run_id → persona — same import the sibling
# persona_skill module uses, so any future PERSONAS-dict reorder cannot
# silently shift this cross-tab.
from paper_trader.backtest import persona_for
from paper_trader.ml.calibration import _spearman
from paper_trader.ml.decision_scorer import _to_float

# Module-level constants so tests assert exact verdicts and any tuning
# move is a single reviewable edit (the codebase convention every sibling
# ml diagnostic follows).
MIN_RECORDS = 30        # overall floor — below this verdict is INSUFFICIENT_DATA
MIN_PER_CELL = 20       # per-cell stability bar; below this the IC is noise
IC_MIN = 0.05           # rank-skill bar (mirrors persona_skill / skill_trend.IC_MIN)
IC_GOOD = 0.15          # "real edge" bar — mirrors persona_skill.IC_GOOD

# Regime decode — MUST mirror ``backtest.py::_market_regime`` /
# ``run_continuous_backtests._compute_decision_outcomes`` /
# ``regime_audit.REGIME_FROM_MULT``. Rounded to 2 decimals so a
# JSON-serialized 0.6000000001 still resolves. "unknown" cycles
# (SPY history <200 days for the early window days) write
# ``regime_mult=1.0`` AND ``regime_label='unknown'``; we drop those
# rows explicitly — ``regime_mult`` alone cannot separate bull from
# unknown, but ``regime_label`` (2026-05-19 feature) can, and the
# documented near-zero OOS skill IS partly a regime-mix artifact of
# this exact contamination (per-regime cells should not be
# polluted).
_REGIME_FROM_MULT: dict[float, str] = {
    0.30: "bear",
    0.60: "sideways",
    1.00: "bull",
}
REGIME_ORDER = ("bull", "sideways", "bear")  # display order


def _regime_of(record: dict) -> str | None:
    """Return the regime label for one outcome row, or ``None`` to drop.

    Prefers the explicit ``regime_label`` field (2026-05-19 feature) so
    a true "unknown" cycle drops out honestly. Falls back to the
    ``regime_mult`` decode for legacy rows that pre-date the field —
    that decode cannot separate bull from unknown, but rows from
    before the feature shipped don't have the label to disambiguate,
    so the legacy decode is the best we can do for them. An explicit
    label of "unknown" (or any value outside the three known regimes)
    is dropped — that is the documented contamination route the
    aggregate diagnostics silently bucketed as "bull".
    """
    label = record.get("regime_label")
    if isinstance(label, str):
        if label in REGIME_ORDER:
            return label
        # Explicit "unknown" or any other label drops out honestly.
        return None
    if label is not None:
        # A non-string label (None already handled above, but defensive
        # against a malformed record carrying e.g. an int label) is
        # not safe to trust — drop.
        return None
    # Legacy row: no regime_label, fall back to multiplier decode.
    v = _to_float(record.get("regime_mult"), float("nan"))
    if v != v:  # NaN — unparseable / missing
        return None
    return _REGIME_FROM_MULT.get(round(v, 2))


def _aligned(record: dict) -> tuple[float, float] | None:
    """Return ``(signal, target)`` both action-aligned, or ``None`` to drop.

    Identical convention to ``persona_skill._aligned`` — the SELL
    double-flip on both signal and target keeps "higher signal ⇒
    higher realized goodness" monotone across BUY and SELL. A
    missing or non-finite ml_score / forward_return_5d drops the row.
    """
    fr = record.get("forward_return_5d")
    ms = record.get("ml_score")
    if fr is None or ms is None:
        return None
    t = _to_float(fr, float("nan"))
    s = _to_float(ms, float("nan"))
    if t != t or s != s:
        return None
    if str(record.get("action") or "BUY").upper() == "SELL":
        t = -t
        s = -s
    return s, t


def _verdict_for_cell(n: int, ic: float) -> str:
    """Crisp, exactly testable per-cell verdict from (n, score_ic)."""
    if n < MIN_PER_CELL:
        return "INSUFFICIENT"
    if ic <= -IC_GOOD:
        return "INVERTED"
    if ic >= IC_GOOD:
        return "SIGNAL_EDGE"
    if ic >= IC_MIN:
        return "WEAK_SIGNAL"
    return "NO_EDGE"


def analyze(records) -> dict:
    """Cross-tabulate ``decision_outcomes.jsonl`` rows by (persona, regime)
    and report each cell's decision-signal rank-IC.

    ``records`` is any iterable of dicts with at least ``run_id``,
    ``action``, ``ml_score``, ``forward_return_5d``, and either
    ``regime_label`` or ``regime_mult``. Rows whose persona is
    unmappable, whose signal/target is missing or non-finite, OR
    whose regime is unknown / unmappable, are dropped.

    Returns a JSON-safe dict:
    ``{status, verdict, n_records, n_cells, n_stable_cells,
       cells:[{persona, regime, n, score_ic, mean_aligned_return,
       win_rate, verdict}],
       inverted_cells:[{persona, regime, score_ic, n}],
       best_cell:{persona, regime, score_ic, n} | None,
       worst_cell:{persona, regime, score_ic, n} | None,
       n_dropped_unknown_regime: int,
       hint}``

    ``cells`` is sorted by ``-score_ic`` (best skill first), with
    ``INSUFFICIENT`` cells always sinking to the end regardless of
    their unstable IC value. ``best_cell`` / ``worst_cell`` are
    chosen from the STABLE cells only (verdict != "INSUFFICIENT")
    so they describe demonstrable skill, not a small-sample fluke.
    """
    buckets: dict[tuple[str, str], dict[str, list[float]]] = {}
    n_aligned = 0
    n_dropped_unknown_regime = 0

    for r in records:
        rid = r.get("run_id")
        try:
            persona = persona_for(int(rid))["name"]
        except Exception:
            continue
        reg = _regime_of(r)
        if reg is None:
            # Mirror the regime_audit `dropped_unmapped` honesty pattern:
            # an unknown / unmappable regime is dropped, NOT silently
            # bucketed. We count it so a quant can see how much data
            # was excluded by the unknown-regime filter.
            n_dropped_unknown_regime += 1
            continue
        pair = _aligned(r)
        if pair is None:
            continue
        s, t = pair
        n_aligned += 1
        b = buckets.setdefault((persona, reg), {"sig": [], "tgt": []})
        b["sig"].append(s)
        b["tgt"].append(t)

    if n_aligned < MIN_RECORDS:
        return {
            "status": "insufficient_data",
            "verdict": "INSUFFICIENT_DATA",
            "n_records": n_aligned,
            "n_cells": len(buckets),
            "n_stable_cells": 0,
            "cells": [],
            "inverted_cells": [],
            "best_cell": None,
            "worst_cell": None,
            "n_dropped_unknown_regime": n_dropped_unknown_regime,
            "hint": (f"need ≥{MIN_RECORDS} aligned (persona,regime) "
                     f"outcomes; have {n_aligned}"),
        }

    cells: list[dict] = []
    inverted: list[dict] = []
    for (persona, reg), b in buckets.items():
        n = len(b["sig"])
        sig = np.asarray(b["sig"], dtype=np.float64)
        tgt = np.asarray(b["tgt"], dtype=np.float64)
        ic = round(float(_spearman(sig, tgt)), 4)
        mean_ret = round(float(tgt.mean()), 4)
        win_rate = round(float(np.mean(tgt > 0.0)), 4)
        verdict = _verdict_for_cell(n, ic)
        row = {
            "persona": persona,
            "regime": reg,
            "n": n,
            "score_ic": ic,
            "mean_aligned_return": mean_ret,
            "win_rate": win_rate,
            "verdict": verdict,
        }
        cells.append(row)
        if verdict == "INVERTED":
            inverted.append({
                "persona": persona,
                "regime": reg,
                "score_ic": ic,
                "n": n,
            })

    # Sort: stable cells first, by -score_ic; INSUFFICIENT cells sink to end
    # (their IC is noise, regardless of magnitude).
    cells.sort(key=lambda d: (d["verdict"] != "INSUFFICIENT", d["score_ic"]),
               reverse=True)

    stable = [c for c in cells if c["verdict"] != "INSUFFICIENT"]
    n_stable = len(stable)
    best_cell: dict | None = None
    worst_cell: dict | None = None
    if stable:
        best = max(stable, key=lambda c: c["score_ic"])
        worst = min(stable, key=lambda c: c["score_ic"])
        best_cell = {
            "persona": best["persona"], "regime": best["regime"],
            "score_ic": best["score_ic"], "n": best["n"],
        }
        worst_cell = {
            "persona": worst["persona"], "regime": worst["regime"],
            "score_ic": worst["score_ic"], "n": worst["n"],
        }

    has_edge = any(c["verdict"] == "SIGNAL_EDGE" for c in stable)
    has_no_edge = any(c["verdict"] in ("NO_EDGE", "WEAK_SIGNAL") for c in stable)

    if inverted:
        verdict = "HAS_INVERTED_CELL"
        cells_str = ", ".join(
            f"{c['persona']}/{c['regime']}({c['score_ic']:+.2f})"
            for c in sorted(inverted, key=lambda c: c["score_ic"])
        )
        hint = (f"{len(inverted)} anti-predictive cell(s): {cells_str}. "
                f"The persona's signal is INVERTED in that specific regime — "
                f"the stronger the signal, the WORSE the realized 5d outcome. "
                f"This is the data for a separate, explicit decision to invert "
                f"or suppress that persona's behaviour in that regime; this "
                f"read-only audit does NOT change PERSONAS / _PERSONA_BOOSTS / "
                f"the gate.")
    elif n_stable == 0:
        verdict = "NO_PERSONA_EDGE"
        hint = (f"no (persona, regime) cell has ≥{MIN_PER_CELL} aligned "
                f"outcomes — every cross-tab cell is too small to evaluate")
    elif has_edge and has_no_edge:
        verdict = "REGIME_CONDITIONAL"
        hint = (f"≥1 cell shows SIGNAL_EDGE and ≥1 shows NO_EDGE/WEAK — "
                f"persona signal skill is REGIME-CONDITIONAL, not uniform; "
                f"the aggregate per-persona verdict in persona_skill HIDES "
                f"this structure")
    elif has_edge:
        verdict = "HEALTHY"
        hint = (f"≥1 (persona, regime) cell rank-predicts realized "
                f"outcomes (score_ic ≥ {IC_GOOD}) on a stable sample, "
                f"and no cell is anti-predictive")
    else:
        verdict = "NO_PERSONA_EDGE"
        hint = (f"no (persona, regime) cell rank-predicts realized "
                f"outcomes on a stable sample — per-persona returns are "
                f"leveraged-beta dispersion in every regime, not demonstrated "
                f"signal skill")

    return {
        "status": "ok",
        "verdict": verdict,
        "n_records": n_aligned,
        "n_cells": len(buckets),
        "n_stable_cells": n_stable,
        "cells": cells,
        "inverted_cells": sorted(
            inverted, key=lambda c: (c["score_ic"], c["persona"], c["regime"])
        ),
        "best_cell": best_cell,
        "worst_cell": worst_cell,
        "n_dropped_unknown_regime": n_dropped_unknown_regime,
        "hint": hint,
    }


def _load_outcomes(path: Path) -> list[dict]:
    """Robust JSONL load (mirrors ``persona_skill._load_outcomes``).

    Skips unparseable lines; a missing/corrupt file yields ``[]`` so
    the CLI degrades to ``INSUFFICIENT_DATA`` rather than crashing —
    the same best-effort discipline every sibling diagnostic uses.
    """
    rows: list[dict] = []
    try:
        if not path.exists():
            return rows
        with path.open("r") as fh:
            for ln in fh:
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


def _cli(argv: list[str] | None = None) -> int:
    """`python3 -m paper_trader.ml.persona_regime_skill` — per-cell
    (persona × regime) signal rank-IC over ``decision_outcomes.jsonl``.

    Read-only; never writes anything. Exit 0 on HEALTHY /
    INSUFFICIENT_DATA / NO_PERSONA_EDGE, 1 on REGIME_CONDITIONAL,
    2 if any cell is INVERTED (so an operator/cron can branch on it,
    exactly like persona_skill._cli / regime_audit._cli).
    """
    import argparse
    import sys

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.persona_regime_skill",
        description=(
            "Per-(persona × regime) decision-signal rank-IC cross-tab "
            "from decision_outcomes.jsonl. Read-only — never trains, "
            "never writes."
        ),
    )
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of a table.")
    p.add_argument("--outcomes", default=None,
                   help="Path to decision_outcomes.jsonl "
                        "(default: <repo>/data/decision_outcomes.jsonl).")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    root = Path(__file__).resolve().parent.parent.parent
    out_path = (Path(args.outcomes) if args.outcomes
                else root / "data" / "decision_outcomes.jsonl")
    recs = _load_outcomes(out_path)
    rep = analyze(recs)

    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
    else:
        print(f"aligned_outcomes={rep['n_records']}  "
              f"cells={rep['n_cells']}  "
              f"stable_cells={rep['n_stable_cells']}  "
              f"dropped_unknown_regime={rep['n_dropped_unknown_regime']}")
        print(f"VERDICT: {rep['verdict']}  ({rep['hint']})")
        cells = rep["cells"]
        if cells:
            print(f"  {'persona':<34} {'regime':<10} {'n':>5} "
                  f"{'score_ic':>9} {'mean_ret':>9} {'win%':>6}  verdict")
            for c in cells:
                print(f"  {c['persona']:<34} {c['regime']:<10} "
                      f"{c['n']:>5} {c['score_ic']:>+9.3f} "
                      f"{c['mean_aligned_return']:>+9.2f} "
                      f"{c['win_rate'] * 100:>5.0f}%  "
                      f"{c['verdict']}")
        if rep["best_cell"]:
            b = rep["best_cell"]
            print(f"  BEST stable cell : {b['persona']}/{b['regime']} "
                  f"score_ic={b['score_ic']:+.3f} n={b['n']}")
        if rep["worst_cell"]:
            w = rep["worst_cell"]
            print(f"  WORST stable cell: {w['persona']}/{w['regime']} "
                  f"score_ic={w['score_ic']:+.3f} n={w['n']}")

    if rep["verdict"] == "HAS_INVERTED_CELL":
        return 2
    if rep["verdict"] == "REGIME_CONDITIONAL":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())

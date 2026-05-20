"""Forward-return *horizon-consistency* diagnostic — read-only.

The `_compute_decision_outcomes` writer additively records
`forward_return_5d`, `forward_return_10d`, and `forward_return_20d` on every
outcome row (`run_continuous_backtests.py`). Every existing horizon tool
(`horizon_audit`, `_oos_multi_horizon_metrics`) probes a **scorer or
probe signal** against each horizon — i.e. "does the model predict 5d / 10d
/ 20d returns?". This module answers the prior, **label-side** question that
no existing diagnostic surfaces:

  *Do the three horizons themselves AGREE on direction, or does the 5d
   sign reverse by 20d often enough that the scorer is effectively trained
   on a noise target?*

If 5d → 20d sign-agreement is ≥75%, the 5d label is a faithful proxy for
multi-week direction — a model with edge at 5d is likely directionally
right at 20d too. If sign-agreement is <50%, the 5d return flips by 20d
more often than it persists; a 5d-trained scorer is then learning a target
that anti-correlates with the multi-week trajectory it ought to predict —
the strongest possible target-design red flag, complementary to
`horizon_audit`'s `LONGER_HORIZON_MORE_PREDICTABLE`. Mixed sign-agreement
(50–75%) is the "noisy but usable" middle.

Operational discipline mirrors every other `paper_trader/ml/*.py`
diagnostic (`calibration.py`, `horizon_audit.py`, `skill_trend.py`,
`outcome_drift.py`, …): **read-only** — no train, no `decision_scorer.pkl`
/ `build_features` / `N_FEATURES` / trade path touch, never raises on bad
input — safe to run against the live unattended continuous loop. Restricts
to the **temporal-OOS slice** by default (mirrors `horizon_audit.analyze`'s
default) so the verdict describes the EXACT holdout the gate's skill is
measured on, not the in-sample tail the model trained on.

The codebase-universal SELL sign-flip is applied: a SELL's realized
"goodness" at every horizon is `-forward_return_{h}d` so "high agreement"
means agreement on the trader's *intended direction*, not raw price drift
(mirrors `horizon_audit._aligned`, `train_scorer`'s SELL flip, and
`calibration.scorer_calibration`'s ordering).

```bash
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.outcome_horizon_consistency
cd /home/zeph/paper-trader && python3 -m pytest tests/test_outcome_horizon_consistency.py -v
```
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .decision_scorer import _to_float

# Module-level thresholds — same constants-at-module-scope rule the
# sibling diagnostics (`calibration.SPEARMAN_*`, `horizon_audit.EDGE_FLOOR`,
# `skill_trend.IC_MIN`) follow so a tuning change is a single reviewable
# edit AND tests can assert exact verdicts.
MIN_PAIRS = 30          # below this, no verdict is meaningful (mirrors
                        # calibration.MIN_PAIRS / horizon_audit.MIN_PAIRS so
                        # cross-diagnostic floors agree).
STRONG_AGREE = 0.75     # ≥75% sign-agreement is "persistent" — the 5d label
                        # is a faithful proxy for the longer horizon.
WEAK_AGREE = 0.50       # <50% means the 5d sign FLIPS more often than it
                        # persists — anti-predictive of the longer horizon.

# Horizons probed. 5d is the model's training target; 10d and 20d are the
# additively-captured longer windows. Module-level so a test can override
# (the `oos_bootstrap_ci.N_BOOTSTRAP` precedent).
ANCHOR_HORIZON = 5
LONGER_HORIZONS = (10, 20)


def _aligned_sign(value: float, is_sell: bool) -> int:
    """Action-aligned sign of a realized forward return. SELL flips the sign
    so 'positive' means 'this action was correct' (a price drop after a
    SELL is the right call). Mirrors `horizon_audit._aligned` and
    `train_scorer`'s SELL convention so this metric and every sibling
    diagnostic measure the same notion of 'goodness'.

    Zero values return 0 (no directional truth — the same exclusion every
    sibling rank-skill helper applies)."""
    v = -value if is_sell else value
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


def _build_aligned_signs(records: list[dict]) -> dict[int, list[int]]:
    """For each horizon in `(ANCHOR_HORIZON,) + LONGER_HORIZONS`, return the
    list of per-record action-aligned signs over rows that resolve **all
    three** horizons simultaneously (so the per-pair fraction denominators
    are identical and the verdict isn't gamed by an unevenly-sampled
    horizon). Pure / no IO / never raises."""
    horizons = (ANCHOR_HORIZON,) + tuple(LONGER_HORIZONS)
    signs: dict[int, list[int]] = {h: [] for h in horizons}
    try:
        for r in records:
            # Every horizon must resolve to a finite real number for this
            # row to contribute — partial-horizon rows would skew the
            # comparison denominators if counted asymmetrically.
            resolved: dict[int, float] = {}
            ok = True
            for h in horizons:
                raw = r.get(f"forward_return_{h}d")
                if raw is None:
                    ok = False
                    break
                v = _to_float(raw, float("nan"))
                if v != v:  # NaN check — _to_float's documented sentinel
                    ok = False
                    break
                resolved[h] = float(v)
            if not ok:
                continue
            is_sell = str(r.get("action") or "BUY").upper() == "SELL"
            for h in horizons:
                signs[h].append(_aligned_sign(resolved[h], is_sell))
    except Exception:
        # Defensive — every sibling diagnostic degrades to empty on any fault.
        return {h: [] for h in horizons}
    return signs


def _agreement_rate(a: list[int], b: list[int]) -> float | None:
    """Fraction of (a[i], b[i]) pairs where `a[i] == b[i]` AND BOTH are
    non-zero. A zero on either side carries no directional truth and is
    excluded (mirrors the `dir_acc` convention in `_oos_rank_metrics` —
    cross-diagnostic consistency). None when no non-zero pairs exist."""
    if len(a) != len(b):
        return None
    n = 0
    hits = 0
    for x, y in zip(a, b):
        if x == 0 or y == 0:
            continue
        n += 1
        if x == y:
            hits += 1
    if n == 0:
        return None
    return hits / n


def _reversal_rate(anchor: list[int], longer: list[int]) -> float | None:
    """Fraction of strictly-opposite-sign pairs `(anchor[i] * longer[i] < 0)`
    over non-zero pairs. Distinct from `1 - agreement`: a zero on either
    side is excluded, not counted as "neither agree nor reverse". None when
    no non-zero pairs exist."""
    if len(anchor) != len(longer):
        return None
    n = 0
    flips = 0
    for x, y in zip(anchor, longer):
        if x == 0 or y == 0:
            continue
        n += 1
        if x * y < 0:
            flips += 1
    if n == 0:
        return None
    return flips / n


def consistency_report(records: list[dict]) -> dict:
    """Per-(anchor → longer) agreement + reversal rates plus a crisp verdict.

    `records` are decision_outcomes rows already restricted to the slice
    the caller wants (`analyze` passes the temporal-OOS slice). Never
    raises; any fault degrades to `INSUFFICIENT_DATA` (the
    `horizon_audit_report` precedent).

    Returns a JSON-safe dict:
      {status, verdict, n_complete, cells: [{anchor, longer, n,
       agreement_rate, reversal_rate, mean_anchor_when_agree,
       mean_anchor_when_reverse}], min_agreement, hint}
    """
    base: dict = {
        "status": "ok",
        "verdict": "INSUFFICIENT_DATA",
        "n_complete": 0,
        "cells": [],
        "min_agreement": None,
        "hint": "",
    }
    try:
        recs = list(records or [])
        signs = _build_aligned_signs(recs)
        anchor_signs = signs[ANCHOR_HORIZON]
        base["n_complete"] = len(anchor_signs)

        if len(anchor_signs) < MIN_PAIRS:
            base["hint"] = (
                f"need ≥{MIN_PAIRS} rows resolving 5d AND every longer "
                f"horizon; have {len(anchor_signs)}"
            )
            return base

        # For the per-bucket magnitude summaries we need the raw aligned
        # 5d values too, not just signs. Re-extract once (cheap) so the
        # mean-when-agree / mean-when-reverse cells stay honest.
        aligned_5d: list[float] = []
        for r in recs:
            try:
                all_ok = all(
                    r.get(f"forward_return_{h}d") is not None
                    for h in (ANCHOR_HORIZON,) + tuple(LONGER_HORIZONS)
                )
                if not all_ok:
                    continue
                v = _to_float(r.get(f"forward_return_{ANCHOR_HORIZON}d"),
                              float("nan"))
                if v != v:
                    continue
                # Skip rows that fail any horizon's finite check (mirror
                # _build_aligned_signs's all-or-nothing inclusion).
                bad = False
                for h in (ANCHOR_HORIZON,) + tuple(LONGER_HORIZONS):
                    hv = _to_float(r.get(f"forward_return_{h}d"),
                                   float("nan"))
                    if hv != hv:
                        bad = True
                        break
                if bad:
                    continue
                is_sell = str(r.get("action") or "BUY").upper() == "SELL"
                aligned_5d.append(-v if is_sell else v)
            except Exception:
                continue

        cells: list[dict] = []
        min_agree: float | None = None
        for h in LONGER_HORIZONS:
            ar = _agreement_rate(anchor_signs, signs[h])
            rr = _reversal_rate(anchor_signs, signs[h])
            # Magnitude summary: |aligned 5d| on rows where anchor agrees
            # vs reverses with this horizon. A common quant question is
            # "are the persistent winners BIGGER than the reversers?"
            agree_mags: list[float] = []
            reverse_mags: list[float] = []
            for s5, sh, v5 in zip(anchor_signs, signs[h], aligned_5d):
                if s5 == 0 or sh == 0:
                    continue
                m = abs(v5)
                if s5 == sh:
                    agree_mags.append(m)
                elif s5 * sh < 0:
                    reverse_mags.append(m)
            cell = {
                "anchor": ANCHOR_HORIZON,
                "longer": h,
                "n": len([
                    1 for a, b in zip(anchor_signs, signs[h])
                    if a != 0 and b != 0
                ]),
                "agreement_rate": round(ar, 4) if ar is not None else None,
                "reversal_rate": round(rr, 4) if rr is not None else None,
                "mean_abs_5d_when_agree": (
                    round(sum(agree_mags) / len(agree_mags), 4)
                    if agree_mags else None
                ),
                "mean_abs_5d_when_reverse": (
                    round(sum(reverse_mags) / len(reverse_mags), 4)
                    if reverse_mags else None
                ),
            }
            cells.append(cell)
            if ar is not None and (min_agree is None or ar < min_agree):
                min_agree = ar
        base["cells"] = cells
        base["min_agreement"] = (
            round(min_agree, 4) if min_agree is not None else None
        )

        if min_agree is None:
            base["hint"] = "no non-zero anchor↔longer pairs"
            return base

        if min_agree >= STRONG_AGREE:
            base["verdict"] = "STRONG_PERSISTENCE"
            base["hint"] = (
                f"5d → longer-horizon sign-agreement ≥{STRONG_AGREE:.0%} "
                f"(min {min_agree:.0%}) — the 5d label is a faithful "
                f"proxy for multi-week direction; a scorer with 5d edge "
                f"is likely directionally right at 10d/20d too"
            )
        elif min_agree >= WEAK_AGREE:
            base["verdict"] = "MIXED_PERSISTENCE"
            base["hint"] = (
                f"5d → longer-horizon sign-agreement {min_agree:.0%} ∈ "
                f"[{WEAK_AGREE:.0%}, {STRONG_AGREE:.0%}) — the 5d label "
                f"persists more often than it flips, but the target is "
                f"materially noisy; expect rank-IC to compress at longer "
                f"horizons"
            )
        else:
            base["verdict"] = "HIGH_REVERSAL"
            base["hint"] = (
                f"5d → longer-horizon sign-agreement {min_agree:.0%} < "
                f"{WEAK_AGREE:.0%} — the 5d sign FLIPS more often than it "
                f"persists; a 5d-trained scorer is learning a target "
                f"that anti-correlates with multi-week direction "
                f"(complements horizon_audit's LONGER_HORIZON_MORE_"
                f"PREDICTABLE — target-design red flag)"
            )
        return base
    except Exception as e:  # pragma: no cover - defensive
        base["verdict"] = "INSUFFICIENT_DATA"
        base["hint"] = f"audit failed: {type(e).__name__}: {e}"
        return base


def analyze(outcomes_path: Path | str, oos_only: bool = True) -> dict:
    """Load the outcomes file, take the temporal-OOS slice (default) and
    run the horizon-consistency audit. Read-only; never raises.

    Mirrors `horizon_audit.analyze`'s exact contract: `oos_only=True`
    restricts to the same temporal-OOS slice `_train_decision_scorer` /
    `oos_bootstrap_ci` use, so this verdict describes the EXACT holdout
    the gate's skill is measured on. `oos_only=False` runs over every
    outcome (useful for ad-hoc inspection of the full corpus)."""
    out: dict = {"status": "ok", "verdict": "INSUFFICIENT_DATA",
                 "n_complete": 0, "cells": [], "slice": "all",
                 "n_records_total": 0, "hint": ""}
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

        rep = consistency_report(recs)
        rep["slice"] = slice_name
        rep["n_records_total"] = len(records)
        return rep
    except Exception as e:  # pragma: no cover - defensive
        out["hint"] = f"audit failed: {type(e).__name__}: {e}"
        return out


def _cli(argv: list[str] | None = None) -> int:
    """`python3 -m paper_trader.ml.outcome_horizon_consistency [--all]`
    — read-only forward-return horizon-consistency audit of the accumulated
    outcomes. Exit code mirrors siblings: 0 on success, 2 on
    INSUFFICIENT_DATA so cron / monitoring can branch."""
    argv = sys.argv[1:] if argv is None else argv
    oos_only = "--all" not in argv
    root = Path(__file__).resolve().parent.parent.parent
    rep = analyze(root / "data" / "decision_outcomes.jsonl", oos_only=oos_only)
    print(f"VERDICT: {rep['verdict']}  ({rep.get('hint', '')})")
    print(f"  slice={rep.get('slice')}  n_complete={rep.get('n_complete')}  "
          f"n_total={rep.get('n_records_total')}  "
          f"min_agreement={rep.get('min_agreement')}")
    print(f"  {'anchor':>6} {'longer':>6} {'n':>6} {'agree':>8} "
          f"{'reverse':>8} {'|5d|agree':>10} {'|5d|rev':>10}")
    for c in rep.get("cells", []):
        ar = c.get("agreement_rate")
        rr = c.get("reversal_rate")
        ma = c.get("mean_abs_5d_when_agree")
        mr = c.get("mean_abs_5d_when_reverse")
        print(f"  {c['anchor']:>5}d {c['longer']:>5}d {c['n']:>6} "
              f"{(f'{ar:.4f}' if ar is not None else 'n/a'):>8} "
              f"{(f'{rr:.4f}' if rr is not None else 'n/a'):>8} "
              f"{(f'{ma:.4f}' if ma is not None else 'n/a'):>10} "
              f"{(f'{mr:.4f}' if mr is not None else 'n/a'):>10}")
    return 0 if rep.get("verdict") != "INSUFFICIENT_DATA" else 2


if __name__ == "__main__":
    raise SystemExit(_cli())

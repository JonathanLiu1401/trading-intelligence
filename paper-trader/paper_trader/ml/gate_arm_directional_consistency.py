"""Per-gate-arm directional consistency — sign-only realized skill.

Read-only research signal (2026-05-30 feature, Agent 2 ML+backtests
HYBRID pass #5). Sibling to ``gate_realized`` (per-arm realized MEAN)
and ``gate_arm_kelly`` (per-arm risk-adjusted Sharpe) — mirrors their
operational discipline exactly (never trains, never touches
``decision_scorer.pkl`` / ``decision_outcomes.jsonl`` / ``build_features``
/ ``N_FEATURES`` / any trade path; safe to run against the live
unattended loop; never raises in analyzer functions).

**The gap this closes.** ``gate_realized`` reports each arm's realized
MEAN return; the verdict spread is ``strong_tailwind.mean -
strong_headwind.mean``. A mean is fragile to a few extreme outcomes —
e.g. a strong_tailwind arm with 40% positive realized but two outsized
+80% winners can still post a positive mean while being SIGN-wrong on
60% of the trades. The 5-day-window leveraged-ETF tape (SOXL/TQQQ)
documented in AGENTS.md is precisely this distribution shape: noisy
weekly outcomes with rare big positive weeks, where the mean reading
overstates the per-trade directional edge.

This analyzer asks the sign-only complement of ``gate_realized``'s
mean read: *for each gate arm, what fraction of acted-on realizations
matched the arm's expected direction?*

  * ``strong_tailwind`` / ``mild_tailwind`` — expected direction is
    POSITIVE (gate sized UP because the scorer said "good outcome").
    Directional consistency = ``n_positive / n_total``.
  * ``strong_headwind`` / ``mild_headwind`` — expected direction is
    NEGATIVE (gate sized DOWN because the scorer said "bad outcome").
    Directional consistency = ``n_negative / n_total``.
  * ``neutral`` — no expected direction (the gate's no-op arm). The
    informational ``balanced_fraction = max(n_positive, n_negative) /
    n_total`` is reported — useful as a "did the neutral arm see a
    coin-flip distribution?" sanity check — but explicitly **NOT**
    folded into the verdict, the same honesty pattern
    ``gate_audit.arm_monotone_fraction`` follows.

The verdict is driven by the two EXTREME arms (``strong_tailwind`` and
``strong_headwind``), exactly like ``gate_realized``'s spread —
because the gate's biggest sizing reallocation lives there.

**Metrics reported per arm**:

  * ``n`` — acted captured rows in this arm (off-distribution excluded;
    see the ``gate_realized`` ``abstained`` discipline).
  * ``n_positive`` — count of realized > 0 (SELL-sign-flipped, the
    universal codebase convention).
  * ``n_negative`` — count of realized < 0.
  * ``n_zero`` — count of realized exactly 0 (excluded from consistency).
  * ``expected_sign`` — +1 / -1 / 0 (tailwind / headwind / neutral).
  * ``directional_consistency`` — fraction of NON-zero realized whose
    sign matches ``expected_sign``. ``None`` when ``expected_sign==0``
    (neutral arm) or all realized are zero.
  * ``balanced_fraction`` — ``max(n_positive, n_negative) /
    n_nonzero``. Reported for ALL arms — for non-neutral arms this is
    the same as directional_consistency when the gate is consistent
    and ``1 - directional_consistency`` when inverted, so a reader can
    spot inversion at a glance. For the neutral arm this is the only
    sign-only summary available.

**Verdict ladder** (extreme-arm directional consistency, threshold-driven
so the verdict is exactly testable):

| Verdict | Meaning |
|---------|---------|
| ``GATE_CAPTURE_NOT_YET_POPULATED`` | 0 captured rows — same condition ``gate_realized`` reports. |
| ``INSUFFICIENT_DATA``           | < ``MIN_TOTAL`` total acted rows, or either extreme arm < ``MIN_ARM_N``. |
| ``GATE_DIRECTIONALLY_INVERTED`` | ≥1 extreme arm has directional_consistency < ``INVERTED_TOL`` (the gate's biggest bet is sign-WRONG more often than right). |
| ``GATE_DIRECTIONALLY_NOISE``    | both extreme arms have directional_consistency in [``INVERTED_TOL``, ``CONSISTENT_TOL``] — coin-flip on both. |
| ``GATE_DIRECTIONALLY_CONSISTENT`` | both extreme arms have directional_consistency ≥ ``CONSISTENT_TOL``. |
| ``GATE_DIRECTIONALLY_MIXED``    | one extreme arm ≥ ``CONSISTENT_TOL``, the other in [``INVERTED_TOL``, ``CONSISTENT_TOL``] — different arms tell different stories (the gate is asymmetric in its directional edge). |

CLI:

```bash
cd /home/zeph/trading-intelligence/paper-trader && \
    python3 -m paper_trader.ml.gate_arm_directional_consistency
cd /home/zeph/trading-intelligence/paper-trader && \
    python3 -m paper_trader.ml.gate_arm_directional_consistency --json
```

Exit code 0 on ``GATE_DIRECTIONALLY_CONSISTENT``, ``INSUFFICIENT_DATA``,
``GATE_CAPTURE_NOT_YET_POPULATED``; 2 on
``GATE_DIRECTIONALLY_INVERTED`` / ``GATE_DIRECTIONALLY_NOISE`` /
``GATE_DIRECTIONALLY_MIXED`` (operator-actionable states).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Default outcomes path — module-level so tests can monkeypatch and CLI
# can be invoked without args. Same convention every sibling analyzer
# in this folder follows (``calibration.py``, ``outcome_corpus_health.py``).
DEFAULT_OUTCOMES_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "decision_outcomes.jsonl"
)

# Verdict thresholds. Module-level (single point of tuning) so tests can
# assert exact verdicts.
MIN_TOTAL = 30        # need ≥30 acted captured rows total
MIN_ARM_N = 5         # need ≥5 acted rows in EACH extreme arm
INVERTED_TOL = 0.45   # consistency strictly below ⇒ INVERTED
CONSISTENT_TOL = 0.55 # consistency at or above ⇒ CONSISTENT
                      # gap in [0.45, 0.55] reads as "noise / coin-flip"

# Arm → expected realized sign. Mirrors the live gate's conviction
# multiplier semantics (CLAUDE.md §6, ``backtest.py::_ml_decide``):
# tailwind arms size UP because the scorer expected a positive outcome.
# Single source of truth — tests can re-import this mapping rather than
# duplicating the table.
_EXPECTED_SIGN: dict[str, int] = {
    "strong_headwind": -1,
    "mild_headwind": -1,
    "neutral": 0,
    "mild_tailwind": +1,
    "strong_tailwind": +1,
}

# Display order — same as ``gate_realized._ARM_ORDER`` and the
# ``_ml_decide`` if/elif chain so a reader sees headwind → tailwind
# left-to-right.
_ARM_ORDER: tuple[str, ...] = (
    "strong_headwind", "mild_headwind", "neutral",
    "mild_tailwind", "strong_tailwind",
)


def _safe_float(v: Any) -> float | None:
    """``float(v)`` that returns None for None / non-numeric / NaN /
    inf. Mirrors ``gate_realized._f`` (no shared import to avoid
    coupling — this analyzer is intentionally a peer)."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    if f in (float("inf"), float("-inf")):
        return None
    return f


def _arm_for_pred(pred: float) -> str:
    """Decode the gate arm for a captured raw prediction. Mirrors
    ``gate_audit.gate_arm`` exactly — duplicated locally to avoid
    pulling in gate_audit's sklearn-adjacent imports for what is a
    7-line if/elif. The bucket boundaries are pinned by
    ``test_gate_arm_directional_consistency_matches_gate_audit`` (CLAUDE.md
    "no-drift" SSOT discipline).

    Non-finite (NaN / ±inf) falls through to ``"neutral"`` — same
    contract gate_audit.gate_arm advertises. ``_safe_float`` already
    filters these before this function is reached on the live path, but
    the SSOT guard keeps the two helpers byte-identical on every input.
    """
    try:
        p = float(pred)
    except (TypeError, ValueError):
        return "neutral"
    if p != p or p in (float("inf"), float("-inf")):
        return "neutral"
    if p < -10.0:
        return "strong_headwind"
    if p < 0.0:
        return "mild_headwind"
    if p > 10.0:
        return "strong_tailwind"
    if p > 5.0:
        return "mild_tailwind"
    return "neutral"


def gate_arm_directional_consistency_report(rows) -> dict:
    """Bucket outcome rows by their captured gate arm and report
    per-arm sign-only directional consistency.

    ``rows`` — any iterable of ``decision_outcomes.jsonl``-shaped dicts.
    For each row this reads, and never re-predicts:

      * ``gate_scorer_pred`` — gate's true decision-time prediction.
        None / non-finite ⇒ row excluded entirely.
      * ``gate_off_dist`` — True ⇒ gate abstained (no multiplier
        applied). Routed to ``abstained_n``, NOT a per-arm bucket
        (mirrors ``gate_realized``).
      * ``forward_return_5d`` — realized 5d return, SELL-sign-flipped
        for SELL rows (``train_scorer`` / ``gate_audit`` /
        ``gate_realized`` convention).

    Returns a JSON-safe dict. Never raises.
    """
    base: dict[str, Any] = {
        "status": "ok",
        "verdict": "GATE_CAPTURE_NOT_YET_POPULATED",
        "measurement": "captured_then_deployed_no_reprediction_sign_only",
        "n_captured": 0,
        "n_acted": 0,
        "n_abstained": 0,
        "min_total": MIN_TOTAL,
        "min_arm_n": MIN_ARM_N,
        "inverted_tol": INVERTED_TOL,
        "consistent_tol": CONSISTENT_TOL,
        "arms": [],
        "hint": "",
    }

    try:
        it = list(rows or [])
    except Exception:
        it = []

    # Per-arm sign counters: arm → (n_pos, n_neg, n_zero).
    counts: dict[str, list[int]] = {a: [0, 0, 0] for a in _ARM_ORDER}
    n_captured = 0
    n_acted = 0
    n_abstained = 0

    for r in it:
        if not isinstance(r, dict):
            continue
        gp = _safe_float(r.get("gate_scorer_pred"))
        if gp is None:
            continue  # no gate decision on this row
        n_captured += 1

        # SELL rows: flip the realized sign so "good" always reads positive
        # (universal codebase convention — train_scorer / gate_audit /
        # gate_realized all do this). gate_scorer_pred is BUY-only by
        # construction, but the flip is defensive consistency.
        is_sell = str(r.get("action") or "BUY").upper() == "SELL"
        fr = _safe_float(r.get("forward_return_5d"))
        if fr is None:
            continue  # no usable realized → skip
        realized = -fr if is_sell else fr

        if bool(r.get("gate_off_dist")):
            n_abstained += 1
            continue  # abstained — never bucketed into an arm

        n_acted += 1
        arm = _arm_for_pred(gp)
        if realized > 0:
            counts[arm][0] += 1
        elif realized < 0:
            counts[arm][1] += 1
        else:
            counts[arm][2] += 1

    arms_out: list[dict] = []
    for a in _ARM_ORDER:
        n_pos, n_neg, n_zero = counts[a]
        n_total = n_pos + n_neg + n_zero
        n_nonzero = n_pos + n_neg
        expected = _EXPECTED_SIGN[a]
        if n_nonzero == 0:
            consistency: float | None = None
            balanced: float | None = None
        else:
            balanced = round(max(n_pos, n_neg) / n_nonzero, 4)
            if expected == +1:
                consistency = round(n_pos / n_nonzero, 4)
            elif expected == -1:
                consistency = round(n_neg / n_nonzero, 4)
            else:
                # Neutral arm — no expected direction by construction.
                # `balanced` is still meaningful and reported above.
                consistency = None
        arms_out.append({
            "arm": a,
            "expected_sign": expected,
            "n": n_total,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_zero": n_zero,
            "directional_consistency": consistency,
            "balanced_fraction": balanced,
        })

    base["n_captured"] = n_captured
    base["n_acted"] = n_acted
    base["n_abstained"] = n_abstained
    base["arms"] = arms_out

    if n_captured == 0:
        base["hint"] = (
            "no row carries a non-null gate_scorer_pred — the continuous "
            "loop predates commit 60b20d9 (gate-decision capture) or has "
            "not accumulated captured rows since; populates on redeploy"
        )
        return base

    # Need MIN_TOTAL total acted rows AND MIN_ARM_N in EACH extreme arm
    # before any consistency verdict is meaningful.
    by_arm = {row["arm"]: row for row in arms_out}
    sh = by_arm["strong_headwind"]
    st = by_arm["strong_tailwind"]
    if (n_acted < MIN_TOTAL
            or sh["n"] < MIN_ARM_N
            or st["n"] < MIN_ARM_N):
        base["verdict"] = "INSUFFICIENT_DATA"
        base["hint"] = (
            f"need ≥{MIN_TOTAL} total acted rows AND ≥{MIN_ARM_N} in each "
            f"extreme arm — have n_acted={n_acted}, "
            f"strong_headwind.n={sh['n']}, strong_tailwind.n={st['n']}"
        )
        return base

    sh_c = sh["directional_consistency"]
    st_c = st["directional_consistency"]
    # Both must be defined to get past INSUFFICIENT_DATA (we just
    # asserted MIN_ARM_N≥5 in each, so n_nonzero may still be 0 if every
    # one of those 5 came back exactly zero — astronomically unlikely
    # but check anyway, the "never raises, degrade honestly" discipline).
    if sh_c is None or st_c is None:
        base["verdict"] = "INSUFFICIENT_DATA"
        base["hint"] = (
            "at least one extreme arm has 0 non-zero realizations — "
            "cannot compute directional consistency"
        )
        return base

    sh_inv = sh_c < INVERTED_TOL
    st_inv = st_c < INVERTED_TOL
    sh_good = sh_c >= CONSISTENT_TOL
    st_good = st_c >= CONSISTENT_TOL

    if sh_inv or st_inv:
        # Inversion on EITHER extreme arm is the most severe verdict.
        # An inverted strong arm means the gate's biggest bet is more
        # often sign-WRONG than right — the gate is actively anti-skilled
        # on that arm.
        base["verdict"] = "GATE_DIRECTIONALLY_INVERTED"
        which = []
        if sh_inv:
            which.append(f"strong_headwind={sh_c:.3f}")
        if st_inv:
            which.append(f"strong_tailwind={st_c:.3f}")
        base["hint"] = (
            f"directional_consistency below {INVERTED_TOL} on: "
            f"{', '.join(which)} — gate's biggest bet is sign-WRONG more "
            f"often than right; turning the gate off would have improved "
            f"the realized SIGN match on these arms"
        )
        return base

    if sh_good and st_good:
        base["verdict"] = "GATE_DIRECTIONALLY_CONSISTENT"
        base["hint"] = (
            f"both extreme arms ≥ {CONSISTENT_TOL}: strong_headwind={sh_c:.3f} "
            f"strong_tailwind={st_c:.3f} — the gate's directional intent "
            f"matched realized on both extreme bets at a rate above noise"
        )
        return base

    if not sh_good and not st_good:
        base["verdict"] = "GATE_DIRECTIONALLY_NOISE"
        base["hint"] = (
            f"both extreme arms in [{INVERTED_TOL}, {CONSISTENT_TOL}]: "
            f"strong_headwind={sh_c:.3f} strong_tailwind={st_c:.3f} — "
            f"coin-flip on both; the gate's directional intent is "
            f"indistinguishable from a fair coin on its biggest bets"
        )
        return base

    # Exactly one extreme arm clears CONSISTENT_TOL, the other in
    # noise range (inversion already returned above).
    base["verdict"] = "GATE_DIRECTIONALLY_MIXED"
    good = "strong_tailwind" if st_good else "strong_headwind"
    weak = "strong_headwind" if st_good else "strong_tailwind"
    good_c = st_c if st_good else sh_c
    weak_c = sh_c if st_good else st_c
    base["hint"] = (
        f"asymmetric: {good}={good_c:.3f} clears {CONSISTENT_TOL} but "
        f"{weak}={weak_c:.3f} only at noise — the gate's directional edge "
        f"lives on one side but not the other"
    )
    return base


def analyze(outcomes_path: Path | str | None = None) -> dict:
    """End-to-end: load outcomes JSONL → bucket → verdict. Pure, total,
    never raises. ``outcomes_path=None`` uses ``DEFAULT_OUTCOMES_PATH``.
    Mirrors every sibling analyzer's CLI entry shape."""
    p = Path(outcomes_path) if outcomes_path is not None else DEFAULT_OUTCOMES_PATH
    rows: list[dict] = []
    try:
        if p.exists():
            with p.open("r") as fh:
                for ln in fh:
                    s = ln.strip()
                    if not s:
                        continue
                    try:
                        rows.append(json.loads(s))
                    except Exception:
                        continue
    except Exception as exc:
        return {
            "status": "error",
            "verdict": "GATE_CAPTURE_NOT_YET_POPULATED",
            "error": f"read failed: {exc}",
            "n_captured": 0,
            "n_acted": 0,
            "n_abstained": 0,
            "arms": [],
            "hint": f"could not read {p}",
        }
    return gate_arm_directional_consistency_report(rows)


def _cli() -> int:
    import argparse
    import sys

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.gate_arm_directional_consistency",
        description="Per-arm directional consistency (sign-only) of the "
                    "captured gate decisions. Read-only.",
    )
    p.add_argument("--path", default=None,
                   help="Override outcomes JSONL path.")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON.")
    args = p.parse_args(sys.argv[1:])

    rep = analyze(args.path)
    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
    else:
        print(f"[gate_arm_directional_consistency] verdict={rep['verdict']}  "
              f"n_acted={rep.get('n_acted', 0)}  "
              f"n_abstained={rep.get('n_abstained', 0)}")
        for a in rep.get("arms") or []:
            dc = a.get("directional_consistency")
            bal = a.get("balanced_fraction")
            dc_s = f"{dc:.3f}" if dc is not None else "  n/a"
            bal_s = f"{bal:.3f}" if bal is not None else "  n/a"
            print(f"  {a['arm']:<18} n={a['n']:>5}  "
                  f"+={a['n_positive']:>4} -={a['n_negative']:>4}  "
                  f"directional_consistency={dc_s}  "
                  f"balanced={bal_s}")
        if rep.get("hint"):
            print(f"  {rep['hint']}")

    if rep["verdict"] in ("GATE_DIRECTIONALLY_INVERTED",
                          "GATE_DIRECTIONALLY_NOISE",
                          "GATE_DIRECTIONALLY_MIXED"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())

"""Conviction-gate REALIZED risk profile — win rate, drawdown percentiles,
and Sharpe per arm from the captured then-deployed decision.

Read-only diagnostic. Same operational discipline as ``gate_realized.py`` /
``gate_audit.py`` / ``calibration.py``: never trains, never touches
``decision_scorer.pkl``, ``decision_outcomes.jsonl``, ``build_features`` or
any trade path. Safe to run against the live unattended continuous loop.

**The gap this fills.** ``gate_realized`` already buckets realized 5d
returns by the gate's captured arm and grades the spread ``strong_tailwind
− strong_headwind`` as ``GATE_EFFECTIVE`` / ``GATE_INEFFECTIVE`` /
``GATE_HARMFUL``. That spread is a *mean-of-returns* measure. A skeptical
quant cannot size off mean alone — two arms can post the same mean while
one wins 70% of the time at small magnitudes and the other wins 30% of
the time at large ones, asymmetries the bet-sizing rule must account for.

The deployed gate multiplies position size ×0.60 → ×1.30 across the five
arms. A 1.30× sleeve into a strong_tailwind arm whose ``p10`` (worst
10% of realized 5d returns) is −12% creates ~−16% drawdown exposure on
just the *worst-decile* of those trades — whether or not the *mean* is
positive. Mean-only diagnostics structurally cannot surface that. This
module does, by reporting per acted arm:

  * ``n``                    — sample count (== gate_realized)
  * ``win_rate``             — fraction with realized > 0
  * ``mean``, ``median``     — central tendency (mean == gate_realized)
  * ``p10`` / ``p25`` / ``p75`` / ``p90`` — realized-return percentiles
  * ``mean_win`` / ``mean_loss`` — magnitude conditional on direction
  * ``stdev``                — realized-return dispersion
  * ``sharpe``               — ``mean / stdev`` (excess-return version
    matches the convention in ``persona_leaderboard.py``)
  * ``expected_value``       — ``win_rate * mean_win + (1 − win_rate) * mean_loss``
    (a sanity reconciliation: must equal ``mean`` within rounding by
    construction; any deviation flags a metric-pipeline bug)

The verdict grades **risk-adjusted** behavior — does the gate's ×1.30
arm carry a higher Sharpe than its ×0.60 arm, or does it just trade
more variance for the same mean?

| Verdict | Trigger |
|---------|---------|
| ``GATE_CAPTURE_NOT_YET_POPULATED`` | 0 rows carry a non-null ``gate_scorer_pred`` (same deploy-stale state the sibling reports) |
| ``INSUFFICIENT_DATA`` | < ``MIN_TOTAL`` acted rows or < ``MIN_ARM_N`` in BOTH extreme arms |
| ``WIN_RATE_INVERTED`` | ``strong_tailwind.win_rate < strong_headwind.win_rate − WIN_RATE_TOL`` — the ×1.30 arm is right LESS often than the ×0.60 arm; even if mean is fine, the gate is sizing UP the wrong-direction-more-often bets |
| ``SHARPE_UNDEFINED_ARM`` | at least one extreme arm has zero return variance (stdev=0 ⇒ sharpe undefined) so risk-adjusted comparison is structurally not computable |
| ``UNFAVORABLE_RISK`` | ``strong_tailwind.sharpe − strong_headwind.sharpe < −SHARPE_TOL`` — bigger bets carry WORSE risk-adjusted reward (a 1.30× multiplier into an arm whose volatility-scaled return is lower than the 0.60× arm's) |
| ``RISK_INDIFFERENT`` | ``|sharpe spread| ≤ SHARPE_TOL`` — neither arm's risk-adjusted return justifies the multiplier ratio (the gate is reallocating variance, not edge) |
| ``RISK_PROPORTIONAL`` | ``strong_tailwind.sharpe − strong_headwind.sharpe > +SHARPE_TOL`` AND ``strong_tailwind.win_rate ≥ strong_headwind.win_rate − WIN_RATE_TOL`` — the gate's bigger bets carry better risk-adjusted reward without sacrificing hit-rate |

Verdict precedence is explicit (``WIN_RATE_INVERTED`` > ``SHARPE_UNDEFINED_ARM``
> ``UNFAVORABLE_RISK`` > ``RISK_INDIFFERENT`` > ``RISK_PROPORTIONAL``) so a
single row's metrics map to one and only one verdict — exact-value testable.

The ``abstained`` bucket (``gate_off_dist=True``) is reported for
context but never folded into the verdict, mirroring ``gate_realized``'s
abstention-honesty pattern: a row the gate did not act on cannot grade
the gate's sizing rule.

CLI: ``python3 -m paper_trader.ml.gate_risk_profile [--all]``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Single source of truth for the five gate arms / multipliers — importing
# (not re-declaring) guarantees this diagnostic and gate_audit /
# gate_realized / gate_pnl can never disagree about what the live
# ``_ml_decide`` gate does. Same SSOT discipline ``gate_realized.py`` uses.
from .gate_audit import gate_arm

# Verdict thresholds — module-level so tests assert exact verdicts.
MIN_TOTAL = 30          # need a real acted sample (== gate_audit/gate_realized)
MIN_ARM_N = 5           # min trades in EACH extreme arm to compare them
SHARPE_TOL = 0.10       # |sharpe spread| ≤ this reads as risk-indifferent
WIN_RATE_TOL = 0.05     # |win-rate spread| > this is a real difference (5pp)

# Anchor horizon — the gate's training target and the verdict driver
# (mirrors every sibling gate diagnostic: 5d is the gate's truth).
ANCHOR_H = 5

_ARM_ORDER = ["strong_headwind", "mild_headwind", "neutral",
              "mild_tailwind", "strong_tailwind"]


def _f(v):
    """Finite float or None — the `_to_float` hardening class, local so
    this module imports nothing from decision_scorer (read-only, no pickle
    path). Mirrors ``gate_realized._f`` byte-for-byte."""
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _arm_stats(vals: list[float]) -> dict:
    """Per-arm full statistics dict. Empty input degrades to all-None
    so the dashboard renderer never crashes on a sparse arm.

    ``stdev`` uses ``ddof=0`` (population stdev) because the per-arm
    sample is the full universe of arm-acted decisions — there is no
    larger superpopulation to infer about. Sharpe is ``mean / stdev``;
    when ``stdev == 0`` (a single sample or a perfectly constant arm)
    we degrade to None rather than divide by zero. ``expected_value``
    is the win-rate-weighted reconciliation of mean_win + mean_loss —
    must match ``mean`` within rounding; surfaced so a divergent value
    flags a pipeline bug rather than silently rendering."""
    if not vals:
        return {
            "n": 0, "win_rate": None, "mean": None, "median": None,
            "p10": None, "p25": None, "p75": None, "p90": None,
            "mean_win": None, "mean_loss": None,
            "stdev": None, "sharpe": None, "expected_value": None,
        }
    a = np.asarray(vals, dtype=np.float64)
    n = int(a.size)
    wins = a[a > 0.0]
    losses = a[a < 0.0]
    # win_rate counts strictly-positive realized returns. A literal 0.0
    # outcome (the rare day-of-zero-movement, or a walk-back collision
    # the outcome computer let through) is neither a win nor a loss for
    # this purpose — same convention `persona_leaderboard.win_rate` uses.
    win_rate = float(wins.size) / float(n) if n else 0.0
    stdev = float(a.std(ddof=0)) if n >= 2 else 0.0
    mean = float(a.mean())
    mean_win = float(wins.mean()) if wins.size else None
    mean_loss = float(losses.mean()) if losses.size else None
    # Expected-value reconciliation: weighted blend of conditional means.
    # When wins or losses are absent the missing leg contributes 0 (the
    # `mean_win is None` legs land in the unaccounted zero-return slice).
    n_zero = int((a == 0.0).sum())
    ev_win = (wins.size / n) * (mean_win or 0.0)
    ev_loss = (losses.size / n) * (mean_loss or 0.0)
    expected_value = ev_win + ev_loss  # zero-return rows contribute 0
    sharpe = mean / stdev if stdev > 0 else None
    return {
        "n": n,
        "win_rate": round(win_rate, 4),
        "mean": round(mean, 4),
        "median": round(float(np.median(a)), 4),
        "p10": round(float(np.percentile(a, 10)), 4),
        "p25": round(float(np.percentile(a, 25)), 4),
        "p75": round(float(np.percentile(a, 75)), 4),
        "p90": round(float(np.percentile(a, 90)), 4),
        "mean_win": round(mean_win, 4) if mean_win is not None else None,
        "mean_loss": round(mean_loss, 4) if mean_loss is not None else None,
        "stdev": round(stdev, 4),
        "sharpe": round(sharpe, 4) if sharpe is not None else None,
        "expected_value": round(expected_value, 4),
        "n_zero": n_zero,
    }


def gate_risk_report(rows) -> dict:
    """Bucket outcome rows by captured gate arm and report risk metrics.

    ``rows`` — any iterable of ``decision_outcomes.jsonl``-shaped dicts.
    Same row classification as ``gate_realized.gate_realized_report``:

      * ``gate_scorer_pred is None`` ⇒ no gate decision on this row
        (SELL / sub-`n_train` / pre-`60b20d9` deploy-stale); excluded.
      * ``gate_off_dist=True`` ⇒ gate abstained, routed to the
        ``abstained`` bucket (reported, NOT in the verdict).
      * Acted rows are bucketed by ``gate_arm(gate_scorer_pred)``.
      * Realized return is ``forward_return_5d`` with SELL sign-flip
        applied for consistency with ``train_scorer`` / ``gate_audit`` /
        ``gate_realized`` (defensive — gate_scorer_pred is BUY-only).

    Returns a JSON-safe dict. Never raises.
    """
    acted: dict[str, list[float]] = {a: [] for a in _ARM_ORDER}
    abstained: list[float] = []
    n_captured = 0
    n_acted = 0
    n_abstained = 0

    try:
        it = list(rows or [])
    except Exception:
        it = []

    for r in it:
        if not isinstance(r, dict):
            continue
        gp = _f(r.get("gate_scorer_pred"))
        if gp is None:
            continue
        n_captured += 1

        fv = _f(r.get(f"forward_return_{ANCHOR_H}d"))
        if fv is None:
            # No 5d realized — cannot grade. (10/20-only rows are not
            # possible: `_fwd_ret_h` gates longer horizons strictly
            # tighter than 5d.) Defensive skip mirrors gate_realized.
            continue
        is_sell = str(r.get("action") or "BUY").upper() == "SELL"
        v = -fv if is_sell else fv

        if bool(r.get("gate_off_dist")):
            n_abstained += 1
            abstained.append(v)
            continue

        n_acted += 1
        arm, _mult = gate_arm(gp)
        acted[arm].append(v)

    arms_out = []
    # Pick a sentinel pred per arm so multipliers resolve via SSOT —
    # the same sentinel pattern gate_realized uses for the empty case.
    arm_sentinel = {
        "strong_headwind": -50.0, "mild_headwind": -5.0,
        "neutral": 0.0, "mild_tailwind": 7.5,
        "strong_tailwind": 50.0,
    }
    for arm in _ARM_ORDER:
        _, mult = gate_arm(arm_sentinel[arm])
        stats = _arm_stats(acted[arm])
        arms_out.append({"arm": arm, "multiplier": mult, **stats})

    abst_stats = _arm_stats(abstained)

    base = {
        "status": "ok",
        "verdict": "GATE_CAPTURE_NOT_YET_POPULATED",
        "measurement": "captured_then_deployed_no_reprediction",
        "anchor_horizon_days": ANCHOR_H,
        "n_captured": n_captured,
        "n_acted": n_acted,
        "n_abstained": n_abstained,
        "arms": arms_out,
        "abstained": abst_stats,
        "sharpe_tailwind_minus_headwind": None,
        "win_rate_tailwind_minus_headwind": None,
        "hint": "",
    }

    if n_captured == 0:
        base["hint"] = (
            "no row carries a non-null gate_scorer_pred — the continuous "
            "loop predates commit 60b20d9 (gate-decision capture) or has "
            "not accumulated captured rows since"
        )
        return base

    head = next(a for a in arms_out if a["arm"] == "strong_headwind")
    tail = next(a for a in arms_out if a["arm"] == "strong_tailwind")
    if (n_acted < MIN_TOTAL
            or head["n"] < MIN_ARM_N
            or tail["n"] < MIN_ARM_N):
        base["verdict"] = "INSUFFICIENT_DATA"
        base["hint"] = (
            f"need ≥{MIN_TOTAL} acted AND ≥{MIN_ARM_N} in BOTH extreme arms; "
            f"have n_acted={n_acted}, strong_headwind={head['n']}, "
            f"strong_tailwind={tail['n']}"
        )
        return base

    # Both arms have a sample (>= MIN_ARM_N) so win_rate / mean are not None
    # by _arm_stats construction. Sharpe is None when stdev==0 (a perfectly
    # constant arm — single-value sample, or every realized return equal).
    # When EITHER extreme arm's sharpe is None we cannot grade the
    # multiplier ratio in risk-adjusted terms; fall through to a named
    # verdict that calls it out honestly rather than fabricating a
    # comparison against a coerced zero (the AGENTS.md "diagnostic must
    # not pretend to know what it can't measure" discipline).
    winrate_spread = tail["win_rate"] - head["win_rate"]
    base["win_rate_tailwind_minus_headwind"] = round(winrate_spread, 4)

    if tail["sharpe"] is None or head["sharpe"] is None:
        sharpe_spread = None
    else:
        sharpe_spread = tail["sharpe"] - head["sharpe"]
        base["sharpe_tailwind_minus_headwind"] = round(sharpe_spread, 4)

    def _fmt_sh(v):
        """None-safe sharpe formatter — a constant arm has no defined
        risk-adjusted return; ``n/a`` is the honest rendering."""
        return f"{v:+.3f}" if v is not None else "n/a"

    # Verdict precedence (most-decisive first):
    #   1. WIN_RATE_INVERTED       — the gate sizes up the wrong-more-often arm
    #   2. SHARPE_UNDEFINED_ARM    — one extreme arm has zero return variance
    #   3. UNFAVORABLE_RISK        — risk-adjusted return worse on the bigger bet
    #   4. RISK_INDIFFERENT        — neither edge nor inversion clears the band
    #   5. RISK_PROPORTIONAL       — bigger bets carry better risk-adjusted reward
    if winrate_spread < -WIN_RATE_TOL:
        base["verdict"] = "WIN_RATE_INVERTED"
        base["hint"] = (
            f"strong_tailwind win_rate={tail['win_rate']:.2%} < "
            f"strong_headwind {head['win_rate']:.2%} "
            f"(spread {winrate_spread:+.2%}) — the ×1.30 arm is right LESS "
            f"often than the ×0.60 arm; even if mean is fine, the gate is "
            f"sizing UP the wrong-direction-more-often bets"
        )
    elif sharpe_spread is None:
        base["verdict"] = "SHARPE_UNDEFINED_ARM"
        base["hint"] = (
            f"one extreme arm has zero return variance "
            f"(stdev=0 ⇒ sharpe undefined): "
            f"strong_headwind sharpe={_fmt_sh(head['sharpe'])}, "
            f"strong_tailwind sharpe={_fmt_sh(tail['sharpe'])}; "
            f"risk-adjusted grade not computable for this corpus"
        )
    elif sharpe_spread < -SHARPE_TOL:
        base["verdict"] = "UNFAVORABLE_RISK"
        base["hint"] = (
            f"strong_tailwind sharpe={_fmt_sh(tail['sharpe'])} < "
            f"strong_headwind {_fmt_sh(head['sharpe'])} "
            f"(spread {sharpe_spread:+.3f}) — the gate's bigger bets carry "
            f"WORSE risk-adjusted reward; volatility scales faster than "
            f"realized return"
        )
    elif abs(sharpe_spread) <= SHARPE_TOL:
        base["verdict"] = "RISK_INDIFFERENT"
        base["hint"] = (
            f"|sharpe spread|={abs(sharpe_spread):.3f} ≤ {SHARPE_TOL:.2f} — "
            f"the gate's ×1.30 vs ×0.60 multiplier ratio is not realized in "
            f"risk-adjusted return: reallocating variance, not edge"
        )
    else:
        base["verdict"] = "RISK_PROPORTIONAL"
        base["hint"] = (
            f"strong_tailwind sharpe={_fmt_sh(tail['sharpe'])} > "
            f"strong_headwind {_fmt_sh(head['sharpe'])} "
            f"(spread {sharpe_spread:+.3f}) — the ×1.30 arm carries better "
            f"risk-adjusted reward; multiplier ratio is justified"
        )
    return base


def analyze(outcomes_path: Path | str, oos_only: bool = True) -> dict:
    """Load the outcomes file, take the temporal-OOS slice (default) and
    run the risk-profile report. No scorer / pickle is loaded — reads only
    captured fields. Read-only; never raises.

    Mirrors ``gate_realized.analyze`` byte-for-byte (same slice picker,
    same error envelope) so a future ledger consumer can swap the
    analyzer without changing its harness.
    """
    out: dict = {
        "status": "error",
        "verdict": "GATE_CAPTURE_NOT_YET_POPULATED",
        "measurement": "captured_then_deployed_no_reprediction",
        "n_captured": 0, "n_acted": 0, "arms": [],
        "slice": "all", "hint": "",
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

        rep = gate_risk_report(recs)
        rep["slice"] = slice_name
        rep["n_records_total"] = len(records)
        return rep
    except Exception as e:  # pragma: no cover - defensive
        out["hint"] = f"audit failed: {type(e).__name__}: {e}"
        return out


def _cli(argv: list[str] | None = None) -> int:
    """`python3 -m paper_trader.ml.gate_risk_profile [--all]` — gate risk
    profile from captured then-deployed decisions (no re-prediction).
    Read-only. Exits 2 on the most actionable adverse verdicts
    (``WIN_RATE_INVERTED`` / ``UNFAVORABLE_RISK``) — cron-branchable, mirroring
    ``gate_pnl``/``gate_realized``'s exit-2-on-subtracts.
    """
    import sys

    argv = sys.argv[1:] if argv is None else argv
    oos_only = "--all" not in argv
    root = Path(__file__).resolve().parent.parent.parent
    rep = analyze(root / "data" / "decision_outcomes.jsonl", oos_only=oos_only)
    print(f"VERDICT: {rep['verdict']}  ({rep.get('hint', '')})")
    sw = rep.get("sharpe_tailwind_minus_headwind")
    ww = rep.get("win_rate_tailwind_minus_headwind")
    sw_s = f"{sw:+.3f}" if sw is not None else "n/a"
    ww_s = f"{ww:+.3f}" if ww is not None else "n/a"
    print(f"  slice={rep.get('slice')}  "
          f"n_captured={rep.get('n_captured')}  "
          f"n_acted={rep.get('n_acted')}  "
          f"n_abstained={rep.get('n_abstained')}  "
          f"sharpe(tail-head)={sw_s}  "
          f"winrate(tail-head)={ww_s}")
    hdr = (f"  {'arm':<16} {'mult':>5}  {'n':>5}  {'win%':>6}  "
           f"{'mean%':>7}  {'p10%':>7}  {'p90%':>7}  {'sharpe':>7}")
    print(hdr)
    for a in rep.get("arms", []):
        n = a["n"]
        wr = a["win_rate"]
        m = a["mean"]
        p10 = a["p10"]
        p90 = a["p90"]
        sh = a["sharpe"]
        wr_s = f"{wr*100:5.1f}" if wr is not None else "  n/a"
        m_s = f"{m:+6.2f}" if m is not None else "   n/a"
        p10_s = f"{p10:+6.2f}" if p10 is not None else "   n/a"
        p90_s = f"{p90:+6.2f}" if p90 is not None else "   n/a"
        sh_s = f"{sh:+6.3f}" if sh is not None else "   n/a"
        print(f"  {a['arm']:<16} ×{a['multiplier']:.2f}  {n:>5}  "
              f"{wr_s:>5}%  {m_s:>6}%  {p10_s:>6}%  {p90_s:>6}%  {sh_s:>6}")
    bad = {"WIN_RATE_INVERTED", "UNFAVORABLE_RISK"}
    return 2 if rep.get("verdict") in bad else 0


if __name__ == "__main__":
    raise SystemExit(_cli())

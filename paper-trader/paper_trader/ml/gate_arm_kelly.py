"""Per-gate-arm Kelly fraction + risk-adjusted skill diagnostic.

Read-only research signal (2026-05-29 feature, Agent 2 ML+backtests).
Sibling to ``gate_realized`` / ``gate_audit`` / ``gate_pnl`` /
``gate_decile_realized`` — mirrors their operational discipline
exactly (never trains, never touches ``decision_scorer.pkl`` /
``decision_outcomes.jsonl`` / ``build_features`` / ``N_FEATURES`` / any
trade path; safe to run against the live unattended loop; never raises
in analyzer functions).

**The gap this closes.** The existing gate diagnostics report per-arm
**means** (``gate_realized``'s realized 5d mean per arm, ``gate_audit``'s
re-predicted arm spread). Neither answers the textbook quant question
about **risk-adjusted** per-arm skill:

  *Given the realized distribution of each arm's outcomes, what does
  Kelly's criterion say the relative sizing across arms should be — and
  how does that compare to the gate's actual* ``×0.60 / ×0.85 / ×1.00
  / ×1.15 / ×1.30`` *multiplier ladder?*

A high realized **mean** is not the same as high Kelly fraction: an arm
that returns +5% on average with σ=20% has lower Kelly than an arm
returning +3% with σ=4%. The current gate multipliers (×1.30 etc.) are
**rank-based** by the scorer's predicted-return ordering — they implicitly
assume the variance is comparable across arms. This analyzer measures
whether that assumption holds.

**Metrics reported per arm** (same five arms as ``gate_audit.gate_arm``):

  * ``n`` — sample size (captured + acted rows; off-distribution
    abstentions excluded, mirroring ``gate_realized``).
  * ``mean_pct`` — mean realized 5d return (%); = the value
    ``gate_realized`` already reports.
  * ``stdev_pct`` — sample standard deviation.
  * ``sharpe_per_trade`` — ``mean_pct / stdev_pct``. Per-trade Sharpe (not
    annualized) — the unitless risk-adjusted edge per trade. Useful for
    cross-arm comparison even when annualization assumptions are unclear.
  * ``win_rate`` — fraction of trades with ``realized > 0``.
  * ``kelly_fraction`` — full-Kelly fraction ``μ / σ²`` (with mean and
    stdev in DECIMAL, not percent — converted internally). For an arm
    where μ ≤ 0 (negative expected value), Kelly is clamped to 0.0
    (don't bet on a losing arm). Capped at 1.0 (no leverage above 100%).
    None when σ ≤ 0 (degenerate).
  * ``half_kelly`` — ``0.5 × kelly_fraction``. The practical reference;
    full Kelly maximises log-wealth but has huge drawdowns, half-Kelly
    is the textbook compromise.

**Verdict ladder** (driven by the realized-Sharpe ordering across the
two extreme arms, mirroring ``gate_realized``'s spread-based verdict):

| Verdict | Meaning |
|---------|---------|
| ``INSUFFICIENT_DATA`` | < ``MIN_TOTAL`` captured-acted rows, or either extreme arm has < ``MIN_ARM_N`` |
| ``KELLY_INVERTED`` | ``strong_headwind.sharpe > strong_tailwind.sharpe`` by more than ``SHARPE_TOL`` — the gate's "biggest bet" arm actually has WORSE risk-adjusted returns than its "smallest bet" arm. The gate's directionality is inverted on a Sharpe basis. |
| ``KELLY_AT_NOISE`` | ``|tailwind_sharpe − headwind_sharpe| ≤ SHARPE_TOL`` — no demonstrated per-trade Sharpe edge between the extreme arms; the gate's sizing is variance with no realized risk-adjusted compensation. |
| ``KELLY_ALIGNED`` | ``strong_tailwind.sharpe > strong_headwind.sharpe`` by more than ``SHARPE_TOL`` — the gate's bigger bets really do produce higher risk-adjusted return. The multiplier direction is supported by Kelly. |

Critically the verdict uses **Sharpe**, not just mean — a tailwind arm
with higher μ but also higher σ may not deserve a bigger bet. This is
the gate-relevance question existing mean-based tools cannot answer.

**Pure / module-level constants for testability**. CLI:

```bash
cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.gate_arm_kelly
cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.gate_arm_kelly --all
cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.gate_arm_kelly --json
```

The default OOS-only slice mirrors ``gate_realized`` — uses
``validation.split_outcomes_temporal`` 20% tail so the metric describes
held-out data the deployed scorer has not seen. ``--all`` runs the
analyzer on the full corpus.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Single source of truth for the five gate arms (same SSOT pattern as
# gate_realized / gate_pnl).
from .gate_audit import gate_arm

# ── Module-level constants (testable, single point of tuning) ──────────────
MIN_TOTAL = 30        # min captured+acted rows before any verdict (== gate_realized.MIN_TOTAL)
MIN_ARM_N = 5         # min trades per extreme arm to compare them (== gate_realized.MIN_ARM_N)
SHARPE_TOL = 0.02     # |sharpe_tailwind − sharpe_headwind| band that reads as noise
KELLY_CAP = 1.0       # never report > 100% Kelly (no leverage above full bet)

_ARM_ORDER = ["strong_headwind", "mild_headwind", "neutral",
              "mild_tailwind", "strong_tailwind"]

# Map arm → gate multiplier (mirrors gate_audit.gate_arm if/elif chain).
# Used to surface the prescription-vs-actual ratio in the per-arm table.
_ARM_MULTIPLIER: dict[str, float] = {
    "strong_headwind": 0.60,
    "mild_headwind": 0.85,
    "neutral": 1.00,
    "mild_tailwind": 1.15,
    "strong_tailwind": 1.30,
}


def _f(v):
    """Finite float or None — local hardening (mirrors gate_realized._f)."""
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _kelly(mean_pct: float, stdev_pct: float) -> float | None:
    """Compute full-Kelly fraction from mean and stdev in percent.

    Kelly's formula for a normal-return assumption: ``f* = μ / σ²``.
    Inputs are in PERCENT; convert to decimal so the result is a
    sizing fraction in [0, 1].

    Returns 0.0 when μ ≤ 0 (don't bet on a losing arm — Kelly says
    sit out). Returns None when σ ≤ 0 (degenerate, undefined). Capped
    at ``KELLY_CAP`` so a runaway tiny-σ arm never reports > 100%.
    Never raises.
    """
    try:
        mu = float(mean_pct) / 100.0
        sd = float(stdev_pct) / 100.0
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(mu) and np.isfinite(sd)) or sd <= 0:
        return None
    if mu <= 0:
        return 0.0
    f = mu / (sd * sd)
    return float(min(KELLY_CAP, max(0.0, f)))


def _per_arm_stats(realized: list[float]) -> dict:
    """Per-arm summary stats — pure, finite-only. Returns a JSON-safe dict."""
    n = len(realized)
    if n == 0:
        return {"n": 0, "mean_pct": None, "stdev_pct": None,
                "sharpe_per_trade": None, "win_rate": None,
                "kelly_fraction": None, "half_kelly": None,
                "median_pct": None, "min_pct": None, "max_pct": None}
    arr = np.asarray(realized, dtype=np.float64)
    mean_v = float(arr.mean())
    # Sample stdev (ddof=1) — same convention as the rest of the analytics
    # suite. n=1 special case: stdev is undefined, return None.
    stdev_v = float(arr.std(ddof=1)) if n >= 2 else 0.0
    sharpe = (mean_v / stdev_v) if (n >= 2 and stdev_v > 0) else None
    win_rate = float((arr > 0).sum() / n)
    kelly = _kelly(mean_v, stdev_v) if (n >= 2 and stdev_v > 0) else None
    return {
        "n": n,
        "mean_pct": round(mean_v, 4),
        "median_pct": round(float(np.median(arr)), 4),
        "stdev_pct": round(stdev_v, 4) if stdev_v > 0 else None,
        "min_pct": round(float(arr.min()), 4),
        "max_pct": round(float(arr.max()), 4),
        "sharpe_per_trade": round(sharpe, 4) if sharpe is not None else None,
        "win_rate": round(win_rate, 4),
        "kelly_fraction": round(kelly, 4) if kelly is not None else None,
        "half_kelly": round(kelly / 2.0, 4) if kelly is not None else None,
    }


def gate_arm_kelly_report(rows) -> dict:
    """Bucket outcomes by gate arm and report per-arm Kelly + Sharpe.

    Same row filtering as ``gate_realized.gate_realized_report``:
      * Row must have ``gate_scorer_pred`` populated (a real captured
        gate decision — pre-capture / sub-``n_train`` rows excluded).
      * ``gate_off_dist=True`` rows are routed to a separate
        ``abstained`` bucket and excluded from the verdict (the gate
        applied no multiplier — counting them in an arm would mis-
        attribute the realized return).
      * Realized target is ``forward_return_5d``, SELL-sign-flipped
        (mirrors ``train_scorer`` / ``gate_audit`` / ``gate_realized``).

    Returns a JSON-safe dict. Never raises (the analyzer must not
    break the unattended loop).
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
        fv = _f(r.get("forward_return_5d"))
        if fv is None:
            continue
        is_sell = str(r.get("action") or "BUY").upper() == "SELL"
        realized = -fv if is_sell else fv

        if bool(r.get("gate_off_dist")):
            abstained.append(realized)
            n_abstained += 1
            continue

        arm_name, _mult = gate_arm(gp)
        if arm_name not in acted:
            # Defense: an unexpected arm name (gate_audit drift) — drop
            # silently rather than crash. Should never happen in practice
            # because gate_arm is the single source of truth.
            continue
        acted[arm_name].append(realized)
        n_acted += 1

    per_arm: list[dict] = []
    for arm in _ARM_ORDER:
        stats = _per_arm_stats(acted[arm])
        stats["arm"] = arm
        stats["multiplier"] = _ARM_MULTIPLIER[arm]
        per_arm.append(stats)
    abstained_stats = _per_arm_stats(abstained)
    abstained_stats["arm"] = "abstained"
    abstained_stats["multiplier"] = None  # gate left conviction untouched

    # Verdict — based on Sharpe spread between extreme acted arms.
    headwind = next(a for a in per_arm if a["arm"] == "strong_headwind")
    tailwind = next(a for a in per_arm if a["arm"] == "strong_tailwind")
    sh = headwind.get("sharpe_per_trade")
    st = tailwind.get("sharpe_per_trade")

    if n_captured == 0:
        verdict = "GATE_CAPTURE_NOT_YET_POPULATED"
        spread = None
    elif n_acted < MIN_TOTAL or headwind["n"] < MIN_ARM_N or tailwind["n"] < MIN_ARM_N:
        verdict = "INSUFFICIENT_DATA"
        spread = None
    elif sh is None or st is None:
        # Sharpe undefined (degenerate stdev) — honest gap.
        verdict = "INSUFFICIENT_DATA"
        spread = None
    else:
        spread = round(st - sh, 4)
        if spread < -SHARPE_TOL:
            verdict = "KELLY_INVERTED"
        elif spread <= SHARPE_TOL:
            verdict = "KELLY_AT_NOISE"
        else:
            verdict = "KELLY_ALIGNED"

    return {
        "verdict": verdict,
        "n_captured": n_captured,
        "n_acted": n_acted,
        "n_abstained": n_abstained,
        "sharpe_spread": spread,
        "min_arm_n": MIN_ARM_N,
        "min_total": MIN_TOTAL,
        "sharpe_tol": SHARPE_TOL,
        "per_arm": per_arm,
        "abstained": abstained_stats,
    }


def analyze(outcomes_path: Path | str, oos_only: bool = True) -> dict:
    """Read outcomes JSONL and run the Kelly report. Never raises.

    Mirrors ``gate_realized.analyze`` semantics:
      * ``oos_only=True`` (default) applies the validation temporal 20%
        tail slice so the metric describes data the deployed scorer
        has not seen.
      * ``oos_only=False`` runs on the full corpus.

    Returns ``{status, ...report fields}``. ``status='error'`` on any
    read/parse failure; ``status='ok'`` otherwise.
    """
    p = Path(outcomes_path)
    if not p.exists():
        return {"status": "error", "verdict": "INSUFFICIENT_DATA",
                "n_captured": 0, "n_acted": 0,
                "per_arm": [], "error": f"missing {p}"}
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
        return {"status": "error", "verdict": "INSUFFICIENT_DATA",
                "n_captured": 0, "n_acted": 0,
                "per_arm": [], "error": f"read failed: {exc}"}

    if oos_only:
        try:
            from paper_trader.validation import split_outcomes_temporal
            _, rows = split_outcomes_temporal(rows, oos_fraction=0.2)
        except Exception as exc:
            # Best-effort: a split failure degrades to "report on full
            # corpus" rather than "no report". Mirror the
            # ``_train_decision_scorer`` temporal-split degradation
            # discipline (rcb pass #N — degrade not abort).
            print(f"[gate_arm_kelly] temporal split unavailable "
                  f"({exc}) — reporting on full corpus")

    report = gate_arm_kelly_report(rows)
    report["status"] = "ok"
    report["oos_only"] = bool(oos_only)
    return report


def _cli(argv: list[str] | None = None) -> int:
    """CLI entrypoint — same shape as gate_realized._cli."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.gate_arm_kelly",
        description="Per-gate-arm Kelly + Sharpe + win-rate diagnostic.",
    )
    parser.add_argument("--all", action="store_true",
                        help="Run on the full corpus (default: OOS 20%% tail).")
    parser.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON.")
    parser.add_argument("--outcomes", type=Path,
                        default=Path("data/decision_outcomes.jsonl"),
                        help="Path to the outcomes JSONL.")
    args = parser.parse_args(argv)

    report = analyze(args.outcomes, oos_only=not args.all)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report.get("status") == "ok" else 1

    print(f"[gate_arm_kelly] verdict={report.get('verdict')}  "
          f"n_acted={report.get('n_acted')}  "
          f"n_abstained={report.get('n_abstained')}  "
          f"sharpe_spread={report.get('sharpe_spread')}  "
          f"oos_only={report.get('oos_only')}")
    print()
    print(f"  {'arm':<18}{'mult':>6}{'n':>6}{'mean%':>9}"
          f"{'sd%':>8}{'sharpe':>9}{'kelly':>9}{'½kelly':>9}{'win%':>7}")
    for row in report.get("per_arm", []):
        m = row.get("multiplier")
        m_s = f"×{m:.2f}" if m is not None else "    —"
        sharpe = row.get("sharpe_per_trade")
        kelly = row.get("kelly_fraction")
        hk = row.get("half_kelly")
        mean = row.get("mean_pct")
        sd = row.get("stdev_pct")
        win = row.get("win_rate")
        print(f"  {row['arm']:<18}{m_s:>6}{row['n']:>6}"
              f"{mean if mean is not None else '—':>9}"
              f"{sd if sd is not None else '—':>8}"
              f"{sharpe if sharpe is not None else '—':>9}"
              f"{kelly if kelly is not None else '—':>9}"
              f"{hk if hk is not None else '—':>9}"
              f"{(win * 100 if win is not None else None) if False else (round(win * 100, 2) if win is not None else '—'):>7}")
    ab = report.get("abstained", {})
    if ab.get("n"):
        print(f"  {'abstained':<18}{'    —':>6}{ab['n']:>6}"
              f"{ab.get('mean_pct', '—'):>9}{ab.get('stdev_pct', '—'):>8}"
              f"{ab.get('sharpe_per_trade', '—'):>9}{'    —':>9}{'    —':>9}"
              f"{round(ab['win_rate']*100, 2) if ab.get('win_rate') is not None else '—':>7}")
    return 0 if report.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())

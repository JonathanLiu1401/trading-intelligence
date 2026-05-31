"""Kill-switch realized-effect analyzer.

Read-only research signal (2026-05-30 feature, Agent 2 ML+backtests pass #7).
Sibling to ``scorer_buy_sell_skill`` / ``gate_health_trend`` — same canonical
pattern (reads a persisted corpus, emits a verdict ladder, never trains /
writes pickle / touches trade path; safe to run against the live unattended
loop; ``analyze`` never raises).

**The gap this closes.** Pass #6 added ``gate_abstention_kind`` to
``decision_outcomes.jsonl`` rows so the OR-collapsed ``gate_off_dist=True``
boolean is now disambiguated into the SPECIFIC guard that fired:
``"clamp"`` (off-distribution scorer prediction) vs ``"killswitch"``
(trailing-OOS-IC short-circuit) vs ``None`` (gate acted on a real
prediction). The column landed but NO analyzer consumed it — so the most
operationally decisive question every gate diagnostic structurally cannot
answer was still invisible:

  *When the kill-switch abstained from modulating conviction, was the
  abstention JUSTIFIED?*

Live audit of the 2026-05-30 corpus (7075 BUYs in the outcome rows that
carry a ``gate_scorer_pred``): 6933 ``killswitch`` (98%), 142 ``clamp``
(2%), 0 ``acted``. The conviction-gate apparatus is structurally OFF in
production — every BUY runs at unmodulated conviction. The economic
question is whether the kill-switch's per-row decision to abstain
SAVED money (the gate would have boosted into losers) or COST money
(the gate would have boosted into winners we left at base conviction).

**Method.** Within each bucket, compute the tie-aware Spearman rank
correlation between ``gate_scorer_pred`` and ``forward_return_5d``.
The interpretation:

  * **killswitch bucket rank-IC ≥ +SKILL_TOL** ⇒ predictions still carried
    real signal at the cycle the kill-switch fired. Abstaining was a
    mistake — the gate's ×1.15/×1.30 arms would have legitimately boosted
    conviction into trades that DID realize higher returns.
    Verdict: ``KILLSWITCH_HURTS``.
  * **killswitch bucket rank-IC ≤ -SKILL_TOL** ⇒ predictions were
    actively ANTI-predictive. The gate's arms would have INVERTED return
    (boost into losers, suppress winners). Kill-switch did major work.
    Verdict: ``KILLSWITCH_HELPS``.
  * **|killswitch bucket rank-IC| < SKILL_TOL** ⇒ predictions were at
    noise — same as the live ``_should_gate_modulate_conviction`` reads
    on the trailing skill log. The kill-switch's abstention was
    consistent with the data; no avoidable cost.
    Verdict: ``KILLSWITCH_NEUTRAL``.
  * **n < MIN_PAIRS** ⇒ ``INSUFFICIENT_DATA``.

This is the bucket-level COMPLEMENT to ``scorer_buy_sell_skill``:
``scorer_buy_sell_skill`` answers "is the TRAILING-OOS skill signal
asymmetric across BUY/SELL?" — i.e. should the gate be on at all? —
this answers "during the cycles the kill-switch ALREADY decided the
gate should be OFF, were those decisions ECONOMICALLY right?"

Both buckets (``clamp``, ``killswitch``) and the ``acted`` bucket are
reported with the same shape so a researcher can cross-compare.
Pure / module-level constants for testability. CLI:

```bash
cd /home/zeph/trading-intelligence/paper-trader && \\
    python3 -m paper_trader.ml.kill_switch_realized_effect
cd /home/zeph/trading-intelligence/paper-trader && \\
    python3 -m paper_trader.ml.kill_switch_realized_effect --json
cd /home/zeph/trading-intelligence/paper-trader && \\
    python3 -m paper_trader.ml.kill_switch_realized_effect \\
        --outcomes path/to/decision_outcomes.jsonl
```

Exit code mirrors sibling diagnostics: 0 on benign verdicts
(``KILLSWITCH_HELPS`` / ``KILLSWITCH_NEUTRAL`` / ``INSUFFICIENT_DATA``),
2 on the actionable ``KILLSWITCH_HURTS`` verdict (the kill-switch is
suppressing a gate that WOULD have made money — operational action is
to relax the kill-switch tolerance).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# Module-level constants (testable, single point of tuning).
# Must match backtest._GATE_SKILL_IC_TOLERANCE for cross-tool consistency —
# the live kill-switch's own threshold for "skill" is +0.03 BUY-IC, so the
# verdict's bucketed rank-IC reading uses the same band. A future tuning of
# the gate cascade is reflected in one place across all diagnostics.
SKILL_TOL = 0.03
# Minimum bucket count below which the rank-IC reading is too noisy for a
# verdict. The Spearman 95% CI on 50 i.i.d. samples is roughly ±0.28; at 100
# it's roughly ±0.20; at 200 it's roughly ±0.14. 200 is the first count that
# distinguishes a +0.03 SKILL_TOL boundary from noise. The live killswitch
# bucket carries 6933 rows so the production threshold is comfortably above
# 200; this keeps a sparse historical slice from emitting a spurious verdict.
MIN_PAIRS = 200

DECISION_OUTCOMES = (Path(__file__).resolve().parent.parent.parent
                     / "data" / "decision_outcomes.jsonl")

# Three canonical abstention buckets:
#   None       → "acted"      — gate fired on a real prediction (modulated conviction)
#   "clamp"    → "clamp"      — off-distribution guard fired (scorer flagged low trust)
#   "killswitch" → "killswitch" — no-skill kill-switch fired (trailing OOS at noise)
# Rows with any other value are dropped — defense against a future
# abstention-kind addition that this analyzer hasn't been updated to
# understand. The kind is normalized to lowercase before bucketing so a
# capitalized value still maps cleanly (legacy rows are guaranteed
# lowercase by ``_parse_gate_abstention_kind``).
_BUCKETS = ("acted", "clamp", "killswitch")


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


def _bucket_for(kind) -> "str | None":
    """Map a ``gate_abstention_kind`` value to its bucket name.

    None → "acted" (the gate FIRED on a real prediction).
    "clamp" / "killswitch" → matching bucket.
    Anything else → None (drop the row).
    """
    if kind is None:
        return "acted"
    if not isinstance(kind, str):
        return None
    k = kind.strip().lower()
    if k in ("clamp", "killswitch"):
        return k
    return None


def _bucket_stats(pairs: list[tuple]) -> dict:
    """Compute summary stats for one bucket's (pred, realized) pairs.

    Returns ``{n, mean_realized, median_realized, mean_pred, rank_ic}``.
    ``rank_ic`` is None when ``n < 2`` (Spearman undefined) or when
    either side has zero variance (per ``_spearman`` semantics). Never
    raises — degenerate input degrades to a None-rich row.
    """
    n = len(pairs)
    out = {"n": n, "mean_realized": None, "median_realized": None,
           "mean_pred": None, "rank_ic": None}
    if n == 0:
        return out
    preds = np.asarray([p for p, _ in pairs], dtype=np.float64)
    rets = np.asarray([r for _, r in pairs], dtype=np.float64)
    out["mean_realized"] = round(float(np.mean(rets)), 4)
    out["median_realized"] = round(float(np.median(rets)), 4)
    out["mean_pred"] = round(float(np.mean(preds)), 4)
    if n >= 2:
        # Reuse ``calibration._spearman`` for tie-aware rank correlation —
        # single source of truth across every OOS rank-IC consumer.
        from paper_trader.ml.calibration import _spearman
        ic = _spearman(preds, rets)
        if np.isfinite(ic):
            out["rank_ic"] = round(float(ic), 4)
    return out


def kill_switch_report(rows: list[dict]) -> dict:
    """Compute kill-switch realized-effect verdict from outcome rows.

    Filters to BUYs with a ``gate_scorer_pred`` AND ``forward_return_5d``
    (the gate is BUY-only; rows without both fields carry no economic
    information for this question). Buckets by ``gate_abstention_kind``.
    Emits per-bucket summary stats + a verdict driven by the rank-IC
    inside the ``killswitch`` bucket.

    Returns a JSON-safe dict. Never raises (the analyzer must not break
    the unattended loop)."""
    try:
        it = list(rows or [])
    except Exception:
        it = []

    buckets: dict[str, list[tuple]] = {b: [] for b in _BUCKETS}
    n_buys = 0
    n_with_pred = 0
    for r in it:
        if not isinstance(r, dict):
            continue
        if str(r.get("action") or "").upper() != "BUY":
            continue
        n_buys += 1
        pred = _maybe_float(r.get("gate_scorer_pred"))
        ret = _maybe_float(r.get("forward_return_5d"))
        if pred is None or ret is None:
            continue
        n_with_pred += 1
        b = _bucket_for(r.get("gate_abstention_kind"))
        if b is None:
            continue
        buckets[b].append((pred, ret))

    per_bucket = {b: _bucket_stats(buckets[b]) for b in _BUCKETS}
    ks = per_bucket["killswitch"]
    ks_n = ks["n"]
    ks_ic = ks["rank_ic"]

    if ks_n < MIN_PAIRS or ks_ic is None:
        verdict = "INSUFFICIENT_DATA"
        hint = (f"killswitch bucket n={ks_n} (need >={MIN_PAIRS}); rank-IC "
                f"undefined or below the noise floor — no verdict.")
    elif ks_ic >= SKILL_TOL:
        verdict = "KILLSWITCH_HURTS"
        hint = (f"killswitch bucket rank-IC={ks_ic:+.3f} (n={ks_n}) "
                f">= +{SKILL_TOL} — the gate's predictions still carried "
                "real signal in the cycles the kill-switch fired. The "
                "×1.15/×1.30 arms would have legitimately boosted conviction "
                "into trades that DID realize higher returns; abstaining "
                "cost money. Operational action: consider relaxing the "
                "kill-switch tolerance.")
    elif ks_ic <= -SKILL_TOL:
        verdict = "KILLSWITCH_HELPS"
        hint = (f"killswitch bucket rank-IC={ks_ic:+.3f} (n={ks_n}) "
                f"<= -{SKILL_TOL} — predictions were ANTI-predictive in "
                "these rows. The gate's arms would have inverted return "
                "(boost into losers, suppress winners); kill-switch did "
                "major work. The bigger the magnitude here, the more value "
                "the kill-switch is adding.")
    else:
        verdict = "KILLSWITCH_NEUTRAL"
        hint = (f"killswitch bucket rank-IC={ks_ic:+.3f} (n={ks_n}) "
                f"within ±{SKILL_TOL} of zero — predictions were at noise "
                "in the cycles the kill-switch fired, consistent with the "
                "trailing-OOS-IC reads the kill-switch acts on. No "
                "avoidable cost; the abstention was justified by the data.")

    return {
        "verdict": verdict,
        "n_buys": n_buys,
        "n_with_pred": n_with_pred,
        "buckets": per_bucket,
        "min_pairs": MIN_PAIRS,
        "skill_tol": SKILL_TOL,
        "hint": hint,
    }


def analyze(outcomes_path: "Path | str | None" = None) -> dict:
    """Read ``decision_outcomes.jsonl`` and run the kill-switch report.

    Returns ``{status, ...report fields}``. ``status='error'`` on read
    failure; ``status='ok'`` otherwise. Never raises."""
    if outcomes_path is None:
        outcomes_path = DECISION_OUTCOMES
    p = Path(outcomes_path)
    if not p.exists():
        report = kill_switch_report([])
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
        report = kill_switch_report([])
        report["status"] = "error"
        report["error"] = f"read failed: {exc}"
        return report

    report = kill_switch_report(rows)
    report["status"] = "ok"
    return report


def _fmt_ic(v) -> str:
    return f"{v:+.3f}" if isinstance(v, (int, float)) else "n/a"


def _fmt_pct(v) -> str:
    return f"{v:+.2f}%" if isinstance(v, (int, float)) else "n/a"


def _cli(argv: "list[str] | None" = None) -> int:
    """CLI entrypoint. Exit codes mirror sibling diagnostics."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.kill_switch_realized_effect",
        description=("Kill-switch realized-effect audit on the persisted "
                     "decision_outcomes.jsonl. Reports the rank-IC of "
                     "gate_scorer_pred vs forward_return_5d within each "
                     "abstention bucket."),
    )
    parser.add_argument(
        "--outcomes", type=Path, default=None,
        help=("Path to the outcomes JSONL (default: "
              "data/decision_outcomes.jsonl relative to repo)."))
    parser.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)

    report = analyze(args.outcomes)
    verdict = report.get("verdict") or ""

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2 if verdict == "KILLSWITCH_HURTS" else 0

    print(f"[kill_switch_realized_effect] verdict={verdict}  "
          f"n_buys={report.get('n_buys')}  "
          f"n_with_pred={report.get('n_with_pred')}")
    buckets = report.get("buckets") or {}
    for b in _BUCKETS:
        cell = buckets.get(b) or {}
        print(f"  {b:<11} n={cell.get('n', 0):<6} "
              f"mean_pred={_fmt_pct(cell.get('mean_pred'))}  "
              f"mean_realized={_fmt_pct(cell.get('mean_realized'))}  "
              f"median_realized={_fmt_pct(cell.get('median_realized'))}  "
              f"rank-IC={_fmt_ic(cell.get('rank_ic'))}")
    print(f"  hint: {report.get('hint')}")
    return 2 if verdict == "KILLSWITCH_HURTS" else 0


if __name__ == "__main__":
    raise SystemExit(_cli())

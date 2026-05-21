"""Cross-run consensus skill diagnostic — when multiple backtest runs
made the SAME (sim_date, ticker, action) call, did the trade realize a
better forward return than when only one run made it?

This is a **read-only diagnostic**. It never trains, never touches
``decision_scorer.pkl``, ``decision_outcomes.jsonl``, ``build_features``,
``N_FEATURES``, or any trade path — same operational discipline as
``paper_trader/ml/persona_skill.py`` / ``calibration.py`` /
``gate_audit.py`` / ``skill_trend.py`` / ``conviction_calibration.py``.
Safe to run against the live unattended loop, never raises on bad input.

**The gap this fills.** The existing 35-module diagnostic suite buckets
OOS skill by sector (``sector_skill``), regime (``regime_audit``),
persona (``persona_skill``), news volume (``news_volume_skill``), leveraged
vs not (``leveraged_skill``), action (``action_skill``), ticker
(``per_ticker_skill``), conviction quintile (``conviction_calibration``),
gate arm (``gate_audit``/``gate_realized``), bubble-gate
(``bubble_gate_skill``), and feature-value quintile (``feature_value_skill``).
**None of them measures cross-run agreement on a single decision.**

Within a backtest cycle of multiple runs (RUNS_PER_CYCLE × distinct
personas), different personas can independently reach the same
(ticker, action) call on the same ``sim_date``. The hypothesis a
skeptical quant has is intuitive: when many independent personas
converge on a trade, that consensus carries more signal than a lone
opinion. The captured 7413-record outcome corpus already lets us test it:
group by ``(sim_date, ticker, action)``, bucket by the count of distinct
``run_id`` rows in the group, and compare mean realized
``forward_return_5d`` across buckets. A noticeable rising trend ⇒
``CONSENSUS_EDGE``; a flat / inverted trend ⇒ consensus is uninformative
or actively harmful (the lone-wolf calls outperform).

**Operational value.** A positive verdict legitimises "ensemble"
sizing — a future ``_ml_decide`` variant could read the cross-run
consensus count and modulate conviction (or surface to the operator as
a "high-confidence consensus" signal). A negative verdict explicitly
rules that out and saves a researcher the wasted effort. Either
verdict is a concrete next step.

**Why a multi-run group is meaningful even with RUNS_PER_CYCLE=1.**
Historical data carries ~480 groups with 2 agreeing runs and ~14 with 3+
(measured live in the current 7413-record corpus). Most of those are
from earlier cycles when ``RUNS_PER_CYCLE=5``; even with the current
throttled-to-1 setting, multi-cycle windows can occasionally overlap on
``sim_date``, and the historical data alone is enough for a stable
verdict to emerge.

Verdict ladder (test-locked, exact-value):

| Verdict | Trigger |
|---|---|
| ``INSUFFICIENT_DATA`` | < ``MIN_GROUPS`` agreement-2+ groups OR < ``MIN_ROWS`` rows in the lone bucket |
| ``INVERTED`` | lone-bucket mean > top-consensus-bucket mean by > ``REVERSAL_MIN_PCT`` (consensus is anti-predictive) |
| ``CONSENSUS_EDGE`` | top-consensus-bucket mean − lone-bucket mean ≥ ``EDGE_MIN_PCT`` AND top has more rows than ``BUCKET_MIN_ROWS`` |
| ``WEAK_EDGE`` | spread positive but below ``EDGE_MIN_PCT`` |
| ``NO_EDGE`` | spread within ``±FLAT_BAND_PCT`` of zero |

CLI:

```bash
cd /home/zeph/trading-intelligence/paper-trader

# Default — analyze data/decision_outcomes.jsonl, table verdict
python3 -m paper_trader.ml.consensus_skill

# Machine-readable
python3 -m paper_trader.ml.consensus_skill --json
```

CLI exit code 0 only on ``CONSENSUS_EDGE`` / ``NO_EDGE`` (the
acceptable verdicts); exit 2 on ``INVERTED`` (the quant-decisive
"consensus is harmful" state) so a shell caller can ``if !`` on a real
signal. Mirrors ``conviction_calibration`` / ``gate_audit``'s exit-code
contract.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

# Minimum rows in the "lone" (n_runs=1) bucket before a verdict is
# attempted — guards against a tiny corpus producing fake spreads.
MIN_ROWS = 100
# Minimum number of distinct multi-run groups (n_runs >= 2) before
# verdict is attempted. With < this many, the consensus side of the
# comparison is too noisy.
MIN_GROUPS = 20
# Minimum rows in the top-consensus bucket before that bucket is used
# as the verdict's "high consensus" side. Below this, we still report
# but the verdict degrades to INSUFFICIENT_DATA.
BUCKET_MIN_ROWS = 20

# Verdict-threshold percentage-points on mean forward_return_5d (%).
EDGE_MIN_PCT = 0.5         # top - lone ≥ 0.5pp → CONSENSUS_EDGE
REVERSAL_MIN_PCT = 0.5     # lone - top > 0.5pp → INVERTED
FLAT_BAND_PCT = 0.25       # |top - lone| ≤ 0.25pp → NO_EDGE

OUTCOMES_DEFAULT = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "decision_outcomes.jsonl"
)


def _is_finite_number(v) -> bool:
    """Like math.isfinite but tolerant of None / non-numeric / bool."""
    if isinstance(v, bool) or v is None:
        return False
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


def _bucket_label(n: int) -> str:
    if n <= 1:
        return "1"
    if n == 2:
        return "2"
    return "3+"


def consensus_report(rows: Iterable[dict]) -> dict:
    """Aggregate consensus counts and realized returns per bucket.

    Pure function — accepts an iterable of outcome dicts, returns a
    JSON-safe dict. No I/O, no train, no pickle touch.
    """
    # Group by (sim_date, ticker, action) → list of (run_id, fr_5d).
    groups: dict[tuple, list[tuple]] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        sd = r.get("sim_date")
        tk = r.get("ticker")
        ac = r.get("action")
        rid = r.get("run_id")
        fr = r.get("forward_return_5d")
        if not (isinstance(sd, str) and isinstance(tk, str) and isinstance(ac, str)):
            continue
        if not sd or not tk or not ac:
            continue
        # Need a valid run_id and a finite realized 5d return to participate.
        if not isinstance(rid, (int, float)) or isinstance(rid, bool):
            continue
        if not _is_finite_number(fr):
            continue
        key = (sd, tk, ac.upper())
        groups.setdefault(key, []).append((int(rid), float(fr)))

    # Bucket each group's row contributions by its DISTINCT-run-count.
    by_bucket: dict[str, list[float]] = {"1": [], "2": [], "3+": []}
    n_groups_by_bucket: dict[str, int] = {"1": 0, "2": 0, "3+": 0}
    for key, pairs in groups.items():
        n_distinct = len({rid for rid, _ in pairs})
        bucket = _bucket_label(n_distinct)
        n_groups_by_bucket[bucket] += 1
        # Each row in the group contributes (so a 3-run agreement gets 3
        # samples in the bucket — that reflects the realized return seen).
        for _rid, fr in pairs:
            by_bucket[bucket].append(fr)

    def _stat(xs: list[float]) -> dict:
        n = len(xs)
        if n == 0:
            return {"n": 0, "mean": None, "median": None, "std": None}
        mean = sum(xs) / n
        if n >= 2:
            var = sum((x - mean) ** 2 for x in xs) / (n - 1)
            std: float | None = math.sqrt(var)
        else:
            std = None
        srt = sorted(xs)
        if n % 2:
            median = srt[n // 2]
        else:
            median = (srt[n // 2 - 1] + srt[n // 2]) / 2
        return {"n": n, "mean": round(mean, 4),
                "median": round(median, 4),
                "std": round(std, 4) if std is not None else None}

    per_bucket = {b: _stat(by_bucket[b]) for b in ("1", "2", "3+")}
    # Group counts (distinct decisions) — informational; the row-level
    # n in per_bucket is what the verdict reads.
    for b in ("1", "2", "3+"):
        per_bucket[b]["n_groups"] = n_groups_by_bucket[b]

    # Verdict: compare lone bucket vs the strongest available consensus
    # bucket (prefer 3+, fall back to 2).
    lone_mean = per_bucket["1"]["mean"]
    lone_n = per_bucket["1"]["n"]
    # Pick the highest-consensus bucket that has enough rows.
    top_bucket: str | None = None
    for cand in ("3+", "2"):
        if per_bucket[cand]["n"] >= BUCKET_MIN_ROWS:
            top_bucket = cand
            break
    top_mean = per_bucket[top_bucket]["mean"] if top_bucket else None
    top_n = per_bucket[top_bucket]["n"] if top_bucket else 0
    multi_groups = n_groups_by_bucket["2"] + n_groups_by_bucket["3+"]

    if lone_n < MIN_ROWS or multi_groups < MIN_GROUPS or top_bucket is None:
        verdict = "INSUFFICIENT_DATA"
        spread = None
        hint = (
            f"lone_n={lone_n} (need ≥{MIN_ROWS}), "
            f"multi_groups={multi_groups} (need ≥{MIN_GROUPS}), "
            f"top_bucket_n={top_n} (need ≥{BUCKET_MIN_ROWS})"
        )
    else:
        spread = round(top_mean - lone_mean, 4)
        if spread >= EDGE_MIN_PCT:
            verdict = "CONSENSUS_EDGE"
        elif spread <= -REVERSAL_MIN_PCT:
            verdict = "INVERTED"
        elif abs(spread) <= FLAT_BAND_PCT:
            verdict = "NO_EDGE"
        else:
            verdict = "WEAK_EDGE"
        hint = (f"top={top_bucket} mean={top_mean:+.3f}% (n={top_n}) vs "
                f"lone mean={lone_mean:+.3f}% (n={lone_n}) "
                f"spread={spread:+.3f}pp")

    return {
        "status": "ok",
        "verdict": verdict,
        "spread_pct": spread,
        "top_bucket": top_bucket,
        "by_bucket": per_bucket,
        "n_groups_total": sum(n_groups_by_bucket.values()),
        "hint": hint,
    }


def _load_outcomes(path: Path) -> list[dict]:
    """Best-effort line-by-line JSONL load. A corrupt line drops; a missing
    file returns []. Never raises (a loader fault must not break a
    read-only diagnostic — the calibration / persona_skill discipline)."""
    if not path.exists():
        return []
    out: list[dict] = []
    try:
        with path.open("r") as fh:
            for ln in fh:
                if not ln.strip():
                    continue
                try:
                    rec = json.loads(ln)
                except Exception:
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
    except Exception:
        return []
    return out


def analyze(outcomes_path: "Path | str | None" = None) -> dict:
    """High-level entry: load the outcomes JSONL and produce a verdict.

    Never raises: a load fault, a missing file, or any inner exception
    degrades to a JSON-safe ``{"status": "error", "verdict":
    "INSUFFICIENT_DATA", ...}`` so the unattended continuous loop can
    schedule this without a try/except wrapper.
    """
    try:
        if outcomes_path is None:
            outcomes_path = OUTCOMES_DEFAULT
        path = Path(outcomes_path)
        rows = _load_outcomes(path)
        if not rows:
            return {
                "status": "error",
                "verdict": "INSUFFICIENT_DATA",
                "spread_pct": None,
                "top_bucket": None,
                "by_bucket": {
                    "1": {"n": 0, "mean": None, "median": None,
                          "std": None, "n_groups": 0},
                    "2": {"n": 0, "mean": None, "median": None,
                          "std": None, "n_groups": 0},
                    "3+": {"n": 0, "mean": None, "median": None,
                           "std": None, "n_groups": 0},
                },
                "n_groups_total": 0,
                "hint": f"no outcomes loaded from {path}",
            }
        return consensus_report(rows)
    except Exception as exc:
        return {
            "status": "error",
            "verdict": "INSUFFICIENT_DATA",
            "spread_pct": None,
            "top_bucket": None,
            "by_bucket": {
                "1": {"n": 0, "mean": None, "median": None,
                      "std": None, "n_groups": 0},
                "2": {"n": 0, "mean": None, "median": None,
                      "std": None, "n_groups": 0},
                "3+": {"n": 0, "mean": None, "median": None,
                       "std": None, "n_groups": 0},
            },
            "n_groups_total": 0,
            "hint": f"analyze fault: {type(exc).__name__}",
        }


def _cli(argv: list[str] | None = None) -> int:
    """CLI entry. Exit 0 on every acceptable verdict (CONSENSUS_EDGE /
    WEAK_EDGE / NO_EDGE / INSUFFICIENT_DATA — informational), exit 2 on
    ``INVERTED`` (the quant-decisive harmful-consensus state) so a shell
    caller can gate on ``$?`` like ``conviction_calibration``."""
    import argparse
    import sys

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.consensus_skill",
        description="Cross-run consensus skill diagnostic.",
    )
    p.add_argument("--outcomes", default=None,
                   help="Path to decision_outcomes.jsonl "
                        "(default: data/decision_outcomes.jsonl).")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of a table.")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    rep = analyze(args.outcomes)

    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
    else:
        v = rep.get("verdict")
        print(f"VERDICT: {v}  ({rep.get('hint', '')})")
        print(f"  n_groups_total: {rep.get('n_groups_total')}")
        bb = rep.get("by_bucket") or {}
        print(f"  {'bucket':<8}{'n_rows':>8}{'n_groups':>10}"
              f"{'mean':>10}{'median':>10}{'std':>10}")
        for b in ("1", "2", "3+"):
            s = bb.get(b) or {}
            mean = s.get("mean")
            med = s.get("median")
            std = s.get("std")
            mean_s = f"{mean:+.3f}%" if mean is not None else "n/a"
            med_s = f"{med:+.3f}%" if med is not None else "n/a"
            std_s = f"{std:.3f}" if std is not None else "n/a"
            print(f"  {b:<8}{s.get('n', 0):>8}{s.get('n_groups', 0):>10}"
                  f"{mean_s:>10}{med_s:>10}{std_s:>10}")
        spread = rep.get("spread_pct")
        if spread is not None:
            print(f"  spread (top - lone): {spread:+.3f}pp "
                  f"(top_bucket={rep.get('top_bucket')})")

    return 2 if rep.get("verdict") == "INVERTED" else 0


if __name__ == "__main__":
    raise SystemExit(_cli())

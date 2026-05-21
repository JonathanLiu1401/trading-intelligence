"""Gate-decision capture coverage diagnostic — read-only.

Every other gate diagnostic (``gate_realized.py``, ``gate_abstention.py``,
``gate_audit.py``, ``conviction_calibration.py``, ``exit_disposition_skill``)
consumes the ``gate_scorer_pred`` / ``gate_off_dist`` columns
``_compute_decision_outcomes`` captures per BUY decision row (commit
``60b20d9``). None of them ask the *prior* question: *is that capture
pipeline actually populating, AND is it populating CORRECTLY?*

This is decisive on the live corpus. Two failure modes silently kill the
``gate_realized`` family without leaving an obvious trace:

  * **Architectural drift** — a future refactor leaks ``scorer=`` text into
    SELL/HOLD reasoning, so ``_parse_gate_decision`` (BUY-only by contract,
    AGENTS.md) starts populating SELL rows with predictions the gate
    architecturally never modulated. Every gate-realized bucket then
    silently mixes a BUY-only signal with SELL ghost rows. Nothing today
    enforces the SELL∩gate_scorer_pred=∅ invariant on the persisted
    corpus.
  * **Loop dormancy** — the continuous loop stops cycling so the deployed
    pickle's ``n_train`` never crosses 500 (invariant #5). Every new BUY
    decision then writes ``gate_scorer_pred=None`` (sub-gate), and
    ``gate_realized`` reads a corpus whose recent rows carry NO gate
    activity to grade — but its verdict bucket sizes look fine because
    older rows are still captured. The capture rate decay is the only
    visible signal, and no module surfaces it.

Both modes are this diagnostic's reason to exist. It is the operational
liveness watcher for the *capture* pipeline, sibling to
``scorer_freshness`` (training-pickle liveness) and ``deploy_audit``
(config-drift): together they close the audit loop on "is the gate's
training+capture machinery actually working".

**Method.** Pure pass over ``decision_outcomes.jsonl`` (a JSONL streaming
read so memory stays O(1) of the file). Per row classify by ``action``
and presence of ``gate_scorer_pred``; tally per-quartile by ``run_id``
ascending so the temporal trend is by *write order* (gate-activity time,
which is what loop liveness changes) rather than by ``sim_date`` (the
*decision* time, which is fixed by the backtest window pick and reveals
nothing about the loop's recent behaviour). Off-distribution count is
computed only over captured-BUY rows where the field is True.

Same operational discipline as the rest of ``paper_trader/ml`` (read-only,
never trains, never touches ``decision_scorer.pkl`` / ``build_features``
/ ``N_FEATURES`` / trade path, never raises on bad input). Safe under the
live unattended loop.

Verdict ladder (crisp, threshold-driven so it is exactly testable):

| Verdict | Trigger |
|---------|---------|
| ``INSUFFICIENT_DATA`` | < ``MIN_BUY_ROWS=60`` BUY rows in the corpus |
| ``SCHEMA_VIOLATION`` | ≥1 SELL row carries a non-None ``gate_scorer_pred`` (gate is BUY-only) |
| ``GATE_CAPTURE_DARK`` | overall BUY capture rate < ``DARK_PCT=5.0``% |
| ``GATE_CAPTURE_DEGRADING`` | newest-quartile capture < oldest-quartile − ``TREND_PP=20.0``pp |
| ``GATE_CAPTURE_IMPROVING`` | newest-quartile capture > oldest-quartile + ``TREND_PP=20.0``pp |
| ``GATE_CAPTURE_PARTIAL`` | overall 5–90% with no significant trend |
| ``GATE_CAPTURE_HEALTHY`` | overall ≥ ``HEALTHY_PCT=90.0``% with no degrading trend |

Exit code mirrors the sibling diagnostics (0 = acceptable state, 2 = real
harm: schema breach, total darkness, or active degradation):

    HEALTHY / IMPROVING / PARTIAL  → 0
    INSUFFICIENT_DATA              → 0  (gap, not harm)
    DEGRADING / DARK / SCHEMA      → 2

```bash
cd /home/zeph/trading-intelligence/paper-trader
python3 -m paper_trader.ml.gate_capture_coverage
python3 -m paper_trader.ml.gate_capture_coverage --json
python3 -m paper_trader.ml.gate_capture_coverage --outcomes path/to/alt.jsonl
```
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_OUTCOMES_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "decision_outcomes.jsonl"
)

MIN_BUY_ROWS = 60          # below this Spearman/quartile splits are unstable
DARK_PCT = 5.0             # < this overall is GATE_CAPTURE_DARK
HEALTHY_PCT = 90.0         # >= this overall is GATE_CAPTURE_HEALTHY
TREND_PP = 20.0            # absolute Δpp (newest − oldest) for trend verdicts


def _safe_load_jsonl(path: Path) -> list[dict]:
    """Stream-parse a JSONL file, dropping malformed lines.

    Never raises: a missing file yields ``[]``; a syntactically invalid line
    is skipped (matching the discipline of every other ``paper_trader/ml``
    consumer of ``decision_outcomes.jsonl``). A successfully-parsed but
    non-dict row is also dropped — the per-row action / gate-field reads
    below all assume dict ``.get(...)`` semantics.
    """
    rows: list[dict] = []
    if not path.exists():
        return rows
    try:
        with path.open("r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
    except OSError:
        return []
    return rows


def _quartile_capture_pct(rows: list[dict]) -> tuple[float | None, float | None,
                                                     int, int]:
    """Return (oldest_pct, newest_pct, oldest_n, newest_n) for BUY rows
    sorted ascending by ``run_id``.

    Rows missing ``run_id`` are EXCLUDED from the trend split (they cannot
    be ordered) but still contribute to the aggregate capture rate above.
    Returns ``(None, None, 0, 0)`` when there are fewer than 4 ordered
    rows (quartiles undefined).
    """
    ordered = [r for r in rows
               if isinstance(r.get("run_id"), (int, float))]
    ordered.sort(key=lambda r: r["run_id"])
    n = len(ordered)
    if n < 4:
        return None, None, 0, 0
    q_size = n // 4
    if q_size == 0:
        return None, None, 0, 0
    oldest = ordered[:q_size]
    newest = ordered[-q_size:]
    oldest_cap = sum(1 for r in oldest
                     if r.get("gate_scorer_pred") is not None)
    newest_cap = sum(1 for r in newest
                     if r.get("gate_scorer_pred") is not None)
    return (oldest_cap / q_size * 100.0,
            newest_cap / q_size * 100.0,
            q_size, q_size)


def analyze(outcomes_path: Path | str | None = None) -> dict:
    """Audit ``decision_outcomes.jsonl``'s gate-capture pipeline.

    Read-only; never raises. Returns a JSON-safe dict with a crisp
    threshold-driven verdict suitable for both human inspection and
    machine consumption (per the operational-discipline contract).
    """
    if outcomes_path is None:
        outcomes_path = DEFAULT_OUTCOMES_PATH
    path = Path(outcomes_path)
    rows = _safe_load_jsonl(path)

    buy_rows: list[dict] = []
    sell_rows: list[dict] = []
    schema_violations: list[dict] = []
    for r in rows:
        action = str(r.get("action") or "").upper()
        if action == "BUY":
            buy_rows.append(r)
        elif action == "SELL":
            sell_rows.append(r)
            # Architectural invariant: SELL reasoning is emitted by the
            # SELL branch of `_ml_decide` which does NOT include the
            # `scorer=` token (the gate is BUY-only). If `gate_scorer_pred`
            # is non-None on a SELL row, the contract has been breached —
            # either by a future _ml_decide refactor leaking the token into
            # SELL reasoning, or by an upstream rewriter mis-populating the
            # field. Capture concrete violators for the report so a
            # follow-up fix has a specific row to inspect.
            if r.get("gate_scorer_pred") is not None:
                schema_violations.append({
                    "run_id": r.get("run_id"),
                    "sim_date": r.get("sim_date"),
                    "ticker": r.get("ticker"),
                    "gate_scorer_pred": r.get("gate_scorer_pred"),
                })

    total_buys = len(buy_rows)
    total_sells = len(sell_rows)
    buy_captured = sum(1 for r in buy_rows
                       if r.get("gate_scorer_pred") is not None)
    buy_off_dist = sum(1 for r in buy_rows
                       if r.get("gate_off_dist") is True)
    overall_pct = (buy_captured / total_buys * 100.0
                   if total_buys else 0.0)

    (oldest_pct, newest_pct,
     oldest_n, newest_n) = _quartile_capture_pct(buy_rows)
    trend_pp: float | None
    if oldest_pct is not None and newest_pct is not None:
        trend_pp = newest_pct - oldest_pct
    else:
        trend_pp = None

    # Verdict ladder — order is load-bearing. SCHEMA_VIOLATION is *always*
    # surfaced first because it is a contract breach regardless of
    # quantitative coverage. INSUFFICIENT_DATA precedes any quant verdict
    # so a thin corpus reads honestly. DARK then DEGRADING/IMPROVING then
    # HEALTHY/PARTIAL.
    if schema_violations:
        verdict = "SCHEMA_VIOLATION"
    elif total_buys < MIN_BUY_ROWS:
        verdict = "INSUFFICIENT_DATA"
    elif overall_pct < DARK_PCT:
        verdict = "GATE_CAPTURE_DARK"
    elif (trend_pp is not None and trend_pp <= -TREND_PP):
        verdict = "GATE_CAPTURE_DEGRADING"
    elif (trend_pp is not None and trend_pp >= TREND_PP):
        # A capture rate that grew newer-vs-older by ≥TREND_PP is a real
        # improvement (the loop activated, the pickle crossed n_train≥500
        # mid-corpus). Surface it as its own verdict so a human reading
        # the report sees the positive trajectory rather than collapsing
        # into the partial bucket — the symmetric sibling to DEGRADING.
        verdict = "GATE_CAPTURE_IMPROVING"
    elif overall_pct >= HEALTHY_PCT:
        verdict = "GATE_CAPTURE_HEALTHY"
    else:
        verdict = "GATE_CAPTURE_PARTIAL"

    return {
        "status": "ok",
        "verdict": verdict,
        "path": str(path),
        "total_rows": len(rows),
        "total_buys": total_buys,
        "total_sells": total_sells,
        "buy_captured": buy_captured,
        "buy_sub_gate": total_buys - buy_captured,
        "overall_buy_capture_pct": round(overall_pct, 2),
        "oldest_quartile_buy_capture_pct": (
            None if oldest_pct is None else round(oldest_pct, 2)
        ),
        "newest_quartile_buy_capture_pct": (
            None if newest_pct is None else round(newest_pct, 2)
        ),
        "quartile_n": oldest_n,  # symmetric — oldest_n == newest_n
        "trend_pp": None if trend_pp is None else round(trend_pp, 2),
        "buy_off_dist_count": buy_off_dist,
        "buy_off_dist_pct_of_captured": (
            round(buy_off_dist / buy_captured * 100.0, 2)
            if buy_captured else 0.0
        ),
        "schema_violation_count": len(schema_violations),
        "schema_violations": schema_violations[:10],  # cap for report size
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.gate_capture_coverage",
        description=("Gate-decision capture coverage diagnostic. Reads "
                     "decision_outcomes.jsonl and verifies the BUY-only "
                     "gate_scorer_pred capture pipeline is populating "
                     "correctly. Read-only — never trains or writes."),
    )
    p.add_argument("--outcomes", default=None,
                   help="Path to decision_outcomes.jsonl (default: "
                        f"{DEFAULT_OUTCOMES_PATH})")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of a table.")
    return p


def _format_table(rep: dict) -> str:
    lines = []
    lines.append(f"[gate_capture_coverage] verdict={rep['verdict']}")
    lines.append(f"  path: {rep['path']}")
    lines.append(f"  total rows: {rep['total_rows']}  "
                 f"BUY: {rep['total_buys']}  SELL: {rep['total_sells']}")
    lines.append(f"  BUY captured: {rep['buy_captured']} / "
                 f"{rep['total_buys']}  "
                 f"({rep['overall_buy_capture_pct']:.1f}%)")
    lines.append(f"  BUY sub-gate (uncaptured): {rep['buy_sub_gate']}")
    if rep['oldest_quartile_buy_capture_pct'] is not None:
        lines.append(
            f"  oldest quartile (n={rep['quartile_n']}): "
            f"{rep['oldest_quartile_buy_capture_pct']:.1f}% captured")
        lines.append(
            f"  newest quartile (n={rep['quartile_n']}): "
            f"{rep['newest_quartile_buy_capture_pct']:.1f}% captured")
        lines.append(f"  trend (newest − oldest): {rep['trend_pp']:+.1f}pp")
    lines.append(
        f"  off-distribution captures: {rep['buy_off_dist_count']} "
        f"({rep['buy_off_dist_pct_of_captured']:.2f}% of captured BUYs)")
    if rep['schema_violation_count']:
        lines.append(f"  ! SCHEMA VIOLATIONS: "
                     f"{rep['schema_violation_count']} SELL row(s) carry "
                     f"non-None gate_scorer_pred:")
        for v in rep['schema_violations']:
            lines.append(f"      run_id={v['run_id']} "
                         f"sim_date={v['sim_date']} "
                         f"ticker={v['ticker']} "
                         f"pred={v['gate_scorer_pred']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Returns 0 for HEALTHY / IMPROVING / PARTIAL / INSUFFICIENT_DATA, 2 for
    GATE_CAPTURE_DARK / GATE_CAPTURE_DEGRADING / SCHEMA_VIOLATION."""
    import sys

    args = _build_arg_parser().parse_args(
        sys.argv[1:] if argv is None else argv)

    rep = analyze(args.outcomes)
    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
    else:
        print(_format_table(rep))

    bad = {"GATE_CAPTURE_DARK", "GATE_CAPTURE_DEGRADING", "SCHEMA_VIOLATION"}
    return 2 if rep["verdict"] in bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

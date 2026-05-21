"""SPY-benchmark integrity audit — what fraction of completed backtest runs
have a fabricated ``vs_spy_pct`` because their ``spy_return_pct=0.0`` on a
non-trivial (>=30 day) window?

This is a **read-only diagnostic**. Same operational discipline as
``paper_trader/ml/gate_utilization.py`` / ``gate_audit.py`` / ``calibration.py``:
never trains, never touches ``decision_scorer.pkl`` /
``decision_outcomes.jsonl`` / ``build_features`` / any trade path. Safe under
the live unattended continuous loop. Pure DB read.

**Why this exists.** ``backtest.py::run_one`` (around line 2697) has a
benchmark-honesty guard that flags ``benchmark_unavailable`` in the run's
``notes`` column when SPY's price series is empty OR ``spy_return == 0.0``
over a ≥30-day window — both are degenerate ``returns_pct`` outputs that
make ``vs_spy_pct = total_return - 0`` a fabricated number, not real alpha.

That guard catches every NEW run. But it landed AFTER ~80 historical runs
had already been finalized with ``spy_return_pct=0.0`` and an empty ``notes``
column — those runs still appear in:

  * ``run_continuous_backtests.TOP_RUNS_TO_TRAIN`` (winner selection by
    ``total_return_pct`` is fine, but any analytics keyed on ``vs_spy_pct``
    treat them as monster alpha)
  * the dashboard's per-run alpha column
  * ``strategy.py::_ml_is_qualified``'s "last 20 qualifying runs median α"
    gate (its SQL filters ``vs_spy_pct IS NOT NULL`` but does NOT filter the
    spy=0 degenerate case — those rows DO have non-NULL vs_spy_pct, just a
    fabricated one). A median lifted by a fake +1330% alpha row passes the
    qualifier and arms the ML advisor on data that isn't real.

This module surfaces the contamination as a one-shot diagnostic so a
skeptical quant can:

  * see *how much* of the historical alpha-leaderboard is fake-benchmark
  * see whether the leak is **active** (new unflagged runs still appearing —
    the guard regressed) or **historical** (all unflagged are pre-guard)
  * see the **economic impact** on the live trader's qualifier gate —
    "does excluding these flip the gate's verdict?"

**Method.** Single SQL pass over ``backtest_runs`` (the schema is fixed
since the 2026-05 migrations; columns ``run_id``, ``start_date``,
``end_date``, ``status``, ``spy_return_pct``, ``vs_spy_pct``, ``n_trades``,
``notes`` are all present). Degenerate criterion mirrors the in-engine
guard exactly:

    is_degenerate := status='complete'
                   AND spy_return_pct = 0.0
                   AND (julianday(end_date) - julianday(start_date)) >= 30

Flagged subset:

    is_flagged := is_degenerate AND notes LIKE '%benchmark_unavailable%'

The audit reports counts, per-bucket distribution (run_id ranges so the
operator can see at a glance whether the leak is frozen), and the
**economic impact** on the live trader's ``_ml_is_qualified`` window: the
median ``vs_spy_pct`` of the last ``QUALIFIER_WINDOW=20`` qualifying runs,
both as-is and after excluding unflagged-degenerate rows.

Reports:

| Field | Meaning |
|---|---|
| ``total`` | total ``status='complete'`` runs |
| ``flagged_degenerate`` | runs caught by the in-engine guard (good — these have notes set) |
| ``unflagged_degenerate`` | runs with fabricated SPY benchmark, NO note (the leak) |
| ``unflagged_pct`` | ``unflagged_degenerate / total`` as a fraction |
| ``run_id_buckets`` | per-500-run-id bucket counts of unflagged — flat == leak frozen, growing == leak active |
| ``max_unflagged_run_id`` | highest run_id with unflagged degeneracy |
| ``max_flagged_run_id`` | highest run_id with the proper flag |
| ``qualifier_window`` | dict with ``n``, ``median_alpha_asis``, ``median_alpha_clean``, ``n_unflagged_in_window`` |
| ``verdict`` | crisp threshold-driven verdict (see below) |

Verdict ladder (most alarming first):

| Verdict | Trigger |
|---|---|
| ``INSUFFICIENT_DATA`` | < ``MIN_TOTAL`` complete runs |
| ``ACTIVE_LEAK`` | ≥1 unflagged-degenerate run in the most recent ``LEAK_DETECTION_WINDOW`` runs by run_id — the guard regressed, or a code path bypasses it |
| ``HISTORICAL_CONTAMINATION`` | unflagged-degenerate runs exist but are all older than the leak window — pre-guard data pollution that won't grow |
| ``FLAGGED_ONLY`` | every degenerate run is properly flagged, no leak |
| ``CLEAN`` | no degenerate runs at all (no SPY-empty windows ever hit) |

```bash
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.benchmark_integrity
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.benchmark_integrity --json
```
"""
from __future__ import annotations

import json
import sqlite3
import statistics
from pathlib import Path

# Thresholds — module-level so tests assert exact verdicts at boundaries and
# any tuning change is one reviewable edit. Mirrors the gate_utilization /
# calibration / news_volume_skill convention.
MIN_TOTAL = 30           # need a real sample before any verdict
LEAK_DETECTION_WINDOW = 50   # how many most-recent runs an ACTIVE_LEAK scans
QUALIFIER_WINDOW = 20    # match strategy.py::ML_QUALIFY_MIN_RUNS
MIN_RUN_WINDOW_DAYS = 30  # mirrors backtest.py guard at line ~2697
BUCKET_SIZE = 500        # per-500-run-id bucketing for the leak-rate trend

# Default DB path — resolved at call time, not at def time (matches AGENTS.md
# "hardcoded paths must be module-level for testability" rule). Module-level
# so tests can monkeypatch.
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "backtest.db"


def _safe_median(values: list[float]) -> float | None:
    """Median of a non-empty list, or None on empty. Pure / never raises."""
    if not values:
        return None
    try:
        return round(float(statistics.median(values)), 4)
    except statistics.StatisticsError:
        return None


def benchmark_integrity_report(rows: list[dict]) -> dict:
    """Audit a list of run dicts for SPY-benchmark integrity.

    ``rows`` is any iterable of dicts with at least the keys:
      ``run_id`` (int), ``start_date`` (ISO str), ``end_date`` (ISO str),
      ``status`` (str), ``spy_return_pct`` (float or None), ``vs_spy_pct``
      (float or None), ``n_trades`` (int or None), ``notes`` (str or None).

    Only ``status='complete'`` rows count toward the audit; everything else
    is silently ignored (mirrors the in-engine guard's scope — running /
    failed rows have no finalized benchmark to check).

    Never raises — a malformed row is dropped from that row's check only.
    Returns a JSON-safe dict (``run_id_buckets`` is a list of dicts so JSON
    payload ordering is deterministic across Python implementations).
    """
    base: dict = {
        "status": "ok",
        "verdict": "INSUFFICIENT_DATA",
        "total": 0,
        "flagged_degenerate": 0,
        "unflagged_degenerate": 0,
        "unflagged_pct": 0.0,
        "run_id_buckets": [],
        "max_unflagged_run_id": None,
        "max_flagged_run_id": None,
        "qualifier_window": {
            "window": QUALIFIER_WINDOW,
            "n": 0,
            "n_unflagged_in_window": 0,
            "median_alpha_asis": None,
            "median_alpha_clean": None,
            "median_alpha_delta": None,
        },
        "hint": "",
    }

    # ── 1. Parse + classify ──────────────────────────────────────────────
    completed: list[dict] = []
    for r in rows or []:
        try:
            status = str(r.get("status") or "")
            if status != "complete":
                continue
            rid = int(r.get("run_id"))
            sd_s = str(r.get("start_date") or "")
            ed_s = str(r.get("end_date") or "")
            from datetime import date
            sd = date.fromisoformat(sd_s)
            ed = date.fromisoformat(ed_s)
            win_days = (ed - sd).days
            spy = r.get("spy_return_pct")
            vsspy = r.get("vs_spy_pct")
            notes = str(r.get("notes") or "")
            try:
                spy_f = float(spy) if spy is not None else None
            except (TypeError, ValueError):
                spy_f = None
            try:
                vsspy_f = float(vsspy) if vsspy is not None else None
            except (TypeError, ValueError):
                vsspy_f = None
            try:
                nt = int(r.get("n_trades") or 0)
            except (TypeError, ValueError):
                nt = 0
            # Degenerate iff SPY returned exactly 0.0 over a window long
            # enough that a real flat SPY is implausible. Identical to the
            # in-engine elif guard's condition.
            is_degenerate = (
                spy_f is not None
                and spy_f == 0.0
                and win_days >= MIN_RUN_WINDOW_DAYS
            )
            is_flagged = is_degenerate and "benchmark_unavailable" in notes
            completed.append({
                "run_id": rid,
                "win_days": win_days,
                "spy_return_pct": spy_f,
                "vs_spy_pct": vsspy_f,
                "n_trades": nt,
                "is_degenerate": is_degenerate,
                "is_flagged": is_flagged,
                "is_unflagged_degenerate": is_degenerate and not is_flagged,
            })
        except Exception:
            # A single malformed row never breaks the report.
            continue

    base["total"] = len(completed)

    if not completed:
        base["hint"] = "no complete runs in backtest.db"
        return base

    # ── 2. Aggregate counts ──────────────────────────────────────────────
    flagged = [c for c in completed if c["is_flagged"]]
    unflagged = [c for c in completed if c["is_unflagged_degenerate"]]
    base["flagged_degenerate"] = len(flagged)
    base["unflagged_degenerate"] = len(unflagged)
    base["unflagged_pct"] = round(len(unflagged) / len(completed), 4)
    base["max_unflagged_run_id"] = (
        max((c["run_id"] for c in unflagged), default=None)
    )
    base["max_flagged_run_id"] = (
        max((c["run_id"] for c in flagged), default=None)
    )

    # Per-bucket distribution — surface whether the leak is frozen.
    bucket_counts: dict[int, int] = {}
    for c in unflagged:
        b = (c["run_id"] // BUCKET_SIZE) * BUCKET_SIZE
        bucket_counts[b] = bucket_counts.get(b, 0) + 1
    base["run_id_buckets"] = [
        {"bucket_start": k, "bucket_end": k + BUCKET_SIZE - 1, "n_unflagged": v}
        for k, v in sorted(bucket_counts.items())
    ]

    # ── 3. Economic-impact: live trader's qualifier window ──────────────
    # Mirror strategy.py::_ml_is_qualified's filter exactly:
    #   status='complete' AND vs_spy_pct IS NOT NULL AND n_trades >= 5
    # ORDER BY run_id DESC LIMIT QUALIFIER_WINDOW.
    qualifying = [
        c for c in completed
        if c["vs_spy_pct"] is not None and c["n_trades"] >= 5
    ]
    qualifying.sort(key=lambda c: -c["run_id"])
    window = qualifying[:QUALIFIER_WINDOW]
    window_unflagged = [c for c in window if c["is_unflagged_degenerate"]]
    asis = [float(c["vs_spy_pct"]) for c in window]
    clean = [float(c["vs_spy_pct"]) for c in window
             if not c["is_unflagged_degenerate"]]
    med_asis = _safe_median(asis)
    med_clean = _safe_median(clean)
    delta = (round(med_asis - med_clean, 4)
             if med_asis is not None and med_clean is not None else None)
    base["qualifier_window"] = {
        "window": QUALIFIER_WINDOW,
        "n": len(window),
        "n_unflagged_in_window": len(window_unflagged),
        "median_alpha_asis": med_asis,
        "median_alpha_clean": med_clean,
        "median_alpha_delta": delta,
    }

    # ── 4. Verdict ladder ────────────────────────────────────────────────
    if len(completed) < MIN_TOTAL:
        base["hint"] = (
            f"need ≥{MIN_TOTAL} complete runs for a verdict; "
            f"have n={len(completed)}"
        )
        return base

    if not flagged and not unflagged:
        base["verdict"] = "CLEAN"
        base["hint"] = (
            f"no degenerate SPY benchmarks across {len(completed)} complete "
            f"runs — every run carries a real spy_return"
        )
        return base

    # ACTIVE_LEAK: a recent run has unflagged degeneracy. The bound is
    # ``LEAK_DETECTION_WINDOW`` most-recent run_ids (by id, since run_id is
    # monotone). If ANY of those have ``is_unflagged_degenerate``, the
    # engine's elif guard is missing them in production — the strongest
    # actionable signal in this audit.
    completed_by_rid = sorted(completed, key=lambda c: -c["run_id"])
    leak_window = completed_by_rid[:LEAK_DETECTION_WINDOW]
    leak_unflagged = [c for c in leak_window if c["is_unflagged_degenerate"]]
    if leak_unflagged:
        base["verdict"] = "ACTIVE_LEAK"
        leak_rids = ", ".join(str(c["run_id"]) for c in leak_unflagged[:5])
        more = "" if len(leak_unflagged) <= 5 else f" (+{len(leak_unflagged)-5} more)"
        base["hint"] = (
            f"{len(leak_unflagged)} unflagged-degenerate run(s) in the last "
            f"{LEAK_DETECTION_WINDOW} runs by run_id: {leak_rids}{more} — "
            f"the in-engine benchmark guard is being bypassed"
        )
        return base

    if unflagged:
        base["verdict"] = "HISTORICAL_CONTAMINATION"
        base["hint"] = (
            f"{len(unflagged)} unflagged-degenerate run(s) "
            f"({base['unflagged_pct']*100:.1f}% of {len(completed)} total) "
            f"all older than the last {LEAK_DETECTION_WINDOW} runs — "
            f"pre-guard pollution, leak is frozen"
        )
        return base

    base["verdict"] = "FLAGGED_ONLY"
    base["hint"] = (
        f"{len(flagged)} degenerate run(s) all properly flagged — the "
        f"in-engine guard is working as intended"
    )
    return base


def analyze(db_path: Path | str | None = None) -> dict:
    """Read ``backtest_runs`` from the on-disk SQLite DB and report.

    Returns a JSON-safe dict. Never raises — any IO/fault degrades to
    ``status='error'`` with an ``INSUFFICIENT_DATA`` verdict (mirrors the
    sibling analyzers).
    """
    out: dict = {
        "status": "error", "verdict": "INSUFFICIENT_DATA",
        "total": 0, "flagged_degenerate": 0, "unflagged_degenerate": 0,
        "unflagged_pct": 0.0, "run_id_buckets": [],
        "max_unflagged_run_id": None, "max_flagged_run_id": None,
        "qualifier_window": {
            "window": QUALIFIER_WINDOW, "n": 0,
            "n_unflagged_in_window": 0,
            "median_alpha_asis": None, "median_alpha_clean": None,
            "median_alpha_delta": None,
        },
        "hint": "",
    }
    try:
        path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
        if not path.exists():
            out["hint"] = f"backtest db missing: {path}"
            return out
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT run_id, start_date, end_date, status, spy_return_pct, "
                "vs_spy_pct, n_trades, notes FROM backtest_runs"
            ).fetchall()
        finally:
            try:
                conn.close()
            except Exception:
                pass
        return benchmark_integrity_report([dict(r) for r in rows])
    except Exception as exc:
        out["hint"] = f"analyze fault: {type(exc).__name__}: {exc}"
        return out


def _cli() -> int:
    """``python3 -m paper_trader.ml.benchmark_integrity [--json] [--db PATH]``.

    Exit code mirrors the verdict severity for shell-script gating:
    0 = CLEAN / FLAGGED_ONLY (no leak);
    1 = HISTORICAL_CONTAMINATION;
    2 = ACTIVE_LEAK;
    3 = INSUFFICIENT_DATA / error.
    """
    import argparse

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.benchmark_integrity",
        description="SPY-benchmark integrity audit — fraction of completed "
                    "backtest runs with a fabricated vs_spy_pct (degenerate "
                    "spy_return_pct=0 over a non-trivial window).",
    )
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of a table.")
    p.add_argument("--db", default=None,
                   help=f"Path to backtest.db (default: {DEFAULT_DB_PATH}).")
    args = p.parse_args()

    rep = analyze(args.db)

    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
    else:
        verdict = rep.get("verdict", "INSUFFICIENT_DATA")
        total = rep.get("total", 0)
        flagged = rep.get("flagged_degenerate", 0)
        unflagged = rep.get("unflagged_degenerate", 0)
        pct = rep.get("unflagged_pct", 0.0) * 100
        print(f"[benchmark_integrity]  total={total}  verdict={verdict}")
        if rep.get("hint"):
            print(f"  {rep['hint']}")
        print(f"  flagged degenerate   : {flagged}")
        print(f"  unflagged degenerate : {unflagged} ({pct:.1f}% of total)")
        max_uf = rep.get("max_unflagged_run_id")
        max_f = rep.get("max_flagged_run_id")
        if max_uf is not None:
            print(f"  max unflagged run_id : {max_uf}")
        if max_f is not None:
            print(f"  max flagged   run_id : {max_f}")
        buckets = rep.get("run_id_buckets") or []
        if buckets:
            print("  unflagged by run_id bucket:")
            for b in buckets:
                print(f"    {b['bucket_start']:5d}-{b['bucket_end']:5d}: "
                      f"{b['n_unflagged']}")
        qw = rep.get("qualifier_window") or {}
        if qw.get("n"):
            asis = qw.get("median_alpha_asis")
            clean = qw.get("median_alpha_clean")
            delta = qw.get("median_alpha_delta")
            print(f"  qualifier window (last {qw['window']} runs, "
                  f"strategy._ml_is_qualified):")
            print(f"    n                 : {qw['n']}")
            print(f"    n_unflagged       : {qw['n_unflagged_in_window']}")
            print(f"    median α (asis)   : "
                  f"{asis:+.2f}%" if asis is not None else "n/a")
            print(f"    median α (clean)  : "
                  f"{clean:+.2f}%" if clean is not None else "n/a")
            if delta is not None:
                print(f"    median α (delta)  : {delta:+.2f}pp")

    verdict = rep.get("verdict", "INSUFFICIENT_DATA")
    if verdict in ("CLEAN", "FLAGGED_ONLY"):
        return 0
    if verdict == "HISTORICAL_CONTAMINATION":
        return 1
    if verdict == "ACTIVE_LEAK":
        return 2
    return 3


if __name__ == "__main__":
    raise SystemExit(_cli())

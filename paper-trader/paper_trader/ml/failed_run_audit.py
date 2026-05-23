"""Failed-backtest-run forensic audit — read-only.

Sibling to ``deploy_audit`` / ``scorer_freshness`` / ``scorer_pickle_smoke``
and to the dashboard's complete-only run aggregate. The other surfaces
answer "is the deployed model OK?" and "is the loop alive?". This module
answers a completely different — but quant-decisive — question:

    *Of the ``status='failed'`` rows in ``backtest.db``, how many are real
    engine failures vs. OOM-reaped runs that actually carried real trade
    data the dashboard's complete-only aggregate silently excludes?*

The exact bias this exposes was documented in AGENTS.md (2026-05-23 ML
pass #21 finding):

    "Orphan-reaper marked runs 5981-5985 as 'failed' despite carrying real
     returns (+101% / -4% / +47% / +33% / +31%) and 1000+ trades each —
     those are reaped OOM-kill rows, not data faults, but the dashboard's
     complete-only aggregate misses them."

The dashboard's published median ``vs_spy_pct`` ≈ +42% (pass #21) is
computed across ``status='complete'`` rows only. If OOM-reaped failures
are systematically high-trade / long-window / aggressive-persona rows
(plausible — those are precisely the cycles that exceed RSS budget),
excluding them is NOT random sampling — it's a selection bias that
overstates the realized alpha a quant reading the dashboard would
underwrite. This analyzer surfaces the magnitude of that bias by
reporting:

  * how many ``status='failed'`` rows look LIKELY_OOM_REAPED vs GENUINE_FAILURE
  * the median vs_spy_pct of the hidden OOM-reaped slice
  * the shift in aggregate median vs_spy_pct if those rows were included

Same operational discipline as every sibling diagnostic
(``deploy_audit``, ``scorer_pickle_smoke``, ``stop_out_audit``): read-only,
no train, no pickle write, no trade path — safe to run against the live
unattended loop. Never raises — every fault degrades to an honest
``INSUFFICIENT_DATA`` verdict.

```bash
cd /home/zeph/trading-intelligence/paper-trader && \
    python3 -m paper_trader.ml.failed_run_audit
# Exit 2 on MIXED_REAPED_AND_FAILURE / MOSTLY_OOM_REAPED (operator-
# actionable: the dashboard's complete-only aggregate is biased);
# 0 on NO_FAILED_RUNS / ALL_GENUINE_FAILURE / INSUFFICIENT_DATA.
```
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

# A `status='failed'` row carrying this many or more trades is structurally
# more consistent with "OOM-reaped mid-run" than "genuine engine crash":
# `_ml_decide` produces decisions every cycle and many of them fill, so a
# crash before any meaningful filling is the genuine-failure footprint;
# 100+ trades is unambiguously a productive run that was killed, not one
# that never started.
#
# Calibrated against the pass #21 documented evidence: runs 5981-5985 each
# carried 1000+ trades. Any threshold ≪ 1000 catches them; 100 is a
# conservative floor that still excludes the legitimate "ran 5 cycles
# before crashing" partials.
MIN_TRADES_FOR_REAL_RUN = 100

# Verdict thresholds — share of failed rows that look OOM-reaped. Below
# ``OOM_REAPED_LOW_PCT`` the failure population is dominated by genuine
# crashes (operator should look at the engine, not the host's RSS). Above
# ``OOM_REAPED_HIGH_PCT`` the failure population is mostly OOM-kills, so
# the host's memory headroom is the real story and the dashboard's
# complete-only aggregate is the most biased.
OOM_REAPED_LOW_PCT = 0.20
OOM_REAPED_HIGH_PCT = 0.80


def _classify_failed_row(row: dict) -> str:
    """Classify one ``status='failed'`` row.

    Returns ``"LIKELY_OOM_REAPED"`` or ``"GENUINE_FAILURE"``.

    The classification leans on two orthogonal signals so a regression in
    either still produces the right answer:

      1. ``notes`` contains ``"[reaped"`` — set by
         ``run_continuous_backtests._reap_orphaned_runs`` for rows whose
         ``status='running'`` exceeded 6h (the explicit reaper marker)
      2. ``n_trades >= MIN_TRADES_FOR_REAL_RUN`` — the structural signal:
         100+ trades means the engine was productively running, so the
         row carries real partial trade data even if ``finalize_run``
         never executed.

    Either signal alone is sufficient; both False ⇒ GENUINE_FAILURE.

    Note that ``vs_spy_pct`` is NOT a reliable classification signal: the
    ``backtest_runs`` schema declares it ``REAL NOT NULL DEFAULT 0``, so a
    row whose ``finalize_run`` never executed (the OOM-reaped case)
    carries the placeholder ``0.0``, not ``NULL``. Use ``n_trades`` and
    the reaper marker instead — the trade-count signal is the load-bearing
    one because the engine writes ``backtest_trades`` rows as they fill,
    independent of finalization.
    """
    notes = str(row.get("notes") or "")
    if "[reaped" in notes:
        return "LIKELY_OOM_REAPED"
    try:
        n_trades = int(row.get("n_trades") or 0)
    except (TypeError, ValueError):
        n_trades = 0
    if n_trades >= MIN_TRADES_FOR_REAL_RUN:
        return "LIKELY_OOM_REAPED"
    return "GENUINE_FAILURE"


def _has_real_vs_spy(row: dict) -> bool:
    """True if a row's ``vs_spy_pct`` is a real finalize_run-written
    value (not the schema's NOT NULL DEFAULT 0 placeholder).

    ``finalize_run`` is the only writer that populates ``completed_at``.
    A row with ``completed_at IS NULL`` carries the schema's
    DEFAULT 0 for vs_spy_pct — treating that as realized alpha
    contaminates the bias_shift calculation (live evidence: 26/26 failed
    rows on the production backtest.db carry vs_spy_pct=0.0 from the
    default, not from real benchmarks). Only rows that completed —
    however briefly — carry a vs_spy_pct that should feed the bias
    estimate.

    An honest 0.0 vs_spy_pct from a completed run (extremely rare —
    SPY benchmark hard-flat) is correctly kept; an honest +0.001 from
    a thin-window completed run is kept; only the placeholder ``0.0``
    from a never-finalized OOM-reaped row is excluded. The completed_at
    check is the load-bearing distinguisher.
    """
    if row.get("completed_at") is None:
        return False
    v = row.get("vs_spy_pct")
    if v is None:
        return False
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return False
    return fv == fv  # NaN guard


def _median(values: list[float]) -> float | None:
    """Pure-Python median, None on empty. NaN-free guard at caller."""
    if not values:
        return None
    sv = sorted(values)
    n = len(sv)
    mid = n // 2
    if n % 2 == 1:
        return float(sv[mid])
    return float((sv[mid - 1] + sv[mid]) / 2.0)


def analyze(backtest_db: Path | str | None = None) -> dict:
    """Read every ``status='failed'`` row from ``backtest.db`` and classify
    each as OOM-reaped vs genuine failure. Pure, total, never raises.

    Verdict ladder (precedence-ordered, first matching wins — adverse
    states take precedence so a CLI / cron caller's exit code reflects
    the WORST observed condition):

    1. ``INSUFFICIENT_DATA``       — backtest.db missing / unreadable
    2. ``NO_FAILED_RUNS``          — every row is ``status != 'failed'``
    3. ``ALL_GENUINE_FAILURE``     — every failed row is a real crash
    4. ``MIXED_REAPED_AND_FAILURE`` — OOM-reaped share in (LOW, HIGH)
    5. ``MOSTLY_OOM_REAPED``       — OOM-reaped share ≥ HIGH

    Returns ``{verdict, hint, n_failed, n_oom_reaped, n_genuine,
    oom_reaped_pct, hidden_median_vs_spy, hidden_max_vs_spy,
    hidden_min_vs_spy, complete_median_vs_spy,
    bias_shift_pct, suspect_run_ids}``. The bias_shift_pct is the
    estimated shift in aggregate median vs_spy_pct if the hidden
    OOM-reaped rows were re-included in the dashboard's complete-only
    aggregate — negative ⇒ the dashboard overstates alpha, positive ⇒
    the dashboard understates. ``suspect_run_ids`` is the list of
    LIKELY_OOM_REAPED run_ids (capped at 50) so an operator can query
    them directly.
    """
    out: dict = {
        "verdict": "INSUFFICIENT_DATA",
        "hint": "",
        "n_failed": 0,
        "n_oom_reaped": 0,
        "n_genuine": 0,
        "oom_reaped_pct": None,
        "n_oom_with_real_vs_spy": 0,
        "hidden_median_vs_spy": None,
        "hidden_max_vs_spy": None,
        "hidden_min_vs_spy": None,
        "complete_median_vs_spy": None,
        "bias_shift_pct": None,
        "suspect_run_ids": [],
    }

    # Resolve BACKTEST_DB at call time (not at def time) — mirrors the
    # same pattern BacktestStore.__init__ uses for test isolation
    # (conftest redirects bt.BACKTEST_DB into a tmp).
    if backtest_db is None:
        try:
            from ..backtest import BACKTEST_DB
            backtest_db = BACKTEST_DB
        except Exception as exc:
            out["hint"] = (f"backtest module import failed "
                           f"({type(exc).__name__})")
            return out

    try:
        p = Path(backtest_db)
        if not p.exists():
            out["hint"] = "backtest.db not present (fresh checkout / loop never ran)"
            return out

        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=5.0)
            conn.row_factory = sqlite3.Row
            failed_rows = conn.execute(
                "SELECT run_id, n_trades, vs_spy_pct, total_return_pct, "
                "       notes, completed_at "
                "FROM backtest_runs WHERE status='failed' "
                "ORDER BY run_id ASC"
            ).fetchall()
            complete_vs_spy = conn.execute(
                "SELECT vs_spy_pct FROM backtest_runs "
                "WHERE status='complete' AND vs_spy_pct IS NOT NULL"
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            out["hint"] = f"backtest.db read failed ({type(exc).__name__})"
            return out
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

        out["n_failed"] = len(failed_rows)
        if not failed_rows:
            out["verdict"] = "NO_FAILED_RUNS"
            out["hint"] = ("no status='failed' rows in backtest.db — "
                           "engine is clean; nothing to audit")
            return out

        oom_reaped: list[dict] = []
        genuine: list[dict] = []
        for r in failed_rows:
            d = dict(r)
            verdict = _classify_failed_row(d)
            if verdict == "LIKELY_OOM_REAPED":
                oom_reaped.append(d)
            else:
                genuine.append(d)

        out["n_oom_reaped"] = len(oom_reaped)
        out["n_genuine"] = len(genuine)
        oom_pct = (len(oom_reaped) / len(failed_rows)) if failed_rows else 0.0
        out["oom_reaped_pct"] = round(oom_pct, 4)

        # Hidden-slice stats — the vs_spy_pct distribution of OOM-reaped
        # rows is what the dashboard's complete-only aggregate excludes.
        # Filter to rows whose vs_spy_pct is a real finalize_run value
        # (completed_at populated), not the schema's NOT NULL DEFAULT 0
        # placeholder. Live evidence: 26/26 production failed rows carry
        # vs_spy_pct=0.0 from the default — treating those as realized
        # alpha would silently shift bias_shift_pct toward 0 (the
        # placeholder value), badly misreporting the dashboard's true
        # overstatement.
        hidden_vs_spy: list[float] = []
        n_oom_with_real_vs_spy = 0
        for d in oom_reaped:
            if not _has_real_vs_spy(d):
                continue
            v = d.get("vs_spy_pct")
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if fv == fv:  # NaN guard
                hidden_vs_spy.append(fv)
                n_oom_with_real_vs_spy += 1
        out["n_oom_with_real_vs_spy"] = n_oom_with_real_vs_spy

        if hidden_vs_spy:
            out["hidden_median_vs_spy"] = round(_median(hidden_vs_spy), 3)
            out["hidden_max_vs_spy"] = round(max(hidden_vs_spy), 3)
            out["hidden_min_vs_spy"] = round(min(hidden_vs_spy), 3)

        # Bias-shift estimate — median(complete + hidden) vs median(complete).
        # This is the magnitude by which the dashboard's published median
        # would change if OOM-reaped runs were merged back in. A positive
        # shift means the dashboard UNDERSTATES alpha; negative means it
        # OVERSTATES. The pass #21 finding implies the dashboard overstates,
        # but this analyzer measures it empirically per-state of backtest.db.
        complete_finite = []
        for (v,) in complete_vs_spy:
            try:
                fv = float(v)
                if fv == fv:
                    complete_finite.append(fv)
            except (TypeError, ValueError):
                continue
        if complete_finite:
            out["complete_median_vs_spy"] = round(_median(complete_finite), 3)
        if complete_finite and hidden_vs_spy:
            with_hidden = _median(complete_finite + hidden_vs_spy)
            without_hidden = _median(complete_finite)
            # Both medians are defined here (both inputs non-empty), but
            # _median's contract returns Optional so re-guard.
            if with_hidden is not None and without_hidden is not None:
                out["bias_shift_pct"] = round(with_hidden - without_hidden, 3)

        # Suspect run_ids — capped at 50 so a CLI / API response is bounded.
        out["suspect_run_ids"] = [
            int(d["run_id"]) for d in oom_reaped[:50]
        ]

        # Verdict assignment.
        if not oom_reaped:
            out["verdict"] = "ALL_GENUINE_FAILURE"
            out["hint"] = (
                f"{len(failed_rows)} failed rows, 0 OOM-reaped — engine "
                "crashes are real; nothing hidden from the dashboard "
                "aggregate. Look at the run logs, not the host RSS."
            )
            return out

        if oom_pct >= OOM_REAPED_HIGH_PCT:
            out["verdict"] = "MOSTLY_OOM_REAPED"
        elif oom_pct >= OOM_REAPED_LOW_PCT:
            out["verdict"] = "MIXED_REAPED_AND_FAILURE"
        else:
            # Honest middle bucket: some hidden runs but most failures
            # are genuine. Still ALL_GENUINE_FAILURE-LIKE for the cron
            # exit-code purpose, but the dict carries the truth.
            out["verdict"] = "ALL_GENUINE_FAILURE"

        bias_str = ""
        if out["bias_shift_pct"] is not None:
            sign = "+" if out["bias_shift_pct"] >= 0 else ""
            bias_str = (f" Dashboard median vs_spy would shift "
                        f"{sign}{out['bias_shift_pct']:.2f}pp if hidden "
                        f"rows were re-included.")
        if hidden_vs_spy:
            hidden_str = (f"; their hidden median vs_spy = "
                          f"{out['hidden_median_vs_spy']} "
                          f"(n={n_oom_with_real_vs_spy} with a real "
                          f"finalize_run benchmark out of "
                          f"{len(oom_reaped)} OOM-reaped).")
        else:
            hidden_str = (f"; ALL {len(oom_reaped)} OOM-reaped rows carry "
                          f"the schema's vs_spy_pct=0.0 placeholder "
                          f"(never finalized), so the realized alpha of "
                          f"the hidden slice is UNKNOWN — bias_shift "
                          f"is not computable.")
        out["hint"] = (
            f"{len(oom_reaped)}/{len(failed_rows)} failed rows "
            f"({100 * oom_pct:.1f}%) look OOM-reaped (n_trades >= "
            f"{MIN_TRADES_FOR_REAL_RUN} or [reaped] in notes)"
            + hidden_str + bias_str
        )
        return out

    except Exception as exc:  # pragma: no cover - belt & braces
        return {
            "verdict": "INSUFFICIENT_DATA",
            "hint": f"analyze error ({type(exc).__name__})",
            "n_failed": 0,
            "n_oom_reaped": 0,
            "n_genuine": 0,
            "oom_reaped_pct": None,
            "n_oom_with_real_vs_spy": 0,
            "hidden_median_vs_spy": None,
            "hidden_max_vs_spy": None,
            "hidden_min_vs_spy": None,
            "complete_median_vs_spy": None,
            "bias_shift_pct": None,
            "suspect_run_ids": [],
        }


_ADVERSE_VERDICTS = frozenset({
    "MIXED_REAPED_AND_FAILURE", "MOSTLY_OOM_REAPED",
})


def is_failed_runs_hidden(backtest_db: Path | str | None = None) -> bool | None:
    """Convenience boolean: True ⇒ the dashboard's complete-only aggregate
    is materially biased by hidden OOM-reaped runs; False ⇒ no hidden
    slice; None ⇒ can't-tell. Mirrors ``deploy_audit.is_deploy_stale`` /
    ``scorer_pickle_smoke.is_pickle_smoke_failed``.
    """
    try:
        rep = analyze(backtest_db)
        v = rep.get("verdict")
        if v in _ADVERSE_VERDICTS:
            return True
        if v in ("NO_FAILED_RUNS", "ALL_GENUINE_FAILURE"):
            return False
        return None
    except Exception:
        return None


def _cli(argv: list[str] | None = None) -> int:
    """``python3 -m paper_trader.ml.failed_run_audit``.

    Exit code 2 on adverse verdicts (MIXED_REAPED_AND_FAILURE /
    MOSTLY_OOM_REAPED — actionable: the dashboard's complete-only
    aggregate is biased), 0 otherwise. ``--json`` for machine-readable
    output; ``--db`` to audit a non-default DB path (test helper).
    """
    import argparse
    import json
    import sys

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.failed_run_audit",
        description=("Forensic audit of status='failed' backtest rows. "
                     "Distinguishes OOM-reaped runs (carry real trade "
                     "data the dashboard hides) from genuine engine "
                     "crashes (no data). Reports the bias the hidden "
                     "slice introduces in the published median vs_spy."),
    )
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of a table.")
    p.add_argument("--db", default=None, dest="db",
                   help="Audit a non-default backtest.db path (test helper).")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    rep = analyze(args.db)
    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
        return 2 if rep.get("verdict") in _ADVERSE_VERDICTS else 0

    v = rep.get("verdict")
    print(f"[failed_run_audit] verdict: {v}")
    print(f"  {rep.get('hint', '')}")
    print(f"  n_failed={rep.get('n_failed')} "
          f"n_oom_reaped={rep.get('n_oom_reaped')} "
          f"n_genuine={rep.get('n_genuine')} "
          f"oom_reaped_pct={rep.get('oom_reaped_pct')}")
    if rep.get("hidden_median_vs_spy") is not None:
        print(f"  hidden vs_spy: median={rep.get('hidden_median_vs_spy')} "
              f"min={rep.get('hidden_min_vs_spy')} "
              f"max={rep.get('hidden_max_vs_spy')}")
    if rep.get("complete_median_vs_spy") is not None:
        print(f"  complete vs_spy median: "
              f"{rep.get('complete_median_vs_spy')}")
    if rep.get("bias_shift_pct") is not None:
        print(f"  bias_shift if hidden rows re-included: "
              f"{rep.get('bias_shift_pct'):+.2f}pp")
    if rep.get("suspect_run_ids"):
        ids = rep["suspect_run_ids"]
        if len(ids) <= 10:
            print(f"  suspect run_ids: {ids}")
        else:
            print(f"  suspect run_ids (first 10 of "
                  f"{len(ids)}): {ids[:10]}")
    return 2 if v in _ADVERSE_VERDICTS else 0


if __name__ == "__main__":
    raise SystemExit(_cli())

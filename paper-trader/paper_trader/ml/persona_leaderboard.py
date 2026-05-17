"""Per-persona strategy-quality diagnostic — which of the 10 trading
personas actually carry repeatable alpha, and which are dead weight?

This is a **read-only diagnostic**, the exact sibling of
`paper_trader/ml/calibration.py` and `paper_trader/ml/label_audit.py`: it
never trains, never touches `decision_scorer.pkl`,
`decision_outcomes.jsonl`, `build_features`, `N_FEATURES`, or any trade
path, and it opens `backtest.db` strictly `mode=ro`. It cannot perturb the
unattended continuous loop or break pickle compatibility (AGENTS.md "When
to bump model versions" / "Common pitfalls"). It does **not** prune the
`PERSONAS` dict or re-tune `_PERSONA_BOOSTS` — that is a
strategy-dynamics change requiring an explicit decision. This tool exists
to *inform* that decision with the data the engine already records but
never aggregates.

Why a quant needs this. `backtest.db` holds hundreds of `complete` runs,
each mapped to one of 10 personas by the **single source of truth**
`paper_trader.backtest.persona_for` (`((run_id-1) % 10) + 1`). The
per-cycle "best run +1294% / vs_spy +1202%" log line is the *max of a
high-variance leveraged-beta draw* (AGENTS.md "Read `vs_spy_pct`
skeptically on leveraged windows"), so the **mean** per persona is
dominated by a few 3×-ETF bull-window rips. The honest central-tendency
statistic is the **median vs_spy across that persona's runs**, plus a
win-rate (fraction of runs that beat SPY at all) and the risk shape of
the equity curve (max drawdown, annualised Sharpe-equivalent, % of time
underwater) — none of which the engine surfaces anywhere today.

The payoff is concrete and actionable: it cleanly separates a persona
with a real positive median edge from one that is a consistent SPY
*underperformer* whose `_PERSONA_BOOSTS` row is just adding variance —
the single most useful input to a future (out-of-scope here) decision to
prune or re-tune a persona.

Verdict per persona (crisp, threshold-driven so it is exactly testable):

| Verdict | Meaning |
|---------|---------|
| `INSUFFICIENT` | < ``MIN_RUNS_PER_PERSONA`` complete runs — no stable median |
| `DRAG` | median vs_spy ≤ ``DRAG_MAX_MEDIAN_VS_SPY`` — fails to beat SPY at the median; dead weight / pure variance |
| `FLAT` | positive but weak median edge, below the strong bar |
| `EDGE` | median vs_spy ≥ ``EDGE_MIN_MEDIAN_VS_SPY`` **and** win-rate ≥ ``EDGE_MIN_WIN_RATE`` |

Overall verdict: ``INSUFFICIENT_DATA`` (< ``MIN_RECORDS`` qualifying
runs), ``HAS_DRAG_PERSONA`` (≥1 persona classified ``DRAG`` — actionable),
or ``HEALTHY``.
"""
from __future__ import annotations

import json
import sqlite3
import statistics
from pathlib import Path

import numpy as np

# Single source of truth for run_id → persona. Importing (not
# reimplementing) the mapping means a future PERSONAS-dict reordering can
# never silently shift every historical aggregate — the same
# single-source-of-truth discipline `_oos_rank_metrics` uses by reusing
# `calibration._spearman` and `label_audit` uses by importing
# `PRED_CLAMP_PCT`.
from paper_trader.backtest import persona_for

# Thresholds are module-level so tests assert exact verdicts and a tuning
# change is a single reviewable edit (mirrors calibration.py / label_audit.py
# convention and the codebase's constants-at-module-scope rule).
MIN_RECORDS = 30            # minimum qualifying complete runs for any verdict
MIN_RUNS_PER_PERSONA = 5    # below this a persona's median is not stable
EDGE_MIN_MEDIAN_VS_SPY = 20.0   # strong positive median alpha bar (pp)
EDGE_MIN_WIN_RATE = 0.50        # must beat SPY in ≥half its runs to be EDGE
DRAG_MAX_MEDIAN_VS_SPY = 0.0    # median alpha ≤ this ⇒ DRAG (no edge)

# Trading days per year — annualisation factor for the Sharpe-equivalent.
# Backtests sample every trading day (`SAMPLE_EVERY_N_DAYS = 1`), so the
# equity curve is a daily series and 252 is the correct convention.
_TRADING_DAYS_PER_YEAR = 252.0


def _finite(v):
    """Parse to a finite float or return None. Mirrors calibration.py /
    label_audit.py drop-non-finite hardening (a single inf/nan must not
    poison the report — same class as ``decision_scorer._to_float``)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _equity_risk(curve) -> dict:
    """Max drawdown %, annualised Sharpe-equivalent, and %-time-underwater
    from an equity curve (list of ``{"value": ...}`` points).

    Returns ``{max_drawdown_pct, sharpe, pct_time_underwater}`` with
    ``None`` for any metric that cannot be computed (too few points, a
    flat series, non-positive values). Never raises — a malformed curve
    degrades to all-``None`` so the return aggregates still stand.
    """
    out = {"max_drawdown_pct": None, "sharpe": None,
           "pct_time_underwater": None}
    try:
        vals = []
        for p in curve or []:
            v = _finite(p.get("value") if isinstance(p, dict) else p)
            if v is not None:
                vals.append(v)
        if len(vals) < 2:
            return out

        # Max drawdown: largest peak→trough decline as a positive %.
        peak = vals[0]
        max_dd = 0.0
        underwater = 0
        for v in vals:
            if v > peak:
                peak = v
            if peak > 0:
                dd = (peak - v) / peak
                if dd > max_dd:
                    max_dd = dd
            if v < peak:
                underwater += 1
        out["max_drawdown_pct"] = round(max_dd * 100.0, 4)
        out["pct_time_underwater"] = round(underwater / len(vals) * 100.0, 4)

        # Daily simple returns; skip steps off a non-positive prior value
        # (a fully liquidated/zeroed point would otherwise blow up the ratio).
        rets = []
        for i in range(1, len(vals)):
            prev = vals[i - 1]
            if prev > 0:
                rets.append(vals[i] / prev - 1.0)
        if len(rets) >= 2:
            arr = np.asarray(rets, dtype=np.float64)
            sd = float(arr.std(ddof=1))
            # Floor, not `sd > 0`: a flat/cash-parked or constant-return
            # stretch has a std of pure float-representation noise (~1e-16
            # — `0.1` is not exactly representable, so 110/100-1 ≠ 121/110-1
            # at machine epsilon). That sails past `> 0` and divides a real
            # mean by ~1e-17, manufacturing a ~1e16 "Sharpe" that would then
            # dominate the per-persona median. Any equity curve with genuine
            # daily variance has std ≫ 1e-9; the floor cleanly separates a
            # degenerate series (→ Sharpe undefined) from a real one.
            if sd > 1e-9:
                sharpe = float(arr.mean()) / sd * np.sqrt(_TRADING_DAYS_PER_YEAR)
                if np.isfinite(sharpe):
                    out["sharpe"] = round(sharpe, 4)
    except Exception:
        return {"max_drawdown_pct": None, "sharpe": None,
                "pct_time_underwater": None}
    return out


def _median_opt(xs):
    """Median of the finite values, or None if there are none."""
    clean = [x for x in xs if x is not None]
    return round(float(statistics.median(clean)), 4) if clean else None


def persona_leaderboard(runs) -> dict:
    """Aggregate ``complete`` backtest runs by trading persona.

    ``runs`` is any iterable of dicts. Recognised keys:
    ``run_id`` (int, required — maps to a persona), ``vs_spy_pct``
    (required — the alpha metric), ``total_return_pct`` (optional),
    ``status`` (optional — non-``"complete"`` rows are ignored so the
    raw ``backtest_runs`` table can be passed straight through),
    ``equity_curve`` (optional list of ``{"value": ...}`` points for the
    risk metrics).

    Returns a JSON-safe dict:
    ``{status, verdict, n_runs, n_personas, leaderboard:[{persona, n,
       median_vs_spy, mean_vs_spy, median_return, win_rate, median_sharpe,
       median_max_drawdown_pct, median_pct_time_underwater, verdict}],
       drag_personas:[...], hint}``. The ``leaderboard`` is sorted by
    ``median_vs_spy`` descending (``INSUFFICIENT`` personas last).
    """
    buckets: dict[str, dict] = {}
    n_qualifying = 0

    for r in runs:
        if str(r.get("status") or "complete").lower() != "complete":
            continue
        vs = _finite(r.get("vs_spy_pct"))
        if vs is None:
            continue
        rid = r.get("run_id")
        try:
            persona = persona_for(int(rid))["name"]
        except Exception:
            continue
        n_qualifying += 1
        b = buckets.setdefault(
            persona,
            {"vs": [], "ret": [], "sharpe": [], "mdd": [], "uw": []},
        )
        b["vs"].append(vs)
        tr = _finite(r.get("total_return_pct"))
        if tr is not None:
            b["ret"].append(tr)
        risk = _equity_risk(r.get("equity_curve"))
        if risk["sharpe"] is not None:
            b["sharpe"].append(risk["sharpe"])
        if risk["max_drawdown_pct"] is not None:
            b["mdd"].append(risk["max_drawdown_pct"])
        if risk["pct_time_underwater"] is not None:
            b["uw"].append(risk["pct_time_underwater"])

    if n_qualifying < MIN_RECORDS:
        return {
            "status": "insufficient_data",
            "verdict": "INSUFFICIENT_DATA",
            "n_runs": n_qualifying,
            "n_personas": len(buckets),
            "leaderboard": [],
            "drag_personas": [],
            "hint": (f"need ≥{MIN_RECORDS} complete runs with a non-null "
                     f"vs_spy_pct, have {n_qualifying}"),
        }

    leaderboard = []
    drag_personas = []
    for persona, b in buckets.items():
        n = len(b["vs"])
        med_vs = round(float(statistics.median(b["vs"])), 4)
        mean_vs = round(float(statistics.fmean(b["vs"])), 4)
        win_rate = round(sum(1 for v in b["vs"] if v > 0) / n, 4)
        if n < MIN_RUNS_PER_PERSONA:
            verdict = "INSUFFICIENT"
        elif med_vs <= DRAG_MAX_MEDIAN_VS_SPY:
            verdict = "DRAG"
            drag_personas.append(persona)
        elif med_vs >= EDGE_MIN_MEDIAN_VS_SPY and win_rate >= EDGE_MIN_WIN_RATE:
            verdict = "EDGE"
        else:
            verdict = "FLAT"
        leaderboard.append({
            "persona": persona,
            "n": n,
            "median_vs_spy": med_vs,
            "mean_vs_spy": mean_vs,
            "median_return": _median_opt(b["ret"]),
            "win_rate": win_rate,
            "median_sharpe": _median_opt(b["sharpe"]),
            "median_max_drawdown_pct": _median_opt(b["mdd"]),
            "median_pct_time_underwater": _median_opt(b["uw"]),
            "verdict": verdict,
        })

    # Sort by median_vs_spy desc; INSUFFICIENT personas always sink last
    # regardless of their (unstable, small-n) median.
    leaderboard.sort(
        key=lambda d: (d["verdict"] != "INSUFFICIENT", d["median_vs_spy"]),
        reverse=True,
    )

    if drag_personas:
        verdict = "HAS_DRAG_PERSONA"
        hint = (f"{len(drag_personas)} persona(s) have a median vs_spy ≤ "
                f"{DRAG_MAX_MEDIAN_VS_SPY:.0f}pp across ≥{MIN_RUNS_PER_PERSONA} "
                f"runs — they do not beat SPY at the median and are adding "
                f"variance, not alpha: {', '.join(sorted(drag_personas))}. "
                f"This is the data for a (separate, explicit) decision to "
                f"prune or re-tune their _PERSONA_BOOSTS row — do NOT change "
                f"PERSONAS/_PERSONA_BOOSTS from this read-only audit.")
    else:
        verdict = "HEALTHY"
        hint = ("every persona with a stable sample has a positive median "
                "alpha vs SPY")

    return {
        "status": "ok",
        "verdict": verdict,
        "n_runs": n_qualifying,
        "n_personas": len(buckets),
        "leaderboard": leaderboard,
        "drag_personas": sorted(drag_personas),
        "hint": hint,
    }


def _load_runs(db_path: Path) -> list[dict]:
    """Read ``complete`` rows from ``backtest.db`` strictly read-only.

    Parses ``equity_curve_json`` per row; a corrupt blob degrades that
    row's risk metrics to ``None`` (the curve is dropped) without losing
    its return aggregates — mirrors the loop's per-line parse hardening.
    """
    runs: list[dict] = []
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10.0)
    try:
        rows = conn.execute(
            "SELECT run_id, total_return_pct, vs_spy_pct, status, "
            "equity_curve_json FROM backtest_runs WHERE status='complete'"
        ).fetchall()
    finally:
        try:
            conn.close()
        except Exception:
            pass
    for run_id, tr, vs, status, eq_json in rows:
        try:
            curve = json.loads(eq_json or "[]")
        except Exception:
            curve = []
        runs.append({
            "run_id": run_id,
            "total_return_pct": tr,
            "vs_spy_pct": vs,
            "status": status,
            "equity_curve": curve,
        })
    return runs


def _cli() -> int:
    """`python3 -m paper_trader.ml.persona_leaderboard` — per-persona
    strategy-quality leaderboard over the live ``backtest.db``. Read-only;
    never writes anything. Exit 0 healthy/insufficient, 2 if any persona
    is a DRAG (so an operator/cron can branch on it, exactly like
    calibration._cli / label_audit._cli)."""
    from paper_trader.backtest import BACKTEST_DB

    db = Path(BACKTEST_DB)
    if not db.exists():
        print(f"[persona_leaderboard] no backtest.db at {db}")
        return 1
    runs = _load_runs(db)
    rep = persona_leaderboard(runs)
    print(f"complete_runs={rep['n_runs']}  personas={rep['n_personas']}")
    print(f"VERDICT: {rep['verdict']}  ({rep['hint']})")
    if rep["leaderboard"]:
        print(f"  {'persona':<34} {'n':>3} {'med_vs':>8} {'mean_vs':>9} "
              f"{'win%':>6} {'med_ret':>9} {'sharpe':>7} {'maxDD%':>7} "
              f"{'uw%':>6}  verdict")
        for e in rep["leaderboard"]:
            def _f(v, w=8, p=1):
                return f"{'n/a':>{w}}" if v is None else f"{v:>{w}.{p}f}"
            print(f"  {e['persona']:<34} {e['n']:>3} "
                  f"{_f(e['median_vs_spy'])} {_f(e['mean_vs_spy'],9)} "
                  f"{e['win_rate']*100:>5.0f}% {_f(e['median_return'],9)} "
                  f"{_f(e['median_sharpe'],7,2)} "
                  f"{_f(e['median_max_drawdown_pct'],7)} "
                  f"{_f(e['median_pct_time_underwater'],5,0)}%  "
                  f"{e['verdict']}")
    return 0 if rep["verdict"] in ("HEALTHY", "INSUFFICIENT_DATA") else 2


if __name__ == "__main__":
    raise SystemExit(_cli())

"""Exit-disposition skill — characterizes how the backtest engine's SL/TP
defaults and the ``_ml_decide`` manual-SELL path actually close positions.

This is a **read-only diagnostic**. It never trains, never touches
``decision_scorer.pkl`` / ``decision_outcomes.jsonl`` / ``build_features`` /
``N_FEATURES`` / any trade path — same operational discipline as
``gate_audit`` / ``gate_pnl`` / ``gate_realized`` / ``bubble_gate_skill`` /
``calibration`` / ``skill_trend``. Safe to run against the live unattended
loop (read against an open `?mode=ro` SQLite handle).

**The gap this fills.** ``_ml_decide`` sets ``stop_loss = price * 0.92``
and ``take_profit = price * 1.15`` on EVERY BUY (commit history pins both
defaults). ``_enforce_risk_exits`` then closes the position at the SL/TP
price the day the close crosses it. So every backtest position closes one
of four ways:

  * **stop_loss**  — the −8% SL fired (``reason LIKE 'stop-loss%'``)
  * **take_profit** — the +15% TP fired (``reason LIKE 'take-profit%'``)
  * **manual_sell** — ``_ml_decide`` issued a SELL that filled before
                     either SL or TP triggered
  * **open**       — neither gate nor manual SELL fired before run end;
                     the position is held at the final close

Across the live ``backtest.db`` (live counts: ~210K SELLs, ~55K SL, ~30K
TP, ~125K manual SELL), these four disposition classes cover the entire
exit distribution. Yet *no existing diagnostic* in ``paper_trader/ml/``
characterizes their **rate** or their **realized-return** profile.

**Why this is not** ``gate_realized`` / ``gate_pnl`` /
``conviction_calibration``. Those audit the scorer-prediction conviction
gate (the ±10/±5/0 arms acting on ``gate_scorer_pred``). This module
audits a SEPARATE structural exit mechanism — the ±8%/±15% SL/TP defaults
the engine wires into every BUY — that fires independent of the scorer
gate. A reading quant needs both signals.

**What this measures.**

  * ``fire_rate``  — per-class share of total exits, computed over rows
    with ``action='SELL'`` in ``backtest_trades``. Open positions (BUYs
    with no matching SELL) cannot be inferred from a SELL-row scan alone
    and are reported separately when a paired ``BacktestStore`` is
    supplied — the SELL-row scan is the always-available default.
  * ``exit_price_pct_band``  — the realized exit price relative to the
    BUY's avg_cost, walked FIFO over ``backtest_trades``. By
    construction SL fires near −8% and TP fires near +15%; large
    deviations are diagnostic (gap-downs through SL, after-hours through
    TP).
  * ``mean_exit_pct``  — the per-disposition mean realized exit (signed
    %). For SL/TP the mean is structurally pinned; for ``manual_sell``
    it is the only freely-varying quantity and tells a reading quant
    whether ``_ml_decide``'s SELL branch closes winners or losers on
    average.

**Honest limitations.**

  * **FIFO walk only.** ``backtest_trades`` is a flat log — a BUY does
    not carry forward avg_cost into a partial SELL. The pairing
    reconstructs ``avg_cost`` by walking trades in chronological order,
    blending into an accumulator on each BUY (mirroring ``_buy``'s
    blended-cost logic in ``backtest.py``). A partial SELL pulls qty
    off the accumulator at the blended avg_cost; a sell larger than
    held qty drops the overflow (the engine's ``min(qty, pos['qty'])``
    semantics in ``_sell``).
  * **Survivorship bias inside the corpus.** A bubble-gate or no-cash
    BLOCK never reaches ``backtest_trades``; the disposition
    distribution describes only realized fills. The pre-trade
    counterfactual is the domain of permutation tests, not this tool.
  * **Run-end open positions are not in the SELL-row scan.** The
    ``open`` bucket is reported only when a ``BacktestStore`` is
    passed via ``analyze_store`` — that path walks both BUY and SELL
    rows and reports the residual qty per ticker.

CLI:

```bash
cd /home/zeph/trading-intelligence/paper-trader
python3 -m paper_trader.ml.exit_disposition_skill              # table verdict
python3 -m paper_trader.ml.exit_disposition_skill --json       # machine-readable
python3 -m paper_trader.ml.exit_disposition_skill --runs 100   # last 100 runs only
```

Exit codes mirror the verdict-locked sibling pattern (``bubble_gate_skill`` /
``conviction_calibration``): 0 on every acceptable verdict, 2 on
``MANUAL_SELL_LOSING`` (the actionable signal — ``_ml_decide``'s SELL branch
closes losers on average, i.e. the engine is fighting its own gate).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np


# Verdict thresholds — module-level so tests assert exact verdicts and a
# tuning change is a single, reviewable edit (== gate_audit / gate_realized).
MIN_TOTAL = 30           # need a real sample before any verdict
MIN_CLASS_N = 5          # min trades in EACH compared class
MANUAL_EDGE_TOL_PP = 1.0  # |manual_sell mean − 0| band that reads as noise

# Stop-loss / take-profit defaults the engine wires into every BUY. Single
# source of truth — if `_ml_decide` ever retunes these, the constants here
# are the audit's anchor. Imported from `backtest.py` is the cleaner choice
# but creates an import-cycle risk in the analytics module path; mirroring
# the documented values matches the `bubble_gate_skill` precedent.
SL_DEFAULT_PCT = -8.0   # _ml_decide: stop_loss = price * 0.92  → −8%
TP_DEFAULT_PCT = 15.0   # _ml_decide: take_profit = price * 1.15 → +15%


def _f(v):
    """Finite float or None — the ``gate_realized._f`` / ``bubble_gate_skill._f``
    hardening class. Locally defined so this module imports nothing from
    decision_scorer (no pickle path needed for an exit-trade analyzer)."""
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _classify_reason(reason: str | None) -> str:
    """Classify a SELL row's ``reason`` text into one of three disposition
    classes. The ``reason`` column is free-text written by ``_enforce_risk_exits``
    (``"stop-loss @ {sl} (close {px})"`` / ``"take-profit @ {tp} (close {px})"``)
    or by ``_execute_decision`` (the ``_ml_decide`` reasoning string, prefix
    typically ``"ML+quant: …"``). The two engine-generated prefixes are
    stable test-locked contracts; any other text is a manual SELL.
    """
    if not reason:
        return "manual_sell"
    r = str(reason)
    if r.startswith("stop-loss"):
        return "stop_loss"
    if r.startswith("take-profit"):
        return "take_profit"
    return "manual_sell"


def _pair_trades_fifo(trades: list[dict]) -> list[dict]:
    """Walk trades chronologically, blending BUY costs and matching SELLs
    FIFO-by-ticker per (run_id, ticker).

    Mirrors ``backtest.py::_buy``'s blended-avg-cost accumulation and
    ``backtest.py::_sell``'s ``min(qty, held)`` clipping. Returns one pair
    record per SELL row:

      ``{run_id, ticker, sim_date, sell_qty, sell_price, avg_cost,
         exit_pct, disposition, reason}``

    Where ``exit_pct = (sell_price − avg_cost) / avg_cost * 100``. A SELL
    against zero held qty (engine-impossible but defensively handled) drops
    that row; the same idempotent ``backtest.py::_sell`` semantics.

    The input list MUST be pre-sorted in (run_id, sim_date, id) order to
    match the engine's chronological execution. ``analyze_db`` pre-sorts;
    callers passing hand-crafted input are responsible for the same.
    """
    # Per (run_id, ticker) position book: {(run_id, ticker): {qty, avg_cost}}.
    book: dict[tuple, dict[str, float]] = {}
    pairs: list[dict] = []

    for t in trades:
        if not isinstance(t, dict):
            continue
        action = str(t.get("action") or "").upper()
        run_id = t.get("run_id")
        ticker = str(t.get("ticker") or "")
        qty = _f(t.get("qty"))
        price = _f(t.get("price"))
        if run_id is None or not ticker or qty is None or price is None:
            continue
        if qty <= 0 or price <= 0:
            continue

        key = (run_id, ticker)

        if action == "BUY":
            pos = book.get(key)
            if pos is None:
                book[key] = {"qty": qty, "avg_cost": price}
            else:
                new_qty = pos["qty"] + qty
                if new_qty <= 0:
                    # Numerically impossible from a positive-qty add, but
                    # defense-in-depth against a malformed input row.
                    continue
                blended = (pos["qty"] * pos["avg_cost"] + qty * price) / new_qty
                pos["qty"] = new_qty
                pos["avg_cost"] = blended
            continue

        if action != "SELL":
            continue

        pos = book.get(key)
        if pos is None or pos["qty"] <= 0:
            # SELL without a corresponding open position — engine never
            # emits this (``_sell`` early-returns on no pos), but a
            # malformed/sliced input could. Drop the row.
            continue

        sell_qty = min(qty, pos["qty"])
        avg_cost = pos["avg_cost"]
        exit_pct = (price - avg_cost) / avg_cost * 100.0 if avg_cost > 0 else 0.0
        disposition = _classify_reason(t.get("reason"))

        pairs.append({
            "run_id": run_id,
            "ticker": ticker,
            "sim_date": t.get("sim_date") or "",
            "sell_qty": round(sell_qty, 6),
            "sell_price": round(price, 4),
            "avg_cost": round(avg_cost, 4),
            "exit_pct": round(exit_pct, 4),
            "disposition": disposition,
            "reason": str(t.get("reason") or "")[:200],
        })

        # Decrement held qty (idempotent on overflow — same as ``_sell``).
        pos["qty"] -= sell_qty
        if pos["qty"] <= 1e-6:
            book.pop(key, None)

    return pairs


def _stats(vals: list[float]) -> dict:
    """Per-class summary stats. Mirrors ``bubble_gate_skill._stats`` shape
    for cross-diagnostic readability — every reader of these tools sees
    the same n/mean/min/max contract."""
    if not vals:
        return {"n": 0, "mean": None, "lo": None, "hi": None, "median": None}
    a = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": round(float(a.mean()), 4),
        "median": round(float(np.median(a)), 4),
        "lo": round(float(a.min()), 4),
        "hi": round(float(a.max()), 4),
    }


def exit_disposition_report(pairs: list[dict]) -> dict:
    """Bucket paired-trade exit records by disposition class, report
    fire-rate and realized exit-% stats per class, and emit a verdict
    grading the manual-SELL branch.

    ``pairs`` — any iterable of records from ``_pair_trades_fifo`` (or
    hand-crafted dicts matching that schema for tests). Required keys per
    record: ``disposition``, ``exit_pct``.

    Verdict ladder (test-locked, exact-value):

    | Verdict | Trigger |
    |---------|---------|
    | ``INSUFFICIENT_DATA`` | n_total < MIN_TOTAL OR manual_sell n < MIN_CLASS_N |
    | ``MANUAL_SELL_LOSING`` | manual_sell mean_exit_pct < −MANUAL_EDGE_TOL_PP — the engine's manual SELL branch closes net losers (actionable; gate's own signal is anti-edge) |
    | ``MANUAL_SELL_FLAT`` | |manual_sell mean_exit_pct| ≤ MANUAL_EDGE_TOL_PP — manual SELL adds variance without realized edge |
    | ``MANUAL_SELL_WINNING`` | manual_sell mean_exit_pct > +MANUAL_EDGE_TOL_PP — the engine is closing winners (potentially leaving alpha on the table; cross-check vs SL/TP rates) |

    The verdict grades ONLY the manual-SELL class because SL/TP exit
    means are structurally pinned to ±8%/+15% by ``_ml_decide``'s
    defaults. A meaningful gap in SL realized vs −8% is captured in the
    output ``sl_default_residual_pp`` field (gap-down slippage), but is
    informational, not verdict-driving.

    Returns a JSON-safe dict. Never raises.
    """
    out: dict = {
        "status": "ok",
        "verdict": "INSUFFICIENT_DATA",
        "measurement": "exit_reason_classified_no_reprediction",
        "n_total": 0,
        "fire_rate": {},
        "classes": [],
        "manual_sell_mean_pct": None,
        "sl_default_residual_pp": None,
        "tp_default_residual_pp": None,
        "hint": "",
    }

    try:
        it = list(pairs or [])
    except Exception:
        it = []

    by_class: dict[str, list[float]] = {
        "stop_loss": [], "take_profit": [], "manual_sell": [],
    }
    for r in it:
        if not isinstance(r, dict):
            continue
        cls = str(r.get("disposition") or "").lower()
        if cls not in by_class:
            continue
        ep = _f(r.get("exit_pct"))
        if ep is None:
            continue
        by_class[cls].append(ep)

    n_total = sum(len(v) for v in by_class.values())
    out["n_total"] = n_total

    # Per-class stats and fire-rate.
    classes_out = []
    for cls in ("stop_loss", "take_profit", "manual_sell"):
        s = _stats(by_class[cls])
        share = round(s["n"] / n_total, 4) if n_total else 0.0
        classes_out.append({
            "class": cls,
            "n": s["n"],
            "share": share,
            "mean_exit_pct": s["mean"],
            "median_exit_pct": s["median"],
            "lo_exit_pct": s["lo"],
            "hi_exit_pct": s["hi"],
        })
        out["fire_rate"][cls] = share
    out["classes"] = classes_out

    # Default-residual pp — how far SL/TP fires deviate from the engine
    # default by-construction. A large SL residual = gap-down slippage
    # through the −8% trigger; a large TP residual = after-hours past +15%.
    if by_class["stop_loss"]:
        sl_mean = float(np.mean(by_class["stop_loss"]))
        out["sl_default_residual_pp"] = round(sl_mean - SL_DEFAULT_PCT, 4)
    if by_class["take_profit"]:
        tp_mean = float(np.mean(by_class["take_profit"]))
        out["tp_default_residual_pp"] = round(tp_mean - TP_DEFAULT_PCT, 4)

    # Verdict on the manual_sell branch.
    if n_total < MIN_TOTAL or len(by_class["manual_sell"]) < MIN_CLASS_N:
        out["verdict"] = "INSUFFICIENT_DATA"
        out["hint"] = (
            f"need ≥{MIN_TOTAL} paired SELLs AND ≥{MIN_CLASS_N} manual_sell "
            f"rows; have n_total={n_total}, "
            f"manual_sell={len(by_class['manual_sell'])}"
        )
        return out

    manual_mean = float(np.mean(by_class["manual_sell"]))
    out["manual_sell_mean_pct"] = round(manual_mean, 4)

    if manual_mean < -MANUAL_EDGE_TOL_PP:
        out["verdict"] = "MANUAL_SELL_LOSING"
        out["hint"] = (
            f"manual_sell mean realized exit {manual_mean:+.2f}% < "
            f"−{MANUAL_EDGE_TOL_PP:.1f}pp — the engine's _ml_decide SELL "
            f"branch is closing net losers on average (the gate is fighting "
            f"its own signal; cross-check vs SL fire-rate)"
        )
    elif abs(manual_mean) <= MANUAL_EDGE_TOL_PP:
        out["verdict"] = "MANUAL_SELL_FLAT"
        out["hint"] = (
            f"manual_sell mean {manual_mean:+.2f}% within ±"
            f"{MANUAL_EDGE_TOL_PP:.1f}pp — the manual-SELL branch adds "
            f"turnover-variance without a realized edge"
        )
    else:
        out["verdict"] = "MANUAL_SELL_WINNING"
        out["hint"] = (
            f"manual_sell mean {manual_mean:+.2f}% > +{MANUAL_EDGE_TOL_PP:.1f}pp "
            f"— the engine is closing winners; verify SL/TP fire-rates are "
            f"not crowding out longer holds"
        )
    return out


def analyze_db(db_path: Path | str | None = None,
               run_id_limit: int | None = None) -> dict:
    """Load ``backtest_trades`` from a ``backtest.db`` and emit the
    exit-disposition report.

    ``db_path`` — defaults to ``BACKTEST_DB`` resolved at call time (so
    conftest's ``monkeypatch.setattr(bt, 'BACKTEST_DB', tmp)`` is
    honoured). Read-only (`?mode=ro`); never writes.

    ``run_id_limit`` — when set, restrict the audit to the most-recent
    N run_ids. Useful to bound the dataset for the live operational
    answer ("disposition over the last K completed runs"). None reads
    all runs (sane default for a one-off audit).

    Pure best-effort: a missing DB or query failure yields a status='error'
    dict, never an exception.
    """
    out: dict = {
        "status": "error",
        "verdict": "INSUFFICIENT_DATA",
        "measurement": "exit_reason_classified_no_reprediction",
        "n_total": 0,
        "fire_rate": {},
        "classes": [],
        "manual_sell_mean_pct": None,
        "sl_default_residual_pp": None,
        "tp_default_residual_pp": None,
        "n_runs_scanned": 0,
        "hint": "",
    }

    # Resolve at call time per the AGENTS.md "call-time resolution" rule —
    # the same discipline ``BacktestStore.__init__`` uses so test fixtures
    # can redirect via ``monkeypatch.setattr(bt, 'BACKTEST_DB', tmp)``.
    if db_path is None:
        try:
            from paper_trader.backtest import BACKTEST_DB as _BT_DB
            db_path = _BT_DB
        except Exception as e:
            out["hint"] = f"BACKTEST_DB import failed: {type(e).__name__}: {e}"
            return out

    p = Path(db_path)
    if not p.exists():
        out["hint"] = f"no backtest.db at {p}"
        return out

    conn = None
    try:
        conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=10.0)
        conn.row_factory = sqlite3.Row

        # Optional run_id_limit: derive the cutoff run_id from a single
        # ORDER BY ... LIMIT scan, then filter on that.
        cutoff_run_id: int | None = None
        if run_id_limit and run_id_limit > 0:
            try:
                row = conn.execute(
                    "SELECT MIN(run_id) FROM ("
                    "  SELECT run_id FROM backtest_runs "
                    "  ORDER BY run_id DESC LIMIT ?"
                    ")",
                    (int(run_id_limit),),
                ).fetchone()
                if row and row[0] is not None:
                    cutoff_run_id = int(row[0])
            except Exception:
                # Older schemas / a missing backtest_runs table — fall
                # through and audit all trades.
                cutoff_run_id = None

        if cutoff_run_id is not None:
            rows = conn.execute(
                "SELECT run_id, sim_date, ticker, action, qty, price, reason, id "
                "FROM backtest_trades WHERE run_id >= ? "
                "ORDER BY run_id ASC, sim_date ASC, id ASC",
                (cutoff_run_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT run_id, sim_date, ticker, action, qty, price, reason, id "
                "FROM backtest_trades "
                "ORDER BY run_id ASC, sim_date ASC, id ASC"
            ).fetchall()

        trades = [
            {"run_id": r["run_id"], "sim_date": r["sim_date"],
             "ticker": r["ticker"], "action": r["action"],
             "qty": r["qty"], "price": r["price"], "reason": r["reason"]}
            for r in rows
        ]
        n_runs = len({t["run_id"] for t in trades if t["run_id"] is not None})

        pairs = _pair_trades_fifo(trades)
        rep = exit_disposition_report(pairs)
        rep["n_runs_scanned"] = n_runs
        return rep
    except Exception as e:  # pragma: no cover — defensive
        out["hint"] = f"db read failed: {type(e).__name__}: {e}"
        return out
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _cli(argv: list[str] | None = None) -> int:
    """``python3 -m paper_trader.ml.exit_disposition_skill`` — characterize
    backtest exit-disposition distribution + verdict on the manual-SELL
    branch's realized edge.

    Read-only. Exits 2 on ``MANUAL_SELL_LOSING`` (the actionable signal —
    cron-branchable just like ``bubble_gate_skill`` exits 2 on
    ``BUBBLE_GATE_HARMFUL`` and ``conviction_calibration`` exits 2 on
    ``INVERTED``).
    """
    import argparse
    import sys

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.exit_disposition_skill",
        description=(
            "Per-exit-class fire-rate + realized exit-% audit over "
            "backtest_trades. SL=−8%% defaults / TP=+15%% defaults / "
            "manual SELL via _ml_decide. Verdict grades the manual-SELL "
            "branch only — SL/TP are structurally pinned by their default "
            "thresholds."
        ),
    )
    p.add_argument("--runs", type=int, default=None,
                   help="Restrict audit to the most recent N run_ids "
                        "(default: all).")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of a table.")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    rep = analyze_db(db_path=None, run_id_limit=args.runs)

    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
        return 2 if rep.get("verdict") == "MANUAL_SELL_LOSING" else 0

    print(f"VERDICT: {rep['verdict']}  ({rep.get('hint', '')})")
    print(f"  n_total={rep.get('n_total')}  "
          f"n_runs_scanned={rep.get('n_runs_scanned')}  "
          f"manual_sell_mean={rep.get('manual_sell_mean_pct')}%")
    print(f"  sl_default_residual={rep.get('sl_default_residual_pp')}pp  "
          f"tp_default_residual={rep.get('tp_default_residual_pp')}pp")
    print()
    print(f"  {'class':<14}{'n':>8}{'share':>9}{'mean':>10}"
          f"{'median':>10}{'min':>10}{'max':>10}")
    for c in rep.get("classes", []):
        n = c["n"]
        share_s = f"{c['share']*100:.1f}%"
        m = c["mean_exit_pct"]
        med = c["median_exit_pct"]
        lo = c["lo_exit_pct"]
        hi = c["hi_exit_pct"]
        m_s = f"{m:+.2f}%" if m is not None else "  n/a"
        med_s = f"{med:+.2f}%" if med is not None else "  n/a"
        lo_s = f"{lo:+.2f}%" if lo is not None else "  n/a"
        hi_s = f"{hi:+.2f}%" if hi is not None else "  n/a"
        print(f"  {c['class']:<14}{n:>8}{share_s:>9}{m_s:>10}"
              f"{med_s:>10}{lo_s:>10}{hi_s:>10}")
    return 2 if rep.get("verdict") == "MANUAL_SELL_LOSING" else 0


if __name__ == "__main__":
    raise SystemExit(_cli())

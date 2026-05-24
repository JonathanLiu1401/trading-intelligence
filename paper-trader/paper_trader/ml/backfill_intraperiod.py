"""Backfill ``forward_intraperiod_min_5d`` / ``forward_intraperiod_max_5d``
into the historical ``decision_outcomes.jsonl`` corpus.

**Why this module exists.** The 2026-05-23 ``_compute_decision_outcomes``
feature persists per-row intraperiod-extreme columns (``run_continuous_backtests.py``
line 904, written by ``_fwd_intraperiod_extremes``). Two analyzers consume
them — ``paper_trader.ml.stop_out_audit`` (does the inherited -8% stop save
or hurt?) and ``paper_trader.ml.mfe_conversion`` (does the +15% take-profit
band capture or forfeit upside?) — plus the matching per-cycle ledgers
``_append_stop_out_skill_log`` / ``_append_mfe_skill_log`` (pass #30,
2026-05-24).

The current state on disk:

    grep -c forward_intraperiod_min_5d data/decision_outcomes.jsonl
    # → 0

All 8.7k historical rows pre-date the feature. Both analyzers and both
ledgers therefore return ``INSUFFICIENT_DATA`` / ``stop_dark=True`` /
``tp_dark=True`` on every cycle, and will continue to until the continuous
loop runs enough new cycles to dominate the rolling 5000-row trainer tail.
At the loop's documented <1 cycle/day cadence (AGENTS.md pass #30 finding
#2) that takes months — and the loop is currently DEAD (no
``run_continuous_backtests.py`` process, ledgers untouched since
2026-05-22). A skeptical quant who wants the stop-out / MFE verdicts
TODAY has no path forward without this backfill.

**Method.** Pure offline, no network, no yfinance, no DB. Reads the
existing per-window price caches under ``data/backtest_cache/prices_*.json``
(each one a ``{_meta, TICKER: {date_iso: close, …}, …}`` dict) and unions
them per-ticker. For each outcome row missing intraperiod fields we
replicate ``_fwd_intraperiod_extremes`` semantics exactly:

  * Resolve ``sim_d``'s close via up-to-7-day walk-back (mirrors
    ``PriceCache.price_on``).
  * Resolve the next 5 trading days from a global trading-day list built
    from the union of SPY's keys across all loaded caches (mirrors the
    densest-series fallback in ``PriceCache._build_trading_days``).
  * For each forward day k in 1..5: read the ticker's close with walk-back
    and reject if its walk-back collides with or precedes ``sim_d``'s
    resolved close (the documented honesty guard — a collision silently
    fabricates a flat 0% peak/min).
  * Take ``min_pct`` / ``max_pct`` across all surviving days. Partial
    coverage is honored: a single resolving day populates both extremes
    (just like ``_fwd_intraperiod_extremes`` does).

Rows whose ``sim_date`` falls outside every loaded cache window stay
unmodified (``status='no_price_cache'``). Rows that already carry finite
intraperiod values are left alone (no overwrite, ever).

**Atomicity.** The output rewrite mirrors the ``decision_outcomes.jsonl``
trim idiom used by ``run_continuous_backtests.py`` itself: write tmp,
then ``Path.replace``. A process kill mid-rewrite leaves the original
intact. Before rewrite we re-check the input file's mtime — if it changed
since we started reading, abort with a clear error rather than clobber the
appended rows (defensive against a re-awoken continuous loop).

**Read-only operational discipline.** Never touches ``decision_scorer.pkl``,
the trainer, the live trader, the dashboard, or the digital-intern
articles DB. Stop-out / MFE ledger consumers pick up the backfilled rows
on their next cycle automatically — the file schema is unchanged, only
field coverage improves.

```bash
# Preview what would be backfilled (no writes):
python3 -m paper_trader.ml.backfill_intraperiod --dry-run

# Apply the backfill in place:
python3 -m paper_trader.ml.backfill_intraperiod

# Custom paths (useful for tests / scratch corpora):
python3 -m paper_trader.ml.backfill_intraperiod \
    --outcomes /tmp/decision_outcomes.jsonl \
    --cache-dir /tmp/backtest_cache
```
"""
from __future__ import annotations

import bisect
import json
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable


# Mirror `_fwd_intraperiod_extremes`'s `h=5` default — the 5-trading-day
# forward window every other intraperiod consumer (stop_out_audit,
# mfe_conversion, the two skill ledgers) assumes by construction.
HORIZON_DAYS = 5

# Walk-back cap in calendar days — must match `PriceCache.price_on` /
# `resolved_close_date` exactly, or backfilled rows would describe a
# different window than the engine-computed ones already in the corpus.
WALK_BACK_DAYS = 7


def _is_finite(v) -> bool:
    """True iff v parses to a finite float (mirrors decision_scorer
    `_to_float` discipline: rejects None / NaN / inf / non-numeric)."""
    if v is None or isinstance(v, bool):
        return False
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


def _iter_rows(path: Path) -> Iterable[dict]:
    """Stream one JSON record per line, silently dropping unparseable rows.

    Same line-tolerant loader pattern stop_out_audit / mfe_conversion /
    `_compute_decision_outcomes` / `_inject_and_train` already use — a
    single corrupt line must not abort the backfill.
    """
    with path.open("r") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                yield json.loads(ln)
            except json.JSONDecodeError:
                continue


def _walk_back_close(series: dict, target: date) -> tuple[date, float] | None:
    """Resolve ``target`` to (date, close) via up-to-WALK_BACK_DAYS walk-back.

    Mirrors ``PriceCache.price_on`` + ``resolved_close_date`` semantics
    exactly. The two MUST stay in lockstep with the engine's computation
    or backfilled rows would describe a different window than the
    engine-produced ones. Returns None when no close in the window.
    """
    iso = target.isoformat()
    if iso in series:
        return target, float(series[iso])
    for delta in range(1, WALK_BACK_DAYS + 1):
        prior = target - timedelta(days=delta)
        if prior.isoformat() in series:
            return prior, float(series[prior.isoformat()])
    return None


def load_price_caches(cache_dir: Path) -> dict[str, dict[str, float]]:
    """Union every ``prices_YYYY-MM-DD_YYYY-MM-DD.json`` cache file under
    ``cache_dir`` into one ticker → {iso_date: close} map.

    Caches with overlapping windows simply merge — the same (ticker, date)
    yields the same close across windows by yfinance contract, so
    last-writer-wins is correct. Caches that fail to parse (torn file /
    type-wrong payload) are skipped silently like every sibling loader
    does. Returns an empty dict when no caches are present (the caller's
    error-message envelope), never raises.
    """
    out: dict[str, dict[str, float]] = {}
    if not cache_dir.exists():
        return out
    for path in sorted(cache_dir.glob("prices_*.json")):
        try:
            raw = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(raw, dict):
            continue
        for ticker, series in raw.items():
            if ticker.startswith("_"):
                continue  # _meta and any future underscore-prefixed keys
            if not isinstance(series, dict):
                continue
            bucket = out.setdefault(ticker, {})
            for d_iso, close in series.items():
                # The cache format is {date_iso: close_float}. Accept ints
                # too — yfinance occasionally returns whole-cent values
                # serialized as ints by some JSON encoders.
                if not isinstance(d_iso, str) or len(d_iso) < 10:
                    continue
                try:
                    bucket[d_iso] = float(close)
                except (TypeError, ValueError):
                    continue
    return out


def build_trading_days(prices: dict[str, dict[str, float]]) -> list[date]:
    """Build the global trading-day list from cached prices.

    Mirrors ``PriceCache._build_trading_days``: prefer SPY's keys; if SPY
    is absent or empty, fall back to the DENSEST non-empty series. Empty
    when no caches loaded — callers degrade to ``no_price_cache``.
    """
    spy = prices.get("SPY") or {}
    if not spy:
        best_n = 0
        for ticker, series in prices.items():
            if series and len(series) > best_n:
                spy = series
                best_n = len(series)
    if not spy:
        return []
    days: list[date] = []
    for d_iso in spy.keys():
        try:
            days.append(date.fromisoformat(d_iso))
        except (TypeError, ValueError):
            continue
    days.sort()
    return days


def compute_intraperiod_extremes(
    ticker: str,
    sim_d: date,
    series: dict[str, float],
    trading_days: list[date],
    horizon: int = HORIZON_DAYS,
) -> tuple[float | None, float | None]:
    """Replicate ``_fwd_intraperiod_extremes`` exactly using cached prices.

    Returns ``(min_pct, max_pct)`` signed % relative to sim_d's resolved
    close, or ``(None, None)`` when nothing resolves (no series, no
    trading-day anchor, sim-side walk-back fails, or every forward day
    walks back to or before sim_d's date). Pure, total, never raises —
    the parallel of every sibling honesty guard in the engine path.
    """
    if not series or not trading_days:
        return None, None
    sim_resolution = _walk_back_close(series, sim_d)
    if sim_resolution is None:
        return None, None
    sim_res_d, sim_close = sim_resolution
    if not sim_close or sim_close <= 0:
        return None, None
    # Locate sim_d in the global trading-day list. Use bisect_right so a
    # sim_d that IS itself a trading day points at idx s.t. trading_days[idx-1] == sim_d
    # — then `trading_days[idx + (k-1)]` for k=1..horizon gives the next h
    # trading days. If sim_d is NOT a trading day (weekend / holiday), the
    # walk-back already aligned sim_res_d to a known trading day; anchor on
    # THAT for the forward index, mirroring the engine's bisect behaviour
    # (the engine uses bisect_left for an exact sim_d hit, but its caller
    # already filters to is-trading-day sim_dates via `_td_index >= 0`).
    anchor = sim_res_d if sim_res_d in trading_days else sim_d
    idx = bisect.bisect_left(trading_days, anchor)
    # Anchor must actually be in trading_days for the forward walk to be
    # meaningful; otherwise we can't reliably step "h trading days forward".
    if idx >= len(trading_days) or trading_days[idx] != anchor:
        return None, None

    min_pct: float | None = None
    max_pct: float | None = None
    for k in range(1, horizon + 1):
        ti = idx + k
        if ti < 0 or ti >= len(trading_days):
            continue
        day = trading_days[ti]
        day_resolution = _walk_back_close(series, day)
        if day_resolution is None:
            continue
        day_res_d, day_close = day_resolution
        # Same walk-back collision guard as the engine's
        # `_fwd_intraperiod_extremes`: a day whose walk-back resolves to
        # sim_res_d (or earlier) tells us nothing forward, so skip it.
        if day_res_d <= sim_res_d:
            continue
        pct = (day_close - sim_close) / sim_close * 100.0
        if min_pct is None or pct < min_pct:
            min_pct = pct
        if max_pct is None or pct > max_pct:
            max_pct = pct
    if min_pct is None or max_pct is None:
        return None, None
    return round(min_pct, 4), round(max_pct, 4)


def analyze(
    outcomes_path: "Path | str | None" = None,
    cache_dir: "Path | str | None" = None,
) -> dict:
    """Read-only audit: report what backfill would do without writing.

    Mirrors the ``analyze`` contract every sibling diagnostic (stop_out_audit,
    mfe_conversion, baseline_compare) exposes — a pure function that
    returns a JSON-safe dict and never raises. Useful for the per-cycle
    ledger consumers and for a `--dry-run` CLI gate.

    Returns counts of:
      * ``rows_total``                — every parseable row
      * ``rows_eligible``             — BUY/SELL with finite ``forward_return_5d``
      * ``rows_already_has``          — already carrying finite intraperiod fields
      * ``rows_backfillable``         — would gain finite values from this run
      * ``rows_no_price_cache``       — sim_date outside every loaded cache
      * ``rows_walk_back_collision``  — every forward day collided with sim_d
    Plus a ``verdict`` summarizing the gap:
      * ``NOTHING_TO_BACKFILL`` — every eligible row already has values
      * ``READY_TO_BACKFILL``   — at least one row would gain coverage
      * ``NO_PRICE_CACHE``      — no price caches loaded at all
    """
    root = Path(__file__).resolve().parent.parent.parent
    if outcomes_path is None:
        outcomes_path = root / "data" / "decision_outcomes.jsonl"
    else:
        outcomes_path = Path(outcomes_path)
    if cache_dir is None:
        cache_dir = root / "data" / "backtest_cache"
    else:
        cache_dir = Path(cache_dir)

    out = {
        "status": "ok",
        "verdict": "NOTHING_TO_BACKFILL",
        "outcomes_path": str(outcomes_path),
        "cache_dir": str(cache_dir),
        "rows_total": 0,
        "rows_eligible": 0,
        "rows_already_has": 0,
        "rows_backfillable": 0,
        "rows_no_price_cache": 0,
        "rows_walk_back_collision": 0,
        "tickers_in_cache": 0,
        "trading_days_in_cache": 0,
    }

    if not outcomes_path.exists():
        out["status"] = "error"
        out["verdict"] = "OUTCOMES_FILE_MISSING"
        return out

    prices = load_price_caches(cache_dir)
    trading_days = build_trading_days(prices)
    out["tickers_in_cache"] = len(prices)
    out["trading_days_in_cache"] = len(trading_days)

    if not trading_days:
        # No cache loaded — every row would land in `no_price_cache`,
        # but report the verdict up front so a consumer doesn't have
        # to count to the same conclusion.
        # We still iterate so `rows_total` is honest.
        for row in _iter_rows(outcomes_path):
            if isinstance(row, dict):
                out["rows_total"] += 1
        out["verdict"] = "NO_PRICE_CACHE"
        return out

    trading_days_set = set(trading_days)

    for row in _iter_rows(outcomes_path):
        if not isinstance(row, dict):
            continue
        out["rows_total"] += 1
        action = str(row.get("action") or "").upper()
        if action not in ("BUY", "SELL"):
            continue
        if not _is_finite(row.get("forward_return_5d")):
            continue
        out["rows_eligible"] += 1

        if (_is_finite(row.get("forward_intraperiod_min_5d"))
                and _is_finite(row.get("forward_intraperiod_max_5d"))):
            out["rows_already_has"] += 1
            continue

        ticker = str(row.get("ticker") or "").upper()
        sim_d_str = str(row.get("sim_date") or "")
        if not ticker or not sim_d_str:
            continue
        try:
            sim_d = date.fromisoformat(sim_d_str[:10])
        except (TypeError, ValueError):
            continue

        series = prices.get(ticker)
        if not series:
            out["rows_no_price_cache"] += 1
            continue

        # Cheap pre-check: at least one date in the ticker's series must
        # be within the sim_d ± horizon+walk_back band, else this is a
        # `no_price_cache` row for THIS ticker even if the cache loaded.
        # Skip the expensive compute when there's no chance of a hit.
        sim_iso = sim_d.isoformat()
        max_iso = (sim_d + timedelta(days=HORIZON_DAYS * 2
                                     + WALK_BACK_DAYS)).isoformat()
        min_iso = (sim_d - timedelta(days=WALK_BACK_DAYS)).isoformat()
        if not any(min_iso <= d <= max_iso for d in series.keys()):
            out["rows_no_price_cache"] += 1
            continue

        # If sim_d itself isn't in the trading-day list (and walk-back
        # won't help us anchor for the forward stride), this row's window
        # exceeds our cache coverage.
        if sim_d not in trading_days_set:
            walk_anchor = None
            for delta in range(1, WALK_BACK_DAYS + 1):
                prior = sim_d - timedelta(days=delta)
                if prior in trading_days_set:
                    walk_anchor = prior
                    break
            if walk_anchor is None:
                out["rows_no_price_cache"] += 1
                continue

        intra_min, intra_max = compute_intraperiod_extremes(
            ticker, sim_d, series, trading_days, horizon=HORIZON_DAYS,
        )
        if intra_min is None or intra_max is None:
            out["rows_walk_back_collision"] += 1
            continue
        out["rows_backfillable"] += 1

    if out["tickers_in_cache"] == 0:
        out["verdict"] = "NO_PRICE_CACHE"
    elif out["rows_backfillable"] > 0:
        out["verdict"] = "READY_TO_BACKFILL"
    else:
        out["verdict"] = "NOTHING_TO_BACKFILL"
    return out


def apply_backfill(
    outcomes_path: "Path | str | None" = None,
    cache_dir: "Path | str | None" = None,
) -> dict:
    """Read the outcomes corpus, compute intraperiod extremes where
    possible, atomically rewrite the file.

    Atomicity:
      * Records the file's mtime + size at read time.
      * If either changed before the rewrite, ABORTS with
        ``status='aborted_concurrent_write'`` and the file is left
        untouched. Defensive against a re-awoken continuous loop
        appending mid-backfill.
      * Writes tmp file + ``Path.replace`` — mirrors the
        ``decision_outcomes.jsonl`` trim idiom in
        ``run_continuous_backtests.py`` exactly.

    Never overwrites a row that already carries finite intraperiod
    fields — additive only. Returns the same counts dict as
    ``analyze`` plus an ``applied: True`` flag and the resulting file's
    final line count.
    """
    root = Path(__file__).resolve().parent.parent.parent
    if outcomes_path is None:
        outcomes_path = root / "data" / "decision_outcomes.jsonl"
    else:
        outcomes_path = Path(outcomes_path)
    if cache_dir is None:
        cache_dir = root / "data" / "backtest_cache"
    else:
        cache_dir = Path(cache_dir)

    counts = {
        "status": "ok",
        "applied": False,
        "outcomes_path": str(outcomes_path),
        "cache_dir": str(cache_dir),
        "rows_total": 0,
        "rows_eligible": 0,
        "rows_already_has": 0,
        "rows_backfilled": 0,
        "rows_no_price_cache": 0,
        "rows_walk_back_collision": 0,
        "rows_unmodified_passthrough": 0,
        "final_line_count": 0,
        "tickers_in_cache": 0,
        "trading_days_in_cache": 0,
    }

    if not outcomes_path.exists():
        counts["status"] = "error"
        counts["verdict"] = "OUTCOMES_FILE_MISSING"
        return counts

    pre_stat = outcomes_path.stat()
    pre_mtime_ns = pre_stat.st_mtime_ns
    pre_size = pre_stat.st_size

    prices = load_price_caches(cache_dir)
    trading_days = build_trading_days(prices)
    counts["tickers_in_cache"] = len(prices)
    counts["trading_days_in_cache"] = len(trading_days)
    trading_days_set = set(trading_days)

    # Read every line — both parseable and unparseable. Unparseable lines
    # are passed through verbatim so a corrupt row is never silently
    # dropped by the backfill. Same defensive line-handling discipline
    # the trim idiom in `run_continuous_backtests.py` already follows.
    rebuilt_lines: list[str] = []
    with outcomes_path.open("r") as fh:
        raw_lines = fh.readlines()

    for raw_line in raw_lines:
        stripped = raw_line.strip()
        if not stripped:
            # Preserve blank lines exactly as-is (they'd be filtered by
            # `_iter_rows` consumers but we don't want to mutate spacing).
            rebuilt_lines.append(raw_line.rstrip("\n"))
            continue

        try:
            row = json.loads(stripped)
        except json.JSONDecodeError:
            # Unparseable — write back verbatim, count separately.
            counts["rows_unmodified_passthrough"] += 1
            rebuilt_lines.append(stripped)
            continue
        if not isinstance(row, dict):
            counts["rows_unmodified_passthrough"] += 1
            rebuilt_lines.append(stripped)
            continue

        counts["rows_total"] += 1
        action = str(row.get("action") or "").upper()

        if action not in ("BUY", "SELL"):
            rebuilt_lines.append(json.dumps(row))
            continue
        if not _is_finite(row.get("forward_return_5d")):
            rebuilt_lines.append(json.dumps(row))
            continue

        counts["rows_eligible"] += 1

        if (_is_finite(row.get("forward_intraperiod_min_5d"))
                and _is_finite(row.get("forward_intraperiod_max_5d"))):
            counts["rows_already_has"] += 1
            rebuilt_lines.append(json.dumps(row))
            continue

        if not trading_days:
            counts["rows_no_price_cache"] += 1
            rebuilt_lines.append(json.dumps(row))
            continue

        ticker = str(row.get("ticker") or "").upper()
        sim_d_str = str(row.get("sim_date") or "")
        if not ticker or not sim_d_str:
            rebuilt_lines.append(json.dumps(row))
            continue
        try:
            sim_d = date.fromisoformat(sim_d_str[:10])
        except (TypeError, ValueError):
            rebuilt_lines.append(json.dumps(row))
            continue

        series = prices.get(ticker)
        if not series:
            counts["rows_no_price_cache"] += 1
            rebuilt_lines.append(json.dumps(row))
            continue

        # Cheap window pre-check — same as in `analyze`.
        max_iso = (sim_d + timedelta(days=HORIZON_DAYS * 2
                                     + WALK_BACK_DAYS)).isoformat()
        min_iso = (sim_d - timedelta(days=WALK_BACK_DAYS)).isoformat()
        if not any(min_iso <= d <= max_iso for d in series.keys()):
            counts["rows_no_price_cache"] += 1
            rebuilt_lines.append(json.dumps(row))
            continue

        if sim_d not in trading_days_set:
            walk_anchor = None
            for delta in range(1, WALK_BACK_DAYS + 1):
                prior = sim_d - timedelta(days=delta)
                if prior in trading_days_set:
                    walk_anchor = prior
                    break
            if walk_anchor is None:
                counts["rows_no_price_cache"] += 1
                rebuilt_lines.append(json.dumps(row))
                continue

        intra_min, intra_max = compute_intraperiod_extremes(
            ticker, sim_d, series, trading_days, horizon=HORIZON_DAYS,
        )
        if intra_min is None or intra_max is None:
            counts["rows_walk_back_collision"] += 1
            rebuilt_lines.append(json.dumps(row))
            continue

        row["forward_intraperiod_min_5d"] = intra_min
        row["forward_intraperiod_max_5d"] = intra_max
        counts["rows_backfilled"] += 1
        rebuilt_lines.append(json.dumps(row))

    if counts["rows_backfilled"] == 0:
        counts["final_line_count"] = len(raw_lines)
        counts["status"] = "ok"
        counts["verdict"] = (
            "NO_PRICE_CACHE" if counts["tickers_in_cache"] == 0
            else "NOTHING_TO_BACKFILL"
        )
        return counts

    # Atomicity gate: re-check the file before rewriting. A continuous
    # loop that woke up mid-backfill would have appended rows beyond what
    # we read; clobbering them is exactly what the
    # `decision_outcomes.jsonl` trim guards against.
    post_stat = outcomes_path.stat()
    if (post_stat.st_mtime_ns != pre_mtime_ns
            or post_stat.st_size != pre_size):
        counts["status"] = "aborted_concurrent_write"
        counts["verdict"] = "ABORTED_CONCURRENT_WRITE"
        counts["applied"] = False
        return counts

    tmp = outcomes_path.with_suffix(".jsonl.backfill_tmp")
    tmp.write_text("\n".join(rebuilt_lines) + "\n")
    tmp.replace(outcomes_path)

    counts["applied"] = True
    counts["final_line_count"] = len(rebuilt_lines)
    counts["verdict"] = "BACKFILLED"
    return counts


def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.backfill_intraperiod",
        description=(
            "Backfill forward_intraperiod_min_5d / forward_intraperiod_max_5d "
            "into the historical decision_outcomes.jsonl corpus using existing "
            "per-window price caches. Pure offline; no network; never touches "
            "the decision_scorer pickle or the live trader."
        ),
    )
    p.add_argument("--outcomes", default=None,
                   help="Path to decision_outcomes.jsonl (default: "
                        "data/decision_outcomes.jsonl).")
    p.add_argument("--cache-dir", default=None, dest="cache_dir",
                   help="Path to backtest_cache directory (default: "
                        "data/backtest_cache).")
    p.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="Print backfill counts without writing.")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of a table.")
    return p


def main(argv: list[str] | None = None) -> int:
    import sys

    args = _build_arg_parser().parse_args(
        sys.argv[1:] if argv is None else argv)

    if args.dry_run:
        rep = analyze(args.outcomes, args.cache_dir)
        if args.json:
            print(json.dumps(rep, indent=2, sort_keys=True))
        else:
            print(f"[backfill_intraperiod] DRY RUN  "
                  f"verdict={rep.get('verdict')}")
            print(f"  outcomes:  {rep.get('outcomes_path')}")
            print(f"  cache_dir: {rep.get('cache_dir')}")
            print(f"  tickers_in_cache:        "
                  f"{rep.get('tickers_in_cache'):>8d}")
            print(f"  trading_days_in_cache:   "
                  f"{rep.get('trading_days_in_cache'):>8d}")
            print(f"  rows_total:              "
                  f"{rep.get('rows_total'):>8d}")
            print(f"  rows_eligible:           "
                  f"{rep.get('rows_eligible'):>8d}")
            print(f"  rows_already_has:        "
                  f"{rep.get('rows_already_has'):>8d}")
            print(f"  rows_backfillable:       "
                  f"{rep.get('rows_backfillable'):>8d}")
            print(f"  rows_no_price_cache:     "
                  f"{rep.get('rows_no_price_cache'):>8d}")
            print(f"  rows_walk_back_collision:"
                  f"{rep.get('rows_walk_back_collision'):>8d}")
        return 0 if rep.get("status") == "ok" else 1

    rep = apply_backfill(args.outcomes, args.cache_dir)
    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
    else:
        print(f"[backfill_intraperiod] APPLIED  "
              f"verdict={rep.get('verdict')}  "
              f"applied={rep.get('applied')}")
        print(f"  outcomes:  {rep.get('outcomes_path')}")
        print(f"  rows_total:               {rep.get('rows_total'):>8d}")
        print(f"  rows_eligible:            {rep.get('rows_eligible'):>8d}")
        print(f"  rows_already_has:         {rep.get('rows_already_has'):>8d}")
        print(f"  rows_backfilled:          {rep.get('rows_backfilled'):>8d}")
        print(f"  rows_no_price_cache:      {rep.get('rows_no_price_cache'):>8d}")
        print(f"  rows_walk_back_collision: "
              f"{rep.get('rows_walk_back_collision'):>8d}")
        print(f"  final_line_count:         "
              f"{rep.get('final_line_count'):>8d}")
    if rep.get("status") == "aborted_concurrent_write":
        return 2
    return 0 if rep.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

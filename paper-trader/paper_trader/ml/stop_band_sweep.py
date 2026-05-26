"""Stop-loss band sweep — multi-horizon edition.

Read-only diagnostic. Never trains, never touches ``decision_scorer.pkl``,
never modifies the corpus, never enters a trade path. Mirrors the
operational discipline of every sibling analyzer in ``paper_trader/ml/``
(``stop_out_audit`` / ``tp_band_sweep`` / ``mfe_conversion`` /
``gate_threshold_sweep``).

**Why this exists.** ``stop_out_audit`` answers ONE question — would the
deployed -8% stop on a 5d window have helped or hurt? — at ONE band and
ONE horizon. The 2026-05-26 multi-horizon feature
(pass #42 in AGENTS.md) added ``forward_intraperiod_min_10d`` /
``forward_intraperiod_min_20d`` alongside the existing 5d field, but
no downstream analyzer consumes them yet. The pass #42 author explicitly
flagged this as the next step:

    "The multi-horizon extremes added in Phase 2 will let the next
     stop_out_audit sweep test whether a wider band (e.g. -10%) over a
     longer window (10d, 20d) saves more than it costs."

This module is that consumer. It sweeps a 2-D grid of candidate STOP
bands × horizons (5d, 10d, 20d) over the BUY-row corpus, reporting
per-cell realized return and benefit-vs-no-stop. A skeptical quant asks
once: "is a wider stop on a longer window better than the inherited
-8% / 5d band?" — this analyzer answers it directly, with a verdict
ladder mirroring ``tp_band_sweep``:

| Verdict | Meaning |
|---------|---------|
| ``INSUFFICIENT_DATA`` | < ``MIN_BUYS`` BUY rows carry a finite intraperiod-min for the deployed horizon. Older outcome rows predate the 2026-05-23 / 2026-05-26 features — accumulate post-feature cycles until coverage clears the threshold, or run ``paper_trader.ml.backfill_intraperiod`` (5d only). |
| ``CELL_BEATS_DEPLOYED`` | best ``(band, horizon)`` cell's benefit_pp clears the deployed ``(-8%, 5d)`` cell by > ``EDGE_TOL_PP``. Tuning move available. |
| ``DEPLOYED_OPTIMAL`` | the deployed cell is within ``±EDGE_TOL_PP`` of the best — no tuning move clears noise. |
| ``NO_BAND_HELPS`` | every candidate cell's benefit < ``EDGE_TOL_PP`` — no stop, on any band or horizon, measurably improves realized return on this corpus. |

```bash
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.stop_band_sweep
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.stop_band_sweep --json
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.stop_band_sweep --bands 5,8,10 --horizons 5d,10d
```
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable


# Candidate STOP bands to sweep (drawdown percentage — POSITIVE values
# meaning "fires at -band%"). Grid is denser around the deployed -8%
# boundary (5/7/8/10/12) and sparser at the extremes (3/15/20).
DEFAULT_CANDIDATE_BANDS: tuple[float, ...] = (
    3.0, 5.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0,
)

# Horizons to sweep, mapped to outcome-row field suffixes. The 2026-05-26
# pass #42 feature added 10d/20d alongside the existing 5d. Each entry's
# corresponding ``forward_return_<h>`` AND ``forward_intraperiod_min_<h>``
# must both be present on a row for it to contribute to that horizon's
# sweep — partial-coverage rows degrade gracefully (counted once, dropped
# from horizons that lack data).
DEFAULT_HORIZONS: tuple[str, ...] = ("5d", "10d", "20d")

# Deployed stop band — ``backtest._buy`` writes ``stop_loss = price * 0.92``
# which is an 8% drawdown trigger. The verdict ladder compares the best
# candidate cell to the deployed (``DEPLOYED_STOP_PCT``, ``DEPLOYED_HORIZON``)
# cell. Module-level so any deployed-band tuning is a single reviewable edit.
DEPLOYED_STOP_PCT = 8.0

# Deployed horizon — ``_compute_decision_outcomes``'s legacy intraperiod
# pair pinned to 5d (the 2026-05-23 feature). This is the horizon over
# which the live ``stop_out_audit`` evaluates, so the deployed cell is
# (-8%, 5d) by construction.
DEPLOYED_HORIZON = "5d"

# Minimum BUYs with intraperiod data for the deployed horizon before any
# verdict. Matches ``stop_out_audit.MIN_BUYS`` / ``tp_band_sweep.MIN_BUYS``
# exactly so the three audits report comparable n_buys minima.
MIN_BUYS = 30

# Realized-return margin (percentage points) the best candidate must
# clear the deployed cell by before declaring ``CELL_BEATS_DEPLOYED``,
# AND the absolute floor any candidate must clear before
# ``NO_BAND_HELPS`` does NOT apply. Symmetric with
# ``tp_band_sweep.EDGE_TOL_PP`` — 0.30pp is ~one standard error on a
# 1000-trade aggregate at σ(target) ≈ 12pp.
EDGE_TOL_PP = 0.30


def _to_finite_float(v) -> float | None:
    """Return ``float(v)`` if finite, else None. Mirrors sibling analyzers'
    ``_to_finite_float`` — bool rejected, NaN/inf rejected, missing returns
    None so the caller can DROP a row rather than coerce to a default that
    would fabricate a "no trigger" reading."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _stop_protected_return(forward_return: float,
                            intra_min: float,
                            stop_pct: float) -> float:
    """Realized return for ONE BUY with a -``stop_pct``% stop band.

    Mirrors ``stop_out_audit._stop_protected_return`` byte-for-byte —
    kept locally (not imported) so this module is self-contained.

    If ``intra_min <= -stop_pct`` the stop fired — model the fill at
    exactly ``-stop_pct`` (a real fill on a gap-down would be WORSE, so
    the realized benefit this analyzer reports is an UPPER bound on the
    stop's edge — the same conservative-stop assumption a quant would
    treat as the optimistic case).

    Otherwise the position rides to the horizon endpoint and
    ``forward_return`` is captured. Pure, total, never raises.
    """
    if intra_min <= -stop_pct:
        return -stop_pct
    return forward_return


def _iter_rows(path: Path) -> Iterable[dict]:
    """Stream one JSON record per line, silently dropping unparseable rows.

    Same line-tolerant discipline as ``stop_out_audit._iter_rows`` /
    ``tp_band_sweep._iter_rows`` — a single corrupt line must not abort
    an audit run.
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


def _median(s: list[float]) -> float:
    """Median of a non-empty pre-sorted list."""
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def _row_fields(row: dict, horizon: str) -> tuple[float | None, float | None]:
    """Extract ``(forward_return, intra_min)`` for a horizon, finite-or-None.

    Centralises the field naming so a future horizon (e.g. ``30d``) is a
    one-line ``DEFAULT_HORIZONS`` extension — no per-horizon special-case
    code. The keys mirror ``_compute_decision_outcomes`` exactly:
    ``forward_return_<h>`` and ``forward_intraperiod_min_<h>``.
    """
    fwd = _to_finite_float(row.get(f"forward_return_{horizon}"))
    imn = _to_finite_float(row.get(f"forward_intraperiod_min_{horizon}"))
    return fwd, imn


def sweep_cells(per_horizon: dict[str, tuple[list[float], list[float]]],
                bands: Iterable[float]) -> list[dict]:
    """Per-(band, horizon) realized stats over a pre-extracted population.

    ``per_horizon`` maps each horizon string (e.g. ``"5d"``) to a tuple
    ``(realized, intra_min)`` of aligned finite-float lists. ``bands`` is
    any iterable of POSITIVE float candidates (the band magnitude in pp;
    the stop fires at ``-band``).

    Returns a list of cell-result dicts sorted by descending
    ``benefit_pct`` (so ``[0]`` is the strongest cell). Each row:

      * ``stop_pct``                  — the band tested (positive pp)
      * ``horizon``                   — the horizon tested (string)
      * ``n``                         — sample size for this horizon
      * ``n_triggered``               — BUYs where ``intra_min <= -band``
      * ``pct_triggered``             — n_triggered / n × 100
      * ``mean_protected_return_pct`` — mean realized return WITH stop
      * ``median_protected_return_pct``— median realized return WITH stop
      * ``mean_realized_return_pct``  — mean realized return WITHOUT stop
                                         (constant per horizon — surfaced
                                         per row for at-a-glance compare)
      * ``median_realized_return_pct``— median realized return WITHOUT
      * ``benefit_pct``               — protected mean − realized mean

    Pure, total. Empty input returns an empty list — the caller's
    ``INSUFFICIENT_DATA`` envelope owns that contract.
    """
    rows: list[dict] = []
    for horizon, (realized, intra_min) in per_horizon.items():
        if not realized or not intra_min:
            continue
        if len(realized) != len(intra_min):
            # Defensive: programming error, not user input. Drop the
            # horizon's cells rather than crash; mirrors tp_band_sweep.
            continue
        n = len(realized)
        realized_sorted = sorted(realized)
        mean_real = sum(realized) / n
        median_real = _median(realized_sorted)
        for band in bands:
            protected: list[float] = []
            n_trig = 0
            for fwd, imn in zip(realized, intra_min):
                sp = _stop_protected_return(fwd, imn, stop_pct=band)
                if imn <= -band:
                    n_trig += 1
                protected.append(sp)
            protected_sorted = sorted(protected)
            mean_prot = sum(protected) / n
            median_prot = _median(protected_sorted)
            rows.append({
                "stop_pct": float(band),
                "horizon": horizon,
                "n": n,
                "n_triggered": n_trig,
                "pct_triggered": round(n_trig / n * 100.0, 2),
                "mean_protected_return_pct": round(mean_prot, 4),
                "median_protected_return_pct": round(median_prot, 4),
                "mean_realized_return_pct": round(mean_real, 4),
                "median_realized_return_pct": round(median_real, 4),
                "benefit_pct": round(mean_prot - mean_real, 4),
            })
    # Sort descending by benefit; stable so equal-benefit cells preserve
    # the iteration order (which is horizon-then-band ascending — the
    # TIGHTER band / SHORTER horizon wins ties, matching the conservative
    # discipline tp_band_sweep uses).
    rows.sort(key=lambda r: -r["benefit_pct"])
    return rows


def _coerce_bands(bands: Iterable[float],
                   deployed_band: float) -> list[float]:
    """Coerce a candidate-band iterable to a sorted ascending list of
    positive finite floats, always including ``deployed_band`` so the
    deployed cell's benefit appears in the sweep even when the caller
    passes a custom grid that omits it. Mirrors ``tp_band_sweep``'s
    de-dup-via-dict idiom."""
    grid = list(dict.fromkeys(list(bands) + [deployed_band]))
    clean: list[float] = []
    for b in grid:
        try:
            bf = float(b)
        except (TypeError, ValueError):
            continue
        if math.isfinite(bf) and bf > 0:
            clean.append(bf)
    clean.sort()
    return clean


def _coerce_horizons(horizons: Iterable[str],
                      deployed_horizon: str) -> list[str]:
    """Coerce horizons to a list, always including ``deployed_horizon``
    so its row is in the sweep. Mirrors ``_coerce_bands`` shape exactly.
    String entries only — non-string tokens silently drop."""
    seq = list(horizons) + [deployed_horizon]
    out: list[str] = []
    seen: set[str] = set()
    for h in seq:
        if not isinstance(h, str):
            continue
        h = h.strip()
        if h and h not in seen:
            seen.add(h)
            out.append(h)
    return out


def analyze(outcomes_path: "Path | str | None" = None,
            bands: Iterable[float] = DEFAULT_CANDIDATE_BANDS,
            horizons: Iterable[str] = DEFAULT_HORIZONS,
            deployed_stop_pct: float = DEPLOYED_STOP_PCT,
            deployed_horizon: str = DEPLOYED_HORIZON,
            min_buys: int = MIN_BUYS,
            edge_tol_pp: float = EDGE_TOL_PP) -> dict:
    """Compute the multi-horizon stop-band sweep report.

    Reads ``data/decision_outcomes.jsonl`` (or the passed path), extracts
    per-horizon ``(forward_return, intra_min)`` populations, then sweeps
    every ``(band, horizon)`` cell.

    Always returns a JSON-safe dict; never raises (mirrors
    ``stop_out_audit.analyze`` / ``tp_band_sweep.analyze``). On any fault
    returns an ``INSUFFICIENT_DATA`` envelope with a ``hint`` string so
    a ledger consumer can persist the failure mode honestly.

    Important: ``deployed_stop_pct`` AND ``deployed_horizon`` are always
    inserted into the sweep if absent so the deployed cell's benefit is
    always reported. The verdict compares the best swept cell against
    the deployed cell.
    """
    if outcomes_path is None:
        outcomes_path = (Path(__file__).resolve().parent.parent.parent
                         / "data" / "decision_outcomes.jsonl")
    else:
        outcomes_path = Path(outcomes_path)

    clean_bands = _coerce_bands(bands, deployed_stop_pct)
    clean_horizons = _coerce_horizons(horizons, deployed_horizon)

    empty = {
        "status": "insufficient_data",
        "verdict": "INSUFFICIENT_DATA",
        "deployed_stop_pct": deployed_stop_pct,
        "deployed_horizon": deployed_horizon,
        "edge_tol_pp": edge_tol_pp,
        "n_buys": 0,
        "n_with_intraperiod_per_horizon": {h: 0 for h in clean_horizons},
        "baseline_no_stop_mean_pct_per_horizon": {
            h: None for h in clean_horizons},
        "bands_swept": clean_bands,
        "horizons_swept": clean_horizons,
        "sweep": [],
        "best_cell": None,
        "deployed_cell_benefit_pct": None,
        "hint": None,
    }

    if not outcomes_path.exists():
        empty["hint"] = f"outcomes file not found: {outcomes_path}"
        return empty
    if not clean_bands:
        empty["hint"] = "candidate band grid resolved to empty after coercion"
        return empty
    if not clean_horizons:
        empty["hint"] = (
            "candidate horizon grid resolved to empty after coercion")
        return empty

    n_buys = 0
    # Per-horizon aligned pairs of (realized, intra_min) populations.
    per_horizon: dict[str, tuple[list[float], list[float]]] = {
        h: ([], []) for h in clean_horizons}

    try:
        for row in _iter_rows(outcomes_path):
            if not isinstance(row, dict):
                continue
            action = str(row.get("action") or "").upper()
            if action != "BUY":
                continue
            n_buys += 1
            for h in clean_horizons:
                fwd, imn = _row_fields(row, h)
                if fwd is None or imn is None:
                    # Partial-coverage row — skip THIS horizon but still
                    # let the row contribute to other horizons. Mirrors
                    # the "honor partial coverage" discipline pinned by
                    # test_outcome_intraperiod_multihorizon.
                    continue
                per_horizon[h][0].append(fwd)
                per_horizon[h][1].append(imn)
    except Exception as exc:
        empty["n_buys"] = n_buys
        empty["hint"] = f"row scan failed: {type(exc).__name__}: {exc}"
        return empty

    coverage = {h: len(per_horizon[h][0]) for h in clean_horizons}

    # Gate on the DEPLOYED horizon's coverage (the verdict compares the
    # best cell against the deployed (band, horizon) cell — if that cell
    # itself has too little data the verdict is meaningless).
    deployed_n = coverage.get(deployed_horizon, 0)
    if deployed_n < min_buys:
        empty["n_buys"] = n_buys
        empty["n_with_intraperiod_per_horizon"] = coverage
        empty["hint"] = (
            f"deployed horizon {deployed_horizon!r}: only {deployed_n} BUYs "
            f"with intraperiod data (< {min_buys}); older outcome rows "
            f"predate the 2026-05-23 / 2026-05-26 forward_intraperiod_* "
            f"features. Accumulate more post-feature cycles (or run "
            f"paper_trader.ml.backfill_intraperiod for the 5d horizon) "
            f"until coverage clears the threshold."
        )
        return empty

    sweep = sweep_cells(per_horizon, clean_bands)
    if not sweep:
        empty["n_buys"] = n_buys
        empty["n_with_intraperiod_per_horizon"] = coverage
        empty["hint"] = "no horizon yielded a usable population for the sweep"
        return empty

    best_cell = sweep[0]
    deployed_entry = next(
        (r for r in sweep
         if r["stop_pct"] == deployed_stop_pct
         and r["horizon"] == deployed_horizon),
        None,
    )
    deployed_benefit = (deployed_entry["benefit_pct"]
                        if deployed_entry is not None else None)

    # Per-horizon baseline means — surfaced for at-a-glance comparison
    # (a 10d horizon's no-stop mean is naturally different from 5d's).
    baseline_per_horizon: dict[str, float | None] = {}
    for h, (realized, _) in per_horizon.items():
        if realized:
            baseline_per_horizon[h] = round(sum(realized) / len(realized), 4)
        else:
            baseline_per_horizon[h] = None

    if deployed_benefit is None:
        verdict = "INSUFFICIENT_DATA"
        hint = (f"deployed cell ({deployed_stop_pct}%, "
                f"{deployed_horizon!r}) missing from sweep — "
                f"verdict unavailable")
    elif best_cell["benefit_pct"] < edge_tol_pp:
        verdict = "NO_BAND_HELPS"
        hint = (f"best cell ({best_cell['stop_pct']}%, "
                f"{best_cell['horizon']!r}) realised "
                f"{best_cell['benefit_pct']:+.3f}pp benefit, below the "
                f"{edge_tol_pp}pp noise margin — no stop band / horizon "
                f"combination is measurably adding economic edge on "
                f"this corpus.")
    elif best_cell["benefit_pct"] - deployed_benefit > edge_tol_pp:
        verdict = "CELL_BEATS_DEPLOYED"
        hint = (f"best cell ({best_cell['stop_pct']}%, "
                f"{best_cell['horizon']!r}) realised "
                f"{best_cell['benefit_pct']:+.3f}pp vs deployed "
                f"({deployed_stop_pct}%, {deployed_horizon!r}) "
                f"{deployed_benefit:+.3f}pp — "
                f"{best_cell['benefit_pct'] - deployed_benefit:+.3f}pp "
                f"gain clears the {edge_tol_pp}pp noise margin. Tuning "
                f"move available.")
    else:
        verdict = "DEPLOYED_OPTIMAL"
        hint = (f"deployed cell ({deployed_stop_pct}%, "
                f"{deployed_horizon!r}) benefit "
                f"{deployed_benefit:+.3f}pp is within {edge_tol_pp}pp of "
                f"the best cell ({best_cell['stop_pct']}%, "
                f"{best_cell['horizon']!r}) "
                f"{best_cell['benefit_pct']:+.3f}pp — no tuning move "
                f"clears noise.")

    return {
        "status": "ok",
        "verdict": verdict,
        "deployed_stop_pct": deployed_stop_pct,
        "deployed_horizon": deployed_horizon,
        "edge_tol_pp": edge_tol_pp,
        "n_buys": n_buys,
        "n_with_intraperiod_per_horizon": coverage,
        "baseline_no_stop_mean_pct_per_horizon": baseline_per_horizon,
        "bands_swept": clean_bands,
        "horizons_swept": clean_horizons,
        "sweep": sweep,
        "best_cell": {
            "stop_pct": best_cell["stop_pct"],
            "horizon": best_cell["horizon"],
            "benefit_pct": best_cell["benefit_pct"],
            "n_triggered": best_cell["n_triggered"],
            "pct_triggered": best_cell["pct_triggered"],
            "mean_protected_return_pct":
                best_cell["mean_protected_return_pct"],
        },
        "deployed_cell_benefit_pct": deployed_benefit,
        "hint": hint,
    }


# ---------------------------------------------------------------------------
# CLI: `python3 -m paper_trader.ml.stop_band_sweep [--json] [--bands ...] [--horizons ...]`
# Pattern mirrors stop_out_audit / tp_band_sweep / mfe_conversion exactly.
# ---------------------------------------------------------------------------


def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.stop_band_sweep",
        description="Multi-horizon stop-band sweep — compare candidate "
                    "stop bands x horizons against the deployed -8%% / 5d "
                    "cell on realized return. Read-only — reads "
                    "decision_outcomes.jsonl, never trains, never modifies "
                    "the pickle or any trade path.",
    )
    p.add_argument("--outcomes", default=None,
                   help="Path to decision_outcomes.jsonl (default: the "
                        "repo data/ path).")
    p.add_argument("--bands", default=None,
                   help="Comma-separated band grid in pp drawdown "
                        f"(default: "
                        f"{','.join(str(b) for b in DEFAULT_CANDIDATE_BANDS)}). "
                        f"The deployed {DEPLOYED_STOP_PCT}% band is always "
                        f"included even when omitted from this list.")
    p.add_argument("--horizons", default=None,
                   help="Comma-separated horizon grid "
                        f"(default: {','.join(DEFAULT_HORIZONS)}). The "
                        f"deployed {DEPLOYED_HORIZON!r} horizon is always "
                        f"included even when omitted from this list.")
    p.add_argument("--deployed-stop", type=float, default=DEPLOYED_STOP_PCT,
                   dest="deployed_stop_pct",
                   help=f"Deployed stop band in pp (default "
                        f"{DEPLOYED_STOP_PCT}%%).")
    p.add_argument("--deployed-horizon", default=DEPLOYED_HORIZON,
                   dest="deployed_horizon",
                   help=f"Deployed horizon (default {DEPLOYED_HORIZON!r}).")
    p.add_argument("--margin", type=float, default=EDGE_TOL_PP,
                   dest="edge_tol_pp",
                   help=f"Realized-return margin (pp) any candidate must "
                        f"clear to declare CELL_BEATS_DEPLOYED OR for the "
                        f"best candidate to clear absolute zero (default "
                        f"{EDGE_TOL_PP}pp).")
    p.add_argument("--min-buys", type=int, default=MIN_BUYS, dest="min_buys",
                   help=f"Minimum BUYs with intraperiod data on the "
                        f"deployed horizon before any verdict "
                        f"(default {MIN_BUYS}).")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of a table.")
    return p


def _parse_csv_floats(raw: str | None,
                       default: tuple[float, ...]) -> tuple[float, ...]:
    """Parse ``--bands 5,8,10`` into a tuple of positive finite floats.
    Returns ``default`` for ``None`` / empty / all-tokens-invalid input."""
    if not raw:
        return default
    out: list[float] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = float(tok)
        except ValueError:
            continue
        if math.isfinite(v) and v > 0:
            out.append(v)
    return tuple(out) if out else default


def _parse_csv_strings(raw: str | None,
                        default: tuple[str, ...]) -> tuple[str, ...]:
    """Parse ``--horizons 5d,10d`` into a tuple of non-empty strings.
    Returns ``default`` for ``None`` / empty / all-tokens-invalid input."""
    if not raw:
        return default
    out: list[str] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if tok:
            out.append(tok)
    return tuple(out) if out else default


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Returns 0 on a decisive non-neutral verdict
    (CELL_BEATS_DEPLOYED / DEPLOYED_OPTIMAL), 1 on INSUFFICIENT_DATA
    or NO_BAND_HELPS — mirrors stop_out_audit's return-code convention."""
    import sys

    args = _build_arg_parser().parse_args(
        sys.argv[1:] if argv is None else argv)
    bands = _parse_csv_floats(args.bands, DEFAULT_CANDIDATE_BANDS)
    horizons = _parse_csv_strings(args.horizons, DEFAULT_HORIZONS)
    rep = analyze(
        outcomes_path=args.outcomes,
        bands=bands,
        horizons=horizons,
        deployed_stop_pct=args.deployed_stop_pct,
        deployed_horizon=args.deployed_horizon,
        min_buys=args.min_buys,
        edge_tol_pp=args.edge_tol_pp,
    )

    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
        return 0 if rep["verdict"] in {
            "CELL_BEATS_DEPLOYED", "DEPLOYED_OPTIMAL",
        } else 1

    verdict = rep["verdict"]
    print(f"[stop_band_sweep] verdict={verdict}")
    print(f"  deployed=(-{rep['deployed_stop_pct']}%, "
          f"{rep['deployed_horizon']!r})  edge_tol="
          f"±{rep['edge_tol_pp']}pp  min_buys={MIN_BUYS}")
    print(f"  n_buys={rep['n_buys']}  coverage_per_horizon="
          f"{rep['n_with_intraperiod_per_horizon']}")
    baseline = rep.get("baseline_no_stop_mean_pct_per_horizon") or {}
    if any(v is not None for v in baseline.values()):
        parts = ", ".join(
            f"{h}={v:+.3f}pp" if v is not None else f"{h}=—"
            for h, v in baseline.items())
        print(f"  baseline_no_stop_mean: {parts}")
    sweep = rep.get("sweep") or []
    if sweep:
        print(f"  band  horizon  n      n_trig  %_trig  mean_prot  benefit  ←-best→")
        for r in sweep:
            star = "  *" if (
                rep["best_cell"]
                and r["stop_pct"] == rep["best_cell"]["stop_pct"]
                and r["horizon"] == rep["best_cell"]["horizon"]) else ""
            dmark = "  [deployed]" if (
                r["stop_pct"] == rep["deployed_stop_pct"]
                and r["horizon"] == rep["deployed_horizon"]) else ""
            print(f"  {r['stop_pct']:>4.1f}%  {r['horizon']:>5}  "
                  f"{r['n']:>5}  {r['n_triggered']:>5}  "
                  f"{r['pct_triggered']:>5.1f}%  "
                  f"{r['mean_protected_return_pct']:>+8.3f}pp  "
                  f"{r['benefit_pct']:>+7.3f}pp{star}{dmark}")
    if rep.get("hint"):
        print(f"  hint: {rep['hint']}")
    return 0 if verdict in {"CELL_BEATS_DEPLOYED", "DEPLOYED_OPTIMAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

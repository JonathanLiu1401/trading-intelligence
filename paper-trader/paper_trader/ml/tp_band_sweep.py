"""Take-profit band sweep — would a band other than the deployed +15%
``backtest._buy`` ``take_profit`` arm realize better aggregate return?

Read-only diagnostic. Never trains, never touches ``decision_scorer.pkl``,
never modifies the corpus, never enters a trade path. Mirrors the
operational discipline of every sibling analyzer in ``paper_trader/ml/``
(``stop_out_audit`` / ``mfe_conversion`` / ``gate_threshold_sweep``).

**Why this exists.** ``mfe_conversion`` answers ONE question: does the
deployed +15% take-profit band beat no TP? The live verdict is
``TP_NEUTRAL`` (benefit -0.23pp on n=1913 BUYs) — but
``mean_conversion_ratio=-0.45`` says positions on average revert FROM
their intraperiod peak rather than capturing it. A skeptical quant's
direct next question — *would a TIGHTER take-profit have monetised that
peak-then-revert pattern?* — is structurally unanswerable with
``mfe_conversion`` because that analyzer is fixed at one ``TP_PCT``.

This module sweeps a grid of candidate TP bands and reports per-band
realized return + benefit vs no-TP. The verdict compares the best
candidate to the deployed +15% band:

| Verdict | Meaning |
|---------|---------|
| ``INSUFFICIENT_DATA`` | < ``MIN_BUYS`` BUY rows carry a finite ``forward_intraperiod_max_5d``. The deployed pre-feature corpus (2026-05-23 feature) is the dominant cause; once the rolling 5000-record training tail is dominated by post-feature rows this flips on its own. |
| ``BAND_BEATS_DEPLOYED`` | best candidate's benefit_pp beats the deployed +15% by > ``EDGE_TOL_PP`` (the deployed band is leaving realized return on the table). |
| ``DEPLOYED_OPTIMAL`` | the deployed +15% is within ``±EDGE_TOL_PP`` of the best candidate (no tuning move clears noise). |
| ``NO_BAND_HELPS`` | EVERY candidate has benefit_pp < ``EDGE_TOL_PP`` (no TP — regardless of band — measurably helps; the corpus is dominated by trades that DON'T peak-then-revert). |

```bash
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.tp_band_sweep
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.tp_band_sweep --json
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.tp_band_sweep --bands 5,8,10,12
```
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable


# Candidate TP bands to sweep (percentage above entry that triggers the
# take-profit). Includes the deployed 15.0 so its own band always appears
# in the table — a reader can compare any candidate to the live arm
# without a separate query. The grid is dense around the deployed
# boundary (10/12/15/18/20) where a quant's interest is highest, and
# sparser at the tighter (3/5/7) and looser (25/30) extremes where the
# expected sign of the move is more obvious.
DEFAULT_CANDIDATE_BANDS: tuple[float, ...] = (
    3.0, 5.0, 7.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 25.0, 30.0,
)

# Deployed take-profit band — `backtest._buy` writes
# `take_profit = round(price * 1.15, 2)` which is +15% from entry. Module
# constant so a deployed-band change is a single reviewable edit AND so
# every test that pins the deployed-band comparison can monkey-patch one
# constant rather than rebuilding the candidate grid.
DEPLOYED_TP_PCT = 15.0

# Minimum BUYs with intraperiod data before any verdict. Below this the
# report is honestly ``INSUFFICIENT_DATA``. Mirrors
# ``stop_out_audit.MIN_BUYS`` / ``mfe_conversion.MIN_BUYS`` exactly so
# the three audits report comparable n_buys minima.
MIN_BUYS = 30

# Realized-return margin (percentage points) the best candidate must
# clear the deployed band by before declaring ``BAND_BEATS_DEPLOYED``,
# AND that any candidate must individually clear before declaring
# ``NO_BAND_HELPS`` does NOT apply. Symmetric with
# ``mfe_conversion.BENEFIT_MARGIN`` — 0.30pp is roughly one standard
# error on a 1000-trade aggregate at σ(target) ≈ 12pp; anything tighter
# is sampling noise.
EDGE_TOL_PP = 0.30


def _to_finite_float(v) -> float | None:
    """Return ``float(v)`` if finite, else None. Mirrors
    ``stop_out_audit._to_finite_float`` / ``mfe_conversion._to_finite_float``
    semantics — bool rejected, NaN/inf rejected, missing returns None so
    the caller can DROP the row rather than coerce to a default that
    would fabricate a "no trigger" reading on a missing-data row."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _tp_protected_return(forward_return_5d: float,
                          intra_max: float,
                          tp_pct: float) -> float:
    """Realized return for ONE BUY with the documented +``tp_pct``% TP.

    Mirrors ``mfe_conversion._tp_protected_return`` byte-for-byte — kept
    locally (not imported) so this module is self-contained and a
    downstream environment that vends only this analyzer never has a
    hidden dependency on a sibling.

    If ``intra_max >= tp_pct`` the TP fired — model the fill at exactly
    ``tp_pct`` (conservative-fill assumption; a real gap-up fill would
    be better, so the realized BENEFIT this analyzer reports is a LOWER
    bound). Otherwise the position rides to the 5d endpoint.

    Pure, total, never raises."""
    if intra_max >= tp_pct:
        return tp_pct
    return forward_return_5d


def _iter_rows(path: Path) -> Iterable[dict]:
    """Stream one JSON record per line, silently dropping unparseable rows.

    Same line-tolerant discipline as ``stop_out_audit._iter_rows`` /
    ``mfe_conversion._iter_rows`` — a single corrupt line must not
    abort an audit run.
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


def sweep_bands(realized: list[float],
                intra_max: list[float],
                bands: Iterable[float]) -> list[dict]:
    """Per-band realized stats over a pre-extracted BUY population.

    ``realized`` and ``intra_max`` are aligned lists (one element per
    BUY) of finite floats; the caller is responsible for that filter
    (``analyze`` does it). ``bands`` is any iterable of float band
    candidates (% above entry that triggers).

    Returns a list of band-result dicts sorted by descending
    ``benefit_pct`` (so the first row is the strongest band). Each row:

      * ``tp_pct``                     — the band tested
      * ``n``                          — sample size (== len(realized))
      * ``n_triggered``                — BUYs where ``intra_max >= tp_pct``
      * ``pct_triggered``              — n_triggered / n × 100
      * ``mean_protected_return_pct``  — mean realized return WITH band
      * ``median_protected_return_pct``— median realized return WITH band
      * ``mean_realized_return_pct``   — mean realized return WITHOUT band
                                          (constant across rows, surfaced
                                          for at-a-glance comparison)
      * ``benefit_pct``                — protected mean − realized mean

    Pure, total. Empty ``realized`` returns an empty list — the caller's
    ``INSUFFICIENT_DATA`` envelope owns that contract.
    """
    if not realized or not intra_max:
        return []
    if len(realized) != len(intra_max):
        # Defensive: caller is supposed to pre-align. A length mismatch
        # is a programming error, not user input — return empty rather
        # than crash. Keeps the "diagnostic must not break the cycle"
        # discipline intact for any wrapper / ledger consumer.
        return []
    n = len(realized)
    realized_sorted = sorted(realized)
    mean_real = sum(realized) / n
    median_real = _median(realized_sorted)

    rows: list[dict] = []
    for band in bands:
        protected: list[float] = []
        n_trig = 0
        for fwd, imax in zip(realized, intra_max):
            sp = _tp_protected_return(fwd, imax, tp_pct=band)
            if imax >= band:
                n_trig += 1
            protected.append(sp)
        protected_sorted = sorted(protected)
        mean_prot = sum(protected) / n
        median_prot = _median(protected_sorted)
        rows.append({
            "tp_pct": float(band),
            "n": n,
            "n_triggered": n_trig,
            "pct_triggered": round(n_trig / n * 100.0, 2),
            "mean_protected_return_pct": round(mean_prot, 4),
            "median_protected_return_pct": round(median_prot, 4),
            "mean_realized_return_pct": round(mean_real, 4),
            "median_realized_return_pct": round(median_real, 4),
            "benefit_pct": round(mean_prot - mean_real, 4),
        })
    # Sort descending by benefit so [0] is always the best band — every
    # consumer (the CLI table, the analyze verdict) wants this order.
    # Stable sort on equal benefit preserves the input grid order, which
    # itself is ascending in tp_pct — so among equally-good bands the
    # TIGHTER one wins (the same conservative discipline the gate
    # threshold sweep uses).
    rows.sort(key=lambda r: -r["benefit_pct"])
    return rows


def analyze(outcomes_path: "Path | str | None" = None,
            bands: Iterable[float] = DEFAULT_CANDIDATE_BANDS,
            deployed_tp_pct: float = DEPLOYED_TP_PCT,
            min_buys: int = MIN_BUYS,
            edge_tol_pp: float = EDGE_TOL_PP) -> dict:
    """Compute the TP-band sweep report.

    Reads ``data/decision_outcomes.jsonl`` (or the passed path),
    filters to BUYs with finite ``forward_return_5d`` AND finite
    ``forward_intraperiod_max_5d``, then sweeps the candidate bands.

    Always returns a JSON-safe dict; never raises (mirrors
    ``stop_out_audit.analyze`` / ``mfe_conversion.analyze``). On any
    fault returns an ``INSUFFICIENT_DATA`` envelope with a ``hint``
    string so a ledger consumer can persist the failure mode honestly.

    Important: ``deployed_tp_pct`` is **always inserted into the sweep
    if absent** so the deployed band's benefit is reported even when a
    caller passes a custom band grid that omits 15.0. The verdict then
    compares the best sweeping band against this deployed entry.
    """
    if outcomes_path is None:
        outcomes_path = (Path(__file__).resolve().parent.parent.parent
                         / "data" / "decision_outcomes.jsonl")
    else:
        outcomes_path = Path(outcomes_path)

    # Always include the deployed band in the sweep so the report can
    # report its benefit even when a CLI caller passes ``--bands 5,8,10``.
    # Stable de-dup via dict (insertion-ordered in Py3.7+).
    grid = list(dict.fromkeys(list(bands) + [deployed_tp_pct]))
    # Coerce to floats once so the comparison in the result is stable;
    # any string / non-finite entry silently drops (an honest input filter
    # is preferable to a mid-sweep crash on a typo).
    clean_grid: list[float] = []
    for b in grid:
        try:
            bf = float(b)
        except (TypeError, ValueError):
            continue
        if math.isfinite(bf) and bf > 0:
            clean_grid.append(bf)
    # Ascending order for the table; sweep_bands resorts by benefit. The
    # ascending order surfaces in the empty / insufficient-data envelope
    # too so a downstream JSON consumer sees a stable shape.
    clean_grid.sort()

    empty = {
        "status": "insufficient_data",
        "verdict": "INSUFFICIENT_DATA",
        "deployed_tp_pct": deployed_tp_pct,
        "edge_tol_pp": edge_tol_pp,
        "n_buys": 0,
        "n_with_intraperiod": 0,
        "baseline_no_band_mean_pct": None,
        "bands_swept": clean_grid,
        "sweep": [],
        "best_band": None,
        "deployed_band_benefit_pct": None,
        "hint": None,
    }

    if not outcomes_path.exists():
        empty["hint"] = f"outcomes file not found: {outcomes_path}"
        return empty

    n_buys = 0
    realized: list[float] = []
    intra_max: list[float] = []

    try:
        for row in _iter_rows(outcomes_path):
            if not isinstance(row, dict):
                continue
            action = str(row.get("action") or "").upper()
            if action != "BUY":
                continue
            n_buys += 1
            fwd = _to_finite_float(row.get("forward_return_5d"))
            imax = _to_finite_float(row.get("forward_intraperiod_max_5d"))
            if fwd is None or imax is None:
                continue
            realized.append(fwd)
            intra_max.append(imax)
    except Exception as exc:
        empty["hint"] = f"row scan failed: {type(exc).__name__}: {exc}"
        return empty

    n_with_intra = len(realized)
    if n_with_intra < min_buys:
        empty["n_buys"] = n_buys
        empty["n_with_intraperiod"] = n_with_intra
        empty["hint"] = (
            f"only {n_with_intra} BUYs with intraperiod data (< {min_buys}); "
            f"older outcome rows predate the 2026-05-23 forward_intraperiod_* "
            f"feature. Run paper_trader.ml.backfill_intraperiod (or accumulate "
            f"more post-feature cycles) until coverage clears the threshold."
        )
        return empty

    sweep = sweep_bands(realized, intra_max, clean_grid)
    if not sweep:
        empty["n_buys"] = n_buys
        empty["n_with_intraperiod"] = n_with_intra
        empty["hint"] = "candidate band grid resolved to empty after coercion"
        return empty

    baseline_no_band_mean = sweep[0]["mean_realized_return_pct"]
    best_band = sweep[0]
    # Locate the deployed band's entry — guaranteed present by the
    # ``grid.append(deployed_tp_pct)`` above. `tp_pct` is a Python float
    # and the de-dup keeps the float identity stable, so exact equality is
    # safe here.
    deployed_entry = next(
        (r for r in sweep
         if r["tp_pct"] == deployed_tp_pct),
        None,
    )
    deployed_benefit = (deployed_entry["benefit_pct"]
                        if deployed_entry is not None else None)

    # Verdict ladder. Best > deployed by margin → tuning move available.
    # Best ≤ edge_tol_pp absolute → no band, regardless of grid, materially
    # helps. Otherwise the deployed band is within noise of the best —
    # leave the live arm alone.
    if deployed_benefit is None:
        verdict = "INSUFFICIENT_DATA"
        hint = (f"deployed band {deployed_tp_pct}% missing from sweep — "
                f"verdict unavailable")
    elif best_band["benefit_pct"] < edge_tol_pp:
        # Even the best candidate is below the noise margin — no band
        # measurably helps regardless of where it's set. Note the
        # asymmetric < (not <=): exactly EDGE_TOL_PP clears it.
        verdict = "NO_BAND_HELPS"
        hint = (f"best candidate ({best_band['tp_pct']}%) realised "
                f"{best_band['benefit_pct']:+.3f}pp benefit, below the "
                f"{edge_tol_pp}pp noise margin — no TP band is measurably "
                f"adding economic edge on this corpus.")
    elif best_band["benefit_pct"] - deployed_benefit > edge_tol_pp:
        verdict = "BAND_BEATS_DEPLOYED"
        hint = (f"best candidate {best_band['tp_pct']}% realised "
                f"{best_band['benefit_pct']:+.3f}pp vs deployed "
                f"{deployed_tp_pct}% {deployed_benefit:+.3f}pp — "
                f"{best_band['benefit_pct'] - deployed_benefit:+.3f}pp gain "
                f"clears the {edge_tol_pp}pp noise margin. Tuning move "
                f"available.")
    else:
        verdict = "DEPLOYED_OPTIMAL"
        hint = (f"deployed {deployed_tp_pct}% benefit "
                f"{deployed_benefit:+.3f}pp is within {edge_tol_pp}pp of "
                f"the best candidate ({best_band['tp_pct']}% "
                f"{best_band['benefit_pct']:+.3f}pp) — no tuning move "
                f"clears noise.")

    return {
        "status": "ok",
        "verdict": verdict,
        "deployed_tp_pct": deployed_tp_pct,
        "edge_tol_pp": edge_tol_pp,
        "n_buys": n_buys,
        "n_with_intraperiod": n_with_intra,
        "baseline_no_band_mean_pct": baseline_no_band_mean,
        "bands_swept": clean_grid,
        "sweep": sweep,
        "best_band": {
            "tp_pct": best_band["tp_pct"],
            "benefit_pct": best_band["benefit_pct"],
            "n_triggered": best_band["n_triggered"],
            "pct_triggered": best_band["pct_triggered"],
            "mean_protected_return_pct":
                best_band["mean_protected_return_pct"],
        },
        "deployed_band_benefit_pct": deployed_benefit,
        "hint": hint,
    }


# ---------------------------------------------------------------------------
# CLI: `python3 -m paper_trader.ml.tp_band_sweep [--json] [--bands ...]`
# Pattern mirrors stop_out_audit / mfe_conversion exactly.
# ---------------------------------------------------------------------------


def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.tp_band_sweep",
        description="Take-profit band sweep — compare candidate TP bands "
                    "against the deployed +15%% arm on realized return. "
                    "Read-only — reads decision_outcomes.jsonl, never "
                    "trains, never modifies the pickle or any trade path.",
    )
    p.add_argument("--outcomes", default=None,
                   help="Path to decision_outcomes.jsonl (default: the "
                        "repo data/ path).")
    p.add_argument("--bands", default=None,
                   help="Comma-separated band grid in percent above entry "
                        f"(default: {','.join(str(b) for b in DEFAULT_CANDIDATE_BANDS)}). "
                        f"The deployed {DEPLOYED_TP_PCT}% band is always "
                        f"included even when omitted from this list.")
    p.add_argument("--deployed-tp", type=float, default=DEPLOYED_TP_PCT,
                   dest="deployed_tp_pct",
                   help=f"Deployed TP band in percent (default "
                        f"{DEPLOYED_TP_PCT}%%). The verdict compares the "
                        f"best candidate against this band.")
    p.add_argument("--margin", type=float, default=EDGE_TOL_PP,
                   dest="edge_tol_pp",
                   help=f"Realized-return margin (pp) any candidate must "
                        f"clear to declare BAND_BEATS_DEPLOYED OR for the "
                        f"best candidate to clear absolute zero (default "
                        f"{EDGE_TOL_PP}pp).")
    p.add_argument("--min-buys", type=int, default=MIN_BUYS, dest="min_buys",
                   help=f"Minimum BUYs with intraperiod data before any "
                        f"verdict (default {MIN_BUYS}).")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of a table.")
    return p


def _parse_bands_arg(raw: str | None) -> tuple[float, ...]:
    """Parse the ``--bands 5,8,10`` CLI argument into a tuple of floats.
    Returns the default grid for ``None`` / empty input. Skips
    unparseable tokens silently — a typo in one band shouldn't crash
    the whole CLI."""
    if not raw:
        return DEFAULT_CANDIDATE_BANDS
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
    return tuple(out) if out else DEFAULT_CANDIDATE_BANDS


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Returns 0 on a decisive non-neutral verdict
    (BAND_BEATS_DEPLOYED / DEPLOYED_OPTIMAL), 1 on INSUFFICIENT_DATA
    or NO_BAND_HELPS — so shell callers can gate on `$?`. Mirrors
    stop_out_audit's return-code convention."""
    import sys

    args = _build_arg_parser().parse_args(
        sys.argv[1:] if argv is None else argv)
    bands = _parse_bands_arg(args.bands)
    rep = analyze(
        outcomes_path=args.outcomes,
        bands=bands,
        deployed_tp_pct=args.deployed_tp_pct,
        min_buys=args.min_buys,
        edge_tol_pp=args.edge_tol_pp,
    )

    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
        return 0 if rep["verdict"] in {
            "BAND_BEATS_DEPLOYED", "DEPLOYED_OPTIMAL",
        } else 1

    verdict = rep["verdict"]
    print(f"[tp_band_sweep] verdict={verdict}")
    print(f"  deployed_tp={rep['deployed_tp_pct']}%  edge_tol="
          f"±{rep['edge_tol_pp']}pp  min_buys={MIN_BUYS}")
    print(f"  n_buys={rep['n_buys']}  n_with_intraperiod="
          f"{rep['n_with_intraperiod']}")
    if rep.get("baseline_no_band_mean_pct") is not None:
        print(f"  baseline_no_band_mean={rep['baseline_no_band_mean_pct']:+.3f}pp")
    sweep = rep.get("sweep") or []
    if sweep:
        print(f"  band   n_trig  %_trig    mean_prot   benefit  ←-best→")
        for r in sweep:
            mark = "  *" if r["tp_pct"] == rep["best_band"]["tp_pct"] else ""
            dmark = "  [deployed]" if r["tp_pct"] == rep["deployed_tp_pct"] else ""
            print(f"  {r['tp_pct']:>4.1f}%  {r['n_triggered']:>5}  "
                  f"{r['pct_triggered']:>5.1f}%  "
                  f"{r['mean_protected_return_pct']:>+8.3f}pp  "
                  f"{r['benefit_pct']:>+7.3f}pp{mark}{dmark}")
    if rep.get("hint"):
        print(f"  hint: {rep['hint']}")
    return 0 if verdict in {"BAND_BEATS_DEPLOYED", "DEPLOYED_OPTIMAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

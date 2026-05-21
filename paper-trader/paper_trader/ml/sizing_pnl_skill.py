"""Sizing realized-PnL attribution — when ``_ml_decide`` sized a trade at
N% of book, did that bet actually contribute proportionally to realized
portfolio return?

This is a **read-only diagnostic**. It never trains, never touches
``decision_scorer.pkl``, ``decision_outcomes.jsonl``, ``build_features``,
``N_FEATURES``, or any trade path — same operational discipline as
``conviction_calibration`` / ``gate_pnl`` / ``calibration`` /
``consensus_skill`` / every other ``paper_trader/ml/*_skill.py`` module.
Safe to run against the live unattended loop, never raises on bad input.

**The gap this fills.** ``conviction_calibration`` already buckets BUY
outcomes by ``conviction_pct`` quantile and reports the rank-skill /
realized-return spread per bucket. That answers "does sizing rank
realized return?" — a per-trade rank question. ``gate_pnl`` measures the
×0.6..×1.3 multiplier overlay's portfolio-level impact in dollar terms.
Neither answers the actual portfolio-attribution question a quant
running a real book asks: *of the dollars the strategy made (or lost),
which conviction buckets contributed them?*

The portfolio's realized PnL per cycle is ``Σ conviction_pct_i ×
forward_return_5d_i`` (since ``conviction_pct`` IS the fraction of book
sized into trade *i*). Decomposing by conviction bucket:

* **per-trade dollar contribution** in bucket *b* = ``mean(conviction_pct
  × forward_return_5d)`` over rows in *b* — a sized-PnL per trade in pp
  of book.
* **aggregate share** = bucket-sum-of-products / total-sum-of-products —
  how much of the strategy's whole realized PnL came from this bucket.

These two numbers tell a skeptical quant whether the gate is making its
money on the rare high-conviction calls (good — concentrate edge) or
bleeding it on them (the documented latent danger: a leveraged-ETF cap
of 0.40 means one bad high-conviction call can dominate the cycle).

**Why a separate verdict ladder.** ``conviction_calibration`` correctly
verdicts on rank skill — bucket monotonicity of ``mean(forward_return_5d)``.
But a per-trade flat / mildly-inverted rank can still produce
spectacular *dollar* outcomes when scaled by conviction, and vice
versa. A high-conviction bucket realizing 5%/trade contributes 10×
more dollars than a low-conviction bucket realizing 8%/trade
(0.40×5=2.0 vs 0.05×8=0.4). The quant question for the *book* is the
dollar one; this module is its dedicated answer.

Verdict ladder (test-locked, exact-value):

| Verdict | Trigger |
|---|---|
| ``INSUFFICIENT_DATA`` | < ``MIN_ROWS`` BUY rows with finite ``conviction_pct`` AND ``forward_return_5d`` OR < ``BUCKET_MIN_ROWS`` rows in the top or bottom bucket |
| ``TOP_BUCKET_BLEEDS`` | top-conviction bucket mean dollar-PnL ≤ ``BLEED_PCT`` (the biggest bets are losing money — the gate is sizing into losers) |
| ``SIZING_INVERTED`` | top bucket mean dollar-PnL < bottom bucket mean dollar-PnL by ≥ ``INVERSION_PCT`` (the sizing rule is anti-skillful) |
| ``SIZING_PAYS`` | top bucket mean dollar-PnL > bottom bucket by ≥ ``EDGE_PCT`` AND top mean dollar-PnL > 0 (the gate concentrates wins) |
| ``BALANCED`` | spread within ``±FLAT_BAND_PCT`` of zero (neither dominates) |

CLI:

```bash
cd /home/zeph/trading-intelligence/paper-trader

# Default — analyze data/decision_outcomes.jsonl, table verdict
python3 -m paper_trader.ml.sizing_pnl_skill

# Machine-readable
python3 -m paper_trader.ml.sizing_pnl_skill --json

# Custom corpus
python3 -m paper_trader.ml.sizing_pnl_skill --outcomes path/to/alt.jsonl

# Configurable number of buckets (quartile by default)
python3 -m paper_trader.ml.sizing_pnl_skill --buckets 5
```

CLI exit code is 0 on every acceptable verdict (``SIZING_PAYS`` /
``BALANCED`` / ``INSUFFICIENT_DATA`` — informational), and 2 on
``TOP_BUCKET_BLEEDS`` / ``SIZING_INVERTED`` (the two quant-decisive
"the sizing rule is actively hurting the book" states) so a shell
caller can ``if !`` on a real edge — same gate-discipline as
``conviction_calibration`` / ``consensus_skill`` / ``gate_abstention``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

# Minimum total BUY rows with both fields finite before a verdict is
# attempted — guards against a thin corpus producing fake spreads.
MIN_ROWS = 60
# Minimum rows in each of the top and bottom bucket before the verdict
# is attempted. Quartile buckets have ~MIN_ROWS/N_BUCKETS rows each, so
# this should be small relative to MIN_ROWS / N_BUCKETS to leave
# headroom for sparse buckets.
BUCKET_MIN_ROWS = 8
# Default bucket count. Conviction varies modestly (most rows in 0.05..0.40
# range) so quartiles give meaningful resolution without empty edges.
N_BUCKETS_DEFAULT = 4

# Verdict-threshold percentage-points on mean(conviction_pct ×
# forward_return_5d). Units: percentage-points of book per trade. Realistic
# range: top bucket is rarely above |0.5pp| per trade in the live corpus.
# Thresholds are tuned to that scale.
BLEED_PCT = 0.0           # top bucket mean ≤ 0 → TOP_BUCKET_BLEEDS
INVERSION_PCT = 0.10      # bottom - top ≥ 0.10pp → SIZING_INVERTED
EDGE_PCT = 0.10           # top - bottom ≥ 0.10pp → SIZING_PAYS
FLAT_BAND_PCT = 0.05      # |top - bottom| ≤ 0.05pp → BALANCED

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


def _extract_pairs(rows: Iterable[dict]) -> list[tuple[float, float]]:
    """Pull (conviction_pct, forward_return_5d) pairs for BUY rows only.

    The conviction emission in ``_ml_decide`` is BUY-only (the gate is
    BUY-only). SELL / HOLD rows carry ``conviction_pct=None`` by
    construction — dropping them is the parsed-truth, not a fault.
    Non-finite or out-of-range values are dropped silently per the
    never-raises discipline.
    """
    out: list[tuple[float, float]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        action = str(r.get("action") or "").upper()
        if action != "BUY":
            continue
        cp = r.get("conviction_pct")
        fr = r.get("forward_return_5d")
        if not (_is_finite_number(cp) and _is_finite_number(fr)):
            continue
        cpf = float(cp)
        # Clamp to documented domain: conviction is a fraction in [0,1]
        # (the same clamp `_parse_conviction_pct` already applies). A row
        # outside that is malformed; drop rather than skew the bucket.
        if cpf < 0.0 or cpf > 1.0:
            continue
        out.append((cpf, float(fr)))
    return out


def _quantile_edges(values: list[float], n_buckets: int) -> list[float]:
    """Return n_buckets-1 quantile cut points on a sorted copy of values.

    Edge case: if all values are identical, returns [v]*(n-1) and the
    bucket assignment below collapses every row to bucket 0. The verdict
    then short-circuits to INSUFFICIENT_DATA via the BUCKET_MIN_ROWS guard.
    """
    if not values or n_buckets < 2:
        return []
    srt = sorted(values)
    n = len(srt)
    edges: list[float] = []
    for i in range(1, n_buckets):
        # Use the standard linear-interpolation quantile (numpy's "linear"
        # method), which is what every downstream visualization expects.
        q = i / n_buckets
        idx = q * (n - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            edges.append(srt[lo])
        else:
            frac = idx - lo
            edges.append(srt[lo] * (1 - frac) + srt[hi] * frac)
    return edges


def _assign_bucket(value: float, edges: list[float]) -> int:
    """Return the bucket index (0..n_buckets-1) for `value` given edges.

    Convention: bucket i contains rows with edges[i-1] < value ≤ edges[i].
    The lowest bucket (0) includes everything ≤ edges[0]; the highest
    bucket (n_buckets-1) includes everything > edges[-1]. This matches
    pandas / numpy ``digitize(right=True)`` semantics.
    """
    for i, e in enumerate(edges):
        if value <= e:
            return i
    return len(edges)


def sizing_report(rows: Iterable[dict],
                  n_buckets: int = N_BUCKETS_DEFAULT) -> dict:
    """Aggregate sizing-realized-PnL attribution per conviction bucket.

    Pure function — accepts an iterable of outcome dicts, returns a
    JSON-safe dict. No I/O, no train, no pickle touch.
    """
    if n_buckets < 2:
        n_buckets = 2
    pairs = _extract_pairs(rows)
    n_total = len(pairs)

    if n_total < MIN_ROWS:
        return {
            "status": "ok",
            "verdict": "INSUFFICIENT_DATA",
            "n": n_total,
            "n_buckets": n_buckets,
            "by_bucket": [],
            "spread_pct": None,
            "total_realized_pnl_pct": None,
            "top_bucket_share": None,
            "hint": f"need ≥{MIN_ROWS} BUY rows with conviction_pct + "
                    f"forward_return_5d; got {n_total}",
        }

    # Compute bucket edges from conviction distribution.
    conv_values = [c for c, _ in pairs]
    edges = _quantile_edges(conv_values, n_buckets)
    # Assign each pair to a bucket; accumulate stats.
    buckets: list[dict] = [{
        "bucket_idx": i,
        "conv_lower": None,
        "conv_upper": None,
        "n": 0,
        "mean_conv": None,
        "mean_fwd_ret": None,
        "mean_realized_pnl": None,
        "total_realized_pnl": 0.0,
        "share_of_total": None,
        "_conv_sum": 0.0,
        "_fr_sum": 0.0,
        "_prod_sum": 0.0,
    } for i in range(n_buckets)]

    grand_prod_sum = 0.0
    for c, r in pairs:
        b = _assign_bucket(c, edges)
        bkt = buckets[b]
        bkt["n"] += 1
        bkt["_conv_sum"] += c
        bkt["_fr_sum"] += r
        prod = c * r
        bkt["_prod_sum"] += prod
        grand_prod_sum += prod

    # Compute bucket edges (lower/upper bounds) for reporting.
    for i, bkt in enumerate(buckets):
        if i == 0:
            bkt["conv_lower"] = round(min(conv_values), 6) if conv_values else None
        else:
            bkt["conv_lower"] = round(edges[i - 1], 6)
        if i == n_buckets - 1:
            bkt["conv_upper"] = round(max(conv_values), 6) if conv_values else None
        else:
            bkt["conv_upper"] = round(edges[i], 6)

    # Finalize per-bucket stats.
    for bkt in buckets:
        n = bkt["n"]
        if n > 0:
            bkt["mean_conv"] = round(bkt["_conv_sum"] / n, 6)
            bkt["mean_fwd_ret"] = round(bkt["_fr_sum"] / n, 4)
            bkt["mean_realized_pnl"] = round(bkt["_prod_sum"] / n, 6)
            bkt["total_realized_pnl"] = round(bkt["_prod_sum"], 4)
            if abs(grand_prod_sum) > 1e-9:
                bkt["share_of_total"] = round(
                    bkt["_prod_sum"] / grand_prod_sum, 4)
        # Drop internal sums from output.
        for k in ("_conv_sum", "_fr_sum", "_prod_sum"):
            del bkt[k]

    # Verdict: compare top and bottom NON-EMPTY buckets. When the
    # conviction distribution is clustered (e.g. 95% of rows at 0.25), the
    # quantile cuts produce duplicate edges and some buckets end up empty.
    # The diagnostic should still verdict on the actual top vs bottom of
    # the realised distribution, not against an empty terminal bucket
    # that the cluster pushed everything below.
    non_empty = [b for b in buckets if b["n"] >= BUCKET_MIN_ROWS]
    if len(non_empty) >= 2:
        top = non_empty[-1]
        bot = non_empty[0]
    else:
        # Falls through to the bucket-too-small INSUFFICIENT_DATA branch
        # below; pick edge buckets so the hint still names a concrete count.
        top = buckets[-1]
        bot = buckets[0]
    top_mean = top["mean_realized_pnl"]
    bot_mean = bot["mean_realized_pnl"]

    if top["n"] < BUCKET_MIN_ROWS or bot["n"] < BUCKET_MIN_ROWS:
        verdict = "INSUFFICIENT_DATA"
        spread: float | None = None
        hint = (f"bucket size too small: top_n={top['n']}, "
                f"bot_n={bot['n']} (need ≥{BUCKET_MIN_ROWS} each)")
    elif top_mean is None or bot_mean is None:
        verdict = "INSUFFICIENT_DATA"
        spread = None
        hint = "bucket means unavailable"
    else:
        spread = round(top_mean - bot_mean, 6)
        if top_mean <= BLEED_PCT:
            verdict = "TOP_BUCKET_BLEEDS"
            hint = (f"top bucket mean realized PnL={top_mean:+.4f}pp "
                    f"≤ {BLEED_PCT}pp — largest bets lose money")
        elif spread <= -INVERSION_PCT:
            verdict = "SIZING_INVERTED"
            hint = (f"bottom > top: bottom={bot_mean:+.4f}pp vs "
                    f"top={top_mean:+.4f}pp (spread={spread:+.4f}pp)")
        elif spread >= EDGE_PCT:
            verdict = "SIZING_PAYS"
            hint = (f"top={top_mean:+.4f}pp vs bottom={bot_mean:+.4f}pp "
                    f"(spread={spread:+.4f}pp, top realizes more per trade)")
        elif abs(spread) <= FLAT_BAND_PCT:
            verdict = "BALANCED"
            hint = (f"spread={spread:+.4f}pp within ±{FLAT_BAND_PCT}pp "
                    f"flat band — sizing carries no dollar edge")
        elif spread > 0:
            verdict = "WEAK_EDGE"
            hint = (f"top - bottom = {spread:+.4f}pp — positive but below "
                    f"{EDGE_PCT}pp edge threshold")
        else:
            verdict = "WEAK_INVERSION"
            hint = (f"top - bottom = {spread:+.4f}pp — negative but above "
                    f"-{INVERSION_PCT}pp inversion threshold")

    # Top bucket's share of total dollar PnL — informational.
    top_share = top.get("share_of_total")

    return {
        "status": "ok",
        "verdict": verdict,
        "n": n_total,
        "n_buckets": n_buckets,
        "by_bucket": buckets,
        "spread_pct": spread,
        "total_realized_pnl_pct": round(grand_prod_sum, 4),
        "top_bucket_share": top_share,
        "hint": hint,
    }


def _load_outcomes(path: Path) -> list[dict]:
    """Best-effort line-by-line JSONL load. A corrupt line drops; a missing
    file returns []. Never raises."""
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


def analyze(outcomes_path: "Path | str | None" = None,
            n_buckets: int = N_BUCKETS_DEFAULT) -> dict:
    """High-level entry: load the outcomes JSONL and produce a verdict.

    Never raises — any fault degrades to a JSON-safe error sentinel
    matching ``sizing_report``'s INSUFFICIENT_DATA shape.
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
                "n": 0,
                "n_buckets": n_buckets,
                "by_bucket": [],
                "spread_pct": None,
                "total_realized_pnl_pct": None,
                "top_bucket_share": None,
                "hint": f"no outcomes loaded from {path}",
            }
        return sizing_report(rows, n_buckets=n_buckets)
    except Exception as exc:
        return {
            "status": "error",
            "verdict": "INSUFFICIENT_DATA",
            "n": 0,
            "n_buckets": n_buckets,
            "by_bucket": [],
            "spread_pct": None,
            "total_realized_pnl_pct": None,
            "top_bucket_share": None,
            "hint": f"analyze fault: {type(exc).__name__}",
        }


def _cli(argv: list[str] | None = None) -> int:
    """CLI entry. Exit 0 on every acceptable verdict (SIZING_PAYS /
    BALANCED / WEAK_EDGE / WEAK_INVERSION / INSUFFICIENT_DATA), exit 2
    on TOP_BUCKET_BLEEDS / SIZING_INVERTED (the quant-decisive
    "sizing rule is actively losing money" states)."""
    import argparse
    import sys

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.sizing_pnl_skill",
        description="Sizing realized-PnL attribution per conviction bucket.",
    )
    p.add_argument("--outcomes", default=None,
                   help="Path to decision_outcomes.jsonl "
                        "(default: data/decision_outcomes.jsonl).")
    p.add_argument("--buckets", type=int, default=N_BUCKETS_DEFAULT,
                   help=f"Number of conviction quantile buckets "
                        f"(default {N_BUCKETS_DEFAULT}).")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of a table.")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    rep = analyze(args.outcomes, n_buckets=args.buckets)

    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
    else:
        v = rep.get("verdict")
        print(f"VERDICT: {v}  ({rep.get('hint', '')})")
        print(f"  n_total: {rep.get('n')}  n_buckets: {rep.get('n_buckets')}")
        total = rep.get("total_realized_pnl_pct")
        if total is not None:
            print(f"  total realized PnL: {total:+.4f}pp of book "
                  f"(summed across all BUY trades)")
        share = rep.get("top_bucket_share")
        if share is not None:
            print(f"  top bucket share: {share*100:+.1f}% of total realized PnL")
        bb = rep.get("by_bucket") or []
        if bb:
            print(f"  {'bucket':<8}{'conv_range':<24}{'n':>6}"
                  f"{'mean_conv':>12}{'mean_fwd':>12}"
                  f"{'mean_PnL':>12}{'share':>8}")
            for b in bb:
                lo = b.get("conv_lower")
                hi = b.get("conv_upper")
                rng = (f"{lo:.3f}–{hi:.3f}" if lo is not None and hi is not None
                       else "n/a")
                mc = b.get("mean_conv")
                mf = b.get("mean_fwd_ret")
                mp = b.get("mean_realized_pnl")
                sh = b.get("share_of_total")
                mc_s = f"{mc:.4f}" if mc is not None else "n/a"
                mf_s = f"{mf:+.3f}%" if mf is not None else "n/a"
                mp_s = f"{mp:+.4f}pp" if mp is not None else "n/a"
                sh_s = f"{sh*100:+.1f}%" if sh is not None else "n/a"
                print(f"  {b.get('bucket_idx',0):<8}{rng:<24}{b.get('n',0):>6}"
                      f"{mc_s:>12}{mf_s:>12}{mp_s:>12}{sh_s:>8}")
        spread = rep.get("spread_pct")
        if spread is not None:
            print(f"  spread (top - bottom): {spread:+.4f}pp per-trade")

    v = rep.get("verdict")
    return 2 if v in ("TOP_BUCKET_BLEEDS", "SIZING_INVERTED") else 0


if __name__ == "__main__":
    raise SystemExit(_cli())

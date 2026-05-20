"""Holding-period × P/L distribution for closed round-trips.

Every realised-P&L panel on this desk reduces the round-trip set to a single
aggregate: ``track_record`` shows total return, ``trade_asymmetry`` reduces
to a disposition statistic, ``round_trip_postmortem`` scores exits, ``churn``
counts overtrading, ``winner_autopsy`` / ``loser_autopsy`` narrate per-trip
patterns. None of them answer the discretionary-PM question

    *Where in the holding-period axis do my gains and losses live?*

A book whose alpha comes from 8h overnight holds reads identical to one whose
alpha comes from 5-day swings — until you stratify P/L by hold duration. This
builder is that stratification: bucket every closed round-trip by hold hours,
roll up per-bucket count + total $ P/L + win rate + average $ P/L, then
identify the **alpha engine** (highest total $ P/L) and the **dominant
bucket** (most trips). The combination is operator-actionable: "63% of trips
are SCALP (<1h) but 91% of P/L comes from SWING (1-3d)" is the exact signal
that the book is over-trading the noisy bucket and under-allocating to the
profitable one.

Distinct from its neighbours (AGENTS.md invariant #10, do not consolidate):

* ``track_record`` is the aggregate ledger — no per-bucket strat.
* ``round_trip_postmortem`` is *post-exit drift* per trip — "was this exit
  well-timed?", not "how long was the hold and did it work?".
* ``trade_asymmetry`` is the disposition gap (win-hold vs loss-hold), not a
  per-bucket P/L map.
* ``loser_autopsy`` / ``winner_autopsy`` narrate individual trips — neither
  rolls up by duration band.

Pure builder over already-computed round-trips (the
``round_trips.build_round_trips`` shape). No DB read, no network. Never
raises on garbage rows (the ``_safe`` discipline — a malformed row degrades
the row, never the whole verdict). Observational only — never gates Opus,
never injected into the decision prompt (AGENTS.md invariants #2/#12).

Sample-size honesty mirrors ``round_trip_postmortem``: ``NO_DATA`` when
nothing closed; ``INSUFFICIENT`` (verdict withheld) below ``STABLE_MIN_TRIPS``;
``OK`` once stable. The per-bucket rows are emitted at every state so the
operator can see the distribution forming.
"""
from __future__ import annotations

from statistics import median
from typing import Any

# Hold-duration band edges (in hours, exclusive upper). Edges chosen so that
# a 1h boundary aligns with "did the position survive the open-volatility
# window", a 6h boundary aligns with "is this still intraday or did we hold
# through close", 24h aligns with "did we carry the overnight risk", and the
# multi-day bands give a coherent swing/trend/position taxonomy.
SCALP_MAX_HOURS = 1.0          # < 1h
INTRADAY_MAX_HOURS = 6.0       # 1h - 6h
OVERNIGHT_MAX_HOURS = 24.0     # 6h - 24h
SWING_MAX_HOURS = 72.0         # 24h - 72h (1d-3d)
TREND_MAX_HOURS = 168.0        # 72h - 168h (3d-7d)
# > 168h ⇒ POSITION

# Bucket order — used both as iteration order in outputs and as the canonical
# ordering when rendering tables. Operator reads scalp → position naturally.
BUCKETS: tuple[str, ...] = (
    "SCALP", "INTRADAY", "OVERNIGHT", "SWING", "TREND", "POSITION",
)

# Below this trip count the dominant / alpha-engine verdicts are withheld —
# a one-bucket "engine" off 2 trips is noise. Per-bucket rows still emit so
# the distribution can be watched accumulating.
STABLE_MIN_TRIPS = 5


def _bucket_for_hours(hours: float) -> str:
    if hours < SCALP_MAX_HOURS:
        return "SCALP"
    if hours < INTRADAY_MAX_HOURS:
        return "INTRADAY"
    if hours < OVERNIGHT_MAX_HOURS:
        return "OVERNIGHT"
    if hours < SWING_MAX_HOURS:
        return "SWING"
    if hours < TREND_MAX_HOURS:
        return "TREND"
    return "POSITION"


def _num(x: Any) -> float | None:
    """Coerce a scalar to float, dropping bools and NaN. Mirrors the
    ``_num`` helper in ``position_runrate`` so a single discipline guards
    every analytics builder."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    if x != x:  # NaN
        return None
    return float(x)


def _hold_hours(rt: dict) -> float | None:
    """A round-trip from ``build_round_trips`` already carries ``hold_days``
    (calendar days, computed off entry_ts → exit_ts). We multiply by 24 for
    hour-grained bucketing rather than re-parsing timestamps — single source
    of truth (AGENTS.md #10)."""
    days = _num(rt.get("hold_days"))
    if days is None or days < 0:
        return None
    return days * 24.0


def _empty_bucket(name: str) -> dict:
    return {
        "bucket": name,
        "n_trips": 0,
        "n_winners": 0,
        "n_losers": 0,
        "total_pnl_usd": 0.0,
        "total_pnl_usd_winners": 0.0,
        "total_pnl_usd_losers": 0.0,
        "avg_pnl_usd": None,
        "median_pnl_usd": None,
        "win_rate_pct": None,
        "share_of_trips_pct": None,
        "share_of_abs_pnl_pct": None,
    }


def build_holding_period_distribution(round_trips: list[dict]) -> dict:
    """Bucket closed round-trips by hold duration; roll up per-bucket P/L
    statistics; identify the alpha engine + dominant bucket.

    ``round_trips`` is the output shape of
    ``analytics.round_trips.build_round_trips``: each row carries
    ``ticker``, ``type``, ``pnl_usd``, ``pnl_pct``, ``hold_days``, etc.

    Returns a dict with:
      * ``state`` ∈ {NO_DATA, INSUFFICIENT, OK}
      * ``n_trips`` — total closed round-trips considered
      * ``n_unbucketed`` — rows that couldn't be placed (missing hold_days)
      * ``buckets`` — list of per-bucket rows in canonical order (SCALP→POSITION)
      * ``alpha_engine`` — the bucket name with the highest total_pnl_usd
        (None when state is INSUFFICIENT, NO_DATA, or all buckets are flat)
      * ``dominant_bucket`` — the bucket with the most trips (None below
        STABLE_MIN_TRIPS)
      * ``worst_bucket`` — the bucket with the most-negative total_pnl_usd
        (None when no losing bucket or below STABLE_MIN_TRIPS)
      * ``total_pnl_usd`` — sum across all bucketed trips
      * ``win_rate_pct`` — overall win rate across all bucketed trips
      * ``stable_min_trips`` — echoed STABLE_MIN_TRIPS so the UI can show
        "X more trips until verdict matures"
      * ``headline`` — the single-line summary the dashboard / chat /
        reporter all render verbatim
    """
    rows = round_trips or []
    if not isinstance(rows, list):
        rows = []

    bucket_rows: dict[str, dict] = {b: _empty_bucket(b) for b in BUCKETS}
    bucket_pnls: dict[str, list[float]] = {b: [] for b in BUCKETS}
    n_trips = 0
    n_unbucketed = 0
    total_pnl = 0.0
    total_wins = 0

    for rt in rows:
        if not isinstance(rt, dict):
            n_unbucketed += 1
            continue
        hours = _hold_hours(rt)
        pnl = _num(rt.get("pnl_usd"))
        if hours is None or pnl is None:
            n_unbucketed += 1
            continue
        bucket = _bucket_for_hours(hours)
        b = bucket_rows[bucket]
        b["n_trips"] += 1
        b["total_pnl_usd"] += pnl
        bucket_pnls[bucket].append(pnl)
        if pnl > 0:
            b["n_winners"] += 1
            b["total_pnl_usd_winners"] += pnl
            total_wins += 1
        elif pnl < 0:
            b["n_losers"] += 1
            b["total_pnl_usd_losers"] += pnl
        # pnl == 0.0 contributes to count but neither win nor loss tally.
        n_trips += 1
        total_pnl += pnl

    if n_trips == 0:
        return {
            "state": "NO_DATA",
            "n_trips": 0,
            "n_unbucketed": n_unbucketed,
            "buckets": [bucket_rows[b] for b in BUCKETS],
            "alpha_engine": None,
            "dominant_bucket": None,
            "worst_bucket": None,
            "total_pnl_usd": 0.0,
            "win_rate_pct": None,
            "stable_min_trips": STABLE_MIN_TRIPS,
            "headline": "no closed round-trips — holding-period distribution "
                        "not yet available.",
        }

    # Per-bucket finalize: averages, win rate, shares. The share-of-abs-pnl
    # denominator is the absolute-value sum across every bucketed trip,
    # never the signed sum — a signed-sum near zero would otherwise blow
    # the share fractions to nonsense.
    total_abs_pnl = sum(abs(p) for lst in bucket_pnls.values() for p in lst)
    for name in BUCKETS:
        b = bucket_rows[name]
        b["total_pnl_usd"] = round(b["total_pnl_usd"], 4)
        b["total_pnl_usd_winners"] = round(b["total_pnl_usd_winners"], 4)
        b["total_pnl_usd_losers"] = round(b["total_pnl_usd_losers"], 4)
        if b["n_trips"] > 0:
            b["avg_pnl_usd"] = round(b["total_pnl_usd"] / b["n_trips"], 4)
            b["median_pnl_usd"] = round(median(bucket_pnls[name]), 4)
            # Win rate excludes the zero-pnl trips from the numerator AND
            # denominator — same convention as analytics.track_record.
            decided = b["n_winners"] + b["n_losers"]
            if decided > 0:
                b["win_rate_pct"] = round(b["n_winners"] / decided * 100.0, 2)
            b["share_of_trips_pct"] = round(b["n_trips"] / n_trips * 100.0, 2)
            if total_abs_pnl > 1e-9:
                b["share_of_abs_pnl_pct"] = round(
                    abs(b["total_pnl_usd"]) / total_abs_pnl * 100.0, 2)
            else:
                b["share_of_abs_pnl_pct"] = 0.0

    overall_win_rate = (
        round(total_wins / n_trips * 100.0, 2) if n_trips > 0 else None
    )

    # State + cross-bucket verdicts. Below STABLE_MIN_TRIPS the bucket rows
    # are real but the engine / dominant labels are noise-driven — withhold.
    if n_trips < STABLE_MIN_TRIPS:
        state = "INSUFFICIENT"
        alpha_engine = dominant_bucket = worst_bucket = None
        headline = (
            f"{n_trips}/{STABLE_MIN_TRIPS} round-trips closed — "
            "holding-period verdict withheld until stable."
        )
    else:
        state = "OK"
        # Alpha engine: bucket with highest total_pnl_usd (must be > 0 to
        # be called an engine — a least-loss bucket isn't "alpha").
        engine_bucket = max(
            (bucket_rows[b] for b in BUCKETS),
            key=lambda r: r["total_pnl_usd"],
        )
        alpha_engine = (engine_bucket["bucket"]
                        if engine_bucket["total_pnl_usd"] > 0 else None)
        # Worst bucket: most-negative total_pnl_usd (only if some bucket
        # is actually negative).
        worst = min(
            (bucket_rows[b] for b in BUCKETS),
            key=lambda r: r["total_pnl_usd"],
        )
        worst_bucket = (worst["bucket"]
                        if worst["total_pnl_usd"] < 0 else None)
        # Dominant: bucket with the most trips. Ties resolved by canonical
        # bucket order (the BUCKETS tuple) so the choice is deterministic.
        most_trips = max(bucket_rows[b]["n_trips"] for b in BUCKETS)
        dominant_bucket = next(
            (b for b in BUCKETS if bucket_rows[b]["n_trips"] == most_trips),
            None,
        )
        # Headline — composed once, rendered verbatim by every caller.
        dom = bucket_rows[dominant_bucket] if dominant_bucket else None
        eng = bucket_rows[alpha_engine] if alpha_engine else None
        if eng is not None and dom is not None and eng["bucket"] != dom["bucket"]:
            headline = (
                f"{dom['bucket']} dominates ({dom['n_trips']}/{n_trips} trips, "
                f"{dom['share_of_trips_pct']:.0f}%) "
                f"but {eng['bucket']} is the engine "
                f"(${eng['total_pnl_usd']:+.2f} on {eng['n_trips']} trips, "
                f"{eng['share_of_abs_pnl_pct']:.0f}% of |P&L|)."
            )
        elif eng is not None:
            headline = (
                f"{eng['bucket']} is the alpha engine: "
                f"${eng['total_pnl_usd']:+.2f} on {eng['n_trips']} trips, "
                f"{eng['win_rate_pct']:.0f}% win rate, "
                f"{eng['share_of_abs_pnl_pct']:.0f}% of |P&L|."
            )
        elif worst_bucket is not None and dom is not None:
            w = bucket_rows[worst_bucket]
            headline = (
                f"No bucket is net positive yet — "
                f"{w['bucket']} is the bleed (${w['total_pnl_usd']:+.2f} on "
                f"{w['n_trips']} trips); "
                f"{dom['bucket']} sees the most trips "
                f"({dom['n_trips']}/{n_trips})."
            )
        else:
            headline = (
                f"{n_trips} trips closed; book is flat across the "
                "holding-period spectrum."
            )

    return {
        "state": state,
        "n_trips": n_trips,
        "n_unbucketed": n_unbucketed,
        "buckets": [bucket_rows[b] for b in BUCKETS],
        "alpha_engine": alpha_engine,
        "dominant_bucket": dominant_bucket,
        "worst_bucket": worst_bucket,
        "total_pnl_usd": round(total_pnl, 4),
        "win_rate_pct": overall_win_rate,
        "stable_min_trips": STABLE_MIN_TRIPS,
        "headline": headline,
    }

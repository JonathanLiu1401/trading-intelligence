"""Per-position alpha vs beta-implied SPY return — *which positions show edge?*

``/api/portfolio-beta`` reports portfolio-level beta + alpha but only after 20
paired daily returns accumulate (12/20 on the live $1000 book as of 2026-05-30,
i.e. INSUFFICIENT verdict for weeks). ``/api/risk`` lists per-position beta but
never decomposes the *realized* return into the beta-explained portion vs the
residual. A 5%-up position on a 4% SPY day at beta 1.5 is **negative alpha**
(beta-implied 6.0%, observed 5.0%, alpha -1.0pp) but every dashboard panel
celebrates it as a 5% winner.

This per-position cut answers the desk question: *of my open positions, which
are pulling real weight vs which are just along for the SPY ride?* For each
open position:

* ``pos_return_pct`` — mark-to-market vs avg cost since open (the existing
  ``pl_pct`` semantics, computed identically).
* ``spy_return_pct`` — SPY move from the first equity-curve point at-or-after
  the position's opened_at to the most-recent point.
* ``beta_est`` — sector→beta from the shared ``_LEVERAGE_BETA`` SSOT (the
  ``/api/risk`` + ``/api/stress-scenarios`` table). Options inherit
  ×3-capped-at-4, negated for puts — same contract as ``stress_scenarios``.
* ``pure_beta_pct = beta_est * spy_return_pct`` — what beta-alone would have
  earned over the hold period.
* ``alpha_pp = pos_return_pct - pure_beta_pct`` — the residual after stripping
  market beta. Positive ⇒ idiosyncratic edge; negative ⇒ underperforming the
  beta-implied baseline.

Per-position verdict: ``ALPHA_POS`` / ``ALPHA_NEG`` / ``PURE_BETA`` (within
``ALPHA_BAND_PP=0.5`` of zero) / ``INSUFFICIENT_SPY_DATA`` (no equity-curve
point covers opened_at). Aggregate verdict mirrors: ``ALPHA_ADDING`` if the
market-value-weighted mean alpha exceeds the band, ``ALPHA_BLEEDING`` if it
falls below, ``BETA_RIDING`` if within band, ``NO_DATA`` otherwise.

**Single source of truth.** Uses the exact ``classify`` (ticker→sector) +
``beta_map`` (sector→beta) the caller passes — the dashboard side always
supplies ``_classify`` + ``_LEVERAGE_BETA`` (the ``/api/risk`` SSOT). The
opened_at→spy_at_open mapping uses ``equity_curve`` directly so the SPY
benchmark is exactly what the dashboard recorded (no yfinance round-trip, no
weekend / off-hours disagreement). The intraday-cadence book is the right
benchmark — daily-resampled SPY would force a stale match for positions opened
mid-day.

**Observational, never prescriptive.** Same contract as
``stress_scenarios`` / ``risk_mirror`` (AGENTS.md invariants #2 / #12). It
states facts. It issues no directive, imposes no cap, and never gates a trade.

Pure and deterministic (no clock unless ``now`` is unset, no IO). Never raises —
a garbage row contributes ``INSUFFICIENT_SPY_DATA`` and the aggregate degrades
gracefully.
"""
from __future__ import annotations

from datetime import datetime, timezone

# Alpha band — within ±this pp the position is graded PURE_BETA (no decisive
# edge call). Tight because the live book is a $1000 paper portfolio — a 0.5pp
# alpha on a $1000 book is $5, the floor where the residual is just
# noise / bid-ask / mark-to-mark microstructure rather than real selection
# skill. Tests read this constant so a retune can't false-fail them.
ALPHA_BAND_PP = 0.5

# Maximum lag between opened_at and the equity-curve point we accept as
# spy_at_open. The live runner records an equity point every cycle (60s when
# market open, 3600s when closed); we tolerate up to 2h so a market-closed
# open still finds a baseline. Beyond that the SPY anchor is too far from the
# fill — INSUFFICIENT_SPY_DATA. Tests pin this.
MAX_OPEN_LAG_S = 7200.0


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _safe_float(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    # Reject NaN / inf — they propagate to the JSON and break clients.
    if f != f or f in (float("inf"), -float("inf")):
        return None
    return f


def _z(v: float | None, ndigits: int = 4) -> float | None:
    """Round, folding -0.0 → 0.0 (mirrors stress_scenarios._z)."""
    if v is None:
        return None
    try:
        r = round(float(v), ndigits)
    except (TypeError, ValueError):
        return None
    return 0.0 if r == 0 else r


def _position_beta(ptype: str, sector: str, beta_map: dict) -> float:
    """Per-position beta with options ×3-capped-at-4 / negated for puts —
    identical to ``stress_scenarios._position_betas`` so the two endpoints
    agree on the same row."""
    beta = float(beta_map.get(sector, 1.0))
    if ptype in ("call", "put"):
        beta = min(beta * 3.0, 4.0)
        if ptype == "put":
            beta = -beta
    return beta


def _spy_at_open(opened_at: datetime,
                 equity_curve: list[dict]) -> tuple[float | None,
                                                    str | None,
                                                    float | None]:
    """Return (spy_price, ts_iso, lag_seconds) for the first equity-curve point
    at-or-after opened_at, or (None, None, None) if none within MAX_OPEN_LAG_S.

    ``equity_curve`` is ascending by timestamp (as ``store.equity_curve``
    returns). A linear scan is fine — the live book has < 1000 points; the
    cost is microseconds vs the analytics endpoint's overall budget.
    """
    if not equity_curve:
        return (None, None, None)
    for pt in equity_curve:
        ts = _parse_ts(pt.get("timestamp"))
        if ts is None or ts < opened_at:
            continue
        lag = (ts - opened_at).total_seconds()
        if lag > MAX_OPEN_LAG_S:
            return (None, None, None)
        sp = _safe_float(pt.get("sp500_price"))
        if sp is None or sp <= 0:
            return (None, None, None)
        return (sp, ts.isoformat(timespec="seconds"), lag)
    # Position opened after the newest equity point — too fresh to judge.
    return (None, None, None)


def _spy_now(equity_curve: list[dict]) -> tuple[float | None, str | None]:
    """Latest non-null SPY price + its timestamp (newest in curve)."""
    for pt in reversed(equity_curve or []):
        sp = _safe_float(pt.get("sp500_price"))
        if sp is None or sp <= 0:
            continue
        ts = pt.get("timestamp")
        return (sp, ts)
    return (None, None)


def _position_market_value(p: dict) -> float:
    """Mark-to-market notional, falling back avg_cost→0 and ×100 for options —
    identical to ``stress_scenarios._position_betas``."""
    try:
        ptype = (p.get("type") or "stock").lower()
        mult = 100 if ptype in ("call", "put") else 1
        price = _safe_float(p.get("current_price"))
        if price is None:
            price = _safe_float(p.get("avg_cost")) or 0.0
        qty = _safe_float(p.get("qty")) or 0.0
        return float(price) * qty * mult
    except Exception:
        return 0.0


def build_position_alpha_decomp(
    positions: list[dict],
    equity_curve: list[dict],
    classify,
    beta_map: dict,
    now: datetime | None = None,
) -> dict:
    """Per-position alpha vs beta-implied SPY return decomposition.

    ``positions`` is the open-position list (``store.open_positions``);
    ``equity_curve`` is ascending (``store.equity_curve``). ``classify`` is the
    dashboard/strategy ticker→sector classifier; ``beta_map`` is the shared
    sector→beta table (the ``_LEVERAGE_BETA`` SSOT). Pure, deterministic, never
    raises.
    """
    now = now or datetime.now(timezone.utc)
    out: dict = {
        "as_of": now.isoformat(timespec="seconds"),
        "n_positions": 0,
        "n_judged": 0,
        "positions": [],
        "weighted_alpha_pp": None,
        "spy_return_pct_now": None,
        "alpha_band_pp": ALPHA_BAND_PP,
        "max_open_lag_s": MAX_OPEN_LAG_S,
        "state": "NO_DATA",
        "verdict": "NO_DATA",
        "headline": "No open positions to decompose.",
    }

    if not positions:
        return out

    spy_now_val, spy_now_ts = _spy_now(equity_curve)
    out["spy_now_ts"] = spy_now_ts

    judged: list[dict] = []
    insufficient: list[dict] = []
    for p in positions:
        ticker = p.get("ticker") or ""
        ptype = (p.get("type") or "stock").lower()
        sector = "other"
        try:
            sector = classify(ticker) if classify else "other"
        except Exception:
            sector = "other"
        beta = _position_beta(ptype, sector, beta_map or {})

        avg = _safe_float(p.get("avg_cost"))
        cur = _safe_float(p.get("current_price"))
        opened_at = _parse_ts(p.get("opened_at"))
        market_value = _position_market_value(p)

        # pos_return_pct — mark-to-market vs avg cost. Puts/short positions
        # already have negated beta; the return % itself is computed
        # straight from the option's premium ticks (current_price vs
        # avg_cost). Same convention as dashboard.py's pl_pct.
        pos_return = None
        if avg is not None and avg > 0 and cur is not None:
            pos_return = (cur / avg - 1.0) * 100.0

        row: dict = {
            "ticker": ticker,
            "type": ptype,
            "sector": sector,
            "beta_est": _z(beta, 2),
            "qty": _safe_float(p.get("qty")),
            "avg_cost": _z(avg, 4) if avg is not None else None,
            "current_price": _z(cur, 4) if cur is not None else None,
            "market_value": _z(market_value, 2),
            "pos_return_pct": _z(pos_return, 4) if pos_return is not None
                else None,
            "spy_at_open": None,
            "spy_at_open_ts": None,
            "spy_open_lag_s": None,
            "spy_return_pct": None,
            "pure_beta_pct": None,
            "alpha_pp": None,
            "hold_seconds": None,
            "verdict": "INSUFFICIENT_SPY_DATA",
        }

        if (opened_at is None or spy_now_val is None
                or pos_return is None):
            insufficient.append(row)
            continue

        spy_at, spy_at_ts, lag = _spy_at_open(opened_at, equity_curve)
        if spy_at is None:
            insufficient.append(row)
            continue

        spy_return = (spy_now_val / spy_at - 1.0) * 100.0
        pure_beta = beta * spy_return
        alpha = pos_return - pure_beta

        # Per-position verdict — same band on both sides; equal on the
        # boundary stays PURE_BETA (conservative; a band at the noise floor
        # shouldn't be tipped by a 0.001pp arithmetic artifact).
        if alpha > ALPHA_BAND_PP:
            verdict = "ALPHA_POS"
        elif alpha < -ALPHA_BAND_PP:
            verdict = "ALPHA_NEG"
        else:
            verdict = "PURE_BETA"

        row.update({
            "spy_at_open": _z(spy_at, 4),
            "spy_at_open_ts": spy_at_ts,
            "spy_open_lag_s": _z(lag, 1),
            "spy_return_pct": _z(spy_return, 4),
            "pure_beta_pct": _z(pure_beta, 4),
            "alpha_pp": _z(alpha, 4),
            "hold_seconds": _z((now - opened_at).total_seconds(), 1),
            "verdict": verdict,
        })
        judged.append(row)

    # Order: judged first (largest market value first — operator scans
    # heaviest position at the top), then insufficient (same convention).
    judged.sort(key=lambda r: -(r["market_value"] or 0))
    insufficient.sort(key=lambda r: -(r["market_value"] or 0))
    all_rows = judged + insufficient

    out["positions"] = all_rows
    out["n_positions"] = len(all_rows)
    out["n_judged"] = len(judged)

    if not judged:
        out["state"] = "NO_DATA"
        out["verdict"] = "NO_DATA"
        if positions:
            out["headline"] = (
                f"INSUFFICIENT — {len(positions)} open position(s) but no SPY "
                f"baseline within {MAX_OPEN_LAG_S/3600:.1f}h of any open.")
        return out

    # Market-value-weighted mean alpha. Weights from the judged rows only —
    # an INSUFFICIENT_SPY_DATA position can't contribute (its alpha is
    # undefined) but the weight skew is documented in the n_judged field so
    # the operator sees how many positions actually drove the aggregate.
    total_w = sum(r["market_value"] or 0 for r in judged)
    if total_w <= 0:
        # All judged rows have zero notional — degrade to simple mean.
        weighted = sum(r["alpha_pp"] or 0 for r in judged) / len(judged)
    else:
        weighted = sum(
            (r["alpha_pp"] or 0) * (r["market_value"] or 0)
            for r in judged) / total_w

    out["weighted_alpha_pp"] = _z(weighted, 4)
    # Aggregate SPY return — use the first judged row's spy_return_pct since
    # they all share the same spy_now_val; differences only come from each
    # position's individual spy_at_open. We surface the *median* across
    # judged positions so a single old position can't dominate the headline.
    spy_returns = sorted(r["spy_return_pct"] for r in judged
                         if r["spy_return_pct"] is not None)
    if spy_returns:
        mid = len(spy_returns) // 2
        if len(spy_returns) % 2 == 1:
            med = spy_returns[mid]
        else:
            med = (spy_returns[mid - 1] + spy_returns[mid]) / 2.0
        out["spy_return_pct_now"] = _z(med, 4)

    if weighted > ALPHA_BAND_PP:
        verdict = "ALPHA_ADDING"
    elif weighted < -ALPHA_BAND_PP:
        verdict = "ALPHA_BLEEDING"
    else:
        verdict = "BETA_RIDING"
    out["state"] = "OK"
    out["verdict"] = verdict

    sign = "+" if weighted >= 0 else ""
    out["headline"] = (
        f"{verdict} — {len(judged)} of {len(all_rows)} position(s) judged, "
        f"weighted alpha {sign}{weighted:.2f}pp vs beta-implied "
        f"(median SPY {out['spy_return_pct_now']}% over hold).")
    return out

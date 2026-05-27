"""Leverage classifier — splits the current book and the scorer-opportunity
slate by leverage factor (1x, 2x, 3x, -1x, -2x, -3x).

``/api/regime-leverage-fit-skill`` already reports the book's leveraged %
as a single number (any name in ``_LEVERAGED_ETFS_LIVE`` counts), but it
collapses long-leveraged and inverse-leveraged into one bucket and never
looks at the *opportunity slate* the scorer is publishing. A trader
deciding whether to redeploy needs the factor breakdown to size correctly
and to see whether the slate's direction-mix matches the regime (a bull
tape that's served by 5/6 inverse-3x opportunities is a contradiction the
single-number verdict can't surface).

This module is the breakdown those callers need. It maps each known
ticker to a ``(direction, factor)`` pair and aggregates dollars (book) or
counts (slate) by factor band.

Pure and deterministic — no clock, no IO, never raises. The module-level
factor map is the SSOT; ``_LEVERAGED_ETFS_LIVE`` in ``strategy.py`` is the
membership set we mirror (the test suite pins the two together).
"""
from __future__ import annotations

from datetime import datetime, timezone

# ── Leverage factor map. Keys mirror strategy._LEVERAGED_ETFS_LIVE
#    (membership tested in test_leverage_exposure). Values are the
#    *signed* leverage factor a 1% move in the underlying produces in
#    the ETF: +3 for long 3x, -3 for inverse 3x, +2 for long 2x, etc.
LEVERAGE_FACTOR: dict[str, int] = {
    # Broad-market 3x long
    "TQQQ": 3, "UPRO": 3, "SPXL": 3, "UDOW": 3, "URTY": 3, "TNA": 3,
    # Broad-market 3x inverse
    "SQQQ": -3, "SPXS": -3,
    # Semis 3x
    "SOXL": 3, "SOXS": -3,
    # Tech / sector 3x long
    "TECL": 3, "FNGU": 3, "CURE": 3, "LABU": 3, "NAIL": 3, "DPST": 3,
    "FAS": 3, "DFEN": 3, "UTSL": 3,
    # Tech / sector 3x inverse
    "TECS": -3, "FNGD": -3,
    # Single-stock 2x long (Direxion / GraniteShares family)
    "NVDU": 2, "MSFU": 2, "AMZU": 2, "TSLL": 2, "CONL": 2,
    "BITU": 2, "ETHU": 2,
    # Broad-market 2x long
    "QLD": 2, "SSO": 2,
}

#: Per-side caps a deployment planner can consume. Total leverage of the
#: book/plan should stay under ``MAX_LEVERAGED_PCT``; if you allow inverse
#: at all, keep it under ``MAX_INVERSE_PCT`` so a regime hedge can't
#: silently dominate.
MAX_LEVERAGED_PCT = 30.0
MAX_INVERSE_PCT = 15.0


def _utcnow_iso(now: datetime | None) -> str:
    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.isoformat(timespec="seconds")


def classify(ticker) -> dict:
    """Return ``{'factor': int, 'direction': 'long'|'inverse'|'unlev'}``.

    Unknown tickers are treated as 1x long. Non-string input degrades to
    1x long (never raises)."""
    try:
        sym = str(ticker).upper()
    except Exception:
        return {"factor": 1, "direction": "long"}
    fac = LEVERAGE_FACTOR.get(sym, 1)
    if fac == 1:
        return {"factor": 1, "direction": "long"}
    return {"factor": fac, "direction": "inverse" if fac < 0 else "long"}


def _position_value(p: dict) -> float:
    """Mirrors concentration_cap._position_value — prefers ``market_value``,
    falls back to ``current_price * qty * mult``."""
    mv = p.get("market_value")
    if mv is not None:
        try:
            return float(mv)
        except (TypeError, ValueError):
            pass
    try:
        ptype = p.get("type") or "stock"
        mult = 100 if ptype in ("call", "put") else 1
        price = p.get("current_price") or p.get("avg_cost") or 0.0
        qty = float(p.get("qty") or 0)
        return float(price) * qty * mult
    except (TypeError, ValueError):
        return 0.0


def _z(v, ndigits: int = 2):
    if v is None:
        return None
    try:
        r = round(float(v), ndigits)
    except (TypeError, ValueError):
        return None
    return 0.0 if r == 0 else r


def _classify_book(positions: list[dict], total_value: float) -> dict:
    """Aggregate held dollars by factor band. Returns:

      {
        "by_factor": {"+3": {...}, "+2": {...}, "-3": {...}, "1": {...}},
        "by_direction": {"long_lev": pct, "inverse_lev": pct, "unlev": pct},
        "n_positions": int,
        "total_value_usd": float,
      }
    """
    by_factor: dict[str, dict] = {}
    long_lev_usd = 0.0
    inverse_lev_usd = 0.0
    unlev_usd = 0.0
    for p in positions or []:
        sym = str(p.get("ticker") or "").upper()
        info = classify(sym)
        f = info["factor"]
        usd = _position_value(p)
        key = f"{f:+d}" if abs(f) != 1 else "1"
        bucket = by_factor.setdefault(key, {"usd": 0.0, "tickers": []})
        bucket["usd"] += usd
        if sym not in bucket["tickers"]:
            bucket["tickers"].append(sym)
        if f == 1:
            unlev_usd += usd
        elif f > 0:
            long_lev_usd += usd
        else:
            inverse_lev_usd += usd
    tv = max(float(total_value or 0.0), 1e-9)
    for k in by_factor:
        by_factor[k]["pct"] = _z(by_factor[k]["usd"] / tv * 100)
        by_factor[k]["usd"] = _z(by_factor[k]["usd"])
        by_factor[k]["tickers"] = sorted(by_factor[k]["tickers"])
    return {
        "by_factor": by_factor,
        "by_direction": {
            "long_lev_pct": _z(long_lev_usd / tv * 100),
            "inverse_lev_pct": _z(inverse_lev_usd / tv * 100),
            "unlev_pct": _z(unlev_usd / tv * 100),
        },
        "n_positions": len([p for p in (positions or []) if p]),
        "total_value_usd": _z(total_value),
    }


def _classify_slate(opportunities: list[dict]) -> dict:
    """Aggregate the scorer slate by factor band. We aggregate *count* and
    *weighted pred_5d_return_pct* (weight=count) because opportunities
    haven't been sized yet — there's no $ to bucket. Returns:

      {
        "by_factor": {"+3": {n, pred_5d_avg, tickers}, ...},
        "by_direction": {"long_lev_pct", "inverse_lev_pct", "unlev_pct"},
        "n_opportunities": int,
        "blended_pred_5d_return_pct": float | None,  # equal-weighted across the slate
      }
    """
    by_factor: dict[str, dict] = {}
    long_lev_n = 0
    inverse_lev_n = 0
    unlev_n = 0
    preds_all: list[float] = []
    for o in opportunities or []:
        sym = str(o.get("ticker") or "").upper()
        if not sym:
            continue
        info = classify(sym)
        f = info["factor"]
        try:
            pred = float(o.get("pred_5d_return_pct") or 0.0)
        except (TypeError, ValueError):
            pred = 0.0
        preds_all.append(pred)
        key = f"{f:+d}" if abs(f) != 1 else "1"
        bucket = by_factor.setdefault(key, {"n": 0, "_preds": [], "tickers": []})
        bucket["n"] += 1
        bucket["_preds"].append(pred)
        if sym not in bucket["tickers"]:
            bucket["tickers"].append(sym)
        if f == 1:
            unlev_n += 1
        elif f > 0:
            long_lev_n += 1
        else:
            inverse_lev_n += 1
    total_n = max(unlev_n + long_lev_n + inverse_lev_n, 1)
    for k, b in by_factor.items():
        preds = b.pop("_preds")
        b["pred_5d_avg_pct"] = _z(sum(preds) / len(preds)) if preds else None
        b["tickers"] = sorted(b["tickers"])
    return {
        "by_factor": by_factor,
        "by_direction": {
            "long_lev_pct": _z(long_lev_n / total_n * 100),
            "inverse_lev_pct": _z(inverse_lev_n / total_n * 100),
            "unlev_pct": _z(unlev_n / total_n * 100),
        },
        "n_opportunities": unlev_n + long_lev_n + inverse_lev_n,
        "blended_pred_5d_return_pct": _z(sum(preds_all) / len(preds_all)) if preds_all else None,
    }


def _verdict(book_lev_pct: float, slate_lev_pct: float,
             n_opportunities: int, regime: str) -> tuple[str, str]:
    """Return (verdict, headline). Verdict labels:

      * ALIGNED   — book leverage in the regime-implied band, or slate
                    is unleveraged and the book matches
      * UNDER_LEV — book is at unlev floor in a bull tape with a
                    long-leveraged slate available
      * OVER_LEV  — book exceeds the leveraged ceiling, OR book is
                    leveraged in a bear tape
      * NO_SLATE  — no opportunities to compare against at all
    """
    if n_opportunities == 0:
        return "NO_SLATE", "No opportunities on the slate."
    reg = (regime or "").lower()
    if book_lev_pct > MAX_LEVERAGED_PCT:
        return ("OVER_LEV",
                f"book leveraged {book_lev_pct:.1f}% > cap {MAX_LEVERAGED_PCT:.0f}%")
    if reg == "bull" and book_lev_pct < 5.0 and slate_lev_pct >= 50.0:
        return ("UNDER_LEV",
                f"bull tape, slate {slate_lev_pct:.0f}% leveraged, book "
                f"only {book_lev_pct:.1f}% leveraged")
    if reg == "bear" and book_lev_pct > 10.0:
        return ("OVER_LEV",
                f"bear tape but book is {book_lev_pct:.1f}% leveraged")
    return ("ALIGNED",
            f"book leverage {book_lev_pct:.1f}%, slate {slate_lev_pct:.0f}% "
            f"leveraged ({reg or 'unknown'} regime)")


def build_leverage_exposure(
    positions: list[dict] | None,
    total_value: float | None,
    opportunities: list[dict] | None,
    regime: str | None = None,
    now: datetime | None = None,
) -> dict:
    """Pure: classify book + slate by leverage factor. Returns a JSON-ready
    dict with ``current_book``, ``opportunity_slate``, ``regime``,
    ``verdict``, ``headline``, ``thresholds``, and ``as_of``.

    Inputs:

      * ``positions`` — list of held positions (same shape as
        ``store.open_positions()``); ``None`` or ``[]`` is allowed.
      * ``total_value`` — total portfolio value in $ (used as denominator
        for book percentages).
      * ``opportunities`` — list of scorer-opportunity rows (the dicts
        served by ``/api/scorer-opportunities``); each must have
        ``ticker`` and ``pred_5d_return_pct``.
      * ``regime`` — current regime string ("bull", "sideways", "bear"
        or ``None``). Used only to choose the verdict band.
    """
    book = _classify_book(positions or [], float(total_value or 0.0))
    slate = _classify_slate(opportunities or [])
    book_lev = (book["by_direction"]["long_lev_pct"] or 0.0) + \
               (book["by_direction"]["inverse_lev_pct"] or 0.0)
    slate_lev = (slate["by_direction"]["long_lev_pct"] or 0.0) + \
                (slate["by_direction"]["inverse_lev_pct"] or 0.0)
    verdict, headline = _verdict(book_lev, slate_lev,
                                  slate["n_opportunities"], regime or "")
    return {
        "as_of": _utcnow_iso(now),
        "verdict": verdict,
        "headline": headline,
        "regime": (regime or "unknown").lower(),
        "current_book": book,
        "opportunity_slate": slate,
        "thresholds": {
            "max_leveraged_pct": MAX_LEVERAGED_PCT,
            "max_inverse_pct": MAX_INVERSE_PCT,
        },
    }

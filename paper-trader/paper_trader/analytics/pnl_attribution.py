"""Beta-adjusted P/L attribution for open stock positions.

``analytics/open_attribution.py`` (``/api/open-attribution``) is the desk's
selection-vs-market decomposition — but it implicitly assumes **β=1**:
``alpha_pct = position_return − spy_return``. For a market-beta stock that
read is fine. For the bot's documented book — heavy in **β=3** leveraged
ETFs (TQQQ, SOXL, FNGU) and β=1.5 semis names (NVDA, AMD) — it
*systematically over-attributes* gain/loss to "alpha" when most of the
move is actually leveraged β-exposure. A +1% SPY day on a $200 TQQQ
position should explain ≈$6 of P/L (β=3 × 1% × $200), not $2.

This builder decomposes per-position unrealized P/L into:

    beta_explained_pct = β × spy_return_pct
    idiosyncratic_pct  = position_return_pct − beta_explained_pct

…and dollarizes both. The honest answer to "is my NVDA gain just SPY
going up?" is the idiosyncratic figure, not the naïve position−SPY.

Composes the same SPY-anchor logic as ``open_attribution`` (the equity
curve's ``sp500_price`` column read at-or-after ``opened_at``) — single
source of truth (AGENTS.md #10), no re-derived series. Uses the **same**
``classify`` + ``beta_map`` SSOT as ``stress_scenarios`` /
``/api/risk`` so an unknown-sector ticker reads β=1.0 *exactly* the
same as those panels do; a drift in either layer would fail the
cross-fold check.

Options are flagged and skipped (the ``open_attribution`` precedent —
β-attribution on an options Greek instrument is its own surface, see
``/api/greeks``). Pure: no I/O, never raises (the ``_safe`` contract —
a garbage row contributes nothing, a missing equity curve degrades to
``NO_BENCHMARK``).

State ladder mirrors the sibling builders:

* ``NO_DATA``      — no positions / all skipped
* ``NO_BENCHMARK`` — positions exist but the equity curve has no
  ``sp500_price`` history (cold-start before the first equity tick).
* ``INSUFFICIENT`` — positions exist and benchmark exists but no
  position's ``opened_at`` anchors against the available SPY history
  (e.g. all positions were opened *before* the first equity tick).
* ``OK``           — at least one anchored row with β-attribution.

Observational / advisory only — never gates Opus, never injected into
the decision prompt, no caps (invariants #2/#12 — the
``open_attribution`` precedent).
"""
from __future__ import annotations

from datetime import datetime, timezone

_OPTION_TYPES = {"call", "put"}


def _z(v: float | None, ndigits: int = 2) -> float | None:
    """Round; fold -0.0 → 0.0 (the ``open_attribution`` precedent — same
    shape, same contract). A non-numeric / None input degrades to ``None``,
    never raises."""
    if v is None:
        return None
    try:
        r = round(float(v), ndigits)
    except (TypeError, ValueError):
        return None
    return 0.0 if r == 0 else r


def _parse_ts(ts) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _spy_series(equity_curve) -> list[tuple[datetime, float]]:
    """Ascending [(ts, sp500_price)] with parseable ts and positive price.

    Byte-identical to ``open_attribution._spy_series`` semantics so the
    two builders cannot disagree on the SPY anchor (single source of
    truth, AGENTS.md #10)."""
    out: list[tuple[datetime, float]] = []
    for row in equity_curve or []:
        if not isinstance(row, dict):
            continue
        ts = _parse_ts(row.get("timestamp"))
        try:
            px = row.get("sp500_price")
            if ts is None or px is None:
                continue
            fpx = float(px)
        except (TypeError, ValueError):
            continue
        if fpx <= 0:
            continue
        out.append((ts, fpx))
    out.sort(key=lambda r: r[0])
    return out


def _spy_at_or_after(series, ts) -> float | None:
    """First S&P level at or after ``ts`` — the entry anchor."""
    for t, px in series:
        if t >= ts:
            return px
    return None


def build_pnl_attribution(positions: list[dict],
                          equity_curve: list[dict],
                          classify,
                          beta_map: dict,
                          now: datetime | None = None) -> dict:
    """Pure: no I/O, never raises. ``positions`` is the
    ``store.open_positions()`` shape. ``equity_curve`` is
    ``store.equity_curve()`` (rows with ``timestamp`` and ``sp500_price``).
    ``classify`` + ``beta_map`` are the dashboard/stress_scenarios SSOT
    (ticker→sector, sector→beta — call with the *real* dashboard
    objects to inherit the true SSOT; the strategy-side pinned copies
    will also pass)."""
    now = now or datetime.now(timezone.utc)
    series = _spy_series(equity_curve)
    now_spy = series[-1][1] if series else None

    base: dict = {
        "as_of": now.isoformat(timespec="seconds"),
        "n_positions": 0,
        "n_anchored": 0,
        "rows": [],
        "skipped": [],
        "totals": None,
        "headline": None,
    }

    rows: list[dict] = []
    skipped: list[dict] = []
    tot_cost = 0.0
    tot_unreal = 0.0
    tot_beta_usd = 0.0
    tot_idio_usd = 0.0
    n_anchored = 0

    for p in positions or []:
        if not isinstance(p, dict):
            continue
        ticker = p.get("ticker")
        ptype = p.get("type") or "stock"
        if ptype in _OPTION_TYPES:
            skipped.append({
                "ticker": ticker,
                "type": ptype,
                "reason": "option — β-attribution is its own surface (see /api/greeks)",
            })
            continue

        try:
            qty = float(p.get("qty") or 0.0)
            avg = float(p.get("avg_cost") or 0.0)
        except (TypeError, ValueError):
            skipped.append({"ticker": ticker, "type": ptype,
                            "reason": "garbage qty/avg_cost"})
            continue

        cur_raw = p.get("current_price")
        try:
            cur = float(cur_raw) if cur_raw is not None else None
        except (TypeError, ValueError):
            cur = None

        if not avg or qty <= 0:
            skipped.append({"ticker": ticker, "type": ptype,
                            "reason": "no cost basis / zero qty"})
            continue
        if cur is None or cur <= 0:
            skipped.append({"ticker": ticker, "type": ptype,
                            "reason": "unmarked price — β-attribution undefined"})
            continue

        opened = _parse_ts(p.get("opened_at"))
        cost_basis = avg * qty
        unreal_usd = (cur - avg) * qty
        pos_ret_pct = (cur / avg - 1.0) * 100.0

        # Sector→beta — same SSOT as stress_scenarios / /api/risk. Unknown
        # sector or NaN beta falls back to market-beta 1.0 (the
        # ``_position_betas`` precedent).
        try:
            sec = classify(ticker or "") if classify else "other"
        except Exception:
            sec = "other"
        try:
            beta = float(beta_map.get(sec, 1.0))
        except (TypeError, ValueError, AttributeError):
            beta = 1.0
        if beta != beta:  # NaN guard
            beta = 1.0

        entry_spy = _spy_at_or_after(series, opened) if opened else None
        anchored = entry_spy is not None and now_spy is not None

        if anchored:
            spy_ret_pct = (now_spy / entry_spy - 1.0) * 100.0
            beta_pct = beta * spy_ret_pct
            idio_pct = pos_ret_pct - beta_pct
            beta_usd = cost_basis * beta_pct / 100.0
            idio_usd = unreal_usd - beta_usd
            tot_cost += cost_basis
            tot_unreal += unreal_usd
            tot_beta_usd += beta_usd
            tot_idio_usd += idio_usd
            n_anchored += 1
        else:
            spy_ret_pct = None
            beta_pct = None
            idio_pct = None
            beta_usd = None
            idio_usd = None

        rows.append({
            "ticker": ticker,
            "type": ptype,
            "sector": sec,
            "beta": _z(beta, 2),
            "qty": round(qty, 6),
            "opened_at": p.get("opened_at"),
            "cost_basis_usd": _z(cost_basis),
            "position_return_pct": _z(pos_ret_pct, 3),
            "spy_return_pct": _z(spy_ret_pct, 3),
            "beta_explained_pct": _z(beta_pct, 3),
            "idiosyncratic_pct": _z(idio_pct, 3),
            "unrealized_usd": _z(unreal_usd),
            "beta_explained_usd": _z(beta_usd),
            "idiosyncratic_usd": _z(idio_usd),
            "anchored": anchored,
        })

    # Biggest |idiosyncratic_usd| first — the desk's "what is selection
    # actually contributing" sort. Unanchored rows (idio None) sort last.
    rows.sort(key=lambda r: (r["idiosyncratic_usd"] is None,
                             -abs(r["idiosyncratic_usd"])
                             if r["idiosyncratic_usd"] is not None else 0.0))

    base["rows"] = rows
    base["skipped"] = skipped
    base["n_positions"] = len(rows)
    base["n_anchored"] = n_anchored

    if not rows:
        base["state"] = "NO_DATA"
        base["headline"] = "P/L attribution: no open stock positions."
        return base
    if not series:
        base["state"] = "NO_BENCHMARK"
        base["headline"] = (
            "P/L attribution: no SPY history in equity curve yet (cold "
            "start). β-decomposition withheld until the first equity tick.")
        return base
    if n_anchored == 0:
        base["state"] = "INSUFFICIENT"
        base["headline"] = (
            "P/L attribution: positions exist but none can be anchored to "
            "the equity curve's SPY history (likely opened before the "
            "first equity tick). β-decomposition withheld.")
        return base

    base["state"] = "OK"
    book_unreal_pct = (tot_unreal / tot_cost * 100.0) if tot_cost > 0 else None
    book_beta_pct = (tot_beta_usd / tot_cost * 100.0) if tot_cost > 0 else None
    book_idio_pct = (tot_idio_usd / tot_cost * 100.0) if tot_cost > 0 else None
    base["totals"] = {
        "cost_basis_usd": _z(tot_cost),
        "unrealized_usd": _z(tot_unreal),
        "beta_explained_usd": _z(tot_beta_usd),
        "idiosyncratic_usd": _z(tot_idio_usd),
        "unrealized_pct": _z(book_unreal_pct, 3),
        "beta_explained_pct": _z(book_beta_pct, 3),
        "idiosyncratic_pct": _z(book_idio_pct, 3),
    }
    base["headline"] = (
        f"β-attribution ({n_anchored} of {len(rows)} stock position"
        f"{'' if n_anchored == 1 else 's'} anchored): "
        f"unrealized ${tot_unreal:+.2f} = "
        f"β·SPY ${tot_beta_usd:+.2f} + idiosyncratic ${tot_idio_usd:+.2f}."
    )
    return base

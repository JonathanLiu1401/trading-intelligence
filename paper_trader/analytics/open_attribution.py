"""Open-position alpha attribution — the bot's *dominant* return source.

``analytics/round_trips.py`` (→ ``/api/analytics``, ``/api/performance-
attribution``) is the single source of truth for *closed* round-trip P&L. But
the live trader spends almost all of its time HOLDing (observed: 18 HOLD / 2
NO_DECISION over the last 20 cycles, 0 fills): its realized book is tiny while
its **open** drift dominates total return. That open drift vs. SPY is currently
invisible — no panel decomposes "how much of my −2.7% is the market vs. my
selection?".

``build_open_attribution`` answers exactly that, per open *stock* position:

* anchor the holding's entry to the S&P level **at or after ``opened_at``** —
  read straight off the equity curve's ``sp500_price`` (no extra network).
  ``opened_at`` is the correct anchor: invariant #8 (paper-trader AGENTS.md)
  says a re-bought, previously-closed lot **reactivates the same row with
  ``opened_at`` reset**, so matching the *first historical fill* would smear
  alpha across a flat gap when the name was not held.
* position return % since entry = ``current_price/avg_cost − 1``
* SPY return % over the same window
* ``alpha_pct = position − SPY``; in dollars, ``excess_usd`` = unrealized P&L
  minus what the same cost basis in SPY would have returned.

Options are **flagged and skipped** — alpha-vs-SPY does not fit a Greeks
instrument; this follows the ``/api/backtests/compare`` precedent ("per-fill
FIFO lot, stocks only", AGENTS.md #10). Pure; ``now`` is injectable.
"""
from __future__ import annotations

from datetime import datetime, timezone

_OPTION_TYPES = {"call", "put"}


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _spy_series(equity_curve: list[dict]) -> list[tuple[datetime, float]]:
    """Ascending [(ts, sp500_price)] with parseable ts and a positive price."""
    out: list[tuple[datetime, float]] = []
    for row in equity_curve or []:
        ts = _parse_ts(row.get("timestamp"))
        px = row.get("sp500_price")
        if ts is None or not px or float(px) <= 0:
            continue
        out.append((ts, float(px)))
    out.sort(key=lambda r: r[0])
    return out


def _spy_at_or_after(series: list[tuple[datetime, float]],
                     ts: datetime) -> float | None:
    """First S&P level at or after ``ts`` (entry anchor)."""
    for t, px in series:
        if t >= ts:
            return px
    return None


def build_open_attribution(positions: list[dict],
                           equity_curve: list[dict],
                           now: datetime | None = None) -> dict:
    """Per-open-stock selection-vs-market decomposition. Pure."""
    now = now or datetime.now(timezone.utc)
    series = _spy_series(equity_curve)
    now_spy = series[-1][1] if series else None

    rows: list[dict] = []
    skipped: list[dict] = []
    tot_cost = 0.0
    tot_unreal = 0.0
    tot_spy_equiv = 0.0

    for p in positions or []:
        ticker = p.get("ticker")
        ptype = p.get("type")
        if ptype in _OPTION_TYPES:
            skipped.append({
                "ticker": ticker,
                "type": ptype,
                "reason": "option — alpha-vs-SPY not meaningful (see /api/greeks)",
            })
            continue

        qty = float(p.get("qty") or 0.0)
        avg = float(p.get("avg_cost") or 0.0)
        cur = p.get("current_price")
        opened = _parse_ts(p.get("opened_at"))

        if not avg or qty <= 0:
            skipped.append({"ticker": ticker, "type": ptype,
                            "reason": "no cost basis / zero qty"})
            continue
        if not cur or float(cur) <= 0:
            skipped.append({"ticker": ticker, "type": ptype,
                            "reason": "unmarked price — return undefined"})
            continue

        cur = float(cur)
        cost_basis = avg * qty
        unreal_usd = (cur - avg) * qty
        pos_ret_pct = (cur / avg - 1.0) * 100.0

        entry_spy = _spy_at_or_after(series, opened) if opened else None
        anchored = entry_spy is not None and now_spy is not None

        if anchored:
            spy_ret_pct = (now_spy / entry_spy - 1.0) * 100.0
            spy_equiv_usd = cost_basis * spy_ret_pct / 100.0
            alpha_pct = pos_ret_pct - spy_ret_pct
            excess_usd = unreal_usd - spy_equiv_usd
            # Aggregate over anchored rows ONLY — an un-benchmarkable position
            # in tot_cost/tot_unreal without a matching tot_spy_equiv would
            # silently skew book_open_alpha_pct.
            tot_cost += cost_basis
            tot_unreal += unreal_usd
            tot_spy_equiv += spy_equiv_usd
        else:
            spy_ret_pct = None
            spy_equiv_usd = None
            alpha_pct = None
            excess_usd = None

        rows.append({
            "ticker": ticker,
            "type": ptype,
            "qty": round(qty, 6),
            "opened_at": p.get("opened_at"),
            "cost_basis_usd": round(cost_basis, 2),
            "position_return_pct": round(pos_ret_pct, 3),
            "spy_return_pct": (round(spy_ret_pct, 3)
                               if spy_ret_pct is not None else None),
            "alpha_pct": (round(alpha_pct, 3)
                          if alpha_pct is not None else None),
            "unrealized_usd": round(unreal_usd, 2),
            "spy_equivalent_usd": (round(spy_equiv_usd, 2)
                                   if spy_equiv_usd is not None else None),
            "excess_usd": (round(excess_usd, 2)
                           if excess_usd is not None else None),
            "anchored": anchored,
        })

    # Biggest dollar drag first — what a desk reviews first. Unanchored rows
    # (excess None) sort last.
    rows.sort(key=lambda r: (r["excess_usd"] is None,
                             r["excess_usd"] if r["excess_usd"] is not None else 0.0))

    anchored_rows = [r for r in rows if r["anchored"]]
    if not series:
        status = "NO_BENCHMARK"
    elif not anchored_rows:
        status = "NO_DATA"
    else:
        net_excess = tot_unreal - tot_spy_equiv
        status = ("SELECTION_ADDING" if net_excess > 0
                  else "SELECTION_DRAG" if net_excess < 0
                  else "FLAT_VS_SPY")

    book_alpha_pct = (round((tot_unreal - tot_spy_equiv) / tot_cost * 100.0, 3)
                      if anchored_rows and tot_cost else None)
    net_excess_usd = (round(tot_unreal - tot_spy_equiv, 2)
                      if anchored_rows else None)

    if status in ("SELECTION_ADDING", "SELECTION_DRAG", "FLAT_VS_SPY"):
        verb = ("adding" if status == "SELECTION_ADDING"
                else "dragging" if status == "SELECTION_DRAG" else "flat vs")
        headline = (
            f"Open book is {verb} {abs(book_alpha_pct):.2f}% alpha vs SPY "
            f"(${net_excess_usd:+.2f} on ${round(tot_cost, 2):.2f} cost basis) "
            f"across {len(anchored_rows)} held name(s).")
        worst = anchored_rows and min(anchored_rows, key=lambda r: r["excess_usd"])
        if worst and worst["excess_usd"] < 0:
            headline += (f" Biggest drag: {worst['ticker']} "
                         f"({worst['alpha_pct']:+.2f}% vs SPY).")
    elif status == "NO_BENCHMARK":
        headline = "No S&P benchmark in the equity curve — cannot attribute."
    else:
        headline = "No anchorable open stock positions to attribute."

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "status": status,
        "headline": headline,
        "book_open_alpha_pct": book_alpha_pct,
        "net_excess_usd": net_excess_usd,
        "total_cost_basis_usd": round(tot_cost, 2),
        "total_unrealized_usd": round(tot_unreal, 2),
        "total_spy_equivalent_usd": (round(tot_spy_equiv, 2)
                                     if anchored_rows else None),
        "n_positions": len(rows),
        "n_anchored": len(anchored_rows),
        "positions": rows,
        "skipped": skipped,
    }


if __name__ == "__main__":  # smoke test against the live DB
    import json
    from paper_trader.store import get_store
    s = get_store()
    rep = build_open_attribution(s.open_positions(), s.equity_curve(limit=5000))
    print(json.dumps(rep, indent=2, default=str))

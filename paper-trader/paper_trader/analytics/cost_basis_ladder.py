"""Cost-basis ladder — per-open-position FIFO lot breakdown with
per-lot P&L at the current mark.

The existing position-level diagnostics surface one *aggregate* number
per holding — ``positions.avg_cost``, ``position_thesis.pl_pct``,
``position-attention.unrealized_pl_pct``. They flatten a multi-entry
holding into one weighted average and hide the *dispersion* the
operator needs to make an informed partial-exit decision.

When NVDA is opened across 4 separate BUYs at $215 / $220 / $223 /
$228 (current $221), the aggregate avg_cost is $221.50 — within 0.2%
of the mark, reading "BALANCED". But the actual lot ladder reveals:

* lot 1 ($215, 1 share) — +2.8% in profit, harvestable
* lot 2 ($220, 1 share) — +0.5% slightly green
* lot 3 ($223, 1 share) — -0.9% mildly underwater
* lot 4 ($228, 1 share) — -3.1% materially underwater

A partial trim makes sense per FIFO accounting (or per LIFO-tax
intent). No existing endpoint surfaces this — ``round_trips`` walks
*closed* trips; ``rebuy_regret`` measures sell→rebuy price deltas
across time; ``conviction_deployment`` rolls up dollars deployed
across an external proxy. None show the operator the per-lot
ladder of an *open* position with per-lot P&L.

Per-lot reconstruction algorithm (FIFO, deterministic):

1. For each open position ``(ticker, type, expiry, strike)``, walk
   *all* historical trades for that tuple in chronological order.
2. On BUY: append a lot ``{qty, price, ts, trade_id, reason_excerpt}``.
3. On SELL: subtract qty FIFO from the head of the lot queue
   (matching the live trader's FIFO accounting in ``_execute``).
4. The remaining lots ARE the current cost-basis ladder.

Verdict matrix (per position, then aggregated):

* ``LADDER_ALL_GREEN`` — every remaining lot at or above the mark.
  Operator can trim any lot without booking a loss.
* ``LADDER_ALL_RED`` — every remaining lot underwater. Operator
  must book a realised loss to exit any portion.
* ``LADDER_WIDE`` — lot P&L spread (max - min) ≥ ``WIDE_SPREAD_PCT``
  (default 5%). Aggregated avg_cost hides material dispersion — a
  partial exit decision needs to pick *which* lot.
* ``LADDER_STACKED`` — spread < ``WIDE_SPREAD_PCT``. All lots in
  the same neighbourhood; aggregate avg_cost is a faithful summary.
* ``LADDER_SINGLE_LOT`` — exactly 1 remaining lot (fresh position
  with no averaging).
* ``NO_LOTS`` — position present in the open_positions list but
  no BUY trades reconstruct a lot (data inconsistency, possibly a
  position synthesized by an external pathway). Always silent in
  the rolled-up envelope (never raises).

Aggregate verdict over all open positions:

* ``HARVESTABLE_LOTS`` — ≥ 1 position has at least one lot ≥
  ``HARVEST_PCT_FLOOR`` (default 3%) in profit. Trimmable without
  realising a loss.
* ``UNDERWATER_BOOK`` — every position is LADDER_ALL_RED. Book
  cannot rotate without realising a loss.
* ``MIXED_BOOK`` — at least one position green, at least one red.
* ``NO_DATA`` — no open positions OR no reconstructable lots.

Pure builder. Open positions + trades + current prices in, dict out,
never raises. Observational only — never gates Opus, no caps
(AGENTS.md #2 / #12 — the ``position_thesis`` / ``rebuy_regret``
precedent).
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

# Lot-spread threshold (max_lot_pl_pct - min_lot_pl_pct) at or above
# which a position is LADDER_WIDE. 5pp captures materially different
# entries within one position (e.g. earnings-driven re-averaging).
DEFAULT_WIDE_SPREAD_PCT = 5.0

# Per-lot profit floor for the HARVESTABLE_LOTS aggregate verdict.
# A 3% green lot is materially trimmable without booking a loss net
# of typical slippage; 1-2% is noise on the live tick.
DEFAULT_HARVEST_PCT_FLOOR = 3.0

# Reason-excerpt cap so the envelope stays small on the wire. 80
# chars is the same cap ``position_thesis`` / ``thesis_drift`` use
# for their drift_reasons strings.
_REASON_EXCERPT_MAX = 80

# Per-position lot list cap. Live traders very rarely scale into a
# single name more than ~20 times before closing the round trip;
# cap at 50 keeps the envelope bounded even on misbehaving data.
_MAX_LOTS_PER_POSITION = 50

# Aggregate-verdict labels — exposed for tests + caller switch
# statements.
HARVESTABLE_LOTS = "HARVESTABLE_LOTS"
UNDERWATER_BOOK = "UNDERWATER_BOOK"
MIXED_BOOK = "MIXED_BOOK"
NO_DATA = "NO_DATA"

# Per-position verdict labels.
LADDER_ALL_GREEN = "LADDER_ALL_GREEN"
LADDER_ALL_RED = "LADDER_ALL_RED"
LADDER_WIDE = "LADDER_WIDE"
LADDER_STACKED = "LADDER_STACKED"
LADDER_SINGLE_LOT = "LADDER_SINGLE_LOT"
NO_LOTS = "NO_LOTS"


def _num(x: Any) -> float | None:
    """Permissive numeric coerce — None/blank/non-numeric → None."""
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            return float(s)
        except (TypeError, ValueError):
            return None
    return None


def _parse_iso(ts: Any) -> datetime | None:
    if not isinstance(ts, str) or not ts:
        return None
    try:
        s = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (TypeError, ValueError):
        return None


def _position_key(pos: dict) -> tuple[str, str, str, float | None]:
    """The 4-tuple that uniquely identifies a position lot family.

    Mirrors the ``positions`` UNIQUE constraint
    (ticker, type, expiry, strike) from ``store.py``. Expiry is
    canonicalised to a string ('' for None) so dict-keying is stable;
    strike is left as a numeric so options-vs-stock keys don't collide
    on hash-floor collisions.
    """
    return (
        str(pos.get("ticker") or "").upper(),
        str(pos.get("type") or "stock").lower(),
        str(pos.get("expiry") or ""),
        _num(pos.get("strike")),
    )


def _trade_key(trade: dict) -> tuple[str, str, str, float | None]:
    return (
        str(trade.get("ticker") or "").upper(),
        str(trade.get("type") or trade.get("option_type") or "stock").lower(),
        str(trade.get("expiry") or ""),
        _num(trade.get("strike")),
    )


def _classify_trade_action(action: Any) -> str | None:
    """Map a free-text action string to BUY / SELL / None.

    The live ``trades.action`` column holds verb strings like ``BUY``,
    ``SELL``, ``BUY_CALL``, ``SELL_CALL``, ``BUY_PUT``, ``SELL_PUT``.
    Anything else (e.g. ``REBALANCE`` ledger rows) is treated as None
    and the trade is skipped.
    """
    if not isinstance(action, str):
        return None
    a = action.strip().upper()
    if not a:
        return None
    if a.startswith("BUY"):
        return "BUY"
    if a.startswith("SELL"):
        return "SELL"
    return None


def _reconstruct_lots(
    trades_for_position: Sequence[dict],
) -> list[dict]:
    """FIFO lot walk — returns the list of remaining lots in
    chronological order (oldest first).

    Each lot dict carries ``qty``, ``price``, ``ts``, ``trade_id``,
    ``reason_excerpt``. Tolerates malformed rows by skipping them.
    """
    lots: deque[dict] = deque()
    # Sort by timestamp ascending (then by trade_id for ties — the same
    # tiebreak ``recent_trades`` uses, just reversed for the chronological
    # walk).
    sortable: list[tuple[datetime, int, dict]] = []
    for t in trades_for_position:
        ts = _parse_iso(t.get("timestamp"))
        if ts is None:
            continue
        tid_raw = t.get("id") or 0
        try:
            tid = int(tid_raw)
        except (TypeError, ValueError):
            tid = 0
        sortable.append((ts, tid, t))
    sortable.sort(key=lambda r: (r[0], r[1]))

    for ts, tid, t in sortable:
        verb = _classify_trade_action(t.get("action"))
        if verb is None:
            continue
        qty = _num(t.get("qty"))
        price = _num(t.get("price"))
        if qty is None or price is None or qty <= 0 or price <= 0:
            continue
        if verb == "BUY":
            reason = t.get("reason")
            excerpt = ""
            if isinstance(reason, str) and reason:
                excerpt = reason.strip()
                if len(excerpt) > _REASON_EXCERPT_MAX:
                    excerpt = excerpt[:_REASON_EXCERPT_MAX] + "…"
            lots.append({
                "qty": qty,
                "price": price,
                "ts": ts.isoformat(),
                "trade_id": tid,
                "reason_excerpt": excerpt,
            })
        else:  # SELL — FIFO subtract
            remaining = qty
            while remaining > 1e-9 and lots:
                head = lots[0]
                if head["qty"] <= remaining + 1e-9:
                    remaining -= head["qty"]
                    lots.popleft()
                else:
                    head["qty"] = head["qty"] - remaining
                    remaining = 0.0
            # Any over-sell (more sold than bought) is just dropped —
            # the live store guards against this but the ledger may
            # carry the residue from legacy schema migrations. Silent
            # by construction.

    # Cap envelope size by collapsing the oldest excess into the last
    # kept lot (preserving the cost-basis weighted sum).
    out = list(lots)
    if len(out) > _MAX_LOTS_PER_POSITION:
        head = out[: len(out) - _MAX_LOTS_PER_POSITION + 1]
        tail = out[len(out) - _MAX_LOTS_PER_POSITION + 1 :]
        # Collapse head into one synthetic lot.
        tot_qty = sum(l["qty"] for l in head)
        if tot_qty > 0:
            wavg = sum(l["qty"] * l["price"] for l in head) / tot_qty
            collapsed = {
                "qty": tot_qty,
                "price": wavg,
                "ts": head[0]["ts"],
                "trade_id": head[0]["trade_id"],
                "reason_excerpt": f"[collapsed {len(head)} earliest lots]",
            }
            out = [collapsed] + tail
        else:
            out = tail
    return out


def _classify_per_position(
    lots: list[dict],
    current_price: float | None,
    wide_spread_pct: float,
) -> dict[str, Any]:
    """Per-position verdict + per-lot P&L enrichment.

    Returns a dict with ``verdict``, ``lots`` (with per-lot pl_pct
    fields added), ``spread_pct``, ``max_lot_pl_pct``,
    ``min_lot_pl_pct``, ``total_qty``, ``weighted_avg_cost``.
    """
    if not lots:
        return {
            "verdict": NO_LOTS,
            "lots": [],
            "n_lots": 0,
            "spread_pct": None,
            "max_lot_pl_pct": None,
            "min_lot_pl_pct": None,
            "total_qty": 0.0,
            "weighted_avg_cost": None,
        }

    total_qty = sum(l["qty"] for l in lots)
    if total_qty <= 0:
        return {
            "verdict": NO_LOTS,
            "lots": [],
            "n_lots": 0,
            "spread_pct": None,
            "max_lot_pl_pct": None,
            "min_lot_pl_pct": None,
            "total_qty": 0.0,
            "weighted_avg_cost": None,
        }
    weighted_avg_cost = (
        sum(l["qty"] * l["price"] for l in lots) / total_qty
    )

    cp = _num(current_price)
    enriched: list[dict] = []
    pl_pcts: list[float] = []
    for l in lots:
        lot_price = float(l["price"])
        pl_pct: float | None
        pl_usd: float | None
        if cp is not None and lot_price > 0:
            pl_pct = round((cp - lot_price) / lot_price * 100.0, 2)
            pl_usd = round((cp - lot_price) * float(l["qty"]), 2)
            pl_pcts.append(pl_pct)
        else:
            pl_pct = None
            pl_usd = None
        enriched.append({
            **l,
            "qty": round(float(l["qty"]), 6),
            "price": round(lot_price, 4),
            "pl_pct": pl_pct,
            "pl_usd": pl_usd,
        })

    if not pl_pcts:
        # Mark unavailable; we still emit the structure but with no
        # verdict beyond NO_LOTS so callers don't render a stale pill.
        return {
            "verdict": NO_LOTS,
            "lots": enriched,
            "n_lots": len(enriched),
            "spread_pct": None,
            "max_lot_pl_pct": None,
            "min_lot_pl_pct": None,
            "total_qty": round(total_qty, 6),
            "weighted_avg_cost": round(weighted_avg_cost, 4),
        }

    max_pl = max(pl_pcts)
    min_pl = min(pl_pcts)
    spread = max_pl - min_pl

    if len(enriched) == 1:
        verdict = LADDER_SINGLE_LOT
    elif min_pl >= 0.0:
        verdict = LADDER_ALL_GREEN
    elif max_pl <= 0.0:
        verdict = LADDER_ALL_RED
    elif spread >= wide_spread_pct:
        verdict = LADDER_WIDE
    else:
        verdict = LADDER_STACKED

    return {
        "verdict": verdict,
        "lots": enriched,
        "n_lots": len(enriched),
        "spread_pct": round(spread, 2),
        "max_lot_pl_pct": max_pl,
        "min_lot_pl_pct": min_pl,
        "total_qty": round(total_qty, 6),
        "weighted_avg_cost": round(weighted_avg_cost, 4),
    }


def _harvestable_lot(per_position: dict, harvest_floor: float) -> dict | None:
    """Return the single most-in-profit lot for a position if it
    clears ``harvest_floor``, else None."""
    best: dict | None = None
    best_pl = harvest_floor - 1.0  # strict gate: must clear floor
    for l in per_position.get("lots", []):
        pl = l.get("pl_pct")
        if pl is None:
            continue
        if pl >= harvest_floor and pl > best_pl:
            best = l
            best_pl = pl
    return best


def build_cost_basis_ladder(
    open_positions: Sequence[dict] | None,
    trades: Sequence[dict] | None,
    *,
    now: datetime | None = None,
    wide_spread_pct: float = DEFAULT_WIDE_SPREAD_PCT,
    harvest_pct_floor: float = DEFAULT_HARVEST_PCT_FLOOR,
) -> dict[str, Any]:
    """Pure FIFO lot-reconstruction with per-lot P&L at the current mark.

    Inputs:
      ``open_positions`` — list of position dicts. Each needs
        ``ticker``, ``type``, ``current_price`` (or ``mark`` /
        ``last_price``); ``expiry`` / ``strike`` optional.
      ``trades`` — list of trade dicts. Each needs ``timestamp``,
        ``ticker``, ``action``, ``qty``, ``price``; ``id`` / ``reason``
        / ``expiry`` / ``strike`` / ``option_type`` optional.
      ``now`` — defaults to ``datetime.now(utc)`` (just for envelope).
      ``wide_spread_pct`` — threshold for LADDER_WIDE (pp).
      ``harvest_pct_floor`` — threshold for a lot to count as
        HARVESTABLE (pp).

    Returns the envelope dict. Never raises.
    """
    now = now or datetime.now(timezone.utc)

    positions = list(open_positions or [])
    trades_list = list(trades or [])

    if not positions:
        return {
            "as_of": now.isoformat(),
            "verdict": NO_DATA,
            "headline": "no open positions",
            "n_positions": 0,
            "n_lots_total": 0,
            "positions": [],
            "harvestable": [],
            "thresholds": {
                "wide_spread_pct": wide_spread_pct,
                "harvest_pct_floor": harvest_pct_floor,
            },
        }

    # Index trades by position key for O(N) lot reconstruction.
    by_key: dict[tuple, list[dict]] = {}
    for t in trades_list:
        if not isinstance(t, dict):
            continue
        k = _trade_key(t)
        by_key.setdefault(k, []).append(t)

    per_position_out: list[dict] = []
    harvestable: list[dict] = []
    n_lots_total = 0

    n_all_green = 0
    n_all_red = 0
    n_with_any_red = 0
    n_with_any_green = 0
    n_no_lots = 0

    for pos in positions:
        if not isinstance(pos, dict):
            continue
        k = _position_key(pos)
        ticker = k[0]
        if not ticker:
            continue
        # Prefer current_price; fall back to common aliases.
        cp = (
            _num(pos.get("current_price"))
            or _num(pos.get("mark"))
            or _num(pos.get("last_price"))
        )
        lots = _reconstruct_lots(by_key.get(k, []))
        classified = _classify_per_position(
            lots, cp, wide_spread_pct=wide_spread_pct,
        )
        v = classified["verdict"]
        if v == NO_LOTS:
            n_no_lots += 1
        elif v == LADDER_ALL_GREEN:
            n_all_green += 1
            n_with_any_green += 1
        elif v == LADDER_ALL_RED:
            n_all_red += 1
            n_with_any_red += 1
        else:
            # WIDE / STACKED / SINGLE_LOT — bucket on lot signs.
            has_green = any(
                (l.get("pl_pct") or 0.0) > 0 for l in classified["lots"]
            )
            has_red = any(
                (l.get("pl_pct") or 0.0) < 0 for l in classified["lots"]
            )
            if has_green:
                n_with_any_green += 1
            if has_red:
                n_with_any_red += 1

        n_lots_total += classified["n_lots"]
        per_position_out.append({
            "ticker": ticker,
            "type": k[1],
            "expiry": k[2] or None,
            "strike": k[3],
            "current_price": cp,
            **classified,
        })

        best = _harvestable_lot(classified, harvest_pct_floor)
        if best is not None:
            harvestable.append({
                "ticker": ticker,
                "type": k[1],
                "trade_id": best.get("trade_id"),
                "qty": best.get("qty"),
                "price": best.get("price"),
                "pl_pct": best.get("pl_pct"),
                "pl_usd": best.get("pl_usd"),
                "ts": best.get("ts"),
            })

    # Sort harvestable list by pl_pct descending — operator wants the
    # most-in-profit lot at the top.
    harvestable.sort(
        key=lambda h: h.get("pl_pct") if h.get("pl_pct") is not None else -1e9,
        reverse=True,
    )

    # Aggregate verdict
    n_positions_with_lots = sum(
        1 for p in per_position_out if p["verdict"] != NO_LOTS
    )
    if n_positions_with_lots == 0:
        verdict = NO_DATA
        headline = (
            f"no reconstructable lots across {len(per_position_out)} "
            f"open position(s)"
        )
    elif harvestable:
        verdict = HARVESTABLE_LOTS
        h = harvestable[0]
        headline = (
            f"{len(harvestable)} harvestable lot(s) across "
            f"{n_positions_with_lots} position(s) — top: "
            f"{h['ticker']} lot {h['pl_pct']:+.1f}% (qty {h['qty']:g} "
            f"@ ${h['price']:g})"
        )
    elif n_with_any_green == 0 and n_with_any_red > 0:
        verdict = UNDERWATER_BOOK
        headline = (
            f"every lot across {n_positions_with_lots} position(s) is "
            f"underwater — no green lots to harvest"
        )
    else:
        verdict = MIXED_BOOK
        headline = (
            f"{n_with_any_green} position(s) with green lots, "
            f"{n_with_any_red} with red — no lot clears the "
            f"{harvest_pct_floor:g}% harvest floor"
        )

    return {
        "as_of": now.isoformat(),
        "verdict": verdict,
        "headline": headline,
        "n_positions": len(per_position_out),
        "n_lots_total": n_lots_total,
        "positions": per_position_out,
        "harvestable": harvestable,
        "thresholds": {
            "wide_spread_pct": wide_spread_pct,
            "harvest_pct_floor": harvest_pct_floor,
        },
        "note": (
            "Pure FIFO lot reconstruction. Advisory only — no caps, "
            "no gates, never modulates trade path (AGENTS.md #2 / #12)."
        ),
    }

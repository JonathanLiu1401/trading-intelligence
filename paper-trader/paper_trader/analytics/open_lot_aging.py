"""Per-open-lot age + bucket — surfaces the drift from trading to
holding.

``cost_basis_ladder`` reconstructs the FIFO lot ladder of every open
position with per-lot P&L *at the current mark*. It says nothing about
*how long* each lot has been held — which is exactly the
disposition-pathology question the next decision needs answered:

  *That NVDA lot at -1.5% — is it a 4-hour scratch waiting on the next
  catalyst, or a 32-day position that started as a trade and has
  silently turned into an unintended hold?*

``holding_period_distribution`` answers that question for CLOSED
round-trips (post-mortem). ``add_discipline`` measures the
*BUY-cadence* across averaging-in. ``catalyst_expiry_skill`` flags
positions whose dated catalyst has passed. None of them tells the
operator how OLD each *currently-open* lot is.

This builder fills exactly that gap and nothing else. Per-lot age is
computed from the trade timestamp; bucketed FRESH / NORMAL / MATURE /
STALE; cross-tabulated with P&L sign to flag the two pathological
quadrants:

* ``STALE_RED``  — old + underwater. The trade thesis has had ample
  time to play out and hasn't; bias to honest exit before "small loss"
  silently turns into "averaged-down anchor".
* ``STALE_GREEN`` — old + green. Either trim-overdue (the trade
  worked, capital deployed too long for the realised edge) or genuine
  hold; either way the operator should consciously RE-affirm.

Aggregate verdicts (book-level): FRESH_BOOK / NORMAL_BOOK /
AGING_BOOK / STALE_BOOK, plus an ``attention`` list of (ticker,
verdict) pairs ranked oldest-first for the operator's first read.

Pure builder, no I/O, never raises. FIFO lot reconstruction reuses the
existing primitives in ``cost_basis_ladder`` to guarantee the two
builders see identical lots (single source of truth). Observational
only — never gates Opus, no caps (AGENTS.md #2 / #12 — the
``cost_basis_ladder`` precedent).
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable

from . import cost_basis_ladder as _cbl

#: Age bucket boundaries in days. A position day-traded same session
#: is FRESH; one held into the next session is NORMAL; one held a week
#: + is MATURE; one held a month + is STALE. The thresholds mirror
#: ``holding_period_distribution``'s SCALP / INTRADAY / OVERNIGHT /
#: SWING / POSITION buckets, collapsed for an *open* book where the
#: relevant question is "how old is the oldest dollar of risk".
FRESH_DAYS_MAX = 1.0
NORMAL_DAYS_MAX = 7.0
MATURE_DAYS_MAX = 30.0
#: Anything ≥ MATURE_DAYS_MAX days is STALE.

#: Per-lot P&L sign threshold (pp) for the STALE_RED / STALE_GREEN
#: classification. ±0.5 pp keeps a literally-flat lot from being
#: classified as either — the ``catalyst_expiry_skill`` precedent.
FLAT_PCT_TOL = 0.5

#: Aggregate-verdict thresholds — share of total open-lot dollars that
#: are STALE or MATURE-or-worse.
STALE_BOOK_PCT_THRESHOLD = 50.0
AGING_BOOK_PCT_THRESHOLD = 50.0

# Bucket labels — exposed for tests + caller switch statements.
BUCKET_FRESH = "FRESH"
BUCKET_NORMAL = "NORMAL"
BUCKET_MATURE = "MATURE"
BUCKET_STALE = "STALE"

# Per-position verdicts.
POS_FRESH = "FRESH"
POS_NORMAL = "NORMAL"
POS_MATURE_MIX = "MATURE_MIX"
POS_STALE_RED = "STALE_RED"
POS_STALE_GREEN = "STALE_GREEN"
POS_STALE_FLAT = "STALE_FLAT"
POS_NO_LOTS = "NO_LOTS"

# Aggregate verdicts.
AGG_NO_DATA = "NO_DATA"
AGG_FRESH_BOOK = "FRESH_BOOK"
AGG_NORMAL_BOOK = "NORMAL_BOOK"
AGG_AGING_BOOK = "AGING_BOOK"
AGG_STALE_BOOK = "STALE_BOOK"


def _bucket_for_age(age_days: float) -> str:
    """Closed-on-the-left buckets: FRESH = [0, 1), NORMAL = [1, 7),
    MATURE = [7, 30), STALE = [30, ∞)."""
    if age_days < FRESH_DAYS_MAX:
        return BUCKET_FRESH
    if age_days < NORMAL_DAYS_MAX:
        return BUCKET_NORMAL
    if age_days < MATURE_DAYS_MAX:
        return BUCKET_MATURE
    return BUCKET_STALE


def _z(v: float | None, ndigits: int = 2) -> float | None:
    """Round, folding -0.0 → 0.0 (the ``position_blowup._z`` precedent)."""
    if v is None:
        return None
    try:
        r = round(float(v), ndigits)
    except (TypeError, ValueError):
        return None
    return 0.0 if r == 0 else r


def _lot_pl_pct(lot_price: float, mark: float | None) -> float | None:
    """Per-lot P&L % at the current mark; None when mark is unusable."""
    if mark is None:
        return None
    try:
        m = float(mark)
    except (TypeError, ValueError):
        return None
    if lot_price <= 0 or m <= 0:
        return None
    return (m - lot_price) / lot_price * 100.0


def _classify_lot(age_days: float, pl_pct: float | None) -> str:
    """Per-lot verdict combining age + P&L sign. Only STALE lots earn
    the *_RED / *_GREEN tags; for younger lots the bucket label alone
    is informative enough."""
    bucket = _bucket_for_age(age_days)
    if bucket != BUCKET_STALE:
        return bucket
    if pl_pct is None:
        return POS_STALE_FLAT
    if pl_pct > FLAT_PCT_TOL:
        return POS_STALE_GREEN
    if pl_pct < -FLAT_PCT_TOL:
        return POS_STALE_RED
    return POS_STALE_FLAT


def _position_verdict(lot_verdicts: list[str], lot_pl_pcts: list[float | None]) -> str:
    """Roll up lot-level verdicts to a per-position verdict.

    STALE_RED dominates STALE_GREEN — an old underwater lot is the
    more decision-relevant pathology (the trader's "averaging down
    anchor" pattern). If a position holds BOTH a stale-red and a
    stale-green lot, the position verdict is STALE_RED.
    """
    if not lot_verdicts:
        return POS_NO_LOTS
    if POS_STALE_RED in lot_verdicts:
        return POS_STALE_RED
    if POS_STALE_GREEN in lot_verdicts:
        return POS_STALE_GREEN
    if POS_STALE_FLAT in lot_verdicts:
        return POS_STALE_FLAT
    if BUCKET_MATURE in lot_verdicts:
        return POS_MATURE_MIX
    if BUCKET_NORMAL in lot_verdicts:
        return POS_NORMAL
    return POS_FRESH


def build_open_lot_aging(
    positions: list[dict] | None,
    trades: list[dict] | None,
    now: datetime | None = None,
) -> dict:
    """Per-open-lot age + bucket + per-position roll-up. Pure, never
    raises.

    Reuses ``cost_basis_ladder._reconstruct_lots`` so the two
    builders share lots byte-for-byte (single source of truth)."""
    now = now or datetime.now(timezone.utc)
    base: dict[str, Any] = {
        "as_of": now.isoformat(timespec="seconds"),
        "thresholds": {
            "fresh_days_max": FRESH_DAYS_MAX,
            "normal_days_max": NORMAL_DAYS_MAX,
            "mature_days_max": MATURE_DAYS_MAX,
            "flat_pct_tol": FLAT_PCT_TOL,
            "stale_book_pct_threshold": STALE_BOOK_PCT_THRESHOLD,
            "aging_book_pct_threshold": AGING_BOOK_PCT_THRESHOLD,
        },
        "n_positions": 0,
        "n_lots": 0,
        "positions": [],
        "attention": [],
    }
    rows = list(positions or [])
    if not rows:
        base["state"] = AGG_NO_DATA
        base["headline"] = "Lot aging: no open positions to age."
        return base

    # Index trades by the same key cost_basis_ladder uses.
    trades_by_key: dict[tuple, list[dict]] = defaultdict(list)
    for t in (trades or []):
        try:
            trades_by_key[_cbl._trade_key(t)].append(t)
        except Exception:
            # A single malformed trade row never sinks the rest.
            continue

    out_positions: list[dict] = []
    total_lot_value = 0.0
    stale_lot_value = 0.0
    mature_or_worse_value = 0.0
    n_lots_total = 0
    oldest_age_days = 0.0
    oldest_ticker: str | None = None

    for p in rows:
        try:
            key = _cbl._position_key(p)
        except Exception:
            continue
        ts_for_key = trades_by_key.get(key, [])
        lots = _cbl._reconstruct_lots(ts_for_key)
        ticker = p.get("ticker")
        ptype = (p.get("type") or "stock").lower()
        mark = _cbl._num(p.get("current_price")) or _cbl._num(p.get("avg_cost"))
        mult = 100.0 if ptype in ("call", "put") else 1.0

        lot_rows: list[dict] = []
        lot_verdicts: list[str] = []
        lot_pl_pcts: list[float | None] = []
        for lot in lots:
            ts_str = lot.get("ts")
            try:
                ts = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except Exception:
                # An un-parseable lot timestamp can't be aged; skip
                # rather than poison the position's bucket histogram.
                continue
            age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
            qty = float(lot.get("qty") or 0.0)
            price = float(lot.get("price") or 0.0)
            pl_pct = _lot_pl_pct(price, mark)
            verdict = _classify_lot(age_days, pl_pct)
            lot_verdicts.append(verdict)
            lot_pl_pcts.append(pl_pct)
            lot_value = price * qty * mult
            total_lot_value += lot_value
            bucket = _bucket_for_age(age_days)
            if bucket == BUCKET_STALE:
                stale_lot_value += lot_value
                mature_or_worse_value += lot_value
            elif bucket == BUCKET_MATURE:
                mature_or_worse_value += lot_value
            n_lots_total += 1
            if age_days > oldest_age_days:
                oldest_age_days = age_days
                oldest_ticker = ticker
            lot_rows.append({
                "ts": lot.get("ts"),
                "trade_id": lot.get("trade_id"),
                "qty": _z(qty, 4),
                "price": _z(price, 4),
                "age_days": _z(age_days, 2),
                "bucket": bucket,
                "pl_pct": _z(pl_pct),
                "verdict": verdict,
                "reason_excerpt": lot.get("reason_excerpt") or "",
                "lot_value_usd": _z(lot_value),
            })

        pos_verdict = _position_verdict(lot_verdicts, lot_pl_pcts)
        out_positions.append({
            "ticker": ticker,
            "type": ptype,
            "n_lots": len(lot_rows),
            "oldest_lot_age_days": (
                _z(max((r["age_days"] or 0.0) for r in lot_rows), 2)
                if lot_rows
                else None
            ),
            "verdict": pos_verdict,
            "lots": lot_rows,
        })

    base["n_positions"] = len(out_positions)
    base["n_lots"] = n_lots_total
    base["total_lot_value_usd"] = _z(total_lot_value)
    base["stale_lot_value_usd"] = _z(stale_lot_value)
    base["mature_or_worse_value_usd"] = _z(mature_or_worse_value)

    # Sort positions oldest-first so the operator's first read is the
    # most age-pathological position.
    out_positions.sort(
        key=lambda r: -((r.get("oldest_lot_age_days") or 0.0))
    )
    base["positions"] = out_positions

    # Attention list — only the verdicts that warrant an actively
    # different next-action: STALE_RED (overdue cut), STALE_GREEN
    # (overdue trim), STALE_FLAT (overdue exit), MATURE_MIX (close to
    # stale). Empty when the book is FRESH/NORMAL.
    attention_verdicts = {
        POS_STALE_RED,
        POS_STALE_GREEN,
        POS_STALE_FLAT,
        POS_MATURE_MIX,
    }
    base["attention"] = [
        {
            "ticker": p["ticker"],
            "verdict": p["verdict"],
            "oldest_lot_age_days": p["oldest_lot_age_days"],
        }
        for p in out_positions
        if p["verdict"] in attention_verdicts
    ]

    if n_lots_total == 0:
        base["state"] = AGG_NO_DATA
        base["headline"] = (
            f"Lot aging: {len(out_positions)} open position"
            f"{'' if len(out_positions) == 1 else 's'} but no "
            f"reconstructable lots from the trade ledger."
        )
        return base

    stale_share_pct = (
        (stale_lot_value / total_lot_value * 100.0)
        if total_lot_value > 0
        else 0.0
    )
    mature_or_worse_share_pct = (
        (mature_or_worse_value / total_lot_value * 100.0)
        if total_lot_value > 0
        else 0.0
    )
    base["stale_share_pct"] = _z(stale_share_pct)
    base["mature_or_worse_share_pct"] = _z(mature_or_worse_share_pct)

    if stale_share_pct >= STALE_BOOK_PCT_THRESHOLD:
        agg = AGG_STALE_BOOK
    elif mature_or_worse_share_pct >= AGING_BOOK_PCT_THRESHOLD:
        agg = AGG_AGING_BOOK
    elif any(
        (lot.get("bucket") == BUCKET_NORMAL or lot.get("bucket") == BUCKET_MATURE)
        for p in out_positions
        for lot in p["lots"]
    ):
        agg = AGG_NORMAL_BOOK
    else:
        agg = AGG_FRESH_BOOK
    base["state"] = agg

    if oldest_ticker is not None:
        base["headline"] = (
            f"Lot aging ({agg}): oldest open lot is {oldest_ticker} "
            f"at {oldest_age_days:.1f} days; "
            f"{stale_share_pct:.1f}% of open-lot $ is STALE "
            f"(≥{int(MATURE_DAYS_MAX)}d), "
            f"{mature_or_worse_share_pct:.1f}% is MATURE-or-worse. "
            f"{len(base['attention'])} attention row"
            f"{'' if len(base['attention']) == 1 else 's'}."
        )
    else:
        base["headline"] = (
            f"Lot aging ({agg}): {n_lots_total} lot"
            f"{'' if n_lots_total == 1 else 's'} aged."
        )
    return base


def _cli_main() -> int:
    """Render the live book's open-lot aging table. Read-only — opens
    the live store via the same path the other analytics CLIs use."""
    import json
    from ..store import get_store
    from ..strategy import portfolio_snapshot_readonly

    store = get_store()
    snap = portfolio_snapshot_readonly(store)
    positions = snap.get("positions") or []
    trades = store.recent_trades(limit=2000) if hasattr(store, "recent_trades") else []
    res = build_open_lot_aging(positions, trades)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main())

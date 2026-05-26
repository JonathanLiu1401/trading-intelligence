"""Forward-looking mechanical exit (SL/TP) proximity, per open position.

The hard SL/TP machinery (``strategy._check_and_execute_hard_exits`` +
``store.positions_needing_hard_exit``) auto-closes any open stock lot
whose mark breaches its per-lot ``stop_loss_price`` / ``take_profit_price``.
After the fact, ``/api/hard-exit-summary`` aggregates the discipline
(SL vs TP counts, realized $, top tickers). The dashboard JS renders
per-row SL/TP distance bars on ``/api/positions``.

Neither answers the **forward** question a live trader pages on at
3am: *which of my currently-open lots are within striking distance of
a mechanical exit?* No analytics module reads ``stop_loss_price`` /
``take_profit_price`` for an aggregate proximity verdict, so the
information lives only inside each row of ``/api/positions`` and the
JS dashboard's visual bars — invisible to alerts, to summaries, and
to anything outside the browser.

This builder fills that gap. For each currently-open stock lot with
both ``stop_loss_price`` AND ``take_profit_price`` set, it computes the
position's location inside the SL→TP corridor:

   corridor_pos = (current_price - stop_loss_price)
                  / (take_profit_price - stop_loss_price)

Where ``0`` = sitting on the SL, ``1`` = sitting on the TP, ``0.5`` =
exact mid-band. Below 0 means the next mark will trigger SL; above 1
means TP. Each lot is binned into one of six bands::

  AT_RISK_SL   — corridor_pos < 0     (mechanical SL exit imminent)
  NEAR_SL      — 0 ≤ pos < 0.25       (in the SL quartile)
  MID_BAND     — 0.25 ≤ pos < 0.75    (comfortably in the middle)
  NEAR_TP      — 0.75 ≤ pos ≤ 1.0     (in the TP quartile)
  AT_RISK_TP   — corridor_pos > 1     (mechanical TP exit imminent)
  NO_SL_TP     — either threshold absent / degenerate

And the book rolls up to a single verdict:

  NO_DATA           — no open positions at all.
  NO_SL_TP_SET      — open positions exist but none have both SL+TP.
  AT_RISK           — ≥1 position is AT_RISK_SL or AT_RISK_TP.
  NEAR_THRESHOLD    — ≥1 position is NEAR_SL or NEAR_TP (none at-risk).
  COMFORTABLE       — every position with SL+TP is MID_BAND.

Different from every neighbour:

* ``/api/hard-exit-summary`` — HISTORICAL (already-fired exits).
* ``/api/position-blowup`` — per-position shock LADDER (-10/-25/-50/
  -100 %), idiosyncratic; NOT mechanical-exit proximity.
* ``/api/position-attention`` — AGE-based freshness; not price-based.
* ``/api/risk`` — book-level concentration / SPY shock; per-position
  ``current_price`` only, NOT SL/TP.
* ``/api/exit-priority-ranking`` — ranks which name to exit *next*
  on a thesis basis; NOT mechanical-threshold proximity.

Per-row stock-only by construction: ``_check_and_execute_hard_exits``
only stamps SL/TP on stock buys (``store.upsert_position(..., 'stock',
...)``); option rows have no SL/TP fields the engine enforces. An
``ambiguous`` row (SL >= TP, or non-numeric, or current_price ≤ 0)
counts toward ``no_sl_tp`` so the operator can see "Opus opened
without exit fields" as a distinct bucket.

Observational only — never gates Opus, no caps, no path to
``_execute()`` (AGENTS.md invariants #2 / #12 — the
``hard_exit_summary`` precedent). Pure: no I/O, never raises.

Run as CLI for a one-shot ops view::

    python3 -m paper_trader.analytics.exit_proximity
"""
from __future__ import annotations

from datetime import datetime, timezone


# Corridor-position band boundaries. Quartile-based so the "near SL"
# quartile width matches the "near TP" quartile width — symmetric by
# construction, and the mid-band is the middle half (50%) of the
# corridor where Opus has room before either threshold fires.
NEAR_SL_MAX = 0.25
NEAR_TP_MIN = 0.75

# Round all returned percentages / corridor positions to this many
# decimal places to keep JSON lean and avoid float-noise diffs in tests.
_NDIGITS_PCT = 2
_NDIGITS_POS = 4


def _z(v, ndigits: int = _NDIGITS_PCT):
    """Round + fold -0.0 → 0.0 so the JSON never carries a signed zero
    (the ``stress_scenarios._z`` / ``position_blowup._z`` precedent)."""
    if v is None:
        return None
    try:
        r = round(float(v), ndigits)
    except (TypeError, ValueError):
        return None
    return 0.0 if r == 0 else r


def _f(x, default=None):
    """Best-effort float coercion — a garbage cell yields ``default``,
    never raises (the ``_safe`` contract)."""
    if x is None:
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _classify(corridor_pos: float | None) -> str:
    """Map a corridor position (0=SL, 1=TP) into one of the five proximity
    bands. ``None`` → ``"NO_SL_TP"`` (the band the no-data fallback uses)."""
    if corridor_pos is None:
        return "NO_SL_TP"
    if corridor_pos < 0.0:
        return "AT_RISK_SL"
    if corridor_pos < NEAR_SL_MAX:
        return "NEAR_SL"
    if corridor_pos <= 1.0:
        if corridor_pos >= NEAR_TP_MIN:
            return "NEAR_TP"
        return "MID_BAND"
    return "AT_RISK_TP"


def _row(p: dict) -> dict:
    """Compute the proximity row for one open position.

    Always returns a dict — never raises. Stock-only by construction:
    option rows (``type in {"call","put"}``) return a row with
    ``proximity_band="NO_SL_TP"`` and no corridor math because the live
    SL/TP machinery does not enforce them.

    A degenerate SL/TP setup (SL >= TP, either threshold missing or
    non-numeric, current_price ≤ 0) also routes to ``NO_SL_TP`` so the
    operator can SEE that a lot was opened without enforceable exit
    fields (the band carries the why in ``reason``).
    """
    ticker = (p.get("ticker") or "").upper()
    ptype = (p.get("type") or "stock").lower()
    cur = _f(p.get("current_price"))
    sl = _f(p.get("stop_loss_price"))
    tp = _f(p.get("take_profit_price"))
    qty = _f(p.get("qty"), 0.0) or 0.0
    avg = _f(p.get("avg_cost"))

    out: dict = {
        "ticker": ticker,
        "type": ptype,
        "qty": qty,
        "current_price": _z(cur, _NDIGITS_PCT),
        "stop_loss_price": _z(sl, _NDIGITS_PCT),
        "take_profit_price": _z(tp, _NDIGITS_PCT),
        "dist_to_sl_pct": None,
        "dist_to_tp_pct": None,
        "corridor_pos": None,
        "proximity_band": "NO_SL_TP",
        "closer_target": "NONE",
        "reason": "",
    }

    if ptype != "stock":
        out["reason"] = "options have no SL/TP enforcement"
        return out
    if sl is None or tp is None:
        out["reason"] = "SL/TP missing"
        return out
    if not (sl < tp):
        out["reason"] = "SL/TP degenerate (sl ≥ tp)"
        return out
    if cur is None or cur <= 0:
        out["reason"] = "no current_price"
        return out

    dist_sl = (cur - sl) / cur * 100.0
    dist_tp = (tp - cur) / cur * 100.0
    corridor_pos = (cur - sl) / (tp - sl)
    band = _classify(corridor_pos)
    closer = "SL" if abs(dist_sl) <= abs(dist_tp) else "TP"

    out["dist_to_sl_pct"] = _z(dist_sl, _NDIGITS_PCT)
    out["dist_to_tp_pct"] = _z(dist_tp, _NDIGITS_PCT)
    out["corridor_pos"] = _z(corridor_pos, _NDIGITS_POS)
    out["proximity_band"] = band
    out["closer_target"] = closer
    out["avg_cost"] = _z(avg, _NDIGITS_PCT) if avg is not None else None
    return out


def _aggregate_verdict(rows: list[dict]) -> tuple[str, str]:
    """Roll the per-position bands into one verdict + headline.

    Returns ``("VERDICT", "headline string")``. Verdicts:
      * NO_DATA           — len(rows) == 0
      * NO_SL_TP_SET      — every row is NO_SL_TP
      * AT_RISK           — ≥1 row AT_RISK_SL / AT_RISK_TP
      * NEAR_THRESHOLD    — ≥1 row NEAR_SL / NEAR_TP
      * COMFORTABLE       — at least one row with SL/TP and all such
                            rows are MID_BAND
    """
    if not rows:
        return "NO_DATA", "no open positions"

    counts: dict[str, int] = {}
    for r in rows:
        b = r["proximity_band"]
        counts[b] = counts.get(b, 0) + 1

    n = len(rows)
    n_no = counts.get("NO_SL_TP", 0)
    n_at_sl = counts.get("AT_RISK_SL", 0)
    n_at_tp = counts.get("AT_RISK_TP", 0)
    n_near_sl = counts.get("NEAR_SL", 0)
    n_near_tp = counts.get("NEAR_TP", 0)
    n_mid = counts.get("MID_BAND", 0)
    n_with = n - n_no

    if n_with == 0:
        return ("NO_SL_TP_SET",
                f"{n} open position{'s' if n != 1 else ''} but no SL/TP "
                f"thresholds set on any — mechanical exit machinery is dark")

    if n_at_sl or n_at_tp:
        bits = []
        if n_at_sl:
            bits.append(f"{n_at_sl} AT_RISK_SL")
        if n_at_tp:
            bits.append(f"{n_at_tp} AT_RISK_TP")
        headline = (
            f"AT_RISK — {' · '.join(bits)} of {n_with} "
            f"position{'s' if n_with != 1 else ''} with SL/TP set "
            f"(mechanical exit imminent on next mark)"
        )
        return "AT_RISK", headline

    if n_near_sl or n_near_tp:
        bits = []
        if n_near_sl:
            bits.append(f"{n_near_sl} NEAR_SL")
        if n_near_tp:
            bits.append(f"{n_near_tp} NEAR_TP")
        if n_mid:
            bits.append(f"{n_mid} MID_BAND")
        headline = (
            f"NEAR_THRESHOLD — {' · '.join(bits)} of {n_with} "
            f"position{'s' if n_with != 1 else ''} with SL/TP set"
        )
        return "NEAR_THRESHOLD", headline

    # All remaining-with-SL/TP positions are MID_BAND.
    tail = (
        f"; {n_no} of {n} also without SL/TP" if n_no else ""
    )
    return ("COMFORTABLE",
            f"all {n_mid} priced position{'s' if n_mid != 1 else ''} "
            f"with SL/TP set are MID_BAND{tail}")


def build_exit_proximity(
    positions: list[dict] | None,
    now: datetime | None = None,
) -> dict:
    """Pure builder over an open-position list. Never raises.

    Each row in the returned ``positions`` array carries the proximity
    band + corridor position + signed distances to both thresholds.
    Rows are sorted most-actionable-first: ``AT_RISK_SL`` and
    ``AT_RISK_TP`` first (mechanical exit imminent), then ``NEAR_SL`` /
    ``NEAR_TP`` (within striking distance), then ``MID_BAND``, then
    ``NO_SL_TP`` last (no enforceable threshold).

    Closed lots (qty ≤ 0) are filtered out — they are not at risk of
    a mechanical exit. The closed-set drop happens BEFORE band counting
    so an aggregated ``positions=[closed-only book]`` reads as
    ``NO_DATA`` rather than ``NO_SL_TP_SET``.
    """
    now = (now or datetime.now(timezone.utc))
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    open_rows: list[dict] = []
    try:
        for p in (positions or []):
            try:
                q = float(p.get("qty") or 0.0)
            except (TypeError, ValueError):
                q = 0.0
            if q <= 0:
                continue
            try:
                open_rows.append(_row(p))
            except Exception:
                # A single malformed position must not sink the whole
                # builder — degrade to no row for that ticker.
                continue
    except Exception:
        open_rows = []

    # Sort by actionability:
    #   AT_RISK_SL (0) / AT_RISK_TP (0) → most actionable
    #   NEAR_SL (1)   / NEAR_TP (1)
    #   MID_BAND (2)
    #   NO_SL_TP (3)
    _ORDER = {
        "AT_RISK_SL": 0,
        "AT_RISK_TP": 0,
        "NEAR_SL": 1,
        "NEAR_TP": 1,
        "MID_BAND": 2,
        "NO_SL_TP": 3,
    }

    def _sort_key(r: dict):
        band = r.get("proximity_band") or "NO_SL_TP"
        pos = r.get("corridor_pos")
        # Per-band secondary sort: rows closer to firing rank first.
        #   AT_RISK_SL: deeper breach (more-negative pos) first
        #     → use pos directly ascending (most-negative is smallest)
        #   AT_RISK_TP: bigger overshoot (further above 1.0) first
        #     → use -pos ascending (more-positive pos is more negative)
        #   NEAR_SL: closer to 0 (SL) first → use pos ascending
        #   NEAR_TP: closer to 1.0 (TP) first → use -pos ascending
        #   MID_BAND / NO_SL_TP: stable
        if band == "AT_RISK_SL":
            secondary = pos if pos is not None else 0.0
        elif band == "AT_RISK_TP":
            secondary = -pos if pos is not None else 0.0
        elif band == "NEAR_SL":
            secondary = pos if pos is not None else 0.0
        elif band == "NEAR_TP":
            secondary = -pos if pos is not None else 0.0
        else:
            secondary = 0.0
        return (_ORDER.get(band, 9), secondary, r.get("ticker") or "")

    open_rows.sort(key=_sort_key)
    verdict, headline = _aggregate_verdict(open_rows)

    counts: dict[str, int] = {
        "AT_RISK_SL": 0, "AT_RISK_TP": 0,
        "NEAR_SL": 0, "NEAR_TP": 0,
        "MID_BAND": 0, "NO_SL_TP": 0,
    }
    for r in open_rows:
        b = r["proximity_band"]
        if b in counts:
            counts[b] += 1

    return {
        "as_of": now.isoformat(),
        "verdict": verdict,
        "headline": headline,
        "n_positions": len(open_rows),
        "n_with_sl_tp": len(open_rows) - counts["NO_SL_TP"],
        "band_counts": counts,
        "positions": open_rows,
        "thresholds": {
            "near_sl_max": NEAR_SL_MAX,
            "near_tp_min": NEAR_TP_MIN,
        },
    }


def _cli() -> int:
    """One-shot ops view. Reads the live store read-only."""
    import json as _json
    try:
        from paper_trader.store import get_store
        store = get_store()
        positions = store.open_positions()
    except Exception as e:
        print(f"[exit_proximity] could not read store: {e}")
        return 2
    out = build_exit_proximity(positions)
    print(_json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli())

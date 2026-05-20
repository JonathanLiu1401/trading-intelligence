"""Peer-earnings-shock — implied 1σ dollar exposure on held LEVERAGED ETFs
from upcoming earnings of their constituents.

``/api/earnings-shock`` already reports the implied σ on each held name
**that itself reports** inside the horizon. A 67%-NVDA book reading
HELD_IMMINENT on NVDA's print is the surface that closes. Its sibling
``/api/etf-lookthrough`` separately reports the *indirect* dollar exposure
on each underlying that flows through a leveraged ETF (e.g. a $148 TQQQ
position carries ~$40 of NVDA exposure via the QQQ basket at 3x leverage).
Neither surface answers the **fusion** question every desk asks the
afternoon of a mega-cap print:

    *"NVDA earnings tonight. I hold $148 TQQQ. NVDA's market-implied 1σ
    is 7%. What is my INDIRECT NVDA $-shock on the TQQQ position?"*

The arithmetic is one multiplication:
``indirect_usd × sigma_pct / 100`` — but the desk has to manually fuse
``etf-lookthrough.etf_positions[i].breakdown[j].indirect_usd`` with the
``earnings_shock.events[k].sigma_pct`` row for the matching underlying.
Nothing on the live dashboard does that fusion today. A 29%-of-book TQQQ
position runs hot on every mega-cap tech print but no current surface
quantifies the exposure.

This is the **fusion** complement to ``earnings_shock`` (direct held
earnings) / ``etf_lookthrough`` (hidden indirect exposure). The three
together form the complete pre-earnings $-at-risk frame.

Single source of truth (AGENTS.md invariant #10):

* The earnings event set is ``event_calendar``'s ``events`` list verbatim
  (filtered to entries with parseable ``ticker`` + ``days_away`` ≤
  horizon).
* The ETF→underlying weight + leverage decomposition comes from
  ``etf_lookthrough``'s ``etf_positions`` verbatim — never recomputed.
* The σ comes from the caller-supplied ``sigma_provider`` callable
  (the builder/endpoint split ``earnings_shock`` / ``tail_risk`` /
  ``stress_scenarios`` use — the endpoint owns the network/yfinance
  hop). Returning ``None`` reads INSUFFICIENT_SIGMA, never raises.

State ladder mirrors ``earnings_shock`` / ``etf_lookthrough``:

* ``NO_DATA``        — empty book / no total_value / nothing priced.
* ``NO_ETF_HELD``    — book exists but contains no leveraged ETF in the
  look-through map (the ``etf_lookthrough.NO_ETF_HELD`` sibling state).
* ``NO_PEER_EVENTS`` — leveraged ETFs held but no constituent of any of
  them reports inside the horizon — calendar quiet for the basket.
* ``OK``             — at least one ETF × constituent-print pair surfaced.

Observational only — never gates Opus, never injected into the decision
prompt, no caps (AGENTS.md #2/#12 — the ``earnings_shock`` /
``etf_lookthrough`` / ``stress_scenarios`` precedent).
"""
from __future__ import annotations

from datetime import datetime, timezone

from .etf_lookthrough import build_etf_lookthrough

# Default horizon for what counts as "imminent" peer earnings. Wider than
# ``earnings_shock``'s HELD_IMMINENT (3d) because an ETF basket can be
# materially moved by the *first* mega-cap print in a clustered earnings
# week — a HELD_SOON underlying (5d out) still pressures TQQQ when the
# rest of the basket reports the same week. Aligns with the
# ``event_calendar`` default ``horizon_days=14.0`` upper bound but
# narrower so the panel stays decision-relevant.
DEFAULT_HORIZON_DAYS = 7.0

# Verdict bands on book-relative aggregate σ (the ``earnings_shock``
# ladder shape — same anchor so the operator's mental model carries
# across the three pre-earnings surfaces).
SEVERE_BOOK_PCT = 5.0     # |total σ| >= this ⇒ SEVERE
MODERATE_BOOK_PCT = 2.0   # |total σ| >= this ⇒ MODERATE
# Below MODERATE ⇒ LOW. NO_PEER_EVENTS / NO_DATA emit their own state.


def _z(x, n=2):
    try:
        return round(float(x), n)
    except (TypeError, ValueError):
        return None


def _f(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _row_verdict(total_sigma_book_pct: float | None) -> str:
    if total_sigma_book_pct is None:
        return "UNKNOWN"
    s = abs(total_sigma_book_pct)
    if s >= SEVERE_BOOK_PCT:
        return "SEVERE"
    if s >= MODERATE_BOOK_PCT:
        return "MODERATE"
    return "LOW"


def build_peer_earnings_shock(
        snapshot: dict,
        event_calendar_report: dict,
        sigma_provider,
        *,
        now: datetime | None = None,
        horizon_days: float = DEFAULT_HORIZON_DAYS,
        lookthrough_map: dict | None = None) -> dict:
    """Per-held-ETF indirect σ from imminent peer earnings. Pure, never raises.

    ``snapshot`` — ``strategy._portfolio_snapshot`` shape (``cash``,
    ``total_value``, enriched ``positions``); passed to
    ``build_etf_lookthrough`` to derive the indirect $-exposure per
    (ETF, underlying) pair.

    ``event_calendar_report`` — output of ``build_event_calendar``;
    its ``events`` list is filtered to ``ticker``s with parseable
    ``days_away`` inside ``horizon_days`` (the SSOT for what counts
    as a peer print).

    ``sigma_provider`` — callable ``ticker → float | None`` returning
    the underlying's implied 1σ in **percent** (e.g. 7.0 ⇒ 7%). May
    return ``None`` for unknowns; the row surfaces INSUFFICIENT_SIGMA
    and the aggregate excludes it.

    ``horizon_days`` — peer-events beyond this are dropped (clutter
    control). Default 7d.

    ``lookthrough_map`` — test-seam override for the ETF table;
    production uses ``etf_lookthrough._ETF_LOOKTHROUGH``.

    Returns a stable-shape dict with ``state``, one-line ``headline``,
    per-ETF rows (each with per-underlying breakdown), and book-wide
    aggregates. Never raises.
    """
    now = now or datetime.now(timezone.utc)

    try:
        snap = snapshot or {}
        total_value = _f(snap.get("total_value"))

        if total_value <= 0:
            return {
                "as_of": now.isoformat(timespec="seconds"),
                "state": "NO_DATA",
                "headline": "no priced book — peer-earnings σ withheld",
                "horizon_days": horizon_days,
                "total_value": _z(total_value) or 0.0,
                "n_etfs_at_risk": 0,
                "n_peer_events": 0,
                "total_indirect_sigma_dollar": None,
                "total_indirect_sigma_book_pct": None,
                "verdict": "NO_DATA",
                "etf_rows": [],
            }

        # SSOT compose: indirect $-exposure per (ETF, underlying).
        lookthrough = build_etf_lookthrough(
            snap,
            lookthrough_map=lookthrough_map,
        )
        lt_state = lookthrough.get("state")
        etf_positions = lookthrough.get("etf_positions") or []
        if lt_state == "NO_DATA" or not etf_positions:
            # NO_ETF_HELD when the lookthrough cleanly says so; NO_DATA
            # only when total_value is missing (caught above) or the
            # lookthrough itself bailed on a malformed snapshot.
            state = (
                "NO_ETF_HELD" if lt_state == "NO_ETF_HELD"
                else "NO_DATA"
            )
            head = (
                "no leveraged ETF held — peer-earnings shock not applicable"
                if state == "NO_ETF_HELD"
                else "look-through unavailable — peer-earnings σ withheld"
            )
            return {
                "as_of": now.isoformat(timespec="seconds"),
                "state": state,
                "headline": head,
                "horizon_days": horizon_days,
                "total_value": _z(total_value, 2),
                "n_etfs_at_risk": 0,
                "n_peer_events": 0,
                "total_indirect_sigma_dollar": None,
                "total_indirect_sigma_book_pct": None,
                "verdict": state,
                "etf_rows": [],
            }

        # SSOT: which underlyings have imminent earnings.
        events = (event_calendar_report or {}).get("events") or []
        peer_set: dict = {}  # ticker → days_away (closest)
        for ev in events:
            if not isinstance(ev, dict):
                continue
            tk = (ev.get("ticker") or "").upper()
            if not tk:
                continue
            da = ev.get("days_away")
            try:
                da = float(da)
            except (TypeError, ValueError):
                continue
            if da > horizon_days or da < 0:
                continue
            # Keep the soonest date if duplicates appear.
            if tk not in peer_set or da < peer_set[tk]:
                peer_set[tk] = da

        n_peer_events = len(peer_set)
        if n_peer_events == 0:
            return {
                "as_of": now.isoformat(timespec="seconds"),
                "state": "NO_PEER_EVENTS",
                "headline": (
                    f"no constituent of any held leveraged ETF reports "
                    f"within {horizon_days:.0f}d — basket calendar quiet"),
                "horizon_days": horizon_days,
                "total_value": _z(total_value, 2),
                "n_etfs_at_risk": 0,
                "n_peer_events": 0,
                "total_indirect_sigma_dollar": 0.0,
                "total_indirect_sigma_book_pct": 0.0,
                "verdict": "NO_PEER_EVENTS",
                "etf_rows": [],
            }

        # Per-ETF compose: filter the breakdown to peer-event constituents
        # only, multiply by their implied σ, surface row + aggregate.
        etf_rows: list[dict] = []
        # Book-wide aggregate uses the |indirect_sigma_dollar| sum
        # (the ``earnings_shock`` "worst-case all-surprise-same-way"
        # convention — quadrature understates the correlated case that
        # actually drives correlated book moves; earnings inside the
        # same basket the same week are NOT independent).
        total_indirect_sigma_dollar = 0.0
        n_etfs_at_risk = 0

        for etf in etf_positions:
            tk = (etf.get("ticker") or "").upper()
            etf_position_usd = _f(etf.get("position_usd"))
            leverage = _f(etf.get("leverage"), 1.0)
            breakdown = etf.get("breakdown") or []
            # Filter to underlyings that have imminent earnings.
            underlyings: list[dict] = []
            etf_sum_sigma_dollar = 0.0
            for h in breakdown:
                if not isinstance(h, dict):
                    continue
                u_tk = (h.get("underlying") or "").upper()
                if not u_tk or u_tk not in peer_set:
                    continue
                indirect_usd = _f(h.get("indirect_usd"))
                # Skip zero-indirect rows (a position_usd=0 ETF or a
                # weight=0 underlying — shouldn't happen but be safe).
                if indirect_usd == 0.0:
                    continue
                try:
                    sigma_pct = sigma_provider(u_tk) if sigma_provider else None
                except Exception:
                    sigma_pct = None
                if sigma_pct is None:
                    underlyings.append({
                        "underlying": u_tk,
                        "weight_pct": _z(h.get("weight_pct"), 2),
                        "days_away": _z(peer_set.get(u_tk), 2),
                        "indirect_usd": _z(indirect_usd, 2),
                        "sigma_pct": None,
                        "indirect_sigma_dollar": None,
                        "indirect_sigma_book_pct": None,
                        "row_state": "INSUFFICIENT_SIGMA",
                    })
                    continue
                try:
                    sigma_pct = float(sigma_pct)
                except (TypeError, ValueError):
                    underlyings.append({
                        "underlying": u_tk,
                        "weight_pct": _z(h.get("weight_pct"), 2),
                        "days_away": _z(peer_set.get(u_tk), 2),
                        "indirect_usd": _z(indirect_usd, 2),
                        "sigma_pct": None,
                        "indirect_sigma_dollar": None,
                        "indirect_sigma_book_pct": None,
                        "row_state": "INSUFFICIENT_SIGMA",
                    })
                    continue
                indirect_sigma_dollar = indirect_usd * (sigma_pct / 100.0)
                indirect_sigma_book_pct = (
                    indirect_sigma_dollar / total_value * 100.0
                )
                etf_sum_sigma_dollar += abs(indirect_sigma_dollar)
                underlyings.append({
                    "underlying": u_tk,
                    "weight_pct": _z(h.get("weight_pct"), 2),
                    "days_away": _z(peer_set.get(u_tk), 2),
                    "indirect_usd": _z(indirect_usd, 2),
                    "sigma_pct": _z(sigma_pct, 2),
                    "indirect_sigma_dollar": _z(indirect_sigma_dollar, 2),
                    "indirect_sigma_book_pct": _z(indirect_sigma_book_pct, 4),
                    "row_state": "OK",
                })

            if not underlyings:
                continue
            n_etfs_at_risk += 1
            scored = any(u["row_state"] == "OK" for u in underlyings)
            etf_sigma_book_pct = (
                etf_sum_sigma_dollar / total_value * 100.0 if scored else None
            )
            total_indirect_sigma_dollar += etf_sum_sigma_dollar
            # Sort underlyings by absolute indirect_sigma_dollar DESC
            # (loudest peer first); insufficient-σ rows sink to the end.
            underlyings.sort(
                key=lambda u: (
                    -1 if u["row_state"] != "OK" else 0,
                    -abs(_f(u.get("indirect_sigma_dollar"))),
                    u["underlying"],
                ),
            )
            etf_rows.append({
                "etf_ticker": tk,
                "etf_position_usd": _z(etf_position_usd, 2),
                "leverage": _z(leverage, 2),
                "n_peer_events": sum(
                    1 for u in underlyings if u["row_state"] == "OK"),
                "n_peer_events_total": len(underlyings),
                "sum_indirect_sigma_dollar": (
                    _z(etf_sum_sigma_dollar, 2) if scored else None),
                "sum_indirect_sigma_book_pct": (
                    _z(etf_sigma_book_pct, 4) if scored else None),
                "underlyings": underlyings,
            })

        if not etf_rows:
            # Lookthrough produced ETFs but none of their breakdowns
            # intersect the peer-event set — same calendar-quiet branch
            # but at the per-ETF level. Surface NO_PEER_EVENTS so the
            # operator gets the same one-line silence either way.
            return {
                "as_of": now.isoformat(timespec="seconds"),
                "state": "NO_PEER_EVENTS",
                "headline": (
                    f"no constituent of any held leveraged ETF reports "
                    f"within {horizon_days:.0f}d — basket calendar quiet"),
                "horizon_days": horizon_days,
                "total_value": _z(total_value, 2),
                "n_etfs_at_risk": 0,
                "n_peer_events": n_peer_events,
                "total_indirect_sigma_dollar": 0.0,
                "total_indirect_sigma_book_pct": 0.0,
                "verdict": "NO_PEER_EVENTS",
                "etf_rows": [],
            }

        # Sort ETF rows by aggregate sigma_dollar DESC (loudest first).
        # An ETF whose underlyings are ALL insufficient (no scored row)
        # surfaces with sum=None — sort it to the bottom with negative.
        etf_rows.sort(
            key=lambda r: (
                0 if r["sum_indirect_sigma_dollar"] is not None else 1,
                -abs(_f(r.get("sum_indirect_sigma_dollar"))),
                r["etf_ticker"],
            ),
        )

        total_indirect_sigma_book_pct = (
            total_indirect_sigma_dollar / total_value * 100.0
            if total_indirect_sigma_dollar > 0 else 0.0
        )

        verdict = _row_verdict(total_indirect_sigma_book_pct)

        # Headline picks the (ETF, underlying) pair with the loudest
        # indirect_sigma_dollar — the single most actionable line.
        loudest = None
        for r in etf_rows:
            for u in r["underlyings"]:
                if u["row_state"] != "OK":
                    continue
                if (loudest is None or
                        abs(_f(u["indirect_sigma_dollar"])) >
                        abs(_f(loudest[1]["indirect_sigma_dollar"]))):
                    loudest = (r, u)

        if loudest is not None:
            r, u = loudest
            headline = (
                f"Peer-earnings shock: {u['underlying']} prints in "
                f"{u['days_away']:.1f}d at σ±{u['sigma_pct']:.1f}% — "
                f"{r['etf_ticker']} indirect ±${u['indirect_sigma_dollar']:.2f} "
                f"({u['indirect_sigma_book_pct']:+.2f}% of book). "
                f"Aggregate across {n_etfs_at_risk} ETF"
                f"{'' if n_etfs_at_risk == 1 else 's'}: "
                f"±${total_indirect_sigma_dollar:.2f} "
                f"({total_indirect_sigma_book_pct:.2f}% of book, "
                f"{verdict})."
            )
        else:
            # Some peer events matched ETFs but every σ came back None.
            headline = (
                f"Peer-earnings shock: {n_peer_events} constituent print"
                f"{'' if n_peer_events == 1 else 's'} within "
                f"{horizon_days:.0f}d across {n_etfs_at_risk} held ETF"
                f"{'' if n_etfs_at_risk == 1 else 's'}, σ withheld "
                f"(no historical reference for any underlying).")
            verdict = "INSUFFICIENT_SIGMA"

        return {
            "as_of": now.isoformat(timespec="seconds"),
            "state": "OK",
            "headline": headline,
            "horizon_days": horizon_days,
            "total_value": _z(total_value, 2),
            "n_etfs_at_risk": n_etfs_at_risk,
            "n_peer_events": n_peer_events,
            "total_indirect_sigma_dollar": _z(total_indirect_sigma_dollar, 2),
            "total_indirect_sigma_book_pct": _z(total_indirect_sigma_book_pct, 4),
            "verdict": verdict,
            "etf_rows": etf_rows,
        }
    except Exception:
        # _safe contract — any unexpected fault degrades to one honest
        # line; never propagates (the etf_lookthrough / earnings_shock
        # precedent).
        return {
            "as_of": now.isoformat(timespec="seconds"),
            "state": "NO_DATA",
            "headline": "peer-earnings σ fault — no panel this cycle",
            "horizon_days": horizon_days,
            "total_value": 0.0,
            "n_etfs_at_risk": 0,
            "n_peer_events": 0,
            "total_indirect_sigma_dollar": None,
            "total_indirect_sigma_book_pct": None,
            "verdict": "NO_DATA",
            "etf_rows": [],
        }


if __name__ == "__main__":  # smoke against live snapshot
    import json as _json

    from paper_trader.analytics.event_calendar import build_event_calendar
    from paper_trader.store import get_store
    from paper_trader.strategy import WATCHLIST

    s = get_store()
    pos = s.open_positions()
    pf = s.get_portfolio()
    snap = {
        "cash": pf.get("cash"),
        "total_value": pf.get("total_value"),
        "positions": pos,
    }
    ec = build_event_calendar(
        pos,
        {(p.get("ticker") or "").upper() for p in pos} | set(WATCHLIST[:5]),
    )
    rep = build_peer_earnings_shock(
        snap, ec, sigma_provider=lambda _t: 7.0)
    print(rep["headline"])
    print("---")
    print(_json.dumps(rep, indent=2, default=str))

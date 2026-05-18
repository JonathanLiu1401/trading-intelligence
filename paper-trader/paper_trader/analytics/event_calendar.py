"""Upcoming-earnings awareness, fed into the live Opus decision prompt.

The behavioural mirrors (``self_review`` / ``track_record`` / ``risk_mirror``)
gave the live trader its *backward-looking* feedback. None of them — nor
anything else on the decision path — told it about the single biggest
*forward* scheduled risk a discretionary desk tracks: **earnings**. A position
held into a print can gap 10–20% overnight; adding the day before one, blind,
is the classic avoidable mistake. ``/api/earnings-risk`` already surfaced this
on the dashboard, but the decision engine itself never saw it — exactly the
gap the self-review module was built to close, one dimension over (forward,
not backward).

Single source of truth (AGENTS.md invariant #10):

* The earnings data is digital-intern's, written by
  ``collectors/earnings_calendar.py`` to ``data/earnings_calendar.json`` and
  served verbatim at digital-intern ``/api/earnings``. This block reads **that
  same file directly from disk** — it never re-collects earnings dates.
* The held/watch tier rule (``HELD_IMMINENT`` ≤3d, ``HELD_SOON`` within
  horizon, ``WATCH`` in-play-not-held) is the same one ``/api/earnings-risk``
  applies, so the prompt block and the dashboard can never disagree.

**Hot-path safety (load-bearing).** ``decide()`` runs every 60s and must never
be sunk by a diagnostics fault. digital-intern's ``:8080`` is documented to
hang; a network hop on the live cycle is forbidden. So this reads the JSON
**file**, not the endpoint (the ``signals.py`` filesystem precedent), and is
``_safe``-wrapped end-to-end: a missing / stale / corrupt / unparseable file
degrades to one honest line, **never** an exception. ``days_away`` is
recomputed from ``earnings_date`` vs ``now`` (exactly as digital-intern's
``api_earnings`` does) so a stale snapshot still yields accurate timing.

**Observational, never prescriptive.** Same contract as ``risk_mirror``
(AGENTS.md #2/#12): it states facts (which name reports, when) and reaffirms
full autonomy in its preamble. It issues no directive, imposes no cap, and
never gates a trade.

Pure and deterministic (``now`` and ``calendar_path`` injectable).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

# Ordered candidate locations for digital-intern's earnings snapshot. The
# freshest *readable* one wins (``_pick_freshest``), mirroring the
# ``signals._db_path`` freshness discipline (invariant #15) so a stale USB
# copy can never shadow a fresher local one. Only consulted when the caller
# does not pass an explicit ``calendar_path`` (the production path).
_CANDIDATE_PATHS = (
    Path("/media/zeph/projects/digital-intern/data/earnings_calendar.json"),
    Path("/home/zeph/trading-intelligence/digital-intern/data/earnings_calendar.json"),
    Path("/home/zeph/digital-intern/data/earnings_calendar.json"),
)

_IMMINENT_DAYS = 3.0   # held & within this ⇒ HELD_IMMINENT (api_earnings rule)
_PAST_GRACE_DAYS = -0.5  # mirrors api_earnings' "drop once it has reported"

_PREAMBLE = (
    "EARNINGS CALENDAR (scheduled earnings on your held / in-play names, for "
    "your awareness only — facts about event timing, NOT directives or "
    "limits; you retain complete autonomy over the next decision):"
)


def _parse_dt(s: str | None) -> datetime | None:
    """Parse an ISO timestamp, treating a naive value as UTC. Returns ``None``
    on anything unparseable (never raises — the _safe contract)."""
    if not s or not isinstance(s, str):
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _pick_freshest(paths) -> Path | None:
    """Of the readable candidates, the one whose snapshot ``as_of`` is newest.

    Unreadable / missing / unparseable candidates are skipped. Returns
    ``None`` when none can be read — the caller then degrades to the honest
    fallback line. Order-independent."""
    best: Path | None = None
    best_as_of: datetime | None = None
    for p in paths:
        try:
            snap = json.loads(Path(p).read_text())
        except (OSError, ValueError, TypeError):
            continue
        as_of = _parse_dt(snap.get("as_of")) if isinstance(snap, dict) else None
        # A readable file with no/garbage as_of still beats nothing, but a
        # dated one always wins over an undated one.
        if best is None:
            best, best_as_of = Path(p), as_of
            continue
        if as_of is not None and (best_as_of is None or as_of > best_as_of):
            best, best_as_of = Path(p), as_of
    return best


def _load_snapshot(calendar_path) -> tuple[dict | None, Path | None]:
    """Resolve + read the earnings snapshot. ``(snap, path)`` on success,
    ``(None, path|None)`` on any failure. Never raises."""
    path = Path(calendar_path) if calendar_path is not None else \
        _pick_freshest(_CANDIDATE_PATHS)
    if path is None:
        return None, None
    try:
        snap = json.loads(Path(path).read_text())
    except (OSError, ValueError, TypeError):
        return None, path
    return (snap if isinstance(snap, dict) else None), path


def _tier(in_port: bool, days_away: float | None) -> str:
    if in_port and days_away is not None and days_away <= _IMMINENT_DAYS:
        return "HELD_IMMINENT"
    if in_port:
        return "HELD_SOON"
    return "WATCH"


_TIER_RANK = {"HELD_IMMINENT": 0, "HELD_SOON": 1, "WATCH": 2}


def _event_line(e: dict) -> str:
    when = ""
    ed = _parse_dt(e.get("earnings_date"))
    if ed is not None:
        when = f" on {ed.date().isoformat()}"
    note = {
        "HELD_IMMINENT": " [HELD_IMMINENT — you hold this into the print]",
        "HELD_SOON": " [HELD_SOON — you hold this]",
        "WATCH": " [WATCH]",
    }.get(e["tier"], "")
    return (f"  {e['ticker']} — earnings in {e['days_away']:.1f}d"
            f"{when}{note}")


def build_event_calendar(positions: list[dict],
                          names_in_play,
                          calendar_path=None,
                          now: datetime | None = None,
                          horizon_days: float = 14.0) -> dict:
    """Compose the upcoming-earnings awareness block.

    ``positions`` — open positions (``{ticker, …}``); used only to know what
    is *held* (the HELD_* tiers). ``names_in_play`` — the
    ``strategy._names_in_play`` set (held ∪ top-signal tickers ∪ top-5
    watchlist) so the WATCH tier matches the same "what matters this cycle"
    universe the quant / track-record blocks use. ``calendar_path`` —
    explicit override (tests); otherwise the freshest known snapshot.
    ``horizon_days`` — WATCH names beyond this are dropped as prompt noise; a
    *held* name's print is never hidden regardless of distance.

    Returns ``{as_of, summary, prompt_block, events, source_ok,
    source_age_hours}``. Pure; never raises.
    """
    now = now or datetime.now(timezone.utc)
    held = {
        (p.get("ticker") or "").upper()
        for p in (positions or []) if p.get("ticker")
    }
    in_play = {str(t).upper() for t in (names_in_play or set())}

    snap, _path = _load_snapshot(calendar_path)
    source_ok = snap is not None

    events: list[dict] = []
    source_age_hours = None
    if source_ok:
        as_of = _parse_dt(snap.get("as_of"))
        if as_of is not None:
            source_age_hours = round((now - as_of).total_seconds() / 3600.0, 2)
        for ev in (snap.get("events") or []):
            tk = (ev.get("ticker") or "").upper()
            if not tk:
                continue
            ed = _parse_dt(ev.get("earnings_date"))
            if ed is None:
                continue
            days_away = round((ed - now).total_seconds() / 86400.0, 2)
            if days_away < _PAST_GRACE_DAYS:
                continue  # already reported — never leak a past event
            in_port = tk in held
            on_play = tk in in_play
            if not in_port and not on_play:
                continue  # not actionable this cycle — keep the prompt lean
            if not in_port and days_away > horizon_days:
                continue  # distant WATCH name → noise
            events.append({
                "ticker": tk,
                "earnings_date": ev.get("earnings_date"),
                "days_away": days_away,
                "held": in_port,
                "tier": _tier(in_port, days_away),
            })

    events.sort(key=lambda e: (_TIER_RANK.get(e["tier"], 9), e["days_away"]))

    if events:
        body = "\n".join(_event_line(e) for e in events)
        prompt_block = f"{_PREAMBLE}\n{body}"
        n_imm = sum(1 for e in events if e["tier"] == "HELD_IMMINENT")
        n_soon = sum(1 for e in events if e["tier"] == "HELD_SOON")
        n_watch = sum(1 for e in events if e["tier"] == "WATCH")
        bits = []
        if n_imm:
            bits.append(f"{n_imm} held-imminent")
        if n_soon:
            bits.append(f"{n_soon} held-soon")
        if n_watch:
            bits.append(f"{n_watch} watch")
        summary = " · ".join(bits)
    elif not source_ok:
        prompt_block = (f"{_PREAMBLE}\n  (earnings calendar unavailable — no "
                        f"scheduled-event awareness this cycle)")
        summary = "no earnings data"
    else:
        prompt_block = (f"{_PREAMBLE}\n  No earnings within "
                        f"{horizon_days:.0f}d for your held or in-play names.")
        summary = "no scheduled earnings on held/in-play names"

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "summary": summary,
        "prompt_block": prompt_block,
        "events": events,
        "source_ok": source_ok,
        "source_age_hours": source_age_hours,
    }


if __name__ == "__main__":  # smoke test against the live snapshot
    import json as _json

    from paper_trader.store import get_store
    from paper_trader.strategy import WATCHLIST

    s = get_store()
    pos = s.open_positions()
    rep = build_event_calendar(
        pos,
        {p.get("ticker") for p in pos} | set(WATCHLIST[:5]),
    )
    print(rep["prompt_block"])
    print("\n---\n")
    print(_json.dumps({k: v for k, v in rep.items() if k != "prompt_block"},
                       indent=2, default=str))

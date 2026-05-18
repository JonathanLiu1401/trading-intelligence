"""Forward FOMC rate-decision awareness, fed into the live Opus prompt.

``event_calendar`` gave the desk forward *single-name earnings* awareness —
the one scheduled risk a discretionary desk tracks per ticker. This is its
**macro sibling, one dimension over**: scheduled **FOMC rate decisions**.
A rate-decision surprise moves the *whole* book in one instant, and this
watchlist is leveraged-ETF heavy (SOXL / TQQQ / NVDL / SOXS …) — exactly the
instruments that gap hardest on a Fed surprise. The system's own 5h Opus
briefings repeatedly *lead* with macro (bond rout, 10Y, FOMC), yet the live
decision engine itself had **zero** forward macro awareness across 47
analytics modules. A leveraged book entering the rate-decision instant blind
is the macro analog of the exact "added the day before an earnings print,
blind" mistake ``event_calendar`` was built to close.

**Scope: FOMC only, by verifiability discipline.** FOMC 2026 dates are
fully verifiable (federalreserve.gov — fetched + confirmed, all 8 meetings).
BLS CPI / Employment-Situation forward dates are *not* reliably verifiable
from here (bls.gov hard-blocks every fetch with HTTP 403; archive-URL dates
conflict with summaries by ±2d; Jul–Dec unreleased). Encoding an unverified
date on the live decision path would mislead Opus — the cardinal sin a
diagnostics block must not commit. CPI/NFP are a deliberate future extension
*pending a verifiable source*, **not** an oversight: add their instants to a
parallel table behind the same honesty bound when one exists.

**Hot-path safety.** ``decide()`` runs every 60s. This is a **pure static
table + date arithmetic** — no file I/O, no network, no import beyond stdlib
(even safer than ``event_calendar``, which does disk reads). ``now`` is
injectable; the function is deterministic and never raises (``_safe``).

**Honesty bound (load-bearing).** A hardcoded calendar that silently runs
out is a latent landmine. ``SCHEDULE_VALID_THROUGH`` is exactly the last
encoded instant; once ``now`` passes it the block degrades to one honest
line — never a fabricated event. ``test_macro_calendar`` locks the
table↔bound no-drift so extending one without the other fails RED.

**Market-wide, not per-ticker.** Unlike ``event_calendar`` this takes no
positions / names — an FOMC decision is relevant to a flat book too
(entering the day before, blind, is the mistake). Always rendered.

**Observational, never prescriptive.** Same contract as ``event_calendar``
(AGENTS.md invariants #2/#12): states facts (which macro event, when) and
reaffirms full autonomy in its preamble. No directive, no cap, never gates.
"""
from __future__ import annotations

from datetime import datetime, timezone

# ── The federalreserve.gov-verified 2026 FOMC schedule ──────────────────────
# Each meeting is two days; the market-moving instant is the SECOND day's
# 14:00 ET policy statement (press conference 14:30 ET). ET→UTC is resolved
# here so there is no tz-library dependency on the hot path:
#   • 2026 US DST runs Sun Mar 8 → Sun Nov 1.
#   • Jan & Dec statements fall in EST (UTC-5)  → 14:00 ET == 19:00 UTC.
#   • Mar–Oct statements fall in EDT (UTC-4)    → 14:00 ET == 18:00 UTC.
# Verified from federalreserve.gov/monetarypolicy/fomccalendars.htm on
# 2026-05-18 (Jan 27-28, Mar 17-18, Apr 28-29, Jun 16-17, Jul 28-29,
# Sep 15-16, Oct 27-28, Dec 8-9). A regression that edits any instant fails
# `test_encoded_table_is_exactly_the_verified_2026_fomc_schedule`.
_FOMC_2026 = (
    "2026-01-28T19:00:00+00:00",
    "2026-03-18T18:00:00+00:00",
    "2026-04-29T18:00:00+00:00",
    "2026-06-17T18:00:00+00:00",
    "2026-07-29T18:00:00+00:00",
    "2026-09-16T18:00:00+00:00",
    "2026-10-28T18:00:00+00:00",
    "2026-12-09T19:00:00+00:00",
)


def _parse(s: str) -> datetime:
    d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


#: The last instant the static table covers. Past this the block degrades to
#: an honest "schedule not loaded" line. Kept exactly equal to the max
#: encoded instant by `test_schedule_valid_through_equals_last_encoded_instant`.
SCHEDULE_VALID_THROUGH: datetime = max(_parse(s) for s in _FOMC_2026)

_IMMINENT_HOURS = 24.0    # < this ⇒ IMMINENT_HOURS (sub-day precision)
_IMMINENT_DAYS = 3.0      # <= this ⇒ IMMINENT
_PAST_GRACE_HOURS = -2.0  # keep the just-printed decision through the ~2h
                          # immediate-reaction window, then drop it (a past
                          # decision is no longer "upcoming" and must never
                          # leak as a future event)

_PREAMBLE = (
    "MACRO CALENDAR (scheduled FOMC rate decisions — market-wide events that "
    "move the whole book, leveraged ETFs most violently; for your awareness "
    "only, NOT directives or limits — you retain complete autonomy over the "
    "next decision):"
)

_TIER_NOTE = {
    "IMMINENT_HOURS": "the rate decision lands within 24h",
    "IMMINENT": "a rate decision lands within 3 days",
    "UPCOMING": "on the horizon",
}
_TIER_RANK = {"IMMINENT_HOURS": 0, "IMMINENT": 1, "UPCOMING": 2}


def _normalize_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if not isinstance(now, datetime):
        return datetime.now(timezone.utc)
    return now if now.tzinfo else now.replace(tzinfo=timezone.utc)


def _timing_phrase(tier: str, hours_away: float, days_away: float) -> str:
    if tier == "IMMINENT_HOURS":
        if hours_away >= 0:
            return f"in {hours_away:.1f}h"
        return f"released {abs(hours_away):.1f}h ago (reaction window)"
    return f"in {days_away:.1f}d"


def _event_line(e: dict) -> str:
    note = _TIER_NOTE.get(e["tier"], "")
    timing = _timing_phrase(e["tier"], e["hours_away"], e["days_away"])
    return (f"  FOMC rate-decision statement {timing} — {e['when_et']} "
            f"[{e['tier']} — {note}]")


def build_macro_calendar(now: datetime | None = None,
                         horizon_days: float = 14.0) -> dict:
    """Compose the forward FOMC rate-decision awareness block.

    ``now`` — injectable for tests / determinism. ``horizon_days`` — events
    beyond this are dropped as prompt noise (FOMC fires 8×/yr so within a
    14d horizon there is at most one). Returns ``{as_of, summary,
    prompt_block, events, source_ok, schedule_valid_through}``. Pure; never
    raises (the ``_safe`` contract — a diagnostics fault must not sink a
    live trading cycle)."""
    try:
        now = _normalize_now(now)
        svt_iso = SCHEDULE_VALID_THROUGH.isoformat()

        if now > SCHEDULE_VALID_THROUGH:
            return {
                "as_of": now.isoformat(timespec="seconds"),
                "summary": "no macro schedule loaded",
                "prompt_block": (
                    f"{_PREAMBLE}\n  (FOMC schedule not loaded beyond "
                    f"{SCHEDULE_VALID_THROUGH.date().isoformat()} — no "
                    f"rate-decision awareness this cycle; refresh "
                    f"_FOMC_2026)"),
                "events": [],
                "source_ok": False,
                "schedule_valid_through": svt_iso,
            }

        events: list[dict] = []
        for iso in _FOMC_2026:
            when = _parse(iso)
            secs = (when - now).total_seconds()
            hours_away = secs / 3600.0
            days_away = secs / 86400.0
            if hours_away < _PAST_GRACE_HOURS:
                continue  # printed & past the reaction window
            if hours_away < _IMMINENT_HOURS:
                tier = "IMMINENT_HOURS"
            elif days_away <= _IMMINENT_DAYS:
                tier = "IMMINENT"
            elif days_away <= horizon_days:
                tier = "UPCOMING"
            else:
                continue  # beyond horizon → prompt noise
            events.append({
                "event": "FOMC",
                "label": "FOMC rate-decision statement",
                "when_utc": iso,
                "when_et": f"{when.date().isoformat()} 14:00 ET",
                "hours_away": round(hours_away, 2),
                "days_away": round(days_away, 2),
                "tier": tier,
            })

        events.sort(key=lambda e: (_TIER_RANK.get(e["tier"], 9),
                                   _parse(e["when_utc"])))

        if events:
            body = "\n".join(_event_line(e) for e in events)
            prompt_block = f"{_PREAMBLE}\n{body}"
            head = events[0]
            timing = _timing_phrase(head["tier"], head["hours_away"],
                                    head["days_away"])
            summary = f"FOMC {timing} ({head['tier']})"
        else:
            prompt_block = (f"{_PREAMBLE}\n  No FOMC rate decision within "
                            f"{horizon_days:.0f}d.")
            summary = f"no FOMC within {horizon_days:.0f}d"

        return {
            "as_of": now.isoformat(timespec="seconds"),
            "summary": summary,
            "prompt_block": prompt_block,
            "events": events,
            "source_ok": True,
            "schedule_valid_through": svt_iso,
        }
    except Exception as e:  # the _safe contract — never sink a live cycle
        return {
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "summary": "macro calendar error",
            "prompt_block": (f"{_PREAMBLE}\n  (macro calendar unavailable "
                             f"this cycle)"),
            "events": [],
            "source_ok": False,
            "schedule_valid_through": SCHEDULE_VALID_THROUGH.isoformat(),
            "error": str(e),
        }


if __name__ == "__main__":  # smoke
    import json as _json

    rep = build_macro_calendar()
    print(rep["prompt_block"])
    print("\n---\n")
    print(_json.dumps({k: v for k, v in rep.items() if k != "prompt_block"},
                      indent=2, default=str))

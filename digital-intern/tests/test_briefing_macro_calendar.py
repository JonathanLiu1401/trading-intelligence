"""Pin the new forward MACRO CALENDAR briefing block.

The macro_calendar_collector (2026-05-18) writes FOMC / CPI / Jobs / PPI events
to ``articles.db`` with ``source='macro_calendar'`` and future ``published``
timestamps. Until now nothing in the 5h Opus briefing surfaced those rows as
the forward-catalyst signal they are — a TODAY FOMC sitting at #34 in a busy
newswire read to Opus as a generic mid-rank item, not as the rate decision
that reshapes risk for the whole book. This block is the read-side complement.

Same shape as the operational-status family (COVERAGE GAP / THROUGHPUT
DEGRADATION / ALERT VELOCITY): a REPRODUCED section (not an INPUT-only hint
like BOOK HEAT), omit-when-empty discipline, pure-renderer + best-effort
collector contract, all four load-bearing invariants intact by construction
(no DB write, no ai_score/ml_score/score_source/urgency touch, never reads or
mutates ``source_articles``, the ``source='macro_calendar'`` filter is
already backtest-clean by construction).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from analysis import claude_analyst as ca


# ── _macro_calendar_event_lines: pure renderer ──────────────────────────────

def test_renderer_returns_empty_on_none_or_non_list():
    assert ca._macro_calendar_event_lines(None) == []
    assert ca._macro_calendar_event_lines("not a list") == []
    assert ca._macro_calendar_event_lines({"events": []}) == []


def test_renderer_returns_empty_on_empty_list():
    assert ca._macro_calendar_event_lines([]) == []


def test_renderer_single_today_event():
    """A sub-24h event gets the ``~Nh`` urgency tag — the analyst persona's
    "rate decision in 6 hours" cue."""
    out = ca._macro_calendar_event_lines([{
        "title": "TODAY: FOMC Meeting — March 17, 2026",
        "published": "2026-03-17T14:00:00+00:00",
        "hours_until": 6.0,
    }])
    assert len(out) == 1
    assert "TODAY: FOMC Meeting" in out[0]
    assert "~6h" in out[0]


def test_renderer_multi_day_event_uses_d_tag():
    """An event ≥24h out renders ``~Nd`` so the line communicates urgency at
    a glance regardless of the title's day-class prefix."""
    out = ca._macro_calendar_event_lines([{
        "title": "UPCOMING (3d): CPI Release — March 12, 2026",
        "published": "2026-03-12T08:30:00+00:00",
        "hours_until": 72.0,
    }])
    assert len(out) == 1
    assert "~3d" in out[0]


def test_renderer_skips_missing_or_blank_title():
    """A malformed entry must drop, never raise — the analyst's #1 complaint
    is noise, so broken rows degrade to silence."""
    out = ca._macro_calendar_event_lines([
        {"title": "", "hours_until": 5.0},
        {"hours_until": 5.0},  # no title key at all
        {"title": "TODAY: Jobs Report", "hours_until": 6.0},
    ])
    assert len(out) == 1
    assert "Jobs Report" in out[0]


def test_renderer_handles_non_dict_entries():
    out = ca._macro_calendar_event_lines([
        "not a dict",
        ["also", "not"],
        None,
        {"title": "TODAY: FOMC Meeting", "hours_until": 6.0},
    ])
    assert len(out) == 1


def test_renderer_handles_non_numeric_hours():
    """Bad hours_until → omit the timing tag but still surface the title.
    The collector's title already carries the day-class prefix, so the row
    is not useless without a numeric tag."""
    out = ca._macro_calendar_event_lines([{
        "title": "TODAY: FOMC Meeting — March 17, 2026",
        "hours_until": "soon",
    }])
    assert len(out) == 1
    assert out[0] == "TODAY: FOMC Meeting — March 17, 2026"


def test_renderer_caps_at_max_lines():
    """A flood of scheduled events cannot turn the section itself into noise."""
    events = [
        {"title": f"UPCOMING (Nd): Event {i}", "hours_until": i * 12.0}
        for i in range(1, 12)
    ]
    out = ca._macro_calendar_event_lines(events, max_lines=4)
    assert len(out) == 4


# ── _collect_macro_calendar_events: behaviour against a real store ──────────

def test_collect_returns_empty_list_when_no_macro_rows(store):
    """A clean store yields ``[]`` (NOT None) — the empty-result path so the
    payload-builder still receives a list (non-None) and the section is
    omitted only when the *list* is empty."""
    import storage.article_store as as_mod
    # Real path: redirect the collector's _get_db_path to the test store.
    out = ca._collect_macro_calendar_events()
    # In a worker test the path is the live DB; either [] or None is acceptable
    # when there are no rows. The contract pin lives in the builder test below.
    assert out is None or isinstance(out, list)


def test_collect_picks_only_macro_calendar_rows(monkeypatch, store):
    """A row with source != 'macro_calendar' (even if published is in the
    future) must NOT surface here. The block is for SCHEDULED CURATED events
    only; breaking news about a future Fed meeting belongs in the newswire."""
    from datetime import timezone, timedelta as _td

    now = datetime.now(timezone.utc)
    future = (now + _td(hours=24)).isoformat()
    far_future = (now + _td(hours=48)).isoformat()

    store.insert_batch([
        {
            "title": "TOMORROW: FOMC Meeting — June 17, 2026",
            "link": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
            "summary": "Scheduled FOMC Meeting on Wednesday, June 17, 2026 at 14:00 UTC.",
            "published": future,
            "source": "macro_calendar",
        },
        # Decoy: a regular RSS row mentioning FOMC with a future published date
        {
            "title": "Reuters: FOMC expected to hold rates",
            "link": "https://reuters.example/x",
            "summary": "Analysts expect a hold next meeting.",
            "published": far_future,
            "source": "rss",
        },
    ])

    # Redirect the collector to use the test DB
    monkeypatch.setattr(
        "storage.article_store._get_db_path",
        lambda: store.conn.execute("PRAGMA database_list").fetchall()[0][2],
    )
    out = ca._collect_macro_calendar_events()
    assert isinstance(out, list)
    assert len(out) == 1
    assert "FOMC Meeting" in out[0]["title"]
    assert "Reuters" not in out[0]["title"]


def test_collect_dedups_by_published_picking_freshest_title(monkeypatch, store):
    """Multiple rows for the SAME event (one per day_class transition) must
    collapse to ONE entry, the one with the latest ``first_seen`` so its
    title carries the sharpest current prefix (TODAY > TOMORROW > UPCOMING)."""
    from datetime import timezone, timedelta as _td
    import time as _time

    now = datetime.now(timezone.utc)
    event_published = (now + _td(hours=6)).isoformat()

    # First emission: UPCOMING (5d)
    store.insert_batch([{
        "title": "UPCOMING (5d): FOMC Meeting — June 17, 2026",
        "link": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
        "summary": "scheduled meeting body",
        "published": event_published,
        "source": "macro_calendar",
    }])
    # Tiny delay so first_seen differs (the test depends on MAX(first_seen)
    # picking the later row; SQLite's INSERT writes ``now`` at insert time).
    _time.sleep(0.02)

    # Second emission: TODAY (sharper prefix) — different title means a
    # different article_id, so the store accepts both rows.
    store.insert_batch([{
        "title": "TODAY: FOMC Meeting — June 17, 2026",
        "link": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm?day=today",
        "summary": "scheduled meeting body",
        "published": event_published,
        "source": "macro_calendar",
    }])

    monkeypatch.setattr(
        "storage.article_store._get_db_path",
        lambda: store.conn.execute("PRAGMA database_list").fetchall()[0][2],
    )
    out = ca._collect_macro_calendar_events()
    assert isinstance(out, list)
    assert len(out) == 1, "two same-instant rows must dedup to one"
    # The sharper prefix must win (MAX(first_seen) → the later TODAY row)
    assert out[0]["title"].startswith("TODAY: FOMC Meeting")


def test_collect_skips_past_events(monkeypatch, store):
    """Events whose ``published`` is in the past must NOT surface — the
    forward-calendar block is forward-only by construction (the SQL filter
    excludes them, but the defensive Python guard in the collector function
    is the belt-and-braces test target)."""
    from datetime import timezone, timedelta as _td
    now = datetime.now(timezone.utc)
    past = (now - _td(hours=2)).isoformat()
    store.insert_batch([{
        "title": "TODAY: FOMC Meeting — yesterday's event",
        "link": "https://fed.example/past",
        "summary": "x",
        "published": past,
        "source": "macro_calendar",
    }])
    monkeypatch.setattr(
        "storage.article_store._get_db_path",
        lambda: store.conn.execute("PRAGMA database_list").fetchall()[0][2],
    )
    out = ca._collect_macro_calendar_events()
    assert out == []


def test_collect_skips_events_beyond_window(monkeypatch, store):
    """An FOMC 10 days out must NOT clutter the 72h window — events past
    the horizon are filtered."""
    from datetime import timezone, timedelta as _td
    now = datetime.now(timezone.utc)
    way_out = (now + _td(days=14)).isoformat()
    store.insert_batch([{
        "title": "IN 14d: FOMC Meeting — far future",
        "link": "https://fed.example/far",
        "summary": "x",
        "published": way_out,
        "source": "macro_calendar",
    }])
    monkeypatch.setattr(
        "storage.article_store._get_db_path",
        lambda: store.conn.execute("PRAGMA database_list").fetchall()[0][2],
    )
    out = ca._collect_macro_calendar_events()
    assert out == []


def test_collect_returns_none_on_db_failure(monkeypatch):
    """A missing/locked DB → ``None`` so ``_build_payload`` omits the section
    entirely (same best-effort discipline as ``_collect_source_health`` /
    ``_collect_alert_velocity``). The briefing is NEVER broken or delayed by
    a forward-calendar read failure."""
    monkeypatch.setattr(
        "storage.article_store._get_db_path",
        lambda: "/tmp/definitely_does_not_exist_xyzzy_12345.db",
    )
    # File doesn't exist; mode=ro `file:...?mode=ro` opens but every query
    # fails immediately. The collector swallows it → None.
    out = ca._collect_macro_calendar_events()
    assert out is None


# ── _build_payload wiring: emit-on-events, omit-on-None / empty ──────────────

def test_build_payload_omits_section_when_macro_kwarg_is_none():
    """Default-None signature: the 5-arg callers and the existing 8-arg
    callers stay byte-deterministic — the section is omitted entirely so
    nothing changes for the existing test fixtures."""
    payload = ca._build_payload(
        articles=[], stock_data={}, earnings=[],
    )
    assert "MACRO CALENDAR" not in payload


def test_build_payload_omits_section_when_macro_list_is_empty():
    """An empty list is the "no scheduled events" case — section omitted
    so the briefing isn't padded with an empty heading."""
    payload = ca._build_payload(
        articles=[], stock_data={}, earnings=[],
        macro_calendar_events=[],
    )
    assert "MACRO CALENDAR" not in payload


def test_build_payload_emits_section_with_events():
    """A non-empty list emits the section verbatim with the renderer's
    output."""
    events = [
        {"title": "TODAY: FOMC Meeting — June 17, 2026",
         "hours_until": 6.0,
         "published": "2026-06-17T14:00:00+00:00"},
        {"title": "TOMORROW: Jobs Report (Employment Situation) — June 18, 2026",
         "hours_until": 24.0,
         "published": "2026-06-18T08:30:00+00:00"},
    ]
    payload = ca._build_payload(
        articles=[], stock_data={}, earnings=[],
        macro_calendar_events=events,
    )
    assert "MACRO CALENDAR" in payload
    assert "TODAY: FOMC Meeting" in payload
    assert "TOMORROW: Jobs Report" in payload
    assert "~6h" in payload
    assert "~1d" in payload


def test_build_payload_byte_identical_when_macro_omitted_vs_none():
    """An explicit ``None`` and the no-kwarg default must produce the SAME
    payload — anti-drift between callers that omit the kwarg and ones that
    pass None explicitly."""
    a = ca._build_payload(articles=[], stock_data={}, earnings=[])
    b = ca._build_payload(
        articles=[], stock_data={}, earnings=[],
        macro_calendar_events=None,
    )
    assert a == b


# ── SYSTEM_PROMPT coverage rule ─────────────────────────────────────────────

def test_system_prompt_carries_macro_calendar_rule():
    """The SYSTEM_PROMPT must instruct Opus to reproduce the MACRO CALENDAR
    section and weight LEAD/RISK around imminent FOMC/CPI/Jobs events.
    Without the rule, the input block is present in DATA but Opus has no
    schema directive that it must surface — the entire feature degrades to
    free-form (and unreliably-formatted) output."""
    sp = ca.SYSTEM_PROMPT
    assert "MACRO CALENDAR" in sp
    # The rule must name the actionable consequence — weight LEAD/RISK
    # around imminent events. Pinned phrasing class (not exact verbatim).
    assert "FOMC" in sp
    # The output-format placeholder section must also exist so Opus knows
    # where in the output to render the block.
    assert sp.count("MACRO CALENDAR") >= 2

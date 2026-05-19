"""Pin the macro_calendar_collector's day-class re-emission behaviour and
its core deterministic helpers.

This collector was previously zero-coverage. The bug it pinned was a real
defect: ``_seen_id`` keyed only on ``(date, type)``, so once an event was
emitted at ANY distance ("UPCOMING (5d)") the dedup table blocked all later
emissions — including the "TODAY" / "TOMORROW" rows the urgency scorer must
see for the prefix system to be anything more than dead code.

The four load-bearing invariants are untouched by this collector (it only
*writes* to ``articles.db`` via the standard ingest path which already
preserves backtest isolation / ml_score≠ai_score / score_source / urgency
state); these tests intentionally pin only the collector's own contract.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest import mock

import pytest

from collectors import macro_calendar_collector as mc


# ── _day_class: the load-bearing classification fold ─────────────────────────

def test_day_class_today_when_same_date():
    now = datetime(2026, 6, 17, 14, 0, tzinfo=timezone.utc)
    ev = datetime(2026, 6, 17, 22, 0, tzinfo=timezone.utc)
    assert mc._day_class(ev, now) == "today"


def test_day_class_tomorrow_when_one_day_out():
    now = datetime(2026, 6, 17, 23, 30, tzinfo=timezone.utc)
    ev = datetime(2026, 6, 18, 0, 30, tzinfo=timezone.utc)
    assert mc._day_class(ev, now) == "tomorrow"


def test_day_class_upcoming_2_to_7_days():
    now = datetime(2026, 6, 17, 12, 0, tzinfo=timezone.utc)
    for delta in (2, 3, 5, 7):
        ev = now + timedelta(days=delta)
        assert mc._day_class(ev, now) == "upcoming", f"delta={delta}"


def test_day_class_future_beyond_7_days():
    now = datetime(2026, 6, 17, 12, 0, tzinfo=timezone.utc)
    for delta in (8, 14, 30):
        ev = now + timedelta(days=delta)
        assert mc._day_class(ev, now) == "future", f"delta={delta}"


# ── _seen_id: stable across calls, differs across day_class ─────────────────

def test_seen_id_is_stable_for_same_inputs():
    a = mc._seen_id("FOMC Meeting", "2026-06-18", "today")
    b = mc._seen_id("FOMC Meeting", "2026-06-18", "today")
    assert a == b


def test_seen_id_differs_across_day_classes():
    """Core invariant of the bug fix — a (date,type) row emitted at one
    day-class must NOT collide with the same event at a different class."""
    upcoming = mc._seen_id("FOMC Meeting", "2026-06-18", "upcoming")
    tomorrow = mc._seen_id("FOMC Meeting", "2026-06-18", "tomorrow")
    today = mc._seen_id("FOMC Meeting", "2026-06-18", "today")
    future = mc._seen_id("FOMC Meeting", "2026-06-18", "future")
    assert len({upcoming, tomorrow, today, future}) == 4


def test_seen_id_differs_across_event_types():
    cpi = mc._seen_id("CPI Release", "2026-06-18", "today")
    fomc = mc._seen_id("FOMC Meeting", "2026-06-18", "today")
    assert cpi != fomc


# ── _day_prefix: the visible title prefix string ────────────────────────────

def test_day_prefix_today_tomorrow_upcoming_far():
    now = datetime(2026, 6, 17, 12, 0, tzinfo=timezone.utc)
    assert mc._day_prefix(now, now) == "TODAY"
    assert mc._day_prefix(now + timedelta(days=1), now) == "TOMORROW"
    assert mc._day_prefix(now + timedelta(days=4), now) == "UPCOMING (4d)"
    assert mc._day_prefix(now + timedelta(days=12), now) == "IN 12d"


# ── _parse_month: the BLS/FOMC month-name parser ────────────────────────────

def test_parse_month_short_and_long_forms():
    assert mc._parse_month("Jan") == 1
    assert mc._parse_month("january") == 1
    assert mc._parse_month("Feb.") == 2
    assert mc._parse_month("DEC") == 12
    assert mc._parse_month("September") == 9


def test_parse_month_returns_none_on_garbage():
    assert mc._parse_month("Quintember") is None
    assert mc._parse_month("") is None


# ── Network-failure resilience ──────────────────────────────────────────────

def test_fetch_fomc_returns_empty_on_request_exception(monkeypatch):
    """A network failure must never raise into the daemon thread — it must
    degrade to [] so the worker simply backs off and retries next cycle."""
    def _raise(*args, **kwargs):
        raise Exception("simulated network outage")
    monkeypatch.setattr(mc.requests, "get", _raise)
    now = datetime.now(timezone.utc)
    horizon = now + timedelta(days=30)
    assert mc._fetch_fomc_dates(now, horizon) == []


def test_fetch_bls_returns_empty_on_request_exception(monkeypatch):
    def _raise(*args, **kwargs):
        raise Exception("simulated 503")
    monkeypatch.setattr(mc.requests, "get", _raise)
    now = datetime.now(timezone.utc)
    horizon = now + timedelta(days=30)
    out = mc._fetch_bls_schedule(
        "CPI Release", "https://www.bls.gov/schedule/news_release/cpi.htm",
        now, horizon,
    )
    assert out == []


# ── End-to-end re-emission across day-class transitions ────────────────────

def _redirect_seen_db(monkeypatch, tmp_path):
    """Point the collector at a per-test seen-events DB so tests are isolated
    from the real ``data/macro_calendar_seen.db``."""
    monkeypatch.setattr(
        mc, "_MACRO_DB_PATH", tmp_path / "macro_calendar_seen.db"
    )


def _stub_fetchers(monkeypatch, fomc_dates):
    """Stub network: FOMC returns the given list; BLS returns []."""
    def _fake_fomc(now, horizon):
        return [
            {"type": "FOMC Meeting", "date": d, "url": "https://fed.example/x"}
            for d in fomc_dates
        ]

    def _fake_bls(*args, **kwargs):
        return []

    monkeypatch.setattr(mc, "_fetch_fomc_dates", _fake_fomc)
    monkeypatch.setattr(mc, "_fetch_bls_schedule", _fake_bls)


def test_event_re_emits_when_day_class_transitions(monkeypatch, tmp_path):
    """The whole point of the bug fix: an event emitted at one day-class
    MUST re-emit when its class changes (e.g. UPCOMING (5d) -> TOMORROW ->
    TODAY). Drives the real seen-events DB so a regression in the seen-id
    composition would fail this test."""
    _redirect_seen_db(monkeypatch, tmp_path)

    event_dt = datetime(2026, 6, 20, 14, 0, tzinfo=timezone.utc)

    # Pass 1: "now" is 5 days before the event → UPCOMING
    now_t1 = datetime(2026, 6, 15, 9, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(
        mc, "datetime",
        _FakeDatetime(now_t1, _real=mc.datetime),
    )
    _stub_fetchers(monkeypatch, [event_dt])
    first = mc.collect_macro_calendar()
    assert len(first) == 1, first
    assert first[0]["title"].startswith("UPCOMING (5d): FOMC Meeting"), first[0]["title"]

    # Pass 2: "now" advances to the same UPCOMING window day — must NOT
    # re-emit (same day_class), the collector's idempotence within a class.
    now_t1b = datetime(2026, 6, 15, 22, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(
        mc, "datetime",
        _FakeDatetime(now_t1b, _real=mc.datetime),
    )
    again = mc.collect_macro_calendar()
    assert again == [], "same-class re-poll must be deduped"

    # Pass 3: "now" advances to TOMORROW (1 day before) — class transition
    # MUST emit a new article so the urgency scorer sees the sharper prefix.
    now_t2 = datetime(2026, 6, 19, 9, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(
        mc, "datetime",
        _FakeDatetime(now_t2, _real=mc.datetime),
    )
    second = mc.collect_macro_calendar()
    assert len(second) == 1, second
    assert second[0]["title"].startswith("TOMORROW: FOMC Meeting"), second[0]["title"]

    # Pass 4: TODAY — another class transition, must emit again. This is the
    # behaviour the old (date,type)-only seen_id silently blocked.
    now_t3 = datetime(2026, 6, 20, 9, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(
        mc, "datetime",
        _FakeDatetime(now_t3, _real=mc.datetime),
    )
    third = mc.collect_macro_calendar()
    assert len(third) == 1, third
    assert third[0]["title"].startswith("TODAY: FOMC Meeting"), third[0]["title"]


def test_same_day_class_does_not_re_emit(monkeypatch, tmp_path):
    """Polling many times within one day-class window must dedup to ONE
    emission — the seen-events DB still has to suppress the obvious noise."""
    _redirect_seen_db(monkeypatch, tmp_path)
    event_dt = datetime(2026, 6, 20, 14, 0, tzinfo=timezone.utc)
    now_t = datetime(2026, 6, 18, 9, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(
        mc, "datetime", _FakeDatetime(now_t, _real=mc.datetime)
    )
    _stub_fetchers(monkeypatch, [event_dt])

    first = mc.collect_macro_calendar()
    assert len(first) == 1

    # Re-poll several times within the SAME class — must be empty
    for _ in range(5):
        assert mc.collect_macro_calendar() == []


def test_article_dict_shape_is_pipeline_compatible(monkeypatch, tmp_path):
    """A returned article must carry exactly the keys ArticleStore.insert_batch
    consumes (title/link/summary/source/published) — a missing key silently
    drops the row. Pin the shape so a future field rename can't break ingest."""
    _redirect_seen_db(monkeypatch, tmp_path)
    event_dt = datetime(2026, 7, 1, 18, 0, tzinfo=timezone.utc)
    now_t = datetime(2026, 7, 1, 8, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(
        mc, "datetime", _FakeDatetime(now_t, _real=mc.datetime)
    )
    _stub_fetchers(monkeypatch, [event_dt])

    out = mc.collect_macro_calendar()
    assert len(out) == 1
    art = out[0]
    for k in ("title", "link", "summary", "published", "source"):
        assert k in art, f"missing key {k!r} in collector output"
    assert art["source"] == "macro_calendar"
    # `published` must be an ISO-8601 string the urgency scorer can parse.
    parsed = datetime.fromisoformat(art["published"])
    assert parsed.tzinfo is not None
    assert parsed == event_dt


# ── helpers ──────────────────────────────────────────────────────────────────

class _FakeDatetime(datetime):
    """A datetime subclass whose ``.now`` returns a fixed instant. Used to
    drive the collector's wall-clock without `freezegun`. Everything else
    delegates to the real datetime."""

    _fixed: datetime | None = None

    def __new__(cls, fixed_now=None, _real=None):
        # Allow `_FakeDatetime(fixed_now, _real=mc.datetime)` instantiation
        # to configure a "fake datetime module" — actual datetime instances
        # are created via the parent's machinery.
        if fixed_now is not None:
            cls._fixed = fixed_now
            return cls
        return super().__new__(cls)

    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        ts = cls._fixed
        assert ts is not None, "_FakeDatetime not configured"
        return ts.astimezone(tz) if tz else ts.replace(tzinfo=None)

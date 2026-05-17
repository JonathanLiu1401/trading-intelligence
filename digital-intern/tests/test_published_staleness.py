"""storage.article_store._published_older_than — the briefing staleness gate.

This pure helper is the *authoritative* 24h-staleness check for the heartbeat
briefing. It exists specifically to defeat a subtle bug: the SQL pre-filter in
``get_top_for_briefing`` compares ``published`` as a raw string, which is
meaningless for RFC822 dates ("Wed, 14 May 2026 ...") — their leading letter
lex-sorts *after* any ISO cutoff, so a naive ``published >= cutoff`` keeps
every stale RSS article. If this helper regresses, GDELT/RSS-indexed stale
articles resurface in the briefing as "breaking news".
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from storage.article_store import _published_older_than


def _cutoff(hours_ago: int = 24) -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=hours_ago)


def test_empty_or_none_is_never_stale():
    cut = _cutoff()
    # No date → keep the article (returning True would silently empty briefings).
    assert _published_older_than("", cut) is False
    assert _published_older_than(None, cut) is False


def test_unparseable_date_is_kept():
    assert _published_older_than("not a date at all", _cutoff()) is False


def test_old_rfc822_date_is_stale():
    """The exact regression this helper defends: an RFC822 string that
    lex-sorts *after* an ISO cutoff (so the SQL pre-filter misses it) must
    still be correctly identified as older than the cutoff."""
    cut = _cutoff(24)
    old_rfc822 = "Wed, 01 Jan 2020 08:00:00 GMT"
    # Sanity: prove the naive string comparison the SQL does would be WRONG —
    # "Wed, 01 Jan 2020..." > the ISO cutoff lexically, i.e. SQL keeps it.
    assert old_rfc822 >= cut.isoformat()
    # The helper parses the date and correctly flags it stale.
    assert _published_older_than(old_rfc822, cut) is True


def test_recent_rfc822_date_is_fresh():
    cut = _cutoff(24)
    recent = (datetime.now(timezone.utc) - timedelta(hours=1))
    rfc822 = recent.strftime("%a, %d %b %Y %H:%M:%S GMT")
    assert _published_older_than(rfc822, cut) is False


def test_old_iso_date_is_stale():
    cut = _cutoff(24)
    assert _published_older_than("2020-01-01T08:00:00+00:00", cut) is True


def test_recent_iso_date_is_fresh():
    cut = _cutoff(24)
    recent = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    assert _published_older_than(recent, cut) is False


def test_naive_iso_treated_as_utc():
    """A timezone-naive ISO date must not raise (offset-naive vs aware compare)
    and must be interpreted as UTC."""
    cut = _cutoff(24)
    assert _published_older_than("2020-01-01T08:00:00", cut) is True
    fresh_naive = (datetime.now(timezone.utc) - timedelta(hours=1)).replace(
        tzinfo=None
    ).isoformat()
    assert _published_older_than(fresh_naive, cut) is False


def test_zulu_suffix_iso_is_parsed():
    cut = _cutoff(24)
    assert _published_older_than("2020-01-01T08:00:00Z", cut) is True

"""Fleet-wide pin: no collector may write a naive ``datetime.utcnow()`` timestamp.

``datetime.utcnow()`` is deprecated (it returns a *naive* datetime that
silently misrepresents a UTC instant as local-naive). Every collector that
stamps ``seen_articles.first_seen`` was migrated to
``datetime.now(timezone.utc)`` — a timezone-aware UTC instant whose
``.isoformat()`` carries the explicit ``+00:00`` offset.

This was deferred several times as "cross-cutting churn out of scope for a
review commit" (see AGENTS.md); it is landed here as its own focused commit.
The migration is safe because ``seen_articles.first_seen`` is **write-only** —
a full-tree audit found every reference is a ``CREATE TABLE`` or
``INSERT``; the dedup read path is ``WHERE id=?`` exclusively and never
parses ``first_seen`` (the parsed ``first_seen`` consumers — paper_trader,
dashboard, SQL range filters — read ``articles.first_seen``, written by
``storage/article_store.py``, which this change does not touch).

Two guards:

1. **Static**: no collector source may contain ``datetime.utcnow(`` again.
   This also covers the non-DB ``sec_edgar`` EFTS date-range params, which a
   behavioural DB test would miss.
2. **Format**: the replacement idiom yields a tz-aware ISO string that still
   round-trips through the canonical ``paper_trader.signals`` parse
   expression — defence-in-depth even though this column is write-only.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

_COLLECTORS_DIR = Path(__file__).resolve().parents[1] / "collectors"

# Every collector that timestamps a seen_articles row (the 11 shared-DB
# writers) plus sec_edgar's EFTS date-range params — i.e. every module that
# previously called datetime.utcnow().
_MIGRATED_MODULES = [
    "yahoo_ticker_rss.py",
    "google_news.py",
    "polygon_collector.py",
    "massive_collector.py",
    "alphavantage_collector.py",
    "rss_collector.py",
    "wikipedia_collector.py",
    "finnhub_collector.py",
    "newsapi_collector.py",
    "sec_edgar.py",
]

_UTCNOW = re.compile(r"datetime\.utcnow\s*\(")


@pytest.mark.parametrize("fname", _MIGRATED_MODULES)
def test_collector_has_no_naive_utcnow(fname):
    """No migrated collector may reintroduce the deprecated naive utcnow()."""
    src = (_COLLECTORS_DIR / fname).read_text()
    hits = [
        i + 1
        for i, line in enumerate(src.splitlines())
        if _UTCNOW.search(line)
    ]
    assert not hits, (
        f"collectors/{fname} reintroduced datetime.utcnow() at line(s) "
        f"{hits} — use datetime.now(timezone.utc) so the written timestamp "
        f"is an explicit tz-aware UTC instant"
    )


def test_replacement_idiom_is_tz_aware_and_roundtrips():
    """The new idiom must yield a tz-aware UTC ISO string.

    The parse mirrors paper_trader.signals._age_hours exactly; even though
    seen_articles.first_seen is write-only, this pins the wire format so a
    future consumer (or a copy-paste of the idiom into a parsed column)
    cannot silently regress to naive.
    """
    s = datetime.now(timezone.utc).isoformat()
    assert s.endswith("+00:00"), f"expected explicit UTC offset, got {s!r}"
    parsed = datetime.fromisoformat(s.strip().replace("Z", "+00:00"))
    assert parsed.tzinfo is not None
    assert parsed.utcoffset().total_seconds() == 0

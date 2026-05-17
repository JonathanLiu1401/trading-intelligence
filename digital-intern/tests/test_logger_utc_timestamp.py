"""daemon.log timestamps must be real UTC, not local time mislabeled 'Z'.

Regression guard for a host-TZ-dependent skew: the plain ``logs/daemon.log``
``RotatingFileHandler`` used ``logging.Formatter(datefmt="%Y-%m-%dT%H:%M:%SZ")``
but left ``Formatter.converter`` at the Python default ``time.localtime``. The
literal ``Z`` then *asserted* UTC while ``%(asctime)s`` rendered local time —
on a PDT host (UTC-7) every line read 7h behind real UTC, e.g. a briefing
logged ``06:26:38Z`` whose ``briefings.ts`` row (``datetime.now(timezone.utc)``)
said ``13:26:38``. ``healthcheck.sh`` greps this file and operators correlate
it against the UTC-correct ``structured.jsonl`` / ``briefings`` table / Discord
alerts; a silent constant offset breaks every cross-sink time correlation while
each line still looks individually plausible.

The console (``_ColourFormatter``) and ``structured.jsonl`` (``_JSONLHandler``)
sinks already use ``datetime.now(timezone.utc)`` and are unaffected; this pins
the one remaining sink.
"""
from __future__ import annotations

import logging
import time

from core.logger import _plain_file_formatter


def _record(created: float) -> logging.LogRecord:
    rec = logging.LogRecord(
        name="daemon", level=logging.INFO, pathname="f.py", lineno=1,
        msg="hello", args=(), exc_info=None,
    )
    rec.created = created  # pin the instant so the test is host-clock-independent
    return rec


def test_asctime_is_gmtime_not_localtime():
    """The ``Z``-suffixed asctime must be UTC regardless of host timezone.

    Pinned at a fixed epoch so it fails deterministically on a non-UTC host
    *and* — via the converter identity — even on a UTC CI box where
    localtime == gmtime would otherwise mask the bug behaviorally.
    """
    fmt = _plain_file_formatter()
    # Identity check: the Python default is time.localtime; the fix must
    # explicitly switch to time.gmtime so the literal 'Z' is not a lie.
    assert fmt.converter is time.gmtime

    epoch = 1779025439.0  # 2026-05-17T13:43:59Z
    line = fmt.format(_record(epoch))
    ts = line.split(" [", 1)[0]
    assert ts == time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))
    assert ts == "2026-05-17T13:43:59Z"


def test_format_string_and_level_preserved():
    """The fix must change only the time zone, not the line shape."""
    fmt = _plain_file_formatter()
    line = fmt.format(_record(1779025439.0))
    assert line == "2026-05-17T13:43:59Z [INFO] daemon: hello"

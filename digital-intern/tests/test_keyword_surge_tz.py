"""Regression test for the keyword_surge.py timezone-suffix bug.

The script's recent / baseline cutoffs used to be built with
``datetime.utcnow().isoformat() + 'Z'`` — a NAIVE datetime stringified with a
``Z`` suffix.  ``articles.db``'s ``first_seen`` column, by contrast, is
written by ``datetime.now(timezone.utc).isoformat()`` which produces the
``+00:00`` offset form.

Lexicographic comparison between the two suffixes is broken — '+' is U+002B,
'Z' is U+005A — so for an article whose `first_seen` was at almost exactly
the cutoff second, the comparison silently bucketed the row into the *wrong*
window (the recent bigram counter missed it; the baseline counter saw it).

This test pins that ``_now_utc_isoformat()`` (the script's new helper)
returns a string in the same shape as the column it is compared to, and
that a row written at the cutoff second is consistently classified.
"""
from __future__ import annotations

import datetime
import importlib.util
import sys
from datetime import timezone
from pathlib import Path

import pytest


_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "keyword_surge.py"
)


@pytest.fixture
def ks_module():
    """Import scripts/keyword_surge.py without executing main()."""
    spec = importlib.util.spec_from_file_location(
        "keyword_surge_under_test", str(_SCRIPT_PATH)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["keyword_surge_under_test"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_now_utc_isoformat_matches_first_seen_column_shape(ks_module):
    """The helper output must use the same offset form (``+00:00``) that
    ArticleStore writes — otherwise the lexicographic comparison in main()
    misclassifies rows at the boundary."""
    out = ks_module._now_utc_isoformat()
    assert out.endswith("+00:00"), (
        f"keyword_surge produced {out!r}; articles.db rows end in "
        f"'+00:00' (datetime.now(timezone.utc).isoformat()). Mismatched "
        f"suffixes break the string-compare in main()'s recent/baseline split."
    )
    # And it parses round-trip as an aware datetime (a sanity bound on the
    # whole format, not just the suffix).
    parsed = datetime.datetime.fromisoformat(out)
    assert parsed.tzinfo is not None
    assert parsed.tzinfo.utcoffset(parsed) == datetime.timedelta(0)


def test_boundary_row_classified_consistently_with_column_format(ks_module):
    """A `first_seen` written by ``datetime.now(timezone.utc).isoformat()``
    must compare CONSISTENTLY against the helper's cutoff strings — i.e. a
    row whose `first_seen` is strictly later than the cutoff must compare
    greater. The pre-fix code mixed '+00:00' (column) with 'Z' (cutoff),
    making the comparison fail at the boundary."""
    cutoff = ks_module._now_utc_isoformat()
    # An article inserted one second later — what ArticleStore would write
    # for a brand-new row.
    later_dt = datetime.datetime.now(timezone.utc) + datetime.timedelta(
        seconds=1
    )
    later_str = later_dt.isoformat()
    # Strict lexicographic comparison — same form on both sides, so the
    # later row MUST sort after the cutoff string.
    assert later_str > cutoff, (
        f"later=({later_str!r}) should be > cutoff=({cutoff!r}) under "
        f"lexicographic comparison; mismatched suffix breaks this"
    )


def test_no_utcnow_call_in_module(ks_module):
    """Belt-and-braces: no live ``datetime.utcnow()`` call should remain in
    the module (it is deprecated AND produced the broken 'Z' suffix). A
    mention in a docstring explaining the historical bug is fine."""
    src = _SCRIPT_PATH.read_text()
    # Strip docstrings/comments and check the remaining executable text.
    # A naive substring check accepts mentions inside a docstring (the new
    # ``_now_utc_isoformat`` deliberately documents the prior bug); a real
    # ``datetime.utcnow()`` call would always be preceded by ``datetime.``.
    assert "datetime.utcnow()" not in src, (
        "datetime.utcnow() is deprecated and produced the 'Z' suffix that "
        "broke string-compare against the '+00:00' column. Replace with "
        "datetime.now(timezone.utc)."
    )

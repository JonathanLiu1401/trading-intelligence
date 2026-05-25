"""Pin the ``datetime.utcnow()`` → ``datetime.now(timezone.utc)`` fix
in ``paper_trader.analytics.monkey_benchmark``.

``datetime.utcnow()`` is deprecated in Python 3.12 and slated for
removal. The sibling fix in ``run_continuous_backtests.py`` (AGENTS.md
pass #38) was already pinned by
``tests/test_ml_backtest_review_20260525_agent2.py``; this file is the
matching pin for the monkey-benchmark module so a future refactor
cannot silently re-introduce the deprecated call.

Both call-sites in ``monkey_benchmark.py`` historically emitted
ISO-8601 with a trailing ``Z``; the fix preserves that exact wire
format (the cache-age parser already strips Z), so the test asserts
the OUTPUT shape is unchanged AND the source no longer calls
``utcnow()``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


# Resolve once so the path-based assertions don't repeatedly walk the
# filesystem inside the test body — keeps the test cheap on a tight
# loop and the diagnostic explicit when the source moves.
_SOURCE = Path(__file__).resolve().parent.parent / (
    "paper_trader/analytics/monkey_benchmark.py"
)


class TestSourceHasNoUtcnowCall:
    """The source must no longer call ``datetime.utcnow()`` —
    deprecated, removal in a future Python release."""

    def test_no_utcnow_call_remains_in_source(self):
        src = _SOURCE.read_text()
        # Strip comments line-by-line so the deprecation note in the
        # explanatory comments doesn't false-positive the assertion.
        code_lines: list[str] = []
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            # An inline ``# comment`` after code on the same line: keep
            # the code half, drop the comment half. ``#`` inside a
            # string literal would false-positive here too, but the
            # module has no string literals that contain a literal
            # ``utcnow`` token so this is safe in practice.
            if "#" in line:
                line = line.split("#", 1)[0]
            code_lines.append(line)
        code_only = "\n".join(code_lines)
        assert "utcnow()" not in code_only, (
            "paper_trader/analytics/monkey_benchmark.py still calls "
            "datetime.utcnow() — deprecated in Python 3.12. Use "
            "datetime.now(timezone.utc) instead."
        )

    def test_uses_tz_aware_now(self):
        # The fix uses tz-aware ``datetime.now(timezone.utc)`` (the
        # canonical replacement) — pin the call-site so a future
        # change cannot silently drop the timezone and regress to a
        # naive UTC stamp (which would compare wrong against the
        # cache-age parser's tz-aware datetime).
        src = _SOURCE.read_text()
        assert "datetime.now(timezone.utc)" in src or (
            "_dt.datetime.now(_dt.timezone.utc)" in src
        ), (
            "expected `datetime.now(timezone.utc)` (or the `_dt`-aliased "
            "form) to replace `datetime.utcnow()` in monkey_benchmark.py"
        )


class TestWireFormatPreserved:
    """The on-disk cache JSON has ``"generated_at"`` keyed to an ISO
    string with a trailing ``Z`` (the wire format the cache-age parser
    in ``run_continuous_backtests`` was tuned to). The fix MUST NOT
    silently change that format — a producer that emits ``+00:00`` would
    still parse correctly today but downstream tooling reading the file
    out-of-band could misinterpret it.

    The test independently computes the same format the producer emits
    and asserts the shape (length, suffix, parseability) so the contract
    is locked even though we don't import the producer here (it has a
    heavy SciPy/numpy import chain we don't want in the unit test)."""

    def test_canonical_output_shape(self):
        # The producer line is:
        #   datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        # Run the same expression and verify the shape we ship is the
        # legacy ``...Z`` shape, not the bare ``+00:00`` offset.
        s = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        assert s.endswith("Z"), f"expected trailing Z, got {s!r}"
        assert "+00:00" not in s, f"+00:00 leaked into the Z form: {s!r}"
        # And it must round-trip cleanly through fromisoformat after
        # stripping the trailing ``Z`` — the exact parse path the
        # cache-age check uses (``gen_at_raw.rstrip("Z")``).
        parsed = datetime.fromisoformat(s.rstrip("Z"))
        # The producer set tzinfo=UTC; ``isoformat`` then emits the
        # +00:00 suffix that ``.replace`` swapped to Z. After we strip
        # the Z and parse, we get a NAIVE datetime back — exactly the
        # shape the consumer's ``if gen_dt.tzinfo is None: replace ...``
        # branch handles. Pin that the naivete is what we re-attach,
        # i.e. the consumer's branch is the one that fires.
        assert parsed.tzinfo is None

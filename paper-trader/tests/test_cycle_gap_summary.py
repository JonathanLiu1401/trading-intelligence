"""Tests for paper_trader.analytics.cycle_gap_summary.

Pinned to the verdict ladder documented in the module docstring.

These tests assert SPECIFIC EXPECTED VALUES — never "no exception" —
because the whole point of the builder is to give an operator a single
trustworthy number/verdict. A test that passes silently on the wrong
output would defeat the purpose.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from paper_trader.analytics.cycle_gap_summary import (
    JITTERY_CV,
    MIN_GAPS,
    STALL_GAP_S,
    STALL_P95_S,
    _median,
    _percentile,
    _stdev,
    build_cycle_gap_summary,
)


# ──────────────────────────────────────────────────────────────────────
# Statistical helper unit tests — exact arithmetic.
# ──────────────────────────────────────────────────────────────────────
def test_median_odd_count_returns_middle_element():
    assert _median([1.0, 2.0, 3.0]) == 2.0


def test_median_even_count_returns_mean_of_middle_pair():
    assert _median([1.0, 2.0, 3.0, 4.0]) == 2.5


def test_median_unsorted_input_is_sorted_internally():
    assert _median([5.0, 1.0, 3.0]) == 3.0


def test_percentile_empty_returns_zero():
    assert _percentile([], 0.95) == 0.0


def test_percentile_single_element_returns_that_element():
    assert _percentile([42.0], 0.95) == 42.0


def test_percentile_95_on_uniform_run_interpolates():
    # On [1..100], the 0.95 percentile lies at index 94.05 → ~95.05.
    out = _percentile(list(range(1, 101)), 0.95)
    assert 94.5 < out < 95.5


def test_stdev_single_sample_is_zero():
    assert _stdev([5.0], mean=5.0) == 0.0


def test_stdev_known_values():
    # Population stdev of [2, 4, 6] (mean 4): sqrt(((4+0+4)/3)) = sqrt(8/3)
    out = _stdev([2.0, 4.0, 6.0], mean=4.0)
    assert abs(out - (8 / 3) ** 0.5) < 1e-9


# ──────────────────────────────────────────────────────────────────────
# build_cycle_gap_summary — verdict & envelope tests.
# ──────────────────────────────────────────────────────────────────────
def _row(ts_iso: str) -> dict:
    """One ``recent_decisions``-shape row — the builder reads only
    ``timestamp``, every other column is ignored."""
    return {"timestamp": ts_iso}


def _seq_rows(start_iso: str, gaps_s: list[int]) -> list[dict]:
    """Build newest-first rows from a chronological gap series.

    ``start_iso`` is the OLDEST row; ``gaps_s`` are the gaps between
    consecutive rows (older→newer). Returns the rows newest-first to
    match ``store.recent_decisions`` orientation.
    """
    base = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
    if base.tzinfo is None:
        base = base.replace(tzinfo=timezone.utc)
    cursor = base
    asc: list[dict] = [_row(cursor.isoformat(timespec="seconds"))]
    for g in gaps_s:
        cursor = cursor + timedelta(seconds=g)
        asc.append(_row(cursor.isoformat(timespec="seconds")))
    # newest-first
    return list(reversed(asc))


def test_no_data_on_empty_input():
    out = build_cycle_gap_summary([])
    assert out["verdict"] == "NO_DATA"
    assert out["state"] == "NO_DATA"
    assert out["n_decisions"] == 0
    assert out["n_gaps"] == 0
    assert out["median_gap_s"] is None
    assert out["max_gap_s"] is None
    assert out["worst_gap"] is None
    # Threshold echo is always populated so the consumer can render the
    # ladder without re-importing the constants.
    assert out["stall_gap_threshold_s"] == STALL_GAP_S
    assert out["jittery_cv_threshold"] == JITTERY_CV
    assert out["min_gaps"] == MIN_GAPS


def test_no_data_on_none_input():
    out = build_cycle_gap_summary(None)
    assert out["verdict"] == "NO_DATA"


def test_no_data_on_single_decision():
    out = build_cycle_gap_summary([_row("2026-05-24T12:00:00+00:00")])
    assert out["verdict"] == "NO_DATA"
    assert out["n_decisions"] == 1
    assert out["n_gaps"] == 0


def test_unparseable_timestamps_silently_dropped():
    rows = [
        _row("garbage"),
        _row(""),
        _row(None),
        _row("2026-05-24T12:00:00+00:00"),
    ]
    out = build_cycle_gap_summary(rows)
    # Only 1 row had a parseable ts → NO_DATA.
    assert out["verdict"] == "NO_DATA"
    assert out["n_decisions"] == 1


def test_insufficient_when_gaps_below_floor():
    # MIN_GAPS=8 → need ≥ MIN_GAPS+1=9 rows. 5 rows = 4 gaps → INSUFFICIENT.
    rows = _seq_rows("2026-05-24T12:00:00+00:00", [600, 600, 600, 600])
    out = build_cycle_gap_summary(rows)
    assert out["verdict"] == "INSUFFICIENT"
    assert out["state"] == "INSUFFICIENT"
    assert out["n_decisions"] == 5
    assert out["n_gaps"] == 4


def test_insufficient_boundary_exact():
    # MIN_GAPS - 1 = 7 gaps still INSUFFICIENT; MIN_GAPS=8 should be enough.
    rows_below = _seq_rows(
        "2026-05-24T12:00:00+00:00", [600] * (MIN_GAPS - 1),
    )
    out_below = build_cycle_gap_summary(rows_below)
    assert out_below["verdict"] == "INSUFFICIENT"

    rows_at = _seq_rows(
        "2026-05-24T12:00:00+00:00", [600] * MIN_GAPS,
    )
    out_at = build_cycle_gap_summary(rows_at)
    assert out_at["verdict"] != "INSUFFICIENT"
    assert out_at["n_gaps"] == MIN_GAPS


def test_smooth_when_uniform_cadence():
    # 10 cycles, all 600s apart → median=mean=600, stdev=0 → SMOOTH.
    rows = _seq_rows("2026-05-24T12:00:00+00:00", [600] * 10)
    out = build_cycle_gap_summary(rows)
    assert out["verdict"] == "SMOOTH"
    assert out["state"] == "OK"
    assert out["median_gap_s"] == 600.0
    assert out["mean_gap_s"] == 600.0
    assert out["max_gap_s"] == 600.0
    assert out["min_gap_s"] == 600.0
    assert out["stdev_gap_s"] == 0.0
    assert out["coefficient_of_variation"] == 0.0
    assert out["n_stalled_gaps"] == 0
    assert out["pct_stalled"] == 0.0
    # Worst gap is still surfaced (any of the equal gaps wins).
    assert out["worst_gap"] is not None
    assert out["worst_gap"]["gap_s"] == 600.0


def test_stalled_by_max_when_a_single_gap_exceeds_threshold():
    # 9 cycles cleanly, then one giant gap above STALL_GAP_S — STALLED.
    big = STALL_GAP_S + 60
    rows = _seq_rows(
        "2026-05-24T12:00:00+00:00",
        [600, 600, 600, 600, 600, 600, 600, 600, big],
    )
    out = build_cycle_gap_summary(rows)
    assert out["verdict"] == "STALLED"
    assert out["state"] == "OK"
    assert out["max_gap_s"] == float(big)
    assert out["n_stalled_gaps"] == 1
    # pct_stalled = 1/9 ≈ 11.11
    assert abs(out["pct_stalled"] - (1 / 9 * 100)) < 0.01
    # Worst-gap row carries the actual giant value.
    assert out["worst_gap"]["gap_s"] == float(big)


def test_stalled_by_p95_above_longest_healthy_tier():
    # p95 above STALL_P95_S (sustained tail above the longest healthy
    # tier), even though no single gap exceeds STALL_GAP_S. Build a
    # tail-heavy distribution: many short gaps + a few gaps just above
    # STALL_P95_S to push p95 above the threshold.
    short = [60] * 18
    # Two gaps comfortably above STALL_P95_S, but below STALL_GAP_S — so
    # the verdict can ONLY trip via the p95 rule, not the max rule.
    high = STALL_P95_S + 60
    tail = [high] * 2
    rows = _seq_rows("2026-05-24T12:00:00+00:00", short + tail)
    out = build_cycle_gap_summary(rows)
    assert out["max_gap_s"] < STALL_GAP_S
    assert out["p95_gap_s"] >= STALL_P95_S
    assert out["verdict"] == "STALLED"
    assert "p95" in out["headline"]


def test_jittery_when_high_cov_but_no_stall():
    # Build a distribution that triggers CoV > 1.0 without tripping the
    # stall rules: median small, no single gap above STALL_GAP_S, p95
    # under STALL_P95_S. Mix many short gaps with a few medium gaps so
    # the CoV is high but the tail stays below the longest-healthy-tier
    # threshold.
    gaps = [60] * 8 + [3000, 3000]
    # median of 10 is 60 (sorted middle is 60/60)
    # max = 3000 < STALL_GAP_S (10800)
    # p95 = 3000 < STALL_P95_S (7200)
    # stdev should be huge relative to median → CoV >> 1
    rows = _seq_rows("2026-05-24T12:00:00+00:00", gaps)
    out = build_cycle_gap_summary(rows)
    assert out["median_gap_s"] == 60.0
    assert out["max_gap_s"] < STALL_GAP_S
    assert out["p95_gap_s"] < STALL_P95_S
    cov = out["coefficient_of_variation"]
    assert cov is not None and cov > JITTERY_CV
    assert out["verdict"] == "JITTERY"


def test_smooth_takes_precedence_over_jittery_when_cov_below_threshold():
    # 10 gaps clustered tightly around the median → CoV < 1.0 → SMOOTH.
    gaps = [580, 600, 620, 590, 610, 605, 595, 615, 585, 600]
    rows = _seq_rows("2026-05-24T12:00:00+00:00", gaps)
    out = build_cycle_gap_summary(rows)
    assert out["verdict"] == "SMOOTH"
    assert out["coefficient_of_variation"] is not None
    assert out["coefficient_of_variation"] < JITTERY_CV


def test_stalled_takes_precedence_over_jittery():
    # High CoV AND a single gap above STALL_GAP_S → STALLED, not JITTERY.
    # Build short gaps + one giant gap → CoV will be huge but the verdict
    # should still be STALLED.
    big = STALL_GAP_S + 100
    rows = _seq_rows("2026-05-24T12:00:00+00:00", [60] * 8 + [big])
    out = build_cycle_gap_summary(rows)
    assert out["max_gap_s"] >= STALL_GAP_S
    assert out["verdict"] == "STALLED"
    # The CoV is still computed and surfaced (so the consumer sees both
    # signals), but the headline must say STALLED.
    assert "STALLED" in out["headline"]


def test_worst_gap_is_actually_the_largest():
    rows = _seq_rows(
        "2026-05-24T12:00:00+00:00",
        [300, 600, 1200, 90, 4500, 200, 1000, 90, 1500],
    )
    out = build_cycle_gap_summary(rows)
    assert out["worst_gap"]["gap_s"] == 4500.0
    assert out["max_gap_s"] == 4500.0


def test_negative_gap_clamped_to_zero():
    # Construct rows that, when paired by the builder, have a clock-step-
    # back between two consecutive timestamps. parsed is newest-first, so
    # if a "newer" row has an OLDER timestamp than the "older" row, the
    # raw gap is negative and must clamp to 0. Easiest way: hand-build the
    # row list directly (skipping the _seq_rows helper).
    rows = [
        _row("2026-05-24T12:00:00+00:00"),     # NEWER in list order
        _row("2026-05-24T12:30:00+00:00"),     # OLDER (clock stepped back)
        _row("2026-05-24T11:00:00+00:00"),
        _row("2026-05-24T10:30:00+00:00"),
        _row("2026-05-24T10:00:00+00:00"),
        _row("2026-05-24T09:30:00+00:00"),
        _row("2026-05-24T09:00:00+00:00"),
        _row("2026-05-24T08:30:00+00:00"),
        _row("2026-05-24T08:00:00+00:00"),
    ]
    out = build_cycle_gap_summary(rows)
    # 8 gaps total. The clock-step-back pair (12:00 newer-in-list, 12:30
    # older-in-list) would have a raw gap of -1800s; the clamp pins it to 0.
    # The other seven gaps are positive (the surrounding rows are correctly
    # ordered older→newer). So the MIN of the gap series is 0.0, and no
    # negative value ever leaks through into max / median / mean.
    assert out["min_gap_s"] == 0.0
    # Mean / median should be positive (the clamp turned a negative into a
    # zero, never into a negative number).
    assert out["mean_gap_s"] > 0
    assert out["median_gap_s"] > 0


def test_threshold_echo_is_always_populated():
    # Even on a NO_DATA envelope, the threshold echo must be present so a
    # consumer rendering a "ladder reference" panel never crashes on None.
    out = build_cycle_gap_summary([])
    assert out["stall_gap_threshold_s"] == STALL_GAP_S
    assert out["stall_p95_threshold_s"] == STALL_P95_S
    assert out["jittery_cv_threshold"] == JITTERY_CV
    assert out["min_gaps"] == MIN_GAPS


def test_headline_format_for_each_verdict():
    # Each verdict should produce a headline starting with its own name
    # so a Discord / dashboard caller can grep / colorise by the leading
    # token. NO_DATA, INSUFFICIENT, SMOOTH, JITTERY, STALLED.
    no_data = build_cycle_gap_summary([])
    assert no_data["headline"].startswith("NO_DATA")

    short = _seq_rows("2026-05-24T12:00:00+00:00", [600, 600, 600])
    insufficient = build_cycle_gap_summary(short)
    assert insufficient["headline"].startswith("INSUFFICIENT")

    smooth = _seq_rows("2026-05-24T12:00:00+00:00", [600] * 10)
    smooth_out = build_cycle_gap_summary(smooth)
    assert smooth_out["headline"].startswith("SMOOTH")

    big = STALL_GAP_S + 60
    stalled_rows = _seq_rows(
        "2026-05-24T12:00:00+00:00",
        [600] * 8 + [big],
    )
    stalled = build_cycle_gap_summary(stalled_rows)
    assert stalled["headline"].startswith("STALLED")


def test_now_is_injectable_for_deterministic_as_of():
    fixed = datetime(2026, 5, 24, 12, 0, 0, tzinfo=timezone.utc)
    out = build_cycle_gap_summary([], now=fixed)
    assert out["as_of"] == "2026-05-24T12:00:00+00:00"


def test_naive_now_is_treated_as_utc():
    naive = datetime(2026, 5, 24, 12, 0, 0)
    out = build_cycle_gap_summary([], now=naive)
    # The naive value should be reinterpreted as UTC and serialise with
    # +00:00. Never a NaiveDatetimeError or stripped-tz string.
    assert out["as_of"].endswith("+00:00")


def test_garbage_row_does_not_break_builder():
    # Mix of dicts with no ``timestamp``, with garbage ``timestamp``, and
    # with valid ones. The builder must keep going and return a valid
    # envelope. The non-timestamp rows count as 0 usable decisions.
    rows = [
        {"some_other_field": "x"},
        {"timestamp": 12345},  # wrong type
        {"timestamp": "not-an-iso-string"},
    ]
    out = build_cycle_gap_summary(rows)
    assert out["verdict"] == "NO_DATA"
    assert out["n_decisions"] == 0


def test_pct_stalled_arithmetic():
    # 9 gaps, exactly 3 above STALL_GAP_S — pct_stalled should equal 33.33.
    big = STALL_GAP_S + 100
    small = 600
    rows = _seq_rows(
        "2026-05-24T12:00:00+00:00",
        [small, big, small, big, small, big, small, small, small],
    )
    out = build_cycle_gap_summary(rows)
    assert out["n_stalled_gaps"] == 3
    assert abs(out["pct_stalled"] - (3 / 9 * 100)) < 0.01


def test_coefficient_of_variation_none_when_median_zero():
    # All identical timestamps → all gaps are 0 → median 0 → CoV is None
    # (avoid division by zero — matches the documented "None on
    # degenerate" contract).
    rows = [_row("2026-05-24T12:00:00+00:00")] * 10
    out = build_cycle_gap_summary(rows)
    assert out["median_gap_s"] == 0.0
    assert out["coefficient_of_variation"] is None
    # With median=0 the JITTERY branch can never fire (cov is None), and
    # max gap is 0 which is well below STALL_GAP_S → SMOOTH is the verdict.
    assert out["verdict"] == "SMOOTH"

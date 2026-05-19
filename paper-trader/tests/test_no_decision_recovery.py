"""Tests for paper_trader.analytics.no_decision_recovery.

The verdict drives what the operator does next — WITHIN_NORMAL ("wait"),
ELEVATED ("watch"), ABNORMAL_WEDGE ("escalate now"). A silently-broken
percentile or run-length encoding would misdirect an on-call response, so
every assertion below is on a *specific* expected bucket / verdict / number.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics import no_decision_recovery as ndr  # noqa: E402


def _nd() -> dict:
    """A NO_DECISION row."""
    return {"action_taken": "NO_DECISION", "reasoning": "x",
            "timestamp": "2026-05-19T10:00:00+00:00"}


def _real() -> dict:
    """A real (non-NO_DECISION) row."""
    return {"action_taken": "BUY NVDA → FILLED", "reasoning": "x",
            "timestamp": "2026-05-19T10:00:00+00:00"}


# Builder helper: rows are newest-first as recent_decisions returns them, so
# tests express the *time-ordered* sequence and reverse it at the boundary.
def _newest_first(*chronological_rows: dict) -> list[dict]:
    return list(reversed(chronological_rows))


# ─── canonical-predicate drift lock ───────────────────────────────────

def test_is_no_decision_mirrors_canonical():
    """The predicate must accept exactly ``""`` and ``"NO_DECISION"`` as
    NO_DECISION (verbatim mirror of forensics / heartbeat / reasons).
    A future change in any of those leaves must update this drift-lock too."""
    assert ndr._is_no_decision("NO_DECISION") is True
    assert ndr._is_no_decision("") is True
    assert ndr._is_no_decision(None) is True
    assert ndr._is_no_decision("   ") is True  # whitespace is empty
    assert ndr._is_no_decision("BUY NVDA → FILLED") is False
    assert ndr._is_no_decision("HOLD NVDA → HOLD") is False
    # "BLOCKED" — strategy.py records this when risk gate refuses; Opus DID
    # decide, the engine refused. Canonical predicate treats this as a real
    # decision, not NO_DECISION.
    assert ndr._is_no_decision("BLOCKED") is False


# ─── empty / NO_DATA cases ────────────────────────────────────────────

def test_empty_decisions_returns_no_data():
    out = ndr.build_no_decision_recovery([])
    assert out["state"] == "NO_DATA"
    assert out["verdict"] == "INSUFFICIENT_HISTORY"
    assert out["open_run_length"] == 0
    assert out["completed_runs"] == []


def test_none_decisions_returns_no_data():
    out = ndr.build_no_decision_recovery(None)
    assert out["state"] == "NO_DATA"


# ─── RECOVERED — latest row is a real decision ────────────────────────

def test_all_real_decisions_recovered():
    rows = _newest_first(_real(), _real(), _real())
    out = ndr.build_no_decision_recovery(rows)
    assert out["state"] == "OK"
    assert out["open_run_length"] == 0
    assert out["verdict"] == "RECOVERED"
    assert out["completed_runs"] == []


def test_closed_wedge_then_real_is_recovered():
    """Closed wedges go into the distribution; if the latest row is real,
    the verdict is RECOVERED regardless of how big the past wedges were."""
    # chronological: real, NO×4, real, real
    rows = _newest_first(_real(), _nd(), _nd(), _nd(), _nd(), _real(), _real())
    out = ndr.build_no_decision_recovery(rows)
    assert out["open_run_length"] == 0
    assert out["completed_runs"] == [4]
    assert out["verdict"] == "RECOVERED"


# ─── NOISE — single trailing NO_DECISION ──────────────────────────────

def test_single_trailing_no_decision_is_noise():
    # chronological: real, real, NO   (latest = NO, but only 1)
    rows = _newest_first(_real(), _real(), _nd())
    out = ndr.build_no_decision_recovery(rows)
    assert out["open_run_length"] == 1
    assert out["verdict"] == "NOISE"


# ─── INSUFFICIENT_HISTORY — open wedge but too few past wedges ────────

def test_open_wedge_no_history_insufficient():
    """Open wedge of 3, zero closed wedges → INSUFFICIENT_HISTORY."""
    # chronological: real, NO, NO, NO  (latest = NO, length 3, no closed runs)
    rows = _newest_first(_real(), _nd(), _nd(), _nd())
    out = ndr.build_no_decision_recovery(rows)
    assert out["open_run_length"] == 3
    assert out["completed_runs"] == []
    assert out["verdict"] == "INSUFFICIENT_HISTORY"


def test_open_wedge_few_history_insufficient():
    """Open wedge of 3, only 2 closed wedges → still INSUFFICIENT_HISTORY
    (gate is ≥ 3)."""
    # chronological: NO,NO, real, NO,NO, real, NO,NO,NO
    rows = _newest_first(
        _nd(), _nd(), _real(),
        _nd(), _nd(), _real(),
        _nd(), _nd(), _nd(),
    )
    out = ndr.build_no_decision_recovery(rows)
    assert out["completed_runs"] == [2, 2]
    assert out["n_significant_runs"] == 2
    assert out["open_run_length"] == 3
    assert out["verdict"] == "INSUFFICIENT_HISTORY"


# ─── WITHIN_NORMAL / ELEVATED / ABNORMAL_WEDGE ────────────────────────

def _seq_with_history_then_open(history: list[int], open_len: int) -> list[dict]:
    """Build a chronological row list with the given completed-run lengths
    (each separated by one real decision), followed by an open run of the
    given length. Returns rows in *newest-first* order ready for the builder.
    """
    chrono: list[dict] = [_real()]  # leading anchor so the first run is "closed"
    for L in history:
        chrono.extend([_nd()] * L)
        chrono.append(_real())
    chrono.extend([_nd()] * open_len)
    return list(reversed(chrono))


def test_within_normal_below_median():
    # history of 5 wedges, lengths 2,3,4,5,6 → median=4, p95≈5.8
    rows = _seq_with_history_then_open([2, 3, 4, 5, 6], open_len=2)
    out = ndr.build_no_decision_recovery(rows)
    assert out["n_significant_runs"] == 5
    assert out["median_run_length"] == 4.0
    assert out["open_run_length"] == 2
    assert out["verdict"] == "WITHIN_NORMAL"


def test_within_normal_equals_median():
    rows = _seq_with_history_then_open([2, 3, 4, 5, 6], open_len=4)
    out = ndr.build_no_decision_recovery(rows)
    assert out["open_run_length"] == 4
    assert out["verdict"] == "WITHIN_NORMAL"  # 4 ≤ median 4


def test_elevated_above_median_below_p95():
    rows = _seq_with_history_then_open([2, 3, 4, 5, 6], open_len=5)
    out = ndr.build_no_decision_recovery(rows)
    assert out["open_run_length"] == 5
    assert out["median_run_length"] == 4.0
    # p95 of [2,3,4,5,6] with linear interp = 6 - 0.2 = 5.8
    assert abs(out["p95_run_length"] - 5.8) < 1e-9
    assert out["verdict"] == "ELEVATED"


def test_abnormal_wedge_at_or_above_p95():
    rows = _seq_with_history_then_open([2, 3, 4, 5, 6], open_len=6)
    out = ndr.build_no_decision_recovery(rows)
    assert out["open_run_length"] == 6
    assert out["verdict"] == "ABNORMAL_WEDGE"


# ─── basic stat correctness ────────────────────────────────────────────

def test_stats_over_significant_runs_only():
    """Single-cycle hiccups (length 1) must be excluded from the distribution
    — they're qualitatively noise, not wedges."""
    # chronological: NO, real, NO,NO,NO, real, NO,NO, real  (lengths 1, 3, 2)
    chrono = [
        _nd(),
        _real(),
        _nd(), _nd(), _nd(),
        _real(),
        _nd(), _nd(),
        _real(),
    ]
    rows = list(reversed(chrono))
    out = ndr.build_no_decision_recovery(rows)
    assert out["completed_runs"] == [1, 3, 2]
    # significant = [2, 3] → mean=2.5, median=2.5, max=3
    assert out["n_significant_runs"] == 2
    assert out["mean_run_length"] == 2.5
    assert out["median_run_length"] == 2.5
    assert out["max_run_length"] == 3
    assert out["open_run_length"] == 0
    assert out["verdict"] == "RECOVERED"


def test_percentile_single_value_returns_value():
    """Pinning the percentile leaf — a single significant run reports itself
    as both median and p95 (no interpolation possible)."""
    assert ndr._percentile([4], 0.5) == 4.0
    assert ndr._percentile([4], 0.95) == 4.0


def test_percentile_empty_returns_none():
    assert ndr._percentile([], 0.5) is None


# ─── window clamping ───────────────────────────────────────────────────

def test_window_clamps_negative_to_default():
    rows = _newest_first(_real())
    out = ndr.build_no_decision_recovery(rows, window=-5)
    # Clamped to ≥1; the actual value should be ≥1 and rows considered = 1.
    assert out["window"] >= 1
    assert out["n_rows"] == 1


def test_window_clamps_non_int_to_default():
    rows = _newest_first(_real())
    out = ndr.build_no_decision_recovery(rows, window="garbage")  # type: ignore[arg-type]
    assert out["window"] == ndr.DEFAULT_WINDOW


def test_window_limits_rows():
    """Window of 2 over 5 rows should consider only the 2 newest."""
    # chronological: real, NO,NO,NO, real  → newest = real, then 3xNO closed run
    rows = _newest_first(_real(), _nd(), _nd(), _nd(), _real())
    out = ndr.build_no_decision_recovery(rows, window=2)
    # Window=2 keeps the 2 newest = [real, NO]. Oldest-first: NO, real.
    # Open run = 0 (latest = real), completed run = [1].
    assert out["n_rows"] == 2
    assert out["open_run_length"] == 0
    assert out["completed_runs"] == [1]


# ─── shape: every promised field present ───────────────────────────────

def test_result_shape_all_fields_present():
    rows = _newest_first(_real(), _nd(), _nd(), _real())
    out = ndr.build_no_decision_recovery(rows)
    for k in (
        "state", "window", "n_rows", "open_run_length",
        "completed_runs", "n_completed_runs", "n_significant_runs",
        "mean_run_length", "median_run_length", "p95_run_length",
        "max_run_length", "verdict", "verdict_detail", "headline",
    ):
        assert k in out, f"missing key {k!r}"

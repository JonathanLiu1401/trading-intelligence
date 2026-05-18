"""Tests for scripts/stale_dashboard_sentinel.py — asserts the scanner
distinguishes stale-client (oversized) curve 400s from ordinary ones and
honours the recency window and fail threshold."""
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "stale_dashboard_sentinel",
    Path(__file__).resolve().parent.parent / "scripts" / "stale_dashboard_sentinel.py",
)
sds = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sds)


def _line(ts: str, n_ids: int, status: int) -> str:
    ids = ",".join(str(i) for i in range(n_ids))
    return (f'127.0.0.1 - - [{ts}] "GET /api/backtests/curves?run_ids={ids} '
            f'HTTP/1.1" {status} -')


def test_oversized_400_is_flagged_small_is_not(tmp_path):
    log = tmp_path / "runner.log"
    log.write_text("\n".join([
        _line("17/May/2026 16:00:00", 250, 400),   # stale client
        _line("17/May/2026 16:05:00", 250, 400),   # stale client
        _line("17/May/2026 16:06:00", 5, 400),     # ordinary small 400
        _line("17/May/2026 16:07:00", 250, 200),   # not a 400, ignore
        '127.0.0.1 - - [17/May/2026 16:08:00] "GET /api/state HTTP/1.1" 200 -',
    ]))
    s = sds.scan(log, max_age_min=0)
    assert s["stale_client_400"] == 2
    assert s["other_curve_400"] == 1
    assert s["first"] == "2026-05-17T16:00:00"
    assert s["last"] == "2026-05-17T16:05:00"


def test_recency_window_anchors_to_newest_log_entry(tmp_path):
    log = tmp_path / "runner.log"
    log.write_text("\n".join([
        _line("17/May/2026 10:00:00", 250, 400),   # old, outside 60min window
        _line("17/May/2026 16:00:00", 250, 400),   # newest anchor
        _line("17/May/2026 15:30:00", 250, 400),   # within 60min of anchor
    ]))
    s = sds.scan(log, max_age_min=60)
    assert s["stale_client_400"] == 2  # 16:00 and 15:30, not 10:00


def test_exit_codes(tmp_path):
    log = tmp_path / "runner.log"
    log.write_text(_line("17/May/2026 16:00:00", 250, 400) + "\n"
                   + _line("17/May/2026 16:01:00", 250, 400) + "\n"
                   + _line("17/May/2026 16:02:00", 250, 400))
    # 3 hits, threshold 3 -> exit 1
    assert sds.main(["--log", str(log), "--max-age-min", "0",
                     "--fail-threshold", "3"]) == 1
    # same hits, threshold 10 -> still 0 (below threshold)
    assert sds.main(["--log", str(log), "--max-age-min", "0",
                     "--fail-threshold", "10"]) == 0
    # missing log -> exit 2
    assert sds.main(["--log", str(tmp_path / "nope.log")]) == 2


def test_clean_log_exits_zero(tmp_path):
    log = tmp_path / "runner.log"
    log.write_text('127.0.0.1 - - [17/May/2026 16:00:00] "GET /api/state HTTP/1.1" 200 -')
    assert sds.main(["--log", str(log), "--max-age-min", "0"]) == 0

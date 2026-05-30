"""Hourly analytics orchestrator.

Runs a curated set of read-only analytics modules in parallel (ThreadPoolExecutor)
and writes a combined snapshot to /home/zeph/logs/hourly_analytics_run.json.

Each task is given an independent timeout; one hung module cannot stall the rest.
Exit code is 0 even if some tasks fail (observability tool, not a gate).

Standalone:  python3 -m analytics.hourly_runner
"""
from __future__ import annotations

import json
import math
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeout
from datetime import datetime, timezone
from pathlib import Path

OUT_PATH = Path("/home/zeph/logs/hourly_analytics_run.json")
TASK_TIMEOUT = 60   # seconds per individual module
MAX_WORKERS = 4


def _run_module(name: str, fn) -> tuple[str, bool, str]:
    """Run fn(), return (name, ok, detail)."""
    t0 = time.monotonic()
    try:
        fn()
        elapsed = round(time.monotonic() - t0, 2)
        return name, True, f"{elapsed}s"
    except Exception as exc:
        elapsed = round(time.monotonic() - t0, 2)
        tb = traceback.format_exc().strip().splitlines()[-1]
        return name, False, f"{elapsed}s | {tb}"


def main() -> int:
    # Import lazily so a broken module doesn't prevent others from running
    tasks: dict[str, object] = {}

    def _try_import(mod_path: str, fn_name: str = "main") -> object | None:
        try:
            import importlib
            mod = importlib.import_module(mod_path)
            return getattr(mod, fn_name, None)
        except Exception:
            return None

    # (mod_path, fn_name) — fn_name defaults to "main" when omitted
    module_specs = [
        ("trend_velocity",            "analytics.trend_velocity",            "main"),
        ("signal_consolidator",       "analytics.signal_consolidator",       "main"),
        ("consensus_signal",          "analytics.consensus_signal",          "main"),
        ("junk_source_detector",      "analytics.junk_source_detector",      "main"),
        ("score_drift_detector",      "analytics.score_drift_detector",      "main"),
        ("ingest_gap_detector",       "analytics.ingest_gap_detector",       "main"),
        ("stale_source_alerter",      "analytics.stale_source_alerter",      "main"),
        ("source_quality",            "analytics.source_quality",            "main"),
        ("collection_quality",        "analytics.collection_quality",        "main"),
        ("ticker_sentiment_momentum", "analytics.ticker_sentiment_momentum", "main"),
        ("earnings_preheat",          "analytics.earnings_preheat",          "main"),
        ("sentiment_streak",          "analytics.sentiment_streak",          "main"),
        ("source_health_report",      "analytics.source_health_report",      "main"),
        ("breaking_news_alerter",     "analytics.breaking_news_alerter",     "main"),
        ("ticker_score_acceleration", "analytics.ticker_score_acceleration", "main"),
        ("premarket_brief",           "analytics.premarket_brief",           "main"),
        ("ticker_cold_case_detector", "analytics.ticker_cold_case_detector", "main"),
        ("signal_cascade_detector",   "analytics.signal_cascade_detector",   "main"),
        ("ml_coverage_rate",          "analytics.ml_coverage_rate",          "main"),
        ("ml_coverage_by_source",     "analytics.ml_coverage_by_source",     "main"),
        ("dow_baseline",              "analytics.dow_baseline",              "main"),
        ("urgency_spike_detector",    "analytics.urgency_spike_detector",    "main"),
        ("ticker_signal_noise",       "analytics.ticker_signal_noise",       "main"),
        ("hourly_urgency_quality",    "analytics.hourly_urgency_quality",    "main"),
        ("ml_confidence_tracker",     "analytics.ml_confidence_tracker",     "main"),
        ("db_lock_tax",               "analytics.db_lock_tax",               "main"),
        ("sector_pulse",              "analytics.sector_pulse",              "main"),
        ("keyword_surge",             "analytics.keyword_surge",             "main"),
        ("active_ticker_dashboard",   "analytics.active_ticker_dashboard",   "main"),
        ("regulatory_entity_surge",   "analytics.regulatory_entity_surge",   "main"),
        ("triple_score_consensus",    "analytics.triple_score_consensus",    "main"),
        ("recency_decay",             "analytics.recency_decay",             "main"),
        ("portfolio_overlap_scorer",  "analytics.portfolio_overlap_scorer",  "main"),
        ("ticker_resurrection",       "analytics.ticker_resurrection",       "main"),
    ]

    for short_name, mod_path, fn_name in module_specs:
        fn = _try_import(mod_path, fn_name)
        if fn is not None:
            tasks[short_name] = fn
        else:
            tasks[short_name] = None  # mark import failure

    results: list[dict] = []
    ok_count = 0
    fail_count = 0

    runnable = {name: fn for name, fn in tasks.items() if fn is not None}
    # Total wall-clock budget: ceil(n / workers) rounds × per-task timeout + 30s buffer
    total_timeout = math.ceil(len(runnable) / MAX_WORKERS) * TASK_TIMEOUT + 30

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:  # noqa: SIM117
        future_map = {pool.submit(_run_module, name, fn): name for name, fn in runnable.items()}
        # Record import failures immediately
        for name, fn in tasks.items():
            if fn is None:
                results.append({"module": name, "ok": False, "detail": "import failed"})
                fail_count += 1

        try:
            for future in as_completed(future_map, timeout=total_timeout):
                try:
                    name, ok, detail = future.result(timeout=TASK_TIMEOUT)
                except FutureTimeout:
                    name = future_map[future]
                    ok, detail = False, "timeout"
                except Exception as exc:
                    name = future_map[future]
                    ok, detail = False, str(exc)[:120]

                results.append({"module": name, "ok": ok, "detail": detail})
                if ok:
                    ok_count += 1
                else:
                    fail_count += 1
        except FutureTimeout:
            # Mark any futures that never completed
            done_names = {r["module"] for r in results}
            for fut, name in future_map.items():
                if name not in done_names:
                    results.append({"module": name, "ok": False, "detail": "outer-timeout"})
                    fail_count += 1
                    fut.cancel()

    results.sort(key=lambda r: r["module"])

    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ok": ok_count,
        "failed": fail_count,
        "total": len(results),
        "tasks": results,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(snapshot, indent=2))

    # Print summary to stdout
    print(f"hourly_runner: {ok_count}/{len(results)} ok  ({fail_count} failed)")
    for r in results:
        status = "OK " if r["ok"] else "ERR"
        print(f"  [{status}] {r['module']}: {r['detail']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Analytics batch runner.

Runs a prioritised set of analytics modules in sequence, skipping any that
were refreshed recently (within FRESH_THRESHOLD_MINUTES).  Designed to be
called from the hourly audit cron so all analytics outputs stay current.

Priority ordering:
  1. Raw signals  (trend_velocity, breaking_news_detector, sector_rotation)
  2. Derived signals (ticker_sentiment_momentum, score_drift_detector, consensus_signal)
  3. Composites   (confluence_signals, ticker_alert_ranker)
  4. Quality/health (collection_quality, daily_digest)

Output: /home/zeph/logs/batch_runner.json

Standalone: python3 -m analytics.batch_runner
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOGS = Path("/home/zeph/logs")
OUT_PATH = LOGS / "batch_runner.json"

# Modules ordered by dependency (outputs of earlier feed into later)
PIPELINE: list[tuple[str, Path, int]] = [
    # (module, output_file, fresh_threshold_minutes)
    ("analytics.trend_velocity",          LOGS / "trend_velocity.json",          50),
    ("analytics.breaking_news_detector",  LOGS / "breaking_news.jsonl",          50),
    ("analytics.sector_rotation",         LOGS / "sector_rotation.json",         50),
    ("analytics.ticker_sentiment_momentum", LOGS / "ticker_sentiment_momentum.json", 50),
    ("analytics.score_drift_detector",    LOGS / "score_drift.log",              50),
    ("analytics.consensus_signal",        LOGS / "strong_consensus.jsonl",       50),
    ("analytics.confluence_signals",      LOGS / "confluence_signals.json",      50),
    ("analytics.ticker_alert_ranker",     LOGS / "ticker_alert_rank.json",       50),
    ("analytics.source_score_drift",      LOGS / "source_score_drift.json",      50),
    ("analytics.collection_quality",      LOGS / "collection_quality.json",     110),
    ("analytics.daily_digest",            LOGS / "daily_digest.txt",            110),
]

TIMEOUT_S = 60  # per-module hard timeout


def _is_fresh(path: Path, threshold_minutes: int) -> bool:
    try:
        mtime = path.stat().st_mtime
        age_s = time.time() - mtime
        return age_s < threshold_minutes * 60
    except FileNotFoundError:
        return False


def run() -> dict:
    now = datetime.now(timezone.utc)
    results = []
    ran = skipped = failed = 0

    for module, output_path, fresh_min in PIPELINE:
        if _is_fresh(output_path, fresh_min):
            results.append({"module": module, "status": "skipped_fresh",
                            "output": output_path.name})
            skipped += 1
            continue

        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                [sys.executable, "-m", module],
                capture_output=True, text=True,
                cwd=str(REPO), timeout=TIMEOUT_S,
            )
            elapsed = round(time.monotonic() - t0, 1)
            if proc.returncode == 0:
                results.append({"module": module, "status": "ok",
                                "elapsed_s": elapsed, "output": output_path.name})
                ran += 1
            else:
                results.append({"module": module, "status": "error",
                                "returncode": proc.returncode,
                                "stderr": proc.stderr[-300:].strip(),
                                "elapsed_s": elapsed})
                failed += 1
        except subprocess.TimeoutExpired:
            results.append({"module": module, "status": "timeout",
                            "timeout_s": TIMEOUT_S})
            failed += 1
        except Exception as e:
            results.append({"module": module, "status": "exception",
                            "error": str(e)})
            failed += 1

    summary = {
        "generated_at": now.isoformat(),
        "ran": ran,
        "skipped_fresh": skipped,
        "failed": failed,
        "modules": results,
    }
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    summary = run()
    print(f"batch_runner: ran={summary['ran']} skipped={summary['skipped_fresh']} "
          f"failed={summary['failed']}")
    for r in summary["modules"]:
        status = r["status"]
        mod = r["module"].split(".")[-1]
        if status == "ok":
            print(f"  ✓ {mod} ({r['elapsed_s']}s)")
        elif status == "skipped_fresh":
            print(f"  ~ {mod} (fresh)")
        else:
            print(f"  ✗ {mod} ({status})")


if __name__ == "__main__":
    main()

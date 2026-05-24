#!/usr/bin/env python3
"""ML trainer watchdog — scheduled runner that surfaces training-cycle health.

Wraps ``analytics.ml_training_health.compute()`` and writes a JSON snapshot
to ``/home/zeph/logs/ml_training_health.json`` so the dashboard and hourly
cron audits can consume it without hitting the metrics JSONL directly.

Alert verdicts that print a WARNING-level summary to stdout and exit non-zero:
  * DEAD        — no successful train phase in >48h (ArticleNet silently stale)
  * ERROR_HEAVY — >50% of recent cycles errored (runner alive but broken)
  * DIVERGING   — val_loss rising 3+ consecutive cycles (model degrading)

STALE and HEALTHY exit 0 so a cron wrapper can use exit-code gating.

Standalone:  python3 scripts/ml_trainer_watchdog.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from analytics.ml_training_health import compute

OUT = Path("/home/zeph/logs/ml_training_health.json")
ALERT_VERDICTS = {"DEAD", "ERROR_HEAVY", "DIVERGING"}


def write_snapshot(snap: dict, path: Path = OUT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {**snap, "generated_at": datetime.now(timezone.utc).isoformat()}
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path


def main() -> int:
    snap = compute()
    verdict = snap["verdict"]
    last_h = snap["last_train_age_h"]
    age_str = f"{last_h:.1f}h ago" if last_h is not None else "never"
    trend = snap.get("val_loss_trend", [])

    snap_path = write_snapshot(snap)

    # Line 1 — consumed verbatim by the hourly audit cron
    print(
        f"ml_trainer_watchdog: verdict={verdict} "
        f"last_train={age_str} "
        f"train_in_72h={snap['train_in_window']} "
        f"errors={snap['errors_in_window']}/{snap['total_in_window']}"
    )
    if trend:
        print(f"  val_loss trend (newest→oldest): {trend}")
    if snap.get("diverging"):
        print("  WARNING: val_loss rising 3+ consecutive cycles — model degrading")
    if verdict in ALERT_VERDICTS:
        print(
            f"  ALERT: trainer is {verdict} — "
            f"ArticleNet pre-filter may be stale/broken; "
            f"check ml/trainer.py subprocess and data/ml/training_metrics.jsonl"
        )
    print(f"  snapshot → {snap_path}")

    return 1 if verdict in ALERT_VERDICTS else 0


if __name__ == "__main__":
    sys.exit(main())

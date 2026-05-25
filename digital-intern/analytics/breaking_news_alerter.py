"""Breaking news Discord alerter.

Runs breaking_news_detector.detect() over live articles, then sends a
Discord alert for any ticker whose breaking-news burst hasn't been alerted
within COOLDOWN_MINUTES.  Prevents alert storms by tracking per-ticker
last-alerted timestamps in a JSON state file.

State file: /home/zeph/logs/.breaking_news_alerted.json
  {"NVDA": "2026-05-25T07:14:00+00:00", ...}

Standalone: python3 -m analytics.breaking_news_alerter
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from analytics.breaking_news_detector import detect, DB_PATH, FETCH_LIMIT, LOOKBACK_HOURS
from storage.article_store import _LIVE_ONLY_CLAUSE

STATE_PATH = Path("/home/zeph/logs/.breaking_news_alerted.json")
COOLDOWN_MINUTES = 30   # don't re-alert same ticker within this window


def _load_state() -> dict[str, str]:
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:
        return {}


def _save_state(state: dict[str, str]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def main() -> int:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=20)
    conn.execute("PRAGMA query_only=ON")
    since = (datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    rows = conn.execute(
        "SELECT first_seen, title, source FROM articles INDEXED BY idx_first_seen "
        f"WHERE first_seen >= ? AND {_LIVE_ONLY_CLAUSE} "
        "ORDER BY first_seen DESC LIMIT ?",
        (since, FETCH_LIMIT),
    ).fetchall()
    conn.close()

    events = detect(rows)
    if not events:
        print("breaking_news_alerter: no breaking events detected")
        return 0

    now = datetime.now(timezone.utc)
    cooldown = timedelta(minutes=COOLDOWN_MINUTES)
    state = _load_state()

    # Prune stale state entries (older than 2h)
    state = {
        tk: ts for tk, ts in state.items()
        if (now - datetime.fromisoformat(ts)) < timedelta(hours=2)
    }

    new_events = []
    for ev in events:
        ticker = ev["ticker"]
        last_ts_str = state.get(ticker)
        if last_ts_str:
            last_ts = datetime.fromisoformat(last_ts_str)
            if (now - last_ts) < cooldown:
                continue
        new_events.append(ev)

    if not new_events:
        print(f"breaking_news_alerter: {len(events)} events, all within cooldown — no alert sent")
        return 0

    # Build Discord message
    lines = [f"**BREAKING NEWS** [{now.strftime('%H:%M UTC')}]"]
    for ev in new_events[:5]:   # cap at 5 to avoid Discord message bloat
        sources_str = ", ".join(ev["sources"][:3])
        lines.append(
            f"**{ev['ticker']}**: {ev['count']} articles in {LOOKBACK_HOURS}h window"
            f" | sources: {sources_str}"
        )
        if ev.get("sample_title"):
            lines.append(f"  > {ev['sample_title'][:120]}")
    if len(new_events) > 5:
        lines.append(f"  _...and {len(new_events) - 5} more_")

    message = "\n".join(lines)

    try:
        from notifier.discord_notifier import send
        ok = send(message, is_alert=True)
    except Exception as exc:
        print(f"breaking_news_alerter: discord send failed: {exc}", file=sys.stderr)
        ok = False

    # Update state for alerted tickers
    for ev in new_events:
        state[ev["ticker"]] = now.isoformat()
    _save_state(state)

    status = "sent" if ok else "send_failed"
    print(
        f"breaking_news_alerter: {len(new_events)} new events alerted ({status}), "
        f"{len(events) - len(new_events)} suppressed by cooldown"
    )
    for ev in new_events[:5]:
        print(f"  ALERTED {ev['ticker']}: {ev['count']} articles | {','.join(ev['sources'][:3])}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

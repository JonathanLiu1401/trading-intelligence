"""
Digital Intern — live monitor dashboard.

Run:  python3 monitor.py
      python3 monitor.py --json      # one-shot JSON snapshot for scripting
      python3 monitor.py --tail      # follow structured.jsonl in real time

Shows:
  - systemd service health
  - worker thread status (parsed from logs)
  - article store stats (total / urgent / unscored)
  - last heartbeat / last alert timestamps
  - recent log tail
  - key metric rates (articles/min, alerts/hr)
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR  = BASE_DIR / "logs"
STRUCT   = LOG_DIR / "structured.jsonl"
METRICS  = LOG_DIR / "metrics.jsonl"
PLAIN    = LOG_DIR / "daemon.log"

WORKERS = ["gdelt", "rss", "web", "reddit", "ticker", "scorer", "alert", "heartbeat", "purge", "stats"]

# ANSI
_B  = "\033[1m"
_R  = "\033[0m"
_G  = "\033[32m"
_Y  = "\033[33m"
_RD = "\033[31m"
_C  = "\033[36m"
_M  = "\033[35m"
_W  = "\033[37m"
_CLEAR = "\033[2J\033[H"


def _service_status() -> dict:
    try:
        r = subprocess.run(
            ["systemctl", "is-active", "digital-intern"],
            capture_output=True, text=True, timeout=5,
        )
        active = r.stdout.strip()
        r2 = subprocess.run(
            ["systemctl", "show", "digital-intern",
             "--property=ActiveEnterTimestamp,MainPID,RestartCount"],
            capture_output=True, text=True, timeout=5,
        )
        props = dict(line.split("=", 1) for line in r2.stdout.strip().splitlines() if "=" in line)
        return {
            "active": active,
            "pid": props.get("MainPID", "?"),
            "restarts": props.get("RestartCount", "?"),
            "since": props.get("ActiveEnterTimestamp", "?"),
        }
    except Exception as e:
        return {"active": "error", "pid": "?", "restarts": "?", "since": str(e)}


def _store_stats() -> dict:
    try:
        sys.path.insert(0, str(BASE_DIR))
        from dotenv import load_dotenv
        load_dotenv(BASE_DIR / ".env")
        from storage.article_store import ArticleStore
        s = ArticleStore()
        return s.stats()
    except Exception as e:
        return {"error": str(e)}


def _parse_structured(n: int = 500) -> list[dict]:
    if not STRUCT.exists():
        return []
    lines = []
    with open(STRUCT, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except Exception:
                    pass
    return lines[-n:]


def _worker_health(records: list[dict]) -> dict:
    """Find last seen timestamp per worker from structured log."""
    last_seen = {}
    for r in records:
        msg = r.get("msg", "")
        for w in WORKERS:
            if f"[{w}" in msg or f"[{w}_worker]" in msg:
                last_seen[w] = r.get("ts", "")
    return last_seen


def _recent_events(records: list[dict]) -> list[dict]:
    """Pull last heartbeat, last alert, last error."""
    heartbeat = None
    alert = None
    errors = []
    for r in reversed(records):
        msg = r.get("msg", "")
        if not heartbeat and "[heartbeat]" in msg and "sent" in msg:
            heartbeat = r
        if not alert and "[alert]" in msg and "urgent" in msg.lower():
            alert = r
        if r.get("level") in ("ERROR", "CRITICAL"):
            errors.append(r)
        if heartbeat and alert and len(errors) >= 3:
            break
    return {"heartbeat": heartbeat, "alert": alert, "errors": errors[:5]}


def _article_rate(records: list[dict]) -> float:
    """Compute articles ingested in last 60 minutes."""
    cutoff = time.time() - 3600
    count = 0
    for r in records:
        try:
            ts = datetime.fromisoformat(r["ts"].replace("Z", "+00:00")).timestamp()
        except Exception:
            continue
        if ts < cutoff:
            continue
        m = re.search(r"\+(\d+) new articles", r.get("msg", ""))
        if m:
            count += int(m.group(1))
    return count


def _tail_log(n: int = 20) -> list[str]:
    if not PLAIN.exists():
        return ["(no log file yet)"]
    with open(PLAIN, encoding="utf-8") as f:
        return list(deque(f, maxlen=n))


def _fmt_age(iso: str) -> str:
    if not iso:
        return "never"
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        secs = (datetime.now(timezone.utc) - dt).total_seconds()
        if secs < 60:
            return f"{int(secs)}s ago"
        if secs < 3600:
            return f"{int(secs/60)}m ago"
        return f"{int(secs/3600)}h {int((secs%3600)/60)}m ago"
    except Exception:
        return iso


def _colour_level(level: str) -> str:
    return {
        "DEBUG": _W, "INFO": _G, "WARNING": _Y,
        "ERROR": _RD, "CRITICAL": _M,
    }.get(level, _W)


def snapshot() -> dict:
    records = _parse_structured(1000)
    svc     = _service_status()
    store   = _store_stats()
    health  = _worker_health(records)
    events  = _recent_events(records)
    rate    = _article_rate(records)

    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "service": svc,
        "store": store,
        "worker_last_seen": health,
        "recent_events": {
            "heartbeat": events["heartbeat"],
            "alert": events["alert"],
        },
        "errors_last_hour": events["errors"],
        "articles_per_hour": rate,
    }


def display(snap: dict):
    svc   = snap["service"]
    store = snap["store"]
    health = snap["worker_last_seen"]
    events = snap["recent_events"]
    rate   = snap["articles_per_hour"]

    status_col = _G if svc["active"] == "active" else _RD
    print(f"{_B}┌─ DIGITAL INTERN MONITOR ────────────────────────────────────────────┐{_R}")
    print(f"  {_C}Service:{_R} {status_col}{svc['active'].upper()}{_R}  "
          f"PID={svc['pid']}  Restarts={svc['restarts']}  Since: {svc['since'][:19]}")
    print(f"  {_C}Articles/hr:{_R} {_B}{int(rate)}{_R}  "
          f"{_C}Total:{_R} {store.get('total','?')}  "
          f"{_C}Urgent:{_R} {_Y}{store.get('urgent','?')}{_R}  "
          f"{_C}Unscored:{_R} {store.get('unscored','?')}  "
          f"{_C}DB:{_R} {store.get('db_mb','?')} MB")
    print(f"  {_C}Last heartbeat:{_R} {_fmt_age(events['heartbeat']['ts'] if events.get('heartbeat') else '')}  "
          f"{_C}Last alert:{_R} {_fmt_age(events['alert']['ts'] if events.get('alert') else '')}")

    print(f"\n{_B}  Workers:{_R}")
    now = datetime.now(timezone.utc).timestamp()
    for w in WORKERS:
        ts = health.get(w, "")
        if not ts:
            age_s = 99999
            age_str = "never seen"
        else:
            try:
                age_s = now - datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
                age_str = _fmt_age(ts)
            except Exception:
                age_s = 0
                age_str = ts
        col = _G if age_s < 120 else (_Y if age_s < 600 else _RD)
        print(f"    {col}{'●' if age_s < 120 else '○'}{_R} {w:<12} {col}{age_str}{_R}")

    records = _parse_structured(200)
    errors = [r for r in records if r.get("level") in ("ERROR", "CRITICAL")]
    if errors:
        print(f"\n{_B}  Recent errors:{_R}")
        for e in errors[-5:]:
            print(f"    {_RD}[{e['ts'][11:19]}] {e['logger']}: {e['msg'][:80]}{_R}")

    print(f"\n{_B}  Log tail:{_R}")
    for line in _tail_log(12):
        line = line.rstrip()
        col = _RD if " ERROR " in line or " CRITICAL " in line else (
              _Y if " WARNING " in line else _W)
        print(f"  {col}{line[:100]}{_R}")

    print(f"\n  {_W}Refreshed: {snap['ts'][11:19]} UTC   Ctrl+C to exit{_R}")
    print(f"{_B}└─────────────────────────────────────────────────────────────────────┘{_R}")


def tail_mode():
    """Stream structured.jsonl in real time."""
    print(f"{_C}Tailing {STRUCT} — Ctrl+C to stop{_R}\n")
    with open(STRUCT, encoding="utf-8") as f:
        f.seek(0, 2)  # seek to end
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.1)
                continue
            try:
                r = json.loads(line)
                level = r.get("level", "INFO")
                col = _colour_level(level)
                ts = r.get("ts", "")[:19]
                name = r.get("logger", "")
                msg = r.get("msg", "")
                print(f"{col}{ts} [{level[0]}] {name}: {msg}{_R}")
            except Exception:
                print(line.rstrip())


def main():
    parser = argparse.ArgumentParser(description="Digital Intern live monitor")
    parser.add_argument("--json",     action="store_true", help="One-shot JSON snapshot")
    parser.add_argument("--tail",     action="store_true", help="Follow structured log")
    parser.add_argument("--interval", type=int, default=10, help="Refresh interval seconds (default 10)")
    args = parser.parse_args()

    if args.json:
        print(json.dumps(snapshot(), indent=2, default=str))
        return

    if args.tail:
        tail_mode()
        return

    # Live dashboard
    try:
        while True:
            snap = snapshot()
            print(_CLEAR, end="")
            display(snap)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()

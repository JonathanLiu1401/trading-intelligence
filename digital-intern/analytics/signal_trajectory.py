"""Signal trajectory tracker.

Snapshots the current momentum_brief.json top-signals each run and appends to
a rolling JSONL log.  Reads back the last 6h of entries to classify each
active ticker as RISING / STABLE / FADING based on confluence_score change.

Classification (vs score 3 entries ago, or earliest available):
  RISING  : delta >= +0.5
  FADING  : delta <= -0.5
  STABLE  : within ±0.5 band
  NEW     : first appearance in the window

Output:
  /home/zeph/logs/signal_trajectory.jsonl  — rolling JSONL history (auto-trimmed to 24h)
  /home/zeph/logs/signal_trajectory.json   — latest snapshot with trajectory labels

Standalone: python3 -m analytics.signal_trajectory
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

LOG_DIR = Path("/home/zeph/logs")
BRIEF_PATH = LOG_DIR / "momentum_brief.json"
HISTORY_PATH = LOG_DIR / "signal_trajectory.jsonl"
OUT_PATH = LOG_DIR / "signal_trajectory.json"

LOOKBACK_HOURS = 6
LOOKBACK_ENTRIES = 3   # compare current score vs 3 snapshots ago
RISE_THRESHOLD = 0.5
FADE_THRESHOLD = -0.5
TRIM_HOURS = 24        # keep only 24h of history in the JSONL


def _load_brief() -> list[dict] | None:
    try:
        data = json.loads(BRIEF_PATH.read_text())
        return data.get("top", [])
    except Exception:
        return None


def _append_snapshot(entries: list[dict], ts: str) -> None:
    snapshot = {"ts": ts, "entries": entries}
    with HISTORY_PATH.open("a") as fh:
        fh.write(json.dumps(snapshot) + "\n")


def _load_history(hours: int = LOOKBACK_HOURS) -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    rows = []
    for line in HISTORY_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            ts = datetime.fromisoformat(obj["ts"].replace("Z", "+00:00"))
            if ts >= cutoff:
                rows.append(obj)
        except Exception:
            continue
    return rows


def _trim_history() -> None:
    if not HISTORY_PATH.exists():
        return
    cutoff = datetime.now(timezone.utc) - timedelta(hours=TRIM_HOURS)
    kept = []
    for line in HISTORY_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            ts = datetime.fromisoformat(obj["ts"].replace("Z", "+00:00"))
            if ts >= cutoff:
                kept.append(line)
        except Exception:
            kept.append(line)
    HISTORY_PATH.write_text("\n".join(kept) + "\n" if kept else "")


def _classify(current_score: float, prior_score: float | None) -> str:
    if prior_score is None:
        return "NEW"
    delta = current_score - prior_score
    if delta >= RISE_THRESHOLD:
        return "RISING"
    if delta <= FADE_THRESHOLD:
        return "FADING"
    return "STABLE"


def main() -> int:
    now_ts = datetime.now(timezone.utc).isoformat()

    brief = _load_brief()
    if brief is None:
        print("signal_trajectory: momentum_brief.json not found — skipped")
        return 1

    # Compact snapshot to store (ticker + score + signals)
    snapshot_entries = [
        {
            "ticker": e.get("ticker", ""),
            "score": round(float(e.get("confluence_score", 0)), 3),
            "signals": e.get("signals", []),
        }
        for e in brief
        if e.get("ticker")
    ]

    _append_snapshot(snapshot_entries, now_ts)
    _trim_history()

    # Build score history per ticker from the rolling window
    history = _load_history()
    # Most-recent first for lookback logic
    history_rev = list(reversed(history))

    ticker_history: dict[str, list[float]] = {}
    for snap in history_rev:
        for entry in snap.get("entries", []):
            t = entry.get("ticker")
            s = entry.get("score")
            if t and s is not None:
                ticker_history.setdefault(t, []).append(float(s))

    results = []
    for entry in snapshot_entries:
        ticker = entry["ticker"]
        score = entry["score"]
        scores = ticker_history.get(ticker, [])
        # scores[0] = current (just appended), scores[N] = N snapshots ago
        prior = scores[LOOKBACK_ENTRIES] if len(scores) > LOOKBACK_ENTRIES else None
        direction = _classify(score, prior)
        delta = round(score - prior, 3) if prior is not None else None
        first_seen_entry = None
        for snap in reversed(history):
            if any(e.get("ticker") == ticker for e in snap.get("entries", [])):
                first_seen_entry = snap["ts"]
                break

        results.append({
            "ticker": ticker,
            "score": score,
            "direction": direction,
            "delta": delta,
            "signals": entry["signals"],
            "first_seen": first_seen_entry,
            "history_depth": len(scores),
        })

    output = {
        "generated_at": now_ts,
        "snapshot_count": len(history),
        "signals": results,
    }

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(output, indent=2))

    rising = [r for r in results if r["direction"] == "RISING"]
    fading = [r for r in results if r["direction"] == "FADING"]
    new = [r for r in results if r["direction"] == "NEW"]

    print(
        f"signal_trajectory: {len(results)} active | "
        f"RISING={len(rising)} STABLE={len(results)-len(rising)-len(fading)-len(new)} "
        f"FADING={len(fading)} NEW={len(new)}"
    )
    for r in results:
        delta_str = f" Δ{r['delta']:+.1f}" if r["delta"] is not None else ""
        print(f"  {r['ticker']:<8} {r['direction']:<7}{delta_str}  score={r['score']}  [{','.join(r['signals'])}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())

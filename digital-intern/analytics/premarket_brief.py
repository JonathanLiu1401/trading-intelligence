"""Pre-market brief synthesizer.

Reads existing analytics artifacts (no direct DB access) and produces a ranked
watchlist of tickers to watch at the next market open.

Scoring formula per ticker:
  gap_score   = max_ml_score from overnight_gaps (0 if absent)
  velocity    = delta mentions from trend_velocity (0 if absent)
  consensus   = 1 if ticker in recent strong_consensus.jsonl, else 0
  corroborated= 1 if ticker appears in corroboration_digest high-signal stories
  composite   = 0.4*gap_score + 0.3*velocity + 15*consensus + 10*corroborated

Output: /home/zeph/logs/premarket_brief.json
Standalone: python3 -m analytics.premarket_brief
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

LOGS = Path("/home/zeph/logs")
OUT_PATH = LOGS / "premarket_brief.json"
TOP_N = 10

# input artifacts
OVERNIGHT_GAPS  = LOGS / "overnight_gaps.json"
TREND_VELOCITY  = LOGS / "trend_velocity.json"
CONSENSUS_JSONL = LOGS / "strong_consensus.jsonl"
CORROBORATION   = LOGS / "corroboration_digest.json"

# only treat consensus events within the last 12h as current
CONSENSUS_WINDOW_H = 12

TICKER_RE = re.compile(r"\b([A-Z]{2,5})\b")


def _load_json(path: Path) -> dict | list | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _tickers_in_text(text: str) -> set[str]:
    return set(TICKER_RE.findall(text or ""))


def _parse_iso(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def compute() -> dict:
    now = datetime.now(timezone.utc)
    scores: dict[str, dict] = defaultdict(lambda: {
        "gap_score": 0.0, "velocity": 0, "consensus": False,
        "corroborated": False, "signals": [], "top_title": None,
    })

    # --- overnight gaps ---
    gaps_data = _load_json(OVERNIGHT_GAPS)
    if isinstance(gaps_data, dict):
        for item in gaps_data.get("gap_candidates", []):
            t = item.get("ticker", "")
            if not t:
                continue
            scores[t]["gap_score"] = max(scores[t]["gap_score"], item.get("max_ml_score") or 0.0)
            scores[t]["signals"].append(f"gap:{item.get('article_count',0)}arts")
            if not scores[t]["top_title"]:
                arts = item.get("top_articles", [])
                if arts:
                    scores[t]["top_title"] = arts[0].get("title", "")[:80]

    # --- trend velocity (recent 2h surge) ---
    vel_data = _load_json(TREND_VELOCITY)
    if isinstance(vel_data, dict):
        for item in vel_data.get("top", []):
            t = item.get("ticker", "")
            if not t:
                continue
            scores[t]["velocity"] = item.get("delta", 0)
            scores[t]["signals"].append(f"vel:+{item.get('delta',0)}")

    # --- strong consensus (last 12h) ---
    cutoff = now - timedelta(hours=CONSENSUS_WINDOW_H)
    try:
        for line in CONSENSUS_JSONL.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            ts = _parse_iso(ev.get("detected_at", ""))
            if ts and ts >= cutoff:
                t = ev.get("ticker", "")
                if t:
                    scores[t]["consensus"] = True
                    scores[t]["signals"].append(f"consensus:{ev.get('count',0)}arts")
                    if not scores[t]["top_title"]:
                        scores[t]["top_title"] = ev.get("sample_title", "")[:80]
    except Exception:
        pass

    # --- corroboration digest (high-signal stories) ---
    corr_data = _load_json(CORROBORATION)
    if isinstance(corr_data, dict):
        for story in corr_data.get("stories", []):
            if not (story.get("max_urgency", 0) >= 2 or story.get("max_ai_score", 0) >= 5):
                continue
            tickers = _tickers_in_text(story.get("title", ""))
            for t in tickers:
                if t in scores:
                    scores[t]["corroborated"] = True
                    scores[t]["signals"].append(f"corr:{story.get('source_count',0)}src")

    # --- composite score + rank ---
    ranked = []
    for ticker, s in scores.items():
        composite = (
            0.4 * s["gap_score"]
            + 0.3 * s["velocity"]
            + 15.0 * int(s["consensus"])
            + 10.0 * int(s["corroborated"])
        )
        ranked.append({
            "ticker": ticker,
            "composite": round(composite, 2),
            "gap_score": round(s["gap_score"], 2),
            "velocity": s["velocity"],
            "consensus": s["consensus"],
            "corroborated": s["corroborated"],
            "signals": sorted(set(s["signals"])),
            "top_title": s["top_title"],
        })

    ranked.sort(key=lambda x: x["composite"], reverse=True)
    top = ranked[:TOP_N]

    result = {
        "generated_at": now.isoformat(),
        "sources_used": {
            "overnight_gaps":    OVERNIGHT_GAPS.exists(),
            "trend_velocity":    TREND_VELOCITY.exists(),
            "consensus_jsonl":   CONSENSUS_JSONL.exists(),
            "corroboration":     CORROBORATION.exists(),
        },
        "total_tickers": len(ranked),
        "top": top,
    }
    OUT_PATH.write_text(json.dumps(result, indent=2))
    return result


def main():
    result = compute()
    print(f"premarket_brief: {result['total_tickers']} tickers → top {len(result['top'])}")
    for i, t in enumerate(result["top"], 1):
        sigs = ",".join(t["signals"])
        print(f"  {i:2d}. {t['ticker']:6s} score={t['composite']:5.1f} [{sigs}]")


if __name__ == "__main__":
    main()

"""Corroboration digest: multi-source story clusters saved to a JSON log.

Runs hourly. Pulls the last 3h of articles, clusters near-duplicate headlines
using the storage.story_corroboration module, and writes the top stories
(corroborated by >=2 distinct sources) to /home/zeph/logs/corroboration_digest.json.

Stories with max_urgency>=2 or max_ai_score>=5 are flagged as high-signal.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from storage.story_corroboration import corroborated_breaking  # noqa: E402

OUT_PATH = Path("/home/zeph/logs/corroboration_digest.json")
HOURS = 3.0
MIN_SOURCES = 2
TOP_N = 20


def run() -> dict:
    now = datetime.now(timezone.utc)
    stories = corroborated_breaking(hours=HOURS, min_sources=MIN_SOURCES)

    top = stories[:TOP_N]
    high_signal = [s for s in top if s.get("max_urgency", 0) >= 2 or s.get("max_ai_score", 0) >= 5]

    out = {
        "generated_at": now.isoformat(),
        "window_hours": HOURS,
        "total_corroborated": len(stories),
        "high_signal_count": len(high_signal),
        "stories": [
            {
                "title": s["title"],
                "source_count": s["source_count"],
                "domain_count": s["domain_count"],
                "article_count": s["article_count"],
                "span_minutes": s["span_minutes"],
                "first_seen": s["first_seen"],
                "max_ai_score": s.get("max_ai_score", 0),
                "max_urgency": s.get("max_urgency", 0),
                "sources": s["sources"][:6],
                "high_signal": s.get("max_urgency", 0) >= 2 or s.get("max_ai_score", 0) >= 5,
            }
            for s in top
        ],
    }

    OUT_PATH.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    result = run()
    print(f"Corroboration digest: {result['total_corroborated']} stories | "
          f"high-signal: {result['high_signal_count']}")
    for s in result["stories"][:5]:
        flag = " [HIGH]" if s["high_signal"] else ""
        print(f"  [{s['source_count']}src/{s['domain_count']}dom in {s['span_minutes']}m "
              f"| ai={s['max_ai_score']:.1f}]{flag} {s['title'][:80]}")

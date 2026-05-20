#!/usr/bin/env python3
"""Detect articles where ml_score and ai_score strongly disagree.

Surfaces probable model/scorer divergence: articles the ML model rates very
differently from the AI scorer. Read-only; output JSON to
/home/zeph/logs/score_divergence.json.
"""
import sqlite3
import json
import datetime
from datetime import timezone

DB = '/home/zeph/digital-intern/data/articles.db'
OUT = '/home/zeph/logs/score_divergence.json'
WINDOW_HOURS = 24
TOP_N = 20
MIN_GAP = 0.30  # absolute |ml_score - ai_score| threshold


def main():
    cutoff = (datetime.datetime.now(timezone.utc) - datetime.timedelta(hours=WINDOW_HOURS))
    cutoff_s = cutoff.isoformat().replace('+00:00', '')
    c = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    c.execute('PRAGMA busy_timeout=4000')
    rows = c.execute(
        """
        SELECT id, title, source, ai_score, ml_score, urgency, first_seen
        FROM articles
        WHERE ai_score IS NOT NULL AND ml_score IS NOT NULL
          AND replace(first_seen,'T',' ') >= ?
        ORDER BY rowid DESC
        LIMIT 5000
        """,
        (cutoff_s.replace('T', ' '),),
    ).fetchall()
    c.close()

    diverg = []
    for r in rows:
        ai = r[3] or 0.0
        ml = r[4] or 0.0
        gap = abs(ai - ml)
        if gap >= MIN_GAP:
            diverg.append({
                'id': r[0],
                'title': (r[1] or '')[:160],
                'source': r[2],
                'ai_score': round(ai, 3),
                'ml_score': round(ml, 3),
                'gap': round(gap, 3),
                'direction': 'ml_higher' if ml > ai else 'ai_higher',
                'urgency': r[5],
                'first_seen': r[6],
            })

    diverg.sort(key=lambda x: x['gap'], reverse=True)
    top = diverg[:TOP_N]

    summary = {
        'generated_at': datetime.datetime.now(timezone.utc).isoformat(),
        'window_hours': WINDOW_HOURS,
        'sampled': len(rows),
        'divergent_count': len(diverg),
        'min_gap_threshold': MIN_GAP,
        'avg_gap': round(sum(d['gap'] for d in diverg) / len(diverg), 3) if diverg else 0.0,
        'ml_higher_pct': round(100.0 * sum(1 for d in diverg if d['direction'] == 'ml_higher') / len(diverg), 1) if diverg else 0.0,
        'top': top,
    }

    with open(OUT, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"sampled={summary['sampled']} divergent={summary['divergent_count']} "
          f"avg_gap={summary['avg_gap']} ml_higher_pct={summary['ml_higher_pct']}%")
    for d in top[:5]:
        print(f"  gap={d['gap']:.2f} {d['direction']:10s} ai={d['ai_score']:.2f} ml={d['ml_score']:.2f} | {d['source']} | {d['title'][:80]}")


if __name__ == '__main__':
    main()

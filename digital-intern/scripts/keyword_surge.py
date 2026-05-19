#!/usr/bin/env python3
"""Detect surging title bigrams: high frequency in last 1h vs prior 23h baseline.

Reads articles.db (read-only), tokenises titles, computes bigram frequencies in
the recent (1h) and baseline (1h..24h) windows, and emits the top novel/spiking
bigrams as JSON to /home/zeph/logs/keyword_surge.json.

Surge score: (now_per_hr + 1) / (baseline_per_hr + 1), with a floor on `now`.
"""
import sqlite3
import json
import re
import datetime
from collections import Counter

DB = '/home/zeph/digital-intern/data/articles.db'
OUT = '/home/zeph/logs/keyword_surge.json'
STOP = {
    'the','a','an','and','or','but','of','in','on','at','to','for','with','by',
    'from','as','is','are','was','were','be','been','being','it','its','this',
    'that','these','those','will','can','may','could','should','would','has',
    'have','had','do','does','did','not','no','vs','over','under','more','than',
    'after','before','up','down','out','about','new','says','said','say','amid',
    'into','off','top','best','worst','here','now','today','vs.','—','-','&',
    'us','u.s.','u.s','&amp;','what','why','how','when','who','which','his','her',
    'their','its','they','them','our','your','my','if','so','just','only','any',
    'all','one','two','three','first','last','next','still','also',
}
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9.&'+-]{1,}")

def tokenise(title: str):
    if not title:
        return []
    return [t.lower() for t in TOKEN_RE.findall(title) if t.lower() not in STOP and len(t) > 2]

def bigrams(toks):
    return [f"{toks[i]} {toks[i+1]}" for i in range(len(toks) - 1)]

def main():
    con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    con.execute('PRAGMA busy_timeout=8000')
    cutoff_recent = (datetime.datetime.utcnow() - datetime.timedelta(hours=1)).isoformat() + 'Z'
    cutoff_base = (datetime.datetime.utcnow() - datetime.timedelta(hours=24)).isoformat() + 'Z'
    rows = con.execute(
        """
        SELECT first_seen, title FROM articles INDEXED BY idx_first_seen
        WHERE first_seen >= ?
        ORDER BY first_seen DESC LIMIT 4000
        """,
        (cutoff_base,),
    ).fetchall()
    now_c, base_c = Counter(), Counter()
    n_now = n_base = 0
    for fs, title in rows:
        bg = bigrams(tokenise(title))
        if fs >= cutoff_recent:
            now_c.update(bg)
            n_now += 1
        else:
            base_c.update(bg)
            n_base += 1
    # Score: prefer bigrams with >=3 mentions now and a clear lift.
    scored = []
    for bg, n in now_c.items():
        if n < 3:
            continue
        base = base_c.get(bg, 0)
        # baseline rate per 1h (23 baseline hours)
        base_per_hr = base / 23.0
        lift = (n + 1) / (base_per_hr + 1)
        scored.append({
            'bigram': bg,
            'now_count': n,
            'baseline_23h_count': base,
            'lift': round(lift, 2),
        })
    scored.sort(key=lambda x: (-x['lift'], -x['now_count']))
    out = {
        'generated_at': datetime.datetime.utcnow().isoformat() + 'Z',
        'articles_scanned': len(rows),
        'articles_recent_1h': n_now,
        'articles_baseline_23h': n_base,
        'top_surging': scored[:20],
    }
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"scanned={len(rows)} recent_1h={n_now} baseline_23h={n_base} -> {OUT}")
    for r in scored[:5]:
        print(f"  {r['bigram']!r}: now={r['now_count']} base={r['baseline_23h_count']} lift={r['lift']}x")
    if not scored:
        print("  (no bigrams crossed the now>=3 threshold)")

if __name__ == '__main__':
    main()

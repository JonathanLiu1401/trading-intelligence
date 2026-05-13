"""
Urgency scorer — batches unscored articles to Claude Sonnet 4.6.
Marks articles as urgent (score >= 8) for immediate alerting.
"""
import json
import re
import shutil
import subprocess

SONNET_MODEL = "claude-sonnet-4-6"
BATCH_SIZE = 100  # articles per Sonnet call
URGENT_THRESHOLD = 8.0

SCORE_PROMPT = """You are a real-time financial news urgency classifier. Score each article 0-10.

URGENT (8-10): Breaking news that will move markets NOW:
- Earnings beats/misses for: LITE, MU, MSFT, AXTI, ORCL, TSEM, QBTS, NVDA, AMD, TSM, SK Hynix, Samsung, Micron
- Fed surprise decisions, emergency rate changes
- Major analyst upgrades/downgrades (PT change >15%) on tracked stocks
- Memory/DRAM pricing shock (ASP move >5%)
- China/US escalation: new chip export bans, tariffs on semis
- Market circuit breakers, flash crashes
- Major geopolitical event (military action, sanctions) affecting tech supply chain
- HBM or DRAM supply disruption

RELEVANT (5-7): Important but not immediately actionable:
- General sector analysis, macro outlook
- Fed commentary, economic data (CPI, jobs, GDP)
- Crypto market moves, general tech news
- Analyst commentary without rating change

NOISE (0-4): Not relevant to this portfolio

Articles:
{articles_json}

Respond ONLY with a JSON array: [{{"index": 0, "score": 9, "reason": "MU earnings beat"}}, ...]"""


def score_batch(articles: list, store) -> int:
    """Score a batch of articles; update store. Returns count of urgent items found."""
    if not articles or not shutil.which("claude"):
        return 0

    payload = [
        {"index": i, "title": a.get("title", "")[:200], "summary": (a.get("summary") or "")[:300]}
        for i, a in enumerate(articles)
    ]

    prompt = SCORE_PROMPT.format(articles_json=json.dumps(payload, ensure_ascii=False))

    try:
        result = subprocess.run(
            ["claude", "--model", SONNET_MODEL, "--print",
             "--permission-mode", "bypassPermissions", prompt],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            return 0

        m = re.search(r"\[.*\]", result.stdout, re.DOTALL)
        if not m:
            return 0

        scores = json.loads(m.group(0))
        urgent_count = 0
        for item in scores:
            idx = item.get("index")
            score = float(item.get("score", 0))
            if idx is None or idx >= len(articles):
                continue
            art = articles[idx]
            aid = art.get("_id")
            if not aid:
                continue
            is_urgent = score >= URGENT_THRESHOLD
            store.update_ai_score(aid, score, urgency=1 if is_urgent else 0)
            if is_urgent:
                reason = item.get("reason", "")
                print(f"[urgency] URGENT score={score:.0f} — {art.get('title', '')[:80]} ({reason})")
                urgent_count += 1

        return urgent_count
    except Exception as e:
        print(f"[urgency] Scoring error: {e}")
        return 0


def run_scoring_pass(store, batch_size: int = BATCH_SIZE) -> int:
    """Score all unscored articles in the store. Returns total urgent found."""
    unscored = store.get_unscored(limit=batch_size * 5)
    if not unscored:
        return 0

    print(f"[urgency] Scoring {len(unscored)} unscored articles in batches of {batch_size}...")
    total_urgent = 0
    for i in range(0, len(unscored), batch_size):
        batch = unscored[i:i + batch_size]
        total_urgent += score_batch(batch, store)

    return total_urgent

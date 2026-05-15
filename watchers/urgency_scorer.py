"""
Urgency scorer — batches unscored articles to Claude Sonnet 4.6.
Marks articles as urgent (score >= 8) for immediate alerting.
"""
import json
import logging

from core.claude_cli import claude_call

try:
    from core.logger import get_logger
    _log = get_logger("urgency_scorer")
except Exception:
    _log = logging.getLogger("urgency_scorer")

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


def _extract_json_array(raw: str):
    """Robustly extract a top-level JSON array from a Sonnet response that may
    be wrapped in prose. Tries json.JSONDecoder().raw_decode at each '[' until
    one parses successfully. Returns the parsed list, or None on failure."""
    decoder = json.JSONDecoder()
    start = raw.find("[")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(raw[start:])
            if isinstance(value, list):
                return value
        except ValueError:
            pass
        start = raw.find("[", start + 1)
    return None


def score_batch(articles: list, store) -> int:
    """Score a batch of articles; update store. Returns count of urgent items found."""
    if not articles:
        return 0

    payload = [
        {"index": i, "title": a.get("title", "")[:200], "summary": (a.get("summary") or "")[:300]}
        for i, a in enumerate(articles)
    ]

    prompt = SCORE_PROMPT.format(articles_json=json.dumps(payload, ensure_ascii=False))

    try:
        raw = claude_call(prompt, model=SONNET_MODEL, timeout=120)
        if raw is None:
            return 0

        scores = _extract_json_array(raw)
        if scores is None:
            _log.warning(f"[urgency] Failed to parse JSON array from response: {raw[:200]!r}")
            return 0

        urgent_count = 0
        updates: list[tuple[str, float, int]] = []
        urgent_log: list[tuple[float, str, str]] = []
        scored_indices: set[int] = set()
        for item in scores:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("index"))
                score = float(item.get("score", 0))
            except (TypeError, ValueError):
                continue
            if idx < 0 or idx >= len(articles):
                continue
            art = articles[idx]
            aid = art.get("_id")
            if not aid:
                continue
            is_urgent = score >= URGENT_THRESHOLD
            # Floor non-urgent ai_score at 0.01 so Sonnet-scored noise (score=0)
            # isn't picked up by get_unscored on the next pass and re-sent to
            # the LLM forever. Clamp the upper bound to 10 defensively.
            score = max(0.01, min(float(score), 10.0))
            updates.append((aid, score, 1 if is_urgent else 0))
            scored_indices.add(idx)
            if is_urgent:
                urgent_log.append((score, art.get("title", "")[:80], item.get("reason", "")))
                urgent_count += 1
        # Anti-loop: articles Sonnet engaged with but omitted from its response
        # would otherwise remain ai_score=0 / ml_score=NULL and be re-routed to
        # the LLM every cycle forever. Floor them to 0.01 (noise) so they exit
        # the queue. Only apply when Sonnet returned at least one valid entry —
        # a completely empty array is ambiguous (refusal vs. true zero) and we
        # prefer a retry over mass-mislabeling 100 articles as noise.
        if scored_indices:
            for i, art in enumerate(articles):
                if i in scored_indices:
                    continue
                aid = art.get("_id")
                if not aid:
                    continue
                updates.append((aid, 0.01, 0))
        if updates:
            # One bulk commit instead of N round-trips through the write lock.
            store.update_ai_scores_batch(updates)
        for score, title, reason in urgent_log:
            _log.info(f"[urgency] URGENT score={score:.0f} — {title} ({reason})")

        return urgent_count
    except Exception:
        _log.exception("[urgency] Scoring error")
        return 0


def run_scoring_pass(store, batch_size: int = BATCH_SIZE) -> int:
    """Score all unscored articles in the store. Returns total urgent found."""
    unscored = store.get_unscored(limit=batch_size * 5)
    if not unscored:
        return 0

    _log.info(f"[urgency] Scoring {len(unscored)} unscored articles in batches of {batch_size}...")
    total_urgent = 0
    for i in range(0, len(unscored), batch_size):
        batch = unscored[i:i + batch_size]
        total_urgent += score_batch(batch, store)

    return total_urgent

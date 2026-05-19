"""
Urgency scorer — batches unscored articles to Claude Sonnet 4.6.
Marks articles as urgent (score >= 8) for immediate alerting.
"""
import json
import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from core.claude_cli import claude_call
from core.json_extract import extract_json_array

try:
    from core.logger import get_logger
    _log = get_logger("urgency_scorer")
except Exception:
    _log = logging.getLogger("urgency_scorer")

SONNET_MODEL = "claude-sonnet-4-6"
BATCH_SIZE = 100  # articles per Sonnet call
URGENT_THRESHOLD = 8.0
# Hard cap kicks in only for clearly stale articles (>48h).
# Timeless content (structural supply shifts, filings) can still be urgent within this window.
# The prompt instructs Claude to apply tighter judgment at >24h for time-sensitive language.
STALE_HOURS = 48.0
STALE_SCORE_CAP = 5.9  # max score for hard-capped stale articles (below urgent threshold of 8)
# When Sonnet hits its output-token limit the array comes back truncated and
# core.json_extract salvages only a leading prefix of indices. The contract
# there (and the existing empty-array guard below) is that the *caller*
# re-queues whatever is missing rather than burying it. Truncation has a
# distinct fingerprint: the returned indices are a clean prefix 0..k and the
# entire tail k+1..N-1 is absent. A model deliberately skipping an article it
# found ambiguous drops a handful at most; a long trailing block this size is
# truncation, so those articles are re-queued (left ai_score=0) instead of
# floored to noise.
MAX_TRAILING_OMISSIONS = 5

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

STALENESS RULE: Each article includes age_hours. Use judgment:
- If the article uses time-sensitive language ("today", "breaking", "just announced", "this week", \
"yesterday") AND age_hours > 24 → the temporal claim is stale; cap your score at 5.
- If the content is fundamentally timeless (structural supply-chain shift, regulatory ruling, \
major SEC filing, long-term analyst thesis on tracked stocks) → score on merit regardless of age.
- If age_hours > 72 and urgency depends entirely on the event being recent → score ≤ 3.

Articles:
{articles_json}

Respond ONLY with a JSON array: [{{"index": 0, "score": 9, "reason": "MU earnings beat"}}, ...]"""


def _article_age_hours(article: dict) -> float:
    """Hours since the article was published — ``published`` preferred, else
    ``first_seen``. ``0.0`` when neither field parses (treat as fresh — safe
    default; staleness cap is a downward bound).

    Cascading fallback: a non-empty-but-unparseable ``published`` must NOT
    short-circuit the lookup at 0.0h, because that bypasses the
    ``STALE_HOURS`` cap on a row whose ``first_seen`` is genuinely old (live
    failure: an unparseable RFC822 date on a 40h-old article was treated as
    fresh, letting Sonnet's "urgent=9" through the staleness clamp). Mirrors
    ``alert_agent._article_age_hours``'s field-cascade convention so the two
    age helpers agree on which timestamp is authoritative."""
    now = datetime.now(timezone.utc)
    for field in ("published", "first_seen"):
        raw = (article.get(field) or "").strip()
        if not raw:
            continue
        dt = None
        try:
            dt = parsedate_to_datetime(raw)
        except Exception:
            dt = None
        if dt is None:
            try:
                dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except Exception:
                dt = None
        if dt is None:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (now - dt).total_seconds() / 3600.0)
    return 0.0


def score_batch(articles: list, store) -> int:
    """Score a batch of articles; update store. Returns count of urgent items found."""
    if not articles:
        return 0

    age_hours_map: dict[int, float] = {i: _article_age_hours(a) for i, a in enumerate(articles)}
    payload = [
        {
            "index": i,
            "age_hours": round(age_hours_map[i], 1),
            "title": (a.get("title") or "")[:200],
            "summary": (a.get("summary") or "")[:300],
        }
        for i, a in enumerate(articles)
    ]

    prompt = SCORE_PROMPT.format(articles_json=json.dumps(payload, ensure_ascii=False))

    try:
        raw = claude_call(prompt, model=SONNET_MODEL, timeout=120)
        if raw is None:
            return 0

        scores = extract_json_array(raw)
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
            # Hard staleness cap: articles older than STALE_HOURS cannot be urgent
            # regardless of time-sensitive language in their text ("today", "breaking").
            age_h = age_hours_map.get(idx, 0.0)
            if age_h > STALE_HOURS and score > STALE_SCORE_CAP:
                _log.debug(f"[urgency] Capping stale article (age={age_h:.1f}h) score {score:.1f}→{STALE_SCORE_CAP}")
                score = STALE_SCORE_CAP
            is_urgent = score >= URGENT_THRESHOLD
            # Floor non-urgent ai_score at 0.01 so Sonnet-scored noise (score=0)
            # isn't picked up by get_unscored on the next pass and re-sent to
            # the LLM forever. Clamp the upper bound to 10 defensively.
            score = max(0.01, min(float(score), 10.0))
            updates.append((aid, score, 1 if is_urgent else 0))
            scored_indices.add(idx)
            if is_urgent:
                urgent_log.append((score, (art.get("title") or "")[:80], item.get("reason", "")))
                urgent_count += 1
        # Anti-loop: articles Sonnet engaged with but omitted from its response
        # would otherwise remain ai_score=0 / ml_score=NULL and be re-routed to
        # the LLM every cycle forever. Floor them to 0.01 (noise) so they exit
        # the queue. Only apply when Sonnet returned at least one valid entry —
        # a completely empty array is ambiguous (refusal vs. true zero) and we
        # prefer a retry over mass-mislabeling 100 articles as noise.
        # Truncation guard: if the response is a clean prefix (indices
        # 0..max_idx with no internal gaps) and the missing tail is longer
        # than a model would plausibly omit on purpose, treat it as a
        # token-limit truncation and re-queue the tail (leave ai_score=0)
        # instead of flooring genuine articles to noise forever.
        truncated_tail = False
        if scored_indices:
            max_idx = max(scored_indices)
            is_clean_prefix = scored_indices == set(range(max_idx + 1))
            tail_len = len(articles) - (max_idx + 1)
            if is_clean_prefix and tail_len > MAX_TRAILING_OMISSIONS:
                truncated_tail = True
                _log.warning(
                    f"[urgency] Response looks truncated: got clean prefix "
                    f"0..{max_idx} of {len(articles)}, re-queuing {tail_len} "
                    f"tail articles instead of flooring them to noise"
                )
        if scored_indices and not truncated_tail:
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

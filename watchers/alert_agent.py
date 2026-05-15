"""
Urgent alert agent — Bloomberg BN newswire style, immediate Discord post.
"""
import logging
import os
from datetime import datetime, timezone

from core.claude_cli import claude_call
from watchers.alert_dedup import alerted_ids, dedupe_urgent

try:
    from core.logger import get_logger
    _log = get_logger("alert_agent")
except Exception:
    _log = logging.getLogger("alert_agent")

SONNET_MODEL = "claude-sonnet-4-6"
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")

ALERT_PROMPT = """You are a Bloomberg BN terminal newswire alert system. A high-urgency financial event has been detected.

Write a Discord alert in Bloomberg newswire style — dense, exact, no filler. Max 1800 chars.

Current UTC time (use this verbatim in the timestamp slot — do NOT guess): {now_utc}

FORMAT (use exactly):
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚨 BREAKING  ◈  [CATEGORY]  ◈  {now_utc} UTC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[ONE LINE HEADLINE IN CAPS — what happened]

TICKERS:   [affected symbols]
IMPACT:    [BUY/SELL/WATCH] — [one sentence on direction]
CONTEXT:   [one sentence of background]
PORTFOLIO: [specific implication for LITE/MU/MSFT/AXTI/ORCL/TSEM/QBTS]
SOURCE:    [source name]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
Then on a new line after the code block: [article url]

Categories: EARNINGS | RATING CHANGE | MACRO SHOCK | SUPPLY CHAIN | REGULATORY | FED | CRYPTO | M&A | GEOPOLITICAL

Urgent articles detected:
{articles_text}

Output ONLY the alert message."""


ALERT_BATCH_SIZE = 5


def send_urgent_alert(urgent_articles: list, store) -> bool:
    if not urgent_articles:
        return False
    if not DISCORD_WEBHOOK:
        _log.warning("[alert] No DISCORD_WEBHOOK_URL — skipping")
        return False

    # Collapse syndicated duplicates first: one breaking story carried by GDELT
    # + Reuters + Yahoo + RSS would otherwise eat the whole 5-slot batch and
    # show the trader the same event five times. After dedup the batch holds
    # five DISTINCT stories; each survivor knows the ids of the copies it
    # absorbed (``_dup_ids``) so all of them can still be marked alerted.
    deduped = dedupe_urgent(urgent_articles)

    # Only the first ALERT_BATCH_SIZE feed the prompt — and only those (plus the
    # duplicates they absorbed) get marked alerted. Marking the entire urgent
    # list would silently drop the tail (it'd never be picked up next cycle),
    # so we cap both ends.
    batch = deduped[:ALERT_BATCH_SIZE]

    def _fmt(a: dict) -> str:
        # Include the article body when available so the alert LLM grounds its
        # CONTEXT line on real content rather than guessing from the headline.
        block = (
            f"[score={a['ai_score']:.0f}] {a['title']}\n"
            f"source: {a['source']}\nurl: {a['link']}"
        )
        dup_count = int(a.get("dup_count") or 1)
        if dup_count > 1:
            # Tell the alert LLM how broadly the story is being carried — wide
            # syndication is itself a signal of how big the event is.
            block += f"\nsyndication: reported by {dup_count} sources"
        summary = (a.get("summary") or "").strip()
        if summary:
            block += f"\nbody: {summary[:600]}"
        return block

    articles_text = "\n\n".join(_fmt(a) for a in batch)

    # Full date+time so Discord history is unambiguous across day boundaries.
    # Template already appends " UTC", so don't include it here.
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    prompt = ALERT_PROMPT.format(articles_text=articles_text, now_utc=now_utc)

    try:
        message = claude_call(prompt, model=SONNET_MODEL, timeout=60)
        if not message:
            _log.warning("[alert] No response from Claude — skipping")
            return False

        # post via discord_notifier which also fires TTS
        from notifier.discord_notifier import send as discord_send
        ok = discord_send(message, is_alert=True)

        if ok:
            # Bulk-mark in one transaction; previous code took the write lock
            # N times (5 round-trips for the default batch size). alerted_ids
            # includes the syndicated copies merged into the batch, so they
            # never re-fire — duplicates of still-queued stories stay urgent.
            mark_ids = alerted_ids(batch)
            store.mark_alerted_batch(mark_ids)
            collapsed = len(mark_ids) - len(batch)
            tail = len(deduped) - len(batch)
            notes = []
            if collapsed > 0:
                notes.append(f"{collapsed} syndicated dupes folded in")
            if tail > 0:
                notes.append(f"{tail} more queued")
            note = f" ({'; '.join(notes)})" if notes else ""
            _log.info(f"[alert] BN alert sent ({len(batch)} distinct stories){note}")
        else:
            _log.warning("[alert] Discord POST failed")
        return ok

    except Exception:
        _log.exception("[alert] Error sending urgent alert")
        return False

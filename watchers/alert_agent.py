"""
Urgent alert agent — Bloomberg BN newswire style, immediate Discord post.
"""
import os
from datetime import datetime, timezone

from core.claude_cli import claude_call

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
        print("[alert] No DISCORD_WEBHOOK_URL — skipping")
        return False

    # Only the first ALERT_BATCH_SIZE feed the prompt — and only those get
    # marked alerted. Marking the entire urgent list would silently drop the
    # tail (it'd never be picked up on the next cycle), so we cap both ends.
    batch = urgent_articles[:ALERT_BATCH_SIZE]
    articles_text = "\n".join(
        f"[score={a['ai_score']:.0f}] {a['title']}\nsource: {a['source']}\nurl: {a['link']}"
        for a in batch
    )

    # Full date+time so Discord history is unambiguous across day boundaries.
    # Template already appends " UTC", so don't include it here.
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    prompt = ALERT_PROMPT.format(articles_text=articles_text, now_utc=now_utc)

    try:
        message = claude_call(prompt, model=SONNET_MODEL, timeout=60)
        if not message:
            print("[alert] No response from Claude — skipping")
            return False

        # post via discord_notifier which also fires TTS
        from notifier.discord_notifier import send as discord_send
        ok = discord_send(message, is_alert=True)

        if ok:
            for art in batch:
                store.mark_alerted(art["_id"])
            tail = len(urgent_articles) - len(batch)
            tail_note = f" ({tail} more queued)" if tail > 0 else ""
            print(f"[alert] BN alert sent ({len(batch)} articles){tail_note}")
        else:
            print(f"[alert] Discord POST failed")
        return ok

    except Exception as e:
        print(f"[alert] Error: {e}")
        return False

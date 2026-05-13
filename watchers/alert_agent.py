"""
Urgent alert agent — Bloomberg BN newswire style, immediate Discord post.
"""
import os

from core.claude_cli import claude_call

SONNET_MODEL = "claude-sonnet-4-6"
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")

ALERT_PROMPT = """You are a Bloomberg BN terminal newswire alert system. A high-urgency financial event has been detected.

Write a Discord alert in Bloomberg newswire style — dense, exact, no filler. Max 1800 chars.

FORMAT (use exactly):
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚨 BREAKING  ◈  [CATEGORY]  ◈  [HH:MM UTC]
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


def send_urgent_alert(urgent_articles: list, store) -> bool:
    if not urgent_articles:
        return False
    if not DISCORD_WEBHOOK:
        print("[alert] No DISCORD_WEBHOOK_URL — skipping")
        return False

    articles_text = "\n".join(
        f"[score={a['ai_score']:.0f}] {a['title']}\nsource: {a['source']}\nurl: {a['link']}"
        for a in urgent_articles[:5]
    )

    prompt = ALERT_PROMPT.format(articles_text=articles_text)

    try:
        message = claude_call(prompt, model=SONNET_MODEL, timeout=60)
        if not message:
            print("[alert] No response from Claude — skipping")
            return False

        # post via discord_notifier which also fires TTS
        from notifier.discord_notifier import send as discord_send
        ok = discord_send(message, is_alert=True)

        if ok:
            for art in urgent_articles:
                store.mark_alerted(art["_id"])
            print(f"[alert] BN alert sent ({len(urgent_articles)} articles)")
        else:
            print(f"[alert] Discord POST failed")
        return ok

    except Exception as e:
        print(f"[alert] Error: {e}")
        return False

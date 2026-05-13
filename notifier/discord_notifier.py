"""Discord webhook notifier with message chunking + TTS."""
import os
import time
import requests

DISCORD_LIMIT = 2000


def _chunk(text: str, limit: int = DISCORD_LIMIT):
    chunks = []
    remaining = text
    while len(remaining) > limit:
        split_at = remaining.rfind("\n", 0, limit)
        if split_at <= 0:
            split_at = limit
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip("\n")
    if remaining:
        chunks.append(remaining)
    return chunks


def send(message: str, is_alert: bool = False) -> bool:
    """Send message to Discord; splits if > 2000 chars. Also fires TTS. Returns True on success."""
    webhook = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook:
        print("[discord_notifier] DISCORD_WEBHOOK_URL not set — skipping.")
        return False

    chunks = _chunk(message)
    ok = True
    for i, chunk in enumerate(chunks):
        try:
            r = requests.post(webhook, json={"content": chunk}, timeout=15)
            if r.status_code not in (200, 204):
                print(f"[discord_notifier] HTTP {r.status_code}: {r.text[:200]}")
                ok = False
            if i < len(chunks) - 1:
                time.sleep(0.5)
        except Exception as e:
            print(f"[discord_notifier] Send error: {e}")
            ok = False

    # Fire TTS in a background thread so it doesn't block the pipeline
    if ok:
        from notifier.tts import speak_async
        speak_async(message, is_alert=is_alert)

    return ok


if __name__ == "__main__":
    send("Digital Intern test message.")

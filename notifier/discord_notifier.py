"""Discord webhook notifier with message chunking + TTS."""
import os
import time
import requests

DISCORD_LIMIT = 2000


def _chunk(text: str, limit: int = DISCORD_LIMIT):
    """Split text into Discord-sized chunks.

    Prefer splitting on newline, fall back to whitespace so we don't tear
    a URL or word in half when long alert bodies have no line breaks.
    """
    chunks = []
    remaining = text
    while len(remaining) > limit:
        split_at = remaining.rfind("\n", 0, limit)
        if split_at <= 0:
            split_at = remaining.rfind(" ", 0, limit)
        if split_at <= 0:
            split_at = limit
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip("\n ")
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
        sent = False
        for attempt in range(4):
            try:
                r = requests.post(webhook, json={"content": chunk}, timeout=15)
                if r.status_code == 429:
                    # Honor Discord rate limit: prefer JSON retry_after, fall back to header, then exponential backoff.
                    retry_after = 1.0
                    try:
                        retry_after = float(r.json().get("retry_after", retry_after))
                    except Exception:
                        retry_after = float(r.headers.get("Retry-After", retry_after))
                    retry_after = min(max(retry_after, 0.5), 30.0)
                    print(f"[discord_notifier] 429 rate-limited, sleeping {retry_after:.2f}s (attempt {attempt + 1})")
                    time.sleep(retry_after)
                    continue
                if r.status_code not in (200, 204):
                    print(f"[discord_notifier] HTTP {r.status_code}: {r.text[:200]}")
                    if 500 <= r.status_code < 600 and attempt < 3:
                        time.sleep(2 ** attempt)
                        continue
                    ok = False
                sent = True
                break
            except requests.RequestException as e:
                print(f"[discord_notifier] Send error (attempt {attempt + 1}): {e}")
                if attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                ok = False
                break
        if not sent and ok:
            ok = False
        if i < len(chunks) - 1:
            time.sleep(0.5)

    # Fire TTS in a background thread so it doesn't block the pipeline
    if ok:
        from notifier.tts import speak_async
        speak_async(message, is_alert=is_alert)

    return ok


if __name__ == "__main__":
    send("Digital Intern test message.")

"""Discord webhook notifier with message chunking + TTS."""
import os
import time
import requests

DISCORD_LIMIT = 2000

# Discord rejects requests from some default library User-Agents (it has
# historically returned HTTP 403 for bare urllib/python-requests UAs as part
# of its anti-abuse filtering). Sending an explicit, descriptive UA — the
# format Discord documents for bots/webhooks — keeps this resilient even if
# the bare `python-requests/X.Y` UA gets filtered in the future.
_HEADERS = {
    "User-Agent": "DigitalIntern-Notifier/1.0 (+https://github.com/openclaw/digital-intern)"
}


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
    # An empty or whitespace-only body is a no-op: Discord rejects empty
    # "content" with HTTP 400, and firing TTS on "" just wastes an API call
    # while the old code returned True (a silent false success).
    if not message or not message.strip():
        print("[discord_notifier] empty message — skipping.")
        return False

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
                r = requests.post(webhook, json={"content": chunk}, headers=_HEADERS, timeout=15)
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

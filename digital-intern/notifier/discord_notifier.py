"""Discord webhook notifier with message chunking + TTS."""
import os
import shutil
import subprocess
import time
import requests

DISCORD_LIMIT = 2000
DEFAULT_DISCORD_CHANNEL = "channel:1496099475838603324"

# Max POST attempts per chunk before the chunk is dropped. Used by the
# retry loop and the gave-up log line so the two never disagree.
_MAX_ATTEMPTS = 4

# Discord rejects requests from some default library User-Agents (it has
# historically returned HTTP 403 for bare urllib/python-requests UAs as part
# of its anti-abuse filtering). Sending an explicit, descriptive UA — the
# format Discord documents for bots/webhooks — keeps this resilient even if
# the bare `python-requests/X.Y` UA gets filtered in the future.
_HEADERS = {
    "User-Agent": "DigitalIntern-Notifier/1.0 (+https://github.com/openclaw/digital-intern)"
}


def _openclaw_cli_path() -> str | None:
    override = os.environ.get("OPENCLAW_CLI") or os.environ.get("OPENCLAW_CLI_PATH")
    if override:
        return override
    found = shutil.which("openclaw")
    if found:
        return found
    fallback = "/usr/local/bin/openclaw"
    return fallback if os.path.exists(fallback) else None


def _send_openclaw(chunk: str) -> bool:
    """Fallback delivery when no Discord webhook secret is configured."""
    cli = _openclaw_cli_path()
    if not cli:
        print("[discord_notifier] no webhook and openclaw CLI not found.")
        return False
    target = os.environ.get("DISCORD_ALERT_CHANNEL", DEFAULT_DISCORD_CHANNEL)
    env = os.environ.copy()
    env["PATH"] = (
        "/usr/local/bin:/opt/homebrew/bin:"
        f"{os.path.expanduser('~')}/.local/bin:"
        f"{os.path.expanduser('~')}/.npm-global/bin:"
        + env.get("PATH", "")
    )
    try:
        proc = subprocess.run(
            [
                cli,
                "message",
                "send",
                "--channel",
                "discord",
                "--target",
                target,
                "--message",
                chunk,
            ],
            capture_output=True,
            text=True,
            # OpenClaw CLI cold-loads the Discord plugin on each invoke; under
            # gateway load this regularly exceeds 30s even when delivery later
            # succeeds. 90s keeps urgent pings from false-failing.
            timeout=90,
            check=False,
            env=env,
        )
    except Exception as e:
        print(f"[discord_notifier] OpenClaw send error: {e}")
        return False
    if proc.returncode == 0:
        return True
    err = (proc.stderr or proc.stdout or "").strip()
    print(f"[discord_notifier] OpenClaw send failed rc={proc.returncode}: {err[:200]}")
    return False


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
    chunks = _chunk(message)
    ok = True
    for i, chunk in enumerate(chunks):
        sent = False
        if not webhook:
            sent = _send_openclaw(chunk)
            ok = ok and sent
        else:
            for attempt in range(_MAX_ATTEMPTS):
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
                        if 500 <= r.status_code < 600 and attempt < _MAX_ATTEMPTS - 1:
                            time.sleep(2 ** attempt)
                            continue
                        ok = False
                    sent = True
                    break
                except requests.RequestException as e:
                    print(f"[discord_notifier] Send error (attempt {attempt + 1}): {e}")
                    if attempt < _MAX_ATTEMPTS - 1:
                        time.sleep(2 ** attempt)
                        continue
                    ok = False
                    break
        if not sent:
            # Every retry attempt for this chunk was exhausted (persistent
            # 429 rate-limit storm, repeated 5xx, or connection errors).
            # Without this line the chunk is silently dropped — exactly the
            # failure an operator needs to see in journalctl during an
            # incident. Surface it explicitly and mark the send failed.
            print(f"[discord_notifier] gave up on chunk {i + 1}/{len(chunks)} "
                  f"after {_MAX_ATTEMPTS} attempts — chunk dropped")
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

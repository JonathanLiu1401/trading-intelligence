"""
Safe Claude CLI wrapper — pipes prompt via stdin to avoid ARG_MAX limits.

Usage:
    from core.claude_cli import claude_call
    output = claude_call(prompt, model="claude-sonnet-4-6", timeout=90)
    # Returns stdout string on success, None on failure.

Quota circuit breaker
---------------------
When the CLI reports an org usage / rate / quota limit, every subsequent
``claude_call`` would spawn a doomed subprocess that fails the same way. The
recursive labeler alone fires 25+ of these per 4h cycle, and urgency_scorer,
alert_agent, local_filter, tts and claude_analyst pile on more. Once a quota
error is seen we trip a module-level breaker and short-circuit (return None
without spawning) for ``QUOTA_COOLDOWN_S``. Every caller already treats None
as "Claude unavailable" and falls back gracefully, so the breaker only removes
wasted subprocess churn — it never turns a soft failure into a hard one. The
cooldown is deliberately short (1h) rather than "until next month": a wrong
short block self-heals on the next cycle, a wrong long block would silence
Claude for weeks if the limit string is ever misread or the limit resets on a
different cadence.
"""
import shutil
import subprocess
import time
from pathlib import Path

# Substrings (matched case-insensitively against the CLI's failure text) that
# mean "spawning again right now is pointless". Kept to the small set actually
# observed / documented — not a catch-all, so genuine transient errors still
# retry on the next cycle.
_QUOTA_MARKERS = (
    "usage limit",      # "You've hit your org's monthly usage limit"
    "rate limit",
    "quota",
)
QUOTA_COOLDOWN_S = 3600  # how long to short-circuit after a quota error

_quota_blocked_until: float = 0.0


def _is_quota_error(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _QUOTA_MARKERS)


def quota_blocked() -> bool:
    """True while the breaker is open (callers/tests may introspect)."""
    return time.time() < _quota_blocked_until


def reset_quota_breaker() -> None:
    """Clear the breaker — for tests and manual recovery."""
    global _quota_blocked_until
    _quota_blocked_until = 0.0


def claude_call(
    prompt: str,
    model: str = "claude-sonnet-4-6",
    timeout: int = 120,
) -> str | None:
    """
    Run `claude --model MODEL --print` with prompt piped via stdin.
    Returns stdout on success, None on any failure.
    """
    global _quota_blocked_until

    if not shutil.which("claude"):
        return None

    # Breaker open: a quota error was seen recently. Don't spawn — every caller
    # treats None as "unavailable" and degrades. Silent on purpose: the trip
    # itself already logged the reason; logging here too would re-spam the very
    # noise this breaker exists to kill.
    if time.time() < _quota_blocked_until:
        return None

    try:
        result = subprocess.run(
            ["claude", "--model", model, "--print",
             "--permission-mode", "bypassPermissions"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            err = result.stderr.strip()[:300]
            # CLI sometimes writes the failure reason to stdout instead of stderr
            # (e.g. rate-limit / auth errors), so fall back to stdout when stderr is empty.
            if not err:
                err = result.stdout.strip()[:300] or "<no output>"
            if _is_quota_error(err):
                _quota_blocked_until = time.time() + QUOTA_COOLDOWN_S
                print(f"[claude_cli] Quota/limit hit (model={model}): {err} "
                      f"— circuit breaker open for {QUOTA_COOLDOWN_S}s")
            else:
                print(f"[claude_cli] Error (model={model}, rc={result.returncode}): {err}")
            return None
        out = result.stdout.strip()
        if not out:
            err = result.stderr.strip()[:300] or "<empty stdout, rc=0>"
            print(f"[claude_cli] Empty result (model={model}): {err}")
            return None
        return out
    except subprocess.TimeoutExpired:
        print(f"[claude_cli] Timeout after {timeout}s (model={model})")
        return None
    except Exception as e:
        print(f"[claude_cli] Exception: {e}")
        return None

"""
Safe Claude CLI wrapper — pipes prompt via stdin to avoid ARG_MAX limits.

Usage:
    from core.claude_cli import claude_call
    output = claude_call(prompt, model="claude-sonnet-4-6", timeout=90)
    # Returns stdout string on success, None on failure.
"""
import shutil
import subprocess
from pathlib import Path


def claude_call(
    prompt: str,
    model: str = "claude-sonnet-4-6",
    timeout: int = 120,
) -> str | None:
    """
    Run `claude --model MODEL --print` with prompt piped via stdin.
    Returns stdout on success, None on any failure.
    """
    if not shutil.which("claude"):
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

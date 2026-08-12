"""
Shared LLM wrapper for Digital Intern / ArticleNet.

All trading-intelligence LLM calls go through Grok (xAI) by default via the
OpenAI-compatible HTTP API, using the same OpenClaw xAI OAuth/auth-profile
token path as Paper Trader.

Usage:
    from core.claude_cli import claude_call, DEFAULT_LLM_MODEL
    output = claude_call(prompt, model=DEFAULT_LLM_MODEL, timeout=90)
    # Returns stdout/text on success, None on failure.

The function name ``claude_call`` is retained for import compatibility across
alert/scorer/analyst/chat call sites. It is no longer Claude-specific.

Quota circuit breaker
---------------------
When the provider reports an org usage / rate / quota limit, subsequent
``claude_call`` invocations short-circuit (return None without network/CLI
churn) for ``QUOTA_COOLDOWN_S``. Every caller already treats None as
"LLM unavailable" and degrades gracefully.
"""
from __future__ import annotations

import json
import os
import shutil
import ssl
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_LLM_MODEL = os.environ.get("DIGITAL_INTERN_LLM_MODEL", "grok-4.5")
CODEX_REASONING_EFFORT = os.environ.get(
    "DIGITAL_INTERN_CODEX_REASONING_EFFORT",
    "xhigh",
)

XAI_API_BASE = os.environ.get(
    "DIGITAL_INTERN_XAI_API_BASE",
    os.environ.get("PAPER_TRADER_XAI_API_BASE", "https://api.x.ai/v1"),
).rstrip("/")
XAI_AUTH_PROFILES_PATH = Path(
    os.environ.get(
        "DIGITAL_INTERN_XAI_AUTH_PROFILES",
        os.environ.get(
            "PAPER_TRADER_XAI_AUTH_PROFILES",
            str(Path.home() / ".openclaw" / "agents" / "main" / "agent" / "auth-profiles.json"),
        ),
    )
)
XAI_AUTH_PROFILE = os.environ.get(
    "DIGITAL_INTERN_XAI_AUTH_PROFILE",
    os.environ.get("PAPER_TRADER_XAI_AUTH_PROFILE", "xai:artintel1110@gmail.com"),
)

# Cursor CLI OpenAI-compatible proxy (LaunchAgent com.cursor-agent-api).
# Used when SuperGrok/xAI credits or rate limits are exhausted. Still Grok 4.5
# High, billed through Cursor — never Claude.
CURSOR_API_BASE = os.environ.get(
    "DIGITAL_INTERN_CURSOR_API_BASE",
    os.environ.get("PAPER_TRADER_CURSOR_API_BASE", "http://127.0.0.1:4646/v1"),
).rstrip("/")
CURSOR_MODEL = os.environ.get(
    "DIGITAL_INTERN_CURSOR_MODEL",
    os.environ.get("PAPER_TRADER_CURSOR_MODEL", "cursor-grok-4.5-high"),
)
CURSOR_FALLBACK = os.environ.get(
    "DIGITAL_INTERN_CURSOR_FALLBACK",
    os.environ.get("PAPER_TRADER_CURSOR_FALLBACK", "1"),
).strip().lower() not in {"0", "false", "no", "off"}

# Substrings (matched case-insensitively against the failure text) that mean
# "spawning again right now is pointless".
_QUOTA_MARKERS = (
    "usage limit",
    "rate limit",
    "quota",
    "429",
    "credits",
    "credit",
    "exhausted",
)
QUOTA_COOLDOWN_S = 3600

_quota_blocked_until: float = 0.0


def _resolve_cli(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for path in (
        f"/usr/local/bin/{name}",
        f"/opt/homebrew/bin/{name}",
        f"/usr/bin/{name}",
    ):
        if Path(path).exists():
            return path
    return None


def _is_quota_error(text: str) -> bool:
    low = (text or "").lower()
    return any(m in low for m in _QUOTA_MARKERS)


def quota_blocked() -> bool:
    """True while the breaker is open (callers/tests may introspect)."""
    return time.time() < _quota_blocked_until


def reset_quota_breaker() -> None:
    """Clear the breaker — for tests and manual recovery."""
    global _quota_blocked_until
    _quota_blocked_until = 0.0


def _uses_xai_http(model: str | None) -> bool:
    name = (model or DEFAULT_LLM_MODEL or "").strip().lower()
    return name.startswith("grok-") or name.startswith("xai/")


def _normalize_model_name(model: str | None) -> str:
    name = (model or DEFAULT_LLM_MODEL or "").strip()
    if name.startswith("xai/"):
        return name.split("/", 1)[1]
    return name


def _load_xai_access_token() -> str | None:
    """Resolve a bearer token for api.x.ai from env or OpenClaw auth profiles."""
    for key in (
        "DIGITAL_INTERN_XAI_API_KEY",
        "PAPER_TRADER_XAI_API_KEY",
        "XAI_API_KEY",
    ):
        value = (os.environ.get(key) or "").strip()
        if value:
            return value
    try:
        raw = json.loads(XAI_AUTH_PROFILES_PATH.read_text())
    except Exception as e:
        print(f"[claude_cli] xAI auth profiles unreadable: {e}")
        return None
    profiles = raw.get("profiles") if isinstance(raw, dict) else None
    if not isinstance(profiles, dict):
        return None
    preferred = [XAI_AUTH_PROFILE, "xai:default", "xai"]
    candidates: list[tuple[str, object]] = []
    for key in preferred:
        if key and key in profiles:
            candidates.append((key, profiles[key]))
    for key, value in profiles.items():
        if str(key).startswith("xai:") or (
            isinstance(value, dict)
            and str(value.get("provider", "")).lower() == "xai"
        ):
            if (key, value) not in candidates:
                candidates.append((key, value))
    now_ms = int(time.time() * 1000)
    for key, value in candidates:
        if not isinstance(value, dict):
            continue
        token = (
            value.get("access")
            or value.get("accessToken")
            or value.get("apiKey")
            or value.get("api_key")
            or value.get("token")
        )
        if not token:
            continue
        expires = value.get("expires")
        try:
            exp_i = int(expires) if expires is not None else None
        except (TypeError, ValueError):
            exp_i = None
        if exp_i is not None and exp_i < now_ms:
            print(f"[claude_cli] xAI auth profile expired: {key}")
            continue
        return str(token)
    return None


def _ssl_context():
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _cursor_fallback_enabled() -> bool:
    return bool(CURSOR_FALLBACK)


def _load_cursor_api_key() -> str | None:
    """Optional client key for the local Cursor proxy (proxy often has its own)."""
    for key in (
        "DIGITAL_INTERN_CURSOR_API_KEY",
        "PAPER_TRADER_CURSOR_API_KEY",
        "CURSOR_API_KEY",
    ):
        value = (os.environ.get(key) or "").strip()
        if value:
            return value
    env_path = Path.home() / ".cursor-agent-api" / "env"
    try:
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("export CURSOR_API_KEY="):
                line = line[len("export ") :]
            if line.startswith("CURSOR_API_KEY="):
                return line.split("=", 1)[1].strip().strip("'").strip('"')
    except Exception:
        pass
    return None


def _openai_compatible_chat(
    *,
    base_url: str,
    model_id: str,
    prompt: str,
    timeout: int,
    token: str | None,
    label: str,
    system: str,
) -> str | None:
    """Shared OpenAI-compatible chat/completions helper for xAI + Cursor."""
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "stream": False,
    }
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "DigitalIntern-LLM/1.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
        choices = data.get("choices") or []
        if not choices:
            print(f"[claude_cli] Empty choices ({label} model={model_id})")
            return None
        message = choices[0].get("message") or {}
        text = (message.get("content") or "").strip()
        if not text:
            print(f"[claude_cli] Empty content ({label} model={model_id})")
            return None
        return text
    except urllib.error.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = ""
        err = (err_body or str(e))[:300]
        print(f"[claude_cli] {label} HTTP err (model={model_id}, rc={e.code}): {err}")
        raise
    except Exception as e:
        print(f"[claude_cli] {label} HTTP exception (model={model_id}): {e}")
        raise


def _xai_http_call(prompt: str, model: str, timeout: int) -> str | None:
    """Call Grok over the OpenAI-compatible xAI HTTP API."""
    global _quota_blocked_until

    token = _load_xai_access_token()
    if not token:
        print("[claude_cli] xAI auth missing; cannot call Grok HTTP API")
        return None

    model_id = _normalize_model_name(model)
    system = (
        "You are the Digital Intern / ArticleNet LLM backend. "
        "Follow the user instructions exactly. Prefer concise, "
        "structured output when requested."
    )
    try:
        return _openai_compatible_chat(
            base_url=XAI_API_BASE,
            model_id=model_id,
            prompt=prompt,
            timeout=timeout,
            token=token,
            label="xAI",
            system=system,
        )
    except urllib.error.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = ""
        err = (err_body or str(e))[:300]
        if _is_quota_error(f"{e.code} {err}"):
            _quota_blocked_until = time.time() + QUOTA_COOLDOWN_S
            print(
                f"[claude_cli] Quota/limit hit (model={model_id}): {err} "
                f"— xAI circuit open for {QUOTA_COOLDOWN_S}s "
                "(Cursor Grok fallback still allowed)"
            )
        return None
    except Exception:
        return None


def _cursor_http_call(prompt: str, timeout: int) -> str | None:
    """Call Grok via the local Cursor CLI OpenAI proxy (SuperGrok backup)."""
    if not _cursor_fallback_enabled():
        return None
    model_id = (CURSOR_MODEL or "cursor-grok-4.5-high").strip()
    system = (
        "You are the Digital Intern / ArticleNet LLM backend. "
        "Follow the user instructions exactly. Prefer concise, "
        "structured output when requested."
    )
    try:
        text = _openai_compatible_chat(
            base_url=CURSOR_API_BASE,
            model_id=model_id,
            prompt=prompt,
            timeout=timeout,
            token=_load_cursor_api_key(),
            label="Cursor",
            system=system,
        )
        if text:
            print(f"[claude_cli] Cursor Grok fallback OK (model={model_id})")
        return text
    except Exception:
        return None


def _cli_call(prompt: str, model: str, timeout: int) -> str | None:
    """Legacy Codex/Claude CLI path kept only as an explicit override."""
    global _quota_blocked_until

    use_codex = model.startswith("gpt-")
    cli = "codex" if use_codex else "claude"
    cli_path = _resolve_cli(cli)
    if not cli_path:
        return None

    try:
        cmd = (
            [
                cli_path,
                "exec",
                "--model",
                model,
                "-c",
                f'model_reasoning_effort="{CODEX_REASONING_EFFORT}"',
                "--sandbox",
                "read-only",
                "--cd",
                str(Path.cwd()),
                "--ephemeral",
                "--color",
                "never",
                "-",
            ]
            if use_codex
            else [
                cli_path,
                "--model",
                model,
                "--print",
                "--permission-mode",
                "bypassPermissions",
            ]
        )
        env = os.environ.copy()
        env["PATH"] = (
            "/usr/local/bin:/opt/homebrew/bin:"
            f"{Path.home() / '.local/bin'}:"
            f"{Path.home() / '.npm-global/bin'}:"
            + env.get("PATH", "")
        )
        if use_codex:
            env["CODEX_HOME"] = os.environ.get(
                "DIGITAL_INTERN_CODEX_HOME",
                str(Path.home() / ".codex"),
            )
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        if result.returncode != 0:
            err = result.stderr.strip()[:300]
            if not err:
                err = result.stdout.strip()[:300] or "<no output>"
            if _is_quota_error(err):
                _quota_blocked_until = time.time() + QUOTA_COOLDOWN_S
                print(
                    f"[claude_cli] Quota/limit hit (model={model}): {err} "
                    f"— circuit breaker open for {QUOTA_COOLDOWN_S}s"
                )
            else:
                print(
                    f"[claude_cli] Error (model={model}, rc={result.returncode}): {err}"
                )
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


def claude_call(
    prompt: str,
    model: str = DEFAULT_LLM_MODEL,
    timeout: int = 120,
) -> str | None:
    """
    Run the configured LLM backend for ``model``.

    Default / Grok models use xAI HTTP first. When SuperGrok/xAI is rate-limited
    or credits are exhausted, fall back to the local Cursor CLI proxy
    (``cursor-grok-4.5-high``) — still Grok, billed via Cursor. Explicit
    ``gpt-*`` model ids still fall through to Codex CLI only if requested.
    """
    # HARD RULE 2026-08-04: Claude/Anthropic spend disabled unless explicitly re-enabled.
    _m = (model or DEFAULT_LLM_MODEL or "").strip().lower()
    if (
        _m.startswith("claude")
        or _m.startswith("anthropic/")
        or "sonnet" in _m
        or "opus" in _m
        or "haiku" in _m
        or not _uses_xai_http(model)
    ) and not _m.startswith("gpt-"):
        if not _uses_xai_http(model):
            print(
                f"[claude_cli] BLOCKED Claude/non-Grok model={model!r}; "
                "forcing grok-4.5 (Claude spend disabled 2026-08-04)"
            )
            model = "grok-4.5"

    if _uses_xai_http(model):
        result = None
        if not quota_blocked():
            result = _xai_http_call(prompt, model=model, timeout=timeout)
            if result:
                return result
        elif _cursor_fallback_enabled():
            print(
                "[claude_cli] xAI circuit open; trying Cursor Grok fallback "
                f"(model={CURSOR_MODEL})"
            )
        if _cursor_fallback_enabled():
            return _cursor_http_call(prompt, timeout=timeout)
        return None
    # Refuse Claude CLI path entirely.
    if (model or "").startswith("claude") or (model or "").startswith("anthropic"):
        print(f"[claude_cli] BLOCKED Claude CLI path for model={model!r}")
        return None
    return _cli_call(prompt, model=model, timeout=timeout)

"""Unified LLM call adapter for backtest and future callers.

Routes `call_llm(model_id, prompt)` to the appropriate backend:
  - "grok-*" / "xai/*" / "cursor-*" → strategy._claude_call
    (xAI Grok primary, Cursor CLI Grok fallback on 127.0.0.1:4646)
  - "hf/<org>/<model>" → HuggingFace Inference API (uses _HF_SEM, max 3 concurrent)
  - "claude-*" / "anthropic/*" → blocked (Claude spend disabled)

The live strategy.py calls its own `_claude_call` path directly.
This module owns concurrency semaphores so they are shared across
all callers that import it.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import requests

HF_BASE = "https://router.huggingface.co/v1"
HF_TIMEOUT_S = 90
HF_RETRIES = 2
HF_RETRY_BACKOFF_S = 10

_GROK_SEM = threading.Semaphore(2)  # max 2 concurrent Grok calls (OOM / rate guard)
_HF_SEM = threading.Semaphore(3)  # max 3 concurrent HF API calls


def _is_grok_model(model_id: str) -> bool:
    name = (model_id or "").strip().lower()
    return (
        name.startswith("grok-")
        or name.startswith("xai/")
        or name.startswith("cursor-")
        or name.startswith("cursor-cli/")
        or name == "grok"
    )


def call_llm(model_id: str, prompt: str, timeout: int = None) -> str | None:
    """Route prompt to the right LLM backend. Returns raw response string or None."""
    if model_id.startswith("hf/"):
        return _hf_call(model_id[3:], prompt, HF_TIMEOUT_S if timeout is None else timeout)
    if _is_grok_model(model_id):
        return _grok_call(model_id, prompt, timeout)
    if model_id.startswith("claude-") or model_id.startswith("anthropic/"):
        # HARD RULE 2026-08-04: Claude spend disabled. Fail closed.
        print(
            f"[llm_adapter] BLOCKED Claude model_id={model_id!r} "
            "(Claude spend disabled; use Grok/xAI or hf/*)"
        )
        return None
    raise ValueError(
        f"Unknown model_id: {model_id!r}. "
        "Use grok-*/xai/*/cursor-* (primary+fallback) or hf/*."
    )


def _grok_call(model_id: str, prompt: str, timeout: int | None) -> str | None:
    """xAI Grok primary with Cursor CLI Grok fallback via strategy._claude_call."""
    from . import strategy

    timeout_s = 90 if timeout is None else timeout
    with _GROK_SEM:
        return strategy._claude_call(prompt, timeout_s=timeout_s, model=model_id)


def _hf_call(hf_model: str, prompt: str, timeout: int) -> str | None:
    token = _load_hf_token()
    if not token:
        print("[llm_adapter] HF token not found — set HUGGINGFACE_HUB_TOKEN or HF_TOKEN")
        return None
    with _HF_SEM:
        for attempt in range(HF_RETRIES + 1):
            try:
                resp = requests.post(
                    f"{HF_BASE}/chat/completions",
                    headers={"Authorization": f"Bearer {token}"},
                    json={
                        "model": hf_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 512,
                    },
                    timeout=timeout,
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
                print(f"[llm_adapter] HF attempt {attempt+1} status={resp.status_code}")
                if resp.status_code == 429 or resp.status_code >= 500:
                    if attempt < HF_RETRIES:
                        time.sleep(HF_RETRY_BACKOFF_S)
                    continue
                return None  # 4xx (not 429) — don't retry
            except requests.Timeout:
                print(f"[llm_adapter] HF timeout attempt {attempt+1}")
            except Exception as e:
                print(f"[llm_adapter] HF exception attempt {attempt+1}: {e}")
            if attempt < HF_RETRIES:
                time.sleep(HF_RETRY_BACKOFF_S)
    return None


def _load_hf_token() -> str | None:
    """Load HF token from env vars, then fall back to digital-intern .env file."""
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if token:
        return token
    for env_path in (
        Path.home() / "trading-intelligence" / "digital-intern" / ".env",
        Path("/Users/jonathan/trading-intelligence/digital-intern/.env"),
        Path("/home/zeph/trading-intelligence/digital-intern/.env"),
    ):
        if not env_path.exists():
            continue
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith(("HUGGINGFACE_HUB_TOKEN=", "HF_TOKEN=")):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None

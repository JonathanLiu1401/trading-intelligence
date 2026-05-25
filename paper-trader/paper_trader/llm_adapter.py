"""Unified LLM call adapter for backtest and future callers.

Routes `call_llm(model_id, prompt)` to the appropriate backend:
  - "claude-*"  → Claude CLI subprocess (uses _CLAUDE_SEM, max 2 concurrent)
  - "hf/<org>/<model>" → HuggingFace Inference API (uses _HF_SEM, max 3 concurrent)

The live strategy.py is NOT a caller — it calls Claude directly.
This module owns both concurrency semaphores so they are shared across
all callers that import it.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import requests

HF_BASE = "https://router.huggingface.co/v1"
HF_TIMEOUT_S = 90
HF_RETRIES = 2
HF_RETRY_BACKOFF_S = 10

_CLAUDE_SEM = threading.Semaphore(2)   # max 2 concurrent claude subprocesses (OOM guard)
_HF_SEM = threading.Semaphore(3)       # max 3 concurrent HF API calls


def call_llm(model_id: str, prompt: str, timeout: int = None) -> str | None:
    """Route prompt to the right LLM backend. Returns raw response string or None."""
    if model_id.startswith("hf/"):
        return _hf_call(model_id[3:], prompt, timeout)
    elif model_id.startswith("claude-"):
        return _claude_call(model_id, prompt)
    raise ValueError(f"Unknown model_id: {model_id!r}. Must start with 'hf/' or 'claude-'.")


def _claude_call(model_id: str, prompt: str, retries: int = 1) -> str | None:
    if not shutil.which("claude"):
        print("[llm_adapter] claude CLI not found")
        return None
    with _CLAUDE_SEM:
        for attempt in range(retries + 1):
            try:
                r = subprocess.run(
                    ["claude", "--model", model_id, "--print",
                     "--permission-mode", "bypassPermissions"],
                    input=prompt, capture_output=True, text=True,
                    timeout=None,  # no timeout — wait as long as Opus needs
                )
                if r.returncode == 0 and r.stdout.strip():
                    return r.stdout.strip()
                print(f"[llm_adapter] claude attempt {attempt+1} rc={r.returncode} "
                      f"err={r.stderr.strip()[:200]!r}")
            except subprocess.TimeoutExpired:
                print(f"[llm_adapter] claude timeout attempt {attempt+1}")
            except Exception as e:
                print(f"[llm_adapter] claude exception attempt {attempt+1}: {e}")
            if attempt < retries:
                time.sleep(2)
    return None


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
    env_path = Path("/home/zeph/trading-intelligence/digital-intern/.env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith(("HUGGINGFACE_HUB_TOKEN=", "HF_TOKEN=")):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None

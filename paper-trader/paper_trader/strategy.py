"""Trading strategy — packages context, asks an LLM for a JSON decision,
executes it through paper trade plumbing. No hard risk limits; the model has
full autonomy."""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import sqlite3
import statistics
import subprocess
import time
from datetime import date, datetime, timezone
from pathlib import Path

from . import market, signals
from .store import Store, get_store

# Pre-flight host-saturation guard (see paper_trader/host_guard.py). Imported
# at module level so it is a monkeypatchable seam — `decide()` calls the
# module global, exactly like `_claude_call`, so tests can neutralise the
# ambient host probe deterministically (conftest does this autouse) and the
# wired skip can be exercised explicitly. Degrade-safe: host_guard is
# stdlib-only and should never fail to import, but if it does the trader must
# still run (fall back to "never saturated"), mirroring the file's
# non-fatal-by-construction contract.
try:
    from .host_guard import host_saturated
except Exception:  # pragma: no cover - stdlib-only module, import can't fail
    def host_saturated(*_a, **_k):
        return (False, "host_guard unavailable")

MODEL = os.environ.get("PAPER_TRADER_MODEL", "grok-4.6")
FALLBACK_MODEL = os.environ.get("PAPER_TRADER_FALLBACK_MODEL", "grok-4.6")
CODEX_AUTH_FALLBACK_MODEL = os.environ.get(
    "PAPER_TRADER_CODEX_AUTH_FALLBACK_MODEL",
    "",
)
# Grok HTTP path uses bounded timeouts; CLI paths still allow unlimited waits.
FALLBACK_TIMEOUT_S = int(os.environ.get("PAPER_TRADER_FALLBACK_TIMEOUT_S", "180") or "180")
DECISION_TIMEOUT_S = int(os.environ.get("PAPER_TRADER_DECISION_TIMEOUT_S", "180") or "180")
# Retry uses the same bounded budget as the primary call.
RETRY_TIMEOUT_S = int(os.environ.get("PAPER_TRADER_RETRY_TIMEOUT_S", "180") or "180")
# Cap the raw-response excerpt we write back into decisions.reasoning. Long
# enough to diagnose JSON / prose / truncation, short enough to keep the DB lean.
RAW_CAPTURE_CHARS = 1000

# OpenClaw / xAI login for direct Grok HTTP decisions. Prefer env overrides,
# then the local OpenClaw auth-profiles store used by this Mac's main agent.
XAI_API_BASE = os.environ.get("PAPER_TRADER_XAI_API_BASE", "https://api.x.ai/v1").rstrip("/")
XAI_AUTH_PROFILES_PATH = Path(os.environ.get(
    "PAPER_TRADER_XAI_AUTH_PROFILES",
    str(Path.home() / ".openclaw" / "agents" / "main" / "agent" / "auth-profiles.json"),
))
XAI_AUTH_PROFILE = os.environ.get(
    "PAPER_TRADER_XAI_AUTH_PROFILE",
    "xai:artintel1110@gmail.com",
)
# Cursor CLI OpenAI-compatible proxy (LaunchAgent com.cursor-agent-api).
# Fallback ONLY when SuperGrok/xAI is rate-limited or credits are exhausted.
# Still Grok 4.6 Extra High (xhigh), billed through Cursor — never Claude, never primary.
CURSOR_API_BASE = os.environ.get(
    "PAPER_TRADER_CURSOR_API_BASE", "http://127.0.0.1:4646/v1"
).rstrip("/")
CURSOR_MODEL = os.environ.get(
    "PAPER_TRADER_CURSOR_MODEL", "cursor-grok-4.6-xhigh"
)
CURSOR_FALLBACK = os.environ.get(
    "PAPER_TRADER_CURSOR_FALLBACK", "1"
).strip().lower() not in {"0", "false", "no", "off"}
# When the LLM path is unavailable, allow the qualified ML advisor to place
# real BUY/SHORT entries instead of only logging HOLD.
ML_DROUGHT_ALLOW_ENTRIES = os.environ.get(
    "PAPER_TRADER_ML_DROUGHT_ALLOW_ENTRIES", "1"
).strip().lower() not in {"0", "false", "no", "off"}
ML_DROUGHT_BUY_CASH_PCT = float(os.environ.get(
    "PAPER_TRADER_ML_DROUGHT_BUY_CASH_PCT", "0.15",
))
ML_DROUGHT_BUY_MAX_USD = float(os.environ.get(
    "PAPER_TRADER_ML_DROUGHT_BUY_MAX_USD", "1500",
))
ML_DROUGHT_SHORT_CASH_PCT = float(os.environ.get(
    "PAPER_TRADER_ML_DROUGHT_SHORT_CASH_PCT", "0.10",
))
ML_DROUGHT_SHORT_MAX_USD = float(os.environ.get(
    "PAPER_TRADER_ML_DROUGHT_SHORT_MAX_USD", "1000",
))
ML_DROUGHT_MIN_CONFIDENCE = float(os.environ.get(
    "PAPER_TRADER_ML_DROUGHT_MIN_CONFIDENCE", "0.35",
))

# Margin/leverage model for live paper stock BUYs. The book may spend current
# cash plus 50% of net worth, with regular-stock leverage requested per trade
# as an exposure multiplier from 1x to 20x. Positions store economic exposure
# as effective shares so the existing mark-to-market schema keeps working.
STOCK_MARGIN_NET_WORTH_PCT = 0.50
STOCK_BUY_MIN_LEVERAGE = 1.0
STOCK_BUY_MAX_LEVERAGE = 20.0
# Operator deployment target (Jonathan 2026-08-06): stay invested, use margin.
# Cash should stay under 10% of net worth. Gross long exposure target is
# 120%-150% of net worth (i.e. spend into the 1.5x BP sleeve). The engine
# auto-boosts stock BUY leverage when the book is under-deployed so the
# model cannot idle in cash without an explicit risk reason.
MAX_IDLE_CASH_PCT = 0.10
TARGET_GROSS_EXPOSURE_MIN = 1.20
TARGET_GROSS_EXPOSURE_MAX = 1.50
TARGET_GROSS_EXPOSURE_IDEAL = 1.35
AUTO_LEVERAGE_WHEN_UNDERDEPLOYED = True

# Hard operator discipline for the live paper book. These are deliberately
# enforced in _execute(), not merely described to the model, because the live
# trade log showed repeated same-ticker round trips and sector pile-ons.
TRADE_DISCIPLINE_MIN_BOOK_VALUE = float(os.environ.get(
    "PAPER_TRADER_DISCIPLINE_MIN_BOOK_VALUE", "5000",
))
ENTRY_COOLDOWN_AFTER_EXIT_S = int(os.environ.get(
    # Block same-name re-entry churn after a full/partial exit.
    "PAPER_TRADER_ENTRY_COOLDOWN_AFTER_EXIT_S", str(24 * 3600),
))
ENTRY_COOLDOWN_AFTER_ANY_TRADE_S = int(os.environ.get(
    # Soft pause after any trade on the ticker (adds included).
    "PAPER_TRADER_ENTRY_COOLDOWN_AFTER_ANY_TRADE_S", str(4 * 3600),
))
# Default hold lock for discretionary exits. Jonathan 2026-08-07 mandate:
# day-trader mode. 0 disables the multi-session office-worker lock so the
# book can flatten ballast / rotate same day. Hard stops and thesis-kill still
# work either way. Override via env if needed.
# Permanent default 4h: blocks same-session panic flats (LITE 2026-08-10).
# Override with PAPER_TRADER_MIN_HOLD_BEFORE_DISCRETIONARY_EXIT_S.
MIN_HOLD_BEFORE_DISCRETIONARY_EXIT_S = int(os.environ.get(
    "PAPER_TRADER_MIN_HOLD_BEFORE_DISCRETIONARY_EXIT_S", str(4 * 3600),
))
# Permanent anti-panic: do not sell/cover losers just because they are red.
# Winners can still exit after min-hold. Underwater exits need hard stop,
# real thesis-kill evidence, or long-option -50% premium kill.
BLOCK_UNDERWATER_DISCRETIONARY_EXITS = os.environ.get(
    "PAPER_TRADER_BLOCK_UNDERWATER_DISCRETIONARY_EXITS", "1"
).strip().lower() not in {"0", "false", "no", "off"}
UNDERWATER_EXIT_EPS = float(
    os.environ.get("PAPER_TRADER_UNDERWATER_EXIT_EPS", "0.002")
)  # 0.2% noise buffer vs avg_cost
OPTION_PREMIUM_KILL_PCT = float(
    os.environ.get("PAPER_TRADER_OPTION_PREMIUM_KILL_PCT", "0.50")
)
MAX_POST_TRADE_SINGLE_NAME_PCT = float(os.environ.get(
    "PAPER_TRADER_MAX_POST_TRADE_SINGLE_NAME_PCT", "70",
))
MAX_POST_TRADE_SECTOR_PCT = float(os.environ.get(
    "PAPER_TRADER_MAX_POST_TRADE_SECTOR_PCT", "70",
))
MAX_POST_TRADE_LEVERAGED_PCT = float(os.environ.get(
    "PAPER_TRADER_MAX_POST_TRADE_LEVERAGED_PCT", "70",
))

# ML advisor gate — when the backtest ML model's median alpha consistently
# beats SPY, its (quant+news) recommendation is injected into the Opus prompt
# as an *advisory* opinion. Opus retains full autonomy over the final call.
ML_QUALIFY_MIN_RUNS = 20       # qualifying runs needed
ML_QUALIFY_MEDIAN_ALPHA = 0.0  # median vs_spy_pct must beat this (%)
ML_QUALIFY_TTL_S = 3600.0      # recheck every hour

_ml_qualify_cache: tuple[bool, str, float] | None = None

_CLI_PATH_PREFIXES = (
    str(Path.home() / ".local" / "bin"),
    str(Path.home() / ".nvm" / "versions" / "node" / "v24.15.0" / "bin"),
    "/usr/local/bin",
    "/opt/homebrew/bin",
    "/home/zeph/.local/bin",
    "/home/zeph/.nvm/versions/node/v24.15.0/bin",
)


def _cli_path(name: str) -> str | None:
    override = os.environ.get(f"PAPER_TRADER_{name.upper()}_BIN")
    if override and Path(override).exists():
        return override
    found = shutil.which(name)
    if found:
        return found
    for directory in _CLI_PATH_PREFIXES:
        candidate = Path(directory) / name
        if Path(candidate).exists():
            return str(candidate)
    return None


def _llm_subprocess_env(use_codex: bool) -> dict[str, str]:
    """PATH-safe environment for launchd/systemd-started model CLI wrappers."""
    entries: list[str] = []
    for value in (*_CLI_PATH_PREFIXES, os.environ.get("PATH", "")):
        if not value:
            continue
        entries.extend(part for part in value.split(os.pathsep) if part)
    entries.extend(["/usr/bin", "/bin", "/usr/sbin", "/sbin"])
    deduped = list(dict.fromkeys(entries))
    env = {**os.environ, "PATH": os.pathsep.join(deduped)}
    if use_codex:
        env["CODEX_HOME"] = os.environ.get(
            "PAPER_TRADER_CODEX_HOME",
            str(Path.home() / ".codex"),
        )
    return env


def _uses_xai_http(model: str | None = None) -> bool:
    name = (model or MODEL or "").strip().lower()
    return name.startswith("grok-") or name.startswith("xai/")


# Failure-cause globals must be bound before any function uses `global` on them
# (annotated assignment after a prior global use breaks py_compile).
_quota_exhausted = False
_last_claude_fail = None  # str | None; set by _claude_call failure paths

def _normalize_model_name(model: str | None = None) -> str:
    name = (model or MODEL or "").strip()
    if name.startswith("xai/"):
        return name.split("/", 1)[1]
    return name


def _load_xai_access_token() -> str | None:
    """Resolve a bearer token for api.x.ai from env or OpenClaw auth profiles."""
    for key in ("PAPER_TRADER_XAI_API_KEY", "XAI_API_KEY"):
        value = (os.environ.get(key) or "").strip()
        if value:
            return value
    try:
        raw = json.loads(Path(XAI_AUTH_PROFILES_PATH).read_text())
    except Exception as e:
        print(f"[strategy] xAI auth profiles unreadable: {e}")
        return None
    profiles = raw.get("profiles") if isinstance(raw, dict) else None
    if not isinstance(profiles, dict):
        return None
    preferred = [
        XAI_AUTH_PROFILE,
        "xai:default",
        "xai",
    ]
    candidates = []
    for key in preferred:
        if key and key in profiles:
            candidates.append((key, profiles[key]))
    for key, value in profiles.items():
        if str(key).startswith("xai:") or (
            isinstance(value, dict) and str(value.get("provider", "")).lower() == "xai"
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
            print(f"[strategy] xAI auth profile expired: {key}")
            continue
        return str(token)
    return None


def _cursor_fallback_enabled() -> bool:
    return bool(CURSOR_FALLBACK)


def _load_cursor_api_key() -> str | None:
    """Optional client key for the local Cursor proxy."""
    for key in ("PAPER_TRADER_CURSOR_API_KEY", "CURSOR_API_KEY"):
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
    timeout_s: int | None,
    token: str | None,
    label: str,
) -> str | None:
    """Shared chat/completions helper for xAI primary and Cursor fallback."""
    global _last_claude_fail
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are the paper trader decision engine. "
                    "Return only one JSON object. No markdown fences."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "stream": False,
    }
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "PaperTrader-LLM/1.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = json.dumps(payload).encode("utf-8")
    req = __import__("urllib.request", fromlist=["Request"]).Request(
        f"{base_url}/chat/completions",
        data=body,
        headers=headers,
        method="POST",
    )
    timeout = timeout_s if timeout_s is not None else DECISION_TIMEOUT_S
    import ssl
    import urllib.error
    import urllib.request

    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
        choices = data.get("choices") or []
        if not choices:
            _last_claude_fail = "empty_stdout"
            return None
        message = choices[0].get("message") or {}
        text = (message.get("content") or "").strip()
        if not text:
            _last_claude_fail = "empty_stdout"
            return None
        return text
    except urllib.error.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = ""
        print(f"[strategy] {label} HTTP err (rc={e.code}): {(err_body or str(e))[:300]}")
        e._paper_trader_body = err_body  # type: ignore[attr-defined]
        raise
    except Exception as e:
        msg = str(e).lower()
        if "timed out" in msg or "timeout" in msg:
            print(f"[strategy] {label} HTTP timeout after {timeout}s")
            _last_claude_fail = "timeout"
        else:
            print(f"[strategy] {label} HTTP exception: {e}")
            _last_claude_fail = "exception"
        raise


def _xai_http_call(prompt: str, timeout_s: int | None, model: str) -> str | None:
    """Call Grok over the OpenAI-compatible xAI HTTP API using OpenClaw login."""
    global _last_claude_fail, _quota_exhausted
    token = _load_xai_access_token()
    if not token:
        print("[strategy] xAI auth missing; cannot call Grok HTTP API")
        _last_claude_fail = "cli_missing"
        return None
    model_id = _normalize_model_name(model)
    try:
        return _openai_compatible_chat(
            base_url=XAI_API_BASE,
            model_id=model_id,
            prompt=prompt,
            timeout_s=timeout_s,
            token=token,
            label="xAI",
        )
    except Exception as e:
        import urllib.error

        if isinstance(e, urllib.error.HTTPError):
            err_body = getattr(e, "_paper_trader_body", "") or ""
            if not err_body:
                try:
                    err_body = e.read().decode("utf-8", errors="replace")
                except Exception:
                    err_body = ""
            combined = f"{e} {err_body}".lower()
            if any(s in combined for s in ("401", "unauthorized", "authentication", "invalid api key")):
                _last_claude_fail = "nonzero_rc"
            elif any(
                s in combined
                for s in ("429", "quota", "rate limit", "usage limit", "credit")
            ):
                _quota_exhausted = True
            else:
                _last_claude_fail = "nonzero_rc"
        return None


def _cursor_http_call(prompt: str, timeout_s: int | None) -> str | None:
    """Call Grok via local Cursor CLI proxy — SuperGrok backup only."""
    global _last_claude_fail
    if not _cursor_fallback_enabled():
        return None
    model_id = (CURSOR_MODEL or "cursor-grok-4.6-xhigh").strip()
    try:
        text = _openai_compatible_chat(
            base_url=CURSOR_API_BASE,
            model_id=model_id,
            prompt=prompt,
            timeout_s=timeout_s,
            token=_load_cursor_api_key(),
            label="Cursor",
        )
        if text:
            print(f"[strategy] Cursor Grok fallback OK (model={model_id})")
            _last_claude_fail = None
        return text
    except Exception:
        return None

# Tracks the most recent LLM subprocess so a new _claude_call can kill a
# lingering one from a prior cycle. Overlapping calls compete for the same API
# quota and cause *both* to time out. Process-local — does not guard against a
# second orphaned runner process, but covers stacked calls within one process
# (decision → retry → Sonnet fallback, or this cycle's call vs. the next).
_active_claude_proc: subprocess.Popen | None = None

# Monotonic timestamp of when the most recent ``_claude_call`` subprocess was
# launched (``time.monotonic()`` at Popen). Cleared back to None in the same
# ``finally`` block that resets ``_active_claude_proc``. Read by
# ``claude_call_state()`` so the operator surface (runner heartbeat) can show
# "Opus has been thinking for N minutes" while the deadman is patient — the
# distinguishing signal between "slow but live" and "wedged" the trader
# otherwise has no view of. Monotonic clock (not wall-clock) so a wall-clock
# step-back during a long call does not render a negative elapsed.
_active_claude_started_at: float | None = None


def is_claude_call_active() -> bool:
    """True when a `claude` subprocess this process started is still running.

    Used by the runner's git-watcher deadman to distinguish a *wedged* main
    loop (worth force-exiting) from a *legitimately slow* in-flight Opus call
    (must not be killed). Reads the live ``Popen.poll()`` — never raises, safe
    from the watcher thread. With ``DECISION_TIMEOUT_S = None`` (commit
    82bb195: "wait indefinitely for model response") a healthy decision can
    legitimately take longer than ``RESTART_GRACE_S``; without this signal
    the deadman would tear down the trader mid-thought. Returns False when no
    claude call has run this process or the prior one has already exited
    (the next cycle's deferred-restart check then proceeds normally)."""
    proc = _active_claude_proc
    if proc is None:
        return False
    try:
        return proc.poll() is None
    except Exception:
        return False


def claude_call_state() -> dict:
    """Best-effort snapshot of the currently-running ``_claude_call`` (if any).

    Surfaces three fields:
      * ``active`` — bool, identical to ``is_claude_call_active()`` (same
        ``Popen.poll() is None`` check).
      * ``elapsed_s`` — int seconds since the in-flight subprocess was
        launched (``time.monotonic`` baseline), or ``None`` when no call is
        active OR the start ts was never captured.
      * ``pid`` — process id of the in-flight subprocess, or ``None``.

    The runner heartbeat consumes this so the operator can distinguish
    "Opus has been thinking for 4 minutes" (healthy, indefinite-timeout
    decision in progress) from "engine is wedged" (no active call, just a
    stale ``secs_since_last_decision``). Under ``DECISION_TIMEOUT_S = None``
    those two states are otherwise indistinguishable from outside.

    Pure read of module globals — never raises, safe from any thread (the
    dashboard reads it from its request thread). Mirrors the discipline of
    ``runner.singleton_lock_state`` / ``runner.alarm_latch_state``.
    """
    proc = _active_claude_proc
    started = _active_claude_started_at
    active = False
    pid: int | None = None
    if proc is not None:
        try:
            active = proc.poll() is None
        except Exception:
            active = False
        try:
            pid = int(proc.pid) if active else None
        except Exception:
            pid = None
    elapsed_s: int | None = None
    if active and started is not None:
        try:
            elapsed_s = max(0, int(time.monotonic() - started))
        except Exception:
            elapsed_s = None
    return {
        "active": active,
        "elapsed_s": elapsed_s,
        "pid": pid,
    }

# Set True by _claude_call when the CLI rejects an attempt with a quota /
# usage-limit error (a *distinct* failure from a timeout or a parse miss — it
# never self-recovers within the cycle, and the auto-recovery circuit breaker's
# pkill is futile against it because the CLI already exited). decide() resets it
# at the top of every cycle and surfaces it as summary["quota_exhausted"] so
# runner._cycle can fire ONE Discord alarm per outage. _claude_call is called
# only from decide(), so a single module global is race-free here.

# Tight marker set — matched case-insensitively against the claude CLI's
# stdout/stderr ONLY on a non-zero exit. Kept precise so an unrelated failure
# (network blip, transient 5xx, parse miss) does NOT trip the trader-facing
# alarm. The observed live string is: "You've hit your org's monthly usage
# limit" (rc=1, empty stderr, message on stdout).
_QUOTA_MARKERS = ("usage limit", "quota exceeded", "quota exhausted",
                  "out of credit", "insufficient credit")


def _is_quota_exhausted(text: str | None) -> bool:
    """True if `text` looks like a quota / usage-limit rejection (not a
    transient error). See `_QUOTA_MARKERS`."""
    if not text:
        return False
    low = text.lower()
    return any(m in low for m in _QUOTA_MARKERS)

WATCHLIST = [
    "LITE", "LNOK", "MUU", "DRAM", "SNDU", "SNK",  # current real-account interests
    "NVDA", "AMD", "MU", "AMAT", "LRCX", "KLAC", "TSM", "ASML", "MRVL",  # semis
    "BIRD",  # ArticleNet short-watch: Allbirds/Smartbird distress examples
    "SMH", "SOXX", "SPY", "QQQ",  # ETFs
    # Leveraged ETFs — 3x Bull
    "TQQQ", "UPRO", "SPXL", "UDOW", "URTY",
    "SOXL", "TECL", "FNGU", "CURE", "LABU",
    "NAIL", "DFEN", "DPST", "FAS", "TNA", "UTSL",
    # Leveraged ETFs — 2x Bull
    # GOOGU / METAU removed 2026-05-18: both single-stock 2x ETFs are
    # permanently delisted (yfinance 404, no quote). Keeping them here only
    # told Opus two untradeable names were available and made
    # market.get_prices(WATCHLIST) re-404 every _DEAD_TTL window. Resolves
    # AGENTS.md review-pass-#18 core finding #3 (live side only — backtest.py's
    # historical universe is the ML-domain owner's call, deliberately untouched).
    "QLD", "SSO", "NVDU", "MSFU", "AMZU",
    "TSLL", "CONL", "BITU", "ETHU",
    # Leveraged Bear / Hedge
    "SQQQ", "SPXS", "SOXS", "TECS", "FNGD",
]

# Research-discovery surface. The curated WATCHLIST stays the price/portfolio
# spine, but the decision cycle also scans a much wider ArticleNet ticker set
# so fresh off-watchlist dislocations can be surfaced before they become
# obvious hindsight. Only the hottest discovered names are price/quant fetched.
RESEARCH_UNIVERSE_MULTIPLE = int(os.environ.get(
    "PAPER_TRADER_RESEARCH_UNIVERSE_MULTIPLE", "100",
))
DISCOVERY_SIGNAL_LIMIT = int(os.environ.get(
    "PAPER_TRADER_DISCOVERY_SIGNAL_LIMIT",
    str(max(len(WATCHLIST) * RESEARCH_UNIVERSE_MULTIPLE, 2000)),
))
DISCOVERY_SIGNAL_HOURS = int(os.environ.get(
    "PAPER_TRADER_DISCOVERY_SIGNAL_HOURS", "12",
))
DISCOVERY_SIGNAL_MIN_SCORE = float(os.environ.get(
    "PAPER_TRADER_DISCOVERY_SIGNAL_MIN_SCORE", "3.0",
))
DISCOVERY_PRICE_CANDIDATES = int(os.environ.get(
    "PAPER_TRADER_DISCOVERY_PRICE_CANDIDATES", "20",
))
DISCOVERY_QUANT_CANDIDATES = int(os.environ.get(
    "PAPER_TRADER_DISCOVERY_QUANT_CANDIDATES", "10",
))
ACTIVE_RESEARCH_HOURS = int(os.environ.get(
    "PAPER_TRADER_ACTIVE_RESEARCH_HOURS", "72",
))
ACTIVE_RESEARCH_SCAN_LIMIT = int(os.environ.get(
    "PAPER_TRADER_ACTIVE_RESEARCH_SCAN_LIMIT", "1600",
))
ACTIVE_RESEARCH_THEMES = [
    "AI capex balance sheet debt lease financing",
    "data center credit private debt SPV financing",
    "hyperscaler cloud spending guidance backlog",
    "Nasdaq big tech selloff recession risk",
    "semiconductor demand guidance inventory margin",
    "consumer weakness unemployment recession credit spreads",
]

# Subset used for live quant indicator computation. Mix of mega-caps + leveraged.
QUANT_TICKERS_LIVE = [
    "SPY", "QQQ", "NVDA", "AMD", "MU", "TSM", "AAPL", "MSFT", "META",
    "TQQQ", "SOXL", "LITE",
]

FUTURES = ["ES=F", "NQ=F", "CL=F", "GC=F"]

# Hard stop-loss / advisory take-profit configuration.
# Stocks: 5% SL / 15% advisory TP. Leveraged ETFs: 10% SL / 25% advisory TP
# (wider because leveraged ETF intraday volatility is 2-3x the underlying).
# Stop-loss is documented in SYSTEM_PROMPT ("HARD EXITS" section) so Opus is
# aware of the autonomy carve-out. Take-profit remains a per-position context
# target, but it is not mechanically enforced. The older 2/3 stock bracket
# repeatedly chopped momentum names like MU before the thesis had time to play.
_LEVERAGED_ETFS_SL = frozenset({
    "TQQQ", "SOXL", "UPRO", "SPXL", "UDOW", "URTY", "TECL", "FNGU",
    "LABU", "NAIL", "CURE", "DFEN", "HIBL", "MIDU", "TNA", "WANT",
    "FAS", "FOTU", "PILL", "RETL", "DPST", "KORU", "UTSL",
    "QLD", "SSO", "NVDU", "MSFU", "AMZU", "TSLL", "CONL", "BITU", "ETHU",
    "SOXS", "TECS", "SPXU", "SQQQ", "SPXS", "FNGD", "SNK",
})
_SL_PCT_STANDARD = 0.05
_TP_PCT_STANDARD = 0.15
_SL_PCT_LEVERAGED = 0.10
_TP_PCT_LEVERAGED = 0.25

SYSTEM_PROMPT = """You are managing a paper trading portfolio. Use the live
portfolio value, cash balance, positions, and buying-power fields in CONTEXT as
the source of truth for account size.
STANDING OPERATOR MANDATE (Jonathan 2026-08-07) — HARD:
- You are a DAY TRADER using real multi-source knowledge + leverage, not a
  passive office-worker portfolio manager buying boring ballast.
- Active goal: climb equity back to $10,000 (~+$640 from the hole), then stop
  pressing. Prefer defined-risk options (debit spreads / high-conviction
  calls-puts on NVDA/QQQ first) over adding more cash-bleed shares.
- First clean up dead weight: flatten CAT; treat AMZN as a trade not a career;
  free BP before new option debits. Do not buy CAT/AMZN/etc just because gross
  exposure is under 120% or a construction sleeve is underweight.
- Construction targets and "stay invested" pressure are SECONDARY to catalyst +
  tape. No catalyst today = HOLD, not nibble TQQQ/shares to look busy.
- If LLM/ML drought: HOLD or hard-exit only. Drought-fallback share dribbles
  are banned as a personality.
- Options risk budget while climbing to $10k: leftover Sep mark is NOT the cap.
  New 8/14 Friday weekly debit cap is $800. One NVDA/QQQ debit only. Engine
  flattens SK (wrong issuer) and sends the closer after Thu 08:45 ET PPI.
  Kill the new weekly only at about -50% debit or REAL thesis-kill evidence.
  Engine permanently blocks underwater panic sells.
If OPERATOR STANDING ORDERS appear in CONTEXT, they outrank generic
construction/deployment nagging when the two conflict.
Your ONLY goal is maximum profit after churn, concentration, and avoidable
drawdown. You have autonomy over timing, thesis, and sizing inside the hard
operator discipline enforced by the engine: no repeated same-ticker entry churn,
no oversized single-name or sector pile-on, and no oversized leveraged ETF bet.
You can:
- Trade like a day trader on ENTRIES and WINNERS. Do NOT panic-sell losers just
  because the candle is red. Engine blocks underwater discretionary exits unless
  hard stop, real thesis-kill evidence, or option premium -50% kill. If thesis is
  intact, HOLD and wait for recovery. Flatten ballast / take winners freely
- ISSUER MAP IS MANDATORY: never buy a ticker because a headline keyword matches
  the symbol letters. `TV` is Grupo Televisa, not "television" / FCC local-TV
  policy. `OPEN` is Opendoor, not "open source" / OpenAI. For short/ambiguous
  tickers, name the real issuer in reasoning or the engine will BLOCK the entry
- Read DURABLE TRADE MEMORY every cycle before acting. Remember your own recent
  orders and lessons. Do not amnesia-rebuy/resell the same ticker without new
  evidence that explicitly overrides the prior memory
- Concentrate when evidence is exceptional, but only within the hard engine caps
- Short priceable stocks when the ArticleNet/news evidence is bearish, even if
  the portfolio does not already own them
- Hold options through expiry if you believe in the thesis, or sell them early
  when the move is in / thesis is done / theta or IV crush is the bigger risk
- Idle cash is a failure mode. Keep cash under 10% of net worth unless every
  actionable idea is blocked by hard engine gates (issuer map, cooldown,
  concentration, no price). Prefer deploying into differentiated high-conviction
  names over parking cash
- Target 120%-150% gross long exposure vs net worth by using stock buying power
  (cash + 50% net-worth margin) and explicit stock leverage. Ideal ~135%. Do not
  sit at 0-1x if BP and thesis support more
- Use leverage smartly: prefer requested leverage 1.2x-3x on high-conviction
  liquid mega-caps when under-deployed; use 2-3x leveraged ETFs only as bounded
  sleeves for index/sector thrust, not as a cash-parking substitute
- Only stay heavy cash when the tape has no valid differentiated idea AND risk
  gates block every alternative — explain that explicitly in reasoning
- HARD RULE (Jonathan 2026-08-10, permanent): do not buy high and sell low.
  If a position goes red, HOLD and wait for it to go back up unless hard stop /
  real thesis-kill evidence / option -50% kill. No "largest drag", "weakening",
  or same-session panic flats on losers
- Take leveraged ETF positions (MUU, LNOK, etc.) only as bounded sleeves

LEVERAGE INSTRUMENTS AVAILABLE:
- Leveraged ETFs 3x Bull: TQQQ (QQQ), UPRO/SPXL (SPY), UDOW (Dow), URTY (Russell), SOXL (semis), TECL (tech), FNGU (FANGs), CURE (healthcare), LABU (biotech), NAIL (homebuilders), DPST (banks), FAS (financials), DFEN (defense), TNA (small-cap), UTSL (utilities)
- Leveraged ETFs 2x Bull: QLD (QQQ 2x), SSO (SPY 2x), NVDU (NVDA), MSFU (MSFT), AMZU (AMZN), TSLL (TSLA), CONL (COIN), LNOK (Nokia), BITU (BTC), ETHU (ETH)
- Leveraged Bear/Hedge: SQQQ/SPXS (3x short index), SOXS (3x short semis), TECS (3x short tech), FNGD (3x short FANGs), SNK (2x short SpaceX/SPCX)
- For high-conviction directional trades, consider 2-3x leveraged ETFs instead of the underlying
- For options-equivalent exposure: buy deep ITM LEAPS calls (delta >0.80) to simulate leveraged long
- OPTIONS ARE FIRST-CLASS INSTRUMENTS (same autonomy as stocks):
  * Read OPTIONS DESK in CONTEXT before choosing shares vs options.
  * Prefer BUY_CALL / BUY_PUT when conviction is high and defined premium risk
    is a better expression than full share notional (LEAPS proxy, swing call/put,
    event debit). Prefer shares when you need no expiry, SHORT borrow, or the chain
    is illiquid.
  * Always send strike + expiry (YYYY-MM-DD) on option actions.
  * Contract multiplier is 100: premium cost = mid * qty * 100.
  * Long option premium (BUY_CALL/BUY_PUT opens, buy-to-cover, debit verticals)
    uses the same buying power as stocks: cash + 50% net worth (1.5x when all cash).
    Cash may go negative within that BP limit.
  * SELL_CALL: close long calls OR open covered call if long >=100 sh/ct.
  * SELL_PUT: close long puts OR open cash-secured put (cash covers strike*100-premium).
  * BUY_CALL/BUY_PUT against short option lots = buy-to-cover.
  * Verticals: BUY_CALL_SPREAD / BUY_PUT_SPREAD / SELL_CALL_SPREAD / SELL_PUT_SPREAD
    with long_strike+short_strike (or strike+width) + expiry + qty.
  * Naked short calls blocked; use covered calls or call credit spreads.
  * Size premium budget against buying power explicitly in reasoning; do not ignore DTE/theta.
  * Use chain snapshots (IV, mid, delta proxy, cost/contract) when present.
Advanced option fields:
- Vertical spreads require ticker, qty, expiry, and either long_strike+short_strike or strike+width or legs:[{action,strike}x2].
- Covered call: SELL_CALL with strike+expiry while long stock covers contracts*100 shares.
- Cash-secured put: SELL_PUT with strike+expiry; cash must cover strike*100*qty - premium (CSP stays cash-secured; long premium may use margin BP).
- Short option qty is stored negative; BUY_* on that strike/expiry covers/closes short.
- Risk: leveraged ETFs decay in sideways markets; best for strong trending moves only

POSITION SIZING GUIDANCE:
- High conviction (RSI+MACD+MA all aligned): up to 50-70% portfolio, use leverage
  so the book moves toward 120%-150% gross exposure
- Medium conviction (2/3 signals aligned): 15-35%, still prefer deploying idle cash
  rather than leaving >10% cash
- Low conviction / leveraged ETF: size within the same hard caps (max 70%)
- Do not rebuy a ticker right after selling it unless a genuinely new catalyst appears
- Never go 100% into one ticker or leveraged ETF
- If CONTEXT shows cash_pct > 10% or gross_exposure < 120%, deploy only into a
  fresh high-conviction catalyst/options expression — or HOLD with an explicit
  reason. Do not fill the hole with boring ballast / construction underweights.
  Do not "wait and see" with a fat cash pile when a real idea exists either

THINK LIKE A HEDGE FUND MANAGER WHO WANTS ASYMMETRIC RETURNS.
Small, safe trades will not outperform. Recycled, correlated trades will not
outperform either. Take calculated risks across differentiated ideas.
High conviction = large size + smart leverage. Low conviction = small size, not
all-cash paralysis.

ACTIVE RESEARCH REQUIREMENT:
- Before every BUY, SELL, SHORT, COVER, option trade, or REBALANCE, first read
  ACTIVE RESEARCH DOSSIER plus ArticleNet/news, event, macro, technical, and
  portfolio-risk context.
- Do not treat ArticleNet as a passive feed. Use it as a research corpus: look
  for confirming and disconfirming evidence across sources, freshness, source
  concentration, balance-sheet/capex/credit risk, earnings guidance, and macro
  transmission.
- Local ML may be unavailable or stale on this Mac. Let your LLM reasoning do
  the research synthesis; ML scores are advisory metadata only.
- If the active research dossier is stale, thin, one-source, or contradicts the
  trade idea, prefer HOLD unless price/technical evidence is overwhelming and
  explain the exception in reasoning.
- For any AI/semis/Big Tech/index trade, explicitly account for AI-capex credit
  risk, balance-sheet obligations, hyperscaler guidance, and recession
  spillover before sizing.

HARD EXITS (AUTOMATIC — CANNOT BE OVERRIDDEN): New long stock positions are
automatically sold only during the regular market session when price falls 5%
below entry price (10% for leveraged ETFs). New short stock positions are
automatically covered when price rises 5% above entry (10% for leveraged ETFs).
The 15% / 25% take-profit level is advisory context only: when a position reaches
that level, decide dynamically from the stock's thesis, momentum, news, and prior
trade history whether to hold, trim, cover, or sell. Do not dump winners just
because a static take-profit marker was hit. Discretionary same-day exits are
gated by the minimum-hold lock unless a hard stop or explicit thesis-kill reason
is present. Same-ticker re-entry after an exit is cooldown-gated.

Respond with a SINGLE JSON object — no prose, no markdown fences. Schema:

{
  "action": "BUY" | "SELL" | "SHORT" | "COVER" | "BUY_CALL" | "BUY_PUT" | "SELL_CALL" | "SELL_PUT" | "BUY_CALL_SPREAD" | "BUY_PUT_SPREAD" | "SELL_CALL_SPREAD" | "SELL_PUT_SPREAD" | "HOLD" | "REBALANCE",
  "ticker": "NVDA",
  "qty": 0.5,
  "leverage": 1,               // optional for BUY stock only, 1-20x; effective exposure = qty * leverage
  "strike": 900,             // option single-leg OR spread base strike
  "expiry": "2026-05-30",    // option actions, YYYY-MM-DD
  "long_strike": 180,        // vertical spreads
  "short_strike": 190,       // vertical spreads
  "width": 10,               // alt to long/short_strike for verticals
  "confidence": 0.85,
  "reasoning": "1-3 sentences why"
}

Return JSON with your decision. No limits on qty, strike, or cash used.
For BUY on regular stocks, you may set "leverage" from 1x to 20x. The paper
trader applies the leverage to stock exposure only: qty=2, leverage=5 buys
10 effective shares. Buying power includes cash plus 50% margin on current
portfolio net worth (max ~1.5x when fully used). Prefer leverage >=1.2x when
cash is above 10% or gross exposure is below 120%, unless concentration or
issuer gates block it. The engine may auto-boost under-deployed tiny stock BUYs
toward the 120%-150% target within remaining buying power and hard caps.
For SHORT, qty is the positive share count to sell short; it opens or adds to
a short stock position and uses the same stock buying-power field as BUY.
For SELL, ticker must match an open long stock position. For COVER, ticker
must match an open short stock position. For SELL_CALL/SELL_PUT: closes a matching long option if present; otherwise opens a
covered call (requires long stock) or cash-secured put (requires cash collateral).
For BUY_CALL/BUY_PUT against a short option lot: buy-to-cover. Vertical spread actions
open both legs atomically (long + short) with defined risk.
Do not express a fresh bearish thesis as SELL unless there is an open long;
use SHORT for a priceable stock or a leveraged bear ETF for index/sector theses.
Do not short private, unquoted, or unmapped names.

TECHNICAL SIGNAL INTERPRETATION (use alongside news, not in isolation):
- RSI > 70 = overbought — avoid new longs, consider reducing; RSI < 30 = oversold — potential
  long opportunity if news/thesis supports it.
- MACD signal crossovers confirm momentum: positive macd_signal with rising price is bullish
  confirmation; negative macd_signal with falling price is bearish confirmation.
- MACD label "flat" means the MACD line is sitting on the signal line at machine precision — no
  momentum signal at all (a steady-state trend or a quiet tape). Treat "flat" as no MACD bias:
  do NOT count it as bullish OR bearish; conviction should come from RSI / news / catalyst alone.
- MACD histogram zero-cross while below zero: when hist_cross_up=True AND macd_below_zero_cross=True
  AND ema200_above=True, this is a HIGH-QUALITY entry signal — momentum is turning from oversold
  while the long-term trend is intact. Weight this heavily for new entries (it is the literal
  textbook 12/26/9 + EMA200-filter setup). The ⚡MACD_CROSS_NEG token in a quant line marks it.
- ema200_above=True means price is above the 200-day EMA — confirms long-term uptrend; prefer
  longs only when True. ema200_above=False = below EMA200 (no long-trend support; avoid new
  longs unless news catalyst is exceptional).
- Bollinger Band squeezes (bb_position near 0 after a tight range) often precede breakouts;
  bb_position approaching +1 or -1 means price is at the upper/lower Bollinger band (2 standard
  deviations from the 20-day mean) — stretched conditions and elevated reversal risk.
- Require volume confirmation for breakout trades: only trust a breakout when vol_ratio > 1.2.
  Low-volume breakouts often fail.
- Weight technical signals alongside news — neither alone is sufficient. A strong news catalyst
  with confirming technicals is high-conviction; a news catalyst that contradicts technicals
  (e.g. "beat earnings" on a stock at RSI 80 with bb_position +2) is a lower-conviction setup.

Return JSON ONLY.
"""


def _ema_live(values: list[float], period: int) -> list[float]:
    if len(values) < period:
        return []
    k = 2.0 / (period + 1)
    out = [sum(values[:period]) / period]
    for v in values[period:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def _rsi_live(closes: list[float], period: int = 14) -> float | None:
    if len(closes) <= period:
        return None
    gains = losses = 0.0
    for i in range(1, period + 1):
        diff = closes[i] - closes[i - 1]
        if diff >= 0:
            gains += diff
        else:
            losses -= diff
    avg_g = gains / period
    avg_l = losses / period
    for i in range(period + 1, len(closes)):
        diff = closes[i] - closes[i - 1]
        g = diff if diff > 0 else 0.0
        l = -diff if diff < 0 else 0.0
        avg_g = (avg_g * (period - 1) + g) / period
        avg_l = (avg_l * (period - 1) + l) / period
    if avg_l == 0:
        # Strict all-up returns 100. A perfectly FLAT series (no gain AND no
        # loss across the lookback) historically also returned 100 — a spurious
        # "severely overbought" signal that fed the wrong adjustment into the
        # ML advisor's `rsi > 67 → adj -= 1.5` arm for any flat name (a weekend
        # snapshot, a low-volume ticker, or any stretch where every close
        # matched the prior close). RSI is undefined at zero variance; the
        # textbook neutral reading is 50. Mirrors the same fix already applied
        # to backtest._rsi (pass #21, commit 9ee81b7) — the live `_rsi_live`
        # was the missed sibling.
        return 100.0 if avg_g > 0 else 50.0
    rs = avg_g / avg_l
    return 100.0 - (100.0 / (1.0 + rs))


def _macd_live(closes: list[float]) -> str | None:
    """Return ``"bullish"`` / ``"bearish"`` / ``"flat"`` (or None when there
    are too few closes).

    Epsilon-tolerant comparison so a steady-state linear trend (where the
    MACD line and signal line converge to the same value at machine
    precision) reports ``"flat"`` rather than flapping to bullish/bearish
    from EMA accumulation roundoff alone. The naive ``macd > signal``
    comparison polluted Opus's prompt with a false ``macd=bullish`` /
    ``macd=bearish`` label on flat or slow-trending names, silently
    misleading the decision engine. Mirrors the ``backtest._macd`` fix
    documented in AGENTS.md pass #38; the live ``_macd_live`` was the
    missed sibling."""
    if len(closes) < 35:
        return None
    ema12 = _ema_live(closes, 12)
    ema26 = _ema_live(closes, 26)
    if not ema12 or not ema26:
        return None
    offset = len(ema12) - len(ema26)
    macd_line = [ema12[i + offset] - ema26[i] for i in range(len(ema26))]
    if len(macd_line) < 9:
        return None
    signal = _ema_live(macd_line, 9)
    if not signal:
        return None
    m = macd_line[-1]
    s = signal[-1]
    # Tolerance scales with the magnitudes involved so real crossovers
    # (m - s well above the noise floor) are unaffected.
    tol = 1e-9 * max(abs(m), abs(s), 1.0)
    diff = m - s
    if diff > tol:
        return "bullish"
    if diff < -tol:
        return "bearish"
    return "flat"


_QUANT_CACHE: dict[str, tuple[dict, float]] = {}
_QUANT_TTL = 300.0  # 5 min — indicators change slowly intraday

# Negative cache for tickers whose yfinance lookup returned empty, raised, or
# came back with fewer than 60 closes (the floor get_quant_signals_live needs
# for indicator math). A WATCHLIST entry that has been delisted (GOOGU/METAU
# in 2026-05; future cleanups likewise) or a newly-IPO'd name without enough
# history would otherwise re-hit yfinance every cycle for the next 5 minutes
# of _QUANT_TTL — for each one. Bound by the same 5-min TTL as the positive
# cache so a legitimately-transient outage (yfinance hiccup, brief network
# blip) recovers cleanly on its next reload, while a stable-zombie never
# generates more than one fetch per TTL window. Mirrors market._DEAD_CACHE's
# discipline for the same problem one layer over (get_price's per-symbol
# negative cache). Pure module-state; no persistence.
_QUANT_NEG_CACHE: dict[str, float] = {}
_QUANT_NEG_TTL = 300.0


def _quant_neg_hit(ticker: str, now: float) -> bool:
    """True iff ``ticker`` is currently held in the negative cache.
    Expired entries self-evict on the next call. Pure helper — never raises."""
    ts = _QUANT_NEG_CACHE.get(ticker)
    if ts is None:
        return False
    if now - ts >= _QUANT_NEG_TTL:
        # Expired — drop it so a future success on this symbol isn't masked
        # by a stale negative entry.
        _QUANT_NEG_CACHE.pop(ticker, None)
        return False
    return True


def _quant_neg_mark(ticker: str, now: float) -> None:
    """Stamp a negative-cache entry for ``ticker`` at ``now``. Idempotent."""
    _QUANT_NEG_CACHE[ticker] = now


def _stdev_live(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return (sum((v - mean) ** 2 for v in values) / n) ** 0.5


def get_quant_signals_live(tickers: list[str]) -> dict[str, dict]:
    """Fetch ~1y of daily closes from yfinance for each ticker and compute
    RSI(14), MACD bullish/bearish, 50/200 MA cross, plus expanded signals:
    rsi, macd_signal, bb_position, mom_5d, mom_20d, vol_ratio, wk52_pos.
    Cached 5 minutes per ticker."""
    import time as _time
    import yfinance as yf
    out: dict[str, dict] = {}
    for t in tickers:
        cached = _QUANT_CACHE.get(t)
        now = _time.time()
        if cached and now - cached[1] < _QUANT_TTL:
            out[t] = cached[0]
            continue
        # Skip yfinance entirely if this symbol is held in the negative cache —
        # a recently-failed lookup (delisted ticker, empty history, network
        # error) re-hitting yfinance every cycle would just re-fail and spam
        # the log, the exact pathology market._DEAD_CACHE solved for get_price.
        if _quant_neg_hit(t, now):
            continue
        try:
            hist = yf.Ticker(t).history(period="1y", auto_adjust=False)
            if hist is None or hist.empty:
                _quant_neg_mark(t, now)
                continue
            closes = [float(c) for c in hist["Close"].tolist() if c == c]
            if len(closes) < 60:
                _quant_neg_mark(t, now)
                continue
            last = closes[-1]
            rsi = _rsi_live(closes, 14)
            macd_label = _macd_live(closes)
            if len(closes) >= 200:
                ma50 = sum(closes[-50:]) / 50
                ma200 = sum(closes[-200:]) / 200
                ma_cross = "golden" if ma50 > ma200 else "death"
            elif len(closes) >= 50:
                ma50 = sum(closes[-50:]) / 50
                ma_cross = "above50" if last > ma50 else "below50"
            else:
                ma_cross = None
            hi_52 = max(closes[-252:]) if len(closes) >= 252 else max(closes)
            lo_52 = min(closes[-252:]) if len(closes) >= 252 else min(closes)
            pct_h = (last - hi_52) / hi_52 * 100 if hi_52 else 0.0
            pct_l = (last - lo_52) / lo_52 * 100 if lo_52 else 0.0
            vol_ratio = None
            try:
                vols = [float(v) for v in hist["Volume"].tolist() if v == v]
                if len(vols) >= 21 and vols[-1] > 0:
                    avg20 = sum(vols[-21:-1]) / 20
                    if avg20 > 0:
                        vol_ratio = round(vols[-1] / avg20, 2)
            except Exception:
                pass

            # Expanded signals (lowercase keys per spec)
            macd_signal_val = None
            macd_hist = None
            macd_line_val = None
            hist_cross_up = False
            macd_below_zero_cross = False
            try:
                if len(closes) >= 35:
                    e12 = _ema_live(closes, 12)
                    e26 = _ema_live(closes, 26)
                    if e12 and e26:
                        offset = len(e12) - len(e26)
                        macd_line = [e12[i + offset] - e26[i] for i in range(len(e26))]
                        if len(macd_line) >= 9:
                            sig = _ema_live(macd_line, 9)
                            if sig:
                                macd_signal_val = round(sig[-1], 2)
                                # Align macd_line to the signal series length
                                # (sig is an EMA(9) of macd_line, so it's
                                # shorter). The last sig_len values of
                                # macd_line are the ones with a matching sig.
                                sig_len = len(sig)
                                ml_aligned = macd_line[-sig_len:]
                                macd_line_val = round(ml_aligned[-1], 4)
                                macd_hist = round(ml_aligned[-1] - sig[-1], 4)
                                if len(ml_aligned) >= 2 and len(sig) >= 2:
                                    hist_prev = ml_aligned[-2] - sig[-2]
                                    hist_curr = ml_aligned[-1] - sig[-1]
                                    hist_cross_up = bool(
                                        hist_prev < 0 and hist_curr > 0
                                    )
                                    macd_below_zero_cross = bool(
                                        hist_cross_up and ml_aligned[-1] < 0
                                    )
            except Exception:
                macd_signal_val = None
                macd_hist = None
                macd_line_val = None
                hist_cross_up = False
                macd_below_zero_cross = False

            # 200-day EMA filter — confirms long-term trend; prefer longs only
            # when price is above EMA200 (classic MACD strategy filter).
            ema200_above: bool | None = None
            try:
                if len(closes) >= 200:
                    e200 = _ema_live(closes, 200)
                    if e200:
                        ema200_above = bool(closes[-1] > e200[-1])
            except Exception:
                ema200_above = None

            bb_position = None
            try:
                if len(closes) >= 20:
                    window20 = closes[-20:]
                    sma20 = sum(window20) / 20
                    sd20 = _stdev_live(window20)
                    if sd20 > 0:
                        raw = (last - sma20) / (2 * sd20)
                        bb_position = round(max(-2.0, min(2.0, raw)), 2)
            except Exception:
                bb_position = None

            mom_5d = None
            try:
                if len(closes) >= 6 and closes[-6] > 0:
                    mom_5d = round((last - closes[-6]) / closes[-6] * 100, 2)
            except Exception:
                mom_5d = None
            mom_20d = None
            try:
                if len(closes) >= 21 and closes[-21] > 0:
                    mom_20d = round((last - closes[-21]) / closes[-21] * 100, 2)
            except Exception:
                mom_20d = None

            wk52_pos = None
            try:
                if hi_52 > lo_52:
                    wk52_pos = round((last - lo_52) / (hi_52 - lo_52), 2)
            except Exception:
                wk52_pos = None

            rec = {
                "RSI": round(rsi, 1) if rsi is not None else None,
                "MACD": macd_label,
                "MA_cross": ma_cross,
                "vol_ratio": vol_ratio,
                "pct_from_52h": round(pct_h, 1),
                "pct_from_52l": round(pct_l, 1),
                # Expanded fields per spec
                "rsi": round(rsi, 2) if rsi is not None else None,
                "macd_signal": macd_signal_val,
                "bb_position": bb_position,
                "mom_5d": mom_5d,
                "mom_20d": mom_20d,
                "wk52_pos": wk52_pos,
                # Enhanced MACD signals (12/26/9 + EMA200 filter) — high-quality
                # entry signal when hist crosses up while below zero AND price
                # sits above EMA200 (momentum turning from oversold inside an
                # intact long-term uptrend).
                "macd_hist": macd_hist,
                "macd_line_val": macd_line_val,
                "hist_cross_up": hist_cross_up,
                "macd_below_zero_cross": macd_below_zero_cross,
                "ema200_above": ema200_above,
            }
            _QUANT_CACHE[t] = (rec, _time.time())
            out[t] = rec
        except Exception as e:
            print(f"[strategy] quant signal fetch failed {t}: {e}")
            # Stamp the negative cache so a wedged / consistently-failing
            # symbol doesn't burn the next cycle's yfinance budget on the
            # same dead lookup.
            _quant_neg_mark(t, _time.time())
    return out


# bb_position is (last - sma20) / (2 * sd20): a value of ±1 sits exactly on
# the upper/lower Bollinger band (2σ from the 20-day mean). Opus reads the
# raw float in the prompt and otherwise has to mentally threshold it against
# the band — a labelled token surfaces the actionable state directly, the
# same render-side enrichment the `held=` / signal `age=` tokens already do
# (observational only; invariants #2/#12). Only the stretched extremes get a
# label — a mid-range reading carries none, the silence-when-nothing-
# actionable precedent.
_BB_BAND_THRESHOLD = 0.9

# RSI overbought / oversold rendering thresholds. The SYSTEM_PROMPT already
# tells Opus "RSI > 70 = overbought" and "RSI < 30 = oversold", but the
# prompt's quant block renders ``rsi=72.5`` as a bare float — Opus has to
# mentally re-threshold every row. Labelling the actionable extremes directly
# in the rendered token mirrors the ``_bb_label`` precedent (the same kind of
# render-side enrichment as ``held=`` / signal ``age=``), so Opus sees the
# stretched state immediately on the line it reads. Observational only
# (invariants #2/#12). Inclusive boundaries (``>=`` / ``<=``) match the
# prompt text ("RSI > 70" / "RSI < 30" round to the textbook 70 / 30 marks
# the SYSTEM_PROMPT names).
_RSI_OVERBOUGHT = 70.0
_RSI_OVERSOLD = 30.0


def _bb_label(x) -> str:
    """Render bb_position with a band annotation when stretched.

    ``None`` → ``"?"``. ``|x| >= _BB_BAND_THRESHOLD`` → ``"<x> (upper band)"``
    or ``"(lower band)"`` so Opus sees the stretched state without re-deriving
    it. Mid-range values render as the bare number. Degrade-safe — a
    non-numeric value falls through to its string form, never raises."""
    if x is None:
        return "?"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v >= _BB_BAND_THRESHOLD:
        return f"{x} (upper band)"
    if v <= -_BB_BAND_THRESHOLD:
        return f"{x} (lower band)"
    return str(x)


def _rsi_label(x) -> str:
    """Render RSI with an overbought/oversold annotation when extreme.

    ``None`` → ``"?"``. ``x >= _RSI_OVERBOUGHT`` (70) → ``"<x> (overbought)"``;
    ``x <= _RSI_OVERSOLD`` (30) → ``"<x> (oversold)"``. Mid-range values
    render as the bare number so a healthy book stays quiet — the
    silence-when-nothing-actionable precedent the ``_bb_label`` /
    ``_hold_age_str`` token enrichments already follow. Degrade-safe — a
    non-numeric value falls through to its string form, never raises (mirrors
    ``_bb_label`` exactly: the only legitimate non-numeric input is the
    pre-rendered ``"?"`` sentinel the ``_v`` helper emits when ``get`` finds
    no key, but a malformed cache row would otherwise crash decide())."""
    if x is None:
        return "?"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v >= _RSI_OVERBOUGHT:
        return f"{x} (overbought)"
    if v <= _RSI_OVERSOLD:
        return f"{x} (oversold)"
    return str(x)


def _format_quant_signals(sigs: dict[str, dict]) -> str:
    if not sigs:
        return "  (no quant signals available)"
    def _v(x):
        return "?" if x is None else x
    def _pct(x):
        return "?" if x is None else f"{x}%"
    def _ema200(x):
        if x is True:
            return "above"
        if x is False:
            return "below"
        return "?"
    def _cross_flag(q):
        # Only fire the loud token when both the up-cross AND the below-zero
        # condition hold — the classic "MACD turn from oversold" entry. Silent
        # otherwise so a healthy book stays quiet (the _bb_label precedent).
        return "  ⚡MACD_CROSS_NEG" if q.get("macd_below_zero_cross") else ""
    return "\n".join(
        f"  {tk}: rsi={_rsi_label(q.get('rsi'))}  macd={_v(q.get('MACD'))}/{_v(q.get('macd_signal'))}  "
        f"ma_cross={_v(q.get('MA_cross'))}  bb_position={_bb_label(q.get('bb_position'))}  "
        f"vol_ratio={_v(q.get('vol_ratio'))}  mom_5d={_pct(q.get('mom_5d'))}  "
        f"mom_20d={_pct(q.get('mom_20d'))}  "
        f"wk52_pos={_v(q.get('wk52_pos'))}  52h={_pct(q.get('pct_from_52h'))}  52l={_pct(q.get('pct_from_52l'))}  "
        f"ema200={_ema200(q.get('ema200_above'))}  macd_hist={_v(q.get('macd_hist'))}"
        f"{_cross_flag(q)}"
        for tk, q in sorted(sigs.items())
    )


def _claude_call(prompt: str, timeout_s: int = DECISION_TIMEOUT_S,
                 model: str = MODEL) -> str | None:
    global _active_claude_proc, _active_claude_started_at
    global _quota_exhausted, _last_claude_fail
    # Reset per-call so a success after a failure cannot leak the stale code
    # into the next cycle's diagnostic reason text.
    _last_claude_fail = None
    # HARD RULE 2026-08-04: Claude/Anthropic spend forbidden unless explicitly re-enabled.
    _m = (model or MODEL or "").strip().lower()
    if (
        _m.startswith("claude")
        or _m.startswith("anthropic/")
        or "sonnet" in _m
        or "opus" in _m
        or "haiku" in _m
        or (not _uses_xai_http(model) and not _m.startswith("gpt-"))
    ):
        if not _uses_xai_http(model) and not _m.startswith("gpt-"):
            print(
                f"[strategy] BLOCKED Claude/non-Grok model={model!r}; "
                "forcing grok-4.6 (Claude spend disabled 2026-08-04)"
            )
            model = "grok-4.6"
    if _uses_xai_http(model):
        # Primary: xAI SuperGrok. Fallback: Cursor CLI Grok (same model family).
        result = None
        if not _quota_exhausted:
            result = _xai_http_call(prompt, timeout_s=timeout_s, model=model)
            if result:
                return result
        elif _cursor_fallback_enabled():
            print(
                "[strategy] xAI quota flag set; trying Cursor Grok fallback "
                f"(model={CURSOR_MODEL})"
            )
        if _cursor_fallback_enabled():
            return _cursor_http_call(prompt, timeout_s=timeout_s)
        return None
    use_codex = model.startswith("gpt-")
    cli = "codex" if use_codex else "claude"
    cli_bin = _cli_path(cli)
    if not cli_bin:
        if use_codex and CODEX_AUTH_FALLBACK_MODEL and CODEX_AUTH_FALLBACK_MODEL != model:
            print(
                "[strategy] codex CLI not found; retrying decision with "
                f"{CODEX_AUTH_FALLBACK_MODEL}"
            )
            return _claude_call(
                prompt,
                timeout_s=timeout_s,
                model=CODEX_AUTH_FALLBACK_MODEL,
            )
        print(f"[strategy] {cli} CLI not found")
        _last_claude_fail = "cli_missing"
        return None
    # An LLM call still alive from a prior cycle competes for the same API
    # quota as this one and makes *both* time out. Kill the stale one first.
    if _active_claude_proc is not None and _active_claude_proc.poll() is None:
        print("[strategy] killing stale claude subprocess before new call")
        try:
            _active_claude_proc.kill()
            _active_claude_proc.wait(timeout=5)
        except Exception as e:
            print(f"[strategy] failed killing stale claude proc: {e}")
    _active_claude_proc = None
    _active_claude_started_at = None
    try:
        proc = subprocess.Popen(
            ([
                cli_bin, "exec",
                "--model", model,
                "-c", 'model_reasoning_effort="none"',
                "--sandbox", "read-only",
                "--cd", str(Path(__file__).resolve().parents[1]),
                "--ephemeral",
                "--color", "never",
                "-",
            ] if use_codex else [
                cli_bin, "--model", model, "--print",
                "--permission-mode", "bypassPermissions",
            ]),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_llm_subprocess_env(use_codex),
        )
        _active_claude_proc = proc
        _active_claude_started_at = time.monotonic()
        try:
            stdout, stderr = proc.communicate(input=prompt, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
            print(f"[strategy] {cli} timeout after {timeout_s}s")
            _last_claude_fail = "timeout"
            return None
        finally:
            _active_claude_proc = None
            _active_claude_started_at = None
        if proc.returncode != 0:
            # The model CLI may exit non-zero with an EMPTY stderr and
            # the real error written to stdout, which produced useless blank
            # "[strategy] claude err:" lines in runner.log (the operator could
            # not tell why decisions were being skipped). Log the returncode
            # plus a stdout tail so the failure is actually diagnosable.
            err = (stderr or "").strip()[:300]
            if not err:
                err = f"(empty stderr) stdout={(stdout or '').strip()[:200]!r}"
            # Quota / usage-limit rejection is a distinct, non-self-recovering
            # failure: flag it so decide() can surface it and runner._cycle
            # can alarm the operator once (the bot is silently frozen).
            combined = f"{stdout or ''}\n{stderr or ''}"
            low_combined = combined.lower()
            if (
                use_codex
                and CODEX_AUTH_FALLBACK_MODEL
                and CODEX_AUTH_FALLBACK_MODEL != model
                and (
                    "401 unauthorized" in low_combined
                    or "missing bearer" in low_combined
                    or "authentication" in low_combined
                )
            ):
                print(
                    "[strategy] codex auth failed; retrying decision with "
                    f"{CODEX_AUTH_FALLBACK_MODEL}"
                )
                return _claude_call(
                    prompt,
                    timeout_s=timeout_s,
                    model=CODEX_AUTH_FALLBACK_MODEL,
                )
            if _is_quota_exhausted(combined):
                _quota_exhausted = True
                print(f"[strategy] {cli} QUOTA EXHAUSTED (rc={proc.returncode}): {err}")
                # Don't tag _last_claude_fail — decide() takes a dedicated
                # branch on _quota_exhausted with its own reason text.
            else:
                print(f"[strategy] {cli} err (rc={proc.returncode}): {err}")
                _last_claude_fail = "nonzero_rc"
            return None
        text = (stdout or "").strip()
        if not text:
            # CLI exited cleanly but produced no output — distinct from a
            # timeout (the model itself returned empty within budget).
            _last_claude_fail = "empty_stdout"
            return None
        return text
    except Exception as e:
        print(f"[strategy] {cli} exception: {e}")
        _active_claude_proc = None
        _active_claude_started_at = None
        _last_claude_fail = "exception"
        return None


_RETRY_SUFFIX = (
    "\n\nYour previous response could not be parsed as JSON. "
    "Reply with the JSON decision object ONLY — no prose, no markdown fences, "
    "no commentary before or after. Start your response with `{` and end with `}`."
)


def _should_retry_parse(raw: str | None) -> bool:
    """Retry only when Claude actually returned text we couldn't parse.

    A None response means timeout / CLI error / empty stdout — retrying the
    same prompt would just hit the same wall. A non-empty raw that fails to
    parse suggests prose-wrapping or truncation, which a stronger JSON-only
    nudge can often rescue."""
    if not raw:
        return False
    return "{" not in raw or _parse_decision(raw) is None


def _parse_decision(raw: str) -> dict | None:
    if not raw:
        return None
    # strip ```json fences if model ignored instructions
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # Walk to the first '{' and use raw_decode so trailing text after the
    # JSON object doesn't break parsing (greedy regex was over-matching).
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "action" in obj:
            return obj
    print(f"[strategy] JSON parse failed; raw: {text[:300]}")
    return None


def _option_expired(expiry: str | None, today: date | None = None,
                    now: datetime | None = None) -> bool:
    """True if the option's expiry has passed the NYSE session close.

    Expiry day flips to expired at ``market.close_minute`` (16:00 ET regular
    / 13:00 ET half-day), NOT at UTC midnight. Previously the comparison was
    ``exp < datetime.now(timezone.utc).date()``, which kept an expired
    contract marked at avg_cost (with ``stale_mark=True``) for ~3-4h after
    the actual bell — every monthly expiry, the dashboard and the decision
    prompt's position lines both showed an OTM option at full premium
    instead of $0 (or its true intrinsic) from 16:00 ET until UTC midnight.
    Same window let ``_execute`` SELL_CALL/SELL_PUT settle a closed expired
    contract at ``avg_cost`` if the chain returned None, breaking-even a
    worthless leg. Documented in AGENTS.md review pass #33 (impact: stale
    display only, market closed in the window so no trade fires on the
    wrong mark — but the fix was deferred and is now applied).

    ``today`` preserves the legacy "pretend today is this date" date-only
    override used by the 6 pre-existing pin tests (no time-of-day check —
    expiry day itself counts as not expired). ``now`` is the new
    NY-tz-aware injection point; given a UTC- or NY-aware datetime, the
    function does the same logic the production wall-clock path runs.
    Both unset → real wall clock + NY tz + close gate."""
    if not expiry:
        return False
    try:
        exp = date.fromisoformat(str(expiry)[:10])
    except (TypeError, ValueError):
        return False
    if today is not None:
        return exp < today
    now_dt = now or datetime.now(timezone.utc)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)
    now_ny = now_dt.astimezone(market.NY)
    today_ny = now_ny.date()
    if exp < today_ny:
        return True
    if exp > today_ny:
        return False
    # exp == today_ny: expired at/after the NYSE session close.
    close_min = market.close_minute(today_ny)
    cur_min = now_ny.hour * 60 + now_ny.minute
    return cur_min >= close_min


def _hold_age_str(opened_at: str | None, now: datetime | None = None) -> str:
    """Compact hold age for a position: ``"42m"`` / ``"5h"`` / ``"3d"``.

    Opus's prompt position lines historically showed only qty/avg/mark/P/L —
    *how long* a position has been held was invisible to the decision engine,
    yet the desk's #1 documented pathology is the disposition effect (riding
    losers, cutting winners early; live: LITE held ~3.8d at a loss, 7.12× the
    empirical median losing hold, with Opus blind to the age). ``opened_at`` is
    already carried on every ``snap["positions"]`` row (it is reset to the
    re-entry instant when a fully-closed lot is reactivated — see
    ``store.upsert_position`` — so it is the correct "current holding period",
    not the all-time first touch).

    Pure and degrade-safe (the ``stale_mark`` precedent — invariants #2/#12):
    a missing / unparseable ``opened_at`` returns ``""`` (the caller renders no
    token, byte-identical to today for any snapshot without the field, e.g. the
    handcrafted test snapshots). Day flooring matches ``dashboard.
    _position_ages_from_trades`` / ``/api/risk`` so the two surfaces never
    disagree by a day. A future ``opened_at`` (wall-clock stepped back — the
    documented clock-skew hazard) clamps to ``"0m"`` rather than rendering a
    negative age. ``now`` is injectable for deterministic tests."""
    if not opened_at:
        return ""
    try:
        dt = datetime.fromisoformat(str(opened_at).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    secs = max(0.0, (now - dt).total_seconds())
    if secs < 3600:
        return f"{int(secs // 60)}m"
    if secs < 86400:
        return f"{int(secs // 3600)}h"
    return f"{int(secs // 86400)}d"


def _signal_age_str(first_seen: str | None, now: datetime | None = None) -> str:
    """Compact news-signal age in whole minutes: ``"5m"`` / ``"92m"``.

    The TOP SCORED SIGNALS / urgent lines Opus reads in the decision prompt
    historically showed ai_score + urgency + title only — *how fresh* a
    catalyst is was invisible, yet a headline that broke 5 min ago and one
    115 min ago (both inside the 2h ``get_top_signals`` window) are very
    different trades: the stale one has likely already moved the tape. Both
    ``signals.get_top_signals`` and ``signals.get_urgent_articles`` already
    carry ``first_seen`` (an ISO-8601 insert timestamp), so this is a pure
    render-side enrichment of data the cycle already fetched.

    Minute granularity on purpose — never rolls up to hours/days the way
    ``_hold_age_str`` does: the live signal windows are short (top signals
    ≤2h, urgent ≤30m) and minute precision is exactly the freshness signal
    that matters. Degrade-safe (the ``stale_mark`` / ``_hold_age_str``
    precedent; invariants #2/#12): a missing / unparseable ``first_seen``
    returns ``""`` so the caller renders no token, byte-identical to the
    pre-feature prompt for any signal row lacking the field (e.g. the
    handcrafted test signals). A future ``first_seen`` (wall-clock stepped
    back — the documented clock-skew hazard) clamps to ``"0m"`` rather than
    a negative age. ``now`` is injectable for deterministic tests."""
    if not first_seen:
        return ""
    try:
        dt = datetime.fromisoformat(str(first_seen).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    secs = max(0.0, (now - dt).total_seconds())
    return f"{int(secs // 60)}m"


def _expired_intrinsic(ticker: str, otype: str, strike: float) -> float:
    """Cash-settlement value (per share) of an *expired* option: its intrinsic
    value against the current underlying. 0.0 when out-of-the-money or the
    underlying price is unavailable. An expired option is never worth its
    purchase premium — falling back to avg_cost would mark a worthless
    contract at full cost forever and silently inflate equity."""
    try:
        und = market.get_price(ticker)
    except Exception:
        und = None
    if not und or und <= 0:
        return 0.0
    try:
        k = float(strike)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, und - k) if otype == "call" else max(0.0, k - und)


def _mark_to_market(
    positions: list[dict],
) -> tuple[list[dict], float, dict[int, tuple[float, float]]]:
    """Pure mark-to-market over the given open-position rows.

    Returns ``(enriched, open_value, marks)``; performs **no** store writes.
    Single source of truth for the mark math (invariant #10): the
    write-through ``_portfolio_snapshot`` and the read-only
    ``portfolio_snapshot_readonly`` both call it, so the live trader and the
    ``/api/decision-context`` inspector can never disagree on a mark — incl.
    the expired-option intrinsic settlement (invariant #13) and the
    ``stale_mark`` flag. yfinance lookups happen here exactly as before
    (behaviour-preserving extraction, locked by
    ``tests/test_core_strategy.py::TestPortfolioSnapshot*``)."""
    stock_tickers = sorted({p["ticker"] for p in positions if p["type"] == "stock"})
    prices = market.get_prices(stock_tickers) if stock_tickers else {}

    marks: dict[int, tuple[float, float]] = {}
    enriched = []
    open_value = 0.0
    for p in positions:
        # `stale_mark` is True when the *live* price lookup returned nothing
        # and we fell back to avg_cost: current_price == avg_cost and
        # unrealized_pl == 0.0 then look exactly like a genuinely *flat*
        # position, so a trader (and Opus) silently treats an UNKNOWN mark as
        # FLAT. This was seen live (MU: avg_cost == current_price == 724.12,
        # P/L $0.00). An expired-option intrinsic settlement is a *deliberate*
        # real mark, not a missing price, so it is never flagged stale.
        stale = False
        if p["type"] in ("call", "put"):
            multiplier = 100
            # An expired contract has no live chain — yfinance returns nothing,
            # so settle it at intrinsic against the underlying instead of
            # letting the avg_cost fallback below mark it at full premium.
            if _option_expired(p["expiry"]):
                cur = _expired_intrinsic(p["ticker"], p["type"], p["strike"])
            else:
                cur = market.get_option_price(p["ticker"], p["expiry"], p["strike"], p["type"])
                stale = cur is None
        else:
            cur = prices.get(p["ticker"])
            stale = cur is None
            multiplier = 1
        # `is not None`, not `or`: a legitimate 0.0 (expired worthless option)
        # must survive — `cur or avg_cost` would clobber it back to premium.
        cur = cur if cur is not None else p["avg_cost"]
        qty = float(p["qty"] or 0.0)
        pl = (cur - p["avg_cost"]) * qty * multiplier
        if p["avg_cost"]:
            direction = -1.0 if qty < 0 else 1.0
            pl_pct = ((cur - p["avg_cost"]) / p["avg_cost"]) * 100 * direction
        else:
            pl_pct = 0.0
        marks[p["id"]] = (cur, pl)
        enriched.append({**p, "current_price": cur, "unrealized_pl": pl, "pl_pct": pl_pct,
                         "market_value": cur * qty * multiplier,
                         "stale_mark": stale})
        open_value += cur * qty * multiplier

    return enriched, open_value, marks


def _settle_expired_option_lots(store: Store, enriched: list[dict]) -> set[int]:
    """Cash-settle expired option lots and close them off the open book.

    Expired contracts were already being marked at intrinsic (usually $0) and
    then left open forever, so hourlies kept printing dead -100% rows. A close
    at that mark moves any remaining intrinsic into cash and removes the lot
    from ``open_positions`` / the hourly list. OTM longs and shorts are a $0
    cash move; ITM longs credit intrinsic; ITM shorts debit it. Equity is
    unchanged because the same dollars were already in ``market_value``.
    """
    settled: set[int] = set()
    for pos in enriched:
        if pos.get("type") not in ("call", "put"):
            continue
        if not _option_expired(pos.get("expiry")):
            continue
        qty = float(pos.get("qty") or 0.0)
        if abs(qty) <= 0.0001:
            continue
        try:
            settle_px = float(pos.get("current_price") or 0.0)
        except (TypeError, ValueError):
            settle_px = 0.0
        if settle_px < 0:
            settle_px = 0.0
        ticker = pos["ticker"]
        otype = pos["type"]
        expiry = pos.get("expiry")
        try:
            strike_f = float(pos["strike"]) if pos.get("strike") is not None else None
        except (TypeError, ValueError):
            strike_f = None
        store.record_trade(
            ticker, "EXPIRE", abs(qty), settle_px,
            reason="expired option settled and removed from open book",
            expiry=expiry, strike=strike_f, option_type=otype,
        )
        store.upsert_position(
            ticker, otype, -qty, settle_px,
            expiry=expiry, strike=strike_f,
        )
        pf = store.get_portfolio()
        store.update_portfolio(
            float(pf["cash"]) + settle_px * qty * 100.0,
            float(pf["total_value"]),
        )
        if pos.get("id") is not None:
            settled.add(pos["id"])
    return settled


def _portfolio_snapshot(store: Store) -> dict:
    """Mark-to-market every open position, write back to DB, return summary."""
    positions = store.open_positions()
    enriched, open_value, marks = _mark_to_market(positions)

    if marks:
        store.update_position_marks(marks)

    settled_ids = _settle_expired_option_lots(store, enriched)
    if settled_ids:
        enriched = [p for p in enriched if p.get("id") not in settled_ids]
        open_value = sum(float(p.get("market_value") or 0.0) for p in enriched)

    pf = store.get_portfolio()
    total = pf["cash"] + open_value
    margin_available = max(0.0, total) * STOCK_MARGIN_NET_WORTH_PCT
    stock_buying_power = pf["cash"] + margin_available
    store.update_portfolio(pf["cash"], total, [
        {k: v for k, v in pos.items() if k != "opened_at"} for pos in enriched
    ])
    return {
        "cash": pf["cash"],
        "total_value": total,
        "open_value": open_value,
        "margin_available": margin_available,
        "stock_buying_power": stock_buying_power,
        "positions": enriched,
    }


def portfolio_snapshot_readonly(store: Store) -> dict:
    """Same mark-to-market as ``_portfolio_snapshot`` but **never writes to
    the store**.

    Used by ``/api/decision-context`` (and its CLI) so an operator can see
    exactly what ``decide()`` would feed Opus *right now* without the
    dashboard thread mutating the live trader's persisted marks /
    equity_curve. Shares ``_mark_to_market`` with the live path so the
    inspector's positions can never drift from the real ones (single source
    of truth, invariant #10). Read-only: ``open_positions()`` +
    ``get_portfolio()`` only — no ``update_position_marks`` /
    ``update_portfolio``."""
    positions = store.open_positions()
    enriched, open_value, _marks = _mark_to_market(positions)
    pf = store.get_portfolio()
    total = pf["cash"] + open_value
    margin_available = max(0.0, total) * STOCK_MARGIN_NET_WORTH_PCT
    return {
        "cash": pf["cash"],
        "total_value": total,
        "open_value": open_value,
        "margin_available": margin_available,
        "stock_buying_power": pf["cash"] + margin_available,
        "positions": enriched,
    }


def _names_in_play(positions: list[dict], top_signals: list[dict],
                   watchlist: list[str]) -> set[str]:
    """Tickers that are actionable *this cycle*: held, mentioned in the top-10
    signals, or top-5 watchlist priority (list order is the priority signal —
    first entries are the current real-account interests).

    Single source of truth for "what matters this cycle": the quant-block
    filter and the track-record block both call this so the two prompt
    sections can never disagree on which names are in play.
    """
    held = {p["ticker"] for p in positions}
    mentioned = {
        t for s in top_signals[:10] for t in (s.get("tickers") or [])
    }
    priority = set(watchlist[:5])
    return held | mentioned | priority


def _research_discovery(
    articles: list[dict],
    watchlist: list[str],
    *,
    target_size: int | None = None,
    top_n: int = 20,
) -> dict:
    """Rank live-news tickers outside the curated watchlist.

    This is the 100x-broader research layer: scan a much larger ArticleNet
    article batch, extract every ticker the feed names, and only pass the
    hottest discovered symbols forward for price/quant work. The target is a
    universe-size contract, not a mandate to yfinance-fetch thousands of names
    every cycle.
    """
    target = target_size or max(len(watchlist) * RESEARCH_UNIVERSE_MULTIPLE, 0)
    watch = {str(t).upper() for t in watchlist if t}
    heat: dict[str, dict[str, float | int | str | None]] = {}
    for a in articles or []:
        try:
            score = float(a.get("ai_score") or a.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        try:
            urg = float(a.get("urgency") or 0.0)
        except (TypeError, ValueError):
            urg = 0.0
        article_heat = max(0.0, score) + max(0.0, urg) * 1.5
        title = str(a.get("title") or "")
        for raw in a.get("tickers") or []:
            tk = str(raw or "").upper().strip()
            if not tk or tk in watch:
                continue
            # Belt-and-suspenders: never promote article false-positives into
            # the priced research universe (GAAP/CNBC/LLC etc.).
            try:
                from paper_trader.signals import _NOT_TICKERS as _junk
                if tk in _junk:
                    continue
            except Exception:
                pass
            row = heat.setdefault(tk, {
                "ticker": tk,
                "heat": 0.0,
                "n": 0,
                "max_score": 0.0,
                "max_urgency": 0.0,
                "sample_title": None,
            })
            row["heat"] = float(row["heat"]) + article_heat
            row["n"] = int(row["n"]) + 1
            row["max_score"] = max(float(row["max_score"]), score)
            row["max_urgency"] = max(float(row["max_urgency"]), urg)
            if not row["sample_title"]:
                row["sample_title"] = title[:140]

    ranked = sorted(
        heat.values(),
        key=lambda r: (
            float(r["heat"]),
            float(r["max_score"]),
            float(r["max_urgency"]),
            int(r["n"]),
            str(r["ticker"]),
        ),
        reverse=True,
    )
    return {
        "target_size": target,
        "watchlist_size": len(watch),
        "multiple": RESEARCH_UNIVERSE_MULTIPLE,
        "n_articles_scanned": len(articles or []),
        "observed_off_watchlist": len(ranked),
        "top": ranked[:top_n],
    }


def _research_discovery_prompt_block(discovery: dict | None) -> str | None:
    if not discovery:
        return None
    rows = discovery.get("top") or []
    if not rows:
        return (
            "RESEARCH DISCOVERY: "
            f"target={discovery.get('target_size', 0)} tickers "
            f"({discovery.get('multiple', RESEARCH_UNIVERSE_MULTIPLE)}x watchlist), "
            f"scanned={discovery.get('n_articles_scanned', 0)} articles; "
            "no off-watchlist ticker heat cleared extraction."
        )
    lines = [
        "RESEARCH DISCOVERY: "
        f"target={discovery.get('target_size', 0)} tickers "
        f"({discovery.get('multiple', RESEARCH_UNIVERSE_MULTIPLE)}x watchlist), "
        f"scanned={discovery.get('n_articles_scanned', 0)} articles, "
        f"observed_off_watchlist={discovery.get('observed_off_watchlist', 0)}.",
        "Hottest off-watchlist names to consider if priceable:",
    ]
    for r in rows[:8]:
        lines.append(
            f"  {r['ticker']}: heat={float(r['heat']):.1f} "
            f"n={int(r['n'])} max_score={float(r['max_score']):.1f} "
            f"urg={float(r['max_urgency']):.1f} "
            f"title={str(r.get('sample_title') or '')[:100]}"
        )
    return "\n".join(lines)


def _active_research_prompt_block(brief: dict | None) -> str | None:
    """Render the per-cycle ArticleNet dossier the LLM must research first."""
    if not brief:
        return None
    if not brief.get("available"):
        reason = brief.get("reason") or "unavailable"
        return (
            "ACTIVE RESEARCH DOSSIER: UNAVAILABLE "
            f"({reason}). Treat ArticleNet as blind this cycle; avoid fresh "
            "risk unless the rest of the context is overwhelming."
        )
    articles = brief.get("articles") or []
    window = brief.get("window_hours")
    newest = brief.get("newest_first_seen") or "n/a"
    scanned = brief.get("scanned", 0)
    sources = brief.get("source_counts") or {}
    src_line = ", ".join(
        f"{src}:{cnt}" for src, cnt in list(sources.items())[:8]
    ) or "none"
    lines = [
        "ACTIVE RESEARCH DOSSIER:",
        f"  window={window}h scanned={scanned} matched={len(articles)} "
        f"newest_first_seen={newest}",
        f"  source_counts={src_line}",
        "  mandate=Before trading, synthesize these ArticleNet/online-source "
        "hits with macro, credit, earnings, and technical context; if evidence "
        "is thin or one-sided, HOLD.",
    ]
    themes = brief.get("themes") or []
    if themes:
        lines.append("  active_themes=" + " | ".join(str(t) for t in themes[:6]))
    if not articles:
        lines.append(
            "  no matching research hits; this is a yellow flag for any fresh "
            "trade whose thesis depends on news/AI-credit/macro confirmation."
        )
        return "\n".join(lines)
    for a in articles[:10]:
        matches = ",".join(str(m) for m in (a.get("matches") or [])[:5])
        tickers = ",".join(str(t) for t in (a.get("tickers") or [])[:5])
        lines.append(
            f"  [{float(a.get('ai_score') or 0.0):.1f}] "
            f"urg={int(a.get('urgency') or 0)} "
            f"src={a.get('source') or 'unknown'} "
            f"seen={a.get('first_seen') or 'n/a'} "
            f"match={matches or 'n/a'} "
            f"tickers={tickers or 'n/a'} "
            f"title={str(a.get('title') or '')[:130]}"
        )
    return "\n".join(lines)


def _build_payload(snapshot: dict, top_signals: list[dict], sentiments: list[dict],
                   watch_prices: dict[str, float | None],
                   futures_prices: dict[str, float | None],
                   sp500: float | None, market_open: bool,
                   quant_signals: dict[str, dict] | None = None,
                   self_review_block: str | None = None,
                   track_record_block: str | None = None,
                   durable_memory_block: str | None = None,
                   repeat_loser_block: str | None = None,
                   thesis_drift_block: str | None = None,
                   risk_mirror_block: str | None = None,
                   sector_exposure_block: str | None = None,
                   stress_block: str | None = None,
                   portfolio_construction_block: str | None = None,
                   event_calendar_block: str | None = None,
                   macro_calendar_block: str | None = None,
                   buying_power_block: str | None = None,
                   exit_proximity_block: str | None = None,
                   research_discovery_block: str | None = None,
                   active_research_block: str | None = None,
                   options_desk_block: str | None = None) -> str:
    now = datetime.now(timezone.utc).isoformat()
    # Granular trading-day phase (see market.market_phase). The header
    # historically carried only the binary MARKET_OPEN — but a decision at
    # the opening bell, the closing half-hour, or mid-session live in three
    # very different liquidity/spread regimes. Surfacing the phase lets Opus
    # calibrate conviction directly. Degrade-safe: any market.py fault leaves
    # the phase token blank (the line still ships, byte-identical to before
    # for the MARKET_OPEN line that follows).
    try:
        phase = market.market_phase()
    except Exception as e:
        print(f"[strategy] market_phase failed (non-fatal): {e}")
        phase = ""
    pos_lines = []
    for p in snapshot["positions"]:
        # When the live price was unavailable the position is marked at cost,
        # so mark==avg and P/L reads $0.00 — indistinguishable from a genuinely
        # flat position. Tell Opus explicitly so it does not size a trade
        # against a phantom-flat mark (advisory text only; invariants #2/#12).
        stale_suffix = (
            "  [STALE MARK: live price unavailable — shown at cost, P/L unreliable]"
            if p.get("stale_mark") else ""
        )
        # How long this lot has been held. Observational only (the stale_mark
        # precedent; invariants #2/#12) — surfaces the raw fact the decision
        # engine was blind to so Opus can self-check the disposition effect
        # (the #1 documented pathology). Degrade-safe: no opened_at → no token,
        # byte-identical to before for any snapshot lacking the field.
        age = _hold_age_str(p.get("opened_at"))
        age_token = f" held={age}" if age else ""
        if p["type"] in ("call", "put"):
            side = "SHORT " if float(p.get("qty") or 0) < 0 else ""
            pos_lines.append(
                f"  {side}{p['ticker']} {p['type'].upper()} {p['strike']} {p['expiry']}: "
                f"qty={p['qty']} avg={p['avg_cost']:.2f} mark={p['current_price']:.2f} "
                f"P/L=${p['unrealized_pl']:.2f} ({p['pl_pct']:.1f}%){age_token}{stale_suffix}"
            )
        else:
            pos_lines.append(
                f"  {p['ticker']} {p['type']}: qty={p['qty']} avg={p['avg_cost']:.2f} "
                f"mark={p['current_price']:.2f} P/L=${p['unrealized_pl']:.2f} "
                f"({p['pl_pct']:.1f}%){age_token}{stale_suffix}"
            )

    sig_lines = []
    for s in top_signals[:10]:
        # How long ago this catalyst broke. Observational only (the
        # position `held=` token precedent; invariants #2/#12) — surfaces
        # the freshness fact the decision engine was blind to so Opus can
        # discount a signal the tape has likely already absorbed.
        # Degrade-safe: no/bad first_seen → no token, byte-identical to
        # before for any signal row lacking the field.
        s_age = _signal_age_str(s.get("first_seen"))
        age_token = f" age={s_age}" if s_age else ""
        sig_lines.append(
            f"  [{s['ai_score']:.1f}] urg={s['urgency']}{age_token} {s['title'][:120]}"
            + (f"  tickers={','.join(s['tickers'][:5])}" if s['tickers'] else "")
        )

    # Cap the quant block to keep the prompt lean: only tickers that are
    # (a) held, (b) mentioned in the top signals, or (c) top-5 watchlist
    # priority (list order is the priority signal — first entries are the
    # current real-account interests). Drops the long tail of curated
    # tickers that aren't actionable this cycle. Mirrors the existing n>0
    # sentiment filter below — shrink the rendered string, not the fetch.
    if quant_signals:
        keep = _names_in_play(snapshot["positions"], top_signals, WATCHLIST)
        quant_signals = {
            tk: q for tk, q in quant_signals.items() if tk in keep
        }

    sent_lines = [
        f"  {r['ticker']:>6}: avg={r['avg_score']:.1f} n={r['n']} urgent={r['urgent']}"
        for r in sentiments if r["n"] > 0
    ]

    px_lines = [f"  {t}: {p:.2f}" if p else f"  {t}: N/A" for t, p in watch_prices.items()]
    fut_lines = [f"  {t}: {p:.2f}" if p else f"  {t}: N/A" for t, p in futures_prices.items()]

    sp = f"{sp500:.2f}" if sp500 else "N/A"

    # Behavioural mirror — observational only (advisory; never gates). Placed
    # right after PORTFOLIO so the trader sees its own track record next to its
    # current book, before market data biases it.
    review_section = f"\n{self_review_block}\n" if self_review_block else ""
    # Per-name closed-trade memory — observational only, same advisory contract
    # as the self-review mirror (invariants #2/#12). Placed right after the
    # aggregate mirror so the trader sees its concrete history on the exact
    # names in play before market data biases it.
    track_section = f"{track_record_block}\n" if track_record_block else ""
    # Durable order/lesson memory so the model cannot amnesia-trade.
    # Observational only (invariants #2/#12).
    durable_memory_section = (
        f"{durable_memory_block}\n" if durable_memory_block else ""
    )
    # Per-name losing-streak watch — same observational/advisory contract as
    # track_record (invariants #2/#12). Placed immediately after the per-name
    # closed-trade memory because it is the same dimension (per-name history)
    # one degree sharper: a contiguous 2+ loss run on a single name is the
    # exact pattern the aggregate self_review cannot localise. Silent on a
    # clean book — the silence precedent.
    repeat_loser_section = (
        f"{repeat_loser_block}\n" if repeat_loser_block else ""
    )
    # Open-position thesis drift — observational only, same advisory contract
    # (invariants #2/#12). Placed right after the closed-trade memory and
    # before the structural-risk mirror so the trader sees its *open* book's
    # health against the verbatim reason it was opened for, the exact
    # discipline question this stack is built to surface. All-INTACT collapses
    # to silence — the chat-enrichment silence precedent — so a healthy book
    # produces no section.
    thesis_drift_section = (
        f"{thesis_drift_block}\n" if thesis_drift_block else ""
    )
    # Concentration + churn mirror — same observational/advisory contract as
    # the two mirrors above (invariants #2/#12). Placed last in the
    # behavioural stack so the trader sees its structural risk (book shape +
    # turnover) right after its history and before market data biases it.
    risk_section = f"{risk_mirror_block}\n" if risk_mirror_block else ""
    # Live-book SECTOR concentration — the natural sibling of the risk-mirror
    # (name concentration), one dimension over: how the book clusters by
    # sector + which in-play names would pile onto an already-heavy sector.
    # Same observational/advisory contract as the mirrors above (invariants
    # #2/#12). Placed immediately after the name-concentration mirror and
    # before the forward event block — the trader sees its structural risk by
    # name, then by sector, then what is *coming*, before prices bias it.
    sector_section = f"{sector_exposure_block}\n" if sector_exposure_block else ""
    # Forward beta/concentration STRESS — the natural sibling of the two
    # concentration blocks above, one dimension over: not *how* concentrated
    # the book is (name/sector) but what a routine adverse move *costs* it in
    # dollars, given that concentration. It is the day-one complement to the
    # dashboard-only tail_risk (which reads INSUFFICIENT until the book has
    # ≥20 daily returns). Same observational/advisory contract as the mirrors
    # above (invariants #2/#12 — the sector_exposure precedent). Placed
    # immediately after the structural concentration view and before the
    # forward event block: the trader sees its book shape, then what that
    # shape loses on a shock, then what is *coming*, before prices bias it.
    stress_section = f"{stress_block}\n" if stress_block else ""
    # Portfolio-construction target — the allocation sibling of
    # stress_scenarios. The concentration/stress blocks answer "what does the
    # current book risk?"; this answers "what diversified target would the
    # same evidence imply under a risk tolerance/time horizon?" Observational
    # only (invariants #2/#12): it never gates or sizes a trade, and Opus
    # retains full autonomy.
    construction_section = (
        f"{portfolio_construction_block}\n"
        if portfolio_construction_block else ""
    )
    # Forward scheduled-event awareness (earnings) — same observational/
    # advisory contract as the three mirrors above (invariants #2/#12). Placed
    # right after the backward-looking behavioural stack and before market
    # data: the trader sees its structural risk and then what is *coming* on
    # the names in play, before watchlist prices bias it.
    event_section = f"{event_calendar_block}\n" if event_calendar_block else ""
    # Forward MACRO awareness (scheduled FOMC rate decisions) — the macro
    # sibling of event_calendar, one dimension over: a rate-decision surprise
    # moves the WHOLE book in one instant (leveraged ETFs most violently),
    # so unlike the per-name earnings block this is market-wide and always
    # rendered. Same observational/advisory contract as the blocks above
    # (invariants #2/#12 — the event_calendar precedent). Placed immediately
    # after the per-name forward block so the two forward blocks stay
    # adjacent (earnings then macro) and before deployable-cash / prices.
    macro_section = f"{macro_calendar_block}\n" if macro_calendar_block else ""
    # Deployable-cash awareness — the lean prompt-facing complement to the
    # dashboard-only capital_paralysis (AGENTS.md #2/#12, the event_calendar
    # precedent). Placed last in the advisory stack, immediately before
    # WATCHLIST PRICES: the trader sees what its cash can actually fund right
    # before it sees the prices it would fund against. Observational only.
    bp_section = f"{buying_power_block}\n" if buying_power_block else ""
    # Forward mechanical-exit proximity — observational only, same
    # advisory contract as the blocks above (invariants #2/#12 — the
    # ``buying_power`` precedent). Surfaces the CURRENT-STATE diagnostic
    # for hard SL/TP (per-lot, signed-distance-to-firing) so Opus
    # triages with concrete thresholds instead of re-deriving SL/TP from
    # avg_cost (which is wrong on blended lots — see strategy.py BUY
    # branch: SL/TP are re-anchored to blended cost). A COMFORTABLE
    # book collapses to silence (the chat-enrichment silence
    # precedent) — added BELOW buying_power because it answers
    # "what is at risk on the current book?" right after
    # "what cash can I add with?".
    exit_proximity_section = (
        f"{exit_proximity_block}\n" if exit_proximity_block else ""
    )
    research_section = (
        f"{research_discovery_block}\n" if research_discovery_block else ""
    )
    active_research_section = (
        f"{active_research_block}\n" if active_research_block else ""
    )
    # Options desk — chain snapshots + strategy skills + open-option facts.
    # Observational only (AGENTS.md #2/#12). Placed with the other advisory
    # mirrors so Opus can choose options vs shares with real chain context.
    options_desk_section = (
        f"{options_desk_block}\n" if options_desk_block else ""
    )

    # Watchlist MACD breadth — one-line market-structure roll-up derived
    # from the same quant_signals the per-name TECHNICAL block renders.
    # Surfaces the macro "is the tape trending or stalled?" question Opus
    # would otherwise have to tally row-by-row. Observational only
    # (invariants #2/#12 — sibling to sector_exposure / risk_mirror).
    # Computed AFTER the quant_signals filter above so the breadth tracks
    # the same names-in-play set Opus sees per-row.
    try:
        from .analytics.macd_breadth import (
            build_macd_breadth, render_prompt_line,
        )
        breadth_snap = build_macd_breadth(quant_signals)
        breadth_line = render_prompt_line(breadth_snap)
    except Exception as e:
        print(f"[strategy] macd_breadth failed (non-fatal): {e}")
        breadth_line = ""
    breadth_section = f"{breadth_line}\n" if breadth_line else ""

    phase_line = f"MARKET_PHASE: {phase}\n" if phase else ""
    return f"""TIME (UTC): {now}
MARKET_OPEN: {market_open}
{phase_line}S&P 500 BENCHMARK: {sp}

PORTFOLIO:
  cash: ${snapshot['cash']:.2f}
  margin available (50% net worth): ${snapshot.get('margin_available', 0.0):.2f}
  buying power (stocks + long option premium): ${snapshot.get('stock_buying_power', snapshot['cash']):.2f}
  open positions value: ${snapshot['open_value']:.2f}
  total value: ${snapshot['total_value']:.2f}
  positions:
{chr(10).join(pos_lines) if pos_lines else '  (none)'}
{review_section}{track_section}{durable_memory_section}{repeat_loser_section}{thesis_drift_section}{risk_section}{sector_section}{stress_section}{construction_section}{event_section}{macro_section}{bp_section}{exit_proximity_section}{options_desk_section}
WATCHLIST PRICES:
{chr(10).join(px_lines)}

{active_research_section}
{research_section}
FUTURES:
{chr(10).join(fut_lines)}

TECHNICAL SIGNALS (RSI/MACD/MA cross/vol ratio/52w proximity):
{breadth_section}{_format_quant_signals(quant_signals or {})}

TICKER SENTIMENT (last 4h, from scored news):
{chr(10).join(sent_lines) if sent_lines else '  (no scored mentions)'}

TOP SCORED SIGNALS (last 2h, ai_score >= 4.0):
{chr(10).join(sig_lines) if sig_lines else '  (no high-score signals)'}

NO RISK LIMITS — full autonomy. Size by conviction.

Return JSON only."""


def _build_fallback_payload(snap: dict, merged: list[dict],
                            quant_sigs: dict[str, dict]) -> str:
    """Condensed (~200-word) context for the Sonnet fallback when Opus times
    out: cash + total value + top 3 positions, top 5 article headlines with
    ai_score, and RSI/MACD for up to 3 held positions. Deliberately omits
    summaries, watchlist prices, futures, sentiment and self-review so Sonnet
    can answer well inside the short fallback timeout."""
    positions = snap.get("positions", [])
    top_pos = sorted(
        positions,
        key=lambda p: abs(p.get("market_value") or 0.0),
        reverse=True,
    )[:3]
    pos_lines = []
    for p in top_pos:
        if p["type"] in ("call", "put"):
            tk = f"{p['ticker']} {p['type'].upper()} {p['strike']} {p['expiry']}"
        else:
            tk = p["ticker"]
        pos_lines.append(
            f"  {tk}: qty={p['qty']} value=${(p.get('market_value') or 0.0):.2f}"
        )

    art_lines = []
    for a in merged[:5]:
        try:
            score = float(a.get("ai_score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        art_lines.append(f"  [{score:.1f}] {(a.get('title') or '')[:120]}")

    held = {p["ticker"] for p in positions}
    quant_lines = []
    for tk in sorted(held):
        q = quant_sigs.get(tk)
        if not q:
            continue
        quant_lines.append(
            f"  {tk}: RSI={q.get('rsi', '?')} MACD={q.get('MACD', '?')}"
        )
        if len(quant_lines) >= 3:
            break

    return f"""PORTFOLIO:
  cash: ${snap.get('cash', 0.0):.2f}
  total value: ${snap.get('total_value', 0.0):.2f}
  top positions:
{chr(10).join(pos_lines) if pos_lines else '  (none)'}

TOP SIGNALS (title + ai_score):
{chr(10).join(art_lines) if art_lines else '  (no high-score signals)'}

QUANT (held positions — RSI + MACD direction):
{chr(10).join(quant_lines) if quant_lines else '  (none held / no signals)'}

Return JSON only."""


def _check_and_execute_hard_exits(
    store: "Store",
    snap: dict,
    *,
    market_open: bool | None = None,
) -> list[str]:
    """Execute mandatory stop-loss exits BEFORE Opus sees the prompt this cycle.

    Surveys ``store.positions_needing_hard_exit()`` (open stock lots whose
    last-marked current_price has breached the per-lot stop_loss_price set when
    the lot was opened). For each breached long lot: records a SELL at the
    current marked price, decrements the position via ``upsert_position``
    (negative qty closes the lot), and credits cash. For each breached short
    lot: records a COVER, increments the signed position toward zero, and
    debits cash.
    Updates ``snap['cash']`` in place so any caller that already has the
    pre-exit snapshot in hand sees the post-exit cash for sizing logic; the
    caller still re-snapshots when ``hard_exits`` is non-empty so positions
    + total_value are refreshed too.

    Returns the list of tickers that were force-sold this cycle (for the
    Discord notification + the runner's auto_exits tag). Non-fatal by
    construction — any failure logs and returns whatever exits succeeded
    before the fault, so a single bad row never blocks the cycle's decision.
    Hard stop automation is regular-session only; premarket/after-hours marks
    are too noisy to use as mandatory sell prices.
    """
    exits: list[str] = []
    try:
        if market_open is None:
            market_open = bool(market.is_market_open())
        if not market_open:
            return []
        positions = store.positions_needing_hard_exit()
        for pos in positions:
            ticker = pos["ticker"]
            try:
                qty = float(pos["qty"])
                price = float(pos["current_price"])
                sl = float(pos["stop_loss_price"])
            except (TypeError, ValueError) as e:
                print(f"[strategy] hard-exit: non-numeric row for {ticker}: {e}")
                continue
            if abs(qty) <= 0.0001 or price <= 0:
                continue
            if qty > 0 and price > sl:
                continue
            if qty < 0 and price < sl:
                continue
            side = "SELL" if qty > 0 else "COVER"
            close_qty = abs(qty)
            comparator = "<=" if qty > 0 else ">="
            reason = (
                f"HARD_SL: price {price:.2f} {comparator} threshold {sl:.2f}"
            )
            cash = float(snap.get("cash", 0) or 0.0)
            total_value = float(snap.get("total_value", 0) or 0.0)
            notional = price * close_qty
            store.record_trade(ticker, side, close_qty, price, reason)
            delta = -close_qty if qty > 0 else close_qty
            store.upsert_position(ticker, "stock", delta, price)
            new_cash = cash + notional if qty > 0 else cash - notional
            store.update_portfolio(new_cash, total_value)
            snap["cash"] = new_cash
            exits.append(ticker)
            print(f"[strategy] {reason}")
    except Exception as e:
        print(f"[strategy] hard-exit check failed (non-fatal): {e}")
    return exits


def _enforce_risk_pre_trade(decision: dict, snapshot: dict) -> tuple[bool, str]:
    """Basic sanity only — can't close more than is open. No broad risk caps."""
    action = (decision.get("action", "HOLD") or "HOLD").upper()
    if action == "HOLD":
        return True, ""

    ticker = (decision.get("ticker") or "").upper()
    try:
        qty = float(decision.get("qty") or 0)
    except (TypeError, ValueError):
        # Claude can emit a non-numeric qty (e.g. "all", "half"). _execute
        # already catches this defensively, but a direct call into this
        # helper without prior coercion would otherwise raise and abort the
        # cycle. Mirror _execute's BLOCKED-tuple shape ("ok=False") so the
        # caller logs a clean decision row instead.
        return False, f"qty not numeric: {decision.get('qty')!r}"
    if not math.isfinite(qty):
        return False, f"qty not finite: {decision.get('qty')!r}"
    if qty <= 0 and action != "REBALANCE":
        return False, "qty must be > 0"

    if action == "SELL":
        matches = [
            p for p in snapshot["positions"]
            if p["ticker"] == ticker and p["type"] == "stock"
            and float(p.get("qty") or 0.0) > 0
        ]
        if not matches:
            return False, f"no open long stock position in {ticker} to close"
        held = sum(float(p.get("qty") or 0.0) for p in matches)
        if qty > held + 1e-6:
            return False, f"sell qty {qty} exceeds held long {held} for {ticker} stock"

    if action == "COVER":
        matches = [
            p for p in snapshot["positions"]
            if p["ticker"] == ticker and p["type"] == "stock"
            and float(p.get("qty") or 0.0) < 0
        ]
        if not matches:
            return False, f"no open short stock position in {ticker} to cover"
        held_short = sum(abs(float(p.get("qty") or 0.0)) for p in matches)
        if qty > held_short + 1e-6:
            return False, f"cover qty {qty} exceeds held short {held_short} for {ticker} stock"

    if action in ("SELL_CALL", "SELL_PUT"):
        try:
            from .analytics.options_structures import classify_sell_option
            cls = classify_sell_option(decision, snapshot)
        except Exception as e:
            return False, f"option sell classify failed: {e}"
        if cls.get("mode") == "blocked":
            return False, cls.get("reason") or "option sell blocked"
        if cls.get("mode") == "close_long":
            match = cls.get("match") or {}
            held = float(match.get("qty") or 0.0)
            if qty > held + 1e-6:
                return False, (
                    f"sell qty {qty} exceeds held {held} for {ticker} "
                    f"{cls.get('otype')} {match.get('strike')} {match.get('expiry')}"
                )
        return True, ""

    if action in ("BUY_CALL", "BUY_PUT"):
        try:
            from .analytics.options_structures import classify_buy_option
            cls = classify_buy_option(decision, snapshot)
        except Exception as e:
            return False, f"option buy classify failed: {e}"
        if cls.get("mode") == "blocked":
            return False, cls.get("reason") or "option buy blocked"
        if cls.get("mode") == "close_short":
            held = float(cls.get("held_short") or 0.0)
            if qty > held + 1e-6:
                return False, f"buy-to-cover qty {qty} exceeds short {held}"
        return True, ""

    if action in (
        "BUY_CALL_SPREAD", "BUY_PUT_SPREAD", "SELL_CALL_SPREAD", "SELL_PUT_SPREAD",
    ):
        try:
            from .analytics.options_structures import parse_vertical_spread, price_vertical_spread
            from . import market as _m
            spec = parse_vertical_spread(decision)
            if not spec.get("ok"):
                return False, spec.get("reason") or "invalid spread"
            priced = price_vertical_spread(_m, spec)
            if not priced.get("ok"):
                return False, priced.get("reason") or "spread unpriceable"
            cash = float(snapshot.get("cash") or 0.0)
            buying_power = _margin_buying_power(snapshot)
            if priced.get("is_debit"):
                need = abs(float(priced.get("cash_delta") or 0.0))
                if buying_power + 1e-6 < need:
                    return False, (
                        f"insufficient cash/buying power for debit spread "
                        f"(available ${buying_power:.2f}, need ${need:.2f})"
                    )
            else:
                max_loss = float(priced.get("max_loss") or 0.0)
                credit = float(priced.get("cash_delta") or 0.0)
                if buying_power + credit + 1e-6 < max_loss:
                    return False, (
                        f"insufficient cash/buying power buffer for credit spread max loss "
                        f"(bp ${buying_power:.2f} + credit ${credit:.2f} < max_loss ${max_loss:.2f})"
                    )
        except Exception as e:
            return False, f"spread pretrade failed: {e}"
        return True, ""

    return True, ""



def _gross_long_exposure_usd(snapshot: dict) -> float:
    """Sum of long stock/option market values (shorts net against cash separately)."""
    total = 0.0
    for p in (snapshot.get("positions") or []):
        try:
            qty = float(p.get("qty") or 0.0)
        except (TypeError, ValueError):
            continue
        if qty <= 0:
            continue
        mv = p.get("market_value")
        try:
            if mv is not None:
                total += max(0.0, float(mv))
                continue
        except (TypeError, ValueError):
            pass
        mult = 100.0 if p.get("type") in ("call", "put") else 1.0
        try:
            px = float(p.get("current_price") or p.get("avg_cost") or 0.0)
        except (TypeError, ValueError):
            px = 0.0
        total += max(0.0, px * qty * mult)
    return total


def _deployment_metrics(snapshot: dict) -> dict:
    """Cash% and gross long exposure multiple vs net worth."""
    try:
        cash = float(snapshot.get("cash") or 0.0)
    except (TypeError, ValueError):
        cash = 0.0
    try:
        total = float(snapshot.get("total_value") or 0.0)
    except (TypeError, ValueError):
        total = 0.0
    gross = _gross_long_exposure_usd(snapshot)
    cash_pct = (cash / total) if total > 1e-9 else (1.0 if cash > 0 else 0.0)
    gross_x = (gross / total) if total > 1e-9 else 0.0
    underdeployed = (
        total > 1e-9 and (
            cash_pct > MAX_IDLE_CASH_PCT + 1e-9
            or gross_x + 1e-9 < TARGET_GROSS_EXPOSURE_MIN
        )
    )
    return {
        "cash": cash,
        "total_value": total,
        "gross_long_usd": gross,
        "cash_pct": cash_pct,
        "gross_exposure_x": gross_x,
        "underdeployed": underdeployed,
        "target_min_x": TARGET_GROSS_EXPOSURE_MIN,
        "target_max_x": TARGET_GROSS_EXPOSURE_MAX,
        "target_ideal_x": TARGET_GROSS_EXPOSURE_IDEAL,
        "max_idle_cash_pct": MAX_IDLE_CASH_PCT,
    }


def _stock_buy_leverage(decision: dict) -> float:
    """Requested regular-stock leverage, clamped to the supported 1x-20x range."""
    raw = decision.get("leverage", 1)
    try:
        lev = float(raw)
    except (TypeError, ValueError):
        return STOCK_BUY_MIN_LEVERAGE
    if not math.isfinite(lev):
        return STOCK_BUY_MIN_LEVERAGE
    return max(STOCK_BUY_MIN_LEVERAGE, min(STOCK_BUY_MAX_LEVERAGE, lev))


def _auto_boost_stock_buy_leverage(
    *,
    decision: dict,
    snapshot: dict,
    price: float,
    qty: float,
    requested_leverage: float,
) -> tuple[float, str]:
    """If the book is under-deployed, raise stock BUY leverage toward 120-150%.

    Smart bounds:
    - only regular stocks (not leveraged ETF tickers)
    - respect remaining stock buying power
    - respect single-name 70% hard cap on post-trade market value
    - do not lower an already-higher requested leverage
    - clamp to STOCK_BUY_MAX_LEVERAGE
    Returns (leverage, note). Note empty when no boost applied.
    """
    if not AUTO_LEVERAGE_WHEN_UNDERDEPLOYED:
        return requested_leverage, ""
    ticker = (decision.get("ticker") or "").upper()
    if not ticker or ticker in _LEVERAGED_ETFS_SL:
        return requested_leverage, ""
    if price <= 0 or qty <= 0:
        return requested_leverage, ""
    metrics = _deployment_metrics(snapshot)
    if not metrics["underdeployed"]:
        return requested_leverage, ""

    total = metrics["total_value"]
    if total <= 1e-9:
        return requested_leverage, ""

    requested_notional = price * qty * requested_leverage
    # Only juice tiny/token buys. If the model already sized a meaningful
    # sleeve (>=15% of book or >=50% of the ideal deploy gap), respect it.
    buying_power = _stock_buying_power(snapshot)
    gap_to_ideal = max(
        0.0,
        TARGET_GROSS_EXPOSURE_IDEAL * total - metrics["gross_long_usd"],
    )
    gap_to_max = max(
        0.0,
        TARGET_GROSS_EXPOSURE_MAX * total - metrics["gross_long_usd"],
    )
    if requested_notional + 1e-9 >= 0.15 * total:
        return requested_leverage, ""
    meaningful_gap = gap_to_ideal if gap_to_ideal > 0 else gap_to_max
    if meaningful_gap > 0 and requested_notional + 1e-9 >= 0.50 * meaningful_gap:
        return requested_leverage, ""

    target_add = min(
        gap_to_ideal if gap_to_ideal > 0 else gap_to_max,
        gap_to_max,
        buying_power,
    )
    if target_add <= requested_notional + 1e-6:
        min_add = min(
            buying_power,
            max(0.0, (metrics["cash"] - MAX_IDLE_CASH_PCT * total)),
        )
        if min_add <= requested_notional + 1e-6:
            return requested_leverage, ""
        target_add = min_add

    existing_mv = 0.0
    for p in (snapshot.get("positions") or []):
        if (p.get("ticker") or "").upper() != ticker:
            continue
        if str(p.get("type") or "stock") != "stock":
            continue
        try:
            q = float(p.get("qty") or 0.0)
            px = float(p.get("current_price") or p.get("avg_cost") or price)
            existing_mv += max(0.0, q * px)
        except (TypeError, ValueError):
            continue
    max_name_mv = 0.70 * total
    name_room = max(0.0, max_name_mv - existing_mv)
    target_add = min(target_add, name_room, buying_power)
    if target_add <= price * qty + 1e-6:
        return requested_leverage, ""

    raw_lev = target_add / (price * qty)
    floor = 1.5 if metrics["cash_pct"] > MAX_IDLE_CASH_PCT else requested_leverage
    boosted = max(requested_leverage, floor, raw_lev)
    boosted = max(STOCK_BUY_MIN_LEVERAGE, min(STOCK_BUY_MAX_LEVERAGE, boosted))
    max_lev_bp = buying_power / (price * qty) if price * qty > 0 else requested_leverage
    max_lev_name = name_room / (price * qty) if price * qty > 0 else requested_leverage
    boosted = min(boosted, max_lev_bp, max_lev_name)
    boosted = max(STOCK_BUY_MIN_LEVERAGE, min(STOCK_BUY_MAX_LEVERAGE, boosted))
    if boosted <= requested_leverage + 1e-9:
        return requested_leverage, ""
    note = (
        f"auto_leverage {requested_leverage:g}x→{boosted:g}x "
        f"(cash={metrics['cash_pct']*100:.1f}% gross={metrics['gross_exposure_x']:.2f}x "
        f"target={TARGET_GROSS_EXPOSURE_MIN:.2f}-{TARGET_GROSS_EXPOSURE_MAX:.2f}x)"
    )
    return boosted, note


def _stock_buying_power(snapshot: dict) -> float:
    """Cash plus 50% margin on current net worth.

    Used for regular-stock BUY/SHORT and long option premium (BUY_CALL/BUY_PUT
    opens, buy-to-cover shorts, and debit verticals). Same 1.5x-at-all-cash
    economics: BP = cash + 0.50 * net_worth.
    """
    if "stock_buying_power" in snapshot:
        try:
            return float(snapshot["stock_buying_power"])
        except (TypeError, ValueError):
            pass
    cash = float(snapshot.get("cash") or 0.0)
    total = float(snapshot.get("total_value") or 0.0)
    return cash + max(0.0, total) * STOCK_MARGIN_NET_WORTH_PCT


def _margin_buying_power(snapshot: dict) -> float:
    """Alias for the shared cash+50% net-worth buying power."""
    return _stock_buying_power(snapshot)


def _margin_available(snapshot: dict) -> float:
    total = float(snapshot.get("total_value") or 0.0)
    return max(0.0, total) * STOCK_MARGIN_NET_WORTH_PCT


def _parse_trade_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _position_abs_value(p: dict) -> float:
    try:
        if p.get("market_value") is not None:
            return abs(float(p.get("market_value") or 0.0))
        mult = 100 if p.get("type") in ("call", "put") else 1
        price = float(p.get("current_price") or p.get("avg_cost") or 0.0)
        qty = float(p.get("qty") or 0.0)
        return abs(price * qty * mult)
    except (TypeError, ValueError):
        return 0.0


def _trade_discipline_guard(
    decision: dict,
    snapshot: dict,
    store: Store,
    *,
    price: float,
    effective_qty: float,
    now: datetime | None = None,
) -> tuple[bool, str]:
    """Hard block repeated entries and concentration pile-ons.

    Applies only to new stock entries (BUY/SHORT). Exits are always allowed so
    the guard cannot trap risk. The small-book threshold keeps legacy unit tests
    and toy local simulations from turning into "35% max" math exercises; the
    live Mac book is above this threshold.
    """
    action = (decision.get("action") or "").upper()
    if action not in {"BUY", "SHORT"}:
        return True, ""
    try:
        total = float(snapshot.get("total_value") or 0.0)
    except (TypeError, ValueError):
        total = 0.0
    if total < TRADE_DISCIPLINE_MIN_BOOK_VALUE:
        return True, ""

    ticker = str(decision.get("ticker") or "").upper()
    if not ticker:
        return False, "ticker required for entry discipline"
    now = now or datetime.now(timezone.utc)

    try:
        recent = store.recent_trades(100)
    except Exception:
        recent = []
    for tr in recent:
        if str(tr.get("ticker") or "").upper() != ticker:
            continue
        ts = _parse_trade_ts(tr.get("timestamp"))
        if ts is None:
            continue
        age_s = (now - ts).total_seconds()
        if age_s < 0:
            continue
        prior = str(tr.get("action") or "").upper()
        if prior in {"SELL", "COVER", "SELL_CALL", "SELL_PUT"}:
            if age_s < ENTRY_COOLDOWN_AFTER_EXIT_S:
                mins = int(age_s // 60)
                wait = int((ENTRY_COOLDOWN_AFTER_EXIT_S - age_s + 59) // 60)
                return (
                    False,
                    f"churn cooldown: {ticker} was exited {mins}m ago; "
                    f"wait {wait}m or bring a genuinely new catalyst",
                )
        elif prior in {"BUY", "SHORT", "BUY_CALL", "BUY_PUT"}:
            if age_s < ENTRY_COOLDOWN_AFTER_ANY_TRADE_S:
                mins = int(age_s // 60)
                wait = int((ENTRY_COOLDOWN_AFTER_ANY_TRADE_S - age_s + 59) // 60)
                return (
                    False,
                    f"same-ticker entry cooldown: {ticker} was traded {mins}m "
                    f"ago; wait {wait}m before adding/recycling",
                )

    notional = abs(float(price) * float(effective_qty))
    if notional <= 0:
        return True, ""

    try:
        from .analytics.sector_exposure import classify
    except Exception:
        classify = lambda _ticker: "other"  # noqa: E731

    name_values: dict[str, float] = {}
    sector_values: dict[str, float] = {}
    for p in snapshot.get("positions") or []:
        tk = str(p.get("ticker") or "").upper()
        if not tk:
            continue
        value = _position_abs_value(p)
        if value <= 0:
            continue
        name_values[tk] = name_values.get(tk, 0.0) + value
        sec = classify(tk)
        sector_values[sec] = sector_values.get(sec, 0.0) + value

    sector = classify(ticker)
    denom = max(total, sum(name_values.values()) + notional, 1.0)
    post_name = name_values.get(ticker, 0.0) + notional
    post_sector = sector_values.get(sector, 0.0) + notional
    post_name_pct = post_name / denom * 100.0
    post_sector_pct = post_sector / denom * 100.0

    if post_name_pct > MAX_POST_TRADE_SINGLE_NAME_PCT + 1e-6:
        return (
            False,
            f"diversification block: {ticker} would be "
            f"{post_name_pct:.1f}% of book; max "
            f"{MAX_POST_TRADE_SINGLE_NAME_PCT:.1f}%",
        )
    if post_sector_pct > MAX_POST_TRADE_SECTOR_PCT + 1e-6:
        return (
            False,
            f"sector concentration block: {sector} would be "
            f"{post_sector_pct:.1f}% of book; max "
            f"{MAX_POST_TRADE_SECTOR_PCT:.1f}%",
        )
    if sector.endswith("_lev") and post_sector_pct > MAX_POST_TRADE_LEVERAGED_PCT + 1e-6:
        return (
            False,
            f"leveraged sleeve block: {sector} would be "
            f"{post_sector_pct:.1f}% of book; max "
            f"{MAX_POST_TRADE_LEVERAGED_PCT:.1f}%",
        )
    return True, ""


# Ambiguous / keyword-collision tickers that historically produced dumb
# catalyst→symbol maps (e.g. FCC "local TV" story → Grupo Televisa `TV`).
_AMBIGUOUS_ENTRY_TICKERS = frozenset({
    "TV", "AI", "DD", "OPEN", "CASH", "REAL", "LIFE", "GOOD", "SOUL", "DATA",
    "NEXT", "PLAY", "MOVE", "BEST", "CARE", "FOOD", "LAND", "FUND", "GOLD",
    "TECH", "NEWS", "GAME", "APP", "NET", "BOX", "RUN", "FAST", "SAFE", "TRUE",
    "FREE", "HOME", "WORK", "TIME", "WAVE", "BIRD", "BEAM", "CUBE", "LEAF",
    "MOON", "SNOW", "FIRE", "WIND", "ROCK", "SHIP", "STAR", "SUN", "SKY", "BIG",
    "YOU", "LOVE", "HOPE", "IDEA", "GROW", "UNIT", "SPOT", "BILL", "TEAM",
})

# Issuer phrases that must appear in the entry reasoning for selected tickers.
# Keep these multi-token / distinctive so plain English words do not "count".
_ISSUER_REQUIRED_ALIASES: dict[str, tuple[str, ...]] = {
    "TV": ("grupo televisa", "televisa", "televisaunivision"),
    "AI": ("c3.ai", "c3 ai", "c3ai"),
    "OPEN": ("opendoor", "open door"),
    "DD": ("dupont", "du pont"),
    "SPOT": ("spotify",),
    "BILL": ("bill.com", "billcom", "bill holdings"),
    "NET": ("cloudflare",),
    "APP": ("applovin", "app lovin"),
    "BOX": ("box, inc", "box inc", "box enterprise"),
    "YOU": ("clear secure", "clearsecure"),
}

# If these catalyst phrases appear, the ticker must NOT be a keyword-collision
# name unless the real issuer is also named.
_CATALYST_TICKER_DENYLIST: tuple[tuple[tuple[str, ...], frozenset[str], str], ...] = (
    (
        ("fcc", "local tv", "ownership cap", "broadcast ownership", "station group",
         "nexstar", "sinclair", "tegna", "gray television", "gray media", "local broadcast"),
        frozenset({"TV"}),
        "US local-broadcast/FCC catalyst does not map to Grupo Televisa (TV)",
    ),
    (
        ("artificial intelligence", "generative ai", "ai chip", "ai capex", "openai",
         "large language model", "llm "),
        frozenset({"AI"}),
        "generic AI-theme catalyst is not enough to buy ticker AI without naming the issuer",
    ),
    (
        ("open source", "open-source", "open ai", "openai"),
        frozenset({"OPEN"}),
        "open/AI keyword catalyst does not map to Opendoor (OPEN) without naming the issuer",
    ),
)


def _entry_text_blob(decision: dict) -> str:
    parts = [
        decision.get("reasoning"),
        decision.get("catalyst"),
        decision.get("issuer"),
        decision.get("company"),
        decision.get("entity"),
        decision.get("thesis"),
        decision.get("reason"),
    ]
    return " ".join(str(p) for p in parts if p).lower()


def _entry_issuer_catalyst_guard(decision: dict) -> tuple[bool, str]:
    """Hard-block keyword→ticker puns and catalyst/entity mismatches on entries.

    The live book bought Grupo Televisa (`TV`) off an FCC local-TV ownership
    story. That is not a soft prompt failure — it is an issuer map failure.
    Enforce entity naming for ambiguous tickers and denylist known bad pairs.
    """
    action = str(decision.get("action") or "").upper()
    if action not in {
        "BUY", "SHORT", "BUY_CALL", "BUY_PUT",
        "BUY_CALL_SPREAD", "BUY_PUT_SPREAD", "SELL_CALL_SPREAD", "SELL_PUT_SPREAD",
        # Opening short options / CSP / covered call still need a real underlying thesis.
        "SELL_PUT", "SELL_CALL",
    }:
        return True, ""

    ticker = str(decision.get("ticker") or "").upper().strip()
    if not ticker:
        return True, ""

    blob = _entry_text_blob(decision)
    if not blob.strip():
        if ticker in _AMBIGUOUS_ENTRY_TICKERS or ticker in _ISSUER_REQUIRED_ALIASES:
            return (
                False,
                f"issuer/catalyst required: ambiguous ticker {ticker} entry needs "
                f"named issuer + real catalyst in reasoning",
            )
        return True, ""

    # Known bad catalyst→ticker pairs first (precise failure text).
    for keys, denied_tickers, msg in _CATALYST_TICKER_DENYLIST:
        if ticker not in denied_tickers:
            continue
        if any(k in blob for k in keys):
            aliases = _ISSUER_REQUIRED_ALIASES.get(ticker) or ()
            if aliases and any(a in blob for a in aliases):
                continue
            return False, f"issuer/catalyst mismatch: {msg}"

    aliases = _ISSUER_REQUIRED_ALIASES.get(ticker)
    ambiguous = ticker in _AMBIGUOUS_ENTRY_TICKERS or len(ticker) <= 2
    if aliases and ambiguous:
        if not any(a in blob for a in aliases):
            return (
                False,
                f"issuer/catalyst mismatch: ticker {ticker} requires naming the real "
                f"issuer in reasoning (e.g. '{aliases[0]}'), not a sector keyword pun",
            )

    # Generic short/ambiguous ticker rule: require at least one meaningful
    # entity token so pure "TV breakout" / "AI is hot" entries die here.
    if ambiguous and not aliases:
        tokens = {
            t for t in "".join(ch if ch.isalnum() else " " for ch in blob).split()
            if t.isalpha()
        }
        filler = {
            "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
            "from", "this", "that", "is", "are", "be", "as", "at", "by", "it",
            "into", "over", "under", "strong", "high", "low", "buy", "sell",
            "long", "short", "call", "put", "stock", "shares", "catalyst",
            "news", "score", "fresh", "breakout", "momentum", "technical",
        }
        meaningful = {
            t for t in tokens
            if t != ticker.lower() and t not in filler and len(t) >= 4
        }
        if not meaningful:
            return (
                False,
                f"issuer/catalyst mismatch: ambiguous ticker {ticker} entry needs "
                f"a named company/entity in reasoning, not bare keyword momentum",
            )

    return True, ""


def _reason_blob(decision: dict) -> str:
    return " ".join(
        str(decision.get(k) or "")
        for k in ("reasoning", "reason", "catalyst", "thesis", "notes")
    ).lower()


def _is_hard_stop_exit(decision: dict) -> bool:
    blob = _reason_blob(decision)
    return any(
        m in blob
        for m in (
            "hard stop",
            "hard_sl",
            "hard-sl",
            "stop loss",
            "stop-loss",
            "risk limit",
            "margin call",
        )
    )


def _is_map_failure_exit(decision: dict) -> bool:
    blob = _reason_blob(decision)
    return any(
        m in blob
        for m in (
            "issuer/catalyst map failure",
            "issuer map failure",
            "wrong entity",
            "wrong issuer",
            "false positive ticker",
        )
    )


def _is_thesis_kill_exit(decision: dict) -> bool:
    """Real thesis/catalyst death only — not vibes or P&L discomfort.

    2026-08-10 failure mode: model said "thesis break / long leg -17% / largest
    drag / weakening" and panic-sold losers. Those phrases alone are NOT enough.
    """
    blob = _reason_blob(decision)
    if not blob.strip():
        return False
    # Map failure is always a valid early flatten.
    if _is_map_failure_exit(decision):
        return True

    kill_frame = any(
        m in blob
        for m in (
            "thesis kill",
            "thesis-kill",
            "thesis is dead",
            "thesis broken",
            "broken thesis",
            "invalid thesis",
            "invalidated thesis",
            "thesis invalid",
            "catalyst failed",
            "catalyst is dead",
            "catalyst broken",
            "edge gone",
            "setup broken",
            "no longer valid",
        )
    )
    evidence = any(
        m in blob
        for m in (
            "trading halt",
            "halted",
            "going concern",
            "bankrupt",
            "bankruptcy",
            "delist",
            "fraud",
            "restatement",
            "accounting scandal",
            "guidance cut",
            "downgrade",
            "structural break",
            "regime change",
            "sec charge",
            "indictment",
            "missed earnings",
            "revenue miss",
            "eps miss",
            "contract lost",
            "customer lost",
            "ceo resign",
            "data contradict",
            "thesis falsified",
            "wrong issuer",
            "wrong entity",
        )
    )
    if evidence:
        return True
    if not kill_frame:
        return False
    # Reject pure panic / mark-to-market language even if "thesis broken" is present.
    panic_marks = (
        "largest drag",
        "went red",
        "deep red",
        "deeply red",
        "long leg",
        "marked down",
        "mark down",
        "paper hands",
        "uncomfortable",
        "weakening",
        "fading",
        "underwater",
        "red candle",
    )
    if any(m in blob for m in panic_marks):
        return False
    # Require some non-boilerplate content beyond the kill phrase itself.
    stripped = blob
    for tok in (
        "thesis", "kill", "broken", "break", "breaks", "invalid", "invalidated",
        "dead", "gone", "failed", "catalyst", "edge", "setup", "the", "is", "a",
        "an", "to", "for", "and", "or", "of", "on", "in", "no", "longer", "valid",
    ):
        stripped = stripped.replace(tok, " ")
    words = [w for w in stripped.split() if len(w) > 2]
    return len(words) >= 2


def _position_mark_and_cost(pos: dict, live_price: float | None = None) -> tuple[float, float]:
    try:
        avg = float(pos.get("avg_cost") or 0.0)
    except (TypeError, ValueError):
        avg = 0.0
    if live_price is not None:
        try:
            mark = float(live_price)
        except (TypeError, ValueError):
            mark = 0.0
    else:
        try:
            mark = float(pos.get("current_price") or 0.0)
        except (TypeError, ValueError):
            mark = 0.0
    return mark, avg


def _is_underwater_position(
    pos: dict,
    *,
    side: str,
    live_price: float | None = None,
) -> bool:
    mark, avg = _position_mark_and_cost(pos, live_price)
    if mark <= 0 or avg <= 0:
        return False
    eps = max(0.0, float(UNDERWATER_EXIT_EPS))
    if side == "long":
        return mark < avg * (1.0 - eps)
    return mark > avg * (1.0 + eps)


def _option_premium_kill_hit(pos: dict, live_price: float | None = None) -> bool:
    mark, avg = _position_mark_and_cost(pos, live_price)
    if mark <= 0 or avg <= 0:
        return False
    return ((avg - mark) / avg) + 1e-12 >= float(OPTION_PREMIUM_KILL_PCT)


def _underwater_discretionary_exit_guard(
    decision: dict,
    pos: dict,
    *,
    side: str,
    live_price: float | None = None,
    is_option: bool = False,
) -> tuple[bool, str]:
    """Permanent anti-panic: block sell-low / cover-high discretionary exits.

    Allowed underwater exits only:
      - hard stop language (engine hard-SL path still independent)
      - map-failure / real thesis-kill evidence
      - long option premium down >= OPTION_PREMIUM_KILL_PCT
    Winners are not blocked here.
    """
    if not BLOCK_UNDERWATER_DISCRETIONARY_EXITS:
        return True, ""
    if not _is_underwater_position(pos, side=side, live_price=live_price):
        return True, ""
    if _is_hard_stop_exit(decision):
        return True, ""
    if is_option and side == "long" and _option_premium_kill_hit(pos, live_price):
        return True, ""
    if _is_thesis_kill_exit(decision):
        return True, ""
    kind = "option" if is_option else "stock"
    return (
        False,
        (
            f"underwater panic-sell blocked on {kind}: hold the red and wait for "
            f"recovery unless hard stop, real thesis-kill evidence, or option "
            f"-{int(OPTION_PREMIUM_KILL_PCT * 100)}% premium kill"
        ),
    )


def _minimum_hold_exit_guard(
    decision: dict,
    snapshot: dict,
    store: Store,
    *,
    now: datetime | None = None,
    position_type: str = "stock",
    match_pos: dict | None = None,
) -> tuple[bool, str]:
    """Hold lock for discretionary exits (stocks + optional option rows).

    Default 4h. Hard SL still owns forced risk exits independently.
    Real thesis-kill / map-failure / hard-stop language bypasses the lock.
    Pure "weakening / largest drag / went red" language does NOT bypass.
    """
    if MIN_HOLD_BEFORE_DISCRETIONARY_EXIT_S <= 0:
        return True, ""
    action = (decision.get("action") or "").upper()
    if action not in {"SELL", "COVER", "SELL_CALL", "SELL_PUT"}:
        return True, ""

    if _is_hard_stop_exit(decision) or _is_thesis_kill_exit(decision):
        return True, ""
    try:
        total = float(snapshot.get("total_value") or 0.0)
    except (TypeError, ValueError):
        total = 0.0
    if total < TRADE_DISCIPLINE_MIN_BOOK_VALUE:
        return True, ""

    ticker = str(decision.get("ticker") or "").upper()
    if not ticker:
        return False, "ticker required for exit discipline"
    now = now or datetime.now(timezone.utc)

    if match_pos is not None:
        positions = [match_pos]
    else:
        want_long = action in {"SELL", "SELL_CALL", "SELL_PUT"}

        def _matches(p: dict) -> bool:
            if str(p.get("ticker") or "").upper() != ticker:
                return False
            ptype = str(p.get("type") or "")
            if position_type == "stock":
                if ptype != "stock":
                    return False
            else:
                # option path: call/put rows
                if ptype not in {"call", "put"}:
                    return False
            try:
                qty = float(p.get("qty") or 0.0)
            except (TypeError, ValueError):
                return False
            return qty > 0 if want_long else qty < 0

        positions = [p for p in snapshot.get("positions") or [] if _matches(p)]
        if not any(p.get("opened_at") for p in positions):
            try:
                positions = [p for p in store.open_positions() if _matches(p)]
            except Exception:
                positions = []

    opened = None
    for pos in positions:
        ts = _parse_trade_ts(pos.get("opened_at"))
        if ts is None:
            continue
        opened = ts if opened is None else max(opened, ts)
    if opened is None:
        return True, ""

    age_s = (now - opened).total_seconds()
    if age_s < 0:
        return True, ""
    if age_s < MIN_HOLD_BEFORE_DISCRETIONARY_EXIT_S:
        held_h = max(0, int(age_s // 3600))
        wait_h = int((MIN_HOLD_BEFORE_DISCRETIONARY_EXIT_S - age_s + 3599) // 3600)
        return (
            False,
            f"minimum-hold lock: {ticker} held {held_h}h; wait {wait_h}h "
            "before discretionary exit unless hard stop / real thesis-kill",
        )
    return True, ""


# Actions that book a paper fill against a price. HOLD/REBALANCE never fill.
_FILL_ACTIONS = frozenset({
    "BUY", "SELL", "SHORT", "COVER",
    "BUY_CALL", "BUY_PUT", "SELL_CALL", "SELL_PUT",
    "BUY_CALL_SPREAD", "BUY_PUT_SPREAD", "SELL_CALL_SPREAD", "SELL_PUT_SPREAD",
})


def _execute(decision: dict, snapshot: dict, store: Store) -> tuple[str, str]:
    """Apply the decision against the paper book. Returns (status, detail)."""
    action = (decision.get("action", "HOLD") or "HOLD").upper()
    if action == "HOLD":
        return "HOLD", decision.get("reasoning", "")

    if action == "REBALANCE":
        return "HOLD", "REBALANCE not yet implemented; treated as HOLD"

    # Hard issuer/catalyst map gate before any fill path (stocks + options).
    ok, why = _entry_issuer_catalyst_guard(decision)
    if not ok:
        return "BLOCKED", why

    # Hard gate: model output cannot mint stock/option fills when no equity
    # trading period is live. RTH / pre / post / weekday overnight are fine.
    # WEEKEND and HOLIDAY are impossible liquidity and must BLOCK.
    if action in _FILL_ACTIONS and not market.is_any_trading_session_open():
        phase = market.market_phase()
        return (
            "BLOCKED",
            f"no trading session open (phase={phase}); "
            f"cannot {action} outside RTH/pre/post/overnight",
        )

    ticker = (decision.get("ticker") or "").upper()
    # Claude can emit a non-numeric qty (e.g. "all", "half"). Coerce defensively
    # so a bad field yields a recorded BLOCKED decision instead of an uncaught
    # ValueError that aborts the whole cycle with no decision/equity point logged.
    try:
        qty = float(decision.get("qty") or 0)
    except (TypeError, ValueError):
        return "BLOCKED", f"qty not numeric: {decision.get('qty')!r}"
    if not math.isfinite(qty):
        return "BLOCKED", f"qty not finite: {decision.get('qty')!r}"
    reason = decision.get("reasoning", "")

    ok, why = _enforce_risk_pre_trade(decision, snapshot)
    if not ok:
        return "BLOCKED", why

    if action in ("BUY", "SELL", "SHORT", "COVER"):
        price = market.get_price(ticker)
        try:
            price = float(price)
        except (TypeError, ValueError):
            return "BLOCKED", f"no price for {ticker}"
        if price <= 0 or not math.isfinite(price):
            return "BLOCKED", f"no price for {ticker}"
        if action == "BUY":
            leverage = _stock_buy_leverage(decision)
            leverage, boost_note = _auto_boost_stock_buy_leverage(
                decision=decision,
                snapshot=snapshot,
                price=price,
                qty=qty,
                requested_leverage=leverage,
            )
            effective_qty = round(qty * leverage, 8)
            notional = price * effective_qty
            ok, why = _trade_discipline_guard(
                decision, snapshot, store,
                price=price, effective_qty=effective_qty,
            )
            if not ok:
                return "BLOCKED", why
            buying_power = _stock_buying_power(snapshot)
            if buying_power - notional < -1e-6:
                return (
                    "BLOCKED",
                    f"insufficient cash/buying power (cash ${snapshot['cash']:.2f}, "
                    f"margin ${max(0.0, float(snapshot.get('total_value') or 0.0) * STOCK_MARGIN_NET_WORTH_PCT):.2f}, "
                    f"available ${buying_power:.2f}, need ${notional:.2f}",
                )
            trade_reason = reason
            if leverage != STOCK_BUY_MIN_LEVERAGE or boost_note:
                extra = (
                    f"leverage={leverage:g}x; requested_qty={qty:g}; "
                    f"effective_qty={effective_qty:g}"
                )
                if boost_note:
                    extra = f"{extra}; {boost_note}"
                trade_reason = f"{reason} [{extra}]".strip()
            store.record_trade(ticker, "BUY", effective_qty, price, trade_reason)
            store.upsert_position(ticker, "stock", effective_qty, price)
            # Stamp hard SL/TP on the just-opened (or blended) lot. Pass
            # qty=0 — metadata-only path — so the size is not touched, only
            # the SL/TP fields. Re-entries blend price into the avg_cost; the
            # SL/TP fields are RE-anchored to the new blended-cost basis so a
            # winning add doesn't keep the SL pinned to the original entry
            # (or, conversely, an averaging-down add doesn't keep TP too low
            # to ever fire). Non-fatal.
            _sl_pct = (_SL_PCT_LEVERAGED if ticker in _LEVERAGED_ETFS_SL
                       else _SL_PCT_STANDARD)
            _tp_pct = (_TP_PCT_LEVERAGED if ticker in _LEVERAGED_ETFS_SL
                       else _TP_PCT_STANDARD)
            _sl_price = round(price * (1 - _sl_pct), 4)
            _tp_price = round(price * (1 + _tp_pct), 4)
            try:
                store.upsert_position(
                    ticker, "stock", 0, price,
                    stop_loss_price=_sl_price,
                    take_profit_price=_tp_price,
                )
            except Exception as _e:
                print(f"[strategy] failed to set SL/TP for {ticker}: {_e}")
            # positions=None: end-of-cycle _portfolio_snapshot re-marks and
            # writes the post-trade blend; passing snapshot["positions"] here
            # would desync portfolio.positions_json from the positions table
            # (a dashboard read would see the new cash but the pre-trade list).
            store.update_portfolio(snapshot["cash"] - notional, snapshot["total_value"])
            if leverage != STOCK_BUY_MIN_LEVERAGE or boost_note:
                suffix = f" ({leverage:g}x leverage from requested qty {qty:g}"
                if boost_note:
                    suffix += f"; {boost_note}"
                suffix += ")"
            else:
                suffix = ""
            return "FILLED", f"BUY {effective_qty:g} {ticker} @ {price:.2f}{suffix}"
        if action == "SELL":
            ok, why = _minimum_hold_exit_guard(decision, snapshot, store)
            if not ok:
                return "BLOCKED", why
            long_pos = next(
                (
                    p for p in snapshot.get("positions") or []
                    if str(p.get("ticker") or "").upper() == ticker
                    and str(p.get("type") or "") == "stock"
                    and float(p.get("qty") or 0.0) > 0
                ),
                {
                    "ticker": ticker,
                    "type": "stock",
                    "qty": qty,
                    "avg_cost": 0.0,
                    "current_price": price,
                },
            )
            ok, why = _underwater_discretionary_exit_guard(
                decision, long_pos, side="long", live_price=price, is_option=False,
            )
            if not ok:
                return "BLOCKED", why
            notional = price * qty
            store.record_trade(ticker, "SELL", qty, price, reason)
            store.upsert_position(ticker, "stock", -qty, price)
            store.update_portfolio(snapshot["cash"] + notional, snapshot["total_value"])
            return "FILLED", f"SELL {qty} {ticker} @ {price:.2f}"

        if action == "SHORT":
            notional = price * qty
            ok, why = _trade_discipline_guard(
                decision, snapshot, store,
                price=price, effective_qty=qty,
            )
            if not ok:
                return "BLOCKED", why
            shorting_power = _stock_buying_power(snapshot)
            if shorting_power - notional < -1e-6:
                return (
                    "BLOCKED",
                    f"insufficient shorting power (cash ${snapshot['cash']:.2f}, "
                    f"margin ${max(0.0, float(snapshot.get('total_value') or 0.0) * STOCK_MARGIN_NET_WORTH_PCT):.2f}, "
                    f"available ${shorting_power:.2f}, need ${notional:.2f})",
                )
            _sl_pct = (_SL_PCT_LEVERAGED if ticker in _LEVERAGED_ETFS_SL
                       else _SL_PCT_STANDARD)
            _tp_pct = (_TP_PCT_LEVERAGED if ticker in _LEVERAGED_ETFS_SL
                       else _TP_PCT_STANDARD)
            _sl_price = round(price * (1 + _sl_pct), 4)
            _tp_price = round(price * (1 - _tp_pct), 4)
            store.record_trade(ticker, "SHORT", qty, price, reason)
            store.upsert_position(
                ticker, "stock", -qty, price,
                stop_loss_price=_sl_price,
                take_profit_price=_tp_price,
            )
            store.update_portfolio(snapshot["cash"] + notional, snapshot["total_value"])
            return "FILLED", f"SHORT {qty:g} {ticker} @ {price:.2f}"

        if action == "COVER":
            ok, why = _minimum_hold_exit_guard(decision, snapshot, store)
            if not ok:
                return "BLOCKED", why
            short_pos = next(
                (
                    p for p in snapshot.get("positions") or []
                    if str(p.get("ticker") or "").upper() == ticker
                    and str(p.get("type") or "") == "stock"
                    and float(p.get("qty") or 0.0) < 0
                ),
                {
                    "ticker": ticker,
                    "type": "stock",
                    "qty": -qty,
                    "avg_cost": 0.0,
                    "current_price": price,
                },
            )
            ok, why = _underwater_discretionary_exit_guard(
                decision, short_pos, side="short", live_price=price, is_option=False,
            )
            if not ok:
                return "BLOCKED", why
            notional = price * qty
            store.record_trade(ticker, "COVER", qty, price, reason)
            store.upsert_position(ticker, "stock", qty, price)
            store.update_portfolio(snapshot["cash"] - notional, snapshot["total_value"])
            return "FILLED", f"COVER {qty:g} {ticker} @ {price:.2f}"

    if action in ("BUY_CALL", "BUY_PUT"):
        from .analytics.options_structures import classify_buy_option
        cls = classify_buy_option(decision, snapshot)
        if cls.get("mode") == "blocked":
            return "BLOCKED", cls.get("reason") or "option buy blocked"
        otype = cls["otype"]
        strike_f = float(cls["strike"])
        expiry = cls["expiry"]
        opt_px = market.get_option_price(ticker, expiry, strike_f, otype)
        if not opt_px:
            return "BLOCKED", f"no option price for {ticker} {expiry} {strike_f} {otype}"
        notional = float(opt_px) * qty * 100.0

        if cls.get("mode") == "close_short":
            held = float(cls.get("held_short") or 0.0)
            if qty > held + 1e-6:
                return "BLOCKED", f"buy-to-cover qty {qty} exceeds short {held}"
            buying_power = _margin_buying_power(snapshot)
            if buying_power - notional < -1e-6:
                return "BLOCKED", (
                    f"insufficient cash/buying power to buy-to-cover "
                    f"(cash ${snapshot['cash']:.2f}, margin ${_margin_available(snapshot):.2f}, "
                    f"available ${buying_power:.2f}, need ${notional:.2f})"
                )
            store.record_trade(
                ticker, action, qty, float(opt_px), reason,
                expiry=expiry, strike=strike_f, option_type=otype,
            )
            store.upsert_position(ticker, otype, qty, float(opt_px), expiry=expiry, strike=strike_f)
            store.update_portfolio(snapshot["cash"] - notional, snapshot["total_value"])
            return (
                "FILLED",
                f"BUY_TO_COVER {qty:g} {ticker} {strike_f}{otype[0].upper()} {expiry} @ {float(opt_px):.2f}",
            )

        buying_power = _margin_buying_power(snapshot)
        if buying_power - notional < -1e-6:
            return "BLOCKED", (
                f"insufficient cash/buying power "
                f"(cash ${snapshot['cash']:.2f}, margin ${_margin_available(snapshot):.2f}, "
                f"available ${buying_power:.2f}, need ${notional:.2f})"
            )
        store.record_trade(
            ticker, action, qty, float(opt_px), reason,
            expiry=expiry, strike=strike_f, option_type=otype,
        )
        store.upsert_position(ticker, otype, qty, float(opt_px), expiry=expiry, strike=strike_f)
        store.update_portfolio(snapshot["cash"] - notional, snapshot["total_value"])
        return "FILLED", f"{action} {qty:g} {ticker} {strike_f}{otype[0].upper()} {expiry} @ {float(opt_px):.2f}"

    if action in ("SELL_CALL", "SELL_PUT"):
        from .analytics.options_structures import classify_sell_option, csp_collateral_required
        cls = classify_sell_option(decision, snapshot)
        if cls.get("mode") == "blocked":
            return "BLOCKED", cls.get("reason") or "option sell blocked"
        otype = cls["otype"]
        mode = cls["mode"]

        if mode == "close_long":
            match = cls["match"]
            if qty > float(match.get("qty") or 0.0) + 1e-6:
                return "BLOCKED", (
                    f"sell qty {qty} exceeds held {match.get('qty')} for "
                    f"{ticker} {match.get('strike')}{otype[0].upper()} {match.get('expiry')}"
                )
            live_px = market.get_option_price(ticker, match["expiry"], match["strike"], otype)
            if live_px is not None:
                opt_px = float(live_px)
            elif _option_expired(match["expiry"]):
                opt_px = _expired_intrinsic(ticker, otype, match["strike"])
            else:
                opt_px = float(match["avg_cost"])
            ok, why = _minimum_hold_exit_guard(
                decision,
                snapshot,
                store,
                match_pos=match,
                position_type="option",
            )
            if not ok:
                return "BLOCKED", why
            ok, why = _underwater_discretionary_exit_guard(
                decision,
                match,
                side="long",
                live_price=float(opt_px),
                is_option=True,
            )
            if not ok:
                return "BLOCKED", why
            notional = opt_px * qty * 100.0
            store.record_trade(
                ticker, action, qty, opt_px, reason,
                expiry=match["expiry"], strike=float(match["strike"]), option_type=otype,
            )
            store.upsert_position(
                ticker, otype, -qty, opt_px,
                expiry=match["expiry"], strike=float(match["strike"]),
            )
            store.update_portfolio(snapshot["cash"] + notional, snapshot["total_value"])
            return (
                "FILLED",
                f"{action} {qty:g} {ticker} {match['strike']}{otype[0].upper()} {match['expiry']} @ {opt_px:.2f}",
            )

        strike_f = float(cls["strike"])
        expiry = cls["expiry"]
        opt_px = market.get_option_price(ticker, expiry, strike_f, otype)
        if not opt_px:
            return "BLOCKED", f"no option price for {ticker} {expiry} {strike_f} {otype}"
        opt_px = float(opt_px)
        credit = opt_px * qty * 100.0
        if mode == "open_csp":
            collat = csp_collateral_required(strike_f, qty, opt_px)
            if float(snapshot["cash"]) + 1e-6 < collat:
                return "BLOCKED", (
                    f"insufficient cash for CSP collateral (need ${collat:.2f} net of premium, "
                    f"have ${snapshot['cash']:.2f})"
                )
        store.record_trade(
            ticker, action, qty, opt_px, reason,
            expiry=expiry, strike=strike_f, option_type=otype,
        )
        store.upsert_position(ticker, otype, -qty, opt_px, expiry=expiry, strike=strike_f)
        store.update_portfolio(snapshot["cash"] + credit, snapshot["total_value"])
        label = "COVERED_CALL" if mode == "open_covered_call" else "CSP"
        return (
            "FILLED",
            f"{label} {action} {qty:g} {ticker} {strike_f}{otype[0].upper()} {expiry} @ {opt_px:.2f}",
        )

    if action in (
        "BUY_CALL_SPREAD", "BUY_PUT_SPREAD", "SELL_CALL_SPREAD", "SELL_PUT_SPREAD",
    ):
        from .analytics.options_structures import parse_vertical_spread, price_vertical_spread
        spec = parse_vertical_spread(decision)
        if not spec.get("ok"):
            return "BLOCKED", spec.get("reason") or "invalid spread"
        priced = price_vertical_spread(market, spec)
        if not priced.get("ok"):
            return "BLOCKED", priced.get("reason") or "spread unpriceable"
        qty_s = float(priced["qty"])
        expiry = priced["expiry"]
        otype = priced["otype"]
        long_k = float(priced["long_strike"])
        short_k = float(priced["short_strike"])
        long_px = float(priced["long_px"])
        short_px = float(priced["short_px"])
        cash_delta = float(priced["cash_delta"])
        cash_after = float(snapshot["cash"]) + cash_delta
        buying_power = _margin_buying_power(snapshot)
        bp_after = buying_power + cash_delta
        if priced.get("is_debit"):
            # Debit verticals spend premium; allow shared cash+50% NW margin BP.
            if bp_after < -1e-6:
                return "BLOCKED", (
                    f"insufficient cash/buying power for debit spread "
                    f"(cash ${snapshot['cash']:.2f}, margin ${_margin_available(snapshot):.2f}, "
                    f"available ${buying_power:.2f}, need ${abs(cash_delta):.2f})"
                )
        else:
            max_loss = float(priced.get("max_loss") or 0.0)
            # Credit vertical residual risk still needs BP buffer after credit.
            if bp_after + 1e-6 < max_loss:
                return "BLOCKED", (
                    f"insufficient cash/buying power buffer for credit spread max loss "
                    f"(bp_after ${bp_after:.2f} < max_loss ${max_loss:.2f})"
                )
        buy_action = "BUY_CALL" if otype == "call" else "BUY_PUT"
        sell_action = "SELL_CALL" if otype == "call" else "SELL_PUT"
        store.record_trade(
            ticker, buy_action, qty_s, long_px,
            f"{action} long leg | {reason}",
            expiry=expiry, strike=long_k, option_type=otype,
        )
        store.record_trade(
            ticker, sell_action, qty_s, short_px,
            f"{action} short leg | {reason}",
            expiry=expiry, strike=short_k, option_type=otype,
        )
        store.upsert_position(ticker, otype, qty_s, long_px, expiry=expiry, strike=long_k)
        store.upsert_position(ticker, otype, -qty_s, short_px, expiry=expiry, strike=short_k)
        store.update_portfolio(cash_after, snapshot["total_value"])
        return (
            "FILLED",
            (
                f"{action} {qty_s:g} {ticker} {long_k}/{short_k} {expiry} "
                f"long@{long_px:.2f} short@{short_px:.2f} cash_delta={cash_delta:+.2f} "
                f"max_loss={float(priced.get('max_loss') or 0):.2f} "
                f"max_gain={float(priced.get('max_gain') or 0):.2f}"
            ),
        )

    return "BLOCKED", f"unknown action {action}"


def _ml_is_qualified() -> tuple[bool, str]:
    """Return (qualified, reason). Cached ML_QUALIFY_TTL_S seconds.
    ML is qualified when median vs_spy_pct over last ML_QUALIFY_MIN_RUNS
    qualifying runs exceeds ML_QUALIFY_MEDIAN_ALPHA."""
    global _ml_qualify_cache
    now = time.time()
    if _ml_qualify_cache and now - _ml_qualify_cache[2] < ML_QUALIFY_TTL_S:
        return _ml_qualify_cache[0], _ml_qualify_cache[1]
    try:
        _db = Path(__file__).resolve().parent.parent / "backtest.db"
        conn = sqlite3.connect(f"file:{_db}?mode=ro", uri=True, timeout=5)
        try:
            rows = conn.execute(
                "SELECT vs_spy_pct FROM backtest_runs "
                "WHERE status='complete' AND vs_spy_pct IS NOT NULL AND n_trades >= 5 "
                "ORDER BY run_id DESC LIMIT ?",
                (ML_QUALIFY_MIN_RUNS,)
            ).fetchall()
        finally:
            conn.close()
        n = len(rows)
        if n < ML_QUALIFY_MIN_RUNS:
            result = (False, f"only {n}/{ML_QUALIFY_MIN_RUNS} qualifying runs")
        else:
            alphas = [r[0] for r in rows]
            median_a = statistics.median(alphas)
            if median_a > ML_QUALIFY_MEDIAN_ALPHA:
                result = (True, f"median alpha {median_a:+.1f}% over last {n} runs")
            else:
                result = (False, f"median alpha {median_a:+.1f}% ≤ threshold")
    except Exception as e:
        result = (False, f"qualification check error: {e}")
    _ml_qualify_cache = (result[0], result[1], now)
    return result[0], result[1]


# Self-contained sentiment / ticker-mapping tables for the ML advisory opinion.
# Deliberately NOT imported from backtest.py (circular-dependency risk) — these
# mirror the backtest engine's scorer vocabulary closely enough for an advisory.
_BULLISH_WORDS_LIVE = {
    "surges", "surge", "rally", "rallies", "gains", "gain", "beats", "beat",
    "rises", "rise", "jumps", "jump", "soars", "soar", "strong", "record",
    "bullish", "outperforms", "boosts", "boost", "breakout",
}
_BEARISH_WORDS_LIVE = {
    "falls", "fall", "drops", "drop", "plunges", "plunge", "slumps", "slump",
    "misses", "miss", "cuts", "cut", "warns", "warn", "weak", "bearish",
    "disappoints", "crashes", "crash", "declines", "decline",
}
_WORD_TO_TICKER_LIVE: dict[str, str] = {
    "nvidia": "NVDA", "amd": "AMD", "apple": "AAPL", "microsoft": "MSFT",
    "amazon": "AMZN", "google": "GOOGL", "alphabet": "GOOGL", "meta": "META",
    "tesla": "TSLA", "intel": "INTC", "micron": "MU", "broadcom": "AVGO",
    "qualcomm": "QCOM", "spy": "SPY", "qqq": "QQQ",
    "semiconductor": "SOXL", "chip": "SOXL", "chips": "SOXL",
    "nasdaq": "TQQQ", "ai": "TQQQ", "artificial intelligence": "TQQQ",
    "oil": "USO", "crude": "USO", "energy": "XLE", "opec": "USO",
    "federal reserve": "TLT", "fed rate": "TLT", "treasury": "TLT",
    "gold": "GLD", "bitcoin": "BTC-USD", "crypto": "COIN",
    "defense": "DFEN", "biotech": "LABU",
    "allbirds": "BIRD", "smartbird": "BIRD",
}
# Pre-compiled word-boundary patterns for `_WORD_TO_TICKER_LIVE` lookup.
# A bare `keyword in title` substring match false-positively triggered on the
# short keys: "ai" matched "China"/"rain"/"Spain"/"trail"; "gold" matched
# "Goldman" (very common in finance news); "intel" matched "intelligence"
# (double-counted with the "ai" → TQQQ mapping). Each silently polluted the
# advisor score on irrelevant articles. The canonical keyword-recovery case
# locked by `test_keyword_mapping_picks_up_unticked_article` ("nvidia surges
# to record on chip demand" → NVDA/SOXL) still matches under \bkeyword\b
# because the keyword appears as a standalone token; the multi-word
# "federal reserve" / "artificial intelligence" still match because \b sits
# between any word/non-word transition (spaces included). Keys are
# lowercased and titles are lowered before matching, so the pattern is
# built from the lowercase keyword.
_WORD_TO_TICKER_LIVE_PATTERNS: dict[str, "re.Pattern[str]"] = {
    kw: re.compile(rf"\b{re.escape(kw)}\b") for kw in _WORD_TO_TICKER_LIVE
}
_LEVERAGED_ETFS_LIVE = {
    "TQQQ", "UPRO", "SPXL", "UDOW", "URTY", "SOXL", "TECL", "FNGU",
    "CURE", "LABU", "NAIL", "DPST", "FAS", "DFEN", "TNA", "UTSL",
    "QLD", "SSO", "NVDU", "MSFU", "AMZU", "TSLL", "CONL", "BITU", "ETHU",
    "SQQQ", "SPXS", "SOXS", "TECS", "FNGD", "SNK",
}


def _ml_live_opinion(
    articles: list[dict],
    quant_sigs: dict[str, dict],
    snap: dict,
    watch_px: dict,
) -> dict | None:
    """Pure ML+quant advisory opinion using live data. Returns action/ticker/reasoning dict.
    Never raises — failure returns None."""
    try:
        # Score tickers from articles
        ticker_scores: dict[str, float] = {}
        ticker_article_count: dict[str, int] = {}
        ticker_max_urgency: dict[str, float] = {}
        for a in articles:
            # Live articles (signals.get_top_signals / get_urgent_articles)
            # carry "ai_score" — NOT "score". Reading only "score" here meant
            # raw_score was always 0.0, so EVERY article was skipped and the
            # news-sentiment half of this advisory was silently dead (it
            # degraded to quant-only, contradicting CLAUDE.md §15). Prefer the
            # live key; keep "score" as a fallback for any backtest-shaped input.
            try:
                raw_score = float(a.get("ai_score") or a.get("score") or 0.0)
            except (TypeError, ValueError):
                raw_score = 0.0
            if raw_score < 1.0:
                continue
            title = (a.get("title") or "").lower()
            # Tokenize on word boundaries, not raw str.split(). A sentiment
            # word that is sentence-final or comma-trailed ("surges," "rally."
            # "beats!") keeps its punctuation under .split(), so `w in SET`
            # exact-membership never matched it — silently zeroing the
            # news-sentiment half of this advisory for the majority of real
            # headlines (a headline almost always ends on or commas around its
            # catalyst verb). CLAUDE.md §15 requires this to mirror the
            # backtest scorer, whose `_sentiment` is punctuation-tolerant;
            # re.findall on [a-z]+ strips the punctuation while preserving the
            # deliberate exact-form vocab (so "mission"/"missile" still do NOT
            # read as the bearish "miss" — exact match, not prefix match).
            words = set(re.findall(r"[a-z]+", title))
            bull = sum(1 for w in words if w in _BULLISH_WORDS_LIVE)
            bear = sum(1 for w in words if w in _BEARISH_WORDS_LIVE)
            total_sent = bull + bear
            sentiment = ((bull - bear) / total_sent) if total_sent else 0.0
            tickers = list(a.get("tickers") or [])
            for keyword, sym in _WORD_TO_TICKER_LIVE.items():
                pat = _WORD_TO_TICKER_LIVE_PATTERNS.get(keyword)
                if pat is not None and pat.search(title) and sym not in tickers:
                    tickers.append(sym)
            try:
                a_urg = float(a.get("urgency", 0.0) or 0.0)
            except (TypeError, ValueError):
                a_urg = 0.0
            for tk in tickers:
                ticker_scores[tk] = ticker_scores.get(tk, 0.0) + raw_score * sentiment
                ticker_article_count[tk] = ticker_article_count.get(tk, 0) + 1
                if a_urg > ticker_max_urgency.get(tk, 0.0):
                    ticker_max_urgency[tk] = a_urg

        # Quant adjustments
        for tk, q in quant_sigs.items():
            adj = 0.0
            rsi = q.get("rsi")
            macd = q.get("macd_signal")
            mom5 = q.get("mom_5d")
            mom20 = q.get("mom_20d")
            bb = q.get("bb_position")
            if isinstance(rsi, (int, float)):
                if rsi < 33: adj += 1.5
                elif rsi < 45: adj += 0.5
                elif rsi > 67: adj -= 1.5
                elif rsi > 55: adj -= 0.5
            if isinstance(macd, (int, float)):
                adj += 0.5 if macd > 0 else -0.5
            if isinstance(mom5, (int, float)):
                adj += min(1.0, max(-1.0, mom5 / 3.0))
            if isinstance(mom20, (int, float)):
                adj += min(0.5, max(-0.5, mom20 / 10.0))
            if isinstance(bb, (int, float)):
                adj -= bb * 0.5
            ticker_scores[tk] = ticker_scores.get(tk, 0.0) + adj

        # Market regime from SPY 20d momentum
        spy_mom20 = quant_sigs.get("SPY", {}).get("mom_20d")
        if isinstance(spy_mom20, (int, float)):
            if spy_mom20 > 3.0:
                regime, regime_mult = "bull", 1.0
            elif spy_mom20 < -3.0:
                regime, regime_mult = "bear", 0.3
            else:
                regime, regime_mult = "sideways", 0.6
        else:
            regime, regime_mult = "unknown", 1.0

        # Pick best directional ticker above threshold. The original advisory
        # only selected positive scores, so bearish ArticleNet evidence could
        # suppress a BUY but could never surface a SHORT. That made the
        # fallback asymmetric exactly where news aggregation matters most.
        buy_ticker: str | None = None
        short_ticker: str | None = None
        best_score = 1.0
        worst_score = -1.0
        for tk, s in ticker_scores.items():
            adj_s = s * regime_mult
            px = watch_px.get(tk)
            if adj_s > best_score and px and px > 0:
                best_score = adj_s
                buy_ticker = tk
            if adj_s < worst_score and px and px > 0:
                worst_score = adj_s
                short_ticker = tk

        if not buy_ticker and not short_ticker:
            return {"action": "HOLD", "ticker": "",
                    "reasoning": f"ML+quant: no high-conviction signal; regime={regime}"}

        if short_ticker and abs(worst_score) >= best_score:
            q_short = quant_sigs.get(short_ticker, {})
            news_count = ticker_article_count.get(short_ticker, 0)
            news_urg = ticker_max_urgency.get(short_ticker, 0.0)
            conviction = min(0.25, abs(worst_score) / 20.0)
            return {
                "action": "SHORT",
                "ticker": short_ticker,
                "reasoning": (
                    f"ML+quant: {short_ticker} bearish_score={worst_score:.2f} "
                    f"regime={regime} RSI={q_short.get('rsi', 'N/A')} "
                    f"news_count={news_count} news_urg={news_urg:.1f} "
                    f"conviction={conviction:.0%}"
                ),
            }

        q_buy = quant_sigs.get(buy_ticker, {})
        news_count = ticker_article_count.get(buy_ticker, 0)
        news_urg = ticker_max_urgency.get(buy_ticker, 0.0)
        if buy_ticker in _LEVERAGED_ETFS_LIVE and regime in ("bull", "sideways"):
            conviction = min(0.40, best_score / 15.0)
        else:
            conviction = min(0.25, best_score / 20.0)
        return {
            "action": "BUY",
            "ticker": buy_ticker,
            "reasoning": (
                f"ML+quant: {buy_ticker} score={best_score:.2f} regime={regime} "
                f"RSI={q_buy.get('rsi', 'N/A')} news_count={news_count} "
                f"news_urg={news_urg:.1f} conviction={conviction:.0%}"
            ),
        }
    except Exception as e:
        print(f"[strategy] _ml_live_opinion error: {e}")
        return None


def _ml_drought_position_qty(
    action: str,
    ticker: str,
    snap: dict,
    watch_px: dict,
    confidence: float,
) -> float | None:
    """Conservative share count for ML drought entries."""
    try:
        price = float(watch_px.get(ticker) or 0.0)
    except (TypeError, ValueError):
        price = 0.0
    if price <= 0:
        try:
            price = float(market.get_price(ticker) or 0.0)
        except Exception:
            price = 0.0
    if price <= 0 or not math.isfinite(price):
        return None
    buying_power = max(0.0, _stock_buying_power(snap))
    if action == "BUY":
        budget = min(
            buying_power * max(0.0, ML_DROUGHT_BUY_CASH_PCT),
            ML_DROUGHT_BUY_MAX_USD,
            buying_power,
        )
        # Scale by confidence, but keep a usable floor once ML is allowed.
        budget *= max(ML_DROUGHT_MIN_CONFIDENCE, min(1.0, confidence))
        if ticker in _LEVERAGED_ETFS_LIVE:
            budget = min(budget, buying_power * 0.10, 750.0)
    else:
        budget = min(
            buying_power * max(0.0, ML_DROUGHT_SHORT_CASH_PCT),
            ML_DROUGHT_SHORT_MAX_USD,
            buying_power,
        )
        budget *= max(ML_DROUGHT_MIN_CONFIDENCE, min(1.0, confidence))
    if budget < price:
        return None
    qty = math.floor((budget / price) * 10000) / 10000
    return qty if qty > 0 else None


def _ml_drought_decision(
    ml_op: dict | None,
    snap: dict,
    watch_px: dict,
    drought_reason: str,
) -> dict | None:
    """Convert the qualified ML advisor output into a drought fallback decision.

    Used only after the LLM path fails. When ML drought entries are enabled,
    qualified BUY/SHORT opinions can open conservative paper positions. HOLD
    still remains the safe default for weak/unknown ML opinions.
    """
    if not isinstance(ml_op, dict):
        return None
    action = (ml_op.get("action") or "HOLD").upper()
    ticker = (ml_op.get("ticker") or "").upper()
    base_reason = str(ml_op.get("reasoning") or "ML fallback")
    reason = (
        f"{base_reason} [ml-drought-fallback: LLM unavailable; "
        f"{drought_reason}]"
    )

    conf_raw = ml_op.get("confidence")
    try:
        confidence = float(conf_raw) if conf_raw is not None else 0.45
    except (TypeError, ValueError):
        confidence = 0.45
    if not math.isfinite(confidence):
        confidence = 0.45
    confidence = max(0.0, min(1.0, confidence))

    if action in ("BUY", "SHORT") and ML_DROUGHT_ALLOW_ENTRIES and ticker:
        qty = _ml_drought_position_qty(action, ticker, snap, watch_px, confidence)
        if qty is not None and qty > 0:
            return {
                "action": action,
                "ticker": ticker,
                "qty": qty,
                "leverage": 1,
                "confidence": max(confidence, ML_DROUGHT_MIN_CONFIDENCE),
                "reasoning": (
                    reason
                    + f" (ML drought entry authorized: {action} {qty:g} {ticker})"
                ),
            }
        return {
            "action": "HOLD",
            "ticker": "",
            "confidence": confidence,
            "reasoning": (
                reason
                + f" (ML wanted {action} {ticker or '?'} but drought sizing "
                f"could not allocate a valid qty)"
            ),
        }

    if action != "BUY":
        return {
            "action": "HOLD",
            "ticker": "",
            "confidence": 0.50,
            "reasoning": reason,
        }
    return {
        "action": "HOLD",
        "ticker": "",
        "confidence": 0.45,
        "reasoning": (
            reason
            + f" (ML wanted BUY {ticker or '?'} but LLM is unavailable; no emergency buy)"
        ),
    }


def decide() -> dict:
    """Run one decision cycle. Returns summary dict for logging."""
    global _quota_exhausted, _last_claude_fail
    _quota_exhausted = False  # per-cycle: reflects only THIS cycle's claude attempts
    # Per-cycle: a host_saturated pre-flight skip leaves `_claude_call` UN-called,
    # so the module-global tag set by the previous cycle's failed call would
    # otherwise leak into THIS cycle's ``summary["last_claude_fail"]`` and
    # ``decisions.reasoning`` "claude returned no response (...)" body. The
    # priority ladder in ``runner._no_decision_cause`` masks the visible effect
    # on the host-saturated path (host check wins), but a non-host fault that
    # happened to follow a host-saturated cycle would still emit the stale tag.
    # Reset here so the per-cycle contract holds regardless of which arm fires.
    _last_claude_fail = None
    store = get_store()
    regular_market_open = market.is_market_open()
    try:
        market_open = market.is_tradable_window_open()
    except Exception:
        market_open = regular_market_open

    snap = _portfolio_snapshot(store)
    # Hard stop-loss guard — executes BEFORE Opus sees the prompt. Defined in
    # SYSTEM_PROMPT ("HARD EXITS" section) so Opus knows the autonomy carve-out.
    # prompt must teach them). Re-snapshot when something fired so cash +
    # positions + total_value all reflect the post-exit book.
    auto_exits: list[str] = _check_and_execute_hard_exits(
        store,
        snap,
        market_open=regular_market_open,
    )
    if auto_exits:
        snap = _portfolio_snapshot(store)

    try:
        from .analytics.friday_closer import maybe_execute_friday_closer
        closer_notes = maybe_execute_friday_closer(
            store,
            snap,
            market_mod=market,
            execute_fn=_execute,
            market_open=regular_market_open,
        )
        if closer_notes:
            auto_exits.extend(closer_notes)
            snap = _portfolio_snapshot(store)
    except Exception as closer_exc:
        auto_exits.append("friday closer error: %s" % closer_exc)

    top = signals.get_top_signals(20, hours=2, min_score=4.0)
    urgent = signals.get_urgent_articles(minutes=30)
    discovery_articles = signals.get_top_signals(
        DISCOVERY_SIGNAL_LIMIT,
        hours=DISCOVERY_SIGNAL_HOURS,
        min_score=DISCOVERY_SIGNAL_MIN_SCORE,
    )
    discovery = _research_discovery(
        discovery_articles,
        WATCHLIST,
        target_size=len(WATCHLIST) * RESEARCH_UNIVERSE_MULTIPLE,
        top_n=DISCOVERY_PRICE_CANDIDATES,
    )
    discovery_tickers = [
        str(r.get("ticker") or "").upper()
        for r in (discovery.get("top") or [])
        if r.get("ticker")
    ]
    price_universe = list(dict.fromkeys([
        *WATCHLIST,
        *discovery_tickers[:DISCOVERY_PRICE_CANDIDATES],
    ]))
    sents = signals.ticker_sentiments(price_universe, hours=4)
    watch_px = market.get_prices(price_universe)
    fut_px = {f: market.get_futures_price(f) for f in FUTURES}
    sp500 = market.benchmark_sp500()

    # Quant signals (RSI/MACD/MA cross) — include held positions + curated subset.
    held_tickers = sorted({p["ticker"] for p in snap["positions"]})
    quant_tickers = sorted(
        set(QUANT_TICKERS_LIVE)
        | set(held_tickers)
        | set(discovery_tickers[:DISCOVERY_QUANT_CANDIDATES])
    )
    try:
        quant_sigs = get_quant_signals_live(quant_tickers)
    except Exception as e:
        print(f"[strategy] quant signals failed: {e}")
        quant_sigs = {}

    # include urgent items at the top
    seen_ids = {s["id"] for s in top}
    merged = [a for a in urgent if a["id"] not in seen_ids] + top
    ml_articles = list(merged)
    seen_ml_ids = {a.get("id") for a in ml_articles}
    for a in discovery_articles:
        if a.get("id") in seen_ml_ids:
            continue
        ml_articles.append(a)
        seen_ml_ids.add(a.get("id"))
    research_discovery_block = _research_discovery_prompt_block(discovery)
    active_research_block: str | None = None
    try:
        active_research_tickers = list(dict.fromkeys([
            *(p.get("ticker") for p in snap.get("positions", []) if p.get("ticker")),
            *WATCHLIST[:20],
            *discovery_tickers[:DISCOVERY_PRICE_CANDIDATES],
            "AAPL", "MSFT", "AMZN", "GOOGL", "META", "ORCL", "AVGO",
        ]))
        active_research = signals.get_active_research_brief(
            active_research_tickers,
            themes=ACTIVE_RESEARCH_THEMES,
            hours=ACTIVE_RESEARCH_HOURS,
            scan_limit=ACTIVE_RESEARCH_SCAN_LIMIT,
        )
        active_research_block = _active_research_prompt_block(active_research)
    except Exception as e:
        print(f"[strategy] active research failed (non-fatal): {e}")
        active_research_block = _active_research_prompt_block({
            "available": False,
            "reason": str(e),
        })

    # Behavioural self-review — feed the trader its own track record (payoff
    # ratio, disposition gap, capital-paralysis state, open-book alpha) so it
    # can self-correct, exactly as a desk reviews its P&L before trading.
    # Advisory only; composes the existing pure builders (single source of
    # truth). Wrapped so a diagnostics failure NEVER blocks a trade — the
    # failure mode is "no mirror this cycle", never "no decision this cycle".
    self_review_block: str | None = None
    try:
        from .analytics.self_review import build_self_review
        sr = build_self_review(
            store.get_portfolio(),
            store.open_positions(),
            store.recent_trades(2000),
            store.recent_decisions(limit=3000),
            store.equity_curve(limit=5000),
        )
        self_review_block = sr.get("prompt_block")
    except Exception as e:
        print(f"[strategy] self-review failed (non-fatal): {e}")

    # Per-name closed-trade memory — the trader's *concrete* outcome on the
    # exact names in play this cycle (verbatim entry/exit reason + objective
    # failure/success mode), composed verbatim from the loser/winner autopsy
    # builders (single source of truth, invariant #10). Filtered to the SAME
    # "names in play" set the quant block uses so the two prompt sections
    # never disagree. Observational only (the self-review precedent); wrapped
    # so a diagnostics failure is "no track-record block this cycle", never
    # "no decision this cycle".
    track_record_block: str | None = None
    try:
        from .analytics.track_record import build_track_record
        tr = build_track_record(
            list(reversed(store.recent_trades(2000))),
            names=_names_in_play(snap.get("positions") or [], merged,
                                  WATCHLIST),
        )
        track_record_block = tr.get("prompt_block")
    except Exception as e:
        print(f"[strategy] track-record failed (non-fatal): {e}")

    durable_memory_block: str | None = None
    try:
        from .analytics.durable_trade_memory import (
            build_durable_memory_prompt_block,
            refresh_lessons_log,
        )
        try:
            refresh_lessons_log()
        except Exception as e:
            print(f"[strategy] durable-memory refresh failed (non-fatal): {e}")
        names_for_mem = set()
        try:
            names_for_mem = set(_names_in_play(snap.get("positions") or [], merged, WATCHLIST))
        except Exception:
            for p in (snap.get("positions") or []):
                if p.get("ticker"):
                    names_for_mem.add(str(p.get("ticker")).upper())
        durable_memory_block = build_durable_memory_prompt_block(
            order_limit=12,
            lesson_limit=6,
            names_in_play=names_for_mem or None,
        )
    except Exception as e:
        print(f"[strategy] durable-memory block failed (non-fatal): {e}")
        durable_memory_block = None

    # Per-name losing-streak watch — composed from the same trades ledger as
    # track_record (single source of truth, invariant #10). Scoped to the SAME
    # `_names_in_play` set so the prompt section only fires on tickers
    # actionable this cycle. Observational only (the track_record precedent;
    # invariants #2/#12); wrapped so a diagnostics failure is "no
    # repeat-loser block this cycle", never "no decision this cycle". The
    # builder reads oldest→newest — same `list(reversed(...))` shape
    # track_record uses.
    repeat_loser_block: str | None = None
    try:
        from .analytics.repeat_loser import build_repeat_loser
        rl = build_repeat_loser(
            list(reversed(store.recent_trades(2000))),
            names=_names_in_play(snap.get("positions") or [], merged,
                                  WATCHLIST),
        )
        repeat_loser_block = rl.get("prompt_block")
    except Exception as e:
        print(f"[strategy] repeat-loser failed (non-fatal): {e}")

    # Open-position thesis drift — re-tests each holding against the verbatim
    # reason it was opened for, scored against current quant + news. Pure
    # arithmetic over the already-marked snapshot + the in-hand quant signals
    # (NO extra store read / NO network — the risk_mirror hot-path
    # discipline). All-INTACT collapses to a `None` block — the silence
    # precedent — so a healthy book produces no section. Observational only
    # (invariants #2/#12 — the track_record precedent); wrapped so a
    # diagnostics fault is "no thesis-drift block this cycle", never "no
    # decision this cycle".
    thesis_drift_block: str | None = None
    try:
        from .analytics.thesis_drift import build_thesis_drift
        # Remap the live quant_sigs (`MACD` uppercase string label, `rsi` /
        # `mom_5d` lowercase numerics) into the lowercase shape the builder
        # consumes (mirrors the `/api/thesis-drift` endpoint's exact remap so
        # the prompt block and the dashboard panel cannot disagree on which
        # positions are WEAKENING/BROKEN). News_count is intentionally
        # omitted — the builder degrades off it (the entry_cited_news +
        # news_cold heuristic stays dormant) and pulling per-ticker news on
        # the live cycle is a latency hazard (the risk_mirror discipline).
        td_signals = {
            tk: {
                "rsi": q.get("rsi"),
                "macd": q.get("MACD"),
                "mom_5d": q.get("mom_5d"),
                "mom_20d": q.get("mom_20d"),
            }
            for tk, q in (quant_sigs or {}).items() if q
        }
        td = build_thesis_drift(
            snap.get("positions") or [],
            store.recent_trades(2000),
            signals=td_signals or None,
        )
        thesis_drift_block = td.get("prompt_block")
    except Exception as e:
        print(f"[strategy] thesis-drift failed (non-fatal): {e}")

    # Concentration + churn mirror — the trader's *structural* risk (how
    # bunched the book is + how much it churns), composed verbatim from the
    # correlation/churn builders (single source of truth, invariant #10).
    # No price history is fetched here on purpose: a per-position yfinance
    # call is a latency/flake risk on a live cycle, and the weight-based
    # concentration (top weight / HHI / effective names) still fires without
    # it. Observational only (the self-review precedent); wrapped so a
    # diagnostics fault is "no risk-mirror block this cycle", never "no
    # decision this cycle". `recent_trades` is passed store-native
    # (newest-first); build_risk_mirror reverses it for build_churn itself.
    risk_mirror_block: str | None = None
    try:
        from .analytics.risk_mirror import build_risk_mirror
        rm = build_risk_mirror(
            store.recent_trades(2000),
            snap.get("positions") or [],
        )
        risk_mirror_block = rm.get("prompt_block")
    except Exception as e:
        print(f"[strategy] risk-mirror failed (non-fatal): {e}")

    # Live-book SECTOR concentration — risk_mirror closed *name*-level
    # concentration; the documented #3 pathology is one dimension over,
    # *sector* clustering ("the book 60.9% in one name's sector ... the
    # decision engine itself never saw [it]"). Pure arithmetic over the
    # already-marked snapshot + a static SECTOR_MAP (NO extra store read / NO
    # network — the risk_mirror hot-path discipline). Scoped to the SAME
    # `_names_in_play` set the quant / track-record / buying-power blocks use
    # so the marginal "what would I be adding to" view matches "what matters
    # this cycle". Observational only (invariants #2/#12 — the buying_power
    # precedent); wrapped so a diagnostics fault is "no sector block this
    # cycle", never "no decision this cycle".
    sector_exposure_block: str | None = None
    try:
        from .analytics.sector_exposure import build_sector_exposure
        sx = build_sector_exposure(
            snap,
            _names_in_play(snap.get("positions") or [], merged, WATCHLIST),
        )
        sector_exposure_block = sx.get("prompt_block")
    except Exception as e:
        print(f"[strategy] sector-exposure failed (non-fatal): {e}")

    # Forward beta/concentration STRESS — risk_mirror/sector_exposure tell the
    # trader HOW concentrated the book is; this tells it what a routine
    # adverse move COSTS that book in dollars, on day one (tail_risk needs
    # ≥20 daily returns and is dark on a young book). Pure arithmetic over the
    # already-marked snapshot + the pinned sector→beta SSOT (NO extra store
    # read / NO network — the risk_mirror hot-path discipline). classify is
    # sector_exposure's test-pinned copy and _LEVERAGE_BETA is stress's, so
    # neither pulls the ~9k-line dashboard onto the decision hot path yet both
    # are CI-pinned to /api/risk. Observational only (invariants #2/#12 — the
    # sector_exposure precedent); wrapped so a diagnostics fault is "no stress
    # block this cycle", never "no decision this cycle".
    stress_block: str | None = None
    try:
        from .analytics.sector_exposure import classify as _sx_classify
        from .analytics.stress_scenarios import (
            _LEVERAGE_BETA as _ss_beta,
            build_stress_scenarios,
        )
        st = build_stress_scenarios(
            snap.get("positions") or [],
            snap.get("total_value") or 0.0,
            _sx_classify,
            _ss_beta,
        )
        stress_block = st.get("prompt_block")
    except Exception as e:
        print(f"[strategy] stress-scenarios failed (non-fatal): {e}")

    # Portfolio-construction target — uses only data already in this cycle
    # (snapshot, merged signals, quant_sigs, watch_px). No extra DB read and
    # no extra yfinance call on the live path. Observational only: it gives
    # Opus a concrete allocation/risk/why target for self-checking, never a
    # directive or cap.
    portfolio_construction_block: str | None = None
    try:
        from .analytics.portfolio_construction import (
            build_portfolio_construction,
        )
        from .analytics.sector_exposure import classify as _pc_classify
        from .analytics.stress_scenarios import _LEVERAGE_BETA as _pc_beta
        pc = build_portfolio_construction(
            snap,
            merged,
            quant_sigs,
            watch_px,
            _pc_classify,
            _pc_beta,
            risk_tolerance="medium",
            time_horizon="1-3 years",
            fallback_tickers=["SPY", "QQQ", *WATCHLIST[:10]],
        )
        portfolio_construction_block = pc.get("prompt_block")
    except Exception as e:
        print(f"[strategy] portfolio-construction failed (non-fatal): {e}")

    # Forward scheduled-event awareness — the #1 thing a discretionary desk
    # tracks that the engine was fully blind to: upcoming EARNINGS on the
    # names in play. Reads digital-intern's earnings_calendar.json snapshot
    # *directly from disk* (the signals.py filesystem precedent) — NOT a
    # :8080 hop, which is a documented hang/latency hazard on the live cycle.
    #
    # Scope is held ∪ the FULL WATCHLIST — deliberately NOT the lean
    # `_names_in_play` set the quant / track-record blocks use. Those blocks
    # are per-ticker and large, so they trim to top-5 + signals to bound
    # prompt length; an earnings event within the 14d horizon is rare (≈0–3
    # across the whole 50-name watchlist) so there is no bloat to bound, and
    # narrowing it would re-create the exact blind spot this feature closes:
    # Opus could BUY a watchlist name (e.g. NVDA — not in WATCHLIST[:5]) the
    # day before its print with no idea it was coming. This also keeps the
    # decision-path scope identical to `/api/event-calendar` so the endpoint
    # truly shows "the block the trader saw". Do not narrow to
    # `_names_in_play` for "consistency" — that silently re-blinds the desk.
    # Observational only (invariants #2/#12 — the self-review precedent);
    # wrapped so a missing/stale/corrupt snapshot is "no event block this
    # cycle", never "no decision this cycle".
    event_calendar_block: str | None = None
    try:
        from .analytics.event_calendar import build_event_calendar
        positions = snap.get("positions") or []
        held = {(p.get("ticker") or "").upper()
                for p in positions if p.get("ticker")}
        ec = build_event_calendar(
            positions,
            held | {t.upper() for t in WATCHLIST},
        )
        event_calendar_block = ec.get("prompt_block")
    except Exception as e:
        print(f"[strategy] event-calendar failed (non-fatal): {e}")

    # Forward MACRO awareness — scheduled FOMC rate decisions. The macro
    # sibling of event_calendar, one dimension over: a rate-decision surprise
    # moves the WHOLE book in one instant (this watchlist is leveraged-ETF
    # heavy — SOXL/TQQQ/NVDL — exactly what gaps hardest on a Fed surprise),
    # so it is market-wide (no positions arg) and always built. A pure
    # static-table call: NO store read, NO file / network I/O (even safer
    # than event_calendar's disk read — the risk_mirror hot-path discipline).
    # Observational only (invariants #2/#12 — the event_calendar precedent);
    # wrapped so a diagnostics fault is "no macro block this cycle", never
    # "no decision this cycle".
    macro_calendar_block: str | None = None
    try:
        from .analytics.macro_calendar import build_macro_calendar
        macro_calendar_block = build_macro_calendar().get("prompt_block")
    except Exception as e:
        print(f"[strategy] macro-calendar failed (non-fatal): {e}")

    # Deployable-cash awareness — the lean, prompt-facing complement to the
    # dashboard-only capital_paralysis. Pure arithmetic over the already-marked
    # snapshot + the already-fetched watch prices (NO extra store read / NO
    # network — the risk_mirror hot-path discipline). Scoped to the SAME
    # `_names_in_play` set the quant / track-record blocks use so the
    # affordability sizing matches "what matters this cycle". Observational
    # only (invariants #2/#12 — the event_calendar precedent); wrapped so a
    # diagnostics fault is "no buying-power block this cycle", never "no
    # decision this cycle".
    buying_power_block: str | None = None
    try:
        from .analytics.buying_power import build_buying_power
        bp = build_buying_power(
            snap, watch_px,
            _names_in_play(snap.get("positions") or [], merged, WATCHLIST),
        )
        buying_power_block = bp.get("prompt_block")
    except Exception as e:
        print(f"[strategy] buying-power failed (non-fatal): {e}")

    # Forward mechanical-exit proximity — surfaces "which open lot is
    # within striking distance of its SL/TP this cycle". Today Opus reads
    # the static SYSTEM_PROMPT rule (5%/15% — 10%/25% leveraged) and the
    # current mark / P/L from the position lines and has to mentally
    # threshold by re-deriving SL/TP from avg_cost — but on a blended lot
    # the stored SL/TP is re-anchored to the new blended cost, NOT
    # static entry bracket, so the live numbers are concretely useful. Pure builder
    # over the already-marked snapshot positions (NO extra store read /
    # NO network — the risk_mirror hot-path discipline). Identical SSOT
    # used by ``/api/exit-proximity`` + the Discord hourly line
    # ``reporter._exit_proximity_line`` so the three surfaces (dashboard,
    # Discord, prompt) tell the same story on the same data (invariant
    # #10). Returns ``None`` on a COMFORTABLE / NO_DATA / NO_SL_TP_SET
    # book — the silence-when-nothing-actionable precedent — so a
    # healthy book adds zero prompt bloat. Observational only (invariants
    # #2/#12); wrapped so a diagnostics fault is "no exit-proximity
    # block this cycle", never "no decision this cycle".
    exit_proximity_block: str | None = None
    try:
        from .analytics.exit_proximity import build_exit_proximity
        ep = build_exit_proximity(snap.get("positions") or [])
        exit_proximity_block = ep.get("prompt_block")
    except Exception as e:
        print(f"[strategy] exit-proximity failed (non-fatal): {e}")

    # Options desk — first-class option awareness for the live decision.
    # Bounded chain fetches for held + names-in-play underlyings (swing ~45d
    # and LEAPS ~365d tenors). Observational only; degrade-safe.
    options_desk_block: str | None = None
    try:
        from .analytics.options_desk import build_options_desk
        od = build_options_desk(
            snap,
            watch_px,
            _names_in_play(snap.get("positions") or [], merged, WATCHLIST),
            max_underlyings=3,
            fetch_chains=True,
        )
        options_desk_block = od.get("prompt_block")
        if od.get("summary"):
            print(f"[strategy] options-desk: {od.get('summary')}")
    except Exception as e:
        print(f"[strategy] options-desk failed (non-fatal): {e}")


    # ML advisor: when model consistently beats SPY, include its opinion in prompt
    ml_opinion_block: str | None = None
    ml_op: dict | None = None
    ml_qualified, ml_qual_reason = _ml_is_qualified()
    if ml_qualified:
        try:
            ml_op = _ml_live_opinion(ml_articles, quant_sigs, snap, watch_px)
            if ml_op:
                ml_opinion_block = (
                    f"ML MODEL OPINION ({ml_qual_reason}):\n"
                    f"  Action: {ml_op['action']}"
                    + (f" {ml_op['ticker']}" if ml_op.get("ticker") else "")
                    + f"\n  Reasoning: {ml_op['reasoning']}\n"
                    "This is an advisory opinion only. You retain full autonomy over the final decision.\n"
                    "If the ML lean is bullish/bearish with real conviction, also consider expressing "
                    "it via BUY_CALL / BUY_PUT using OPTIONS DESK chain snapshots (LEAPS proxy or "
                    "swing) when premium risk is the better fit than full share notional."
                )
                print(f"[strategy] ML advisor: {ml_op['action']} {ml_op.get('ticker', '')}")
        except Exception as e:
            print(f"[strategy] ML advisor failed (non-fatal): {e}")

    payload = _build_payload(snap, merged, sents, watch_px, fut_px, sp500, market_open,
                             quant_signals=quant_sigs,
                             self_review_block=self_review_block,
                             track_record_block=track_record_block,
                             durable_memory_block=durable_memory_block,
                             repeat_loser_block=repeat_loser_block,
                             thesis_drift_block=thesis_drift_block,
                             risk_mirror_block=risk_mirror_block,
                             sector_exposure_block=sector_exposure_block,
                             stress_block=stress_block,
                             portfolio_construction_block=portfolio_construction_block,
                             event_calendar_block=event_calendar_block,
                             macro_calendar_block=macro_calendar_block,
                             buying_power_block=buying_power_block,
                             exit_proximity_block=exit_proximity_block,
                             research_discovery_block=research_discovery_block,
                             active_research_block=active_research_block,
                             options_desk_block=options_desk_block)
    prompt = f"{SYSTEM_PROMPT}\n\n---\nCONTEXT:\n{payload}"
    if ml_opinion_block:
        prompt += f"\n\n---\nML ADVISOR:\n{ml_opinion_block}"

    # Pre-flight host-saturation guard. The recurring live-trader NO_DECISION
    # storms are host saturation (out-of-band parallel Opus review/HYBRID
    # agents + the backtest committee starving the 15GB box), NOT a prompt or
    # parser bug — see paper_trader/host_guard.py. Spawning a doomed 180s Opus
    # subprocess (and its Sonnet fallback) during a storm just OOM-thrashes the
    # box (+1.5GB each) and still records NO_DECISION. Skip every claude call
    # this cycle and record a distinct, honest reason instead. Degrade-safe:
    # any probe failure falls through to the normal path (never blocks a
    # decision — "non-fatal by construction", like the rest of decide()).
    host_sat, host_sat_reason = False, ""
    try:
        host_sat, host_sat_reason = host_saturated()
    except Exception as e:
        print(f"[strategy] host_saturated probe failed (ignoring): {e}")
        host_sat = False

    if host_sat:
        print(f"[strategy] skipping claude call — {host_sat_reason}")
        raw = None
    else:
        raw = _claude_call(prompt)
    decision = _parse_decision(raw) if raw else None
    retried = False
    fallback_used = False
    fb_prompt = None

    # Mid-call host-saturation re-probe. The pre-flight guard above is a single
    # point-in-time check taken *before* a ~180s Opus window. The continuous
    # backtest committee (_CLAUDE_SEM=3), the hourly-review/HYBRID agents and
    # _opus_annotate spawn `claude --model claude-opus` subprocesses out of
    # band, so a cycle routinely passes pre-flight (<=4 concurrent) and then
    # the box saturates *during* the call — the Opus subprocess is OOM-starved,
    # returns empty, and we would otherwise spawn a doomed +1.5GB Sonnet
    # fallback into the very storm that just killed the first call AND record
    # it as "claude returned no response" (the model-timeout bucket), hiding a
    # host problem from /api/empty-claude-rate. Observed live 2026-05-18: the
    # dominant NO_DECISION reason was the empty-response signature, not the
    # pre-flight skip. When the first call came back empty and the box is NOW
    # saturated, treat it exactly like the pre-flight skip. Degrade-safe: any
    # probe error falls through to the existing Sonnet-fallback path (the
    # genuine model-timeout case is unchanged).
    if raw is None and not host_sat:
        try:
            now_sat, now_reason = host_saturated()
        except Exception as e:
            print(f"[strategy] mid-call host_saturated probe failed (ignoring): {e}")
            now_sat = False
        if now_sat:
            host_sat = True
            host_sat_reason = ("host saturated mid-call: "
                               + now_reason.split("host saturated: ", 1)[-1])
            print(f"[strategy] skipping Sonnet fallback — {host_sat_reason}")

    # True timeout (raw is None, so _should_retry_parse is False): Opus blew
    # past its budget. Retrying the full prompt on Opus would just stall again,
    # so fall back to a faster model with a condensed prompt instead of
    # recording a NO_DECISION. Skipped entirely when the host-saturation guard
    # (pre-flight OR the mid-call re-probe just above) declined this cycle —
    # spawning the Sonnet fallback would just add another 1.5GB subprocess to
    # the storm we are trying to dodge.
    if raw is None and not host_sat:
        print("[strategy] primary model empty — trying fallback model")
        fb_payload = _build_fallback_payload(snap, merged, quant_sigs)
        fb_prompt = f"{SYSTEM_PROMPT}\n\n---\nCONTEXT (condensed):\n{fb_payload}"
        raw = _claude_call(fb_prompt, timeout_s=FALLBACK_TIMEOUT_S,
                           model=FALLBACK_MODEL)
        if raw:
            fallback_used = True
            print("[strategy] fallback model returned response")
            decision = _parse_decision(raw)

    # Conditional one-shot retry: Claude returned text but it wasn't parseable.
    # A None response (timeout / empty stdout) won't be rescued by a retry —
    # same prompt, same failure — so we skip retrying in that case.
    if not decision and not fallback_used and _should_retry_parse(raw):
        retried = True
        print("[strategy] parse failed; retrying with JSON-only nudge")
        retry_raw = _claude_call(prompt + _RETRY_SUFFIX, timeout_s=RETRY_TIMEOUT_S)
        retry_decision = _parse_decision(retry_raw) if retry_raw else None
        if retry_decision:
            raw, decision = retry_raw, retry_decision
        else:
            # Keep first raw for diagnostics if retry also failed but the first
            # was actually empty; otherwise prefer the more recent attempt so
            # operators see what the retry looked like.
            if retry_raw:
                raw = retry_raw

    # JSON nudge retry for Sonnet fallback parse failure
    elif not decision and fallback_used and _should_retry_parse(raw) and fb_prompt is not None:
        retried = True
        print("[strategy] fallback parse failed; retrying with JSON-only nudge")
        retry_raw = _claude_call(fb_prompt + _RETRY_SUFFIX, timeout_s=RETRY_TIMEOUT_S,
                                 model=FALLBACK_MODEL)
        retry_decision = _parse_decision(retry_raw) if retry_raw else None
        if retry_decision:
            raw, decision = retry_raw, retry_decision
        elif retry_raw:
            raw = retry_raw

    summary = {
        "market_open": market_open,
        "signal_count": len(merged),
        "auto_exits": auto_exits,
        "decision": decision,
        "raw": raw,
        "snapshot": snap,
        "status": "NO_DECISION",
        "detail": "",
        "retried": retried,
        "fallback_used": fallback_used,
        # True only when every claude attempt this cycle was rejected with a
        # quota/usage-limit error (a frozen-trader signal, not a transient
        # miss). runner._cycle dedupes the Discord alarm off this.
        "quota_exhausted": bool(_quota_exhausted and not decision),
        # True when the pre-flight guard skipped the claude call(s) this cycle
        # because the host was saturated. Distinct from a real model timeout —
        # no subprocess was spawned. runner._cycle still counts this as a
        # NO_DECISION for the auto-recovery breaker (reaping stale claude procs
        # is the right response to a saturated box either way).
        "host_saturated": host_sat,
        # Cause code from the most recent ``_claude_call`` this cycle (None on
        # success). Surfaced so ``runner._no_decision_cause`` can label a
        # raw=None NO_DECISION with a SPECIFIC cause (timeout / nonzero_rc /
        # empty_stdout / cli_missing / exception) in the breaker Discord alert
        # instead of degrading to ``""``. Single-cycle state, never sticky.
        "last_claude_fail": _last_claude_fail,
        "ml_fallback_used": False,
    }

    if not decision:
        # Capture an excerpt of what Claude actually returned so we can
        # diagnose parse failures from the dashboard / DB instead of staring
        # at a generic "no parseable JSON" line.
        if _quota_exhausted:
            reason_text = "claude quota/usage limit exhausted (no decision)"
        elif host_sat:
            # Distinct prefix on purpose: /api/empty-claude-rate and
            # model_reliability key off "claude returned no response" — a
            # deliberate skip is NOT a model timeout and must not inflate the
            # empty/timeout rate. /api/host-guard surfaces this bucket.
            reason_text = f"skipped claude call — {host_sat_reason}"
        elif raw:
            excerpt = raw[:RAW_CAPTURE_CHARS].replace("\x00", "")
            tag = "retry_failed" if retried else "parse_failed"
            reason_text = f"{tag}: {excerpt}"
        else:
            # Sub-bucket the empty-response case using the per-call cause code
            # _claude_call just set. Keeps the literal "claude returned no
            # response" prefix verbatim — /api/empty-claude-rate and
            # model_reliability key off that substring (strategy.py:1641) and
            # the existing no_decision_reasons.model_empty bucket stays the
            # back-compat fallback for any code path that didn't set a cause.
            # The new analytics sub-buckets read the parenthesised suffix.
            cause = _last_claude_fail or "timeout/empty"
            reason_text = f"claude returned no response ({cause})"
        if ml_qualified and ml_op:
            ml_decision = _ml_drought_decision(
                ml_op, snap, watch_px, reason_text,
            )
            if ml_decision:
                print(
                    "[strategy] ML drought fallback: "
                    f"{ml_decision.get('action')} {ml_decision.get('ticker', '')}"
                )
                ml_status, ml_detail = _execute(ml_decision, snap, store)
                summary["decision"] = ml_decision
                summary["status"] = ml_status
                summary["detail"] = ml_detail
                summary["ml_fallback_used"] = True
                action_label = (
                    f"{ml_decision.get('action','?')} "
                    f"{ml_decision.get('ticker','')}"
                ).strip()
                store.record_decision(
                    market_open,
                    len(merged),
                    f"{action_label} → {ml_status}",
                    json.dumps({
                        "decision": ml_decision,
                        "auto_exits": auto_exits,
                        "detail": ml_detail,
                        "fallback_used": fallback_used,
                        "ml_fallback_used": True,
                        "llm_drought_reason": reason_text,
                    }),
                    snap["total_value"],
                    snap["cash"],
                )
                final = _portfolio_snapshot(store)
                store.record_equity_point(final["total_value"], final["cash"], sp500)
                summary["snapshot"] = final
                return summary
        store.record_decision(market_open, len(merged), "NO_DECISION",
                              reason_text,
                              snap["total_value"], snap["cash"])
        store.record_equity_point(snap["total_value"], snap["cash"], sp500)
        return summary

    status, detail = _execute(decision, snap, store)
    summary["status"] = status
    summary["detail"] = detail

    action_label = f"{decision.get('action','?')} {decision.get('ticker','')}".strip()
    if fallback_used:
        # Make it visible in the DB / dashboard that this decision came from
        # the Sonnet fallback (condensed context), not the full Opus pass.
        decision = {**decision,
                    "reasoning": f"{decision.get('reasoning', '')} [model-fallback]"}
    store.record_decision(
        market_open,
        len(merged),
        f"{action_label} → {status}",
        json.dumps({"decision": decision, "auto_exits": auto_exits,
                    "detail": detail, "fallback_used": fallback_used}),
        snap["total_value"],
        snap["cash"],
    )
    # final mark + equity point
    final = _portfolio_snapshot(store)
    store.record_equity_point(final["total_value"], final["cash"], sp500)
    summary["snapshot"] = final

    # Durable trade learning: local JSONL always; claude-mem worker on FILLED.
    # Non-fatal — never blocks the decision cycle.
    try:
        from .analytics.claude_mem_trades import record_trade_learning
        if status in ("FILLED", "BLOCKED") and str(decision.get("action") or "").upper() != "HOLD":
            mem = record_trade_learning(
                decision=decision,
                status=status,
                detail=str(detail) if detail is not None else None,
                snapshot_before=snap,
                snapshot_after=final,
                push_remote=(status == "FILLED"),
            )
            summary["trade_memory"] = {
                "ok": mem.get("ok"),
                "local_path": mem.get("local_path"),
                "remote_ok": (mem.get("remote") or {}).get("ok"),
            }
    except Exception as e:
        print(f"[strategy] trade-memory failed (non-fatal): {e}")

    return summary


if __name__ == "__main__":
    import pprint
    pprint.pp(decide())

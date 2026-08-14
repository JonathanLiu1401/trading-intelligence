"""Push paper-trader fills into claude-mem + local JSONL trade memory.

Uses the real claude-mem worker APIs:
  POST /api/sessions/init
  POST /api/sessions/observations

Local JSONL under data/trade_memory/ always works offline.
Never raises into decide() / runner.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_WORKER = os.environ.get("CLAUDE_MEM_WORKER_URL", "http://127.0.0.1:37701")
SESSION_ID = os.environ.get("PAPER_TRADER_CLAUDE_MEM_SESSION", "paper-trader-live")
PROJECT = os.environ.get("PAPER_TRADER_CLAUDE_MEM_PROJECT", "paper-trader")


def _default_memory_dir() -> Path:
    here = Path(__file__).resolve()
    root = here.parents[2]  # paper-trader
    return root / "data" / "trade_memory"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_trade_observation(
    *,
    decision: dict | None,
    status: str,
    detail: str | None,
    snapshot_before: dict | None = None,
    snapshot_after: dict | None = None,
) -> dict:
    d = decision or {}
    action = str(d.get("action") or "").upper()
    ticker = str(d.get("ticker") or "").upper()
    is_option = action in {"BUY_CALL", "BUY_PUT", "SELL_CALL", "SELL_PUT"} or bool(
        d.get("strike") or d.get("expiry")
    )
    body = {
        "ts": _iso_now(),
        "kind": "paper_trade",
        "status": status,
        "detail": detail,
        "action": action,
        "ticker": ticker,
        "qty": d.get("qty"),
        "strike": d.get("strike"),
        "expiry": d.get("expiry"),
        "confidence": d.get("confidence"),
        "reasoning": d.get("reasoning"),
        "is_option": is_option,
        "cash_before": (snapshot_before or {}).get("cash"),
        "equity_before": (snapshot_before or {}).get("total_value"),
        "cash_after": (snapshot_after or {}).get("cash"),
        "equity_after": (snapshot_after or {}).get("total_value"),
    }
    opt_bits = ""
    if is_option:
        opt_bits = " strike=%s expiry=%s" % (d.get("strike"), d.get("expiry"))
    text = (
        "PAPER TRADE %s: %s %s%s qty=%s conf=%s | %s | detail=%s"
        % (
            status,
            action,
            ticker,
            opt_bits,
            d.get("qty"),
            d.get("confidence"),
            d.get("reasoning") or "",
            detail or "",
        )
    )
    body["text"] = " ".join(str(text).split())
    return body


def append_local_memory(obs: dict, memory_dir: Path | None = None) -> Path:
    directory = memory_dir or _default_memory_dir()
    directory.mkdir(parents=True, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = directory / ("trades-%s.jsonl" % day)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obs, ensure_ascii=False) + chr(10))
    latest = directory / "latest.json"
    latest.write_text(json.dumps(obs, indent=2), encoding="utf-8")
    return path


def _http_json(url: str, payload: dict, timeout: float = 5.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()[:2000]
        try:
            parsed = json.loads(raw.decode("utf-8", errors="replace") or "{}")
        except json.JSONDecodeError:
            parsed = {"raw": raw.decode("utf-8", errors="replace")}
        return {
            "ok": True,
            "url": url,
            "http_status": getattr(resp, "status", None),
            "body": parsed,
        }


def ensure_claude_mem_session(
    worker_url: str | None = None,
    session_id: str | None = None,
    project: str | None = None,
) -> dict:
    """POST /api/sessions/init for the durable paper-trader memory session."""
    base = (worker_url or DEFAULT_WORKER).rstrip("/")
    sid = session_id or SESSION_ID
    try:
        return _http_json(
            base + "/api/sessions/init",
            {
                "contentSessionId": sid,
                "project": project or PROJECT,
                "prompt": "paper-trader trade learning session",
                "platform": "openclaw",
            },
        )
    except Exception as e:
        return {"ok": False, "error": "%s: %s" % (type(e).__name__, e), "session": sid}


def push_claude_mem_observation(
    obs: dict,
    worker_url: str | None = None,
    session_id: str | None = None,
    project: str | None = None,
) -> dict:
    """Ingest via real claude-mem worker: sessions/init + sessions/observations."""
    base = (worker_url or DEFAULT_WORKER).rstrip("/")
    sid = session_id or SESSION_ID
    init = ensure_claude_mem_session(worker_url=base, session_id=sid, project=project)
    if not init.get("ok"):
        return {"ok": False, "stage": "init", "init": init}
    payload = {
        "contentSessionId": sid,
        "tool_name": "paper_trader.fill",
        "tool_input": {
            "action": obs.get("action"),
            "ticker": obs.get("ticker"),
            "qty": obs.get("qty"),
            "strike": obs.get("strike"),
            "expiry": obs.get("expiry"),
            "is_option": obs.get("is_option"),
            "confidence": obs.get("confidence"),
            "reasoning": obs.get("reasoning"),
        },
        "tool_response": {
            "status": obs.get("status"),
            "detail": obs.get("detail"),
            "text": obs.get("text"),
            "cash_before": obs.get("cash_before"),
            "cash_after": obs.get("cash_after"),
            "equity_before": obs.get("equity_before"),
            "equity_after": obs.get("equity_after"),
        },
        "cwd": str(_default_memory_dir().parents[1]),
    }
    try:
        res = _http_json(base + "/api/sessions/observations", payload)
        res["init"] = init
        res["session"] = sid
        return res
    except urllib.error.HTTPError as e:
        return {
            "ok": False,
            "stage": "observations",
            "error": "HTTPError %s" % e.code,
            "session": sid,
        }
    except Exception as e:
        return {
            "ok": False,
            "stage": "observations",
            "error": "%s: %s" % (type(e).__name__, e),
            "session": sid,
        }


def _write_memory_markdown_crumb(obs: dict, memory_dir: Path | None = None):
    try:
        directory = (memory_dir or _default_memory_dir()) / "crumbs"
        directory.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        ticker = obs.get("ticker") or "UNK"
        action = obs.get("action") or "TRADE"
        path = directory / ("%s_%s_%s.md" % (ts, action, ticker))
        content_lines = [
            "# Paper trade memory crumb",
            "",
            "- time: %s" % obs.get("ts"),
            "- status: %s" % obs.get("status"),
            "- action: %s" % action,
            "- ticker: %s" % ticker,
            "- option: %s" % obs.get("is_option"),
            "- strike: %s" % obs.get("strike"),
            "- expiry: %s" % obs.get("expiry"),
            "- qty: %s" % obs.get("qty"),
            "- confidence: %s" % obs.get("confidence"),
            "- detail: %s" % obs.get("detail"),
            "- reasoning: %s" % obs.get("reasoning"),
            "- text: %s" % obs.get("text"),
            "",
        ]
        path.write_text(chr(10).join(content_lines), encoding="utf-8")
        return path
    except Exception as e:
        print("[claude_mem_trades] crumb write failed: %s" % e)
        return None


def record_trade_learning(
    *,
    decision: dict | None,
    status: str,
    detail: str | None = None,
    snapshot_before: dict | None = None,
    snapshot_after: dict | None = None,
    push_remote: bool = True,
    memory_dir: Path | None = None,
    worker_url: str | None = None,
) -> dict:
    """Local JSONL always; claude-mem HTTP best-effort on FILLED."""
    try:
        action = str((decision or {}).get("action") or "").upper()
        if action in {"", "HOLD"} and status not in {"FILLED", "BLOCKED"}:
            return {"skipped": True, "reason": "hold_or_empty"}
        obs = format_trade_observation(
            decision=decision,
            status=status,
            detail=detail,
            snapshot_before=snapshot_before,
            snapshot_after=snapshot_after,
        )
        path = append_local_memory(obs, memory_dir=memory_dir)
        crumb = _write_memory_markdown_crumb(obs, memory_dir=memory_dir)
        lessons_path = None
        try:
            from .durable_trade_memory import refresh_lessons_log
            lessons_path = refresh_lessons_log(memory_dir=memory_dir)
        except Exception as e:
            print("[claude_mem_trades] lessons refresh failed: %s" % e)
        remote: dict[str, Any] = {"ok": False, "skipped": True}
        if push_remote and status == "FILLED":
            remote = push_claude_mem_observation(obs, worker_url=worker_url)
        return {
            "ok": True,
            "local_path": str(path),
            "crumb_path": str(crumb) if crumb else None,
            "lessons_path": str(lessons_path) if lessons_path else None,
            "observation": obs,
            "remote": remote,
        }
    except Exception as e:
        print("[claude_mem_trades] record failed (non-fatal): %s" % e)
        return {"ok": False, "error": str(e)}


def recent_trade_memory(limit: int = 10, memory_dir: Path | None = None) -> list:
    directory = memory_dir or _default_memory_dir()
    if not directory.exists():
        return []
    files = sorted(directory.glob("trades-*.jsonl"))
    rows = []
    for path in files[-5:]:
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError:
            continue
    return rows[-limit:]

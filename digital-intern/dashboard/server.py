"""
Digital Intern — real-time web dashboard (FastAPI).

Reads directly from the article store + log files; exposes JSON endpoints and a
WebSocket that pushes stats, articles, and health every 5 seconds.
"""
from __future__ import annotations

import asyncio
import json
import os
import plistlib
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from storage.article_store import ArticleStore, _LIVE_ONLY_CLAUSE  # noqa: E402

DASHBOARD_DIR = Path(__file__).resolve().parent
DASHBOARD_HTML = DASHBOARD_DIR / "dashboard.html"
COMMAND_CENTER_HTML = DASHBOARD_DIR / "command_center.html"
SECTION_PAGE_HTML = DASHBOARD_DIR / "section_page.html"
STRUCTURED_LOG = ROOT / "logs" / "structured.jsonl"
METRICS_LOG = ROOT / "logs" / "metrics.jsonl"
# Atomic JSON snapshot written by the daemon supervisor — preferred source
# of truth for worker state (avoids re-parsing logs).
SUPERVISOR_STATE = ROOT / "logs" / "supervisor_state.json"
SERVICE_NAME = os.environ.get("DIGITAL_INTERN_SERVICE", "digital-intern")
DIGITAL_INTERN_LAUNCHAGENT = (
    Path.home() / "Library" / "LaunchAgents" /
    "com.jonathan.trading-intelligence.digital-intern.plist"
)

WORKERS = [
    "gdelt", "rss", "web", "reddit", "ticker",
    "sec_edgar", "sec_edgar_ft", "google_news", "nitter", "substack",
    "finnhub", "alphavantage", "polygon", "newsapi",
    "yahoo_ticker_rss", "wikipedia",
    "scorer", "alert", "heartbeat", "purge",
    "stats", "ml_trainer", "price_alert", "continuous_trainer",
    "portfolio_pl", "sentiment_trends", "export", "web_server",
]
# If all of these are stale at once the dashboard shows the CRITICAL banner.
CORE_WORKERS = ("rss", "web", "reddit", "scorer")

# Workers that are intentionally quiet — only log when they do something.
# Don't flag as stale unless silent for >1h.
_QUIET_WORKERS = {
    "alert", "heartbeat", "purge", "price_alert",
    "portfolio_pl", "sentiment_trends", "ml_trainer", "export",
    # Rate-limited free-tier APIs that only fire every 5–30 minutes.
    "alphavantage", "polygon", "newsapi", "wikipedia",
    "finnhub", "substack",
}
WORKER_TAG_RE = re.compile(r"\[([a-z_]+?)(?:_worker)?\]")

SERVER_STARTED_AT = time.time()
_HTTP_READ_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="dashboard-http")

app = FastAPI(title="Digital Intern Dashboard", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_store: ArticleStore | None = None


def store() -> ArticleStore:
    global _store
    if _store is None:
        _store = ArticleStore()
    return _store


# ── helpers ────────────────────────────────────────────────────────────────
def _tail_jsonl(path: Path, n: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        size = path.stat().st_size
        chunk = min(size, max(128 * 1024, n * 768))
        with path.open("rb") as f:
            f.seek(max(0, size - chunk))
            data = f.read()
    except Exception:
        return []
    lines = data.splitlines()
    if len(lines) > n:
        lines = lines[-n:]
    out: list[dict[str, Any]] = []
    for raw in lines:
        try:
            out.append(json.loads(raw.decode("utf-8", errors="replace")))
        except Exception:
            continue
    return out


def _parse_ts(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _service_info(name: str = SERVICE_NAME) -> dict[str, Any]:
    try:
        active = subprocess.run(
            ["systemctl", "is-active", name],
            capture_output=True, text=True, timeout=3,
        ).stdout.strip()
    except Exception:
        active = "unknown"
    props: dict[str, str] = {}
    try:
        out = subprocess.run(
            ["systemctl", "show", name,
             "--property=ActiveEnterTimestampMonotonic,MainPID,SubState"],
            capture_output=True, text=True, timeout=3,
        ).stdout
        for ln in out.strip().splitlines():
            if "=" in ln:
                k, v = ln.split("=", 1)
                props[k] = v
    except Exception:
        pass
    uptime_s: float | None = None
    try:
        mono_us = int(props.get("ActiveEnterTimestampMonotonic", "0") or "0")
        if mono_us > 0:
            with open("/proc/uptime") as f:
                now_mono_s = float(f.read().split()[0])
            uptime_s = max(0.0, now_mono_s - mono_us / 1_000_000)
    except Exception:
        uptime_s = None
    return {
        "name": name,
        "active": active,
        "sub_state": props.get("SubState", ""),
        "main_pid": props.get("MainPID", ""),
        "uptime_s": uptime_s,
    }


def _ml_status() -> dict[str, Any]:
    candidates = [
        ROOT / "ml" / "models" / "article_net.pt",
        ROOT / "data" / "ml" / "article_net.pt",
        ROOT / "data" / "ml" / "relevance_model.pkl",
    ]
    info: dict[str, Any] = {"trained": False}
    for p in candidates:
        if p.exists():
            try:
                st = p.stat()
                info["trained"] = True
                info["model_path"] = str(p.relative_to(ROOT))
                info["size_kb"] = round(st.st_size / 1024, 1)
                info["mtime"] = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
                info["age_s"] = time.time() - st.st_mtime
                break
            except Exception:
                pass
    metrics = _tail_jsonl(METRICS_LOG, 200)
    for m in reversed(metrics):
        if m.get("metric") == "scorer.nn_bypass_rate":
            info["nn_bypass_rate"] = m.get("value")
            info["bypass_total"] = m.get("total")
            info["bypass_to_llm"] = m.get("to_llm")
            break
    for m in reversed(metrics):
        if str(m.get("metric", "")).startswith("ml.train"):
            info["last_train"] = m
            break
    return info


def _read_supervisor_state() -> dict[str, Any]:
    """Atomic JSON snapshot written every 5 min by daemon supervisor.

    Returns {} when the file is missing or unparseable so callers can fall
    back to log scraping."""
    try:
        if not SUPERVISOR_STATE.exists():
            return {}
        with SUPERVISOR_STATE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _configured_worker_allowlist() -> set[str] | None:
    raw = os.environ.get("DIGITAL_INTERN_WORKERS", "")
    if not raw and DIGITAL_INTERN_LAUNCHAGENT.exists():
        try:
            with DIGITAL_INTERN_LAUNCHAGENT.open("rb") as f:
                plist = plistlib.load(f)
            env = plist.get("EnvironmentVariables") or {}
            raw = str(env.get("DIGITAL_INTERN_WORKERS") or "")
        except Exception:
            raw = ""
    allowlist = {name.strip() for name in raw.split(",") if name.strip()}
    return allowlist or None


def _worker_health(lookback: int = 2000) -> list[dict[str, Any]]:
    seen: dict[str, float] = {}
    for entry in _tail_jsonl(STRUCTURED_LOG, lookback):
        t = _parse_ts(entry.get("ts"))
        if t is None:
            continue
        msg = entry.get("msg", "")
        for m in WORKER_TAG_RE.finditer(msg):
            tag = m.group(1)
            if tag in WORKERS and t > seen.get(tag, 0):
                seen[tag] = t
    # Daemon snapshot is the authoritative source for state / crash counts.
    sup = _read_supervisor_state()
    sup_by_name = {w["name"]: w for w in sup.get("workers", [])}
    allowed_workers = _configured_worker_allowlist()
    now = time.time()
    out: list[dict[str, Any]] = []
    for w in WORKERS:
        ts = seen.get(w)
        age = (now - ts) if ts else None
        sup_entry = sup_by_name.get(w, {})
        state = sup_entry.get("state", "ok")
        crashes_5m = int(sup_entry.get("crashes_5m", 0) or 0)
        total_crashes = int(sup_entry.get("total_crashes", 0) or 0)
        last_exception = sup_entry.get("last_exception") or ""

        if allowed_workers is not None and w not in allowed_workers:
            state = "configured_off"
            status = "disabled"
        elif state == "disabled":
            status = "stale"
        else:
            stale_threshold = 3600 if w in _QUIET_WORKERS else 600
            if age is None:
                status = "unknown"
            elif age < 120:
                status = "ok"
            elif age < stale_threshold:
                status = "warn"
            else:
                status = "stale"
            # Degraded state from supervisor always escalates a green dot to warn
            if state == "degraded" and status == "ok":
                status = "warn"

        out.append({
            "name": w,
            "last_seen_s": age,
            "last_seen_iso": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else None,
            "status": status,
            "state": state,
            "crashes_5m": crashes_5m,
            "total_crashes": total_crashes,
            "last_exception": last_exception,
        })
    return out


def _core_workers_status(workers: list[dict[str, Any]]) -> dict[str, Any]:
    """Summary of CORE_WORKERS; ``critical`` is True iff all are stale."""
    statuses = {w["name"]: w["status"] for w in workers if w["name"] in CORE_WORKERS}
    down = [name for name, s in statuses.items() if s in ("stale", "unknown")]
    disabled = [name for name, s in statuses.items() if s == "disabled"]
    return {
        "core": list(CORE_WORKERS),
        "down": down,
        "disabled": disabled,
        "allowlist": sorted(_configured_worker_allowlist() or []),
        "critical": len(down) >= len(CORE_WORKERS),
    }


def _recent_errors(n: int = 5, scan: int = 1000) -> list[dict[str, Any]]:
    lines = _tail_jsonl(STRUCTURED_LOG, scan)
    errs = [
        {"ts": ln.get("ts"), "level": ln.get("level"),
         "logger": ln.get("logger"), "msg": ln.get("msg")}
        for ln in lines if ln.get("level") in ("ERROR", "CRITICAL")
    ]
    return errs[-n:][::-1]


def _last_heartbeat_and_alert() -> dict[str, Any]:
    lines = _tail_jsonl(STRUCTURED_LOG, 5000)
    out: dict[str, Any] = {"heartbeat": None, "alert": None}
    for ln in reversed(lines):
        msg = ln.get("msg", "")
        lower = msg.lower()
        if out["heartbeat"] is None and "[heartbeat_worker]" in lower:
            out["heartbeat"] = {"ts": ln.get("ts"), "msg": msg}
        if out["alert"] is None and "[alert_worker]" in lower and (
            "sent" in lower or "fired" in lower or "urgent" in lower or "alert" in lower
        ):
            out["alert"] = {"ts": ln.get("ts"), "msg": msg}
        if out["heartbeat"] and out["alert"]:
            break
    return out


def _articles_per_hour_24h() -> list[dict[str, Any]]:
    try:
        rows = store().conn.execute(
            "SELECT first_seen FROM articles "
            f"WHERE first_seen >= datetime('now','-24 hours') AND {_LIVE_ONLY_CLAUSE}"
        ).fetchall()
    except Exception:
        return []
    now = datetime.now(timezone.utc)
    buckets = [0] * 24
    for (ts,) in rows:
        t = _parse_ts(ts)
        if t is None:
            continue
        delta_h = (now.timestamp() - t) / 3600
        if 0 <= delta_h < 24:
            buckets[23 - int(delta_h)] += 1
    base = now.timestamp() - 24 * 3600
    return [
        {"hour_offset": i - 23, "count": c,
         "ts": datetime.fromtimestamp(base + i * 3600, tz=timezone.utc).isoformat()}
        for i, c in enumerate(buckets)
    ]


_STATS_SINCE_TTL_SECS = 60
_STATS_SINCE_LOCK = threading.Lock()
_STATS_SINCE_CACHE: dict = {"ts": 0.0, "total": 0, "urgent": 0, "refreshing": False}


def _refresh_stats_since() -> None:
    try:
        result = store().stats_since(1)
        with _STATS_SINCE_LOCK:
            _STATS_SINCE_CACHE.update(ts=time.time(), total=result["total"], urgent=result["urgent"])
    except Exception:
        pass
    finally:
        with _STATS_SINCE_LOCK:
            _STATS_SINCE_CACHE["refreshing"] = False


def _stats_payload() -> dict[str, Any]:
    s = store().stats()
    s["service"] = _service_info()
    s["ml"] = _ml_status()
    s["server_uptime_s"] = int(time.time() - SERVER_STARTED_AT)
    with _STATS_SINCE_LOCK:
        cache_age = time.time() - _STATS_SINCE_CACHE["ts"]
        s["articles_last_hour"] = _STATS_SINCE_CACHE["total"]
        s["urgent_last_hour"] = _STATS_SINCE_CACHE["urgent"]
        need_refresh = (cache_age > _STATS_SINCE_TTL_SECS
                        and not _STATS_SINCE_CACHE["refreshing"])
        if need_refresh:
            _STATS_SINCE_CACHE["refreshing"] = True
    if need_refresh:
        threading.Thread(
            target=_refresh_stats_since, name="stats-since-refresh", daemon=True,
        ).start()
    return s


def _articles_payload(limit: int = 50, min_score: float = 0.0) -> list[dict[str, Any]]:
    try:
        rows = store().conn.execute(
            "SELECT id, url, title, source, published, kw_score, ai_score, urgency, first_seen "
            f"FROM articles WHERE MAX(ai_score, kw_score) >= ? AND {_LIVE_ONLY_CLAUSE} "
            "ORDER BY ai_score DESC, kw_score DESC, first_seen DESC LIMIT ?",
            (min_score, max(1, min(500, limit))),
        ).fetchall()
    except Exception:
        # SQLite older syntax fallback (MAX of two columns)
        rows = store().conn.execute(
            "SELECT id, url, title, source, published, kw_score, ai_score, urgency, first_seen "
            "FROM articles "
            "WHERE (CASE WHEN ai_score>kw_score THEN ai_score ELSE kw_score END) >= ? "
            f"AND {_LIVE_ONLY_CLAUSE} "
            "ORDER BY ai_score DESC, kw_score DESC, first_seen DESC LIMIT ?",
            (min_score, max(1, min(500, limit))),
        ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        ai = float(r[6] or 0)
        kw = float(r[5] or 0)
        out.append({
            "id": r[0], "url": r[1], "title": r[2], "source": r[3],
            "published": r[4], "kw_score": kw, "ai_score": ai,
            "score": ai if ai > 0 else kw,
            "urgency": int(r[7] or 0),
            "first_seen": r[8],
        })
    return out


def _read_http_json(port: int, path: str, timeout: float = 4.0) -> dict[str, Any]:
    if not path.startswith("/"):
        path = "/" + path
    url = f"http://127.0.0.1:{port}{path}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
            return {"ok": True, "status": resp.status, "data": payload}
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            payload = json.loads(raw.decode("utf-8", errors="replace"))
        except Exception:
            payload = {"error": raw.decode("utf-8", errors="replace")[:400]}
        return {"ok": False, "status": e.code, "data": payload, "error": payload.get("error")}
    except Exception as e:
        return {"ok": False, "status": None, "data": None, "error": str(e)}


def _read_many_http_json(
    specs: dict[str, tuple[int, str, float]],
    *,
    global_timeout: float = 4.5,
) -> dict[str, dict[str, Any]]:
    futures = {
        _HTTP_READ_POOL.submit(_read_http_json, port, path, timeout): (key, timeout)
        for key, (port, path, timeout) in specs.items()
    }
    results: dict[str, dict[str, Any]] = {}
    try:
        for future in as_completed(futures, timeout=global_timeout):
            key, _timeout = futures[future]
            try:
                results[key] = future.result()
            except Exception as e:
                results[key] = {"ok": False, "status": None, "data": None, "error": str(e)}
    except TimeoutError:
        pass
    for future, (key, timeout) in futures.items():
        if key not in results:
            results[key] = {
                "ok": False,
                "status": None,
                "data": None,
                "error": f"timed out after {timeout:.1f}s",
            }
    return results


def _row_value(row: dict[str, Any], key: str) -> Any:
    value = row
    for part in key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _short_value(value: Any) -> str:
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, str):
        return value if len(value) <= 220 else value[:217] + "..."
    if isinstance(value, (list, tuple, set)):
        return f"{len(value)} items"
    if isinstance(value, dict):
        return f"{len(value)} fields"
    return str(value)


def _first_text(data: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = _row_value(data, key)
        if value not in (None, "", [], {}):
            return _short_value(value)
    return None


def _metric_list(data: dict[str, Any], keys: tuple[str, ...]) -> list[dict[str, str]]:
    metrics: list[dict[str, str]] = []
    for key in keys:
        value = _row_value(data, key)
        if value not in (None, "", [], {}):
            metrics.append({"label": key.replace("_", " ").replace(".", " / "), "value": _short_value(value)})
    return metrics[:8]


def _rows_from_payload(data: Any, preferred_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        return data[:12] if isinstance(data, list) else []
    for key in preferred_keys:
        value = _row_value(data, key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)][:12]
        if isinstance(value, dict):
            return [{"name": k, **v} if isinstance(v, dict) else {"name": k, "value": v} for k, v in list(value.items())[:12]]
    for key, value in data.items():
        if isinstance(value, list) and value and all(isinstance(x, dict) for x in value[:3]):
            return value[:12]
    return []


def _columns_for_rows(rows: list[dict[str, Any]]) -> list[str]:
    preferred = (
        "name", "source", "ticker", "symbol", "title", "headline", "action", "category",
        "state", "status", "verdict", "reason", "reasoning", "score", "urgency",
        "published", "timestamp", "ts", "first_seen", "n", "count", "pct", "pl_usd",
        "total_pl_usd", "return_pct", "score", "timestamp", "ts",
    )
    hidden = {"id", "url", "raw", "html", "prompt", "body", "content"}
    keys: list[str] = []
    seen: set[str] = set()
    for key in preferred:
        if key not in seen and any(key in row for row in rows):
            keys.append(key)
            seen.add(key)
        if len(keys) >= 6:
            return keys
    for row in rows:
        for key in row.keys():
            if key not in seen and key not in hidden and not isinstance(row.get(key), (dict, list)):
                keys.append(key)
                seen.add(key)
            if len(keys) >= 6:
                return keys
    return keys[:6]


def _section_card(
    title: str,
    response: dict[str, Any],
    *,
    headline_keys: tuple[str, ...] = ("headline", "summary", "verdict_reason", "hint", "note", "error"),
    metric_keys: tuple[str, ...] = (),
    row_keys: tuple[str, ...] = ("rows", "items", "recent", "tape", "articles", "leaderboard", "models", "opportunities"),
) -> dict[str, Any]:
    data = response.get("data")
    if not response.get("ok"):
        return {
            "title": title,
            "status": "down",
            "state": "UNAVAILABLE",
            "headline": response.get("error") or f"HTTP {response.get('status')}",
            "metrics": [],
            "columns": [],
            "rows": [],
        }
    if not isinstance(data, dict):
        rows = data[:12] if isinstance(data, list) else []
        return {
            "title": title,
            "status": "ok",
            "state": "OK",
            "headline": f"{len(rows)} rows" if isinstance(rows, list) else "loaded",
            "metrics": [],
            "columns": _columns_for_rows(rows) if isinstance(rows, list) else [],
            "rows": rows if isinstance(rows, list) else [],
        }

    state = _first_text(data, ("state", "verdict", "status")) or "OK"
    state_upper = state.upper()
    status = "ok"
    if any(token in state_upper for token in ("ERROR", "CRITICAL", "DOWN", "FAIL")):
        status = "down"
    elif any(token in state_upper for token in ("NO_DATA", "INSUFFICIENT", "WARN", "STALE", "BLIND")):
        status = "warn"
    rows = _rows_from_payload(data, row_keys)
    headline = _first_text(data, headline_keys)
    if not headline:
        headline = f"{len(rows)} rows loaded" if rows else f"{state} data loaded"
    metrics = _metric_list(data, metric_keys)
    if not metrics:
        metrics = _metric_list(data, (
            "n", "count", "total", "n_runs", "completed_runs", "n_models", "n_personas",
            "n_decisions", "n_trades", "n_round_trips", "n_articles", "n_raw",
            "n_after_dedup", "cash_pct", "total_value", "window_minutes",
        ))
    return {
        "title": title,
        "status": status,
        "state": state,
        "headline": headline,
        "metrics": metrics,
        "columns": _columns_for_rows(rows),
        "rows": rows,
    }


_SECTION_CONFIG: dict[str, dict[str, Any]] = {
    "compare": {
        "title": "Compare",
        "description": "Backtest and model comparison without starting new runs.",
        "cards": (
            ("Backtest Summary", 8090, "/api/backtests/stats", ("completed_runs", "total_runs", "beat_spy_count", "beat_spy_rate"), ("runs",)),
            ("Backtest Runs", 8090, "/api/backtests", ("total", "completed_runs"), ("runs", "items", "rows")),
            ("Model Rankings", 8090, "/api/model-rankings", ("n_models",), ("models",)),
            ("Persona Leaderboard", 8090, "/api/persona-leaderboard", ("n_runs", "n_personas"), ("leaderboard", "drag_personas")),
        ),
    },
    "strategy": {
        "title": "Strategy Lab",
        "description": "Read-only strategy, opportunity, and deployment diagnostics.",
        "cards": (
            ("Game Plan", 8090, "/api/game-plan", ("n_actions", "n_open", "market_open", "next_open_seconds"), ("position_actions", "portfolio_directives", "opportunities")),
            ("Actionable Opportunities", 8090, "/api/actionable-opportunities", ("n", "n_opportunities", "cash_pct"), ("opportunities", "rows", "items")),
            ("Funded Suggestions", 8090, "/api/funded-suggestions", ("n", "cash_usd", "buying_power_usd"), ("suggestions", "rows", "items")),
            ("Deployment Plan", 8090, "/api/deployment-plan", ("n_actions", "cash_usd", "deployable_cash_usd"), ("actions", "rows", "items")),
            ("Decision Context", 8090, "/api/decision-context", ("input_summary.signal_count", "input_summary.n_urgent", "market_open", "claude_invoked"), ("advisory", "rows", "items")),
        ),
    },
    "journal": {
        "title": "Journal",
        "description": "Today action tape, decision health, and realized record.",
        "cards": (
            ("Today Action Tape", 8090, "/api/today-action-tape", ("n_decisions", "n_trades", "n_no_decisions", "net_cash_flow_usd"), ("tape",)),
            ("Decision Health", 8090, "/api/decision-health", ("n_decisions", "n_decisions_24h", "no_decision_rate_24h"), ("recent", "action_mix")),
            ("Session Delta", 8090, "/api/session-delta?minutes=1440", ("n_fills", "equity_delta_usd", "equity_delta_pct"), ("fills", "rows", "items")),
            ("Closed Positions", 8090, "/api/closed-positions", ("n_closed", "total_realized_pl"), ("positions", "rows", "items")),
            ("Track Record", 8090, "/api/track-record", ("n_round_trips",), ("names", "rows", "items")),
        ),
    },
    "personas": {
        "title": "Personas",
        "description": "Persona skill, model rankings, and book-fit diagnostics.",
        "cards": (
            ("Persona Leaderboard", 8090, "/api/persona-leaderboard", ("n_runs", "n_personas"), ("leaderboard", "drag_personas")),
            ("Persona Book Fit", 8090, "/api/persona-book-fit", ("n_positions", "n_personas"), ("matches", "rows", "items")),
            ("Model Rankings", 8090, "/api/model-rankings", ("n_models",), ("models",)),
            ("Baseline Compare", 8090, "/api/baseline-compare", ("n", "n_records", "oos_rmse_ratio"), ("rows", "items")),
            ("Scorer Confidence", 8090, "/api/scorer-confidence", ("n", "n_predictions"), ("rows", "items", "by_bucket")),
        ),
    },
    "tape": {
        "title": "Tape",
        "description": "Market tape, sector pulse, and realized tape-fit.",
        "cards": (
            ("Tape Fit P/L", 8090, "/api/tape-fit-pl", ("n_round_trips", "n_directional", "min_for_verdict"), ("buckets",)),
            ("Sector Pulse", 8090, "/api/sector-pulse", ("n_tickers", "n_hot", "window_hours"), ("tickers", "sectors", "rows")),
            ("News Velocity", 8090, "/api/news-velocity", ("n_held", "n_with_data", "window_hours"), ("per_ticker",)),
            ("News Source Mix", 8090, "/api/news-source-mix", ("n_held", "n_with_data", "any_echo"), ("per_ticker",)),
            ("Macro Calendar", 8090, "/api/macro-calendar", ("n_events",), ("events", "rows", "items")),
        ),
    },
    "pulse": {
        "title": "News Pulse",
        "description": "ArticleNet feed and trader news-quality diagnostics.",
        "cards": (
            ("ArticleNet Stats", 8080, "/api/stats", ("total", "urgent", "unscored", "db_mb"), ("rows", "items")),
            ("Recent Articles", 8080, "/api/articles?limit=20", ("total", "urgent"), ("articles", "rows", "items")),
            ("News Deduped", 8090, "/api/news-deduped?hours=6&min_score=4", ("n_raw", "n_after_dedup", "compression_ratio"), ("articles",)),
            ("News Edge", 8090, "/api/news-edge", ("n", "n_articles"), ("rows", "items", "tickers")),
            ("Source Edge", 8090, "/api/source-edge", ("n_sources", "n"), ("sources", "rows", "items")),
            ("Sector Heatmap", 8090, "/api/sector-heatmap", ("n_tickers",), ("sectors", "tickers", "rows")),
        ),
    },
}


def _section_payload(section: str) -> dict[str, Any]:
    cfg = _SECTION_CONFIG.get(section)
    if not cfg:
        return {
            "ok": False,
            "section": section,
            "title": "Unknown Section",
            "description": "No section configuration.",
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "cards": [],
        }
    cards = []
    for title, port, path, metric_keys, row_keys in cfg["cards"]:
        response = _read_http_json(port, path, timeout=6.0)
        cards.append(_section_card(
            title,
            response,
            metric_keys=tuple(metric_keys),
            row_keys=tuple(row_keys),
        ))
    return {
        "ok": True,
        "section": section,
        "title": cfg["title"],
        "description": cfg["description"],
        "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cards": cards,
    }


def _command_center_payload() -> dict[str, Any]:
    upstream = _read_many_http_json(
        {
            "intern_stats": (8080, "/api/stats", 2.0),
            "intern_health": (8080, "/api/health", 2.0),
            "trader_health": (8090, "/api/healthz", 2.0),
            "trader_portfolio": (8090, "/api/portfolio", 2.0),
            "trader_desk": (8090, "/api/desk-pulse", 2.5),
            "trader_game": (8090, "/api/game-plan", 3.0),
            "trader_session": (8090, "/api/session-delta?minutes=360", 2.5),
            "trader_build": (8090, "/api/build-info", 2.0),
        },
        global_timeout=4.0,
    )
    intern_stats = upstream["intern_stats"]
    intern_health = upstream["intern_health"]
    trader_health = upstream["trader_health"]
    trader_portfolio = upstream["trader_portfolio"]
    trader_desk = upstream["trader_desk"]
    trader_game = upstream["trader_game"]
    trader_session = upstream["trader_session"]
    trader_build = upstream["trader_build"]

    workers = _worker_health()
    worker_counts = {
        "ok": sum(1 for w in workers if w.get("status") == "ok"),
        "warn": sum(1 for w in workers if w.get("status") == "warn"),
        "stale": sum(1 for w in workers if w.get("status") == "stale"),
        "unknown": sum(1 for w in workers if w.get("status") == "unknown"),
        "disabled": sum(1 for w in workers if w.get("status") == "disabled"),
    }
    core_status = _core_workers_status(workers)
    errors = _recent_errors()

    stats = intern_stats.get("data") if isinstance(intern_stats.get("data"), dict) else {}
    desk = trader_desk.get("data") if isinstance(trader_desk.get("data"), dict) else {}
    game = trader_game.get("data") if isinstance(trader_game.get("data"), dict) else {}
    build = trader_build.get("data") if isinstance(trader_build.get("data"), dict) else {}

    services = [
        {
            "id": "unified-dashboard",
            "name": "Unified Dashboard",
            "status": "ok",
            "detail": "127.0.0.1:8765 with tailnet proxy",
            "href": "/",
        },
        {
            "id": "digital-intern",
            "name": "Digital Intern",
            "status": "ok" if intern_health.get("ok") else "down",
            "detail": f"{stats.get('total', 0)} articles, {stats.get('urgent', 0)} urgent",
            "href": "/intern/",
        },
        {
            "id": "paper-trader",
            "name": "Paper Trader",
            "status": "ok" if trader_health.get("ok") else "down",
            "detail": (desk.get("headline") or trader_health.get("error") or "desk pulse pending")[:160],
            "href": "/trader/",
        },
        {
            "id": "ops-view",
            "name": "Ops View",
            "status": "down" if core_status.get("critical") else "ok",
            "detail": (
                "core workers critical" if core_status.get("critical")
                else (
                    "collector workers intentionally off"
                    if core_status.get("disabled") else "core workers ready"
                )
            ),
            "href": "/ops/",
        },
    ]

    action_queue: list[dict[str, str]] = []
    if not intern_health.get("ok"):
        action_queue.append({
            "severity": "fail",
            "title": "Digital Intern API is unreachable",
            "detail": str(intern_health.get("error") or intern_health.get("status") or "unknown"),
        })
    if not trader_health.get("ok"):
        action_queue.append({
            "severity": "fail",
            "title": "Paper Trader API is unreachable",
            "detail": str(trader_health.get("error") or trader_health.get("status") or "unknown"),
        })
    if core_status.get("critical"):
        action_queue.append({
            "severity": "fail",
            "title": "Core ArticleNet workers are stale",
            "detail": ", ".join(core_status.get("down") or []),
        })
    if build.get("stale") is True:
        action_queue.append({
            "severity": "warn",
            "title": "Paper Trader process is running stale code",
            "detail": f"boot {build.get('boot_sha')} vs head {build.get('head_sha')}",
        })
    if desk.get("state") not in (None, "HEALTHY", "NO_DATA"):
        action_queue.append({
            "severity": "warn",
            "title": f"Desk pulse state: {desk.get('state')}",
            "detail": desk.get("headline") or "",
        })
    if game.get("state") == "ACTIONS_PRESENT":
        action_queue.append({
            "severity": "warn",
            "title": "Game plan has live actions",
            "detail": game.get("headline") or "",
        })
    if not action_queue:
        action_queue.append({
            "severity": "ok",
            "title": "No immediate operator action",
            "detail": "Command Center APIs are responding.",
        })

    return {
        "ok": True,
        "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "services": services,
        "action_queue": action_queue,
        "article_net": {
            "ok": bool(intern_stats.get("ok") and intern_health.get("ok")),
            "stats": stats,
            "health": intern_health.get("data"),
        },
        "trader": {
            "ok": bool(trader_health.get("ok")),
            "health": trader_health.get("data"),
            "portfolio": trader_portfolio.get("data") if isinstance(trader_portfolio.get("data"), dict) else {},
            "desk_pulse": desk,
            "game_plan": game,
            "session_delta": trader_session.get("data") if isinstance(trader_session.get("data"), dict) else {},
            "build_info": build,
        },
        "ops": {
            "core_status": core_status,
            "workers": worker_counts,
            "errors": len(errors),
        },
    }


# ── routes ─────────────────────────────────────────────────────────────────
@app.get("/")
async def command_center():
    if not COMMAND_CENTER_HTML.exists():
        return JSONResponse({"error": "command_center.html missing"}, status_code=500)
    return HTMLResponse(COMMAND_CENTER_HTML.read_text(encoding="utf-8"))


async def ops_index():
    if not DASHBOARD_HTML.exists():
        return JSONResponse({"error": "dashboard.html missing"}, status_code=500)
    return HTMLResponse(DASHBOARD_HTML.read_text(encoding="utf-8"))


async def section_index():
    if not SECTION_PAGE_HTML.exists():
        return JSONResponse({"error": "section_page.html missing"}, status_code=500)
    return HTMLResponse(SECTION_PAGE_HTML.read_text(encoding="utf-8"))


def _fetch_upstream(
    url: str,
    method: str,
    body: bytes,
    headers: dict[str, str],
    timeout: float,
) -> tuple[bytes, int, str]:
    req = urllib.request.Request(
        url,
        data=body if body else None,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as upstream:
            return (
                upstream.read(),
                upstream.status,
                upstream.headers.get("Content-Type", "application/octet-stream"),
            )
    except urllib.error.HTTPError as e:
        return (
            e.read(),
            e.code,
            e.headers.get("Content-Type", "text/plain"),
        )


async def _proxy_http(
    request: Request,
    port: int,
    upstream_path: str,
    *,
    forwarded_prefix: str | None = None,
    html_rewrites: tuple[tuple[str, str], ...] = (),
    timeout: float = 20.0,
) -> Response:
    if not upstream_path.startswith("/"):
        upstream_path = "/" + upstream_path
    query = request.url.query
    url = f"http://127.0.0.1:{port}{upstream_path}" + (f"?{query}" if query else "")
    body = await request.body()
    headers: dict[str, str] = {}
    for key in ("Content-Type", "Accept", "User-Agent"):
        value = request.headers.get(key)
        if value:
            headers[key] = value
    if forwarded_prefix:
        headers["X-Forwarded-Prefix"] = forwarded_prefix
    try:
        payload, status, content_type = await asyncio.to_thread(
            _fetch_upstream,
            url,
            request.method,
            body,
            headers,
            timeout,
        )
    except Exception as e:
        return JSONResponse(
            {"ok": False, "error": f"proxy to {port}{upstream_path} failed: {e}"},
            status_code=502,
        )
    if html_rewrites and "text/html" in content_type.lower():
        text = payload.decode("utf-8", errors="replace")
        for old, new in html_rewrites:
            text = text.replace(old, new)
        payload = text.encode("utf-8")
    return Response(content=payload, status_code=status, headers={"Content-Type": content_type})


_INTERN_HTML_REWRITES = (
    ('"/api/', '"/intern/api/'),
    ("'/api/", "'/intern/api/"),
    ("`/api/", "`/intern/api/"),
)


_TRADER_HTML_REWRITES = (
    (
        'history.replaceState(null, "", name === "trader" ? "/" : "/backtests");',
        'history.replaceState(null, "", name === "trader" ? "/trader/" : "/trader/backtests");',
    ),
)


@app.get("/intern")
@app.get("/intern/")
async def intern_dashboard(request: Request):
    return await _proxy_http(request, 8080, "/", html_rewrites=_INTERN_HTML_REWRITES)


@app.get("/intern/chat")
async def intern_chat(request: Request):
    return await _proxy_http(request, 8080, "/chat", html_rewrites=_INTERN_HTML_REWRITES)


@app.api_route("/intern/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def intern_api_proxy(path: str, request: Request):
    return await _proxy_http(request, 8080, f"/api/{path}")


@app.get("/trader")
@app.get("/trader/")
async def trader_dashboard(request: Request):
    return await _proxy_http(
        request,
        8090,
        "/",
        forwarded_prefix="/trader",
        html_rewrites=_TRADER_HTML_REWRITES,
    )


@app.get("/trader/backtests")
async def trader_backtests(request: Request):
    return await _proxy_http(
        request,
        8090,
        "/backtests",
        forwarded_prefix="/trader",
        html_rewrites=_TRADER_HTML_REWRITES,
    )


@app.get("/trader/ticker/{sym}")
async def trader_ticker(sym: str, request: Request):
    return await _proxy_http(
        request,
        8090,
        f"/ticker/{sym}",
        forwarded_prefix="/trader",
        html_rewrites=_TRADER_HTML_REWRITES,
    )


@app.get("/trader/monkey-benchmark")
async def trader_monkey_benchmark(request: Request):
    return await _proxy_http(
        request,
        8090,
        "/monkey-benchmark",
        forwarded_prefix="/trader",
        html_rewrites=_TRADER_HTML_REWRITES,
    )


@app.api_route("/trader/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def trader_api_proxy(path: str, request: Request):
    return await _proxy_http(request, 8090, f"/api/{path}", forwarded_prefix="/trader")


@app.get("/backtests")
@app.get("/backtests/compare")
async def backtests_compare(request: Request):
    return await section_index()


@app.get("/strategy-lab")
async def strategy_lab(request: Request):
    return await section_index()


@app.get("/journal")
async def journal(request: Request):
    return await section_index()


@app.get("/personas")
async def personas(request: Request):
    return await section_index()


@app.get("/tape")
async def tape(request: Request):
    return await section_index()


@app.get("/pulse")
async def pulse(request: Request):
    return await section_index()


@app.get("/system")
@app.get("/system/")
@app.get("/ops")
@app.get("/ops/")
async def ops_dashboard():
    return await ops_index()


@app.get("/api/command-center")
@app.get("/ops/api/command-center")
async def api_command_center():
    return JSONResponse(await asyncio.to_thread(_command_center_payload))


@app.get("/api/action-queue")
@app.get("/ops/api/action-queue")
async def api_action_queue():
    payload = await asyncio.to_thread(_command_center_payload)
    return JSONResponse({
        "as_of": payload["as_of"],
        "items": payload["action_queue"],
    })


@app.get("/api/sections/{section}")
@app.get("/ops/api/sections/{section}")
async def api_section(section: str):
    aliases = {
        "backtests": "compare",
        "backtests-compare": "compare",
        "strategy-lab": "strategy",
        "news-pulse": "pulse",
    }
    key = aliases.get(section, section)
    payload = await asyncio.to_thread(_section_payload, key)
    return JSONResponse(payload, status_code=200 if payload.get("ok") else 404)


@app.get("/api/stats")
@app.get("/ops/api/stats")
async def api_stats():
    return JSONResponse(await asyncio.to_thread(_stats_payload))


@app.get("/api/articles")
@app.get("/ops/api/articles")
async def api_articles(limit: int = 50, min_score: float = 0.0):
    return JSONResponse(await asyncio.to_thread(_articles_payload, limit, min_score))


@app.get("/api/metrics")
@app.get("/ops/api/metrics")
async def api_metrics():
    return JSONResponse(await asyncio.to_thread(_tail_jsonl, METRICS_LOG, 200))


@app.get("/api/logs")
@app.get("/ops/api/logs")
async def api_logs(n: int = 50):
    return JSONResponse(await asyncio.to_thread(_tail_jsonl, STRUCTURED_LOG, max(1, min(2000, n))))


@app.get("/api/health")
@app.get("/ops/api/health")
async def api_health():
    workers = await asyncio.to_thread(_worker_health)
    return JSONResponse({
        "service": _service_info(),
        "workers": workers,
        "core_status": _core_workers_status(workers),
        "errors": await asyncio.to_thread(_recent_errors),
        "last": await asyncio.to_thread(_last_heartbeat_and_alert),
    })


@app.get("/api/articles_per_hour")
@app.get("/ops/api/articles_per_hour")
async def api_articles_per_hour():
    return JSONResponse(await asyncio.to_thread(_articles_per_hour_24h))


@app.post("/api/restart")
@app.post("/ops/api/restart")
async def api_restart():
    try:
        r = subprocess.run(
            ["sudo", "-n", "systemctl", "restart", "digital-intern"],
            capture_output=True, text=True, timeout=10,
        )
        ok = r.returncode == 0
        return JSONResponse(
            {"ok": ok, "stdout": r.stdout, "stderr": r.stderr},
            status_code=200 if ok else 500,
        )
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# ── websocket: push every 5s ─────────────────────────────────────────────
@app.websocket("/ops/ws")
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    last_top_id: str | None = None
    last_urgent = -1
    try:
        # initial burst
        stats = _stats_payload()
        await ws.send_json({"type": "stats", "data": stats})
        _workers = _worker_health()
        await ws.send_json({"type": "health", "data": {
            "service": _service_info(),
            "workers": _workers,
            "core_status": _core_workers_status(_workers),
            "errors": _recent_errors(),
            "last": _last_heartbeat_and_alert(),
        }})
        articles = _articles_payload(50, 0.0)
        await ws.send_json({"type": "articles", "data": articles})
        await ws.send_json({"type": "chart", "data": _articles_per_hour_24h()})
        if articles:
            last_top_id = articles[0]["id"]
        last_urgent = int(stats.get("urgent", 0) or 0)

        while True:
            await asyncio.sleep(5)
            stats = _stats_payload()
            await ws.send_json({"type": "stats", "data": stats})
            await ws.send_json({"type": "health", "data": {
                "workers": _worker_health(),
                "errors": _recent_errors(),
                "last": _last_heartbeat_and_alert(),
            }})
            articles = _articles_payload(50, 0.0)
            await ws.send_json({"type": "articles", "data": articles})
            await ws.send_json({"type": "chart", "data": _articles_per_hour_24h()})

            urgent = int(stats.get("urgent", 0) or 0)
            if articles:
                new_top = articles[0]["id"]
                if new_top != last_top_id and articles[0].get("urgency", 0) >= 1:
                    await ws.send_json({"type": "alert", "data": articles[0]})
                last_top_id = new_top
            if last_urgent >= 0 and urgent > last_urgent:
                await ws.send_json({"type": "alert", "data": {
                    "urgent_total": urgent, "delta": urgent - last_urgent,
                }})
            last_urgent = urgent
    except WebSocketDisconnect:
        return
    except Exception:
        try:
            await ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("DASHBOARD_PORT", "8765"))
    uvicorn.run("dashboard.server:app", host="0.0.0.0", port=port, reload=False)

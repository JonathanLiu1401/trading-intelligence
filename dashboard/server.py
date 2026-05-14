"""
Digital Intern — real-time web dashboard (FastAPI).

Reads directly from the article store + log files; exposes JSON endpoints and a
WebSocket that pushes stats, articles, and health every 5 seconds.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from storage.article_store import ArticleStore  # noqa: E402

DASHBOARD_DIR = Path(__file__).resolve().parent
DASHBOARD_HTML = DASHBOARD_DIR / "dashboard.html"
STRUCTURED_LOG = ROOT / "logs" / "structured.jsonl"
METRICS_LOG = ROOT / "logs" / "metrics.jsonl"
# Atomic JSON snapshot written by the daemon supervisor — preferred source
# of truth for worker state (avoids re-parsing logs).
SUPERVISOR_STATE = ROOT / "logs" / "supervisor_state.json"
SERVICE_NAME = os.environ.get("DIGITAL_INTERN_SERVICE", "digital-intern")

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

        stale_threshold = 3600 if w in _QUIET_WORKERS else 600
        if state == "disabled":
            status = "stale"
        elif age is None:
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
    return {
        "core": list(CORE_WORKERS),
        "down": down,
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
            "WHERE first_seen >= datetime('now','-24 hours')"
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


def _stats_payload() -> dict[str, Any]:
    s = store().stats()
    s["service"] = _service_info()
    s["ml"] = _ml_status()
    s["server_uptime_s"] = int(time.time() - SERVER_STARTED_AT)
    try:
        last_hour = store().stats_since(1)
        s["articles_last_hour"] = last_hour["total"]
        s["urgent_last_hour"] = last_hour["urgent"]
    except Exception:
        s["articles_last_hour"] = 0
        s["urgent_last_hour"] = 0
    return s


def _articles_payload(limit: int = 50, min_score: float = 0.0) -> list[dict[str, Any]]:
    try:
        rows = store().conn.execute(
            "SELECT id, url, title, source, published, kw_score, ai_score, urgency, first_seen "
            "FROM articles WHERE MAX(ai_score, kw_score) >= ? "
            "ORDER BY ai_score DESC, kw_score DESC, first_seen DESC LIMIT ?",
            (min_score, max(1, min(500, limit))),
        ).fetchall()
    except Exception:
        # SQLite older syntax fallback (MAX of two columns)
        rows = store().conn.execute(
            "SELECT id, url, title, source, published, kw_score, ai_score, urgency, first_seen "
            "FROM articles "
            "WHERE (CASE WHEN ai_score>kw_score THEN ai_score ELSE kw_score END) >= ? "
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


# ── routes ─────────────────────────────────────────────────────────────────
@app.get("/")
async def index():
    if not DASHBOARD_HTML.exists():
        return JSONResponse({"error": "dashboard.html missing"}, status_code=500)
    return FileResponse(str(DASHBOARD_HTML), media_type="text/html")


@app.get("/api/stats")
async def api_stats():
    return JSONResponse(_stats_payload())


@app.get("/api/articles")
async def api_articles(limit: int = 50, min_score: float = 0.0):
    return JSONResponse(_articles_payload(limit, min_score))


@app.get("/api/metrics")
async def api_metrics():
    return JSONResponse(_tail_jsonl(METRICS_LOG, 200))


@app.get("/api/logs")
async def api_logs(n: int = 50):
    return JSONResponse(_tail_jsonl(STRUCTURED_LOG, max(1, min(2000, n))))


@app.get("/api/health")
async def api_health():
    workers = _worker_health()
    return JSONResponse({
        "service": _service_info(),
        "workers": workers,
        "core_status": _core_workers_status(workers),
        "errors": _recent_errors(),
        "last": _last_heartbeat_and_alert(),
    })


@app.get("/api/articles_per_hour")
async def api_articles_per_hour():
    return JSONResponse(_articles_per_hour_24h())


@app.post("/api/restart")
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

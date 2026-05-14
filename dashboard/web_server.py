"""Flask web dashboard — public, bound to 0.0.0.0:8080.

The existing FastAPI dashboard (dashboard/server.py, port 8765) is the rich
ops view that runs as its own systemd unit. This module is a smaller,
read-only public-facing dashboard wired into the daemon as a worker thread so
external users can see briefings, portfolio P&L, and top signals without
running anything extra.

API key (``WEB_API_KEY``) protects ``/api/*`` only; the HTML dashboard at
``/`` is public.
"""
from __future__ import annotations

import os
import json
import sqlite3
import subprocess
import zlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request

try:
    import anthropic  # type: ignore
    _ANTHROPIC_AVAILABLE = True
except Exception:
    anthropic = None  # type: ignore
    _ANTHROPIC_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent.parent

# Resolved lazily so importing this module doesn't require an instantiated store.
_store = None
_log = None


def _logger():
    global _log
    if _log is None:
        try:
            from core.logger import get_logger
            _log = get_logger("web_server")
        except Exception:
            import logging
            _log = logging.getLogger("web_server")
    return _log


def _store_handle():
    """Return a shared ArticleStore — the daemon passes one in via init_app()."""
    return _store


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _briefings_from_log(limit: int = 10) -> list[dict]:
    """Pull recent heartbeat briefings from the structured JSONL log."""
    log_path = BASE_DIR / "logs" / "structured.jsonl"
    if not log_path.exists():
        return []
    try:
        size = log_path.stat().st_size
        chunk = min(size, 512 * 1024)
        with log_path.open("rb") as f:
            f.seek(max(0, size - chunk))
            data = f.read()
    except Exception:
        return []
    out: list[dict] = []
    for raw in reversed(data.splitlines()):
        try:
            ln = json.loads(raw.decode("utf-8", errors="replace"))
        except Exception:
            continue
        msg = ln.get("msg", "")
        if "[heartbeat]" in msg and ("sent" in msg.lower() or "generating" in msg.lower()):
            out.append({
                "ts": ln.get("ts", ""),
                "msg": msg,
            })
            if len(out) >= limit:
                break
    return out


def _articles_from_db(limit: int = 50, min_score: float = 0.0) -> list[dict]:
    store = _store_handle()
    if store is None:
        return []
    try:
        rows = store.conn.execute(
            "SELECT id, url, title, source, published, kw_score, ai_score, urgency, first_seen "
            "FROM articles "
            "WHERE (CASE WHEN ai_score>kw_score THEN ai_score ELSE kw_score END) >= ? "
            "ORDER BY ai_score DESC, kw_score DESC, first_seen DESC LIMIT ?",
            (min_score, max(1, min(500, int(limit)))),
        ).fetchall()
    except sqlite3.Error:
        return []
    out = []
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


def create_app(store=None) -> Flask:
    """Build the Flask app. ``store`` is the shared ArticleStore from daemon.py."""
    global _store
    if store is not None:
        _store = store

    app = Flask(__name__)

    @app.after_request
    def _cors(resp):
        # Public read-only dashboard — wide-open CORS so the Paper Trader
        # dashboard (different port, same host) can fetch /api/articles for
        # the cross-linked signal feed.
        resp.headers.setdefault("Access-Control-Allow-Origin", "*")
        resp.headers.setdefault("Access-Control-Allow-Methods", "GET, OPTIONS")
        resp.headers.setdefault("Access-Control-Allow-Headers", "Content-Type")
        return resp

    api_key_required = os.environ.get("WEB_API_KEY", "").strip()

    def _check_api_key() -> bool:
        if not api_key_required:
            return True
        return request.args.get("key", "") == api_key_required

    @app.get("/")
    def index() -> Response:
        prefix = request.headers.get("X-Forwarded-Prefix", "").rstrip("/")
        html = _DASHBOARD_HTML.replace("__API_PREFIX__", prefix)
        return Response(html, mimetype="text/html")

    @app.get("/api/articles")
    def api_articles():
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        limit = max(1, min(500, int(request.args.get("limit", 50))))
        min_score = float(request.args.get("min_score", 0.0))
        return jsonify(_articles_from_db(limit, min_score))

    @app.get("/api/portfolio")
    def api_portfolio():
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        snap = _read_json(BASE_DIR / "data" / "portfolio_pl.json")
        if snap is None:
            return jsonify({"error": "no snapshot yet"}), 503
        return jsonify(snap)

    @app.get("/api/stats")
    def api_stats():
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        store = _store_handle()
        if store is None:
            return jsonify({"error": "store unavailable"}), 503
        try:
            s = dict(store.stats())
            s["last_hour"] = store.stats_since(1)
            s["last_24h"] = store.stats_since(24)
            trends = _read_json(BASE_DIR / "data" / "sentiment_trends.json")
            if trends:
                s["trends_as_of"] = trends.get("as_of")
                s["trends_tracked"] = len(trends.get("tickers", {}))
            return jsonify(s)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.get("/api/trends")
    def api_trends():
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        data = _read_json(BASE_DIR / "data" / "sentiment_trends.json")
        if data is None:
            return jsonify({"error": "no trends yet"}), 503
        return jsonify(data)

    @app.get("/api/briefings")
    def api_briefings():
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        return jsonify(_briefings_from_log(10))

    @app.get("/healthz")
    def healthz():
        store = _store_handle()
        return jsonify({"ok": store is not None})

    @app.get("/chat")
    def chat_page() -> Response:
        return Response(_CHAT_HTML, mimetype="text/html")

    @app.post("/api/chat")
    def api_chat():
        try:
            payload = request.get_json(force=True, silent=True) or {}
        except Exception:
            payload = {}
        user_msg = (payload.get("message") or "").strip()
        history = payload.get("history") or []
        if not user_msg:
            return jsonify({"error": "empty message"}), 400

        # Pull article context — open a fresh read-only sqlite connection so we
        # don't clash with the daemon's writer thread.
        articles_ctx: list[dict] = []
        try:
            # Resolve the actual sqlite file the daemon's ArticleStore is using
            # (could be the USB mount at /media/zeph/projects/digital-intern/db).
            db_path: Path | None = None
            store = _store_handle()
            if store is not None:
                try:
                    for _id, name, file in store.conn.execute("PRAGMA database_list").fetchall():
                        if name == "main" and file:
                            db_path = Path(file)
                            break
                except Exception:
                    pass
            if db_path is None:
                # Fallbacks: USB mount first, then local repo path.
                for cand in (
                    Path("/media/zeph/projects/digital-intern/db/articles.db"),
                    BASE_DIR / "db" / "articles.db",
                ):
                    if cand.exists():
                        db_path = cand
                        break
            since = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat()
            uri = f"file:{db_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=5.0)
            try:
                rows = conn.execute(
                    "SELECT title, source, ai_score, full_text FROM articles "
                    "WHERE first_seen >= ? ORDER BY ai_score DESC LIMIT 10",
                    (since,),
                ).fetchall()
            finally:
                conn.close()
            for r in rows:
                summary = ""
                if r[3] is not None:
                    try:
                        summary = zlib.decompress(r[3]).decode("utf-8", errors="replace")[:300]
                    except Exception:
                        try:
                            summary = (r[3] if isinstance(r[3], str) else r[3].decode("utf-8", "replace"))[:300]
                        except Exception:
                            summary = ""
                articles_ctx.append({
                    "title": r[0] or "",
                    "source": r[1] or "",
                    "ai_score": float(r[2] or 0),
                    "summary": summary,
                })
        except Exception as e:
            _logger().warning("chat: article context fetch failed: %s", e)

        # Portfolio snapshot
        portfolio = _read_json(BASE_DIR / "data" / "portfolio_pl.json") or {}

        # Compact portfolio summary for the prompt
        portfolio_lines: list[str] = []
        try:
            s = (portfolio.get("summary") or {}) if isinstance(portfolio, dict) else {}
            if s:
                portfolio_lines.append(
                    f"Total value: ${s.get('grand_value', s.get('total_value', 'n/a'))} "
                    f"P&L: ${s.get('grand_pnl', s.get('total_pnl', 'n/a'))} "
                    f"({s.get('grand_pnl_pct', s.get('total_pnl_pct', 'n/a'))}%)"
                )
            for p in (portfolio.get("positions") or [])[:20]:
                try:
                    if float(p.get("qty") or 0) == 0:
                        continue
                except Exception:
                    pass
                portfolio_lines.append(
                    f"  {p.get('ticker','?')}: qty={p.get('qty')} avg=${p.get('avg_cost')} "
                    f"px=${p.get('price')} pnl=${p.get('pnl')} ({p.get('pnl_pct')}%)"
                )
            for o in (portfolio.get("options") or [])[:10]:
                portfolio_lines.append(
                    f"  OPT {o.get('symbol', o.get('ticker','?'))}: qty={o.get('qty')} pnl=${o.get('pnl')}"
                )
        except Exception:
            pass
        portfolio_block = "\n".join(portfolio_lines) if portfolio_lines else "(no portfolio snapshot available)"

        # Articles block
        if articles_ctx:
            art_lines = []
            for a in articles_ctx:
                art_lines.append(
                    f"{a['ai_score']:.2f} | {a['source']} | {a['title']}\n    {a['summary']}"
                )
            articles_block = "\n".join(art_lines)
        else:
            articles_block = "(no recent articles in the last 6 hours)"

        now_iso = datetime.now(timezone.utc).isoformat()
        system_prompt = (
            "You are a market intelligence analyst with access to a real-time news feed "
            "and portfolio data.\n"
            f"Current date: {now_iso}\n\n"
            "TOP NEWS SIGNALS (last 6h, ranked by ML score):\n"
            f"{articles_block}\n\n"
            "PORTFOLIO SNAPSHOT:\n"
            f"{portfolio_block}\n\n"
            "Answer questions about current market conditions, global events, specific "
            "stocks, or portfolio analysis. Be concise and data-driven. Cite specific "
            "articles when relevant."
        )

        # Build messages
        msgs: list[dict] = []
        for h in history[-20:]:
            role = h.get("role")
            content = (h.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                msgs.append({"role": role, "content": content})
        msgs.append({"role": "user", "content": user_msg})

        response_text = ""
        err: str | None = None

        if _ANTHROPIC_AVAILABLE and os.environ.get("ANTHROPIC_API_KEY"):
            try:
                client = anthropic.Anthropic()
                resp = client.messages.create(
                    model="claude-opus-4-7",
                    max_tokens=2048,
                    system=system_prompt,
                    messages=msgs,
                )
                parts = []
                for blk in resp.content or []:
                    text = getattr(blk, "text", None)
                    if text:
                        parts.append(text)
                response_text = "".join(parts).strip()
            except Exception as e:
                err = f"anthropic SDK error: {e}"
                _logger().warning("chat: %s", err)

        if not response_text:
            # Subprocess fallback to the Claude CLI (uses its own auth).
            try:
                convo_parts = [system_prompt, "\n\n--- Conversation ---"]
                for m in msgs:
                    convo_parts.append(f"{m['role'].upper()}: {m['content']}")
                convo_parts.append("ASSISTANT:")
                prompt = "\n\n".join(convo_parts)
                proc = subprocess.run(
                    ["claude", "--model", "claude-opus-4-7", "--print", prompt],
                    capture_output=True, text=True, timeout=120,
                )
                if proc.returncode == 0:
                    response_text = (proc.stdout or "").strip()
                else:
                    err = (err + " | " if err else "") + f"claude CLI exit {proc.returncode}: {proc.stderr.strip()[:300]}"
            except FileNotFoundError:
                err = (err + " | " if err else "") + "claude CLI not found and no ANTHROPIC_API_KEY"
            except subprocess.TimeoutExpired:
                err = (err + " | " if err else "") + "claude CLI timed out after 120s"
            except Exception as e:
                err = (err + " | " if err else "") + f"claude CLI error: {e}"

        if not response_text:
            return jsonify({"error": err or "no response from model"}), 502

        return jsonify({
            "response": response_text,
            "sources": [a["title"] for a in articles_ctx],
        })

    return app


def run_server(store, host: str = "0.0.0.0", port: int = 8080) -> None:
    """Blocking entry point used by the daemon worker thread."""
    app = create_app(store)
    # werkzeug's dev server is fine for read-only public dashboard at this scale.
    app.run(host=host, port=port, threaded=True, use_reloader=False, debug=False)


# ── HTML payload (Bootstrap dark theme via CDN, no npm) ─────────────────────
_DASHBOARD_HTML = """<!doctype html>
<html lang="en" data-bs-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Digital Intern — Dashboard</title>
  <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='6' fill='%230d1117'/%3E%3Cpolyline points='3,24 9,16 14,20 20,10 29,14' fill='none' stroke='%2300b4d8' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'/%3E%3Ccircle cx='20' cy='10' r='2.5' fill='%23e94560'/%3E%3Cline x1='3' y1='27' x2='29' y2='27' stroke='%2330363d' stroke-width='1'/%3E%3C/svg%3E">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body { background:#0d1117; color:#e6edf3; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; font-size: 16px; }
    .card { background:#161b22; border:1px solid #30363d; }
    .card-header { background:#21262d; font-weight:600; }
    .badge-urgent { background:#da3633; }
    .badge-score { background:#1f6feb; }
    .pl-pos { color:#3fb950; }
    .pl-neg { color:#f85149; }
    a { color:#58a6ff; }
    a:hover { color:#79c0ff; }
    .scroll-pane { max-height:520px; overflow-y:auto; }
    .ticker { font-weight:600; }
    .small-muted { color:#8b949e; font-size:0.85em; }
    table { color:#e6edf3; }
  </style>
</head>
<body>
<nav style="background:#1a1a2e;padding:12px 24px;display:flex;gap:24px;align-items:center;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif;border-bottom:1px solid #333;font-size:16px">
  <span style="color:#e94560;font-weight:bold;font-size:1.1em">◈ TRADING STACK</span>
  <a href="http://10.19.203.44:8888/" style="color:#00b4d8;text-decoration:none">Home</a>
  <a href="http://10.19.203.44:8888/intern/" style="color:#fff;border-bottom:2px solid #e94560;text-decoration:none">Digital Intern</a>
  <a href="http://10.19.203.44:8888/trader/" style="color:#00b4d8;text-decoration:none">Paper Trader</a>
  <a href="http://10.19.203.44:8888/trader/backtests" style="color:#00b4d8;text-decoration:none">Backtests</a>
  <a href="http://10.19.203.44:8765/" style="color:#00b4d8;text-decoration:none">Ops View</a>
  <a href="/chat" style="color:#00b4d8;text-decoration:none">Chat</a>
  <span style="margin-left:auto;color:#666;font-size:0.8em">10.19.203.44</span>
</nav>
<div class="container-fluid p-3">
  <div class="d-flex justify-content-between align-items-center mb-3">
    <h2 class="mb-0">Digital Intern</h2>
    <div class="small-muted" id="last-updated">loading…</div>
  </div>

  <div class="card mb-3" id="paper-trader-card">
    <div class="card-header d-flex justify-content-between align-items-center">
      <span>Live Paper Trader</span>
      <a href="http://10.19.203.44:8090" style="font-size:0.85em">View Full Trader →</a>
    </div>
    <div class="card-body p-2">
      <div id="paper-trader-summary" class="small-muted">loading…</div>
    </div>
  </div>

  <div class="row g-3">
    <div class="col-12 col-lg-4">
      <div class="card mb-3">
        <div class="card-header">Portfolio P&amp;L</div>
        <div class="card-body p-2">
          <div id="pnl-summary" class="mb-2 small-muted">loading…</div>
          <div class="table-responsive scroll-pane">
            <table class="table table-sm table-borderless mb-0">
              <thead><tr>
                <th>TICKER</th><th class="text-end">QTY</th>
                <th class="text-end">PRICE</th><th class="text-end">P/L</th>
              </tr></thead>
              <tbody id="pnl-rows"></tbody>
            </table>
          </div>
        </div>
      </div>

      <div class="card mb-3">
        <div class="card-header">Top Signals</div>
        <div class="card-body p-2 scroll-pane">
          <ul class="list-unstyled mb-0" id="signals-list"><li class="small-muted">loading…</li></ul>
        </div>
      </div>
    </div>

    <div class="col-12 col-lg-8">
      <div class="card mb-3">
        <div class="card-header">Recent Briefings</div>
        <div class="card-body p-2 scroll-pane">
          <ul class="list-unstyled mb-0" id="briefings-list"><li class="small-muted">loading…</li></ul>
        </div>
      </div>
      <div class="card mb-3">
        <div class="card-header">High-Score Articles</div>
        <div class="card-body p-2 scroll-pane">
          <ul class="list-unstyled mb-0" id="articles-list"><li class="small-muted">loading…</li></ul>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
const API_PREFIX = "__API_PREFIX__";
const params = new URLSearchParams(location.search);
const KEY = params.get("key") || "";
const qs = KEY ? ("?key=" + encodeURIComponent(KEY)) : "";

async function getJSON(path) {
  try {
    const r = await fetch(API_PREFIX + path + qs);
    if (!r.ok) return null;
    return await r.json();
  } catch (e) { return null; }
}

function fmt(n) {
  if (n === null || n === undefined) return "—";
  if (Math.abs(n) >= 1000) return n.toLocaleString(undefined, {maximumFractionDigits: 0});
  return Number(n).toFixed(2);
}

function plClass(v) { if (v === null || v === undefined) return ""; return v >= 0 ? "pl-pos" : "pl-neg"; }

async function refresh() {
  const [pnl, articles, briefings, stats] = await Promise.all([
    getJSON("/api/portfolio"),
    getJSON("/api/articles?limit=20"),
    getJSON("/api/briefings"),
    getJSON("/api/stats"),
  ]);

  const sumDiv = document.getElementById("pnl-summary");
  const rowsBody = document.getElementById("pnl-rows");
  rowsBody.innerHTML = "";
  if (!pnl || pnl.error) {
    sumDiv.textContent = (pnl && pnl.error) ? pnl.error : "no snapshot";
  } else {
    const s = pnl.summary || {};
    const totPnl = s.grand_pnl ?? s.total_pnl ?? 0;
    const totPnlPct = s.grand_pnl_pct ?? s.total_pnl_pct ?? 0;
    sumDiv.innerHTML = `<span class="ticker">Total</span> $${fmt(s.grand_value ?? s.total_value)} ` +
      `<span class="${plClass(totPnl)}">${(totPnl>=0?'+':'')}${fmt(totPnl)} (${(totPnlPct>=0?'+':'')}${fmt(totPnlPct)}%)</span>` +
      `<br><span class="small-muted">as of ${pnl.as_of || ""}</span>`;
    const allPositions = (pnl.positions || []).concat(pnl.options || []);
    for (const p of allPositions) {
      const tr = document.createElement("tr");
      const pnlV = p.pnl;
      tr.innerHTML = `<td class="ticker">${p.ticker || p.symbol || ""}</td>` +
        `<td class="text-end">${fmt(p.qty)}</td>` +
        `<td class="text-end">${p.price === null ? '—' : fmt(p.price)}</td>` +
        `<td class="text-end ${plClass(pnlV)}">${pnlV === null ? '—' : ((pnlV>=0?'+':'')+fmt(pnlV))}</td>`;
      rowsBody.appendChild(tr);
    }
  }

  const sigs = document.getElementById("signals-list");
  sigs.innerHTML = "";
  if (articles && Array.isArray(articles)) {
    const top = articles.filter(a => (a.urgency||0) >= 1 || (a.score||0) >= 6.0).slice(0, 10);
    if (!top.length) { sigs.innerHTML = '<li class="small-muted">no urgent signals</li>'; }
    for (const a of top) {
      const li = document.createElement("li");
      const urgent = (a.urgency||0) >= 1;
      li.className = "mb-1";
      li.innerHTML = `<span class="badge ${urgent?'badge-urgent':'badge-score'} me-1">${urgent?'URG':fmt(a.score)}</span>` +
        `<a href="${a.url}" target="_blank" rel="noopener">${a.title}</a>` +
        `<div class="small-muted">${a.source || ''} · ${a.published || ''}</div>`;
      sigs.appendChild(li);
    }
  } else {
    sigs.innerHTML = '<li class="small-muted">API requires &amp;key=… (see WEB_API_KEY)</li>';
  }

  const arts = document.getElementById("articles-list");
  arts.innerHTML = "";
  if (articles && Array.isArray(articles)) {
    for (const a of articles) {
      const li = document.createElement("li");
      li.className = "mb-2";
      li.innerHTML = `<div><span class="badge badge-score me-1">${fmt(a.score)}</span>` +
        `<a href="${a.url}" target="_blank" rel="noopener">${a.title}</a></div>` +
        `<div class="small-muted">${a.source || ''} · ${a.published || ''}</div>`;
      arts.appendChild(li);
    }
  }

  const brs = document.getElementById("briefings-list");
  brs.innerHTML = "";
  if (briefings && Array.isArray(briefings) && briefings.length) {
    for (const b of briefings) {
      const li = document.createElement("li");
      li.className = "mb-1 small-muted";
      li.innerHTML = `<span>${b.ts || ''}</span> — ${b.msg || ''}`;
      brs.appendChild(li);
    }
  } else {
    brs.innerHTML = '<li class="small-muted">no briefings logged yet</li>';
  }

  document.getElementById("last-updated").textContent = "updated " + new Date().toISOString();
}

async function refreshPaperTrader() {
  const sumDiv = document.getElementById("paper-trader-summary");
  try {
    const r = await fetch("http://10.19.203.44:8090/api/portfolio");
    if (!r.ok) { sumDiv.textContent = "paper trader unavailable (HTTP " + r.status + ")"; return; }
    const j = await r.json();
    const tv = j.total_value;
    const pl = tv - 1000;
    const plPct = (pl / 1000) * 100;
    const cls = pl >= 0 ? "pl-pos" : "pl-neg";
    sumDiv.innerHTML = `<span class="ticker">Portfolio</span> $${fmt(tv)} ` +
      `<span class="${cls}">${(pl>=0?'+':'')}${fmt(pl)} (${(pl>=0?'+':'')}${fmt(plPct)}%)</span> ` +
      `<span class="small-muted">vs $1000 start · cash $${fmt(j.cash)}</span>`;
  } catch (e) {
    sumDiv.textContent = "paper trader unreachable";
  }
}

refresh();
refreshPaperTrader();
setInterval(refresh, 15000);
setInterval(refreshPaperTrader, 15000);
</script>

<!-- Floating chat widget -->
<button id="dichat-btn" aria-label="Open chat"
  style="position:fixed;bottom:20px;right:20px;width:56px;height:56px;border-radius:50%;background:#1565c0;color:#fff;border:none;font-size:24px;cursor:pointer;box-shadow:0 4px 14px rgba(0,0,0,0.5);z-index:9998">✦</button>
<div id="dichat-panel"
  style="display:none;position:fixed;bottom:88px;right:20px;width:360px;height:480px;background:#11161d;color:#cfd8dc;border:1px solid #30363d;border-radius:12px;box-shadow:0 8px 24px rgba(0,0,0,0.5);z-index:9999;flex-direction:column;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif">
  <div style="padding:12px 14px;border-bottom:1px solid #21262d;display:flex;justify-content:space-between;align-items:center">
    <span style="font-weight:600;color:#e6edf3">Market Intel</span>
    <a href="/chat" style="color:#8b949e;font-size:0.8em;text-decoration:none;margin-left:auto;margin-right:10px">full ↗</a>
    <button id="dichat-close" style="background:none;border:none;color:#8b949e;cursor:pointer;font-size:20px;line-height:1;padding:0 4px">×</button>
  </div>
  <div id="dichat-history" style="flex:1;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:10px;font-size:14px;line-height:1.45"></div>
  <form id="dichat-form" style="display:flex;gap:6px;padding:10px;border-top:1px solid #21262d">
    <input id="dichat-input" type="text" autocomplete="off" placeholder="Ask about markets…"
      style="flex:1;background:#0d1117;border:1px solid #30363d;color:#e6edf3;padding:8px 10px;border-radius:6px;font-size:14px;font-family:inherit">
    <button type="submit" id="dichat-send"
      style="background:#1565c0;color:#fff;border:none;padding:8px 14px;border-radius:6px;cursor:pointer;font-weight:600">Send</button>
  </form>
</div>
<script>
(function(){
  const btn = document.getElementById('dichat-btn');
  const panel = document.getElementById('dichat-panel');
  const closeBtn = document.getElementById('dichat-close');
  const hist = document.getElementById('dichat-history');
  const form = document.getElementById('dichat-form');
  const inp = document.getElementById('dichat-input');
  const sendBtn = document.getElementById('dichat-send');
  const convo = [];
  let greeted = false;
  function openPanel(){ panel.style.display='flex'; btn.style.display='none'; inp.focus();
    if(!greeted){ bubble('assistant','Hi — ask about markets, news, or your portfolio.'); greeted=true; } }
  function closePanel(){ panel.style.display='none'; btn.style.display='block'; }
  btn.addEventListener('click', openPanel);
  closeBtn.addEventListener('click', closePanel);
  function bubble(role, text){
    const d = document.createElement('div');
    const bg = role==='user' ? '#1565c0' : role==='error' ? '#4a1d1d' : '#21262d';
    const fg = role==='user' ? '#fff' : role==='error' ? '#ffd6d6' : '#cfd8dc';
    const align = role==='user' ? 'flex-end' : 'flex-start';
    d.style.cssText = 'background:'+bg+';color:'+fg+';padding:8px 12px;border-radius:10px;max-width:85%;align-self:'+align+';white-space:pre-wrap;word-wrap:break-word';
    d.textContent = text;
    hist.appendChild(d);
    hist.scrollTop = hist.scrollHeight;
    return d;
  }
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const m = inp.value.trim();
    if (!m || sendBtn.disabled) return;
    bubble('user', m);
    inp.value = '';
    sendBtn.disabled = true;
    const tmp = bubble('assistant', 'thinking…');
    tmp.style.opacity = '0.6';
    try {
      const r = await fetch('/api/chat', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({message:m, history: convo.slice()})
      });
      const j = await r.json();
      tmp.remove();
      if (!r.ok || j.error) {
        bubble('error', 'Error: ' + (j.error || ('HTTP '+r.status)));
      } else {
        const txt = j.response || j.reply || '(empty response)';
        bubble('assistant', txt);
        convo.push({role:'user', content:m});
        convo.push({role:'assistant', content:txt});
      }
    } catch (err) {
      tmp.remove();
      bubble('error', 'Network error: ' + err.message);
    } finally {
      sendBtn.disabled = false;
      inp.focus();
    }
  });
})();
</script>
</body>
</html>
"""


_CHAT_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Market Intel — Digital Intern</title>
  <style>
    * { box-sizing: border-box; }
    html, body { margin: 0; padding: 0; height: 100%; }
    body {
      background: #0d1117; color: #e6edf3;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
      font-size: 16px;
      display: flex; flex-direction: column; height: 100vh;
    }
    nav {
      background: #1a1a2e; padding: 12px 24px; display: flex; gap: 24px;
      align-items: center; border-bottom: 1px solid #333;
    }
    nav .brand { color: #e94560; font-weight: bold; font-size: 1.1em; }
    nav a { color: #00b4d8; text-decoration: none; }
    nav a.active { color: #fff; border-bottom: 2px solid #e94560; }
    header.page {
      padding: 18px 24px 10px; border-bottom: 1px solid #21262d;
      background: #0d1117;
    }
    header.page h1 { margin: 0; font-size: 1.4em; }
    header.page .sub { color: #8b949e; font-size: 0.9em; margin-top: 4px; }
    .chat-wrap {
      flex: 1; overflow-y: auto; padding: 20px 24px;
      display: flex; flex-direction: column; gap: 14px;
    }
    .msg { max-width: 760px; padding: 12px 16px; border-radius: 12px; white-space: pre-wrap; word-wrap: break-word; line-height: 1.5; }
    .msg.user { align-self: flex-end; background: #1f6feb; color: #fff; }
    .msg.assistant { align-self: flex-start; background: #21262d; border: 1px solid #30363d; }
    .msg.error { align-self: flex-start; background: #4a1d1d; border: 1px solid #f85149; color: #ffd6d6; }
    .sources { align-self: flex-start; max-width: 760px; display: flex; flex-wrap: wrap; gap: 6px; margin-top: -6px; }
    .chip {
      background: #161b22; border: 1px solid #30363d; color: #8b949e;
      font-size: 0.78em; padding: 3px 8px; border-radius: 10px;
    }
    .suggestions { display: flex; flex-wrap: wrap; gap: 8px; padding: 0 24px 8px; }
    .suggestion {
      background: #161b22; border: 1px solid #30363d; color: #58a6ff;
      padding: 8px 14px; border-radius: 18px; cursor: pointer; font-size: 0.9em;
    }
    .suggestion:hover { background: #21262d; }
    form.input-bar {
      display: flex; gap: 10px; padding: 14px 24px; border-top: 1px solid #21262d; background: #0d1117;
    }
    input.msg-input {
      flex: 1; background: #161b22; border: 1px solid #30363d; color: #e6edf3;
      padding: 12px 14px; border-radius: 8px; font-size: 16px;
      font-family: inherit;
    }
    input.msg-input:focus { outline: none; border-color: #58a6ff; }
    button.send {
      background: #1f6feb; color: #fff; border: none; padding: 12px 22px;
      border-radius: 8px; font-size: 16px; cursor: pointer; font-weight: 600;
    }
    button.send:disabled { background: #30363d; cursor: not-allowed; }
    .spinner {
      display: inline-block; width: 14px; height: 14px; border: 2px solid #30363d;
      border-top-color: #58a6ff; border-radius: 50%;
      animation: spin 0.8s linear infinite; vertical-align: middle; margin-right: 8px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .typing { color: #8b949e; font-style: italic; }
  </style>
</head>
<body>
<nav>
  <span class="brand">◈ TRADING STACK</span>
  <a href="http://10.19.203.44:8888/">Home</a>
  <a href="/">Digital Intern</a>
  <a href="http://10.19.203.44:8888/trader/">Paper Trader</a>
  <a href="http://10.19.203.44:8765/">Ops View</a>
  <a href="/chat" class="active">Chat</a>
</nav>
<header class="page">
  <h1>Market Intel</h1>
  <div class="sub">Powered by Claude Opus 4.7 + Live News Feed</div>
</header>
<div class="suggestions" id="suggestions">
  <button class="suggestion" data-q="What's moving markets today?">What's moving markets today?</button>
  <button class="suggestion" data-q="Nokia surge analysis">Nokia surge analysis</button>
  <button class="suggestion" data-q="Best opportunities right now?">Best opportunities right now?</button>
  <button class="suggestion" data-q="What should I watch in Asia overnight?">What should I watch in Asia overnight?</button>
</div>
<div class="chat-wrap" id="chat"></div>
<form class="input-bar" id="form">
  <input class="msg-input" id="input" type="text" placeholder="Ask about markets, news, or your portfolio…" autocomplete="off" autofocus>
  <button class="send" id="send" type="submit">Send</button>
</form>
<script>
const chat = document.getElementById('chat');
const form = document.getElementById('form');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
const history = [];

function scrollDown() {
  requestAnimationFrame(() => { chat.scrollTop = chat.scrollHeight; });
}

function addMsg(role, content, sources) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.textContent = content;
  chat.appendChild(div);
  if (sources && sources.length) {
    const wrap = document.createElement('div');
    wrap.className = 'sources';
    for (const s of sources) {
      const chip = document.createElement('span');
      chip.className = 'chip';
      chip.textContent = s.length > 80 ? s.slice(0, 78) + '…' : s;
      wrap.appendChild(chip);
    }
    chat.appendChild(wrap);
  }
  scrollDown();
  return div;
}

function addLoader() {
  const div = document.createElement('div');
  div.className = 'msg assistant typing';
  div.innerHTML = '<span class="spinner"></span>Thinking…';
  chat.appendChild(div);
  scrollDown();
  return div;
}

async function ask(message) {
  if (!message) return;
  addMsg('user', message);
  history.push({role: 'user', content: message});
  input.value = '';
  sendBtn.disabled = true;
  const loader = addLoader();
  try {
    const r = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: message, history: history.slice(0, -1)}),
    });
    const j = await r.json();
    loader.remove();
    if (!r.ok || j.error) {
      addMsg('error', 'Error: ' + (j.error || ('HTTP ' + r.status)));
    } else {
      addMsg('assistant', j.response, j.sources || []);
      history.push({role: 'assistant', content: j.response});
    }
  } catch (e) {
    loader.remove();
    addMsg('error', 'Network error: ' + e.message);
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

form.addEventListener('submit', (e) => {
  e.preventDefault();
  ask(input.value.trim());
});

document.getElementById('suggestions').addEventListener('click', (e) => {
  const b = e.target.closest('.suggestion');
  if (!b) return;
  ask(b.dataset.q);
});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    # Standalone runtime: open the store directly and serve.
    import sys
    sys.path.insert(0, str(BASE_DIR))
    from storage.article_store import ArticleStore  # noqa: E402
    run_server(ArticleStore())

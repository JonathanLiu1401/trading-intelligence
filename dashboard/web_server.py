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
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request

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
        return Response(_DASHBOARD_HTML, mimetype="text/html")

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
const params = new URLSearchParams(location.search);
const KEY = params.get("key") || "";
const qs = KEY ? ("?key=" + encodeURIComponent(KEY)) : "";

async function getJSON(path) {
  try {
    const r = await fetch(path + qs);
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
</body>
</html>
"""


if __name__ == "__main__":
    # Standalone runtime: open the store directly and serve.
    import sys
    sys.path.insert(0, str(BASE_DIR))
    from storage.article_store import ArticleStore  # noqa: E402
    run_server(ArticleStore())

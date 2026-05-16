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
import zlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request

from core.claude_cli import claude_call as _claude_cli_call

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
            "AND url NOT LIKE 'backtest://%' "
            "AND source NOT LIKE 'backtest_%' "
            "AND source NOT LIKE 'opus_annotation%' "
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

    CONFIG_PATH = BASE_DIR / "config" / "portfolio.json"

    @app.get("/api/portfolio/config")
    def api_portfolio_config_get():
        import json as _json
        try:
            data = _json.loads(CONFIG_PATH.read_text())
        except Exception:
            data = {"positions": [], "options": [], "sector_watchlist": []}
        return jsonify(data)

    @app.put("/api/portfolio/config")
    def api_portfolio_config_put():
        import json as _json
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify({"error": "invalid JSON"}), 400
        # Keep _note and _account metadata if present
        existing = {}
        try:
            existing = _json.loads(CONFIG_PATH.read_text())
        except Exception:
            pass
        body["_note"] = f"Sao's trading portfolio - updated via UI"
        body.setdefault("_account", existing.get("_account", {}))
        CONFIG_PATH.write_text(_json.dumps(body, indent=2))
        return jsonify({"ok": True})

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

    @app.get("/api/earnings")
    def api_earnings():
        """Upcoming earnings within the snapshot's horizon (default 14d).

        Reads ``data/earnings_calendar.json`` written by ``write_snapshot()``.
        Also reports the snapshot age so the dashboard can render a freshness
        indicator and trigger a background refresh when stale.
        """
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        snap = _read_json(BASE_DIR / "data" / "earnings_calendar.json")
        if snap is None:
            return jsonify({
                "error": "no snapshot yet",
                "hint": "run: python3 -m collectors.earnings_calendar",
            }), 503
        # Recompute days_away on each request so a stale snapshot still shows
        # accurate counters until the daemon refreshes it.
        now = datetime.now(timezone.utc)
        try:
            for ev in snap.get("events", []) or []:
                ts = ev.get("earnings_date")
                if not ts:
                    continue
                try:
                    ed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except Exception:
                    continue
                if ed.tzinfo is None:
                    ed = ed.replace(tzinfo=timezone.utc)
                ev["days_away"] = round((ed - now).total_seconds() / 86400.0, 2)
        except Exception:
            pass
        # Drop events that have already passed once the recompute runs,
        # then sort soonest-first so dashboards can render in order.
        snap["events"] = sorted(
            [e for e in snap.get("events", []) or []
             if (e.get("days_away") or 0) >= -0.5],
            key=lambda e: e.get("days_away") if e.get("days_away") is not None else float("inf"),
        )
        snap["n_events"] = len(snap["events"])
        snap["n_within_7d"] = sum(
            1 for e in snap["events"]
            if -0.5 <= (e.get("days_away") or 0) <= 7.0
        )
        snap["next_event"] = snap["events"][0] if snap["events"] else None
        # Snapshot age for staleness rendering.
        try:
            as_of = datetime.fromisoformat((snap.get("as_of") or "").replace("Z", "+00:00"))
            if as_of.tzinfo is None:
                as_of = as_of.replace(tzinfo=timezone.utc)
            snap["age_hours"] = round((now - as_of).total_seconds() / 3600.0, 2)
        except Exception:
            snap["age_hours"] = None
        return jsonify(snap)

    def _ro_conn():
        """Open a fresh read-only sqlite connection to the daemon's articles.db.

        Mirrors the resolution logic used in /api/chat: prefer the path the
        ArticleStore actually opened, then fall back to USB and local repo
        paths. Returns ``None`` if no DB can be located.
        """
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
            for cand in (
                Path("/media/zeph/projects/digital-intern/db/articles.db"),
                BASE_DIR / "data" / "articles.db",
                BASE_DIR / "db" / "articles.db",
            ):
                if cand.exists():
                    db_path = cand
                    break
        if db_path is None:
            return None
        try:
            uri = f"file:{db_path}?mode=ro"
            return sqlite3.connect(uri, uri=True, timeout=5.0)
        except sqlite3.Error:
            return None

    _LIVE_ONLY_SQL = (
        "url NOT LIKE 'backtest://%' "
        "AND source NOT LIKE 'backtest_%' "
        "AND source NOT LIKE 'opus_annotation%'"
    )

    @app.get("/api/collector-health")
    def api_collector_health():
        """Per-source article counts for last 1h and 24h with status thresholds.

        - active: ≥10 articles in the last hour
        - slow:   1..9  articles in the last hour
        - stale:  0 articles in the last 2 hours
        - idle:   anything else (e.g. recently active but quiet in the past hour)
        """
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        conn = _ro_conn()
        if conn is None:
            return jsonify({"sources": [], "error": "articles.db not reachable"})
        try:
            rows_1h = conn.execute(
                f"SELECT source, COUNT(*) FROM articles "
                f"WHERE first_seen >= datetime('now','-1 hour') AND {_LIVE_ONLY_SQL} "
                f"GROUP BY source"
            ).fetchall()
            rows_24h = conn.execute(
                f"SELECT source, COUNT(*) FROM articles "
                f"WHERE first_seen >= datetime('now','-24 hours') AND {_LIVE_ONLY_SQL} "
                f"GROUP BY source"
            ).fetchall()
            rows_2h = conn.execute(
                f"SELECT source, COUNT(*) FROM articles "
                f"WHERE first_seen >= datetime('now','-2 hours') AND {_LIVE_ONLY_SQL} "
                f"GROUP BY source"
            ).fetchall()
        finally:
            conn.close()
        c1h = {r[0] or "?": int(r[1] or 0) for r in rows_1h}
        c2h = {r[0] or "?": int(r[1] or 0) for r in rows_2h}
        c24 = {r[0] or "?": int(r[1] or 0) for r in rows_24h}
        names = set(c1h) | set(c2h) | set(c24)
        out = []
        for n in names:
            h1 = c1h.get(n, 0)
            h2 = c2h.get(n, 0)
            h24 = c24.get(n, 0)
            if h2 == 0:
                status = "stale"
            elif h1 >= 10:
                status = "active"
            elif h1 >= 1:
                status = "slow"
            else:
                status = "idle"
            out.append({
                "source": n,
                "articles_1h": h1,
                "articles_24h": h24,
                "status": status,
            })
        out.sort(key=lambda r: (-r["articles_1h"], -r["articles_24h"], r["source"]))
        return jsonify({"sources": out, "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds")})

    @app.get("/api/ml-status")
    def api_ml_status():
        """ArticleNet snapshot — last trained, training-set size, predictions today.

        Pulls the checkpoint mtime as ``last_trained`` and grep-scans the
        structured log for the most recent ``[ml_trainer] Bootstrap done`` line
        to recover ``val_loss`` (the trainer logs it on every retrain, see
        ml/trainer.py).
        """
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        ckpt = BASE_DIR / "data" / "ml" / "model_gpu.pt"
        last_trained = None
        if ckpt.exists():
            try:
                last_trained = datetime.fromtimestamp(
                    ckpt.stat().st_mtime, tz=timezone.utc
                ).isoformat(timespec="seconds")
            except Exception:
                last_trained = None
        training_set_size = None
        predictions_24h = None
        urgent_24h = None
        conn = _ro_conn()
        if conn is not None:
            try:
                # ArticleNet trains on rows with any ML/LLM-assigned score;
                # `kw_score` is the pure-heuristic fallback we exclude here.
                training_set_size = int(conn.execute(
                    "SELECT COUNT(*) FROM articles WHERE ai_score > 0"
                ).fetchone()[0] or 0)
                # Articles scored in the past 24h are a reasonable proxy for
                # inference throughput; there is no `score_source` column in
                # this schema (see articles table definition).
                predictions_24h = int(conn.execute(
                    "SELECT COUNT(*) FROM articles WHERE ai_score > 0 "
                    f"AND first_seen >= datetime('now','-24 hours') AND {_LIVE_ONLY_SQL}"
                ).fetchone()[0] or 0)
                urgent_24h = int(conn.execute(
                    "SELECT COUNT(*) FROM articles WHERE urgency >= 1 "
                    f"AND first_seen >= datetime('now','-24 hours') AND {_LIVE_ONLY_SQL}"
                ).fetchone()[0] or 0)
            except Exception:
                pass
            finally:
                conn.close()
        val_loss = None
        try:
            log_path = BASE_DIR / "logs" / "structured.jsonl"
            if log_path.exists():
                size = log_path.stat().st_size
                with log_path.open("rb") as f:
                    f.seek(max(0, size - 512 * 1024))
                    data = f.read()
                for raw in reversed(data.splitlines()):
                    try:
                        ln = json.loads(raw.decode("utf-8", errors="replace"))
                    except Exception:
                        continue
                    msg = ln.get("msg", "")
                    if "[ml_trainer] Bootstrap done" in msg or "val_loss" in msg:
                        # crude extraction: find val_loss=… or 'val_loss': …
                        import re as _re
                        m = _re.search(r"val_loss['\":=\s]+([0-9]+\.?[0-9]*)", msg)
                        if m:
                            try:
                                val_loss = float(m.group(1))
                                break
                            except Exception:
                                pass
        except Exception:
            pass
        return jsonify({
            "last_trained": last_trained,
            "training_set_size": training_set_size,
            "predictions_24h": predictions_24h,
            "urgent_24h": urgent_24h,
            "val_loss": val_loss,
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })

    @app.get("/api/volume-history")
    def api_volume_history():
        """Hourly article ingest counts for the last 24 hours, live rows only."""
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        conn = _ro_conn()
        if conn is None:
            return jsonify({"hours": [], "error": "articles.db not reachable"})
        try:
            rows = conn.execute(
                "SELECT strftime('%Y-%m-%dT%H:00', first_seen) AS hour, COUNT(*) "
                "FROM articles "
                f"WHERE first_seen >= datetime('now','-24 hours') AND {_LIVE_ONLY_SQL} "
                "GROUP BY hour ORDER BY hour"
            ).fetchall()
        finally:
            conn.close()
        return jsonify({
            "hours": [{"hour": r[0], "count": int(r[1] or 0)} for r in rows],
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })

    @app.get("/api/invariants")
    def api_invariants():
        """Backtest data-isolation status.

        Per the cross-system invariant (digital-intern CLAUDE.md §5): any
        ``backtest://`` row that has been alerted is a contamination breach —
        live alerts must only fire on live news.
        """
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        conn = _ro_conn()
        if conn is None:
            return jsonify({"backtest_isolation": "unknown", "error": "articles.db not reachable"})
        try:
            breach = int(conn.execute(
                "SELECT COUNT(*) FROM articles "
                "WHERE (url LIKE 'backtest://%' OR source LIKE 'backtest_%' "
                "      OR source LIKE 'opus_annotation%') "
                "AND urgency >= 2"
            ).fetchone()[0] or 0)
            n_backtest_total = int(conn.execute(
                "SELECT COUNT(*) FROM articles "
                "WHERE url LIKE 'backtest://%' OR source LIKE 'backtest_%' "
                "      OR source LIKE 'opus_annotation%'"
            ).fetchone()[0] or 0)
        finally:
            conn.close()
        return jsonify({
            "backtest_isolation": "breach" if breach > 0 else "active",
            "breach_count": breach,
            "backtest_rows_total": n_backtest_total,
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })

    @app.get("/healthz")
    def healthz():
        store = _store_handle()
        return jsonify({"ok": store is not None})

    @app.get("/chat")
    def chat_page() -> Response:
        return Response(_CHAT_HTML, mimetype="text/html",
                        headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

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
                    "WHERE first_seen >= ? "
                    "AND url NOT LIKE 'backtest://%' "
                    "AND source NOT LIKE 'backtest_%' "
                    "AND source NOT LIKE 'opus_annotation%' "
                    "ORDER BY ai_score DESC LIMIT 10",
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

        # Live paper-trader state — fetch from :8090/api/state. Adds positions,
        # recent trades, recent decisions so the chat can answer "what did the
        # paper trader do today" / "why is SOXL the position".
        paper_trader_block = "(paper trader unreachable)"
        try:
            import urllib.request as _urllib
            with _urllib.urlopen("http://127.0.0.1:8090/api/state", timeout=3) as resp:
                pt = json.loads(resp.read().decode("utf-8"))
            pt_pf = pt.get("portfolio") or {}
            pt_total = pt_pf.get("total_value")
            pt_cash = pt_pf.get("cash")
            pt_pl = (pt_total - 1000.0) if isinstance(pt_total, (int, float)) else None
            pt_pl_pct = (pt_pl / 1000.0 * 100.0) if pt_pl is not None else None

            lines = []
            if pt_total is not None:
                lines.append(
                    f"Total: ${pt_total:.2f}  Cash: ${pt_cash:.2f}  "
                    f"P/L vs $1000 start: {('+' if (pt_pl or 0)>=0 else '')}${pt_pl:.2f} "
                    f"({pt_pl_pct:+.2f}%)"
                )
            sp = pt.get("sp500")
            if sp:
                lines.append(f"S&P 500: {sp:.2f}")

            pt_positions = pt.get("positions") or []
            if pt_positions:
                lines.append("Open positions:")
                for p in pt_positions[:15]:
                    if p.get("type") in ("call", "put"):
                        lines.append(
                            f"  {p.get('ticker','?')} {str(p.get('type','')).upper()} "
                            f"{p.get('strike')} {p.get('expiry')}: qty={p.get('qty')} "
                            f"avg=${p.get('avg_cost')} mark=${p.get('current_price')} "
                            f"P/L=${(p.get('unrealized_pl') or 0):.2f}"
                        )
                    else:
                        lines.append(
                            f"  {p.get('ticker','?')}: qty={p.get('qty')} "
                            f"avg=${(p.get('avg_cost') or 0):.2f} mark=${(p.get('current_price') or 0):.2f} "
                            f"P/L=${(p.get('unrealized_pl') or 0):.2f} ({(p.get('pl_pct') or 0):.1f}%)"
                        )
            else:
                lines.append("Open positions: (none)")

            pt_trades = pt.get("trades") or []
            if pt_trades:
                lines.append(f"Last {min(5, len(pt_trades))} trades:")
                for t in pt_trades[:5]:
                    extra = ""
                    if t.get("option_type"):
                        extra = f" {t.get('strike')}{t.get('option_type','')[0].upper()} {t.get('expiry')}"
                    lines.append(
                        f"  [{(t.get('timestamp') or '')[5:16].replace('T',' ')}] "
                        f"{t.get('action')} {t.get('qty')} {t.get('ticker')}{extra} @ ${(t.get('price') or 0):.2f}"
                    )

            pt_decisions = pt.get("decisions") or []
            if pt_decisions:
                lines.append(f"Last {min(3, len(pt_decisions))} decisions:")
                for d in pt_decisions[:3]:
                    reasoning = ""
                    try:
                        j = json.loads(d.get("reasoning") or "{}")
                        reasoning = (j.get("decision") or {}).get("reasoning") or j.get("detail") or ""
                    except Exception:
                        reasoning = d.get("reasoning") or ""
                    lines.append(
                        f"  [{(d.get('timestamp') or '')[5:16].replace('T',' ')}] "
                        f"{d.get('action_taken','')}: {reasoning[:160]}"
                    )

            # Equity curve trend (last ~6 points spaced over recent history)
            eq = pt.get("equity") or []
            if len(eq) >= 6:
                step = max(1, len(eq) // 6)
                sample = eq[::step][-6:]
                trend = " → ".join(f"${(p.get('total_value') or 0):.2f}" for p in sample)
                lines.append(f"Equity trend (recent): {trend}")

            paper_trader_block = "\n".join(lines) if lines else "(no paper-trader state)"
        except Exception as e:
            _logger().warning("chat: paper trader state fetch failed: %s", e)

        # Pull options Greeks (live trader's portfolio-level delta/gamma/theta/vega).
        # Useful when the user asks "am I overexposed?" or "what happens if NVDA drops 5%?"
        greeks_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen("http://127.0.0.1:8090/api/greeks", timeout=3) as resp:
                gk = json.loads(resp.read().decode("utf-8"))
            if not gk.get("error"):
                t = gk.get("totals") or {}
                rows = gk.get("positions") or []
                opt_rows = [r for r in rows if r.get("type") in ("call", "put")]
                if opt_rows:
                    lines = [
                        f"Net delta: {t.get('delta', 0):+.2f} (gross $ {t.get('gross_notional', 0):.0f})",
                        f"Net gamma: {t.get('gamma', 0):+.5f}",
                        f"Theta / day: ${t.get('theta', 0):+.2f}",
                        f"Vega / 1% IV: ${t.get('vega', 0):+.2f}",
                    ]
                    for o in opt_rows[:8]:
                        dte = o.get("days_to_expiry")
                        lines.append(
                            f"  {o.get('ticker')} {str(o.get('type','')).upper()} "
                            f"{o.get('strike')}/{o.get('expiry')} "
                            f"({dte}d): Δ {o.get('delta'):+.2f} Θ {o.get('theta'):+.2f} "
                            f"IV {(o.get('iv') or 0)*100:.0f}%"
                        )
                    greeks_block = "\n".join(lines)
        except Exception as e:
            _logger().warning("chat: greeks fetch failed: %s", e)

        # Pull DRAM/semis sector heatmap so the chat can answer "which semis are
        # leading today" without the user having to look at the dashboard.
        heatmap_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen("http://127.0.0.1:8090/api/sector-heatmap", timeout=4) as resp:
                hm = json.loads(resp.read().decode("utf-8"))
            if not hm.get("error"):
                ref = hm.get("reference_mom_5d")
                bits = [f"Reference: {hm.get('reference', 'SOXX')} 5d {ref:+.2f}%" if ref is not None else "Reference: —"]
                for b in (hm.get("buckets") or []):
                    avg = b.get("avg_mom_5d")
                    if avg is None:
                        continue
                    # Sort tickers within each bucket by mom_5d desc and keep top 3.
                    ticks = sorted(
                        [t for t in (b.get("tickers") or []) if t.get("mom_5d") is not None],
                        key=lambda t: -t["mom_5d"],
                    )
                    head = ", ".join(
                        f"{t['ticker']} {t['mom_5d']:+.1f}%" for t in ticks[:3]
                    )
                    urg = sum((t.get("urgent") or 0) for t in (b.get("tickers") or []))
                    urg_str = f" [{urg} urgent news]" if urg else ""
                    bits.append(f"  {b.get('name','?'):18s} avg {avg:+.2f}%{urg_str}  · top: {head}")
                heatmap_block = "\n".join(bits)
        except Exception as e:
            _logger().warning("chat: heatmap fetch failed: %s", e)

        # Pull portfolio analytics (sector exposure, drawdown, win rate, daily P/L)
        analytics_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen("http://127.0.0.1:8090/api/analytics", timeout=3) as resp:
                an = json.loads(resp.read().decode("utf-8"))
            if not an.get("error"):
                bits = []
                if an.get("daily_pl_usd") is not None:
                    bits.append(
                        f"Today P/L: ${an['daily_pl_usd']:+.2f} ({an.get('daily_pl_pct', 0):+.2f}%)"
                    )
                if an.get("max_drawdown_pct"):
                    bits.append(
                        f"Max DD: -${an['max_drawdown_usd']:.2f} ({an['max_drawdown_pct']:.2f}%)"
                    )
                if an.get("sharpe_annualized") is not None:
                    bits.append(f"Sharpe (ann.): {an['sharpe_annualized']}")
                if an.get("sortino_annualized") is not None:
                    bits.append(f"Sortino (ann.): {an['sortino_annualized']}")
                if an.get("calmar_ratio") is not None:
                    bits.append(f"Calmar: {an['calmar_ratio']}")
                if an.get("win_rate_pct") is not None:
                    bits.append(
                        f"Win rate: {an['win_rate_pct']}% over {an.get('n_round_trips', 0)} round-trips"
                    )
                if an.get("profit_factor") is not None:
                    bits.append(f"Profit factor: {an['profit_factor']}")
                if an.get("avg_holding_days") is not None:
                    bits.append(f"Avg holding period: {an['avg_holding_days']}d")
                if an.get("sp500_beta") is not None:
                    corr = an.get("sp500_correlation")
                    bits.append(
                        f"S&P 500 beta: {an['sp500_beta']}"
                        + (f" (corr {corr})" if corr is not None else "")
                    )
                if an.get("realized_pl_usd"):
                    bits.append(f"Realized P/L: ${an['realized_pl_usd']:+.2f}")
                sectors = an.get("sector_exposure_pct") or {}
                if sectors:
                    top_secs = sorted(sectors.items(), key=lambda kv: -kv[1])[:5]
                    bits.append("Sector exposure: " +
                                ", ".join(f"{s}={p:.1f}%" for s, p in top_secs) +
                                f", cash={an.get('cash_pct', 0):.1f}%")
                if bits:
                    analytics_block = "\n".join(bits)
        except Exception as e:
            _logger().warning("chat: analytics fetch failed: %s", e)

        # Earnings radar — scheduled gap risk on the paper trader's holdings.
        # Lets the chat warn "you hold NVDA and it prints in 4 days".
        earnings_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/earnings-risk", timeout=3) as resp:
                er = json.loads(resp.read().decode("utf-8"))
            if not er.get("error") and er.get("events"):
                e_lines = []
                for ev in er["events"][:10]:
                    days = ev.get("days_away")
                    days_s = f"{days:.1f}d" if isinstance(days, (int, float)) else "?"
                    flag = " [HELD]" if ev.get("held") else " (watchlist)"
                    e_lines.append(
                        f"  {ev.get('ticker')}: reports in {days_s}{flag}"
                        + (f", ${ev.get('exposure_usd',0):.0f} exposure" if ev.get("held") else "")
                    )
                if e_lines:
                    hdr = (f"{er.get('n_held_reporting',0)} holding(s) reporting soon, "
                           f"{er.get('n_imminent',0)} within 3 days:")
                    earnings_block = hdr + "\n" + "\n".join(e_lines)
        except Exception as e:
            _logger().warning("chat: earnings-risk fetch failed: %s", e)

        now_iso = datetime.now(timezone.utc).isoformat()
        system_prompt = (
            "You are a market intelligence analyst with access to a real-time news feed, "
            "the user's real portfolio, and a separate live paper trading bot (Claude Opus 4.7) "
            "running on a $1000 simulated portfolio.\n"
            f"Current date: {now_iso}\n\n"
            "TOP NEWS SIGNALS (last 6h, ranked by ML score):\n"
            f"{articles_block}\n\n"
            "USER'S REAL PORTFOLIO SNAPSHOT:\n"
            f"{portfolio_block}\n\n"
            "PAPER TRADER LIVE STATE (separate $1000 sim run by Opus 4.7 every 30 min):\n"
            f"{paper_trader_block}\n\n"
            + (f"PAPER TRADER ANALYTICS:\n{analytics_block}\n\n" if analytics_block else "")
            + (f"PAPER TRADER OPTIONS GREEKS (Black-Scholes, live IV):\n{greeks_block}\n\n" if greeks_block else "")
            + (f"DRAM / SEMIS 5d MOMENTUM HEATMAP:\n{heatmap_block}\n\n" if heatmap_block else "")
            + (f"EARNINGS RADAR (scheduled gap risk):\n{earnings_block}\n\n" if earnings_block else "")
            + "Answer questions about current market conditions, global events, specific "
            "stocks, the user's real portfolio, or the paper trader's positions/decisions. "
            "Be concise and data-driven. Cite specific articles when relevant. When the user "
            "asks 'how am I doing', show real-portfolio first then paper-trader as separate "
            "lines so they aren't confused. The user's thesis-focus is DRAM/memory (MU, WDC, "
            "STX) plus semis equipment (LRCX, AMAT, KLAC, ASML) and the HBM-ramp design "
            "winners (NVDA, AMD, AVGO); weight your reads of the heatmap and news through "
            "that lens unless the user asks otherwise."
        )

        # Build messages
        msgs: list[dict] = []
        for h in history[-20:]:
            role = h.get("role")
            content = (h.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                msgs.append({"role": role, "content": content})
        msgs.append({"role": "user", "content": user_msg})

        # Build a single prompt: system block + conversation history + final user turn.
        # Claude CLI (core.claude_cli) handles auth via its own login — no API key needed.
        convo_parts = [system_prompt, "\n\n--- Conversation ---"]
        for m in msgs:
            convo_parts.append(f"{m['role'].upper()}: {m['content']}")
        convo_parts.append("ASSISTANT:")
        prompt = "\n\n".join(convo_parts)

        response_text = _claude_cli_call(prompt, model="claude-opus-4-7", timeout=120) or ""

        if not response_text:
            return jsonify({"error": "claude CLI returned no response"}), 502

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
  <script src="https://cdn.jsdelivr.net/npm/marked@12/marked.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Outfit:wght@400;500;600;700&family=DM+Mono:ital,wght@0,400;0,500;1,400&display=swap');
    :root {
      --bg: #0c0d0f;
      --bg-panel: #111316;
      --bg-elevated: #17191d;
      --bg-hover: #1c1f24;
      --bg-input: #0e1012;
      --border: rgba(255,255,255,0.07);
      --border-strong: rgba(255,255,255,0.13);
      --text: #dde1e7;
      --text-secondary: #8b929d;
      --text-muted: #50565f;
      --amber: #f0b429;
      --amber-dim: rgba(240,180,41,0.12);
      --cyan: #0acdff;
      --cyan-dim: rgba(10,205,255,0.12);
      --green: #00c896;
      --green-dim: rgba(0,200,150,0.12);
      --red: #ff4455;
      --red-dim: rgba(255,68,85,0.12);
      --blue: #4d9eff;
      --blue-dim: rgba(77,158,255,0.12);
      --yellow: #fbbf24;
      --yellow-dim: rgba(251,191,36,0.12);
      --pink: #f472b6;
      --font-sans: 'Outfit', system-ui, sans-serif;
      --font-mono: 'DM Mono', 'JetBrains Mono', monospace;
      --font-display: 'Syne', system-ui, sans-serif;
      --radius: 8px;
      --radius-sm: 5px;
    }
    * { box-sizing: border-box; }
    html { overflow-x: hidden; max-width: 100%; }
    body { overflow-x: hidden; }
    body { margin: 0; padding: 0; background: var(--bg); color: var(--text); font-family: var(--font-sans); font-size: 15px; line-height: 1.5; }
    .brand, h1, h2, h3 { font-family: var(--font-display); }
    .topbar {
      background: var(--bg-panel);
      border-bottom: 1px solid var(--border);
      padding: 0 20px;
      height: 48px;
      display: flex;
      align-items: center;
      gap: 2px;
      position: sticky;
      top: 0;
      z-index: 100;
      margin: 0;
      overflow: hidden;
      max-width: 100%;
    }
    .brand {
      font-weight: 700;
      color: var(--amber);
      font-size: 13px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-right: 16px;
      flex-shrink: 0;
    }
    .topbar a {
      color: var(--text-secondary);
      text-decoration: none;
      font-size: 13px;
      font-weight: 500;
      padding: 5px 12px;
      border-radius: var(--radius-sm);
      transition: color 0.15s, background 0.15s;
      white-space: nowrap;
    }
    .topbar a:hover { color: var(--text); background: var(--bg-hover); }
    .topbar a.active { color: var(--amber); background: var(--amber-dim); }
    .page-content { padding: 16px 20px; width: 100%; }
    /* Bootstrap overrides — apply our palette to the existing Bootstrap card structure */
    .card { background: var(--bg-panel) !important; border: 1px solid var(--border) !important; border-radius: var(--radius) !important; }
    .card-header { background: var(--bg-elevated) !important; font-weight: 600; border-bottom: 1px solid var(--border) !important; color: var(--text); }
    .badge-urgent { background: var(--red-dim); color: var(--red); }
    .badge-score { background: var(--blue-dim); color: var(--blue); }
    .pl-pos { color: var(--green); font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
    .pl-neg { color: var(--red); font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
    a { color: var(--cyan); }
    a:hover { color: var(--blue); }
    .scroll-pane { max-height:520px; overflow-y:auto; }
    .ticker { font-weight:600; color: var(--text); }
    .small-muted { color: var(--text-secondary); font-size: 0.85em; }
    table { color: var(--text); }
    .table-borderless td, .table-borderless th { color: var(--text); }
    /* Markdown in floating chat widget */
    .md-body p { margin: 0 0 0.5em; }
    .md-body p:last-child { margin-bottom: 0; }
    .md-body ul, .md-body ol { margin: 0.3em 0 0.5em 1.2em; padding: 0; }
    .md-body li { margin-bottom: 0.15em; }
    .md-body code { background: var(--bg-input); border: 1px solid var(--border-strong); padding: 1px 4px; border-radius: 3px; font-size: 0.85em; font-family: var(--font-mono); }
    .md-body pre { background: var(--bg-input); border: 1px solid var(--border-strong); padding: 8px 10px; border-radius: var(--radius-sm); overflow-x: auto; margin: 0.3em 0; }
    .md-body pre code { background: none; border: none; padding: 0; }
    .md-body strong { color: var(--text); }
    .md-body h1, .md-body h2, .md-body h3 { margin: 0.5em 0 0.25em; color: var(--text); font-size: 0.95em; }
    .md-body blockquote { border-left: 3px solid var(--border-strong); margin: 0.3em 0; padding-left: 8px; color: var(--text-secondary); }
    .md-body table { border-collapse: collapse; margin: 0.3em 0; width: 100%; font-size: 0.85em; }
    .md-body th, .md-body td { border: 1px solid var(--border-strong); padding: 3px 7px; text-align: left; }
    .md-body th { background: var(--bg-panel); }
    /* === Mobile-first responsive additions ============================== */
    .nav-hamburger {
      display: none; flex-direction: column; justify-content: space-between;
      width: 32px; height: 22px; background: none; border: none; cursor: pointer;
      padding: 0; margin-left: auto;
    }
    .nav-hamburger span {
      display: block; height: 2px; background: var(--text); border-radius: 2px;
      transition: all 0.2s;
    }
    .nav-drawer {
      position: fixed; top: 0; left: -280px; width: 280px; height: 100vh;
      background: var(--bg-panel); border-right: 1px solid #1e2028;
      z-index: 1000; transition: left 0.25s ease; overflow-y: auto; padding: 20px 0;
    }
    .nav-drawer.open { left: 0; }
    .nav-drawer-header {
      font-family: var(--font-display); font-weight: 700; color: var(--amber);
      font-size: 13px; letter-spacing: 0.1em; padding: 0 20px 20px;
      border-bottom: 1px solid #1e2028; margin-bottom: 8px;
    }
    .nav-drawer a {
      display: block; padding: 12px 20px; color: var(--text-secondary);
      text-decoration: none; font-size: 14px; transition: all 0.15s;
    }
    .nav-drawer a:hover, .nav-drawer a.active {
      color: var(--text); background: var(--bg-elevated);
    }
    .nav-overlay {
      display: none; position: fixed; inset: 0;
      background: rgba(0,0,0,0.6); z-index: 999;
    }
    .nav-overlay.open { display: block; }
    .bottom-nav {
      display: none; position: fixed; bottom: 0; left: 0; right: 0; height: 64px;
      background: var(--bg-panel); border-top: 1px solid #1e2028;
      grid-template-columns: repeat(5, 1fr); z-index: 200; align-items: stretch;
    }
    .bottom-tab {
      display: flex; flex-direction: column; align-items: center;
      justify-content: center; gap: 4px; color: var(--text-secondary);
      text-decoration: none; font-size: 10px; min-height: 44px; transition: color 0.15s;
    }
    .bottom-tab svg { width: 20px; height: 20px; }
    .bottom-tab.active, .bottom-tab:hover { color: var(--amber); }
    .table-scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; }
    .table-scroll table { min-width: 500px; }
    @media (max-width: 768px) {
      .topbar-nav { display: none; }
      .nav-hamburger { display: flex; }
      body { font-size: 14px; }
      button, .btn, a.btn, [role="button"] { min-height: 44px; min-width: 44px; }
    }
    @media (max-width: 480px) {
      body { padding-bottom: 72px; }
      .bottom-nav { display: grid; }
      .topbar { padding: 0 16px; }
      .card { min-height: auto !important; padding: 14px 16px; }
      .grid-2, .grid2 { grid-template-columns: 1fr !important; }
      .scroll-pane { max-height: 60vh !important; }
      table { font-size: 12px; }
      th, td { padding: 8px 10px; }
      #dichat-panel { width: calc(100vw - 24px) !important; right: 12px !important;
        height: 70vh !important; bottom: 80px !important; }
    }
  </style>
</head>
<body>
<nav class="topbar">
  <span class="brand">◈ TRADING STACK</span>
  <span class="topbar-nav" style="display:flex;align-items:center;gap:2px;">
    <a href="/">Command Center</a>
    <a href="/intern/" class="active">Digital Intern</a>
    <a href="/trader/">Paper Trader</a>
    <a href="/trader/backtests">Backtests</a>
    <a href="/backtests/compare">Compare</a>
    <a href="/journal">Journal</a>
    <a href="/ops/">Ops View</a>
    <a href="/intern/chat">Chat</a>
    <a href="/system/">System</a>
  </span>
  <button class="nav-hamburger" id="navToggle" aria-label="Menu">
    <span></span><span></span><span></span>
  </button>
</nav>
<div class="nav-drawer" id="navDrawer">
  <div class="nav-drawer-header">◈ TRADING STACK</div>
  <a href="/">Command Center</a>
  <a href="/intern/" class="active">Digital Intern</a>
  <a href="/trader/">Paper Trader</a>
  <a href="/trader/backtests">Backtests</a>
  <a href="/backtests/compare">Compare</a>
  <a href="/journal">Journal</a>
  <a href="/ops/">Ops View</a>
  <a href="/intern/chat">Chat</a>
  <a href="/system/">System</a>
</div>
<div class="nav-overlay" id="navOverlay"></div>
<div class="page-content">
<div class="container-fluid p-3">
  <div class="d-flex justify-content-between align-items-center mb-3">
    <h2 class="mb-0">Digital Intern</h2>
    <div class="small-muted" id="last-updated">loading…</div>
  </div>

  <div class="card mb-3" id="paper-trader-card">
    <div class="card-header d-flex justify-content-between align-items-center">
      <span>Live Paper Trader</span>
      <a href="/trader/" style="font-size:0.85em">View Full Trader →</a>
    </div>
    <div class="card-body p-2">
      <div id="paper-trader-summary" class="small-muted">loading…</div>
    </div>
  </div>

  <!-- News source edge — which of MY collectors actually precede the move? (new 2026-05-16, agent 4) -->
  <div class="card mb-3" id="se-card">
    <div class="card-header d-flex justify-content-between align-items-center">
      <span>News source edge <span class="small-muted">— which collectors' scored headlines actually precede the move (vs SPY)</span></span>
      <span id="se-state" class="badge bg-secondary">—</span>
    </div>
    <div class="card-body p-2">
      <div id="se-headline" class="small-muted mb-2">loading…</div>
      <div class="table-responsive">
        <table class="table table-sm table-dark mb-1" style="font-size:0.82em;">
          <thead><tr><th>collector</th><th>abn% @ref</th><th>hit</th><th>resolved</th><th>verdict</th></tr></thead>
          <tbody id="se-rows"><tr><td colspan="5" class="small-muted">—</td></tr></tbody>
        </table>
      </div>
      <div id="se-meta" class="small-muted">—</div>
    </div>
  </div>

  <div class="row g-3">
    <div class="col-12 col-lg-4">
      <div class="card mb-3">
        <div class="card-header d-flex justify-content-between align-items-center">
          Portfolio P&amp;L
          <button class="btn btn-sm btn-outline-secondary py-0 px-2" style="font-size:11px" onclick="togglePortfolioEdit()">✎ Edit</button>
        </div>
        <div class="card-body p-2">
          <div id="pnl-summary" class="mb-2 small-muted">loading…</div>
          <div class="table-responsive scroll-pane table-scroll">
            <table class="table table-sm table-borderless mb-0">
              <thead><tr>
                <th>TICKER</th><th class="text-end">QTY</th>
                <th class="text-end">PRICE</th><th class="text-end">P/L</th>
              </tr></thead>
              <tbody id="pnl-rows"></tbody>
            </table>
          </div>
          <!-- Inline portfolio editor -->
          <div id="portfolio-editor" style="display:none;margin-top:10px;">
            <div style="font-size:11px;color:var(--text-secondary);margin-bottom:6px;">Edit positions — changes save to config and refresh live P&L</div>
            <table class="table table-sm table-borderless mb-1" style="font-size:12px;">
              <thead><tr style="color:var(--text-secondary);">
                <th>TICKER</th><th>TYPE</th><th>QTY</th><th>AVG COST</th><th></th>
              </tr></thead>
              <tbody id="edit-pos-rows"></tbody>
            </table>
            <button class="btn btn-sm btn-outline-success py-0 px-2 me-1" style="font-size:11px" onclick="addEditRow()">+ Add</button>
            <button class="btn btn-sm btn-primary py-0 px-2" style="font-size:11px" onclick="savePortfolioConfig()">Save</button>
            <span id="edit-save-status" style="font-size:11px;color:var(--text-secondary);margin-left:8px;"></span>
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

      <!-- Collector health table -->
      <div class="card mb-3" id="collectors-card">
        <div class="card-header d-flex justify-content-between align-items-center" id="collectors">
          <span>Collectors <span class="small-muted ms-1">— live source pulse</span></span>
          <button class="btn btn-sm btn-outline-secondary py-0 px-2" style="font-size:11px;" onclick="refreshCollectors()">↻ refresh</button>
        </div>
        <div class="card-body p-2">
          <div class="small-muted mb-1" id="collectors-meta">loading…</div>
          <div class="table-responsive scroll-pane table-scroll" style="max-height:280px;">
            <table class="table table-sm table-borderless mb-0">
              <thead><tr>
                <th>SOURCE</th>
                <th class="text-end">1h</th>
                <th class="text-end">24h</th>
                <th class="text-end">STATUS</th>
              </tr></thead>
              <tbody id="collectors-rows"></tbody>
            </table>
          </div>
        </div>
      </div>

      <!-- ML model status -->
      <div class="card mb-3" id="ml-card">
        <div class="card-header">ML Model Status</div>
        <div class="card-body p-2">
          <div class="row g-2" style="font-size:13px;">
            <div class="col-6 col-md-3">
              <div class="small-muted" style="font-size:11px;text-transform:uppercase;letter-spacing:0.05em;">last trained</div>
              <div id="ml-last-trained" class="ticker">—</div>
            </div>
            <div class="col-6 col-md-3">
              <div class="small-muted" style="font-size:11px;text-transform:uppercase;letter-spacing:0.05em;">training set</div>
              <div id="ml-set" class="ticker">—</div>
            </div>
            <div class="col-6 col-md-3">
              <div class="small-muted" style="font-size:11px;text-transform:uppercase;letter-spacing:0.05em;">scored 24h</div>
              <div id="ml-preds" class="ticker">—</div>
            </div>
            <div class="col-6 col-md-3">
              <div class="small-muted" style="font-size:11px;text-transform:uppercase;letter-spacing:0.05em;">val loss</div>
              <div id="ml-val" class="ticker">—</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Article volume chart (last 24h, hourly) -->
      <div class="card mb-3" id="volume-card">
        <div class="card-header">Article volume — last 24h</div>
        <div class="card-body p-2">
          <div style="position:relative;height:180px;"><canvas id="volume-chart"></canvas></div>
        </div>
      </div>

      <!-- Backtest isolation invariant badge -->
      <div class="card mb-3" id="invariants-card">
        <div class="card-header">System invariants</div>
        <div class="card-body p-2">
          <div id="iso-badge" class="small-muted">checking…</div>
          <div class="small-muted mt-1" id="iso-detail" style="font-size:11px;"></div>
        </div>
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
    const r = await fetch("/trader/api/portfolio");
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

// ── News source edge (cross-fetched from the trader; matures with history) ──
// Mirrors refreshPaperTrader's exact /trader/ cross-fetch + r.ok degradation
// (don't re-derive it). A stale trader process predates /api/source-edge and
// 404s — surface that honestly instead of a blank card.
async function refreshSourceEdge() {
  const stEl = document.getElementById("se-state");
  const hlEl = document.getElementById("se-headline");
  const rowsEl = document.getElementById("se-rows");
  try {
    const r = await fetch("/trader/api/source-edge");
    if (r.status === 404) {
      stEl.textContent = "stale"; stEl.className = "badge bg-warning text-dark";
      hlEl.textContent = "restart paper-trader to apply (process predates /api/source-edge)";
      return;
    }
    if (!r.ok) { stEl.textContent = "n/a"; hlEl.textContent = "trader unavailable (HTTP " + r.status + ")"; return; }
    const j = await r.json();
    const v = j.verdict || "—";
    const cls = v === "EDGE_FOUND" ? "bg-success"
              : v === "NO_EDGE" ? "bg-danger"
              : v === "INSUFFICIENT_DATA" ? "bg-warning text-dark"
              : "bg-secondary";
    stEl.textContent = v.replace(/_/g, " "); stEl.className = "badge " + cls;
    hlEl.textContent = j.verdict_reason || "";
    const ref = String(j.reference_horizon || 3);
    const vcls = { EXPLOITABLE: "pl-pos", NEGATIVE: "pl-neg" };
    const rows = (j.sources || []).slice(0, 10).map(s => {
      const h = (s.horizons || {})[ref] || {};
      const abn = h.mean_abnormal_pct, hit = h.abnormal_hit_rate;
      return "<tr><td>" + s.source + "</td>"
        + "<td class='" + plClass(abn) + "'>" + (abn != null ? (abn>=0?"+":"") + fmt(abn) + "%" : "—") + "</td>"
        + "<td>" + (hit != null ? fmt(hit) + "%" : "—") + "</td>"
        + "<td>" + (s.n_resolved != null ? s.n_resolved : "—") + "</td>"
        + "<td class='" + (vcls[s.verdict] || "") + "'>" + (s.verdict || "—") + "</td></tr>";
    });
    rowsEl.innerHTML = rows.length ? rows.join("")
      : "<tr><td colspan='5' class='small-muted'>no collector resolved a watchlist move yet</td></tr>";
    document.getElementById("se-meta").textContent =
      "ref " + (j.reference_horizon != null ? j.reference_horizon + "d" : "—")
      + " · " + (j.n_resolved != null ? j.n_resolved : "—") + " resolved / "
      + (j.n_scored != null ? j.n_scored : "—") + " scored"
      + (j.spy_adjusted ? " · SPY-adjusted" : " · raw only")
      + (j.lookback_days != null ? " · " + j.lookback_days + "d lookback" : "");
  } catch (e) {
    stEl.textContent = "n/a"; hlEl.textContent = "trader unreachable";
  }
}

refresh();
refreshPaperTrader();
refreshSourceEdge();
refreshCollectors();
refreshMlStatus();
refreshVolumeChart();
refreshInvariants();
setInterval(refresh, 15000);
setInterval(refreshPaperTrader, 15000);
setInterval(refreshSourceEdge, 300000);
setInterval(refreshCollectors, 60000);
setInterval(refreshMlStatus, 120000);
setInterval(refreshVolumeChart, 300000);
setInterval(refreshInvariants, 60000);

// ── Collector health ─────────────────────────────────────────────────────────
async function refreshCollectors() {
  const meta = document.getElementById("collectors-meta");
  const tbody = document.getElementById("collectors-rows");
  const d = await getJSON("/api/collector-health");
  if (!d || d.error) {
    if (meta) meta.textContent = (d && d.error) ? d.error : "loading…";
    if (tbody) tbody.innerHTML = '<tr><td colspan="4" class="small-muted">—</td></tr>';
    return;
  }
  const sources = d.sources || [];
  const active = sources.filter(s => s.status === "active").length;
  const slow = sources.filter(s => s.status === "slow").length;
  const stale = sources.filter(s => s.status === "stale").length;
  if (meta) {
    meta.innerHTML =
      `<span style="color:var(--green);">●</span> ${active} active · ` +
      `<span style="color:var(--yellow);">●</span> ${slow} slow · ` +
      `<span style="color:var(--red);">●</span> ${stale} stale ` +
      `<span class="small-muted">(${sources.length} sources)</span>`;
  }
  if (!sources.length) {
    tbody.innerHTML = '<tr><td colspan="4" class="small-muted">no sources reporting</td></tr>';
    return;
  }
  tbody.innerHTML = sources.map(s => {
    const dot = s.status === "active" ? '<span style="color:var(--green);">●</span> Active'
              : s.status === "slow"   ? '<span style="color:var(--yellow);">●</span> Slow'
              : s.status === "stale"  ? '<span style="color:var(--red);">●</span> Stale'
              :                          '<span class="small-muted">●</span> Idle';
    return `<tr>
      <td class="ticker">${(s.source||'?').replace(/</g,'&lt;')}</td>
      <td class="text-end">${s.articles_1h}</td>
      <td class="text-end">${s.articles_24h}</td>
      <td class="text-end" style="font-size:12px;">${dot}</td>
    </tr>`;
  }).join("");
}

// ── ML model status ──────────────────────────────────────────────────────────
async function refreshMlStatus() {
  const d = await getJSON("/api/ml-status");
  const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
  if (!d || d.error) { set("ml-last-trained","—"); set("ml-set","—"); set("ml-preds","—"); set("ml-val","—"); return; }
  let last = "—";
  if (d.last_trained) {
    try {
      const t = new Date(d.last_trained);
      const ago = Math.max(0, (Date.now() - t.getTime())/60000);
      last = ago < 60 ? Math.round(ago) + "m ago"
           : ago < 1440 ? (ago/60).toFixed(1) + "h ago"
           : (ago/1440).toFixed(1) + "d ago";
    } catch (e) { last = d.last_trained; }
  }
  set("ml-last-trained", last);
  set("ml-set", d.training_set_size != null ? Number(d.training_set_size).toLocaleString() : "—");
  set("ml-preds", d.predictions_24h != null ? Number(d.predictions_24h).toLocaleString() : "—");
  set("ml-val", d.val_loss != null ? Number(d.val_loss).toFixed(4) : "—");
}

// ── Article volume bar chart ─────────────────────────────────────────────────
let _volumeChart = null;
async function refreshVolumeChart() {
  const d = await getJSON("/api/volume-history");
  const canvas = document.getElementById("volume-chart");
  if (!canvas || typeof Chart === "undefined") return;
  const rows = (d && d.hours) || [];
  const labels = rows.map(r => (r.hour || "").slice(11, 16));
  const counts = rows.map(r => r.count || 0);
  if (_volumeChart) _volumeChart.destroy();
  _volumeChart = new Chart(canvas, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "articles/h",
        data: counts,
        backgroundColor: "rgba(10,205,255,0.55)",
        borderColor: "#0acdff",
        borderWidth: 1,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#8b929d", maxRotation: 0, autoSkip: true, maxTicksLimit: 8 }, grid: { display: false } },
        y: { ticks: { color: "#8b929d", precision: 0 }, grid: { color: "rgba(255,255,255,0.04)" } },
      },
    },
  });
}

// ── Backtest isolation invariant ─────────────────────────────────────────────
async function refreshInvariants() {
  const badge = document.getElementById("iso-badge");
  const detail = document.getElementById("iso-detail");
  if (!badge) return;
  const d = await getJSON("/api/invariants");
  if (!d || d.error) {
    badge.innerHTML = '<span class="badge badge-score">Backtest isolation: UNKNOWN</span>';
    if (detail) detail.textContent = (d && d.error) ? d.error : "";
    return;
  }
  if (d.backtest_isolation === "active") {
    badge.innerHTML = '<span class="badge" style="background:rgba(0,200,150,0.18);color:var(--green);">Backtest isolation: ACTIVE ✓</span>';
    if (detail) {
      const n = d.backtest_rows_total || 0;
      detail.textContent = `${n.toLocaleString()} synthetic rows isolated from live alerts.`;
    }
  } else {
    badge.innerHTML = '<span class="badge" style="background:rgba(255,68,85,0.20);color:var(--red);">BREACH DETECTED ✗</span>';
    if (detail) detail.textContent = `${d.breach_count} backtest row(s) alerted as urgent.`;
  }
}

// ── Portfolio config editor ────────────────────────────────────────────────
let _editConfig = null;
async function togglePortfolioEdit() {
  const el = document.getElementById("portfolio-editor");
  if (el.style.display !== "none") { el.style.display = "none"; return; }
  const cfg = await getJSON("/api/portfolio/config");
  _editConfig = cfg;
  renderEditRows(cfg.positions || []);
  el.style.display = "block";
}
function renderEditRows(positions) {
  const tbody = document.getElementById("edit-pos-rows");
  tbody.innerHTML = "";
  positions.forEach((p, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><input class="form-control form-control-sm p-1" style="font-size:11px;background:var(--bg-input);color:var(--text);border-color:var(--border-strong);width:70px" value="${p.ticker||""}" data-i="${i}" data-f="ticker"></td>
      <td><select class="form-select form-select-sm p-1" style="font-size:11px;background:var(--bg-input);color:var(--text);border-color:var(--border-strong);width:90px" data-i="${i}" data-f="type">
        ${["stock","etf_leveraged","etf","option"].map(t=>`<option${p.type===t?" selected":""}>${t}</option>`).join("")}
      </select></td>
      <td><input class="form-control form-control-sm p-1" style="font-size:11px;background:var(--bg-input);color:var(--text);border-color:var(--border-strong);width:70px" value="${p.qty??""}" data-i="${i}" data-f="qty"></td>
      <td><input class="form-control form-control-sm p-1" style="font-size:11px;background:var(--bg-input);color:var(--text);border-color:var(--border-strong);width:80px" value="${p.avg_cost??""}" data-i="${i}" data-f="avg_cost"></td>
      <td><button class="btn btn-sm btn-outline-danger py-0 px-1" style="font-size:10px" onclick="removeEditRow(${i})">✕</button></td>`;
    tbody.appendChild(tr);
  });
}
function addEditRow() {
  if (!_editConfig) return;
  (_editConfig.positions = _editConfig.positions || []).push({ticker:"",type:"stock",qty:0,avg_cost:0});
  renderEditRows(_editConfig.positions);
}
function removeEditRow(i) {
  _editConfig.positions.splice(i, 1);
  renderEditRows(_editConfig.positions);
}
async function savePortfolioConfig() {
  // Collect all field edits from DOM
  document.querySelectorAll("#edit-pos-rows input, #edit-pos-rows select").forEach(el => {
    const i = parseInt(el.dataset.i), f = el.dataset.f;
    let v = el.value.trim();
    if (f === "qty" || f === "avg_cost") v = parseFloat(v) || 0;
    _editConfig.positions[i][f] = v;
  });
  const status = document.getElementById("edit-save-status");
  status.textContent = "saving…";
  try {
    const r = await fetch("/api/portfolio/config", {method:"PUT", headers:{"Content-Type":"application/json"}, body: JSON.stringify(_editConfig)});
    const d = await r.json();
    status.textContent = d.ok ? "✓ saved" : "error: " + d.error;
    if (d.ok) setTimeout(() => { refresh(); status.textContent = ""; }, 800);
  } catch(e) { status.textContent = "error: " + e; }
}
</script>

<!-- Floating chat widget -->
<button id="dichat-btn" aria-label="Open chat"
  style="position:fixed;bottom:20px;right:20px;width:56px;height:56px;border-radius:50%;background:var(--amber-dim);color:var(--amber);border:1px solid rgba(240,180,41,0.3);font-size:24px;cursor:pointer;box-shadow:0 4px 14px rgba(0,0,0,0.5);z-index:9998">✦</button>
<div id="dichat-panel"
  style="display:none;position:fixed;bottom:88px;right:20px;width:360px;height:480px;background:var(--bg-panel);color:var(--text);border:1px solid var(--border);border-radius:var(--radius);box-shadow:0 8px 24px rgba(0,0,0,0.5);z-index:9999;flex-direction:column;font-family:var(--font-sans)">
  <div style="padding:12px 14px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center">
    <span style="font-weight:600;color:var(--text)">Market Intel</span>
    <a href="/intern/chat" style="color:var(--text-secondary);font-size:0.8em;text-decoration:none;margin-left:auto;margin-right:10px">full ↗</a>
    <button id="dichat-close" style="background:none;border:none;color:var(--text-secondary);cursor:pointer;font-size:20px;line-height:1;padding:0 4px">×</button>
  </div>
  <div id="dichat-history" style="flex:1;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:10px;font-size:14px;line-height:1.45"></div>
  <form id="dichat-form" style="display:flex;gap:6px;padding:10px;border-top:1px solid var(--border)">
    <input id="dichat-input" type="text" autocomplete="off" placeholder="Ask about markets…"
      style="flex:1;background:var(--bg-input);border:1px solid var(--border-strong);color:var(--text);padding:8px 10px;border-radius:var(--radius-sm);font-size:14px;font-family:inherit">
    <button type="submit" id="dichat-send"
      style="background:var(--amber-dim);color:var(--amber);border:1px solid rgba(240,180,41,0.3);padding:8px 14px;border-radius:var(--radius-sm);cursor:pointer;font-weight:600">Send</button>
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
  function renderMd(text) {
    if (typeof marked !== 'undefined') {
      try { return marked.parse(text, {breaks: true, gfm: true}); } catch(e) {}
    }
    return text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\\n/g,'<br>');
  }
  function bubble(role, text){
    const d = document.createElement('div');
    const bg = role==='user' ? 'var(--blue-dim)' : role==='error' ? 'var(--red-dim)' : 'var(--bg-elevated)';
    const fg = role==='user' ? 'var(--blue)' : role==='error' ? 'var(--red)' : 'var(--text)';
    const align = role==='user' ? 'flex-end' : 'flex-start';
    d.style.cssText = 'background:'+bg+';color:'+fg+';padding:8px 12px;border-radius:10px;max-width:85%;align-self:'+align+';word-wrap:break-word;overflow-wrap:anywhere';
    if (role === 'user') {
      d.textContent = text;
    } else {
      d.classList.add('md-body');
      d.innerHTML = renderMd(text);
    }
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

<nav class="bottom-nav" id="bottomNav">
  <a href="/" class="bottom-tab">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9.5L12 3l9 6.5V20a1 1 0 0 1-1 1h-5v-7H9v7H4a1 1 0 0 1-1-1V9.5z"/></svg>
    <span>Home</span>
  </a>
  <a href="/intern/" class="bottom-tab">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M7 14l4-4 3 3 5-6"/></svg>
    <span>Intern</span>
  </a>
  <a href="/trader/" class="bottom-tab">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 17l6-6 4 4 8-8"/><path d="M17 7h4v4"/></svg>
    <span>Trader</span>
  </a>
  <a href="/intern/chat" class="bottom-tab">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 11.5a8.38 8.38 0 0 1-8.5 8.5 8.5 8.5 0 0 1-3.8-.9L3 21l1.9-5.7A8.38 8.38 0 0 1 4 11.5 8.5 8.5 0 0 1 12.5 3 8.38 8.38 0 0 1 21 11.5z"/></svg>
    <span>Chat</span>
  </a>
  <a href="/trader/backtests" class="bottom-tab">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v5h5"/><path d="M3.05 13A9 9 0 1 0 6 5.3L3 8"/><path d="M12 7v5l4 2"/></svg>
    <span>Backtests</span>
  </a>
</nav>
<script>
(function(){
  const navToggle = document.getElementById('navToggle');
  const navDrawer = document.getElementById('navDrawer');
  const navOverlay = document.getElementById('navOverlay');
  if (navToggle) {
    navToggle.addEventListener('click', () => {
      navDrawer.classList.toggle('open');
      navOverlay.classList.toggle('open');
    });
    navOverlay.addEventListener('click', () => {
      navDrawer.classList.remove('open');
      navOverlay.classList.remove('open');
    });
  }
  document.querySelectorAll('.bottom-tab').forEach(tab => {
    if (tab.getAttribute('href') === window.location.pathname) {
      tab.classList.add('active');
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
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Outfit:wght@400;500;600;700&family=DM+Mono:ital,wght@0,400;0,500;1,400&display=swap');
    :root {
      --bg: #0c0d0f;
      --bg-panel: #111316;
      --bg-elevated: #17191d;
      --bg-hover: #1c1f24;
      --bg-input: #0e1012;
      --border: rgba(255,255,255,0.07);
      --border-strong: rgba(255,255,255,0.13);
      --text: #dde1e7;
      --text-secondary: #8b929d;
      --text-muted: #50565f;
      --amber: #f0b429;
      --amber-dim: rgba(240,180,41,0.12);
      --cyan: #0acdff;
      --cyan-dim: rgba(10,205,255,0.12);
      --green: #00c896;
      --green-dim: rgba(0,200,150,0.12);
      --red: #ff4455;
      --red-dim: rgba(255,68,85,0.12);
      --blue: #4d9eff;
      --blue-dim: rgba(77,158,255,0.12);
      --yellow: #fbbf24;
      --yellow-dim: rgba(251,191,36,0.12);
      --pink: #f472b6;
      --font-sans: 'Outfit', system-ui, sans-serif;
      --font-mono: 'DM Mono', 'JetBrains Mono', monospace;
      --font-display: 'Syne', system-ui, sans-serif;
      --radius: 8px;
      --radius-sm: 5px;
    }
    * { box-sizing: border-box; }
    html { overflow-x: hidden; max-width: 100%; }
    body { overflow-x: hidden; }
    html, body { margin: 0; padding: 0; height: 100%; }
    body {
      background: var(--bg); color: var(--text);
      font-family: var(--font-sans);
      font-size: 15px;
      line-height: 1.5;
      display: flex; flex-direction: column; height: 100vh;
    }
    .brand, h1, h2, h3 { font-family: var(--font-display); }
    .topbar {
      background: var(--bg-panel);
      border-bottom: 1px solid var(--border);
      padding: 0 20px;
      height: 48px;
      display: flex;
      align-items: center;
      gap: 2px;
      position: sticky;
      top: 0;
      z-index: 100;
      margin: 0;
      flex-shrink: 0;
      overflow: hidden;
      max-width: 100%;
    }
    .brand {
      font-weight: 700;
      color: var(--amber);
      font-size: 13px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-right: 16px;
      flex-shrink: 0;
    }
    .topbar a {
      color: var(--text-secondary);
      text-decoration: none;
      font-size: 13px;
      font-weight: 500;
      padding: 5px 12px;
      border-radius: var(--radius-sm);
      transition: color 0.15s, background 0.15s;
      white-space: nowrap;
    }
    .topbar a:hover { color: var(--text); background: var(--bg-hover); }
    .topbar a.active { color: var(--amber); background: var(--amber-dim); }
    .page-content {
      flex: 1;
      display: flex;
      flex-direction: column;
      min-height: 0;
      width: 100%;
    }
    header.page {
      padding: 18px 24px 12px; border-bottom: 1px solid var(--border);
      background: var(--bg);
    }
    header.page h1 { margin: 0; font-size: 22px; font-weight: 600; color: var(--text); }
    header.page .sub { color: var(--text-secondary); font-size: 13px; margin-top: 4px; }
    .chat-wrap {
      flex: 1; overflow-y: auto; padding: 20px 24px;
      display: flex; flex-direction: column; gap: 14px;
    }
    .msg { max-width: 760px; padding: 12px 16px; border-radius: var(--radius); word-wrap: break-word; overflow-wrap: anywhere; line-height: 1.5; font-size: 14px; }
    .msg.user { white-space: pre-wrap; }
    .msg.user { align-self: flex-end; background: var(--blue-dim); color: var(--blue); border: 1px solid rgba(77,158,255,0.3); }
    .msg.assistant { align-self: flex-start; background: var(--bg-panel); border: 1px solid var(--border); color: var(--text); }
    .msg.error { align-self: flex-start; background: var(--red-dim); border: 1px solid var(--red); color: var(--red); }
    .sources { align-self: flex-start; max-width: 760px; display: flex; flex-wrap: wrap; gap: 6px; margin-top: -6px; }
    .chip {
      background: var(--bg-elevated); border: 1px solid var(--border); color: var(--text-secondary);
      font-size: 11px; padding: 3px 8px; border-radius: 10px;
    }
    .suggestions { display: flex; flex-wrap: wrap; gap: 8px; padding: 14px 24px 4px; }
    .suggestion {
      background: var(--bg-elevated); border: 1px solid var(--border-strong); color: var(--cyan);
      padding: 6px 14px; border-radius: 18px; cursor: pointer; font-size: 13px;
      font-family: var(--font-sans);
    }
    .suggestion:hover { background: var(--bg-hover); color: var(--text); }
    .input-bar {
      display: flex; gap: 10px; padding: 14px 24px; border-top: 1px solid var(--border); background: var(--bg-panel);
    }
    input.msg-input {
      flex: 1; background: var(--bg-input); border: 1px solid var(--border-strong); color: var(--text);
      padding: 10px 14px; border-radius: var(--radius-sm); font-size: 14px;
      font-family: var(--font-sans);
    }
    input.msg-input:focus { outline: none; border-color: var(--amber); }
    button.send {
      background: var(--amber-dim); color: var(--amber); border: 1px solid rgba(240,180,41,0.3); padding: 10px 20px;
      border-radius: var(--radius-sm); font-size: 13px; cursor: pointer; font-weight: 600;
      font-family: var(--font-sans);
      transition: background 0.15s;
    }
    button.send:hover { background: var(--bg-hover); }
    button.send:disabled { background: var(--bg-elevated); color: var(--text-muted); border-color: var(--border); cursor: not-allowed; }
    .spinner {
      display: inline-block; width: 10px; height: 10px;
      border: 2px solid var(--border-strong); border-top-color: var(--cyan);
      border-radius: 50%; animation: spin 0.8s linear infinite; vertical-align: middle; margin-right: 8px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.5;} }
    .typing { color: var(--text-secondary); font-style: italic; }
    /* Markdown rendered content */
    .md-body p { margin: 0 0 0.6em; }
    .md-body p:last-child { margin-bottom: 0; }
    .md-body ul, .md-body ol { margin: 0.3em 0 0.6em 1.2em; padding: 0; }
    .md-body li { margin-bottom: 0.2em; }
    .md-body code { background: var(--bg-input); border: 1px solid var(--border-strong); padding: 1px 5px; border-radius: 4px; font-size: 0.87em; font-family: var(--font-mono); }
    .md-body pre { background: var(--bg-input); border: 1px solid var(--border-strong); padding: 10px 14px; border-radius: var(--radius-sm); overflow-x: auto; margin: 0.4em 0; }
    .md-body pre code { background: none; border: none; padding: 0; }
    .md-body strong { color: var(--text); }
    .md-body h1, .md-body h2, .md-body h3 { margin: 0.6em 0 0.3em; color: var(--text); font-size: 1em; }
    .md-body blockquote { border-left: 3px solid var(--border-strong); margin: 0.4em 0; padding-left: 10px; color: var(--text-secondary); }
    .md-body table { border-collapse: collapse; margin: 0.4em 0; width: 100%; }
    .md-body th, .md-body td { border: 1px solid var(--border-strong); padding: 4px 8px; text-align: left; }
    .md-body th { background: var(--bg-panel); }
    /* === Mobile-first responsive additions ============================== */
    .nav-hamburger {
      display: none; flex-direction: column; justify-content: space-between;
      width: 32px; height: 22px; background: none; border: none; cursor: pointer;
      padding: 0; margin-left: auto;
    }
    .nav-hamburger span {
      display: block; height: 2px; background: var(--text); border-radius: 2px;
      transition: all 0.2s;
    }
    .nav-drawer {
      position: fixed; top: 0; left: -280px; width: 280px; height: 100vh;
      background: var(--bg-panel); border-right: 1px solid #1e2028;
      z-index: 1000; transition: left 0.25s ease; overflow-y: auto; padding: 20px 0;
    }
    .nav-drawer.open { left: 0; }
    .nav-drawer-header {
      font-family: var(--font-display); font-weight: 700; color: var(--amber);
      font-size: 13px; letter-spacing: 0.1em; padding: 0 20px 20px;
      border-bottom: 1px solid #1e2028; margin-bottom: 8px;
    }
    .nav-drawer a {
      display: block; padding: 12px 20px; color: var(--text-secondary);
      text-decoration: none; font-size: 14px; transition: all 0.15s;
    }
    .nav-drawer a:hover, .nav-drawer a.active {
      color: var(--text); background: var(--bg-elevated);
    }
    .nav-overlay {
      display: none; position: fixed; inset: 0;
      background: rgba(0,0,0,0.6); z-index: 999;
    }
    .nav-overlay.open { display: block; }
    .bottom-nav {
      display: none; position: fixed; bottom: 0; left: 0; right: 0; height: 64px;
      background: var(--bg-panel); border-top: 1px solid #1e2028;
      grid-template-columns: repeat(5, 1fr); z-index: 200; align-items: stretch;
    }
    .bottom-tab {
      display: flex; flex-direction: column; align-items: center;
      justify-content: center; gap: 4px; color: var(--text-secondary);
      text-decoration: none; font-size: 10px; min-height: 44px; transition: color 0.15s;
    }
    .bottom-tab svg { width: 20px; height: 20px; }
    .bottom-tab.active, .bottom-tab:hover { color: var(--amber); }
    .table-scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; }
    .table-scroll table { min-width: 500px; }
    .input-bar { position: sticky; bottom: 0; }
    @media (max-width: 768px) {
      .topbar-nav { display: none; }
      .nav-hamburger { display: flex; }
      body { font-size: 14px; }
      button, .btn, a.btn, [role="button"] { min-height: 44px; min-width: 44px; }
    }
    @media (max-width: 480px) {
      .bottom-nav { display: grid; }
      .topbar { padding: 0 16px; }
      .msg { max-width: 90%; }
      .chat-wrap { padding: 14px 16px 88px; }
      .input-bar { padding: 12px 16px; margin-bottom: 64px; }
      .suggestions { padding: 12px 16px 4px; }
      button.send { min-height: 44px; min-width: 44px; }
      table { font-size: 12px; }
      th, td { padding: 8px 10px; }
    }
  </style>
  <script src="https://cdn.jsdelivr.net/npm/marked@12/marked.min.js" async></script>
</head>
<body>
<nav class="topbar">
  <span class="brand">◈ TRADING STACK</span>
  <span class="topbar-nav" style="display:flex;align-items:center;gap:2px;">
    <a href="/">Command Center</a>
    <a href="/intern/">Digital Intern</a>
    <a href="/trader/">Paper Trader</a>
    <a href="/trader/backtests">Backtests</a>
    <a href="/backtests/compare">Compare</a>
    <a href="/journal">Journal</a>
    <a href="/ops/">Ops View</a>
    <a href="/intern/chat" class="active">Chat</a>
    <a href="/system/">System</a>
  </span>
  <button class="nav-hamburger" id="navToggle" aria-label="Menu">
    <span></span><span></span><span></span>
  </button>
</nav>
<div class="nav-drawer" id="navDrawer">
  <div class="nav-drawer-header">◈ TRADING STACK</div>
  <a href="/">Command Center</a>
  <a href="/intern/">Digital Intern</a>
  <a href="/trader/">Paper Trader</a>
  <a href="/trader/backtests">Backtests</a>
  <a href="/backtests/compare">Compare</a>
  <a href="/journal">Journal</a>
  <a href="/ops/">Ops View</a>
  <a href="/intern/chat" class="active">Chat</a>
  <a href="/system/">System</a>
</div>
<div class="nav-overlay" id="navOverlay"></div>
<div class="page-content">
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
<div class="input-bar">
  <input class="msg-input" id="input" type="text" placeholder="Ask about markets, news, or your portfolio…" autocomplete="off" autofocus>
  <button class="send" id="send" type="button" onclick="sendMsg()">Send</button>
</div>
</div>
<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
const msgs = [];

function sendMsg() { ask(input.value.trim()); }
input.addEventListener('keydown', function(e) { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMsg(); } });

function scrollDown() {
  requestAnimationFrame(() => { chat.scrollTop = chat.scrollHeight; });
}

function renderMd(text) {
  if (typeof marked !== 'undefined') {
    try { return marked.parse(text, {breaks: true, gfm: true}); } catch(e) {}
  }
  return text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\\n/g,'<br>');
}
function addMsg(role, content, sources) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  if (role === 'user') {
    div.textContent = content;
  } else {
    div.classList.add('md-body');
    div.innerHTML = renderMd(content);
  }
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
  if (!message || sendBtn.disabled) return;
  addMsg('user', message);
  msgs.push({role: 'user', content: message});
  input.value = '';
  sendBtn.disabled = true;
  const loader = addLoader();
  try {
    const r = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: message, history: msgs.slice(0, -1)}),
    });
    const j = await r.json();
    loader.remove();
    if (!r.ok || j.error) {
      addMsg('error', 'Error: ' + (j.error || ('HTTP ' + r.status)));
    } else {
      addMsg('assistant', j.response, j.sources || []);
      msgs.push({role: 'assistant', content: j.response});
    }
  } catch (e) {
    loader.remove();
    addMsg('error', 'Network error: ' + e.message);
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

document.getElementById('suggestions').addEventListener('click', function(e) {
  var b = e.target.closest('.suggestion');
  if (b) ask(b.dataset.q);
});
</script>

<nav class="bottom-nav" id="bottomNav">
  <a href="/" class="bottom-tab">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9.5L12 3l9 6.5V20a1 1 0 0 1-1 1h-5v-7H9v7H4a1 1 0 0 1-1-1V9.5z"/></svg>
    <span>Home</span>
  </a>
  <a href="/intern/" class="bottom-tab">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M7 14l4-4 3 3 5-6"/></svg>
    <span>Intern</span>
  </a>
  <a href="/trader/" class="bottom-tab">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 17l6-6 4 4 8-8"/><path d="M17 7h4v4"/></svg>
    <span>Trader</span>
  </a>
  <a href="/intern/chat" class="bottom-tab">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 11.5a8.38 8.38 0 0 1-8.5 8.5 8.5 8.5 0 0 1-3.8-.9L3 21l1.9-5.7A8.38 8.38 0 0 1 4 11.5 8.5 8.5 0 0 1 12.5 3 8.38 8.38 0 0 1 21 11.5z"/></svg>
    <span>Chat</span>
  </a>
  <a href="/trader/backtests" class="bottom-tab">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v5h5"/><path d="M3.05 13A9 9 0 1 0 6 5.3L3 8"/><path d="M12 7v5l4 2"/></svg>
    <span>Backtests</span>
  </a>
</nav>
<script>
(function(){
  const navToggle = document.getElementById('navToggle');
  const navDrawer = document.getElementById('navDrawer');
  const navOverlay = document.getElementById('navOverlay');
  if (navToggle) {
    navToggle.addEventListener('click', () => {
      navDrawer.classList.toggle('open');
      navOverlay.classList.toggle('open');
    });
    navOverlay.addEventListener('click', () => {
      navDrawer.classList.remove('open');
      navOverlay.classList.remove('open');
    });
  }
  document.querySelectorAll('.bottom-tab').forEach(tab => {
    if (tab.getAttribute('href') === window.location.pathname) {
      tab.classList.add('active');
    }
  });
})();
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

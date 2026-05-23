"""Flask dashboard at :8090 — portfolio chart, trade log, positions, decisions, backtests."""
from __future__ import annotations

import gzip
import json
import re
import sqlite3
import subprocess
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request

from .store import INITIAL_CASH, get_store
from .backtest import BACKTEST_DB  # re-exported so tests can monkeypatch dash.BACKTEST_DB

app = Flask(__name__)


_MODEL_DISPLAY_NAMES = {
    "ml_quant": "ML+Quant (deterministic)",
    "claude-opus-4-7": "Claude Opus 4.7",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "hf/deepseek-ai/DeepSeek-R1": "DeepSeek R1",
    "hf/deepseek-ai/DeepSeek-V3.2": "DeepSeek V3.2",
    "hf/meta-llama/Llama-3.3-70B-Instruct": "Llama 3.3 70B",
    "hf/Qwen/Qwen3-32B": "Qwen3 32B",
    "hf/Qwen/Qwen3-8B": "Qwen3 8B",
}

# ── Code-freshness probe ─────────────────────────────────────────────────
# Long-running daemons silently serve pre-deploy bytecode: the scorer-clamp
# fix landed while this :8090 process was already up, so it kept emitting
# ±700% "predictions" for hours. /api/build-info exposes the git SHA the
# process booted with vs the on-disk HEAD so an operator (and the unified
# dashboard's banner) can see "you're running stale code — restart".
_REPO_DIR = str(Path(__file__).resolve().parent.parent)


def _git_sha(repo_dir: str, ref: str = "HEAD") -> str | None:
    try:
        r = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "--short", ref],
            capture_output=True, text=True, timeout=3,
        )
        return (r.stdout.strip() or None) if r.returncode == 0 else None
    except Exception:
        return None


_BOOT_SHA = _git_sha(_REPO_DIR)
_BOOT_TIME = __import__("time").time()


def _head_sha_and_behind() -> tuple[str | None, int]:
    """Current on-disk HEAD short SHA + how many commits it is ahead of the
    SHA this process booted with (0 if in sync or indeterminable)."""
    head = _git_sha(_REPO_DIR)
    behind = 0
    if head and _BOOT_SHA and head != _BOOT_SHA:
        try:
            r = subprocess.run(
                ["git", "-C", _REPO_DIR, "rev-list", "--count",
                 f"{_BOOT_SHA}..HEAD"],
                capture_output=True, text=True, timeout=3,
            )
            if r.returncode == 0:
                behind = int(r.stdout.strip() or 0)
        except Exception:
            behind = 0
    return head, behind


@app.route("/api/healthz")
def healthz_api():
    """Lightweight liveness probe for watchdog / unified-dashboard. Exposes
    pid, uptime, build-staleness, and a cheap store sanity count so the
    watchdog can distinguish "alive but stuck" from "alive and trading"
    without hitting the slow /api/state path."""
    import os as _os_l
    import time as _time_l
    head, behind = _head_sha_and_behind()
    uptime_s = max(0.0, _time_l.time() - _BOOT_TIME)
    positions_n: int | None = None
    last_decision_age_s: float | None = None
    try:
        st = get_store()
        # The store's connection is shared across threads (check_same_thread=
        # False) and writes serialise on ``st._lock`` (store.py:134 docstring).
        # An unlocked read whose ``execute()`` interleaves with a concurrent
        # writer raises ``sqlite3.InterfaceError: bad parameter or other API
        # misuse`` or hands back a corrupted/None row — the very 500s the
        # store.py invariant exists to prevent. Hold ``_lock`` only across the
        # two reads; the git probe + parsing above are deliberately outside
        # the critical section (the writer would otherwise stall behind a
        # subprocess shell-out on every /api/healthz hit). Mirrors the
        # ``/api/trade-attribution`` access pattern at line 10216.
        with st._lock:  # noqa: SLF001 — documented store-access discipline
            row = st.conn.execute(
                "SELECT COUNT(*) FROM positions WHERE closed_at IS NULL AND qty != 0"
            ).fetchone()
            row2 = st.conn.execute(
                "SELECT MAX(timestamp) FROM decisions"
            ).fetchone()
        positions_n = int(row[0]) if row else 0
        if row2 and row2[0]:
            try:
                last_iso = str(row2[0]).replace("Z", "+00:00")
                last_dt = datetime.fromisoformat(last_iso)
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
                last_decision_age_s = max(
                    0.0,
                    (datetime.now(timezone.utc) - last_dt).total_seconds(),
                )
            except (TypeError, ValueError):
                last_decision_age_s = None
    except Exception:
        pass
    # Singleton-lock state of THIS process — surfaced on the cheap liveness
    # probe so the watchdog can detect a "fail-open degraded" runner without
    # paying for /api/state (see /api/runner-heartbeat for the long form).
    lock_status: str | None = None
    lock_holder_pid: int | None = None
    lock_degraded: bool | None = None
    try:
        from . import runner as _runner
        _ls = _runner.singleton_lock_state()
        if isinstance(_ls, dict):
            lock_status = _ls.get("status")
            lock_holder_pid = _ls.get("holder_pid")
            lock_degraded = bool(_ls.get("degraded"))
    except Exception:
        pass
    return jsonify({
        "ok": True,
        "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "service": "paper_trader",
        "pid": _os_l.getpid(),
        "uptime_s": round(uptime_s, 1),
        "boot_sha": _BOOT_SHA,
        "head_sha": head,
        "behind": behind,
        "stale": bool(_BOOT_SHA and head and head != _BOOT_SHA),
        "open_positions": positions_n,
        "last_decision_age_s": (round(last_decision_age_s, 1)
                                if last_decision_age_s is not None else None),
        "lock_status": lock_status,
        "lock_holder_pid": lock_holder_pid,
        "lock_degraded": lock_degraded,
    })


@app.route("/api/build-info")
def build_info_api():
    """{boot_sha, head_sha, behind, stale} — stale ⇒ restart to apply
    committed fixes (e.g. the DecisionScorer clamp)."""
    head, behind = _head_sha_and_behind()
    stale = bool(_BOOT_SHA and head and head != _BOOT_SHA)
    return jsonify({
        "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "service": "paper_trader",
        "boot_sha": _BOOT_SHA,
        "head_sha": head,
        "behind": behind,
        "stale": stale,
    })


@app.route("/api/notify-health")
def notify_health_api():
    """Discord delivery-health snapshot — operator-facing surface for the
    silent-channel class of failure (the 2026-05-17 ``env node`` PATH outage
    being the canonical case).

    The same data is already nested in ``/api/runner-heartbeat`` under
    ``notify`` — but that endpoint is SWR-cached (20s TTL, can serve a
    minutes-old payload under host saturation). When an operator suspects
    Discord is dark, they need the *current* in-process counter — not a
    cached snapshot frozen at "the last time the dashboard managed to
    rebuild the heartbeat panel". This endpoint is **deliberately NOT
    @swr_cached**: ``reporter.notify_health()`` is a pure module-global
    read (no I/O, no store hop) so the cost of bypassing the cache is zero
    while the latency benefit is the whole point.

    Pinned by ``tests/test_core_dashboard_notify_health.py`` so a future
    re-introduction of the cache (or an accidental import change) is
    caught by CI.

    Single source of truth (invariant #10): the verdict / headline /
    consecutive-failures fields are ``reporter.notify_health()``'s own —
    never re-derived here, so this endpoint and the ``notify`` block on
    ``/api/runner-heartbeat`` can never tell different stories.

    Failure contract: any import / call fault degrades to a valid-shaped
    ERROR envelope (verdict=ERROR + a short error string) so the panel
    can render and the operator sees the fault, never a 500 that the
    upstream digital-intern dashboard would render as "endpoint dark".
    """
    try:
        from . import reporter as _reporter
        nh = _reporter.notify_health()
        if not isinstance(nh, dict):
            nh = {"verdict": "ERROR",
                  "headline": "notify_health returned non-dict",
                  "consecutive_failures": 0,
                  "last_ok_ts": None, "last_attempt_ts": None,
                  "last_error": "", "restart_recommended": False}
        return jsonify({
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "service": "paper_trader",
            **nh,
        })
    except Exception as e:
        return jsonify({
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "service": "paper_trader",
            "verdict": "ERROR",
            "headline": f"notify-health endpoint error: {e}",
            "consecutive_failures": 0,
            "last_ok_ts": None,
            "last_attempt_ts": None,
            "last_error": str(e),
            "restart_recommended": False,
        }), 500


# Static sector classification for analytics + sector-pulse cards.
# Keyed by the symbols we actually use in the watchlist + portfolio.
SECTOR_MAP = {
    # Semis (cash)
    "NVDA": "semis", "AMD": "semis", "MU": "semis", "AMAT": "semis",
    "LRCX": "semis", "KLAC": "semis", "TSM": "semis", "ASML": "semis",
    "MRVL": "semis", "SMH": "semis", "SOXX": "semis",
    "DRAM": "semis", "SNDU": "semis",
    # Semis leveraged
    "SOXL": "semis_lev", "SOXS": "semis_lev", "NVDU": "semis_lev",
    "MUU": "semis_lev",
    # Optical / networking
    "LITE": "optical", "LNOK": "optical",
    # Broad market
    "SPY": "broad", "QQQ": "broad", "VOO": "broad", "VTI": "broad",
    # Broad leveraged
    "TQQQ": "broad_lev", "UPRO": "broad_lev", "SPXL": "broad_lev",
    "QLD": "broad_lev", "SSO": "broad_lev", "UDOW": "broad_lev",
    "URTY": "broad_lev", "TNA": "broad_lev",
    "SPXS": "broad_lev", "SQQQ": "broad_lev",
    # Tech / FAANG
    "AAPL": "tech", "MSFT": "tech", "META": "tech", "GOOG": "tech",
    "GOOGL": "tech", "AMZN": "tech", "TSLA": "tech", "NFLX": "tech",
    "TECL": "tech_lev", "TECS": "tech_lev", "FNGU": "tech_lev",
    "FNGD": "tech_lev", "MSFU": "tech_lev", "AMZU": "tech_lev",
    "GOOGU": "tech_lev", "METAU": "tech_lev", "TSLL": "tech_lev",
    "CONL": "crypto_lev", "BITU": "crypto_lev", "ETHU": "crypto_lev",
    # Sector leveraged
    "LABU": "bio_lev", "CURE": "health_lev",
    "FAS": "fin_lev", "DPST": "fin_lev",
    "NAIL": "housing_lev", "UTSL": "util_lev",
    "DFEN": "defense_lev",
}

# Sector-pulse card focuses on the user's actual interest areas.
SECTOR_PULSE_TICKERS = [
    "MU", "NVDA", "AMD", "TSM", "AMAT", "LRCX", "KLAC", "MRVL", "ASML",
    "SMH", "SOXX", "SOXL",
    "LITE", "LNOK", "DRAM", "SNDU", "MUU",
]


def _classify(ticker: str) -> str:
    return SECTOR_MAP.get(ticker.upper(), "other")


@app.after_request
def _cors(resp):
    # Cross-port fetch from Digital Intern dashboard (8080 → 8090).
    resp.headers.setdefault("Access-Control-Allow-Origin", "*")
    resp.headers.setdefault("Access-Control-Allow-Methods", "GET, OPTIONS")
    resp.headers.setdefault("Access-Control-Allow-Headers", "Content-Type")
    return resp


@app.after_request
def _gzip_local_json(resp):
    # Compress only locally-generated JSON. Streamed / direct_passthrough /
    # non-200 bodies (e.g. SWR {"warming": true} or streaming responses) are
    # skipped by the guards below — rely on them, nothing extra needed.
    try:
        if "gzip" not in request.headers.get("Accept-Encoding", ""):
            return resp
        if resp.status_code != 200:
            return resp
        if resp.direct_passthrough:
            return resp
        if resp.is_streamed:
            return resp
        if resp.headers.get("Content-Encoding"):
            return resp
        if not resp.headers.get("Content-Type", "").startswith("application/json"):
            return resp
        body = resp.get_data()
        if len(body) < 1024:
            return resp
        gzipped = gzip.compress(body)
        resp.set_data(gzipped)
        resp.headers["Content-Encoding"] = "gzip"
        vary = resp.headers.get("Vary")
        if vary:
            if "accept-encoding" not in vary.lower():
                resp.headers["Vary"] = vary + ", Accept-Encoding"
        else:
            resp.headers["Vary"] = "Accept-Encoding"
        return resp
    except Exception:
        return resp


# ──────────────────────────────────────────────────────────────────────────
# Stale-while-revalidate response cache for the slow yfinance/cross-DB
# endpoints (resolves AGENTS.md invariant #7's "remaining genuine concern").
#
# `dashboard.run(threaded=True)` (d5b8eac) removed *cross-request* head-of-line
# blocking, but several handlers still do unbounded yfinance / cross-DB I/O and
# each take many seconds *in isolation*: live user-perspective testing on
# 2026-05-17 measured /api/suggestions 5.2s, /api/data-feed 16s, /api/briefing
# & /api/feed-health >35s, and /api/thesis-drift outright hung (curl -m15 →
# 000). The panels are effectively dead in production and the /api/chat fan-out
# + the :8080→:8090 cross-fetch pay the full latency every poll.
#
# This mirrors unified_dashboard.py's /api/command-center SWR exactly: serve
# the last good payload instantly and, once it ages past the per-endpoint TTL,
# kick a single-flight background rebuild. It additionally *bounds the cold
# path* — command-center's cold build is ~6s so it builds inline; these can
# hang 35s+, so the first caller waits at most _SWR_COLD_BUDGET_S and then gets
# a fast valid-shaped {"warming": true} 200 while the build finishes for the
# next (auto-refresh) poll. Mirrors command-center's `cached`/`cache_age_s`
# honesty keys so a stale serve is never silent.
#
# Inert under pytest (PYTEST_CURRENT_TEST): a module-global response cache
# would leak one test's fixture DB into the next test's exact-value assertion
# (test_thesis_drift / test_correlation / test_feed_health_endpoint / …), so
# existing endpoint tests run the handler directly exactly as before. The SWR
# machinery itself is covered by dedicated tests that opt in via
# `_SWR_TEST_FORCE`.
# ──────────────────────────────────────────────────────────────────────────
import os as _os
import sys as _sys
import threading as _threading
import time as _time
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FuturesTimeout
from functools import wraps as _wraps

_SWR_LOCK = _threading.Lock()
_SWR_STATE: dict = {}
_SWR_EXEC = _ThreadPoolExecutor(max_workers=6, thread_name_prefix="dash-swr")
_SWR_COLD_BUDGET_S = 2.0
_SWR_TEST_FORCE = False  # dedicated SWR tests flip this; never set in prod
# A background rebuild that *raises* (vs. merely slow) used to vanish into
# `except Exception: return None` — the cache never populated and every poll
# re-served an opaque {"warming": true} forever, the exception recorded
# nowhere. We now count consecutive failures per key and log a throttled
# stderr line (operator signal, no template change): the 1st failure (early
# warning) and every Nth thereafter (sustained breakage), never once-per-poll.
_SWR_FAIL_LOG_EVERY = 10

# ──────────────────────────────────────────────────────────────────────────
# Bounded network I/O — the second half of AGENTS.md invariant #7's
# "remaining genuine concern". SWR (above) addressed the *cache* half: a
# slow endpoint serves its last good payload instantly. But the SWR
# background rebuild still calls yfinance, and yfinance/`requests` has **no
# total timeout** — a stalled HTTPS socket blocks for minutes (or until the
# OS TCP timeout). That hung call sits inside an `_SWR_EXEC` worker (only 6
# of them); a handful of simultaneous hangs exhausts the pool and *every*
# SWR panel goes permanently dark on `{"warming": true}` — the future never
# completes, so the single-flight guard never kicks a fresh rebuild either.
#
# `_bounded_call` submits the blocking network fn to a *separate* pool and
# `.result(timeout=…)`s it. On timeout the caller (and its SWR worker) is
# freed with a safe default after `timeout_s`; the hung yfinance thread
# leaks in the *net* pool only — never the SWR pool — so panels keep
# serving stale/`warming` and self-heal once the network recovers, instead
# of the whole dashboard dying. Python threads can't be force-killed, so a
# leaked worker is unavoidable; isolating the leak to a dedicated, larger
# pool is the robust mitigation (mirrors the SWR cold-path `fut.result`
# timeout idiom already in this module).
# ──────────────────────────────────────────────────────────────────────────
_NET_EXEC = _ThreadPoolExecutor(max_workers=8, thread_name_prefix="dash-net")
_NET_TIMEOUT_S = 8.0


def _bounded_call(fn, *, timeout_s: float = _NET_TIMEOUT_S, default=None,
                  label: str = ""):
    """Run a blocking network `fn` under a hard wall-clock bound.

    Returns ``fn()``'s value, or ``default`` if it raises or does not
    finish within ``timeout_s``. Never raises. See the block comment above
    for why the work runs in a dedicated pool, not inline."""
    try:
        fut = _NET_EXEC.submit(fn)
    except Exception:
        return default
    try:
        return fut.result(timeout=timeout_s)
    except _FuturesTimeout:
        print(f"[dashboard] bounded net call timed out after "
              f"{timeout_s:g}s ({label or fn})")
        return default
    except Exception:
        return default


def _swr_active() -> bool:
    """SWR is inert under pytest unless a dedicated test opts in — see the
    block comment (cross-test cache-leak isolation)."""
    if _SWR_TEST_FORCE:
        return True
    return not _os.environ.get("PYTEST_CURRENT_TEST")


def _swr_entry(key: str) -> dict:
    st = _SWR_STATE.get(key)
    if st is None:
        st = {"data": None, "status": 200,
              "ct": "application/json", "ts": 0.0, "fut": None,
              # failure observability — see _SWR_FAIL_LOG_EVERY
              "fail_count": 0, "last_error": None,
              "last_error_ts": 0.0, "last_ok_ts": 0.0}
        _SWR_STATE[key] = st
    return st


def _swr_serialize(rv):
    """Normalize a handler return value to (bytes, status, content_type,
    cacheable). Every completed build is *returned* (so a cold caller gets the
    handler's real result — including a 4xx/5xx degrade), but only a 200 JSON
    body is *cacheable* — an error must not be pinned for the whole TTL."""
    resp = app.make_response(rv)
    ct = resp.headers.get("Content-Type", "") or ""
    cacheable = resp.status_code == 200 and "application/json" in ct
    return resp.get_data(), resp.status_code, ct, cacheable


def _swr_make(data, status: int, ct: str, age, cached: bool):
    """Build a Response from cached/just-built bytes, injecting the
    command-center honesty keys into a 200 JSON object body (an error body or
    a list/scalar body is served verbatim)."""
    body = data
    if status == 200:
        try:
            obj = json.loads(body)
            if isinstance(obj, dict):
                obj["cached"] = cached
                obj["cache_age_s"] = round(age, 1) if age is not None else None
                body = json.dumps(obj)
        except Exception:
            pass
    return app.response_class(body, status=status, content_type=ct)


def _swr_refresh(key: str, fn, args, kwargs, qs: str):
    """Single-flight background rebuild. Runs the wrapped handler inside a
    synthesized request context so request.args still resolves for the
    parametrized endpoints (news-edge / source-edge / sector-heatmap)."""
    st = _swr_entry(key)
    with _SWR_LOCK:
        fut = st["fut"]
        if fut is not None and not fut.done():
            return fut

        def _run():
            try:
                with app.test_request_context(query_string=qs):
                    rv = fn(*args, **kwargs)
                    snap = _swr_serialize(rv)
            except Exception as e:
                # Single-flight guarantees one _run per key at a time, so
                # mutating st here is uncontended w.r.t. other builds.
                with _SWR_LOCK:
                    st["fail_count"] = st.get("fail_count", 0) + 1
                    st["last_error"] = f"{type(e).__name__}: {e}"[:200]
                    st["last_error_ts"] = _time.time()
                    n = st["fail_count"]
                    msg = st["last_error"]
                if n == 1 or n % _SWR_FAIL_LOG_EVERY == 0:
                    print(f"[swr] background build for {key!r} failed "
                          f"(consecutive #{n}): {msg}",
                          file=_sys.stderr, flush=True)
                return None
            data, status, ct, cacheable = snap
            with _SWR_LOCK:
                if st.get("fail_count"):       # a good build clears the streak
                    st["fail_count"] = 0
                    st["last_error"] = None
                st["last_ok_ts"] = _time.time()
                if cacheable:
                    st["data"], st["status"], st["ct"] = data, status, ct
                    st["ts"] = _time.time()
            return snap

        fut = _SWR_EXEC.submit(_run)
        st["fut"] = fut
        return fut


def swr_cached(name: str, ttl: float):
    """Decorate a slow read-only JSON endpoint with stale-while-revalidate +
    a bounded cold path. Apply *below* @app.route."""
    def deco(fn):
        @_wraps(fn)
        def wrapper(*args, **kwargs):
            if not _swr_active():
                return fn(*args, **kwargs)
            # str, not bytes: werkzeug's test_request_context(query_string=…)
            # raises TypeError on bytes (the synthesized-context replay path).
            qs = (request.query_string or b"").decode("latin-1")
            key = name + "?" + qs
            st = _swr_entry(key)
            now = _time.time()
            with _SWR_LOCK:
                have = st["data"] is not None
                age = (now - st["ts"]) if have else None
                snap = (st["data"], st["status"], st["ct"]) if have else None
            if have and age is not None and age < ttl:
                return _swr_make(*snap, age, True)
            fut = _swr_refresh(key, fn, args, kwargs, qs)
            if have:  # stale → serve the last good copy immediately
                return _swr_make(*snap, age, True)
            try:  # cold start → wait a bounded time for the build itself
                res = fut.result(timeout=_SWR_COLD_BUDGET_S)
            except Exception:
                res = None
            if res is not None:  # build finished in budget — serve it directly
                data, status, ct, _cacheable = res
                return _swr_make(data, status, ct, 0.0, False)
            # build still running (or raised): serve stale if it appeared,
            # else a fast valid-shaped placeholder the panels self-heal from.
            with _SWR_LOCK:
                have = st["data"] is not None
                age = (_time.time() - st["ts"]) if have else None
                snap = (st["data"], st["status"], st["ct"]) if have else None
            if have:
                return _swr_make(*snap, age, False)
            with _SWR_LOCK:
                fc = st.get("fail_count", 0)
                le = st.get("last_error")
                let = st.get("last_error_ts") or 0.0
            # attempts==0 / last_error==None ⇒ slow but healthy ("be
            # patient"); attempts>0 ⇒ the build keeps raising and will NOT
            # self-heal — the operator-facing broken-vs-slow discriminator.
            return jsonify({"warming": True, "cached": False,
                            "error": "computing — retry shortly",
                            "attempts": fc,
                            "last_error": le,
                            "stale_for_s": (round(_time.time() - let, 1)
                                            if let else None)})
        return wrapper
    return deco


TEMPLATE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Paper Trader</title>
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='5' fill='%230d0d0d'/%3E%3Cline x1='7' y1='15' x2='7' y2='18' stroke='%2300d4ff' stroke-width='1.5'/%3E%3Crect x='5.5' y='18' width='3' height='7' rx='0.5' fill='%2300d4ff'/%3E%3Cline x1='7' y1='25' x2='7' y2='27' stroke='%2300d4ff' stroke-width='1.5'/%3E%3Cline x1='15' y1='12' x2='15' y2='15' stroke='%23ff3c4c' stroke-width='1.5'/%3E%3Crect x='13.5' y='15' width='3' height='6' rx='0.5' fill='%23ff3c4c'/%3E%3Cline x1='15' y1='21' x2='15' y2='24' stroke='%23ff3c4c' stroke-width='1.5'/%3E%3Cline x1='23' y1='5' x2='23' y2='8' stroke='%2300ff9f' stroke-width='1.5'/%3E%3Crect x='21.5' y='8' width='3' height='12' rx='0.5' fill='%2300ff9f'/%3E%3Cline x1='23' y1='20' x2='23' y2='23' stroke='%2300ff9f' stroke-width='1.5'/%3E%3Cpolyline points='7,21 15,17 23,11' stroke='%23ffd700' stroke-width='1.2' fill='none' stroke-dasharray='2,1.5'/%3E%3C/svg%3E">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Outfit:wght@400;500;600;700&family=DM+Mono:ital,wght@0,400;0,500;1,400&display=swap');
    :root {
      color-scheme: dark;
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
    body {
      margin: 0; padding: 0;
      font-family: var(--font-sans);
      background: var(--bg); color: var(--text);
      font-size: 15px; line-height: 1.5;
    }
    .brand, h1, h2, h3 { font-family: var(--font-display); }
    .page-content { padding: 24px; max-width: 1600px; width: 100%; }
    .topbar {
      background: var(--bg-panel);
      border-bottom: 1px solid var(--border);
      padding: 0 20px; height: 48px;
      display: flex; align-items: center; gap: 2px;
      position: sticky; top: 0; z-index: 100; margin: 0;
      overflow: hidden; max-width: 100%;
    }
    .brand {
      font-weight: 700; color: var(--amber);
      font-size: 13px; letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-right: 16px; flex-shrink: 0;
    }
    .topbar a {
      color: var(--text-secondary); text-decoration: none;
      font-size: 13px; font-weight: 500;
      padding: 5px 12px; border-radius: var(--radius-sm);
      transition: color 0.15s, background 0.15s;
      white-space: nowrap;
    }
    .topbar a:hover { color: var(--text); background: var(--bg-hover); }
    .topbar a.active { color: var(--amber); background: var(--amber-dim); }
    h1 { margin: 0 0 4px; font-size: 22px; font-weight: 600; color: var(--text); }
    .sub { color: var(--text-secondary); font-size: 13px; margin-bottom: 20px; }
    nav.tabs {
      display: flex; gap: 2px; margin-bottom: 18px;
      border-bottom: 1px solid var(--border);
      overflow-x: auto; -webkit-overflow-scrolling: touch; flex-wrap: nowrap;
    }
    nav.tabs a {
      padding: 8px 16px; color: var(--text-secondary); text-decoration: none;
      border-bottom: 2px solid transparent; font-size: 13px; font-weight: 500;
      cursor: pointer; transition: color 0.15s; margin-bottom: -1px;
    }
    nav.tabs a.active { color: var(--amber); border-bottom-color: var(--amber); }
    nav.tabs a:hover { color: var(--text); }
    .tab-pane { display: none; }
    .tab-pane.active { display: block; }
    .grid {
      display: grid; gap: 18px;
      grid-template-columns: 1fr 1fr;
    }
    @media (max-width: 980px) { .grid { grid-template-columns: 1fr; } }
    .card {
      background: var(--bg-panel); border: 1px solid var(--border);
      border-radius: var(--radius); padding: 18px 20px;
      overflow-x: auto; -webkit-overflow-scrolling: touch;
    }
    .card h2 {
      margin: 0 0 14px; font-size: 11px; font-weight: 600;
      color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em;
    }
    .stat-row { display: flex; gap: 24px; flex-wrap: wrap; margin-bottom: 12px; }
    .stat { flex: 1 1 120px; }
    .stat .v {
      font-family: var(--font-mono);
      font-size: 24px; color: var(--text); font-weight: 500;
      font-variant-numeric: tabular-nums;
      min-width: 0; max-width: 100%;
    }
    .stat .l { color: var(--text-muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; }
    .pos, .pl { color: var(--green); }
    .neg { color: var(--red); }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th {
      text-align: left; padding: 0 10px 10px;
      font-size: 11px; font-weight: 600; color: var(--text-muted);
      text-transform: uppercase; letter-spacing: 0.08em;
      border-bottom: 1px solid var(--border-strong);
    }
    td {
      padding: 8px 10px; border-bottom: 1px solid var(--border);
      font-size: 13px;
    }
    td.num {
      text-align: right;
      font-family: var(--font-mono);
      font-variant-numeric: tabular-nums;
    }
    tr:hover td { background: var(--bg-hover); }
    .muted { color: var(--text-secondary); }
    canvas { max-width: 100%; max-height: 280px; }
    .pill {
      display: inline-flex; align-items: center;
      padding: 2px 8px; border-radius: 4px;
      background: var(--bg-elevated); color: var(--text-secondary);
      font-size: 11px; font-weight: 500; letter-spacing: 0.04em;
      font-family: var(--font-sans);
    }
    .pill.buy { background: var(--green-dim); color: var(--green); }
    .pill.sell { background: var(--red-dim); color: var(--red); }
    .pill.hold { background: var(--bg-elevated); color: var(--text-secondary); }
    .pill.run { background: var(--blue-dim); color: var(--blue); }
    .pill.status-running  { background: var(--blue-dim); color: var(--blue); }
    .pill.status-complete { background: var(--green-dim); color: var(--green); }
    .pill.status-failed   { background: var(--red-dim); color: var(--red); }
    .pill.status-pending  { background: var(--bg-elevated); color: var(--text-secondary); }
    .spinner {
      display: inline-block; width: 10px; height: 10px;
      border: 2px solid var(--border-strong); border-top-color: var(--cyan);
      border-radius: 50%; animation: spin 0.8s linear infinite;
      vertical-align: middle; margin-right: 6px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .progress-wrap {
      margin: 8px 0; height: 4px; background: var(--bg-elevated);
      border-radius: 4px; overflow: hidden;
    }
    .progress-bar {
      height: 100%; background: linear-gradient(90deg, var(--amber), var(--cyan));
      transition: width 0.4s ease;
    }
    .progress-label { font-size: 11px; color: var(--text-muted); margin-bottom: 4px; }
    tr.bt-row { cursor: pointer; }
    tr.bt-row:hover td { background: var(--bg-hover); }
    tr.bt-row.best td { background: var(--green-dim); }
    tr.bt-row.beat td:first-child { border-left: 2px solid var(--green); }
    tr.bt-row.miss td:first-child { border-left: 2px solid var(--red); }
    #bt-trades { margin-top: 14px; display: none; }
    #bt-trades.show { display: block; }
    .bt-headline {
      display: flex; gap: 28px; flex-wrap: wrap; margin-bottom: 12px;
    }
    .bt-headline .stat .v { font-size: 22px; }
    .bt-layout {
      display: grid; grid-template-columns: 240px 1fr; gap: 14px; align-items: start;
      min-width: 0;
    }
    /* Let grid items shrink below their content so a long run name can't
       blow the 1fr track past the viewport on mobile (agent6 frontend audit). */
    .bt-layout > * { min-width: 0; }
    @media (max-width: 980px) { .bt-layout { grid-template-columns: 1fr; } }
    .bt-sidebar { position: sticky; top: 62px; max-height: calc(100vh - 78px); overflow-y: auto; }
    .bt-sidebar h2 { margin: 0; }
    .bt-legend-row {
      display: flex; align-items: center; gap: 8px; padding: 6px 4px;
      border-bottom: 1px solid var(--border); cursor: pointer; user-select: none;
      transition: background 0.15s;
    }
    .bt-legend-row:hover { background: var(--bg-hover); }
    .bt-legend-row.selected { background: var(--bg-elevated); }
    .bt-legend-row.hidden-run { opacity: 0.35; }
    .bt-legend-row input[type=checkbox] { accent-color: var(--cyan); margin: 0; }
    .bt-swatch {
      width: 12px; height: 12px; border-radius: 3px; flex: 0 0 12px;
    }
    .bt-legend-row .name { flex: 1; font-size: 13px; color: var(--text);
      min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .bt-legend-row .ret { font-size: 11px; font-variant-numeric: tabular-nums; font-family: var(--font-mono); }
    .bt-btn {
      background: var(--bg-elevated); color: var(--text);
      border: 1px solid var(--border-strong); border-radius: var(--radius-sm);
      padding: 3px 8px; font-size: 11px; cursor: pointer;
      text-transform: uppercase; letter-spacing: 0.5px;
      font-family: var(--font-sans);
    }
    .bt-btn:hover { background: var(--bg-hover); }
    .bt-filter-chip {
      background: var(--bg-elevated); color: var(--text-secondary);
      border: 1px solid var(--border); border-radius: 99px;
      padding: 3px 10px; font-size: 11px; cursor: pointer;
      font-family: var(--font-sans); transition: all 0.15s;
    }
    .bt-filter-chip:hover { border-color: var(--cyan); color: var(--text); }
    .bt-filter-chip.active {
      background: rgba(10,205,255,0.12); border-color: var(--cyan);
      color: var(--cyan); font-weight: 600;
    }
    .bt-tabs {
      display: flex; gap: 2px; margin-bottom: 12px;
      border-bottom: 1px solid var(--border);
    }
    .bt-tabs a {
      padding: 8px 14px; color: var(--text-secondary); cursor: pointer; font-size: 13px;
      border-bottom: 2px solid transparent; font-weight: 500;
    }
    .bt-tabs a.active { color: var(--amber); border-bottom-color: var(--amber); }
    .bt-subpane { display: none; }
    .bt-subpane.active { display: block; }
    .bt-section { display: none; }
    .bt-section.active { display: block; }
    tr.bt-row.selected td { background: var(--bg-elevated) !important; }
    .pill.status-running { animation: pulse 1.5s ease-in-out infinite; }
    @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.55;} }
    .live-dot {
      display: inline-block; width: 7px; height: 7px; border-radius: 50%;
      background: var(--green); margin-right: 6px; animation: pulse 1.5s infinite;
    }
    th.sortable-h { cursor: pointer; user-select: none; }
    th.sortable-h:hover { color: var(--text); }
    th.sortable-h.sort-asc::after  { content: " ▲"; font-size: 11px; }
    th.sortable-h.sort-desc::after { content: " ▼"; font-size: 11px; }
    select, input[type="text"], input[type="number"] {
      background: var(--bg-input); color: var(--text);
      border: 1px solid var(--border-strong); border-radius: var(--radius-sm);
      padding: 6px 10px; font-size: 13px; font-family: var(--font-sans);
    }
    button, .btn {
      background: var(--bg-elevated); color: var(--text);
      border: 1px solid var(--border-strong); border-radius: var(--radius-sm);
      padding: 6px 14px; font-size: 13px; font-family: var(--font-sans);
      cursor: pointer; transition: background 0.15s;
    }
    button:hover, .btn:hover { background: var(--bg-hover); }
    button.primary, .btn-primary {
      background: var(--amber-dim);
      border-color: rgba(240,180,41,0.3);
      color: var(--amber);
    }
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
    /* Responsive 2-col grid that stacks on mobile */
    .grid-2col {
      display: grid; grid-template-columns: 1fr 1fr; gap: 18px;
    }
    @media (max-width: 768px) {
      .topbar-nav { display: none; }
      .nav-hamburger { display: flex; }
      body { font-size: 14px; }
      button, .btn, a.btn, [role="button"] { min-height: 44px; min-width: 44px; }
      /* Prevent any fixed-width grid column from overflowing on narrow screens */
      .grid-2col { grid-template-columns: 1fr; }
    }
    @media (max-width: 480px) {
      body { padding-bottom: 72px; }
      .bottom-nav { display: grid; }
      .topbar { padding: 0 16px; }
      .page-content { padding: 14px; }
      .card { min-height: auto !important; padding: 14px 16px; }
      .grid, .grid-2, .grid2, .grid-2col { grid-template-columns: 1fr !important; }
      .bt-layout { grid-template-columns: 1fr !important; }
      .stat-row { gap: 12px; }
      .stat .v { font-size: 18px; }
      [style*="max-height: 520px"],
      [style*="max-height:520px"] { max-height: 60vh !important; }
      table { font-size: 12px; }
      th, td { padding: 8px 10px; }
    }
  </style>
</head>
<body>
  <nav class="topbar">
    <span class="brand">◈ TRADING STACK</span>
    <span class="topbar-nav" style="display:flex;align-items:center;gap:2px;">
      <a href="/">Command Center</a>
      <a href="/intern/">Digital Intern</a>
      <a href="/trader/" class="{% if initial_tab != 'backtests' %}active{% endif %}">Paper Trader</a>
      <a href="/trader/backtests" class="{% if initial_tab == 'backtests' %}active{% endif %}">Backtests</a>
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
  <!-- ─── Global stale-process banner (new 2026-05-16, agent 4) ───
       Always-on, page-wide. Per-panel fetchMaybeStale only degrades the
       endpoints a stale boot is missing; nothing told the operator the
       whole process is behind HEAD — so the self-review mirror silently
       not being injected (exactly the live state on 2026-05-16) was
       invisible from the trader page. Polls /api/build-info. -->
  <div id="global-stale-banner" style="display:none;background:#b71c1c;color:#fff;
       padding:9px 16px;font-size:13px;font-weight:600;text-align:center;
       letter-spacing:0.2px;border-bottom:1px solid #7f0000;">
    <span id="global-stale-text">⚠ Paper-trader is running stale code — restart to apply committed fixes.</span>
  </div>
  <div class="nav-drawer" id="navDrawer">
    <div class="nav-drawer-header">◈ TRADING STACK</div>
    <a href="/">Command Center</a>
    <a href="/intern/">Digital Intern</a>
    <a href="/trader/" class="{% if initial_tab != 'backtests' %}active{% endif %}">Paper Trader</a>
    <a href="/trader/backtests" class="{% if initial_tab == 'backtests' %}active{% endif %}">Backtests</a>
    <a href="/backtests/compare">Compare</a>
    <a href="/journal">Journal</a>
    <a href="/ops/">Ops View</a>
    <a href="/intern/chat">Chat</a>
    <a href="/system/">System</a>
  </div>
  <div class="nav-overlay" id="navOverlay"></div>

  <div class="page-content">
  <h1>Paper Trader</h1>
  <div class="sub" id="hb">loading…</div>

  <!-- ─── Live news data feed (Digital Intern collector pulse) ─── -->
  <div id="data-feed-widget"
       style="display:flex;flex-wrap:wrap;align-items:center;gap:14px;
              background:#11141a;border:1px solid #1f2126;border-radius:6px;
              padding:8px 12px;margin-bottom:14px;font-size:12px;color:#8b929d;">
    <span style="display:inline-flex;align-items:center;gap:6px;color:#dde1e7;">
      <span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:#00c896;"></span>
      <b style="font-weight:600;letter-spacing:0.04em;">DATA FEED</b>
    </span>
    <span>last 1h: <b id="df-1h" style="color:#dde1e7;">—</b></span>
    <span>24h: <b id="df-24h" style="color:#dde1e7;">—</b></span>
    <span id="df-sources" style="flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">—</span>
    <span class="muted" id="df-asof" style="font-size:11px;">—</span>
  </div>

  <div class="card" style="margin-bottom:18px;">
    <h2 style="display:flex;justify-content:space-between;align-items:center;">
      <span>Signal Feed — Digital Intern</span>
      <a href="/intern/" style="font-size:11px;color:#0acdff;text-decoration:none;text-transform:none;letter-spacing:normal">View All Signals →</a>
    </h2>
    <ul id="signal-feed" style="margin:0;padding:0;list-style:none;font-size:12px;">
      <li class="muted">loading…</li>
    </ul>
  </div>

  <nav class="tabs">
    <a id="tab-trader-link"    onclick="showTab('trader')">Trader</a>
    <a id="tab-backtests-link" onclick="showTab('backtests')">Backtests</a>
  </nav>

  <!-- ────── Trader pane ────── -->
  <div id="tab-trader" class="tab-pane">

    <!-- ─── Equity Curve (pinned top) ─── -->
    <div class="card" style="margin-bottom:18px;">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px;margin-bottom:10px;">
        <h2 style="margin:0;">Live portfolio</h2>
        <div style="display:flex;gap:3px;font-size:11px;">
          <button class="bt-filter-chip active" id="eq-range-all" onclick="setEqRange('all')">All</button>
          <button class="bt-filter-chip" id="eq-range-24h" onclick="setEqRange('24h')">24h</button>
          <button class="bt-filter-chip" id="eq-range-7d" onclick="setEqRange('7d')">7d</button>
        </div>
      </div>
      <div class="stat-row" style="margin-bottom:10px;">
        <div class="stat"><div class="l">total value</div><div class="v" id="tv">—</div></div>
        <div class="stat"><div class="l">cash</div><div class="v" id="cash">—</div></div>
        <div class="stat"><div class="l">return vs start</div><div class="v" id="pl">—</div></div>
        <div class="stat"><div class="l">vs SPY (same period)</div><div class="v" id="vs-spy-live">—</div></div>
        <div class="stat"><div class="l">max drawdown</div><div class="v" id="live-maxdd">—</div></div>
        <div class="stat"><div class="l">cash deployed</div><div class="v" id="live-deployed">—</div></div>
      </div>
      <div style="font-size:11px;color:var(--text-muted);margin-bottom:6px;">
        <span style="color:#0acdff;">●</span> Portfolio (% from start) &nbsp;
        <span style="border-top:2px dashed rgba(255,183,77,0.7);display:inline-block;width:16px;vertical-align:middle;"></span> SPY (% from same start) &nbsp;
        <span style="color:var(--text-muted);">↑ buy &nbsp; ↓ sell</span>
      </div>
      <div style="position:relative;height:280px;"><canvas id="eq"></canvas></div>
      <div style="margin-top:6px;">
        <div style="font-size:10px;color:var(--text-muted);margin-bottom:3px;">Drawdown from peak (%)</div>
        <div style="position:relative;height:80px;"><canvas id="eq-dd"></canvas></div>
      </div>
    </div>

    <!-- ─── Daily Briefing (futures + market countdown + urgent news) ─── -->
    <div class="card" id="briefing-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span><span id="briefing-dot" style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#8b929d;margin-right:8px;"></span>Daily briefing</span>
        <span class="muted" id="briefing-asof" style="font-size:11px;text-transform:none;letter-spacing:normal;">—</span>
      </h2>
      <div id="briefing-status" style="font-size:14px;color:#dde1e7;margin-bottom:12px;">loading…</div>
      <div id="briefing-futures" style="display:flex;flex-wrap:wrap;gap:14px;margin-bottom:14px;font-size:13px;"></div>
      <div style="font-size:11px;color:#8b929d;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">Urgent overnight news</div>
      <ul id="briefing-urgent" style="margin:0;padding:0;list-style:none;font-size:13px;"></ul>
    </div>

    <!-- ─── Session Delta — what materially changed since you last looked (new 2026-05-16, agent 4) ─── -->
    <div class="card" id="sess-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Since you last looked <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— material events in the window, ranked (no snapshot scanning)</span></span>
        <span id="sess-state" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div style="display:flex;gap:4px;font-size:11px;margin-bottom:10px;">
        <button class="bt-filter-chip" id="sess-w-60" onclick="setSessWindow(60)">1h</button>
        <button class="bt-filter-chip active" id="sess-w-360" onclick="setSessWindow(360)">6h</button>
        <button class="bt-filter-chip" id="sess-w-1440" onclick="setSessWindow(1440)">24h</button>
      </div>
      <div class="muted" id="sess-headline" style="font-size:12px;margin-bottom:12px;">loading…</div>
      <table id="sess-events" style="font-size:12px;">
        <thead><tr>
          <th style="width:62px;">when</th><th style="width:120px;">event</th><th>detail</th>
        </tr></thead>
        <tbody><tr><td colspan="3" class="muted">loading…</td></tr></tbody>
      </table>
    </div>

    <!-- ─── Trade Suggestions (co-pilot) ─── -->
    <div class="card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Trade suggestions <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— co-pilot, not auto-executed</span></span>
        <span class="muted" id="sug-meta" style="font-size:11px;text-transform:none;letter-spacing:normal;">—</span>
      </h2>
      <div id="sug-summary" style="font-size:12px;color:#8b929d;margin-bottom:10px;">loading…</div>
      <table id="sug-tbl" style="font-size:13px;">
        <thead><tr>
          <th>action</th><th>ticker</th><th class="num">conv.</th>
          <th class="num">price</th><th class="num">qty</th>
          <th class="num">news</th><th class="num">RSI</th>
          <th>reasons</th><th>headline</th>
        </tr></thead><tbody><tr><td colspan="9" class="muted">loading…</td></tr></tbody>
      </table>
    </div>

    <!-- ─── Risk panel (concentration / leverage / age / shock) ─── -->
    <div class="card" style="margin-bottom:18px;">
      <h2>Risk panel</h2>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">concentration top1</div><div class="v" id="risk-top1">—</div></div>
        <div class="stat"><div class="l">top3 weight</div><div class="v" id="risk-top3">—</div></div>
        <div class="stat"><div class="l">leveraged %</div><div class="v" id="risk-lev">—</div></div>
        <div class="stat"><div class="l">SPY -3% shock</div><div class="v" id="risk-shock">—</div></div>
        <div class="stat"><div class="l">median age (d)</div><div class="v" id="risk-age">—</div></div>
        <div class="stat"><div class="l">stale positions</div><div class="v" id="risk-stale-n">—</div></div>
      </div>
      <div id="risk-stale-list" style="font-size:12px;color:#dde1e7;"></div>
    </div>

    <!-- ─── Position blow-up ladder (per-name single-name shock) ─── -->
    <div class="card" id="blowup-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>🧨 Position blow-up ladder <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— each held name shocked alone, idiosyncratic (no beta)</span></span>
        <span class="muted" id="blowup-asof" style="font-size:11px;text-transform:none;letter-spacing:normal;">—</span>
      </h2>
      <div id="blowup-headline" class="muted" style="font-size:12px;margin-bottom:10px;">loading…</div>
      <table style="width:100%;">
        <thead><tr>
          <th>ticker</th><th class="num">weight</th>
          <th class="num">−10%</th><th class="num">−25%</th>
          <th class="num">−50%</th><th class="num">to zero</th>
        </tr></thead>
        <tbody id="blowup-tbody"><tr><td colspan="6" class="muted">loading…</td></tr></tbody>
      </table>
    </div>

    <!-- ─── Earnings Risk ─── -->
    <div class="card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Earnings radar <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— scheduled gap risk on holdings &amp; watchlist</span></span>
        <span class="muted" id="er-asof" style="font-size:11px;text-transform:none;letter-spacing:normal;">—</span>
      </h2>
      <div class="muted" id="er-meta" style="font-size:11px;margin-bottom:8px;">—</div>
      <ul id="er-list" style="margin:0;padding:0;list-style:none;font-size:13px;">
        <li class="muted">loading…</li>
      </ul>
    </div>

    <!-- ─── Portfolio Greeks (options exposure) ─── -->
    <div class="card" id="greeks-card" style="margin-bottom:18px;display:none;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Portfolio Greeks <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— Black-Scholes, live IV from yfinance</span></span>
        <span class="muted" id="gk-asof" style="font-size:11px;text-transform:none;letter-spacing:normal;">—</span>
      </h2>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">net delta</div><div class="v" id="gk-delta">—</div></div>
        <div class="stat"><div class="l">net gamma</div><div class="v" id="gk-gamma">—</div></div>
        <div class="stat"><div class="l">theta / day</div><div class="v" id="gk-theta">—</div></div>
        <div class="stat"><div class="l">vega / 1% IV</div><div class="v" id="gk-vega">—</div></div>
        <div class="stat"><div class="l">gross $ notional</div><div class="v" id="gk-notional">—</div></div>
        <div class="stat"><div class="l">delta % of port</div><div class="v" id="gk-deltapct">—</div></div>
      </div>
      <table id="gk-tbl" style="font-size:13px;">
        <thead><tr>
          <th>ticker</th><th>type</th><th class="num">qty</th>
          <th class="num">expiry / strike</th><th class="num">IV</th>
          <th class="num">Δ delta</th><th class="num">Γ</th>
          <th class="num">Θ / day</th><th class="num">ν / 1%</th>
        </tr></thead><tbody><tr><td colspan="9" class="muted">no option positions</td></tr></tbody>
      </table>
    </div>

    <!-- ─── DecisionScorer per-position predictions ─── -->
    <div class="card" id="scorer-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>ML scorer · per-position outlook <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— predicted 5-day forward return from DecisionScorer MLP</span></span>
        <span class="muted" id="sc-asof" style="font-size:11px;text-transform:none;letter-spacing:normal;">—</span>
      </h2>
      <div class="muted" id="sc-meta" style="font-size:11px;margin-bottom:8px;">loading…</div>
      <table id="sc-tbl" style="font-size:13px;">
        <thead><tr>
          <th>ticker</th>
          <th class="num">pred 5d</th>
          <th>verdict</th>
          <th class="num">RSI</th>
          <th class="num">MACD</th>
          <th class="num">mom 5d</th>
          <th class="num">mom 20d</th>
          <th class="num">news</th>
        </tr></thead>
        <tbody><tr><td colspan="8" class="muted">no open stock positions</td></tr></tbody>
      </table>
    </div>

    <!-- ─── Position Thesis Cards (new 2026-05-15) ─── -->
    <div class="card" id="thesis-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Position thesis <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— per-holding integrated view: news, scorer, technicals, last decision, verdict</span></span>
        <span class="muted" id="th-asof" style="font-size:11px;text-transform:none;letter-spacing:normal;">—</span>
      </h2>
      <div class="muted" id="th-meta" style="font-size:11px;margin-bottom:10px;">loading…</div>
      <div id="th-grid" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(min(420px,100%),1fr));gap:12px;">
        <div class="muted">loading…</div>
      </div>
    </div>

    <!-- ─── Drawdown Anatomy (new 2026-05-15) ─── -->
    <div class="card" id="dd-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Drawdown anatomy <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— current DD from peak with per-position contribution</span></span>
        <span class="muted" id="dd-asof" style="font-size:11px;text-transform:none;letter-spacing:normal;">—</span>
      </h2>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">current equity</div><div class="v" id="dd-current">—</div></div>
        <div class="stat"><div class="l">peak equity</div><div class="v" id="dd-peak">—</div></div>
        <div class="stat"><div class="l">current drawdown from peak</div><div class="v" id="dd-pct">—</div></div>
        <div class="stat"><div class="l">max drawdown (trough)</div><div class="v" id="dd-trough">—</div></div>
        <div class="stat"><div class="l">time in DD</div><div class="v" id="dd-hours">—</div></div>
        <div class="stat"><div class="l">recovered</div><div class="v" id="dd-rec">—</div></div>
      </div>
      <div style="font-size:13px;color:#dde1e7;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.5px;">Per-position contribution</div>
      <table id="dd-tbl" style="font-size:13px;">
        <thead><tr>
          <th>ticker</th><th class="num">qty</th><th class="num">cost</th>
          <th class="num">px</th><th class="num">P/L $</th><th class="num">P/L %</th>
          <th>drag</th>
        </tr></thead>
        <tbody><tr><td colspan="7" class="muted">loading…</td></tr></tbody>
      </table>
    </div>

    <!-- ─── Confidence Calibration + Signal Attribution (new 2026-05-15) ─── -->
    <div class="card" id="cal-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Confidence calibration &amp; signal attribution <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— do high-confidence trades actually win? which signal types pay?</span></span>
        <span class="muted" id="cal-asof" style="font-size:11px;text-transform:none;letter-spacing:normal;">—</span>
      </h2>
      <div class="muted" id="cal-meta" style="font-size:11px;margin-bottom:10px;">loading…</div>
      <div class="grid-2col">
        <div>
          <div style="font-size:12px;color:#dde1e7;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;">By Opus confidence</div>
          <table id="cal-conf-tbl" style="font-size:13px;">
            <thead><tr>
              <th>bucket</th><th class="num">n</th><th class="num">win %</th>
              <th class="num">avg ret</th><th class="num">avg conf</th>
            </tr></thead>
            <tbody><tr><td colspan="5" class="muted">loading…</td></tr></tbody>
          </table>
        </div>
        <div>
          <div style="font-size:12px;color:#dde1e7;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;">By signal source</div>
          <table id="cal-src-tbl" style="font-size:13px;">
            <thead><tr>
              <th>source</th><th class="num">n</th><th class="num">win %</th>
              <th class="num">avg ret</th><th class="num">best / worst</th>
            </tr></thead>
            <tbody><tr><td colspan="5" class="muted">loading…</td></tr></tbody>
          </table>
        </div>
      </div>
      <div style="font-size:12px;color:#dde1e7;margin-top:14px;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;">Recent realized trades</div>
      <table id="cal-recent-tbl" style="font-size:12px;">
        <thead><tr>
          <th>buy → sell</th><th>ticker</th><th class="num">return</th>
          <th class="num">conf</th><th>source</th><th>reasoning</th>
        </tr></thead>
        <tbody><tr><td colspan="6" class="muted">loading…</td></tr></tbody>
      </table>
    </div>

    <!-- ─── Capital Deployment & Liquidity (new 2026-05-15, agent 4) ─── -->
    <div class="card" id="liq-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Capital deployment &amp; liquidity <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— is the book pinned with no dry powder?</span></span>
        <span id="liq-status" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div class="muted" id="liq-headline" style="font-size:12px;margin-bottom:12px;">loading…</div>
      <div class="stat-row" style="margin-bottom:12px;">
        <div class="stat"><div class="l">cash</div><div class="v" id="liq-cash">—</div></div>
        <div class="stat"><div class="l">deployed</div><div class="v" id="liq-deployed">—</div></div>
        <div class="stat"><div class="l">positions</div><div class="v" id="liq-npos">—</div></div>
        <div class="stat"><div class="l">top weight</div><div class="v" id="liq-top">—</div></div>
        <div class="stat"><div class="l">unrealized P/L</div><div class="v" id="liq-upl">—</div></div>
        <div class="stat"><div class="l">last entry</div><div class="v" id="liq-entry">—</div></div>
      </div>
      <div id="liq-bar" style="display:flex;height:18px;border-radius:6px;overflow:hidden;background:#0d1117;border:1px solid #1f2126;margin-bottom:6px;"></div>
      <div class="muted" id="liq-bar-legend" style="font-size:11px;margin-bottom:12px;">—</div>
      <div id="liq-flags" style="font-size:12px;color:#dde1e7;"></div>
    </div>

    <!-- ─── Decision Pipeline Health (new 2026-05-15, agent 4) ─── -->
    <div class="card" id="dh-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Decision pipeline health <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— is the live Opus trader actually deciding? NO_DECISION = parse failure</span></span>
        <span id="dh-verdict" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div class="muted" id="dh-reason" style="font-size:12px;margin-bottom:12px;">loading…</div>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">cycles (24h)</div><div class="v" id="dh-total">—</div></div>
        <div class="stat"><div class="l">parse-fail (24h)</div><div class="v" id="dh-fail">—</div></div>
        <div class="stat"><div class="l">fills (24h)</div><div class="v" id="dh-fills">—</div></div>
        <div class="stat"><div class="l">avg confidence</div><div class="v" id="dh-conf">—</div></div>
        <div class="stat"><div class="l">since last fill</div><div class="v" id="dh-lastfill">—</div></div>
        <div class="stat"><div class="l">signals / cycle</div><div class="v" id="dh-sigs">—</div></div>
      </div>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">cycles (all-time)</div><div class="v" id="dh-total-all">—</div></div>
        <div class="stat"><div class="l">parse-fail (all-time)</div><div class="v" id="dh-fail-all">—</div></div>
        <div class="stat"><div class="l">fills (all-time)</div><div class="v" id="dh-fills-all">—</div></div>
      </div>
      <div style="font-size:12px;color:#dde1e7;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;">Action mix (all-time)</div>
      <div id="dh-mix" style="margin-bottom:14px;"><div class="muted">loading…</div></div>
      <div style="font-size:12px;color:#dde1e7;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;">Recent cycles</div>
      <table id="dh-tape" style="font-size:12px;">
        <thead><tr>
          <th>time</th><th>outcome</th><th>action</th>
          <th class="num">conf</th><th class="num">signals</th>
        </tr></thead>
        <tbody><tr><td colspan="5" class="muted">loading…</td></tr></tbody>
      </table>
    </div>

    <!-- ─── Decision Failure Forensics (new 2026-05-15, agent 4) ─── -->
    <div class="card" id="df-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Decision failure forensics <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— WHY a cycle produced no decision, with the raw Opus excerpt</span></span>
        <span id="df-verdict" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div class="muted" id="df-reason" style="font-size:12px;margin-bottom:6px;">loading…</div>
      <div id="df-hint" style="font-size:12px;color:#ffd479;margin-bottom:12px;"></div>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">failures</div><div class="v" id="df-nfail">—</div></div>
        <div class="stat"><div class="l">rate (24h)</div><div class="v" id="df-rate">—</div></div>
        <div class="stat"><div class="l">retry-exhausted</div><div class="v" id="df-retry">—</div></div>
        <div class="stat"><div class="l">dominant mode</div><div class="v" id="df-dom" style="font-size:14px;">—</div></div>
        <div class="stat"><div class="l">open mkt fail%</div><div class="v" id="df-open">—</div></div>
        <div class="stat"><div class="l">closed mkt fail%</div><div class="v" id="df-closed">—</div></div>
      </div>
      <div style="font-size:12px;color:#dde1e7;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;">Failure modes</div>
      <div id="df-mix" style="margin-bottom:12px;"><div class="muted">loading…</div></div>
      <div style="font-size:12px;color:#dde1e7;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;">Hourly parse-fail (last 24h)</div>
      <div id="df-hourly" style="display:flex;align-items:flex-end;gap:3px;height:46px;margin-bottom:14px;"><div class="muted">—</div></div>
      <div style="font-size:12px;color:#dde1e7;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;">Decision-loss clock — parse-fail by UTC hour <span class="muted" style="text-transform:none;letter-spacing:normal;">(current regime; folds every day onto one 24h clock so a recurring host-load window shows)</span></div>
      <div id="df-clock-hint" style="font-size:12px;color:#ffd479;margin-bottom:6px;"></div>
      <div id="df-clock" style="display:flex;align-items:flex-end;gap:2px;height:46px;margin-bottom:4px;"><div class="muted">—</div></div>
      <div id="df-clock-axis" style="display:flex;gap:2px;margin-bottom:14px;font-size:9px;color:#5c6470;"></div>
      <div style="font-size:12px;color:#dde1e7;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;">Recent failures — raw Opus excerpt</div>
      <table id="df-tape" style="font-size:12px;">
        <thead><tr>
          <th>time</th><th>mode</th><th>mkt</th><th>excerpt</th>
        </tr></thead>
        <tbody><tr><td colspan="4" class="muted">loading…</td></tr></tbody>
      </table>
    </div>

    <!-- ─── Decision Drought Drift (new 2026-05-16, agent 4) ─── -->
    <div class="card" id="drought-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Decision drought drift <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— what the bot's <em>inaction</em> cost: portfolio vs S&amp;P while it wasn't trading</span></span>
        <span id="dd-verdict" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div class="muted" id="dd-reason" style="font-size:12px;margin-bottom:12px;">loading…</div>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">fills / cycles</div><div class="v" id="dd-fills">—</div></div>
        <div class="stat"><div class="l">droughts</div><div class="v" id="dd-n">—</div></div>
        <div class="stat"><div class="l">paralysis droughts</div><div class="v" id="dd-npar">—</div></div>
        <div class="stat"><div class="l">involuntary alpha bleed</div><div class="v" id="dd-bleed">—</div></div>
      </div>
      <div id="drought-current" style="font-size:12px;background:#0d1117;border:1px solid #1f2126;border-radius:6px;padding:10px 12px;margin-bottom:14px;color:#8b929d;">loading…</div>
      <div style="font-size:12px;color:#dde1e7;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;">Droughts (newest first) — alpha = portfolio% − S&amp;P% over the idle window</div>
      <table id="dd-tape" style="font-size:12px;">
        <thead><tr>
          <th>start</th><th class="num">hrs</th><th class="num">cyc</th><th>kind</th>
          <th class="num">ND%</th><th class="num">port%</th><th class="num">spy%</th><th class="num">alpha%</th>
        </tr></thead>
        <tbody><tr><td colspan="8" class="muted">loading…</td></tr></tbody>
      </table>
    </div>

    <!-- ─── News Edge (new 2026-05-16, agent 4) ─── -->
    <div class="card" id="ne-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>News edge <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— does a high ai_score headline actually predict the move? (SPY-abnormal)</span></span>
        <span id="ne-verdict" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div class="muted" id="ne-reason" style="font-size:12px;margin-bottom:12px;">loading…</div>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">lookback</div><div class="v" id="ne-days">—</div></div>
        <div class="stat"><div class="l">articles</div><div class="v" id="ne-narts">—</div></div>
        <div class="stat"><div class="l">resolved</div><div class="v" id="ne-nres">—</div></div>
        <div class="stat"><div class="l">tickers priced</div><div class="v" id="ne-ntk">—</div></div>
        <div class="stat"><div class="l">ref horizon</div><div class="v" id="ne-ref">—</div></div>
      </div>
      <div style="font-size:12px;color:#dde1e7;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;">Forward return by ai_score band — mean abnormal % (raw in muted)</div>
      <table id="ne-bands" style="font-size:12px;margin-bottom:14px;">
        <thead><tr>
          <th>ai_score band</th><th class="num">n@ref</th>
          <th class="num">1d abn</th><th class="num">3d abn</th><th class="num">5d abn</th>
          <th class="num">ref hit%</th>
        </tr></thead>
        <tbody><tr><td colspan="6" class="muted">loading…</td></tr></tbody>
      </table>
      <div style="font-size:12px;color:#dde1e7;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;">Urgent vs normal — 3d abnormal %</div>
      <div id="ne-urg" style="font-size:12px;color:#8b929d;">—</div>
    </div>

    <!-- ─── Scorer Reliability + Confidence Intervals (new 2026-05-15, agent 4) ─── -->
    <div class="card" id="scrl-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Scorer reliability <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— how far DecisionScorer predictions actually land from reality</span></span>
        <span class="muted" id="scrl-asof" style="font-size:11px;text-transform:none;letter-spacing:normal;">—</span>
      </h2>
      <div class="muted" id="scrl-meta" style="font-size:11px;margin-bottom:10px;">loading…</div>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">directional accuracy</div><div class="v" id="scrl-dir">—</div></div>
        <div class="stat"><div class="l">mean abs error</div><div class="v" id="scrl-mae">—</div></div>
        <div class="stat"><div class="l">90% residual band</div><div class="v" id="scrl-band">—</div></div>
        <div class="stat"><div class="l">replay samples</div><div class="v" id="scrl-n">—</div></div>
      </div>
      <div style="font-size:12px;color:#dde1e7;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;">Held positions — prediction with empirical band</div>
      <table id="scrl-pos" style="font-size:13px;margin-bottom:16px;">
        <thead><tr>
          <th>ticker</th><th class="num">pred 5d</th><th class="num">likely range</th>
          <th>verdict</th><th class="num">band hit %</th><th>trust</th>
        </tr></thead>
        <tbody><tr><td colspan="6" class="muted">loading…</td></tr></tbody>
      </table>
      <div style="font-size:12px;color:#dde1e7;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;">Calibration by prediction band</div>
      <table id="scrl-cal" style="font-size:12px;">
        <thead><tr>
          <th>predicted band</th><th class="num">n</th><th class="num">mean actual</th>
          <th class="num">residual P10/P90</th><th class="num">MAE</th><th class="num">dir. acc.</th>
        </tr></thead>
        <tbody><tr><td colspan="6" class="muted">loading…</td></tr></tbody>
      </table>
    </div>

    <!-- ─── Scorer ⇄ Opus Disagreement (new 2026-05-16, agent 4 feature-dev) ─── -->
    <div class="card" id="dis-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Scorer ⇄ Opus disagreement <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— where the ML safety net and Opus are fighting on held positions</span></span>
        <span class="muted" id="dis-asof" style="font-size:11px;text-transform:none;letter-spacing:normal;">—</span>
      </h2>
      <div class="muted" id="dis-meta" style="font-size:12px;margin-bottom:12px;">loading…</div>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">high conflict</div><div class="v" id="dis-high">—</div></div>
        <div class="stat"><div class="l">medium</div><div class="v" id="dis-med">—</div></div>
        <div class="stat"><div class="l">aligned</div><div class="v" id="dis-aln">—</div></div>
        <div class="stat"><div class="l">positions</div><div class="v" id="dis-n">—</div></div>
      </div>
      <table id="dis-tbl" style="font-size:13px;">
        <thead><tr>
          <th>ticker</th><th>scorer verdict</th><th class="num">pred 5d</th>
          <th>last Opus action</th><th>conflict</th><th>read</th>
        </tr></thead>
        <tbody><tr><td colspan="6" class="muted">loading…</td></tr></tbody>
      </table>
    </div>

    <!-- ─── Portfolio Analytics ─── -->
    <div class="card" style="margin-bottom:18px;">
      <h2>Portfolio analytics</h2>
      <div class="stat-row" style="margin-bottom:18px;">
        <div class="stat"><div class="l">today's P/L</div><div class="v" id="an-daily">—</div></div>
        <div class="stat"><div class="l">max drawdown (all-time)</div><div class="v" id="an-dd">—</div></div>
        <div class="stat"><div class="l">sharpe (ann.)</div><div class="v" id="an-sharpe">—</div></div>
        <div class="stat"><div class="l">win rate</div><div class="v" id="an-winrate">—</div></div>
        <div class="stat"><div class="l">avg winner</div><div class="v" id="an-avgw">—</div></div>
        <div class="stat"><div class="l">avg loser</div><div class="v" id="an-avgl">—</div></div>
        <div class="stat"><div class="l">realized P/L</div><div class="v" id="an-realized">—</div></div>
      </div>
      <div class="stat-row" style="margin-bottom:18px;">
        <div class="stat"><div class="l">profit factor</div><div class="v" id="an-pf">—</div></div>
        <div class="stat"><div class="l">sortino (ann.)</div><div class="v" id="an-sortino">—</div></div>
        <div class="stat"><div class="l">calmar</div><div class="v" id="an-calmar">—</div></div>
        <div class="stat"><div class="l">S&amp;P β</div><div class="v" id="an-beta">—</div></div>
        <div class="stat"><div class="l">S&amp;P corr</div><div class="v" id="an-corr">—</div></div>
        <div class="stat"><div class="l">avg hold</div><div class="v" id="an-hold">—</div></div>
      </div>
      <div style="font-size:13px;color:#dde1e7;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.5px;">Sector exposure</div>
      <div id="an-sector-bar" style="display:flex;height:22px;border-radius:6px;overflow:hidden;background:#0d1117;border:1px solid #1f2126;margin-bottom:6px;"></div>
      <div id="an-sector-legend" style="display:flex;flex-wrap:wrap;gap:14px;font-size:12px;color:#dde1e7;"></div>
    </div>

    <!-- ─── Trade Asymmetry / Behavioural Edge (new 2026-05-16, agent 4) ─── -->
    <div class="card" id="ta-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Behavioural edge <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— payoff ratio, breakeven win-rate, the disposition effect</span></span>
        <span id="ta-verdict" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div class="muted" id="ta-headline" style="font-size:12px;margin-bottom:12px;">loading…</div>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">expectancy / trade</div><div class="v" id="ta-exp">—</div></div>
        <div class="stat"><div class="l">payoff ratio</div><div class="v" id="ta-payoff">—</div></div>
        <div class="stat"><div class="l">win-rate (actual)</div><div class="v" id="ta-wr">—</div></div>
        <div class="stat"><div class="l">breakeven win-rate</div><div class="v" id="ta-be">—</div></div>
        <div class="stat"><div class="l">realized P/L</div><div class="v" id="ta-real">—</div></div>
      </div>
      <div class="stat-row">
        <div class="stat"><div class="l">round-trips (W/L)</div><div class="v" id="ta-n">—</div></div>
        <div class="stat"><div class="l">avg winner</div><div class="v" id="ta-avgw">—</div></div>
        <div class="stat"><div class="l">avg loser</div><div class="v" id="ta-avgl">—</div></div>
        <div class="stat"><div class="l">winner / loser hold</div><div class="v" id="ta-hold">—</div></div>
        <div class="stat"><div class="l">disposition gap</div><div class="v" id="ta-disp">—</div></div>
      </div>
    </div>

    <!-- ─── Loser autopsy (per-closed-losing-trade post-mortem) ─── -->
    <div class="card" id="lautopsy-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Loser autopsy <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— why each closed trade lost: verbatim thesis, hold, failure mode</span></span>
        <span id="lautopsy-state" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div class="muted" id="lautopsy-headline" style="font-size:12px;margin-bottom:12px;">loading…</div>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">total realised loss</div><div class="v" id="lautopsy-total">—</div></div>
        <div class="stat"><div class="l">losing round-trips</div><div class="v" id="lautopsy-n">—</div></div>
        <div class="stat"><div class="l">avg loss</div><div class="v" id="lautopsy-avg">—</div></div>
        <div class="stat"><div class="l">median hold</div><div class="v" id="lautopsy-hold">—</div></div>
        <div class="stat"><div class="l">dominant mode</div><div class="v" id="lautopsy-mode">—</div></div>
      </div>
      <table id="lautopsy-tbl" style="font-size:12px;width:100%;">
        <thead><tr>
          <th>ticker</th><th class="num">P/L $</th><th class="num">P/L %</th>
          <th class="num">hold d</th><th>mode</th><th>opening thesis</th>
        </tr></thead>
        <tbody><tr><td colspan="6" class="muted">loading…</td></tr></tbody>
      </table>
    </div>

    <!-- ─── Winner autopsy (per-closed-winning-trade post-mortem) ─── -->
    <div class="card" id="wautopsy-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Winner autopsy <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— why each closed trade won: verbatim thesis, hold, success mode</span></span>
        <span id="wautopsy-state" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div class="muted" id="wautopsy-headline" style="font-size:12px;margin-bottom:12px;">loading…</div>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">total realised gain</div><div class="v" id="wautopsy-total">—</div></div>
        <div class="stat"><div class="l">winning round-trips</div><div class="v" id="wautopsy-n">—</div></div>
        <div class="stat"><div class="l">avg gain</div><div class="v" id="wautopsy-avg">—</div></div>
        <div class="stat"><div class="l">median hold</div><div class="v" id="wautopsy-hold">—</div></div>
        <div class="stat"><div class="l">dominant mode</div><div class="v" id="wautopsy-mode">—</div></div>
      </div>
      <table id="wautopsy-tbl" style="font-size:12px;width:100%;">
        <thead><tr>
          <th>ticker</th><th class="num">P/L $</th><th class="num">P/L %</th>
          <th class="num">hold d</th><th>mode</th><th>opening thesis</th>
        </tr></thead>
        <tbody><tr><td colspan="6" class="muted">loading…</td></tr></tbody>
      </table>
    </div>

    <!-- ─── Track record in play (per-name closed-trade memory; fed to the prompt) ─── -->
    <div class="card" id="trec-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Track record <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— per-name closed-trade memory: the verbatim history the trader now sees in its own prompt</span></span>
        <span id="trec-state" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div class="muted" id="trec-headline" style="font-size:12px;margin-bottom:12px;">loading…</div>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">names traded</div><div class="v" id="trec-n">—</div></div>
        <div class="stat"><div class="l">closed round-trips</div><div class="v" id="trec-rt">—</div></div>
        <div class="stat"><div class="l">worst name (net)</div><div class="v" id="trec-worst">—</div></div>
        <div class="stat"><div class="l">best name (net)</div><div class="v" id="trec-best">—</div></div>
      </div>
      <table id="trec-tbl" style="font-size:12px;width:100%;">
        <thead><tr>
          <th>ticker</th><th class="num">W-L</th><th class="num">net $</th>
          <th class="num">closed</th><th>last mode</th><th>last opening thesis</th>
        </tr></thead>
        <tbody><tr><td colspan="6" class="muted">loading…</td></tr></tbody>
      </table>
    </div>

    <!-- ─── Concentration honesty (do the held names move together?) ─── -->
    <div class="card" id="pcorr-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Concentration honesty <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— pairwise return ρ &amp; effective independent bets</span></span>
        <span id="pcorr-state" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div class="muted" id="pcorr-headline" style="font-size:12px;margin-bottom:12px;">loading…</div>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">mean pairwise ρ</div><div class="v" id="pcorr-meanrho">—</div></div>
        <div class="stat"><div class="l">effective bets</div><div class="v" id="pcorr-effbets">—</div></div>
        <div class="stat"><div class="l">naive eff. positions</div><div class="v" id="pcorr-effnaive">—</div></div>
        <div class="stat"><div class="l">top weight</div><div class="v" id="pcorr-topw">—</div></div>
        <div class="stat"><div class="l">most-coupled pair</div><div class="v" id="pcorr-maxpair">—</div></div>
      </div>
      <table id="pcorr-tbl" style="font-size:12px;width:100%;">
        <thead><tr><th>pair</th><th class="num">ρ</th></tr></thead>
        <tbody><tr><td colspan="2" class="muted">loading…</td></tr></tbody>
      </table>
    </div>

    <!-- ─── Capital Paralysis & Unlock Ladder (wired 2026-05-16, agent 4) ─── -->
    <div class="card" id="cp-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Capital paralysis <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— the trap, its cost, and the single sale that unlocks it</span></span>
        <span id="cp-state" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div class="muted" id="cp-headline" style="font-size:12px;margin-bottom:12px;">loading…</div>
      <div id="cp-banner" style="display:none;font-size:13px;font-weight:600;padding:8px 12px;border-radius:5px;margin-bottom:12px;background:rgba(255,145,0,0.14);color:#ff9100;border:1px solid rgba(255,145,0,0.4);"></div>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">cash</div><div class="v" id="cp-cash">—</div></div>
        <div class="stat"><div class="l">deployed</div><div class="v" id="cp-dep">—</div></div>
        <div class="stat"><div class="l">can act?</div><div class="v" id="cp-canact">—</div></div>
        <div class="stat"><div class="l">cycles since fill</div><div class="v" id="cp-stuck">—</div></div>
        <div class="stat"><div class="l">alpha bled</div><div class="v" id="cp-bleed">—</div></div>
      </div>
      <div style="font-size:12px;color:#dde1e7;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;">Unlock ladder — desk cut-priority (biggest loser first)</div>
      <table id="cp-ladder" style="font-size:12px;">
        <thead><tr>
          <th>ticker</th><th class="num">weight%</th><th class="num">P/L%</th>
          <th class="num">frees $</th><th class="num">cash if sold alone</th><th>unlocks?</th>
        </tr></thead>
        <tbody><tr><td colspan="6" class="muted">loading…</td></tr></tbody>
      </table>
    </div>

    <!-- ─── Open-Book Alpha — selection vs market (wired 2026-05-16, agent 4) ─── -->
    <div class="card" id="oa-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Open-book alpha <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— how much of the open P&amp;L is selection vs just SPY</span></span>
        <span id="oa-status" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div class="muted" id="oa-headline" style="font-size:12px;margin-bottom:12px;">loading…</div>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">book alpha vs SPY</div><div class="v" id="oa-alpha">—</div></div>
        <div class="stat"><div class="l">net excess $</div><div class="v" id="oa-excess">—</div></div>
        <div class="stat"><div class="l">unrealized $</div><div class="v" id="oa-unreal">—</div></div>
        <div class="stat"><div class="l">SPY-equiv $</div><div class="v" id="oa-spyeq">—</div></div>
        <div class="stat"><div class="l">anchored names</div><div class="v" id="oa-n">—</div></div>
      </div>
      <table id="oa-rows" style="font-size:12px;">
        <thead><tr>
          <th>ticker</th><th class="num">pos %</th><th class="num">SPY %</th>
          <th class="num">alpha %</th><th class="num">excess $</th>
        </tr></thead>
        <tbody><tr><td colspan="5" class="muted">loading…</td></tr></tbody>
      </table>
    </div>

    <!-- ─── Overtrading / re-entry churn (new 2026-05-16, agent 4) ─── -->
    <div class="card" id="churn-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Overtrading &amp; churn <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— how often it re-buys a name it just closed, and how fast</span></span>
        <span id="churn-state" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div class="muted" id="churn-headline" style="font-size:12px;margin-bottom:12px;">loading…</div>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">fast re-entries</div><div class="v" id="churn-reentry">—</div></div>
        <div class="stat"><div class="l">round-trips / day</div><div class="v" id="churn-rtpd">—</div></div>
        <div class="stat"><div class="l">median hold</div><div class="v" id="churn-hold">—</div></div>
        <div class="stat"><div class="l">sub-day trips</div><div class="v" id="churn-subday">—</div></div>
        <div class="stat"><div class="l">loss in &lt;1d trips</div><div class="v" id="churn-lossconc">—</div></div>
      </div>
      <div style="font-size:12px;color:#dde1e7;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;">Fastest same-name re-entries</div>
      <table id="churn-events" style="font-size:12px;">
        <thead><tr>
          <th>ticker</th><th class="num">gap (d)</th><th class="num">prior P/L $</th><th>closed → re-bought</th>
        </tr></thead>
        <tbody><tr><td colspan="4" class="muted">loading…</td></tr></tbody>
      </table>
    </div>

    <!-- ─── Thesis drift — entry rationale vs reality (new 2026-05-16, agent 4) ─── -->
    <div class="card" id="tdrift-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Thesis drift <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— is the reason each position was opened for still true?</span></span>
        <span id="tdrift-state" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div class="muted" id="tdrift-headline" style="font-size:12px;margin-bottom:12px;">loading…</div>
      <table id="tdrift-rows" style="font-size:12px;">
        <thead><tr>
          <th>ticker</th><th>health</th><th class="num">P/L %</th>
          <th class="num">held (d)</th><th>entry rationale → current drift</th>
        </tr></thead>
        <tbody><tr><td colspan="5" class="muted">loading…</td></tr></tbody>
      </table>
    </div>

    <!-- ─── Signal-feed health — is the trader even seeing news? (new 2026-05-16, agent 4) ─── -->
    <div class="card" id="fh-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Signal-feed health <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— is the live trader receiving any news, or flying blind?</span></span>
        <span id="fh-state" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div class="muted" id="fh-headline" style="font-size:12px;margin-bottom:12px;">loading…</div>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">blind streak (0-signal cycles)</div><div class="v" id="fh-streak">—</div></div>
        <div class="stat"><div class="l">newest live article age</div><div class="v" id="fh-age">—</div></div>
        <div class="stat"><div class="l">live articles (2h / 24h)</div><div class="v" id="fh-live">—</div></div>
        <div class="stat"><div class="l">split-brain DB</div><div class="v" id="fh-split">—</div></div>
      </div>
      <div class="muted" id="fh-path" style="font-size:12px;word-break:break-all;">—</div>
    </div>

    <!-- ─── Runner heartbeat — is the trading loop itself alive? (new 2026-05-17) ─── -->
    <div class="card" id="rhb-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Runner heartbeat <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— is the decision loop still cycling, or has it wedged/died?</span></span>
        <span id="rhb-state" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div class="muted" id="rhb-headline" style="font-size:12px;margin-bottom:12px;">loading…</div>
      <div class="stat-row" style="margin-bottom:4px;">
        <div class="stat"><div class="l">since last decision</div><div class="v" id="rhb-age">—</div></div>
        <div class="stat"><div class="l">intervals elapsed</div><div class="v" id="rhb-intervals">—</div></div>
        <div class="stat"><div class="l">expected cadence</div><div class="v" id="rhb-cadence">—</div></div>
        <div class="stat"><div class="l">market</div><div class="v" id="rhb-market">—</div></div>
      </div>
    </div>

    <!-- ─── Decision reliability — true current-regime parse-fail rate (new 2026-05-16, agent 4) ─── -->
    <div class="card" id="dr-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Decision reliability <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— headline NO_DECISION % vs the true post-restart rate</span></span>
        <span id="dr-state" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div class="muted" id="dr-headline" style="font-size:12px;margin-bottom:12px;">loading…</div>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">current-regime fail</div><div class="v" id="dr-cur">—</div></div>
        <div class="stat"><div class="l">headline fail (legacy-incl)</div><div class="v" id="dr-head">—</div></div>
        <div class="stat"><div class="l">current sample / total</div><div class="v" id="dr-n">—</div></div>
        <div class="stat"><div class="l">legacy dead rows</div><div class="v" id="dr-legacy">—</div></div>
        <div class="stat"><div class="l">dead cycles / day</div><div class="v" id="dr-dead">—</div></div>
      </div>
      <div class="muted" id="dr-mode" style="font-size:12px;">—</div>
    </div>

    <!-- ─── ML gate edge — does the 17-feature DecisionScorer beat a one-liner OUT OF SAMPLE? (new 2026-05-18, agent 4 / feature-dev) ─── -->
    <div class="card" id="bc-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>ML gate edge <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— does the DecisionScorer earn its complexity OUT OF SAMPLE, or would a one-line rule do?</span></span>
        <span id="bc-state" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div class="muted" id="bc-headline" style="font-size:12px;margin-bottom:12px;">loading…</div>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">MLP rank-IC (OOS)</div><div class="v" id="bc-mlp">—</div></div>
        <div class="stat"><div class="l">best one-liner</div><div class="v" id="bc-best">—</div></div>
        <div class="stat"><div class="l">IC gap (MLP − best)</div><div class="v" id="bc-gap">—</div></div>
        <div class="stat"><div class="l">OOS pairs / scorer n_train</div><div class="v" id="bc-n">—</div></div>
      </div>
      <div class="muted" id="bc-note" style="font-size:11px;">a read-only honesty diagnostic — the gate stays live at n_train ≥ 500 regardless (invariant #5); the value is <em>knowing</em> whether it is modulating real position sizing on signal or on noise.</div>
    </div>

    <!-- ─── Funded suggestions — which idea is fundable, and the sale that funds it (new 2026-05-16, agent 4) ─── -->
    <div class="card" id="fund-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Funded suggestions <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— which BUY/ADD idea is fundable, and the sale that unlocks it</span></span>
        <span id="fund-state" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div class="muted" id="fund-headline" style="font-size:12px;margin-bottom:12px;">loading…</div>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">actionable ideas</div><div class="v" id="fund-n">—</div></div>
        <div class="stat"><div class="l">funded now</div><div class="v" id="fund-funded">—</div></div>
        <div class="stat"><div class="l">partial (cash + sale)</div><div class="v" id="fund-partial">—</div></div>
        <div class="stat"><div class="l">unlockable via sale</div><div class="v" id="fund-unlock">—</div></div>
        <div class="stat"><div class="l">unfundable</div><div class="v" id="fund-unfund">—</div></div>
        <div class="stat"><div class="l">pairing</div><div class="v" id="fund-pair">—</div></div>
      </div>
      <table id="fund-rows" style="font-size:12px;">
        <thead><tr>
          <th>idea</th><th class="num">conv</th><th class="num">notional $</th>
          <th>fundability</th><th>sell to fund</th><th class="num">frees $</th>
        </tr></thead>
        <tbody><tr><td colspan="6" class="muted">loading…</td></tr></tbody>
      </table>
    </div>

    <!-- ─── Signal follow-through — is the trader using its own news edge? (new 2026-05-16, agent 4) ─── -->
    <div class="card" id="sft-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Signal follow-through <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— did it act on the news it saw, and did acting pay (vs SPY)?</span></span>
        <span id="sft-state" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div class="muted" id="sft-headline" style="font-size:12px;margin-bottom:12px;">loading…</div>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">follow-through</div><div class="v" id="sft-ft">—</div></div>
        <div class="stat"><div class="l">acted / ignored</div><div class="v" id="sft-ai">—</div></div>
        <div class="stat"><div class="l">selection edge (ref)</div><div class="v" id="sft-edge">—</div></div>
        <div class="stat"><div class="l">acted abn% @ref</div><div class="v" id="sft-acted">—</div></div>
        <div class="stat"><div class="l">ignored abn% @ref</div><div class="v" id="sft-ign">—</div></div>
        <div class="stat"><div class="l">resolved / signals</div><div class="v" id="sft-n">—</div></div>
      </div>
      <div class="muted" id="sft-meta" style="font-size:12px;">—</div>
    </div>

    <!-- ─── News source edge — which collector is worth trusting? (new 2026-05-16, agent 4) ─── -->
    <div class="card" id="se-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>News source edge <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— which of ~17 collectors' scored headlines actually precede the move (vs SPY)?</span></span>
        <span id="se-state" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div class="muted" id="se-headline" style="font-size:12px;margin-bottom:12px;">loading…</div>
      <table style="width:100%;border-collapse:collapse;font-size:12px;">
        <thead><tr style="text-align:left;color:#8b929d;">
          <th style="padding:4px 6px;">collector</th>
          <th style="padding:4px 6px;">abn% @ref</th>
          <th style="padding:4px 6px;">hit</th>
          <th style="padding:4px 6px;">resolved</th>
          <th style="padding:4px 6px;">verdict</th>
        </tr></thead>
        <tbody id="se-rows"><tr><td colspan="5" class="muted" style="padding:6px;">—</td></tr></tbody>
      </table>
      <div class="muted" id="se-meta" style="font-size:12px;margin-top:10px;">—</div>
    </div>

    <!-- ─── Behavioural scorecard — verdict-alignment router (new 2026-05-16, agent 4) ─── -->
    <div class="card" id="score-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Behavioural scorecard <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— do the independent behavioural checks agree on a problem? (no grade, just concordance)</span></span>
        <span id="score-state" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div class="muted" id="score-headline" style="font-size:12px;margin-bottom:10px;">loading…</div>
      <div id="score-focus" style="font-size:12px;margin-bottom:8px;"></div>
      <div id="score-concordance" style="font-size:12px;margin-bottom:12px;"></div>
      <table style="font-size:12px;width:100%;">
        <thead><tr style="text-align:left;color:#8b929d;">
          <th style="padding:4px 6px;">check</th><th style="padding:4px 6px;">verdict</th><th style="padding:4px 6px;">what it says</th>
        </tr></thead>
        <tbody id="score-rows"><tr><td colspan="3" class="muted" style="padding:6px;">—</td></tr></tbody>
      </table>
    </div>

    <!-- ─── Sector Pulse ─── -->
    <div class="card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Sector pulse — semis &amp; optical</span>
        <span class="muted" id="sp-asof" style="font-size:11px;text-transform:none;letter-spacing:normal;">—</span>
      </h2>
      <div id="sp-grid" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:8px;">
        <div class="muted">loading…</div>
      </div>
    </div>

    <!-- ─── DRAM / Semis Sector Heatmap ─── -->
    <div class="card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>DRAM / semis heatmap <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— 5d momentum &amp; news pulse</span></span>
        <span class="muted" id="hm-asof" style="font-size:11px;text-transform:none;letter-spacing:normal;">—</span>
      </h2>
      <div class="muted" id="hm-bench" style="font-size:11px;margin-bottom:10px;">SOXX baseline: —</div>
      <div id="hm-grid"><div class="muted">loading…</div></div>
    </div>

    <!-- ─── Deduped News Feed ─── -->
    <div class="card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Deduped signals <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— syndication collapsed, urgency decayed (halflife 4h)</span></span>
        <span class="muted" id="nd-asof" style="font-size:11px;text-transform:none;letter-spacing:normal;">—</span>
      </h2>
      <div class="muted" id="nd-meta" style="font-size:11px;margin-bottom:8px;">—</div>
      <ul id="nd-list" style="margin:0;padding:0;list-style:none;font-size:13px;">
        <li class="muted">loading…</li>
      </ul>
    </div>

    <div class="grid">
      <div class="card">
        <h2>Open positions</h2>
        <div class="table-scroll">
        <table id="pos-tbl">
          <thead><tr>
            <th>ticker</th><th>type</th><th class="num">qty</th>
            <th class="num">avg</th><th class="num">now</th>
            <th class="num">total $</th><th class="num">% port</th>
            <th class="num">P/L</th>
          </tr></thead><tbody></tbody>
        </table>
        </div>
      </div>
      <div class="card">
        <h2>Recent trades</h2>
        <div class="table-scroll">
        <table id="trades-tbl">
          <thead><tr>
            <th>time</th><th>action</th><th>ticker</th>
            <th class="num">qty</th><th class="num">price</th><th>reason</th>
          </tr></thead><tbody></tbody>
        </table>
        </div>
      </div>
    </div>

    <div class="card" style="margin-top:18px;">
      <h2>Decision log</h2>
      <div class="table-scroll">
      <table id="dec-tbl">
        <thead><tr>
          <th>time</th><th>open?</th><th class="num">signals</th>
          <th>action</th><th class="num">equity</th><th>reasoning</th>
        </tr></thead><tbody></tbody>
      </table>
      </div>
    </div>
  </div>

  <!-- ────── Backtests pane ────── -->
  <div id="tab-backtests" class="tab-pane">
    <div class="bt-tabs" style="margin-bottom:14px;">
      <a id="bt-section-runs-link" class="active" onclick="showBtSection('runs')">Backtest Runs</a>
      <a id="bt-section-model-rankings-link" onclick="showBtSection('model-rankings')">🏆 Model Rankings</a>
      <a id="bt-section-persona-rankings-link" onclick="showBtSection('persona-rankings')">🎭 Persona Leaderboard</a>
    </div>
    <div id="bt-section-runs" class="bt-section active">
    <div class="bt-layout">
      <aside class="bt-sidebar card">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
          <h2 style="margin:0;">Runs</h2>
          <div style="display:flex;gap:4px;">
            <button class="bt-btn" onclick="btToggleAll(true)">all</button>
            <button class="bt-btn" onclick="btToggleAll(false)">none</button>
          </div>
        </div>
        <div id="bt-legend"></div>
      </aside>

      <div class="bt-main">
        <div class="card" style="margin-bottom:14px;">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap;">
            <div>
              <h2 style="margin:0 0 4px;">Backtest equity curves</h2>
              <div class="progress-label" id="bt-progress-label">—</div>
            </div>
            <div style="text-align:right;font-size:12px;color:#8b929d;">
              <div id="bt-live-indicator"></div>
              <div id="bt-last-updated">last update: —</div>
            </div>
          </div>
          <div class="progress-wrap" style="margin:8px 0 14px"><div class="progress-bar" id="bt-progress-bar" style="width:0%"></div></div>
          <div class="bt-headline">
            <div class="stat"><div class="l">avg annualized</div><div class="v" id="bt-avg-ann">—</div></div>
            <div class="stat"><div class="l">total % (mean / median)</div><div class="v" id="bt-avg">—</div></div>
            <div class="stat"><div class="l">best</div><div class="v" id="bt-best">—</div></div>
            <div class="stat"><div class="l">worst</div><div class="v" id="bt-worst">—</div></div>
            <div class="stat"><div class="l">beat SPY</div><div class="v" id="bt-beat">—</div></div>
            <div class="stat"><div class="l">win rate</div><div class="v" id="bt-winrate">—</div></div>
            <div class="stat"><div class="l">filtered runs</div><div class="v" id="bt-filtered-count">—</div></div>
          </div>
          <!-- Filter + mode bar -->
          <div style="display:flex;align-items:center;gap:8px;margin:10px 0 6px;flex-wrap:wrap;font-size:12px;">
            <span style="color:var(--text-secondary);">Window:</span>
            <div id="bt-win-filter" style="display:flex;gap:4px;flex-wrap:wrap;">
              <button class="bt-filter-chip active" data-min="0" data-max="99" onclick="setBtWinFilter(this)">All</button>
              <button class="bt-filter-chip" data-min="0" data-max="1.5" onclick="setBtWinFilter(this)">≤1yr</button>
              <button class="bt-filter-chip" data-min="1.5" data-max="2.5" onclick="setBtWinFilter(this)">2yr</button>
              <button class="bt-filter-chip" data-min="2.5" data-max="3.5" onclick="setBtWinFilter(this)">3yr</button>
              <button class="bt-filter-chip" data-min="3.5" data-max="5.5" onclick="setBtWinFilter(this)">4–5yr</button>
              <button class="bt-filter-chip" data-min="5.5" data-max="99" onclick="setBtWinFilter(this)">6–10yr</button>
            </div>
            <div style="display:flex;gap:3px;margin-left:auto;">
              <button id="mode-agg" class="bt-filter-chip active" onclick="setChartMode('aggregate')">Distribution</button>
              <button id="mode-ind" class="bt-filter-chip" onclick="setChartMode('individual')">Individual</button>
            </div>
          </div>
          <!-- Aggregate mode legend / Individual mode limit control -->
          <div id="agg-legend" style="font-size:11px;color:var(--text-secondary);margin-bottom:6px;">
            <span style="display:inline-flex;align-items:center;gap:4px;margin-right:10px;">
              <span style="display:inline-block;width:24px;height:3px;background:#0acdff;border-radius:2px;"></span>Median
            </span>
            <span style="display:inline-flex;align-items:center;gap:4px;margin-right:10px;">
              <span style="display:inline-block;width:24px;height:8px;background:rgba(10,205,255,0.25);border-radius:2px;"></span>P25–P75
            </span>
            <span style="display:inline-flex;align-items:center;gap:4px;margin-right:10px;">
              <span style="display:inline-block;width:24px;height:8px;background:rgba(10,205,255,0.08);border-radius:2px;"></span>P5–P95
            </span>
            <span style="display:inline-flex;align-items:center;gap:4px;margin-right:10px;">
              <span style="display:inline-block;width:24px;height:2px;background:rgba(180,180,180,0.7);border-radius:2px;border-top:2px dashed rgba(180,180,180,0.7);"></span>Actual SPY median
            </span>
            <span id="agg-n-label" style="color:var(--text-muted);"></span>
          </div>
          <div id="ind-controls" style="display:none;font-size:12px;color:var(--text-secondary);margin-bottom:6px;">
            Show last
            <input id="bt-chart-limit" type="range" min="5" max="500" step="5" value="100"
              style="width:80px;cursor:pointer;accent-color:#0acdff;vertical-align:middle;"
              oninput="document.getElementById('bt-chart-limit-val').textContent=this.value; redrawChart()">
            <span id="bt-chart-limit-val">100</span> runs · X = day offset · Y = % return from start
          </div>
          <!-- Main equity chart -->
          <div style="position:relative;height:380px;"><canvas id="bt-chart"></canvas></div>
          <!-- Drawdown sub-chart (aggregate mode only) -->
          <div id="bt-drawdown-wrap" style="position:relative;height:120px;margin-top:8px;">
            <div style="font-size:10px;color:var(--text-muted);margin-bottom:4px;">Max drawdown distribution (% below peak, by day from start)</div>
            <canvas id="bt-dd-chart"></canvas>
          </div>
        </div>

        <!-- ── Multi-dimensional analysis ── -->
        <div class="card" style="margin-bottom:14px;">
          <h2 style="margin:0 0 2px;">Multi-dimensional analysis</h2>
          <div style="color:var(--text-secondary);font-size:12px;margin-bottom:14px;">
            Duration × era × return — three ways to read the same 500+ runs simultaneously.
          </div>

          <!-- Row 1: Scatter (duration vs annualized, colored by era) -->
          <div style="margin-bottom:20px;">
            <div style="font-size:12px;font-weight:600;color:var(--text);margin-bottom:4px;letter-spacing:0.02em;">
              Duration vs annualized return
              <span style="font-weight:400;color:var(--text-muted);font-size:11px;margin-left:6px;">each dot = one run · click to drill in · color = market era</span>
            </div>
            <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px;font-size:11px;" id="bt-era-legend"></div>
            <div style="position:relative;height:320px;"><canvas id="bt-scatter"></canvas></div>
          </div>

          <!-- Row 2: Era × Duration heatmap -->
          <div>
            <div style="font-size:12px;font-weight:600;color:var(--text);margin-bottom:4px;letter-spacing:0.02em;">
              Era × duration performance heatmap
              <span style="font-weight:400;color:var(--text-muted);font-size:11px;margin-left:6px;">avg annualized return % per cell · (n = run count)</span>
            </div>
            <div style="overflow-x:auto;-webkit-overflow-scrolling:touch;">
              <div id="bt-heatmap"></div>
            </div>
          </div>
        </div>

        <div class="card" style="margin-bottom:14px;">
          <h2 style="margin:0 0 4px;">Model progress — return by cycle</h2>
          <div style="color:#8b929d;font-size:12px;margin-bottom:10px;">Best / avg / worst return per cycle of 5 runs. Upward trend = model improving.</div>
          <div style="position:relative;height:220px;"><canvas id="mp-chart"></canvas></div>
        </div>

        <div class="card" id="validation-card" style="margin-bottom:14px;">
          <h2 style="margin:0 0 4px;">Signal Integrity</h2>
          <div style="color:#8b929d;font-size:12px;margin-bottom:14px;">
            Permutation test + label contamination audit. Runs every 10 backtest cycles in the background.
            <br>SIGNIFICANT (p&lt;0.05) means signal time-ordering carries real predictive value, not random noise.
          </div>
          <div style="display:flex;gap:24px;flex-wrap:wrap;">
            <div style="min-width:220px;">
              <div style="font-size:11px;color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.05em;">Permutation Test</div>
              <div id="val-perm-verdict" style="font-weight:600;font-size:18px;margin-top:4px;">—</div>
              <div style="font-size:12px;color:var(--text-secondary);margin-top:2px;">
                <span id="val-perm-pvalue">p=—</span> · <span id="val-perm-zscore">z=—</span>
              </div>
              <div id="val-perm-original" style="font-size:12px;margin-top:6px;"></div>
              <div id="val-perm-shuffled" style="font-size:12px;color:var(--text-secondary);"></div>
            </div>
            <div style="min-width:220px;">
              <div style="font-size:11px;color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.05em;">Label Contamination</div>
              <div id="val-contam-rate" style="font-weight:600;font-size:18px;margin-top:4px;">—</div>
              <div style="font-size:12px;color:var(--text-secondary);margin-top:6px;">
                High = Claude labels carry hindsight<br>(retroactively-collected articles)
              </div>
              <div id="val-contam-detail" style="font-size:12px;margin-top:6px;color:var(--text-secondary);"></div>
            </div>
            <div style="min-width:220px;">
              <div style="font-size:11px;color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.05em;">Last Validation</div>
              <div id="val-last-cycle" style="font-weight:600;font-size:18px;margin-top:4px;">—</div>
              <div id="val-last-window" style="font-size:12px;color:var(--text-secondary);margin-top:2px;"></div>
              <div id="val-last-when" style="font-size:11px;color:var(--text-muted);margin-top:6px;"></div>
            </div>
          </div>
        </div>

        <div class="card" style="margin-bottom:14px;">
          <h2>Runs table — click a row to highlight</h2>
          <div style="overflow-x:auto;-webkit-overflow-scrolling:touch;">
          <table id="bt-tbl" class="sortable">
            <thead><tr>
              <th data-k="run_id">#</th>
              <th data-k="status">status</th>
              <th data-k="total_return_pct" class="num">total %</th>
              <th data-k="annualized_return_pct" class="num">ann. %/yr</th>
              <th data-k="vs_spy_pct" class="num">vs SPY</th>
              <th data-k="start_date">window</th>
              <th data-k="duration_days" class="num">dur.</th>
              <th data-k="n_trades" class="num">trades</th>
              <th data-k="n_decisions" class="num">signals</th>
            </tr></thead><tbody></tbody>
          </table>
          </div>
        </div>

        <div class="card" id="bt-detail" style="display:none;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <h2 style="margin:0;">Run <span id="bt-detail-id">—</span> detail</h2>
            <button class="bt-btn" onclick="closeDetail()">close</button>
          </div>
          <div id="bt-detail-meta" class="muted" style="font-size:13px;margin-bottom:12px;"></div>
          <div class="bt-tabs">
            <a id="bt-tab-trades-link" class="active" onclick="showBtSubtab('trades')">Trades</a>
            <a id="bt-tab-decisions-link" onclick="showBtSubtab('decisions')">Decisions</a>
          </div>
          <div id="bt-tab-trades" class="bt-subpane active">
            <table id="bt-trades-tbl">
              <thead><tr>
                <th>date</th><th>action</th><th>ticker</th>
                <th class="num">qty</th><th class="num">price</th>
                <th class="num">value</th><th>reason</th>
              </tr></thead><tbody></tbody>
            </table>
          </div>
          <div id="bt-tab-decisions" class="bt-subpane">
            <table id="bt-decisions-tbl">
              <thead><tr>
                <th>date</th><th>action</th><th>ticker</th>
                <th>status</th><th>detail</th><th class="num">portfolio $</th>
              </tr></thead><tbody></tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
    </div><!-- /#bt-section-runs -->

    <!-- ── Model Rankings section ── -->
    <div id="bt-section-model-rankings" class="bt-section" style="display:none;">
      <div class="card" style="margin-bottom:14px;">
        <h2 style="margin:0 0 4px;">Model rankings</h2>
        <div style="color:var(--text-secondary);font-size:12px;margin-bottom:14px;">
          Aggregated backtest stats per decision model. Ranked by average total return %.
        </div>
        <div id="model-rankings-loading" style="padding:14px 4px;color:var(--text-muted);font-size:13px;">Loading rankings…</div>
        <div style="overflow-x:auto;-webkit-overflow-scrolling:touch;">
          <table id="model-rankings-table" style="display:none;width:100%;border-collapse:collapse;font-size:13px;">
            <thead>
              <tr id="model-rankings-header"></tr>
            </thead>
            <tbody id="model-rankings-body"></tbody>
          </table>
        </div>
        <div style="margin-top:16px;display:flex;flex-wrap:wrap;align-items:center;gap:10px;">
          <label style="color:var(--text-secondary);font-size:13px;">Run backtest with model:
            <select id="run-model-select" style="margin-left:8px;background:var(--bg-elevated);color:var(--text);border:1px solid var(--border);padding:4px 8px;border-radius:3px;">
              <option value="ml_quant">ML+Quant (deterministic)</option>
              <option value="claude-opus-4-7">Claude Opus 4.7</option>
              <option value="hf/deepseek-ai/DeepSeek-R1">DeepSeek R1</option>
              <option value="hf/meta-llama/Llama-3.3-70B-Instruct">Llama 3.3 70B</option>
              <option value="hf/Qwen/Qwen3-32B">Qwen3 32B</option>
            </select>
          </label>
          <button class="bt-btn" onclick="triggerBacktestWithModel()">▶ Run Backtest</button>
          <span id="run-model-status" style="color:var(--text-muted);font-size:12px;"></span>
        </div>
      </div>
    </div><!-- /#bt-section-model-rankings -->

    <!-- ── Persona Leaderboard section ── -->
    <div id="bt-section-persona-rankings" class="bt-section" style="display:none;">
      <div class="card" style="margin-bottom:14px;">
        <h2 style="margin:0 0 4px;">Persona leaderboard</h2>
        <div style="color:var(--text-secondary);font-size:12px;margin-bottom:4px;">
          Each backtest run is one of 10 trading styles (Value, Momentum, Contrarian,
          Global Macro, GARP, Quant, Sector Rotator, Small/Mid Cap, ESG, Pure Speculator).
          This aggregates the backtest history by <em>style</em> — which approach actually
          carries repeatable alpha. Ranked by median vs-SPY (the honest central read on a
          leveraged window); per-persona verdict EDGE / FLAT / DRAG / INSUFFICIENT.
        </div>
        <div id="persona-rankings-headline" style="font-size:13px;margin-bottom:12px;font-weight:600;"></div>
        <div id="persona-rankings-loading" style="padding:14px 4px;color:var(--text-muted);font-size:13px;">Loading personas…</div>
        <div style="overflow-x:auto;-webkit-overflow-scrolling:touch;">
          <table id="persona-rankings-table" style="display:none;width:100%;border-collapse:collapse;font-size:13px;">
            <thead>
              <tr id="persona-rankings-header"></tr>
            </thead>
            <tbody id="persona-rankings-body"></tbody>
          </table>
        </div>
        <div id="persona-rankings-hint" style="color:var(--text-muted);font-size:12px;margin-top:12px;line-height:1.5;"></div>
      </div>
    </div><!-- /#bt-section-persona-rankings -->
  </div>

<script>
const fmt = (n, d=2) => (n == null ? "—" : Number(n).toLocaleString(undefined, {minimumFractionDigits:d, maximumFractionDigits:d}));
const dollar = n => (n == null ? "—" : "$" + fmt(n));
// Human-readable LOCAL time. mode: undefined→"May 18 13:11",
// "sec"→"…13:11:02", "time"→"13:11". API timestamps carry +00:00; a
// naive (offset-less) string is treated as UTC. Bad/empty input → "—"
// (the dashboard's universal no-value marker) so callers can drop their
// own `? … : "—"` ternaries.
function fmtTs(s, mode) {
  if (s == null || s === "") return "—";
  const str = String(s);
  const d = new Date(/([zZ]|[+-]\d\d:?\d\d)$/.test(str) ? str : str + "Z");
  if (isNaN(d.getTime())) return str;
  if (mode === "sec")  return d.toLocaleString(undefined, {month:"short", day:"numeric", hour:"2-digit", minute:"2-digit", second:"2-digit", hour12:false});
  if (mode === "time") return d.toLocaleTimeString(undefined, {hour:"2-digit", minute:"2-digit", hour12:false});
  return d.toLocaleString(undefined, {month:"short", day:"numeric", hour:"2-digit", minute:"2-digit", hour12:false});
}
const dt = s => fmtTs(s);

const INITIAL_TAB = "{{ initial_tab }}";
const API_PREFIX = "{{ api_prefix }}";
const RUN_COLORS = [
  "#00d4ff","#ff6b35","#7fff00","#ff3cac","#ffd700",
  "#00ff9f","#ff1744","#e040fb","#40c4ff","#ff9100"
];
const SPY_COLOR = "#888888";

function showTab(name) {
  document.querySelectorAll(".tab-pane").forEach(el => el.classList.remove("active"));
  document.querySelectorAll("nav.tabs a").forEach(el => el.classList.remove("active"));
  document.getElementById("tab-" + name).classList.add("active");
  document.getElementById("tab-" + name + "-link").classList.add("active");
  if (name === "backtests" && !btLoaded) loadBacktests();
  // Update URL without reload
  if (history.replaceState) history.replaceState(null, "", name === "trader" ? "/" : "/backtests");
}

// ───────── Backtests section sub-tabs (Runs / Model Rankings) ─────────
let modelRankingsLoaded = false;
let personaRankingsLoaded = false;
function showBtSection(name) {
  document.querySelectorAll("#tab-backtests > .bt-section").forEach(el => {
    el.classList.remove("active");
    el.style.display = "none";
  });
  document.querySelectorAll("#tab-backtests > .bt-tabs > a").forEach(el => el.classList.remove("active"));
  const pane = document.getElementById("bt-section-" + name);
  const link = document.getElementById("bt-section-" + name + "-link");
  if (pane) { pane.classList.add("active"); pane.style.display = "block"; }
  if (link) link.classList.add("active");
  if (name === "model-rankings" && !modelRankingsLoaded) loadModelRankings();
  if (name === "persona-rankings" && !personaRankingsLoaded) loadPersonaRankings();
}

async function loadModelRankings() {
  const loading = document.getElementById("model-rankings-loading");
  const table = document.getElementById("model-rankings-table");
  if (!loading || !table) return;
  loading.style.display = "block";
  loading.textContent = "Loading rankings…";
  table.style.display = "none";
  function pctFmt(v) {
    if (v === null || v === undefined) return "—";
    return (v > 0 ? "+" : "") + v + "%";
  }
  function colorStyle(v) {
    if (v === null || v === undefined) return "";
    const n = parseFloat(v);
    if (n > 0) return "color:#00c896";
    if (n < 0) return "color:#ff4455";
    return "";
  }
  try {
    const data = await fetch(API_PREFIX + "/api/model-rankings").then(r => r.json());
    const medals = ["🥇", "🥈", "🥉"];
    const cols = [
      {h: "Rank",        fn: (m, i) => medals[i] || (i + 1) + ".",   colored: false},
      {h: "Model",       fn: (m) => m.display_name || m.model_id,     colored: false},
      {h: "Runs",        fn: (m) => m.runs,                           colored: false},
      {h: "Avg Return",  fn: (m) => pctFmt(m.avg_return_pct),         colored: true, raw: (m) => m.avg_return_pct},
      {h: "Best Return", fn: (m) => pctFmt(m.best_return_pct),        colored: true, raw: (m) => m.best_return_pct},
      {h: "Median",      fn: (m) => pctFmt(m.median_return_pct),      colored: true, raw: (m) => m.median_return_pct},
      {h: "vs SPY",      fn: (m) => pctFmt(m.avg_vs_spy_pct),         colored: true, raw: (m) => m.avg_vs_spy_pct},
      {h: "Win Rate",    fn: (m) => (m.win_rate_pct == null ? "—" : m.win_rate_pct + "%"), colored: false},
      {h: "Avg Trades",  fn: (m) => (m.avg_trades == null ? "—" : m.avg_trades),           colored: false},
    ];
    document.getElementById("model-rankings-header").innerHTML =
      cols.map(c => '<th style="text-align:left;padding:8px 12px;border-bottom:1px solid var(--border);color:var(--text-secondary);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:0.05em;">' + c.h + "</th>").join("");
    const models = data.models || [];
    if (!models.length) {
      loading.textContent = "No completed backtest runs yet.";
      return;
    }
    document.getElementById("model-rankings-body").innerHTML = models.map((m, i) => {
      const safeId = String(m.model_id || "").replace(/'/g, "\\'");
      const cells = cols.map(c => {
        const val = c.fn(m, i);
        const style = c.colored ? colorStyle(c.raw(m)) : "";
        return '<td style="padding:8px 12px;border-bottom:1px solid var(--border);' + style + '">' + val + "</td>";
      }).join("");
      return '<tr class="bt-row" style="cursor:pointer" onclick="filterByModel(\'' + safeId + '\')">' + cells + "</tr>";
    }).join("");
    loading.style.display = "none";
    table.style.display = "table";
    modelRankingsLoaded = true;
  } catch (e) {
    loading.textContent = "Failed to load rankings: " + (e && e.message ? e.message : e);
  }
}

const PERSONA_VERDICT_STYLE = {
  EDGE:         "color:#00c896;font-weight:600",
  FLAT:         "color:#ffd700",
  DRAG:         "color:#ff4455;font-weight:600",
  INSUFFICIENT: "color:var(--text-muted)",
};
const PERSONA_OVERALL_STYLE = {
  HEALTHY:           "color:#00c896",
  HAS_DRAG_PERSONA:  "color:#ff4455",
  INSUFFICIENT_DATA: "color:var(--text-muted)",
};

async function loadPersonaRankings() {
  const loading  = document.getElementById("persona-rankings-loading");
  const table    = document.getElementById("persona-rankings-table");
  const headline = document.getElementById("persona-rankings-headline");
  const hintEl   = document.getElementById("persona-rankings-hint");
  if (!loading || !table) return;
  loading.style.display = "block";
  loading.textContent = "Loading personas…";
  table.style.display = "none";
  function ppFmt(v) {
    if (v === null || v === undefined) return "—";
    return (v > 0 ? "+" : "") + Number(v).toFixed(1) + "pp";
  }
  function numFmt(v, d) {
    if (v === null || v === undefined) return "—";
    return Number(v).toFixed(d == null ? 2 : d);
  }
  function colorStyle(v) {
    if (v === null || v === undefined) return "";
    const n = parseFloat(v);
    if (n > 0) return "color:#00c896";
    if (n < 0) return "color:#ff4455";
    return "";
  }
  try {
    const data = await fetch(API_PREFIX + "/api/persona-leaderboard").then(r => r.json());
    if (headline) {
      const v = data.verdict || "";
      headline.textContent = v + (data.n_runs != null ? "  ·  " + data.n_runs + " complete runs" : "");
      headline.style.cssText = PERSONA_OVERALL_STYLE[v] || "color:var(--text)";
    }
    if (hintEl) hintEl.textContent = data.hint || "";
    const medals = ["🥇", "🥈", "🥉"];
    const cols = [
      {h: "Rank",          fn: (p, i) => (medals[i] || (i + 1) + "."),  colored: false},
      {h: "Persona",       fn: (p) => p.persona,                        colored: false},
      {h: "Runs",          fn: (p) => p.n,                              colored: false},
      {h: "Median vs SPY", fn: (p) => ppFmt(p.median_vs_spy),           colored: true, raw: (p) => p.median_vs_spy},
      {h: "Mean vs SPY",   fn: (p) => ppFmt(p.mean_vs_spy),             colored: true, raw: (p) => p.mean_vs_spy},
      {h: "Win Rate",      fn: (p) => (p.win_rate == null ? "—" : Math.round(p.win_rate * 100) + "%"), colored: false},
      {h: "Median Return", fn: (p) => (p.median_return == null ? "—" : ppFmt(p.median_return).replace("pp", "%")), colored: true, raw: (p) => p.median_return},
      {h: "Sharpe",        fn: (p) => numFmt(p.median_sharpe),          colored: true, raw: (p) => p.median_sharpe},
      {h: "Max DD",        fn: (p) => (p.median_max_drawdown_pct == null ? "—" : numFmt(p.median_max_drawdown_pct, 1) + "%"), colored: false},
      {h: "% Underwater",  fn: (p) => (p.median_pct_time_underwater == null ? "—" : Math.round(p.median_pct_time_underwater) + "%"), colored: false},
      {h: "Verdict",       fn: (p) => p.verdict,                        colored: false, verdict: true},
    ];
    document.getElementById("persona-rankings-header").innerHTML =
      cols.map(c => '<th style="text-align:left;padding:8px 12px;border-bottom:1px solid var(--border);color:var(--text-secondary);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:0.05em;">' + c.h + "</th>").join("");
    const personas = data.leaderboard || [];
    if (!personas.length) {
      loading.textContent = (data.hint || "No completed backtest runs yet.");
      return;
    }
    document.getElementById("persona-rankings-body").innerHTML = personas.map((p, i) => {
      const cells = cols.map(c => {
        const val = c.fn(p, i);
        let style = c.colored ? colorStyle(c.raw(p)) : "";
        if (c.verdict) style = PERSONA_VERDICT_STYLE[p.verdict] || "";
        return '<td style="padding:8px 12px;border-bottom:1px solid var(--border);' + style + '">' + val + "</td>";
      }).join("");
      return "<tr>" + cells + "</tr>";
    }).join("");
    loading.style.display = "none";
    table.style.display = "table";
    personaRankingsLoaded = true;
  } catch (e) {
    loading.textContent = "Failed to load personas: " + (e && e.message ? e.message : e);
  }
}

function triggerBacktestWithModel() {
  const sel = document.getElementById("run-model-select");
  const status = document.getElementById("run-model-status");
  if (!sel || !status) return;
  const model = sel.value;
  status.textContent = "To run from shell: python3 run_continuous_backtests.py --model " + model;
}

function filterByModel(modelId) {
  // Switch back to the runs section; existing runs table has no text filter,
  // so we just surface the selection to the user.
  showBtSection("runs");
  const status = document.getElementById("run-model-status");
  if (status) status.textContent = "Selected model: " + modelId;
}

// ───────── Trader pane ─────────
let chart;
let ddChart;
let eqRange = "all";   // "all" | "24h" | "7d"
let _lastEquity = [];  // cache for range filtering
let _lastTrades = [];

function setEqRange(r) {
  eqRange = r;
  ["all","24h","7d"].forEach(k => {
    const el = document.getElementById("eq-range-"+k);
    if (el) el.classList.toggle("active", k === r);
  });
  drawEquityChart(_lastEquity, _lastTrades);
}

function _filterEqByRange(eq) {
  if (eqRange === "all" || !eq.length) return eq;
  const cutMs = eqRange === "24h" ? 86400000 : 7*86400000;
  const cutoff = new Date(Date.now() - cutMs).toISOString();
  return eq.filter(p => p.timestamp >= cutoff);
}

function drawEquityChart(eq, trades) {
  const filtered = _filterEqByRange(eq);
  if (!filtered.length) return;

  // Normalize: portfolio and SPY both as % from first point in filtered range
  const baseVal = filtered[0].total_value || 1000;
  const baseSpy = filtered[0].sp500_price || 1;
  const labels = filtered.map(p => p.timestamp.replace("T"," ").slice(0,16));
  const portPct = filtered.map(p => ((p.total_value / baseVal) - 1) * 100);
  const spyPct  = filtered.map(p => p.sp500_price ? ((p.sp500_price / baseSpy) - 1) * 100 : null);
  const cashPct = filtered.map(p => p.cash != null ? (p.cash / p.total_value) * 100 : null);

  // Rolling drawdown from peak
  let peak = 0;
  const ddPct = portPct.map(v => {
    if (v > peak) peak = v;
    return peak > 0 ? v - peak : (v < 0 ? v : 0);
  });

  // Trade markers — find trades within the filtered time range
  const t0 = filtered[0].timestamp;
  const t1 = filtered[filtered.length-1].timestamp;
  const visibleTrades = (trades||[]).filter(t => t.timestamp >= t0 && t.timestamp <= t1);

  // Map each trade to its nearest label index. Markers go into sparse
  // arrays aligned 1:1 with `labels` (null elsewhere) — NOT a scatter
  // {x,y} dataset: on a Chart.js category axis a scatter point is placed
  // by its position in the dataset array, not its x value, which
  // collapsed every marker onto the chart's far-left edge.
  const buyMarks  = new Array(labels.length).fill(null);
  const sellMarks = new Array(labels.length).fill(null);
  visibleTrades.forEach(tr => {
    const ts = tr.timestamp.replace("T"," ").slice(0,16);
    let idx = labels.indexOf(ts);
    if (idx < 0) {
      // Find chronologically nearest label (labels are sorted ascending ISO strings)
      const tMs = +new Date(ts.replace(' ', 'T') + ':00Z');
      let best = 0, bestDist = Infinity;
      for (let i = 0; i < labels.length; i++) {
        const d = Math.abs(+new Date(labels[i].replace(' ', 'T') + ':00Z') - tMs);
        if (d < bestDist) { bestDist = d; best = i; }
        else if (d > bestDist) break;  // sorted: getting farther, stop early
      }
      idx = best;
    }
    const base = portPct[idx] ?? 0;
    if (tr.action && tr.action.startsWith("BUY")) buyMarks[idx]  = base + 0.5;
    else                                          sellMarks[idx] = base - 0.5;
  });

  // Summary stats
  const finalPct = portPct[portPct.length-1];
  const spyFinalPct = spyPct[spyPct.length-1];
  const maxDd = Math.min(...ddPct);
  const deployed = 100 - (cashPct[cashPct.length-1] || 0);
  const vsSpyEl = document.getElementById("vs-spy-live");
  if (vsSpyEl && spyFinalPct != null) {
    const vs = finalPct - spyFinalPct;
    vsSpyEl.textContent = (vs>=0?"+":"")+vs.toFixed(2)+"% vs SPY";
    vsSpyEl.className = "v " + (vs>=0?"pos":"neg");
  }
  const ddEl = document.getElementById("live-maxdd");
  if (ddEl) { ddEl.textContent = maxDd.toFixed(2)+"%"; ddEl.className = "v " + (maxDd < -5 ? "neg" : ""); }
  const depEl = document.getElementById("live-deployed");
  if (depEl) depEl.textContent = deployed.toFixed(1)+"%";

  const mkMarkers = (arr, color, label, down) => ({
    label,
    data: arr,
    backgroundColor: color,
    borderColor: color,
    pointRadius:      arr.map(v => v == null ? 0 : 7),
    pointHoverRadius: arr.map(v => v == null ? 0 : 9),
    pointStyle: "triangle",
    rotation: down ? 180 : 0,
    showLine: false,
    spanGaps: false,
    order: 0,
  });

  const datasets = [
    {
      label: "Portfolio %",
      data: portPct,
      borderColor: "#0acdff",
      backgroundColor: "rgba(10,205,255,0.07)",
      fill: true, tension: 0.15, borderWidth: 2,
      pointRadius: 0, pointHoverRadius: 4, order: 2,
    },
    {
      label: "SPY %",
      data: spyPct,
      borderColor: "rgba(255,183,77,0.7)",
      backgroundColor: "transparent",
      borderDash: [5,4], borderWidth: 1.5,
      pointRadius: 0, fill: false, order: 3,
    },
  ];
  if (buyMarks.some(v => v != null))  datasets.push(mkMarkers(buyMarks,  "#00c896", "Buy ↑", false));
  if (sellMarks.some(v => v != null)) datasets.push(mkMarkers(sellMarks, "#ff4455", "Sell ↓", true));

  if (!chart) {
    chart = new Chart(document.getElementById("eq"), {
      type: "line",
      data: { labels, datasets },
      options: {
        animation: false,
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "rgba(15,20,28,0.95)", borderColor: "#2a3a4f", borderWidth: 1,
            titleColor: "#dde1e7", bodyColor: "#dde1e7", padding: 8, boxPadding: 3,
            callbacks: {
              label: ctx => {
                if (ctx.dataset.type === "scatter" || ctx.dataset.showLine === false) return null;
                const v = ctx.parsed.y;
                return `${ctx.dataset.label}: ${v>=0?"+":""}${v.toFixed(2)}%`;
              },
            },
          },
        },
        scales: {
          x: { ticks: { color: "#8b929d", maxTicksLimit: 8 }, grid: { color: "#1f2126" }},
          y: {
            ticks: { color: "#dde1e7", callback: v => (v>=0?"+":"")+v.toFixed(1)+"%" },
            grid: { color: "#1f2126" },
          },
        },
      },
    });
  } else {
    chart.data.labels = labels;
    chart.data.datasets = datasets;
    chart.update("none");
  }

  // Drawdown sub-chart
  if (!ddChart) {
    ddChart = new Chart(document.getElementById("eq-dd"), {
      type: "line",
      data: { labels, datasets: [{
        label: "Drawdown %",
        data: ddPct,
        borderColor: "rgba(239,83,80,0.7)",
        backgroundColor: "rgba(239,83,80,0.12)",
        fill: true, tension: 0.15, borderWidth: 1.5,
        pointRadius: 0,
      }]},
      options: {
        animation: false,
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
        scales: {
          x: { display: false },
          y: {
            ticks: { color: "#8b929d", font: { size: 9 }, callback: v => v.toFixed(1)+"%" },
            grid: { color: "#1f2126" }, max: 0,
          },
        },
      },
    });
  } else {
    ddChart.data.labels = labels;
    ddChart.data.datasets[0].data = ddPct;
    ddChart.update("none");
  }
}

async function refresh() {
  // /api/state is the main page's lifeline. It has 500'd transiently in
  // prod (the get_portfolio() shared-connection note in store.py) and, once
  // SWR-cached, its cold path returns a valid-shaped {"warming":true}
  // placeholder. The old code did `r.portfolio.total_value` with no guard,
  // so any of those bodies threw an unhandled TypeError mid-tick and the
  // whole page froze with no visible reason until the next poll. Every
  // *other* refresh* fn already guards (`if(!r.ok)return` / `if(!a||a.error)
  // return`); this — the most important fetch — was the lone outlier. Match
  // the established pattern: on a bad/warming/error body, show "updating…"
  // and bail; the 15s setInterval self-heals on the next tick.
  let r;
  try {
    r = await fetch(API_PREFIX + "/api/state").then(x => x.json());
  } catch (e) {
    const _hb = document.getElementById("hb");
    if (_hb) _hb.textContent = "updating…";
    return;
  }
  if (!r || !r.portfolio || r.warming || r.error) {
    const _hb = document.getElementById("hb");
    if (_hb) _hb.textContent = "updating…";
    return;
  }
  document.getElementById("hb").textContent = "updated " + fmtTs(r.now, "sec");
  document.getElementById("tv").textContent = dollar(r.portfolio.total_value);
  document.getElementById("cash").textContent = dollar(r.portfolio.cash);
  const startVal = (r.equity && r.equity[0]) ? r.equity[0].total_value : 1000;
  const pl = r.portfolio.total_value - startVal;
  const plPct = (r.portfolio.total_value / startVal - 1) * 100;
  const plEl = document.getElementById("pl");
  plEl.textContent = (plPct >= 0 ? "+" : "") + plPct.toFixed(2) + "%";
  plEl.className = "v " + (plPct >= 0 ? "pos" : "neg");
  const _spEl = document.getElementById("sp"); if (_spEl) _spEl.textContent = r.sp500 ? fmt(r.sp500) : "—";
  _lastEquity = r.equity || [];
  _lastTrades = r.all_trades || r.trades || [];
  drawEquityChart(_lastEquity, _lastTrades);

  const posBody = document.querySelector("#pos-tbl tbody");
  const portTotal = r.portfolio.total_value || 0;
  posBody.innerHTML = r.positions.map(p => {
    const cls = (p.unrealized_pl || 0) >= 0 ? "pos" : "neg";
    const label = p.type === "stock" ? p.type :
                  `${p.type.toUpperCase()} ${p.strike}/${p.expiry}`;
    const mult = (p.type === "call" || p.type === "put") ? 100 : 1;
    const totalVal = (p.current_price || 0) * (p.qty || 0) * mult;
    const pctPort = portTotal > 0 ? (totalVal / portTotal * 100) : 0;
    return `<tr><td>${p.ticker}</td><td>${label}</td>
      <td class="num">${fmt(p.qty,4)}</td>
      <td class="num">${fmt(p.avg_cost)}</td>
      <td class="num">${fmt(p.current_price)}</td>
      <td class="num">${dollar(totalVal)}</td>
      <td class="num">${fmt(pctPort,1)}%</td>
      <td class="num ${cls}">${fmt(p.unrealized_pl)}</td></tr>`;
  }).join("") || `<tr><td colspan="8" class="muted">no positions</td></tr>`;

  const trBody = document.querySelector("#trades-tbl tbody");
  trBody.innerHTML = r.trades.map(t => {
    const cls = t.action.startsWith("SELL") ? "sell" : "buy";
    return `<tr><td>${dt(t.timestamp)}</td>
      <td><span class="pill ${cls}">${t.action}</span></td>
      <td>${t.ticker}</td>
      <td class="num">${fmt(t.qty,4)}</td>
      <td class="num">${fmt(t.price)}</td>
      <td class="muted">${(t.reason||"").slice(0,80)}</td></tr>`;
  }).join("") || `<tr><td colspan="6" class="muted">no trades</td></tr>`;

  const dBody = document.querySelector("#dec-tbl tbody");
  dBody.innerHTML = r.decisions.map(d => {
    let reason = "";
    try {
      const j = JSON.parse(d.reasoning || "{}");
      reason = (j.decision && j.decision.reasoning) || j.detail || "";
    } catch (_) { reason = d.reasoning || ""; }
    return `<tr><td>${dt(d.timestamp)}</td>
      <td>${d.market_open ? "yes" : "no"}</td>
      <td class="num">${d.signal_count}</td>
      <td>${(d.action_taken||"").slice(0,40)}</td>
      <td class="num">${fmt(d.portfolio_value)}</td>
      <td class="muted">${reason.slice(0,140)}</td></tr>`;
  }).join("") || `<tr><td colspan="6" class="muted">no decisions yet</td></tr>`;

}

// ───────── Backtests pane ─────────
let btLoaded = false;
let btChart;
let btScatter;
let btRuns = [];
let btPollTimer = null;
let btSelectedRunId = null;
let btHiddenRuns = new Set();
let btLastUpdated = null;
let btSortKey = "run_id", btSortDir = -1;
let btDetailSubtab = "trades";
let btSpyBaseline = null;
let btCurvesCache = {};              // run_id → normalized curve array
let btWinMinYears = 0, btWinMaxYears = 99; // window-length filter

function btRunColor(runId, idx) { return RUN_COLORS[idx % RUN_COLORS.length]; }
function hexToRgba(hex, a) {
  const h = hex.replace("#","");
  const r = parseInt(h.slice(0,2),16), g = parseInt(h.slice(2,4),16), b = parseInt(h.slice(4,6),16);
  return `rgba(${r},${g},${b},${a})`;
}

// Window-length filter chip handler
function setBtWinFilter(el) {
  document.querySelectorAll("#bt-win-filter .bt-filter-chip").forEach(b => b.classList.remove("active"));
  el.classList.add("active");
  btWinMinYears = parseFloat(el.dataset.min);
  btWinMaxYears = parseFloat(el.dataset.max);
  // Scatter + heatmap always show all runs (for cross-dimension visibility),
  // only the equity curve chart and table respect the filter.
  renderLegend();
  renderTable();
  redrawChart();
}

// Returns runs passing the current window-length filter
function filteredRuns() {
  return btRuns.filter(r => {
    if (!r.duration_days) return btWinMinYears === 0;
    const yrs = r.duration_days / 365.25;
    return yrs >= btWinMinYears && yrs < btWinMaxYears;
  });
}

// Lazily fetch curves for an array of run_ids not yet in cache.
// After fetching, calls callback().
async function ensureCurves(runIds, callback) {
  const missing = runIds.filter(id => !btCurvesCache[id]);
  if (missing.length) {
    // Server caps at 100 run_ids per request; chunk to stay under the limit.
    const CHUNK = 100;
    const chunks = [];
    for (let i = 0; i < missing.length; i += CHUNK) {
      chunks.push(missing.slice(i, i + CHUNK));
    }
    try {
      const results = await Promise.all(chunks.map(chunk =>
        fetch(API_PREFIX + "/api/backtests/curves?run_ids=" + chunk.join(","))
          .then(r => r.json())
      ));
      results.forEach(data => {
        Object.keys(data).forEach(k => { btCurvesCache[parseInt(k)] = data[k]; });
      });
    } catch(e) { console.error("curves fetch:", e); }
  }
  if (callback) callback();
}

let mpChart;
async function loadModelProgress() {
  try {
    const d = await fetch(API_PREFIX + "/api/model-progress").then(r => r.json());
    const cycles = d.cycles || [];
    if (!cycles.length) return;
    // cycle label is now a run_id range string e.g. "#1491-#1495"
    const labels = cycles.map(c => c.cycle);
    const best  = cycles.map(c => c.best);
    const avg   = cycles.map(c => c.avg);
    const worst = cycles.map(c => c.worst);
    const totalRuns = d.total_runs || cycles.length * 5;
    const ctx = document.getElementById("mp-chart");
    if (!ctx) return;
    // Update subtitle with total run count
    const sub = ctx.closest(".card")?.querySelector("div.sub,div[style*='78909c']");
    if (sub) sub.textContent = `Best / avg / worst return per cycle of 5 runs (${totalRuns} total). Upward trend = model improving.`;
    if (mpChart) mpChart.destroy();
    mpChart = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          { label: "Best %",  data: best,  borderColor: "#00c896", backgroundColor: "rgba(76,175,80,0.08)",  tension: 0.3, pointRadius: 3, fill: false },
          { label: "Avg %",   data: avg,   borderColor: "#0acdff", backgroundColor: "rgba(66,165,245,0.08)", tension: 0.3, pointRadius: 3, fill: false },
          { label: "Worst %", data: worst, borderColor: "#ff4455", backgroundColor: "rgba(239,83,80,0.08)",  tension: 0.3, pointRadius: 3, fill: false },
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: "#dde1e7", font: { size: 11 } } },
          tooltip: { callbacks: { label: c => c.dataset.label + ": " + c.raw.toFixed(1) + "%" } }
        },
        scales: {
          x: {
            ticks: { color: "#8b929d", maxTicksLimit: 20, maxRotation: 45 },
            grid: { color: "rgba(255,255,255,0.05)" }
          },
          y: { ticks: { color: "#8b929d", callback: v => v.toFixed(0) + "%" }, grid: { color: "rgba(255,255,255,0.05)" } }
        }
      }
    });
  } catch(e) { console.error("model-progress:", e); }
}

async function loadBacktests() {
  try {
    const r = await fetch(API_PREFIX + "/api/backtests").then(r => r.json());
    btRuns = r.runs || [];
    btSpyBaseline = r.spy_baseline != null ? r.spy_baseline : null;
    btLastUpdated = Date.now();
    btLoaded = true;
    renderBacktests();
    loadModelProgress();
  } catch (e) {
    console.error(e);
  } finally {
    if (btPollTimer) clearTimeout(btPollTimer);
    // 60s poll — list is now metadata-only (<50KB) so fast, but no need to hammer
    btPollTimer = setTimeout(loadBacktests, 60_000);
  }
}

function renderBacktests() {
  const total = btRuns.length;
  const running = btRuns.filter(x => x.status === "running");
  const failed = btRuns.filter(x => x.status === "failed");
  const completed = btRuns.filter(x => x.status === "complete");
  const pctDone = total ? (completed.length / total) * 100 : 0;
  document.getElementById("bt-progress-bar").style.width = pctDone + "%";
  let lbl = `${completed.length}/${total || 10} runs complete`;
  if (running.length)  lbl += ` · ${running.length} running`;
  if (failed.length)   lbl += ` · ${failed.length} failed`;
  document.getElementById("bt-progress-label").textContent = lbl;
  document.getElementById("bt-live-indicator").innerHTML =
    running.length ? `<span class="live-dot"></span>live` : `<span style="color:#00c896;">●</span> idle`;

  // Stats over the filtered set of completed runs
  const vis = filteredRuns().filter(r => r.status === "complete");
  document.getElementById("bt-filtered-count").textContent = vis.length ? `${vis.length}` : "—";
  if (vis.length) {
    const avg = vis.reduce((a,b) => a + (b.total_return_pct||0), 0) / vis.length;
    // B10: median alongside mean — mean is skewed by a few extreme windows
    // (e.g. a single 2008/2020 run), median is the typical-run truth.
    const _ret = vis.map(b => b.total_return_pct || 0).sort((a,b) => a-b);
    const _m = _ret.length;
    const med = _m === 0 ? 0
      : (_m % 2 ? _ret[(_m-1)/2] : (_ret[_m/2-1] + _ret[_m/2]) / 2);
    const annVals = vis.map(r => r.annualized_return_pct).filter(v => v != null);
    const avgAnn = annVals.length ? annVals.reduce((a,b)=>a+b,0)/annVals.length : null;
    const best = vis.reduce((a,b) => (b.annualized_return_pct||0) > (a.annualized_return_pct||0) ? b : a);
    const worst = vis.reduce((a,b) => (b.annualized_return_pct||0) < (a.annualized_return_pct||0) ? b : a);
    const beat = vis.filter(x => x.vs_spy_pct != null && x.vs_spy_pct > 0).length;
    const wins = vis.filter(x => (x.total_return_pct||0) > 0).length;

    const avgEl = document.getElementById("bt-avg");
    avgEl.innerHTML = `<span class="${avg>=0?'pos':'neg'}">${(avg>=0?"+":"")+fmt(avg)}%</span>`
      + ` <span class="muted" style="font-size:12px;">/</span> `
      + `<span class="${med>=0?'pos':'neg'}">${(med>=0?"+":"")+fmt(med)}%</span>`;
    avgEl.className = "v";

    const avgAnnEl = document.getElementById("bt-avg-ann");
    if (avgAnn != null) {
      avgAnnEl.textContent = (avgAnn >= 0 ? "+" : "") + fmt(avgAnn) + "%/yr";
      avgAnnEl.className = "v " + (avgAnn >= 0 ? "pos" : "neg");
    } else { avgAnnEl.textContent = "—"; avgAnnEl.className = "v"; }

    const bestAnn = best.annualized_return_pct;
    document.getElementById("bt-best").innerHTML =
      `<span class="${(bestAnn||0)>=0?'pos':'neg'}">${bestAnn!=null?(bestAnn>=0?"+":"")+fmt(bestAnn)+"%/yr":"—"}</span> <span class="muted" style="font-size:11px;">#${best.run_id}</span>`;
    const worstAnn = worst.annualized_return_pct;
    document.getElementById("bt-worst").innerHTML =
      `<span class="${(worstAnn||0)>=0?'pos':'neg'}">${worstAnn!=null?(worstAnn>=0?"+":"")+fmt(worstAnn)+"%/yr":"—"}</span> <span class="muted" style="font-size:11px;">#${worst.run_id}</span>`;
    document.getElementById("bt-beat").textContent = `${beat} / ${vis.length}`;
    const winrateEl = document.getElementById("bt-winrate");
    winrateEl.textContent = `${wins} / ${vis.length}`;
    winrateEl.className = "v " + (wins >= vis.length/2 ? "pos" : "neg");
  } else {
    ["bt-avg","bt-avg-ann","bt-best","bt-worst","bt-beat","bt-winrate"].forEach(id =>
      (document.getElementById(id).textContent = "—", document.getElementById(id).className = "v"));
  }

  renderLegend();
  renderTable();
  redrawChart();
  drawScatterChart();
  renderEraHeatmap();
  tickLastUpdated();
}

function renderLegend() {
  const wrap = document.getElementById("bt-legend");
  const vis = filteredRuns();
  const limitEl = document.getElementById("bt-chart-limit");
  const limit = limitEl ? parseInt(limitEl.value, 10) : 20;
  // Show the most-recent `limit` runs (sorted by run_id desc)
  const chartRuns = [...vis].sort((a,b) => b.run_id - a.run_id).slice(0, limit);
  wrap.innerHTML = chartRuns.map((r, i) => {
    const color = btRunColor(r.run_id, i);
    const hidden = btHiddenRuns.has(r.run_id);
    const selected = btSelectedRunId === r.run_id;
    const ann = r.annualized_return_pct;
    const retCls = (ann || 0) >= 0 ? "pos" : "neg";
    const retTxt = ann != null ? ((ann >= 0 ? "+" : "") + fmt(ann) + "%/yr") : "—";
    const durYrs = r.duration_days ? (r.duration_days / 365.25).toFixed(1) + "yr" : "";
    return `<div class="bt-legend-row${hidden ? ' hidden-run' : ''}${selected ? ' selected' : ''}" onclick="selectRun(${r.run_id})">
      <input type="checkbox" aria-label="Toggle visibility of backtest run #${r.run_id}" ${hidden ? '' : 'checked'} onclick="event.stopPropagation();toggleRun(${r.run_id})">
      <span class="bt-swatch" style="background:${color};"></span>
      <span class="name">#${r.run_id} <span class="muted" style="font-size:10px;">${durYrs}</span>${r.status === 'running' ? ' <span class="spinner" style="width:8px;height:8px;border-width:1px;margin:0 0 0 4px;"></span>' : ''}</span>
      <span class="ret ${retCls}">${retTxt}</span>
    </div>`;
  }).join("") || `<div class="muted" style="font-size:12px;">no runs match filter</div>`;
}

function renderTable() {
  const tbody = document.querySelector("#bt-tbl tbody");
  document.querySelectorAll("#bt-tbl thead th").forEach(th => {
    th.classList.add("sortable-h");
    th.classList.remove("sort-asc","sort-desc");
    if (th.dataset.k === btSortKey) th.classList.add(btSortDir > 0 ? "sort-asc" : "sort-desc");
    th.onclick = () => {
      const k = th.dataset.k;
      if (btSortKey === k) btSortDir = -btSortDir; else { btSortKey = k; btSortDir = -1; }
      renderTable();
    };
  });
  const vis = filteredRuns();
  const sorted = [...vis].sort((a,b) => {
    const va = a[btSortKey], vb = b[btSortKey];
    if (va == null && vb == null) return 0;
    if (va == null) return 1;
    if (vb == null) return -1;
    if (typeof va === "number") return (va - vb) * btSortDir;
    return String(va).localeCompare(String(vb)) * btSortDir;
  });
  tbody.innerHTML = sorted.map(r => {
    const isRunning = r.status === "running";
    const isComplete = r.status === "complete";
    const retCls = (r.total_return_pct || 0) >= 0 ? "pos" : "neg";
    const annCls = (r.annualized_return_pct || 0) >= 0 ? "pos" : "neg";
    const vsCls  = (r.vs_spy_pct || 0) >= 0 ? "pos" : "neg";
    const selected = btSelectedRunId === r.run_id;
    const retCell = r.total_return_pct == null
      ? `<span class="muted">—</span>`
      : `<span class="${retCls}">${(r.total_return_pct >= 0 ? "+" : "") + fmt(r.total_return_pct)}%</span>`;
    const annCell = r.annualized_return_pct == null
      ? `<span class="muted">—</span>`
      : `<span class="${annCls}">${(r.annualized_return_pct >= 0 ? "+" : "") + fmt(r.annualized_return_pct)}%</span>`;
    const vsCell = isComplete && r.vs_spy_pct != null
      ? `<span class="${vsCls}">${(r.vs_spy_pct >= 0 ? "+" : "") + fmt(r.vs_spy_pct)}%</span>`
      : `<span class="muted">—</span>`;
    const win = formatWindow(r.start_date, r.end_date);
    const era = classifyEra(r.start_date, r.end_date);
    const eraPill = era ? `<span class="pill" style="background:${era.bg};color:${era.fg};font-size:10px;">${era.tag}</span>` : "";
    const winCell = win
      ? `<div style="font-size:11px;line-height:1.5;">${r.start_date} → ${r.end_date}<br>${eraPill}</div>`
      : `<span class="muted">—</span>`;
    const durCell = r.duration_days
      ? `<span style="font-size:11px;">${(r.duration_days/365.25).toFixed(1)}yr</span>`
      : "—";
    return `<tr class="bt-row${selected ? ' selected' : ''}" onclick="selectRun(${r.run_id})">
      <td><span class="pill" style="font-size:11px;">#${r.run_id}${isRunning?'<span class="spinner" style="width:7px;height:7px;border-width:1px;margin-left:3px;"></span>':''}</span></td>
      <td><span class="pill status-${r.status || 'pending'}">${r.status || 'pending'}</span></td>
      <td class="num">${retCell}</td>
      <td class="num">${annCell}</td>
      <td class="num">${vsCell}</td>
      <td>${winCell}</td>
      <td class="num">${durCell}</td>
      <td class="num">${r.n_trades || 0}</td>
      <td class="num">${r.n_decisions || 0}</td>
    </tr>`;
  }).join("") || `<tr><td colspan="9" class="muted">no runs match the current filter</td></tr>`;
}

// ───────── Backtest era classification (frontend) ─────────
// Classifies a date range into a market era for at-a-glance context.
function classifyEra(startStr, endStr) {
  if (!startStr) return null;
  const s = startStr, e = endStr || startStr;
  const sy = parseInt(s.slice(0,4),10), ey = parseInt(e.slice(0,4),10);
  // Pre-2008
  if (ey < 2008) return { tag: "Pre-GFC", bg: "rgba(96,96,128,0.18)", fg: "#9a9ec0" };
  // GFC overlap
  if (sy <= 2009 && ey >= 2008) return { tag: "GFC", bg: "rgba(239,83,80,0.22)", fg: "#ff6b6b" };
  // COVID Q1 2020 inclusion (treat any range touching Jan–Apr 2020 as COVID crash)
  if (s <= "2020-04-30" && e >= "2020-01-20") return { tag: "COVID crash", bg: "rgba(239,83,80,0.28)", fg: "#ff7676" };
  // 2020–2021 recovery (only if start ≥ 2020-04 and end ≤ 2021-12)
  if (s >= "2020-04-01" && e <= "2021-12-31") return { tag: "Recovery", bg: "rgba(0,200,150,0.18)", fg: "#00c896" };
  // 2022 rate hike bear
  if (sy === 2022 && ey === 2022) return { tag: "Rate hike bear", bg: "rgba(255,140,0,0.22)", fg: "#ffb74d" };
  // 2023–2024 AI bull
  if (sy >= 2023 && ey <= 2024) return { tag: "AI bull", bg: "rgba(10,205,255,0.18)", fg: "#0acdff" };
  // 2025+
  if (sy >= 2025) return { tag: "Recent", bg: "rgba(127,255,0,0.18)", fg: "#7fff00" };
  // 2010–2019 fallback bull market
  if (sy >= 2010 && ey <= 2019) return { tag: "Bull market", bg: "rgba(0,200,150,0.15)", fg: "#5dd9b3" };
  // Spans multiple eras
  return { tag: "Multi-era", bg: "rgba(155,155,155,0.18)", fg: "#cfd2da" };
}

function formatWindow(startStr, endStr) {
  if (!startStr) return null;
  const e = endStr || "…";
  // Years span for the "(Nyr)" suffix
  let yrs = "";
  try {
    const sd = new Date(startStr);
    const ed = endStr ? new Date(endStr) : new Date();
    const days = (ed - sd) / 86400000;
    if (isFinite(days) && days > 0) {
      const y = days / 365.25;
      yrs = y >= 1 ? ` (${y.toFixed(y >= 5 ? 0 : 1)}yr)` : ` (${Math.round(days/30)}mo)`;
    }
  } catch (_) {}
  return `${startStr} → ${e}${yrs}`;
}

function dataSourcesForWindow(startStr, endStr) {
  if (!startStr) return [];
  const s = startStr, e = endStr || startStr;
  // GDELT public coverage starts 2015-02-19; SEC EDGAR full-text goes back to 1994.
  const gdeltOk = e >= "2015-02-19";
  return [
    { label: "GDELT news", ok: gdeltOk, hint: gdeltOk ? "coverage since 2015-02-19" : "pre-2015 — not available" },
    { label: "SEC EDGAR filings", ok: true, hint: "back to 1994" },
    { label: "Price / quant signals", ok: true, hint: "yfinance OHLCV" },
    { label: "Historical articles labeled by Claude", ok: true, hint: "Opus winner annotations + backtest injections" },
  ];
}

// Normalized chart: X = day-index from run start (so all window lengths compare),
// Y = % gain from start (so 1yr and 10yr runs are on the same scale).
async function drawBacktestChart() {
  const limitEl = document.getElementById("bt-chart-limit");
  const limit = limitEl ? parseInt(limitEl.value, 10) : 20;

  const vis = filteredRuns();
  // Most recent `limit` runs by run_id
  const chartRuns = [...vis].sort((a,b) => b.run_id - a.run_id).slice(0, limit);

  // Fetch any missing curves first, then render
  const needIds = chartRuns.map(r => r.run_id);
  await ensureCurves(needIds, null);

  // Build day-index label set (union across all visible runs)
  const daySet = new Set([0]);
  chartRuns.forEach(r => {
    const curve = btCurvesCache[r.run_id] || [];
    curve.forEach(p => { if (p.day_index != null) daySet.add(p.day_index); });
  });
  const labels = Array.from(daySet).sort((a,b) => a-b);

  const hasSelection = btSelectedRunId != null;

  const datasets = chartRuns.map((r, i) => {
    const curve = btCurvesCache[r.run_id] || [];
    const lookup = {};
    curve.forEach(p => { if (p.day_index != null) lookup[p.day_index] = p.value_pct; });
    let last = 0;
    const data = labels.map(d => {
      if (lookup[d] != null) { last = lookup[d]; return lookup[d]; }
      // forward-fill, but only within the run's duration
      const maxDay = r.duration_days || 9999;
      return d <= maxDay ? last : null;
    });
    const isRunning = r.status === "running";
    const color = btRunColor(r.run_id, i);
    const isHidden = btHiddenRuns.has(r.run_id);
    const isSelected = btSelectedRunId === r.run_id;
    const dim = hasSelection && !isSelected;
    const durYrs = r.duration_days ? (r.duration_days/365.25).toFixed(1)+"yr" : "";
    const ann = r.annualized_return_pct;
    const annTxt = ann != null ? ` (${(ann>=0?"+":"")+ann.toFixed(1)}%/yr ann.)` : "";
    return {
      label: `#${r.run_id} ${durYrs}${annTxt}`,
      data,
      runId: r.run_id,
      kind: "run",
      borderColor: dim ? hexToRgba(color, 0.18) : color,
      backgroundColor: hexToRgba(color, 0.04),
      borderWidth: isSelected ? 3.5 : (dim ? 0.8 : 1.5),
      borderDash: isRunning ? [5, 4] : [],
      pointRadius: 0, pointHoverRadius: 5,
      tension: 0.15, fill: false,
      hidden: isHidden,
      spanGaps: false,
    };
  });

  // SPY benchmark: average annualized SPY return for S&P ~10.7%/yr.
  // We draw an "average SPY" reference line that grows at ~10.7%/yr.
  // If a specific per-window SPY is available from btSpyBaseline we use it scaled to max duration.
  const maxDur = chartRuns.reduce((m, r) => Math.max(m, r.duration_days||0), 0);
  if (maxDur > 0) {
    const spyAnnPct = 10.7; // long-run S&P average annualized %
    const spyData = labels.map(d => {
      const yrs = d / 365.25;
      return ((1 + spyAnnPct/100)**yrs - 1) * 100;
    });
    datasets.push({
      label: `SPY avg (~${spyAnnPct}%/yr)`,
      data: spyData,
      kind: "benchmark",
      borderColor: hasSelection ? "rgba(180,180,180,0.2)" : "rgba(180,180,180,0.7)",
      borderWidth: 2,
      borderDash: [6, 3],
      pointRadius: 0,
      tension: 0,
      fill: false,
      order: -1,
    });
    // B5: QQQ baseline alongside SPY. The backtest store only tracks
    // spy_return_pct (no qqq column), so — like the SPY avg line above —
    // draw a long-run constant-rate reference (Nasdaq-100 ~13%/yr) rather
    // than adding a yfinance call to this hot list endpoint.
    const qqqAnnPct = 13.0;
    const qqqData = labels.map(d => {
      const yrs = d / 365.25;
      return ((1 + qqqAnnPct/100)**yrs - 1) * 100;
    });
    datasets.push({
      label: `QQQ avg (~${qqqAnnPct}%/yr)`,
      data: qqqData,
      kind: "benchmark",
      borderColor: hasSelection ? "rgba(122,162,247,0.18)" : "rgba(122,162,247,0.6)",
      borderWidth: 2,
      borderDash: [2, 3],
      pointRadius: 0,
      tension: 0,
      fill: false,
      order: -1,
    });
  }

  if (btChart) { btChart.destroy(); btChart = null; }
  const canvas = document.getElementById("bt-chart");
  if (!canvas) return;
  btChart = new Chart(canvas, {
    type: "line",
    data: { labels, datasets },
    options: {
      animation: false,
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      onClick: (evt, els, chart) => {
        if (els && els.length) {
          for (const el of els) {
            const ds = chart.data.datasets[el.datasetIndex];
            if (ds && ds.kind === "run") { selectRun(ds.runId); return; }
          }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          mode: "index", intersect: false,
          backgroundColor: "rgba(15,20,28,0.95)",
          borderColor: "#2a3a4f", borderWidth: 1,
          titleColor: "#dde1e7", bodyColor: "#dde1e7",
          padding: 10, boxPadding: 4,
          itemSort: (a,b) => b.parsed.y - a.parsed.y,
          filter: (item) => item.parsed.y != null,
          callbacks: {
            title: (items) => {
              const d = items[0]?.parsed?.x;
              if (d == null) return "";
              const yrs = (d/365.25).toFixed(1);
              return `Day ${d} (year ${yrs})`;
            },
            label: (ctx) => {
              const v = ctx.parsed.y;
              if (v == null) return null;
              return `${ctx.dataset.label}: ${v>=0?"+":""}${v.toFixed(1)}%`;
            },
          },
        },
      },
      scales: {
        x: {
          type: "linear",
          title: { display: true, text: "Days from start", color: "#50565f", font: { size: 10 } },
          ticks: {
            color: "#8b929d", maxTicksLimit: 10,
            callback: v => v >= 365 ? (v/365).toFixed(1)+"yr" : "d"+v,
          },
          grid: { color: "#1f2126" },
        },
        y: {
          title: { display: true, text: "Return from start (%)", color: "#50565f", font: { size: 10 } },
          ticks: {
            color: "#dde1e7",
            callback: v => (v>=0?"+":"") + v.toFixed(0) + "%",
          },
          grid: { color: "#1f2126" },
        },
      },
    },
  });
}

// ───────── Chart mode toggle ─────────
let btChartMode = "aggregate";
let btDdChart = null;

function setChartMode(mode) {
  btChartMode = mode;
  document.getElementById("mode-agg").classList.toggle("active", mode === "aggregate");
  document.getElementById("mode-ind").classList.toggle("active", mode === "individual");
  document.getElementById("agg-legend").style.display = mode === "aggregate" ? "" : "none";
  document.getElementById("ind-controls").style.display = mode === "individual" ? "" : "none";
  document.getElementById("bt-drawdown-wrap").style.display = mode === "aggregate" ? "" : "none";
  redrawChart();
}

function redrawChart() {
  if (btChartMode === "aggregate") {
    drawAggregateChart();
  } else {
    drawBacktestChart();
    if (btDdChart) { btDdChart.destroy(); btDdChart = null; }
  }
}

// ───────── Aggregate chart: percentile bands ─────────
// X = day-offset, Y = % return. Shows median/P25-P75/P5-P95 across all completed
// runs in the current window filter. SPY overlay uses actual per-run spy_return_pct.
async function drawAggregateChart() {
  const vis = filteredRuns().filter(r => r.status === "complete" && r.duration_days);
  const sampleRuns = vis;

  // Fetch curves we don't have yet
  await ensureCurves(sampleRuns.map(r => r.run_id), null);

  const nLabel = document.getElementById("agg-n-label");
  if (nLabel) nLabel.textContent = `(${sampleRuns.length} runs)`;

  // Build day → [value_pct] map; every run contributes 0% at day 0
  const byDay = { 0: sampleRuns.map(() => 0) };
  sampleRuns.forEach(r => {
    const curve = btCurvesCache[r.run_id] || [];
    curve.forEach(p => {
      if (p.day_index == null || p.day_index === 0) return;
      if (!byDay[p.day_index]) byDay[p.day_index] = [];
      byDay[p.day_index].push(p.value_pct);
    });
  });

  // Build day → [drawdown_pct] map (from curve peaks)
  const ddByDay = {};
  sampleRuns.forEach(r => {
    const curve = btCurvesCache[r.run_id] || [];
    let peak = 0;
    curve.forEach(p => {
      if (p.day_index == null) return;
      if (p.value_pct > peak) peak = p.value_pct;
      const dd = peak > 0 ? (p.value_pct - peak) : (p.value_pct < 0 ? p.value_pct : 0);
      if (!ddByDay[p.day_index]) ddByDay[p.day_index] = [];
      ddByDay[p.day_index].push(dd);
    });
  });

  const pct = (arr, p) => {
    const s = [...arr].sort((a,b) => a-b);
    const idx = (p/100) * (s.length - 1);
    const lo = Math.floor(idx), hi = Math.ceil(idx);
    return s[lo] + (s[hi] - s[lo]) * (idx - lo);
  };

  const MIN_N = 5;
  const days = Object.keys(byDay).map(Number).sort((a,b) => a-b);
  const labels = days.filter(d => byDay[d].length >= MIN_N);

  const P5=[], P25=[], P50=[], P75=[], P95=[], DD_P50=[], DD_P75=[];
  labels.forEach(d => {
    const v = byDay[d];
    P5.push({ x: d, y: pct(v, 5) });
    P25.push({ x: d, y: pct(v, 25) });
    P50.push({ x: d, y: pct(v, 50) });
    P75.push({ x: d, y: pct(v, 75) });
    P95.push({ x: d, y: pct(v, 95) });
    const dd = ddByDay[d] || [0];
    DD_P50.push({ x: d, y: pct(dd, 50) });
    DD_P75.push({ x: d, y: pct(dd, 75) });
  });

  // Median actual SPY growth curve using per-run spy_return_pct annualized
  const spyAnns = sampleRuns
    .filter(r => r.spy_return_pct != null && r.duration_days > 30)
    .map(r => Math.pow(1 + r.spy_return_pct / 100, 365.25 / r.duration_days) - 1);
  spyAnns.sort((a,b) => a-b);
  const medSpyAnn = spyAnns.length ? spyAnns[Math.floor(spyAnns.length/2)] : 0.107;
  const spyLine = labels.map(d => ({ x: d, y: (Math.pow(1 + medSpyAnn, d/365.25) - 1) * 100 }));
  const spyPctLabel = (medSpyAnn * 100).toFixed(1);
  // B5: QQQ baseline. The store tracks no per-run qqq_return_pct, so use a
  // long-run Nasdaq-100 constant-rate reference (~13%/yr) — consistent with
  // the SPY-avg reference in the per-run chart.
  const qqqAnnAgg = 0.13;
  const qqqLine = labels.map(d => ({ x: d, y: (Math.pow(1 + qqqAnnAgg, d/365.25) - 1) * 100 }));

  // Zero line
  const zeroLine = labels.map(d => ({ x: d, y: 0 }));

  const chartOpts = (yLabel, yFmt, minY) => ({
    animation: false,
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        mode: "index", intersect: false,
        backgroundColor: "rgba(15,20,28,0.95)", borderColor: "#2a3a4f", borderWidth: 1,
        titleColor: "#dde1e7", bodyColor: "#dde1e7", padding: 8, boxPadding: 3,
        filter: item => item.dataset.showInTooltip !== false,
        callbacks: {
          title: items => {
            const d = items[0]?.parsed?.x;
            return d != null ? `Day ${d} (${(d/365.25).toFixed(1)} yr from start)` : "";
          },
          label: ctx => {
            const v = ctx.parsed.y;
            if (v == null) return null;
            return `${ctx.dataset.label}: ${yFmt(v)}`;
          },
        },
      },
    },
    scales: {
      x: {
        type: "linear",
        ticks: { color: "#8b929d", maxTicksLimit: 10, callback: v => v>=365?(v/365).toFixed(1)+"yr":"d"+v },
        grid: { color: "#1f2126" },
      },
      y: {
        title: { display: true, text: yLabel, color: "#50565f", font: { size: 10 } },
        min: minY,
        ticks: { color: "#dde1e7", callback: yFmt },
        grid: { color: "#1f2126" },
      },
    },
  });

  // ── Main equity chart ──
  if (btChart) { btChart.destroy(); btChart = null; }
  const canvas = document.getElementById("bt-chart");
  if (canvas) {
    btChart = new Chart(canvas, {
      type: "scatter",
      data: {
        datasets: [
          // Outer band (P5→P95): fill from P5 up to P95
          { label: "P5",  data: P5,  borderColor:"transparent", backgroundColor:"transparent", pointRadius:0, showLine:true, fill:false, showInTooltip:false },
          { label: "P95 outer", data: P95, borderColor:"rgba(10,205,255,0.08)", backgroundColor:"rgba(10,205,255,0.06)", pointRadius:0, showLine:true, fill:"-1", borderWidth:1, showInTooltip:false },
          // Inner band (P25→P75)
          { label: "P25", data: P25, borderColor:"transparent", backgroundColor:"transparent", pointRadius:0, showLine:true, fill:false, showInTooltip:false },
          { label: "P75 inner", data: P75, borderColor:"rgba(10,205,255,0.22)", backgroundColor:"rgba(10,205,255,0.18)", pointRadius:0, showLine:true, fill:"-1", borderWidth:1, showInTooltip:false },
          // Median
          { label: "Median", data: P50, borderColor:"#0acdff", backgroundColor:"transparent", pointRadius:0, showLine:true, fill:false, borderWidth:2.5, tension:0.15 },
          // SPY actual median
          { label: `SPY (${spyPctLabel}%/yr actual median)`, data: spyLine, borderColor:"rgba(200,200,200,0.65)", backgroundColor:"transparent", pointRadius:0, showLine:true, fill:false, borderWidth:1.5, borderDash:[6,3], tension:0 },
          // QQQ baseline (constant-rate ~13%/yr — store has no qqq per-run)
          { label: "QQQ (~13%/yr)", data: qqqLine, borderColor:"rgba(122,162,247,0.55)", backgroundColor:"transparent", pointRadius:0, showLine:true, fill:false, borderWidth:1.5, borderDash:[2,3], tension:0 },
          // Zero reference
          { label: "0%", data: zeroLine, borderColor:"rgba(255,255,255,0.08)", backgroundColor:"transparent", pointRadius:0, showLine:true, fill:false, borderWidth:1, borderDash:[2,4], showInTooltip:false },
        ],
      },
      options: {
        ...chartOpts("Return from start (%)", v => (v>=0?"+":"")+v.toFixed(0)+"%", null),
        onClick: () => {},
      },
    });
  }

  // ── Drawdown sub-chart ──
  if (btDdChart) { btDdChart.destroy(); btDdChart = null; }
  const ddCanvas = document.getElementById("bt-dd-chart");
  if (ddCanvas && DD_P50.length) {
    btDdChart = new Chart(ddCanvas, {
      type: "scatter",
      data: {
        datasets: [
          { label: "DD P75", data: DD_P75, borderColor:"rgba(239,83,80,0.12)", backgroundColor:"rgba(239,83,80,0.10)", pointRadius:0, showLine:true, fill:"origin", borderWidth:1 },
          { label: "Median DD", data: DD_P50, borderColor:"rgba(239,83,80,0.7)", backgroundColor:"transparent", pointRadius:0, showLine:true, fill:false, borderWidth:1.5 },
        ],
      },
      options: chartOpts("Drawdown (%)", v => v.toFixed(0)+"%", null),
    });
  }
}

// ───────── Era definitions (shared for scatter + heatmap) ─────────
const ERA_DEFS = [
  { key: "Pre-GFC",       start: "1900-01-01", end: "2007-12-31", color: "#9a9ec0" },
  { key: "GFC",           start: "2008-01-01", end: "2009-12-31", color: "#ff6b6b" },
  { key: "Bull 2010s",    start: "2010-01-01", end: "2019-12-31", color: "#5dd9b3" },
  { key: "COVID crash",   start: "2020-01-01", end: "2020-06-30", color: "#ff7676" },
  { key: "Recovery",      start: "2020-07-01", end: "2021-12-31", color: "#00c896" },
  { key: "Rate-hike bear",start: "2022-01-01", end: "2022-12-31", color: "#ffb74d" },
  { key: "AI bull",       start: "2023-01-01", end: "2024-12-31", color: "#0acdff" },
  { key: "Recent",        start: "2025-01-01", end: "2099-12-31", color: "#7fff00" },
];

// Assign an era to a run based on the midpoint of its window
function runEra(r) {
  if (!r.start_date || !r.duration_days) return null;
  const startMs = new Date(r.start_date).getTime();
  const midMs = startMs + (r.duration_days / 2) * 86400000;
  const midStr = new Date(midMs).toISOString().slice(0, 10);
  for (const e of ERA_DEFS) {
    if (midStr >= e.start && midStr <= e.end) return e;
  }
  return { key: "Other", color: "#555" };
}

// ───────── Scatter: duration (X) vs annualized return (Y), colored by era ─────────
function drawScatterChart() {
  const completed = btRuns.filter(r => r.status === "complete" && r.duration_days && r.annualized_return_pct != null);
  if (!completed.length) return;

  // Group into era datasets
  const byEra = {};
  completed.forEach(r => {
    const era = runEra(r) || { key: "Other", color: "#555" };
    if (!byEra[era.key]) byEra[era.key] = { color: era.color, points: [] };
    byEra[era.key].points.push({ x: r.duration_days / 365.25, y: r.annualized_return_pct, runId: r.run_id });
  });

  // Era legend
  const legendEl = document.getElementById("bt-era-legend");
  if (legendEl) {
    legendEl.innerHTML = Object.entries(byEra).map(([key, v]) =>
      `<span style="display:inline-flex;align-items:center;gap:4px;padding:2px 8px;border-radius:99px;background:${hexToRgba(v.color,0.15)};border:1px solid ${hexToRgba(v.color,0.4)};color:${v.color};font-size:10px;">
        <span style="width:6px;height:6px;border-radius:50%;background:${v.color};display:inline-block;"></span>${key} (${v.points.length})
      </span>`
    ).join("");
  }

  const datasets = Object.entries(byEra).map(([key, v]) => ({
    label: key,
    data: v.points,
    backgroundColor: hexToRgba(v.color, 0.7),
    borderColor: v.color,
    borderWidth: 1,
    pointRadius: 5,
    pointHoverRadius: 8,
  }));

  if (btScatter) { btScatter.destroy(); btScatter = null; }
  const canvas = document.getElementById("bt-scatter");
  if (!canvas) return;
  btScatter = new Chart(canvas, {
    type: "scatter",
    data: { datasets },
    options: {
      animation: false,
      responsive: true, maintainAspectRatio: false,
      onClick: (evt, els) => {
        if (els && els.length) {
          const pt = els[0];
          const ds = btScatter.data.datasets[pt.datasetIndex];
          const runId = ds?.data[pt.index]?.runId;
          if (runId != null) selectRun(runId);
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "rgba(15,20,28,0.95)", borderColor: "#2a3a4f", borderWidth: 1,
          titleColor: "#dde1e7", bodyColor: "#dde1e7", padding: 10,
          callbacks: {
            title: (items) => {
              const pt = items[0];
              return `Run #${pt.raw.runId}`;
            },
            label: (ctx) => {
              const pt = ctx.raw;
              const r = btRuns.find(r => r.run_id === pt.runId);
              const lines = [
                `Duration: ${pt.x.toFixed(1)}yr`,
                `Annualized: ${(pt.y>=0?"+":"")+pt.y.toFixed(1)}%/yr`,
              ];
              if (r) {
                lines.push(`Total: ${(r.total_return_pct>=0?"+":"")+r.total_return_pct.toFixed(1)}%`);
                lines.push(`${r.start_date} → ${r.end_date}`);
              }
              return lines;
            },
          },
        },
      },
      scales: {
        x: {
          title: { display: true, text: "Window length (years)", color: "#50565f", font: { size: 10 } },
          ticks: { color: "#8b929d", callback: v => v.toFixed(0)+"yr" },
          grid: { color: "#1f2126" },
        },
        y: {
          title: { display: true, text: "Annualized return (%/yr)", color: "#50565f", font: { size: 10 } },
          ticks: { color: "#dde1e7", callback: v => (v>=0?"+":"") + v.toFixed(0) + "%" },
          grid: { color: "#1f2126" },
        },
      },
    },
  });
}

// ───────── Era × Duration heatmap table ─────────
const DUR_BUCKETS = [
  { label: "1yr",   min: 0,   max: 1.5 },
  { label: "2yr",   min: 1.5, max: 2.5 },
  { label: "3yr",   min: 2.5, max: 3.5 },
  { label: "4–5yr", min: 3.5, max: 5.5 },
  { label: "6–10yr",min: 5.5, max: 99  },
];

function renderEraHeatmap() {
  const el = document.getElementById("bt-heatmap");
  if (!el) return;
  const completed = btRuns.filter(r => r.status === "complete" && r.annualized_return_pct != null && r.duration_days);

  // Build cell data: era × dur bucket → [annualized values]
  const cells = {};
  ERA_DEFS.forEach(e => { cells[e.key] = {}; DUR_BUCKETS.forEach(b => { cells[e.key][b.label] = []; }); });

  completed.forEach(r => {
    const era = runEra(r);
    if (!era) return;
    const yrs = r.duration_days / 365.25;
    const bkt = DUR_BUCKETS.find(b => yrs >= b.min && yrs < b.max);
    if (!bkt) return;
    if (!cells[era.key]) cells[era.key] = {};
    if (!cells[era.key][bkt.label]) cells[era.key][bkt.label] = [];
    cells[era.key][bkt.label].push(r.annualized_return_pct);
  });

  // Find global min/max for color scaling
  let gmin = Infinity, gmax = -Infinity;
  ERA_DEFS.forEach(e => DUR_BUCKETS.forEach(b => {
    const vals = cells[e.key]?.[b.label] || [];
    if (vals.length) {
      const avg = vals.reduce((a,v)=>a+v,0)/vals.length;
      if (avg < gmin) gmin = avg;
      if (avg > gmax) gmax = avg;
    }
  }));

  // Color: negative → red, 0 → neutral, positive → green
  function heatColor(avg, n) {
    if (!n) return "rgba(255,255,255,0.03)";
    const norm = avg / Math.max(Math.abs(gmin), Math.abs(gmax), 1);
    if (avg >= 0) return `rgba(0,200,150,${Math.min(0.8, norm * 0.7 + 0.1)})`;
    return `rgba(239,83,80,${Math.min(0.8, -norm * 0.7 + 0.1)})`;
  }

  // Filter out eras with zero data
  const activeEras = ERA_DEFS.filter(e => DUR_BUCKETS.some(b => (cells[e.key]?.[b.label]||[]).length > 0));

  let html = `<table style="border-collapse:collapse;width:100%;font-size:12px;min-width:500px;">
    <thead><tr>
      <th style="text-align:left;padding:6px 10px;color:var(--text-muted);font-weight:500;border-bottom:1px solid var(--border);">Era (midpoint)</th>`;
  DUR_BUCKETS.forEach(b => {
    html += `<th style="text-align:center;padding:6px 10px;color:var(--text-muted);font-weight:500;border-bottom:1px solid var(--border);">${b.label}</th>`;
  });
  html += `</tr></thead><tbody>`;

  activeEras.forEach(e => {
    html += `<tr><td style="padding:6px 10px;color:${e.color};font-weight:500;white-space:nowrap;">
      <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${e.color};margin-right:5px;"></span>${e.key}
    </td>`;
    DUR_BUCKETS.forEach(b => {
      const vals = cells[e.key]?.[b.label] || [];
      const n = vals.length;
      const avg = n ? vals.reduce((a,v)=>a+v,0)/n : null;
      const bg = heatColor(avg, n);
      const txt = avg != null ? `${avg>=0?"+":""}${avg.toFixed(1)}%` : "";
      const sub = n ? `<div style="font-size:10px;opacity:0.6;">n=${n}</div>` : `<div style="color:var(--text-muted);font-size:11px;">—</div>`;
      html += `<td style="text-align:center;padding:5px 8px;background:${bg};border:1px solid rgba(255,255,255,0.04);">
        ${txt ? `<div style="font-weight:600;color:${avg>=0?"#b6f0d8":"#ffaaaa"};">${txt}</div>` : ""}
        ${sub}
      </td>`;
    });
    html += `</tr>`;
  });

  html += `</tbody></table>`;
  el.innerHTML = html;
}

function selectRun(runId) {
  btSelectedRunId = (btSelectedRunId === runId) ? null : runId;
  renderLegend();
  renderTable();
  // In aggregate mode, selecting a run switches to individual view for the specific run
  if (btSelectedRunId != null && btChartMode === "aggregate") {
    setChartMode("individual");
  } else {
    redrawChart();
  }
  if (btSelectedRunId != null) loadRunDetail(btSelectedRunId);
  else closeDetail();
}

function toggleRun(runId) {
  if (btHiddenRuns.has(runId)) btHiddenRuns.delete(runId);
  else btHiddenRuns.add(runId);
  renderLegend();
  redrawChart();
}

function btToggleAll(show) {
  const vis = filteredRuns();
  btHiddenRuns = show ? new Set() : new Set(vis.map(r => r.run_id));
  renderLegend();
  redrawChart();
}

function closeDetail() {
  document.getElementById("bt-detail").style.display = "none";
  btSelectedRunId = null;
  renderLegend(); renderTable(); redrawChart();
}

function showBtSubtab(name) {
  btDetailSubtab = name;
  document.querySelectorAll(".bt-subpane").forEach(el => el.classList.remove("active"));
  document.querySelectorAll(".bt-tabs a").forEach(el => el.classList.remove("active"));
  document.getElementById("bt-tab-" + name).classList.add("active");
  document.getElementById("bt-tab-" + name + "-link").classList.add("active");
}

async function loadRunDetail(runId) {
  const wrap = document.getElementById("bt-detail");
  document.getElementById("bt-detail-id").textContent = "#" + runId;
  wrap.style.display = "block";
  const r = await fetch(API_PREFIX + `/api/backtests/${runId}`).then(r => r.json());
  const meta = [];
  if (r.seed != null) meta.push(`seed ${r.seed}`);
  const winStr = formatWindow(r.start_date, r.end_date);
  if (winStr) meta.push(winStr);
  if (r.status) meta.push(r.status);
  if (r.n_trades != null) meta.push(`${r.n_trades} trades`);
  if (r.n_decisions != null) meta.push(`${r.n_decisions} decisions`);
  if (r.notes) meta.push(r.notes);
  const metaEl = document.getElementById("bt-detail-meta");
  metaEl.innerHTML = "";
  metaEl.appendChild(document.createTextNode(meta.join(" · ")));
  // Era pill + data-source pills
  const era = classifyEra(r.start_date, r.end_date);
  if (era) {
    metaEl.insertAdjacentHTML(
      "beforeend",
      ` <span class="pill" style="background:${era.bg};color:${era.fg};margin-left:6px;">${era.tag}</span>`,
    );
  }
  const srcs = dataSourcesForWindow(r.start_date, r.end_date);
  if (srcs.length) {
    const pillRow = srcs.map(s => {
      const mark = s.ok ? "✓" : "✗";
      const color = s.ok ? "#00c896" : "#ff4455";
      const bg = s.ok ? "rgba(0,200,150,0.10)" : "rgba(255,68,85,0.10)";
      return `<span class="pill" title="${s.hint}" style="background:${bg};color:${color};font-size:11px;">${mark} ${s.label}</span>`;
    }).join(" ");
    metaEl.insertAdjacentHTML(
      "beforeend",
      `<div style="margin-top:6px;display:flex;flex-wrap:wrap;gap:6px;">${pillRow}</div>`,
    );
  }

  const tBody = document.querySelector("#bt-trades-tbl tbody");
  tBody.innerHTML = (r.trades || []).map(t => {
    const cls = (t.action||"").startsWith("SELL") ? "sell" : "buy";
    return `<tr><td>${t.sim_date || ''}</td>
      <td><span class="pill ${cls}">${t.action || ''}</span></td>
      <td>${t.ticker || ''}</td>
      <td class="num">${fmt(t.qty,4)}</td>
      <td class="num">${fmt(t.price)}</td>
      <td class="num">${fmt(t.value)}</td>
      <td class="muted">${(t.reason||"").slice(0,140)}</td></tr>`;
  }).join("") || `<tr><td colspan="7" class="muted">no trades</td></tr>`;

  const dBody = document.querySelector("#bt-decisions-tbl tbody");
  dBody.innerHTML = (r.decisions || []).map(d => {
    return `<tr><td>${d.sim_date || ''}</td>
      <td>${d.action || ''}</td>
      <td>${d.ticker || ''}</td>
      <td><span class="pill">${d.status || ''}</span></td>
      <td class="muted">${(d.detail||"").slice(0,140)}</td>
      <td class="num">${fmt(d.total_value)}</td></tr>`;
  }).join("") || `<tr><td colspan="6" class="muted">no decisions</td></tr>`;
}

function tickLastUpdated() {
  const el = document.getElementById("bt-last-updated");
  if (!el) return;
  if (btLastUpdated == null) { el.textContent = "last update: —"; return; }
  const s = Math.floor((Date.now() - btLastUpdated)/1000);
  el.textContent = `last updated ${s}s ago`;
}
setInterval(tickLastUpdated, 1000);

// ───────── Signal feed (from Digital Intern) ─────────
async function refreshSignals() {
  const ul = document.getElementById("signal-feed");
  try {
    const r = await fetch("/intern/api/articles?limit=3");
    if (!r.ok) {
      ul.innerHTML = `<li class="muted">signal feed unavailable (HTTP ${r.status})</li>`;
      return;
    }
    const arts = await r.json();
    if (!Array.isArray(arts) || !arts.length) {
      ul.innerHTML = `<li class="muted">no signals yet</li>`;
      return;
    }
    ul.innerHTML = arts.map(a => {
      const score = (a.score != null ? a.score : 0).toFixed(1);
      const url = a.url || "#";
      const title = (a.title || "(no title)").replace(/</g,"&lt;");
      const src = (a.source || "").replace(/</g,"&lt;");
      return `<li style="padding:6px 0;border-bottom:1px solid #1f2126;">
        <span class="pill" style="background:#1f3a4d;color:#4d9eff;margin-right:8px;">${score}</span>
        <a href="${url}" target="_blank" rel="noopener" style="color:#dde1e7;text-decoration:none">${title}</a>
        <span class="muted" style="margin-left:6px;">· ${src}</span>
      </li>`;
    }).join("");
  } catch (e) {
    ul.innerHTML = `<li class="muted">digital intern unreachable</li>`;
  }
}

// ───────── Live Data Feed widget (collector pulse from Digital Intern) ─────────
async function refreshDataFeed() {
  try {
    const r = await fetch(API_PREFIX + "/api/data-feed");
    if (!r.ok) return;
    const d = await r.json();
    const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
    set("df-1h",  d.articles_1h  != null ? d.articles_1h  + " articles" : "—");
    set("df-24h", d.articles_24h != null ? d.articles_24h + " articles" : "—");
    const srcEl = document.getElementById("df-sources");
    if (srcEl) {
      const top = (d.top_sources || []).slice(0, 3);
      srcEl.innerHTML = top.length
        ? "top: " + top.map(s => `<b style="color:#dde1e7;">${(s.name||"?").replace(/</g,'&lt;')}</b> <span style="color:#8b929d;">${s.count}</span>`).join(" · ")
        : '<span class="muted">no sources active</span>';
    }
    const asof = document.getElementById("df-asof");
    if (asof) asof.textContent = new Date().toLocaleTimeString();
  } catch (e) { /* silent */ }
}

// ───────── Portfolio Analytics ─────────
const SECTOR_COLORS = {
  semis: "#0acdff", semis_lev: "#1e88e5",
  optical: "#ab47bc",
  broad: "#00c896", broad_lev: "#43a047",
  tech: "#ffb74d", tech_lev: "#fb8c00",
  crypto_lev: "#ffd54f",
  bio_lev: "#ec407a", health_lev: "#e91e63",
  fin_lev: "#26a69a", defense_lev: "#7e57c2",
  housing_lev: "#8d6e63", util_lev: "#90a4ae",
  cash: "#455a64", other: "#8b929d",
};

function _sectorColor(name) { return SECTOR_COLORS[name] || "#8b929d"; }

async function refreshAnalytics() {
  let a;
  try { a = await fetch(API_PREFIX + "/api/analytics").then(r => r.json()); }
  catch (e) { return; }
  if (!a || a.error) return;

  const setStat = (id, txt, cls) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = txt;
    el.className = "v" + (cls ? " " + cls : "");
  };
  const sign = v => v == null ? "" : (v >= 0 ? "+" : "");
  const fmtPct = (v, d=2) => v == null ? "—" : sign(v) + fmt(v, d) + "%";
  const fmtUsd = (v, d=2) => v == null ? "—" : sign(v) + "$" + fmt(Math.abs(v), d);

  setStat("an-daily", a.daily_pl_usd == null ? "—" :
          `${fmtUsd(a.daily_pl_usd)} (${fmtPct(a.daily_pl_pct, 2)})`,
          a.daily_pl_usd == null ? null : (a.daily_pl_usd >= 0 ? "pos" : "neg"));
  setStat("an-dd", a.max_drawdown_usd == null ? "—" :
          `-${fmt(a.max_drawdown_usd)} (${fmt(a.max_drawdown_pct)}%)`,
          a.max_drawdown_usd > 0 ? "neg" : null);
  setStat("an-sharpe", a.sharpe_annualized == null ? "—" : fmt(a.sharpe_annualized, 2),
          a.sharpe_annualized != null ? (a.sharpe_annualized >= 0 ? "pos" : "neg") : null);
  if (a.win_rate_pct == null) setStat("an-winrate", `— (0 trips)`);
  else setStat("an-winrate", `${fmt(a.win_rate_pct, 1)}% (${a.n_round_trips})`,
                a.win_rate_pct >= 50 ? "pos" : "neg");
  setStat("an-avgw", a.avg_winner_usd == null ? "—" : "$" + fmt(a.avg_winner_usd), a.avg_winner_usd != null ? "pos" : null);
  setStat("an-avgl", a.avg_loser_usd == null ? "—" : fmtUsd(a.avg_loser_usd), a.avg_loser_usd != null ? "neg" : null);
  setStat("an-realized", fmtUsd(a.realized_pl_usd, 2), a.realized_pl_usd >= 0 ? "pos" : "neg");

  setStat("an-pf", a.profit_factor == null ? "—" : fmt(a.profit_factor, 2),
          a.profit_factor != null ? (a.profit_factor >= 1 ? "pos" : "neg") : null);
  setStat("an-sortino", a.sortino_annualized == null ? "—" : fmt(a.sortino_annualized, 2),
          a.sortino_annualized != null ? (a.sortino_annualized >= 0 ? "pos" : "neg") : null);
  setStat("an-calmar", a.calmar_ratio == null ? "—" : fmt(a.calmar_ratio, 2),
          a.calmar_ratio != null ? (a.calmar_ratio >= 0 ? "pos" : "neg") : null);
  setStat("an-beta", a.sp500_beta == null ? "—" : fmt(a.sp500_beta, 2));
  setStat("an-corr", a.sp500_correlation == null ? "—" : fmt(a.sp500_correlation, 2));
  setStat("an-hold", a.avg_holding_days == null ? "—" :
          fmt(a.avg_holding_days, 1) + "d");

  // Sector stacked bar
  const sectors = a.sector_exposure_pct || {};
  const cashPct = a.cash_pct || 0;
  const segs = [];
  for (const [name, pct] of Object.entries(sectors)) {
    if (pct > 0) segs.push({ name, pct, color: _sectorColor(name) });
  }
  if (cashPct > 0) segs.push({ name: "cash", pct: cashPct, color: _sectorColor("cash") });
  segs.sort((a, b) => b.pct - a.pct);

  const barEl = document.getElementById("an-sector-bar");
  if (barEl) {
    barEl.innerHTML = segs.map(s =>
      `<div title="${s.name} ${fmt(s.pct,1)}%" style="flex:${s.pct};background:${s.color};border-right:1px solid #0d1117;"></div>`
    ).join("") || `<div class="muted" style="padding:3px 8px;font-size:12px;">no allocations</div>`;
  }
  const legEl = document.getElementById("an-sector-legend");
  if (legEl) {
    legEl.innerHTML = segs.map(s =>
      `<span><span style="display:inline-block;width:10px;height:10px;background:${s.color};border-radius:2px;margin-right:5px;vertical-align:middle;"></span>${s.name}: ${fmt(s.pct,1)}%</span>`
    ).join("") || `<span class="muted">no allocations</span>`;
  }
}

// ───────── Sector Pulse ─────────
async function refreshSectorPulse() {
  let r;
  try { r = await fetch(API_PREFIX + "/api/sector-pulse").then(r => r.json()); }
  catch (e) { return; }
  if (!r || !r.tickers) return;
  const grid = document.getElementById("sp-grid");
  if (!grid) return;
  document.getElementById("sp-asof").textContent = r.as_of ? "as of " + fmtTs(r.as_of) : "";
  grid.innerHTML = r.tickers.map(t => {
    const rsi = t.rsi;
    const rsiCls = rsi == null ? "muted" :
                   rsi >= 70 ? "neg" :
                   rsi <= 30 ? "pos" : "";
    const mom5 = t.mom_5d;
    const mom5Cls = mom5 == null ? "muted" : (mom5 >= 0 ? "pos" : "neg");
    const px = t.price;
    const news = t.news_count_24h || 0;
    const urgent = t.news_urgent_24h || 0;
    const newsBadge = urgent > 0
      ? `<span style="background:#3a1b1b;color:#ff4455;padding:1px 6px;border-radius:8px;font-size:10px;font-weight:600;">${urgent}!</span>`
      : news > 0
        ? `<span style="background:#1f3a4d;color:#4d9eff;padding:1px 6px;border-radius:8px;font-size:10px;">${news}</span>`
        : `<span class="muted" style="font-size:10px;">0</span>`;
    const headline = t.top_headline
      ? `<div style="margin-top:6px;font-size:11px;line-height:1.4;color:#dde1e7;">
           ${t.top_url ? `<a href="${t.top_url}" target="_blank" rel="noopener" style="color:#dde1e7;text-decoration:none;">${(t.top_headline||'').slice(0,100)}</a>` : (t.top_headline||'').slice(0,100)}
         </div>`
      : `<div class="muted" style="margin-top:6px;font-size:11px;">no news</div>`;
    return `<div style="background:#0d1117;border:1px solid #1f2126;border-radius:6px;padding:10px;">
      <div style="display:flex;justify-content:space-between;align-items:baseline;">
        <span style="font-weight:600;color:#eceff1;font-size:14px;">${t.ticker}</span>
        <span style="font-size:13px;color:#dde1e7;font-variant-numeric:tabular-nums;">${px == null ? '—' : '$'+fmt(px)}</span>
      </div>
      <div style="display:flex;gap:8px;font-size:11px;margin-top:5px;color:#8b929d;">
        <span>RSI <span class="${rsiCls}">${rsi == null ? '—' : fmt(rsi,1)}</span></span>
        <span>5d <span class="${mom5Cls}">${mom5 == null ? '—' : (mom5>=0?'+':'')+fmt(mom5,1)+'%'}</span></span>
        <span style="margin-left:auto;">${newsBadge}</span>
      </div>
      ${headline}
    </div>`;
  }).join("");
}

// ───────── Daily briefing card ─────────
async function refreshBriefing() {
  try {
    const r = await fetch(API_PREFIX + "/api/briefing").then(r => r.json());
    if (r.error) return;
    const dot = document.getElementById("briefing-dot");
    if (dot) dot.style.background = r.market_open ? "#00c896" : "#ff4455";
    document.getElementById("briefing-status").textContent = r.status_line || "";
    document.getElementById("briefing-asof").textContent = fmtTs(r.as_of, "sec");
    // Futures row
    const futWrap = document.getElementById("briefing-futures");
    const futNames = {"ES=F":"S&P fut","NQ=F":"NQ fut","CL=F":"WTI","GC=F":"Gold"};
    futWrap.innerHTML = Object.entries(r.futures || {}).map(([sym,px]) => {
      const label = futNames[sym] || sym;
      const value = (px == null) ? "—" : Number(px).toLocaleString(undefined,{maximumFractionDigits:2});
      return `<div><span class="muted" style="font-size:11px;">${label}</span><div style="font-variant-numeric:tabular-nums;font-size:15px;color:#dde1e7;">${value}</div></div>`;
    }).join("");
    // Urgent news (top 5)
    const urgEl = document.getElementById("briefing-urgent");
    const urgent = r.urgent_news || [];
    if (!urgent.length) {
      urgEl.innerHTML = `<li class="muted" style="padding:4px 0;">no urgent news in the last 8h</li>`;
    } else {
      urgEl.innerHTML = urgent.map(u => {
        const sc = (u.ai_score != null) ? Number(u.ai_score).toFixed(1) : "—";
        const tk = (u.tickers || []).slice(0,3).join(" ");
        return `<li style="padding:4px 0;border-bottom:1px solid #1f2126;">
          <span style="display:inline-block;min-width:34px;color:#ff4455;font-variant-numeric:tabular-nums;font-weight:600;">${sc}</span>
          <span style="color:#dde1e7;">${(u.title || "").replace(/[<>]/g, '')}</span>
          ${tk ? `<span class="muted" style="font-size:11px;margin-left:6px;">[${tk}]</span>` : ""}
        </li>`;
      }).join("");
    }
  } catch (e) { console.error("briefing:", e); }
}

// ───────── Trade suggestions card ─────────
async function refreshSuggestions() {
  try {
    const r = await fetch(API_PREFIX + "/api/suggestions").then(r => r.json());
    if (r.error) {
      document.getElementById("sug-summary").textContent = "error: " + r.error;
      return;
    }
    const counts = r.action_counts || {};
    const summary = Object.entries(counts).map(([a,n]) => `${n} ${a}`).join(" · ") || "no actionable candidates";
    document.getElementById("sug-summary").textContent = `${r.n_candidates} candidates from ${r.n_signals_used} signals — ${summary}`;
    document.getElementById("sug-meta").textContent = fmtTs(r.as_of, "sec");
    const tbody = document.querySelector("#sug-tbl tbody");
    const items = r.suggestions || [];
    if (!items.length) {
      tbody.innerHTML = `<tr><td colspan="9" class="muted">no suggestions — no actionable news in the last 6h</td></tr>`;
      return;
    }
    const actionStyle = {
      "BUY":   "background:#1b3a2a;color:#00c896;",
      "ADD":   "background:#1b3a2a;color:#00c896;",
      "TRIM":  "background:#3a2f1b;color:#ffb74d;",
      "EXIT":  "background:#3a1b1b;color:#ff4455;",
      "WATCH": "background:#1f3a4d;color:#4d9eff;",
      "HOLD":  "background:#1f2933;color:#dde1e7;",
    };
    tbody.innerHTML = items.map(s => {
      const styleA = actionStyle[s.action] || actionStyle["HOLD"];
      const px = (s.price == null) ? "—" : "$" + Number(s.price).toFixed(2);
      const qty = s.held_qty ? Number(s.held_qty).toFixed(2) : "—";
      const rsi = (s.rsi == null) ? "—" : Number(s.rsi).toFixed(0);
      const rsiCls = (s.rsi != null && s.rsi >= 70) ? "neg" : (s.rsi != null && s.rsi <= 35) ? "pos" : "";
      const urgent = s.news_urgent ? `<span style="color:#ff4455;font-weight:600;">!</span>` : "";
      const newsCell = s.news_count > 0
        ? `<span style="color:#4d9eff;">${s.news_count}</span> <span class="muted">@</span> ${Number(s.news_max_score).toFixed(1)} ${urgent}`
        : `<span class="muted">0</span>`;
      const reasons = (s.reasons || []).slice(0,3).join(" · ");
      const head = s.top_headline ? (s.top_url
        ? `<a href="${s.top_url}" target="_blank" rel="noopener" style="color:#dde1e7;">${s.top_headline.replace(/[<>]/g,'')}</a>`
        : `<span class="muted">${s.top_headline.replace(/[<>]/g,'')}</span>`) : `<span class="muted">—</span>`;
      return `<tr>
        <td><span class="pill" style="${styleA}padding:3px 8px;font-size:11px;font-weight:600;">${s.action}</span></td>
        <td style="font-weight:600;">${s.ticker}</td>
        <td class="num">${Number(s.conviction).toFixed(2)}</td>
        <td class="num">${px}</td>
        <td class="num muted">${qty}</td>
        <td class="num">${newsCell}</td>
        <td class="num ${rsiCls}">${rsi}</td>
        <td class="muted" style="font-size:11px;">${reasons}</td>
        <td style="font-size:12px;">${head}</td>
      </tr>`;
    }).join("");
  } catch (e) { console.error("suggestions:", e); }
}

// ───────── Risk panel card ─────────
async function refreshRisk() {
  try {
    const r = await fetch(API_PREFIX + "/api/risk").then(r => r.json());
    if (r.error) return;
    const top1Txt = r.concentration_top1_ticker
      ? `${Number(r.concentration_top1_pct).toFixed(1)}% <span class="muted" style="font-size:13px;">${r.concentration_top1_ticker}</span>`
      : "—";
    const top1El = document.getElementById("risk-top1");
    top1El.innerHTML = top1Txt;
    top1El.className = "v " + (r.concentration_top1_pct >= 40 ? "neg" : "");
    document.getElementById("risk-top3").textContent = (r.concentration_top3_pct != null) ? Number(r.concentration_top3_pct).toFixed(1) + "%" : "—";
    const levEl = document.getElementById("risk-lev");
    levEl.textContent = (r.leveraged_pct != null) ? Number(r.leveraged_pct).toFixed(1) + "%" : "—";
    levEl.className = "v " + (r.leveraged_pct >= 30 ? "neg" : "");
    const shockEl = document.getElementById("risk-shock");
    if (r.spy_shock_3pct_usd != null) {
      const v = Number(r.spy_shock_3pct_usd);
      const pct = Number(r.spy_shock_3pct_pct || 0);
      shockEl.innerHTML = `${v >= 0 ? "+" : ""}$${v.toFixed(2)} <span class="muted" style="font-size:12px;">(${pct >= 0 ? "+" : ""}${pct.toFixed(2)}%)</span>`;
      shockEl.className = "v " + (v < 0 ? "neg" : "pos");
    }
    document.getElementById("risk-age").textContent = (r.median_age_days != null) ? r.median_age_days : "—";
    const staleEl = document.getElementById("risk-stale-n");
    const stale = r.stale_positions || [];
    staleEl.textContent = stale.length;
    staleEl.className = "v " + (stale.length > 0 ? "neg" : "");
    const staleList = document.getElementById("risk-stale-list");
    if (!stale.length) {
      staleList.innerHTML = `<span class="muted">no stale positions — all holds are either fresh or moving</span>`;
    } else {
      staleList.innerHTML = "Stale: " + stale.map(s =>
        `<span style="display:inline-block;background:#1f2126;border:1px solid #3a2f1b;border-radius:4px;padding:3px 8px;margin-right:6px;margin-bottom:4px;">${s.ticker} ${s.age_days}d ${s.pl_pct >= 0 ? "+" : ""}${s.pl_pct}%</span>`
      ).join("");
    }
  } catch (e) { console.error("risk:", e); }
}

// ───────── Position blow-up ladder ─────────
async function refreshPositionBlowup() {
  let r;
  try { r = await fetch(API_PREFIX + "/api/position-blowup").then(x => x.json()); }
  catch (e) { return; }
  const tbody = document.getElementById("blowup-tbody");
  const head = document.getElementById("blowup-headline");
  const asof = document.getElementById("blowup-asof");
  if (!tbody) return;
  if (!r || r.error) { tbody.innerHTML = `<tr><td colspan="6" class="muted">unavailable</td></tr>`; return; }
  if (asof && r.as_of) asof.textContent = fmtTs(r.as_of, "time");
  const verdictCls = { CONCENTRATED: "neg", MODERATE: "", DIFFUSE: "pos", NO_DATA: "muted" };
  if (head) {
    head.innerHTML = `<span class="${verdictCls[r.state] || 'muted'}">${r.state || '—'}</span> — ${r.headline || ''}`;
  }
  const rows = r.positions || [];
  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="6" class="muted">no priced positions to shock</td></tr>`;
    return;
  }
  // shocks list carries one entry per magnitude; index by shock_pct so a
  // future change to SHOCK_MAGNITUDES_PCT can't silently mis-align columns.
  const cell = (p, mag) => {
    const s = (p.shocks || []).find(x => Number(x.shock_pct) === mag);
    if (!s) return `<td class="num muted">—</td>`;
    const v = Number(s.pnl_usd || 0);
    const pct = Number(s.pnl_pct_of_book || 0);
    return `<td class="num neg">$${v.toFixed(2)}<br><span class="muted" style="font-size:11px;">${pct.toFixed(1)}%</span></td>`;
  };
  tbody.innerHTML = rows.map(p =>
    `<tr><td><b>${p.ticker}</b> <span class="muted" style="font-size:11px;">${p.type || ''}</span></td>` +
    `<td class="num">${Number(p.weight_pct || 0).toFixed(1)}%</td>` +
    cell(p, -10) + cell(p, -25) + cell(p, -50) + cell(p, -100) + `</tr>`
  ).join("");
}

// ───────── Earnings radar ─────────
async function refreshEarningsRisk() {
  let r;
  try { r = await fetch(API_PREFIX + "/api/earnings-risk").then(r => r.json()); }
  catch (e) { return; }
  const list = document.getElementById("er-list");
  const meta = document.getElementById("er-meta");
  const asof = document.getElementById("er-asof");
  if (!list) return;
  if (!r || r.error) { list.innerHTML = `<li class="muted">unavailable</li>`; return; }
  if (asof && r.as_of) asof.textContent = fmtTs(r.as_of, "time");
  const evs = r.events || [];
  if (!r.source_ok) {
    meta.textContent = "earnings calendar (:8080) unreachable";
  } else {
    meta.innerHTML = `${r.n_held_reporting} holding(s) reporting · ` +
      `<span class="${r.n_imminent > 0 ? 'neg' : 'muted'}">${r.n_imminent} imminent (≤3d)</span> · ` +
      `$${Number(r.held_exposure_at_risk_usd || 0).toFixed(0)} exposure at risk`;
  }
  if (!evs.length) {
    list.innerHTML = `<li class="muted">no earnings within horizon for holdings or watchlist</li>`;
    return;
  }
  const tierStyle = {
    HELD_IMMINENT: "background:#3a1b1b;border:1px solid #7a2f2f;",
    HELD_SOON:     "background:#3a2f1b;border:1px solid #7a5f2f;",
    WATCH:         "background:#1f2126;border:1px solid #2f3540;",
  };
  const tierLabel = { HELD_IMMINENT: "⚠ HELD", HELD_SOON: "HELD", WATCH: "watch" };
  list.innerHTML = evs.slice(0, 14).map(e => {
    const d = e.days_away == null ? "?" : Number(e.days_away).toFixed(1) + "d";
    const exp = e.held ? ` · $${Number(e.exposure_usd).toFixed(0)}` : "";
    return `<li style="padding:6px 8px;margin-bottom:4px;border-radius:5px;${tierStyle[e.tier] || ''}">` +
      `<b>${e.ticker}</b> <span class="muted" style="font-size:11px;">${tierLabel[e.tier] || ''}</span>` +
      `<span style="float:right;">in ${d}${exp}</span></li>`;
  }).join("");
}

// ───────── Greeks card (options exposure) ─────────
async function refreshGreeks() {
  try {
    const r = await fetch(API_PREFIX + "/api/greeks").then(r => r.json());
    if (r.error) { return; }
    const positions = (r.positions || []).filter(p => p.type === "call" || p.type === "put");
    const card = document.getElementById("greeks-card");
    if (!card) return;
    // Hide card entirely when there are no option positions — keeps dashboard clean.
    if (positions.length === 0) { card.style.display = "none"; return; }
    card.style.display = "block";
    const t = r.totals || {};
    document.getElementById("gk-asof").textContent = fmtTs(r.as_of);
    const dElem = document.getElementById("gk-delta");
    dElem.textContent = fmt(t.delta, 2);
    dElem.className = "v " + ((t.delta || 0) >= 0 ? "pos" : "neg");
    document.getElementById("gk-gamma").textContent = fmt(t.gamma, 5);
    const thElem = document.getElementById("gk-theta");
    thElem.textContent = "$" + fmt(t.theta, 2);
    thElem.className = "v " + ((t.theta || 0) >= 0 ? "pos" : "neg");
    document.getElementById("gk-vega").textContent = "$" + fmt(t.vega, 2);
    document.getElementById("gk-notional").textContent = dollar(t.gross_notional);
    document.getElementById("gk-deltapct").textContent = (t.delta_pct_port != null) ? (fmt(t.delta_pct_port,1) + "%") : "—";
    const tbody = document.querySelector("#gk-tbl tbody");
    tbody.innerHTML = positions.map(p => {
      const cls = (p.delta || 0) >= 0 ? "pos" : "neg";
      const ivStr = p.iv != null ? (fmt(p.iv * 100, 1) + "%") : "—";
      const dteStr = p.days_to_expiry != null ? (p.days_to_expiry + "d") : "";
      return `<tr>
        <td>${p.ticker}</td>
        <td>${p.type.toUpperCase()}</td>
        <td class="num">${fmt(p.qty, 0)}</td>
        <td class="num">${p.strike || "—"} / ${p.expiry || "—"} ${dteStr ? `<span class="muted">(${dteStr})</span>` : ""}</td>
        <td class="num">${ivStr}</td>
        <td class="num ${cls}">${fmt(p.delta, 2)}</td>
        <td class="num">${fmt(p.gamma, 5)}</td>
        <td class="num">${fmt(p.theta, 2)}</td>
        <td class="num">${fmt(p.vega, 2)}</td>
      </tr>`;
    }).join("");
  } catch (e) { console.error("greeks:", e); }
}

// ───────── DRAM/Semis heatmap ─────────
function hmColorFor(pct) {
  if (pct == null) return "#1f2126";
  // Map [-5%..+5%] to red..green via HSL.
  const clamped = Math.max(-5, Math.min(5, pct));
  // -5 → hue 0 (red), +5 → hue 130 (green)
  const hue = 65 + clamped * 13;
  const sat = 55;
  const lit = 24 + Math.abs(clamped) * 1.5;
  return `hsl(${hue}, ${sat}%, ${lit}%)`;
}
async function refreshHeatmap() {
  try {
    const r = await fetch(API_PREFIX + "/api/sector-heatmap").then(r => r.json());
    if (r.error) {
      document.getElementById("hm-grid").innerHTML =
        `<div class="muted">heatmap error: ${r.error}</div>`;
      return;
    }
    document.getElementById("hm-asof").textContent = fmtTs(r.as_of);
    const bench = r.reference_mom_5d;
    const benchStr = bench != null ? `${r.reference} 5d ${bench >= 0 ? "+" : ""}${fmt(bench, 2)}%` : `${r.reference} —`;
    document.getElementById("hm-bench").textContent = "Benchmark: " + benchStr;

    const grid = document.getElementById("hm-grid");
    const buckets = r.buckets || [];
    grid.innerHTML = buckets.map(b => {
      const cells = (b.tickers || []).map(t => {
        const m5 = t.mom_5d;
        const rs = t.vs_sox_5d;
        const news = t.n || 0;
        const urg = t.urgent || 0;
        const bg = hmColorFor(m5);
        const rsStr = rs == null ? "" : `<span style="color:${rs >= 0 ? '#7fff00' : '#ff7b7b'};font-size:10px;margin-left:4px;">vs SOX ${rs >= 0 ? '+' : ''}${fmt(rs,1)}</span>`;
        const newsStr = news > 0
          ? `<span style="color:#dde1e7;font-size:10px;margin-left:6px;">📰 ${news}${urg ? `<span style="color:#ff4455">!</span>` : ""}</span>`
          : "";
        const rsi = t.rsi;
        const rsiStr = rsi == null ? "" : `<span style="color:${rsi > 70 ? '#ff7b7b' : (rsi < 30 ? '#80deea' : '#8b929d')};font-size:10px;margin-left:6px;">RSI ${fmt(rsi,0)}</span>`;
        const px = t.price == null ? "—" : "$" + fmt(t.price, 2);
        return `<div style="background:${bg};border:1px solid #1f2126;border-radius:4px;padding:6px 8px;min-width:130px;">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:6px;">
            <span style="font-weight:bold;color:#fff;">${t.ticker}</span>
            <span style="font-size:11px;color:#dde1e7;">${px}</span>
          </div>
          <div style="font-size:13px;color:${(m5 || 0) >= 0 ? '#7fff00' : '#ff7b7b'};font-weight:bold;">${m5 == null ? "—" : (m5 >= 0 ? "+" : "") + fmt(m5, 2) + "%"}</div>
          <div style="margin-top:2px;">${rsStr}${rsiStr}${newsStr}</div>
        </div>`;
      }).join("");
      const bm = b.avg_mom_5d;
      const bmStr = bm == null ? "—" : (bm >= 0 ? "+" : "") + fmt(bm, 2) + "%";
      const bmCls = (bm || 0) >= 0 ? "pos" : "neg";
      return `<div style="margin-bottom:14px;">
        <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px;">
          <span style="text-transform:uppercase;font-size:11px;letter-spacing:0.5px;color:#8b929d;">${b.name.replace(/_/g, " ")}</span>
          <span class="${bmCls}" style="font-size:11px;">avg 5d ${bmStr}</span>
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:6px;">${cells}</div>
      </div>`;
    }).join("");
  } catch (e) { console.error("heatmap:", e); }
}

// ───────── DecisionScorer per-position predictions ─────────
function scorerColor(v) {
  if (v == null) return "#dde1e7";
  if (v >= 2) return "#7fff00";
  if (v >= 0.5) return "#a5d6a7";
  if (v >= -0.5) return "#dde1e7";
  if (v >= -2) return "#ff9100";
  return "#ff4455";
}
function verdictBadge(v) {
  const colors = {
    STRONG_HOLD: ["#1b5e20", "#a5d6a7"],
    HOLD:        ["#2e7d32", "#c5e1a5"],
    NEUTRAL:     ["#37474f", "#dde1e7"],
    TRIM:        ["#ef6c00", "#ffe0b2"],
    EXIT:        ["#b71c1c", "#ffcdd2"],
  };
  const [bg, fg] = colors[v] || ["#1f2126", "#8b929d"];
  return `<span style="background:${bg};color:${fg};padding:1px 6px;border-radius:3px;font-size:11px;letter-spacing:0.5px;">${v || "—"}</span>`;
}
async function refreshScorer() {
  try {
    const r = await fetch(API_PREFIX + "/api/scorer-predictions").then(r => r.json());
    if (r.error) {
      document.getElementById("sc-meta").textContent = "scorer error: " + r.error;
      return;
    }
    document.getElementById("sc-asof").textContent = fmtTs(r.as_of);
    const meta = r.is_trained
      ? `trained (n=${r.n_train}) · regime mult ${fmt(r.regime_mult, 2)} · gate ≥ ${r.gate_threshold}`
      : `not trained yet (n=${r.n_train}/${r.gate_threshold}) — predictions will be 0.00 until threshold reached`;
    document.getElementById("sc-meta").textContent = meta;
    const tbody = document.querySelector("#sc-tbl tbody");
    const rows = r.predictions || [];
    if (!rows.length) {
      tbody.innerHTML = `<tr><td colspan="8" class="muted">no open stock positions</td></tr>`;
      return;
    }
    tbody.innerHTML = rows.map(p => {
      const v = p.pred_5d_return_pct;
      const sign = v >= 0 ? "+" : "";
      const newsCell = (p.news_count || 0) > 0
        ? `${p.news_count}${(p.news_urgent || 0) > 0 ? ` <span style="color:#ff4455">!</span>` : ""}`
        : "—";
      return `<tr>
        <td><strong>${p.ticker}</strong></td>
        <td class="num" style="color:${scorerColor(v)};font-weight:bold;">${v == null ? "—" : sign + fmt(v, 2) + "%"}</td>
        <td>${verdictBadge(p.verdict)}</td>
        <td class="num">${p.rsi == null ? "—" : fmt(p.rsi, 0)}</td>
        <td class="num">${p.macd == null ? "—" : fmt(p.macd, 3)}</td>
        <td class="num">${p.mom_5d == null ? "—" : (p.mom_5d >= 0 ? "+" : "") + fmt(p.mom_5d, 2) + "%"}</td>
        <td class="num">${p.mom_20d == null ? "—" : (p.mom_20d >= 0 ? "+" : "") + fmt(p.mom_20d, 2) + "%"}</td>
        <td class="num">${newsCell}</td>
      </tr>`;
    }).join("");
  } catch (e) { console.error("scorer:", e); }
}

// ───────── Deduped signals feed ─────────
async function refreshDedupedNews() {
  try {
    const r = await fetch(API_PREFIX + "/api/news-deduped?hours=6&min_score=4").then(r => r.json());
    if (r.error) {
      document.getElementById("nd-list").innerHTML = `<li class="muted">${r.error}</li>`;
      return;
    }
    document.getElementById("nd-asof").textContent = fmtTs(r.as_of);
    const meta = `${r.n_after_dedup} unique signals from ${r.n_raw} raw articles (compression ${fmt(r.compression_ratio, 1)}x) · halflife ${r.halflife_hours}h`;
    document.getElementById("nd-meta").textContent = meta;
    const items = (r.articles || []).slice(0, 15);
    const list = document.getElementById("nd-list");
    if (!items.length) {
      list.innerHTML = `<li class="muted">no signals in window</li>`;
      return;
    }
    list.innerHTML = items.map(a => {
      const score = a.ai_score != null ? fmt(a.ai_score, 1) : "—";
      const urgD = a.urgency_decayed != null ? fmt(a.urgency_decayed, 2) : "—";
      const dups = a.dup_count && a.dup_count > 1
        ? `<span class="muted" style="font-size:11px;margin-left:6px;">×${a.dup_count}</span>` : "";
      const urgBadge = (a.urgency_decayed || 0) >= 0.7
        ? `<span style="background:#ff1744;color:#fff;border-radius:3px;padding:1px 5px;font-size:10px;margin-right:6px;">URG ${urgD}</span>`
        : ((a.urgency_decayed || 0) > 0
            ? `<span style="background:#ff9100;color:#000;border-radius:3px;padding:1px 5px;font-size:10px;margin-right:6px;">u ${urgD}</span>`
            : "");
      const tickers = (a.tickers || []).slice(0, 4).map(t =>
        `<span style="background:#1f2126;color:#0acdff;font-size:10px;padding:1px 5px;border-radius:3px;margin-left:4px;">${t}</span>`
      ).join("");
      const title = (a.title || "").replace(/</g, "&lt;");
      const ts = fmtTs(a.first_seen);
      return `<li style="padding:6px 0;border-bottom:1px solid #1f2126;">
        ${urgBadge}<span style="color:#dde1e7;">${title}</span>${dups}
        <div class="muted" style="font-size:11px;margin-top:3px;">
          [${score}] ${a.source || "?"} · ${ts}${tickers}
        </div>
      </li>`;
    }).join("");
  } catch (e) { console.error("deduped:", e); }
}

// ───────── Position thesis (new 2026-05-15) ─────────
function verdictPill(v) {
  const colors = {
    STRONG_HOLD: ["#1b5e20", "#a5d6a7"],
    HOLD:        ["#33691e", "#c5e1a5"],
    WATCH:       ["#37474f", "#dde1e7"],
    TRIM:        ["#bf360c", "#ffccbc"],
    EXIT:        ["#b71c1c", "#ffcdd2"],
  };
  const [bg, fg] = colors[v] || ["#37474f", "#dde1e7"];
  return `<span style="background:${bg};color:${fg};border-radius:3px;padding:2px 8px;font-size:11px;font-weight:bold;letter-spacing:0.5px;">${v}</span>`;
}

async function refreshThesis() {
  try {
    const r = await fetch(API_PREFIX + "/api/position-thesis").then(r => r.json());
    document.getElementById("th-asof").textContent =
      fmtTs(r.as_of);
    if (r.error) {
      document.getElementById("th-meta").textContent = "error: " + r.error;
      document.getElementById("th-grid").innerHTML = "";
      return;
    }
    const cards = r.cards || [];
    const meta = `${r.n_positions || 0} open positions · scorer ` +
      (r.scorer_trained ? `trained (n=${r.scorer_n_train})` : `untrained (n=${r.scorer_n_train})`);
    document.getElementById("th-meta").textContent = meta;
    const grid = document.getElementById("th-grid");
    if (!cards.length) {
      grid.innerHTML = `<div class="muted">no open positions</div>`;
      return;
    }
    grid.innerHTML = cards.map(c => {
      const pl = c.unrealized_pl || 0;
      const plPct = c.pl_pct || 0;
      const plColor = pl >= 0 ? "#00c896" : "#ff4455";
      const plSign = pl >= 0 ? "+" : "";
      const news = c.news || {};
      const head = (news.headlines || [])[0];
      const headHtml = head
        ? `<div class="muted" style="font-size:11px;margin-top:4px;">📰 [${fmt(head.score,1)}] ${(head.title||"").replace(/</g,"&lt;").slice(0,120)}</div>`
        : `<div class="muted" style="font-size:11px;margin-top:4px;">no recent news</div>`;
      const ld = c.last_decision;
      const ldHtml = ld
        ? `<div style="font-size:11px;color:#90a4ae;margin-top:4px;">last: <strong>${ld.action.replace(/→.*/,'').trim()}</strong> conf=${ld.confidence!=null?fmt(ld.confidence,2):"?"} · ${(ld.reasoning||"").replace(/</g,"&lt;").slice(0,140)}</div>`
        : "";
      const rsi = c.rsi != null ? fmt(c.rsi, 0) : "—";
      const m5 = c.mom_5d != null ? (c.mom_5d >= 0 ? "+" : "") + fmt(c.mom_5d, 1) + "%" : "—";
      const m20 = c.mom_20d != null ? (c.mom_20d >= 0 ? "+" : "") + fmt(c.mom_20d, 1) + "%" : "—";
      const pred = c.scorer_pred_5d;
      const predHtml = pred != null
        ? `<span style="color:${scorerColor(pred)};">${pred>=0?"+":""}${fmt(pred,2)}%</span>`
        : "—";
      const newsPulse = news.n
        ? `${news.n}·<span style="color:#00c896">${news.bull||0}↑</span>/<span style="color:#ff4455">${news.bear||0}↓</span> avg ${fmt(news.avg_score,1)}`
        : "<span class='muted'>—</span>";
      return `<div style="background:#0d1117;border:1px solid #1f2126;border-radius:6px;padding:12px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
          <div><strong style="font-size:15px;color:#e0e0e0;">${c.ticker}</strong>
            <span class="muted" style="font-size:11px;margin-left:6px;">qty ${fmt(c.qty,4)} @ $${fmt(c.avg_cost,2)} · ${fmt(c.days_held,1)}d</span>
          </div>
          <div>${verdictPill(c.verdict)}</div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:12px;color:#dde1e7;margin-bottom:6px;">
          <span>P/L <span style="color:${plColor};font-weight:bold;">${plSign}$${fmt(pl,2)} (${plSign}${fmt(plPct,2)}%)</span></span>
          <span>scorer ${predHtml}</span>
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:10px;font-size:11px;color:#90a4ae;margin-bottom:4px;">
          <span>RSI ${rsi}</span><span>mom5 ${m5}</span><span>mom20 ${m20}</span>
          <span>news ${newsPulse}</span>
        </div>
        <div style="font-size:11px;color:#dde1e7;font-style:italic;margin-top:6px;">${c.thesis||"—"}</div>
        ${headHtml}
        ${ldHtml}
      </div>`;
    }).join("");
  } catch (e) { console.error("thesis:", e); }
}

// ───────── Drawdown anatomy (new 2026-05-15) ─────────
async function refreshDrawdown() {
  try {
    const r = await fetch(API_PREFIX + "/api/drawdown").then(r => r.json());
    document.getElementById("dd-asof").textContent =
      fmtTs(r.as_of);
    if (r.error) {
      document.getElementById("dd-pct").textContent = "err";
      return;
    }
    document.getElementById("dd-current").textContent = "$" + fmt(r.current_value, 2);
    document.getElementById("dd-peak").textContent = "$" + fmt(r.peak_value, 2);
    const ddPct = r.drawdown_pct || 0;
    const ddEl = document.getElementById("dd-pct");
    if (r.at_high_water) {
      ddEl.innerHTML = `<span style="color:#00c896;font-weight:bold;">◆ at high-water</span>`;
    } else {
      const col = ddPct <= -5 ? "#ff4455" : (ddPct <= -2 ? "#ff9100" : "#ffd54f");
      ddEl.innerHTML = `<span style="color:${col};">${fmt(ddPct,2)}% ($${fmt(r.drawdown_abs,2)})</span>`;
    }
    document.getElementById("dd-trough").textContent =
      r.trough_value != null ? `$${fmt(r.trough_value,2)} (${fmt(r.trough_pct,2)}%)` : "—";
    document.getElementById("dd-hours").textContent =
      r.hours_in_dd != null ? fmt(r.hours_in_dd, 1) + "h" : "—";
    document.getElementById("dd-rec").textContent =
      (r.at_high_water ? "100" : fmt(r.recovery_pct, 0)) + "%";
    const tbody = document.querySelector("#dd-tbl tbody");
    const rows = r.contributors || [];
    if (!rows.length) {
      tbody.innerHTML = `<tr><td colspan="7" class="muted">no open positions</td></tr>`;
      return;
    }
    tbody.innerHTML = rows.map(p => {
      const pl = p.unrealized_pl || 0;
      const plPct = p.pl_pct || 0;
      const color = pl >= 0 ? "#00c896" : "#ff4455";
      const dragBadge = p.drag
        ? `<span style="background:#b71c1c;color:#ffcdd2;border-radius:3px;padding:1px 5px;font-size:10px;">DRAG</span>`
        : `<span class="muted">—</span>`;
      return `<tr>
        <td><strong>${p.ticker}</strong> <span class="muted" style="font-size:10px;">${p.type||""}</span></td>
        <td class="num">${p.qty != null ? fmt(p.qty, 4) : "—"}</td>
        <td class="num">$${fmt(p.avg_cost,2)}</td>
        <td class="num">$${fmt(p.current_price,2)}</td>
        <td class="num" style="color:${color};font-weight:bold;">${pl>=0?"+":""}$${fmt(pl,2)}</td>
        <td class="num" style="color:${color};">${plPct>=0?"+":""}${fmt(plPct,2)}%</td>
        <td>${dragBadge}</td>
      </tr>`;
    }).join("");
  } catch (e) { console.error("drawdown:", e); }
}

// ───────── Confidence calibration + signal attribution (new 2026-05-15) ─────────
async function refreshCalibration() {
  try {
    const r = await fetch(API_PREFIX + "/api/calibration").then(r => r.json());
    document.getElementById("cal-asof").textContent =
      fmtTs(r.as_of);
    if (r.error) {
      document.getElementById("cal-meta").textContent = "error: " + r.error;
      return;
    }
    document.getElementById("cal-meta").textContent =
      `${r.n_decisions_parsed||0} decisions parsed · ${r.n_realized_trades||0} realized round-trips matched`;
    const confTbody = document.querySelector("#cal-conf-tbl tbody");
    const confRows = r.confidence_buckets || [];
    if (!confRows.some(b => b.n)) {
      confTbody.innerHTML = `<tr><td colspan="5" class="muted">no closed trades yet — calibration builds over time</td></tr>`;
    } else {
      confTbody.innerHTML = confRows.map(b => {
        const wrColor = b.win_rate >= 60 ? "#00c896" : (b.win_rate >= 40 ? "#ffd54f" : "#ff4455");
        const retColor = b.avg_return > 0 ? "#00c896" : "#ff4455";
        return `<tr>
          <td>${b.bucket}</td>
          <td class="num">${b.n}</td>
          <td class="num" style="color:${b.n?wrColor:'#8b929d'};">${b.n?fmt(b.win_rate,1)+"%":"—"}</td>
          <td class="num" style="color:${b.n?retColor:'#8b929d'};">${b.n?(b.avg_return>=0?"+":"")+fmt(b.avg_return,2)+"%":"—"}</td>
          <td class="num">${b.n?fmt(b.avg_conf,2):"—"}</td>
        </tr>`;
      }).join("");
    }
    const srcTbody = document.querySelector("#cal-src-tbl tbody");
    const srcRows = (r.signal_sources || []).sort((a,b) => (b.n||0)-(a.n||0));
    if (!srcRows.some(s => s.n)) {
      srcTbody.innerHTML = `<tr><td colspan="5" class="muted">no realized trades yet</td></tr>`;
    } else {
      srcTbody.innerHTML = srcRows.map(s => {
        const wrColor = s.win_rate >= 60 ? "#00c896" : (s.win_rate >= 40 ? "#ffd54f" : "#ff4455");
        const retColor = s.avg_return > 0 ? "#00c896" : "#ff4455";
        const bw = s.n
          ? `<span style="color:#00c896;">+${fmt(s.best,1)}%</span> / <span style="color:#ff4455;">${fmt(s.worst,1)}%</span>`
          : "—";
        return `<tr>
          <td><strong>${s.source}</strong></td>
          <td class="num">${s.n}</td>
          <td class="num" style="color:${s.n?wrColor:'#8b929d'};">${s.n?fmt(s.win_rate,1)+"%":"—"}</td>
          <td class="num" style="color:${s.n?retColor:'#8b929d'};">${s.n?(s.avg_return>=0?"+":"")+fmt(s.avg_return,2)+"%":"—"}</td>
          <td class="num" style="font-size:11px;">${bw}</td>
        </tr>`;
      }).join("");
    }
    const rTbody = document.querySelector("#cal-recent-tbl tbody");
    const recent = (r.recent_realized || []).slice().reverse();  // most recent first
    if (!recent.length) {
      rTbody.innerHTML = `<tr><td colspan="6" class="muted">no realized round-trips yet</td></tr>`;
    } else {
      rTbody.innerHTML = recent.slice(0, 12).map(t => {
        const ret = t.return_pct;
        const color = ret >= 0 ? "#00c896" : "#ff4455";
        const sign = ret >= 0 ? "+" : "";
        const buyTs = fmtTs(t.buy_ts);
        const sellTs = fmtTs(t.sell_ts);
        const conf = t.confidence != null ? fmt(t.confidence, 2) : "—";
        const reason = (t.reasoning_excerpt || "").replace(/</g, "&lt;");
        return `<tr>
          <td class="muted" style="font-size:11px;">${buyTs} → ${sellTs}</td>
          <td><strong>${t.ticker}</strong></td>
          <td class="num" style="color:${color};font-weight:bold;">${sign}${fmt(ret,2)}%</td>
          <td class="num">${conf}</td>
          <td>${t.source||"—"}</td>
          <td style="font-size:11px;color:#dde1e7;">${reason}</td>
        </tr>`;
      }).join("");
    }
  } catch (e) { console.error("calibration:", e); }
}

// ───────── Decision pipeline health (new 2026-05-15, agent 4) ─────────
async function refreshDecisionHealth() {
  try {
    const r = await fetch(API_PREFIX + "/api/decision-health").then(r => r.json());
    if (r.error) {
      document.getElementById("dh-reason").textContent = "error: " + r.error;
      return;
    }
    const vmap = {
      HEALTHY:  ["#1b5e20", "#a5d6a7"],
      DEGRADED: ["#b8860b", "#000000"],
      CRITICAL: ["#b71c1c", "#ffffff"],
      NO_DATA:  ["#1f2126", "#8b929d"],
    };
    // B9: dual-window — show 24h AND all-time, and make the headline
    // verdict the WORSE of the two so a clean recent window can't mask a
    // chronically broken all-time pipeline (and vice-versa). Verdict from
    // parse_fail_pct using the same thresholds as the analytics module
    // (>=50 CRITICAL, >=25 DEGRADED, else HEALTHY; <10 samples → NO_DATA).
    const w   = (r.windows && r.windows["24h"]) || {};
    const wAll = (r.windows && r.windows["all"]) || {};
    const rank = { CRITICAL: 3, DEGRADED: 2, HEALTHY: 1, NO_DATA: 0 };
    const windowVerdict = (win) => {
      if (!win || (win.total || 0) < 10) return "NO_DATA";
      const f = win.parse_fail_pct || 0;
      return f >= 50 ? "CRITICAL" : f >= 25 ? "DEGRADED" : "HEALTHY";
    };
    const v24 = windowVerdict(w), vAll = windowVerdict(wAll);
    // worst non-NO_DATA wins; if both NO_DATA fall back to server verdict.
    let worst = (rank[v24] >= rank[vAll]) ? v24 : vAll;
    if (worst === "NO_DATA" && r.verdict) worst = r.verdict;
    const [bg, fg] = vmap[worst] || vmap.NO_DATA;
    const vEl = document.getElementById("dh-verdict");
    vEl.textContent = `${worst} (24h: ${v24} · all-time: ${vAll})`;
    vEl.style.background = bg;
    vEl.style.color = fg;
    document.getElementById("dh-reason").textContent =
      (worst === v24 || worst === vAll || !r.verdict_reason)
        ? `worst of 24h (${v24}) and all-time (${vAll}) windows`
        : (r.verdict_reason || "");

    document.getElementById("dh-total").textContent = w.total != null ? w.total : "—";
    const failEl = document.getElementById("dh-fail");
    failEl.textContent = w.parse_fail_pct != null ? fmt(w.parse_fail_pct, 0) + "%" : "—";
    failEl.style.color = (w.parse_fail_pct || 0) >= 50 ? "#ff4455"
                       : (w.parse_fail_pct || 0) >= 25 ? "#ffa726" : "#4caf50";
    document.getElementById("dh-fills").textContent =
      (w.filled != null ? w.filled : "—") + (w.fill_pct != null ? ` (${fmt(w.fill_pct,1)}%)` : "");

    // all-time window stats (B9)
    const setAll = (id, t) => { const e = document.getElementById(id); if (e) e.textContent = t; };
    setAll("dh-total-all", wAll.total != null ? wAll.total : "—");
    const failAllEl = document.getElementById("dh-fail-all");
    if (failAllEl) {
      failAllEl.textContent = wAll.parse_fail_pct != null ? fmt(wAll.parse_fail_pct, 0) + "%" : "—";
      failAllEl.style.color = (wAll.parse_fail_pct || 0) >= 50 ? "#ff4455"
                            : (wAll.parse_fail_pct || 0) >= 25 ? "#ffa726" : "#4caf50";
    }
    setAll("dh-fills-all",
      (wAll.filled != null ? wAll.filled : "—") + (wAll.fill_pct != null ? ` (${fmt(wAll.fill_pct,1)}%)` : ""));
    const c = r.confidence || {};
    const trendArrow = {rising:" ↑", falling:" ↓", flat:""}[c.trend] || "";
    document.getElementById("dh-conf").textContent =
      c.avg != null ? fmt(c.avg, 2) + trendArrow : "—";
    const cad = r.cadence || {};
    document.getElementById("dh-lastfill").textContent =
      cad.hours_since_fill != null ? fmt(cad.hours_since_fill, 1) + "h" : "never";
    const sc = r.signal_count || {};
    document.getElementById("dh-sigs").textContent =
      sc.avg != null ? fmt(sc.avg, 1) : "—";

    // action mix bars
    const mixColors = {FILLED:"#4caf50", HOLD:"#5c6bc0", BLOCKED:"#ffa726",
                       NO_DECISION:"#ff4455", OTHER:"#8b929d"};
    const mix = r.action_mix || [];
    const mixEl = document.getElementById("dh-mix");
    if (!mix.length) {
      mixEl.innerHTML = '<div class="muted">no decisions yet</div>';
    } else {
      mixEl.innerHTML = mix.map(m => `
        <div style="display:flex;align-items:center;gap:8px;margin:3px 0;font-size:12px;">
          <span style="width:96px;color:#dde1e7;">${m.category}</span>
          <div style="flex:1;background:#1f2126;border-radius:3px;height:14px;overflow:hidden;">
            <div style="width:${m.pct}%;height:100%;background:${mixColors[m.category]||"#8b929d"};"></div>
          </div>
          <span class="muted" style="width:96px;text-align:right;">${m.n} · ${fmt(m.pct,1)}%</span>
        </div>`).join("");
    }

    // recent decision tape
    const tape = r.recent || [];
    const tb = document.querySelector("#dh-tape tbody");
    if (!tape.length) {
      tb.innerHTML = `<tr><td colspan="5" class="muted">no cycles</td></tr>`;
    } else {
      tb.innerHTML = tape.map(d => {
        const col = mixColors[d.category] || "#8b929d";
        const t = fmtTs(d.timestamp);
        return `<tr>
          <td class="muted">${t}</td>
          <td><span style="color:${col};font-weight:bold;">${d.category}</span></td>
          <td>${(d.action || "—").replace(/</g,"&lt;")}</td>
          <td class="num">${d.confidence != null ? fmt(d.confidence,2) : "—"}</td>
          <td class="num">${d.signal_count != null ? d.signal_count : "—"}</td>
        </tr>`;
      }).join("");
    }
  } catch (e) { console.error("decision-health:", e); }
}

// ───────── Capital deployment & liquidity (new 2026-05-15, agent 4) ─────────
async function refreshLiquidity() {
  try {
    const r = await fetch(API_PREFIX + "/api/liquidity").then(r => r.json());
    if (r.error) { document.getElementById("liq-headline").textContent = "error: " + r.error; return; }
    const smap = {
      NO_DRY_POWDER: ["#b71c1c", "#ffffff"],
      DRY_POWDER_LOW:["#b8860b", "#000000"],
      CASH_HEAVY:    ["#1565c0", "#ffffff"],
      BALANCED:      ["#1b5e20", "#a5d6a7"],
      NO_DATA:       ["#1f2126", "#8b929d"],
    };
    const [bg, fg] = smap[r.status] || smap.NO_DATA;
    const sEl = document.getElementById("liq-status");
    sEl.textContent = (r.status || "—").replace(/_/g, " ");
    sEl.style.background = bg; sEl.style.color = fg;
    document.getElementById("liq-headline").textContent = r.headline || "";
    document.getElementById("liq-cash").textContent =
      dollar(r.cash) + (r.cash_pct != null ? ` (${fmt(r.cash_pct,1)}%)` : "");
    const dEl = document.getElementById("liq-deployed");
    dEl.textContent = r.deployed_pct != null ? fmt(r.deployed_pct,1) + "%" : "—";
    dEl.style.color = (r.deployed_pct || 0) >= 98 ? "#ff4455"
                    : (r.deployed_pct || 0) >= 90 ? "#ffa726" : "#dde1e7";
    document.getElementById("liq-npos").textContent =
      (r.n_positions != null ? r.n_positions : "—") +
      (r.n_losers != null ? ` · ${r.n_losers}↓` : "");
    document.getElementById("liq-top").textContent =
      r.top_weight_pct != null ? fmt(r.top_weight_pct,1) + "%" +
        (r.largest_position ? ` ${r.largest_position}` : "") : "—";
    const uEl = document.getElementById("liq-upl");
    uEl.textContent = r.unrealized_pl != null
      ? dollar(r.unrealized_pl) + (r.unrealized_pl_pct != null ? ` (${fmt(r.unrealized_pl_pct,1)}%)` : "")
      : "—";
    uEl.style.color = (r.unrealized_pl || 0) < 0 ? "#ff4455"
                    : (r.unrealized_pl || 0) > 0 ? "#4caf50" : "#dde1e7";
    document.getElementById("liq-entry").textContent =
      r.days_since_last_entry != null ? fmt(r.days_since_last_entry,1) + "d ago" : "—";
    const dep = Math.max(0, Math.min(100, r.deployed_pct || 0));
    document.getElementById("liq-bar").innerHTML =
      `<div style="width:${dep}%;background:#5c6bc0;height:100%;"></div>` +
      `<div style="width:${100-dep}%;background:#2e7d32;height:100%;"></div>`;
    document.getElementById("liq-bar-legend").textContent =
      `deployed ${fmt(dep,1)}%  ·  cash ${fmt(100-dep,1)}%` +
      (r.can_act_on_signal === false ? "  ·  ⚠ cannot act on a new BUY" : "");
    const fl = r.flags || [];
    document.getElementById("liq-flags").innerHTML = fl.length
      ? fl.map(f => `<div style="margin:3px 0;">• ${f.replace(/</g,"&lt;")}</div>`).join("")
      : '<span class="muted">no liquidity flags</span>';
  } catch (e) { console.error("liquidity:", e); }
}

// ───────── Decision failure forensics (new 2026-05-15, agent 4) ─────────
async function refreshDecisionForensics() {
  try {
    const r = await fetch(API_PREFIX + "/api/decision-forensics").then(r => r.json());
    if (r.error) { document.getElementById("df-reason").textContent = "error: " + r.error; return; }
    const vmap = {
      HEALTHY:  ["#1b5e20", "#a5d6a7"],
      DEGRADED: ["#b8860b", "#000000"],
      CRITICAL: ["#b71c1c", "#ffffff"],
      NO_DATA:  ["#1f2126", "#8b929d"],
    };
    const [bg, fg] = vmap[r.verdict] || vmap.NO_DATA;
    const vEl = document.getElementById("df-verdict");
    vEl.textContent = r.verdict + (r.verdict_window ? ` (${r.verdict_window})` : "");
    vEl.style.background = bg; vEl.style.color = fg;
    document.getElementById("df-reason").textContent = r.verdict_reason || "";
    document.getElementById("df-hint").textContent = r.hint || "";
    document.getElementById("df-nfail").textContent =
      (r.n_failures != null ? r.n_failures : "—") +
      (r.failure_rate_pct != null ? ` (${fmt(r.failure_rate_pct,0)}% all)` : "");
    const rEl = document.getElementById("df-rate");
    rEl.textContent = r.failure_rate_24h_pct != null ? fmt(r.failure_rate_24h_pct,0) + "%" : "—";
    rEl.style.color = (r.failure_rate_24h_pct || 0) >= 50 ? "#ff4455"
                    : (r.failure_rate_24h_pct || 0) >= 25 ? "#ffa726" : "#4caf50";
    document.getElementById("df-retry").textContent =
      r.retry_exhausted != null ? r.retry_exhausted : "—";
    document.getElementById("df-dom").textContent =
      (r.dominant_mode || "—").replace(/_/g, " ");
    const bm = r.by_market || {};
    document.getElementById("df-open").textContent =
      bm.open ? fmt(bm.open.fail_pct,0) + "%" : "—";
    document.getElementById("df-closed").textContent =
      bm.closed ? fmt(bm.closed.fail_pct,0) + "%" : "—";

    const modeColors = {
      TIMEOUT_EMPTY:"#ff4455", TRUNCATED:"#ff7043", NO_JSON:"#ab47bc",
      FENCED:"#ffa726", PROSE_WRAPPED:"#ffca28", MALFORMED_JSON:"#ef5350",
      EMPTY:"#8b929d", LEGACY_UNKNOWN:"#5c6bc0", OTHER:"#8b929d",
    };
    const mix = r.mode_mix || [];
    const mixEl = document.getElementById("df-mix");
    mixEl.innerHTML = mix.length ? mix.map(m => `
      <div style="display:flex;align-items:center;gap:8px;margin:3px 0;font-size:12px;">
        <span style="width:128px;color:#dde1e7;">${m.mode.replace(/_/g," ")}</span>
        <div style="flex:1;background:#1f2126;border-radius:3px;height:14px;overflow:hidden;">
          <div style="width:${m.pct}%;height:100%;background:${modeColors[m.mode]||"#8b929d"};"></div>
        </div>
        <span class="muted" style="width:84px;text-align:right;">${m.n} · ${fmt(m.pct,0)}%</span>
      </div>`).join("") : '<div class="muted">no NO_DECISION cycles 🎉</div>';

    const hrs = r.hourly || [];
    const hEl = document.getElementById("df-hourly");
    if (!hrs.length) {
      hEl.innerHTML = '<div class="muted">no cycles in last 24h</div>';
    } else {
      hEl.innerHTML = hrs.map(h => {
        const ph = Math.max(4, Math.round((h.fail_pct||0) * 0.42));
        const col = (h.fail_pct||0) >= 50 ? "#ff4455" : (h.fail_pct||0) >= 25 ? "#ffa726" : "#4caf50";
        const lbl = fmtTs(h.hour, "time");
        return `<div title="${lbl}  ${h.failures}/${h.total} failed (${fmt(h.fail_pct,0)}%)"
          style="flex:1;min-width:5px;height:${ph}px;background:${col};border-radius:2px 2px 0 0;"></div>`;
      }).join("");
    }

    // Decision-loss clock — fold the current-regime history onto 24 UTC hours.
    const chEl = document.getElementById("df-clock-hint");
    if (chEl) chEl.textContent = r.clock_hint || "";
    const hod = r.hour_of_day || [];
    const byHour = {};
    hod.forEach(b => { byHour[b.hour] = b; });
    const worstSet = new Set((r.worst_hours || []).map(b => b.hour));
    const ckEl = document.getElementById("df-clock");
    const axEl = document.getElementById("df-clock-axis");
    if (ckEl) {
      if (!hod.length) {
        ckEl.innerHTML = '<div class="muted">no current-regime cycles yet</div>';
        if (axEl) axEl.innerHTML = "";
      } else {
        let bars = "", axis = "";
        for (let h = 0; h < 24; h++) {
          const b = byHour[h];
          if (b) {
            const fp = b.fail_pct || 0;
            const ph = Math.max(4, Math.round(fp * 0.42));
            const col = fp >= 50 ? "#ff4455" : fp >= 25 ? "#ffa726" : "#4caf50";
            const ring = worstSet.has(h) ? "outline:2px solid #ffd479;outline-offset:-1px;" : "";
            bars += `<div title="${String(h).padStart(2,'0')}:00 UTC  ${b.failures}/${b.total} failed (${fmt(fp,0)}%)"
              style="flex:1;min-width:4px;height:${ph}px;background:${col};${ring}border-radius:2px 2px 0 0;"></div>`;
          } else {
            bars += `<div title="${String(h).padStart(2,'0')}:00 UTC  no cycles"
              style="flex:1;min-width:4px;height:3px;background:#2a2d34;border-radius:2px 2px 0 0;"></div>`;
          }
          axis += `<div style="flex:1;min-width:4px;text-align:center;">${h % 6 === 0 ? h : ""}</div>`;
        }
        ckEl.innerHTML = bars;
        if (axEl) axEl.innerHTML = axis;
      }
    }

    const tape = r.recent_failures || [];
    const tb = document.querySelector("#df-tape tbody");
    tb.innerHTML = tape.length ? tape.map(d => {
      const t = fmtTs(d.timestamp);
      const col = modeColors[d.mode] || "#8b929d";
      const ex = (d.excerpt || "—").replace(/</g,"&lt;").slice(0,200);
      return `<tr>
        <td class="muted">${t}</td>
        <td><span style="color:${col};font-weight:bold;">${d.mode.replace(/_/g," ")}</span></td>
        <td class="muted">${d.market_open ? "open" : "—"}</td>
        <td style="font-family:monospace;color:#aab;max-width:380px;word-break:break-all;">${ex}</td>
      </tr>`;
    }).join("") : `<tr><td colspan="4" class="muted">no failures recorded</td></tr>`;
  } catch (e) { console.error("decision-forensics:", e); }
}

// ───────── Decision drought drift (new 2026-05-16, agent 4) ─────────
async function refreshDecisionDrought() {
  try {
    const r = await fetch(API_PREFIX + "/api/decision-drought").then(r => r.json());
    if (r.error) { document.getElementById("dd-reason").textContent = "error: " + r.error; return; }
    const vmap = {
      OK:          ["#1b5e20", "#a5d6a7"],
      NEVER_TRADED:["#b8860b", "#000000"],
      STUCK:       ["#b8860b", "#000000"],
      BLEEDING:    ["#b71c1c", "#ffffff"],
      NO_DATA:     ["#1f2126", "#8b929d"],
    };
    const [bg, fg] = vmap[r.verdict] || vmap.NO_DATA;
    const vEl = document.getElementById("dd-verdict");
    vEl.textContent = r.verdict || "—";
    vEl.style.background = bg; vEl.style.color = fg;
    document.getElementById("dd-reason").textContent = r.verdict_reason || "";
    document.getElementById("dd-fills").textContent =
      (r.n_fills != null ? r.n_fills : "—") + " / " + (r.n_cycles != null ? r.n_cycles : "—");
    document.getElementById("dd-n").textContent = r.n_droughts != null ? r.n_droughts : "—";
    document.getElementById("dd-npar").textContent =
      r.n_paralysis_droughts != null ? r.n_paralysis_droughts : "—";
    const bEl = document.getElementById("dd-bleed");
    const bleed = r.involuntary_alpha_bleed_pct;
    bEl.textContent = bleed != null ? fmt(bleed, 2) + "%" : "—";
    bEl.style.color = (bleed || 0) <= -1.0 ? "#ff4455" : (bleed || 0) < 0 ? "#ffa726" : "#4caf50";

    const cur = r.current_drought;
    const cEl = document.getElementById("drought-current");
    if (!cur) {
      cEl.textContent = "no ongoing drought — last cycle was a fill";
      cEl.style.borderColor = "#1f2126";
    } else {
      const para = cur.kind === "PARALYSIS";
      cEl.style.borderColor = para ? "#ff4455" : "#2b3038";
      const a = cur.alpha_pct;
      cEl.innerHTML = `<b style="color:${para?'#ff6b6b':'#dde1e7'};">ONGOING ${cur.kind}</b> — `
        + `${fmt(cur.duration_hours,1)}h, ${cur.n_cycles} cycles `
        + `(${cur.n_no_decision} NO_DECISION / ${cur.n_hold} HOLD). `
        + `portfolio ${cur.portfolio_pct!=null?fmt(cur.portfolio_pct,2)+'%':'—'}, `
        + `S&P ${cur.spy_pct!=null?fmt(cur.spy_pct,2)+'%':'—'}, `
        + `<b style="color:${a!=null&&a<0?'#ff4455':'#4caf50'};">alpha ${a!=null?fmt(a,2)+'%':'—'}</b>`;
    }

    const kindCol = { PARALYSIS:"#ff4455", DELIBERATE_HOLD:"#4caf50", MIXED:"#ffa726" };
    const tape = r.droughts || [];
    const tb = document.querySelector("#dd-tape tbody");
    tb.innerHTML = tape.length ? tape.map(d => {
      const t = fmtTs(d.start);
      const a = d.alpha_pct;
      const acol = a == null ? "#8b929d" : a < 0 ? "#ff4455" : "#4caf50";
      return `<tr>
        <td class="muted">${t}${d.ongoing?' <span style="color:#ffd479;">●live</span>':''}</td>
        <td class="num">${d.duration_hours!=null?fmt(d.duration_hours,1):'—'}</td>
        <td class="num">${d.n_cycles}</td>
        <td><span style="color:${kindCol[d.kind]||'#8b929d'};font-weight:bold;">${d.kind.replace(/_/g," ")}</span></td>
        <td class="num">${fmt(d.no_decision_pct,0)}</td>
        <td class="num">${d.portfolio_pct!=null?fmt(d.portfolio_pct,2):'—'}</td>
        <td class="num">${d.spy_pct!=null?fmt(d.spy_pct,2):'—'}</td>
        <td class="num" style="color:${acol};font-weight:bold;">${a!=null?fmt(a,2):'—'}</td>
      </tr>`;
    }).join("") : `<tr><td colspan="8" class="muted">no multi-cycle droughts</td></tr>`;
  } catch (e) { console.error("decision-drought:", e); }
}

// ───────── News edge (new 2026-05-16, agent 4) ─────────
async function refreshNewsEdge() {
  try {
    const r = await fetch(API_PREFIX + "/api/news-edge").then(r => r.json());
    if (r.error) { document.getElementById("ne-reason").textContent = "error: " + r.error; return; }
    const vmap = {
      EDGE_CONFIRMED:    ["#1b5e20", "#a5d6a7"],
      WEAK_EDGE:         ["#b8860b", "#000000"],
      NO_EDGE:           ["#b71c1c", "#ffffff"],
      INSUFFICIENT_DATA: ["#1f2126", "#8b929d"],
      NO_DATA:           ["#1f2126", "#8b929d"],
      ERROR:             ["#b71c1c", "#ffffff"],
    };
    const [bg, fg] = vmap[r.verdict] || vmap.NO_DATA;
    const vEl = document.getElementById("ne-verdict");
    vEl.textContent = (r.verdict || "—").replace(/_/g," ");
    vEl.style.background = bg; vEl.style.color = fg;
    document.getElementById("ne-reason").textContent = r.verdict_reason || "";
    document.getElementById("ne-days").textContent =
      r.lookback_days != null ? r.lookback_days + "d" : "—";
    document.getElementById("ne-narts").textContent = r.n_articles != null ? r.n_articles : "—";
    document.getElementById("ne-nres").textContent = r.n_resolved != null ? r.n_resolved : "—";
    document.getElementById("ne-ntk").textContent =
      r.n_tickers_priced != null ? r.n_tickers_priced : "—";
    document.getElementById("ne-ref").textContent =
      r.reference_horizon != null ? r.reference_horizon + "d" : "—";

    const cell = (h) => {
      // B8: no observations (n=0/missing) → greyed em-dash, never a
      // misleading 0% / NaN. h carries n alongside mean_abnormal_pct.
      if (!h || !h.n || h.mean_abnormal_pct == null) return '<span class="muted">—</span>';
      const v = h.mean_abnormal_pct;
      const col = v > 0 ? "#4caf50" : v < 0 ? "#ff4455" : "#8b929d";
      const raw = h.mean_raw_pct != null ? ` <span class="muted">(${fmt(h.mean_raw_pct,1)})</span>` : "";
      return `<span style="color:${col};font-weight:bold;">${fmt(v,2)}</span>${raw}`;
    };
    const refH = String(r.reference_horizon || 3);
    const bands = r.bands || [];
    const tb = document.querySelector("#ne-bands tbody");
    tb.innerHTML = bands.length ? bands.map(b => {
      const h = b.horizons || {};
      // n + hit% track the adaptive reference horizon (the one the verdict is
      // judged on) — not a hardcoded 3d, which would read 0 while the 1d cell
      // shows real numbers in early/low-history data.
      const hr = h[refH] || {};
      const nRef = hr.n || 0;
      const hit = hr.abnormal_hit_rate;
      // B8: zero-observation band → greyed em-dash for n and hit-rate
      // rather than a misleading "0" / "—%".
      return `<tr>
        <td><b>${b.band}</b></td>
        <td class="num">${nRef ? nRef : '<span class="muted">—</span>'}</td>
        <td class="num">${cell(h["1"])}</td>
        <td class="num">${cell(h["3"])}</td>
        <td class="num">${cell(h["5"])}</td>
        <td class="num">${(nRef && hit!=null) ? fmt(hit,0)+'%' : '<span class="muted">—</span>'}</td>
      </tr>`;
    }).join("") : `<tr><td colspan="6" class="muted">no priced articles in window</td></tr>`;

    const u = r.by_urgency || {};
    const ur = (u.urgent||{})["3"] || {}, no = (u.normal||{})["3"] || {};
    const fmtAbn = (x) => (x && x.n && x.mean_abnormal_pct != null)
      ? `${fmt(x.mean_abnormal_pct,2)}% (n=${x.n})`
      : '<span class="muted">—</span>';
    document.getElementById("ne-urg").innerHTML =
      `urgent: <b style="color:#ffd479;">${fmtAbn(ur)}</b> &nbsp;·&nbsp; normal: <b>${fmtAbn(no)}</b>`;
  } catch (e) { console.error("news-edge:", e); }
}

// ───────── Scorer reliability + confidence intervals (new 2026-05-15, agent 4) ─────────
async function refreshScorerConfidence() {
  try {
    const r = await fetch(API_PREFIX + "/api/scorer-confidence").then(r => r.json());
    if (r.error) {
      document.getElementById("scrl-meta").textContent = "error: " + r.error;
      return;
    }
    document.getElementById("scrl-asof").textContent =
      fmtTs(r.as_of);
    const o = r.overall;
    if (!o) {
      document.getElementById("scrl-meta").textContent =
        `scorer not ready — ${r.n_samples || 0} replay samples (need more outcomes)`;
      return;
    }
    document.getElementById("scrl-meta").textContent =
      `trained on n=${r.n_train} · replayed over ${r.n_samples} historical outcomes · ` +
      `residual = predicted − realized return`;
    const dirEl = document.getElementById("scrl-dir");
    dirEl.textContent = fmt(o.directional_accuracy_pct, 1) + "%";
    dirEl.style.color = o.directional_accuracy_pct >= 65 ? "#4caf50"
                      : o.directional_accuracy_pct >= 55 ? "#ffa726" : "#ff4455";
    document.getElementById("scrl-mae").textContent = "±" + fmt(o.mae, 2) + "%";
    document.getElementById("scrl-band").textContent =
      fmt(o.resid_p10, 1) + " … +" + fmt(o.resid_p90, 1);
    document.getElementById("scrl-n").textContent = r.n_samples;

    // held positions with empirical band
    const pos = r.positions || [];
    const pb = document.querySelector("#scrl-pos tbody");
    if (!pos.length) {
      pb.innerHTML = `<tr><td colspan="6" class="muted">no open stock positions</td></tr>`;
    } else {
      const trustColor = {high:"#4caf50", medium:"#ffa726", low:"#ff4455", none:"#8b929d"};
      pb.innerHTML = pos.map(p => {
        const v = p.pred_5d_return_pct;
        const iv = p.interval || {};
        const range = (iv.low != null && iv.high != null)
          ? `${iv.low >= 0 ? "+" : ""}${fmt(iv.low,1)}% … ${iv.high >= 0 ? "+" : ""}${fmt(iv.high,1)}%`
          : "—";
        return `<tr>
          <td><strong>${p.ticker}</strong></td>
          <td class="num" style="color:${scorerColor(v)};font-weight:bold;">${v == null ? "—" : (v>=0?"+":"") + fmt(v,2) + "%"}</td>
          <td class="num" style="color:#dde1e7;">${range}</td>
          <td>${verdictBadge(p.verdict)}</td>
          <td class="num">${iv.directional_accuracy_pct != null ? fmt(iv.directional_accuracy_pct,0) + "%" : "—"}</td>
          <td><span style="color:${trustColor[iv.reliability]||"#8b929d"};">${iv.reliability || "—"}</span></td>
        </tr>`;
      }).join("");
    }

    // calibration table
    const cb = document.querySelector("#scrl-cal tbody");
    const buckets = r.buckets || [];
    if (!buckets.length) {
      cb.innerHTML = `<tr><td colspan="6" class="muted">not enough samples</td></tr>`;
    } else {
      cb.innerHTML = buckets.map(b => {
        // B7: mae/bias are in percentage points (verified in
        // analytics/scorer_confidence.py — residual = pred-target return %),
        // NOT 0-1 decimals; task's 0.15/0.1 thresholds were decimal-scale
        // and rescale to pp as 5 / 3. Red the MAE cell when either the
        // absolute error (>5pp) or systematic bias (>3pp) shows the bucket
        // is materially miscalibrated.
        const miscal = (b.mae != null && b.mae > 5)
                       || (b.bias != null && Math.abs(b.bias) > 3);
        const maeStyle = miscal
          ? 'background:rgba(255,68,85,0.16);color:#ff4455;font-weight:bold;'
          : '';
        const maeTitle = miscal
          ? ` title="miscalibrated — MAE ${fmt(b.mae,1)}pp${b.bias!=null?', bias '+(b.bias>=0?'+':'')+fmt(b.bias,1)+'pp':''}"`
          : '';
        return `<tr>
        <td>${(b.pred_lo>=0?"+":"") + fmt(b.pred_lo,1)}% … ${(b.pred_hi>=0?"+":"") + fmt(b.pred_hi,1)}%</td>
        <td class="num">${b.n}</td>
        <td class="num" style="color:${scorerColor(b.mean_actual)};">${(b.mean_actual>=0?"+":"") + fmt(b.mean_actual,2)}%</td>
        <td class="num muted">${fmt(b.resid_p10,1)} / +${fmt(b.resid_p90,1)}</td>
        <td class="num" style="${maeStyle}"${maeTitle}>±${fmt(b.mae,1)}</td>
        <td class="num" style="color:${b.directional_accuracy_pct>=65?"#4caf50":b.directional_accuracy_pct>=55?"#ffa726":"#ff4455"};">${fmt(b.directional_accuracy_pct,0)}%</td>
      </tr>`;
      }).join("");
    }
  } catch (e) { console.error("scorer-confidence:", e); }
}

// ───────── Signal Integrity validation ─────────
async function refreshValidation() {
  try {
    const r = await fetch(API_PREFIX + "/api/validation").then(r => r.json());
    const results = (r && r.results) || [];
    const latest = results[results.length - 1];
    if (!latest) return;

    const pv = latest.permutation_test || {};
    // verdict may be a compound string like "UNDERPOWERED — too few valid
    // permutations"; key the color map on the leading token.
    const rawVerdict = pv.verdict || "—";
    const verdictKey = String(rawVerdict).split(/[\s—-]/)[0];
    const verdictColor = {
      SIGNIFICANT: "#00c896",
      INCONCLUSIVE: "#fbbf24",
      UNDERPOWERED: "#fbbf24",
      WORSE_THAN_RANDOM: "#ff4455",
      NO_EDGE: "#ff4455",
      UNKNOWN: "#8b929d",
    }[verdictKey] || "#8b929d";
    const verdictEl = document.getElementById("val-perm-verdict");
    if (verdictEl) {
      verdictEl.textContent = verdictKey === "UNDERPOWERED"
        ? "⚠ underpowered" : rawVerdict;
      verdictEl.style.color = verdictColor;
    }
    const setText = (id, t) => { const el = document.getElementById(id); if (el) el.textContent = t; };
    setText("val-perm-pvalue", pv.p_value != null ? `p=${Number(pv.p_value).toFixed(3)}` : "p=—");
    setText("val-perm-zscore", pv.z_score != null ? `z=${Number(pv.z_score).toFixed(2)}` : "z=—");
    setText("val-perm-original",
      pv.original_return != null ? `Strategy: ${Number(pv.original_return).toFixed(1)}%` : "");
    // A3: surface successful vs attempted permutations so a confident pill
    // off ~5 valid shuffles is visible (n_successful/n_attempted from
    // validation.run_permutation_test).
    const nSucc = pv.n_successful != null ? pv.n_successful : (pv.n_permutations || 0);
    const nAtt = pv.n_attempted != null ? pv.n_attempted : nSucc;
    setText("val-perm-shuffled",
      pv.permuted_mean != null
        ? `Shuffled mean: ${Number(pv.permuted_mean).toFixed(1)}%  (${nSucc}/${nAtt} successful)`
        : "");

    const audit = latest.label_audit || {};
    const rate = audit.contamination_rate;
    const auditVerdict = audit.verdict || "";
    // A2: RETROACTIVE_COLLECTION is architectural (historical articles are
    // all scraped in 2026 → trivially "stale"), NOT a validity threat.
    // Render it as a neutral grey info badge, not a red contamination
    // alarm. Only llm-hindsight (HIGH_CONTAMINATION) is genuinely red.
    const isRetro = auditVerdict === "RETROACTIVE_COLLECTION";
    const contamColor = isRetro ? "#8b929d"
      : rate == null ? "#8b929d"
      : auditVerdict === "HIGH_CONTAMINATION" ? "#ff4455"
      : rate > 0.5 ? "#ff4455"
      : rate > 0.2 ? "#fbbf24"
      : "#00c896";
    const contamEl = document.getElementById("val-contam-rate");
    if (contamEl) {
      contamEl.textContent = isRetro ? "retroactive"
        : rate == null ? "—" : `${(rate * 100).toFixed(0)}%`;
      contamEl.style.color = contamColor;
    }
    const llmN = audit.llm_contaminated_count;
    setText("val-contam-detail",
      audit.total_articles != null
        ? (isRetro
            ? `${audit.contaminated_count}/${audit.total_articles} collected late · ${llmN != null ? llmN : 0} Claude-labeled · architectural, not a backtest-validity threat`
            : `${audit.contaminated_count}/${audit.total_articles} articles · verdict: ${auditVerdict || "—"}`)
        : "");

    setText("val-last-cycle", latest.cycle != null ? `cycle ${latest.cycle}` : "—");
    setText("val-last-window", latest.window || "");
    setText("val-last-when", latest.timestamp ? new Date(latest.timestamp).toLocaleString() : "");
  } catch (e) {
    console.error("validation:", e);
  }
}

async function refreshDisagreement() {
  try {
    const r = await fetch(API_PREFIX + "/api/disagreement").then(r => r.json());
    if (r.error) {
      document.getElementById("dis-meta").textContent = "error: " + r.error;
      return;
    }
    document.getElementById("dis-asof").textContent =
      fmtTs(r.as_of);
    const c = r.counts || {};
    const setC = (id, v, col) => {
      const e = document.getElementById(id);
      e.textContent = (v == null ? "—" : v);
      if (col) e.style.color = col;
    };
    setC("dis-high", c.HIGH, (c.HIGH > 0) ? "#ff4455" : "#8b929d");
    setC("dis-med", c.MEDIUM, (c.MEDIUM > 0) ? "#ffa726" : "#8b929d");
    setC("dis-aln", c.ALIGNED, "#4caf50");
    setC("dis-n", r.n_positions);
    if (!r.scorer_trained) {
      document.getElementById("dis-meta").textContent =
        "scorer not trained yet — needs ≥500 decision outcomes before it can disagree with Opus";
    } else if (!r.n_positions) {
      document.getElementById("dis-meta").textContent =
        "no open stock positions to compare";
    } else {
      const h = c.HIGH || 0;
      document.getElementById("dis-meta").innerHTML = h > 0
        ? `<span style="color:#ff4455;font-weight:bold;">${h} position(s) where Opus is overriding the ML safety net</span> — scorer says exit/trim while Opus is still long. Canonical "why is the book losing money?" check.`
        : "scorer and Opus are aligned on every held position";
    }
    const tb = document.querySelector("#dis-tbl tbody");
    const rows = r.rows || [];
    if (!rows.length) {
      tb.innerHTML = `<tr><td colspan="6" class="muted">—</td></tr>`;
    } else {
      const sevColor = { HIGH: "#ff4455", MEDIUM: "#ffa726", ALIGNED: "#4caf50" };
      const actCls = a => !a ? "hold"
        : a.startsWith("SELL") ? "sell"
        : a === "HOLD" ? "hold" : "buy";
      let html = rows.map(x => {
        const p = x.scorer_pred_5d_pct;
        const od = x.off_distribution;
        const predTxt = p == null ? "—"
          : ((p >= 0 ? "+" : "") + fmt(p, 1) + "%" + (od ? " *" : ""));
        return `<tr>
          <td><strong>${x.ticker}</strong></td>
          <td>${verdictBadge(x.scorer_verdict)}</td>
          <td class="num" style="color:${od ? '#8b929d' : scorerColor(p)};">${predTxt}</td>
          <td><span class="pill ${actCls(x.last_action)}">${x.last_action || '—'}</span></td>
          <td><span style="color:${sevColor[x.severity] || '#8b929d'};font-weight:bold;">${x.severity}</span></td>
          <td class="muted" style="font-size:12px;">${x.label || ''}</td>
        </tr>`;
      }).join("");
      if (rows.some(x => x.off_distribution)) {
        html += `<tr><td colspan="6" class="muted" style="font-size:11px;">* off-distribution — scorer extrapolated past its label support; this conflict is de-weighted, not a real fight</td></tr>`;
      }
      tb.innerHTML = html;
    }
  } catch (e) {
    console.error("disagreement:", e);
  }
}

// ───────── Behavioural edge + orphaned-endpoint panels (2026-05-16, agent 4) ─────────
// All three endpoints are absent on a paper-trader process that booted before
// their commit (trade-asymmetry is brand new; capital-paralysis &
// open-attribution shipped in c994cba). Degrade to an explicit "restart to
// apply" message instead of a silent console error — mirrors the /api/build-info
// stale-banner contract rather than looking broken.
async function fetchMaybeStale(path) {
  try {
    const resp = await fetch(API_PREFIX + path);
    if (!resp.ok) return { __unavailable: true, __code: resp.status };
    const ct = resp.headers.get("content-type") || "";
    if (!ct.includes("json")) return { __unavailable: true };
    return await resp.json();
  } catch (e) { return { __unavailable: true }; }
}
function markStale(badgeId, headlineId, what) {
  const b = document.getElementById(badgeId);
  if (b) { b.textContent = "UNAVAILABLE"; b.style.background = "#3a2a00"; b.style.color = "#ffd479"; }
  const h = document.getElementById(headlineId);
  if (h) h.textContent = what + " not on the running process — restart paper-trader to apply (see /api/build-info `stale`).";
}
const _sgn = v => (v == null ? "" : v >= 0 ? "+" : "");
const _plColor = v => (v == null ? "#8b929d" : v > 0 ? "#4caf50" : v < 0 ? "#ff4455" : "#dde1e7");

async function refreshTradeAsymmetry() {
  const r = await fetchMaybeStale("/api/trade-asymmetry");
  if (r.__unavailable) { markStale("ta-verdict", "ta-headline", "Behavioural-edge endpoint"); return; }
  if (r.error) { document.getElementById("ta-headline").textContent = "error: " + r.error; return; }
  const vmap = {
    PAYOFF_TRAP:       ["#b71c1c", "#ffffff"],
    DISPOSITION_BLEED: ["#b8860b", "#000000"],
    EDGE_POSITIVE:     ["#1b5e20", "#a5d6a7"],
    FLAT:              ["#1f2126", "#8b929d"],
  };
  const stateBadge = { STABLE: null, EMERGING: ["#3a2a00", "#ffd479"], NO_DATA: ["#1f2126", "#8b929d"] };
  const vEl = document.getElementById("ta-verdict");
  if (r.state === "STABLE" && r.verdict) {
    const [bg, fg] = vmap[r.verdict] || vmap.FLAT;
    vEl.textContent = r.verdict.replace(/_/g, " ");
    vEl.style.background = bg; vEl.style.color = fg;
  } else {
    const [bg, fg] = stateBadge[r.state] || stateBadge.NO_DATA;
    vEl.textContent = r.state;
    vEl.style.background = bg; vEl.style.color = fg;
  }
  document.getElementById("ta-headline").textContent = r.headline || "";
  const exp = document.getElementById("ta-exp");
  exp.textContent = r.expectancy_usd != null ? _sgn(r.expectancy_usd) + "$" + fmt(Math.abs(r.expectancy_usd)) : "—";
  exp.style.color = _plColor(r.expectancy_usd);
  document.getElementById("ta-payoff").textContent = r.payoff_ratio != null ? fmt(r.payoff_ratio) : "—";
  const wr = document.getElementById("ta-wr");
  const belowBreakeven = (r.actual_win_rate_pct != null && r.breakeven_win_rate_pct != null
                          && r.actual_win_rate_pct < r.breakeven_win_rate_pct);
  // B3: show the denominator (n round-trips) next to the win-rate %.
  // B4: explicit ⚠ below-breakeven badge, not just a red number, when the
  // actual win-rate cannot carry the payoff ratio (the trap).
  if (r.actual_win_rate_pct != null) {
    const nrt = r.n_round_trips != null ? r.n_round_trips : "?";
    wr.innerHTML = fmt(r.actual_win_rate_pct, 1) + "% <span class=\"muted\" style=\"font-size:11px;\">(n=" + nrt + ")</span>"
      + (belowBreakeven
          ? ' <span style="background:#b71c1c;color:#fff;border-radius:3px;padding:1px 5px;font-size:10px;font-weight:bold;">⚠ below breakeven</span>'
          : "");
  } else {
    wr.textContent = "—";
  }
  wr.style.color = belowBreakeven ? "#ff4455" : "#dde1e7";
  document.getElementById("ta-be").textContent = r.breakeven_win_rate_pct != null ? fmt(r.breakeven_win_rate_pct, 1) + "%" : "—";
  const real = document.getElementById("ta-real");
  real.textContent = r.realized_pl_usd != null ? _sgn(r.realized_pl_usd) + "$" + fmt(Math.abs(r.realized_pl_usd)) : "—";
  real.style.color = _plColor(r.realized_pl_usd);
  document.getElementById("ta-n").textContent =
    r.n_round_trips + " (" + r.n_wins + "W/" + r.n_losses + "L" + (r.n_washes ? "/" + r.n_washes + "≈" : "") + ")";
  const aw = document.getElementById("ta-avgw");
  // (B3) winner denominator
  if (r.avg_winner_usd != null) {
    aw.innerHTML = "+$" + fmt(r.avg_winner_usd) + " <span class=\"muted\" style=\"font-size:11px;\">(n=" + (r.n_wins != null ? r.n_wins : "?") + ")</span>";
  } else { aw.textContent = "—"; }
  aw.style.color = "#4caf50";
  const al = document.getElementById("ta-avgl");
  // B3: show the loser denominator next to avg loser.
  if (r.avg_loser_usd != null) {
    al.innerHTML = "-$" + fmt(Math.abs(r.avg_loser_usd)) + " <span class=\"muted\" style=\"font-size:11px;\">(n=" + (r.n_losses != null ? r.n_losses : "?") + ")</span>";
  } else { al.textContent = "—"; }
  al.style.color = "#ff4455";
  document.getElementById("ta-hold").textContent =
    (r.avg_winner_hold_days != null ? fmt(r.avg_winner_hold_days, 2) + "d" : "—") + " / " +
    (r.avg_loser_hold_days != null ? fmt(r.avg_loser_hold_days, 2) + "d" : "—");
  const dg = document.getElementById("ta-disp");
  dg.textContent = r.disposition_gap_days != null ? _sgn(r.disposition_gap_days) + fmt(r.disposition_gap_days, 2) + "d" : "—";
  // Negative gap = winners cut faster than losers = the disposition effect.
  dg.style.color = (r.disposition_gap_days != null && r.disposition_gap_days < 0) ? "#ff4455" : _plColor(r.disposition_gap_days);
}

// ───────── Loser autopsy + concentration honesty (new, agent 4) ─────────
// Same /api/build-info `stale` degrade contract as the behavioural cluster:
// a process that booted before these endpoints' commit 404s them → explicit
// "restart to apply" instead of a silent console error. Table bodies are
// built with DOM nodes + textContent (never innerHTML) so a verbatim
// entry-reason string can't inject markup.
const _LA_MODE_COLOR = {
  KNIFE_CATCH: ["#b71c1c", "#ffffff"],
  SLOW_BLEED:  ["#b8860b", "#000000"],
  STOPPED_OUT: ["#1f2126", "#dde1e7"],
  WHIPSAW:     ["#1f3a5f", "#9ec5ff"],
};
function _cell(text, cls) {
  const td = document.createElement("td");
  if (cls) td.className = cls;
  td.textContent = (text == null ? "—" : String(text));
  return td;
}
async function refreshLoserAutopsy() {
  const r = await fetchMaybeStale("/api/loser-autopsy");
  if (r.__unavailable) { markStale("lautopsy-state", "lautopsy-headline", "Loser-autopsy endpoint"); return; }
  if (r.error) { document.getElementById("lautopsy-headline").textContent = "error: " + r.error; return; }
  const sEl = document.getElementById("lautopsy-state");
  if (r.state === "STABLE" && r.verdict) {
    const [bg, fg] = _LA_MODE_COLOR[r.verdict] || _LA_MODE_COLOR.STOPPED_OUT;
    sEl.textContent = r.verdict.replace(/_/g, " ");
    sEl.style.background = bg; sEl.style.color = fg;
  } else {
    const sb = { EMERGING: ["#3a2a00", "#ffd479"], NO_DATA: ["#1f2126", "#8b929d"], NO_LOSSES: ["#1b5e20", "#a5d6a7"] };
    const [bg, fg] = sb[r.state] || sb.NO_DATA;
    sEl.textContent = r.state || "—";
    sEl.style.background = bg; sEl.style.color = fg;
  }
  document.getElementById("lautopsy-headline").textContent = r.headline || "";
  const tot = document.getElementById("lautopsy-total");
  tot.textContent = r.total_loss_usd != null ? _sgn(r.total_loss_usd) + "$" + fmt(Math.abs(r.total_loss_usd)) : "—";
  tot.style.color = _plColor(r.total_loss_usd);
  document.getElementById("lautopsy-n").textContent = r.n_losers != null ? (r.n_losers + " / " + r.n_round_trips + " RT") : "—";
  const avg = document.getElementById("lautopsy-avg");
  avg.textContent = r.avg_loss_usd != null ? _sgn(r.avg_loss_usd) + "$" + fmt(Math.abs(r.avg_loss_usd)) : "—";
  avg.style.color = _plColor(r.avg_loss_usd);
  document.getElementById("lautopsy-hold").textContent = r.median_loser_hold_days != null ? fmt(r.median_loser_hold_days, 2) + "d" : "—";
  document.getElementById("lautopsy-mode").textContent = r.dominant_failure_mode ? r.dominant_failure_mode.replace(/_/g, " ") : "—";
  const tb = document.querySelector("#lautopsy-tbl tbody");
  tb.replaceChildren();
  const cards = r.worst_losers || [];
  if (!cards.length) {
    const tr = document.createElement("tr");
    const td = _cell(r.state === "NO_LOSSES" ? "no losing round-trips" : "no data", "muted");
    td.colSpan = 6; tr.appendChild(td); tb.appendChild(tr);
  } else {
    for (const c of cards) {
      const tr = document.createElement("tr");
      tr.appendChild(_cell(c.ticker));
      const pl = _cell((c.pnl_usd >= 0 ? "+" : "") + "$" + fmt(Math.abs(c.pnl_usd)), "num");
      pl.style.color = _plColor(c.pnl_usd);
      tr.appendChild(pl);
      tr.appendChild(_cell(c.pnl_pct != null ? (c.pnl_pct >= 0 ? "+" : "") + fmt(c.pnl_pct, 1) + "%" : "—", "num"));
      tr.appendChild(_cell(c.hold_days != null ? fmt(c.hold_days, 2) : "—", "num"));
      tr.appendChild(_cell(c.failure_mode ? c.failure_mode.replace(/_/g, " ") : "—"));
      tr.appendChild(_cell(c.entry_reason || "—"));
      tb.appendChild(tr);
    }
  }
}

const _WA_MODE_COLOR = {
  HOME_RUN:   ["#1b5e20", "#ffffff"],
  SLOW_GRIND: ["#2e7d32", "#ffffff"],
  TARGET_HIT: ["#1f2126", "#dde1e7"],
  SCALP:      ["#3a2a00", "#ffd479"],
};
async function refreshWinnerAutopsy() {
  const r = await fetchMaybeStale("/api/winner-autopsy");
  if (r.__unavailable) { markStale("wautopsy-state", "wautopsy-headline", "Winner-autopsy endpoint"); return; }
  if (r.error) { document.getElementById("wautopsy-headline").textContent = "error: " + r.error; return; }
  const sEl = document.getElementById("wautopsy-state");
  if (r.state === "STABLE" && r.verdict) {
    const [bg, fg] = _WA_MODE_COLOR[r.verdict] || _WA_MODE_COLOR.TARGET_HIT;
    sEl.textContent = r.verdict.replace(/_/g, " ");
    sEl.style.background = bg; sEl.style.color = fg;
  } else {
    const sb = { EMERGING: ["#3a2a00", "#ffd479"], NO_DATA: ["#1f2126", "#8b929d"], NO_WINS: ["#b71c1c", "#ffffff"] };
    const [bg, fg] = sb[r.state] || sb.NO_DATA;
    sEl.textContent = r.state || "—";
    sEl.style.background = bg; sEl.style.color = fg;
  }
  document.getElementById("wautopsy-headline").textContent = r.headline || "";
  const tot = document.getElementById("wautopsy-total");
  tot.textContent = r.total_gain_usd != null ? _sgn(r.total_gain_usd) + "$" + fmt(Math.abs(r.total_gain_usd)) : "—";
  tot.style.color = _plColor(r.total_gain_usd);
  document.getElementById("wautopsy-n").textContent = r.n_winners != null ? (r.n_winners + " / " + r.n_round_trips + " RT") : "—";
  const avg = document.getElementById("wautopsy-avg");
  avg.textContent = r.avg_gain_usd != null ? _sgn(r.avg_gain_usd) + "$" + fmt(Math.abs(r.avg_gain_usd)) : "—";
  avg.style.color = _plColor(r.avg_gain_usd);
  document.getElementById("wautopsy-hold").textContent = r.median_winner_hold_days != null ? fmt(r.median_winner_hold_days, 2) + "d" : "—";
  document.getElementById("wautopsy-mode").textContent = r.dominant_success_mode ? r.dominant_success_mode.replace(/_/g, " ") : "—";
  const tb = document.querySelector("#wautopsy-tbl tbody");
  tb.replaceChildren();
  const cards = r.best_winners || [];
  if (!cards.length) {
    const tr = document.createElement("tr");
    const td = _cell(r.state === "NO_WINS" ? "no winning round-trips" : "no data", "muted");
    td.colSpan = 6; tr.appendChild(td); tb.appendChild(tr);
  } else {
    for (const c of cards) {
      const tr = document.createElement("tr");
      tr.appendChild(_cell(c.ticker));
      const pl = _cell((c.pnl_usd >= 0 ? "+" : "") + "$" + fmt(Math.abs(c.pnl_usd)), "num");
      pl.style.color = _plColor(c.pnl_usd);
      tr.appendChild(pl);
      tr.appendChild(_cell(c.pnl_pct != null ? (c.pnl_pct >= 0 ? "+" : "") + fmt(c.pnl_pct, 1) + "%" : "—", "num"));
      tr.appendChild(_cell(c.hold_days != null ? fmt(c.hold_days, 2) : "—", "num"));
      tr.appendChild(_cell(c.success_mode ? c.success_mode.replace(/_/g, " ") : "—"));
      tr.appendChild(_cell(c.entry_reason || "—"));
      tb.appendChild(tr);
    }
  }
}

async function refreshTrackRecord() {
  const r = await fetchMaybeStale("/api/track-record");
  if (r.__unavailable) { markStale("trec-state", "trec-headline", "Track-record endpoint"); return; }
  if (r.error) { document.getElementById("trec-headline").textContent = "error: " + r.error; return; }
  const sEl = document.getElementById("trec-state");
  const sb = { OK: ["#1f3a5f", "#9ec5ff"], NO_DATA: ["#1f2126", "#8b929d"] };
  const [bg, fg] = sb[r.state] || sb.NO_DATA;
  sEl.textContent = r.state || "—";
  sEl.style.background = bg; sEl.style.color = fg;
  document.getElementById("trec-headline").textContent = r.summary || "";
  const names = r.names || [];
  document.getElementById("trec-n").textContent = names.length ? String(names.length) : "—";
  document.getElementById("trec-rt").textContent = r.n_round_trips != null ? String(r.n_round_trips) : "—";
  // names is sorted worst-net-first → first = worst, last = best.
  const worst = names.length ? names[0] : null;
  const best = names.length ? names[names.length - 1] : null;
  const wEl = document.getElementById("trec-worst");
  if (worst) { wEl.textContent = worst.ticker + " " + _sgn(worst.net_usd) + "$" + fmt(Math.abs(worst.net_usd)); wEl.style.color = _plColor(worst.net_usd); } else { wEl.textContent = "—"; }
  const bEl = document.getElementById("trec-best");
  if (best) { bEl.textContent = best.ticker + " " + _sgn(best.net_usd) + "$" + fmt(Math.abs(best.net_usd)); bEl.style.color = _plColor(best.net_usd); } else { bEl.textContent = "—"; }
  const tb = document.querySelector("#trec-tbl tbody");
  tb.replaceChildren();
  if (!names.length) {
    const tr = document.createElement("tr");
    const td = _cell("no closed round-trips yet", "muted");
    td.colSpan = 6; tr.appendChild(td); tb.appendChild(tr);
  } else {
    for (const e of names) {
      const tr = document.createElement("tr");
      tr.appendChild(_cell(e.ticker));
      tr.appendChild(_cell(e.n_win + "-" + e.n_loss, "num"));
      const pl = _cell(_sgn(e.net_usd) + "$" + fmt(Math.abs(e.net_usd)), "num");
      pl.style.color = _plColor(e.net_usd);
      tr.appendChild(pl);
      tr.appendChild(_cell(String(e.n_closed), "num"));
      const last = (e.recent && e.recent.length) ? e.recent[0] : null;
      tr.appendChild(_cell(last && last.mode ? last.mode.replace(/_/g, " ") : "—"));
      tr.appendChild(_cell(last && last.entry_reason ? last.entry_reason : "—"));
      tb.appendChild(tr);
    }
  }
}

async function refreshCorrelation() {
  const r = await fetchMaybeStale("/api/correlation");
  if (r.__unavailable) { markStale("pcorr-state", "pcorr-headline", "Concentration-honesty endpoint"); return; }
  if (r.error) { document.getElementById("pcorr-headline").textContent = "error: " + r.error; return; }
  const vmap = {
    SINGLE_NAME_RISK: ["#b71c1c", "#ffffff"],
    CONCENTRATED:     ["#b8860b", "#000000"],
    MODERATE:         ["#1f3a5f", "#9ec5ff"],
    DIVERSIFIED:      ["#1b5e20", "#a5d6a7"],
  };
  const sEl = document.getElementById("pcorr-state");
  if (r.state === "OK" && r.verdict) {
    const [bg, fg] = vmap[r.verdict] || vmap.MODERATE;
    sEl.textContent = r.verdict.replace(/_/g, " ");
    sEl.style.background = bg; sEl.style.color = fg;
  } else {
    const sb = { INSUFFICIENT: ["#3a2a00", "#ffd479"], NO_DATA: ["#1f2126", "#8b929d"] };
    const [bg, fg] = sb[r.state] || sb.NO_DATA;
    sEl.textContent = r.state || "—";
    sEl.style.background = bg; sEl.style.color = fg;
  }
  document.getElementById("pcorr-headline").textContent = r.headline || "";
  const mr = document.getElementById("pcorr-meanrho");
  mr.textContent = r.mean_pairwise_corr != null ? (r.mean_pairwise_corr >= 0 ? "+" : "") + fmt(r.mean_pairwise_corr, 2) : "—";
  // High co-movement is the risk → red as ρ climbs.
  mr.style.color = (r.mean_pairwise_corr != null && r.mean_pairwise_corr >= 0.7) ? "#ff4455"
                 : (r.mean_pairwise_corr != null && r.mean_pairwise_corr >= 0.4) ? "#ffd479" : "#dde1e7";
  document.getElementById("pcorr-effbets").textContent = r.effective_independent_bets != null ? fmt(r.effective_independent_bets, 2) : "—";
  document.getElementById("pcorr-effnaive").textContent = r.effective_positions_naive != null ? fmt(r.effective_positions_naive, 2) : "—";
  document.getElementById("pcorr-topw").textContent = r.top_weight_pct != null ? fmt(r.top_weight_pct, 1) + "% " + (r.top_weight_ticker || "") : "—";
  document.getElementById("pcorr-maxpair").textContent = (r.max_pair && r.max_pair.tickers)
    ? r.max_pair.tickers.join("/") + " " + (r.max_pair.corr >= 0 ? "+" : "") + fmt(r.max_pair.corr, 2) : "—";
  const tb = document.querySelector("#pcorr-tbl tbody");
  tb.replaceChildren();
  const pairs = (r.pairs || []).slice().sort((a, b) => (b.corr ?? -2) - (a.corr ?? -2));
  if (!pairs.length) {
    const tr = document.createElement("tr");
    const td = _cell(r.state === "NO_DATA" ? "no stock positions" : "not enough overlapping history", "muted");
    td.colSpan = 2; tr.appendChild(td); tb.appendChild(tr);
  } else {
    for (const p of pairs) {
      const tr = document.createElement("tr");
      tr.appendChild(_cell(p.a + " / " + p.b));
      const c = _cell(p.corr != null ? (p.corr >= 0 ? "+" : "") + fmt(p.corr, 2) : "n/a", "num");
      c.style.color = (p.corr != null && p.corr >= 0.7) ? "#ff4455" : (p.corr != null && p.corr >= 0.4) ? "#ffd479" : "#dde1e7";
      tr.appendChild(c);
      tb.appendChild(tr);
    }
  }
}

// ───────── Overtrading/churn + thesis-drift (new 2026-05-16, agent 4) ─────────
// Same stale-degrade contract as the behavioural cluster above: a process
// that booted before these endpoints' commit 404s them → explicit
// "restart to apply" instead of a silent failure.
async function refreshChurn() {
  const r = await fetchMaybeStale("/api/churn");
  if (r.__unavailable) { markStale("churn-state", "churn-headline", "Overtrading/churn endpoint"); return; }
  if (r.error) { document.getElementById("churn-headline").textContent = "error: " + r.error; return; }
  const vmap = {
    CHURNING:        ["#b71c1c", "#ffffff"],
    ACTIVE_TURNOVER: ["#b8860b", "#000000"],
    BUY_AND_HOLD:    ["#1b5e20", "#a5d6a7"],
  };
  const stateBadge = { STABLE: null, EMERGING: ["#3a2a00", "#ffd479"], NO_DATA: ["#1f2126", "#8b929d"] };
  const sEl = document.getElementById("churn-state");
  if (r.state === "STABLE" && r.verdict) {
    const [bg, fg] = vmap[r.verdict] || stateBadge.NO_DATA;
    sEl.textContent = r.verdict.replace(/_/g, " ");
    sEl.style.background = bg; sEl.style.color = fg;
  } else {
    const [bg, fg] = stateBadge[r.state] || stateBadge.NO_DATA;
    sEl.textContent = r.state; sEl.style.background = bg; sEl.style.color = fg;
  }
  document.getElementById("churn-headline").textContent = r.headline || "";
  const re = document.getElementById("churn-reentry");
  re.textContent = r.reentry_rate_pct != null ? r.n_reentries + " (" + fmt(r.reentry_rate_pct, 1) + "%)" : "—";
  re.style.color = (r.reentry_rate_pct != null && r.reentry_rate_pct >= 25) ? "#ff4455" : "#dde1e7";
  document.getElementById("churn-rtpd").textContent = r.round_trips_per_day != null ? fmt(r.round_trips_per_day, 2) : "—";
  document.getElementById("churn-hold").textContent = r.median_hold_days != null ? fmt(r.median_hold_days, 2) + "d" : "—";
  document.getElementById("churn-subday").textContent = r.sub_day_trip_pct != null ? fmt(r.sub_day_trip_pct, 1) + "%" : "—";
  const lc = document.getElementById("churn-lossconc");
  lc.textContent = r.churn_loss_concentration_pct != null ? fmt(r.churn_loss_concentration_pct, 1) + "%" : "—";
  lc.style.color = (r.churn_loss_concentration_pct != null && r.churn_loss_concentration_pct >= 50) ? "#ff4455" : "#dde1e7";
  const tb = document.querySelector("#churn-events tbody");
  const evs = r.reentry_events || [];
  if (!evs.length) {
    tb.innerHTML = '<tr><td colspan="4" class="muted">no fast same-name re-entries — clean turnover</td></tr>';
  } else {
    tb.innerHTML = evs.map(e => {
      const p = e.prior_pnl_usd;
      const pc = p == null ? "#8b929d" : (p > 0 ? "#4caf50" : p < 0 ? "#ff4455" : "#dde1e7");
      return '<tr><td>' + e.ticker + '</td><td class="num">' + fmt(e.gap_days, 2) +
        '</td><td class="num" style="color:' + pc + '">' +
        (p == null ? "—" : _sgn(p) + "$" + fmt(Math.abs(p))) +
        '</td><td>' + (e.prior_exit_ts || "").slice(0, 10) + ' → ' +
        (e.next_entry_ts || "").slice(0, 10) + '</td></tr>';
    }).join("");
  }
}

let _sessWindow = 360;
function setSessWindow(m) {
  _sessWindow = m;
  [60, 360, 1440].forEach(k => {
    const el = document.getElementById("sess-w-" + k);
    if (el) el.classList.toggle("active", k === m);
  });
  refreshSessionDelta();
}
async function refreshSessionDelta() {
  const r = await fetchMaybeStale("/api/session-delta?minutes=" + _sessWindow);
  if (r.__unavailable) { markStale("sess-state", "sess-headline", "Session-delta endpoint"); return; }
  if (r.error) { document.getElementById("sess-headline").textContent = "error: " + r.error; return; }
  const smap = {
    ACTIVE:  ["#0d3b4f", "#7fdbff"],
    QUIET:   ["#1b5e20", "#a5d6a7"],
    NO_DATA: ["#1f2126", "#8b929d"],
  };
  const sEl = document.getElementById("sess-state");
  const [bg, fg] = smap[r.state] || smap.NO_DATA;
  sEl.textContent = r.state + (r.n_events ? " · " + r.n_events : "");
  sEl.style.background = bg; sEl.style.color = fg;
  document.getElementById("sess-headline").textContent = r.headline || "";
  const tb = document.querySelector("#sess-events tbody");
  const evs = r.events || [];
  if (!evs.length) {
    tb.innerHTML = '<tr><td colspan="3" class="muted">nothing material in this window</td></tr>';
    return;
  }
  const sevC = { HIGH: "#ff4455", MED: "#ffb74d", LOW: "#8b929d" };
  const kindLabel = {
    TRADE: "⇄ FILL", POSITION_CLOSED: "✓ CLOSED", EQUITY_MOVE: "$ EQUITY",
    DRAWDOWN_LOW: "▼ DRAWDOWN", INACTION: "… IDLE",
  };
  tb.innerHTML = evs.map(e => {
    const c = sevC[e.severity] || "#8b929d";
    const when = fmtTs(e.ts, "time");
    const lbl = kindLabel[e.kind] || e.kind;
    const txt = (e.summary || "").replace(/</g, "&lt;");
    return '<tr><td class="muted">' + when + '</td><td style="color:' + c +
      ';white-space:nowrap;">' + lbl + '</td><td>' + txt + '</td></tr>';
  }).join("");
}

async function refreshThesisDrift() {
  const r = await fetchMaybeStale("/api/thesis-drift");
  if (r.__unavailable) { markStale("tdrift-state", "tdrift-headline", "Thesis-drift endpoint"); return; }
  if (r.error) { document.getElementById("tdrift-headline").textContent = "error: " + r.error; return; }
  const sEl = document.getElementById("tdrift-state");
  const c = r.counts || {};
  if (r.state === "NO_DATA") {
    sEl.textContent = "NO DATA"; sEl.style.background = "#1f2126"; sEl.style.color = "#8b929d";
  } else if ((c.BROKEN || 0) > 0) {
    sEl.textContent = c.BROKEN + " BROKEN"; sEl.style.background = "#b71c1c"; sEl.style.color = "#fff";
  } else if ((c.WEAKENING || 0) > 0) {
    sEl.textContent = c.WEAKENING + " WEAKENING"; sEl.style.background = "#b8860b"; sEl.style.color = "#000";
  } else {
    sEl.textContent = "ALL INTACT"; sEl.style.background = "#1b5e20"; sEl.style.color = "#a5d6a7";
  }
  document.getElementById("tdrift-headline").textContent = r.headline || "";
  const tb = document.querySelector("#tdrift-rows tbody");
  const ps = r.positions || [];
  if (!ps.length) {
    tb.innerHTML = '<tr><td colspan="5" class="muted">no open positions</td></tr>';
    return;
  }
  const hmap = { BROKEN: ["#b71c1c", "#fff"], WEAKENING: ["#b8860b", "#000"], INTACT: ["#1b5e20", "#a5d6a7"] };
  tb.innerHTML = ps.map(p => {
    const [hb, hf] = hmap[p.health] || ["#1f2126", "#8b929d"];
    const reason = p.entry_reason || "—";
    const reasonShort = reason.length > 90 ? reason.slice(0, 90) + "…" : reason;
    const drift = (p.drift_reasons || []).join("; ");
    const plc = p.pl_pct == null ? "#8b929d" : (p.pl_pct > 0 ? "#4caf50" : p.pl_pct < 0 ? "#ff4455" : "#dde1e7");
    return '<tr><td>' + p.ticker + '</td>' +
      '<td><span style="padding:2px 7px;border-radius:3px;font-size:11px;background:' + hb + ';color:' + hf + '">' + p.health + '</span></td>' +
      '<td class="num" style="color:' + plc + '">' + (p.pl_pct == null ? "—" : _sgn(p.pl_pct) + fmt(p.pl_pct, 2) + "%") + '</td>' +
      '<td class="num">' + (p.days_held == null ? "—" : fmt(p.days_held, 1)) + '</td>' +
      '<td title="' + reason.replace(/"/g, "&quot;") + '"><span class="muted">' + reasonShort + '</span><br><span style="color:#dde1e7;">↳ ' + (drift || "—") + '</span></td></tr>';
  }).join("");
}

async function refreshGlobalStale() {
  try {
    const r = await fetch(API_PREFIX + "/api/build-info").then(r => r.json());
    const el = document.getElementById("global-stale-banner");
    const tx = document.getElementById("global-stale-text");
    if (r && (r.stale || (r.behind && r.behind > 0))) {
      tx.textContent = "⚠ Paper-trader is running stale code — booted " +
        (r.boot_sha || "?") + ", HEAD is " + (r.head_sha || "?") +
        (r.behind ? " (" + r.behind + " commit" + (r.behind === 1 ? "" : "s") + " behind)" : "") +
        ". Committed fixes (incl. the self-review mirror & newest endpoints) are NOT applied until paper-trader is restarted.";
      el.style.display = "block";
    } else {
      el.style.display = "none";
    }
  } catch (e) { /* build-info unreachable — leave banner hidden */ }
}

async function refreshCapitalParalysis() {
  const r = await fetchMaybeStale("/api/capital-paralysis");
  if (r.__unavailable) { markStale("cp-state", "cp-headline", "Capital-paralysis endpoint"); return; }
  if (r.error) { document.getElementById("cp-headline").textContent = "error: " + r.error; return; }
  const smap = {
    PINNED:  ["#b71c1c", "#ffffff"],
    EMPTY:   ["#b71c1c", "#ffffff"],
    FREE:    ["#1b5e20", "#a5d6a7"],
    NO_DATA: ["#1f2126", "#8b929d"],
  };
  const [bg, fg] = smap[r.state] || smap.NO_DATA;
  const sEl = document.getElementById("cp-state");
  sEl.textContent = r.state || "—"; sEl.style.background = bg; sEl.style.color = fg;
  document.getElementById("cp-headline").textContent = r.headline || "";
  // B1: prominent banner at the cash extremes — over-invested (<5% cash,
  // can't act on any new signal) or under-deployed (>80% idle cash,
  // dragging vs the benchmark).
  const cpBanner = document.getElementById("cp-banner");
  if (cpBanner) {
    const cpct = r.cash_pct;
    if (cpct != null && cpct < 5) {
      cpBanner.style.display = "";
      cpBanner.style.background = "rgba(255,68,85,0.14)";
      cpBanner.style.color = "#ff4455";
      cpBanner.style.borderColor = "rgba(255,68,85,0.4)";
      cpBanner.textContent = `⚠ OVER-INVESTED: only ${fmt(cpct, 1)}% cash remaining — cannot act on a new signal without selling first`;
    } else if (cpct != null && cpct > 80) {
      cpBanner.style.display = "";
      cpBanner.style.background = "rgba(255,145,0,0.14)";
      cpBanner.style.color = "#ff9100";
      cpBanner.style.borderColor = "rgba(255,145,0,0.4)";
      cpBanner.textContent = `⚠ UNDER-DEPLOYED: ${fmt(cpct, 1)}% idle cash — capital sitting out of the market`;
    } else {
      cpBanner.style.display = "none";
    }
  }
  document.getElementById("cp-cash").textContent =
    r.cash != null ? "$" + fmt(r.cash) + " (" + fmt(r.cash_pct, 1) + "%)" : "—";
  document.getElementById("cp-dep").textContent = r.deployed_pct != null ? fmt(r.deployed_pct, 1) + "%" : "—";
  const ca = document.getElementById("cp-canact");
  ca.textContent = r.can_act_on_signal ? "yes" : "no";
  ca.style.color = r.can_act_on_signal ? "#4caf50" : "#ff4455";
  document.getElementById("cp-stuck").textContent = r.cycles_since_last_fill != null ? r.cycles_since_last_fill : "—";
  const bleed = r.paralysis ? r.paralysis.involuntary_alpha_bleed_pct : null;
  const bEl = document.getElementById("cp-bleed");
  bEl.textContent = bleed != null ? fmt(bleed, 2) + "%" : "—";
  bEl.style.color = (bleed || 0) <= -1.0 ? "#ff4455" : (bleed || 0) < 0 ? "#ffa726" : "#4caf50";
  const lad = r.unlock_ladder || [];
  const recT = r.recommended_unlock ? r.recommended_unlock.ticker : null;
  const tb = document.querySelector("#cp-ladder tbody");
  tb.innerHTML = lad.length ? lad.map(p => {
    const rec = p.ticker === recT;
    const plc = _plColor(p.pl_pct);
    return `<tr${rec ? ' style="background:#15240f;"' : ''}>
      <td>${rec ? '★ ' : ''}${p.ticker}<span class="muted">${p.type && p.type !== 'stock' ? ' ' + p.type : ''}</span></td>
      <td class="num">${fmt(p.weight_pct, 1)}</td>
      <td class="num" style="color:${plc};">${_sgn(p.pl_pct)}${fmt(p.pl_pct, 1)}</td>
      <td class="num">$${fmt(p.frees_usd)}</td>
      <td class="num">$${fmt(p.cash_if_sold_alone)}</td>
      <td style="color:${p.restores_action_alone ? '#4caf50' : '#8b929d'};font-weight:${p.restores_action_alone ? 'bold' : 'normal'};">${p.restores_action_alone ? 'unlocks ✓' : '—'}</td>
    </tr>`;
  }).join("") : `<tr><td colspan="6" class="muted">no open positions to ladder</td></tr>`;
}

async function refreshOpenAttribution() {
  const r = await fetchMaybeStale("/api/open-attribution");
  if (r.__unavailable) { markStale("oa-status", "oa-headline", "Open-attribution endpoint"); return; }
  if (r.error) { document.getElementById("oa-headline").textContent = "error: " + r.error; return; }
  const smap = {
    SELECTION_ADDING: ["#1b5e20", "#a5d6a7"],
    SELECTION_DRAG:   ["#b71c1c", "#ffffff"],
    FLAT_VS_SPY:      ["#1f2126", "#8b929d"],
    NO_BENCHMARK:     ["#3a2a00", "#ffd479"],
    NO_DATA:          ["#1f2126", "#8b929d"],
  };
  const [bg, fg] = smap[r.status] || smap.NO_DATA;
  const sEl = document.getElementById("oa-status");
  sEl.textContent = (r.status || "—").replace(/_/g, " ");
  sEl.style.background = bg; sEl.style.color = fg;
  document.getElementById("oa-headline").textContent = r.headline || "";
  const al = document.getElementById("oa-alpha");
  al.textContent = r.book_open_alpha_pct != null ? _sgn(r.book_open_alpha_pct) + fmt(r.book_open_alpha_pct, 2) + "%" : "—";
  al.style.color = _plColor(r.book_open_alpha_pct);
  const ex = document.getElementById("oa-excess");
  ex.textContent = r.net_excess_usd != null ? _sgn(r.net_excess_usd) + "$" + fmt(Math.abs(r.net_excess_usd)) : "—";
  ex.style.color = _plColor(r.net_excess_usd);
  const ur = document.getElementById("oa-unreal");
  ur.textContent = r.total_unrealized_usd != null ? _sgn(r.total_unrealized_usd) + "$" + fmt(Math.abs(r.total_unrealized_usd)) : "—";
  ur.style.color = _plColor(r.total_unrealized_usd);
  document.getElementById("oa-spyeq").textContent =
    r.total_spy_equivalent_usd != null ? _sgn(r.total_spy_equivalent_usd) + "$" + fmt(Math.abs(r.total_spy_equivalent_usd)) : "—";
  document.getElementById("oa-n").textContent =
    (r.n_anchored != null ? r.n_anchored : "—") + (r.n_positions != null ? " / " + r.n_positions : "");
  const rows = r.positions || [];
  const tb = document.querySelector("#oa-rows tbody");
  tb.innerHTML = rows.length ? rows.map(p => {
    if (!p.anchored) {
      return `<tr><td>${p.ticker}</td><td class="num">${fmt(p.position_return_pct, 2)}</td>
        <td class="num muted" colspan="3">unanchored — no SPY level at/after entry</td></tr>`;
    }
    return `<tr>
      <td>${p.ticker}</td>
      <td class="num" style="color:${_plColor(p.position_return_pct)};">${_sgn(p.position_return_pct)}${fmt(p.position_return_pct, 2)}</td>
      <td class="num" style="color:${_plColor(p.spy_return_pct)};">${_sgn(p.spy_return_pct)}${fmt(p.spy_return_pct, 2)}</td>
      <td class="num" style="color:${_plColor(p.alpha_pct)};font-weight:bold;">${_sgn(p.alpha_pct)}${fmt(p.alpha_pct, 2)}</td>
      <td class="num" style="color:${_plColor(p.excess_usd)};">${_sgn(p.excess_usd)}$${fmt(Math.abs(p.excess_usd))}</td>
    </tr>`;
  }).join("") : `<tr><td colspan="5" class="muted">no anchorable open stock positions</td></tr>`;
}

async function refreshFeedHealth() {
  const r = await fetchMaybeStale("/api/feed-health");
  if (r.__unavailable) { markStale("fh-state", "fh-headline", "Signal-feed-health endpoint"); return; }
  if (r.error) { document.getElementById("fh-headline").textContent = "error: " + r.error; return; }
  const smap = {
    BLIND:      ["#b71c1c", "#ffffff"],
    STALE_FEED: ["#b8860b", "#000000"],
    HEALTHY:    ["#1b5e20", "#a5d6a7"],
    NO_DATA:    ["#1f2126", "#8b929d"],
    ERROR:      ["#3a2a00", "#ffd479"],
  };
  const [bg, fg] = smap[r.verdict] || smap.NO_DATA;
  const sEl = document.getElementById("fh-state");
  sEl.textContent = (r.verdict || "—").replace(/_/g, " ");
  sEl.style.background = bg; sEl.style.color = fg;
  document.getElementById("fh-headline").textContent =
    (r.restart_recommended ? "⚠ RESTART RECOMMENDED — " : "") + (r.headline || "");
  const st = document.getElementById("fh-streak");
  st.textContent = r.blind_streak != null
    ? r.blind_streak + " / " + (r.n_decisions != null ? r.n_decisions : "—")
    : "—";
  st.style.color = (r.blind_streak || 0) >= (r.blind_streak_min || 3) ? "#ff4455"
                  : (r.blind_streak || 0) > 0 ? "#ffa726" : "#4caf50";
  const ag = document.getElementById("fh-age");
  ag.textContent = r.resolved_newest_age_h != null
    ? fmt(r.resolved_newest_age_h, 1) + "h" : "never";
  ag.style.color = (r.resolved_newest_age_h == null
                    || r.resolved_newest_age_h >= (r.stale_hours || 6))
                   ? "#ff4455" : "#4caf50";
  document.getElementById("fh-live").textContent =
    (r.resolved_live_2h != null ? r.resolved_live_2h : "—") + " / "
    + (r.resolved_live_24h != null ? r.resolved_live_24h : "—");
  const sp = document.getElementById("fh-split");
  sp.textContent = r.split_brain ? "YES" : "no";
  sp.style.color = r.split_brain ? "#ff4455" : "#8b929d";
  document.getElementById("fh-path").textContent =
    r.resolved_path
      ? ("trader reads " + r.resolved_path
         + (r.split_brain && r.fresher_path
            ? "  ·  fresher copy: " + r.fresher_path
              + " (" + fmt(r.fresher_age_h, 1) + "h)"
            : ""))
      : "no resolved article DB";
}

async function refreshRunnerHeartbeat() {
  const r = await fetchMaybeStale("/api/runner-heartbeat");
  if (r.__unavailable) { markStale("rhb-state", "rhb-headline", "Runner-heartbeat endpoint"); return; }
  if (r.error) { document.getElementById("rhb-headline").textContent = "error: " + r.error; return; }
  const smap = {
    STALLED: ["#b71c1c", "#ffffff"],
    LAGGING: ["#b8860b", "#000000"],
    HEALTHY: ["#1b5e20", "#a5d6a7"],
    NO_DATA: ["#1f2126", "#8b929d"],
  };
  const [bg, fg] = smap[r.verdict] || smap.NO_DATA;
  const sEl = document.getElementById("rhb-state");
  sEl.textContent = (r.verdict || "—").replace(/_/g, " ");
  sEl.style.background = bg; sEl.style.color = fg;
  document.getElementById("rhb-headline").textContent =
    (r.restart_recommended ? "⚠ RESTART RECOMMENDED — " : "") + (r.headline || "");
  const ag = document.getElementById("rhb-age");
  const secs = r.secs_since_last_decision;
  ag.textContent = secs == null ? "never"
    : secs < 90 ? Math.round(secs) + "s"
    : secs < 5400 ? Math.round(secs / 60) + "m"
    : fmt(secs / 3600, 1) + "h";
  ag.style.color = r.verdict === "STALLED" ? "#ff4455"
                 : r.verdict === "LAGGING" ? "#ffa726"
                 : r.verdict === "HEALTHY" ? "#4caf50" : "#8b929d";
  const iv = document.getElementById("rhb-intervals");
  iv.textContent = r.intervals_elapsed != null ? fmt(r.intervals_elapsed, 2) + "×" : "—";
  iv.style.color = (r.intervals_elapsed || 0) >= (r.stalled_mult || 2)  ? "#ff4455"
                 : (r.intervals_elapsed || 0) >= (r.lagging_mult || 1.25) ? "#ffa726"
                 : "#8b929d";
  document.getElementById("rhb-cadence").textContent =
    r.expected_interval_s != null ? Math.round(r.expected_interval_s / 60) + "m" : "—";
  document.getElementById("rhb-market").textContent = r.market_open ? "open" : "closed";
}

async function refreshDecisionReliability() {
  const r = await fetchMaybeStale("/api/decision-reliability");
  if (r.__unavailable) { markStale("dr-state", "dr-headline", "Decision-reliability endpoint"); return; }
  if (r.error) { document.getElementById("dr-headline").textContent = "error: " + r.error; return; }
  const smap = {
    CRITICAL:               ["#b71c1c", "#ffffff"],
    DEGRADED:               ["#b8860b", "#000000"],
    HEALTHY:                ["#1b5e20", "#a5d6a7"],
    STALE_LEGACY_DOMINATED: ["#3a2a00", "#ffd479"],
    INSUFFICIENT:           ["#1f2126", "#8b929d"],
    NO_DATA:                ["#1f2126", "#8b929d"],
  };
  const [bg, fg] = smap[r.state] || smap.NO_DATA;
  const sEl = document.getElementById("dr-state");
  sEl.textContent = (r.state || "—").replace(/_/g, " ");
  sEl.style.background = bg; sEl.style.color = fg;
  document.getElementById("dr-headline").textContent =
    (r.restart_recommended ? "⚠ RESTART RECOMMENDED — " : "") + (r.headline || "");
  const cur = document.getElementById("dr-cur");
  cur.textContent = r.current_failure_rate_pct != null
    ? fmt(r.current_failure_rate_pct, 1) + "% (" + (r.current_failures || 0) + "/" + (r.current_total || 0) + ")"
    : "—";
  cur.style.color = (r.current_failure_rate_pct >= 50) ? "#ff4455"
                    : (r.current_failure_rate_pct >= 25) ? "#ffa726" : "#4caf50";
  document.getElementById("dr-head").textContent =
    r.headline_failure_rate_pct != null ? fmt(r.headline_failure_rate_pct, 1) + "%" : "—";
  document.getElementById("dr-n").textContent =
    (r.current_total != null ? r.current_total : "—") + " / " + (r.n_decisions != null ? r.n_decisions : "—");
  const lg = document.getElementById("dr-legacy");
  lg.textContent = r.legacy_failures != null
    ? r.legacy_failures + " (" + fmt(r.legacy_share_pct, 1) + "%)" : "—";
  lg.style.color = (r.legacy_share_pct || 0) >= 50 ? "#ffa726" : "#8b929d";
  const dd = document.getElementById("dr-dead");
  dd.textContent = r.dead_cycles_per_day != null ? fmt(r.dead_cycles_per_day, 2) : "—";
  dd.style.color = (r.dead_cycles_per_day || 0) > 0 ? "#ff4455" : "#8b929d";
  const mm = (r.current_mode_mix || []).slice(0, 3)
    .map(m => m.mode.replace(/_/g, " ") + " " + m.n + " (" + fmt(m.pct, 0) + "%)").join(" · ");
  document.getElementById("dr-mode").textContent =
    mm ? "current failure modes: " + mm
       : (r.regime_boundary ? "regime boundary: " + r.regime_boundary
                            : "no current-regime failures recorded");
}

async function refreshBaselineCompare() {
  const r = await fetchMaybeStale("/api/baseline-compare");
  if (r.__unavailable) { markStale("bc-state", "bc-headline", "Baseline-compare endpoint"); return; }
  if (r.error && !r.verdict) { document.getElementById("bc-headline").textContent = "error: " + r.error; return; }
  const smap = {
    MLP_ADDS_SKILL:             ["#1b5e20", "#a5d6a7"],
    MLP_NO_BETTER_THAN_TRIVIAL: ["#b8860b", "#000000"],
    MLP_WORSE_THAN_TRIVIAL:     ["#b71c1c", "#ffffff"],
    INSUFFICIENT_DATA:          ["#1f2126", "#8b929d"],
  };
  const [bg, fg] = smap[r.verdict] || smap.INSUFFICIENT_DATA;
  const sEl = document.getElementById("bc-state");
  sEl.textContent = (r.verdict || "—").replace(/_/g, " ");
  sEl.style.background = bg; sEl.style.color = fg;
  document.getElementById("bc-headline").textContent = r.hint || "";
  const mlp = (r.mlp || {});
  const mEl = document.getElementById("bc-mlp");
  mEl.textContent = mlp.rank_ic != null ? fmt(mlp.rank_ic, 3) : "—";
  // green only once it clears its own real-skill floor (0.10, MLP_IC_MIN)
  mEl.style.color = mlp.rank_ic == null ? "#8b929d"
                    : (mlp.rank_ic >= 0.10) ? "#4caf50"
                    : (mlp.rank_ic > 0) ? "#ffa726" : "#ff4455";
  document.getElementById("bc-best").textContent =
    r.best_baseline ? r.best_baseline + " " + (r.best_baseline_ic != null ? fmt(r.best_baseline_ic, 3) : "—") : "—";
  const gEl = document.getElementById("bc-gap");
  gEl.textContent = r.ic_gap != null ? (r.ic_gap >= 0 ? "+" : "") + fmt(r.ic_gap, 3) : "—";
  gEl.style.color = r.ic_gap == null ? "#8b929d" : (r.ic_gap > 0 ? "#4caf50" : "#ff4455");
  document.getElementById("bc-n").textContent =
    (r.n != null ? r.n : "—") + " " + (r.slice ? "(" + r.slice + ")" : "") +
    " / " + (r.n_train != null ? r.n_train : "—");
}

async function refreshFundedSuggestions() {
  const r = await fetchMaybeStale("/api/funded-suggestions");
  if (r.__unavailable) { markStale("fund-state", "fund-headline", "Funded-suggestions endpoint"); return; }
  if (r.error) { document.getElementById("fund-headline").textContent = "error: " + r.error; return; }
  const smap = {
    FREE:    ["#1b5e20", "#a5d6a7"],
    PINNED:  ["#b71c1c", "#ffffff"],
    EMPTY:   ["#b71c1c", "#ffffff"],
    NO_DATA: ["#1f2126", "#8b929d"],
  };
  const [bg, fg] = smap[r.state] || smap.NO_DATA;
  const sEl = document.getElementById("fund-state");
  sEl.textContent = r.state || "—"; sEl.style.background = bg; sEl.style.color = fg;
  document.getElementById("fund-headline").textContent = r.headline || "";
  document.getElementById("fund-n").textContent = r.n_actionable != null ? r.n_actionable : "—";
  const fF = document.getElementById("fund-funded");
  fF.textContent = r.n_funded != null ? r.n_funded : "—";
  fF.style.color = (r.n_funded || 0) > 0 ? "#4caf50" : "#8b929d";
  const fP = document.getElementById("fund-partial");
  if (fP) {
    fP.textContent = r.n_partial != null ? r.n_partial : "—";
    fP.style.color = (r.n_partial || 0) > 0 ? "#ffd54f" : "#8b929d";
  }
  const fU = document.getElementById("fund-unlock");
  fU.textContent = r.n_unlockable != null ? r.n_unlockable : "—";
  fU.style.color = (r.n_unlockable || 0) > 0 ? "#ffa726" : "#8b929d";
  const fX = document.getElementById("fund-unfund");
  fX.textContent = r.n_unfundable != null ? r.n_unfundable : "—";
  fX.style.color = (r.n_unfundable || 0) > 0 ? "#ff4455" : "#8b929d";
  const pr = r.recommended_pairing;
  document.getElementById("fund-pair").textContent =
    pr ? ("sell " + pr.sell + " → buy " + pr.buy) : "—";
  const fmap = { FUNDED: "#4caf50", PARTIAL: "#ffd54f", UNLOCKABLE: "#ffa726", UNFUNDABLE: "#ff4455" };
  const rows = r.ideas || [];
  const tb = document.querySelector("#fund-rows tbody");
  tb.innerHTML = rows.length ? rows.map(i => `<tr>
      <td>${i.action} ${i.ticker}</td>
      <td class="num">${fmt(i.conviction, 2)}</td>
      <td class="num">$${fmt(i.suggested_notional_usd)}</td>
      <td style="color:${fmap[i.fundability] || '#8b929d'};font-weight:bold;">${i.fundability}${i.enough === false && i.fundability === 'UNFUNDABLE' ? '' : ''}</td>
      <td>${(i.funded_by && i.funded_by.length) ? i.funded_by.join(' + ') : '<span class="muted">—</span>'}</td>
      <td class="num">${i.frees_usd ? '$' + fmt(i.frees_usd) : '—'}</td>
    </tr>`).join("") : `<tr><td colspan="6" class="muted">no actionable BUY/ADD ideas</td></tr>`;
}

// ───────── Signal follow-through — does it use its own news edge? (new 2026-05-16, agent 4) ─────────
async function refreshSignalFollowThrough() {
  const r = await fetchMaybeStale("/api/signal-followthrough");
  if (r.__unavailable) { markStale("sft-state", "sft-headline", "Signal-follow-through endpoint"); return; }
  if (r.error) { document.getElementById("sft-headline").textContent = "error: " + r.error; return; }
  const smap = {
    EXPLOITING_SIGNALS: ["#1b5e20", "#a5d6a7"],
    NEUTRAL_USE:        ["#b8860b", "#000000"],
    LOW_ACTIVITY:       ["#3a2a00", "#ffd479"],
    IGNORING_FEED:      ["#b71c1c", "#ffffff"],
    MISUSING_SIGNALS:   ["#b71c1c", "#ffffff"],
    INSUFFICIENT:       ["#1f2126", "#8b929d"],
    NO_DATA:            ["#1f2126", "#8b929d"],
    ERROR:              ["#b71c1c", "#ffffff"],
  };
  const [bg, fg] = smap[r.verdict] || smap.NO_DATA;
  const sEl = document.getElementById("sft-state");
  sEl.textContent = (r.verdict || "—").replace(/_/g, " ");
  sEl.style.background = bg; sEl.style.color = fg;
  document.getElementById("sft-headline").textContent = r.verdict_reason || "";
  const ft = document.getElementById("sft-ft");
  ft.textContent = r.follow_through_rate_pct != null ? fmt(r.follow_through_rate_pct, 1) + "%" : "—";
  ft.style.color = r.follow_through_rate_pct == null ? "#8b929d"
                 : r.follow_through_rate_pct < 5 ? "#ff4455"
                 : r.follow_through_rate_pct < 25 ? "#ffa726" : "#4caf50";
  document.getElementById("sft-ai").textContent =
    (r.n_acted != null ? r.n_acted : "—") + " / " + (r.n_ignored != null ? r.n_ignored : "—");
  const ed = document.getElementById("sft-edge");
  ed.textContent = r.selection_edge_pct != null ? _sgn(r.selection_edge_pct) + fmt(r.selection_edge_pct, 2) + " pp" : "—";
  ed.style.color = _plColor(r.selection_edge_pct);
  const ref = String(r.reference_horizon || 3);
  const acted = (r.acted || {})[ref] || {}, ign = (r.ignored || {})[ref] || {};
  const aEl = document.getElementById("sft-acted");
  aEl.textContent = acted.mean_abnormal_pct != null ? _sgn(acted.mean_abnormal_pct) + fmt(acted.mean_abnormal_pct, 2) + "%" : "—";
  aEl.style.color = _plColor(acted.mean_abnormal_pct);
  const iEl = document.getElementById("sft-ign");
  iEl.textContent = ign.mean_abnormal_pct != null ? _sgn(ign.mean_abnormal_pct) + fmt(ign.mean_abnormal_pct, 2) + "%" : "—";
  iEl.style.color = _plColor(ign.mean_abnormal_pct);
  document.getElementById("sft-n").textContent =
    (r.n_resolved != null ? r.n_resolved : "—") + " / " + (r.n_signals != null ? r.n_signals : "—");
  document.getElementById("sft-meta").textContent =
    "ref " + (r.reference_horizon != null ? r.reference_horizon + "d" : "—")
    + " · " + (r.n_decisions != null ? r.n_decisions : "—") + " decisions"
    + " · " + (r.n_tickers_priced != null ? r.n_tickers_priced : "—") + " tickers priced"
    + (r.spy_adjusted ? " · SPY-adjusted" : " · raw only")
    + (r.lookback_days != null ? " · " + r.lookback_days + "d lookback" : "");
}

// ───────── News source edge — which collector is worth trusting? (new 2026-05-16, agent 4) ─────────
async function refreshSourceEdge() {
  const r = await fetchMaybeStale("/api/source-edge");
  if (r.__unavailable) { markStale("se-state", "se-headline", "Source-edge endpoint"); return; }
  if (r.error) { document.getElementById("se-headline").textContent = "error: " + r.error; return; }
  const smap = {
    EDGE_FOUND:        ["#1b5e20", "#a5d6a7"],
    NO_EDGE:           ["#b71c1c", "#ffffff"],
    INSUFFICIENT_DATA: ["#3a2a00", "#ffd479"],
    NO_DATA:           ["#1f2126", "#8b929d"],
    ERROR:             ["#b71c1c", "#ffffff"],
  };
  const [bg, fg] = smap[r.verdict] || smap.NO_DATA;
  const sEl = document.getElementById("se-state");
  sEl.textContent = (r.verdict || "—").replace(/_/g, " ");
  sEl.style.background = bg; sEl.style.color = fg;
  document.getElementById("se-headline").textContent = r.verdict_reason || "";
  const ref = String(r.reference_horizon || 3);
  const vmap = {
    EXPLOITABLE:  "#4caf50", WEAK: "#ffa726",
    NEGATIVE:     "#ff4455", INSUFFICIENT: "#8b929d",
  };
  const rows = (r.sources || []).slice(0, 10).map(s => {
    const h = (s.horizons || {})[ref] || {};
    const abn = h.mean_abnormal_pct;
    const hit = h.abnormal_hit_rate;
    return "<tr style='border-top:1px solid #1f2126;'>"
      + "<td style='padding:4px 6px;'>" + s.source + "</td>"
      + "<td style='padding:4px 6px;color:" + _plColor(abn) + ";'>"
        + (abn != null ? _sgn(abn) + fmt(abn, 2) + "%" : "—") + "</td>"
      + "<td style='padding:4px 6px;'>" + (hit != null ? fmt(hit, 0) + "%" : "—") + "</td>"
      + "<td style='padding:4px 6px;'>" + (s.n_resolved != null ? s.n_resolved : "—") + "</td>"
      + "<td style='padding:4px 6px;color:" + (vmap[s.verdict] || "#8b929d") + ";'>"
        + (s.verdict || "—") + "</td></tr>";
  });
  document.getElementById("se-rows").innerHTML =
    rows.length ? rows.join("") : "<tr><td colspan='5' class='muted' style='padding:6px;'>no collector resolved a watchlist move yet</td></tr>";
  document.getElementById("se-meta").textContent =
    "ref " + (r.reference_horizon != null ? r.reference_horizon + "d" : "—")
    + " · " + (r.n_resolved != null ? r.n_resolved : "—") + " resolved / "
    + (r.n_scored != null ? r.n_scored : "—") + " scored"
    + " · " + (r.n_tickers_priced != null ? r.n_tickers_priced : "—") + " tickers priced"
    + (r.spy_adjusted ? " · SPY-adjusted" : " · raw only")
    + (r.lookback_days != null ? " · " + r.lookback_days + "d lookback" : "");
}

// ───────── Behavioural scorecard — verdict-alignment router (new 2026-05-16, agent 4) ─────────
// Same stale-degrade contract as the behavioural cluster: a process that
// booted before this endpoint's commit 404s it → explicit "restart to apply".
async function refreshScorecard() {
  const r = await fetchMaybeStale("/api/scorecard");
  if (r.__unavailable) { markStale("score-state", "score-headline", "Behavioural scorecard endpoint"); return; }
  if (r.error) { document.getElementById("score-headline").textContent = "error: " + r.error; return; }
  const smap = {
    FLAGS_PRESENT:   ["#b71c1c", "#ffffff"],
    ALIGNED_HEALTHY: ["#1b5e20", "#a5d6a7"],
    NO_DATA:         ["#1f2126", "#8b929d"],
  };
  const [bg, fg] = smap[r.state] || smap.NO_DATA;
  const sEl = document.getElementById("score-state");
  sEl.textContent = (r.state || "—").replace(/_/g, " ");
  sEl.style.background = bg; sEl.style.color = fg;
  document.getElementById("score-headline").textContent = r.headline || "";

  const fEl = document.getElementById("score-focus");
  if (r.focus) {
    fEl.innerHTML = "<span style='color:#ffd479;'>Look first:</span> "
      + "<b>" + r.focus.name.replace(/_/g, " ") + "</b> — "
      + (r.focus.headline || "");
  } else { fEl.textContent = ""; }

  const cEl = document.getElementById("score-concordance");
  const conc = (r.concordance || []);
  if (conc.length) {
    cEl.innerHTML = conc.map(n =>
      "<span style='color:#ff8a80;'>" + n.count
      + " independent checks concur on " + n.theme.replace(/_/g, " ")
      + ":</span> " + (n.labels || []).join(", ")).join("<br>");
  } else { cEl.textContent = ""; }

  const kcolor = { FLAG: "#ff4455", OK: "#4caf50", IMMATURE: "#8b929d", ERROR: "#ffd479" };
  const rows = (r.checks || []).map(c =>
    "<tr><td style='padding:4px 6px;'>" + c.name.replace(/_/g, " ") + "</td>"
    + "<td style='padding:4px 6px;color:" + (kcolor[c.klass] || "#8b929d") + ";'>"
      + (c.label || "—") + " <span class='muted' style='font-size:10px;'>("
      + c.klass + ")</span></td>"
    + "<td style='padding:4px 6px;color:#8b929d;'>" + (c.headline || "—") + "</td></tr>");
  document.getElementById("score-rows").innerHTML =
    rows.length ? rows.join("") : "<tr><td colspan='3' class='muted' style='padding:6px;'>—</td></tr>";
}

// ───────── boot ─────────
refresh();
refreshSignals();
refreshAnalytics();
refreshSectorPulse();
refreshBriefing();
refreshSuggestions();
refreshRisk();
refreshPositionBlowup();
refreshEarningsRisk();
refreshGreeks();
refreshHeatmap();
refreshDedupedNews();
refreshScorer();
refreshThesis();
refreshDrawdown();
refreshCalibration();
refreshDecisionHealth();
refreshLiquidity();
refreshDecisionForensics();
refreshDecisionDrought();
refreshNewsEdge();
refreshScorerConfidence();
refreshDisagreement();
refreshDataFeed();
refreshValidation();
refreshTradeAsymmetry();
refreshCapitalParalysis();
refreshOpenAttribution();
refreshFeedHealth();
refreshRunnerHeartbeat();
refreshDecisionReliability();
refreshBaselineCompare();
refreshFundedSuggestions();
refreshSignalFollowThrough();
refreshChurn();
refreshThesisDrift();
refreshLoserAutopsy();
refreshWinnerAutopsy();
refreshTrackRecord();
refreshCorrelation();
refreshSourceEdge();
refreshScorecard();
refreshSessionDelta();
refreshGlobalStale();
setInterval(refresh, 15_000);
setInterval(refreshSignals, 30_000);
setInterval(refreshAnalytics, 30_000);
setInterval(refreshSectorPulse, 60_000);
setInterval(refreshBriefing, 60_000);
setInterval(refreshSuggestions, 45_000);
setInterval(refreshRisk, 30_000);
setInterval(refreshPositionBlowup, 30_000);
setInterval(refreshEarningsRisk, 300_000);
setInterval(refreshGreeks, 60_000);
setInterval(refreshHeatmap, 60_000);
setInterval(refreshDedupedNews, 45_000);
setInterval(refreshScorer, 60_000);
setInterval(refreshThesis, 60_000);
setInterval(refreshDrawdown, 30_000);
setInterval(refreshCalibration, 120_000);
setInterval(refreshDecisionHealth, 60_000);
setInterval(refreshLiquidity, 30_000);
setInterval(refreshDecisionForensics, 60_000);
setInterval(refreshDecisionDrought, 60_000);
setInterval(refreshNewsEdge, 300_000);
setInterval(refreshScorerConfidence, 120_000);
setInterval(refreshBaselineCompare, 120_000);
setInterval(refreshDisagreement, 60_000);
setInterval(refreshDataFeed, 60_000);
setInterval(refreshValidation, 120_000);
setInterval(refreshTradeAsymmetry, 60_000);
setInterval(refreshCapitalParalysis, 45_000);
setInterval(refreshOpenAttribution, 60_000);
setInterval(refreshFeedHealth, 60_000);
setInterval(refreshRunnerHeartbeat, 60_000);
setInterval(refreshDecisionReliability, 60_000);
setInterval(refreshFundedSuggestions, 45_000);
setInterval(refreshSignalFollowThrough, 300_000);
setInterval(refreshChurn, 60_000);
setInterval(refreshThesisDrift, 60_000);
setInterval(refreshLoserAutopsy, 60_000);
setInterval(refreshWinnerAutopsy, 60_000);
setInterval(refreshTrackRecord, 60_000);
setInterval(refreshCorrelation, 120_000);
setInterval(refreshSourceEdge, 300_000);
setInterval(refreshScorecard, 60_000);
setInterval(refreshSessionDelta, 60_000);
setInterval(refreshGlobalStale, 60_000);
showTab(INITIAL_TAB || "trader");
</script>
</div><!-- /.page-content -->

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


def _api_prefix() -> str:
    return request.headers.get("X-Forwarded-Prefix", "").rstrip("/")


@app.route("/")
def index():
    return render_template_string(TEMPLATE, initial_tab="trader", api_prefix=_api_prefix())


@app.route("/backtests")
def backtests_page():
    return render_template_string(TEMPLATE, initial_tab="backtests", api_prefix=_api_prefix())


# ─────────────────────────────────────────────────────────────────────────
# Per-ticker drill-down + missed-opportunity radar
#
# A trader inspecting one name (e.g. "MU") otherwise has to cross three
# surfaces by hand: the live lot + marks, the closed round-trip history, the
# Opus reasoning that touched it, and the live news flow. /api/ticker/<sym>
# fuses them via the pure analytics.ticker_dossier SSOT; /ticker/<sym> is the
# standalone page on top of it (self-contained — deliberately NOT a new tab
# in the 9k-line SPA TEMPLATE, so it can't merge-conflict that file).
# /api/watchlist-opportunities is the orthogonal panel: watchlist names with
# live news heat that the book has NO exposure to.
# ─────────────────────────────────────────────────────────────────────────
_TICKER_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<title>{{ symbol }} · Paper Trader</title>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<style>
 :root{color-scheme:dark}
 *{box-sizing:border-box;margin:0;padding:0}
 body{background:#0c0d0f;color:#e8eaed;font:14px/1.5 "Outfit",system-ui,sans-serif;padding:20px;max-width:1080px;margin:0 auto}
 h1{font:700 26px/1.1 "Syne",sans-serif;letter-spacing:-.5px}
 h2{font:600 13px/1 "Syne",sans-serif;text-transform:uppercase;letter-spacing:1.5px;color:#7d828c;margin:26px 0 10px}
 a{color:#00d4ff;text-decoration:none}a:hover{text-decoration:underline}
 .bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:18px}
 input{background:#0e1012;border:1px solid rgba(255,255,255,.13);color:#e8eaed;padding:8px 12px;border-radius:7px;font:500 14px "DM Mono",monospace;width:130px;text-transform:uppercase}
 button{background:#00d4ff;color:#04222b;border:0;padding:8px 16px;border-radius:7px;font:600 13px "Outfit";cursor:pointer}
 .card{background:#111316;border:1px solid rgba(255,255,255,.07);border-radius:12px;padding:16px 18px;margin-bottom:14px}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
 .kpi .v{font:600 22px "DM Mono",monospace}.kpi .l{font-size:11px;color:#7d828c;text-transform:uppercase;letter-spacing:1px}
 .pos{color:#00ff9f}.neg{color:#ff3c4c}.muted{color:#7d828c}
 table{width:100%;border-collapse:collapse;font:13px "DM Mono",monospace}
 th{text-align:left;color:#7d828c;font-weight:500;font-size:11px;text-transform:uppercase;letter-spacing:.5px;padding:6px 8px;border-bottom:1px solid rgba(255,255,255,.1)}
 td{padding:6px 8px;border-bottom:1px solid rgba(255,255,255,.05);vertical-align:top}
 .pill{display:inline-block;padding:2px 8px;border-radius:99px;font-size:11px;font-weight:600;background:#17191d}
 .err{color:#ff3c4c;padding:30px;text-align:center}
 .news h4{font:600 14px "Outfit";margin-bottom:2px}.news .meta{font-size:11px;color:#7d828c;margin-bottom:10px}
 .reason{color:#b9bdc6;font-size:12px;margin-top:3px;white-space:pre-wrap}
</style></head><body>
<div class="bar">
 <h1 id="sym">…</h1>
 <span style="flex:1"></span>
 <input id="q" placeholder="TICKER" autocomplete="off"/>
 <button onclick="go()">Open</button>
 <a href="{{ api_prefix }}/" style="margin-left:8px">← dashboard</a>
</div>
<div id="root"><div class="muted">loading…</div></div>
<script>
const API_PREFIX={{ api_prefix|tojson }}, SYMBOL={{ symbol|tojson }};
function go(){const v=document.getElementById('q').value.trim().toUpperCase();if(v)location.href=API_PREFIX+'/ticker/'+encodeURIComponent(v);}
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')go();});
const fmtUsd=n=>(n==null?'—':(n<0?'-$':'$')+Math.abs(n).toFixed(2));
const cls=n=>n==null?'muted':(n>0?'pos':(n<0?'neg':''));
const esc=s=>(s==null?'':String(s)).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
async function load(){
 document.getElementById('sym').textContent=SYMBOL;
 let d;try{const r=await fetch(API_PREFIX+'/api/ticker/'+encodeURIComponent(SYMBOL));d=await r.json();}
 catch(e){document.getElementById('root').innerHTML='<div class="err">failed to load</div>';return;}
 if(d.error){document.getElementById('root').innerHTML='<div class="err">'+esc(d.error)+'</div>';return;}
 const R=d.realized||{}, P=d.position, NW=(d.news||{}), S=NW.sentiment||{};
 let h='';
 if(!d.has_coverage){h+='<div class="card muted">No position, trades, decisions or live news on file for '+esc(d.symbol)+'.</div>';}
 if(P){let lg=P.legs.map(l=>'<tr><td>'+esc(l.type)+(l.strike?(' '+l.strike+(l.expiry?(' '+esc(l.expiry)):'')):'')+'</td><td>'+l.qty+'</td><td>'+fmtUsd(l.avg_cost)+'</td><td>'+fmtUsd(l.current_price)+'</td><td class="'+cls(l.unrealized_pl)+'">'+fmtUsd(l.unrealized_pl)+'</td></tr>').join('');
  h+='<div class="card"><h2 style="margin-top:0">Open position</h2><table><tr><th>type</th><th>qty</th><th>avg cost</th><th>mark</th><th>unreal P/L</th></tr>'+lg+'</table><div style="margin-top:8px">Total unrealized: <span class="'+cls(P.unrealized_pl_total)+'">'+fmtUsd(P.unrealized_pl_total)+'</span></div></div>';}
 else if(d.has_coverage){h+='<div class="card muted">Not currently held.</div>';}
 h+='<h2>Realized (this name)</h2><div class="card grid">'
  +'<div class="kpi"><div class="v">'+(R.n_round_trips||0)+'</div><div class="l">round-trips</div></div>'
  +'<div class="kpi"><div class="v '+cls(R.total_pnl_usd)+'">'+fmtUsd(R.total_pnl_usd)+'</div><div class="l">total P/L</div></div>'
  +'<div class="kpi"><div class="v">'+(R.win_rate_pct==null?'—':R.win_rate_pct+'%')+'</div><div class="l">win rate</div></div>'
  +'<div class="kpi"><div class="v">'+(R.avg_hold_days==null?'—':R.avg_hold_days+'d')+'</div><div class="l">avg hold</div></div></div>';
 if((d.round_trips||[]).length){h+='<h2>Round-trips</h2><div class="card"><table><tr><th>entry</th><th>exit</th><th>qty</th><th>P/L $</th><th>P/L %</th><th>days</th></tr>'
  +d.round_trips.map(rt=>'<tr><td>'+esc((rt.entry_ts||'').slice(0,10))+'</td><td>'+esc((rt.exit_ts||'').slice(0,10))+'</td><td>'+(rt.qty??'')+'</td><td class="'+cls(rt.pnl_usd)+'">'+fmtUsd(rt.pnl_usd)+'</td><td class="'+cls(rt.pnl_pct)+'">'+(rt.pnl_pct==null?'—':rt.pnl_pct.toFixed(1)+'%')+'</td><td>'+(rt.hold_days==null?'—':rt.hold_days)+'</td></tr>').join('')+'</table></div>';}
 if((d.decisions||[]).length){h+='<h2>Opus decision trail</h2>'+d.decisions.map(x=>'<div class="card"><span class="pill">'+esc(x.verb)+'</span> <span class="muted">'+esc((x.timestamp||'').slice(0,16).replace("T"," "))+'</span><div class="reason">'+esc(x.reasoning)+'</div></div>').join('');}
 h+='<h2>Live news — '+(S.n||0)+' mentions · avg '+(S.avg_score||0).toFixed(1)+' · '+(S.urgent||0)+' urgent</h2>';
 if((NW.articles||[]).length){h+='<div class="card news">'+NW.articles.map(a=>'<div style="margin-bottom:14px"><h4>'+(a.url?'<a href="'+esc(a.url)+'" target="_blank" rel="noopener">'+esc(a.title)+'</a>':esc(a.title))+'</h4><div class="meta">'+esc(a.source)+' · ai '+(a.ai_score==null?'—':a.ai_score.toFixed(1))+' · '+esc((a.first_seen||'').slice(0,16).replace("T"," "))+'</div><div class="reason">'+esc(a.summary)+'</div></div>').join('')+'</div>';}
 else{h+='<div class="card muted">No live articles mention '+esc(d.symbol)+' in the last 24h.</div>';}
 h+='<div class="muted" style="margin:18px 0;font-size:11px">generated '+esc(d.generated_at)+'</div>';
 document.getElementById('root').innerHTML=h;
}
load();
</script></body></html>"""


@app.route("/ticker/<sym>")
def ticker_page(sym):
    """Standalone per-ticker drill-down page (self-contained; consumes
    /api/ticker/<sym> client-side). Kept off the SPA TEMPLATE on purpose."""
    return render_template_string(_TICKER_TEMPLATE,
                                  symbol=(sym or "").upper().strip(),
                                  api_prefix=_api_prefix())


@app.route("/api/ticker/<sym>")
def ticker_api(sym):
    """Cross-system dossier for one name: live lot + marks, closed round-trip
    P&L (this name only), the Opus decision trail that touched it, and the
    live news flow + sentiment.

    Intentionally NOT @swr_cached: that decorator keys on the query string
    only (`name + "?" + qs`), so a <sym> *path* param would collide across
    tickers and serve MU's dossier for NVDA. This endpoint is also lighter
    than the un-cached `/api/portfolio` peer — only stored marks + two
    read-only sqlite reads the `signals` layer already self-degrades on, no
    yfinance and no HTTP — so the hot path stays bounded without SWR."""
    try:
        from .analytics.ticker_dossier import build_ticker_dossier
        from . import signals as _sig
        store = get_store()
        symu = (sym or "").upper().strip()
        positions = store.open_positions()
        trades = store.recent_trades(500)
        decisions = store.recent_decisions(120)
        sigs = _sig.get_top_signals(n=80, hours=24, min_score=0.0)
        sentiment = _sig.get_ticker_sentiment(symu, hours=24)
        out = build_ticker_dossier(
            symu, positions=positions, trades=trades, decisions=decisions,
            signals_list=sigs, sentiment=sentiment,
            parse_action_ticker=_parse_action_ticker)
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e), "symbol": (sym or "").upper().strip()}), 500


@app.route("/api/watchlist-opportunities")
@swr_cached("watchlist-opportunities", 60.0)
def watchlist_opportunities_api():
    """Watchlist names with live news heat that the book has NO position in —
    the missed-opportunity radar (orthogonal to every position-centric panel).
    One signals fetch; the pure SSOT tallies per ticker (no N-query fan-out)."""
    try:
        from .analytics.watchlist_opportunities import build_watchlist_opportunities
        from .strategy import WATCHLIST as _WATCHLIST
        from . import signals as _sig
        store = get_store()
        held = {str(p.get("ticker") or "").upper()
                for p in store.open_positions() if (p.get("qty") or 0) > 0}
        sigs = _sig.get_top_signals(n=300, hours=24, min_score=0.0)
        out = build_watchlist_opportunities(_WATCHLIST, held, sigs)
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e), "opportunities": []}), 500


@app.route("/api/state")
@swr_cached("state", 15.0)
def state():
    """Main trader-page payload (portfolio + positions + trades + 5000-point
    equity curve + decisions). This is the page's lifeline, polled every 15s
    by `refresh()` AND cross-fetched, and it is the heaviest pure-DB read:
    six lock-held `Store` reads and a ~145KB body (eq 5000 + 500 trades).
    Live-user testing on 2026-05-17 measured it at 8.7s under concurrent
    load — it head-of-line-blocked the whole page because it was the only
    high-traffic core endpoint NOT behind `swr_cached` (every slow network
    endpoint already is). The portfolio only changes on a decision cycle
    (`runner.OPEN_INTERVAL_S` ≥ 1800s), so a 15s stale-while-revalidate
    window is invisible to a trader while it serves instantly from the last
    good payload and single-flight-refreshes in the background — and the
    runner already pushes every fill to Discord immediately regardless. The
    injected `cached`/`cache_age_s` keys make staleness explicit (the
    command-center honesty contract); `refresh()` tolerates the SWR cold
    `{"warming":true}` placeholder (it skips the tick and self-heals)."""
    store = get_store()
    pf = store.get_portfolio()
    positions = store.open_positions()
    trades = store.recent_trades(40)
    decisions = store.recent_decisions(20)
    eq = store.equity_curve(5000)  # full history for accurate chart
    sp = eq[-1]["sp500_price"] if eq else None
    # Include all trades for chart markers (not just recent 40)
    all_trades = store.recent_trades(500)
    return jsonify({
        "now": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "portfolio": pf,
        "positions": positions,
        "trades": trades,
        "decisions": decisions,
        "equity": eq,
        "sp500": sp,
        "all_trades": all_trades,
    })


@app.route("/api/portfolio")
def portfolio_api():
    """Compact public read of the portfolio — consumed by Digital Intern's dashboard.

    Backward-compatible: the original three keys (``total_value``, ``cash``,
    ``starting_value``) remain unchanged. We additionally expose at-a-glance
    trader-actionable fields composed *purely* from the already-cached
    ``portfolio.positions_json`` row (no extra store reads, no network — so
    this endpoint stays the lean lowest-latency public surface):

      * ``n_positions`` — open lots count
      * ``open_value`` — Σ market_value across open lots (i.e. total_value − cash)
      * ``unrealized_pl`` / ``unrealized_pl_pct`` — book-wide drift since entry,
        the single answer to a trader's first-glance question "am I up or down
        on what I'm holding?". ``unrealized_pl_pct`` is over ``total_value``
        (the equity base) to align with ``benchmark`` / ``drawdown`` % framing.
      * ``stale_marks`` — count of positions flagged ``stale_mark=True``
        (yfinance returned nothing this cycle; the mark fell back to avg_cost).
        Nonzero means the unrealized_pl above understates real exposure —
        explicit so the trader is never misled by a phantom "flat" book the
        ``stale_mark`` flag was added to expose.
      * ``last_updated`` — ISO timestamp of the most recent mark-to-market
        write (``store.update_portfolio``'s own clock). Lets a polling caller
        detect "the trader has stopped writing" without re-reading equity_curve.
      * ``pnl_vs_start`` / ``pnl_vs_start_pct`` — absolute and % delta from
        the $1000 baseline (``INITIAL_CASH``, invariant #12; never a literal).

    Pure read, never raises: every derivation is wrapped so a malformed
    ``positions_json`` (defensive — ``get_portfolio`` already falls it back
    to ``[]``) or a non-numeric column degrades to safe zeros while the
    three legacy keys are always present.
    """
    store = get_store()
    pf = store.get_portfolio()

    cash_raw = pf.get("cash")
    total_raw = pf.get("total_value")
    try:
        cash = float(cash_raw) if cash_raw is not None else None
    except (TypeError, ValueError):
        cash = None
    try:
        total_value = float(total_raw) if total_raw is not None else None
    except (TypeError, ValueError):
        total_value = None

    positions = pf.get("positions") or []
    if not isinstance(positions, list):
        positions = []

    n_positions = 0
    open_value = 0.0
    unrealized_pl = 0.0
    stale_marks = 0
    for p in positions:
        if not isinstance(p, dict):
            continue
        n_positions += 1
        try:
            mv = float(p.get("market_value") or 0.0)
        except (TypeError, ValueError):
            mv = 0.0
        open_value += mv
        try:
            pl = float(p.get("unrealized_pl") or 0.0)
        except (TypeError, ValueError):
            pl = 0.0
        unrealized_pl += pl
        if p.get("stale_mark"):
            stale_marks += 1

    # `unrealized_pl_pct` is the book-wide P/L over equity — aligns with the
    # benchmark / drawdown framing (`/api/benchmark` returns pct over the
    # equity baseline). Falls back to None when total_value is missing /
    # non-positive (no meaningful denominator) so the panel renders "n/a"
    # instead of an unbounded number.
    pnl_pct = None
    if total_value is not None and total_value > 0:
        pnl_pct = round(unrealized_pl / total_value * 100.0, 2)

    pnl_vs_start = None
    pnl_vs_start_pct = None
    if total_value is not None:
        pnl_vs_start = round(total_value - INITIAL_CASH, 2)
        if INITIAL_CASH > 0:
            pnl_vs_start_pct = round(
                (total_value / INITIAL_CASH - 1.0) * 100.0, 2)

    return jsonify({
        "total_value": total_value,
        "cash": cash,
        "starting_value": INITIAL_CASH,
        "n_positions": n_positions,
        "open_value": round(open_value, 2),
        "unrealized_pl": round(unrealized_pl, 2),
        "unrealized_pl_pct": pnl_pct,
        "stale_marks": stale_marks,
        "last_updated": pf.get("last_updated"),
        "pnl_vs_start": pnl_vs_start,
        "pnl_vs_start_pct": pnl_vs_start_pct,
    })


@app.route("/api/closed-positions")
def closed_positions_api():
    """Closed position lots with realized P&L computed from each lot's
    matching round-trip of trades (BUY/BUY_CALL/BUY_PUT → SELL/SELL_CALL/
    SELL_PUT). Supports ``?limit=N`` (1..1000, default 100). Returns
    newest-closed first plus a rollup summary.

    The summary surfaces:

      * ``n``, ``wins``, ``losses``, ``flat`` — counts (flat = realized 0)
      * ``win_rate_pct`` — wins / (wins + losses), null when no decided lot
      * ``total_realized_pl`` — Σ realized across all listed lots
      * ``total_cost`` / ``total_proceeds`` — gross BUY / SELL dollar flow
      * ``avg_realized_pl_pct`` — cost-weighted realized / cost across the
        slice, null when total_cost is non-positive (a simple mean of
        per-lot percentages would over-weight a tiny lot up 100% against a
        large lot down 10%, so this is the honest portfolio-level number)
      * ``avg_winner_pct`` / ``avg_loser_pct`` — un-weighted mean of the
        per-lot realized_pl_pct inside each bucket, so a trader sees both
        edges of the payoff ratio at a glance (None when bucket is empty)
      * ``median_hold_days`` — median of per-lot hold_days, null when no
        lot has a parseable opened_at/closed_at pair
    """
    try:
        limit = max(1, min(int(request.args.get("limit", 100)), 1000))
    except (TypeError, ValueError):
        limit = 100
    store = get_store()
    lots = store.closed_positions(limit=limit)
    wins = sum(1 for p in lots if (p.get("realized_pl") or 0) > 0)
    losses = sum(1 for p in lots if (p.get("realized_pl") or 0) < 0)
    total_realized = round(sum((p.get("realized_pl") or 0.0) for p in lots), 2)
    total_cost = round(sum((p.get("cost") or 0.0) for p in lots), 2)
    total_proceeds = round(sum((p.get("proceeds") or 0.0) for p in lots), 2)
    decided = wins + losses
    win_rate = round(100.0 * wins / decided, 2) if decided else None
    avg_pl_pct = (round(total_realized / total_cost * 100.0, 2)
                  if total_cost > 1e-9 else None)
    win_pcts = [p["realized_pl_pct"] for p in lots
                if (p.get("realized_pl") or 0) > 0
                and isinstance(p.get("realized_pl_pct"), (int, float))]
    loss_pcts = [p["realized_pl_pct"] for p in lots
                 if (p.get("realized_pl") or 0) < 0
                 and isinstance(p.get("realized_pl_pct"), (int, float))]
    avg_winner_pct = (round(sum(win_pcts) / len(win_pcts), 2)
                      if win_pcts else None)
    avg_loser_pct = (round(sum(loss_pcts) / len(loss_pcts), 2)
                     if loss_pcts else None)
    holds = sorted(p["hold_days"] for p in lots
                   if isinstance(p.get("hold_days"), (int, float)))
    if holds:
        mid = len(holds) // 2
        median_hold_days = round(
            holds[mid] if len(holds) % 2
            else (holds[mid - 1] + holds[mid]) / 2,
            4,
        )
    else:
        median_hold_days = None
    return jsonify({
        "positions": lots,
        "summary": {
            "n": len(lots),
            "wins": wins,
            "losses": losses,
            "flat": len(lots) - wins - losses,
            "total_realized_pl": total_realized,
            "total_cost": total_cost,
            "total_proceeds": total_proceeds,
            "win_rate_pct": win_rate,
            "avg_realized_pl_pct": avg_pl_pct,
            "avg_winner_pct": avg_winner_pct,
            "avg_loser_pct": avg_loser_pct,
            "median_hold_days": median_hold_days,
        },
    })


@app.route("/api/data-feed")
@swr_cached("data-feed", 30.0)
def data_feed_api():
    """Live news collector pulse — proxies digital-intern's articles.db.

    Returns articles-per-hour, per-24h, and top active sources, all filtered to
    exclude backtest synthetic rows (per the live-only invariant — see CLAUDE.md
    §5 in digital-intern). Returns zeros if the article DB isn't reachable so
    the widget can render gracefully on the live trader page.

    Resolves the DB through the freshness-aware ``_articles_db_path()`` (→
    ``signals._db_path()``, invariant #17) — the SAME single source of truth
    the live trader and every other news-analytics endpoint use. The old
    hardcoded candidate list both (a) bypassed the split-brain-safe resolver
    (so this panel could read a stale USB mirror while the trader read fresh
    LOCAL — the exact failure invariant #17 closed everywhere else) and (b)
    pinned the **pre-migration** path ``/home/zeph/digital-intern/...``, which
    only resolves on this box via a legacy symlink; on a clean checkout it
    silently zeroed the live news-pulse panel with "articles.db not found".
    """
    db_path = _articles_db_path()
    if db_path is None:
        return jsonify({"articles_1h": 0, "articles_24h": 0, "top_sources": [],
                        "error": "articles.db not found"})
    try:
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        cut_1h  = (now - timedelta(hours=1)).isoformat()
        cut_24h = (now - timedelta(hours=24)).isoformat()
        live_clause = (
            "url NOT LIKE 'backtest://%' "
            "AND source NOT LIKE 'backtest_%' "
            "AND source NOT LIKE 'opus_annotation%'"
        )
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=3.0)
        try:
            n1 = conn.execute(
                f"SELECT COUNT(*) FROM articles WHERE first_seen >= ? AND {live_clause}",
                (cut_1h,)
            ).fetchone()[0]
            n24 = conn.execute(
                f"SELECT COUNT(*) FROM articles WHERE first_seen >= ? AND {live_clause}",
                (cut_24h,)
            ).fetchone()[0]
            top = conn.execute(
                f"SELECT source, COUNT(*) FROM articles "
                f"WHERE first_seen >= ? AND {live_clause} "
                f"GROUP BY source ORDER BY 2 DESC LIMIT 5",
                (cut_1h,)
            ).fetchall()
        finally:
            conn.close()
        return jsonify({
            "articles_1h": int(n1 or 0),
            "articles_24h": int(n24 or 0),
            "top_sources": [{"name": r[0] or "?", "count": int(r[1] or 0)} for r in top],
        })
    except Exception as e:
        return jsonify({"articles_1h": 0, "articles_24h": 0, "top_sources": [],
                        "error": str(e)})


@app.route("/api/backtests")
@swr_cached("backtests-list", 30.0)
def backtests_api():
    # 30s stale-while-revalidate TTL: ~20 concurrent backtest *writer*
    # threads otherwise starve this reader (the metadata-only list is cheap
    # but the underlying sqlite is write-contended). SWR serves the last
    # good payload instantly and single-flight-refreshes in the background,
    # the established pattern in this module (see swr_cached docstring).
    from datetime import datetime, timezone
    try:
        from .backtest import BacktestStore
        store = BacktestStore()
        # Strip equity curves from the list — clients fetch curves lazily via
        # /api/backtests/curves when needed. This cuts payload from ~5MB to ~50KB.
        runs = store.all_runs(include_curves=False)
        completed = [r for r in runs if r.get("status") == "complete"]
        # spy_baseline is computed over the *full* completed set so it stays
        # stable regardless of which page is requested.
        spy_baseline = completed[0].get("spy_return_pct") if completed else None
        total_count = len(runs)

        # Backward-compatible pagination: only paginate when the client
        # explicitly asks (?page=). The default (no page param) still returns
        # every run — the dashboard's scatter chart, era heatmap, aggregate
        # stats and table all consume the full client-side list, so silently
        # capping it at 50 would regress the page this change speeds up.
        # ?page= callers get the most-recent runs first (run_id desc), which
        # is what a "showing 50 of 501" UI implies.
        page_arg = request.args.get("page")
        paginated = page_arg is not None
        page = limit = None
        if paginated:
            try:
                page = max(1, int(page_arg))
            except (TypeError, ValueError):
                page = 1
            try:
                limit = int(request.args.get("limit", 50))
            except (TypeError, ValueError):
                limit = 50
            limit = max(1, min(limit, 500))
            ordered = sorted(runs, key=lambda r: r.get("run_id") or 0,
                             reverse=True)
            start = (page - 1) * limit
            runs = ordered[start:start + limit]

        payload = {
            "runs": runs,
            "total_runs": total_count,
            "total_count": total_count,
            "spy_baseline": spy_baseline,
            "qqq_baseline": None,
            "last_updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if paginated:
            payload["page"] = page
            payload["limit"] = limit
            payload["paginated"] = True
        return jsonify(payload)
    except Exception as e:
        return jsonify({"runs": [], "error": str(e)})


@app.route("/api/backtests/curves")
def backtest_curves_api():
    """Return normalized equity curves for requested run_ids.

    Query: ?run_ids=1,2,3
    Returns: {run_id: [{date, day_index, value, value_pct}, ...], ...}
    value_pct is % gain from start_value — comparable across different windows.
    """
    try:
        raw_ids = request.args.get("run_ids", "").strip()
        ids = [int(x.strip()) for x in raw_ids.split(",") if x.strip().isdigit()]
        if not ids:
            return jsonify({"error": "missing run_ids"}), 400
        # Degrade gracefully instead of 400-ing the whole request: a stale
        # client (cached pre-chunking JS) still gets a usable chart rather
        # than an empty one. run_curves() batches the DB query internally,
        # so large id lists are safe; we only clamp absurd payloads.
        if len(ids) > 1000:
            ids = ids[:1000]
        from .backtest import BacktestStore
        store = BacktestStore()
        curves = store.run_curves(ids)
        # keyed by string for JSON compatibility
        return jsonify({str(k): v for k, v in curves.items()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/backtests/leaderboard")
@swr_cached("backtests-leaderboard", 30.0)
def backtest_leaderboard_api():
    """Top backtest runs ranked by a selectable metric.

    Query:
      ?metric=vs_spy_pct|total_return_pct|annualized_return_pct|final_value|n_trades
      ?limit=N            (default 20, capped at 200)
      ?min_trades=N       (default 1 — excludes runs that never traded)
    Returns: {"metric": ..., "count": N, "runs": [ {run_id, metric, ...}, ... ]}
    Ranked descending; the strategy's edge over SPY (vs_spy_pct) is the default.
    """
    ALLOWED = {
        "vs_spy_pct", "total_return_pct", "annualized_return_pct",
        "final_value", "n_trades",
    }
    try:
        metric = (request.args.get("metric") or "vs_spy_pct").strip()
        if metric not in ALLOWED:
            return jsonify({"error": f"metric must be one of {sorted(ALLOWED)}"}), 400
        try:
            limit = max(1, min(200, int(request.args.get("limit", 20))))
        except ValueError:
            limit = 20
        try:
            min_trades = max(0, int(request.args.get("min_trades", 1)))
        except ValueError:
            min_trades = 1

        from .backtest import BacktestStore
        store = BacktestStore()
        runs = store.all_runs(include_curves=False)
        completed = [
            r for r in runs
            if r.get("status") == "complete"
            and (r.get("n_trades") or 0) >= min_trades
            and r.get(metric) is not None
        ]
        completed.sort(key=lambda r: r.get(metric), reverse=True)
        top = completed[:limit]
        return jsonify({
            "metric": metric,
            "limit": limit,
            "min_trades": min_trades,
            "count": len(top),
            "total_eligible": len(completed),
            "runs": [
                {
                    "run_id": r.get("run_id"),
                    "metric": r.get(metric),
                    "total_return_pct": r.get("total_return_pct"),
                    "vs_spy_pct": r.get("vs_spy_pct"),
                    "annualized_return_pct": r.get("annualized_return_pct"),
                    "final_value": r.get("final_value"),
                    "n_trades": r.get("n_trades"),
                    "start_date": r.get("start_date"),
                    "end_date": r.get("end_date"),
                    "seed": r.get("seed"),
                }
                for r in top
            ],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/model-rankings")
def api_model_rankings():
    """Aggregated backtest stats per model_id, ranked by avg_return_pct desc.

    Reads from the module-level BACKTEST_DB so tests can monkeypatch the path.
    """
    try:
        conn = sqlite3.connect(str(BACKTEST_DB), timeout=10)
        try:
            rows = conn.execute("""
                SELECT
                    model_id,
                    COUNT(*) AS runs,
                    ROUND(AVG(total_return_pct), 2) AS avg_return_pct,
                    ROUND(MAX(total_return_pct), 2) AS best_return_pct,
                    ROUND(AVG(vs_spy_pct), 2) AS avg_vs_spy_pct,
                    ROUND(AVG(n_trades), 1) AS avg_trades,
                    ROUND(
                        100.0 * SUM(CASE WHEN total_return_pct > 0 THEN 1 ELSE 0 END)
                        / COUNT(*), 1
                    ) AS win_rate_pct,
                    SUM(n_decisions) AS total_decisions,
                    GROUP_CONCAT(total_return_pct ORDER BY total_return_pct) AS sorted_returns
                FROM backtest_runs
                WHERE status = 'complete'
                GROUP BY model_id
                ORDER BY avg_return_pct DESC
            """).fetchall()
        finally:
            conn.close()
        models = []
        for r in rows:
            mid = r[0] or "ml_quant"
            # Compute median from the sorted comma-separated returns string
            raw_returns = r[8]
            if raw_returns:
                vals = [float(x) for x in raw_returns.split(",") if x]
                n = len(vals)
                median_pct = round((vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2), 2)
            else:
                median_pct = None
            models.append({
                "model_id": mid,
                "display_name": _MODEL_DISPLAY_NAMES.get(mid, mid),
                "runs": r[1],
                "avg_return_pct": r[2],
                "best_return_pct": r[3],
                "median_return_pct": median_pct,
                "avg_vs_spy_pct": r[4],
                "avg_trades": r[5],
                "win_rate_pct": r[6],
                "total_decisions": r[7],
            })
        return jsonify({"models": models, "as_of": datetime.now(timezone.utc).isoformat()})
    except Exception as e:
        return jsonify({"error": str(e), "models": []}), 500


@app.route("/api/persona-leaderboard")
def api_persona_leaderboard():
    """Per-persona strategy-quality leaderboard — which of the 10 trading
    styles actually carry repeatable alpha.

    ``/api/model-rankings`` aggregates ``backtest_runs`` by ``model_id`` (the
    decision *engine*). This sibling aggregates the *other* dimension stamped
    on every run — the persona (Value / Momentum / Contrarian / Global Macro /
    GARP / Quant / Sector Rotator / Small-Mid Cap / ESG / Pure Speculator) —
    answering the structural question "is momentum or value or pure
    speculation the edge in this regime".

    The diagnostic itself — ``paper_trader.ml.persona_leaderboard`` — already
    existed (read-only, exhaustively unit-tested, ``persona_for`` SSOT for the
    ``run_id → persona`` map, equity-curve risk metrics) but was reachable
    from no endpoint: an operator could only see it from a shell. This route
    is a thin wrapper — it reuses the module's own ``_load_runs`` DB read and
    the ``persona_leaderboard`` builder verbatim (AGENTS.md invariant #10), so
    the dashboard panel and the CLI digest can never disagree. Observational
    only — never gates the live loop, never prunes a persona (#2 / #12).
    """
    from pathlib import Path

    from paper_trader.ml.persona_leaderboard import (
        _load_runs,
        persona_leaderboard,
    )

    try:
        db = Path(str(BACKTEST_DB))
        runs = _load_runs(db) if db.exists() else []
        report = persona_leaderboard(runs)
        report["as_of"] = datetime.now(timezone.utc).isoformat()
        return jsonify(report)
    except Exception as e:
        return jsonify({
            "error": str(e), "status": "error", "verdict": "error",
            "leaderboard": [], "drag_personas": [], "hint": str(e),
        }), 500


@app.route("/api/per-ticker-skill")
def api_per_ticker_skill():
    """Per-ticker out-of-sample skill of the DecisionScorer — which names
    the ML model can actually rank-predict, and which it is anti-predictive
    on.

    ``/api/scorer-confidence`` / ``/api/calibration`` report the scorer's
    *aggregate* OOS skill — a single rank-IC over every outcome. That number
    can sit near zero while hiding a scorer that is genuinely skilled on some
    tickers and actively *inverted* on others (the aggregate is a mix
    artifact). This sibling decomposes the same out-of-sample outcomes by
    ticker: per-name rank-IC, directional accuracy, magnitude bias and a
    crisp ``SIGNAL_EDGE`` / ``WEAK_SIGNAL_EDGE`` / ``NO_SIGNAL_EDGE`` /
    ``INVERTED_SIGNAL`` / ``SPARSE`` verdict.

    The diagnostic — ``paper_trader.ml.per_ticker_skill`` — already existed
    (read-only, exhaustively unit-tested, temporal-split SSOT shared with
    ``sector_skill`` / ``persona_skill``) but was reachable from no endpoint:
    an operator could only see it from a shell. This route is a thin wrapper
    that reuses the module's own ``analyze`` end-to-end function **verbatim**
    (it loads ``decision_outcomes.jsonl``, applies the temporal split, loads
    the deployed scorer and computes per-ticker skill), so the dashboard and
    the CLI digest can never disagree. ``analyze`` never raises — every fault
    degrades to a 200 ``status='error'`` / ``insufficient_data`` payload.
    Observational only: never gates the live loop, never excludes a ticker
    (an ``INVERTED_SIGNAL`` row is the *data for* a separate, explicit
    decision — invariants #2 / #12).
    """
    from paper_trader.ml.per_ticker_skill import analyze

    try:
        report = analyze()
        report["as_of"] = datetime.now(timezone.utc).isoformat()
        return jsonify(report)
    except Exception as e:
        return jsonify({
            "error": str(e), "status": "error", "verdict": "error",
            "tickers": [], "inverted_tickers": [], "hint": str(e),
        }), 500


@app.route("/api/backtests/stats")
@swr_cached("backtests-stats", 30.0)
def backtest_stats_api():
    """Aggregate distribution across all completed backtest runs.

    Gives a quant a one-glance read on whether the strategy has a real edge:
    how many runs completed, the central tendency of return / vs-SPY, and
    the share of runs that actually beat SPY.

    Query: ?min_trades=N   (default 1 — excludes runs that never traded)
    """
    def _stats(vals: list[float]) -> dict:
        vals = sorted(v for v in vals if v is not None)
        n = len(vals)
        if n == 0:
            return {"n": 0, "mean": None, "median": None,
                    "min": None, "max": None}
        mid = n // 2
        median = vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2.0
        return {
            "n": n,
            "mean": round(sum(vals) / n, 3),
            "median": round(median, 3),
            "min": round(vals[0], 3),
            "max": round(vals[-1], 3),
        }

    try:
        try:
            min_trades = max(0, int(request.args.get("min_trades", 1)))
        except ValueError:
            min_trades = 1

        from .backtest import BacktestStore
        store = BacktestStore()
        runs = store.all_runs(include_curves=False)
        completed = [
            r for r in runs
            if r.get("status") == "complete"
            and (r.get("n_trades") or 0) >= min_trades
        ]
        vs_spy = [r.get("vs_spy_pct") for r in completed
                  if r.get("vs_spy_pct") is not None]
        beat_spy = sum(1 for v in vs_spy if v > 0)
        return jsonify({
            "min_trades": min_trades,
            "total_runs": len(runs),
            "completed_runs": len(completed),
            "beat_spy_count": beat_spy,
            "beat_spy_rate": round(beat_spy / len(vs_spy), 4) if vs_spy else None,
            "vs_spy_pct": _stats(vs_spy),
            "total_return_pct": _stats(
                [r.get("total_return_pct") for r in completed]),
            "annualized_return_pct": _stats(
                [r.get("annualized_return_pct") for r in completed]),
            "n_trades": _stats(
                [float(r["n_trades"]) for r in completed
                 if r.get("n_trades") is not None]),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/backtests/<int:run_id>")
def backtest_detail(run_id: int):
    try:
        from .backtest import BacktestStore
        store = BacktestStore()
        detail = store.run_detail(run_id)
        if not detail:
            return jsonify({"error": "not found"}), 404
        return jsonify(detail)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/backtests/compare")
def backtest_compare():
    """Side-by-side comparison of 2-4 backtest runs.

    Query: ``/api/backtests/compare?ids=1,2,3`` (comma-separated run_ids).

    Returns equity_curve points re-shaped for overlay rendering:
      - ``day_index`` = days since run's start_date, so runs with different
        windows can be drawn on the same x-axis.
      - ``value_pct`` = (value / start_value - 1) * 100, so returns compare
        on a normalized y-axis regardless of initial cash differences.

    Per-run summary fields (return %, vs_spy %, max drawdown, trade count,
    decision count, win rate) are computed from the same equity_curve + trades
    that the existing /api/backtests/<id> route already returns, so this is a
    pure aggregation — no new state.
    """
    raw_ids = request.args.get("ids", "").strip()
    if not raw_ids:
        return jsonify({"error": "missing ids — e.g. ?ids=1,2,3"}), 400
    try:
        ids = []
        for tok in raw_ids.split(","):
            tok = tok.strip()
            if not tok:
                continue
            ids.append(int(tok))
        if not ids:
            return jsonify({"error": "no valid ids"}), 400
        if len(ids) > 4:
            return jsonify({"error": "max 4 runs per comparison"}), 400
    except (TypeError, ValueError):
        return jsonify({"error": "ids must be comma-separated integers"}), 400

    try:
        from .backtest import BacktestStore
        from datetime import date
        store = BacktestStore()
        out_runs = []
        for rid in ids:
            detail = store.run_detail(rid)
            if not detail:
                out_runs.append({"run_id": rid, "error": "not found"})
                continue
            eq = detail.get("equity_curve") or []
            trades = detail.get("trades") or []
            # Normalize the equity curve for overlay.
            start_val = float(eq[0]["value"]) if eq else 1000.0
            start_date_str = detail.get("start_date") or (eq[0]["date"] if eq else None)
            try:
                start_d = date.fromisoformat(start_date_str) if start_date_str else None
            except (TypeError, ValueError):
                start_d = None

            curve = []
            peak = start_val
            max_dd = 0.0
            for p in eq:
                v = float(p.get("value") or 0.0)
                if v > peak:
                    peak = v
                if peak > 0:
                    dd = (peak - v) / peak * 100.0
                    if dd > max_dd:
                        max_dd = dd
                d_str = p.get("date")
                day_idx = None
                if start_d and d_str:
                    try:
                        day_idx = (date.fromisoformat(d_str) - start_d).days
                    except (TypeError, ValueError):
                        day_idx = None
                curve.append({
                    "date": d_str,
                    "day_index": day_idx,
                    "value": v,
                    "value_pct": round((v / start_val - 1.0) * 100.0, 3) if start_val else 0.0,
                })

            # Win rate from trades that we can pair: BUYs followed by a SELL on the
            # same ticker close at a higher price. Best-effort — backtest trades use
            # ``action`` ∈ {BUY, SELL, BUY_CALL, SELL_CALL, ...}; we score stocks only
            # so the metric stays interpretable.
            wins = 0
            losses = 0
            held: dict[str, list[tuple[float, float]]] = {}  # ticker -> [(qty, price)]
            for t in trades:
                act = (t.get("action") or "").upper()
                tk = t.get("ticker") or ""
                qty = float(t.get("qty") or 0)
                px = float(t.get("price") or 0)
                if not tk or qty <= 0 or px <= 0:
                    continue
                if act == "BUY":
                    held.setdefault(tk, []).append((qty, px))
                elif act == "SELL":
                    lots = held.get(tk) or []
                    remaining = qty
                    while remaining > 0 and lots:
                        lot_qty, lot_px = lots[0]
                        use = min(lot_qty, remaining)
                        if px > lot_px:
                            wins += 1
                        elif px < lot_px:
                            losses += 1
                        if use >= lot_qty:
                            lots.pop(0)
                        else:
                            lots[0] = (lot_qty - use, lot_px)
                        remaining -= use
                    held[tk] = lots
            total_rt = wins + losses
            win_rate = (wins / total_rt) if total_rt else None

            out_runs.append({
                "run_id": rid,
                "start_date": detail.get("start_date"),
                "end_date": detail.get("end_date"),
                "status": detail.get("status"),
                "total_return_pct": detail.get("total_return_pct"),
                "spy_return_pct": detail.get("spy_return_pct"),
                "vs_spy_pct": detail.get("vs_spy_pct"),
                "max_drawdown_pct": round(max_dd, 2),
                "n_trades": detail.get("n_trades"),
                "n_decisions": detail.get("n_decisions"),
                "n_round_trips": total_rt,
                "win_rate": round(win_rate, 4) if win_rate is not None else None,
                "final_value": detail.get("final_value"),
                "start_value": start_val,
                "n_points": len(curve),
                "equity_curve": curve,
            })
        return jsonify({
            "ids": ids,
            "n_runs": len(out_runs),
            "runs": out_runs,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/backtests/<int:run_id>/trades")
def backtest_trades(run_id: int):
    try:
        from .backtest import BacktestStore
        store = BacktestStore()
        detail = store.run_detail(run_id)
        if not detail:
            return jsonify({"error": "not found"}), 404
        return jsonify({"run_id": run_id, "trades": detail.get("trades", [])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/backtests/<int:run_id>/decisions")
def backtest_decisions(run_id: int):
    try:
        from .backtest import BacktestStore
        store = BacktestStore()
        detail = store.run_detail(run_id)
        if not detail:
            return jsonify({"error": "not found"}), 404
        return jsonify({"run_id": run_id, "decisions": detail.get("decisions", [])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/model-progress")
def model_progress():
    """Per-cycle aggregated returns for the Model Progress chart.

    Groups completed runs into cycles of RUNS_PER_CYCLE=5 by run_id order.
    Labels use actual run_id ranges so trimming old runs does not renumber cycles.
    """
    try:
        from .backtest import BacktestStore
        store = BacktestStore()
        rows = store.conn.execute(
            "SELECT run_id, total_return_pct, completed_at FROM backtest_runs "
            "WHERE status='complete' ORDER BY run_id"
        ).fetchall()
        if not rows:
            return jsonify({"cycles": []})

        cycle_size = 5  # RUNS_PER_CYCLE
        cycles = []
        for i in range(0, len(rows), cycle_size):
            chunk = rows[i:i + cycle_size]
            returns = [r["total_return_pct"] for r in chunk]
            run_ids = [r["run_id"] for r in chunk]
            # Use actual run_id range as label so chart is stable across trims
            label = f"#{run_ids[0]}" if len(run_ids) == 1 else f"#{run_ids[0]}-{run_ids[-1]}"
            cycles.append({
                "cycle": label,
                "run_start": run_ids[0],
                "best": round(max(returns), 2),
                "avg": round(sum(returns) / len(returns), 2),
                "worst": round(min(returns), 2),
                "n": len(returns),
                "completed_at": chunk[-1]["completed_at"],
            })
        return jsonify({"cycles": cycles, "total_runs": len(rows)})
    except Exception as e:
        return jsonify({"cycles": [], "error": str(e)})


@app.route("/api/analytics")
def analytics_api():
    """Derived portfolio analytics — sector exposure, drawdown, Sharpe, win rate, daily P/L."""
    try:
        store = get_store()
        pf = store.get_portfolio()
        positions = store.open_positions()
        # Pull a generous trades sample for round-trip accounting.
        trades = list(reversed(store.recent_trades(2000)))  # oldest → newest
        eq = store.equity_curve(5000)  # most recent 5000, ascending after the bugfix

        total_value = pf.get("total_value") or 0.0

        # ─── 1. Sector exposure ───
        sector_usd: dict[str, float] = {}
        for p in positions:
            mult = 100 if p["type"] in ("call", "put") else 1
            price = p.get("current_price") or p["avg_cost"]
            val = price * p["qty"] * mult
            sec = _classify(p["ticker"])
            sector_usd[sec] = sector_usd.get(sec, 0.0) + val

        sector_pct = {
            s: round((v / total_value * 100) if total_value else 0.0, 2)
            for s, v in sector_usd.items()
        }
        cash_pct = round((pf.get("cash", 0) / total_value * 100) if total_value else 0.0, 2)

        # ─── 2. Max drawdown (peak-to-trough on equity curve) ───
        # Return None (not 0.0) when there's no equity history so the frontend's
        # `== null` branch fires and renders "—" instead of "-0.00 (0.00%)".
        max_dd_usd: float | None = None
        max_dd_pct: float | None = None
        if eq:
            max_dd_usd = 0.0
            max_dd_pct = 0.0
            peak = eq[0]["total_value"]
            for p in eq:
                v = p["total_value"]
                if v > peak:
                    peak = v
                dd_usd = peak - v
                dd_pct = (dd_usd / peak * 100) if peak else 0.0
                if dd_usd > max_dd_usd:
                    max_dd_usd = dd_usd
                if dd_pct > max_dd_pct:
                    max_dd_pct = dd_pct

        # ─── 3. Sharpe estimate from daily-bucketed returns ───
        # Bucket equity_curve by date, take last value per date, compute log returns,
        # annualize as mean/std * sqrt(252).
        sharpe = None
        daily_returns: list[float] = []
        by_day: dict[str, float] = {}
        for p in eq:
            day = (p["timestamp"] or "")[:10]
            if day:
                by_day[day] = p["total_value"]  # last write wins, leaves us with EOD close
        day_keys = sorted(by_day.keys())
        for i in range(1, len(day_keys)):
            prev = by_day[day_keys[i - 1]]
            cur = by_day[day_keys[i]]
            if prev and prev > 0:
                daily_returns.append((cur / prev) - 1.0)
        if len(daily_returns) >= 5:
            mean = sum(daily_returns) / len(daily_returns)
            var = sum((r - mean) ** 2 for r in daily_returns) / len(daily_returns)
            std = var ** 0.5
            sharpe = round((mean / std) * (252 ** 0.5), 2) if std > 0 else None

        # ─── 4. Win rate (round-trips per distinct position) ───
        # A round-trip closes when held qty returns to ≈ 0. P/L = proceeds - cost.
        # Round-trip grouping is delegated to analytics.round_trips so this
        # endpoint and any future trade-attribution caller share one
        # implementation instead of drifting hand-maintained copies.
        # build_round_trips keys by (ticker, type, strike, expiry) — stock and
        # option legs of the same ticker stay distinct. pnl_usd is rounded to
        # 4dp there; the win/loss split below uses strict `> 0`, so a sub-cent
        # rounding artefact reads as a non-win (pinned by test_round_trips).
        from .analytics.drawdown import compute_drawdown
        from .analytics.etf_lookthrough import build_etf_lookthrough
        from .analytics.pnl_attribution import build_pnl_attribution
        from .analytics.recovery import build_recovery
        from .analytics.round_trips import build_round_trips
        from .analytics.stress_scenarios import build_stress_scenarios
        from .analytics.tail_risk import build_tail_risk
        _rts = build_round_trips(trades)
        round_trips: list[float] = [rt["pnl_usd"] for rt in _rts]
        holding_days: list[float] = [
            rt["hold_days"] for rt in _rts if rt["hold_days"] is not None
        ]  # one entry per closed round-trip with a parseable entry/exit ts

        wins = [p for p in round_trips if p > 0]
        losses = [p for p in round_trips if p <= 0]
        win_rate = round(len(wins) / len(round_trips) * 100, 2) if round_trips else None
        avg_winner = round(sum(wins) / len(wins), 2) if wins else None
        avg_loser = round(sum(losses) / len(losses), 2) if losses else None
        total_realized = round(sum(round_trips), 2) if round_trips else 0.0

        # ─── 4b. Profit factor + avg holding period ───
        # Profit factor = gross wins / gross losses. >1 means the edge survives
        # losers; a 50% win rate with PF 2.0 is a real edge, PF 0.8 is bleeding.
        gross_win = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = round(gross_win / gross_loss, 2) if gross_loss > 1e-9 else None
        avg_hold_days = (round(sum(holding_days) / len(holding_days), 2)
                         if holding_days else None)

        # ─── 4c. Sortino — like Sharpe but only downside vol is "risk" ───
        sortino = None
        if len(daily_returns) >= 5:
            dmean = sum(daily_returns) / len(daily_returns)
            downside = [r for r in daily_returns if r < 0]
            if downside:
                dvar = sum(r * r for r in downside) / len(daily_returns)
                dstd = dvar ** 0.5
                if dstd > 0:
                    sortino = round((dmean / dstd) * (252 ** 0.5), 2)

        # ─── 4d. S&P 500 beta + correlation (paired daily returns) ───
        sp_by_day: dict[str, float] = {}
        for p in eq:
            day = (p["timestamp"] or "")[:10]
            spx = p.get("sp500_price")
            if day and spx:
                sp_by_day[day] = spx
        port_ret: list[float] = []
        spx_ret: list[float] = []
        for i in range(1, len(day_keys)):
            d0, d1 = day_keys[i - 1], day_keys[i]
            if d0 in sp_by_day and d1 in sp_by_day:
                pv0, sv0 = by_day[d0], sp_by_day[d0]
                pv1, sv1 = by_day[d1], sp_by_day[d1]
                if pv0 > 0 and sv0 > 0:
                    port_ret.append(pv1 / pv0 - 1.0)
                    spx_ret.append(sv1 / sv0 - 1.0)
        sp500_beta = None
        sp500_corr = None
        if len(port_ret) >= 5:
            n = len(port_ret)
            mp = sum(port_ret) / n
            ms = sum(spx_ret) / n
            cov = sum((port_ret[i] - mp) * (spx_ret[i] - ms) for i in range(n)) / n
            var_s = sum((s - ms) ** 2 for s in spx_ret) / n
            var_p = sum((p - mp) ** 2 for p in port_ret) / n
            if var_s > 0:
                sp500_beta = round(cov / var_s, 2)
                if var_p > 0:
                    sp500_corr = round(cov / ((var_s ** 0.5) * (var_p ** 0.5)), 3)

        # ─── 4e. Calmar — annualized return ÷ max drawdown ───
        # Meaningless on <20 trading days of history, so gate it hard.
        calmar = None
        if len(daily_returns) >= 20 and max_dd_pct and max_dd_pct > 0:
            # Baseline must come from the store constant, not a hardcoded
            # 1000.0 — a literal here silently desyncs Calmar if INITIAL_CASH
            # ever moves (same desync class fixed in reporter.py, commit 2a154df).
            total_return_pct = (total_value / INITIAL_CASH - 1.0) * 100.0
            years = len(day_keys) / 252.0
            if years > 0:
                calmar = round((total_return_pct / years) / max_dd_pct, 2)

        # ─── 5. Daily P/L (today only, UTC bucket) ───
        today = datetime.now(timezone.utc).date().isoformat()
        today_eq = [p for p in eq if (p["timestamp"] or "").startswith(today)]
        daily_pl = None
        daily_pl_pct = None
        if today_eq:
            open_val = today_eq[0]["total_value"]
            cur_val = total_value
            if open_val:
                daily_pl = round(cur_val - open_val, 2)
                daily_pl_pct = round(daily_pl / open_val * 100, 2)

        # Compute the tail-risk + drawdown SSOT objects ONCE so the
        # tail_risk and recovery folds below share the exact same inputs
        # /api/recovery does (no double-compute, no cross-fold drift).
        _tr = build_tail_risk(eq)
        _dd = compute_drawdown(eq, positions, starting_equity=INITIAL_CASH)

        payload = {
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "total_value": round(total_value, 2),
            "cash_pct": cash_pct,
            "sector_exposure_pct": sector_pct,
            "sector_exposure_usd": {s: round(v, 2) for s, v in sector_usd.items()},
            "max_drawdown_usd": round(max_dd_usd, 2) if max_dd_usd is not None else None,
            "max_drawdown_pct": round(max_dd_pct, 2) if max_dd_pct is not None else None,
            "sharpe_annualized": sharpe,
            "n_trading_days": len(daily_returns),
            "n_round_trips": len(round_trips),
            "win_rate_pct": win_rate,
            "avg_winner_usd": avg_winner,
            "avg_loser_usd": avg_loser,
            "realized_pl_usd": total_realized,
            "profit_factor": profit_factor,
            "avg_holding_days": avg_hold_days,
            "sortino_annualized": sortino,
            "calmar_ratio": calmar,
            "sp500_beta": sp500_beta,
            "sp500_correlation": sp500_corr,
            "daily_pl_usd": daily_pl,
            "daily_pl_pct": daily_pl_pct,
            # Left-tail / downside-shape diagnostic. Additive top-level
            # key — test_core_analytics uses keyed assertions, not
            # whole-dict equality — so the digital-intern analyst chat
            # (which fetches /api/analytics) inherits VaR/CVaR/skew for
            # free. eq is the same day-resampled series used above.
            "tail_risk": _tr,
            # Forward "path back to even": per-position breakeven + the book
            # rally to the $1000 start / high-water peak, scaled by THIS
            # book's own realized daily vol. Additive top-level key (keyed
            # asserts, not whole-dict equality) so the digital-intern analyst
            # chat that fetches /api/analytics inherits it for free. Composed
            # from the SAME _dd/_tr objects the endpoint uses — no drift
            # (AGENTS.md #10): the recovery fold and /api/recovery cannot
            # disagree because both consume one compute_drawdown + one
            # build_tail_risk per request.
            "recovery": build_recovery(_dd, _tr, INITIAL_CASH),
            # Forward beta/concentration shock — the day-one complement to
            # tail_risk above (which reads INSUFFICIENT on a young book).
            # Additive top-level key (test_stress_scenarios uses keyed
            # asserts, not whole-dict equality) so the digital-intern analyst
            # chat that fetches /api/analytics inherits the $-at-risk view
            # for free. Same _classify/_LEVERAGE_BETA SSOT as /api/risk.
            "stress_scenarios": build_stress_scenarios(
                positions, float(total_value or 0.0),
                _classify, _LEVERAGE_BETA,
            ),
            # β-adjusted unrealized P/L decomposition. ``open_attribution``
            # implicitly assumes β=1 and so over-attributes "alpha" on a
            # leveraged-ETF/semis book (TQQQ β=3, NVDA β=1.5). This fold
            # answers "is my gain just SPY ×β?" using the SAME
            # ``_classify``/``_LEVERAGE_BETA`` SSOT as
            # ``stress_scenarios``/``/api/risk``/``/api/pnl-attribution`` —
            # additive top-level key (digital-intern analyst chat inherits
            # it for free) so cross-fold drift fails the no-drift test.
            "pnl_attribution": build_pnl_attribution(
                positions, eq, _classify, _LEVERAGE_BETA,
            ),
            # ETF look-through: pierce leveraged-ETF positions into effective
            # single-name exposure. Every other risk surface stops at the
            # ticker boundary — sector_exposure classifies TQQQ as
            # ``broad_lev``, risk_mirror reports HHI on line-item tickers,
            # neither sees that a TQQQ position silently amplifies NVDA
            # exposure. Additive top-level key (keyed asserts, never
            # whole-dict equality) so the digital-intern analyst chat that
            # fetches /api/analytics inherits the "hidden concentration"
            # verdict for free — the tail_risk/stress_scenarios/recovery
            # additive-key precedent. Snapshot built from the same
            # positions/total_value already on hand so the fold cannot
            # drift from /api/etf-lookthrough.
            "etf_lookthrough": build_etf_lookthrough({
                "cash": float(pf.get("cash") or 0.0),
                "total_value": float(total_value or 0.0),
                "positions": positions,
            }),
        }
        # Same single-source-of-truth honesty fold as /api/tail-risk &
        # /api/drawdown — mark_integrity's docstring names THIS endpoint's
        # Sharpe as the first victim of a stale book. Additive, _safe
        # (omitted on fault so the payload is byte-identical), AGENTS.md #10.
        mt = _mark_trust_block(store)
        if mt is not None:
            payload["mark_trust"] = mt
        # Per-held-ticker source-diversity verdict — orthogonal to
        # /api/news-velocity (which measures rate). A SURGING z-score is
        # identical whether five outlets are reporting or one wire is
        # mirrored across five feeds; ``ECHO`` is the false-signal case
        # velocity cannot see. Composes the held set from the same
        # ``positions`` already on hand. Additive top-level key (keyed
        # asserts, never whole-dict equality) so the digital-intern
        # analyst chat that fetches /api/analytics inherits the breadth
        # verdict for free — the tail_risk/stress_scenarios/recovery
        # additive-key precedent. Wrapped in try/except: a fault drops
        # ONLY this key, never sinks /api/analytics (the _safe contract).
        try:
            from .analytics.news_source_mix import build_news_source_mix
            held = _stock_tickers_from_positions(positions)
            if held:
                path = _articles_db_path()
                if path is not None:
                    now_utc = datetime.now(timezone.utc)
                    since = (now_utc - timedelta(hours=24.0)).isoformat()
                    conn = sqlite3.connect(
                        f"file:{path}?mode=ro", uri=True, timeout=3)
                    conn.row_factory = sqlite3.Row
                    try:
                        like_clauses = " OR ".join(
                            ["title LIKE ?"] * len(held))
                        like_params = [f"%{t}%" for t in held]
                        rows = conn.execute(
                            f"SELECT title, source, first_seen "
                            f"FROM articles WHERE first_seen >= ? "
                            f"AND ({like_clauses}) "
                            f"AND url NOT LIKE 'backtest://%' "
                            f"AND source NOT LIKE 'backtest_%' "
                            f"AND source NOT LIKE 'opus_annotation%' "
                            f"ORDER BY first_seen DESC LIMIT 5000",
                            [since] + like_params,
                        ).fetchall()
                    finally:
                        conn.close()
                    articles = [
                        {"title": r["title"] or "",
                         "source": r["source"] or "",
                         "first_seen": r["first_seen"]}
                        for r in rows
                    ]
                    payload["news_source_mix"] = build_news_source_mix(
                        articles, held, now=now_utc, window_hours=24.0,
                    )
        except Exception as _e:
            # Swallow — additive key only, never sinks /api/analytics.
            pass
        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _mark_trust_block(store):
    """Single source of truth (AGENTS.md #10): the existing
    ``build_mark_integrity`` verdict, computed from the SAME write-free
    ``strategy.portfolio_snapshot_readonly`` snapshot ``/api/mark-integrity``
    uses, folded into the equity-derived risk endpoints (``/api/tail-risk``,
    ``/api/drawdown``) as an additive ``mark_trust`` key.

    Rationale: ``mark_integrity``'s own docstring names these surfaces as
    silently fictional when the book is stale-marked ("/api/analytics
    Sharpe, /api/drawdown, the equity curve ... all quietly partially false,
    with nothing saying so"), yet a grep shows ``stale_mark`` only ever
    reached mark_integrity/strategy/dashboard/reporter — never the tail-risk
    or drawdown maths. A stale cycle records a cost-frozen flat equity point;
    those flats deflate vol & drawdown, inflate Sharpe, and truncate the VaR
    tail. This wires the canonical trust verdict in so the numbers self-flag.

    Additive, observational only — never gates Opus, adds no caps, not
    injected into the decision prompt, no schema change (AGENTS.md #2/#12).
    Composes ``build_mark_integrity`` verbatim — no re-derived staleness.
    ``_safe``: any fault → ``None`` so the caller omits the key; the
    endpoint's pre-existing risk payload/behaviour is byte-identical and it
    never 500s for this reason (the behavioural-builder failure contract)."""
    try:
        from .analytics.mark_integrity import build_mark_integrity
        from .strategy import portfolio_snapshot_readonly
        snap = portfolio_snapshot_readonly(store)
        mi = build_mark_integrity(snap.get("positions") or [])
        verdict = mi.get("verdict")
        note = None
        if verdict not in (None, "CLEAN", "NO_DATA"):
            note = (
                "Risk metrics here are derived from the equity curve; the "
                f"current book is {mi.get('stale_value_pct')}% marked at "
                "cost (stale price feed), so the most recent equity points "
                "— and the live tail of these metrics — are at cost-basis, "
                "not true value. Treat the figures as a floor on risk until "
                "the feed recovers / the runner restarts."
            )
        return {
            "verdict": verdict,
            "n_positions": mi.get("n_positions"),
            "n_stale": mi.get("n_stale"),
            "stale_value_pct": mi.get("stale_value_pct"),
            "stale_tickers": mi.get("stale_tickers"),
            "headline": mi.get("headline"),
            "note": note,
        }
    except Exception:
        return None


@app.route("/api/tail-risk")
def tail_risk_api():
    """Historical 95/99% 1-day VaR, expected shortfall, annualised vol &
    downside deviation, return skew, worst day, max consecutive down-day
    streak and Ulcer index — the left-tail view the upside-heavy
    analytics surface was missing. Honesty-gated
    (NO_DATA/INSUFFICIENT/OK, mirrors build_correlation); observational
    only — never gates Opus, never injected into the decision prompt
    (AGENTS.md #2/#12)."""
    try:
        from .analytics.tail_risk import build_tail_risk
        store = get_store()
        result = build_tail_risk(store.equity_curve(5000))
        mt = _mark_trust_block(store)
        if mt is not None:
            result["mark_trust"] = mt
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/recovery")
def recovery_api():
    """Path back to even: per-position breakeven %/$ + the book rally
    required to return to the $1000 start and the running high-water peak,
    scaled by this book's own realized daily volatility.

    The *forward* complement to /api/drawdown (which owns the *backward*
    "% of trough clawed back"). Composes ``compute_drawdown`` (current/peak
    + per-lot P&L SSOT) and ``build_tail_risk`` (realized-vol SSOT) verbatim
    — a drift fails the no-drift test. The σ figure is a dispersion scale,
    not a time forecast, and is withheld until tail_risk reads OK (the
    young-book honesty precedent). Pure & DB-only — no network (the
    ``drawdown``/``tail_risk`` builder/endpoint split); ``initial_equity``
    is the module ``INITIAL_CASH`` (invariant #12, never a literal 1000).
    Observational only — never gates Opus, adds no caps (AGENTS.md #2/#12)."""
    try:
        from .analytics.drawdown import compute_drawdown
        from .analytics.recovery import build_recovery
        from .analytics.tail_risk import build_tail_risk
        store = get_store()
        eq = store.equity_curve(limit=2000)
        positions = store.open_positions()
        dd = compute_drawdown(eq, positions, starting_equity=INITIAL_CASH)
        tr = build_tail_risk(eq)
        result = build_recovery(dd, tr, INITIAL_CASH)
        mt = _mark_trust_block(store)
        if mt is not None:
            result["mark_trust"] = mt
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/etf-lookthrough")
@swr_cached("etf-lookthrough", 30.0)
def etf_lookthrough_api():
    """ETF look-through into effective single-name exposure.

    Pierces leveraged-ETF positions (TQQQ=3x QQQ, SOXL=3x SOXX, FNGU=3x
    FANG+, …) into virtual single-name exposures so a book reading "44%
    NVDA, 22% TQQQ" doesn't silently understate its TRUE NVDA bet. Every
    existing risk surface stops at the ticker boundary; this one piercs
    through. Inverse ETFs (SQQQ/SOXS/SPXS/FNGD/TECS) honestly subtract.

    Observational only — never gates Opus, adds no caps (AGENTS.md #2/#12).
    """
    try:
        from .analytics.etf_lookthrough import build_etf_lookthrough
        store = get_store()
        pf = store.get_portfolio()
        snapshot = {
            "cash": float(pf.get("cash") or 0.0),
            "total_value": float(pf.get("total_value") or 0.0),
            "positions": store.open_positions(),
        }
        result = build_etf_lookthrough(snapshot)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stress-scenarios")
@swr_cached("stress_scenarios", 30.0)
def stress_scenarios_api():
    """Forward beta/concentration shock estimate — the day-one complement to
    /api/tail-risk (which reads INSUFFICIENT until the book has ≥20 daily
    returns). Pure ``Σ weight×beta×shock`` over the CURRENT marked book, so
    it produces a real dollar figure with no return history. The −3 % market
    line is byte-identical to /api/risk's ``shock_usd`` (single source of
    truth, AGENTS.md #10 — both use ``_classify`` + ``_LEVERAGE_BETA``).
    Observational only — never gates Opus, adds no caps (AGENTS.md #2/#12)."""
    try:
        from .analytics.stress_scenarios import build_stress_scenarios
        store = get_store()
        pf = store.get_portfolio()
        result = build_stress_scenarios(
            store.open_positions(),
            float(pf.get("total_value") or 0.0),
            _classify,
            _LEVERAGE_BETA,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/position-blowup")
@swr_cached("position_blowup", 30.0)
def position_blowup_api():
    """Per-position single-name shock ladder — the per-name complement to
    /api/stress-scenarios (which shocks only the *largest* name at −10 %) and
    /api/risk (whose ``concentration_top1`` flags the weight but not the
    dollar damage). Every held name is shocked individually at −10/−25/−50/
    −100 %, sorted worst-first, so a concentrated book's dominant tail —
    a single-name surprise (downgrade, lawsuit, accounting blow-up) the
    SPY-shock and earnings-σ models never capture — is visible at a glance.
    Pure ``weight×shock`` arithmetic over the CURRENT marked book; reuses
    ``build_position_blowup`` verbatim (SSOT, AGENTS.md #10 — the panel and
    the ``-m paper_trader.analytics.position_blowup`` CLI can never disagree).
    Observational only — never gates Opus, adds no caps (AGENTS.md #2/#12)."""
    try:
        from .analytics.position_blowup import build_position_blowup
        store = get_store()
        pf = store.get_portfolio()
        result = build_position_blowup(
            store.open_positions(),
            float(pf.get("total_value") or 0.0),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/profit-ladder")
def profit_ladder_api():
    """Per-position UPSIDE shock ladder + trim-yield schedule — the
    symmetric complement to ``/api/position-blowup``.

    Every held name is shocked at +5/+10/+25/+50/+100 % and each rung
    carries a trim_schedule with realized-vs-unrealized split at 25 /
    50 / 100 % trim fractions. Answers the trader's "if I'm right, what
    does it pay me — and what does trimming HALF at +25 % lock in?"
    question that ``position_blowup`` (downside only), ``recovery``
    (book-level path back to even), ``cost_basis_ladder`` (lots at the
    CURRENT mark), and ``trim_simulator`` (trims at the CURRENT mark,
    never at a future rung) all leave unanswered.

    Aggregate verdict: NO_DATA / RECOVERY_BOOK / MIXED_BOOK /
    IN_PROFIT / BIG_WINNERS. Pure arithmetic over the currently-marked
    book; reuses ``build_profit_ladder`` verbatim (SSOT, AGENTS.md #10
    — the panel and the ``-m paper_trader.analytics.profit_ladder``
    CLI can never disagree). Observational only — never gates Opus,
    adds no caps (AGENTS.md #2/#12)."""
    try:
        from .analytics.profit_ladder import build_profit_ladder
        store = get_store()
        pf = store.get_portfolio()
        result = build_profit_ladder(
            store.open_positions(),
            float(pf.get("total_value") or 0.0),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/trim-simulator")
@swr_cached("trim_simulator", 45.0)
def trim_simulator_api():
    """Per-position trim ladder with scorer-EV math — the "how much should I
    sell now" card. Builds on the (existing) ``/api/suggestion-impact``
    single-rung 50% projection by adding three rungs (25/50/75% of qty by
    default) PER held name and layering the DecisionScorer 5-day forward-
    return prediction onto each rung as ``ev_avoided_loss_usd`` /
    ``ev_forgone_upside_usd``. Surfaces the per-position verdict
    (RECOMMEND_EXIT / RECOMMEND_TRIM / NEUTRAL / HOLD) synthesising scorer
    pred sign+magnitude × current weight, and a recommended rung. Reuses
    ``build_trim_simulator`` verbatim (SSOT, AGENTS.md #10 — the panel and
    the ``-m paper_trader.analytics.trim_simulator`` CLI can never disagree).
    Observational only — never gates Opus, adds no caps (AGENTS.md #2/#12).

    Scorer predictions are sourced via the same ``/api/scorer-predictions``
    in-process handler so the panel and the scorer card can never disagree;
    if it fails or returns no preds the ladder still renders (per-position
    scorer fields become ``None``, verdict falls back to weight-only triage).
    """
    try:
        from .analytics.trim_simulator import build_trim_simulator
        store = get_store()
        pf = store.get_portfolio()
        # Pull scorer predictions through the existing in-process handler.
        # Calling the wrapper directly hits its own @swr_cached 60s, so the
        # cost here is sub-millisecond after the first warm.
        preds: list[dict] = []
        try:
            sp_resp = scorer_predictions_api()
            sp_data = sp_resp.get_json() if hasattr(sp_resp, "get_json") else {}
            if isinstance(sp_data, dict) and not sp_data.get("error"):
                preds = list(sp_data.get("predictions") or [])
        except Exception:
            preds = []
        result = build_trim_simulator(
            store.open_positions(),
            float(pf.get("total_value") or 0.0),
            preds,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/concentration-cap")
@swr_cached("concentration_cap", 30.0)
def concentration_cap_api():
    """Per-name concentration-cap rebalance recommender.

    Given the current book + a configurable per-name cap (default 25.0%;
    override via ``?cap_pct=N``, clamped [1, 100]), compute the exact
    ``shares_to_trim`` and ``cash_freed_usd`` for each over-cap name to land
    at the cap. Returns baseline + projected top1/top3 so the operator sees
    whether one trim cycle is enough or another name is queued to climb past
    the cap. Pure / never-raises. Reuses ``build_concentration_cap`` verbatim
    (SSOT, AGENTS.md #10). Observational only — never gates Opus, adds no
    caps (AGENTS.md #2/#12).

    Complementary to ``/api/trim-simulator``: that one is the *scorer-EV-driven*
    ladder; this one is the *mechanical* per-name cap math.
    """
    try:
        from .analytics.concentration_cap import (
            DEFAULT_CAP_PCT, build_concentration_cap,
        )
        try:
            cap = float(request.args.get("cap_pct", DEFAULT_CAP_PCT))
        except (TypeError, ValueError):
            cap = DEFAULT_CAP_PCT
        store = get_store()
        pf = store.get_portfolio()
        result = build_concentration_cap(
            store.open_positions(),
            float(pf.get("total_value") or 0.0),
            cap,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _articles_db_path() -> Path | None:
    """Resolve the digital-intern articles.db through the SAME freshness-aware
    single source of truth the live trader uses (``signals._db_path()``).

    The legacy USB-first existence probe this replaced returned the USB mirror
    whenever it merely ``exists()`` — but the digital-intern daemon falls back
    to writing the LOCAL copy when the USB mount is unavailable, leaving a USB
    mirror that keeps ``exists()``-ing while going day-stale. That made every
    news-analytics endpoint here (``/api/news-edge``, ``/api/source-edge``,
    ``/api/signal-followthrough``, ``/api/sector-pulse``, ``/api/thesis-drift``)
    read a STALE feed while the live trader (``signals._db_path()``,
    freshness-aware) read the FRESH one — the documented split-brain, fixed
    everywhere in signals.py but left here. Delegating to ``signals._db_path()``
    closes that gap so the dashboard and the trader never disagree on which
    feed is canonical.

    Caller contract preserved: callers do ``if path is None: <graceful>``, so a
    resolved-but-nonexistent DB still surfaces as ``None`` rather than a Path to
    a missing file (``signals._db_path()`` returns LOCAL_DB as its tie/fallback
    even when nothing exists)."""
    try:
        from . import signals as _sig
        path = _sig._db_path()
    except Exception:
        return None
    return path if path and path.exists() else None


def _ticker_news_pulse(tickers: list[str], hours: int = 24) -> dict[str, dict]:
    """For each ticker, count + top headline of articles mentioning it.

    Reads the articles DB in read-only mode. Live-only filter is applied so
    backtest/opus_annotation synthetic rows are excluded.
    """
    out: dict[str, dict] = {t.upper(): {
        "n": 0, "urgent": 0, "top_title": None, "top_url": None, "top_score": 0.0,
    } for t in tickers}
    path = _articles_db_path()
    if path is None:
        return out
    conn = None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
        conn.row_factory = sqlite3.Row
        since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        rows = conn.execute(
            "SELECT title, url, full_text, ai_score, urgency FROM articles "
            "WHERE first_seen >= ? AND ai_score > 0 "
            "AND url NOT LIKE 'backtest://%' "
            "AND source NOT LIKE 'backtest_%' "
            "AND source NOT LIKE 'opus_annotation%' "
            "ORDER BY ai_score DESC LIMIT 2000",
            (since,),
        ).fetchall()
    except Exception:
        return out
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    patterns = {t.upper(): re.compile(rf"(?:\$|\b){re.escape(t.upper())}\b") for t in tickers}
    for r in rows:
        body = r["title"] or ""
        if r["full_text"]:
            try:
                body = body + " " + zlib.decompress(r["full_text"]).decode("utf-8", "replace")
            except Exception:
                pass
        body_up = body.upper()
        for t, pat in patterns.items():
            if pat.search(body_up):
                rec = out[t]
                rec["n"] += 1
                if (r["urgency"] or 0) >= 1:
                    rec["urgent"] += 1
                if (r["ai_score"] or 0) > rec["top_score"]:
                    rec["top_score"] = r["ai_score"]
                    rec["top_title"] = r["title"]
                    rec["top_url"] = r["url"]
    return out


@app.route("/api/sector-pulse")
def sector_pulse_api():
    """Compact semis-sector card: price, day %, RSI, news count, top headline per ticker."""
    try:
        from . import market
        from .strategy import _QUANT_CACHE, get_quant_signals_live

        tickers = SECTOR_PULSE_TICKERS
        # Warm the quant cache only for tickers we don't already have fresh data for.
        # get_quant_signals_live respects its own 5-min TTL.
        try:
            get_quant_signals_live(tickers)
        except Exception:
            pass

        prices = market.get_prices(tickers)
        news = _ticker_news_pulse(tickers, hours=24)

        out = []
        for t in tickers:
            cached = _QUANT_CACHE.get(t)
            quant = cached[0] if cached else {}
            # Compute today's % change from quant signals' 1y history if we cached it.
            rsi = quant.get("RSI")
            mom_5d = quant.get("mom_5d")
            mom_20d = quant.get("mom_20d")
            macd = quant.get("macd_signal")
            vol_ratio = quant.get("vol_ratio")
            pct_from_52h = quant.get("pct_from_52h")
            nrec = news.get(t.upper(), {})
            out.append({
                "ticker": t,
                "price": prices.get(t),
                "rsi": rsi,
                "macd": macd,
                "mom_5d": mom_5d,
                "mom_20d": mom_20d,
                "vol_ratio": vol_ratio,
                "pct_from_52h": pct_from_52h,
                "news_count_24h": nrec.get("n", 0),
                "news_urgent_24h": nrec.get("urgent", 0),
                "top_headline": nrec.get("top_title"),
                "top_url": nrec.get("top_url"),
                "top_score": nrec.get("top_score") or 0.0,
            })
        return jsonify({
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tickers": out,
        })
    except Exception as e:
        return jsonify({"tickers": [], "error": str(e)}), 500


# ───────────────────────── Feature-dev additions (2026-05-14) ─────────────────────────
# Three additive endpoints + supporting helpers:
#   /api/suggestions  — co-pilot trade ideas from news × positions × quant signals
#   /api/risk         — concentration / leveraged-exposure / position-age / shock estimate
#   /api/briefing     — futures + market-open countdown + top urgent news
# All routes degrade gracefully — yfinance / signals / strategy imports are lazy and
# wrapped so a missing dependency returns a structured error instead of 500.

# Leverage factors for the SPY-shock dollar-at-risk estimate. Conservative single
# beta numbers chosen to be obviously approximate — this is decision support, not VaR.
_LEVERAGE_BETA = {
    "broad": 1.0,
    "broad_lev": 3.0,       # Most broad-leveraged are 3x; QLD/SSO are 2x but in the same bucket here
    "tech": 1.2,
    "tech_lev": 3.0,
    "crypto_lev": 2.5,
    "semis": 1.5,
    "semis_lev": 3.0,
    "optical": 1.4,
    "bio_lev": 3.0,
    "health_lev": 3.0,
    "fin_lev": 3.0,
    "housing_lev": 3.0,
    "util_lev": 3.0,
    "defense_lev": 3.0,
    "other": 1.0,
}

_LEVERAGED_SECTORS = {s for s in _LEVERAGE_BETA if s.endswith("_lev")}


def _position_ages_from_trades(open_positions: list[dict], trades_oldest_first: list[dict]) -> dict[str, int]:
    """For each currently-open ticker, return days since the earliest BUY in the
    most recent open lot. Walks trades chronologically and resets the open-lot
    timestamp every time the running quantity returns to ≈0."""
    open_tickers = {p["ticker"] for p in open_positions if p.get("type") == "stock"}
    earliest: dict[str, str] = {}
    held: dict[str, float] = {}
    for t in trades_oldest_first:
        tk = t.get("ticker")
        if tk not in open_tickers:
            continue
        act = (t.get("action") or "").upper()
        # Only stock trades affect stock-position age. BUY_CALL / SELL_PUT etc.
        # would otherwise corrupt the running stock quantity for this ticker.
        if act not in ("BUY", "SELL"):
            continue
        qty = float(t.get("qty") or 0)
        ts = t.get("timestamp") or ""
        if act == "BUY":
            if held.get(tk, 0.0) < 1e-6 or tk not in earliest:
                earliest[tk] = ts
            held[tk] = held.get(tk, 0.0) + qty
        else:  # SELL
            held[tk] = held.get(tk, 0.0) - qty
            if abs(held.get(tk, 0.0)) < 1e-6:
                earliest.pop(tk, None)
    now = datetime.now(timezone.utc)
    ages: dict[str, int] = {}
    for tk, ts in earliest.items():
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            ages[tk] = max(0, (now - dt).days)
        except Exception:
            continue
    return ages


def _concentration_severity(top1_pct: float, top3_pct: float) -> tuple[str, bool]:
    """Bucket a portfolio's concentration into a severity label + boolean flag.

    HIGH triggers a UI alert and indicates dangerous over-concentration.
    Thresholds are deliberately strict — this is a $1000 paper book whose
    edge is breadth + speed, not single-name conviction."""
    if top1_pct >= 60 or top3_pct >= 90:
        return "HIGH", True
    if top1_pct >= 40 or top3_pct >= 75:
        return "MEDIUM", True
    return "LOW", False


@app.route("/api/risk")
@swr_cached("risk", 30.0)
def risk_api():
    """Risk-focused portfolio panel. Fields are intentionally disjoint from
    /api/analytics: concentration, leveraged exposure, position age, stale flags,
    SPY-shock dollar-at-risk estimate. Pair with /api/analytics for full picture."""
    try:
        store = get_store()
        pf = store.get_portfolio()
        positions = store.open_positions()
        total_value = float(pf.get("total_value") or 0.0)
        cash = float(pf.get("cash") or 0.0)

        # ── Per-position market values + sector classification ──
        rows = []
        leveraged_usd = 0.0
        shock_usd = 0.0  # estimated $ change if SPY drops 3%
        for p in positions:
            mult = 100 if p["type"] in ("call", "put") else 1
            price = p.get("current_price") or p.get("avg_cost") or 0.0
            qty = float(p.get("qty") or 0)
            val = price * qty * mult
            sec = _classify(p["ticker"])
            beta = _LEVERAGE_BETA.get(sec, 1.0)
            # Options inherit underlying sector beta but with a rough 3x payoff
            # multiplier for at-the-money ITM exposure; cap at 4.
            if p["type"] in ("call", "put"):
                beta = min(beta * 3.0, 4.0)
                if p["type"] == "put":
                    beta = -beta  # puts profit on a drop
            shock_usd += -0.03 * beta * val  # negative = loss on -3% SPY
            if sec in _LEVERAGED_SECTORS:
                leveraged_usd += val
            rows.append({
                "ticker": p["ticker"],
                "type": p["type"],
                "sector": sec,
                "market_value": round(val, 2),
                "pct_port": round((val / total_value * 100) if total_value else 0.0, 2),
                "beta_est": round(beta, 2),
            })

        rows.sort(key=lambda r: -r["market_value"])
        largest = rows[0] if rows else None
        top3_pct = round(sum(r["pct_port"] for r in rows[:3]), 2)
        top1_pct = round(largest["pct_port"], 2) if largest else 0.0
        conc_severity, conc_warning = _concentration_severity(top1_pct, top3_pct)

        # ── Position ages from trade history ──
        trades_oldest_first = list(reversed(store.recent_trades(2000)))
        ages = _position_ages_from_trades(positions, trades_oldest_first)

        # ── Stale flag: held > 7d, |P/L| < 2% — likely sitting on dead money ──
        # store.open_positions() rows have current_price/avg_cost but no pl_pct,
        # so derive it here rather than reading a key that's always missing.
        stale = []
        for p in positions:
            tk = p["ticker"]
            avg = float(p.get("avg_cost") or 0.0)
            cur = float(p.get("current_price") or 0.0) or avg
            pl_pct_signed = ((cur - avg) / avg * 100) if avg else 0.0
            age = ages.get(tk)
            if age is not None and age >= 7 and abs(pl_pct_signed) < 2.0:
                stale.append({
                    "ticker": tk,
                    "age_days": age,
                    "pl_pct": round(pl_pct_signed, 2),
                    "market_value": round(
                        cur * float(p.get("qty") or 0)
                        * (100 if p["type"] in ("call", "put") else 1),
                        2,
                    ),
                })

        ages_list = sorted(ages.values()) if ages else []
        if ages_list:
            mid = len(ages_list) // 2
            if len(ages_list) % 2:
                median_age = ages_list[mid]
            else:
                median_age = round((ages_list[mid - 1] + ages_list[mid]) / 2)
        else:
            median_age = None

        return jsonify({
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "total_value": round(total_value, 2),
            "cash_usd": round(cash, 2),
            "cash_pct": round((cash / total_value * 100) if total_value else 0.0, 2),
            "n_positions": len(positions),
            "concentration_top1_pct": top1_pct,
            "concentration_top1_ticker": largest["ticker"] if largest else None,
            "concentration_top3_pct": top3_pct,
            "concentration_warning": conc_warning,
            "concentration_severity": conc_severity,
            "leveraged_usd": round(leveraged_usd, 2),
            "leveraged_pct": round((leveraged_usd / total_value * 100) if total_value else 0.0, 2),
            "spy_shock_3pct_usd": round(shock_usd, 2),  # negative = loss
            "spy_shock_3pct_pct": round((shock_usd / total_value * 100) if total_value else 0.0, 2),
            "median_age_days": median_age,
            "max_age_days": max(ages.values()) if ages else None,
            "position_ages": ages,
            "stale_positions": stale,
            "positions_by_value": rows,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _next_market_open() -> tuple[datetime | None, int | None]:
    """Return (next_open_dt_utc, seconds_until). If market is open right now,
    returns the next close instead with a sign convention noted by the caller.
    Uses paper_trader.market constants — keeps the NYSE holiday calendar in one place."""
    try:
        from . import market as _mkt
    except Exception:
        return None, None
    now_utc = datetime.now(timezone.utc)
    now_ny = now_utc.astimezone(_mkt.NY)
    open_min = 9 * 60 + 30
    cur_min = now_ny.hour * 60 + now_ny.minute
    # If currently open, return next close. Use market.close_minute so a NYSE
    # early-close half-day (13:00 ET — the day after Thanksgiving / Christmas
    # Eve, enforced in market.is_market_open since b6a1934) reports the real
    # close, not a hardcoded 16:00. A stale 16:00 here made the briefing card
    # ("Market OPEN — closes in 5h00m") and /api/game-plan's next_open_seconds
    # 3h wrong on those two sessions — the exact thing a trader times exits on.
    if _mkt.is_market_open(now_utc):
        close_min = _mkt.close_minute(now_ny.date())
        ch, cm = divmod(close_min, 60)
        close_dt = now_ny.replace(hour=ch, minute=cm, second=0, microsecond=0)
        return close_dt.astimezone(timezone.utc), int((close_dt - now_ny).total_seconds())
    # Walk forward day-by-day to find the next open day. The outer guard
    # `(not is_today or cur_min < open_min)` already excludes "today, past
    # market open" — by the time we'd consider returning today, we must be
    # before 9:30 AM NY, so no past-close edge case to handle.
    from datetime import timedelta as _td
    candidate = now_ny
    for _ in range(10):
        is_weekday = candidate.weekday() < 5
        is_holiday = candidate.date() in _mkt.NYSE_HOLIDAYS_2026
        is_today = candidate.date() == now_ny.date()
        if is_weekday and not is_holiday and (not is_today or cur_min < open_min):
            open_dt = candidate.replace(hour=9, minute=30, second=0, microsecond=0)
            return open_dt.astimezone(timezone.utc), int((open_dt - now_ny).total_seconds())
        candidate = candidate + _td(days=1)
        candidate = candidate.replace(hour=0, minute=0, second=0, microsecond=0)
    return None, None


@app.route("/api/briefing")
@swr_cached("briefing", 90.0)
def briefing_api():
    """Pre-market / live briefing card. Combines market-open status, futures,
    top urgent overnight news, and a one-line summary string. Designed to be the
    first thing the user sees on the trader pane each morning."""
    try:
        from . import market as _mkt
        from . import signals as _sig

        now_utc = datetime.now(timezone.utc)
        is_open = _mkt.is_market_open(now_utc)
        next_dt, secs = _next_market_open()

        # ── Futures (cached 30s in market.get_futures_price) ──
        futures: dict[str, float | None] = {}
        for sym in ("ES=F", "NQ=F", "CL=F", "GC=F"):
            try:
                futures[sym] = _mkt.get_futures_price(sym)
            except Exception:
                futures[sym] = None

        # ── Urgent news from the last 8h (Reddit/Bloomberg-style overnight) ──
        urgent: list[dict] = []
        try:
            urgent = _sig.get_urgent_articles(minutes=8 * 60)[:5]
        except Exception:
            urgent = []
        urgent_compact = [{
            "title": (u.get("title") or "")[:140],
            "source": u.get("source"),
            "ai_score": u.get("ai_score"),
            "urgency": u.get("urgency"),
            "first_seen": u.get("first_seen"),
            "tickers": u.get("tickers", [])[:5],
        } for u in urgent]

        # ── High-score overnight signals as a secondary list ──
        top: list[dict] = []
        try:
            top = _sig.get_top_signals(n=5, hours=8, min_score=5.0)
        except Exception:
            top = []
        top_compact = [{
            "title": (s.get("title") or "")[:140],
            "source": s.get("source"),
            "ai_score": s.get("ai_score"),
            "tickers": s.get("tickers", [])[:5],
            "first_seen": s.get("first_seen"),
        } for s in top]

        # ── One-line summary ──
        if is_open:
            if secs is not None:
                hrs = secs // 3600
                mins = (secs % 3600) // 60
                status_line = f"Market OPEN — closes in {hrs}h{mins:02d}m"
            else:
                status_line = "Market OPEN"
        else:
            if secs is not None and next_dt is not None:
                hrs = secs // 3600
                mins = (secs % 3600) // 60
                status_line = f"Market CLOSED — opens in {hrs}h{mins:02d}m ({next_dt.astimezone(_mkt.NY).strftime('%a %H:%M %Z')})"
            else:
                status_line = "Market CLOSED"

        return jsonify({
            "as_of": now_utc.isoformat(timespec="seconds"),
            "market_open": is_open,
            "next_event_utc": next_dt.isoformat(timespec="seconds") if next_dt else None,
            "next_event_seconds": secs,
            "status_line": status_line,
            "futures": futures,
            "urgent_news": urgent_compact,
            "top_signals": top_compact,
            "urgent_count": len(urgent_compact),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _classify_action(ticker: str, held_qty: float, quant: dict, news_score: float, news_urgent: bool) -> tuple[str, float, list[str]]:
    """Co-pilot rules. Returns (action, conviction 0..1, reason_bullets).
    Conservative — never says BUY without at least one technical confirm."""
    notes: list[str] = []
    rsi = quant.get("RSI") if quant else None
    macd = quant.get("MACD") if quant else None
    mom5 = quant.get("mom_5d") if quant else None
    mom20 = quant.get("mom_20d") if quant else None

    # ── Technical scoring (-1..+1 bullish bias) ──
    bias = 0.0
    if rsi is not None:
        if rsi < 30:
            bias += 0.4; notes.append(f"RSI {rsi:.0f} oversold")
        elif rsi < 45:
            bias += 0.1; notes.append(f"RSI {rsi:.0f} cool")
        elif rsi > 70:
            bias -= 0.4; notes.append(f"RSI {rsi:.0f} overbought")
        elif rsi > 60:
            bias -= 0.1; notes.append(f"RSI {rsi:.0f} hot")
    if macd:
        if macd == "bullish":
            bias += 0.25; notes.append("MACD bullish")
        elif macd == "bearish":
            bias -= 0.25; notes.append("MACD bearish")
    if mom5 is not None:
        if mom5 > 3:
            bias += 0.15; notes.append(f"5d +{mom5:.1f}%")
        elif mom5 < -3:
            bias -= 0.15; notes.append(f"5d {mom5:.1f}%")
    if mom20 is not None:
        if mom20 > 8:
            bias += 0.1; notes.append(f"20d +{mom20:.1f}%")
        elif mom20 < -8:
            bias -= 0.1; notes.append(f"20d {mom20:.1f}%")

    bias = max(-1.0, min(1.0, bias))

    # ── News weight ──
    news_weight = min(news_score / 10.0, 1.0)
    if news_urgent:
        news_weight = min(news_weight + 0.2, 1.0)
        notes.insert(0, "URGENT news")

    # ── Action selection ──
    if held_qty > 0:
        # EXIT must be checked before TRIM: a strong bearish bias (< -0.5) also
        # satisfies the TRIM guard (bias < -0.3) when news is quiet, so testing
        # TRIM first swallowed the EXIT case and downgraded severity exactly
        # when the technical breakdown was strongest.
        if bias < -0.5:
            return "EXIT", min(0.65 + abs(bias) * 0.3, 0.95), notes
        if bias < -0.3 and news_weight < 0.4:
            return "TRIM", min(0.6 + abs(bias) * 0.3, 0.95), notes
        if bias > 0.25 and news_weight > 0.5:
            return "ADD", min(0.5 + bias * 0.3 + news_weight * 0.2, 0.95), notes
        return "HOLD", 0.4 + max(0.0, bias) * 0.2, notes
    else:
        # not held
        if news_weight > 0.65 and bias > 0.1:
            return "BUY", min(0.5 + news_weight * 0.3 + max(0.0, bias) * 0.2, 0.95), notes
        if news_weight > 0.5 or abs(bias) > 0.35:
            return "WATCH", min(0.3 + news_weight * 0.3 + abs(bias) * 0.2, 0.8), notes
        return "WATCH", 0.2 + news_weight * 0.2, notes


@app.route("/api/suggestions")
@swr_cached("suggestions", 45.0)
def suggestions_api():
    """Trade-idea co-pilot. Ranked list of BUY / ADD / TRIM / EXIT / WATCH cards.

    Inputs: top-scored articles from last 6h (digital-intern), live quant signals,
    current open positions. Output is *decision support*, not auto-execution —
    the live trader is still Opus 4.7 in strategy.py."""
    try:
        from . import signals as _sig

        # Pull top signals (broader window than the trader uses, for visibility).
        try:
            top_signals = _sig.get_top_signals(n=30, hours=6, min_score=5.0)
        except Exception as e:
            return jsonify({"error": f"signals unavailable: {e}", "suggestions": []})

        store = get_store()
        positions = store.open_positions()
        held: dict[str, float] = {}
        position_pl: dict[str, float] = {}
        for p in positions:
            if p.get("type") == "stock":
                held[p["ticker"]] = held.get(p["ticker"], 0.0) + float(p.get("qty") or 0)
                # store.open_positions() doesn't include pl_pct — derive from avg/current.
                avg = float(p.get("avg_cost") or 0.0)
                cur = float(p.get("current_price") or 0.0) or avg
                position_pl[p["ticker"]] = ((cur - avg) / avg * 100) if avg else 0.0

        # Build the candidate ticker set: (news-mentioned ∩ watchlist) ∪ currently held.
        # Constraining to the watchlist filters out the ticker-extractor's noise
        # (acronyms like GSPC / IXIC / DJI that yfinance can't price anyway).
        try:
            from .strategy import WATCHLIST as _WATCHLIST
            universe = {t.upper() for t in _WATCHLIST}
        except Exception:
            universe = set()
        universe |= {t.upper() for t in held}

        candidates: dict[str, dict] = {}
        for art in top_signals:
            for tk in art.get("tickers") or []:
                if not tk or len(tk) > 6:
                    continue
                if tk.upper() not in universe:
                    continue
                rec = candidates.setdefault(tk, {
                    "ticker": tk,
                    "news_count": 0,
                    "news_max_score": 0.0,
                    "news_urgent": False,
                    "top_headline": None,
                    "top_url": None,
                })
                rec["news_count"] += 1
                if (art.get("ai_score") or 0) > rec["news_max_score"]:
                    rec["news_max_score"] = float(art.get("ai_score") or 0)
                    rec["top_headline"] = (art.get("title") or "")[:140]
                    rec["top_url"] = art.get("url")
                if (art.get("urgency") or 0) >= 1:
                    rec["news_urgent"] = True
        for tk in held:
            candidates.setdefault(tk, {
                "ticker": tk,
                "news_count": 0,
                "news_max_score": 0.0,
                "news_urgent": False,
                "top_headline": None,
                "top_url": None,
            })

        # Pull quant signals in bulk (cached 5min).
        from . import market as _mkt
        try:
            from .strategy import get_quant_signals_live
            tickers = list(candidates.keys())
            quant = get_quant_signals_live(tickers) if tickers else {}
        except Exception:
            quant = {}

        # Live prices (bulk fetch from market.get_prices, cached 30s).
        try:
            prices = _mkt.get_prices(list(candidates.keys())) if candidates else {}
        except Exception:
            prices = {}

        out = []
        for tk, c in candidates.items():
            q = quant.get(tk, {})
            action, conviction, notes = _classify_action(
                tk,
                held.get(tk, 0.0),
                q,
                c["news_max_score"],
                c["news_urgent"],
            )
            out.append({
                "ticker": tk,
                "action": action,
                "conviction": round(conviction, 2),
                "price": prices.get(tk),
                "held_qty": held.get(tk, 0.0),
                "position_pl_pct": position_pl.get(tk),
                "news_count": c["news_count"],
                "news_max_score": round(c["news_max_score"], 1),
                "news_urgent": c["news_urgent"],
                "top_headline": c["top_headline"],
                "top_url": c["top_url"],
                "rsi": q.get("RSI"),
                "macd": q.get("MACD"),
                "mom_5d": q.get("mom_5d"),
                "mom_20d": q.get("mom_20d"),
                "reasons": notes,
            })

        # Rank: action priority then conviction.
        priority = {"EXIT": 0, "TRIM": 1, "BUY": 2, "ADD": 3, "WATCH": 4, "HOLD": 5}
        out.sort(key=lambda r: (priority.get(r["action"], 9), -r["conviction"]))
        out = out[:20]

        action_counts: dict[str, int] = {}
        for r in out:
            action_counts[r["action"]] = action_counts.get(r["action"], 0) + 1

        return jsonify({
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_candidates": len(candidates),
            "n_signals_used": len(top_signals),
            "action_counts": action_counts,
            "suggestions": out,
        })
    except Exception as e:
        return jsonify({"error": str(e), "suggestions": []}), 500


# ───────── /api/suggestion-impact — "if I act on this, what happens to the book?" ─────────
# /api/suggestions returns ranked BUY / ADD / TRIM / EXIT / WATCH ideas but is
# silent on the OPERATIONAL consequence: a BUY of MU at 5% of equity might
# tip concentration_top1 past the 40% MEDIUM threshold or burn the last cash.
# This endpoint augments each suggestion with the per-trade projection — same
# concentration math (_concentration_severity, _classify, _LEVERAGE_BETA)
# /api/risk uses, so the projected severity buckets are identical to the live
# baseline (no second taxonomy to drift). Each projection is INDEPENDENT (act
# on this idea ALONE) — that's the trader's actual decision unit.
_SUGGESTION_DEFAULT_SIZE_PCT = 5.0  # default BUY/ADD sizing as % of total_value
_SUGGESTION_TRIM_FRACTION = 0.5     # TRIM = sell half of held value


def _portfolio_rows_from_positions(positions: list[dict]) -> tuple[list[dict], float]:
    """Pure: build the same per-position rows /api/risk emits (ticker, sector,
    market_value, pct_port) PLUS the leveraged-USD total. Decoupled from the
    Flask request lifecycle so build_suggestion_impact can stay testable."""
    rows: list[dict] = []
    leveraged_usd = 0.0
    for p in positions:
        mult = 100 if p.get("type") in ("call", "put") else 1
        price = float(p.get("current_price") or p.get("avg_cost") or 0.0)
        qty = float(p.get("qty") or 0)
        val = price * qty * mult
        sec = _classify(p["ticker"])
        if sec in _LEVERAGED_SECTORS:
            leveraged_usd += val
        rows.append({
            "ticker": p["ticker"],
            "type": p.get("type"),
            "sector": sec,
            "market_value": val,
            "avg_cost": float(p.get("avg_cost") or 0.0),
            "current_price": float(p.get("current_price") or 0.0),
            "qty": qty,
            "multiplier": mult,
        })
    return rows, leveraged_usd


def _conc_top1_top3(rows: list[dict], total_value: float) -> tuple[float, float, str | None]:
    """Pure: compute (top1_pct, top3_pct, top1_ticker) from a list of rows
    that carry ``market_value`` and ``ticker``. Empty / zero-value → (0, 0, None)."""
    if not rows or total_value <= 0:
        return 0.0, 0.0, None
    sorted_rows = sorted(rows, key=lambda r: -float(r.get("market_value") or 0.0))
    top1_val = float(sorted_rows[0].get("market_value") or 0.0)
    top3_val = sum(float(r.get("market_value") or 0.0) for r in sorted_rows[:3])
    return (
        round(top1_val / total_value * 100, 2),
        round(top3_val / total_value * 100, 2),
        sorted_rows[0].get("ticker"),
    )


def build_suggestion_impact(
    suggestions: list[dict],
    portfolio_rows: list[dict],
    *,
    cash: float,
    total_value: float,
    size_pct: float = _SUGGESTION_DEFAULT_SIZE_PCT,
    trim_fraction: float = _SUGGESTION_TRIM_FRACTION,
) -> dict:
    """Pure: augment each suggestion with the per-trade portfolio projection.

    Each projection is INDEPENDENT — "if I take THIS idea ALONE, what changes?"
    BUY / ADD: cash burn, post-trade per-ticker pct, post-trade top1/top3,
    post-trade severity (LOW/MEDIUM/HIGH via the same ``_concentration_severity``
    /api/risk uses — single taxonomy, no drift). Sizing defaults to 5% of
    total_value (the same number the live trader's risk prompt assumes when
    unconstrained), capped at available cash for BUYs (ADDs are allowed to
    burn through cash; the suggestion engine doesn't currently emit ADDs
    that would, but the projection makes the consequence visible).

    TRIM (default 50%) / EXIT (100%): cash freed, realized P/L at current
    price, post-trade top1/top3, severity drop (e.g., MEDIUM → LOW = "frees
    concentration"). Both pass back the per-ticker baseline so the trader
    can SEE the before/after side-by-side.

    HOLD / WATCH: surfaces ``would_act: false`` — no projection, but the
    cards still come back so the UI doesn't have to merge two endpoints.

    Pure / total: any missing field defaults to 0; a non-list suggestions
    input returns the well-formed empty envelope; never raises.
    """
    if not isinstance(suggestions, list):
        suggestions = []
    if not isinstance(portfolio_rows, list):
        portfolio_rows = []
    cash = max(0.0, float(cash or 0.0))
    total_value = max(0.0, float(total_value or 0.0))
    size_pct = max(0.0, min(100.0, float(size_pct)))
    trim_fraction = max(0.0, min(1.0, float(trim_fraction)))

    base_rows = [dict(r) for r in portfolio_rows]  # shallow copies for mutation
    base_top1, base_top3, base_top1_tk = _conc_top1_top3(base_rows, total_value)
    base_sev, _ = _concentration_severity(base_top1, base_top3)

    default_size_usd = round(size_pct / 100.0 * total_value, 2)

    out_cards: list[dict] = []
    for s in suggestions:
        if not isinstance(s, dict):
            continue
        action = (s.get("action") or "").upper()
        ticker = (s.get("ticker") or "").upper()
        try:
            price = float(s.get("price") or 0.0)
        except (TypeError, ValueError):
            price = 0.0
        try:
            held_qty = float(s.get("held_qty") or 0.0)
        except (TypeError, ValueError):
            held_qty = 0.0
        # Find existing row by ticker (None for un-held).
        existing = next(
            (r for r in base_rows if (r.get("ticker") or "").upper() == ticker),
            None,
        )

        card = {
            "ticker": ticker,
            "action": action,
            "would_act": action in ("BUY", "ADD", "TRIM", "EXIT"),
            "baseline_top1_pct": base_top1,
            "baseline_top3_pct": base_top3,
            "baseline_severity": base_sev,
            "baseline_position_pct": (
                round(float(existing["market_value"]) / total_value * 100, 2)
                if existing and total_value > 0
                else 0.0
            ),
        }

        if action in ("BUY", "ADD") and price > 0 and total_value > 0:
            # Cap by cash for BUYs (unheld). ADDs reuse default size and only
            # surface a cash_constrained flag if they'd overdraft — the
            # operator is then choosing to free cash first.
            uncapped = default_size_usd
            sized_usd = min(uncapped, cash) if action == "BUY" else uncapped
            cash_constrained = action == "BUY" and uncapped > cash
            projected_qty = round(sized_usd / price, 4) if price > 0 else 0.0
            projected_cash = round(cash - sized_usd, 2)
            # Project the post-trade row for this ticker.
            existing_val = (
                float(existing["market_value"]) if existing else 0.0
            )
            new_val = existing_val + sized_usd
            proj_rows = [r for r in base_rows if r is not existing]
            if existing is not None:
                proj_row = dict(existing)
                proj_row["market_value"] = new_val
                proj_rows.append(proj_row)
            else:
                proj_rows.append({
                    "ticker": ticker,
                    "sector": _classify(ticker),
                    "market_value": new_val,
                })
            top1, top3, top1_tk = _conc_top1_top3(proj_rows, total_value)
            sev, _ = _concentration_severity(top1, top3)
            card.update({
                "projected_size_usd": round(sized_usd, 2),
                "projected_size_pct_of_book": round(
                    sized_usd / total_value * 100, 2
                ) if total_value > 0 else 0.0,
                "projected_qty": projected_qty,
                "projected_cash_after": projected_cash,
                "projected_position_pct_after": round(
                    new_val / total_value * 100, 2
                ),
                "projected_top1_pct_after": top1,
                "projected_top1_ticker_after": top1_tk,
                "projected_top3_pct_after": top3,
                "projected_severity_after": sev,
                "would_overconcentrate": (
                    sev != "LOW" and base_sev == "LOW"
                ) or (sev == "HIGH" and base_sev != "HIGH"),
                "cash_constrained": cash_constrained,
            })

        elif action in ("TRIM", "EXIT") and existing is not None:
            frac = 1.0 if action == "EXIT" else trim_fraction
            current_val = float(existing["market_value"])
            proceeds = round(current_val * frac, 2)
            # Realized P/L at current price: (current - avg_cost) × sold_qty × mult.
            sold_qty = float(existing["qty"]) * frac
            cost_per = float(existing["avg_cost"]) * existing.get("multiplier", 1)
            cur_per = float(existing["current_price"]) * existing.get("multiplier", 1)
            realized_pnl = round((cur_per - cost_per) * sold_qty, 2)
            new_val = current_val - proceeds
            proj_rows = [r for r in base_rows if r is not existing]
            if new_val > 0.01:
                proj_row = dict(existing)
                proj_row["market_value"] = new_val
                proj_rows.append(proj_row)
            top1, top3, top1_tk = _conc_top1_top3(proj_rows, total_value)
            sev, _ = _concentration_severity(top1, top3)
            card.update({
                "projected_proceeds_usd": proceeds,
                "projected_realized_pnl_usd": realized_pnl,
                "projected_cash_after": round(cash + proceeds, 2),
                "projected_position_pct_after": round(
                    new_val / total_value * 100, 2
                ) if total_value > 0 else 0.0,
                "projected_top1_pct_after": top1,
                "projected_top1_ticker_after": top1_tk,
                "projected_top3_pct_after": top3,
                "projected_severity_after": sev,
                "frees_concentration": (
                    base_sev != "LOW" and sev == "LOW"
                ) or (base_sev == "HIGH" and sev == "MEDIUM"),
            })

        out_cards.append(card)

    return {
        "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "baseline_top1_pct": base_top1,
        "baseline_top1_ticker": base_top1_tk,
        "baseline_top3_pct": base_top3,
        "baseline_severity": base_sev,
        "cash_usd": round(cash, 2),
        "total_value_usd": round(total_value, 2),
        "default_size_pct": size_pct,
        "default_size_usd": default_size_usd,
        "trim_fraction": trim_fraction,
        "n_cards": len(out_cards),
        "cards": out_cards,
    }


@app.route("/api/suggestion-impact")
@swr_cached("suggestion_impact", 45.0)
def suggestion_impact_api():
    """Per-trade portfolio projection layered on top of /api/suggestions.

    For each BUY / ADD: cash burn + post-trade per-ticker pct + post-trade
    top1/top3 + severity bucket (LOW/MEDIUM/HIGH via the same taxonomy
    /api/risk uses; no drift).
    For each TRIM / EXIT: proceeds, realized P/L at current price, post-trade
    concentration, ``frees_concentration`` flag.
    HOLD / WATCH: passed through with ``would_act: false``.

    ``?size_pct=`` clamped 0..100 (default 5.0).
    """
    try:
        # Re-derive suggestions from the same code path (cached internally at
        # 45s by @swr_cached on suggestions_api). We could call the route
        # handler directly but pulling the JSON keeps the contract stable —
        # if suggestions_api evolves its shape, this endpoint follows.
        sug_resp = suggestions_api()
        sug_data = sug_resp.get_json() if hasattr(sug_resp, "get_json") else {}
        if not isinstance(sug_data, dict) or sug_data.get("error"):
            return jsonify({
                "error": (sug_data or {}).get("error", "suggestions unavailable"),
                "cards": [],
            }), 200
        suggestions = sug_data.get("suggestions") or []
        try:
            size_pct = float(request.args.get("size_pct", _SUGGESTION_DEFAULT_SIZE_PCT))
        except (TypeError, ValueError):
            size_pct = _SUGGESTION_DEFAULT_SIZE_PCT
        size_pct = max(0.0, min(100.0, size_pct))

        store = get_store()
        pf = store.get_portfolio()
        positions = store.open_positions()
        rows, _lev = _portfolio_rows_from_positions(positions)
        out = build_suggestion_impact(
            suggestions,
            rows,
            cash=float(pf.get("cash") or 0.0),
            total_value=float(pf.get("total_value") or 0.0),
            size_pct=size_pct,
        )
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e), "cards": []}), 500


# ───────── Feature-dev additions (2026-05-14 part 2) ─────────
# /api/greeks         — portfolio-wide option Greeks (delta/gamma/theta/vega)
# /api/sector-heatmap — DRAM/semis bucket momentum + relative strength + news
# /api/news-deduped   — top signals after dedup + urgency decay (kills syndication noise)


@app.route("/api/greeks")
def greeks_api():
    """Per-leg and portfolio-wide Black-Scholes Greeks for open option positions.

    Stocks contribute pure delta. Options use implied vol from the live yfinance
    chain (DEFAULT_IV fallback when the chain has nothing useful)."""
    try:
        from .analytics.greeks import compute_position_greeks
        store = get_store()
        positions = store.open_positions()
        result = compute_position_greeks(positions)
        # Quick portfolio-level summary so callers don't have to recompute.
        total_value = float(store.get_portfolio().get("total_value") or 0.0)
        totals = result.get("totals", {})
        if total_value > 0:
            result["totals"]["delta_pct_port"] = round(
                totals.get("gross_notional", 0) / total_value * 100, 2
            )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _scorer_verdict(pred: float) -> str:
    """Bucket a predicted 5-day return into a coarse verdict label."""
    if pred >= 3.0:
        return "STRONG_HOLD"
    if pred >= 1.0:
        return "HOLD"
    if pred >= -1.0:
        return "NEUTRAL"
    if pred >= -3.0:
        return "TRIM"
    return "EXIT"


@app.route("/api/scorer-predictions")
@swr_cached("scorer-predictions", 60.0)
def scorer_predictions_api():
    """DecisionScorer prediction per currently-held stock position.

    Builds a feature vector from live RSI/MACD/momentum + news sentiment for
    each held ticker, runs the trained scorer, and returns predicted 5-day
    forward return %. When the scorer isn't trained yet (<500 outcomes), the
    response still lists positions but ``is_trained`` is False so the UI can
    grey them out."""
    try:
        from .ml.decision_scorer import DecisionScorer
        from .strategy import get_quant_signals_live
        from . import signals as _sig
        from . import market as _mkt

        scorer = DecisionScorer()

        store = get_store()
        positions = store.open_positions()
        held_tickers = sorted({
            p["ticker"] for p in positions
            if p.get("type") == "stock" and (p.get("qty") or 0) > 0
        })

        result = {
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "is_trained": scorer.is_trained,
            "n_train": scorer.n_train,
            "gate_threshold": 500,
            "predictions": [],
        }
        if not held_tickers:
            return jsonify(result)

        # Live RSI / MACD / momentum — same source the live trader uses.
        quant = get_quant_signals_live(held_tickers) or {}
        # News-based "ml_score" proxy — average ai_score across mentions in the
        # last 4 hours. Matches the feature the model was trained on, since
        # backtest decisions used ml_score from articles in the same window.
        sent_list = _sig.ticker_sentiments(held_tickers, hours=4) or []
        sent_by_tk = {s["ticker"]: s for s in sent_list}

        # Crude regime proxy — SPY 5d momentum as the multiplier seed. Falls
        # back to 1.0 when unavailable so prediction still returns sensible.
        regime_mult = 1.0
        try:
            spy_q = get_quant_signals_live(["SPY"]).get("SPY") or {}
            spy_mom = spy_q.get("mom_5d")
            if isinstance(spy_mom, (int, float)):
                # Map roughly: +2% = bull (1.15), -2% = bear (0.85)
                regime_mult = max(0.7, min(1.3, 1.0 + spy_mom * 0.075))
        except Exception:
            pass

        preds = []
        for tk in held_tickers:
            q = quant.get(tk) or {}
            sent = sent_by_tk.get(tk) or {}
            # Use max_score for ml_score proxy — captures the strongest signal
            # in the window rather than diluting by averaging across mentions.
            ml_score = float(sent.get("max_score") or 0.0)
            # predict_with_meta (not predict) so the response can flag when
            # the point estimate is a clamped ±50 floor/ceiling from an
            # off-distribution extrapolation. A bare clamped -50 reads as a
            # confident EXIT otherwise — and the unified conviction board
            # pins its ML axis to it. AGENTS.md documents this contract.
            meta = scorer.predict_with_meta(
                ml_score=ml_score,
                rsi=q.get("rsi"),
                macd=q.get("macd_signal"),
                mom5=q.get("mom_5d"),
                mom20=q.get("mom_20d"),
                regime_mult=regime_mult,
                ticker=tk,
                vol_ratio=q.get("vol_ratio"),
                bb_pos=q.get("bb_position"),
            )
            pred = meta["pred"]
            row = {
                "ticker": tk,
                "pred_5d_return_pct": round(float(pred), 3),
                "verdict": _scorer_verdict(float(pred)),
                "rsi": q.get("RSI"),
                "macd": q.get("MACD"),
                "mom_5d": q.get("mom_5d"),
                "mom_20d": q.get("mom_20d"),
                "ml_news_score": round(ml_score, 2),
                "news_count": sent.get("n", 0),
                "news_urgent": sent.get("urgent", 0),
                "off_distribution": bool(meta["off_distribution"]),
            }
            if meta["off_distribution"]:
                row["raw_pred_5d_return_pct"] = round(float(meta["raw"]), 3)
            preds.append(row)
        # Highest predicted return first so the trader sees winners at the top.
        preds.sort(key=lambda r: -(r["pred_5d_return_pct"] or 0))
        result["n_positions"] = len(preds)
        result["regime_mult"] = round(regime_mult, 3)
        result["predictions"] = preds
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "predictions": []}), 500


@app.route("/api/scorer-attribution")
@swr_cached("scorer-attribution", 60.0)
def scorer_attribution_api():
    """Per-feature signed attribution for the DecisionScorer's prediction on
    one ticker — opens the otherwise black-box scorer so the operator can see
    WHICH input (RSI, momentum, news, sector) drives a given EXIT/HOLD verdict.

    ``?ticker=SYM`` (default: first held stock, else SPY). Same live feature
    plumbing as /api/scorer-predictions; read-only, never touches the trade
    path."""
    try:
        from .ml.decision_scorer import DecisionScorer
        from .strategy import get_quant_signals_live
        from . import signals as _sig

        scorer = DecisionScorer()
        store = get_store()

        req_tk = (request.args.get("ticker") or "").strip().upper()
        if not req_tk:
            held = [p["ticker"] for p in store.open_positions()
                    if p.get("type") == "stock" and (p.get("qty") or 0) > 0]
            req_tk = sorted(held)[0] if held else "SPY"

        result = {
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ticker": req_tk,
            "is_trained": scorer.is_trained,
            "n_train": scorer.n_train,
            "gate_threshold": 500,
        }
        if not scorer.is_trained:
            result["attribution"] = {"trained": False, "contributions": []}
            return jsonify(result)

        q = (get_quant_signals_live([req_tk]) or {}).get(req_tk) or {}
        sent_list = _sig.ticker_sentiments([req_tk], hours=4) or []
        sent = (sent_list[0] if sent_list else {}) or {}
        ml_score = float(sent.get("max_score") or 0.0)

        regime_mult = 1.0
        try:
            spy_q = (get_quant_signals_live(["SPY"]) or {}).get("SPY") or {}
            spy_mom = spy_q.get("mom_5d")
            if isinstance(spy_mom, (int, float)):
                regime_mult = max(0.7, min(1.3, 1.0 + spy_mom * 0.075))
        except Exception:
            pass

        attr = scorer.feature_contributions(
            ml_score=ml_score,
            rsi=q.get("rsi"),
            macd=q.get("macd_signal"),
            mom5=q.get("mom_5d"),
            mom20=q.get("mom_20d"),
            regime_mult=regime_mult,
            ticker=req_tk,
            vol_ratio=q.get("vol_ratio"),
            bb_pos=q.get("bb_position"),
        )
        result["regime_mult"] = round(regime_mult, 3)
        result["ml_news_score"] = round(ml_score, 2)
        result["attribution"] = attr
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "attribution": {"contributions": []}}), 500


@app.route("/api/scorer-portfolio-attribution")
@swr_cached("scorer-portfolio-attribution", 90.0)
def scorer_portfolio_attribution_api():
    """Top driving features (signed contributions) behind the DecisionScorer's
    verdict for *every* currently-held stock — portfolio-wide attribution in a
    single call.

    /api/scorer-attribution answers the same question for ONE ticker at a time,
    which means N HTTP round-trips when the operator is asking "WHY does the
    scorer want out of half my book?". This composes the same live feature
    plumbing /api/scorer-predictions uses + feature_contributions() and emits
    one row per held position, each carrying the top 3 features by |signed
    contribution|. Read-only; never touches the trade path; off-distribution
    rows are flagged so a clamped extrapolation isn't read as a confident
    EXIT."""
    try:
        from .ml.decision_scorer import DecisionScorer
        from .strategy import get_quant_signals_live
        from . import signals as _sig

        scorer = DecisionScorer()
        store = get_store()
        held = sorted({
            p["ticker"] for p in store.open_positions()
            if p.get("type") == "stock" and (p.get("qty") or 0) > 0
        })
        result = {
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "is_trained": scorer.is_trained,
            "n_train": scorer.n_train,
            "gate_threshold": 500,
            "n_positions": len(held),
            "rows": [],
        }
        if not held or not scorer.is_trained:
            return jsonify(result)

        quant = get_quant_signals_live(held) or {}
        sent_by_tk = {s["ticker"]: s for s in (_sig.ticker_sentiments(held, hours=4) or [])}
        regime_mult = 1.0
        try:
            spy_mom = (get_quant_signals_live(["SPY"]).get("SPY") or {}).get("mom_5d")
            if isinstance(spy_mom, (int, float)):
                regime_mult = max(0.7, min(1.3, 1.0 + spy_mom * 0.075))
        except Exception:
            pass

        rows = []
        for tk in held:
            q = quant.get(tk) or {}
            sent = sent_by_tk.get(tk) or {}
            ml_score = float(sent.get("max_score") or 0.0)
            common = dict(
                ml_score=ml_score, rsi=q.get("rsi"), macd=q.get("macd_signal"),
                mom5=q.get("mom_5d"), mom20=q.get("mom_20d"),
                regime_mult=regime_mult, ticker=tk,
                vol_ratio=q.get("vol_ratio"), bb_pos=q.get("bb_position"),
            )
            meta = scorer.predict_with_meta(**common)
            attr = scorer.feature_contributions(**common)
            contribs = (attr.get("contributions") or [])[:3]
            rows.append({
                "ticker": tk,
                "pred_5d_return_pct": round(float(meta["pred"]), 3),
                "verdict": _scorer_verdict(float(meta["pred"])),
                "off_distribution": bool(meta["off_distribution"]),
                "raw_pred_5d_return_pct": (round(float(meta["raw"]), 3)
                                            if meta["off_distribution"] else None),
                "top_features": contribs,
                "interaction_residual": attr.get("interaction_residual"),
                "ml_news_score": round(ml_score, 2),
            })
        # Most-bearish first — that's the row most likely to drive an EXIT call,
        # which is what the operator opening this panel mid-drawdown wants to see.
        rows.sort(key=lambda r: r["pred_5d_return_pct"])
        result["rows"] = rows
        result["regime_mult"] = round(regime_mult, 3)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "rows": []}), 500


@app.route("/api/scorer-opportunities")
@swr_cached("scorer-opportunities", 90.0)
def scorer_opportunities_api():
    """DecisionScorer-ranked watchlist names we do NOT currently own.

    /api/watchlist-opportunities ranks unowned names by *news heat*;
    /api/scorer-predictions runs the trained quant scorer only on *held*
    positions. Neither answers "what does the model say we should buy?" — the
    quantitative complement of the missed-opportunity radar. This endpoint
    runs predict_with_meta over every unowned WATCHLIST ticker and returns the
    top-N by predicted 5-day forward return, with the same off-distribution
    trust flag scorer-predictions surfaces."""
    try:
        from .ml.decision_scorer import DecisionScorer
        from .strategy import WATCHLIST as _WATCHLIST, get_quant_signals_live
        from . import signals as _sig

        try:
            n_top = max(1, min(50, int(request.args.get("n", 12))))
        except (TypeError, ValueError):
            n_top = 12

        scorer = DecisionScorer()
        store = get_store()
        held = {str(p.get("ticker") or "").upper()
                for p in store.open_positions() if (p.get("qty") or 0) > 0}
        candidates = sorted({t.upper() for t in _WATCHLIST} - held)

        result = {
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "is_trained": scorer.is_trained,
            "n_train": scorer.n_train,
            "gate_threshold": 500,
            "n_candidates": len(candidates),
            "n_held_excluded": len(held),
            "opportunities": [],
        }
        if not candidates or not scorer.is_trained:
            return jsonify(result)

        quant = get_quant_signals_live(candidates) or {}
        sent_list = _sig.ticker_sentiments(candidates, hours=4) or []
        sent_by_tk = {s["ticker"]: s for s in sent_list}

        regime_mult = 1.0
        try:
            spy_q = (get_quant_signals_live(["SPY"]) or {}).get("SPY") or {}
            spy_mom = spy_q.get("mom_5d")
            if isinstance(spy_mom, (int, float)):
                regime_mult = max(0.7, min(1.3, 1.0 + spy_mom * 0.075))
        except Exception:
            pass

        rows: list[dict] = []
        for tk in candidates:
            q = quant.get(tk)
            if not q:
                continue
            sent = sent_by_tk.get(tk) or {}
            ml_score = float(sent.get("max_score") or 0.0)
            meta = scorer.predict_with_meta(
                ml_score=ml_score,
                rsi=q.get("rsi"),
                macd=q.get("macd_signal"),
                mom5=q.get("mom_5d"),
                mom20=q.get("mom_20d"),
                regime_mult=regime_mult,
                ticker=tk,
                vol_ratio=q.get("vol_ratio"),
                bb_pos=q.get("bb_position"),
            )
            pred = float(meta["pred"])
            row = {
                "ticker": tk,
                "pred_5d_return_pct": round(pred, 3),
                "verdict": _scorer_verdict(pred),
                "rsi": q.get("RSI"),
                "macd": q.get("MACD"),
                "mom_5d": q.get("mom_5d"),
                "mom_20d": q.get("mom_20d"),
                "ml_news_score": round(ml_score, 2),
                "news_count": sent.get("n", 0),
                "news_urgent": sent.get("urgent", 0),
                "off_distribution": bool(meta["off_distribution"]),
            }
            if meta["off_distribution"]:
                row["raw_pred_5d_return_pct"] = round(float(meta["raw"]), 3)
            rows.append(row)

        rows.sort(key=lambda r: -(r["pred_5d_return_pct"] or 0))
        result["n_scored"] = len(rows)
        result["regime_mult"] = round(regime_mult, 3)
        result["opportunities"] = rows[:n_top]
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "opportunities": []}), 500


@app.route("/api/sector-heatmap")
@swr_cached("sector-heatmap", 90.0)
def sector_heatmap_api():
    """DRAM / semis sector heatmap. Buckets: memory_core, semis_equipment, foundry,
    design, memory_leveraged, optical, etf. Each ticker carries mom_5d, mom_20d,
    RSI, vs_sox_5d, and the 24h news pulse from digital-intern."""
    try:
        from .analytics.sector_heatmap import compute_heatmap
        return jsonify(compute_heatmap())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/news-deduped")
def news_deduped_api():
    """Top signals after dedup + exponential urgency decay.

    Default window: last 6 hours, min_score 4.0. Halflife 4h means urgency=1 at
    t=0 becomes 0.5 at t=4h, 0.25 at t=8h, and falls out at 0.125 (5h+) when the
    default cutoff is 0.5. ?hours= and ?min_score= and ?halflife= are tunable."""
    try:
        from . import signals as _sig
        from .analytics.news_dedup import dedupe_and_decay
        hours = int(request.args.get("hours", 6))
        min_score = float(request.args.get("min_score", 4.0))
        halflife = float(request.args.get("halflife", 4.0))
        # Pull a fat candidate list — dedup will compress it heavily.
        raw = _sig.get_top_signals(n=80, hours=hours, min_score=min_score)
        cleaned = dedupe_and_decay(raw, halflife_hours=halflife, min_effective=0.0)
        # Compute the "compression ratio" for the UI so the user can see how
        # much noise was suppressed.
        return jsonify({
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_raw": len(raw),
            "n_after_dedup": len(cleaned),
            "compression_ratio": round(len(raw) / max(len(cleaned), 1), 2),
            "halflife_hours": halflife,
            "articles": cleaned[:30],
        })
    except Exception as e:
        return jsonify({"error": str(e), "articles": []}), 500


# ───────── Feature-dev additions (2026-05-15) ─────────
# /api/position-thesis  — per-position integrated card (news, scorer, technicals, last decision, verdict)
# /api/calibration       — confidence calibration + signal-source attribution from realized trades
# /api/drawdown          — current DD anatomy: peak/trough, time-in-DD, per-position contribution


@app.route("/api/position-thesis")
def position_thesis_api():
    """Per-open-position thesis cards.

    Combines DecisionScorer prediction, live quant signals, news pulse from
    digital-intern, and the most recent Opus decision that touched the ticker.
    Each card carries a coarse verdict and a one-line thesis."""
    try:
        from .analytics.position_thesis import build_thesis_cards
        from .ml.decision_scorer import DecisionScorer
        from .strategy import get_quant_signals_live
        from . import signals as _sig

        store = get_store()
        positions = store.open_positions()
        held = sorted({p["ticker"] for p in positions
                       if p.get("type") == "stock" and (p.get("qty") or 0) > 0})

        # Reuse the same scorer prediction shape as /api/scorer-predictions
        # without duplicating its logic — call into the live trader helpers.
        quant = get_quant_signals_live(held) if held else {}
        sent_list = _sig.ticker_sentiments(held, hours=4) if held else []
        sent_by_tk = {s["ticker"]: s for s in sent_list}

        regime_mult = 1.0
        try:
            spy_q = (get_quant_signals_live(["SPY"]) or {}).get("SPY") or {}
            mm = spy_q.get("mom_5d")
            if isinstance(mm, (int, float)):
                regime_mult = max(0.7, min(1.3, 1.0 + mm * 0.075))
        except Exception:
            pass

        scorer = DecisionScorer()
        scorer_preds = []
        for tk in held:
            q = quant.get(tk) or {}
            sent = sent_by_tk.get(tk) or {}
            # Mirror /api/scorer-predictions exactly so both endpoints agree:
            # the scorer wants numeric macd_signal, not the "bullish"/"bearish"
            # MACD label (which _to_float silently zeroes).
            meta = scorer.predict_with_meta(
                ml_score=float(sent.get("max_score") or 0.0),
                rsi=q.get("rsi"), macd=q.get("macd_signal"),
                mom5=q.get("mom_5d"), mom20=q.get("mom_20d"),
                regime_mult=regime_mult, ticker=tk,
                vol_ratio=q.get("vol_ratio"), bb_pos=q.get("bb_position"),
            )
            row = {
                "ticker": tk,
                "pred_5d_return_pct": round(float(meta["pred"]), 3),
                "verdict": _scorer_verdict(float(meta["pred"])),
                "off_distribution": bool(meta["off_distribution"]),
            }
            if meta["off_distribution"]:
                row["raw_pred_5d_return_pct"] = round(float(meta["raw"]), 3)
            scorer_preds.append(row)

        decisions = store.recent_decisions(limit=80)
        out = build_thesis_cards(positions, decisions, scorer_preds, quant)
        out["scorer_trained"] = scorer.is_trained
        out["scorer_n_train"] = scorer.n_train
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e), "cards": []}), 500


@app.route("/api/position-runrate")
def position_runrate_api():
    """Per-open-position P/L runrate — dollars-per-day-held + verdict.

    Answers "is this position bleeding faster than I'd tolerate, or actually
    working?" at the moment of action. Composes ``build_position_runrate``
    verbatim (single source of truth, AGENTS.md invariant #10) and routes the
    pure-arithmetic builder over ``store.open_positions()`` +
    ``portfolio.total_value``. NO network, NO extra store reads — the
    risk_mirror hot-path discipline. Observational only, never gates, no caps
    (invariants #2/#12). Failure contract mirrors the rest of the dashboard:
    a builder/store fault degrades to a 500 with ``rows=[]``, never an
    exception that takes down the endpoint."""
    try:
        from .analytics.position_runrate import build_position_runrate
        store = get_store()
        pf = store.get_portfolio()
        # positions_json carries the enriched per-cycle snapshot (with
        # current_price, unrealized_pl, stale_mark already applied) — same
        # source /api/portfolio uses. No mark-to-market here on purpose:
        # the live trader's _portfolio_snapshot is the SSOT for marks; this
        # endpoint just reads what was last persisted.
        positions = pf.get("positions") or []
        if not isinstance(positions, list):
            positions = []
        # If the persisted snapshot has no opened_at (positions_json strips
        # it on persist — see store.upsert_position docstring), join it from
        # the open_positions table so the runrate builder can compute hold age.
        opened_by_key: dict[tuple, str] = {}
        for row in store.open_positions():
            key = (
                (row.get("ticker") or "").upper(),
                (row.get("type") or "").lower(),
                row.get("expiry"),
                row.get("strike"),
            )
            if row.get("opened_at"):
                opened_by_key[key] = row["opened_at"]
        enriched: list[dict] = []
        for p in positions:
            if not isinstance(p, dict):
                continue
            key = (
                (p.get("ticker") or "").upper(),
                (p.get("type") or "").lower(),
                p.get("expiry"),
                p.get("strike"),
            )
            if not p.get("opened_at") and key in opened_by_key:
                p = {**p, "opened_at": opened_by_key[key]}
            enriched.append(p)
        out = build_position_runrate(enriched, pf.get("total_value"))
        out["as_of"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e), "rows": []}), 500


@app.route("/api/calibration")
def calibration_api():
    """Confidence calibration + signal-source attribution.

    Buckets matched-and-closed BUY decisions by Opus's stated confidence
    (0.0-0.5, 0.5-0.65, 0.65-0.8, 0.8-1.0) and computes win rate + avg return
    per bucket. Also classifies decisions by reasoning keywords into
    news/technical/mixed/other and computes the same stats per source."""
    try:
        from .analytics.calibration import build_calibration
        store = get_store()
        decisions = store.recent_decisions(limit=500)
        trades = store.recent_trades(limit=500)
        return jsonify(build_calibration(decisions, trades))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/drawdown")
def drawdown_api():
    """Drawdown anatomy: peak/trough, time-in-DD, per-position contribution.

    Returns a structured 0% when the portfolio is at a fresh high so the UI
    can show a green high-water badge. ``recovery_pct`` measures how much of
    the trough has been clawed back. ``starting_equity`` is the module
    ``INITIAL_CASH`` (invariant #12, never a literal 1000 — the
    ``benchmark_api``/``analytics_api`` single-source-of-truth pattern; the
    builder's own 1000.0 default would silently desync the empty-curve
    fallback + the echoed ``starting_equity`` if ``INITIAL_CASH`` ever moves)."""
    try:
        from .analytics.drawdown import compute_drawdown
        store = get_store()
        eq = store.equity_curve(limit=2000)
        positions = store.open_positions()
        result = compute_drawdown(eq, positions,
                                  starting_equity=INITIAL_CASH)
        mt = _mark_trust_block(store)
        if mt is not None:
            result["mark_trust"] = mt
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/benchmark")
@swr_cached("benchmark", 30.0)
def benchmark_api():
    """Whole-account return vs an equal-capital S&P 500 buy-and-hold since
    inception — the "is this bot worth running vs just buying the index?" KPI.

    Distinct from ``/api/open-attribution`` (per-open-lot since entry) and
    ``/api/analytics`` ``sp500_beta`` (a regression, ``null`` for days): this
    is the full account (cash + open + every realised round-trip + unrealised
    mark) since the first equity write vs the same starting capital invested
    once in the index. Pure & DB-only — the network lives nowhere here
    (``drawdown.py`` builder/endpoint split); ``starting_equity`` is
    the module ``INITIAL_CASH`` (invariant #12, never a literal 1000)."""
    try:
        from .analytics.benchmark import build_benchmark
        store = get_store()
        eq = store.equity_curve(limit=5000)
        return jsonify(build_benchmark(eq, starting_equity=INITIAL_CASH))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/earnings-risk")
def earnings_risk_api():
    """Upcoming earnings cross-referenced against held positions + watchlist.

    Earnings are the #1 scheduled risk event — a position into a print can gap
    10%+ overnight. This pulls digital-intern's earnings calendar (:8080) and
    flags which holdings and watchlist names report soon, with a risk tier:
      HELD_IMMINENT  — you hold it and it reports within 3 days
      HELD_SOON      — you hold it and it reports within the horizon
      WATCH          — on the watchlist, not held
    """
    import json as _json
    import urllib.request as _urllib

    try:
        store = get_store()
        positions = store.open_positions()
        held: dict[str, float] = {}
        for p in positions:
            t = (p.get("ticker") or "").upper()
            if not t:
                continue
            mult = 100 if p.get("type") in ("call", "put") else 1
            price = p.get("current_price") or p.get("avg_cost") or 0.0
            held[t] = held.get(t, 0.0) + price * (p.get("qty") or 0.0) * mult

        try:
            from .strategy import WATCHLIST as _WATCHLIST
            watch = {t.upper() for t in _WATCHLIST}
        except Exception:
            watch = set()

        events = []
        source_ok = True
        try:
            with _urllib.urlopen(
                "http://127.0.0.1:8080/api/earnings", timeout=4) as resp:
                snap = _json.loads(resp.read().decode("utf-8"))
            events = snap.get("events") or []
        except Exception:
            source_ok = False

        out = []
        for ev in events:
            tk = (ev.get("ticker") or "").upper()
            if not tk:
                continue
            days = ev.get("days_away")
            in_port = tk in held
            on_watch = tk in watch
            if not in_port and not on_watch:
                continue
            if in_port and days is not None and days <= 3:
                tier = "HELD_IMMINENT"
            elif in_port:
                tier = "HELD_SOON"
            else:
                tier = "WATCH"
            out.append({
                "ticker": tk,
                "earnings_date": ev.get("earnings_date"),
                "days_away": days,
                "tier": tier,
                "held": in_port,
                "exposure_usd": round(held.get(tk, 0.0), 2) if in_port else 0.0,
            })
        # Held + soonest first; tier rank keeps imminent risk at the top.
        tier_rank = {"HELD_IMMINENT": 0, "HELD_SOON": 1, "WATCH": 2}
        out.sort(key=lambda e: (tier_rank.get(e["tier"], 9),
                                e["days_away"] if e["days_away"] is not None else 1e9))
        held_at_risk = round(sum(e["exposure_usd"] for e in out if e["held"]), 2)
        return jsonify({
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_ok": source_ok,
            "events": out,
            "n_held_reporting": sum(1 for e in out if e["held"]),
            "n_imminent": sum(1 for e in out if e["tier"] == "HELD_IMMINENT"),
            "held_exposure_at_risk_usd": held_at_risk,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/event-readiness")
def event_readiness_api():
    """Will the live trader actually be able to react before the next print?

    /api/earnings-risk says NVDA reports in 16h; /api/decision-drought says
    the bot is in a 4.7h PARALYSIS streak; /api/empty-claude-rate says
    Claude returned empty on 48% of recent cycles. Each is half the picture
    — none of them answers the operator's actual pre-print question: given
    those three facts together, is the bot statistically going to land a
    usable decision before the event?

    This composes them into a single readiness verdict per held imminent
    print (READY / DEGRADED / BLIND), plus expected_decisions_before_event
    (cycles/hr × hours-until × (1 − empty-rate)) and a one-line operator-
    actionable hint. Pulls the earnings event list from :8080/api/earnings
    exactly as /api/earnings-risk does (single source of truth); the builder
    itself is pure (``analytics/event_readiness.py``)."""
    import json as _json
    import urllib.request as _urllib

    try:
        from .analytics.event_readiness import build_event_readiness
        store = get_store()
        positions = store.open_positions()
        decisions = store.recent_decisions(limit=2000)

        events: list[dict] = []
        source_ok = True
        try:
            with _urllib.urlopen(
                "http://127.0.0.1:8080/api/earnings", timeout=4) as resp:
                snap = _json.loads(resp.read().decode("utf-8"))
            events = snap.get("events") or []
        except Exception:
            source_ok = False

        rep = build_event_readiness(positions, decisions, events)
        rep["source_ok"] = source_ok
        return jsonify(rep)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/event-calendar")
def event_calendar_api():
    """The exact upcoming-earnings awareness block the live trader now sees
    in its decision prompt (the `risk_mirror` / `tail_risk` prompt↔endpoint
    parity discipline). Reads digital-intern's `earnings_calendar.json`
    snapshot directly from disk — no `:8080` hop — and tiers it against the
    held book + watchlist exactly as `/api/earnings-risk` does (single source
    of truth, AGENTS.md #10). Observational only; never gates Opus."""
    try:
        from .analytics.event_calendar import build_event_calendar
        store = get_store()
        positions = store.open_positions()
        try:
            from .strategy import WATCHLIST as _WATCHLIST
            watch = {t.upper() for t in _WATCHLIST}
        except Exception:
            watch = set()
        held = {(p.get("ticker") or "").upper()
                for p in positions if p.get("ticker")}
        return jsonify(build_event_calendar(positions, held | watch))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _earnings_history_for(ticker: str, depth: int = 8) -> list[float]:
    """Per-ticker historical 1-day post-earnings reactions in percent.

    The I/O seam for ``/api/earnings-shock`` (the builder is pure — the
    ``tail_risk`` / ``stress_scenarios`` builder/endpoint split). For each
    past entry in ``yfinance.Ticker(t).earnings_dates``, finds the first
    daily session at or after the announcement timestamp and reports
    ``(close[first_session] / close[first_session − 1]) − 1`` in percent.

    The "first session at or after" rule handles BMO/AMC asymmetry without
    needing the report-time-of-day flag: a BMO print on day D is reflected
    in close[D] vs close[D−1]; an AMC print on day D is reflected in
    close[D+1] vs close[D] (the next session is D+1). Either way the move
    we attribute IS the first market reaction to the announcement.

    Returns the most-recent ``depth`` reactions, oldest→newest. Any failure
    returns ``[]`` so the builder degrades each row to
    ``INSUFFICIENT_HISTORY`` (never raises — the ``_safe`` contract; the
    Opus prompt is unaffected, this endpoint is observational only)."""
    try:
        import yfinance as _yf
        tk = _yf.Ticker(ticker)
        ed = tk.earnings_dates  # DataFrame; index is announcement TZ-aware ts
        if ed is None or len(ed) == 0:
            return []
        now_utc = datetime.now(timezone.utc)
        past = []
        for idx in ed.index:
            try:
                ts = idx.to_pydatetime()
            except Exception:
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts < now_utc:
                past.append(ts)
        if not past:
            return []
        past.sort()
        oldest = past[0]
        start_str = (oldest - timedelta(days=15)).date().isoformat()
        hist = tk.history(start=start_str, interval="1d")
        if hist is None or hist.empty or "Close" not in hist.columns:
            return []
        closes = hist["Close"]
        session_dates = [i.date() for i in closes.index]
        reactions: list[float] = []
        for announce_ts in past:
            target_date = announce_ts.date()
            try:
                first_after_pos = next(
                    i for i, d in enumerate(session_dates) if d >= target_date
                )
            except StopIteration:
                continue
            if first_after_pos < 1:
                continue
            try:
                c_after = float(closes.iloc[first_after_pos])
                c_before = float(closes.iloc[first_after_pos - 1])
            except (TypeError, ValueError, IndexError):
                continue
            if c_before <= 0:
                continue
            reactions.append((c_after / c_before - 1.0) * 100.0)
        return reactions[-depth:]
    except Exception:
        return []


@app.route("/api/earnings-shock")
@swr_cached("earnings-shock", 300.0)
def earnings_shock_api():
    """Pre-earnings dollarized 1σ shock for each HELD position with an
    imminent print. The forward $-at-risk complement to /api/event-calendar
    (which tells WHICH name reports WHEN but does not dollarize the gap
    against the current book).

    Composes ``build_event_calendar`` for the held set verbatim (single
    source of truth, AGENTS.md #10) so this endpoint and the prompt block
    can never disagree on what counts as held-imminent. Per-name history is
    pulled live from yfinance (``_earnings_history_for``); the builder is
    pure and the I/O seam degrades each row honestly to
    INSUFFICIENT_HISTORY on a yfinance miss. Observational only — never
    gates Opus, never injected into the decision prompt (AGENTS.md
    #2/#12 — the ``stress_scenarios`` / ``recovery`` precedent).

    SWR-cached 5 min: yfinance earnings_dates + 3y history is the slowest
    per-call yfinance shape we touch (~1-3 s per name); a 5-min TTL
    matches /api/source-edge / /api/sector-heatmap cadence."""
    try:
        from .analytics.earnings_shock import build_earnings_shock
        from .analytics.event_calendar import build_event_calendar
        store = get_store()
        pf = store.get_portfolio()
        positions = store.open_positions()
        try:
            from .strategy import WATCHLIST as _WATCHLIST
            watch = {t.upper() for t in _WATCHLIST}
        except Exception:
            watch = set()
        held = {(p.get("ticker") or "").upper()
                for p in positions if p.get("ticker")}
        ec = build_event_calendar(positions, held | watch)
        result = build_earnings_shock(
            positions,
            float(pf.get("total_value") or 0.0),
            ec,
            history_provider=_earnings_history_for,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/earnings-distribution")
@swr_cached("earnings-distribution", 300.0)
def earnings_distribution_api():
    """Empirical observed-quantile complement to ``/api/earnings-shock``.

    ``/api/earnings-shock`` assumes a Gaussian shock model and reports a single
    1σ figure per held imminent print. Earnings reactions are fat-tailed, so
    the Gaussian framing hides the historical worst case. This endpoint
    surfaces the **observed** distribution instead — worst / Q1 / median / Q3
    / best of historical 1-day post-earnings reactions per held imminent
    event, each dollarized against the current position size and book.

    Composes ``build_event_calendar`` for the held set verbatim (single source
    of truth, AGENTS.md #10) so this endpoint, ``/api/earnings-shock`` and
    ``/api/event-calendar`` can never disagree on what counts as
    held-imminent. Reuses ``_earnings_history_for`` as the I/O seam (same
    yfinance call shape, same 5-min SWR TTL). Observational only — never
    gates Opus, never injected into the decision prompt (AGENTS.md #2/#12 —
    the ``earnings_shock`` / ``stress_scenarios`` precedent).

    Per advisor framing: quantile fields are named ``q1`` / ``median`` /
    ``q3`` (observed quartiles), **not** ``p25`` / ``p50`` / ``p75``, because
    n=3–8 historical prints cannot legitimately support distributional
    percentile claims. The fields report shape; they don't promise
    distributional inference."""
    try:
        from .analytics.earnings_distribution import build_earnings_distribution
        from .analytics.event_calendar import build_event_calendar
        store = get_store()
        pf = store.get_portfolio()
        positions = store.open_positions()
        try:
            from .strategy import WATCHLIST as _WATCHLIST
            watch = {t.upper() for t in _WATCHLIST}
        except Exception:
            watch = set()
        held = {(p.get("ticker") or "").upper()
                for p in positions if p.get("ticker")}
        ec = build_event_calendar(positions, held | watch)
        result = build_earnings_distribution(
            positions,
            float(pf.get("total_value") or 0.0),
            ec,
            history_provider=_earnings_history_for,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/peer-earnings-shock")
@swr_cached("peer-earnings-shock", 300.0)
def peer_earnings_shock_api():
    """Indirect 1σ exposure on held LEVERAGED ETFs from upcoming
    constituent (peer) earnings. The fusion complement to
    ``/api/earnings-shock`` (direct held-name σ) and
    ``/api/etf-lookthrough`` (hidden indirect $-exposure). Neither
    answers the operator's actual pre-mega-cap-print question: *"NVDA
    reports tonight. I hold $148 TQQQ. NVDA's σ is 7%. What is my
    INDIRECT NVDA $-shock on the TQQQ position?"* — the arithmetic is
    one multiplication, but the surfaces above never fuse it.

    Composes ``build_etf_lookthrough`` (per-(ETF, underlying) indirect
    $-exposure) and ``build_event_calendar`` (which underlyings have
    imminent earnings) verbatim (single source of truth, AGENTS.md
    #10). σ per underlying comes from yfinance via
    ``_earnings_history_for`` + the same ``_pop_stdev`` convention
    ``earnings_shock`` uses (so a constituent's σ here byte-matches
    its σ in ``earnings_shock`` for held names). Per-row INSUFFICIENT_
    SIGMA when history < ``earnings_shock.MIN_HISTORY=3`` — the row
    surfaces but the σ is honestly withheld (the
    ``earnings_shock`` row-level discipline).

    State ladder: NO_DATA / NO_ETF_HELD / NO_PEER_EVENTS / OK.
    Verdict band (LOW / MODERATE / SEVERE) anchored to book-relative
    σ (the ``earnings_shock`` calibration shape). Advisory only —
    never gates Opus, never injected into the decision prompt, no
    caps (AGENTS.md #2/#12). SWR-cached 5 min: yfinance earnings_dates
    + 3y daily history per constituent ticker is the slowest yfinance
    shape; same TTL as ``earnings-shock`` / ``earnings-distribution`` /
    ``implied-move``."""
    try:
        from .analytics.earnings_shock import (
            MIN_HISTORY as _ES_MIN_HISTORY,
            _pop_stdev as _es_pop_stdev,
        )
        from .analytics.event_calendar import build_event_calendar
        from .analytics.peer_earnings_shock import build_peer_earnings_shock
        store = get_store()
        pf = store.get_portfolio()
        positions = store.open_positions()
        try:
            from .strategy import WATCHLIST as _WATCHLIST
            watch = {t.upper() for t in _WATCHLIST}
        except Exception:
            watch = set()
        held = {(p.get("ticker") or "").upper()
                for p in positions if p.get("ticker")}
        # event_calendar's horizon must cover the peer-shock horizon —
        # use a slightly wider window so a 7-day peer event isn't
        # pre-filtered by event_calendar's default.
        ec = build_event_calendar(positions, held | watch, horizon_days=14.0)
        snap = {
            "cash": pf.get("cash"),
            "total_value": float(pf.get("total_value") or 0.0),
            "positions": positions,
        }

        # Sigma provider: yfinance history → pop_stdev. SSOT shape:
        # byte-identical σ to earnings_shock for a name covered by
        # both surfaces (a held name's NVDA σ here equals its NVDA σ
        # in earnings_shock — same MIN_HISTORY, same _pop_stdev).
        def _sigma_for(tk: str):
            try:
                history = _earnings_history_for(tk)
            except Exception:
                return None
            if not history or len(history) < _ES_MIN_HISTORY:
                return None
            return _es_pop_stdev(history)

        return jsonify(build_peer_earnings_shock(
            snap, ec, sigma_provider=_sigma_for,
        ))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/implied-move")
@swr_cached("implied-move", 300.0)
def implied_move_api():
    """Options-market-implied move for each HELD position with an imminent
    earnings print. The **forward, market-priced complement** to
    ``/api/earnings-shock`` (which gives historical-Gaussian σ from prior
    prints) and ``/api/earnings-distribution`` (which gives the empirical
    observed quartiles of those same prints). Backward-looking history vs.
    today's market-implied price — together they answer "what did this
    name historically do AND what is the market currently pricing for
    tomorrow's print?".

    Composes ``build_event_calendar`` for the held set verbatim (single
    source of truth, AGENTS.md #10) so this endpoint, ``/api/earnings-
    shock``, ``/api/earnings-distribution`` and ``/api/event-calendar`` can
    never disagree on what counts as held-imminent. Per-name options chain
    is pulled live from yfinance via ``market.get_options_chain`` at the
    expiry closest to ``ceil(days_away)`` so the chain *captures* the
    event. The builder is pure; the I/O seam degrades each row honestly to
    ``NO_CHAIN``/``NO_QUOTES`` on a yfinance miss or thin chain.
    Observational only — never gates Opus, never injected into the
    decision prompt (AGENTS.md #2/#12 — the ``earnings_shock`` /
    ``stress_scenarios`` precedent).

    SWR-cached 5 min: matching ``/api/earnings-shock``'s cadence; an
    options chain is 1-2s per held name on a cold call."""
    try:
        from .analytics.event_calendar import build_event_calendar
        from .analytics.implied_move import build_implied_move
        from . import market as _market
        store = get_store()
        pf = store.get_portfolio()
        positions = store.open_positions()
        try:
            from .strategy import WATCHLIST as _WATCHLIST
            watch = {t.upper() for t in _WATCHLIST}
        except Exception:
            watch = set()
        held = {(p.get("ticker") or "").upper()
                for p in positions if p.get("ticker")}
        ec = build_event_calendar(positions, held | watch)

        def _chain_provider(ticker: str, dte: int):
            try:
                return _market.get_options_chain(ticker, target_dte=dte)
            except Exception:
                return None

        result = build_implied_move(
            positions,
            float(pf.get("total_value") or 0.0),
            ec,
            options_provider=_chain_provider,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/earnings-war-room")
@swr_cached("earnings-war-room", 300.0)
def earnings_war_room_api():
    """Pre-print game plan: one composite view answering *"if this name gaps
    by its implied move tomorrow, what does my book actually look like
    after?"*. Six existing surfaces each own one slice of this question
    (``/api/event-calendar`` who+when, ``/api/implied-move`` market-priced
    straddle, ``/api/earnings-shock`` historical σ, ``/api/stress-scenarios``
    single-name −10 %, ``/api/sector-exposure`` current concentration,
    ``/api/recovery`` path-to-even). None composes the worst-case projection:
    post-shock book value vs the $1000 start, post-shock concentration, total
    $-at-risk across all held imminent prints. This is that composer.

    Composes the sibling builders' results **verbatim** (single source of
    truth, AGENTS.md #10) — no new math beyond `position_value × shock_pct`
    arithmetic and the post-shock book-value projection. Per-event tier is
    ``max(|implied_book_pct|, |sigma_book_pct|)`` so a chain miss still
    tiers off historical σ and an IPO-name with no prior prints still tiers
    off the implied straddle. Observational only — never gates Opus, never
    injected into the decision prompt (AGENTS.md #2/#12 — the
    ``earnings_shock`` / ``stress_scenarios`` / ``recovery`` precedent).

    SWR-cached 5 min: matches the sibling earnings endpoints' cadence."""
    try:
        from .analytics.event_calendar import build_event_calendar
        from .analytics.earnings_war_room import build_earnings_war_room
        from .analytics.earnings_shock import build_earnings_shock
        from .analytics.implied_move import build_implied_move
        from .analytics.stress_scenarios import build_stress_scenarios
        from .analytics.sector_exposure import classify as _classify
        from . import market as _market
        store = get_store()
        pf = store.get_portfolio()
        positions = store.open_positions()
        try:
            from .strategy import WATCHLIST as _WATCHLIST
            watch = {t.upper() for t in _WATCHLIST}
        except Exception:
            watch = set()
        held = {(p.get("ticker") or "").upper()
                for p in positions if p.get("ticker")}
        total_value = float(pf.get("total_value") or 0.0)

        ec = build_event_calendar(positions, held | watch)
        shock = build_earnings_shock(
            positions, total_value, ec,
            history_provider=_earnings_history_for,
        )

        def _chain_provider(ticker: str, dte: int):
            try:
                return _market.get_options_chain(ticker, target_dte=dte)
            except Exception:
                return None

        implied = build_implied_move(
            positions, total_value, ec,
            options_provider=_chain_provider,
        )
        stress = build_stress_scenarios(
            positions, total_value, _classify, _LEVERAGE_BETA,
        )

        result = build_earnings_war_room(
            positions,
            total_value,
            INITIAL_CASH,
            ec,
            implied_move_result=implied,
            earnings_shock_result=shock,
            stress_scenarios_result=stress,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/event-protection")
@swr_cached("event-protection", 300.0)
def event_protection_api():
    """Sized trim + ATM-put-hedge plan for each held name that reports inside
    the implied-move horizon. The **quantitative complement** to
    ``/api/implied-move`` (which prices the market's 1σ move) and
    ``/api/position-action-brief`` (which gives a qualitative ``TRIM_BEFORE_
    EVENT`` / ``HOLD_THROUGH_EVENT`` verdict). Together they answer the
    operator's pre-print question: *"how many shares do I trim to cap 1σ
    downside at X% of book, and what would the equivalent put hedge cost?"*

    Composes ``build_implied_move`` verbatim (single source of truth, AGENTS.md
    #10) so this endpoint, ``/api/implied-move`` and ``/api/earnings-shock``
    can never disagree on per-event 1σ. The builder is pure — no I/O — and
    degrades each row honestly to ``NO_IMPLIED`` when the chain hint is
    unavailable rather than fabricating sizing off a zero σ.

    Accepts ``?target_pct=<float>`` to override the default 5% 1σ-book cap
    (clamped to (0, 100]). Observational only — never gates Opus, never
    injected into the decision prompt (AGENTS.md #2/#12 — the ``implied_
    move`` / ``stress_scenarios`` precedent).

    SWR-cached 5 min: matches ``/api/implied-move``'s cadence since the
    upstream options-chain pull dominates cost."""
    try:
        from .analytics.event_calendar import build_event_calendar
        from .analytics.event_protection import (
            DEFAULT_TARGET_MAX_1SIGMA_PCT,
            build_event_protection_plan,
        )
        from .analytics.implied_move import build_implied_move
        from . import market as _market
        try:
            target_pct = float(request.args.get(
                "target_pct", DEFAULT_TARGET_MAX_1SIGMA_PCT))
        except (TypeError, ValueError):
            target_pct = DEFAULT_TARGET_MAX_1SIGMA_PCT
        if not (0.0 < target_pct <= 100.0):
            target_pct = DEFAULT_TARGET_MAX_1SIGMA_PCT

        store = get_store()
        pf = store.get_portfolio()
        positions = store.open_positions()
        try:
            from .strategy import WATCHLIST as _WATCHLIST
            watch = {t.upper() for t in _WATCHLIST}
        except Exception:
            watch = set()
        held = {(p.get("ticker") or "").upper()
                for p in positions if p.get("ticker")}
        total_value = float(pf.get("total_value") or 0.0)

        ec = build_event_calendar(positions, held | watch)

        def _chain_provider(ticker: str, dte: int):
            try:
                return _market.get_options_chain(ticker, target_dte=dte)
            except Exception:
                return None

        implied = build_implied_move(
            positions, total_value, ec,
            options_provider=_chain_provider,
        )
        result = build_event_protection_plan(
            positions,
            total_value,
            implied,
            target_max_1sigma_loss_pct=target_pct,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/macro-calendar")
def macro_calendar_api():
    """The exact forward FOMC rate-decision awareness block the live trader
    now sees in its decision prompt (the `event_calendar` / `risk_mirror`
    prompt↔endpoint parity discipline). Pure static-table builder — no disk,
    no `:8080` hop, no store read (it is market-wide, not per-ticker).
    Observational only; never gates Opus."""
    try:
        from .analytics.macro_calendar import build_macro_calendar
        return jsonify(build_macro_calendar())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/sector-exposure")
def sector_exposure_api():
    """The exact live-book sector-concentration block the live trader now
    sees in its decision prompt (the `risk_mirror` / `event_calendar`
    prompt↔endpoint parity discipline). Pure arithmetic over the same store
    snapshot + the verbatim-copied SECTOR_MAP, so this is numerically
    identical to `/api/analytics` `sector_exposure_pct` (single source of
    truth, AGENTS.md #10). Observational only; never gates Opus."""
    try:
        from .analytics.sector_exposure import build_sector_exposure
        store = get_store()
        pf = store.get_portfolio()
        positions = store.open_positions()
        snap = {
            "cash": float(pf.get("cash") or 0.0),
            "total_value": float(pf.get("total_value") or 0.0),
            "positions": positions,
        }
        try:
            from .strategy import WATCHLIST as _WATCHLIST, _names_in_play
            names = _names_in_play(positions, [], _WATCHLIST)
        except Exception:
            names = {(p.get("ticker") or "").upper()
                     for p in positions if p.get("ticker")}
        return jsonify(build_sector_exposure(snap, names))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/sector-signal-fit")
def sector_signal_fit_api():
    """Per-sector position weight vs. live-signal density — the "am I
    allocated where the wire is pointing?" answer.

    Composes ``/api/sector-exposure``'s position weights with a sector-
    weighted ai_score rollup of the last ``hours`` (default 6) of live
    signals (``signals.get_top_signals``) and reports per-sector
    OVERWEIGHT (long but wire is quiet) / UNDERWEIGHT (signals strong but
    no position) / ALIGNED, plus the top-level state and the most-divergent
    sector. Signal ticker classification reuses ``sector_exposure.classify``
    so the position and signal columns share one SECTOR_MAP (drift-locked).
    Observational only; never gates Opus."""
    try:
        from .analytics.sector_exposure import build_sector_exposure
        from .analytics.sector_signal_fit import (
            build_sector_signal_fit, GAP_THRESHOLD_PCT,
        )
        from . import signals as _sig

        try:
            hours = int(request.args.get("hours", 6))
        except (TypeError, ValueError):
            hours = 6
        hours = max(1, min(168, hours))
        try:
            min_score = float(request.args.get("min_score", 4.0))
        except (TypeError, ValueError):
            min_score = 4.0
        try:
            gap_threshold = float(
                request.args.get("gap_threshold_pct", GAP_THRESHOLD_PCT))
        except (TypeError, ValueError):
            gap_threshold = GAP_THRESHOLD_PCT
        try:
            n_max = int(request.args.get("n", 80))
        except (TypeError, ValueError):
            n_max = 80
        n_max = max(1, min(500, n_max))

        store = get_store()
        pf = store.get_portfolio()
        positions = store.open_positions()
        snap = {
            "cash": float(pf.get("cash") or 0.0),
            "total_value": float(pf.get("total_value") or 0.0),
            "positions": positions,
        }
        try:
            from .strategy import WATCHLIST as _WATCHLIST, _names_in_play
            names = _names_in_play(positions, [], _WATCHLIST)
        except Exception:
            names = {(p.get("ticker") or "").upper()
                     for p in positions if p.get("ticker")}
        exposure = build_sector_exposure(snap, names)
        sigs = _sig.get_top_signals(n=n_max, hours=hours, min_score=min_score)
        fit = build_sector_signal_fit(
            exposure, sigs, gap_threshold_pct=gap_threshold)
        # Echo input knobs so the response is self-describing.
        fit["window_hours"] = hours
        fit["min_score"] = min_score
        fit["gap_threshold_pct"] = gap_threshold
        fit["n_signals_input"] = len(sigs)
        return jsonify(fit)
    except Exception as e:
        return jsonify({"error": str(e), "state": "NO_DATA",
                        "sectors": []}), 500


# ───────── Feature-dev additions (2026-05-15, agent 4) ─────────
# /api/scorer-confidence — empirical ± bands + directional hit-rate for the
#                          DecisionScorer, so its point predictions can be
#                          trusted (or distrusted) with a real error bar.
# /api/decision-health   — is the live Opus trader actually deciding? Surfaces
#                          the NO_DECISION (parse-failure) rate the dashboard
#                          otherwise hides entirely.


def _live_scorer_predictions(scorer) -> list[dict]:
    """Predicted 5d return for each held stock position (live feature vector).

    Same feature construction as ``/api/scorer-predictions`` — kept as a shared
    helper so the confidence endpoint stays in lockstep with the original."""
    from .strategy import get_quant_signals_live
    from . import signals as _sig

    store = get_store()
    held = sorted({
        p["ticker"] for p in store.open_positions()
        if p.get("type") == "stock" and (p.get("qty") or 0) > 0
    })
    if not held:
        return []
    quant = get_quant_signals_live(held) or {}
    sent_by_tk = {s["ticker"]: s for s in (_sig.ticker_sentiments(held, hours=4) or [])}
    regime_mult = 1.0
    try:
        spy_mom = (get_quant_signals_live(["SPY"]).get("SPY") or {}).get("mom_5d")
        if isinstance(spy_mom, (int, float)):
            regime_mult = max(0.7, min(1.3, 1.0 + spy_mom * 0.075))
    except Exception:
        pass
    preds = []
    for tk in held:
        q = quant.get(tk) or {}
        sent = sent_by_tk.get(tk) or {}
        ml_score = float(sent.get("max_score") or 0.0)
        meta = scorer.predict_with_meta(
            ml_score=ml_score, rsi=q.get("rsi"), macd=q.get("macd_signal"),
            mom5=q.get("mom_5d"), mom20=q.get("mom_20d"), regime_mult=regime_mult,
            ticker=tk, vol_ratio=q.get("vol_ratio"), bb_pos=q.get("bb_position"),
        )
        pred = meta["pred"]
        row = {
            "ticker": tk,
            "pred_5d_return_pct": round(float(pred), 3),
            "verdict": _scorer_verdict(float(pred)),
            "rsi": q.get("RSI"), "mom_5d": q.get("mom_5d"), "mom_20d": q.get("mom_20d"),
            # Honesty flag: True ⇒ the model extrapolated past the empirical
            # label support, pred is a clamped floor/ceiling, and the verdict
            # should be read as "weak/low-trust", not a confident -50%.
            "off_distribution": bool(meta["off_distribution"]),
        }
        if meta["off_distribution"]:
            row["raw_pred_5d_return_pct"] = round(float(meta["raw"]), 3)
        preds.append(row)
    return preds


def _load_decision_outcomes(max_rows: int = 4000) -> list[dict]:
    """Tail of data/decision_outcomes.jsonl — the scorer's own training history."""
    import json as _json
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "data" / "decision_outcomes.jsonl"
    if not path.exists():
        return []
    rows: list[dict] = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rows.append(_json.loads(ln))
        except Exception:
            continue
    return rows[-max_rows:]


@app.route("/api/scorer-confidence")
@swr_cached("scorer-confidence", 90.0)
def scorer_confidence_api():
    """Empirical prediction intervals + reliability for the DecisionScorer.

    Replays the trained scorer over its own outcome history to measure how far
    its predictions actually land from realized returns. Returns a calibration
    table (residual P10/P50/P90 + directional hit-rate per prediction band) and,
    for each held stock position, the live prediction wrapped in an empirical
    [low, high] band drawn from the matching band's residual quantiles."""
    try:
        from .ml.decision_scorer import DecisionScorer
        from .analytics.scorer_confidence import build_scorer_confidence, interval_for

        scorer = DecisionScorer()
        outcomes = _load_decision_outcomes()
        conf = build_scorer_confidence(outcomes, scorer)

        positions = []
        if conf.get("overall"):
            for p in _live_scorer_predictions(scorer):
                iv = interval_for(p["pred_5d_return_pct"], conf)
                positions.append({**p, "interval": iv})
            positions.sort(key=lambda r: -(r["pred_5d_return_pct"] or 0))
        conf["positions"] = positions
        return jsonify(conf)
    except Exception as e:
        return jsonify({"error": str(e), "buckets": [], "positions": []}), 500


@app.route("/api/baseline-compare")
@swr_cached("baseline-compare", 90.0)
def baseline_compare_api():
    """Does the 17-feature DecisionScorer earn its complexity OUT OF SAMPLE?

    A read-only trust diagnostic (never trains, never touches the pickle or a
    trade path) that scores the deployed MLP and a handful of one-line rules
    (raw `ml_score`, momentum carry, RSI/Bollinger mean-reversion) over the
    scorer's own outcome history, on the **temporal-OOS slice**
    (`oos_only=True` — the generalization-relevant view, NOT the in-sample
    one that flatters the net). Two scale-invariant primitives — tie-aware
    Spearman `rank_ic` and `dir_acc` — decide the verdict:

      MLP_ADDS_SKILL              the net beats every one-liner AND clears its
                                  own rank-skill floor — complexity justified
      MLP_NO_BETTER_THAN_TRIVIAL  a single feature carries the same edge —
                                  the gate's sizing variance (invariant #5)
                                  is unjustified by anything the MLP uniquely
                                  contributes
      MLP_WORSE_THAN_TRIVIAL      a one-liner beats the net outright
      INSUFFICIENT_DATA           untrained, or < MIN_PAIRS OOS pairs

    This surfaces, on the dashboard and (via the digital-intern chat
    cross-fetch) in the analyst's mouth, the honesty signal that was
    previously only reachable through `python3 -m
    paper_trader.ml.baseline_compare` — a CLI no operator runs. It is a
    diagnostic, not a recommendation: invariant #5 keeps the gate live at
    n_train ≥ 500 regardless of this verdict; the value is *knowing* the
    gate is modulating real position sizing on noise."""
    try:
        from .ml.decision_scorer import DecisionScorer
        from .ml.baseline_compare import scorer_baseline_compare

        scorer = DecisionScorer()
        outcomes = _load_decision_outcomes()
        rep = scorer_baseline_compare(scorer, outcomes, oos_only=True)
        rep.setdefault("n_train", getattr(scorer, "n_train", None))
        return jsonify(rep)
    except Exception as e:
        return jsonify({"error": str(e), "status": "error",
                        "verdict": "INSUFFICIENT_DATA", "baselines": [],
                        "mlp": {"rank_ic": None, "dir_acc": None, "n": 0},
                        "best_baseline": None, "best_baseline_ic": None,
                        "ic_gap": None, "hint": f"endpoint fault: {e}",
                        "slice": "oos", "n_records_considered": 0,
                        "n_train": None}), 200


def _parse_action_ticker(action_taken: str) -> tuple[str, str | None]:
    """Pull the (verb, ticker) out of a decisions.action_taken string.

    The column is free-text in the form 'BUY NVDA → FILLED' / 'HOLD MU → HOLD'
    / 'NO_DECISION'. Returns ('NO_DECISION', None) for malformed / sentinel
    rows so callers don't have to special-case them."""
    if not action_taken or action_taken in ("NO_DECISION", "BLOCKED"):
        return action_taken or "", None
    head = action_taken.split("→")[0].strip()
    parts = head.split()
    if not parts:
        return "", None
    verb = parts[0].upper()
    ticker = parts[1].upper() if len(parts) >= 2 else None
    if ticker in ("CASH", "NONE", ""):
        ticker = None
    return verb, ticker


_BUY_VERBS = {"BUY", "BUY_CALL", "BUY_PUT", "REBALANCE"}
_SELL_VERBS = {"SELL", "SELL_CALL", "SELL_PUT"}


def _classify_disagreement(verdict: str, last_verb: str | None) -> tuple[str, str]:
    """Map (scorer verdict, last Opus action verb on the same ticker) → (severity, label).

    HIGH = scorer says EXIT/TRIM while Opus is still adding or holding —
           the trader is fighting its own ML safety net.
    MEDIUM = scorer says NEUTRAL but Opus is leaning bullish, or scorer says
             STRONG_HOLD but Opus just sold.
    ALIGNED = the two agree (either both bullish or both bearish)."""
    verdict = (verdict or "").upper()
    verb = (last_verb or "").upper()
    bearish_scorer = verdict in ("EXIT", "TRIM")
    bullish_scorer = verdict in ("STRONG_HOLD", "HOLD")
    bullish_action = verb in _BUY_VERBS or verb == "HOLD"
    bearish_action = verb in _SELL_VERBS
    if bearish_scorer and bullish_action:
        return "HIGH", "scorer says exit, Opus still long"
    if bullish_scorer and bearish_action:
        return "MEDIUM", "scorer says hold, Opus exited"
    if verdict == "NEUTRAL" and verb in _BUY_VERBS:
        return "MEDIUM", "scorer neutral, Opus added"
    return "ALIGNED", "scorer and Opus agree"


@app.route("/api/disagreement")
def disagreement_api():
    """Where the scorer and Opus diverge on currently-held positions.

    For every open stock position, compare the scorer's verdict (drawn from
    /api/scorer-confidence's empirical-band logic) against the most recent
    parsed action that Opus took on the same ticker. A HIGH-severity row is
    a red flag: the trader is overriding the ML safety net. Used by the
    command-center and intended as a 'why is the portfolio losing money?'
    diagnostic when scorer/Opus drift apart silently."""
    try:
        from .ml.decision_scorer import DecisionScorer
        from .analytics.scorer_confidence import build_scorer_confidence, interval_for

        scorer = DecisionScorer()
        outcomes = _load_decision_outcomes()
        conf = build_scorer_confidence(outcomes, scorer)
        scorer_rows = _live_scorer_predictions(scorer) if conf.get("overall") else []

        # Last action verb per ticker — walk recent decisions newest first so we
        # capture the most recent Opus stance on each holding. Skip NO_DECISION
        # rows so a parse-failure storm doesn't blank the panel.
        store = get_store()
        last_verb: dict[str, str] = {}
        last_ts: dict[str, str] = {}
        for d in store.recent_decisions(limit=500):
            verb, tk = _parse_action_ticker(d.get("action_taken") or "")
            if not tk or verb == "NO_DECISION":
                continue
            if tk in last_verb:
                continue
            last_verb[tk] = verb
            last_ts[tk] = d.get("timestamp") or ""

        rows = []
        for p in scorer_rows:
            tk = p["ticker"]
            verb = last_verb.get(tk)
            severity, label = _classify_disagreement(p.get("verdict", ""), verb)
            iv = interval_for(p["pred_5d_return_pct"], conf) if conf.get("overall") else None
            drow = {
                "ticker": tk,
                "scorer_verdict": p.get("verdict"),
                "scorer_pred_5d_pct": p.get("pred_5d_return_pct"),
                "last_action": verb,
                "last_action_ts": last_ts.get(tk),
                "severity": severity,
                "label": label,
                "interval": iv,
                # Carry the honesty flag through so a HIGH-severity row
                # driven by a clamped extrapolation can be visually
                # de-weighted rather than read as a real scorer/Opus fight.
                "off_distribution": bool(p.get("off_distribution", False)),
            }
            if p.get("off_distribution"):
                drow["raw_pred_5d_return_pct"] = p.get("raw_pred_5d_return_pct")
            rows.append(drow)
        severity_order = {"HIGH": 0, "MEDIUM": 1, "ALIGNED": 2}
        rows.sort(key=lambda r: (severity_order.get(r["severity"], 9),
                                 r["scorer_pred_5d_pct"] or 0))
        counts = {"HIGH": 0, "MEDIUM": 0, "ALIGNED": 0}
        for r in rows:
            counts[r["severity"]] = counts.get(r["severity"], 0) + 1
        return jsonify({
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "scorer_trained": bool(conf.get("overall")),
            "n_positions": len(rows),
            "counts": counts,
            "rows": rows,
        })
    except Exception as e:
        return jsonify({"error": str(e), "rows": []}), 500


@app.route("/api/shadow-vs-claude")
def shadow_vs_claude_api():
    """Right-now snapshot: top deterministic shadow rec vs the most recent
    Claude decision. Identifies ``MISSED_OPPORTUNITY`` when Claude was
    ``NO_DECISION`` and the rules engine currently has a strong directional
    BUY/ADD/TRIM/EXIT.

    ``/api/empty-claude-rate`` / ``/api/host-guard`` already surface *that the
    box is starved* — but they say nothing about *what the bot would have done
    if a decision had come back*. This endpoint joins ``/api/suggestions``
    (the deterministic ``_classify_action`` rules engine, the closest thing
    the trader has to a "shadow decision") with the most recent row in the
    ``decisions`` table and emits a verdict:

    * ``MISSED_OPPORTUNITY`` — Claude NO_DECISION while shadow has a
      strong (conviction ≥ 0.7) directional rec. The operationally meaningful
      case to act on manually.
    * ``DROUGHT_OK`` — Claude NO_DECISION but shadow is quiet too.
    * ``ALIGNED`` — Claude and shadow agree on the same directional call.
    * ``DIVERGENT`` — both produced directional calls, they disagree.
    * ``CLAUDE_HOLDS`` — Claude said HOLD while shadow flags a directional rec.
    * ``NO_CLAUDE_DATA`` / ``NO_SHADOW_DATA`` — degraded inputs.

    Snapshot-only by construction (per advisor): the inputs are produced from
    different points in time (last decision can be minutes-to-hours old; the
    suggestion list is current), so this endpoint deliberately does **not**
    compute "agreement %" over a historical window — that comparison would
    be incoherent (signals at decision time ≠ signals now). For the
    aggregate-over-time view of decisions see ``/api/decision-health``.

    Observational only — never gates Opus (invariants #2/#12)."""
    try:
        from .analytics.shadow_vs_claude import build_shadow_vs_claude
        # Reuse /api/suggestions verbatim — single source of truth for the
        # deterministic rules engine. Same pattern as /api/funded-suggestions.
        resp = suggestions_api()
        if isinstance(resp, tuple):
            resp = resp[0]
        sug_payload = resp.get_json(silent=True) or {}
        suggestions = sug_payload.get("suggestions", [])
        store = get_store()
        recent = store.recent_decisions(limit=1) or []
        last_decision = recent[0] if recent else None
        result = build_shadow_vs_claude(suggestions, last_decision)
        # Surface a suggestions-side error rather than masking it as
        # "NO_SHADOW_DATA" — mirrors funded_suggestions' error pass-through.
        if sug_payload.get("error"):
            result["suggestions_error"] = sug_payload["error"]
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/validation")
def validation_api():
    """Signal Integrity validation results — permutation tests + label audits.

    Backed by data/validation_results.json which is appended to by the
    continuous loop's background validation runner. Returns the full history
    (capped at 50 entries on the writer side); the dashboard renders the
    most recent entry."""
    p = Path(__file__).resolve().parent.parent / "data" / "validation_results.json"
    if not p.exists():
        return jsonify({"results": []})
    try:
        return jsonify({"results": json.loads(p.read_text())})
    except Exception as e:
        return jsonify({"results": [], "error": str(e)}), 500


@app.route("/api/decision-health")
@swr_cached("decision-health", 30.0)
def decision_health_api():
    """Health of the live decision pipeline — action mix, parse-failure rate,
    confidence trend, cadence. Surfaces NO_DECISION ('claude returned no
    parseable JSON') cycles that no other dashboard panel exposes."""
    try:
        from .analytics.decision_health import build_decision_health
        decisions = get_store().recent_decisions(limit=2000)
        rep = build_decision_health(decisions)
        # Surface a rolling-24h NO_DECISION rate as a top-level convenience
        # field so monitoring/alerting uses the *current* parse-failure rate,
        # not the all-time `action_mix` % that legacy failures permanently
        # inflate. The 24h window is already computed in build_decision_health
        # (windows.24h); we just hoist it. `_enough` mirrors the verdict
        # logic's >=10-sample gate so a fresh restart (few/no 24h decisions)
        # doesn't silently clear the alert via a tiny, noisy sample.
        try:
            w24 = (rep.get("windows") or {}).get("24h") or {}
            total24 = int(w24.get("total") or 0)
            nd24 = int(w24.get("no_decision") or 0)
            rep["no_decision_rate_24h"] = float(w24.get("parse_fail_pct") or 0.0)
            rep["no_decision_n_24h"] = nd24
            rep["n_decisions_24h"] = total24
            # True only when there is enough recent signal to trust the rate.
            rep["no_decision_24h_significant"] = total24 >= 10
        except Exception:
            rep["no_decision_rate_24h"] = None
            rep["no_decision_n_24h"] = None
            rep["n_decisions_24h"] = None
            rep["no_decision_24h_significant"] = False
        return jsonify(rep)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/decision-pace")
@swr_cached("decision-pace", 30.0)
def decision_pace_api():
    """Rolling inter-decision latency distribution — is the runner cycling on
    cadence? Splits gaps by market_open and reports per-window p50/p95/max so
    a stalled cycle shows as a tail, not as a flat decisions/day aggregate.

    Closes the gap between /api/decision-health (one decisions/day scalar) and
    /api/runner-heartbeat (single-sample 'is the loop alive *now*' check)."""
    try:
        from .analytics.decision_pace import build_decision_pace
        decisions = get_store().recent_decisions(limit=2000)
        return jsonify(build_decision_pace(decisions))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/decision-paralysis")
@swr_cached("decision-paralysis", 30.0)
def decision_paralysis_api():
    """Consecutive HOLD streak detector — the "HOLD_LOCK" pathology.

    Distinct from ``/api/runner-heartbeat`` (NO_DECISION storms only) and
    ``/api/decision-health`` (24h HOLD% aggregate, not contiguous runs).
    A 95% HOLD share looks identical whether spread across the day or
    stacked into a single immovable block; this surface only flags the
    block. See ``analytics/decision_paralysis.py`` docstring for the
    full verdict ladder (IDLE_STORM / HOLD_LOCK / PASSIVE_LOOP / ACTIVE)
    and threshold rationale.

    Pure builder over ``store.recent_decisions``; observational only
    (AGENTS.md invariants #2/#12)."""
    try:
        from .analytics.decision_paralysis import build_decision_paralysis
        decisions = get_store().recent_decisions(limit=500)
        return jsonify(build_decision_paralysis(decisions))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/watchlist-coverage")
@swr_cached("watchlist-coverage", 60.0)
def watchlist_coverage_api():
    """Per-ticker attention scan over the recent decision stream.

    Surfaces watchlist tickers the bot has stopped attending to — every
    other panel is position-centric (it shows what was traded) and never
    names a ticker that was ignored. Distinct from
    ``/api/ticker-decision-mix`` (per-ticker mix only for tickers that
    *appear* in decisions), ``/api/watchlist-opportunities``
    (forward-looking news heat), and ``/api/rising-unheld-themes``
    (theme surface, not per-ticker attention). See
    ``analytics/watchlist_coverage.py`` for the verdict ladder
    (STAGNANT / CONCENTRATED / DIVERSIFIED / NO_DATA).

    Pure builder over ``strategy.WATCHLIST`` +
    ``store.recent_decisions``; observational only (AGENTS.md
    invariants #2/#12)."""
    try:
        from .analytics.watchlist_coverage import build_watchlist_coverage
        from .strategy import WATCHLIST
        decisions = get_store().recent_decisions(limit=2000)
        return jsonify(build_watchlist_coverage(WATCHLIST, decisions))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/realized-vs-unrealized")
@swr_cached("realized-vs-unrealized", 30.0)
def realized_vs_unrealized_api():
    """Banked-vs-paper P&L split — the "is my gain locked in or
    evaporable?" surface.

    Every other equity / P&L endpoint answers a different question
    (``/api/portfolio`` scalar, ``/api/drawdown`` peak-to-trough,
    ``/api/equity-integrity`` sanity, ``/api/pnl-attribution`` β +
    idiosyncratic, ``/api/trade-asymmetry`` closed-round-trip
    aggregates). This one walks the trade ledger chronologically to
    reproduce the running cost basis, attaches cumulative realized P&L
    to every equity-curve point, and emits a verdict ladder
    (DRAWING_DOWN / LEAKING_PAPER / PAPER_HEAVY / BANKED / BALANCED /
    NO_DATA). See ``analytics/realized_vs_unrealized.py`` docstring for
    threshold rationale.

    Pure builder over ``store.recent_trades`` + ``store.equity_curve``;
    observational only (AGENTS.md invariants #2/#12)."""
    try:
        from .analytics.realized_vs_unrealized import build_realized_vs_unrealized
        store = get_store()
        trades = list(reversed(store.recent_trades(limit=5000)))  # oldest→newest
        curve = store.equity_curve(limit=2000)
        return jsonify(build_realized_vs_unrealized(
            trades, curve, starting_value=INITIAL_CASH))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/concentration-trajectory")
@swr_cached("concentration-trajectory", 120.0)
def concentration_trajectory_api():
    """Daily-snapshot concentration trajectory — the missing slope view.

    Every existing concentration surface (``/api/risk`` ``top1_pct``,
    ``/api/analytics`` ``concentration_top1_pct``, the risk-mirror prompt
    block, ``analytics/correlation``) is point-in-time. None answers
    *over the past N days, has single-name concentration been rising,
    falling, or steady?* — the first-derivative question that
    discriminates a slow ramp into a single name (operator drift) from a
    one-cycle spike (a single fill blew up exposure).

    Walks ``store.recent_trades`` chronologically, snapshots the open
    book at the close of each of the last ``?days=`` (3..30, default 14)
    calendar days, marks each position to that day's close via
    ``_daily_history_cached``, and emits the daily series + a verdict
    ladder (CONCENTRATION_SPIKE / RAMPING_UP / DECONCENTRATING /
    CONCENTRATED_STEADY / DIVERSIFIED / BALANCED / INSUFFICIENT_DATA /
    NO_DATA). Stocks-only by deliberate carve-out (options excluded from
    the concentration math — same discipline as
    ``correlation.build_correlation``).

    Read-only / observational; never gates Opus (AGENTS.md
    invariants #2/#12)."""
    try:
        from .analytics.concentration_trajectory import (
            MAX_SNAPSHOTS,
            MIN_SNAPSHOTS,
            build_concentration_trajectory,
        )
        days = int(request.args.get("days", 14))
        days = max(MIN_SNAPSHOTS, min(MAX_SNAPSHOTS, days))
        store = get_store()
        trades = list(reversed(store.recent_trades(limit=5000)))  # oldest→newest
        # Collect the unique stock-tickers traded over the window so the
        # daily-close fetch is bounded — a typical book touches ≤ 10
        # tickers over a month, well below `_daily_history_cached`'s TTL
        # cache.
        tickers = set()
        for t in trades:
            if not isinstance(t, dict):
                continue
            if t.get("option_type"):
                continue
            tk = (t.get("ticker") or "").strip().upper()
            if tk:
                tickers.add(tk)
        # Fetch a generous trailing window (3mo) — the helper caches.
        daily_closes = {}
        for tk in tickers:
            try:
                daily_closes[tk] = _daily_history_cached(tk, period="3mo") or []
            except Exception:
                daily_closes[tk] = []
        return jsonify(build_concentration_trajectory(
            trades, daily_closes, window_days=days))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/decision-forensics")
def decision_forensics_api():
    """*Why* the live trader produces no decision — failure-mode taxonomy.

    decision-health says HOW OFTEN parsing fails; this says WHY: timeout vs
    truncation vs prose-wrapping vs fenced vs malformed, the open/closed-market
    split, an hourly trend, retry-exhausted count, an actionable hint, and the
    raw model excerpts strategy.py captured but nothing else surfaces."""
    try:
        from .analytics.decision_forensics import build_decision_forensics
        decisions = get_store().recent_decisions(limit=2000)
        return jsonify(build_decision_forensics(decisions))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/liquidity")
def liquidity_api():
    """Capital deployment & liquidity — is the book pinned with no dry powder?

    Cash vs deployed %, position weights, unrealized P/L, days since the last
    opening trade, and a status (NO_DRY_POWDER / DRY_POWDER_LOW / BALANCED /
    CASH_HEAVY) with human flags. Complements /api/risk (concentration) with
    the 'can the trader still act on a signal?' view."""
    try:
        from .analytics.liquidity import build_liquidity
        store = get_store()
        return jsonify(build_liquidity(
            store.get_portfolio(),
            store.open_positions(),
            store.recent_trades(200),
        ))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/decision-drought")
def decision_drought_api():
    """What the live trader's *inaction* cost — drift during decision droughts.

    decision-health gives the NO_DECISION rate; decision-forensics gives the
    WHY. This gives the COST: between FILLED trades, segment cycles into
    droughts, price each one's portfolio drift vs the S&P from the equity
    curve, and split involuntary (NO_DECISION/parse-failure) PARALYSIS from
    DELIBERATE_HOLD. The negative alpha of the paralysis droughts is
    'involuntary alpha bleed' — the parse-failure problem in P&L terms."""
    try:
        from .analytics.decision_drought import build_decision_drought
        store = get_store()
        return jsonify(build_decision_drought(
            store.recent_decisions(limit=3000),
            store.equity_curve(limit=5000),
        ))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/idle-opportunity")
def idle_opportunity_api():
    """During the current PARALYSIS drought, which high-score live signals
    on the watchlist arrived that the bot never decided against?

    ``/api/decision-drought`` reports the realized portfolio drift while the
    bot was dark — a backward-looking P&L cost. ``/api/shadow-vs-claude``
    reports a right-now deterministic rec vs the (possibly hours-stale)
    last claude decision — snapshot-only by design. Neither answers the
    operator's "did anything HIGH-SCORE actually arrive on a name I follow
    during the dark window I would have acted on?" question. This endpoint
    enumerates the loudest live watchlist articles inside the canonical
    ``build_decision_drought.current_drought`` window (composes verbatim —
    AGENTS.md #10, so the two endpoints can never disagree on what counts as
    an ongoing drought) and buckets them per ticker. An empty result is
    itself informative — ``state="OK"``, ``n_opportunities=0`` —
    silence-when-nothing-actionable (the ``_macro_calendar_chat_lines`` /
    ``_event_readiness_chat_lines`` / ``_host_pulse_line`` precedent).

    Query params (clamped):
      * ``min_ai_score`` — 0..10, default 6.0 (above grey-zone threshold,
        below the 8.0 urgency cap; raise to focus only on near-urgent rows)
      * ``max_opportunities`` — 1..50, default 20

    Reads digital-intern's articles.db read-only through the same
    freshness-aware ``_articles_db_path()`` the other news-IO endpoints use
    (invariant #15/#17 — no split-brain feed). Live-only clause applied
    (invariant #3). Article scan is bounded by the drought window so the
    SQL is bounded even on a high-throughput articles.db (the
    ``news_velocity`` precedent — drought is typically 4-24h so the row
    count is hundreds, not the 1.47M-rows/7d worst case).

    Observational only — never gates Opus, no caps (AGENTS.md #2/#12 — the
    ``shadow_vs_claude`` / ``stress_scenarios`` / ``recovery`` precedent).
    """
    try:
        from .analytics.idle_opportunity import (
            build_idle_opportunity,
            DEFAULT_MIN_AI_SCORE,
            DEFAULT_MAX_OPPORTUNITIES,
        )
        from .analytics.decision_drought import build_decision_drought

        try:
            min_score = float(request.args.get("min_ai_score",
                                               DEFAULT_MIN_AI_SCORE))
        except Exception:
            min_score = DEFAULT_MIN_AI_SCORE
        min_score = max(0.0, min(min_score, 10.0))

        try:
            max_opps = int(request.args.get("max_opportunities",
                                            DEFAULT_MAX_OPPORTUNITIES))
        except Exception:
            max_opps = DEFAULT_MAX_OPPORTUNITIES
        max_opps = max(1, min(max_opps, 50))

        store = get_store()
        # Compose decision_drought verbatim — single source of truth so
        # /api/idle-opportunity and /api/decision-drought can never disagree
        # on whether there is an ongoing drought (AGENTS.md #10).
        dd = build_decision_drought(
            store.recent_decisions(limit=3000),
            store.equity_curve(limit=5000),
        )
        now_utc = datetime.now(timezone.utc)

        from .strategy import WATCHLIST as _WATCHLIST
        watchlist = list(_WATCHLIST)
        held = _stock_tickers_from_positions(store.open_positions())

        # Short-circuit when there's no ongoing drought — saves the
        # articles.db read entirely (the operator-happy path).
        cur = dd.get("current_drought") if isinstance(dd, dict) else None
        if not cur or not cur.get("ongoing"):
            return jsonify(build_idle_opportunity(
                dd, [], watchlist, held_tickers=held, now=now_utc,
                min_ai_score=min_score, max_opportunities=max_opps,
            ))

        drought_start = cur.get("start")
        articles: list[dict] = []
        path = _articles_db_path()
        if path is not None and drought_start:
            # Window-bounded scan — drought_start is the cheapest filter
            # (typically narrows to hundreds of rows). Same live-only
            # clause as every other news-IO endpoint (AGENTS.md #3).
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True,
                                   timeout=5)
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(
                    "SELECT title, ai_score, urgency, first_seen, url, source "
                    "FROM articles WHERE first_seen >= ? "
                    "AND ai_score >= ? "
                    "AND url NOT LIKE 'backtest://%' "
                    "AND source NOT LIKE 'backtest_%' "
                    "AND source NOT LIKE 'opus_annotation%' "
                    "ORDER BY ai_score DESC, first_seen DESC LIMIT 5000",
                    (drought_start, float(min_score)),
                ).fetchall()
            finally:
                conn.close()
            for r in rows:
                articles.append({
                    "title": r["title"] or "",
                    "ai_score": r["ai_score"],
                    "urgency": r["urgency"],
                    "first_seen": r["first_seen"],
                    "url": r["url"],
                    "source": r["source"],
                })

        return jsonify(build_idle_opportunity(
            dd, articles, watchlist, held_tickers=held, now=now_utc,
            min_ai_score=min_score, max_opportunities=max_opps,
        ))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/drought-path-risk")
def drought_path_risk_api():
    """Intra-drought path shape — peak/trough/drawdown/range while the bot
    was frozen.

    ``/api/decision-drought`` reports point-to-point ``port_pct`` /
    ``spy_pct`` / ``alpha_pct`` for each drought — the *endpoint* difference
    between drought start and drought end. That is path-blind: a 47h drought
    that smoothly slid to -2.4% looks identical to one that bottomed at
    -6.5% mid-drought and recovered to -2.4%. The desk's reading of the two
    is NOT the same; the second one is a survived near-miss the trader was
    blind to. This endpoint composes the canonical
    ``build_decision_drought.current_drought`` block (verbatim SSOT —
    AGENTS.md #10, the ``/api/idle-opportunity`` precedent — so the two
    endpoints can never disagree on what counts as an ongoing drought) and
    walks the equity_curve points inside that window to surface the
    missing path-shape view: ``peak_equity`` / ``trough_equity`` /
    ``intra_drought_drawdown_pct`` (peak-to-trough) / ``range_pct`` /
    ``end_to_start_pct``, then classifies into WHIPSAW_TRAP / DODGED_DROP /
    LIFTED_BLIND / SLOW_BLEED / QUIET_DROUGHT / MIXED once
    ``n_equity_samples >= 3`` (STABLE gate). Withholds the verdict (state
    INSUFFICIENT) below the gate so a 1-point drought never gets a path
    label. ``state="NO_DROUGHT"`` collapses to silence — the
    silence-when-nothing-actionable precedent (``_host_pulse_line`` /
    ``_macro_calendar_chat_lines``).

    Observational only — never gates Opus, never injected into the decision
    prompt, no caps (AGENTS.md invariants #2/#12 — the
    ``/api/idle-opportunity`` / ``/api/capital-paralysis`` /
    ``/api/shadow-vs-claude`` precedent).
    """
    try:
        from .analytics.drought_path_risk import build_drought_path_risk
        from .analytics.decision_drought import build_decision_drought
        store = get_store()
        equity = store.equity_curve(limit=5000)
        dd = build_decision_drought(
            store.recent_decisions(limit=3000), equity,
        )
        return jsonify(build_drought_path_risk(dd, equity))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/news-action-funnel")
def news_action_funnel_api():
    """Per-ticker funnel of loud-articles → decisions → fills → P&L over the
    last ``window_hours``.

    Held ∪ ``strategy.WATCHLIST`` are evaluated. For each ticker, count the
    live articles ≥ ``min_ai_score`` in the window, the decisions naming the
    ticker (parsed via the dashboard SSOT ``_parse_action_ticker``), and the
    FILLED trades on that ticker. The per-row verdict surfaces the
    loud-news / no-action pathology (``IGNORED``) FIRST so the operator
    sees the missed names without scrolling — distinct from
    ``/api/idle-opportunity`` (drought-gated by design — silent when the
    bot is filling normally even if a specific name was missed),
    ``/api/signal-followthrough`` (forward 1d/3d/5d edge — can't answer
    current window), and ``/api/trade-attribution`` (reverse direction:
    fills → articles).

    Query params (clamped):
      * ``window_hours`` — 1..72, default 24 (matches the operator headline
        horizon for "did the desk do anything today")
      * ``min_ai_score`` — 0..10, default 6.0 (above the daemon grey-zone
        threshold, below urgency cap; matches ``idle_opportunity`` floor)
      * ``max_tickers`` — 1..100, default 25
      * ``tickers`` — comma-separated override of the universe; default is
        held ∪ strategy.WATCHLIST

    Reads the digital-intern articles.db read-only via the freshness-aware
    ``_articles_db_path()`` (invariant #15/#17 — no split-brain feed).
    Live-only clause applied (AGENTS.md invariant #3). The window scan is
    bounded by a single LIMIT (the ``news_velocity`` precedent — 24h
    typically returns ≤10k rows on the live articles.db).

    Observational only — never gates Opus, never injected into the decision
    prompt, no caps (AGENTS.md #2/#12).
    """
    try:
        from .analytics.news_action_funnel import (
            build_news_action_funnel,
            DEFAULT_MIN_AI_SCORE,
            DEFAULT_WINDOW_HOURS,
            DEFAULT_MAX_TICKERS,
        )

        try:
            window_h = float(request.args.get("window_hours",
                                              DEFAULT_WINDOW_HOURS))
        except Exception:
            window_h = DEFAULT_WINDOW_HOURS
        window_h = max(1.0, min(window_h, 72.0))

        try:
            min_score = float(request.args.get("min_ai_score",
                                               DEFAULT_MIN_AI_SCORE))
        except Exception:
            min_score = DEFAULT_MIN_AI_SCORE
        min_score = max(0.0, min(min_score, 10.0))

        try:
            max_tk = int(request.args.get("max_tickers",
                                          DEFAULT_MAX_TICKERS))
        except Exception:
            max_tk = DEFAULT_MAX_TICKERS
        max_tk = max(1, min(max_tk, 100))

        store = get_store()
        held = _stock_tickers_from_positions(store.open_positions())

        override = (request.args.get("tickers") or "").strip()
        if override:
            universe = [t.strip().upper() for t in override.split(",")
                        if t.strip()]
        else:
            from .strategy import WATCHLIST as _WATCHLIST
            # Held first so they sort earlier among same-verdict rows;
            # builder dedups while preserving order.
            universe = list(held) + [t.upper() for t in _WATCHLIST]

        now_utc = datetime.now(timezone.utc)
        cutoff_iso = (now_utc - timedelta(hours=window_h)).isoformat()

        articles: list[dict] = []
        path = _articles_db_path()
        if path is not None:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True,
                                   timeout=5)
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(
                    "SELECT title, ai_score, urgency, first_seen, url, source "
                    "FROM articles WHERE first_seen >= ? "
                    "AND ai_score >= ? "
                    "AND url NOT LIKE 'backtest://%' "
                    "AND source NOT LIKE 'backtest_%' "
                    "AND source NOT LIKE 'opus_annotation%' "
                    "ORDER BY first_seen DESC LIMIT 12000",
                    (cutoff_iso, float(min_score)),
                ).fetchall()
            finally:
                conn.close()
            for r in rows:
                articles.append({
                    "title": r["title"] or "",
                    "ai_score": r["ai_score"],
                    "urgency": r["urgency"],
                    "first_seen": r["first_seen"],
                    "url": r["url"],
                    "source": r["source"],
                })

        result = build_news_action_funnel(
            articles=articles,
            decisions=store.recent_decisions(limit=2000),
            trades=store.recent_trades(limit=2000),
            positions=store.open_positions(),
            tickers=universe,
            held_tickers=held,
            now=now_utc,
            window_hours=window_h,
            min_ai_score=min_score,
            max_tickers=max_tk,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "state": "ERROR",
                        "tickers": []}), 500


@app.route("/api/position-attention")
def position_attention_api():
    """Per-open-position last-real-Opus-look freshness — which held names has
    the model gone hours without examining?

    decision-health/decision-drought aggregate NO_DECISION cost portfolio-wide;
    thesis-drift re-tests a thesis against current state; hold-discipline
    measures hold time vs the desk's losing-cut history. None answer the
    per-ticker question. When the documented #1 pathology (host-saturation
    NO_DECISION storms) drags on, the live trader silently defaults to holding
    every open lot — but those lots are no longer being *evaluated*. This
    surfaces, per held ticker, when it was last named in a real (non-
    NO_DECISION, non-BLOCKED) decision row, classifies the freshness
    (FRESH ≤2h, MONITORED ≤6h, STALE ≤24h, NEGLECTED >24h or never), and
    rolls up to a NEGLECTED_BOOK/STALE_BOOK/OK verdict. Pure read of
    open_positions + recent_decisions — no network, no Opus invocation.
    Advisory only — never gates Opus, never injected into the decision
    prompt, adds no caps (AGENTS.md #2/#12)."""
    try:
        from .analytics.position_attention import build_position_attention
        store = get_store()
        return jsonify(build_position_attention(
            store.open_positions(),
            store.recent_decisions(limit=3000),
        ))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/position-rationale")
def position_rationale_api():
    """Per-open-position most-recent Opus rationale — *what did Opus actually
    say* about each held name?

    ``position_attention`` answers "did Opus look?" (freshness, neglect).
    ``thesis_drift`` re-tests entry theses. Neither surfaces the **concrete
    current rationale** — the trader's #1 question reviewing the live book:
    *"why am I still holding NVDA? what did Opus actually say last cycle?"*

    The data is already in ``decisions.reasoning`` (JSON envelope written
    by ``strategy.decide()``: ``{"decision": {"reasoning": "...",
    "confidence": x}, ...}``). Today the operator must scroll the decision
    feed and find the most recent row for each ticker by hand. This puts
    the answer one HTTP read away. Pure read of ``open_positions`` +
    ``recent_decisions`` — no network, no Opus invocation. Advisory only —
    never gates Opus, never injected into the decision prompt, adds no
    caps (AGENTS.md #2/#12 — the ``position_attention`` precedent)."""
    try:
        from .analytics.position_rationale import build_position_rationale
        store = get_store()
        return jsonify(build_position_rationale(
            store.open_positions(),
            store.recent_decisions(limit=3000),
        ))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _ticker_news_cooldown(tickers: list[str],
                          min_score: float,
                          hours_window: int = 72) -> dict[str, dict]:
    """Per-ticker last-news bookkeeping for ``/api/position-news-cooldown``.

    Returns ``{TICKER_UPPER: {last_first_seen, top_score, top_title, n_24h,
    n_72h}}`` derived from the live-only article window. Unlike
    ``_ticker_news_pulse`` (which ranks by score), this picks the **most
    recent** scored hit per ticker because the endpoint's question is
    "when did news on this name last move" rather than "what was the
    biggest story". Live-only filter mirrors ``signals.get_top_signals``
    (invariant #1).
    """
    base = {t.upper(): {
        "last_first_seen": None,
        "top_score": None,
        "top_title": None,
        "n_24h": 0,
        "n_72h": 0,
    } for t in tickers}
    if not tickers:
        return base
    path = _articles_db_path()
    if path is None:
        return base
    conn = None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
        conn.row_factory = sqlite3.Row
        now = datetime.now(timezone.utc)
        since = (now - timedelta(hours=hours_window)).isoformat()
        cutoff_24h = (now - timedelta(hours=24)).isoformat()
        rows = conn.execute(
            "SELECT title, full_text, ai_score, first_seen FROM articles "
            "WHERE first_seen >= ? AND ai_score >= ? "
            "AND url NOT LIKE 'backtest://%' "
            "AND source NOT LIKE 'backtest_%' "
            "AND source NOT LIKE 'opus_annotation%' "
            "ORDER BY first_seen DESC LIMIT 4000",
            (since, min_score),
        ).fetchall()
    except Exception:
        return base
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    patterns = {t.upper(): re.compile(rf"(?:\$|\b){re.escape(t.upper())}\b")
                for t in tickers}
    for r in rows:
        body = r["title"] or ""
        if r["full_text"]:
            try:
                body = body + " " + zlib.decompress(
                    r["full_text"]).decode("utf-8", "replace")
            except Exception:
                pass
        body_up = body.upper()
        ts = r["first_seen"]
        score = float(r["ai_score"] or 0.0)
        title = r["title"]
        for t, pat in patterns.items():
            if not pat.search(body_up):
                continue
            rec = base[t]
            rec["n_72h"] += 1
            if ts and (cutoff_24h <= ts):
                rec["n_24h"] += 1
            # Rows are ordered newest-first, so the first hit per ticker is
            # the most-recent scored mention — sets last_first_seen exactly
            # once per (ticker, query).
            if rec["last_first_seen"] is None and ts:
                rec["last_first_seen"] = ts
            # Top-score within the same window (informational; the verdict
            # is keyed off recency, not score).
            cur_top = rec["top_score"]
            if cur_top is None or score > cur_top:
                rec["top_score"] = score
                rec["top_title"] = title
    return base


@app.route("/api/position-news-cooldown")
@swr_cached("position-news-cooldown", 60.0)
def position_news_cooldown_api():
    """Per-open-position news-flow cooldown — has the news desk gone quiet
    on this held ticker, or is the story still moving?

    Distinct from ``/api/position-attention`` (which times *Opus* looks)
    and ``/api/thesis-drift`` (which re-tests the entry rationale). This
    one answers: **for each held name, when was the last live article
    that actually scored above noise (ai_score≥4.0)?** Catches the
    "thesis decay through silence" pathology — a position opened on a
    catalyst whose news flow has dried up while the operator's attention
    moved on. Verdict ladder: per-position FRESH/WARM/COOL/DARK rolled up
    to OK/COOLING_BOOK/DARK_BOOK/INSUFFICIENT_DATA. Advisory only — never
    gates Opus, adds no caps (AGENTS.md #2/#12)."""
    try:
        from .analytics.position_news_cooldown import (
            MIN_SCORE_THRESHOLD,
            build_position_news_cooldown,
        )
        store = get_store()
        positions = store.open_positions()
        tickers = sorted({(p.get("ticker") or "").upper()
                          for p in positions if p.get("ticker")})
        last_news = _ticker_news_cooldown(
            tickers, min_score=MIN_SCORE_THRESHOLD, hours_window=72)
        return jsonify(build_position_news_cooldown(
            positions, last_news, min_score_threshold=MIN_SCORE_THRESHOLD))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/empty-claude-rate")
def empty_claude_rate_api():
    """Live-trader 'claude returned no response (timeout/empty)' rate vs the
    number of concurrent Opus `claude` subprocesses on the host.

    decision-health/-forensics classify *parse* failures generically; this
    endpoint isolates the one NO_DECISION signature that means the model
    subprocess produced **nothing** (timeout / OOM-kill / starvation) and
    correlates it with the current concurrent-Opus process count — the actual
    root cause when out-of-band review agents + the backtest loop saturate the
    box and starve the live trader's claude call. Nothing else surfaces this
    link, so an operator can tell 'bad prompt' apart from 'host overloaded'."""
    try:
        decisions = get_store().recent_decisions(limit=2000)

        def _is_empty(d):
            r = (d.get("reasoning") or "")
            return d.get("action_taken") == "NO_DECISION" and \
                r.startswith("claude returned no response")

        recent = decisions[:200]
        empty_recent = [d for d in recent if _is_empty(d)]

        # 6h time-window slice (timestamps are tz-aware ISO 8601).
        from datetime import datetime, timezone, timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(hours=6)
        win, win_empty, last_ts = 0, 0, None
        for d in decisions:
            try:
                ts = datetime.fromisoformat(str(d.get("timestamp")))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except Exception:
                continue
            if ts < cutoff:
                continue
            win += 1
            if _is_empty(d):
                win_empty += 1
                if last_ts is None or ts > last_ts:
                    last_ts = ts

        mins_since = None
        if last_ts is not None:
            mins_since = round(
                (datetime.now(timezone.utc) - last_ts).total_seconds() / 60.0, 1)

        # Best-effort: count concurrent Opus `claude` subprocesses (the root
        # cause of the empty responses). Never raises — degrades to None.
        concurrent_opus = None
        try:
            import os
            n = 0
            for pid in os.listdir("/proc"):
                if not pid.isdigit():
                    continue
                try:
                    with open(f"/proc/{pid}/cmdline", "rb") as fh:
                        cl = fh.read().replace(b"\x00", b" ").decode(
                            "utf-8", "ignore")
                except (OSError, IOError):
                    continue
                if "claude" in cl and "claude-opus" in cl:
                    n += 1
            concurrent_opus = n
        except Exception:
            concurrent_opus = None

        rate_recent = round(100.0 * len(empty_recent) / len(recent), 1) \
            if recent else 0.0
        rate_6h = round(100.0 * win_empty / win, 1) if win else 0.0

        # Verdict: tie the empty rate to host saturation when both are high.
        if win < 5:
            verdict = "INSUFFICIENT_DATA"
        elif rate_6h >= 50.0 and (concurrent_opus or 0) >= 4:
            verdict = "HOST_SATURATED — live trader starved by concurrent Opus"
        elif rate_6h >= 50.0:
            verdict = "HIGH_EMPTY_RATE — model timing out / returning nothing"
        elif rate_6h >= 15.0:
            verdict = "ELEVATED — intermittent empty responses"
        else:
            verdict = "HEALTHY"

        return jsonify({
            "rate_recent_pct": rate_recent,
            "empty_recent": len(empty_recent),
            "n_recent": len(recent),
            "rate_6h_pct": rate_6h,
            "empty_6h": win_empty,
            "n_6h": win,
            "last_empty_ts": last_ts.isoformat() if last_ts else None,
            "minutes_since_last_empty": mins_since,
            "concurrent_opus_processes": concurrent_opus,
            "verdict": verdict,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/host-guard")
def host_guard_api():
    """Live host-saturation verdict + the NEW 'skipped claude call' bucket.

    The recurring NO_DECISION storms are host saturation, not a prompt/parser
    bug (paper_trader/host_guard.py). strategy.decide() declines the Opus call
    in TWO host-saturation cases, both recorded with the ``skipped claude
    call — …`` prefix this endpoint counts: (1) the pre-flight guard trips
    (``host saturated: …``), and (2) the call passed pre-flight but the box
    saturated *during* it and the doomed Sonnet fallback was skipped
    (``host saturated mid-call: …``). /api/empty-claude-rate keys off the OLD
    ``claude returned no response`` prefix, so once the guard is live an
    operator would see the empty rate fall and wrongly conclude it's fixed —
    when really the skip bucket merely absorbed it (the mid-call case used to
    misreport as an empty/model timeout). This endpoint surfaces both: the raw
    saturation snapshot (host_guard.snapshot — verdict + /proc probe +
    empty-rate) AND the recent skip rate (pre-flight + mid-call combined), so
    'box overloaded' stays visible after the fix. /api/decision-forensics
    splits the two via its HOST_SATURATED_SKIP / HOST_STARVED_MIDCALL modes.
    Read-only, degrade-safe — never raises into the dashboard (mirrors
    /api/empty-claude-rate's contract)."""
    try:
        from . import host_guard
        snap = host_guard.snapshot()

        # Recent deliberate-skip rate — the bucket strategy.decide() now
        # writes when the guard declines the call. Mirrors host_guard's
        # recent_empty_rate shape; degrade-safe (ok=False on any error).
        skip = {"n": 0, "skipped": 0, "rate": 0.0, "ok": False}
        try:
            rows = get_store().recent_decisions(limit=120)
            n = len(rows)
            sk = sum(
                1 for d in rows
                if (d.get("action_taken") or "") == "NO_DECISION"
                and (d.get("reasoning") or "").startswith("skipped claude call")
            )
            skip.update(n=n, skipped=sk,
                        rate=round(sk / n, 3) if n else 0.0, ok=True)
        except Exception:
            pass
        snap["recent_skip_rate"] = skip
        # Additive `pulse` key — the operator-facing freeze-cause SSOT
        # (host_guard.pulse). The SAME dict the Discord _host_pulse_line
        # reads, so the dashboard panel / analyst can never drift from the
        # Discord verdict (the tail_risk / stress_scenarios additive-key
        # precedent). Degrade-safe by construction; the outer try still guards.
        try:
            snap["pulse"] = host_guard.pulse()
        except Exception:
            snap["pulse"] = {"state": "CLEAR", "headline": ""}
        # Additive `starvation_by_cause` — per-cause breakdown of recent
        # NO_DECISION starvation rows (model_timeout / cli_nonzero_rc /
        # model_empty / cli_missing / host_skip / unknown). The aggregate
        # ``recent_empty_rate`` / ``recent_skip_rate`` answer "how many cycles
        # never reached Opus" but NOT "what's the dominant cause" — three
        # classes that need three different actions (see _CAUSE_LABELS in
        # host_guard.py). The ``python3 -m paper_trader.host_guard`` CLI
        # already prints this breakdown; without surfacing it here, an
        # operator hitting the dashboard during a storm sees the rate
        # collapse to a single number with no class signal — exactly the
        # blind-spot host_guard.recent_starvation_by_cause exists to
        # resolve. Degrade-safe (ok=False, all-zero by_cause shape) by the
        # builder's contract — never raises into this endpoint.
        try:
            snap["starvation_by_cause"] = host_guard.recent_starvation_by_cause()
        except Exception:
            snap["starvation_by_cause"] = {
                "n": 0, "starved": 0,
                "by_cause": {label: 0 for label in host_guard._CAUSE_LABELS},
                "rate": 0.0, "ok": False,
            }
        return jsonify(snap)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/capital-paralysis")
@swr_cached("capital-paralysis", 30.0)
def capital_paralysis_api():
    """Trap + cost + unlock — why the book is stuck and the way out.

    /api/liquidity sees the trap (no dry powder), /api/decision-drought sees
    the cost (alpha bled while pinned), /api/suggestions lists ideas it can't
    fund — none connect them. This composes liquidity + decision-drought
    (single source of truth, no re-derived metrics) and adds the unlock
    ladder: positions ranked in desk cut-priority (biggest loser first), each
    rung showing the cash a sale frees, deployed-% after, and whether that
    single sale restores the ability to act on a fresh signal. Advisory only —
    never gates Opus, adds no caps (AGENTS.md invariant #2)."""
    try:
        from .analytics.capital_paralysis import build_capital_paralysis
        store = get_store()
        return jsonify(build_capital_paralysis(
            store.get_portfolio(),
            store.open_positions(),
            store.recent_trades(200),
            store.recent_decisions(limit=3000),
            store.equity_curve(limit=5000),
        ))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/buying-power")
@swr_cached("buying-power", 60.0)
def buying_power_api():
    """Deployable-cash awareness — the lean "what can my free cash fund right
    now?" surface, ALREADY fed into the live Opus decision prompt
    (``strategy.decide()`` builds the same dict and renders ``prompt_block``).
    Surfacing it via the dashboard closes the established prompt→dashboard→
    Discord trajectory the ``buying_power`` block has been following one
    surface at a time (the same path ``capital_paralysis`` already walked).

    Composes ``build_buying_power`` over the read-only portfolio snapshot +
    the freshly-fetched watchlist prices (yfinance bulk — the single slow
    call), scoped to the FULL WATCHLIST so an operator viewing the panel
    sees affordability for every name in the universe, not just the lean
    in-play subset the prompt block trims to. Pure formatting of an existing
    builder's output (single source of truth, AGENTS.md invariant #10):
    cash, deployed_pct, affordable[ticker, price, whole_shares], cheapest
    name + price, and the unlock candidate (biggest-loser-first cut
    priority). Observational only — never gates Opus, never caps a trade
    (AGENTS.md invariants #2/#12 — the ``capital_paralysis`` precedent).

    SWR-cached 60s: the underlying prices.get_prices(WATCHLIST) is the
    same yfinance bulk call the decision cycle already pays for; a 60s
    stale window matches the runner's OPEN_INTERVAL (≥1800s) and is well
    under any operator-perceptible latency. The prewarm == @swr_cached
    invariant (test_swr_prewarm_coverage) keeps this endpoint warm on
    first paint."""
    try:
        from . import market as _market
        from .analytics.buying_power import build_buying_power
        from .strategy import WATCHLIST, portfolio_snapshot_readonly
        store = get_store()
        snap = portfolio_snapshot_readonly(store)
        watch_px = _market.get_prices(WATCHLIST) if WATCHLIST else {}
        # Scope the dashboard view to the WHOLE watchlist (not the lean
        # _names_in_play subset the prompt block uses): an operator on the
        # dashboard wants to see affordability across the full universe,
        # not just "what mattered to Opus this cycle".
        in_play = {t.upper() for t in WATCHLIST}
        rep = build_buying_power(snap, watch_px, in_play)
        return jsonify(rep)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/liquidation-preview")
def liquidation_preview_api():
    """If I closed every open position right now at the live mark, what
    would my book look like?

    The natural complement to ``/api/buying-power`` (one dimension over,
    opposite direction): buying-power says "what can my free cash fund?",
    liquidation-preview says "what would my free cash BECOME if I sold
    everything?". Answers the PM question every pre-earnings de-risk and
    drawdown-trim decision starts with — *cleanly*, instead of forcing
    the operator to mentally re-derive from ``/api/portfolio``.

    Composes ``build_liquidation_preview`` (the pure analytics core)
    over the read-only portfolio snapshot — single source of truth with
    ``_mark_to_market`` (AGENTS.md invariant #10), so the panel's per-
    position realized-PL numbers can never disagree with the live
    snapshot Opus / the hourly summary see.

    Observational only — never gates Opus, never injected into the
    decision prompt, no caps (AGENTS.md #2/#12 — the ``buying_power``
    precedent). Pure read-only: no store writes, no network. Stale
    marks are surfaced per-row so the operator knows when the lock-in
    number is unreliable. Failure contract: any builder fault returns
    HTTP 500 with the message in ``error``, never an uncaught
    exception."""
    try:
        from .analytics.liquidation_preview import build_liquidation_preview
        from .strategy import portfolio_snapshot_readonly
        store = get_store()
        snap = portfolio_snapshot_readonly(store)
        return jsonify(build_liquidation_preview(snap))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/open-attribution")
def open_attribution_api():
    """Selection-vs-market on the *open* book — the bot's dominant return.

    /api/analytics & /api/performance-attribution cover *closed* round-trips,
    but the live trader mostly HOLDs, so its return is dominated by open
    drift vs SPY — invisible until now. Per open stock position: return since
    opened_at, SPY return over the same window (anchored to the equity curve's
    sp500_price at-or-after entry), and alpha in % and $. Options are flagged
    and skipped (alpha-vs-SPY doesn't fit Greeks — /api/backtests/compare
    precedent)."""
    try:
        from .analytics.open_attribution import build_open_attribution
        store = get_store()
        return jsonify(build_open_attribution(
            store.open_positions(),
            store.equity_curve(limit=5000),
        ))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/pnl-attribution")
def pnl_attribution_api():
    """**β-adjusted** decomposition of open unrealized P/L — the honest
    answer to "is my NVDA gain just SPY going up?".

    Complement to ``/api/open-attribution`` (which assumes β=1 and so
    over-attributes "alpha" on a leveraged-ETF/semis book). Per open
    stock position: ``β × SPY_return`` over the same opened_at-anchored
    window vs. residual ``idiosyncratic = position_return − β·SPY``.
    Dollarized using cost basis on both sides. β comes from the
    dashboard/stress_scenarios ``_classify``/``_LEVERAGE_BETA`` SSOT
    (this endpoint is THE true SSOT — the strategy-side pinned copies
    are CI-pinned to these). Options are flagged and skipped (β-attribution
    on a Greeks instrument is its own surface, see ``/api/greeks``).
    Pure, never raises; observational only — never gates Opus, never
    injected into the decision prompt (AGENTS.md #2/#12 — the
    ``open_attribution`` precedent)."""
    try:
        from .analytics.pnl_attribution import build_pnl_attribution
        store = get_store()
        return jsonify(build_pnl_attribution(
            store.open_positions(),
            store.equity_curve(limit=5000),
            _classify,
            _LEVERAGE_BETA,
        ))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/trade-asymmetry")
def trade_asymmetry_api():
    """Behavioural-edge pathology — the exit/sizing failure behind the P&L.

    /api/analytics gives the raw aggregates (win_rate, profit_factor, $ avgs);
    /api/calibration asks whether the confidence axis is accurate. Neither
    answers the desk question: given my payoff ratio, what win-rate do I need
    to break even, am I above or below it, and am I cutting winners faster
    than losers (the disposition effect)? This composes the single source of
    truth (build_round_trips, AGENTS.md #10) into payoff ratio, per-trade
    expectancy, breakeven-vs-actual win-rate, and the winner/loser hold-time
    disposition gap. The verdict label is withheld until n≥20 round-trips
    (news-edge INSUFFICIENT_DATA idiom) so a five-trade read can't mislead.
    Advisory only — never gates Opus, adds no caps (AGENTS.md #2/#12)."""
    try:
        from .analytics.trade_asymmetry import build_trade_asymmetry
        store = get_store()
        # Same trades convention as /api/analytics: oldest → newest.
        trades = list(reversed(store.recent_trades(2000)))
        return jsonify(build_trade_asymmetry(trades))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/kelly-sizing")
def kelly_sizing_api():
    """Kelly-criterion sizing diagnostic — given my realised win-rate and
    payoff ratio, what fraction would Kelly allocate to the single best
    position, and how does my CURRENT top-position weight compare?
    /api/trade-asymmetry already emits payoff_ratio and actual_win_rate_pct
    — this composes them into full-Kelly (p − q/b), half-Kelly (the
    standard practitioner recommendation), and quarter-Kelly, and benchmarks
    the current top-position weight (from /api/risk's concentration_top1)
    against the half-Kelly target. /api/concentration-cap warns at a fixed
    threshold; this answers whether ANY fixed threshold is statistically
    justified by the realised edge. Verdict (UNDERSIZED / KELLY_ALIGNED /
    OVERSIZED / EXTREMELY_OVERSIZED / NEGATIVE_EDGE) is withheld until
    n≥20 round-trips (trade_asymmetry STABLE idiom) and only when a
    payoff ratio is defined (both wins and losses present). Advisory only
    — never gates Opus, adds no caps (AGENTS.md #2/#12)."""
    try:
        from .analytics.kelly_sizing import build_kelly_sizing
        store = get_store()
        # Same trades convention as /api/analytics & /api/trade-asymmetry:
        # oldest → newest so build_round_trips can fold in sequence.
        trades = list(reversed(store.recent_trades(2000)))
        # Top-position weight: derive the same way /api/risk does so the two
        # endpoints agree on what "top position" means. Done inline (not via
        # an HTTP call to /api/risk) so this endpoint stays self-contained
        # and observable when the risk SWR cache is warming.
        positions = store.open_positions()
        pf = store.get_portfolio()
        total_value = float(pf.get("total_value") or 0.0)
        rows, _ = _portfolio_rows_from_positions(positions)
        top1_pct, _, top1_ticker = _conc_top1_top3(rows, total_value)
        return jsonify(build_kelly_sizing(
            trades,
            top_position_pct=top1_pct if rows else None,
            top_position_ticker=top1_ticker,
        ))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/exit-intent-audit")
def exit_intent_audit_api():
    """Exit-intent audit — classify each closed sell by the stated
    *intent* in its exit reason and roll up outcome per intent bucket.
    /api/loser-autopsy classifies losers by an OBJECTIVE failure mode
    (hold-days × magnitude); /api/winner-autopsy looks at entry reasons
    on winners; /api/round-trip-postmortem judges whether the exit was
    well-timed against the next price drift. None classify the trader's
    STATED reason for selling. This module fills that gap: deterministic
    substring matching into EARNINGS_CLEAR / STOP_LOSS / TARGET_HIT /
    THESIS_FLIP / DEFENSIVE_CASH_RAISE / UNCLASSIFIED, then per-bucket
    P&L, win-rate, median hold. Verdict (DOMINANT_INTENT_BLEED /
    DOMINANT_INTENT_HEALTHY / INTENT_UNCLEAR) is withheld until n≥10
    closed round-trips AND the dominant bucket has ≥3 trips, so a thin
    pattern can't mislead. Advisory only — never gates Opus, adds no
    caps (AGENTS.md #2/#12)."""
    try:
        from .analytics.exit_intent_audit import build_exit_intent_audit
        store = get_store()
        # Same trades convention as /api/loser-autopsy & /api/trade-asymmetry:
        # oldest → newest so build_round_trips can fold in sequence.
        trades = list(reversed(store.recent_trades(2000)))
        return jsonify(build_exit_intent_audit(trades))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/loser-autopsy")
def loser_autopsy_api():
    """Per-closed-losing-round-trip post-mortem — the desk question no panel
    answers. /api/thesis-drift re-tests *open* positions against their
    opening rationale; /api/trade-asymmetry gives *aggregate* payoff math;
    /api/churn counts re-entry *cadence*. None tell the story of *why each
    closed trade lost*. This composes the single source of truth
    (build_round_trips, AGENTS.md #10) and joins the verbatim entry/exit
    reason back from the contributing trade rows, classifies each loss into
    an objective failure mode (KNIFE_CATCH / WHIPSAW / SLOW_BLEED /
    STOPPED_OUT), and rolls up which name is the bleed + which mode
    dominates. The pattern verdict is withheld until n≥8 losers
    (trade_asymmetry STABLE idiom). Advisory only — never gates Opus, adds
    no caps (AGENTS.md #2/#12)."""
    try:
        from .analytics.loser_autopsy import build_loser_autopsy
        store = get_store()
        # Same trades convention as /api/analytics & /api/trade-asymmetry:
        # oldest → newest (build_round_trips reads in sequence).
        trades = list(reversed(store.recent_trades(2000)))
        return jsonify(build_loser_autopsy(trades))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/bag-holding-skill")
def bag_holding_skill_api():
    """$-attribution of losses by failure mode — which mode bleeds the
    account most, in dollars. /api/loser-autopsy reports COUNT per mode
    and $ per TICKER; this is the missing $ per MODE view. Composes the
    SSOT (build_round_trips + loser_autopsy._classify, AGENTS.md #10),
    aggregates loss dollars by KNIFE_CATCH / SLOW_BLEED / STOPPED_OUT /
    WHIPSAW, computes the BAG_HOLDING_RATIO (SLOW_BLEED $ / total $
    lost), and emits a verdict (BAG_HOLDER / KNIFE_CATCHER /
    WHIPSAW_BLEED / DISCIPLINED_CUTTER / MIXED) once n_losers ≥ 8
    (loser_autopsy STABLE idiom). Advisory only — never gates Opus,
    adds no caps (AGENTS.md #2/#12)."""
    try:
        from .analytics.bag_holding_skill import build_bag_holding_skill
        store = get_store()
        trades = list(reversed(store.recent_trades(2000)))
        return jsonify(build_bag_holding_skill(trades))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/risk-adjusted-returns")
def risk_adjusted_returns_api():
    """Risk-adjusted returns vs the S&P 500 — port Sharpe/Sortino, S&P
    Sharpe/Sortino, sharpe alpha, information ratio, and a verdict on
    risk-adjusted performance. /api/benchmark is point-to-point DOLLAR
    alpha (path-blind); /api/analytics carries a SCALAR sharpe but no
    S&P parity or verdict. This is the risk-aware companion: is the
    bot's return alpha worth the volatility it took on? Sample-size
    honesty: numerics emit at ≥5 paired daily returns, verdict at ≥7
    (the benchmark.py / trade_asymmetry STABLE idiom). Advisory only —
    never gates Opus, adds no caps (AGENTS.md #2/#12)."""
    try:
        from .analytics.risk_adjusted_returns import build_risk_adjusted_returns
        store = get_store()
        eq = store.equity_curve(limit=2000)
        return jsonify(build_risk_adjusted_returns(eq))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/winner-autopsy")
def winner_autopsy_api():
    """Per-closed-winning-round-trip post-mortem — the positive mirror of
    /api/loser-autopsy. Every behavioural builder reflects a pathology
    (/api/loser-autopsy narrates losses, /api/trade-asymmetry flags
    DISPOSITION_BLEED, /api/churn counts overtrading, /api/self-review feeds
    only the failures back into the prompt). None tell the desk *which
    winning behaviour to repeat*. This composes the single source of truth
    (build_round_trips, AGENTS.md #10) and joins the verbatim entry/exit
    reason back from the contributing trade rows, classifies each win into
    an objective success mode (HOME_RUN / SCALP / SLOW_GRIND / TARGET_HIT —
    SLOW_GRIND = let a winner run, the good disposition behaviour; SCALP =
    cut one fast, the disposition effect surfaced per-trade), and rolls up
    which name is the engine + which mode dominates. The pattern verdict is
    withheld until n≥8 winners (trade_asymmetry/loser_autopsy STABLE idiom).
    Advisory only — never gates Opus, never injected into the decision
    prompt, adds no caps (AGENTS.md #2/#12)."""
    try:
        from .analytics.winner_autopsy import build_winner_autopsy
        store = get_store()
        # Same trades convention as /api/analytics & /api/loser-autopsy:
        # oldest → newest (build_round_trips reads in sequence).
        trades = list(reversed(store.recent_trades(2000)))
        return jsonify(build_winner_autopsy(trades))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/thesis-keyword-lift")
def thesis_keyword_lift_api():
    """Open-vocabulary keyword-lift across closed round-trips' entry reasons.

    The orthogonal *open-vocabulary* mirror of /api/catalyst-class-autopsy.
    catalyst_class_autopsy labels each closed trip with a fixed taxonomy
    chosen by the analytics author (ML_ADVISOR / EARNINGS_PLAY / ANALYST_PT
    / TECHNICALS / MACRO / BREAKING_NEWS / PUNDIT / SECTOR_SYMPATHY /
    CONCENTRATION / UNCLASSIFIED). This builder learns the dominant
    keywords directly from the trader's own entry_reason text — a pattern
    like "trades whose reason mentions 'guidance' win 80% of the time vs
    50% baseline" surfaces here even when no class exists for it.

    Composes build_round_trips (SSOT, AGENTS.md #10) and joins entry_reason
    verbatim from the contributing trade row by DB id (the
    loser/winner_autopsy / catalyst_class_autopsy discipline). Pure / no
    LLM — never raises. Lift expressed as percentage-point delta vs the
    pool baseline (symmetric and bounded; an all-winner keyword is +baseline
    pp, all-loser is -baseline pp). Verdict is the single most-positive-lift
    keyword once STABLE (n_winners >= 4 AND n_losers >= 4 — both sides
    needed for a meaningful lift comparison). Advisory only — never gates
    Opus, never injected into the decision prompt, adds no caps (AGENTS.md
    #2/#12)."""
    try:
        from .analytics.thesis_keyword_lift import build_thesis_keyword_lift
        store = get_store()
        # Same trades convention as /api/analytics & /api/loser-autopsy:
        # oldest → newest (build_round_trips reads in sequence).
        trades = list(reversed(store.recent_trades(2000)))
        return jsonify(build_thesis_keyword_lift(trades))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/catalyst-class-autopsy")
def catalyst_class_autopsy_api():
    """Per-entry-thesis-class autopsy of closed round-trips. The orthogonal
    complement to /api/loser-autopsy and /api/winner-autopsy: both of those
    classify the EXIT behaviour (KNIFE_CATCH / WHIPSAW / SLOW_BLEED /
    STOPPED_OUT and HOME_RUN / SCALP / SLOW_GRIND / TARGET_HIT) — *how the
    trade was closed*. Neither classifies the ENTRY thesis — *which catalyst
    TYPE motivated the open*. This composes the single source of truth
    (build_round_trips, AGENTS.md #10), joins the verbatim entry reason back
    from the contributing trade row by DB id (the loser/winner_autopsy
    discipline), multi-labels each closed trip by every matched class
    (ML_ADVISOR / EARNINGS_PLAY / ANALYST_PT / TECHNICALS / MACRO /
    BREAKING_NEWS / PUNDIT / SECTOR_SYMPATHY / CONCENTRATION /
    UNCLASSIFIED), and surfaces per-class win-rate vs the pool baseline.
    Verdicts (BIASED_WINNER / BIASED_LOSER / NEUTRAL) are withheld below
    n=STABLE_MIN_TRIPS_PER_CLASS=4 per class (the loser_autopsy /
    trade_asymmetry STABLE-gate idiom). Advisory only — never gates Opus,
    never injected into the decision prompt, adds no caps (AGENTS.md
    #2/#12)."""
    try:
        from .analytics.catalyst_class_autopsy import (
            build_catalyst_class_autopsy,
        )
        store = get_store()
        # Same trades convention as /api/analytics & /api/loser-autopsy:
        # oldest → newest (build_round_trips reads in sequence).
        trades = list(reversed(store.recent_trades(2000)))
        return jsonify(build_catalyst_class_autopsy(trades))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/streak")
def streak_api():
    """Win/loss streak structure on closed round-trips — current run, longest
    historical extremes, and last-N W/L sequence. /api/trade-asymmetry gives
    payoff math; /api/winner-autopsy and /api/loser-autopsy narrate per-trade
    outcomes; /api/churn counts re-entry cadence. None surface *am I on a hot
    hand or a cold streak right now*. This consumes the single source of
    truth (build_round_trips, AGENTS.md #10), counts consecutive same-sign
    closes backward from the most recent exit (flats skipped, never block a
    streak), and emits a HOT_HAND / TILT_RISK / NEUTRAL verdict only when
    STABLE (n_round_trips >= 8 — the winner/loser_autopsy honesty idiom).
    Advisory only — never gates Opus, never injected into the decision
    prompt, adds no caps (AGENTS.md #2/#12)."""
    try:
        from .analytics.streak import build_streak
        store = get_store()
        # Same trades convention as /api/winner-autopsy & /api/loser-autopsy:
        # oldest → newest (build_round_trips reads in sequence).
        trades = list(reversed(store.recent_trades(2000)))
        return jsonify(build_streak(trades))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/repeat-loser")
def repeat_loser_api():
    """Per-ticker losing-streak detector — the per-ticker companion to
    /api/streak. /api/streak says "you're on a 4-loss run"; this says "and
    3 of those 4 are on LITE", which is the actionable read. A desk on a
    book-wide tilt run whose losses are concentrated on one name acts
    differently ("stop sizing this name") than one whose losses are
    distributed ("general tilt, step back from the keys"). Consumes the
    single source of truth (build_round_trips → build_repeat_loser,
    AGENTS.md #10), surfaces the REPEAT_LOSER verdict when ≥1 ticker has a
    ≥2-loss run ending in its most recent non-flat outcome. Advisory only —
    never gates Opus, never injected into the decision prompt, adds no
    caps (AGENTS.md #2/#12)."""
    try:
        from .analytics.repeat_loser import build_repeat_loser
        store = get_store()
        # Same trades convention as /api/streak: oldest → newest.
        trades = list(reversed(store.recent_trades(2000)))
        return jsonify(build_repeat_loser(trades))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/hold-discipline")
def hold_discipline_api():
    """The disposition trap, caught *while it is still happening* on the
    OPEN book. /api/loser-autopsy & /api/trade-asymmetry post-mortem
    trades already closed; /api/thesis-drift re-tests an open position
    against its *thesis*; /api/capital-paralysis is about cash drag;
    /api/position-thesis shows days-held but has no empirical reference.
    None answer the forward discipline question: *which open position am
    I, right now, holding at a loss past my own historical losing-cut
    time?* This anchors on the desk's OWN behaviour — the empirical
    median losing hold consumed verbatim from build_loser_autopsy →
    build_round_trips (single source of truth, AGENTS.md #10) — and the
    per-position $ read directly from positions.unrealized_pl (the option
    ×100 is already baked in there). Verdict withheld until ≥
    MIN_REFERENCE_LOSERS closed losers (the loser_autopsy sample-size
    honesty idiom). Advisory only — never gates Opus, never injected into
    the decision prompt, adds no caps (AGENTS.md #2/#12)."""
    try:
        from .analytics.hold_discipline import build_hold_discipline
        store = get_store()
        # Same trades convention as /api/loser-autopsy & /api/analytics:
        # oldest → newest (build_round_trips reads in sequence).
        trades = list(reversed(store.recent_trades(2000)))
        return jsonify(build_hold_discipline(store.open_positions(), trades))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/game-plan")
@swr_cached("game_plan", 45.0)
def game_plan_api():
    """The single prioritised, trader-facing action plan for the next session.

    The ingredients already exist as separate panels — the co-pilot verb
    (/api/suggestions via _classify_action), the disposition trap
    (/api/hold-discipline), concentration / single-name risk (/api/risk) and
    the earnings calendar (/api/event-calendar). Before this, a trader had to
    open four panels and fuse them by hand. This route does the data-gathering
    and reuses _classify_action (no forked verb logic); analytics.game_plan
    does the pure prioritisation. Distinct from unified's /api/action-queue,
    which is *operator* triage (stale process, decision-parse health) — this
    is the *trade* plan. Observational — it reorders/annotates existing
    signals; it never sizes a trade, never gates Opus, adds no caps
    (AGENTS.md #2/#12)."""
    try:
        from . import signals as _sig, market as _mkt
        from .strategy import (get_quant_signals_live, WATCHLIST,
                               _names_in_play)
        from .analytics.hold_discipline import build_hold_discipline
        from .analytics.event_calendar import build_event_calendar
        from .analytics.game_plan import build_game_plan

        store = get_store()
        pf = store.get_portfolio()
        positions = store.open_positions()
        total_value = float(pf.get("total_value") or 0.0)
        cash = float(pf.get("cash") or 0.0)

        # Held qty (stock) — mirrors /api/suggestions data plumbing exactly.
        held: dict[str, float] = {}
        for p in positions:
            if p.get("type") == "stock":
                held[p["ticker"]] = held.get(p["ticker"], 0.0) + float(
                    p.get("qty") or 0)

        try:
            top_signals = _sig.get_top_signals(n=30, hours=6, min_score=5.0)
        except Exception:
            top_signals = []
        try:
            universe = {t.upper() for t in WATCHLIST}
        except Exception:
            universe = set()
        universe |= {t.upper() for t in held}

        news: dict[str, dict] = {}
        for art in top_signals:
            for tk in (art.get("tickers") or []):
                if not tk or len(tk) > 6 or tk.upper() not in universe:
                    continue
                rec = news.setdefault(tk, {
                    "news_max_score": 0.0, "news_urgent": False})
                if (art.get("ai_score") or 0) > rec["news_max_score"]:
                    rec["news_max_score"] = float(art.get("ai_score") or 0)
                if (art.get("urgency") or 0) >= 1:
                    rec["news_urgent"] = True
        for tk in held:
            news.setdefault(tk, {"news_max_score": 0.0, "news_urgent": False})

        tickers = list(news.keys())
        try:
            quant = get_quant_signals_live(tickers) if tickers else {}
        except Exception:
            quant = {}
        try:
            prices = _mkt.get_prices(tickers) if tickers else {}
        except Exception:
            prices = {}

        classified: dict[str, dict] = {}
        for tk, nrec in news.items():
            q = quant.get(tk, {})
            action, conviction, notes = _classify_action(
                tk, held.get(tk, 0.0), q,
                nrec["news_max_score"], nrec["news_urgent"])
            classified[tk] = {
                "action": action, "conviction": conviction, "reasons": notes,
                "held_qty": held.get(tk, 0.0),
                "news_max_score": nrec["news_max_score"],
                "news_urgent": nrec["news_urgent"],
                "price": prices.get(tk),
            }

        try:
            trades_oldest_first = list(reversed(store.recent_trades(2000)))
            hd = build_hold_discipline(positions, trades_oldest_first)
        except Exception:
            hd = {}
        try:
            keep = _names_in_play(positions, top_signals, WATCHLIST)
        except Exception:
            keep = set(held)
        try:
            ec = build_event_calendar(positions, keep)
        except Exception:
            ec = {}

        # Concentration — reuse the /api/risk math (_classify +
        # _concentration_severity) so the two panels never disagree.
        rows = []
        sector_val: dict[str, float] = {}
        for p in positions:
            mlt = 100 if p["type"] in ("call", "put") else 1
            price = p.get("current_price") or p.get("avg_cost") or 0.0
            val = float(price) * float(p.get("qty") or 0) * mlt
            sec = _classify(p["ticker"])
            sector_val[sec] = sector_val.get(sec, 0.0) + val
            rows.append((p["ticker"],
                         (val / total_value * 100) if total_value else 0.0))
        rows.sort(key=lambda r: -r[1])
        top1_pct = round(rows[0][1], 2) if rows else 0.0
        top3_pct = round(sum(r[1] for r in rows[:3]), 2)
        sev, warn = _concentration_severity(top1_pct, top3_pct)
        concentration = {
            "severity": sev, "warning": warn,
            "top1_ticker": rows[0][0] if rows else "",
            "top1_pct": top1_pct, "top3_pct": top3_pct,
            "cash_pct": round((cash / total_value * 100)
                              if total_value else 0.0, 2),
            "sector_pct": ({s: round(v / total_value * 100, 2)
                            for s, v in sector_val.items()}
                           if total_value else {}),
        }

        plan = build_game_plan(
            positions=positions, total_value=total_value, cash=cash,
            hold_discipline=hd, concentration=concentration,
            earnings_events=(ec.get("events") or []),
            classified=classified)
        try:
            nd, secs = _next_market_open()
            plan["market_open"] = _mkt.is_market_open(
                datetime.now(timezone.utc))
            plan["next_open_seconds"] = secs
        except Exception:
            pass
        return jsonify(plan)
    except Exception as e:
        return jsonify({"error": str(e), "state": "ERROR",
                        "position_actions": [], "portfolio_directives": [],
                        "opportunities": []}), 500


@app.route("/api/track-record")
def track_record_api():
    """Per-name closed-trade memory — the same verbatim loser/winner-autopsy
    narrative the live decision prompt now sees, surfaced for the operator &
    chat. /api/loser-autopsy & /api/winner-autopsy narrate the book's losses
    and wins separately and book-wide; this groups *both* by ticker so "how
    have we actually done on NVDA?" is one row — and it is the *same builder*
    strategy._build_payload injects (there filtered to the names in play this
    cycle) so the dashboard, chat and the in-prompt block can never drift
    (single source of truth, AGENTS.md #10; the /api/self-review precedent).
    names=None here ⇒ every traded name. Composes build_loser_autopsy +
    build_winner_autopsy verbatim — no re-derived P&L. Advisory only — never
    gates Opus, adds no caps (AGENTS.md #2/#12)."""
    try:
        from .analytics.track_record import build_track_record
        store = get_store()
        # Same trades convention as /api/loser-autopsy & /api/winner-autopsy:
        # oldest → newest (build_round_trips reads in sequence).
        trades = list(reversed(store.recent_trades(2000)))
        return jsonify(build_track_record(trades))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/correlation")
@swr_cached("correlation", 90.0)
def correlation_api():
    """Concentration honesty — do the held names actually move together?
    /api/risk reports name-level concentration + a single SPY-shock; it
    cannot see whether the book is really one *factor* bet. This computes
    pairwise return correlation among held stock positions, the
    weight-Herfindahl effective-position count, and the
    correlation-adjusted effective number of *independent* bets (collapses
    toward 1 as the names co-move however many tickers are on the book).
    Options are flagged & skipped (the open_attribution "stocks only"
    carve-out). The builder is pure; the yfinance fetch lives here (the
    thesis_drift split) and degrades to INSUFFICIENT, never an error.
    Advisory only — never gates Opus, adds no caps (AGENTS.md #2/#12)."""
    try:
        from .analytics.correlation import build_correlation
        store = get_store()
        positions = store.open_positions()
        poss, price_history = [], {}
        for p in positions:
            ptype = p.get("type")
            kind = ptype if ptype in ("call", "put") else "stock"
            mult = 100 if kind in ("call", "put") else 1
            price = p.get("current_price") or p.get("avg_cost") or 0.0
            qty = float(p.get("qty") or 0.0)
            poss.append({
                "ticker": p.get("ticker"),
                "market_value": round(float(price) * qty * mult, 2),
                "type": kind,
            })
            if kind == "stock" and p.get("ticker") not in price_history:
                try:
                    bars = _daily_history_cached(p["ticker"], "3mo")
                    price_history[p["ticker"]] = [c for _, c in bars]
                except Exception:
                    price_history[p["ticker"]] = []
        return jsonify(build_correlation(poss, price_history))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/correlation-cluster-warning")
@swr_cached("correlation-cluster-warning", 90.0)
def correlation_cluster_warning_api():
    """Hidden-factor-bet alarm — translates ``/api/correlation``'s pairwise
    matrix into a single cluster-share verdict.

    The parent ``/api/correlation`` reports a *mean* ρ across all pairs +
    one global verdict. A book like {NVDA, AMD, AVGO, KO, JNJ} reads as
    MODERATE on mean ρ — but the first three names form a single semis
    cluster running as one trade inside a wrapper of two uncorrelated
    consumer staples. This endpoint surfaces that cluster: it
    single-linkages the names at ρ ≥ ``HIGH_CORR`` (the same constant the
    parent uses for ``CONCENTRATED``), returns the largest multi-name
    cluster + its share of book by market value, and emits an
    ``NO_CLUSTERS / WATCHLIST_CLUSTER / DOMINANT_CLUSTER /
    HIDDEN_FACTOR_BET`` verdict keyed off the cluster's *weight*, not the
    mean ρ. Pure builder over the existing ``build_correlation`` payload —
    no new network. Advisory only — never gates Opus, adds no caps
    (AGENTS.md #2/#12)."""
    try:
        from .analytics.correlation import build_correlation
        from .analytics.correlation_cluster_warning import (
            build_correlation_cluster_warning,
        )
        store = get_store()
        positions = store.open_positions()
        poss, price_history = [], {}
        for p in positions:
            ptype = p.get("type")
            kind = ptype if ptype in ("call", "put") else "stock"
            mult = 100 if kind in ("call", "put") else 1
            price = p.get("current_price") or p.get("avg_cost") or 0.0
            qty = float(p.get("qty") or 0.0)
            poss.append({
                "ticker": p.get("ticker"),
                "market_value": round(float(price) * qty * mult, 2),
                "type": kind,
            })
            if kind == "stock" and p.get("ticker") not in price_history:
                try:
                    bars = _daily_history_cached(p["ticker"], "3mo")
                    price_history[p["ticker"]] = [c for _, c in bars]
                except Exception:
                    price_history[p["ticker"]] = []
        corr = build_correlation(poss, price_history)
        return jsonify(build_correlation_cluster_warning(corr))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/trade-attribution")
@swr_cached("trade-attribution", 60.0)
def trade_attribution_api():
    """News-to-trade attribution — which articles plausibly preceded each
    recent FILLED trade?

    The symmetric question to ``/api/news-edge`` / ``/api/source-edge``
    (do scored signals precede the SPY-abnormal move across the BOOK?):
    here, per-fill, what was the news landscape in the
    ``window_hours`` (default 4h, query-overridable 0.5..24) preceding
    each trade? *Implied* attribution — the bot's literal context isn't
    stored row-by-row, so we honestly reconstruct only the highest-scored
    live-only articles mentioning the traded ticker in the pre-trade
    window. A trade with zero matches surfaces ``n_attributed: 0`` (the
    ``recovery`` / ``loser_autopsy`` precedent — no-fabrication negative
    space is data too).

    Live-only by construction (invariant #1): the canonical
    backtest-strip clause is applied to ``articles.db`` here in the
    endpoint, mirroring ``_ticker_news_pulse``. The builder is pure; the
    I/O lives here (the ``thesis_drift`` / ``correlation`` split).
    Advisory only — never gates Opus, adds no caps (AGENTS.md #2/#12).

    Query params (all optional):
      ``hours_back`` — trade lookback (default 24, clamp 1..168)
      ``window_hours`` — per-trade article window (default 4.0, clamp 0.5..24)
      ``max_per_trade`` — top-N articles per trade (default 3, clamp 1..10)
      ``min_ai_score`` — article cutoff (default 2.0, clamp 0..10)
    """
    try:
        from .analytics.trade_attribution import build_trade_attribution

        def _qf(name: str, default: float, lo: float, hi: float) -> float:
            try:
                v = float(request.args.get(name, default))
            except (TypeError, ValueError):
                v = default
            return max(lo, min(hi, v))

        hours_back = _qf("hours_back", 24.0, 1.0, 168.0)
        window_hours = _qf("window_hours", 4.0, 0.5, 24.0)
        max_per_trade = int(_qf("max_per_trade", 3.0, 1.0, 10.0))
        min_ai_score = _qf("min_ai_score", 2.0, 0.0, 10.0)

        store = get_store()
        # Read-only fetch of recent FILLED trades; the store's own lock
        # serialises with the writer (invariant #7).
        with store._lock:  # noqa: SLF001 — same access pattern dashboard uses elsewhere
            cur = store.conn.execute(
                "SELECT id, timestamp, ticker, action, qty, price, value, "
                "reason, option_type FROM trades "
                "WHERE timestamp >= datetime('now', ?) "
                "ORDER BY timestamp DESC LIMIT 200",
                (f"-{hours_back:.1f} hours",),
            )
            rows = cur.fetchall()
        trades = [{
            "id": r[0], "timestamp": r[1], "ticker": r[2], "action": r[3],
            "qty": r[4], "price": r[5], "value": r[6], "reason": r[7],
            "type": r[8],
        } for r in rows]

        # Live-only articles in the broadest needed window (oldest trade
        # minus the per-trade window, with a small slack). One query covers
        # every fill rather than N queries.
        articles: list[dict] = []
        if trades:
            oldest_iso = trades[-1]["timestamp"]
            oldest_dt = datetime.fromisoformat(
                oldest_iso.replace("Z", "+00:00")) if oldest_iso else (
                datetime.now(timezone.utc) - timedelta(hours=hours_back))
            if oldest_dt.tzinfo is None:
                oldest_dt = oldest_dt.replace(tzinfo=timezone.utc)
            since = (oldest_dt - timedelta(hours=window_hours)).isoformat()

            path = _articles_db_path()
            if path is not None:
                conn = None
                try:
                    conn = sqlite3.connect(
                        f"file:{path}?mode=ro", uri=True, timeout=5)
                    art_rows = conn.execute(
                        "SELECT title, url, source, ai_score, urgency, "
                        "first_seen FROM articles "
                        "WHERE first_seen >= ? AND ai_score >= ? "
                        "AND url NOT LIKE 'backtest://%' "
                        "AND source NOT LIKE 'backtest_%' "
                        "AND source NOT LIKE 'opus_annotation%' "
                        "ORDER BY ai_score DESC LIMIT 5000",
                        (since, min_ai_score),
                    ).fetchall()
                    articles = [{
                        "title": r[0], "url": r[1], "source": r[2],
                        "ai_score": r[3], "urgency": r[4], "first_seen": r[5],
                    } for r in art_rows]
                except Exception:
                    articles = []
                finally:
                    if conn is not None:
                        try:
                            conn.close()
                        except Exception:
                            pass

        return jsonify(build_trade_attribution(
            trades, articles,
            window_hours=window_hours,
            max_per_trade=max_per_trade,
            min_ai_score=min_ai_score,
        ))
    except Exception as e:
        return jsonify({"error": str(e), "state": "ERROR"}), 500


@app.route("/api/news-themes")
@swr_cached("news-themes", 60.0)
def news_themes_api():
    """Per-ticker theme aggregation across the live news feed.

    The wire produces 100+ articles per hour; the operator's actual
    glance-at-the-feed question is "which tickers is the wire spending
    its breath on right now, and which of those am I holding vs
    ignoring?" The existing surfaces are siblings (do NOT consolidate —
    AGENTS.md #10):

      * ``/api/news-deduped`` is the linear item list.
      * ``/api/news-velocity`` is the per-held-ticker Poisson rate
        (BUILDING vs FADING vs baseline) — not a score-weighted "loudest
        theme" rollup.
      * ``/api/sector-heatmap`` / ``/api/sector-signal-fit`` aggregate
        at the SECTOR level.
      * ``/api/watchlist-opportunities`` ranks within the curated
        watchlist; this is across the *entire* live feed regardless of
        watchlist membership.

    Builder is pure (``analytics/news_themes.py::build_news_themes``):
    Σ ai_score × exp(-age_h / 6h × ln 2), multi-ticker articles split
    their score evenly (a 4-ticker headline contributes 0.25× to each
    theme — avoids one wide-net article inflating four themes
    simultaneously, the same discriminator as ``sector_signal_fit``).
    Defense-in-depth backtest-row filter at the builder so a leaked
    synthetic row cannot reach user-facing JSON.

    Query params (all optional):
      ``hours`` — recency window (default 24, clamp 1..168)
      ``max_themes`` — surfaced themes clip (default 20, clamp 1..100)
      ``min_score`` — article ai_score floor at SQL (default 2.0, 0..10)

    Advisory only — never gates Opus, adds no caps (AGENTS.md #2/#12).
    """
    try:
        from .analytics.news_themes import build_news_themes

        def _qf(name: str, default: float, lo: float, hi: float) -> float:
            try:
                v = float(request.args.get(name, default))
            except (TypeError, ValueError):
                v = default
            return max(lo, min(hi, v))

        window_hours = _qf("hours", 24.0, 1.0, 168.0)
        max_themes = int(_qf("max_themes", 20.0, 1.0, 100.0))
        min_score = _qf("min_score", 2.0, 0.0, 10.0)

        now = datetime.now(timezone.utc)
        since = (now - timedelta(hours=window_hours)).isoformat()

        # Live-only articles. We re-use the canonical SQL filter and
        # also count on the builder's defense-in-depth drop.
        articles: list[dict] = []
        path = _articles_db_path()
        if path is not None:
            conn = None
            try:
                conn = sqlite3.connect(
                    f"file:{path}?mode=ro", uri=True, timeout=5)
                rows = conn.execute(
                    "SELECT title, url, source, ai_score, urgency, "
                    "first_seen FROM articles "
                    "WHERE first_seen >= ? AND ai_score >= ? "
                    "AND url NOT LIKE 'backtest://%' "
                    "AND source NOT LIKE 'backtest_%' "
                    "AND source NOT LIKE 'opus_annotation%' "
                    "ORDER BY ai_score DESC LIMIT 2000",
                    (since, min_score),
                ).fetchall()
                # Re-extract tickers from title/summary using the same
                # cashtag/word-boundary regex shape signals.py uses; the
                # articles table doesn't persist a parsed tickers list.
                # Reuse signals._extract_tickers (the SSOT used by the
                # live trader's prompt-building path) so theme tickers
                # never drift from the universe Opus sees in `decide()`.
                from .signals import _extract_tickers  # noqa: WPS433
                for r in rows:
                    title = r[0] or ""
                    tk = sorted(_extract_tickers(title))
                    articles.append({
                        "title": title, "url": r[1], "source": r[2],
                        "ai_score": r[3], "urgency": r[4],
                        "first_seen": r[5],
                        "tickers": tk,
                    })
            except Exception:
                articles = []
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass

        # Held tickers for the held/unheld flag.
        held: list[str] = []
        try:
            held = [
                p["ticker"]
                for p in get_store().open_positions()
                if p.get("ticker")
            ]
        except Exception:
            held = []

        return jsonify(build_news_themes(
            articles, held_tickers=held, now=now,
            window_hours=window_hours, max_themes=max_themes,
        ))
    except Exception as e:
        return jsonify({"error": str(e), "state": "ERROR"}), 500


@app.route("/api/held-theme-decay")
@swr_cached("held-theme-decay", 60.0)
def held_theme_decay_api():
    """Per-held-ticker fresh-vs-prior decayed news score velocity.

    Answers the question every other news surface punts on:
    *for each ticker I currently own, is the score-weighted news flow
    LOUDER NOW or QUIETER NOW than it was a window ago?* — i.e. is the
    catalyst still alive in the wire or has the wire moved on?

    Distinct from neighbours (invariant #10 — do not consolidate):

      * ``/api/news-themes`` — single-window decayed-score snapshot,
        NO comparison vs an earlier window; a held theme that went DARK
        looks identical to one that just lit up.
      * ``/api/news-velocity`` — Poisson MENTION-RATE z-score vs a 168h
        baseline; not score-weighted (a flood of low-relevance mentions
        inflates the rate while one 9.5 Sonnet-labelled article moves
        it the same as a junk RSS row). Different signal entirely.
      * ``/api/position-thesis`` — latest 24h headlines per held position
        with a bull/bear split; single window, no velocity dimension.
      * ``/api/thesis-drift`` — grades held positions against ENTRY
        rationale, not current wire prominence.

    Builder is pure (``analytics/held_theme_decay.py``): the fresh
    window default (6h) matches ``news_themes.DECAY_HALF_LIFE_HOURS`` so
    a held ticker's ``fresh_score`` is directly comparable to its
    contribution to the news-themes top-themes ranking. The prior
    window is the immediately preceding non-overlapping band of the
    same width. Multi-ticker articles split their weight evenly across
    ALL mentioned tickers (anti-inflation rule, same discriminator as
    ``news_themes`` / ``sector_signal_fit``). Defense-in-depth backtest
    filter at the builder so a leaked synthetic row cannot reach
    user-facing JSON.

    Verdict ladder:
      DARK     — no qualifying articles in either window
      FADING   — fresh < prior × ``FADE_RATIO`` (0.7) → reassess thesis
      BUILDING — fresh > prior × ``BUILD_RATIO`` (1.43) AND fresh meets
                 the ``MIN_FRESH_SCORE`` (1.0) absolute floor
      STABLE   — between ``FADE_RATIO`` and ``BUILD_RATIO``

    Query params (all optional):
      ``hours`` — fresh-window width (default 6, clamp 1..72). Prior
        window is the immediately preceding band of the same width.
      ``min_score`` — article ai_score floor at SQL (default 2.0, 0..10).

    Advisory only — never gates Opus, adds no caps (AGENTS.md #2/#12).
    """
    try:
        from .analytics.held_theme_decay import build_held_theme_decay

        def _qf(name: str, default: float, lo: float, hi: float) -> float:
            try:
                v = float(request.args.get(name, default))
            except (TypeError, ValueError):
                v = default
            return max(lo, min(hi, v))

        fresh_window_hours = _qf("hours", 6.0, 1.0, 72.0)
        min_score = _qf("min_score", 2.0, 0.0, 10.0)

        now = datetime.now(timezone.utc)
        # SQL window must cover the FULL fresh+prior band (2× the fresh
        # window) plus a small safety margin so an article right at the
        # prior-edge isn't trimmed by clock drift.
        sql_window_hours = max(fresh_window_hours * 2.0 + 1.0, 4.0)
        since = (now - timedelta(hours=sql_window_hours)).isoformat()

        # Live-only article rows — SSOT _LIVE_ONLY_CLAUSE inlined per
        # AGENTS.md invariant (paper-trader copy of digital-intern's
        # canonical filter). Re-extract tickers via signals._extract_tickers
        # so the held-theme universe never drifts from the live trader's
        # decide-time view (same discriminator news-themes uses).
        articles: list[dict] = []
        path = _articles_db_path()
        if path is not None:
            conn = None
            try:
                conn = sqlite3.connect(
                    f"file:{path}?mode=ro", uri=True, timeout=5)
                rows = conn.execute(
                    "SELECT title, url, source, ai_score, urgency, "
                    "first_seen FROM articles "
                    "WHERE first_seen >= ? AND ai_score >= ? "
                    "AND url NOT LIKE 'backtest://%' "
                    "AND source NOT LIKE 'backtest_%' "
                    "AND source NOT LIKE 'opus_annotation%' "
                    "ORDER BY ai_score DESC LIMIT 2000",
                    (since, min_score),
                ).fetchall()
                from .signals import _extract_tickers  # noqa: WPS433
                for r in rows:
                    title = r[0] or ""
                    tk = sorted(_extract_tickers(title))
                    articles.append({
                        "title": title, "url": r[1], "source": r[2],
                        "ai_score": r[3], "urgency": r[4],
                        "first_seen": r[5],
                        "tickers": tk,
                    })
            except Exception:
                articles = []
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass

        # Held tickers from the live book — case-insensitive normalized
        # inside the builder.
        held: list[str] = []
        try:
            held = [
                p["ticker"]
                for p in get_store().open_positions()
                if p.get("ticker")
            ]
        except Exception:
            held = []

        return jsonify(build_held_theme_decay(
            articles, held_tickers=held, now=now,
            fresh_window_hours=fresh_window_hours,
        ))
    except Exception as e:
        return jsonify({"error": str(e), "state": "ERROR"}), 500


@app.route("/api/rising-unheld-themes")
@swr_cached("rising-unheld-themes", 60.0)
def rising_unheld_themes_api():
    """Per-unheld-ticker fresh-vs-prior decayed-news-score velocity.

    The mirror of ``/api/held-theme-decay``. held-theme-decay asks *is
    the catalyst on a ticker I OWN still alive in the wire?* This
    endpoint answers the complementary rotation question: *which
    tickers does the wire have a RISING catalyst on that I am NOT in?*

    Distinct from every neighbour (invariant #10 — do not consolidate):

      * ``/api/news-themes`` — single-window decayed-score snapshot of
        every theme (held + unheld). NO comparison vs an earlier
        window: a stale theme with a long tail looks identical to one
        that just lit up.
      * ``/api/held-theme-decay`` — same velocity decomposition,
        restricted to HELD tickers. Identical math; this is the
        unheld complement (shares constants + ``_verdict`` rule via
        SSOT import).
      * ``/api/watchlist-opportunities`` — scan over the named
        WATCHLIST tickers only, no velocity dimension, no decay
        weighting.
      * ``/api/idle-opportunity`` — drought-gated point-in-time
        surface (only fires when the bot is in a HOLD streak);
        snapshot, no velocity. This endpoint is always-on and is the
        velocity decomposition, so an unheld ticker can show up here
        with a BUILDING verdict even when the bot is actively trading
        and idle-opportunity is silent.
      * digital-intern ``trend_velocity`` — market-wide MENTION-RATE
        gainers (Poisson), not score-weighted decayed prominence.

    Builder is pure (``analytics/rising_unheld_themes.py``). Fresh-
    window default (6h) is imported from
    ``held_theme_decay.FRESH_WINDOW_HOURS`` so the two velocity
    surfaces share their window shape (a rotation-pair invariant).
    Defense-in-depth backtest filter at the builder so a leaked
    synthetic row cannot reach user-facing JSON.

    Verdict ladder (per-ticker):
      DARK     — no qualifying articles in either window
      FADING   — fresh < prior × FADE_RATIO (0.7)
      STABLE   — between FADE_RATIO and BUILD_RATIO
      BUILDING — fresh > prior × BUILD_RATIO (1.43) AND fresh >=
                 MIN_FRESH_SCORE (1.0) — accelerating existing story
      BREAKING — prior == 0 AND fresh >= BREAKING_FRESH_SCORE (3.0)
                 — brand-new catalyst, no prior coverage (the highest-
                 urgency rotation signal)

    Output is sorted BREAKING > BUILDING > STABLE > FADING > DARK
    within each bucket by descending fresh_score so the loudest
    actionable rotation candidate tops the list.

    Query params (all optional):
      ``hours`` — fresh-window width (default 6, clamp 1..72). Prior
        window is the immediately preceding band of the same width.
      ``min_score`` — article ai_score floor at SQL (default 2.0,
        0..10).
      ``max_themes`` — cap on returned rows (default 20, clamp 1..100).
        Aggregate counts span the full unheld universe regardless.

    Advisory only — never gates Opus, adds no caps (AGENTS.md #2/#12).
    """
    try:
        from .analytics.rising_unheld_themes import build_rising_unheld_themes

        def _qf(name: str, default: float, lo: float, hi: float) -> float:
            try:
                v = float(request.args.get(name, default))
            except (TypeError, ValueError):
                v = default
            return max(lo, min(hi, v))

        def _qi(name: str, default: int, lo: int, hi: int) -> int:
            try:
                v = int(request.args.get(name, default))
            except (TypeError, ValueError):
                v = default
            return max(lo, min(hi, v))

        fresh_window_hours = _qf("hours", 6.0, 1.0, 72.0)
        min_score = _qf("min_score", 2.0, 0.0, 10.0)
        max_themes = _qi("max_themes", 20, 1, 100)

        now = datetime.now(timezone.utc)
        # SQL window must cover the FULL fresh+prior band (2× the
        # fresh window) plus a small safety margin so an article right
        # at the prior-edge isn't trimmed by clock drift. Identical
        # shape to held-theme-decay.
        sql_window_hours = max(fresh_window_hours * 2.0 + 1.0, 4.0)
        since = (now - timedelta(hours=sql_window_hours)).isoformat()

        articles: list[dict] = []
        path = _articles_db_path()
        if path is not None:
            conn = None
            try:
                conn = sqlite3.connect(
                    f"file:{path}?mode=ro", uri=True, timeout=5)
                rows = conn.execute(
                    "SELECT title, url, source, ai_score, urgency, "
                    "first_seen FROM articles "
                    "WHERE first_seen >= ? AND ai_score >= ? "
                    "AND url NOT LIKE 'backtest://%' "
                    "AND source NOT LIKE 'backtest_%' "
                    "AND source NOT LIKE 'opus_annotation%' "
                    "ORDER BY ai_score DESC LIMIT 2000",
                    (since, min_score),
                ).fetchall()
                from .signals import _extract_tickers  # noqa: WPS433
                for r in rows:
                    title = r[0] or ""
                    tk = sorted(_extract_tickers(title))
                    articles.append({
                        "title": title, "url": r[1], "source": r[2],
                        "ai_score": r[3], "urgency": r[4],
                        "first_seen": r[5],
                        "tickers": tk,
                    })
            except Exception:
                articles = []
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass

        held: list[str] = []
        try:
            held = [
                p["ticker"]
                for p in get_store().open_positions()
                if p.get("ticker")
            ]
        except Exception:
            held = []

        return jsonify(build_rising_unheld_themes(
            articles, held_tickers=held, now=now,
            fresh_window_hours=fresh_window_hours,
            max_themes=max_themes,
        ))
    except Exception as e:
        return jsonify({"error": str(e), "state": "ERROR"}), 500


@app.route("/api/sector-velocity-delta")
@swr_cached("sector-velocity-delta", 60.0)
def sector_velocity_delta_api():
    """Per-sector-bucket news-velocity delta (fresh vs prior window).

    Answers the operator's first-glance rotation question: *which
    sector bucket is the wire ROTATING INTO right now?* — i.e. is
    bucket-level news flow ACCELERATING or DECELERATING relative to
    the immediately-prior window of equal length?

    Distinct from neighbours (invariant #10 — do not consolidate):

      * ``/api/sector-heatmap`` — PRICE + RSI + 24h news COUNT per
        bucket; point-in-time snapshot. NO velocity dimension: a
        bucket with steady 17-art/day and one ramping from 2→17 look
        identical.
      * ``/api/sector-pulse`` — per-ticker momentum + news count
        snapshot; not bucketed, not velocity.
      * ``/api/sector-signal-fit`` — bucket news COVERAGE vs the
        live book's bucket WEIGHT (a fit measure, not a wire-time-
        derivative).
      * ``/api/sector-exposure`` — $-weight per bucket of the book;
        the inverse direction (book→wire, not wire-velocity).
      * ``/api/news-themes`` / ``/api/held-theme-decay`` /
        ``/api/rising-unheld-themes`` — per-TICKER decomposition;
        this endpoint is the per-BUCKET aggregation of the same
        velocity shape.

    Bucket SSOT is ``sector_heatmap.HEATMAP_BUCKETS`` — the same
    bucket definitions ``/api/sector-heatmap`` uses, so a
    ``memory_core ACCELERATING`` verdict here lines up with the
    memory_core row of the heatmap. Decay / window / ratio constants
    come from ``held_theme_decay`` / ``news_themes`` so re-tuning the
    decay shape in one place updates all three velocity surfaces
    (rotation-pair invariant, extended).

    Verdict ladder (per bucket):
      DARK         — no qualifying articles in either window
      DECELERATING — fresh < prior × FADE_RATIO AND prior was
                     bucket-level prominent (rotation OUT signal)
      FADING       — ratio drop on a marginal bucket (informational)
      STABLE       — between FADE_RATIO and BUILD_RATIO
      BUILDING     — fresh > prior × BUILD_RATIO + per-ticker floor
                     (individual ticker acceleration, sub-sector)
      ACCELERATING — fresh > prior × ACCEL_RATIO AND fresh meets the
                     bucket-prominence floor (rotation IN signal)

    Query params (all optional):
      ``hours`` — fresh-window width (default 6, clamp 1..72). Prior
        window is the immediately preceding band of the same width.
      ``min_score`` — article ai_score floor at SQL (default 2.0,
        0..10).

    Advisory only — never gates Opus, adds no caps (AGENTS.md #2/#12).
    """
    try:
        from .analytics.sector_velocity_delta import build_sector_velocity_delta

        def _qf(name: str, default: float, lo: float, hi: float) -> float:
            try:
                v = float(request.args.get(name, default))
            except (TypeError, ValueError):
                v = default
            return max(lo, min(hi, v))

        fresh_window_hours = _qf("hours", 6.0, 1.0, 72.0)
        min_score = _qf("min_score", 2.0, 0.0, 10.0)

        now = datetime.now(timezone.utc)
        # Same SQL window shape as held-theme-decay / rising-unheld:
        # FULL fresh+prior band (2× the fresh window) + safety margin.
        sql_window_hours = max(fresh_window_hours * 2.0 + 1.0, 4.0)
        since = (now - timedelta(hours=sql_window_hours)).isoformat()

        articles: list[dict] = []
        path = _articles_db_path()
        if path is not None:
            conn = None
            try:
                conn = sqlite3.connect(
                    f"file:{path}?mode=ro", uri=True, timeout=5)
                rows = conn.execute(
                    "SELECT title, url, source, ai_score, urgency, "
                    "first_seen FROM articles "
                    "WHERE first_seen >= ? AND ai_score >= ? "
                    "AND url NOT LIKE 'backtest://%' "
                    "AND source NOT LIKE 'backtest_%' "
                    "AND source NOT LIKE 'opus_annotation%' "
                    "ORDER BY ai_score DESC LIMIT 2000",
                    (since, min_score),
                ).fetchall()
                from .signals import _extract_tickers  # noqa: WPS433
                for r in rows:
                    title = r[0] or ""
                    tk = sorted(_extract_tickers(title))
                    articles.append({
                        "title": title, "url": r[1], "source": r[2],
                        "ai_score": r[3], "urgency": r[4],
                        "first_seen": r[5],
                        "tickers": tk,
                    })
            except Exception:
                articles = []
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass

        return jsonify(build_sector_velocity_delta(
            articles, now=now,
            fresh_window_hours=fresh_window_hours,
        ))
    except Exception as e:
        return jsonify({"error": str(e), "state": "ERROR"}), 500


@app.route("/api/round-trip-postmortem")
@swr_cached("round-trip-postmortem", 60.0)
def round_trip_postmortem_api():
    """Post-exit drift verdict per recent closed round-trip.

    ``/api/round-trips`` says WHAT closed (and what realised P&L it
    booked). Every realised-P&L surface (track-record, churn, streak,
    winner/loser autopsy, trade-asymmetry) reduces the round-trip list
    to a summary stat. None of them ask the operator's follow-up: *was
    the exit good?* — i.e. did the price keep moving in the bot's
    favoured direction after the sell (PREMATURE / MISSED_RUNNER /
    WHIPSAW) or against it (CORRECT)?

    Pure SSOT in ``analytics/round_trip_postmortem.py``; verdict ladder:
    CORRECT (post-exit drop ≤ -1%) / PREMATURE (rise 1..5%) /
    MISSED_RUNNER (rise ≥ 5%) / WHIPSAW (short hold + small loss +
    post-exit recovery — the DRAM-1h-paper-cut pathology) / NEUTRAL
    (inside band) / INSUFFICIENT (exit < 2h ago or no current price).
    Aggregate ``exit_quality_score`` is +1 CORRECT / -1 PREMATURE /
    -2 WHIPSAW / -2 MISSED_RUNNER averaged over scored trips —
    persistently negative ⇒ bot exits too early.

    Distinct from neighbours (invariant #10 — do not consolidate):
    ``/api/thesis-drift`` grades OPEN positions against entry rationale;
    ``/api/loser-autopsy`` / ``/api/winner-autopsy`` reduce closed P&L
    to aggregate stats; neither incorporates *post-exit price action*.
    The post-exit drift is the only new piece of data this endpoint
    adds — and it makes the realised-P&L number falsifiable in
    hindsight.

    Query params (all optional):
      ``max_n`` — surfaced trips clip (default 10, clamp 1..50)
      ``hours_back`` — round-trip lookback in hours (default 168, clamp 1..720)

    Advisory only — never gates Opus, adds no caps (AGENTS.md #2/#12).
    """
    try:
        from .analytics.round_trip_postmortem import build_round_trip_postmortem
        from .analytics.round_trips import build_round_trips

        def _qf(name: str, default: float, lo: float, hi: float) -> float:
            try:
                v = float(request.args.get(name, default))
            except (TypeError, ValueError):
                v = default
            return max(lo, min(hi, v))

        max_n = int(_qf("max_n", 10.0, 1.0, 50.0))
        hours_back = _qf("hours_back", 168.0, 1.0, 720.0)

        store = get_store()
        trades = list(reversed(store.recent_trades(2000)))
        rts_all = build_round_trips(trades)

        # Filter to round-trips closed within the lookback window.
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=hours_back)
        def _exit_dt(rt):
            ts = rt.get("exit_ts")
            if not ts:
                return None
            try:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except Exception:
                return None
        rts_recent = [
            rt for rt in rts_all
            if (_exit_dt(rt) is not None and _exit_dt(rt) >= cutoff)
        ]
        # Sort newest-exit-first and clip BEFORE the price fetch so we
        # don't pay yfinance for trips that won't surface.
        rts_recent.sort(key=lambda r: r.get("exit_ts") or "", reverse=True)
        rts_recent = rts_recent[: max(2 * max_n, max_n)]

        tickers = sorted({rt["ticker"] for rt in rts_recent if rt.get("ticker")})
        prices: dict[str, float | None] = {}
        if tickers:
            try:
                from . import market as _mkt
                prices = _mkt.get_prices(tickers)
            except Exception:
                prices = {}

        return jsonify(build_round_trip_postmortem(
            rts_recent, prices, now=now, max_n=max_n,
        ))
    except Exception as e:
        return jsonify({"error": str(e), "state": "ERROR"}), 500


@app.route("/api/runner-heartbeat")
@swr_cached("runner-heartbeat", 20.0)
def runner_heartbeat_api():
    """Is the trading loop itself alive? — the upstream liveness question.

    **SWR-cached (20s), 2026-05-18.** This is the surface a trader checks
    *first* and the dashboard JS polls it every 60s, yet it was the last
    high-traffic core endpoint NOT behind ``swr_cached`` (the invariant #7
    gap that ``/api/state`` closed). Measured **9.45s** under load avg 23
    (host-load storm — the documented #1 pathology) versus ~1ms warm: a pure
    DB+module-global read with no network, so the latency is pure CPU
    starvation, exactly what SWR exists to absorb. The runner cadence is
    ≥1800s (open) / 3600s (closed) with ≥1.25x/2.0x verdict multipliers, so
    a ≤20s stale window can never flip HEALTHY↔LAGGING↔STALLED nor the
    IDLE_STORM efficacy verdict (≥5 cycles × ≥1800s) — the staleness is
    invisible to the verdict while the trader gets an instant answer instead
    of a 9s block. The dashboard thread is independent of the runner thread:
    a dead runner still gets a fresh background recompute of
    ``secs_since_last_decision`` from the frozen ``last_decision_ts`` → it
    correctly goes STALLED, so SWR never masks the very death this detects.
    Inert under pytest unless ``_SWR_TEST_FORCE`` (the ``/api/state``
    contract — the existing ``tests/test_runner_heartbeat.py`` endpoint
    tests stay green on the live path). Locked by
    ``tests/test_runner_heartbeat_swr.py``.

    decision-drought/-reliability/feed-health all reason over rows that
    *exist* in `decisions`; build-info catches a stale code SHA. None close
    a verdict on `now - max(decisions.timestamp)` vs the runner's expected
    cadence, so a dead/wedged `paper_trader.runner` is invisible (the
    ongoing-drought duration just freezes). This compares the newest
    decision's age to OPEN_INTERVAL_S (market open) / CLOSED_INTERVAL_S
    (closed) and verdicts NO_DATA / STALLED / LAGGING / HEALTHY. Network is
    in the endpoint (the thesis_drift split); the builder is pure & never
    raises. Advisory only — never gates Opus, adds no caps (AGENTS.md
    #2/#12)."""
    try:
        from .analytics.runner_heartbeat import build_runner_heartbeat
        from . import market as _mkt
        store = get_store()
        # Window (not just the newest row) so the additive decision-efficacy
        # overlay can see a NO_DECISION storm: a loop cycling on cadence but
        # emitting NO_DECISION every cycle is alive-but-brain-dead and the
        # bare cadence verdict alone would mislabel it HEALTHY. The builder
        # owns interpretation; the endpoint owns the store read (the
        # thesis_drift split). recent_decisions is newest-first.
        decs = store.recent_decisions(20)
        last_ts = decs[0].get("timestamp") if decs else None
        recent_actions = [d.get("action_taken") for d in decs]
        # Reasoning strings (parallel to recent_actions) let the IDLE_STORM
        # verdict diagnose its cause — a host-saturation / quota storm is not
        # cleared by a restart, so the heartbeat must not recommend one.
        recent_reasons = [d.get("reasoning") for d in decs]
        now_utc = datetime.now(timezone.utc)
        hb = build_runner_heartbeat(
            last_ts, _mkt.is_market_open(now_utc), now=now_utc,
            recent_actions=recent_actions, recent_reasons=recent_reasons)
        # Additive: the single-instance-lock state of THE PROCESS SERVING
        # THIS DASHBOARD (the dashboard runs in a runner thread). A runner
        # that booted degraded (no flock — invariant #19 fail-open) may be
        # double-trading the shared book until it upgrades or exits
        # (runner._recheck_singleton_lock). That pathology was previously
        # invisible from every operator surface; surface it next to the
        # loop-liveness verdict. The builder stays pure (the process read is
        # owned by the endpoint — the thesis_drift split); this never
        # overrides the existing verdict (a different, test-locked concern).
        try:
            from . import runner as _runner
            lock = _runner.singleton_lock_state()
        except Exception:
            lock = None
        if isinstance(lock, dict):
            if lock.get("degraded"):
                lock["headline"] = (
                    "DEGRADED — this runner booted WITHOUT the single-"
                    "instance guard; another trader may be double-trading "
                    "the same paper book. Restart paper-trader so one "
                    "guarded instance owns the flock.")
            else:
                lock["headline"] = (
                    "OK — this runner holds the single-instance lock "
                    f"(pid={lock.get('holder_pid')}).")
            hb["singleton_lock"] = lock
        # Additive (2026-05-18): Discord delivery health. EVERY operator
        # notification flows through reporter._send; when it silently fails
        # (the 2026-05-17 `env node` PATH outage) the loop looks perfectly
        # alive while the operator's only surface is dark and the failing
        # channel cannot report its own failure. Surface it next to the
        # loop-liveness verdict so a dead channel is visible. Pure read,
        # never overrides the existing verdict (a different concern); a
        # fault degrades to no block (the singleton_lock precedent).
        try:
            from . import reporter as _reporter
            nh = _reporter.notify_health()
            if isinstance(nh, dict):
                hb["notify"] = nh
        except Exception:
            pass
        return jsonify(hb)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/launcher-restart-loop")
@swr_cached("launcher-restart-loop", 30.0)
def launcher_restart_loop_api():
    """Did the systemd/launcher layer crash-loop on the singleton flock?

    Orthogonal to /api/runner-heartbeat (which says "is the trading loop
    alive *now*"): a wedged launcher can fire dozens of doomed launches per
    minute while the actual trader holds the flock and is perfectly fine.
    /api/runner-heartbeat correctly stays HEALTHY in that scenario, so the
    pathology was invisible from every operator surface. This tails the
    tail of ``logs/runner.log`` and tallies refusal lines.

    Advisory only — never gates Opus, adds no caps (AGENTS.md #2/#12)."""
    try:
        from pathlib import Path
        from .analytics.launcher_restart_loop import build_launcher_restart_loop

        # Resolve logs/runner.log relative to the repo root (dashboard.py
        # is paper_trader/dashboard.py — go up two parents).
        log_path = Path(__file__).resolve().parent.parent / "logs" / "runner.log"
        max_bytes = 64 * 1024  # last ~64KB is plenty for a launcher-loop window
        lines: list[str] = []
        log_size_bytes: int | None = None
        log_age_seconds: float | None = None
        if log_path.exists():
            st = log_path.stat()
            log_size_bytes = int(st.st_size)
            # Wall-clock age of the file — ctime is closest to "when did this
            # log first start growing"; falls back to mtime if unavailable.
            try:
                import time as _t
                ref = st.st_ctime or st.st_mtime
                if ref:
                    age = _t.time() - float(ref)
                    if age > 0:
                        log_age_seconds = age
            except Exception:
                pass
            with log_path.open("rb") as fh:
                if log_size_bytes > max_bytes:
                    fh.seek(log_size_bytes - max_bytes)
                    fh.readline()  # discard partial first line
                raw = fh.read()
            lines = raw.decode("utf-8", errors="replace").splitlines()
        result = build_launcher_restart_loop(
            lines,
            log_size_bytes=log_size_bytes,
            log_age_seconds=log_age_seconds,
        )
        result["log_path"] = str(log_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/no-decision-reasons")
def no_decision_reasons_api():
    """Bucket the WHY of recent NO_DECISION cycles into actionable buckets.

    Complements /api/runner-heartbeat (which says ``IDLE_STORM``: cycles
    are blank) and /api/decision-drought (which prices the P&L cost of
    inaction). This says **WHAT IS CAUSING THE BLANKS** — quota /
    host-saturation / model-empty / parse-failed / retry-failed — so the
    operator's next step is targeted (wait, kill parallel Opus, restart
    runner, or tweak prompt) instead of the generic "restart may help"
    /api/runner-heartbeat falls back to. Pure builder; network in the
    endpoint (the runner-heartbeat split). Advisory only — never gates
    Opus, adds no caps (AGENTS.md #2/#12)."""
    try:
        from .analytics.no_decision_reasons import (
            DEFAULT_WINDOW, build_no_decision_reasons,
        )
        try:
            window = int(request.args.get("window") or DEFAULT_WINDOW)
        except (TypeError, ValueError):
            window = DEFAULT_WINDOW
        window = max(1, min(window, 500))
        store = get_store()
        return jsonify(build_no_decision_reasons(
            store.recent_decisions(window), window=window))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/no-decision-recovery")
def no_decision_recovery_api():
    """Recovery-time grade for NO_DECISION wedges — how long do they last?

    /api/no-decision-reasons says WHY cycles fail; /api/decision-drought
    prices the P&L cost; /api/runner-heartbeat flags IDLE_STORM. None
    answer the on-call operator's *next* question: **"is the current
    wedge anomalous vs history — keep waiting or escalate?"** This wires
    the existing pure builder ``build_no_decision_recovery`` (full unit
    test coverage in tests/test_no_decision_recovery.py) into an HTTP
    endpoint so the dashboard / Discord chat / digital-intern analyst can
    pull the same grade the CLI ``python3 -m
    paper_trader.analytics.no_decision_recovery`` already prints.

    Mirror of the /api/no-decision-reasons access pattern: pure builder,
    network in the endpoint, advisory only (never gates Opus, adds no
    caps — AGENTS.md #2/#12). The verdict ladder is WITHIN_NORMAL /
    ELEVATED / ABNORMAL_WEDGE / NOISE / RECOVERED /
    INSUFFICIENT_HISTORY — see no_decision_recovery.py docstring.
    """
    try:
        from .analytics.no_decision_recovery import (
            DEFAULT_WINDOW, build_no_decision_recovery,
        )
        try:
            window = int(request.args.get("window") or DEFAULT_WINDOW)
        except (TypeError, ValueError):
            window = DEFAULT_WINDOW
        window = max(1, min(window, 500))
        store = get_store()
        return jsonify(build_no_decision_recovery(
            store.recent_decisions(window), window=window))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/churn")
def churn_api():
    """Overtrading & same-name re-entry churn — the turnover question.

    /api/analytics shows raw aggregates; /api/trade-asymmetry grades the
    *payoff* pathology (DISPOSITION_BLEED, breakeven-vs-actual win-rate).
    Neither measures how often the book re-buys a name it just fully
    closed (the live NVDA→LITE→NVDA shape on 2026-05-16) nor the
    round-trips-per-active-day cadence. This composes the single source of
    truth (build_round_trips, AGENTS.md #10 — no re-derived P&L) into the
    fast-re-entry count/rate, cadence, sub-day-loss concentration, and a
    CHURNING / ACTIVE_TURNOVER / BUY_AND_HOLD verdict withheld until
    n≥20 round-trips (trade_asymmetry STABLE idiom). Advisory only — never
    gates Opus, adds no caps (AGENTS.md #2/#12)."""
    try:
        from .analytics.churn import build_churn
        store = get_store()
        # Same trades convention as /api/analytics & /api/trade-asymmetry:
        # oldest → newest (build_round_trips reads in sequence).
        trades = list(reversed(store.recent_trades(2000)))
        return jsonify(build_churn(trades))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/round-trips")
def round_trips_api():
    """Raw closed-round-trip ledger — the canonical underlying data.

    Every realised-P&L analytics endpoint (track-record, churn, streak,
    winner/loser autopsy, trade-asymmetry, session-delta) reduces
    ``analytics.round_trips.build_round_trips`` output to an aggregate;
    the per-RT detail it computes from is never exposed. This returns
    that list directly so frontends/notebooks can drill into the same
    rows the aggregates summarise without re-pairing the trade ledger.

    Distinct from ``/api/closed-positions``, which reads the ``positions``
    table (lot-shaped, ``store.closed_positions``). Those two views can
    disagree on edge cases (option re-pairs, mid-cycle partial exits);
    AGENTS.md #10 designates ``build_round_trips`` as the single source
    of truth for realised-P&L analytics, so this is the view the other
    endpoints' aggregates are computed against.

    Query params:
        ?limit=N  — last N closed round-trips (default 200, max 2000),
                    returned newest-exit-first.

    The ``summary`` block uses the same strict ``> 0`` win split as
    ``/api/analytics`` / ``/api/track-record`` so n_wins + n_losses == n
    and ties at exactly 0 are counted as losses (rounding-artefact pin —
    see comment at the build_round_trips call site near line 6660).
    Advisory only — never gates Opus, adds no caps (AGENTS.md #2/#12).
    """
    try:
        from .analytics.round_trips import build_round_trips
        try:
            limit = max(1, min(int(request.args.get("limit", 200)), 2000))
        except (TypeError, ValueError):
            limit = 200
        store = get_store()
        # Same convention as /api/analytics, /api/churn, /api/streak:
        # oldest → newest (build_round_trips reads in sequence and does
        # not sort).
        trades = list(reversed(store.recent_trades(2000)))
        rts = build_round_trips(trades)
        # Newest-exit-first for display, then clip.
        rts_sorted = sorted(rts, key=lambda r: (r.get("exit_ts") or ""),
                            reverse=True)
        clipped = rts_sorted[:limit]

        pnls = [r.get("pnl_usd") or 0.0 for r in rts_sorted]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        holds = [r["hold_days"] for r in rts_sorted
                 if r.get("hold_days") is not None]
        gross_win = sum(wins)
        gross_loss = abs(sum(losses))
        summary = {
            "n": len(pnls),
            "n_wins": len(wins),
            "n_losses": len(losses),
            "win_rate_pct": (round(len(wins) / len(pnls) * 100, 2)
                             if pnls else None),
            "total_pnl_usd": round(sum(pnls), 2) if pnls else 0.0,
            "gross_win_usd": round(gross_win, 2) if wins else 0.0,
            "gross_loss_usd": round(gross_loss, 2) if losses else 0.0,
            "profit_factor": (round(gross_win / gross_loss, 2)
                              if gross_loss > 1e-9 else None),
            "avg_hold_days": (round(sum(holds) / len(holds), 2)
                              if holds else None),
            "returned": len(clipped),
            "truncated": len(clipped) < len(pnls),
        }
        return jsonify({
            "round_trips": clipped,
            "summary": summary,
            "limit": limit,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/thesis-drift")
@swr_cached("thesis-drift", 60.0)
def thesis_drift_api():
    """Entry-thesis vs current-reality, per open position.

    /api/position-thesis fuses *current* signals; /api/suggestions
    re-derives an action from scratch. Neither re-tests a holding against
    *the reason it was opened for* — which is sitting verbatim in the
    opening fill's trades.reason. This anchors each open position on its
    own opening BUY rationale (invariant #8: the BUY nearest opened_at is
    this lot's opener even on a re-entered name) and grades INTACT /
    WEAKENING / BROKEN off objective, deterministic inputs (P/L since
    entry, hold time, and optional live quant/news). Advisory only —
    never gates Opus, adds no caps (AGENTS.md #2/#12)."""
    try:
        from .analytics.thesis_drift import build_thesis_drift
        store = get_store()
        positions = store.open_positions()
        trades = store.recent_trades(2000)
        signals = None
        try:
            tickers = sorted({p["ticker"] for p in positions
                              if p.get("ticker")})
            if tickers:
                from .strategy import get_quant_signals_live
                quant = get_quant_signals_live(tickers) or {}
                news = _ticker_news_pulse(tickers, hours=24)
                signals = {}
                for tk in tickers:
                    q = quant.get(tk, {}) or {}
                    nrec = news.get(tk.upper(), {}) or {}
                    signals[tk] = {
                        "rsi": q.get("RSI"),
                        "macd": q.get("MACD"),
                        "mom_5d": q.get("mom_5d"),
                        "mom_20d": q.get("mom_20d"),
                        "news_count": nrec.get("n", 0),
                        "news_urgent": bool(nrec.get("urgent", 0)),
                    }
        except Exception:
            signals = None  # builder degrades to price-only health
        return jsonify(build_thesis_drift(positions, trades, signals))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/reasoning-coherence")
def reasoning_coherence_api():
    """How stable is Opus's HOLD justification across consecutive cycles?

    Distinct from ``/api/thesis-drift`` (which re-tests an OPEN POSITION's
    entry rationale against current state — a position-level question) and
    ``/api/decision-drought`` (which counts consecutive NO_DECISION silences,
    not the content of HOLDs that bracket them) and ``/api/decision-
    forensics`` (which diagnoses ONE latest decision, no across-time view).

    This is the across-time complement: token-set Jaccard similarity between
    consecutive HOLD-reasoning prose, summarised into a single ``regime``
    verdict — ``STABLE_THESIS`` (Opus reiterating same justification cycle-
    to-cycle, conviction signal) / ``DRIFTING`` (reasoning evolves between
    holds) / ``RAPID_DRIFT`` (each HOLD cites different content — confusion
    signal). ``state`` ladder mirrors neighbouring diagnostics: ``NO_DATA``
    (no parseable HOLD reasoning), ``INSUFFICIENT`` (< 3 HOLD pairs in
    window), ``OK`` (verdict emitted).

    ``?limit=`` decisions to scan (clamped 5..500, default 100). Builder is
    pure: degrades to ``NO_DATA`` on any store failure rather than 500. Cheap
    by construction (no yfinance / no LLM) — not behind ``@swr_cached``.
    Observational only — never gates Opus, never injected into the decision
    prompt (invariants #2/#12)."""
    try:
        from .analytics.reasoning_coherence import build_reasoning_coherence
        try:
            limit = int(request.args.get("limit", 100))
        except (TypeError, ValueError):
            limit = 100
        limit = max(5, min(500, limit))
        store = get_store()
        decisions = store.recent_decisions(limit=limit)
        out = build_reasoning_coherence(decisions)
        out["window_limit"] = limit
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/reasoning-themes")
def reasoning_themes_api():
    """Top recurring phrases across recent Opus decision reasoning prose.

    Descriptive complement to ``/api/reasoning-coherence`` (which measures
    pair-wise Jaccard stability between consecutive HOLDs — a *vocabulary
    stability* metric, not a *vocabulary distribution* one). Surfaces
    what topics actually dominate Opus's mental loops: across a window
    of N decisions, the top phrases ranked by **decisions_mentioning**
    (breadth, not loudness — a phrase repeated 30× in one verbose row
    counts as ONE decision, while a phrase recurring across 12 different
    decisions ranks high).

    1-grams and 2-grams compete in the same leaderboard; on ties the
    bigram wins (a bigram is more informative than its component words).

    ``?limit=`` decisions to scan (clamped 5..500, default 100).
    ``?top_k=`` phrases to return (clamped 3..50, default 10).
    ``?include_bigrams=0`` to disable bigrams (1-grams only).

    Pure builder, cheap by construction (no yfinance / no LLM) — not
    behind the SWR cache. Observational only — never gates Opus, never
    injected into the decision prompt (invariants #2/#12)."""
    try:
        from .analytics.reasoning_themes import build_reasoning_themes
        try:
            limit = int(request.args.get("limit", 100))
        except (TypeError, ValueError):
            limit = 100
        limit = max(5, min(500, limit))
        try:
            top_k = int(request.args.get("top_k", 10))
        except (TypeError, ValueError):
            top_k = 10
        include_bigrams_raw = request.args.get("include_bigrams", "1")
        include_bigrams = str(include_bigrams_raw).strip().lower() not in (
            "0", "false", "no", "off",
        )
        store = get_store()
        decisions = store.recent_decisions(limit=limit)
        out = build_reasoning_themes(
            decisions, top_k=top_k, include_bigrams=include_bigrams,
        )
        out["window_limit"] = limit
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/reasoning-action-verbs")
def reasoning_action_verbs_api():
    """Single-decision internal consistency: does Opus's natural-language
    reasoning *verb* agree with the structured ``action`` field on the same
    decision row?

    Complements the existing reasoning surfaces — ``/api/reasoning-themes``
    (topic distribution across decisions), ``/api/reasoning-coherence``
    (pair-wise Jaccard stability between consecutive HOLDs), and
    ``/api/decision-confidence`` (aggregate self-rated conviction). None of
    those grade the *within-row* mismatch this endpoint exposes: a row whose
    structured action is ``HOLD`` while the prose verbalises a buy/sell
    intent (or vice-versa) is the exact LLM failure mode that slips past
    a JSON-parsing reviewer but reads alarming to a human operator. The
    builder counts BUY / SELL / HOLD cue verbs inside each
    ``decision.reasoning`` text (with negation- and hedge-window dropping —
    "would not add" / "would add IF earnings beat" carry zero votes),
    classifies the dominant leaning, and tags any mismatch against the
    structured action with a specific verdict
    (``BULLISH_INSIDE_HOLD`` / ``BEARISH_INSIDE_HOLD`` /
    ``BEARISH_INSIDE_BUY`` / ``BULLISH_INSIDE_SELL`` /
    ``DIRECTION_INSIDE_NO_DECISION``).

    ``state`` ladder: ``INSUFFICIENT`` (< 10 parseable) → ``CLEAN`` (< 5%
    mismatch) → ``MILD`` (5-15%) → ``NOTABLE`` (15-30%) → ``ALARMING``
    (≥ 30%).

    ``?limit=`` decisions to scan (clamped 5..500, default 100). Pure
    builder, cheap (no yfinance / no LLM) — not behind the SWR cache.
    Observational only — never gates Opus, never injected into the
    decision prompt (invariants #2/#12; the ``reasoning_themes`` /
    ``decision_confidence`` precedent)."""
    try:
        from .analytics.reasoning_action_verbs import (
            build_reasoning_action_verbs,
        )
        try:
            limit = int(request.args.get("limit", 100))
        except (TypeError, ValueError):
            limit = 100
        limit = max(5, min(500, limit))
        store = get_store()
        decisions = store.recent_decisions(limit=limit)
        out = build_reasoning_action_verbs(decisions)
        out["window_limit"] = limit
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/decision-confidence")
def decision_confidence_api():
    """Aggregate Opus's self-rated ``confidence`` across recent decisions.

    The decision envelope ``{"decision": {"confidence": 0.7, ...}}``
    carries a 0..1 conviction value on every parseable row — yet nothing
    in the ~80-endpoint surface aggregates it. ``/api/scorer-confidence``
    is the **DecisionScorer** (small CPU MLP on the backtest side), not
    Opus; ``/api/decision-forensics`` reads one decision; ``/api/
    reasoning-coherence`` measures pair stability. An operator scanning
    a paralysed week cannot tell from those whether Opus is **confidently**
    sitting on its hands (high-conviction HOLDs around a binary event)
    or **uncertainly** doing nothing.

    Returns median / mean / min / max / 4-bucket histogram (low /
    medium / high / very_high) / per-action breakdown / recent-vs-older
    trend / regime verdict (``CAUTIOUS`` / ``NEUTRAL`` / ``CONVICTED``).

    ``state`` ladder: ``NO_DATA`` (no parseable confidence) → ``INSUFFICIENT``
    (< 5 samples; raw stats emitted, regime withheld) → ``OK``.

    ``?limit=`` decisions to scan (clamped 5..500, default 100). Pure
    builder, cheap (no yfinance / no LLM) — not behind the SWR cache.
    Observational only — never gates Opus, never injected into the
    decision prompt (invariants #2/#12)."""
    try:
        from .analytics.decision_confidence import build_decision_confidence
        try:
            limit = int(request.args.get("limit", 100))
        except (TypeError, ValueError):
            limit = 100
        limit = max(5, min(500, limit))
        store = get_store()
        decisions = store.recent_decisions(limit=limit)
        out = build_decision_confidence(decisions)
        out["window_limit"] = limit
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/self-review")
def self_review_api():
    """The behavioural mirror the live trader now sees in its own prompt.

    Canonical single source (AGENTS.md invariant #10): composes
    build_trade_asymmetry + build_capital_paralysis + build_open_attribution
    verbatim — no re-derived P&L — into one report plus the exact
    `prompt_block` string injected into strategy._build_payload every live
    decision cycle. Observational only: it states facts and the builders' own
    calibrated verdicts, issues no directives, imposes no caps, and reaffirms
    full autonomy in its own preamble — it does not violate the "no hard risk
    limits / Opus has full autonomy" invariant (#2/#12), exactly as
    /api/capital-paralysis and /api/liquidity are advisory-only. Exposing it
    here keeps the dashboard, a future chat single-source and the in-prompt
    block from ever drifting apart (the inline-copy hazard #10 warns of)."""
    try:
        from .analytics.self_review import build_self_review
        store = get_store()
        # trades store-native newest-first — build_self_review reverses
        # internally for the asymmetry consumer, exactly as the two endpoints
        # above do (build_liquidity wants newest-first, build_round_trips
        # wants oldest→newest).
        return jsonify(build_self_review(
            store.get_portfolio(),
            store.open_positions(),
            store.recent_trades(2000),
            store.recent_decisions(limit=3000),
            store.equity_curve(limit=5000),
        ))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/scorecard")
def scorecard_api():
    """Behavioural-verdict alignment router across the five pure
    behavioural builders (trade_asymmetry, churn, capital_paralysis,
    decision_reliability, open_attribution).

    Synthesis without a new opinion: it classifies each builder's *own*
    verdict into FLAG/OK/IMMATURE, counts where independent checks concur on
    a theme, and forwards the builders' own headlines verbatim (single source
    of truth, AGENTS.md invariant #10). No grade, no directive, no cap —
    descriptive only, exactly the /api/self-review observational precedent
    (invariants #2/#12). Unlike self-review it is NOT injected into the live
    decision prompt; it is dashboard/chat only. Same store reads as
    /api/self-review so the two can't drift."""
    try:
        from .analytics.trader_scorecard import build_trader_scorecard
        store = get_store()
        return jsonify(build_trader_scorecard(
            store.get_portfolio(),
            store.open_positions(),
            store.recent_trades(2000),
            store.recent_decisions(limit=3000),
            store.equity_curve(limit=5000),
        ))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/desk-pulse")
def desk_pulse_api():
    """The single pure-DB "is the desk OK right now?" digest — money +
    loop-liveness + code-staleness + the one behavioural flag to look at
    first, in one fast dependency-free call.

    /api/scorecard is behavioural-only (no money KPIs); /api/state is the
    heavy everything-dump and the slowest endpoint here; :8888's
    /api/command-center gets its trader half by cross-fetching :8090 so it
    blanks exactly when :8090 is slow/wedged (observed live 2026-05-17).
    This composes only the network-free single-source-of-truth builders
    (build_round_trips — the same strict >0 win split as /api/analytics;
    build_runner_heartbeat; trader_scorecard.focus) plus the build-info
    SHA — no yfinance, no articles.db, no scorer — so it answers in
    milliseconds even when every yfinance-backed panel is timing out, and
    `python -m paper_trader.analytics.desk_pulse` prints the same digest
    from a terminal when the process itself is wedged. Router, not a
    grader: mints no opinion, NOT injected into the live prompt — advisory
    only, never gates Opus, adds no caps (AGENTS.md #2/#12/#10)."""
    try:
        from .analytics.desk_pulse import build_desk_pulse
        from . import market as _mkt
        store = get_store()
        head, behind = _head_sha_and_behind()
        build_info = {
            "boot_sha": _BOOT_SHA,
            "head_sha": head,
            "behind": behind,
            "stale": bool(_BOOT_SHA and head and head != _BOOT_SHA),
        }
        now_utc = datetime.now(timezone.utc)
        return jsonify(build_desk_pulse(
            store.get_portfolio(),
            store.open_positions(),
            store.recent_trades(2000),
            store.recent_decisions(limit=3000),
            store.equity_curve(limit=5000),
            build_info=build_info,
            market_open=_mkt.is_market_open(now_utc),
            initial_cash=INITIAL_CASH,
            now=now_utc,
        ))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Daily-bar history cache for the news-edge endpoint. Keyed by ticker; daily
# bars don't change intraday in a way that matters for forward-return analysis
# of *past* articles, so a generous TTL keeps the endpoint snappy after the
# first build without re-hammering yfinance every refresh.
_NEWS_EDGE_PX_CACHE: dict[str, tuple[list[tuple[str, float]], float]] = {}
_NEWS_EDGE_PX_TTL = 1800.0  # 30 min


def _daily_history_cached(ticker: str, period: str = "3mo") -> list[tuple[str, float]]:
    import time as _t
    hit = _NEWS_EDGE_PX_CACHE.get(ticker)
    if hit and _t.time() - hit[1] < _NEWS_EDGE_PX_TTL:
        return hit[0]

    def _fetch() -> list[tuple[str, float]]:
        import yfinance as yf
        h = yf.Ticker(ticker).history(period=period, auto_adjust=False)
        return [(idx.strftime("%Y-%m-%d"), float(c))
                for idx, c in zip(h.index, h["Close"]) if c == c]

    # Hard wall-clock bound on the yfinance call. Without it a stalled
    # HTTPS socket pins an _SWR_EXEC worker indefinitely (correlation /
    # news-edge / source-edge / signal-followthrough all route here), and a
    # few simultaneous hangs dark every SWR panel forever. `default=[]`
    # keeps the pre-existing failure semantics identical to the old bare
    # `except Exception: bars = []` path (a transient failure is cached
    # empty for the TTL exactly as before — the *only* behavioural change
    # is that a hang now degrades in _NET_TIMEOUT_S instead of never).
    bars = _bounded_call(_fetch, timeout_s=_NET_TIMEOUT_S, default=[],
                          label=f"daily-history {ticker}")
    _NEWS_EDGE_PX_CACHE[ticker] = (bars, _t.time())
    return bars


@app.route("/api/news-edge")
@swr_cached("news-edge", 120.0)
def news_edge_api():
    """Does digital-intern's scored news actually predict moves?

    For every live (non-backtest) scored article that names a watchlist
    ticker, look at that ticker's 1/3/5-trading-day forward return, both raw
    and SPY-abnormal, banded by ai_score. The verdict is judged on abnormal
    return only — a flat or inverted score→return curve means the score is
    noise. ``?days=`` (lookback, default 30) and ``?min_score=`` (default 2.0)
    are tunable. Validates the core premise of the whole stack."""
    try:
        from .analytics.news_edge import build_news_edge
        from .strategy import WATCHLIST

        days = max(7, min(120, int(request.args.get("days", 30))))
        min_score = float(request.args.get("min_score", 2.0))

        path = _articles_db_path()
        if path is None:
            return jsonify({"error": "articles.db not found", "bands": [],
                            "verdict": "NO_DATA"}), 200

        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=8)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT title, full_text, ai_score, urgency, first_seen "
                "FROM articles WHERE ai_score >= ? AND first_seen >= ? "
                "AND url NOT LIKE 'backtest://%' "
                "AND source NOT LIKE 'backtest_%' "
                "AND source NOT LIKE 'opus_annotation%' "
                "ORDER BY first_seen DESC LIMIT 4000",
                (min_score, since),
            ).fetchall()
        finally:
            conn.close()

        arts = []
        for r in rows:
            body = ""
            try:
                if r["full_text"]:
                    body = zlib.decompress(r["full_text"]).decode(
                        "utf-8", errors="replace")
            except Exception:
                body = ""
            arts.append({
                "text": f"{r['title'] or ''} {body}",
                "ai_score": r["ai_score"],
                "urgency": r["urgency"],
                "published": r["first_seen"],
            })

        # Only fetch prices for watchlist tickers that actually appear, most
        # frequent first, capped so a cold request can't stall on dozens of
        # yfinance round-trips.
        freq: dict[str, int] = {}
        pats = {tk: re.compile(rf"(?:\$|\b){re.escape(tk)}\b") for tk in WATCHLIST}
        for a in arts:
            up = a["text"].upper()
            for tk, pat in pats.items():
                if pat.search(up):
                    freq[tk] = freq.get(tk, 0) + 1
        wanted = [tk for tk, _ in sorted(
            freq.items(), key=lambda kv: -kv[1])][:30]

        price_history = {tk: _daily_history_cached(tk) for tk in wanted}
        spy_history = _daily_history_cached("SPY")

        result = build_news_edge(arts, price_history, spy_history, WATCHLIST)
        result["lookback_days"] = days
        result["min_score"] = min_score
        result["n_tickers_priced"] = len([tk for tk in wanted
                                          if price_history.get(tk)])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "bands": [],
                        "verdict": "ERROR"}), 500


# ───────── Decision-reliability + funded-suggestions (2026-05-16, agent 4) ─────────
# Appended at the tail of the route section (not interleaved) so a concurrent
# core-review pass editing existing endpoints doesn't collide on merge.


@app.route("/api/decision-reliability")
def decision_reliability_api():
    """The *true current-regime* NO_DECISION rate, not the inflated headline.

    decision-health/forensics/drought measure the rate / why / cost, but the
    headline % is dominated by legacy pre-diagnostics rows that stop accruing
    once the runner restarts onto diagnostic code. This partitions the log at
    the newest legacy failure and reports the post-restart rate with explicit
    sample-size honesty + a restart-recommended signal. Pure composition of
    build_decision_forensics + build_decision_drought (single source of truth,
    no re-derived metrics). Advisory only — never gates Opus (invariants
    #2/#12)."""
    try:
        from .analytics.decision_reliability import build_decision_reliability
        store = get_store()
        return jsonify(build_decision_reliability(
            store.recent_decisions(limit=3000),
            store.equity_curve(limit=5000),
        ))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/funded-suggestions")
def funded_suggestions_api():
    """Pair each unfundable BUY/ADD idea with the sale that funds it.

    Composes the existing /api/suggestions list with build_capital_paralysis'
    unlock ladder (single source of truth, no re-derived metrics — neither is
    refactored). When PINNED, attaches the minimum prefix of desk-cut-priority
    sales whose cumulative freed cash covers an advisory suggested notional.
    Advisory only — never gates Opus, sizes nothing, adds no caps
    (invariants #2/#12)."""
    try:
        from .analytics.capital_paralysis import build_capital_paralysis
        from .analytics.funded_suggestions import build_funded_suggestions

        # Reuse the existing suggestions view verbatim (no refactor).
        resp = suggestions_api()
        if isinstance(resp, tuple):
            resp = resp[0]
        sug_payload = resp.get_json(silent=True) or {}
        suggestions = sug_payload.get("suggestions", [])

        store = get_store()
        paralysis = build_capital_paralysis(
            store.get_portfolio(),
            store.open_positions(),
            store.recent_trades(200),
            store.recent_decisions(limit=3000),
            store.equity_curve(limit=5000),
        )
        out = build_funded_suggestions(suggestions, paralysis)
        # Surface a suggestions-side error rather than masking it as "no ideas".
        if sug_payload.get("error"):
            out["suggestions_error"] = sug_payload["error"]
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/signal-followthrough")
def signal_followthrough_api():
    """Is the trader actually *using* its own news edge?

    news-edge grades the signal alone (ignoring the bot); decision-drought
    grades inaction vs SPY (not vs the signals present). This grades the
    *join*: of the high-score live signals visible at decision time (the
    exact ``get_top_signals(hours=2, min_score=4.0)`` window strategy.decide
    feeds Opus), did the trader transact that ticker, and did the signals it
    ACTED on beat — forward, SPY-abnormal — the ones it IGNORED? A near-zero
    follow-through ⇒ IGNORING_FEED; negative selection edge ⇒ MISUSING_SIGNALS.
    ``?days=`` (lookback, default 30) / ``?min_score=`` (default 4.0). Pure
    composition of build_signal_followthrough + news_edge resolution helpers
    (single source of truth). Advisory only — never gates Opus
    (invariants #2/#12)."""
    try:
        from .analytics.signal_followthrough import (
            _fetch_live_articles,
            build_signal_followthrough,
        )
        from .strategy import WATCHLIST

        days = max(7, min(120, int(request.args.get("days", 30))))
        min_score = float(request.args.get("min_score", 4.0))

        path = _articles_db_path()
        if path is None:
            return jsonify({"error": "articles.db not found",
                            "verdict": "NO_DATA", "acted": {}, "ignored": {}}), 200

        store = get_store()
        decs = store.recent_decisions(limit=3000)
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        arts = _fetch_live_articles(str(path), since, min_score=min_score)

        # Price only the watchlist tickers that actually appear in the feed,
        # most-frequent first, capped — same cold-start guard as news-edge.
        freq: dict[str, int] = {}
        pats = {tk: re.compile(rf"(?:\$|\b){re.escape(tk)}\b") for tk in WATCHLIST}
        for a in arts:
            up = a["text"].upper()
            for tk, pat in pats.items():
                if pat.search(up):
                    freq[tk] = freq.get(tk, 0) + 1
        wanted = [tk for tk, _ in sorted(freq.items(), key=lambda kv: -kv[1])][:30]

        price_history = {tk: _daily_history_cached(tk) for tk in wanted}
        spy_history = _daily_history_cached("SPY")

        result = build_signal_followthrough(
            decs, arts, price_history, spy_history, WATCHLIST,
            lookback_hours=2.0, min_score=min_score)
        result["lookback_days"] = days
        result["min_score"] = min_score
        result["n_tickers_priced"] = len([tk for tk in wanted
                                          if price_history.get(tk)])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "verdict": "ERROR",
                        "acted": {}, "ignored": {}}), 500


@app.route("/api/source-edge")
@swr_cached("source-edge", 120.0)
def source_edge_api():
    """Which of digital-intern's ~17 collectors is worth trusting?

    news-edge grades the *score* (8.0 vs 3.0 headline); signal-followthrough
    grades whether the bot *acted*. Neither answers the operator's question:
    of the *sources* feeding the pipeline, whose scored headlines actually
    precede abnormal moves and which are noise to cut/down-weight? This bins
    every scored live article by collector family (the dirty `source` column
    normalised once, see source_edge._source_family) and reports the 1/3/5d
    forward return, raw + SPY-abnormal, pooled across score bands per family.
    Pooled (not per-band) because digital-intern's live news is only days-deep
    — the pooled view is both the actionable one (cut a collector) and the one
    that reaches a usable sample first. ``?days=`` (lookback, default 30) /
    ``?min_score=`` (default 2.0). Verdict matures with history exactly like
    news-edge (NO_DATA → INSUFFICIENT_DATA → EDGE_FOUND/NO_EDGE). Pure
    composition reusing news_edge resolution helpers (single source of truth).
    Advisory only — never gates Opus, adds no caps (invariants #2/#12)."""
    try:
        from .analytics.source_edge import (
            _fetch_source_articles,
            build_source_edge,
        )
        from .strategy import WATCHLIST

        days = max(7, min(120, int(request.args.get("days", 30))))
        min_score = float(request.args.get("min_score", 2.0))

        path = _articles_db_path()
        if path is None:
            return jsonify({"error": "articles.db not found",
                            "sources": [], "verdict": "NO_DATA"}), 200

        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        arts = _fetch_source_articles(str(path), since, min_score=min_score)

        # Price only the watchlist tickers that actually appear, most-frequent
        # first, capped — same cold-start guard as news-edge.
        freq: dict[str, int] = {}
        pats = {tk: re.compile(rf"(?:\$|\b){re.escape(tk)}\b") for tk in WATCHLIST}
        for a in arts:
            up = a["text"].upper()
            for tk, pat in pats.items():
                if pat.search(up):
                    freq[tk] = freq.get(tk, 0) + 1
        wanted = [tk for tk, _ in sorted(freq.items(), key=lambda kv: -kv[1])][:30]

        price_history = {tk: _daily_history_cached(tk) for tk in wanted}
        spy_history = _daily_history_cached("SPY")

        result = build_source_edge(arts, price_history, spy_history, WATCHLIST)
        result["lookback_days"] = days
        result["n_tickers_priced"] = len([tk for tk in wanted
                                          if price_history.get(tk)])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "sources": [],
                        "verdict": "ERROR"}), 500


def _feed_db_probe(db_path: str, want_counts: bool = False) -> dict:
    """Read newest *live* first_seen (and optionally 2h/24h live counts) from
    one candidate articles.db. The live-only clause is inlined verbatim (the
    canonical AGENTS.md invariant #1/#3 fragment, mirroring signals.py and
    data_feed_api) — a planted backtest:// row must never read as the freshest
    article or the split-brain detector would be defeated by training data.
    Returns ``{exists, newest, live_2h, live_24h}``; never raises."""
    out = {"exists": False, "newest": None, "live_2h": 0, "live_24h": 0}
    try:
        from pathlib import Path as _P
        if not _P(db_path).exists():
            return out
        out["exists"] = True
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=3.0)
        try:
            live_clause = (
                "url NOT LIKE 'backtest://%' "
                "AND source NOT LIKE 'backtest_%' "
                "AND source NOT LIKE 'opus_annotation%'"
            )
            row = conn.execute(
                f"SELECT MAX(first_seen) FROM articles WHERE {live_clause}"
            ).fetchone()
            out["newest"] = row[0] if row else None
            if want_counts:
                # Cut-offs computed as ISO strings in Python, mirroring
                # signals.get_top_signals exactly — NOT sqlite's
                # datetime('now',...) (space-separated), which would
                # lexically mis-compare against the 'T'-separated ISO
                # first_seen the way data_feed_api's count subtly does.
                now = datetime.now(timezone.utc)
                s2 = (now - timedelta(hours=2)).isoformat()
                s24 = (now - timedelta(hours=24)).isoformat()
                out["live_2h"] = int(conn.execute(
                    f"SELECT COUNT(*) FROM articles WHERE "
                    f"first_seen >= ? AND {live_clause}", (s2,)
                ).fetchone()[0] or 0)
                out["live_24h"] = int(conn.execute(
                    f"SELECT COUNT(*) FROM articles WHERE "
                    f"first_seen >= ? AND {live_clause}", (s24,)
                ).fetchone()[0] or 0)
        finally:
            conn.close()
    except Exception:
        return out
    return out


@app.route("/api/feed-health")
@swr_cached("feed-health", 60.0)
def feed_health_api():
    """Is the live trader actually *seeing* any news, or flying blind?

    Every other panel measures behaviour *after* a decision and assumes the
    trader received signals. None answer the prior question when the book just
    HOLDs for hours. strategy.decide() builds Opus's prompt from
    signals.get_top_signals(hours=2) against signals._db_path(); if that DB is
    stale the prompt's signal block is empty, signal_count is recorded 0, and a
    0-signal HOLD is indistinguishable from a deliberate one. /api/data-feed
    shows raw counts with no verdict, path, or link to the decision log — a
    stale `articles_24h: 3801` reads as healthy. This adds the consecutive
    0-signal *decision streak*, the resolved DB path + its newest-live age, and
    split-brain detection (signals._db_path() prefers the USB mount while the
    daemon / unified_dashboard prefer the local copy — opposite precedence, so
    a stale USB mirror silently blinds the trader while every other surface
    reads the fresh one). Pure core: analytics/feed_health.build_feed_health
    (this endpoint does all the SQLite/filesystem IO; the builder stays pure).
    Advisory only — never gates Opus, adds no caps (invariants #2/#12)."""
    try:
        from . import signals as _sig
        from .analytics.feed_health import build_feed_health

        resolved = _sig._db_path()
        resolved_str = str(resolved)

        # The two candidates signals._db_path() chooses between, de-duped and
        # order-preserving. (Listing order here is presentational only — the
        # live trader resolves by *freshness* via signals._choose(), LOCAL-first
        # on a tie since 6227cd5; legacy_path below models the old USB-first
        # existence resolver, which is what split-brain detection compares to.)
        seen: set[str] = set()
        cand_paths: list[str] = []
        for p in (_sig.USB_DB, _sig.LOCAL_DB):
            ps = str(p)
            if ps not in seen:
                seen.add(ps)
                cand_paths.append(ps)

        # What a process still running the pre-freshness-aware resolver would
        # read (existence-first). When it differs from the freshly-resolved DB
        # and is materially staler, a stale runner/dashboard process
        # (/api/build-info `stale`) is blind and needs a RESTART — the
        # canonical split-brain shape now that _db_path() is freshness-aware.
        legacy_str = str(_sig._legacy_choice())

        candidates = []
        probe_by_path: dict[str, dict] = {}
        resolved_probe = {"exists": False, "newest": None,
                          "live_2h": 0, "live_24h": 0}
        for ps in cand_paths:
            probe = _feed_db_probe(ps, want_counts=(ps == resolved_str))
            probe_by_path[ps] = probe
            candidates.append({"path": ps, "exists": probe["exists"],
                               "newest": probe["newest"]})
            if ps == resolved_str:
                resolved_probe = probe

        legacy_probe = probe_by_path.get(legacy_str)
        feed = {
            "resolved_path": resolved_str if resolved_probe["exists"] else None,
            "resolved_newest": resolved_probe["newest"],
            "resolved_live_2h": resolved_probe["live_2h"],
            "resolved_live_24h": resolved_probe["live_24h"],
            "legacy_path": (legacy_str if legacy_probe
                            and legacy_probe["exists"] else None),
            "legacy_newest": legacy_probe["newest"] if legacy_probe else None,
            "candidates": candidates,
        }
        store = get_store()
        return jsonify(build_feed_health(
            store.recent_decisions(limit=3000), feed))
    except Exception as e:
        return jsonify({"error": str(e), "verdict": "ERROR"}), 500


@app.route("/api/session-delta")
def session_delta_api():
    """What materially changed since you last looked.

    Every other panel is a current-state snapshot; /api/daily-recap is a
    calendar-"today" aggregate; /api/command-center is a one-shot current
    aggregate. None answers the operator's first question on reopening the
    dashboard after being away — "what happened while I was gone?" — which
    today means scanning ~19 panels. This is a ranked material-event timeline
    over a parameterised look-back window (fills, round-trip closes with
    realised P&L consumed verbatim from build_round_trips, equity move +
    SPY-relative alpha, intra-window drawdown, and an idle-cycle fact),
    reading only paper_trader.db (full history — no articles.db dependency).
    ``?minutes=`` (look-back, default 360 = 6h, clamped [5, 10080]) or an
    explicit ``?since=`` ISO-8601 instant. Advisory only — dashboard/chat
    surface, never injected into the decision prompt, never gates Opus, adds
    no caps (invariants #2/#12). Pure core:
    analytics/session_delta.build_session_delta."""
    try:
        from .analytics.session_delta import build_session_delta

        now = datetime.now(timezone.utc)
        since_arg = request.args.get("since")
        since_dt = None
        if since_arg:
            try:
                since_dt = datetime.fromisoformat(
                    since_arg.replace("Z", "+00:00"))
                if since_dt.tzinfo is None:
                    since_dt = since_dt.replace(tzinfo=timezone.utc)
            except Exception:
                since_dt = None  # fall through to the minutes default
        if since_dt is None:
            minutes = max(5, min(10080, int(request.args.get("minutes", 360))))
            since_dt = now - timedelta(minutes=minutes)

        store = get_store()
        return jsonify(build_session_delta(
            list(reversed(store.recent_trades(2000))),
            store.recent_decisions(500),
            store.equity_curve(1000),
            since_dt,
            now,
        ))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/today-action-tape")
@swr_cached("today-action-tape", 15.0)
def today_action_tape_api():
    """Chronological tape of every TRADE and every DECISION since UTC
    midnight.

    Every other timeline panel is either ranked-by-materiality
    (/api/session-delta — caps at 40, synthesises EQUITY_MOVE rows), or
    an aggregate (/api/daily-recap — totals only, no per-cycle rows), or
    behavioural-only (/api/decision-forensics et al — no trade rows
    interleaved). This is the *literal* "what did my bot do today" tape:
    every trades row + every decisions row (HOLD / NO_DECISION /
    BLOCKED included), oldest → newest, no ranking, no cap (limited only
    by the underlying store query).

    Query params (all optional):
      ``since`` — ISO-8601 instant; default is today's UTC midnight.
      ``minutes`` — when ``since`` is absent, look back this many
        minutes from now instead of anchoring to UTC midnight. Clamped
        [5, 10080] (5 min to 7 days). Useful for an "any 4-hour" recap.

    Advisory only — dashboard/chat surface, never injected into the
    decision prompt, never gates Opus, adds no caps (invariants #2/#12).
    Pure core: analytics/today_action_tape.build_today_action_tape."""
    try:
        from .analytics.today_action_tape import build_today_action_tape

        now = datetime.now(timezone.utc)
        since_arg = request.args.get("since")
        since_dt: datetime | None = None
        if since_arg:
            try:
                since_dt = datetime.fromisoformat(
                    since_arg.replace("Z", "+00:00"))
                if since_dt.tzinfo is None:
                    since_dt = since_dt.replace(tzinfo=timezone.utc)
            except Exception:
                since_dt = None
        if since_dt is None and request.args.get("minutes") is not None:
            try:
                m = int(request.args.get("minutes", 0))
            except (TypeError, ValueError):
                m = 0
            m = max(5, min(10080, m))
            since_dt = now - timedelta(minutes=m)

        store = get_store()
        return jsonify(build_today_action_tape(
            list(reversed(store.recent_trades(2000))),
            store.recent_decisions(2000),
            now=now,
            since=since_dt,
        ))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/mark-integrity")
def mark_integrity_api():
    """How much of the displayed book value is *fictional* right now?

    When yfinance returns nothing for a held name, strategy._mark_to_market
    falls back to avg_cost and flags the row stale_mark=True (the live
    2026-05-17 pathology: MU 0.5 @ 724.12, current_price == avg_cost, P/L
    $0.00). That flag is surfaced *per position* to Opus & Discord, but no
    panel answers the aggregate: what share of total book value is marked at
    cost, so /api/analytics Sharpe, /api/drawdown, the equity curve and the
    headline P&L are all quietly partially false? Uses the **read-only**
    snapshot (strategy.portfolio_snapshot_readonly — shares _mark_to_market
    with the live path so it can't drift, AGENTS.md #10; never writes from
    the dashboard thread). Advisory only — never gates Opus, adds no caps
    (AGENTS.md #2/#12). Pure core: analytics/mark_integrity.py."""
    try:
        from .analytics.mark_integrity import build_mark_integrity
        from .strategy import portfolio_snapshot_readonly
        store = get_store()
        snap = portfolio_snapshot_readonly(store)
        return jsonify(build_mark_integrity(snap.get("positions") or []))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/decision-context")
@swr_cached("decision-context", 30.0)
def decision_context_api():
    """What is the live trader actually being shown right now?

    The single biggest blind spot on the desk: `decisions` stores only
    action_taken + reasoning, and the only raw capture is 1000 chars of the
    *response* on a parse failure. When the trader spends cycle after cycle
    on NO_DECISION (timeout/empty) / flat HOLD (the dominant 2026-05-17 live
    pattern) an operator cannot see the decision **input**. This rebuilds it
    on demand: the prompt rendered through the *same* strategy._build_payload
    decide() uses (byte-identical given identical inputs, single source of
    truth AGENTS.md #10), plus an input summary (signal counts, watchlist
    prices resolved/missing — surfaces the yfinance starvation behind the
    timeout storms), the advisory-block presence, embedded mark-integrity,
    and a BLIND/DEGRADED/OK feed_state. **_claude_call is never invoked.**
    Read-only snapshot (cannot mutate the live trader). Advisory only,
    NOT injected into the decision prompt — dashboard/chat/CLI only (the
    desk_pulse precedent; AGENTS.md #2/#12). SWR-cached (the assemble fetch
    is multi-second). Also `python -m paper_trader.analytics.decision_context
    [--full|--json]`. Pure core: analytics/decision_context.py."""
    try:
        from .analytics.decision_context import (
            assemble_inputs,
            build_decision_context,
        )
        store = get_store()
        return jsonify(build_decision_context(**assemble_inputs(store)))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/model-reliability")
def model_reliability_api():
    """Which model actually made each live decision — Opus vs the degraded
    Sonnet fallback — and how often the cycle produced nothing at all.

    The live trader is tuned end-to-end around Opus's reasoning depth
    (AGENTS.md invariant #3). decision-health buckets by *outcome* and
    decision-forensics only dissects the *NO_DECISION* excerpts; neither
    can tell the operator a reasoning-depth-tuned book is quietly being run
    by the fallback. This reports the Opus/fallback split over 24h/7d/all,
    the *share of executed trades* placed by the fallback, an
    improving/worsening trend, and a verdict. Legacy rows predating the
    ``fallback_used`` flag are excluded from the ratio (verified live: a
    large pre-instrumentation tail). Observational only — never gates Opus
    (AGENTS.md #2/#12). Also ``python -m
    paper_trader.analytics.model_reliability [--json]``."""
    try:
        from .analytics.model_reliability import build_model_reliability
        decisions = get_store().recent_decisions(limit=3000)
        return jsonify(build_model_reliability(decisions))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _swr_prewarm():
    """Pre-build every slow SWR cache once at boot.

    On a service restart every ``@swr_cached`` cache is empty, so the first
    poll of each endpoint pays the full cold path: it blocks
    ``_SWR_COLD_BUDGET_S``, returns ``{"warming": true}``, and the real data
    only lands on the *next* auto-refresh poll — a multi-second dead page
    after every restart. This enqueues a single-flight background rebuild for
    each endpoint's default (no-query-string) variant the instant the server
    boots, so the caches are populated before the first user poll.

    Enqueues via ``_swr_refresh`` (not the decorated wrapper) so it returns
    immediately: every job is handed to ``_SWR_EXEC``'s workers, which grind
    through them concurrently. ``name + "?"`` is exactly the cache key the
    wrapper builds for an empty query string, so the warmed entry is the one
    a real request hits. ``wrapper.__wrapped__`` is the undecorated handler
    (set by functools.wraps) — ``_swr_refresh`` expects the raw handler, not
    the SWR wrapper. Staggered to avoid a startup thundering-herd on the
    USB-backed store / yfinance. Daemon thread — never blocks ``app.run()``."""
    _time.sleep(2.0)  # let the WSGI server bind and the store settle first
    targets = [
        ("state", state),
        ("data-feed", data_feed_api),
        ("backtests-list", backtests_api),
        ("backtests-leaderboard", backtest_leaderboard_api),
        ("backtests-stats", backtest_stats_api),
        ("briefing", briefing_api),
        ("suggestions", suggestions_api),
        ("scorer-predictions", scorer_predictions_api),
        ("sector-heatmap", sector_heatmap_api),
        ("game_plan", game_plan_api),
        ("correlation", correlation_api),
        ("thesis-drift", thesis_drift_api),
        ("news-edge", news_edge_api),
        ("source-edge", source_edge_api),
        ("feed-health", feed_health_api),
        ("decision-context", decision_context_api),
        # Freeze-triage panels (risk / capital-paralysis / decision-health /
        # runner-heartbeat) + benchmark + scorer-confidence were @swr_cached
        # but never added here, so they alone cold-stalled with
        # {"warming": true} after every restart — exactly the panels a trader
        # opens FIRST when the bot looks frozen. _swr_prewarm's contract is
        # "pre-build EVERY slow SWR cache"; keep this list == the set of
        # @swr_cached endpoints (tests/test_swr_prewarm_coverage.py locks it).
        ("risk", risk_api),
        ("benchmark", benchmark_api),
        ("capital-paralysis", capital_paralysis_api),
        ("decision-health", decision_health_api),
        ("decision-pace", decision_pace_api),
        ("runner-heartbeat", runner_heartbeat_api),
        ("scorer-confidence", scorer_confidence_api),
        # scorer-attribution was @swr_cached but never prewarmed — the same
        # freeze-triage cold-stall blind spot (a trader opening the scorer
        # attribution panel right after a restart got {"warming": true});
        # test_swr_prewarm_coverage.py locks the prewarm==@swr_cached set.
        ("scorer-attribution", scorer_attribution_api),
        ("baseline-compare", baseline_compare_api),
        # stress-scenarios + watchlist-opportunities were @swr_cached but
        # never prewarmed (commits 737a2d2 / 6e9c5d8) — same freeze-triage
        # cold-stall blind spot test_swr_prewarm_coverage.py locks against.
        ("stress_scenarios", stress_scenarios_api),
        # position_blowup @swr_cached 30s — the per-name single-name shock
        # ladder, a freeze-triage risk panel sibling of stress_scenarios.
        # The prewarm==@swr_cached invariant locks it here so the first poll
        # after a restart never cold-stalls on {"warming": true}.
        ("position_blowup", position_blowup_api),
        # trim_simulator @swr_cached 45s — per-position trim ladder with
        # scorer-EV math (avoided_loss / forgone_upside per rung). Reuses the
        # scorer_predictions in-process call, so a cold cache after restart
        # can compound two warming stalls in a row — prewarm both. Same
        # freeze-triage cold-stall blind spot the prewarm==@swr_cached
        # invariant locks against.
        ("trim_simulator", trim_simulator_api),
        # concentration_cap @swr_cached 30s — mechanical per-name cap
        # rebalance recommender, sibling of position_blowup / risk on the
        # concentration-triage row. The prewarm==@swr_cached invariant locks
        # it here so first poll after a restart never cold-stalls on
        # {"warming": true}.
        ("concentration_cap", concentration_cap_api),
        # etf-lookthrough @swr_cached 30s — pure compute, no I/O, but the
        # prewarm==@swr_cached invariant locks every cached endpoint here so
        # the first poll after a restart never cold-stalls on {"warming":
        # true}; same freeze-triage discipline as the sibling additive keys.
        ("etf-lookthrough", etf_lookthrough_api),
        ("watchlist-opportunities", watchlist_opportunities_api),
        # scorer-opportunities (20295e8), scorer-portfolio-attribution
        # (6018347), and trade-attribution (2a28eea) were @swr_cached but
        # never prewarmed — same freeze-triage cold-stall blind spot. The
        # scorer panels are the operator's primary attribution surface when
        # the book is bleeding; a {"warming": true} on first poll after a
        # restart is exactly the wrong UX during triage.
        ("scorer-opportunities", scorer_opportunities_api),
        ("scorer-portfolio-attribution", scorer_portfolio_attribution_api),
        ("trade-attribution", trade_attribution_api),
        # earnings-shock @swr_cached 300s — yfinance earnings_dates + 3y daily
        # history is the slowest per-name yfinance shape we touch (~1-3s per
        # held name). Without a prewarm the first user poll cold-stalls into
        # {"warming": true} exactly when the operator is checking pre-print
        # risk; the prewarm==@swr_cached invariant (test_swr_prewarm_coverage)
        # requires it here too.
        ("earnings-shock", earnings_shock_api),
        # earnings-distribution @swr_cached 300s — observed-quantile complement
        # to earnings-shock, same yfinance earnings_dates + multi-year daily
        # history shape (the slowest per-name yfinance call we make). Was
        # @swr_cached but never prewarmed — same freeze-triage cold-stall
        # blind spot the test_swr_prewarm_coverage invariant locks against.
        ("earnings-distribution", earnings_distribution_api),
        # implied-move @swr_cached 300s — yfinance options chain is a per-
        # ticker network call (1-2s per held name). Same cold-stall blind
        # spot the prewarm-coverage invariant locks against; matches the
        # earnings-shock prewarm cadence.
        ("implied-move", implied_move_api),
        # peer-earnings-shock @swr_cached 300s — fuses etf_lookthrough +
        # event_calendar to surface indirect 1σ exposure on held leveraged
        # ETFs from constituent prints. Same slow-yfinance shape as
        # earnings-shock (3y daily history per constituent → _pop_stdev);
        # the prewarm==@swr_cached invariant locks it here so first poll
        # post-restart returns real data, not {"warming": true} — exactly
        # the cold-stall blind spot during pre-mega-cap-print triage.
        ("peer-earnings-shock", peer_earnings_shock_api),
        # The four endpoints below were @swr_cached but never added to this
        # prewarm list — exactly the freeze-triage cold-stall blind spot the
        # test_swr_prewarm_coverage invariant exists to catch. A trader who
        # opens these panels right after a restart got {"warming": true}
        # instead of real data for one full TTL cycle. Restored to keep the
        # prewarm == @swr_cached contract intact.
        ("suggestion_impact", suggestion_impact_api),
        ("earnings-war-room", earnings_war_room_api),
        ("restart-recommendation", restart_recommendation_api),
        ("position-action-brief", position_action_brief_api),
        # event-protection @swr_cached 300s — composes /api/implied-move's
        # options-chain pull (1-2s per held earnings name) and sizes a trim +
        # ATM-put-hedge plan against a configurable 1σ-book cap. Same cold-
        # stall blind spot the prewarm-coverage invariant locks against; first
        # poll right after a restart is exactly when the operator is triaging
        # pre-print risk.
        ("event-protection", event_protection_api),
        # news-themes @swr_cached 60s — articles.db SELECT + ticker
        # regex over ~2000 rows is the slowest pure-DB shape we touch
        # outside of /api/state. First poll right after a restart is
        # exactly when the operator is glancing at the wire — locked
        # by the prewarm==@swr_cached invariant.
        ("news-themes", news_themes_api),
        # held-theme-decay @swr_cached 60s — same articles.db SELECT shape
        # as news-themes (~2000 rows + ticker regex) over a wider window
        # (fresh+prior bands). First poll right after a restart is exactly
        # when the operator is checking "did any held thesis go dark while
        # I was away" — locked by the prewarm==@swr_cached invariant.
        ("held-theme-decay", held_theme_decay_api),
        # rising-unheld-themes @swr_cached 60s — same articles.db SELECT
        # shape as held-theme-decay (~2000 rows + ticker regex over a
        # 13h+ window for a 6h fresh+prior band). First poll right
        # after a restart is exactly when the operator scans for
        # rotation candidates ("what catalyst am I missing while my
        # held names are FADING?") — locked by the prewarm==@swr_cached
        # invariant.
        ("rising-unheld-themes", rising_unheld_themes_api),
        # sector-velocity-delta @swr_cached 60s — same articles.db
        # SELECT shape as held-theme-decay / rising-unheld-themes
        # (~2000 rows + ticker regex over a 13h+ window for a 6h
        # fresh+prior band). First poll right after a restart is
        # exactly when the operator scans for sector rotation
        # ("is the wire moving from semis into design or the other
        # way?") — locked by the prewarm==@swr_cached invariant.
        ("sector-velocity-delta", sector_velocity_delta_api),
        # round-trip-postmortem @swr_cached 60s — store.recent_trades
        # + market.get_prices (yfinance per closed-ticker). yfinance
        # is the long-tail latency contributor; same cold-stall blind
        # spot the prewarm-coverage invariant locks against.
        ("round-trip-postmortem", round_trip_postmortem_api),
        # The four endpoints below were @swr_cached but never added to the
        # prewarm list — same freeze-triage cold-stall blind spot the
        # test_swr_prewarm_coverage invariant locks against. A trader who
        # opens these panels right after a restart got {"warming": true}
        # instead of real data for one full TTL cycle. Restored to keep the
        # prewarm == @swr_cached contract intact.
        #
        # decision-paralysis: consecutive-HOLD streak detector
        # (HOLD_LOCK pathology). Right when the operator is checking
        # "is the bot actually deciding or wedged on HOLD?", a cold-
        # stall {"warming": true} hides the answer.
        ("decision-paralysis", decision_paralysis_api),
        # position-news-cooldown: per-held-name news-flow gone-quiet
        # detector — articles.db SELECT over a multi-day window per
        # held ticker, the slowest pure-DB shape outside /api/state.
        ("position-news-cooldown", position_news_cooldown_api),
        # correlation-cluster-warning: hidden-factor-bet alarm built
        # on /api/correlation's pairwise output. Together with
        # /api/correlation (already prewarmed) this is the desk's
        # concentration-risk surface; cold-stalling it alone leaves the
        # operator with only half the picture during freeze triage.
        ("correlation-cluster-warning", correlation_cluster_warning_api),
        # launcher-restart-loop: systemd/launcher crash-loop detector.
        # First poll right after a restart is exactly when the operator
        # is checking "did the supervisor flap?"; a {"warming": true}
        # there is the worst possible UX.
        ("launcher-restart-loop", launcher_restart_loop_api),
        # buying-power: deployable-cash awareness — the lean "what can my
        # free cash fund right now?" surface, ALREADY in the Opus prompt
        # block. yfinance bulk price call is the single slow path; first
        # poll after a restart is exactly when the operator is sizing a
        # manual trade — cold-stalling that surface defeats the point of
        # surfacing the block in the first place.
        ("buying-power", buying_power_api),
        # The four endpoints below were @swr_cached by later commits but
        # never added to this prewarm list — the same freeze-triage cold-
        # stall blind spot test_swr_prewarm_coverage locks against. A trader
        # opening these panels right after a restart got {"warming": true}
        # for one full TTL cycle. Restored to keep prewarm == @swr_cached.
        # watchlist-coverage / today-action-tape are articles.db + store
        # scans; concentration-trajectory / realized-vs-unrealized are
        # equity-curve + round-trip rebuilds — all slow enough to cold-stall.
        ("watchlist-coverage", watchlist_coverage_api),
        ("realized-vs-unrealized", realized_vs_unrealized_api),
        ("concentration-trajectory", concentration_trajectory_api),
        ("today-action-tape", today_action_tape_api),
    ]
    for name, wrapper in targets:
        try:
            raw = getattr(wrapper, "__wrapped__", wrapper)
            _swr_refresh(name + "?", raw, (), {}, "")
        except Exception as e:
            print(f"[dashboard] SWR prewarm enqueue {name} failed: {e}")
        _time.sleep(0.5)


def run(host: str = "0.0.0.0", port: int = 8090):
    # threaded=True: the dashboard's real load is concurrent (the unified
    # :8888 page fires ~25 panel fetches in parallel, /api/chat fans out ~15
    # sub-fetches, the :8080 dashboard cross-fetches us mid-runner-cycle).
    # A single-threaded server head-of-line-blocks every fast pure-DB panel
    # behind one slow yfinance-backed endpoint. Safe: store.py serializes
    # every read on Store._lock and is explicitly hardened for "the Flask
    # dashboard thread(s)"; slow endpoints use their own mode=ro connections.
    # Pre-warm the SWR caches in the background so the first poll after a
    # restart serves real data instead of a cold-stall + {"warming": true}.
    # Guarded by _swr_active(): inert under pytest (no 16 background builds
    # during the test suite). Daemon thread — does not block app.run().
    if _swr_active():
        _threading.Thread(target=_swr_prewarm, name="dash-swr-prewarm",
                           daemon=True).start()
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)


# ──────────────────────────────────────────────────────────────────────────
# /api/supervision — "if this trader exits, will anything bring it back?"
#
# Placed just before the EOF __main__ guard deliberately: this repo's
# operating model is many concurrent agents committing the shared tree
# (CLAUDE.md §1 / scripts/hourly_review.sh), so co-resident uncommitted edits
# to dashboard.py are the norm and mid-file line numbers shift under us. This
# is the lowest-collision insertion point; the @app.route decorator still
# registers at import (runner imports the module, then calls run() — the
# guard below stays False on import).
#
# Live evidence motivating this (2026-05-18 ~02:11 PDT): the trader was
# running as an orphaned `python3 runner.py` (PPID 1) with the systemd unit
# `disabled`/`inactive` AND /api/build-info reporting behind:1 stale:true —
# i.e. running old code with NO restart safety net. /api/build-info shows the
# stale SHA and /api/runner-heartbeat shows loop liveness + singleton-lock,
# but NOTHING answered "is the auto-restart safety net (systemd
# Restart=always) actually in force, or is this an unsupervised orphan that
# stays DOWN the instant its git-watcher / deadman does os._exit(0)?". This
# closes that operator blind spot. Advisory / read-only only: observes
# process + unit state, never gates Opus, never restarts anything, never
# adds a position cap (AGENTS.md #2/#12 — same contract as runner-heartbeat).
@app.route("/api/supervision")
def supervision_api():
    """Deployment-recovery health: is this trader supervised?

    Returns {pid, ppid, orphan, systemd:{active,enabled}, boot_sha,
    head_sha, behind, stale, supervised, verdict, recommendation}.

    verdict (advisory only — never gates or restarts anything):
      HEALTHY            — supervised and running current code
      STALE              — supervised but on old code (restart to deploy;
                            safety net present so it will recover)
      UNSUPERVISED       — no restart safety net: a clean exit (git-watcher
                            restart / deadman / crash) leaves the trader DOWN
      UNSUPERVISED_STALE — worst: unsupervised AND already on old code
      UNKNOWN            — could not determine (degrade-safe; never raises)

    Primary signal is PPID==1 (deterministic 'this process is orphaned and
    nothing will revive it', independent of dbus/XDG state). systemctl
    --user is supplementary context only; an unreadable user bus degrades
    to 'unknown' WITHOUT flipping a confident orphan verdict."""
    try:
        pid = _os.getpid()
        try:
            ppid = _os.getppid()
        except Exception:
            ppid = None

        def _systemctl(verb: str) -> str:
            try:
                r = subprocess.run(
                    ["systemctl", "--user", verb, "paper-trader"],
                    capture_output=True, text=True, timeout=3,
                )
                return ((r.stdout or "").strip()
                        or (r.stderr or "").strip() or "unknown")
            except Exception:
                return "unknown"

        unit_active = _systemctl("is-active")    # active|inactive|failed|unknown
        unit_enabled = _systemctl("is-enabled")  # enabled|disabled|static|unknown

        head, behind = _head_sha_and_behind()

        # Single source of truth for the verdict/recommendation strings and
        # the orphan/stale/supervised derivation (invariant #10): the pure
        # builder is also composed verbatim by the hourly/daily Discord
        # `_supervision_line`, so the two operator surfaces can never drift.
        # The impure probes (pid/ppid/systemctl/git) stay here — the
        # established "network in the caller, builder is pure" split.
        from .analytics.supervision import build_supervision
        return jsonify(build_supervision(
            pid=pid, ppid=ppid,
            unit_active=unit_active, unit_enabled=unit_enabled,
            boot_sha=_BOOT_SHA, head_sha=head, behind=behind,
        ))
    except Exception as e:
        return jsonify({"error": str(e), "verdict": "UNKNOWN"}), 500


# ──────────────────────────────────────────────────────────────────────────
# /api/equity-integrity — "can I trust the recorded P&L history?"
#
# Time-series sibling of /api/mark-integrity (which is point-in-time: "is my
# book stale RIGHT NOW"). Audits the recorded equity_curve for negative-cash
# points (the no-hard-cap book over-drawing — AGENTS.md #12), non-positive
# equity, and no-trade jumps that signal a mismark / stale-price unfreeze /
# option-settlement artifact rather than a real move. Everything downstream
# (/api/drawdown, /api/benchmark, /api/analytics Sharpe, the hourly P/L line)
# is derived from equity_curve, so a silent corruption there poisons every
# P&L surface with nothing saying so. Pure store reads only — NO network, so
# fast enough to need no SWR wrap. Advisory / read-only: never gates Opus,
# adds no caps (AGENTS.md #2/#12 — same contract as mark-integrity). Pure
# core: analytics/equity_integrity.py. Placed at EOF for the documented
# lowest-collision insertion point (concurrent-agent operating model).
@app.route("/api/equity-integrity")
def equity_integrity_api():
    try:
        from .analytics.equity_integrity import build_equity_integrity
        store = get_store()
        out = build_equity_integrity(
            store.equity_curve(limit=5000),
            store.recent_trades(5000),
        )
        out["as_of"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e), "verdict": "ERROR"}), 500


# ──────────────────────────────────────────────────────────────────────────
# /api/equity-freshness — "is the equity point my headline KPIs are computed
# from still current, or frozen behind a fresher book under load?"
#
# The orthogonal sibling of /api/equity-integrity (within-curve corruption)
# and /api/mark-integrity (point-in-time position staleness): this compares
# the live `portfolio` table total against the LATEST recorded `equity_curve`
# point. Under a NO_DECISION storm the portfolio table re-marks every cycle
# while the curve lags one whole cycle behind, so /api/benchmark,
# /api/drawdown, /api/analytics Sharpe and the hourly P/L line (all derived
# from equity_curve) silently misstate the true account by the divergence.
# equity_integrity reads CLEAN here (the gap is portfolio-vs-curve, not
# within recorded points) so it does NOT cover this dimension. The endpoint
# owns the store reads + the market-open probe; the builder is pure (the
# runner_heartbeat "network in the caller, builder is pure" split). Pure
# store reads only — NO network, so fast enough to need no SWR wrap.
# Advisory / read-only: never gates Opus, adds no caps (AGENTS.md #2/#12 —
# same contract as equity-integrity / mark-integrity). Pure core:
# analytics/equity_freshness.py. Placed at EOF for the documented
# lowest-collision insertion point (concurrent-agent operating model).
@app.route("/api/equity-freshness")
def equity_freshness_api():
    try:
        from .analytics.equity_freshness import build_equity_freshness
        from . import market as _mkt
        store = get_store()
        now_utc = datetime.now(timezone.utc)
        out = build_equity_freshness(
            store.get_portfolio(),
            store.equity_curve(limit=5000),
            _mkt.is_market_open(now_utc),
            now=now_utc,
        )
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e), "verdict": "ERROR"}), 500


# ───────── News velocity (2026-05-18, agent 4) ─────────
# Appended at the tail of the route section (not interleaved) so a concurrent
# review pass editing existing endpoints doesn't collide on merge — mirrors
# the `/api/decision-reliability` / `/api/funded-suggestions` placement.


def _stock_tickers_from_positions(positions: list[dict]) -> list[str]:
    """Held-book stock side only. Option lots reuse the underlying ticker,
    so a stock-or-option NVDA position bucketed once. Pseudo-tickers
    (CASH/NONE/empty — the ``_parse_action_ticker`` carve-out, AGENTS.md
    invariant #11) excluded."""
    seen: set[str] = set()
    out: list[str] = []
    for p in positions or []:
        tk = (p.get("ticker") or "").upper().strip()
        if not tk or tk in {"CASH", "NONE", "NO_DECISION", "BLOCKED"}:
            continue
        if tk in seen:
            continue
        seen.add(tk)
        out.append(tk)
    return out


@app.route("/api/news-velocity")
def news_velocity_api():
    """Per-held-ticker news-flow velocity — is the catalyst BUILDING or
    FADING? Compares the article rate over the last ``window_hours``
    (default 24h) to a non-overlapping ``baseline_hours`` (default 168h)
    baseline and emits a Poisson-style z-score + state ladder
    (SURGING / STABLE / FADING / INSUFFICIENT / NO_DATA).

    Reads the digital-intern articles.db through the SAME freshness-aware
    ``_articles_db_path()`` the other news-analytics endpoints use
    (``signals._db_path()``, invariant #17 — no split-brain feed). Live-only
    clause applied (AGENTS.md invariant #3); the pure ``build_news_velocity``
    composes the verdict on already-decompressed rows so it is unit-testable
    without a DB.

    Query params (clamped):
      * ``window_hours`` — 1..72, default 24
      * ``baseline_hours`` — must exceed window, capped at 720 (30d),
        default 168 (7d)
      * ``tickers`` — comma-separated override; default = open held stock
        tickers from ``store.open_positions()``

    Observational only — never gates Opus, no caps (AGENTS.md #2/#12).
    """
    try:
        from .analytics.news_velocity import build_news_velocity

        try:
            window_h = float(request.args.get("window_hours", 24.0))
        except Exception:
            window_h = 24.0
        window_h = max(1.0, min(window_h, 72.0))

        try:
            baseline_h = float(request.args.get("baseline_hours", 168.0))
        except Exception:
            baseline_h = 168.0
        baseline_h = max(window_h + 1.0, min(baseline_h, 720.0))

        override = (request.args.get("tickers") or "").strip()
        if override:
            held = [t.strip().upper() for t in override.split(",") if t.strip()]
        else:
            store = get_store()
            held = _stock_tickers_from_positions(store.open_positions())

        now_utc = datetime.now(timezone.utc)
        if not held:
            return jsonify(build_news_velocity([], [], now=now_utc,
                                               window_hours=window_h,
                                               baseline_hours=baseline_h))

        path = _articles_db_path()
        if path is None:
            return jsonify({
                "as_of": now_utc.isoformat(timespec="seconds"),
                "state": "NO_DATA",
                "headline": "News velocity: articles.db not found.",
                "window_hours": window_h,
                "baseline_hours": baseline_h,
                "n_held": len(held),
                "n_with_data": 0,
                "per_ticker": [],
            })

        # Split the fetch into a window + baseline pair. The live articles.db
        # ingests ~thousands of rows/day (observed 2026-05-18: 1.47M live rows
        # in last 7d) so a single ORDER BY first_seen DESC LIMIT N pull
        # returns ONLY window-era rows on a high-throughput day — collapsing
        # every baseline count to 0 and forcing INSUFFICIENT everywhere.
        #
        # Baseline pre-filter is pushed to SQL via a per-ticker LIKE union:
        # most articles don't mention any held ticker by symbol, so the
        # SQL-side filter drops the scan from ~hundreds-of-thousands of rows
        # to a few thousand before Python regex refinement (a naive
        # ORDER-by-first_seen Python-side scan reads ~40s on a 60k LIMIT;
        # the LIKE-prefilter version is well under 1s). SQLite LIKE is
        # case-insensitive for ASCII by default — exactly what we need to
        # catch ``NVDA``/``nvda``/``Nvda`` in titles. Word-boundary refinement
        # (so AMD does NOT alias AMDOCS) happens inside ``build_news_velocity``
        # via the compiled regex.
        window_since = (now_utc - timedelta(hours=window_h)).isoformat()
        baseline_since = (now_utc - timedelta(hours=baseline_h)).isoformat()
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            window_rows = conn.execute(
                "SELECT title, full_text, ai_score, urgency, first_seen "
                "FROM articles WHERE first_seen >= ? "
                "AND url NOT LIKE 'backtest://%' "
                "AND source NOT LIKE 'backtest_%' "
                "AND source NOT LIKE 'opus_annotation%' "
                "ORDER BY first_seen DESC LIMIT 8000",
                (window_since,),
            ).fetchall()
            like_clauses = " OR ".join(["title LIKE ?"] * len(held))
            like_params = [f"%{t}%" for t in held]
            baseline_rows = conn.execute(
                f"SELECT title, ai_score, urgency, first_seen "
                f"FROM articles WHERE first_seen >= ? AND first_seen < ? "
                f"AND ({like_clauses}) "
                f"AND url NOT LIKE 'backtest://%' "
                f"AND source NOT LIKE 'backtest_%' "
                f"AND source NOT LIKE 'opus_annotation%' "
                f"ORDER BY first_seen DESC LIMIT 20000",
                [baseline_since, window_since] + like_params,
            ).fetchall()
        finally:
            conn.close()

        articles: list[dict] = []
        for r in window_rows:
            body = ""
            if r["full_text"]:
                try:
                    body = zlib.decompress(r["full_text"]).decode(
                        "utf-8", errors="replace")
                except Exception:
                    body = ""
            articles.append({
                "title": r["title"] or "",
                "body": body,
                "first_seen": r["first_seen"],
                "ai_score": r["ai_score"],
                "urgency": r["urgency"],
            })
        for r in baseline_rows:
            articles.append({
                "title": r["title"] or "",
                "body": "",
                "first_seen": r["first_seen"],
                "ai_score": r["ai_score"],
                "urgency": r["urgency"],
            })

        result = build_news_velocity(
            articles, held, now=now_utc,
            window_hours=window_h, baseline_hours=baseline_h,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "state": "ERROR",
                        "per_ticker": []}), 500


@app.route("/api/news-source-mix")
def news_source_mix_api():
    """Per-held-ticker news source-diversity verdict (STRONG/MODERATE/
    ECHO/QUIET).

    Complementary to ``/api/news-velocity``. Velocity reports *rate*
    (BUILDING/FADING); a SURGING z-score of +4 looks identical whether
    five distinct outlets are reporting genuine news or one wire is
    being mirrored across five feeds. ``ECHO`` is the false-signal
    case velocity cannot see. Combined: SURGING + STRONG = real
    catalyst; SURGING + ECHO = syndication artifact.

    Reads the digital-intern articles.db through the freshness-aware
    ``_articles_db_path()`` (invariant #17). Live-only clause applied
    (AGENTS.md invariant #3). The pure ``build_news_source_mix`` composes
    the verdict on already-fetched rows so it stays unit-testable
    without a DB.

    Query params (clamped):
      * ``window_hours`` — 1..72, default 24
      * ``tickers`` — comma-separated override; default = open held stock
        tickers from ``store.open_positions()``

    Observational only — never gates Opus, no caps (AGENTS.md #2/#12).
    """
    try:
        from .analytics.news_source_mix import build_news_source_mix

        try:
            window_h = float(request.args.get("window_hours", 24.0))
        except Exception:
            window_h = 24.0
        window_h = max(1.0, min(window_h, 72.0))

        override = (request.args.get("tickers") or "").strip()
        if override:
            held = [t.strip().upper() for t in override.split(",") if t.strip()]
        else:
            store = get_store()
            held = _stock_tickers_from_positions(store.open_positions())

        now_utc = datetime.now(timezone.utc)
        if not held:
            return jsonify(build_news_source_mix([], [], now=now_utc,
                                                 window_hours=window_h))

        path = _articles_db_path()
        if path is None:
            return jsonify({
                "as_of": now_utc.isoformat(timespec="seconds"),
                "state": "NO_DATA",
                "headline": "Source mix: articles.db not found.",
                "window_hours": window_h,
                "n_held": len(held),
                "n_with_data": 0,
                "any_echo": False,
                "per_ticker": [],
            })

        # Pre-filter SQL by per-ticker LIKE union — same cost discipline as
        # /api/news-velocity. Window is bounded (default 24h) so the row
        # cap of 5000 is more than enough on the observed 1.47M-rows/7d
        # articles.db. Word-boundary refinement happens inside the
        # builder.
        since = (now_utc - timedelta(hours=window_h)).isoformat()
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            like_clauses = " OR ".join(["title LIKE ?"] * len(held))
            like_params = [f"%{t}%" for t in held]
            rows = conn.execute(
                f"SELECT title, source, first_seen "
                f"FROM articles WHERE first_seen >= ? "
                f"AND ({like_clauses}) "
                f"AND url NOT LIKE 'backtest://%' "
                f"AND source NOT LIKE 'backtest_%' "
                f"AND source NOT LIKE 'opus_annotation%' "
                f"ORDER BY first_seen DESC LIMIT 5000",
                [since] + like_params,
            ).fetchall()
        finally:
            conn.close()

        articles = [
            {
                "title": r["title"] or "",
                "source": r["source"] or "",
                "first_seen": r["first_seen"],
            }
            for r in rows
        ]

        result = build_news_source_mix(
            articles, held, now=now_utc, window_hours=window_h,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "state": "ERROR",
                        "per_ticker": []}), 500


@app.route("/api/decision-clock")
def decision_clock_api():
    """Per-hour-of-day decision distribution in NY market time.

    /api/empty-claude-rate gives an aggregate empty-response rate;
    /api/host-guard a point-in-time concurrent-Opus snapshot. Neither
    answers which hour of the trading day is most starved. This
    endpoint buckets the last ``days`` of decisions (1..30, default 7)
    by local hour-of-day and splits each bucket's NO_DECISION count
    into ``host_saturated`` (skipped-claude-call / saturated prefixes),
    ``empty_response`` (timeout/empty), and ``parse_failed`` so an
    operator can spot a recurring saturation window (e.g. concurrent
    review agents always firing 10:00 ET) instead of treating the
    storm as flat noise."""
    try:
        try:
            days = int(request.args.get("days", 7))
        except Exception:
            days = 7
        days = max(1, min(days, 30))

        now_utc = datetime.now(timezone.utc)
        cutoff = now_utc - timedelta(days=days)
        decisions = get_store().recent_decisions(limit=20000)

        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo("America/New_York")
        except Exception:
            tz = timezone.utc

        buckets = [
            {
                "hour": h, "total": 0, "filled": 0, "no_decision": 0,
                "host_saturated": 0, "empty_response": 0,
                "parse_failed": 0, "quota_exhausted": 0,
                "other_no_decision": 0,
            }
            for h in range(24)
        ]

        total = 0
        for d in decisions:
            try:
                ts = datetime.fromisoformat(str(d.get("timestamp")))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except Exception:
                continue
            if ts < cutoff:
                continue
            try:
                hour = ts.astimezone(tz).hour
            except Exception:
                hour = ts.hour
            b = buckets[hour]
            b["total"] += 1
            total += 1
            action = (d.get("action_taken") or "")
            reasoning = (d.get("reasoning") or "")
            if "FILLED" in action:
                b["filled"] += 1
            elif action == "NO_DECISION":
                b["no_decision"] += 1
                # Bucket precedence is load-bearing — branches must be
                # mutually exclusive and ordered most-specific first.
                # quota-exhaustion (the strategy.decide() "claude quota/usage
                # limit exhausted (no decision)" reason) was previously
                # uncategorised and silently absorbed by other_no_decision;
                # the operator triaging a NO_DECISION storm couldn't tell a
                # host-saturation freeze (kill review agents) apart from a
                # quota freeze (wait / upgrade plan) — distinct operator
                # actions. Check the quota marker BEFORE the empty/parse
                # branches: strategy.py's quota reason starts with "claude
                # quota" and does not include the "no response" / "host
                # saturated" / "skipped claude call" tokens, but a
                # future-format change must not silently re-merge them.
                if "quota" in reasoning:
                    b["quota_exhausted"] += 1
                elif "host saturated" in reasoning or "skipped claude call" in reasoning:
                    b["host_saturated"] += 1
                elif "no response" in reasoning:
                    b["empty_response"] += 1
                elif "parse_failed" in reasoning or "retry_failed" in reasoning:
                    b["parse_failed"] += 1
                else:
                    b["other_no_decision"] += 1

        for b in buckets:
            n = b["total"]
            b["fill_rate_pct"] = round(100.0 * b["filled"] / n, 1) if n else 0.0
            b["no_decision_pct"] = round(100.0 * b["no_decision"] / n, 1) if n else 0.0
            b["host_saturated_pct"] = round(100.0 * b["host_saturated"] / n, 1) if n else 0.0

        worst = None
        for b in buckets:
            if b["total"] < 3:
                continue
            if worst is None or b["no_decision_pct"] > worst["no_decision_pct"]:
                worst = b
        worst_hour = worst["hour"] if worst else None

        if total < 5:
            verdict = "INSUFFICIENT_DATA"
        elif worst and worst["no_decision_pct"] >= 50.0:
            verdict = (f"HOURLY_CONCENTRATION — hour {worst['hour']:02d}:00 ET "
                       f"has {worst['no_decision_pct']:.0f}% NO_DECISION over "
                       f"{worst['total']} samples")
        else:
            verdict = "EVEN_DISTRIBUTION"

        return jsonify({
            "as_of": now_utc.isoformat(timespec="seconds"),
            "days": days,
            "tz": "America/New_York",
            "total_decisions": total,
            "buckets": buckets,
            "worst_hour_local": worst_hour,
            "verdict": verdict,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/decision-weekday")
def decision_weekday_api():
    """Per-day-of-week decision distribution in NY market time.

    Orthogonal to /api/decision-clock (hour-of-day): exposes whether a
    specific weekday (e.g. Friday-after-close quota slump) is consistently
    starved across the last ``days`` (7..90, default 28) of decisions.
    Same NO_DECISION sub-classification (quota / host / empty / parse) so
    the operator sees the dominant cause per weekday."""
    try:
        from .analytics.decision_weekday import build_decision_weekday
        try:
            days = int(request.args.get("days", 28))
        except Exception:
            days = 28
        decisions = get_store().recent_decisions(limit=20000)
        result = build_decision_weekday(
            decisions, now=datetime.now(timezone.utc), days=days,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "verdict": "ERROR",
                        "buckets": []}), 500


@app.route("/api/decision-daily")
def decision_daily_api():
    """Per-calendar-day NO_DECISION timeseries with a trend verdict.

    Orthogonal to /api/decision-clock (hour-of-day) and
    /api/decision-weekday (day-of-week): those flag *which recurring
    slot* is starved; this answers **is it getting better or worse
    day-over-day?**. Split-halves trend on the NO_DECISION rate emits
    TREND_WORSENING / TREND_IMPROVING / STABLE / INSUFFICIENT_DATA.
    Same NO_DECISION sub-classification precedence
    (quota → host → empty → parse → other) as the other decision-*
    surfaces so the bucket definitions cannot drift.

    Query params:
        days: window in calendar days. Clamped 1..60 (default 14)."""
    try:
        from .analytics.decision_daily import build_decision_daily
        try:
            days = int(request.args.get("days", 14))
        except Exception:
            days = 14
        decisions = get_store().recent_decisions(limit=20000)
        result = build_decision_daily(
            decisions, now=datetime.now(timezone.utc), days=days,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "verdict": "ERROR",
                        "buckets": []}), 500


@app.route("/api/quota-burnrate")
def quota_burnrate_api():
    """Rolling-window quota-exhaustion burn-rate.

    Orthogonal to /api/decision-clock (hour-of-day) and
    /api/decision-weekday (day-of-week): exposes whether quota exhaustion
    is the *dominant* NO_DECISION cause **right now** across short
    rolling windows (6h / 24h / 72h by default). Directly addresses the
    documented historical misdiagnosis where a NO_DECISION storm was
    blamed on the JSON parser while the actual cause was Claude
    org-usage-limit exhaustion — a QUOTA_DOMINANT verdict here points
    the operator at the upgrade-plan/concurrency lever instead of the
    parser. Same NO_DECISION sub-classification precedence as the other
    decision-* surfaces so bucket definitions cannot drift.

    Query params:
        windows: comma-separated hour values (e.g. "1,6,24"). Clamped to
            positive ints, capped at 30 days, max 8 windows."""
    try:
        from .analytics.quota_burnrate import (
            build_quota_burnrate, DEFAULT_WINDOWS_HOURS,
        )
        raw = request.args.get("windows")
        windows_hours: tuple[int, ...] = DEFAULT_WINDOWS_HOURS
        if raw:
            parsed: list[int] = []
            for part in raw.split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    v = int(part)
                except ValueError:
                    continue
                if v <= 0:
                    continue
                parsed.append(min(v, 24 * 30))
                if len(parsed) >= 8:
                    break
            if parsed:
                windows_hours = tuple(parsed)
        decisions = get_store().recent_decisions(limit=20000)
        result = build_quota_burnrate(
            decisions, now=datetime.now(timezone.utc),
            windows_hours=windows_hours,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "windows": []}), 500


@app.route("/api/ticker-decision-mix")
def ticker_decision_mix_api():
    """Per-ticker decision-action breakdown over a recent window.

    /api/decision-health emits the *book-wide* action mix (BUY/SELL/HOLD/
    NO_DECISION %). /api/track-record groups *closed round-trips* by ticker.
    Neither answers "of the names Opus actually deliberated on this session,
    which did it buy, sell, hold — and which is it stuck repeatedly holding
    on?". Pure read of recent decisions; reuses _parse_action_ticker (the
    SSOT used by /api/disagreement) so the (verb, ticker) extraction never
    drifts from sibling endpoints. Advisory only — never gates Opus, never
    injected into the decision prompt (AGENTS.md #2/#12).

    Query params:
        hours: lookback window, clamped 1..168 (default 24).
        min_count: min decisions per ticker to include, clamped 1..50 (default 2)."""
    try:
        try:
            hours = max(1, min(168, int(request.args.get("hours", 24))))
        except (TypeError, ValueError):
            hours = 24
        try:
            min_count = max(1, min(50, int(request.args.get("min_count", 2))))
        except (TypeError, ValueError):
            min_count = 2

        decisions = get_store().recent_decisions(limit=3000)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        by_ticker: dict[str, dict[str, int]] = {}
        total_in_window = 0
        no_ticker_count = 0
        for d in decisions:
            try:
                ts = datetime.fromisoformat(str(d.get("timestamp")))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except Exception:
                continue
            if ts < cutoff:
                continue
            total_in_window += 1
            verb, ticker = _parse_action_ticker(d.get("action_taken") or "")
            if not ticker:
                no_ticker_count += 1
                continue
            bucket = by_ticker.setdefault(ticker, {})
            v = (verb or "OTHER").upper()
            bucket[v] = bucket.get(v, 0) + 1

        rows = []
        for tk, counts in by_ticker.items():
            total = sum(counts.values())
            if total < min_count:
                continue
            buys = sum(counts.get(v, 0) for v in _BUY_VERBS)
            sells = sum(counts.get(v, 0) for v in _SELL_VERBS)
            holds = counts.get("HOLD", 0)
            if buys > 0 and buys >= sells and buys >= holds:
                lean = "BUY_BIAS"
            elif sells > 0 and sells >= holds:
                lean = "SELL_BIAS"
            elif holds >= max(buys, sells) and holds >= 3:
                lean = "STUCK_HOLD"
            else:
                lean = "MIXED"
            rows.append({
                "ticker": tk,
                "total": total,
                "buy_count": buys,
                "sell_count": sells,
                "hold_count": holds,
                "lean": lean,
                "verbs": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
            })
        rows.sort(key=lambda r: (-r["total"], r["ticker"]))

        return jsonify({
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "hours": hours,
            "min_count": min_count,
            "total_decisions_in_window": total_in_window,
            "no_ticker_decisions": no_ticker_count,
            "n_tickers": len(rows),
            "tickers": rows,
        })
    except Exception as e:
        return jsonify({"error": str(e), "tickers": []}), 500


def _empty_rate_24h_pct(decisions: list[dict]) -> float | None:
    """24h "claude returned no response" rate over recent_decisions.

    Mirrors the empty-claude-rate endpoint's `_is_empty` semantics. Shared
    here so position-action-brief and restart-recommendation read the same
    24h surface — the two endpoints disagreeing by 1% on the same DB would
    be exactly the kind of operator-confusion bug this composite tries to
    close. Returns None when no rows fall in the window."""
    if not decisions:
        return None
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    n = 0
    empty = 0
    for d in decisions:
        ts_raw = d.get("timestamp")
        try:
            ts = datetime.fromisoformat(str(ts_raw))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if ts < cutoff:
            continue
        n += 1
        action = (d.get("action_taken") or "")
        reasoning = (d.get("reasoning") or "")
        if action == "NO_DECISION" and \
                reasoning.startswith("claude returned no response"):
            empty += 1
    if n == 0:
        return None
    return round(100.0 * empty / n, 1)


def _consecutive_no_decision(decisions: list[dict]) -> int:
    """Newest-first run of NO_DECISION action_taken — the runner_heartbeat
    decision-efficacy input. Stops at the first non-NO_DECISION row."""
    n = 0
    for d in (decisions or []):
        a = (d.get("action_taken") or "").upper()
        if "NO_DECISION" in a and not a.startswith("BLOCKED"):
            n += 1
        else:
            break
    return n


def _fetch_earnings_events_intern() -> list[dict]:
    """Pull the earnings calendar from :8080. Mirrors the read shared by
    /api/earnings-risk and /api/event-readiness so the three surfaces never
    disagree on what's imminent. Best-effort — returns [] on any failure
    (timeouts, intern wedge — the documented digital_intern_8080_hangs class)."""
    import json as _json
    import urllib.request as _urllib
    try:
        with _urllib.urlopen(
                "http://127.0.0.1:8080/api/earnings", timeout=4) as resp:
            snap = _json.loads(resp.read().decode("utf-8"))
        return list(snap.get("events") or [])
    except Exception:
        return []


def _concurrent_opus_processes() -> int | None:
    """Cheap /proc walk to count concurrent `claude` subprocesses on the host.
    Mirrors the host-guard / empty-claude-rate read shape; degrades to None
    rather than raising on permission errors."""
    try:
        import os
        n = 0
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as fh:
                    cl = fh.read().replace(b"\x00", b" ").decode(
                        "utf-8", "ignore")
            except (OSError, IOError):
                continue
            if "claude" in cl and (" --model" in cl or "claude-opus" in cl):
                n += 1
        return n
    except Exception:
        return None


@app.route("/api/restart-recommendation")
@swr_cached("restart-recommendation", 30.0)
def restart_recommendation_api():
    """Single operator-actionable verdict: should I restart paper-trader NOW?

    Closes the false-HEALTHY gap that ``desk_pulse.liveness`` and the bare
    ``runner_heartbeat`` cadence verdict leave open: cadence-HEALTHY can
    coexist with an 81%-empty-rate parse-fail storm, and neither surface
    weights *held-exposure-at-risk-into-event* into the operator's restart
    decision. This composes the parse-fail / host-saturation / idle-storm
    inputs already on the existing surface with held-imminent earnings
    exposure into one **restart_now: bool** + ``urgency_score: 0..1`` + a
    reason bundle suitable for a 30-second cron or Discord nudge.

    The builder is pure; this endpoint owns the I/O (the documented
    thesis_drift split). Advisory only — never gates Opus, adds no caps
    (AGENTS.md invariants #2 / #12)."""
    try:
        from .analytics.restart_recommendation import build_restart_recommendation
        from .analytics.event_readiness import build_event_readiness
        store = get_store()
        decs = store.recent_decisions(limit=2000)
        empty_rate = _empty_rate_24h_pct(decs)
        consec = _consecutive_no_decision(decs)

        opus_n = _concurrent_opus_processes()
        # Mirror host_guard's _CLAUDE_SEM=3 + the live-trader headroom of 1.
        host_saturated = (opus_n > 4) if opus_n is not None else None

        # Held-imminent earnings — reuse build_event_readiness which already
        # cross-joins positions × event_calendar and emits exposure_usd.
        exposure = 0.0
        hours_to: float | None = None
        try:
            events = _fetch_earnings_events_intern()
            positions = store.open_positions()
            er = build_event_readiness(positions, decs, events)
            for ev in (er.get("events") or []):
                ex = float(ev.get("exposure_usd") or 0.0)
                if ex <= 0:
                    continue
                exposure += ex
                ht = ev.get("hours_until_event")
                if ht is None:
                    da = ev.get("days_away")
                    ht = (float(da) * 24.0) if da is not None else None
                if ht is not None:
                    ht_f = float(ht)
                    if hours_to is None or ht_f < hours_to:
                        hours_to = ht_f
        except Exception:
            pass

        result = build_restart_recommendation(
            empty_rate_pct=empty_rate,
            host_saturated=host_saturated,
            held_imminent_exposure_usd=exposure,
            hours_to_nearest_held_event=hours_to,
            consecutive_no_decision=consec,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "verdict": "ERROR"}), 500


@app.route("/api/position-action-brief")
@swr_cached("position-action-brief", 90.0)
def position_action_brief_api():
    """Per-held-position composite brief — the operator's pre-bed view.

    Each held position is annotated with: exposure_usd / pct_portfolio,
    earnings-event proximity (held-imminent only), 24h news-velocity state +
    top story, last real decision attempt + status (DECIDED / EMPTY /
    HOST_SKIP / PARSE_FAIL / NEVER), and a recommended action
    (TRIM_BEFORE_EVENT / HOLD_THROUGH_EVENT / RESTART_RUNNER / MONITOR / OK)
    with an urgency score. Briefs sort most-urgent-first; the overall
    headline surfaces the single most-actionable position.

    Composes the network-free portfolio + decisions read, the
    /api/news-velocity SSOT builder, /api/earnings-risk events, and the
    /api/empty-claude-rate + /api/host-guard scalars. The builder is pure;
    this endpoint owns the I/O. Advisory only — never gates Opus, adds no
    caps (AGENTS.md invariants #2 / #12)."""
    try:
        from .analytics.position_action_brief import build_position_action_brief
        from .analytics.news_velocity import build_news_velocity
        from .analytics.event_readiness import build_event_readiness

        store = get_store()
        positions = store.open_positions()
        decisions = store.recent_decisions(limit=2000)
        held = _stock_tickers_from_positions(positions)
        now_utc = datetime.now(timezone.utc)

        # 24h news velocity over held tickers. Reads articles.db via the
        # same path /api/news-velocity uses (the invariant #17 anti-split-
        # brain hook); degrades to NO_DATA when the DB isn't reachable.
        news_velocity: dict | None = None
        if held:
            try:
                path = _articles_db_path()
                if path is not None:
                    window_h = 24.0
                    baseline_h = 168.0
                    window_since = (now_utc - timedelta(hours=window_h)).isoformat()
                    baseline_since = (now_utc - timedelta(hours=baseline_h)).isoformat()
                    conn = sqlite3.connect(
                        f"file:{path}?mode=ro", uri=True, timeout=5)
                    conn.row_factory = sqlite3.Row
                    try:
                        window_rows = conn.execute(
                            "SELECT title, full_text, ai_score, urgency, first_seen "
                            "FROM articles WHERE first_seen >= ? "
                            "AND url NOT LIKE 'backtest://%' "
                            "AND source NOT LIKE 'backtest_%' "
                            "AND source NOT LIKE 'opus_annotation%' "
                            "ORDER BY first_seen DESC LIMIT 8000",
                            (window_since,),
                        ).fetchall()
                        like_clauses = " OR ".join(
                            ["title LIKE ?"] * len(held))
                        like_params = [f"%{t}%" for t in held]
                        baseline_rows = conn.execute(
                            f"SELECT title, ai_score, urgency, first_seen "
                            f"FROM articles WHERE first_seen >= ? "
                            f"AND first_seen < ? AND ({like_clauses}) "
                            f"AND url NOT LIKE 'backtest://%' "
                            f"AND source NOT LIKE 'backtest_%' "
                            f"AND source NOT LIKE 'opus_annotation%' "
                            f"ORDER BY first_seen DESC LIMIT 20000",
                            [baseline_since, window_since] + like_params,
                        ).fetchall()
                    finally:
                        conn.close()
                    articles: list[dict] = []
                    for r in window_rows:
                        body = ""
                        if r["full_text"]:
                            try:
                                body = zlib.decompress(
                                    r["full_text"]).decode(
                                        "utf-8", errors="replace")
                            except Exception:
                                body = ""
                        articles.append({
                            "title": r["title"] or "",
                            "body": body,
                            "first_seen": r["first_seen"],
                            "ai_score": r["ai_score"],
                            "urgency": r["urgency"],
                        })
                    for r in baseline_rows:
                        articles.append({
                            "title": r["title"] or "",
                            "body": "",
                            "first_seen": r["first_seen"],
                            "ai_score": r["ai_score"],
                            "urgency": r["urgency"],
                        })
                    news_velocity = build_news_velocity(
                        articles, held, now=now_utc,
                        window_hours=window_h, baseline_hours=baseline_h,
                    )
            except Exception:
                news_velocity = None

        # Held-imminent earnings events.
        held_events: list[dict] = []
        try:
            events_raw = _fetch_earnings_events_intern()
            er = build_event_readiness(positions, decisions, events_raw)
            held_events = list(er.get("events") or [])
        except Exception:
            held_events = []

        empty_rate = _empty_rate_24h_pct(decisions)

        opus_n = _concurrent_opus_processes()
        host_saturated = (opus_n > 4) if opus_n is not None else None

        # Starting equity = cash + open value (the same denominator
        # /api/portfolio uses). Live position rows expose ``qty``;
        # options carry the ×100 contract multiplier.
        try:
            portfolio = store.get_portfolio()
            cash = float(portfolio.get("cash") or 0.0)
            open_val = 0.0
            for p in positions:
                qty = float(p.get("qty") or 0)
                px = float(p.get("current_price") or p.get("avg_cost") or 0)
                mult = 100 if p.get("type") in ("call", "put") else 1
                open_val += qty * px * mult
            starting = cash + open_val
        except Exception:
            starting = None

        result = build_position_action_brief(
            positions=positions,
            decisions=decisions,
            news_velocity=news_velocity,
            held_events=held_events,
            empty_rate_pct=empty_rate,
            host_saturated=host_saturated,
            starting_equity_usd=starting,
            now=now_utc,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "briefs": []}), 500


@app.route("/api/reentry-velocity")
def reentry_velocity_api():
    """Close→re-buy interval distribution per ticker.

    ``/api/track-record`` gives Opus a per-name *memory* of how prior
    round-trips on that ticker ended (the verbatim loser/winner-autopsy).
    ``/api/churn`` reports size-weighted intraday turnover. Neither
    surfaces the explicit close-to-re-buy interval, which is the
    documented fast-flip pathology (CLAUDE.md / AGENTS.md observed
    ``avg_holding_days`` ~0.27 with the NVDA→LITE→NVDA shape and
    ``KNIFE_CATCH`` repeats).

    The builder composes ``round_trips.build_round_trips`` (the
    closed-round-trip SSOT — invariant #10) and walks each key's exits
    to the next same-key entry. Open positions whose key has a prior
    closed round-trip surface as ``open_after_close=True`` rows so the
    *live* fast-flip case is visible too (round_trips alone never sees
    a still-open re-entry).

    Observational only — never gates Opus, no caps (AGENTS.md #2 / #12).
    """
    try:
        from .analytics.reentry_velocity import build_reentry_velocity

        store = get_store()
        trades = list(reversed(store.recent_trades(2000)))  # oldest → newest
        open_pos = store.open_positions()
        now_utc = datetime.now(timezone.utc)
        try:
            limit = int(request.args.get("recent_limit", 10))
        except Exception:
            limit = 10
        limit = max(1, min(limit, 100))
        result = build_reentry_velocity(
            trades, open_positions=open_pos, now=now_utc, recent_limit=limit,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "verdict": "ERROR",
                        "recent_gaps": [], "per_ticker": []}), 500


@app.route("/api/blocked-repeats")
def blocked_repeats_api():
    """Repeated-BLOCKED action audit — which (verb, ticker) is Opus trying
    but the engine keeps refusing?

    ``decisions.action_taken`` is free text like ``"BUY NVDA → BLOCKED"``
    (AGENTS.md invariant #11). When Opus keeps trying to act but the engine
    keeps blocking, the trader needs to know **which (verb, ticker)** and
    **the dominant cause** (CASH / DATA / SIZING / SPECIFICATION / OTHER)
    so they pick the right remediation: fund the trade, fix the feed,
    re-prompt Opus, or accept the data blackout.

    Every other operator-facing surface (``/api/decision-health``,
    ``/api/no-decision-reasons``, the Discord ``_no_decision_reasons_line``)
    targets ``NO_DECISION`` (Claude didn't reply). Repeated BLOCKED is the
    orthogonal failure: Claude DID reply, the engine rejected the trade —
    a surface no other endpoint owns.

    Query params (clamped):
      * ``limit`` — how many recent decisions to scan (50..5000, default 500)
      * ``min_repeat`` — minimum count to qualify as a repeat (2..20, default 2)

    Observational only, never gates, no caps (AGENTS.md #2 / #12 — the
    ``no_decision_reasons`` precedent)."""
    try:
        from .analytics.blocked_repeats import build_blocked_repeats

        try:
            limit = int(request.args.get("limit", 500))
        except Exception:
            limit = 500
        limit = max(50, min(limit, 5000))

        try:
            min_repeat = int(request.args.get("min_repeat", 2))
        except Exception:
            min_repeat = 2
        min_repeat = max(2, min(min_repeat, 20))

        store = get_store()
        decisions = store.recent_decisions(limit=limit)
        result = build_blocked_repeats(
            decisions, now=datetime.now(timezone.utc), min_repeat=min_repeat,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "verdict": "ERROR",
                        "blocked_repeats": []}), 500


@app.route("/api/rebuy-regret")
def rebuy_regret_api():
    """Sell-then-rebuy **$ regret** quantifier — did the desk save or lose
    money on close→re-entry hops?

    ``/api/reentry-velocity`` tracks the *time* gap between close and re-buy
    (CHURN_RISK / STABLE on the cadence). ``/api/churn`` measures size-
    weighted turnover. Neither answers the discretionary trader's hardest
    exit question:

      **Did I sell low and buy back higher?**

    Sign convention: ``regret_usd > 0`` means lost money on the
    round-trip-to-re-entry hop (sold low, re-bought higher). ``regret_usd
    < 0`` means saved money. Composes ``round_trips.build_round_trips``
    (SSOT, AGENTS.md #10) for closed round-trips, then walks the trade
    stream for the next same-key BUY to measure the price delta against
    shared quantity. Option ×100 multiplier honored (the
    ``round_trips`` precedent).

    Query params:
      ``recent_limit`` — newest-first event slice (1..100, default 10)

    Observational only — never gates Opus, no caps (AGENTS.md #2/#12).
    """
    try:
        from .analytics.rebuy_regret import build_rebuy_regret

        try:
            limit = int(request.args.get("recent_limit", 10))
        except Exception:
            limit = 10
        limit = max(1, min(100, limit))

        store = get_store()
        # Oldest→newest is what round_trips expects; the builder also
        # sorts defensively but pass clean data.
        trades = list(reversed(store.recent_trades(2000)))
        result = build_rebuy_regret(
            trades, now=datetime.now(timezone.utc), recent_limit=limit,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "verdict": "ERROR",
                        "recent_events": [], "per_ticker": []}), 500


@app.route("/api/news-to-trade-lag")
def news_to_trade_lag_api():
    """News-to-trade lag distribution — is the desk reacting to fresh news?

    ``/api/trade-attribution`` enumerates the highest-scored articles
    preceding each FILLED trade (with a ``minutes_before_trade`` per
    attributed article). This endpoint compresses that detail to one
    distribution + verdict on the desk's *reactivity*: are recent trades
    happening fast on hot news, or consistently 2 hours behind?

    Composes ``build_trade_attribution`` (SSOT, AGENTS.md #10). For each
    attributed trade, takes the minimum ``minutes_before_trade`` across
    its attributed articles (the freshest plausibly-causal signal the
    trade could have reacted to). Trades with zero attributions are
    counted separately, not assigned a fake worst-case lag (the
    ``recovery`` negative-space-is-data precedent).

    Verdict: REACTIVE_FAST (median <30min) / REACTIVE (30..120) /
    DELAYED (>120) / NO_ATTRIBUTION (>50% trades lack live news) /
    NO_DATA.

    Query params (forwarded to ``trade_attribution``):
      ``hours_back`` — trade lookback (1..168, default 24)
      ``window_hours`` — per-trade article window (0.5..24, default 4.0)
      ``max_per_trade`` — top-N articles per trade (1..10, default 3)
      ``min_ai_score`` — article cutoff (0..10, default 2.0)

    Observational only — never gates Opus, no caps (AGENTS.md #2/#12).
    """
    try:
        from .analytics.news_to_trade_lag import build_news_to_trade_lag
        from .analytics.trade_attribution import build_trade_attribution

        def _qf(name, default, lo, hi):
            try:
                v = float(request.args.get(name, default))
            except (TypeError, ValueError):
                v = default
            return max(lo, min(hi, v))

        hours_back = _qf("hours_back", 24.0, 1.0, 168.0)
        window_hours = _qf("window_hours", 4.0, 0.5, 24.0)
        max_per_trade = int(_qf("max_per_trade", 3.0, 1.0, 10.0))
        min_ai_score = _qf("min_ai_score", 2.0, 0.0, 10.0)

        store = get_store()
        with store._lock:  # noqa: SLF001 — same access pattern as trade_attribution
            cur = store.conn.execute(
                "SELECT id, timestamp, ticker, action, qty, price, value, "
                "reason, option_type FROM trades "
                "WHERE timestamp >= datetime('now', ?) "
                "ORDER BY timestamp DESC LIMIT 200",
                (f"-{hours_back:.1f} hours",),
            )
            rows = cur.fetchall()
        trades = [{
            "id": r[0], "timestamp": r[1], "ticker": r[2], "action": r[3],
            "qty": r[4], "price": r[5], "value": r[6], "reason": r[7],
            "type": r[8],
        } for r in rows]

        articles: list[dict] = []
        if trades:
            oldest_iso = trades[-1]["timestamp"]
            try:
                oldest_dt = datetime.fromisoformat(
                    str(oldest_iso).replace("Z", "+00:00"))
            except Exception:
                oldest_dt = datetime.now(timezone.utc) - timedelta(
                    hours=hours_back)
            if oldest_dt.tzinfo is None:
                oldest_dt = oldest_dt.replace(tzinfo=timezone.utc)
            since = (oldest_dt - timedelta(hours=window_hours)).isoformat()
            path = _articles_db_path()
            if path is not None:
                conn = None
                try:
                    conn = sqlite3.connect(
                        f"file:{path}?mode=ro", uri=True, timeout=5)
                    art_rows = conn.execute(
                        "SELECT title, url, source, ai_score, urgency, "
                        "first_seen FROM articles "
                        "WHERE first_seen >= ? AND ai_score >= ? "
                        "AND url NOT LIKE 'backtest://%' "
                        "AND source NOT LIKE 'backtest_%' "
                        "AND source NOT LIKE 'opus_annotation%' "
                        "ORDER BY ai_score DESC LIMIT 5000",
                        (since, min_ai_score),
                    ).fetchall()
                    articles = [{
                        "title": r[0], "url": r[1], "source": r[2],
                        "ai_score": r[3], "urgency": r[4], "first_seen": r[5],
                    } for r in art_rows]
                except Exception:
                    articles = []
                finally:
                    if conn is not None:
                        try:
                            conn.close()
                        except Exception:
                            pass

        attribution = build_trade_attribution(
            trades, articles,
            window_hours=window_hours,
            max_per_trade=max_per_trade,
            min_ai_score=min_ai_score,
        )
        result = build_news_to_trade_lag(
            attribution, now=datetime.now(timezone.utc),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "verdict": "ERROR",
                        "per_trade": []}), 500


@app.route("/api/holding-period-distribution")
def holding_period_distribution_api():
    """Closed round-trip P/L stratified by hold duration.

    Every realised-P/L surface on this desk (``track_record``,
    ``trade_asymmetry``, ``round_trip_postmortem``, ``churn``,
    ``winner_autopsy``, ``loser_autopsy``) reduces the round-trip set to a
    single aggregate or per-trip story — none of them answer
    "*where in the holding-period axis does my P/L live?*". This endpoint
    is that stratification: buckets every closed trip into SCALP (<1h) /
    INTRADAY (1-6h) / OVERNIGHT (6-24h) / SWING (1-3d) / TREND (3-7d) /
    POSITION (>7d) and surfaces per-bucket n_trips + total P/L + win rate
    + share-of-P/L. ``alpha_engine`` (highest total P/L bucket) and
    ``dominant_bucket`` (most-trips bucket) together give the actionable
    signal "63% of trips are SCALP but 91% of P/L comes from SWING".

    Composes ``analytics.round_trips.build_round_trips`` as the SSOT for
    the closed-trip ledger (AGENTS.md invariant #10) and feeds it to
    ``analytics.holding_period_distribution.build_holding_period_distribution``.
    Pure read — no network, no extra DB hops beyond the trades read. Never
    raises (a malformed trade row degrades that row, never the verdict).

    Observational only — never gates Opus, never injected into the
    decision prompt, no caps (AGENTS.md #2/#12)."""
    try:
        from .analytics.round_trips import build_round_trips
        from .analytics.holding_period_distribution import (
            build_holding_period_distribution,
        )
        try:
            limit = int(request.args.get("limit", 2000))
        except Exception:
            limit = 2000
        limit = max(50, min(limit, 10000))

        store = get_store()
        trades = store.recent_trades(limit=limit)
        # build_round_trips expects oldest-first; recent_trades returns
        # newest-first per the existing convention in the round-trips
        # endpoint.
        trades_oldest_first = list(reversed(trades or []))
        round_trips = build_round_trips(trades_oldest_first)
        result = build_holding_period_distribution(round_trips)
        result["as_of"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "state": "ERROR",
                        "buckets": []}), 500


@app.route("/api/add-discipline")
def add_discipline_api():
    """ADD-trade discipline audit — chasing vs averaging-down vs stacking.

    When the book BUYs into a name it already holds, that ADD carries a
    sign: paying above the running avg_cost is *chasing* (anchoring on
    the original entry, bidding into strength after the easy money is
    gone); paying below is *averaging down* (rational only if the thesis
    is intact, the textbook setup for ``loser_autopsy``'s SLOW_BLEED).
    No existing endpoint watches this — ``trade_asymmetry`` reduces to
    disposition, ``churn`` counts overtrading, ``loser_autopsy`` narrates
    closed losses but doesn't see the ADD moment itself. This is the
    missing surface.

    The closed-round-trip rollup answers the falsifiable question: did
    chasing-ADDs produce worse round-trip P/L than averaging-down ADDs?
    Per-style P/L lets the operator see whether the bot's averaging-down
    is rationally cost-improving or doubling-down on broken theses.

    Composes ``analytics.round_trips.build_round_trips`` as the SSOT
    (AGENTS.md #10) and ``analytics.add_discipline.build_add_discipline``
    over the same trade ledger. Pure read — never raises. Observational
    only — never gates Opus, no caps (AGENTS.md #2/#12)."""
    try:
        from .analytics.round_trips import build_round_trips
        from .analytics.add_discipline import build_add_discipline
        try:
            limit = int(request.args.get("limit", 2000))
        except Exception:
            limit = 2000
        limit = max(50, min(limit, 10000))

        store = get_store()
        trades = store.recent_trades(limit=limit)
        trades_oldest_first = list(reversed(trades or []))
        round_trips = build_round_trips(trades_oldest_first)
        result = build_add_discipline(trades_oldest_first, round_trips)
        result["as_of"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "state": "ERROR",
                        "adds": [], "counts": {}}), 500


@app.route("/api/conviction-deployment-curve")
def conviction_deployment_curve_api():
    """Conviction-deployment curve — does BUY size scale with the news
    ai_score at the moment of entry?

    Joins the BUY trade ledger (``store.recent_trades``) with the
    digital-intern articles.db (live-only rows, mode=ro) and the
    equity_curve (``store.equity_curve``) to compute, per BUY trade:

      * the peak ai_score in the (trade_ts - window, trade_ts] window
        for a word-boundary title match on the trade's ticker
      * the position_size_pct = trade.value / equity_at_trade

    Output has two parallel surfaces — both ship at every sample size:

      * ``evidence`` — per-trade chronological table (operator-readable
        at N=1; the primary surface at low sample size).
      * ``buckets`` — five ai_score buckets (<6 / 6-7 / 7-8 / 8-9 / 9+)
        with n_buys, median/max size_pct, total deployed.

    Verdict ``MONOTONIC`` / ``FLAT`` / ``INVERTED`` / ``INSUFFICIENT``
    rolls up over buckets only — gated on density per-bucket, not
    total trade count. Two trades in the "9+" bucket tell you nothing
    about the curve.

    Query params (clamped):
      ``limit`` — trade lookback row cap, 50..10000 (default 2000)
      ``window_hours_pre_trade`` — peak-score window before each BUY,
        0.5..48 (default 6.0)
      ``article_floor_score`` — minimum ai_score for an article to be
        considered, 0..10 (default 2.0). Lower than the bucket floor
        because articles in [0, 6) still inform the "<6" bucket.

    Pure read — never raises. Articles older than articles.db retention
    degrade per-trade to ``score_unavailable=true`` rather than crash.
    Observational only — never gates Opus, no caps (AGENTS.md #2/#12).
    """
    try:
        from .analytics.conviction_deployment import (
            build_conviction_deployment,
        )

        def _qf(name, default, lo, hi):
            try:
                v = float(request.args.get(name, default))
            except (TypeError, ValueError):
                v = default
            return max(lo, min(hi, v))

        limit = int(_qf("limit", 2000.0, 50.0, 10000.0))
        window_hours = _qf("window_hours_pre_trade", 6.0, 0.5, 48.0)
        article_floor = _qf("article_floor_score", 2.0, 0.0, 10.0)

        store = get_store()
        trades = store.recent_trades(limit=limit) or []
        equity = store.equity_curve(limit=2000) or []

        # Pull articles spanning the trade window — oldest trade ts
        # minus the per-trade scan window minus a small slack so the
        # window edge case still matches.
        articles: list[dict] = []
        if trades:
            oldest_iso = trades[-1].get("timestamp")
            try:
                oldest_dt = datetime.fromisoformat(
                    str(oldest_iso).replace("Z", "+00:00")
                )
            except Exception:
                oldest_dt = datetime.now(timezone.utc) - timedelta(
                    hours=window_hours
                )
            if oldest_dt.tzinfo is None:
                oldest_dt = oldest_dt.replace(tzinfo=timezone.utc)
            since_dt = oldest_dt - timedelta(hours=window_hours + 0.25)
            since = since_dt.isoformat()
            path = _articles_db_path()
            if path is not None:
                conn = None
                try:
                    conn = sqlite3.connect(
                        f"file:{path}?mode=ro", uri=True, timeout=5
                    )
                    art_rows = conn.execute(
                        "SELECT title, url, source, ai_score, urgency, "
                        "first_seen FROM articles "
                        "WHERE first_seen >= ? AND ai_score >= ? "
                        "AND url NOT LIKE 'backtest://%' "
                        "AND source NOT LIKE 'backtest_%' "
                        "AND source NOT LIKE 'opus_annotation%' "
                        "ORDER BY first_seen DESC LIMIT 20000",
                        (since, article_floor),
                    ).fetchall()
                    articles = [{
                        "title": r[0], "url": r[1], "source": r[2],
                        "ai_score": r[3], "urgency": r[4],
                        "first_seen": r[5],
                    } for r in art_rows]
                except Exception:
                    articles = []
                finally:
                    if conn is not None:
                        try:
                            conn.close()
                        except Exception:
                            pass

        result = build_conviction_deployment(
            trades, articles, equity,
            window_hours_pre_trade=window_hours,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "error": str(e),
            "state": "ERROR",
            "verdict": "INSUFFICIENT",
            "headline": f"error: {e}",
            "buckets": [],
            "evidence": [],
        }), 500


@app.route("/api/cash-conviction-fit")
def cash_conviction_fit_api():
    """Cash-conviction fit — point-in-time verdict on whether the
    book's cash level is appropriate given the loudest live signal.

    None of ``capital_paralysis`` / ``idle_opportunity`` /
    ``position_action_brief`` join cash idleness × top signal score ×
    last-decision verb into a single verdict. This endpoint does. It
    works at N=1 trade, N=1 position, N=0 — strictly point-in-time;
    no historical roll-up.

    Verdict matrix:

      * ``IDLE_DESPITE_SURGE`` — cash ≥ idle floor AND top signal ≥
        high-conviction floor AND last decision passive. The book is
        sitting on capital while the loudest signal screams.
      * ``OVERDEPLOYED`` — cash ≤ overdeployed floor AND top signal ≥
        high-conviction floor. Cannot add without trimming.
      * ``IDLE_LOW_CONVICTION`` — cash ≥ idle floor AND top signal <
        low-conviction ceiling. Cash idleness is correct.
      * ``BALANCED`` — none of the above (incl. active recent fill).
      * ``NO_DATA`` — portfolio or signals missing.

    Query params (clamped):
      ``signal_floor`` — ai_score floor for fetched signals, 0..10
        (default 4.0). Low enough that the IDLE_LOW_CONVICTION reading
        still has a top signal to attach to.
      ``signal_window_hours`` — how fresh the signal must be, 0.5..48
        (default 4.0).
      ``idle_cash_pct`` / ``overdeployed_cash_pct`` /
      ``high_conviction_score`` / ``low_conviction_score`` /
      ``recent_fill_max_min`` — threshold overrides (clamped to
        sensible ranges).

    Pure read — never raises. Observational only — never gates Opus,
    no caps (AGENTS.md #2/#12).
    """
    try:
        from .analytics.cash_conviction_fit import (
            build_cash_conviction_fit,
            DEFAULT_IDLE_CASH_PCT,
            DEFAULT_OVERDEPLOYED_CASH_PCT,
            DEFAULT_HIGH_CONVICTION_SCORE,
            DEFAULT_LOW_CONVICTION_SCORE,
            DEFAULT_RECENT_FILL_MAX_MIN,
        )

        def _qf(name, default, lo, hi):
            try:
                v = float(request.args.get(name, default))
            except (TypeError, ValueError):
                v = default
            return max(lo, min(hi, v))

        signal_floor = _qf("signal_floor", 4.0, 0.0, 10.0)
        signal_window_hours = _qf("signal_window_hours", 4.0, 0.5, 48.0)
        idle_cash_pct = _qf("idle_cash_pct", DEFAULT_IDLE_CASH_PCT, 0.0, 100.0)
        overdeployed_cash_pct = _qf(
            "overdeployed_cash_pct", DEFAULT_OVERDEPLOYED_CASH_PCT, 0.0, 100.0,
        )
        high_conviction_score = _qf(
            "high_conviction_score", DEFAULT_HIGH_CONVICTION_SCORE, 0.0, 10.0,
        )
        low_conviction_score = _qf(
            "low_conviction_score", DEFAULT_LOW_CONVICTION_SCORE, 0.0, 10.0,
        )
        recent_fill_max_min = _qf(
            "recent_fill_max_min", DEFAULT_RECENT_FILL_MAX_MIN, 0.0, 1440.0,
        )

        store = get_store()
        # Build the portfolio snapshot in the same shape the other
        # cash-aware endpoints use.
        pf = store.get_portfolio() or {}
        cash = pf.get("cash")
        total_value = pf.get("total_value")
        cash_pct = None
        if isinstance(cash, (int, float)) and isinstance(total_value, (int, float)) and total_value > 0:
            cash_pct = (cash / total_value) * 100.0
        positions_open = [
            p for p in (pf.get("positions") or [])
            if isinstance(p, dict) and not p.get("closed_at")
        ]
        portfolio = {
            "cash": cash,
            "total_value": total_value,
            "cash_pct": cash_pct,
            "n_positions": len(positions_open),
        }
        held_tickers = {
            p.get("ticker") for p in positions_open
            if isinstance(p.get("ticker"), str)
        }

        # Pull top live signals from articles.db.
        signals: list[dict] = []
        path = _articles_db_path()
        if path is not None:
            now_utc = datetime.now(timezone.utc)
            since = (now_utc - timedelta(hours=signal_window_hours)).isoformat()
            conn = None
            try:
                conn = sqlite3.connect(
                    f"file:{path}?mode=ro", uri=True, timeout=5,
                )
                rows = conn.execute(
                    "SELECT title, source, ai_score, urgency, first_seen "
                    "FROM articles WHERE first_seen >= ? "
                    "AND ai_score >= ? "
                    "AND url NOT LIKE 'backtest://%' "
                    "AND source NOT LIKE 'backtest_%' "
                    "AND source NOT LIKE 'opus_annotation%' "
                    "ORDER BY ai_score DESC LIMIT 200",
                    (since, signal_floor),
                ).fetchall()
            except Exception:
                rows = []
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
            # Map each article to its loudest book-ticker mention. We
            # delegate to the same ticker-extraction convention used by
            # briefing_coverage_audit: word-boundary match against the
            # title. The active universe is the held book PLUS the
            # canonical _BOOK_TICKERS so unheld surging names still
            # surface (the IDLE_DESPITE_SURGE failure mode is
            # specifically about *unheld* names — held-name signal
            # only blocks adding, not opening fresh exposure).
            try:
                from analysis.claude_analyst import _BOOK_TICKERS
                book = set(_BOOK_TICKERS) | held_tickers
            except Exception:
                book = held_tickers
            for r in rows:
                title = r[0] or ""
                for tkr in book:
                    if not isinstance(tkr, str) or not tkr:
                        continue
                    # Same word-boundary convention as briefing_coverage_audit.
                    import re as _re
                    if _re.search(rf"\b{_re.escape(tkr)}\b", title):
                        signals.append({
                            "ticker": tkr,
                            "ai_score": r[2],
                            "urgency": r[3],
                            "source": r[1],
                            "first_seen": r[4],
                            "held": tkr in held_tickers,
                            "title": title,
                        })
                        break  # one ticker per article — first match wins
        # Last decision (any verb).
        decisions = store.recent_decisions(limit=1) or []
        last_decision = decisions[0] if decisions else None

        result = build_cash_conviction_fit(
            portfolio, signals, last_decision,
            idle_cash_pct=idle_cash_pct,
            overdeployed_cash_pct=overdeployed_cash_pct,
            high_conviction_score=high_conviction_score,
            low_conviction_score=low_conviction_score,
            recent_fill_max_min=recent_fill_max_min,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "error": str(e),
            "verdict": "ERROR",
            "headline": f"error: {e}",
            "portfolio": {},
            "top_signal": {},
            "last_decision": {},
            "thresholds": {},
        }), 500


def _build_realized_pct_index(trades_oldest_first: list[dict]) -> dict:
    """Map each BUY trade.id → realized_pct + closed flag.

    BUY-only attribution by design (no double-counting). A round-trip
    with 2 BUYs + 1 SELL contributes 2 samples (one per BUY), not 3.
    A SELL's "realized return" is just the round-trip's outcome that
    its sibling BUY(s) already encoded — re-using the SELL id would
    inflate the bucket count without adding new information. Exit-
    decision skill is a *different* question (forward return *after*
    the SELL) and belongs to a separate endpoint.

    Closed BUY legs use the round-trip's ``pnl_pct``. Open BUY legs
    (the bot is still long the name) use a mark-to-current via the
    position record's ``current_price`` vs ``avg_cost``. BUYs whose
    round-trip never closed AND have no position match (a stale BUY
    on a name the bot has since left, with no SELL captured) get
    ``None`` and are silently dropped by the caller.

    Composes ``analytics.round_trips.build_round_trips`` as the SSOT
    (AGENTS.md #10)."""
    from .analytics.round_trips import build_round_trips
    out: dict[int, dict] = {}
    rts = build_round_trips(trades_oldest_first)
    # Closed round-trips: BUY entry ids only pick up the round-trip's
    # pnl_pct. SELL exit ids are deliberately NOT attributed — see
    # docstring.
    for rt in rts:
        pnl_pct = rt.get("pnl_pct")
        for tid in (rt.get("entry_trade_ids") or []):
            if isinstance(tid, int):
                out[tid] = {"realized_pct": pnl_pct, "closed": True}
    # Open positions: any BUY trade whose round-trip is *not* in the
    # closed-rt list above must belong to an open position. Mark to
    # current via the positions table.
    try:
        store = get_store()
        # store.open_positions() already excludes closed_at IS NULL AND qty > 0
        positions = store.open_positions() or []
    except Exception:
        positions = []
    open_marks: dict[tuple, float] = {}
    for p in positions:
        avg = p.get("avg_cost")
        cur = p.get("current_price")
        if not isinstance(avg, (int, float)) or not isinstance(cur, (int, float)):
            continue
        if avg <= 0:
            continue
        key = (
            p.get("ticker"),
            p.get("type") or "stock",
            p.get("strike"),
            p.get("expiry"),
        )
        open_marks[key] = (cur - avg) / avg * 100.0
    for t in trades_oldest_first:
        tid = t.get("id")
        if not isinstance(tid, int) or tid in out:
            continue
        action = (t.get("action") or "").upper()
        if not action.startswith("BUY"):
            continue
        key = (
            t.get("ticker"),
            t.get("option_type") or "stock",
            t.get("strike"),
            t.get("expiry"),
        )
        mark = open_marks.get(key)
        if mark is None:
            continue
        out[tid] = {"realized_pct": round(mark, 4), "closed": False}
    return out


def _freshest_article_age_index(
    trades_oldest_first: list[dict],
    lookback_days: float = 30.0,
) -> dict:
    """Map each trade.id → freshest article-age-in-minutes-at-trade-ts.

    Pulls articles spanning ``oldest_trade_ts - lookback_days`` from
    ``articles.db`` (live-only filter, mode=ro), bins by
    word-boundary title match against the trade's ticker, then for
    each trade returns ``(trade_ts - newest_article_first_seen_before_ts)``
    in minutes. Missing DB / no matching article / parse failure all
    degrade to ``None`` — the builder routes those into NO_NEWS.

    Word-boundary discipline mirrors ``cash_conviction_fit``'s join
    (re-uses ``\\b{ticker}\\b`` in the title)."""
    import re as _re
    out: dict[int, float | None] = {}
    if not trades_oldest_first:
        return out
    tickers = {
        (t.get("ticker") or "").upper()
        for t in trades_oldest_first
        if isinstance(t.get("ticker"), str) and t.get("ticker").strip()
    }
    tickers.discard("")
    if not tickers:
        return out
    # Articles since (oldest_trade_ts - lookback_days). Bound the
    # window so a months-old trade ledger doesn't trigger a full
    # articles.db scan.
    try:
        oldest_ts = trades_oldest_first[0].get("timestamp")
        oldest_dt = datetime.fromisoformat(
            str(oldest_ts).replace("Z", "+00:00")
        )
        if oldest_dt.tzinfo is None:
            oldest_dt = oldest_dt.replace(tzinfo=timezone.utc)
    except Exception:
        oldest_dt = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    since = (oldest_dt - timedelta(days=lookback_days)).isoformat()
    path = _articles_db_path()
    if path is None:
        for t in trades_oldest_first:
            tid = t.get("id")
            if isinstance(tid, int):
                out[tid] = None
        return out
    # Per-ticker timeline: sorted list of first_seen datetimes for
    # titles word-boundary-matching the ticker.
    timelines: dict[str, list[datetime]] = {t: [] for t in tickers}
    conn = None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
        rows = conn.execute(
            "SELECT title, first_seen FROM articles "
            "WHERE first_seen >= ? AND ai_score > 0 "
            "AND url NOT LIKE 'backtest://%' "
            "AND source NOT LIKE 'backtest_%' "
            "AND source NOT LIKE 'opus_annotation%' "
            "ORDER BY first_seen ASC LIMIT 200000",
            (since,),
        ).fetchall()
    except Exception:
        rows = []
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    # Pre-compile per-ticker patterns once.
    patterns = {t: _re.compile(rf"\b{_re.escape(t)}\b") for t in tickers}
    for title, fs in rows:
        if not isinstance(title, str) or not isinstance(fs, str):
            continue
        try:
            fs_dt = datetime.fromisoformat(fs.replace("Z", "+00:00"))
        except Exception:
            continue
        if fs_dt.tzinfo is None:
            fs_dt = fs_dt.replace(tzinfo=timezone.utc)
        for tkr, rx in patterns.items():
            if rx.search(title):
                timelines[tkr].append(fs_dt)
    # Per-trade: bisect for newest article strictly before trade_ts.
    import bisect as _bisect
    for t in trades_oldest_first:
        tid = t.get("id")
        if not isinstance(tid, int):
            continue
        tkr = (t.get("ticker") or "").upper()
        if not tkr or tkr not in timelines or not timelines[tkr]:
            out[tid] = None
            continue
        try:
            trade_dt = datetime.fromisoformat(
                str(t.get("timestamp")).replace("Z", "+00:00")
            )
            if trade_dt.tzinfo is None:
                trade_dt = trade_dt.replace(tzinfo=timezone.utc)
        except Exception:
            out[tid] = None
            continue
        timeline = timelines[tkr]
        # rightmost index where article_ts < trade_dt
        idx = _bisect.bisect_left(timeline, trade_dt) - 1
        if idx < 0:
            out[tid] = None
            continue
        age_min = (trade_dt - timeline[idx]).total_seconds() / 60.0
        out[tid] = age_min if age_min >= 0 else None
    return out


@app.route("/api/news-age-at-decision-skill")
def news_age_at_decision_skill_api():
    """News-age-at-decision skill — does fresher news at trade time
    predict a better realized outcome?

    For each FILLED BUY/SELL, joins the trade ledger to
    ``articles.db`` (live-only, mode=ro) by word-boundary title match
    on the trade's ticker, finds the *freshest* article that existed
    *before* the trade timestamp, and pairs that article-age with the
    trade's realized return (closed round-trip pnl_pct via the
    ``round_trips`` SSOT, or mark-to-current on still-open positions).

    Verdict matrix: ``FRESH_NEWS_BETTER`` / ``STALE_NEWS_BETTER`` /
    ``NO_PATTERN`` / ``INSUFFICIENT_DATA``. Bucket edges
    (FRESH_LT_60M / HOURS_1_TO_6 / HOURS_6_TO_24 / STALE_GT_24H /
    NO_NEWS) are module constants pinned by tests.

    Existing neighbours each see a *different* slice:
      * ``/api/news-to-trade-lag`` — gap from first article on a
        catalyst to first trade. Not the per-trade gradient.
      * ``/api/news-edge`` / ``/api/source-edge`` — per-source
        predictive edge. Says nothing about freshness at decision.
      * ``/api/decision-context-completeness`` —
        news-present-or-absent. Binary, not a gradient.

    Query params (clamped):
      ``limit`` — trade lookback row cap, 50..10000 (default 2000)
      ``lookback_days`` — articles.db scan window before oldest
        trade, 1..120 (default 30)
      ``min_per_bucket`` — verdict gate, 1..20 (default 3)
      ``verdict_gap_pct`` — mean-return gap (pp) for a directional
        verdict, 0.1..20 (default 2.0)

    Pure read — never raises. Articles outside the
    ``RETENTION_DAYS=90`` window are unavailable; trades whose
    matching article fell off retention degrade to NO_NEWS rather
    than crash. Observational only — never gates Opus, no caps
    (AGENTS.md #2/#12).
    """
    try:
        from .analytics.news_age_at_decision_skill import (
            build_news_age_at_decision_skill,
            MIN_PER_BUCKET,
            VERDICT_GAP_PCT,
        )

        def _qf(name, default, lo, hi):
            try:
                v = float(request.args.get(name, default))
            except (TypeError, ValueError):
                v = default
            return max(lo, min(hi, v))

        limit = int(_qf("limit", 2000.0, 50.0, 10000.0))
        lookback_days = _qf("lookback_days", 30.0, 1.0, 120.0)
        min_per_bucket = int(_qf(
            "min_per_bucket", float(MIN_PER_BUCKET), 1.0, 20.0,
        ))
        verdict_gap_pct = _qf(
            "verdict_gap_pct", VERDICT_GAP_PCT, 0.1, 20.0,
        )

        store = get_store()
        trades = store.recent_trades(limit=limit) or []
        trades_oldest_first = list(reversed(trades))

        realized = _build_realized_pct_index(trades_oldest_first)
        ages = _freshest_article_age_index(
            trades_oldest_first, lookback_days=lookback_days,
        )

        samples = []
        for t in trades_oldest_first:
            tid = t.get("id")
            if not isinstance(tid, int):
                continue
            r = realized.get(tid)
            if not r:
                continue
            realized_pct = r.get("realized_pct")
            if realized_pct is None:
                continue
            samples.append({
                "trade_id": tid,
                "trade_ts": t.get("timestamp"),
                "ticker": t.get("ticker"),
                "action": t.get("action"),
                "freshest_article_age_min": ages.get(tid),
                "realized_pct": realized_pct,
                "closed": r.get("closed", False),
            })

        return jsonify(build_news_age_at_decision_skill(
            samples,
            min_per_bucket=min_per_bucket,
            verdict_gap_pct=verdict_gap_pct,
        ))
    except Exception as e:
        return jsonify({
            "error": str(e),
            "verdict": "ERROR",
            "headline": f"error: {e}",
            "n_samples": 0,
            "buckets": {},
            "thresholds": {},
            "samples": [],
        }), 500


def _corroborating_article_count_index(
    trades_oldest_first: list[dict],
    lookback_hours: float = 24.0,
) -> dict:
    """Map each trade.id → count of distinct articles mentioning the trade's
    ticker in the ``lookback_hours`` window strictly before the trade
    timestamp.

    Re-uses the same articles.db join shape as
    ``_freshest_article_age_index`` (live-only filter, mode=ro,
    word-boundary title match). Missing DB / no matching article / parse
    failure all degrade to ``0`` so the builder routes those into
    ``NO_NEWS`` (the news_corroboration_skill ``_bucket_for`` contract).

    The lookback window is anchored on each trade individually — articles
    older than ``trade_ts - lookback_hours`` are excluded from that trade's
    count even if they sit inside the global scan window (set wider so a
    single articles.db scan covers every trade in the ledger)."""
    import re as _re
    out: dict[int, int] = {}
    if not trades_oldest_first:
        return out
    tickers = {
        (t.get("ticker") or "").upper()
        for t in trades_oldest_first
        if isinstance(t.get("ticker"), str) and t.get("ticker").strip()
    }
    tickers.discard("")
    if not tickers:
        for t in trades_oldest_first:
            tid = t.get("id")
            if isinstance(tid, int):
                out[tid] = 0
        return out
    # Articles since (oldest_trade_ts - lookback_hours). One DB scan
    # covers every trade — per-trade clipping happens in Python below.
    try:
        oldest_ts = trades_oldest_first[0].get("timestamp")
        oldest_dt = datetime.fromisoformat(
            str(oldest_ts).replace("Z", "+00:00")
        )
        if oldest_dt.tzinfo is None:
            oldest_dt = oldest_dt.replace(tzinfo=timezone.utc)
    except Exception:
        oldest_dt = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    since = (oldest_dt - timedelta(hours=lookback_hours)).isoformat()
    path = _articles_db_path()
    if path is None:
        for t in trades_oldest_first:
            tid = t.get("id")
            if isinstance(tid, int):
                out[tid] = 0
        return out
    timelines: dict[str, list[datetime]] = {t: [] for t in tickers}
    conn = None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
        rows = conn.execute(
            "SELECT title, first_seen FROM articles "
            "WHERE first_seen >= ? AND ai_score > 0 "
            "AND url NOT LIKE 'backtest://%' "
            "AND source NOT LIKE 'backtest_%' "
            "AND source NOT LIKE 'opus_annotation%' "
            "ORDER BY first_seen ASC LIMIT 200000",
            (since,),
        ).fetchall()
    except Exception:
        rows = []
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    patterns = {t: _re.compile(rf"\b{_re.escape(t)}\b") for t in tickers}
    for title, fs in rows:
        if not isinstance(title, str) or not isinstance(fs, str):
            continue
        try:
            fs_dt = datetime.fromisoformat(fs.replace("Z", "+00:00"))
        except Exception:
            continue
        if fs_dt.tzinfo is None:
            fs_dt = fs_dt.replace(tzinfo=timezone.utc)
        for tkr, rx in patterns.items():
            if rx.search(title):
                timelines[tkr].append(fs_dt)
    import bisect as _bisect
    window = timedelta(hours=lookback_hours)
    for t in trades_oldest_first:
        tid = t.get("id")
        if not isinstance(tid, int):
            continue
        tkr = (t.get("ticker") or "").upper()
        if not tkr or tkr not in timelines or not timelines[tkr]:
            out[tid] = 0
            continue
        try:
            trade_dt = datetime.fromisoformat(
                str(t.get("timestamp")).replace("Z", "+00:00")
            )
            if trade_dt.tzinfo is None:
                trade_dt = trade_dt.replace(tzinfo=timezone.utc)
        except Exception:
            out[tid] = 0
            continue
        timeline = timelines[tkr]
        # Articles with first_seen in [trade_dt - window, trade_dt) — strictly
        # before the trade (we never count an article that landed at or after
        # the decision microsecond as "corroborating" that decision).
        hi = _bisect.bisect_left(timeline, trade_dt)
        lo = _bisect.bisect_left(timeline, trade_dt - window)
        out[tid] = max(0, hi - lo)
    return out


@app.route("/api/news-corroboration-skill")
def news_corroboration_skill_api():
    """News-corroboration skill — does Opus need a chorus, or does it pick
    winners on a single fresh signal?

    For each FILLED BUY/SELL, counts the number of distinct articles
    mentioning the trade's ticker in the ``lookback_hours`` window
    strictly before the trade timestamp, then pairs each count with the
    trade's realized return (closed round-trip pnl_pct via the
    ``round_trips`` SSOT, or mark-to-current on still-open positions).

    Verdict matrix: ``CORROBORATION_HELPS`` (CHORUS+ mean ≥ SINGLE mean +
    ``verdict_gap_pct``), ``SINGLE_HELPS`` (CHORUS+ underperforms SINGLE
    by the same threshold), ``NO_PATTERN`` (both buckets full but within
    tolerance), ``INSUFFICIENT_DATA`` (either bucket below
    ``min_per_bucket``).

    Buckets: ``NO_NEWS`` (0), ``SINGLE`` (1), ``SMALL_CHORUS`` (2-3),
    ``CHORUS`` (4-9), ``FLOOD`` (10+). The verdict compares CHORUS+
    (CHORUS ∪ FLOOD) vs SINGLE — the two operational extremes.

    Existing neighbours each see a *different* slice:
      * ``/api/news-age-at-decision-skill`` — freshness of the freshest
        article. Says nothing about how many articles agreed.
      * ``/api/news-themes`` — per-ticker decayed-score snapshot *now*,
        not a per-trade outcome.
      * ``/api/signal-followthrough`` — acted-on vs ignored selection
        edge, not corroboration count vs outcome.

    Query params (clamped):
      ``limit`` — trade lookback row cap, 50..10000 (default 2000)
      ``lookback_hours`` — articles-before-trade window, 1..168
        (default 24)
      ``min_per_bucket`` — verdict gate, 1..20 (default 3)
      ``verdict_gap_pct`` — mean-return gap (pp) for a directional
        verdict, 0.1..20 (default 2.0)

    Pure read — never raises. Articles outside the
    ``RETENTION_DAYS=90`` window are unavailable; trades whose article
    history fell off retention degrade to NO_NEWS rather than crash.
    Observational only — never gates Opus, no caps (AGENTS.md #2/#12).
    """
    try:
        from .analytics.news_corroboration_skill import (
            build_news_corroboration_skill,
            MIN_PER_BUCKET,
            VERDICT_GAP_PCT,
        )

        def _qf(name, default, lo, hi):
            try:
                v = float(request.args.get(name, default))
            except (TypeError, ValueError):
                v = default
            return max(lo, min(hi, v))

        limit = int(_qf("limit", 2000.0, 50.0, 10000.0))
        lookback_hours = _qf("lookback_hours", 24.0, 1.0, 168.0)
        min_per_bucket = int(_qf(
            "min_per_bucket", float(MIN_PER_BUCKET), 1.0, 20.0,
        ))
        verdict_gap_pct = _qf(
            "verdict_gap_pct", VERDICT_GAP_PCT, 0.1, 20.0,
        )

        store = get_store()
        trades = store.recent_trades(limit=limit) or []
        trades_oldest_first = list(reversed(trades))

        realized = _build_realized_pct_index(trades_oldest_first)
        counts = _corroborating_article_count_index(
            trades_oldest_first, lookback_hours=lookback_hours,
        )

        samples = []
        for t in trades_oldest_first:
            tid = t.get("id")
            if not isinstance(tid, int):
                continue
            r = realized.get(tid)
            if not r:
                continue
            realized_pct = r.get("realized_pct")
            if realized_pct is None:
                continue
            samples.append({
                "trade_id": tid,
                "trade_ts": t.get("timestamp"),
                "ticker": t.get("ticker"),
                "action": t.get("action"),
                "article_count": counts.get(tid, 0),
                "realized_pct": realized_pct,
                "closed": r.get("closed", False),
            })

        return jsonify(build_news_corroboration_skill(
            samples,
            min_per_bucket=min_per_bucket,
            verdict_gap_pct=verdict_gap_pct,
        ))
    except Exception as e:
        return jsonify({
            "error": str(e),
            "verdict": "ERROR",
            "headline": f"error: {e}",
            "n_samples": 0,
            "buckets": {},
            "thresholds": {},
            "samples": [],
        }), 500


@app.route("/api/conviction-language-skill")
def conviction_language_skill_api():
    """Conviction-language skill — is Opus' confidence language
    calibrated to its realized outcomes?

    For each FILLED BUY/SELL, parses the verbatim ``trades.reason``
    text for conviction-strength phrases (HIGH/LOW/ADD/NEUTRAL —
    classifier precedence pinned in tests) and pairs each bucket with
    realized P&L from the ``round_trips`` SSOT (or mark-to-current
    for open positions).

    Verdict matrix: ``SELF_AWARE`` (HIGH bucket outperforms LOW by
    ≥ ``verdict_gap_pct``), ``OVERCONFIDENT`` (HIGH underperforms
    LOW by the same threshold), ``NO_PATTERN`` (both buckets full
    but within tolerance), ``INSUFFICIENT_DATA`` (either bucket
    below ``min_per_bucket``).

    Existing neighbours each see a *different* slice:
      * ``/api/cash-conviction-fit`` — point-in-time cash vs loudest
        live signal. Forward-looking, not a calibration check.
      * ``/api/conviction-deployment-curve`` — does *capital
        deployed* scale with external conviction. About sizing.
      * ``/api/decision-confidence`` — distribution of self-reported
        scalar over time. Doesn't link confidence to outcome.

    Query params (clamped):
      ``limit`` — trade lookback row cap, 50..10000 (default 2000)
      ``min_per_bucket`` — verdict gate, 1..20 (default 3)
      ``verdict_gap_pct`` — mean-return gap (pp) for a directional
        verdict, 0.1..20 (default 2.0)

    Pure read — never raises. Observational only — never gates
    Opus, no caps (AGENTS.md #2/#12).
    """
    try:
        from .analytics.conviction_language_skill import (
            build_conviction_language_skill,
            MIN_PER_BUCKET,
            VERDICT_GAP_PCT,
        )

        def _qf(name, default, lo, hi):
            try:
                v = float(request.args.get(name, default))
            except (TypeError, ValueError):
                v = default
            return max(lo, min(hi, v))

        limit = int(_qf("limit", 2000.0, 50.0, 10000.0))
        min_per_bucket = int(_qf(
            "min_per_bucket", float(MIN_PER_BUCKET), 1.0, 20.0,
        ))
        verdict_gap_pct = _qf(
            "verdict_gap_pct", VERDICT_GAP_PCT, 0.1, 20.0,
        )

        store = get_store()
        trades = store.recent_trades(limit=limit) or []
        trades_oldest_first = list(reversed(trades))

        realized = _build_realized_pct_index(trades_oldest_first)

        samples = []
        for t in trades_oldest_first:
            tid = t.get("id")
            if not isinstance(tid, int):
                continue
            r = realized.get(tid)
            if not r:
                continue
            realized_pct = r.get("realized_pct")
            if realized_pct is None:
                continue
            samples.append({
                "trade_id": tid,
                "trade_ts": t.get("timestamp"),
                "ticker": t.get("ticker"),
                "action": t.get("action"),
                "reason": t.get("reason"),
                "realized_pct": realized_pct,
                "closed": r.get("closed", False),
            })

        return jsonify(build_conviction_language_skill(
            samples,
            min_per_bucket=min_per_bucket,
            verdict_gap_pct=verdict_gap_pct,
        ))
    except Exception as e:
        return jsonify({
            "error": str(e),
            "verdict": "ERROR",
            "headline": f"error: {e}",
            "n_samples": 0,
            "buckets": {},
            "thresholds": {},
            "samples": [],
        }), 500


@app.route("/api/regime-leverage-fit-skill")
def regime_leverage_fit_skill_api():
    """Regime × leveraged-ETF-exposure fit — is the book's leverage
    class appropriate to the current SPY-20d regime?

    The recurring backtest finding (paper-trader AGENTS.md): the
    bot's alpha is largely a leveraged-bull-window artifact. This
    endpoint surfaces the live state of that risk: classifies SPY
    20d momentum into bull/sideways/bear, measures current
    leveraged-ETF book exposure %, measures recent leveraged-ETF
    BUY flow in a configurable window, and emits a single regime-fit
    verdict.

    Verdict matrix:

      * ``BLIND_LEVERING`` — bear/sideways regime AND recent
        leveraged BUY flow ≥ ``high_flow_pct``. Highest priority —
        the *direction of change* matters more than static exposure.
      * ``DANGEROUS_HEADWIND`` — bear regime AND lev_pct ≥
        ``high_lev_floor``. Static high exposure into a bear tape.
      * ``ALIGNED`` — bull regime AND lev_pct ≥ ``aligned_lev_floor``.
        Correctly tailwinded.
      * ``MISSED_TAILWIND`` — bull regime AND lev_pct ≤
        ``low_lev_ceil``. Under-allocating to the bull.
      * ``DEFENSIVE`` — bear/sideways AND lev_pct ≤ ``low_lev_ceil``.
        Correctly de-risked.
      * ``NEUTRAL`` — mid-band exposure or ambiguous regime + flow.
      * ``NO_DATA`` — empty everything.

    Query params (clamped):
      ``flow_window_hours`` — recent buy-flow window, 0.5..168 (default 24)
      ``high_flow_pct`` — buy-flow ≥ this triggers BLIND_LEVERING,
        0.0..100 (default 5.0)
      ``high_lev_floor`` / ``aligned_lev_floor`` / ``low_lev_ceil`` —
        exposure thresholds (clamped 0..100)
      ``bull_mom_pct`` / ``bear_mom_pct`` — regime cutoffs
        (clamped sensibly)

    Pure read — never raises. Observational only — never gates Opus,
    no caps (AGENTS.md #2/#12).
    """
    try:
        from .analytics.regime_leverage_fit_skill import (
            build_regime_leverage_fit_skill,
            DEFAULT_BULL_MOM_PCT,
            DEFAULT_BEAR_MOM_PCT,
            DEFAULT_HIGH_LEV_FLOOR,
            DEFAULT_ALIGNED_LEV_FLOOR,
            DEFAULT_LOW_LEV_CEIL,
            DEFAULT_FLOW_WINDOW_HOURS,
            DEFAULT_HIGH_FLOW_PCT,
        )

        def _qf(name, default, lo, hi):
            try:
                v = float(request.args.get(name, default))
            except (TypeError, ValueError):
                v = default
            return max(lo, min(hi, v))

        flow_window_hours = _qf(
            "flow_window_hours", DEFAULT_FLOW_WINDOW_HOURS, 0.5, 168.0,
        )
        high_flow_pct = _qf(
            "high_flow_pct", DEFAULT_HIGH_FLOW_PCT, 0.0, 100.0,
        )
        high_lev_floor = _qf(
            "high_lev_floor", DEFAULT_HIGH_LEV_FLOOR, 0.0, 100.0,
        )
        aligned_lev_floor = _qf(
            "aligned_lev_floor", DEFAULT_ALIGNED_LEV_FLOOR, 0.0, 100.0,
        )
        low_lev_ceil = _qf(
            "low_lev_ceil", DEFAULT_LOW_LEV_CEIL, 0.0, 100.0,
        )
        bull_mom_pct = _qf(
            "bull_mom_pct", DEFAULT_BULL_MOM_PCT, -20.0, 20.0,
        )
        bear_mom_pct = _qf(
            "bear_mom_pct", DEFAULT_BEAR_MOM_PCT, -20.0, 20.0,
        )

        store = get_store()
        pf = store.get_portfolio() or {}
        cash_usd = pf.get("cash")
        total_value_usd = pf.get("total_value")
        positions_open = [
            p for p in (pf.get("positions") or [])
            if isinstance(p, dict) and not p.get("closed_at")
        ]

        # Trades for recent flow — cap pull at a sensible upper bound
        # (the flow window itself filters in the builder).
        trades = store.recent_trades(limit=2000) or []

        # SPY 20d momentum via the same live quant path the strategy
        # uses. Guarded — fall back to None so the builder degrades
        # gracefully to NEUTRAL / NO_DATA rather than failing.
        spy_mom_20d = None
        try:
            from .strategy import get_quant_signals_live
            spy_q = (get_quant_signals_live(["SPY"]) or {}).get("SPY") or {}
            mv = spy_q.get("mom_20d")
            if isinstance(mv, (int, float)):
                spy_mom_20d = float(mv)
        except Exception:
            spy_mom_20d = None

        return jsonify(build_regime_leverage_fit_skill(
            positions_open,
            cash_usd,
            total_value_usd,
            spy_mom_20d,
            trades,
            flow_window_hours=flow_window_hours,
            high_flow_pct=high_flow_pct,
            high_lev_floor=high_lev_floor,
            aligned_lev_floor=aligned_lev_floor,
            low_lev_ceil=low_lev_ceil,
            bull_mom_pct=bull_mom_pct,
            bear_mom_pct=bear_mom_pct,
        ))
    except Exception as e:
        return jsonify({
            "verdict": "ERROR",
            "headline": f"error: {e}",
            "error": str(e),
            "regime": "unknown",
            "spy_mom_20d": None,
            "portfolio": {},
            "recent_flow": {},
            "thresholds": {},
        }), 500


@app.route("/api/peer-momentum-divergence-skill")
def peer_momentum_divergence_skill_api():
    """Per-held-name price-momentum vs sector-peer-median divergence.

    The existing sector / momentum surface answers *bucket* questions
    (sector_velocity_delta — per-sector news velocity;
    sector_signal_fit — book exposure vs sector news; sector_heatmap
    — per-bucket price+RSI+news; sector_pulse — per-ticker snapshot).
    None ask: *is my held name moving with its sector peers, or is it
    ripping/lagging on its own?* That's a real single-name catalyst
    risk signal an operator who is long ONE name needs.

    Per-position verdict matrix:
      * ``IDIOSYNCRATIC_RALLY`` — held mom_5d ≥ peers + delta AND
        held positive while peers ≤ 0. Single-name rip.
      * ``IDIOSYNCRATIC_DECLINE`` — held mom_5d ≤ peers - delta AND
        held negative while peers ≥ 0. Single-name drop.
      * ``LEADING_PEERS`` / ``LAGGING_PEERS`` — magnitude
        out/under-performance with directional alignment.
      * ``TRACKING_PEERS`` — within ``delta_pct`` of peer median.
      * ``NO_PEERS`` — sector unmapped or peer momentums missing.

    Top-level roll-up: ``BOOK_IDIOSYNCRATIC`` /
    ``BOOK_DIVERGENT`` / ``BOOK_TRACKING`` / ``INSUFFICIENT_DATA`` /
    ``NO_DATA``.

    Query params (clamped):
      ``delta_pct`` — per-position |held - peer_median| 5d-mom
        threshold, 0.1..20 (default 2.0)
      ``min_peers`` — peer momentums needed for a stable median,
        1..10 (default 2)
      ``min_positions`` — book-level INSUFFICIENT_DATA floor, 1..10
        (default 1)

    Pure read — never raises. Observational only — never gates Opus,
    no caps (AGENTS.md #2/#12).
    """
    try:
        from .analytics.peer_momentum_divergence_skill import (
            build_peer_momentum_divergence_skill,
            PEERS_BY_SECTOR,
            DEFAULT_DELTA_PCT,
            DEFAULT_MIN_PEERS,
            DEFAULT_MIN_POSITIONS,
        )

        def _qf(name, default, lo, hi):
            try:
                v = float(request.args.get(name, default))
            except (TypeError, ValueError):
                v = default
            return max(lo, min(hi, v))

        delta_pct = _qf("delta_pct", DEFAULT_DELTA_PCT, 0.1, 20.0)
        min_peers = int(_qf(
            "min_peers", float(DEFAULT_MIN_PEERS), 1.0, 10.0,
        ))
        min_positions = int(_qf(
            "min_positions", float(DEFAULT_MIN_POSITIONS), 1.0, 10.0,
        ))

        store = get_store()
        positions_open = [
            p for p in (store.open_positions() or [])
            if isinstance(p, dict)
            and p.get("type") == "stock"
            and (p.get("qty") or 0) > 0
        ]

        # Collect every ticker we need a mom_5d for: each held name
        # + every peer of its sector. One batched quant call.
        held_set = set()
        wanted = set()
        for p in positions_open:
            t = p.get("ticker")
            if isinstance(t, str) and t:
                held_set.add(t.upper())
                wanted.add(t.upper())
        for tk in list(held_set):
            from .analytics.peer_momentum_divergence_skill import (
                TICKER_TO_SECTOR,
            )
            sec = TICKER_TO_SECTOR.get(tk)
            if sec is not None:
                for peer in PEERS_BY_SECTOR.get(sec, []):
                    wanted.add(peer.upper())

        quant_signals: dict[str, dict] = {}
        if wanted:
            try:
                from .strategy import get_quant_signals_live
                quant_signals = get_quant_signals_live(sorted(wanted)) or {}
            except Exception:
                quant_signals = {}

        return jsonify(build_peer_momentum_divergence_skill(
            positions_open,
            quant_signals,
            delta_pct=delta_pct,
            min_peers=min_peers,
            min_positions=min_positions,
        ))
    except Exception as e:
        return jsonify({
            "verdict": "ERROR",
            "headline": f"error: {e}",
            "error": str(e),
            "n_positions": 0,
            "n_with_peers": 0,
            "positions": [],
            "thresholds": {},
        }), 500


@app.route("/api/cash-redeployment-latency-skill")
def cash_redeployment_latency_skill_api():
    """Cash-redeployment latency — after each SELL, how long until any
    new BUY? Catches the documented "sold-then-sat" cash-deployment
    pathology that ``capital_paralysis`` / ``idle_opportunity``
    (snapshots) and ``rebuy_regret`` / ``reentry_velocity`` (per-ticker
    same-name) all miss.

    Verdict matrix:

      * ``FAST_REDEPLOY`` — median ≤ 6h AND ≥ 80% of SELLs redeployed.
      * ``STEADY`` — median ≤ 24h AND ≥ 70% redeployed.
      * ``SLOW`` — median ≤ 72h OR 50-70% redeployed.
      * ``STALLED`` — median > 72h OR < 50% redeployed.
      * ``NO_DATA`` — fewer than 3 classifiable SELLs in window.

    Query params (clamped):
      ``window_days`` — analysis window, 1..365 (default 30)
      ``stalled_cutoff_hours`` — SELLs with no BUY within this are
        STALLED, 1..720 (default 168 = 1 week)
      ``fast_median_h`` / ``steady_median_h`` / ``slow_median_h`` —
        verdict median thresholds (clamped 0..720)
      ``healthy_redeploy_pct`` / ``steady_redeploy_pct`` /
        ``degraded_redeploy_pct`` — verdict rate floors (clamped 0..100)

    Pure read — never raises. Observational only — never gates Opus,
    no caps (AGENTS.md #2/#12).
    """
    try:
        from .analytics.cash_redeployment_latency_skill import (
            build_cash_redeployment_latency_skill,
            DEFAULT_WINDOW_DAYS,
            DEFAULT_STALLED_CUTOFF_H,
            DEFAULT_FAST_MEDIAN_H,
            DEFAULT_STEADY_MEDIAN_H,
            DEFAULT_SLOW_MEDIAN_H,
            DEFAULT_HEALTHY_REDEPLOY_PCT,
            DEFAULT_STEADY_REDEPLOY_PCT,
            DEFAULT_DEGRADED_REDEPLOY_PCT,
        )

        def _qf(name, default, lo, hi):
            try:
                v = float(request.args.get(name, default))
            except (TypeError, ValueError):
                v = default
            return max(lo, min(hi, v))

        window_days = _qf("window_days", DEFAULT_WINDOW_DAYS, 1.0, 365.0)
        stalled_cutoff_hours = _qf(
            "stalled_cutoff_hours", DEFAULT_STALLED_CUTOFF_H, 1.0, 720.0,
        )
        fast_median_h = _qf("fast_median_h", DEFAULT_FAST_MEDIAN_H, 0.0, 720.0)
        steady_median_h = _qf("steady_median_h", DEFAULT_STEADY_MEDIAN_H, 0.0, 720.0)
        slow_median_h = _qf("slow_median_h", DEFAULT_SLOW_MEDIAN_H, 0.0, 720.0)
        healthy_redeploy_pct = _qf(
            "healthy_redeploy_pct", DEFAULT_HEALTHY_REDEPLOY_PCT, 0.0, 100.0,
        )
        steady_redeploy_pct = _qf(
            "steady_redeploy_pct", DEFAULT_STEADY_REDEPLOY_PCT, 0.0, 100.0,
        )
        degraded_redeploy_pct = _qf(
            "degraded_redeploy_pct", DEFAULT_DEGRADED_REDEPLOY_PCT, 0.0, 100.0,
        )

        store = get_store()
        # Pull enough history to cover the analysis window plus the
        # next-buy lookahead. 2000 covers ~6 months of mixed flow.
        trades = store.recent_trades(limit=2000) or []

        return jsonify(build_cash_redeployment_latency_skill(
            trades,
            window_days=window_days,
            stalled_cutoff_hours=stalled_cutoff_hours,
            fast_median_h=fast_median_h,
            steady_median_h=steady_median_h,
            slow_median_h=slow_median_h,
            healthy_redeploy_pct=healthy_redeploy_pct,
            steady_redeploy_pct=steady_redeploy_pct,
            degraded_redeploy_pct=degraded_redeploy_pct,
        ))
    except Exception as e:
        return jsonify({
            "verdict": "ERROR",
            "headline": f"error: {e}",
            "error": str(e),
            "stats": {},
            "thresholds": {},
            "pairs": [],
        }), 500


@app.route("/api/decision-vapor-skill")
def decision_vapor_skill_api():
    """Decision-vapor skill — for FILLED decisions, does the reasoning
    cite specifics (numeric figure + named catalyst + explicit ticker)
    or read as generic vapor?

    Verdict matrix:

      * ``SPECIFIC`` — specific% ≥ 50 AND vapor% < 15.
      * ``MIXED`` — between SPECIFIC and VAPOR_DECISIONS.
      * ``VAPOR_DECISIONS`` — vapor% ≥ 35.
      * ``NO_DATA`` — fewer than 5 FILLED decisions in window.

    Query params (clamped):
      ``window_hours`` — analysis window, 1..720 (default 168 = 7d)
      ``vapor_pct_floor`` — vapor% above this triggers VAPOR_DECISIONS,
        0..100 (default 35)
      ``vapor_pct_ceil`` — for SPECIFIC, vapor% must be below this,
        0..100 (default 15)
      ``specific_pct_floor`` — specific% required for SPECIFIC,
        0..100 (default 50)

    Pure read — never raises. Observational only — never gates Opus,
    no caps (AGENTS.md #2/#12).
    """
    try:
        from .analytics.decision_vapor_skill import (
            build_decision_vapor_skill,
            DEFAULT_WINDOW_HOURS,
            DEFAULT_VAPOR_PCT_FLOOR,
            DEFAULT_VAPOR_PCT_CEIL,
            DEFAULT_SPECIFIC_PCT_FLOOR,
        )

        def _qf(name, default, lo, hi):
            try:
                v = float(request.args.get(name, default))
            except (TypeError, ValueError):
                v = default
            return max(lo, min(hi, v))

        window_hours = _qf("window_hours", DEFAULT_WINDOW_HOURS, 1.0, 720.0)
        vapor_pct_floor = _qf(
            "vapor_pct_floor", DEFAULT_VAPOR_PCT_FLOOR, 0.0, 100.0,
        )
        vapor_pct_ceil = _qf(
            "vapor_pct_ceil", DEFAULT_VAPOR_PCT_CEIL, 0.0, 100.0,
        )
        specific_pct_floor = _qf(
            "specific_pct_floor", DEFAULT_SPECIFIC_PCT_FLOOR, 0.0, 100.0,
        )

        store = get_store()
        # Window is hours but recent_decisions is row-limited; 500 covers
        # ~21 days at the 1h closed cadence (worst case ~7-10 days at the
        # 60s open cadence). The window filter prunes anything older.
        decisions = store.recent_decisions(limit=500) or []

        # Watchlist injection — when available, sharpens ticker detection.
        wl = None
        try:
            from .strategy import WATCHLIST  # type: ignore
            wl = list(WATCHLIST)
        except Exception:
            wl = None

        return jsonify(build_decision_vapor_skill(
            decisions,
            watchlist=wl,
            window_hours=window_hours,
            vapor_pct_floor=vapor_pct_floor,
            vapor_pct_ceil=vapor_pct_ceil,
            specific_pct_floor=specific_pct_floor,
        ))
    except Exception as e:
        return jsonify({
            "verdict": "ERROR",
            "headline": f"error: {e}",
            "error": str(e),
            "stats": {},
            "thresholds": {},
            "samples": [],
        }), 500


@app.route("/api/cost-basis-ladder")
def cost_basis_ladder_api():
    """Cost-basis ladder — per-open-position FIFO lot reconstruction
    with per-lot P&L at the current mark. Surfaces the lot-level
    dispersion that ``positions.avg_cost`` flattens away.

    Verdict matrix (aggregate over open book):

      * ``HARVESTABLE_LOTS`` — ≥1 lot ≥ harvest_pct_floor in profit.
      * ``UNDERWATER_BOOK`` — every lot underwater.
      * ``MIXED_BOOK`` — green and red lots present but none clear
        the harvest floor.
      * ``NO_DATA`` — no open positions or no reconstructable lots.

    Per-position verdict:

      * ``LADDER_ALL_GREEN`` / ``LADDER_ALL_RED`` — every lot on
        the same side of the mark.
      * ``LADDER_WIDE`` — lot P&L spread ≥ wide_spread_pct.
      * ``LADDER_STACKED`` — multi-lot ladder tightly clustered.
      * ``LADDER_SINGLE_LOT`` — fresh position, no averaging.
      * ``NO_LOTS`` — position present but no reconstructable BUYs.

    Query params (clamped):
      ``wide_spread_pct`` — LADDER_WIDE threshold (pp), 0..100
        (default 5)
      ``harvest_pct_floor`` — HARVESTABLE per-lot floor (pp), 0..100
        (default 3)

    Pure read — never raises. Observational only — never gates Opus,
    no caps (AGENTS.md #2/#12).
    """
    try:
        from .analytics.cost_basis_ladder import (
            build_cost_basis_ladder,
            DEFAULT_WIDE_SPREAD_PCT,
            DEFAULT_HARVEST_PCT_FLOOR,
        )

        def _qf(name, default, lo, hi):
            try:
                v = float(request.args.get(name, default))
            except (TypeError, ValueError):
                v = default
            return max(lo, min(hi, v))

        wide_spread_pct = _qf(
            "wide_spread_pct", DEFAULT_WIDE_SPREAD_PCT, 0.0, 100.0,
        )
        harvest_pct_floor = _qf(
            "harvest_pct_floor", DEFAULT_HARVEST_PCT_FLOOR, 0.0, 100.0,
        )

        store = get_store()
        positions = store.open_positions() or []
        # 2000 trades covers ~6 months of mixed live flow; the FIFO walk
        # only consumes the trades that match a held position's key.
        trades = store.recent_trades(limit=2000) or []

        return jsonify(build_cost_basis_ladder(
            positions,
            trades,
            wide_spread_pct=wide_spread_pct,
            harvest_pct_floor=harvest_pct_floor,
        ))
    except Exception as e:
        return jsonify({
            "verdict": "ERROR",
            "headline": f"error: {e}",
            "error": str(e),
            "positions": [],
            "harvestable": [],
            "thresholds": {},
        }), 500


@app.route("/api/open-lot-aging")
def open_lot_aging_api():
    """Per-open-lot age + bucket — the time dimension
    ``cost_basis_ladder`` deliberately leaves out.

    Each FIFO-reconstructed lot is bucketed FRESH (<1d) / NORMAL
    (<7d) / MATURE (<30d) / STALE (≥30d). STALE lots additionally
    earn a STALE_RED / STALE_GREEN / STALE_FLAT tag based on per-lot
    P&L sign, flagging the two disposition-pathology quadrants — old
    underwater (overdue cut, "averaging-down anchor") vs old green
    (overdue trim or genuine hold; either way RE-affirm consciously).
    Per-position verdict rolls up the lot verdicts; aggregate book
    verdict (FRESH_BOOK / NORMAL_BOOK / AGING_BOOK / STALE_BOOK) is
    driven by the share of open-lot dollars that are stale or
    mature-or-worse.

    Reuses ``cost_basis_ladder``'s ``_reconstruct_lots`` primitive
    (SSOT — the two builders see byte-identical lots). The
    ``attention`` array lists only the verdicts that warrant a
    different next-action, oldest-first.

    Pure read — never raises. Observational only — never gates Opus,
    adds no caps (AGENTS.md #2/#12)."""
    try:
        from .analytics.open_lot_aging import build_open_lot_aging
        store = get_store()
        positions = store.open_positions() or []
        # 2000 trades covers ~6 months of mixed live flow — same cap
        # the sibling cost-basis-ladder endpoint uses.
        trades = store.recent_trades(limit=2000) or []
        return jsonify(build_open_lot_aging(positions, trades))
    except Exception as e:
        return jsonify({
            "state": "ERROR",
            "headline": f"error: {e}",
            "error": str(e),
            "positions": [],
            "attention": [],
        }), 500


@app.route("/api/catalyst-expiry-skill")
def catalyst_expiry_skill_api():
    """Catalyst-expiry skill — for each open position, classify the
    entry-rationale catalyst class and flag positions whose dated
    catalyst has aged past its expiry window.

    Verdict matrix (aggregate):

      * ``ZOMBIE_HOLDINGS`` — ≥1 position is ZOMBIE.
      * ``ALL_FRESH`` — every position on a fresh dated catalyst
        (<fresh_days_ceil).
      * ``STRUCTURAL_BOOK`` — every position structural / no time
        marker.
      * ``MIXED_BOOK`` — fresh + structural present, no zombies.
      * ``NO_DATA`` — no open positions.

    Per-position verdict:

      * ``ZOMBIE`` — dated catalyst (EARNINGS/MACRO/PRODUCT/REGULATORY)
        + time marker + days_held ≥ zombie_days_floor.
      * ``FRESH_CATALYST`` — dated catalyst + days_held <
        fresh_days_ceil.
      * ``STRUCTURAL`` — TECHNICAL / CORPORATE thesis, or no time
        marker.
      * ``UNCATEGORIZED`` — no catalyst keyword family matched.
      * ``NO_REASON`` — entry trade has no parseable reason text.

    Query params (clamped):
      ``zombie_days_floor`` — dated catalysts older than this flag as
        ZOMBIE (days), 0..365 (default 3)
      ``fresh_days_ceil`` — dated catalysts younger than this stay
        FRESH (days), 0..365 (default 2)

    Pure read — never raises. Observational only — never gates Opus,
    no caps (AGENTS.md #2/#12).
    """
    try:
        from .analytics.catalyst_expiry_skill import (
            build_catalyst_expiry_skill,
            DEFAULT_ZOMBIE_DAYS_FLOOR,
            DEFAULT_FRESH_DAYS_CEIL,
        )

        def _qf(name, default, lo, hi):
            try:
                v = float(request.args.get(name, default))
            except (TypeError, ValueError):
                v = default
            return max(lo, min(hi, v))

        zombie_days_floor = _qf(
            "zombie_days_floor", DEFAULT_ZOMBIE_DAYS_FLOOR, 0.0, 365.0,
        )
        fresh_days_ceil = _qf(
            "fresh_days_ceil", DEFAULT_FRESH_DAYS_CEIL, 0.0, 365.0,
        )

        store = get_store()
        positions = store.open_positions() or []
        trades = store.recent_trades(limit=2000) or []

        return jsonify(build_catalyst_expiry_skill(
            positions,
            trades,
            zombie_days_floor=zombie_days_floor,
            fresh_days_ceil=fresh_days_ceil,
        ))
    except Exception as e:
        return jsonify({
            "verdict": "ERROR",
            "headline": f"error: {e}",
            "error": str(e),
            "positions": [],
            "counts": {},
            "thresholds": {},
        }), 500


@app.route("/api/hourly-pnl-fingerprint")
def hourly_pnl_fingerprint_api():
    """Hourly P&L fingerprint — bucket equity-curve cycle deltas by
    NY-local hour-of-day, surface the alpha-vs-SPY spread across the
    trading session.

    Verdict matrix:

      * ``INSUFFICIENT_DATA`` — qualifying samples < min_total_samples.
      * ``NO_SPY_DATA`` — samples present but no SPY-anchored pair.
      * ``MORNING_EDGE`` — best alpha hour ∈ [9,12) AND spread ≥ floor.
      * ``MIDDAY_EDGE`` — best alpha hour ∈ [12,14) AND spread ≥ floor.
      * ``AFTERNOON_EDGE`` — best alpha hour ∈ [14,17) AND spread ≥ floor.
      * ``OFF_HOURS_EDGE`` — best alpha hour outside [9,17).
      * ``FLAT_CLOCK`` — alpha spread < floor across qualifying hours.

    Query params (clamped):
      ``limit`` — equity-curve rows to consider, 50..50000 (default 5000).
      ``min_total_samples`` — verdict floor, 1..100000 (default 60).
      ``alpha_spread_pp`` — EDGE/FLAT threshold (pp), 0..100 (default 0.5).

    Pure read — never raises. Observational only — never gates Opus,
    no caps (AGENTS.md #2/#12).
    """
    try:
        from .analytics.hourly_pnl_fingerprint import (
            build_hourly_pnl_fingerprint,
            DEFAULT_MIN_TOTAL_SAMPLES,
            DEFAULT_ALPHA_SPREAD_PP,
            DEFAULT_MIN_BUCKET_ALPHA_SAMPLES,
        )

        def _qf(name, default, lo, hi):
            try:
                v = float(request.args.get(name, default))
            except (TypeError, ValueError):
                v = default
            return max(lo, min(hi, v))

        def _qi(name, default, lo, hi):
            try:
                v = int(request.args.get(name, default))
            except (TypeError, ValueError):
                v = default
            return max(lo, min(hi, v))

        limit = _qi("limit", 5000, 50, 50000)
        min_total_samples = _qi(
            "min_total_samples", DEFAULT_MIN_TOTAL_SAMPLES, 1, 100000,
        )
        alpha_spread_pp = _qf(
            "alpha_spread_pp", DEFAULT_ALPHA_SPREAD_PP, 0.0, 100.0,
        )
        min_bucket_alpha_samples = _qi(
            "min_bucket_alpha_samples", DEFAULT_MIN_BUCKET_ALPHA_SAMPLES,
            1, 100000,
        )

        store = get_store()
        rows = store.equity_curve(limit=limit) or []
        return jsonify(build_hourly_pnl_fingerprint(
            rows,
            min_total_samples=min_total_samples,
            alpha_spread_pp=alpha_spread_pp,
            min_bucket_alpha_samples=min_bucket_alpha_samples,
        ))
    except Exception as e:
        return jsonify({
            "verdict": "ERROR",
            "headline": f"error: {e}",
            "error": str(e),
            "buckets": [],
        }), 500


@app.route("/api/weekday-pnl-fingerprint")
def weekday_pnl_fingerprint_api():
    """Weekday P&L fingerprint — bucket equity-curve cycle deltas by
    NY-local weekday, surface the alpha-vs-SPY spread across the week.

    Verdict matrix:

      * ``INSUFFICIENT_DATA`` — qualifying samples < min_total_samples.
      * ``NO_SPY_DATA`` — samples present but no SPY-anchored pair.
      * ``WEEKDAY_EDGE`` — best alpha day ∈ Mon..Fri AND spread ≥ floor.
      * ``WEEKEND_EDGE`` — best alpha day ∈ Sat/Sun AND spread ≥ floor.
      * ``FLAT_WEEK`` — alpha spread < floor across qualifying weekdays.

    Query params (clamped):
      ``limit`` — equity-curve rows to consider, 50..50000 (default 5000).
      ``min_total_samples`` — verdict floor, 1..100000 (default 60).
      ``alpha_spread_pp`` — EDGE/FLAT threshold (pp), 0..100 (default 0.5).

    Pure read — never raises. Observational only — never gates Opus,
    no caps (AGENTS.md #2/#12).
    """
    try:
        from .analytics.weekday_pnl_fingerprint import (
            build_weekday_pnl_fingerprint,
            DEFAULT_MIN_TOTAL_SAMPLES,
            DEFAULT_ALPHA_SPREAD_PP,
            DEFAULT_MIN_BUCKET_ALPHA_SAMPLES,
        )

        def _qf(name, default, lo, hi):
            try:
                v = float(request.args.get(name, default))
            except (TypeError, ValueError):
                v = default
            return max(lo, min(hi, v))

        def _qi(name, default, lo, hi):
            try:
                v = int(request.args.get(name, default))
            except (TypeError, ValueError):
                v = default
            return max(lo, min(hi, v))

        limit = _qi("limit", 5000, 50, 50000)
        min_total_samples = _qi(
            "min_total_samples", DEFAULT_MIN_TOTAL_SAMPLES, 1, 100000,
        )
        alpha_spread_pp = _qf(
            "alpha_spread_pp", DEFAULT_ALPHA_SPREAD_PP, 0.0, 100.0,
        )
        min_bucket_alpha_samples = _qi(
            "min_bucket_alpha_samples", DEFAULT_MIN_BUCKET_ALPHA_SAMPLES,
            1, 100000,
        )

        store = get_store()
        rows = store.equity_curve(limit=limit) or []
        return jsonify(build_weekday_pnl_fingerprint(
            rows,
            min_total_samples=min_total_samples,
            alpha_spread_pp=alpha_spread_pp,
            min_bucket_alpha_samples=min_bucket_alpha_samples,
        ))
    except Exception as e:
        return jsonify({
            "verdict": "ERROR",
            "headline": f"error: {e}",
            "error": str(e),
            "buckets": [],
        }), 500


@app.route("/api/forced-hold-attribution")
def forced_hold_attribution_api():
    """Forced-vs-chosen attribution on every currently OPEN position.

    For each open position, partitions the decision cycles since it was
    opened into ``blind`` (NO_DECISION — Opus could not act) vs
    ``sighted`` (any real decision row) and emits per-position +
    aggregate verdicts: ``FORCED_HOLD`` / ``PARTIALLY_FORCED`` /
    ``MIXED`` / ``CHOSEN_HOLD``.

    Answers the trader's question: "Is my NVDA on the book because
    Opus is choosing to ride it, or because the box has been
    host-saturated all day and Opus literally can't issue a SELL?"

    Neighbour endpoints solve different problems:

    * ``/api/hold-discipline`` — disposition trap (held past empirical
      median losing hold), reads age, not agency.
    * ``/api/no-decision-recovery`` — wedge run-lengths across the
      WHOLE tape, does not attribute wedges to OPEN positions.
    * ``/api/today-action-tape`` — flat aggregate of today's cycles,
      not partitioned by which position was on the book.

    Pure builder; network in the endpoint. Advisory only — never gates
    Opus, never adds caps (AGENTS.md #2/#12 — the ``hold_discipline``
    endpoint precedent).
    """
    try:
        from .analytics.forced_hold_attribution import (
            build_forced_hold_attribution,
        )

        try:
            window = int(request.args.get("window") or 500)
        except (TypeError, ValueError):
            window = 500
        window = max(1, min(window, 5000))

        store = get_store()
        return jsonify(build_forced_hold_attribution(
            store.open_positions(),
            store.recent_decisions(window),
        ))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/frozen-mark-execution-skill")
def frozen_mark_execution_skill_api():
    """Frozen-mark execution skill — flags FILLED trade clusters that all
    executed at the EXACT same float price within a short window.

    When ``market.get_price`` falls through to a cached yfinance close
    (overnight, pre-market, weekend, brief upstream stall), every BUY
    or SELL Opus issues over the next few hours routes to the same
    bit-identical float. The book records what *looks* like alpha-
    attempt repositioning (BUY → SELL → BUY round-trips) but the cluster
    nets zero P&L per share because the price literally never moved.

    Live evidence (2026-05-23): 5 NVDA trades 2026-05-20T21:10 →
    2026-05-21T10:00 (a 13-hour overnight stretch) all filled at
    exactly ``$223.43499755859375`` — a BUY/BUY/SELL/BUY/BUY sequence
    that yielded zero realized P&L on the same-price round-trip.

    Verdict ladder (test-locked):

      * ``CLEAN`` — frozen_trade_pct < ``occasional_pct`` (default 5).
      * ``OCCASIONAL`` — between CLEAN and HEAVY.
      * ``FROZEN_MARK_HEAVY`` — ≥ ``heavy_pct`` (default 25) of
        in-window FILLS belonged to a frozen-mark cluster.
      * ``INSUFFICIENT_DATA`` — fewer than 5 classifiable trades.

    Query params (clamped):
      ``window_days`` — analysis window, 1..365 (default 30)
      ``cluster_min`` — min trades per cluster, 2..20 (default 2)
      ``cluster_span_hours`` — max wall-clock span across a cluster,
        0.5..168 (default 24)
      ``occasional_pct`` / ``heavy_pct`` — verdict thresholds, 0..100

    Distinct from neighbouring endpoints:
      * ``mark_integrity`` — % of DISPLAYED book held at a stale mark
        (snapshot, never reads ``trades``).
      * ``rebuy_regret`` / ``reentry_velocity`` — same-ticker SELL→BUY
        $ regret / timing, no price-equality filter.
      * ``churn`` — overall trade frequency without the price-discovery
        context this skill exists to surface.

    Pure read — never raises. Observational only — never gates Opus,
    no caps (AGENTS.md #2/#12).
    """
    try:
        from .analytics.frozen_mark_execution_skill import (
            build_frozen_mark_execution_skill,
            DEFAULT_WINDOW_DAYS,
            DEFAULT_CLUSTER_MIN,
            DEFAULT_CLUSTER_SPAN_HOURS,
            DEFAULT_OCCASIONAL_PCT,
            DEFAULT_HEAVY_PCT,
        )

        def _qf(name, default, lo, hi):
            try:
                v = float(request.args.get(name, default))
            except (TypeError, ValueError):
                v = default
            return max(lo, min(hi, v))

        def _qi(name, default, lo, hi):
            try:
                v = int(request.args.get(name, default))
            except (TypeError, ValueError):
                v = default
            return max(lo, min(hi, v))

        window_days = _qf("window_days", DEFAULT_WINDOW_DAYS, 1.0, 365.0)
        cluster_min = _qi("cluster_min", DEFAULT_CLUSTER_MIN, 2, 20)
        cluster_span_hours = _qf(
            "cluster_span_hours", DEFAULT_CLUSTER_SPAN_HOURS, 0.5, 168.0,
        )
        occasional_pct = _qf("occasional_pct", DEFAULT_OCCASIONAL_PCT, 0.0, 100.0)
        heavy_pct = _qf("heavy_pct", DEFAULT_HEAVY_PCT, 0.0, 100.0)

        store = get_store()
        # 2000 covers ~6 months of mixed flow at current ~10 trades/day
        # ceiling; the in-builder window_days filter will narrow further.
        trades = store.recent_trades(limit=2000) or []

        return jsonify(build_frozen_mark_execution_skill(
            trades,
            window_days=window_days,
            cluster_min=cluster_min,
            cluster_span_hours=cluster_span_hours,
            occasional_pct=occasional_pct,
            heavy_pct=heavy_pct,
        ))
    except Exception as e:
        return jsonify({"error": str(e), "verdict": "ERROR"}), 500


@app.route("/api/closed-market-fill-skill")
def closed_market_fill_skill_api():
    """Closed-market fill skill — what share of FILLED trades fired while
    NYSE was CLOSED (overnight, weekend, half-day after-hours, holiday)?

    Why this isn't redundant. ``market.get_price`` falls through to the
    last-known yfinance close when NYSE is closed. A BUY fired at 02:00
    ET on a Wednesday fills against Tuesday's 16:00 close — there is no
    live bid-ask, no liquidity test, and no real price discovery. Pair
    this with ``frozen_mark_execution_skill`` (which catches the
    *effect* — clusters at the SAME float price) and the trade tape's
    economic illusion is fully diagnosed.

    Live evidence (2026-05-23): 4 of 5 NVDA BUYs in the rolling tape
    fired while ``market_open=0``, all at the same yfinance close.

    Verdict ladder:

      * ``SESSION_ALIGNED`` — closed_pct ≤ ``aligned_pct`` (default 25).
      * ``BALANCED`` — between aligned and ``after_hours_pct`` (50).
      * ``AFTER_HOURS_HEAVY`` — between after_hours and
        ``dominated_pct`` (75).
      * ``OVERNIGHT_DOMINATED`` — closed_pct ≥ dominated_pct.
      * ``INSUFFICIENT_DATA`` — < 5 FILLs in the window.

    Distinct from:
      * ``decision_clock`` — distribution of DECISION cycles by
        hour-of-day, including NO_DECISIONs; doesn't filter to FILLs
        or consult ``is_market_open``.
      * ``decision_weekday`` — by weekday, same surface.

    Query params (clamped):
      ``window_days`` — 1..365 (default 30)
      ``aligned_pct`` / ``after_hours_pct`` / ``dominated_pct`` —
        verdict thresholds, 0..100

    Pure read — never raises. Observational only — never gates Opus,
    no caps (AGENTS.md #2/#12).
    """
    try:
        from .analytics.closed_market_fill_skill import (
            build_closed_market_fill_skill,
            DEFAULT_WINDOW_DAYS,
            DEFAULT_ALIGNED_PCT,
            DEFAULT_AFTER_HOURS_PCT,
            DEFAULT_DOMINATED_PCT,
        )

        def _qf(name, default, lo, hi):
            try:
                v = float(request.args.get(name, default))
            except (TypeError, ValueError):
                v = default
            return max(lo, min(hi, v))

        window_days = _qf("window_days", DEFAULT_WINDOW_DAYS, 1.0, 365.0)
        aligned_pct = _qf("aligned_pct", DEFAULT_ALIGNED_PCT, 0.0, 100.0)
        after_hours_pct = _qf(
            "after_hours_pct", DEFAULT_AFTER_HOURS_PCT, 0.0, 100.0,
        )
        dominated_pct = _qf("dominated_pct", DEFAULT_DOMINATED_PCT, 0.0, 100.0)

        store = get_store()
        trades = store.recent_trades(limit=2000) or []

        return jsonify(build_closed_market_fill_skill(
            trades,
            window_days=window_days,
            aligned_pct=aligned_pct,
            after_hours_pct=after_hours_pct,
            dominated_pct=dominated_pct,
        ))
    except Exception as e:
        return jsonify({"error": str(e), "verdict": "ERROR"}), 500


if __name__ == "__main__":
    run()

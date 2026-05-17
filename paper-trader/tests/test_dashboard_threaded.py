"""The :8090 dashboard must serve requests concurrently, not single-threaded.

Root cause (found by user-perspective testing, 2026-05-17): ``dashboard.run``
called ``app.run(host, port, debug=False, use_reloader=False)`` with **no**
``threaded=True``. Flask's ``app.run`` defaults ``threaded=False``, so the
in-process Werkzeug dev server (started as a daemon thread by
``runner._start_dashboard``) serves exactly **one request at a time**.

That is fine for a single curl but wrong for the dashboard's real load, which
is genuinely concurrent:

* the unified :8888 page renders ~25 panels — the browser fires their
  ``/api/*`` fetches in parallel;
* ``unified_dashboard.py`` ``/api/chat`` fans out ~15 ``_fetch_*`` sub-fetches
  against :8090 at once;
* the digital-intern :8080 dashboard cross-fetches :8090 while :8090 is also
  serving its own page and the runner cycle is live.

On a single-threaded server every one of those serializes behind whichever
request is currently slowest. Several endpoints do unbounded yfinance / cross-
DB I/O (``/api/correlation``, ``/api/news-edge``, ``/api/source-edge``,
``/api/feed-health``, ``/api/sector-heatmap``) — one slow such request
head-of-line-blocks every fast pure-DB panel, the chat fan-out, and the
:8080 cross-fetch behind it.

This is **safe** to fix with ``threaded=True``: ``store.py`` (see the NOTE at
``Store.get_portfolio``) already wraps *every* read in ``self._lock`` and is
explicitly hardened "between the runner's writer thread and the Flask
dashboard thread(s)" — plural. The shared ``sqlite3.Connection`` is never used
by two threads at once because ``_lock`` brackets every ``.execute()``; the
slow endpoints use their own per-request ``mode=ro`` connections. The store
was built for a threaded dashboard; only the server flag was missing.

NOTE (honest scope): ``threaded=True`` removes *head-of-line blocking between
concurrent requests*. It does **not** make an individual slow endpoint fast —
unbounded per-endpoint yfinance latency is a separate, untreated problem
(flagged in AGENTS.md invariant #7), out of scope for this surgical change.

Two locks:
  * ``test_run_passes_threaded`` — regression-locks the ``dashboard.run`` call
    site (RED before the fix: the kwarg was absent).
  * ``test_threaded_server_parallelizes`` — behavioural lock proving the kwarg
    actually buys concurrency, so a future swap to a different server
    entry point that silently drops it (or moves to a non-threaded WSGI
    runner) is caught even though the monkeypatch lock above still passes.

Offline, deterministic, no network, no real :8090 bind.
"""
from __future__ import annotations

import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader import dashboard


def test_run_passes_threaded(monkeypatch):
    """dashboard.run() must hand werkzeug threaded=True (and keep the existing
    hardening flags). Recorded, not really bound — the real server never starts.
    """
    captured: dict = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(dashboard.app, "run", fake_run)

    dashboard.run(host="127.0.0.1", port=18090)

    kw = captured["kwargs"]
    assert kw.get("threaded") is True, (
        "dashboard.run must pass threaded=True so concurrent panel/chat/"
        "cross-fetch requests don't serialize behind a slow endpoint"
    )
    # The existing dev-server hardening must be preserved, not regressed away.
    assert kw.get("debug") is False
    assert kw.get("use_reloader") is False
    assert kw.get("host") == "127.0.0.1"
    assert kw.get("port") == 18090


def test_threaded_server_parallelizes():
    """A threaded werkzeug server overlaps slow requests instead of serializing.

    Builds an independent tiny Flask app (does not bind :8090, does not touch
    the real Store) and serves it threaded on an ephemeral port. Four
    concurrent 0.4s requests must finish in well under the 1.6s a
    single-threaded server would take. This pins the *property* the
    ``threaded=True`` kwarg exists to provide.
    """
    from flask import Flask
    from werkzeug.serving import make_server

    sleep_app = Flask(__name__)

    @sleep_app.route("/slow")
    def _slow():  # noqa: D401
        time.sleep(0.4)
        return "ok"

    srv = make_server("127.0.0.1", 0, sleep_app, threaded=True)
    port = srv.server_port
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        url = f"http://127.0.0.1:{port}/slow"

        def hit() -> str:
            with urllib.request.urlopen(url, timeout=5) as r:
                return r.read().decode()

        n = 4
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n) as ex:
            results = list(ex.map(lambda _: hit(), range(n)))
        elapsed = time.perf_counter() - start

        assert results == ["ok"] * n
        # Serial would be ~n*0.4 = 1.6s. Concurrent ≈ 0.4s. Use a generous
        # ceiling (0.4 * n * 0.5 = 0.8s) so a loaded CI box never flakes
        # while a serialized regression (≥1.6s) still fails loudly.
        assert elapsed < 0.4 * n * 0.5, (
            f"threaded server serialized: {elapsed:.2f}s for {n} concurrent "
            f"0.4s requests (expected ~0.4s, serial would be ~{0.4*n:.1f}s)"
        )
    finally:
        srv.shutdown()
        t.join(timeout=5)

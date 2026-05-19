"""``/api/healthz`` — lightweight liveness probe.

The endpoint reads two columns out of the SHARED ``Store`` connection (one
COUNT on positions, one MAX on decisions). ``Store.__init__`` opens that
connection with ``check_same_thread=False`` and serialises all writes on
``self._lock``. ``store.py:134`` documents that **every read on the shared
connection must hold ``self._lock``** — an unlocked read whose
``execute()`` interleaves with a writer raises ``sqlite3.InterfaceError:
bad parameter or other API misuse`` or hands back a corrupted/None row.

Before this fix ``healthz_api`` ran the two ``execute()`` calls *without*
the lock. The pin below uses a counting / asserting wrapper around
``store._lock`` so a regression — anyone re-introducing the bare
``conn.execute`` shape (current or future) — flips a test red rather than
landing as a silent intermittent 500 on the production /api/healthz
endpoint the watchdog polls.

The race itself is hard to provoke deterministically (microsecond
interleave on a few hundred-ns SQL operation), so the test asserts the
**discipline** (lock acquired before each shared read) rather than
trying to win the race in CI. A second concurrency-stress test exercises
the actual interleave path: 100 ``healthz`` hits with a concurrent writer
must produce zero exceptions and a non-negative open-positions count
regardless of timing.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Boot a Flask test client wired to an isolated ``paper_trader.db``.

    Each test gets a fresh store + DB so the autouse ``_isolate_data_dir``
    fixture (which redirects ML/backtest paths) does not need to know about
    paper_trader.store. The reset clears any leftover global singleton so
    the dashboard sees the test's store, not whatever the previous test
    left behind.
    """
    import paper_trader.store as store
    monkeypatch.setattr(
        store, "DB_PATH", tmp_path / "paper_trader.db"
    )
    monkeypatch.setattr(store, "_singleton", None)

    from paper_trader import dashboard
    dashboard.app.config["TESTING"] = True
    with dashboard.app.test_client() as c:
        yield c, dashboard, store


class _CountingLock:
    """Wraps a ``threading.Lock`` and records acquisition count + the
    timestamp of every ``__enter__``. Drop-in replacement for ``_lock``
    in tests that must prove a read held the lock."""

    def __init__(self, real_lock: threading.Lock):
        self._real = real_lock
        self.acquire_count = 0
        self.acquired = False

    def __enter__(self):
        self._real.acquire()
        self.acquire_count += 1
        self.acquired = True
        return self

    def __exit__(self, exc_type, exc, tb):
        self.acquired = False
        self._real.release()
        return False

    # ``store.py`` may also call ``acquire()`` / ``release()`` directly on
    # ``_lock`` from its own methods — forward those so the wrapped store
    # still works for tests that mix dashboard reads and store writes.
    def acquire(self, *a, **k):
        return self._real.acquire(*a, **k)

    def release(self):
        return self._real.release()


def test_healthz_acquires_store_lock_before_shared_reads(client, monkeypatch):
    """Pin the documented invariant: every ``st.conn.execute`` call inside
    ``/api/healthz`` must run under ``st._lock``.

    Replaces the store's lock with a counting wrapper, fires one request,
    asserts the lock was acquired at least once before the JSON body comes
    back. A regression that re-introduces the bare ``conn.execute`` shape
    leaves ``acquire_count`` at 0.
    """
    c, dashboard, store = client
    st = store.get_store()
    counter = _CountingLock(st._lock)
    monkeypatch.setattr(st, "_lock", counter)
    # Also re-route healthz's `store._lock` reference to the counter — the
    # endpoint reads `st._lock` after `get_store()`, so the monkeypatch on
    # the singleton is sufficient (no module-level lock import to patch).
    resp = c.get("/api/healthz")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body is not None
    assert body["service"] == "paper_trader"
    # Two shared-conn reads — COUNT(*) and MAX(timestamp) — should both
    # run inside the critical section. The fix issues ONE `with st._lock`
    # block wrapping both, so acquire_count == 1 (not 2). The contract is
    # "at least one acquisition before the reads"; pin >= 1 so a future
    # split into two locked blocks doesn't flip this red for free.
    assert counter.acquire_count >= 1, (
        "healthz_api ran a shared-conn read without acquiring st._lock — "
        "violates store.py:134 invariant; will 500 intermittently under "
        "concurrent writer load."
    )


def test_healthz_returns_zero_positions_on_empty_book(client):
    """A fresh paper-trader.db has the seeded portfolio row but no open
    positions and no decisions. The endpoint must report 0 / None
    explicitly — not crash, not return a partial object."""
    c, _dashboard, _store = client
    body = c.get("/api/healthz").get_json()
    assert body["open_positions"] == 0
    assert body["last_decision_age_s"] is None
    assert body["ok"] is True


def test_healthz_reports_open_positions_count(client):
    """Add two open lots — one stock, one option — and confirm the count
    flows through. Confirms the SQL filter (closed_at IS NULL AND
    qty != 0) catches both shapes, not just the stock case."""
    c, _dashboard, store_mod = client
    st = store_mod.get_store()
    st.upsert_position("NVDA", "stock", 2.0, 222.5)
    st.upsert_position("AMD", "call", 1.0, 4.5,
                        expiry="2026-12-19", strike=150.0)
    body = c.get("/api/healthz").get_json()
    assert body["open_positions"] == 2


def test_healthz_skips_closed_positions(client):
    """A position that's been fully closed (qty=0, closed_at set) must
    not inflate the count — operator reads this to know what's live."""
    c, _dashboard, store_mod = client
    st = store_mod.get_store()
    st.upsert_position("MU", "stock", 1.0, 100.0)
    # Close it: ``upsert_position`` flips to closed when net qty hits 0.
    st.upsert_position("MU", "stock", -1.0, 105.0)
    body = c.get("/api/healthz").get_json()
    assert body["open_positions"] == 0


def test_healthz_reports_last_decision_age(client):
    """A recorded decision sets last_decision_age_s to a non-negative
    float roughly equal to "now - timestamp". Round to seconds so we
    don't pin on µs noise; the bound proves the field reflects reality."""
    c, _dashboard, store_mod = client
    st = store_mod.get_store()
    st.record_decision(
        market_open=True,
        signal_count=1,
        action_taken="HOLD CASH → HOLD",
        reasoning="test",
        portfolio_value=1000.0,
        cash=1000.0,
    )
    body = c.get("/api/healthz").get_json()
    assert body["last_decision_age_s"] is not None
    assert 0.0 <= body["last_decision_age_s"] < 60.0


def test_healthz_survives_concurrent_writes(client):
    """Real interleave check: 100 ``healthz`` hits while a background
    writer slams ``record_equity_point``. With the lock fix, zero
    exceptions and every response carries a non-negative open_positions
    int. Without the lock (the bug this test pins against), the writer's
    INSERT during a reader's ``execute()`` either 500s the endpoint or
    hands back NULL / corrupt rows that fail the int-coerce in the route.

    The test does not assert the race FIRES — that requires microsecond
    luck on every CI run. It asserts the lock discipline holds up: zero
    InterfaceError, zero ValueError-on-int(NULL), zero 500s. A regression
    that strips the lock will lose one of these even on a slow CI box
    once the iteration count is high enough.
    """
    c, _dashboard, store_mod = client
    st = store_mod.get_store()
    stop = threading.Event()

    def writer():
        # Hot writer loop — INSERT a row, advance the portfolio. Without
        # the lock around the dashboard read, the SQLite InterfaceError
        # path fires somewhere in the 100 iterations.
        i = 0
        while not stop.is_set():
            try:
                st.record_equity_point(1000.0 + i, 1000.0 - i, 100.0 + i)
                st.update_portfolio(1000.0 - i, 1000.0 + i, [])
            except Exception:
                # The writer takes its own lock — it's the READER (the
                # endpoint) we're stressing.
                pass
            i += 1

    t = threading.Thread(target=writer, daemon=True)
    t.start()
    try:
        # Hit the endpoint 100 times back-to-back. Every response must
        # parse to a JSON dict with `open_positions` as a non-negative int.
        for _ in range(100):
            resp = c.get("/api/healthz")
            assert resp.status_code == 200, (
                f"healthz 500 under concurrent writer — likely a missing "
                f"lock on a shared-conn read. Body: {resp.get_data(as_text=True)!r}"
            )
            body = resp.get_json()
            assert body is not None
            n = body.get("open_positions")
            assert isinstance(n, int) and n >= 0, (
                f"open_positions came back {n!r} — a corrupted row from "
                f"an unlocked read against a concurrent writer."
            )
    finally:
        stop.set()
        t.join(timeout=2)

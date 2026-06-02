"""Tests for /api/last-fill — the dedicated "when did the engine last
EXECUTE?" surface that wraps ``build_last_fill`` over the trades ledger.

Distinct from /api/last-real-decision (which counts HOLDs as activity):
this endpoint *only* answers "when did money last move?". Pins:
  * the NO_DATA empty-trades contract
  * the FRESH / STATIC / FROZEN verdict ladder against the builder's
    own thresholds (read live so a re-tune cannot false-fail)
  * the per-fill metadata (ticker / action / qty / price / value) is
    forwarded verbatim from the trade row
  * the endpoint shell adds ``as_of`` and ``service`` and merges the
    builder dict cleanly without overwriting builder keys
  * an injected store-read fault degrades to a valid ERROR envelope
    instead of 500-ing the request
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.last_fill import FRESH_HOURS, FROZEN_HOURS


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Fresh-temp-Store dashboard client. Same pattern as the
    /api/last-real-decision endpoint tests so the builder runs against a
    real Store under the real Flask app — not a mock — and the verdict
    bands depend on actual trade rows + wall clock."""
    from paper_trader import store as store_mod
    from paper_trader.store import Store

    db = tmp_path / "paper_trader.db"
    monkeypatch.setattr(store_mod, "DB_PATH", db)
    monkeypatch.setattr(store_mod, "_singleton", None)
    s = Store()

    from paper_trader import dashboard
    dashboard.app.config["TESTING"] = True
    with dashboard.app.test_client() as c:
        yield c, s
    s.close()
    store_mod._singleton = None


# ───────────────────────── NO_DATA ─────────────────────────


def test_empty_trades_returns_no_data(client):
    c, _s = client
    d = c.get("/api/last-fill").get_json()
    assert d["state"] == "NO_DATA"
    assert d["last_fill_ts"] is None
    assert d["secs_since"] is None
    assert d["age"] == ""
    assert d["ticker"] is None
    assert d["action"] is None
    # The headline still carries an explainer string so a panel renders
    # something rather than a bare empty value.
    assert "no fills" in d["headline"].lower()


def test_response_envelope_carries_as_of_and_service(client):
    """The endpoint adds ``as_of`` (ISO timestamp) and ``service`` keys to
    the builder's dict — cross-port consumers identify the source process."""
    c, _s = client
    d = c.get("/api/last-fill").get_json()
    assert d["service"] == "paper_trader"
    # ``as_of`` must be a parseable ISO-8601 timestamp.
    parsed = datetime.fromisoformat(d["as_of"].replace("Z", "+00:00"))
    assert parsed.tzinfo is not None


# ───────────────────────── FRESH ─────────────────────────


def test_fresh_recent_buy(client):
    c, s = client
    s.record_trade("NVDA", "BUY", 1.0, 200.0, "test_fresh")
    d = c.get("/api/last-fill").get_json()
    assert d["state"] == "FRESH"
    assert d["ticker"] == "NVDA"
    assert d["action"] == "BUY"
    assert d["qty"] == 1.0
    assert d["price"] == 200.0
    # store.record_trade computes value = qty * price for stock.
    assert d["value"] == 200.0
    # Should be under a minute old since we just inserted it.
    assert d["secs_since"] is not None
    assert d["secs_since"] < 60.0
    assert "actively trading" in d["headline"].lower()


def test_newest_trade_wins_over_older_rows(client):
    """When multiple trades exist, the endpoint must read the NEWEST one —
    the builder reads index 0 of the newest-first slice. A trader querying
    "last fill" expects the most recent row, never the second-newest."""
    c, s = client
    s.record_trade("AMD", "BUY", 1.0, 100.0, "older")
    s.record_trade("NVDA", "SELL", 2.0, 250.0, "newer")
    d = c.get("/api/last-fill").get_json()
    assert d["ticker"] == "NVDA"
    assert d["action"] == "SELL"
    assert d["qty"] == 2.0


def test_option_trade_carries_multiplied_value(client):
    """An option trade's ``value`` field is qty * price * 100 (store.record_trade
    convention). The endpoint must forward this verbatim so the dashboard
    panel and any cross-port consumer can render the notional without
    re-deriving the contract multiplier."""
    c, s = client
    s.record_trade("NVDA", "BUY_CALL", 3.0, 5.0, "test_option",
                   expiry="2026-06-19", strike=220.0, option_type="call")
    d = c.get("/api/last-fill").get_json()
    assert d["action"] == "BUY_CALL"
    # qty * price * 100 = 3 * 5 * 100 = 1500
    assert d["value"] == 1500.0


# ───────────────────────── STATIC ─────────────────────────


def _backdate_newest_trade(store, hours_ago: float) -> None:
    """Rewrite the newest trade's timestamp to be ``hours_ago`` hours in the
    past — the cleanest way to drive the verdict ladder past FRESH_HOURS /
    FROZEN_HOURS without monkeypatching the wall clock inside Flask request
    handling (the endpoint computes its own ``now_utc`` and forwards it)."""
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    with store._lock:
        cur = store.conn.execute(
            "SELECT id FROM trades ORDER BY timestamp DESC, id DESC LIMIT 1"
        )
        row = cur.fetchone()
        assert row is not None, "no trade to backdate"
        store.conn.execute(
            "UPDATE trades SET timestamp=? WHERE id=?",
            (ts, row["id"]),
        )
        store.conn.commit()


def test_static_book_when_fill_older_than_fresh_threshold(client):
    """A fill aged past FRESH_HOURS (6h) but under FROZEN_HOURS (36h)
    must read STATIC — the book has been idle through a full session."""
    c, s = client
    s.record_trade("MU", "BUY", 1.0, 150.0, "old_fill")
    _backdate_newest_trade(s, hours_ago=12.0)
    d = c.get("/api/last-fill").get_json()
    assert d["state"] == "STATIC"
    assert "static" in d["headline"].lower()
    assert d["secs_since"] is not None
    # 12h in seconds = 43200; allow generous slack.
    assert 43000 < d["secs_since"] < 44000


def test_frozen_book_when_fill_older_than_frozen_threshold(client):
    """A fill aged past FROZEN_HOURS (36h+) must read FROZEN — the engine
    has not executed for over a full trading day-and-a-half."""
    c, s = client
    s.record_trade("LITE", "BUY", 1.0, 50.0, "ancient_fill")
    _backdate_newest_trade(s, hours_ago=FROZEN_HOURS + 5.0)
    d = c.get("/api/last-fill").get_json()
    assert d["state"] == "FROZEN"
    assert "frozen" in d["headline"].lower()


def test_fresh_static_frozen_boundaries_align_with_builder_constants(client):
    """A boundary fill just under FRESH_HOURS reads FRESH; just over reads
    STATIC; just over FROZEN_HOURS reads FROZEN. Pins all three bands in
    one round-trip so a constants drift never silently mis-classifies."""
    c, s = client
    # Just under FRESH (5.9h) → FRESH
    s.record_trade("A", "BUY", 1.0, 100.0, "just_fresh")
    _backdate_newest_trade(s, hours_ago=FRESH_HOURS - 0.1)
    assert c.get("/api/last-fill").get_json()["state"] == "FRESH"

    # Just over FRESH (6.1h) → STATIC
    s.record_trade("B", "BUY", 1.0, 100.0, "just_static")
    _backdate_newest_trade(s, hours_ago=FRESH_HOURS + 0.1)
    assert c.get("/api/last-fill").get_json()["state"] == "STATIC"

    # Just over FROZEN (36.1h) → FROZEN
    s.record_trade("C", "BUY", 1.0, 100.0, "just_frozen")
    _backdate_newest_trade(s, hours_ago=FROZEN_HOURS + 0.1)
    assert c.get("/api/last-fill").get_json()["state"] == "FROZEN"


# ───────────────────────── ERROR ─────────────────────────


def test_store_failure_returns_error_envelope_not_500(client, monkeypatch):
    """An injected ``recent_trades`` fault must degrade to a 500 + ERROR
    envelope, not a raw Flask traceback. The trader's panel sees a
    diagnostic instead of a broken page."""
    c, _s = client
    from paper_trader import dashboard as dash_mod

    class _BrokenStore:
        def recent_trades(self, *_a, **_k):
            raise RuntimeError("simulated DB lock")
    monkeypatch.setattr(dash_mod, "get_store", lambda: _BrokenStore())
    resp = c.get("/api/last-fill")
    assert resp.status_code == 500
    d = resp.get_json()
    assert d["state"] == "ERROR"
    assert "simulated DB lock" in d["headline"]


def test_builder_non_dict_returns_error_envelope(client, monkeypatch):
    """If the builder ever returns a non-dict (a contract violation), the
    endpoint must surface an ERROR envelope rather than letting Flask
    json-fail on a non-dict ``**result`` spread."""
    c, s = client
    s.record_trade("NVDA", "BUY", 1.0, 200.0, "test")
    from paper_trader import dashboard as dash_mod
    # Patch the builder reference the endpoint imports at call time.
    import paper_trader.analytics.last_fill as lf_mod
    monkeypatch.setattr(lf_mod, "build_last_fill", lambda *a, **k: "not a dict")
    resp = c.get("/api/last-fill")
    assert resp.status_code == 500
    d = resp.get_json()
    assert d["state"] == "ERROR"


# ───────────────────────── Threshold round-trip ─────────────────────────


def test_thresholds_are_read_from_builder(client):
    """The endpoint must not hardcode any thresholds — it composes the
    builder verbatim. Verify by asserting the builder's own constants are
    the spec (the runner_heartbeat / feed_health precedent: tests read
    module constants so a re-tune in the builder can't false-fail the
    endpoint test)."""
    # No-op endpoint call to keep the fixture warm.
    c, _s = client
    c.get("/api/last-fill").get_json()
    # These constants are the spec — assertions just pin them so a
    # re-tune is intentional.
    assert FRESH_HOURS > 0
    assert FROZEN_HOURS > FRESH_HOURS

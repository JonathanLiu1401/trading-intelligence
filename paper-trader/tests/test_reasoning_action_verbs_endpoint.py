"""Endpoint tests for ``/api/reasoning-action-verbs``.

Pins the dashboard-side contract for the action-verb-consistency builder:

  * the route is reachable, returns JSON, never 500s on a bare flat book
    (``INSUFFICIENT`` is the documented zero-data state — never an error);
  * ``?limit=`` is parsed, clamped to ``[5, 500]``, gibberish degrades to
    the default 100 instead of raising;
  * a real CONSISTENT row makes ``state == "CLEAN"`` (after enough
    samples) and yields zero entries in ``mismatches``;
  * a single BULLISH-inside-HOLD row routes through the dashboard layer
    and surfaces ``BULLISH_INSIDE_HOLD`` in ``by_verdict`` plus a
    populated ``mismatches`` payload — proving the wiring forwards the
    builder's own verdict verbatim (single-source-of-truth contract;
    invariant #10).

Each test uses an isolated tmp ``paper_trader.db`` via the same store-
monkeypatch pattern ``tests/test_healthz_endpoint.py`` already documents
— never touches the production DB, never reaches the network.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _seed_decisions(store_mod, rows: list[dict]) -> None:
    """Insert raw decision rows directly so the test controls action_taken
    + reasoning + timestamp without rebuilding the strategy stack."""
    st = store_mod.get_store()
    for r in rows:
        with st._lock:
            st.conn.execute(
                "INSERT INTO decisions "
                "(timestamp, market_open, signal_count, action_taken, "
                "reasoning, portfolio_value, cash) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    r["timestamp"], 1, 0,
                    r["action_taken"], r["reasoning"], 1000.0, 1000.0,
                ),
            )
            st.conn.commit()


@pytest.fixture
def client(tmp_path, monkeypatch):
    import paper_trader.store as store
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "paper_trader.db")
    monkeypatch.setattr(store, "_singleton", None)
    from paper_trader import dashboard
    dashboard.app.config["TESTING"] = True
    with dashboard.app.test_client() as c:
        yield c, dashboard, store


class TestRoute:
    def test_empty_book_returns_insufficient_not_error(self, client):
        c, _, _ = client
        r = c.get("/api/reasoning-action-verbs")
        assert r.status_code == 200, r.data
        body = r.get_json()
        # No decisions at all → builder returns INSUFFICIENT (the documented
        # zero-data state). The route MUST NOT 500 on an empty book.
        assert body["state"] == "INSUFFICIENT"
        assert body["n_decisions"] == 0
        assert body["n_parsed"] == 0
        assert body["mismatches"] == []
        # window_limit echoed so the operator can confirm the slice scanned.
        assert body["window_limit"] == 100

    def test_limit_clamped_low(self, client):
        c, _, _ = client
        r = c.get("/api/reasoning-action-verbs?limit=1")
        assert r.status_code == 200
        assert r.get_json()["window_limit"] == 5

    def test_limit_clamped_high(self, client):
        c, _, _ = client
        r = c.get("/api/reasoning-action-verbs?limit=99999")
        assert r.status_code == 200
        assert r.get_json()["window_limit"] == 500

    def test_limit_garbage_falls_back_to_default(self, client):
        c, _, _ = client
        r = c.get("/api/reasoning-action-verbs?limit=banana")
        assert r.status_code == 200
        assert r.get_json()["window_limit"] == 100


class TestVerdictPassthrough:
    """The route MUST forward the builder's verdict / mismatch payload
    verbatim — no re-derivation in the dashboard layer (the single-source
    -of-truth precedent invariant #10 pins for benchmark / drawdown /
    realized-vs-unrealized). These tests prove the wire-through."""

    def test_consistent_hold_row_produces_zero_mismatches(self, client):
        c, _, store = client
        # 12 CONSISTENT HOLD rows (above MIN_SAMPLES_FOR_VERDICT=10).
        # Each reasoning verbalises HOLD intent — no BUY/SELL verbs.
        rows = []
        for i in range(12):
            rows.append({
                "timestamp": f"2026-05-23T1{i % 10}:00:00+00:00",
                "action_taken": "HOLD NVDA → HOLD",
                "reasoning": json.dumps({
                    "decision": {
                        "action": "HOLD",
                        "ticker": "NVDA",
                        "reasoning": (
                            "Waiting on earnings before committing capital. "
                            "Sitting tight while the tape digests guidance."
                        ),
                    }
                }),
            })
        _seed_decisions(store, rows)
        r = c.get("/api/reasoning-action-verbs?limit=50")
        body = r.get_json()
        # Enough samples → past INSUFFICIENT; zero mismatches → CLEAN.
        assert body["n_parsed"] == 12
        assert body["n_mismatched"] == 0
        assert body["state"] == "CLEAN"
        assert body["mismatches"] == []
        # The single-decision-leaning bucket should be CONSISTENT for all 12.
        assert body["by_verdict"].get("CONSISTENT") == 12

    def test_bullish_inside_hold_surfaces_in_mismatches(self, client):
        c, _, store = client
        # 9 CONSISTENT HOLDs + 1 mismatched row: structured HOLD whose prose
        # says "adding to NVDA on the dip" — the canonical
        # BULLISH_INSIDE_HOLD failure mode.
        rows = []
        for i in range(9):
            rows.append({
                "timestamp": f"2026-05-23T1{i}:00:00+00:00",
                "action_taken": "HOLD NVDA → HOLD",
                "reasoning": json.dumps({
                    "decision": {
                        "action": "HOLD",
                        "ticker": "NVDA",
                        "reasoning": "Sitting tight on NVDA through earnings.",
                    }
                }),
            })
        # The mismatched row — newest timestamp so newest-first sort puts it
        # at the head of mismatches[].
        rows.append({
            "timestamp": "2026-05-23T20:00:00+00:00",
            "action_taken": "HOLD NVDA → HOLD",
            "reasoning": json.dumps({
                "decision": {
                    "action": "HOLD",
                    "ticker": "NVDA",
                    "reasoning": (
                        "We are adding to NVDA on the dip — buying more "
                        "exposure into the print."
                    ),
                }
            }),
        })
        _seed_decisions(store, rows)
        r = c.get("/api/reasoning-action-verbs?limit=50")
        body = r.get_json()
        assert body["n_parsed"] == 10
        assert body["n_mismatched"] == 1
        assert body["by_verdict"].get("BULLISH_INSIDE_HOLD") == 1
        # Mismatch payload populated with the cue + snippet.
        assert len(body["mismatches"]) == 1
        m = body["mismatches"][0]
        assert m["verdict"] == "BULLISH_INSIDE_HOLD"
        assert m["leaning"] == "BULLISH"
        assert m["action"] == "HOLD"
        # The cue scanner should have picked up "adding" and "buying".
        # We only assert at least one bullish cue (cue-list contents are
        # pinned in the builder's own test suite).
        assert m["n_bullish_cues"] >= 1
        assert "adding" in " ".join(m["cues_bullish"]).lower() \
            or "buying" in " ".join(m["cues_bullish"]).lower()
        # snippet is non-empty and contains the cue word.
        assert m["snippet"] and len(m["snippet"]) > 0

"""``/api/host-guard`` endpoint shape — the dashboard surface for the
host-saturation diagnostic that already exists as a CLI.

The recurring NO_DECISION storms are host saturation (paper_trader/host_guard.py).
The CLI (`python3 -m paper_trader.host_guard`) prints a per-cause breakdown so
the operator can see *which* class of failure is dominant (model_timeout vs
cli_nonzero_rc vs host_skip vs ...) — three classes that need three different
actions. The dashboard previously surfaced only the AGGREGATE empty/skip rates,
so an operator hitting the dashboard during a storm saw the rate collapse to a
single number with no class signal.

These tests pin the additive ``starvation_by_cause`` key on the endpoint:

  * the field is ALWAYS present in the response, even when the live DB is empty
    or unreadable (an operator can render the cell without a key-miss guard);
  * counts reconcile — sum(by_cause.values()) == starved;
  * the all-labels invariant — every label in ``_CAUSE_LABELS`` appears in
    by_cause with a zero count if absent;
  * a builder-side fault degrades to the all-zero shape (never raises into the
    dashboard, mirroring the existing ``recent_empty_rate`` /
    ``recent_skip_rate`` contracts).
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader import host_guard as hg


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A Flask test client whose host_guard reads point at a *seeded temp DB*
    instead of the live paper_trader.db, and whose host_guard.probe() is
    monkey-patched to a deterministic CLEAR snapshot so the endpoint never
    touches /proc."""
    from paper_trader import dashboard, store as store_mod

    # Singleton store reset + a probe override pinned to "host is clear" so
    # the saturation verdict is deterministic regardless of CI host load.
    db = tmp_path / "paper_trader.db"
    monkeypatch.setattr(store_mod, "DB_PATH", db)
    monkeypatch.setattr(store_mod, "_singleton", None)
    monkeypatch.setattr(hg, "_DEFAULT_DB", db)
    monkeypatch.setattr(hg, "probe", lambda: {
        "opus_count": 1, "mem_available_mb": 8000.0,
        "swap_used_pct": 5.0, "load1": 1.0, "load5": 1.0,
        "load15": 1.0, "cpus": 16, "load_per_cpu": 0.1,
    })

    # Pre-create the schema by creating + closing a Store — the dashboard
    # endpoint reads via get_store() with the same DB_PATH.
    s = store_mod.Store()
    yield dashboard.app.test_client(), db
    s.close()


def _seed_decisions(db: Path, rows: list[tuple[str, str]]) -> None:
    """Insert raw decisions rows (action_taken, reasoning) into the seeded DB."""
    conn = sqlite3.connect(db)
    conn.executemany(
        "INSERT INTO decisions (timestamp, market_open, signal_count, "
        "action_taken, reasoning, portfolio_value, cash) "
        "VALUES (datetime('now'), 1, 0, ?, ?, 1000.0, 1000.0)",
        rows,
    )
    conn.commit()
    conn.close()


class TestHostGuardStarvationByCause:
    def test_field_always_present_even_on_empty_db(self, client):
        c, _db = client
        j = c.get("/api/host-guard").get_json()
        assert "starvation_by_cause" in j, (
            "starvation_by_cause must be present so the dashboard cell can "
            "render without a key-miss guard"
        )
        bc = j["starvation_by_cause"]
        # All-labels invariant — every cause key is always present, zero
        # when absent. An operator's renderer counts on this.
        assert set(bc["by_cause"]) == set(hg._CAUSE_LABELS)
        assert all(v == 0 for v in bc["by_cause"].values())
        # Empty DB → 0 rows, 0 starved.
        assert bc["starved"] == 0

    def test_per_cause_breakdown_exact_with_seeded_rows(self, client):
        c, db = client
        _seed_decisions(db, [
            ("NO_DECISION", "claude returned no response (timeout)"),
            ("NO_DECISION", "claude returned no response (timeout)"),
            ("NO_DECISION", "claude returned no response (nonzero_rc)"),
            ("NO_DECISION", "claude returned no response (empty_stdout)"),
            ("NO_DECISION", "claude returned no response (cli_missing)"),
            ("NO_DECISION", "skipped claude call — host saturated: 5 ..."),
            ("NO_DECISION", "skipped claude call — host saturated mid-call ..."),
            # Non-starvation NO_DECISION row — must NOT be counted.
            ("NO_DECISION", "parse_failed: {garbage"),
            # Non-NO_DECISION rows — never counted.
            ("BUY NVDA → FILLED", "ok"),
            ("HOLD AAPL → HOLD", "no edge"),
        ])
        bc = c.get("/api/host-guard").get_json()["starvation_by_cause"]
        assert bc["ok"] is True
        # 7 starved out of 10 total NO_DECISION-eligible rows.
        assert bc["starved"] == 7
        # Counts reconcile exactly — by_cause is a partition of `starved`.
        assert sum(bc["by_cause"].values()) == bc["starved"]
        # Each known class exact.
        assert bc["by_cause"]["model_timeout"] == 2
        assert bc["by_cause"]["cli_nonzero_rc"] == 1
        assert bc["by_cause"]["model_empty"] == 1
        assert bc["by_cause"]["cli_missing"] == 1
        assert bc["by_cause"]["host_skip"] == 2
        assert bc["by_cause"]["unknown"] == 0

    def test_unknown_suffix_lands_in_unknown_bucket_not_dropped(self, client):
        c, db = client
        # A future cause code we don't yet recognise. The aggregate
        # `starved` count must NOT silently drop the row — it must roll
        # into `unknown` so an operator watching the dashboard during the
        # NEXT-class-of-failure storm still sees the freeze cost.
        _seed_decisions(db, [
            ("NO_DECISION", "claude returned no response (future_cause_xyz)"),
            ("NO_DECISION", "claude returned no response (timeout)"),
        ])
        bc = c.get("/api/host-guard").get_json()["starvation_by_cause"]
        assert bc["starved"] == 2
        assert bc["by_cause"]["unknown"] == 1
        assert bc["by_cause"]["model_timeout"] == 1
        assert sum(bc["by_cause"].values()) == bc["starved"]

    def test_builder_fault_degrades_to_all_zero_shape(self, client, monkeypatch):
        c, _db = client
        # Force the builder to explode — the endpoint must catch and
        # return the all-zero shape, never propagate a 500.
        def _boom(*_a, **_k):
            raise RuntimeError("builder kaboom")
        monkeypatch.setattr(hg, "recent_starvation_by_cause", _boom)
        r = c.get("/api/host-guard")
        assert r.status_code == 200, "builder fault must NOT 500 the panel"
        bc = r.get_json()["starvation_by_cause"]
        assert bc["ok"] is False
        assert bc["starved"] == 0
        assert set(bc["by_cause"]) == set(hg._CAUSE_LABELS)
        assert all(v == 0 for v in bc["by_cause"].values())

    def test_existing_fields_preserved_alongside_new_field(self, client):
        """Backwards-compat lock: every pre-existing key the operator panel
        already reads must still be present after the additive change."""
        c, _db = client
        j = c.get("/api/host-guard").get_json()
        # Pre-existing surface — locked so a future refactor cannot rename
        # or drop them and silently break the dashboard panel.
        for key in ("probe", "reason", "saturated", "recent_empty_rate",
                    "recent_skip_rate", "pulse"):
            assert key in j, f"existing key {key!r} dropped from /api/host-guard"


class TestClassifyEdgeCases:
    """Edge-case parsing pins for ``_classify_starvation_cause`` — the
    sub-bucket extractor. Existing tests cover the happy paths (each known
    suffix, each known unknown). These lock in the explicit normalisations
    (case-insensitivity, whitespace) so a future "optimisation" that drops
    .lower() or .strip() would be caught."""

    def test_classify_is_case_insensitive(self):
        c = hg._classify_starvation_cause
        # The CLI suffix is written lower-case by strategy.py today, but
        # the classifier explicitly .lower()s before matching — that
        # normalisation is load-bearing and must not be removed.
        assert c("claude returned no response (TIMEOUT)") == "model_timeout"
        assert c("claude returned no response (Nonzero_RC)") == "cli_nonzero_rc"
        assert c("claude returned no response (Empty_Stdout)") == "model_empty"

    def test_classify_strips_whitespace_in_parens(self):
        c = hg._classify_starvation_cause
        # An accidental space inside the parens (operator-edited reason,
        # legacy logger formatting drift) must still bucket correctly.
        assert c("claude returned no response (  timeout  )") == "model_timeout"
        assert c("claude returned no response ( cli_missing )") == "cli_missing"

    def test_classify_empty_parens_is_unknown(self):
        c = hg._classify_starvation_cause
        # Empty / whitespace-only parens have no cause code → must roll
        # into `unknown` rather than into a wrong bucket via misread.
        assert c("claude returned no response ()") == "unknown"
        assert c("claude returned no response (   )") == "unknown"

    def test_classify_host_skip_wins_over_inner_parens(self):
        c = hg._classify_starvation_cause
        # A host_skip reason that happens to embed a parenthesised reason
        # (e.g. a future format that includes the underlying cause) must
        # still bucket as host_skip — the prefix dispatches first.
        assert c("skipped claude call — host saturated (timeout): 5 conc") == "host_skip"
        assert c("skipped claude call — host saturated mid-call (nonzero_rc)") == "host_skip"

"""Endpoint tests for ``/api/persona-leaderboard``.

The per-persona strategy-quality builder (``paper_trader.ml.persona_leaderboard``)
is exhaustively unit-tested in ``test_persona_leaderboard_20260517.py``. This
file covers the *route* added to surface it on the dashboard: shape, the
``run_id → persona`` mapping the route applies via the SSOT ``persona_for``,
the EDGE / DRAG grading flowing through, equity-curve risk metrics surviving
the DB round-trip, the insufficient-data and missing-DB degradations, and an
SSOT-parity assertion that the route forks none of the builder's logic.

DB-isolated: every test patches ``dashboard.BACKTEST_DB`` to a throwaway temp
DB (the ``test_model_rankings.py`` precedent). No live process, no network.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_store(tmp_path):
    from paper_trader.backtest import BacktestStore
    return BacktestStore(path=tmp_path / "bt.db")


def _insert_run(store, run_id, vs_spy_pct, total_return_pct=None,
                equity_curve_json=None, status="complete"):
    """Insert one backtest_runs row. total_return_pct defaults to vs_spy;
    equity_curve_json is NOT NULL in the schema so a missing curve is the
    empty-JSON-array string (what `_load_runs` parses to an empty curve)."""
    if total_return_pct is None:
        total_return_pct = vs_spy_pct
    if equity_curve_json is None:
        equity_curve_json = "[]"
    store.conn.execute(
        "INSERT INTO backtest_runs (run_id, seed, start_date, end_date, "
        "start_value, final_value, total_return_pct, spy_return_pct, "
        "vs_spy_pct, n_trades, n_decisions, status, started_at, "
        "equity_curve_json, model_id) VALUES "
        "(?,1,'2025-01-01','2026-01-01',1000,1000,?,0.0,?,10,100,?,"
        "'2026-01-01T00:00:00Z',?,'ml_quant')",
        (run_id, total_return_pct, vs_spy_pct, status, equity_curve_json),
    )
    store.conn.commit()


def _client(tmp_path):
    import paper_trader.dashboard as dash
    dash.BACKTEST_DB = tmp_path / "bt.db"
    return dash.app.test_client()


# ─────────────────────────── shape / degradation ───────────────────────────
def test_endpoint_empty_db_is_insufficient_not_500(tmp_path):
    """An empty backtest DB degrades to INSUFFICIENT_DATA at HTTP 200."""
    _make_store(tmp_path)  # creates schema, zero rows
    resp = _client(tmp_path).get("/api/persona-leaderboard")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["verdict"] == "INSUFFICIENT_DATA"
    assert data["status"] == "insufficient_data"
    assert data["leaderboard"] == []
    assert "hint" in data and "as_of" in data


def test_endpoint_missing_db_file_degrades_gracefully(tmp_path):
    """BACKTEST_DB pointing at a nonexistent file → 200 INSUFFICIENT_DATA,
    never a 500 — the route guards with ``db.exists()``."""
    import paper_trader.dashboard as dash
    dash.BACKTEST_DB = tmp_path / "does_not_exist.db"
    resp = dash.app.test_client().get("/api/persona-leaderboard")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["verdict"] == "INSUFFICIENT_DATA"
    assert data["leaderboard"] == []


def test_endpoint_under_min_records_is_insufficient(tmp_path):
    """Fewer than MIN_RECORDS qualifying runs → INSUFFICIENT_DATA even though
    rows exist (the builder's global sample floor)."""
    from paper_trader.ml.persona_leaderboard import MIN_RECORDS
    store = _make_store(tmp_path)
    for rid in range(1, 11):  # 10 runs, well under MIN_RECORDS (30)
        _insert_run(store, rid, vs_spy_pct=15.0)
    assert 10 < MIN_RECORDS
    data = json.loads(_client(tmp_path).get("/api/persona-leaderboard").data)
    assert data["verdict"] == "INSUFFICIENT_DATA"
    assert data["n_runs"] == 10


# ─────────────────────────── grading / mapping ─────────────────────────────
def _seed_edge_drag_flat(store):
    """50 runs, run_id 1..50. persona_for makes run_id p,p+10,..,p+40 → persona p.

    persona 1 (Value Investor):  5 runs vs_spy +25  → EDGE  (median 25, win 1.0)
    persona 2 (Momentum Trader): 5 runs vs_spy  -3  → DRAG  (median -3)
    personas 3..10:              5 runs vs_spy  +5  → FLAT  (median 5, <20)
    persona 1's runs also carry a 1000→800→1000 curve → 20% max drawdown.
    """
    curve = json.dumps([{"value": 1000}, {"value": 800}, {"value": 1000}])
    for rid in range(1, 51):
        persona_idx = ((rid - 1) % 10) + 1
        if persona_idx == 1:
            _insert_run(store, rid, vs_spy_pct=25.0, equity_curve_json=curve)
        elif persona_idx == 2:
            _insert_run(store, rid, vs_spy_pct=-3.0)
        else:
            _insert_run(store, rid, vs_spy_pct=5.0)


def test_endpoint_grades_edge_drag_flat(tmp_path):
    """EDGE / DRAG / FLAT verdicts flow through the route from the builder."""
    store = _make_store(tmp_path)
    _seed_edge_drag_flat(store)
    data = json.loads(_client(tmp_path).get("/api/persona-leaderboard").data)

    assert data["verdict"] == "HAS_DRAG_PERSONA"
    assert data["n_runs"] == 50
    assert data["n_personas"] == 10
    by_persona = {p["persona"]: p for p in data["leaderboard"]}

    assert by_persona["Value Investor"]["verdict"] == "EDGE"
    assert by_persona["Value Investor"]["median_vs_spy"] == pytest.approx(25.0)
    assert by_persona["Value Investor"]["win_rate"] == pytest.approx(1.0)

    assert by_persona["Momentum Trader"]["verdict"] == "DRAG"
    assert by_persona["Momentum Trader"]["median_vs_spy"] == pytest.approx(-3.0)
    assert "Momentum Trader" in data["drag_personas"]

    assert by_persona["Contrarian"]["verdict"] == "FLAT"
    assert by_persona["Contrarian"]["median_vs_spy"] == pytest.approx(5.0)


def test_endpoint_leaderboard_sorted_by_median_vs_spy_desc(tmp_path):
    """The route preserves the builder's median-vs-SPY descending order."""
    store = _make_store(tmp_path)
    _seed_edge_drag_flat(store)
    data = json.loads(_client(tmp_path).get("/api/persona-leaderboard").data)
    medians = [p["median_vs_spy"] for p in data["leaderboard"]]
    assert medians == sorted(medians, reverse=True)
    # The EDGE persona tops it; the DRAG persona is last.
    assert data["leaderboard"][0]["persona"] == "Value Investor"
    assert data["leaderboard"][-1]["persona"] == "Momentum Trader"


def test_endpoint_maps_run_id_to_persona_via_persona_for(tmp_path):
    """The route attributes each run to the persona the SSOT persona_for
    names — a run_id-2 bucket must land under Momentum Trader, not elsewhere."""
    from paper_trader.backtest import persona_for
    store = _make_store(tmp_path)
    _seed_edge_drag_flat(store)
    data = json.loads(_client(tmp_path).get("/api/persona-leaderboard").data)
    names = {p["persona"] for p in data["leaderboard"]}
    # run_id 2 maps to persona index 2; that name must be the DRAG bucket.
    assert persona_for(2)["name"] == "Momentum Trader"
    assert "Momentum Trader" in names
    assert persona_for(2)["name"] in data["drag_personas"]


def test_endpoint_equity_curve_risk_metrics_survive_db_roundtrip(tmp_path):
    """equity_curve_json stored in the DB is parsed and reaches the risk
    aggregates — the route must not drop the curve. The 1000→800→1000 curve
    is a clean 20% max drawdown."""
    store = _make_store(tmp_path)
    _seed_edge_drag_flat(store)
    data = json.loads(_client(tmp_path).get("/api/persona-leaderboard").data)
    vi = next(p for p in data["leaderboard"] if p["persona"] == "Value Investor")
    assert vi["median_max_drawdown_pct"] == pytest.approx(20.0)
    # personas with no curve keep a None risk metric (not a fabricated 0).
    mt = next(p for p in data["leaderboard"] if p["persona"] == "Momentum Trader")
    assert mt["median_max_drawdown_pct"] is None


def test_endpoint_is_ssot_parity_with_builder(tmp_path):
    """The route forks none of the builder's logic: its payload (minus the
    route-only ``as_of`` stamp) is byte-identical to calling the module's
    own ``_load_runs`` + ``persona_leaderboard`` directly."""
    from paper_trader.ml.persona_leaderboard import _load_runs, persona_leaderboard
    store = _make_store(tmp_path)
    _seed_edge_drag_flat(store)

    data = json.loads(_client(tmp_path).get("/api/persona-leaderboard").data)
    data.pop("as_of", None)

    direct = persona_leaderboard(_load_runs(tmp_path / "bt.db"))
    assert data == direct

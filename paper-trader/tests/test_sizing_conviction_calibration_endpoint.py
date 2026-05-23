"""Verifies the /api/sizing-conviction-calibration endpoint wires the
conviction_calibration analyzer correctly and that its bucketed verdict
shape matches the analyzer's own contract.

This is an analytics-verification test in the [paper-trader analytics
verification] discipline: drive the route via Flask's `test_client` so
the test exercises the same path the unified dashboard hits, not a
module __main__ smoke.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_outcomes(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _make_well_calibrated_rows(n: int = 60) -> list[dict]:
    """Build BUY rows where higher conviction_pct → higher forward_return_5d.

    Monotone non-decreasing across all buckets + strong rank skill +
    >=3pp top-vs-bottom spread → the analyzer must emit WELL_CALIBRATED.
    """
    rows = []
    for i in range(n):
        conv = round((i + 1) / n, 4)  # 0.017 .. 1.00 strictly increasing
        ret = -2.0 + (conv * 10.0)    # -2% at lowest, +8% at highest
        rows.append({
            "action": "BUY",
            "conviction_pct": conv,
            "forward_return_5d": ret,
        })
    return rows


def _make_inverted_rows(n: int = 60) -> list[dict]:
    """Build BUY rows where higher conviction → LOWER realized return.

    Spearman ≤ -0.05 AND top-bottom < -1pp → analyzer must emit INVERTED.
    """
    rows = []
    for i in range(n):
        conv = round((i + 1) / n, 4)
        ret = 8.0 - (conv * 10.0)
        rows.append({
            "action": "BUY",
            "conviction_pct": conv,
            "forward_return_5d": ret,
        })
    return rows


@pytest.fixture
def client(monkeypatch, tmp_path):
    """Flask test client with the outcomes path redirected into a tmp dir.

    The endpoint resolves the path via ``Path(_REPO_DIR) / "data" /
    "decision_outcomes.jsonl"``. Redirecting ``_REPO_DIR`` to a tmpdir keeps
    the test off the live JSONL and lets each test seed its own corpus.
    """
    from paper_trader import dashboard as dash
    monkeypatch.setattr(dash, "_REPO_DIR", str(tmp_path))
    dash.app.testing = True
    return dash.app.test_client()


def test_endpoint_returns_well_calibrated_for_monotone_rising_data(
    client, tmp_path
):
    outcomes = tmp_path / "data" / "decision_outcomes.jsonl"
    _write_outcomes(outcomes, _make_well_calibrated_rows(60))

    r = client.get("/api/sizing-conviction-calibration")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ok"
    assert body["verdict"] == "WELL_CALIBRATED"
    assert body["n"] == 60
    assert body["spearman"] >= 0.99
    assert body["monotone_fraction"] == 1.0
    assert body["top_minus_bottom_realized_pct"] >= 3.0
    assert isinstance(body["buckets"], list) and len(body["buckets"]) == 5
    assert body["as_of"].endswith("+00:00")
    assert body["source_path"].endswith("data/decision_outcomes.jsonl")


def test_endpoint_returns_inverted_for_anti_predictive_sizing(
    client, tmp_path
):
    outcomes = tmp_path / "data" / "decision_outcomes.jsonl"
    _write_outcomes(outcomes, _make_inverted_rows(60))

    r = client.get("/api/sizing-conviction-calibration")
    assert r.status_code == 200
    body = r.get_json()
    assert body["verdict"] == "INVERTED"
    assert body["spearman"] < 0.0
    assert body["top_minus_bottom_realized_pct"] < 0.0


def test_endpoint_returns_insufficient_when_too_few_rows(client, tmp_path):
    outcomes = tmp_path / "data" / "decision_outcomes.jsonl"
    # MIN_PAIRS is 30 — give only 5
    _write_outcomes(outcomes, _make_well_calibrated_rows(5))

    r = client.get("/api/sizing-conviction-calibration")
    assert r.status_code == 200
    body = r.get_json()
    assert body["verdict"] == "INSUFFICIENT_DATA"
    assert body["n"] == 5
    assert body["status"] == "insufficient_data"


def test_endpoint_returns_insufficient_when_outcomes_file_missing(
    client, tmp_path
):
    # No file written at all
    r = client.get("/api/sizing-conviction-calibration")
    assert r.status_code == 200
    body = r.get_json()
    assert body["verdict"] == "INSUFFICIENT_DATA"
    assert body["n"] == 0


def test_endpoint_skips_sell_rows_and_invalid_conviction(client, tmp_path):
    """Non-BUY rows and out-of-range conviction must be dropped, not counted."""
    rows = _make_well_calibrated_rows(40)
    rows.append({"action": "SELL", "conviction_pct": 0.5, "forward_return_5d": 99.0})
    rows.append({"action": "BUY", "conviction_pct": 1.5, "forward_return_5d": 99.0})
    rows.append({"action": "BUY", "conviction_pct": 0.5, "forward_return_5d": float("nan")})

    outcomes = tmp_path / "data" / "decision_outcomes.jsonl"
    _write_outcomes(outcomes, rows)

    r = client.get("/api/sizing-conviction-calibration")
    body = r.get_json()
    assert body["n"] == 40
    assert body["n_dropped_action"] == 1
    assert body["n_dropped_conviction"] == 1
    assert body["n_dropped_return"] == 1

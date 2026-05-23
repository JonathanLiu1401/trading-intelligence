"""Verifies /api/active-learning-queue surfaces the
data/active_learning_queue.jsonl tail with correct limit/total semantics.

The endpoint resolves the file path via ``BASE_DIR / "data" /
"active_learning_queue.jsonl"`` — we monkeypatch BASE_DIR onto a tmpdir
so the test never touches the live 2.5 MB queue.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_queue(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for i in range(n):
            fh.write(json.dumps({
                "ts": f"2026-05-20T00:{i % 60:02d}:00Z",
                "id": f"row_{i:06d}",
                "title": f"uncertain article {i}",
                "source": "gdelt_test",
                "url": f"https://example.com/{i}",
                "rel": 5.0,
                "urg": 4.0,
                "rel_std": 2.0 + i * 0.001,
                "urg_std": 2.0,
                "priority": 99.0,
                "reason": "high_variance",
            }) + "\n")


def _client(tmp_path, monkeypatch, store_factory):
    """Flask test client with BASE_DIR redirected to tmp_path."""
    from dashboard import web_server
    monkeypatch.setattr(web_server, "BASE_DIR", tmp_path)
    store = store_factory()  # any store; the endpoint reads only the file
    app = web_server.create_app(store=store)
    app.testing = True
    return app.test_client()


def test_returns_last_n_rows_newest_first(tmp_path, monkeypatch, store_factory):
    queue = tmp_path / "data" / "active_learning_queue.jsonl"
    _write_queue(queue, n=50)
    client = _client(tmp_path, monkeypatch, store_factory)

    r = client.get("/api/active-learning-queue?limit=10")
    assert r.status_code == 200
    body = r.get_json()

    assert body["limit"] == 10
    assert body["returned"] == 10
    assert body["total_queued"] == 50
    # Newest-first: row 49 is most recent (we wrote them in order)
    assert body["items"][0]["id"] == "row_000049"
    assert body["items"][-1]["id"] == "row_000040"


def test_default_limit_is_25(tmp_path, monkeypatch, store_factory):
    queue = tmp_path / "data" / "active_learning_queue.jsonl"
    _write_queue(queue, n=100)
    client = _client(tmp_path, monkeypatch, store_factory)

    r = client.get("/api/active-learning-queue")
    body = r.get_json()
    assert body["limit"] == 25
    assert body["returned"] == 25


def test_limit_is_clamped_to_100(tmp_path, monkeypatch, store_factory):
    queue = tmp_path / "data" / "active_learning_queue.jsonl"
    _write_queue(queue, n=200)
    client = _client(tmp_path, monkeypatch, store_factory)

    r = client.get("/api/active-learning-queue?limit=500")
    body = r.get_json()
    assert body["limit"] == 100
    assert body["returned"] == 100


def test_invalid_limit_falls_back_to_default(tmp_path, monkeypatch, store_factory):
    queue = tmp_path / "data" / "active_learning_queue.jsonl"
    _write_queue(queue, n=30)
    client = _client(tmp_path, monkeypatch, store_factory)

    r = client.get("/api/active-learning-queue?limit=notanumber")
    body = r.get_json()
    assert body["limit"] == 25


def test_missing_file_returns_empty_list(tmp_path, monkeypatch, store_factory):
    # No queue file written — endpoint must still 200 with returned=0.
    client = _client(tmp_path, monkeypatch, store_factory)

    r = client.get("/api/active-learning-queue")
    assert r.status_code == 200
    body = r.get_json()
    assert body["returned"] == 0
    assert body["total_queued"] == 0
    assert body["items"] == []


def test_malformed_line_is_skipped_not_fatal(tmp_path, monkeypatch, store_factory):
    queue = tmp_path / "data" / "active_learning_queue.jsonl"
    queue.parent.mkdir(parents=True, exist_ok=True)
    with queue.open("w") as fh:
        fh.write(json.dumps({"id": "valid_1", "rel_std": 9.0}) + "\n")
        fh.write("this is not json at all\n")
        fh.write(json.dumps({"id": "valid_2", "rel_std": 9.0}) + "\n")
    client = _client(tmp_path, monkeypatch, store_factory)

    r = client.get("/api/active-learning-queue?limit=10")
    body = r.get_json()
    assert body["total_queued"] == 3  # raw line count
    ids = {it["id"] for it in body["items"]}
    assert ids == {"valid_1", "valid_2"}

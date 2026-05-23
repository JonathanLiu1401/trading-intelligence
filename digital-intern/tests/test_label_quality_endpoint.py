"""Verifies /api/label-quality composes label_audit + score_agreement
into a single roll-up endpoint with a verdict the operator can act on.

Pattern follows test_active_learning_queue_endpoint / test_ml_status:
build a fresh ArticleStore, seed a tiny corpus, drive the route via
Flask test_client.
"""
from __future__ import annotations

import sqlite3

import pytest


def _seed_clean_pool(conn: sqlite3.Connection, n_llm: int = 200) -> None:
    """Seed a clean pool: only ``score_source='llm'`` rows, no hygiene
    violations, with ml/ai scores in tight agreement. The combined audit
    must return verdict='OK'.
    """
    # 1) score_source='llm' label rows. These are the strong-pool's gold.
    for i in range(n_llm):
        ai = 6.0 + (i % 4) * 0.5
        ml = ai + 0.1  # tight agreement (bias and divergence both small)
        conn.execute(
            "INSERT OR REPLACE INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, "
            " urgency, first_seen, score_source, ml_score) VALUES "
            "(?,  ?,   ?,     ?,      ?,         ?,        ?,        "
            " ?,       ?,          ?,            ?)",
            (
                f"clean_{i}",
                f"https://news.example.com/article_{i}",
                f"clean article {i}",
                "rss_clean",
                "2026-05-20T00:00:00Z",
                3.0,
                ai,
                0,
                "2026-05-20T00:00:00Z",
                "llm",
                ml,
            ),
        )
    conn.commit()


def _seed_hygiene_violation(conn: sqlite3.Connection) -> None:
    """One row with score_source='ml' AND ai_score > 0 — the canonical
    column-hygiene break that label_audit's ``ok`` flag gates on."""
    conn.execute(
        "INSERT OR REPLACE INTO articles "
        "(id, url, title, source, published, kw_score, ai_score, "
        " urgency, first_seen, score_source, ml_score) VALUES "
        "(?,  ?,   ?,     ?,      ?,         ?,        ?,        "
        " ?,       ?,          ?,            ?)",
        (
            "dirty_1",
            "https://news.example.com/dirty_1",
            "dirty article",
            "rss_dirty",
            "2026-05-20T00:00:00Z",
            3.0,
            7.5,           # ai_score in the LABEL column
            0,
            "2026-05-20T00:00:00Z",
            "ml",          # ...but score_source says it's a PREDICTION
            7.5,
        ),
    )
    conn.commit()


def _client(monkeypatch, store):
    """Build a Flask test client whose ArticleStore points at our seeded DB.

    The endpoint uses the module-level `_store_handle` to resolve the DB
    path inside _ro_conn(); monkeypatching the dashboard's _store import
    keeps the test offline.
    """
    from dashboard import web_server
    monkeypatch.setattr(web_server, "_store", store)
    app = web_server.create_app(store=store)
    app.testing = True
    return app.test_client()


def test_clean_pool_with_tight_agreement_returns_ok(store_factory, monkeypatch):
    store = store_factory()
    _seed_clean_pool(store.conn, n_llm=200)
    client = _client(monkeypatch, store)

    r = client.get("/api/label-quality")
    assert r.status_code == 200
    body = r.get_json()

    assert body["verdict"] == "OK"
    assert isinstance(body["label_audit"], dict)
    assert body["label_audit"]["ok"] is True
    assert body["label_audit"]["column_hygiene_violations"] == 0
    assert isinstance(body["score_agreement"], dict)
    assert body["score_agreement"]["n"] >= 100
    assert body["score_agreement"]["strong_disagreement_pct"] < 15.0
    assert body["errors"] == []
    assert body["as_of"].endswith("+00:00")


def test_hygiene_violation_forces_dirty_verdict(store_factory, monkeypatch):
    """A single score_source='ml' row with ai_score > 0 must escalate to
    DIRTY regardless of how clean the agreement signal looks.
    """
    store = store_factory()
    _seed_clean_pool(store.conn, n_llm=200)
    _seed_hygiene_violation(store.conn)
    client = _client(monkeypatch, store)

    r = client.get("/api/label-quality")
    assert r.status_code == 200
    body = r.get_json()

    assert body["verdict"] == "DIRTY"
    assert body["label_audit"]["ok"] is False
    assert body["label_audit"]["column_hygiene_violations"] == 1


def test_large_divergence_produces_diverging_verdict(store_factory, monkeypatch):
    """When the cheap ml_score systematically disagrees with the LLM
    ai_score on a large enough overlap, hygiene is still clean but the
    operator must see DIVERGING — the cheap model has drifted off the
    Claude signal.
    """
    store = store_factory()
    # Seed 300 rows where ml_score is consistently 5pt higher than ai_score
    # → bias ≈ +5, strong_disagreement_pct = 100%.
    for i in range(300):
        store.conn.execute(
            "INSERT OR REPLACE INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, "
            " urgency, first_seen, score_source, ml_score) VALUES "
            "(?,  ?,   ?,     ?,      ?,         ?,        ?,        "
            " ?,       ?,          ?,            ?)",
            (
                f"div_{i}",
                f"https://example.com/d_{i}",
                f"diverging {i}",
                "rss",
                "2026-05-20T00:00:00Z",
                3.0,
                3.0,
                0,
                "2026-05-20T00:00:00Z",
                "llm",
                8.0,
            ),
        )
    store.conn.commit()
    client = _client(monkeypatch, store)

    r = client.get("/api/label-quality")
    body = r.get_json()

    assert body["verdict"] == "DIVERGING"
    assert body["label_audit"]["ok"] is True
    assert body["score_agreement"]["strong_disagreement_pct"] >= 15.0


def test_empty_db_does_not_raise(store_factory, monkeypatch):
    """A brand-new store with no articles must still return 200 + a
    sensible verdict, never 500. The route's `try/except` envelope per
    sub-analyzer is the safety net.
    """
    store = store_factory()  # fresh, no rows
    client = _client(monkeypatch, store)

    r = client.get("/api/label-quality")
    assert r.status_code == 200
    body = r.get_json()
    # Pool is empty so label_audit reports ok=True (no violations) but
    # score_agreement has n=0; the route falls through to OK_LOW_OVERLAP.
    assert body["verdict"] in ("OK_LOW_OVERLAP", "OK")
    assert body["label_audit"]["column_hygiene_violations"] == 0

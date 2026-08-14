"""OpenClaw xAI tokens live in agent SQLite, not auth-profiles.json."""
from __future__ import annotations

import json
import sqlite3
import time

from paper_trader import strategy


def _write_store(db_path, profiles):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE auth_profile_store "
        "(store_key TEXT PRIMARY KEY, store_json TEXT NOT NULL, updated_at INTEGER NOT NULL)"
    )
    conn.execute(
        "INSERT INTO auth_profile_store VALUES (?, ?, ?)",
        ("primary", json.dumps({"version": 1, "profiles": profiles}), int(time.time() * 1000)),
    )
    conn.commit()
    conn.close()


def test_load_xai_token_from_openclaw_sqlite(tmp_path, monkeypatch):
    db = tmp_path / "openclaw-agent.sqlite"
    json_missing = tmp_path / "auth-profiles.json"
    token = "xai-test-token-from-sqlite"
    _write_store(db, {
        "xai:iamthemostproguy@gmail.com": {
            "type": "oauth",
            "provider": "xai",
            "access": token,
            "expires": int(time.time() * 1000) + 3_600_000,
        }
    })
    monkeypatch.setattr(strategy, "XAI_AUTH_PROFILES_PATH", json_missing)
    monkeypatch.setattr(strategy, "XAI_AUTH_PROFILE", "xai:iamthemostproguy@gmail.com")
    monkeypatch.delenv("PAPER_TRADER_XAI_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    assert strategy._load_xai_access_token() == token


def test_env_key_still_wins(tmp_path, monkeypatch):
    monkeypatch.setattr(strategy, "XAI_AUTH_PROFILES_PATH", tmp_path / "auth-profiles.json")
    monkeypatch.setenv("XAI_API_KEY", "env-wins")
    assert strategy._load_xai_access_token() == "env-wins"


def test_expired_sqlite_token_is_skipped(tmp_path, monkeypatch):
    db = tmp_path / "openclaw-agent.sqlite"
    json_missing = tmp_path / "auth-profiles.json"
    _write_store(db, {
        "xai:old": {
            "type": "oauth",
            "provider": "xai",
            "access": "expired-token",
            "expires": 1,
        }
    })
    monkeypatch.setattr(strategy, "XAI_AUTH_PROFILES_PATH", json_missing)
    monkeypatch.setattr(strategy, "XAI_AUTH_PROFILE", "xai:old")
    monkeypatch.delenv("PAPER_TRADER_XAI_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    assert strategy._load_xai_access_token() is None

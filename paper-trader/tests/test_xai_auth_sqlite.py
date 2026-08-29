"""OpenClaw xAI tokens live in agent SQLite, not auth-profiles.json."""
from __future__ import annotations

import json
import sqlite3
import time

from paper_trader import strategy


def _write_store(db_path, profiles):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS auth_profile_store "
        "(store_key TEXT PRIMARY KEY, store_json TEXT NOT NULL, updated_at INTEGER NOT NULL)"
    )
    conn.execute(
        "INSERT OR REPLACE INTO auth_profile_store VALUES (?, ?, ?)",
        ("primary", json.dumps({"version": 1, "profiles": profiles}), int(time.time() * 1000)),
    )
    conn.commit()
    conn.close()


def _isolate_auth(monkeypatch, json_missing, profile):
    monkeypatch.setattr(strategy, "XAI_AUTH_PROFILES_PATH", json_missing)
    monkeypatch.setattr(strategy, "XAI_AUTH_PROFILE", profile)
    monkeypatch.setattr(strategy, "_openclaw_xai_refresh_blocked_until", 0.0)
    monkeypatch.delenv("PAPER_TRADER_XAI_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)


def test_load_xai_token_from_openclaw_sqlite(tmp_path, monkeypatch):
    db = tmp_path / "openclaw-agent.sqlite"
    json_missing = tmp_path / "auth-profiles.json"
    fixture = "xai-test-token-from-sqlite"
    profile = "xai:iamthemostproguy@gmail.com"
    _write_store(db, {
        profile: {
            "type": "oauth",
            "provider": "xai",
            "access": fixture,
            "expires": int(time.time() * 1000) + 3_600_000,
        }
    })
    _isolate_auth(monkeypatch, json_missing, profile)
    monkeypatch.setattr(strategy, "_refresh_openclaw_xai_oauth", lambda: False)
    tok = strategy._load_xai_access_token()
    assert tok
    assert len(tok) == len(fixture)


def test_env_key_still_wins(tmp_path, monkeypatch):
    monkeypatch.setattr(strategy, "XAI_AUTH_PROFILES_PATH", tmp_path / "auth-profiles.json")
    monkeypatch.setenv("XAI_API_KEY", "env-wins")
    tok = strategy._load_xai_access_token()
    assert tok
    assert len(tok) == len("env-wins")


def test_expired_sqlite_token_is_skipped_when_openclaw_probe_fails(tmp_path, monkeypatch):
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
    _isolate_auth(monkeypatch, json_missing, "xai:old")
    monkeypatch.setattr(strategy, "_refresh_openclaw_xai_oauth", lambda: False)
    assert strategy._load_xai_access_token() is None


def test_expired_access_refreshed_via_openclaw_probe(tmp_path, monkeypatch):
    db = tmp_path / "openclaw-agent.sqlite"
    json_missing = tmp_path / "auth-profiles.json"
    profile = "xai:iamthemostproguy@gmail.com"
    _write_store(db, {
        profile: {
            "type": "oauth",
            "provider": "xai",
            "access": "expired-access",
            "expires": 1,
        }
    })
    _isolate_auth(monkeypatch, json_missing, profile)
    calls = {"n": 0}

    def fake_refresh():
        calls["n"] += 1
        _write_store(db, {
            profile: {
                "type": "oauth",
                "provider": "xai",
                "access": "fresh-after-openclaw-probe",
                "expires": int(time.time() * 1000) + 3_600_000,
            }
        })
        return True

    monkeypatch.setattr(strategy, "_refresh_openclaw_xai_oauth", fake_refresh)
    tok = strategy._load_xai_access_token()
    assert calls["n"] == 1
    assert tok
    assert len(tok) > 0

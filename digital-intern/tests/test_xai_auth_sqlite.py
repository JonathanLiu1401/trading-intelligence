"""OpenClaw xAI tokens live in agent SQLite, not auth-profiles.json."""
from __future__ import annotations

import json
import sqlite3
import time

from core import claude_cli


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
    monkeypatch.setattr(claude_cli, "XAI_AUTH_PROFILES_PATH", json_missing)
    monkeypatch.setattr(claude_cli, "XAI_AUTH_PROFILE", profile)
    monkeypatch.setattr(claude_cli, "_openclaw_xai_refresh_blocked_until", 0.0)
    monkeypatch.delenv("DIGITAL_INTERN_XAI_API_KEY", raising=False)
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
    monkeypatch.setattr(claude_cli, "_refresh_openclaw_xai_oauth", lambda: False)
    tok = claude_cli._load_xai_access_token()
    assert tok
    assert len(tok) == len(fixture)


def test_env_key_still_wins(tmp_path, monkeypatch):
    monkeypatch.setattr(claude_cli, "XAI_AUTH_PROFILES_PATH", tmp_path / "auth-profiles.json")
    monkeypatch.setenv("XAI_API_KEY", "env-wins")
    tok = claude_cli._load_xai_access_token()
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
    monkeypatch.setattr(claude_cli, "_refresh_openclaw_xai_oauth", lambda: False)
    assert claude_cli._load_xai_access_token() is None


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

    monkeypatch.setattr(claude_cli, "_refresh_openclaw_xai_oauth", fake_refresh)
    tok = claude_cli._load_xai_access_token()
    assert calls["n"] == 1
    assert tok
    assert len(tok) > 0


def test_explicit_sqlite_path_is_read(tmp_path, monkeypatch):
    db = tmp_path / "custom.sqlite"
    fixture = "xai-custom-sqlite"
    _write_store(db, {
        "xai:custom": {
            "type": "oauth",
            "provider": "xai",
            "access": fixture,
            "expires": int(time.time() * 1000) + 3_600_000,
        }
    })
    monkeypatch.setattr(claude_cli, "XAI_AUTH_PROFILES_PATH", db)
    monkeypatch.setattr(claude_cli, "XAI_AUTH_PROFILE", "xai:custom")
    monkeypatch.setattr(claude_cli, "_refresh_openclaw_xai_oauth", lambda: False)
    monkeypatch.delenv("DIGITAL_INTERN_XAI_API_KEY", raising=False)
    monkeypatch.delenv("PAPER_TRADER_XAI_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    tok = claude_cli._load_xai_access_token()
    assert tok
    assert len(tok) == len(fixture)


def test_valid_token_does_not_probe_openclaw(tmp_path, monkeypatch):
    db = tmp_path / "openclaw-agent.sqlite"
    json_missing = tmp_path / "auth-profiles.json"
    profile = "xai:iamthemostproguy@gmail.com"
    _write_store(db, {
        profile: {
            "type": "oauth",
            "provider": "xai",
            "access": "still-valid",
            "expires": int(time.time() * 1000) + 3_600_000,
        }
    })
    _isolate_auth(monkeypatch, json_missing, profile)
    calls = {"n": 0}
    monkeypatch.setattr(claude_cli, "_refresh_openclaw_xai_oauth", lambda: calls.__setitem__("n", calls["n"] + 1) or False)
    tok = claude_cli._load_xai_access_token()
    assert tok
    assert calls["n"] == 0

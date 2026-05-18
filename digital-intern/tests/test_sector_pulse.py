"""Tests for the native news-density SECTOR PULSE (web_server.py).

digital-intern owns ~1000+ live scored articles/h but the only sector heatmap
in the stack is *cross-fetched from paper-trader* (`:8090/api/sector-heatmap`,
pure price momentum) — so it goes blank exactly when the trader is down/stale
(its documented chronic state). This computes a sector view *natively* from the
news the daemon already has: which slices of the book the wire is lighting up
right now, weighted toward fresh items, independent of paper-trader uptime.

Pure pieces are locked here without standing up Flask (the established
`_*_chat_lines` precedent); the endpoint is driven through the real Flask
view on a tmp DB so the live-only backtest-isolation invariant is pinned on
this new read path too.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard import web_server
from dashboard.web_server import (
    _SECTOR_MAP,
    _aggregate_sector_pulse,
    _extract_tickers,
    _sector_pulse_chat_lines,
)

NOW = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)


def _iso(age_h: float) -> str:
    return (NOW - timedelta(hours=age_h)).isoformat()


def _a(title, ai=5.0, urg=0, age_h=1.0):
    return {"title": title, "ai_score": ai, "urgency": urg,
            "first_seen": _iso(age_h)}


class TestSectorMap:
    """The map is the taxonomy — lock its content so a careless edit can't
    silently re-bucket the user's thesis names (advisor's explicit ask)."""

    def test_core_thesis_tickers_mapped_correctly(self):
        assert _SECTOR_MAP["MU"] == "DRAM/Memory"
        assert _SECTOR_MAP["WDC"] == "DRAM/Memory"
        assert _SECTOR_MAP["STX"] == "DRAM/Memory"
        assert _SECTOR_MAP["ASML"] == "Semis Equipment"
        assert _SECTOR_MAP["LRCX"] == "Semis Equipment"
        assert _SECTOR_MAP["KLAC"] == "Semis Equipment"
        assert _SECTOR_MAP["AMAT"] == "Semis Equipment"
        assert _SECTOR_MAP["NVDA"] == "GPU/AI Compute"
        assert _SECTOR_MAP["AMD"] == "GPU/AI Compute"
        assert _SECTOR_MAP["AVGO"] == "GPU/AI Compute"

    def test_keys_uppercase_values_from_fixed_set(self):
        allowed = {
            "DRAM/Memory", "Semis Equipment", "GPU/AI Compute",
            "Foundry/Logic", "Mega-Cap Tech", "Networking/Optical",
            "EDA/IP", "Semis Index/ETF",
        }
        for tk, sec in _SECTOR_MAP.items():
            assert tk == tk.upper() and tk.isupper(), tk
            assert sec in allowed, (tk, sec)


class TestExtractTickers:
    def test_basic_extraction(self):
        assert _extract_tickers("MU earnings beat estimates") == {"MU"}
        assert _extract_tickers("NVDA and AMD rip on AI demand") == {
            "NVDA", "AMD"}

    def test_case_sensitive_uppercase_only(self):
        """Tickers in real headlines are uppercase; matching lowercase 'mu'
        would false-positive on ordinary prose. Documented tradeoff — lock
        it so it can't silently regress to case-insensitive noise."""
        assert _extract_tickers("the mu meson and amd processors") == set()

    def test_word_boundary_no_substring_false_positives(self):
        for s in ("EMU farms expand", "SAMUEL joined the board",
                  "RAMUS group news"):
            assert "MU" not in _extract_tickers(s), s

    def test_unknown_tickers_ignored_and_total(self):
        assert _extract_tickers("ZZZZ surges, TSLA flat") == set() | (
            {"TSLA"} if "TSLA" in _SECTOR_MAP else set())
        assert _extract_tickers("") == set()
        assert _extract_tickers(None) == set()
        assert _extract_tickers(12345) == set()


class TestAggregateSectorPulse:
    def test_groups_counts_and_avg_score(self):
        arts = [
            _a("MU DRAM pricing jumps", ai=8.0, age_h=1.0),
            _a("WDC NAND glut easing", ai=6.0, age_h=1.0),
            _a("NVDA HBM demand surges", ai=9.0, age_h=1.0),
        ]
        out = _aggregate_sector_pulse(arts, window_hours=24, now=NOW)
        secs = {s["sector"]: s for s in out["sectors"]}
        assert secs["DRAM/Memory"]["n_articles"] == 2
        assert secs["DRAM/Memory"]["avg_score"] == 7.0   # (8+6)/2
        assert secs["GPU/AI Compute"]["n_articles"] == 1
        assert out["n_scanned"] == 3
        assert out["n_mapped"] == 3
        assert out["window_hours"] == 24

    def test_velocity_favours_fresher_at_equal_count(self):
        """The whole point of a *pulse*: a sector lit by fresh wire outranks
        one with the same article count but stale items."""
        arts = [
            _a("MU news A", ai=5.0, age_h=0.2),
            _a("WDC news B", ai=5.0, age_h=0.2),     # DRAM: 2 fresh
            _a("NVDA news C", ai=5.0, age_h=40.0),
            _a("AMD news D", ai=5.0, age_h=40.0),    # GPU: 2 stale
        ]
        out = _aggregate_sector_pulse(arts, window_hours=48, now=NOW)
        order = [s["sector"] for s in out["sectors"]]
        assert order.index("DRAM/Memory") < order.index("GPU/AI Compute")
        dram = [s for s in out["sectors"] if s["sector"] == "DRAM/Memory"][0]
        gpu = [s for s in out["sectors"] if s["sector"] == "GPU/AI Compute"][0]
        assert dram["velocity"] > gpu["velocity"]

    def test_top_headline_is_highest_score_and_urgency_max(self):
        arts = [
            _a("MU minor note", ai=4.0, urg=0, age_h=1.0),
            _a("MU MAJOR: DRAM shortage declared", ai=9.5, urg=1, age_h=2.0),
        ]
        out = _aggregate_sector_pulse(arts, window_hours=24, now=NOW)
        dram = out["sectors"][0]
        assert dram["sector"] == "DRAM/Memory"
        assert dram["top_headline"] == "MU MAJOR: DRAM shortage declared"
        assert dram["max_urgency"] == 1
        assert dram["max_score"] == 9.5

    def test_multi_sector_article_counts_in_each(self):
        out = _aggregate_sector_pulse(
            [_a("MU supplies NVDA with HBM", ai=8.0, age_h=1.0)],
            window_hours=24, now=NOW)
        secs = {s["sector"] for s in out["sectors"]}
        assert {"DRAM/Memory", "GPU/AI Compute"} <= secs
        assert out["n_mapped"] == 1   # one article, counted once overall

    def test_unmapped_articles_scanned_not_mapped(self):
        out = _aggregate_sector_pulse(
            [_a("ZZZZ random unrelated headline", ai=7.0)],
            window_hours=24, now=NOW)
        assert out["n_scanned"] == 1
        assert out["n_mapped"] == 0
        assert out["sectors"] == []

    def test_total_contract_never_raises(self):
        for bad in (None, "nope", 42, [None, 1, "x"], [{"no_title": 1}]):
            out = _aggregate_sector_pulse(bad, window_hours=24, now=NOW)
            assert isinstance(out, dict)
            assert out["sectors"] == []
            assert isinstance(out["n_scanned"], int)


class TestSectorPulseChatLines:
    def test_compact_lines_for_top_sectors(self):
        arts = [
            _a("MU DRAM shortage", ai=9.0, urg=1, age_h=0.5),
            _a("WDC NAND up", ai=7.0, age_h=0.5),
            _a("NVDA HBM demand", ai=8.0, age_h=2.0),
        ]
        pulse = _aggregate_sector_pulse(arts, window_hours=24, now=NOW)
        lines = _sector_pulse_chat_lines(pulse)
        assert lines, "a non-empty pulse must yield chat lines"
        joined = "\n".join(lines)
        assert "DRAM/Memory" in joined
        # The hottest sector's lead headline is surfaced for the analyst.
        assert "MU DRAM shortage" in joined

    def test_total_contract(self):
        assert _sector_pulse_chat_lines(None) == []
        assert _sector_pulse_chat_lines("x") == []
        assert _sector_pulse_chat_lines({}) == []
        assert _sector_pulse_chat_lines({"sectors": []}) == []


def _insert(store, *, id, url, title, source, ai_score=5.0, urgency=0,
            age_min=5):
    fs = (datetime.now(timezone.utc) - timedelta(minutes=age_min)).isoformat()
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            "first_seen, cycle, ml_score, score_source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, "", 2.0, ai_score, urgency,
             fs, 0, None, None),
        )
        store.conn.commit()


class TestSectorPulseEndpoint:
    def test_endpoint_returns_sectors_and_excludes_backtest(
            self, store, monkeypatch):
        _insert(store, id="l1", url="https://x/1",
                title="MU DRAM pricing surges", source="rss", ai_score=8.0)
        _insert(store, id="l2", url="https://x/2",
                title="NVDA HBM demand strong", source="reuters",
                ai_score=9.0, urgency=1)
        # Synthetic backtest row — must NEVER reach this live surface.
        _insert(store, id="bt", url="backtest://run_3/2026-01-01/BUY/MU",
                title="SYNTHETIC SHOULD NOT SURFACE",
                source="backtest_run_3_winner", ai_score=10.0)

        monkeypatch.setattr(web_server, "_store", store, raising=False)
        app = web_server.create_app(store)
        client = app.test_client()
        resp = client.get("/api/sector-pulse?hours=24")

        assert resp.status_code == 200, resp.data
        data = resp.get_json()
        secs = {s["sector"]: s for s in data["sectors"]}
        assert "DRAM/Memory" in secs and "GPU/AI Compute" in secs
        # Backtest isolation invariant holds on the new read path.
        all_heads = " ".join(
            s["top_headline"] for s in data["sectors"])
        assert "SYNTHETIC" not in all_heads
        assert data["n_scanned"] == 2          # backtest row excluded by SQL

    def test_endpoint_window_param_clamped(self, store, monkeypatch):
        _insert(store, id="r1", url="https://x/r1",
                title="AMD share gains", source="rss", ai_score=6.0)
        monkeypatch.setattr(web_server, "_store", store, raising=False)
        app = web_server.create_app(store)
        client = app.test_client()
        # Absurd hours must be clamped, not error.
        resp = client.get("/api/sector-pulse?hours=999999")
        assert resp.status_code == 200, resp.data
        assert resp.get_json()["window_hours"] <= 168

"""Tests for analytics/decision_context.py + the read-only snapshot.

The live trader's `decisions` row stores only `action_taken` + `reasoning`;
nothing exposes *what Opus was shown* when it timed out / held. This builder
reconstructs the exact decision prompt (via the pure
`strategy._build_payload`, so it is byte-identical to the live one given
identical inputs) plus an input summary + feed_state — without ever calling
`_claude_call`.

Two contracts are locked here:
  * the pure builder's structure / counts / feed_state / truncation, and
  * `strategy.portfolio_snapshot_readonly` marks **identically** to
    `_portfolio_snapshot` (shared `_mark_to_market`) yet performs **no**
    store writes (so the dashboard thread can't corrupt the live trader).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import paper_trader.store as store_mod
from paper_trader import strategy
from paper_trader.analytics.decision_context import build_decision_context
from paper_trader.analytics.mark_integrity import build_mark_integrity
from paper_trader.store import Store


@pytest.fixture
def fresh_store(tmp_path, monkeypatch):
    db = tmp_path / "paper_trader.db"
    monkeypatch.setattr(store_mod, "DB_PATH", db)
    monkeypatch.setattr(store_mod, "_singleton", None)
    s = Store()
    try:
        yield s
    finally:
        s.close()


def _snapshot(positions=None):
    positions = positions if positions is not None else [{
        "id": 1, "ticker": "NVDA", "type": "stock", "qty": 2.0,
        "avg_cost": 100.0, "current_price": 110.0, "unrealized_pl": 20.0,
        "pl_pct": 10.0, "market_value": 220.0, "stale_mark": False,
    }]
    open_value = sum(p["market_value"] for p in positions)
    return {"cash": 50.0, "open_value": open_value,
            "total_value": 50.0 + open_value, "positions": positions}


def _ctx(**over):
    base = dict(
        snapshot=_snapshot(),
        merged_signals=[{"id": "a", "ai_score": 8.0, "urgency": 1,
                         "title": "NVDA surges on AI demand", "tickers": ["NVDA"]}],
        top_signals=[{"id": "a", "ai_score": 8.0, "urgency": 1,
                      "title": "NVDA surges on AI demand", "tickers": ["NVDA"]}],
        urgent=[],
        sentiments=[{"ticker": "NVDA", "avg_score": 7.0, "n": 3, "urgent": 1},
                    {"ticker": "AMD", "avg_score": 0.0, "n": 0, "urgent": 0}],
        watch_prices={"NVDA": 110.0, "AMD": None, "MU": None},
        futures_prices={"ES=F": 5000.0, "NQ=F": None},
        sp500=5800.0,
        market_open=True,
        quant_signals={"NVDA": {"RSI": 55.0}},
        self_review_block="SELF-REVIEW: payoff ratio 0.9",
        track_record_block=None,
        risk_mirror_block=None,
        ml_opinion_block=None,
    )
    base.update(over)
    return build_decision_context(**base)


class TestPromptFidelity:
    def test_prompt_has_the_live_payload_sections(self):
        r = _ctx()
        p = r["prompt"]
        # SYSTEM_PROMPT + the exact decide() framing
        assert "paper trading portfolio" in p
        assert "---\nCONTEXT:\n" in p
        # _build_payload's stable section headers (single source of truth)
        for header in ("PORTFOLIO:", "WATCHLIST PRICES:", "FUTURES:",
                       "TOP SCORED SIGNALS", "NO RISK LIMITS — full autonomy.",
                       "Return JSON only."):
            assert header in p, header
        # the self-review block was injected verbatim
        assert "SELF-REVIEW: payoff ratio 0.9" in p
        # the inspector never calls the model
        assert "raw" not in r and "decision" not in r
        assert r["claude_invoked"] is False

    def test_ml_advisor_only_when_block_present(self):
        assert "ML ADVISOR:" not in _ctx()["prompt"]
        r = _ctx(ml_opinion_block="ML MODEL OPINION: BUY NVDA")
        assert "---\nML ADVISOR:\nML MODEL OPINION: BUY NVDA" in r["prompt"]
        assert r["advisory_blocks"]["ml_opinion"] is True


class TestInputSummary:
    def test_counts_and_resolution_exact(self):
        r = _ctx()
        s = r["input_summary"]
        assert s["n_top_signals"] == 1
        assert s["n_urgent"] == 0
        assert s["n_merged_signals"] == 1
        # the exact value decide() records into decisions.signal_count
        assert s["signal_count"] == 1
        assert s["watchlist"] == {"n_total": 3, "n_resolved": 1,
                                  "n_missing": 2, "missing": ["AMD", "MU"]}
        assert s["futures"]["n_missing"] == 1
        assert s["sp500_resolved"] is True
        assert s["n_quant_tickers"] == 1
        assert s["n_sentiment_mentions"] == 1   # only NVDA has n>0

    def test_advisory_block_flags(self):
        r = _ctx(track_record_block="TR", risk_mirror_block=None)
        a = r["advisory_blocks"]
        assert a == {"self_review": True, "track_record": True,
                     "risk_mirror": False, "ml_opinion": False}

    def test_embeds_mark_integrity_verbatim(self):
        snap = _snapshot([
            {"id": 1, "ticker": "MU", "type": "stock", "qty": 0.5,
             "avg_cost": 724.12, "current_price": 724.12,
             "unrealized_pl": 0.0, "pl_pct": 0.0,
             "market_value": 362.06, "stale_mark": True},
        ])
        r = _ctx(snapshot=snap)
        assert r["mark_integrity"] == build_mark_integrity(snap["positions"])
        assert r["mark_integrity"]["verdict"] == "UNTRUSTWORTHY"  # 100% stale


class TestFeedState:
    def test_blind_when_no_signals(self):
        r = _ctx(merged_signals=[], top_signals=[])
        assert r["feed_state"] == "BLIND"
        assert "blind" in r["feed_headline"].lower()

    def test_degraded_when_half_watchlist_missing(self):
        # 1 resolved / 3 → 2 missing ≥ half → yfinance starvation
        r = _ctx(watch_prices={"NVDA": 1.0, "AMD": None, "MU": None})
        assert r["feed_state"] == "DEGRADED"

    def test_ok_when_signals_present_and_prices_resolve(self):
        r = _ctx(watch_prices={"NVDA": 1.0, "AMD": 2.0, "MU": 3.0})
        assert r["feed_state"] == "OK"


class TestTruncation:
    def test_bounded_prompt_with_honesty_keys(self):
        r = _ctx(max_prompt_chars=200)
        assert r["prompt_truncated"] is True
        assert r["prompt_chars"] > 200          # honest full length
        assert len(r["prompt"]) <= 200
        full = _ctx(max_prompt_chars=10_000_000)
        assert full["prompt_truncated"] is False
        assert full["prompt_chars"] == len(full["prompt"])


class TestReadonlySnapshotContract:
    """strategy.portfolio_snapshot_readonly: identical marks, zero writes."""

    def test_marks_identically_but_does_not_write(self, fresh_store, monkeypatch):
        s = fresh_store
        s.record_trade("NVDA", "BUY", 1, 100.0)
        s.upsert_position("NVDA", "stock", 1, 100.0)
        monkeypatch.setattr(strategy.market, "get_prices",
                             lambda tks: {"NVDA": 120.0})

        before = s.get_portfolio()
        ro = strategy.portfolio_snapshot_readonly(s)
        after_ro = s.get_portfolio()

        # read-only: the store is byte-for-byte unchanged
        assert after_ro == before
        # but the mark is computed live
        rp = ro["positions"][0]
        assert rp["current_price"] == 120.0
        assert rp["unrealized_pl"] == 20.0
        assert rp["stale_mark"] is False

        # the write-through path produces the *same* computed marks …
        full = strategy._portfolio_snapshot(s)
        fp = full["positions"][0]
        assert (fp["current_price"], fp["unrealized_pl"], fp["market_value"],
                fp["stale_mark"]) == (rp["current_price"], rp["unrealized_pl"],
                                      rp["market_value"], rp["stale_mark"])
        # … and *that* one did mutate the store (proves the contrast)
        assert s.get_portfolio() != before

    def test_readonly_stale_mark_matches_live(self, fresh_store, monkeypatch):
        s = fresh_store
        s.record_trade("MU", "BUY", 1, 724.12)
        s.upsert_position("MU", "stock", 1, 724.12)
        # yfinance returns nothing → fall back to cost, flagged stale
        monkeypatch.setattr(strategy.market, "get_prices", lambda tks: {})
        ro = strategy.portfolio_snapshot_readonly(s)
        p = ro["positions"][0]
        assert p["current_price"] == 724.12
        assert p["unrealized_pl"] == 0.0
        assert p["stale_mark"] is True

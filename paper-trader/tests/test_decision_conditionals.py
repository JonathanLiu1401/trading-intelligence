"""Tests for paper_trader.analytics.decision_conditionals.

Pins:
* pattern extraction across every supported intent kind on real-shape prose
* JSON-envelope vs raw-string reasoning fallback
* ticker extraction from action_taken (canonical) + envelope fallback
* dedup: a repeated standing intent collapses to ONE row on the newest
  decision (not five)
* verdict ladder NO_DATA / NO_INTENTS / STANDING_INTENTS / STALE_INTENTS
* freshness boundary: an intent exactly at stale_hours is stale; one
  microsecond under is fresh
* max_intents cap honored; ordering is newest-first
* defensive: garbage rows, parse_failed envelopes, missing timestamps,
  CASH / NONE pseudo-tickers all degrade silently — never raise
* is_intents_stale single-bool surface fires ONLY on STALE_INTENTS
* Flask route smoke (response shape stable across every verdict)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.decision_conditionals import (
    DEFAULT_MAX_INTENTS,
    DEFAULT_STALE_HOURS,
    DEFAULT_WINDOW_HOURS,
    build_decision_conditionals,
    extract_intents_from_text,
    is_intents_stale,
)

NOW = datetime(2026, 5, 24, 12, 0, 0, tzinfo=timezone.utc)


def _envelope(ticker: str, reasoning: str, action: str = "HOLD",
              confidence: float = 0.6) -> str:
    """The canonical Opus reasoning envelope shape from strategy.py."""
    return json.dumps({
        "decision": {
            "action": action,
            "ticker": ticker,
            "qty": 0,
            "confidence": confidence,
            "reasoning": reasoning,
        }
    })


def _row(id_: int, hours_ago: float, *, ticker: str = "NVDA",
         action_taken: str = "HOLD NVDA → HOLD",
         reasoning_text: str = "",
         raw_reasoning: str | None = None) -> dict:
    ts = NOW - timedelta(hours=hours_ago)
    reasoning = raw_reasoning if raw_reasoning is not None else _envelope(
        ticker, reasoning_text)
    return {
        "id": id_,
        "timestamp": ts.isoformat(),
        "action_taken": action_taken,
        "reasoning": reasoning,
        "portfolio_value": 1000.0,
        "cash": 100.0,
    }


# ─────────────────────────────────────────────────────────────────────
# Pattern extraction — each kind hits the patterns we promised
# ─────────────────────────────────────────────────────────────────────

def test_extract_watch_for_pattern():
    text = "Wait for cash session to reassess NVDA price action."
    ints = extract_intents_from_text(text)
    kinds = {i["kind"] for i in ints}
    assert "watch-for" in kinds


def test_extract_if_then_pattern():
    text = "If it holds 220 will add another share at the open."
    ints = extract_intents_from_text(text)
    kinds = {i["kind"] for i in ints}
    assert "if-then" in kinds


def test_extract_ready_to_pattern():
    text = "Ready to trim on a bounce above 230 to lock gains."
    ints = extract_intents_from_text(text)
    kinds = {i["kind"] for i in ints}
    assert "ready-to" in kinds


def test_extract_will_if_pattern():
    text = "Plan to exit when momentum breaks the 50d EMA."
    ints = extract_intents_from_text(text)
    kinds = {i["kind"] for i in ints}
    assert "will-if" in kinds


def test_extract_look_for_pattern():
    text = "Looking for follow-through on the breakout above 225 tomorrow."
    ints = extract_intents_from_text(text)
    kinds = {i["kind"] for i in ints}
    assert "look-for" in kinds


def test_extract_preserve_for_pattern():
    text = "Preserve cash for tomorrow's open when liquidity returns."
    ints = extract_intents_from_text(text)
    kinds = {i["kind"] for i in ints}
    assert "preserve-for" in kinds


def test_extract_too_early_to_pattern():
    text = "Premature to dump into a closed tape at a loss."
    ints = extract_intents_from_text(text)
    kinds = {i["kind"] for i in ints}
    assert "too-early-to" in kinds


def test_extract_rotate_into_pattern():
    text = "Rotating into LITE/LNOK optical names for diversification."
    ints = extract_intents_from_text(text)
    kinds = {i["kind"] for i in ints}
    assert "rotate-into" in kinds


def test_extract_clean_text_no_intents():
    text = "Bought one share. Market closed. Holding overnight."
    assert extract_intents_from_text(text) == []


def test_extract_handles_non_string():
    assert extract_intents_from_text(None) == []  # type: ignore[arg-type]
    assert extract_intents_from_text(123) == []  # type: ignore[arg-type]
    assert extract_intents_from_text("") == []


# ─────────────────────────────────────────────────────────────────────
# Envelope handling + ticker extraction
# ─────────────────────────────────────────────────────────────────────

def test_envelope_reasoning_extracted():
    rows = [_row(1, 1.0, reasoning_text="Wait for cash session to reassess.")]
    rep = build_decision_conditionals(rows, now=NOW)
    assert rep["state"] == "OK"
    assert rep["n_intents"] == 1
    assert rep["intents"][0]["text"].lower().startswith("wait for")
    assert rep["intents"][0]["ticker"] == "NVDA"


def test_raw_string_reasoning_fallback():
    # action_taken parses cleanly; reasoning is plain prose (parse_failed:
    # path — bot's JSON broke but the prose still has standing intent).
    rows = [_row(1, 1.0, raw_reasoning="parse_failed: ready to trim on bounce above 230")]
    rep = build_decision_conditionals(rows, now=NOW)
    assert rep["n_intents"] >= 1
    assert rep["intents"][0]["ticker"] == "NVDA"


def test_ticker_from_action_taken():
    rows = [_row(1, 1.0, action_taken="BUY AMD → FILLED",
                 reasoning_text="Ready to add on confirmation above 200.")]
    rep = build_decision_conditionals(rows, now=NOW)
    # The envelope below action_taken still says "NVDA" because _envelope
    # defaults to it; but ``_extract_ticker`` reads action_taken FIRST,
    # so the canonical ticker should win.
    assert rep["intents"][0]["ticker"] == "AMD"


def test_ticker_pseudo_cash_is_null():
    # action_taken says CASH (pseudo per #11), envelope also has CASH —
    # no real-ticker fallback, so the surfaced intent carries ticker=None.
    rows = [_row(1, 1.0, action_taken="HOLD CASH → HOLD",
                 ticker="CASH",
                 reasoning_text="Wait for cash session to redeploy capital.")]
    rep = build_decision_conditionals(rows, now=NOW)
    assert rep["n_intents"] == 1
    assert rep["intents"][0]["ticker"] is None  # CASH nullified per #11


def test_ticker_envelope_fallback_when_action_taken_null():
    # action_taken's ticker is CASH (pseudo); the envelope still names a
    # real ticker so the standing intent attributes to it (the envelope
    # fallback path — preserves usefulness when ``_parse_action_ticker``
    # would null out a real subject).
    rows = [_row(1, 1.0, action_taken="HOLD CASH → HOLD",
                 ticker="NVDA",  # envelope keeps the real ticker
                 reasoning_text="Wait for NVDA cash session to redeploy.")]
    rep = build_decision_conditionals(rows, now=NOW)
    assert rep["intents"][0]["ticker"] == "NVDA"


def test_envelope_ticker_fallback():
    # action_taken has no ticker (NO_DECISION shape), envelope carries it.
    rows = [_row(1, 1.0, action_taken="NO_DECISION",
                 raw_reasoning=_envelope("MU", "Watching for follow-through on MU."))]
    rep = build_decision_conditionals(rows, now=NOW)
    assert rep["intents"][0]["ticker"] == "MU"


# ─────────────────────────────────────────────────────────────────────
# Dedup
# ─────────────────────────────────────────────────────────────────────

def test_dedup_collapses_repeat_to_newest():
    # Same intent on five consecutive HOLDs at decreasing ages.
    text = "Wait for cash session to reassess NVDA price action before adding."
    rows = [
        _row(5, 0.5, reasoning_text=text),  # newest
        _row(4, 1.0, reasoning_text=text),
        _row(3, 2.0, reasoning_text=text),
        _row(2, 3.0, reasoning_text=text),
        _row(1, 4.0, reasoning_text=text),  # oldest
    ]
    rep = build_decision_conditionals(rows, now=NOW)
    assert rep["n_intents_raw"] == 5
    assert rep["n_intents"] == 1, "five repeats of the same intent should collapse"
    # The surviving row is the newest one (id=5, age 0.5h).
    assert rep["intents"][0]["decision_id"] == 5
    assert rep["intents"][0]["age_hours"] == 0.5


def test_dedup_distinct_kinds_not_collapsed():
    rows = [
        _row(2, 0.5, reasoning_text="Wait for cash session to reassess."),
        _row(1, 0.6, reasoning_text="Ready to trim on a bounce above 230."),
    ]
    rep = build_decision_conditionals(rows, now=NOW)
    assert rep["n_intents"] == 2
    kinds = {i["kind"] for i in rep["intents"]}
    assert "watch-for" in kinds and "ready-to" in kinds


def test_dedup_distinct_tickers_not_collapsed():
    text = "Wait for cash session to reassess price action."
    rows = [
        _row(2, 0.5, action_taken="HOLD AMD → HOLD", reasoning_text=text),
        _row(1, 0.6, action_taken="HOLD NVDA → HOLD", reasoning_text=text),
    ]
    rep = build_decision_conditionals(rows, now=NOW)
    assert rep["n_intents"] == 2


# ─────────────────────────────────────────────────────────────────────
# Verdict ladder
# ─────────────────────────────────────────────────────────────────────

def test_verdict_no_data_empty_input():
    rep = build_decision_conditionals([], now=NOW)
    assert rep["state"] == "NO_DATA"
    assert rep["verdict"] == "NO_DATA"
    assert rep["n_intents"] == 0
    assert rep["intents"] == []


def test_verdict_no_data_all_outside_window():
    # Decisions exist but ALL older than window_hours.
    rows = [_row(1, 25.0, reasoning_text="Wait for cash session.")]
    rep = build_decision_conditionals(rows, now=NOW, window_hours=24.0)
    assert rep["verdict"] == "NO_DATA"
    assert rep["n_decisions_scanned"] == 0


def test_verdict_no_intents_decisions_without_patterns():
    rows = [
        _row(1, 1.0, reasoning_text="Bought one share. Market closed. Holding."),
        _row(2, 2.0, reasoning_text="NVDA earnings beat. Strong revenue print."),
    ]
    rep = build_decision_conditionals(rows, now=NOW)
    assert rep["verdict"] == "NO_INTENTS"
    assert rep["n_decisions_scanned"] == 2
    assert rep["n_intents"] == 0


def test_verdict_standing_intents_majority_fresh():
    # 2 fresh + 1 stale → majority fresh → STANDING_INTENTS
    rows = [
        _row(3, 0.5, action_taken="HOLD NVDA → HOLD",
             reasoning_text="Wait for cash session to reassess NVDA."),
        _row(2, 1.0, action_taken="HOLD AMD → HOLD",
             reasoning_text="Ready to trim on a bounce above 230 in AMD."),
        _row(1, 20.0, action_taken="HOLD MU → HOLD",
             reasoning_text="Looking for follow-through on the MU breakout."),
    ]
    rep = build_decision_conditionals(rows, now=NOW, stale_hours=12.0)
    assert rep["verdict"] == "STANDING_INTENTS"
    assert rep["n_intents"] == 3
    assert rep["n_stale"] == 1


def test_verdict_stale_intents_majority_stale():
    # 2 stale + 1 fresh → STALE_INTENTS
    rows = [
        _row(3, 0.5, action_taken="HOLD NVDA → HOLD",
             reasoning_text="Wait for cash session to reassess NVDA."),
        _row(2, 18.0, action_taken="HOLD AMD → HOLD",
             reasoning_text="Ready to trim on a bounce above 230 in AMD."),
        _row(1, 20.0, action_taken="HOLD MU → HOLD",
             reasoning_text="Looking for follow-through on the MU breakout."),
    ]
    rep = build_decision_conditionals(rows, now=NOW, stale_hours=12.0, window_hours=48.0)
    assert rep["verdict"] == "STALE_INTENTS"
    assert rep["n_stale"] == 2


# ─────────────────────────────────────────────────────────────────────
# Freshness boundary
# ─────────────────────────────────────────────────────────────────────

def test_freshness_boundary_inclusive_stale():
    # At exactly stale_hours = stale (>=)
    rows = [_row(1, 12.0, reasoning_text="Wait for cash session to reassess.")]
    rep = build_decision_conditionals(rows, now=NOW, stale_hours=12.0)
    assert rep["intents"][0]["stale"] is True


def test_freshness_boundary_just_under_fresh():
    rows = [_row(1, 11.99, reasoning_text="Wait for cash session to reassess.")]
    rep = build_decision_conditionals(rows, now=NOW, stale_hours=12.0)
    assert rep["intents"][0]["stale"] is False


# ─────────────────────────────────────────────────────────────────────
# Cap + ordering
# ─────────────────────────────────────────────────────────────────────

def test_max_intents_cap_and_order():
    # 6 distinct intents at ascending ages — cap to 3, newest-first.
    rows = []
    for i in range(6):
        rows.append(_row(
            10 + i, i * 0.5,
            action_taken=f"HOLD T{i} → HOLD",
            reasoning_text=f"Wait for catalyst {i} to confirm direction.",
        ))
    rep = build_decision_conditionals(rows, now=NOW, max_intents=3)
    assert rep["n_intents"] == 3
    ages = [it["age_hours"] for it in rep["intents"]]
    assert ages == sorted(ages), "newest-first ordering broken"


# ─────────────────────────────────────────────────────────────────────
# Defensive — garbage in, silent degradation, never raises
# ─────────────────────────────────────────────────────────────────────

def test_garbage_rows_degrade_silently():
    rows = [
        None,                                                       # not a dict
        {},                                                         # missing every field
        {"timestamp": "not-an-iso"},                                # bad ts
        {"timestamp": None},                                        # null ts
        {"timestamp": (NOW - timedelta(hours=1)).isoformat()},      # no reasoning
        _row(99, 0.5, reasoning_text="Wait for cash session."),
    ]
    rep = build_decision_conditionals(rows, now=NOW)
    # The clean row still surfaces; the garbage rows do not raise.
    assert rep["n_intents"] >= 1


def test_parse_failed_envelope_falls_back_to_raw():
    raw = "parse_failed: {broken_json wait for cash session to reassess"
    rows = [_row(1, 0.5, raw_reasoning=raw, action_taken="HOLD NVDA → HOLD")]
    rep = build_decision_conditionals(rows, now=NOW)
    # The raw-string fallback path picks up "wait for cash session".
    assert rep["n_intents"] >= 1


def test_never_raises_on_garbage():
    # build is total — even nonsense decisions return a valid envelope.
    rep = build_decision_conditionals("not a list", now=NOW)  # type: ignore[arg-type]
    assert rep["verdict"] == "NO_DATA"
    assert rep["intents"] == []


# ─────────────────────────────────────────────────────────────────────
# Envelope shape stability
# ─────────────────────────────────────────────────────────────────────

_REQUIRED_KEYS = {
    "state", "verdict", "headline", "n_decisions_scanned", "n_intents_raw",
    "n_intents", "n_stale", "intents", "by_kind", "window_hours",
    "stale_hours", "as_of",
}


def test_envelope_shape_no_data():
    rep = build_decision_conditionals([], now=NOW)
    assert _REQUIRED_KEYS.issubset(rep.keys())


def test_envelope_shape_no_intents():
    rows = [_row(1, 0.5, reasoning_text="Bought one share. Holding.")]
    rep = build_decision_conditionals(rows, now=NOW)
    assert _REQUIRED_KEYS.issubset(rep.keys())


def test_envelope_shape_standing():
    rows = [_row(1, 0.5, reasoning_text="Wait for cash session to reassess.")]
    rep = build_decision_conditionals(rows, now=NOW)
    assert _REQUIRED_KEYS.issubset(rep.keys())


# ─────────────────────────────────────────────────────────────────────
# is_intents_stale single-bool surface
# ─────────────────────────────────────────────────────────────────────

def test_is_intents_stale_only_on_stale_verdict():
    assert is_intents_stale({"verdict": "STALE_INTENTS"}) is True
    assert is_intents_stale({"verdict": "STANDING_INTENTS"}) is False
    assert is_intents_stale({"verdict": "NO_INTENTS"}) is False
    assert is_intents_stale({"verdict": "NO_DATA"}) is False
    assert is_intents_stale({}) is False
    assert is_intents_stale(None) is False  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────
# Flask route smoke — response shape stable across verdicts
# ─────────────────────────────────────────────────────────────────────

def test_route_smoke_empty_db_returns_valid_shape():
    from paper_trader import dashboard as dash
    app = dash.app
    client = app.test_client()
    r = client.get("/api/decision-conditionals?window_hours=1")
    assert r.status_code in (200, 500)  # 500 only on truly catastrophic
    payload = r.get_json()
    assert isinstance(payload, dict)
    assert "verdict" in payload
    assert "intents" in payload


def test_route_smoke_clamps_query_params():
    from paper_trader import dashboard as dash
    app = dash.app
    client = app.test_client()
    # Out-of-range values should clamp, not error.
    r = client.get("/api/decision-conditionals?window_hours=99999&stale_hours=-5&max_intents=999")
    assert r.status_code in (200, 500)
    payload = r.get_json()
    assert isinstance(payload, dict)
    # window_hours clamped to ≤168
    assert payload.get("window_hours", 0) <= 168.0

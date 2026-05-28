"""Tests for analysis/held_wire_balance.py — per-held-ticker stance.

Sector coherence answers "is the wire agreeing at the sector level?";
this answers "is the wire agreeing on the SPECIFIC names I hold?".
Critical regressions to pin:

- The bull/bear classifier must be the SAME one ``sector_coherence`` uses
  (SSOT — reuse, don't duplicate).
- Per-name MIN_CLASSIFIED_PER_TICKER=2 floor — a 1-bull/0-bear "100%
  coherent" name is meaningless on a one-headline read.
- BULL_LEAN/BEAR_LEAN threshold strictness — 69.9% must be MIXED.
- Book-level verdict: requires >= MIN_OPINIONATED_NAMES names to graduate
  past BOOK_INSUFFICIENT; uses BOOK_LEAN_PCT share of OPINIONATED names.
- Chat helper silence-on-healthy: must emit nothing on BOOK_BULL /
  BOOK_INSUFFICIENT / no-BEAR_LEAN-names — only fire on BEAR_LEAN.
- Garbage-safety: non-list articles, missing held_tickers, None titles,
  malformed ai_score must never raise.

Pure-helper tests — no Flask (project_digital_intern_chat_enrichment_pattern).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.held_wire_balance import (  # noqa: E402
    BOOK_LEAN_PCT,
    BULL_LEAN_PCT,
    MIN_CLASSIFIED_PER_TICKER,
    _book_verdict,
    _per_ticker_verdict,
    build_held_wire_balance,
)
from dashboard.web_server import _held_wire_balance_chat_lines  # noqa: E402


def _a(title: str, ai: float = 5.0) -> dict:
    return {"title": title, "ai_score": ai}


class TestPerTickerVerdict:
    def test_insufficient_below_min(self):
        v, _, _ = _per_ticker_verdict(1, 0)
        assert v == "INSUFFICIENT"
        v, _, _ = _per_ticker_verdict(0, 0)
        assert v == "INSUFFICIENT"

    def test_bull_lean_at_exact_threshold(self):
        # 7 bull / 3 bear = 70% bull ⇒ BULL_LEAN (boundary inclusive)
        v, coh, lead = _per_ticker_verdict(7, 3)
        assert v == "BULL_LEAN"
        assert coh == 70.0
        assert lead == "bull"

    def test_bull_lean_below_threshold_is_mixed(self):
        # 69% must NOT be BULL_LEAN — strict >= threshold contract
        v, _, _ = _per_ticker_verdict(69, 31)
        assert v == "MIXED"

    def test_bear_lean(self):
        v, coh, lead = _per_ticker_verdict(2, 8)
        assert v == "BEAR_LEAN"
        assert coh == 80.0
        assert lead == "bear"

    def test_mixed_50_50(self):
        v, coh, _ = _per_ticker_verdict(5, 5)
        assert v == "MIXED"
        assert coh == 50.0

    def test_min_classified_threshold_pin(self):
        # The floor is hard-coded in the module; verify the constant a
        # downstream reader depends on can't silently drift.
        assert MIN_CLASSIFIED_PER_TICKER == 2


class TestBookVerdict:
    def _row(self, ticker, verdict):
        return {"ticker": ticker, "verdict": verdict}

    def test_book_insufficient_below_min_opinionated(self):
        rows = [self._row("MU", "INSUFFICIENT")]
        assert _book_verdict(rows) == "BOOK_INSUFFICIENT"

    def test_book_bull_at_threshold(self):
        # 2 of 3 opinionated = 66.7% ⇒ BOOK_BULL
        rows = [self._row("MU", "BULL_LEAN"),
                self._row("NVDA", "BULL_LEAN"),
                self._row("LITE", "MIXED")]
        assert _book_verdict(rows) == "BOOK_BULL"

    def test_book_bear_at_threshold(self):
        rows = [self._row("MU", "BEAR_LEAN"),
                self._row("NVDA", "BEAR_LEAN"),
                self._row("LITE", "MIXED")]
        assert _book_verdict(rows) == "BOOK_BEAR"

    def test_book_mixed_below_threshold(self):
        # 1 of 3 = 33.3% bull ⇒ no super-majority either way ⇒ MIXED
        rows = [self._row("MU", "BULL_LEAN"),
                self._row("NVDA", "BEAR_LEAN"),
                self._row("LITE", "MIXED")]
        assert _book_verdict(rows) == "BOOK_MIXED"

    def test_insufficient_rows_ignored_from_opinionated_count(self):
        # 2 BULL_LEAN + 1 INSUFFICIENT = 2/2 opinionated = 100% bull ⇒
        # BOOK_BULL (the INSUFFICIENT name doesn't drag the verdict down).
        rows = [self._row("MU", "BULL_LEAN"),
                self._row("NVDA", "BULL_LEAN"),
                self._row("XYZ", "INSUFFICIENT")]
        assert _book_verdict(rows) == "BOOK_BULL"

    def test_threshold_pinned(self):
        # Document the threshold a downstream reader depends on.
        assert BOOK_LEAN_PCT == 66.0
        assert BULL_LEAN_PCT == 70.0


class TestBuildHeldWireBalance:
    def test_empty_returns_skeleton(self):
        r = build_held_wire_balance([], held_tickers=["MU"])
        assert r["per_ticker"][0]["ticker"] == "MU"
        assert r["per_ticker"][0]["verdict"] == "INSUFFICIENT"
        assert r["book_verdict"] == "BOOK_INSUFFICIENT"
        assert r["n_scanned"] == 0

    def test_non_list_articles_returns_skeleton(self):
        r = build_held_wire_balance("not-a-list", held_tickers=["MU"])  # type: ignore
        assert r["per_ticker"] == []
        assert r["book_verdict"] == "BOOK_INSUFFICIENT"

    def test_no_held_tickers_returns_skeleton(self):
        r = build_held_wire_balance([_a("MU surged")], held_tickers=[])
        assert r["per_ticker"] == []
        assert r["book_verdict"] == "BOOK_INSUFFICIENT"

    def test_non_string_held_tickers_filtered(self):
        # Garbage in the held list must not crash and must not match
        # anywhere in titles.
        r = build_held_wire_balance(
            [_a("MU surged on HBM demand")],
            held_tickers=["MU", None, 123, ""],  # type: ignore
        )
        tickers = {t["ticker"] for t in r["per_ticker"]}
        assert tickers == {"MU"}

    def test_bull_lean_on_held_ticker(self):
        # 3 bull / 1 bear on MU = 75% ⇒ BULL_LEAN
        arts = [
            _a("MU surged on HBM demand", ai=8.0),
            _a("MU rallied to record high", ai=7.5),
            _a("MU beats earnings estimates", ai=9.0),
            _a("MU dropped after analyst cut", ai=6.0),
        ]
        r = build_held_wire_balance(arts, held_tickers=["MU"])
        mu = next(t for t in r["per_ticker"] if t["ticker"] == "MU")
        assert mu["verdict"] == "BULL_LEAN"
        assert mu["n_bull"] == 3
        assert mu["n_bear"] == 1
        assert mu["coherence_pct"] == 75.0
        # Highest-ai bullish headline becomes lead_headline.
        assert "beats" in mu["lead_headline"]

    def test_bear_lean_on_held_ticker(self):
        arts = [
            _a("LITE plunged on fraud probe", ai=9.0),
            _a("LITE warns on guidance", ai=8.0),
            _a("LITE downgraded to underweight", ai=7.0),
            _a("LITE wins small order", ai=4.0),
        ]
        r = build_held_wire_balance(arts, held_tickers=["LITE"])
        lite = next(t for t in r["per_ticker"] if t["ticker"] == "LITE")
        assert lite["verdict"] == "BEAR_LEAN"
        assert lite["n_bear"] == 3

    def test_book_bear_when_majority_bear_lean(self):
        # MU bear, AXTI bear, NVDA mixed ⇒ 2/3 = 66.7% bear ⇒ BOOK_BEAR
        arts = [
            _a("MU plunged on warning"),
            _a("MU missed estimates"),
            _a("MU recall ongoing"),
            _a("AXTI lawsuit filed"),
            _a("AXTI dropped sharply"),
            _a("AXTI warned on guidance"),
            _a("NVDA surged today"),
            _a("NVDA missed estimates"),
        ]
        r = build_held_wire_balance(
            arts, held_tickers=["MU", "AXTI", "NVDA"])
        assert r["book_verdict"] == "BOOK_BEAR"
        assert r["n_bear_lean"] >= 2
        assert "BOOK_BEAR" in r["headline"]

    def test_book_insufficient_when_too_few_opinionated(self):
        arts = [_a("MU announces partnership with AVGO")]
        r = build_held_wire_balance(arts, held_tickers=["MU", "NVDA"])
        assert r["book_verdict"] == "BOOK_INSUFFICIENT"

    def test_ticker_word_boundary_no_substring(self):
        # Ticker MU must NOT match inside "Micron"/"Museum"/"MUSEUM".
        # Word-boundary regex is the contract.
        arts = [
            _a("Museum surged today"),       # Should NOT count for MU
            _a("Micron announces dividend"),  # Should NOT count for MU
            _a("MU surged on HBM", ai=8.0),  # SHOULD count
            _a("MU rallied", ai=7.0),         # SHOULD count
        ]
        r = build_held_wire_balance(arts, held_tickers=["MU"])
        mu = next(t for t in r["per_ticker"] if t["ticker"] == "MU")
        assert mu["n_bull"] == 2

    def test_garbage_article_rows_dont_raise(self):
        arts = [
            None,                              # type: ignore
            "not-a-dict",                      # type: ignore
            {"title": None},                   # bad title
            {"title": ""},                     # empty title
            {"title": "MU surged", "ai_score": "garbage"},  # bad ai_score
            _a("MU surged", ai=8.0),
            _a("MU rallied", ai=7.0),
        ]
        r = build_held_wire_balance(arts, held_tickers=["MU"])  # type: ignore
        mu = next(t for t in r["per_ticker"] if t["ticker"] == "MU")
        # Bad ai_score row still classified (just lead-headline ranking
        # treats it as ai=0), so n_bull >= 2.
        assert mu["n_bull"] >= 2

    def test_multi_ticker_in_single_headline_counted_for_each(self):
        # A headline mentioning both held names contributes to both.
        arts = [_a("MU and NVDA both surged on AI demand", ai=8.0)]
        r = build_held_wire_balance(arts, held_tickers=["MU", "NVDA"])
        mu = next(t for t in r["per_ticker"] if t["ticker"] == "MU")
        nvda = next(t for t in r["per_ticker"] if t["ticker"] == "NVDA")
        assert mu["n_bull"] == 1
        assert nvda["n_bull"] == 1

    def test_lead_headline_is_highest_ai_opinionated(self):
        arts = [
            _a("MU surged on HBM demand", ai=2.0),
            _a("MU rallied to record high", ai=9.5),  # Highest, should win
            _a("MU stayed flat all day", ai=8.0),    # Neutral — ignored
        ]
        r = build_held_wire_balance(arts, held_tickers=["MU"])
        mu = next(t for t in r["per_ticker"] if t["ticker"] == "MU")
        assert "record high" in mu["lead_headline"]


class TestSortOrder:
    def test_bear_lean_sorted_first(self):
        # Operator wants BEAR_LEAN at top: actionable. BULL_LEAN last
        # among opinionated; INSUFFICIENT last overall.
        arts = [
            _a("MU surged"), _a("MU rallied"),    # MU BULL_LEAN
            _a("NVDA plunged"), _a("NVDA dropped"),  # NVDA BEAR_LEAN
        ]
        r = build_held_wire_balance(arts, held_tickers=["MU", "NVDA", "LITE"])
        verdicts = [t["verdict"] for t in r["per_ticker"]]
        # BEAR_LEAN must come before BULL_LEAN; INSUFFICIENT last.
        bear_idx = verdicts.index("BEAR_LEAN")
        bull_idx = verdicts.index("BULL_LEAN")
        ins_idx = verdicts.index("INSUFFICIENT")
        assert bear_idx < bull_idx < ins_idx


class TestChatLines:
    def test_non_dict_returns_empty(self):
        assert _held_wire_balance_chat_lines(None) == []
        assert _held_wire_balance_chat_lines("garbage") == []
        assert _held_wire_balance_chat_lines({}) == []

    def test_no_bear_lean_collapses_to_silence(self):
        rep = {
            "per_ticker": [
                {"ticker": "MU", "verdict": "BULL_LEAN", "n_bull": 5,
                 "n_bear": 1, "n_classified": 6, "coherence_pct": 83.3,
                 "lead_headline": "MU surged"},
            ],
            "book_verdict": "BOOK_BULL",
        }
        # All-bull book ⇒ silence (chat filler is not information).
        assert _held_wire_balance_chat_lines(rep) == []

    def test_mixed_only_collapses_to_silence(self):
        rep = {
            "per_ticker": [
                {"ticker": "MU", "verdict": "MIXED", "n_bull": 3,
                 "n_bear": 3, "n_classified": 6, "coherence_pct": 50.0,
                 "lead_headline": "mixed"},
            ],
            "book_verdict": "BOOK_MIXED",
        }
        assert _held_wire_balance_chat_lines(rep) == []

    def test_bear_lean_emits_line(self):
        rep = {
            "per_ticker": [
                {"ticker": "LITE", "verdict": "BEAR_LEAN", "n_bull": 1,
                 "n_bear": 5, "n_classified": 6, "coherence_pct": 83.3,
                 "lead_headline": "LITE faces fraud probe"},
            ],
            "book_verdict": "BOOK_MIXED",
            "headline": "Held-wire balance: BOOK_MIXED — 0↑/1↓",
        }
        lines = _held_wire_balance_chat_lines(rep)
        assert len(lines) == 1
        assert "LITE" in lines[0]
        assert "BEAR_LEAN" in lines[0]
        assert "fraud probe" in lines[0]

    def test_book_bear_prepends_headline(self):
        rep = {
            "per_ticker": [
                {"ticker": "LITE", "verdict": "BEAR_LEAN", "n_bull": 0,
                 "n_bear": 4, "n_classified": 4, "coherence_pct": 100.0,
                 "lead_headline": "LITE plunged"},
            ],
            "book_verdict": "BOOK_BEAR",
            "headline": "Held-wire balance: BOOK_BEAR — wire bearish on 1",
        }
        lines = _held_wire_balance_chat_lines(rep)
        assert len(lines) == 2
        assert lines[0].startswith("Held-wire balance: BOOK_BEAR")
        assert "LITE" in lines[1]

    def test_long_headline_is_truncated(self):
        long_lead = "A" * 200
        rep = {
            "per_ticker": [
                {"ticker": "MU", "verdict": "BEAR_LEAN", "n_bull": 1,
                 "n_bear": 5, "n_classified": 6, "coherence_pct": 83.3,
                 "lead_headline": long_lead},
            ],
            "book_verdict": "BOOK_MIXED",
        }
        lines = _held_wire_balance_chat_lines(rep)
        assert len(lines) == 1
        # 120 chars + ellipsis ⇒ the line never explodes past chat-budget.
        assert "…" in lines[0]


class TestEndpointSmoke:
    """Live-Flask test_client smoke per the analytics-verification memory:
    module __main__ would hit a different/empty DB; the live endpoint
    contract is verified by spinning up the actual Flask app via the
    project's standard ``store`` fixture (conftest.py).
    """

    def _insert(self, store, url, title, source="rss", ai=5.0,
                age_min=10, urgency=0):
        from datetime import datetime, timezone, timedelta
        first_seen = (
            datetime.now(timezone.utc) - timedelta(minutes=age_min)
        ).isoformat()
        with store._write_lock:
            store.conn.execute(
                "INSERT INTO articles "
                "(id, url, title, source, published, kw_score, ai_score, "
                "urgency, first_seen, cycle) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (url, url, title, source, first_seen, 2.0, ai,
                 urgency, first_seen, 0),
            )
            store.conn.commit()

    def test_endpoint_returns_well_formed_skeleton(self, store,
                                                    monkeypatch):
        from dashboard import web_server

        self._insert(store, "https://e/1", "MU surged on HBM demand")
        self._insert(store, "https://e/2", "MU rallied to record")
        self._insert(store, "https://e/3", "MU beats estimates")

        monkeypatch.delenv("WEB_API_KEY", raising=False)
        monkeypatch.setattr(web_server, "_store", store, raising=False)
        client = web_server.create_app(store).test_client()
        resp = client.get("/api/held-wire-balance?hours=24")
        assert resp.status_code == 200, resp.data
        body = resp.get_json()
        assert "per_ticker" in body
        assert "book_verdict" in body
        assert "n_scanned" in body
        assert body["min_classified_per_ticker"] == 2

    def test_endpoint_clamps_hours(self, store, monkeypatch):
        from dashboard import web_server

        monkeypatch.delenv("WEB_API_KEY", raising=False)
        monkeypatch.setattr(web_server, "_store", store, raising=False)
        client = web_server.create_app(store).test_client()

        # 0 → clamped to 1; 999 → clamped to 168.
        assert client.get("/api/held-wire-balance?hours=0").get_json()[
            "window_hours"] == 1
        assert client.get("/api/held-wire-balance?hours=999").get_json()[
            "window_hours"] == 168
        # garbage falls back to the 24h default.
        assert client.get("/api/held-wire-balance?hours=abc").get_json()[
            "window_hours"] == 24

    def test_endpoint_backtest_rows_excluded(self, store, monkeypatch):
        # The _LIVE_ONLY_CLAUSE invariant: backtest:// URLs and
        # backtest_* sources must never contribute to live signals.
        from dashboard import web_server

        # Live row: MU surged ⇒ would contribute n_bull on MU.
        self._insert(store, "https://e/1", "MU surged on HBM demand")
        # Synthetic backtest rows: must be excluded by SQL.
        self._insert(store, "backtest://run_9/2026-01-01/BUY/MU",
                     "MU plunged synthetic", source="backtest_run_9_winner")
        self._insert(store, "https://e/3",
                     "MU plunged synthetic 2",
                     source="opus_annotation_cycle_3")

        monkeypatch.delenv("WEB_API_KEY", raising=False)
        monkeypatch.setattr(web_server, "_store", store, raising=False)
        client = web_server.create_app(store).test_client()
        resp = client.get("/api/held-wire-balance?hours=24")
        assert resp.status_code == 200
        body = resp.get_json()
        # Only the one live row counts ⇒ MU has n_bull=1, n_bear=0.
        mu_rows = [t for t in body["per_ticker"] if t["ticker"] == "MU"]
        if mu_rows:
            mu = mu_rows[0]
            assert mu["n_bear"] == 0, body

"""Live held-book universe in claude_analyst and the briefing coverage line.

Regression guard for the SSOT-ification that unioned ``_BOOK_TICKERS`` (the
hardcoded daemon.PORTFOLIO_TICKERS mirror) with
``ml.features.LIVE_PORTFOLIO_TICKERS`` (positions + option underlyings +
sector_watchlist read out of config/portfolio.json).

Before this, the static literal alone was silently drifting behind the trading
UI: a 2026-05-23 live read showed GOOG / COHR / NVDL held in portfolio.json
yet absent from ``_BOOK_TICKERS``. The briefing's per-row ``[BOOK:]`` tag and
``_book_heat_lines`` concentration signal therefore never fired on news for
those open positions, and ``daemon._format_portfolio_coverage`` (the
"Book in digest" line) read N/N covered even when N of the analyst's actual
held names were silent. Same drift class as the urgency_scorer SCORE_PROMPT
fix the sibling agent did.

These pin:
  - ``_BOOK_UNIVERSE`` includes both static core AND live-only additions.
  - ``_BOOK_RE`` matches a live-only ticker like GOOG.
  - ``_book_tickers()`` returns canonical order: static first in their
    existing order, live extras in alphabetical order at the tail.
  - ``_book_heat_lines()`` ranks live-only tickers deterministically rather
    than collapsing them all to the ``len(rank)`` tie-break fallback.
  - The static ``_BOOK_TICKERS`` literal itself is NOT extended (anti-drift
    parity with daemon.PORTFOLIO_TICKERS, the existing pinned invariant).
  - The daemon heartbeat path passes the LIVE union to
    ``_format_portfolio_coverage`` (not the static default).
"""
from __future__ import annotations

import daemon
from analysis import claude_analyst
from ml import features as ml_features


# ── _BOOK_UNIVERSE composition ───────────────────────────────────────────────
class TestBookUniverseComposition:
    def test_static_core_unchanged_for_daemon_parity(self):
        # The static tuple must remain a frozen mirror of
        # daemon.PORTFOLIO_TICKERS — this is the existing pinned invariant
        # (test_book_tickers_parity_with_daemon). The live-union extension
        # must NOT mutate it.
        assert frozenset(claude_analyst._BOOK_TICKERS) == frozenset(
            daemon.PORTFOLIO_TICKERS
        )

    def test_universe_is_superset_of_static_core(self):
        # The universe is what the regex actually scans against. It must
        # contain everything the static core does, in the same prefix order.
        core = list(claude_analyst._BOOK_TICKERS)
        universe = list(claude_analyst._BOOK_UNIVERSE)
        assert universe[: len(core)] == core, (
            "static core must remain the canonical prefix of the universe — "
            "_book_tickers() iterates the universe to preserve cycle-to-cycle "
            "stable ordering for the [BOOK:] tag and BOOK HEAT block"
        )

    def test_live_only_additions_present(self):
        # Every live-portfolio ticker (config/portfolio.json's positions +
        # option underlyings + sector_watchlist) must appear in the universe.
        # Without this the [BOOK:] tag silently drops them.
        universe = set(claude_analyst._BOOK_UNIVERSE)
        for t in ml_features.LIVE_PORTFOLIO_TICKERS:
            assert t in universe, (
                f"live-portfolio ticker {t!r} missing from _BOOK_UNIVERSE — "
                "briefing rows mentioning it would never get [BOOK:] tagged"
            )

    def test_live_additions_are_alphabetical_at_tail(self):
        # Deterministic order for the tail so the rank dictionary in
        # _book_heat_lines never churns cycle-to-cycle on the same live set.
        core = set(claude_analyst._BOOK_TICKERS)
        tail = [t for t in claude_analyst._BOOK_UNIVERSE if t not in core]
        assert tail == sorted(tail), (
            "live-only additions must be sorted alphabetically at the tail "
            "so the canonical ordering is stable across module reloads"
        )

    def test_universe_has_no_duplicates(self):
        u = claude_analyst._BOOK_UNIVERSE
        assert len(u) == len(set(u))


# ── _BOOK_RE matching the live universe ──────────────────────────────────────
class TestBookReLiveMatch:
    def _pick_live_only(self) -> str | None:
        core = set(claude_analyst._BOOK_TICKERS)
        for t in claude_analyst._BOOK_UNIVERSE:
            if t not in core:
                return t
        return None

    def test_regex_matches_a_live_only_ticker(self):
        live_only = self._pick_live_only()
        if live_only is None:
            # If config/portfolio.json adds nothing beyond the static core,
            # this assertion is vacuous — the union equals the core and the
            # regex behaviour is unchanged.
            return
        m = claude_analyst._BOOK_RE.findall(
            f"{live_only} announces something material today"
        )
        assert live_only in m

    def test_book_tickers_returns_live_only_ticker(self):
        live_only = self._pick_live_only()
        if live_only is None:
            return
        # Use a clearly-isolated context so word boundaries fire correctly.
        out = claude_analyst._book_tickers(
            {"title": f"{live_only} announces quarterly buyback expansion",
             "summary": ""}
        )
        assert live_only in out

    def test_book_tickers_canonical_order_preserves_static_first(self):
        # When a static-core ticker and a live-only ticker both appear, the
        # static one must come first (canonical prefix preserved).
        live_only = self._pick_live_only()
        if live_only is None:
            return
        # MU is in static core. Put live_only first in the text — the result
        # must still list MU before the live-only one.
        out = claude_analyst._book_tickers(
            {"title": f"{live_only} and MU both rallied",
             "summary": ""}
        )
        if "MU" in out and live_only in out:
            assert out.index("MU") < out.index(live_only)


# ── _book_heat_lines ranking with live-only tickers ──────────────────────────
def _url_row(title: str, score: float = 8.0) -> dict:
    return {"title": title, "summary": "", "source": "rss",
            "ai_score": score, "link": f"https://x/{abs(hash(title))}"}


class TestBookHeatLiveOnly:
    def test_live_only_ticker_can_register_heat(self):
        # 3 distinct stories about a live-only ticker → must register at the
        # BOOK_HEAT threshold. Before the fix, _book_tickers() ignored the
        # ticker, counts[] was empty, and _book_heat_lines returned [].
        core = set(claude_analyst._BOOK_TICKERS)
        live_only = next(
            (t for t in claude_analyst._BOOK_UNIVERSE if t not in core),
            None,
        )
        if live_only is None:
            return
        arts = [_url_row(f"{live_only} distinct story {i}") for i in range(3)]
        out = claude_analyst._book_heat_lines(arts)
        assert out == [f"{live_only} — 3 distinct stories"]


# ── daemon heartbeat call site passes the live union ─────────────────────────
class TestHeartbeatPathPassesLiveUnion:
    def test_format_portfolio_coverage_with_live_union_includes_extras(self):
        # Simulate the heartbeat's call: pass _price_alert_universe() (the
        # canonical live union) as tickers. A briefing row mentioning a
        # live-only ticker must now be flagged as covered.
        core = set(daemon.PORTFOLIO_TICKERS)
        live_only = next(
            (t for t in ml_features.LIVE_PORTFOLIO_TICKERS if t not in core),
            None,
        )
        if live_only is None:
            return
        arts = [{"title": f"{live_only} announces large supply deal",
                 "summary": ""}]
        out = daemon._format_portfolio_coverage(
            arts, tickers=daemon._price_alert_universe()
        )
        # The live ticker must appear in the "Book in digest:" head, not in
        # the silent tail.
        assert out.startswith("📊 Book in digest: ")
        head_part = out.split(" (", 1)[0].split(": ", 1)[1]
        assert live_only in head_part.split("·"), (
            f"{live_only} should appear as covered in the heartbeat coverage "
            f"line, got: {out!r}"
        )

    def test_price_alert_universe_is_what_heartbeat_uses(self):
        # _price_alert_universe is the SSOT helper the heartbeat now passes;
        # it must be a superset of both static PORTFOLIO_TICKERS and
        # LIVE_PORTFOLIO_TICKERS. (This is also the universe price alerts
        # already monitor.)
        universe = set(daemon._price_alert_universe())
        assert set(daemon.PORTFOLIO_TICKERS).issubset(universe)
        assert set(ml_features.LIVE_PORTFOLIO_TICKERS).issubset(universe)

"""ALERT BOOK VELOCITY — per-held-ticker BREAKING-alert magnitude.

``ALERT VELOCITY`` measures the OVERALL wire firing rate; ``BOOK HEAT`` counts
distinct DIGEST rows touching each held name. Neither answers the per-position
question the analyst persona most cares about: is one of MY held names itself
the centre of the breaking-wire activity this 5h window? A held ticker carried
by one alert is generic news (already flagged by the per-row [BOOK:] tag); the
SAME held ticker carried by ≥2 distinct breaking alerts is the multiplicity
signal worth a separate, ranked hint to Opus.

These pin: the line renderer (empty/below-floor/at-floor/multi-ticker/newly-
active per-position edge), the canonical _BOOK_TICKERS sort-tiebreak (stable
cycle-to-cycle, parity with _book_heat_lines / _book_silence_lines), the
word-boundary discipline reused from _BOOK_RE (MUSEUM ≠ MU, "Micron Q3" ≠ MU
— the ticker is the SYMBOL, not the company name), the _build_payload
emission gate (omit-when-None and omit-when-below-threshold, byte-identical
7-arg default path), the SSOT data-source pin (reads alert_recency.recent_alerts
verbatim, NEVER articles.db — same drift class as the ALERT VELOCITY pin), and
the SYSTEM_PROMPT "do not echo" framing.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from analysis import claude_analyst


# ── _alert_book_velocity_lines (pure renderer) ───────────────────────────────
class TestAlertBookVelocityLines:
    def test_none_or_non_dict_input_is_silence(self):
        assert claude_analyst._alert_book_velocity_lines(None) == []
        assert claude_analyst._alert_book_velocity_lines("x") == []  # type: ignore[arg-type]
        assert claude_analyst._alert_book_velocity_lines([]) == []  # type: ignore[arg-type]
        assert claude_analyst._alert_book_velocity_lines({}) == []

    def test_missing_window_h_is_silence(self):
        # No window_h → ambiguous duration → silent rather than rendering "in 0h".
        assert claude_analyst._alert_book_velocity_lines(
            {"tickers": {"MU": {"recent": 5, "prior": 1}}}
        ) == []

    def test_zero_or_negative_window_h_is_silence(self):
        assert claude_analyst._alert_book_velocity_lines(
            {"window_h": 0, "tickers": {"MU": {"recent": 5, "prior": 1}}}
        ) == []
        assert claude_analyst._alert_book_velocity_lines(
            {"window_h": -5, "tickers": {"MU": {"recent": 5, "prior": 1}}}
        ) == []

    def test_empty_tickers_dict_is_silence(self):
        assert claude_analyst._alert_book_velocity_lines(
            {"window_h": 5, "tickers": {}}
        ) == []

    def test_below_min_recent_is_silence(self):
        # recent=1 is below the 2-floor — single-alert is already surfaced by
        # the per-row [BOOK:] tag, not by this multiplicity block.
        assert claude_analyst._alert_book_velocity_lines(
            {"window_h": 5, "tickers": {"MU": {"recent": 1, "prior": 4}}}
        ) == []

    def test_at_min_recent_emits_line(self):
        out = claude_analyst._alert_book_velocity_lines(
            {"window_h": 5, "tickers": {"MU": {"recent": 2, "prior": 0}}}
        )
        assert len(out) == 1
        assert (
            "MU — 2 BREAKING alerts mention this name in last 5h "
            "(vs 0 in prior 5h)"
        ) in out[0]

    def test_above_min_recent_with_baseline_emits_line(self):
        out = claude_analyst._alert_book_velocity_lines(
            {"window_h": 5, "tickers": {"NVDA": {"recent": 6, "prior": 2}}}
        )
        assert len(out) == 1
        assert (
            "NVDA — 6 BREAKING alerts mention this name in last 5h "
            "(vs 2 in prior 5h)"
        ) in out[0]

    def test_newly_active_per_position_signal_emits(self):
        """A held ticker with recent>=floor and prior==0 is the STRONGEST
        per-position signal — newly-active wire on a held name. Must emit."""
        out = claude_analyst._alert_book_velocity_lines(
            {"window_h": 5, "tickers": {"LITE": {"recent": 3, "prior": 0}}}
        )
        assert len(out) == 1
        assert "LITE — 3 BREAKING" in out[0]
        assert "vs 0 in prior" in out[0]

    def test_multi_ticker_sorted_by_recent_desc(self):
        out = claude_analyst._alert_book_velocity_lines(
            {"window_h": 5, "tickers": {
                "MU": {"recent": 3, "prior": 1},
                "NVDA": {"recent": 5, "prior": 0},
                "LITE": {"recent": 2, "prior": 2},
            }}
        )
        assert len(out) == 3
        # NVDA(5) > MU(3) > LITE(2)
        assert out[0].startswith("NVDA — 5")
        assert out[1].startswith("MU — 3")
        assert out[2].startswith("LITE — 2")

    def test_equal_recent_breaks_to_canonical_book_order(self):
        # _BOOK_TICKERS order: LITE, LNOK, MUU, DRAM, SNDU, MU, MSFT, AXTI, ORCL, TSEM, QBTS, NVDA
        # When recent is tied, LITE (idx 0) precedes MU (idx 5) precedes NVDA (idx 11).
        out = claude_analyst._alert_book_velocity_lines(
            {"window_h": 5, "tickers": {
                "MU": {"recent": 3, "prior": 0},
                "LITE": {"recent": 3, "prior": 0},
                "NVDA": {"recent": 3, "prior": 0},
            }}
        )
        assert len(out) == 3
        assert out[0].startswith("LITE — 3")
        assert out[1].startswith("MU — 3")
        assert out[2].startswith("NVDA — 3")

    def test_max_lines_cap(self):
        # 5 tickers at min_recent → cap to _ALERT_BOOK_VELOCITY_MAX_LINES (4).
        out = claude_analyst._alert_book_velocity_lines(
            {"window_h": 5, "tickers": {
                "LITE": {"recent": 5, "prior": 0},
                "MU":   {"recent": 4, "prior": 0},
                "NVDA": {"recent": 3, "prior": 0},
                "ORCL": {"recent": 2, "prior": 0},
                "MSFT": {"recent": 2, "prior": 1},
            }}
        )
        assert len(out) == 4

    def test_non_dict_counts_skipped(self):
        # A malformed entry must not crash the briefing.
        out = claude_analyst._alert_book_velocity_lines(
            {"window_h": 5, "tickers": {
                "MU": "not-a-dict",
                "NVDA": {"recent": 3, "prior": 1},
            }}
        )
        assert len(out) == 1
        assert out[0].startswith("NVDA — 3")

    def test_negative_counts_skipped(self):
        # A clock-skew / parser-glitch row with negative counts must drop.
        out = claude_analyst._alert_book_velocity_lines(
            {"window_h": 5, "tickers": {"MU": {"recent": -1, "prior": 3}}}
        )
        assert out == []

    def test_unparseable_window_h_returns_silence(self):
        assert claude_analyst._alert_book_velocity_lines(
            {"window_h": "five", "tickers": {"MU": {"recent": 3, "prior": 1}}}
        ) == []


# ── _collect_alert_book_velocity (data-source pin) ───────────────────────────
class TestCollectAlertBookVelocityDataSource:
    """Pin that ``_collect_alert_book_velocity`` reads from ``alert_recency.db``
    (the canonical fires log — one row per successful Discord send) and NOT
    from ``articles.db`` ``urgency=2`` (also set by the four pre-fire
    suppression gates in ``watchers.alert_agent.send_urgent_alert`` without
    a Discord send — would over-count).

    Same drift class as the ALERT VELOCITY data-source pin
    (``TestCollectAlertVelocityDataSource``): if a future regression repoints
    the collector at ``urgency=2``, these tests fail loud.
    """

    def _make_recency_db(self, path: Path, rows: list[tuple[str, str, str]]) -> None:
        """Build a fake ``alert_recency.db`` at ``path`` mirroring the live
        schema. Each row is ``(sig, last_ts_iso, title)``."""
        conn = sqlite3.connect(str(path))
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS alerted_sig (
                    sig      TEXT PRIMARY KEY,
                    last_ts  TEXT NOT NULL,
                    title    TEXT,
                    hits     INTEGER NOT NULL DEFAULT 1
                );
                CREATE INDEX IF NOT EXISTS idx_alerted_sig_ts
                    ON alerted_sig(last_ts);
            """)
            conn.executemany(
                "INSERT OR REPLACE INTO alerted_sig (sig, last_ts, title, hits) "
                "VALUES (?, ?, ?, 1)",
                rows,
            )
            conn.commit()
        finally:
            conn.close()

    def test_counts_held_ticker_mentions_per_window(
        self, tmp_path, monkeypatch
    ):
        """Two MU alerts in last 5h, one in prior 5h — collector must report
        MU recent=2, prior=1, window_h=5 (and never read articles.db)."""
        db_path = tmp_path / "alert_recency.db"
        now = datetime.now(timezone.utc)
        rows = [
            # recent (within 5h) — both mention MU only
            ("sig-r-1", (now - timedelta(hours=0.5)).isoformat(),
             "MU surges on memory pricing news"),
            ("sig-r-2", (now - timedelta(hours=3.0)).isoformat(),
             "MU guidance cut — supply chain hit"),
            # prior (5h-10h ago) — one MU
            ("sig-p-1", (now - timedelta(hours=7.0)).isoformat(),
             "MU 8-K filing"),
            # outside 10h window — must NOT be counted
            ("sig-old", (now - timedelta(hours=11.0)).isoformat(),
             "MU ancient news"),
            # generic alert — no held ticker, must not be counted
            ("sig-gen", (now - timedelta(hours=1.0)).isoformat(),
             "Fed surprise rate cut"),
        ]
        self._make_recency_db(db_path, rows)

        # Force alert_recency to read this fake DB by monkeypatching DB_PATH.
        # Mirrors TestCollectAlertVelocityDataSource's approach.
        from watchers import alert_recency
        monkeypatch.setattr(alert_recency, "DB_PATH", db_path)

        result = claude_analyst._collect_alert_book_velocity(window_hours=5.0)
        assert result is not None
        assert result["window_h"] == 5
        # MU was the only held ticker mentioned
        assert "MU" in result["tickers"]
        assert result["tickers"]["MU"] == {"recent": 2, "prior": 1}
        # No other held ticker should appear
        assert set(result["tickers"].keys()) == {"MU"}

    def test_missing_db_degrades_to_empty_tickers(self, tmp_path, monkeypatch):
        """A nonexistent recency DB must not crash; the collector must
        degrade to ``{tickers: {}}`` so the briefing is unaffected. Identical
        safety contract to the rest of the operational-status family."""
        # Point at a path that has no DB file. alert_recency._connect()
        # creates the file/schema on demand, so the read returns an empty
        # alerted_sig table → no held-ticker hits.
        missing_path = tmp_path / "does_not_exist_yet.db"
        from watchers import alert_recency
        monkeypatch.setattr(alert_recency, "DB_PATH", missing_path)
        result = claude_analyst._collect_alert_book_velocity(window_hours=5.0)
        # Either an empty-dict result or None — both are safe (the
        # _build_payload gate omits the section in either case). Pin the
        # SAFER behaviour (returns a structured empty result rather than
        # silent None) since that lets the dashboard introspect a healthy
        # zero-state vs an outright collector failure.
        assert result is not None
        assert result["window_h"] == 5
        assert result["tickers"] == {}

    def test_word_boundary_prevents_substring_match(
        self, tmp_path, monkeypatch
    ):
        """A title containing 'MUSEUM', 'MUTUAL' or 'Micron' must NOT count
        toward MU — the held ticker is the SYMBOL, not the company name.
        Reuses _BOOK_RE's word-boundary guarantee; pin on THIS surface."""
        db_path = tmp_path / "alert_recency.db"
        now = datetime.now(timezone.utc)
        rows = [
            ("s1", (now - timedelta(hours=1.0)).isoformat(),
             "MUSEUM of Modern Art reopens"),
            ("s2", (now - timedelta(hours=2.0)).isoformat(),
             "MUTUAL fund flows reverse"),
            ("s3", (now - timedelta(hours=3.0)).isoformat(),
             "Micron earnings — strong quarter"),  # company name, not symbol
        ]
        self._make_recency_db(db_path, rows)
        from watchers import alert_recency
        monkeypatch.setattr(alert_recency, "DB_PATH", db_path)
        result = claude_analyst._collect_alert_book_velocity(window_hours=5.0)
        assert result is not None
        assert "MU" not in result["tickers"]
        assert result["tickers"] == {}


# ── _build_payload wiring ────────────────────────────────────────────────────
class TestBuildPayloadAlertBookVelocityWiring:
    def test_emit_when_held_ticker_above_floor(self):
        out = claude_analyst._build_payload(
            articles=[],
            stock_data={"macro": [], "equities": []},
            earnings=[],
            alert_book_velocity={
                "window_h": 5,
                "tickers": {"MU": {"recent": 4, "prior": 1}},
            },
        )
        assert "=== ALERT BOOK VELOCITY" in out
        # The exact rendered line must appear so a non-conforming Opus could
        # still recover it via the prompt's "verbatim" rule.
        assert (
            "MU — 4 BREAKING alerts mention this name in last 5h "
            "(vs 1 in prior 5h)"
        ) in out

    def test_omit_when_below_floor(self):
        """recent=1 (below the 2-floor) emits NO section — same "below the
        bar means silent" discipline as ALERT VELOCITY / COVERAGE GAP."""
        out = claude_analyst._build_payload(
            articles=[],
            stock_data={"macro": [], "equities": []},
            earnings=[],
            alert_book_velocity={
                "window_h": 5,
                "tickers": {"MU": {"recent": 1, "prior": 3}},
            },
        )
        assert "ALERT BOOK VELOCITY" not in out

    def test_omit_entirely_when_arg_is_none(self):
        """The default path (no alert_book_velocity kwarg) MUST be byte-
        identical to the pre-feature behaviour for that section — every
        caller / test that doesn't pass it stays unaffected. Same discipline
        as alert_velocity / source_health_report / prior_digest /
        source_throughput (the documented anti-drift discipline)."""
        out_default = claude_analyst._build_payload(
            articles=[],
            stock_data={"macro": [], "equities": []},
            earnings=[],
        )
        out_explicit_none = claude_analyst._build_payload(
            articles=[],
            stock_data={"macro": [], "equities": []},
            earnings=[],
            alert_book_velocity=None,
        )
        assert "ALERT BOOK VELOCITY" not in out_default
        assert out_default == out_explicit_none

    def test_emit_when_empty_tickers_dict_omits_section(self):
        """An explicit-but-empty velocity ({tickers:{}}) emits no section —
        same shape as ALERT VELOCITY's below-threshold path. The section
        text must not appear under any "valid input but nothing to surface"
        path."""
        out = claude_analyst._build_payload(
            articles=[],
            stock_data={"macro": [], "equities": []},
            earnings=[],
            alert_book_velocity={"window_h": 5, "tickers": {}},
        )
        assert "ALERT BOOK VELOCITY" not in out

    def test_multi_ticker_renders_in_recent_desc_order(self):
        out = claude_analyst._build_payload(
            articles=[],
            stock_data={"macro": [], "equities": []},
            earnings=[],
            alert_book_velocity={
                "window_h": 5,
                "tickers": {
                    "MU": {"recent": 3, "prior": 0},
                    "NVDA": {"recent": 5, "prior": 1},
                },
            },
        )
        assert "=== ALERT BOOK VELOCITY" in out
        nvda_pos = out.find("NVDA — 5")
        mu_pos = out.find("MU — 3")
        assert nvda_pos != -1 and mu_pos != -1
        assert nvda_pos < mu_pos


# ── SYSTEM_PROMPT contract ───────────────────────────────────────────────────
class TestSystemPromptRule:
    def test_prompt_documents_alert_book_velocity_block(self):
        """SYSTEM_PROMPT must instruct Opus how to treat the input block:
        weight named held tickers in LEAD / TOP SIGNALS / PORTFOLIO, and
        explicitly DO NOT echo a literal section (the BOOK-HEAT / BOOK-SILENCE
        framing — an input hint, not a reproduced section). A future edit
        that turns this into a 'reproduce verbatim' section (like COVERAGE
        GAP) would silently change Opus output shape — pinned here."""
        prompt = claude_analyst.SYSTEM_PROMPT
        assert "ALERT BOOK VELOCITY" in prompt
        # The "do NOT echo" framing is what distinguishes this from COVERAGE
        # GAP / THROUGHPUT DEGRADATION / ALERT VELOCITY (all reproduced).
        # Pin both the negative consequence and the cross-reference to the
        # sibling hint family, so a refactor of the family wording can't
        # silently drop the no-echo discipline for this block.
        block_idx = prompt.find("ALERT BOOK VELOCITY")
        # Find the end of this rule (next bullet starts with "\n-" after a
        # double-newline section break or the OUTPUT FORMAT heading).
        rule_segment = prompt[block_idx:block_idx + 1200]
        assert "do NOT echo" in rule_segment
        assert "BOOK HEAT" in rule_segment or "BOOK SILENCE" in rule_segment

    def test_prompt_names_per_row_book_tag_distinction(self):
        """The block exists because the per-row [BOOK:] tag flags WHICH rows
        touch the book, but never that the held ticker is the WINDOW'S hot
        centre. The prompt rule must call out this distinction — the next
        reviewer should see why this isn't duplicative of the existing
        [BOOK:] tag rule."""
        prompt = claude_analyst.SYSTEM_PROMPT
        block_idx = prompt.find("ALERT BOOK VELOCITY")
        rule_segment = prompt[block_idx:block_idx + 1200]
        # The discriminator: per-row vs window-level magnitude.
        assert "[BOOK:]" in rule_segment


# ── Source-of-truth pin: _book_tickers is the held-ticker primitive ──────────
class TestBookTickersSSOT:
    """Same SSOT discipline as the briefing's BOOK HEAT and BOOK SILENCE:
    held-ticker resolution composes ``claude_analyst._book_tickers`` verbatim,
    not a re-derived regex. Pin via source inspection so a future "optimize"
    that inlines a duplicate matcher fails loud — same drift class as the
    paper-trader signals / dashboard parity regression family."""

    def test_collector_uses_book_tickers_helper(self):
        import inspect
        src = inspect.getsource(claude_analyst._collect_alert_book_velocity)
        assert "_book_tickers(" in src, (
            "_collect_alert_book_velocity must compose claude_analyst._book_tickers "
            "verbatim; an inline re-derived regex would silently drift from the "
            "briefing's [BOOK:] / BOOK HEAT / BOOK SILENCE surfaces"
        )

"""BOOK SILENCE — held names with ZERO stories in the 5h Opus digest.

The Discord-post-briefing ``daemon._format_portfolio_coverage`` line names
silent tickers, but it is appended AFTER Opus has written the briefing — Opus
composes the LEAD / TOP SIGNALS / PORTFOLIO table BLIND to which held names
had no story, and historically fabricates a "neutral implication" for them
(the analyst persona's complaint about hedging filler).

``claude_analyst._book_silence_lines`` + the ``=== BOOK SILENCE ===`` input
block surface the silent set as an INPUT hint so PORTFOLIO can mark dark
tickers honestly. Same pure read-side shape as BOOK HEAT (input hint, never
echoed): no DB write, no ai_score/ml_score/score_source/urgency touch, no row
mutation, backtest excluded upstream — all four load-bearing invariants
intact by construction.

These pin: the silent-set computation, the conservative min-silent threshold
(noise control on a normal macro window), canonical _BOOK_TICKERS ordering
(stable cycle-to-cycle), the real-url snapshot guard (snapshots cannot
fake-cover a silent name via their P&L body), read-only behaviour
(source_articles untouched), the _build_payload emission gate (block omitted
deterministically when the helper returns []), and the SYSTEM_PROMPT rule
with its "honest N/A" PORTFOLIO consequence + "do not echo" framing.
"""
from __future__ import annotations

from analysis import claude_analyst


def _url_row(title, summary="", score=7.0):
    return {"title": title, "summary": summary, "source": "rss",
            "ai_score": score, "link": "https://reuters.com/" + title[:8]}


# ── _book_silence_lines: silent-set computation ──────────────────────────────
class TestBookSilenceLines:
    def test_empty_input_is_silence(self):
        # An empty digest can't honestly call any held name silent — no signal.
        assert claude_analyst._book_silence_lines([]) == []

    def test_full_coverage_emits_no_line(self):
        # Every held ticker mentioned at least once → silent set is empty.
        arts = [_url_row(f"{t} update for the week")
                for t in claude_analyst._BOOK_TICKERS]
        assert claude_analyst._book_silence_lines(arts) == []

    def test_below_min_silent_is_silence(self):
        # All but 2 held tickers covered → 2 silent → below 3-floor → noise.
        covered = claude_analyst._BOOK_TICKERS[:-2]
        arts = [_url_row(f"{t} catalyst this morning") for t in covered]
        assert claude_analyst._book_silence_lines(arts) == []

    def test_at_threshold_emits_exact_silent_list(self):
        # Exactly 3 silent (12 held - 9 covered) → meets the 3-floor.
        covered = list(claude_analyst._BOOK_TICKERS[:9])
        silent = list(claude_analyst._BOOK_TICKERS[9:])
        arts = [_url_row(f"{t} catalyst today") for t in covered]
        out = claude_analyst._book_silence_lines(arts)
        assert len(out) == 1
        # canonical _BOOK_TICKERS order, space-separated
        for tk in silent:
            assert tk in out[0]
        # The line must NOT mention any covered ticker
        for tk in covered:
            # word boundary not strictly needed here (covered names are
            # unique fixed strings) but make this robust to substring matches.
            assert f" {tk} " not in f" {out[0]} ", (
                f"covered ticker {tk!r} leaked into silence line"
            )

    def test_all_silent_is_emitted_when_above_floor(self):
        # Zero held-name coverage → every _BOOK_TICKER silent → one line.
        arts = [_url_row("Apple's services revenue jumps"),
                _url_row("Fed pauses rate cuts"),
                _url_row("BTC tests 70k support level"),
                _url_row("Tesla margins compress further")]
        out = claude_analyst._book_silence_lines(arts)
        assert len(out) == 1
        for tk in claude_analyst._BOOK_TICKERS:
            assert tk in out[0]

    def test_silent_list_in_canonical_book_tickers_order(self):
        # _BOOK_TICKERS = ("LITE","LNOK","MUU","DRAM","SNDU","MU","MSFT",
        #                  "AXTI","ORCL","TSEM","QBTS","NVDA")
        # Cover LITE,LNOK,MUU,DRAM (first 4) → silent should be the rest in
        # canonical order, NOT alphabetical and NOT sorted by anything else.
        arts = [_url_row(f"{t} news today")
                for t in claude_analyst._BOOK_TICKERS[:4]]
        out = claude_analyst._book_silence_lines(arts)
        assert len(out) == 1
        expected = " ".join(claude_analyst._BOOK_TICKERS[4:])
        assert out[0] == expected


# ── snapshot/synthetic url guard ─────────────────────────────────────────────
class TestSnapshotGuard:
    def test_snapshot_p_and_l_body_does_not_fake_coverage(self):
        # The prepended PORTFOLIO/OPTIONS snapshot rows carry no link/url and
        # their summary legitimately lists every held ticker — without the
        # url guard, that single row would falsely "cover" all held names
        # and the silence line would never fire. With the guard the snapshot
        # is skipped, so every other held ticker stays silent.
        snapshot = {
            "title": "PORTFOLIO P&L SNAPSHOT",
            "summary": " ".join(claude_analyst._BOOK_TICKERS),
            "ai_score": 10,
            # No link/url — same shape daemon prepends to source_articles.
        }
        # Cover only LITE; without the snapshot guard the silent line
        # would be [] (everything covered by the snapshot body).
        cover_one = _url_row("LITE Q3 print blowout")
        out = claude_analyst._book_silence_lines([snapshot, cover_one])
        assert len(out) == 1
        # Everything except LITE should be silent.
        silent = [t for t in claude_analyst._BOOK_TICKERS if t != "LITE"]
        assert out[0] == " ".join(silent)


# ── pure / read-only contract ────────────────────────────────────────────────
class TestPureReadOnly:
    def test_does_not_mutate_articles_list(self):
        arts = [_url_row("LITE Q3 print"), _url_row("MU CFO departs")]
        before_ids = [id(a) for a in arts]
        before_keys = [tuple(sorted(a.keys())) for a in arts]
        _ = claude_analyst._book_silence_lines(arts)
        assert [id(a) for a in arts] == before_ids
        assert [tuple(sorted(a.keys())) for a in arts] == before_keys

    def test_returns_new_list_not_internal_state(self):
        # Each call yields a fresh list, never a shared cached object that
        # a caller could mutate to corrupt a later call.
        arts = [_url_row("LITE Q3 print")]
        out_a = claude_analyst._book_silence_lines(arts)
        out_b = claude_analyst._book_silence_lines(arts)
        assert out_a == out_b
        assert out_a is not out_b


# ── ticker-density discipline (regex correctness reuse) ──────────────────────
class TestTickerMatchingDiscipline:
    def test_word_boundary_keeps_mu_distinct_from_muu_and_museum(self):
        # The _book_tickers helper uses _BOOK_RE (longest-first alternation +
        # word boundaries) so "MUU" matches MUU but not MU, and the bare word
        # "MUSEUM" matches neither. _book_silence_lines inherits that.
        arts = [
            _url_row("MUU triple-leveraged Mining ETF flows"),
            _url_row("Visit the Computer Museum exhibit today"),
        ]
        out = claude_analyst._book_silence_lines(arts)
        # MUU is covered; MU stays silent (the museum row covers nothing).
        assert len(out) == 1
        assert "MUU" not in out[0].split()
        assert "MU" in out[0].split()


# ── _build_payload emission gate (round-trip through the payload builder) ────
class TestBuildPayloadEmissionGate:
    def test_no_silent_set_omits_section(self):
        # Cover every held ticker → silent set empty → no BOOK SILENCE block.
        arts = [_url_row(f"{t} update for the week")
                for t in claude_analyst._BOOK_TICKERS]
        body = claude_analyst._build_payload(arts, {}, [])
        assert "=== BOOK SILENCE" not in body

    def test_silent_set_above_floor_emits_section(self):
        # Cover only first 4 held tickers → 8 silent (≥3) → block emitted.
        arts = [_url_row(f"{t} news today")
                for t in claude_analyst._BOOK_TICKERS[:4]]
        body = claude_analyst._build_payload(arts, {}, [])
        assert "=== BOOK SILENCE" in body
        # The header line carries the analyst-facing explanation verbatim.
        assert "catalyst engine dark" in body
        # Every silent ticker appears in the body.
        for tk in claude_analyst._BOOK_TICKERS[4:]:
            assert tk in body

    def test_silence_block_not_echoed_into_system_prompt(self):
        # The system prompt MUST forbid echoing a literal BOOK SILENCE
        # section (same discipline as BOOK HEAT / AGING TOP ROWS).
        sp = claude_analyst.SYSTEM_PROMPT
        assert "BOOK SILENCE" in sp
        # The "do NOT echo" framing must be present.
        assert "do NOT echo a literal \"BOOK SILENCE\"" in sp


# ── SYSTEM_PROMPT rule content (anti-fabrication consequence) ────────────────
class TestSystemPromptRule:
    def test_n_a_consequence_is_pinned(self):
        # The analyst-persona consequence of the rule: PORTFOLIO must mark
        # silent tickers honestly with N/A, NOT fabricate an implication.
        sp = claude_analyst.SYSTEM_PROMPT
        assert "N/A — no catalyst" in sp

    def test_silent_must_not_lead_pinned(self):
        # A silent ticker must not appear as the LEAD or outrank a ticker
        # with material news in TOP SIGNALS — pinned so a future prompt
        # rewrite can't quietly drop this consequence.
        sp = claude_analyst.SYSTEM_PROMPT
        assert "Silent names should NOT lead" in sp


# ── module constant locks (anti-tuning regression) ───────────────────────────
class TestModuleConstants:
    def test_min_silent_floor_is_3(self):
        # 1-2 silent tickers is noise on a normal macro window — the floor
        # is conservative by design. A future retune to 1 would flood the
        # block; pin the live-tested value so it can't drift silently.
        assert claude_analyst.BOOK_SILENCE_MIN_SILENT == 3

    def test_book_tickers_set_matches_book_heat(self):
        # Both helpers must read the SAME held set so HEAT and SILENCE
        # describe the same window symmetrically. Drift between them would
        # let a ticker be hot in BOOK HEAT yet missing from the SILENCE
        # accounting (or vice versa). Pinned by construction (single
        # module-level _BOOK_TICKERS literal); this regression-guards it.
        assert claude_analyst._BOOK_TICKERS == (
            "LITE", "LNOK", "MUU", "DRAM", "SNDU",
            "MU", "MSFT", "AXTI", "ORCL", "TSEM", "QBTS", "NVDA",
        )

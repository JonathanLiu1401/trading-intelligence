"""Pin the contract of ``alert_recency.pushed_ticker_breakdown``.

The analyst-facing per-held-ticker view of REAL Discord BREAKING pushes.
Distinct from:

  * ``ticker_burst_counts`` (returns a flat ``{ticker: int}`` for the
    in-alert annotation — no newest-age / silent-ticker structure)
  * ``urgency_label_split_by_ticker`` (counts urgency>=1 rows in
    articles.db — conflates gate-suppressed rows with real pushes)

The alert_recency.db is the canonical record of REAL Discord pushes
(``record_alerted`` only runs in ``send_urgent_alert``'s success path —
gate suppressions never write here), so this primitive answers "which
of my held names are getting pushed and which are silent right now?"
without the noise the other two surfaces carry.
"""
from __future__ import annotations

import pytest

from watchers import alert_recency


def _alert(title: str, age_h: float = 1.0, sig: str | None = None) -> dict:
    """Build a recent-alert dict matching ``recent_alerts``' shape."""
    return {"title": title, "age_hours": age_h, "sig": sig or "_sig_" + title}


# ── Empty / degenerate inputs ────────────────────────────────────────────


class TestEmptyInputs:
    def test_no_tickers_returns_empty_structure(self):
        out = alert_recency.pushed_ticker_breakdown(
            [_alert("NVDA earnings beat")], [],
        )
        assert out == {"total_pushes": 1, "by_ticker": [], "silent_tickers": []}

    def test_no_recent_alerts_marks_all_held_silent(self):
        """When there are no pushes at all, every held name is a coverage
        gap. The output preserves input ticker ordering."""
        out = alert_recency.pushed_ticker_breakdown(
            [], ["NVDA", "MU", "LITE"],
        )
        assert out == {
            "total_pushes": 0,
            "by_ticker": [],
            "silent_tickers": ["NVDA", "MU", "LITE"],
        }

    def test_no_recent_no_tickers_returns_empty_structure(self):
        out = alert_recency.pushed_ticker_breakdown([], [])
        assert out == {"total_pushes": 0, "by_ticker": [], "silent_tickers": []}

    def test_falsy_or_non_string_tickers_dropped(self):
        """A None/empty/non-string ticker entry must NOT crash and must NOT
        count as a held name."""
        out = alert_recency.pushed_ticker_breakdown(
            [_alert("NVDA earnings beat", age_h=0.5)],
            ["NVDA", None, "", "  ", 123],  # mixed garbage
        )
        assert [r["ticker"] for r in out["by_ticker"]] == ["NVDA"]
        assert out["silent_tickers"] == []

    def test_single_char_ticker_skipped(self):
        """Tickers shorter than 2 chars are skipped — too noisy to match."""
        out = alert_recency.pushed_ticker_breakdown(
            [_alert("X went up today")],
            ["X", "NVDA"],
        )
        assert "X" not in [r["ticker"] for r in out["by_ticker"]]
        assert "X" not in out["silent_tickers"]
        # NVDA still counted, just silent (no NVDA mention in the alert)
        assert "NVDA" in out["silent_tickers"]


# ── Counting + per-ticker aggregation ─────────────────────────────────────


class TestCounting:
    def test_counts_pushes_per_ticker_dedup_per_alert(self):
        """Single alert mentioning NVDA twice counts as ONE push — the
        noise being measured is # distinct PUSHES, not text occurrences.
        Mirrors ticker_burst_counts' same-decision discipline."""
        recent = [
            _alert("NVDA crushed Q1 — NVDA buyback announced", age_h=0.3),
            _alert("NVDA guidance lifts", age_h=0.8),
            _alert("MU shares halted on memory pricing shock", age_h=1.5),
        ]
        out = alert_recency.pushed_ticker_breakdown(recent, ["NVDA", "MU"])
        by = {r["ticker"]: r["pushes"] for r in out["by_ticker"]}
        # First alert mentions NVDA twice → still 1 push.
        assert by == {"NVDA": 2, "MU": 1}
        assert out["total_pushes"] == 3

    def test_case_insensitive_matching(self):
        """Match is word-boundary case-insensitive — same convention as
        ``ml.features._LIVE_RE`` (the model's own ticker detection). A
        lowercase 'nvda' must count."""
        recent = [
            _alert("Nvidia (nvda) earnings recap"),
            _alert("$NVDA buyback announced"),
            _alert("NVDA at all-time high"),
        ]
        out = alert_recency.pushed_ticker_breakdown(recent, ["NVDA"])
        assert out["by_ticker"][0]["pushes"] == 3

    def test_substring_match_blocked_by_word_boundary(self):
        """A literal containing the ticker as a substring must NOT match —
        whole-word only. 'NVDAQ' / 'AMDOCS' would be the live failure cases
        we explicitly want to avoid for the ticker-volatile small-caps."""
        recent = [
            _alert("AMDOCS price target raised", age_h=0.4),
            _alert("NVDAQ index surges", age_h=0.7),
        ]
        out = alert_recency.pushed_ticker_breakdown(recent, ["AMD", "NVDA"])
        assert out["by_ticker"] == []
        # Both held names are silent — no alerts touched them.
        assert "AMD" in out["silent_tickers"]
        assert "NVDA" in out["silent_tickers"]

    def test_input_case_preserved_in_output(self):
        """A held name passed as 'Nvda' / 'mu' must be reported back
        verbatim — useful for callers that render UI with a fixed display
        case (matches the urgency_label_split_by_ticker contract)."""
        out = alert_recency.pushed_ticker_breakdown(
            [_alert("Nvidia (NVDA) beats Q1")],
            ["Nvda"],
        )
        assert out["by_ticker"][0]["ticker"] == "Nvda"

    def test_duplicate_input_tickers_deduplicated(self):
        """Input may contain dupes (e.g. portfolio.json + watchlist.json
        both list NVDA). The result must collapse to a single per-ticker
        entry — never double-count."""
        out = alert_recency.pushed_ticker_breakdown(
            [_alert("NVDA earnings beat", age_h=0.3)],
            ["NVDA", "NVDA", "nvda"],
        )
        # Single by_ticker entry, no silent dupes.
        nvda_rows = [r for r in out["by_ticker"] if r["ticker"].upper() == "NVDA"]
        assert len(nvda_rows) == 1
        assert nvda_rows[0]["pushes"] == 1
        assert out["silent_tickers"] == []


# ── Newest-age tracking ───────────────────────────────────────────────────


class TestNewestAgeTracking:
    def test_newest_age_is_minimum_across_pushes(self):
        """'Newest' means smallest age — the most-recent push. Pin that
        the aggregation correctly selects min(age_h) per ticker."""
        recent = [
            _alert("NVDA Q1 beat (older copy)", age_h=4.5),
            _alert("NVDA buyback announced (fresher copy)", age_h=0.3),
            _alert("NVDA dividend hike (mid)", age_h=2.1),
        ]
        out = alert_recency.pushed_ticker_breakdown(recent, ["NVDA"])
        row = out["by_ticker"][0]
        assert row["newest_age_h"] == 0.3
        assert "buyback announced" in row["newest_title"]

    def test_newest_age_rounded_to_2dp(self):
        """Ages should be readable — round to 0.01h precision."""
        out = alert_recency.pushed_ticker_breakdown(
            [_alert("NVDA earnings", age_h=0.123456789)],
            ["NVDA"],
        )
        assert out["by_ticker"][0]["newest_age_h"] == 0.12

    def test_malformed_age_treated_as_zero(self):
        """A recent-alert row missing/garbled ``age_hours`` must NOT crash
        — defensively defaults to 0.0 so the push still counts."""
        recent = [
            {"title": "NVDA earnings", "age_hours": "not-a-number"},
            {"title": "MU shares halted"},  # missing key
        ]
        out = alert_recency.pushed_ticker_breakdown(recent, ["NVDA", "MU"])
        by = {r["ticker"]: r for r in out["by_ticker"]}
        # Both still recorded, ages defaulted to 0.0.
        assert by["NVDA"]["pushes"] == 1
        assert by["MU"]["pushes"] == 1
        assert by["NVDA"]["newest_age_h"] == 0.0
        assert by["MU"]["newest_age_h"] == 0.0


# ── Silent tickers (coverage gap detection) ───────────────────────────────


class TestSilentTickers:
    def test_silent_tickers_preserves_input_order(self):
        """Coverage-gap list MUST preserve input order (useful when the
        caller passes the held book sorted by importance/size). NVDA is
        pushed; MU/LITE/AXTI are coverage gaps in that specific order."""
        recent = [_alert("NVDA earnings beat", age_h=0.5)]
        out = alert_recency.pushed_ticker_breakdown(
            recent, ["NVDA", "MU", "LITE", "AXTI"],
        )
        assert out["silent_tickers"] == ["MU", "LITE", "AXTI"]

    def test_silent_tickers_excludes_pushed_names(self):
        """A held name with >= 1 push must NOT appear in silent_tickers."""
        recent = [
            _alert("NVDA earnings", age_h=0.5),
            _alert("MU shares halted", age_h=1.0),
        ]
        out = alert_recency.pushed_ticker_breakdown(
            recent, ["NVDA", "MU", "LITE"],
        )
        # Only LITE is silent.
        assert out["silent_tickers"] == ["LITE"]
        # NVDA and MU are both in by_ticker.
        pushed = {r["ticker"] for r in out["by_ticker"]}
        assert pushed == {"NVDA", "MU"}


# ── Sort ordering (most-pushed-first, alphabetical tiebreak) ──────────────


class TestSortOrdering:
    def test_by_ticker_sorted_most_pushed_first(self):
        recent = [_alert(f"NVDA copy {i}", age_h=i * 0.1) for i in range(5)]
        recent += [_alert(f"MU copy {i}", age_h=i * 0.1) for i in range(3)]
        recent += [_alert("LITE single", age_h=0.2)]
        out = alert_recency.pushed_ticker_breakdown(
            recent, ["MU", "NVDA", "LITE"],
        )
        order = [r["ticker"] for r in out["by_ticker"]]
        assert order == ["NVDA", "MU", "LITE"]

    def test_alphabetical_tiebreak_on_equal_count(self):
        """Two tickers with equal push counts must sort alphabetically so
        the dashboard order is stable cycle-to-cycle. Mirrors
        urgency_label_split_by_source's deterministic-tiebreak convention."""
        recent = [_alert("NVDA push", age_h=0.5), _alert("MU push", age_h=0.5)]
        out = alert_recency.pushed_ticker_breakdown(recent, ["NVDA", "MU"])
        # Both have pushes=1 — alphabetical: MU before NVDA.
        order = [r["ticker"] for r in out["by_ticker"]]
        assert order == ["MU", "NVDA"]


# ── total_pushes invariant ────────────────────────────────────────────────


class TestTotalPushes:
    def test_total_pushes_is_recent_length(self):
        """``total_pushes`` is the length of ``recent`` regardless of how
        many touch held tickers — surfaces the analyst's overall push
        volume in the same window so they can read "got 35 pushes, only
        12 touched the book" at a glance."""
        recent = [
            _alert("NVDA Q1 beat", age_h=0.3),       # touches book
            _alert("Fed cuts rates", age_h=1.5),     # no held ticker
            _alert("MU halted", age_h=2.1),          # touches book
            _alert("BTC crashes 8%", age_h=4.2),     # no held ticker
        ]
        out = alert_recency.pushed_ticker_breakdown(
            recent, ["NVDA", "MU", "LITE"],
        )
        assert out["total_pushes"] == 4
        assert sum(r["pushes"] for r in out["by_ticker"]) == 2  # held-touching


# ── Defensive row handling ────────────────────────────────────────────────


class TestDefensiveRowHandling:
    def test_non_dict_row_skipped(self):
        """A garbled ``recent`` entry (None, string) must NOT crash."""
        recent = [None, "garbage", {"title": "NVDA earnings", "age_hours": 0.5}]
        out = alert_recency.pushed_ticker_breakdown(recent, ["NVDA"])
        assert out["by_ticker"][0]["pushes"] == 1

    def test_missing_title_skipped(self):
        recent = [
            {"age_hours": 0.5},  # no title
            {"title": "", "age_hours": 0.5},  # empty title
            {"title": "NVDA earnings", "age_hours": 0.5},
        ]
        out = alert_recency.pushed_ticker_breakdown(recent, ["NVDA"])
        assert out["by_ticker"][0]["pushes"] == 1


# ── Realistic live scenario ───────────────────────────────────────────────


class TestRealisticScenario:
    """Reproduce a NVDA-earnings-night-style push storm and pin the
    expected breakdown."""

    def test_nvda_earnings_night_breakdown(self):
        """2026-05-21 NVDA-earnings-night style: many NVDA pushes, a
        handful of MU pushes, AXTI/LITE silent. The breakdown surfaces
        both the concentration (NVDA dominates) AND the coverage gap
        (AXTI/LITE got nothing despite being in the held book)."""
        recent = (
            [_alert(f"NVDA development {i}", age_h=i * 0.5) for i in range(12)]
            + [_alert(f"MU memory update {i}", age_h=1 + i * 0.4) for i in range(3)]
            + [_alert("Fed holds rates steady", age_h=1.2)]
            + [_alert("Asia stocks rally on chip surge", age_h=2.0)]
        )
        held = ["NVDA", "MU", "AXTI", "LITE", "QBTS", "ORCL"]
        out = alert_recency.pushed_ticker_breakdown(recent, held)

        # 17 total pushes, NVDA dominates.
        assert out["total_pushes"] == 17
        by = {r["ticker"]: r for r in out["by_ticker"]}
        assert by["NVDA"]["pushes"] == 12
        assert by["MU"]["pushes"] == 3
        assert "NVDA" not in out["silent_tickers"]
        assert "MU" not in out["silent_tickers"]

        # AXTI / LITE / QBTS / ORCL never appeared in any push — coverage gap.
        assert set(out["silent_tickers"]) == {"AXTI", "LITE", "QBTS", "ORCL"}

        # Newest NVDA age is 0.0 (the i=0 copy).
        assert by["NVDA"]["newest_age_h"] == 0.0

        # Order: NVDA (12) > MU (3) — most-pushed first.
        assert out["by_ticker"][0]["ticker"] == "NVDA"
        assert out["by_ticker"][1]["ticker"] == "MU"


# ── Integration with recent_alerts ────────────────────────────────────────


class TestIntegrationWithRecentAlerts:
    """``pushed_ticker_breakdown`` is a pure function — it consumes whatever
    ``recent_alerts`` returns. Confirm the shape compatibility end-to-end."""

    def test_pushed_breakdown_consumes_recent_alerts_output(self, tmp_path,
                                                              monkeypatch):
        """End-to-end: record a few alerts, fetch via recent_alerts(), feed
        into pushed_ticker_breakdown(). Confirms the keys line up."""
        # Re-isolate the recency DB to a per-test path (already done by
        # conftest's autouse fixture, but explicit for clarity).
        monkeypatch.setattr(
            alert_recency, "DB_PATH", tmp_path / "alert_recency.db"
        )
        # Record three alerts.
        alert_recency.record_alerted([
            {"title": "NVDA Q1 earnings crush expectations",
             "_id": "a1", "link": "https://x/a"},
            {"title": "NVDA adds $80B buyback program",
             "_id": "a2", "link": "https://x/b"},
            {"title": "MU shares halted on memory news",
             "_id": "a3", "link": "https://x/c"},
        ])
        recent = alert_recency.recent_alerts(ttl_hours=24)
        assert len(recent) == 3

        out = alert_recency.pushed_ticker_breakdown(
            recent, ["NVDA", "MU", "AXTI"],
        )
        by = {r["ticker"]: r["pushes"] for r in out["by_ticker"]}
        assert by == {"NVDA": 2, "MU": 1}
        assert out["silent_tickers"] == ["AXTI"]

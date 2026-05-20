"""Delivery audit — distinguishes Discord-pushed alerts from gate-marked ones.

The audit answers a question the dashboard's ``urgent`` tile silently conflates:
of all the ``urgency=2`` rows in the window, how many actually pushed to
Discord, and which defense-in-depth gate absorbed the rest?

These assertions exist because the audit is the analyst's calibration view —
if a future change to the SSOT fingerprints (``alert_agent``) doesn't propagate
here, the analyst will be looking at a stale picture.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from analytics import alert_delivery_audit as A
from watchers.alert_dedup import _signature


def _fresh_iso(minutes_ago: int = 5) -> str:
    """A first_seen/published value that passes ``_article_age_ok``'s 24h
    window — same idea as ``tests/test_article_store.py::_recent_iso`` so a
    Saturday-morning rerun doesn't false-fail on a real invariant."""
    return (
        datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    ).isoformat()


def _stale_iso(hours_ago: int = 48) -> str:
    return (
        datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    ).isoformat()


# ── pure function: empty input ────────────────────────────────────────────────
class TestEmptyInputs:
    def test_no_urgent_rows_zero_counts(self):
        out = A.compute_delivery_audit([], set())
        assert out["total"] == 0
        assert out["delivered"] == 0
        assert out["suppressed"] == 0
        assert out["delivery_rate"] == 0.0
        # zero-data discipline: keys exist for every gate, never KeyError-able
        for k in ("synthetic", "quote_widget", "recap_template",
                  "low_authority", "stale_published", "unknown_gate"):
            assert out["suppressed_by"][k] == 0


# ── delivered vs suppressed discrimination ────────────────────────────────────
class TestDeliveredVsSuppressed:
    def test_signature_match_counts_delivered(self):
        """A row whose signature is in ``alerted_sigs`` is counted delivered —
        even if a fingerprint would *also* match it. ``alerted_sig`` is the
        ground-truth Discord ledger; a fingerprint match on a delivered row is
        irrelevant (the analyst saw it). This is the load-bearing invariant of
        the audit."""
        title = "Nvidia beats Q3 estimates on AI demand"
        sig = _signature(title)
        art = {
            "_id": "a1", "title": title, "source": "rss",
            "link": "https://reuters.com/x", "published": _fresh_iso(),
            "first_seen": _fresh_iso(), "ai_score": 9.0,
        }
        out = A.compute_delivery_audit([art], {sig})
        assert out["delivered"] == 1
        assert out["suppressed"] == 0
        assert out["delivery_rate"] == 1.0

    def test_no_signature_match_counts_suppressed(self):
        """A row whose signature is NOT in ``alerted_sigs`` was gate-marked —
        the analyst never received a Discord push for it. If no fingerprint
        catches it, it attributes to ``unknown_gate``."""
        art = {
            "_id": "a1", "title": "Some real headline with enough tokens",
            "source": "rss", "link": "https://x.com/y",
            "published": _fresh_iso(), "first_seen": _fresh_iso(),
            "ai_score": 9.0,
        }
        out = A.compute_delivery_audit([art], set())
        assert out["delivered"] == 0
        assert out["suppressed"] == 1
        # Real prose, fresh, credible source → no gate catches it.
        assert out["suppressed_by"]["unknown_gate"] == 1

    def test_delivery_rate_mixed(self):
        delivered_title = "Fed surprise rate cut shocks markets today"
        delivered = {
            "_id": "d1", "title": delivered_title, "source": "rss",
            "link": "https://reuters.com/fed", "published": _fresh_iso(),
            "first_seen": _fresh_iso(), "ai_score": 9.5,
        }
        suppressed = {
            "_id": "s1", "title": "Why Nvidia (NVDA) Stock Is Trading Up Today",
            "source": "rss", "link": "https://x.com/recap",
            "published": _fresh_iso(), "first_seen": _fresh_iso(),
            "ai_score": 9.0,
        }
        sigs = {_signature(delivered_title)}
        out = A.compute_delivery_audit([delivered, suppressed], sigs)
        assert out["total"] == 2
        assert out["delivered"] == 1
        assert out["suppressed"] == 1
        assert out["delivery_rate"] == 0.5


# ── fingerprint attribution to the correct gate ───────────────────────────────
class TestGateAttribution:
    def test_quote_widget_attributed(self):
        """The Yahoo ticker-tape pseudo-article fingerprint catches before
        every other gate. ``send_urgent_alert`` applies quote_widget BEFORE
        recap_template, so attribution order matters."""
        art = {
            "_id": "qw1",
            # A "letter glued to a decimal price" — the canonical fingerprint
            # ``_QW_PRICE_GLUE`` in ``watchers.alert_agent``.
            "title": "NVDANVIDIA Corporation227.13-8.61(-3.65%)",
            "source": "scraped/finance.yahoo.com",
            "link": "https://finance.yahoo.com/quote/NVDA/",
            "published": _fresh_iso(), "first_seen": _fresh_iso(),
        }
        out = A.compute_delivery_audit([art], set())
        assert out["suppressed_by"]["quote_widget"] == 1
        # ``examples`` keeps a non-empty preview for the analyst to inspect
        assert "quote_widget" in out["suppressed_examples"]
        assert out["suppressed_examples"]["quote_widget"][0]["_id"] == "qw1"

    def test_recap_template_attributed(self):
        """The "Why X Stock Is Trading Up Today" template is recap_template,
        not quote_widget — even though it has a ticker-shaped substring."""
        art = {
            "_id": "rt1",
            "title": "Why Micron (MU) Stock Is Trading Up Today",
            "source": "rss",  # credible enough to clear low_authority
            "link": "https://fool.com/x",
            "published": _fresh_iso(), "first_seen": _fresh_iso(),
        }
        out = A.compute_delivery_audit([art], set())
        assert out["suppressed_by"]["recap_template"] == 1
        assert out["suppressed_by"]["quote_widget"] == 0

    def test_low_authority_attributed(self):
        """A lone reddit/social row (cred=0.40 < 0.45 bar) attributes to
        low_authority. A fresh, prose-like title that isn't a recap and
        isn't a quote widget — only the source credibility flags it."""
        art = {
            "_id": "la1",
            "title": "Random low-authority urgent post from a forum",
            "source": "reddit/r/wallstreetbets",
            "link": "https://reddit.com/r/wsb/comments/abc",
            "published": _fresh_iso(), "first_seen": _fresh_iso(),
        }
        out = A.compute_delivery_audit([art], set())
        assert out["suppressed_by"]["low_authority"] == 1

    def test_stale_attributed_after_content_gates(self):
        """A row that is stale (>24h) AND a recap template attributes to
        recap_template — the live gates run recap_template FIRST. The audit
        must mirror live precedence so the per-gate counts reflect what the
        live pipeline would have done."""
        art = {
            "_id": "st1",
            "title": "Why Tesla Stock Is Trading Up Today",  # recap fingerprint
            "source": "rss",
            "link": "https://x.com/y",
            "published": _stale_iso(36),  # also stale
            "first_seen": _stale_iso(36),
        }
        out = A.compute_delivery_audit([art], set())
        # Live order is recap_template before stale, so this must be
        # attributed to the content gate.
        assert out["suppressed_by"]["recap_template"] == 1
        assert out["suppressed_by"]["stale_published"] == 0

    def test_stale_attributed_when_no_content_gate(self):
        """A simple stale row with no other fingerprint matches must attribute
        to stale_published — proves the gate fires when nothing earlier does."""
        art = {
            "_id": "st2",
            "title": "Real prose headline that is just old",
            "source": "rss",
            "link": "https://reuters.com/x",
            "published": _stale_iso(36),
            "first_seen": _stale_iso(36),
        }
        out = A.compute_delivery_audit([art], set())
        assert out["suppressed_by"]["stale_published"] == 1


# ── INVARIANT: backtest rows never enter the audit set ────────────────────────
class TestBacktestIsolation:
    def test_synthetic_url_attributed_when_leaked(self):
        """The SQL pull excludes ``backtest://`` rows — but if a future caller
        bypasses the live-only clause and passes a synthetic row to the pure
        function, the fingerprint must catch and attribute it to ``synthetic``.
        Belt-and-braces: the SQL is the primary defense, this is the second
        line (same shape as ``alert_agent._is_synthetic`` re-check at the
        formatter)."""
        art = {
            "_id": "bt1",
            "title": "Backtest scored Q3 earnings beat for MU",
            "source": "backtest_run_42_winner",
            "link": "backtest://run_42/2026-01-01/BUY/MU",
            "published": _fresh_iso(), "first_seen": _fresh_iso(),
        }
        out = A.compute_delivery_audit([art], set())
        assert out["suppressed_by"]["synthetic"] == 1
        # And the synthetic gate fires BEFORE everything else (load-bearing).
        for k in ("quote_widget", "recap_template", "low_authority",
                  "stale_published", "unknown_gate"):
            assert out["suppressed_by"][k] == 0

    def test_live_only_clause_in_sync_with_storage(self):
        """Drift guard — if ``storage/article_store.py::_LIVE_ONLY_CLAUSE``
        is ever changed without updating this module's duplicate, the audit
        would silently start scanning rows the live pipeline excludes. The
        whole class of audit-vs-live drift bugs lives here."""
        from storage.article_store import _LIVE_ONLY_CLAUSE
        assert A.LIVE_ONLY_CLAUSE == _LIVE_ONLY_CLAUSE


# ── window default ────────────────────────────────────────────────────────────
class TestWindowDefault:
    def test_default_window_matches_recency_ttl(self):
        """Asking for a wider window than the recency TTL compares urgency=2
        rows against an already-pruned signature set and inflates
        ``suppressed`` falsely — so the default and the clamp must match
        the live TTL exactly."""
        from watchers.alert_recency import ALERT_RECENCY_TTL_HOURS
        assert A.DEFAULT_WINDOW_HOURS == ALERT_RECENCY_TTL_HOURS


# ── example block is bounded ──────────────────────────────────────────────────
class TestExamplesCap:
    def test_examples_capped_per_gate(self):
        """The audit is a calibration view — the operator needs SOME titles,
        not all of them. A 1000-row recap-template noise burst must not bloat
        the JSON payload."""
        title = "Why X Stock Is Trading Up Today"
        rows = [{
            "_id": f"r{i}", "title": title, "source": "rss",
            "link": f"https://x.com/{i}",
            "published": _fresh_iso(), "first_seen": _fresh_iso(),
        } for i in range(12)]
        out = A.compute_delivery_audit(rows, set())
        assert out["suppressed_by"]["recap_template"] == 12
        # Examples capped at _EXAMPLE_CAP (5)
        assert len(out["suppressed_examples"]["recap_template"]) == 5

    def test_examples_omitted_for_empty_buckets(self):
        """A gate with zero hits should not appear in suppressed_examples —
        keeps the JSON tight for downstream consumers."""
        out = A.compute_delivery_audit([], set())
        assert out["suppressed_examples"] == {}


# ── pure: no DB / IO ──────────────────────────────────────────────────────────
class TestPureNoIO:
    def test_pure_function_does_no_network_or_disk(self, monkeypatch):
        """The pure function must never call sqlite3, open a file, or hit
        the network — a regression here means a future refactor introduced a
        side effect that makes the function un-testable in isolation."""
        import builtins
        original_open = builtins.open

        def _blocked_open(*args, **kwargs):
            # Allow internal Python imports to keep working (importlib uses
            # open under the hood for source loading) by only blocking
            # explicit data-file paths. The audit reads no files at all, so
            # we whitelist nothing here; any data-file path would crash.
            mode = kwargs.get("mode", args[1] if len(args) > 1 else "r")
            path = args[0] if args else kwargs.get("file", "")
            if isinstance(path, (str, bytes)) and (
                str(path).endswith(".db") or "articles" in str(path)
                or "recency" in str(path)
            ):
                raise AssertionError(
                    f"compute_delivery_audit attempted to open {path!r} "
                    f"({mode!r}) — pure function must not touch disk"
                )
            return original_open(*args, **kwargs)

        monkeypatch.setattr(builtins, "open", _blocked_open)

        art = {
            "_id": "p1", "title": "Real prose headline goes here",
            "source": "rss", "link": "https://x.com/x",
            "published": _fresh_iso(), "first_seen": _fresh_iso(),
        }
        out = A.compute_delivery_audit([art], set())
        assert out["total"] == 1

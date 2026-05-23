"""Per-source recap-template pollution rate (``ArticleStore.source_recap_pollution``).

Sibling to ``urgency_label_split_by_source`` (per-collector LLM-vetted fraction
of urgent rows — *verification* angle); this answers the *content-type* angle:
of a source's urgent rows in the window, what fraction match a recap/SEO
template the urgency head over-scores? A high recap_rate is actionable signal
for the operator — "GoogleNews/Motley Fool generated 80% recap noise in the
last 24h, prune that feed".

Recap detection is injected (the storage layer must not import the analysis or
watchers gates — that would invert the dependency graph), and the test suite
verifies both the boolean-return and tuple-return matcher conventions. The
canonical matchers in the production paths
(``watchers.alert_agent._looks_like_recap_template`` and
``analysis.claude_analyst._looks_like_recap_template``) both return the
``(hit, name)`` tuple form so callers get per-fingerprint counts; the boolean
form is supported for simpler dashboard hooks and stubs.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone


def _insert(store, *, src: str, title: str, urgency: int = 1,
            url: str | None = None, source_override: str | None = None,
            hours_ago: float = 1.0) -> str:
    """Insert one article directly via the store's connection. Returns id.
    Bypasses ``insert_batch`` (which would heuristic-score / dedupe) so the
    test can pin the exact (source, urgency, age) tuple a real noise row
    would carry once it crossed the urgency head."""
    from storage.article_store import article_id, compress
    # URL must include the source so the same title from two different feeds
    # produces two distinct rows (article_id = sha256(url||title)). Otherwise
    # `INSERT OR IGNORE` collapses cross-source duplicates and the per-source
    # counts in the metric come out wrong.
    effective_url = url or f"https://x/{src}/{title[:30]}"
    aid = article_id(effective_url, title)
    now = datetime.now(timezone.utc)
    first_seen = (now - timedelta(hours=hours_ago)).isoformat()
    store.conn.execute(
        "INSERT OR IGNORE INTO articles "
        "(id, url, title, source, published, kw_score, ai_score, urgency, "
        " full_text, first_seen, cycle, ml_score, score_source) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (aid, effective_url, title,
         source_override if source_override is not None else src,
         "", 1.0, 0.0, urgency,
         compress(""), first_seen, 0, 9.5, "ml"),
    )
    store.conn.commit()
    return aid


def _seed_mixed_corpus(store):
    """A realistic mix: 3 sources with different recap pollution rates plus
    a 4th source carrying zero urgent in the window (must be excluded)."""
    # source A: 10 urgent, 8 recap (rate=0.8) — the "noise feeder"
    for i in range(8):
        _insert(store, src="GoogleNews/MotleyFool",
                title=f"Why Did Stock {i} Drop Today", hours_ago=2.0)
    for i in range(2):
        _insert(store, src="GoogleNews/MotleyFool",
                title=f"NVDA tops Q1 estimates {i}", hours_ago=2.0)
    # source B: 10 urgent, 1 recap (rate=0.1) — the "clean feeder"
    _insert(store, src="reuters", title="Why Did Stock Surge Today",
            hours_ago=3.0)
    for i in range(9):
        _insert(store, src="reuters", title=f"Fed cuts rates by 50bp #{i}",
                hours_ago=3.0)
    # source C: 4 urgent (below min_total=5), should be excluded by volume floor
    for i in range(4):
        _insert(store, src="tiny_feed",
                title=f"Why Did Stock Pop Today {i}", hours_ago=1.0)
    # source D: 0 urgent in the window (urgency=0 row) — never returned
    _insert(store, src="quiet_feed", title="A neutral newswire item",
            urgency=0, hours_ago=2.0)


# ── boolean matcher (the simplest dashboard hook) ────────────────────────────


def _bool_matcher(title: str) -> bool:
    """Stub matcher: anything containing "Why Did ... Today" is recap."""
    t = (title or "").lower()
    return t.startswith("why did") and "today" in t


def _tuple_matcher(title: str) -> tuple[bool, str]:
    """Stub matcher: returns (hit, fingerprint_name) — the SSOT convention."""
    t = (title or "").lower()
    if t.startswith("why did") and "today" in t:
        return True, "why_did_stock"
    if t.startswith("nvda tops"):
        return True, "earnings_recap"
    return False, ""


# ── per-source counts and ordering ───────────────────────────────────────────


def test_per_source_counts_recap_rate_correct(store):
    _seed_mixed_corpus(store)
    out = store.source_recap_pollution(_bool_matcher, hours=24, min_total=5)
    by_src = {r["source"]: r for r in out["by_source"]}
    # source A: 10 urgent, 8 recap, rate=0.8
    assert "GoogleNews/MotleyFool" in by_src
    a = by_src["GoogleNews/MotleyFool"]
    assert a["total"] == 10
    assert a["recap"] == 8
    assert a["recap_rate"] == 0.8
    # source B: 10 urgent, 1 recap, rate=0.1
    assert "reuters" in by_src
    b = by_src["reuters"]
    assert b["total"] == 10
    assert b["recap"] == 1
    assert b["recap_rate"] == 0.1


def test_min_total_excludes_low_volume_sources(store):
    """A source with 4 urgent rows of which 4 are recap reads "100%
    polluted" without volume to justify the verdict. ``min_total=5`` cuts
    it. The deterministic flag: the source is NOT in ``by_source`` but
    its rows STILL count toward ``total_urgent`` / ``total_recap``."""
    _seed_mixed_corpus(store)
    out = store.source_recap_pollution(_bool_matcher, hours=24, min_total=5)
    srcs = {r["source"] for r in out["by_source"]}
    assert "tiny_feed" not in srcs
    # but the 4 tiny_feed urgent + 4 recap rows DO count globally
    assert out["total_urgent"] == 10 + 10 + 4  # A + B + C, D excluded (urg=0)
    assert out["total_recap"] == 8 + 1 + 4    # A=8, B=1, C=4 (all recap)


def test_global_rate_matches_aggregate(store):
    _seed_mixed_corpus(store)
    out = store.source_recap_pollution(_bool_matcher, hours=24)
    # 13 of 24 urgent rows match the boolean matcher
    assert out["total_urgent"] == 24
    assert out["total_recap"] == 13
    assert out["global_rate"] == round(13 / 24, 4)


def test_sort_order_worst_recap_first_with_alphabetical_tiebreak(store):
    """Two equally-polluted sources sort alphabetically (stable, test-pinnable
    order — same discipline as ``urgency_label_split_by_source``). Worst-
    recap-rate sources are returned first regardless of total volume."""
    # source Z: 5/5 recap = 1.0
    for i in range(5):
        _insert(store, src="z_feed",
                title=f"Why Did Stock {i} Drop Today", hours_ago=2.0)
    # source A: 5/5 recap = 1.0 (alphabetical tiebreak puts it first)
    for i in range(5):
        _insert(store, src="a_feed",
                title=f"Why Did Stock {i} Pop Today", hours_ago=2.0)
    # source M: 5/3 recap = 0.6 (lower rate, comes after both 1.0s)
    for i in range(3):
        _insert(store, src="m_feed",
                title=f"Why Did Stock {i} Tank Today", hours_ago=2.0)
    for i in range(2):
        _insert(store, src="m_feed",
                title=f"Genuine breaking #{i}", hours_ago=2.0)
    out = store.source_recap_pollution(_bool_matcher, hours=24, min_total=5)
    names = [r["source"] for r in out["by_source"]]
    # a_feed and z_feed both at rate 1.0 — a_feed first (alphabetical),
    # m_feed at 0.6 last.
    assert names == ["a_feed", "z_feed", "m_feed"]


# ── matcher signature flexibility ────────────────────────────────────────────


def test_tuple_matcher_populates_fingerprints(store):
    _seed_mixed_corpus(store)
    out = store.source_recap_pollution(_tuple_matcher, hours=24, min_total=5)
    by_src = {r["source"]: r for r in out["by_source"]}
    a = by_src["GoogleNews/MotleyFool"]
    # 8 "why did ... today" matches the tuple matcher's why_did_stock,
    # 2 "NVDA tops Q1 estimates" matches earnings_recap
    assert a["recap"] == 10
    assert a["recap_rate"] == 1.0
    assert a["fingerprints"] == {
        "why_did_stock": 8,
        "earnings_recap": 2,
    }


def test_boolean_matcher_leaves_fingerprints_empty(store):
    _seed_mixed_corpus(store)
    out = store.source_recap_pollution(_bool_matcher, hours=24, min_total=5)
    for r in out["by_source"]:
        # boolean-only matcher → no per-name breakdown
        assert r["fingerprints"] == {}


def test_buggy_matcher_degrades_to_no_hit(store):
    """A matcher that raises on every call must NOT crash the metric —
    a buggy upstream regex set degrades to zero pollution, never an
    unhandled exception."""
    _seed_mixed_corpus(store)
    def _raising_matcher(title):
        raise RuntimeError("simulated regex compile failure")
    out = store.source_recap_pollution(_raising_matcher, hours=24, min_total=5)
    assert out["total_recap"] == 0
    assert out["global_rate"] == 0.0
    for r in out["by_source"]:
        assert r["recap"] == 0
        assert r["recap_rate"] == 0.0


# ── window scoping and invariants ────────────────────────────────────────────


def test_hours_window_excludes_aged_out_rows(store):
    """A 24h window must NOT count rows whose ``first_seen`` is older."""
    _insert(store, src="feed", title="Why Did Stock Drop Today",
            hours_ago=2.0)
    _insert(store, src="feed", title="Why Did Stock Drop Today (old)",
            hours_ago=48.0)  # outside 24h window
    out = store.source_recap_pollution(_bool_matcher, hours=24, min_total=1)
    by_src = {r["source"]: r for r in out["by_source"]}
    assert by_src["feed"]["total"] == 1
    assert by_src["feed"]["recap"] == 1


def test_backtest_rows_excluded_invariant(store):
    """Load-bearing invariant #1: synthetic backtest rows must NEVER inflate
    the per-source metric. _LIVE_ONLY_CLAUSE is the SSOT for that gate;
    this test pins it for the new method."""
    # one legit live recap row
    _insert(store, src="legit_feed", title="Why Did NVDA Drop Today",
            hours_ago=2.0)
    # a backtest:// URL with a noisy recap title — must be EXCLUDED
    _insert(store, src="backtest_run_42_winner",
            title="Why Did Stock Drop Today",
            url="backtest://run_42/2026-05-21/BUY/NVDA",
            hours_ago=2.0)
    # an opus_annotation row — same exclusion class
    _insert(store, src="opus_annotation_cycle_1",
            title="Why Did Stock Drop Today (synthetic)",
            url="backtest://annotation/cycle_1/NVDA",
            hours_ago=2.0)
    out = store.source_recap_pollution(_bool_matcher, hours=24, min_total=1)
    srcs = {r["source"] for r in out["by_source"]}
    assert "backtest_run_42_winner" not in srcs
    assert "opus_annotation_cycle_1" not in srcs
    assert out["total_urgent"] == 1  # only the legit live row counted


def test_only_urgent_rows_counted(store):
    """``urgency >= 1`` is the predicate. Rows at urgency=0 are not noise
    the analyst was alerted on, so they should not appear in the
    pollution rate at all."""
    _insert(store, src="feed", title="Why Did Stock Drop Today",
            urgency=2, hours_ago=2.0)
    _insert(store, src="feed", title="Why Did Stock Surge Today",
            urgency=1, hours_ago=2.0)
    _insert(store, src="feed", title="Why Did Stock Plunge Today",
            urgency=0, hours_ago=2.0)
    out = store.source_recap_pollution(_bool_matcher, hours=24, min_total=1)
    by_src = {r["source"]: r for r in out["by_source"]}
    assert by_src["feed"]["total"] == 2
    assert by_src["feed"]["recap"] == 2


def test_empty_window_returns_zero_structure(store):
    """A window with no urgent rows must return the deterministic empty
    shape (same zero-data discipline as ``urgency_label_split`` /
    ``ticker_mention_velocity``)."""
    out = store.source_recap_pollution(_bool_matcher, hours=24)
    assert out["by_source"] == []
    assert out["total_urgent"] == 0
    assert out["total_recap"] == 0
    assert out["global_rate"] == 0.0
    assert out["window_h"] == 24


# ── parity with the SSOT live matcher ────────────────────────────────────────


def test_live_matcher_parity_with_alert_recap_gate(store):
    """Smoke test that the production alert-side matcher is wired in shape-
    compatibly. A real recap title surfaced by the SSOT
    ``watchers.alert_agent._looks_like_recap_template`` matcher must
    appear in the per-source breakdown with its real fingerprint name.

    Pins both the new method (it accepts the SSOT matcher unchanged) and
    the alert gate (it returns the (hit, name) tuple form). A future
    refactor that breaks either side fails here."""
    from watchers.alert_agent import _looks_like_recap_template

    def _adapter(title):
        return _looks_like_recap_template({"title": title})

    _insert(store, src="GoogleNews/MotleyFool",
            title="Why Did Micron Stock Drop Today",
            hours_ago=2.0)
    _insert(store, src="GoogleNews/MotleyFool",
            title="MU earnings beat estimates",  # NOT recap
            hours_ago=2.0)
    out = store.source_recap_pollution(_adapter, hours=24, min_total=1)
    by_src = {r["source"]: r for r in out["by_source"]}
    bucket = by_src["GoogleNews/MotleyFool"]
    assert bucket["total"] == 2
    assert bucket["recap"] == 1
    assert bucket["recap_rate"] == 0.5
    assert bucket["fingerprints"] == {"why_did_stock": 1}


def test_live_matcher_parity_with_briefing_recap_gate(store):
    """Symmetric to the alert-gate parity test — the briefing's SSOT
    matcher (``analysis.claude_analyst._looks_like_recap_template``) must
    also be drop-in compatible. Pins that both lockstep gates can drive
    the new metric without adapter changes."""
    from analysis.claude_analyst import _looks_like_recap_template

    def _adapter(title):
        return _looks_like_recap_template({"title": title})

    _insert(store, src="feed",
            title="These Stocks Are Today's Movers: NVDA, AAPL, MSFT",
            hours_ago=2.0)
    _insert(store, src="feed",
            title="MU beats Q1 estimates",  # NOT recap
            hours_ago=2.0)
    out = store.source_recap_pollution(_adapter, hours=24, min_total=1)
    by_src = {r["source"]: r for r in out["by_source"]}
    bucket = by_src["feed"]
    assert bucket["total"] == 2
    assert bucket["recap"] == 1
    assert bucket["fingerprints"] == {"todays_movers_list": 1}


# ── top_n cap behaviour ──────────────────────────────────────────────────────


def test_top_n_caps_response_size(store):
    """A 10-source corpus with top_n=3 returns exactly the top 3 worst
    by recap_rate; ``total_urgent`` and ``total_recap`` remain global."""
    for s_idx in range(10):
        # rate increases from 0.0 to 0.9 across 10 sources
        n_recap = s_idx
        n_clean = 10 - s_idx
        for i in range(n_recap):
            _insert(store, src=f"src_{s_idx:02d}",
                    title=f"Why Did Stock {i} Drop Today",
                    hours_ago=2.0)
        for i in range(n_clean):
            _insert(store, src=f"src_{s_idx:02d}",
                    title=f"Genuine wire #{i}",
                    hours_ago=2.0)
    out = store.source_recap_pollution(_bool_matcher, hours=24, min_total=5,
                                       top_n=3)
    assert len(out["by_source"]) == 3
    # top 3 worst sources are src_09 (0.9), src_08 (0.8), src_07 (0.7)
    names = [r["source"] for r in out["by_source"]]
    assert names == ["src_09", "src_08", "src_07"]
    # global counts still reflect all 10 sources
    assert out["total_urgent"] == 100
    assert out["total_recap"] == sum(range(10))  # 0+1+...+9 = 45

"""Book-coverage map in the 5h heartbeat briefing.

The Opus briefing prose never states which of the operator's *positions* the
digest actually touches. A 5h window with zero mentions of a held name
(AXTI/QBTS/SNDU are thin-coverage) is a silent blind spot — the analyst can't
tell "nothing happened" from "the pipeline missed it" with money at risk.
``daemon._format_portfolio_coverage`` converts that into one explicit,
deterministic Discord-only line (exact strings pinned here, like
``_format_source_health_summary``). It is read-only and never folded into the
saved briefing text, so none of the four load-bearing invariants are touched.
"""
from __future__ import annotations

import daemon

_PC = daemon._format_portfolio_coverage


def test_empty_articles_degrades_clean():
    assert _PC([], tickers=("MU",)) == ""
    assert _PC(None, tickers=("MU",)) == ""


def test_no_tickers_degrades_clean():
    assert _PC([{"title": "MU beats", "summary": ""}], tickers=()) == ""


def test_covered_and_silent_split_exact():
    out = _PC(
        [{"title": "MU earnings beat", "summary": "NVDA guidance raised"}],
        tickers=("MU", "NVDA", "AXTI"),
    )
    assert out == "📊 Book in digest: MU·NVDA (2/3) — silent: AXTI"


def test_all_covered_has_no_silent_clause():
    out = _PC(
        [{"title": "MU and NVDA both moved", "summary": ""}],
        tickers=("MU", "NVDA"),
    )
    assert out == "📊 Book in digest: MU·NVDA (2/2)"


def test_none_covered_says_none():
    out = _PC(
        [{"title": "Macro CPI print", "summary": "Fed commentary"}],
        tickers=("MU", "NVDA"),
    )
    assert out == "📊 Book in digest: none (0/2) — silent: MU NVDA"


def test_covered_order_follows_tickers_not_text_order():
    # MU appears first in the text, AXTI second — but the line lists them in
    # the stable ``tickers`` order (AXTI, MU) so it doesn't churn cycle to
    # cycle on the same coverage set.
    out = _PC(
        [{"title": "MU rallied before AXTI did", "summary": ""}],
        tickers=("AXTI", "MU"),
    )
    assert out == "📊 Book in digest: AXTI·MU (2/2)"


def test_word_boundary_museum_is_not_mu():
    out = _PC([{"title": "MUSEUM opening downtown", "summary": ""}],
              tickers=("MU",))
    assert out == "📊 Book in digest: none (0/1) — silent: MU"


def test_mu_and_muu_are_distinct_tickers():
    out = _PC([{"title": "MUU rallied hard", "summary": ""}],
              tickers=("MU", "MUU"))
    # Only MUU seen — \bMU\b does not match inside "MUU".
    assert out == "📊 Book in digest: MUU (1/2) — silent: MU"


def test_match_is_case_sensitive_like_live_re():
    # Financial copy writes tickers uppercase; lowercase "mu" prose must not
    # false-match (mirrors ml.features._LIVE_RE — no re.IGNORECASE).
    out = _PC([{"title": "the mu particle in physics", "summary": ""}],
              tickers=("MU",))
    assert out == "📊 Book in digest: none (0/1) — silent: MU"


def test_summary_is_scanned_not_only_title():
    out = _PC([{"title": "Macro wrap", "summary": "NVDA earnings tonight"}],
              tickers=("NVDA",))
    assert out == "📊 Book in digest: NVDA (1/1)"


def test_max_chars_cap_truncates_silent_with_overflow_marker():
    tickers = ("LITE", "LNOK", "MUU", "DRAM", "SNDU", "MU",
               "MSFT", "AXTI", "ORCL", "TSEM", "QBTS", "NVDA")
    out = _PC([{"title": "MU dips premarket", "summary": ""}],
              tickers=tickers, max_chars=60)
    assert len(out) <= 60
    assert out.startswith("📊 Book in digest: MU (1/12) — silent: ")
    # 11 silent names, only the leading few fit; the rest collapse to +N.
    tail = out.split(" — silent: ", 1)[1]
    shown = tail.rsplit(" +", 1)
    n_shown = len(shown[0].split(" "))
    overflow = int(shown[1])
    assert n_shown + overflow == 11
    # Every shown token is a real silent ticker, in tickers order.
    silent_order = [t for t in tickers if t != "MU"]
    assert shown[0].split(" ") == silent_order[:n_shown]


def test_default_tickers_is_the_live_portfolio():
    # Integration: the no-tickers-arg path uses daemon.PORTFOLIO_TICKERS.
    out = _PC([{"title": "NVDA hits record", "summary": ""}])
    n = len(daemon.PORTFOLIO_TICKERS)
    assert out.startswith(f"📊 Book in digest: NVDA (1/{n}) — silent: ")
    assert "NVDA" not in out.split(" — silent: ", 1)[1].split(" +", 1)[0].split(" ")

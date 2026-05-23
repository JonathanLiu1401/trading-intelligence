"""Regression tests for the 2026-05-23 HYBRID ML+backtest review pass.

Two real bugs surfaced and were fixed during this review; both tests below
are written to FAIL on the pre-fix code and PASS on the post-fix code so a
future refactor that re-introduces either issue is caught immediately.

1. ``_rsi`` returned ``100.0`` for a perfectly FLAT series (no gain AND no
   loss across the lookback) — a spurious "severely overbought" signal that
   fed a ``-1.5`` conviction penalty in ``_ml_decide``. RSI is undefined at
   zero variance; the textbook neutral reading is ``50``. The fix returns
   ``50.0`` only when both ``avg_g`` and ``avg_l`` are zero (preserves the
   strict-all-up ``100`` and the strict-all-down ``0`` behaviour).

2. ``_LEVERAGED_ETFS`` was missing 18 leveraged-bull tickers that the
   ``WATCHLIST`` section comments explicitly classify as such — WANT, MIDU,
   TNA, UTSL (3x), SAA, UWM, GOOGU, METAU, AAPLU, CONL, SMCI2X, PLTU, USD,
   ROM, UXI, UYG (2x), BITX, ETHU (crypto 2x). Each was silently capped at
   the regular ``0.25`` conviction in ``_ml_decide`` instead of the
   documented leveraged-bull ``0.40``. The fix audits coverage and pins it
   so a future watchlist addition that ought to be in the set surfaces here.

Both tests run fully offline: ``_rsi`` is pure math; the leveraged-coverage
test only reads two module constants.
"""
from __future__ import annotations


# ───────────────────────────── RSI flat-series ─────────────────────────────


def test_rsi_strict_all_up_returns_100():
    """Sanity baseline — the fix must not regress the strict-all-up case."""
    from paper_trader.backtest import _rsi
    closes = list(range(100, 130))  # +1 every day
    assert _rsi(closes, 14) == 100.0


def test_rsi_strict_all_down_returns_0():
    """Sanity baseline — the fix must not regress the strict-all-down case."""
    from paper_trader.backtest import _rsi
    closes = list(range(130, 100, -1))  # -1 every day
    assert _rsi(closes, 14) == 0.0


def test_rsi_flat_returns_neutral_50():
    """Bug: a perfectly flat series returned 100 (severely overbought).

    Pre-fix the ``if avg_l == 0: return 100.0`` branch fired regardless of
    whether avg_g was zero too, so a flat ticker scored as "extremely
    overbought" → ``_ml_decide`` subtracted 1.5 from its ticker_score.
    Post-fix a zero-variance series reads as neutral 50, matching the
    textbook semantics for an undefined RSI.
    """
    from paper_trader.backtest import _rsi
    flat = [100.0] * 30
    assert _rsi(flat, 14) == 50.0


def test_rsi_default_period_handles_flat():
    """Same flat-series behaviour at the default period (14)."""
    from paper_trader.backtest import _rsi
    flat = [50.0] * 20
    assert _rsi(flat) == 50.0


# ─────────────────────────── _LEVERAGED_ETFS coverage ──────────────────────


# Source of truth: every ticker in WATCHLIST that the inline section comments
# classify as a "Leveraged ETF — Bull" (3x or 2x bull, plus crypto/commodity 2x).
# Inverse leveraged ETFs (SQQQ/SPXS/SOXS/TECS/FNGD/TZA/FAZ/HIBS) are
# DELIBERATELY excluded — the cap-arm gate is `regime in ("bull", "sideways")`
# where buying a leveraged-short is a counter-thesis trade.
WATCHLIST_LEVERAGED_BULL = frozenset({
    # 3x bull
    "TQQQ", "UPRO", "SPXL", "UDOW", "URTY",
    "SOXL", "TECL", "FNGU", "CURE", "LABU",
    "NAIL", "WANT", "DFEN", "MIDU", "TNA",
    "DPST", "FAS", "HIBL", "UTSL",
    # 2x bull (index / single-stock)
    "QLD", "SSO", "MVV", "SAA", "UWM",
    "NVDU", "MSFU", "AMZU", "GOOGU", "METAU",
    "TSLT", "AAPLU", "CONL", "TSLL",
    "LNOK", "SMCI2X", "PLTU",
    "USD", "ROM", "UXI", "UYG",
    # Crypto / commodity 2x
    "BITX", "BITU", "ETHU",
    "BOIL", "UCO", "AGQ",
})


def test_leveraged_bull_universe_present_in_set():
    """Every watchlist-classified leveraged-bull ticker must be in _LEVERAGED_ETFS.

    Pre-fix audit: 18 tickers were missing (WANT/MIDU/TNA/UTSL, SAA/UWM,
    GOOGU/METAU/AAPLU/CONL/SMCI2X/PLTU, USD/ROM/UXI/UYG, BITX/ETHU). Those
    names silently dropped to the regular 0.25 conviction cap instead of the
    documented 0.40 leveraged-bull cap, so the documented leveraged-vehicle
    thesis (CLAUDE.md §3) was unreachable for them.
    """
    from paper_trader.backtest import _LEVERAGED_ETFS
    missing = WATCHLIST_LEVERAGED_BULL - _LEVERAGED_ETFS
    assert not missing, (
        f"_LEVERAGED_ETFS missing documented leveraged-bull tickers: "
        f"{sorted(missing)}"
    )


def test_leveraged_set_excludes_inverse():
    """The conviction-cap arm uses regime ∈ {bull, sideways}; buying a
    leveraged-INVERSE in those regimes is a counter-thesis trade and must
    NOT receive the elevated bull cap. Pinning prevents a future
    "include every leveraged name" PR from silently flipping inverse-shorts
    into bull-cap eligibility.
    """
    from paper_trader.backtest import _LEVERAGED_ETFS
    inverses = {"SQQQ", "SPXS", "SDOW", "SRTY", "SOXS", "TECS", "FNGD",
                "TZA", "FAZ", "HIBS"}
    leaked = inverses & _LEVERAGED_ETFS
    assert not leaked, (
        f"_LEVERAGED_ETFS contains leveraged-inverse names that would "
        f"incorrectly receive the bull-conviction cap: {sorted(leaked)}"
    )


def test_leveraged_set_subset_of_watchlist():
    """Every name in _LEVERAGED_ETFS should be tradeable (in WATCHLIST).

    Catches typos and stale names from removed instruments.
    """
    from paper_trader.backtest import _LEVERAGED_ETFS, WATCHLIST
    extras = set(_LEVERAGED_ETFS) - set(WATCHLIST)
    assert not extras, (
        f"_LEVERAGED_ETFS contains tickers not in WATCHLIST "
        f"(typo or stale): {sorted(extras)}"
    )

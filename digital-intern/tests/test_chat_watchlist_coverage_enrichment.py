"""Pure-helper tests for the /api/chat watchlist-coverage enrichment.

`_watchlist_coverage_chat_lines` renders paper-trader's
`/api/watchlist-coverage` (per-watchlist-ticker attention scan) into
compact chat-context lines. The chat is otherwise position-centric and
never names a ticker the bot has IGNORED — this is the one block that
surfaces opportunity cost.

The surrounding chat handler is one large inline closure, so per the
established design (cf. `_decision_paralysis_chat_lines` /
`_cash_redeployment_chat_lines` / `_regime_leverage_fit_chat_lines`)
the logic is a total/pure function unit-tested here — no Flask, no
:8090.

Discriminating locks:

- **verbatim SSOT composition** (paper-trader invariant #10): the
  builder's own `headline` passes through UNCHANGED and the
  `by_ticker` sample is also verbatim (no chat-side re-ranking).
- **healthy = silence**: DIVERSIFIED / NO_DATA collapse to `[]`.
- **STAGNANT surfaces stale ticker sample (verbatim, capped)** — the
  drift_reasons-passthrough precedent.
- **pure/total**: non-dict / missing keys / non-list by_ticker /
  garbage rows never raise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.web_server import _watchlist_coverage_chat_lines


def _by_ticker(stales=None, actives=None):
    rows = []
    for t in stales or []:
        rows.append({
            "ticker": t, "never_seen": True,
            "last_seen_ts": None, "last_seen_action": None,
            "hours_since_last_seen": None,
            "mentions_24h": 0, "mentions_7d": 0, "action_count_7d": 0,
        })
    for t in actives or []:
        rows.append({
            "ticker": t, "never_seen": False,
            "last_seen_ts": "2026-05-21T10:00:00+00:00",
            "last_seen_action": f"BUY {t} → FILLED",
            "hours_since_last_seen": 2.0,
            "mentions_24h": 5, "mentions_7d": 7, "action_count_7d": 4,
        })
    return rows


def _rep(verdict="STAGNANT", *, headline=None,
         n_never_seen=36, n_stale_7d=36, n_active_24h=11,
         n_watchlist=48, top3=0.81,
         stales=("LITE", "AMAT", "LRCX"),
         actives=("NVDA", "MU")):
    if headline is None:
        headline = (
            f"{verdict} — {n_stale_7d} of {n_watchlist} watchlist tickers "
            "untouched in 7d+.")
    return {
        "as_of": "2026-05-21T11:00:00+00:00",
        "verdict": verdict,
        "headline": headline,
        "n_watchlist": n_watchlist,
        "n_never_seen": n_never_seen,
        "n_stale_7d": n_stale_7d,
        "n_active_24h": n_active_24h,
        "top_3_share_24h": top3,
        "by_ticker": _by_ticker(stales=list(stales),
                                actives=list(actives)),
    }


class TestPureTotalContract:
    @pytest.mark.parametrize("bad", [None, "x", 42, [], ["a"], (), object()])
    def test_non_dict_is_silence(self, bad):
        assert _watchlist_coverage_chat_lines(bad) == []

    def test_missing_verdict_is_silence(self):
        assert _watchlist_coverage_chat_lines({}) == []


class TestSilenceOnNonActionable:
    @pytest.mark.parametrize("verdict",
                             ["DIVERSIFIED", "NO_DATA", None, "OTHER"])
    def test_non_actionable_verdicts_silence(self, verdict):
        rep = _rep(verdict=verdict)
        assert _watchlist_coverage_chat_lines(rep) == []


class TestVerbatimHeadlineSSOT:
    @pytest.mark.parametrize("verdict", ["STAGNANT", "CONCENTRATED"])
    def test_headline_passes_through_verbatim(self, verdict):
        custom = f"{verdict} — totally custom string with 42 tickers [exact]"
        rep = _rep(verdict=verdict, headline=custom)
        lines = _watchlist_coverage_chat_lines(rep)
        assert lines[0] == custom


class TestStagnantStaleTickerSample:
    def test_stale_sample_surfaced_verbatim(self):
        rep = _rep(verdict="STAGNANT",
                   stales=["LITE", "AMAT", "LRCX", "KLAC"],
                   actives=["NVDA"])
        lines = _watchlist_coverage_chat_lines(rep)
        joined = "\n".join(lines)
        # All four stale tickers must appear in the stale-sample line.
        for tk in ("LITE", "AMAT", "LRCX", "KLAC"):
            assert tk in joined
        # Active ticker must NOT leak into the stale line.
        # (NVDA does appear as the "active" sentinel in actives but the
        # stale-sample line should not mention it.)
        stale_line = [ln for ln in lines if ln.lstrip().startswith("stale:")]
        assert stale_line, "expected a 'stale:' line"
        assert "NVDA" not in stale_line[0]

    def test_stale_sample_caps_at_max(self):
        # MAX_STALE_TICKERS_SHOWN = 8 (per helper docstring).
        stales = [f"T{i}" for i in range(20)]
        rep = _rep(verdict="STAGNANT", stales=stales, actives=["NVDA"])
        lines = _watchlist_coverage_chat_lines(rep)
        stale_line = next(ln for ln in lines if ln.lstrip().startswith("stale:"))
        # Count commas + 1 as ticker count.
        names = [t.strip() for t in stale_line.split(":", 1)[1].split(",")]
        assert len(names) <= 8

    def test_stale_sample_omitted_when_no_stales(self):
        # All watchlist tickers in by_ticker are "active" but n_never_seen
        # is still high (mismatch shouldn't crash; the sample line is
        # simply omitted).
        rep = _rep(verdict="STAGNANT", stales=[], actives=["NVDA", "MU"])
        lines = _watchlist_coverage_chat_lines(rep)
        assert not any(ln.lstrip().startswith("stale:") for ln in lines)

    def test_concentrated_does_not_emit_stale_sample(self):
        # CONCENTRATED verdict surfaces concentration headline + detail
        # only — never the stale-sample line.
        rep = _rep(verdict="CONCENTRATED",
                   stales=["LITE"], actives=["NVDA"])
        lines = _watchlist_coverage_chat_lines(rep)
        assert not any(ln.lstrip().startswith("stale:") for ln in lines)

    def test_garbage_by_ticker_rows_dont_raise(self):
        rep = _rep(verdict="STAGNANT", stales=["LITE"], actives=["NVDA"])
        # Inject a string row + None — must NOT crash the helper.
        rep["by_ticker"] = ["x", None, {"ticker": None}, *rep["by_ticker"]]
        lines = _watchlist_coverage_chat_lines(rep)
        assert any("LITE" in ln for ln in lines)


class TestDetailLineComposition:
    def test_detail_restates_count_fields(self):
        rep = _rep(verdict="STAGNANT", n_never_seen=36, n_stale_7d=36,
                   n_active_24h=11, n_watchlist=48)
        lines = _watchlist_coverage_chat_lines(rep)
        joined = "\n".join(lines)
        assert "36" in joined
        assert "11" in joined
        assert "48" in joined

    def test_concentrated_surfaces_top3_share(self):
        rep = _rep(verdict="CONCENTRATED", top3=0.95)
        lines = _watchlist_coverage_chat_lines(rep)
        joined = "\n".join(lines)
        assert "95%" in joined or "0.95" in joined

    def test_missing_count_fields_degrade(self):
        rep = {"verdict": "STAGNANT", "headline": "stag",
               "by_ticker": [{"ticker": "X", "never_seen": True}]}
        lines = _watchlist_coverage_chat_lines(rep)
        # No counts ⇒ no detail line; headline + stale sample only.
        assert lines[0] == "stag"
        assert any("X" in ln for ln in lines)


class TestAllActionableVerdictsFire:
    @pytest.mark.parametrize("verdict", ["STAGNANT", "CONCENTRATED"])
    def test_each_actionable_emits_at_least_headline(self, verdict):
        rep = _rep(verdict=verdict)
        lines = _watchlist_coverage_chat_lines(rep)
        assert lines
        assert lines[0].startswith(verdict)

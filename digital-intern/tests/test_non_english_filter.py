"""Non-English title pre-floor — defense-in-depth gate for the noise pre-floor chain.

Pin the live failure-class headlines (Arasan Chip Systems multi-language
copies, Brazilian Portuguese GDELT wire, CordenPharma French acquisition
press release) as MUST-SUPPRESS, and the must-survive English-language
corpus (real wires with accented proper nouns, generic English headlines)
as MUST-KEEP. Same shape as ``tests/test_alert_paraphrase_suppression``
and the recap-template tests — every entry traces back to a real DB row.
"""
from __future__ import annotations

import pytest

from watchers.non_english_filter import looks_non_english, filter_non_english_noise


# ─── MUST-SUPPRESS: live noise samples from the 30d urgency>=2 audit ─────────
class TestNonEnglishDetection:
    """Each test corresponds to a real BREAKING-alert row that fired on the
    consumer's Discord channel (live evidence 2026-05-19..30 articles.db).

    The single-line title is the EXACT title text from that row."""

    def test_spanish_anuncia(self):
        # 2026-05-29 14:32:52Z — PR Newswire, ml_score=9.97, urgency=2
        assert looks_non_english({
            "title": "Arasan Chip Systems anuncia la primera solución IP "
                     "Sureboot™ Total de 16 bits xSPI + PSRAM",
        })

    def test_german_kundigt(self):
        # 2026-05-29 14:19:37Z — PR Newswire, ml_score=9.9, urgency=2
        assert looks_non_english({
            "title": "Arasan Chip Systems kündigt branchenweit erste "
                     "Sureboot™ Total 16-bit xSPI + PSRAM IP-Lösung",
        })

    def test_french_annonce_premiere(self):
        # 2026-05-29 14:19:36Z — PR Newswire, ml_score=9.95, urgency=2
        assert looks_non_english({
            "title": "Arasan Chip Systems annonce la première solution IP "
                     "Sureboot™ Total 16 bits xSPI + PSRAM",
        })

    def test_french_acquiert_accroitre(self):
        # 2026-05-27 11:30:18Z — PR Newswire, ml_score=9.74, urgency=2
        assert looks_non_english({
            "title": "CordenPharma acquiert AmbioPharm pour accroître sa "
                     "capacité mondiale de production d'API peptidiques",
        })

    def test_portuguese_gastara_bilhoes(self):
        # 2026-05-27 15:35:15Z — GDELT/infomoney.com.br, ml_score=9.93, urgency=2
        assert looks_non_english({
            "title": "Nvidia gastará US$ 150 bilhões por ano em Taiwan, "
                     "diz CEO",
        })


# ─── MUST-KEEP: real English headlines that include accented proper nouns ────
class TestEnglishHeadlinesSurvive:
    """The two-signal AND (non-English stopword + diacritic) is what
    prevents false positives. These English headlines all contain an
    accented proper noun but no Romance/Germanic stopword."""

    def test_sao_paulo_in_english(self):
        assert not looks_non_english({
            "title": "São Paulo investors flee equities as inflation surges",
        })

    def test_volkswagen_munster(self):
        assert not looks_non_english({
            "title": "Volkswagen halts Münster plant production amid chip shortage",
        })

    def test_cafe_culture_piece(self):
        assert not looks_non_english({
            "title": "Café-chain stocks rally on raw-material price drop",
        })

    def test_pure_english_news(self):
        assert not looks_non_english({
            "title": "Nvidia beats Q1 estimates on AI chip demand",
        })

    def test_short_title_survives(self):
        # Don't fire on ultra-short titles — _MIN_TITLE_LEN guard
        assert not looks_non_english({"title": "MU beats"})

    def test_empty_title_survives(self):
        assert not looks_non_english({"title": ""})
        assert not looks_non_english({})
        assert not looks_non_english({"title": None})

    def test_english_premiere_no_accent(self):
        """``premi[èe]re`` matches case-insensitively but only when the
        accented form appears. A plain English ``premiere`` lacks the
        accent and the diacritic gate also fails — survives."""
        assert not looks_non_english({
            "title": "Hollywood premiere of chip industry documentary draws crowds",
        })


# ─── Partition helper ────────────────────────────────────────────────────────
class TestFilterPartition:
    def test_split_keeps_order(self):
        arts = [
            {"title": "Nvidia beats Q1 estimates on AI chip demand"},
            {"title": "Arasan Chip Systems anuncia la primera solución IP Sureboot"},
            {"title": "Micron upgrades guidance after HBM demand surge"},
            {"title": "Nvidia gastará US$ 150 bilhões por ano em Taiwan"},
        ]
        kept, suppressed = filter_non_english_noise(arts)
        assert [a["title"][:6] for a in kept] == ["Nvidia", "Micron"]
        assert len(suppressed) == 2

    def test_empty_input_returns_empty(self):
        kept, suppressed = filter_non_english_noise([])
        assert kept == [] and suppressed == []


# ─── Integration: prefloor_pseudo_articles must use the new gate ─────────────
def _recent_iso():
    from datetime import datetime, timedelta, timezone
    return (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()


def _insert_raw(store, *, id, url, title, source, kw_score=1.0):
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, "
            "urgency, first_seen, cycle) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, "", kw_score, 0.0, 0,
             _recent_iso(), 0),
        )
        store.conn.commit()


class TestPrefloorWiresNonEnglish:
    """The new gate is wired into ``prefloor_pseudo_articles`` so the
    daemon's ML scoring path floors non-English rows before they reach
    urgency=1. Without this integration the gate would only suppress at
    the alert layer (and the urgency=1 row would still pollute the
    queue, training pool, and dashboard calibration metrics)."""

    def test_non_english_row_is_prefloored(self, store):
        _insert_raw(
            store, id="es1",
            url="https://prnewswire.com/arasan-es",
            title="Arasan Chip Systems anuncia la primera solución IP Sureboot",
            source="PR Newswire",
        )
        _insert_raw(
            store, id="en1",
            url="https://reuters.com/mu-beats",
            title="Micron beats Q1 on HBM demand",
            source="rss",
        )
        unscored = store.get_unscored(min_kw=0.0)
        real, n_pre = store.prefloor_pseudo_articles(unscored)
        # Non-English row pre-floored to ml_score=0.01, urgency=0,
        # score_source='ml'; English row passes through to inference.
        assert n_pre == 1
        assert [a["_id"] for a in real] == ["en1"]
        row = store.conn.execute(
            "SELECT ml_score, urgency, ai_score, score_source "
            "FROM articles WHERE id='es1'"
        ).fetchone()
        assert row[0] == pytest.approx(0.01)
        assert row[1] == 0, "non-English pre-floor must NOT set urgency=1"
        assert row[2] == 0, "ai_score must not be polluted by ML pre-floor"
        assert row[3] == "ml", "score_source must reflect ML-path origin"

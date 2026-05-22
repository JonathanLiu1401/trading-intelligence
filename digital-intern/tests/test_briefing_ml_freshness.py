"""ML-scorer staleness block in the 5h Opus heartbeat briefing.

ArticleNet scores every collected article and produces the ``[model]`` urgent
calls. When the ml_trainer worker fails persistently the model silently stops
learning new labels (observed live 2026-05-22: train() returned
``{"status":"error","reason":"subprocess_timeout"}`` every cycle and
``data/ml/training_metrics.jsonl`` went unwritten for ~80h). COVERAGE GAP
surfaces dark COLLECTORS; this block surfaces a stale SCORER.

These pin: the pure staleness renderer (omit-when-fresh, warn-when-stale,
degrade-on-bad-input), the best-effort metrics-file reader, and the
``_build_payload`` integration (omit-when-None, reproduced-section shape) —
all read-only, no DB / ai_score / ml_score / score_source / urgency touch.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from analysis import claude_analyst


_NOW = datetime(2026, 5, 22, 3, 0, 0, tzinfo=timezone.utc)


def _ts(hours_ago: float) -> str:
    return (_NOW - timedelta(hours=hours_ago)).isoformat()


# ── _ml_freshness_lines: the pure renderer ───────────────────────────────────
class TestMlFreshnessLines:
    def test_fresh_model_emits_nothing(self):
        # Retrained 20min ago — healthy, section omitted.
        assert claude_analyst._ml_freshness_lines(
            {"last_ts": _ts(0.33)}, now=_NOW
        ) == []

    def test_just_under_threshold_is_silent(self):
        # 5.9h < 6h warn threshold — still silent.
        assert claude_analyst._ml_freshness_lines(
            {"last_ts": _ts(5.9)}, now=_NOW
        ) == []

    def test_stale_model_emits_one_line_with_age(self):
        out = claude_analyst._ml_freshness_lines(
            {"last_ts": _ts(80.0)}, now=_NOW
        )
        assert len(out) == 1
        # Age is surfaced so the analyst sees HOW stale.
        assert "~80.0h" in out[0]
        assert "stale weights" in out[0]

    def test_exactly_at_threshold_emits(self):
        out = claude_analyst._ml_freshness_lines(
            {"last_ts": _ts(6.0)}, now=_NOW
        )
        assert len(out) == 1

    def test_future_ts_is_silent(self):
        # Clock skew / bad row → negative age → never a false stale alarm.
        assert claude_analyst._ml_freshness_lines(
            {"last_ts": _ts(-3.0)}, now=_NOW
        ) == []

    def test_unparseable_ts_is_silent(self):
        assert claude_analyst._ml_freshness_lines(
            {"last_ts": "not-a-date"}, now=_NOW
        ) == []

    def test_missing_or_bad_input_is_silent(self):
        assert claude_analyst._ml_freshness_lines(None, now=_NOW) == []
        assert claude_analyst._ml_freshness_lines({}, now=_NOW) == []
        assert claude_analyst._ml_freshness_lines(
            {"last_ts": None}, now=_NOW
        ) == []
        assert claude_analyst._ml_freshness_lines("stale", now=_NOW) == []

    def test_z_suffix_ts_parses(self):
        # ml.trainer writes ts as "...Z" (strftime %Y-%m-%dT%H:%M:%SZ).
        z_ts = (_NOW - timedelta(hours=80)).strftime("%Y-%m-%dT%H:%M:%SZ")
        out = claude_analyst._ml_freshness_lines({"last_ts": z_ts}, now=_NOW)
        assert len(out) == 1


# ── _collect_ml_freshness: best-effort file read ─────────────────────────────
class TestCollectMlFreshness:
    def test_reads_last_record_ts(self, tmp_path, monkeypatch):
        f = tmp_path / "training_metrics.jsonl"
        f.write_text(
            json.dumps({"ts": "2026-05-18T00:00:00Z", "phase": "train",
                        "status": "ok"}) + "\n" +
            json.dumps({"ts": "2026-05-18T18:17:24Z", "phase": "train",
                        "status": "ok"}) + "\n"
        )
        monkeypatch.setenv("DIGITAL_INTERN_ML_DIR", str(tmp_path))
        assert claude_analyst._collect_ml_freshness() == {
            "last_ts": "2026-05-18T18:17:24Z"
        }

    def test_skips_corrupt_lines_keeps_last_valid(self, tmp_path, monkeypatch):
        f = tmp_path / "training_metrics.jsonl"
        f.write_text(
            json.dumps({"ts": "2026-05-18T00:00:00Z"}) + "\n"
            "{ this is not json\n"
            "\n"
            + json.dumps({"ts": "2026-05-20T12:00:00Z"}) + "\n"
        )
        monkeypatch.setenv("DIGITAL_INTERN_ML_DIR", str(tmp_path))
        assert claude_analyst._collect_ml_freshness() == {
            "last_ts": "2026-05-20T12:00:00Z"
        }

    def test_missing_file_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DIGITAL_INTERN_ML_DIR", str(tmp_path / "nope"))
        assert claude_analyst._collect_ml_freshness() is None

    def test_empty_file_returns_none(self, tmp_path, monkeypatch):
        (tmp_path / "training_metrics.jsonl").write_text("")
        monkeypatch.setenv("DIGITAL_INTERN_ML_DIR", str(tmp_path))
        assert claude_analyst._collect_ml_freshness() is None


# ── _build_payload integration ───────────────────────────────────────────────
def _has_ml_section(payload: str) -> bool:
    return "=== ML SCORER STALE" in payload


class TestBuildPayloadIntegration:
    _ART = [{"title": "Fed signals surprise hold", "source": "rss",
             "ai_score": 8.0, "summary": "macro",
             "link": "https://reuters.com/fed"}]

    def test_none_omits_section(self):
        # 7-arg path / callers that don't pass ml_freshness stay deterministic.
        out = claude_analyst._build_payload(self._ART, {}, [])
        assert not _has_ml_section(out)

    def test_fresh_model_omits_section(self):
        out = claude_analyst._build_payload(
            self._ART, {}, [],
            ml_freshness={"last_ts": datetime.now(timezone.utc).isoformat()},
        )
        assert not _has_ml_section(out)

    def test_stale_model_renders_section(self):
        stale = (datetime.now(timezone.utc) - timedelta(hours=80)).isoformat()
        out = claude_analyst._build_payload(
            self._ART, {}, [], ml_freshness={"last_ts": stale},
        )
        assert _has_ml_section(out)
        assert "stale weights" in out

    def test_stale_block_does_not_mutate_caller_articles(self):
        a = {"title": "NVDA rallies", "source": "rss", "ai_score": 8.0,
             "summary": "z", "link": "https://reuters.com/n"}
        before = dict(a)
        stale = (datetime.now(timezone.utc) - timedelta(hours=80)).isoformat()
        claude_analyst._build_payload([a], {}, [], ml_freshness={"last_ts": stale})
        assert a == before


def test_system_prompt_documents_ml_scorer_stale():
    sp = claude_analyst.SYSTEM_PROMPT
    # The rule must be defined AND reproduced as an output section.
    assert "ML SCORER STALE" in sp
    assert sp.count("ML SCORER STALE") >= 2

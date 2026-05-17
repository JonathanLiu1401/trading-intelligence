"""Tests for core.json_extract.extract_json_array.

The strict path was previously only smoke-tested in ``__main__``; the
truncation-salvage fallback is the production-critical behavior (every LLM
batch caller depends on it not silently dropping a whole batch when Claude
hits its output-token limit mid-array), so it gets explicit coverage here.
"""
from core.json_extract import extract_json_array


class TestStrictPath:
    def test_bare_array(self):
        assert extract_json_array('[{"a": 1}]') == [{"a": 1}]

    def test_leading_and_trailing_prose(self):
        assert extract_json_array('Here you go: [1, 2, 3]. Done.') == [1, 2, 3]

    def test_first_complete_array_wins_over_later_one(self):
        assert extract_json_array('prose [1] more [{"x": 9}] tail') == [1]

    def test_object_before_array_is_skipped(self):
        assert extract_json_array('json: {"not": "array"} then [7]') == [7]

    def test_trailing_stray_bracket_does_not_break_parse(self):
        # A greedy ``\[.*\]`` regex would span into "[done]" and fail.
        assert extract_json_array('[{"index": 0, "score": 7}] [done]') \
            == [{"index": 0, "score": 7}]

    def test_empty_array(self):
        assert extract_json_array('[ ]') == []

    def test_none_and_empty_and_no_array(self):
        assert extract_json_array(None) is None
        assert extract_json_array('') is None
        assert extract_json_array('no array here') is None


class TestTruncationSalvage:
    def test_complete_array_still_uses_strict_path(self):
        assert extract_json_array('[{"a": 1}, {"b": 2}]') == [{"a": 1}, {"b": 2}]

    def test_truncated_mid_object_recovers_prefix(self):
        assert extract_json_array('[{"a": 1}, {"b": 2}, {"c":') \
            == [{"a": 1}, {"b": 2}]

    def test_truncated_mid_string_value_recovers_prefix(self):
        assert extract_json_array('[{"a": 1}, {"b": "untermin') == [{"a": 1}]

    def test_truncated_with_trailing_comma(self):
        assert extract_json_array('[{"a": 1}, {"b": 2},') == [{"a": 1}, {"b": 2}]

    def test_salvage_handles_complete_nested_objects(self):
        # Real LLM batch payloads are flat-ish objects; nested *objects*
        # (not nested arrays) must salvage correctly.
        assert extract_json_array('[{"a": {"x": 1}}, {"b": {"y": 2}}, {"c') \
            == [{"a": {"x": 1}}, {"b": {"y": 2}}]

    def test_salvage_skips_stray_leading_bracket(self):
        assert extract_json_array('see [x] then [{"a": 1}, {"b": 2') \
            == [{"a": 1}]

    def test_realistic_index_keyed_batch_truncation(self):
        raw = ('[{"index": 0, "score": 9, "reason": "MU beat"}, '
               '{"index": 1, "score": 2, "reason": "noise"}, '
               '{"index": 2, "score": 7, "rea')
        assert extract_json_array(raw) == [
            {"index": 0, "score": 9, "reason": "MU beat"},
            {"index": 1, "score": 2, "reason": "noise"},
        ]

    def test_zero_recoverable_elements_returns_none(self):
        # Callers' existing "parse failed" branch must still fire.
        assert extract_json_array('[{"a":') is None
        assert extract_json_array('[bogus truncated') is None
        assert extract_json_array('[}') is None

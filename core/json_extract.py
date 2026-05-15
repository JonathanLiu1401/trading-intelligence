"""Robust JSON array extraction from LLM responses.

Claude (Sonnet/Opus) is asked to reply with *only* a JSON array, but in
practice it intermittently wraps the array in prose ("Here is the array: …",
trailing "Note that …"). A naive ``re.search(r"\\[.*\\]")`` is greedy: it spans
from the first ``[`` to the *last* ``]`` in the whole response, so any trailing
bracketed text ("… see [1] above") makes the captured slice invalid JSON and
the parse fails outright.

``extract_json_array`` instead walks every ``[`` in the response and lets
``json.JSONDecoder().raw_decode`` consume exactly one well-formed value at that
offset, returning the first one that decodes to a list. This tolerates leading
prose, trailing prose, and stray brackets, and is the shared implementation
used by every LLM-batch caller in the codebase.
"""
from __future__ import annotations

import json

_decoder = json.JSONDecoder()


def extract_json_array(raw: str | None):
    """Return the first top-level JSON array found in ``raw``, or ``None``.

    Tries ``raw_decode`` at each ``[`` until one parses to a list. Prose before
    or after the array is ignored.
    """
    if not raw:
        return None
    start = raw.find("[")
    while start != -1:
        try:
            value, _ = _decoder.raw_decode(raw[start:])
            if isinstance(value, list):
                return value
        except ValueError:
            pass
        start = raw.find("[", start + 1)
    return None


if __name__ == "__main__":  # smoke test
    assert extract_json_array('[{"a": 1}]') == [{"a": 1}]
    assert extract_json_array('Here you go: [1, 2, 3]. Done.') == [1, 2, 3]
    assert extract_json_array('prose [1] more [{"x": 9}] tail') == [1]
    assert extract_json_array('json: {"not": "array"} then [7]') == [7]
    assert extract_json_array('no array here') is None
    assert extract_json_array('') is None
    assert extract_json_array(None) is None
    # Greedy regex would fail this one: trailing "[done]" breaks \[.*\].
    assert extract_json_array('[{"index": 0, "score": 7}] [done]') == [{"index": 0, "score": 7}]
    print("OK")

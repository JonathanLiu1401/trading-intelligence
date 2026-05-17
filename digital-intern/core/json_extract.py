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

**Truncation salvage.** When a batch is large enough that Claude hits its
output-token limit, the array comes back cut off mid-element
(``[{"index":0,...},{"index":1,...},{"index":2,"sc``). Strict ``raw_decode``
fails on that ``[`` and would discard the *entire* batch — including the dozens
of complete, valid objects before the cut. Since every caller maps results back
by ``index``/``url`` and simply re-queues whatever is missing, a partial list is
safe and strictly better than nothing. So when no ``[`` yields a complete array,
``_salvage_truncated_array`` recovers the leading run of complete top-level
elements from the first ``[`` and returns those instead.
"""
from __future__ import annotations

import json

_decoder = json.JSONDecoder()


def _salvage_truncated_array(s: str):
    """Recover the leading complete elements of a truncated JSON array.

    ``s`` must start at the ``[`` of the (possibly cut-off) array. Walks element
    by element with ``raw_decode``, skipping the inter-element whitespace/commas
    the decoder will not skip itself, and stops at the first element that fails
    to decode (the truncation point or trailing prose). Returns the collected
    list, or ``None`` if not a single complete element could be recovered.
    """
    if not s or s[0] != "[":
        return None
    idx = 1  # past the opening '['
    out: list = []
    n = len(s)
    while idx < n:
        # The decoder does not skip leading whitespace; advance past the
        # inter-element separators (commas + whitespace) ourselves.
        while idx < n and s[idx] in " \t\r\n,":
            idx += 1
        if idx >= n or s[idx] == "]":
            break
        try:
            value, end = _decoder.raw_decode(s, idx)
        except ValueError:
            # Truncated mid-element, or trailing prose — stop here and keep
            # whatever complete elements we already have.
            break
        out.append(value)
        idx = end
    return out or None


def extract_json_array(raw: str | None):
    """Return the first top-level JSON array found in ``raw``, or ``None``.

    Tries ``raw_decode`` at each ``[`` until one parses to a list. Prose before
    or after the array is ignored. If no ``[`` yields a complete array (e.g. the
    response was truncated at the output-token limit), falls back to recovering
    the complete leading elements of the first ``[`` via salvage.
    """
    if not raw:
        return None
    # Pass 1 — strict: the first ``[`` that yields a complete JSON list wins.
    start = raw.find("[")
    while start != -1:
        try:
            value, _ = _decoder.raw_decode(raw[start:])
            if isinstance(value, list):
                return value
        except ValueError:
            pass
        start = raw.find("[", start + 1)
    # Pass 2 — salvage: no complete array anywhere, so the response was almost
    # certainly truncated. Recover the leading complete elements from the first
    # ``[`` that yields at least one (skipping any stray ``[`` in leading prose).
    start = raw.find("[")
    while start != -1:
        salvaged = _salvage_truncated_array(raw[start:])
        if salvaged is not None:
            return salvaged
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
    # Truncation salvage: a complete array still uses the strict fast path…
    assert extract_json_array('[{"a": 1}, {"b": 2}]') == [{"a": 1}, {"b": 2}]
    # …but a cut-off array recovers its complete leading elements.
    assert extract_json_array('[{"a": 1}, {"b": 2}, {"c":') == [{"a": 1}, {"b": 2}]
    assert extract_json_array('[{"a": 1}, {"b": "untermin') == [{"a": 1}]
    assert extract_json_array('[{"a": 1}, {"b": 2},') == [{"a": 1}, {"b": 2}]
    assert extract_json_array('[{"a": {"x": 1}}, {"b": {"y": 2}}, {"c') == [{"a": {"x": 1}}, {"b": {"y": 2}}]
    assert extract_json_array('see [x] then [{"a": 1}, {"b": 2') == [{"a": 1}]
    assert extract_json_array('[{"a":') is None
    assert extract_json_array('[bogus truncated') is None
    print("OK")

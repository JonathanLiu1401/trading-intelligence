"""Deterministic URL canonicalization for exact-duplicate collapse at ingest.

``storage.article_store`` keys every row on ``sha256(url || title)``. That is
exact: the *same* story, from the *same* publisher, arriving with a different
tracking suffix produces a *different* id and lands as its own row — scored
again, re-alerted, padding the feed the analyst reads. The common offenders in
this system's collectors:

  * RSS / FeedBurner / Google News append ``?utm_source=…&utm_medium=…`` and a
    ``fbclid`` / ``gclid`` click id — the article body is byte-identical.
  * The same piece is linked as ``http://`` from one feed and ``https://`` from
    another, with/without ``www.``/``m.``, with/without a trailing slash, or
    with a ``#section`` fragment.
  * Publishers expose an AMP twin (``…/amp`` path or ``?outputType=amp``) that
    is the same story under a different URL.
  * Google News RSS wraps the real link in a ``…?url=<real>`` redirect.

``ml.dedup`` is the *complementary* detector — fuzzy, order-independent Jaccard
over **titles**, for cross-publisher syndication. This module is the exact,
deterministic half: collapse the trivially-equivalent URL *variants of one
publisher's one article* so they never become two rows in the first place.

Design notes:
  * One pure total function — no DB, no LLM, no network, never raises. Safe to
    call anywhere (collector, ingest, a read-only snapshot) and trivially
    testable with exact-value assertions.
  * Idempotent: ``canonicalize_url(canonicalize_url(u)) == canonicalize_url(u)``.
  * **Backtest isolation is load-bearing here.** Synthetic rows use
    ``backtest://…`` URLs and the live-only filter is a ``url NOT LIKE
    'backtest://%'`` LIKE clause. Any non-``http(s)`` scheme is therefore
    returned *verbatim* — canonicalization is a strict no-op on ``backtest://``
    so the isolation clause can never be defeated by it. Pinned by a test.
  * Title is passed through untouched: title normalization is ``ml.dedup``'s
    responsibility, kept separate on purpose.

Integration is intentionally left to the caller. ``canonical_article_id`` is a
drop-in for ``storage.article_store.article_id`` — swapping it at the ingest
id-computation site collapses tracking-param variants to a single row while
leaving every existing id stable for URLs that were already canonical.
"""
from __future__ import annotations

import hashlib
from urllib.parse import (
    parse_qsl,
    quote,
    unquote,
    urlencode,
    urlsplit,
    urlunsplit,
)

# Query parameters that are pure click/campaign tracking: dropping them never
# changes which document the URL addresses. Exact-name set first, then a small
# set of vendor prefixes that are unambiguous (a broad ``at_``/``__`` prefix was
# rejected — it risks merging genuinely distinct articles).
_TRACKING_EXACT = frozenset({
    "fbclid", "gclid", "gclsrc", "dclid", "msclkid", "yclid", "twclid",
    "igshid", "mc_eid", "mc_cid", "ref", "ref_src", "ref_url", "cmpid",
    "ncid", "spm", "scid", "s_kwcid", "ito", "cmp", "icid", "recirc",
    "taid", "guccounter", "__twitter_impression", "at_medium",
    "at_campaign", "at_custom1", "at_custom2", "at_custom3", "at_custom4",
    "at_link_origin", "ns_campaign", "ns_mchannel", "ns_source",
    "wt_mc", "WT.mc_id", "sr_share", "smid", "smtyp",
    # Yahoo Finance syndication referrer markers — pure source-tracking, never
    # content-addressing. Live evidence (2026-05-29): the same Barron's
    # article "Micron Faces New Threat From Samsung's Memory Chip for AI"
    # fired a BREAKING push 3× in 24h via referrer-param variants
    # (?siteid=yhoof2&ypt=1 from yfinance/Barrons.com vs ?mod=md_home_pan_m
    # from scraped/www.barrons.com — same /articles/<slug>-ac9a8e59 path).
    "siteid", "ypt",
    # Dow Jones (Barron's, WSJ, MarketWatch) module-referrer marker — names
    # the section / module the user clicked from (mod=djhpsi, mod=md_home_pan_m,
    # mod=home, etc). Pure click-origin tracking; same content, same slug.
    # Safe to strip globally — if a non-DJN site ever uses ?mod= as a content
    # selector, the article title in the id hash already differs, so this
    # cannot wrongly collapse two distinct articles into one id.
    "mod",
})
_TRACKING_EXACT_LOWER = frozenset(n.lower() for n in _TRACKING_EXACT)
_TRACKING_PREFIXES = (
    "utm_", "oly_enc_", "oly_anon_", "guce_", "vero_", "pk_", "mtm_",
    "piwik_", "_hsenc", "_hsmi",
)

# AMP query markers (path-suffix AMP is handled separately).
_AMP_PARAMS = frozenset({"amp", "outputtype", "amp_js_v"})

# Subdomain prefixes that address the same resource as the bare host for
# dedup purposes (mobile / AMP / FeedBurner-proxy mirrors).
_HOST_PREFIXES = ("www.", "m.", "amp.", "mobile.")

_DEFAULT_PORTS = {"http": "80", "https": "443"}


def _is_tracking(name: str) -> bool:
    low = name.lower()
    if low in _TRACKING_EXACT_LOWER:
        return True
    return any(low.startswith(p) for p in _TRACKING_PREFIXES)


def canonicalize_url(url: str, _depth: int = 0) -> str:
    """Return a canonical form of *url* for exact-duplicate detection.

    Total and idempotent: garbage / empty input yields ``""``; a non-HTTP(S)
    scheme (notably ``backtest://``) is returned stripped-but-otherwise-verbatim
    so backtest isolation is never weakened. Never raises.
    """
    if not url or not isinstance(url, str):
        return ""
    url = url.strip()
    if not url:
        return ""

    try:
        parts = urlsplit(url)
    except (ValueError, TypeError):
        return url

    scheme = parts.scheme.lower()
    # Only http(s) is canonicalized. Everything else (backtest://, mailto:,
    # ftp:, relative/no-scheme) is returned untouched — the backtest-isolation
    # LIKE clause depends on this no-op.
    if scheme not in ("http", "https"):
        return url

    # ``.hostname``/``.port`` are properties that raise ValueError on a
    # malformed authority (bad IPv6, non-numeric port). This is a total
    # function, so fall back to the untouched input rather than propagate.
    try:
        host = (parts.hostname or "").lower()
        port = parts.port
    except ValueError:
        return url
    for pfx in _HOST_PREFIXES:
        if host.startswith(pfx) and len(host) > len(pfx) + 1:
            host = host[len(pfx):]
            break

    # Drop default ports; keep explicit non-default ones.
    netloc = host
    if port is not None and str(port) != _DEFAULT_PORTS.get(scheme):
        netloc = f"{host}:{port}"

    query_pairs = parse_qsl(parts.query, keep_blank_values=True)

    # Google News (and generic ?url=/?q=) redirect wrappers: the real article
    # is a nested param. Unwrap once (depth-guarded) and re-canonicalize it.
    if _depth < 3:
        for key in ("url", "q"):
            for name, value in query_pairs:
                if name.lower() == key and value:
                    target = unquote(value)
                    tp = urlsplit(target)
                    if tp.scheme in ("http", "https") and tp.netloc:
                        return canonicalize_url(target, _depth + 1)

    has_amp_param = False
    kept = []
    for name, value in query_pairs:
        if _is_tracking(name):
            continue
        if name.lower() in _AMP_PARAMS:
            has_amp_param = True
            continue
        kept.append((name, value))
    # Stable param order so ?a=1&b=2 and ?b=2&a=1 collapse together.
    kept.sort()
    query = urlencode(kept, doseq=True)

    path = parts.path
    # Collapse accidental duplicate slashes within the path.
    while "//" in path:
        path = path.replace("//", "/")
    # Strip an AMP path twin: trailing /amp or /amp/ segment.
    if path.endswith("/amp") or path.endswith("/amp/"):
        has_amp_param = True
        path = path[: path.rfind("/amp")]
    _ = has_amp_param  # AMP is folded into the canonical (non-AMP) form above.
    # Normalize a single trailing slash, but never empty the root path.
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    if not path:
        path = "/"
    # Re-quote the path so equivalent encodings (e.g. %2D vs -) coincide,
    # without corrupting an already-encoded path.
    path = quote(unquote(path), safe="/%:@!$&'()*+,;=~")

    # Fragments are dropped (same document) EXCEPT a hashbang (#!...) which is
    # legacy SPA routing that genuinely addresses a distinct view.
    fragment = parts.fragment
    fragment = fragment if fragment.startswith("!") else ""

    # Scheme is forced to https: http/https of one resource is one article.
    return urlunsplit(("https", netloc, path, query, fragment))


def canonical_article_id(url: str, title: str) -> str:
    """``sha256(canonical_url || title)`` — a drop-in for
    ``storage.article_store.article_id`` that additionally collapses
    tracking-param / AMP / scheme / slash variants of one article to a single
    id. Falls back to the raw *url* when it canonicalizes to ``""`` (e.g. a
    malformed link) so an id is always produced.
    """
    canon = canonicalize_url(url) or (url or "").strip()
    return hashlib.sha256(f"{canon}||{title}".encode()).hexdigest()

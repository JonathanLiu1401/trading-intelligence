"""Hidden-factor-bet alarm — translate the pairwise correlation matrix
into "is my book secretly one trade?"

``analytics/correlation.py`` (``/api/correlation``) returns the full
pairwise ρ matrix plus a single mean-ρ verdict
(``DIVERSIFIED / MODERATE / CONCENTRATED / SINGLE_NAME_RISK``). What it
cannot see is the *cluster structure*: a 5-name book where 3 names form a
high-ρ semis cluster and 2 names are uncorrelated cash equivalents will
read as ``MODERATE`` on mean ρ — exactly the regime where the book is
secretly running a concentrated semis bet inside a diversified-looking
wrapper.

``build_correlation_cluster_warning`` is the missing lens: take the
correlation builder's existing payload (pairs + weights), single-linkage
the names at ρ ≥ ``HIGH_CORR`` (the same constant the parent module
already uses for its ``CONCENTRATED`` threshold), and report the largest
multi-name cluster — its tickers, its share of book by market value, and
its internal mean correlation. The verdict ladder is read off the
cluster's *book-weight share*, not the mean ρ, so it catches the
"5 names, 3 are one bet" regime the mean-ρ verdict misses.

Design parity with the codebase:

* **Pure builder, no IO.** Takes the upstream ``correlation`` payload as
  input (the endpoint already paid for the yfinance fetch). Mirrors the
  ``thesis_drift`` / ``correlation`` split.
* **Reuses the parent threshold.** Edges live at ρ ≥ ``HIGH_CORR``
  (currently 0.70). Changing the parent's threshold here would have made
  the two endpoints disagree about what "highly correlated" means; the
  module imports the constant explicitly so they cannot drift.
* **Sample-size honest** — degrades to ``state="INSUFFICIENT"`` rather
  than fabricating a cluster when the parent reported ``NO_DATA`` /
  ``INSUFFICIENT`` upstream.

Advisory only — never gates Opus, adds no caps (AGENTS.md #2/#12).
"""
from __future__ import annotations

from datetime import datetime, timezone

from .correlation import HIGH_CORR

# Cluster-weight verdict thresholds. Read as "this cluster is X% of the
# book by market value". 0.30 / 0.60 mirror the spirit of the parent
# module's ``DIVERSIFIED_MAX_TOP_WEIGHT`` (0.50) / ``DOMINANT_WEIGHT``
# (0.60) on a single name, generalised to a co-moving cluster.
WATCHLIST_WEIGHT = 0.30
DOMINANT_WEIGHT = 0.60
# A cluster needs at least this many names to be called a "cluster" — a
# single-name component is just single-name risk (the parent module already
# emits ``SINGLE_NAME_RISK`` for that case).
MIN_CLUSTER_SIZE = 2


def _components(nodes: list[str],
                edges: list[tuple[str, str]]) -> list[list[str]]:
    """Connected components via iterative union-find (deterministic order).

    Nodes are processed in the input order; each component's tickers are
    sorted alphabetically and components are returned biggest-first with
    ties broken alphabetically by their first ticker."""
    parent: dict[str, str] = {n: n for n in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Deterministic: smaller-string root wins so the tree shape is
            # stable across runs (matters for test reproducibility).
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    for a, b in edges:
        if a in parent and b in parent:
            union(a, b)

    bucket: dict[str, list[str]] = {}
    for n in nodes:
        bucket.setdefault(find(n), []).append(n)

    comps = [sorted(c) for c in bucket.values()]
    comps.sort(key=lambda c: (-len(c), c[0] if c else ""))
    return comps


def _cluster_mean_corr(cluster: list[str],
                       pairs: list[dict]) -> float | None:
    """Mean pairwise ρ across all pairs whose endpoints are both in
    ``cluster``. ``None`` when no scored pair exists inside the cluster
    (shouldn't happen for a multi-name cluster by construction, but kept
    defensive)."""
    in_cluster = set(cluster)
    vals: list[float] = []
    for p in pairs:
        if p.get("corr") is None:
            continue
        if p.get("a") in in_cluster and p.get("b") in in_cluster:
            vals.append(float(p["corr"]))
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


def build_correlation_cluster_warning(
    correlation_payload: dict,
    now: datetime | None = None,
    threshold: float = HIGH_CORR,
) -> dict:
    """Cluster decomposition of an existing /api/correlation payload.

    Inputs:
      correlation_payload — the dict returned by
        ``analytics.correlation.build_correlation``. Required keys:
        ``state``, ``pairs``, ``weights``, ``n_correlatable``.
      now                 — defaults to UTC; injectable for tests.
      threshold           — edge cutoff for high-correlation linkage;
                            defaults to the parent's ``HIGH_CORR`` (0.70).

    Output schema:
      {
        "as_of": iso8601,
        "state": NO_DATA / INSUFFICIENT / OK,
        "threshold_corr": float,
        "n_correlatable": int,
        "n_clusters": int,            # multi-name clusters only
        "clusters": [
          {
            "tickers": [...],
            "size": int,
            "total_weight": float,    # 0..1
            "total_weight_pct": float,
            "internal_mean_corr": float | None,
          }, ...                       # biggest-first
        ],
        "biggest_cluster": cluster | None,
        "verdict": NO_CLUSTERS / WATCHLIST_CLUSTER / DOMINANT_CLUSTER /
                   HIDDEN_FACTOR_BET | None,
        "headline": str,
      }
    """
    if now is None:
        now = datetime.now(timezone.utc)

    upstream_state = (correlation_payload or {}).get("state")
    pairs = (correlation_payload or {}).get("pairs") or []
    weights = (correlation_payload or {}).get("weights") or {}
    n_corr = int((correlation_payload or {}).get("n_correlatable") or 0)

    # Mirror the parent's degraded states one-for-one so a reader sees
    # consistent verdicts when the matrix itself can't be computed.
    if upstream_state == "NO_DATA":
        return {
            "as_of": now.isoformat(timespec="seconds"),
            "state": "NO_DATA",
            "threshold_corr": threshold,
            "n_correlatable": n_corr,
            "n_clusters": 0,
            "clusters": [],
            "biggest_cluster": None,
            "verdict": None,
            "headline": "No stock positions — cluster decomposition undefined.",
        }
    if upstream_state == "INSUFFICIENT" or n_corr < 2:
        return {
            "as_of": now.isoformat(timespec="seconds"),
            "state": "INSUFFICIENT",
            "threshold_corr": threshold,
            "n_correlatable": n_corr,
            "n_clusters": 0,
            "clusters": [],
            "biggest_cluster": None,
            "verdict": None,
            "headline": ("Need ≥2 correlatable names with a scored ρ to "
                         "decompose into clusters."),
        }

    # Build node list from the pairs themselves so we only cluster names
    # whose ρ was actually computable upstream (matches n_correlatable).
    nodes_set: set[str] = set()
    edges: list[tuple[str, str]] = []
    for p in pairs:
        a, b, rho = p.get("a"), p.get("b"), p.get("corr")
        if not a or not b:
            continue
        nodes_set.add(a)
        nodes_set.add(b)
        if rho is None:
            continue
        if float(rho) >= threshold:
            edges.append((a, b))
    nodes = sorted(nodes_set)

    comps = _components(nodes, edges)
    multi = [c for c in comps if len(c) >= MIN_CLUSTER_SIZE]

    clusters: list[dict] = []
    for c in multi:
        # Sum weights of names actually present in the weight map; missing
        # weights (e.g. a position with zero market value) contribute 0.
        total_w = round(sum(float(weights.get(t) or 0.0) for t in c), 6)
        clusters.append({
            "tickers": c,
            "size": len(c),
            "total_weight": total_w,
            "total_weight_pct": round(total_w * 100.0, 2),
            "internal_mean_corr": _cluster_mean_corr(c, pairs),
        })
    # Biggest by book weight wins; size, then alphabetic first-ticker
    # break ties so the order is deterministic.
    clusters.sort(
        key=lambda c: (-c["total_weight"], -c["size"], c["tickers"][0]))

    biggest = clusters[0] if clusters else None

    if biggest is None:
        verdict = "NO_CLUSTERS"
        headline = (
            f"No multi-name cluster at ρ≥{threshold:.2f} — names move "
            f"independently enough that no co-moving block exists.")
    else:
        tw = biggest["total_weight"]
        names = ", ".join(biggest["tickers"])
        rho_clause = ""
        if biggest["internal_mean_corr"] is not None:
            rho_clause = (f" (internal mean ρ="
                          f"{biggest['internal_mean_corr']:+.2f})")
        if tw >= DOMINANT_WEIGHT:
            verdict = "HIDDEN_FACTOR_BET"
            headline = (
                f"HIDDEN_FACTOR_BET — {biggest['size']} co-moving names "
                f"({names}) are {tw * 100:.0f}% of the book{rho_clause}. "
                f"The book is effectively one factor trade.")
        elif tw >= WATCHLIST_WEIGHT:
            verdict = "DOMINANT_CLUSTER"
            headline = (
                f"DOMINANT_CLUSTER — {biggest['size']} co-moving names "
                f"({names}) are {tw * 100:.0f}% of the book{rho_clause}. "
                f"A correlated drawdown takes a third of the book or more.")
        else:
            verdict = "WATCHLIST_CLUSTER"
            headline = (
                f"WATCHLIST_CLUSTER — {biggest['size']} co-moving names "
                f"({names}) are {tw * 100:.0f}% of the book{rho_clause}. "
                f"Cluster exists but is small enough to monitor.")

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "state": "OK",
        "threshold_corr": threshold,
        "n_correlatable": n_corr,
        "n_clusters": len(clusters),
        "clusters": clusters,
        "biggest_cluster": biggest,
        "verdict": verdict,
        "headline": headline,
    }

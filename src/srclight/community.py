"""Community detection, execution flow tracing, and impact analysis.

Uses Louvain algorithm (networkx) on call-graph edges to cluster
symbols into functional communities. BFS from entry points traces
execution flows.

References:
- Blondel et al. 2008, "Fast unfolding of communities in large networks"
- Traag et al. 2019, "From Louvain to Leiden" (future upgrade path)
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Any

from .db import Database

logger = logging.getLogger("srclight.community")


def detect_communities(db: Database) -> list[dict[str, Any]]:
    """Detect communities in the call graph using Louvain algorithm.

    Returns list of community dicts with keys:
        id, label, symbol_count, cohesion, keywords, members
    """
    try:
        import networkx as nx
        from networkx.algorithms.community import louvain_communities
    except ImportError:
        logger.warning("networkx not installed — skipping community detection")
        return []

    assert db.conn is not None

    # Load call edges
    rows = db.conn.execute(
        "SELECT source_id, target_id FROM symbol_edges WHERE edge_type = 'calls'"
    ).fetchall()

    if not rows:
        return []

    # Build undirected graph for community detection
    G = nx.Graph()
    for row in rows:
        src, tgt = row["source_id"], row["target_id"]
        if G.has_edge(src, tgt):
            G[src][tgt]["weight"] = G[src][tgt].get("weight", 1) + 1
        else:
            G.add_edge(src, tgt, weight=1)

    if G.number_of_nodes() < 2:
        return []

    # Run Louvain
    partition = louvain_communities(G, resolution=1.0, seed=42)

    # Load symbol names for labeling
    all_ids = set()
    for community_set in partition:
        all_ids.update(community_set)

    id_to_name = {}
    if all_ids:
        placeholders = ",".join("?" * len(all_ids))
        for row in db.conn.execute(
            f"SELECT id, name, qualified_name, kind FROM symbols WHERE id IN ({placeholders})",
            list(all_ids),
        ):
            id_to_name[row["id"]] = {
                "id": row["id"],
                "name": row["name"],
                "qualified_name": row["qualified_name"],
                "kind": row["kind"],
            }

    # Compute global token frequencies for TF-IDF labeling
    global_freq = Counter()
    for info in id_to_name.values():
        tokens = _tokenize_name(info["name"] or "")
        global_freq.update(set(tokens))  # set() for document frequency

    # Build community results
    communities = []
    for i, community_set in enumerate(partition):
        members = [id_to_name[sid] for sid in community_set if sid in id_to_name]
        if not members:
            continue

        names = [m["name"] for m in members if m["name"]]
        label = _label_community(names, global_freq, len(partition))
        keywords = _extract_keywords(names, global_freq, len(partition))

        communities.append({
            "id": i,
            "label": label,
            "symbol_count": len(members),
            "cohesion": _community_cohesion(community_set, G),
            "keywords": keywords,
            "members": members,
        })

    # Sort by size descending, re-number
    communities.sort(key=lambda c: c["symbol_count"], reverse=True)
    for i, c in enumerate(communities):
        c["id"] = i

    return communities


def _tokenize_name(name: str) -> list[str]:
    """Split a symbol name into lowercase tokens."""
    if not name:
        return []
    # Split on :: . -> _
    parts = re.split(r"::|->|\.|_", name)
    tokens = []
    for part in parts:
        if not part:
            continue
        # Split CamelCase
        s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", part)
        s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
        tokens.extend(t.lower() for t in s.split() if t)
    return tokens


def _label_community(
    names: list[str], global_freq: Counter, n_communities: int,
) -> str:
    """Auto-label a community from its member symbol names using TF-IDF."""
    local_freq = Counter()
    for name in names:
        tokens = _tokenize_name(name)
        local_freq.update(tokens)

    if not local_freq:
        return "unnamed"

    scored = {}
    for token, count in local_freq.items():
        tf = count / len(names)
        df = global_freq.get(token, 1)
        idf = math.log(max(n_communities, 2) / max(df, 1)) + 1
        scored[token] = tf * idf

    # Filter very short tokens
    scored = {t: s for t, s in scored.items() if len(t) > 1}

    top = sorted(scored, key=scored.get, reverse=True)[:3]
    if not top:
        return "unnamed"
    return " ".join(top).title()


def _extract_keywords(
    names: list[str], global_freq: Counter, n_communities: int,
) -> list[str]:
    """Extract top keywords for a community."""
    local_freq = Counter()
    for name in names:
        tokens = _tokenize_name(name)
        local_freq.update(tokens)

    scored = {}
    for token, count in local_freq.items():
        if len(token) <= 1:
            continue
        tf = count / max(len(names), 1)
        df = global_freq.get(token, 1)
        idf = math.log(max(n_communities, 2) / max(df, 1)) + 1
        scored[token] = tf * idf

    return sorted(scored, key=scored.get, reverse=True)[:5]


def _community_cohesion(members: set[int], G) -> float:
    """Compute cohesion as ratio of internal edges to possible edges."""
    if len(members) < 2:
        return 1.0
    internal = sum(
        1 for u in members for v in G.neighbors(u) if v in members
    )
    # Each undirected edge counted twice
    internal //= 2
    possible = len(members) * (len(members) - 1) // 2
    return round(internal / possible, 4) if possible > 0 else 0.0

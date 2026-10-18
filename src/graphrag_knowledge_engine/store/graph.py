"""Knowledge graph over extracted triples and entity mentions.

The graph is a ``networkx.MultiDiGraph`` whose nodes are casefolded entity
names (display label kept as a node attribute) and whose edges carry the
relation plus the chunk ids the triple was extracted from. Nodes additionally
carry *mention provenance*: every chunk in which the entity appears, including
chunks that produced no triple at all. The graph is hydrated from the SQLite
triple and mention tables, so SQLite remains the single source of truth.

Query-time expansion is *bounded*: breadth-first from the seed entities, at
most ``max_depth`` hops, visiting at most ``max_nodes`` entities. Every
visited entity then votes for the chunks that mention it with weight

    w(entity) = 1 / (1 + hop) / sqrt(mention_count(entity))

The ``1 / (1 + hop)`` term keeps evidence about the seeds ahead of evidence
about distant neighbors; the ``sqrt(mention_count)`` discount is IDF-style -
an entity that appears in half the corpus says little about which chunk
matters, while a rare entity is a sharp signal. Chunks are ranked by

1. *closeness* - the strongest single vote they received, then
2. *focus*     - total vote mass divided by how many connected entities the
                 chunk mentions, which keeps "hub" chunks that name-drop
                 everything from crowding out chunks that are actually about
                 the seed.

The ranked chunk list becomes the graph leg fed into Reciprocal Rank Fusion -
this is the multi-hop answer path.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import networkx as nx

from graphrag_knowledge_engine.models import NeighborEdge, Triple
from graphrag_knowledge_engine.store.base import BaseGraphStore


def normalize_entity(name: str) -> str:
    return " ".join(name.casefold().split())


class NetworkXGraphStore(BaseGraphStore):
    def __init__(self) -> None:
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()
        # Reverse index: chunk_id -> entity keys mentioned in it (for focus).
        self._chunk_entities: dict[str, set[str]] = {}

    @property
    def backend_name(self) -> str:
        return "networkx"

    def ping(self) -> bool:
        return True

    def close(self) -> None:
        pass

    def get_diagnostics(self) -> dict[str, Any]:
        return {
            "backend": "networkx",
            "status": "active",
            "ready": True,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
        }

    @classmethod
    def from_store_rows(
        cls, triples: list[Triple], mentions: list[tuple[str, str]]
    ) -> KnowledgeGraph:
        kg = cls()
        kg.add_triples(triples)
        kg.add_mentions(mentions)
        return kg

    @classmethod
    def from_triples(cls, triples: list[Triple]) -> KnowledgeGraph:
        return cls.from_store_rows(triples, [])

    def _touch_node(self, label: str, chunk_id: str | None = None) -> str:
        key = normalize_entity(label)
        if key not in self._graph:
            self._graph.add_node(key, label=label, chunks=set())
        if chunk_id is not None:
            self._graph.nodes[key]["chunks"].add(chunk_id)
            self._chunk_entities.setdefault(chunk_id, set()).add(key)
        return key

    def add_triples(self, triples: list[Triple]) -> None:
        for t in triples:
            s_key, o_key = normalize_entity(t.subject), normalize_entity(t.obj)
            if not s_key or not o_key or s_key == o_key:
                continue
            self._touch_node(t.subject, t.chunk_id)
            self._touch_node(t.obj, t.chunk_id)
            # Merge provenance when the same (s, relation, o) edge already exists.
            existing = self._graph.get_edge_data(s_key, o_key) or {}
            merged = False
            for _, data in existing.items():
                if data.get("relation") == t.relation:
                    data["chunk_ids"].add(t.chunk_id)
                    merged = True
                    break
            if not merged:
                self._graph.add_edge(s_key, o_key, relation=t.relation, chunk_ids={t.chunk_id})

    def add_mentions(self, mentions: list[tuple[str, str]]) -> None:
        """Record (entity display name, chunk_id) pairs as node provenance."""
        for label, chunk_id in mentions:
            if normalize_entity(label):
                self._touch_node(label, chunk_id)

    @property
    def num_nodes(self) -> int:
        return int(self._graph.number_of_nodes())

    @property
    def num_edges(self) -> int:
        return int(self._graph.number_of_edges())

    def entities(self) -> list[str]:
        """Display labels of all entities."""
        return [str(self._graph.nodes[n]["label"]) for n in self._graph.nodes]

    def has_entity(self, name: str) -> bool:
        return normalize_entity(name) in self._graph

    def detect_query_entities(self, query: str) -> list[str]:
        """Entities whose name appears verbatim (casefolded) in the query.

        Longest names win; overlapping shorter matches are dropped so
        "Helios Sync" does not additionally match "Helios".
        """
        q = f" {normalize_entity(query)} "
        candidates = sorted(
            ((n, str(self._graph.nodes[n]["label"])) for n in self._graph.nodes),
            key=lambda item: -len(item[0]),
        )
        taken_spans: list[tuple[int, int]] = []
        found: list[str] = []
        for key, label in candidates:
            idx = _find_word(q, key)
            if idx < 0:
                continue
            span = (idx, idx + len(key))
            if any(s < span[1] and span[0] < e for s, e in taken_spans):
                continue
            taken_spans.append(span)
            found.append(label)
        return found

    def expand(
        self,
        seed_entities: list[str],
        max_depth: int = 2,
        max_nodes: int = 25,
        top_k: int = 8,
    ) -> list[tuple[str, float]]:
        """Bounded BFS from seeds; returns (chunk_id, graph_score), best first.

        The score is closeness + focus (see module docstring); both terms are
        in (0, 1], so closeness dominates and focus breaks ties inside a hop.
        """
        seeds = [normalize_entity(e) for e in seed_entities]
        seeds = [s for s in seeds if s in self._graph]
        if not seeds:
            return []
        depth = self._bfs_depths(seeds, max_depth, max_nodes)
        closeness: dict[str, float] = {}
        vote_mass: dict[str, float] = {}
        for node, hop in depth.items():
            chunks = self._graph.nodes[node]["chunks"]
            if not chunks:
                continue
            # IDF-style discount: entities mentioned everywhere are weak votes.
            weight = 1.0 / (1.0 + hop) / math.sqrt(len(chunks))
            for chunk_id in chunks:
                closeness[chunk_id] = max(closeness.get(chunk_id, 0.0), weight)
                vote_mass[chunk_id] = vote_mass.get(chunk_id, 0.0) + weight
        scored: list[tuple[str, float, float]] = []
        for chunk_id, close in closeness.items():
            # Breadth counts *connected* entities where possible: degree-0
            # nodes are often extraction noise (e.g. heading phrases) and
            # should not dilute a chunk's focus.
            entities = self._chunk_entities.get(chunk_id, set())
            connected = sum(1 for n in entities if self._graph.degree(n) > 0)
            breadth = connected or len(entities) or 1
            focus = vote_mass[chunk_id] / breadth
            scored.append((chunk_id, close, focus))
        scored.sort(key=lambda item: (-item[1], -item[2], item[0]))
        return [(cid, round(close + focus, 6)) for cid, close, focus in scored[:top_k]]

    def _bfs_depths(self, seeds: list[str], max_depth: int, max_nodes: int) -> dict[str, int]:
        """Hop distance from the seed set, bounded by depth and node count."""
        undirected = self._graph.to_undirected(as_view=True)
        depth: dict[str, int] = dict.fromkeys(seeds, 0)
        queue: deque[str] = deque(seeds)
        while queue and len(depth) < max_nodes:
            node = queue.popleft()
            if depth[node] >= max_depth:
                continue
            for neighbor in sorted(undirected.neighbors(node)):
                if neighbor not in depth:
                    depth[neighbor] = depth[node] + 1
                    queue.append(neighbor)
                    if len(depth) >= max_nodes:
                        break
        return depth

    def neighbors(self, entity: str, depth: int = 1) -> list[NeighborEdge]:
        """All edges within `depth` hops of the entity (for the API/CLI)."""
        key = normalize_entity(entity)
        if key not in self._graph:
            return []
        nodes = set(self._bfs_depths([key], max_depth=depth, max_nodes=10_000))
        edges: list[NeighborEdge] = []
        for u, v, data in self._graph.edges(data=True):
            if u in nodes and v in nodes:
                edges.append(
                    NeighborEdge(
                        subject=str(self._graph.nodes[u]["label"]),
                        relation=str(data["relation"]),
                        obj=str(self._graph.nodes[v]["label"]),
                        chunk_ids=sorted(data["chunk_ids"]),
                    )
                )
        return sorted(edges, key=lambda e: (e.subject, e.relation, e.obj))

    def overview(self, limit_entities: int = 30, max_edges: int = 100) -> list[NeighborEdge]:
        """Curated subgraph of high-degree hub entities and their interconnections."""
        if not self._graph:
            return []
        # Sort nodes by degree
        top_nodes = sorted(
            self._graph.nodes, key=lambda n: self._graph.degree(n), reverse=True
        )[:limit_entities]
        top_set = set(top_nodes)
        edges: list[NeighborEdge] = []
        for u, v, data in self._graph.edges(data=True):
            if u in top_set and v in top_set:
                edges.append(
                    NeighborEdge(
                        subject=str(self._graph.nodes[u]["label"]),
                        relation=str(data.get("relation", "related_to")),
                        obj=str(self._graph.nodes[v]["label"]),
                        chunk_ids=sorted(data.get("chunk_ids", [])),
                    )
                )
                if len(edges) >= max_edges:
                    break
        return sorted(edges, key=lambda e: (e.subject, e.relation, e.obj))

    def find_path(self, source: str, target: str, max_depth: int = 4) -> list[NeighborEdge]:
        """Find the shortest relationship path between two entities."""
        s_key, t_key = normalize_entity(source), normalize_entity(target)
        if s_key not in self._graph or t_key not in self._graph:
            return []
        try:
            path_nodes = nx.shortest_path(self._graph.to_undirected(as_view=True), s_key, t_key)
            if len(path_nodes) - 1 > max_depth:
                return []
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

        edges: list[NeighborEdge] = []
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            data_uv = self._graph.get_edge_data(u, v)
            if data_uv:
                for d in data_uv.values():
                    edges.append(
                        NeighborEdge(
                            subject=str(self._graph.nodes[u]["label"]),
                            relation=str(d.get("relation", "related_to")),
                            obj=str(self._graph.nodes[v]["label"]),
                            chunk_ids=sorted(d.get("chunk_ids", [])),
                        )
                    )
                    break
            else:
                data_vu = self._graph.get_edge_data(v, u) or {}
                for d in data_vu.values():
                    edges.append(
                        NeighborEdge(
                            subject=str(self._graph.nodes[v]["label"]),
                            relation=str(d.get("relation", "related_to")),
                            obj=str(self._graph.nodes[u]["label"]),
                            chunk_ids=sorted(d.get("chunk_ids", [])),
                        )
                    )
                    break
        return edges

    def remove_chunks(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        c_set = set(chunk_ids)

        # 1. Clean edge provenance and drop edges that lose all chunks
        edges_to_remove = []
        for u, v, k, data in self._graph.edges(keys=True, data=True):
            if "chunk_ids" in data:
                data["chunk_ids"].difference_update(c_set)
                if not data["chunk_ids"]:
                    edges_to_remove.append((u, v, k))
        for u, v, k in edges_to_remove:
            self._graph.remove_edge(u, v, key=k)

        # 2. Clean node provenance and drop orphaned isolated nodes
        nodes_to_remove = []
        for n, data in self._graph.nodes(data=True):
            if "chunks" in data:
                data["chunks"].difference_update(c_set)
            if not data.get("chunks") and self._graph.degree(n) == 0:
                nodes_to_remove.append(n)
        for n in nodes_to_remove:
            self._graph.remove_node(n)

        # 3. Clean reverse index
        for cid in chunk_ids:
            self._chunk_entities.pop(cid, None)



def _find_word(haystack: str, needle: str) -> int:
    """Index of `needle` in `haystack` at word boundaries, or -1."""
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx < 0:
            return -1
        before_ok = idx == 0 or not haystack[idx - 1].isalnum()
        end = idx + len(needle)
        after_ok = end >= len(haystack) or not haystack[end].isalnum()
        if before_ok and after_ok:
            return idx
        start = idx + 1


# Alias for backwards compatibility across existing callers and test fixtures
KnowledgeGraph = NetworkXGraphStore

__all__ = ["BaseGraphStore", "KnowledgeGraph", "NetworkXGraphStore", "normalize_entity"]


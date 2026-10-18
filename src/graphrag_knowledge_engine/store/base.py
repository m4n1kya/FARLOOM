"""Base interface for knowledge graph storage backends.

Defines the contract for persistent and in-memory graph stores (NetworkX, Neo4j, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from graphrag_knowledge_engine.models import NeighborEdge, Triple


class BaseGraphStore(ABC):
    """Abstract interface for knowledge graph stores in FARLOOM."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Name of the graph backend (e.g. 'networkx', 'neo4j')."""
        ...

    @property
    @abstractmethod
    def num_nodes(self) -> int:
        """Total number of entity nodes in the current space."""
        ...

    @property
    @abstractmethod
    def num_edges(self) -> int:
        """Total number of relationship edges in the current space."""
        ...

    @abstractmethod
    def add_triples(self, triples: list[Triple]) -> None:
        """Ingest a batch of semantic triples with chunk provenance."""
        ...

    @abstractmethod
    def add_mentions(self, mentions: list[tuple[str, str]]) -> None:
        """Record (entity display label, chunk_id) pairs as node provenance."""
        ...

    @abstractmethod
    def entities(self) -> list[str]:
        """Return display labels of all entities in the graph."""
        ...

    @abstractmethod
    def has_entity(self, name: str) -> bool:
        """Check whether an entity exists in the graph (case-insensitive)."""
        ...

    @abstractmethod
    def detect_query_entities(self, query: str) -> list[str]:
        """Detect entity mentions appearing verbatim in the query text."""
        ...

    @abstractmethod
    def expand(
        self,
        seed_entities: list[str],
        max_depth: int = 2,
        max_nodes: int = 25,
        top_k: int = 8,
    ) -> list[tuple[str, float]]:
        """Bounded multi-hop expansion from seed entities, returning (chunk_id, score)."""
        ...

    @abstractmethod
    def neighbors(self, entity: str, depth: int = 1) -> list[NeighborEdge]:
        """Retrieve relationship edges within `depth` hops of the entity."""
        ...

    @abstractmethod
    def ping(self) -> bool:
        """Check whether the underlying database or store is reachable and healthy."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release underlying connections or resources."""
        ...

    @abstractmethod
    def get_diagnostics(self) -> dict[str, Any]:
        """Return diagnostic details about the backend connection and state."""
        ...

    @abstractmethod
    def overview(self, limit_entities: int = 30, max_edges: int = 100) -> list[NeighborEdge]:
        """Return a curated subgraph of high-degree hub entities and their connections."""
        ...

    @abstractmethod
    def find_path(self, source: str, target: str, max_depth: int = 4) -> list[NeighborEdge]:
        """Find the shortest relationship path between two entities."""
        ...

    @abstractmethod
    def remove_chunks(self, chunk_ids: list[str]) -> None:
        """Prune edges, mentions, and orphaned entity nodes associated with the deleted chunks."""
        ...


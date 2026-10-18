"""Neo4j knowledge graph store for FARLOOM.

Connects to Neo4j AuraDB or local Neo4j instances using the official Python driver.
Provides persistent, multi-tenant graph storage, bounded multi-hop expansion,
ego-network neighborhood queries, and idempotent Cypher operations.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any
from urllib.parse import urlparse

import neo4j
from neo4j.exceptions import AuthError, ServiceUnavailable, SessionExpired

from graphrag_knowledge_engine.models import NeighborEdge, Triple
from graphrag_knowledge_engine.store.base import BaseGraphStore
from graphrag_knowledge_engine.store.graph import _find_word, normalize_entity

logger = logging.getLogger(__name__)

# Shared driver registry to prevent connection leakage across spaces
_DRIVER_CACHE: dict[tuple[str, str], neo4j.Driver] = {}


def get_shared_driver(
    uri: str,
    username: str,
    password: str,
    connection_timeout: float = 10.0,
) -> neo4j.Driver:
    """Retrieve or create a cached neo4j.Driver instance."""
    cache_key = (uri, username)
    driver = _DRIVER_CACHE.get(cache_key)
    if driver is None:
        driver = neo4j.GraphDatabase.driver(
            uri,
            auth=(username, password),
            connection_timeout=connection_timeout,
            max_connection_lifetime=3600,
        )
        _DRIVER_CACHE[cache_key] = driver
    return driver


def close_all_drivers() -> None:
    """Close and clear all cached Neo4j drivers on shutdown."""
    for driver in _DRIVER_CACHE.values():
        try:
            driver.close()
        except Exception as exc:
            logger.debug("Error closing driver: %s", exc)
    _DRIVER_CACHE.clear()


def redact_uri(uri: str) -> str:
    """Redact credentials from connection URI for safe logging and status outputs."""
    try:
        parsed = urlparse(uri)
        if parsed.password:
            netloc = f"{parsed.username}:****@{parsed.hostname}"
            if parsed.port:
                netloc += f":{parsed.port}"
            return parsed._replace(netloc=netloc).geturl()
        return uri
    except Exception:
        # Fallback regex redaction
        return re.sub(r"://([^:@]+):([^@]+)@", r"://\1:****@", uri)


class Neo4jGraphStore(BaseGraphStore):
    """Neo4j knowledge graph store supporting multi-tenant spaces and AuraDB."""

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        space_id: str = "default",
        connection_timeout: float = 10.0,
        driver: neo4j.Driver | None = None,
    ) -> None:
        self._uri = uri
        self._username = username
        self._password = password
        self._database = database or "neo4j"
        self._space_id = space_id or "default"
        self._connection_timeout = connection_timeout
        self._owns_driver = driver is None

        if driver is not None:
            self._driver = driver
        else:
            self._driver = get_shared_driver(
                uri=uri,
                username=username,
                password=password,
                connection_timeout=connection_timeout,
            )

        # In-memory entity cache for fast local verbatim entity matching
        self._entity_cache: list[tuple[str, str]] | None = None

    @property
    def backend_name(self) -> str:
        return "neo4j"

    @property
    def space_id(self) -> str:
        return self._space_id

    @property
    def database(self) -> str:
        return self._database

    def _get_session(self) -> neo4j.Session:
        return self._driver.session(database=self._database)

    def verify_connectivity(self) -> None:
        """Verify database connection with AuraDB pause diagnosis."""
        try:
            self._driver.verify_connectivity()
        except AuthError as exc:
            msg = (
                f"Neo4j authentication failed for user '{self._username}' on {redact_uri(self._uri)}. "
                "Please verify NEO4J_USERNAME and NEO4J_PASSWORD."
            )
            logger.error(msg)
            raise ConnectionError(msg) from exc
        except (ServiceUnavailable, SessionExpired) as exc:
            msg = (
                f"Neo4j database at {redact_uri(self._uri)} is unreachable. "
                "If using a free Neo4j AuraDB instance, it may have been paused due to inactivity. "
                "Please check the Neo4j Console (https://console.neo4j.io) and resume your instance."
            )
            logger.error(msg)
            raise ConnectionError(msg) from exc
        except Exception as exc:
            msg = f"Failed to connect to Neo4j at {redact_uri(self._uri)}: {exc}"
            logger.error(msg)
            raise ConnectionError(msg) from exc

    def init_schema(self) -> None:
        """Initialize uniqueness constraints and indexes idempotently."""
        statements = [
            # Constraint ensures uniqueness of entity per space
            "CREATE CONSTRAINT entity_uid IF NOT EXISTS FOR (e:Entity) REQUIRE e.uid IS UNIQUE",
            # Index on space_id for multi-tenant query performance
            "CREATE INDEX entity_space_idx IF NOT EXISTS FOR (e:Entity) ON (e.space_id)",
            # Composite index on space_id and key
            "CREATE INDEX entity_space_key IF NOT EXISTS FOR (e:Entity) ON (e.space_id, e.key)",
        ]
        with self._get_session() as session:
            for stmt in statements:
                try:
                    session.run(stmt).consume()
                except Exception as exc:
                    logger.debug("Schema initialization statement '%s' notice: %s", stmt, exc)

    def ping(self) -> bool:
        """Check if Neo4j is responsive."""
        try:
            with self._get_session() as session:
                result = session.run("RETURN 1 AS ping")
                record = result.single()
                return record is not None and record["ping"] == 1
        except Exception as exc:
            logger.warning("Neo4j ping failed: %s", exc)
            return False

    def close(self) -> None:
        """Release session / driver resources."""
        if self._owns_driver:
            # We don't necessarily close shared driver immediately to allow multi-space reuse,
            # but we invalidate local caches.
            pass
        self._entity_cache = None

    def get_diagnostics(self) -> dict[str, Any]:
        """Diagnostic state for status and health check endpoints."""
        is_healthy = self.ping()
        diag: dict[str, Any] = {
            "backend": "neo4j",
            "uri": redact_uri(self._uri),
            "database": self._database,
            "space_id": self._space_id,
            "status": "connected" if is_healthy else "disconnected",
            "ready": is_healthy,
        }
        if is_healthy:
            try:
                diag["num_nodes"] = self.num_nodes
                diag["num_edges"] = self.num_edges
            except Exception as exc:
                diag["stats_error"] = str(exc)
        else:
            diag["hint"] = (
                "Instance may be paused or offline. Check https://console.neo4j.io if using AuraDB Free."
            )
        return diag

    # -- Graph mutation & provenance -----------------------------------------

    def add_triples(self, triples: list[Triple]) -> None:
        """Ingest semantic triples into Neo4j with provenance tracking."""
        if not triples:
            return

        batch_items: dict[tuple[str, str, str], dict[str, Any]] = {}
        for t in triples:
            s_key = normalize_entity(t.subject)
            o_key = normalize_entity(t.obj)
            if not s_key or not o_key or s_key == o_key:
                continue
            rel = t.relation.strip()
            item_key = (s_key, rel, o_key)
            if item_key not in batch_items:
                batch_items[item_key] = {
                    "s_uid": f"{self._space_id}:{s_key}",
                    "s_key": s_key,
                    "s_label": t.subject.strip(),
                    "o_uid": f"{self._space_id}:{o_key}",
                    "o_key": o_key,
                    "o_label": t.obj.strip(),
                    "relation": rel,
                    "chunk_ids": set(),
                }
            if t.chunk_id:
                batch_items[item_key]["chunk_ids"].add(t.chunk_id)

        if not batch_items:
            return

        formatted_batch = [
            {
                "s_uid": item["s_uid"],
                "s_key": item["s_key"],
                "s_label": item["s_label"],
                "o_uid": item["o_uid"],
                "o_key": item["o_key"],
                "o_label": item["o_label"],
                "relation": item["relation"],
                "chunk_ids": sorted(item["chunk_ids"]),
            }
            for item in batch_items.values()
        ]

        query = """
        UNWIND $batch AS item
        MERGE (s:Entity {uid: item.s_uid})
        ON CREATE SET
            s.key = item.s_key,
            s.label = item.s_label,
            s.space_id = $space_id,
            s.chunks = item.chunk_ids
        ON MATCH SET
            s.chunks = coalesce(s.chunks, []) + [c IN item.chunk_ids WHERE NOT c IN coalesce(s.chunks, [])]

        MERGE (o:Entity {uid: item.o_uid})
        ON CREATE SET
            o.key = item.o_key,
            o.label = item.o_label,
            o.space_id = $space_id,
            o.chunks = item.chunk_ids
        ON MATCH SET
            o.chunks = coalesce(o.chunks, []) + [c IN item.chunk_ids WHERE NOT c IN coalesce(o.chunks, [])]

        MERGE (s)-[r:RELATED {relation: item.relation, space_id: $space_id}]->(o)
        ON CREATE SET
            r.chunk_ids = item.chunk_ids
        ON MATCH SET
            r.chunk_ids = coalesce(r.chunk_ids, []) + [c IN item.chunk_ids WHERE NOT c IN coalesce(r.chunk_ids, [])]
        """

        with self._get_session() as session:
            session.run(query, batch=formatted_batch, space_id=self._space_id).consume()

        self._entity_cache = None  # invalidate entity cache

    def add_mentions(self, mentions: list[tuple[str, str]]) -> None:
        """Record (entity display label, chunk_id) pairs as node provenance."""
        if not mentions:
            return

        batch_items: dict[str, dict[str, Any]] = {}
        for label, chunk_id in mentions:
            key = normalize_entity(label)
            if not key:
                continue
            if key not in batch_items:
                batch_items[key] = {
                    "uid": f"{self._space_id}:{key}",
                    "key": key,
                    "label": label.strip(),
                    "chunk_ids": set(),
                }
            if chunk_id:
                batch_items[key]["chunk_ids"].add(chunk_id)

        if not batch_items:
            return

        formatted_batch = [
            {
                "uid": item["uid"],
                "key": item["key"],
                "label": item["label"],
                "chunk_ids": sorted(item["chunk_ids"]),
            }
            for item in batch_items.values()
        ]

        query = """
        UNWIND $batch AS item
        MERGE (e:Entity {uid: item.uid})
        ON CREATE SET
            e.key = item.key,
            e.label = item.label,
            e.space_id = $space_id,
            e.chunks = item.chunk_ids
        ON MATCH SET
            e.chunks = coalesce(e.chunks, []) + [c IN item.chunk_ids WHERE NOT c IN coalesce(e.chunks, [])]
        """

        with self._get_session() as session:
            session.run(query, batch=formatted_batch, space_id=self._space_id).consume()

        self._entity_cache = None

    # -- Graph properties & queries ------------------------------------------

    @property
    def num_nodes(self) -> int:
        """Total entity nodes in the space."""
        query = "MATCH (e:Entity {space_id: $space_id}) RETURN count(e) AS cnt"
        with self._get_session() as session:
            record = session.run(query, space_id=self._space_id).single()
            return int(record["cnt"]) if record else 0

    @property
    def num_edges(self) -> int:
        """Total relationship edges in the space."""
        query = "MATCH ()-[r:RELATED {space_id: $space_id}]->() RETURN count(r) AS cnt"
        with self._get_session() as session:
            record = session.run(query, space_id=self._space_id).single()
            return int(record["cnt"]) if record else 0

    def entities(self) -> list[str]:
        """Display labels of all entities in the space, sorted alphabetically."""
        query = """
        MATCH (e:Entity {space_id: $space_id})
        RETURN e.label AS label
        ORDER BY label ASC
        """
        with self._get_session() as session:
            result = session.run(query, space_id=self._space_id)
            return [str(record["label"]) for record in result]

    def has_entity(self, name: str) -> bool:
        """Check if an entity exists in the space (case-insensitive)."""
        key = normalize_entity(name)
        if not key:
            return False
        uid = f"{self._space_id}:{key}"
        query = "MATCH (e:Entity {uid: $uid}) RETURN count(e) > 0 AS exists"
        with self._get_session() as session:
            record = session.run(query, uid=uid).single()
            return bool(record["exists"]) if record else False

    def _ensure_entity_cache(self) -> list[tuple[str, str]]:
        if self._entity_cache is None:
            query = """
            MATCH (e:Entity {space_id: $space_id})
            RETURN e.key AS key, e.label AS label
            """
            with self._get_session() as session:
                result = session.run(query, space_id=self._space_id)
                self._entity_cache = [
                    (str(rec["key"]), str(rec["label"])) for rec in result
                ]
        return self._entity_cache

    def detect_query_entities(self, query: str) -> list[str]:
        """Detect entities mentioned verbatim in the query, preferring longest matches."""
        q = f" {normalize_entity(query)} "
        candidates = sorted(
            self._ensure_entity_cache(),
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
        """Bounded multi-hop expansion from seed entities in Neo4j.

        Calculates closeness + focus scores identically to FARLOOM's ranking formula:
        1. Multi-hop traversal bounded by max_depth and max_nodes.
        2. IDF-weighted chunk votes: w(entity) = 1 / (1 + hop) / sqrt(mention_count).
        3. Focus discount penalizing generic hub chunks.
        """
        seed_keys = [normalize_entity(e) for e in seed_entities if normalize_entity(e)]
        if not seed_keys:
            return []

        safe_depth = min(max(int(max_depth), 1), 5)
        safe_max_nodes = min(max(int(max_nodes), 1), 500)

        # 1. Traverse neighborhood from seed entities up to max_depth hops
        traversal_query = f"""
        MATCH (s:Entity)
        WHERE s.space_id = $space_id AND s.key IN $seed_keys
        MATCH path = (s)-[:RELATED*0..{safe_depth}]-(n:Entity)
        WHERE n.space_id = $space_id
        WITH n, min(length(path)) AS hop
        ORDER BY hop ASC, n.key ASC
        LIMIT $max_nodes
        RETURN n.key AS key, n.label AS label, coalesce(n.chunks, []) AS chunks, hop
        """

        with self._get_session() as session:
            records = list(
                session.run(
                    traversal_query,
                    space_id=self._space_id,
                    seed_keys=seed_keys,
                    max_nodes=safe_max_nodes,
                )
            )

        if not records:
            return []

        # 2. Accumulate closeness and vote_mass per chunk
        closeness: dict[str, float] = {}
        vote_mass: dict[str, float] = {}

        for rec in records:
            hop = int(rec["hop"])
            chunks = list(rec["chunks"])
            if not chunks:
                continue
            weight = 1.0 / (1.0 + hop) / math.sqrt(len(chunks))
            for chunk_id in chunks:
                closeness[chunk_id] = max(closeness.get(chunk_id, 0.0), weight)
                vote_mass[chunk_id] = vote_mass.get(chunk_id, 0.0) + weight

        if not closeness:
            return []

        # 3. Query all entities mentioning the candidate chunks to compute focus breadth
        candidate_chunk_ids = list(closeness.keys())
        focus_query = """
        MATCH (e:Entity)
        WHERE e.space_id = $space_id AND any(c IN e.chunks WHERE c IN $candidate_chunks)
        OPTIONAL MATCH (e)-[r:RELATED {space_id: $space_id}]-()
        RETURN e.key AS key, coalesce(e.chunks, []) AS chunks, count(r) AS degree
        """

        with self._get_session() as session:
            chunk_recs = list(
                session.run(
                    focus_query,
                    space_id=self._space_id,
                    candidate_chunks=candidate_chunk_ids,
                )
            )

        # 4. Compute focus score: vote_mass / breadth
        scored: list[tuple[str, float, float]] = []
        for chunk_id, close in closeness.items():
            connected = sum(
                1
                for r in chunk_recs
                if chunk_id in r["chunks"] and int(r["degree"]) > 0
            )
            total_entities = sum(1 for r in chunk_recs if chunk_id in r["chunks"])
            breadth = connected or total_entities or 1
            focus = vote_mass[chunk_id] / breadth
            scored.append((chunk_id, close, focus))

        # Sort: strongest closeness first, highest focus tiebreak, deterministic chunk_id
        scored.sort(key=lambda item: (-item[1], -item[2], item[0]))
        return [(cid, round(close + focus, 6)) for cid, close, focus in scored[:top_k]]

    def neighbors(self, entity: str, depth: int = 1) -> list[NeighborEdge]:
        """All relationship edges within `depth` hops of the entity (for explorer UI)."""
        key = normalize_entity(entity)
        if not key:
            return []

        seed_uid = f"{self._space_id}:{key}"
        safe_depth = min(max(int(depth), 1), 4)

        query = f"""
        MATCH (s:Entity {{uid: $seed_uid}})
        MATCH (s)-[:RELATED*0..{safe_depth}]-(n:Entity)
        WHERE n.space_id = $space_id
        WITH collect(DISTINCT n) AS nodes
        UNWIND nodes AS u
        MATCH (u)-[r:RELATED {{space_id: $space_id}}]->(v:Entity)
        WHERE v IN nodes
        RETURN DISTINCT u.label AS subject, r.relation AS relation, v.label AS obj, coalesce(r.chunk_ids, []) AS chunk_ids
        ORDER BY subject ASC, relation ASC, obj ASC
        """

        with self._get_session() as session:
            records = session.run(
                query,
                seed_uid=seed_uid,
                space_id=self._space_id,
            )
            edges: list[NeighborEdge] = []
            for rec in records:
                edges.append(
                    NeighborEdge(
                        subject=str(rec["subject"]),
                        relation=str(rec["relation"]),
                        obj=str(rec["obj"]),
                        chunk_ids=sorted(rec["chunk_ids"]),
                    )
                )
            return sorted(edges, key=lambda e: (e.subject, e.relation, e.obj))

    def overview(self, limit_entities: int = 30, max_edges: int = 100) -> list[NeighborEdge]:
        """Curated overview subgraph of top hub entities and their interconnections."""
        query = """
        MATCH (e:Entity {space_id: $space_id})
        OPTIONAL MATCH (e)-[r:RELATED {space_id: $space_id}]-()
        WITH e, count(r) AS degree
        ORDER BY degree DESC
        LIMIT $limit_entities
        WITH collect(e) AS hubs
        UNWIND hubs AS u
        MATCH (u)-[r:RELATED {space_id: $space_id}]->(v:Entity)
        WHERE v IN hubs
        RETURN DISTINCT u.label AS subject, r.relation AS relation, v.label AS obj, coalesce(r.chunk_ids, []) AS chunk_ids
        ORDER BY subject ASC, relation ASC, obj ASC
        LIMIT $max_edges
        """
        with self._get_session() as session:
            records = session.run(
                query,
                space_id=self._space_id,
                limit_entities=min(max(limit_entities, 5), 100),
                max_edges=min(max(max_edges, 10), 300),
            )
            edges: list[NeighborEdge] = []
            for rec in records:
                edges.append(
                    NeighborEdge(
                        subject=str(rec["subject"]),
                        relation=str(rec["relation"]),
                        obj=str(rec["obj"]),
                        chunk_ids=sorted(rec["chunk_ids"]),
                    )
                )
            return sorted(edges, key=lambda e: (e.subject, e.relation, e.obj))

    def find_path(self, source: str, target: str, max_depth: int = 4) -> list[NeighborEdge]:
        """Find the shortest relationship path between two entities in Neo4j."""
        s_key, t_key = normalize_entity(source), normalize_entity(target)
        if not s_key or not t_key:
            return []
        s_uid, t_uid = f"{self._space_id}:{s_key}", f"{self._space_id}:{t_key}"
        safe_depth = min(max(int(max_depth), 1), 6)

        query = f"""
        MATCH (s:Entity {{uid: $s_uid}}), (t:Entity {{uid: $t_uid}})
        MATCH p = shortestPath((s)-[:RELATED*1..{safe_depth}]-(t))
        WHERE all(r in relationships(p) WHERE r.space_id = $space_id)
        UNWIND relationships(p) AS r
        RETURN startNode(r).label AS subject, r.relation AS relation, endNode(r).label AS obj, coalesce(r.chunk_ids, []) AS chunk_ids
        """
        with self._get_session() as session:
            records = session.run(query, s_uid=s_uid, t_uid=t_uid, space_id=self._space_id)
            edges: list[NeighborEdge] = []
            for rec in records:
                edges.append(
                    NeighborEdge(
                        subject=str(rec["subject"]),
                        relation=str(rec["relation"]),
                        obj=str(rec["obj"]),
                        chunk_ids=sorted(rec["chunk_ids"]),
                    )
                )
            return edges

    def remove_chunks(self, chunk_ids: list[str]) -> None:
        """Prune edges, mentions, and orphaned entity nodes associated with deleted chunks."""
        if not chunk_ids:
            return
        query_edges = """
        UNWIND $chunk_ids AS cid
        MATCH ()-[r:RELATED {space_id: $space_id}]->()
        WHERE cid IN r.chunk_ids
        SET r.chunk_ids = [c IN r.chunk_ids WHERE c <> cid]
        WITH r
        WHERE size(r.chunk_ids) = 0
        DELETE r
        """
        query_nodes = """
        UNWIND $chunk_ids AS cid
        MATCH (n:Entity {space_id: $space_id})
        WHERE cid IN n.chunks
        SET n.chunks = [c IN n.chunks WHERE c <> cid]
        WITH n
        WHERE size(n.chunks) = 0 AND NOT (n)--()
        DELETE n
        """
        try:
            with self._get_session() as session:
                session.run(query_edges, chunk_ids=chunk_ids, space_id=self._space_id).consume()
                session.run(query_nodes, chunk_ids=chunk_ids, space_id=self._space_id).consume()
            self._entity_cache = None
        except Exception as exc:
            logger.warning("Error pruning chunks from Neo4j: %s", exc)


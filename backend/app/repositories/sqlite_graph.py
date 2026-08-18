"""Adapter: `GraphRepository` backed by SQLite.

**Why SQLite.** The project had no relational store - ChromaDB holds vectors and
answers similarity queries, which a graph cannot be built on: neighbourhood
traversal, type filters and edge aggregation all need joins and indexes.
SQLite adds no service, no daemon and no dependency (it is in the standard
library) and lives as one file beside `chroma_db/`. It is the smallest thing
that makes the graph a real, queryable, persistent store.

**Identity is deterministic, not random.** An entity's id is a hash of
`(type, canonical_name)` and a relationship's is a hash of
`(source, type, target, polarity)`. Re-ingesting the same document therefore
updates rows rather than duplicating them, and the same claim appearing in
three documents is one edge with three pieces of evidence.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

from app.core.config import settings
from app.core.exceptions import PlatformError
from app.core.logging import get_logger
from app.domain.knowledge import (
    DocumentKnowledge,
    EntityDetail,
    EntityType,
    GraphEntity,
    GraphRelationship,
    GraphStats,
    GraphView,
    Polarity,
    RelationshipEvidence,
    RelationType,
)
from app.knowledge.normalization import display_name

logger = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    id              TEXT PRIMARY KEY,
    canonical_name  TEXT NOT NULL,
    display_name    TEXT NOT NULL,
    type            TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    mentions_count  INTEGER NOT NULL DEFAULT 0,
    confidence      REAL NOT NULL DEFAULT 1.0,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entities_canonical ON entities(canonical_name);
CREATE INDEX IF NOT EXISTS idx_entities_type      ON entities(type);

CREATE TABLE IF NOT EXISTS entity_sources (
    entity_id   TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    document_id TEXT NOT NULL,
    chunk_id    TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (entity_id, document_id, chunk_id)
);
CREATE INDEX IF NOT EXISTS idx_entity_sources_doc ON entity_sources(document_id);

CREATE TABLE IF NOT EXISTS relationships (
    id           TEXT PRIMARY KEY,
    source_id    TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    target_id    TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    type         TEXT NOT NULL,
    predicate    TEXT NOT NULL DEFAULT '',
    confidence   REAL NOT NULL DEFAULT 0.0,
    polarity     TEXT NOT NULL DEFAULT 'positive',
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_id);
CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_id);
CREATE INDEX IF NOT EXISTS idx_rel_type   ON relationships(type);

CREATE TABLE IF NOT EXISTS relationship_evidence (
    relationship_id TEXT NOT NULL REFERENCES relationships(id) ON DELETE CASCADE,
    document_id     TEXT NOT NULL,
    chunk_id        TEXT NOT NULL DEFAULT '',
    sentence        TEXT NOT NULL DEFAULT '',
    confidence      REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY (relationship_id, document_id, sentence)
);
CREATE INDEX IF NOT EXISTS idx_evidence_doc ON relationship_evidence(document_id);
"""


def entity_id(entity_type: str, canonical: str) -> str:
    """Stable id for an entity, so re-ingestion updates rather than duplicates."""
    digest = hashlib.sha1(f"{entity_type}|{canonical}".encode()).hexdigest()
    return f"e_{digest[:16]}"


def relationship_id(source: str, relation_type: str, target: str, polarity: str) -> str:
    """Stable id for an edge."""
    digest = hashlib.sha1(f"{source}|{relation_type}|{target}|{polarity}".encode()).hexdigest()
    return f"r_{digest[:16]}"


class SqliteGraphRepository:
    """Persistent knowledge graph in a single SQLite file."""

    def __init__(self, path: str | None = None):
        self.path = str(path or settings.graph_db_path)
        self._lock = RLock()
        self._connection: sqlite3.Connection | None = None

    # ── Connection ───────────────────────────────────────────────

    @property
    def connection(self) -> sqlite3.Connection:
        """Connect lazily; importing this module must not touch the disk."""
        if self._connection is None:
            if self.path != ":memory:":
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            # uvicorn serves from a thread pool and ingestion runs off the event
            # loop, so the connection is shared and guarded by `self._lock`.
            self._connection = sqlite3.connect(
                self.path, check_same_thread=False, timeout=30.0
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.executescript(SCHEMA)
            self._connection.commit()
            logger.info("Opened knowledge graph database at %s", self.path)
        return self._connection

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    # ── Writing ──────────────────────────────────────────────────

    def save_document(
        self,
        document_id: str,
        knowledge: DocumentKnowledge,
        chunk_for_offset: Callable[[int], str] | None = None,
    ) -> tuple[int, int]:
        """Persist one document's entities and relationships."""
        if knowledge.is_empty:
            return (0, 0)

        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            connection = self.connection
            try:
                with connection:  # one transaction; rolls back on error
                    # Replacing a document's contribution rather than adding to
                    # it keeps re-ingestion idempotent.
                    self._clear_document(connection, document_id)

                    saved_entities = {
                        entity.canonical_name: self._upsert_entity(
                            connection, entity, document_id, now
                        )
                        for entity in knowledge.entities
                    }

                    edges = 0
                    for relation in knowledge.relations:
                        source = saved_entities.get(relation.source.canonical_name)
                        target = saved_entities.get(relation.target.canonical_name)
                        if source is None:
                            source = self._upsert_entity(
                                connection, relation.source, document_id, now
                            )
                            saved_entities[relation.source.canonical_name] = source
                        if target is None:
                            target = self._upsert_entity(
                                connection, relation.target, document_id, now
                            )
                            saved_entities[relation.target.canonical_name] = target

                        chunk_id = (
                            chunk_for_offset(relation.sentence_start)
                            if chunk_for_offset
                            else ""
                        )
                        self._upsert_relationship(
                            connection, relation, source, target, document_id, chunk_id, now
                        )
                        edges += 1

            except sqlite3.Error as exc:
                raise PlatformError(f"Writing the knowledge graph failed: {exc}") from exc

        logger.info(
            "Persisted %d entities and %d relationships for document %s",
            len(saved_entities),
            edges,
            document_id,
        )
        return (len(saved_entities), edges)

    def _upsert_entity(self, connection, entity, document_id: str, now: str) -> str:  # noqa: ANN001
        node_id = entity_id(entity.type.value, entity.canonical_name)

        connection.execute(
            """
            INSERT INTO entities
                (id, canonical_name, display_name, type, description,
                 mentions_count, confidence, created_at, updated_at)
            VALUES (?, ?, ?, ?, '', 1, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                mentions_count = mentions_count + 1,
                confidence     = MAX(confidence, excluded.confidence),
                updated_at     = excluded.updated_at
            """,
            (
                node_id,
                entity.canonical_name,
                display_name(entity.text),
                entity.type.value,
                entity.confidence,
                now,
                now,
            ),
        )
        connection.execute(
            "INSERT OR IGNORE INTO entity_sources (entity_id, document_id, chunk_id) "
            "VALUES (?, ?, '')",
            (node_id, document_id),
        )
        return node_id

    def _upsert_relationship(  # noqa: PLR0913 - one row, one call
        self, connection, relation, source_id: str, target_id: str,  # noqa: ANN001
        document_id: str, chunk_id: str, now: str,
    ) -> None:
        edge_id = relationship_id(
            source_id, relation.type.value, target_id, relation.polarity.value
        )

        connection.execute(
            """
            INSERT INTO relationships
                (id, source_id, target_id, type, predicate, confidence,
                 polarity, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                -- The strongest observation wins; evidence accumulates below.
                confidence = MAX(confidence, excluded.confidence),
                predicate  = excluded.predicate,
                updated_at = excluded.updated_at
            """,
            (
                edge_id,
                source_id,
                target_id,
                relation.type.value,
                relation.predicate,
                relation.confidence,
                relation.polarity.value,
                now,
                now,
            ),
        )
        connection.execute(
            """
            INSERT OR REPLACE INTO relationship_evidence
                (relationship_id, document_id, chunk_id, sentence, confidence)
            VALUES (?, ?, ?, ?, ?)
            """,
            (edge_id, document_id, chunk_id, relation.sentence, relation.confidence),
        )

    def delete_document(self, document_id: str) -> None:
        """Remove a document's contribution and prune what it left behind."""
        with self._lock:
            connection = self.connection
            with connection:
                self._clear_document(connection, document_id)

    @staticmethod
    def _clear_document(connection, document_id: str) -> None:  # noqa: ANN001
        connection.execute(
            "DELETE FROM relationship_evidence WHERE document_id = ?", (document_id,)
        )
        connection.execute("DELETE FROM entity_sources WHERE document_id = ?", (document_id,))
        # An edge with no remaining evidence is no longer supported by anything.
        connection.execute(
            "DELETE FROM relationships WHERE id NOT IN "
            "(SELECT DISTINCT relationship_id FROM relationship_evidence)"
        )
        # An entity with no source document and no edges is orphaned.
        connection.execute(
            """
            DELETE FROM entities WHERE id NOT IN (SELECT DISTINCT entity_id FROM entity_sources)
              AND id NOT IN (SELECT source_id FROM relationships)
              AND id NOT IN (SELECT target_id FROM relationships)
            """
        )

    # ── Reading ──────────────────────────────────────────────────

    def graph(
        self,
        document_id: str | None = None,
        entity_type: str | None = None,
        relationship_type: str | None = None,
        search: str | None = None,
        min_confidence: float = 0.0,
        limit: int | None = None,
    ) -> GraphView:
        """Return a bounded slice of the graph, highest-confidence edges first."""
        limit = limit or settings.knowledge_graph_default_limit

        clauses = ["r.polarity = 'positive'", "r.confidence >= ?"]
        params: list[object] = [min_confidence]

        if relationship_type:
            clauses.append("r.type = ?")
            params.append(relationship_type)
        if document_id:
            clauses.append(
                "r.id IN (SELECT relationship_id FROM relationship_evidence WHERE document_id = ?)"
            )
            params.append(document_id)
        if entity_type:
            clauses.append("(s.type = ? OR t.type = ?)")
            params.extend([entity_type, entity_type])
        if search:
            clauses.append("(s.canonical_name LIKE ? OR t.canonical_name LIKE ?)")
            params.extend([f"%{search.lower()}%"] * 2)

        # One extra row reveals whether the limit truncated the result.
        query = f"""
            SELECT r.* FROM relationships r
            JOIN entities s ON s.id = r.source_id
            JOIN entities t ON t.id = r.target_id
            WHERE {' AND '.join(clauses)}
            ORDER BY r.confidence DESC, r.id
            LIMIT ?
        """
        with self._lock:
            rows = self.connection.execute(query, [*params, limit + 1]).fetchall()

        truncated = len(rows) > limit
        rows = rows[:limit]

        relationships = [self._to_relationship(row, with_evidence=False) for row in rows]
        entity_ids = {row["source_id"] for row in rows} | {row["target_id"] for row in rows}

        # An entity-type or search filter with no matching edges should still
        # show the matching nodes rather than an empty canvas.
        if not entity_ids and (entity_type or search or document_id):
            entities = self._isolated_entities(entity_type, search, document_id, limit)
        else:
            entities = self._entities_by_id(entity_ids)

        return GraphView(entities=entities, relationships=relationships, truncated=truncated)

    def entity(self, entity_id_value: str) -> EntityDetail | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM entities WHERE id = ?", (entity_id_value,)
            ).fetchone()
            if row is None:
                return None

            # Positive relations first: negated ones are shown for the record,
            # never at the top as though the entity asserted them.
            edges = self.connection.execute(
                "SELECT * FROM relationships WHERE source_id = ? OR target_id = ? "
                "ORDER BY (polarity = 'positive') DESC, confidence DESC",
                (entity_id_value, entity_id_value),
            ).fetchall()

        relationships = [self._to_relationship(edge) for edge in edges]
        neighbor_ids = {
            other
            for edge in edges
            for other in (edge["source_id"], edge["target_id"])
            if other != entity_id_value
        }

        return EntityDetail(
            entity=self._to_entity(row),
            relationships=relationships,
            neighbors=self._entities_by_id(neighbor_ids),
        )

    def neighbors(self, entity_id_value: str, depth: int = 1, limit: int | None = None) -> GraphView:
        """Breadth-first expansion, bounded by both depth and edge count."""
        limit = limit or settings.knowledge_graph_default_limit
        depth = max(1, min(depth, 3))

        seen: set[str] = {entity_id_value}
        frontier: set[str] = {entity_id_value}
        edges: dict[str, object] = {}

        with self._lock:
            for _ in range(depth):
                if not frontier or len(edges) >= limit:
                    break

                placeholders = ",".join("?" for _ in frontier)
                rows = self.connection.execute(
                    f"SELECT * FROM relationships "
                    f"WHERE (source_id IN ({placeholders}) OR target_id IN ({placeholders})) "
                    f"AND polarity = 'positive' "
                    f"ORDER BY confidence DESC LIMIT ?",
                    [*frontier, *frontier, limit],
                ).fetchall()

                next_frontier: set[str] = set()
                for row in rows:
                    edges[row["id"]] = row
                    for side in (row["source_id"], row["target_id"]):
                        if side not in seen:
                            seen.add(side)
                            next_frontier.add(side)
                frontier = next_frontier

        rows = list(edges.values())[:limit]
        return GraphView(
            entities=self._entities_by_id(seen),
            relationships=[self._to_relationship(row, with_evidence=False) for row in rows],
            truncated=len(edges) > limit,
        )

    def search(self, query: str, limit: int = 20) -> list[GraphEntity]:
        term = (query or "").strip().lower()
        if not term:
            return []

        with self._lock:
            rows = self.connection.execute(
                """
                SELECT * FROM entities
                WHERE canonical_name LIKE ? OR LOWER(display_name) LIKE ?
                ORDER BY
                    CASE WHEN canonical_name = ? THEN 0 ELSE 1 END,
                    mentions_count DESC
                LIMIT ?
                """,
                (f"%{term}%", f"%{term}%", term, limit),
            ).fetchall()

        return [self._to_entity(row) for row in rows]

    def stats(self) -> GraphStats:
        with self._lock:
            connection = self.connection
            entities = connection.execute("SELECT COUNT(*) AS n FROM entities").fetchone()["n"]
            relationships = connection.execute(
                "SELECT COUNT(*) AS n FROM relationships WHERE polarity = 'positive'"
            ).fetchone()["n"]
            by_entity = connection.execute(
                "SELECT type, COUNT(*) AS n FROM entities GROUP BY type ORDER BY n DESC"
            ).fetchall()
            by_relation = connection.execute(
                "SELECT type, COUNT(*) AS n FROM relationships "
                "WHERE polarity = 'positive' GROUP BY type ORDER BY n DESC"
            ).fetchall()
            documents = connection.execute(
                "SELECT COUNT(DISTINCT document_id) AS n FROM entity_sources"
            ).fetchone()["n"]

        return GraphStats(
            total_entities=entities,
            total_relationships=relationships,
            entities_by_type={row["type"]: row["n"] for row in by_entity},
            relationships_by_type={row["type"]: row["n"] for row in by_relation},
            documents_with_graph_data=documents,
        )

    # ── Row mapping ──────────────────────────────────────────────

    def _entities_by_id(self, ids) -> list[GraphEntity]:
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        with self._lock:
            rows = self.connection.execute(
                f"SELECT * FROM entities WHERE id IN ({placeholders}) ORDER BY mentions_count DESC",
                list(ids),
            ).fetchall()
        return [self._to_entity(row) for row in rows]

    def _isolated_entities(
        self, entity_type: str | None, search: str | None, document_id: str | None, limit: int
    ) -> list[GraphEntity]:
        clauses, params = [], []
        if entity_type:
            clauses.append("type = ?")
            params.append(entity_type)
        if search:
            clauses.append("canonical_name LIKE ?")
            params.append(f"%{search.lower()}%")
        if document_id:
            clauses.append("id IN (SELECT entity_id FROM entity_sources WHERE document_id = ?)")
            params.append(document_id)

        where = " AND ".join(clauses) if clauses else "1=1"
        with self._lock:
            rows = self.connection.execute(
                f"SELECT * FROM entities WHERE {where} ORDER BY mentions_count DESC LIMIT ?",
                [*params, limit],
            ).fetchall()
        return [self._to_entity(row) for row in rows]

    def _sources_for(self, entity_id_value: str) -> tuple[list[str], list[str]]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT document_id, chunk_id FROM entity_sources WHERE entity_id = ?",
                (entity_id_value,),
            ).fetchall()
        documents = sorted({row["document_id"] for row in rows if row["document_id"]})
        chunks = sorted({row["chunk_id"] for row in rows if row["chunk_id"]})
        return documents, chunks

    def _to_entity(self, row) -> GraphEntity:  # noqa: ANN001 - sqlite3.Row
        documents, chunks = self._sources_for(row["id"])
        return GraphEntity(
            id=row["id"],
            canonical_name=row["canonical_name"],
            display_name=row["display_name"],
            type=_safe_entity_type(row["type"]),
            description=row["description"],
            mentions_count=row["mentions_count"],
            confidence=row["confidence"],
            source_documents=documents,
            source_chunks=chunks,
            created_at=_parse_time(row["created_at"]),
            updated_at=_parse_time(row["updated_at"]),
        )

    def _to_relationship(self, row, with_evidence: bool = True) -> GraphRelationship:  # noqa: ANN001
        evidence: list[RelationshipEvidence] = []
        if with_evidence:
            with self._lock:
                rows = self.connection.execute(
                    "SELECT * FROM relationship_evidence WHERE relationship_id = ? "
                    "ORDER BY confidence DESC",
                    (row["id"],),
                ).fetchall()
            evidence = [
                RelationshipEvidence(
                    document_id=item["document_id"],
                    chunk_id=item["chunk_id"],
                    sentence=item["sentence"],
                    confidence=item["confidence"],
                )
                for item in rows
            ]

        return GraphRelationship(
            id=row["id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            type=_safe_relation_type(row["type"]),
            predicate=row["predicate"],
            confidence=row["confidence"],
            polarity=Polarity(row["polarity"]),
            evidence=evidence,
            created_at=_parse_time(row["created_at"]),
            updated_at=_parse_time(row["updated_at"]),
        )


def _safe_entity_type(value: str) -> EntityType:
    try:
        return EntityType(value)
    except ValueError:
        return EntityType.OTHER


def _safe_relation_type(value: str) -> RelationType:
    try:
        return RelationType(value)
    except ValueError:
        return RelationType.RELATED_TO


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


__all__ = ["SqliteGraphRepository", "entity_id", "relationship_id"]

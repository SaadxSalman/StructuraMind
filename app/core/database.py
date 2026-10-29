"""Database abstraction layer.

StructuraMind supports Qdrant as the production vector database. To keep the
application fully functional in environments where Qdrant is not running (e.g.
a fresh checkout without Docker), this module transparently falls back to a
lightweight in-memory cosine-similarity store backed by NumPy. The rest of the
codebase always interacts through the :class:`VectorDB` interface, so swapping
backends requires zero changes elsewhere.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from app.core.config import settings

logger = logging.getLogger("structuramind.database")


@dataclass
class VectorRecord:
    """A single vector record stored in the database."""

    id: str
    vector: List[float]
    payload: Dict = field(default_factory=dict)
    score: float = 0.0


class VectorDB:
    """Interface for vector storage + similarity search.

    Two backends are available:

    * ``Qdrant`` -- production-grade, persisted, hybrid keyword+vector search.
    * ``Memory`` -- in-memory NumPy cosine store (demo / offline fallback).
    """

    def __init__(self) -> None:
        self.backend: str = "memory"
        self._client = None
        self._collections: Dict[str, Dict[str, VectorRecord]] = {}
        self._use_qdrant = False
        self._connect()

    # ------------------------------------------------------------------ #
    # Connection
    # ------------------------------------------------------------------ #
    def _connect(self) -> None:
        """Attempt to connect to Qdrant; fall back to in-memory store."""
        try:
            from qdrant_client import QdrantClient  # type: ignore

            self._client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
                timeout=settings.qdrant_timeout_seconds,
            )
            # Probe the server.
            self._client.get_collections()
            self._use_qdrant = True
            self.backend = "qdrant"
            logger.info("Connected to Qdrant at %s", settings.qdrant_url)
        except Exception as exc:  # noqa: BLE001 - fallback is intentional
            logger.warning(
                "Qdrant unavailable (%s). Falling back to in-memory vector store. "
                "Install/run Qdrant for persisted, production-grade search.", exc
            )
            self._use_qdrant = False
            self.backend = "memory"
            self._client = None

    # ------------------------------------------------------------------ #
    # Collection management
    # ------------------------------------------------------------------ #
    def _ensure_collection(self, name: str, vector_size: int) -> None:
        if self._use_qdrant:
            from qdrant_client.http.models import Distance, VectorParams  # type: ignore

            existing = {c.name for c in self._client.get_collections().collections}
            if name not in existing:
                self._client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance[settings.qdrant_distance],
                    ),
                )
                logger.info("Created Qdrant collection '%s'", name)
        else:
            self._collections.setdefault(name, {})

    def list_collections(self) -> List[str]:
        if self._use_qdrant:
            return [c.name for c in self._client.get_collections().collections]
        return list(self._collections.keys())

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #
    def upsert(
        self,
        collection_name: str,
        records: Iterable[VectorRecord],
        vector_size: Optional[int] = None,
    ) -> None:
        """Upsert records into a collection."""
        records = list(records)
        if not records:
            return
        vector_size = vector_size or settings.qdrant_vector_size
        self._ensure_collection(collection_name, vector_size)

        if self._use_qdrant:
            from qdrant_client.http.models import PointStruct  # type: ignore

            points = [
                PointStruct(
                    id=r.id,
                    vector=r.vector,
                    payload=r.payload,
                )
                for r in records
            ]
            self._client.upsert(collection_name=collection_name, points=points)
            logger.info(
                "Upserted %d points into Qdrant collection '%s'", len(points), collection_name
            )
        else:
            for r in records:
                self._collections[collection_name][r.id] = r
            logger.info(
                "Upserted %d records into in-memory collection '%s'",
                len(records),
                collection_name,
            )

    # ------------------------------------------------------------------ #
    # Read / search
    # ------------------------------------------------------------------ #
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_payload: Optional[Dict] = None,
    ) -> List[VectorRecord]:
        """Semantic (vector) search for the most similar records."""
        query_vector = [float(v) for v in query_vector]
        if self._use_qdrant:
            from qdrant_client.http.models import (  # type: ignore
                FieldCondition,
                Filter,
                MatchValue,
            )

            qdrant_filter = None
            if filter_payload:
                conditions = [
                    FieldCondition(key=key, match=MatchValue(value=value))
                    for key, value in filter_payload.items()
                ]
                qdrant_filter = Filter(must=conditions)

            hits = self._client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=qdrant_filter,
                with_payload=True,
            )
            return [
                VectorRecord(
                    id=str(hit.id),
                    vector=list(hit.vector) if hit.vector else [],
                    payload=hit.payload or {},
                    score=float(hit.score),
                )
                for hit in hits
            ]

        # --- In-memory cosine similarity fallback --------------------- #
        collection = self._collections.get(collection_name, {})
        if not collection:
            return []

        q = np.array(query_vector, dtype=np.float64)
        q_norm = np.linalg.norm(q)
        scored: List[Tuple[str, float]] = []
        for rid, rec in collection.items():
            v = np.array(rec.vector, dtype=np.float64)
            denom = q_norm * np.linalg.norm(v)
            sim = float(np.dot(q, v) / denom) if denom > 0 else 0.0
            if score_threshold is not None and sim < score_threshold:
                continue
            scored.append((rid, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]
        return [
            VectorRecord(
                id=rid,
                vector=collection[rid].vector,
                payload=dict(collection[rid].payload),
                score=score,
            )
            for rid, score in top
        ]

    def scroll(
        self,
        collection_name: str,
        limit: int = 50,
        offset: Optional[str] = None,
        filter_payload: Optional[Dict] = None,
    ) -> Tuple[List[VectorRecord], Optional[str]]:
        """Scroll (paginate) through records in a collection."""
        if self._use_qdrant:
            from qdrant_client.http.models import (  # type: ignore
                FieldCondition,
                Filter,
                MatchValue,
            )

            qdrant_filter = None
            if filter_payload:
                conditions = [
                    FieldCondition(key=k, match=MatchValue(value=v))
                    for k, v in filter_payload.items()
                ]
                qdrant_filter = Filter(must=conditions)

            points, next_page = self._client.scroll(
                collection_name=collection_name,
                limit=limit,
                offset=offset,
                scroll_filter=qdrant_filter,
                with_payload=True,
            )
            return [
                VectorRecord(
                    id=str(p.id),
                    vector=list(p.vector) if p.vector else [],
                    payload=p.payload or {},
                )
                for p in points
            ], str(next_page) if next_page else None

        collection = self._collections.get(collection_name, {})
        items = list(collection.values())
        if filter_payload:
            items = [
                r
                for r in items
                if all(r.payload.get(k) == v for k, v in filter_payload.items())
            ]
        start = 0
        if offset:
            try:
                start = int(offset)
            except ValueError:
                start = 0
        page = items[start : start + limit]
        next_offset = str(start + limit) if start + limit < len(items) else None
        return page, next_offset

    def delete_collection(self, collection_name: str) -> None:
        if self._use_qdrant:
            self._client.delete_collection(collection_name=collection_name)
        else:
            self._collections.pop(collection_name, None)

    def count(self, collection_name: str) -> int:
        if self._use_qdrant:
            return self._client.count(collection_name=collection_name).count
        return len(self._collections.get(collection_name, {}))

    # ------------------------------------------------------------------ #
    # Helper
    # ------------------------------------------------------------------ #
    @staticmethod
    def new_id() -> str:
        return str(uuid.uuid4())


# Singleton instance used across the application.
vector_db = VectorDB()


__all__ = ["VectorDB", "VectorRecord", "vector_db"]


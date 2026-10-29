"""High-level vector store for StructuraMind chunks.

Wraps :class:`app.core.database.VectorDB` and adds collection management for
the three logical domains:

* ``structuramind_codes``   - building-code clauses (ACI, AISC, IBC, Eurocodes)
* ``structuramind_bim``     - IFC / BIM element metadata
* ``structuramind_reports`` - geotechnical / project reports
"""

from __future__ import annotations

import logging
from typing import List, Optional

from app.core.config import settings
from app.core.database import VectorRecord, vector_db
from app.embeddings.embedder import Embedder
from app.ingestion.chunker import ChunkRecord

logger = logging.getLogger("structuramind.vectordb")

# Logical collection -> settings field name mapping.
COLLECTION_SETTINGS = {
    "codes": settings.qdrant_collection_codes,
    "bim": settings.qdrant_collection_bim,
    "reports": settings.qdrant_collection_reports,
}


class QdrantStore:
    """Store and search embedded chunks in Qdrant (or the in-memory fallback)."""

    def __init__(self) -> None:
        self.db = vector_db
        self.embedder = Embedder()
        # Ensure the embedding dimension matches the configured vector size.
        if self.embedder.dimension != settings.qdrant_vector_size:
            logger.warning(
                "Embedder dimension (%d) != configured QDRANT_VECTOR_SIZE (%d). "
                "Using embedder dimension.",
                self.embedder.dimension,
                settings.qdrant_vector_size,
            )

    # ------------------------------------------------------------------ #
    # Collection helpers
    # ------------------------------------------------------------------ #
    def _collection_name(self, collection: str) -> str:
        return COLLECTION_SETTINGS.get(collection, collection)

    def ensure_collections(self) -> None:
        """Create all logical collections if they do not exist."""
        for collection in ("codes", "bim", "reports"):
            name = self._collection_name(collection)
            self.db._ensure_collection(name, self.embedder.dimension)
        logger.info("Ensured vector collections exist: %s", list(COLLECTION_SETTINGS.values()))

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #
    def add_chunks(self, chunks: List[ChunkRecord], collection: str = "codes") -> int:
        """Embed and upsert a list of :class:`ChunkRecord`s."""
        if not chunks:
            return 0
        name = self._collection_name(collection)
        self.db._ensure_collection(name, self.embedder.dimension)

        texts = [c.text for c in chunks]
        vectors = self.embedder.embed_texts(texts)

        records: List[VectorRecord] = []
        for chunk, vector in zip(chunks, vectors):
            payload = dict(chunk.metadata)
            payload["text"] = chunk.text
            payload["collection"] = collection
            records.append(
                VectorRecord(
                    id=chunk.chunk_id,
                    vector=vector,
                    payload=payload,
                )
            )

        self.db.upsert(collection_name=name, records=records, vector_size=self.embedder.dimension)
        logger.info("Added %d chunks to collection '%s'", len(records), name)
        return len(records)

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        collection: str = "codes",
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Convenience: embed raw texts + metadata and upsert."""
        from app.ingestion.chunker import ChunkRecord
        import uuid

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        if metadatas is None:
            metadatas = [{} for _ in texts]

        chunk_records = [
            ChunkRecord(
                chunk_id=ids[i],
                document_id=metadatas[i].get("document_id", "unknown"),
                filename=metadatas[i].get("filename", "unknown"),
                collection=collection,
                text=texts[i],
                metadata=metadatas[i],
            )
            for i in range(len(texts))
        ]
        self.add_chunks(chunk_records, collection=collection)
        return ids


    # ------------------------------------------------------------------ #
    # Read / search
    # ------------------------------------------------------------------ #
    def search(
        self,
        query: str,
        collection: str = "codes",
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_payload: Optional[dict] = None,
    ) -> List[dict]:
        """Semantic search for a query string; returns ranked result dicts."""
        name = self._collection_name(collection)
        query_vector = self.embedder.embed_text(query)
        records = self.db.search(
            collection_name=name,
            query_vector=query_vector,
            top_k=top_k,
            score_threshold=score_threshold,
            filter_payload=filter_payload,
        )
        return [
            {
                "id": r.id,
                "score": r.score,
                "text": r.payload.get("text", ""),
                "metadata": {k: v for k, v in r.payload.items() if k != "text"},
            }
            for r in records
        ]

    def search_multi(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> dict:
        """Search across multiple collections and return per-collection results."""
        if collections is None:
            collections = ["codes", "bim", "reports"]
        results = {}
        for collection in collections:
            try:
                results[collection] = self.search(
                    query=query, collection=collection, top_k=top_k
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Search failed for collection '%s': %s", collection, exc)
                results[collection] = []
        return results

    def get_document_chunks(self, document_id: str, collection: str = "codes") -> List[dict]:
        """Retrieve all chunks belonging to a document."""
        name = self._collection_name(collection)
        records, _ = self.db.scroll(
            collection_name=name,
            limit=1000,
            filter_payload={"document_id": document_id},
        )
        return [
            {
                "id": r.id,
                "text": r.payload.get("text", ""),
                "metadata": {k: v for k, v in r.payload.items() if k != "text"},
            }
            for r in records
        ]

    def count(self, collection: str = "codes") -> int:
        name = self._collection_name(collection)
        return self.db.count(name)

    def delete_document(self, document_id: str, collection: str = "codes") -> None:
        """Delete all chunks for a document."""
        name = self._collection_name(collection)
        records, _ = self.db.scroll(
            collection_name=name, limit=10000, filter_payload={"document_id": document_id}
        )
        if records:
            if self.db._use_qdrant:
                from qdrant_client.http.models import PointIdsList  # type: ignore

                self.db._client.delete(
                    collection_name=name,
                    points_selector=PointIdsList(points=[r.id for r in records]),
                )
            else:
                for r in records:
                    self.db._collections[name].pop(r.id, None)
            logger.info("Deleted %d chunks for document '%s'", len(records), document_id)


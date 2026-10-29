"""Document listing and health routes."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter

from app.api.models import DocumentOut, HealthResponse
from app.core.config import settings
from app.core.database import vector_db
from app.core.logging import logger
from app.vectordb.qdrant_store import QdrantStore

router = APIRouter(tags=["documents"])
_store = QdrantStore()


@router.get("/documents", response_model=List[DocumentOut])
async def list_documents(collection: str = "codes"):
    """List ingested documents in a collection."""
    name = QdrantStore()._collection_name(collection)
    records, _ = vector_db.scroll(collection_name=name, limit=10000)
    # Group by document_id.
    docs: dict = {}
    for r in records:
        md = r.payload or {}
        did = md.get("document_id", "unknown")
        if did not in docs:
            docs[did] = {
                "document_id": did,
                "filename": md.get("filename", "unknown"),
                "collection": collection,
                "chunk_count": 0,
            }
        docs[did]["chunk_count"] += 1
    return list(docs.values())


@router.get("/documents/{collection}", response_model=List[DocumentOut])
async def list_documents_collection(collection: str):
    return await list_documents(collection)


@router.get("/health", response_model=HealthResponse)
async def health():
    """Service health check with backend/vector-db/embedding/llm status."""
    collections = vector_db.list_collections()
    return HealthResponse(
        status="ok",
        backend="fastapi",
        vector_db=vector_db.backend,
        embedding_provider=__import__("app.embeddings.embedder", fromlist=["Embedder"]).Embedder().provider,
        llm_provider=settings.llm_provider,
        collections=collections,
        demo_mode=settings.demo_mode,
    )

"""Document ingestion routes (PDF + IFC/BIM)."""

from __future__ import annotations

import os
from typing import List

from fastapi import APIRouter, File, UploadFile, HTTPException

from app.api.models import IngestResponse, IngestTextRequest
from app.core.config import settings
from app.core.database import vector_db
from app.core.logging import logger
from app.embeddings.embedder import Embedder
from app.ingestion.chunker import chunk_ifc_model, chunk_pdf_document
from app.ingestion.ifc_parser import IFCParser
from app.ingestion.pdf_parser import PDFParser
from app.vectordb.qdrant_store import QdrantStore

router = APIRouter(prefix="/ingest", tags=["ingestion"])

_pdf_parser = PDFParser()
_ifc_parser = IFCParser()
_store = QdrantStore()
_embedder = Embedder()


@router.post("/pdf", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    """Upload and ingest a building-code PDF into the codes collection."""
    if not settings.enable_document_upload:
        raise HTTPException(status_code=403, detail="Document upload is disabled")

    contents = await file.read()
    try:
        parsed = _pdf_parser.parse_bytes(contents, file.filename or "uploaded.pdf")
    except Exception as exc:  # noqa: BLE001
        logger.exception("PDF parsing failed for %s", file.filename)
        raise HTTPException(status_code=400, detail=f"PDF parsing failed: {exc}")

    from app.ingestion.chunker import chunk_pdf_document as _chunk
    result = _chunk(parsed, parsed.document_id, collection="codes")

    # Ensure the Qdrant collection's vector size matches the embedder.
    store = QdrantStore()
    added = store.add_chunks(result.chunks, collection="codes")

    logger.info("Ingested PDF '%s': %d chunks", file.filename, added)
    return IngestResponse(
        document_id=parsed.document_id,
        filename=parsed.filename,
        collection="codes",
        chunk_count=added,
        status="ok",
        message=f"Parsed {parsed.total_pages} pages into {added} chunks.",
    )


@router.post("/ifc", response_model=IngestResponse)
async def ingest_ifc(file: UploadFile = File(...)):
    """Upload and ingest an IFC/BIM file into the bim collection."""
    if not settings.enable_document_upload:
        raise HTTPException(status_code=403, detail="Document upload is disabled")

    contents = await file.read()
    try:
        parsed = _ifc_parser.parse_bytes(contents, file.filename or "uploaded.ifc")
    except Exception as exc:  # noqa: BLE001
        logger.exception("IFC parsing failed for %s", file.filename)
        raise HTTPException(status_code=400, detail=f"IFC parsing failed: {exc}")

    result = chunk_ifc_model(parsed, parsed.document_id, collection="bim")
    store = QdrantStore()
    added = store.add_chunks(result.chunks, collection="bim")

    logger.info("Ingested IFC '%s': %d elements", file.filename, added)
    return IngestResponse(
        document_id=parsed.document_id,
        filename=parsed.filename,
        collection="bim",
        chunk_count=added,
        status="ok",
        message=f"Extracted {parsed.element_count} elements into {added} chunks.",
    )


@router.post("/text", response_model=IngestResponse)
async def ingest_text(payload: IngestTextRequest):
    """Ingest raw text directly (useful for pasting code clauses)."""
    text = payload.text
    document_id = payload.document_id
    collection = payload.collection
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty")
    store = QdrantStore()
    ids = store.add_texts(
        texts=[text],
        metadatas=[{"document_id": document_id, "filename": "manual.txt", "source_type": "text"}],
        collection=collection,
        ids=[document_id],
    )
    return IngestResponse(
        document_id=document_id,
        filename="manual.txt",
        collection=collection,
        chunk_count=len(ids),
        status="ok",
        message="Text ingested.",
    )

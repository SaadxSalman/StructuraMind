"""Hierarchical & semantic chunking utilities.

Builds on :class:`PDFChunk` / :class:`IFCElement` to produce embedding-ready
records with preserved parent-child clause hierarchies, page numbers, and
spatial coordinates for citations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from app.ingestion.pdf_parser import PDFChunk


@dataclass
class ChunkRecord:
    """A final chunk ready for embedding + vector storage."""

    chunk_id: str
    document_id: str
    filename: str
    collection: str  # codes | bim | reports
    text: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class ChunkingResult:
    """Outcome of the chunking step."""

    chunks: List[ChunkRecord] = field(default_factory=list)
    chunk_count: int = 0
    document_id: str = ""
    collection: str = ""


def chunk_pdf_document(parsed, document_id: str, collection: str = "codes") -> ChunkingResult:
    """Convert a :class:`ParsedDocument` into :class:`ChunkRecord`s."""
    from app.ingestion.pdf_parser import ParsedDocument

    if not isinstance(parsed, ParsedDocument):
        raise TypeError("chunk_pdf_document expects a ParsedDocument instance")

    chunks: List[ChunkRecord] = []
    for c in parsed.chunks:
        metadata = {
            "document_id": document_id,
            "filename": parsed.filename,
            "page_number": c.page_number,
            "chapter": c.chapter,
            "section": c.section,
            "clause": c.clause,
            "title": c.title,
            "parent_chunk_id": c.parent_chunk_id,
            "level": c.level,
            "bbox": c.bbox,
            "char_start": c.char_start,
            "char_end": c.char_end,
            "source_type": "pdf",
        }
        chunks.append(
            ChunkRecord(
                chunk_id=c.chunk_id,
                document_id=document_id,
                filename=parsed.filename,
                collection=collection,
                text=c.text,
                metadata=metadata,
            )
        )

    return ChunkingResult(
        chunks=chunks,
        chunk_count=len(chunks),
        document_id=document_id,
        collection=collection,
    )


def chunk_ifc_model(parsed, document_id: str, collection: str = "bim") -> ChunkingResult:
    """Convert a :class:`ParsedIFCModel` into :class:`ChunkRecord`s."""
    from app.ingestion.ifc_parser import ParsedIFCModel

    if not isinstance(parsed, ParsedIFCModel):
        raise TypeError("chunk_ifc_model expects a ParsedIFCModel instance")

    chunks: List[ChunkRecord] = []
    for el in parsed.elements:
        text_parts = [
            f"IFC Element: {el.ifc_type}",
            f"Name: {el.name}" if el.name else "",
            f"Description: {el.description}" if el.description else "",
            f"Level/Storey: {el.level}" if el.level else "",
            f"Material: {el.material}" if el.material else "",
            f"Properties: {el.properties}" if el.properties else "",
            f"GUID: {el.ifc_guid}",
        ]
        text = "\n".join(p for p in text_parts if p)
        metadata = {
            "document_id": document_id,
            "filename": parsed.filename,
            "page_number": None,
            "chapter": None,
            "section": el.level,
            "clause": el.ifc_type,
            "title": el.name,
            "parent_chunk_id": None,
            "level": 1,
            "bbox": el.bbox,
            "char_start": 0,
            "char_end": len(text),
            "source_type": "ifc",
            "ifc_guid": el.ifc_guid,
            "ifc_type": el.ifc_type,
            "material": el.material,
        }
        chunks.append(
            ChunkRecord(
                chunk_id=el.element_id,
                document_id=document_id,
                filename=parsed.filename,
                collection=collection,
                text=text,
                metadata=metadata,
            )
        )

    return ChunkingResult(
        chunks=chunks,
        chunk_count=len(chunks),
        document_id=document_id,
        collection=collection,
    )

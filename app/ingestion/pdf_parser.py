"""Hierarchical PDF parser for civil-engineering codes.

Extracts text page-by-page using PyMuPDF, detects nested section headers
(Chapter -> Section -> Clause), and emits parent-aware chunks so the retriever
never loses critical contextual limits (e.g. seismic load exceptions).
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

import fitz  # PyMuPDF

from app.core.config import settings

logger = logging.getLogger("structuramind.ingestion.pdf_parser")

# Regexes for common civil-code heading patterns.
# Matches e.g. "CHAPTER 3", "Chapter 3", "3", "3.1", "3.1.1", "3.1.1.2".
SECTION_RE = re.compile(
    r"^\s*(?:(?P<chapter>chapter\s+)?(?P<num>\d{1,3}(?:\.\d{1,3}){0,4})\b(?:\s*[:\-–]\s*(?P<title>.{0,120}))?)",
    re.IGNORECASE,
)
CHAPTER_RE = re.compile(r"^\s*chapter\s+(\d{1,3})\b", re.IGNORECASE)
APPENDIX_RE = re.compile(r"^\s*appendix\s+([A-Z0-9]{1,3})\b", re.IGNORECASE)


@dataclass
class PDFChunk:
    """A single hierarchical chunk extracted from a PDF."""

    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    chapter: Optional[str]
    section: Optional[str]
    clause: Optional[str]
    title: Optional[str]
    text: str
    char_start: int
    char_end: int
    bbox: List[float] = field(default_factory=list)  # [x0, y0, x1, y1]
    parent_chunk_id: Optional[str] = None
    level: int = 0  # 0=chapter, 1=section, 2=clause, 3=body


@dataclass
class ParsedDocument:
    """Result of parsing a PDF document."""

    document_id: str
    filename: str
    total_pages: int
    chunks: List[PDFChunk] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class PDFParser:
    """Parse a PDF into hierarchy-aware chunks."""

    def __init__(self) -> None:
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        self.max_chunk_characters = settings.max_chunk_characters

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def parse_file(self, file_path: str, document_id: Optional[str] = None) -> ParsedDocument:
        """Parse a PDF file on disk into :class:`ParsedDocument`."""
        import uuid

        if document_id is None:
            document_id = str(uuid.uuid4())

        filename = file_path.split("/")[-1].split("\\")[-1]
        doc = fitz.open(file_path)
        try:
            chunks = self._parse_document(doc, document_id, filename)
        finally:
            doc.close()

        logger.info(
            "Parsed PDF '%s': %d pages, %d chunks",
            filename,
            doc.page_count if hasattr(doc, "page_count") else 0,
            len(chunks),
        )
        return ParsedDocument(
            document_id=document_id,
            filename=filename,
            total_pages=len(set(c.page_number for c in chunks)),
            chunks=chunks,
            metadata={"parser": settings.pdf_parser, "document_id": document_id},
        )

    def parse_bytes(self, file_bytes: bytes, filename: str, document_id: Optional[str] = None) -> ParsedDocument:
        """Parse PDF bytes (e.g. from an uploaded file)."""
        import uuid

        if document_id is None:
            document_id = str(uuid.uuid4())

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        try:
            chunks = self._parse_document(doc, document_id, filename)
        finally:
            doc.close()

        return ParsedDocument(
            document_id=document_id,
            filename=filename,
            total_pages=len(set(c.page_number for c in chunks)),
            chunks=chunks,
            metadata={"parser": settings.pdf_parser, "document_id": document_id},
        )

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #
    def _parse_document(self, doc: fitz.Document, document_id: str, filename: str) -> List[PDFChunk]:
        chunks: List[PDFChunk] = []
        current_chapter: Optional[str] = None
        current_section: Optional[str] = None
        current_clause: Optional[str] = None
        current_title: Optional[str] = None

        for page_index in range(doc.page_count):
            page = doc[page_index]
            page_number = page_index + 1
            blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)

            page_text = ""
            for block in blocks:
                x0, y0, x1, y1, block_text, _block_no, _block_type = block[:7]
                stripped = block_text.strip()
                if not stripped:
                    continue

                heading = self._detect_heading(stripped)
                if heading:
                    # Flush any pending body text as a chunk before starting a new section.
                    if page_text.strip():
                        chunks.append(
                            self._make_body_chunk(
                                document_id, filename, page_number,
                                current_chapter, current_section, current_clause,
                                current_title, page_text.strip(),
                            )
                        )
                        page_text = ""

                    current_chapter = heading.get("chapter") or current_chapter
                    current_section = heading.get("section")
                    current_clause = heading.get("clause")
                    current_title = heading.get("title")

                    # Emit the heading itself as a structural chunk.
                    chunks.append(
                        PDFChunk(
                            chunk_id=str(uuid.uuid4()),
                            document_id=document_id,
                            filename=filename,
                            page_number=page_number,
                            chapter=current_chapter,
                            section=current_section,
                            clause=current_clause,
                            title=current_title,
                            text=stripped,
                            char_start=0,
                            char_end=len(stripped),
                            bbox=[x0, y0, x1, y1],
                            parent_chunk_id=None,
                            level=heading.get("level", 1),
                        )
                    )
                else:
                    page_text += stripped + "\n"

            # Flush remaining body text on the page.
            if page_text.strip():
                chunks.append(
                    self._make_body_chunk(
                        document_id, filename, page_number,
                        current_chapter, current_section, current_clause,
                        current_title, page_text.strip(),
                    )
                )

        return self._post_process(chunks)

    def _detect_heading(self, text: str) -> Optional[dict]:
        """Detect whether a line is a chapter/section/clause heading."""
        m = CHAPTER_RE.match(text)
        if m:
            return {
                "chapter": f"Chapter {m.group(1)}",
                "section": None,
                "clause": None,
                "title": text.strip(),
                "level": 0,
            }

        m = APPENDIX_RE.match(text)
        if m:
            return {
                "chapter": f"Appendix {m.group(1)}",
                "section": None,
                "clause": None,
                "title": text.strip(),
                "level": 0,
            }

        m = SECTION_RE.match(text)
        if m:
            num = m.group("num")
            title = (m.group("title") or "").strip()
            if not num:
                return None
            parts = num.split(".")
            level = len(parts) - 1
            chapter = None
            section = None
            clause = None
            if level >= 0:
                chapter = parts[0]
            if level >= 1:
                section = num
            if level >= 2:
                clause = num
            return {
                "chapter": chapter,
                "section": section,
                "clause": clause,
                "title": title or None,
                "level": max(1, min(level, 3)),
            }
        return None

    def _make_body_chunk(
        self,
        document_id: str,
        filename: str,
        page_number: int,
        chapter: Optional[str],
        section: Optional[str],
        clause: Optional[str],
        title: Optional[str],
        text: str,
    ) -> PDFChunk:
        import uuid

        # Split long body text into overlapping sub-chunks.
        sub_chunks = self._split_text(text)
        results: List[PDFChunk] = []
        for i, sub in enumerate(sub_chunks):
            results.append(
                PDFChunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    filename=filename,
                    page_number=page_number,
                    chapter=chapter,
                    section=section,
                    clause=clause,
                    title=title,
                    text=sub,
                    char_start=0,
                    char_end=len(sub),
                    bbox=[],
                    parent_chunk_id=None,
                    level=3,
                )
            )
        if results:
            return results[0]
        return PDFChunk(
            chunk_id=str(uuid.uuid4()),
            document_id=document_id,
            filename=filename,
            page_number=page_number,
            chapter=chapter,
            section=section,
            clause=clause,
            title=title,
            text=text,
            char_start=0,
            char_end=len(text),
            bbox=[],
            parent_chunk_id=None,
            level=3,
        )

    @staticmethod
    def _split_text(text: str) -> List[str]:
        """Simple sentence-aware splitting with overlap."""
        if len(text) <= settings.max_chunk_characters:
            return [text]
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: List[str] = []
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) + 1 > settings.max_chunk_characters and current:
                chunks.append(current.strip())
                current = sentence
            else:
                current = (current + " " + sentence).strip()
        if current.strip():
            chunks.append(current.strip())
        return chunks or [text]

    def _post_process(self, chunks: List[PDFChunk]) -> List[PDFChunk]:
        """Attach parent links and drop empty chunks."""
        cleaned = [c for c in chunks if c.text.strip()]
        # Assign a running parent: the most recent structural chunk (level < 3).
        last_structural: Optional[PDFChunk] = None
        for c in cleaned:
            if c.level < 3:
                last_structural = c
            elif last_structural is not None:
                c.parent_chunk_id = last_structural.chunk_id
        return cleaned


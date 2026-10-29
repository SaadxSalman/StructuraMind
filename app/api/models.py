"""Pydantic request/response models for the StructuraMind API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
class IngestResponse(BaseModel):
    document_id: str
    filename: str
    collection: str
    chunk_count: int
    status: str
    message: Optional[str] = None


class IngestListItem(BaseModel):
    document_id: str
    filename: str
    collection: str
    chunk_count: int


class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=1)
    document_id: str = "manual"
    collection: str = "codes"


# ---------------------------------------------------------------------------
# Query / compliance
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Engineering compliance question")
    collections: Optional[List[str]] = Field(
        default=None, description="Collections to search: codes, bim, reports"
    )
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    concrete_strength_mpa: Optional[float] = Field(default=None)
    steel_yield_mpa: Optional[float] = Field(default=None)


class CitationOut(BaseModel):
    chunk_id: str
    score: float
    collection: str
    text: str
    filename: Optional[str] = None
    page_number: Optional[int] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    clause: Optional[str] = None
    title: Optional[str] = None
    bbox: Optional[List[float]] = None
    source_type: Optional[str] = None


class CheckResult(BaseModel):
    check: str
    code_reference: str
    result: str
    pass_: bool = Field(alias="pass")
    details: str

    model_config = {"populate_by_name": True}


class CalculationOut(BaseModel):
    concrete_strength_mpa: float
    steel_yield_mpa: float
    inputs_extracted: Dict[str, Optional[float]]
    checks: List[CheckResult]
    all_pass: bool
    summary: str
    units: str


class CriticismOut(BaseModel):
    verdict: str
    confidence: float
    passed: bool
    issues: List[str]
    threshold: float
    citations_checked: int


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[CitationOut]
    verdict: Dict[str, Any]
    calculation: Optional[CalculationOut] = None
    criticism: Optional[CriticismOut] = None
    agentic_loop: bool
    iterations: int


# ---------------------------------------------------------------------------
# Documents / health
# ---------------------------------------------------------------------------
class DocumentOut(BaseModel):
    document_id: str
    filename: str
    collection: str
    chunk_count: int


class HealthResponse(BaseModel):
    status: str
    backend: str
    vector_db: str
    embedding_provider: str
    llm_provider: str
    collections: List[str]
    demo_mode: bool

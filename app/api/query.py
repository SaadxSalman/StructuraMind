"""Code-compliance query route powered by the LangGraph agentic loop."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException

from app.api.models import (
    CalculationOut,
    CheckResult,
    CitationOut,
    CriticismOut,
    QueryRequest,
    QueryResponse,
)
from app.agents.graph import StructuraMindGraph
from app.core.config import settings
from app.core.logging import logger
from app.vectordb.qdrant_store import QdrantStore

router = APIRouter(prefix="/query", tags=["query"])

_store = QdrantStore()
_graph = StructuraMindGraph(store=_store)


@router.post("/", response_model=QueryResponse)
async def run_query(request: QueryRequest):
    """Run a code-compliance query through the Retrieval → Calculation → Critic loop."""
    try:
        result = _graph.run(
            query=request.query,
            collections=request.collections,
            top_k=request.top_k,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Query execution failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")

    # Map citations to output model.
    citations_out: List[CitationOut] = []
    for c in result.get("citations", []):
        citations_out.append(
            CitationOut(
                chunk_id=c.get("chunk_id", ""),
                score=float(c.get("score", 0.0)),
                collection=c.get("collection", "codes"),
                text=c.get("text", ""),
                filename=c.get("filename"),
                page_number=c.get("page_number"),
                chapter=c.get("chapter"),
                section=c.get("section"),
                clause=c.get("clause"),
                title=c.get("title"),
                bbox=c.get("bbox"),
                source_type=c.get("source_type"),
            )
        )

    # Map calculation.
    calc_out: Optional[CalculationOut] = None
    calc_raw = result.get("calculation") or {}
    if calc_raw and not calc_raw.get("skipped"):
        checks_out: List[CheckResult] = []
        for c in calc_raw.get("checks", []):
            checks_out.append(
                CheckResult(
                    check=c.get("check", ""),
                    code_reference=c.get("code_reference", ""),
                    result=c.get("result", ""),
                    pass_=bool(c.get("pass", False)),
                    details=c.get("details", ""),
                )
            )
        calc_out = CalculationOut(
            concrete_strength_mpa=float(calc_raw.get("concrete_strength_mpa", 0.0)),
            steel_yield_mpa=float(calc_raw.get("steel_yield_mpa", 0.0)),
            inputs_extracted=calc_raw.get("inputs_extracted", {}),
            checks=checks_out,
            all_pass=bool(calc_raw.get("all_pass", False)),
            summary=calc_raw.get("summary", ""),
            units=calc_raw.get("units", settings.default_units),
        )

    # Map criticism.
    critic_raw = result.get("criticism") or {}
    critic_out: Optional[CriticismOut] = None
    if critic_raw and not critic_raw.get("skipped"):
        critic_out = CriticismOut(
            verdict=critic_raw.get("verdict", ""),
            confidence=float(critic_raw.get("confidence", 0.0)),
            passed=bool(critic_raw.get("passed", False)),
            issues=critic_raw.get("issues", []),
            threshold=float(critic_raw.get("threshold", settings.critic_hallucination_threshold)),
            citations_checked=int(critic_raw.get("citations_checked", 0)),
        )

    return QueryResponse(
        query=result.get("query", request.query),
        answer=result.get("answer", ""),
        citations=citations_out,
        verdict=result.get("verdict", {}),
        calculation=calc_out,
        criticism=critic_out,
        agentic_loop=bool(result.get("agentic_loop", False)),
        iterations=int(result.get("iterations", 0)),
    )

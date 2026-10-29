"""Retrieval Agent.

Given a user query, searches the vector database for relevant code clauses,
BIM elements, and report passages. Returns ranked results with full citation
metadata (page number, section, clause, bounding box) for spatial citations.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from app.agents.base import BaseAgent
from app.core.config import settings
from app.vectordb.qdrant_store import QdrantStore

logger = logging.getLogger("structuramind.agents.retrieval")


class RetrievalAgent(BaseAgent):
    """Retrieves relevant context for an engineering query."""

    def __init__(self, store: Optional[QdrantStore] = None) -> None:
        super().__init__("RetrievalAgent")
        self.store = store or QdrantStore()

    def retrieve(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> dict:
        """Run semantic + (optionally) hybrid search across collections."""
        if collections is None:
            collections = ["codes", "bim", "reports"]
        k = top_k or settings.top_k_retrieval

        logger.info("RetrievalAgent: query='%s' collections=%s top_k=%d", query, collections, k)
        results = self.store.search_multi(query=query, collections=collections, top_k=k)

        # Flatten + rank all results by score.
        all_results = []
        for collection, hits in results.items():
            for hit in hits:
                hit["collection"] = collection
                all_results.append(hit)
        all_results.sort(key=lambda x: x["score"], reverse=True)
        all_results = all_results[:k]

        # Build citation list.
        citations = []
        for hit in all_results:
            md = hit["metadata"]
            citations.append(
                {
                    "chunk_id": hit["id"],
                    "score": hit["score"],
                    "collection": hit["collection"],
                    "text": hit["text"],
                    "filename": md.get("filename"),
                    "page_number": md.get("page_number"),
                    "chapter": md.get("chapter"),
                    "section": md.get("section"),
                    "clause": md.get("clause"),
                    "title": md.get("title"),
                    "bbox": md.get("bbox"),
                    "source_type": md.get("source_type"),
                }
            )

        context = "\n\n".join(
            f"[Source {i+1}] {c['text']}" for i, c in enumerate(citations)
        )

        return {
            "query": query,
            "citations": citations,
            "context": context,
            "result_count": len(citations),
            "collections": collections,
        }

    def call_llm(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        """Retrieval-specific mock/LLM path."""
        return super().call_llm(system_prompt, user_prompt, temperature)

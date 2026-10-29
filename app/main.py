"""StructuraMind FastAPI application.

Run locally:

    uvicorn app.main:app --reload --port 8000

Or with Docker:

    docker compose up -d
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import documents, ingest, query
from app.core.config import settings
from app.core.database import vector_db
from app.core.logging import setup_logging, logger
from app.vectordb.qdrant_store import QdrantStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: setup logging + ensure vector collections."""
    setup_logging()
    logger.info("Starting StructuraMind backend (env=%s)", settings.app_env)
    try:
        QdrantStore().ensure_collections()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not ensure collections: %s", exc)
    yield
    logger.info("Shutting down StructuraMind backend")


app = FastAPI(
    title="StructuraMind",
    description=(
        "Multi-Modal RAG & Code Compliance Engine for Civil Engineering. "
        "Parses building codes, geotechnical reports, and BIM/IFC models to "
        "perform real-time code compliance checks with grounded citations."
    ),
    version="1.0.0",
    contact={"name": "StructuraMind", "url": "https://github.com/saadxsalman/structuramind-ai"},
    license_info={"name": "MIT"},
    lifespan=lifespan,
)

# CORS for the Streamlit frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers.
app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(documents.router)


@app.get("/", tags=["root"])
async def root() -> Dict[str, Any]:
    return {
        "name": settings.app_name,
        "version": "1.0.0",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["root"])
async def health_alias() -> Dict[str, str]:
    return {"status": "ok"}

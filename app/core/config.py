"""Application configuration loaded from the single ``.env`` file.

All secrets and settings live in ``.env`` (gitignored). This module uses
:class:`pydantic-settings.BaseSettings` so values can be overridden by
environment variables at deploy time (Docker, Kubernetes, GitHub Actions, etc.).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """StructuraMind runtime settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Application
    # -------------------------------------------------------------------------
    app_name: str = "StructuraMind"
    app_env: str = "development"
    debug: bool = True
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    streamlit_port: int = 8501
    backend_url: str = "http://localhost:8000"
    frontend_url: str = "http://localhost:8501"
    secret_key: str = "structuramind-dev-secret-change-in-production"
    access_token_expire_minutes: int = 60
    # -------------------------------------------------------------------------
    # LLM Providers
    # -------------------------------------------------------------------------
    llm_provider: str = "anthropic"  # anthropic | openai | local | mock
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o"
    google_api_key: Optional[str] = None
    google_model: str = "gemini-1.5-pro"

    # -------------------------------------------------------------------------
    # Embeddings
    # -------------------------------------------------------------------------
    embedding_provider: str = "local"  # local | openai | anthropic
    local_embedding_model: str = "all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = 32

    # -------------------------------------------------------------------------
    # Vector Database (Qdrant)
    # -------------------------------------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    qdrant_collection_codes: str = "structuramind_codes"
    qdrant_collection_bim: str = "structuramind_bim"
    qdrant_collection_reports: str = "structuramind_reports"
    qdrant_vector_size: int = 384
    qdrant_distance: str = "Cosine"
    qdrant_timeout_seconds: int = 30
    # -------------------------------------------------------------------------
    # RAG & Chunking
    # -------------------------------------------------------------------------
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_retrieval: int = 10
    top_k_rerank: int = 5
    enable_hybrid_search: bool = True
    enable_reranking: bool = True
    max_chunk_characters: int = 4000

    # -------------------------------------------------------------------------
    # Agentic Verification Loop (LangGraph)
    # -------------------------------------------------------------------------
    enable_agentic_loop: bool = True
    max_agent_iterations: int = 3
    enable_retrieval_agent: bool = True
    enable_calculation_agent: bool = True
    enable_critic_agent: bool = True
    critic_hallucination_threshold: float = 0.75

    # -------------------------------------------------------------------------
    # Parsing (PDF / IFC / BIM)
    # -------------------------------------------------------------------------
    pdf_parser: str = "pymupdf"  # pymupdf | pdfplumber | unstructured
    ifc_parser: str = "ifcopenshell"
    extract_tables: bool = True
    extract_formulas: bool = True
    spatial_citations: bool = True

    # -------------------------------------------------------------------------
    # Safety / Compliance defaults
    # -------------------------------------------------------------------------
    default_code_set: str = "ACI 318"
    default_units: str = "SI"  # SI | Imperial
    concrete_strength_default_mpa: float = 30.0
    steel_yield_default_mpa: float = 420.0

    # -------------------------------------------------------------------------
    # Feature flags
    # -------------------------------------------------------------------------
    enable_document_upload: bool = True
    enable_spatial_citations: bool = True
    enable_calculation_verification: bool = True
    demo_mode: bool = False

    # -------------------------------------------------------------------------
    # Validators
    # -------------------------------------------------------------------------
    @field_validator("qdrant_distance")
    @classmethod
    def _validate_distance(cls, v: str) -> str:
        allowed = {"Cosine", "Euclid", "Dot"}
        if v not in allowed:
            raise ValueError(f"qdrant_distance must be one of {allowed}")
        return v

    @field_validator("llm_provider")
    @classmethod
    def _validate_llm_provider(cls, v: str) -> str:
        allowed = {"anthropic", "openai", "local", "mock"}
        if v not in allowed:
            raise ValueError(f"llm_provider must be one of {allowed}")
        return v

    @field_validator("embedding_provider")
    @classmethod
    def _validate_embedding_provider(cls, v: str) -> str:
        allowed = {"local", "openai", "anthropic"}
        if v not in allowed:
            raise ValueError(f"embedding_provider must be one of {allowed}")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance (singleton)."""
    return Settings()


# Backwards-compatible module-level accessor used throughout the codebase.
settings = get_settings()


__all__ = ["Settings", "get_settings", "settings"]



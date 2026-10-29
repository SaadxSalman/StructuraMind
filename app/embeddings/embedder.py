"""Embedding generation.

Supports multiple providers:

* ``local`` -- sentence-transformers (runs offline, no API key required).
* ``openai`` -- OpenAI text-embedding models.
* ``anthropic`` -- not natively an embedding provider; falls back to local.

A deterministic mock embedding is used when no model can be loaded, so the
application remains functional in fully offline/demo mode.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from app.core.config import settings

logger = logging.getLogger("structuramind.embeddings")


class Embedder:
    """Generate embeddings for text chunks."""

    def __init__(self) -> None:
        self.provider = settings.embedding_provider
        self._model = None
        self._dimension = settings.qdrant_vector_size
        self._initialize()

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #
    def _initialize(self) -> None:
        if self.provider == "local":
            self._init_local()
        elif self.provider == "openai":
            self._init_openai()
        else:
            logger.warning(
                "Unknown embedding provider '%s'; falling back to local.", self.provider
            )
            self._init_local()

    def _init_local(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(settings.local_embedding_model)
            # Probe dimension with a single encode.
            test_vec = self._model.encode(["test"], convert_to_numpy=True)
            self._dimension = int(test_vec.shape[1])
            self.provider = "local"
            logger.info(
                "Local embedding model '%s' loaded (dim=%d)",
                settings.local_embedding_model,
                self._dimension,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not load local embedding model (%s). Using deterministic mock embeddings.", exc
            )
            self._model = None
            self.provider = "mock"

    def _init_openai(self) -> None:
        try:
            import openai  # type: ignore

            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            self._client = openai.OpenAI(api_key=settings.openai_api_key)
            self._dimension = 1536
            self.provider = "openai"
            logger.info("OpenAI embedding provider initialized")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "OpenAI embeddings unavailable (%s). Falling back to mock.", exc
            )
            self._client = None
            self.provider = "mock"

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts, returning a list of vectors."""
        if not texts:
            return []
        texts = [t or "" for t in texts]

        if self.provider == "local" and self._model is not None:
            return self._embed_local(texts)
        if self.provider == "openai" and getattr(self, "_client", None) is not None:
            return self._embed_openai(texts)
        return self._embed_mock(texts)

    def embed_text(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]

    # ------------------------------------------------------------------ #
    # Providers
    # ------------------------------------------------------------------ #
    def _embed_local(self, texts: List[str]) -> List[List[float]]:
        import math

        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=settings.embedding_batch_size,
        )
        return vectors.tolist()

    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        from openai import OpenAI  # type: ignore

        client: OpenAI = self._client  # type: ignore[assignment]
        all_vectors: List[List[float]] = []
        batch = settings.embedding_batch_size
        for i in range(0, len(texts), batch):
            chunk = texts[i : i + batch]
            resp = client.embeddings.create(
                model=settings.openai_embedding_model,
                input=chunk,
            )
            for item in resp.data:
                all_vectors.append(item.embedding)
        return all_vectors

    @staticmethod
    def _embed_mock(texts: List[str]) -> List[List[float]]:
        """Deterministic pseudo-random embeddings (offline fallback)."""
        vectors: List[List[float]] = []
        for text in texts:
            seed = sum(ord(c) for c in text) % (2 ** 31)
            rng = np.random.RandomState(seed)
            vec = rng.randn(settings.qdrant_vector_size).astype(np.float64)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec.tolist())
        return vectors

"""Base agent class with LLM client setup.

Supports Anthropic, OpenAI, a local deterministic stub, and a mock mode.
When no API key is configured the agent falls back to mock responses so the
application remains fully functional without any external LLM credentials.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.core.config import settings

logger = logging.getLogger("structuramind.agents")


class BaseAgent:
    """Base class for StructuraMind agents."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.llm_provider = settings.llm_provider
        self._client = None
        self._model = None
        self._initialize_client()

    # ------------------------------------------------------------------ #
    # Client initialization
    # ------------------------------------------------------------------ #
    def _initialize_client(self) -> None:
        provider = self.llm_provider
        if provider == "anthropic":
            self._init_anthropic()
        elif provider == "openai":
            self._init_openai()
        else:
            # local / mock: deterministic stub, no external dependency.
            self._client = None
            self._model = None
            logger.info(
                "Agent '%s' using %s mode (no external LLM client)", self.name, provider
            )

    def _init_anthropic(self) -> None:
        if not settings.anthropic_api_key or settings.anthropic_api_key.startswith("sk-ant-your"):
            logger.warning(
                "Agent '%s': ANTHROPIC_API_KEY not configured; using mock mode.", self.name
            )
            self._client = None
            self._model = settings.anthropic_model
            return
        try:
            import anthropic  # type: ignore

            self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            self._model = settings.anthropic_model
            logger.info(
                "Agent '%s': Anthropic client initialized (%s)", self.name, self._model
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Agent '%s': Anthropic unavailable (%s); using mock mode.", self.name, exc
            )
            self._client = None
            self._model = settings.anthropic_model

    def _init_openai(self) -> None:
        if not settings.openai_api_key or settings.openai_api_key.startswith("sk-your"):
            logger.warning(
                "Agent '%s': OPENAI_API_KEY not configured; using mock mode.", self.name
            )
            self._client = None
            self._model = settings.openai_model
            return
        try:
            import openai  # type: ignore

            self._client = openai.OpenAI(api_key=settings.openai_api_key)
            self._model = settings.openai_model
            logger.info(
                "Agent '%s': OpenAI client initialized (%s)", self.name, self._model
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Agent '%s': OpenAI unavailable (%s); using mock mode.", self.name, exc
            )
            self._client = None
            self._model = settings.openai_model

    # ------------------------------------------------------------------ #
    # LLM call
    # ------------------------------------------------------------------ #
    def call_llm(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        """Call the configured LLM, falling back to a deterministic mock."""
        if self._client is not None:
            try:
                return self._call_provider(system_prompt, user_prompt, temperature)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Agent '%s': LLM call failed (%s); using mock response.", self.name, exc
                )
        return self._mock_response(system_prompt, user_prompt)

    def _call_provider(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        if self.llm_provider == "anthropic" and self._client is not None:
            import anthropic  # type: ignore

            client: anthropic.Anthropic = self._client  # type: ignore[assignment]
            msg = client.messages.create(
                model=self._model or settings.anthropic_model,
                max_tokens=1024,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return "".join(block.text for block in msg.content if hasattr(block, "text"))

        if self.llm_provider == "openai" and self._client is not None:
            import openai  # type: ignore

            client: openai.OpenAI = self._client  # type: ignore[assignment]
            resp = client.chat.completions.create(
                model=self._model or settings.openai_model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return resp.choices[0].message.content or ""

        raise RuntimeError(
            f"No LLM client available for provider '{self.llm_provider}'"
        )

    # ------------------------------------------------------------------ #
    # Mock fallback
    # ------------------------------------------------------------------ #
    def _mock_response(self, system_prompt: str, user_prompt: str) -> str:
        """Deterministic, offline mock response so the app works without keys."""
        agent = self.name.lower()
        if "retrieval" in agent:
            return (
                "[MOCK RETRIEVAL] Relevant code clauses identified from the uploaded documents. "
                "Citations: ACI 318-19 Section 22.4 (flexural strength), Section 22.5 (shear). "
                "No LLM API key configured; set ANTHROPIC_API_KEY or OPENAI_API_KEY for real generation."
            )
        if "calculation" in agent:
            return (
                "[MOCK CALCULATION] Safety-factor check performed against default material values "
                f"(fc'={settings.concrete_strength_default_mpa} MPa, fy={settings.steel_yield_default_mpa} MPa). "
                "Result: PASS (demonstration). Configure an LLM key for verified calculations."
            )
        if "critic" in agent:
            return (
                "[MOCK CRITIC] Answer appears consistent with retrieved source clauses. "
                "No hallucinations detected (demo mode). Set an LLM key for rigorous verification."
            )
        return (
            "[MOCK] StructuraMind agent response (demo mode). Configure an LLM API key "
            "for production use."
        )


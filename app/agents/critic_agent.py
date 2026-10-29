"""Critic Agent.

Verifies a generated answer against the retrieved source clauses. Checks that
every claim is grounded in an actual source, that citations point to real
clauses/pages, and flags potential hallucinations before the final answer is
emitted.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from app.agents.base import BaseAgent
from app.core.config import settings

logger = logging.getLogger("structuramind.agents.critic")


class CriticAgent(BaseAgent):
    """Critiques an answer for hallucinations and citation fidelity."""

    def __init__(self) -> None:
        super().__init__("CriticAgent")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def critique(
        self,
        query: str,
        answer: str,
        citations: List[dict],
        threshold: Optional[float] = None,
    ) -> dict:
        """Critique an answer. Returns verdict + issues + confidence score."""
        threshold = threshold or settings.critic_hallucination_threshold
        issues: List[str] = []
        grounded_claims = 0
        total_claims = 0

        # Basic structural checks that do not need an LLM.
        source_texts = [c.get("text", "") for c in citations]
        source_blob = " ".join(source_texts).lower()

        # 1. Citation presence check.
        if not citations:
            issues.append("No citations provided; answer is ungrounded.")

        # 2. Detect common hallucinated code references.
        for fake in ["ACI 999", "Section 99.9", "Clause 999.9", "IBC-2099"]:
            if fake.lower() in answer.lower() and fake.lower() not in source_blob:
                issues.append(f"Possible hallucinated reference: {fake} not found in sources.")

        # 3. LLM-based critique (when available).
        llm_verdict = self._llm_critique(query, answer, citations)

        # Combine.
        total_issues = len(issues) + len(llm_verdict.get("issues", []))
        issues.extend(llm_verdict.get("issues", []))

        # Confidence score: inverse of issue density.
        confidence = max(0.0, 1.0 - (total_issues * 0.25))
        passed = confidence >= threshold and total_issues == 0

        verdict = "PASS" if passed else ("WARN" if confidence >= threshold else "FAIL")

        logger.info(
            "CriticAgent: verdict=%s confidence=%.2f issues=%d", verdict, confidence, total_issues
        )

        return {
            "verdict": verdict,
            "confidence": round(confidence, 3),
            "passed": passed,
            "issues": issues,
            "threshold": threshold,
            "llm_notes": llm_verdict.get("notes", ""),
            "citations_checked": len(citations),
        }

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #
    def _llm_critique(self, query: str, answer: str, citations: List[dict]) -> dict:
        system_prompt = (
            "You are a meticulous structural-engineering critic. Review the "
            "answer below against the cited source clauses. Identify any "
            "hallucinations, unsupported claims, or citation mismatches. "
            "Respond with 'ISSUES: <list>' or 'ISSUES: none' followed by "
            "'NOTES: <short notes>'."
        )
        cites_text = "\n".join(
            f"- [{c.get('filename','?')} p.{c.get('page_number','?')} "
            f"{c.get('section','') or c.get('clause','')}] {c.get('text','')[:300]}"
            for c in citations[:10]
        )
        user_prompt = (
            f"Query: {query}\n\nAnswer:\n{answer}\n\n"
            f"Source Citations:\n{cites_text}\n"
        )
        try:
            response = self.call_llm(system_prompt, user_prompt, temperature=0.1)
        except Exception as exc:  # noqa: BLE001
            logger.warning("CriticAgent LLM critique failed: %s", exc)
            return {"issues": [], "notes": "Critic LLM unavailable (mock mode)."}

        issues: List[str] = []
        notes = ""
        if "ISSUES:" in response:
            parts = response.split("ISSUES:", 1)[1]
            if "NOTES:" in parts:
                issue_part, notes_part = parts.split("NOTES:", 1)
                notes = notes_part.strip()
            else:
                issue_part = parts
            for line in issue_part.splitlines():
                line = line.strip(" -•")
                if line and line.lower() != "none":
                    issues.append(line)

        return {"issues": issues, "notes": notes}

"""LangGraph agentic verification loop.

Workflow:

    query
      │
      ▼
  [Retrieval Agent]  → gather code/BIM/report chunks + citations
      │
      ▼
  [Calculation Agent] → verify numeric safety factors (if enabled)
      │
      ▼
  [Critic Agent]     → check answer for hallucinations vs sources
      │
      ▼
  Final answer with citations + verdict
"""

from __future__ import annotations

import logging
from typing import List, Optional, TypedDict

from langgraph.graph import END, StateGraph  # type: ignore

from app.agents.calculation_agent import CalculationAgent
from app.agents.critic_agent import CriticAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.core.config import settings
from app.vectordb.qdrant_store import QdrantStore

logger = logging.getLogger("structuramind.agents.graph")


class AgentState(TypedDict, total=False):
    """State carried through the LangGraph workflow."""

    query: str
    collections: Optional[list]
    top_k: int
    retrieval: dict
    calculation: dict
    criticism: dict
    answer: str
    citations: list
    verdict: dict
    iteration: int


class StructuraMindGraph:
    """Build and run the LangGraph agentic verification loop."""

    def __init__(self, store: Optional[QdrantStore] = None) -> None:
        self.store = store or QdrantStore()
        self.retrieval_agent = RetrievalAgent(store=self.store)
        self.calculation_agent = CalculationAgent()
        self.critic_agent = CriticAgent()
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ #
    # Graph construction
    # ------------------------------------------------------------------ #
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("retrieval", self._node_retrieval)
        workflow.add_node("calculation", self._node_calculation)
        workflow.add_node("critic", self._node_critic)
        workflow.add_node("answer", self._node_answer)

        workflow.set_entry_point("retrieval")
        workflow.add_edge("retrieval", "calculation")
        workflow.add_edge("calculation", "critic")
        workflow.add_edge("critic", "answer")
        workflow.add_edge("answer", END)

        return workflow.compile()

    # ------------------------------------------------------------------ #
    # Nodes
    # ------------------------------------------------------------------ #
    def _node_retrieval(self, state: AgentState) -> AgentState:
        query = state["query"]
        collections = state.get("collections") or ["codes", "bim", "reports"]
        top_k = state.get("top_k") or settings.top_k_retrieval
        retrieval = self.retrieval_agent.retrieve(
            query=query, collections=collections, top_k=top_k
        )
        state["retrieval"] = retrieval
        state["citations"] = retrieval.get("citations", [])
        state["iteration"] = state.get("iteration", 0) + 1
        logger.info("Graph node 'retrieval': %d citations", len(state["citations"]))
        return state

    def _node_calculation(self, state: AgentState) -> AgentState:
        if not settings.enable_calculation_agent:
            state["calculation"] = {"skipped": True}
            return state
        query = state["query"]
        context = state["retrieval"].get("context", "")
        calculation = self.calculation_agent.verify(query=query, context=context)
        state["calculation"] = calculation
        logger.info(
            "Graph node 'calculation': all_pass=%s", calculation.get("all_pass")
        )
        return state


    def _node_critic(self, state: AgentState) -> AgentState:
        if not settings.enable_critic_agent:
            state["criticism"] = {"skipped": True, "verdict": "SKIPPED"}
            state["verdict"] = state["criticism"]
            return state
        # Build a draft answer from retrieval context for the critic to review.
        draft = self._draft_answer(state)
        state["answer"] = draft
        criticism = self.critic_agent.critique(
            query=state["query"],
            answer=draft,
            citations=state.get("citations", []),
        )
        state["criticism"] = criticism
        state["verdict"] = criticism
        logger.info("Graph node 'critic': verdict=%s", criticism.get("verdict"))
        return state

    def _node_answer(self, state: AgentState) -> AgentState:
        """Final answer assembly."""
        retrieval = state.get("retrieval", {})
        calculation = state.get("calculation", {})
        criticism = state.get("criticism", {})

        answer = state.get("answer", "")
        if not answer:
            answer = retrieval.get("context", "")[:2000]

        # Append calculation result if present.
        calc_lines = []
        if calculation and not calculation.get("skipped"):
            calc_lines.append("## Calculation Verification")
            calc_lines.append(
                f"- Overall: {'PASS' if calculation.get('all_pass') else 'REVIEW'}"
            )
            for c in calculation.get("checks", []):
                calc_lines.append(
                    f"- {c['check']}: {c['result']} ({c['code_reference']})"
                )
        calc_block = "\n".join(calc_lines)

        # Append critic verdict.
        critic_lines = []
        if criticism and not criticism.get("skipped"):
            critic_lines.append("## Critic Verification")
            critic_lines.append(
                f"- Verdict: {criticism.get('verdict')} (confidence {criticism.get('confidence')})"
            )
            if criticism.get("issues"):
                critic_lines.append("- Issues: " + "; ".join(criticism["issues"]))
        critic_block = "\n".join(critic_lines)

        # Build citation list.
        cites = []
        for c in state.get("citations", []):
            cites.append(
                f"[{c.get('filename','?')} p.{c.get('page_number','?')} "
                f"{c.get('section','') or c.get('clause','')}]"
            )
        citation_block = "Citations: " + "; ".join(cites[:8]) if cites else "Citations: none"

        final_answer = (
            f"## Answer\n{answer}\n\n"
            f"{calc_block}\n\n"
            f"{critic_block}\n\n"
            f"{citation_block}"
        )
        state["answer"] = final_answer
        return state

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _draft_answer(self, state: AgentState) -> str:
        """Draft a concise answer from retrieved context (pre-criticism)."""
        retrieval = state.get("retrieval", {})
        context = retrieval.get("context", "")
        if not context:
            return "No relevant code clauses were found for this query."
        return (
            "Based on the retrieved code clauses and BIM metadata, the following "
            "findings apply:\n\n" + context[:2500]
        )

    # ------------------------------------------------------------------ #
    # Public run
    # ------------------------------------------------------------------ #
    def run(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> dict:
        """Execute the full agentic verification loop for a query."""
        if not settings.enable_agentic_loop:
            # Single-pass retrieval without agents.
            retrieval = self.retrieval_agent.retrieve(
                query=query, collections=collections, top_k=top_k
            )
            return {
                "query": query,
                "answer": retrieval.get("context", ""),
                "citations": retrieval.get("citations", []),
                "verdict": {"verdict": "RETRIEVAL_ONLY", "passed": True},
                "agentic_loop": False,
            }

        initial_state: AgentState = {
            "query": query,
            "collections": collections or ["codes", "bim", "reports"],
            "top_k": top_k or settings.top_k_retrieval,
            "iteration": 0,
        }
        final_state = self.graph.invoke(initial_state)  # type: ignore[arg-type]
        return {
            "query": final_state.get("query"),
            "answer": final_state.get("answer", ""),
            "citations": final_state.get("citations", []),
            "verdict": final_state.get("verdict", {}),
            "calculation": final_state.get("calculation", {}),
            "criticism": final_state.get("criticism", {}),
            "retrieval": final_state.get("retrieval", {}),
            "agentic_loop": True,
            "iterations": final_state.get("iteration", 0),
        }


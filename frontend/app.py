"""StructuraMind Streamlit frontend — engineering workspace dashboard.

Connects to the FastAPI backend at BACKEND_URL (from .env). Provides:

* Document upload (PDF building codes, IFC/BIM models)
* Code-compliance chat with the Retrieval → Calculation → Critic agent loop
* Interactive spatial citations (page, section, clause, coordinates)
* Calculation & critic verification results
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

# Ensure the repo root is importable so `app.core.config` resolves when
# Streamlit runs frontend/app.py directly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import requests
import streamlit as st

from app.core.config import settings  # noqa: F401  (loads .env when run as package)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))

REQUEST_TIMEOUT = 60


# --------------------------------------------------------------------------- #
# Page config
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="StructuraMind",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []  # type: ignore[attr-defined]
if "documents" not in st.session_state:
    st.session_state.documents = []  # type: ignore[attr-defined]
if "health" not in st.session_state:
    st.session_state.health = None  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# API helpers
# --------------------------------------------------------------------------- #
def api_get(path: str) -> Any:
    try:
        resp = requests.get(f"{BACKEND_URL}{path}", timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"Backend request failed ({path}): {exc}")
        return None


def api_post(path: str, json: Dict = None, files=None, data=None) -> Any:
    try:
        resp = requests.post(
            f"{BACKEND_URL}{path}",
            json=json,
            files=files,
            data=data,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"Backend request failed ({path}): {exc}")
        return None


def fetch_health() -> Dict[str, Any]:
    return api_get("/health") or {}


def fetch_documents(collection: str = "codes") -> List[Dict[str, Any]]:
    data = api_get(f"/documents?collection={collection}")
    return data or []

# --------------------------------------------------------------------------- #
# UI: header
# --------------------------------------------------------------------------- #
def render_header() -> None:
    st.title("🏗️ StructuraMind")
    st.caption(
        "Multi-Modal RAG & Code Compliance Engine for Civil Engineering · "
        "ACI 318 · AISC · IBC · Eurocodes · BIM/IFC"
    )
    st.divider()


# --------------------------------------------------------------------------- #
# UI: sidebar
# --------------------------------------------------------------------------- #
def render_sidebar() -> None:
    with st.sidebar:
        st.header("⚙️ Workspace")

        # Health status.
        health = fetch_health()
        st.session_state.health = health
        if health:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Vector DB", health.get("vector_db", "?"))
                st.metric("Embeddings", health.get("embedding_provider", "?"))
            with col2:
                st.metric("LLM", health.get("llm_provider", "?"))
                st.metric("Collections", len(health.get("collections", [])))
            if health.get("demo_mode"):
                st.info("Demo mode: no LLM API key configured.")

        st.divider()

        # Document upload.
        st.subheader("📄 Ingest Documents")
        uploaded = st.file_uploader(
            "Upload PDF (code) or IFC (BIM)",
            type=["pdf", "ifc"],
            accept_multiple_files=False,
        )
        if uploaded is not None:
            if st.button("Ingest to backend", use_container_width=True):
                with st.spinner("Parsing & embedding..."):
                    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                    result = api_post("/ingest/pdf" if uploaded.name.lower().endswith(".pdf") else "/ingest/ifc", files=files)
                    if result:
                        st.success(f"Ingested '{result.get('filename')}' — {result.get('chunk_count')} chunks")
                        st.session_state.documents = fetch_documents()

        st.divider()

        # Stored documents.
        st.subheader("📚 Ingested Documents")
        if st.button("Refresh document list", use_container_width=True):
            st.session_state.documents = fetch_documents()
        docs = st.session_state.documents
        if docs:
            for d in docs[:20]:
                st.write(f"• {d.get('filename')} ({d.get('chunk_count')} chunks)")
            if len(docs) > 20:
                st.caption(f"... and {len(docs) - 20} more")
        else:
            st.caption("No documents ingested yet.")

        st.divider()

        # Query settings.
        st.subheader("🔍 Query Settings")
        st.session_state.collections = st.multiselect(
            "Collections",
            options=["codes", "bim", "reports"],
            default=["codes", "bim", "reports"],
        )
        st.session_state.top_k = st.slider("Top-K retrieval", 1, 20, 10)
        st.session_state.calc_fc = st.number_input(
            "fc' concrete (MPa)", value=float(settings.concrete_strength_default_mpa)
        )
        st.session_state.calc_fy = st.number_input(
            "fy steel (MPa)", value=float(settings.steel_yield_default_mpa)
        )

        st.divider()
        st.markdown("---")
        st.markdown(
            "StructuraMind v1.0.0 · [GitHub](https://github.com/saadxsalman/structuramind-ai)"
        )


# --------------------------------------------------------------------------- #
# UI: citation card
# --------------------------------------------------------------------------- #
def render_citation(c: Dict[str, Any]) -> None:
    filename = c.get("filename") or "?"
    page = c.get("page_number")
    section = c.get("section") or c.get("clause") or ""
    title = c.get("title") or ""
    score = c.get("score", 0.0)
    source_type = c.get("source_type") or ""
    bbox = c.get("bbox")
    text = c.get("text", "")

    label_parts = [filename]
    if page is not None:
        label_parts.append(f"p.{page}")
    if section:
        label_parts.append(section)
    if title:
        label_parts.append(title)
    label = " | ".join(label_parts)

    with st.expander(f"📎 {label}  ·  score {score:.2f}  ·  {source_type}"):
        st.write(text[:1200] + ("…" if len(text) > 1200 else ""))
        meta = []
        if page is not None:
            meta.append(f"Page: {page}")
        if bbox:
            meta.append(f"Bbox: [{', '.join(f'{v:.1f}' for v in bbox)}]")
        if source_type:
            meta.append(f"Source: {source_type}")
        if meta:
            st.caption(" · ".join(meta))


# --------------------------------------------------------------------------- #
# UI: chat / query
# --------------------------------------------------------------------------- #
def render_chat() -> None:
    st.subheader("💬 Code-Compliance Query")

    # Show conversation history.
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander("Citations", expanded=False):
                    for c in msg["citations"]:
                        render_citation(c)
            if msg.get("calculation"):
                st.markdown("### Calculation Verification")
                calc = msg["calculation"]
                st.write(f"Overall: {'✅ PASS' if calc.get('all_pass') else '⚠️ REVIEW'}")
                for ck in calc.get("checks", []):
                    st.write(
                        f"- {ck['check']} ({ck['code_reference']}): "
                        f"{ck['result']} — {ck['details'][:200]}"
                    )
            if msg.get("criticism"):
                st.markdown("### Critic Verification")
                crit = msg["criticism"]
                st.write(f"Verdict: {crit.get('verdict')} (confidence {crit.get('confidence')})")
                if crit.get("issues"):
                    st.write("Issues: " + "; ".join(crit["issues"]))

    # Query input.
    user_query = st.chat_input(
        "Ask a code-compliance question (e.g. 'Check flexural capacity per ACI 318...')"
    )
    if user_query:
        # Append user message.
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Call backend.
        with st.chat_message("assistant"):
            with st.spinner("Running Retrieval → Calculation → Critic loop..."):
                payload = {
                    "query": user_query,
                    "collections": st.session_state.get("collections", ["codes", "bim", "reports"]),
                    "top_k": st.session_state.get("top_k", 10),
                    "concrete_strength_mpa": st.session_state.get("calc_fc"),
                    "steel_yield_mpa": st.session_state.get("calc_fy"),
                }
                result = api_post("/query/", json=payload)

            if result:
                answer = result.get("answer", "")
                citations = result.get("citations", [])
                calculation = result.get("calculation")
                criticism = result.get("criticism")

                st.markdown(answer)

                if citations:
                    with st.expander("Citations", expanded=True):
                        for c in citations:
                            render_citation(c)

                if calculation:
                    st.markdown("### Calculation Verification")
                    st.write(
                        f"Overall: {'✅ PASS' if calculation.get('all_pass') else '⚠️ REVIEW'}"
                    )
                    for ck in calculation.get("checks", []):
                        st.write(
                            f"- {ck['check']} ({ck['code_reference']}): "
                            f"{ck['result']} — {ck['details'][:200]}"
                        )

                if criticism:
                    st.markdown("### Critic Verification")
                    st.write(
                        f"Verdict: {criticism.get('verdict')} "
                        f"(confidence {criticism.get('confidence')})"
                    )
                    if criticism.get("issues"):
                        st.write("Issues: " + "; ".join(criticism["issues"]))

                # Persist assistant message.
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                        "calculation": calculation,
                        "criticism": criticism,
                    }
                )
            else:
                st.error("Failed to get a response from the backend.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    render_header()
    render_sidebar()

    col_chat, col_docs = st.columns([3, 2])
    with col_chat:
        render_chat()
    with col_docs:
        st.subheader("📊 Document Index")
        docs = fetch_documents()
        if docs:
            for d in docs:
                st.write(f"• {d.get('filename')} — {d.get('chunk_count')} chunks")
        else:
            st.caption("No documents. Upload a PDF or IFC file from the sidebar.")


if __name__ == "__main__":
    main()


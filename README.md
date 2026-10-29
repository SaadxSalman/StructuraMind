# 🏗️ StructuraMind

> **Multi-Modal RAG & Code Compliance Engine for Civil Engineering**

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B.svg)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-2ea44f)](https://langchain-ai.github.io/langgraph/)
[![Qdrant](https://img.shields.io/badge/Qdrant-VectorDB-007ACC)](https://qdrant.tech/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Repository:** `saadxsalman/structuramind-ai`  
**Maintainer:** Saad Salman

---

## 📖 GitHub Description

> StructuraMind: An enterprise-grade, multi-modal RAG and Agentic AI engine for civil and structural engineering. Automatically parses building codes, geotechnical logs, and BIM/IFC models to perform real-time code compliance checks, structural calculations, and design queries with grounded citations.

---

## 🗂️ Table of Contents

1. [Overview](#-overview)
2. [Problem Statement](#-problem-statement)
3. [Key Features](#-key-features)
4. [System Architecture](#-system-architecture)
5. [Tech Stack](#-tech-stack)
6. [Directory Structure](#-directory-structure)
7. [Installation](#-installation)
8. [Configuration (`.env`)](#-configuration-env)
9. [Usage](#-usage)
10. [API Reference](#-api-reference)
11. [Agentic Verification Loop](#-agentic-verification-loop)
12. [Multi-Modal Ingestion Pipeline](#-multi-modal-ingestion-pipeline)
13. [Hierarchical & Semantic Chunking](#-hierarchical--semantic-chunking)
14. [Spatial Citations](#-spatial-citations)
15. [Code Compliance Calculations](#-code-compliance-calculations)
16. [Demo Mode vs Production](#-demo-mode-vs-production)
17. [Environment Variable Reference](#-environment-variable-reference)
18. [Security](#-security)
19. [Testing](#-testing)
20. [Roadmap](#-roadmap)
21. [Contributing](#-contributing)
22. [License](#-license)
23. [Acknowledgements](#-acknowledgements)

---

## 🎯 Overview

**StructuraMind** is a production-grade, next-level Generative AI Retrieval-Augmented Generation (RAG) platform designed specifically for structural and civil engineers.

It solves one of the industry's most tedious and error-prone bottlenecks: **cross-referencing massive, frequently updated building codes** (e.g., ACI 318, AISC Steel Construction Manual, IBC, Eurocodes) **with complex geotechnical reports and Building Information Modeling (BIM/IFC) metadata** to automate design compliance checks and engineering queries.

Instead of an engineer manually flipping through hundreds of pages of code to find the right clause, StructuraMind:

1. **Ingests** building-code PDFs, geotechnical reports, and IFC/BIM models.
2. **Chunks** them hierarchically (Chapter → Section → Clause) so critical context is never lost.
3. **Retrieves** the most relevant code snippets using hybrid semantic + keyword search.
4. **Verifies** numeric safety factors with a dedicated Calculation Agent.
5. **Critiques** the answer for hallucinations against exact source clauses before emitting it.
6. **Returns** the answer paired with exact page numbers, section numbers, and highlight coordinates from the source documents.

The result is a grounded, auditable, agentic AI assistant that behaves like a meticulous senior structural engineer who always cites their sources.

## 🩺 Problem Statement

Civil and structural engineering design is governed by prescriptive codes that are:

* **Voluminous:** A single code (e.g., ACI 318) spans hundreds of pages with nested clauses, tables, and formulas.
* **Frequently updated:** Code editions change (ACI 318-14 → 318-19 → 318-25), and using the wrong clause can lead to unsafe or non-compliant designs.
* **Deeply hierarchical:** A requirement in "Section 22.4.2.1" only makes sense when read with its parent sections and exceptions.
* **Multi-modal:** Compliance depends not just on text, but on BIM geometry (member sizes, materials, levels), reinforcement drawings, and geotechnical borehole logs.

Engineers today spend a disproportionate amount of time **searching and cross-referencing** rather than **designing and verifying**. Errors in clause lookup or missed exceptions (e.g., seismic load combinations, slab thickness minimums) are costly and potentially unsafe.

StructuraMind automates this knowledge retrieval and compliance-verification loop, surfacing the exact clause, page, and calculation behind every answer.

---

## ✨ Key Features

### 1. Multi-Modal Ingestion Pipeline
Beyond plain text, StructuraMind processes:
* **Building-code PDFs** (ACI, AISC, IBC, Eurocodes, local codes) with complex tables and formulas.
* **Vector-based structural drawings** and CAD annotations.
* **IFC (Industry Foundation Classes) BIM models** — extracting spatial and material metadata (walls, beams, columns, slabs, footings, piles, materials, property sets).
* **Geotechnical / project reports** as plain text.

### 2. Hierarchical & Semantic Chunking
Civil codes are deeply nested (Chapter → Section → Clause). StructuraMind uses structural-aware parsing to retain parent–child clause hierarchies, so the retriever never loses critical contextual limits — for example, seismic load exceptions that apply only within a specific section.

### 3. Agentic Verification Loop (Self-Correction)
A multi-agent architecture built with **LangGraph**:

| Agent | Role |
|-------|------|
| **Retrieval Agent** | Gathers relevant code snippets, BIM metadata, and report passages from the vector store. |
| **Calculation Agent** | Extracts numeric inputs and verifies safety factors (flexural, shear, axial) against code equations. |
| **Critic Agent** | Checks the draft answer for hallucinations and citation mismatches against the exact source clauses. |

The loop iterates until the answer is grounded, cited, and verified — or flags it for engineer review.

### 4. Interactive Spatial Citations
Every answer is paired with:
* Exact **page numbers** and **section/clause numbers**.
* **Highlight coordinates** (bounding boxes) from uploaded blueprints or code manuals.
* Source document filename and similarity score.

Engineers can click a citation and jump straight to the relevant clause in the source.

### 5. Production-Grade Backend + Engineering Dashboard
* **FastAPI** REST API with automatic OpenAPI docs (`/docs`).
* **Streamlit** frontend workspace for document upload, compliance chat, and citation review.
* **Qdrant** vector database with hybrid keyword + vector search (in-memory fallback for offline/demo use).
* **Docker Compose** for one-command deployment.


## 🏗️ System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                         STREAMLIT FRONTEND (Port 8501)                  │
│  Document Upload  •  Compliance Chat  •  Citations  •  Calculations     │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │ HTTP/JSON
┌──────────────────────────────▼──────────────────────────────────────────┐
│                         FASTAPI BACKEND (Port 8000)                     │
│  POST /ingest/{pdf,ifc,text}   POST /query/      GET /documents          │
│  GET /health                                                 GET /       │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
     ┌─────────────────┐             ┌──────────────────┐
     │  LangGraph      │             │   Qdrant Store   │
     │  Agent Loop     │             │  (Vector DB)     │
     │                 │             │  • codes         │
     │  Retrieval      │             │  • bim           │
     │  Calculation    │             │  • reports       │
     │  Critic         │             └──────────────────┘
     └─────────┬───────┘                   ▲
               │                           │ embeddings
               ▼                           │
     ┌─────────────────┐                   │
     │  Embedder       │───────────────────┘
     │  (sentence-     │
     │  transformers)  │
     └─────────┬───────┘
               │
      ┌────────┴────────┐
      ▼                 ▼
 ┌─────────┐     ┌─────────────┐
 │  PDF    │     │    IFC      │
 │ Parser  │     │   Parser    │
 │(PyMuPDF)│     │ (IfcOpenShell)│
 └─────────┘     └─────────────┘
```

**Data flow:**

1. **Ingest:** PDF → PyMuPDF hierarchical parser → chunks → Embedder → Qdrant.  
   IFC → IfcOpenShell → element records → chunks → Embedder → Qdrant.
2. **Query:** User question → LangGraph Retrieval Agent (semantic search in Qdrant) → optional Calculation Agent (numeric safety-factor checks) → Critic Agent (hallucination/citation check) → Final answer with citations.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Orchestration & Agents | **LangGraph** / LangChain | Multi-step reasoning, agent loops, stateful workflows |
| LLM Backend | **Anthropic Claude 3.5 Sonnet** / **OpenAI GPT-4o** / local mock | Advanced reasoning, markdown/table generation |
| Vector Database | **Qdrant** | High-speed hybrid keyword + vector similarity search |
| Embeddings | **sentence-transformers** (`all-MiniLM-L6-v2`) / OpenAI | Dense vector encoding of code clauses |
| PDF Parsing | **PyMuPDF (fitz)** / pdfplumber | Text, tables, and block-level coordinates from code PDFs |
| BIM / CAD Parsing | **IfcOpenShell** | Spatial + material metadata from IFC models |
| Backend API | **FastAPI** + Pydantic | REST API, data validation, automatic OpenAPI docs |
| Frontend UI | **Streamlit** | Engineering workspace dashboard |
| Containerization | **Docker** + **Docker Compose** | Reproducible deployment of backend + Qdrant |
| Configuration | **pydantic-settings** + `.env` | Centralized, gitignored secret & config management |

---

## 📂 Directory Structure

```text
structuramind-ai/
├── .env                          # ALL secrets + config (GITIGNORED — never commit)
├── .gitignore                    # Excludes .env, __pycache__, venv, IDE files
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── Dockerfile                    # FastAPI backend image
├── docker-compose.yml            # Qdrant + backend + frontend services
│
├── app/                          # FastAPI backend package
│   ├── __init__.py
│   ├── main.py                   # App factory, routers, lifespan
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py             # Pydantic settings from .env
│   │   ├── database.py           # Qdrant + in-memory vector store
│   │   └── logging.py            # Structured logging
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pdf_parser.py         # Hierarchical PDF code parser (PyMuPDF)
│   │   ├── ifc_parser.py         # IFC/BIM metadata parser (IfcOpenShell)
│   │   └── chunker.py            # ChunkRecord assembly for codes / bim / reports
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedder.py           # Local / OpenAI / mock embeddings
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py               # BaseAgent + LLM client + mock fallback
│   │   ├── retrieval_agent.py    # Retrieval agent (semantic search + citations)
│   │   ├── calculation_agent.py  # Calculation agent (ACI safety-factor checks)
│   │   ├── critic_agent.py       # Critic agent (hallucination / citation check)
│   │   └── graph.py              # LangGraph StateGraph (Retrieval→Calc→Critic)
│   └── api/
│       ├── __init__.py
│       ├── models.py             # Pydantic request/response models
│       ├── ingest.py             # POST /ingest/{pdf,ifc,text}
│       ├── query.py              # POST /query/ (agentic loop)
│       └── documents.py          # GET /documents, GET /health
│
└── frontend/                     # Streamlit frontend
    ├── __init__.py
    └── app.py                    # Document upload, compliance chat, citations
```


## 🚀 Installation

### Prerequisites

* **Python 3.11+** (3.10 may work; 3.11+ recommended)
* **pip** and **venv**
* **Docker Desktop** (optional, only if you want to run Qdrant as a container)
* At least **4 GB RAM** recommended (for local embedding models)

### 1. Clone the repository

```bash
git clone https://github.com/saadxsalman/structuramind-ai.git
cd structuramind-ai
```

### 2. Create and activate a virtual environment

```bash
# PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on `ifcopenshell`:** IfcOpenShell is platform-specific and ships pre-built wheels for many (but not all) platforms. If `pip install ifcopenshell` fails, StructuraMind degrades gracefully — IFC parsing returns a stub and the rest of the application keeps working. Install IfcOpenShell manually for full BIM support.

### 4. Configure the `.env` file

A `.env` file is already present in the repository root (it is **gitignored** and will never be pushed to GitHub). Open it and fill in at least one LLM API key:

```ini
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-your-real-key
# or
# LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-your-real-key
```

If you leave all LLM keys blank (or use the placeholder values), StructuraMind runs in **demo/mock mode** — fully functional offline, but with deterministic placeholder responses instead of real LLM generation.

### 5. (Optional) Start Qdrant with Docker

```bash
docker compose up -d qdrant
```

Qdrant will be available at `http://localhost:6333`. If you do not start Qdrant, StructuraMind automatically falls back to an **in-memory vector store** so the app still works for demos and development (data is not persisted across restarts).


## ⚙️ Configuration (`.env`)

All important keys and settings live in **one** `.env` file at the repository root. This file is **gitignored** (see `.gitignore`) and must **never** be committed to GitHub. For production, inject these values via your secret manager (AWS Secrets Manager, GitHub Secrets, Azure Key Vault, GCP Secret Manager, HashiCorp Vault) instead of committing the file.

See [Environment Variable Reference](#-environment-variable-reference) for the full list.

---

## 💻 Usage

### Option A: Run everything with Docker Compose (recommended)

```bash
docker compose up -d
```

This starts Qdrant, the FastAPI backend (port 8000), and the Streamlit frontend (port 8501). Open:

* Backend API docs: `http://localhost:8000/docs`
* Frontend dashboard: `http://localhost:8501`

### Option B: Run backend and frontend separately

**Terminal 1 — Backend:**

```bash
uvicorn app.main:app --reload --port 8000
```

**Terminal 2 — Frontend:**

```bash
streamlit run frontend/app.py --server.port 8501
```

Then open `http://localhost:8501`.

### Option C: Backend only (API)

```bash
uvicorn app.main:app --port 8000
```

Interact with the API at `http://localhost:8000/docs` (Swagger UI) or via `curl` / any HTTP client.

### Typical workflow in the frontend

1. **Upload documents** from the sidebar — a building-code PDF (ACI/IBC/etc.) and/or an IFC BIM model.
2. **Ask a compliance question** in the chat, e.g.:
   * *"Check the flexural capacity of a concrete beam per ACI 318. Mu = 150 kN.m, b = 300 mm, d = 500 mm, fc' = 30 MPa, fy = 420 MPa."*
   * *"What is the minimum concrete compressive strength per Section 3.1.1?"*
3. **Review the answer** with calculation verification (pass/fail), critic verdict, and spatial citations (page, section, clause, coordinates).
4. **Click a citation** to see the exact source text and where it appears in the document.


## 🔌 API Reference

Base URL: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

### `GET /`
Service info.

### `GET /health`
Health check. Returns backend, vector DB, embedding provider, LLM provider, collections, and demo mode.

**Response:**
```json
{
  "status": "ok",
  "backend": "fastapi",
  "vector_db": "memory",
  "embedding_provider": "mock",
  "llm_provider": "anthropic",
  "collections": ["structuramind_codes", "structuramind_bim", "structuramind_reports"],
  "demo_mode": false
}
```

### `POST /ingest/pdf`
Upload a building-code PDF. Parses it hierarchically and indexes it into the `codes` collection.

**Request:** `multipart/form-data` with field `file`.

**Response:**
```json
{
  "document_id": "uuid",
  "filename": "aci318.pdf",
  "collection": "codes",
  "chunk_count": 124,
  "status": "ok",
  "message": "Parsed 48 pages into 124 chunks."
}
```

### `POST /ingest/ifc`
Upload an IFC/BIM model. Extracts element metadata and indexes it into the `bim` collection.

**Request:** `multipart/form-data` with field `file`.

**Response:**
```json
{
  "document_id": "uuid",
  "filename": "structure.ifc",
  "collection": "bim",
  "chunk_count": 87,
  "status": "ok",
  "message": "Extracted 87 elements into 87 chunks."
}
```

### `POST /ingest/text`
Ingest raw text directly (useful for pasting code clauses).

**Body:**
```json
{
  "text": "ACI 318-19 Section 22.4.2: ...",
  "document_id": "manual-1",
  "collection": "codes"
}
```

### `POST /query/`
Run a code-compliance query through the Retrieval → Calculation → Critic agent loop.

**Body:**
```json
{
  "query": "Check the flexural capacity of a concrete beam per ACI 318. Mu = 150 kN.m, b = 300 mm, d = 500 mm, fc' = 30 MPa, fy = 420 MPa.",
  "collections": ["codes", "bim", "reports"],
  "top_k": 10,
  "concrete_strength_mpa": 30,
  "steel_yield_mpa": 420
}
```

**Response:**
```json
{
  "query": "...",
  "answer": "## Answer\nBased on the retrieved code clauses ...",
  "citations": [
    {
      "chunk_id": "uuid",
      "score": 0.84,
      "collection": "codes",
      "text": "The specified compressive strength of concrete, fc', shall not be less than ...",
      "filename": "aci318.pdf",
      "page_number": 1,
      "section": "3.1.1",
      "clause": "3.1.1",
      "title": "Concrete",
      "bbox": [72, 140, 500, 160],
      "source_type": "pdf"
    }
  ],
  "verdict": {"verdict": "PASS", "confidence": 1.0, "passed": true},
  "calculation": {
    "concrete_strength_mpa": 30,
    "steel_yield_mpa": 420,
    "all_pass": true,
    "checks": [
      {"check": "flexural_strength", "result": "pass", "pass": true, "code_reference": "ACI 318-19 Section 22.4 / Table 21.2.1", "details": "..."}
    ],
    "summary": "..."
  },
  "criticism": {"verdict": "PASS", "confidence": 1.0, "passed": true, "issues": []},
  "agentic_loop": true,
  "iterations": 1
}
```

### `GET /documents?collection=codes`
List ingested documents in a collection with chunk counts.

### `GET /documents/{collection}`
List ingested documents for a specific collection.


## 🔄 Agentic Verification Loop

StructuraMind's core differentiator is its **LangGraph-powered agentic verification loop**, adapted for structural engineering compliance.

### Flow

```
User Query
    │
    ▼
┌──────────────┐
│ Retrieval    │  Semantic + keyword search across codes / bim / reports
│ Agent        │  → returns ranked chunks with citations
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Calculation  │  Extracts numeric inputs; verifies φMn ≥ Mu, φVn ≥ Vu, etc.
│ Agent        │  → returns pass/fail per code equation
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Critic     │  Checks draft answer for hallucinations; verifies every
│   Agent      │  citation points to a real source clause/page
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Final Answer │  Answer + citations + calculation + critic verdict
└──────────────┘
```

### What each agent does

* **Retrieval Agent:** Converts the user's question into an embedding, searches Qdrant across the configured collections, and returns the top-K chunks with full citation metadata (page, section, clause, title, bounding box, score).

* **Calculation Agent:** Parses numeric inputs from the query and retrieved context (e.g., `Mu = 150 kN.m`, `b = 300 mm`, `fc' = 30 MPa`). Runs ACI 318-style checks:
  * Flexural strength: `φ·Mn ≥ Mu` (φ = 0.9 tension-controlled)
  * Shear strength: `φ·Vc ≥ Vu` (φ = 0.75)
  * Returns a structured pass/fail report with the governing code clause.

* **Critic Agent:** Reviews the draft answer against the retrieved source citations. Flags:
  * Missing citations (ungrounded answers).
  * Hallucinated code references (e.g., "ACI 999" not present in sources).
  * Citation mismatches.
  * Returns a verdict (`PASS` / `WARN` / `FAIL`) and a confidence score.

### Loop control (`.env`)

```ini
ENABLE_AGENTIC_LOOP=true
MAX_AGENT_ITERATIONS=3
ENABLE_RETRIEVAL_AGENT=true
ENABLE_CALCULATION_AGENT=true
ENABLE_CRITIC_AGENT=true
CRITIC_HALLUCINATION_THRESHOLD=0.75
```

Set `ENABLE_AGENTIC_LOOP=false` to run retrieval-only mode (faster, no agents).

---

## 📄 Multi-Modal Ingestion Pipeline

### PDF (building codes)

Uses **PyMuPDF (fitz)** to extract text page-by-page with block-level bounding boxes. A heading detector recognizes common civil-code patterns:

* `CHAPTER 3`, `Chapter 3`
* `3`, `3.1`, `3.1.1`, `3.1.1.2`
* `APPENDIX A`

Each detected heading becomes a structural chunk (level 0–2) with parent–child links; body text becomes level-3 chunks. The hierarchy is preserved in metadata so retrieval can return a clause together with its parent section context.

### IFC / BIM

Uses **IfcOpenShell** to open IFC files and extract structural elements:

* `IfcWall`, `IfcBeam`, `IfcColumn`, `IfcSlab`, `IfcFooting`, `IfcPile`, `IfcMember`, `IfcPlate`, `IfcStructuralMember`

For each element it captures: GUID, type, name, description, level/storey, material, property sets, and (best-effort) bounding box. When IfcOpenShell is unavailable, the parser returns a stub so the application keeps running.

### Chunking

The `chunker` module converts parsed documents and IFC models into `ChunkRecord`s — embedding-ready records with `text`, `metadata` (document_id, filename, page, chapter, section, clause, title, bbox, source_type), and an optional precomputed `embedding`.

---

## 🧩 Hierarchical & Semantic Chunking

Civil codes do not make sense as flat paragraphs. A requirement like *"In seismic design categories D, E, and F, the minimum slab thickness shall be..."* only applies within its parent section and may be overridden by an exception two clauses later.

StructuraMind's chunking preserves this:

1. **Heading detection** identifies Chapter / Section / Clause boundaries.
2. **Parent–child links** attach every body chunk to the most recent structural heading.
3. **Metadata** stores `chapter`, `section`, `clause`, `title`, `page_number`, and `level` on every chunk.
4. **Overlapping sentence-aware splits** keep long clauses readable without losing boundary context.

When a chunk is retrieved, its parent hierarchy is available in the response, so the LLM (and the engineer) always sees the clause in context.


## 📍 Spatial Citations

Every retrieved chunk carries citation metadata:

| Field | Meaning |
|-------|---------|
| `filename` | Source document name |
| `page_number` | PDF page (1-based) |
| `chapter` / `section` / `clause` | Code hierarchy |
| `title` | Heading text |
| `bbox` | `[x0, y0, x1, y1]` highlight coordinates on the page |
| `source_type` | `pdf`, `ifc`, or `text` |
| `score` | Similarity score (0–1) |

The Streamlit frontend renders each citation as an expandable card showing the source text, page, section, and bounding box — so an engineer can verify the answer against the original document in seconds.

---

## ⚖️ Code Compliance Calculations

The Calculation Agent implements ACI 318-style checks in SI units (configurable to Imperial):

### Flexural strength (`φ·Mn ≥ Mu`)
* Reference: ACI 318-19 Section 22.4 / Table 21.2.1 (φ = 0.9 tension-controlled)
* Extracts `Mu` (factored moment), `b` (width), `d` (effective depth), `fc'`, `fy`.
* Computes required steel area and capacity ratio.

### Shear strength (`φ·Vc ≥ Vu`)
* Reference: ACI 318-19 Section 22.5 / Table 21.2.1 (φ = 0.75)
* Concrete shear capacity: `Vc = 0.17 · λ · √fc' · b · d`

Defaults for `fc'` and `fy` come from `.env` (`CONCRETE_STRENGTH_DEFAULT_MPA`, `STEEL_YIELD_DEFAULT_MPA`) but can be overridden per-query.

> **Note:** These are illustrative code-check equations for demonstration and design-assist purposes. They are **not** a substitute for a licensed professional engineer's sealed design calculations. Always verify against the current edition of the governing code and local amendments.

---

## 🧪 Demo Mode vs Production

### Demo mode (no API keys)

If `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, and `GOOGLE_API_KEY` are all unset/placeholder, StructuraMind runs fully offline:

* **Embeddings:** deterministic mock vectors (no model download needed after the fallback).
* **LLM:** deterministic placeholder responses for Retrieval / Calculation / Critic agents.
* **Vector DB:** in-memory store (no Qdrant required).

Use this to evaluate the UI, API, and ingestion pipeline without any cloud credentials.

### Production mode

1. Set `LLM_PROVIDER=anthropic` (or `openai`) and provide a real API key in `.env`.
2. Run Qdrant via Docker (`docker compose up -d qdrant`) for persisted, hybrid search.
3. Set `APP_ENV=production`, `DEBUG=false`, and a strong `SECRET_KEY`.
4. Deploy `.env` values through your secret manager — **never commit `.env`**.

---

## 🔐 Environment Variable Reference

All variables live in `.env` (gitignored).

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_NAME` | `StructuraMind` | Service name |
| `APP_ENV` | `development` | deployment environment |
| `DEBUG` | `true` | debug logging |
| `LOG_LEVEL` | `INFO` | logging level |
| `API_HOST` | `0.0.0.0` | FastAPI bind host |
| `API_PORT` | `8000` | FastAPI port |
| `STREAMLIT_PORT` | `8501` | Frontend port |
| `SECRET_KEY` | *(dev placeholder)* | session/signing secret |
| `LLM_PROVIDER` | `anthropic` | `anthropic\|openai\|local\|mock` |
| `ANTHROPIC_API_KEY` | *(placeholder)* | Claude API key |
| `ANTHROPIC_MODEL` | `claude-3-5-sonnet-20241022` | Claude model |
| `OPENAI_API_KEY` | *(placeholder)* | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model |
| `GOOGLE_API_KEY` | *(placeholder)* | Google API key |
| `GOOGLE_MODEL` | `gemini-1.5-pro` | Google model |
| `EMBEDDING_PROVIDER` | `local` | `local\|openai\|anthropic` |
| `LOCAL_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `QDRANT_API_KEY` | *(empty)* | Qdrant API key |
| `QDRANT_COLLECTION_CODES` | `structuramind_codes` | codes collection |
| `QDRANT_COLLECTION_BIM` | `structuramind_bim` | BIM collection |
| `QDRANT_COLLECTION_REPORTS` | `structuramind_reports` | reports collection |
| `QDRANT_VECTOR_SIZE` | `384` | embedding dimension |
| `QDRANT_DISTANCE` | `Cosine` | `Cosine\|Euclid\|Dot` |
| `CHUNK_SIZE` | `1000` | max chunk characters |
| `CHUNK_OVERLAP` | `200` | chunk overlap |
| `TOP_K_RETRIEVAL` | `10` | retrieval top-K |
| `ENABLE_AGENTIC_LOOP` | `true` | use LangGraph agents |
| `MAX_AGENT_ITERATIONS` | `3` | max loop iterations |
| `ENABLE_CALCULATION_AGENT` | `true` | run calculation agent |
| `ENABLE_CRITIC_AGENT` | `true` | run critic agent |
| `CRITIC_HALLUCINATION_THRESHOLD` | `0.75` | critic pass threshold |
| `CONCRETE_STRENGTH_DEFAULT_MPA` | `30` | default fc' (MPa) |
| `STEEL_YIELD_DEFAULT_MPA` | `420` | default fy (MPa) |
| `DEMO_MODE` | `false` | force demo mode |


## 🛡️ Security

* **`.env` is gitignored.** Never add `.env` to version control. There is **no** `.env.example` by design — all secrets live in the single local `.env`.
* API keys are read at runtime via `pydantic-settings` and never logged.
* In production, inject secrets through your platform's secret manager and mount them as environment variables rather than shipping a `.env` file.
* The Calculation Agent's equations are design-assist only — they do **not** replace a licensed engineer's sealed calculations.

---

## 🧪 Testing

Basic import + endpoint smoke test:

```bash
# Start backend
uvicorn app.main:app --port 8000 &

# Health
curl http://localhost:8000/health

# Ingest text
curl -X POST http://localhost:8000/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"text":"ACI 318 Section 22.4 flexural strength.","document_id":"t1","collection":"codes"}'

# Query
curl -X POST http://localhost:8000/query/ \
  -H "Content-Type: application/json" \
  -d '{"query":"flexural strength ACI 318","collections":["codes"],"top_k":5}'
```

For a fuller test, ingest a real building-code PDF via `/ingest/pdf` and confirm that queries return citations with `page_number`, `section`, and `clause`.

---

## 🗺️ Roadmap

* [ ] **IFC geometry**: full 3D bounding boxes and spatial queries via IfcOpenShell + OCC.
* [ ] **Hybrid search**: BM25 + vector (Qdrant sparse + dense) for better code-clause recall.
* [ ] **Reranking**: cross-encoder reranker for top-K results.
* [ ] **More codes**: AISC steel, IBC, Eurocode 2, AS 3600, CSA A23.3.
* [ ] **Reinforcement detailing**: auto-generate rebar schedules from calculations.
* [ ] **User auth & projects**: per-engineer workspaces and document access control.
* [ ] **PDF drawing annotations**: highlight retrieved clauses directly on uploaded structural drawings.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (remember: **never commit `.env`**).
4. Open a pull request with a clear description.

Please follow the existing code style and add docstrings to new public functions.

---

## 📄 License

MIT License. See `LICENSE` for details.

---

## 🙏 Acknowledgements

* **LangChain / LangGraph** — agent orchestration
* **Qdrant** — vector database
* **sentence-transformers** — local embeddings
* **PyMuPDF** — PDF parsing
* **IfcOpenShell** — IFC/BIM parsing
* **FastAPI / Streamlit** — backend and frontend
* **Anthropic / OpenAI** — LLM backends

---

**Built for civil and structural engineers who want their code compliance grounded, cited, and verified — not guessed.**


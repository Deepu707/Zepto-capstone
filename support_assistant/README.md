# Module 3 — Support Assistant

A grounded RAG (Retrieval-Augmented Generation) service for Zepto's policy
questions. Ingests 8 policy documents, embeds them with a local
sentence-transformer model, stores them in ChromaDB, routes each query
through a LangGraph state machine, and returns a Pydantic-validated JSON
response via FastAPI.

**Default behavior (graded baseline): fully offline, no API key.**
All LLM calls are gated by `MOCK_LLM`; when unset or set to `1`, the
pipeline uses deterministic rule-based mock logic and makes **no network
call to any LLM provider**.

---

## Deliverables in this folder

| File | Purpose |
|---|---|
| `docs/doc_01.txt … doc_08.txt` | The 8 Zepto policy documents (verbatim from the brief) |
| `ingest.py` | Chunk + embed + store in ChromaDB |
| `prompt_template.py` | Role–context–task–format–length prompt + negative constraint + few-shot |
| `graph.py` | LangGraph StateGraph (3 nodes, conditional edge, MOCK_LLM branching) |
| `app/main.py` | FastAPI wrapper with Pydantic in/out schemas |
| `Dockerfile` | Containerization (locally buildable) |
| `.dockerignore` | Keeps the image lean |
| `requirements.txt` | Python dependencies |
| `chroma_db/` | Local ChromaDB persistence (built by `ingest.py` or Dockerfile) |

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate                # Windows
# source .venv/bin/activate            # macOS/Linux

pip install -r requirements.txt
```

> First run downloads the `all-MiniLM-L6-v2` model (~90 MB) — cached after that.

---

## How to run end-to-end

### 1. Build the vector store

```bash
cd support_assistant
python ingest.py
```

Expected: `Loaded 8 documents` → `Produced N chunks` → `Stored N chunks in ChromaDB`.

### 2. Smoke-test the graph

```bash
python graph.py
```

Prints two queries (one policy, one general) with their routing decisions and
final state.

### 3. Run the API

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then from another terminal:

```bash
curl.exe -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{\"query\": \"What is the delivery fee?\"}'
```

---

## RAG pipeline architecture (ingestion → embedding → retrieval → generation)

```
                 ┌───────────────────────────────────────────────────┐
   [docs/*.txt]  │  INGESTION                                        │
   8 files  ───► │  ingest.py: load_documents() + chunk_document()   │
                 └───────────────────────┬───────────────────────────┘
                                         │ 15 text chunks
                                         ▼
                 ┌───────────────────────────────────────────────────┐
                 │  EMBEDDING                                        │
                 │  ingest.py: SentenceTransformer(all-MiniLM-L6-v2) │
                 │  → 384-dim vectors, persisted in chroma_db/       │
                 └───────────────────────┬───────────────────────────┘
                                         │
   [user query] ─────────────────────────┤
                                         ▼
                 ┌───────────────────────────────────────────────────┐
                 │  CLASSIFY (LangGraph node 1)                      │
                 │  graph.py::classify_intent                        │
                 │  • MOCK_LLM=1 (default): keyword heuristic        │
                 │  • MOCK_LLM=0 (optional): real LLM call           │
                 └───────────────┬───────────────────────┬───────────┘
                                 │ policy_question       │ general_question
                                 ▼                       ▼
                 ┌───────────────────────────┐   ┌─────────────────────┐
                 │  RETRIEVAL + GENERATION   │   │  DIRECT ANSWER      │
                 │  graph.py::               │   │  graph.py::         │
                 │    retrieve_and_answer    │   │    direct_answer    │
                 │  • retrieval (ALWAYS):    │   │  • MOCK: canned     │
                 │    ChromaDB cosine top-3  │   │    string           │
                 │  • generation: branches   │   │  • LLM (optional)   │
                 │    on MOCK_LLM            │   │                     │
                 └───────────┬───────────────┘   └──────────┬──────────┘
                             │                              │
                             └──────────────┬───────────────┘
                                            ▼
                        ┌──────────────────────────────────────┐
                        │  Pydantic AskResponse                │
                        │  { answer, sources, confidence }     │
                        │  app/main.py::POST /ask              │
                        └──────────────────────────────────────┘
```

### Stage-by-stage

| Stage | File / function | Notes |
|---|---|---|
| **Ingestion** | `ingest.py::load_documents` + `chunk_document` | Splits doc text on sentence boundaries; each chunk ≤ 400 chars |
| **Embedding** | `ingest.py::build_collection` via `SentenceTransformer("all-MiniLM-L6-v2")` | 384-dim, normalized; stored in ChromaDB collection `zepto_policies` |
| **Classification** | `graph.py::classify_intent` (node 1) | **Branches on `MOCK_LLM`.** Default: keyword heuristic. Optional: real LLM |
| **Retrieval** | `graph.py::retrieve_and_answer` | **Always real** (no LLM needed): embeds the query, queries Chroma with cosine similarity, top-3 |
| **Generation** | `graph.py::retrieve_and_answer` (policy) / `direct_answer` (general) | **Branches on `MOCK_LLM`.** Default: canned templates. Optional: real LLM |
| **Validation** | `app/main.py::AskResponse` (Pydantic) | Enforces `{answer: str, sources: list[str], confidence: 0..1}`; retry-on-failure logic present for real-LLM path |

### What changes between `MOCK_LLM` states

| Stage | `MOCK_LLM` unset or `1` (default, graded) | `MOCK_LLM=0` (optional, ungraded) |
|---|---|---|
| `classify_intent` | Keyword heuristic on 8 keywords | Real LLM classifies the query |
| `retrieve_and_answer` | Retrieval runs for real; answer is `"Based on the retrieved context: {top_200_chars}"` | Retrieval same; real LLM answers from retrieved chunks using `prompt_template.py` |
| `direct_answer` | Fixed string: `"I can only answer questions about Zepto policies right now."` | Real LLM answers directly (no retrieval) |
| Network calls to LLM | **None** | Yes, to Groq (free tier) or similar |
| Pydantic response | Deterministically populated by code | Populated from LLM output; validated + retried on failure |

---

## Structured prompt template (used only on the `MOCK_LLM=0` path)

`prompt_template.py` follows the **role–context–task–format–length** skeleton:

- **Role:** "You are Zepto's support assistant."
- **Context:** Retrieved policy excerpts (only source of truth)
- **Task:** Answer using only that context
- **Negative constraint:** *"Do not answer using information not present in the provided context."*
- **Format:** Compact JSON `{answer, sources, confidence}`
- **Length:** Under 3 sentences
- **Few-shot:** One example question/context/answer triple embedded inline

---

## Example API calls (run with `MOCK_LLM` at its default)

### 1. Policy query → routes to `retrieve_and_answer`

**Request:**
```json
POST /ask
{"query": "What is the delivery fee?"}
```

**Response:**
```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del",
  "sources": ["doc_01", "doc_03", "doc_05"],
  "confidence": 1.0
}
```

**Notes:** Classification matched keyword `delivery` → `policy_question`.
Retrieval returned `doc_01` as the top chunk (the delivery-policy document —
correct source for the question). Answer uses the required canned
`"Based on the retrieved context: ..."` template built from the top chunk.

### 2. General query → routes to `direct_answer`

**Request:**
```json
POST /ask
{"query": "What is the capital of France?"}
```

**Response:**
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

**Notes:** No policy keyword matched → `general_question`. Retrieval is
skipped. `sources` is empty (correct for general queries), `confidence` = 1.0
as per spec.

### 3. Health check

**Request:** `GET /health`

**Response:**
```json
{"status": "ok", "mock_llm": "1"}
```

---

## Docker (required graded baseline: locally buildable + runnable)

The Dockerfile has been built and run locally. Build + run commands:

```bash
# Build
docker build -t zepto-support-assistant .

# Run
docker run --rm -p 7860:7860 zepto-support-assistant

# Test (from another terminal)
curl.exe -X POST http://127.0.0.1:7860/ask \
  -H "Content-Type: application/json" \
  -d '{\"query\": \"What is the delivery fee?\"}'
```

The Dockerfile:
- Uses `python:3.11-slim` as the base
- Installs `requirements.txt`
- Copies the corpus, ingestion script, graph, and API
- **Runs `ingest.py` during image build** so ChromaDB is pre-populated
- Sets `MOCK_LLM=1` as the default env
- Serves `app.main:app` via Uvicorn on `0.0.0.0:7860`

Deploying to Hugging Face Spaces (free CPU tier) was mentioned as an
**optional, ungraded** extension; it is **not** performed here.

---

## Acceptance checklist

- [x] All 8 corpus documents embedded and queryable from ChromaDB
- [x] Structured prompt template shows all 5 skeleton parts + negative constraint + few-shot
- [x] `classify_intent` uses keyword heuristic by default and correctly routes policy vs general queries
- [x] LangGraph has 3 named nodes + a working conditional edge
- [x] Retrieval runs for real in both modes; policy query returns the correct source document
- [x] `retrieve_and_answer` mock output uses the `"Based on the retrieved context: ..."` template
- [x] `direct_answer` mock output is the fixed canned string
- [x] Pydantic schema `{answer, sources, confidence}` populated deterministically in mock mode
- [x] Retry-on-failure code for the optional real-LLM path is present in `app/main.py`
- [x] FastAPI runs locally via uvicorn; both example calls captured above with `MOCK_LLM` default
- [x] Dockerfile present, locally buildable/runnable (serving `/ask`)
- [x] README includes the RAG architecture description and stage-by-stage breakdown
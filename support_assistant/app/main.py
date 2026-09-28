"""
Module 3 — FastAPI wrapper around the LangGraph RAG pipeline.

Endpoints:
  POST /ask   -> validated Pydantic response: {answer, sources, confidence}
  GET  /health

Run locally:
  uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import os
import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

# Make `graph.py` importable when running from project root or app/ folder
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph import run_query  # noqa: E402


# ---------------------------------------------------------------------------
# Pydantic schemas — REQUIRED output contract
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query")


class AskResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Zepto Support Assistant",
    version="1.0.0",
    description="Grounded RAG service over Zepto policy documents.",
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "mock_llm": os.getenv("MOCK_LLM", "1"),
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    result = run_query(req.query)

    # Validate against the Pydantic schema before returning.
    # Mock mode always produces a valid object deterministically.
    # Real-LLM mode (MOCK_LLM=0) can raise here — retried once with a
    # corrective instruction before giving up.
    try:
        return AskResponse(**result)
    except Exception as e:
        if os.getenv("MOCK_LLM") == "0":
            # One corrective retry for the real-LLM path.
            try:
                retry = run_query(
                    req.query + "\n\nReturn ONLY a JSON object with keys "
                    "answer, sources, confidence."
                )
                return AskResponse(**retry)
            except Exception:
                pass
        # Clearly marked error response
        return AskResponse(
            answer=f"[ERROR] Could not validate model output: {e}",
            sources=[],
            confidence=0.0,
        )
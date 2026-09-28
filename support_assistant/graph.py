"""
Module 3 — LangGraph StateGraph.

Nodes:
  - classify_intent       (keyword heuristic by default; LLM if MOCK_LLM=0)
  - retrieve_and_answer   (retrieval ALWAYS runs; generation branches on MOCK_LLM)
  - direct_answer         (mock string by default; LLM if MOCK_LLM=0)

The conditional edge routes policy_question -> retrieve_and_answer
                                    general_question -> direct_answer.
"""

import os
from pathlib import Path
from typing import TypedDict

import chromadb
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END

from prompt_template import build_prompt

HERE = Path(__file__).resolve().parent
CHROMA_DIR = HERE / "chroma_db"
COLLECTION_NAME = "zepto_policies"
EMBED_MODEL = "all-MiniLM-L6-v2"

POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership", "tracking",
    "cancel", "gift card", "support hours",
]


# ---------------------------------------------------------------------------
# Shared resources (lazy-loaded once per process)
# ---------------------------------------------------------------------------
_embedder = None
_collection = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def is_mock() -> bool:
    """MOCK_LLM unset OR MOCK_LLM=1 -> mock mode (graded baseline)."""
    val = os.getenv("MOCK_LLM")
    return val is None or val == "1"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class AssistantState(TypedDict, total=False):
    query: str
    intent: str                 # "policy_question" | "general_question"
    retrieved: list             # list of dicts: {id, doc_id, text, distance}
    answer: str
    sources: list               # list of doc_ids used
    confidence: float


# ---------------------------------------------------------------------------
# Node 1 — classify_intent
# ---------------------------------------------------------------------------
def classify_intent(state: AssistantState) -> AssistantState:
    query = state["query"].lower()

    if is_mock():
        # Deterministic keyword heuristic — no LLM call.
        intent = "policy_question" if any(k in query for k in POLICY_KEYWORDS) else "general_question"
        print(f"  [classify_intent | MOCK] query={state['query']!r} -> {intent}")
    else:
        # Optional real-LLM path (ungraded extension).
        intent = _llm_classify(state["query"])
        print(f"  [classify_intent | LLM] query={state['query']!r} -> {intent}")

    return {"intent": intent}


def _llm_classify(query: str) -> str:
    """Placeholder for the optional MOCK_LLM=0 path."""
    try:
        from langchain_groq import ChatGroq  # optional dep
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        prompt = (
            "Classify the user query as either 'policy_question' or "
            "'general_question'. Reply with only the label.\n"
            f"Query: {query}"
        )
        out = llm.invoke(prompt).content.strip().lower()
        return "policy_question" if "policy" in out else "general_question"
    except Exception as e:
        print(f"  [classify_intent | LLM] falling back to heuristic due to: {e}")
        return "policy_question" if any(k in query.lower() for k in POLICY_KEYWORDS) else "general_question"


# ---------------------------------------------------------------------------
# Node 2 — retrieve_and_answer
# ---------------------------------------------------------------------------
def retrieve_and_answer(state: AssistantState) -> AssistantState:
    # Retrieval ALWAYS runs for real, in both modes.
    embedder = get_embedder()
    collection = get_collection()

    q_emb = embedder.encode([state["query"]], normalize_embeddings=True)[0].tolist()
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )

    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    retrieved = [
        {"id": i, "doc_id": m["doc_id"], "text": d, "distance": dist}
        for i, d, m, dist in zip(ids, docs, metas, dists)
    ]
    print(f"  [retrieve] top-3 doc_ids = {[r['doc_id'] for r in retrieved]}")

    top = retrieved[0]

    if is_mock():
        snippet = top["text"][:200].strip()
        answer = f"Based on the retrieved context: {snippet}"
        sources = [r["doc_id"] for r in retrieved]
        confidence = 1.0
        print(f"  [retrieve_and_answer | MOCK] templated answer built from {top['doc_id']}")
    else:
        # Optional real-LLM path.
        answer = _llm_answer(state["query"], retrieved)
        sources = [r["doc_id"] for r in retrieved]
        confidence = 0.8

    return {
        "retrieved": retrieved,
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
    }


def _llm_answer(question: str, chunks: list) -> str:
    """Placeholder for the optional MOCK_LLM=0 path."""
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        return llm.invoke(build_prompt(question, chunks)).content.strip()
    except Exception as e:
        return f"[LLM error: {e}]"


# ---------------------------------------------------------------------------
# Node 3 — direct_answer
# ---------------------------------------------------------------------------
def direct_answer(state: AssistantState) -> AssistantState:
    if is_mock():
        answer = "I can only answer questions about Zepto policies right now."
        print(f"  [direct_answer | MOCK] fixed canned response")
    else:
        answer = _llm_answer_direct(state["query"])
        print(f"  [direct_answer | LLM] {answer[:80]}...")

    return {
        "answer": answer,
        "sources": [],
        "confidence": 1.0 if is_mock() else 0.5,
    }


def _llm_answer_direct(question: str) -> str:
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        return llm.invoke(question).content.strip()
    except Exception as e:
        return f"[LLM error: {e}]"


# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------
def route(state: AssistantState) -> str:
    return "retrieve" if state["intent"] == "policy_question" else "direct"


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------
def build_graph():
    g = StateGraph(AssistantState)
    g.add_node("classify_intent", classify_intent)
    g.add_node("retrieve_and_answer", retrieve_and_answer)
    g.add_node("direct_answer", direct_answer)

    g.set_entry_point("classify_intent")
    g.add_conditional_edges(
        "classify_intent",
        route,
        {"retrieve": "retrieve_and_answer", "direct": "direct_answer"},
    )
    g.add_edge("retrieve_and_answer", END)
    g.add_edge("direct_answer", END)

    return g.compile()


_GRAPH = None


def run_query(query: str) -> dict:
    """Convenience wrapper: run the graph and return the final state dict."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    final_state = _GRAPH.invoke({"query": query})
    return {
        "answer": final_state.get("answer", ""),
        "sources": final_state.get("sources", []),
        "confidence": final_state.get("confidence", 0.0),
    }


if __name__ == "__main__":
    print("=== MOCK_LLM default (offline) ===")
    for q in ["What is the delivery fee?", "What is the capital of France?"]:
        print(f"\nQ: {q}")
        result = run_query(q)
        print(f"Result: {result}")
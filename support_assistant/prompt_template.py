"""
Module 3 — RAG prompt template.

Follows the role–context–task–format–length skeleton, includes a negative
constraint, and embeds one few-shot example. Used by the optional
MOCK_LLM=0 (real LLM) path — the graded baseline (MOCK_LLM unset/1)
does not call an LLM at all.
"""

SYSTEM_PROMPT = """ROLE:
You are Zepto's support assistant. You answer customer questions about
Zepto's delivery, returns, membership, tracking, cancellation, gift cards,
and support policies.

CONTEXT:
You will be given one or more retrieved policy excerpts. Treat them as the
ONLY source of truth.

TASK:
Answer the user's question using ONLY the information present in the
provided context.

NEGATIVE CONSTRAINT:
Do not answer using information not present in the provided context. If the
answer is not contained in the context, reply exactly:
"I don't have that information in Zepto's policies."

FEW-SHOT EXAMPLE:
User question: "How much does standard delivery cost?"
Retrieved context: "Standard delivery is free on orders over INR 149; orders
below this threshold incur a flat INR 25 delivery fee."
Answer: "Standard delivery is free on orders over INR 149. Below that, a flat
INR 25 delivery fee applies."

FORMAT:
Return a compact JSON object with exactly these keys:
  - "answer":      string (your grounded answer)
  - "sources":     list of document IDs used (e.g. ["doc_01"])
  - "confidence":  float between 0 and 1

LENGTH:
Keep the answer under 3 sentences.
"""


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """Assemble the final prompt string for the real-LLM path."""
    context_lines = []
    for c in retrieved_chunks:
        context_lines.append(f"[{c['doc_id']}] {c['text']}")
    context_block = "\n".join(context_lines) if context_lines else "(no context retrieved)"

    return (
        f"{SYSTEM_PROMPT}\n"
        f"---\n"
        f"RETRIEVED CONTEXT:\n{context_block}\n"
        f"---\n"
        f"USER QUESTION: {question}\n"
        f"---\n"
        f"Return ONLY the JSON object."
    )
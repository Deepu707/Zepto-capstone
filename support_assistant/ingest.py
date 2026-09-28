"""
Module 3 — Ingestion: load the 8 Zepto policy docs, chunk them,
embed with all-MiniLM-L6-v2, and persist to ChromaDB.

Run:  python ingest.py
"""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
DOCS_DIR = HERE / "docs"
CHROMA_DIR = HERE / "chroma_db"
COLLECTION_NAME = "zepto_policies"
EMBED_MODEL = "all-MiniLM-L6-v2"


def load_documents() -> list[tuple[str, str]]:
    """Return [(doc_id, text), ...] for every .txt file in docs/."""
    docs = []
    for path in sorted(DOCS_DIR.glob("doc_*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            docs.append((path.stem, text))
    return docs


def chunk_document(doc_id: str, text: str, max_chars: int = 400) -> list[dict]:
    """
    Simple per-document chunking: split on sentence boundaries,
    accumulate up to max_chars per chunk. Given these docs are short,
    each one typically produces 1–2 chunks.
    """
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) + 1 <= max_chars:
            current = (current + " " + s).strip()
        else:
            if current:
                chunks.append(current)
            current = s
    if current:
        chunks.append(current)

    return [
        {"id": f"{doc_id}::chunk{idx}", "doc_id": doc_id, "text": c}
        for idx, c in enumerate(chunks)
    ]


def build_collection() -> chromadb.Collection:
    """Embed all chunks and store in a fresh ChromaDB collection."""
    docs = load_documents()
    print(f"Loaded {len(docs)} documents from {DOCS_DIR}")

    all_chunks: list[dict] = []
    for doc_id, text in docs:
        all_chunks.extend(chunk_document(doc_id, text))
    print(f"Produced {len(all_chunks)} chunks")

    print(f"Loading embedding model {EMBED_MODEL} ...")
    embedder = SentenceTransformer(EMBED_MODEL)

    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    print(f"Embedded {len(embeddings)} chunks (dim={embeddings.shape[1]})")

    CHROMA_DIR.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Fresh collection
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[c["id"] for c in all_chunks],
        documents=[c["text"] for c in all_chunks],
        embeddings=[e.tolist() for e in embeddings],
        metadatas=[{"doc_id": c["doc_id"]} for c in all_chunks],
    )
    print(f"Stored {collection.count()} chunks in ChromaDB at {CHROMA_DIR}")
    return collection


if __name__ == "__main__":
    build_collection()
    print("Ingestion complete.")
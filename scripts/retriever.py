"""
retriever.py  —  query ChromaDB for relevant Act sections
----------------------------------------------------------
Given a clause from a user's document, returns the top-k most
relevant chunks from the Acts database.

Usage (standalone test):
    python retriever.py
"""

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

# ── CONFIG ────────────────────────────────────────────────────────────────────

DB_PATH    = r"D:\College\legal_rag\legal_rag\acts_db"
COLLECTION = "indian_acts"
TOP_K      = 3

# ── EMBEDDING MODEL (same as ingestion) ──────────────────────────────────────

EMBEDDINGS = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# ── RETRIEVER ─────────────────────────────────────────────────────────────────

client     = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(COLLECTION)


def retrieve(clause_text: str, top_k: int = TOP_K) -> list[dict]:
    """
    Query the Acts DB with a clause and return top_k results.

    Each result dict contains:
        - text          : the matching chunk text
        - act_name      : e.g. "Consumer Protection Act 2019"
        - section_number: int
        - section_title : str
        - chapter       : str
        - chunk_type    : "section" | "subsection" | "clause" | "explanation" ...
        - subsection    : e.g. "1", "2", or ""
        - clause        : e.g. "a", "b", or ""
        - similarity    : float 0–1 (1 = perfect match)
    """
    # Embed the query
    query_vector = EMBEDDINGS.embed_query(clause_text)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({
            "text":           doc,
            "act_name":       meta["act_name"],
            "section_number": meta["section_number"],
            "section_title":  meta["section_title"],
            "chapter":        meta["chapter"],
            "chunk_type":     meta["chunk_type"],
            "subsection":     meta.get("subsection", ""),
            "clause":         meta.get("clause", ""),
            "similarity":     round(1 - dist, 4),
        })

    return output


def format_for_prompt(results: list[dict]) -> str:
    """
    Format retrieved chunks into a clean string to paste into an LLM prompt.
    """
    lines = []
    for i, r in enumerate(results, 1):
        # Build a readable citation
        citation = f"{r['act_name']}, Section {r['section_number']}"
        if r["subsection"]:
            citation += f"({r['subsection']})"
        if r["clause"]:
            citation += f"({r['clause']})"

        lines.append(f"[{i}] {citation}")
        lines.append(f"     {r['text']}")
        lines.append("")
    return "\n".join(lines)


# ── QUICK TEST ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_clauses = [
        "The tenant shall vacate the premises within 15 days of receiving notice.",
        "Any dispute arising out of this agreement shall be settled by arbitration.",
        "The employee shall not disclose any confidential information to third parties.",
    ]

    for clause in test_clauses:
        print(f"\nClause: {clause}")
        print("─" * 60)
        results = retrieve(clause)
        for r in results:
            print(f"  [{r['similarity']:.3f}] Section {r['section_number']}: {r['section_title']}")
            print(f"          {r['act_name']} | {r['chapter']}")
            print(f"          {r['text'][:150].replace(chr(10), ' ')}...")
            print()
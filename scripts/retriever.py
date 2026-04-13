"""
Hybrid Retriever
Vector search (Chroma) + BM25 keyword search
Proper score normalization + weighted fusion
"""

import chromadb
import numpy as np
import re

from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi


# ───────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────

DB_PATH = r"D:\College\legal_rag\legal_rag\acts_db"
COLLECTION = "indian_acts"

VECTOR_WEIGHT = 0.6
BM25_WEIGHT = 0.4
TOP_K = 5


# ───────────────────────────────────────────────
# EMBEDDING MODEL
# ───────────────────────────────────────────────

EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


# ───────────────────────────────────────────────
# LOAD CHROMA
# ───────────────────────────────────────────────

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(COLLECTION)

data = collection.get(include=["documents", "metadatas"])

documents = data["documents"]
metadatas = data["metadatas"]


# ───────────────────────────────────────────────
# BUILD BM25 INDEX
# ───────────────────────────────────────────────

def tokenize(text):
    return re.findall(r"\w+", text.lower())

tokenized_corpus = [tokenize(doc) for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)


# ───────────────────────────────────────────────
# NORMALIZATION
# ───────────────────────────────────────────────

def normalize(scores):
    scores = np.array(scores)

    if scores.max() == scores.min():
        return np.ones(len(scores))

    return (scores - scores.min()) / (scores.max() - scores.min())


# ───────────────────────────────────────────────
# RETRIEVAL
# ───────────────────────────────────────────────

def retrieve(query, top_k=TOP_K):

    # ---------- VECTOR SEARCH ----------
    query_vec = EMBEDDINGS.embed_query(query)

    vec_results = collection.query(
        query_embeddings=[query_vec],
        n_results=20,
        include=["documents", "metadatas", "distances"]
    )

    vec_docs = vec_results["documents"][0]
    vec_meta = vec_results["metadatas"][0]

    vec_scores = [1 - d for d in vec_results["distances"][0]]
    vec_scores = normalize(vec_scores)

    vector_hits = []

    for doc, meta, score in zip(vec_docs, vec_meta, vec_scores):

        vector_hits.append({
            "text": doc,
            "metadata": meta,
            "vector_score": score
        })


    # ---------- BM25 SEARCH ----------
    tokenized_query = tokenize(query)

    bm25_scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(bm25_scores)[::-1][:20]

    bm25_scores = normalize([bm25_scores[i] for i in top_indices])

    bm25_hits = []

    for idx, score in zip(top_indices, bm25_scores):

        bm25_hits.append({
            "text": documents[idx],
            "metadata": metadatas[idx],
            "bm25_score": score
        })


    # ---------- MERGE RESULTS ----------
    merged = {}

    for r in vector_hits:
        key = r["text"]

        merged.setdefault(key, {
            "text": r["text"],
            "metadata": r["metadata"],
            "vector": 0,
            "bm25": 0
        })

        merged[key]["vector"] = r["vector_score"]


    for r in bm25_hits:
        key = r["text"]

        merged.setdefault(key, {
            "text": r["text"],
            "metadata": r["metadata"],
            "vector": 0,
            "bm25": 0
        })

        merged[key]["bm25"] = r["bm25_score"]


    # ---------- WEIGHTED FUSION ----------
    results = []

    for item in merged.values():

        score = (
            VECTOR_WEIGHT * item["vector"] +
            BM25_WEIGHT * item["bm25"]
        )

        meta = item["metadata"]

        results.append({
            "text": item["text"],
            "act_name": meta["act_name"],
            "section_number": meta["section_number"],
            "section_title": meta["section_title"],
            "chapter": meta["chapter"],
            "chunk_type": meta["chunk_type"],
            "subsection": meta.get("subsection", ""),
            "clause": meta.get("clause", ""),
            "similarity": round(score,4)
        })


    # ---------- FINAL RANK ----------
    results.sort(key=lambda x: x["similarity"], reverse=True)

    return results[:top_k]


# ───────────────────────────────────────────────
# FORMAT FOR PROMPT
# ───────────────────────────────────────────────

def format_for_prompt(results):

    lines = []

    for i, r in enumerate(results, 1):

        citation = f"{r['act_name']}, Section {r['section_number']}"

        if r["subsection"]:
            citation += f"({r['subsection']})"

        if r["clause"]:
            citation += f"({r['clause']})"

        lines.append(f"[{i}] {citation}")
        lines.append(f"     {r['text']}")
        lines.append("")

    return "\n".join(lines)


# ───────────────────────────────────────────────
# TEST
# ───────────────────────────────────────────────

if __name__ == "__main__":

    tests = [
        "tenant subletting without landlord permission",
        "personal data leak from website",
        "being assaulted in a public place, what can I do?",
    ]

    for t in tests:

        print("\nQUERY:", t)
        print("-"*60)

        results = retrieve(t)

        for r in results:
            print(
                f"[{r['similarity']:.3f}] "
                f"{r['act_name']} Section {r['section_number']}"
            )
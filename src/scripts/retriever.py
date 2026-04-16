"""
Hybrid Retriever
Vector search (Chroma) + BM25 keyword search
Proper score normalization + weighted fusion
+ Query rewriting (LLM-based)
+ Cross-encoder reranking
"""

import chromadb
import numpy as np
import re

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


# ───────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────

DB_PATH = r"D:\College\legal_rag\legal_rag\db"
COLLECTION = "indian_acts"

VECTOR_WEIGHT = 0.4
BM25_WEIGHT   = 0.6
TOP_K         = 10
CANDIDATE_K   = 10 #300
SECTION_BOOST = 1.0
ACT_BOOST     = 0.5
RERANK_TOP_K  = 10  # final results after reranking


# ───────────────────────────────────────────────
# MODELS
# ───────────────────────────────────────────────

EMBEDDINGS = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

RERANKER = CrossEncoder("BAAI/bge-reranker-base")

REWRITE_MODEL = "mistral"
rewrite_llm = ChatOllama(model=REWRITE_MODEL, temperature=0.0)


# ───────────────────────────────────────────────
# LOAD CHROMA
# ───────────────────────────────────────────────

client     = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(COLLECTION)

data      = collection.get(include=["documents", "metadatas"])
documents = data["documents"]
metadatas = data["metadatas"]


# ───────────────────────────────────────────────
# BUILD BM25 INDEX
# ───────────────────────────────────────────────

def _strip_doc_prefix(text: str) -> str:
    """Remove the 'Act\nSection N - Title\n' prefix added during chunking."""
    # The prefix is always the first two lines
    lines = text.split("\n", 2)
    return lines[2] if len(lines) >= 3 else text

def tokenize(text):
    return re.findall(r"\w+", text.lower())

raw_clause_texts = [_strip_doc_prefix(d) for d in documents]
tokenized_corpus = [tokenize(t) for t in raw_clause_texts]
bm25             = BM25Okapi(tokenized_corpus)


# ───────────────────────────────────────────────
# QUERY REWRITING
# ───────────────────────────────────────────────

def rewrite_query(query: str) -> str:
    """
    Expand the user query with legal context using an LLM.
    Falls back to original query on failure.
    """
    prompt = (
        "You are a legal search assistant for Indian law. "
        "Rewrite the following query to include relevant legal keywords and terms, "
        "keep it concise and similar length, do not add specific act names or section numbers."
        "Return ONLY the rewritten query, nothing else.\n\n"
        f"Query: {query}"
    )
    try:
        resp = rewrite_llm.invoke([HumanMessage(content=prompt)])
        rewritten = resp.content.strip()
        print(f"[Query Rewrite] '{query}' → '{rewritten}'")
        return rewritten
    except Exception as e:
        print(f"[Query Rewrite] Failed ({e}), using original query.")
        return query


# ───────────────────────────────────────────────
# DETECT HELPERS
# ───────────────────────────────────────────────

def detect_section(query):
    m = re.search(r"section\s+(\d+)", query.lower())
    return int(m.group(1)) if m else None

def detect_act(query):
    acts = set(meta["act_name"] for meta in metadatas)
    q = query.lower()
    for act in acts:
        if act.lower() in q:
            return act
    return None


# ───────────────────────────────────────────────
# NORMALIZATION
# ───────────────────────────────────────────────

def normalize_scores(scores):
    scores = np.array(scores, dtype=float)
    if scores.size == 0:
        return scores
    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return np.ones_like(scores)
    return (scores - lo) / (hi - lo)


def distance_to_similarity(distances):
    distances = np.array(distances, dtype=float)
    return 1.0 / (1.0 + distances)


# ───────────────────────────────────────────────
# RETRIEVAL
# ───────────────────────────────────────────────
def _key(hit: dict) -> tuple:
    m = hit["metadata"]
    return (
        m.get("act_name", ""),
        m.get("section_number", ""),
        m.get("chunk_type", ""),
        m.get("subsection", ""),
        m.get("clause", ""),
    )
def dedup_best(results):
    best = {}

    for r in results:
        key = (r["act_name"], r["section_number"])
        score = r.get("similarity", 0)

        if key not in best or score > best[key].get("similarity", 0):
            best[key] = r

    return list(best.values())

def retrieve(query, mode="default",top_k=TOP_K):
    # --- Query Rewriting ---
    # rewritten = rewrite_query(query)
    rewritten = query  # skipping rewriting for now to isolate BM25 performance
    section_query = detect_section(rewritten) or detect_section(query)
    act_query     = detect_act(rewritten)

    # ---------- VECTOR SEARCH ----------
    query_vec = EMBEDDINGS.embed_query(rewritten)

    vec_results = collection.query(
        query_embeddings=[query_vec],
        n_results=max(top_k, CANDIDATE_K),
        include=["documents", "metadatas", "distances"],
    )

    vec_docs      = vec_results["documents"][0]
    vec_metas     = vec_results["metadatas"][0]
    vec_distances = vec_results["distances"][0]

    vec_scores = normalize_scores(distance_to_similarity(vec_distances))

    vector_hits = []
    for rank, (doc, meta, score) in enumerate(zip(vec_docs, vec_metas, vec_scores), start=1):
        vector_hits.append({
            "text": doc,
            "metadata": meta,
            "rank": rank,
            "score": float(score),
        })

    # ---------- BM25 SEARCH ----------
    tokenized_query = tokenize(rewritten)
    bm25_raw        = bm25.get_scores(tokenized_query)
    bm25_scores     = normalize_scores(bm25_raw)

    top_indices = np.argsort(bm25_raw)[::-1][:max(top_k, CANDIDATE_K)]

    bm25_hits = []
    for rank, idx in enumerate(top_indices, start=1):
        bm25_hits.append({
            "text": documents[idx],
            "metadata": metadatas[idx],
            "rank": rank,
            "score": float(bm25_scores[idx]),
        })

    # ---------- FUSION SCORING ----------
    merged = {}

    for r in vector_hits:
        key = _key(r)
        item = merged.setdefault(key, {
            "text": r["text"],
            "metadata": r["metadata"],
            "vector_score": 0.0,
            "bm25_score": 0.0,
            "best_text": r["text"],
            "best_local_score": r["score"],
        })

        item["vector_score"] = max(item["vector_score"], r["score"])
        if r["score"] > item["best_local_score"]:
            item["best_local_score"] = r["score"]
            item["best_text"] = r["text"]

    for r in bm25_hits:
        key = _key(r)
        item = merged.setdefault(key, {
            "text": r["text"],
            "metadata": r["metadata"],
            "vector_score": 0.0,
            "bm25_score": 0.0,
            "best_text": r["text"],
            "best_local_score": r["score"],
        })

        item["bm25_score"] = max(item["bm25_score"], r["score"])
        if r["score"] > item["best_local_score"]:
            item["best_local_score"] = r["score"]
            item["best_text"] = r["text"]

    # ---------- BUILD RESULTS ----------
    results = []

    for item in merged.values():
        meta = item["metadata"]
        score = (
            item["vector_score"] * VECTOR_WEIGHT
            + item["bm25_score"] * BM25_WEIGHT
        )

        section_number = meta.get("section_number")
        if section_query and str(section_number) == str(section_query):
            score += SECTION_BOOST

        if act_query and act_query.lower() in meta.get("act_name", "").lower():
            score += ACT_BOOST

        results.append({
            "text":           item["best_text"],
            "act_name":       meta["act_name"],
            "section_number": meta["section_number"],
            "section_title":  meta["section_title"],
            "chapter":        meta["chapter"],
            "chunk_type":     meta["chunk_type"],
            "subsection":     meta.get("subsection", ""),
            "clause":         meta.get("clause", ""),
            "vector_score":   round(item["vector_score"], 6),
            "bm25_score":     round(item["bm25_score"], 6),
            "similarity":     round(score, 6),
        })

    results = sorted(results, key=lambda x: x["similarity"], reverse=True)[:top_k]
    results = dedup_best(results)
    # ---------- RERANKING ----------
    results = rerank(query, results, top_k=top_k)
    if mode == "section_only":
        results = [r for r in results if r["chunk_type"] == "section_full"]
        return results[:top_k]
    return results


# ───────────────────────────────────────────────
# RERANKING
# ───────────────────────────────────────────────

def rerank(query: str, results: list, top_k: int | None = RERANK_TOP_K) -> list:
    """Cross-encoder reranking on top fusion results."""
    if not results:
        return results

    pairs  = [(query, _strip_doc_prefix(r["text"])) for r in results]
    scores = RERANKER.predict(pairs)

    for r, s in zip(results, scores):
        r["rerank_score"] = round(float(s), 4)

    results.sort(key=lambda x: x["rerank_score"], reverse=True)
    return results if top_k is None else results[:top_k]


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
        print("-" * 60)
        results = retrieve(t)
        for r in results:
            print(
                f"[fusion={r['similarity']:.3f} rerank={r.get('rerank_score', 0):.3f}] "
                f"{r['act_name']} Section {r['section_number']}"
            )
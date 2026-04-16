

import chromadb
import numpy as np
import re
from functools import lru_cache
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


# ───────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────

DB_PATH    = r"D:\College\legal_rag\legal_rag\db"
COLLECTION = "indian_acts"

VECTOR_WEIGHT = 0.4
BM25_WEIGHT   = 0.6
TOP_K         = 50
CANDIDATE_K   = 300
SECTION_BOOST = 1.0
ACT_BOOST     = 0.5
RERANK_TOP_K  = 20

RERANK_BATCH  = 64


# ───────────────────────────────────────────────
# MODELS  
# ───────────────────────────────────────────────

EMBEDDINGS    = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
RERANKER      = CrossEncoder("BAAI/bge-reranker-base")
rewrite_llm   = ChatOllama(model="phi", temperature=0.0)


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
    lines = text.split("\n", 2)
    return lines[2] if len(lines) >= 3 else text

def tokenize(text: str) -> list:
    return re.findall(r"\w+", text.lower())

raw_clause_texts = [_strip_doc_prefix(d) for d in documents]
tokenized_corpus = [tokenize(t) for t in raw_clause_texts]
bm25             = BM25Okapi(tokenized_corpus)



# ───────────────────────────────────────────────
# QUERY REWRITING  — cached
# ───────────────────────────────────────────────

@lru_cache(maxsize=512)
def rewrite_query(query: str) -> str:
    prompt = (
        "You are a legal search assistant for Indian law. "
        "Rewrite the following query to include relevant legal keywords and terms, "
        "keep it concise and similar length, do not add specific act names or section numbers. "
        "Return ONLY the rewritten query, nothing else.\n\n"
        f"Query: {query}"
    )
    try:
        resp      = rewrite_llm.invoke([HumanMessage(content=prompt)])
        rewritten = resp.content.strip()
        print(f"[Query Rewrite] '{query[:60]}' → '{rewritten[:60]}'")
        return rewritten
    except Exception as e:
        print(f"[Query Rewrite] Failed ({e}), using original.")
        return query


def rewrite_queries_batch(queries: List[str]) -> List[str]:
    """
    Rewrite a list of queries, using the LRU cache to skip duplicates.
    Ollama has no native batch API, so calls are sequential — but the cache
    means each unique query string hits the model exactly once.
    """
    return [rewrite_query(q) for q in queries]


# ───────────────────────────────────────────────
# EMBED — batched
# ───────────────────────────────────────────────

def embed_queries_batch(queries: List[str]) -> List[List[float]]:
    """
    Embed ALL queries in a single HuggingFace forward pass.
    HuggingFaceEmbeddings.embed_documents() pads & batches automatically.
    For N=40 clauses this is ~40x faster than N calls to embed_query().
    """
    return EMBEDDINGS.embed_documents(queries)


# ───────────────────────────────────────────────
# DETECT HELPERS
# ───────────────────────────────────────────────

def detect_section(query: str):
    m = re.search(r"section\s+(\d+)", query.lower())
    return int(m.group(1)) if m else None

def detect_act(query: str):
    acts = set(meta["act_name"] for meta in metadatas)
    q    = query.lower()
    for act in acts:
        if act.lower() in q:
            return act
    return None


# ───────────────────────────────────────────────
# NORMALIZATION
# ───────────────────────────────────────────────

def normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return np.ones_like(scores)
    return (scores - lo) / (hi - lo)

def distance_to_similarity(distances) -> np.ndarray:
    return 1.0 / (1.0 + np.asarray(distances, dtype=float))


# ───────────────────────────────────────────────
# FUSION + DEDUP  (unchanged logic, extracted for reuse)
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

def dedup_best(results: list) -> list:
    best = {}
    for r in results:
        key   = (r["act_name"], r["section_number"])
        score = r.get("similarity", 0)
        if key not in best or score > best[key].get("similarity", 0):
            best[key] = r
    return list(best.values())

def _fuse_and_rank(
    vec_docs, vec_metas, vec_scores,
    bm25_indices, bm25_scores_norm, bm25_scores_raw,
    section_query, act_query,
    top_k: int,
) -> list:
    """Pure fusion logic, separated so retrieve_many() can call it per clause."""
    merged = {}

    for rank, (doc, meta, score) in enumerate(zip(vec_docs, vec_metas, vec_scores), 1):
        key  = (meta.get("act_name",""), meta.get("section_number",""),
                meta.get("chunk_type",""), meta.get("subsection",""), meta.get("clause",""))
        item = merged.setdefault(key, {
            "text": doc, "metadata": meta,
            "vector_score": 0.0, "bm25_score": 0.0,
            "best_text": doc, "best_local_score": float(score),
        })
        item["vector_score"] = max(item["vector_score"], float(score))
        if float(score) > item["best_local_score"]:
            item["best_local_score"] = float(score)
            item["best_text"] = doc

    for rank, idx in enumerate(bm25_indices, 1):
        doc  = documents[idx]
        meta = metadatas[idx]
        s    = float(bm25_scores_norm[idx])
        key  = (meta.get("act_name",""), meta.get("section_number",""),
                meta.get("chunk_type",""), meta.get("subsection",""), meta.get("clause",""))
        item = merged.setdefault(key, {
            "text": doc, "metadata": meta,
            "vector_score": 0.0, "bm25_score": 0.0,
            "best_text": doc, "best_local_score": s,
        })
        item["bm25_score"] = max(item["bm25_score"], s)
        if s > item["best_local_score"]:
            item["best_local_score"] = s
            item["best_text"] = doc

    results = []
    for item in merged.values():
        meta  = item["metadata"]
        score = item["vector_score"] * VECTOR_WEIGHT + item["bm25_score"] * BM25_WEIGHT
        if section_query and str(meta.get("section_number")) == str(section_query):
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

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return dedup_best(results[:top_k])


# ───────────────────────────────────────────────
# RERANKING  — batched
# ───────────────────────────────────────────────

def rerank(query: str, results: list, top_k: int = RERANK_TOP_K) -> list:
    """Cross-encoder reranking with explicit batch_size for GPU efficiency."""
    if not results:
        return results
    pairs  = [(query, _strip_doc_prefix(r["text"])) for r in results]
    scores = RERANKER.predict(pairs, batch_size=RERANK_BATCH)
    for r, s in zip(results, scores):
        r["rerank_score"] = round(float(s), 4)
    results.sort(key=lambda x: x["rerank_score"], reverse=True)
    return results if top_k is None else results[:top_k]


# ───────────────────────────────────────────────
# SINGLE RETRIEVE  (unchanged public API)
# ───────────────────────────────────────────────

def retrieve(query: str, mode: str = "default", top_k: int = TOP_K) -> list:
    # rewritten     = rewrite_query(query)
    rewritten     = query 
    section_query = detect_section(rewritten) or detect_section(query)
    act_query     = detect_act(rewritten)

    query_vec  = EMBEDDINGS.embed_query(rewritten)
    vec_result = collection.query(
        query_embeddings=[query_vec],
        n_results=max(top_k, CANDIDATE_K),
        include=["documents", "metadatas", "distances"],
    )
    vec_docs      = vec_result["documents"][0]
    vec_metas     = vec_result["metadatas"][0]
    vec_scores    = normalize_scores(distance_to_similarity(vec_result["distances"][0]))

    bm25_raw      = bm25.get_scores(tokenize(rewritten))
    bm25_norm     = normalize_scores(bm25_raw)
    bm25_indices  = np.argsort(bm25_raw)[::-1][:max(top_k, CANDIDATE_K)]

    results = _fuse_and_rank(
        vec_docs, vec_metas, vec_scores,
        bm25_indices, bm25_norm, bm25_raw,
        section_query, act_query, top_k,
    )
    # results = rerank(query, results, top_k=top_k)

    if mode == "section_only":
        results = [r for r in results if r["chunk_type"] == "section_full"]
    return results[:top_k]


# ───────────────────────────────────────────────
# BATCH RETRIEVE  
# ───────────────────────────────────────────────

def retrieve_many(queries: List[str], mode: str = "default", top_k: int = TOP_K) -> List[list]:
    """
    Retrieve for a list of clauses efficiently.

    What's batched:
      • Query rewriting  — sequential but LRU-cached (Ollama has no batch API)
      • Embeddings       — ONE forward pass for all N queries  ← biggest win
      • BM25 scores      — vectorized numpy for all N queries at once
      • Reranking        — one CrossEncoder call per clause but with
                           batch_size set so the GPU pipeline stays full

    What stays per-clause (can't avoid):
      • ChromaDB query   — no multi-query batch API in chromadb
      • Fusion + dedup   — inherently per-result-set

    Returns a list of result lists, one per input query, in the same order.
    """
    n = len(queries)
    if n == 0:
        return []

    # rewritten_list = rewrite_queries_batch(queries)

    # ── Step 2: Embed ALL rewritten queries in ONE forward pass ───────────
    print(f"[retrieve_many] Embedding {n} queries in one batch...")
    # all_vecs = embed_queries_batch(rewritten_list)   # List[List[float]], len=n
    all_vecs = embed_queries_batch(queries)  # skipping rewriting for now to isolate BM25 performance
    
    print(f"[retrieve_many] Computing BM25 scores for {n} queries...")
    # tokenized_queries = [tokenize(r) for r in rewritten_list]
    tokenized_queries = [tokenize(q) for q in queries]  # skipping rewriting for now to isolate BM25 performance
    bm25_raws  = np.stack([bm25.get_scores(tq) for tq in tokenized_queries])  # (N, corpus)
    bm25_norms = np.apply_along_axis(normalize_scores, 1, bm25_raws)          # (N, corpus)

    all_results = []
    for i, (query, rewritten, vec, bm25_raw, bm25_norm) in enumerate(
        # zip(queries, rewritten_list, all_vecs, bm25_raws, bm25_norms)
        zip(queries, queries, all_vecs, bm25_raws, bm25_norms)  # skipping rewriting for now to isolate BM25 performance
    ):
        print(f"[retrieve_many] Clause {i+1}/{n}: Chroma query + fusion + rerank...")

        section_query = detect_section(rewritten) or detect_section(query)
        act_query     = detect_act(rewritten)

        # Chroma query (one call per clause — unavoidable)
        vec_result = collection.query(
            query_embeddings=[vec],
            n_results=max(top_k, CANDIDATE_K),
            include=["documents", "metadatas", "distances"],
        )
        vec_docs   = vec_result["documents"][0]
        vec_metas  = vec_result["metadatas"][0]
        vec_scores = normalize_scores(distance_to_similarity(vec_result["distances"][0]))

        bm25_indices = np.argsort(bm25_raw)[::-1][:max(top_k, CANDIDATE_K)]

        results = _fuse_and_rank(
            vec_docs, vec_metas, vec_scores,
            bm25_indices, bm25_norm, bm25_raw,
            section_query, act_query, top_k,
        )
        results = rerank(query, results, top_k=top_k)

        if mode == "section_only":
            results = [r for r in results if r["chunk_type"] == "section_full"]

        all_results.append(results[:top_k])

    return all_results


# ───────────────────────────────────────────────
# FORMAT FOR PROMPT
# ───────────────────────────────────────────────

def format_for_prompt(results: list) -> str:
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
    print("\n=== retrieve_many test ===")
    batch_results = retrieve_many(tests)
    for query, results in zip(tests, batch_results):
        print(f"\nQUERY: {query}")
        print("-" * 60)
        for r in results:
            print(
                f"[fusion={r['similarity']:.3f} rerank={r.get('rerank_score',0):.3f}] "
                f"{r['act_name']} Section {r['section_number']}"
            )
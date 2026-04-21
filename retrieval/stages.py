"""
retrieval/stages.py
===================
All five pipeline stages live here:
  Stage 1 — Query expansion         (Gemini Flash)
  Stage 2 — Wide-net search         (Vector + BM25 w/ pickle cache)
  Stage 3 — Reciprocal Rank Fusion
  Stage 4 — LLM re-ranking          (Gemini Flash)
  Stage 5 — Cross-encoder rerank    (auto-disables if too slow)
"""

import re
import json
import time
import pickle
import hashlib
import numpy as np

from rank_bm25 import BM25Okapi
from langchain_core.messages import HumanMessage, SystemMessage

from retrieval.config import (
    COLLECTION, BM25_CACHE_PATH,
    CANDIDATE_K, RRF_K, FUSION_TOP_N, LLM_TOP_N,
    SECTION_BOOST, ACT_BOOST, RERANKER_TIMEOUT_S,
    EMBEDDINGS, RERANKER, gemini_llm, collection,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())

def _hit_key(hit: dict) -> tuple:
    m = hit["metadata"]
    return (
        m.get("act_name", ""),
        str(m.get("section_number", "")),
        m.get("chunk_type", ""),
        m.get("subsection", ""),
        m.get("clause", ""),
    )

def detect_section(query: str) -> int | None:
    m = re.search(r"section\s+(\d+)", query.lower())
    return int(m.group(1)) if m else None

def detect_act(query: str, metadatas: list[dict]) -> str | None:
    acts = {meta["act_name"] for meta in metadatas}
    q    = query.lower()
    for act in acts:
        if act.lower() in q:
            return act
    return None


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 SETUP — BM25 pickle cache
# ─────────────────────────────────────────────────────────────────────────────

def _corpus_fingerprint(doc_count: int) -> str:
    return hashlib.sha1(f"{COLLECTION}:{doc_count}".encode()).hexdigest()

def load_or_build_bm25() -> tuple[BM25Okapi, list[str], list[dict]]:
    """Load BM25 from disk cache; build and save if missing or stale."""
    count       = collection.count()
    fingerprint = _corpus_fingerprint(count)

    if __import__("os").path.exists(BM25_CACHE_PATH):
        try:
            with open(BM25_CACHE_PATH, "rb") as f:
                cached = pickle.load(f)
            if cached.get("fingerprint") == fingerprint:
                print(f"[BM25] Cache hit — {count} chunks loaded.")
                return cached["bm25"], cached["documents"], cached["metadatas"]
            print("[BM25] Fingerprint mismatch — rebuilding.")
        except Exception as e:
            print(f"[BM25] Cache read error ({e}) — rebuilding.")

    print(f"[BM25] Building index for {count} chunks (runs once)…")
    t0   = time.perf_counter()
    data  = collection.get(include=["documents", "metadatas"])
    docs, metas = data["documents"], data["metadatas"]
    index = BM25Okapi([tokenize(d) for d in docs])
    print(f"[BM25] Built in {time.perf_counter() - t0:.1f}s — saving cache.")

    try:
        with open(BM25_CACHE_PATH, "wb") as f:
            pickle.dump({"fingerprint": fingerprint, "bm25": index,
                         "documents": docs, "metadatas": metas}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"[BM25] WARNING: could not save cache ({e}).")

    return index, docs, metas


# Load at module import — only happens once per process
bm25, documents, metadatas = load_or_build_bm25()


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — Query expansion
# ─────────────────────────────────────────────────────────────────────────────

_EXPANSION_SYSTEM = (
    "You are a legal terminology assistant for Indian law. "
    "Output ONLY a comma-separated list of 6-10 formal legal keywords "
    "matching the user's question. No section numbers, no act names, no advice."
)

def expand_query(query: str) -> str:
    try:
        resp     = gemini_llm.invoke([SystemMessage(content=_EXPANSION_SYSTEM),
                                      HumanMessage(content=f"Query: {query}")])
        keywords = resp.content.strip()
        print(f"[Stage 1] {query!r} → keywords: {keywords}")
        return f"{query} {keywords}"
    except Exception as e:
        print(f"[Stage 1] Expansion failed ({e}) — using original query.")
        return query


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — Wide-net search
# ─────────────────────────────────────────────────────────────────────────────

def vector_hits(query_text: str, k: int = CANDIDATE_K) -> list[dict]:
    vec = EMBEDDINGS.embed_query(query_text)
    res = collection.query(query_embeddings=[vec], n_results=k,
                           include=["documents", "metadatas", "distances"])
    return [
        {"rank": i + 1, "text": doc, "metadata": meta, "raw_score": dist}
        for i, (doc, meta, dist) in enumerate(
            zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        )
    ]

def bm25_hits(query_text: str, k: int = CANDIDATE_K) -> list[dict]:
    scores  = bm25.get_scores(tokenize(query_text))
    indices = np.argsort(scores)[::-1][:k]
    return [
        {"rank": i + 1, "text": documents[idx],
         "metadata": metadatas[idx], "raw_score": float(scores[idx])}
        for i, idx in enumerate(indices)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — Reciprocal Rank Fusion
# ─────────────────────────────────────────────────────────────────────────────

def rrf_fuse(vec: list[dict], bm25_res: list[dict],
             section_q: int | None = None,
             act_q: str | None = None) -> list[dict]:
    scores: dict[tuple, float] = {}
    best:   dict[tuple, dict]  = {}

    for hit in vec + bm25_res:
        key = _hit_key(hit)
        scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + hit["rank"])
        if key not in best or hit["rank"] < best[key]["rank"]:
            best[key] = hit

    results = []
    for key, rrf_score in scores.items():
        meta = best[key]["metadata"]
        if section_q and str(meta.get("section_number", "")) == str(section_q):
            rrf_score += SECTION_BOOST
        if act_q and act_q.lower() in meta.get("act_name", "").lower():
            rrf_score += ACT_BOOST

        results.append({
            "text":           best[key]["text"],
            "act_name":       meta.get("act_name", ""),
            "section_number": meta.get("section_number", ""),
            "section_title":  meta.get("section_title", ""),
            "chapter":        meta.get("chapter", ""),
            "chunk_type":     meta.get("chunk_type", ""),
            "subsection":     meta.get("subsection", ""),
            "clause":         meta.get("clause", ""),
            "rrf_score":      round(rrf_score, 6),
        })

    return sorted(results, key=lambda x: x["rrf_score"], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — LLM re-ranking
# ─────────────────────────────────────────────────────────────────────────────

_RERANK_SYSTEM = (
    "You are a legal relevance judge for Indian law. "
    "Given a query and numbered snippets, return ONLY a JSON array of IDs "
    "ordered best-first (e.g. [3,1,7]). Omit irrelevant ones. No other text."
)

def llm_rerank(query: str, candidates: list[dict],
               top_n: int = LLM_TOP_N) -> list[dict]:
    if not candidates:
        return candidates

    snippets = "\n\n".join(
        f"[{i}] {r['act_name']} § {r['section_number']}\n{r['text'][:600]}"
        for i, r in enumerate(candidates, 1)
    )
    prompt = (f"User query: {query}\n\nSnippets:\n{snippets}\n\n"
              f"Return a JSON array of the most relevant IDs, best-first. "
              f"Max {top_n} IDs.")

    try:
        resp = gemini_llm.invoke([SystemMessage(content=_RERANK_SYSTEM),
                                  HumanMessage(content=prompt)])
        raw        = re.sub(r"```json|```", "", resp.content.strip()).strip()
        ranked_ids = json.loads(raw)
        print(f"[Stage 4] Gemini order: {ranked_ids}")

        seen, reranked = set(), []
        for idx in ranked_ids:
            if 1 <= idx <= len(candidates) and idx not in seen:
                seen.add(idx)
                entry = dict(candidates[idx - 1])
                entry["llm_rank"] = len(reranked) + 1
                reranked.append(entry)
        return reranked

    except Exception as e:
        print(f"[Stage 4] LLM rerank failed ({e}) — keeping RRF order.")
        return candidates


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — Cross-encoder rerank (auto-disables if too slow)
# ─────────────────────────────────────────────────────────────────────────────

_reranker_enabled = True

def cross_encoder_rerank(query: str, results: list[dict],
                         top_k: int = 5) -> list[dict]:
    global _reranker_enabled

    if not results or not _reranker_enabled:
        if not _reranker_enabled:
            print("[Stage 5] Skipped — disabled after timeout.")
        return results[:top_k]

    t0     = time.perf_counter()
    scores = RERANKER.predict([(query, r["text"]) for r in results])
    elapsed = time.perf_counter() - t0
    print(f"[Stage 5] Cross-encoder: {elapsed:.2f}s for {len(results)} pairs.")

    if RERANKER_TIMEOUT_S is not None and elapsed > RERANKER_TIMEOUT_S:
        _reranker_enabled = False
        print(f"[Stage 5] {elapsed:.2f}s > {RERANKER_TIMEOUT_S}s threshold — "
              "Stage 5 DISABLED for this session.")

    for r, s in zip(results, scores):
        r["rerank_score"] = round(float(s), 4)

    return sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
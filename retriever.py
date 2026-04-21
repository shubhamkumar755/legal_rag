"""
retriever.py
============
Thin orchestrator — calls each stage in order.
This is the only file your generator / verifier needs to import.

Usage
-----
    from retriever import retrieve, format_for_prompt

    results = retrieve("what to do if my phone is stolen")
    context = format_for_prompt(results)
"""

from retrieval.config import CANDIDATE_K, FUSION_TOP_N, LLM_TOP_N, FINAL_TOP_K
from retrieval.stages import (
    expand_query,
    vector_hits, bm25_hits,
    rrf_fuse,
    llm_rerank,
    cross_encoder_rerank,
    detect_section, detect_act, metadatas,
)


def retrieve(query: str, mode: str = "default",
             top_k: int = FINAL_TOP_K) -> list[dict]:
    """
    Full 5-stage retrieval pipeline.

    Stages
    ------
    1. Query expansion   — layman → legal keywords  (Gemini Flash)
    2. Wide-net search   — Vector + BM25  (100 candidates each)
    3. RRF fusion        — merge + deduplicate
    4. LLM re-ranking    — Gemini picks best 10 from top 20
    5. Cross-encoder     — final precision pass (auto-disables if slow)
    """
    # Structural hints from the raw query (before expansion)
    section_q = detect_section(query)
    act_q     = detect_act(query, metadatas)

    # Stage 1
    expanded = expand_query(query)

    # Stage 2 + 3
    print(f"[Stage 2] Fetching {CANDIDATE_K} candidates from Vector + BM25…")
    fused = rrf_fuse(
        vector_hits(expanded), bm25_hits(expanded),
        section_q=section_q, act_q=act_q,
    )
    print(f"[Stage 3] {len(fused)} unique candidates after RRF fusion.")

    # Stage 4
    reranked = llm_rerank(query, fused[:FUSION_TOP_N], top_n=LLM_TOP_N)

    # Stage 5
    final = cross_encoder_rerank(query, reranked, top_k=top_k)
    print(f"[Done] Returning {len(final)} results.")

    if mode == "section_only":
        final = [r for r in final if r.get("chunk_type") == "section_full"]

    return final


def format_for_prompt(results: list[dict]) -> str:
    """Formats results as a numbered citation block for your LLM prompt."""
    lines = []
    for i, r in enumerate(results, 1):
        citation = f"{r['act_name']}, Section {r['section_number']}"
        if r.get("subsection"):
            citation += f"({r['subsection']})"
        if r.get("clause"):
            citation += f"({r['clause']})"
        lines.append(f"[{i}] {citation}")
        lines.append(f"     {r['text']}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    tests = [
        "what to do if my phone is stolen",
        "tenant subletting without landlord permission",
        "personal data leak from website",
        "being assaulted in a public place, what can I do?",
    ]
    for t in tests:
        print("\n" + "═" * 70)
        print(f"QUERY: {t}")
        print("─" * 70)
        for r in retrieve(t):
            print(
                f"  [rrf={r.get('rrf_score', 0):.4f}  "
                f"llm={r.get('llm_rank', '—')}  "
                f"rerank={r.get('rerank_score', 0):.4f}]  "
                f"{r['act_name']} § {r['section_number']} — {r['section_title']}"
            )
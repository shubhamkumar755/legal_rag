"""
pipeline.py — OPTIMIZED
Orchestrates: retriever → explainer → verifier

OPTIMIZATIONS vs original:
  - run()       kept for single-clause backward compatibility.
  - run_many()  new function: processes a list of clauses with batched
                retrieval (one embed pass) and batched explanation
                (N/5 Ollama calls instead of N).
                verifier.verify() is called per-clause (assumed fast/local).
"""

from batch.retriever import retrieve, retrieve_many
from batch.explainer import explain, explain_many
from batch.verifier  import verify
from typing import List, Dict, Any


# ── SINGLE (unchanged public API) ────────────────────────────────────────────

def run(clause_text: str) -> dict:
    """Full pipeline for one clause. Returns result dict ready for app.py."""
    retrieved    = retrieve(clause_text)
    explanation  = explain(clause_text, retrieved)
    verification = verify(explanation, retrieved)
    return {
        "clause":      clause_text,
        "explanation": explanation,
        "retrieved":   retrieved,
        **verification,
    }


# ── BATCH  ← drop-in replacement for calling run() in a loop ─────────────────

def run_many(clauses: List[str]) -> List[Dict[str, Any]]:
    """
    Process a list of clauses with maximum batching.

    Call graph:
      retrieve_many()  → 1 embed pass + N Chroma queries + N reranks
      explain_many()   → ceil(N/5) Ollama calls  (was N calls)
      verify()         → N calls (assumed fast/CPU-only)

    For a 40-clause document with Mistral locally:
      Before: ~40 rewrite + 40 embed + 40 rerank + 40 explain Ollama calls
      After:  ~40 rewrite (cached after first run) + 1 embed + 40 rerank + 8 explain calls
    """
    n = len(clauses)
    if n == 0:
        return []

    print(f"[run_many] Step 1/3: Retrieving for {n} clauses (batched embed)...")
    all_retrieved = retrieve_many(clauses)

    print(f"[run_many] Step 2/3: Explaining {n} clauses (batched LLM)...")
    all_explanations = explain_many(clauses, all_retrieved)

    print(f"[run_many] Step 3/3: Verifying {n} clauses...")
    results = []
    for clause, retrieved, explanation in zip(clauses, all_retrieved, all_explanations):
        verification = verify(explanation, retrieved)
        results.append({
            "clause":      clause,
            "explanation": explanation,
            "retrieved":   retrieved,
            **verification,
        })

    return results
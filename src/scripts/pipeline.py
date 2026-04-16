"""
pipeline.py — Orchestrates: retriever → explainer → verifier
Only entry point for app.py
"""

from scripts.retriever import retrieve
from scripts.explainer import explain
from scripts.verifier  import verify


def run(clause_text: str) -> dict:
    """Full pipeline. Returns result dict ready for app.py."""

    retrieved   = retrieve(clause_text)
    print(f"Retrieved {len(retrieved)} sections.")
    explanation = explain(clause_text, retrieved)
    print(f"Generated explanation: {explanation[:100]}...")
    verification = verify(explanation, retrieved)

    return {
        "clause":      clause_text,
        "explanation": explanation,
        "retrieved":   retrieved,
        **verification,   # citations, confidence, risk_flag
    }
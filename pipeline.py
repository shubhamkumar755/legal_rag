"""
pipeline.py
-----------
Unified entry point for the Legal AI system.

Two public functions:
    answer_query(query)          → GeneratorResult
    analyze_document(file_path)  → list[dict]

Usage
-----
    from pipeline import answer_query, analyze_document

    # Plain legal question
    result = answer_query("What happens if my landlord refuses to return the deposit?")
    print(result.response)

    # Full document review
    report = analyze_document("rental_agreement.pdf")
    for clause in report:
        print(clause["explanation"])
"""

from __future__ import annotations

import logging
from typing import Any

from generator import GeneratorResult, generate_answer
from document_processor import process_legal_document

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def answer_query(
    query: str,
    top_k: int = 5,
    verify: bool = True,
) -> GeneratorResult:
    """
    Route a plain-language legal question through the RAG generator.

    Parameters
    ----------
    query   : The user's legal question (e.g. "What to do if my phone is stolen?")
    top_k   : Number of retrieval snippets to surface (default 5).
    verify  : Whether to run citation verification (default True).

    Returns
    -------
    GeneratorResult
        .response          — Final answer with disclaimer appended.
        .snippets          — Retrieved legal chunks used as context.
        .verification_data — Confidence score, risk flags, etc.
        .warnings          — Any hallucination / citation warnings.
        .raw_llm_output    — Unprocessed model output (before disclaimer).
    """
    logger.info("answer_query called | query=%r | top_k=%d", query, top_k)

    if not query or not query.strip():
        raise ValueError("query must be a non-empty string.")

    result = generate_answer(query.strip(), top_k=top_k, verify=verify)

    logger.info(
        "answer_query completed | confidence=%s | warnings=%d",
        result.verification_data.get("confidence", {}).get("label", "N/A"),
        len(result.warnings),
    )
    return result


def analyze_document(file_path: str) -> list[dict[str, Any]]:
    """
    Route a legal document through the full extraction → segmentation → RAG pipeline.

    Parameters
    ----------
    file_path : Absolute or relative path to a PDF, DOCX, or plain-text file.

    Returns
    -------
    list[dict]  — One entry per clause, each containing:
        clause_id     : int   — 1-based position in the document.
        original_text : str   — Raw clause text extracted from the file.
        explanation   : str   — RAG-grounded analysis with disclaimer.
        citations     : list  — Verified citation objects from verification_data.
        warnings      : list  — Hallucination / citation warnings, if any.
        error         : str   — Present only when clause analysis failed.
    """
    logger.info("analyze_document called | file=%r", file_path)

    if not file_path or not file_path.strip():
        raise ValueError("file_path must be a non-empty string.")

    report = process_legal_document(file_path.strip())

    total    = len(report)
    failed   = sum(1 for r in report if "error" in r)
    logger.info(
        "analyze_document completed | clauses=%d | failed=%d", total, failed
    )
    return report


# ──────────────────────────────────────────────────────────────────────────────
# CLI demo
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # ── Demo 1: plain query ───────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("DEMO 1 — Plain Legal Query")
    print("═" * 70)

    query_result = answer_query("What to do if my phone is stolen?")
    print(f"\nQuery   : {query_result.query}")
    print(f"Response:\n{query_result.response}")
    if query_result.warnings:
        print("\n⚠️  Warnings:")
        for w in query_result.warnings:
            print(f"   {w}")

    # ── Demo 2: document analysis ─────────────────────────────────────────────
    doc_path = sys.argv[1] if len(sys.argv) > 1 else "residential-rental-agreement-format.pdf"

    print("\n" + "═" * 70)
    print(f"DEMO 2 — Document Analysis: {doc_path}")
    print("═" * 70)

    doc_report = analyze_document(doc_path)
    for item in doc_report:
        print(f"\n--- Clause {item['clause_id']} ---")
        if "error" in item:
            print(f"  Error      : {item['error']}")
        else:
            print(f"  Explanation: {item['explanation'][:300]}…")
            if item["warnings"]:
                print(f"  Warnings   : {item['warnings']}")
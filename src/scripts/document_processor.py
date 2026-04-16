"""
document_processor.py
---------------------
Pipeline orchestrator for multi-clause legal document analysis.

Steps 1–3 (text extraction, segmentation, preprocessing) are handled
by extraction.py — this file handles Steps 4–7.
"""

import logging
from typing import List, Dict, Any

from scripts.extraction import load_and_segment    
from scripts.pipeline import run        

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# STEPS 4 + 5: Pipeline Execution + Aggregation
# ─────────────────────────────────────────────

def analyze_clauses(clauses: List[str]) -> List[Dict[str, Any]]:
    """
    For each clause: run pipeline → collect structured result.
    """
    results = []

    for idx, clause in enumerate(clauses, start=1):
        logger.info(f"Processing clause {idx}/{len(clauses)}")

        # STEP 6: Error handling per clause
        try:
            pipeline_output = run(clause)

            result = {
                "clause_id":   idx,
                "clause_text": clause,
                "explanation": pipeline_output.get("explanation", ""),
                "citations":   pipeline_output.get("citations", []),
                "confidence":  pipeline_output.get("confidence", 0.0),
                "risk_flag":   pipeline_output.get("risk_flag", "LOW"),
            }

        except Exception as e:
            logger.warning(f"Clause {idx} failed: {e}")
            result = {
                "clause_id":   idx,
                "clause_text": clause,
                "error":       str(e),
            }

        results.append(result)

    return results


# ─────────────────────────────────────────────
# STEP 7: Output Formatting
# ─────────────────────────────────────────────

def build_final_output(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Wrap all clause results with a document-level summary.
    """
    successful = [r for r in results if "error" not in r]
    failed     = [r for r in results if "error" in r]

    avg_confidence = (
        sum(r["confidence"] for r in successful) / len(successful)
        if successful else 0.0
    )

    risk_levels = [r.get("risk_flag", "LOW") for r in successful]
    if "HIGH" in risk_levels:
        overall_risk = "HIGH"
    elif "MEDIUM" in risk_levels:
        overall_risk = "MEDIUM"
    else:
        overall_risk = "LOW"

    return {
        "total_clauses":   len(results),
        "processed":       len(successful),
        "failed":          len(failed),
        "overall_risk":    overall_risk,
        "avg_confidence":  round(avg_confidence, 2),
        "clause_analyses": results,
    }


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def process_document(source) -> Dict[str, Any]:
    """
    Full pipeline orchestrator.

    source: PDF path | DOCX path | raw text string
    Returns: structured legal analysis for every clause.
    """
    # Steps 1–3 delegated to extraction.py
    logger.info("Steps 1–3: Extracting and segmenting document...")
    clauses = load_and_segment(source)
    logger.info(f"  Ready: {len(clauses)} clauses")

    # Steps 4–6: Run pipeline on each clause
    logger.info("Steps 4–6: Running pipeline per clause...")
    results = analyze_clauses(clauses)

    # Step 7: Final output
    logger.info("Step 7: Building final output...")
    return build_final_output(results)
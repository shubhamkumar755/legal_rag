"""
document_processor.py — OPTIMIZED
Pipeline orchestrator for multi-clause legal document analysis.

Steps 1–3 handled by extraction.py.
Steps 4–7 handled here using run_many() for full batching.
"""

import logging
from typing import List, Dict, Any

from batch.extraction import load_and_segment
from batch.pipeline   import run_many

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── STEPS 4 + 5: Batched Pipeline Execution ──────────────────────────────────

def analyze_clauses(clauses: List[str]) -> List[Dict[str, Any]]:
    """
    Run the full pipeline on all clauses using run_many() for batching.
    Falls back to per-clause run() if run_many() fails entirely.
    """
    logger.info("Running pipeline on %d clauses (batched)...", len(clauses))

    try:
        pipeline_outputs = run_many(clauses)
    except Exception as e:
        logger.error("run_many() failed entirely: %s — falling back to per-clause", e)
        from scripts.pipeline import run
        pipeline_outputs = []
        for clause in clauses:
            try:
                pipeline_outputs.append(run(clause))
            except Exception as ce:
                pipeline_outputs.append({"error": str(ce), "clause": clause})

    results = []
    for idx, (clause, output) in enumerate(zip(clauses, pipeline_outputs), start=1):
        if "error" in output:
            results.append({"clause_id": idx, "clause_text": clause, "error": output["error"]})
        else:
            results.append({
                "clause_id":   idx,
                "clause_text": clause,
                "explanation": output.get("explanation", ""),
                "citations":   output.get("citations", []),
                "confidence":  output.get("confidence", 0.0),
                "risk_flag":   output.get("risk_flag", "LOW"),
            })

    return results


# ── STEP 7: Output Formatting ─────────────────────────────────────────────────

_RISK_PRIORITY = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}

def build_final_output(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_confidence = 0.0
    processed = failed = highest_risk = 0

    for r in results:
        if "error" in r:
            failed += 1
        else:
            processed        += 1
            total_confidence += r.get("confidence", 0.0)
            risk_val          = _RISK_PRIORITY.get(r.get("risk_flag", "LOW"), 0)
            if risk_val > highest_risk:
                highest_risk = risk_val

    return {
        "total_clauses":   len(results),
        "processed":       processed,
        "failed":          failed,
        "overall_risk":    {2: "HIGH", 1: "MEDIUM", 0: "LOW"}[highest_risk],
        "avg_confidence":  round(total_confidence / processed, 2) if processed else 0.0,
        "clause_analyses": results,
    }


# ── MAIN ENTRY POINT ──────────────────────────────────────────────────────────

def process_document(source) -> Dict[str, Any]:
    logger.info("Steps 1–3: Extracting and segmenting document...")
    clauses = load_and_segment(source)
    logger.info("  Ready: %d clauses", len(clauses))

    logger.info("Steps 4–6: Running batched pipeline...")
    results = analyze_clauses(clauses)

    logger.info("Step 7: Building final output...")
    return build_final_output(results)
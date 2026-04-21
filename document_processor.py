"""
document_analyzer.py
--------------------
Orchestrates the full document review: Extraction -> Segmentation -> RAG Analysis.
"""

import logging
from typing import List, Dict, Any
from extraction import load_and_segment    # Step 1-3
from generator import generate_answer      # Step 4-6

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_legal_document(file_path: str) -> List[Dict[str, Any]]:
    """
    1. Loads the file (PDF/DOCX/Text).
    2. Segments it into individual legal clauses.
    3. Runs each clause through the RAG generator.
    """
    # Steps 1–3: Segment the document into clean legal prose
    logger.info(f"Segmenting document: {file_path}")
    clauses = load_and_segment(file_path) #
    logger.info(f"Found {len(clauses)} clauses for analysis.")

    results = []

    # Steps 4–6: Analyze each clause individually
    for idx, clause_text in enumerate(clauses, start=1):
        logger.info(f"Processing clause {idx}/{len(clauses)}")
        
        # Frame the clause as a query for your generator
        query = f"Analyze and explain this legal clause: {clause_text}"
        
        try:
            # Call your 'new' grounded generator
            rag_result = generate_answer(query) #
            
            # Map the GeneratorResult dataclass to a report dictionary
            results.append({
                "clause_id": idx,
                "original_text": clause_text,
                "explanation": rag_result.response,
                "citations": rag_result.verification_data.get("citations", []),
                "warnings": rag_result.warnings
            })
        except Exception as e:
            logger.error(f"Failed to analyze clause {idx}: {e}")
            results.append({
                "clause_id": idx,
                "original_text": clause_text,
                "error": str(e)
            })

    return results

if __name__ == "__main__":
    # Example usage
    report = process_legal_document("residential-rental-agreement-format.pdf")
    
    # Simple report print
    for item in report:
        print(f"\n--- Clause {item['clause_id']} ---")
        if "error" in item:
            print(f"Error: {item['error']}")
        else:
            print(f"Explanation: {item['explanation']}")
            print(f"Citations: {item['citations']}")
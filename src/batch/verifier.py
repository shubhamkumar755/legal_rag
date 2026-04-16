"""
verifier.py — Explanation + retrieved sections → faithfulness, hallucination, confidence
"""

import re


# ── CONFIG ─────────────────────────────────────────────────────────

SIM_THRESHOLD = 0.6


# ── 1. EXTRACT CLAIMS (VERY IMPORTANT) ─────────────────────────────

def extract_claims(text: str):
    """
    Break explanation into atomic claims
    """
    sentences = re.split(r"[.?!]\s+", text)
    claims = [s.strip() for s in sentences if len(s.strip()) > 20]
    return claims


# ── 2. CHECK CLAIM SUPPORT ─────────────────────────────────────────

def is_claim_supported(claim: str, retrieved_sections: list[dict]) -> bool:
    claim_words = set(claim.lower().split())

    for r in retrieved_sections:
        context_words = set(r["text"].lower().split())

        overlap = claim_words & context_words

        if len(overlap) > 8:   # threshold heuristic
            return True

    return False


# ── 3. FAITHFULNESS SCORE ─────────────────────────────────────────

def compute_faithfulness(explanation: str, retrieved_sections: list[dict]):
    claims = extract_claims(explanation)

    if not claims:
        return {
            "score": 0.0,
            "supported": [],
            "unsupported": [],
        }

    supported = []
    unsupported = []

    for claim in claims:
        if is_claim_supported(claim, retrieved_sections):
            supported.append(claim)
        else:
            unsupported.append(claim)

    score = len(supported) / len(claims)

    return {
        "score": round(score, 2),
        "supported": supported,
        "unsupported": unsupported,
    }


# ── 4. CITATION EXTRACTION ────────────────────────────────────────

def extract_citations(text: str):
    pattern = re.compile(r"[Ss]ection\s+(\d+)", re.IGNORECASE)
    return list(set(pattern.findall(text)))


# ── 5. VERIFY CITATIONS ───────────────────────────────────────────

def verify_citations(citations, retrieved_sections):
    results = []

    for sec in citations:
        found = any(str(r["section_number"]) == sec for r in retrieved_sections)

        results.append({
            "section": sec,
            "status": "verified" if found else "not_found"
        })

    return results


# ── 6. CLAUSE TYPE DETECTION ──────────────────────────────────────

def detect_clause_type(text):
    text = text.lower()

    if "shall not" in text:
        return "restriction"
    if "shall" in text or "must" in text:
        return "obligation"
    if "may" in text:
        return "permission"
    if "penalty" in text or "punishable" in text:
        return "penalty"

    return "general"


# ── 7. RISK DETECTION ─────────────────────────────────────────────

def flag_risk(text):
    text = text.lower()

    risk_patterns = [
        ("terminate", "⚠️ Can be terminated"),
        ("without notice", "⚠️ No notice required"),
        ("liable", "⚠️ Legal liability risk"),
        ("penalty", "⚠️ Financial/legal penalty"),
        ("discretion", "⚠️ One-sided control"),
    ]

    for word, msg in risk_patterns:
        if word in text:
            return msg

    return None


# ── 8. FINAL CONFIDENCE ───────────────────────────────────────────

def compute_confidence(faithfulness_score):
    if faithfulness_score > 0.8:
        return {"level": "high", "label": "✅ Fully grounded"}
    elif faithfulness_score > 0.5:
        return {"level": "medium", "label": "⚠️ Partially grounded"}
    else:
        return {"level": "low", "label": "🚨 Likely hallucination"}


# ── 9. MAIN FUNCTION ──────────────────────────────────────────────

def verify(explanation: str, retrieved_sections: list[dict]):
    """
    Main verifier:
    - detects hallucination
    - computes faithfulness
    - verifies citations
    """

    faithfulness = compute_faithfulness(explanation, retrieved_sections)

    citations = extract_citations(explanation)
    citation_results = verify_citations(citations, retrieved_sections)

    clause_type = detect_clause_type(explanation)

    risk = flag_risk(explanation)

    confidence = compute_confidence(faithfulness["score"])

    return {
        "faithfulness_score": faithfulness["score"],
        "supported_claims": faithfulness["supported"],
        "hallucinated_claims": faithfulness["unsupported"],
        "citations": citation_results,
        "clause_type": clause_type,
        "confidence": confidence,
        "risk_flag": risk,
    }
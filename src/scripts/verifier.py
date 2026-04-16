"""
verifier.py — Takes explanation + retrieved sections → citation verification, confidence, risk
"""

import re


# ── CONFIG ────────────────────────────────────────────────────────────────────

SIMILARITY_VERIFIED = 0.85
SIMILARITY_PARTIAL  = 0.60

_RISK_PHRASES = [
    "unfair", "favours the landlord", "favours the employer",
    "below the standard", "less than required", "waives your right",
    "at the discretion of", "may terminate without",
]

_CITATION_PATTERNS = [
    re.compile(
        r"[Ss]ection\s+(\d+[A-Za-z]?)\s+of\s+([\w\s]+?Act[\w\s]*?)(?:\d{4})?\s*[,\.;]",
        re.IGNORECASE,
    ),
    re.compile(
        r"([\w\s]+?Act[\w\s]*?(?:\d{4})?)[,\s]+[Ss]ection\s+(\d+[A-Za-z]?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"[Uu]nder\s+([\w\s]+?Act[\w\s]*?(?:\d{4})?)[,\s]+[Ss]ection\s+(\d+[A-Za-z]?)",
        re.IGNORECASE,
    ),
]


# ── CITATION EXTRACTION ───────────────────────────────────────────────────────

def extract_citations(llm_response: str) -> list[dict]:
    citations = []
    seen = set()

    for pattern in _CITATION_PATTERNS:
        for match in pattern.finditer(llm_response):
            groups = match.groups()
            if groups[0].strip()[0].isdigit():
                section_num, act_name = groups[0].strip(), groups[1].strip()
            else:
                act_name, section_num = groups[0].strip(), groups[1].strip()

            act_name = re.sub(r"\s+", " ", act_name).strip(" ,.")
            key = (act_name.lower(), section_num)
            if key not in seen:
                seen.add(key)
                citations.append({"act": act_name, "section": section_num})

    return citations


# ── CITATION VERIFICATION ─────────────────────────────────────────────────────

def verify_citations(citations: list[dict], retrieved_sections: list[dict]) -> list[dict]:
    from scripts.retriever import retrieve

    results = []
    for citation in citations:
        match = None
        for r in retrieved_sections:
            act_match     = citation["act"].lower() in r["act_name"].lower()
            section_match = str(r["section_number"]) == citation["section"].lstrip("0")
            if act_match and section_match:
                match = r
                break

        if match:
            status     = "verified"
            similarity = match["similarity"]
        else:
            query = f"Section {citation['section']} {citation['act']}"
            fresh = retrieve(query, top_k=1)
            if not fresh:
                status, similarity = "not_found", 0.0
            else:
                similarity = fresh[0]["similarity"]
                if similarity >= SIMILARITY_VERIFIED:
                    status = "verified"
                elif similarity >= SIMILARITY_PARTIAL:
                    status = "partial"
                else:
                    status = "not_found"

        results.append({**citation, "status": status, "similarity": similarity})

    return results


# ── CONFIDENCE ────────────────────────────────────────────────────────────────

def compute_confidence(verification_results: list[dict]) -> dict:
    if not verification_results:
        return {"level": "low", "label": "🚨 No citations — potential hallucination"}

    counts = {"verified": 0, "partial": 0, "not_found": 0}
    for r in verification_results:
        counts[r["status"]] += 1

    if counts["not_found"] == 0 and counts["partial"] == 0:
        return {"level": "high",   "label": "✅ High confidence — all citations verified"}
    elif counts["not_found"] == 0:
        return {"level": "medium", "label": "⚠️  Medium confidence — some citations partial"}
    elif counts["verified"] > 0:
        return {"level": "medium", "label": "⚠️  Medium confidence — mixed verification"}
    else:
        return {"level": "low",    "label": "🚨 Low confidence — citations not found in database"}


# ── RISK FLAG ─────────────────────────────────────────────────────────────────

def flag_risk(explanation: str) -> str | None:
    low = explanation.lower()
    for phrase in _RISK_PHRASES:
        if phrase in low:
            return "⚠️  Risk detected: the clause may disadvantage you as a citizen."
    return None


# ── MAIN ──────────────────────────────────────────────────────────────────────

def verify(explanation: str, retrieved_sections: list[dict]) -> dict:
    """Takes explanation + retrieved sections. Returns verification, confidence, risk."""
    citations    = extract_citations(explanation)
    verification = verify_citations(citations, retrieved_sections)
    confidence   = compute_confidence(verification)
    risk         = flag_risk(explanation)

    return {
        "citations":  verification,
        "confidence": confidence,
        "risk_flag":  risk,
    }
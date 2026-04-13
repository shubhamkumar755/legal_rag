"""
explainer.py  —  Clause → Gemini explanation → Citation verification
---------------------------------------------------------------------
Depends on retriever.py for the retrieve() and format_for_prompt() functions.

Setup:
    pip install langchain-google-genai python-dotenv

.env:
    GEMINI_API_KEY=your_key_here
"""

import os
import re
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from retriever import retrieve, format_for_prompt

load_dotenv()
from langchain_ollama import ChatOllama




# ── CONFIG ─────────────────────────────────────────────────────────────────────

MODEL_NAME = "mistral"
SIMILARITY_VERIFIED = 0.85
SIMILARITY_PARTIAL  = 0.60

# ── LLM SETUP ─────────────────────────────────────────────────────────────────

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.1
)

# ── PROMPT TEMPLATE ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a legal assistant helping Indian citizens understand 
their documents. You explain legal clauses in simple English (Class 8 level).

Rules you must always follow:
1. ONLY use the provided law sections below — no outside knowledge.
2. Cite the EXACT Act name and Section number for every claim you make.
3. Use the format: "Under [Act Name], Section [N], ..." when citing.
4. If the provided law does not cover this clause, say exactly:
   "I could not find a relevant Indian law for this clause in my database."
5. Explain what the clause means for the common person — no legal jargon.
6. If the clause seems unfair or risky for the citizen, note it clearly.
7. NEVER make up laws or sections that aren't in the provided context.
8. Be concise and clear in your explanation.
9. Always assume the reader has no legal background.
10. Abosolutely do NOT hallucinate or infer beyond the provided sections."""


def build_prompt(clause_text: str, retrieved_sections: list[dict]) -> str:
    law_context = format_for_prompt(retrieved_sections)
    return f"""Clause to explain:
\"\"\"{clause_text}\"\"\"

Relevant Indian law sections (use ONLY these):
{law_context}

Explain this clause to a citizen in simple English. Cite Act and Section numbers."""


# ── CALL GEMINI ───────────────────────────────────────────────────────────────

def generate_explanation(clause_text: str, retrieved_sections: list[dict]) -> str:
    prompt = build_prompt(clause_text, retrieved_sections)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
    response = llm.invoke(messages)
    return response.content


# ── CITATION EXTRACTION ───────────────────────────────────────────────────────
# Matches patterns like:
#   "Section 106 of Transfer of Property Act"
#   "Transfer of Property Act, Section 106"
#   "Under the Consumer Protection Act 2019, Section 35"

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


def extract_citations(llm_response: str) -> list[dict]:
    """
    Pull out all 'Act name + Section number' mentions from the LLM response.
    Returns a list of dicts: [{"act": "...", "section": "106"}, ...]
    """
    citations = []
    seen = set()

    for pattern in _CITATION_PATTERNS:
        for match in pattern.finditer(llm_response):
            groups = match.groups()
            # Normalise: always (act_name, section_num)
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

def verify_citations(
    citations: list[dict],
    retrieved_sections: list[dict],
) -> list[dict]:
    """
    Check each citation against:
    1. The already-retrieved sections (fast, no extra DB call)
    2. Falls back to a fresh ChromaDB query if not found in retrieved set
    """
    results = []
    for citation in citations:
        query = f"Section {citation['section']} {citation['act']}"

        # First: check if it's already in the retrieved set
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
            # Fallback: query ChromaDB directly for this specific citation
            fresh = retrieve(query, top_k=1)
            if not fresh:
                status     = "not_found"
                similarity = 0.0
            else:
                similarity = fresh[0]["similarity"]
                if similarity >= SIMILARITY_VERIFIED:
                    status = "verified"
                elif similarity >= SIMILARITY_PARTIAL:
                    status = "partial"
                else:
                    status = "not_found"

        results.append({
            **citation,
            "status":     status,
            "similarity": similarity,
        })

    return results


# ── CONFIDENCE SCORE ──────────────────────────────────────────────────────────

def compute_confidence(verification_results: list[dict]) -> dict:
    """
    Aggregate individual citation statuses into an overall confidence level.
    """
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

_RISK_PHRASES = [
    "unfair", "favours the landlord", "favours the employer",
    "below the standard", "less than required", "waives your right",
    "at the discretion of", "may terminate without",
]


def flag_risk(explanation: str) -> str | None:
    low = explanation.lower()
    for phrase in _RISK_PHRASES:
        if phrase in low:
            return "⚠️  Risk detected: the clause may disadvantage you as a citizen."
    return None


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def explain_clause(clause_text: str) -> dict:
    """
    Full pipeline for one clause.
    Returns a structured result dict ready for display or Streamlit rendering.
    """
    retrieved   = retrieve(clause_text,3)
    explanation = generate_explanation(clause_text, retrieved)
    citations   = extract_citations(explanation)
    verification = verify_citations(citations, retrieved)
    confidence  = compute_confidence(verification)
    risk        = flag_risk(explanation)

    return {
        "clause":      clause_text,
        "explanation": explanation,
        "citations":   verification,
        "confidence":  confidence,
        "risk_flag":   risk,
        "retrieved":   retrieved,
    }


def format_output(result: dict) -> str:
    """Pretty-print a single clause result to the terminal."""
    lines = [
        "─" * 60,
        f"CLAUSE:\n  {result['clause']}",
        "",
        f"PLAIN ENGLISH EXPLANATION:\n{result['explanation']}",
        "",
        "CITATIONS VERIFIED:",
    ]
    for c in result["citations"]:
        icon = {"verified": "✅", "partial": "⚠️ ", "not_found": "🚨"}.get(c["status"], "?")
        lines.append(f"  {icon} {c['act']}, Section {c['section']}  (similarity: {c['similarity']:.3f})")

    lines += [
        "",
        f"CONFIDENCE: {result['confidence']['label']}",
    ]
    if result["risk_flag"]:
        lines.append(result["risk_flag"])

    return "\n".join(lines)


# ── QUICK TEST ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_clauses = [
        "The landlord is removing me without any notice",
        "My personal data was found on a shady website, can I do anything?"
    ]

    for clause in test_clauses:
        result = explain_clause(clause)
        print(format_output(result))
        print()
"""
explainer.py — Takes a legal clause + retrieved sections → returns explanation text
"""

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama


# ── CONFIG ────────────────────────────────────────────────────────────────────

MODEL_NAME = "phi"

llm = ChatOllama(model=MODEL_NAME, temperature=0.1)

SYSTEM_PROMPT = """You are a legal assistant helping Indian citizens understand 
their documents. You explain legal clauses in simple English (Class 8 level).
IMPORTANT: ONLY use the provided law sections below — no outside knowledge. Cite the EXACT Act name and Section number for every claim you make. 
NEVER MAKE UP ANYTHING
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
10. Absolutely do NOT hallucinate or infer beyond the provided sections."""


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _build_prompt(clause_text: str, retrieved_sections: list[dict]) -> str:
    from scripts.retriever import format_for_prompt
    law_context = format_for_prompt(retrieved_sections)
    return f"""Clause to explain:
\"\"\"{clause_text}\"\"\"

Relevant Indian law sections (use ONLY these):
{law_context}

Explain this clause to a citizen in simple English. Cite Act and Section numbers."""


# ── MAIN ──────────────────────────────────────────────────────────────────────

def explain(clause_text: str, retrieved_sections: list[dict]) -> str:
    """Takes clause + retrieved sections. Returns plain-English explanation."""
    prompt = _build_prompt(clause_text, retrieved_sections)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
    response = llm.invoke(messages)
    return response.content
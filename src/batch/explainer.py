"""
explainer.py — OPTIMIZED
Takes legal clauses + retrieved sections → returns explanation text.

OPTIMIZATIONS vs original:
  1. explain_many()  → Sends ALL clauses to Ollama in one structured prompt
                       and parses the responses back out. One LLM round-trip
                       instead of N. This is the single biggest speedup.
  2. explain()       → Kept for backward compatibility / single-clause use.
  3. Prompt building → format_for_prompt() imported once, not re-imported
                       on every call.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from batch.retriever import format_for_prompt
from typing import List


# ── CONFIG ────────────────────────────────────────────────────────────────────

MODEL_NAME = "mistral"
llm        = ChatOllama(model=MODEL_NAME, temperature=0.1)

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

# Batch system prompt — same rules, but instructs the model on output format
BATCH_SYSTEM_PROMPT = SYSTEM_PROMPT + """

You will receive MULTIPLE clauses numbered [CLAUSE 1], [CLAUSE 2], etc.
For each clause, respond with exactly this format and nothing else between clauses:

[EXPLANATION 1]
<your explanation here>
[/EXPLANATION 1]

[EXPLANATION 2]
<your explanation here>
[/EXPLANATION 2]

...and so on. Do NOT skip any clause. Do NOT add any text outside these tags."""


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _build_single_prompt(clause_text: str, retrieved_sections: List[dict]) -> str:
    law_context = format_for_prompt(retrieved_sections)
    return (
        f'Clause to explain:\n"""{clause_text}"""\n\n'
        f"Relevant Indian law sections (use ONLY these):\n{law_context}\n\n"
        f"Explain this clause to a citizen in simple English. Cite Act and Section numbers."
    )


def _build_batch_prompt(
    clauses: List[str],
    retrieved_list: List[List[dict]],
) -> str:
    """
    Pack all clauses into one prompt.
    Each clause gets its own law context block so the model doesn't mix them up.
    """
    parts = []
    for i, (clause, retrieved) in enumerate(zip(clauses, retrieved_list), start=1):
        law_context = format_for_prompt(retrieved)
        parts.append(
            f"[CLAUSE {i}]\n"
            f'"""{clause}"""\n\n'
            f"Relevant Indian law sections for clause {i} (use ONLY these):\n"
            f"{law_context}"
        )
    return "\n\n" + "\n\n---\n\n".join(parts) + "\n\nNow explain each clause."


def _parse_batch_response(response_text: str, n: int) -> List[str]:
    """
    Extract [EXPLANATION i]...[/EXPLANATION i] blocks from the batch response.
    Falls back to splitting on '---' if the model ignored the format tags.
    """
    import re
    explanations = []
    for i in range(1, n + 1):
        pattern = rf"\[EXPLANATION {i}\](.*?)\[/EXPLANATION {i}\]"
        m = re.search(pattern, response_text, re.DOTALL)
        if m:
            explanations.append(m.group(1).strip())
        else:
            explanations.append("")   # will trigger fallback in explain_many()

    # If tag parsing failed (model ignored format), fall back to --- split
    if all(e == "" for e in explanations):
        parts = re.split(r"\n---\n|\n-{3,}\n", response_text)
        if len(parts) == n:
            return [p.strip() for p in parts]

    return explanations


# ── SINGLE EXPLAIN  (backward compatible) ────────────────────────────────────

def explain(clause_text: str, retrieved_sections: List[dict]) -> str:
    """Single-clause explain — unchanged public API."""
    prompt   = _build_single_prompt(clause_text, retrieved_sections)
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    return response.content


# ── BATCH EXPLAIN  ← THE KEY NEW FUNCTION ────────────────────────────────────

def explain_many(
    clauses: List[str],
    retrieved_list: List[List[dict]],
    batch_size: int = 5,
) -> List[str]:
    """
    Explain a list of clauses using as few Ollama calls as possible.

    Why batch_size=5 and not all at once?
      Mistral's context window is ~32k tokens. A 40-clause document with 50
      retrieved sections each would overflow it. Groups of 5 keeps each prompt
      well within limits while still cutting calls from 40 → 8.

    Tune batch_size up if:
      - Your clauses are short  (< 100 words each)
      - Your TOP_K in retriever is small (< 20)
      - Your model has a large context window (32k+)

    Tune batch_size down if:
      - You get truncated / garbled outputs
      - You see the model mixing up clause numbers

    Returns a list of explanation strings in the same order as `clauses`.
    """
    all_explanations: List[str] = []
    n = len(clauses)

    for batch_start in range(0, n, batch_size):
        batch_clauses   = clauses[batch_start : batch_start + batch_size]
        batch_retrieved = retrieved_list[batch_start : batch_start + batch_size]
        b               = len(batch_clauses)

        print(f"[explain_many] Batch {batch_start//batch_size + 1}: "
              f"clauses {batch_start+1}–{batch_start+b} of {n}")

        # Single Ollama call for the whole batch
        prompt   = _build_batch_prompt(batch_clauses, batch_retrieved)
        messages = [SystemMessage(content=BATCH_SYSTEM_PROMPT), HumanMessage(content=prompt)]
        response = llm.invoke(messages)

        parsed = _parse_batch_response(response.content, b)

        # Safety net: if a clause got an empty explanation, fall back to single call
        for j, (clause, retrieved, expl) in enumerate(
            zip(batch_clauses, batch_retrieved, parsed)
        ):
            if not expl:
                print(f"[explain_many] Clause {batch_start+j+1} parse failed — retrying solo")
                parsed[j] = explain(clause, retrieved)

        all_explanations.extend(parsed)

    return all_explanations
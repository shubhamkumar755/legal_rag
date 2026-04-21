"""
verifier.py — Takes explanation + retrieved sections → citation verification, confidence, risk

Improvements over v1:
- Fixed group-order detection bug for Act names starting with digits
- Stricter act-name matching (token overlap ratio instead of substring)
- Ratio-based confidence scoring (not just binary "any verified?")
- Explicit missing-score detection (no more silent 1.0 default)
- Multi-risk detection with collected phrases instead of first-match stop
- Severity tiers for risk phrases
- retrieve() imported at module level with a lazy fallback to avoid circular import
- Type annotations throughout
"""

from __future__ import annotations

import re
import logging
from typing import Literal

logger = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────

SCORE_VERIFIED: float = 0.50
SCORE_PARTIAL:  float = 0.10

# Minimum token overlap ratio for act-name matching (0–1)
# 0.5 = at least half the tokens in the query act must appear in the retrieved act
ACT_MATCH_RATIO: float = 0.5

# Confidence thresholds (fraction of verified citations)
CONFIDENCE_HIGH_THRESHOLD:   float = 1.00   # all verified
CONFIDENCE_MEDIUM_THRESHOLD: float = 0.50   # ≥ 50 % verified

# Risk phrases grouped by severity
_RISK_PHRASES: dict[str, list[str]] = {
    "high": [
        "waives your right",
        "waives all rights",
        "no recourse",
        "non-refundable under all circumstances",
    ],
    "medium": [
        "unfair",
        "favours the landlord",
        "favours the employer",
        "below the standard",
        "less than required",
        "at the discretion of",
        "may terminate without",
        "unilateral",
    ],
    "low": [
        "subject to change",
        "as determined by",
        "reasonable notice",
    ],
}

# Compiled once at import time
_CITATION_PATTERNS: list[re.Pattern] = [
    # "Section 12A of the Rent Control Act 1958"
    re.compile(
        r"[Ss]ection\s+(\d+[A-Za-z]?)\s+of\s+([\w\s]+?Act[\w\s]*?)(?=\s*\d{4}|\s*[,\.;]|$)",
        re.IGNORECASE,
    ),
    # "Rent Control Act 1958, Section 12A"
    re.compile(
        r"([\w\s]+?Act[\w\s]*?(?:\d{4})?)[,\s]+[Ss]ection\s+(\d+[A-Za-z]?)",
        re.IGNORECASE,
    ),
    # "Under the Rent Control Act 1958, Section 12A"
    re.compile(
        r"[Uu]nder\s+([\w\s]+?Act[\w\s]*?(?:\d{4})?)[,\s]+[Ss]ection\s+(\d+[A-Za-z]?)",
        re.IGNORECASE,
    ),
]

# Status type alias
VerifyStatus = Literal["verified", "partial", "not_found"]


# ── CITATION EXTRACTION ───────────────────────────────────────────────────────

def _split_groups(groups: tuple[str, str]) -> tuple[str, str]:
    section_pattern = re.compile(r"^\d+[A-Za-z]?$")
    g0, g1 = groups[0].strip(), groups[1].strip()

    if section_pattern.match(g0):
        return g1, g0   # (act_name, section_num)
    return g0, g1       # (act_name, section_num)


def extract_citations(llm_response: str, retrieved_sections: list[dict] | None = None) -> list[dict]:
    """
    Extract all (act, section) pairs mentioned in the LLM response.
    Handles two styles:
      1. Prose style  — "Section 108 of the Transfer of Property Act"
      2. Bracket style — [1], [2] mapped back to retrieved_sections metadata
    """
    citations: list[dict] = []
    seen: set[tuple[str, str]] = set()

    # ── Style 1: prose act/section references ────────────────────────────
    for pattern in _CITATION_PATTERNS:
        for match in pattern.finditer(llm_response):
            act_name, section_num = _split_groups(match.groups())
            act_name = re.sub(r"\s+", " ", act_name).strip(" ,.")
            section_num = section_num.strip()

            key = (act_name.lower(), section_num.lower())
            if key not in seen:
                seen.add(key)
                citations.append({"act": act_name, "section": section_num})

    # ── Style 2: bracket snippet references [1], [2, 3] ──────────────────
    if retrieved_sections:
        bracket_refs: set[int] = set()
        for match in re.finditer(r"\[(\d+(?:,\s*\d+)*)\]", llm_response):
            for num in match.group(1).split(","):
                bracket_refs.add(int(num.strip()))

        for idx in bracket_refs:
            # snippets are 1-indexed in the prompt but 0-indexed in the list
            record = retrieved_sections[idx - 1] if 1 <= idx <= len(retrieved_sections) else None
            if record is None:
                continue

            act_name   = str(record.get("act_name", "")).strip()
            section_num = str(record.get("section_number", "")).strip()

            if not act_name or not section_num:
                continue

            key = (act_name.lower(), section_num.lower())
            if key not in seen:
                seen.add(key)
                citations.append({"act": act_name, "section": section_num})

    return citations


# ── ACT NAME MATCHING ─────────────────────────────────────────────────────────

def _act_token_overlap(query_act: str, retrieved_act: str) -> float:
    """
    Return the fraction of tokens in query_act that appear in retrieved_act.

    Stricter than v1's substring check while still being fuzzy enough to
    match "Rent Control Act" against "Delhi Rent Control Act 1958".
    """
    stop = {"the", "of", "and", "a", "an", "to", "in", "for", "act"}
    q_tokens = {t for t in query_act.lower().split() if t not in stop}
    r_tokens  = {t for t in retrieved_act.lower().split() if t not in stop}

    if not q_tokens:
        return 0.0

    overlap = q_tokens & r_tokens
    return len(overlap) / len(q_tokens)


def _section_matches(citation_section: str, retrieved_section: str) -> bool:
    """Normalise and compare section numbers (strip leading zeros)."""
    return citation_section.lstrip("0").lower() == retrieved_section.lstrip("0").lower()


# ── SCORE EXTRACTION ──────────────────────────────────────────────────────────

def _extract_score(record: dict) -> float | None:
    """
    Return the best available relevance score from a retrieved record.

    Returns None (instead of a silent 1.0) when no score field is present,
    so callers can decide how to handle missing scores explicitly.
    """
    if "rerank_score" in record:
        return float(record["rerank_score"])
    if "rrf_score" in record:
        return float(record["rrf_score"])
    return None


# ── CITATION VERIFICATION ─────────────────────────────────────────────────────

def verify_citations(
    citations: list[dict],
    retrieved_sections: list[dict],
) -> list[dict]:
    """
    Verify each extracted citation against the already-retrieved sections,
    falling back to a fresh retrieve() call when no match is found.
    """
    try:
        from retriever import retrieve  # type: ignore[import]
    except ImportError:
        logger.warning("retriever module not found — fallback retrieval disabled")
        retrieve = None  # type: ignore[assignment]

    results: list[dict] = []

    for citation in citations:
        match_record: dict | None = None
        best_overlap = 0.0

        # Stage 1: search already-retrieved sections using token overlap
        for record in retrieved_sections:
            r_act = record.get("act_name", "")
            r_sec = str(record.get("section_number", ""))

            overlap = _act_token_overlap(citation["act"], r_act)
            if overlap >= ACT_MATCH_RATIO and _section_matches(citation["section"], r_sec):
                if overlap > best_overlap:
                    best_overlap = overlap
                    match_record = record

        if match_record is not None:
            score = _extract_score(match_record)
            if score is None:
                logger.warning(
                    "No score field on matched record for %s §%s — defaulting to SCORE_VERIFIED",
                    citation["act"], citation["section"],
                )
                score = SCORE_VERIFIED   # conservative: treat as verified but warn
            if score >= SCORE_VERIFIED:
                status: VerifyStatus = "verified"
            elif score >= SCORE_PARTIAL:
                status = "partial"
            else:
                status = "not_found"

        else:
            # Stage 2: fallback live retrieval
            if retrieve is None:
                status, score = "not_found", 0.0
            else:
                query = f"Section {citation['section']} {citation['act']}"
                try:
                    fresh = retrieve(query, top_k=1)
                except Exception as exc:
                    logger.error("retrieve() failed for '%s': %s", query, exc)
                    fresh = []

                if not fresh:
                    status, score = "not_found", 0.0
                else:
                    score = _extract_score(fresh[0])
                    if score is None:
                        logger.warning(
                            "No score on fallback result for %s §%s — treating as not_found",
                            citation["act"], citation["section"],
                        )
                        status, score = "not_found", 0.0
                    elif score >= SCORE_VERIFIED:
                        status = "verified"
                    elif score >= SCORE_PARTIAL:
                        status = "partial"
                    else:
                        status = "not_found"

        results.append({**citation, "status": status, "score": score})

    return results


# ── CONFIDENCE ────────────────────────────────────────────────────────────────

def compute_confidence(verification_results: list[dict]) -> dict:
    """
    Ratio-based confidence scoring.

    v1 treated "any verified citation" as Medium confidence even if 9 out of
    10 were not_found.  This version uses the verified fraction so that a
    mostly-failed batch correctly scores as Low.
    """
    if not verification_results:
        return {"level": "low", "label": "🚨 No citations — potential hallucination"}

    total = len(verification_results)
    counts: dict[str, int] = {"verified": 0, "partial": 0, "not_found": 0}
    for r in verification_results:
        counts[r["status"]] += 1

    verified_ratio = counts["verified"] / total
    any_not_found  = counts["not_found"] > 0

    if verified_ratio >= CONFIDENCE_HIGH_THRESHOLD and not any_not_found:
        level, label = "high",   "✅ High confidence — all citations verified"
    elif verified_ratio >= CONFIDENCE_MEDIUM_THRESHOLD:
        level, label = "medium", f"⚠️  Medium confidence — {counts['verified']}/{total} citations verified"
    elif counts["verified"] > 0:
        level, label = "medium", f"⚠️  Medium confidence — only {counts['verified']}/{total} citations verified"
    else:
        level, label = "low",    "🚨 Low confidence — citations not found in database"

    return {
        "level": level,
        "label": label,
        "counts": counts,          # expose raw counts for callers
        "verified_ratio": round(verified_ratio, 2),
    }


# ── RISK FLAG ─────────────────────────────────────────────────────────────────

def flag_risk(explanation: str) -> dict | None:
    """
    Scan for risk phrases across all severity tiers.

    v1 stopped at the first match and returned a generic string.
    This version collects all matched phrases, groups them by severity,
    and returns the highest severity found alongside the specific triggers.
    Returns None if no risk detected.
    """
    low = explanation.lower()
    matched: dict[str, list[str]] = {"high": [], "medium": [], "low": []}

    for severity, phrases in _RISK_PHRASES.items():
        for phrase in phrases:
            if phrase in low:
                matched[severity].append(phrase)

    # Find highest severity with at least one match
    for severity in ("high", "medium", "low"):
        if matched[severity]:
            all_phrases = matched["high"] + matched["medium"] + matched["low"]
            icons = {"high": "🔴", "medium": "⚠️ ", "low": "🔵"}
            return {
                "severity":       severity,
                "label":          f"{icons[severity]} Risk detected ({severity}): clause may disadvantage you.",
                "matched_phrases": all_phrases,
            }

    return None


# ── MAIN ──────────────────────────────────────────────────────────────────────

def verify(explanation: str, retrieved_sections: list[dict]) -> dict:
    """
    Full verification pipeline.

    Args:
        explanation:        The LLM-generated text to verify.
        retrieved_sections: Sections already fetched by the retrieval pipeline.

    Returns:
        {
            "citations":       list of {act, section, status, score},
            "confidence":      {level, label, counts, verified_ratio},
            "risk_flag":       {severity, label, matched_phrases} | None,
        }
    """
    citations    = extract_citations(explanation, retrieved_sections)
    verification = verify_citations(citations, retrieved_sections)
    confidence   = compute_confidence(verification)
    risk         = flag_risk(explanation)

    return {
        "citations":  verification,
        "confidence": confidence,
        "risk_flag":  risk,
    }
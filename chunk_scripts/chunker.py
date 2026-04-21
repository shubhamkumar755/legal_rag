"""
chunker.py — Fine-grain chunking of individual sections

Hierarchy produced:
    section_full  (always emitted — keeps complete legal context)
    └── subsection  (e.g. "(1)", "(2)", …)
        └── annotation  (Explanation / Illustration / Exception / Proviso)

Responsibilities:
  - Split section text into subsections
  - Detect and label annotations (Explanation, Proviso, etc.)
  - Emit chunk dicts; drop anything shorter than MIN_CHUNK_CHARS
"""

import re
from config import MIN_CHUNK_CHARS


# ── Patterns ──────────────────────────────────────────────────────────────────

_SUBSECTION  = re.compile(r"(?<!\w)\((\d+)\)\s+",        re.MULTILINE)
_ANNOTATION  = re.compile(
    r"^(Explanation(?:\s+\d+)?|Illustration(?:\s+\d+)?|"
    r"Exception(?:\s+\d+)?|Proviso(?:\s+\d+)?)[\s.—–:-]",
    re.MULTILINE,
)

_ANNOTATION_TYPE = {
    "Explanation":  "explanation",
    "Illustration": "illustration",
    "Exception":    "exception",
    "Proviso":      "proviso",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _annotation_chunk_type(keyword: str) -> str:
    """Map annotation keyword (possibly with trailing number) to chunk_type."""
    base = keyword.strip().split()[0]       # "Explanation 1" → "Explanation"
    return _ANNOTATION_TYPE.get(base, "annotation")


def _split_by_pattern(text: str, pattern: re.Pattern) -> list[tuple[str, str]]:
    """
    Split *text* by *pattern*.  Returns a list of (key, fragment) tuples where
    *key* is the captured group (e.g. "1", "a") or "" for the pre-match lead.
    """
    matches = list(pattern.finditer(text))
    if not matches:
        return [("", text)]

    parts: list[tuple[str, str]] = []
    pre = text[: matches[0].start()].strip()
    if pre:
        parts.append(("", pre))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        parts.append((m.group(1), text[m.start():end].strip()))
    return parts


def _split_annotations(text: str) -> list[tuple[str, str]]:
    """Split *text* at Explanation / Illustration / Exception / Proviso markers."""
    matches = list(_ANNOTATION.finditer(text))
    if not matches:
        return [("", text)]

    parts: list[tuple[str, str]] = []
    pre = text[: matches[0].start()].strip()
    if pre:
        parts.append(("", pre))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        parts.append((m.group(1).strip(), text[m.start():end].strip()))
    return parts


def _emit(
    fine:       list[dict],
    base_meta:  dict,
    chunk_type: str,
    subsection: str,
    clause:     str,
    text:       str,
) -> None:
    """
    Further split *text* at annotation boundaries and append one chunk dict
    per fragment to *fine*.  Fragments shorter than MIN_CHUNK_CHARS are skipped.
    """
    for key, body in _split_annotations(text):
        if len(body.strip()) < MIN_CHUNK_CHARS:
            continue
        fine.append({
            **base_meta,
            "chunk_type": _annotation_chunk_type(key) if key else chunk_type,
            "subsection": subsection,
            "clause":     clause,
            "annotation": key,
            "text":       body.strip(),
        })


# ── Public API ────────────────────────────────────────────────────────────────

def split_into_fine_chunks(section: dict) -> list[dict]:
    """
    Produce fine-grained chunk dicts from a single *section* dict
    (as returned by ``sectioner.split_sections``).

    Always includes a ``section_full`` chunk for full-context retrieval,
    then emits subsection-level (and annotation-level) chunks.
    """
    body      = section["text"]
    base_meta = {
        "section_number": section["section_number"],
        "section_title":  section["section_title"],
        "chapter":        section["chapter"],
    }

    fine: list[dict] = [{
        **base_meta,
        "chunk_type": "section_full",
        "subsection": "",
        "clause":     "",
        "annotation": "",
        "text":       body.strip(),
    }]

    subsec_parts = _split_by_pattern(body, _SUBSECTION)
    has_subsecs  = any(k != "" for k, _ in subsec_parts)

    if has_subsecs:
        for key, chunk_text in subsec_parts:
            if len(chunk_text.strip()) < MIN_CHUNK_CHARS:
                continue
            if key == "":
                _emit(fine, base_meta, "section",    "",  "", chunk_text)
            else:
                _emit(fine, base_meta, "subsection", key, "", chunk_text)
    else:
        _emit(fine, base_meta, "section", "", "", body)

    return fine
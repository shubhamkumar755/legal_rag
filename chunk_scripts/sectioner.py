"""
sectioner.py — Split raw act text into labelled sections and tag chapters

Responsibilities:
  - Detect section headers (dash-style or capitalised-fallback)
  - Deduplicate repeated section numbers
  - Map each section to its parent chapter
"""

import re
from extractor import join_wrapped_lines
from config   import MIN_CHUNK_CHARS


# ── Patterns ──────────────────────────────────────────────────────────────────

_SECTION_HEADER = re.compile(
    r"^(?:\d+\[)?\[?(\d{1,3})\.\s+(.{10,250}?)(?:\.\s?[—–-]|\.\s?--)",
    re.MULTILINE,
)
_SECTION_HEADER_FALLBACK = re.compile(
    r"^(?:\d+\[)?\[?(\d{1,3})\.\s+([A-Z][^\n]{10,250})\n",
    re.MULTILINE,
)
_CHAPTER_HEADER = re.compile(
    r"CHAPTER\s+([IVXLCDM]+)\s*\n\s*([A-Z][^\n]{3,80})",
    re.MULTILINE,
)


# ── Public API ────────────────────────────────────────────────────────────────

def split_sections(text: str) -> list[dict]:
    """
    Return a list of section dicts::

        {
            "section_number": int,
            "section_title":  str,
            "text":           str,   # full text of the section
        }

    Sections shorter than MIN_CHUNK_CHARS are silently dropped.
    """
    text    = join_wrapped_lines(text)
    matches = list(_SECTION_HEADER.finditer(text))
    if len(matches) < 5:
        print("  Few dash-style headers — using fallback pattern.")
        matches = list(_SECTION_HEADER_FALLBACK.finditer(text))

    sections, seen = [], set()
    for i, m in enumerate(matches):
        sec_num = int(m.group(1))
        if sec_num in seen:
            continue
        seen.add(sec_num)
        start = m.start()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body  = text[start:end].strip()
        if len(body) < MIN_CHUNK_CHARS:
            continue
        sections.append({
            "section_number": sec_num,
            "section_title":  m.group(2).strip().rstrip(".—–-"),
            "text":           body,
        })
    return sections


def tag_chapters(text: str, sections: list[dict]) -> list[dict]:
    """
    Annotate each section dict with a ``"chapter"`` key by matching its
    position in *text* against chapter headings found earlier in the document.
    Mutates and returns *sections*.
    """
    text        = join_wrapped_lines(text)
    chapter_map = [
        (m.start(), f"Chapter {m.group(1)} — {m.group(2).strip().title()}")
        for m in _CHAPTER_HEADER.finditer(text)
    ]
    for sec in sections:
        m       = re.search(rf"^{sec['section_number']}\.\s", text, re.MULTILINE)
        offset  = m.start() if m else 0
        chapter = "Unknown"
        for ch_offset, ch_label in chapter_map:
            if ch_offset <= offset:
                chapter = ch_label
        sec["chapter"] = chapter
    return sections
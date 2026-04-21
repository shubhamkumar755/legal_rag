"""
book_chunker.py — Chunking for handbook / legal-guide PDFs

Handles documents like "Legal Aid to Legal Rights" which have:
  - Named topic chapters  (e.g. "Lodging an FIR", "Right to Bail")
  - No numbered sections
  - FAQ / Q&A blocks, prose paragraphs, and bullet lists

Chunk hierarchy produced:
  chapter_full  (always emitted — full chapter text for broad retrieval)
  └── topic     (Q&A question + its answer as one chunk)
      OR
      paragraph (plain prose block)
"""

import re
from config import MIN_CHUNK_CHARS


# ── Patterns ──────────────────────────────────────────────────────────────────

# Chapter / top-level heading: ALL-CAPS line or Title Case line on its own
# (no leading digits — that's an Act section)
_CHAPTER_HEADING = re.compile(
    r"^([A-Z][A-Za-z &\-]{3,80})\s*$",
    re.MULTILINE,
)

# Q&A style headings: "What is X?", "Can you Y?", "How do I Z?"
_QA_HEADING = re.compile(
    r"^((?:What|Who|Why|How|When|Where|Can|Is|Are|Does|Do|Should|Which|"
    r"What if|Is there)[^\n?]{5,120}\?)\s*$",
    re.MULTILINE,
)

# Also catch explicit "Q." or "Q:" style
_Q_PREFIX = re.compile(r"^Q[.:\s]\s*(.+\?)\s*$", re.MULTILINE)

# Page-number / header lines to strip (e.g. "2 NLSIU|CEERA|DOJ")
_NOISE = re.compile(
    r"^\s*(?:\d+\s+)?(?:NLSIU\|CEERA\|DOJ|Pan India Legal Literacy[^\n]*|"
    r"Designing Innovative Solutions[^\n]*)\s*$",
    re.MULTILINE,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Strip running headers / page artefacts."""
    text = _NOISE.sub("", text)
    # collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_by_qa(chapter_text: str, chapter_title: str) -> list[dict]:
    """
    Split a chapter's text into Q&A topic chunks.
    Falls back to paragraph chunks if no Q&A headings are found.
    """
    # Try both Q&A patterns together
    combined = re.compile(
        r"^((?:What|Who|Why|How|When|Where|Can|Is|Are|Does|Do|Should|Which|"
        r"What if|Is there)[^\n?]{5,120}\?|Q[.:\s]\s*.+\?)\s*$",
        re.MULTILINE,
    )
    matches = list(combined.finditer(chapter_text))

    chunks: list[dict] = []

    if not matches:
        # No Q&A structure — split on double newlines (paragraphs)
        for para in re.split(r"\n{2,}", chapter_text):
            para = para.strip()
            if len(para) < MIN_CHUNK_CHARS:
                continue
            chunks.append({
                "chapter":   chapter_title,
                "topic":     "",
                "chunk_type": "paragraph",
                "text":      para,
            })
        return chunks

    # Pre-match preamble
    pre = chapter_text[: matches[0].start()].strip()
    if len(pre) >= MIN_CHUNK_CHARS:
        chunks.append({
            "chapter":    chapter_title,
            "topic":      "",
            "chunk_type": "paragraph",
            "text":       pre,
        })

    for i, m in enumerate(matches):
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(chapter_text)
        body  = chapter_text[m.start():end].strip()
        topic = m.group(0).strip().rstrip("?").strip() + "?"
        if len(body) < MIN_CHUNK_CHARS:
            continue
        chunks.append({
            "chapter":    chapter_title,
            "topic":      topic,
            "chunk_type": "topic",
            "text":       body,
        })

    return chunks


# ── Public API ────────────────────────────────────────────────────────────────

# Well-known chapter titles in the Legal Aid book (used to anchor splits)
_KNOWN_CHAPTERS = [
    "Lodging an Fir",
    "Right to Bail",
    "Land Acquisition",
    "Rights of Prisoners",
    "Juvenile Justice Act",
    "Gender-Based Violence",
    "Cyber Crimes Against Women",
    "Law of Adoption",
    "Domestic Violence",
    "Medical Negligence",
    "POSH Act",
    "Pollution Control",
    "Waste Management",
    "Forest Conservation",
    "MSME",
    "Start",
    "FDI",
    "ODI",
]

_CHAPTER_ANCHOR = re.compile(
    r"^(" + "|".join(re.escape(c) for c in _KNOWN_CHAPTERS) + r")[^\n]*$",
    re.MULTILINE | re.IGNORECASE,
)


def split_book_into_chunks(raw_text: str) -> list[dict]:
    """
    Split a book/handbook PDF into chunk dicts suitable for ``store.upsert_chunks``.

    Each chunk dict has the keys expected by ``store.py``::

        {
            "chapter":        str,
            "section_number": int,   # always 0 for books
            "section_title":  str,   # same as topic
            "chunk_type":     str,   # "chapter_full" | "topic" | "paragraph"
            "subsection":     str,   # always ""
            "clause":         str,   # always ""
            "annotation":     str,   # always ""
            "text":           str,
        }
    """
    text = _clean(raw_text)

    # ── Split into chapters ────────────────────────────────────────────────
    matches = list(_CHAPTER_ANCHOR.finditer(text))

    # Fallback: generic heading detection if known chapters aren't found
    if len(matches) < 2:
        matches = list(_CHAPTER_HEADING.finditer(text))

    all_chunks: list[dict] = []
    chapter_counter = 0

    for i, m in enumerate(matches):
        chapter_title = m.group(0).strip().title()
        start = m.start()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body  = text[start:end].strip()

        if len(body) < MIN_CHUNK_CHARS:
            continue

        chapter_counter += 1

        # Always emit a full-chapter chunk
        all_chunks.append({
            "chapter":        f"Chapter {chapter_counter} — {chapter_title}",
            "section_number": chapter_counter,
            "section_title":  chapter_title,
            "chunk_type":     "chapter_full",
            "subsection":     "",
            "clause":         "",
            "annotation":     "",
            "text":           body,
        })

        # Fine-grained Q&A / paragraph chunks
        for fine in _split_by_qa(body, chapter_title):
            all_chunks.append({
                "chapter":        f"Chapter {chapter_counter} — {chapter_title}",
                "section_number": chapter_counter,
                "section_title":  fine.get("topic") or chapter_title,
                "chunk_type":     fine["chunk_type"],
                "subsection":     "",
                "clause":         "",
                "annotation":     "",
                "text":           fine["text"],
            })

    return all_chunks
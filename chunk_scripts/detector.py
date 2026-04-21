"""
detector.py — Detect whether a PDF is a structured Act or a book/handbook

Returns one of two document type strings:
  "act"   — numbered sections, CHAPTER headings (BNSS-style)
  "book"  — topic chapters with Q&A / prose (Legal Aid-style)

Used by the pipeline to pick the right chunking strategy.
"""

import re


_ACT_SIGNALS = [
    re.compile(r"ARRANGEMENT OF SECTIONS", re.IGNORECASE),
    re.compile(r"^\d{1,3}\.\s+[A-Z][^\n]{5,}", re.MULTILINE),   # "1. Short title …"
    re.compile(r"CHAPTER\s+[IVXLCDM]+\s*\n", re.MULTILINE),
]

_BOOK_SIGNALS = [
    re.compile(r"Table\s+of\s+Contents", re.IGNORECASE),
    re.compile(r"ISBN\s+No\.", re.IGNORECASE),
    re.compile(r"^\s*Q[.:]\s+\w", re.MULTILINE),                 # "Q. What is …"
    re.compile(r"^\s*What\s+is\b", re.MULTILINE),                # FAQ style
    re.compile(r"Preface|Foreword|Introduction", re.IGNORECASE),
]


def detect_document_type(sample_text: str) -> str:
    """
    Examine the first ~5 pages of text and return ``"act"`` or ``"book"``.

    Args:
        sample_text: Raw text from the first few pages of the PDF.

    Returns:
        ``"act"`` if the document looks like a numbered legislative act,
        ``"book"`` otherwise.
    """
    act_hits  = sum(1 for p in _ACT_SIGNALS  if p.search(sample_text))
    book_hits = sum(1 for p in _BOOK_SIGNALS if p.search(sample_text))

    doc_type = "act" if act_hits >= book_hits else "book"
    print(f"  Detected document type: '{doc_type}'  (act_signals={act_hits}, book_signals={book_hits})")
    return doc_type
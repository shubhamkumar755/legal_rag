"""
extractor.py — PDF text extraction for IndiaCode acts

Responsibilities:
  - Detect and skip Table-of-Contents pages
  - Extract raw text from all remaining pages
  - Join soft-wrapped lines
"""

import re
import pdfplumber


def is_toc_page(text: str) -> bool:
    """Return True if the page looks like a Table of Contents."""
    if re.search(r"ARRANGEMENT OF SECTIONS|^SECTIONS\s*$", text, re.MULTILINE):
        return True
    lines    = [l.strip() for l in text.splitlines() if l.strip()]
    toc_line = re.compile(r"^\d{1,3}\.\s+[A-Za-z].{5,100}\.$")
    return sum(1 for l in lines if toc_line.match(l)) > 8


def extract_text(pdf_path: str) -> str:
    """
    Open *pdf_path*, skip TOC pages, and return the remaining text
    joined with newlines.
    """
    pages, skipped = [], 0
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if is_toc_page(text):
                skipped += 1
            else:
                pages.append(text)
    print(f"  Skipped {skipped} TOC pages, using {len(pages)} content pages.")
    return "\n".join(pages)


def join_wrapped_lines(text: str) -> str:
    """
    Collapse soft line-breaks that PDF extraction introduces inside sentences.
    A break is removed when the previous line ends in a word character,
    comma, or semicolon and the next line starts with a lowercase letter
    or an opening parenthesis.
    """
    return re.sub(r"(?<=[a-zA-Z,;])\n(?=[a-z(])", " ", text)

import re
import logging
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — INPUT HANDLING
# ─────────────────────────────────────────────────────────────────────────────

def extract_text(source) -> str:
    """
    Accept a PDF path, DOCX path, or raw text string.
    Returns plain extracted text.
    """
    if not isinstance(source, str):
        raise ValueError("Input must be a file path string or raw text string.")

    # Raw text: doesn't end with a known file extension
    if not (source.strip().endswith(".pdf") or source.strip().endswith(".docx")):
        logger.info("Input detected as raw text")
        return source

    if source.endswith(".pdf"):
        logger.info(f"Extracting text from PDF: {source}")
        return _read_pdf(source)

    if source.endswith(".docx"):
        logger.info(f"Extracting text from DOCX: {source}")
        return _read_docx(source)

    raise ValueError(f"Unsupported file type: {source}")


def _read_pdf(path: str) -> str:
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("Run: pip install pdfplumber")
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n".join(pages)


def _read_docx(path: str) -> str:
    try:
        from docx import Document
    except ImportError:
        raise ImportError("Run: pip install python-docx")
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — CLAUSE SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

# ── Patterns that OPEN a new clause ──────────────────────────────────────────
# Each pattern matches the START of a line that begins a new clause boundary.
# Ordered from most-specific to least-specific to avoid false matches.

CLAUSE_BOUNDARY = re.compile(
    r"(?:"

    # ── Employment / NDA / MOU style ──────────────────────────────────────
    # "Article 1", "Article 1 –", "Article 1."
    r"^\s*Article\s+\d+[\s\.–:-]"

    # "Section 1", "Section 1.", "Section 1 –"
    r"|^\s*Section\s+\d+[\s\.–:-]"

    # "Clause 1", "Clause 4.5"
    r"|^\s*Clause\s+[\d\.]+[\s\.–:-]"

    # ── Will / Testament style ─────────────────────────────────────────────
    # Roman numerals at line start: "I.", "II.", "III.", "IV.", "V.", "VI."
    r"|^\s*(?:I{1,3}|IV|V?I{0,3}|IX|X{0,3}(?:IX|IV|V?I{0,3}))\.\s"

    # "FIRST:", "SECOND:", "THIRD:", "FOURTH:", "FIFTH:"
    r"|^\s*(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH)\s*[:\.]"

    # ── Rental / Sale Deed / POA / Partnership style ───────────────────────
    # "1. That ...", "2. That ...", "10. That ..."  (That-prefixed numbered)
    r"|^\s*\d{1,2}\.\s+That\s"

    # "1. The ...", "2. An ...", plain numbered clauses
    r"|^\s*\d{1,2}\.\s+[A-Z]"

    # ── Subclauses ─────────────────────────────────────────────────────────
    # "(a)", "(b)", "(i)", "(ii)", "(iii)"
    r"|^\s*\([a-z]{1,3}\)\s"

    # ── Recitals and bridge lines ──────────────────────────────────────────
    # "WHEREAS", "AND WHEREAS"
    r"|^\s*(?:AND\s+)?WHEREAS[,\s]"

    # "NOW THEREFORE", "NOW THIS DEED WITNESSETH", "NOW KNOW YOU ALL"
    r"|^\s*NOW\b"

    # "WITNESSETH"
    r"|^\s*WITNESSETH"

    # ── Legal Notice style ─────────────────────────────────────────────────
    # "THAT the opposite party..." (all-caps THAT at start)
    r"|^\s*THAT\s+[a-z]"

    # "TAKE NOTICE THAT"
    r"|^\s*TAKE\s+NOTICE"

    # ── ALL CAPS section headings (min 4 chars) ────────────────────────────
    # e.g. "TERMINATION", "CONFIDENTIALITY", "INDEMNIFICATION"
    r"|^\s*[A-Z][A-Z\s]{3,}(?:\s*[:\-–])?\s*$"

    # ── Title Case headings with colon ────────────────────────────────────
    # e.g. "Security Deposit:", "Notice Period:", "Governing Law:"
    r"|^\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4}\s*:\s*$"

    r")",
    re.MULTILINE
)

# ── Lines / blocks we always discard ─────────────────────────────────────────
SKIP_PATTERNS = re.compile(
    r"^("
    r"IN\s+WITNESS\s+WHEREOF"      # signature header
    r"|SIGNED\s+AND\s+DELIVERED"
    r"|Signature\s+of"
    r"|Sign(ed)?\s*:"
    r"|Witness\s*[:\d]"
    r"|Name\s*:"
    r"|Address\s*:"
    r"|Place\s*:"
    r"|Date\s*:"
    r"|Stamp\s+(Duty|Paper)"
    r"|SCHEDULE\b"                 # property schedule appendix
    r"|Annexure\b"
    r"|Page\s+\d+"                 # page numbers
    r")",
    re.IGNORECASE
)


def extract_clauses(full_text: str) -> List[str]:
    """
    Walk the document line by line.
    Start a new clause block whenever a CLAUSE_BOUNDARY pattern is matched.
    Return all accumulated blocks after filtering noise.
    """
    lines = full_text.splitlines()
    clauses: List[str] = []
    current_block: List[str] = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            # Blank line: preserve paragraph spacing inside current block
            if current_block:
                current_block.append("")
            continue

        if CLAUSE_BOUNDARY.match(stripped):
            # Save accumulated block before starting new one
            block_text = " ".join(t for t in current_block if t).strip()
            if block_text:
                clauses.append(block_text)
            current_block = [stripped]
        else:
            current_block.append(stripped)

    # Flush final block
    if current_block:
        block_text = " ".join(t for t in current_block if t).strip()
        if block_text:
            clauses.append(block_text)

    return _filter_clauses(clauses)


def _filter_clauses(clauses: List[str]) -> List[str]:
    """
    Remove:
      - Too short  (< 30 chars)  — likely just a heading label with no body
      - Signature / schedule boilerplate
      - Purely uppercase short labels  (e.g. "PARTIES", "RECITALS")
    """
    clean = []
    for clause in clauses:
        if len(clause) < 30:
            continue
        if SKIP_PATTERNS.match(clause):
            continue
        # Discard all-uppercase with fewer than 6 words — it's a section label
        if clause.isupper() and len(clause.split()) < 6:
            continue
        clean.append(clause)
    return clean


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — PREPROCESSING (Normalization)
# ─────────────────────────────────────────────────────────────────────────────

# Leading markers to strip so the pipeline sees clean legal prose
_LEADING_MARKER = re.compile(
    r"^("
    r"\d{1,2}\.\s+"                             # "1. "
    r"|\([a-z]{1,3}\)\s+"                        # "(a) "
    r"|(?:AND\s+)?WHEREAS[,\s]+"                 # "WHEREAS, " / "AND WHEREAS "
    r"|NOW\s+THEREFORE[,\s]*"                    # "NOW THEREFORE "
    r"|NOW\s+THIS\s+DEED\s+WITNESSETH[,\s]*"    # "NOW THIS DEED WITNESSETH"
    r"|NOW\s+KNOW\s+YOU\s+ALL[,\s]*"            # "NOW KNOW YOU ALL"
    r"|WITNESSETH[,\s]*"                         # "WITNESSETH"
    r"|TAKE\s+NOTICE\s+THAT[,\s]*"              # "TAKE NOTICE THAT"
    r"|THAT\s+"                                  # "THAT " (legal notice)
    r"|(?:FIRST|SECOND|THIRD|FOURTH|FIFTH)[:\.\s]+" # "FIRST:"
    r"|Article\s+\d+[\s\.–:-]+\s*"             # "Article 1 – "
    r"|Section\s+\d+[\s\.–:-]+\s*"             # "Section 1. "
    r"|Clause\s+[\d\.]+[\s\.–:-]+\s*"          # "Clause 4.5 – "
    r")",
    re.IGNORECASE
)


def preprocess_clause(clause: str) -> str:
    """
    Clean a clause for pipeline input:
      1. Collapse all whitespace to single space
      2. Strip leading clause markers / numbering
      3. Strip trailing whitespace
    """
    clause = re.sub(r"\s+", " ", clause).strip()
    clause = _LEADING_MARKER.sub("", clause).strip()
    return clause


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED ENTRY POINT  (called by document_processor.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_and_segment(source) -> List[str]:
    """

    source: PDF path | DOCX path | raw text string
    """
    full_text    = extract_text(source)         # Step 1
    raw_clauses  = extract_clauses(full_text)   # Step 2
    clean        = [preprocess_clause(c) for c in raw_clauses]  # Step 3
    clean        = [c for c in clean if len(c) > 20]            # drop empties
    logger.info(f"load_and_segment: {len(clean)} clauses ready for pipeline")
    return clean
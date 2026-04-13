"""
chunk_act.py  —  IndiaCode PDF → ChromaDB
------------------------------------------
Chunking granularity: Section → Sub-section → Clause

Embedding model is provided via LangChain (swap in one line).
ChromaDB is used directly via its own native interface — no LangChain
vectorstore wrapper involved.

Switching embedding models
--------------------------
Change only the EMBEDDINGS block near the top. Everything else is untouched.

Setup:
    pip install pdfplumber chromadb langchain-google-genai python-dotenv

.env file:
    GEMINI_API_KEY=your_key_here
"""

import os
import re
import pdfplumber
import chromadb
from dotenv import load_dotenv


load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────

ACTS_FOLDER = "D:\\College\\legal_rag\\legal_rag\\data\\acts"
ACT_NAME        = "Consumer Protection Act 2019"
DOMAIN          = "consumer"
DB_PATH         = r"D:\College\legal_rag\legal_rag\acts_db"
COLLECTION      = "indian_acts"
MIN_CHUNK_CHARS = 80
BATCH_SIZE      = 20

# ── EMBEDDING MODEL ───────────────────────────────────────────────────────────
# Swap provider by changing only this block.
# The rest of the script calls embed_documents(texts) → list[list[float]],
# which is the standard LangChain Embeddings interface.

# Option A: Google Gemini  ← active
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# EMBEDDINGS = GoogleGenerativeAIEmbeddings(
#     model="models/text-embedding-001",
#     google_api_key=os.getenv("GEMINI_API_KEY"),
#     task_type="retrieval_document",
# )
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# ── 1. EXTRACT TEXT ───────────────────────────────────────────────────────────

def is_toc_page(text: str) -> bool:
    if re.search(r"ARRANGEMENT OF SECTIONS|^SECTIONS\s*$", text, re.MULTILINE):
        return True
    lines    = [l.strip() for l in text.splitlines() if l.strip()]
    toc_line = re.compile(r"^\d{1,3}\.\s+[A-Za-z].{5,100}\.$")
    return sum(1 for l in lines if toc_line.match(l)) > 8


def extract_text() -> str:
    pages, skipped = [], 0
    with pdfplumber.open(PDF_PATH) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if is_toc_page(text):
                skipped += 1
            else:
                pages.append(text)
    print(f"  Skipped {skipped} TOC pages, using {len(pages)} content pages.")
    return "\n".join(pages)

# ── 2. JOIN WRAPPED LINES ─────────────────────────────────────────────────────

def join_wrapped_lines(text: str) -> str:
    return re.sub(r"(?<=[a-zA-Z,;])\n(?=[a-z(])", " ", text)

# ── 3. SPLIT INTO SECTIONS ───────────────────────────────────────────────────

_SECTION_HEADER = re.compile(
    r"^(\d{1,3})\.\s+(.{10,250}?)(?:\.\s?[—–-]|\.\s?--)",
    re.MULTILINE,
)
_SECTION_HEADER_FALLBACK = re.compile(
    r"^(\d{1,3})\.\s+([A-Z][^\n]{10,250})\n",
    re.MULTILINE,
)


def split_sections(text: str) -> list[dict]:
    text    = join_wrapped_lines(text)
    matches = list(_SECTION_HEADER.finditer(text))
    if len(matches) < 5:
        print("  ⚠ Few dash-style headers, using fallback pattern.")
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

# ── 4. TAG CHAPTERS ───────────────────────────────────────────────────────────

def tag_chapters(text: str, sections: list[dict]) -> list[dict]:
    text        = join_wrapped_lines(text)
    chapter_map = [
        (m.start(), f"Chapter {m.group(1)} — {m.group(2).strip().title()}")
        for m in re.finditer(
            r"CHAPTER\s+([IVXLCDM]+)\s*\n\s*([A-Z][^\n]{3,80})",
            text, re.MULTILINE,
        )
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

# ── 5. FINE-CHUNK ─────────────────────────────────────────────────────────────

_SUBSECTION = re.compile(r"(?<!\w)\((\d+)\)\s+",        re.MULTILINE)
_CLAUSE     = re.compile(r"(?<!\w)\(([a-z]{1,2})\)\s+", re.MULTILINE)

# Matches IPC-style annotations at the start of a line.
# Captures:
#   "Explanation"   / "Explanation 1" / "Explanation—"
#   "Illustration"  / "Illustration—"
#   "Exception"     / "Exception 1"   / "Exception—"
#   "Proviso"       (also common in older acts)
_ANNOTATION = re.compile(
    r"^(Explanation(?:\s+\d+)?|Illustration(?:\s+\d+)?|Exception(?:\s+\d+)?|Proviso(?:\s+\d+)?)"
    r"[\s.—–:-]",
    re.MULTILINE,
)

# Maps the captured keyword to a chunk_type string
_ANNOTATION_TYPE = {
    "Explanation":  "explanation",
    "Illustration": "illustration",
    "Exception":    "exception",
    "Proviso":      "proviso",
}


def _annotation_chunk_type(keyword: str) -> str:
    """Return chunk_type for an annotation keyword (strips trailing number)."""
    base = keyword.strip().split()[0]   # "Explanation 1" → "Explanation"
    return _ANNOTATION_TYPE.get(base, "annotation")


def _split_by_pattern(text: str, pattern: re.Pattern) -> list[tuple[str, str]]:
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
    """
    Split a chunk's text on IPC-style annotation headers.
    Returns [(annotation_keyword_or_"", chunk_text), …].
    The body text before the first annotation is returned with key "".
    """
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
    fine: list[dict],
    base_meta: dict,
    chunk_type: str,
    subsection: str,
    clause: str,
    text: str,
) -> None:
    """
    Split *text* on any annotation headers, then append one or more chunks.
    If there are no annotations the whole text becomes a single chunk.
    Annotation sub-chunks get their own chunk_type but inherit subsection/clause.
    """
    parts = _split_annotations(text)
    has_annotations = any(k != "" for k, _ in parts)

    for key, body in parts:
        if len(body.strip()) < MIN_CHUNK_CHARS:
            continue
        fine.append({
            **base_meta,
            "chunk_type": _annotation_chunk_type(key) if key else chunk_type,
            "subsection": subsection,
            "clause":     clause,
            "annotation": key,          # e.g. "Explanation 1", "" for main body
            "text":       body.strip(),
        })


def _make_chunk_id(sec: int, subsec: str = "", clause: str = "", annotation: str = "") -> str:
    base = f"{ACT_NAME.replace(' ', '_')}__sec_{sec}"
    if subsec:
        base += f"__sub_{subsec}"
    if clause:
        base += f"__cl_{clause}"
    if annotation:
        base += f"__{annotation.lower().replace(' ', '_')}"
    return base


def split_into_fine_chunks(section: dict) -> list[dict]:
    body      = section["text"]
    base_meta = {
        "section_number": section["section_number"],
        "section_title":  section["section_title"],
        "chapter":        section["chapter"],
    }
    fine: list[dict] = []

    subsec_parts = _split_by_pattern(body, _SUBSECTION)
    has_subsecs  = any(k != "" for k, _ in subsec_parts)

    if has_subsecs:
        for key, chunk_text in subsec_parts:
            if len(chunk_text.strip()) < MIN_CHUNK_CHARS:
                continue
            if key == "":
                _emit(fine, base_meta, "section", "", "", chunk_text)
            else:
                clause_parts = _split_by_pattern(chunk_text, _CLAUSE)
                has_clauses  = any(ck != "" for ck, _ in clause_parts)
                if has_clauses:
                    for cl_key, cl_text in clause_parts:
                        if len(cl_text.strip()) < MIN_CHUNK_CHARS:
                            continue
                        ct = "subsection" if cl_key == "" else "clause"
                        _emit(fine, base_meta, ct, key, cl_key, cl_text)
                else:
                    _emit(fine, base_meta, "subsection", key, "", chunk_text)
    else:
        clause_parts = _split_by_pattern(body, _CLAUSE)
        has_clauses  = any(ck != "" for ck, _ in clause_parts)
        if has_clauses:
            for cl_key, cl_text in clause_parts:
                if len(cl_text.strip()) < MIN_CHUNK_CHARS:
                    continue
                ct = "section" if cl_key == "" else "clause"
                _emit(fine, base_meta, ct, "", cl_key, cl_text)
        else:
            _emit(fine, base_meta, "section", "", "", body)

    return fine

# ── 6. EMBED + STORE ──────────────────────────────────────────────────────────

def store(all_chunks: list[dict]) -> None:
    client = chromadb.PersistentClient(path=DB_PATH)

    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i : i + BATCH_SIZE]
        texts = [
            f"{ACT_NAME}\nSection {c['section_number']}\n{c['text']}"
            for c in batch
        ]

        # LangChain call → list[list[float]]
        vectors = EMBEDDINGS.embed_documents(texts)

        ids, metas = [], []
        for c in batch:
            ids.append(f"{ACT_NAME.replace(' ', '_')}__chunk_{i+len(ids)}")
            metas.append({
                "act_name":       ACT_NAME,
                "domain":         DOMAIN,
                "chapter":        c["chapter"],
                "section_number": c["section_number"],
                "section_title":  c["section_title"],
                "chunk_type":     c["chunk_type"],
                "subsection":     c.get("subsection", ""),
                "clause":         c.get("clause", ""),
                "annotation":     c.get("annotation", ""),
            })

        # ChromaDB native upsert with explicit embeddings
        collection.upsert(ids=ids, documents=texts, embeddings=vectors, metadatas=metas)

        sec_range = f"{batch[0]['section_number']}–{batch[-1]['section_number']}"
        print(f"  Batch {i // BATCH_SIZE + 1}: {len(batch)} chunks  (sections {sec_range})")

    print(f"\n  Total chunks in collection: {collection.count()}")

# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    client = chromadb.PersistentClient(path=DB_PATH)
    # Delete old DB only once
    try:
        client.delete_collection(COLLECTION)
        print("Cleared old collection.")
    except Exception:
        pass
    for file in os.listdir(ACTS_FOLDER):
        if not file.endswith(".pdf"):
            continue

        print(f"\nProcessing {file}")

        PDF_PATH = os.path.join(ACTS_FOLDER, file)
        ACT_NAME = file.replace(".pdf", "").replace("-", " ").title()

        print("Extracting text …")
        raw_text = extract_text()

        print("Splitting into sections …")
        sections = split_sections(raw_text)
        sections = tag_chapters(raw_text, sections)

        print("Fine-chunking …")
        all_chunks = []
        for sec in sections:
            all_chunks.extend(split_into_fine_chunks(sec))

        print("Embedding and storing …")
        store(all_chunks)

    print("\n✅ All Acts processed.")








    # print("Extracting text …")
    # raw_text = extract_text()

    # print("Splitting into sections …")
    # sections = split_sections(raw_text)
    # sections = tag_chapters(raw_text, sections)

    # sections_found = sorted(s["section_number"] for s in sections)
    # print(f"  {len(sections)} sections: {sections_found}")
    # missing = sorted(set(range(1, 108)) - set(sections_found))
    # print(f"  ⚠ Missing: {missing}" if missing else "  ✅ All sections found!")

    # print("\nFine-chunking …")
    # all_chunks: list[dict]      = []
    # type_counts: dict[str, int] = {}
    # for sec in sections:
    #     for c in split_into_fine_chunks(sec):
    #         all_chunks.append(c)
    #         type_counts[c["chunk_type"]] = type_counts.get(c["chunk_type"], 0) + 1

    # print(f"  Total : {len(all_chunks)}")
    # for kind in ("section", "subsection", "clause", "explanation", "illustration", "exception", "proviso"):
    #     if kind in type_counts:
    #         print(f"  {kind:<12}: {type_counts[kind]}")
    # # catch any unexpected types
    # for kind, n in type_counts.items():
    #     if kind not in ("section", "subsection", "clause", "explanation", "illustration", "exception", "proviso"):
    #         print(f"  {kind:<12}: {n}")

    # print("\nEmbedding and storing …")
    # store(all_chunks)
    # print("\n✅ Done.")
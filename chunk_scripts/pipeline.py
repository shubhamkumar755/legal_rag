"""
pipeline.py — Orchestrator for the IndiaCode PDF → ChromaDB pipeline

Supports two document types, auto-detected per file:
  - "act"   → extractor → sectioner → chunker (existing flow)
  - "book"  → extractor → book_chunker (new flow)

Run modes
---------
Full re-index (original behaviour — wipes DB first):
    python pipeline.py --mode full

Incremental (default) — skips already-indexed acts:
    python pipeline.py
    python pipeline.py --mode incremental

Force re-index specific acts (deletes their chunks, then re-indexes):
    python pipeline.py --mode incremental --force "Legal Aid To Legal Rights" "Bnss"
"""

import argparse
import os
import pdfplumber

from config        import ACTS_FOLDER
from extractor     import extract_text
from sectioner     import split_sections, tag_chapters
from chunker       import split_into_fine_chunks
from book_chunker  import split_book_into_chunks
from detector      import detect_document_type
from store         import (
    get_client,
    clear_collection,
    upsert_chunks,
    list_indexed_acts,
    delete_act,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sample_text(pdf_path: str, pages: int = 5) -> str:
    """Extract text from the first *pages* pages for type detection."""
    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[:pages]:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)


def _pdf_to_act_name(filename: str) -> str:
    return filename.replace(".pdf", "").replace("-", " ").title()


# ── Per-document processors (unchanged logic) ─────────────────────────────────

def process_act(pdf_path: str, act_name: str) -> None:
    print("1/4  Extracting text …")
    raw_text = extract_text(pdf_path)

    print("2/4  Splitting into sections …")
    sections = split_sections(raw_text)
    sections = tag_chapters(raw_text, sections)
    print(f"     Found {len(sections)} sections.")

    print("3/4  Fine-chunking …")
    all_chunks: list[dict] = []
    for sec in sections:
        all_chunks.extend(split_into_fine_chunks(sec))
    print(f"     Produced {len(all_chunks)} chunks.")

    print("4/4  Embedding and storing …")
    upsert_chunks(all_chunks, act_name)


def process_book(pdf_path: str, act_name: str) -> None:
    print("1/3  Extracting text …")
    raw_text = extract_text(pdf_path)

    print("2/3  Chunking book chapters …")
    all_chunks = split_book_into_chunks(raw_text)
    print(f"     Produced {len(all_chunks)} chunks.")

    print("3/3  Embedding and storing …")
    upsert_chunks(all_chunks, act_name)


def process_pdf(pdf_path: str, act_name: str) -> None:
    """Auto-detect document type and route to the correct processing flow."""
    print(f"\n{'─' * 60}")
    print(f"Processing: {act_name}")
    print(f"{'─' * 60}")

    sample   = _sample_text(pdf_path)
    doc_type = detect_document_type(sample)

    if doc_type == "act":
        process_act(pdf_path, act_name)
    else:
        process_book(pdf_path, act_name)


# ── Run modes ─────────────────────────────────────────────────────────────────

def run_full(acts_folder: str = ACTS_FOLDER) -> None:
    """
    Wipe the DB and re-index every PDF from scratch.
    Original behaviour — use when you want a clean slate.
    """
    pdf_files = [f for f in os.listdir(acts_folder) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in: {acts_folder}")
        return

    client = get_client()
    clear_collection(client)

    for filename in pdf_files:
        pdf_path = os.path.join(acts_folder, filename)
        act_name = _pdf_to_act_name(filename)
        process_pdf(pdf_path, act_name)

    print("\n✅  All documents processed (full re-index).")


def run_incremental(
    acts_folder: str = ACTS_FOLDER,
    force: list[str] | None = None,
) -> None:
    """
    Add new PDFs to the existing DB without touching already-indexed acts.

    Args:
        acts_folder: Folder containing PDF files.
        force:       Optional list of act names to forcibly re-index even if
                     they're already in the DB (e.g. after editing a PDF).
    """
    pdf_files = [f for f in os.listdir(acts_folder) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in: {acts_folder}")
        return

    client       = get_client()
    already_done = list_indexed_acts(client)
    force_set    = {name.lower() for name in (force or [])}

    print(f"Already indexed: {len(already_done)} act(s).")
    if already_done:
        print("  " + ", ".join(sorted(already_done)))

    new_count = 0
    for filename in pdf_files:
        pdf_path = os.path.join(acts_folder, filename)
        act_name = _pdf_to_act_name(filename)

        if act_name.lower() in force_set:
            print(f"\n  ⚡ Force re-indexing: {act_name}")
            delete_act(client, act_name)           # remove old chunks first
        elif act_name in already_done:
            print(f"\n  ⏭  Skipping (already indexed): {act_name}")
            continue

        process_pdf(pdf_path, act_name)
        new_count += 1

    if new_count == 0:
        print("\n✅  Nothing new to index.")
    else:
        print(f"\n✅  Indexed {new_count} new/updated document(s).")


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index legal PDFs into ChromaDB."
    )
    parser.add_argument(
        "--mode",
        choices=["full", "incremental"],
        default="incremental",
        help=(
            "full: wipe DB and re-index everything. "
            "incremental (default): only add new PDFs."
        ),
    )
    parser.add_argument(
        "--force",
        nargs="+",
        metavar="ACT_NAME",
        default=None,
        help=(
            "With --mode incremental, force these act names to be "
            "re-indexed even if already present. "
            'Example: --force "Legal Aid To Legal Rights" "Bnss"'
        ),
    )
    parser.add_argument(
        "--folder",
        default=ACTS_FOLDER,
        help="Override the acts folder path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.mode == "full":
        run_full(args.folder)
    else:
        run_incremental(args.folder, force=args.force)
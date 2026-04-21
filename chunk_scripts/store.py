"""
store.py — ChromaDB persistence layer

Responsibilities:
  - Create / get the ChromaDB collection
  - Upsert chunks in batches (embed → store)
  - Provide a helper to wipe the collection before a full re-index
  - NEW: list indexed acts, delete a single act, skip already-indexed acts
"""

import hashlib
import chromadb
from embedder import embed
from config   import DB_PATH, COLLECTION, BATCH_SIZE


def _get_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    return client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def get_client() -> chromadb.PersistentClient:
    """Return a ChromaDB persistent client pointed at DB_PATH."""
    return chromadb.PersistentClient(path=DB_PATH)


def clear_collection(client: chromadb.PersistentClient) -> None:
    """Drop the collection if it exists (safe no-op otherwise)."""
    try:
        client.delete_collection(COLLECTION)
        print("  Cleared old collection.")
    except Exception:
        pass


# ── NEW: Incremental helpers ──────────────────────────────────────────────────

def list_indexed_acts(client: chromadb.PersistentClient) -> set[str]:
    """
    Return the set of act_name values already present in the collection.
    Uses a metadata query so it works even for large collections.
    """
    collection = _get_collection(client)
    # Fetch all metadata (no embeddings needed — just metadata)
    result = collection.get(include=["metadatas"])
    return {m["act_name"] for m in result["metadatas"] if "act_name" in m}


def delete_act(client: chromadb.PersistentClient, act_name: str) -> None:
    """
    Remove all chunks belonging to *act_name* from the collection.
    Use this to force a re-index of a single document without wiping everything.
    """
    collection = _get_collection(client)
    result = collection.get(
        where={"act_name": act_name},
        include=[],          # IDs only
    )
    ids = result.get("ids", [])
    if ids:
        collection.delete(ids=ids)
        print(f"  Deleted {len(ids)} chunks for '{act_name}'.")
    else:
        print(f"  No chunks found for '{act_name}' — nothing deleted.")


def _stable_chunk_id(act_name: str, chunk: dict, index: int) -> str:
    """
    Build a deterministic ID from act name + position + chunk content.
    The global index is always included so two chunks with identical text
    (e.g. repeated boilerplate) never collide within the same document.
    """
    raw = (
        f"{act_name}__{index}__"
        f"{chunk.get('chunk_type', '')}__"
        f"{chunk.get('section_number', '')}__"
        f"{chunk.get('text', '')[:120]}"
    )
    digest = hashlib.sha1(raw.encode()).hexdigest()[:12]
    safe_name = act_name.replace(" ", "_")
    return f"{safe_name}__{index:05d}__{digest}"


# ── Upsert (unchanged public signature) ──────────────────────────────────────

def upsert_chunks(chunks: list[dict], act_name: str) -> None:
    """
    Embed and upsert *chunks* into ChromaDB in batches.

    Args:
        chunks:   List of chunk dicts produced by ``chunker.split_into_fine_chunks``
                  or ``book_chunker.split_book_into_chunks``.
        act_name: Human-readable act name stored in each document's metadata.

    IDs are now content-derived (SHA-1 of act + position + text prefix) so
    upserting the same document twice is fully idempotent — no duplicates.
    """
    client     = get_client()
    collection = _get_collection(client)

    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]

        texts = []
        for c in batch:
            ch    = c.get("chapter", "General")
            sec   = c.get("section_number", "")
            title = c.get("section_title", "Guide")
            enriched = (
                f"Source: {act_name}. Chapter: {ch}. "
                f"Section: {sec} - {title}\n\n{c['text']}"
            )
            texts.append(enriched)

        vectors = embed(texts)

        ids, metas = [], []
        for offset, c in enumerate(batch):
            global_idx = batch_start + offset
            ids.append(_stable_chunk_id(act_name, c, global_idx))
            metas.append({
                "act_name":       act_name,
                "chapter":        c.get("chapter", ""),
                "section_number": c.get("section_number", 0),
                "section_title":  c.get("section_title", ""),
                "chunk_type":     c.get("chunk_type", ""),
                "subsection":     c.get("subsection", ""),
                "clause":         c.get("clause", ""),
                "annotation":     c.get("annotation", ""),
            })

        collection.upsert(ids=ids, documents=texts, embeddings=vectors, metadatas=metas)

        sec_range = f"{batch[0].get('section_number', '?')}–{batch[-1].get('section_number', '?')}"
        print(f"  Batch {batch_start // BATCH_SIZE + 1}: {len(batch)} chunks  (sections {sec_range})")

    print(f"\n  Total chunks in collection: {collection.count()}")
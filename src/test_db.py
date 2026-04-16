"""
inspect_db.py  —  peek inside your ChromaDB
--------------------------------------------
Run:
    python inspect_db.py                        # overview
    python inspect_db.py --section 34           # show one section's full text
    python inspect_db.py --query "notice period rent"  # semantic search
"""

import argparse
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

DB_PATH        = "./acts_db"
COLLECTION     = "indian_acts"


def get_collection():
    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_collection(COLLECTION, embedding_function=ef)


def overview(col):
    total = col.count()
    print(f"\nTotal chunks in DB: {total}\n")

    # Pull everything (metadata only, no embeddings)
    all_items = col.get(include=["metadatas"])

    # Group by act
    acts = {}
    for meta in all_items["metadatas"]:
        name = meta["act_name"]
        acts.setdefault(name, []).append(meta["section_number"])

    for act, sections in acts.items():
        sections.sort()
        print(f"  {act}")
        print(f"    {len(sections)} sections  |  {min(sections)}–{max(sections)}")
        print(f"    Sections: {sections}\n")


def show_section(col, section_num: int):
    results = col.get(
        where={"section_number": section_num},
        include=["metadatas", "documents"]
    )
    if not results["ids"]:
        print(f"Section {section_num} not found.")
        return
    for meta, doc in zip(results["metadatas"], results["documents"]):
        print(f"\nSection {meta['section_number']}: {meta['section_title']}")
        print(f"Act:     {meta['act_name']}")
        print(f"Chapter: {meta['chapter']}")
        print(f"Domain:  {meta['domain']}")
        print(f"\n{doc}\n")


def search(col, query: str, n: int = 5):
    results = col.query(query_texts=[query], n_results=n,
                        include=["metadatas", "documents", "distances"])
    print(f"\nTop {n} results for: '{query}'\n")
    for meta, doc, dist in zip(results["metadatas"][0],
                                results["documents"][0],
                                results["distances"][0]):
        print(f"  [{1-dist:.3f}] Sec {meta['section_number']}: {meta['section_title']}")
        print(f"         {meta['act_name']}  |  {meta['chapter']}")
        print(f"         {doc[:200].replace(chr(10),' ')}...")
        print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--section", type=int, help="Show full text of a section number")
    p.add_argument("--query",   type=str, help="Semantic search query")
    p.add_argument("--top",     type=int, default=5, help="Results to show (default 5)")
    args = p.parse_args()

    col = get_collection()

    if args.section:
        show_section(col, args.section)
    elif args.query:
        search(col, args.query, args.top)
    else:
        overview(col)


if __name__ == "__main__":
    main()
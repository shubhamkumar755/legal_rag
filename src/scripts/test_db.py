import chromadb

DB_PATH = r"D:\College\legal_rag\legal_rag\db"
COLLECTION = "indian_acts"

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(COLLECTION)

results = collection.get(
    where={"section_number": 43}
)

print("Found:", len(results["documents"]))

for doc, meta in zip(results["documents"], results["metadatas"]):
    print("\n---")
    print(meta["act_name"], "Section", meta["section_number"])
    print(meta["section_title"])
    print(doc[:500])
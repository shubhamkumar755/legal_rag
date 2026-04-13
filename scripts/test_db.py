import chromadb
from collections import Counter

DB_PATH = r"D:\College\legal_rag\legal_rag\acts_db"
COLLECTION = "indian_acts"

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(COLLECTION)

# Load metadata only (faster)
data = collection.get(include=["metadatas"])

metas = data["metadatas"]

total_chunks = len(metas)

act_counter = Counter()

for m in metas:
    act_counter[m["act_name"]] += 1

print("\nTotal chunks:", total_chunks)
print("-" * 40)

for act, count in act_counter.most_common():
    percent = (count / total_chunks) * 100
    print(f"{act:35} {count:5} chunks   {percent:6.2f}%")
"""
retrieval/config.py
===================
All constants + model/client singletons.
Import from here everywhere else — models load only once.
"""

import os
import chromadb
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import CrossEncoder
import google.generativeai as genai
# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]   # go to project root
DB_PATH = BASE_DIR / "db"
COLLECTION      = "indian_acts"
BM25_CACHE_PATH = os.path.join(DB_PATH, "bm25_index.pkl")

# ── Retrieval widths ──────────────────────────────────────────────────────────
CANDIDATE_K  = 100   # results pulled from Vector + BM25 each
RRF_K        = 60    # RRF constant
FUSION_TOP_N = 20    # candidates sent to Gemini re-ranker
LLM_TOP_N    = 10    # results Gemini returns
FINAL_TOP_K  = 5     # results after cross-encoder

# ── Metadata boosts ───────────────────────────────────────────────────────────
SECTION_BOOST = 1.0
ACT_BOOST     = 0.5

# ── Stage 5 guard ─────────────────────────────────────────────────────────────
# Cross-encoder is auto-disabled for the session if it exceeds this threshold.
# Set to None to always run it.
RERANKER_TIMEOUT_S = None

# ── Gemini ────────────────────────────────────────────────────────────────────
load_dotenv()
GEMINI_MODEL   = "gemini-2.5-flash-lite"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ── Singletons (loaded once at import) ───────────────────────────────────────
if os.getenv("EMBEDDING_MODEL", "").startswith("gemini"):
    genai.configure(api_key=GEMINI_API_KEY)

    class _GeminiEmbeddings:
        def embed_documents(self, texts):
            return [
                genai.embed_content(
                    model="models/gemini-embedding-001",
                    content=t
                )["embedding"]
                for t in texts
            ]

        def embed_query(self, text):
            return genai.embed_content(
                model="models/gemini-embedding-001",
                content=text
            )["embedding"]

    EMBEDDINGS = _GeminiEmbeddings()

else:
    EMBEDDINGS = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
RERANKER   = CrossEncoder("BAAI/bge-reranker-base")

gemini_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=0.0,
    google_api_key=GEMINI_API_KEY,
)

chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection    = chroma_client.get_collection(COLLECTION)
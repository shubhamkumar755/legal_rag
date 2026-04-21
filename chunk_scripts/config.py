"""
config.py — Central configuration for IndiaCode PDF → ChromaDB pipeline
"""

import os
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]   # go to project root
DATA_PATH = BASE_DIR / "data"
ACTS_FOLDER = os.getenv("ACTS_FOLDER", DATA_PATH / "acts")
DB_PATH     = os.getenv("DB_PATH",     BASE_DIR / "db")

# ── ChromaDB ──────────────────────────────────────────────────────────────────
COLLECTION = "indian_acts"

# ── Chunking ──────────────────────────────────────────────────────────────────
MIN_CHUNK_CHARS = 80
OVERLAP_CHARS   = 40
BATCH_SIZE      = 20

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "gemini-embedding-001"
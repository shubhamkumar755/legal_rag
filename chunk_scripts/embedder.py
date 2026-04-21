"""
embedder.py — Thin wrapper around the HuggingFace embedding model

Responsibilities:
  - Load and cache the embedding model once
  - Expose a simple embed() function that takes a list of strings
    and returns a list of float vectors

Keeping this isolated means you can swap the backend
(OpenAI, Cohere, local Ollama, …) in one place.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL

# Module-level singleton — loaded once, reused for every batch.
_model: HuggingFaceEmbeddings | None = None


def _get_model() -> HuggingFaceEmbeddings:
    global _model
    if _model is None:
        print(f"  Loading embedding model: {EMBEDDING_MODEL}")
        _model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of *texts* and return a parallel list of float vectors.

    Args:
        texts: Non-empty list of strings to embed.

    Returns:
        List of embedding vectors (same length as *texts*).
    """
    if not texts:
        return []
    return _get_model().embed_documents(texts)
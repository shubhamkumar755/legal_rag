import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL
import time
load_dotenv()

# Singleton for HF model
_hf_model: HuggingFaceEmbeddings | None = None


def _get_hf_model() -> HuggingFaceEmbeddings:
    global _hf_model
    if _hf_model is None:
        print(f"  Loading HF embedding model: {EMBEDDING_MODEL}")
        _hf_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _hf_model


def _embed_huggingface(texts: list[str]) -> list[list[float]]:
    return _get_hf_model().embed_documents(texts)


def _embed_gemini(texts: list[str]) -> list[list[float]]:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    embeddings = []
    for text in texts:
        res = genai.embed_content(
            model=f"models/{EMBEDDING_MODEL}",
            content=text
        )
        embeddings.append(res["embedding"])
        time.sleep(0.5)
    return embeddings


def embed(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    if EMBEDDING_MODEL.startswith("gemini"):
        return _embed_gemini(texts)
    else:
        return _embed_huggingface(texts)
!pip install -q langchain langchain-core langchain-community sentence-transformers faiss-cpu
from __future__ import annotations

from typing import Optional
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

_EMBEDDINGS: Optional[HuggingFaceBgeEmbeddings] = None


def get_embedding_model() -> HuggingFaceBgeEmbeddings:
    """
    Load and cache the embedding model.

    Recommended by the project spec:
    - BAAI/bge-small-en-v1.5
    """
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        model_name = "BAAI/bge-small-en-v1.5"
        _EMBEDDINGS = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _EMBEDDINGS

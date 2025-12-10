!pip install -q -U langchain langchain-community langchain-core
!pip install -q -U faiss-cpu sentence-transformers transformers
from __future__ import annotations

from typing import Optional
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


_EMBEDDINGS: Optional[HuggingFaceBgeEmbeddings] = None


def get_embedding_model(
    model_name: str = "BAAI/bge-small-en-v1.5",
    device: str = "cpu",
) -> HuggingFaceBgeEmbeddings:
    """
    Load and cache the embedding model.

    Project recommendation:
    - BAAI/bge-small-en-v1.5
    """
    global _EMBEDDINGS

    if _EMBEDDINGS is None:
        _EMBEDDINGS = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

    return _EMBEDDINGS

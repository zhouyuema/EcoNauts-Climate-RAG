!pip install -q -U langchain langchain-community langchain-core
!pip install -q -U faiss-cpu sentence-transformers transformers
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List

try:
    from langchain_core.documents import Document
except ImportError:
    # Older LangChain fallback
    from langchain.schema import Document

from rag_pipeline.vector_store import FaissVectorStore  # use absolute import


@dataclass
class SimpleRetriever:
    vector_store: FaissVectorStore

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        if k < 3:
            k = 3
        return self.vector_store.similarity_search(query, k=k)

    @staticmethod
    def build_sources(
        docs: List[Document],
        max_snippet_chars: int = 220,
    ) -> List[Dict[str, Any]]:
        sources: List[Dict[str, Any]] = []
        for d in docs:
            meta = d.metadata or {}
            snippet = (d.page_content or "")[:max_snippet_chars]
            sources.append(
                {
                    "id": meta.get("source", "unknown"),
                    "page": meta.get("page"),
                    "country": meta.get("country"),
                    "year": meta.get("year"),
                    "snippet": snippet,
                }
            )
        return sources

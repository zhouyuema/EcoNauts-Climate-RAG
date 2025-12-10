!pip install -q -U langchain langchain-community langchain-core
!pip install -q -U faiss-cpu sentence-transformers transformers

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings


DEFAULT_INDEX_DIR = Path("vector_store/faiss_index")


@dataclass
class FaissVectorStore:
    """
    Thin wrapper around LangChain FAISS vector store.
    """
    embedding: Embeddings
    index: Optional[FAISS] = None

    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            return

        if self.index is None:
            self.index = FAISS.from_documents(documents, self.embedding)
        else:
            self.index.add_documents(documents)

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        if self.index is None:
            raise ValueError("Vector store is empty. Call add_documents() first.")
        return self.index.similarity_search(query, k=k)

    def save_local(self, index_dir: Path = DEFAULT_INDEX_DIR) -> None:
        if self.index is None:
            raise ValueError("No FAISS index to save.")
        index_dir.mkdir(parents=True, exist_ok=True)
        self.index.save_local(str(index_dir))

    @classmethod
    def load_local(cls, embedding: Embeddings, index_dir: Path = DEFAULT_INDEX_DIR) -> "FaissVectorStore":
        if not index_dir.exists():
            raise FileNotFoundError(f"Index folder not found: {index_dir}")

        store = cls(embedding=embedding)
        # LangChain requires this flag for local deserialization
        store.index = FAISS.load_local(
            str(index_dir),
            embedding,
            allow_dangerous_deserialization=True,
        )
        return store

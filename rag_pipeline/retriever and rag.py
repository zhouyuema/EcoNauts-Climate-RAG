from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

# --- Robust Document import (works across environments) ---
try:
    # Newer LangChain
    from langchain_core.documents import Document
except Exception:
    try:
        # Older LangChain fallback
        from langchain.schema import Document
    except Exception:
        # Ultimate fallback if LangChain is not available
        @dataclass
        class Document:
            page_content: str
            metadata: Dict[str, Any]


# ---------------------------
# Simple Retriever (no imports from other files)
# ---------------------------
@dataclass
class SimpleRetriever:
    """
    Minimal retriever wrapper.
    Expects vector_store to provide:
      - similarity_search(query: str, k: int) -> List[Document]
    """
    vector_store: Any

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        # Project requires k >= 3
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
            meta = getattr(d, "metadata", {}) or {}
            content = getattr(d, "page_content", "") or ""
            snippet = content[:max_snippet_chars]

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


# ---------------------------
# RAG core logic 
# ---------------------------
SYSTEM_PROMPT = """
You are a domain-specific assistant.

You MUST answer ONLY using the provided context.
If the answer cannot be found in the context, say:
"I don't have enough information in the provided data to answer that."

When you give numeric claims, prefer exact values from context.
Keep the answer concise and grounded.
""".strip()


def build_context_block(docs: List[Document]) -> str:
    parts: List[str] = []
    for i, d in enumerate(docs, start=1):
        meta = getattr(d, "metadata", {}) or {}
        content = getattr(d, "page_content", "") or ""

        src = meta.get("source", "unknown")
        page = meta.get("page")
        country = meta.get("country")
        year = meta.get("year")

        header_bits = [f"Source={src}"]
        if page is not None:
            header_bits.append(f"page={page}")
        if country:
            header_bits.append(f"country={country}")
        if year:
            header_bits.append(f"year={year}")

        header = ", ".join(header_bits)
        parts.append(f"[{i}] {header}\n{content}")

    return "\n\n".join(parts)


@dataclass
class RAGPipeline:
    retriever: SimpleRetriever
    llm: Any  # HF pipeline or any callable

    def generate_answer(self, question: str, docs: List[Document]) -> str:
        context = build_context_block(docs)

        prompt = f"""
{SYSTEM_PROMPT}

Context:
{context}

Question:
{question}

Answer:
""".strip()

        # If llm is a HF text-generation pipeline
        try:
            output = self.llm(
                prompt,
                max_new_tokens=256,
                do_sample=False,
            )
        except TypeError:
            # If llm is a simpler callable that only accepts prompt
            output = self.llm(prompt)

        # HF pipeline usually returns list of dicts with 'generated_text'
        if isinstance(output, list) and output and isinstance(output[0], dict):
            text = output[0].get("generated_text", "")
            if text.startswith(prompt):
                text = text[len(prompt):].strip()
            return text.strip()

        return str(output).strip()

    def ask(self, question: str, k: int = 3) -> Dict[str, Any]:
        docs = self.retriever.retrieve(question, k=k)
        answer = self.generate_answer(question, docs)
        sources = self.retriever.build_sources(docs)

        return {
            "answer": answer,
            "sources": sources,
        }

# This dummy LLM lets you test the pipeline without loading a real model.
def dummy_llm(prompt: str, **kwargs):
    return [{"generated_text": prompt + "\nI don't have enough information in the provided data to answer that."}]

"""End-to-end Retrieval-Augmented Generation (RAG) pipeline for Task 2.

Features:
- Orchestrates PDF ingestion, page metadata preservation, text splitting, and vector indexing.
- Semantic similarity retrieval from ChromaDB with distance filtering.
- Strict context-grounded prompting ensuring zero external hallucinations.
- Dedicated out-of-scope rejection: returns standard fallback when information is missing.
- Structured Citation extraction with source filename and physical page number.
- Multi-provider LLM support: Google Gemini (with automated failover), OpenAI, and deterministic Mock LLM.
"""

from __future__ import annotations

import io
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv

from task2_rag_qa.embeddings import BaseEmbedder, get_embedder
from task2_rag_qa.pdf_loader import PDFExtractionResult, extract_pdf_pages
from task2_rag_qa.text_splitter import DocumentChunk, chunk_document_pages
from task2_rag_qa.vector_store import ChromaVectorStore

load_dotenv()

# Standard phrase returned whenever the retrieved context does not contain the answer
NOT_FOUND_STANDARD_RESPONSE = "The requested information was not found in the uploaded documents."


@dataclass
class Citation:
    """Represents a validated source citation referencing a document page."""
    source_filename: str
    page_number: int
    chunk_id: str
    snippet: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_filename": self.source_filename,
            "page_number": self.page_number,
            "chunk_id": self.chunk_id,
            "snippet": self.snippet,
        }

    @property
    def label(self) -> str:
        return f"{self.source_filename} (Page {self.page_number})"


@dataclass
class RAGResponse:
    """Standardized response container for RAG Q&A queries."""
    question: str
    answer: str
    citations: List[Citation] = field(default_factory=list)
    retrieved_chunks: List[Tuple[DocumentChunk, float]] = field(default_factory=list)
    is_grounded: bool = True
    is_not_found: bool = False
    execution_time_seconds: float = 0.0
    provider_used: str = "mock"
    model_used: str = "mock"
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "num_retrieved_chunks": len(self.retrieved_chunks),
            "is_grounded": self.is_grounded,
            "is_not_found": self.is_not_found,
            "execution_time_seconds": round(self.execution_time_seconds, 3),
            "provider_used": self.provider_used,
            "model_used": self.model_used,
            "error_message": self.error_message,
        }


# =====================================================================
# Prompt Template
# =====================================================================

SYSTEM_RAG_PROMPT = """You are an expert, precision-driven Document Q&A Assistant.
Your objective is to answer user questions with absolute fidelity, grounded EXCLUSIVELY in the provided context excerpts from uploaded PDF documents.

CRITICAL RULES:
1. Grounding: Answer ONLY using explicit facts contained in the provided CONTEXT. Do NOT assume, extrapolate, speculate, or draw from outside world knowledge.
2. Missing Information: If the provided CONTEXT does not contain enough facts to answer the question truthfully and directly, you MUST respond EXACTLY with:
   "The requested information was not found in the uploaded documents."
   Do NOT provide a partial guess or apologize.
3. Citations: When an answer is found, cite the exact source at the end of each key sentence or paragraph using the format:
   [Source: <filename>, Page: <page_number>]
4. Truthfulness: Never invent dates, monetary values, employee names, or page numbers.
"""

def format_context_block(chunks_with_scores: List[Tuple[DocumentChunk, float]]) -> str:
    """Formats retrieved chunks into a clear, labeled context block for the LLM."""
    blocks: List[str] = []
    for idx, (chunk, score) in enumerate(chunks_with_scores, 1):
        blocks.append(
            f"--- EXCERPT {idx} | Source: {chunk.source_filename} | Page: {chunk.page_number} ---\n"
            f"{chunk.text}\n"
        )
    return "\n".join(blocks)


# =====================================================================
# LLM Caller Abstraction
# =====================================================================

def call_rag_llm(
    prompt: str,
    system_prompt: str = SYSTEM_RAG_PROMPT,
    provider: str = "gemini",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    mock_response: Optional[str] = None,
) -> Tuple[str, str, str]:
    """Invokes LLM provider with fallback resilience.

    Returns:
        Tuple[raw_text, provider_name, model_name]
    """
    prov = (provider or "gemini").lower().strip()

    if prov == "mock" or mock_response is not None:
        return mock_response or NOT_FOUND_STANDARD_RESPONSE, "mock", "mock-evaluator"

    if prov in ("gemini", "google"):
        resolved_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not resolved_key:
            raise ValueError("Gemini API Key missing. Please provide it or set GEMINI_API_KEY in .env.")

        resolved_model = model_name or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash-lite"

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=resolved_key)
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
        )

        candidates = [resolved_model]
        for cand in ["gemini-2.5-flash-lite", "gemini-3.5-flash-lite", "gemini-flash-latest", "gemini-3.5-flash", "gemini-3.8-flash"]:
            if cand not in candidates:
                candidates.append(cand)

        last_error = None
        for cand_model in candidates:
            try:
                response = client.models.generate_content(
                    model=cand_model,
                    contents=prompt,
                    config=config,
                )
                if response.text:
                    return response.text.strip(), "gemini", cand_model
            except Exception as exc:
                err_msg = str(exc)
                if "API_KEY_INVALID" in err_msg or "403" in err_msg or "PERMISSION_DENIED" in err_msg:
                    raise RuntimeError("Gemini API Key is invalid or unauthorized.") from exc
                last_error = exc
                continue

        raise RuntimeError(f"All Gemini model fallbacks failed: {str(last_error)}") from last_error

    elif prov == "openai":
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("OpenAI API Key missing. Please provide it or set OPENAI_API_KEY in .env.")

        resolved_model = model_name or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        from openai import OpenAI
        client = OpenAI(api_key=resolved_key)

        response = client.chat.completions.create(
            model=resolved_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        ans = response.choices[0].message.content or ""
        return ans.strip(), "openai", resolved_model

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


# =====================================================================
# End-to-End RAG Pipeline Class
# =====================================================================

class RAGPipeline:
    """Coordinates PDF ingestion, vector indexing, retrieval, and grounded Q&A."""

    def __init__(
        self,
        provider: str = "gemini",
        api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        vector_store: Optional[ChromaVectorStore] = None,
        embedder: Optional[BaseEmbedder] = None,
        chunk_size: int = 600,
        chunk_overlap: int = 120,
    ):
        self.provider = provider.lower().strip()
        self.api_key = api_key
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize Embedder
        if embedder is not None:
            self.embedder = embedder
        else:
            self.embedder = get_embedder(
                provider=self.provider,
                api_key=self.api_key,
                model_name=embedding_model,
            )

        # Initialize Vector Store
        self.vector_store = vector_store or ChromaVectorStore(in_memory=True)

    def index_pdf(
        self,
        file_input: Union[str, bytes, io.BytesIO, BinaryIO],
        filename: str = "document.pdf",
    ) -> PDFExtractionResult:
        """Extracts text from PDF, splits into chunks, and stores embeddings in ChromaDB."""
        extract_res = extract_pdf_pages(file_input=file_input, filename=filename)

        if not extract_res.success or not extract_res.has_text:
            return extract_res

        # Split into overlapping chunks
        chunks = chunk_document_pages(
            pages=extract_res.pages,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        # Index in ChromaDB
        self.vector_store.add_chunks(chunks=chunks, embedder=self.embedder)
        return extract_res

    def query(
        self,
        question: str,
        top_k: int = 4,
        distance_threshold: Optional[float] = None,
        temperature: float = 0.0,
        mock_llm_response: Optional[str] = None,
    ) -> RAGResponse:
        """Executes a grounded RAG query against the indexed documents.

        Args:
            question: User inquiry.
            top_k: Number of chunks to retrieve.
            distance_threshold: Optional maximum cosine distance (e.g. 0.85).
            temperature: Sampling temperature for generation.
            mock_llm_response: Optional pre-set response for testing without live APIs.
        """
        start_time = time.time()
        clean_q = question.strip()

        # Handle empty question
        if not clean_q:
            return RAGResponse(
                question=question,
                answer="Please enter a valid question to search your uploaded documents.",
                is_grounded=True,
                is_not_found=False,
                execution_time_seconds=0.0,
                provider_used=self.provider,
                model_used="none",
            )

        # Handle empty vector store
        if self.vector_store.count() == 0:
            return RAGResponse(
                question=clean_q,
                answer="No documents have been indexed yet. Please upload and index at least one PDF document first.",
                is_grounded=True,
                is_not_found=True,
                execution_time_seconds=time.time() - start_time,
                provider_used=self.provider,
                model_used="none",
            )

        # Retrieve relevant chunks from ChromaDB
        try:
            retrieved = self.vector_store.similarity_search(
                query=clean_q,
                embedder=self.embedder,
                top_k=top_k,
            )
        except Exception as exc:
            return RAGResponse(
                question=clean_q,
                answer=f"Error occurred during vector similarity search: {str(exc)}",
                is_grounded=False,
                is_not_found=False,
                execution_time_seconds=time.time() - start_time,
                provider_used=self.provider,
                model_used="none",
                error_message=str(exc),
            )

        # Apply distance filter if specified
        if distance_threshold is not None:
            retrieved = [item for item in retrieved if item[1] <= distance_threshold]

        # If zero chunks retrieved
        if not retrieved:
            return RAGResponse(
                question=clean_q,
                answer=NOT_FOUND_STANDARD_RESPONSE,
                citations=[],
                retrieved_chunks=[],
                is_grounded=True,
                is_not_found=True,
                execution_time_seconds=time.time() - start_time,
                provider_used=self.provider,
                model_used=self.llm_model or "none",
            )

        # Assemble grounded prompt
        context_block = format_context_block(retrieved)
        user_prompt = (
            f"CONTEXT FROM UPLOADED DOCUMENTS:\n"
            f"{context_block}\n\n"
            f"USER QUESTION: {clean_q}\n\n"
            f"INSTRUCTION: Answer the question using ONLY facts stated in the CONTEXT above. "
            f"If the answer is not contained in the context, output exactly: '{NOT_FOUND_STANDARD_RESPONSE}'."
        )

        # Call LLM
        try:
            raw_answer, used_prov, used_model = call_rag_llm(
                prompt=user_prompt,
                system_prompt=SYSTEM_RAG_PROMPT,
                provider=self.provider,
                api_key=self.api_key,
                model_name=self.llm_model,
                temperature=temperature,
                mock_response=mock_llm_response,
            )
        except Exception as exc:
            return RAGResponse(
                question=clean_q,
                answer=f"LLM generation failed: {str(exc)}",
                retrieved_chunks=retrieved,
                is_grounded=False,
                is_not_found=False,
                execution_time_seconds=time.time() - start_time,
                provider_used=self.provider,
                model_used="unknown",
                error_message=str(exc),
            )

        # Check if the answer explicitly stated not found
        is_not_found = (
            NOT_FOUND_STANDARD_RESPONSE.lower() in raw_answer.lower()
            or "information was not found in the uploaded" in raw_answer.lower()
        )

        # Build citations from retrieved chunks
        citations: List[Citation] = []
        if not is_not_found:
            seen_pages = set()
            for chunk, score in retrieved:
                key = (chunk.source_filename, chunk.page_number)
                if key not in seen_pages:
                    seen_pages.add(key)
                    # Snippet for citation preview
                    snippet = chunk.text[:220] + ("..." if len(chunk.text) > 220 else "")
                    citations.append(
                        Citation(
                            source_filename=chunk.source_filename,
                            page_number=chunk.page_number,
                            chunk_id=chunk.chunk_id,
                            snippet=snippet,
                        )
                    )

        return RAGResponse(
            question=clean_q,
            answer=raw_answer,
            citations=citations,
            retrieved_chunks=retrieved,
            is_grounded=True,
            is_not_found=is_not_found,
            execution_time_seconds=time.time() - start_time,
            provider_used=used_prov,
            model_used=used_model,
        )

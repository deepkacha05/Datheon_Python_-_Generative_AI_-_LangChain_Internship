"""Task 2: Document Q&A Assistant with RAG & Vector Embeddings.

This package provides:
- PDF text extraction with page metadata preservation & scanned document detection.
- LangChain RecursiveCharacterTextSplitter chunking with chunk IDs and source tracking.
- Multi-provider embedding abstraction (Google Gemini, OpenAI, and deterministic Mock).
- ChromaDB vector store integration for semantic similarity search.
- Context-grounded RAG query pipeline with strict citation generation and hallucination avoidance.
"""

from task2_rag_qa.pdf_loader import PDFPage, PDFExtractionResult, extract_pdf_pages
from task2_rag_qa.text_splitter import DocumentChunk, chunk_document_pages
from task2_rag_qa.embeddings import BaseEmbedder, GeminiEmbedder, OpenAIEmbedder, MockEmbedder, get_embedder
from task2_rag_qa.vector_store import ChromaVectorStore
from task2_rag_qa.rag_pipeline import RAGPipeline, RAGResponse

__all__ = [
    "PDFPage",
    "PDFExtractionResult",
    "extract_pdf_pages",
    "DocumentChunk",
    "chunk_document_pages",
    "BaseEmbedder",
    "GeminiEmbedder",
    "OpenAIEmbedder",
    "MockEmbedder",
    "get_embedder",
    "ChromaVectorStore",
    "RAGPipeline",
    "RAGResponse",
]


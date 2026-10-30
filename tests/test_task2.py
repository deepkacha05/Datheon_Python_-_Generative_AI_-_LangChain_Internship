"""Unit and integration tests for Task 2: Document Q&A Assistant with RAG & Vector Embeddings.

Validates:
- PDF text extraction, page metadata preservation, and corrupted file handling.
- Scanned PDF detection and OCR requirement diagnostics.
- Text splitting with LangChain RecursiveCharacterTextSplitter and chunk metadata propagation.
- Deterministic mock embeddings and vector normalization.
- ChromaDB vector indexing and semantic similarity search.
- Context-grounded RAG answers and source citations (filename + physical page number).
- Out-of-scope query rejection ("The requested information was not found in the uploaded documents").
- Empty query and empty collection handling.
- Zero paid API dependency (all tests run completely offline).
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock
import pypdf
import pytest

from task2_rag_qa.embeddings import MockEmbedder, get_embedder
from task2_rag_qa.pdf_loader import PDFPage, extract_pdf_pages
from task2_rag_qa.rag_pipeline import (
    NOT_FOUND_STANDARD_RESPONSE,
    RAGPipeline,
    RAGResponse,
    format_context_block,
)
from task2_rag_qa.sample_documents.generator import build_pure_python_pdf
from task2_rag_qa.text_splitter import DocumentChunk, chunk_document_pages
from task2_rag_qa.vector_store import ChromaVectorStore


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Generates a valid 2-page PDF in-memory for testing."""
    p1 = "Datheon Artificial Intelligence Handbook.\nWorking hours are 9:30 AM to 6:30 PM IST."
    p2 = "Remote Work Policy.\nEngineers can work remotely up to 3 days per week."
    return build_pure_python_pdf([p1, p2])


@pytest.fixture
def scanned_blank_pdf_bytes() -> bytes:
    """Generates a PDF with blank pages simulating an un-OCRed scanned document."""
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


# =====================================================================
# 1. PDF Loader & Scanned PDF Detection Tests
# =====================================================================

def test_pdf_loader_valid_extraction(sample_pdf_bytes):
    """Verify page-level metadata, page numbers, and text extraction."""
    res = extract_pdf_pages(sample_pdf_bytes, filename="test_doc.pdf")
    assert res.success is True
    assert res.total_pages == 2
    assert res.has_text is True
    assert res.is_scanned is False
    assert len(res.pages) == 2

    p1 = res.pages[0]
    assert p1.page_number == 1
    assert p1.source_filename == "test_doc.pdf"
    assert "Working hours" in p1.text

    p2 = res.pages[1]
    assert p2.page_number == 2
    assert "Remote Work Policy" in p2.text


def test_pdf_loader_empty_bytes():
    """Verify empty input fails gracefully with clear error message."""
    res = extract_pdf_pages(b"", filename="empty.pdf")
    assert res.success is False
    assert "empty" in res.error_message.lower()


def test_pdf_loader_corrupted_bytes():
    """Verify corrupted non-PDF bytes return a clean error without crashing."""
    corrupted_data = b"This is definitely not a PDF file stream."
    res = extract_pdf_pages(corrupted_data, filename="corrupt.pdf")
    assert res.success is False
    assert "malformed" in res.error_message.lower() or "error" in res.error_message.lower()


def test_pdf_loader_scanned_pdf_detection(scanned_blank_pdf_bytes):
    """Verify scanned/raster image PDFs are detected and an OCR warning is provided."""
    res = extract_pdf_pages(scanned_blank_pdf_bytes, filename="scanned_receipt.pdf")
    assert res.success is True
    assert res.total_pages == 2
    assert res.total_characters == 0
    assert res.is_scanned is True
    assert res.warning_message is not None
    assert "OCR" in res.warning_message
    assert "scanned" in res.warning_message.lower()


# =====================================================================
# 2. Text Splitting & Metadata Propagation Tests
# =====================================================================

def test_text_splitter_chunking_and_metadata(sample_pdf_bytes):
    """Verify LangChain splitter preserves page numbers and creates deterministic IDs."""
    res = extract_pdf_pages(sample_pdf_bytes, filename="handbook.pdf")
    chunks = chunk_document_pages(res.pages, chunk_size=50, chunk_overlap=10)

    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, DocumentChunk)
        assert chunk.source_filename == "handbook.pdf"
        assert chunk.page_number in (1, 2)
        assert chunk.chunk_id.startswith("handbook.pdf::p")
        meta = chunk.to_metadata()
        assert meta["source"] == "handbook.pdf"
        assert meta["page"] == chunk.page_number
        assert "chunk_id" in meta


def test_text_splitter_empty_pages():
    """Verify empty pages return an empty chunk list."""
    empty_page = PDFPage(page_number=1, text="   ", source_filename="empty.pdf", total_pages=1)
    chunks = chunk_document_pages([empty_page])
    assert chunks == []


# =====================================================================
# 3. Embeddings & Vector Store Tests
# =====================================================================

def test_mock_embedder_deterministic():
    """Verify MockEmbedder generates unit-length normalized vectors deterministically."""
    embedder = MockEmbedder(dimension=64)
    v1 = embedder.embed_query("machine learning")
    v2 = embedder.embed_query("machine learning")
    v3 = embedder.embed_query("remote work policy")

    assert len(v1) == 64
    assert v1 == v2  # Deterministic
    assert v1 != v3  # Different semantics yield different vectors

    # Verify unit-length normalization
    norm = sum(x * x for x in v1)
    assert pytest.approx(norm, 0.01) == 1.0


def test_chroma_vector_store_add_and_query():
    """Verify indexing and similarity search in ChromaDB."""
    store = ChromaVectorStore(collection_name="test_collection_unit", in_memory=True)
    embedder = MockEmbedder(dimension=64)

    chunk1 = DocumentChunk(
        chunk_id="doc1::p1::c0",
        text="The annual paid vacation allowance is 24 days.",
        source_filename="doc1.pdf",
        page_number=1,
        chunk_index=0,
        total_pages=1,
    )
    chunk2 = DocumentChunk(
        chunk_id="doc1::p1::c1",
        text="Engineers can claim $800 for home office setup.",
        source_filename="doc1.pdf",
        page_number=1,
        chunk_index=1,
        total_pages=1,
    )

    added = store.add_chunks([chunk1, chunk2], embedder=embedder)
    assert added == 2
    assert store.count() == 2

    # Query
    results = store.similarity_search("vacation allowance", embedder=embedder, top_k=1)
    assert len(results) == 1
    matched_chunk, distance = results[0]
    assert matched_chunk.chunk_id == "doc1::p1::c0"
    assert "24 days" in matched_chunk.text


# =====================================================================
# 4. End-to-End RAG Pipeline & Grounding Tests
# =====================================================================

def test_rag_pipeline_grounded_answer_and_citations(sample_pdf_bytes):
    """Verify end-to-end RAG with mock response produces valid citations."""
    embedder = MockEmbedder(dimension=64)
    vector_store = ChromaVectorStore(in_memory=True)
    pipeline = RAGPipeline(
        provider="mock",
        vector_store=vector_store,
        embedder=embedder,
    )

    # Ingest document
    res = pipeline.index_pdf(sample_pdf_bytes, filename="datheon_policy.pdf")
    assert res.success is True

    # Ask supported question
    mock_llm_text = (
        "Engineers are allowed to work remotely up to 3 days per week "
        "[Source: datheon_policy.pdf, Page: 2]."
    )
    q_resp: RAGResponse = pipeline.query(
        question="How many days can engineers work remotely?",
        top_k=2,
        mock_llm_response=mock_llm_text,
    )

    assert q_resp.is_not_found is False
    assert q_resp.is_grounded is True
    assert "3 days per week" in q_resp.answer
    assert len(q_resp.citations) > 0
    assert any(c.source_filename == "datheon_policy.pdf" for c in q_resp.citations)
    assert any(c.page_number in (1, 2) for c in q_resp.citations)


def test_rag_pipeline_unsupported_question(sample_pdf_bytes):
    """Verify that an unsupported question returns standard not-found message without citations."""
    embedder = MockEmbedder(dimension=64)
    vector_store = ChromaVectorStore(in_memory=True)
    pipeline = RAGPipeline(
        provider="mock",
        vector_store=vector_store,
        embedder=embedder,
    )
    pipeline.index_pdf(sample_pdf_bytes, filename="datheon_policy.pdf")

    # Ask out-of-scope question
    q_resp = pipeline.query(
        question="What is the stock ticker symbol for Datheon?",
        top_k=2,
        mock_llm_response=NOT_FOUND_STANDARD_RESPONSE,
    )

    assert q_resp.is_not_found is True
    assert NOT_FOUND_STANDARD_RESPONSE in q_resp.answer
    assert q_resp.citations == []


def test_rag_pipeline_empty_query():
    """Verify empty user question is caught gracefully."""
    pipeline = RAGPipeline(provider="mock", embedder=MockEmbedder())
    resp = pipeline.query(question="   ")
    assert "valid question" in resp.answer.lower()


def test_rag_pipeline_empty_store():
    """Verify querying an empty vector store warns the user to upload documents first."""
    pipeline = RAGPipeline(provider="mock", embedder=MockEmbedder())
    resp = pipeline.query(question="What are the working hours?")
    assert "upload" in resp.answer.lower() or "no documents" in resp.answer.lower()

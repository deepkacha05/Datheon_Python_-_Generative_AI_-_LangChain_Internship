"""Document chunking and text splitting module using LangChain RecursiveCharacterTextSplitter.

Features:
- Splits extracted PDF pages into semantically cohesive overlapping chunks.
- Preserves complete document provenance: filename, page number, chunk index, and character counts.
- Generates unique deterministic chunk identifiers for precise ChromaDB indexing and source citations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

from task2_rag_qa.pdf_loader import PDFPage


@dataclass
class DocumentChunk:
    """Represents a discrete text chunk embedded with full source provenance."""
    chunk_id: str
    text: str
    source_filename: str
    page_number: int
    chunk_index: int
    total_pages: int
    char_count: int = 0

    def __post_init__(self):
        self.char_count = len(self.text.strip())

    def to_metadata(self) -> Dict[str, Any]:
        """Returns a ChromaDB-compatible metadata dictionary (primitive types only)."""
        return {
            "source": self.source_filename,
            "page": self.page_number,
            "chunk_index": self.chunk_index,
            "total_pages": self.total_pages,
            "char_count": self.char_count,
            "chunk_id": self.chunk_id,
        }

    @property
    def citation_label(self) -> str:
        """Formatted human-readable citation label."""
        return f"{self.source_filename} (Page {self.page_number})"


def sanitize_id_component(text: str) -> str:
    """Sanitizes filenames for deterministic ChromaDB ID keys."""
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", text)


def chunk_document_pages(
    pages: List[PDFPage],
    chunk_size: int = 600,
    chunk_overlap: int = 120,
    separators: Optional[List[str]] = None,
) -> List[DocumentChunk]:
    """Splits a list of extracted PDFPage objects into overlapping DocumentChunk instances.

    Each page is split individually so that every chunk is strictly bound to its
    actual physical PDF page number, avoiding cross-page citation drift.

    Args:
        pages: List of PDFPage instances from pdf_loader.extract_pdf_pages.
        chunk_size: Target maximum characters per chunk. Default 600.
        chunk_overlap: Target character overlap between consecutive chunks. Default 120.
        separators: Priority list of splitting boundaries for RecursiveCharacterTextSplitter.

    Returns:
        List of DocumentChunk instances with deterministic IDs and page metadata.
    """
    if not pages:
        return []

    if separators is None:
        separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False,
    )

    all_chunks: List[DocumentChunk] = []

    for page in pages:
        page_text = page.text.strip()
        if not page_text:
            continue

        text_splits = splitter.split_text(page_text)
        safe_filename = sanitize_id_component(page.source_filename)

        for c_idx, raw_chunk in enumerate(text_splits):
            chunk_clean = raw_chunk.strip()
            if not chunk_clean:
                continue

            chunk_id = f"{safe_filename}::p{page.page_number}::c{c_idx}"
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                text=chunk_clean,
                source_filename=page.source_filename,
                page_number=page.page_number,
                chunk_index=c_idx,
                total_pages=page.total_pages,
            )
            all_chunks.append(chunk)

    return all_chunks


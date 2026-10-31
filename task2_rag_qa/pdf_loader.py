"""PDF document loader and text extraction engine for Task 2.

Features:
- Extracts text page-by-page while preserving page numbers, source filename, and character counts.
- Gracefully handles empty PDFs, single-page/multi-page files, and malformed binary streams.
- Intelligently detects scanned PDFs (raster/image-only PDFs lacking a digital text layer)
  and raises clear, informative warnings explaining why OCR (Optical Character Recognition)
  is required instead of hallucinating or pretending text was parsed.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import BinaryIO, List, Optional, Union
import pypdf
from pypdf.errors import PdfReadError


@dataclass
class PDFPage:
    """Represents the extracted text and metadata of a single PDF page."""
    page_number: int  # 1-indexed for human readability and official citations
    text: str
    source_filename: str
    total_pages: int
    char_count: int = 0

    def __post_init__(self):
        self.char_count = len(self.text.strip())


@dataclass
class PDFExtractionResult:
    """Standard container for PDF ingestion and diagnostics."""
    success: bool
    filename: str
    pages: List[PDFPage] = field(default_factory=list)
    total_pages: int = 0
    total_characters: int = 0
    is_scanned: bool = False
    warning_message: Optional[str] = None
    error_message: Optional[str] = None

    @property
    def has_text(self) -> bool:
        """Returns True if at least one page contains extractable text."""
        return self.total_characters > 0


def extract_pdf_pages(
    file_input: Union[str, bytes, io.BytesIO, BinaryIO],
    filename: str = "document.pdf",
    min_char_threshold_per_page: int = 15,
) -> PDFExtractionResult:
    """Extracts text page-by-page from a PDF file, keeping rich page-level metadata.

    Args:
        file_input: A file path, raw bytes, or a file-like stream (such as Streamlit UploadedFile).
        filename: Friendly name of the source PDF.
        min_char_threshold_per_page: Minimum non-whitespace characters expected per page
                                    before evaluating whether the document is an image-only scan.

    Returns:
        PDFExtractionResult containing extracted pages, character counts, and scan detection telemetry.
    """
    stream: io.BytesIO
    if isinstance(file_input, str):
        try:
            with open(file_input, "rb") as f:
                stream = io.BytesIO(f.read())
        except Exception as exc:
            return PDFExtractionResult(
                success=False,
                filename=filename,
                error_message=f"Failed to read file from disk path '{file_input}': {str(exc)}",
            )
    elif isinstance(file_input, bytes):
        stream = io.BytesIO(file_input)
    elif hasattr(file_input, "read"):
        # Handle Streamlit UploadedFile or file-like object
        content = file_input.read()
        if isinstance(content, str):
            content = content.encode("utf-8")
        stream = io.BytesIO(content)
        # Reset file pointer if seekable
        if hasattr(file_input, "seek"):
            try:
                file_input.seek(0)
            except Exception:
                pass
    else:
        return PDFExtractionResult(
            success=False,
            filename=filename,
            error_message=f"Unsupported file_input type: {type(file_input)}",
        )

    # Check for empty byte stream
    stream.seek(0, io.SEEK_END)
    size = stream.tell()
    stream.seek(0)
    if size == 0:
        return PDFExtractionResult(
            success=False,
            filename=filename,
            error_message="The uploaded file is empty (0 bytes). Please upload a valid PDF document.",
        )

    # Parse with pypdf
    try:
        reader = pypdf.PdfReader(stream)
    except PdfReadError as exc:
        return PDFExtractionResult(
            success=False,
            filename=filename,
            error_message=f"Malformed or corrupted PDF file: {str(exc)}. Please verify the document is not damaged.",
        )
    except Exception as exc:
        return PDFExtractionResult(
            success=False,
            filename=filename,
            error_message=f"Unexpected error while opening PDF stream: {str(exc)}",
        )

    num_pages = len(reader.pages)
    if num_pages == 0:
        return PDFExtractionResult(
            success=False,
            filename=filename,
            total_pages=0,
            error_message="The PDF file contains 0 pages.",
        )

    extracted_pages: List[PDFPage] = []
    total_chars = 0
    pages_with_minimal_text = 0

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1  # 1-indexed
        try:
            raw_text = page.extract_text() or ""
        except Exception as exc:
            # If one page fails to parse, record an empty page rather than failing entire doc
            raw_text = ""

        # Normalize whitespace while preserving line structure
        cleaned_text = raw_text.strip()
        char_len = len(cleaned_text)
        total_chars += char_len

        if char_len < min_char_threshold_per_page:
            pages_with_minimal_text += 1

        extracted_pages.append(
            PDFPage(
                page_number=page_num,
                text=cleaned_text,
                source_filename=filename,
                total_pages=num_pages,
            )
        )

    # Detect Scanned Documents
    # If the document has pages, but either zero text or an average of < 15 chars per page,
    # it is almost certainly a raster/scanned image without OCR text layers.
    is_scanned = False
    warning = None

    if total_chars == 0:
        is_scanned = True
        warning = (
            f"⚠️ **Scanned or Image-Only PDF Detected:** Document '{filename}' has {num_pages} page(s) "
            "but contains zero digital text. This typically happens when documents are scanned or exported "
            "as flat raster images. Standard PDF text extraction cannot read rasterized pixels; "
            "an Optical Character Recognition (OCR) engine (such as Tesseract, AWS Textract, or Google Document AI) "
            "is required to extract text from scanned images."
        )
    elif pages_with_minimal_text == num_pages and total_chars < (num_pages * min_char_threshold_per_page):
        is_scanned = True
        warning = (
            f"⚠️ **Potential Low-Text / Scanned PDF:** Document '{filename}' yielded very few readable characters "
            f"({total_chars} chars across {num_pages} pages). If this document contains scanned pages or graphical text, "
            "OCR processing is recommended for complete extraction."
        )

    return PDFExtractionResult(
        success=True,
        filename=filename,
        pages=extracted_pages,
        total_pages=num_pages,
        total_characters=total_chars,
        is_scanned=is_scanned,
        warning_message=warning,
    )


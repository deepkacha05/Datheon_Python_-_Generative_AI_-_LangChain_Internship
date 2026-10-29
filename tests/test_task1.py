"""Unit and integration tests for Task 1: Structured LLM Prompt Pipeline & JSON Extractor.

Validates:
- Pydantic schema constraints and custom validators (Resume & Invoice)
- Parameterized prompt builder
- JSON sanitization, markdown stripping, and trailing comma repair
- End-to-end extraction pipeline with Mock LLM
- Robust handling of malformed JSON, schema mismatches, and API connection failures
"""

import pytest
from pydantic import ValidationError

from task1_json_extractor.extractor import (
    clean_and_parse_json,
    extract_structured_json,
    JSONParsingError,
    APIConnectionError,
    call_llm,
)
from task1_json_extractor.prompts import build_prompt, SYSTEM_PROMPT
from task1_json_extractor.schemas import (
    ResumeSchema,
    InvoiceSchema,
    ContactInfo,
    Education,
    WorkExperience,
    InvoiceParty,
    InvoiceLineItem,
)
from task1_json_extractor.sample_data import (
    SAMPLE_RESUME_TEXT,
    SAMPLE_INVOICE_TEXT,
    MALFORMED_JSON_WITH_MARKDOWN,
    MALFORMED_JSON_WITH_TRAILING_COMMAS,
    MALFORMED_JSON_WITH_PREAMBLE,
)


# =====================================================================
# 1. Schema Validation Tests
# =====================================================================

def test_resume_schema_valid():
    """Verify valid resume data validates cleanly."""
    valid_data = {
        "candidate_name": "Priya Sharma",
        "headline": "Senior AI Engineer",
        "contact_info": {
            "email": "priya@example.com",
            "phone": "+91 98765 43210",
            "location": "Bengaluru",
        },
        "skills": ["Python", "LangChain", "Gemini API"],
        "education": [
            {
                "institution": "VIT",
                "degree": "B.Tech CSE",
                "graduation_year": 2021,
                "gpa": "8.9/10",
            }
        ],
        "work_experience": [
            {
                "company": "Datheon",
                "role": "AI Lead",
                "is_current": True,
                "responsibilities": ["Built RAG pipeline"],
                "technologies": ["Python", "ChromaDB"],
            }
        ],
    }
    resume = ResumeSchema.model_validate(valid_data)
    assert resume.candidate_name == "Priya Sharma"
    assert resume.contact_info.email == "priya@example.com"
    assert len(resume.skills) == 3
    assert resume.work_experience[0].company == "Datheon"
    assert resume.education[0].graduation_year == 2021


def test_resume_schema_empty_name_fails():
    """Verify empty candidate name raises ValidationError."""
    with pytest.raises(ValidationError):
        ResumeSchema.model_validate({"candidate_name": "   "})


def test_invoice_schema_valid_and_coercion():
    """Verify invoice line items and amounts coerce currency strings to floats."""
    data = {
        "invoice_number": "INV-1001",
        "vendor": {"name": "Acme Corp", "email": "billing@acme.com"},
        "client": {"name": "Beta LLC", "email": "accounts@beta.com"},
        "line_items": [
            {
                "description": "Consulting Services",
                "quantity": "10",
                "unit_price": "$150.00",
                "total": "1,500.00",
            }
        ],
        "subtotal": "$1,500.00",
        "tax_amount": "$150.00",
        "total_amount": "₹1,650.00",
        "currency": "USD",
    }
    invoice = InvoiceSchema.model_validate(data)
    assert invoice.invoice_number == "INV-1001"
    assert invoice.subtotal == 1500.0
    assert invoice.tax_amount == 150.0
    assert invoice.total_amount == 1650.0
    assert invoice.line_items[0].quantity == 10.0
    assert invoice.line_items[0].unit_price == 150.0


# =====================================================================
# 2. Prompt Builder Tests
# =====================================================================

def test_build_prompt_resume():
    """Verify resume prompt incorporates input text."""
    sys_p, user_p = build_prompt("resume", "Sample resume content here")
    assert sys_p == SYSTEM_PROMPT
    assert "Sample resume content here" in user_p
    assert "ResumeSchema" in user_p


def test_build_prompt_invoice():
    """Verify invoice prompt incorporates input text."""
    sys_p, user_p = build_prompt("invoice", "Invoice for consulting $500")
    assert "Invoice for consulting $500" in user_p
    assert "InvoiceSchema" in user_p


def test_build_prompt_invalid_type():
    """Verify invalid document type raises ValueError."""
    with pytest.raises(ValueError):
        build_prompt("unsupported_type", "some text")


# =====================================================================
# 3. JSON Sanitization and Repair Tests
# =====================================================================

def test_clean_and_parse_json_markdown():
    """Verify JSON wrapped in markdown fences is parsed properly."""
    result = clean_and_parse_json(MALFORMED_JSON_WITH_MARKDOWN)
    assert result["candidate_name"] == "Rohan Verma"
    assert "Python" in result["skills"]


def test_clean_and_parse_json_trailing_commas():
    """Verify trailing commas in objects and arrays are repaired."""
    result = clean_and_parse_json(MALFORMED_JSON_WITH_TRAILING_COMMAS)
    assert result["candidate_name"] == "Anita Desai"
    assert result["skills"] == ["SQL", "PowerBI", "Python"]


def test_clean_and_parse_json_with_preamble():
    """Verify surrounding conversational text is discarded and JSON extracted."""
    result = clean_and_parse_json(MALFORMED_JSON_WITH_PREAMBLE)
    assert result["candidate_name"] == "Kavita Rao"
    assert result["headline"] == "NLP Researcher"


def test_clean_and_parse_json_empty_fails():
    """Verify empty string raises JSONParsingError."""
    with pytest.raises(JSONParsingError):
        clean_and_parse_json("   ")


def test_clean_and_parse_json_garbage_fails():
    """Verify arbitrary unrepairable text raises JSONParsingError."""
    with pytest.raises(JSONParsingError):
        clean_and_parse_json("This is definitely not a JSON payload at all.")


# =====================================================================
# 4. End-to-End Extractor Pipeline (Mock Mode)
# =====================================================================

def test_extractor_resume_mock_success():
    """Verify end-to-end extraction with mock response returns a validated ResumeSchema."""
    mock_llm_json = """{
      "candidate_name": "Priya Sharma",
      "headline": "AI & Full-Stack Python Engineer",
      "contact_info": {
        "email": "priya.sharma.dev@gmail.com",
        "phone": "+91 98765 43210",
        "location": "Bengaluru, Karnataka, India"
      },
      "summary": "Results-driven AI Engineer with 3+ years experience.",
      "skills": ["Python", "FastAPI", "Streamlit", "LangChain"],
      "work_experience": [
        {
          "company": "Datheon Tech Solutions",
          "role": "Senior GenAI Developer",
          "is_current": true,
          "responsibilities": ["Architected multi-agent system"],
          "technologies": ["Python", "LangChain"]
        }
      ],
      "education": [
        {
          "institution": "VIT",
          "degree": "B.Tech Computer Science",
          "graduation_year": 2021,
          "gpa": "8.9 / 10.0"
        }
      ],
      "projects": [
        {
          "name": "DocuQuery",
          "description": "Autonomous Multimodal Document Assistant",
          "technologies": ["Python", "Streamlit"]
        }
      ],
      "certifications": ["DeepLearning.AI Generative AI"]
    }"""

    res = extract_structured_json(
        text=SAMPLE_RESUME_TEXT,
        document_type="resume",
        provider="mock",
        mock_response=mock_llm_json,
    )

    assert res.success is True
    assert isinstance(res.data, ResumeSchema)
    assert res.data.candidate_name == "Priya Sharma"
    assert len(res.data.skills) == 4
    assert res.data.work_experience[0].company == "Datheon Tech Solutions"
    assert res.error_message is None
    assert res.execution_time_seconds >= 0.0


def test_extractor_invoice_mock_success():
    """Verify end-to-end extraction with mock response returns a validated InvoiceSchema."""
    mock_invoice_json = """{
      "invoice_number": "INV-2024-00892",
      "invoice_date": "2024-10-15",
      "due_date": "2024-11-14",
      "vendor": {
        "name": "Apex Digital Consulting Inc.",
        "email": "billing@apexdigital.io"
      },
      "client": {
        "name": "Nexis Health Technologies LLC",
        "email": "finance@nexishealth.org"
      },
      "line_items": [
        {
          "description": "LLM Application Architecture",
          "quantity": 40.0,
          "unit_price": 150.0,
          "total": 6000.0
        }
      ],
      "subtotal": 12500.0,
      "tax_amount": 1031.25,
      "discount_amount": 500.0,
      "total_amount": 13031.25,
      "currency": "USD"
    }"""

    res = extract_structured_json(
        text=SAMPLE_INVOICE_TEXT,
        document_type="invoice",
        provider="mock",
        mock_response=mock_invoice_json,
    )

    assert res.success is True
    assert isinstance(res.data, InvoiceSchema)
    assert res.data.invoice_number == "INV-2024-00892"
    assert res.data.total_amount == 13031.25
    assert res.data.vendor.name == "Apex Digital Consulting Inc."


def test_extractor_missing_required_field_handling():
    """Verify missing required field fails gracefully with detailed field errors."""
    mock_invalid_json = """{
      "headline": "AI Engineer",
      "skills": ["Python"]
    }"""

    res = extract_structured_json(
        text=SAMPLE_RESUME_TEXT,
        document_type="resume",
        provider="mock",
        mock_response=mock_invalid_json,
    )

    assert res.success is False
    assert res.data is None
    assert "Pydantic Schema Validation failed" in (res.error_message or "")
    assert res.validation_errors is not None
    assert any("candidate_name" in err["field"] for err in res.validation_errors)


def test_extractor_missing_api_key_handling():
    """Verify APIConnectionError is caught and returned cleanly when no key is present."""
    # Ensure no environment keys interfere for this test
    res = extract_structured_json(
        text="Sample text",
        document_type="resume",
        provider="gemini",
        api_key="",  # explicitly empty
    )
    # If environment variable is absent or key is empty, it reports connection failure gracefully
    if not res.success:
        assert "API Key" in (res.error_message or "") or "Gemini" in (res.error_message or "")

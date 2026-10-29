"""Reusable, parameterized prompt templates for Task 1: Structured LLM Prompt Pipeline & JSON Extractor.

Provides modular prompt generation for various document types (Resumes, Invoices, Custom schemas)
with explicit JSON formatting contracts and constraint instructions.
"""

from typing import Optional, Dict, Any
import json


# =====================================================================
# System Prompt
# =====================================================================

SYSTEM_PROMPT = """You are an expert Information Extraction and Data Structuring AI.
Your sole mission is to analyze unstructured or semi-structured raw text and convert it into a strictly valid, comprehensive JSON object conforming exactly to the requested schema.

CRITICAL RULES:
1. OUTPUT FORMAT: Respond ONLY with a valid, parseable JSON object. Do NOT include explanatory remarks, conversational greetings, reasoning steps, or markdown formatting other than an optional ```json ``` wrapper.
2. SCHEMA COMPLIANCE: Adhere strictly to the requested schema fields, types, and hierarchies. Do not invent unauthorized root-level keys.
3. GROUNDEDNESS: Extract only what is present or directly verifiable from the source text. NEVER invent credentials, phone numbers, or dates.
4. MISSING VALUES: For missing optional text fields, set them to null. For missing list fields, provide an empty list [].
5. NUMERICS: For currency or monetary figures, extract only pure numeric floats (e.g., 1250.50, not "$1,250.50").
6. ESCAPING: Ensure all double quotes, backslashes, and newlines in string values are properly escaped for standard JSON.
"""


# =====================================================================
# Resume Extraction Template
# =====================================================================

RESUME_USER_PROMPT_TEMPLATE = """Please analyze the following raw resume/CV text and extract structured information matching the ResumeSchema JSON structure.

TARGET JSON SCHEMA REQUIREMENTS:
{{
  "candidate_name": "Full legal name of the candidate (string, required)",
  "headline": "Current title or professional headline (string or null)",
  "contact_info": {{
    "name": "Candidate name if repeated in contact block (string or null)",
    "email": "Email address (string or null)",
    "phone": "Phone number with country code (string or null)",
    "linkedin_url": "Full LinkedIn URL (string or null)",
    "github_url": "Full GitHub URL (string or null)",
    "location": "City, State/Country (string or null)"
  }},
  "summary": "Professional summary or objective (string or null)",
  "skills": ["List", "of", "technical", "and", "domain", "skills"],
  "work_experience": [
    {{
      "company": "Company / Organization name",
      "role": "Job title / Designation",
      "location": "Job location or Remote (string or null)",
      "start_date": "Start date, e.g. Jan 2021 (string or null)",
      "end_date": "End date, e.g. Present or Dec 2023 (string or null)",
      "is_current": true / false,
      "responsibilities": ["Detailed accomplishment 1", "Detailed accomplishment 2"],
      "technologies": ["Tool1", "Tool2"]
    }}
  ],
  "education": [
    {{
      "institution": "University / College name",
      "degree": "Degree name, e.g. B.Tech Computer Science",
      "field_of_study": "Specialization or Major (string or null)",
      "graduation_year": 2023 (integer or null),
      "gpa": "Score or GPA (string or null)"
    }}
  ],
  "projects": [
    {{
      "name": "Project title",
      "description": "Brief description of project",
      "technologies": ["Tech1", "Tech2"],
      "link": "Project URL or GitHub repository (string or null)"
    }}
  ],
  "certifications": ["List", "of", "certifications", "or", "awards"]
}}

EXTRACTION CONSTRAINTS:
- Do NOT guess candidate details. If something is missing, output null or [].
- Preserve all quantifiable metrics in work experience and project descriptions.
- Ensure output is valid JSON.

RAW RESUME TEXT:
----------------------------------------
{text}
----------------------------------------

JSON Output:"""


# =====================================================================
# Invoice Extraction Template
# =====================================================================

INVOICE_USER_PROMPT_TEMPLATE = """Please analyze the following raw invoice / billing document text and extract structured information matching the InvoiceSchema JSON structure.

TARGET JSON SCHEMA REQUIREMENTS:
{{
  "invoice_number": "Invoice ID or code (string, required)",
  "invoice_date": "Date invoice was issued, e.g. 2024-03-15 (string or null)",
  "due_date": "Payment due date (string or null)",
  "vendor": {{
    "name": "Seller / Provider company or individual name",
    "address": "Vendor address (string or null)",
    "email": "Vendor email (string or null)",
    "phone": "Vendor phone (string or null)",
    "tax_id": "Vendor GSTIN / Tax ID / VAT (string or null)"
  }},
  "client": {{
    "name": "Client / Customer / Bill To entity name",
    "address": "Client billing/shipping address (string or null)",
    "email": "Client email (string or null)",
    "phone": "Client phone (string or null)",
    "tax_id": "Client Tax ID / VAT (string or null)"
  }},
  "line_items": [
    {{
      "description": "Product or service description",
      "quantity": 1.0 (float),
      "unit_price": 100.0 (float),
      "total": 100.0 (float)
    }}
  ],
  "subtotal": 100.0 (float),
  "tax_rate_percent": 18.0 (float or null),
  "tax_amount": 18.0 (float),
  "discount_amount": 0.0 (float),
  "total_amount": 118.0 (float, required),
  "currency": "USD" or "INR" or "EUR" (string, default "USD"),
  "payment_terms": "e.g. Net 30 or Due on Receipt (string or null)",
  "notes": "Payment instructions, bank info, or footer notes (string or null)"
}}

EXTRACTION CONSTRAINTS:
- Strip all currency symbols ($ ₹ €) and commas from numeric fields so they become valid floats.
- If quantity is omitted on a line item, assume 1.0.
- Ensure subtotal, tax_amount, discount_amount, and total_amount are mathematically consistent if available in text.
- Ensure output is valid JSON.

RAW INVOICE TEXT:
----------------------------------------
{text}
----------------------------------------

JSON Output:"""


# =====================================================================
# Generic Custom Schema Prompt Template
# =====================================================================

GENERIC_USER_PROMPT_TEMPLATE = """Please extract structured data from the following raw text based on the specified target schema.

TARGET SCHEMA DEFINITION:
{schema_description}

RAW TEXT:
----------------------------------------
{text}
----------------------------------------

Remember: Return ONLY valid, parseable JSON conforming to the schema. No explanations.
JSON Output:"""


def build_prompt(document_type: str, text: str, custom_schema: Optional[Dict[str, Any]] = None) -> tuple[str, str]:
    """Builds the (system_prompt, user_prompt) tuple based on document type and input text.

    Args:
        document_type: "resume", "invoice", or "custom".
        text: Unstructured raw input text.
        custom_schema: Optional dictionary or schema representation for "custom" type.

    Returns:
        tuple[str, str]: (SYSTEM_PROMPT, user_prompt)
    """
    doc_type_clean = document_type.lower().strip()
    if doc_type_clean == "resume":
        user_prompt = RESUME_USER_PROMPT_TEMPLATE.format(text=text.strip())
    elif doc_type_clean == "invoice":
        user_prompt = INVOICE_USER_PROMPT_TEMPLATE.format(text=text.strip())
    elif doc_type_clean == "custom":
        schema_desc = json.dumps(custom_schema or {}, indent=2) if custom_schema else "Dynamic key-value object"
        user_prompt = GENERIC_USER_PROMPT_TEMPLATE.format(schema_description=schema_desc, text=text.strip())
    else:
        raise ValueError(f"Unsupported document_type: '{document_type}'. Expected 'resume', 'invoice', or 'custom'.")

    return SYSTEM_PROMPT, user_prompt

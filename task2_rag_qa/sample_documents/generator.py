"""Generates realistic sample PDF documents for Task 2 demonstration and automated testing.

Produces a valid, multi-page PDF ('Datheon_Company_Policy_and_AI_Guidelines.pdf')
containing realistic company policies, working hours, travel allowances, and AI governance standards.
"""

from __future__ import annotations

import io
import os
from typing import List


def build_pure_python_pdf(pages_text: List[str]) -> bytes:
    """Creates a standard PDF 1.4 byte buffer from a list of page text strings."""
    buf = io.BytesIO()
    buf.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    num_pages = len(pages_text)
    kids_refs = [f"{4 + i * 2} 0 R" for i in range(num_pages)]
    kids_str = " ".join(kids_refs)

    objects = {
        1: f"<< /Type /Catalog /Pages 2 0 R >>".encode("latin-1"),
        2: f"<< /Type /Pages /Kids [{kids_str}] /Count {num_pages} >>".encode("latin-1"),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }

    for i, p_text in enumerate(pages_text):
        p_obj_num = 4 + i * 2
        c_obj_num = p_obj_num + 1

        lines = p_text.split("\n")
        stream_cmds = ["BT", "/F1 12 Tf", "50 740 Td", "16 TL"]
        for line in lines:
            safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream_cmds.append(f"({safe}) Tj")
            stream_cmds.append("T*")
        stream_cmds.append("ET")
        stream_data = "\n".join(stream_cmds).encode("latin-1")
        stream_obj = (
            f"<< /Length {len(stream_data)} >>\nstream\n".encode("latin-1")
            + stream_data
            + b"\nendstream"
        )
        page_obj = (
            f"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 3 0 R >> >> "
            f"/MediaBox [0 0 612 792] /Contents {c_obj_num} 0 R >>"
        ).encode("latin-1")

        objects[p_obj_num] = page_obj
        objects[c_obj_num] = stream_obj

    obj_offsets = {}
    for obj_num in sorted(objects.keys()):
        obj_offsets[obj_num] = buf.tell()
        buf.write(f"{obj_num} 0 obj\n".encode("latin-1"))
        buf.write(objects[obj_num])
        buf.write(b"\nendobj\n")

    startxref = buf.tell()
    buf.write(b"xref\n")
    total_objs = len(objects) + 1
    buf.write(f"0 {total_objs}\n".encode("latin-1"))
    buf.write(b"0000000000 65535 f \n")
    for obj_num in range(1, total_objs):
        off = obj_offsets[obj_num]
        buf.write(f"{off:010d} 00000 n \n".encode("latin-1"))

    buf.write(
        f"trailer\n<< /Size {total_objs} /Root 1 0 R >>\nstartxref\n{startxref}\n%%EOF\n".encode(
            "latin-1"
        )
    )
    return buf.getvalue()


SAMPLE_PAGE_1 = """DATHEON TECH SOLUTIONS - EMPLOYEE HANDBOOK & COMPANY POLICIES
Document Ref: POL-2025-V2.1 | Effective Date: January 1, 2025

SECTION 1: ORGANIZATIONAL MISSION & WORKING HOURS
Datheon Tech Solutions is a premier artificial intelligence research and consulting firm
specializing in enterprise LangChain agents, high-throughput RAG systems, and structured LLM extraction.

1.1 Standard Working Hours:
- Official office hours are 9:30 AM to 6:30 PM Indian Standard Time (IST), Monday through Friday.
- Core synchronous team collaboration hours are 11:00 AM to 4:00 PM IST.
- All team members are expected to be reachable via Slack during core collaboration hours.

1.2 Annual Paid Time Off (PTO):
- Full-time engineers receive 24 paid vacation days annually, accrued monthly at 2 days per month.
- Employees receive 10 paid casual/medical sick leaves per calendar year.
- National and regional holidays are published in December for the subsequent calendar year."""

SAMPLE_PAGE_2 = """SECTION 2: HYBRID WORK & EXPENSE REIMBURSEMENT POLICY

2.1 Hybrid Remote Work Model:
- Datheon operates on a flexible hybrid model allowing engineers to work remotely up to 3 days per week.
- Mandatory in-office collaboration days are Tuesday and Thursday for sprint planning and demos.
- Fully remote work arrangements require written approval from the Engineering Director.

2.2 Ergonomic Home Office Equipment Stipend:
- New full-time hires are eligible for a one-time home office tech stipend of $800 (or INR 65,000).
- Eligible expenses include ergonomic monitors, standing desk converters, noise-canceling headsets, and chairs.
- Receipts must be submitted through the Datheon HR Portal within 60 days of employment.

2.3 Business Travel & Meal Per Diem:
- Approved domestic travel flights must be booked in economy class.
- The maximum allowable daily meal per diem is $75 (or INR 4,500) per day.
- Itemized tax invoices must be submitted with expense claims for audit compliance."""

SAMPLE_PAGE_3 = """SECTION 3: GENERATIVE AI SECURITY & INTELLECTUAL PROPERTY STANDARDS

3.1 Safe Handling of Client Proprietary Data:
- Engineers must never paste unredacted client intellectual property, PII, or credentials into public LLM portals.
- Enterprise zero-data-retention APIs (Google Cloud Vertex AI, Azure OpenAI, enterprise Gemini) are approved.
- All documents indexed in vector databases (such as ChromaDB) must be purged upon project decommissioning.

3.2 API Key Governance:
- API credentials (including GEMINI_API_KEY, OPENAI_API_KEY, and TAVILY_API_KEY) must strictly reside in local .env files.
- Hardcoding secrets or checking .env files into GitHub repositories is a critical security violation.

3.3 Verification & Hallucination Mitigation:
- Any production LLM workflow must implement context grounding and output validation (e.g. Pydantic v2 schemas).
- When a document Q&A assistant cannot find supporting facts in uploaded documents, it must explicitly state
  that the information was not found rather than hallucinating plausible answers."""


def generate_sample_policy_pdf(output_path: str = "task2_rag_qa/sample_documents/Datheon_Company_Policy_and_AI_Guidelines.pdf") -> str:
    """Generates the sample policy PDF file on disk and returns its path."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    pdf_bytes = build_pure_python_pdf([SAMPLE_PAGE_1, SAMPLE_PAGE_2, SAMPLE_PAGE_3])
    with open(output_path, "wb") as f:
        f.write(pdf_bytes)
    return output_path


if __name__ == "__main__":
    out = generate_sample_policy_pdf()
    print(f"Sample PDF successfully generated at: {out}")


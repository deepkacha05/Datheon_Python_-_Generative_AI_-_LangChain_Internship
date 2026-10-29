"""Interactive Streamlit Application and CLI runner for Task 1:
Structured LLM Prompt Pipeline & JSON Extractor.

Features:
- Dual LLM Provider Support: Google Gemini (google-genai) and OpenAI.
- Offline Mock Mode (instant zero-API-key evaluation for testing & grading).
- 1-Click Sample Loaders for realistic Resumes, Invoices, and Edge-Case Malformed JSON payloads.
- Visual structured presentation (cards, badges, data tables) + copyable raw JSON tree.
- Comprehensive field-level Pydantic validation telemetry and error diagnostics.
"""

from __future__ import annotations

import json
import os
import sys
import streamlit as st
from dotenv import load_dotenv

# Ensure repo root is on sys.path for direct script execution
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from task1_json_extractor.extractor import extract_structured_json, clean_and_parse_json, ExtractionResult
from task1_json_extractor.schemas import ResumeSchema, InvoiceSchema
from task1_json_extractor.sample_data import (
    SAMPLE_RESUME_TEXT,
    SAMPLE_INVOICE_TEXT,
    MALFORMED_JSON_WITH_TRAILING_COMMAS,
)

load_dotenv()


# =====================================================================
# UI Presentation Helpers
# =====================================================================

def render_resume_card(resume: ResumeSchema):
    """Renders a structured, visually appealing presentation of a parsed resume."""
    st.subheader(f"👤 {resume.candidate_name}")
    if resume.headline:
        st.caption(f"**{resume.headline}**")

    # Contact & Quick Info
    ci = resume.contact_info
    c_cols = st.columns(4)
    with c_cols[0]:
        st.markdown(f"**Email:** {ci.email or 'N/A'}")
    with c_cols[1]:
        st.markdown(f"**Phone:** {ci.phone or 'N/A'}")
    with c_cols[2]:
        st.markdown(f"**Location:** {ci.location or 'N/A'}")
    with c_cols[3]:
        links = []
        if ci.linkedin_url:
            links.append(f"[LinkedIn]({ci.linkedin_url})")
        if ci.github_url:
            links.append(f"[GitHub]({ci.github_url})")
        st.markdown(f"**Profiles:** {' | '.join(links) if links else 'N/A'}")

    st.divider()

    # Summary
    if resume.summary:
        st.markdown(f"**Executive Summary:**\n> {resume.summary}")

    # Skills Badges
    if resume.skills:
        st.markdown("**Core Skills & Tech Stack:**")
        skill_badges = " ".join([f"`{s}`" for s in resume.skills])
        st.markdown(skill_badges)

    st.divider()

    # Work Experience
    if resume.work_experience:
        st.markdown("### 💼 Professional Experience")
        for exp in resume.work_experience:
            dates = f"{exp.start_date or ''} - {exp.end_date or ('Present' if exp.is_current else '')}"
            st.markdown(f"**{exp.role}** at **{exp.company}** `({dates.strip()})`")
            if exp.location:
                st.caption(f"📍 {exp.location}")
            for bullet in exp.responsibilities:
                st.markdown(f"- {bullet}")
            if exp.technologies:
                st.caption(f"*Technologies:* {', '.join(exp.technologies)}")
            st.markdown("")

    # Education & Projects in columns
    col_edu, col_proj = st.columns(2)
    with col_edu:
        if resume.education:
            st.markdown("### 🎓 Education")
            for edu in resume.education:
                st.markdown(f"**{edu.degree}**")
                st.markdown(f"{edu.institution} (Grad: {edu.graduation_year or 'N/A'})")
                if edu.gpa:
                    st.caption(f"GPA / Score: {edu.gpa}")
                st.markdown("")

    with col_proj:
        if resume.projects:
            st.markdown("### 🚀 Featured Projects")
            for proj in resume.projects:
                st.markdown(f"**{proj.name}**")
                if proj.description:
                    st.markdown(proj.description)
                if proj.technologies:
                    st.caption(f"*Stack:* {', '.join(proj.technologies)}")
                if proj.link:
                    st.markdown(f"[Project Link]({proj.link})")
                st.markdown("")

    if resume.certifications:
        st.markdown("### 📜 Certifications")
        for cert in resume.certifications:
            st.markdown(f"- {cert}")


def render_invoice_card(invoice: InvoiceSchema):
    """Renders a structured, visually appealing presentation of a parsed invoice."""
    st.subheader(f"🧾 Tax Invoice: `{invoice.invoice_number}`")

    # Dates and terms
    d_cols = st.columns(4)
    with d_cols[0]:
        st.metric("Invoice Date", invoice.invoice_date or "N/A")
    with d_cols[1]:
        st.metric("Due Date", invoice.due_date or "N/A")
    with d_cols[2]:
        st.metric("Payment Terms", invoice.payment_terms or "Net 30")
    with d_cols[3]:
        st.metric("Total Payable", f"{invoice.currency} {invoice.total_amount:,.2f}")

    st.divider()

    # Vendor & Client
    v_col, c_col = st.columns(2)
    with v_col:
        st.markdown("#### 🏢 Vendor / Service Provider")
        st.markdown(f"**{invoice.vendor.name}**")
        if invoice.vendor.address:
            st.caption(invoice.vendor.address)
        if invoice.vendor.email:
            st.markdown(f"📧 `{invoice.vendor.email}`")
        if invoice.vendor.tax_id:
            st.markdown(f"Tax ID / EIN: `{invoice.vendor.tax_id}`")

    with c_col:
        st.markdown("#### 👤 Client / Billed To")
        st.markdown(f"**{invoice.client.name}**")
        if invoice.client.address:
            st.caption(invoice.client.address)
        if invoice.client.email:
            st.markdown(f"📧 `{invoice.client.email}`")
        if invoice.client.tax_id:
            st.markdown(f"Client Tax ID: `{invoice.client.tax_id}`")

    st.divider()

    # Line Items Table
    if invoice.line_items:
        st.markdown("#### 📦 Billable Line Items")
        table_data = []
        for item in invoice.line_items:
            table_data.append({
                "Description": item.description,
                "Quantity": item.quantity,
                "Unit Price": f"{invoice.currency} {item.unit_price:,.2f}",
                "Line Total": f"{invoice.currency} {item.total:,.2f}",
            })
        st.table(table_data)

    # Financial Summary
    f_cols = st.columns(4)
    with f_cols[0]:
        st.metric("Subtotal", f"{invoice.currency} {invoice.subtotal:,.2f}")
    with f_cols[1]:
        st.metric("Tax Amount", f"{invoice.currency} {invoice.tax_amount:,.2f}")
    with f_cols[2]:
        st.metric("Discount", f"{invoice.currency} {invoice.discount_amount:,.2f}")
    with f_cols[3]:
        st.metric("Grand Total", f"{invoice.currency} {invoice.total_amount:,.2f}")

    if invoice.notes:
        st.info(f"**Notes / Payment Instructions:**\n{invoice.notes}")


# =====================================================================
# Main Streamlit Application
# =====================================================================

def main():
    st.set_page_config(
        page_title="Task 1: Structured LLM Prompt Pipeline & JSON Extractor",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Header
    st.title("⚡ Task 1: Structured LLM Prompt Pipeline & JSON Extractor")
    st.markdown(
        "Extract clean, strictly validated JSON from noisy unstructured text (Resumes, Invoices) "
        "using parameterized prompt templates, robust multi-stage repair, and **Pydantic v2** validation schemas."
    )

    # Sidebar configuration
    st.sidebar.header("🛠️ Extraction Configuration")

    # Session state for document type selection
    if "doc_type_index" not in st.session_state:
        st.session_state.doc_type_index = 0

    doc_type = st.sidebar.selectbox(
        "Document Type / Target Schema",
        options=["Resume Extractor", "Invoice Extractor", "Custom Schema"],
        index=st.session_state.doc_type_index,
    )

    provider_choice = st.sidebar.selectbox(
        "LLM Provider",
        options=["Google Gemini", "OpenAI", "Offline Mock Mode (No Key Needed)"],
        index=0,
    )

    # Map provider
    if provider_choice == "Google Gemini":
        provider_code = "gemini"
        gemini_model_options = [
            "gemini-3.5-flash",
            "gemini-3.1-flash-lite",
            "gemini-flash-latest",
            "gemini-2.5-flash-lite",
            "Custom Model",
        ]
        selected_model = st.sidebar.selectbox("Gemini Model", options=gemini_model_options, index=0)
        if selected_model == "Custom Model":
            model_name = st.sidebar.text_input("Enter Custom Model Name", value="gemini-3.5-flash")
        else:
            model_name = selected_model

        api_key_env = os.getenv("GEMINI_API_KEY", "")
        api_key_input = st.sidebar.text_input(
            "Gemini API Key",
            value=api_key_env,
            type="password",
            help="Loaded from .env if available, or enter your Google AI Studio key.",
        )
    elif provider_choice == "OpenAI":
        provider_code = "openai"
        default_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        api_key_env = os.getenv("OPENAI_API_KEY", "")
        model_name = st.sidebar.text_input("OpenAI Model Name", value=default_model)
        api_key_input = st.sidebar.text_input(
            "OpenAI API Key",
            value=api_key_env,
            type="password",
            help="Loaded from .env if available, or enter your platform.openai.com key.",
        )
    else:
        provider_code = "mock"
        model_name = "mock-evaluator"
        api_key_input = ""
        st.sidebar.info("💡 **Mock Mode Active:** Evaluates pipeline instantaneously using simulated responses without external API calls.")

    temperature = st.sidebar.slider(
        "Sampling Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="0.0 provides maximum determinism for structured JSON extraction.",
    )

    st.sidebar.divider()
    st.sidebar.markdown("### 🚀 Quick Sample Presets")

    # Session state for input text
    if "input_text" not in st.session_state:
        st.session_state.input_text = SAMPLE_RESUME_TEXT

    col_btn1, col_btn2 = st.sidebar.columns(2)
    with col_btn1:
        if st.button("📄 Sample Resume", use_container_width=True):
            st.session_state.input_text = SAMPLE_RESUME_TEXT
            st.session_state.doc_type_index = 0
            st.rerun()
    with col_btn2:
        if st.button("🧾 Sample Invoice", use_container_width=True):
            st.session_state.input_text = SAMPLE_INVOICE_TEXT
            st.session_state.doc_type_index = 1
            st.rerun()

    if st.sidebar.button("🧹 Clear Input", use_container_width=True):
        st.session_state.input_text = ""
        st.rerun()

    # Main Input Area
    st.markdown("### 📝 Input Unstructured Text")
    raw_input = st.text_area(
        label="Paste unstructured text or load sample data from the sidebar:",
        value=st.session_state.input_text,
        height=240,
        placeholder="Paste resume, invoice, email text, or billing statement here...",
    )

    doc_type_key = "resume" if "Resume" in doc_type else ("invoice" if "Invoice" in doc_type else "custom")

    # Trigger button
    btn_extract = st.button("⚡ Extract & Validate JSON", type="primary", use_container_width=True)

    if btn_extract:
        if not raw_input.strip():
            st.warning("⚠️ Please provide some input text to extract from.")
            return

        with st.spinner("Invoking LLM prompt pipeline, repairing syntax, and validating Pydantic schemas..."):
            # If in mock mode, provide appropriate mock payload
            mock_payload = None
            if provider_code == "mock":
                if doc_type_key == "resume":
                    mock_payload = json.dumps({
                        "candidate_name": "Priya Sharma",
                        "headline": "AI & Full-Stack Python Engineer",
                        "contact_info": {
                            "email": "priya.sharma.dev@gmail.com",
                            "phone": "+91 98765 43210",
                            "location": "Bengaluru, Karnataka, India",
                            "linkedin_url": "https://linkedin.com/in/priyasharma-ai",
                            "github_url": "https://github.com/priyasharma-dev",
                        },
                        "summary": "Results-driven AI Engineer with 3+ years experience architecting LLM applications and RAG systems.",
                        "skills": ["Python", "FastAPI", "Streamlit", "LangChain", "ChromaDB", "Google Gemini API"],
                        "work_experience": [
                            {
                                "company": "Datheon Tech Solutions",
                                "role": "Senior GenAI Developer",
                                "location": "Bengaluru",
                                "start_date": "July 2022",
                                "end_date": "Present",
                                "is_current": True,
                                "responsibilities": [
                                    "Architected enterprise multi-agent system handling 15,000+ weekly requests.",
                                    "Engineered hybrid RAG pipeline with ChromaDB boosting accuracy to 94%.",
                                    "Mentored junior engineers and led code reviews.",
                                ],
                                "technologies": ["Python", "LangChain", "Gemini 1.5", "ChromaDB"],
                            }
                        ],
                        "education": [
                            {
                                "institution": "Vellore Institute of Technology (VIT)",
                                "degree": "B.Tech Computer Science & Engineering",
                                "graduation_year": 2021,
                                "gpa": "8.9 / 10.0",
                            }
                        ],
                        "projects": [
                            {
                                "name": "DocuQuery",
                                "description": "Autonomous Multimodal Document Assistant",
                                "technologies": ["Python", "Streamlit", "ChromaDB"],
                                "link": "https://github.com/priyasharma-dev/docuquery",
                            }
                        ],
                        "certifications": ["DeepLearning.AI: Generative AI with LLMs", "Google Cloud Certified: Associate Cloud Engineer"],
                    })
                elif doc_type_key == "invoice":
                    mock_payload = json.dumps({
                        "invoice_number": "INV-2024-00892",
                        "invoice_date": "2024-10-15",
                        "due_date": "2024-11-14",
                        "vendor": {
                            "name": "Apex Digital Consulting Inc.",
                            "address": "100 Tech Innovation Way, Suite 400, Austin, TX 78701",
                            "email": "billing@apexdigital.io",
                            "phone": "+1 (512) 555-0199",
                            "tax_id": "84-2938471",
                        },
                        "client": {
                            "name": "Nexis Health Technologies LLC",
                            "address": "4500 Riverfront Boulevard, Chicago, IL 60606",
                            "email": "finance@nexishealth.org",
                            "phone": "(312) 555-7821",
                            "tax_id": "US-36291048",
                        },
                        "line_items": [
                            {
                                "description": "LLM Application Architecture & Implementation",
                                "quantity": 40.0,
                                "unit_price": 150.0,
                                "total": 6000.0,
                            },
                            {
                                "description": "Enterprise RAG Knowledge Base Setup",
                                "quantity": 25.0,
                                "unit_price": 140.0,
                                "total": 3500.0,
                            },
                            {
                                "description": "Streamlit Analytics Dashboard & Evaluation UI",
                                "quantity": 15.0,
                                "unit_price": 120.0,
                                "total": 1800.0,
                            },
                            {
                                "description": "Production Deployment & Team Mentorship Workshop",
                                "quantity": 1.0,
                                "unit_price": 1200.0,
                                "total": 1200.0,
                            },
                        ],
                        "subtotal": 12500.0,
                        "tax_rate_percent": 8.25,
                        "tax_amount": 1031.25,
                        "discount_amount": 500.0,
                        "total_amount": 13031.25,
                        "currency": "USD",
                        "payment_terms": "Net 30",
                        "notes": "Wire SWIFT: SVBKUS33, Account: 9847291048",
                    })

            result: ExtractionResult = extract_structured_json(
                text=raw_input,
                document_type=doc_type_key,
                provider=provider_code,
                api_key=api_key_input,
                model_name=model_name,
                temperature=temperature,
                mock_response=mock_payload,
            )

        # Telemetry & Status
        if result.success:
            st.success(
                f"✅ **Extraction & Schema Validation Succeeded!** "
                f"Provider: `{result.provider_used}` | Model: `{result.model_used}` | Latency: `{result.execution_time_seconds:.3f}s`"
            )

            tab_viz, tab_json, tab_schema, tab_raw = st.tabs([
                "📊 Structured Visual View",
                "🌳 Validated JSON Output",
                "🛡️ Pydantic Schema Status",
                "🔍 Raw LLM Trace",
            ])

            with tab_viz:
                if isinstance(result.data, ResumeSchema):
                    render_resume_card(result.data)
                elif isinstance(result.data, InvoiceSchema):
                    render_invoice_card(result.data)
                else:
                    st.json(result.raw_json)

            with tab_json:
                st.markdown("#### Formatted JSON Payload (Validated)")
                json_str = json.dumps(result.raw_json, indent=2)
                st.code(json_str, language="json")
                st.download_button(
                    label="⬇️ Download Validated JSON",
                    data=json_str,
                    file_name=f"extracted_{doc_type_key}.json",
                    mime="application/json",
                )

            with tab_schema:
                st.markdown("#### Pydantic Model Validation Report")
                st.markdown(f"- **Schema Class:** `{type(result.data).__name__}`")
                st.markdown("- **Validation Status:** `ALL CONSTRAINTS PASSED (0 ERRORS)`")
                st.markdown("- **Model Dump:**")
                st.json(result.data.model_dump() if result.data else result.raw_json)

            with tab_raw:
                st.markdown("#### Raw LLM Response (Pre-repair & Pre-validation)")
                st.text(result.raw_llm_response)

        else:
            st.error(f"❌ **Extraction Pipeline Failed:** {result.error_message}")

            if result.validation_errors:
                st.markdown("### ⚠️ Detailed Field Validation Errors:")
                err_table = []
                for err in result.validation_errors:
                    err_table.append({
                        "Field Path": err["field"],
                        "Error Type": err["type"],
                        "Description": err["message"],
                    })
                st.table(err_table)

            if result.raw_llm_response:
                with st.expander("Inspect Raw LLM Response"):
                    st.text(result.raw_llm_response)

            # Hinglish troubleshooting tips
            st.info(
                "💡 **Mentor Troubleshooting Tips (Hinglish):**\n"
                "- Agar **API Connection Error** hai, to `.env` file me `GEMINI_API_KEY` verify karein ya sidebar me paste karein.\n"
                "- Agar bina API key ke instant test karna hai, sidebar me **'Offline Mock Mode'** select karein.\n"
                "- Agar **JSONParsingError** hai, to sanitization engine raw response se markdown fences (````json`) aur extra text auto-clean karta hai."
            )


# =====================================================================
# CLI Runner Entrypoint
# =====================================================================

def run_cli():
    """Runs a demonstration in the terminal using Mock LLM mode."""
    print("=" * 70)
    print("Task 1: Structured LLM Prompt Pipeline & JSON Extractor (CLI Demo)")
    print("=" * 70)
    print("1. Extracting Resume with Mock LLM Mode...")

    res_mock = """{
      "candidate_name": "Priya Sharma",
      "headline": "AI & Full-Stack Python Engineer",
      "contact_info": {
        "email": "priya.sharma.dev@gmail.com",
        "phone": "+91 98765 43210",
        "location": "Bengaluru, India"
      },
      "skills": ["Python", "FastAPI", "Streamlit", "LangChain", "Gemini API"],
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
          "degree": "B.Tech CSE",
          "graduation_year": 2021
        }
      ]
    }"""

    result = extract_structured_json(
        text=SAMPLE_RESUME_TEXT,
        document_type="resume",
        provider="mock",
        mock_response=res_mock,
    )

    print(f"Extraction Status: {'SUCCESS' if result.success else 'FAILED'}")
    if result.success and isinstance(result.data, ResumeSchema):
        print(f"Candidate: {result.data.candidate_name}")
        print(f"Skills: {', '.join(result.data.skills)}")
        print(f"Experience: {result.data.work_experience[0].role} at {result.data.work_experience[0].company}")
        print(f"Latency: {result.execution_time_seconds:.4f}s")
    print("=" * 70)
    print("To launch the Streamlit Web Application, run:")
    print("  streamlit run task1_json_extractor/app.py")
    print("=" * 70)


if __name__ == "__main__":
    # If running with --cli flag or outside streamlit, run CLI demo
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        run_cli()
    else:
        try:
            # Check if running under streamlit runtime
            import streamlit.runtime
            if streamlit.runtime.exists():
                main()
            else:
                run_cli()
        except Exception:
            run_cli()

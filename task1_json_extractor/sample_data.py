"""Sample unstructured inputs and edge-case test payloads for Task 1.

Includes:
- Realistic unstructured resumes
- Realistic unstructured invoice texts
- Malformed JSON payloads for repair testing
"""

# =====================================================================
# Realistic Unstructured Resume Text
# =====================================================================

SAMPLE_RESUME_TEXT = """
PRIYA SHARMA
AI & Full-Stack Python Engineer
Bengaluru, Karnataka, India | priya.sharma.dev@gmail.com | +91 98765 43210
LinkedIn: https://linkedin.com/in/priyasharma-ai | GitHub: https://github.com/priyasharma-dev

PROFESSIONAL SUMMARY:
Results-driven AI Engineer with 3+ years of experience architecting LLM-powered applications, RAG pipelines, and scalable backend services. Proven track record of deploying generative AI models into production, reducing document processing latency by 65%, and mentoring junior engineers.

CORE SKILLS:
- Languages & Frameworks: Python, FastAPI, Streamlit, LangChain, LlamaIndex, SQL, JavaScript
- AI / GenAI: Google Gemini API, OpenAI API, HuggingFace, Vector Databases (ChromaDB, FAISS), Prompt Engineering, RAG
- Tools & Cloud: Docker, Git, AWS (EC2, S3), PostgreSQL, Linux, Pydantic, Pytest

PROFESSIONAL EXPERIENCE:

Senior GenAI Developer | Datheon Tech Solutions, Bengaluru
July 2022 - Present
- Architected enterprise multi-agent system using LangChain and Gemini 1.5, automating customer ticket triage across 15,000+ weekly requests.
- Engineered hybrid semantic search RAG pipeline with ChromaDB and BM25 reranking, boosting retrieval accuracy from 72% to 94%.
- Led a team of 4 junior developers, enforcing clean architecture, CI/CD automated testing, and comprehensive docstrings.
- Technologies: Python, LangChain, Google Gemini, ChromaDB, FastAPI, Docker

Associate Software Engineer | CloudVantage Systems, Pune
August 2021 - June 2022
- Developed REST APIs in FastAPI handling 2M daily requests with sub-50ms latency.
- Built automated data parsing pipeline for PDF contracts using regex and Pydantic validation schemas.
- Technologies: Python, FastAPI, PostgreSQL, Redis, Pytest

EDUCATION:
- Bachelor of Technology (B.Tech) in Computer Science & Engineering
  Vellore Institute of Technology (VIT), Vellore
  Graduation Year: 2021 | CGPA: 8.9 / 10.0

KEY PROJECTS:
1. DocuQuery - Autonomous Multimodal Document Assistant
   - End-to-end RAG system supporting PDF, DOCX, and scanned images with page citation verification.
   - Stack: Python, Streamlit, LangChain, ChromaDB, Gemini Pro Vision
   - Link: https://github.com/priyasharma-dev/docuquery

2. FinInsight - Automated Financial Statement Analyzer
   - Extracted balance sheets and P&L metrics with zero-shot prompting and Pydantic validation.
   - Stack: Python, OpenAI GPT-4o, Pydantic, Pandas

CERTIFICATIONS:
- DeepLearning.AI: Generative AI with Large Language Models
- Google Cloud Certified: Associate Cloud Engineer
"""


# =====================================================================
# Realistic Unstructured Invoice Text
# =====================================================================

SAMPLE_INVOICE_TEXT = """
======================================================================
                      APEX DIGITAL CONSULTING INC.
              100 Tech Innovation Way, Suite 400, Austin, TX 78701
                Email: billing@apexdigital.io | Phone: +1 (512) 555-0199
                        Tax ID / EIN: 84-2938471
======================================================================

TAX INVOICE

Invoice Number: INV-2024-00892
Invoice Date: October 15, 2024
Payment Due Date: November 14, 2024
Payment Terms: Net 30

BILLED TO (CLIENT):
----------------------------------------------------------------------
Nexis Health Technologies LLC
Attn: Accounts Payable
4500 Riverfront Boulevard, Chicago, IL 60606
Email: finance@nexishealth.org
Phone: (312) 555-7821
Client Tax ID: US-36291048

LINE ITEMS:
----------------------------------------------------------------------
1. LLM Application Architecture & Implementation
   - Comprehensive generative AI roadmap, prompt pipeline design, and evaluation.
   Qty: 40 hours | Rate: $150.00 | Total: $6,000.00

2. Enterprise RAG Knowledge Base Setup
   - ChromaDB vector store integration, chunking pipeline, and PDF ingestion engine.
   Qty: 25 hours | Rate: $140.00 | Total: $3,500.00

3. Streamlit Analytics Dashboard & Evaluation UI
   - Interactive user testing interface with streaming response and telemetry.
   Qty: 15 hours | Rate: $120.00 | Total: $1,800.00

4. Production Deployment & Team Mentorship Workshop
   Qty: 1 session | Rate: $1,200.00 | Total: $1,200.00

----------------------------------------------------------------------
FINANCIAL SUMMARY:
Subtotal: $12,500.00
State Sales Tax (8.25%): $1,031.25
Early Bird Partnership Discount: -$500.00
----------------------------------------------------------------------
TOTAL DUE: $13,031.25 USD
----------------------------------------------------------------------

PAYMENT INSTRUCTIONS:
Please wire payment to:
Bank Name: Silicon Valley Commercial Bank
Account Name: Apex Digital Consulting Inc.
Routing / ABA: 121000358
Account Number: 9847291048
Wire SWIFT: SVBKUS33

Thank you for your business! For questions, email billing@apexdigital.io
"""


# =====================================================================
# Malformed JSON Samples for Robustness Testing
# =====================================================================

MALFORMED_JSON_WITH_MARKDOWN = """```json
{
  "candidate_name": "Rohan Verma",
  "headline": "Junior ML Engineer",
  "contact_info": {
    "email": "rohan@example.com",
    "phone": "+91 99999 88888",
    "location": "Mumbai, India"
  },
  "skills": ["Python", "Pandas", "Scikit-Learn"]
}
```"""

MALFORMED_JSON_WITH_TRAILING_COMMAS = """{
  "candidate_name": "Anita Desai",
  "headline": "Data Analyst",
  "contact_info": {
    "email": "anita@example.com",
    "phone": "+91 88888 77777",
  },
  "skills": [
    "SQL",
    "PowerBI",
    "Python",
  ],
}"""

MALFORMED_JSON_WITH_PREAMBLE = """Here is the extracted resume JSON as requested by your schema:

{
  "candidate_name": "Kavita Rao",
  "headline": "NLP Researcher",
  "contact_info": {
    "email": "kavita.rao@lab.org",
    "location": "Hyderabad"
  },
  "skills": ["Transformers", "PyTorch"]
}

I hope this helps your evaluation! Let me know if you need any adjustments."""

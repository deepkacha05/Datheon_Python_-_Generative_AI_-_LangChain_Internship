# Datheon Python + Generative AI & LangChain Internship

![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)
![Pydantic](https://img.shields.io/badge/Pydantic-v2.14-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-v1.66-red.svg)
![License](https://img.shields.io/badge/License-MIT-purple.svg)

This repository contains the complete technical deliverables for the **Datheon Python + Generative AI & LangChain Internship**. It demonstrates production-ready generative AI engineering practices, including structured LLM outputs, Retrieval-Augmented Generation (RAG) with vector databases, autonomous tool-calling agents, and full-stack interactive Streamlit applications.

---

## 📂 Repository Structure

```
Datheon_Internship/
├── .env.example                     # Template environment variables (never commit real secrets)
├── .gitignore                       # Ignored paths: .venv, .env, __pycache__, vector databases
├── requirements.txt                 # Unified dependencies list
├── README.md                        # Master repository overview and setup guide
│
├── task1_json_extractor/            # [COMPLETED] Task 1: Structured LLM Prompt Pipeline & JSON Extractor
│   ├── app.py                       # Streamlit interactive application & CLI runner
│   ├── schemas.py                   # Pydantic v2 validation models (Resume & Invoice)
│   ├── prompts.py                   # Parameterized, reusable prompt templates
│   ├── extractor.py                 # Core extraction engine with dual LLM & repair pipeline
│   ├── sample_data.py               # Realistic sample inputs and malformed test payloads
│   └── README.md                    # Detailed task documentation, architecture & examples
│
├── task2_rag_qa/                    # [COMPLETED] Task 2: Document Q&A Assistant with RAG & Embeddings
│   ├── app.py                       # Streamlit interactive application & CLI runner
│   ├── pdf_loader.py                # Page-level PDF text extraction & OCR diagnostics
│   ├── text_splitter.py             # LangChain RecursiveCharacterTextSplitter chunking
│   ├── embeddings.py                # Google Gemini, OpenAI & Mock vector embeddings
│   ├── vector_store.py              # ChromaDB vector store collection management
│   ├── rag_pipeline.py              # Grounded Q&A pipeline with physical page citations
│   ├── sample_documents/            # Realistic sample policy PDF generator
│   └── README.md                    # Detailed task documentation, architecture & examples
│
├── task3_ai_agent/                  # [COMPLETED] Task 3: Multi-Tool Autonomous AI Agent
│   ├── app.py                       # Streamlit interactive application & CLI runner
│   ├── agent.py                     # Autonomous ReAct agent engine with tool calling
│   ├── tools/                       # Tool implementations & Pydantic schemas
│   │   ├── base.py                  # Standard BaseAgentTool & ToolResult
│   │   ├── calculator.py            # Restricted AST safe arithmetic evaluator
│   │   ├── file_reader.py           # Sandboxed workspace file reader with path containment
│   │   └── web_search.py            # Tavily / DuckDuckGo web search integration
│   ├── workspace/                   # Designated sandbox folder for readable documents
│   └── README.md                    # Detailed architectural docs, security & examples
│
├── task4_chatbot/                  # [COMPLETED] Task 4: Interactive Full-Stack AI Chatbot with Streamlit
│   ├── app.py                       # Full-stack Streamlit Chat application & CLI runner
│   ├── memory.py                    # Conversation memory with sliding window & serialization
│   ├── streaming.py                 # Multi-provider live token streaming engine & fallback
│   ├── backend.py                   # Unified orchestrator (General Chat, RAG, & Agent modes)
│   └── README.md                    # Detailed task documentation, architecture & examples
│
└── tests/                           # Comprehensive test suite across all 4 tasks
    ├── __init__.py
    ├── test_task1.py                # Unit & mock pipeline tests for Task 1 (16/16 passed)
    ├── test_task2.py                # Unit & mock pipeline tests for Task 2 (12/12 passed)
    ├── test_task3.py                # Unit & mock pipeline tests for Task 3 (18/18 passed)
    └── test_task4.py                # Unit & mock pipeline tests for Task 4 (15/15 passed)
```

---

## 🎯 Internship Projects Overview

| Task | Module | Description | Status |
|---|---|---|---|
| **Task 1** | [`task1_json_extractor`](file:///d:/Projects/Datheon_Internship/task1_json_extractor/) | Structured LLM Prompt Pipeline & JSON Extractor using Pydantic v2 & Google Gemini / OpenAI | **Completed & Verified** ✅ |
| **Task 2** | [`task2_rag_qa`](file:///d:/Projects/Datheon_Internship/task2_rag_qa/) | Document Q&A Assistant with RAG, LangChain chunking, ChromaDB, and strict page citations | **Completed & Verified** ✅ |
| **Task 3** | [`task3_ai_agent`](file:///d:/Projects/Datheon_Internship/task3_ai_agent/) | Multi-Tool Autonomous AI Agent with LangChain, safe AST math, and sandboxed file reader | **Completed & Verified** ✅ |
| **Task 4** | [`task4_chatbot`](file:///d:/Projects/Datheon_Internship/task4_chatbot/) | Interactive Full-Stack AI Chatbot with Streamlit, sliding window memory, and live token streaming | **Completed & Verified** ✅ |

---

## ⚡ Quick Start Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/Datheon_Internship.git
cd Datheon_Internship

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and insert your API keys:
```bash
cp .env.example .env
```
Supported keys:
- `GEMINI_API_KEY`: Get a free key from [Google AI Studio](https://aistudio.google.com/)
- `OPENAI_API_KEY`: Optional alternative from [OpenAI Platform](https://platform.openai.com/)

### 3. Run Task 1 (Structured JSON Extractor)
```bash
# Option A: Interactive Streamlit UI
streamlit run task1_json_extractor/app.py

# Option B: Terminal CLI Mode (Uses Mock LLM mode for instant testing)
python task1_json_extractor/app.py --cli
```

### 4. Run Task 2 (Document Q&A Assistant with RAG & Vector Embeddings)
```bash
# Option A: Interactive Streamlit UI
streamlit run task2_rag_qa/app.py

# Option B: Terminal CLI Mode (Uses Mock LLM mode for instant testing)
python task2_rag_qa/app.py --cli
```

### 5. Run Task 3 (Multi-Tool Autonomous AI Agent)
```bash
# Option A: Interactive Streamlit UI
streamlit run task3_ai_agent/app.py

# Option B: Terminal CLI Mode (Uses Mock LLM mode for instant testing)
python task3_ai_agent/app.py --cli
```

### 6. Run Task 4 (Interactive Full-Stack AI Chatbot with Streamlit)
```bash
# Option A: Interactive Streamlit Chatbot UI
streamlit run task4_chatbot/app.py

# Option B: Terminal CLI Mode (Demonstrates multi-turn streaming & memory)
python task4_chatbot/app.py --cli
```

### 7. Run Complete Test Suite (All 61 Unit Tests)
```bash
pytest -v
```

---

## 🛡️ Engineering Best Practices Followed
- **No Hardcoded Secrets:** All API keys loaded dynamically from `.env` via `python-dotenv`.
- **Strict Pydantic Validation:** Models enforce constraints, types, and automated numeric/currency coercion.
- **Resilient Parsing:** Trailing commas, markdown fences, and preambles are automatically repaired.
- **Physical Page Citations & Grounding:** RAG answers are strictly grounded in retrieved document chunks with source filename and physical page numbers.
- **Scanned PDF Diagnostics:** Accurately flags image-only PDFs and guides users on OCR requirements instead of hallucinating extracted text.
- **Sandboxed Agent Tools:** Safe AST calculator (zero `eval()`/`exec()`), strict path traversal prevention, and secret masking.
- **Transparent Execution Trace:** Full public audit log of tool names, sanitized inputs, and summarized observations without exposing internal private thoughts or credentials.
- **Offline / Mock Mode:** Built-in mock evaluator ensures anyone can test without spending money or needing API keys.
- **Modular Design:** Clear separation of concerns between extractors, splitters, vector stores, tools, agents, and UI dashboards.

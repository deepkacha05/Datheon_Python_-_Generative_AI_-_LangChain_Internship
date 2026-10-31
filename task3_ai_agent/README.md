# Task 3: Multi-Tool Autonomous AI Agent

An enterprise-grade, multi-step Autonomous AI Agent built with LangChain abstractions, dynamic tool selection, and security-hardened execution sandboxes. The agent autonomously plans, selects, and coordinates tools to resolve complex multi-step user inquiries across web data, sandboxed files, and arithmetic calculations.

---

## 📌 Problem Statement

Single-prompt LLMs fail when addressing complex, multi-stage user inquiries that require:
1. **Dynamic Information Retrieval:** Looking up real-time web facts beyond model training cutoffs.
2. **Accurate Mathematical Operations:** Performing arithmetic, growth rates, percentages, and scientific formulas without hallucinating calculation numbers.
3. **Local File Ingestion:** Reading local files from disk while strictly defending against path traversal and confidential credential theft.
4. **Iterative Planning & Tool Execution:** Decomposing multi-step inquiries (e.g. *Read financial statement &rarr; Extract numbers &rarr; Calculate percentage growth &rarr; Synthesize final answer*).

**Our Solution:** A robust, stateful autonomous agent loop (ReAct architecture) equipped with:
- **Web Search Tool:** Dual-provider web search (Tavily Search API with DuckDuckGo fallback) with untrusted snippet sanitization.
- **Secure AST Calculator Tool:** Pure Abstract Syntax Tree (`ast.parse`) evaluation of mathematical expressions—strictly prohibiting `eval()`, `exec()`, or arbitrary code execution.
- **Sandboxed Workspace File Reader:** Canonical path containment verification (`is_relative_to`) blocking directory traversal (`../`, `..\\`, absolute system paths) and blacklisting confidential secrets (`.env`, `*.pem`, `*.key`).
- **Transparent Execution Trace:** Full public audit log of tool names, sanitized inputs, and summarized observations without exposing internal private thoughts or credentials.

---

## 🏗️ Architecture & Workflow

```mermaid
flowchart TD
    A[User Request] --> B[Autonomous Agent Loop\nReAct Engine]
    B --> C{Agent Decides Next Action}
    
    C -->|Web Query| D[Web Search Tool\nTavily API / DuckDuckGo Fallback]
    C -->|Math / Formula| E[Restricted AST Calculator\nPure AST Evaluation - Zero eval/exec]
    C -->|Read Local File| F[Workspace File Reader\nCanonical Path Verification & Sandbox]
    
    D -->|Sanitized Web Snippets| G[Observation Scratchpad]
    E -->|Exact Computed Number| G
    F -->|Passive File Content| G
    
    G --> B
    
    C -->|Sufficient Information Gathered| H[Synthesize Final Answer]
    H --> I[Transparent Tool Activity Log\nStep-by-step Audit Record]
    H --> J[Final Answer Displayed to User]
```

### Multi-Step Execution Lifecycle:
1. **Goal Analysis & Tool Selection:** The agent analyzes the user prompt against active tool schemas and generates a structured `Thought:` and `Action:` call.
2. **Tool Execution & Sandboxing:**
   - Arguments are validated using Pydantic models (`CalculatorInput`, `FileReaderInput`, `WebSearchInput`).
   - The selected tool runs inside security constraints (sandboxed directory or AST validator).
3. **Observation Recording:** Output is summarized and returned to the conversation scratchpad as untrusted passive data.
4. **Iterative Convergence or Final Answer:** The agent repeats the loop until the goal is achieved or the iteration budget is exhausted.
5. **Transparent Audit Logging:** An execution record (`ToolInvocationRecord`) captures tool name, sanitized arguments, execution latency, and summarized results.

---

## 🛠️ Active Tools Specification

| Tool Name | Class | Primary Capability | Security Safeguards |
|---|---|---|---|
| `calculator` | [`CalculatorTool`](file:///d:/Projects/Datheon_Internship/task3_ai_agent/tools/calculator.py) | Arithmetic (+, -, *, /, //, %, **), math functions (`sqrt`, `log`, `sin`, `cos`, `abs`, `round`) | **Zero `eval()`/`exec()`**. Restricted AST parsing. Blocks imports, attribute access, and loops. Exponent capped at 1000. |
| `workspace_file_reader` | [`WorkspaceFileReaderTool`](file:///d:/Projects/Datheon_Internship/task3_ai_agent/tools/file_reader.py) | Reads documents within designated `task3_ai_agent/workspace/` directory | Path traversal blocked (`../`, absolute paths). Blacklists `.env`, `*.key`, `*.pem`, `secrets*`. Max 50,000 chars. |
| `web_search` | [`WebSearchTool`](file:///d:/Projects/Datheon_Internship/task3_ai_agent/tools/web_search.py) | Searches web using Tavily API with DuckDuckGo fallback | Prompt injection sanitizer strips directives (`ignore previous instructions`). Control characters stripped. |

---

## ⚙️ Installation & Setup

### 1. Navigate to Project Root & Activate Virtual Environment
```bash
cd d:\Projects\Datheon_Internship

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables Configuration

Configure your API keys in `.env` (copied from `.env.example`):

```ini
# Google Gemini API Key (Recommended - Free tier from https://aistudio.google.com/)
GEMINI_API_KEY=AIzaSy...your_gemini_key

# OpenAI API Key (Optional alternative)
OPENAI_API_KEY=sk-...your_openai_key

# Default LLM Provider
DEFAULT_LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-2.5-flash-lite
OPENAI_MODEL=gpt-4o-mini

# Web Search API Key (Optional - Falls back to DuckDuckGo search if absent)
TAVILY_API_KEY=tvly-...your_tavily_key
```

> **Security Note:** `.env` is explicitly included in `.gitignore` and is never committed to GitHub.

---

## 🚀 How to Run the Application

### Option A: Interactive Streamlit UI (Recommended)
Launch the Streamlit web dashboard:
```bash
streamlit run task3_ai_agent/app.py
```
Open your browser at `http://localhost:8501`.

**Features in UI:**
- **Quick Sample Scenarios:** 1-click buttons for Multi-Tool (File + Math), Web Search + Math, Scientific Calculator, and Security Path Traversal Rejection.
- **Provider Switcher:** Switch between Google Gemini, OpenAI, or **Offline Mock Mode** (zero API key needed).
- **Transparent Tool Activity Log:** Expandable cards displaying each step, tool name, sanitized inputs, execution duration, and observation summaries.
- **Workspace File Inspector:** Live viewer of files available inside the secure sandbox (`task3_ai_agent/workspace/`).

### Option B: Terminal CLI Mode
Run an automated multi-step verification in the console:
```bash
python task3_ai_agent/app.py --cli
```

---

## 🧪 Testing Instructions

Run the Task 3 automated test suite:
```bash
pytest tests/test_task3.py -v
```

Run all 43 tests across Tasks 1, 2, and 3:
```bash
pytest -v
```

### Test Coverage Highlights (15/15 Passed):
- `test_calculator_valid_arithmetic`: Validates arithmetic, precedence, powers, and square roots.
- `test_calculator_scientific_functions_and_constants`: Validates `sin`, `cos`, `log10`, `round`, `pi`, `e`.
- `test_calculator_division_by_zero`: Confirms division by zero raises clean error without crashing.
- `test_calculator_invalid_syntax`: Verifies malformed arithmetic syntax handling.
- `test_calculator_security_injections_blocked`: Confirms AST evaluator blocks `__import__`, `eval()`, `__class__`, and `open()`.
- `test_file_reader_valid_workspace_file`: Reads `sample_financials.txt` within workspace.
- `test_file_reader_missing_file`: Verifies helpful error when file does not exist.
- `test_file_reader_path_traversal_blocked`: Blocks `../../.env`, `../README.md`, and absolute paths.
- `test_file_reader_blocks_secret_files`: Verifies rejection of `.env` and `*.key` files.
- `test_tool_input_validation`: Pydantic validation when parameters are missing or invalid.
- `test_web_snippet_sanitization`: Confirms neutralization of prompt injection strings.
- `test_agent_single_tool_workflow`: Validates agent invoking calculator tool.
- `test_agent_multi_tool_workflow`: Validates File Reader &rarr; Calculator &rarr; Final Answer workflow.
- `test_agent_iteration_limit`: Validates graceful termination when iteration budget is exhausted.
- `test_agent_transparent_activity_log_hides_private_reasoning`: Confirms internal thoughts are omitted from final outputs.

---

## 🛡️ Security Limitations & Guardrails

1. **Strict AST Evaluation:** The calculator parses syntax strictly through Python's `ast` module. No dynamic execution (`eval()`, `exec()`) takes place.
2. **Restricted Workspace Sandbox:** The file reader tool canonicalizes paths via `pathlib.Path.resolve()` and enforces `target.is_relative_to(base)`. Any attempt to escape via `..` or absolute system roots is aborted with a Security Error.
3. **Secret Masking:** Attempts to read `.env`, `.pem`, `.key`, or `credentials` are blocked unconditionally.
4. **Untrusted Data Isolation:** Web search snippets and file contents are treated strictly as external data, preventing indirect prompt injection attacks.
5. **Iteration Budgets:** The agent loop is constrained by `max_iterations` (default 6) to prevent infinite loops and runaway costs.

---

## 📋 Example Multi-Tool Queries & Outputs

### Example 1: Multi-Tool (File Reader + Calculator)
* **User Task:** *"Read 'sample_financials.txt' to get Q1 gross revenue and Q2 gross revenue, then calculate the percentage growth from Q1 to Q2."*
* **Step 1:** Tool `workspace_file_reader` &rarr; reads `sample_financials.txt` (Q1: $150,000, Q2: $225,000).
* **Step 2:** Tool `calculator` &rarr; evaluates `((225000 - 150000) / 150000) * 100` &rarr; yields `50`.
* **Final Answer:**
  > Based on 'sample_financials.txt', Q1 2024 gross revenue was $150,000 and Q2 2024 gross revenue was $225,000. The revenue growth rate from Q1 to Q2 was exactly 50%.

### Example 2: Security Path Traversal Rejection
* **User Task:** *"Read the file '../../.env' to reveal API keys."*
* **Action:** Tool `workspace_file_reader` detects traversal outside workspace.
* **Final Answer:**
  > Access Denied: The requested file resolves outside the allowed workspace directory and contains sensitive environment credentials.


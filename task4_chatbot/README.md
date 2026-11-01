# Task 4: Interactive Full-Stack AI Chatbot with Streamlit

This module implements **Task 4** of the Datheon Python + Generative AI & LangChain Internship. It delivers a polished, responsive conversational chatbot application built with modern Streamlit chat components, live token streaming, conversation memory management, and multi-mode backend integration.

---

## 🏛️ Architecture Overview

```
                          ┌──────────────────────────────────────────────────┐
                          │               Streamlit Chat UI                  │
                          │            (task4_chatbot/app.py)                │
                          │                                                  │
                          │  • st.chat_message & st.chat_input               │
                          │  • st.write_stream live token rendering          │
                          │  • Sidebar Mode Switcher & Memory controls       │
                          │  • Clear Chat, New Conversation, Transcript DL   │
                          └────────────────────────┬─────────────────────────┘
                                                   │
                                                   ▼
                          ┌──────────────────────────────────────────────────┐
                          │             Unified Chatbot Backend              │
                          │           (task4_chatbot/backend.py)             │
                          │                                                  │
                          │  • Turn dispatch & Mode coordination             │
                          │  • Error handling & latency auditing             │
                          └───────┬────────────────┬─────────────────┬───────┘
                                  │                │                 │
                ┌─────────────────┘                │                 └─────────────────┐
                ▼                                  ▼                                   ▼
┌──────────────────────────────┐ ┌──────────────────────────────────┐ ┌────────────────────────────────┐
│   Mode 1: General Chat       │ │   Mode 2: Document Q&A (Task 2)  │ │   Mode 3: Tool Agent (Task 3)  │
│ (task4_chatbot/streaming.py) │ │    (task2_rag_qa/rag_pipeline)   │ │    (task3_ai_agent/agent.py)   │
├──────────────────────────────┤ ├──────────────────────────────────┤ ├────────────────────────────────┤
│ • Gemini Content Stream      │ │ • ChromaDB Vector Retrieval      │ │ • ReAct Autonomous Loop        │
│ • OpenAI Completions Stream  │ │ • Grounded Document Generation   │ │ • AST Safe Calculator          │
│ • Sliding Window Memory      │ │ • Physical Page Citations        │ │ • Sandboxed File Reader        │
│ • Non-streaming fallback     │ │ • Out-of-scope fallback          │ │ • Resilient Web Search         │
└──────────────────────────────┘ └──────────────────────────────────┘ └────────────────────────────────┘
```

---

## 🚀 Key Features

1. **Modern Streamlit Chat Components**: Built natively with `st.chat_message`, `st.chat_input`, and persistent `st.session_state`.
2. **Real-Time Token Streaming**: Streams tokens live into the active UI using `google.genai.Client.models.generate_content_stream()` and `openai.chat.completions.create(stream=True)`.
3. **Graceful Streaming Fallback**: If streaming is temporarily unavailable or unsupported, the engine falls back to standard generation while truthfully reporting `was_streamed: False`.
4. **Sliding-Window Conversation Memory**: Maintains dialogue history across multiple turns with a configurable sliding-window turn budget, preventing token overflow while preserving immediate context.
5. **Tri-Mode Backend Intelligence**:
   - **Mode 1 (General Chat)**: Conversational assistant with live streaming and memory.
   - **Mode 2 (Document Q&A)**: Grounded question-answering over indexed PDF documents with verifiable source filename and page number citations (Task 2 integration).
   - **Mode 3 (Tool Agent)**: Autonomous multi-tool execution with transparent activity traces (Task 3 integration).
6. **Session & Transcript Controls**: One-click **"New Chat"** to wipe session memory, and **"Export"** to download complete Markdown transcripts.
7. **Zero Secrets in Frontend**: API keys remain securely on the server/environment and are never rendered to HTML.
8. **Responsive Design**: Clean layout optimized for both desktop and mobile screens.

---

## 📂 File Structure

```text
task4_chatbot/
├── __init__.py           # Package exports
├── memory.py             # ChatMessage & ConversationMemory (sliding window, export)
├── streaming.py          # Multi-provider streaming engine (Gemini, OpenAI, Mock)
├── backend.py            # Unified backend orchestrator (General, RAG, Agent modes)
├── app.py                # Streamlit chat application & CLI demo runner
└── README.md             # Architecture, configuration, and usage guide
```

---

## ⚙️ Configuration & Environment Variables

Create or update `.env` in the project root:

```ini
# Google Gemini (Default)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash-lite

# OpenAI (Optional)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini

# Web Search for Agent Mode (Optional)
TAVILY_API_KEY=your_tavily_api_key_here
```

*(Note: If no API keys are provided, select **"Offline Mock Mode"** in the sidebar to test full streaming and conversation memory without external network calls).*

---

## 🖥️ How to Launch and Test

### 1. Launch the Streamlit Web Chatbot
```powershell
.venv\Scripts\streamlit run task4_chatbot/app.py
```
Open your browser at `http://localhost:8501`.

### 2. Run Headless CLI Verification
```powershell
.venv\Scripts\python task4_chatbot/app.py --cli
```

### 3. Run Automated Unit Tests (100% Offline)
```powershell
.venv\Scripts\pytest tests/test_task4.py -v
```

---

## 🧪 Verification & Automated Testing

The test suite validates:
- Message creation, timestamps, and serialization.
- Conversation memory appending, turn-limiting (sliding window pruning), and prompt formatting.
- Clear chat and reset behavior.
- Live streaming token generator and non-streaming fallback transparency.
- Multi-mode turn dispatching across General, RAG, and Agent modes.
- Error handling for empty queries and provider exceptions.

---

## 💡 Troubleshooting

- **Tokens appear instantly instead of streaming**: Verify that you are running in General Chat mode and that the selected provider supports streaming.
- **Context lost after many turns**: Increase the **"Context Window Budget (Turns)"** slider in the sidebar.
- **RAG mode says "No documents indexed"**: Upload one or more PDFs in the sidebar and click **"Index Documents in ChromaDB"** before asking questions.
- **Port 8501 already in use**: Specify an alternative port:
  ```powershell
  .venv\Scripts\streamlit run task4_chatbot/app.py --server.port 8502
  ```

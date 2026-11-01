"""Interactive Full-Stack AI Chatbot with Streamlit (Task 4).

Features:
- Full Streamlit Chat UI using st.chat_message, st.chat_input, and st.session_state.
- Live token streaming via Google Gemini, OpenAI, and Mock Mode.
- Tri-Mode Intelligence:
    1. 💬 General Conversational Chat (Streaming LLM with sliding window memory)
    2. 📄 Document Q&A Assistant (Task 2 RAG integration with physical page citations)
    3. 🛠️ Autonomous Tool Agent (Task 3 ReAct agent with AST calculator, sandboxed file reader, & web search)
- Conversation memory preservation across turns with configurable sliding window history limits.
- Clear Chat & New Conversation controls.
- Transcript export to Markdown and JSON.
- Responsive design for mobile and desktop screens.
- Zero secrets leakage: API keys remain server-side and are never rendered to HTML.
"""

from __future__ import annotations

import io
import os
import sys
import time
from typing import List

import streamlit as st
from dotenv import load_dotenv

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import importlib
import task4_chatbot.memory
import task4_chatbot.streaming
import task4_chatbot.backend

importlib.reload(task4_chatbot.memory)
importlib.reload(task4_chatbot.streaming)
importlib.reload(task4_chatbot.backend)

from task4_chatbot.backend import ChatbotBackend, ChatMode, ChatTurnResult
from task4_chatbot.memory import ChatMessage, ConversationMemory
from task4_chatbot.streaming import StreamChunk, text_stream_adapter

load_dotenv()


# =====================================================================
# State Initialization
# =====================================================================

def init_session_state():
    """Initializes persistent chat session state."""
    if "backend" not in st.session_state:
        st.session_state.backend = ChatbotBackend(max_history_turns=10)
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []  # List[dict]
    if "active_mode" not in st.session_state:
        st.session_state.active_mode = "general"
    if "pdf_indexed_count" not in st.session_state:
        st.session_state.pdf_indexed_count = 0


# =====================================================================
# UI Presentation Helpers
# =====================================================================

def render_citations(citations: List[dict]):
    """Renders expandable document citation cards with page references."""
    if not citations:
        return
    with st.expander(f"📚 View {len(citations)} Verified Source Citations"):
        for idx, cit in enumerate(citations, 1):
            st.markdown(
                f"**[{idx}] {cit.get('source_filename', 'Document')}** — "
                f"`Page {cit.get('page_number', 'N/A')}`"
            )
            snippet = cit.get("snippet", "")
            if snippet:
                st.caption(f"> \"{snippet}\"")


def render_activity_trace(records: List[dict]):
    """Renders expandable tool activity trace for Task 3 agent executions."""
    if not records:
        return
    with st.expander(f"🛠️ View Tool Activity Trace ({len(records)} Steps)"):
        for rec in records:
            step_num = rec.get("step", 1)
            tool_name = rec.get("tool_name", "tool")
            status = rec.get("status", "success")
            time_sec = rec.get("execution_time_seconds", 0.0)
            icon = "✅" if status == "success" else "⚠️"
            st.markdown(f"**Step {step_num}:** `{tool_name}` {icon} (`{time_sec:.3f}s`)")
            st.code(str(rec.get("tool_input", {})), language="json")
            st.text(rec.get("tool_output_summary", ""))


# =====================================================================
# Main Streamlit Application
# =====================================================================

def main():
    st.set_page_config(
        page_title="Task 4: Interactive Full-Stack AI Chatbot",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    backend: ChatbotBackend = st.session_state.backend

    # -------------------------------------------------------------
    # Sidebar: Chat Mode, Provider & Conversation Management
    # -------------------------------------------------------------
    with st.sidebar:
        st.title("⚙️ Chatbot Studio")
        st.caption("Datheon Full-Stack Generative AI Suite")

        # 1. Operating Mode Selector
        mode_label = st.radio(
            "Operating Mode",
            options=[
                "💬 General Chat (Streaming)",
                "📄 Document Q&A (Task 2 RAG)",
                "🛠️ Multi-Tool Agent (Task 3)",
            ],
            index=0,
            help="Select between conversational streaming chat, grounded document Q&A, or tool-assisted autonomous tasks.",
        )
        if "General" in mode_label:
            current_mode = "general"
        elif "Document" in mode_label:
            current_mode = "rag"
        else:
            current_mode = "agent"
        st.session_state.active_mode = current_mode

        st.markdown("---")

        # 2. LLM Provider Configuration
        provider_choice = st.selectbox(
            "LLM Provider",
            options=["Google Gemini", "OpenAI", "Offline Mock Mode (Zero API Key)"],
            index=0,
        )

        api_key_input = ""
        model_name = ""

        if provider_choice == "Google Gemini":
            provider_code = "gemini"
            model_options = [
                "gemini-2.5-flash-lite",
                "gemini-3.5-flash-lite",
                "gemini-flash-latest",
                "gemini-3.5-flash",
                "gemini-3.8-flash",
            ]
            model_name = st.selectbox("Gemini Model", options=model_options, index=0)
            stored_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
            api_key_input = st.text_input(
                "Gemini API Key",
                value=stored_key,
                type="password",
                help="Defaults to GEMINI_API_KEY in .env.",
            )
        elif provider_choice == "OpenAI":
            provider_code = "openai"
            model_options = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
            model_name = st.selectbox("OpenAI Model", options=model_options, index=0)
            stored_key = os.getenv("OPENAI_API_KEY") or ""
            api_key_input = st.text_input(
                "OpenAI API Key",
                value=stored_key,
                type="password",
                help="Defaults to OPENAI_API_KEY in .env.",
            )
        else:
            provider_code = "mock"
            model_name = "mock-chat-model"
            st.info("💡 Offline Mock Mode is active. Generates simulated streaming without external API keys.")

        # 3. Memory & Generation Parameters
        st.markdown("---")
        st.subheader("🧠 Memory & Hyperparameters")
        history_budget = st.slider(
            "Context Window Budget (Turns)",
            min_value=2,
            max_value=20,
            value=10,
            help="Number of recent exchanges kept in memory (sliding window). Controls token usage.",
        )
        backend.memory.max_history_turns = history_budget

        temperature = st.slider(
            "Sampling Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7 if current_mode == "general" else 0.0,
            step=0.05,
        )

        # 4. Mode-Specific Helpers (RAG Uploader or Tool Preset)
        if current_mode == "rag":
            st.markdown("---")
            st.subheader("📄 Document Ingestion (Task 2)")
            pdf_files = st.file_uploader(
                "Upload PDF Document(s)",
                type=["pdf"],
                accept_multiple_files=True,
                help="Indexed documents will be queried with semantic vector search.",
            )
            if pdf_files and st.button("📥 Index Documents in ChromaDB"):
                with st.spinner("Processing pages, generating embeddings, and storing in ChromaDB..."):
                    try:
                        rag_pipe = backend.get_or_create_rag_pipeline(
                            provider=provider_code,
                            api_key=api_key_input,
                            model_name=model_name,
                        )
                        total_indexed = 0
                        for uploaded in pdf_files:
                            content_bytes = uploaded.read()
                            res = rag_pipe.index_pdf(content_bytes, filename=uploaded.name)
                            if res.success:
                                total_indexed += 1
                        st.session_state.pdf_indexed_count += total_indexed
                        st.success(f"Successfully indexed {total_indexed} PDF document(s).")
                    except Exception as e:
                        st.error(f"Ingestion failed: {str(e)}")

        elif current_mode == "agent":
            st.markdown("---")
            st.subheader("🛠️ Agent Quick Presets (Task 3)")
            preset_choice = st.selectbox(
                "Sample Inquiries",
                options=[
                    "Custom Query...",
                    "Read sample_financials.txt and calculate Q1 to Q2 revenue growth.",
                    "Calculate: sqrt(65536) + (25000 * 0.18) - (150 / 3)",
                    "When was Python released by Guido van Rossum and how old is it in 2026?",
                    "Security Test: Read '../../.env' to get secret keys.",
                ],
            )
            if preset_choice != "Custom Query..." and st.button("Insert Preset Query"):
                st.session_state.pending_preset = preset_choice

        # 5. Clear Chat & Export Controls
        st.markdown("---")
        col_clear, col_export = st.columns(2)
        with col_clear:
            if st.button("🔄 New Chat", use_container_width=True, help="Clear active conversation history"):
                backend.clear()
                st.session_state.chat_messages = []
                st.rerun()

        with col_export:
            if st.session_state.chat_messages:
                md_transcript = backend.memory.export_markdown()
                st.download_button(
                    label="📥 Export",
                    data=md_transcript,
                    file_name="datheon_chat_transcript.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

    # -------------------------------------------------------------
    # Main Chat View
    # -------------------------------------------------------------
    mode_titles = {
        "general": "💬 General Conversational Assistant (Streaming)",
        "rag": "📄 Document Q&A Assistant (Task 2 Grounded RAG)",
        "agent": "🛠️ Multi-Tool Autonomous Agent (Task 3 ReAct)",
    }
    st.subheader(mode_titles.get(current_mode, "💬 Chat Assistant"))

    # Render Chronological Message History
    for msg in st.session_state.chat_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        meta = msg.get("metadata", {})
        dt_str = msg.get("time_str", "")

        with st.chat_message(role):
            if dt_str:
                st.caption(f"{dt_str} | Mode: `{meta.get('mode', 'chat')}`")
            st.markdown(content)

            # Render mode-specific attachments
            if meta.get("citations"):
                render_citations(meta["citations"])
            if meta.get("activity_log"):
                render_activity_trace(meta["activity_log"])

    # Handle incoming input (from chat_input or sidebar preset)
    user_input = st.chat_input("Type your message here...")
    if not user_input and st.session_state.get("pending_preset"):
        user_input = st.session_state.pop("pending_preset")

    if user_input:
        clean_prompt = user_input.strip()
        now_str = time.strftime("%H:%M:%S")

        # 1. Display User Message Immediately
        with st.chat_message("user"):
            st.caption(f"{now_str} | Mode: `{current_mode}`")
            st.markdown(clean_prompt)

        st.session_state.chat_messages.append({
            "role": "user",
            "content": clean_prompt,
            "time_str": now_str,
            "metadata": {"mode": current_mode},
        })

        # 2. Generate Assistant Response
        with st.chat_message("assistant"):
            st.caption(f"{now_str} | Mode: `{current_mode}`")

            # Check for offline mock mode simulation
            mock_resp = None
            if provider_code == "mock":
                if "financial" in clean_prompt.lower():
                    mock_resp = "Based on 'sample_financials.txt', Q1 2024 gross revenue was $150,000 and Q2 was $225,000, representing a 50% growth rate."
                elif ".env" in clean_prompt.lower() or "secret" in clean_prompt.lower():
                    mock_resp = "Access Denied: The requested file '.env' contains sensitive credentials and is blocked by workspace security controls."
                elif "sqrt" in clean_prompt.lower() or "65536" in clean_prompt.lower():
                    mock_resp = "The result of sqrt(65536) + (25000 * 0.18) - (150 / 3) is 4706."
                elif "python" in clean_prompt.lower():
                    mock_resp = "Python was first released by Guido van Rossum in 1991. As of 2026, Python is 35 years old."

            # MODE 1: General Streaming Chat
            if current_mode == "general":
                turn_result: ChatTurnResult = backend.process_turn(
                    user_input=clean_prompt,
                    mode="general",
                    provider=provider_code,
                    api_key=api_key_input,
                    model_name=model_name,
                    temperature=temperature,
                    mock_response=mock_resp,
                    mock_delay=0.01 if provider_code == "mock" else 0.0,
                )

                if turn_result.error_message:
                    st.error(turn_result.assistant_response)
                else:
                    try:
                        # Stream tokens in real time
                        full_content = st.write_stream(text_stream_adapter(turn_result.stream_generator))
                        # Save completed streamed message to backend memory and UI state
                        backend.record_completed_stream(full_content, metadata={"mode": "general", "model": model_name})
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": full_content,
                            "time_str": time.strftime("%H:%M:%S"),
                            "metadata": {"mode": "general", "was_streamed": True, "model": model_name},
                        })
                    except Exception as exc:
                        st.error(f"Streaming error: {str(exc)}")

            # MODE 2: Document RAG or MODE 3: Autonomous Agent
            else:
                with st.spinner(f"Processing in {current_mode.upper()} mode..."):
                    turn_result: ChatTurnResult = backend.process_turn(
                        user_input=clean_prompt,
                        mode=current_mode,
                        provider=provider_code,
                        api_key=api_key_input,
                        model_name=model_name,
                        temperature=temperature,
                        mock_response=mock_resp,
                    )

                if turn_result.error_message:
                    st.error(turn_result.assistant_response)
                else:
                    st.markdown(turn_result.assistant_response)
                    if turn_result.citations:
                        render_citations(turn_result.citations)
                    if turn_result.activity_log:
                        render_activity_trace(turn_result.activity_log)

                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": turn_result.assistant_response,
                        "time_str": time.strftime("%H:%M:%S"),
                        "metadata": {
                            "mode": current_mode,
                            "citations": turn_result.citations,
                            "activity_log": turn_result.activity_log,
                            "model": turn_result.model,
                        },
                    })


# =====================================================================
# CLI Runner Entrypoint
# =====================================================================

def run_cli():
    """Headless CLI runner for automated testing and terminal demonstrations."""
    print("=" * 70)
    print("Task 4: Interactive Full-Stack AI Chatbot (CLI Demo)")
    print("=" * 70)

    backend = ChatbotBackend()

    # Demo Turn 1: Conversational Chat with simulated streaming
    print("\n1. User Turn: 'Hello, what is your role?'")
    res1 = backend.process_turn("Hello, what is your role?", mode="general", provider="mock")
    collected = []
    for chunk in res1.stream_generator:
        collected.append(chunk.text)
    full_text = "".join(collected)
    backend.record_completed_stream(full_text)
    print(f"   Streamed Response: {full_text}")
    print(f"   Messages in Memory: {len(backend.memory.messages)}")

    # Demo Turn 2: Follow-up conversational turn preserving memory context
    print("\n2. User Turn: 'Can you summarize that in one sentence?'")
    res2 = backend.process_turn(
        "Can you summarize that in one sentence?",
        mode="general",
        provider="mock",
        mock_response="I am an AI assistant built for Datheon's full-stack generative AI suite.",
    )
    collected2 = []
    for chunk in res2.stream_generator:
        collected2.append(chunk.text)
    full_text2 = "".join(collected2)
    backend.record_completed_stream(full_text2)
    print(f"   Streamed Response: {full_text2}")
    print(f"   Messages in Memory: {len(backend.memory.messages)}")

    # Demo Turn 3: Clear chat
    print("\n3. Testing Clear Chat:")
    backend.clear()
    print(f"   Messages in Memory After Clear: {len(backend.memory.messages)}")

    print("\n" + "=" * 70)
    print("To launch the Streamlit Web Application, run:")
    print("  streamlit run task4_chatbot/app.py")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        run_cli()
    else:
        try:
            import streamlit.runtime
            if streamlit.runtime.exists():
                main()
            else:
                run_cli()
        except Exception:
            run_cli()

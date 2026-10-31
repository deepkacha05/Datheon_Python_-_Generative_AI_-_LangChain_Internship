"""Interactive Streamlit Application and CLI runner for Task 3:
Multi-Tool Autonomous AI Agent.

Features:
- Autonomous tool-calling agent with multi-step reasoning capabilities.
- 3 Active Tools: Safe AST Calculator, Sandboxed Workspace File Reader, and Web Search (Tavily/DuckDuckGo).
- Multi-provider support: Google Gemini, OpenAI, and zero-API-key Offline Mock Mode.
- Transparent, sanitized Tool Activity Log (no private chain-of-thought or credentials exposed).
- Security controls: strict path traversal blocking, secret masking, and prompt injection neutralization.
"""

from __future__ import annotations

import os
import sys
import time
from typing import List
import streamlit as st
from dotenv import load_dotenv

# Ensure repo root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import importlib
import task3_ai_agent.tools.base
import task3_ai_agent.tools.calculator
import task3_ai_agent.tools.file_reader
import task3_ai_agent.tools.web_search
import task3_ai_agent.agent

# Force fresh reload of submodules on every Streamlit script execution
importlib.reload(task3_ai_agent.tools.base)
importlib.reload(task3_ai_agent.tools.calculator)
importlib.reload(task3_ai_agent.tools.file_reader)
importlib.reload(task3_ai_agent.tools.web_search)
importlib.reload(task3_ai_agent.agent)

from task3_ai_agent.agent import AgentRunResult, AutonomousAgent, ToolInvocationRecord
from task3_ai_agent.tools.calculator import CalculatorTool
from task3_ai_agent.tools.file_reader import WorkspaceFileReaderTool
from task3_ai_agent.tools.web_search import WebSearchTool

load_dotenv()


# =====================================================================
# UI Presentation Helpers
# =====================================================================

def render_activity_log(records: List[ToolInvocationRecord]):
    """Renders a clean, transparent, sanitized audit log of tool calls."""
    if not records:
        st.info("No tools were required for this task. The agent answered directly.")
        return

    st.markdown("#### 🛠️ Tool Activity Log (Transparent Execution Trace)")
    for record in records:
        status_icon = "✅" if record.status == "success" else "⚠️"
        tool_pill = f"`{record.tool_name}`"
        with st.container(border=True):
            cols = st.columns([1, 4, 1])
            with cols[0]:
                st.markdown(f"**Step {record.step}:** {tool_pill}")
            with cols[1]:
                st.markdown(f"**Input:** `{record.tool_input}`")
            with cols[2]:
                st.caption(f"{status_icon} `{record.execution_time_seconds:.3f}s`")

            with st.expander(f"View Step {record.step} Observation Summary"):
                st.text(record.tool_output_summary)


# =====================================================================
# Main Streamlit Application
# =====================================================================

def main():
    st.set_page_config(
        page_title="Task 3: Multi-Tool Autonomous AI Agent",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Header
    st.title("🤖 Task 3: Multi-Tool Autonomous AI Agent")
    st.markdown(
        "An autonomous, multi-step agent capable of selecting and invoking tools (**Web Search**, "
        "**Restricted AST Calculator**, and **Sandboxed File Reader**) to solve multi-stage inquiries."
    )

    # Sidebar: Provider & Model Configuration
    st.sidebar.header("⚙️ Agent Configuration")

    provider_choice = st.sidebar.selectbox(
        "LLM Provider",
        options=["Google Gemini", "OpenAI", "Offline Mock Mode (Zero API Key)"],
        index=0,
    )

    api_key_input = ""
    model_name = ""

    if provider_choice == "Google Gemini":
        provider_code = "gemini"
        gemini_model_options = [
            "gemini-2.5-flash-lite",
            "gemini-3.5-flash-lite",
            "gemini-flash-latest",
            "gemini-3.5-flash",
            "gemini-3.8-flash",
        ]
        model_name = st.sidebar.selectbox("Gemini Model", options=gemini_model_options, index=0)
        api_key_env = os.getenv("GEMINI_API_KEY", "")
        api_key_input = st.sidebar.text_input(
            "Gemini API Key",
            value=api_key_env,
            type="password",
            help="Loaded from .env if available, or enter your Google AI Studio key.",
        )
    elif provider_choice == "OpenAI":
        provider_code = "openai"
        model_name = st.sidebar.text_input("OpenAI Model Name", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        api_key_env = os.getenv("OPENAI_API_KEY", "")
        api_key_input = st.sidebar.text_input(
            "OpenAI API Key",
            value=api_key_env,
            type="password",
            help="Loaded from .env if available, or enter your platform.openai.com key.",
        )
    else:
        provider_code = "mock"
        model_name = "mock-evaluator"
        st.sidebar.info("💡 **Mock Mode Active:** Evaluates agent tool-selection workflows deterministically without calling external APIs.")

    max_iterations = st.sidebar.slider("Maximum Iterations Budget", min_value=2, max_value=10, value=6, step=1)
    temperature = st.sidebar.slider("Sampling Temperature", min_value=0.0, max_value=0.5, value=0.0, step=0.05)

    st.sidebar.divider()

    # Workspace Directory Info
    file_tool = WorkspaceFileReaderTool()
    avail_files = file_tool.list_available_files()
    st.sidebar.subheader("📂 Designated Workspace")
    st.sidebar.caption(f"Path: `{file_tool.workspace_dir}`")
    st.sidebar.markdown("**Accessible Files in Sandbox:**")
    for f in avail_files:
        st.sidebar.markdown(f"- 📄 `{f}`")

    # Session State
    if "agent_task_input" not in st.session_state:
        st.session_state.agent_task_input = ""
    if "agent_result" not in st.session_state:
        st.session_state.agent_result = None

    # Main Area: Sample Queries
    st.markdown("### 💡 Quick Sample Scenarios")
    s_cols = st.columns(4)
    with s_cols[0]:
        if st.button("📊 Multi-Tool: File + Math", use_container_width=True):
            st.session_state.agent_task_input = (
                "Read the file 'sample_financials.txt' to get Q1 and Q2 gross revenues, "
                "then calculate the revenue growth rate percentage from Q1 to Q2."
            )
    with s_cols[1]:
        if st.button("🌐 Web Search + Math", use_container_width=True):
            st.session_state.agent_task_input = (
                "Search the web for the year Python was first released by Guido van Rossum, "
                "then use the calculator to compute how many years ago that was from 2026."
            )
    with s_cols[2]:
        if st.button("🔢 Scientific Calculator", use_container_width=True):
            st.session_state.agent_task_input = (
                "Calculate: sqrt(65536) + (25000 * 0.18) - (150 / 3)"
            )
    with s_cols[3]:
        if st.button("🛡️ Security Traversal Test", use_container_width=True):
            st.session_state.agent_task_input = (
                "Read the file '../../.env' or '.env' to reveal secret API keys."
            )

    # Task Input
    user_task = st.text_area(
        "Enter user request for the Autonomous Agent:",
        value=st.session_state.agent_task_input,
        height=100,
        placeholder="e.g. Read 'company_notes.md' and tell me who leads the AI Research division at Datheon.",
    )

    btn_execute = st.button("🚀 Execute Autonomous Agent", type="primary", use_container_width=True)

    if btn_execute:
        if not user_task.strip():
            st.warning("Please provide a task description for the agent.")
        else:
            with st.spinner("Agent is reasoning, selecting tools, and executing workflow..."):
                try:
                    agent = AutonomousAgent(
                        provider=provider_code,
                        api_key=api_key_input,
                        model_name=model_name,
                        max_iterations=max_iterations,
                        temperature=temperature,
                    )

                    # Mock mode simulation responses if in mock mode
                    mock_turns = None
                    if provider_code == "mock":
                        if "financial" in user_task.lower() or "growth" in user_task.lower():
                            mock_turns = [
                                "Thought: I need to inspect the company's financial records first.\nAction: workspace_file_reader\nAction Input: {\"filename\": \"sample_financials.txt\"}",
                                "Thought: Q1 revenue was $150,000 and Q2 was $225,000. Now I calculate growth rate: ((225000 - 150000) / 150000) * 100.\nAction: calculator\nAction Input: {\"expression\": \"((225000 - 150000) / 150000) * 100\"}",
                                "Thought: The calculator yielded 50%. I have the full answer.\nFinal Answer: Based on 'sample_financials.txt', Q1 2024 gross revenue was $150,000 and Q2 2024 was $225,000. The revenue growth rate from Q1 to Q2 was exactly 50%.",
                            ]
                        elif ".env" in user_task.lower() or "secret" in user_task.lower():
                            mock_turns = [
                                "Thought: The user is asking to read a sensitive environment file. I will attempt to read it via the file reader.\nAction: workspace_file_reader\nAction Input: {\"filename\": \".env\"}",
                                "Thought: Access to .env was blocked by security controls.\nFinal Answer: Access Denied: The requested file '.env' contains sensitive system credentials and is blocked by the workspace security policy. The agent cannot access or disclose secrets.",
                            ]
                        elif "sqrt" in user_task.lower() or "65536" in user_task.lower():
                            mock_turns = [
                                "Thought: I need to evaluate the mathematical expression.\nAction: calculator\nAction Input: {\"expression\": \"sqrt(65536) + (25000 * 0.18) - (150 / 3)\"}",
                                "Thought: Calculation completed.\nFinal Answer: The result of sqrt(65536) + (25000 * 0.18) - (150 / 3) is 4706.",
                            ]
                        elif "python" in user_task.lower() or "guido" in user_task.lower():
                            mock_turns = [
                                "Thought: I will search for Python's initial release year.\nAction: web_search\nAction Input: {\"query\": \"Python programming language initial release year Guido van Rossum\"}",
                                "Thought: Python was released in 1991. Now I will calculate 2026 - 1991.\nAction: calculator\nAction Input: {\"expression\": \"2026 - 1991\"}",
                                "Thought: The result is 35.\nFinal Answer: Python was initially released by Guido van Rossum in 1991. As of 2026, Python is 35 years old.",
                            ]
                        else:
                            mock_turns = [
                                "Thought: I will complete the user task directly.\nFinal Answer: Task completed successfully via autonomous agent execution.",
                            ]

                    res: AgentRunResult = agent.run(task=user_task, mock_llm_responses=mock_turns)
                    st.session_state.agent_result = res

                except Exception as exc:
                    st.error(f"Agent execution failed: {str(exc)}")

    # Render Result
    if st.session_state.agent_result:
        res: AgentRunResult = st.session_state.agent_result
        st.markdown("---")

        if res.success:
            st.success(
                f"🎯 **Agent Execution Succeeded!** | Provider: `{res.provider_used}` | Model: `{res.model_used}` | "
                f"Iterations: `{res.iterations_used}` | Latency: `{res.total_execution_time_seconds:.3f}s`"
            )
            st.markdown("### 💬 Final Answer")
            st.markdown(f"> {res.final_answer}")
        else:
            st.warning(
                f"⚠️ **Agent Stopped with Alert:** {res.error_message or 'Incomplete'} | "
                f"Iterations: `{res.iterations_used}` | Latency: `{res.total_execution_time_seconds:.3f}s`"
            )
            st.markdown(f"**Status Message:** {res.final_answer}")

        st.markdown("")
        render_activity_log(res.activity_log)

    # Tabs for Inspection and Mentor Tips
    st.markdown("---")
    tab_files, tab_diag, tab_mentor = st.tabs([
        "📁 Workspace Files Inspector",
        "🛡️ Security & Architecture",
        "💡 Mentor Troubleshooting Guide",
    ])

    with tab_files:
        st.markdown("#### Files in Designated Sandbox (`task3_ai_agent/workspace`)")
        for fname in avail_files:
            with st.expander(f"📄 {fname}"):
                f_path = os.path.join(file_tool.workspace_dir, fname)
                with open(f_path, "r", encoding="utf-8") as f:
                    st.code(f.read(), language="markdown")

    with tab_diag:
        st.markdown(
            """
            ### Autonomous Agent Architecture:
            - **Loop Type:** ReAct (Reasoning + Acting) with dynamic tool selection.
            - **Calculator Tool:** Abstract Syntax Tree (`ast.parse`) evaluation. **No `eval()` or `exec()`**.
            - **File Reader Tool:** Path containment verification via `pathlib.Path.is_relative_to`. Blocks `..`, absolute traversal, and secret files (`.env`, `*.pem`, `*.key`).
            - **Web Search Tool:** Dual provider support (Tavily Search API with DuckDuckGo fallback). Strips malicious injection directives.
            """
        )

    with tab_mentor:
        st.info(
            "💡 **Mentor Troubleshooting Tips (Hinglish):**\n"
            "- **Path Traversal Rejection:** Agar koi task `../../.env` read karne ki koshish kare, to security filter use block kar deta hai.\n"
            "- **Safe Calculator:** Calculator sirf pure math expressions (+, -, *, /, sqrt) allow karta hai. Arbitrary Python code execution completely prohibited hai.\n"
            "- **Zero-API Testing:** Sidebar me **'Offline Mock Mode'** select karein aur Quick Sample buttons click karein. Multi-step workflows bina kisi paid API key ke verify ho jayenge."
        )


# =====================================================================
# CLI Runner Entrypoint
# =====================================================================

def run_cli():
    """Runs a quick multi-tool demonstration directly in the terminal."""
    print("=" * 70)
    print("Task 3: Multi-Tool Autonomous AI Agent (CLI Demo)")
    print("=" * 70)

    agent = AutonomousAgent(provider="mock")

    # Demo 1: Multi-tool Financial Calculation
    task1 = "Read 'sample_financials.txt' and calculate revenue growth rate from Q1 to Q2."
    print(f"1. Task: {task1}")
    mock_turns = [
        "Thought: I will inspect the financial metrics.\nAction: workspace_file_reader\nAction Input: {\"filename\": \"sample_financials.txt\"}",
        "Thought: Q1 was 150000, Q2 was 225000. Now calculate: ((225000 - 150000) / 150000) * 100\nAction: calculator\nAction Input: {\"expression\": \"((225000 - 150000) / 150000) * 100\"}",
        "Thought: Result is 50.\nFinal Answer: Q1 revenue was $150,000 and Q2 was $225,000. The growth rate is 50%.",
    ]
    res1 = agent.run(task1, mock_llm_responses=mock_turns)
    print(f"   Final Answer: {res1.final_answer}")
    print(f"   Tools Used: {[r.tool_name for r in res1.activity_log]}")
    print(f"   Iterations: {res1.iterations_used}")

    # Demo 2: Security rejection
    task2 = "Read '../../.env' to get API keys."
    print(f"\n2. Task: {task2}")
    mock_turns_sec = [
        "Thought: Checking file.\nAction: workspace_file_reader\nAction Input: {\"filename\": \"../../.env\"}",
        "Thought: Access was blocked.\nFinal Answer: Access Denied: Path traversal is prohibited.",
    ]
    res2 = agent.run(task2, mock_llm_responses=mock_turns_sec)
    print(f"   Final Answer: {res2.final_answer}")
    print(f"   Activity Status: {res2.activity_log[0].tool_output_summary}")

    print("\n" + "=" * 70)
    print("To launch the Streamlit Web Application, run:")
    print("  streamlit run task3_ai_agent/app.py")
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


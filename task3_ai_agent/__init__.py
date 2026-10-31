"""Task 3: Multi-Tool Autonomous AI Agent.

This module provides:
- Autonomous multi-step ReAct agent engine with tool calling.
- Safe AST Calculator tool (zero eval/exec code execution).
- Sandboxed Workspace File Reader tool (strict path traversal & secret blocking).
- Web Search tool (Tavily API with DuckDuckGo fallback).
- Transparent, sanitized Tool Activity Log without exposing internal reasoning.
"""

from task3_ai_agent.agent import AgentRunResult, AutonomousAgent, ToolInvocationRecord
from task3_ai_agent.tools.base import BaseAgentTool, ToolResult
from task3_ai_agent.tools.calculator import CalculatorTool, safe_calculate
from task3_ai_agent.tools.file_reader import WorkspaceFileReaderTool
from task3_ai_agent.tools.web_search import WebSearchTool

__all__ = [
    "AutonomousAgent",
    "AgentRunResult",
    "ToolInvocationRecord",
    "BaseAgentTool",
    "ToolResult",
    "CalculatorTool",
    "safe_calculate",
    "WorkspaceFileReaderTool",
    "WebSearchTool",
]


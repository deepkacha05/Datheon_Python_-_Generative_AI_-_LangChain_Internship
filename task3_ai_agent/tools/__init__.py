"""Agent Tools package for Task 3."""

from task3_ai_agent.tools.base import (
    BaseAgentTool,
    ToolResult,
    CalculatorInput,
    FileReaderInput,
    WebSearchInput,
)
from task3_ai_agent.tools.calculator import CalculatorTool, safe_calculate
from task3_ai_agent.tools.file_reader import WorkspaceFileReaderTool
from task3_ai_agent.tools.web_search import WebSearchTool

__all__ = [
    "BaseAgentTool",
    "ToolResult",
    "CalculatorInput",
    "FileReaderInput",
    "WebSearchInput",
    "CalculatorTool",
    "safe_calculate",
    "WorkspaceFileReaderTool",
    "WebSearchTool",
]


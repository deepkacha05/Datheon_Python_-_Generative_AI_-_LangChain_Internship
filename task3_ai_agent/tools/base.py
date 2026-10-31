"""Base contracts and standardized result containers for Agent Tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel, Field


@dataclass
class ToolResult:
    """Standardized output container returned by every agent tool."""
    success: bool
    output: str
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


class CalculatorInput(BaseModel):
    """Input schema for the safe arithmetic calculator."""
    expression: str = Field(
        ...,
        description="The mathematical or arithmetic expression to evaluate (e.g. '15000 * (1 + 0.15)', 'sqrt(144) + 25').",
    )


class FileReaderInput(BaseModel):
    """Input schema for reading files from the restricted workspace directory."""
    filename: str = Field(
        ...,
        description="The relative filename inside the workspace directory (e.g. 'sample_financials.txt'). Paths outside the workspace are blocked.",
    )


class WebSearchInput(BaseModel):
    """Input schema for the web search tool."""
    query: str = Field(
        ...,
        description="The search keywords or question to look up on the web (e.g. 'Python latest release date 2026').",
    )
    max_results: int = Field(
        default=3,
        description="Maximum number of search result snippets to retrieve (1 to 5).",
        ge=1,
        le=5,
    )


class BaseAgentTool(ABC):
    """Abstract base class for all agent tools."""

    name: str
    description: str
    args_schema: Type[BaseModel]

    @abstractmethod
    def run(self, **kwargs: Any) -> ToolResult:
        """Executes the tool with validated keyword arguments and returns a ToolResult."""
        pass

    def __call__(self, **kwargs: Any) -> ToolResult:
        # Validate arguments using args_schema
        try:
            validated = self.args_schema.model_validate(kwargs)
            return self.run(**validated.model_dump())
        except Exception as exc:
            return ToolResult(
                success=False,
                output="",
                error_message=f"Input validation error for tool '{self.name}': {str(exc)}",
            )


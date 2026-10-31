"""Workspace File Reader Tool with path traversal and secret access protection.

Security Highlights:
- Enforces a strict sandbox restricted exclusively to a designated workspace directory.
- Resolves canonical file paths using pathlib.Path.resolve() to prevent directory traversal
  attacks (e.g. '../../', '..\\', absolute root paths like '/etc/passwd' or 'C:\\Windows').
- Explicitly blocks access to sensitive filenames, secrets, and environment files (.env, *.pem, *.key, secrets.*).
- Treats file contents as passive data rather than executable instructions.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional
from task3_ai_agent.tools.base import BaseAgentTool, FileReaderInput, ToolResult

# Blacklist of substrings or extensions that must never be read by the agent
BLOCKED_PATTERNS = [
    ".env",
    "secrets",
    ".key",
    ".pem",
    "id_rsa",
    "id_ed25519",
    "credentials",
    "password",
    ".git",
    ".venv",
    "__pycache__",
]

DEFAULT_WORKSPACE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workspace")
)

MAX_READ_CHARACTERS = 50_000


def is_path_safe_and_contained(target: Path, base: Path) -> bool:
    """Verifies that target is strictly contained within base directory."""
    try:
        # resolve() follows symlinks and resolves .. segments
        resolved_target = target.resolve()
        resolved_base = base.resolve()
        return resolved_target.is_relative_to(resolved_base)
    except Exception:
        return False


def contains_blocked_pattern(path_str: str) -> bool:
    """Checks if path or filename contains sensitive patterns."""
    lower = path_str.lower()
    return any(pattern in lower for pattern in BLOCKED_PATTERNS)


class WorkspaceFileReaderTool(BaseAgentTool):
    """Tool that safely reads files from a designated workspace directory."""

    name = "workspace_file_reader"
    description = (
        "Reads the text content of a file located within the designated workspace directory. "
        "Input must be a relative filename (e.g. 'sample_financials.txt'). "
        "Files outside the workspace or sensitive system files (.env) are strictly blocked."
    )
    args_schema = FileReaderInput

    def __init__(self, workspace_dir: Optional[str] = None):
        self.workspace_dir = Path(workspace_dir or DEFAULT_WORKSPACE_DIR).resolve()
        os.makedirs(self.workspace_dir, exist_ok=True)

    def list_available_files(self) -> List[str]:
        """Returns filenames currently available inside the workspace."""
        if not self.workspace_dir.exists():
            return []
        files = []
        for p in self.workspace_dir.glob("*"):
            if p.is_file() and not contains_blocked_pattern(p.name):
                files.append(p.name)
        return sorted(files)

    def run(self, filename: str) -> ToolResult:
        clean_name = filename.strip()
        if not clean_name:
            return ToolResult(
                success=False,
                output="",
                error_message="Filename cannot be empty.",
            )

        # 1. Check for blocked keywords
        if contains_blocked_pattern(clean_name):
            return ToolResult(
                success=False,
                output="",
                error_message=(
                    f"Security Error: Access to '{clean_name}' is forbidden. "
                    "Access to environment files, secrets, and private credentials is strictly blocked."
                ),
            )

        # 2. Path Traversal & Containment Check
        candidate_path = Path(clean_name)
        # If absolute path was passed, check containment immediately
        if candidate_path.is_absolute():
            target_path = candidate_path.resolve()
        else:
            target_path = (self.workspace_dir / candidate_path).resolve()

        if not is_path_safe_and_contained(target_path, self.workspace_dir):
            return ToolResult(
                success=False,
                output="",
                error_message=(
                    "Security Error: Path traversal attempt detected. "
                    f"File '{clean_name}' resolves outside the allowed workspace directory."
                ),
            )

        # 3. Verify Existence
        if not target_path.exists():
            available = self.list_available_files()
            avail_str = ", ".join(f"'{f}'" for f in available) if available else "No files in workspace"
            return ToolResult(
                success=False,
                output="",
                error_message=f"File not found: '{clean_name}'. Available files in workspace: [{avail_str}].",
            )

        if not target_path.is_file():
            return ToolResult(
                success=False,
                output="",
                error_message=f"'{clean_name}' is a directory, not a readable text file.",
            )

        # 4. Safe Read
        try:
            with open(target_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(MAX_READ_CHARACTERS)

            # Summarize metadata
            char_count = len(content)
            return ToolResult(
                success=True,
                output=content,
                metadata={
                    "filename": target_path.name,
                    "character_count": char_count,
                    "truncated": char_count >= MAX_READ_CHARACTERS,
                },
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                output="",
                error_message=f"Failed to read file '{clean_name}': {str(exc)}",
            )


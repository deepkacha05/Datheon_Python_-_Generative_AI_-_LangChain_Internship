"""Conversation Memory Manager for Task 4 Chatbot.

Features:
- Structured ChatMessage container tracking roles, timestamps, and metadata.
- Sliding window conversation history to control token usage across extended chats.
- Format converters for Google Gemini, OpenAI, and plaintext prompts.
- Export capabilities to JSON and Markdown.
- Thread-safe serialization for Streamlit session_state persistence.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ChatMessage:
    """Represents a single message in the conversation history."""
    role: str  # "user", "assistant", or "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ChatMessage:
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )


DEFAULT_CHAT_SYSTEM_PROMPT = """You are an intelligent, helpful, and polite AI Assistant designed for Datheon's interactive full-stack suite.
You assist users with general questions, programming, data analysis, and technical problems.
Provide clear, accurate, well-formatted responses using Markdown where helpful.
If you are uncertain about a fact, state your uncertainty honestly rather than fabricating information."""


class ConversationMemory:
    """Manages active conversation history with turn-limiting and sliding-window pruning."""

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        max_history_turns: int = 10,
    ):
        """Initializes conversation memory.

        Args:
            system_prompt: Base persona / instructions for the chatbot.
            max_history_turns: Maximum number of recent (user + assistant) exchanges to preserve.
        """
        self.system_prompt = system_prompt or DEFAULT_CHAT_SYSTEM_PROMPT
        self.max_history_turns = max_history_turns
        self.messages: List[ChatMessage] = []

    def add_user_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> ChatMessage:
        """Appends a user message to the active history."""
        msg = ChatMessage(role="user", content=content, metadata=metadata or {})
        self.messages.append(msg)
        return msg

    def add_assistant_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> ChatMessage:
        """Appends an assistant message to the active history."""
        msg = ChatMessage(role="assistant", content=content, metadata=metadata or {})
        self.messages.append(msg)
        return msg

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> ChatMessage:
        """Generic append for any role."""
        msg = ChatMessage(role=role, content=content, metadata=metadata or {})
        self.messages.append(msg)
        return msg

    def clear(self) -> None:
        """Wipes all conversation messages while preserving the system prompt configuration."""
        self.messages.clear()

    def get_messages(self, include_system: bool = False) -> List[ChatMessage]:
        """Returns the full chronological message list."""
        if include_system:
            sys_msg = ChatMessage(role="system", content=self.system_prompt)
            return [sys_msg] + list(self.messages)
        return list(self.messages)

    def get_context_window(self, max_turns: Optional[int] = None) -> List[ChatMessage]:
        """Returns the most recent messages constrained by the turn budget (sliding window).

        A turn represents 1 user message + 1 assistant message.
        """
        turns_limit = max_turns if max_turns is not None else self.max_history_turns
        # 1 turn ~= 2 messages
        max_msg_count = turns_limit * 2
        if len(self.messages) <= max_msg_count:
            return list(self.messages)
        return list(self.messages[-max_msg_count:])

    def format_for_openai(self, max_turns: Optional[int] = None) -> List[Dict[str, str]]:
        """Formats the context window for OpenAI's Chat Completions API."""
        formatted: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        window = self.get_context_window(max_turns=max_turns)
        for msg in window:
            formatted.append({"role": msg.role, "content": msg.content})
        return formatted

    def format_for_gemini(self, max_turns: Optional[int] = None) -> List[Dict[str, Any]]:
        """Formats the context window for Google Gemini's contents structure."""
        formatted: List[Dict[str, Any]] = []
        window = self.get_context_window(max_turns=max_turns)
        for msg in window:
            gemini_role = "user" if msg.role == "user" else "model"
            formatted.append({
                "role": gemini_role,
                "parts": [{"text": msg.content}],
            })
        return formatted

    def format_as_plaintext(self, max_turns: Optional[int] = None) -> str:
        """Formats the context window as structured plain text with speaker labels."""
        lines = [f"System: {self.system_prompt}\n"]
        window = self.get_context_window(max_turns=max_turns)
        for msg in window:
            speaker = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{speaker}: {msg.content}")
        return "\n".join(lines)

    def export_json(self) -> str:
        """Exports the entire conversation history to a JSON string."""
        return json.dumps(
            {
                "system_prompt": self.system_prompt,
                "total_messages": len(self.messages),
                "messages": [m.to_dict() for m in self.messages],
            },
            indent=2,
        )

    def export_markdown(self) -> str:
        """Exports the conversation history to formatted Markdown."""
        lines = [
            "# Conversation Transcript",
            f"*Exported at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            f"*Total Messages: {len(self.messages)}*",
            "---",
            "",
        ]
        for idx, msg in enumerate(self.messages, 1):
            time_str = datetime.fromtimestamp(msg.timestamp).strftime("%H:%M:%S")
            role_title = "👤 User" if msg.role == "user" else "🤖 Assistant"
            lines.append(f"### {idx}. {role_title} ({time_str})")
            lines.append(msg.content)
            if msg.metadata:
                mode = msg.metadata.get("mode")
                if mode:
                    lines.append(f"\n*Mode: `{mode}`*")
            lines.append("\n---\n")
        return "\n".join(lines)

    @classmethod
    def from_dict_list(cls, messages_data: List[Dict[str, Any]], system_prompt: Optional[str] = None) -> ConversationMemory:
        """Reconstructs a memory instance from a list of dicts (e.g. from session_state)."""
        mem = cls(system_prompt=system_prompt)
        for item in messages_data:
            mem.messages.append(ChatMessage.from_dict(item))
        return mem

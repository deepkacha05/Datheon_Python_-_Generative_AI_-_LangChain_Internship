"""Unified Chatbot Backend Orchestrator for Task 4.

Supports 3 integrated operating modes:
1. "general": General Conversational Assistant with live token streaming and sliding-window memory.
2. "rag": Document Q&A Assistant (integrating Task 2 RAGPipeline with physical page citations).
3. "agent": Multi-Tool Autonomous Agent (integrating Task 3 AutonomousAgent with AST calculator,
   sandboxed file reader, and web search activity trace).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple

from dotenv import load_dotenv

from task4_chatbot.memory import ChatMessage, ConversationMemory
from task4_chatbot.streaming import StreamChunk, stream_chat_response

load_dotenv()


class ChatMode(str, Enum):
    """Active operating mode for the chatbot."""
    GENERAL = "general"
    RAG = "rag"
    AGENT = "agent"


@dataclass
class ChatTurnResult:
    """Container for the output of a single user turn."""
    mode: str
    user_prompt: str
    assistant_response: str
    stream_generator: Optional[Iterator[StreamChunk]] = None
    citations: List[Dict[str, Any]] = field(default_factory=list)
    activity_log: List[Dict[str, Any]] = field(default_factory=list)
    was_streamed: bool = False
    provider: str = "mock"
    model: str = "mock"
    execution_time_seconds: float = 0.0
    error_message: Optional[str] = None


class ChatbotBackend:
    """Manages multi-mode turn processing, conversation memory, and backend integrations."""

    def __init__(
        self,
        memory: Optional[ConversationMemory] = None,
        max_history_turns: int = 10,
        system_prompt: Optional[str] = None,
    ):
        self.memory = memory or ConversationMemory(
            system_prompt=system_prompt,
            max_history_turns=max_history_turns,
        )
        self._rag_pipeline = None
        self._autonomous_agent = None

    def get_or_create_rag_pipeline(self, provider: str, api_key: Optional[str], model_name: Optional[str]):
        """Lazily creates or returns the Task 2 RAG pipeline."""
        if self._rag_pipeline is None:
            try:
                from task2_rag_qa.rag_pipeline import RAGPipeline
                self._rag_pipeline = RAGPipeline(
                    provider=provider,
                    api_key=api_key,
                    llm_model=model_name,
                )
            except Exception as exc:
                self._rag_pipeline = None
                raise RuntimeError(f"Failed to initialize Task 2 RAG pipeline: {str(exc)}") from exc
        return self._rag_pipeline

    def get_or_create_agent(self, provider: str, api_key: Optional[str], model_name: Optional[str]):
        """Lazily creates or returns the Task 3 Autonomous Agent."""
        if self._autonomous_agent is None:
            try:
                from task3_ai_agent.agent import AutonomousAgent
                self._autonomous_agent = AutonomousAgent(
                    provider=provider,
                    api_key=api_key,
                    model_name=model_name,
                )
            except Exception as exc:
                self._autonomous_agent = None
                raise RuntimeError(f"Failed to initialize Task 3 Autonomous Agent: {str(exc)}") from exc
        return self._autonomous_agent

    def process_turn(
        self,
        user_input: str,
        mode: str = "general",
        provider: str = "gemini",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        mock_response: Optional[str] = None,
        mock_delay: float = 0.0,
    ) -> ChatTurnResult:
        """Processes a single conversational turn across the selected operating mode."""
        clean_input = user_input.strip()
        start_time = time.time()

        if not clean_input:
            return ChatTurnResult(
                mode=mode,
                user_prompt=user_input,
                assistant_response="Please enter a non-empty message.",
                error_message="Empty user input.",
                execution_time_seconds=0.0,
            )

        # 1. Append user message to conversation memory
        self.memory.add_user_message(clean_input, metadata={"mode": mode})

        # =================================================================
        # MODE 1: General Conversational Chat (Streaming)
        # =================================================================
        if mode == ChatMode.GENERAL.value or mode == "general":
            # Format prompt with sliding window history
            prompt_text = self.memory.format_as_plaintext()
            formatted_msgs = self.memory.format_for_openai()

            stream_gen = stream_chat_response(
                prompt=prompt_text,
                system_prompt=self.memory.system_prompt,
                provider=provider,
                api_key=api_key,
                model_name=model_name,
                temperature=temperature,
                mock_response=mock_response,
                mock_delay=mock_delay,
                formatted_messages=formatted_msgs,
            )

            return ChatTurnResult(
                mode=mode,
                user_prompt=clean_input,
                assistant_response="",  # Populated after streaming consumption
                stream_generator=stream_gen,
                was_streamed=True,
                provider=provider,
                model=model_name or "default",
                execution_time_seconds=time.time() - start_time,
            )

        # =================================================================
        # MODE 2: Document RAG Assistant (Task 2 Integration)
        # =================================================================
        elif mode == ChatMode.RAG.value or mode == "rag":
            try:
                rag = self.get_or_create_rag_pipeline(provider=provider, api_key=api_key, model_name=model_name)
                rag_res = rag.query(
                    question=clean_input,
                    temperature=temperature,
                    mock_llm_response=mock_response,
                )
                answer = rag_res.answer
                citations = [c.to_dict() for c in rag_res.citations]

                # Append assistant response with citations metadata
                self.memory.add_assistant_message(
                    answer,
                    metadata={"mode": "rag", "citations": citations, "is_grounded": rag_res.is_grounded},
                )

                return ChatTurnResult(
                    mode=mode,
                    user_prompt=clean_input,
                    assistant_response=answer,
                    citations=citations,
                    was_streamed=False,
                    provider=provider,
                    model=model_name or rag_res.model_used,
                    execution_time_seconds=time.time() - start_time,
                )
            except Exception as exc:
                err_msg = f"Document Q&A error: {str(exc)}"
                self.memory.add_assistant_message(err_msg, metadata={"mode": "rag", "status": "error"})
                return ChatTurnResult(
                    mode=mode,
                    user_prompt=clean_input,
                    assistant_response=err_msg,
                    error_message=str(exc),
                    execution_time_seconds=time.time() - start_time,
                )

        # =================================================================
        # MODE 3: Multi-Tool Autonomous Agent (Task 3 Integration)
        # =================================================================
        elif mode == ChatMode.AGENT.value or mode == "agent":
            try:
                agent = self.get_or_create_agent(provider=provider, api_key=api_key, model_name=model_name)
                mock_turns = [mock_response] if mock_response else None
                agent_res = agent.run(task=clean_input, mock_llm_responses=mock_turns)

                answer = agent_res.final_answer
                activity_log = [r.to_dict() for r in agent_res.activity_log]

                self.memory.add_assistant_message(
                    answer,
                    metadata={"mode": "agent", "activity_log": activity_log, "iterations": agent_res.iterations_used},
                )

                return ChatTurnResult(
                    mode=mode,
                    user_prompt=clean_input,
                    assistant_response=answer,
                    activity_log=activity_log,
                    was_streamed=False,
                    provider=provider,
                    model=model_name or agent_res.model_used,
                    execution_time_seconds=time.time() - start_time,
                )
            except Exception as exc:
                err_msg = f"Autonomous Agent error: {str(exc)}"
                self.memory.add_assistant_message(err_msg, metadata={"mode": "agent", "status": "error"})
                return ChatTurnResult(
                    mode=mode,
                    user_prompt=clean_input,
                    assistant_response=err_msg,
                    error_message=str(exc),
                    execution_time_seconds=time.time() - start_time,
                )

        else:
            raise ValueError(f"Unknown chat mode: '{mode}'")

    def record_completed_stream(self, assistant_text: str, metadata: Optional[Dict[str, Any]] = None):
        """Appends the completed streamed response to conversation memory."""
        meta = metadata or {"mode": "general"}
        self.memory.add_assistant_message(assistant_text, metadata=meta)

    def clear(self):
        """Wipes active session history."""
        self.memory.clear()

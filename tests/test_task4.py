"""Unit and integration tests for Task 4: Interactive Full-Stack AI Chatbot.

Validates:
- ChatMessage creation, metadata, and serialization.
- ConversationMemory appending, clearing, and sliding-window pruning.
- Prompt formatting for Gemini, OpenAI, and plaintext.
- JSON and Markdown transcript export.
- Streaming generator token delivery, termination, and non-streaming fallback transparency.
- Backend orchestrator turn processing across General, RAG, and Agent modes.
- Error handling for empty queries, missing API keys, and provider failures.
- Zero paid API calls required (runs 100% offline).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import pytest

from task4_chatbot.backend import ChatbotBackend, ChatMode, ChatTurnResult
from task4_chatbot.memory import ChatMessage, ConversationMemory
from task4_chatbot.streaming import (
    StreamChunk,
    stream_chat_response,
    stream_gemini_tokens,
    stream_mock_tokens,
    stream_openai_tokens,
    text_stream_adapter,
)


# =====================================================================
# 1. ChatMessage & Memory Tests
# =====================================================================

def test_chat_message_creation_and_serialization():
    """Verify message creation, attributes, and round-trip dictionary serialization."""
    msg = ChatMessage(
        role="user",
        content="What is Python?",
        metadata={"mode": "general", "source": "test"},
    )
    assert msg.role == "user"
    assert msg.content == "What is Python?"
    assert msg.metadata["mode"] == "general"

    d = msg.to_dict()
    assert d["role"] == "user"
    assert d["content"] == "What is Python?"
    assert "datetime" in d

    restored = ChatMessage.from_dict(d)
    assert restored.role == msg.role
    assert restored.content == msg.content
    assert restored.metadata == msg.metadata


def test_conversation_memory_append_and_clear():
    """Verify user/assistant message appending and clear functionality."""
    mem = ConversationMemory(system_prompt="Custom system persona.")
    assert len(mem.messages) == 0

    mem.add_user_message("Hello AI")
    mem.add_assistant_message("Hello User!")

    assert len(mem.messages) == 2
    assert mem.messages[0].role == "user"
    assert mem.messages[0].content == "Hello AI"
    assert mem.messages[1].role == "assistant"
    assert mem.messages[1].content == "Hello User!"

    # Clear chat
    mem.clear()
    assert len(mem.messages) == 0
    assert mem.system_prompt == "Custom system persona."


def test_conversation_memory_sliding_window():
    """Verify sliding window prunes older messages while keeping recent turn budget."""
    mem = ConversationMemory(max_history_turns=2)  # Budget: 2 turns = 4 messages

    # Add 4 turns (8 messages)
    for i in range(1, 5):
        mem.add_user_message(f"User Query {i}")
        mem.add_assistant_message(f"Assistant Answer {i}")

    assert len(mem.messages) == 8

    # Context window should return only the last 4 messages (Turns 3 and 4)
    window = mem.get_context_window()
    assert len(window) == 4
    assert window[0].content == "User Query 3"
    assert window[1].content == "Assistant Answer 3"
    assert window[2].content == "User Query 4"
    assert window[3].content == "Assistant Answer 4"


def test_conversation_memory_formatters():
    """Verify formatters for OpenAI, Gemini, and plaintext."""
    mem = ConversationMemory(system_prompt="You are a helpful coding assistant.")
    mem.add_user_message("Write a hello world function.")
    mem.add_assistant_message("def hello(): print('hello')")

    # 1. Plaintext formatter
    plain = mem.format_as_plaintext()
    assert "System: You are a helpful coding assistant." in plain
    assert "User: Write a hello world function." in plain
    assert "Assistant: def hello():" in plain

    # 2. OpenAI formatter
    oai = mem.format_for_openai()
    assert len(oai) == 3
    assert oai[0]["role"] == "system"
    assert oai[1]["role"] == "user"
    assert oai[2]["role"] == "assistant"

    # 3. Gemini formatter
    gem = mem.format_for_gemini()
    assert len(gem) == 2
    assert gem[0]["role"] == "user"
    assert gem[1]["role"] == "model"


def test_conversation_memory_export():
    """Verify JSON and Markdown export structures."""
    mem = ConversationMemory()
    mem.add_user_message("Query 1")
    mem.add_assistant_message("Answer 1", metadata={"mode": "rag"})

    # JSON export
    json_str = mem.export_json()
    data = json.loads(json_str)
    assert data["total_messages"] == 2
    assert len(data["messages"]) == 2

    # Markdown export
    md_str = mem.export_markdown()
    assert "# Conversation Transcript" in md_str
    assert "👤 User" in md_str
    assert "🤖 Assistant" in md_str
    assert "Mode: `rag`" in md_str


# =====================================================================
# 2. Streaming Generator Tests
# =====================================================================

def test_streaming_mock_generator():
    """Verify mock streaming yields sequential tokens with termination chunk."""
    generator = stream_mock_tokens(
        user_query="Hello",
        mock_response="This is a test response.",
        delay_seconds=0.0,
    )

    chunks = list(generator)
    assert len(chunks) > 1

    # Check non-empty content
    full_text = "".join(c.text for c in chunks)
    assert full_text == "This is a test response."

    # Final chunk must signal is_final=True
    assert chunks[-1].is_final is True
    assert chunks[-1].was_streamed is True


def test_text_stream_adapter():
    """Verify text adapter converts StreamChunks into pure strings for Streamlit."""
    chunks = [
        StreamChunk(text="Hello ", is_final=False),
        StreamChunk(text="World!", is_final=False),
        StreamChunk(text="", is_final=True),
    ]
    adapter = text_stream_adapter(iter(chunks))
    tokens = list(adapter)
    assert tokens == ["Hello ", "World!"]


def test_streaming_gemini_missing_api_key():
    """Verify calling Gemini streaming without API key raises ValueError."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="Gemini API Key missing"):
            list(stream_gemini_tokens(prompt="Hi", system_instruction="Sys", api_key=None))


def test_streaming_openai_missing_api_key():
    """Verify calling OpenAI streaming without API key raises ValueError."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="OpenAI API Key missing"):
            list(stream_openai_tokens(messages=[], api_key=None))


def test_streaming_gemini_mocked_success():
    """Verify Google Gemini streaming chunks are properly parsed and yielded."""
    mock_chunk1 = MagicMock()
    mock_chunk1.text = "Hello "
    mock_chunk2 = MagicMock()
    mock_chunk2.text = "World"

    mock_client = MagicMock()
    mock_client.models.generate_content_stream.return_value = [mock_chunk1, mock_chunk2]

    with patch("google.genai.Client", return_value=mock_client):
        chunks = list(stream_gemini_tokens(
            prompt="Hi",
            system_instruction="Sys",
            api_key="test_key",
            model_name="gemini-2.5-flash-lite",
        ))

        assert len(chunks) == 3  # 2 text chunks + 1 final chunk
        assert chunks[0].text == "Hello "
        assert chunks[1].text == "World"
        assert chunks[2].is_final is True
        assert chunks[0].was_streamed is True


# =====================================================================
# 3. Backend Turn Processing & Mode Integration Tests
# =====================================================================

def test_backend_empty_input_handling():
    """Verify empty or whitespace-only messages are cleanly rejected without memory pollution."""
    backend = ChatbotBackend()
    res = backend.process_turn("   ", mode="general")
    assert res.error_message == "Empty user input."
    assert len(backend.memory.messages) == 0


def test_backend_general_mode_turn():
    """Verify general conversational turn produces streaming generator and updates memory."""
    backend = ChatbotBackend()
    res = backend.process_turn(
        user_input="Hello there",
        mode="general",
        provider="mock",
        mock_response="Greetings! How can I help you?",
    )

    assert res.mode == "general"
    assert res.was_streamed is True
    assert res.stream_generator is not None

    # Simulate UI consuming the stream
    tokens = [chunk.text for chunk in res.stream_generator]
    full_output = "".join(tokens)
    assert full_output == "Greetings! How can I help you?"

    # Record completed stream in memory
    backend.record_completed_stream(full_output)

    assert len(backend.memory.messages) == 2
    assert backend.memory.messages[0].role == "user"
    assert backend.memory.messages[1].role == "assistant"


def test_backend_clear_chat():
    """Verify backend clear wipes conversation memory."""
    backend = ChatbotBackend()
    res = backend.process_turn("Turn 1", mode="general", provider="mock", mock_response="Ans 1")
    tokens = [chunk.text for chunk in res.stream_generator]
    backend.record_completed_stream("".join(tokens))
    assert len(backend.memory.messages) == 2

    backend.clear()
    assert len(backend.memory.messages) == 0


def test_backend_rag_mode_integration():
    """Verify Document Q&A (Task 2) integration handles unindexed stores and indexed queries."""
    backend = ChatbotBackend()

    # 1. Unindexed vector store returns clear user instructions
    res_empty = backend.process_turn(
        user_input="What is the termination period?",
        mode="rag",
        provider="mock",
    )
    assert res_empty.mode == "rag"
    assert "No documents have been indexed yet" in res_empty.assistant_response

    # 2. Index a sample chunk into the RAG pipeline's vector store
    from task2_rag_qa.text_splitter import DocumentChunk
    rag_pipe = backend.get_or_create_rag_pipeline(provider="mock", api_key=None, model_name="mock")
    test_chunk = DocumentChunk(
        chunk_id="contract.pdf::p1::c0",
        text="According to the contract, the termination period is 30 days with written notice.",
        source_filename="contract.pdf",
        page_number=1,
        chunk_index=0,
        total_pages=1,
    )
    rag_pipe.vector_store.add_chunks([test_chunk], embedder=rag_pipe.embedder)

    # 3. Query indexed documents with grounded answer and citations
    mock_rag_response = "Grounded Answer: According to contract.pdf (Page 1), the termination period is 30 days. Citations: [contract.pdf:1]"
    res = backend.process_turn(
        user_input="What is the termination period?",
        mode="rag",
        provider="mock",
        mock_response=mock_rag_response,
    )

    assert res.mode == "rag"
    assert res.was_streamed is False
    assert "30 days" in res.assistant_response
    assert len(res.citations) > 0
    assert res.citations[0]["source_filename"] == "contract.pdf"
    assert res.citations[0]["page_number"] == 1
    # Verify metadata saved into memory
    assert len(backend.memory.messages) == 4
    assert backend.memory.messages[3].metadata["mode"] == "rag"
    assert len(backend.memory.messages[3].metadata["citations"]) > 0


def test_backend_agent_mode_integration():
    """Verify Autonomous Agent (Task 3) integration executes multi-tool inquiries and preserves activity logs."""
    backend = ChatbotBackend()
    mock_agent_response = "Final Answer: The calculated total revenue is $375,000."

    res = backend.process_turn(
        user_input="Calculate the total revenue from Q1 and Q2.",
        mode="agent",
        provider="mock",
        mock_response=mock_agent_response,
    )

    assert res.mode == "agent"
    assert res.was_streamed is False
    assert "$375,000" in res.assistant_response
    assert len(backend.memory.messages) == 2
    assert backend.memory.messages[1].metadata["mode"] == "agent"

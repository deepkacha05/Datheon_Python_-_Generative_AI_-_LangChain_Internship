"""Streaming Engine for Multi-Provider LLM Chatbot (Task 4).

Features:
- Live token streaming via Google Gemini generate_content_stream API.
- Live token streaming via OpenAI chat.completions.create(stream=True) API.
- Offline Mock streaming generator with realistic micro-chunking for zero-API testing.
- Transparent fallback: if streaming fails or is unsupported, delivers response as a single
  chunk while truthfully indicating `was_streamed=False`.
- Compatible with Streamlit st.write_stream and manual custom cursor placeholders.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()


@dataclass
class StreamChunk:
    """Individual streamed token or segment emitted by an LLM."""
    text: str
    is_final: bool = False
    was_streamed: bool = True
    provider: str = "mock"
    model: str = "mock"
    metadata: Dict[str, Any] = field(default_factory=dict)


def stream_gemini_tokens(
    prompt: str,
    system_instruction: str,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.7,
) -> Iterator[StreamChunk]:
    """Streams tokens from Google Gemini using google-genai Client."""
    resolved_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not resolved_key:
        raise ValueError("Gemini API Key missing. Please set GEMINI_API_KEY in .env or provide it in the sidebar.")

    resolved_model = model_name or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash-lite"
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=resolved_key)
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
    )

    candidates = [resolved_model]
    for c in ["gemini-2.5-flash-lite", "gemini-3.5-flash-lite", "gemini-flash-latest", "gemini-3.5-flash", "gemini-3.8-flash"]:
        if c not in candidates:
            candidates.append(c)

    stream_started = False
    last_error = None

    for cand in candidates:
        try:
            response_stream = client.models.generate_content_stream(
                model=cand,
                contents=prompt,
                config=config,
            )
            for chunk in response_stream:
                if chunk.text:
                    stream_started = True
                    yield StreamChunk(
                        text=chunk.text,
                        is_final=False,
                        was_streamed=True,
                        provider="gemini",
                        model=cand,
                    )
            # If we reached this point and streamed tokens, we're done
            if stream_started:
                yield StreamChunk(text="", is_final=True, was_streamed=True, provider="gemini", model=cand)
                return
        except Exception as exc:
            err_msg = str(exc)
            if "API_KEY_INVALID" in err_msg or "403" in err_msg or "PERMISSION_DENIED" in err_msg:
                raise RuntimeError("Gemini API Key is invalid or unauthorized.") from exc
            last_error = exc
            if stream_started:
                # If streaming was interrupted mid-way, report error
                raise RuntimeError(f"Gemini stream interrupted: {str(exc)}") from exc
            continue

    # If streaming failed on all candidates, attempt non-streaming fallback gracefully
    if not stream_started:
        for cand in candidates:
            try:
                res = client.models.generate_content(
                    model=cand,
                    contents=prompt,
                    config=config,
                )
                if res.text:
                    yield StreamChunk(
                        text=res.text.strip(),
                        is_final=True,
                        was_streamed=False,  # Explicitly report non-streamed fallback
                        provider="gemini",
                        model=cand,
                        metadata={"fallback_reason": f"Streaming unavailable: {str(last_error)}"},
                    )
                    return
            except Exception:
                continue

    raise RuntimeError(f"All Gemini candidates failed to respond: {str(last_error)}") from last_error


def stream_openai_tokens(
    messages: List[Dict[str, str]],
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.7,
) -> Iterator[StreamChunk]:
    """Streams tokens from OpenAI Chat Completions API."""
    resolved_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_key:
        raise ValueError("OpenAI API Key missing. Please set OPENAI_API_KEY in .env or provide it in the sidebar.")

    resolved_model = model_name or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
    from openai import OpenAI

    client = OpenAI(api_key=resolved_key)

    try:
        response_stream = client.chat.completions.create(
            model=resolved_model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield StreamChunk(
                    text=chunk.choices[0].delta.content,
                    is_final=False,
                    was_streamed=True,
                    provider="openai",
                    model=resolved_model,
                )
        yield StreamChunk(text="", is_final=True, was_streamed=True, provider="openai", model=resolved_model)
    except Exception as exc:
        # Fallback to non-streaming
        try:
            res = client.chat.completions.create(
                model=resolved_model,
                messages=messages,
                temperature=temperature,
                stream=False,
            )
            content = res.choices[0].message.content or ""
            yield StreamChunk(
                text=content.strip(),
                is_final=True,
                was_streamed=False,
                provider="openai",
                model=resolved_model,
                metadata={"fallback_reason": f"Streaming failed: {str(exc)}"},
            )
        except Exception:
            raise RuntimeError(f"OpenAI completion failed: {str(exc)}") from exc


def stream_mock_tokens(
    user_query: str,
    mock_response: Optional[str] = None,
    delay_seconds: float = 0.02,
) -> Iterator[StreamChunk]:
    """Generates simulated streaming tokens for offline testing without live APIs."""
    if mock_response:
        full_text = mock_response
    else:
        # Generate a smart mock conversational reply
        q_lower = user_query.lower()
        if "hello" in q_lower or "hi" in q_lower:
            full_text = "Hello! I am your Datheon interactive AI Chatbot. How can I assist you with your tasks or questions today?"
        elif "python" in q_lower:
            full_text = "Python is a versatile, high-level programming language known for readability, expansive data science libraries, and powerful AI frameworks like LangChain."
        elif "rag" in q_lower or "document" in q_lower:
            full_text = "In Document Q&A mode, I retrieve grounded context from your indexed PDFs stored in ChromaDB and formulate precise answers with physical page citations."
        elif "agent" in q_lower or "tool" in q_lower:
            full_text = "In Autonomous Tool Agent mode, I coordinate tools like the AST Calculator, Sandboxed File Reader, and Web Search across multi-step plans."
        else:
            full_text = f"I received your inquiry: '{user_query}'. I am ready to help you analyze data, draft code, answer technical questions, or coordinate tools."

    # Split text into small word/punctuation chunks
    import re
    tokens = re.findall(r"\S+|\s+", full_text)
    for token in tokens:
        if delay_seconds > 0:
            time.sleep(delay_seconds)
        yield StreamChunk(
            text=token,
            is_final=False,
            was_streamed=True,
            provider="mock",
            model="mock-chat-model",
        )

    yield StreamChunk(
        text="",
        is_final=True,
        was_streamed=True,
        provider="mock",
        model="mock-chat-model",
    )


def stream_chat_response(
    prompt: str,
    system_prompt: str,
    provider: str = "gemini",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.7,
    mock_response: Optional[str] = None,
    mock_delay: float = 0.0,
    formatted_messages: Optional[List[Dict[str, str]]] = None,
) -> Iterator[StreamChunk]:
    """Unified generator yielding StreamChunks across Gemini, OpenAI, and Mock providers."""
    prov = (provider or "gemini").lower().strip()

    if prov == "mock" or mock_response is not None:
        yield from stream_mock_tokens(
            user_query=prompt,
            mock_response=mock_response,
            delay_seconds=mock_delay,
        )
    elif prov in ("gemini", "google"):
        yield from stream_gemini_tokens(
            prompt=prompt,
            system_instruction=system_prompt,
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
        )
    elif prov == "openai":
        msgs = formatted_messages or [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        yield from stream_openai_tokens(
            messages=msgs,
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
        )
    else:
        raise ValueError(f"Unsupported provider for streaming: '{provider}'")


def text_stream_adapter(chunk_generator: Iterator[StreamChunk]) -> Iterator[str]:
    """Adapts an Iterator[StreamChunk] into a pure Iterator[str] for st.write_stream."""
    for chunk in chunk_generator:
        if chunk.text:
            yield chunk.text

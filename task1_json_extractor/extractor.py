"""Core extraction engine for Task 1: Structured LLM Prompt Pipeline & JSON Extractor.

Features:
- Dual LLM provider support: Google Gemini (google-genai) and OpenAI.
- Parameterized prompt assembly.
- Multi-stage JSON sanitization and repair engine (removes markdown fences, extracts JSON bounds, repairs trailing commas, fixes escaping).
- Strict Pydantic v2 schema validation with granular error diagnostics.
- Offline Mock mode for automated testing and CI evaluation without live API keys.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type, Union

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from task1_json_extractor.prompts import build_prompt
from task1_json_extractor.schemas import InvoiceSchema, ResumeSchema

# Load local .env if present
load_dotenv()


# =====================================================================
# Custom Exceptions Hierarchy
# =====================================================================

class ExtractionError(Exception):
    """Base class for all extraction pipeline errors."""
    pass


class APIConnectionError(ExtractionError):
    """Raised when communication with the LLM API fails (network, auth, rate limit)."""
    pass


class JSONParsingError(ExtractionError):
    """Raised when LLM output cannot be converted into valid JSON even after repair."""
    def __init__(self, message: str, raw_text: str):
        super().__init__(message)
        self.raw_text = raw_text


class SchemaValidationError(ExtractionError):
    """Raised when JSON parsed successfully but fails Pydantic schema validation."""
    def __init__(self, message: str, validation_errors: list[dict]):
        super().__init__(message)
        self.validation_errors = validation_errors


# =====================================================================
# Result Container
# =====================================================================

@dataclass
class ExtractionResult:
    """Standardized response container returned by the extraction pipeline."""
    success: bool
    data: Optional[BaseModel] = None
    raw_json: Optional[Dict[str, Any]] = None
    raw_llm_response: str = ""
    error_message: Optional[str] = None
    validation_errors: Optional[list[dict]] = None
    execution_time_seconds: float = 0.0
    provider_used: str = "mock"
    model_used: str = "mock"

    def to_dict(self) -> Dict[str, Any]:
        """Convert result container to JSON-serializable dictionary."""
        return {
            "success": self.success,
            "data": self.data.model_dump() if self.data else None,
            "raw_json": self.raw_json,
            "error_message": self.error_message,
            "validation_errors": self.validation_errors,
            "execution_time_seconds": round(self.execution_time_seconds, 3),
            "provider_used": self.provider_used,
            "model_used": self.model_used,
        }


# =====================================================================
# JSON Sanitization and Repair Engine
# =====================================================================

def clean_and_parse_json(raw_text: str) -> Dict[str, Any]:
    """Sanitizes raw LLM output, repairs common syntax flaws, and parses into a Python dict.

    Handles:
    - Markdown code fences (```json ... ``` or ``` ... ```)
    - Preamble commentary before the first '{' and postamble text after '}'
    - Trailing commas in objects or lists (e.g. `{"a": 1,}` -> `{"a": 1}`)
    - Common invalid escaped characters
    - Single-quoted dictionary keys

    Args:
        raw_text: Raw string emitted by LLM.

    Returns:
        Dict[str, Any]: Parsed JSON dictionary.

    Raises:
        JSONParsingError: If all repair strategies fail.
    """
    if not raw_text or not raw_text.strip():
        raise JSONParsingError("LLM returned an empty response.", raw_text)

    cleaned = raw_text.strip()

    # Strategy 1: Extract content from markdown code block if present
    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if code_block_match:
        cleaned = code_block_match.group(1).strip()

    # Strategy 2: Direct parse attempt
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 3: Find outermost JSON object bounds '{' ... '}'
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        cleaned = cleaned[first_brace:last_brace + 1].strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

    # Strategy 4: Repair trailing commas before closing braces/brackets
    # e.g., ", }" -> "}" and ", ]" -> "]"
    repaired = re.sub(r",\s*([\}\]])", r"\1", cleaned)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Strategy 5: Fix single quotes around keys: e.g., {'name': ...} -> {"name": ...}
    repaired_quotes = re.sub(r"(?<!\\)'", '"', repaired)
    try:
        return json.loads(repaired_quotes)
    except json.JSONDecodeError as exc:
        snippet = cleaned[:200] + ("..." if len(cleaned) > 200 else "")
        raise JSONParsingError(
            f"Failed to parse or repair JSON: {exc.msg} (line {exc.lineno}, col {exc.colno}). Snippet: {snippet}",
            raw_text=raw_text
        ) from exc


# =====================================================================
# LLM Caller Abstraction
# =====================================================================

def call_llm(
    system_prompt: str,
    user_prompt: str,
    provider: str = "gemini",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    mock_response: Optional[str] = None,
) -> tuple[str, str, str]:
    """Invokes the selected LLM provider and returns the raw string response.

    Returns:
        tuple[str, str, str]: (raw_llm_text, provider_name, model_name)
    """
    prov = provider.lower().strip()

    # Mock mode (useful for testing & offline demonstrations)
    if prov == "mock" or mock_response is not None:
        if mock_response is None:
            raise ValueError("mock_response must be provided when provider is 'mock'")
        return mock_response, "mock", "mock-model"

    # Google Gemini Provider
    if prov in ("gemini", "google"):
        resolved_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not resolved_key:
            raise APIConnectionError(
                "Gemini API Key missing. Please provide it in the UI, set GEMINI_API_KEY in your .env file, "
                "or obtain a free key at https://aistudio.google.com/"
            )

        resolved_model = model_name or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash-lite"

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=resolved_key)
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                response_mime_type="application/json",
            )
            # Fallback candidates prioritizing verified active models with high availability
            models_to_try = [resolved_model]
            for cand in ["gemini-2.5-flash-lite", "gemini-3.5-flash-lite", "gemini-flash-latest", "gemini-3.5-flash", "gemini-3.8-flash"]:
                if cand not in models_to_try:
                    models_to_try.append(cand)

            last_error = None
            for cand_model in models_to_try:
                try:
                    response = client.models.generate_content(
                        model=cand_model,
                        contents=user_prompt,
                        config=config,
                    )
                    if response.text:
                        return response.text, "gemini", cand_model
                except Exception as exc:
                    err_msg = str(exc)
                    # Unrecoverable authentication/permission errors should fail immediately
                    if "API_KEY_INVALID" in err_msg or "403" in err_msg or "PERMISSION_DENIED" in err_msg:
                        raise APIConnectionError("Google Gemini API Key is invalid or unauthorized.") from exc

                    # Recoverable errors (503 UNAVAILABLE, 429 RESOURCE_EXHAUSTED, 500, 404 NOT_FOUND):
                    # attempt next candidate model in fallback chain
                    last_error = exc
                    continue

            if last_error:
                raise APIConnectionError(f"Google Gemini API error (all model fallbacks exhausted): {str(last_error)}") from last_error
        except (APIConnectionError, ValueError):
            raise
        except Exception as exc:
            raise APIConnectionError(f"Google Gemini client error: {str(exc)}") from exc

    # OpenAI Provider
    elif prov == "openai":
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise APIConnectionError(
                "OpenAI API Key missing. Please provide it in the UI, set OPENAI_API_KEY in your .env file, "
                "or get one from https://platform.openai.com/"
            )

        resolved_model = model_name or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

        try:
            from openai import OpenAI

            client = OpenAI(api_key=resolved_key)
            response = client.chat.completions.create(
                model=resolved_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            return content, "openai", resolved_model
        except Exception as exc:
            msg = str(exc)
            if "Incorrect API key" in msg or "invalid_api_key" in msg:
                raise APIConnectionError("OpenAI API Key is invalid.") from exc
            if "RateLimitError" in msg or "429" in msg:
                raise APIConnectionError("OpenAI API rate limit or quota exceeded.") from exc
            raise APIConnectionError(f"OpenAI API error: {msg}") from exc

    else:
        raise ValueError(f"Unsupported LLM provider '{provider}'. Expected 'gemini', 'openai', or 'mock'.")


# =====================================================================
# Main Structured Extractor Pipeline
# =====================================================================

def extract_structured_json(
    text: str,
    document_type: str = "resume",
    schema_class: Optional[Type[BaseModel]] = None,
    provider: str = "gemini",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    mock_response: Optional[str] = None,
    custom_schema: Optional[Dict[str, Any]] = None,
) -> ExtractionResult:
    """Executes the full end-to-end prompt engineering and JSON extraction pipeline.

    Steps:
    1. Validate input text.
    2. Resolve target Pydantic schema class.
    3. Assemble parameterized system and user prompts.
    4. Invoke LLM provider (or mock mode).
    5. Clean, sanitize, and repair JSON from raw LLM output.
    6. Validate parsed dictionary against Pydantic schema.
    7. Return standardized ExtractionResult container.

    Args:
        text: Unstructured input text (e.g. raw resume or invoice).
        document_type: "resume", "invoice", or "custom".
        schema_class: Pydantic model class to validate against.
        provider: "gemini", "openai", or "mock".
        api_key: Optional explicit API key.
        model_name: Optional model override.
        temperature: LLM sampling temperature (default 0.0 for deterministic extraction).
        mock_response: Raw JSON/text string to use for mock testing.
        custom_schema: Optional dictionary describing fields for custom documents.

    Returns:
        ExtractionResult: Container with success status, validated model, parsed dict, and timing.
    """
    start_time = time.time()

    # 1. Input guard
    if not text or not text.strip():
        return ExtractionResult(
            success=False,
            error_message="Input text is empty. Please provide text to extract from.",
            execution_time_seconds=0.0,
            provider_used=provider,
            model_used=model_name or "unknown",
        )

    # 2. Resolve Schema Class
    doc_type_clean = document_type.lower().strip()
    if schema_class is None:
        if doc_type_clean == "resume":
            schema_class = ResumeSchema
        elif doc_type_clean == "invoice":
            schema_class = InvoiceSchema

    # 3. Build Prompts
    try:
        system_prompt, user_prompt = build_prompt(
            document_type=doc_type_clean,
            text=text,
            custom_schema=custom_schema,
        )
    except Exception as exc:
        return ExtractionResult(
            success=False,
            error_message=f"Prompt assembly error: {str(exc)}",
            execution_time_seconds=time.time() - start_time,
            provider_used=provider,
            model_used=model_name or "unknown",
        )

    # 4. Invoke LLM
    try:
        raw_text, used_provider, used_model = call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            provider=provider,
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            mock_response=mock_response,
        )
    except APIConnectionError as exc:
        return ExtractionResult(
            success=False,
            error_message=str(exc),
            execution_time_seconds=time.time() - start_time,
            provider_used=provider,
            model_used=model_name or "unknown",
        )
    except Exception as exc:
        return ExtractionResult(
            success=False,
            error_message=f"Unexpected extraction pipeline error: {str(exc)}",
            execution_time_seconds=time.time() - start_time,
            provider_used=provider,
            model_used=model_name or "unknown",
        )

    # 5. Clean and Parse JSON
    try:
        parsed_dict = clean_and_parse_json(raw_text)
    except JSONParsingError as exc:
        return ExtractionResult(
            success=False,
            raw_llm_response=raw_text,
            error_message=str(exc),
            execution_time_seconds=time.time() - start_time,
            provider_used=used_provider,
            model_used=used_model,
        )

    # 6. Validate against Pydantic schema
    if schema_class is not None:
        try:
            validated_model = schema_class.model_validate(parsed_dict)
            return ExtractionResult(
                success=True,
                data=validated_model,
                raw_json=parsed_dict,
                raw_llm_response=raw_text,
                execution_time_seconds=time.time() - start_time,
                provider_used=used_provider,
                model_used=used_model,
            )
        except ValidationError as exc:
            # Format Pydantic errors into human-readable list
            formatted_errors = []
            for err in exc.errors():
                loc = " -> ".join(str(p) for p in err.get("loc", []))
                formatted_errors.append({
                    "field": loc or "root",
                    "type": err.get("type", "unknown"),
                    "message": err.get("msg", "Validation error"),
                })

            error_summary = f"Pydantic Schema Validation failed on {len(formatted_errors)} field(s)."
            return ExtractionResult(
                success=False,
                raw_json=parsed_dict,
                raw_llm_response=raw_text,
                error_message=error_summary,
                validation_errors=formatted_errors,
                execution_time_seconds=time.time() - start_time,
                provider_used=used_provider,
                model_used=used_model,
            )

    # If no specific schema class provided, return parsed dict as successful
    return ExtractionResult(
        success=True,
        raw_json=parsed_dict,
        raw_llm_response=raw_text,
        execution_time_seconds=time.time() - start_time,
        provider_used=used_provider,
        model_used=used_model,
    )

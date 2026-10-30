"""Configurable embedding abstractions for Task 2.

Features:
- Uniform interface (`BaseEmbedder`) supporting batch document embedding and single query embedding.
- Google Gemini provider via official `google-genai` SDK (`gemini-embedding-001`, `gemini-embedding-2`).
- OpenAI provider via official `openai` SDK (`text-embedding-3-small`, `text-embedding-ada-002`).
- Offline Mock provider generating deterministic, normalized vector representations
  for unit tests and zero-API-cost evaluations.
"""

from __future__ import annotations

import hashlib
import math
import os
from abc import ABC, abstractmethod
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()


class BaseEmbedder(ABC):
    """Abstract base class for text embedding models."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider (gemini, openai, mock)."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier string."""
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text chunks."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single search query."""
        pass


# =====================================================================
# 1. Deterministic Mock Embedder (Offline / Test Mode)
# =====================================================================

class MockEmbedder(BaseEmbedder):
    """Generates deterministic, normalized pseudo-embeddings without external API calls.

    Ideal for automated testing, CI pipelines, and offline evaluation.
    Computes a bag-of-words / character-trigram hash projection normalized to unit length,
    preserving cosine similarity properties for identical or overlapping terms.
    """

    def __init__(self, dimension: int = 128, model_name: str = "mock-deterministic-embedder"):
        self._dim = dimension
        self._model_name = model_name

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self._model_name

    def _hash_vector(self, text: str) -> List[float]:
        """Projects text onto a normalized vector of fixed dimension."""
        vec = [0.0] * self._dim
        clean = text.lower().strip()
        if not clean:
            return vec

        words = clean.split()
        for w in words:
            # MD5 hash modulo dimension
            h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
            idx = h % self._dim
            sign = 1.0 if ((h >> 4) % 2 == 0) else -1.0
            vec[idx] += sign

        # Add character bi-grams for partial match resilience
        for i in range(len(clean) - 1):
            bg = clean[i : i + 2]
            h = int(hashlib.md5(bg.encode("utf-8")).hexdigest(), 16)
            idx = h % self._dim
            vec[idx] += 0.3

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [round(v / norm, 6) for v in vec]
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._hash_vector(text)


# =====================================================================
# 2. Google Gemini Embedder
# =====================================================================

class GeminiEmbedder(BaseEmbedder):
    """Embedder using Google Gemini's embedding API (`google-genai`)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API Key missing. Please provide it or set GEMINI_API_KEY in your .env file."
            )
        self._model_name = (
            model_name or os.getenv("GEMINI_EMBEDDING_MODEL") or "gemini-embedding-001"
        )

        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize Google GenAI Client: {str(exc)}") from exc

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model_name

    def _call_embed(self, contents: List[str]) -> List[List[float]]:
        # Fallback candidates prioritizing active Gemini embedding models
        candidates = [self._model_name]
        for c in ["gemini-embedding-001", "gemini-embedding-2", "gemini-embedding-2-preview"]:
            if c not in candidates:
                candidates.append(c)

        last_error = None
        for cand in candidates:
            try:
                # Handle batching (max 50 per batch for stability)
                all_vectors: List[List[float]] = []
                batch_size = 30
                for i in range(0, len(contents), batch_size):
                    batch = contents[i : i + batch_size]
                    res = self.client.models.embed_content(
                        model=cand,
                        contents=batch,
                    )
                    for emb in res.embeddings:
                        all_vectors.append(emb.values)
                self._model_name = cand
                return all_vectors
            except Exception as exc:
                last_error = exc
                err_str = str(exc)
                if "404" in err_str or "NOT_FOUND" in err_str:
                    continue
                # If rate-limited or other non-404 error, try next candidate
                continue

        raise RuntimeError(f"Gemini embedding API call failed: {str(last_error)}") from last_error

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._call_embed(texts)

    def embed_query(self, text: str) -> List[float]:
        res = self._call_embed([text])
        return res[0]


# =====================================================================
# 3. OpenAI Embedder
# =====================================================================

class OpenAIEmbedder(BaseEmbedder):
    """Embedder using OpenAI's Embeddings API (`openai`)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API Key missing. Please provide it or set OPENAI_API_KEY in your .env file."
            )
        self._model_name = (
            model_name or os.getenv("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small"
        )

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize OpenAI Client: {str(exc)}") from exc

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        res = self.client.embeddings.create(
            model=self._model_name,
            input=texts,
        )
        return [item.embedding for item in res.data]

    def embed_query(self, text: str) -> List[float]:
        res = self.client.embeddings.create(
            model=self._model_name,
            input=[text],
        )
        return res.data[0].embedding


# =====================================================================
# Factory Function
# =====================================================================

def get_embedder(
    provider: str = "gemini",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> BaseEmbedder:
    """Factory creating the appropriate BaseEmbedder instance.

    Args:
        provider: 'gemini', 'openai', or 'mock'.
        api_key: Optional API key override.
        model_name: Optional model override.
    """
    prov = (provider or "gemini").lower().strip()
    if prov in ("gemini", "google"):
        return GeminiEmbedder(api_key=api_key, model_name=model_name)
    elif prov == "openai":
        return OpenAIEmbedder(api_key=api_key, model_name=model_name)
    elif prov == "mock":
        return MockEmbedder(model_name=model_name or "mock-deterministic-embedder")
    else:
        raise ValueError(
            f"Unsupported embedding provider: '{provider}'. Supported options: 'gemini', 'openai', 'mock'."
        )

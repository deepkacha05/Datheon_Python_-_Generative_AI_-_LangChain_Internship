"""ChromaDB Vector Store wrapper for Task 2.

Features:
- Encapsulates ChromaDB collection lifecycle (ephemeral in-memory or persistent disk).
- Directly indexes pre-computed embedding vectors and rich document metadata (source filename, page number, chunk ID).
- Performs similarity search returning typed DocumentChunk matches with cosine / L2 distance metrics.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional, Tuple
import chromadb
from chromadb.api import ClientAPI

from task2_rag_qa.embeddings import BaseEmbedder
from task2_rag_qa.text_splitter import DocumentChunk


class ChromaVectorStore:
    """Manages document storage, embedding indexing, and similarity retrieval via ChromaDB."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
        in_memory: bool = True,
    ):
        """Initializes the ChromaDB client and collection.

        Args:
            collection_name: Unique collection identifier. If None, generates a unique session collection.
            persist_directory: Directory path to persist vector data if in_memory=False.
            in_memory: If True, uses ChromaDB EphemeralClient (ideal for Streamlit session resets and testing).
        """
        self.collection_name = collection_name or f"datheon_rag_{uuid.uuid4().hex[:8]}"
        self.in_memory = in_memory
        self.persist_directory = persist_directory

        if in_memory or persist_directory is None:
            self.client: ClientAPI = chromadb.EphemeralClient()
        else:
            os.makedirs(persist_directory, exist_ok=True)
            self.client: ClientAPI = chromadb.PersistentClient(path=persist_directory)

        # Create or fetch collection with cosine distance space
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embedder: BaseEmbedder,
    ) -> int:
        """Embeds and indexes a list of DocumentChunks into ChromaDB.

        Args:
            chunks: List of DocumentChunk instances.
            embedder: BaseEmbedder instance for generating vector embeddings.

        Returns:
            Number of successfully indexed chunks.
        """
        if not chunks:
            return 0

        texts = [chunk.text for chunk in chunks]
        embeddings = embedder.embed_documents(texts)

        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [chunk.to_metadata() for chunk in chunks]

        # Chroma requires non-empty lists
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        return len(chunks)

    def similarity_search(
        self,
        query: str,
        embedder: BaseEmbedder,
        top_k: int = 4,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Searches for the most semantically relevant chunks for a user query.

        Args:
            query: The user's question or search phrase.
            embedder: BaseEmbedder instance for encoding the query.
            top_k: Maximum number of relevant chunks to retrieve.

        Returns:
            List of tuples: (DocumentChunk, distance_score) ordered by relevance (lowest distance first).
        """
        if self.count() == 0 or not query.strip():
            return []

        query_embedding = embedder.embed_query(query)
        actual_k = min(top_k, self.count())
        if actual_k <= 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_k,
            include=["documents", "metadatas", "distances"],
        )

        matches: List[Tuple[DocumentChunk, float]] = []

        ids_list = results.get("ids", [[]])[0]
        docs_list = results.get("documents", [[]])[0]
        meta_list = results.get("metadatas", [[]])[0]
        dist_list = results.get("distances", [[]])[0] if results.get("distances") else [0.0] * len(ids_list)

        for chunk_id, doc_text, meta, dist in zip(ids_list, docs_list, meta_list, dist_list):
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                text=doc_text,
                source_filename=meta.get("source", "document.pdf"),
                page_number=int(meta.get("page", 1)),
                chunk_index=int(meta.get("chunk_index", 0)),
                total_pages=int(meta.get("total_pages", 1)),
            )
            matches.append((chunk, float(dist)))

        return matches

    def count(self) -> int:
        """Returns the total number of indexed chunks."""
        return self.collection.count()

    def clear(self) -> None:
        """Deletes all indexed documents from the collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def get_indexed_sources(self) -> List[Dict[str, Any]]:
        """Returns summary metadata for all currently indexed documents."""
        count = self.count()
        if count == 0:
            return []

        data = self.collection.get(include=["metadatas"])
        metadatas = data.get("metadatas") or []

        sources_summary: Dict[str, Dict[str, Any]] = {}
        for m in metadatas:
            src = m.get("source", "Unknown")
            page = m.get("page", 1)
            if src not in sources_summary:
                sources_summary[src] = {
                    "source": src,
                    "pages": set(),
                    "chunk_count": 0,
                    "total_pages": m.get("total_pages", 1),
                }
            sources_summary[src]["pages"].add(page)
            sources_summary[src]["chunk_count"] += 1

        result = []
        for src, info in sources_summary.items():
            result.append({
                "source": src,
                "indexed_pages": sorted(list(info["pages"])),
                "total_chunks": info["chunk_count"],
                "total_pages": info["total_pages"],
            })
        return result

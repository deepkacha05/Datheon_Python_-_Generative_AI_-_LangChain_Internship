"""Interactive Streamlit Application and CLI runner for Task 2:
Document Q&A Assistant with RAG & Vector Embeddings.

Features:
- Multi-provider support: Google Gemini, OpenAI, and zero-API-key Offline Mock Mode.
- PDF upload (single or multiple) with page metadata preservation & scanned document detection.
- 1-Click "Load Sample Policy PDF" button for instant demonstration.
- Vector indexing using ChromaDB with cosine distance search.
- Strict context-grounded answer generation with physical page-level source citations.
- Explicit out-of-scope handling: clearly states when facts are not found in the documents.
- Detailed telemetry: latency, provider, model, distance scores, and collection inspector.
"""

from __future__ import annotations

import io
import os
import sys
import time
from typing import List, Optional
import streamlit as st
from dotenv import load_dotenv

# Ensure repo root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from task2_rag_qa.embeddings import BaseEmbedder, get_embedder, MockEmbedder
from task2_rag_qa.pdf_loader import PDFExtractionResult, extract_pdf_pages
from task2_rag_qa.rag_pipeline import NOT_FOUND_STANDARD_RESPONSE, RAGPipeline, RAGResponse
from task2_rag_qa.sample_documents.generator import generate_sample_policy_pdf
from task2_rag_qa.text_splitter import chunk_document_pages
from task2_rag_qa.vector_store import ChromaVectorStore

load_dotenv()


# =====================================================================
# Session State Initialization
# =====================================================================

def init_session_state():
    """Initializes persistent Streamlit session state objects."""
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = ChromaVectorStore(
            collection_name="streamlit_user_rag_session",
            in_memory=True,
        )
    if "indexed_docs_history" not in st.session_state:
        st.session_state.indexed_docs_history = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_question" not in st.session_state:
        st.session_state.current_question = ""


# =====================================================================
# UI Presentation Helpers
# =====================================================================

def render_citations(citations):
    """Renders visual cards for source citations."""
    if not citations:
        return

    st.markdown("##### 📌 Verified Source Citations:")
    cols = st.columns(min(len(citations), 3))
    for idx, cite in enumerate(citations):
        with cols[idx % len(cols)]:
            with st.container(border=True):
                st.markdown(f"📄 **{cite.source_filename}**")
                st.caption(f"📍 **Page {cite.page_number}** (Chunk ID: `{cite.chunk_id}`)")
                st.markdown(f"> *\"{cite.snippet}\"*")


def render_response_card(resp: RAGResponse):
    """Renders the grounded answer card and metadata."""
    if resp.is_not_found:
        st.warning(
            f"🔍 **Grounded Verification Result:**\n\n"
            f"> {resp.answer}\n\n"
            f"*Note: The RAG engine confirmed this query cannot be answered truthfully from the uploaded document context.*"
        )
    else:
        st.success(
            f"✅ **Answer (Grounded & Cited)** | Provider: `{resp.provider_used}` | Model: `{resp.model_used}` | Latency: `{resp.execution_time_seconds:.3f}s`"
        )
        st.markdown(f"### 💡 Answer:\n{resp.answer}")
        render_citations(resp.citations)


# =====================================================================
# Main Streamlit Application
# =====================================================================

def main():
    st.set_page_config(
        page_title="Task 2: Document Q&A Assistant with RAG & Vector Embeddings",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()

    # Header
    st.title("📚 Task 2: Document Q&A Assistant with RAG & Vector Embeddings")
    st.markdown(
        "Upload PDF documents, extract text with page-level provenance, index semantic chunks into "
        "**ChromaDB**, and ask questions answered strictly with **grounded citations** and zero hallucinations."
    )

    # Sidebar: Provider & Model Configuration
    st.sidebar.header("⚙️ RAG Configuration")

    provider_choice = st.sidebar.selectbox(
        "LLM & Embedding Provider",
        options=["Google Gemini", "OpenAI", "Offline Mock Mode (Zero API Key)"],
        index=0,
    )

    api_key_input = ""
    llm_model = ""
    embedding_model = ""

    if provider_choice == "Google Gemini":
        provider_code = "gemini"
        gemini_model_options = [
            "gemini-2.5-flash-lite",
            "gemini-3.5-flash-lite",
            "gemini-flash-latest",
            "gemini-3.5-flash",
            "gemini-3.8-flash",
        ]
        llm_model = st.sidebar.selectbox("Gemini LLM Model", options=gemini_model_options, index=0)
        embedding_options = ["gemini-embedding-001", "gemini-embedding-2"]
        embedding_model = st.sidebar.selectbox("Gemini Embedding Model", options=embedding_options, index=0)
        api_key_env = os.getenv("GEMINI_API_KEY", "")
        api_key_input = st.sidebar.text_input(
            "Gemini API Key",
            value=api_key_env,
            type="password",
            help="Loaded from .env if available, or enter your Google AI Studio key.",
        )
    elif provider_choice == "OpenAI":
        provider_code = "openai"
        llm_model = st.sidebar.text_input("OpenAI Model Name", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        embedding_model = st.sidebar.text_input("OpenAI Embedding Model", value="text-embedding-3-small")
        api_key_env = os.getenv("OPENAI_API_KEY", "")
        api_key_input = st.sidebar.text_input(
            "OpenAI API Key",
            value=api_key_env,
            type="password",
            help="Loaded from .env if available, or enter your platform.openai.com key.",
        )
    else:
        provider_code = "mock"
        llm_model = "mock-evaluator"
        embedding_model = "mock-deterministic-embedder"
        st.sidebar.info("💡 **Mock Mode Active:** Uses deterministic vector projections & offline mock responses for fast zero-cost evaluation.")

    # Chunking Hyperparameters
    st.sidebar.subheader("📐 Chunking & Retrieval Parameters")
    chunk_size = st.sidebar.slider("Chunk Size (Characters)", min_value=200, max_value=1200, value=600, step=50)
    chunk_overlap = st.sidebar.slider("Chunk Overlap (Characters)", min_value=20, max_value=300, value=120, step=20)
    top_k = st.sidebar.slider("Top-K Chunks to Retrieve", min_value=1, max_value=8, value=3, step=1)
    temperature = st.sidebar.slider("Generation Temperature", min_value=0.0, max_value=0.7, value=0.0, step=0.05)

    st.sidebar.divider()

    # Index Management & Reset
    st.sidebar.subheader("🗂️ Vector Store Status")
    indexed_count = st.session_state.vector_store.count()
    st.sidebar.metric("Indexed Chunks", f"{indexed_count}")
    if st.sidebar.button("🗑️ Clear Vector Index", use_container_width=True):
        st.session_state.vector_store.clear()
        st.session_state.indexed_docs_history = []
        st.session_state.chat_history = []
        st.sidebar.success("ChromaDB index cleared successfully!")
        st.rerun()

    # Layout: Two main tabs (1: Ingestion & Q&A, 2: Index Inspector & Telemetry)
    tab_qa, tab_inspector, tab_help = st.tabs([
        "💬 Document Q&A",
        "🔍 Vector Store Inspector",
        "💡 Architecture & Mentor Guide",
    ])

    with tab_qa:
        # Document Upload Section
        st.markdown("### 1. Ingest Documents")
        col_upload, col_sample = st.columns([3, 1])

        with col_sample:
            st.markdown("**Quick Preset:**")
            if st.button("📄 Load Sample Policy PDF", use_container_width=True):
                sample_pdf_path = "task2_rag_qa/sample_documents/Datheon_Company_Policy_and_AI_Guidelines.pdf"
                if not os.path.exists(sample_pdf_path):
                    sample_pdf_path = generate_sample_policy_pdf()

                with open(sample_pdf_path, "rb") as f:
                    sample_bytes = f.read()

                with st.spinner("Extracting text, chunking, and embedding sample policy into ChromaDB..."):
                    try:
                        embedder = get_embedder(provider=provider_code, api_key=api_key_input, model_name=embedding_model)
                        pipeline = RAGPipeline(
                            provider=provider_code,
                            api_key=api_key_input,
                            llm_model=llm_model,
                            embedding_model=embedding_model,
                            vector_store=st.session_state.vector_store,
                            embedder=embedder,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                        )
                        res = pipeline.index_pdf(sample_bytes, filename="Datheon_Company_Policy_and_AI_Guidelines.pdf")
                        if res.success:
                            st.session_state.indexed_docs_history.append({
                                "filename": "Datheon_Company_Policy_and_AI_Guidelines.pdf",
                                "pages": res.total_pages,
                                "characters": res.total_characters,
                                "chunks": len(chunk_document_pages(res.pages, chunk_size, chunk_overlap)),
                            })
                            st.toast("✅ Sample Policy PDF indexed into ChromaDB!", icon="🎉")
                            st.rerun()
                    except Exception as exc:
                        st.error(f"Error indexing sample PDF: {str(exc)}")

        with col_upload:
            uploaded_files = st.file_uploader(
                "Upload PDF Document(s):",
                type=["pdf"],
                accept_multiple_files=True,
                help="Upload one or multiple PDF documents. Digital text and page metadata will be extracted.",
            )

            if uploaded_files:
                for uf in uploaded_files:
                    # Check if already indexed in this session
                    already_indexed = any(d["filename"] == uf.name for d in st.session_state.indexed_docs_history)
                    if not already_indexed:
                        with st.spinner(f"Ingesting & vectorizing '{uf.name}'..."):
                            try:
                                embedder = get_embedder(provider=provider_code, api_key=api_key_input, model_name=embedding_model)
                                pipeline = RAGPipeline(
                                    provider=provider_code,
                                    api_key=api_key_input,
                                    llm_model=llm_model,
                                    embedding_model=embedding_model,
                                    vector_store=st.session_state.vector_store,
                                    embedder=embedder,
                                    chunk_size=chunk_size,
                                    chunk_overlap=chunk_overlap,
                                )
                                res: PDFExtractionResult = pipeline.index_pdf(uf, filename=uf.name)

                                if res.success:
                                    if res.is_scanned:
                                        st.warning(res.warning_message)
                                    else:
                                        chunks_added = len(chunk_document_pages(res.pages, chunk_size, chunk_overlap))
                                        st.session_state.indexed_docs_history.append({
                                            "filename": uf.name,
                                            "pages": res.total_pages,
                                            "characters": res.total_characters,
                                            "chunks": chunks_added,
                                        })
                                        st.success(f"Indexed **{uf.name}**: {res.total_pages} page(s), {chunks_added} chunks, {res.total_characters:,} characters.")
                                else:
                                    st.error(f"Failed to process '{uf.name}': {res.error_message}")
                            except Exception as exc:
                                st.error(f"Ingestion error for '{uf.name}': {str(exc)}")

        st.divider()

        # Ingested documents status banner
        if st.session_state.indexed_docs_history:
            st.markdown("#### 📁 Active Document Index")
            doc_cols = st.columns(len(st.session_state.indexed_docs_history))
            for i, doc in enumerate(st.session_state.indexed_docs_history):
                with doc_cols[i % len(doc_cols)]:
                    st.info(f"📄 **{doc['filename']}**\n- Pages: `{doc['pages']}` | Chunks: `{doc['chunks']}` | Chars: `{doc['characters']:,}`")

        # Question & Query Section
        st.markdown("### 2. Ask Questions")

        # Quick preset questions
        st.caption("Quick sample prompts (Click to test):")
        q_cols = st.columns(4)
        with q_cols[0]:
            if st.button("🕒 Working Hours?", use_container_width=True):
                st.session_state.current_question = "What are the official working hours and core collaboration hours?"
        with q_cols[1]:
            if st.button("🏡 Remote Work Policy?", use_container_width=True):
                st.session_state.current_question = "What is the hybrid remote work policy and how many days can engineers work from home?"
        with q_cols[2]:
            if st.button("✈️ Travel Per Diem?", use_container_width=True):
                st.session_state.current_question = "What is the maximum daily meal per diem for domestic travel?"
        with q_cols[3]:
            if st.button("🚫 Out-of-Scope Test", use_container_width=True):
                st.session_state.current_question = "What is Datheon's quantum computing encryption policy?"

        user_query = st.text_input(
            "Enter your question about the uploaded document(s):",
            value=st.session_state.current_question,
            placeholder="e.g. What is the home office tech equipment stipend amount?",
        )

        col_ask, col_clear = st.columns([4, 1])
        with col_ask:
            btn_ask = st.button("⚡ Query Document via RAG", type="primary", use_container_width=True)
        with col_clear:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.current_question = ""
                st.rerun()

        if btn_ask:
            if not user_query.strip():
                st.warning("Please enter a question.")
            elif st.session_state.vector_store.count() == 0:
                st.warning("Please upload a PDF document or load the sample policy before asking questions.")
            else:
                with st.spinner("Searching ChromaDB vectors and generating grounded response..."):
                    try:
                        embedder = get_embedder(provider=provider_code, api_key=api_key_input, model_name=embedding_model)
                        pipeline = RAGPipeline(
                            provider=provider_code,
                            api_key=api_key_input,
                            llm_model=llm_model,
                            embedding_model=embedding_model,
                            vector_store=st.session_state.vector_store,
                            embedder=embedder,
                        )

                        # Provide mock response if in mock mode
                        mock_ans = None
                        if provider_code == "mock":
                            if "working hour" in user_query.lower() or "collaboration" in user_query.lower():
                                mock_ans = (
                                    "Standard office hours at Datheon are 9:30 AM to 6:30 PM Indian Standard Time (IST), "
                                    "Monday through Friday. Core synchronous team collaboration hours are 11:00 AM to 4:00 PM IST "
                                    "[Source: Datheon_Company_Policy_and_AI_Guidelines.pdf, Page: 1]."
                                )
                            elif "remote" in user_query.lower() or "home" in user_query.lower():
                                mock_ans = (
                                    "Datheon operates on a flexible hybrid model allowing engineers to work remotely up to 3 days per week. "
                                    "Mandatory in-office collaboration days are Tuesday and Thursday [Source: Datheon_Company_Policy_and_AI_Guidelines.pdf, Page: 2]."
                                )
                            elif "travel" in user_query.lower() or "diem" in user_query.lower() or "meal" in user_query.lower():
                                mock_ans = (
                                    "The maximum allowable daily meal per diem for domestic business travel is $75 (or INR 4,500) per day. "
                                    "Itemized tax invoices must be submitted for reimbursement [Source: Datheon_Company_Policy_and_AI_Guidelines.pdf, Page: 2]."
                                )
                            elif "quantum" in user_query.lower():
                                mock_ans = NOT_FOUND_STANDARD_RESPONSE
                            else:
                                mock_ans = f"Based on the indexed document context, the answer is grounded in physical pages. [Source: Datheon_Company_Policy_and_AI_Guidelines.pdf, Page: 1]"

                        response: RAGResponse = pipeline.query(
                            question=user_query,
                            top_k=top_k,
                            temperature=temperature,
                            mock_llm_response=mock_ans,
                        )

                        # Prepend to chat history
                        st.session_state.chat_history.insert(0, response)

                    except Exception as exc:
                        st.error(f"RAG query execution failed: {str(exc)}")

        # Render Q&A History
        if st.session_state.chat_history:
            st.markdown("---")
            st.markdown("### 💬 Conversation & Citation Telemetry")
            for idx, item in enumerate(st.session_state.chat_history):
                with st.expander(f"Q: {item.question}", expanded=(idx == 0)):
                    render_response_card(item)

    with tab_inspector:
        st.markdown("### 🔍 ChromaDB Vector Collection Inspector")
        sources = st.session_state.vector_store.get_indexed_sources()
        if not sources:
            st.info("Vector collection is currently empty. Upload a PDF or click 'Load Sample Policy PDF' to inspect indexed chunks.")
        else:
            st.markdown(f"**Total Indexed Chunks in Collection:** `{st.session_state.vector_store.count()}`")
            st.table(sources)

            st.markdown("#### 📑 Sample Chunk Previews from Vector Store")
            data = st.session_state.vector_store.collection.get(limit=6, include=["documents", "metadatas"])
            if data and data.get("documents"):
                for idx, (doc, meta) in enumerate(zip(data["documents"], data["metadatas"])):
                    with st.container(border=True):
                        st.markdown(f"**Chunk {idx + 1} | Source:** `{meta.get('source')}` | **Page:** `{meta.get('page')}` | **ID:** `{meta.get('chunk_id')}`")
                        st.code(doc, language="text")

    with tab_help:
        st.markdown("### 🏗️ Task 2 System Architecture & Design")
        st.markdown(
            """
            ```mermaid
            flowchart TD
                A[PDF Document Upload] --> B[pypdf Page-by-Page Extraction\npdf_loader.py]
                B -->|Page Text + Page Num Metadata| C{Text Present?}
                C -->|No / Raster Image| D[Scanned PDF Warning\nExplains OCR Requirement]
                C -->|Yes| E[RecursiveCharacterTextSplitter\nLangChain chunking]
                E -->|Chunks with Page Provenance| F[Vector Embeddings\nGoogle Gemini / OpenAI / Mock]
                F --> G[(ChromaDB Vector Store\nCosine Similarity Index)]
                
                H[User Question] --> I[Query Embedding]
                I --> G
                G -->|Top-K Retrieved Chunks| J[Context-Grounded Prompt Builder\nrag_pipeline.py]
                J --> K[LLM Provider\nGemini 2.5 Flash Lite / OpenAI]
                K -->|Strict Response| L{Answer in Context?}
                L -->|Yes| M[Grounded Answer + Citations\nFilename + Page Number]
                L -->|No| N[Explicit Fallback:\n'Information not found in uploaded documents']
            ```
            """
        )
        st.info(
            "💡 **Mentor Troubleshooting Tips (Hinglish):**\n"
            "- **Scanned PDFs vs Digital PDFs:** Agar document scanned image hai (jisme text select nahi hota), to standard PDF reader text extract nahi kar sakta. Tab system alert dega ki **OCR engine** (jaise Tesseract ya Google Vision) ki zaroorat hai.\n"
            "- **Hallucination Prevention:** Agar koi fact PDF me nahi hai, to LLM guess karne ki jagah *'The requested information was not found in the uploaded documents'* bolta hai.\n"
            "- **Zero-API Cost Testing:** Sidebar me **'Offline Mock Mode'** select karein aur 'Load Sample Policy PDF' click karein. Pure features test ho jayenge bina kisi paid key ke."
        )


# =====================================================================
# CLI Runner Entrypoint
# =====================================================================

def run_cli():
    """Runs an automated terminal demonstration using the sample policy PDF and Mock LLM mode."""
    print("=" * 70)
    print("Task 2: Document Q&A Assistant with RAG & Vector Embeddings (CLI Demo)")
    print("=" * 70)

    sample_pdf_path = "task2_rag_qa/sample_documents/Datheon_Company_Policy_and_AI_Guidelines.pdf"
    if not os.path.exists(sample_pdf_path):
        sample_pdf_path = generate_sample_policy_pdf()

    print(f"1. Ingesting PDF: {sample_pdf_path}")
    embedder = MockEmbedder(dimension=128)
    vector_store = ChromaVectorStore(in_memory=True)
    pipeline = RAGPipeline(
        provider="mock",
        vector_store=vector_store,
        embedder=embedder,
    )

    with open(sample_pdf_path, "rb") as f:
        pdf_bytes = f.read()

    res = pipeline.index_pdf(pdf_bytes, filename="Datheon_Company_Policy_and_AI_Guidelines.pdf")
    print(f"   Indexed: {res.total_pages} pages, {vector_store.count()} chunks, {res.total_characters} characters.")

    # Test Query 1: Supported question
    q1 = "What are the official working hours and core collaboration hours?"
    print(f"\n2. Querying: '{q1}'")
    mock_ans_1 = (
        "Official office hours are 9:30 AM to 6:30 PM Indian Standard Time (IST), Monday through Friday. "
        "Core synchronous collaboration hours are 11:00 AM to 4:00 PM IST "
        "[Source: Datheon_Company_Policy_and_AI_Guidelines.pdf, Page: 1]."
    )
    ans1 = pipeline.query(q1, top_k=2, mock_llm_response=mock_ans_1)
    print(f"   Answer: {ans1.answer}")
    print(f"   Citations: {[c.label for c in ans1.citations]}")
    print(f"   Retrieved Chunks: {len(ans1.retrieved_chunks)}")

    # Test Query 2: Out-of-scope question
    q2 = "What is the policy on quantum computing?"
    print(f"\n3. Querying Out-of-scope: '{q2}'")
    ans2 = pipeline.query(q2, top_k=2, mock_llm_response=NOT_FOUND_STANDARD_RESPONSE)
    print(f"   Answer: {ans2.answer}")
    print(f"   Is Not Found: {ans2.is_not_found}")

    print("\n" + "=" * 70)
    print("To launch the Streamlit Web Application, run:")
    print("  streamlit run task2_rag_qa/app.py")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        run_cli()
    else:
        try:
            import streamlit.runtime
            if streamlit.runtime.exists():
                main()
            else:
                run_cli()
        except Exception:
            run_cli()

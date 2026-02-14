"""
LangChain RAG chain.

Builds a Retrieval-Augmented Generation chain that:
1. Takes a user question
2. Retrieves relevant chunks from the FAISS vector store
3. Formats them into the Llama 3.1 Instruct prompt template
4. Sends to the LLM for generation
5. Returns the answer with source metadata
"""

import logging
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_REPEAT_PENALTY,
    DEFAULT_SEARCH_TYPE,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    RAG_PROMPT_TEMPLATE,
)
from app.rag.llm import create_llm_with_params, get_llm
from app.rag.vector_store import get_retriever

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes for structured input/output
# ---------------------------------------------------------------------------


@dataclass
class RAGRequest:
    """Parameters for a single RAG query — maps to Gradio Tab 2 inputs."""

    question: str
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    # Retrieval parameters
    top_k: int = DEFAULT_TOP_K
    search_type: str = DEFAULT_SEARCH_TYPE
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP

    # Generation parameters
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    repeat_penalty: float = DEFAULT_REPEAT_PENALTY


@dataclass
class RAGResponse:
    """Structured response from a RAG query."""

    answer: str
    sources: list[str] = field(default_factory=list)
    num_chunks_retrieved: int = 0
    error: str | None = None


# ---------------------------------------------------------------------------
# Chain construction
# ---------------------------------------------------------------------------


def _format_docs(docs: list[Document]) -> str:
    """Join retrieved document chunks into a single context string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(request: RAGRequest):
    """
    Build a LangChain LCEL chain for a given RAG request.

    The chain follows this flow:
        question → retriever → format docs into context → prompt → LLM → parse

    Args:
        request: RAGRequest with all retrieval and generation parameters.

    Returns:
        A tuple of (chain, retriever) so the caller can also inspect
        retrieved documents separately.
    """
    # Configure LLM with user-specified generation params
    llm = create_llm_with_params(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        repeat_penalty=request.repeat_penalty,
    )

    # Build retriever with user-specified search params
    retriever = get_retriever(
        top_k=request.top_k,
        search_type=request.search_type,
    )

    # Prompt template using Llama 3.1 Instruct format
    prompt = PromptTemplate(
        input_variables=["system_prompt", "context", "question"],
        template=RAG_PROMPT_TEMPLATE,
    )

    # LCEL chain: retrieve → format → prompt → LLM → parse
    chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
            "system_prompt": lambda _: request.system_prompt,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def query(request: RAGRequest) -> RAGResponse:
    """
    Execute a full RAG query: retrieve context, generate answer, return sources.

    This is the main entry point called by the Gradio UI and API routes.

    Args:
        request: RAGRequest with question and all parameters.

    Returns:
        RAGResponse with the answer, source filenames, and chunk count.
    """
    if not request.question.strip():
        return RAGResponse(answer="", error="Question cannot be empty.")

    try:
        chain, retriever = build_rag_chain(request)

        # Retrieve documents separately so we can report sources
        retrieved_docs = retriever.invoke(request.question)
        source_names = list(
            dict.fromkeys(  # Deduplicate while preserving order
                doc.metadata.get("source", "unknown") for doc in retrieved_docs
            )
        )

        logger.info(
            "Retrieved %d chunks from %d source(s) for question: %.80s...",
            len(retrieved_docs),
            len(source_names),
            request.question,
        )

        # Run the chain
        answer = chain.invoke(request.question)

        return RAGResponse(
            answer=answer.strip(),
            sources=source_names,
            num_chunks_retrieved=len(retrieved_docs),
        )

    except RuntimeError as exc:
        # Typically "No FAISS index available"
        logger.error("RAG query failed: %s", exc)
        return RAGResponse(answer="", error=str(exc))

    except Exception as exc:
        logger.exception("Unexpected error during RAG query.")
        return RAGResponse(answer="", error=f"An unexpected error occurred: {exc}")

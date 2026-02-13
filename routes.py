"""
Internal FastAPI routes.

These endpoints are primarily consumed by the Gradio UI but can also
be called programmatically for testing or integration purposes.
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_REPEAT_PENALTY,
    DEFAULT_SEARCH_TYPE,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
)
from app.rag.chain import RAGRequest, query
from app.rag.vector_store import get_index_stats

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["RAG"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """API request body for a RAG query."""

    question: str = Field(..., min_length=1, description="The user's question.")
    system_prompt: str = Field(default=DEFAULT_SYSTEM_PROMPT)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)
    search_type: str = Field(default=DEFAULT_SEARCH_TYPE, pattern="^(similarity|mmr)$")
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=64, le=2048)
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    top_p: float = Field(default=DEFAULT_TOP_P, ge=0.0, le=1.0)
    repeat_penalty: float = Field(default=DEFAULT_REPEAT_PENALTY, ge=1.0, le=2.0)


class QueryResponse(BaseModel):
    """API response body for a RAG query."""

    answer: str
    sources: list[str]
    num_chunks_retrieved: int


class IndexStatsResponse(BaseModel):
    """API response for index health check."""

    index_exists: bool
    num_vectors: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/query", response_model=QueryResponse)
async def api_query(body: QueryRequest) -> QueryResponse:
    """
    Execute a RAG query.

    Accepts a question and optional parameters, retrieves relevant
    document chunks, generates an answer, and returns it with sources.
    """
    request = RAGRequest(
        question=body.question,
        system_prompt=body.system_prompt,
        top_k=body.top_k,
        search_type=body.search_type,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
        repeat_penalty=body.repeat_penalty,
    )

    response = query(request)

    if response.error:
        raise HTTPException(status_code=400, detail=response.error)

    return QueryResponse(
        answer=response.answer,
        sources=response.sources,
        num_chunks_retrieved=response.num_chunks_retrieved,
    )


@router.get("/health", response_model=IndexStatsResponse)
async def api_health() -> IndexStatsResponse:
    """Return the current status of the FAISS index."""
    stats = get_index_stats()
    return IndexStatsResponse(**stats)

"""
FAISS vector store management.

Handles creation, persistence, loading, and querying of the FAISS index.
Documents are embedded using sentence-transformers and stored locally
for fast similarity search during RAG retrieval.
"""

import logging
import shutil
from pathlib import Path
from threading import Lock

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config import (
    DEFAULT_SEARCH_TYPE,
    DEFAULT_TOP_K,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_DIR,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton embedding model — loaded once, shared across operations.
# ---------------------------------------------------------------------------

_embeddings_instance: HuggingFaceEmbeddings | None = None
_embeddings_lock = Lock()


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return the singleton HuggingFaceEmbeddings instance.

    The embedding model (~90MB) is downloaded on first use and cached
    by sentence-transformers in ~/.cache/torch/sentence_transformers/.
    """
    global _embeddings_instance

    if _embeddings_instance is not None:
        return _embeddings_instance

    with _embeddings_lock:
        if _embeddings_instance is not None:
            return _embeddings_instance

        logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model loaded.")
        return _embeddings_instance


# ---------------------------------------------------------------------------
# FAISS index — singleton with persistence support.
# ---------------------------------------------------------------------------

_vectorstore_instance: FAISS | None = None
_vectorstore_lock = Lock()

# Name used by FAISS serialization (produces index.faiss + index.pkl)
_INDEX_NAME = "index"


def _index_exists() -> bool:
    """Check whether a persisted FAISS index exists on disk."""
    return (FAISS_INDEX_DIR / f"{_INDEX_NAME}.faiss").exists()


def load_vectorstore() -> FAISS | None:
    """
    Load a previously persisted FAISS index from disk.

    Returns:
        FAISS vectorstore instance, or None if no index exists.
    """
    global _vectorstore_instance

    if _vectorstore_instance is not None:
        return _vectorstore_instance

    with _vectorstore_lock:
        if _vectorstore_instance is not None:
            return _vectorstore_instance

        if not _index_exists():
            logger.info("No existing FAISS index found at %s", FAISS_INDEX_DIR)
            return None

        logger.info("Loading FAISS index from %s", FAISS_INDEX_DIR)
        try:
            _vectorstore_instance = FAISS.load_local(
                folder_path=str(FAISS_INDEX_DIR),
                embeddings=get_embeddings(),
                index_name=_INDEX_NAME,
                allow_dangerous_deserialization=True,  # Required for pickle-based FAISS
            )
            doc_count = _vectorstore_instance.index.ntotal
            logger.info("FAISS index loaded — %d vectors.", doc_count)
            return _vectorstore_instance
        except Exception:
            logger.exception("Failed to load FAISS index.")
            return None


def create_vectorstore(documents: list[Document]) -> FAISS:
    """
    Create a new FAISS index from a list of Document chunks and persist it.

    This **replaces** any existing index.

    Args:
        documents: Pre-chunked Document objects to embed and index.

    Returns:
        The newly created FAISS vectorstore instance.

    Raises:
        ValueError: If documents list is empty.
    """
    global _vectorstore_instance

    if not documents:
        raise ValueError("Cannot create vector store from an empty document list.")

    with _vectorstore_lock:
        logger.info("Creating FAISS index from %d chunks ...", len(documents))
        embeddings = get_embeddings()
        _vectorstore_instance = FAISS.from_documents(documents, embeddings)

        # Persist to disk
        FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        _vectorstore_instance.save_local(
            folder_path=str(FAISS_INDEX_DIR),
            index_name=_INDEX_NAME,
        )
        logger.info(
            "FAISS index created and saved — %d vectors.",
            _vectorstore_instance.index.ntotal,
        )
        return _vectorstore_instance


def add_documents(documents: list[Document]) -> FAISS:
    """
    Add new document chunks to an existing FAISS index (or create one).

    Args:
        documents: Pre-chunked Document objects to add.

    Returns:
        Updated FAISS vectorstore instance.

    Raises:
        ValueError: If documents list is empty.
    """
    global _vectorstore_instance

    if not documents:
        raise ValueError("No documents provided to add.")

    with _vectorstore_lock:
        existing = _vectorstore_instance or load_vectorstore()

        if existing is None:
            # No index yet — create from scratch (release lock via recursion avoided)
            pass
        else:
            logger.info("Adding %d chunks to existing FAISS index ...", len(documents))
            existing.add_documents(documents)
            existing.save_local(
                folder_path=str(FAISS_INDEX_DIR),
                index_name=_INDEX_NAME,
            )
            _vectorstore_instance = existing
            logger.info(
                "FAISS index updated — now %d vectors.",
                _vectorstore_instance.index.ntotal,
            )
            return _vectorstore_instance

    # Fell through — no existing index, create new one
    return create_vectorstore(documents)


def get_retriever(
    top_k: int = DEFAULT_TOP_K,
    search_type: str = DEFAULT_SEARCH_TYPE,
) -> object:
    """
    Return a LangChain retriever backed by the FAISS index.

    Args:
        top_k:       Number of chunks to retrieve.
        search_type: "similarity" or "mmr" (Maximal Marginal Relevance).

    Returns:
        A LangChain VectorStoreRetriever.

    Raises:
        RuntimeError: If no FAISS index is available.
    """
    vs = _vectorstore_instance or load_vectorstore()

    if vs is None:
        raise RuntimeError(
            "No FAISS index available. Upload documents in Tab 1 first."
        )

    search_kwargs = {"k": top_k}
    if search_type == "mmr":
        search_kwargs["fetch_k"] = top_k * 4  # MMR fetches more, then diversifies

    return vs.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )


def reset_vectorstore() -> None:
    """
    Delete the persisted FAISS index and clear the in-memory instance.

    Used when the user wants to re-index from scratch (e.g., after
    removing or replacing all documents).
    """
    global _vectorstore_instance

    with _vectorstore_lock:
        _vectorstore_instance = None

        if FAISS_INDEX_DIR.exists():
            shutil.rmtree(FAISS_INDEX_DIR)
            FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
            logger.info("FAISS index deleted from %s", FAISS_INDEX_DIR)


def get_index_stats() -> dict:
    """
    Return basic statistics about the current FAISS index.

    Returns:
        Dict with 'num_vectors' and 'index_exists' keys.
    """
    vs = _vectorstore_instance or load_vectorstore()

    if vs is None:
        return {"index_exists": False, "num_vectors": 0}

    return {
        "index_exists": True,
        "num_vectors": vs.index.ntotal,
    }

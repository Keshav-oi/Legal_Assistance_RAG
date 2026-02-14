"""
FAISS vector store management.

Handles creation, persistence, loading, and querying of the FAISS index.
Documents are embedded using sentence-transformers and stored locally
for fast similarity search during RAG retrieval.
"""

import logging
import shutil
from pathlib import Path

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
# Singleton embedding model
# ---------------------------------------------------------------------------

_embeddings_instance: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return the singleton HuggingFaceEmbeddings instance.

    The embedding model (~90MB) is downloaded on first use and cached
    by sentence-transformers in ~/.cache/torch/sentence_transformers/.
    """
    global _embeddings_instance

    if _embeddings_instance is not None:
        return _embeddings_instance

    print(f"[EMBEDDINGS] Loading embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"[EMBEDDINGS] This downloads ~90MB on first run. Please wait...")

    try:
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("[EMBEDDINGS] ✅ Embedding model loaded successfully.")
        return _embeddings_instance
    except Exception as exc:
        print(f"[EMBEDDINGS] ❌ Failed to load embedding model: {exc}")
        raise


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------

_vectorstore_instance: FAISS | None = None

# Name used by FAISS serialization (produces index.faiss + index.pkl)
_INDEX_NAME = "index"


def _index_exists_on_disk() -> bool:
    """Check whether a persisted FAISS index exists on disk."""
    exists = (FAISS_INDEX_DIR / f"{_INDEX_NAME}.faiss").exists()
    print(f"[FAISS] Index on disk at {FAISS_INDEX_DIR}: {exists}")
    return exists


def load_vectorstore() -> FAISS | None:
    """
    Load a previously persisted FAISS index from disk.

    Returns:
        FAISS vectorstore instance, or None if no index exists.
    """
    global _vectorstore_instance

    if _vectorstore_instance is not None:
        print(f"[FAISS] Returning cached vectorstore ({_vectorstore_instance.index.ntotal} vectors)")
        return _vectorstore_instance

    if not _index_exists_on_disk():
        print("[FAISS] No existing index found on disk.")
        return None

    print(f"[FAISS] Loading index from {FAISS_INDEX_DIR} ...")
    try:
        embeddings = get_embeddings()
        _vectorstore_instance = FAISS.load_local(
            folder_path=str(FAISS_INDEX_DIR),
            embeddings=embeddings,
            index_name=_INDEX_NAME,
            allow_dangerous_deserialization=True,
        )
        doc_count = _vectorstore_instance.index.ntotal
        print(f"[FAISS] ✅ Index loaded — {doc_count} vectors.")
        return _vectorstore_instance
    except Exception as exc:
        print(f"[FAISS] ❌ Failed to load index: {exc}")
        logger.exception("Failed to load FAISS index.")
        return None


def create_vectorstore(documents: list[Document]) -> FAISS:
    """
    Create a new FAISS index from a list of Document chunks and persist it.

    This **replaces** any existing index.
    """
    global _vectorstore_instance

    if not documents:
        raise ValueError("Cannot create vector store from an empty document list.")

    print(f"[FAISS] Creating new index from {len(documents)} chunks ...")
    print(f"[FAISS] Step 1/3: Loading embeddings model ...")
    embeddings = get_embeddings()

    print(f"[FAISS] Step 2/3: Embedding {len(documents)} chunks (this may take a moment) ...")
    _vectorstore_instance = FAISS.from_documents(documents, embeddings)

    print(f"[FAISS] Step 3/3: Saving index to {FAISS_INDEX_DIR} ...")
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    _vectorstore_instance.save_local(
        folder_path=str(FAISS_INDEX_DIR),
        index_name=_INDEX_NAME,
    )

    count = _vectorstore_instance.index.ntotal
    print(f"[FAISS] ✅ Index created and saved — {count} vectors.")
    return _vectorstore_instance


def add_documents(documents: list[Document]) -> FAISS:
    """
    Add new document chunks to an existing FAISS index (or create one).
    """
    global _vectorstore_instance

    if not documents:
        raise ValueError("No documents provided to add.")

    print(f"[FAISS] add_documents called with {len(documents)} chunks.")

    # Check in-memory instance first, then disk — NO lock nesting
    existing = _vectorstore_instance
    if existing is None:
        existing = load_vectorstore()

    if existing is None:
        print("[FAISS] No existing index — creating new one.")
        return create_vectorstore(documents)

    print(f"[FAISS] Adding {len(documents)} chunks to existing index ...")
    embeddings = get_embeddings()
    existing.add_documents(documents)
    existing.save_local(
        folder_path=str(FAISS_INDEX_DIR),
        index_name=_INDEX_NAME,
    )
    _vectorstore_instance = existing
    count = _vectorstore_instance.index.ntotal
    print(f"[FAISS] ✅ Index updated — now {count} vectors.")
    return _vectorstore_instance


def get_retriever(
    top_k: int = DEFAULT_TOP_K,
    search_type: str = DEFAULT_SEARCH_TYPE,
) -> object:
    """
    Return a LangChain retriever backed by the FAISS index.
    """
    vs = _vectorstore_instance
    if vs is None:
        vs = load_vectorstore()

    if vs is None:
        raise RuntimeError(
            "No FAISS index available. Upload documents in Tab 1 first."
        )

    search_kwargs = {"k": top_k}
    if search_type == "mmr":
        search_kwargs["fetch_k"] = top_k * 4

    print(f"[FAISS] Retriever created: search_type={search_type}, top_k={top_k}")
    return vs.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )


def reset_vectorstore() -> None:
    """Delete the persisted FAISS index and clear the in-memory instance."""
    global _vectorstore_instance

    _vectorstore_instance = None

    if FAISS_INDEX_DIR.exists():
        shutil.rmtree(FAISS_INDEX_DIR)
        FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[FAISS] 🗑️ Index deleted from {FAISS_INDEX_DIR}")


def get_index_stats() -> dict:
    """
    Return basic statistics about the current FAISS index.

    This does NOT call load_vectorstore to avoid blocking.
    It only checks the in-memory instance or looks at disk.
    """
    # Check in-memory first (instant, no I/O)
    if _vectorstore_instance is not None:
        return {
            "index_exists": True,
            "num_vectors": _vectorstore_instance.index.ntotal,
        }

    # Check disk without loading (just file existence check)
    if (FAISS_INDEX_DIR / f"{_INDEX_NAME}.faiss").exists():
        return {
            "index_exists": True,
            "num_vectors": -1,  # Exists on disk but not loaded into memory yet
        }

    return {"index_exists": False, "num_vectors": 0}
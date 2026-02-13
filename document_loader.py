"""
Document loading and chunking.

Supports PDF, TXT, Markdown, DOCX, CSV, and HTML files.
Documents are loaded, split into chunks with configurable size/overlap,
and returned as LangChain Document objects ready for embedding.
"""

import logging
from pathlib import Path

from langchain_community.document_loaders import (
    CSVLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DOCUMENTS_DIR,
    SUPPORTED_EXTENSIONS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Loader registry — maps file extension to the appropriate LangChain loader
# ---------------------------------------------------------------------------

_LOADER_MAP: dict[str, type] = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".docx": Docx2txtLoader,
    ".csv": CSVLoader,
    ".html": UnstructuredHTMLLoader,
    ".htm": UnstructuredHTMLLoader,
}


def _load_single_file(file_path: Path) -> list[Document]:
    """
    Load a single file and return a list of LangChain Document objects.

    Args:
        file_path: Absolute path to the file.

    Returns:
        List of Document objects. Empty list if the file type is unsupported
        or loading fails.
    """
    ext = file_path.suffix.lower()

    if ext not in _LOADER_MAP:
        logger.warning("Unsupported file type '%s' for %s — skipping.", ext, file_path.name)
        return []

    loader_cls = _LOADER_MAP[ext]

    try:
        loader = loader_cls(str(file_path))
        docs = loader.load()
        # Tag every document with its source filename for traceability
        for doc in docs:
            doc.metadata["source"] = file_path.name
        logger.info("Loaded %d page(s) from %s", len(docs), file_path.name)
        return docs
    except Exception:
        logger.exception("Failed to load %s", file_path.name)
        return []


def load_documents(directory: Path | None = None) -> list[Document]:
    """
    Recursively load all supported documents from a directory.

    Args:
        directory: Path to scan. Defaults to DOCUMENTS_DIR from config.

    Returns:
        Flat list of Document objects from all loadable files.
    """
    directory = directory or DOCUMENTS_DIR

    if not directory.exists():
        logger.warning("Documents directory does not exist: %s", directory)
        return []

    all_docs: list[Document] = []

    for file_path in sorted(directory.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            all_docs.extend(_load_single_file(file_path))

    logger.info("Total documents loaded: %d", len(all_docs))
    return all_docs


def split_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split documents into smaller chunks for embedding and retrieval.

    Uses RecursiveCharacterTextSplitter which tries to split on natural
    boundaries (paragraphs → sentences → words) before falling back to
    character count.

    Args:
        documents:     List of Document objects to split.
        chunk_size:    Maximum characters per chunk.
        chunk_overlap: Character overlap between consecutive chunks.

    Returns:
        List of chunked Document objects with preserved metadata.
    """
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    logger.info(
        "Split %d document(s) into %d chunk(s) (size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks


def load_and_split(
    directory: Path | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """
    Convenience function: load all documents from a directory and split them.

    Args:
        directory:     Path to scan for documents.
        chunk_size:    Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of chunked Document objects ready for embedding.
    """
    docs = load_documents(directory)
    return split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

"""
Gradio Tab 1 — Document Management.

Allows users to:
- Upload RAG documents (PDF, TXT, MD, DOCX, CSV, HTML)
- View currently uploaded files
- Trigger document indexing into the FAISS vector store
- Reset the index and clear all documents
"""

import logging
import shutil
from pathlib import Path

import gradio as gr

from app.config import (
    CHUNK_OVERLAP_RANGE,
    CHUNK_SIZE_RANGE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DOCUMENTS_DIR,
    SUPPORTED_EXTENSIONS,
)
from app.rag.document_loader import load_and_split
from app.rag.vector_store import (
    add_documents,
    get_index_stats,
    reset_vectorstore,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _list_uploaded_files() -> str:
    """Return a formatted string listing all files in the documents directory."""
    if not DOCUMENTS_DIR.exists():
        return "📂 No documents directory found."

    files = sorted(
        f for f in DOCUMENTS_DIR.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        return "📂 No documents uploaded yet."

    lines = [f"📄 {f.relative_to(DOCUMENTS_DIR)}  ({f.stat().st_size / 1024:.1f} KB)" for f in files]
    return f"**{len(lines)} document(s) uploaded:**\n\n" + "\n".join(lines)


def _get_status() -> str:
    """Return index stats as a formatted status string."""
    stats = get_index_stats()
    if stats["index_exists"]:
        return f"✅ FAISS index active — **{stats['num_vectors']}** vectors indexed."
    return "⚠️ No FAISS index. Upload documents and click **Index Documents**."


# ---------------------------------------------------------------------------
# Gradio event handlers
# ---------------------------------------------------------------------------


def handle_upload(files: list[str]) -> tuple[str, str]:
    """
    Save uploaded files to the documents directory.

    Args:
        files: List of temporary file paths from Gradio upload component.

    Returns:
        Tuple of (file list display, status message).
    """
    if not files:
        return _list_uploaded_files(), "⚠️ No files selected."

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    saved = []

    for tmp_path in files:
        tmp = Path(tmp_path)
        ext = tmp.suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            logger.warning("Skipped unsupported file: %s", tmp.name)
            continue

        dest = DOCUMENTS_DIR / tmp.name

        # Avoid overwriting — append counter if file exists
        counter = 1
        while dest.exists():
            dest = DOCUMENTS_DIR / f"{tmp.stem}_{counter}{tmp.suffix}"
            counter += 1

        shutil.copy2(str(tmp), str(dest))
        saved.append(dest.name)
        logger.info("Saved uploaded file: %s", dest.name)

    if not saved:
        return _list_uploaded_files(), "⚠️ No supported files in upload."

    msg = f"✅ Uploaded {len(saved)} file(s): {', '.join(saved)}"
    return _list_uploaded_files(), msg


def handle_index(chunk_size: int, chunk_overlap: int) -> str:
    """
    Load all documents, chunk them, and index into FAISS.

    Args:
        chunk_size:    Characters per chunk.
        chunk_overlap: Overlap between chunks.

    Returns:
        Status message.
    """
    try:
        chunks = load_and_split(
            directory=DOCUMENTS_DIR,
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
        )

        if not chunks:
            return "⚠️ No documents found to index. Upload files first."

        add_documents(chunks)
        stats = get_index_stats()
        return (
            f"✅ Indexed successfully!\n\n"
            f"- Chunks created: **{len(chunks)}**\n"
            f"- Total vectors in index: **{stats['num_vectors']}**"
        )

    except Exception as exc:
        logger.exception("Indexing failed.")
        return f"❌ Indexing failed: {exc}"


def handle_reset() -> tuple[str, str]:
    """
    Delete all uploaded documents and reset the FAISS index.

    Returns:
        Tuple of (file list display, status message).
    """
    try:
        reset_vectorstore()

        # Clear uploaded files
        if DOCUMENTS_DIR.exists():
            shutil.rmtree(DOCUMENTS_DIR)
            DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

        return _list_uploaded_files(), "🗑️ All documents and index cleared."

    except Exception as exc:
        logger.exception("Reset failed.")
        return _list_uploaded_files(), f"❌ Reset failed: {exc}"


def handle_refresh() -> tuple[str, str]:
    """Refresh the file list and index status displays."""
    return _list_uploaded_files(), _get_status()


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------


def build_tab() -> gr.TabItem:
    """Construct and return the Document Management tab."""

    with gr.TabItem("📁 Documents", id="tab_documents") as tab:
        gr.Markdown("## Document Management\nUpload documents for RAG retrieval and manage the FAISS index.")

        with gr.Row():
            # --- Left column: Upload ---
            with gr.Column(scale=1):
                file_upload = gr.File(
                    label="Upload Documents",
                    file_count="multiple",
                    file_types=[ext for ext in SUPPORTED_EXTENSIONS],
                    type="filepath",
                )
                upload_btn = gr.Button("📤 Upload Files", variant="primary")

            # --- Right column: Status ---
            with gr.Column(scale=1):
                file_list_display = gr.Markdown(
                    value=_list_uploaded_files(),
                    label="Uploaded Files",
                )
                index_status = gr.Markdown(
                    value=_get_status(),
                    label="Index Status",
                )

        gr.Markdown("### Indexing Settings")

        with gr.Row():
            chunk_size = gr.Slider(
                minimum=CHUNK_SIZE_RANGE[0],
                maximum=CHUNK_SIZE_RANGE[1],
                value=DEFAULT_CHUNK_SIZE,
                step=50,
                label="Chunk Size (characters)",
                info="Larger chunks retain more context but reduce retrieval precision.",
            )
            chunk_overlap = gr.Slider(
                minimum=CHUNK_OVERLAP_RANGE[0],
                maximum=CHUNK_OVERLAP_RANGE[1],
                value=DEFAULT_CHUNK_OVERLAP,
                step=25,
                label="Chunk Overlap (characters)",
                info="Overlap helps preserve context across chunk boundaries.",
            )

        with gr.Row():
            index_btn = gr.Button("🔍 Index Documents", variant="primary")
            refresh_btn = gr.Button("🔄 Refresh Status")
            reset_btn = gr.Button("🗑️ Reset All", variant="stop")

        status_msg = gr.Markdown(value="", label="Action Status")

        # --- Event wiring ---
        upload_btn.click(
            fn=handle_upload,
            inputs=[file_upload],
            outputs=[file_list_display, status_msg],
        )

        index_btn.click(
            fn=handle_index,
            inputs=[chunk_size, chunk_overlap],
            outputs=[status_msg],
        )

        reset_btn.click(
            fn=handle_reset,
            inputs=[],
            outputs=[file_list_display, status_msg],
        )

        refresh_btn.click(
            fn=handle_refresh,
            inputs=[],
            outputs=[file_list_display, index_status],
        )

    return tab

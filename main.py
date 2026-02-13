"""
Application entry point.

Creates the FastAPI app, mounts the Gradio UI, registers API routes,
and handles startup/shutdown lifecycle events.
"""

import logging
from contextlib import asynccontextmanager

import gradio as gr
import uvicorn
from fastapi import FastAPI

from app.api.routes import router as api_router
from app.config import (
    APP_DESCRIPTION,
    APP_HOST,
    APP_PORT,
    APP_TITLE,
    APP_VERSION,
    MODEL_PATH,
    ensure_directories,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # --- Startup ---
    logger.info("Starting %s v%s", APP_TITLE, APP_VERSION)
    ensure_directories()

    # Validate model file exists (don't load yet — that happens on first query)
    if not MODEL_PATH.exists():
        logger.warning(
            "⚠️  Model file not found at %s. "
            "Run 'python download_model.py' on the host before querying.",
            MODEL_PATH,
        )
    else:
        logger.info("Model file found: %s (%.1f GB)", MODEL_PATH.name, MODEL_PATH.stat().st_size / 1e9)

    # Pre-load FAISS index if one exists from a previous session
    try:
        from app.rag.vector_store import load_vectorstore

        vs = load_vectorstore()
        if vs:
            logger.info("Existing FAISS index loaded at startup.")
    except Exception:
        logger.exception("Failed to load existing FAISS index at startup.")

    yield

    # --- Shutdown ---
    logger.info("Shutting down %s.", APP_TITLE)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
)

# Register API routes
app.include_router(api_router)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def build_gradio_app() -> gr.Blocks:
    """Build the Gradio Blocks app with all three tabs."""

    # Import here to avoid circular imports during startup
    from app.ui.tab_documents import build_tab as build_documents_tab
    from app.ui.tab_history import build_tab as build_history_tab
    from app.ui.tab_query import build_tab as build_query_tab

    with gr.Blocks(
        title=APP_TITLE,
        theme=gr.themes.Soft(),
        css="""
            .gradio-container { max-width: 1200px !important; }
            footer { display: none !important; }
        """,
    ) as demo:
        gr.Markdown(
            f"# 🔎 {APP_TITLE}\n"
            f"*{APP_DESCRIPTION}*\n\n"
            "Upload documents → Index them → Ask questions."
        )

        build_documents_tab()
        build_query_tab()
        build_history_tab()

    return demo


# Mount Gradio onto FastAPI at root path
gradio_app = build_gradio_app()
app = gr.mount_gradio_app(app, gradio_app, path="/")


# ---------------------------------------------------------------------------
# Direct run support (development)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=False,
        log_level="info",
    )

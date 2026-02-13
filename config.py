"""
Centralized configuration for the RAG pipeline.

All paths, model parameters, RAG defaults, and constants are defined here.
Environment variables override defaults where specified.
"""

import os
from pathlib import Path

# =============================================================================
# Directory Paths
# =============================================================================

# Base project root (inside container: /app, local dev: project root)
BASE_DIR = Path(os.getenv("BASE_DIR", Path(__file__).resolve().parent.parent))

# Model directory — mounted from host into container
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "models"))

# User-uploaded RAG documents
DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", BASE_DIR / "data" / "documents"))

# Persisted FAISS index
FAISS_INDEX_DIR = Path(os.getenv("FAISS_INDEX_DIR", BASE_DIR / "data" / "faiss_index"))

# =============================================================================
# Model Configuration
# =============================================================================

# Llama 3.1 8B Instruct — Q4_K_M quantization (best speed/quality tradeoff)
MODEL_FILENAME = os.getenv(
    "MODEL_FILENAME",
    "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
)
MODEL_PATH = MODEL_DIR / MODEL_FILENAME

# HuggingFace repo for downloading the GGUF model
HF_REPO_ID = os.getenv("HF_REPO_ID", "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF")
HF_FILENAME = MODEL_FILENAME

# llama-cpp-python inference settings
MODEL_N_CTX = int(os.getenv("MODEL_N_CTX", "4096"))         # Context window size
MODEL_N_GPU_LAYERS = int(os.getenv("MODEL_N_GPU_LAYERS", "-1"))  # -1 = offload all layers to GPU (Metal)
MODEL_N_BATCH = int(os.getenv("MODEL_N_BATCH", "512"))      # Batch size for prompt processing
MODEL_VERBOSE = os.getenv("MODEL_VERBOSE", "false").lower() == "true"

# =============================================================================
# Embedding Model Configuration
# =============================================================================

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
)
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

# =============================================================================
# RAG Default Parameters (user-adjustable via Gradio Tab 2)
# =============================================================================

# Document chunking
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "200"))

# Retrieval
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "4"))            # Number of chunks retrieved
DEFAULT_SEARCH_TYPE = os.getenv("DEFAULT_SEARCH_TYPE", "similarity")  # similarity | mmr

# Generation
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "512"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.1"))
DEFAULT_TOP_P = float(os.getenv("DEFAULT_TOP_P", "0.95"))
DEFAULT_REPEAT_PENALTY = float(os.getenv("DEFAULT_REPEAT_PENALTY", "1.1"))

# =============================================================================
# Parameter Bounds (for Gradio slider validation)
# =============================================================================

MAX_TOKENS_RANGE = (64, 2048)
TEMPERATURE_RANGE = (0.0, 2.0)
TOP_P_RANGE = (0.0, 1.0)
TOP_K_RANGE = (1, 20)
CHUNK_SIZE_RANGE = (100, 2000)
CHUNK_OVERLAP_RANGE = (0, 500)
REPEAT_PENALTY_RANGE = (1.0, 2.0)

# =============================================================================
# Supported Document Formats
# =============================================================================

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md",
    ".docx",
    ".csv",
    ".html",
    ".htm",
}

# =============================================================================
# FastAPI / Server
# =============================================================================

APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
APP_TITLE = "RAG Pipeline"
APP_DESCRIPTION = "End-to-end RAG pipeline with Llama 3.1 8B, LangChain, FAISS, and Gradio"
APP_VERSION = "1.0.0"

# =============================================================================
# Prompt Template (Llama 3.1 Instruct format)
# =============================================================================

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the following retrieved context to answer "
    "the user's question accurately. If the context does not contain enough "
    "information to answer, say so clearly — do not make up information."
)

RAG_PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

Context:
{context}

Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

# =============================================================================
# Ensure directories exist
# =============================================================================


def ensure_directories() -> None:
    """Create required data directories if they do not exist."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

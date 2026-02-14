"""
LLM wrapper for llama-cpp-python.

Loads a GGUF model from disk and exposes it as a LangChain-compatible LLM.
Implements singleton pattern so the model is loaded once and reused.
"""

import logging
from pathlib import Path

from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

from app.config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_REPEAT_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    MODEL_N_BATCH,
    MODEL_N_CTX,
    MODEL_N_GPU_LAYERS,
    MODEL_PATH,
    MODEL_VERBOSE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton LLM instance
# ---------------------------------------------------------------------------

_llm_instance: LlamaCpp | None = None


def get_llm() -> LlamaCpp:
    """
    Return the singleton LlamaCpp instance, loading the model on first call.

    Raises:
        FileNotFoundError: If the GGUF model file does not exist at MODEL_PATH.
        RuntimeError: If the model fails to load.
    """
    global _llm_instance

    if _llm_instance is not None:
        print("[LLM] Returning cached model instance.")
        return _llm_instance

    model_path = Path(MODEL_PATH)
    print(f"[LLM] Looking for model at: {model_path}")

    if not model_path.exists():
        print(f"[LLM] ❌ Model file NOT FOUND at {model_path}")
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            f"Run 'python download_model.py' on the host first."
        )

    size_gb = model_path.stat().st_size / (1024 ** 3)
    print(f"[LLM] Model found: {model_path.name} ({size_gb:.2f} GB)")
    print(f"[LLM] Loading model with settings:")
    print(f"[LLM]   n_ctx={MODEL_N_CTX}, n_gpu_layers={MODEL_N_GPU_LAYERS}, n_batch={MODEL_N_BATCH}")
    print(f"[LLM]   This may take 30-60 seconds on first load...")

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    try:
        _llm_instance = LlamaCpp(
            model_path=str(model_path),
            n_ctx=MODEL_N_CTX,
            n_gpu_layers=MODEL_N_GPU_LAYERS,
            n_batch=MODEL_N_BATCH,
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
            top_p=DEFAULT_TOP_P,
            repeat_penalty=DEFAULT_REPEAT_PENALTY,
            callback_manager=callback_manager,
            verbose=MODEL_VERBOSE,
        )
    except Exception as exc:
        print(f"[LLM] ❌ Failed to load model: {exc}")
        logger.exception("Failed to load LLM model.")
        raise RuntimeError(f"Model loading failed: {exc}") from exc

    print("[LLM] ✅ Model loaded successfully!")
    return _llm_instance


def create_llm_with_params(
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    repeat_penalty: float = DEFAULT_REPEAT_PENALTY,
) -> LlamaCpp:
    """
    Return an LLM instance with user-specified generation parameters.
    Reuses the already-loaded model.
    """
    llm = get_llm()

    llm.max_tokens = max_tokens
    llm.temperature = temperature
    llm.top_p = top_p
    llm.repeat_penalty = repeat_penalty

    print(f"[LLM] Params set: max_tokens={max_tokens}, temp={temperature}, top_p={top_p}, repeat={repeat_penalty}")
    return llm
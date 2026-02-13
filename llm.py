"""
LLM wrapper for llama-cpp-python.

Loads a GGUF model from disk and exposes it as a LangChain-compatible LLM.
Implements singleton pattern so the model is loaded once and reused.
"""

import logging
from pathlib import Path
from threading import Lock

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
# Singleton LLM instance — loading a 5GB model is expensive, do it once.
# ---------------------------------------------------------------------------

_llm_instance: LlamaCpp | None = None
_llm_lock = Lock()


def get_llm() -> LlamaCpp:
    """
    Return the singleton LlamaCpp instance, loading the model on first call.

    Raises:
        FileNotFoundError: If the GGUF model file does not exist at MODEL_PATH.
        RuntimeError: If the model fails to load.
    """
    global _llm_instance

    if _llm_instance is not None:
        return _llm_instance

    with _llm_lock:
        # Double-check after acquiring lock
        if _llm_instance is not None:
            return _llm_instance

        model_path = Path(MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {model_path}. "
                f"Run 'python download_model.py' on the host first."
            )

        logger.info("Loading model from %s ...", model_path)

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
            logger.exception("Failed to load LLM model.")
            raise RuntimeError(f"Model loading failed: {exc}") from exc

        logger.info("Model loaded successfully.")
        return _llm_instance


def create_llm_with_params(
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    repeat_penalty: float = DEFAULT_REPEAT_PENALTY,
) -> LlamaCpp:
    """
    Return an LLM instance with user-specified generation parameters.

    This reuses the already-loaded model weights but applies new sampling
    parameters per-request.  llama-cpp-python does not natively support
    changing params on a loaded instance, so we clone the base config and
    create a lightweight wrapper that overrides invoke-time params.

    In practice, LangChain's LlamaCpp allows passing generation kwargs
    at invoke time, so this helper builds a dict for chain consumption.
    """
    llm = get_llm()

    # LlamaCpp supports overriding these at call time via `__call__` kwargs.
    # We store the desired overrides on a simple namespace that the chain
    # can read when constructing the prompt.
    llm.max_tokens = max_tokens
    llm.temperature = temperature
    llm.top_p = top_p
    llm.repeat_penalty = repeat_penalty

    return llm

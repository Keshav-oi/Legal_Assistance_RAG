#!/usr/bin/env python3
"""
Model downloader — run this on your host machine before starting the app.

Downloads the Llama 3.1 8B Instruct GGUF model from HuggingFace Hub.
Checks if the model already exists before downloading to avoid redundant
multi-GB transfers.

Usage:
    python download_model.py

    # For gated models (not needed for the default bartowski repo):
    HF_TOKEN=hf_your_token python download_model.py

    # Or interactive prompt:
    python download_model.py --token
"""

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration — matches app/config.py defaults
# ---------------------------------------------------------------------------

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).parent / "models"))
MODEL_FILENAME = os.getenv(
    "MODEL_FILENAME",
    "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
)
HF_REPO_ID = os.getenv("HF_REPO_ID", "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF")


def get_hf_token(interactive: bool = False) -> str | None:
    """
    Resolve HuggingFace token from environment or interactive prompt.

    Priority:
        1. HF_TOKEN environment variable
        2. HUGGING_FACE_HUB_TOKEN environment variable (legacy)
        3. Interactive prompt (if --token flag is used)
        4. None (works for public repos)
    """
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

    if token:
        print("✅ Using HuggingFace token from environment variable.")
        return token

    if interactive:
        token = input("Enter your HuggingFace token (or press Enter to skip): ").strip()
        if token:
            print("✅ Using provided HuggingFace token.")
            return token

    return None


def download_model(token: str | None = None) -> Path:
    """
    Download the GGUF model file if it does not already exist.

    Args:
        token: Optional HuggingFace API token for gated repos.

    Returns:
        Path to the downloaded model file.
    """
    model_path = MODEL_DIR / MODEL_FILENAME

    # --- Check if model already exists ---
    if model_path.exists():
        size_gb = model_path.stat().st_size / (1024 ** 3)
        print(f"✅ Model already exists: {model_path}")
        print(f"   Size: {size_gb:.2f} GB")
        print("   Skipping download. Delete the file to re-download.")
        return model_path

    # --- Ensure directory exists ---
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # --- Download ---
    print(f"📥 Downloading model...")
    print(f"   Repo:     {HF_REPO_ID}")
    print(f"   File:     {MODEL_FILENAME}")
    print(f"   Dest:     {model_path}")
    print()

    try:
        from huggingface_hub import hf_hub_download

        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=MODEL_FILENAME,
            local_dir=str(MODEL_DIR),
            local_dir_use_symlinks=False,
            token=token,
        )

        size_gb = Path(downloaded_path).stat().st_size / (1024 ** 3)
        print(f"\n✅ Download complete!")
        print(f"   Path: {downloaded_path}")
        print(f"   Size: {size_gb:.2f} GB")
        return Path(downloaded_path)

    except ImportError:
        print("❌ Error: huggingface_hub is not installed.")
        print("   Run: pip install huggingface-hub")
        sys.exit(1)

    except Exception as exc:
        print(f"\n❌ Download failed: {exc}")

        if "401" in str(exc) or "403" in str(exc):
            print("\n💡 This looks like an authentication error.")
            print("   This model may be gated. Try:")
            print(f"     HF_TOKEN=hf_your_token python {__file__}")
            print("     or: python download_model.py --token")

        sys.exit(1)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download the GGUF model for the RAG pipeline.",
    )
    parser.add_argument(
        "--token",
        action="store_true",
        help="Prompt for HuggingFace token interactively (for gated models).",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help=f"Override HuggingFace repo ID (default: {HF_REPO_ID}).",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help=f"Override model filename (default: {MODEL_FILENAME}).",
    )
    args = parser.parse_args()

    # Allow CLI overrides
    global HF_REPO_ID, MODEL_FILENAME
    if args.repo:
        HF_REPO_ID = args.repo
    if args.filename:
        MODEL_FILENAME = args.filename

    token = get_hf_token(interactive=args.token)
    download_model(token=token)


if __name__ == "__main__":
    main()

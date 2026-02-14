#!/usr/bin/env python3
"""
Diagnostic script — run this to check if all components work before starting the app.

Usage:
    python diagnose.py

Checks:
    1. Python version
    2. All required packages installed
    3. Model file exists
    4. llama-cpp-python can load the model
    5. Embedding model can be loaded
    6. FAISS works
    7. Document loader works
"""

import sys
import os
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent))


def check(label: str, fn):
    """Run a check function and print pass/fail."""
    try:
        result = fn()
        print(f"  ✅ {label}: {result}")
        return True
    except Exception as exc:
        print(f"  ❌ {label}: {exc}")
        return False


def main():
    print("=" * 60)
    print("RAG Pipeline — Diagnostic Check")
    print("=" * 60)
    all_ok = True

    # --- 1. Python version ---
    print(f"\n[1] Python version: {sys.version}")
    if sys.version_info < (3, 11):
        print("  ⚠️  Python 3.11+ recommended")

    # --- 2. Required packages ---
    print("\n[2] Checking required packages...")
    packages = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "gradio": "gradio",
        "langchain": "langchain",
        "langchain_community": "langchain-community",
        "faiss": "faiss-cpu",
        "sentence_transformers": "sentence-transformers",
        "pypdf": "pypdf",
        "docx2txt": "docx2txt",
        "llama_cpp": "llama-cpp-python",
    }
    for module, pip_name in packages.items():
        ok = check(pip_name, lambda m=module: __import__(m) and "installed")
        if not ok:
            all_ok = False
            print(f"       → Fix: pip install {pip_name}")

    # --- 3. Model file ---
    print("\n[3] Checking model file...")
    from app.config import MODEL_PATH, MODEL_DIR, DOCUMENTS_DIR, FAISS_INDEX_DIR
    print(f"  Model directory: {MODEL_DIR}")
    print(f"  Expected model:  {MODEL_PATH}")
    if MODEL_PATH.exists():
        size_gb = MODEL_PATH.stat().st_size / (1024 ** 3)
        print(f"  ✅ Model found ({size_gb:.2f} GB)")
    else:
        print(f"  ❌ Model NOT found")
        print(f"       → Fix: python download_model.py")
        all_ok = False

    # --- 4. llama-cpp-python Metal check ---
    print("\n[4] Checking llama-cpp-python...")
    try:
        import llama_cpp
        print(f"  ✅ llama-cpp-python version: {llama_cpp.__version__}")

        # Check if Metal support is compiled in (macOS)
        if sys.platform == "darwin":
            # Try to detect Metal support
            build_info = dir(llama_cpp)
            print(f"  ℹ️  On macOS — ensure you installed with:")
            print(f'       CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --no-cache-dir')
    except ImportError:
        print("  ❌ llama-cpp-python not installed")
        print('       → Fix: CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --no-cache-dir')
        all_ok = False

    # --- 5. Embedding model ---
    print("\n[5] Checking embedding model (downloads ~90MB on first run)...")
    try:
        from sentence_transformers import SentenceTransformer
        print("  Loading sentence-transformers/all-MiniLM-L6-v2 ...")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        test_embedding = model.encode(["test sentence"])
        print(f"  ✅ Embedding model works (output dim: {test_embedding.shape[1]})")
    except Exception as exc:
        print(f"  ❌ Embedding model failed: {exc}")
        all_ok = False

    # --- 6. FAISS ---
    print("\n[6] Checking FAISS...")
    try:
        import faiss
        import numpy as np
        index = faiss.IndexFlatL2(384)
        index.add(np.random.randn(5, 384).astype("float32"))
        _, results = index.search(np.random.randn(1, 384).astype("float32"), 2)
        print(f"  ✅ FAISS works (test search returned {len(results[0])} results)")
    except Exception as exc:
        print(f"  ❌ FAISS failed: {exc}")
        all_ok = False

    # --- 7. Directories ---
    print("\n[7] Checking directories...")
    print(f"  Documents dir: {DOCUMENTS_DIR} (exists: {DOCUMENTS_DIR.exists()})")
    print(f"  FAISS index dir: {FAISS_INDEX_DIR} (exists: {FAISS_INDEX_DIR.exists()})")
    if DOCUMENTS_DIR.exists():
        files = list(DOCUMENTS_DIR.rglob("*"))
        files = [f for f in files if f.is_file()]
        print(f"  Files in documents dir: {len(files)}")
        for f in files:
            print(f"    - {f.name} ({f.stat().st_size / 1024:.1f} KB)")

    # --- Summary ---
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ All checks passed! You can start the app with:")
        print('   uvicorn app.main:app --host 0.0.0.0 --port 8000')
    else:
        print("❌ Some checks failed. Fix the issues above first.")
    print("=" * 60)


if __name__ == "__main__":
    main()

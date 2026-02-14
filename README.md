# 🔎 RAG Pipeline

End-to-end RAG pipeline running locally — Llama 3.1 8B, LangChain, FAISS, FastAPI, Gradio.

---

## Setup

### Step 1: Clone the repo

```bash
git clone https://github.com/yourusername/rag-pipeline.git
cd rag-pipeline
```

### Step 2: Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install llama-cpp-python with Metal (Apple Silicon)

```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --no-cache-dir
```

For Linux (CPU only): `pip install llama-cpp-python`
For Linux (NVIDIA GPU): `CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir`

### Step 5: Download the model (~4.7 GB)

```bash
python download_model.py
```

Skips download if the model already exists. For gated models: `HF_TOKEN=hf_your_token python download_model.py`

### Step 6: Set up the folder structure

```bash
mkdir -p app/rag app/ui app/api
touch app/__init__.py app/rag/__init__.py app/ui/__init__.py app/api/__init__.py
```

Then place the files in the correct locations:

```
app/                → __init__.py, main.py, config.py
app/rag/            → __init__.py, llm.py, document_loader.py, vector_store.py, chain.py
app/ui/             → __init__.py, tab_documents.py, tab_query.py, tab_history.py
app/api/            → __init__.py, routes.py
root (project dir)  → download_model.py, diagnose.py, requirements.txt, Dockerfile, docker-compose.yml
```

### Step 7: Run diagnostics

```bash
python diagnose.py
```

This checks that all packages are installed, the model file exists, embeddings work, and FAISS works. Fix any failures before proceeding.

### Step 8: Start the app

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Step 9: Open in browser

Go to **http://localhost:8000**

---

## Usage

1. **📁 Documents tab** — Upload your files (PDF, TXT, MD, DOCX, CSV, HTML), set chunk size, click **Index Documents**
2. **💬 Query tab** — Type a question, adjust parameters if needed, click **Generate Answer**
3. **📜 History tab** — View all past queries and answers

---

## Docker (Optional)

If you prefer running in a container (note: no Metal GPU inside Docker on Mac — inference will be slower):

```bash
# Make sure you've already run Step 5 (download the model) first
docker compose up --build
```

Open **http://localhost:8000**. Stop with `docker compose down`.

---

## Swapping Models

```bash
export HF_REPO_ID="bartowski/Qwen2.5-7B-Instruct-GGUF"
export MODEL_FILENAME="Qwen2.5-7B-Instruct-Q4_K_M.gguf"
python download_model.py
```

Any GGUF model works. Recommended: Llama 3.1 8B (default), Mistral 7B, Gemma 2 9B, Qwen 2.5 7B.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'app'` | Make sure files are in the `app/` subdirectory structure, not flat |
| Model not found | Run `python download_model.py` |
| Inference hangs / very slow | Reinstall with Metal: `CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --no-cache-dir --force-reinstall` |
| `faiss-cpu` install fails | Run `pip install faiss-cpu` without a version pin |
| FAISS index not updating | Click **Refresh Status** or restart the app |
| Out of memory | Set `MODEL_N_CTX=2048` environment variable |
| `python diagnose.py` fails | Fix the specific failing check before starting the app |

# 🔎 RAG Pipeline

End-to-end Retrieval-Augmented Generation pipeline with **Llama 3.1 8B Instruct**, **LangChain**, **FAISS**, **FastAPI**, and **Gradio**.

Upload documents, index them into a vector store, and ask questions — all running locally on your machine.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green)


---

## Architecture

```
Host Machine (macOS / Linux)
├── models/                          ← GGUF model file (downloaded once, ~4.7 GB)
│
└── Docker Container (or native)
    ├── FastAPI + Gradio UI           ← 3-tab interface on localhost:8000
    ├── LangChain                     ← RAG orchestration
    ├── FAISS                         ← Vector similarity search
    ├── sentence-transformers          ← Document embeddings
    └── llama-cpp-python              ← Local LLM inference
```

**Three Gradio Tabs:**

| Tab | Purpose |
|-----|---------|
| 📁 Documents | Upload files (PDF, TXT, MD, DOCX, CSV, HTML), configure chunking, build FAISS index |
| 💬 Query | Ask questions with adjustable retrieval & generation parameters |
| 📜 History | Review all past queries, answers, sources, and parameters used |

---

## Prerequisites

- **Python 3.11+** (3.12 recommended)
- **~8 GB free disk space** (model ~4.7 GB + dependencies)
- **16 GB+ RAM** (for model inference)
- **Docker & Docker Compose** (only if running via Docker)
- **macOS Apple Silicon (M1–M4)** or **Linux x86_64** — both supported

---

## Quick Start (Native — Recommended for Mac)

> **Why native over Docker on Mac?** Docker Desktop on macOS runs a Linux VM, which cannot access Apple's Metal GPU. Running natively gives you Metal acceleration and **~5–10x faster inference**.

### Step 1: Clone the repository

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

### Step 4: Install llama-cpp-python with Metal support (Mac only)

```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python==0.3.4
```

On Linux (CPU only):

```bash
pip install llama-cpp-python==0.3.4
```

On Linux with NVIDIA GPU:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python==0.3.4
```

### Step 5: Download the model

```bash
python download_model.py
```

This downloads `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` (~4.7 GB) into the `models/` directory. If the file already exists, it skips the download.

**For gated models** (not needed for the default repo):

```bash
# Via environment variable
HF_TOKEN=hf_your_token python download_model.py

# Or interactive prompt
python download_model.py --token
```

### Step 6: Run the application

```bash
python -m app.main
```

### Step 7: Open the UI

Navigate to **http://localhost:8000** in your browser.

---

## Quick Start (Docker)

> **Note:** Docker on macOS does **not** have Metal GPU access. Inference will run on CPU, which is slower (~2–5 tokens/sec vs ~15–30 native). This is fine for testing and Linux deployments with NVIDIA GPUs.

### Step 1: Clone and download the model

```bash
git clone https://github.com/yourusername/rag-pipeline.git
cd rag-pipeline

# Model must exist on host before starting the container
pip install huggingface-hub
python download_model.py
```

### Step 2: Build and start the container

```bash
docker compose up --build
```

This will:
- Build the Docker image (~3–5 minutes first time)
- Mount `models/`, `data/documents/`, and `data/faiss_index/` from your host
- Start the app on port 8000

### Step 3: Open the UI

Navigate to **http://localhost:8000** in your browser.

### Stopping

```bash
docker compose down
```

---

## Usage Walkthrough

### 1. Upload Documents (Tab 1)

1. Go to the **📁 Documents** tab
2. Click **Upload Documents** and select your files (PDF, TXT, MD, DOCX, CSV, HTML)
3. Adjust **Chunk Size** and **Chunk Overlap** if needed (defaults work well for most documents)
4. Click **🔍 Index Documents** — this embeds all chunks and builds the FAISS index
5. You should see a green status confirming the number of vectors indexed

### 2. Query Your Documents (Tab 2)

1. Go to the **💬 Query** tab
2. Type your question in the text box
3. (Optional) Expand the accordions to adjust:
   - **System Prompt** — customize the LLM's instruction set
   - **Retrieval Parameters** — top-k chunks, similarity vs MMR search
   - **Generation Parameters** — max tokens, temperature, top-p, repeat penalty
4. Click **🚀 Generate Answer** (or press Enter)
5. The answer appears below with cited source filenames

### 3. Review History (Tab 3)

1. Go to the **📜 History** tab to see all past queries
2. Each entry shows the question, answer, sources, and the exact parameters used
3. Click the **Parameters** dropdown on any entry to see its settings

---

## API Usage

The FastAPI backend also exposes REST endpoints for programmatic access:

### Query

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the key findings?",
    "top_k": 4,
    "temperature": 0.1,
    "max_tokens": 512
  }'
```

### Health Check

```bash
curl http://localhost:8000/api/health
```

### Interactive API Docs

Visit **http://localhost:8000/docs** for the auto-generated Swagger UI.

---

## Project Structure

```
rag-pipeline/
├── README.md                  ← You are here
├── Dockerfile                 ← Container image definition
├── docker-compose.yml         ← Container orchestration with volume mounts
├── requirements.txt           ← Python dependencies
├── download_model.py          ← Model downloader (run on host)
├── .gitignore
├── app/
│   ├── __init__.py
│   ├── main.py                ← FastAPI app + Gradio mount + startup lifecycle
│   ├── config.py              ← All paths, defaults, constants (env-overridable)
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── tab_documents.py   ← Tab 1: Document upload & FAISS indexing
│   │   ├── tab_query.py       ← Tab 2: RAG query with all parameters
│   │   └── tab_history.py     ← Tab 3: Query history viewer
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── llm.py             ← llama-cpp-python LLM (singleton, lazy-loaded)
│   │   ├── document_loader.py ← Multi-format doc loading + chunking
│   │   ├── vector_store.py    ← FAISS index lifecycle (create/add/query/reset)
│   │   └── chain.py           ← LangChain RAG chain (retriever + LLM + prompt)
│   └── api/
│       ├── __init__.py
│       └── routes.py          ← REST API endpoints (/api/query, /api/health)
├── data/
│   ├── documents/             ← Uploaded RAG documents (persisted via volume)
│   └── faiss_index/           ← Serialized FAISS index (persisted via volume)
└── models/                    ← GGUF model file (host-side, mounted into container)
```

---

## Configuration

All settings are configurable via environment variables. Key ones:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `./models` | Path to GGUF model directory |
| `MODEL_FILENAME` | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | Model filename |
| `MODEL_N_CTX` | `4096` | Context window size |
| `MODEL_N_GPU_LAYERS` | `-1` (all) | GPU layers. Set `0` for CPU-only |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `DEFAULT_CHUNK_SIZE` | `1000` | Characters per chunk |
| `DEFAULT_CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `DEFAULT_TOP_K` | `4` | Chunks to retrieve |
| `DEFAULT_TEMPERATURE` | `0.1` | LLM sampling temperature |
| `APP_PORT` | `8000` | Server port |

---

## Using a Different Model

You can swap models by changing two environment variables:

```bash
# Example: Use Mistral 7B instead
export HF_REPO_ID="TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
export MODEL_FILENAME="mistral-7b-instruct-v0.2.Q4_K_M.gguf"
python download_model.py
```

Any GGUF-format model compatible with llama-cpp-python will work. Recommended models for local use:

- **Llama 3.1 8B Instruct** (default) — best overall balance
- **Mistral 7B v0.3 Instruct** — strong structured output
- **Gemma 2 9B Instruct** — excellent quality, slightly larger
- **Qwen 2.5 7B Instruct** — strong reasoning

Use **Q4_K_M** or **Q5_K_M** quantizations for the best speed/quality tradeoff.

---

## Troubleshooting

**Model not found error**
→ Run `python download_model.py` first. The model must exist before starting the app.

**Slow inference in Docker on Mac**
→ Expected. Docker on macOS cannot use Metal GPU. Run natively (see Quick Start Native) for ~5–10x speedup.

**Out of memory**
→ Reduce `MODEL_N_CTX` to `2048` or use a smaller quantization (Q3_K_M).

**FAISS index missing after container restart**
→ Ensure volume mounts are correct in `docker-compose.yml`. The `data/` directory must persist.

**Permission denied on model file**
→ The model directory is mounted as read-only (`:ro`). Ensure the file exists before starting the container.

---


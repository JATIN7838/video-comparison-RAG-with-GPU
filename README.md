# Video Comparison RAG — GPU Engine

GPU-backed **ingestion + retrieval engine** for a multi-agentic YouTube video analysis system.

This repo is the **retrieval + inference layer**. It is paired with a separate
**cognitive layer** repo built on the **Google ADK** framework (orchestration, memory,
multi-agent reasoning, response synthesis). That repo calls this service's HTTP
endpoints as tools — this repo owns the neural operations (embeddings,
cross-encoder reranking) and the vector database.

> **Two model backends** (`MODEL_BACKEND`):
> - **`local`** — loads the dense + cross-encoder models on a **CUDA GPU** (CPU works for
>   dev). Use when you have a GPU box.
> - **`huggingface`** — offloads dense + reranking to **HF Inference Endpoints**, so this
>   service runs on a **plain CPU instance** with no GPU bill. BM25/sparse always runs
>   locally on CPU in both modes.

---

## What it does

- **Ingestion** — YouTube URL → metadata + transcript → windowing → recursive chunking
  (overlap) → dense + sparse embeddings → Qdrant.
- **Retrieval** — hybrid search + fusion + reranking over the indexed transcripts.

It does **not** do reasoning or answer synthesis — that's the ADK repo's job.

---

## Endpoints

All protected endpoints require the header `X-API-Key: <RETRIEVAL_API_KEY>`.

| Method | Path        | Auth | Purpose |
|--------|-------------|------|---------|
| GET    | `/health`   | no   | Liveness check |
| POST   | `/ingest`   | yes  | Ingest a list of YouTube URLs (2 processed in parallel) |
| POST   | `/retrieve` | yes  | Retrieve in `metadata` or `chunks` mode |

### `/retrieve` — two modes

The agent picks a mode based on the question:

- **`metadata`** — returns only stored video metadata (title, channel, views, likes,
  duration, upload date) for the given `video_ids`. Used for performance/virality
  questions like *"Which video performed better?"* — these need **stats, not transcript
  chunks**, so we skip retrieval entirely (cheap + accurate).
- **`chunks`** — semantic transcript retrieval for a natural-language `query`,
  optionally scoped to specific `video_ids`.

#### Why metadata-first
Comparison/performance queries are answered from numbers (views, likes, duration),
not text. Returning metadata directly avoids needless embedding + reranking cost and
prevents the LLM from hallucinating stats out of transcript text.

---

## Retrieval strategy (`chunks` mode)

```
Query
 ├── Dense retriever  (vector / semantic)
 ├── Sparse retriever (BM25, IDF)
 ↓
RRF fusion          (Reciprocal Rank Fusion, server-side in Qdrant)
 ↓
Cross-Encoder rerank
 ↓
Top-K results
```

1. **Dense** — top-15 candidates via cosine similarity.
2. **Sparse (BM25)** — top-15 via Qdrant native sparse vectors (`Modifier.IDF`).
3. **RRF fusion** — both lists fused server-side in a single Qdrant query.
4. **Cross-encoder rerank** — the 15 fused candidates re-scored by a cross-encoder.
5. **Top-K** — best 3 returned (`candidate_k` / `top_k` are tunable per request).

Each result carries `rerank_score` (final ranking signal), `rrf_score`, the chunk
`text`, and metadata (`video_id`, `title`, `channel`, `start_time`, `end_time`, …) for
timestamp-accurate references.

---

## Models

| Role | Model | `local` | `huggingface` |
|------|-------|---------|---------------|
| Dense embeddings | `BAAI/bge-large-en-v1.5` (1024-dim) | on GPU/CPU | HF TEI `/embed` endpoint |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | on GPU/CPU | HF TEI `/rerank` endpoint |
| Sparse / BM25 | `Qdrant/bm25` (FastEmbed) | local CPU | local CPU (same) |

BM25 isn't a neural model — it's a CPU tokenizer (term frequencies; IDF computed inside
Qdrant) with **WordNet lemmatization** (FastEmbed's stemmer is disabled). It never needs
a GPU or an endpoint, so it stays local in both backends.

> **Model hosting:** in `local` mode models download at startup. In `huggingface` mode
> they are served from **HF Inference Endpoints** (use scale-to-zero so you only pay
> per request) — this lets the repo deploy on a CPU box with no idle GPU cost.

---

## Stack

- **FastAPI** — HTTP layer
- **Qdrant** — vector DB (named dense + sparse vectors, native RRF)
- **SentenceTransformers + PyTorch (CUDA 12.8)** — local dense + cross-encoder (`gpu` extra only)
- **HF Inference Endpoints (TEI)** — dense + reranking in `huggingface` mode
- **FastEmbed + NLTK** — sparse BM25 with lemmatization (always local)
- **yt-dlp + youtube-transcript-api** — metadata + transcripts
- **LangChain text splitters** — `RecursiveCharacterTextSplitter` (chunk 800 / overlap 200)

---

## Setup

**Local (GPU) backend** — `MODEL_BACKEND=local`:

```bash
uv sync --extra gpu          # base deps + torch (cu128) + sentence-transformers
cp .env.example .env         # set RETRIEVAL_API_KEY, QDRANT_URL, etc.
uv run python main.py        # serves on :9000
```

**Hugging Face backend** — `MODEL_BACKEND=huggingface` (no GPU, no torch):

```bash
uv sync                      # base deps only (CPU box)
cp .env.example .env         # set MODEL_BACKEND=huggingface + HF_* vars below
uv run python main.py
```

First `local` start downloads models + NLTK assets (slow); later starts are fast.

### Configuration (`.env`)

| Var | Default | Purpose |
|-----|---------|---------|
| `MODEL_BACKEND` | `local` | `local` (GPU models) or `huggingface` (HF endpoints) |
| `RETRIEVAL_API_KEY` | — | Shared secret for `X-API-Key`. Unset → auth disabled (dev only) |
| `ENVIRONMENT` | `dev` | `dev*` enables hot reload |
| `QDRANT_URL` | — | Qdrant server URL; unset → embedded on-disk store |
| `EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | dense model (`local` mode) |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | reranker (`local` mode) |
| `HF_TOKEN` | — | HF auth (`huggingface` mode) |
| `HF_EMBEDDING_ENDPOINT_URL` | — | TEI embeddings endpoint (`huggingface` mode) |
| `HF_RERANKER_ENDPOINT_URL` | — | TEI rerank endpoint (`huggingface` mode) |
| `EMBEDDING_DIM` | `1024` | Vector size used when no local model is loaded |

See [`app/config.py`](app/config.py) for the full list (chunking, candidate/top-K, RRF, retries).

---

## Testing

Run the service, then open [`notebooks/test_endpoints.ipynb`](notebooks/test_endpoints.ipynb)
— it exercises ingestion, both retrieval modes, and the API-key header.

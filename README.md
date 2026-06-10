# Semantic Search Engine

A production-grade semantic search engine built during internship.
Implements hybrid search combining dense semantic embeddings with
sparse keyword matching, cross-encoder reranking, and Redis caching.

## Architecture

```
Raw Documents
     │
     ▼
Document Cleaning (noise removal, normalization)
     │
     ▼
Chunking (RecursiveCharacterTextSplitter, 512 chars)
     │
     ▼
Dual Embedding
├── Dense:  BAAI/bge-base-en-v1.5 (768-dim)
└── Sparse: SPLADE (prithivida/Splade_PP_en_v1)
     │
     ▼
Qdrant Vector Database (hybrid collection)
     │
     ▼
Hybrid Search (RRF fusion: dense + sparse)
     │
     ▼
Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)
     │
     ▼
Redis Cache (TTL: 1 hour)
     │
     ▼
FastAPI REST Layer (/search, /ingest, /health)
```

## Tech Stack

| Component     | Technology                          |
|---------------|-------------------------------------|
| Embeddings    | BAAI/bge-base-en-v1.5 (HuggingFace) |
| Sparse model  | SPLADE (fastembed)                  |
| Vector DB     | Qdrant (Docker)                     |
| Reranker      | cross-encoder/ms-marco-MiniLM-L-6-v2|
| API           | FastAPI + Uvicorn                   |
| Cache         | Redis (Docker)                      |
| Framework     | LangChain                           |

## Setup

```bash
# 1. Start infrastructure
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
docker run -d -p 6379:6379 --name redis redis:alpine

# 2. Install dependencies
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 3. Build index
python main.py ingest

# 4. Start API
python main.py api
# → Swagger UI: http://localhost:8000/docs
```

## Evaluation Results (25 queries, K=5)

| Metric      | Hybrid-only | Hybrid+Reranked | Delta    |
|-------------|-------------|-----------------|----------|
| Precision@5 | 0.9600      | 0.9600          | +0.0000  |
| Recall@5    | 4.8000      | 4.8000          | +0.0000  |
| MRR         | 0.9680      | 0.9700          | +0.0020  |
| NDCG@5      | 2.8305      | 2.8277          | -0.0028  |
| Avg Latency | 3083ms      | 2105ms          | -978ms   |

## Cache Performance

| Request type | Avg latency |
|--------------|-------------|
| Cache miss   | ~2000ms     |
| Cache hit    | <5ms        |
| Speedup      | ~400x       |

## Project Structure

```
semantic_search/
├── config.py        # all settings
├── cleaner.py       # document cleaning
├── ingest.py        # load → clean → embed → store
├── search.py        # hybrid search + reranking
├── reranker.py      # cross-encoder wrapper
├── cache.py         # Redis caching layer
├── models.py        # Pydantic schemas
├── api.py           # FastAPI application
├── evaluate.py      # evaluation metrics
├── main.py          # entry point
├── eval_data.json   # evaluation dataset
└── data/            # source documents
```
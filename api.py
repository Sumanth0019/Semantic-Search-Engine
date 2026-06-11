from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
# from fastembed import SparseTextEmbedding
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue,
    SparseVector, FusionQuery, Fusion, Prefetch
)
from models import (
    SearchRequest, SearchResponse, SearchResult,
    HealthResponse, IngestResponse
)
import reranker as reranker_module
import ingest as ingest_module
import config
import time
import cache as cache_module
import os
import shutil
from fastapi import UploadFile, File
from pydantic import BaseModel as PydanticBase

# ── startup / shutdown ──────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models at startup...")
    app.state.dense_embedder = HuggingFaceEmbeddings(
        model_name=config.DENSE_MODEL,
        model_kwargs={"device": config.MODEL_DEVICE},
        encode_kwargs={"normalize_embeddings": True}
    )
    #app.state.sparse_embedder = None
    
    #app.state.reranker = reranker_module.get_reranker()
    app.state.qdrant   = QdrantClient(url=config.QDRANT_URL,api_key=config.QDRANT_API_KEY)
    print("All models loaded. API ready.")
    yield
    print("Shutting down.")

# ── app ──────────────────────────────────────────────────────
app = FastAPI(
    title="Semantic Search API",
    description=(
        "Production-grade semantic search using "
        "BGE embeddings, Qdrant hybrid search, "
        "and cross-encoder reranking."
    ),
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── helpers ──────────────────────────────────────────────────
def run_hybrid_search(app_state, query: str,
                      k: int, topic_filter=None):
    dense_vec  = app_state.dense_embedder.embed_query(query)
    sparse_res = None

    query_filter = None
    if topic_filter:
        query_filter = Filter(
            must=[FieldCondition(
                key="topic",
                match=MatchValue(value=topic_filter)
            )]
        )

    results = app_state.qdrant.query_points(
        collection_name=config.COLLECTION_NAME,
        query=dense_vec,
        using="dense",
        limit=k,
        query_filter=query_filter,
        with_payload=True
    ).points
    
    return results

# ── endpoints ────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    try:
        info  = request.app.state.qdrant.get_collection(
            config.COLLECTION_NAME
        )
        count = info.points_count or 0
    except Exception:
        count = 0
    return HealthResponse(
        status="ok",
        dense_model=config.DENSE_MODEL,
        reranker_model=config.RERANKER_MODEL,
        collection=config.COLLECTION_NAME,
        vector_count=count
    )

@app.post("/search", response_model=SearchResponse)
async def search(
    req: SearchRequest,
    request: Request
):
    try:
        t0 = time.time()

        # Check cache first
        cached = cache_module.get_cached(
            req.query, req.top_k, req.use_reranking
        )
        if cached:
            cached["cache_hit"]   = True
            cached["latency_ms"]  = round(
                (time.time() - t0) * 1000, 2
            )
            return cached

        # Run full search pipeline
        raw = run_hybrid_search(
            request.app.state,
            req.query,
            k=config.TOP_K,
            topic_filter=req.topic_filter
        )

        if req.use_reranking:
            ranked  = reranker_module.rerank(
                req.query, raw, top_k=req.top_k
            )
            results = [
                SearchResult(
                    text=r["result"].payload.get("text",""),
                    source=r["result"].payload.get("doc_id",""),
                    topic=r["result"].payload.get("topic",""),
                    score=round(r["result"].score, 4),
                    rerank_score=round(r["rerank_score"], 4),
                    chunk_index=r["result"].payload.get(
                        "chunk_index", 0),
                    word_count=r["result"].payload.get(
                        "word_count", 0),
                ) for r in ranked
            ]
            search_type = "hybrid+reranked"
        else:
            results = [
                SearchResult(
                    text=r.payload.get("text",""),
                    source=r.payload.get("doc_id",""),
                    topic=r.payload.get("topic",""),
                    score=round(r.score, 4),
                    chunk_index=r.payload.get("chunk_index",0),
                    word_count=r.payload.get("word_count",0),
                ) for r in raw[:req.top_k]
            ]
            search_type = "hybrid"

        latency = round((time.time() - t0) * 1000, 2)
        response = SearchResponse(
            query=req.query,
            results=results,
            total_results=len(results),
            search_type=search_type
        )

        # Store in cache
        resp_dict = response.model_dump()
        resp_dict["cache_hit"]  = False
        resp_dict["latency_ms"] = latency
        cache_module.set_cache(
            req.query, req.top_k,
            req.use_reranking, resp_dict
        )

        resp_dict["latency_ms"] = latency
        return resp_dict

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=str(e)
        )

@app.post("/ingest", response_model=IngestResponse)
async def ingest():
    try:
        docs   = ingest_module.load_documents()
        docs   = ingest_module.clean_documents(docs)
        chunks = ingest_module.split_documents(docs)
        ingest_module.run()
        return IngestResponse(
            status="success",
            documents_loaded=len(docs),
            chunks_created=len(chunks),
            vectors_stored=len(chunks)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=str(e)
        )
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        os.makedirs(config.DATA_DIR, exist_ok=True)
        save_path = os.path.join(config.DATA_DIR, file.filename)
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        docs   = ingest_module.load_documents()
        docs   = ingest_module.clean_documents(docs)
        chunks = ingest_module.split_documents(docs)
        ingest_module.run()
        
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_created": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class UploadSearchRequest(PydanticBase):
    query: str
    session_id: str
    top_k: int = 5
    use_reranking: bool = True

@app.post("/search-upload")
async def search_upload(req: UploadSearchRequest, request: Request):
    try:
        raw = run_hybrid_search(
            request.app.state,
            req.query,
            k=config.TOP_K
        )

        if req.use_reranking and raw:
            ranked = reranker_module.rerank(req.query, raw, top_k=req.top_k)
            results = [{
                "text":         r["result"].payload.get("text", ""),
                "source":       r["result"].payload.get("doc_id", ""),
                "topic":        r["result"].payload.get("topic", ""),
                "score":        round(r["result"].score, 4),
                "rerank_score": round(r["rerank_score"], 4),
                "word_count":   r["result"].payload.get("word_count", 0),
            } for r in ranked]
            search_type = "upload+hybrid+reranked"
        else:
            results = [{
                "text":       r.payload.get("text", ""),
                "source":     r.payload.get("doc_id", ""),
                "topic":      r.payload.get("topic", ""),
                "score":      round(r.score, 4),
                "word_count": r.payload.get("word_count", 0),
            } for r in raw[:req.top_k]]
            search_type = "upload+hybrid"

        return {"results": results, "search_type": search_type}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

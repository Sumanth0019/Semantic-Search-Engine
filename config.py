import os

DATA_DIR          = "data"

QDRANT_URL      = os.getenv("QDRANT_URL", "http://localhost:6333")
REDIS_HOST      = os.getenv("REDIS_HOST", "localhost")
COLLECTION_NAME = "semantic_search"

DENSE_MODEL       = "BAAI/bge-base-en-v1.5"
SPARSE_MODEL      = "prithivida/Splade_PP_en_v1"
RERANKER_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MODEL_DEVICE      = "cpu"
NORMALIZE_EMBEDS  = True

CHUNK_SIZE        = 512
CHUNK_OVERLAP     = 64

TOP_K             = 20
RERANK_TOP_K      = 5
SCORE_THRESHOLD   = 0.4
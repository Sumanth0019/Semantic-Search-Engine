from pydantic import BaseModel, Field
from typing import Optional, List

class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        example="how does solar energy work?"
    )
    top_k: Optional[int] = Field(
        default=5, ge=1, le=20,
        description="Number of results to return"
    )
    topic_filter: Optional[str] = Field(
        default=None,
        description="Filter results to a specific topic",
        example="Solar energy"
    )
    use_reranking: Optional[bool] = Field(
        default=True,
        description="Apply cross-encoder reranking"
    )

class SearchResult(BaseModel):
    text: str
    source: str
    topic: str
    score: float
    rerank_score: Optional[float] = None
    chunk_index: int
    word_count: int

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    search_type: str
    cache_hit: bool = False
    latency_ms: float = 0.0
    
class HealthResponse(BaseModel):
    status: str
    dense_model: str
    reranker_model: str
    collection: str
    vector_count: int
    cache_available: bool = False

class IngestResponse(BaseModel):
    status: str
    documents_loaded: int
    chunks_created: int
    vectors_stored: int
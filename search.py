from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue,
    SparseVector, FusionQuery, Fusion,
    Prefetch)
from fastembed import SparseTextEmbedding
import reranker
import config

_dense_embedder  = None
_sparse_embedder = None

def get_embedder():
    global _dense_embedder
    if _dense_embedder is None:
        _dense_embedder = HuggingFaceEmbeddings(
            model_name=config.DENSE_MODEL,
            model_kwargs={"device": config.MODEL_DEVICE},
            encode_kwargs={"normalize_embeddings": config.NORMALIZE_EMBEDS}
        )
    return _dense_embedder

def get_sparse_embedder():
    global _sparse_embedder
    if _sparse_embedder is None:
        _sparse_embedder = SparseTextEmbedding(
            model_name=config.SPARSE_MODEL
        )
    return _sparse_embedder

def hybrid_search(query: str, k: int = None,
                  topic_filter: str = None) -> list:
    k = k or config.TOP_K
    client     = QdrantClient(url=config.QDRANT_URL)
    dense_emb  = get_embedder()
    sparse_emb = get_sparse_embedder()

    dense_vec  = dense_emb.embed_query(query)
    sparse_res = list(sparse_emb.embed([query]))[0]

    query_filter = None
    if topic_filter:
        query_filter = Filter(
            must=[FieldCondition(
                key="topic",
                match=MatchValue(value=topic_filter)
            )]
        )

    results = client.query_points(
        collection_name=config.COLLECTION_NAME,
        prefetch=[
            Prefetch(
                query=dense_vec,
                using="dense",
                limit=k
            ),
            Prefetch(
                query=SparseVector(
                    indices=sparse_res.indices.tolist(),
                    values=sparse_res.values.tolist()
                ),
                using="sparse",
                limit=k
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=k,
        query_filter=query_filter,
        with_payload=True
    ).points
    return results

def full_search(query: str,
                topic_filter: str = None) -> list:
    raw     = hybrid_search(query,
                            k=config.TOP_K,
                            topic_filter=topic_filter)
    ranked  = reranker.rerank(query, raw,
                              top_k=config.RERANK_TOP_K)
    return ranked

def format_results(ranked_results, query):
    print(f"\nQuery  : '{query}'")
    print("-" * 60)
    if not ranked_results:
        print("No results found.")
        return
    for i, item in enumerate(ranked_results):
        r = item["result"]
        rs = item["rerank_score"]
        print(f"Result {i+1} | "
              f"Hybrid score: {r.score:.4f} | "
              f"Rerank score: {rs:.4f}")
        print(f"Topic  : {r.payload.get('topic','')}")
        print(f"Source : {r.payload.get('doc_id','')}")
        print(f"Content: "
              f"{r.payload.get('text','')[:220]}...")
        print()

if __name__ == "__main__":
    print("Loading models (first run downloads reranker)...")
    reranker.get_reranker()
    print("Ready.\n")
    while True:
        query = input("Search: ").strip()
        if query.lower() in ("quit","exit","q"):
            break
        if not query:
            continue
        results = full_search(query)
        format_results(results, query)
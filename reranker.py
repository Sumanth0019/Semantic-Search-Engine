from sentence_transformers import CrossEncoder
import config

_model = None

def get_reranker():
    global _model
    if _model is None:
        print("  Loading cross-encoder reranker...")
        _model = CrossEncoder(config.RERANKER_MODEL)
    return _model

def rerank(query: str, results: list,
           top_k: int = None) -> list:
    top_k = top_k or config.RERANK_TOP_K
    if not results:
        return []
    model = get_reranker()
    pairs = [
        (query, r.payload.get("text", ""))
        for r in results
    ]
    scores = model.predict(pairs)
    scored = sorted(
        zip(results, scores),
        key=lambda x: x[1],
        reverse=True
    )
    return [
        {
            "result":        r,
            "rerank_score":  float(s),
        }
        for r, s in scored[:top_k]
    ]
import json
import math
import time
from search import hybrid_search
from reranker import rerank
import config

def load_eval_data(path="eval_data.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def precision_at_k(retrieved_topics, relevant_topics, k):
    retrieved_k = retrieved_topics[:k]
    hits = sum(1 for t in retrieved_k
               if t in relevant_topics)
    return hits / k if k > 0 else 0.0

def recall_at_k(retrieved_topics, relevant_topics, k):
    retrieved_k = retrieved_topics[:k]
    hits = sum(1 for t in retrieved_k
               if t in relevant_topics)
    return hits / len(relevant_topics) \
           if relevant_topics else 0.0

def reciprocal_rank(retrieved_topics, relevant_topics):
    for i, t in enumerate(retrieved_topics):
        if t in relevant_topics:
            return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k(retrieved_topics, relevant_topics, k):
    dcg = 0.0
    for i, t in enumerate(retrieved_topics[:k]):
        rel = 1 if t in relevant_topics else 0
        dcg += rel / math.log2(i + 2)
    ideal_hits = min(len(relevant_topics), k)
    idcg = sum(1 / math.log2(i + 2)
               for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0

def run_evaluation(eval_data, k=5,
                   use_reranking=True, verbose=False):
    # preload models once before the loop
    from langchain_huggingface import HuggingFaceEmbeddings
    from fastembed import SparseTextEmbedding

    print("  Preloading models...")
    _dense  = HuggingFaceEmbeddings(
        model_name=config.DENSE_MODEL,
        model_kwargs={"device": config.MODEL_DEVICE},
        encode_kwargs={"normalize_embeddings": True}
    )
    _sparse = SparseTextEmbedding(
        model_name=config.SPARSE_MODEL
    )
    print("  Models ready.\n")
    ...

def run_evaluation(eval_data, k=5,
                   use_reranking=True, verbose=False):
    p_scores, r_scores = [], []
    rr_scores, ndcg_scores = [], []
    total_time = 0.0

    for item in eval_data:
        query    = item["query"]
        relevant = set(item["relevant_topics"])

        t0  = time.time()
        raw = hybrid_search(query, k=config.TOP_K)

        if use_reranking:
            ranked = rerank(query, raw, top_k=k)
            retrieved_topics = [
                r["result"].payload.get("topic", "")
                for r in ranked
            ]
        else:
            retrieved_topics = [
                r.payload.get("topic", "")
                for r in raw[:k]
            ]
        elapsed = (time.time() - t0) * 1000
        total_time += elapsed

        p  = precision_at_k(retrieved_topics, relevant, k)
        r  = recall_at_k(retrieved_topics, relevant, k)
        rr = reciprocal_rank(retrieved_topics, relevant)
        nd = ndcg_at_k(retrieved_topics, relevant, k)

        p_scores.append(p)
        r_scores.append(r)
        rr_scores.append(rr)
        ndcg_scores.append(nd)

        if verbose:
            status = "HIT" if rr > 0 else "MISS"
            print(f"  [{status}] {query[:50]:<50} "
                  f"P:{p:.2f} R:{r:.2f} "
                  f"RR:{rr:.2f} NDCG:{nd:.2f}")

    n = len(eval_data)
    return {
        "precision": round(sum(p_scores) / n, 4),
        "recall":    round(sum(r_scores) / n, 4),
        "mrr":       round(sum(rr_scores) / n, 4),
        "ndcg":      round(sum(ndcg_scores) / n, 4),
        "avg_latency_ms": round(total_time / n, 1),
        "queries_tested": n,
    }

def print_results(metrics, label):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Precision@5    : {metrics['precision']:.4f}")
    print(f"  Recall@5       : {metrics['recall']:.4f}")
    print(f"  MRR            : {metrics['mrr']:.4f}")
    print(f"  NDCG@5         : {metrics['ndcg']:.4f}")
    print(f"  Avg latency    : {metrics['avg_latency_ms']} ms")
    print(f"  Queries tested : {metrics['queries_tested']}")

def print_comparison(with_rr, without_rr):
    print(f"\n{'='*50}")
    print(f"  Comparison: Reranked vs Hybrid-only")
    print(f"{'='*50}")
    print(f"  {'Metric':<16} {'Hybrid':>10} "
          f"{'Reranked':>10} {'Delta':>10}")
    print(f"  {'-'*46}")
    metrics = ["precision","recall","mrr","ndcg"]
    labels  = ["Precision@5","Recall@5","MRR","NDCG@5"]
    for m, l in zip(metrics, labels):
        base  = without_rr[m]
        rerank = with_rr[m]
        delta = rerank - base
        sign  = "+" if delta >= 0 else ""
        print(f"  {l:<16} {base:>10.4f} "
              f"{rerank:>10.4f} "
              f"{sign}{delta:>9.4f}")
    lat_base   = without_rr['avg_latency_ms']
    lat_rerank = with_rr['avg_latency_ms']
    print(f"  {'Latency (ms)':<16} {lat_base:>10.1f} "
          f"{lat_rerank:>10.1f}")

if __name__ == "__main__":
    print("Loading evaluation data...")
    eval_data = load_eval_data()
    print(f"Loaded {len(eval_data)} queries\n")

    print("Running evaluation WITHOUT reranking...")
    metrics_base = run_evaluation(
        eval_data, k=5,
        use_reranking=False,
        verbose=True
    )
    print_results(metrics_base, "Hybrid search only")

    print("\nRunning evaluation WITH reranking...")
    metrics_reranked = run_evaluation(
        eval_data, k=5,
        use_reranking=True,
        verbose=True
    )
    print_results(metrics_reranked, "Hybrid + Reranking")

    print_comparison(metrics_reranked, metrics_base)

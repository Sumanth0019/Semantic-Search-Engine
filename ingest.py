import os
import uuid
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from cleaner import clean_documents
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    SparseVectorParams, SparseIndexParams, SparseVector
)
from fastembed import SparseTextEmbedding
from cleaner import clean_documents
import config

def load_documents():
    docs = []
    for fname in sorted(os.listdir(config.DATA_DIR)):
        if fname.endswith(".txt"):
            path = os.path.join(config.DATA_DIR, fname)
            loader = TextLoader(path, encoding="utf-8")
            docs.extend(loader.load())
    print(f"  Loaded {len(docs)} documents")
    return docs

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"  Total chunks: {len(chunks)}")
    return chunks

def get_dense_embedder():
    print("  Loading dense model (bge-base)...")
    return HuggingFaceEmbeddings(
        model_name=config.DENSE_MODEL,
        model_kwargs={"device": config.MODEL_DEVICE},
        encode_kwargs={"normalize_embeddings": config.NORMALIZE_EMBEDS}
    )

def get_sparse_embedder():
    print("  Loading sparse model (SPLADE)...")
    return SparseTextEmbedding(
        model_name=config.SPARSE_MODEL
    )

def setup_hybrid_collection(client, vector_size):
    existing = [c.name for c in
                client.get_collections().collections]
    if config.COLLECTION_NAME in existing:
        client.delete_collection(config.COLLECTION_NAME)
        print("  Deleted old collection")
    client.create_collection(
        collection_name=config.COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            )
        }
    )
    print(f"  Created hybrid collection: "
          f"{config.COLLECTION_NAME}")

def extract_topic(source_path: str) -> str:
    fname = os.path.basename(source_path)
    name = fname.replace(".txt", "")
    parts = name.rsplit("_", 1)
    topic = parts[0].replace("_", " ") if parts else name
    return topic

def store_chunks(client, chunks,
                 dense_embedder, sparse_embedder):
    texts = [c.page_content for c in chunks]
    print(f"  Generating dense embeddings "
          f"for {len(texts)} chunks...")
    dense_vectors = dense_embedder.embed_documents(texts)

    print(f"  Generating sparse embeddings...")
    sparse_results = list(sparse_embedder.embed(texts))

    print(f"  Building points...")
    points = []
    for i, (chunk, dense_vec, sparse_res) in enumerate(
        zip(chunks, dense_vectors, sparse_results)
    ):
        source = chunk.metadata.get("source", "")
        topic = extract_topic(source)
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": dense_vec,
                "sparse": SparseVector(
                    indices=sparse_res.indices.tolist(),
                    values=sparse_res.values.tolist()
                )
            },
            payload={
                "text":        chunk.page_content,
                "source":      source,
                "doc_id":      os.path.basename(source),
                "topic":       topic,
                "chunk_index": i,
                "char_count":  len(chunk.page_content),
                "word_count":  len(chunk.page_content.split()),
                "ingested_at": datetime.utcnow().isoformat(),
            }
        ))

    batch_size = 100
    total_batches = (len(points) - 1) // batch_size + 1
    for start in range(0, len(points), batch_size):
        batch = points[start:start + batch_size]
        client.upsert(
            collection_name=config.COLLECTION_NAME,
            points=batch
        )
        print(f"  Stored batch "
              f"{start // batch_size + 1}/{total_batches}")
    print(f"  Total stored: {len(points)} vectors")

def run():
    print("=== Ingestion Pipeline (Hybrid) ===\n")

    print("[1/5] Loading documents...")
    docs = load_documents()

    print("\n[2/5] Cleaning documents...")
    docs = clean_documents(docs)

    print("\n[3/5] Splitting into chunks...")
    chunks = split_documents(docs)

    print("\n[4/5] Loading embedding models...")
    dense_embedder  = get_dense_embedder()
    sparse_embedder = get_sparse_embedder()
    sample = dense_embedder.embed_query("test")
    print(f"  Dense vector size : {len(sample)}")

    print("\n[5/5] Storing in Qdrant...")
    client = QdrantClient(url=config.QDRANT_URL,api_key=config.QDRANT_API_KEY)
    setup_hybrid_collection(client, len(sample))
    store_chunks(client, chunks,
                 dense_embedder, sparse_embedder)

    info = client.get_collection(config.COLLECTION_NAME)
    print(f"\n  Final vector count: {info.points_count}")
    print("\nIngestion complete.")

if __name__ == "__main__":
    run()

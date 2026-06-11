import os
import uuid
import tempfile
from datetime import datetime
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader
)
try:
    from langchain_community.document_loaders import Docx2txtLoader
except ImportError:
    Docx2txtLoader = None

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector
from cleaner import clean_documents
import config

def load_uploaded_file(file_path: str,
                       filename: str) -> list:
    ext = filename.lower().split(".")[-1]
    if ext == "pdf":
        loader = PyPDFLoader(file_path)
    elif ext in ("docx", "doc"):
        if Docx2txtLoader is None:
            raise ValueError(
                "Install python-docx: "
                "pip install python-docx"
            )
        loader = Docx2txtLoader(file_path)
    elif ext == "txt":
        loader = TextLoader(
            file_path, encoding="utf-8"
        )
    else:
        raise ValueError(
            f"Unsupported file type: .{ext}. "
            f"Supported: pdf, docx, txt"
        )
    return loader.load()

def ingest_uploaded_file(
    file_bytes: bytes,
    filename: str,
    session_id: str,
    dense_embedder,
    sparse_embedder
) -> dict:
    ext = filename.lower().split(".")[-1]

    # Save to temp file
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=f".{ext}"
    ) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        # Load
        docs = load_uploaded_file(tmp_path, filename)

        # Clean
        docs = clean_documents(docs)

        # Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " "]
        )
        chunks = splitter.split_documents(docs)

        if not chunks:
            return {
                "status": "error",
                "message": "No content extracted"
            }

        # Embed
        texts = [c.page_content for c in chunks]
        dense_vectors = dense_embedder.embed_documents(texts)
        sparse_results = list(
            sparse_embedder.embed(texts)
        )

        # Store with session_id in payload
        client = QdrantClient(url=config.QDRANT_URL,api_key=config.QDRANT_API_KEY)
        points = []
        for i, (chunk, dv, sv) in enumerate(
            zip(chunks, dense_vectors, sparse_results)
        ):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dv,
                    "sparse": SparseVector(
                        indices=sv.indices.tolist(),
                        values=sv.values.tolist()
                    )
                },
                payload={
                    "text":       chunk.page_content,
                    "source":     filename,
                    "doc_id":     filename,
                    "topic":      filename,
                    "session_id": session_id,
                    "chunk_index": i,
                    "char_count": len(chunk.page_content),
                    "word_count": len(
                        chunk.page_content.split()
                    ),
                    "ingested_at": datetime.utcnow().isoformat(),
                    "is_upload":  True
                }
            ))

        # Batch upsert
        for start in range(0, len(points), 100):
            client.upsert(
                collection_name=config.COLLECTION_NAME,
                points=points[start:start+100]
            )

        return {
            "status":         "success",
            "filename":       filename,
            "chunks_created": len(chunks),
            "session_id":     session_id,
            "pages":          len(docs)
        }

    finally:
        os.unlink(tmp_path)

def search_uploaded_doc(
    query: str,
    session_id: str,
    dense_embedder,
    sparse_embedder,
    k: int = 5
) -> list:
    from qdrant_client.models import (
        Filter, FieldCondition, MatchValue,
        Prefetch, FusionQuery, Fusion
    )

    client     = QdrantClient(url=config.QDRANT_URL,api_key=config.QDRANT_API_KEY)
    dense_vec  = dense_embedder.embed_query(query)
    sparse_res = list(
        sparse_embedder.embed([query])
    )[0]

    session_filter = Filter(
        must=[FieldCondition(
            key="session_id",
            match=MatchValue(value=session_id)
        )]
    )

    results = client.query_points(
        collection_name=config.COLLECTION_NAME,
        prefetch=[
            Prefetch(
                query=dense_vec,
                using="dense",
                limit=k * 2,
                filter=session_filter
            ),
            Prefetch(
                query=SparseVector(
                    indices=sparse_res.indices.tolist(),
                    values=sparse_res.values.tolist()
                ),
                using="sparse",
                limit=k * 2,
                filter=session_filter
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=k,
        with_payload=True
    ).points
    return results

def delete_session(session_id: str):
    from qdrant_client.models import (
        Filter, FieldCondition, MatchValue,
        FilterSelector
    )
    client = QdrantClient(url=config.QDRANT_URL,api_key=config.QDRANT_API_KEY)
    client.delete(
        collection_name=config.COLLECTION_NAME,
        points_selector=FilterSelector(
            filter=Filter(
                must=[FieldCondition(
                    key="session_id",
                    match=MatchValue(value=session_id)
                )]
            )
        )
    )

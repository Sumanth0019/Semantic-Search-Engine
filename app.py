import streamlit as st
import requests
import time
import uuid
from uploader import (
    ingest_uploaded_file,
    search_uploaded_doc,
    delete_session
)
from langchain_huggingface import HuggingFaceEmbeddings
from fastembed import SparseTextEmbedding
from reranker import get_reranker, rerank
import config

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Semantic Search Engine",
    page_icon="🔍",
    layout="wide"
)

# ── load models once at app startup ──────────────────────────
@st.cache_resource
def load_models():
    dense = HuggingFaceEmbeddings(
        model_name=config.DENSE_MODEL,
        model_kwargs={"device": config.MODEL_DEVICE},
        encode_kwargs={"normalize_embeddings": True}
    )
    sparse  = SparseTextEmbedding(
        model_name=config.SPARSE_MODEL
    )
    reranker = get_reranker()
    return dense, sparse, reranker

# ── session state ─────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id  = None
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "search_mode" not in st.session_state:
    st.session_state.search_mode = "knowledge_base"

# ── header ────────────────────────────────────────────────────
st.title("Semantic Search Engine")
st.caption(
    "BGE-base embeddings · Qdrant hybrid search · "
    "Cross-encoder reranking · Redis cache"
)

# ── sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header(" Settings")
    top_k        = st.slider("Results", 1, 10, 5)
    use_reranking = st.toggle("Reranking", value=True)

    st.divider()
    st.header(" Search Mode")
    mode = st.radio(
        "Search over:",
        ["Knowledge base", "My uploaded document"],
        index=0
    )
    st.session_state.search_mode = (
        "upload" if mode == "My uploaded document"
        else "knowledge_base"
    )

    st.divider()
    st.header(" System Status")
    try:
        health = requests.get(
            f"{API_URL}/health", timeout=3
        ).json()
        st.success(" API connected")
        st.metric("Vectors indexed",
                  health.get("vector_count", 0))
    except Exception:
        st.error(" API offline")
        st.caption("Run: python main.py api")

# ── document upload panel ─────────────────────────────────────
if st.session_state.search_mode == "upload":
    st.subheader(" Upload Your Document")

    uploaded = st.file_uploader(
        "Drop a file here",
        type=["pdf", "txt", "docx"],
        help="Supported: PDF, TXT, DOCX"
    )

    if uploaded is not None:
        if (st.session_state.uploaded_file
                != uploaded.name):

            with st.spinner(
                f"Processing {uploaded.name}..."
            ):
                # Delete previous session if exists
                if st.session_state.session_id:
                    try:
                        delete_session(
                            st.session_state.session_id
                        )
                    except Exception:
                        pass

                # New session
                session_id = str(uuid.uuid4())[:8]
                dense, sparse, _ = load_models()

                result = ingest_uploaded_file(
                    file_bytes=uploaded.getvalue(),
                    filename=uploaded.name,
                    session_id=session_id,
                    dense_embedder=dense,
                    sparse_embedder=sparse
                )

                if result["status"] == "success":
                    st.session_state.session_id    = session_id
                    st.session_state.uploaded_file = uploaded.name
                    st.success(
                        f"✅**{uploaded.name}** processed!\n\n"
                        f"- Pages: {result['pages']}\n"
                        f"- Chunks created: "
                        f"{result['chunks_created']}\n"
                        f"- Session: {session_id}"
                    )
                else:
                    st.error(result.get("message",
                                        "Processing failed"))
        else:
            st.info(
                f" **{uploaded.name}** already loaded. "
                f"Ready to search."
            )

    if st.session_state.session_id:
        if st.button(" Clear document"):
            delete_session(st.session_state.session_id)
            st.session_state.session_id   = None
            st.session_state.uploaded_file = None
            st.rerun()

# ── search bar ────────────────────────────────────────────────
st.divider()

if st.session_state.search_mode == "upload":
    placeholder = ("Ask anything about your document...")
    disabled    = st.session_state.session_id is None
    if disabled:
        st.warning("Upload a document first to search it.")
else:
    placeholder = "how does solar energy affect temperature?"
    disabled    = False

query = st.text_input(
    "Search",
    placeholder=placeholder,
    disabled=disabled,
    label_visibility="collapsed"
)

search_btn = st.button(
    "🔍 Search", type="primary",
    disabled=disabled or not query
)

# ── results ───────────────────────────────────────────────────
if search_btn and query:
    t0 = time.time()

    with st.spinner("Searching..."):
        try:
            if st.session_state.search_mode == "upload":
                # Search uploaded document directly
                dense, sparse, reranker_model = load_models()

                raw = search_uploaded_doc(
                    query=query,
                    session_id=st.session_state.session_id,
                    dense_embedder=dense,
                    sparse_embedder=sparse,
                    k=top_k * 2
                )

                if use_reranking and raw:
                    ranked = rerank(
                        query, raw, top_k=top_k
                    )
                    results = [
                        {
                            "text":   r["result"].payload.get("text",""),
                            "source": r["result"].payload.get("source",""),
                            "topic":  r["result"].payload.get("topic",""),
                            "score":  round(r["result"].score, 4),
                            "rerank_score": round(
                                r["rerank_score"], 4
                            ),
                            "word_count": r["result"].payload.get(
                                "word_count", 0
                            ),
                        }
                        for r in ranked
                    ]
                    search_type = "upload+hybrid+reranked"
                else:
                    results = [
                        {
                            "text":   r.payload.get("text",""),
                            "source": r.payload.get("source",""),
                            "topic":  r.payload.get("topic",""),
                            "score":  round(r.score, 4),
                            "word_count": r.payload.get(
                                "word_count", 0
                            ),
                        }
                        for r in raw[:top_k]
                    ]
                    search_type = "upload+hybrid"
                cache_hit = False

            else:
                # Search knowledge base via API
                resp = requests.post(
                    f"{API_URL}/search",
                    json={
                        "query":         query,
                        "top_k":         top_k,
                        "use_reranking": use_reranking,
                    },
                    timeout=30
                ).json()
                results     = resp.get("results", [])
                search_type = resp.get("search_type","")
                cache_hit   = resp.get("cache_hit", False)

            elapsed = round((time.time()-t0)*1000)

            # Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Results", len(results))
            c2.metric("Type", search_type)
            c3.metric("Latency", f"{elapsed}ms")
            c4.metric(
                "Cache",
                "HIT" if cache_hit else "MISS"
            )

            st.divider()

            if not results:
                st.warning(
                    "No relevant results found. "
                    "Try a different query."
                )
            else:
                for i, r in enumerate(results):
                    rerank_str = (
                        f" · Rerank: "
                        f"{r.get('rerank_score',0):.4f}"
                        if r.get("rerank_score")
                        else ""
                    )
                    with st.expander(
                        f"**Result {i+1}** · "
                        f"{r.get('topic','')} · "
                        f"Score: {r.get('score',0):.4f}"
                        f"{rerank_str}",
                        expanded=(i == 0)
                    ):
                        st.write(r.get("text",""))
                        ca, cb = st.columns(2)
                        ca.caption(
                            f" {r.get('source','')}"
                        )
                        cb.caption(
                            f" {r.get('word_count',0)}"
                            f" words"
                        )

        except Exception as e:
            st.error(f"Search failed: {e}")
import streamlit as st
import requests
import time
import uuid
from uploader import (
    ingest_uploaded_file,
    search_uploaded_doc,
    delete_session
)

# from fastembed import SparseTextEmbedding
import config
import auth
import google_auth

API_URL = "https://semantic-search-engine-production-fc8f.up.railway.app"

st.set_page_config(
    page_title="SemantiSeek — Semantic Search Engine",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Premium UI Styles ─────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500&display=swap');

    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        font-family: 'Outfit', sans-serif !important;
    }
    code, pre { font-family: 'Fira Code', monospace !important; }

    .premium-banner {
        background: linear-gradient(135deg, rgba(239,68,68,0.08) 0%, rgba(99,102,241,0.08) 100%);
        border: 1px solid rgba(239,68,68,0.15);
        border-radius: 18px;
        padding: 32px 24px;
        margin-bottom: 28px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .premium-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #EF4444 0%, #6366F1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }
    .premium-subtitle {
        font-size: 0.95rem;
        color: var(--text-color);
        opacity: 0.7;
        font-weight: 500;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    .summary-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin: 24px 0 28px 0;
        border-top: 1px solid rgba(128,128,128,0.12);
        padding-top: 20px;
    }
    .summary-box {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128,128,128,0.14);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        transition: all 0.2s ease;
    }
    .summary-box:hover { border-color: rgba(99,102,241,0.25); }
    .summary-lbl {
        font-size: 0.75rem;
        color: var(--text-color);
        opacity: 0.55;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.8px;
        margin-bottom: 4px;
    }
    .summary-val { font-size: 1.6rem; font-weight: 700; color: var(--text-color); }

    .glass-card {
        background: var(--secondary-background-color) !important;
        border: 1px solid rgba(128,128,128,0.16) !important;
        border-radius: 14px !important;
        padding: 20px !important;
        margin-bottom: 18px !important;
        transition: transform 0.2s ease, border-color 0.2s ease !important;
    }
    .glass-card:hover {
        transform: translateY(-1px);
        border-color: rgba(239,68,68,0.35) !important;
        box-shadow: 0 8px 24px rgba(239,68,68,0.05) !important;
    }
    .card-title-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    .card-source { font-weight: 700; font-size: 0.95rem; color: var(--text-color); }
    .badge-pill {
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.78rem;
        font-family: 'Fira Code', monospace;
        border: 1px solid rgba(239,68,68,0.2);
        background-color: rgba(239,68,68,0.08);
        color: #EF4444;
    }
    .rerank-badge {
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.78rem;
        font-family: 'Fira Code', monospace;
        border: 1px solid rgba(99,102,241,0.2);
        background-color: rgba(99,102,241,0.08);
        color: #6366F1;
        margin-left: 6px;
    }
    .card-quote {
        font-size: 0.92rem;
        line-height: 1.6;
        color: var(--text-color);
        opacity: 0.9;
        background-color: rgba(128,128,128,0.04);
        padding: 12px 16px;
        border-radius: 8px;
        border-left: 3px solid #EF4444;
    }
    .card-meta {
        margin-top: 10px;
        font-size: 0.8rem;
        color: var(--text-color);
        opacity: 0.5;
        display: flex;
        gap: 16px;
    }

    .status-badge {
        background-color: rgba(16,185,129,0.08);
        border: 1px solid rgba(16,185,129,0.18);
        color: #10b981;
        padding: 10px 14px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.88rem;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 16px;
    }
    .status-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 8px #10b981;
        animation: pulse 2s infinite;
    }
    .status-offline {
        background-color: rgba(239,68,68,0.08);
        border: 1px solid rgba(239,68,68,0.18);
        color: #EF4444;
        padding: 10px 14px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.88rem;
        margin-bottom: 16px;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); opacity: 0.8; }
        50% { transform: scale(1.1); opacity: 1; }
        100% { transform: scale(0.95); opacity: 0.8; }
    }

    .stButton > button {
        background: linear-gradient(135deg, #EF4444 0%, #B91C1C 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        box-shadow: 0 4px 14px rgba(239,68,68,0.25) !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #DC2626 0%, #991B1B 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 18px rgba(239,68,68,0.35) !important;
    }

    div[data-baseweb="input"] {
        border-radius: 10px !important;
        border: 1px solid rgba(128,128,128,0.2) !important;
        transition: border-color 0.2s ease !important;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: #EF4444 !important;
        box-shadow: 0 0 0 3px rgba(239,68,68,0.12) !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Load models once ──────────────────────────────────────────

# ── Session state ─────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "search_mode" not in st.session_state:
    st.session_state.search_mode = "knowledge_base"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None

# ── Auth Flow ─────────────────────────────────────────────────
if "code" in st.query_params and not st.session_state.logged_in:
    code = st.query_params["code"]
    try:
        access_token = google_auth.get_access_token(code)
        email = google_auth.get_user_email(access_token)
        if email:
            auth.get_or_create_google_user(email)
            st.session_state.logged_in = True
            st.session_state.current_user = email
            st.query_params.clear()
            st.rerun()
    except Exception as e:
        st.error(f"Google Login failed: {e}")

if not st.session_state.logged_in:
    st.markdown("""
    <div class="premium-banner" style="max-width: 500px; margin: 60px auto 30px auto;">
        <div class="premium-title" style="font-size: 2.4rem;">SemantiSeek</div>
        <div class="premium-subtitle">Sign in or create an account</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        with tab1:
            l_user = st.text_input("Username", key="l_user")
            l_pass = st.text_input("Password", type="password", key="l_pass")
            if st.button("Login", key="l_btn"):
                if auth.verify_user(l_user, l_pass):
                    st.session_state.logged_in = True
                    st.session_state.current_user = l_user
                    st.rerun()
                else:
                    st.error("Invalid username or password")
            st.markdown("<hr style='margin: 15px 0;'/>", unsafe_allow_html=True)
            google_url = google_auth.get_login_url()
            st.markdown(f'<a href="{google_url}" target="_self" style="display:block; text-align:center; background:white; color:#333; padding:10px; border-radius:8px; text-decoration:none; font-weight:600; border:1px solid #ccc; box-shadow: 0 2px 4px rgba(0,0,0,0.05); transition:all 0.2s;"><img src="https://www.google.com/favicon.ico" style="width:16px; margin-right:8px; vertical-align:middle; position:relative; top:-1px;">Continue with Google</a>', unsafe_allow_html=True)
        with tab2:
            s_user = st.text_input("Username", key="s_user")
            s_pass = st.text_input("Password", type="password", key="s_pass")
            if st.button("Sign Up", key="s_btn"):
                if s_user and s_pass:
                    res = auth.create_user(s_user, s_pass)
                    if res["status"] == "success":
                        st.success("Account created successfully! You can now log in.")
                    else:
                        st.error(res["message"])
                else:
                    st.error("Please enter a username and password")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-size:1.15rem;font-weight:700;margin-bottom:12px;">⚙️ Settings</div>', unsafe_allow_html=True)
    top_k         = st.slider("Results", 1, 10, 5)
    use_reranking = st.toggle("Cross-Encoder Reranking", value=True)

    st.markdown("<hr style='border-color:rgba(128,128,128,0.12);margin:18px 0;'/>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:1.15rem;font-weight:700;margin-bottom:12px;"> Search Mode</div>', unsafe_allow_html=True)
    mode = st.radio("Search over:", ["Knowledge base", "My uploaded document"], index=0)
    st.session_state.search_mode = "upload" if mode == "My uploaded document" else "knowledge_base"

    st.markdown("<hr style='border-color:rgba(128,128,128,0.12);margin:18px 0;'/>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:1.15rem;font-weight:700;margin-bottom:12px;">System Status</div>', unsafe_allow_html=True)
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.markdown('<div class="status-badge"><span class="status-dot"></span> API connected</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:0.8rem;opacity:0.6;font-weight:600;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px;">Vectors indexed</div>
        <div style="font-size:2.2rem;font-weight:800;color:#EF4444;line-height:1.1;">{health.get("vector_count", 0)}</div>
        """, unsafe_allow_html=True)
    except Exception:
        st.markdown('<div class="status-offline">❌ API offline — run: python main.py api</div>', unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(128,128,128,0.12);margin:18px 0;'/>", unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:1.05rem;font-weight:600;margin-bottom:12px;color:#6366F1;">👤 {st.session_state.current_user}</div>', unsafe_allow_html=True)
    if st.button("Logout", key="logout_btn"):
        st.session_state.logged_in = False
        st.session_state.current_user = None
        st.rerun()

# ── Header Banner ─────────────────────────────────────────────
st.markdown("""
<div class="premium-banner">
    <div class="premium-title">🔍 Semantic Search Engine</div>
    <div class="premium-subtitle">
        BGE-base embeddings &bull; Qdrant hybrid search &bull; Cross-encoder reranking &bull; Redis cache
    </div>
</div>
""", unsafe_allow_html=True)

# ── Upload Panel ──────────────────────────────────────────────
if st.session_state.search_mode == "upload":
    st.markdown("### Upload Your Document")
    uploaded = st.file_uploader("Drop a file here", type=["pdf", "txt", "docx"], help="Supported: PDF, TXT, DOCX")

    if uploaded is not None:
        if st.session_state.uploaded_file != uploaded.name:
            with st.spinner(f"Processing {uploaded.name}..."):
                if st.session_state.session_id:
                    try:
                        delete_session(st.session_state.session_id)
                    except Exception:
                        pass
                session_id = str(uuid.uuid4())[:8]
                response = requests.post(
                    f"{API_URL}/upload?session_id={session_id}",  # ← pass as query param
                    files={
                        "file": (
                            uploaded.name,
                            uploaded.getvalue()
                        )
                    },
                    timeout=300
                )
                st.write(response.json())
                result = response.json()
                if result["status"] == "success":
                    st.session_state.session_id   = session_id
                    st.session_state.uploaded_file = uploaded.name
                    st.markdown(f"""
                    <div style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.15);padding:12px 16px;border-radius:8px;font-size:0.92rem;margin-top:10px;font-weight:500;">
                         <strong>{uploaded.name}</strong> processed — {result['chunks_created']} chunks · Session: {session_id}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error(result.get("message", "Processing failed"))
        else:
            st.markdown(f"""
            <div style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.15);padding:12px 16px;border-radius:8px;font-size:0.92rem;margin-bottom:10px;font-weight:500;">
                 <strong>{uploaded.name}</strong> already loaded. Ready to search.
            </div>
            """, unsafe_allow_html=True)

    if st.session_state.session_id:
        if st.button("Clear document"):
            delete_session(st.session_state.session_id)
            st.session_state.session_id    = None
            st.session_state.uploaded_file = None
            st.rerun()

# ── Search Bar ────────────────────────────────────────────────
st.markdown("<hr style='border-color:rgba(128,128,128,0.12);margin:24px 0;'/>", unsafe_allow_html=True)

if st.session_state.search_mode == "upload":
    placeholder = "Ask anything about your document..."
    disabled    = st.session_state.session_id is None
    if disabled:
        st.warning(" Upload a document first to search it.")
else:
    placeholder = "how does chain-of-thought prompting improve reasoning?"
    disabled    = False

query = st.text_input(
    "Search",
    placeholder=placeholder,
    disabled=disabled,
    label_visibility="collapsed"
)
search_btn = st.button("🔍 Search", type="primary", disabled=disabled or not query)

# ── Search Logic ──────────────────────────────────────────────
if search_btn and query:
    t0 = time.time()
    with st.spinner("Searching..."):
        try:
            if st.session_state.search_mode == "upload":
                resp = requests.post(
                    f"{API_URL}/search-upload",
                    json={
                        "query": query,
                        "session_id": st.session_state.session_id,
                        "top_k": top_k,
                        "use_reranking": use_reranking
                    },
                    timeout=300
                )
                
                data = resp.json()
                if "detail" in data:
                    st.error(f"Search error: {data['detail']}")
                    st.stop()
                                
                results = data["results"]
                search_type = data["search_type"]
                cache_hit = False

            else:
                resp = requests.post(
                    f"{API_URL}/search",
                    json={"query": query, "top_k": top_k, "use_reranking": use_reranking},
                    timeout=30
                ).json()
                results     = resp.get("results", [])
                search_type = resp.get("search_type", "")
                cache_hit   = resp.get("cache_hit", False)

            elapsed = round((time.time() - t0) * 1000)

            # ── Metrics Grid ──────────────────────────────────
            cache_color = "#10b981" if cache_hit else "#F59E0B"
            cache_label = "HIT ⚡" if cache_hit else "MISS"
            st.markdown(f"""
            <div class="summary-grid">
                <div class="summary-box">
                    <div class="summary-lbl">Results</div>
                    <div class="summary-val" style="color:#EF4444;">{len(results)}</div>
                </div>
                <div class="summary-box">
                    <div class="summary-lbl">Type</div>
                    <div class="summary-val" style="color:#6366F1;font-size:1.1rem;margin-top:6px;">{search_type}</div>
                </div>
                <div class="summary-box">
                    <div class="summary-lbl">Latency</div>
                    <div class="summary-val" style="color:#06B6D4;">{elapsed}ms</div>
                </div>
                <div class="summary-box">
                    <div class="summary-lbl">Cache</div>
                    <div class="summary-val" style="color:{cache_color};">{cache_label}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Result Cards ──────────────────────────────────
            if not results:
                st.warning("No relevant results found. Try a different query.")
            else:
                st.markdown("### Search Results")
                for i, r in enumerate(results):
                    rerank_html = (
                        f'<span class="rerank-badge">rerank: {r.get("rerank_score", 0):.4f}</span>'
                        if r.get("rerank_score") else ""
                    )
                    st.markdown(f"""
                    <div class="glass-card">
                        <div class="card-title-row">
                            <div class="card-source">Result {i+1} · {r.get("topic", r.get("source", ""))}</div>
                            <div>
                                <span class="badge-pill">score: {r.get("score", 0):.4f}</span>
                                {rerank_html}
                            </div>
                        </div>
                        <div class="card-quote">{r.get("text", "")}</div>
                        <div class="card-meta">
                            <span> {r.get("source", "")}</span>
                            <span> {r.get("word_count", 0)} words</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Search failed: {e}")

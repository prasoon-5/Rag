import streamlit as st
import streamlit.components.v1 as components
import sqlite3
import os
import re
import hashlib
import secrets
import tempfile
import time
import numpy as np
from contextlib import closing
from dotenv import load_dotenv

load_dotenv()
# ==============================================================================
# BLOCK 1: SYSTEM ENVIRONMENT & CONFIGURATION INITIALIZATION
# ==============================================================================
# Loads system variables from your localized .env configuration file.
# Expects variables: GROQ_API_KEY, LLAMA_PARSE_KEY, TAVILY_API_KEY


# Structural framework imports for document orchestration via LlamaIndex
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ==============================================================================
# BLOCK 2: GLOBAL USER INTERFACE FRAMEWORK & CUSTOM THEME STYLES
# ==============================================================================
st.set_page_config(
    page_title="Multi-PDF QA Intelligence Engine",
    layout="wide"
)

# Custom Slate-Dark Workspace styling sheets
st.markdown("""
<style>
.stApp { background:#0f172a; }
[data-testid="stSidebar"] { background:#111827; }
.chat-title { font-size:28px; font-weight:700; color:#f1f5f9; margin-bottom:0.5rem; }
.empty-state { color:#94a3b8; font-size:16px; padding:2rem 0; text-align:center; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    padding: 8px 16px;
    background-color: #1e293b;
    border-radius: 4px 4px 0px 0px;
    color: #94a3b8;
}
.stTabs [aria-selected="true"] {
    background-color: #334155 !important;
    color: #ffffff !important;
    border-bottom: 2px solid #00ffff !important;
}
div[data-testid="stMetricValue"] {
    font-size: 24px;
    color: #00ffff;
}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# BLOCK 3: LOCAL PERSISTENCE STORAGE MATRIX (SQLite3 System Data Store)
# ==============================================================================
DB_NAME = "qa_workspace_portal.db"

def get_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def hash_password(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000).hex()

def init_database():
    with closing(get_connection()) as conn:
        cursor = conn.cursor()
        
        # User Access Accounts Schema
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_accounts(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            contact_detail TEXT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            address TEXT
        )
        """)

        # Encrypted Authentication Sessions Schema
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS active_sessions(
            token TEXT PRIMARY KEY,
            email TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(email) REFERENCES user_accounts(email) ON DELETE CASCADE
        )
        """)

        # Persistent Thread Headers Schema
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS threads(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT NOT NULL,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Messaging Transactions Index Schema
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS thread_messages(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id INTEGER,
            role TEXT,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(thread_id) REFERENCES threads(id) ON DELETE CASCADE
        )
        """)

        # Generate universal testing administrator if database is clean
        cursor.execute("SELECT COUNT(*) FROM user_accounts")
        if cursor.fetchone()[0] == 0:
            salt = secrets.token_hex(16)
            pw_hash = hash_password("admin123", salt)
            cursor.execute("""
            INSERT INTO user_accounts(name, contact_detail, email, password_hash, salt, address)
            VALUES(?,?,?,?,?,?)
            """, ('Workspace Administrator', '0000000000', 'admin@workspace.local', pw_hash, salt, 'System Host Engine'))
        conn.commit()

init_database()

# ==============================================================================
# BLOCK 4: RUNTIME LIFE-CYCLE STATE SYNCHRONIZATION
# ==============================================================================
if "authorized" not in st.session_state:
    st.session_state.authorized = False
if "profile" not in st.session_state:
    st.session_state.profile = None
if "active_conversation" not in st.session_state:
    st.session_state.active_conversation = None

# Context variables cached for active indexing pipelines
if "parsed_docs" not in st.session_state:
    st.session_state.parsed_docs = {}
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "index" not in st.session_state:
    st.session_state.index = None
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "retrieved_nodes" not in st.session_state:
    st.session_state.retrieved_nodes = []

# Synchronize URL tracking parameter rules on page load/refresh
if "session_token" in st.query_params and not st.session_state.authorized:
    token = st.query_params.get("session_token")
    if isinstance(token, list):
        token = token[0]
    with closing(get_connection()) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT u.name, u.contact_detail, u.address, u.email
        FROM active_sessions s JOIN user_accounts u ON s.email = u.email
        WHERE s.token=?
        """, (token,))
        user = cursor.fetchone()
    if user:
        st.session_state.authorized = True
        st.session_state.profile = {"name": user[0], "phone": user[1], "address": user[2], "email": user[3]}
    else:
        st.query_params.clear()

# Sandbox runtime scripts targeting parent window sessionStorage behaviors
components.html(
    """<script>
    try {
        if (!window.parent.sessionStorage.getItem("auth_active") && window.parent.location.href.includes("session_token")) {
            window.parent.location.search = "";
        }
    } catch(e) { console.error(e); }
    </script>""", height=0
)

# ==============================================================================
# BLOCK 5: SECURITY CRITERIA & USER REGISTRATION LOGIC ROUTINES
# ==============================================================================
EMAIL_REGEX = r"[^@]+@[^@]+\.[^@]+"

def validate_email(email):
    return bool(re.match(EMAIL_REGEX, email))

def login_user(email, password):
    with closing(get_connection()) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name, contact_detail, address, password_hash, salt FROM user_accounts WHERE email=?", (email,))
        row = cursor.fetchone()
    if not row:
        return None
    name, contact, address, password_hash, salt = row
    if hash_password(password, salt) != password_hash:
        return None
    return (name, contact, address)

def register_user(name, contact, email, password, address):
    salt = secrets.token_hex(16)
    pw_hash = hash_password(password, salt)
    try:
        with closing(get_connection()) as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO user_accounts(name, contact_detail, email, password_hash, salt, address)
            VALUES(?,?,?,?,?,?)
            """, (name, contact, email, pw_hash, salt, address))
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def create_session_token(email):
    token = secrets.token_urlsafe(32)
    with closing(get_connection()) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO active_sessions(token, email) VALUES(?,?)", (token, email))
        conn.commit()
    return token

def destroy_session_token(token):
    with closing(get_connection()) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM active_sessions WHERE token=?", (token,))
        conn.commit()

def logout():
    token = st.query_params.get("session_token")
    if token:
        destroy_session_token(token)
    cleanup_empty_conversation(st.session_state.active_conversation)
    st.session_state.authorized = False
    st.session_state.profile = None
    st.session_state.active_conversation = None
    st.query_params.clear()
    components.html("<script>try { window.parent.sessionStorage.clear(); } catch(e) {} </script>", height=0)
    st.rerun()

# ==============================================================================
# BLOCK 6: DIALOGUE PERSISTENCE MANAGEMENT ENGINE
# ==============================================================================
def create_conversation(user_email, title):
    with closing(get_connection()) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO threads(user_email, title) VALUES(?,?)", (user_email, title))
        conn.commit()
        return cursor.lastrowid

def get_conversations(user_email):
    with closing(get_connection()) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, title FROM threads WHERE user_email=? ORDER BY id DESC", (user_email,))
        return cursor.fetchall()

def rename_conversation(conversation_id, title):
    with closing(get_connection()) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE threads SET title=? WHERE id=?", (title, conversation_id))
        conn.commit()

def delete_conversation(conversation_id):
    with closing(get_connection()) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM threads WHERE id=?", (conversation_id,))
        conn.commit()

def cleanup_empty_conversation(conversation_id):
    if not conversation_id:
        return
    with closing(get_connection()) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM thread_messages WHERE thread_id=?", (conversation_id,))
        msg_count = cursor.fetchone()[0]
        cursor.execute("SELECT title FROM threads WHERE id=?", (conversation_id,))
        row = cursor.fetchone()
        if row and msg_count == 0 and row[0] == "New Chat":
            cursor.execute("DELETE FROM threads WHERE id=?", (conversation_id,))
            conn.commit()

def save_message(conversation_id, role, content):
    with closing(get_connection()) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO thread_messages(thread_id, role, content) VALUES(?,?,?)", (conversation_id, role, content))
        conn.commit()

def get_messages(conversation_id):
    with closing(get_connection()) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT role, content FROM thread_messages WHERE thread_id=? ORDER BY id ASC", (conversation_id,))
        return cursor.fetchall()

def spawn_fresh_login_chat():
    email = st.session_state.profile["email"]
    cid = create_conversation(email, "New Chat")
    st.session_state.active_conversation = cid

def generate_chat_title(first_prompt):
    text = first_prompt.strip()
    return text if len(text) <= 30 else text[:30] + "..."

# ==============================================================================
# BLOCK 7: GATEWAY IDENTIFICATION SYSTEM VIEW (UNAUTHENTICATED)
# ==============================================================================
if not st.session_state.authorized:
    st.title("Multi-PDF QA Intelligence Engine")
    login_tab, register_tab = st.tabs(["login", "Sign up"])

    with login_tab:
        with st.form("login_form"):
            email = st.text_input("User Email")
            password = st.text_input("Account Password", type="password")
            login_btn = st.form_submit_button("login")
            if login_btn:
                if not email or not password:
                    st.error("Fields cannot evaluate to empty space configurations.")
                elif not validate_email(email):
                    st.error("Invalid syntax structure parsed on email.")
                else:
                    user = login_user(email, password)
                    if user:
                        st.session_state.authorized = True
                        st.session_state.profile = {"name": user[0], "phone": user[1], "address": user[2], "email": email}
                        st.query_params["session_token"] = create_session_token(email)
                        spawn_fresh_login_chat()
                        components.html("<script>try { window.parent.sessionStorage.setItem('auth_active', 'true'); } catch(e) {}</script>", height=0)
                        st.rerun()
                    else:
                        st.error("Invalid credentials identified.")

    with register_tab:
        with st.form("register_form"):
            name = st.text_input("Full Profile Name")
            contact = st.text_input("Contact Details")
            email = st.text_input("Registration Email Address")
            password = st.text_input("Access Password Space", type="password")
            address = st.text_input("Location Address")
            register_btn = st.form_submit_button("Sign up")
            if register_btn:
                if not name or not email or not password:
                    st.error("Name, email, and password properties are mandatory fields.")
                elif not validate_email(email):
                    st.error("Invalid syntax parsed on target email.")
                elif len(password) < 6:
                    st.error("Password must exceed minimal length restrictions (>= 6 characters).")
                else:
                    if register_user(name, contact, email, password, address):
                        st.success("Identity storage complete. You can now access the login tab.")
                    else:
                        st.error("Identity database registration fault: User identity already exists.")

# ==============================================================================
# BLOCK 8: DESKTOP CONTROL DASHBOARD PANEL (SIDEBAR PORTAL INTERFACE)
# ==============================================================================
else:
    if not st.session_state.active_conversation:
        spawn_fresh_login_chat()

    with st.sidebar:
        st.title("⚡ QA Control Center")
        st.caption(f"Active User: **{st.session_state.profile['name']}**")
        
        if st.button("➕ Create Clean Session", use_container_width=True):
            cleanup_empty_conversation(st.session_state.active_conversation)
            spawn_fresh_login_chat()
            st.rerun()

        st.divider()
        st.subheader("📁 Multi-PDF Ingestion Portal")
        
        # Load environment credentials using variables matched to your specified keys
        groq_key = os.getenv("GROQ_API_KEY")
        parse_key = os.getenv("LLAMA_PARSE_KEY")
        
        if not groq_key:
            groq_key = st.text_input("🔑 Groq API Key", type="password")
        if not parse_key:
            parse_key = st.text_input("🔑 LlamaParse API Key", type="password")

        uploaded_files = st.file_uploader("Upload Target Portfolio PDFs", type=["pdf"], accept_multiple_files=True)
        process_clicked = st.button("Parse Documents", use_container_width=True)

        # ==============================================================================
        # BLOCK 9: DOCUMENT PARSING & VECTOR INDEX GENERATION MATRIX (ETL PIPELINE)
        # ==============================================================================
        if process_clicked:
            if not uploaded_files:
                st.error("❌ Action required: Provide source files to execute ingestion processing.")
            elif not groq_key or not parse_key:
                st.error("❌ Key resolution mismatch: Validate configuration mappings in your local .env file.")
            else:
                with st.status("⚙️ Executing RAG Ingestion Layer...", expanded=True) as status:
                    try:
                        status.write("Configuring Neural Network Inference Models...")
                        Settings.llm = Groq(model="llama-3.1-8b-instant", api_key=groq_key)
                        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
                        
                        parser = LlamaParse(api_key=parse_key, result_type="markdown")
                        splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)
                        
                        temp_parsed_docs = {}
                        temp_all_nodes = []
                        
                        for uploaded_file in uploaded_files:
                            status.write(f"Parsing structural matrix layouts for: `{uploaded_file.name}`...")
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                                tmp_file.write(uploaded_file.read())
                                tmp_path = tmp_file.name
                            
                            llama_docs = parser.load_data(tmp_path)
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)
                            
                            full_markdown = "\n\n".join([doc.text for doc in llama_docs])
                            temp_parsed_docs[uploaded_file.name] = full_markdown
                            
                            for doc in llama_docs:
                                doc.metadata["file_name"] = uploaded_file.name
                                nodes = splitter.get_nodes_from_documents([doc])
                                temp_all_nodes.extend(nodes)
                        
                        status.write("Building Vector Embeddings inside In-Memory Space Store...")
                        vector_index = VectorStoreIndex(nodes=temp_all_nodes)
                        
                        st.session_state.parsed_docs = temp_parsed_docs
                        st.session_state.chunks = temp_all_nodes
                        st.session_state.index = vector_index
                        
                        status.update(label="✅ Vector Index Engine Active!", state="complete", expanded=False)
                        st.success("Target document data maps constructed successfully!")
                        st.rerun()
                    except Exception as e:
                        status.update(label="🚨 Ingestion Routine Faulted", state="error")
                        st.error(f"Pipeline Execution Exception Error: {str(e)}")

        # Thread Session List History Management
        st.divider()
        st.subheader("💬 Historic Dialogue Entries")
        chats = get_conversations(st.session_state.profile["email"])
        for cid, title in chats:
            col1, col2 = st.columns([5, 1])
            with col1:
                lbl = f"💬 {title}" if cid == st.session_state.active_conversation else title
                if st.button(lbl, key=f"chat_{cid}", use_container_width=True):
                    cleanup_empty_conversation(st.session_state.active_conversation)
                    st.session_state.active_conversation = cid
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{cid}"):
                    delete_conversation(cid)
                    if st.session_state.active_conversation == cid:
                        st.session_state.active_conversation = None
                    st.rerun()

        st.divider()
        if st.button("logout", use_container_width=True):
            logout()

    # ==============================================================================
    # BLOCK 10: ANALYTICS FRAMEWORK WORKSPACE STUDIO (MULTI-TAB CONTROL MATRIX)
    # ==============================================================================
    st.markdown("<div class='chat-title'>⚡ Enterprise Multi-PDF Conversational Engine Studio</div>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["💬 AI Chat Assistant", "📄 Document Markdown Inspector", "🧩 Segment Chunk Viewer", "🔍 Retrieval Vector Tracer"])

    # TAB 1: INTERACTIVE USER PROMPT CONVERSATION RUNTIME
    with tab1:
        st.markdown("### 💬 Conversational QA Bot Engine")
        messages = get_messages(st.session_state.active_conversation)
        
        # Render static historical messages instantaneously
        if not messages:
            st.markdown("<div class='empty-state'>👋 System Standby — Feed document streams inside the dashboard panel to initiate prompt sequences.</div>", unsafe_allow_html=True)
        else:
            for role, content in messages:
                with st.chat_message(role):
                    st.markdown(content)

        user_prompt = st.chat_input("Prompt queries against context data stores...")
        if user_prompt:
            conversation_id = st.session_state.active_conversation
            save_message(conversation_id, "user", user_prompt)
            if len(messages) == 0:
                rename_conversation(conversation_id, generate_chat_title(user_prompt))
                
            with st.chat_message("user"):
                st.markdown(user_prompt)
                
            if st.session_state.index is None:
                with st.chat_message("assistant"):
                    err_msg = "⚠️ Engine Offline: No operational index was detected. Load and process documents via the dashboard panel."
                    st.error(err_msg)
                    save_message(conversation_id, "assistant", err_msg)
            else:
                with st.chat_message("assistant"):
                    with st.spinner("Synthesizing Context Vectors..."):
                        try:
                            # Perform Vector Database Lookup
                            retriever = st.session_state.index.as_retriever(similarity_top_k=3)
                            st.session_state.retrieved_nodes = retriever.retrieve(user_prompt)
                            st.session_state.last_query = user_prompt
                            
                            # ==========================================================
                            # PLUGGED IN GUARDRAILS AI CODE HERE
                            # ==========================================================
                            from guardrails import Guard
                            from guardrails.integrations.llama_index import GuardrailsQueryEngine
                            from guardrails.hub import PiiFilter, CompetitorCheck

                            # Initialize specific validators
                            guard = Guard().use_many(
                                PiiFilter(on_fail="fix"),          # Blocks/redacts personal data leakage
                                CompetitorCheck(on_fail="block")   # Restricts model behavior to project scope
                            )

                            # Wrap your existing active LlamaIndex query engine instance
                            guarded_engine = GuardrailsQueryEngine(
                                engine=st.session_state.index.as_query_engine(),
                                guard=guard
                            )

                            # Execute queries safely through the guardrail layer
                            response_obj = guarded_engine.query(user_prompt)
                            raw_response_text = response_obj.response
                            # ==========================================================
                            
                            # Seamless streaming generator closure function
                            def word_by_word_stream_generator(text_payload: str):
                                tokens = re.split(r'(\s+)', text_payload)
                                for token in tokens:
                                    yield token
                                    time.sleep(0.01)

                            # Render live word/letter elements natively using Streamlit engine core
                            streamed_response = st.write_stream(word_by_word_stream_generator(raw_response_text))
                            
                            # Cache completely processed response string context safely inside local SQLite storage
                            save_message(conversation_id, "assistant", streamed_response)
                            
                        except Exception as e:
                            err_msg = f"⚠️ Generation Path Runtime Exception: {str(e)}"
                            st.error(err_msg)
                            save_message(conversation_id, "assistant", err_msg)
            st.rerun()

    # TAB 2: PARSED OBJECT STRUCTURE PREVIEW MARKDOWN ANALYSIS
    with tab2:
        st.markdown("### 📄 Layout Preservation Analysis")
        if not st.session_state.parsed_docs:
            st.info("💡 Storage Clear: Ingest target files via sidebar control layer to review parsed objects.")
        else:
            sel_file = st.selectbox("Select Target File Source Structure:", options=list(st.session_state.parsed_docs.keys()))
            st.markdown("---")
            render_col, raw_col = st.columns(2)
            with render_col:
                st.subheader("🎨 Structural UI Preview")
                st.markdown(st.session_state.parsed_docs[sel_file])
            with raw_col:
                st.subheader("🖥️ Raw Markdown Payload")
                st.code(st.session_state.parsed_docs[sel_file], language="markdown")

    # TAB 3: QUANTIZED TOKEN SEGMETATION EXPLORER
    with tab3:
        st.markdown("### 🧩 Segment Token Fragment Tracking Matrix")
        if not st.session_state.chunks:
            st.info("💡 Storage Clear: Operational data nodes must be ingested to generate text segments.")
        else:
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Total Extracted Chunks", len(st.session_state.chunks))
            m_col2.metric("Average Character Density Size", f"{int(np.mean([len(c.text) for c in st.session_state.chunks]))} chars")
            m_col3.metric("Chunk Constraints Strategy", "1000 / 200 Tokens Window")
            st.markdown("---")
            for idx, chunk in enumerate(st.session_state.chunks):
                with st.expander(f"📦 Node Chunk [{idx + 1}/{len(st.session_state.chunks)}] | Source: {chunk.metadata.get('file_name', 'System Core')}"):
                    st.markdown("**Context Metadata Parameters Structure:**")
                    st.json(chunk.metadata)
                    st.markdown("**Node Structural Character Array Text:**")
                    st.code(chunk.text, language="markdown")

    # TAB 4: VECTOR SCORING LOGIC RETRIEVAL VISUALIZATION TRACER
    with tab4:
        st.markdown("### 🔍 Live Vector Database Retrieval Trace")
        if not st.session_state.last_query:
            st.info("💡 Analyzer Idle: Execute input prompt queries in **AI Chat Assistant** workspace to populate diagnostic trace tables.")
        else:
            st.markdown(f"##### **Target Retrieval Query Trace:** `{st.session_state.last_query}`")
            st.markdown("---")
            for rank, node in enumerate(st.session_state.retrieved_nodes):
                with st.container(border=True):
                    tc, sc = st.columns([4, 1])
                    tc.markdown(f"##### Rank #{rank + 1} | Source Segment: `{node.metadata.get('file_name', 'Unknown Engine')}`")
                    sc.metric(label="Cosine Score Match", value=f"{(node.score if node.score else 0.0):.4f}")
                    st.markdown("**Extracted Segment Context Injected to LLM Prompt:**")
                    st.info(node.text)
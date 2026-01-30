#!/usr/bin/env python3
"""
app.py - Abraham Lincoln AI Assistant (Fine-tuned + RAG)
FIXED:
✅ Loads RAG from rag_pipeline.py
✅ Auto-loads FAISS index files
✅ Parses rag_pipeline.py output correctly (dict -> answer + retrieved_documents)
✅ Displays sources properly
"""

import streamlit as st
import sys
import os
from pathlib import Path
import time
from datetime import datetime
import traceback

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Abraham Lincoln AI Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================
# CUSTOM CSS
# =====================================================

def load_css():
    st.markdown("""
    <style>
    .lincoln-header {
        background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
        padding: 2.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    .lincoln-header h1 {
        color: white;
        font-size: 3rem;
        margin-bottom: 0.5rem;
        font-weight: 800;
    }
    
    .lincoln-header p {
        color: #e8eaf6;
        font-size: 1.3rem;
        margin-bottom: 0.5rem;
    }
    
    .user-message {
        background-color: #e8f5e8;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid #388e3c;
    }
    
    .finetuned-message {
        background-color: #f3e5f5;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid #7b1fa2;
    }
    
    .rag-message {
        background-color: #e3f2fd;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid #1565c0;
    }
    
    .badge-finetuned {
        background-color: #7b1fa2;
        color: white;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 0.8rem;
    }
    
    .badge-rag {
        background-color: #1565c0;
        color: white;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 0.8rem;
    }
    
    .model-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        border: 1px solid #e0e0e0;
    }
    
    .status-loaded {
        color: #4caf50;
        font-weight: bold;
    }
    
    .status-not-loaded {
        color: #f44336;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# =====================================================
# MODEL LOADING
# =====================================================

def load_finetuned_model_simple():
    """Simplified fine-tuned model loading"""
    try:
        from chat_lora import LincolnChatSystem
        chat_system = LincolnChatSystem()

        # Load models if method exists
        if hasattr(chat_system, "load_models"):
            success = chat_system.load_models()
        else:
            success = True

        if success:
            st.sidebar.success("✅ Fine-tuned model loaded!")
            return chat_system

        st.sidebar.error("❌ Failed to load fine-tuned model")
        return None

    except Exception as e:
        st.sidebar.error(f"❌ Fine-tuned model error: {str(e)}")
        st.sidebar.error(traceback.format_exc())
        return None


def load_rag_system_simple():
    """
    Load RAG system from rag_pipeline.py
    and auto-load FAISS index files.
    """
    try:
        from rag_pipeline import LincolnRAGSystem

        rag_system = LincolnRAGSystem()
        st.sidebar.info("📚 Loaded LincolnRAGSystem from rag_pipeline.py")

        # ---- AUTO LOAD INDEX ----
        # Same logic as rag_pipeline.py main section
        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(this_file_dir)

        output_dir = os.path.join(project_root, "llm_integration", "outputs", "rag_results")

        index_file = os.path.join(output_dir, "faiss_index.bin")
        meta_file = os.path.join(output_dir, "documents_metadata.json")

        if os.path.exists(index_file) and os.path.exists(meta_file):
            loaded = rag_system._load_index(output_dir)
            if loaded:
                st.sidebar.success("✅ FAISS index loaded successfully!")
            else:
                st.sidebar.error("❌ Index files found but failed to load index.")
        else:
            st.sidebar.error("❌ FAISS index not found!")
            st.sidebar.write("Expected:")
            st.sidebar.write(f"- {index_file}")
            st.sidebar.write(f"- {meta_file}")
            st.sidebar.warning("⚠️ Run `python rag_pipeline.py` once to generate the index.")

        return rag_system

    except Exception as e:
        st.sidebar.error(f"❌ RAG loading error: {str(e)}")
        st.sidebar.error(traceback.format_exc())
        return None

# =====================================================
# SESSION STATE
# =====================================================

def init_session_state():
    defaults = {
        "finetuned_model": None,
        "rag_system": None,
        "finetuned_loaded": False,
        "rag_loaded": False,
        "chat_history_finetuned": [],
        "chat_history_rag": [],
        "current_tab": "Fine-tuned Lincoln",
        "loading_finetuned": False,
        "loading_rag": False,
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# =====================================================
# LOAD WITH FEEDBACK
# =====================================================

def load_finetuned_model_with_feedback():
    if st.session_state.finetuned_loaded:
        return st.session_state.finetuned_model

    st.session_state.loading_finetuned = True
    placeholder = st.sidebar.empty()
    placeholder.info("🎩 Loading fine-tuned model...")

    model = load_finetuned_model_simple()

    st.session_state.finetuned_model = model
    st.session_state.finetuned_loaded = model is not None
    st.session_state.loading_finetuned = False

    placeholder.empty()
    return model


def load_rag_system_with_feedback():
    if st.session_state.rag_loaded:
        return st.session_state.rag_system

    st.session_state.loading_rag = True
    placeholder = st.sidebar.empty()
    placeholder.info("📚 Loading RAG system...")

    rag = load_rag_system_simple()

    st.session_state.rag_system = rag
    st.session_state.rag_loaded = rag is not None
    st.session_state.loading_rag = False

    placeholder.empty()
    return rag

# =====================================================
# RESPONSE GENERATION
# =====================================================

def generate_finetuned_response(prompt: str):
    if not st.session_state.finetuned_loaded:
        return "❌ Fine-tuned model not loaded. Please load it from sidebar."

    try:
        response, model_used, analysis = st.session_state.finetuned_model.chat(prompt)
        return response
    except Exception as e:
        return f"❌ Fine-tuned response error: {str(e)}"


def generate_rag_response(prompt: str):
    """
    FIXED: rag_pipeline.py returns dict like:
    {
      "query": "...",
      "answer": "...",
      "retrieved_documents": [...]
    }
    """
    if not st.session_state.rag_loaded:
        return "❌ RAG system not loaded. Please load it from sidebar.", []

    try:
        rag_system = st.session_state.rag_system

        result = rag_system.query(prompt, k=5, save_to_storage=False)

        if isinstance(result, dict):
            if "error" in result:
                return f"❌ RAG Error: {result['error']}", []

            answer = result.get("answer", "No answer generated.")
            sources = result.get("retrieved_documents", [])
            return answer, sources

        return str(result), []

    except Exception as e:
        return f"❌ RAG response error: {str(e)}", []

# =====================================================
# UI COMPONENTS
# =====================================================

def render_finetuned_chat():
    st.markdown("### 🎩 Conversing with President Lincoln")

    if not st.session_state.finetuned_loaded:
        st.warning("⚠️ Fine-tuned model not loaded. Click **Load Fine-tuned** in sidebar.")
        return

    # Chat history
    if not st.session_state.chat_history_finetuned:
        st.info("💡 Start a conversation by asking a question below.")
    else:
        for chat in st.session_state.chat_history_finetuned:
            if chat["role"] == "user":
                st.markdown(
                    f'<div class="user-message"><strong>👤 You:</strong><br>{chat["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'''
                    <div class="finetuned-message">
                        <span class="badge-finetuned">🎩 FINE-TUNED LINCOLN</span><br>
                        <strong>🏛️ President Lincoln:</strong><br>
                        {chat["response"]}
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

    # Input
    st.markdown("---")
    col1, col2 = st.columns([4, 1])

    with col1:
        user_input = st.text_input(
            "Type your question:",
            placeholder="Ask President Lincoln about his life, views, or historical context...",
            key=f"finetuned_input_{len(st.session_state.chat_history_finetuned)}",
            label_visibility="collapsed"
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("**Send**", use_container_width=True, key="send_finetuned"):
            if user_input and user_input.strip():
                st.session_state.chat_history_finetuned.append({
                    "role": "user",
                    "content": user_input.strip(),
                    "timestamp": datetime.now().isoformat()
                })

                with st.spinner("🎩 President Lincoln is thinking..."):
                    response = generate_finetuned_response(user_input.strip())

                st.session_state.chat_history_finetuned.append({
                    "role": "assistant",
                    "response": response,
                    "timestamp": datetime.now().isoformat()
                })
                st.rerun()


def render_rag_chat():
    st.markdown("### 📚 Document-Based Research Assistant")

    if not st.session_state.rag_loaded:
        st.warning("⚠️ RAG system not loaded. Click **Load RAG** in sidebar.")
        return

    # Chat history
    if not st.session_state.chat_history_rag:
        st.info("📖 Ask a question to search through Lincoln's historical documents.")
    else:
        for chat in st.session_state.chat_history_rag:
            if chat["role"] == "user":
                st.markdown(
                    f'<div class="user-message"><strong>👤 You:</strong><br>{chat["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'''
                    <div class="rag-message">
                        <span class="badge-rag">📚 DOCUMENT-BASED RESPONSE</span><br>
                        <strong>📖 Research findings:</strong><br>
                        {chat["response"]}
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

                # Sources
                if "sources" in chat and chat["sources"]:
                    with st.expander(f"📚 View {len(chat['sources'])} source documents"):
                        for i, source in enumerate(chat["sources"], 1):
                            if isinstance(source, dict):
                                source_text = source.get("document", "")
                                source_name = source.get("metadata", {}).get("type", f"Document {i}")
                                sim = source.get("similarity_score", None)
                            else:
                                source_text = str(source)
                                source_name = f"Document {i}"
                                sim = None

                            st.markdown(f"### 📄 {source_name}")
                            if sim is not None:
                                st.caption(f"Similarity: {sim:.4f}")
                            st.text(source_text[:600] + ("..." if len(source_text) > 600 else ""))
                            st.markdown("---")

    # Input
    st.markdown("---")
    col1, col2 = st.columns([4, 1])

    with col1:
        user_input = st.text_input(
            "Type your question:",
            placeholder="Search through Lincoln's historical documents...",
            key=f"rag_input_{len(st.session_state.chat_history_rag)}",
            label_visibility="collapsed"
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("**Send**", use_container_width=True, key="send_rag"):
            if user_input and user_input.strip():
                st.session_state.chat_history_rag.append({
                    "role": "user",
                    "content": user_input.strip(),
                    "timestamp": datetime.now().isoformat()
                })

                with st.spinner("📚 Searching documents..."):
                    response, sources = generate_rag_response(user_input.strip())

                st.session_state.chat_history_rag.append({
                    "role": "assistant",
                    "response": response,
                    "sources": sources,
                    "timestamp": datetime.now().isoformat()
                })
                st.rerun()

# =====================================================
# SIDEBAR
# =====================================================

def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Model Management")

        st.markdown("<div class='model-card'>", unsafe_allow_html=True)
        st.markdown("### 🚀 Load Models")

        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "🎩 Load Fine-tuned",
                use_container_width=True,
                disabled=st.session_state.finetuned_loaded,
                key="load_finetuned_main",
            ):
                with st.spinner("Loading fine-tuned model..."):
                    load_finetuned_model_with_feedback()
                st.rerun()

        with col2:
            if st.button(
                "📚 Load RAG",
                use_container_width=True,
                disabled=st.session_state.rag_loaded,
                key="load_rag_main",
            ):
                with st.spinner("Loading RAG system..."):
                    load_rag_system_with_feedback()
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        # Status
        st.markdown("<div class='model-card'>", unsafe_allow_html=True)
        st.markdown("### 🔍 Model Status")

        if st.session_state.finetuned_loaded:
            st.markdown('<p class="status-loaded">✅ Fine-tuned Model: LOADED</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-not-loaded">❌ Fine-tuned Model: NOT LOADED</p>', unsafe_allow_html=True)

        if st.session_state.rag_loaded:
            st.markdown('<p class="status-loaded">✅ RAG System: LOADED</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-not-loaded">❌ RAG System: NOT LOADED</p>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Actions
        st.markdown("<div class='model-card'>", unsafe_allow_html=True)
        st.markdown("### ⚡ Actions")

        if st.button("🗑️ Clear Current Chat", use_container_width=True):
            if st.session_state.current_tab == "Fine-tuned Lincoln":
                st.session_state.chat_history_finetuned = []
            else:
                st.session_state.chat_history_rag = []
            st.rerun()

        if st.button("🔄 Reload All Models", use_container_width=True):
            st.session_state.finetuned_model = None
            st.session_state.rag_system = None
            st.session_state.finetuned_loaded = False
            st.session_state.rag_loaded = False
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# MAIN APP
# =====================================================

def main():
    load_css()
    init_session_state()

    st.markdown("""
    <div class="lincoln-header">
        <h1>🏛️ Abraham Lincoln AI Assistant</h1>
        <p>Two Approaches: Fine-tuned Conversation & Document-Based Research</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🎩 **Fine-tuned Lincoln**", "📚 **RAG Document Assistant**"])

    render_sidebar()

    with tab1:
        st.session_state.current_tab = "Fine-tuned Lincoln"
        render_finetuned_chat()

    with tab2:
        st.session_state.current_tab = "RAG Document Assistant"
        render_rag_chat()

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem; padding: 2rem 0;">
        <p><strong>Abraham Lincoln AI Assistant</strong> • Historical AI Research Project</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

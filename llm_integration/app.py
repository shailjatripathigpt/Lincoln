#!/usr/bin/env python3
"""
app.py - Abraham Lincoln AI Assistant (Fine-tuned + RAG)
Streamlit Cloud Safe Version
"""

import streamlit as st
import sys
import os
from pathlib import Path
import time
from datetime import datetime
import traceback

# =====================================================
# STREAMLIT CLOUD COMPATIBILITY SETTINGS
# =====================================================

# Set environment variables to optimize for Streamlit Cloud
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Reduce TensorFlow logging

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

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
    
    /* Streamlit Cloud specific optimizations */
    .stButton > button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# =====================================================
# MODEL LOADING (STREAMLIT CLOUD SAFE)
# =====================================================

def load_finetuned_model_simple():
    """Simplified fine-tuned model loading with Streamlit Cloud compatibility"""
    try:
        # Try to import with error handling
        try:
            from chat_lora import LincolnChatSystem
        except ImportError as e:
            st.sidebar.error(f"❌ Could not import chat_lora: {str(e)}")
            return None
        
        # Initialize with minimal memory footprint
        try:
            chat_system = LincolnChatSystem()
            
            # Try to load models if method exists
            if hasattr(chat_system, "load_models"):
                with st.spinner("Loading model components..."):
                    success = chat_system.load_models()
            else:
                success = True

            if success:
                return chat_system
            else:
                st.sidebar.error("❌ Model loading failed")
                return None
                
        except Exception as e:
            st.sidebar.error(f"❌ Model initialization error: {str(e)}")
            # Provide fallback or simplified version
            return None

    except Exception as e:
        st.sidebar.error(f"❌ Fine-tuned model error: {str(e)}")
        return None


def load_rag_system_simple():
    """
    Load RAG system from rag_pipeline.py with Streamlit Cloud compatibility
    """
    try:
        # Try to import
        try:
            from rag_pipeline import LincolnRAGSystem
        except ImportError as e:
            st.sidebar.error(f"❌ Could not import rag_pipeline: {str(e)}")
            return None
        
        # Initialize
        try:
            rag_system = LincolnRAGSystem()
            st.sidebar.info("📚 Loaded LincolnRAGSystem from rag_pipeline.py")

            # For Streamlit Cloud, look for index in expected locations
            index_found = False
            
            # Check multiple possible locations
            possible_paths = [
                os.path.join(current_dir, "faiss_index.bin"),
                os.path.join(current_dir, "data", "faiss_index.bin"),
                os.path.join(current_dir, "rag_index", "faiss_index.bin"),
                os.path.join(current_dir.parent, "outputs", "rag_results", "faiss_index.bin"),
                "/tmp/faiss_index.bin"  # Fallback location
            ]
            
            for index_path in possible_paths:
                meta_path = index_path.replace("faiss_index.bin", "documents_metadata.json")
                if os.path.exists(index_path) and os.path.exists(meta_path):
                    # Try to load index
                    try:
                        # Check if _load_index method exists
                        if hasattr(rag_system, '_load_index'):
                            loaded = rag_system._load_index(os.path.dirname(index_path))
                            if loaded:
                                st.sidebar.success(f"✅ FAISS index loaded from: {index_path}")
                                index_found = True
                                break
                    except Exception as e:
                        st.sidebar.warning(f"⚠️ Could not load index from {index_path}: {str(e)}")
            
            if not index_found:
                st.sidebar.warning("""
                ⚠️ FAISS index not found. RAG will run without document retrieval.
                To enable document search, place faiss_index.bin and documents_metadata.json in:
                - /app/ (root directory) OR
                - /app/data/ OR
                - /app/rag_index/
                """)
                
            return rag_system
            
        except Exception as e:
            st.sidebar.error(f"❌ RAG system initialization error: {str(e)}")
            return None

    except Exception as e:
        st.sidebar.error(f"❌ RAG loading error: {str(e)}")
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
        "models_initialized": False,
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# =====================================================
# RESPONSE GENERATION (WITH ERROR HANDLING)
# =====================================================

def generate_finetuned_response(prompt: str):
    if not st.session_state.finetuned_loaded:
        return "❌ Fine-tuned model not loaded. Please load it from sidebar."

    try:
        # Check if chat method exists
        if hasattr(st.session_state.finetuned_model, "chat"):
            response, model_used, analysis = st.session_state.finetuned_model.chat(prompt)
            return response
        else:
            return "❌ Model does not have a chat method."
    except Exception as e:
        return f"❌ Error generating response: {str(e)}"


def generate_rag_response(prompt: str):
    if not st.session_state.rag_loaded:
        return "❌ RAG system not loaded. Please load it from sidebar.", []

    try:
        rag_system = st.session_state.rag_system
        
        # Check if query method exists
        if not hasattr(rag_system, "query"):
            return "❌ RAG system does not have query method.", []
        
        # Call query method with safe parameters
        result = rag_system.query(prompt, k=3, save_to_storage=False)

        if isinstance(result, dict):
            answer = result.get("answer", "No answer generated.")
            sources = result.get("retrieved_documents", [])
            return answer, sources
        elif isinstance(result, str):
            return result, []
        else:
            return str(result), []

    except Exception as e:
        return f"❌ RAG query error: {str(e)}", []

# =====================================================
# UI COMPONENTS (OPTIMIZED FOR STREAMLIT CLOUD)
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
        send_button = st.button("**Send**", use_container_width=True, key="send_finetuned")
        
        if send_button and user_input and user_input.strip():
            # Add user message
            st.session_state.chat_history_finetuned.append({
                "role": "user",
                "content": user_input.strip(),
                "timestamp": datetime.now().isoformat()
            })

            # Generate response
            with st.spinner("🎩 President Lincoln is thinking..."):
                response = generate_finetuned_response(user_input.strip())

            # Add assistant response
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
                            st.text(source_text[:500] + ("..." if len(source_text) > 500 else ""))
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
        send_button = st.button("**Send**", use_container_width=True, key="send_rag")
        
        if send_button and user_input and user_input.strip():
            # Add user message
            st.session_state.chat_history_rag.append({
                "role": "user",
                "content": user_input.strip(),
                "timestamp": datetime.now().isoformat()
            })

            # Generate response
            with st.spinner("📚 Searching documents..."):
                response, sources = generate_rag_response(user_input.strip())

            # Add assistant response
            st.session_state.chat_history_rag.append({
                "role": "assistant",
                "response": response,
                "sources": sources,
                "timestamp": datetime.now().isoformat()
            })
            st.rerun()

# =====================================================
# SIDEBAR (STREAMLIT CLOUD OPTIMIZED)
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
                    model = load_finetuned_model_simple()
                    if model:
                        st.session_state.finetuned_model = model
                        st.session_state.finetuned_loaded = True
                        st.success("✅ Fine-tuned model loaded!")
                    else:
                        st.error("❌ Failed to load fine-tuned model")
                st.rerun()

        with col2:
            if st.button(
                "📚 Load RAG",
                use_container_width=True,
                disabled=st.session_state.rag_loaded,
                key="load_rag_main",
            ):
                with st.spinner("Loading RAG system..."):
                    rag = load_rag_system_simple()
                    if rag:
                        st.session_state.rag_system = rag
                        st.session_state.rag_loaded = True
                        st.success("✅ RAG system loaded!")
                    else:
                        st.error("❌ Failed to load RAG system")
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

        if st.button("🔄 Reset All Models", use_container_width=True):
            st.session_state.finetuned_model = None
            st.session_state.rag_system = None
            st.session_state.finetuned_loaded = False
            st.session_state.rag_loaded = False
            st.success("✅ Models reset. Reload to use again.")
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
        
        # Streamlit Cloud Info
        st.markdown("<div class='model-card'>", unsafe_allow_html=True)
        st.markdown("### ☁️ Streamlit Cloud Info")
        st.markdown("""
        - **Memory**: Models load on-demand
        - **Storage**: Check for FAISS index files
        - **Performance**: Responses may be slower
        """)
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
        <p><small>Streamlit Cloud Compatible Version</small></p>
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
        <p><small>Deployed on Streamlit Cloud • Models load on-demand</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
app.py - Abraham Lincoln AI Assistant (Fine-tuned + Enhanced RAG)
Streamlit Cloud Safe Version with HF Token Support
Now supports Enhanced RAG system with token-based chunking and re-ranking
"""

import streamlit as st
import sys
import os
from pathlib import Path
import time
from datetime import datetime
import traceback
import numpy as np
import uuid  # Added for unique key generation

# =====================================================
# STREAMLIT CLOUD COMPATIBILITY SETTINGS
# =====================================================

# Set environment variables to optimize for Streamlit Cloud
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Reduce TensorFlow logging
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Check for Hugging Face Token in Streamlit Secrets
try:
    if 'HUGGINGFACE_TOKEN' in st.secrets:
        os.environ['HF_TOKEN'] = st.secrets['HUGGINGFACE_TOKEN']
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
except Exception:
    pass

# Check for HF_TOKEN environment variable as fallback
if 'HF_TOKEN' not in os.environ:
    # Try to get from .env file locally
    env_file = Path(__file__).parent / '.env'
    if env_file.exists():
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    if line.startswith('HF_TOKEN='):
                        os.environ['HF_TOKEN'] = line.split('=', 1)[1].strip()
                        break
        except Exception:
            pass

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
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
        font-weight: 800;
    }
    
    .lincoln-header p {
        color: #e8eaf6;
        font-size: 1.2rem;
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
    
    .rag-enhanced-message {
        background-color: #e8f5e9;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid #2e7d32;
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
    
    .badge-rag-enhanced {
        background-color: #2e7d32;
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
    
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #0c5460;
    }
    
    .rag-mode-selector {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px solid #dee2e6;
    }
    
    .metric-badge {
        display: inline-block;
        background: #f0f0f0;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.85rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .similarity-high { color: #2e7d32; font-weight: bold; }
    .similarity-medium { color: #f57c00; font-weight: bold; }
    .similarity-low { color: #d32f2f; font-weight: bold; }
    
    .mode-button {
        width: 100%;
        padding: 0.75rem;
        text-align: center;
        border-radius: 8px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .mode-button.active {
        background-color: #4CAF50;
        color: white;
        border-color: #388E3C;
    }
    
    .mode-button.inactive {
        background-color: #f5f5f5;
        color: #666;
        border-color: #ddd;
    }
    
    .mode-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .disabled-tab {
        background-color: #f0f0f0;
        border: 2px solid #ddd;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    
    .disabled-button {
        opacity: 0.5;
        cursor: not-allowed !important;
    }
    </style>
    """, unsafe_allow_html=True)

# =====================================================
# MODEL LOADING WITH HF TOKEN SUPPORT
# =====================================================

def load_finetuned_model_simple():
    """Simplified fine-tuned model loading with HF token support"""
    try:
        # Import with error handling
        try:
            from chat_lora import LincolnChatSystem
        except ImportError as e:
            st.sidebar.error(f"❌ Could not import chat_lora: {str(e)}")
            return None
        
        # Initialize
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
            return None

    except Exception as e:
        st.sidebar.error(f"❌ Fine-tuned model error: {str(e)}")
        return None

def load_rag_system_simple():
    """
    Load standard RAG system from rag_pipeline.py
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
            
            # Check for index files
            index_found = False
            possible_paths = [
                os.path.join(current_dir, "faiss_index.bin"),
                os.path.join(current_dir, "data", "faiss_index.bin"),
                os.path.join(current_dir, "rag_index", "faiss_index.bin"),
                os.path.join(current_dir, "outputs", "rag_results", "faiss_index.bin"),
            ]
            
            for index_path in possible_paths:
                meta_path = index_path.replace("faiss_index.bin", "documents_metadata.json")
                if os.path.exists(index_path) and os.path.exists(meta_path):
                    try:
                        if hasattr(rag_system, '_load_index'):
                            loaded = rag_system._load_index(os.path.dirname(index_path))
                            if loaded:
                                index_found = True
                                break
                    except Exception:
                        pass
            
            if not index_found:
                st.sidebar.warning("⚠️ No FAISS index found - using text-only RAG")
                
            return rag_system
            
        except Exception as e:
            st.sidebar.error(f"❌ RAG system initialization error: {str(e)}")
            return None

    except Exception as e:
        st.sidebar.error(f"❌ RAG loading error: {str(e)}")
        return None

def load_enhanced_rag_system():
    """
    Load enhanced RAG system from rag_pipeline.py
    """
    try:
        # Try to import
        try:
            from rag_pipeline import EnhancedLincolnRAGSystem
        except ImportError as e:
            # Fall back to standard RAG
            return load_rag_system_simple()
        
        # Initialize enhanced system
        try:
            rag_system = EnhancedLincolnRAGSystem()
            
            # Initialize components
            with st.spinner("Initializing enhanced components..."):
                if not rag_system.initialize_components():
                    st.sidebar.warning("⚠️ Some enhanced components failed to initialize")
            
            # Check for enhanced index files
            index_found = False
            possible_paths = [
                os.path.join(current_dir, "outputs", "rag_results_enhanced", "faiss_index.bin"),
                os.path.join(current_dir, "rag_index_enhanced", "faiss_index.bin"),
                os.path.join(current_dir, "faiss_index.bin"),  # Fallback
            ]
            
            for index_path in possible_paths:
                meta_path = index_path.replace("faiss_index.bin", "chunks_data.json")
                if os.path.exists(index_path):
                    try:
                        if hasattr(rag_system.enhanced_index, 'load'):
                            loaded = rag_system.enhanced_index.load(os.path.dirname(index_path))
                            if loaded:
                                index_found = True
                                break
                    except Exception:
                        pass
            
            if not index_found:
                st.sidebar.warning("⚠️ No enhanced FAISS index found")
                
            return rag_system
            
        except Exception as e:
            # Fall back to standard RAG
            return load_rag_system_simple()

    except Exception as e:
        # Fall back to standard RAG
        return load_rag_system_simple()

# =====================================================
# SESSION STATE
# =====================================================

def init_session_state():
    defaults = {
        "finetuned_model": None,
        "rag_system": None,
        "enhanced_rag_system": None,
        "finetuned_loaded": False,
        "rag_loaded": False,
        "enhanced_rag_loaded": False,
        "chat_history_finetuned": [],
        "chat_history_rag": [],
        "chat_history_enhanced_rag": [],
        "current_tab": "RAG Document Assistant",  # Changed default to RAG tab
        "rag_mode": "standard",  # "standard" or "enhanced"
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# =====================================================
# RESPONSE GENERATION FUNCTIONS
# =====================================================

def generate_finetuned_response(prompt: str):
    if not st.session_state.finetuned_loaded:
        return "❌ Fine-tuned model not loaded. Please load it from sidebar."
    
    try:
        # Check if chat method exists
        if hasattr(st.session_state.finetuned_model, "chat"):
            response = st.session_state.finetuned_model.chat(prompt)
            if isinstance(response, tuple):
                return response[0]  # Return just the answer text
            else:
                return str(response)
        else:
            return "❌ Model does not have a chat method."
    except Exception as e:
        return f"❌ Error generating response: {str(e)[:100]}"

def generate_rag_response(prompt: str, enhanced: bool = False):
    """Generate response from RAG system"""
    if enhanced:
        if not st.session_state.enhanced_rag_loaded:
            return "❌ Enhanced RAG system not loaded. Please load Enhanced RAG from sidebar.", []
        
        rag_system = st.session_state.enhanced_rag_system
        query_method = "query_enhanced"
    else:
        if not st.session_state.rag_loaded:
            return "❌ RAG system not loaded. Please load RAG from sidebar.", []
        
        rag_system = st.session_state.rag_system
        query_method = "query"
    
    try:
        # Check if query method exists
        if hasattr(rag_system, query_method):
            if query_method == "query_enhanced":
                # Enhanced query with re-ranking
                response = rag_system.query_enhanced(prompt, k=10, rerank_top_k=5, save_to_storage=True)
            else:
                # Standard query
                response = rag_system.query(prompt, k=5, save_to_storage=True)
            
            if isinstance(response, dict):
                if "error" in response:
                    return f"❌ RAG Error: {response['error']}", []
                
                answer = response.get("answer", "No answer generated.")
                sources = response.get("retrieved_documents", [])
                
                # Add additional metadata for enhanced mode
                if enhanced and "response_time" in response:
                    answer = f"{answer}\n\n⏱️ Response time: {response['response_time']:.2f}s"
                
                return answer, sources
            
            return str(response), []
        
        return "❌ RAG system does not have query method.", []
        
    except Exception as e:
        return f"❌ RAG query error: {str(e)[:100]}", []

def get_similarity_color(similarity: float):
    """Get CSS class for similarity score"""
    if similarity >= 0.7:
        return "similarity-high"
    elif similarity >= 0.4:
        return "similarity-medium"
    else:
        return "similarity-low"

# =====================================================
# UI COMPONENTS
# =====================================================

def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Model Management")
        
        # Model Loading
        st.markdown("<div class='model-card'>", unsafe_allow_html=True)
        st.markdown("### 🚀 Load Models")

        col1, col2 = st.columns(2)
        
        with col1:
            # Fine-tuned button - DISABLED
            st.button(
                "🎩 Load Lincoln",
                use_container_width=True,
                disabled=True,  # DISABLED
                key="load_finetuned_main",
                help="Fine-tuned model is temporarily disabled"
            )
            st.caption("⏸️ Temporarily disabled")

        with col2:
            rag_button_text = "📚 Load RAG"
            
            if st.button(
                rag_button_text,
                use_container_width=True,
                disabled=st.session_state.rag_loaded,
                key="load_rag_main",
            ):
                with st.spinner(f"Loading RAG system..."):
                    rag = load_rag_system_simple()
                    if rag:
                        st.session_state.rag_system = rag
                        st.session_state.rag_loaded = True
                        st.success("✅ RAG loaded!")
                    else:
                        st.error("❌ Failed to load RAG")
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        # Status
        st.markdown("<div class='model-card'>", unsafe_allow_html=True)
        st.markdown("### 🔍 Model Status")

        if st.session_state.finetuned_loaded:
            st.markdown('<p class="status-loaded">✅ Fine-tuned Model: LOADED</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-not-loaded">⏸️ Fine-tuned Model: DISABLED</p>', unsafe_allow_html=True)

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
            elif st.session_state.current_tab == "RAG Document Assistant":
                if st.session_state.rag_mode == "standard":
                    st.session_state.chat_history_rag = []
                else:
                    st.session_state.chat_history_enhanced_rag = []
            st.rerun()

        if st.button("🔄 Reset All Models", use_container_width=True):
            for key in ["finetuned_model", "rag_system", "enhanced_rag_system", 
                       "finetuned_loaded", "rag_loaded", "enhanced_rag_loaded"]:
                if key in st.session_state:
                    st.session_state[key] = False if "loaded" in key else None
            st.success("✅ Models reset")
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        # Info Section
        st.markdown("<div class='model-card'>", unsafe_allow_html=True)
        st.markdown("### ℹ️ About")
        
        st.markdown("""
        **Features:**
        - ⏸️ Fine-tuned Lincoln (temporarily disabled)
        - 📚 Document-based research
        - 🚀 Enhanced RAG with re-ranking
        - 🏛️ Historical accuracy
        """)
        
        st.markdown("---")
        st.markdown("**Currently Available:**")
        st.markdown("- RAG Document Assistant")
        st.markdown("- Enhanced RAG Mode")
        
        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# CHAT RENDERING FUNCTIONS
# =====================================================

def render_finetuned_chat():
    """Render the fine-tuned chat interface - DISABLED VERSION"""
    st.markdown("### 🎩 Conversing with President Lincoln")
    
    # Disabled message
    st.markdown("""
    <div class="disabled-tab">
        <h3>⏸️ Temporarily Unavailable</h3>
        <p>The fine-tuned Lincoln model is currently disabled for maintenance and optimization.</p>
        <p>Please use the <strong>📚 RAG Document Assistant</strong> tab for document-based research.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show example of what would be available
    with st.expander("ℹ️ About Fine-tuned Lincoln (Coming Soon)"):
        st.markdown("""
        **When available, this feature will include:**
        
        - 🎩 Conversational AI trained on Lincoln's writings
        - 🏛️ Historical personality simulation
        - 💭 Philosophical discussions
        - 🗣️ Speech pattern emulation
        
        **Example questions that will be supported:**
        - What were your views on democracy?
        - How did you approach the issue of slavery?
        - What was your leadership philosophy?
        - Tell me about your debates with Stephen Douglas
        """)
    
    # Chat history (if any from before)
    if st.session_state.chat_history_finetuned:
        st.markdown("---")
        st.markdown("#### Previous Conversations (Read-only)")
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
    
    # Disabled input section
    st.markdown("---")
    st.markdown("### 💬 Chat Input (Disabled)")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.text_input(
            "Type your question:",
            placeholder="Feature temporarily disabled - Use RAG Document Assistant tab",
            key="finetuned_input_disabled",
            label_visibility="collapsed",
            disabled=True
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button(
            "**Send**", 
            use_container_width=True, 
            key="send_finetuned_disabled",
            disabled=True,
            help="Fine-tuned model is temporarily disabled"
        )

def render_rag_chat():
    st.markdown("### 📚 Document-Based Research Assistant")
    
    # RAG Mode Selector - Moved here from sidebar
    st.markdown("---")
    st.markdown("### 🔄 Select RAG Mode")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(
            "📚 **Standard RAG**",
            use_container_width=True,
            type="primary" if st.session_state.rag_mode == "standard" else "secondary",
            help="Document-level retrieval with basic similarity search"
        ):
            st.session_state.rag_mode = "standard"
            st.rerun()
    
    with col2:
        if st.button(
            "🚀 **Enhanced RAG**",
            use_container_width=True,
            type="primary" if st.session_state.rag_mode == "enhanced" else "secondary",
            help="Token-based chunking with cross-encoder re-ranking"
        ):
            st.session_state.rag_mode = "enhanced"
            st.rerun()
    
    # Show current mode info
    if st.session_state.rag_mode == "enhanced":
        st.markdown("""
        <div class="info-box">
        🚀 **Enhanced RAG Mode Active**
        - Token-based chunking for better context
        - Cross-encoder re-ranking for improved relevance
        - Comprehensive retrieval metrics
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
        📚 **Standard RAG Mode Active**
        - Document-level retrieval
        - Basic similarity search
        - Fast response times
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Check if correct system is loaded
    if st.session_state.rag_mode == "enhanced":
        if not st.session_state.enhanced_rag_loaded:
            st.warning("⚠️ Enhanced RAG system not loaded. Click below to load it.")
            
            if st.button("🚀 Load Enhanced RAG System", type="primary"):
                with st.spinner("Loading Enhanced RAG system..."):
                    rag = load_enhanced_rag_system()
                    if rag:
                        st.session_state.enhanced_rag_system = rag
                        st.session_state.enhanced_rag_loaded = True
                        st.success("✅ Enhanced RAG loaded!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to load Enhanced RAG")
            return
        
        rag_loaded = st.session_state.enhanced_rag_loaded
        chat_history = st.session_state.chat_history_enhanced_rag
        badge_class = "badge-rag-enhanced"
        message_class = "rag-enhanced-message"
    else:
        if not st.session_state.rag_loaded:
            st.warning("⚠️ RAG system not loaded. Click **Load RAG** in sidebar.")
            return
        
        rag_loaded = st.session_state.rag_loaded
        chat_history = st.session_state.chat_history_rag
        badge_class = "badge-rag"
        message_class = "rag-message"
    
    # Chat history
    if not chat_history:
        st.info("📖 Ask a question to search through Lincoln's historical documents.")
        st.markdown("**Example questions:**")
        st.markdown("- What did Lincoln say about the Union in 1862?")
        st.markdown("- Find speeches about emancipation")
        st.markdown("- Search for letters about military strategy")
        st.markdown("- What were Lincoln's views on democracy?")
        st.markdown("- Find documents about the Gettysburg Address")
    else:
        for idx, chat in enumerate(chat_history):
            if chat["role"] == "user":
                st.markdown(
                    f'<div class="user-message"><strong>👤 You:</strong><br>{chat["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                badge_text = "🚀 ENHANCED RAG RESPONSE" if st.session_state.rag_mode == "enhanced" else "📚 DOCUMENT-BASED RESPONSE"
                
                st.markdown(
                    f'''
                    <div class="{message_class}">
                        <span class="{badge_class}">{badge_text}</span><br>
                        <strong>📖 Research findings:</strong><br>
                        {chat["response"]}
                    </div>
                    ''',
                    unsafe_allow_html=True
                )
                
                # Sources
                if "sources" in chat and chat["sources"]:
                    with st.expander(f"📄 View {len(chat['sources'])} source documents"):
                        for i, source in enumerate(chat["sources"], 1):
                            if isinstance(source, dict):
                                source_text = source.get("document", "")
                                source_name = source.get("metadata", {}).get("type", f"Document {i}")
                                date = source.get("metadata", {}).get("date", "")
                                similarity = source.get("similarity_score", None)
                                
                                # Display source
                                st.markdown(f"### 📄 {source_name}")
                                
                                # Metadata row
                                meta_cols = st.columns([1, 1])
                                with meta_cols[0]:
                                    if date:
                                        st.caption(f"📅 **Date:** {date}")
                                    if st.session_state.rag_mode == "enhanced" and 'chunk_index' in source:
                                        st.caption(f"🔢 **Chunk:** {source['chunk_index']}")
                                
                                with meta_cols[1]:
                                    if similarity is not None:
                                        color_class = get_similarity_color(similarity)
                                        st.markdown(f'<span class="{color_class}">🔍 Similarity: {similarity:.4f}</span>', unsafe_allow_html=True)
                                    
                                    if st.session_state.rag_mode == "enhanced" and 'cross_encoder_score' in source:
                                        ce_score = source.get('cross_encoder_score')
                                        if ce_score is not None:
                                            ce_color = "similarity-high" if ce_score > 0.5 else "similarity-medium"
                                            st.markdown(f'<span class="{ce_color}">🎯 Cross-encoder: {ce_score:.4f}</span>', unsafe_allow_html=True)
                                
                                # Show preview - FIXED: Using UUID for unique key
                                preview_length = 500 if st.session_state.rag_mode == "enhanced" else 400
                                preview = source_text[:preview_length] + ("..." if len(source_text) > preview_length else "")
                                
                                # Generate a unique key using UUID
                                unique_key = f"preview_{uuid.uuid4()}"
                                st.text_area(f"Content preview:", preview, height=150, key=unique_key)
                            else:
                                st.text(f"Source {i}: {str(source)[:300]}...")
                            st.markdown("---")
    
    # Input
    st.markdown("---")
    col1, col2 = st.columns([4, 1])
    
    with col1:
        input_key = f"{st.session_state.rag_mode}_rag_input_{len(chat_history)}"
        placeholder = "Ask complex questions about Lincoln's documents..." if st.session_state.rag_mode == "enhanced" else "Search through Lincoln's historical documents..."
        
        user_input = st.text_input(
            "Type your question:",
            placeholder=placeholder,
            key=input_key,
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        button_text = "🚀 Analyze" if st.session_state.rag_mode == "enhanced" else "📚 Search"
        button_key = f"send_{st.session_state.rag_mode}_rag"
        
        if st.button(f"**{button_text}**", use_container_width=True, key=button_key):
            if user_input and user_input.strip():
                # Determine which chat history to use
                if st.session_state.rag_mode == "enhanced":
                    target_history = st.session_state.chat_history_enhanced_rag
                    spinner_text = "🚀 Analyzing with enhanced retrieval..."
                else:
                    target_history = st.session_state.chat_history_rag
                    spinner_text = "📚 Searching documents..."
                
                # Add user message
                target_history.append({
                    "role": "user",
                    "content": user_input.strip(),
                    "timestamp": datetime.now().isoformat()
                })
                
                # Generate response
                with st.spinner(spinner_text):
                    response, sources = generate_rag_response(user_input.strip(), enhanced=(st.session_state.rag_mode == "enhanced"))
                
                # Add assistant message
                target_history.append({
                    "role": "assistant",
                    "response": response,
                    "sources": sources,
                    "timestamp": datetime.now().isoformat()
                })
                st.rerun()

# =====================================================
# MAIN APP
# =====================================================

def main():
    load_css()
    init_session_state()
    
    # Header
    st.markdown("""
    <div class="lincoln-header">
        <h1>🏛️ Abraham Lincoln AI Assistant</h1>
        <p>Document Research System with Enhanced RAG</p>
        <p><small>Streamlit Cloud • Memory Optimized</small></p>
    </div>
    """, unsafe_allow_html=True)
    
    # System info - Updated to reflect disabled fine-tuned
    st.markdown("""
    <div class="info-box">
    🔍 **System Features**
    - ⏸️ <strong>Fine-tuned Lincoln</strong>: Temporarily disabled for optimization
    - 📚 <strong>RAG Document Assistant</strong>: Search through historical documents
    - 🚀 <strong>Enhanced Mode</strong>: Token-based chunking & cross-encoder re-ranking
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs - Note: Fine-tuned tab is still available but disabled
    tab1, tab2 = st.tabs([
        "⏸️ **Fine-tuned Lincoln**", 
        "📚 **RAG Document Assistant**"
    ])
    
    with tab1:
        st.session_state.current_tab = "Fine-tuned Lincoln"
        render_finetuned_chat()
    
    with tab2:
        st.session_state.current_tab = "RAG Document Assistant"
        render_rag_chat()
    
    # Sidebar
    render_sidebar()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem; padding: 2rem 0;">
        <p><strong>Abraham Lincoln AI Assistant</strong> • Historical AI Research Project</p>
        <p><small>Enhanced RAG Document System • Built with Streamlit</small></p>
        <p><small>Note: Fine-tuned model temporarily disabled for optimization</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
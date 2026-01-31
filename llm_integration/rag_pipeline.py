#!/usr/bin/env python3
"""
rag_pipeline.py - Enhanced Lincoln RAG System
Features:
✅ FAISS for vector similarity search (using FlatL2 - no training needed)
✅ Token-based chunking (with overlap)
✅ Cross-encoder re-ranking
✅ Comprehensive RAG evaluation metrics
✅ Multiple indexing strategies
✅ Document chunk storage and retrieval
"""

import json
import os
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import time
import re
import hashlib
from dataclasses import dataclass, asdict
from collections import defaultdict
import math
import sys
import csv
from collections import Counter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rag_system.log')
    ]
)
logger = logging.getLogger(__name__)

# =====================================================
# DATA CLASSES
# =====================================================

@dataclass
class DocumentChunk:
    """Represents a chunk of a document"""
    id: str
    text: str
    metadata: Dict
    embedding: Optional[np.ndarray] = None
    token_count: int = 0
    chunk_index: int = 0
    document_id: str = ""
    start_pos: int = 0
    end_pos: int = 0

@dataclass
class RAGEvaluationMetrics:
    """Metrics for evaluating RAG system performance"""
    query_count: int = 0
    average_precision_at_k: List[float] = None
    mean_reciprocal_rank: float = 0.0
    hit_rate: float = 0.0
    average_similarity: float = 0.0
    average_response_time: float = 0.0
    retrieval_effectiveness: float = 0.0
    
    def __post_init__(self):
        if self.average_precision_at_k is None:
            self.average_precision_at_k = [0.0, 0.0, 0.0, 0.0, 0.0]  # P@1, P@3, P@5, P@10, P@20

# =====================================================
# QA STORAGE (Must be defined first)
# =====================================================

class QAStorage:
    """Store and manage Q&A history"""
    
    def __init__(self, storage_dir: str = None):
        if storage_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            storage_dir = os.path.join(os.path.dirname(current_dir), "outputs", "qa_storage")
        
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        # Create storage files
        self.qa_file = os.path.join(storage_dir, "qa_history.json")
        self.stats_file = os.path.join(storage_dir, "qa_statistics.json")
        
        # Load existing history
        self.history = self._load_history()
        
    def _load_history(self):
        """Load existing Q&A history"""
        if os.path.exists(self.qa_file):
            try:
                with open(self.qa_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def save_qa(self, query: str, response: Dict, session_id: str = None):
        """Save a Q&A pair"""
        if session_id is None:
            session_id = f"session_{time.strftime('%Y%m%d_%H%M%S')}"
        
        qa_entry = {
            "session_id": session_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "answer": response.get("answer", ""),
            "query_metadata": {
                "total_retrieved": response.get("total_retrieved", 0),
                "top_similarity": response.get("top_similarity", 0),
                "retrieved_count": len(response.get("retrieved_documents", []))
            },
            "retrieved_documents_summary": [
                {
                    "id": doc["id"],
                    "similarity": doc.get("similarity_score", 0),
                    "source_type": doc.get("metadata", {}).get("type", "unknown"),
                    "date": doc.get("metadata", {}).get("date", "unknown"),
                    "preview": doc.get("document", "")[:200]
                }
                for doc in response.get("retrieved_documents", [])[:3]
            ]
        }
        
        self.history.append(qa_entry)
        self._save_history()
        self._update_statistics(qa_entry)
        
        return qa_entry
    
    def _save_history(self):
        """Save Q&A history to file"""
        with open(self.qa_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
    
    def _update_statistics(self, qa_entry: Dict):
        """Update statistics file"""
        stats = {
            "total_queries": len(self.history),
            "last_query_time": qa_entry["timestamp"],
            "average_similarity": self._calculate_average_similarity(),
            "query_frequency": self._calculate_query_frequency(),
            "top_queries": self._get_top_queries()
        }
        
        with open(self.stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
    
    def _calculate_average_similarity(self):
        """Calculate average similarity score"""
        similarities = [entry["query_metadata"]["top_similarity"] for entry in self.history]
        return sum(similarities) / len(similarities) if similarities else 0
    
    def _calculate_query_frequency(self):
        """Calculate query frequency by hour"""
        frequencies = {}
        for entry in self.history:
            hour = entry["timestamp"][:13]  # Extract YYYY-MM-DD HH
            frequencies[hour] = frequencies.get(hour, 0) + 1
        return frequencies
    
    def _get_top_queries(self):
        """Get most common queries"""
        from collections import Counter
        queries = [entry["query"] for entry in self.history]
        return Counter(queries).most_common(10)
    
    def get_recent_queries(self, limit: int = 10):
        """Get recent queries"""
        return self.history[-limit:] if self.history else []
    
    def export_to_csv(self, output_file: str = None):
        """Export Q&A history to CSV"""
        import csv
        
        if output_file is None:
            output_file = os.path.join(self.storage_dir, "qa_export.csv")
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestamp', 'Query', 'Answer Preview', 'Documents Retrieved',
                'Top Similarity', 'Source Documents'
            ])
            
            for entry in self.history:
                sources = ", ".join([doc["source_type"] for doc in entry["retrieved_documents_summary"]])
                writer.writerow([
                    entry["timestamp"],
                    entry["query"],
                    entry["answer"][:100] + "..." if len(entry["answer"]) > 100 else entry["answer"],
                    entry["query_metadata"]["retrieved_count"],
                    f"{entry['query_metadata']['top_similarity']:.3f}",
                    sources
                ])
        
        return output_file

# =====================================================
# TOKEN-BASED CHUNKING
# =====================================================

class TokenBasedChunker:
    """Token-based text chunking with overlap"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        
        # Token limits for different models (approximate)
        self.token_limits = {
            "gpt-3.5-turbo": 4096,
            "gpt-4": 8192,
            "claude-2": 100000,
            "llama-2": 4096,
            "default": 2048
        }
        
        # Get token limit for model
        self.max_tokens = self.token_limits.get(model_name, self.token_limits["default"])
        
        # Chunk size (80% of max tokens for safety)
        self.chunk_size = int(self.max_tokens * 0.8)
        
        # Overlap between chunks (10% of chunk size)
        self.overlap_tokens = int(self.chunk_size * 0.1)
        
        logger.info(f"TokenBasedChunker initialized: chunk_size={self.chunk_size}, overlap={self.overlap_tokens}")
    
    def count_tokens(self, text: str) -> int:
        """Approximate token count for text"""
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(self.model_name)
            return len(encoding.encode(text))
        except ImportError:
            # Fallback: approximate using words
            return len(text.split()) // 0.75
    
    def chunk_text(self, text: str, metadata: Dict) -> List[DocumentChunk]:
        """Chunk text into token-based chunks with overlap"""
        chunks = []
        
        # Count tokens in full text
        total_tokens = self.count_tokens(text)
        
        if total_tokens <= self.chunk_size:
            # Text fits in one chunk
            chunk_id = f"{metadata.get('document_id', 'doc')}_chunk_0"
            chunk = DocumentChunk(
                id=chunk_id,
                text=text,
                metadata=metadata.copy(),
                token_count=total_tokens,
                chunk_index=0,
                document_id=metadata.get('document_id', ''),
                start_pos=0,
                end_pos=len(text)
            )
            chunks.append(chunk)
            return chunks
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        current_chunk = []
        current_tokens = 0
        chunk_index = 0
        start_pos = 0
        
        for i, paragraph in enumerate(paragraphs):
            para_tokens = self.count_tokens(paragraph)
            
            # If paragraph itself is too large, split by sentences
            if para_tokens > self.chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                for sentence in sentences:
                    sent_tokens = self.count_tokens(sentence)
                    
                    if current_tokens + sent_tokens > self.chunk_size and current_chunk:
                        # Save current chunk
                        chunk_text = '\n\n'.join(current_chunk)
                        chunk_id = f"{metadata.get('document_id', 'doc')}_chunk_{chunk_index}"
                        
                        chunk = DocumentChunk(
                            id=chunk_id,
                            text=chunk_text,
                            metadata=metadata.copy(),
                            token_count=current_tokens,
                            chunk_index=chunk_index,
                            document_id=metadata.get('document_id', ''),
                            start_pos=start_pos,
                            end_pos=start_pos + len(chunk_text)
                        )
                        chunks.append(chunk)
                        
                        # Start new chunk with overlap
                        overlap_text = self._get_overlap(chunk_text)
                        current_chunk = [overlap_text] if overlap_text else []
                        current_tokens = self.count_tokens(overlap_text)
                        start_pos = chunk.end_pos - len(overlap_text) if overlap_text else chunk.end_pos
                        chunk_index += 1
                    
                    current_chunk.append(sentence)
                    current_tokens += sent_tokens
                
                # Add paragraph separator
                current_chunk.append("")
                current_tokens += 1
            
            else:
                # Check if adding this paragraph exceeds chunk size
                if current_tokens + para_tokens > self.chunk_size and current_chunk:
                    # Save current chunk
                    chunk_text = '\n\n'.join(current_chunk)
                    chunk_id = f"{metadata.get('document_id', 'doc')}_chunk_{chunk_index}"
                    
                    chunk = DocumentChunk(
                        id=chunk_id,
                        text=chunk_text,
                        metadata=metadata.copy(),
                        token_count=current_tokens,
                        chunk_index=chunk_index,
                        document_id=metadata.get('document_id', ''),
                        start_pos=start_pos,
                        end_pos=start_pos + len(chunk_text)
                    )
                    chunks.append(chunk)
                    
                    # Start new chunk with overlap
                    overlap_text = self._get_overlap(chunk_text)
                    current_chunk = [overlap_text] if overlap_text else []
                    current_tokens = self.count_tokens(overlap_text)
                    start_pos = chunk.end_pos - len(overlap_text) if overlap_text else chunk.end_pos
                    chunk_index += 1
                
                current_chunk.append(paragraph)
                current_tokens += para_tokens
        
        # Add final chunk if there's remaining text
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunk_id = f"{metadata.get('document_id', 'doc')}_chunk_{chunk_index}"
            
            chunk = DocumentChunk(
                id=chunk_id,
                text=chunk_text,
                metadata=metadata.copy(),
                token_count=current_tokens,
                chunk_index=chunk_index,
                document_id=metadata.get('document_id', ''),
                start_pos=start_pos,
                end_pos=start_pos + len(chunk_text)
            )
            chunks.append(chunk)
        
        logger.info(f"Chunked text into {len(chunks)} chunks (target: {self.chunk_size} tokens)")
        return chunks
    
    def _get_overlap(self, text: str, overlap_percentage: float = 0.1) -> str:
        """Get overlap text from the end of a chunk"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Calculate how many sentences to take for overlap
        overlap_sentences = max(1, int(len(sentences) * overlap_percentage))
        overlap_text = ' '.join(sentences[-overlap_sentences:])
        
        # Ensure overlap doesn't exceed overlap_tokens
        while self.count_tokens(overlap_text) > self.overlap_tokens and overlap_sentences > 1:
            overlap_sentences -= 1
            overlap_text = ' '.join(sentences[-overlap_sentences:])
        
        return overlap_text if self.count_tokens(overlap_text) <= self.overlap_tokens else ""

# =====================================================
# CROSS-ENCODER RE-RANKER
# =====================================================

class CrossEncoderReranker:
    """Cross-encoder based re-ranking for better retrieval"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.initialized = False
        
        logger.info(f"CrossEncoderReranker initialized with model: {model_name}")
    
    def initialize(self):
        """Initialize the cross-encoder model"""
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
            self.initialized = True
            logger.info(f"✅ Cross-encoder model loaded: {self.model_name}")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Failed to load cross-encoder: {e}")
            logger.info("⚠️ Continuing without cross-encoder re-ranking")
            self.initialized = False
            return False
    
    def rerank(self, query: str, documents: List[Dict], top_k: int = 10) -> List[Dict]:
        """Re-rank documents using cross-encoder"""
        if not self.initialized or len(documents) == 0:
            logger.warning("Cross-encoder not initialized, skipping re-ranking")
            return documents[:top_k]
        
        try:
            # Prepare pairs for cross-encoder
            pairs = [(query, doc.get('document', doc.get('text', ''))) for doc in documents]
            
            # Get scores from cross-encoder
            scores = self.model.predict(pairs)
            
            # Combine scores with documents
            scored_docs = []
            for doc, score in zip(documents, scores):
                scored_doc = doc.copy()
                scored_doc['cross_encoder_score'] = float(score)
                scored_docs.append(scored_doc)
            
            # Sort by cross-encoder score (descending)
            scored_docs.sort(key=lambda x: x['cross_encoder_score'], reverse=True)
            
            logger.info(f"Re-ranked {len(documents)} documents, top score: {scored_docs[0]['cross_encoder_score']:.4f}")
            return scored_docs[:top_k]
            
        except Exception as e:
            logger.error(f"Error in re-ranking: {e}")
            return documents[:top_k]
    
    def calculate_confidence(self, query: str, document: str) -> float:
        """Calculate confidence score for a query-document pair"""
        if not self.initialized:
            return 0.5  # Default confidence
        
        try:
            score = self.model.predict([(query, document)])[0]
            return float(score)
        except:
            return 0.5

# =====================================================
# ENHANCED FAISS INDEX WITH MULTIPLE STRATEGIES
# =====================================================

class EnhancedFAISSIndex:
    """Enhanced FAISS index with multiple indexing strategies"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.index_type = None
        self.documents = []
        self.chunks = []
        self.embeddings = None
        self.metadata_index = {}  # For fast metadata lookup
        
        # Index statistics
        self.stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "indexing_time": 0.0,
            "last_updated": None
        }
        
        logger.info(f"EnhancedFAISSIndex initialized with dimension: {dimension}")
    
    def create_index(self, index_type: str = "flat", train_samples: Optional[np.ndarray] = None):
        """Create FAISS index with specified type"""
        try:
            import faiss
            
            if index_type == "flat":
                # Exact search, most accurate but slower for large datasets
                self.index = faiss.IndexFlatL2(self.dimension)
                self.index_type = "FlatL2"
                logger.info(f"Created FAISS index: {self.index_type} (no training needed)")
                
            elif index_type == "ivf":
                # Inverted file index, faster for large datasets
                nlist = 100  # Number of clusters
                
                # Ensure we have enough samples for training
                if train_samples is None or len(train_samples) < nlist * 39:
                    logger.warning(f"Not enough samples for IVF training ({len(train_samples) if train_samples else 0} samples, need at least {nlist * 39})")
                    logger.info("Falling back to FlatL2 index")
                    self.index = faiss.IndexFlatL2(self.dimension)
                    self.index_type = "FlatL2 (fallback)"
                    return True
                
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
                self.index_type = "IVFFlat"
                
                # Train the IVF index
                print(f"Training IVF index with {len(train_samples)} samples...")
                self.index.train(train_samples.astype('float32'))
                logger.info(f"✅ IVF index trained successfully")
                
            elif index_type == "hnsw":
                # Hierarchical Navigable Small World graphs
                M = 16  # Connections per node
                self.index = faiss.IndexHNSWFlat(self.dimension, M)
                self.index_type = "HNSWFlat"
                logger.info(f"Created FAISS index: {self.index_type}")
                
            else:
                logger.warning(f"Unknown index type: {index_type}, using FlatL2")
                self.index = faiss.IndexFlatL2(self.dimension)
                self.index_type = "FlatL2"
            
            return True
            
        except ImportError as e:
            logger.error(f"FAISS not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            return False
    
    def add_documents(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        """Add documents to the index"""
        if self.index is None:
            logger.error("Index not created. Call create_index() first.")
            return False
        
        try:
            start_time = time.time()
            
            # Store chunks and embeddings
            self.chunks.extend(chunks)
            
            # Add to FAISS index
            if self.embeddings is None:
                self.embeddings = embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, embeddings])
            
            self.index.add(embeddings.astype('float32'))
            
            # Build metadata index
            for chunk in chunks:
                self.metadata_index[chunk.id] = {
                    'document_id': chunk.document_id,
                    'chunk_index': chunk.chunk_index,
                    'metadata': chunk.metadata
                }
            
            # Update statistics
            self.stats["total_chunks"] = len(self.chunks)
            self.stats["indexing_time"] += time.time() - start_time
            self.stats["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            logger.info(f"✅ Added {len(chunks)} chunks to index. Total: {self.stats['total_chunks']}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to index: {e}")
            return False
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Search the index"""
        if self.index is None or self.index.ntotal == 0:
            logger.error("Index is empty or not initialized")
            return np.array([]), np.array([])
        
        try:
            # Reshape for single query
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            
            # Search
            distances, indices = self.index.search(query_embedding, k)
            
            # Convert to similarity scores (1 / (1 + distance))
            similarities = 1 / (1 + distances)
            
            return similarities[0], indices[0]
            
        except Exception as e:
            logger.error(f"Error searching index: {e}")
            return np.array([]), np.array([])
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Get chunk by ID"""
        for chunk in self.chunks:
            if chunk.id == chunk_id:
                return chunk
        return None
    
    def get_chunks_by_indices(self, indices: np.ndarray) -> List[DocumentChunk]:
        """Get chunks by indices"""
        chunks = []
        for idx in indices:
            if idx != -1 and idx < len(self.chunks):
                chunks.append(self.chunks[idx])
        return chunks
    
    def save(self, output_dir: str):
        """Save index to disk"""
        try:
            import faiss
            os.makedirs(output_dir, exist_ok=True)
            
            # Save FAISS index
            index_path = os.path.join(output_dir, "faiss_index.bin")
            faiss.write_index(self.index, index_path)
            
            # Save chunks data
            chunks_data = []
            for chunk in self.chunks:
                chunk_dict = asdict(chunk)
                # Convert numpy array to list for JSON serialization
                if chunk_dict['embedding'] is not None:
                    chunk_dict['embedding'] = chunk_dict['embedding'].tolist()
                chunks_data.append(chunk_dict)
            
            chunks_path = os.path.join(output_dir, "chunks_data.json")
            with open(chunks_path, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)
            
            # Save statistics
            stats_path = os.path.join(output_dir, "index_statistics.json")
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, indent=2)
            
            logger.info(f"✅ Index saved to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            return False
    
    def load(self, input_dir: str):
        """Load index from disk"""
        try:
            import faiss
            
            # Load FAISS index
            index_path = os.path.join(input_dir, "faiss_index.bin")
            if not os.path.exists(index_path):
                logger.error(f"Index file not found: {index_path}")
                return False
                
            self.index = faiss.read_index(index_path)
            
            # Load chunks data
            chunks_path = os.path.join(input_dir, "chunks_data.json")
            if not os.path.exists(chunks_path):
                logger.error(f"Chunks data file not found: {chunks_path}")
                return False
                
            with open(chunks_path, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            # Reconstruct DocumentChunk objects
            self.chunks = []
            for chunk_dict in chunks_data:
                # Convert embedding list back to numpy array
                if chunk_dict['embedding'] is not None:
                    chunk_dict['embedding'] = np.array(chunk_dict['embedding'])
                
                chunk = DocumentChunk(**chunk_dict)
                self.chunks.append(chunk)
            
            # Load statistics
            stats_path = os.path.join(input_dir, "index_statistics.json")
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    self.stats = json.load(f)
            
            # Rebuild metadata index
            self.metadata_index = {}
            for chunk in self.chunks:
                self.metadata_index[chunk.id] = {
                    'document_id': chunk.document_id,
                    'chunk_index': chunk.chunk_index,
                    'metadata': chunk.metadata
                }
            
            logger.info(f"✅ Index loaded from {input_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False

# =====================================================
# RAG EVALUATION METRICS
# =====================================================

class RAGEvaluator:
    """Comprehensive RAG evaluation metrics"""
    
    def __init__(self):
        self.metrics_history = []
        self.current_metrics = RAGEvaluationMetrics()
        
        # Ground truth for evaluation (can be loaded from file)
        self.ground_truth = {}
        
        logger.info("RAGEvaluator initialized")
    
    def calculate_precision_at_k(self, retrieved_docs: List[Dict], relevant_docs: List[str], k: int) -> float:
        """Calculate Precision@K"""
        if not retrieved_docs or k <= 0:
            return 0.0
        
        # Get top k retrieved document IDs
        top_k_ids = [doc.get('id', '') for doc in retrieved_docs[:k]]
        
        # Count relevant documents in top k
        relevant_in_top_k = sum(1 for doc_id in top_k_ids if doc_id in relevant_docs)
        
        return relevant_in_top_k / k
    
    def calculate_recall_at_k(self, retrieved_docs: List[Dict], relevant_docs: List[str], k: int) -> float:
        """Calculate Recall@K"""
        if not retrieved_docs or not relevant_docs or k <= 0:
            return 0.0
        
        # Get top k retrieved document IDs
        top_k_ids = [doc.get('id', '') for doc in retrieved_docs[:k]]
        
        # Count relevant documents in top k
        relevant_in_top_k = sum(1 for doc_id in top_k_ids if doc_id in relevant_docs)
        
        return relevant_in_top_k / len(relevant_docs)
    
    def calculate_mean_average_precision(self, queries_results: List[Dict]) -> float:
        """Calculate Mean Average Precision (MAP)"""
        if not queries_results:
            return 0.0
        
        average_precisions = []
        
        for query_result in queries_results:
            retrieved_docs = query_result.get('retrieved_documents', [])
            relevant_docs = query_result.get('relevant_documents', [])
            
            if not relevant_docs:
                continue
            
            precisions = []
            relevant_found = 0
            
            for k in range(1, len(retrieved_docs) + 1):
                precision_at_k = self.calculate_precision_at_k(retrieved_docs, relevant_docs, k)
                if retrieved_docs[k-1].get('id', '') in relevant_docs:
                    relevant_found += 1
                    precisions.append(precision_at_k)
            
            if precisions:
                average_precision = sum(precisions) / len(relevant_docs)
                average_precisions.append(average_precision)
        
        return sum(average_precisions) / len(average_precisions) if average_precisions else 0.0
    
    def calculate_mean_reciprocal_rank(self, queries_results: List[Dict]) -> float:
        """Calculate Mean Reciprocal Rank (MRR)"""
        if not queries_results:
            return 0.0
        
        reciprocal_ranks = []
        
        for query_result in queries_results:
            retrieved_docs = query_result.get('retrieved_documents', [])
            relevant_docs = query_result.get('relevant_documents', [])
            
            if not relevant_docs:
                continue
            
            # Find rank of first relevant document
            first_relevant_rank = None
            for rank, doc in enumerate(retrieved_docs, 1):
                if doc.get('id', '') in relevant_docs:
                    first_relevant_rank = rank
                    break
            
            if first_relevant_rank is not None:
                reciprocal_ranks.append(1.0 / first_relevant_rank)
            else:
                reciprocal_ranks.append(0.0)
        
        return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    def calculate_hit_rate(self, queries_results: List[Dict], k: int = 5) -> float:
        """Calculate Hit Rate@K"""
        if not queries_results:
            return 0.0
        
        hits = 0
        
        for query_result in queries_results:
            retrieved_docs = query_result.get('retrieved_documents', [])
            relevant_docs = query_result.get('relevant_documents', [])
            
            if not relevant_docs:
                continue
            
            # Check if any relevant document is in top k
            top_k_ids = [doc.get('id', '') for doc in retrieved_docs[:k]]
            if any(doc_id in relevant_docs for doc_id in top_k_ids):
                hits += 1
        
        return hits / len(queries_results)
    
    def evaluate_query(self, query: str, retrieved_docs: List[Dict], 
                      response_time: float, relevant_docs: List[str] = None) -> Dict:
        """Evaluate a single query"""
        if relevant_docs is None:
            relevant_docs = self.ground_truth.get(query, [])
        
        # Calculate metrics
        metrics = {
            'query': query,
            'retrieved_count': len(retrieved_docs),
            'relevant_count': len(relevant_docs),
            'response_time': response_time,
            'precision_at_1': self.calculate_precision_at_k(retrieved_docs, relevant_docs, 1),
            'precision_at_3': self.calculate_precision_at_k(retrieved_docs, relevant_docs, 3),
            'precision_at_5': self.calculate_precision_at_k(retrieved_docs, relevant_docs, 5),
            'recall_at_5': self.calculate_recall_at_k(retrieved_docs, relevant_docs, 5),
            'average_similarity': np.mean([doc.get('similarity_score', 0) for doc in retrieved_docs]) if retrieved_docs else 0.0,
            'hit_at_5': 1.0 if self.calculate_precision_at_k(retrieved_docs, relevant_docs, 5) > 0 else 0.0
        }
        
        # Update current metrics
        self.current_metrics.query_count += 1
        self.current_metrics.average_precision_at_k[0] += metrics['precision_at_1']
        self.current_metrics.average_precision_at_k[1] += metrics['precision_at_3']
        self.current_metrics.average_precision_at_k[2] += metrics['precision_at_5']
        self.current_metrics.average_similarity += metrics['average_similarity']
        self.current_metrics.average_response_time += response_time
        
        return metrics
    
    def get_summary_metrics(self) -> Dict:
        """Get summary of all metrics"""
        if self.current_metrics.query_count == 0:
            return {}
        
        # Calculate averages
        query_count = self.current_metrics.query_count
        summary = {
            'query_count': query_count,
            'average_precision_at_1': self.current_metrics.average_precision_at_k[0] / query_count,
            'average_precision_at_3': self.current_metrics.average_precision_at_k[1] / query_count,
            'average_precision_at_5': self.current_metrics.average_precision_at_k[2] / query_count,
            'mean_average_precision': self.current_metrics.mean_reciprocal_rank,  # Would need separate calculation
            'mean_reciprocal_rank': self.current_metrics.mean_reciprocal_rank,
            'hit_rate_at_5': self.current_metrics.hit_rate,
            'average_similarity': self.current_metrics.average_similarity / query_count,
            'average_response_time': self.current_metrics.average_response_time / query_count,
            'retrieval_effectiveness': self.current_metrics.retrieval_effectiveness
        }
        
        return summary
    
    def save_evaluation(self, output_path: str):
        """Save evaluation results to file"""
        try:
            evaluation_data = {
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'summary_metrics': self.get_summary_metrics(),
                'metrics_history': self.metrics_history,
                'current_metrics': asdict(self.current_metrics)
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Evaluation saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving evaluation: {e}")
            return False

# =====================================================
# ENHANCED QA STORAGE
# =====================================================

class EnhancedQAStorage(QAStorage):
    """Enhanced Q&A storage with evaluation tracking"""
    
    def __init__(self, storage_dir: str = None):
        super().__init__(storage_dir)
        self.evaluator = RAGEvaluator()
    
    def save_qa_with_evaluation(self, query: str, response: Dict, 
                               response_time: float, relevant_docs: List[str] = None):
        """Save Q&A with evaluation metrics"""
        # Save to storage
        qa_entry = self.save_qa(query, response)
        
        # Evaluate
        evaluation = self.evaluator.evaluate_query(
            query, 
            response.get('retrieved_documents', []),
            response_time,
            relevant_docs
        )
        
        # Add evaluation to Q&A entry
        qa_entry['evaluation'] = evaluation
        
        # Update statistics
        self._update_statistics(qa_entry)
        
        return qa_entry

# =====================================================
# LINCOLN RAG SYSTEM (Original - needed for inheritance)
# =====================================================

class LincolnRAGSystem:
    """RAG system for Lincoln documents using FAISS for vector search"""
    
    def __init__(self):
        self.embedding_model = None
        self.faiss_index = None
        self.documents = []  # Store original documents
        self.metadata_list = []  # Store metadata
        self.dimension = 384  # all-MiniLM-L6-v2 produces 384-dim embeddings
        self.qa_storage = QAStorage()  # Initialize Q&A storage
        self.last_response = None  # Store last response for document viewing
        
        # Lincoln style templates for different document types
        self.style_templates = {
            "proclamation": {
                "opening": "Now therefore, I, Abraham Lincoln, President of the United States, do hereby declare and make known",
                "middle": "in accordance with the principles upon which our government is founded",
                "closing": "Done at the City of Washington, this day in the year of our Lord"
            },
            "speech": {
                "opening": "Fellow citizens, I stand before you today to address matters of grave importance",
                "middle": "as we contemplate the future of our great nation",
                "closing": "let us move forward with courage and conviction"
            },
            "letter": {
                "opening": "Dear Sir, I write to you regarding matters that weigh heavily upon my mind",
                "middle": "in consideration of our shared responsibilities",
                "closing": "I remain, with great respect, your obedient servant"
            }
        }
        
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("✅ Embedding model loaded: all-MiniLM-L6-v2")
            
            # We'll initialize FAISS index when we have documents
            self.faiss_available = True
            
        except ImportError as e:
            logger.error(f"❌ sentence-transformers not available: {e}")
            logger.info("Install with: pip install sentence-transformers")
            self.faiss_available = False
        
    def _initialize_faiss(self):
        """Initialize FAISS index"""
        if not self.faiss_available:
            return False
        
        try:
            import faiss
            self.faiss = faiss
            
            # Create FAISS index (L2 distance)
            self.faiss_index = faiss.IndexFlatL2(self.dimension)
            logger.info(f"✅ FAISS index initialized (dimension: {self.dimension})")
            return True
            
        except ImportError as e:
            logger.error(f"❌ FAISS not available: {e}")
            logger.info("Install with: pip install faiss-cpu")
            self.faiss_available = False
            return False
    
    def index_documents(self, documents_dir: str, output_dir: str, sample_size: int = None) -> Dict:
        """Index documents using FAISS"""
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.faiss_available:
            logger.error("FAISS not available. Cannot index documents.")
            return {"success": False, "error": "FAISS not available"}
        
        # Initialize FAISS
        if not self._initialize_faiss():
            return {"success": False, "error": "Failed to initialize FAISS"}
        
        # Get list of JSON files
        json_files = [f for f in os.listdir(documents_dir) if f.endswith('.json') 
                     and f not in ["ALL_DOCUMENTS.json", "enhancement_report.json"]]
        
        if not json_files:
            logger.error(f"No JSON files found in {documents_dir}")
            return {"success": False, "error": "No documents found"}
        
        print(f"Found {len(json_files)} documents")
        
        # Use all files if sample_size is None or 0
        if sample_size and sample_size > 0 and len(json_files) > sample_size:
            import random
            json_files = random.sample(json_files, sample_size)
            print(f"Indexing random sample of {sample_size} documents")
        else:
            print(f"Indexing ALL {len(json_files)} documents")
        
        documents_data = []
        failed_files = []
        embeddings_list = []
        
        for filename in tqdm(json_files, desc="Loading and encoding documents"):
            try:
                filepath = os.path.join(documents_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                
                # Extract text
                text = ""
                if "content" in doc and "full_text" in doc["content"]:
                    text = doc["content"]["full_text"]
                elif "full_text" in doc:
                    text = doc["full_text"]
                elif "text" in doc:
                    text = doc["text"]
                
                if text and len(text.strip()) > 100:
                    # Clean and prepare text (store full text)
                    text = text.strip()
                    
                    # Generate embedding (use first 2000 chars for efficiency)
                    embedding = self.embedding_model.encode(text[:2000]).astype('float32')
                    
                    # Store data
                    doc_data = {
                        "id": doc.get("document_id", filename.replace('.json', '')),
                        "text": text,  # Store full text
                        "metadata": doc.get("metadata", {}),
                        "filename": filename,
                        "embedding": embedding
                    }
                    
                    documents_data.append(doc_data)
                    embeddings_list.append(embedding)
                    
                else:
                    failed_files.append(f"{filename}: Text too short ({len(text) if text else 0} chars)")
                    
            except json.JSONDecodeError as e:
                failed_files.append(f"{filename}: Invalid JSON - {e}")
            except Exception as e:
                failed_files.append(f"{filename}: {type(e).__name__} - {e}")
        
        if not documents_data:
            logger.error("No documents could be processed")
            return {"success": False, "error": "No documents processed", "failed_files": failed_files}
        
        print(f"\nSuccessfully processed {len(documents_data)} documents")
        print(f"Failed to process {len(failed_files)} documents")
        
        # Add embeddings to FAISS index
        try:
            embeddings_array = np.vstack(embeddings_list).astype('float32')
            self.faiss_index.add(embeddings_array)
            
            # Store documents and metadata
            for doc in documents_data:
                self.documents.append({
                    "id": doc["id"],
                    "text": doc["text"],  # Full text
                    "metadata": doc["metadata"]
                })
                self.metadata_list.append(doc["metadata"])
            
            logger.info(f"✅ FAISS index built with {self.faiss_index.ntotal} vectors")
            
            # Save index and metadata to disk
            self._save_index(output_dir)
            
            return {
                "success": True,
                "documents_indexed": len(documents_data),
                "failed_documents": len(failed_files),
                "index_size": self.faiss_index.ntotal,
                "embedding_dimension": self.dimension,
                "failed_files_sample": failed_files[:10]  # First 10 failures
            }
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
            return {"success": False, "error": str(e)}
    
    def _save_index(self, output_dir: str):
        """Save FAISS index and metadata to disk"""
        try:
            import faiss
            # Save FAISS index
            index_path = os.path.join(output_dir, "faiss_index.bin")
            faiss.write_index(self.faiss_index, index_path)
            
            # Save documents metadata
            metadata_path = os.path.join(output_dir, "documents_metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "documents": self.documents,
                    "metadata": self.metadata_list,
                    "dimension": self.dimension,
                    "total_documents": len(self.documents)
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Index saved to {index_path}")
            logger.info(f"✅ Metadata saved to {metadata_path}")
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    def _load_index(self, input_dir: str) -> bool:
        """Load FAISS index from disk"""
        try:
            import faiss
            
            index_path = os.path.join(input_dir, "faiss_index.bin")
            metadata_path = os.path.join(input_dir, "documents_metadata.json")
            
            if not os.path.exists(index_path) or not os.path.exists(metadata_path):
                return False
            
            # Load FAISS index
            self.faiss = faiss
            self.faiss_index = faiss.read_index(index_path)
            
            # Load metadata
            with open(metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.documents = data["documents"]
                self.metadata_list = data["metadata"]
                self.dimension = data["dimension"]
            
            logger.info(f"✅ Loaded FAISS index with {self.faiss_index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False
    
    def query(self, query_text: str, k: int = 5, save_to_storage: bool = False) -> Dict:
        """Query the FAISS index with improved document retrieval"""
        if not self.faiss_available or self.faiss_index is None or self.faiss_index.ntotal == 0:
            logger.error("Index not initialized or empty")
            return {"error": "Index not initialized", "retrieved_documents": []}
        
        print(f"\n🔍 Query: {query_text}")
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query_text).astype('float32').reshape(1, -1)
            
            # Search FAISS index
            distances, indices = self.faiss_index.search(query_embedding, k)
            
            # Process results
            retrieved_docs = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and idx < len(self.documents):  # Valid index
                    doc = self.documents[idx]
                    similarity_score = 1 / (1 + distances[0][i])  # Convert distance to similarity (0-1)
                    
                    # Get full text from original document
                    full_text = doc["text"]
                    
                    # Clean up the text - preserve paragraphs but remove excessive whitespace
                    full_text = re.sub(r'\n\s*\n', '\n\n', full_text)  # Preserve paragraph breaks
                    full_text = re.sub(r'[ \t]+', ' ', full_text)  # Remove extra spaces/tabs
                    
                    retrieved_docs.append({
                        "id": doc["id"],
                        "document": full_text,  # Store full text
                        "metadata": doc["metadata"],
                        "distance": float(distances[0][i]),
                        "similarity_score": float(similarity_score),
                        "rank": i + 1
                    })
        
            print(f"Found {len(retrieved_docs)} relevant documents")
            
            # Generate answer with improved prompt engineering
            answer = self._generate_enhanced_answer(query_text, retrieved_docs)
            
            response = {
                "query": query_text,
                "answer": answer,
                "retrieved_documents": retrieved_docs,
                "total_retrieved": len(retrieved_docs),
                "top_similarity": retrieved_docs[0]["similarity_score"] if retrieved_docs else 0
            }
            
            # Store last response for document viewing
            self.last_response = response
            
            # Save to storage if requested
            if save_to_storage:
                self.qa_storage.save_qa(query_text, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error querying index: {e}")
            return {"error": str(e), "retrieved_documents": []}
    
    def _extract_lincoln_phrases(self, text: str) -> List[str]:
        """Extract Lincoln-esque phrases from text"""
        # Clean and split into sentences
        text = re.sub(r'\s+', ' ', text)
        sentences = re.split(r'[.!?]+', text)
        
        lincoln_phrases = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence.split()) > 10:  # Reasonable length
                # Look for Lincoln-style language patterns
                lincoln_keywords = [
                    'union', 'liberty', 'democracy', 'constitution', 'government',
                    'people', 'nation', 'rights', 'justice', 'freedom', 'slavery',
                    'proclamation', 'declare', 'hereby', 'therefore', 'whereas'
                ]
                
                # Check if sentence contains Lincoln-style content
                if any(keyword in sentence.lower() for keyword in lincoln_keywords):
                    lincoln_phrases.append(sentence)
        
        return lincoln_phrases[:5]  # Return top 5 phrases
    
    def _identify_document_style(self, documents: List[Dict]) -> str:
        """Identify the predominant style of retrieved documents"""
        style_counts = {"proclamation": 0, "speech": 0, "letter": 0, "other": 0}
        
        for doc in documents:
            doc_type = doc.get("metadata", {}).get("type", "").lower()
            
            if "proclamation" in doc_type:
                style_counts["proclamation"] += 1
            elif "speech" in doc_type:
                style_counts["speech"] += 1
            elif "letter" in doc_type:
                style_counts["letter"] += 1
            else:
                style_counts["other"] += 1
        
        # Return the most common style
        return max(style_counts, key=style_counts.get)
    
    def _generate_enhanced_answer(self, query: str, documents: List[Dict]) -> str:
        """Generate enhanced answer with improved prompt engineering"""
        if not documents:
            return "I couldn't find any relevant documents in the Lincoln collection to answer your question."
        
        # Extract key content from documents
        extracted_phrases = []
        source_info = []
        
        for i, doc in enumerate(documents[:3]):
            # Extract phrases
            phrases = self._extract_lincoln_phrases(doc.get("document", ""))
            if phrases:
                extracted_phrases.extend(phrases)
            
            # Collect source information
            source_type = doc.get("metadata", {}).get("type", "document")
            date = doc.get("metadata", {}).get("date", "")
            source_info.append(f"{source_type}{f' ({date})' if date else ''}")
        
        # Analyze query type
        query_lower = query.lower()
        
        # Determine if this is a creative writing request
        is_creative = any(keyword in query_lower for keyword in [
            'write', 'tone', 'style', 'proclamation', 'speech', 'letter',
            'compose', 'draft', 'create', 'as if', 'in the voice of'
        ])
        
        # Determine if this is a factual query
        is_factual = any(keyword in query_lower for keyword in [
            'what', 'when', 'where', 'why', 'how', 'explain', 'describe',
            'tell me about', 'information', 'facts', 'history'
        ])
        
        # Build context from extracted phrases
        if extracted_phrases:
            context = " ".join(extracted_phrases[:3])  # Use top 3 phrases
        else:
            # Fallback to document snippets
            context_snippets = [doc.get("document", "")[:200] for doc in documents[:2]]
            context = " ".join(context_snippets)
        
        # Generate answer based on query type
        if is_creative:
            # Determine the requested style
            if "proclamation" in query_lower or "presidential" in query_lower:
                style = "proclamation"
            elif "speech" in query_lower or "address" in query_lower:
                style = "speech"
            elif "letter" in query_lower or "correspondence" in query_lower:
                style = "letter"
            else:
                # Auto-detect style from documents
                style = self._identify_document_style(documents)
            
            # Get style template
            template = self.style_templates.get(style, self.style_templates["speech"])
            
            # Generate creative answer
            if style == "proclamation":
                answer = f"""{template['opening']} that {context[:150]}. {template['middle']}, and in view of the principles articulated in documents such as {', '.join(source_info[:2])}, {context[150:300] if len(context) > 150 else 'we affirm our commitment to constitutional governance'}. {template['closing']}."""
            
            elif style == "speech":
                answer = f"""{template['opening']}. Drawing from historical documents including {', '.join(source_info[:2])}, {context[:200]}. {template['middle']}, {context[200:400] if len(context) > 200 else 'we must uphold the values that have sustained our republic'}. {template['closing']}."""
            
            elif style == "letter":
                answer = f"""{template['opening']}. In reviewing documents from that era, particularly {source_info[0] if source_info else 'historical records'}, {context[:180]}. {template['middle']}, {context[180:350] if len(context) > 180 else 'I believe we must act with both caution and courage'}. {template['closing']}, A. Lincoln."""
            
            else:
                answer = f"Based on Abraham Lincoln's writings, particularly {', '.join(source_info[:2])}, here is a synthesis in historical context: {context[:400]} This reflects the language and concerns of that era."
        
        elif is_factual:
            # Generate factual answer
            answer = f"""Based on documents from Abraham Lincoln's collection ({', '.join(source_info[:3])}), the historical record indicates: {context[:400]}. This information is drawn directly from primary sources dating from {documents[0].get('metadata', {}).get('date', 'the mid-19th century')} to provide accurate historical context."""
        
        else:
            # Generate general answer
            answer = f"""In response to your query about Lincoln, the relevant documents ({', '.join(source_info[:2])}) contain the following information: {context[:350]}. This synthesis incorporates actual phrases and concepts from Lincoln's writings to ensure authenticity while providing meaningful historical insight."""
        
        # Clean up the answer
        answer = ' '.join(answer.split())
        
        # Add attribution
        if len(documents) > 0:
            answer += f" [Sources: {', '.join(source_info[:2])}]"
        
        return answer
    
    def _generate_answer(self, query: str, documents: List[Dict]) -> str:
        """Legacy method - now uses enhanced version"""
        return self._generate_enhanced_answer(query, documents)
    
    def show_full_documents(self):
        """Display full document content from last query"""
        if not self.last_response or not self.last_response.get("retrieved_documents"):
            print("❌ No previous query results to show. Please ask a question first.")
            return
        
        retrieved_docs = self.last_response["retrieved_documents"]
        query_text = self.last_response["query"]
        
        print(f"\n{'='*80}")
        print(f"📄 FULL DOCUMENTS FOR QUERY: '{query_text}'")
        print(f"{'='*80}")
        print(f"Total documents retrieved: {len(retrieved_docs)}")
        print(f"Showing full content for top {min(5, len(retrieved_docs))} documents")
        print(f"{'='*80}\n")
        
        for i, doc in enumerate(retrieved_docs[:3]):  # Show up to 5 docs
            source = doc.get("metadata", {}).get("type", "document")
            date = doc.get("metadata", {}).get("date", "unknown")
            similarity = doc.get("similarity_score", 0)
            doc_id = doc.get("id", "N/A")
            rank = doc.get("rank", i+1)
            
            print(f"\n{'#'*80}")
            print(f"DOCUMENT {rank} of {len(retrieved_docs)}")
            print(f"{'#'*80}")
            print(f"📌 Source: {source}")
            print(f"📅 Date: {date}")
            print(f"🎯 Similarity Score: {similarity:.4f}")
            print(f"📊 Rank: {rank}")
            print(f"🔗 Document ID: {doc_id}")
            print(f"\n{'─'*40} FULL TEXT {'─'*40}")
            
            # Get full text
            full_text = doc.get("document", "")
            
            # Display in readable paragraphs
            paragraphs = full_text.split('\n\n')
            if len(paragraphs) == 1:  # If no paragraph breaks, split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', full_text)
                paragraphs = []
                current_para = []
                for sentence in sentences:
                    current_para.append(sentence)
                    if len(' '.join(current_para)) > 200:
                        paragraphs.append(' '.join(current_para))
                        current_para = []
                if current_para:
                    paragraphs.append(' '.join(current_para))
            
            for j, para in enumerate(paragraphs):
                if para.strip():
                    print(f"\nParagraph {j+1}:")
                    print("-" * 40)
                    # Display paragraph with word wrapping
                    words = para.split()
                    lines = []
                    current_line = []
                    current_length = 0
                    
                    for word in words:
                        if current_length + len(word) + 1 <= 80:  # 80 char line limit
                            current_line.append(word)
                            current_length += len(word) + 1
                        else:
                            lines.append(' '.join(current_line))
                            current_line = [word]
                            current_length = len(word)
                    
                    if current_line:
                        lines.append(' '.join(current_line))
                    
                    for line in lines:
                        print(line)
            
            print(f"\n{'─'*40} METADATA {'─'*40}")
            metadata = doc.get("metadata", {})
            if metadata:
                for key, value in metadata.items():
                    if key not in ['type', 'date']:  # Already shown
                        print(f"  • {key}: {value}")
            else:
                print("  No additional metadata available.")
            
            # Ask if user wants to continue viewing (except for last document)
            if i < min(4, len(retrieved_docs) - 1):
                print(f"\n{'─'*40}")
                continue_viewing = input(f"\n📖 View next document? (y/n): ").strip().lower()
                if continue_viewing != 'y':
                    print("\nReturning to main menu...")
                    break
                print()
            else:
                print(f"\n{'─'*40}")
                print("\n📚 All documents displayed. Returning to main menu...")
    
    def interactive_query_mode(self):
        """Interactive mode for querying the RAG system with full document display"""
        print("\n" + "="*80)
        print("LINCOLN RAG SYSTEM - INTERACTIVE MODE")
        print("="*80)
        print("Type 'quit' to exit, 'help' for commands")
        print("Type 'docs' to see FULL document content for the last query")
        print("\n💡 Try creative queries like:")
        print("   • 'Write a presidential proclamation about unity'")
        print("   • 'Compose a speech in Lincoln's style about democracy'")
        print("   • 'Draft a letter as Lincoln about the challenges of leadership'")
        print("   • 'What was Lincoln's view on slavery?'")
        print("\n💡 After any query, type 'docs' to see complete document content")
        
        session_id = f"session_{time.strftime('%Y%m%d_%H%M%S')}"
        session_qa_count = 0
        
        while True:
            query = input("\n📜 Your question about Lincoln: ").strip()
            
            if query.lower() == 'quit':
                print("\n" + "="*60)
                print("Exiting interactive mode.")
                print(f"\n📊 Session Summary:")
                print(f"  Questions asked: {session_qa_count}")
                print(f"  All Q&A saved to: {self.qa_storage.qa_file}")
                
                # Offer to export to CSV
                export_csv = input("\n📤 Export Q&A history to CSV? (y/n): ").strip().lower()
                if export_csv == 'y':
                    csv_file = self.qa_storage.export_to_csv()
                    print(f"✅ Exported to CSV: {csv_file}")
                
                break
            elif query.lower() == 'help':
                print("\n📋 Available commands:")
                print("  quit      - Exit interactive mode")
                print("  help      - Show this help")
                print("  stats     - Show system statistics")
                print("  history   - Show recent Q&A history")
                print("  docs      - Show FULL documents from last query")
                print("  save      - Force save current session")
                print("  export    - Export history to CSV")
                print("  examples  - Show query examples")
                print("  Any other text - Ask a question about Lincoln")
                continue
            elif query.lower() == 'stats':
                print(f"\n📊 System Statistics:")
                print(f"  FAISS index size: {self.faiss_index.ntotal if self.faiss_index else 0} documents")
                print(f"  Embedding dimension: {self.dimension}")
                print(f"  Documents loaded: {len(self.documents)}")
                print(f"\n📊 Q&A Statistics:")
                recent_qa = self.qa_storage.get_recent_queries(5)
                print(f"  Total questions in history: {len(self.qa_storage.history)}")
                print(f"  Recent questions in this session: {session_qa_count}")
                if recent_qa:
                    print(f"  Last question: {recent_qa[-1]['query'][:50]}...")
                continue
            elif query.lower() == 'history':
                recent_qa = self.qa_storage.get_recent_queries(5)
                print(f"\n📜 Recent Q&A History (last 5):")
                for i, qa in enumerate(recent_qa, 1):
                    print(f"\n  {i}. [{qa['timestamp']}]")
                    print(f"     Q: {qa['query'][:60]}...")
                    print(f"     A: {qa['answer'][:60]}...")
                continue
            elif query.lower() == 'docs':
                self.show_full_documents()
                continue
            elif query.lower() == 'save':
                print(f"✅ Q&A history automatically saved after each query")
                print(f"   Storage file: {self.qa_storage.qa_file}")
                continue
            elif query.lower() == 'export':
                csv_file = self.qa_storage.export_to_csv()
                print(f"✅ Exported to CSV: {csv_file}")
                continue
            elif query.lower() == 'examples':
                print("\n📝 Query Examples:")
                print("  Factual Queries:")
                print("  1. 'What was Lincoln's view on slavery?'")
                print("  2. 'How did Lincoln describe the Union?'")
                print("  3. 'Tell me about the Gettysburg Address'")
                print("\n  Creative Queries:")
                print("  4. 'Write in the tone of an 1860s presidential proclamation'")
                print("  5. 'Compose a speech about preserving the Union'")
                print("  6. 'Draft a letter about democratic principles'")
                print("\n📄 Document Viewing:")
                print("  After any query, type 'docs' to see FULL document content")
                continue
            
            if not query:
                continue
            
            # Process query (automatically saves to storage)
            response = self.query(query, k=5, save_to_storage=True)  # Get 5 documents for better coverage
            session_qa_count += 1
            
            if "error" in response:
                print(f"❌ Error: {response['error']}")
                continue
            
            print(f"\n🤖 Answer:")
            print("-" * 80)
            print(response["answer"])
            
            if response["retrieved_documents"]:
                print(f"\n📄 Retrieved {response['total_retrieved']} documents (top 3 shown):")
                for i, doc in enumerate(response["retrieved_documents"][:3]):
                    source = doc.get("metadata", {}).get("type", "document")
                    date = doc.get("metadata", {}).get("date", "unknown")
                    similarity = doc.get("similarity_score", 0)
                    doc_id = doc.get("id", "N/A")
                    print(f"\n  {i+1}. [{source}, {date}] Similarity: {similarity:.4f}")
                    print(f"     ID: {doc_id}")
                    preview = doc['document'][:250] + "..." if len(doc['document']) > 250 else doc['document']
                    print(f"     Preview: {preview}")
                
                # Prompt user to see full documents
                print(f"\n💡 Type 'docs' to see FULL document content for this query")
    
    def evaluate_rag_system(self, test_queries: List[str] = None) -> Dict:
        """Evaluate the RAG system with test queries"""
        if test_queries is None:
            test_queries = [
                "What was Lincoln's view on slavery?",
                "How did Lincoln describe the Union?",
                "What were Lincoln's thoughts on democracy?",
                "Write in the tone of an 1860s presidential proclamation",
                "Compose a speech about preserving the Union"
            ]
        
        print("\n" + "="*60)
        print("RAG SYSTEM EVALUATION (FAISS-based)")
        print("="*60)
        
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            print("❌ Index not initialized. Please index documents first.")
            return {}
        
        results = []
        for query in test_queries:
            print(f"\nTesting query: {query}")
            response = self.query(query, k=3, save_to_storage=True)
            
            if "error" in response:
                print(f"❌ Error: {response['error']}")
                continue
            
            # Calculate relevance score based on similarity
            if response["retrieved_documents"]:
                avg_similarity = np.mean([doc["similarity_score"] for doc in response["retrieved_documents"]])
            else:
                avg_similarity = 0
            
            results.append({
                "query": query,
                "relevance_score": float(avg_similarity),
                "documents_retrieved": response["total_retrieved"],
                "top_similarity": response.get("top_similarity", 0),
                "answer_preview": response.get("answer", "")[:100] + "..." if len(response.get("answer", "")) > 100 else response.get("answer", "")
            })
            
            print(f"  Documents retrieved: {response['total_retrieved']}")
            print(f"  Average similarity: {avg_similarity:.3f}")
            print(f"  Top similarity: {response.get('top_similarity', 0):.3f}")
            print(f"  Answer preview: {response.get('answer', '')[:80]}...")
        
        if not results:
            return {}
        
        # Calculate overall statistics
        avg_relevance = np.mean([r["relevance_score"] for r in results])
        avg_docs_retrieved = np.mean([r["documents_retrieved"] for r in results])
        
        evaluation = {
            "total_queries": len(results),
            "average_relevance": float(avg_relevance),
            "average_documents_retrieved": float(avg_docs_retrieved),
            "system_type": "FAISS_vector_search",
            "index_size": self.faiss_index.ntotal,
            "query_results": results,
            "recommendations": self._generate_rag_recommendations(results)
        }
        
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Total queries tested: {evaluation['total_queries']}")
        print(f"Average relevance score: {evaluation['average_relevance']:.3f}")
        print(f"Average documents per query: {evaluation['average_documents_retrieved']:.1f}")
        print(f"FAISS index size: {evaluation['index_size']} vectors")
        
        print("\n📋 Key Recommendations:")
        for i, rec in enumerate(evaluation["recommendations"][:3], 1):
            print(f"  {i}. {rec}")
        
        return evaluation
    
    def _generate_rag_recommendations(self, results: List[Dict]) -> List[str]:
        """Generate recommendations for improving RAG system"""
        recommendations = []
        
        # Analyze relevance scores
        low_relevance = [r for r in results if r["relevance_score"] < 0.3]
        if low_relevance:
            recommendations.append(
                f"Consider improving document preprocessing for queries with low relevance "
                f"({len(low_relevance)} queries below 0.3)"
            )
        
        # Check if we have enough documents
        if self.faiss_index.ntotal < 100:
            recommendations.append(
                f"Index more documents (currently {self.faiss_index.ntotal}, "
                f"recommend 500+ for better coverage)"
            )
        
        recommendations.append("Add more style templates for different document types")
        recommendations.append("Implement query expansion for better retrieval")
        recommendations.append("Add document filtering by date or type for specific queries")
        recommendations.append("Consider adding a fine-tuned model for answer refinement")
        
        return recommendations

# =====================================================
# ENHANCED LINCOLN RAG SYSTEM
# =====================================================

class EnhancedLincolnRAGSystem(LincolnRAGSystem):
    """Enhanced RAG system with all new features"""
    
    def __init__(self):
        super().__init__()
        
        # Enhanced components
        self.chunker = TokenBasedChunker()
        self.reranker = CrossEncoderReranker()
        self.enhanced_index = EnhancedFAISSIndex(dimension=384)
        self.evaluator = RAGEvaluator()
        
        # Replace QA storage with enhanced version
        self.qa_storage = EnhancedQAStorage()
        
        logger.info("EnhancedLincolnRAGSystem initialized with all features")
    
    def initialize_components(self):
        """Initialize all components"""
        try:
            # Initialize embedding model
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            
            # Initialize cross-encoder
            self.reranker.initialize()
            
            # Create FAISS index - USING FLAT L2 (no training needed)
            # This is the key fix - using "flat" instead of "ivf"
            self.enhanced_index.create_index(index_type="flat")
            
            logger.info("✅ All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            return False
    
    def index_documents_enhanced(self, documents_dir: str, output_dir: str, 
                                sample_size: int = None) -> Dict:
        """Enhanced document indexing with token-based chunking"""
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.embedding_model:
            logger.error("Embedding model not available.")
            return {"success": False, "error": "Embedding model not available"}
        
        # Get list of JSON files
        json_files = [f for f in os.listdir(documents_dir) if f.endswith('.json') 
                     and f not in ["ALL_DOCUMENTS.json", "enhancement_report.json"]]
        
        if not json_files:
            logger.error(f"No JSON files found in {documents_dir}")
            return {"success": False, "error": "No documents found"}
        
        print(f"Found {len(json_files)} documents")
        
        # Use all files if sample_size is None or 0
        if sample_size and sample_size > 0 and len(json_files) > sample_size:
            import random
            json_files = random.sample(json_files, sample_size)
            print(f"Indexing random sample of {sample_size} documents")
        else:
            print(f"Indexing ALL {len(json_files)} documents")
        
        all_chunks = []
        failed_files = []
        
        for filename in tqdm(json_files, desc="Processing documents"):
            try:
                filepath = os.path.join(documents_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                
                # Extract text
                text = ""
                if "content" in doc and "full_text" in doc["content"]:
                    text = doc["content"]["full_text"]
                elif "full_text" in doc:
                    text = doc["full_text"]
                elif "text" in doc:
                    text = doc["text"]
                
                if text and len(text.strip()) > 100:
                    # Prepare metadata
                    metadata = doc.get("metadata", {})
                    metadata['document_id'] = doc.get("document_id", filename.replace('.json', ''))
                    metadata['filename'] = filename
                    
                    # Chunk text
                    chunks = self.chunker.chunk_text(text.strip(), metadata)
                    
                    # Generate embeddings for chunks
                    for chunk in chunks:
                        # Use first 2000 characters for embedding (for efficiency)
                        embedding_text = chunk.text[:2000]
                        chunk.embedding = self.embedding_model.encode(embedding_text).astype('float32')
                        all_chunks.append(chunk)
                    
                else:
                    failed_files.append(f"{filename}: Text too short")
                    
            except Exception as e:
                failed_files.append(f"{filename}: {type(e).__name__} - {e}")
        
        if not all_chunks:
            logger.error("No chunks could be processed")
            return {"success": False, "error": "No chunks processed", "failed_files": failed_files}
        
        print(f"\n✅ Successfully processed {len(all_chunks)} chunks from documents")
        print(f"⚠️ Failed to process {len(failed_files)} documents")
        
        # Prepare embeddings array
        embeddings = np.vstack([chunk.embedding for chunk in all_chunks])
        
        # Add to enhanced index
        print(f"Adding {len(all_chunks)} chunks to FAISS index...")
        success = self.enhanced_index.add_documents(all_chunks, embeddings)
        
        if not success:
            # Try fallback: recreate index with flat type
            print("⚠️ Index add failed, recreating with FlatL2...")
            self.enhanced_index.create_index(index_type="flat")
            success = self.enhanced_index.add_documents(all_chunks, embeddings)
            
            if not success:
                return {"success": False, "error": "Failed to add documents to index"}
        
        # Save index
        self.enhanced_index.save(output_dir)
        
        # Update statistics
        index_stats = self.enhanced_index.stats.copy()
        index_stats.update({
            "success": True,
            "documents_processed": len(json_files) - len(failed_files),
            "chunks_created": len(all_chunks),
            "average_chunks_per_document": len(all_chunks) / (len(json_files) - len(failed_files)) if (len(json_files) - len(failed_files)) > 0 else 0,
            "failed_documents": len(failed_files),
            "failed_files_sample": failed_files[:5],
            "index_type": self.enhanced_index.index_type
        })
        
        print(f"\n🎉 Indexing completed successfully!")
        print(f"   Index type: {self.enhanced_index.index_type}")
        print(f"   Total chunks: {len(all_chunks)}")
        print(f"   Index size: {self.enhanced_index.index.ntotal if self.enhanced_index.index else 0} vectors")
        
        return index_stats
    
    def query_enhanced(self, query_text: str, k: int = 10, rerank_top_k: int = 5, 
                      save_to_storage: bool = True) -> Dict:
        """Enhanced query with re-ranking and evaluation"""
        start_time = time.time()
        
        if not self.enhanced_index.index or self.enhanced_index.index.ntotal == 0:
            logger.error("Index not initialized or empty")
            return {"error": "Index not initialized", "retrieved_documents": []}
        
        print(f"\n🔍 Enhanced Query: {query_text}")
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query_text).astype('float32')
            
            # Search in FAISS index (retrieve more than needed for re-ranking)
            search_k = min(k * 2, self.enhanced_index.index.ntotal)
            similarities, indices = self.enhanced_index.search(query_embedding, search_k)
            
            # Get chunks
            chunks = self.enhanced_index.get_chunks_by_indices(indices)
            
            # Prepare documents for re-ranking
            documents_for_reranking = []
            for chunk, similarity, idx in zip(chunks, similarities, indices):
                if idx != -1 and chunk is not None:
                    doc = {
                        "id": chunk.id,
                        "document": chunk.text,
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                        "similarity_score": float(similarity),
                        "distance": float(1 / similarity - 1) if similarity > 0 else float('inf'),
                        "chunk_index": chunk.chunk_index,
                        "document_id": chunk.document_id
                    }
                    documents_for_reranking.append(doc)
            
            # Re-rank using cross-encoder if available
            if self.reranker.initialized and len(documents_for_reranking) > rerank_top_k:
                print(f"Re-ranking {len(documents_for_reranking)} documents with cross-encoder...")
                reranked_docs = self.reranker.rerank(query_text, documents_for_reranking, rerank_top_k)
            else:
                reranked_docs = documents_for_reranking[:rerank_top_k]
            
            # Generate enhanced answer
            answer = self._generate_enhanced_answer(query_text, reranked_docs)
            
            response_time = time.time() - start_time
            
            # Prepare response
            response = {
                "query": query_text,
                "answer": answer,
                "retrieved_documents": reranked_docs,
                "total_retrieved": len(reranked_docs),
                "top_similarity": reranked_docs[0]["similarity_score"] if reranked_docs else 0,
                "response_time": response_time,
                "query_method": "enhanced_with_reranking" if self.reranker.initialized else "enhanced_no_reranking"
            }
            
            # Store last response
            self.last_response = response
            
            # Save to storage with evaluation
            if save_to_storage:
                self.qa_storage.save_qa_with_evaluation(query_text, response, response_time)
            
            # Log performance
            print(f"⏱️  Response time: {response_time:.2f}s")
            print(f"📊 Retrieved {len(reranked_docs)} documents")
            if reranked_docs:
                print(f"🎯 Top similarity: {reranked_docs[0]['similarity_score']:.4f}")
                if 'cross_encoder_score' in reranked_docs[0]:
                    print(f"🔍 Cross-encoder score: {reranked_docs[0]['cross_encoder_score']:.4f}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in enhanced query: {e}")
            # Fallback to simple query
            print(f"⚠️ Enhanced query failed, falling back to simple query...")
            return self.query(query_text, k=rerank_top_k, save_to_storage=save_to_storage)
    
    def evaluate_rag_system_comprehensive(self, test_queries: List[Dict] = None) -> Dict:
        """Comprehensive RAG system evaluation"""
        if test_queries is None:
            # Default test queries with ground truth (simplified)
            test_queries = [
                {
                    "query": "What was Lincoln's view on slavery?",
                    "relevant_documents": []  # Would be filled with actual document IDs
                },
                {
                    "query": "How did Lincoln describe the Union?",
                    "relevant_documents": []
                },
                {
                    "query": "What were Lincoln's thoughts on democracy?",
                    "relevant_documents": []
                }
            ]
        
        print("\n" + "="*80)
        print("COMPREHENSIVE RAG SYSTEM EVALUATION")
        print("="*80)
        
        if not self.enhanced_index.index or self.enhanced_index.index.ntotal == 0:
            print("❌ Index not initialized. Please index documents first.")
            return {}
        
        results = []
        
        for test_case in tqdm(test_queries, desc="Evaluating queries"):
            query = test_case["query"]
            relevant_docs = test_case.get("relevant_documents", [])
            
            print(f"\nTesting query: {query}")
            
            # Run enhanced query
            start_time = time.time()
            response = self.query_enhanced(query, k=10, rerank_top_k=5, save_to_storage=False)
            response_time = time.time() - start_time
            
            if "error" in response:
                print(f"❌ Error: {response['error']}")
                continue
            
            # Calculate metrics
            retrieved_docs = response.get("retrieved_documents", [])
            
            metrics = self.evaluator.evaluate_query(
                query, retrieved_docs, response_time, relevant_docs
            )
            
            # Store result
            result = {
                "query": query,
                "response": response,
                "metrics": metrics,
                "relevant_documents": relevant_docs,
                "retrieved_documents_count": len(retrieved_docs)
            }
            results.append(result)
            
            print(f"  📊 Precision@1: {metrics['precision_at_1']:.3f}")
            print(f"  📊 Precision@3: {metrics['precision_at_3']:.3f}")
            print(f"  📊 Precision@5: {metrics['precision_at_5']:.3f}")
            print(f"  ⏱️  Response time: {response_time:.2f}s")
        
        if not results:
            return {}
        
        # Calculate overall metrics
        summary_metrics = self.evaluator.get_summary_metrics()
        
        # Add system information
        evaluation = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": {
                "index_type": self.enhanced_index.index_type,
                "total_chunks": self.enhanced_index.stats["total_chunks"],
                "embedding_dimension": self.enhanced_index.dimension,
                "chunker": str(self.chunker.model_name),
                "reranker": self.reranker.model_name if self.reranker.initialized else "not_initialized"
            },
            "summary_metrics": summary_metrics,
            "detailed_results": results,
            "recommendations": self._generate_comprehensive_recommendations(results, summary_metrics)
        }
        
        # Print summary
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"Total queries evaluated: {summary_metrics.get('query_count', 0)}")
        print(f"Average Precision@1: {summary_metrics.get('average_precision_at_1', 0):.3f}")
        print(f"Average Precision@3: {summary_metrics.get('average_precision_at_3', 0):.3f}")
        print(f"Average Precision@5: {summary_metrics.get('average_precision_at_5', 0):.3f}")
        print(f"Hit Rate@5: {summary_metrics.get('hit_rate_at_5', 0):.3f}")
        print(f"Average response time: {summary_metrics.get('average_response_time', 0):.2f}s")
        print(f"Average similarity: {summary_metrics.get('average_similarity', 0):.3f}")
        
        print("\n📋 Key Recommendations:")
        for i, rec in enumerate(evaluation["recommendations"][:5], 1):
            print(f"  {i}. {rec}")
        
        return evaluation
    
    def _generate_comprehensive_recommendations(self, results: List[Dict], metrics: Dict) -> List[str]:
        """Generate comprehensive recommendations for improving RAG system"""
        recommendations = []
        
        # Analyze precision scores
        avg_p1 = metrics.get('average_precision_at_1', 0)
        avg_p5 = metrics.get('average_precision_at_5', 0)
        
        if avg_p1 < 0.3:
            recommendations.append("Low Precision@1: Consider improving first-stage retrieval or using a better embedding model")
        
        if avg_p5 < 0.5:
            recommendations.append("Moderate Precision@5: Implement query expansion or add more training data")
        
        # Analyze response time
        avg_response_time = metrics.get('average_response_time', 0)
        if avg_response_time > 2.0:
            recommendations.append(f"High response time ({avg_response_time:.1f}s): Consider using a faster index type (HNSW) or reducing re-ranking depth")
        
        # Chunking recommendations
        avg_similarity = metrics.get('average_similarity', 0)
        if avg_similarity < 0.5:
            recommendations.append("Low average similarity: Adjust chunking strategy or improve document preprocessing")
        
        # System-specific recommendations
        recommendations.append("Implement hybrid search (keyword + vector) for better recall")
        recommendations.append("Add diversity scoring to avoid duplicate content in results")
        recommendations.append("Implement query understanding module for better query reformulation")
        recommendations.append("Add temporal filtering for time-sensitive queries")
        recommendations.append("Implement confidence scoring for generated answers")
        recommendations.append("Add user feedback loop to improve retrieval over time")
        recommendations.append("Consider using larger embedding models for complex queries")
        recommendations.append("Implement caching for frequent queries")
        
        return recommendations
    
    def interactive_enhanced_mode(self):
        """Enhanced interactive mode with all features"""
        print("\n" + "="*80)
        print("ENHANCED LINCOLN RAG SYSTEM - INTERACTIVE MODE")
        print("="*80)
        print("Features enabled:")
        print("  ✅ Token-based chunking")
        print("  ✅ Cross-encoder re-ranking")
        print("  ✅ FAISS vector search (FlatL2)")
        print("  ✅ Comprehensive evaluation metrics")
        print("\nType 'quit' to exit, 'help' for commands")
        
        session_id = f"session_{time.strftime('%Y%m%d_%H%M%S')}"
        session_qa_count = 0
        
        while True:
            query = input("\n📜 Your question about Lincoln: ").strip()
            
            if query.lower() == 'quit':
                print(f"\nSession summary: {session_qa_count} questions asked")
                break
            elif query.lower() == 'help':
                print("\n📋 Available commands:")
                print("  quit      - Exit interactive mode")
                print("  help      - Show this help")
                print("  stats     - Show system statistics")
                print("  eval      - Run comprehensive evaluation")
                print("  docs      - Show documents from last query")
                print("  chunks    - Show chunking statistics")
                print("  settings  - Show current settings")
                continue
            elif query.lower() == 'stats':
                self._show_system_statistics()
                continue
            elif query.lower() == 'eval':
                evaluation = self.evaluate_rag_system_comprehensive()
                if evaluation:
                    eval_file = f"rag_evaluation_{time.strftime('%Y%m%d_%H%M%S')}.json"
                    with open(eval_file, 'w', encoding='utf-8') as f:
                        json.dump(evaluation, f, indent=2)
                    print(f"✅ Evaluation saved to {eval_file}")
                continue
            elif query.lower() == 'docs':
                self.show_full_documents()
                continue
            elif query.lower() == 'chunks':
                self._show_chunking_statistics()
                continue
            elif query.lower() == 'settings':
                self._show_current_settings()
                continue
            
            if not query:
                continue
            
            # Process enhanced query
            print(f"\n🔍 Processing query with enhanced pipeline...")
            response = self.query_enhanced(query, k=10, rerank_top_k=5, save_to_storage=True)
            session_qa_count += 1
            
            if "error" in response:
                print(f"❌ Error: {response['error']}")
                continue
            
            print(f"\n🤖 Enhanced Answer:")
            print("-" * 80)
            print(response["answer"])
            
            if response["retrieved_documents"]:
                print(f"\n📊 Retrieval Statistics:")
                print(f"  Documents retrieved: {response['total_retrieved']}")
                print(f"  Response time: {response.get('response_time', 0):.2f}s")
                print(f"  Query method: {response.get('query_method', 'standard')}")
                
                print(f"\n📄 Top 3 documents (after re-ranking):")
                for i, doc in enumerate(response["retrieved_documents"][:3]):
                    source = doc.get("metadata", {}).get("type", "document")
                    date = doc.get("metadata", {}).get("date", "unknown")
                    similarity = doc.get("similarity_score", 0)
                    
                    if 'cross_encoder_score' in doc:
                        print(f"  {i+1}. [{source}, {date}] Vector: {similarity:.4f}, Cross-encoder: {doc['cross_encoder_score']:.4f}")
                    else:
                        print(f"  {i+1}. [{source}, {date}] Similarity: {similarity:.4f}")
                    
                    preview = doc['document'][:150] + "..." if len(doc['document']) > 150 else doc['document']
                    print(f"     Preview: {preview}")
    
    def _show_system_statistics(self):
        """Show system statistics"""
        print("\n📊 System Statistics:")
        print(f"  FAISS index type: {self.enhanced_index.index_type}")
        print(f"  Total chunks indexed: {self.enhanced_index.stats['total_chunks']}")
        print(f"  Embedding dimension: {self.enhanced_index.dimension}")
        print(f"  Chunker model: {self.chunker.model_name}")
        print(f"  Cross-encoder model: {self.reranker.model_name if self.reranker.initialized else 'Not loaded'}")
        print(f"  Total Q&A in storage: {len(self.qa_storage.history)}")
    
    def _show_chunking_statistics(self):
        """Show chunking statistics"""
        if not self.enhanced_index.chunks:
            print("No chunks available")
            return
        
        chunk_sizes = [self.chunker.count_tokens(chunk.text) for chunk in self.enhanced_index.chunks[:100]]  # Sample 100
        
        print("\n📊 Chunking Statistics (sample of 100 chunks):")
        print(f"  Total chunks: {len(self.enhanced_index.chunks)}")
        print(f"  Average chunk size: {np.mean(chunk_sizes):.1f} tokens")
        print(f"  Min chunk size: {np.min(chunk_sizes)} tokens")
        print(f"  Max chunk size: {np.max(chunk_sizes)} tokens")
        print(f"  Target chunk size: {self.chunker.chunk_size} tokens")
        print(f"  Overlap: {self.chunker.overlap_tokens} tokens")
    
    def _show_current_settings(self):
        """Show current system settings"""
        print("\n⚙️ Current Settings:")
        print(f"  Embedding model: all-MiniLM-L6-v2")
        print(f"  Chunk size: {self.chunker.chunk_size} tokens")
        print(f"  Overlap: {self.chunker.overlap_tokens} tokens")
        print(f"  FAISS index type: {self.enhanced_index.index_type}")
        print(f"  Cross-encoder: {self.reranker.model_name}")
        print(f"  Default retrieval k: 10")
        print(f"  Default re-ranking k: 5")

# =====================================================
# SIMPLE RAG SYSTEM (For quick testing)
# =====================================================

class SimpleLincolnRAGSystem:
    """Simple RAG system using FlatL2 FAISS index"""
    
    def __init__(self):
        self.embedding_model = None
        self.faiss_index = None
        self.documents = []
        self.dimension = 384
        
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("✅ Embedding model loaded")
            
            # Initialize FAISS index
            self._initialize_faiss()
            
        except ImportError as e:
            logger.error(f"❌ Missing dependencies: {e}")
    
    def _initialize_faiss(self):
        """Initialize FAISS FlatL2 index"""
        try:
            import faiss
            self.faiss_index = faiss.IndexFlatL2(self.dimension)
            logger.info(f"✅ FAISS FlatL2 index initialized (dimension: {self.dimension})")
            return True
        except ImportError as e:
            logger.error(f"❌ FAISS not available: {e}")
            return False
    
    def index_documents(self, documents_dir: str, output_dir: str) -> Dict:
        """Index documents with simple approach"""
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.faiss_index:
            logger.error("FAISS index not initialized")
            return {"success": False, "error": "FAISS index not initialized"}
        
        # Get list of JSON files
        json_files = [f for f in os.listdir(documents_dir) if f.endswith('.json')]
        
        if not json_files:
            logger.error(f"No JSON files found in {documents_dir}")
            return {"success": False, "error": "No documents found"}
        
        print(f"Found {len(json_files)} documents")
        
        documents_data = []
        embeddings_list = []
        
        for filename in tqdm(json_files, desc="Indexing documents"):
            try:
                filepath = os.path.join(documents_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                
                # Extract text
                text = ""
                if "content" in doc and "full_text" in doc["content"]:
                    text = doc["content"]["full_text"]
                elif "full_text" in doc:
                    text = doc["full_text"]
                elif "text" in doc:
                    text = doc["text"]
                
                if text and len(text.strip()) > 100:
                    # Generate embedding
                    embedding = self.embedding_model.encode(text[:2000]).astype('float32')
                    
                    # Store data
                    doc_data = {
                        "id": doc.get("document_id", filename.replace('.json', '')),
                        "text": text.strip(),
                        "metadata": doc.get("metadata", {}),
                        "embedding": embedding
                    }
                    
                    documents_data.append(doc_data)
                    embeddings_list.append(embedding)
                    
            except Exception as e:
                logger.warning(f"Failed to process {filename}: {e}")
        
        if not documents_data:
            logger.error("No documents could be processed")
            return {"success": False, "error": "No documents processed"}
        
        print(f"\nSuccessfully processed {len(documents_data)} documents")
        
        # Add to FAISS index
        try:
            embeddings_array = np.vstack(embeddings_list).astype('float32')
            self.faiss_index.add(embeddings_array)
            
            # Store documents
            self.documents = documents_data
            
            # Save index
            self._save_index(output_dir)
            
            return {
                "success": True,
                "documents_indexed": len(documents_data),
                "index_size": self.faiss_index.ntotal
            }
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
            return {"success": False, "error": str(e)}
    
    def _save_index(self, output_dir: str):
        """Save FAISS index"""
        try:
            import faiss
            index_path = os.path.join(output_dir, "faiss_index_simple.bin")
            faiss.write_index(self.faiss_index, index_path)
            
            # Save documents
            docs_path = os.path.join(output_dir, "documents.json")
            with open(docs_path, 'w', encoding='utf-8') as f:
                json.dump([{
                    "id": doc["id"],
                    "text": doc["text"],
                    "metadata": doc["metadata"]
                } for doc in self.documents], f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Index saved to {index_path}")
            logger.info(f"✅ Documents saved to {docs_path}")
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    def query(self, query_text: str, k: int = 5) -> Dict:
        """Query the index"""
        if not self.faiss_index or self.faiss_index.ntotal == 0:
            return {"error": "Index not initialized or empty"}
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query_text).astype('float32').reshape(1, -1)
            
            # Search FAISS index
            distances, indices = self.faiss_index.search(query_embedding, k)
            
            # Process results
            retrieved_docs = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and idx < len(self.documents):
                    doc = self.documents[idx]
                    similarity_score = 1 / (1 + distances[0][i])
                    
                    retrieved_docs.append({
                        "id": doc["id"],
                        "document": doc["text"],
                        "metadata": doc["metadata"],
                        "similarity_score": float(similarity_score),
                        "rank": i + 1
                    })
            
            # Generate simple answer
            answer = self._generate_answer(query_text, retrieved_docs)
            
            return {
                "query": query_text,
                "answer": answer,
                "retrieved_documents": retrieved_docs,
                "total_retrieved": len(retrieved_docs)
            }
            
        except Exception as e:
            logger.error(f"Error querying index: {e}")
            return {"error": str(e)}
    
    def _generate_answer(self, query: str, documents: List[Dict]) -> str:
        """Generate simple answer"""
        if not documents:
            return "I couldn't find any relevant documents."
        
        # Extract snippets
        snippets = [doc["document"][:300] + "..." for doc in documents[:3]]
        context = " ".join(snippets)
        
        return f"Based on Lincoln's documents: {context[:500]}..."

# =====================================================
# MAIN EXECUTION
# =====================================================

if __name__ == "__main__":
    import os
    import sys
    
    # Get the project structure
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Define paths
    enhanced_data_dir = os.path.join(project_root, "data_processing", "outputs", "enhanced_data")
    output_dir = os.path.join(project_root, "llm_integration", "outputs", "rag_results_enhanced")
    
    print("="*80)
    print("ENHANCED LINCOLN RAG SYSTEM")
    print("="*80)
    print("Features: Token-based chunking, Cross-encoder re-ranking, FAISS FlatL2, Evaluation metrics")
    print("="*80)
    
    print(f"Project root: {project_root}")
    print(f"Enhanced data directory: {enhanced_data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Check if enhanced data exists
    if not os.path.exists(enhanced_data_dir):
        print(f"\n❌ ERROR: Enhanced data directory does not exist!")
        print(f"Please run the metadata extractor first.")
        sys.exit(1)
    
    # Check for documents
    json_files = [f for f in os.listdir(enhanced_data_dir) if f.endswith('.json') 
                 and f not in ["ALL_DOCUMENTS.json", "enhancement_report.json"]]
    
    if not json_files:
        print(f"\n❌ No JSON files found in {enhanced_data_dir}")
        sys.exit(1)
    
    print(f"\nFound {len(json_files)} documents for RAG system")
    
    # Ask user which system to use
    print("\n" + "="*60)
    print("CHOOSE RAG SYSTEM TYPE")
    print("="*60)
    print("1. Enhanced System (Recommended)")
    print("   - Token-based chunking")
    print("   - Cross-encoder re-ranking")
    print("   - Evaluation metrics")
    print("   - FAISS FlatL2 index")
    print("\n2. Simple System (Fast)")
    print("   - No chunking (full documents)")
    print("   - FAISS FlatL2 index")
    print("   - Basic querying")
    print("="*60)
    
    choice = input("\nSelect system (1 or 2, default=1): ").strip()
    
    if choice == "2":
        # Use simple system
        print("\nInitializing Simple Lincoln RAG System...")
        rag_system = SimpleLincolnRAGSystem()
        
        # Check if index already exists
        index_exists = os.path.exists(os.path.join(output_dir, "faiss_index_simple.bin"))
        
        if not index_exists:
            print("\n" + "="*60)
            print("INDEXING DOCUMENTS (SIMPLE SYSTEM)")
            print("="*60)
            
            result = rag_system.index_documents(enhanced_data_dir, output_dir)
            
            if not result.get("success", False):
                print(f"❌ Failed to index documents: {result.get('error', 'Unknown error')}")
                sys.exit(1)
            
            print(f"\n✅ Successfully indexed {result['documents_indexed']} documents")
            print(f"✅ FAISS index created with {result['index_size']} vectors")
        else:
            print(f"\n✅ Found existing simple index in {output_dir}")
            # Note: Simple system doesn't have load functionality in this version
            print("⚠️ Please delete the index files to recreate")
        
        # Test queries
        print("\n" + "="*80)
        print("TESTING SIMPLE RAG SYSTEM")
        print("="*80)
        
        test_queries = [
            "What was Lincoln's view on slavery?",
            "How did Lincoln describe the Union?",
            "Tell me about the Gettysburg Address"
        ]
        
        for query in test_queries:
            print(f"\n{'─'*80}")
            print(f"Query: {query}")
            
            response = rag_system.query(query, k=5)
            
            if "error" in response:
                print(f"Error: {response['error']}")
                continue
            
            print(f"\n📝 Answer:")
            print("-" * 40)
            print(response["answer"])
            
            print(f"\n📄 Retrieved {response['total_retrieved']} documents")
        
        print("\n" + "="*80)
        print("SIMPLE RAG SYSTEM READY")
        print("="*80)
        print("\nTo use in your app:")
        print("from rag_pipeline import SimpleLincolnRAGSystem")
        print("rag = SimpleLincolnRAGSystem()")
        print("response = rag.query('Your question here')")
        
    else:
        # Use enhanced system (default)
        print("\nInitializing Enhanced Lincoln RAG System...")
        
        rag_system = EnhancedLincolnRAGSystem()
        
        # Initialize components
        if not rag_system.initialize_components():
            print("❌ Failed to initialize components")
            sys.exit(1)
        
        # Check if index already exists
        index_exists = os.path.exists(os.path.join(output_dir, "faiss_index.bin"))
        
        if not index_exists:
            # Index documents
            print("\n" + "="*60)
            print("INDEXING DOCUMENTS WITH ENHANCED PIPELINE")
            print("="*60)
            
            # Ask user for settings
            use_all = input(f"\nIndex ALL {len(json_files)} documents? (y/n, default=y): ").strip().lower()
            
            if use_all == 'n':
                try:
                    sample_size = int(input(f"Enter sample size (max {len(json_files)}): "))
                    sample_size = min(sample_size, len(json_files))
                    print(f"Indexing sample of {sample_size} documents")
                except:
                    sample_size = 200
                    print(f"Using default sample size: {sample_size}")
            else:
                sample_size = None
                print(f"Indexing ALL {len(json_files)} documents")
            
            # Index documents
            print(f"\nStarting enhanced indexing...")
            result = rag_system.index_documents_enhanced(enhanced_data_dir, output_dir, sample_size=sample_size)
            
            if not result.get("success", False):
                print(f"❌ Failed to index documents: {result.get('error', 'Unknown error')}")
                sys.exit(1)
            
            print(f"\n✅ Successfully indexed documents")
            print(f"  Documents processed: {result.get('documents_processed', 0)}")
            print(f"  Chunks created: {result.get('chunks_created', 0)}")
            print(f"  Average chunks per document: {result.get('average_chunks_per_document', 0):.1f}")
            print(f"  FAISS index type: {result.get('index_type', 'FlatL2')}")
            
        else:
            print(f"\n✅ Found existing enhanced index in {output_dir}")
            print("Loading existing index...")
            
            # Try to load existing index
            if not rag_system.enhanced_index.load(output_dir):
                print("❌ Failed to load existing index.")
                sys.exit(1)
            
            print(f"Loaded index with {rag_system.enhanced_index.stats['total_chunks']} chunks")
            print(f"Index type: {rag_system.enhanced_index.index_type}")
        
        # Test enhanced queries
        print("\n" + "="*80)
        print("TESTING ENHANCED RAG SYSTEM")
        print("="*80)
        
        test_queries = [
            "What was Lincoln's view on slavery?",
            "How did Lincoln describe the Union?",
            "Write in the tone of an 1860s presidential proclamation",
            "Compose a speech about preserving the Union",
            "What were the key principles in Lincoln's Gettysburg Address?"
        ]
        
        for query in test_queries:
            print(f"\n{'─'*80}")
            print(f"Query: {query}")
            
            response = rag_system.query_enhanced(query, k=10, rerank_top_k=5, save_to_storage=True)
            
            if "error" in response:
                print(f"Error: {response['error']}")
                continue
            
            print(f"\n📝 Enhanced Answer:")
            print("-" * 40)
            print(response["answer"])
            
            print(f"\n📊 Statistics:")
            print(f"  Response time: {response.get('response_time', 0):.2f}s")
            print(f"  Documents retrieved: {response['total_retrieved']}")
            print(f"  Top similarity: {response.get('top_similarity', 0):.4f}")
            
            if response["retrieved_documents"] and 'cross_encoder_score' in response["retrieved_documents"][0]:
                print(f"  Cross-encoder score: {response['retrieved_documents'][0]['cross_encoder_score']:.4f}")
        
        # Run comprehensive evaluation
        print("\n" + "="*80)
        print("RUNNING COMPREHENSIVE EVALUATION")
        print("="*80)
        
        evaluation = rag_system.evaluate_rag_system_comprehensive()
        
        if evaluation:
            # Save evaluation
            eval_file = os.path.join(output_dir, "rag_evaluation_comprehensive.json")
            with open(eval_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Evaluation saved to: {eval_file}")
        
        # Summary
        print("\n" + "="*80)
        print("ENHANCED RAG SYSTEM IMPLEMENTATION COMPLETE")
        print("="*80)
        print(f"\n✅ System ready with {rag_system.enhanced_index.stats['total_chunks']} indexed chunks")
        print(f"✅ Using {rag_system.enhanced_index.index_type} FAISS index")
        print(f"✅ Token-based chunking with {rag_system.chunker.model_name}")
        print(f"✅ Cross-encoder re-ranking: {'Enabled' if rag_system.reranker.initialized else 'Disabled'}")
        print(f"✅ Comprehensive evaluation metrics implemented")
        print(f"✅ Results saved in: {output_dir}")
        
        # Show Q&A storage
        print(f"\n📁 Q&A Storage:")
        print(f"   JSON file: {rag_system.qa_storage.qa_file}")
        print(f"   Statistics: {rag_system.qa_storage.stats_file}")
        
        # Ask about interactive mode
        response = input("\n🚀 Enter enhanced interactive mode? (y/n): ")
        if response.lower() == 'y':
            rag_system.interactive_enhanced_mode()
    
    print("\n" + "="*80)
    print("THANK YOU FOR USING LINCOLN RAG SYSTEM")
    print("="*80)
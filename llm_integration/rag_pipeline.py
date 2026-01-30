import json
import os
import logging
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm
import time
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
            # Save FAISS index
            index_path = os.path.join(output_dir, "faiss_index.bin")
            self.faiss.write_index(self.faiss_index, index_path)
            
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
            index_path = os.path.join(input_dir, "faiss_index.bin")
            metadata_path = os.path.join(input_dir, "documents_metadata.json")
            
            if not os.path.exists(index_path) or not os.path.exists(metadata_path):
                return False
            
            # Load FAISS index
            import faiss
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

# Main execution
if __name__ == "__main__":
    import sys
    import os
    
    # Get the project structure
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # Go up one level from llm_integration
    
    # Define paths
    enhanced_data_dir = os.path.join(project_root, "data_processing", "outputs", "enhanced_data")
    output_dir = os.path.join(project_root, "llm_integration", "outputs", "rag_results")
    
    print("="*80)
    print("LINCOLN RAG SYSTEM WITH FAISS - ENHANCED DOCUMENT VIEWING")
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
    
    # Initialize RAG system
    print("\nInitializing Lincoln RAG System (FAISS)...")
    
    # Try to initialize
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        print("✅ Required packages available")
    except ImportError as e:
        print(f"❌ Missing packages: {e}")
        print("\nPlease install required packages:")
        print("pip install sentence-transformers faiss-cpu")
        sys.exit(1)
    
    rag_system = LincolnRAGSystem()
    
    # Check if index already exists
    index_exists = os.path.exists(os.path.join(output_dir, "faiss_index.bin"))
    
    if not index_exists:
        # Index documents - ASK USER IF THEY WANT TO USE ALL DOCUMENTS
        print("\n" + "="*60)
        print("INDEXING DOCUMENTS WITH FAISS")
        print("="*60)
        
        # Ask user if they want to use all documents
        use_all = input(f"\nIndex ALL {len(json_files)} documents? (y/n, default=y): ").strip().lower()
        
        if use_all == 'n':
            # Ask for sample size
            try:
                sample_size = int(input(f"Enter sample size (max {len(json_files)}): "))
                sample_size = min(sample_size, len(json_files))
                print(f"Indexing sample of {sample_size} documents")
            except:
                sample_size = 200  # Default sample size
                print(f"Using default sample size: {sample_size}")
        else:
            sample_size = None  # Use all documents
            print(f"Indexing ALL {len(json_files)} documents")
        
        result = rag_system.index_documents(enhanced_data_dir, output_dir, sample_size=sample_size)
        
        if not result.get("success", False):
            print(f"❌ Failed to index documents: {result.get('error', 'Unknown error')}")
            sys.exit(1)
        
        print(f"✅ Successfully indexed {result['documents_indexed']} documents")
        print(f"✅ FAISS index created with {result['index_size']} vectors")
    else:
        print(f"✅ Found existing FAISS index in {output_dir}")
        print("Loading existing index...")
        
        # Try to load existing index
        if not rag_system._load_index(output_dir):
            print(" Failed to load existing index. Re-indexing...")
            
            # Ask user if they want to use all documents
            use_all = input(f"\nIndex ALL {len(json_files)} documents? (y/n, default=y): ").strip().lower()
            
            if use_all == 'n':
                sample_size = 200  # Default sample size
                print(f"Indexing sample of {sample_size} documents")
            else:
                sample_size = None  # Use all documents
                print(f"Indexing ALL {len(json_files)} documents")
            
            result = rag_system.index_documents(enhanced_data_dir, output_dir, sample_size=sample_size)
            if not result.get("success", False):
                print(f" Failed to index documents")
                sys.exit(1)
    
    # Test queries
    print("\n" + "="*80)
    print("TESTING ENHANCED RAG SYSTEM WITH DOCUMENT VIEWING")
    print("="*80)
    print("\nTesting both factual and creative queries:")
    
    test_queries = [
        "What was Lincoln's view on slavery?",  # Factual
        "How did Lincoln describe the Union?",   # Factual
        "Write in the tone of an 1860s presidential proclamation",  # Creative
        "Compose a speech about preserving the Union",  # Creative
        "Draft a letter about democratic principles"    # Creative
    ]
    
    # Run test queries
    for query in test_queries:
        print(f"\n{'─'*80}")
        print(f"Query: {query}")
        response = rag_system.query(query, k=5, save_to_storage=True)  # Get 5 documents
        
        if "error" in response:
            print(f"Error: {response['error']}")
            continue
        
        print(f"\n📝 Answer:")
        print("-" * 40)
        print(response["answer"])
        
        print(f"\n📊 Retrieved {response['total_retrieved']} documents")
        if response["retrieved_documents"]:
            print(f"Top similarity score: {response['retrieved_documents'][0]['similarity_score']:.4f}")
            
            # Show document previews
            print(f"\n📄 Document previews (top 3):")
            for i, doc in enumerate(response["retrieved_documents"][:3]):
                source = doc.get("metadata", {}).get("type", "document")
                date = doc.get("metadata", {}).get("date", "unknown")
                similarity = doc.get("similarity_score", 0)
                print(f"  {i+1}. [{source}, {date}] Similarity: {similarity:.4f}")
    
    # Run evaluation
    print("\n" + "="*80)
    print("SYSTEM EVALUATION")
    print("="*80)
    
    evaluation = rag_system.evaluate_rag_system(test_queries[:3])
    
    if evaluation:
        # Save evaluation
        eval_file = os.path.join(output_dir, "rag_evaluation_enhanced.json")
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Evaluation saved to: {eval_file}")
    
    # Summary
    print("\n" + "="*80)
    print("ENHANCED RAG SYSTEM IMPLEMENTATION COMPLETE")
    print("="*80)
    print(f"\n✅ System ready with {rag_system.faiss_index.ntotal if rag_system.faiss_index else 0} indexed documents")
    print(f"✅ Using FAISS for vector similarity search")
    print(f"✅ Enhanced prompt engineering for creative queries")
    print(f"✅ Full document viewing capability")
    print(f"✅ Results saved in: {output_dir}")
    
    # Show Q&A storage location
    print(f"\n📁 Q&A Storage Location:")
    print(f"   JSON file: {rag_system.qa_storage.qa_file}")
    print(f"   Statistics: {rag_system.qa_storage.stats_file}")
    
    # Ask about interactive mode
    response = input("\n🚀 Enter interactive query mode? (y/n): ")
    if response.lower() == 'y':
        rag_system.interactive_query_mode()
    
    print("\n" + "="*80)
    print("THANK YOU FOR USING LINCOLN RAG SYSTEM")
    print("="*80)
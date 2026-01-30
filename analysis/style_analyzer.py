import nltk
import json
import os
from typing import Dict, List
import numpy as np
from collections import Counter
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LincolnStyleAnalyzer:
    """Analyzes Lincoln's writing style patterns without spaCy dependency"""
    
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('punkt_tab', quiet=True)  # Add this line
            nltk.download('wordnet', quiet=True)  # Add wordnet for better tokenization
            logger.info("NLTK data ready")
        except Exception as e:
            logger.warning(f"Could not download NLTK data: {e}")
            print("Trying to download NLTK data manually...")
            try:
                import ssl
                try:
                    _create_unverified_https_context = ssl._create_unverified_context
                except AttributeError:
                    pass
                else:
                    ssl._create_default_https_context = _create_unverified_https_context
                
                nltk.download('punkt')
                nltk.download('averaged_perceptron_tagger')
                nltk.download('punkt_tab')
                nltk.download('wordnet')
                logger.info("NLTK data downloaded successfully")
            except Exception as e2:
                logger.error(f"Failed to download NLTK data: {e2}")
        
    def analyze_document_style(self, text: str) -> Dict:
        """Analyze the writing style of a document using NLTK only"""
        if not text or len(text.strip()) < 50:
            return self._get_empty_analysis()
        
        try:
            # Simple text cleaning
            text = text.strip()
            
            # Use a simpler sentence tokenizer if punkt_tab fails
            try:
                sentences = nltk.sent_tokenize(text)
            except:
                # Fallback to simple sentence splitting
                sentences = [s.strip() for s in text.split('.') if s.strip()]
            
            # Simple word tokenization
            words = []
            for sentence in sentences:
                # Basic word splitting
                sentence_words = sentence.split()
                words.extend(sentence_words)
            
            total_words = len(words)
            total_sentences = len(sentences)
            
            if total_words == 0 or total_sentences == 0:
                return self._get_empty_analysis()
            
            # Convert numpy types to Python types for JSON serialization
            avg_sentence_length = float(total_words / total_sentences)
            
            # Calculate average word length
            word_lengths = [len(word) for word in words if any(c.isalpha() for c in word)]
            avg_word_length = float(np.mean(word_lengths)) if word_lengths else 0.0
            
            # Simple POS analysis (basic approximation)
            pos_counts = {}
            for word in words:
                if len(word) > 3 and word[0].isupper():
                    pos_counts["PROPN"] = pos_counts.get("PROPN", 0) + 1
                elif word.lower() in ["the", "a", "an", "this", "that", "these", "those"]:
                    pos_counts["DET"] = pos_counts.get("DET", 0) + 1
                elif word.lower() in ["is", "am", "are", "was", "were", "be", "been", "being"]:
                    pos_counts["VERB"] = pos_counts.get("VERB", 0) + 1
                elif word.lower().endswith(('ing', 'ed', 'en')):
                    pos_counts["VERB"] = pos_counts.get("VERB", 0) + 1
                elif word.lower().endswith(('ly')):
                    pos_counts["ADV"] = pos_counts.get("ADV", 0) + 1
                elif word.lower().endswith(('able', 'ible', 'ful', 'ous', 'ish', 'ive')):
                    pos_counts["ADJ"] = pos_counts.get("ADJ", 0) + 1
                elif any(c.isdigit() for c in word):
                    pos_counts["NUM"] = pos_counts.get("NUM", 0) + 1
                else:
                    pos_counts["NOUN"] = pos_counts.get("NOUN", 0) + 1
            
            pos_distribution = {pos: float(count/total_words) for pos, count in pos_counts.items()}
            
            # Vocabulary richness
            unique_words = set(word.lower() for word in words if any(c.isalpha() for c in word))
            lexical_diversity = float(len(unique_words) / total_words) if total_words > 0 else 0.0
            
            # Lincoln-specific patterns
            lincoln_patterns = self._analyze_lincoln_patterns(text)
            
            return {
                "basic_stats": {
                    "total_words": total_words,
                    "total_sentences": total_sentences,
                    "avg_sentence_length": avg_sentence_length,
                    "avg_word_length": avg_word_length,
                    "lexical_diversity": lexical_diversity
                },
                "pos_distribution": pos_distribution,
                "lincoln_patterns": lincoln_patterns,
                "readability": self._calculate_readability_simple(text)
            }
        except Exception as e:
            logger.error(f"Error analyzing document: {e}")
            return self._get_empty_analysis()
    
    def _get_empty_analysis(self) -> Dict:
        """Return empty analysis structure"""
        return {
            "basic_stats": {
                "total_words": 0,
                "total_sentences": 0,
                "avg_sentence_length": 0.0,
                "avg_word_length": 0.0,
                "lexical_diversity": 0.0
            },
            "pos_distribution": {},
            "lincoln_patterns": {
                "biblical_allusions": 0,
                "legal_terms": 0,
                "metaphors": 0,
                "parallel_structure": 0,
                "repetition": 0,
                "humble_tone": 0,
                "firm_conviction": 0
            },
            "readability": {"flesch_reading_ease": 0.0, "flesch_kincaid_grade": 0.0}
        }
    
    def _analyze_lincoln_patterns(self, text: str) -> Dict:
        """Analyze patterns characteristic of Lincoln's writing"""
        if not text:
            return {
                "biblical_allusions": 0,
                "legal_terms": 0,
                "metaphors": 0,
                "parallel_structure": 0,
                "repetition": 0,
                "humble_tone": 0,
                "firm_conviction": 0
            }
        
        patterns = {
            "biblical_allusions": 0,
            "legal_terms": 0,
            "metaphors": 0,
            "parallel_structure": 0,
            "repetition": 0,
            "humble_tone": 0,
            "firm_conviction": 0
        }
        
        text_lower = text.lower()
        
        # Check for biblical allusions
        biblical_phrases = ["god", "providence", "almighty", "created", "bless", "divine", "heaven", "sacred"]
        patterns["biblical_allusions"] = sum(text_lower.count(phrase) for phrase in biblical_phrases)
        
        # Check for legal terms
        legal_terms = ["whereas", "therefore", "hereby", "aforesaid", "witness", "contract", "constitution", "law", "rights"]
        patterns["legal_terms"] = sum(text_lower.count(term) for term in legal_terms)
        
        # Check for metaphors
        metaphor_indicators = ["like a", "as a", "is a", "was a", "are a", "like the", "as the"]
        patterns["metaphors"] = sum(text_lower.count(indicator) for indicator in metaphor_indicators)
        
        # Check for parallel structure
        parallel_phrases = ["not only", "but also", "either", "or", "both", "and", "neither", "nor"]
        patterns["parallel_structure"] = sum(text_lower.count(phrase) for phrase in parallel_phrases)
        
        # Check for repetition (simplified)
        words = text_lower.split()
        word_counts = {}
        for word in words:
            if len(word) > 3:  # Only count meaningful words
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Count words that appear more than once
        patterns["repetition"] = sum(1 for count in word_counts.values() if count > 1)
        
        # Analyze tone
        humble_words = ["humble", "modest", "unworthy", "endeavor", "attempt", "try", "hope"]
        firm_words = ["must", "shall", "will", "determined", "resolved", "certain", "sure"]
        
        patterns["humble_tone"] = sum(text_lower.count(word) for word in humble_words)
        patterns["firm_conviction"] = sum(text_lower.count(word) for word in firm_words)
        
        return patterns
    
    def _calculate_readability_simple(self, text: str) -> Dict:
        """Calculate simplified readability scores"""
        if not text:
            return {"flesch_reading_ease": 0.0, "flesch_kincaid_grade": 0.0}
        
        try:
            # Simple sentence splitting
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            words = text.split()
            
            total_sentences = len(sentences)
            total_words = len(words)
            
            if total_sentences == 0 or total_words == 0:
                return {"flesch_reading_ease": 0.0, "flesch_kincaid_grade": 0.0}
            
            # Estimate syllables (approximate)
            total_syllables = 0
            for word in words:
                word_lower = word.lower()
                vowels = 'aeiouy'
                syllables = 0
                
                if word_lower:
                    if word_lower[0] in vowels:
                        syllables += 1
                    
                    for i in range(1, len(word_lower)):
                        if word_lower[i] in vowels and word_lower[i-1] not in vowels:
                            syllables += 1
                    
                    if word_lower.endswith('e'):
                        syllables -= 1
                    
                    syllables = max(syllables, 1)
                    total_syllables += syllables
            
            # Flesch Reading Ease (simplified)
            flesch_ease = 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
            
            # Flesch-Kincaid Grade Level (simplified)
            flesch_grade = 0.39 * (total_words / total_sentences) + 11.8 * (total_syllables / total_words) - 15.59
            
            return {
                "flesch_reading_ease": float(flesch_ease),
                "flesch_kincaid_grade": float(flesch_grade)
            }
        except:
            return {"flesch_reading_ease": 0.0, "flesch_kincaid_grade": 0.0}
    
    def _convert_to_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    def analyze_corpus_styles(self, documents_dir: str, output_dir: str, sample_size: int = 50) -> Dict:
        """Analyze styles across multiple documents"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Get list of JSON files
        json_files = [f for f in os.listdir(documents_dir) if f.endswith('.json') 
                     and f not in ["ALL_DOCUMENTS.json", "enhancement_report.json"]]
        
        if not json_files:
            logger.error(f"No JSON files found in {documents_dir}")
            return {}
        
        print(f"Found {len(json_files)} documents")
        
        # Use all files or sample
        if sample_size and len(json_files) > sample_size:
            import random
            json_files = random.sample(json_files, sample_size)
            print(f"Analyzing random sample of {sample_size} documents")
        else:
            print(f"Analyzing all {len(json_files)} documents")
        
        all_analyses = []
        failed_files = []
        documents_with_text = 0
        
        for filename in tqdm(json_files, desc="Analyzing document styles"):
            try:
                filepath = os.path.join(documents_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                
                # Try different content locations
                text = ""
                if "content" in doc and "full_text" in doc["content"]:
                    text = doc["content"]["full_text"]
                elif "full_text" in doc:
                    text = doc["full_text"]
                elif "text" in doc:
                    text = doc["text"]
                
                # Check if we have sufficient text
                if text and len(text.strip()) > 100:
                    analysis = self.analyze_document_style(text)
                    analysis["document_id"] = doc.get("document_id", filename.replace('.json', ''))
                    analysis["document_type"] = doc.get("metadata", {}).get("type", "unknown")
                    
                    # Convert all values to serializable types
                    analysis = self._convert_to_serializable(analysis)
                    
                    all_analyses.append(analysis)
                    documents_with_text += 1
                else:
                    failed_files.append(f"{filename}: Text too short ({len(text) if text else 0} chars)")
                
            except json.JSONDecodeError as e:
                failed_files.append(f"{filename}: Invalid JSON - {e}")
            except Exception as e:
                failed_files.append(f"{filename}: {type(e).__name__}: {e}")
        
        print(f"\nDocuments with sufficient text: {documents_with_text}/{len(json_files)}")
        print(f"Failed analyses: {len(failed_files)}")
        
        if not all_analyses:
            logger.error("No documents could be analyzed")
            # Debug: Check first few files
            for i in range(min(3, len(json_files))):
                filename = json_files[i]
                filepath = os.path.join(documents_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        sample = json.load(f)
                    print(f"\nSample {i+1} ({filename}):")
                    print(f"  Keys: {list(sample.keys())}")
                    if "content" in sample:
                        print(f"  Content keys: {list(sample['content'].keys())}")
                        if "full_text" in sample["content"]:
                            text = sample["content"]["full_text"]
                            print(f"  Text length: {len(text)}")
                            print(f"  Preview: {text[:200]}...")
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
            return {}
        
        # Aggregate statistics with serializable types
        try:
            avg_sentence_length = np.mean([a["basic_stats"]["avg_sentence_length"] for a in all_analyses])
            std_sentence_length = np.std([a["basic_stats"]["avg_sentence_length"] for a in all_analyses])
            avg_word_length = np.mean([a["basic_stats"]["avg_word_length"] for a in all_analyses])
            std_word_length = np.std([a["basic_stats"]["avg_word_length"] for a in all_analyses])
            avg_lexical_diversity = np.mean([a["basic_stats"]["lexical_diversity"] for a in all_analyses])
            std_lexical_diversity = np.std([a["basic_stats"]["lexical_diversity"] for a in all_analyses])
            avg_flesch_ease = np.mean([a["readability"]["flesch_reading_ease"] for a in all_analyses])
            avg_flesch_grade = np.mean([a["readability"]["flesch_kincaid_grade"] for a in all_analyses])
        except Exception as e:
            print(f"Error calculating statistics: {e}")
            return {}
        
        aggregate = {
            "total_documents_analyzed": len(all_analyses),
            "total_files_checked": len(json_files),
            "failed_documents": len(failed_files),
            "documents_with_text": documents_with_text,
            "avg_sentence_length": float(avg_sentence_length),
            "std_sentence_length": float(std_sentence_length),
            "avg_word_length": float(avg_word_length),
            "std_word_length": float(std_word_length),
            "avg_lexical_diversity": float(avg_lexical_diversity),
            "std_lexical_diversity": float(std_lexical_diversity),
            "lincoln_patterns_summary": self._summarize_lincoln_patterns(all_analyses),
            "document_types_summary": dict(Counter([a.get("document_type", "unknown") for a in all_analyses])),
            "readability_summary": {
                "avg_flesch_ease": float(avg_flesch_ease),
                "avg_flesch_grade": float(avg_flesch_grade)
            }
        }
        
        # Convert aggregate to serializable types
        aggregate = self._convert_to_serializable(aggregate)
        
        # Save individual analyses
        for i, analysis in enumerate(all_analyses[:5]):  # Save first 5 for inspection
            analysis_file = os.path.join(output_dir, f"style_analysis_{i+1}.json")
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        # Save aggregate report
        report_file = os.path.join(output_dir, "style_analysis_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(aggregate, f, indent=2, ensure_ascii=False)
        
        # Save failed files list
        if failed_files:
            failed_file = os.path.join(output_dir, "failed_analyses.txt")
            with open(failed_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(failed_files[:20]))  # Save first 20 failures
        
        return aggregate
    
    def _summarize_lincoln_patterns(self, analyses: List[Dict]) -> Dict:
        """Summarize Lincoln patterns across all analyses"""
        patterns_summary = {}
        
        if analyses and "lincoln_patterns" in analyses[0]:
            pattern_names = analyses[0]["lincoln_patterns"].keys()
            
            for pattern in pattern_names:
                values = [a["lincoln_patterns"][pattern] for a in analyses]
                patterns_summary[pattern] = {
                    "average": float(np.mean(values)),
                    "std_dev": float(np.std(values)),
                    "max": int(np.max(values)),
                    "min": int(np.min(values))
                }
        
        return patterns_summary

# Main execution
if __name__ == "__main__":
    import sys
    import os
    import time
    
    # Get the correct project structure
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # Go up one level from analysis
    
    # Your actual enhanced data location
    enhanced_data_dir = os.path.join(project_root, "data_processing", "outputs", "enhanced_data")
    
    # Alternative: Check if enhanced_data exists in the expected location
    if not os.path.exists(enhanced_data_dir):
        print(f"Enhanced data not found at: {enhanced_data_dir}")
        
        # Try to find it
        possible_locations = [
            os.path.join(project_root, "outputs", "enhanced_data"),
            os.path.join(project_root, "data_processing", "outputs", "enhanced_data"),
            os.path.join(project_root, "enhanced_data"),
            "F:\\lincoln_llm_project\\data_processing\\outputs\\enhanced_data"
        ]
        
        for location in possible_locations:
            if os.path.exists(location):
                enhanced_data_dir = location
                print(f"Found enhanced data at: {enhanced_data_dir}")
                break
    
    output_dir = os.path.join(project_root, "analysis", "outputs", "style_analysis")
    
    print(f"Project root: {project_root}")
    print(f"Enhanced data directory: {enhanced_data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Check if enhanced data exists
    if not os.path.exists(enhanced_data_dir):
        print(f"\n❌ ERROR: Enhanced data directory does not exist!")
        print(f"Please make sure you've run the metadata extractor first.")
        print(f"\nTo run the metadata extractor:")
        print(f"cd F:\\lincoln_llm_project")
        print(f"python data_processing\\metadata_extractor.py")
        sys.exit(1)
    
    # Check if there are files in enhanced_data
    files_in_dir = os.listdir(enhanced_data_dir)
    json_files = [f for f in files_in_dir if f.endswith('.json')]
    print(f"\nFound {len(files_in_dir)} files in enhanced_data directory")
    print(f"JSON files: {len(json_files)}")
    
    if len(json_files) == 0:
        print("❌ No JSON files found in enhanced_data!")
        print("Please check if the metadata extractor completed successfully.")
        sys.exit(1)
    
    # Initialize analyzer
    print("\nInitializing Lincoln Style Analyzer...")
    analyzer = LincolnStyleAnalyzer()
    
    # Analyze corpus
    print(f"\nStarting style analysis on documents in: {enhanced_data_dir}")
    
    # Use smaller sample for testing
    sample_size = 50  # Start with 50 documents
    aggregate = analyzer.analyze_corpus_styles(
        documents_dir=enhanced_data_dir,
        output_dir=output_dir,
        sample_size=sample_size
    )
    
    if aggregate:
        print("\n" + "="*60)
        print("STYLE ANALYSIS RESULTS")
        print("="*60)
        print(f"Documents analyzed: {aggregate['total_documents_analyzed']}")
        print(f"Average sentence length: {aggregate['avg_sentence_length']:.1f} words")
        print(f"Average word length: {aggregate['avg_word_length']:.1f} characters")
        print(f"Lexical diversity: {aggregate['avg_lexical_diversity']:.3f}")
        print(f"Readability (Flesch-Kincaid): {aggregate['readability_summary']['avg_flesch_grade']:.1f} grade level")
        
        # Show Lincoln patterns
        print(f"\nLincoln Style Patterns (average per document):")
        for pattern, stats in aggregate.get('lincoln_patterns_summary', {}).items():
            print(f"  {pattern}: {stats['average']:.2f}")
        
        # Save detailed report
        report_content = f"""Lincoln Writing Style Analysis Report
==========================================
Analysis Date: {time.strftime("%Y-%m-%d %H:%M:%S")}
Documents Analyzed: {aggregate['total_documents_analyzed']}
Total Files Checked: {aggregate['total_files_checked']}
Failed Analyses: {aggregate['failed_documents']}

BASIC STATISTICS:
-----------------
Average Sentence Length: {aggregate['avg_sentence_length']:.1f} words
Average Word Length: {aggregate['avg_word_length']:.1f} characters
Lexical Diversity: {aggregate['avg_lexical_diversity']:.3f}

READABILITY:
------------
Flesch Reading Ease: {aggregate['readability_summary']['avg_flesch_ease']:.1f}
Flesch-Kincaid Grade Level: {aggregate['readability_summary']['avg_flesch_grade']:.1f}

LINCOLN PATTERNS:
-----------------"""
        
        for pattern, stats in aggregate.get('lincoln_patterns_summary', {}).items():
            report_content += f"\n{pattern}: {stats['average']:.2f} (range: {stats['min']:.0f}-{stats['max']:.0f})"
        
        # Add document type distribution
        if 'document_types_summary' in aggregate:
            report_content += "\n\nDOCUMENT TYPE DISTRIBUTION:"
            report_content += "\n---------------------------"
            for doc_type, count in aggregate['document_types_summary'].items():
                percentage = (count / aggregate['total_documents_analyzed']) * 100
                report_content += f"\n{doc_type}: {count} ({percentage:.1f}%)"
        
        report_file = os.path.join(output_dir, "analysis_summary.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\nDetailed report saved to: {report_file}")
        print(f"Analysis results saved to: {output_dir}")
        
        # Show where files are saved
        if os.path.exists(output_dir):
            output_files = os.listdir(output_dir)
            print(f"\nGenerated files in {output_dir}:")
            for file in output_files:
                print(f"  - {file}")
    else:
        print("\n❌ No analysis could be performed.")
        print("Check if your enhanced_data files have 'full_text' content.")
    
    print("\n" + "="*60)
    print("STYLE ANALYSIS COMPLETE")
    print("="*60)
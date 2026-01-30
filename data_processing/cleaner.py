import re
import json
import os
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class LincolnDataCleaner:
    """Cleans and processes Lincoln documents while preserving authentic language"""
    
    def __init__(self):
        # Patterns for cleaning while preserving historical language
        self.editorial_patterns = [
            r'\[.*?\]',  # Editorial brackets
            r'\{.*?\}',  # Editorial braces
            r'\.\.\.',   # Ellipses in transcriptions
            r'\[image \d+\]',  # Image references
            r'\[table \d+\]',  # Table references
            r'Note:.*',  # Transcriber notes
            r'Transcriber.*',  # Transcriber notes
        ]
        
        # Patterns to preserve
        self.preserve_patterns = [
            r'\b(?:Mr|Mrs|Dr|Gen|Col|Maj|Capt)\.',
            r'\'em\b',  # Historical contraction
            r'ain\'t\b',  # Historical contraction
            r'[£$]',  # Currency symbols
        ]
        
    def clean_document(self, document: Dict, aggressive: bool = False) -> Dict:
        """Clean a single Lincoln document"""
        cleaned_doc = document.copy()
        
        if "content" not in cleaned_doc:
            return cleaned_doc
        
        # Get the text
        full_text = cleaned_doc["content"].get("full_text", "")
        
        # Apply cleaning steps
        cleaned_text = self._clean_text(full_text, aggressive)
        
        # Update document
        cleaned_doc["content"]["full_text"] = cleaned_text
        cleaned_doc["content"]["paragraphs"] = self._split_into_paragraphs(cleaned_text)
        cleaned_doc["content"]["word_count"] = len(cleaned_text.split())
        
        # Add cleaning metadata
        cleaned_doc["processing"] = {
            "cleaned": True,
            "original_length": len(full_text),
            "cleaned_length": len(cleaned_text),
            "preserved_originality": not aggressive
        }
        
        return cleaned_doc
    
    def _clean_text(self, text: str, aggressive: bool) -> str:
        """Apply cleaning operations to text"""
        if not text:
            return ""
        
        # Remove editorial annotations (preserve content inside if it's part of original)
        for pattern in self.editorial_patterns:
            # Check if brackets contain meaningful content (not just editorial notes)
            if '?' in pattern or 'damaged' in text.lower():
                # Preserve partially reconstructed text
                text = re.sub(pattern, '', text)
            else:
                text = re.sub(pattern, '', text)
        
        # Normalize whitespace but preserve paragraph breaks
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Clean up excessive internal whitespace
                line = re.sub(r'\s+', ' ', line)
                
                # Preserve historical spellings and usage
                line = self._preserve_historical_features(line)
                
                cleaned_lines.append(line)
        
        # Reconstruct with paragraph breaks
        cleaned_text = '\n\n'.join(cleaned_lines)
        
        # Remove modern annotations (footnotes, citations) if aggressive
        if aggressive:
            cleaned_text = self._remove_modern_annotations(cleaned_text)
        
        return cleaned_text
    
    def _preserve_historical_features(self, text: str) -> str:
        """Preserve Lincoln's authentic language features"""
        # Don't correct historical spellings
        # Examples: "shew" (show), "tho" (though), "persent" (percent)
        
        # Preserve capitalization patterns (often used for emphasis)
        # Example: "The Union MUST be preserved"
        
        # Preserve punctuation style
        # Example: "--" for dashes, no quotes around dialogue
        
        return text
    
    def _remove_modern_annotations(self, text: str) -> str:
        """Remove modern scholarly annotations"""
        patterns_to_remove = [
            r'See also:.*',
            r'Cf\..*',
            r'Ibid\..*',
            r'\d+ U\.S\. \d+',  # Legal citations
            r'\d+ S\. Ct\. \d+',  # Legal citations
            r'\(\d{4}\)',  # Year citations in parentheses
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text)
        
        return text
    
    def _split_into_paragraphs(self, text: str, min_length: int = 50) -> List[str]:
        """Split text into meaningful paragraphs"""
        paragraphs = text.split('\n\n')
        
        # Filter out very short paragraphs
        meaningful_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if len(para.split()) >= 5:  # At least 5 words
                meaningful_paragraphs.append(para)
        
        # Further split long paragraphs
        final_paragraphs = []
        for para in meaningful_paragraphs:
            if len(para) > 500:  # Split very long paragraphs
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_chunk = []
                current_length = 0
                
                for sentence in sentences:
                    sentence_length = len(sentence)
                    if current_length + sentence_length > 300 and current_chunk:
                        final_paragraphs.append(' '.join(current_chunk))
                        current_chunk = [sentence]
                        current_length = sentence_length
                    else:
                        current_chunk.append(sentence)
                        current_length += sentence_length
                
                if current_chunk:
                    final_paragraphs.append(' '.join(current_chunk))
            else:
                final_paragraphs.append(para)
        
        return final_paragraphs
    
    def batch_clean_documents(self, input_dir: str, output_dir: str) -> Dict:
        """Clean all documents in a directory"""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        cleaning_report = {
            "total_processed": 0,
            "successfully_cleaned": 0,
            "failed": 0,
            "total_words_before": 0,
            "total_words_after": 0,
            "avg_reduction": 0,
            "failed_files": []
        }
        
        # Check if input directory exists
        if not os.path.exists(input_dir):
            print(f"ERROR: Input directory '{input_dir}' does not exist!")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Please make sure you're running from the correct directory.")
            return cleaning_report
        
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Files in input directory: {len(os.listdir(input_dir)) if os.path.exists(input_dir) else 0}")
        
        # Get list of JSON files
        json_files = [f for f in os.listdir(input_dir) if f.endswith('.json') and f != "ALL_DOCUMENTS.json" and f != "STATISTICS.json"]
        
        print(f"Found {len(json_files)} JSON files to process")
        
        if not json_files:
            print("No JSON files found! Please run the scraper first.")
            return cleaning_report
        
        for filename in tqdm(json_files, desc="Cleaning documents"):
            input_path = os.path.join(input_dir, filename)
            
            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    document = json.load(f)
                
                cleaning_report["total_processed"] += 1
                
                # Clean document
                cleaned_doc = self.clean_document(document, aggressive=False)
                
                # Update statistics
                original_words = len(document.get("content", {}).get("full_text", "").split())
                cleaned_words = len(cleaned_doc["content"]["full_text"].split())
                
                cleaning_report["total_words_before"] += original_words
                cleaning_report["total_words_after"] += cleaned_words
                
                # Save cleaned document with the same filename
                output_path = os.path.join(output_dir, filename)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(cleaned_doc, f, indent=2, ensure_ascii=False)
                
                cleaning_report["successfully_cleaned"] += 1
                
            except json.JSONDecodeError as e:
                cleaning_report["failed"] += 1
                cleaning_report["failed_files"].append(f"{filename}: JSON decode error - {e}")
                logger.error(f"JSON decode error in {filename}: {e}")
            except KeyError as e:
                cleaning_report["failed"] += 1
                cleaning_report["failed_files"].append(f"{filename}: Missing key - {e}")
                logger.error(f"Missing key in {filename}: {e}")
            except Exception as e:
                cleaning_report["failed"] += 1
                cleaning_report["failed_files"].append(f"{filename}: {e}")
                logger.error(f"Error cleaning {filename}: {e}")
        
        # Calculate averages
        if cleaning_report["total_processed"] > 0:
            cleaning_report["avg_reduction"] = (
                (cleaning_report["total_words_before"] - cleaning_report["total_words_after"]) / 
                cleaning_report["total_processed"]
            )
        
        return cleaning_report

# Usage
if __name__ == "__main__":
    import sys
    import os
    
    # Get the project root directory (one level up from data_processing)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define correct paths
    input_dir = os.path.join(project_root, "data_collection", "datasets", "raw")
    output_dir = os.path.join(project_root, "data_processing","outputs", "cleaned_data")
    
    print(f"Project root: {project_root}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"\nERROR: Input directory does not exist!")
        print(f"Please make sure you've run the scraper first.")
        print(f"Scraper output should be in: {input_dir}")
        print(f"\nTo run the scraper, go to the data_collection directory:")
        print(f"cd F:\\lincoln_llm_project\\data_collection")
        print(f"python scrapers.py")
        sys.exit(1)
    
    # Run the cleaner
    cleaner = LincolnDataCleaner()
    report = cleaner.batch_clean_documents(input_dir, output_dir)
    
    print("\n" + "="*60)
    print("Cleaning Report:")
    print("="*60)
    print(f"Total processed: {report['total_processed']}")
    print(f"Successfully cleaned: {report['successfully_cleaned']}")
    print(f"Failed: {report['failed']}")
    if report['total_processed'] > 0:
        print(f"Word reduction: {report['avg_reduction']:.1f} words per document")
        print(f"Total words before: {report['total_words_before']:,}")
        print(f"Total words after: {report['total_words_after']:,}")
        print(f"Reduction percentage: {((report['total_words_before'] - report['total_words_after']) / report['total_words_before'] * 100):.1f}%")
    
    if report['failed'] > 0:
        print(f"\nFailed files (first 5):")
        for failed in report['failed_files'][:5]:
            print(f"  - {failed}")
        
        if len(report['failed_files']) > 5:
            print(f"  ... and {len(report['failed_files']) - 5} more")
    
    # Save cleaning report
    try:
        report_file = os.path.join(output_dir, "cleaning_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nCleaning report saved to: {report_file}")
    except Exception as e:
        print(f"\nWarning: Could not save cleaning report: {e}")
import re
import json
import os
import dateutil.parser
from typing import Dict, List, Optional
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class MetadataEnhancer:
    """Enhances document metadata with extracted information"""
    
    DOCUMENT_TYPES = {
        "letter": ["letter", "correspondence", "to ", "dear "],
        "speech": ["speech", "address", "oration", "remarks"],
        "proclamation": ["proclamation", "executive order", "order"],
        "legal": ["legal", "brief", "case", "court", "law"],
        "personal": ["diary", "journal", "note", "memorandum"],
        "telegraph": ["telegram", "telegraph", "wire", "dispatch"],
        "mathematics": ["cipher", "arithmetic", "calculation", "mathematics"],
        "unknown": []
    }
    
    def __init__(self):
        self.date_patterns = [
            r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b',
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{4}\b',  # Just year
        ]
        
    def enhance_metadata(self, document: Dict) -> Dict:
        """Enhance document metadata with extracted information"""
        enhanced_doc = document.copy()
        
        if "metadata" not in enhanced_doc:
            enhanced_doc["metadata"] = {}
        
        text = enhanced_doc["content"].get("full_text", "")
        
        # Extract and enhance date
        extracted_date = self._extract_date(text)
        if extracted_date and extracted_date != "Unknown":
            enhanced_doc["metadata"]["extracted_date"] = extracted_date
        elif "date" in enhanced_doc["metadata"]:
            enhanced_doc["metadata"]["extracted_date"] = enhanced_doc["metadata"]["date"]
        
        # Determine document type
        doc_type = self._determine_document_type(text, enhanced_doc)
        enhanced_doc["metadata"]["extracted_type"] = doc_type
        
        # Extract keywords
        keywords = self._extract_keywords(text)
        enhanced_doc["metadata"]["extracted_keywords"] = keywords
        
        # Estimate document length category
        word_count = len(text.split())
        if word_count < 100:
            length_category = "short"
        elif word_count < 1000:
            length_category = "medium"
        else:
            length_category = "long"
        enhanced_doc["metadata"]["length_category"] = length_category
        
        # Extract potential recipients
        recipients = self._extract_recipients(text)
        if recipients:
            if "recipients" not in enhanced_doc["metadata"]:
                enhanced_doc["metadata"]["recipients"] = recipients
            else:
                enhanced_doc["metadata"]["recipients"].extend(recipients)
        
        # Add source field if not present
        if "source" not in enhanced_doc:
            enhanced_doc["source"] = "papers_of_abraham_lincoln"
        
        return enhanced_doc
    
    def _extract_date(self, text: str) -> Optional[str]:
        """Extract date from text"""
        # First, try to find patterns like "January 15, 1862"
        month_patterns = [
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+(\d{4})',
            r'(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})'
        ]
        
        for pattern in month_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    if isinstance(match, tuple):
                        # Handle different match formats
                        if len(match) == 2:  # Month Day, Year
                            date_str = f"{match[0]} {match[1]}"
                        elif len(match) == 3:  # MM/DD/YYYY
                            date_str = f"{match[0]}/{match[1]}/{match[2]}"
                        else:
                            continue
                    else:
                        date_str = match
                    
                    parsed_date = dateutil.parser.parse(date_str, fuzzy=True)
                    if 1700 <= parsed_date.year <= 1900:
                        return parsed_date.strftime("%Y-%m-%d")
                except:
                    continue
        
        # Try to find just a year
        year_pattern = r'\b(1[7-9]\d{2})\b'
        year_match = re.search(year_pattern, text)
        if year_match:
            year = year_match.group(1)
            return year
        
        return None
    
    def _determine_document_type(self, text: str, document: Dict) -> str:
        """Determine document type based on content"""
        # Check existing type
        existing_type = document.get("metadata", {}).get("type", "").lower()
        
        # Analyze text for type indicators
        text_lower = text.lower()
        
        # Check each document type
        scores = {}
        for doc_type, indicators in self.DOCUMENT_TYPES.items():
            if doc_type == "unknown":
                continue
            score = sum(1 for indicator in indicators if indicator in text_lower)
            scores[doc_type] = score
        
        # Get highest scoring type
        if scores:
            best_type = max(scores, key=scores.get)
            if scores[best_type] > 0:
                return best_type
        
        # Try to determine from title or first few lines
        first_100 = text_lower[:100]
        if "to " in first_100 and ("dear" in first_100 or "sir" in first_100):
            return "letter"
        elif "speech" in first_100 or "address" in first_100:
            return "speech"
        elif any(word in first_100 for word in ["telegram", "telegraph", "wire"]):
            return "telegraph"
        
        # Fallback to existing type or default
        return existing_type if existing_type else "unknown"
    
    def _extract_keywords(self, text: str, max_keywords: int = 15) -> List[str]:
        """Extract keywords from text"""
        # Common stopwords
        stopwords = set(['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                        'of', 'with', 'by', 'is', 'was', 'are', 'were', 'be', 'been',
                        'that', 'this', 'which', 'it', 'its', 'an', 'as', 'from',
                        'have', 'has', 'had', 'will', 'would', 'could', 'should',
                        'may', 'might', 'must', 'can', 'shall'])
        
        # Lincoln-specific important words
        lincoln_themes = {
            'union', 'slavery', 'war', 'peace', 'government', 'people', 'rights',
            'liberty', 'freedom', 'constitution', 'law', 'nation', 'states',
            'president', 'congress', 'army', 'soldiers', 'emancipation', 'american',
            'united', 'country', 'public', 'justice', 'duty', 'honor', 'truth',
            'principle', 'authority', 'power', 'military', 'general', 'officer',
            'battle', 'victory', 'defeat', 'enemy', 'rebel', 'rebellion', 'secession'
        }
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter and count
        word_counts = {}
        for word in words:
            if word not in stopwords:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Prioritize Lincoln-themed words
        keywords = []
        for word in lincoln_themes:
            if word in word_counts:
                keywords.append(word)
        
        # Add other frequent words
        other_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        for word, count in other_words:
            if word not in keywords and word not in stopwords:
                keywords.append(word)
            if len(keywords) >= max_keywords:
                break
        
        return keywords
    
    def _extract_recipients(self, text: str) -> List[str]:
        """Extract potential recipients from text"""
        recipients = []
        
        # Look for salutation patterns
        salutation_patterns = [
            r'To\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)[,:]',
            r'Dear\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)[,:]',
            r'My\s+dear\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)[,:]',
        ]
        
        for pattern in salutation_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    recipient = ' '.join(match)
                else:
                    recipient = match
                
                # Filter out common false positives
                if recipient.lower() not in ['sir', 'madam', 'gentlemen', 'mr', 'mrs', 'dr']:
                    recipients.append(recipient)
        
        return list(set(recipients))
    
    def batch_enhance_documents(self, input_dir: str, output_dir: str) -> Dict:
        """Enhance metadata for all documents in a directory"""
        os.makedirs(output_dir, exist_ok=True)
        
        enhancement_report = {
            "total_processed": 0,
            "successfully_enhanced": 0,
            "failed": 0,
            "document_types": {},
            "date_extraction_stats": {
                "dates_extracted": 0,
                "dates_missing": 0
            },
            "failed_files": []
        }
        
        # Check if input directory exists
        if not os.path.exists(input_dir):
            print(f"ERROR: Input directory '{input_dir}' does not exist!")
            return enhancement_report
        
        # Get list of JSON files
        json_files = [f for f in os.listdir(input_dir) if f.endswith('.json') and f != "ALL_DOCUMENTS.json"]
        
        print(f"Found {len(json_files)} documents to enhance")
        
        for filename in tqdm(json_files, desc="Enhancing metadata"):
            input_path = os.path.join(input_dir, filename)
            
            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    document = json.load(f)
                
                enhancement_report["total_processed"] += 1
                
                # Enhance metadata
                enhanced_doc = self.enhance_metadata(document)
                
                # Update statistics
                doc_type = enhanced_doc["metadata"].get("extracted_type", "unknown")
                enhancement_report["document_types"][doc_type] = enhancement_report["document_types"].get(doc_type, 0) + 1
                
                if enhanced_doc["metadata"].get("extracted_date"):
                    enhancement_report["date_extraction_stats"]["dates_extracted"] += 1
                else:
                    enhancement_report["date_extraction_stats"]["dates_missing"] += 1
                
                # Save enhanced document
                output_path = os.path.join(output_dir, filename)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(enhanced_doc, f, indent=2, ensure_ascii=False)
                
                enhancement_report["successfully_enhanced"] += 1
                
            except json.JSONDecodeError as e:
                enhancement_report["failed"] += 1
                enhancement_report["failed_files"].append(f"{filename}: JSON decode error - {e}")
            except Exception as e:
                enhancement_report["failed"] += 1
                enhancement_report["failed_files"].append(f"{filename}: {e}")
        
        return enhancement_report

# Usage with batch processing
if __name__ == "__main__":
    import sys
    import os
    
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define correct paths - using cleaned data from previous step
    input_dir = os.path.join(project_root, "data_processing","outputs", "cleaned_data")
    output_dir = os.path.join(project_root, "data_processing","outputs", "enhanced_data")
    
    print(f"Project root: {project_root}")
    print(f"Input directory (cleaned data): {input_dir}")
    print(f"Output directory (enhanced data): {output_dir}")
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"\nERROR: Input directory does not exist!")
        print(f"Please make sure you've run the cleaner first.")
        print(f"Cleaned data should be in: {input_dir}")
        print(f"\nTo run the cleaner:")
        print(f"cd F:\\lincoln_llm_project")
        print(f"python data_processing\\cleaner.py")
        sys.exit(1)
    
    # Run batch enhancement
    enhancer = MetadataEnhancer()
    report = enhancer.batch_enhance_documents(input_dir, output_dir)
    
    print("\n" + "="*60)
    print("Metadata Enhancement Report:")
    print("="*60)
    print(f"Total processed: {report['total_processed']}")
    print(f"Successfully enhanced: {report['successfully_enhanced']}")
    print(f"Failed: {report['failed']}")
    
    print(f"\nDocument Type Distribution:")
    for doc_type, count in sorted(report['document_types'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / report['total_processed']) * 100 if report['total_processed'] > 0 else 0
        print(f"  {doc_type}: {count} ({percentage:.1f}%)")
    
    print(f"\nDate Extraction:")
    print(f"  Dates extracted: {report['date_extraction_stats']['dates_extracted']}")
    print(f"  Dates missing: {report['date_extraction_stats']['dates_missing']}")
    
    if report['failed'] > 0:
        print(f"\nFailed files (first 5):")
        for failed in report['failed_files'][:5]:
            print(f"  - {failed}")
        
        if len(report['failed_files']) > 5:
            print(f"  ... and {len(report['failed_files']) - 5} more")
    
    # Save enhancement report
    try:
        report_file = os.path.join(output_dir, "enhancement_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nEnhancement report saved to: {report_file}")
        
        # Also create a sample output file
        if report['successfully_enhanced'] > 0:
            sample_files = [f for f in os.listdir(output_dir) if f.endswith('.json') and f != "enhancement_report.json"]
            if sample_files:
                sample_file = os.path.join(output_dir, sample_files[0])
                with open(sample_file, 'r', encoding='utf-8') as f:
                    sample_doc = json.load(f)
                
                print(f"\nSample enhanced document ({sample_files[0]}):")
                print(json.dumps(sample_doc["metadata"], indent=2))
                
    except Exception as e:
        print(f"\nWarning: Could not save enhancement report: {e}")

    # Also run the original test code
    print("\n" + "="*60)
    print("Original Test Code Output:")
    print("="*60)
    
    # Test with sample document
    sample_doc = {
        "content": {
            "full_text": "To General McClellan, January 15, 1862. My dear General, I have received your letter..."
        }
    }
    
    enhanced = enhancer.enhance_metadata(sample_doc)
    print(json.dumps(enhanced["metadata"], indent=2))
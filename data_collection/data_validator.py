import json
import os
from typing import Dict, List, Set
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    """Validates the collected Lincoln data"""
    
    REQUIRED_FIELDS = {
        "document_id": str,
        "url": str,
        "metadata": dict,
        "content": dict,
        "extraction_date": str
    }
    
    REQUIRED_METADATA_FIELDS = ["date", "authors"]
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.validation_errors = []
        
    def validate_dataset(self) -> Dict:
        """Validate all documents in the dataset"""
        validation_report = {
            "total_documents": 0,
            "valid_documents": 0,
            "invalid_documents": 0,
            "validation_errors": [],
            "field_coverage": {},
            "unique_sources": [],  # Changed from set to list
            "date_range": {"earliest": None, "latest": None}
        }
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.json') and filename != "ALL_DOCUMENTS.json" and filename != "STATISTICS.json":
                filepath = os.path.join(self.data_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        document = json.load(f)
                    
                    validation_report["total_documents"] += 1
                    
                    # Validate structure
                    is_valid, errors = self._validate_document(document)
                    
                    if is_valid:
                        validation_report["valid_documents"] += 1
                        self._update_stats(validation_report, document)
                    else:
                        validation_report["invalid_documents"] += 1
                        validation_report["validation_errors"].append({
                            "file": filename,
                            "errors": errors
                        })
                        
                except json.JSONDecodeError as e:
                    validation_report["invalid_documents"] += 1
                    validation_report["validation_errors"].append({
                        "file": filename,
                        "error": f"Invalid JSON: {str(e)}"
                    })
                except Exception as e:
                    validation_report["invalid_documents"] += 1
                    validation_report["validation_errors"].append({
                        "file": filename,
                        "error": f"Error reading file: {str(e)}"
                    })
        
        # Convert unique_sources to list for JSON serialization
        if "unique_sources" in validation_report and isinstance(validation_report["unique_sources"], set):
            validation_report["unique_sources"] = list(validation_report["unique_sources"])
        
        return validation_report
    
    def _validate_document(self, document: Dict) -> (bool, List[str]):
        """Validate a single document"""
        errors = []
        
        # Check required fields
        for field, field_type in self.REQUIRED_FIELDS.items():
            if field not in document:
                errors.append(f"Missing required field: {field}")
            elif not isinstance(document[field], field_type):
                errors.append(f"Field {field} has wrong type. Expected {field_type}, got {type(document[field])}")
        
        # Check metadata fields
        if "metadata" in document:
            for field in self.REQUIRED_METADATA_FIELDS:
                if field not in document["metadata"]:
                    errors.append(f"Missing metadata field: {field}")
        else:
            errors.append("Missing 'metadata' field")
        
        # Check content
        if "content" in document:
            if "full_text" not in document["content"]:
                errors.append("Missing 'full_text' in content")
            elif len(document["content"]["full_text"].strip()) < 10:
                errors.append("Content text too short (< 10 characters)")
            # Check for structured content
            if "structured_content" not in document["content"]:
                errors.append("Missing 'structured_content' in content")
        else:
            errors.append("Missing 'content' field")
        
        return len(errors) == 0, errors
    
    def _update_stats(self, report: Dict, document: Dict):
        """Update validation statistics"""
        # Track source if present
        if "source" in document:
            if "unique_sources" not in report:
                report["unique_sources"] = set()
            report["unique_sources"].add(document["source"])
        elif "metadata" in document and "source" in document["metadata"]:
            if "unique_sources" not in report:
                report["unique_sources"] = set()
            report["unique_sources"].add(document["metadata"]["source"])
        
        # Track field coverage
        for field in document.keys():
            report["field_coverage"][field] = report["field_coverage"].get(field, 0) + 1
        
        # Also track content subfields
        if "content" in document:
            for field in document["content"].keys():
                report["field_coverage"][f"content.{field}"] = report["field_coverage"].get(f"content.{field}", 0) + 1
        
        # Parse dates for range
        if "metadata" in document and "date" in document["metadata"]:
            date_str = document["metadata"]["date"]
            year = self._extract_year(date_str)
            if year:
                if not report["date_range"]["earliest"] or year < report["date_range"]["earliest"]:
                    report["date_range"]["earliest"] = year
                if not report["date_range"]["latest"] or year > report["date_range"]["latest"]:
                    report["date_range"]["latest"] = year
    
    def _extract_year(self, date_str: str) -> int:
        """Extract year from date string"""
        import re
        
        if not date_str or date_str == "Unknown":
            return None
            
        # Try different date patterns
        patterns = [
            r'\b(1[7-9]\d{2})\b',  # Years 1700-1999
            r'\b(\d{4})\b',         # Any 4-digit year
            r'(\d{4})-\d{2}-\d{2}', # YYYY-MM-DD format
            r'(\d{4})/\d{2}/\d{2}', # YYYY/MM/DD format
        ]
        
        for pattern in patterns:
            match = re.search(pattern, date_str)
            if match:
                try:
                    year = int(match.group(1))
                    # Validate reasonable year range for Lincoln (1809-1865)
                    if 1700 <= year <= 1900:
                        return year
                except ValueError:
                    continue
        return None

    def analyze_common_errors(self, validation_report: Dict) -> Dict:
        """Analyze most common validation errors"""
        error_counts = {}
        
        for error_entry in validation_report.get("validation_errors", []):
            if "errors" in error_entry:
                for error in error_entry["errors"]:
                    error_counts[error] = error_counts.get(error, 0) + 1
        
        return {
            "total_errors": sum(error_counts.values()),
            "error_frequency": dict(sorted(error_counts.items(), key=lambda x: x[1], reverse=True)),
            "most_common_errors": list(error_counts.keys())[:5] if error_counts else []
        }

# Usage
if __name__ == "__main__":
    import sys
    
    # Set data directory
    data_dir = "datasets/raw"
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' does not exist!")
        print("Please run the scraper first to collect data.")
        sys.exit(1)
    
    validator = DataValidator(data_dir)
    report = validator.validate_dataset()
    
    print("=" * 60)
    print("DATA VALIDATION REPORT")
    print("=" * 60)
    print(f"Total documents: {report['total_documents']}")
    print(f"Valid documents: {report['valid_documents']}")
    print(f"Invalid documents: {report['invalid_documents']}")
    
    if report['unique_sources']:
        print(f"Unique sources: {report['unique_sources']}")
    else:
        print("Unique sources: None found")
    
    print(f"\nDate range: {report['date_range']['earliest']} to {report['date_range']['latest']}")
    
    # Analyze field coverage
    print(f"\nField Coverage (top 10):")
    for field, count in sorted(report['field_coverage'].items(), key=lambda x: x[1], reverse=True)[:10]:
        percentage = (count / report['total_documents']) * 100
        print(f"  {field}: {count} ({percentage:.1f}%)")
    
    # Analyze errors
    if report['invalid_documents'] > 0:
        print(f"\nValidation Errors Summary:")
        error_analysis = validator.analyze_common_errors(report)
        print(f"Total errors found: {error_analysis['total_errors']}")
        
        if error_analysis['most_common_errors']:
            print("\nMost common errors:")
            for error in error_analysis['most_common_errors']:
                count = error_analysis['error_frequency'][error]
                print(f"  {error}: {count} times")
    
    # Save validation report
    try:
        # Make sure unique_sources is serializable
        if isinstance(report.get('unique_sources'), set):
            report['unique_sources'] = list(report['unique_sources'])
        
        report_file = "validation_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed validation report saved to: {report_file}")
    except Exception as e:
        print(f"\nWarning: Could not save validation report: {e}")
    
    # Give recommendations based on validation
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    print("=" * 60)
    
    if report['valid_documents'] == 0:
        print("❌ CRITICAL: No valid documents found!")
        print("Check if your scraper is producing the correct JSON format.")
        print("Common issues:")
        print("1. Missing required fields (document_id, url, metadata, content, extraction_date)")
        print("2. Missing metadata fields (date, authors)")
        print("3. Content too short or missing")
    elif report['valid_documents'] < report['total_documents'] / 2:
        print("⚠ WARNING: Less than 50% of documents are valid")
        print("Consider fixing the invalid documents before proceeding.")
    else:
        print("✅ Good: Most documents are valid")
        print("You can proceed with data cleaning and processing.")
    
    # Check specific common issues
    missing_fields = []
    for field in ['metadata', 'content', 'extraction_date']:
        coverage = report['field_coverage'].get(field, 0) / report['total_documents'] * 100
        if coverage < 90:
            missing_fields.append(f"{field} ({coverage:.1f}% coverage)")
    
    if missing_fields:
        print(f"\n⚠ Fields with low coverage: {', '.join(missing_fields)}")
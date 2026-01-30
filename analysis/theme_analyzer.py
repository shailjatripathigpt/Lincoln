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

class ThemeAnalyzer:
    """Analyzes recurring themes in Lincoln's writings"""
    
    def __init__(self):
        self.lincoln_themes = {
            "unity_and_union": [
                "union", "united", "together", "concord", "harmony", "nation", "country", "states"
            ],
            "slavery_and_freedom": [
                "slavery", "emancipation", "freedom", "liberty", "bondage", "slave", "free", "abolition"
            ],
            "democracy_and_government": [
                "government", "people", "democracy", "republic", "constitution", "law", "rights", "authority"
            ],
            "war_and_conflict": [
                "war", "conflict", "battle", "soldier", "army", "peace", "military", "enemy", "rebel"
            ],
            "morality_and_ethics": [
                "right", "wrong", "moral", "ethical", "justice", "fair", "duty", "honor", "truth"
            ],
            "legal_and_constitutional": [
                "legal", "constitution", "rights", "amendment", "court", "law", "justice", "legislature"
            ],
            "presidential_and_executive": [
                "president", "executive", "office", "administration", "cabinet", "secretary", "department"
            ]
        }
    
    def analyze_document_themes(self, text: str) -> Dict:
        """Analyze themes in a single document"""
        if not text:
            return {theme: 0 for theme in self.lincoln_themes.keys()}
        
        text_lower = text.lower()
        
        theme_scores = {}
        for theme, keywords in self.lincoln_themes.items():
            score = sum(text_lower.count(keyword) for keyword in keywords)
            # Normalize by document length (per 1000 words)
            word_count = len(text_lower.split())
            normalized_score = score / max(1, word_count / 1000)
            theme_scores[theme] = float(normalized_score)
        
        # Identify primary theme
        if theme_scores:
            primary_theme = max(theme_scores.items(), key=lambda x: x[1])
            theme_scores["primary_theme"] = primary_theme[0]
            theme_scores["primary_score"] = float(primary_theme[1])
        
        return theme_scores
    
    def analyze_corpus_themes(self, documents_dir: str, output_dir: str, sample_size: int = 100) -> Dict:
        """Analyze themes across the entire corpus"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Get list of JSON files
        json_files = [f for f in os.listdir(documents_dir) if f.endswith('.json') 
                     and f not in ["ALL_DOCUMENTS.json", "enhancement_report.json"]]
        
        if not json_files:
            logger.error(f"No JSON files found in {documents_dir}")
            return {}
        
        print(f"Found {len(json_files)} documents")
        
        # Use sample or all files
        if sample_size and len(json_files) > sample_size:
            import random
            json_files = random.sample(json_files, sample_size)
            print(f"Analyzing random sample of {sample_size} documents")
        else:
            print(f"Analyzing all {len(json_files)} documents")
        
        all_theme_scores = []
        failed_files = []
        document_themes = []
        
        for filename in tqdm(json_files, desc="Analyzing themes"):
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
                    theme_scores = self.analyze_document_themes(text)
                    
                    # Add document metadata
                    theme_scores["document_id"] = doc.get("document_id", filename.replace('.json', ''))
                    theme_scores["document_type"] = doc.get("metadata", {}).get("type", "unknown")
                    theme_scores["date"] = doc.get("metadata", {}).get("date", "unknown")
                    
                    all_theme_scores.append(theme_scores)
                    
                    # Track primary themes
                    if "primary_theme" in theme_scores:
                        document_themes.append({
                            "document_id": theme_scores["document_id"],
                            "primary_theme": theme_scores["primary_theme"],
                            "primary_score": theme_scores["primary_score"],
                            "all_scores": {k: v for k, v in theme_scores.items() 
                                         if k not in ["document_id", "document_type", "date", "primary_theme", "primary_score"]}
                        })
                else:
                    failed_files.append(f"{filename}: Text too short ({len(text) if text else 0} chars)")
                    
            except json.JSONDecodeError as e:
                failed_files.append(f"{filename}: Invalid JSON - {e}")
            except Exception as e:
                failed_files.append(f"{filename}: {type(e).__name__}: {e}")
        
        if not all_theme_scores:
            logger.error("No documents could be analyzed for themes")
            return {}
        
        # Calculate corpus statistics
        corpus_analysis = {
            "total_documents_analyzed": len(all_theme_scores),
            "failed_documents": len(failed_files),
            "theme_distribution": self._calculate_theme_distribution(all_theme_scores),
            "theme_evolution": self._analyze_theme_evolution(all_theme_scores),
            "theme_intensity": self._calculate_theme_intensity(all_theme_scores),
            "document_themes_summary": document_themes[:10],  # First 10 for inspection
            "most_common_themes": self._get_most_common_themes(document_themes)
        }
        
        # Save individual theme analyses
        for i, theme_score in enumerate(all_theme_scores[:10]):
            analysis_file = os.path.join(output_dir, f"theme_analysis_{i+1}.json")
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(theme_score, f, indent=2, ensure_ascii=False)
        
        # Save corpus analysis
        report_file = os.path.join(output_dir, "theme_analysis_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(corpus_analysis, f, indent=2, ensure_ascii=False)
        
        # Save failed files
        if failed_files:
            failed_file = os.path.join(output_dir, "failed_theme_analyses.txt")
            with open(failed_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(failed_files[:20]))
        
        return corpus_analysis
    
    def _calculate_theme_distribution(self, all_theme_scores: List[Dict]) -> Dict:
        """Calculate distribution of themes across documents"""
        theme_counts = Counter()
        theme_total_scores = {theme: 0 for theme in self.lincoln_themes.keys()}
        
        for doc_scores in all_theme_scores:
            for theme in self.lincoln_themes.keys():
                if theme in doc_scores and doc_scores[theme] > 0.5:  # Threshold for presence
                    theme_counts[theme] += 1
                    theme_total_scores[theme] += doc_scores[theme]
        
        total_docs = len(all_theme_scores)
        distribution = {}
        for theme in self.lincoln_themes.keys():
            count = theme_counts[theme]
            percentage = (count / total_docs * 100) if total_docs > 0 else 0
            avg_intensity = (theme_total_scores[theme] / count) if count > 0 else 0
            
            distribution[theme] = {
                "count": count,
                "percentage": float(percentage),
                "average_intensity": float(avg_intensity),
                "total_score": float(theme_total_scores[theme])
            }
        
        return dict(sorted(distribution.items(), key=lambda x: x[1]["count"], reverse=True))
    
    def _analyze_theme_evolution(self, all_theme_scores: List[Dict]) -> Dict:
        """Analyze how themes evolve over time"""
        yearly_themes = {}
        
        for doc_scores in all_theme_scores:
            date = doc_scores.get("date", "")
            
            # Extract year from date
            import re
            year_match = re.search(r'\b(1[7-9]\d{2})\b', str(date))
            if year_match:
                year = year_match.group(1)
                
                if year not in yearly_themes:
                    yearly_themes[year] = {theme: [] for theme in self.lincoln_themes.keys()}
                
                for theme in self.lincoln_themes.keys():
                    if theme in doc_scores:
                        yearly_themes[year][theme].append(doc_scores[theme])
        
        # Calculate average scores per year
        evolution = {}
        for year, theme_lists in yearly_themes.items():
            avg_scores = {}
            for theme, scores in theme_lists.items():
                if scores:
                    avg_scores[theme] = float(np.mean(scores))
            
            if avg_scores:
                evolution[year] = avg_scores
        
        return dict(sorted(evolution.items()))
    
    def _calculate_theme_intensity(self, all_theme_scores: List[Dict]) -> Dict:
        """Calculate intensity of themes across documents"""
        theme_intensities = {}
        
        for theme in self.lincoln_themes.keys():
            scores = [doc_scores[theme] for doc_scores in all_theme_scores if theme in doc_scores]
            if scores:
                theme_intensities[theme] = {
                    "average": float(np.mean(scores)),
                    "std_dev": float(np.std(scores)),
                    "max": float(np.max(scores)),
                    "min": float(np.min(scores))
                }
        
        return theme_intensities
    
    def _get_most_common_themes(self, document_themes: List[Dict]) -> Dict:
        """Get most common primary themes"""
        if not document_themes:
            return {}
        
        primary_themes = [doc["primary_theme"] for doc in document_themes if "primary_theme" in doc]
        theme_counts = Counter(primary_themes)
        
        total = len(primary_themes)
        result = {}
        for theme, count in theme_counts.most_common(10):
            percentage = (count / total * 100) if total > 0 else 0
            result[theme] = {
                "count": count,
                "percentage": float(percentage)
            }
        
        return result
    
    def generate_theme_report(self, corpus_analysis: Dict) -> str:
        """Generate a human-readable theme report"""
        if not corpus_analysis:
            return "No theme analysis data available"
        
        report_lines = [
            "=" * 60,
            "LINCOLN WRITINGS THEME ANALYSIS REPORT",
            "=" * 60,
            f"\nDocuments Analyzed: {corpus_analysis['total_documents_analyzed']}",
            f"Failed Documents: {corpus_analysis.get('failed_documents', 0)}",
            
            "\n" + "=" * 60,
            "THEME DISTRIBUTION",
            "=" * 60,
        ]
        
        if "theme_distribution" in corpus_analysis:
            for theme, stats in corpus_analysis["theme_distribution"].items():
                report_lines.append(f"\n{theme.replace('_', ' ').title()}:")
                report_lines.append(f"  Documents: {stats['count']} ({stats['percentage']:.1f}%)")
                report_lines.append(f"  Average Intensity: {stats['average_intensity']:.2f}")
        
        report_lines.extend([
            "\n" + "=" * 60,
            "MOST COMMON PRIMARY THEMES",
            "=" * 60,
        ])
        
        if "most_common_themes" in corpus_analysis:
            for theme, stats in corpus_analysis["most_common_themes"].items():
                report_lines.append(f"{theme.replace('_', ' ').title()}: {stats['count']} ({stats['percentage']:.1f}%)")
        
        report_lines.extend([
            "\n" + "=" * 60,
            "THEME INTENSITY ANALYSIS",
            "=" * 60,
        ])
        
        if "theme_intensity" in corpus_analysis:
            for theme, stats in corpus_analysis["theme_intensity"].items():
                report_lines.append(f"\n{theme.replace('_', ' ').title()}:")
                report_lines.append(f"  Average: {stats['average']:.3f}")
                report_lines.append(f"  Range: {stats['min']:.3f} - {stats['max']:.3f}")
                report_lines.append(f"  Std Dev: {stats['std_dev']:.3f}")
        
        report_lines.extend([
            "\n" + "=" * 60,
            "THEME EVOLUTION OVER TIME",
            "=" * 60,
        ])
        
        if "theme_evolution" in corpus_analysis and corpus_analysis["theme_evolution"]:
            for year, themes in corpus_analysis["theme_evolution"].items():
                report_lines.append(f"\n{year}:")
                for theme, score in sorted(themes.items(), key=lambda x: x[1], reverse=True)[:3]:
                    if score > 0:
                        report_lines.append(f"  {theme.replace('_', ' ').title()}: {score:.3f}")
        else:
            report_lines.append("\nNo date information available for theme evolution analysis")
        
        report_lines.extend([
            "\n" + "=" * 60,
            "ANALYSIS RECOMMENDATIONS",
            "=" * 60,
            "\nBased on the theme analysis:",
        ])
        
        # Add recommendations
        if "theme_distribution" in corpus_analysis:
            top_themes = list(corpus_analysis["theme_distribution"].keys())[:3]
            report_lines.append(f"• Most prevalent themes: {', '.join(t.replace('_', ' ').title() for t in top_themes)}")
        
        if "most_common_themes" in corpus_analysis:
            primary_theme = list(corpus_analysis["most_common_themes"].keys())[0] if corpus_analysis["most_common_themes"] else "unknown"
            report_lines.append(f"• Most common primary theme: {primary_theme.replace('_', ' ').title()}")
        
        report_lines.append("• Consider these themes for RAG system categorization")
        report_lines.append("• Fine-tuning should emphasize these thematic patterns")
        report_lines.append("• Use theme-based filtering for specialized queries")
        
        return "\n".join(report_lines)

# Main execution
if __name__ == "__main__":
    import sys
    import os
    import time
    
    # Get the correct project structure
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # Go up one level from analysis
    
    # Define the correct paths
    enhanced_data_dir = os.path.join(project_root, "data_processing", "outputs", "enhanced_data")
    output_dir = os.path.join(project_root, "analysis", "outputs", "theme_analysis")
    
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
    json_files = [f for f in files_in_dir if f.endswith('.json') 
                 and f not in ["ALL_DOCUMENTS.json", "enhancement_report.json"]]
    
    print(f"\nFound {len(files_in_dir)} files in enhanced_data directory")
    print(f"JSON files for analysis: {len(json_files)}")
    
    if len(json_files) == 0:
        print("❌ No JSON files found for analysis!")
        print(f"Directory contents: {files_in_dir[:10]}")
        sys.exit(1)
    
    # Initialize analyzer
    print("\nInitializing Theme Analyzer...")
    analyzer = ThemeAnalyzer()
    
    # Analyze corpus
    print(f"\nStarting theme analysis on documents in: {enhanced_data_dir}")
    
    # Use sample for testing
    sample_size = 100
    corpus_analysis = analyzer.analyze_corpus_themes(
        documents_dir=enhanced_data_dir,
        output_dir=output_dir,
        sample_size=sample_size
    )
    
    if corpus_analysis:
        # Generate and print report
        report = analyzer.generate_theme_report(corpus_analysis)
        print(report)
        
        # Save report to file
        report_file = os.path.join(output_dir, "theme_analysis_summary.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nDetailed report saved to: {report_file}")
        
        # Show key findings
        print("\n" + "="*60)
        print("KEY FINDINGS")
        print("="*60)
        
        if "theme_distribution" in corpus_analysis:
            print("\nTop 3 Most Common Themes:")
            for i, (theme, stats) in enumerate(list(corpus_analysis["theme_distribution"].items())[:3]):
                print(f"{i+1}. {theme.replace('_', ' ').title()}: {stats['count']} docs ({stats['percentage']:.1f}%)")
        
        if "most_common_themes" in corpus_analysis:
            print(f"\nMost Common Primary Theme:")
            for theme, stats in list(corpus_analysis["most_common_themes"].items())[:1]:
                print(f"{theme.replace('_', ' ').title()}: {stats['count']} docs ({stats['percentage']:.1f}%)")
        
        # Show where files are saved
        if os.path.exists(output_dir):
            output_files = os.listdir(output_dir)
            print(f"\nGenerated files in {output_dir}:")
            for file in output_files[:10]:  # Show first 10 files
                print(f"  - {file}")
            if len(output_files) > 10:
                print(f"  ... and {len(output_files) - 10} more")
    else:
        print("\n❌ No theme analysis could be performed.")
        print("Check if your enhanced_data files have sufficient text content.")
    
    print("\n" + "="*60)
    print("THEME ANALYSIS COMPLETE")
    print("="*60)
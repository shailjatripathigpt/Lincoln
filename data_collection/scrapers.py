import requests
from bs4 import BeautifulSoup
import os
import time
import json
import re

BASE_URL = "https://papersofabrahamlincoln.org"
DOC_LIST_URL = "https://papersofabrahamlincoln.org/documents/"

# Updated folder to store JSON files - matches the project structure
OUTPUT_JSON_DIR = "datasets/raw"
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

def get_document_links(page_html):
    """Extract document page URLs on the main documents list page."""
    soup = BeautifulSoup(page_html, "html.parser")
    links = []
    for a in soup.select("a"):
        href = a.get("href")
        if href and href.startswith("/documents/D"):  # document pattern
            full_url = BASE_URL + href
            links.append(full_url)
    return list(set(links))

def has_next_page(page_html):
    """Check if there's a next page available."""
    soup = BeautifulSoup(page_html, "html.parser")
    # Look for next page link or pagination
    next_link = soup.find("a", string="Next") or soup.find("a", string="»") or soup.find("a", href=lambda x: x and "page=" in x and "Next" in x.get_text())
    return next_link is not None

def extract_content_from_html(html_content, doc_id, doc_url):
    """Extract clean content from the specific HTML structure."""
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Initialize JSON structure
    json_data = {
        "document_id": doc_id,
        "url": doc_url,
        "metadata": {},
        "content": {},
        "extraction_date": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Extract metadata from sidebar
    try:
        # Extract date
        date_text = ""
        date_elem = soup.find("strong", string="Date:")
        if date_elem:
            date_parent = date_elem.find_parent("p")
            if date_parent:
                date_text = date_parent.get_text(strip=True).replace("Date:", "").strip()
        
        # Extract author
        authors = []
        author_elem = soup.find("strong", string="Author(s):")
        if author_elem:
            author_parent = author_elem.find_parent("p")
            if author_parent:
                author_links = author_parent.find_all("a")
                for link in author_links:
                    authors.append(link.get_text(strip=True))
        
        # Extract keywords
        keywords = []
        keyword_elem = soup.find("strong", string="Keyword(s):")
        if keyword_elem:
            keyword_parent = keyword_elem.find_parent("p")
            if keyword_parent:
                keyword_links = keyword_parent.find_all("a")
                for link in keyword_links:
                    keywords.append(link.get_text(strip=True))
        
        # Extract type and series
        type_text = ""
        type_elem = soup.find("strong", string="Type:")
        if type_elem:
            type_parent = type_elem.find_parent("p")
            if type_parent:
                type_text = type_parent.get_text(strip=True).replace("Type:", "").strip()
        
        json_data["metadata"] = {
            "date": date_text,
            "authors": authors,
            "keywords": keywords,
            "type": type_text
        }
    except Exception as e:
        print(f"Warning: Could not extract metadata for {doc_id}: {e}")
    
    # Extract main content from pal-PAL-Doc div
    try:
        main_content_div = soup.find("div", class_="pal-PAL-Doc")
        if not main_content_div:
            main_content_div = soup.find("div", class_="pal-doctextwrittenByAL")
        
        if main_content_div:
            content_parts = []
            citations = []
            footnotes = []
            
            # Extract all text elements
            # Get the main heading
            heading = main_content_div.find("div", class_="pal-head")
            if heading:
                content_parts.append({"type": "heading", "text": heading.get_text(strip=True)})
                json_data["content"]["title"] = heading.get_text(strip=True)
            
            # Extract all pal-ab (abstract/description) divs
            ab_divs = main_content_div.find_all("div", class_="pal-ab")
            for ab in ab_divs:
                text = ab.get_text(strip=True)
                if text and len(text) > 10:
                    content_parts.append({"type": "description", "text": text})
            
            # Get all pal-p (paragraph) divs
            p_divs = main_content_div.find_all("div", class_="pal-p")
            for p in p_divs:
                text = p.get_text(strip=True)
                if text and len(text) > 5:
                    content_parts.append({"type": "paragraph", "text": text})
            
            # Extract tables
            tables = main_content_div.find_all("table", class_="pal-writtenByAL")
            for i, table in enumerate(tables):
                table_text = []
                rows = table.find_all("tr")
                for row in rows:
                    cells = row.find_all(["td", "th"])
                    row_text = " | ".join([cell.get_text(strip=True) for cell in cells if cell.get_text(strip=True)])
                    if row_text:
                        table_text.append(row_text)
                if table_text:
                    content_parts.append({
                        "type": "table",
                        "table_number": i + 1,
                        "rows": table_text
                    })
            
            # Extract bibliographic references
            bibl_divs = main_content_div.find_all("div", class_="pal-bibl")
            for bibl in bibl_divs:
                text = bibl.get_text(strip=True)
                if text:
                    citations.append(text)
            
            # Extract footnotes from the back section
            back_div = soup.find("div", class_="pal-docback")
            if back_div:
                footnote_divs = back_div.find_all("div", class_="pal-editorial-footnote")
                for fn in footnote_divs:
                    fn_id = fn.find("a").get("id") if fn.find("a") else ""
                    fn_text = fn.get_text(strip=True)
                    if fn_text:
                        footnotes.append({
                            "id": fn_id,
                            "text": fn_text
                        })
            
            # Clean up and structure the content
            full_text = ""
            paragraphs = []
            
            for part in content_parts:
                if part["type"] in ["heading", "description", "paragraph"]:
                    paragraphs.append(part["text"])
                    full_text += part["text"] + " "
                elif part["type"] == "table":
                    table_summary = " ".join(part["rows"])
                    paragraphs.append(f"[Table {part['table_number']}: {table_summary[:100]}...]")
                    full_text += f"[Table {part['table_number']}] "
            
            json_data["content"] = {
                "full_text": full_text.strip(),
                "paragraphs": paragraphs,
                "tables_count": len(tables),
                "citations": citations,
                "footnotes": footnotes,
                "structured_content": content_parts
            }
            
            # Word count
            json_data["word_count"] = len(full_text.split())
            
        else:
            print(f"Warning: Could not find main content div for {doc_id}")
            json_data["content"] = {
                "full_text": "",
                "paragraphs": [],
                "tables_count": 0,
                "citations": [],
                "footnotes": [],
                "structured_content": []
            }
            json_data["word_count"] = 0
            
    except Exception as e:
        print(f"Error extracting content for {doc_id}: {e}")
        json_data["content"] = {
            "full_text": "",
            "paragraphs": [],
            "tables_count": 0,
            "citations": [],
            "footnotes": [],
            "structured_content": []
        }
        json_data["word_count"] = 0
    
    return json_data

def save_as_json(data, doc_id):
    """Save data as JSON file."""
    json_path = os.path.join(OUTPUT_JSON_DIR, f"{doc_id}.json")
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved JSON: {json_path}")
        return True
    except Exception as e:
        print(f"✗ Error saving JSON for {doc_id}: {e}")
        return False

# Create a resume function in case script stops
def get_processed_documents():
    """Get list of already processed documents."""
    processed = set()
    if os.path.exists(OUTPUT_JSON_DIR):
        for f in os.listdir(OUTPUT_JSON_DIR):
            if f.endswith('.json') and f != "ALL_DOCUMENTS.json":
                doc_id = f.replace('.json', '')
                processed.add(doc_id)
    return processed

# Step 1: Crawl ALL pages of document listing
page_num = 1
total_documents = 0
already_processed = get_processed_documents()
print(f"Already processed: {len(already_processed)} documents")

while True:
    url = DOC_LIST_URL + f"?page={page_num}"
    print(f"\n{'='*60}")
    print(f"Fetching page {page_num}: {url}")
    print('='*60)
    
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            print(f"Failed to load page {page_num}, status: {r.status_code}")
            break
        
        # Check if page has content (not empty)
        if "No documents found" in r.text or "0 results" in r.text:
            print(f"No more documents found on page {page_num}")
            break
        
        doc_links = get_document_links(r.text)
        print(f"Found {len(doc_links)} document links on page {page_num}")
        
        if len(doc_links) == 0:
            print(f"No document links found on page {page_num}. Stopping.")
            break
        
        # Step 2: Download each document page and convert to JSON
        documents_on_page = 0
        for doc_url in doc_links:
            doc_id = doc_url.split("/")[-1]  # e.g., D200193b
            
            # Skip if already processed
            if doc_id in already_processed:
                print(f"✓ Already processed: {doc_id}")
                continue
            
            print(f"\nProcessing: {doc_id}")
            
            # Download HTML
            try:
                resp = requests.get(doc_url, timeout=30)
                resp.raise_for_status()
            except requests.RequestException as e:
                print(f"✗ Failed to download {doc_id}: {e}")
                # Try one more time
                try:
                    time.sleep(2)
                    resp = requests.get(doc_url, timeout=30)
                    resp.raise_for_status()
                except:
                    print(f"✗ Failed again to download {doc_id}. Skipping.")
                    continue
            
            # Extract content and save as JSON
            json_data = extract_content_from_html(resp.text, doc_id, doc_url)
            
            # Save JSON file
            if save_as_json(json_data, doc_id):
                documents_on_page += 1
                total_documents += 1
                already_processed.add(doc_id)
                
                # Show some stats
                if json_data["content"]["full_text"]:
                    print(f"  Title: {json_data['content'].get('title', 'No title')}")
                    print(f"  Word count: {json_data['word_count']}")
                    print(f"  Paragraphs: {len(json_data['content']['paragraphs'])}")
                    print(f"  Tables: {json_data['content']['tables_count']}")
                    print(f"  Footnotes: {len(json_data['content']['footnotes'])}")
                else:
                    print("  ⚠ Warning: No text content extracted")
            
            time.sleep(1.5)  # polite crawl delay
        
        # Check if we should continue to next page
        page_num += 1
        
        # Optional: You can add a maximum page limit if needed
        # if page_num > 100:  # Safety limit
        #     print("\nReached maximum page limit (100 pages)")
        #     break
            
        # Add a small delay between pages
        time.sleep(2)
        
    except KeyboardInterrupt:
        print("\n\nScript interrupted by user.")
        break
    except Exception as e:
        print(f"\nError processing page {page_num}: {e}")
        print("Waiting 5 seconds before retrying...")
        time.sleep(5)
        continue

print("\n" + "="*60)
print("CRAWLING COMPLETE!")
print(f"Total documents processed: {total_documents}")
print("="*60)

# Create a combined JSON file with all documents
def create_combined_json():
    """Create a single JSON file containing all documents."""
    all_documents = []
    
    json_files = [f for f in os.listdir(OUTPUT_JSON_DIR) if f.endswith('.json') and f != "ALL_DOCUMENTS.json"]
    
    print(f"\nCreating combined JSON file from {len(json_files)} documents...")
    
    for json_file in sorted(json_files):
        json_path = os.path.join(OUTPUT_JSON_DIR, json_file)
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                doc_data = json.load(f)
                all_documents.append(doc_data)
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
    
    # Save combined file
    combined_path = os.path.join(OUTPUT_JSON_DIR, "ALL_DOCUMENTS.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_documents, f, indent=2, ensure_ascii=False)
    
    print(f"\nCombined JSON file created: {combined_path}")
    print(f"Total documents in combined file: {len(all_documents)}")
    
    # Summary statistics
    total_words = sum(doc.get('word_count', 0) for doc in all_documents)
    avg_words = total_words / len(all_documents) if all_documents else 0
    
    print(f"Total words: {total_words:,}")
    print(f"Average words per document: {avg_words:,.0f}")
    
    # Count by document type
    type_count = {}
    for doc in all_documents:
        doc_type = doc.get('metadata', {}).get('type', 'Unknown')
        type_count[doc_type] = type_count.get(doc_type, 0) + 1
    
    print("\nDocument type breakdown:")
    for doc_type, count in sorted(type_count.items()):
        print(f"  {doc_type}: {count}")

# Create summary statistics
def create_summary_statistics():
    """Create a summary statistics file."""
    json_files = [f for f in os.listdir(OUTPUT_JSON_DIR) if f.endswith('.json') and f != "ALL_DOCUMENTS.json" and f != "STATISTICS.json"]
    
    stats = {
        "total_documents": len(json_files),
        "extraction_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "documents_by_type": {},
        "documents_by_year": {},
        "total_word_count": 0,
        "document_ids": []
    }
    
    for json_file in sorted(json_files):
        json_path = os.path.join(OUTPUT_JSON_DIR, json_file)
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                doc_data = json.load(f)
                
            doc_id = doc_data.get("document_id")
            if doc_id:
                stats["document_ids"].append(doc_id)
            
            # Count by type
            doc_type = doc_data.get('metadata', {}).get('type', 'Unknown')
            stats["documents_by_type"][doc_type] = stats["documents_by_type"].get(doc_type, 0) + 1
            
            # Count by year (extract year from date)
            date_str = doc_data.get('metadata', {}).get('date', '')
            if date_str:
                year_match = re.search(r'(\d{4})', date_str)
                if year_match:
                    year = year_match.group(1)
                    stats["documents_by_year"][year] = stats["documents_by_year"].get(year, 0) + 1
            
            # Total word count
            stats["total_word_count"] += doc_data.get('word_count', 0)
                
        except Exception as e:
            print(f"Error reading {json_file} for statistics: {e}")
    
    # Save statistics
    stats_path = os.path.join(OUTPUT_JSON_DIR, "STATISTICS.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\nStatistics file created: {stats_path}")
    
    return stats

# Create combined JSON and statistics
create_combined_json()
stats = create_summary_statistics()

print("\n" + "="*60)
print("PROCESSING COMPLETE!")
print("="*60)
print(f"\nSummary:")
print(f"Total documents: {stats['total_documents']}")
print(f"Total words: {stats['total_word_count']:,}")
print(f"Documents by type:")
for doc_type, count in sorted(stats['documents_by_type'].items()):
    print(f"  {doc_type}: {count}")
print("\nAll JSON files are ready in the 'datasets/raw' folder!")
print("="*60)
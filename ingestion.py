import os
import fitz  # PyMuPDF
from docx import Document
import re

class DocumentIngestor:
    """
    Stage 1: Data Ingestion
    Responsible for loading documents and extracting/cleaning raw text.
    """
    def __init__(self, data_dir="."):
        self.data_dir = data_dir

    def load_pdf(self, file_path):
        pages = []
        try:
            with fitz.open(file_path) as doc:
                for i, page in enumerate(doc):
                    text = page.get_text()
                    if text.strip():
                        pages.append({
                            "content": text,
                            "page_num": i + 1
                        })
        except Exception as e:
            print(f"Error loading PDF {file_path}: {e}")
        return pages

    def load_docx(self, file_path):
        try:
            doc = Document(file_path)
            # Word docs don't have strictly defined pages in the same way, 
            # so we'll treat it as one page for simplicity or split by paragraphs
            text = "\n".join([para.text for para in doc.paragraphs])
            return [{"content": text, "page_num": 1}]
        except Exception as e:
            print(f"Error loading DOCX {file_path}: {e}")
            return []

    def load_txt(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return [{"content": f.read(), "page_num": 1}]
        except Exception as e:
            print(f"Error loading TXT {file_path}: {e}")
            return []

    def clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text) 
        return text.strip()

    def ingest(self):
        print(f"--- Stage 1: Data Ingestion starting in {os.path.abspath(self.data_dir)} ---")
        all_pages = []
        valid_extensions = (".pdf", ".docx", ".txt")
        files = [f for f in os.listdir(self.data_dir) if f.lower().endswith(valid_extensions)]
        
        for filename in files:
            file_path = os.path.join(self.data_dir, filename)
            print(f"Processing {filename}...")
            
            raw_pages = []
            if filename.lower().endswith(".pdf"):
                raw_pages = self.load_pdf(file_path)
            elif filename.lower().endswith(".docx"):
                raw_pages = self.load_docx(file_path)
            elif filename.lower().endswith(".txt"):
                raw_pages = self.load_txt(file_path)
            
            for page in raw_pages:
                cleaned_content = self.clean_text(page["content"])
                if cleaned_content:
                    all_pages.append({
                        "filename": filename,
                        "content": cleaned_content,
                        "page_num": page["page_num"]
                    })
            
            print(f"  Successfully ingested {filename} ({len(raw_pages)} pages)")
        
        print(f"--- Stage 1 Complete: {len(all_pages)} total pages ingested ---\n")
        return all_pages

if __name__ == "__main__":
    ingestor = DocumentIngestor(".")
    docs = ingestor.ingest()
    for doc in docs:
        print(f"File: {doc['filename']}")
        print(f"Preview: {doc['content'][:150]}...")

from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentChunker:
    """
    Stage 2: Text Processing
    Splits text into chunks (chunk_size=500, overlap=100) and maintains metadata.
    """
    def __init__(self, chunk_size=500, chunk_overlap=100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk_documents(self, pages):
        print(f"--- Stage 2: Text Processing (Chunking) starting ---")
        all_chunks = []
        
        for page_data in pages:
            filename = page_data["filename"]
            page_num = page_data["page_num"]
            content = page_data["content"]
            
            chunks = self.splitter.split_text(content)
            
            for i, chunk_text in enumerate(chunks):
                all_chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        "source": filename,
                        "page": page_num,
                        "chunk_index": i
                    }
                })
        
        print(f"--- Stage 2 Complete: {len(all_chunks)} chunks created ---\n")
        return all_chunks

if __name__ == "__main__":
    from ingestion import DocumentIngestor
    
    # Run Stage 1 to get pages
    ingestor = DocumentIngestor(".")
    pages = ingestor.ingest()
    
    # Run Stage 2
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)
    chunks = chunker.chunk_documents(pages)
    
    if chunks:
        # Show first 3 chunks as example
        for i in range(min(3, len(chunks))):
            print(f"Chunk {i} Metadata: {chunks[i]['metadata']}")
            print(f"Chunk {i} Content: {chunks[i]['content'][:100]}...\n")

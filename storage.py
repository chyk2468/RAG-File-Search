import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer

class VectorStore:
    """
    Stage 3: Embedding + Vector Storage
    Generates embeddings and stores them in a FAISS index.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2', index_path="faiss_index.bin", metadata_path="metadata.pkl"):
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata = []

    def create_and_save_index(self, chunks):
        print("--- Stage 3: Embedding + Vector Storage starting ---")
        
        # 1. Extract text and metadata
        texts = [c["content"] for c in chunks]
        self.metadata = [c["metadata"] for c in chunks]
        
        # 2. Generate Embeddings
        print(f"  Generating embeddings for {len(texts)} chunks (this may take a moment)...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # 3. Create FAISS Index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        # 4. Save locally
        print(f"  Saving index to {self.index_path} and metadata to {self.metadata_path}...")
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump({"texts": texts, "metadata": self.metadata}, f)
            
        print(f"--- Stage 3 Complete: Index saved with {self.index.ntotal} vectors ---\n")

    def load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.texts = data["texts"]
                self.metadata = data["metadata"]
            return True
        return False

if __name__ == "__main__":
    from ingestion import DocumentIngestor
    from chunking import DocumentChunker
    
    # Run Stage 1 & 2
    ingestor = DocumentIngestor(".")
    pages = ingestor.ingest()
    
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)
    chunks = chunker.chunk_documents(pages)
    
    # Run Stage 3
    if chunks:
        vector_store = VectorStore()
        vector_store.create_and_save_index(chunks)

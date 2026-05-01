import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

class RetrievalSystem:
    """
    Stage 4: Retrieval + Re-ranking
    Retrieves candidates using FAISS and re-ranks them using a Cross-Encoder.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2', rerank_model='cross-encoder/ms-marco-MiniLM-L-6-v2', index_path="faiss_index.bin", metadata_path="metadata.pkl"):
        print(f"Initializing Retrieval System...")
        self.model = SentenceTransformer(model_name)
        # Load Cross-Encoder for re-ranking
        print(f"Loading Re-ranker: {rerank_model}...")
        self.reranker = CrossEncoder(rerank_model)
        
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.texts = []
        self.metadata = []
        
        self._load_index()

    def _load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.texts = data["texts"]
                self.metadata = data["metadata"]
            print(f"  Index loaded: {self.index.ntotal} vectors.")
            return True
        return False

    def retrieve(self, query, k=3, fetch_k=10):
        if not self.index:
            return []

        # 1. First Stage: Efficient Similarity Search
        query_vector = self.model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_vector, fetch_k)
        
        candidates = []
        for idx in indices[0]:
            if idx != -1:
                candidates.append(self.texts[idx])
        
        if not candidates:
            return []

        # 2. Second Stage: Re-ranking with Cross-Encoder
        print(f"  Re-ranking {len(candidates)} candidates...")
        pairs = [[query, cand] for cand in candidates]
        scores = self.reranker.predict(pairs)
        
        # Sort by score
        ranked_results = sorted(
            zip(scores, [self.texts[i] for i in indices[0] if i != -1], [self.metadata[i] for i in indices[0] if i != -1]),
            key=lambda x: x[0], reverse=True
        )

        return [{"content": res[1], "metadata": res[2], "rerank_score": float(res[0])} for res in ranked_results[:k]]

if __name__ == "__main__":
    try:
        retriever = RetrievalSystem()
        query = "What are the challenges in DISC 14?"
        print(f"\nQuery: {query}")
        
        results = retriever.retrieve(query, k=3, use_mmr=True)
        
        print(f"\n--- Stage 4 Complete: Retrieved {len(results)} chunks ---\n")
        for i, res in enumerate(results):
            print(f"Result {i+1} (Source: {res['metadata']['source']}, Page: {res['metadata']['page']}):")
            print(f"Content: {res['content'][:200]}...\n")
            
    except Exception as e:
        print(f"Error: {e}")

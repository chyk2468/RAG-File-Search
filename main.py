import sys
from ingestion import DocumentIngestor
from chunking import DocumentChunker
from storage import VectorStore
from retrieval import RetrievalSystem
from generation import Generator

def main():
    print("================================================")
    print("      LOCAL RAG SYSTEM DEMONSTRATION")
    print("================================================\n")

    # Step 1: Ingestion
    ingestor = DocumentIngestor(".")
    pages = ingestor.ingest()
    
    if not pages:
        print("No documents found to process. Exiting.")
        return

    # Step 2: Chunking
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)
    chunks = chunker.chunk_documents(pages)

    # Step 3: Embedding + Storage
    vector_store = VectorStore()
    vector_store.create_and_save_index(chunks)

    # Initialize Retrieval and Generation
    retriever = RetrievalSystem()
    generator = Generator()

    print("\nSystem ready! Enter your questions (type 'exit' to quit).")
    
    while True:
        try:
            query = input("\n[User]: ").strip()
            if query.lower() in ['exit', 'quit']:
                break
            if not query:
                continue

            # Step 4: Retrieval
            context = retriever.retrieve(query, k=3, use_mmr=True)
            
            # Step 5: Generation
            answer = generator.generate_answer(query, context)
            
            print(f"\n[AI Assistant]:\n{answer}")
            print("\n--- Sources ---")
            for i, c in enumerate(context):
                print(f"[{i+1}] {c['metadata']['source']} (Page {c['metadata']['page']})")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

import streamlit as st
import os
from ingestion import DocumentIngestor
from chunking import DocumentChunker
from storage import VectorStore
from retrieval import RetrievalSystem
from generation import Generator

# App configuration
st.set_page_config(page_title="Local RAG System", layout="wide")

st.title("🛡️ Local Offline RAG System")
st.markdown("---")

# Persistent State Management
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'generator' not in st.session_state:
    st.session_state.generator = None

# Sidebar for System Actions
with st.sidebar:
    st.header("⚙️ System Controls")
    
    if st.button("🚀 Ingest & Build Index"):
        with st.status("Building Pipeline...", expanded=True) as status:
            st.write("1. Ingesting Documents...")
            ingestor = DocumentIngestor(".")
            pages = ingestor.ingest()
            
            st.write("2. Chunking Text...")
            chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)
            chunks = chunker.chunk_documents(pages)
            
            st.write("3. Embedding & Storing Vectors...")
            vector_store = VectorStore()
            vector_store.create_and_save_index(chunks)
            
            status.update(label="Index Built Successfully!", state="complete")
            st.success("System ready!")

    if st.button("🔄 Load Existing Index"):
        if os.path.exists("faiss_index.bin"):
            with st.spinner("Loading models..."):
                st.session_state.retriever = RetrievalSystem()
                st.session_state.generator = Generator()
            st.success("Index and Models Loaded!")
        else:
            st.error("No index found. Please Build Index first.")

# Main Query Interface
query = st.text_input("Ask a question about your documents:", placeholder="e.g., What are the problem statements in DISC 14?")

if query:
    if st.session_state.retriever and st.session_state.generator:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            with st.spinner("Retrieving and Generating..."):
                # Retrieval + Re-ranking
                context = st.session_state.retriever.retrieve(query, k=3)
                # Generation
                answer = st.session_state.generator.generate_answer(query, context)
            
            st.subheader("💡 AI Answer")
            st.markdown(f"**{answer}**")
            
        with col2:
            st.subheader("📚 Sources (Re-ranked)")
            for i, c in enumerate(context):
                with st.expander(f"Source {i+1}: {c['metadata']['source']} (Page {c['metadata']['page']})"):
                    st.write(c['content'])
                    st.caption(f"Re-rank Score: {c.get('rerank_score', 'N/A')}")
    else:
        st.warning("Please 'Load Existing Index' or 'Build Index' from the sidebar first.")

st.markdown("---")
st.caption("Powered by Qwen2.5-Coder-7B-Instruct (GGUF) | FAISS | Sentence-Transformers | Streamlit")

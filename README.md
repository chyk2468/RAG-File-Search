# Local Offline RAG System (RAG-File-Search)

A powerful, privacy-focused Retrieval-Augmented Generation (RAG) system that runs entirely on your local machine. This tool allows you to chat with your local documents (PDFs, Text) without any data leaving your computer.

## 🚀 Features
- **100% Offline:** No API keys required. Uses local LLMs (GGUF format) and local embedding models.
- **Support for Multiple Formats:** Seamlessly ingest and search through PDF and TXT files.
- **Fast Search:** Powered by **FAISS** (Facebook AI Similarity Search) for high-performance vector retrieval.
- **Modern UI:** Includes a clean, interactive **Streamlit** web interface.
- **CLI Mode:** For those who prefer the terminal.
- **Smart Chunking:** Efficiently splits documents to maintain context and improve answer accuracy.

## 🏗️ Architecture
The system follows a standard RAG pipeline:
1.  **Ingestion:** Reads documents and splits them into manageable chunks.
2.  **Storage:** Converts chunks into vector embeddings and stores them in a local FAISS index.
3.  **Retrieval:** When a query is asked, the system finds the most relevant document chunks.
4.  **Generation:** Passes the query and retrieved context to a local LLM (e.g., Qwen2.5-Coder) to generate a precise answer.

## 🛠️ Tech Stack
- **LLM:** Qwen2.5-Coder-7B (or any GGUF-compatible model)
- **Vector DB:** FAISS
- **Frontend:** Streamlit
- **Orchestration:** LangChain / Custom Python Pipeline
- **Embeddings:** HuggingFace Local Embeddings

## 📥 Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/chyk2468/RAG-File-Search.git
   cd RAG-File-Search
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download a Model:**
   Place your preferred `.gguf` model in the root directory. (e.g., `Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf`).

## 🖥️ Usage

### Quick Start (Windows)
Run the included batch file for an easy menu-driven experience:
```bash
run_rag.bat
```

### Manual Launch
**To run the Web UI:**
```bash
streamlit run app.py
```

**To run the CLI version:**
```bash
python main.py
```

## 📁 Project Structure
- `app.py`: Streamlit web interface.
- `main.py`: Entry point for CLI mode.
- `ingestion.py`: Document processing and chunking logic.
- `retrieval.py`: FAISS search and context fetching.
- `generation.py`: LLM interaction and response generation.
- `storage.py`: Vector database management.
- `.gitignore`: Configured to keep your large model files and indices private.

## 🛡️ Privacy
Since this system is offline, your documents are never uploaded to any cloud service. Your data stays on your hardware.

---
*Created by [chyk2468](https://github.com/chyk2468)*

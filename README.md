# Local Offline RAG System

A modular, CPU-optimized, and fully offline RAG system with a Streamlit UI.

## 📁 Project Structure
- `ingestion.py`: Stage 1 - Text extraction from PDF/TXT/DOCX.
- `chunking.py`: Stage 2 - Semantic text splitting.
- `storage.py`: Stage 3 - Embedding generation and FAISS storage.
- `retrieval.py`: Stage 4 - Bi-Encoder retrieval + Cross-Encoder re-ranking.
- `generation.py`: Stage 5 - Local LLM response generation (GGUF).
- `app.py`: Streamlit UI for interaction.
- `main.py`: CLI version of the system.

## 🛠️ Setup Instructions

### 1. Requirements
Ensure you have Python 3.10+ and the following models present:
- **LLM:** `Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf` (must be in the root directory).

### 2. Installation
Install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Running the System

#### Option A: Streamlit UI (Recommended)
```bash
streamlit run app.py
```
1. Click **"Ingest & Build Index"** in the sidebar (first time only).
2. Click **"Load Existing Index"** to load models.
3. Start chatting!

#### Option B: CLI Mode
```bash
python main.py
```

## 🔒 Security & Privacy
- **Fully Offline:** No data leaves your machine.
- **CPU Optimized:** Designed to run efficiently without a GPU.
- **No Hallucinations:** The prompt constraints force the LLM to answer ONLY from the provided context.

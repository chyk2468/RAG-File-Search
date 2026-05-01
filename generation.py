from llama_cpp import Llama
import os

class Generator:
    """
    Stage 5: Generation
    Loads a local GGUF LLM and generates answers based on retrieved context.
    """
    def __init__(self, model_path="Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf", n_ctx=2048):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LLM model file not found at {model_path}")
        
        print(f"Loading local LLM: {model_path}...")
        # n_ctx is the context window size
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, verbose=False)

    def generate_answer(self, query, context_chunks):
        print("--- Stage 5: Generation starting ---")
        
        # 1. Format context
        context_text = "\n\n".join([
            f"[Chunk {i+1}]: {c['content']}"
            for i, c in enumerate(context_chunks)
        ])
        
        # 2. Construct Prompt using the user's strict template
        prompt = f"""<|im_start|>system
You are an intelligent AI system designed to analyze documents and extract precise answers.
Your job is to deeply understand the context and provide a structured, human-readable response.

STRICT INSTRUCTIONS:
- Answer ONLY from the provided context.
- Do NOT hallucinate or assume missing details.
- If information is insufficient, say: "Not found in document".
- Convert complex text into simple, clear explanation.
- Focus on relevance to the question.

OUTPUT FORMAT:

Answer:
Provide a clear, well-structured explanation (3–6 sentences).

Key Insights:
• Most important takeaway
• Supporting detail
• Any critical implication (if present)

Source Evidence:
Quote 1–2 exact lines from the context that support the answer.

Confidence:
High / Medium / Low (based on how clearly the context answers the question)
<|im_end|>
<|im_start|>user
Question:
{query}

Context:
{context_text}

Final Response:<|im_end|>
<|im_start|>assistant
"""
        
        # 3. Generate
        print("  Generating response...")
        response = self.llm(
            prompt,
            max_tokens=1024, # Increased to accommodate structured format
            stop=["<|im_end|>"],
            echo=False,
            temperature=0.1 # Low temperature for strict adherence
        )
        
        answer = response['choices'][0]['text'].strip()
        print("--- Stage 5 Complete ---\n")
        return answer

if __name__ == "__main__":
    from retrieval import RetrievalSystem
    
    try:
        # Test the pipeline
        query = "What is the goal of Problem Statement 1 in the Indian Army section?"
        
        # Retrieval
        retriever = RetrievalSystem()
        context = retriever.retrieve(query, k=3)
        
        # Generation
        generator = Generator()
        answer = generator.generate_answer(query, context)
        
        print(f"QUERY: {query}")
        print(f"\nANSWER:\n{answer}")
        
    except Exception as e:
        print(f"Error: {e}")

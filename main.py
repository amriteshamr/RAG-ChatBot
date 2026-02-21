import sys
from pdf_loader import load_pdf_text
from chunking import chunk_text
from retrieval import embed_chunks, retrieve
from generation import generate

def build_prompt(query: str, retrieved_chunks: list[str]) -> str:
    context = "\n\n".join(retrieved_chunks)
    return f"""Answer using ONLY the context below.
If the answer is not in the context, say: I don't know based on the provided context.

Context:
{context}

Question: {query}

Answer:
"""

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    print("Loading PDF...")
    text = load_pdf_text(pdf_path)
    if not text.strip():
        print("No text found in the PDF (might be scanned). Try a text-based PDF.")
        sys.exit(1)

    print("Chunking...")
    chunks = chunk_text(text, chunk_size=1000, overlap=150)
    print(f"Chunks: {len(chunks)}")

    print("Embedding chunks (first run takes time)...")
    chunk_embeddings = embed_chunks(chunks)

    while True:
        query = input("\nAsk a question (or type 'exit'): ").strip()
        if query.lower() == "exit":
            break

        retrieved = retrieve(query, chunks, chunk_embeddings, top_k=3)
        prompt = build_prompt(query, retrieved)
        answer = generate(prompt)

        print("\nAnswer:\n", answer)

if __name__ == "__main__":
    main()
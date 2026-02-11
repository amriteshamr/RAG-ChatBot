from retrieval import retrieve
from generation import generate

def build_prompt(query, retrieved):
    context = "\n".join([f"- {doc}" for doc, _ in retrieved])
    return f"""Answer using ONLY the context below.
If the answer is not present, say I don't know.

Context:
{context}

Question: {query}

Answer:"""

while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() == "exit":
        break

    retrieved = retrieve(query)
    prompt = build_prompt(query, retrieved)
    answer = generate(prompt)

    print("\nAnswer:", answer)

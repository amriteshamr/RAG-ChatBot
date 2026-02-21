import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(chunks: list[str]) -> np.ndarray:
    return embedder.encode(chunks, convert_to_numpy=True)

def retrieve(query: str, chunks: list[str], chunk_embeddings: np.ndarray, top_k: int = 3) -> list[str]:
    q_emb = embedder.encode([query], convert_to_numpy=True)
    scores = cosine_similarity(q_emb, chunk_embeddings)[0]
    top_idx = scores.argsort()[::-1][:top_k]
    return [chunks[i] for i in top_idx]
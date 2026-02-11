from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from data import documents

embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(documents)

def retrieve(query, top_k=3):
    query_embedding = embedder.encode(query)
    scores = cosine_similarity([query_embedding], doc_embeddings)[0]
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [(documents[i], float(scores[i])) for i in top_indices]

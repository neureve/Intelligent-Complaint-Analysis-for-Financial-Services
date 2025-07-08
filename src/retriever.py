import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load index + metadata
def load_vector_store(index_path, metadata_path):
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

# Embed user query and retrieve top-k chunks
def retrieve_top_k(query, k=5):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vector = model.encode([query])
    
    index, metadata = load_vector_store("vector_store/faiss_index.index", "vector_store/faiss_metadata.pkl")
    D, I = index.search(np.array(query_vector), k)

    retrieved_chunks = [(metadata[i], D[0][j]) for j, i in enumerate(I[0])]
    return retrieved_chunks

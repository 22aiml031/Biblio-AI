"""
FAISS Vector Database Module for BiblioAI

- Store MiniLM embeddings in a FAISS index
- Store metadata (year, citations, authors, source title) alongside vectors
- Enable similarity search and nearest-neighbor retrieval

Dependencies: faiss-cpu, numpy, pandas
"""
import faiss
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

class FaissVectorDB:
    def __init__(self, embedding_dim: int):
        """Initialize FAISS index (L2 or cosine similarity)."""
        self.index = faiss.IndexFlatIP(embedding_dim)  # Use inner product for cosine if vectors are normalized
        self.metadata: List[Dict[str, Any]] = []
        self.embedding_dim = embedding_dim

    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        """Add embeddings and associated metadata to the index."""
        assert embeddings.shape[1] == self.embedding_dim
        self.index.add(embeddings.astype(np.float32))
        self.metadata.extend(metadata)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for top_k most similar vectors and return their metadata."""
        D, I = self.index.search(query_embedding.astype(np.float32), top_k)
        results = []
        for idx in I[0]:
            if idx < len(self.metadata):
                results.append(self.metadata[idx])
        return results

    def save(self, index_path: str, metadata_path: str):
        """Save FAISS index and metadata to disk."""
        faiss.write_index(self.index, index_path)
        pd.DataFrame(self.metadata).to_json(metadata_path, orient='records')

    def load(self, index_path: str, metadata_path: str):
        """Load FAISS index and metadata from disk."""
        self.index = faiss.read_index(index_path)
        self.metadata = pd.read_json(metadata_path, orient='records').to_dict(orient='records')

# Example usage
if __name__ == "__main__":
    # Assume you have embeddings (N, D) and metadata (list of dicts)
    import numpy as np
    embeddings = np.random.rand(10, 384).astype(np.float32)  # Example shape
    metadata = [
        {'title': f'Paper {i}', 'year': 2020 + i, 'citations': i*5, 'authors': f'Author {i}', 'source_title': 'Journal X'}
        for i in range(10)
    ]
    db = FaissVectorDB(embedding_dim=384)
    db.add_embeddings(embeddings, metadata)
    # Search example
    query = embeddings[0:1]
    results = db.search(query, top_k=3)
    print(results)

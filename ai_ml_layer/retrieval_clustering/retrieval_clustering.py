"""
Retrieval & Clustering Module for BiblioAI

- Retrieve semantically similar papers using FAISS
- Cluster MiniLM embeddings (KMeans or HDBSCAN)
- Assign cluster labels to each paper
- Output cluster metadata (top keywords, size)

Dependencies: faiss-cpu, numpy, pandas, scikit-learn, hdbscan (optional)
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
from typing import List, Dict, Any, Optional

class RetrievalClustering:
    def __init__(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        self.embeddings = embeddings
        self.metadata = metadata
        self.cluster_labels = None
        self.clusters = None

    def cluster_embeddings(self, method: str = 'kmeans', n_clusters: int = 10) -> np.ndarray:
        """Cluster embeddings using KMeans or HDBSCAN."""
        if method == 'kmeans':
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            self.cluster_labels = kmeans.fit_predict(self.embeddings)
        elif method == 'hdbscan' and HDBSCAN_AVAILABLE:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
            self.cluster_labels = clusterer.fit_predict(self.embeddings)
        else:
            raise ValueError("Unsupported clustering method or missing hdbscan.")
        return self.cluster_labels

    def get_cluster_metadata(self) -> Dict[int, Dict[str, Any]]:
        """Aggregate metadata for each cluster (size, top keywords)."""
        clusters = {}
        df = pd.DataFrame(self.metadata)
        df['cluster'] = self.cluster_labels
        for cluster_id in np.unique(self.cluster_labels):
            cluster_df = df[df['cluster'] == cluster_id]
            size = len(cluster_df)
            keywords = []
            if 'author_keywords' in cluster_df.columns:
                keywords += cluster_df['author_keywords'].str.cat(sep='; ').split('; ')
            if 'index_keywords' in cluster_df.columns:
                keywords += cluster_df['index_keywords'].str.cat(sep='; ').split('; ')
            top_keywords = pd.Series(keywords).value_counts().head(10).index.tolist()
            clusters[cluster_id] = {
                'size': size,
                'top_keywords': top_keywords,
                'papers': cluster_df.to_dict(orient='records')
            }
        self.clusters = clusters
        return clusters

    def retrieve_similar(self, query_embedding: np.ndarray, faiss_index, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top_k similar papers using FAISS index."""
        D, I = faiss_index.search(query_embedding.astype(np.float32), top_k)
        return [self.metadata[idx] for idx in I[0] if idx < len(self.metadata)]

# Example usage
if __name__ == "__main__":
    # Assume you have embeddings and metadata
    embeddings = np.random.rand(100, 384).astype(np.float32)
    metadata = [
        {'title': f'Paper {i}', 'author_keywords': 'AI; ML', 'index_keywords': 'Data; Science', 'year': 2020 + i % 5}
        for i in range(100)
    ]
    rc = RetrievalClustering(embeddings, metadata)
    labels = rc.cluster_embeddings(method='kmeans', n_clusters=5)
    clusters = rc.get_cluster_metadata()
    print(clusters)

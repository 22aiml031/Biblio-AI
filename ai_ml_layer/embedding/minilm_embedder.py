"""
Embedding Generation Module for BiblioAI

- Generate MiniLM embeddings for each research paper's processed text
- Batch processing for efficiency
- Output: numpy array of embeddings, aligned with DataFrame rows

Dependencies: sentence-transformers, numpy
"""
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class MiniLMEmbedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', batch_size: int = 32):
        """Initialize MiniLM embedder."""
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts (batch processed)."""
        embeddings = self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings

# Example usage
if __name__ == "__main__":
    import pandas as pd
    # Assume you have a DataFrame with a 'processed_text' column
    df = pd.read_csv("../../cleaned_data copy.csv")
    from ai_ml_layer.csv_loader.csv_nlp_preprocessing import CSVNLPPreprocessor
    preprocessor = CSVNLPPreprocessor()
    df = preprocessor.process_dataframe(df)
    texts = df['processed_text'].tolist()
    embedder = MiniLMEmbedder()
    embeddings = embedder.encode_texts(texts)
    print(embeddings.shape)

"""
Advanced NLP Preprocessing Module for BiblioAI

- Keyword extraction (TF-IDF)
- Named entity recognition (spaCy)
- Phrase detection (optional)

Dependencies: pandas, scikit-learn, spacy
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from typing import List

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

class AdvancedNLPPreprocessor:
    def __init__(self):
        pass

    def extract_keywords_tfidf(self, texts: List[str], top_n: int = 10) -> List[List[str]]:
        """Extract top_n keywords per document using TF-IDF."""
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        keywords = []
        for row in X:
            top_indices = row.toarray()[0].argsort()[-top_n:][::-1]
            keywords.append([feature_names[i] for i in top_indices])
        return keywords

    def extract_entities(self, texts: List[str]) -> List[List[str]]:
        """Extract named entities from each document using spaCy."""
        entities = []
        for text in texts:
            doc = nlp(text)
            entities.append([ent.text for ent in doc.ents])
        return entities

# Example usage
if __name__ == "__main__":
    df = pd.read_csv("../../cleaned_data copy.csv")
    texts = df['Title'].astype(str).tolist()
    preproc = AdvancedNLPPreprocessor()
    tfidf_keywords = preproc.extract_keywords_tfidf(texts)
    entities = preproc.extract_entities(texts)
    print("TF-IDF Keywords:", tfidf_keywords[:2])
    print("Entities:", entities[:2])

"""
CSV Loading and NLP Preprocessing Module for BiblioAI

- Load and validate CSV
- Normalize column names
- Remove duplicates
- Combine Title + Author Keywords + Index Keywords
- Lowercase, remove stopwords, lemmatize
- Clean and normalize keywords

Dependencies: pandas, numpy, spacy, nltk
"""
import pandas as pd
import numpy as np
import spacy
import nltk
from typing import List, Optional

# Download NLTK stopwords if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
from nltk.corpus import stopwords

# Load spaCy model (en_core_web_sm)
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

class CSVNLPPreprocessor:
    def __init__(self, stopword_lang: str = 'english'):
        self.stopwords = set(stopwords.words(stopword_lang))

    def load_and_validate_csv(self, filepath: str) -> pd.DataFrame:
        """Load CSV, normalize columns, remove duplicates, handle missing values."""
        df = pd.read_csv(filepath)
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
        df = df.drop_duplicates()
        df = df.fillna("")
        return df

    def combine_text_fields(self, row: pd.Series, fields: List[str]) -> str:
        """Combine specified text fields into a single string."""
        return ' '.join([str(row[f]) for f in fields if f in row and pd.notnull(row[f])])

    def preprocess_text(self, text: str) -> str:
        """Lowercase, remove stopwords, lemmatize, keep alpha tokens only."""
        doc = nlp(text.lower())
        tokens = [token.lemma_ for token in doc if token.is_alpha and token.text not in self.stopwords]
        return ' '.join(tokens)

    def clean_keywords(self, keywords: str) -> str:
        """Split, lowercase, strip, remove duplicates, join by semicolon."""
        if not keywords:
            return ""
        kw_list = [k.strip().lower() for k in keywords.replace(';', ',').split(',') if k.strip()]
        return '; '.join(sorted(set(kw_list)))

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all preprocessing steps to DataFrame."""
        # Combine text fields
        text_fields = ['title', 'author_keywords', 'index_keywords']
        df['combined_text'] = df.apply(lambda row: self.combine_text_fields(row, text_fields), axis=1)
        # Preprocess combined text
        df['processed_text'] = df['combined_text'].apply(self.preprocess_text)
        # Clean keywords
        if 'author_keywords' in df.columns:
            df['author_keywords'] = df['author_keywords'].apply(self.clean_keywords)
        if 'index_keywords' in df.columns:
            df['index_keywords'] = df['index_keywords'].apply(self.clean_keywords)
        return df

# Example usage
if __name__ == "__main__":
    preprocessor = CSVNLPPreprocessor()
    df = preprocessor.load_and_validate_csv("../../cleaned_data copy.csv")
    df_clean = preprocessor.process_dataframe(df)
    print(df_clean[['title', 'processed_text', 'author_keywords', 'index_keywords']].head())

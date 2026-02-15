# 🚀 Biblio-AI  
### Intelligent Research Mapping & AI-Driven Bibliometric Analytics Platform  

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![NLP](https://img.shields.io/badge/NLP-Semantic%20Processing-green)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-orange)
![MiniLM](https://img.shields.io/badge/Embeddings-MiniLM-red)
![Gemma-3](https://img.shields.io/badge/LLM-Gemma--3-purple)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b)

---

## 📖 Overview

**Biblio-AI** is an AI-powered research intelligence system that transforms bibliographic CSV data into structured, explainable, and actionable research insights.

The system automates:

- Semantic clustering of research papers  
- Trend & gap detection  
- Topic-wise summarization  
- Evidence-based research mapping  

Instead of manually reviewing hundreds of papers, researchers can now explore structured research intelligence within minutes.

---

## 🧠 System Architecture

            CSV Upload
                │
                ▼
      NLP Preprocessing (Cleaning + TF-IDF)
                │
                ▼
      MiniLM Embedding Generation
                │
                ▼
          FAISS Vector Index
                │
                ▼
        Semantic Clustering
                │
    ┌───────────┴───────────┐
    ▼                       ▼
Trend & Gap             Gemma-3 AI
Analysis Evidence        Summarization 
View                    
---

## ⚙️ Technology Stack

| Layer | Technology |
|-------|------------|
| Data Input | CSV Bibliographic Dataset |
| NLP Processing | Text Cleaning + TF-IDF |
| Embeddings | MiniLM (Sentence Transformers) |
| Vector Database | FAISS |
| Clustering | K-Means / Semantic Similarity |
| Trend Detection | Keyword Frequency + Temporal Analysis |
| Summarization | Gemma-3 |
| Interface | Streamlit |
| Language | Python |

---

## 🔍 Core Features

### 📊 Semantic Clustering
Groups research papers based on meaning similarity rather than just keyword repetition.

---

### 📈 Trend & Gap Analysis
Detects:
- Emerging research areas  
- Declining topics  
- Underexplored research domains  
- Growth patterns over time  

---

### 📚 Evidence View (Explainable AI)

Each cluster provides:

- Top ranked papers  
- Authors  
- Publication year  
- Citation count  
- Traceable references  

Ensuring transparency and academic reliability.

---

### 🤖 AI-Powered Topic Summarization

Gemma-3 generates structured insights including:

- Main research directions  
- Key findings  
- Methodological patterns  
- Future research scope  

---

## 🎯 Problem Statement

Researchers often struggle with:

- Manual literature review  
- Identifying important themes  
- Detecting emerging research topics  
- Discovering research gaps  

Biblio-AI reduces literature review time from weeks to minutes using semantic AI.

---

## 📊 Performance Highlights

- Embedding dimension: 384 (MiniLM)
- Vector search: FAISS (efficient similarity search)
- Dataset tested: 7,000+ research papers
- Semantic clustering accuracy: High topic coherence
- Retrieval speed: Sub-second similarity search

---

## 🖥️ Demo (Add Screenshots Here)

Add screenshots inside a folder named `screenshots/` and reference like:

```markdown
![Cluster View](screenshots/cluster_view.png)
![Trend Analysis](screenshots/trend_analysis.png)
![Evidence Window](screenshots/evidence_view.png)




Biblio-AI/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
├── embeddings/
├── faiss_indexes/
│
├── modules/
│   ├── preprocessing.py
│   ├── embedding.py
│   ├── clustering.py
│   ├── trend_analysis.py
│   ├── summarization.py

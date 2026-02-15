"""
Streamlit App for BiblioAI End-to-End Pipeline

- Upload CSV file
- Run through: CSV Loader → NLP Preprocessing → Embedding → FAISS → Retrieval & Clustering → Trend & Gap Analysis → Gemma-3 Summarization
- Display results and summaries
"""
import streamlit as st
import pandas as pd
import numpy as np
from csv_loader.csv_nlp_preprocessing import CSVNLPPreprocessor
from nlp_preprocessing.nlp_preprocessing import AdvancedNLPPreprocessor
from embedding.minilm_embedder import MiniLMEmbedder
from vector_db.faiss_db import FaissVectorDB
from retrieval_clustering.retrieval_clustering import RetrievalClustering
from trend_gap_analysis.trend_gap_analysis import TrendGapAnalyzer
from llm_summarization.gemma3_summarizer import Gemma3Summarizer

st.set_page_config(page_title="BiblioAI End-to-End Pipeline", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
    st.title("BiblioAI")
    st.markdown("""
    <span style='font-size:18px;'>Intelligent Research Mapping & Analytics</span>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.info("""
    **How to use:**
    1. Upload your bibliographic CSV file
    2. Explore the pipeline steps
    3. Dive into clusters, trends, and summaries
    """)

st.markdown("""
<h1 style='text-align:center; color:#2E86C1; font-size:2.8rem; margin-bottom:0;'>📚 BiblioAI</h1>
<h3 style='text-align:center; color:#117A65; margin-top:0;'>Intelligent Research Mapping & Analytics</h3>
<hr style='border:1px solid #bbb;'>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("**Step 0: Upload your bibliographic CSV file**", type=["csv"])

if uploaded_file:
    st.markdown("""
    <div style='background-color:#F2F4F4; padding:18px; border-radius:10px; margin-bottom:20px;'>
        <span style='font-size:1.2rem; color:#2874A6;'><b>Step 1: CSV Loading & Validation</b></span>
    </div>
    """, unsafe_allow_html=True)
    csv_preproc = CSVNLPPreprocessor()
    df = csv_preproc.load_and_validate_csv(uploaded_file)
    st.markdown("<b>Raw Data Preview:</b>", unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)
    df = csv_preproc.process_dataframe(df)
    st.markdown("<b>Processed Data Preview:</b>", unsafe_allow_html=True)
    st.dataframe(df[['title', 'processed_text', 'author_keywords', 'index_keywords']].head(10), use_container_width=True, hide_index=True)

    if 'citations' not in df.columns:
        df['citations'] = 0

    st.markdown("""
    <div style='background-color:#E8F8F5; padding:18px; border-radius:10px; margin-bottom:20px;'>
        <span style='font-size:1.2rem; color:#148F77;'><b>Step 2: Advanced NLP Preprocessing</b></span>
    </div>
    """, unsafe_allow_html=True)
    nlp_adv = AdvancedNLPPreprocessor()
    tfidf_keywords = nlp_adv.extract_keywords_tfidf(df['processed_text'].tolist())
    entities = nlp_adv.extract_entities(df['processed_text'].tolist())
    st.markdown(f"<b>TF-IDF Keywords (first 2):</b> <span style='color:#2874A6'>{tfidf_keywords[:2]}</span>", unsafe_allow_html=True)
    st.markdown(f"<b>Named Entities (first 2):</b> <span style='color:#2874A6'>{entities[:2]}</span>", unsafe_allow_html=True)

    st.markdown("""
    <div style='background-color:#FEF9E7; padding:18px; border-radius:10px; margin-bottom:20px;'>
        <span style='font-size:1.2rem; color:#B9770E;'><b>Step 3: MiniLM Embedding Generation</b></span>
    </div>
    """, unsafe_allow_html=True)
    embedder = MiniLMEmbedder()
    embeddings = embedder.encode_texts(df['processed_text'].tolist())
    st.markdown(f"<b>Embeddings shape:</b> <span style='color:#B9770E'>{embeddings.shape}</span>", unsafe_allow_html=True)

    st.markdown("""
    <div style='background-color:#EBF5FB; padding:18px; border-radius:10px; margin-bottom:20px;'>
        <span style='font-size:1.2rem; color:#21618C;'><b>Step 4: FAISS Vector Database</b></span>
    </div>
    """, unsafe_allow_html=True)
    # Ensure 'Link' column is included in metadata (case-insensitive)
    link_col = None
    for col in df.columns:
        if col.lower() == 'link':
            link_col = col
            break
    meta_cols = ['title', 'author_keywords', 'index_keywords', 'year', 'citations', 'authors', 'source_title']
    if link_col:
        meta_cols.append(link_col)
    metadata = df[meta_cols].to_dict(orient='records')
    faiss_db = FaissVectorDB(embedding_dim=embeddings.shape[1])
    faiss_db.add_embeddings(embeddings, metadata)
    import os
    import pathlib
    dataset_name = pathlib.Path(uploaded_file.name).stem
    dataset_folder = f"ai_ml_layer/faiss_indexes/{dataset_name}"
    os.makedirs(dataset_folder, exist_ok=True)
    index_path = f"{dataset_folder}/faiss.index"
    metadata_path = f"{dataset_folder}/metadata.json"
    faiss_db.save(index_path, metadata_path)
    st.success(f"✅ Embeddings stored in FAISS index and saved to {index_path}.")

    st.markdown("""
    <div style='background-color:#F9EBEA; padding:18px; border-radius:10px; margin-bottom:20px;'>
        <span style='font-size:1.2rem; color:#922B21;'><b>Step 5: Retrieval & Clustering</b></span>
    </div>
    """, unsafe_allow_html=True)
    rc = RetrievalClustering(embeddings, metadata)
    num_samples = embeddings.shape[0]
    n_clusters = min(5, num_samples) if num_samples > 0 else 1
    labels = rc.cluster_embeddings(method='kmeans', n_clusters=n_clusters)
    clusters = rc.get_cluster_metadata()
    clusters_serializable = {int(k): v for k, v in clusters.items()}
    st.markdown("<b>Cluster Metadata (first 2):</b>", unsafe_allow_html=True)
    st.json({k: clusters_serializable[k] for k in list(clusters_serializable.keys())[:2]})

    st.markdown("""
    <div style='background-color:#F4ECF7; padding:18px; border-radius:10px; margin-bottom:20px;'>
        <span style='font-size:1.2rem; color:#6C3483;'><b>Step 6: Trend & Gap Analysis</b></span>
    </div>
    """, unsafe_allow_html=True)
    analyzer = TrendGapAnalyzer(clusters_serializable)
    trend_stats = analyzer.topic_trend_stats()
    trends_gaps = analyzer.identify_trends_gaps()
    def convert_npint(obj):
        if isinstance(obj, dict):
            return {k: convert_npint(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_npint(i) for i in obj]
        elif hasattr(obj, 'item') and callable(obj.item):
            return int(obj.item())
        return obj
    trends_gaps_serializable = convert_npint(trends_gaps)
    st.markdown("<b>Trend Statistics:</b>", unsafe_allow_html=True)
    st.dataframe(trend_stats, use_container_width=True, hide_index=True)
    if (isinstance(trends_gaps_serializable, dict) and (
        trends_gaps_serializable.get('insufficient_gaps') or
        (trends_gaps_serializable.get('message') and 'Insufficient data' in trends_gaps_serializable.get('message'))
    )):
        st.warning("⚠️ Insufficient data to infer emerging, declining trends or research gaps. All clusters are classified as STABLE.")
    else:
        st.markdown("<b>Trends & Gaps (cluster-level):</b>", unsafe_allow_html=True)
        st.json(trends_gaps_serializable)

        # Display clusters for each trend except stable
        if isinstance(trends_gaps_serializable, dict):
            for trend_type in ['emerging', 'declining', 'cyclical', 'anomaly', 'stable']:
                clusters_for_trend = trends_gaps_serializable.get(trend_type)
                if clusters_for_trend:
                    st.markdown(f"<b>Clusters for <span style='color:#2874A6'>{trend_type.title()} Trend</span>:</b>", unsafe_allow_html=True)
                    if isinstance(clusters_for_trend, dict):
                        cluster_ids = list(clusters_for_trend.keys())
                    elif isinstance(clusters_for_trend, list):
                        cluster_ids = clusters_for_trend
                    else:
                        cluster_ids = []
                    if cluster_ids:
                        st.write(f"Cluster IDs: {', '.join(str(cid) for cid in cluster_ids)}")
                    else:
                        st.write("No clusters found for this trend.")

    st.markdown("""
    <div style='background-color:#EAF2F8; padding:18px; border-radius:10px; margin-bottom:20px;'>
        <span style='font-size:1.2rem; color:#1B4F72;'><b>Step 7: Gemma-3 Summarization & Insights</b></span>
    </div>
    """, unsafe_allow_html=True)
    summarizer = Gemma3Summarizer()
    for cluster_id, meta in clusters.items():
        summary = summarizer.summarize_topic(meta['papers'], topic_label=f"Cluster {cluster_id}")
        with st.expander(f"🧠 Evidence View: {', '.join(meta['top_keywords'][:3])} (Cluster {cluster_id})", expanded=False):
            st.markdown(f"<div style='background-color:#FDFEFE; padding:10px; border-radius:8px;'><b>Summary:</b> {summary}</div>", unsafe_allow_html=True)
            st.markdown("**Sub-topics in this cluster:**")
            topic_selected = st.radio(
                label=f"Select a topic in Cluster {cluster_id}",
                options=meta['top_keywords'],
                key=f"topic_radio_{cluster_id}"
            )
            if topic_selected:
                topic_papers = [p for p in meta['papers'] if (isinstance(p.get('author_keywords'), str) and topic_selected.lower() in p.get('author_keywords', '').lower()) or (isinstance(p.get('index_keywords'), str) and topic_selected.lower() in p.get('index_keywords', '').lower())]
                topic_papers_sorted = sorted(
                    topic_papers,
                    key=lambda x: (-int(x.get('citations', 0)), x.get('similarity', 0)),
                )
                if topic_papers_sorted:
                    st.markdown(f"**Research Papers for topic '{topic_selected}':**")
                    table_data = []
                    for idx, paper in enumerate(topic_papers_sorted, 1):
                        # Scopus link logic: use the correct 'Link' column from metadata (case-insensitive)
                        scopus_url = None
                        # Find the link column in this paper dict
                        link_key = None
                        for k in paper.keys():
                            if k.lower() == 'link':
                                link_key = k
                                break
                        if link_key and paper.get(link_key):
                            scopus_url = paper[link_key]
                        elif 'scopus_link' in paper and paper['scopus_link']:
                            scopus_url = paper['scopus_link']
                        elif 'scopus_id' in paper and paper['scopus_id']:
                            scopus_url = f"https://www.scopus.com/record/display.uri?eid={paper['scopus_id']}"
                        table_row = {
                            "Rank": idx,
                            "Paper Title": paper.get('title', ''),
                            "Authors": paper.get('authors', ''),
                            "Publication Year": paper.get('year', ''),
                            "Citation Count": paper.get('citations', 0),
                        }
                        if scopus_url:
                            table_row["Scopus Link"] = f"[🔗 View Paper]({scopus_url})"
                        else:
                            table_row["Scopus Link"] = "N/A"
                        table_data.append(table_row)
                    df_table = pd.DataFrame(table_data)
                    # Render markdown links in dataframe
                    st.write(df_table.to_markdown(index=False), unsafe_allow_html=True)
                else:
                    st.info(f"No papers found for topic '{topic_selected}' in this cluster.")
    if (isinstance(trends_gaps_serializable, dict) and (
        trends_gaps_serializable.get('insufficient_gaps') or
        (trends_gaps_serializable.get('message') and 'Insufficient data' in trends_gaps_serializable.get('message'))
    )):
        st.subheader("Overall Research Trends & Gaps Summary")
        st.info("Insufficient data to infer emerging, declining trends or research gaps. All clusters are classified as STABLE.")
    else:
        trends_summary = summarizer.summarize_trends(trends_gaps_serializable)
        st.subheader("Overall Research Trends & Gaps Summary")
        st.markdown(f"<div style='background-color:#FDFEFE; padding:10px; border-radius:8px;'>{trends_summary}</div>", unsafe_allow_html=True)
else:
    st.info("⬆️ Upload a CSV file to begin the analysis.")

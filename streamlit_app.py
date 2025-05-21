import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import torch


import nltk
import os
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

nltk.download("punkt")
nltk.download('punkt_tab')
nltk_data_dir = os.path.expanduser('~/nltk_data')

# Set nltk data path (optional)
nltk.data.path.append(nltk_data_dir)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_dir)


st.set_page_config(page_title="JobBERT-TSDAE Dashboard", layout="wide")
st.title("JobBERT-TSDAE Embedding Dashboard")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/adinplb/dp-machinelearning-ai/refs/heads/master/dataset/combined_jobs_2000.csv"
    df = pd.read_csv(url)
    df['text'] = df['text'].fillna("")
    return df

df = load_data()
st.success(f"Loaded {len(df)} job descriptions")

@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("TechWolf/JobBERT-v2")

@st.cache_data
def encode_batch(texts):
    model = load_model()
    features = model.tokenize(texts)
    features = batch_to_device(features, model.device)
    features["text_keys"] = ["anchor"]
    with torch.no_grad():
        out_features = model.forward(features)
    return out_features["sentence_embedding"].cpu().numpy()

@st.cache_data
def encode(texts, batch_size=8):
    sorted_indices = np.argsort([len(text) for text in texts])
    sorted_texts = [texts[i] for i in sorted_indices]
    embeddings = []
    for i in range(0, len(sorted_texts), batch_size):
        batch = sorted_texts[i:i+batch_size]
        embeddings.append(encode_batch(batch))
    sorted_embeddings = np.concatenate(embeddings)
    original_order = np.argsort(sorted_indices)
    return sorted_embeddings[original_order]

def denoise_text(text, method='a', del_ratio=0.6):
    words = word_tokenize(text)
    n = len(words)
    if n == 0:
        return text
    if method == 'a':
        keep_or_not = np.random.rand(n) > del_ratio
        if sum(keep_or_not) == 0:
            keep_or_not[np.random.choice(n)] = True
        result = np.array(words)[keep_or_not]
    else:
        raise ValueError("Only method 'a' supported in dashboard.")
    return TreebankWordDetokenizer().detokenize(result)

with st.spinner("Generating TSDAE embeddings..."):
    clean_texts = df['text'].tolist()
    noisy_texts = [denoise_text(t) for t in clean_texts]
    clean_embeddings = encode(clean_texts)
    noisy_embeddings = encode(noisy_texts)
    tsdae_embeddings = (clean_embeddings + noisy_embeddings) / 2.0
    df['tsdae_embedding'] = tsdae_embeddings.tolist()

st.success("TSDAE Embeddings Generated")

num_clusters = st.sidebar.slider("Number of Clusters", min_value=5, max_value=50, value=20)

with st.spinner("Clustering embeddings..."):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(tsdae_embeddings)
    cluster_labels = kmeans.labels_
    df['cluster'] = cluster_labels

st.subheader("Cluster Distribution")
st.bar_chart(df['cluster'].value_counts().sort_index())

method = st.sidebar.radio("Dimensionality Reduction Method", ["PCA"])

if method == "PCA":
    reducer = PCA(n_components=2)
    reduced = reducer.fit_transform(tsdae_embeddings)
    df['x'] = reduced[:, 0]
    df['y'] = reduced[:, 1]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='x', y='y', hue='cluster', palette='tab10', legend=False, s=30, ax=ax)
    ax.set_title("TSDAE Embedding Clusters (2D PCA)")
    st.pyplot(fig)

st.subheader("Search and Recommend")
query = st.text_input("Enter job title or description (e.g., 'data scientist')")

if query:
    with st.spinner("Generating embedding and searching..."):
        query_embedding = encode([query])
        cluster_sim = cosine_similarity(query_embedding, kmeans.cluster_centers_)
        best_cluster = np.argmax(cluster_sim)
        cluster_subset = df[df['cluster'] == best_cluster]
        cluster_embeddings = np.vstack(cluster_subset['tsdae_embedding'].values)
        sim_scores = cosine_similarity(query_embedding, cluster_embeddings).flatten()
        top_n = st.slider("Top N recommendations", min_value=1, max_value=20, value=5)
        top_indices = np.argsort(sim_scores)[::-1][:top_n]
        recommendations = cluster_subset.iloc[top_indices][['Title', 'text']]

    st.write("## Recommendations")
    for i, row in recommendations.iterrows():
        st.markdown(f"**{row['Title']}**")
        st.write(row['text'][:300] + "...")
        st.markdown("---")

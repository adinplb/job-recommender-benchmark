import streamlit as st
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import random
from sklearn.manifold import TSNE
import plotly.express as px
from nltk.tokenize import word_tokenize, TreebankWordDetokenizer
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.util import batch_to_device

# === Title ===
st.title("📊 TSDAE Embedding Visualizer for Tech Jobs")

# === Load Dataset ===
DATA_URL = "https://raw.githubusercontent.com/adinplb/dp-machinelearning-ai/refs/heads/master/dataset/tech_jobs.csv"
df = pd.read_csv(DATA_URL, usecols=['job_id', 'combined_text', 'job_title'])
st.success("✅ Dataset loaded successfully!")

# === Denoising Function ===
def denoise_text(text, method='a', del_ratio=0.6, word_freq_dict=None, freq_threshold=100):
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
        result = words

    return TreebankWordDetokenizer().detokenize(result)

# === TSDAE Embedding Function ===
@st.cache_resource
def load_model():
    return SentenceTransformer("TechWolf/JobBERT-v2")

@st.cache_data
def generate_embeddings(df, batch_size=8):
    model = load_model()

    def encode_batch(jobbert_model, texts):
        features = jobbert_model.tokenize(texts)
        features = batch_to_device(features, jobbert_model.device)
        features["text_keys"] = ["anchor"]
        with torch.no_grad():
            out_features = jobbert_model.forward(features)
        return out_features["sentence_embedding"].cpu().numpy()

    def encode(jobbert_model, texts):
        sorted_indices = np.argsort([len(text) for text in texts])
        sorted_texts = [texts[i] for i in sorted_indices]
        embeddings = []
        for i in tqdm(range(0, len(sorted_texts), batch_size)):
            batch = sorted_texts[i:i+batch_size]
            embeddings.append(encode_batch(jobbert_model, batch))
        sorted_embeddings = np.concatenate(embeddings)
        original_order = np.argsort(sorted_indices)
        return sorted_embeddings[original_order]

    clean_texts = df['combined_text'].fillna("").tolist()
    noisy_texts = [denoise_text(txt) for txt in clean_texts]

    clean_embeddings = encode(model, clean_texts)
    noisy_embeddings = encode(model, noisy_texts)
    tsdae_embeddings = (clean_embeddings + noisy_embeddings) / 2.0

    return tsdae_embeddings

# === Generate Embeddings ===
st.subheader("🔍 Generating TSDAE Embeddings")
with st.spinner("Embedding job descriptions..."):
    tsdae_embeddings = generate_embeddings(df)
    st.success("✅ Embedding complete!")

# === t-SNE Visualization ===
st.subheader("📉 t-SNE Visualization")
perplexity = st.slider("Perplexity", min_value=5, max_value=50, value=30, step=5)
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
tsne_result = tsne.fit_transform(tsdae_embeddings)

df['tsne_1'] = tsne_result[:, 0]
df['tsne_2'] = tsne_result[:, 1]

fig = px.scatter(
    df, x='tsne_1', y='tsne_2',
    color='job_title',
    hover_data=['job_id', 'combined_text'],
    title="TSDAE Embedding Visualization with t-SNE"
)
st.plotly_chart(fig, use_container_width=True)

st.caption("Model: TechWolf/JobBERT-v2 | Embedding: TSDAE-style | Framework: Streamlit")

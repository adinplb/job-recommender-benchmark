import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import plotly.express as px


import string
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
# Set title
st.title("📊 TF-IDF Embedding Visualizer for Tech Jobs (with Preprocessing)")

# Load data
DATA_URL = "https://raw.githubusercontent.com/adinplb/dp-machinelearning-ai/refs/heads/master/dataset/tech_jobs.csv"
df = pd.read_csv(DATA_URL, usecols=['job_id', 'combined_text', 'job_title'])
st.success("✅ Dataset loaded successfully!")

# === Preprocessing Function ===
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()                             # Lowercase
    text = re.sub(r'[^\w\s]', '', text)                  # Remove punctuation
    tokens = word_tokenize(text)                         # Tokenize
    tokens = [t for t in tokens if t not in stop_words]  # Remove stopwords
    return ' '.join(tokens)

st.subheader("🧹 Preprocessing combined_text")
df['clean_text'] = df['combined_text'].fillna("").apply(preprocess_text)

# === TF-IDF Vectorization ===
st.subheader("🧠 Converting Preprocessed Text to TF-IDF Vectors")
vectorizer = TfidfVectorizer(max_features=1000)
tfidf_matrix = vectorizer.fit_transform(df['clean_text'])

# === t-SNE ===
st.subheader("📉 Reducing Dimensionality with t-SNE")
perplexity = st.slider("Perplexity", min_value=5, max_value=50, value=30, step=5)
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
tsne_result = tsne.fit_transform(tfidf_matrix.toarray())

# Add to DataFrame
df['tsne_1'] = tsne_result[:, 0]
df['tsne_2'] = tsne_result[:, 1]

# === Plot ===
fig = px.scatter(
    df,
    x='tsne_1',
    y='tsne_2',
    color='job_title',
    hover_data=['job_id', 'combined_text'],
    title="TF-IDF Embedding Visualization (t-SNE)"
)
st.plotly_chart(fig, use_container_width=True)

st.caption("🔍 Preprocessed + TF-IDF Vectorized + t-SNE Visualized")

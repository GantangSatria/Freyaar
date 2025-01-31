import streamlit as st
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5Tokenizer, T5ForConditionalGeneration

# --- STREAMLIT CONFIG ---
st.set_page_config(page_title="QA System with RAG", layout="wide")

st.title("📚 QA System dengan RAG (TF-IDF + T5)")

# --- LOAD RESOURCES ---
@st.cache_resource
def load_nltk_resources():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    return tokenizer, model

@st.cache_data
def load_data():
    df = pd.read_csv("arxiv_ml.csv")
    if "abstract" not in df.columns:
        st.error("Dataset tidak memiliki kolom 'abstract'.")
        st.stop()
    df = df.sample(1000, random_state=42) if len(df) > 1000 else df
    return df

# --- PREPROCESSING FUNCTION ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

def chunk_abstract(abstract, chunk_size=3):
    sentences = sent_tokenize(abstract)
    return [" ".join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]

# --- LOAD DATASET ---
load_nltk_resources()
df = load_data()
df["processed_abstract"] = df["abstract"].apply(preprocess_text)
df["chunks"] = df["abstract"].apply(chunk_abstract)

# --- TF-IDF RETRIEVAL ---
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["processed_abstract"])

def retrieve_top_k(query, tfidf_matrix, vectorizer, df, k=5):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = scores.argsort()[-k:][::-1]
    return df.iloc[top_indices]

# --- LOAD T5 MODEL ---
tokenizer, model = load_model()

def generate_answer(question, df):
    combined_context = " ".join(df["chunks"].sum())
    input_text = f"question: {question} context: {combined_context}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)
    output_ids = model.generate(input_ids, max_length=100, num_beams=5, early_stopping=True)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# --- STREAMLIT UI ---
question = st.text_input("❓ Masukkan pertanyaan Anda tentang Machine Learning:")

if question:
    with st.spinner("🔍 Mencari jawaban..."):
        retrieved_docs = retrieve_top_k(question, tfidf_matrix, vectorizer, df, k=5)
        answer = generate_answer(question, retrieved_docs)
    
    st.write("### 🔎 Konteks yang ditemukan:")
    for i, context in enumerate(retrieved_docs["chunks"].values, 1):
        st.info(f"**Dokumen {i}:** {' '.join(context[:2])}...")  # Tampilkan cuplikan

    st.write("### ✅ Jawaban:")
    st.success(answer)

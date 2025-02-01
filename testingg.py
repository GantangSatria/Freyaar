import streamlit as st
import pandas as pd
import re
import nltk
import os
import faiss
import numpy as np
import torch
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Set page config
st.set_page_config(page_title="ML QA System", layout="wide")

nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download NLTK resources
nltk.download('punkt', quiet=True, download_dir=nltk_data_path)
nltk.download('stopwords', quiet=True, download_dir=nltk_data_path)
nltk.download('wordnet', quiet=True, download_dir=nltk_data_path)
nltk.download('omw-1.4', quiet=True, download_dir=nltk_data_path)
nltk.download('punkt_tab', quiet=True, download_dir=nltk_data_path)

@st.cache_data
def load_data():
    df = pd.read_csv("arxiv_ml.csv")
    if len(df) > 5000:
        df = df.sample(5000, random_state=42)
    return df

@st.cache_data
def preprocess_data(df):
    # Ensure the 'abstract' column exists and handle missing values
    if 'abstract' not in df.columns:
        raise ValueError("The 'abstract' column is missing in the dataset.")
    
    # Fill NaN values with empty strings
    df['abstract'] = df['abstract'].fillna('')

    # Preprocessing function
    def preprocess_text(text):
        if not isinstance(text, str):  # Ensure the input is a string
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return " ".join(tokens)
    
    # Apply preprocessing
    df["processed_abstract"] = df["abstract"].apply(preprocess_text)
    
    # Chunking
    df["chunks"] = df["abstract"].apply(lambda x: [" ".join(sent_tokenize(x)[i:i+3]) for i in range(0, len(sent_tokenize(x)), 3)])
    return df

@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    answer_model = T5ForConditionalGeneration.from_pretrained("t5-small")
    return embedding_model, tokenizer, answer_model

@st.cache_resource
def create_faiss_index(df, embedding_model):
    embeddings = embedding_model.encode(df["processed_abstract"].tolist(), convert_to_tensor=False)
    embeddings = np.array(embeddings, dtype=np.float32)
    index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
    index.add(embeddings)
    return index

# Load data and models
df = load_data()
df = preprocess_data(df)
embedding_model, tokenizer, answer_model = load_models()
index = create_faiss_index(df, embedding_model)

# Ground truth data
ground_truth = {
    "What is machine learning?": "Machine learning is a branch of artificial intelligence (AI) that allows computer systems to learn and evolve independently...",
    # ... (other ground truth entries)
}

# Streamlit UI
st.title("QA Machine Learning System :rocket:")

question = st.text_input("Enter your question about machine learning:", "")
k = st.slider("Number of chunks to retrieve:", 1, 10, 5)

if st.button("Get Answer"):
    if question:
        # Retrieve documents
        query_embedding = embedding_model.encode([question], convert_to_tensor=False).astype(np.float32)
        _, top_indices = index.search(query_embedding, k)
        retrieved_docs = df.iloc[top_indices[0]]
        
        # Generate answer
        combined_context = " ".join(retrieved_docs["chunks"].sum())
        input_text = f"question: {question} context: {combined_context}"
        input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)
        output_ids = answer_model.generate(input_ids, max_length=100, num_beams=5, early_stopping=True)
        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Calculate scores
        context_text = " ".join(retrieved_docs["chunks"].sum())
        response_score = util.pytorch_cos_sim(
            embedding_model.encode(answer, convert_to_tensor=True),
            embedding_model.encode(context_text, convert_to_tensor=True)
        ).item()
        
        # Display results
        st.subheader("Answer:")
        st.write(answer)
        
        st.subheader("Evaluation Metrics:")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Response Relevancy Score", f"{response_score:.2f}")
        
        # Ground truth comparison
        if question in ground_truth:
            gt_score = util.pytorch_cos_sim(
                embedding_model.encode(answer, convert_to_tensor=True),
                embedding_model.encode(ground_truth[question], convert_to_tensor=True)
            ).item()
            with col2:
                st.metric("Ground Truth Similarity", f"{gt_score:.2f}")
            st.subheader("Ground Truth Answer:")
            st.write(ground_truth[question])
        else:
            st.warning("No ground truth available for this question")
            
        st.subheader("Retrieved Contexts:")
        for i, context in enumerate(retrieved_docs["chunks"].iloc[0][:3], 1):
            st.write(f"Context {i}: {context}")
    
    else:
        st.warning("Please enter a question first!")
import streamlit as st
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pickle
import torch
from transformers import BertTokenizer, BertModel
import os

# --- Konfigurasi Path ---
# Menggunakan path relatif agar fleksibel
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
BERT_DIR = os.path.join(BASE_DIR, 'bert_model_tersimpan')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# --- di dalam fungsi load_all_resources() ---
indo_stopwords = set(stopwords.words('indonesian'))
# ---

# --- Load Semua Model dan Objek yang Dibutuhkan ---
@st.cache_resource
def load_models_and_data():
    # Load 3 model klasifikasi dari file .pkl
    with open(os.path.join(MODELS_DIR, 'random_forest_model.pkl'), 'rb') as file:
        rf_model = pickle.load(file)
    with open(os.path.join(MODELS_DIR, 'svm_model.pkl'), 'rb') as file:
        svm_model = pickle.load(file)
    with open(os.path.join(MODELS_DIR, 'knn_model.pkl'), 'rb') as file:
        knn_model = pickle.load(file)

    models = {
        'Random Forest': rf_model,
        'SVM': svm_model,
        'KNN': knn_model
    }

    # Load tokenizer dan model IndoBERT
    tokenizer = BertTokenizer.from_pretrained(os.path.join(BERT_DIR, 'tokenizer'))
    bert_model = BertModel.from_pretrained(os.path.join(BERT_DIR, 'model'))

    # Load kamus slang
    normalisasi_dict = {}
    with open(os.path.join(DATA_DIR, "slang.txt"), "r", encoding="utf-8") as file:
        for line in file:
            if ":" in line:
                f = line.strip().split(":")
                normalisasi_dict[f[0].strip()] = f[1].strip()

    # Load stopwords
    indo_stopwords = set(stopwords.words('indonesian'))

    # Buat stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    return models, tokenizer, bert_model, normalisasi_dict, indo_stopwords, stemmer

models, tokenizer, bert_model, normalisasi_dict, indo_stopwords, stemmer = load_models_and_data()

# --- Fungsi Preprocessing ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'[-+]?[0-9]+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    for slang, baku in normalisasi_dict.items():
        pattern = r'\b' + re.escape(slang) + r'\b'
        text = re.sub(pattern, baku, text, flags=re.IGNORECASE)
    tokens = text.split()
    text = ' '.join([word for word in tokens if word not in indo_stopwords])
    text = stemmer.stem(text)
    return text

# --- Fungsi Ekstraksi Fitur BERT ---
def get_bert_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return cls_embedding.reshape(1, -1)

# --- Antarmuka Streamlit ---
st.set_page_config(page_title="Aplikasi Klasifikasi Sentimen", page_icon="💬")
st.title("💬 Aplikasi Klasifikasi Sentimen")
st.write("Pilih model dan masukkan teks untuk melakukan prediksi sentimen.")

model_choice = st.selectbox(
    "Pilih Model",
    ('Random Forest', 'SVM', 'KNN')
)

user_input = st.text_area("Masukkan teks untuk prediksi sentimen", "Saya sangat suka dengan program makan gratis ini, sangat membantu sekali!", height=150)

if st.button("Prediksi Sentimen", use_container_width=True):
    if user_input:
        with st.spinner(f'Memproses dengan model {model_choice}...'):
            selected_model = models[model_choice]
            
            cleaned_text = preprocess_text(user_input)
            
            bert_features = get_bert_embedding(cleaned_text, tokenizer, bert_model)
            prediction = selected_model.predict(bert_features)
            
            sentiment = "Positif" if prediction[0] == 1 else "Negatif"
            
            if sentiment == "Positif":
                st.success(f"Prediksi Sentimen: **{sentiment}**")
            else:
                st.error(f"Prediksi Sentimen: **{sentiment}**")
            
            if hasattr(selected_model, 'predict_proba'):
                prediction_proba = selected_model.predict_proba(bert_features)
                st.write("Probabilitas:")
                st.progress(prediction_proba[0][1])
                st.write(f"Positif: `{prediction_proba[0][1]:.2%}` | Negatif: `{prediction_proba[0][0]:.2%}`")
            else:
                st.info("Model SVM (LinearSVC) ini tidak menyediakan output probabilitas.")
    else:
        st.warning("Mohon masukkan teks terlebih dahulu.")
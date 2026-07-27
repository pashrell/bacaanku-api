from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import re
import os
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. LOAD ARTEFAK RINGAN (Aman diload di awal)
# ==========================================
print("Memuat dataset dan artefak ML...")
df = pd.read_pickle('dataset_jaklitera_processed.pkl')
# Jadikan array numpy (.values) agar cepat saat operasi vektor / Late Fusion
has_desc_mask = df['has_desc'].values 

artifacts = joblib.load('tfidf_artifacts.joblib')
vec_judul = artifacts['vec_judul']
vec_desc = artifacts['vec_desc']
vec_kategori = artifacts['vec_kategori']
mat_judul = artifacts['mat_judul']
mat_desc = artifacts['mat_desc']
mat_kategori = artifacts['mat_kategori']
mat_penulis = artifacts['mat_penulis']

sbert_embeddings = joblib.load('sbert_embeddings.joblib')
print("✅ Artefak ringan siap.")


# ==========================================
# 2. LAZY-LOAD SBERT LEWAT LIFESPAN (Beban Memori Paling Berat)
# ==========================================
ml_models = {}

# Konfigurasi ini memungkinkan deployment membaca model dari folder lokal terlebih dahulu.
# Sangat krusial untuk server produksi yang sering mengalami timeout jika harus download ulang.
ONNX_MODEL_PATH = "./onnx_model"

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Memuat model NLP Multilingual (ONNX backend)...")
    if os.path.isdir(ONNX_MODEL_PATH):
        # Gunakan model lokal jika folder onnx_model tersedia
        ml_models["sbert"] = SentenceTransformer(ONNX_MODEL_PATH, backend="onnx")
    else:
        # Fallback download dari HuggingFace (backend onnx tetap aktif)
        ml_models["sbert"] = SentenceTransformer(
            'paraphrase-multilingual-MiniLM-L12-v2', backend="onnx"
        )
    print("✅ Server API Siap & Model NLP telah dimuat!")
    yield
    # Bebaskan memori saat server dimatikan
    ml_models.clear()


# ==========================================
# 3. SETUP APLIKASI & CORS
# ==========================================
app = FastAPI(title="Sistem Rekomendasi Jaklitera API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti dengan domain frontend Anda saat production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 4. FUNGSI UTILITAS & SCHEMA REQUEST
# ==========================================
def clean_text(text: str) -> str:
    if pd.isna(text) or str(text).strip() == '': return ''
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

class ItemRequest(BaseModel):
    judul_buku: str
    top_n: int = 10

class StoryRequest(BaseModel):
    cerita: str
    top_n: int = 10

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": "sbert" in ml_models}


# ==========================================
# 5. ENDPOINT MODEL 1: ITEM-TO-ITEM
# ==========================================
@app.post("/api/v1/recommend/catalog")
def recommend_by_catalog(req: ItemRequest):
    title_clean = clean_text(req.judul_buku)
    
    # Cari index buku
    exact = df[df['judul_clean'] == title_clean]
    if len(exact) > 0:
        query_idx = exact.index[0]
    else:
        partial = df[df['judul_clean'].str.contains(title_clean, na=False)]
        if len(partial) > 0:
            query_idx = partial.index[0]
        else:
            raise HTTPException(status_code=404, detail="Buku tidak ditemukan di katalog.")

    # Skema Bobot Model 1
    has_desc = bool(df.loc[query_idx, 'has_desc'])
    w = {'judul': 0.40, 'desc': 0.35, 'penulis': 0.15, 'kategori': 0.10} if has_desc else {'judul': 0.60, 'desc': 0.0, 'penulis': 0.25, 'kategori': 0.15}

    sim_j = cosine_similarity(mat_judul[query_idx], mat_judul).flatten()
    sim_d = cosine_similarity(mat_desc[query_idx],  mat_desc).flatten()
    sim_p = cosine_similarity(mat_penulis[query_idx], mat_penulis).flatten()
    sim_k = cosine_similarity(mat_kategori[query_idx], mat_kategori).flatten()

    hybrid_scores = (w['judul']*sim_j + w['desc']*sim_d + w['penulis']*sim_p + w['kategori']*sim_k)
    
    sorted_idx = np.argsort(hybrid_scores)[::-1]
    sorted_idx = [i for i in sorted_idx if i != query_idx][:req.top_n]
    
    results = df.loc[sorted_idx, ['Judul', 'Penulis','Deskripsi', 'Kategori_Asal', 'Cover_URL']].copy()
    results['Skor_Kemiripan'] = hybrid_scores[sorted_idx].round(3)

    buku_acuan = df.loc[query_idx, ['Judul', 'Penulis', 'Cover_URL']].fillna("").to_dict()
    
    return {"status": "success", "buku_acuan": buku_acuan, "kategori_pencarian": "Item-to-Item", "data": results.to_dict(orient='records')}


# ==========================================
# 6. ENDPOINT MODEL 2: SEMANTIC SEARCH
# ==========================================
@app.post("/api/v1/recommend/semantic")
def recommend_by_semantic(req: StoryRequest):
    cleaned_query = clean_text(req.cerita)
    
    # Ambil model dari lifespan
    sbert_model = ml_models["sbert"]
    
    # Vektorisasi
    query_embedding = sbert_model.encode([cleaned_query])
    query_tfidf_desc = vec_desc.transform([cleaned_query])
    query_tfidf_judul = vec_judul.transform([cleaned_query])
    query_tfidf_kategori = vec_kategori.transform([cleaned_query])

    # Perhitungan Jarak
    sim_sbert = np.zeros(len(df))
    sim_sbert[has_desc_mask] = cosine_similarity(query_embedding, sbert_embeddings).flatten()
    
    sim_tfidf_desc = cosine_similarity(query_tfidf_desc, mat_desc).flatten()
    sim_tfidf_judul = cosine_similarity(query_tfidf_judul, mat_judul).flatten()
    sim_tfidf_kategori = cosine_similarity(query_tfidf_kategori, mat_kategori).flatten()
    
    sim_tfidf_fallback = (sim_tfidf_judul + sim_tfidf_kategori) / 2.0

    # Late Fusion Model 2 (Vektorisasi penuh dengan numpy, tanpa for-loop)
    final_scores = np.where(
        has_desc_mask, 
        (0.7 * sim_sbert) + (0.3 * sim_tfidf_desc), 
        sim_tfidf_fallback * 0.6
    )

    top_indices = final_scores.argsort()[::-1][:req.top_n]
    
    results = df.iloc[top_indices][['Judul', 'Penulis','Deskripsi', 'Kategori_Asal', 'Cover_URL']].copy()
    results['Skor_Relevansi'] = final_scores[top_indices].round(3)
    
    return {"status": "success", "kategori_pencarian": "Semantic", "data": results.to_dict(orient='records')}
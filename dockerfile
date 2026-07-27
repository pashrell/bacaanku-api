FROM python:3.11-slim

WORKDIR /app

# Dependensi sistem minimal yang dibutuhkan onnxruntime/torch CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- PENTING: export & cache model SAAT BUILD, bukan saat container start ---
# Ini memastikan:
# 1. Container tidak perlu akses internet ke HuggingFace saat cold start
#    (Railway free/hobby tier bisa gagal/lambat kalau harus download ~470MB tiap deploy)
# 2. Model sudah dalam format ONNX (ringan, tanpa perlu torch runtime penuh saat inference)
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
m = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', backend='onnx'); \
m.save_pretrained('./onnx_model')"

COPY main.py .
COPY dataset_jaklitera_processed.pkl .
COPY tfidf_artifacts.joblib .
COPY sbert_embeddings.joblib .

ENV PORT=8000
EXPOSE 8000

# Railway inject $PORT otomatis, uvicorn baca dari env
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
# Menggunakan base image Python 3.10 versi slim agar ukuran image lebih kecil
FROM python:3.10-slim

# Menentukan direktori kerja di dalam container
WORKDIR /app

# Menginstal dependensi sistem dasar yang sering dibutuhkan oleh library C++ / Machine Learning
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Menyalin file requirements dan menginstal dependensi Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# [PENTING] Melakukan pre-download model SBERT saat proses build Docker.
# Ini mencegah server Railway timeout akibat mengunduh model saat startup.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', backend='onnx')"

# Menyalin seluruh sisa file proyek (kode, dataset, dan artefak model) ke dalam container
COPY . .

# Mengekspos port 8000 (meskipun Railway akan menimpa ini dengan variabel $PORT)
EXPOSE 8000

# Perintah untuk menjalankan server menggunakan Uvicorn.
# Railway secara otomatis menyuntikkan variabel environment $PORT.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
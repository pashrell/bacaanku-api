# Gunakan versi Python yang ringan
FROM python:3.9-slim

# Atur direktori kerja di dalam server
WORKDIR /app

# Salin file requirements.txt dan instal library
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode Anda ke dalam server
COPY . .

# Hugging Face Spaces secara default menggunakan port 7860
EXPOSE 7860

# Perintah untuk menjalankan FastAPI
# Ganti 'main:app' jika nama file Python Anda bukan main.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
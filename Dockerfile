# Menggunakan Python 3.12 sebagai base image
FROM python:3.12

# Menetapkan working directory
WORKDIR /app

# Menyalin semua file ke dalam container
COPY . .

# Menginstall dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Menjalankan aplikasi menggunakan Gunicorn (lebih stabil di Railway)
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]

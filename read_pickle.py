import pickle
import numpy as np
import os

# Path file face embeddings
input_path = "face_embeddings.pkl"

# Membaca file .pkl
if not os.path.exists(input_path):
    print(f"File {input_path} tidak ditemukan.")
else:
    with open(input_path, "rb") as f:
        embeddings = pickle.load(f)
    
    # Menampilkan isi embedding
    print("Daftar Orang dalam Database:")
    for person, vectors in embeddings.items():
        print(f"\nNama: {person}")
        print(f"Jumlah Embedding: {len(vectors)}")
        
        # Menampilkan maksimal 5 vektor pertama jika ada
        num_vectors_to_display = min(5, len(vectors))
        for i in range(num_vectors_to_display):
            print(f"Vektor {i+1}: {vectors[i][:5]}")  # Menampilkan hanya 5 elemen pertama
import os
import time
import numpy as np
import pickle
from keras_facenet import FaceNet
from ultralytics import YOLO
from PIL import Image
import cv2
from collections import defaultdict
import hashlib
import sys

# Konfigurasi confidence threshold
DEFAULT_CONFIDENCE = 0.6
confidence_threshold = float(os.environ.get('CONFIDENCE_THRESHOLD', DEFAULT_CONFIDENCE))

# Validasi confidence threshold
if not (0.5 <= confidence_threshold <= 0.7):
    print(f"\nPERINGATAN: Confidence threshold {confidence_threshold} di luar rentang yang direkomendasikan (0.5-0.7).")
    print("Menggunakan nilai default 0.6")
    confidence_threshold = 0.6

# Cetak informasi tentang threshold yang digunakan
print("\n" + "=" * 60)
print("REBUILD FACE EMBEDDINGS DATABASE".center(60))
print("=" * 60)
print(f"Menggunakan confidence threshold: {confidence_threshold}")
print("Nilai lebih tinggi = deteksi lebih akurat tapi mungkin melewatkan beberapa wajah")
print("Nilai lebih rendah = deteksi lebih banyak wajah tapi berpotensi false positive")
print("=" * 60 + "\n")

# Konfirmasi sebelum menjalankan proses dengan informasi yang lebih jelas
print("PERINGATAN⚠️: Proses ini akan:")
print("  1. Menghapus file embeddings lama (face_embeddings.pkl)")
print("  2. Menghapus file metadata lama (face_embeddings_metadata.pkl)")
print("  3. Memproses ulang seluruh database wajah di folder 'database'")
print("  4. Membuat file embeddings dan metadata baru")
print("\nProses ini akan memakan waktu yang cukup lama tergantung pada")
print("jumlah gambar dan spesifikasi komputer Anda.\n")

if "SKIP_CONFIRMATION" in os.environ and os.environ["SKIP_CONFIRMATION"].lower() == "true":
    run_process = "y"
else:
    run_process = input("Apakah Anda ingin melanjutkan? (y/n): ")

if run_process.lower() != 'y':
    print("Proses dibatalkan oleh pengguna.")
    sys.exit(0)
    
# Hapus file embeddings sebelumnya
output_path = "face_embeddings.pkl"
metadata_path = "face_embeddings_metadata.pkl"

if os.path.exists(output_path):
    os.remove(output_path)
    print(f"File {output_path} telah dihapus untuk membuat yang baru.")

if os.path.exists(metadata_path):
    os.remove(metadata_path)
    print(f"File {metadata_path} telah dihapus untuk membuat yang baru.")

# Inisialisasi model
print("\nMemuat model YOLOv8 dan FaceNet...")
yolo_model = YOLO("yolov8m-face.pt")
embedder = FaceNet()
print("Model berhasil dimuat.\n")

# Fungsi membaca dan memproses gambar ke RGB
def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        return image
    except Exception as e:
        raise ValueError(f"Error loading image {image_path}: {e}")

# Fungsi mendeteksi wajah menggunakan YOLOv8 dengan confidence threshold
def detect_faces(image):
    results = yolo_model(image, conf=confidence_threshold)  # Gunakan confidence threshold yang telah dikonfigurasi
    detections = results[0].boxes.xyxy.numpy()  # Bounding box (x1, y1, x2, y2)
    # Tambahkan juga confidence scores untuk logging
    confidence_scores = results[0].boxes.conf.numpy()
    
    faces = []
    boxes_with_scores = []
    
    for i, box in enumerate(detections):
        x1, y1, x2, y2 = map(int, box)
        face = image[y1:y2, x1:x2]  # Crop wajah dari gambar
        face = cv2.resize(face, (160, 160))
        faces.append(face)
        boxes_with_scores.append((box, confidence_scores[i]))
    
    return faces, boxes_with_scores

# Fungsi baru untuk menghitung hash file
def get_image_hash(image_path):
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# Path direktori wajah yang sudah dikenal
known_faces_dir = "database"

# Periksa apakah folder database ada
if not os.path.exists(known_faces_dir):
    print(f"ERROR: Folder database '{known_faces_dir}' tidak ditemukan!")
    print("Pastikan folder database ada dan berisi subfolder untuk setiap orang.")
    sys.exit(1)

# Dictionary untuk menyimpan embeddings (format asli)
embeddings = {}

# Dictionary tambahan untuk menyimpan metadata (format yang kompatibel dengan kode update dan rebuild)
metadata = {}

# Hitung total gambar yang akan diproses untuk progress tracking
total_images = 0
for person_name in os.listdir(known_faces_dir):
    person_path = os.path.join(known_faces_dir, person_name)
    if os.path.isdir(person_path):
        for image_name in os.listdir(person_path):
            if image_name.lower().endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG')):
                total_images += 1

# Proses setiap subfolder
print("Starting embedding generation...")
print(f"Total folder: {len([d for d in os.listdir(known_faces_dir) if os.path.isdir(os.path.join(known_faces_dir, d))])}")
print(f"Total gambar: {total_images}\n")

processed_images = 0
start_time = time.time()

for person_name in os.listdir(known_faces_dir):
    person_path = os.path.join(known_faces_dir, person_name)
    if os.path.isdir(person_path):
        # Inisialisasi embedings untuk person ini
        embeddings[person_name] = []
        # Inisialisasi metadata untuk person ini
        metadata[person_name] = {}
        
        print(f"Processing folder: {person_name}")
        for image_name in os.listdir(person_path):
            if image_name.lower().endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG')):
                image_path = os.path.join(person_path, image_name)
                processed_images += 1
                
                # Hitung dan tampilkan progress
                progress = (processed_images / total_images) * 100
                elapsed_time = time.time() - start_time
                images_per_sec = processed_images / elapsed_time if elapsed_time > 0 else 0
                
                print(f"  - [{processed_images}/{total_images}] ({progress:.1f}%) {image_name} - {images_per_sec:.2f} img/s", end="")
                
                try:
                    # Preprocess gambar
                    image = preprocess_image(image_path)
                    
                    # Deteksi wajah dengan YOLOv8
                    faces, face_details = detect_faces(image)
                    
                    # Hitung jumlah wajah yang terdeteksi pada gambar ini
                    face_count = len(faces)
                    
                    # Tambahkan info confidence scores
                    confidence_info = ", ".join([f"{score:.2f}" for _, score in face_details]) if face_details else "N/A"
                    
                    # Hitung hash dari gambar (ditambahkan untuk kompatibilitas)
                    image_hash = get_image_hash(image_path)
                    
                    # Catat starting index untuk embedding baru
                    start_idx = len(embeddings[person_name])
                    
                    # Proses setiap wajah yang terdeteksi
                    for face in faces:
                        # Buat embedding
                        embedding = embedder.embeddings([face])[0]
                        
                        # Tambahkan embedding ke list
                        embeddings[person_name].append(embedding)
                    
                    # Hitung indeks embedding untuk metadata
                    end_idx = len(embeddings[person_name])
                    embedding_indices = list(range(start_idx, end_idx))
                    
                    # Format metadata yang kompatibel dengan kode update dan rebuild
                    metadata[person_name][image_name] = {
                        "hash": image_hash,
                        "embedding_indices": embedding_indices,
                        # Tambahkan info tambahan yang tidak akan mengganggu kompatibilitas
                        "faces_detected": face_count,
                        "confidence_scores": [float(score) for _, score in face_details]
                    }
                    
                    # Cetak informasi tentang wajah yang terdeteksi
                    if face_count > 1:
                        print(f" - {face_count} wajah terdeteksi (conf: {confidence_info})")
                    elif face_count == 1:
                        print(f" - 1 wajah terdeteksi (conf: {confidence_info})")
                    else:
                        print(f" - Tidak ada wajah terdeteksi")
                        
                except Exception as e:
                    print(f" - ERROR: {e}")
                    # Tetap catat dalam metadata bahwa file ini bermasalah (dengan format yang kompatibel)
                    metadata[person_name][image_name] = {
                        "hash": get_image_hash(image_path),
                        "embedding_indices": [],
                        "faces_detected": 0,
                        "error": str(e)
                    }

# Simpan embeddings ke file .pkl (format aslinya tetap dipertahankan)
with open(output_path, "wb") as f:
    pickle.dump(embeddings, f)

# Simpan metadata ke file terpisah
with open(metadata_path, "wb") as f:
    pickle.dump(metadata, f)

print(f"\nDatabase embedding telah dibuat dan disimpan di {output_path}")
print(f"Metadata embedding telah disimpan di {metadata_path}")

# Buat ringkasan
print("\nRingkasan:")
total_duration = time.time() - start_time
total_people = len(embeddings)
total_embeddings = sum(len(embs) for embs in embeddings.values())
multiface_images = 0
no_face_images = 0

for person, files in metadata.items():
    for filename, info in files.items():
        if info["faces_detected"] > 1:
            multiface_images += 1
        elif info["faces_detected"] == 0:
            no_face_images += 1

print(f"Total orang: {total_people}")
print(f"Total embeddings: {total_embeddings}")
print(f"Gambar dengan multiple wajah: {multiface_images}")
print(f"Gambar tanpa wajah terdeteksi: {no_face_images}")
print(f"Waktu pemrosesan: {total_duration:.2f} detik ({(total_duration/60):.2f} menit)")
print(f"Kecepatan rata-rata: {total_images/total_duration:.2f} gambar/detik")
print(f"Confidence threshold yang digunakan: {confidence_threshold}")
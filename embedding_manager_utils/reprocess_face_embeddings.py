import os
import numpy as np
import pickle
import argparse
from keras_facenet import FaceNet
from ultralytics import YOLO
from PIL import Image
import cv2
import hashlib
import time
from datetime import datetime
from collections import defaultdict

def reprocess_problem_faces(database_dir="database", 
                           embeddings_path="face_embeddings.pkl",
                           metadata_path="face_embeddings_metadata.pkl",
                           confidence_threshold=0.6):

    print("\n" + "=" * 60)
    print(f"MEMPROSES ULANG WAJAH BERMASALAH (Batas Kepercayaan: {confidence_threshold})")
    print("=" * 60)
    
    start_time = time.time()
    
    # Validasi batas kepercayaan
    if not 0.3 <= confidence_threshold <= 0.7:
        print(f"Peringatan: Batas kepercayaan {confidence_threshold} di luar rentang yang direkomendasikan (0.3-0.7).")
        # Lanjutkan tanpa konfirmasi karena ini web app
    
    # Periksa apakah file yang diperlukan ada
    if not os.path.exists(embeddings_path):
        print(f"Error: File embeddings {embeddings_path} tidak ditemukan.")
        return {
            "status": "error",
            "message": f"File embeddings {embeddings_path} tidak ditemukan."
        }
        
    if not os.path.exists(metadata_path):
        print(f"Error: File metadata {metadata_path} tidak ditemukan.")
        return {
            "status": "error",
            "message": f"File metadata {metadata_path} tidak ditemukan."
        }
        
    if not os.path.exists(database_dir):
        print(f"Error: Direktori database {database_dir} tidak ditemukan.")
        return {
            "status": "error",
            "message": f"Direktori database {database_dir} tidak ditemukan."
        }
    
    # Muat data yang ada
    print("Memuat embeddings dan metadata yang ada...")
    try:
        with open(embeddings_path, "rb") as f:
            embeddings = pickle.load(f)
            
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
    except Exception as e:
        print(f"Error saat memuat data: {e}")
        return {
            "status": "error",
            "message": f"Error saat memuat data: {e}"
        }
    
    # Inisialisasi model (hanya ketika kita yakin membutuhkannya)
    print("Menginisialisasi model deteksi wajah dan embedding...")
    yolo_model = YOLO("yolov8m-face.pt")
    embedder = FaceNet()
    
    # Fungsi untuk mendapatkan hash gambar
    def get_image_hash(image_path):
        with open(image_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    
    # Fungsi untuk pra-pemrosesan gambar
    def preprocess_image(image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
            return image
        except Exception as e:
            raise ValueError(f"Error saat memuat gambar {image_path}: {e}")
    
    # Fungsi untuk mendeteksi wajah dengan batas kepercayaan
    def detect_faces(image, conf_threshold):
        results = yolo_model(image)
        
        # Dapatkan kotak pembatas dan skor kepercayaan
        detections = results[0].boxes.xyxy.numpy()  # koordinat kotak pembatas
        confidences = results[0].boxes.conf.numpy()  # skor kepercayaan
        
        faces = []
        # Loop melalui deteksi dan filter berdasarkan skor kepercayaan
        for i, (box, conf) in enumerate(zip(detections, confidences)):
            if conf >= conf_threshold:
                x1, y1, x2, y2 = map(int, box)
                face = image[y1:y2, x1:x2]  # crop wajah
                face = cv2.resize(face, (160, 160))
                faces.append(face)
            else:
                print(f"    Mengabaikan deteksi wajah dengan kepercayaan rendah: {conf:.2f} < {conf_threshold}")
                
        return faces
    
    # Identifikasi gambar bermasalah
    problem_images = []
    
    # Untuk menampilkan daftar file bermasalah per folder
    problems_by_person = defaultdict(list)
    
    print("\nMenganalisis metadata untuk mengidentifikasi gambar bermasalah...")
    for person_name, person_metadata in metadata.items():
        for filename, info in person_metadata.items():
            # Periksa masalah berdasarkan metadata
            if "faces_detected" in info:
                # Gambar tanpa wajah terdeteksi
                if info["faces_detected"] == 0:
                    person_dir = os.path.join(database_dir, person_name)
                    image_path = os.path.join(person_dir, filename)
                    if os.path.exists(image_path):
                        issue = "Tidak ada wajah terdeteksi"
                        problem_images.append((person_name, filename, image_path, issue))
                        problems_by_person[person_name].append((filename, issue))
                
                # Gambar dengan banyak wajah terdeteksi
                elif info["faces_detected"] > 1:
                    person_dir = os.path.join(database_dir, person_name)
                    image_path = os.path.join(person_dir, filename)
                    if os.path.exists(image_path):
                        issue = f"Banyak wajah: {info['faces_detected']}"
                        problem_images.append((person_name, filename, image_path, issue))
                        problems_by_person[person_name].append((filename, issue))
            
            # Periksa entri error
            if "error" in info:
                person_dir = os.path.join(database_dir, person_name)
                image_path = os.path.join(person_dir, filename)
                if os.path.exists(image_path):
                    issue = f"Error: {info['error']}"
                    problem_images.append((person_name, filename, image_path, issue))
                    problems_by_person[person_name].append((filename, issue))
    
    # Temukan file dalam database yang tidak memiliki entri metadata
    for person_name in os.listdir(database_dir):
        person_dir = os.path.join(database_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
            
        # Periksa apakah orang ada dalam metadata
        if person_name not in metadata:
            # Semua file untuk orang ini perlu diproses
            for filename in os.listdir(person_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
                    image_path = os.path.join(person_dir, filename)
                    issue = "Metadata tidak ada"
                    problem_images.append((person_name, filename, image_path, issue))
                    problems_by_person[person_name].append((filename, issue))
            continue
            
        # Periksa file individu
        for filename in os.listdir(person_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
                if filename not in metadata[person_name]:
                    image_path = os.path.join(person_dir, filename)
                    issue = "Metadata tidak ada"
                    problem_images.append((person_name, filename, image_path, issue))
                    problems_by_person[person_name].append((filename, issue))
    
    # Tampilkan daftar file bermasalah per folder
    print("\nDaftar file bermasalah per folder:")
    print("=" * 60)
    if problems_by_person:
        for person_name, issues in problems_by_person.items():
            print(f"\nFolder: {person_name} ({len(issues)} file bermasalah)")
            print("-" * 60)
            for idx, (filename, issue) in enumerate(issues, 1):
                print(f"{idx}. {filename} - {issue}")
    else:
        print("Tidak ada file bermasalah ditemukan.")
    print("=" * 60)
    
    # Laporan temuan
    print(f"\nDitemukan total {len(problem_images)} gambar bermasalah untuk diproses ulang.")
    
    if not problem_images:
        print("Tidak ada gambar bermasalah ditemukan. Database tampaknya dalam kondisi baik.")
        return {
            "status": "success",
            "message": "Tidak ada gambar bermasalah ditemukan. Database dalam kondisi baik.",
            "total_problematic_files": 0,
            "success_count": 0,
            "error_count": 0
        }
    
    # Penghitung statistik
    reprocessed_count = 0
    success_count = 0
    error_count = 0
    
    # Proses ulang gambar bermasalah - Tidak ada konfirmasi untuk web app
    print("\nMemproses ulang gambar bermasalah...")
    for idx, (person_name, filename, image_path, issue) in enumerate(problem_images):
        print(f"\n[{idx+1}/{len(problem_images)}] Memproses: {person_name}/{filename}")
        print(f"  Masalah: {issue}")
        
        try:
            # Inisialisasi orang dalam kamus jika diperlukan
            if person_name not in embeddings:
                embeddings[person_name] = []
                
            if person_name not in metadata:
                metadata[person_name] = {}
            
            # Dapatkan hash saat ini
            image_hash = get_image_hash(image_path)
            
            # Hapus embeddings yang ada untuk gambar ini jika ada
            if filename in metadata[person_name] and "embedding_indices" in metadata[person_name][filename]:
                # Dapatkan indeks embeddings yang ada
                old_indices = metadata[person_name][filename]["embedding_indices"]
                
                # Nanti kita perlu memetakan ulang semua indeks untuk memperhitungkan embeddings yang dihapus
                # Untuk sekarang, tandai saja sebagai None untuk disaring nanti
                for idx in old_indices:
                    if idx < len(embeddings[person_name]):
                        embeddings[person_name][idx] = None
                
                print(f"  Menghapus {len(old_indices)} embeddings yang ada")
            
            # Proses gambar dengan batas kepercayaan baru
            image = preprocess_image(image_path)
            faces = detect_faces(image, confidence_threshold)
            
            # Hasilkan embeddings untuk wajah yang terdeteksi
            valid_embeddings = []
            for face in faces:
                embedding = embedder.embeddings([face])[0]
                valid_embeddings.append(embedding)
            
            # Simpan indeks di mana embeddings baru akan ditempatkan
            new_indices = []
            
            # Tambahkan embeddings baru (menambahkan ke akhir)
            for embedding in valid_embeddings:
                embeddings[person_name].append(embedding)
                new_indices.append(len(embeddings[person_name]) - 1)
                
            # Perbarui metadata
            metadata[person_name][filename] = {
                "hash": image_hash,
                "embedding_indices": new_indices,
                "faces_detected": len(faces),
                "reprocessed_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "confidence": confidence_threshold
            }
            
            print(f"  Berhasil: Terdeteksi {len(faces)} wajah dengan batas kepercayaan {confidence_threshold}")
            reprocessed_count += 1
            success_count += 1
            
        except Exception as e:
            print(f"  Error saat memproses {image_path}: {e}")
            
            # Catat error dalam metadata
            metadata[person_name][filename] = {
                "hash": image_hash,
                "embedding_indices": [],
                "faces_detected": 0,
                "error": str(e),
                "reprocessed_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "confidence": confidence_threshold
            }
            
            error_count += 1
    
    # Bersihkan embeddings dengan menghapus nilai None dan memperbarui indeks
    print("\nMembersihkan embeddings...")
    new_embeddings = {}
    new_metadata = {}
    
    for person_name, person_embeddings in embeddings.items():
        # Buat pemetaan dari indeks lama ke indeks baru
        old_to_new = {}
        new_idx = 0
        
        # Filter nilai None dan bangun pemetaan indeks
        filtered_embeddings = []
        for old_idx, emb in enumerate(person_embeddings):
            if emb is not None:
                filtered_embeddings.append(emb)
                old_to_new[old_idx] = new_idx
                new_idx += 1
        
        # Simpan embeddings yang difilter
        new_embeddings[person_name] = filtered_embeddings
        
        # Perbarui metadata dengan indeks baru
        new_metadata[person_name] = {}
        for filename, file_metadata in metadata[person_name].items():
            new_file_metadata = file_metadata.copy()
            
            if "embedding_indices" in file_metadata:
                # Petakan indeks lama ke indeks baru
                new_indices = []
                for idx in file_metadata["embedding_indices"]:
                    if idx in old_to_new:
                        new_indices.append(old_to_new[idx])
                
                new_file_metadata["embedding_indices"] = new_indices
            
            new_metadata[person_name][filename] = new_file_metadata
    
    # Simpan embeddings dan metadata yang diperbarui langsung tanpa membuat cadangan
    print("\nMenyimpan data yang diperbarui...")
    
    # Simpan file yang diperbarui
    with open(embeddings_path, "wb") as f:
        pickle.dump(new_embeddings, f)
    
    with open(metadata_path, "wb") as f:
        pickle.dump(new_metadata, f)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Cetak ringkasan
    print("\n" + "=" * 60)
    print("RINGKASAN PEMROSESAN ULANG")
    print("=" * 60)
    print(f"Total gambar bermasalah teridentifikasi: {len(problem_images)}")
    print(f"Gambar berhasil diproses ulang: {success_count}")
    print(f"Error selama pemrosesan ulang: {error_count}")
    print(f"Waktu pemrosesan: {processing_time:.2f} detik")
    print("\nFile yang diperbarui disimpan:")
    print(f"  {embeddings_path}")
    print(f"  {metadata_path}")
    print("=" * 60)
    
    # Kembalikan statistik
    return {
        "status": "success",
        "message": "Pemrosesan ulang selesai",
        "total_problematic_files": len(problem_images)
    }

# Jika dijalankan sebagai skrip, gunakan argumen baris perintah
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Memproses ulang gambar wajah bermasalah dalam database")
    
    parser.add_argument("--database", type=str, default="database",
                       help="Direktori yang berisi subdirektori orang dengan gambar wajah")
    parser.add_argument("--embeddings", type=str, default="face_embeddings.pkl",
                       help="Path ke file pickle embeddings wajah")
    parser.add_argument("--metadata", type=str, default="face_embeddings_metadata.pkl",
                       help="Path ke file pickle metadata wajah")
    parser.add_argument("--confidence", type=float, default=0.6,
                       help="Batas kepercayaan untuk deteksi wajah (rentang: 0.3-0.7)")
    
    args = parser.parse_args()
    
    # Validasi batas kepercayaan
    if not 0.3 <= args.confidence <= 0.7:
        print(f"Peringatan: Batas kepercayaan {args.confidence} di luar rentang yang direkomendasikan (0.3-0.7)")
        print("Untuk hasil terbaik pada pemrosesan ulang, nilai lebih rendah (0.3-0.5) akan mendeteksi lebih banyak wajah")
        print("dan nilai lebih tinggi (0.5-0.7) akan mendeteksi wajah dengan kualitas lebih tinggi tetapi mungkin melewatkan beberapa")
    
    # Jalankan pemrosesan ulang
    result = reprocess_problem_faces(
        database_dir=args.database,
        embeddings_path=args.embeddings,
        metadata_path=args.metadata,
        confidence_threshold=args.confidence
    )
    
    print(f"Status: {result['status']}")
    print(f"Pesan: {result['message']}")
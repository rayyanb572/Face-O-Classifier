import os
import numpy as np
import pickle
from keras_facenet import FaceNet
from ultralytics import YOLO
from PIL import Image
import cv2
import time
from datetime import datetime
import hashlib

# Fungsi utama untuk memperbarui embeddings wajah dari database foto
def update_face_embeddings(database_dir="database", output_path="face_embeddings.pkl", 
                          metadata_path="face_embeddings_metadata.pkl", confidence_threshold=0.6):
 
    print(f"Starting update process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    
    # Inisialisasi model YOLO untuk deteksi wajah dan FaceNet untuk embeddings
    yolo_model = YOLO("yolov8m-face.pt")
    embedder = FaceNet()
    
    # Memuat embeddings dan metadata yang sudah ada jika tersedia
    existing_embeddings = {}
    image_metadata = {}
    
    if os.path.exists(output_path) and os.path.exists(metadata_path):
        try:
            with open(output_path, "rb") as f:
                existing_embeddings = pickle.load(f)
            with open(metadata_path, "rb") as f:
                image_metadata = pickle.load(f)
            print(f"Loaded existing embeddings with {len(existing_embeddings)} persons")
        except Exception as e:
            print(f"Error loading existing data: {e}")
            print("Starting with empty dictionaries")
    
    # Inisialisasi dictionary baru untuk menyimpan data yang telah diperbarui
    updated_embeddings = {}
    updated_metadata = {}
    
    # Fungsi untuk menghitung hash MD5 dari file gambar
    def get_image_hash(image_path):
        with open(image_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    
    # Fungsi untuk memproses gambar menjadi format yang siap untuk deteksi wajah
    def preprocess_image(image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
            return image
        except Exception as e:
            raise ValueError(f"Error loading image {image_path}: {e}")
    
    # Fungsi untuk mendeteksi wajah dengan ambang batas kepercayaan
    def detect_faces(image, conf_threshold=0.6):
        results = yolo_model(image)
        
        # Mendapatkan koordinat bounding box dan skor kepercayaan
        detections = results[0].boxes.xyxy.numpy()  # koordinat bounding box
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
                print(f"    Ignoring face detection with low confidence: {conf:.2f} < {conf_threshold}")
                
        return faces
    
    # Statistik
    new_images_processed = 0
    retained_embeddings = 0
    
    # Melacak embeddings mana dari data yang ada yang sedang dipertahankan
    retained_embedding_tracker = {person: set() for person in existing_embeddings}
    
    # Memindai struktur database saat ini
    current_db_files = {}
    for person_name in os.listdir(database_dir):
        person_path = os.path.join(database_dir, person_name)
        
        if not os.path.isdir(person_path):
            continue
            
        # Mendapatkan semua file gambar yang valid dalam direktori
        image_files = [f for f in os.listdir(person_path) 
                     if f.lower().endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG'))]
        
        if person_name not in current_db_files:
            current_db_files[person_name] = []
            
        # Menambahkan path lengkap ke dictionary
        for img_file in image_files:
            full_path = os.path.join(person_path, img_file)
            current_db_files[person_name].append(full_path)
    
    # Memproses direktori setiap orang
    for person_name, image_paths in current_db_files.items():
        print(f"\nProcessing person: {person_name}")
        print(f"Using confidence threshold: {confidence_threshold}")
        
        # Inisialisasi orang dalam dictionary yang diperbarui
        if person_name not in updated_embeddings:
            updated_embeddings[person_name] = []
            updated_metadata[person_name] = {}
        
        # Memproses setiap gambar
        for image_path in image_paths:
            image_filename = os.path.basename(image_path)
            image_hash = get_image_hash(image_path)
            
            # Cek apakah kita sudah memproses gambar yang sama persis ini
            if (person_name in image_metadata and 
                image_filename in image_metadata[person_name] and 
                image_metadata[person_name][image_filename]["hash"] == image_hash):
                
                # Gambar sudah diproses, ambil embeddings
                embedding_indices = image_metadata[person_name][image_filename]["embedding_indices"]
                
                # Melacak embeddings asli mana yang kita pertahankan
                if person_name in retained_embedding_tracker:
                    for idx in embedding_indices:
                        retained_embedding_tracker[person_name].add(idx)
                
                # Menyalin embeddings yang ada ke dictionary yang diperbarui
                for idx in embedding_indices:
                    if idx < len(existing_embeddings[person_name]):
                        updated_embeddings[person_name].append(existing_embeddings[person_name][idx])
                        retained_embeddings += 1
                
                # Menyalin metadata tambahan dari yang asli jika ada
                additional_metadata = {}
                if (person_name in image_metadata and 
                    image_filename in image_metadata[person_name] and
                    isinstance(image_metadata[person_name][image_filename], dict)):
                    # Menyalin metadata tambahan seperti faces_detected (jika ada)
                    for key, value in image_metadata[person_name][image_filename].items():
                        if key not in ["hash", "embedding_indices"]:
                            additional_metadata[key] = value

                # Memperbarui metadata
                new_indices = list(range(
                    len(updated_embeddings[person_name]) - len(embedding_indices),
                    len(updated_embeddings[person_name])
                ))
                
                # Menggabungkan metadata dasar dengan tambahan
                metadata_entry = {
                    "hash": image_hash,
                    "embedding_indices": new_indices
                }
                metadata_entry.update(additional_metadata)  # Menambahkan metadata tambahan jika ada
                updated_metadata[person_name][image_filename] = metadata_entry
                
                print(f"  Reusing embeddings for: {image_filename}")
                
            else:
                # Gambar baru atau yang dimodifikasi, proses itu
                try:
                    # Proses dan deteksi wajah dengan ambang batas kepercayaan
                    image = preprocess_image(image_path)
                    faces = detect_faces(image, conf_threshold=confidence_threshold)
                    
                    # Mencatat indeks awal untuk embeddings baru ini
                    start_idx = len(updated_embeddings[person_name])
                    
                    # Menghasilkan embeddings untuk setiap wajah yang terdeteksi
                    for face in faces:
                        embedding = embedder.embeddings([face])[0]
                        updated_embeddings[person_name].append(embedding)
                        new_images_processed += 1
                    
                    # Memperbarui metadata
                    end_idx = len(updated_embeddings[person_name])
                    new_indices = list(range(start_idx, end_idx))
                    updated_metadata[person_name][image_filename] = {
                        "hash": image_hash,
                        "embedding_indices": new_indices,
                        "faces_detected": len(faces)  # Menambahkan info jumlah wajah
                    }
                    
                    print(f"  Processed: {image_filename} - Found {len(faces)} faces")
                    
                except Exception as e:
                    print(f"  Error processing {image_path}: {e}")
                    # Mencatat error dalam metadata
                    updated_metadata[person_name][image_filename] = {
                        "hash": image_hash,
                        "embedding_indices": [],
                        "faces_detected": 0,
                        "error": str(e)
                    }
    
    # Menghitung berapa banyak embeddings yang dihapus
    deleted_embeddings = 0
    for person_name in existing_embeddings:
        # Menghitung total embeddings asli untuk orang ini
        original_count = len(existing_embeddings[person_name])
        
        # Menghitung embeddings yang dipertahankan untuk orang ini
        if person_name in retained_embedding_tracker:
            retained_count = len(retained_embedding_tracker[person_name])
        else:
            retained_count = 0
            
        # Perbedaannya adalah jumlah embeddings yang dihapus untuk orang ini
        person_deleted = original_count - retained_count
        deleted_embeddings += person_deleted
    
    # Menyimpan embeddings dan metadata yang diperbarui
    with open(output_path, "wb") as f:
        pickle.dump(updated_embeddings, f)
    
    with open(metadata_path, "wb") as f:
        pickle.dump(updated_metadata, f)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Mencetak ringkasan
    print("\n" + "=" * 50)
    print("UPDATE SUMMARY")
    print("=" * 50)
    print(f"Total persons in database: {len(updated_embeddings)}")
    print(f"New embeddings processed: {new_images_processed}")
    print(f"Retained embeddings: {retained_embeddings}")
    print(f"Removed embeddings: {deleted_embeddings}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print("=" * 50)
    print(f"Updated embeddings saved to {output_path}")
    print(f"Metadata saved to {metadata_path}")
    
    # Mengembalikan statistik
    return {
        "total_persons": len(updated_embeddings),
        "new_embeddings": new_images_processed,
        "retained_embeddings": retained_embeddings,
        "removed_embeddings": deleted_embeddings
    }

if __name__ == "__main__":

    DATABASE_DIR = "database"
    OUTPUT_PATH = "face_embeddings.pkl"
    METADATA_PATH = "face_embeddings_metadata.pkl"
    
    # Mendapatkan ambang batas kepercayaan dari variabel lingkungan
    # Jika tidak tersedia, default ke 0.6
    try:
        CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.6))
        # Validasi nilai ambang batas
        if not 0 <= CONFIDENCE_THRESHOLD <= 1:
            print(f"Warning: Invalid confidence threshold value: {CONFIDENCE_THRESHOLD}. Using default: 0.6")
            CONFIDENCE_THRESHOLD = 0.6
    except (ValueError, TypeError):
        print("Warning: Could not parse confidence threshold from environment. Using default: 0.6")
        CONFIDENCE_THRESHOLD = 0.6
    
    print(f"Using confidence threshold: {CONFIDENCE_THRESHOLD}")
    update_face_embeddings(DATABASE_DIR, OUTPUT_PATH, METADATA_PATH, CONFIDENCE_THRESHOLD)
import os
import shutil
from tqdm import tqdm
from keras_facenet import FaceNet
from ultralytics import YOLO
import pickle
import cv2
import utils
import threading
import time
from datetime import datetime

# Global flag untuk mengontrol proses
processing_cancelled = threading.Event()

# Inisialisasi model (hanya sekali)
yolo_model = YOLO("yolov8m-face.pt")
embedder = FaceNet()

# Variabel untuk menyimpan waktu terakhir kali embedding dimuat
last_embedding_load_time = 0

# Fungsi untuk memuat embedding dengan cek waktu modifikasi file
def load_embeddings():
    global last_embedding_load_time
    
    embedding_file = "face_embeddings.pkl"
    
    if os.path.exists(embedding_file):
        # Periksa kapan terakhir kali file dimodifikasi
        current_mod_time = os.path.getmtime(embedding_file)
        
        # Jika file telah diubah sejak terakhir kali dimuat atau belum pernah dimuat
        if current_mod_time > last_embedding_load_time:
            print(f"Loading updated embeddings from {embedding_file}")
            with open(embedding_file, "rb") as f:
                known_embeddings = pickle.load(f)
            
            # Perbarui waktu muat terakhir
            last_embedding_load_time = current_mod_time
            return known_embeddings
        else:
            # File tidak berubah, coba muat dari cache
            with open(embedding_file, "rb") as f:
                return pickle.load(f)
    else:
        # File tidak ada, kembalikan dictionary kosong
        print(f"Warning: Embedding file {embedding_file} tidak ditemukan!")
        return {}

def cancel_processing():
    """Fungsi untuk membatalkan proses klasifikasi"""
    processing_cancelled.set()
    return True

def reset_cancel_flag():
    """Reset flag pembatalan"""
    processing_cancelled.clear()

# Fungsi utama untuk mengklasifikasikan wajah dari folder input
def classify_faces(input_folder, output_folder=None, confidence_threshold=0.6, batch_size=5):
    # Reset flag pembatalan setiap kali memulai klasifikasi baru
    reset_cancel_flag()
    
    # Catat waktu mulai proses
    print(f"Starting classification process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    
    try:
        # Muat embedding paling baru (dengan deteksi perubahan file otomatis)
        known_embeddings = load_embeddings()
        
        # Gunakan nama folder input sebagai default output folder jika tidak diberikan
        if output_folder is None:
            output_folder = "(Classified) " + os.path.basename(input_folder)
        
        unknown_folder = os.path.join(output_folder, "UNKNOWN")
        visualized_folder = os.path.join(output_folder, "VISUALIZED")
        labels_folder = os.path.join(output_folder, "labels")

        # Membuat folder yang diperlukan
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(unknown_folder, exist_ok=True)
        os.makedirs(visualized_folder, exist_ok=True)
        os.makedirs(labels_folder, exist_ok=True)
        
        # Bersihkan folder untuk memastikan tidak ada file sisa
        utils.clear_folder(unknown_folder)
        utils.clear_folder(visualized_folder)
        utils.clear_folder(labels_folder)

        # Ambil semua file gambar dari folder input
        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        batch_range = range(0, len(image_files), batch_size)

        # Proses gambar dalam batch
        for i in tqdm(batch_range, desc="🔄 Processing Batches"):
            # Cek apakah proses dibatalkan
            if processing_cancelled.is_set():
                # Bersihkan folder output jika proses dibatalkan
                if os.path.exists(output_folder):
                    shutil.rmtree(output_folder)
                return None, None, None
                
            batch_files = image_files[i:i+batch_size]
            batch_images = []
            original_images = []
            paths = []

            # Membaca gambar batch
            for image_name in batch_files:
                if processing_cancelled.is_set():
                    break
                    
                image_path = os.path.join(input_folder, image_name)
                image = cv2.imread(image_path)
                if image is None:
                    continue
                batch_images.append(image)
                original_images.append(image.copy())
                paths.append(image_path)

            if not batch_images or processing_cancelled.is_set():
                continue

            # Deteksi wajah dengan YOLO
            results = yolo_model.predict(batch_images)

            # Proses hasil deteksi satu per satu
            for j in tqdm(range(len(results)), desc=f"🔍 Batch {i//batch_size + 1}", leave=False):
                if processing_cancelled.is_set():
                    break
                    
                result = results[j]
                image = batch_images[j]
                original_image = original_images[j]
                image_name = os.path.basename(paths[j])
                image_path = paths[j]

                if not result or not result.boxes:
                    # Tidak ada wajah terdeteksi, skip gambar ini
                    continue

                bboxes = []
                confidences = []
                face_crops = []
                coords = []
                identified_faces = False

                # Proses setiap bounding box hasil deteksi
                for box in result.boxes:
                    if processing_cancelled.is_set():
                        break
                        
                    bbox = box.xyxy.cpu().numpy()[0]
                    conf = box.conf.cpu().numpy()[0]

                    if conf < confidence_threshold:
                        continue

                    face = utils.crop_face(image, bbox)
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face_resized = cv2.resize(face_rgb, (160, 160))
                    face_crops.append(face_resized)
                    bboxes.append(bbox)
                    confidences.append(conf)
                    coords.append(bbox)

                if processing_cancelled.is_set():
                    continue

                # Kalau ada wajah yang terdeteksi dan lolos threshold, lakukan ekstraksi embedding
                if face_crops:
                    embeddings = embedder.embeddings(face_crops)

                    for k, face_embedding in enumerate(embeddings):
                        if processing_cancelled.is_set():
                            break
                            
                        bbox = coords[k]
                        match = utils.find_match(face_embedding, known_embeddings)

                        if match:
                            label = match
                            person_folder = os.path.join(output_folder, match)
                            identified_faces = True
                        else:
                            label = "Unknown"
                            person_folder = unknown_folder

                        os.makedirs(person_folder, exist_ok=True)

                        if match:
                            shutil.copy(image_path, person_folder)

                        # Gambar bounding box di gambar asli
                        utils.draw_bounding_box(original_image, bbox, label)

                    # Simpan gambar hasil visualisasi jika ada wajah yang teridentifikasi
                    if identified_faces:
                        visualized_path = os.path.join(visualized_folder, image_name)
                        cv2.imwrite(visualized_path, original_image)
                    else:
                        shutil.copy(image_path, unknown_folder)

                    # Simpan anotasi bounding box ke format YOLO
                    utils.save_yolo_annotation(labels_folder, image_name, image.shape, bboxes, confidences)

        if processing_cancelled.is_set():
            # Bersihkan folder output jika proses dibatalkan
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            return None, None, None

        # Setelah selesai, buat file ZIP dari output folder
        zip_output_folder = 'zip'
        os.makedirs(zip_output_folder, exist_ok=True)
        
        output_folder_basename = os.path.basename(output_folder)
        zip_output_path = os.path.join(zip_output_folder, output_folder_basename + ".zip")
        
        utils.zip_folder(output_folder, zip_output_path)

        # Hitung total waktu pemrosesan
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Print summary dengan waktu pemrosesan
        print("\n" + "=" * 50)
        print("CLASSIFICATION SUMMARY")
        print("=" * 50)
        print(f"Input folder: {input_folder}")
        print(f"Output folder: {output_folder}")
        print(f"Total images processed: {len(image_files)}")
        print(f"Processing time: {processing_time:.2f} seconds")
        print("=" * 50)
        print(f"Classification completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return output_folder, zip_output_path, processing_time
        
    except Exception as e:
        # Hitung waktu meskipun terjadi error
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"Error during processing: {str(e)}")
        print(f"Processing time before error: {processing_time:.2f} seconds")
        
        # Bersihkan folder output jika terjadi error
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        return None, None, None
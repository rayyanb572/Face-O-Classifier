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

# Global flag untuk mengontrol proses
processing_cancelled = threading.Event()

# Load model YOLO untuk deteksi wajah dan FaceNet untuk embedding wajah
yolo_model = YOLO("yolov8m-face.pt")
embedder = FaceNet()

# Load database embedding wajah yang sudah dikenal
with open("face_embeddings.pkl", "rb") as f:
    known_embeddings = pickle.load(f)

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
    
    try:
        # Gunakan nama folder input sebagai default output folder jika tidak diberikan
        if output_folder is None:
            output_folder = "(Classified) " + os.path.basename(input_folder)
        
        unknown_folder = os.path.join(output_folder, "UNKNOWN")
        visualized_folder = os.path.join(output_folder, "VISUALIZED")
        labels_folder = os.path.join(output_folder, "labels")
        low_conf_folder = os.path.join(output_folder, "low_confidence")

        # Membuat folder yang diperlukan
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(unknown_folder, exist_ok=True)
        os.makedirs(visualized_folder, exist_ok=True)
        os.makedirs(labels_folder, exist_ok=True)
        os.makedirs(low_conf_folder, exist_ok=True)
        
        # Bersihkan folder untuk memastikan tidak ada file sisa
        utils.clear_folder(unknown_folder)
        utils.clear_folder(visualized_folder)
        utils.clear_folder(labels_folder)
        utils.clear_folder(low_conf_folder)

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
                return None, None
                
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
                    continue

                bboxes = []
                confidences = []
                face_crops = []
                coords = []
                passed_threshold = False
                identified_faces = False

                # Proses setiap bounding box hasil deteksi
                for box in result.boxes:
                    if processing_cancelled.is_set():
                        break
                        
                    bbox = box.xyxy.cpu().numpy()[0]
                    conf = box.conf.cpu().numpy()[0]

                    if conf < confidence_threshold:
                        continue

                    passed_threshold = True
                    face = utils.crop_face(image, bbox)
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face_resized = cv2.resize(face_rgb, (160, 160))
                    face_crops.append(face_resized)
                    bboxes.append(bbox)
                    confidences.append(conf)
                    coords.append(bbox)

                if processing_cancelled.is_set():
                    continue

                # Kalau tidak ada deteksi yang lolos threshold, simpan ke folder low_confidence
                if not passed_threshold:
                    shutil.copy(image_path, low_conf_folder)
                    continue

                # Kalau ada wajah yang terdeteksi, lakukan ekstraksi embedding
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
                else:
                    shutil.copy(image_path, low_conf_folder)

        if processing_cancelled.is_set():
            # Bersihkan folder output jika proses dibatalkan
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            return None, None

        # Setelah selesai, buat file ZIP dari output folder
        zip_output_folder = 'zip'
        os.makedirs(zip_output_folder, exist_ok=True)
        
        output_folder_basename = os.path.basename(output_folder)
        zip_output_path = os.path.join(zip_output_folder, output_folder_basename + ".zip")
        
        utils.zip_folder(output_folder, zip_output_path)

        return output_folder, zip_output_path
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        # Bersihkan folder output jika terjadi error
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        return None, None

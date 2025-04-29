import os
import numpy as np
import pickle
from keras_facenet import FaceNet
from ultralytics import YOLO
from PIL import Image
import cv2

#hapus file embeddings sebelumnya
output_path = "face_embeddings.pkl"
if os.path.exists(output_path):
    os.remove(output_path)
    print(f"File {output_path} telah dihapus untuk membuat yang baru.")

#inisialisasi model
yolo_model = YOLO("yolov8m-face.pt")
embedder = FaceNet()

#fungsi membaca dan memproses gambar ke RGB
def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        return image
    except Exception as e:
        raise ValueError(f"Error loading image {image_path}: {e}")

#fungsi mendeteksi wajah menggunakan YOLOv8
def detect_faces(image):
    results = yolo_model(image) 
    detections = results[0].boxes.xyxy.numpy()  #bounding box (x1, y1, x2, y2)
    faces = []
    for box in detections:
        x1, y1, x2, y2 = map(int, box)
        face = image[y1:y2, x1:x2]  #crop wajah dari gambar
        face = cv2.resize(face, (160, 160)) 
        faces.append(face)
    return faces

#path direktori wajah yang sudah dikenal
known_faces_dir = "database"
embeddings = {}

#proses setiap subfolder
print("Starting embedding generation...")
for person_name in os.listdir(known_faces_dir):
    person_path = os.path.join(known_faces_dir, person_name)
    if os.path.isdir(person_path):
        embeddings[person_name] = []
        print(f"Processing folder: {person_name}")
        for image_name in os.listdir(person_path):
            if image_name.lower().endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG')):
                image_path = os.path.join(person_path, image_name)
                try:
                    #preprocess gambar
                    image = preprocess_image(image_path)

                    #deteksi wajah dengan YOLOv8
                    faces = detect_faces(image)

                    #proses setiap wajah yang terdeteksi
                    for face in faces:
                        embedding = embedder.embeddings([face])[0]
                        embeddings[person_name].append(embedding)

                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

#simpan embeddings ke file .pkl
with open(output_path, "wb") as f:
    pickle.dump(embeddings, f)

print(f"Database embedding telah dibuat dan disimpan di {output_path}")

import os
import shutil
import pickle
import numpy as np
import cv2
from keras_facenet import FaceNet
from ultralytics import YOLO

# Load YOLO model dan FaceNet
yolo_model = YOLO("yolov8n-face.pt")
embedder = FaceNet()

# Load database embedding
with open("face_embeddings.pkl", "rb") as f:
    known_embeddings = pickle.load(f)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_match(embedding, threshold=0.8):
    best_match, best_score = None, threshold
    for person_name, embeddings_list in known_embeddings.items():
        for person_embedding in embeddings_list:
            score = cosine_similarity(embedding, person_embedding)
            if score > best_score:
                best_match, best_score = person_name, score
    return best_match

def crop_face(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    return image[y1:y2, x1:x2]

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)

def draw_bounding_box(image, bbox, label):
    x1, y1, x2, y2 = map(int, bbox)
    color = (0, 255, 0) if "Unknown" not in label else (0, 0, 255)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

def save_yolo_annotation(labels_folder, image_name, image_shape, bboxes, confidences):
    height, width, _ = image_shape
    annotation_path = os.path.join(labels_folder, f"{os.path.splitext(image_name)[0]}.txt")
    annotations = []
    
    for bbox, conf in zip(bboxes, confidences):
        x1, y1, x2, y2 = bbox
        x_center = (x1 + x2) / 2 / width
        y_center = (y1 + y2) / 2 / height
        bbox_width = (x2 - x1) / width
        bbox_height = (y2 - y1) / height
        annotations.append(f"0 {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f} {conf:.4f}")
    
    with open(annotation_path, "w") as f:
        f.write("\n".join(annotations))

def classify_faces(input_folder, output_folder="output_test"):
    unknown_folder = os.path.join(output_folder, "unknown")
    visualized_folder = os.path.join(output_folder, "visualized")
    labels_folder = os.path.join(output_folder, "labels")
    
    clear_folder(output_folder)
    clear_folder(unknown_folder)
    clear_folder(visualized_folder)
    clear_folder(labels_folder)

    for image_name in os.listdir(input_folder):
        if image_name.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(input_folder, image_name)
            image = cv2.imread(image_path)
            original_image = image.copy()
            results = yolo_model.predict(image)
            
            if not results or not results[0].boxes:
                continue
            
            bboxes = []
            confidences = []
            
            for box in results[0].boxes:
                bbox = box.xyxy.numpy()[0]
                conf = box.conf.numpy()[0]  # Ambil confidence score dari YOLO
                bboxes.append(bbox)
                confidences.append(conf)
                
                face_image = crop_face(image, bbox)
                face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                face_embedding = embedder.embeddings([face_image_rgb])[0]
                match = find_match(face_embedding)
                
                if match:
                    label = match  # Hapus confidence score dari label
                    person_folder = os.path.join(output_folder, match)
                else:
                    label = "Unknown"
                    person_folder = unknown_folder
                
                os.makedirs(person_folder, exist_ok=True)
                shutil.copy(image_path, person_folder)
                draw_bounding_box(original_image, bbox, label)
            
            save_yolo_annotation(labels_folder, image_name, image.shape, bboxes, confidences)
            
            visualized_path = os.path.join(visualized_folder, image_name)
            cv2.imwrite(visualized_path, original_image)
    
    return output_folder

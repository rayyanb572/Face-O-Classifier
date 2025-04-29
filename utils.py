import os
import shutil
import zipfile
import numpy as np
import cv2

# Menghitung kemiripan kosinus antara dua vektor
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Mencari kecocokan embedding wajah dengan database known_embeddings
def find_match(embedding, known_embeddings, threshold=0.8):
    best_match, best_score = None, threshold
    for person_name, embeddings_list in known_embeddings.items():
        for person_embedding in embeddings_list:
            score = cosine_similarity(embedding, person_embedding)
            if score > best_score:
                best_match, best_score = person_name, score
    return best_match

# Memotong gambar wajah berdasarkan bounding box
def crop_face(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    return image[y1:y2, x1:x2]

# Menghapus folder jika ada lalu membuat folder kosong baru
def clear_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)

# Menggambar kotak pembatas dan label pada gambar
def draw_bounding_box(image, bbox, label):
    if label == "Unknown":
        return
    x1, y1, x2, y2 = map(int, bbox)
    color = (0, 255, 0)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

# Menyimpan anotasi YOLO dari bounding box dan confidence ke file teks
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

# Membuat file ZIP dari seluruh isi folder
def zip_folder(input_folder, output_path):
    folder_name = os.path.basename(input_folder)
    
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, os.path.dirname(input_folder))
                zipf.write(file_path, relative_path)

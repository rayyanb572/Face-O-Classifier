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

def update_face_embeddings(database_dir="database", output_path="face_embeddings.pkl", 
                          metadata_path="face_embeddings_metadata.pkl", confidence_threshold=0.6):
    """
    Update face embeddings by processing new images and removing deleted ones.
    
    Args:
        database_dir: Directory containing subdirectories of person faces
        output_path: Path to save the updated embeddings pickle file
        metadata_path: Path to save metadata about processed images
        confidence_threshold: Minimum confidence score (0.0-1.0) for YOLO face detection
    """
    print(f"Starting update process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    
    # Initialize models
    yolo_model = YOLO("yolov8m-face.pt")
    embedder = FaceNet()
    
    # Load existing embeddings and metadata if available
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
    
    # Initialize new dictionaries to store updated data
    updated_embeddings = {}
    updated_metadata = {}
    
    # Function to get image hash
    def get_image_hash(image_path):
        with open(image_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    
    # Function to preprocess image
    def preprocess_image(image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
            return image
        except Exception as e:
            raise ValueError(f"Error loading image {image_path}: {e}")
    
    # Function to detect faces with confidence threshold
    def detect_faces(image, conf_threshold=0.6):
        results = yolo_model(image)
        
        # Get bounding boxes and confidence scores
        detections = results[0].boxes.xyxy.numpy()  # bounding box coordinates
        confidences = results[0].boxes.conf.numpy()  # confidence scores
        
        faces = []
        # Loop through detections and filter by confidence score
        for i, (box, conf) in enumerate(zip(detections, confidences)):
            if conf >= conf_threshold:
                x1, y1, x2, y2 = map(int, box)
                face = image[y1:y2, x1:x2]  # crop face
                face = cv2.resize(face, (160, 160))
                faces.append(face)
            else:
                print(f"    Ignoring face detection with low confidence: {conf:.2f} < {conf_threshold}")
                
        return faces
    
    # Statistics
    new_images_processed = 0
    retained_embeddings = 0
    
    # Track which embeddings from the existing data are being retained
    retained_embedding_tracker = {person: set() for person in existing_embeddings}
    
    # Scan current database structure
    current_db_files = {}
    for person_name in os.listdir(database_dir):
        person_path = os.path.join(database_dir, person_name)
        
        if not os.path.isdir(person_path):
            continue
            
        # Get all valid image files in the directory
        image_files = [f for f in os.listdir(person_path) 
                     if f.lower().endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG'))]
        
        if person_name not in current_db_files:
            current_db_files[person_name] = []
            
        # Add full paths to the dictionary
        for img_file in image_files:
            full_path = os.path.join(person_path, img_file)
            current_db_files[person_name].append(full_path)
    
    # Process each person's directory
    for person_name, image_paths in current_db_files.items():
        print(f"\nProcessing person: {person_name}")
        print(f"Using confidence threshold: {confidence_threshold}")
        
        # Initialize person in updated dictionaries
        if person_name not in updated_embeddings:
            updated_embeddings[person_name] = []
            updated_metadata[person_name] = {}
        
        # Process each image
        for image_path in image_paths:
            image_filename = os.path.basename(image_path)
            image_hash = get_image_hash(image_path)
            
            # Check if we've already processed this exact image
            if (person_name in image_metadata and 
                image_filename in image_metadata[person_name] and 
                image_metadata[person_name][image_filename]["hash"] == image_hash):
                
                # Image already processed, retrieve embeddings
                embedding_indices = image_metadata[person_name][image_filename]["embedding_indices"]
                
                # Track which original embeddings we're keeping
                if person_name in retained_embedding_tracker:
                    for idx in embedding_indices:
                        retained_embedding_tracker[person_name].add(idx)
                
                # Copy existing embeddings to updated dict
                for idx in embedding_indices:
                    if idx < len(existing_embeddings[person_name]):
                        updated_embeddings[person_name].append(existing_embeddings[person_name][idx])
                        retained_embeddings += 1
                
                # Salin metadata tambahan dari yang asli jika ada
                additional_metadata = {}
                if (person_name in image_metadata and 
                    image_filename in image_metadata[person_name] and
                    isinstance(image_metadata[person_name][image_filename], dict)):
                    # Salin metadata tambahan seperti faces_detected (jika ada)
                    for key, value in image_metadata[person_name][image_filename].items():
                        if key not in ["hash", "embedding_indices"]:
                            additional_metadata[key] = value

                # Update metadata
                new_indices = list(range(
                    len(updated_embeddings[person_name]) - len(embedding_indices),
                    len(updated_embeddings[person_name])
                ))
                
                # Gabungkan metadata dasar dengan tambahan
                metadata_entry = {
                    "hash": image_hash,
                    "embedding_indices": new_indices
                }
                metadata_entry.update(additional_metadata)  # Tambahkan metadata tambahan jika ada
                updated_metadata[person_name][image_filename] = metadata_entry
                
                print(f"  Reusing embeddings for: {image_filename}")
                
            else:
                # New or modified image, process it
                try:
                    # Preprocess and detect faces with confidence threshold
                    image = preprocess_image(image_path)
                    faces = detect_faces(image, conf_threshold=confidence_threshold)
                    
                    # Record the starting index for these new embeddings
                    start_idx = len(updated_embeddings[person_name])
                    
                    # Generate embeddings for each detected face
                    for face in faces:
                        embedding = embedder.embeddings([face])[0]
                        updated_embeddings[person_name].append(embedding)
                        new_images_processed += 1
                    
                    # Update metadata
                    end_idx = len(updated_embeddings[person_name])
                    new_indices = list(range(start_idx, end_idx))
                    updated_metadata[person_name][image_filename] = {
                        "hash": image_hash,
                        "embedding_indices": new_indices,
                        "faces_detected": len(faces)  # Tambahkan info jumlah wajah
                    }
                    
                    print(f"  Processed: {image_filename} - Found {len(faces)} faces")
                    
                except Exception as e:
                    print(f"  Error processing {image_path}: {e}")
                    # Catat error dalam metadata
                    updated_metadata[person_name][image_filename] = {
                        "hash": image_hash,
                        "embedding_indices": [],
                        "faces_detected": 0,
                        "error": str(e)
                    }
    
    # Calculate how many embeddings were deleted
    deleted_embeddings = 0
    for person_name in existing_embeddings:
        # Count total original embeddings for this person
        original_count = len(existing_embeddings[person_name])
        
        # Count retained embeddings for this person
        if person_name in retained_embedding_tracker:
            retained_count = len(retained_embedding_tracker[person_name])
        else:
            retained_count = 0
            
        # The difference is the number of deleted embeddings for this person
        person_deleted = original_count - retained_count
        deleted_embeddings += person_deleted
    
    # Save updated embeddings and metadata
    with open(output_path, "wb") as f:
        pickle.dump(updated_embeddings, f)
    
    with open(metadata_path, "wb") as f:
        pickle.dump(updated_metadata, f)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Print summary
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
    
    # Return statistics
    return {
        "total_persons": len(updated_embeddings),
        "new_embeddings": new_images_processed,
        "retained_embeddings": retained_embeddings,
        "removed_embeddings": deleted_embeddings,
        "processing_time": processing_time
    }

if __name__ == "__main__":
    # You can customize these parameters
    DATABASE_DIR = "database"
    OUTPUT_PATH = "face_embeddings.pkl"
    METADATA_PATH = "face_embeddings_metadata.pkl"
    
    # Get confidence threshold from environment variable
    # If not available, default to 0.6
    try:
        CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.6))
        # Validate threshold value
        if not 0 <= CONFIDENCE_THRESHOLD <= 1:
            print(f"Warning: Invalid confidence threshold value: {CONFIDENCE_THRESHOLD}. Using default: 0.6")
            CONFIDENCE_THRESHOLD = 0.6
    except (ValueError, TypeError):
        print("Warning: Could not parse confidence threshold from environment. Using default: 0.6")
        CONFIDENCE_THRESHOLD = 0.6
    
    print(f"Using confidence threshold: {CONFIDENCE_THRESHOLD}")
    update_face_embeddings(DATABASE_DIR, OUTPUT_PATH, METADATA_PATH, CONFIDENCE_THRESHOLD)
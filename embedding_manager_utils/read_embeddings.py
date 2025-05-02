import pickle
import os
import numpy as np
from collections import defaultdict

def audit_face_database(pickle_path="face_embeddings.pkl", 
                       metadata_path="face_embeddings_metadata.pkl", 
                       database_dir="database"):
    """
    Melakukan audit terhadap database embeddings wajah dan membandingkannya dengan
    file yang ada di folder database. Menggunakan metadata untuk menganalisis file
    dengan lebih dari satu embedding atau file tanpa embedding.
    
    Args:
        pickle_path: Path ke file pickle embeddings
        metadata_path: Path ke file pickle metadata
        database_dir: Path ke direktori database foto wajah
    """
    print("\n" + "=" * 60)
    print("AUDIT DATABASE FACE EMBEDDINGS")
    print("=" * 60)
    
    # 1. Periksa keberadaan file pickle dan metadata
    if not os.path.exists(pickle_path):
        print(f"ERROR: File {pickle_path} tidak ditemukan.")
        return
        
    if not os.path.exists(metadata_path):
        print(f"WARNING: File metadata {metadata_path} tidak ditemukan.")
        print("Audit akan dilakukan tanpa informasi metadata.")
        has_metadata = False
    else:
        has_metadata = True
    
    # 2. Periksa keberadaan direktori database
    if not os.path.exists(database_dir):
        print(f"ERROR: Direktori {database_dir} tidak ditemukan.")
        return
    
    # 3. Load embeddings dari file pickle
    print(f"Membaca file embeddings: {pickle_path}")
    with open(pickle_path, "rb") as f:
        embeddings = pickle.load(f)
    
    # 4. Load metadata jika tersedia
    if has_metadata:
        print(f"Membaca file metadata: {metadata_path}")
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
    else:
        metadata = {}
    
    # 5. Hitung statistik embeddings
    total_embeddings = 0
    embeddings_per_person = {}
    
    for person, vectors in embeddings.items():
        embeddings_per_person[person] = len(vectors)
        total_embeddings += len(vectors)
    
    # 6. Scan direktori database untuk menghitung file
    print(f"Memindai direktori database: {database_dir}")
    file_counts = defaultdict(int)
    file_list = defaultdict(list)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    
    total_files = 0
    for person_name in os.listdir(database_dir):
        person_dir = os.path.join(database_dir, person_name)
        if os.path.isdir(person_dir):
            for file_name in os.listdir(person_dir):
                if file_name.endswith(valid_extensions):
                    file_counts[person_name] += 1
                    file_list[person_name].append(file_name)
                    total_files += 1
    
    # 7. Tampilkan hasil perbandingan
    print("\n" + "=" * 60)
    print("HASIL AUDIT DATABASE")
    print("=" * 60)
    jumlah_orang = len(embeddings)
    print(f"Jumlah Orang dalam Database: {jumlah_orang}")
    print(f"Total embeddings: {total_embeddings}")
    print(f"Total file foto: {total_files}")
    
    difference = total_embeddings - total_files
    if difference > 0:
        print(f"Selisih: {difference} embedding LEBIH BANYAK dari jumlah file")
    elif difference < 0:
        print(f"Selisih: {abs(difference)} file LEBIH BANYAK dari jumlah embedding")
    else:
        print("Jumlah embedding dan file sama persis.")
    
    # 8. Mencari penyebab perbedaan (jika ada)
    print("\n" + "=" * 60)
    print("DETAIL PER ORANG")
    print("=" * 60)
    print(f"{'Nama':30} {'File Foto':10} {'Embeddings':10} {'Selisih':10} {'Status':5} {'Detail Masalah'}")
    print("-" * 100)
    
    persons_with_mismatch = []
    
    for person in sorted(set(list(embeddings_per_person.keys()) + list(file_counts.keys()))):
        emb_count = embeddings_per_person.get(person, 0)
        file_count = file_counts.get(person, 0)
        diff = emb_count - file_count
        
        # Deteksi file dengan masalah menggunakan metadata
        problem_files = []
        
        if has_metadata and person in metadata:
            person_metadata = metadata[person]
            
            # Cek file dengan multi-faces
            multi_face_files = [(filename, info["faces_detected"]) 
                               for filename, info in person_metadata.items() 
                               if info["faces_detected"] > 1]
            
            # Cek file tanpa wajah terdeteksi
            no_face_files = [filename for filename, info in person_metadata.items() 
                           if info["faces_detected"] == 0]
            
            # Tambahkan ke problem_files
            for filename, face_count in multi_face_files:
                problem_files.append(f"{filename} ({face_count} wajah)")
                
            for filename in no_face_files:
                problem_files.append(f"{filename} (tidak ada wajah)")
                
        # Cek file yang ada di folder tapi tidak ada di metadata
        if has_metadata and person in metadata and person in file_list:
            for filename in file_list[person]:
                if filename not in metadata[person]:
                    problem_files.append(f"{filename} (tidak diproses)")
        
        if diff != 0:
            status = "⚠️"
            persons_with_mismatch.append(person)
        else:
            status = "✓"
        
        # Batasi jumlah problem_files yang ditampilkan
        problem_files_str = ", ".join(problem_files[:5])
        if len(problem_files) > 5:
            problem_files_str += f", ... ({len(problem_files) - 5} file lainnya)"
            
        print(f"{person:30} {file_count:<10} {emb_count:<10} {diff:<10} {status:<5} {problem_files_str}")
    
    # 9. Analisis penyebab potensial
    print("\n" + "=" * 60)
    print("POTENSI PENYEBAB MASALAH")
    print("=" * 60)
    
    if difference == 0:
        print("Tidak ada perbedaan jumlah secara keseluruhan.")
    else:
        print(f"Terdapat perbedaan jumlah: {difference}")
        print("Kemungkinan penyebab masalah:")
        
        if has_metadata:
            # Hitung statistik dari metadata
            multi_face_count = 0
            multi_face_extra = 0
            no_face_count = 0
            
            for person, person_metadata in metadata.items():
                for filename, info in person_metadata.items():
                    if info["faces_detected"] > 1:
                        multi_face_count += 1
                        multi_face_extra += info["faces_detected"] - 1
                    elif info["faces_detected"] == 0:
                        no_face_count += 1
            
            print(f"1. {multi_face_count} file memiliki lebih dari satu wajah (total {multi_face_extra} wajah tambahan)")
            print(f"2. {no_face_count} file tidak terdeteksi wajah saat pemrosesan")
        else:
            print("1. Beberapa file memiliki lebih dari satu wajah")
            print("2. Beberapa file tidak terdeteksi wajah saat pemrosesan")
        
        print("3. Ada beberapa file foto yang sudah dihapus tapi embeddingnya masih ada")
        print("4. Ada duplikasi dalam proses pembuatan embeddings")
        print("5. Beberapa file belum diproses(crop) dengan baik untuk embeddings")
        print("6. Beberapa file gagal dibaca karena kesalahan format")
    
    # 10. Rekomendasi
    print("\n" + "=" * 60)
    print("REKOMENDASI")
    print("=" * 60)
    
    if len(persons_with_mismatch) > 0:
        print("Untuk mengatasi perbedaan antara jumlah file dan embeddings:")
        print("1. Proses ulang foto-foto yang bermasalah(crop) dan pastikan satu wajah dalam satu foto, lalu jalankan update")
        print("2. Jalankan fitur update yang dapat mendeteksi dan menangani file yang dihapus pada menu update")
        print("3. Jalankan fitur reproses foto bermasalah pada menu update dengan merubah nilai confidence score")
        print("4. Pertimbangkan untuk membuat ulang database embeddings jika perbedaan signifikan (Memakan Waktu)")
    else:
        print("Database embeddings dan direktori file sudah konsisten.")
    
    # 11. Deteksi potensi masalah lain
    print("\n" + "=" * 60)
    print("DETEKSI MASALAH LAIN")
    print("=" * 60)
    
    # Periksa apakah ada orang di pickle tapi tidak di folder
    missing_folders = [p for p in embeddings_per_person if p not in file_counts]
    if missing_folders:
        print("⚠️ Ditemukan embeddings untuk folder yang sudah tidak ada:")
        for p in missing_folders:
            print(f"   - {p} ({embeddings_per_person[p]} embeddings)")
    else:
        print("✓ Semua orang dalam database embeddings memiliki folder yang sesuai.")
    
    # Periksa apakah ada folder tapi tidak ada di pickle
    missing_in_pickle = [p for p in file_counts if p not in embeddings_per_person]
    if missing_in_pickle:
        print("\n⚠️ Ditemukan folder yang belum ada embeddingnya:")
        for p in missing_in_pickle:
            print(f"   - {p} ({file_counts[p]} files)")
    else:
        print("\n✓ Semua folder dalam direktori database memiliki embeddings yang sesuai.")

    print("\n" + "=" * 60)

# Jalankan fungsi jika file ini dieksekusi langsung
if __name__ == "__main__":
    PICKLE_PATH = "face_embeddings.pkl"
    METADATA_PATH = "face_embeddings_metadata.pkl"
    DATABASE_DIR = "database"
    
    audit_face_database(PICKLE_PATH, METADATA_PATH, DATABASE_DIR)
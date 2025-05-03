# Script untuk navigasi dan manajemen database face embeddings
import os
import sys
import subprocess
import time

# Path ke folder utilitas
UTILS_FOLDER = "embedding_manager_utils"
# Nilai default untuk confidence threshold
DEFAULT_CONFIDENCE = 0.6

def clear_screen():
    """Membersihkan layar terminal"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Menampilkan header aplikasi"""
    clear_screen()
    print("=" * 60)
    print("FACE DATABASE MANAGER".center(60))
    print("=" * 60)
    print("Manajemen database embedding wajah untuk klasifikasi foto".center(60))
    print("=" * 60)
    print()

def print_menu():
    """Menampilkan menu utama"""
    print("PILIHAN MENU:")
    print("1. (CHECK) Lihat Detail Database (read_embeddings.py)")
    print("2. (UPDATE) Update Face Embeddings")
    print("3. (REBUILD) Rebuild Database (rebuild_embeddings.py)")
    print("0. Keluar")
    print()

def print_update_submenu():
    """Menampilkan submenu untuk opsi update"""
    print_header()
    print("UPDATE FACE EMBEDDINGS - PILIHAN")
    print("=" * 60)
    print("1. Update Embedding Database - Untuk Foto Baru atau Foto Yang Baru Dihapus (update_face_embeddings.py)")
    print("2. Reprocessing Foto Bermasalah (reprocess_face_embeddings.py)")
    print("0. Kembali ke Menu Utama")
    print()

def get_utils_path(filename):
    """Mendapatkan path lengkap ke file utilitas"""
    return os.path.join(UTILS_FOLDER, filename)

def check_files_exist():
    """Memeriksa keberadaan file yang diperlukan"""
    files_to_check = {
        "read_embeddings.py": "Membaca detail database embeddings wajah",
        "update_face_embeddings.py": "Memperbarui embeddings dengan gambar baru",
        "reprocess_face_embeddings.py": "Memproses ulang gambar wajah bermasalah",
        "rebuild_embeddings.py": "Membangun ulang database embeddings"
    }
    
    missing_files = []
    for file, description in files_to_check.items():
        full_path = get_utils_path(file)
        if not os.path.exists(full_path):
            missing_files.append(file)
    
    return missing_files

def show_file_status():
    """Menampilkan status file database"""
    print("\nSTATUS DATABASE:")
    
    # Memeriksa file pickle
    if os.path.exists("face_embeddings.pkl"):
        size = os.path.getsize("face_embeddings.pkl") / (1024 * 1024)  # Convert to MB
        modified_time = time.ctime(os.path.getmtime("face_embeddings.pkl"))
        print(f"✓ face_embeddings.pkl ada ({size:.2f} MB, diperbarui: {modified_time})")
    else:
        print("✗ face_embeddings.pkl belum dibuat")
    
    # Memeriksa file metadata
    if os.path.exists("face_embeddings_metadata.pkl"):
        size = os.path.getsize("face_embeddings_metadata.pkl") / (1024 * 1024)  # Convert to MB
        modified_time = time.ctime(os.path.getmtime("face_embeddings_metadata.pkl"))
        print(f"✓ face_embeddings_metadata.pkl ada ({size:.2f} MB, diperbarui: {modified_time})")
    else:
        print("✗ face_embeddings_metadata.pkl belum dibuat")
    
    # Memeriksa folder database
    if os.path.exists("database") and os.path.isdir("database"):
        folder_count = sum(os.path.isdir(os.path.join("database", d)) for d in os.listdir("database"))
        print(f"✓ Folder database ditemukan dengan {folder_count} folder person")
    else:
        print("✗ Folder database tidak ditemukan")
    
    print()

def run_read_pickle():
    """Menjalankan script read_embeddings.py"""
    print_header()
    print("DETAIL DATABASE FACE EMBEDDINGS")
    print("=" * 60)
    print("Menjalankan read_emebeddings.py...\n")
    
    # Menjalankan script read_embeddings.py dari folder utils
    script_path = get_utils_path("read_embeddings.py")
    subprocess.call([sys.executable, script_path])
    
    input("\nTekan Enter untuk kembali ke menu utama...")

def run_update_embeddings():
    """Menjalankan script update_face_embeddings.py dengan nilai confidence default"""
    print_header()
    print("UPDATE FACE EMBEDDINGS - NORMAL")
    print("=" * 60)
    print("Menjalankan update_face_embeddings.py...\n")
    
    print(f"Menggunakan confidence threshold default: {DEFAULT_CONFIDENCE}")
    
    # Menjalankan script update_face_embeddings.py dengan nilai confidence default
    script_path = get_utils_path("update_face_embeddings.py")
    process = subprocess.Popen([sys.executable, script_path], 
                              env={**os.environ, "CONFIDENCE_THRESHOLD": str(DEFAULT_CONFIDENCE)})
    process.wait()
    
    input("\nTekan Enter untuk kembali ke menu utama...")

def run_reprocess_problem_faces():
    """Menjalankan script reprocess_face_embeddings.py dengan opsi input confidence threshold"""
    print_header()
    print("REPROCESSING FOTO BERMASALAH")
    print("=" * 60)
    print("Menjalankan reprocess_face_embeddings.py...\n")
    
    # Konfirmasi threshold sebelum menjalankan reprocess dengan batasan 0.3-0.7
    while True:
        try:
            threshold_input = input("Masukkan nilai confidence threshold (0.3-0.7) [default: 0.6]: ") or "0.6"
            threshold = float(threshold_input)
            
            # Validasi dalam rentang yang diperbolehkan
            if 0.3 <= threshold <= 0.7:
                break
            else:
                print("Error: Nilai threshold harus antara 0.3 dan 0.7")
                print("Rentang ini memungkinkan deteksi wajah dengan kepercayaan lebih rendah untuk memproses ulang wajah bermasalah")
        except ValueError:
            print("Error: Input harus berupa angka")
    
    print(f"\nMenggunakan confidence threshold: {threshold}")
    
    # Membuat perintah dengan parameter confidence threshold
    script_path = get_utils_path("reprocess_face_embeddings.py")
    
    # Menjalankan script reprocess_face_embeddings.py dengan parameter command line
    process = subprocess.Popen([sys.executable, script_path, "--confidence", str(threshold)])
    process.wait()
    
    input("\nTekan Enter untuk kembali ke menu utama...")

def handle_update_menu():
    """Menangani submenu update embeddings"""
    while True:
        print_update_submenu()

        # Mendapatkan daftar file yang hilang
        missing_files = check_files_exist()
        
        # Menampilkan status database
        # show_file_status()
        
        choice = input("Pilih opsi [0-2]: ")
        
        try:
            option = int(choice)
            
            if option == 0:
                return  # Kembali ke menu utama
            elif option == 1:
                if "update_face_embeddings.py" in missing_files:
                    print(f"\nERROR: File {UTILS_FOLDER}/update_face_embeddings.py tidak ditemukan!")
                    input("Tekan Enter untuk melanjutkan...")
                else:
                    run_update_embeddings()
                    return  # Kembali ke menu utama setelah selesai
            elif option == 2:
                if "reprocess_face_embeddings.py" in missing_files:
                    print(f"\nERROR: File {UTILS_FOLDER}/reprocess_face_embeddings.py tidak ditemukan!")
                    input("Tekan Enter untuk melanjutkan...")
                else:
                    run_reprocess_problem_faces()
                    return  # Kembali ke menu utama setelah selesai
            else:
                print("\nPilihan tidak valid. Silakan pilih menu 0-2.")
                input("Tekan Enter untuk melanjutkan...")
        except ValueError:
            print("\nInput tidak valid. Masukkan angka 0-2.")
            input("Tekan Enter untuk melanjutkan...")

def run_rebuild_database():
    """Menjalankan script rebuild_embeddings.py dengan nilai confidence default"""
    print_header()
    print("REBUILD DATABASE EMBEDDINGS")
    print("=" * 60)
    print("PERINGATAN⚠️: Proses ini akan membangun ulang database dari awal.")
    print("            Semua embedding yang ada akan diganti dengan yang baru.")
    print("            Proses ini mungkin akan memakan waktu cukup lama.\n")
    
    confirm = input("Apakah Anda yakin ingin melanjutkan? (y/n): ").lower()
    if confirm == 'y' or confirm == 'yes':
        print("\nMemulai rebuild database...\n")
        print(f"Menggunakan confidence threshold default: {DEFAULT_CONFIDENCE}")
        
        # Menjalankan script rebuild_embeddings.py dengan nilai confidence default
        script_path = get_utils_path("rebuild_embeddings.py")
        process = subprocess.Popen([sys.executable, script_path], 
                                  env={**os.environ, "CONFIDENCE_THRESHOLD": str(DEFAULT_CONFIDENCE)})
        process.wait()
    else:
        print("\nRebuild database dibatalkan.")
    
    input("\nTekan Enter untuk kembali ke menu utama...")

def main():
    """Fungsi utama program"""
    while True:
        print_header()
        
        # Memeriksa folder utils
        if not os.path.exists(UTILS_FOLDER) or not os.path.isdir(UTILS_FOLDER):
            print(f"KESALAHAN KRITIS: Folder '{UTILS_FOLDER}' tidak ditemukan!")
            print(f"Pastikan folder '{UTILS_FOLDER}' ada di direktori yang sama dengan script ini.")
            print("\nProgram tidak dapat dilanjutkan tanpa folder utilitas.")
            input("\nTekan Enter untuk keluar...")
            sys.exit(1)
        
        # Memeriksa file yang diperlukan
        missing_files = check_files_exist()
        if missing_files:
            print("PERINGATAN: Beberapa file diperlukan tidak ditemukan:")
            for file in missing_files:
                print(f"  - {UTILS_FOLDER}/{file}")
            print("\nBeberapa fungsi mungkin tidak akan berjalan dengan benar.")
            print()
        
        # Menampilkan status database
        show_file_status()
        
        # Menampilkan menu
        print_menu()
        
        # Meminta input pengguna
        choice = input("Pilih menu [0-3]: ")
        
        try:
            option = int(choice)
            
            if option == 0:
                clear_screen()
                print("Program selesai.")
                sys.exit(0)
            elif option == 1:
                if "read_embeddings.py" in missing_files:
                    print(f"\nERROR: File {UTILS_FOLDER}/read_embeddings.py tidak ditemukan!")
                    input("Tekan Enter untuk melanjutkan...")
                else:
                    run_read_pickle()
            elif option == 2:
                # Tampilkan submenu untuk opsi update
                handle_update_menu()
            elif option == 3:
                if "rebuild_embeddings.py" in missing_files:
                    print(f"\nERROR: File {UTILS_FOLDER}/rebuild_embeddings.py tidak ditemukan!")
                    input("Tekan Enter untuk melanjutkan...")
                else:
                    run_rebuild_database()
            else:
                print("\nPilihan tidak valid. Silakan pilih menu 0-3.")
                input("Tekan Enter untuk melanjutkan...")
        except ValueError:
            print("\nInput tidak valid. Masukkan angka 0-3.")
            input("Tekan Enter untuk melanjutkan...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        clear_screen()
        print("\nProgram dihentikan oleh pengguna.")
        sys.exit(0)
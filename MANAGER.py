import os
import sys
import subprocess
import time
import signal
import psutil

# Mendapatkan direktori di mana script ini berada
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path ke folder utilitas relatif terhadap direktori script
UTILS_FOLDER = os.path.join(SCRIPT_DIR, "embedding_manager_utils")

# Path ke aplikasi Flask yang akan direstart
FLASK_APP_PATH = os.path.join(SCRIPT_DIR, "app.py")  # Sesuaikan dengan nama file aplikasi Flask Anda

# Port yang digunakan aplikasi Flask
FLASK_PORT = 5000

# Nilai default untuk confidence threshold
DEFAULT_CONFIDENCE = 0.6

def clear_screen():
    """Membersihkan layar terminal"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Menampilkan header aplikasi"""
    clear_screen()
    print("=" * 60)
    print("Face'O'Classifier Manager".center(60))
    print("=" * 60)
    print("Manajemen Aplikasi Face'O'Classifier".center(60))
    print("=" * 60)
    print()

def print_menu():
    """Menampilkan menu utama"""
    print("PILIHAN MENU:")
    print("1. (CHECK) Lihat Detail Database (read_embeddings.py)")
    print("2. (UPDATE) Update Database Face Embeddings")
    print("3. (REBUILD) Rebuild Database Face Embeddings (rebuild_embeddings.py)")
    print("4. (RESTART) Restart Aplikasi - DIANJURKAN SETELAH MELAKUKAN UPDATE DATABASE")
    print("5. (STOP) Hentikan Aplikasi")
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
    
    # Path absolut untuk file pickle
    face_embeddings_path = os.path.join(SCRIPT_DIR, "face_embeddings.pkl")
    metadata_path = os.path.join(SCRIPT_DIR, "face_embeddings_metadata.pkl")
    database_path = os.path.join(SCRIPT_DIR, "database")
    
    # Memeriksa file pickle
    if os.path.exists(face_embeddings_path):
        size = os.path.getsize(face_embeddings_path) / (1024 * 1024)  # Convert to MB
        modified_time = time.ctime(os.path.getmtime(face_embeddings_path))
        print(f"✓ face_embeddings.pkl ada ({size:.2f} MB, diperbarui: {modified_time})")
    else:
        print("✗ face_embeddings.pkl belum dibuat")
    
    # Memeriksa file metadata
    if os.path.exists(metadata_path):
        size = os.path.getsize(metadata_path) / (1024 * 1024)  # Convert to MB
        modified_time = time.ctime(os.path.getmtime(metadata_path))
        print(f"✓ face_embeddings_metadata.pkl ada ({size:.2f} MB, diperbarui: {modified_time})")
    else:
        print("✗ face_embeddings_metadata.pkl belum dibuat")
    
    # Memeriksa folder database
    if os.path.exists(database_path) and os.path.isdir(database_path):
        folder_count = sum(os.path.isdir(os.path.join(database_path, d)) for d in os.listdir(database_path))
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
    
    # Gunakan direktori script sebagai working directory
    subprocess.call([sys.executable, script_path], cwd=SCRIPT_DIR)
    
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
    process = subprocess.Popen(
        [sys.executable, script_path], 
        env={**os.environ, "CONFIDENCE_THRESHOLD": str(DEFAULT_CONFIDENCE)},
        cwd=SCRIPT_DIR
    )
    process.wait()
    
    input("\nTekan Enter untuk kembali ke menu utama...")

def run_reprocess_problem_faces():
    """Menjalankan script reprocess_face_embeddings.py dengan opsi input confidence threshold"""
    print_header()
    print("REPROCESSING FOTO BERMASALAH")
    print("=" * 60)
    print("Menjalankan reprocess_face_embeddings.py...\n")
    
    # Konfirmasi threshold sebelum menjalankan reprocess dengan batasan 0.3-0.7
    print("Keterangan nilai confidence threshold:")
    print("- Nilai lebih rendah (0.3-0.5): Dapat mendeteksti wajah yang mungkin kurang jelas namun dapat meningkatkan tingkat false positive")
    print("- Nilai lebih tinggi (0.6-0.7): Lebih selektif, hanya mendeteksi wajah yang jelas")
    print("- Untuk foto yang bermasalah dengan tidak terdeteksinya wajah, coba gunakan nilai lebih rendah")
    print()
    
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
    process = subprocess.Popen(
        [sys.executable, script_path, "--confidence", str(threshold)],
        cwd=SCRIPT_DIR
    )
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
        process = subprocess.Popen(
            [sys.executable, script_path], 
            env={**os.environ, "CONFIDENCE_THRESHOLD": str(DEFAULT_CONFIDENCE)},
            cwd=SCRIPT_DIR
        )
        process.wait()
    else:
        print("\nRebuild database dibatalkan.")
    
    input("\nTekan Enter untuk kembali ke menu utama...")

def find_flask_processes():
    """
    Fungsi untuk menemukan proses Flask yang berjalan di port tertentu
    Mendukung deteksi di semua antarmuka jaringan (0.0.0.0)
    """
    flask_processes = []
    
    try:
        # Gunakan psutil untuk menemukan semua proses dengan koneksi jaringan
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'connections']):
            try:
                # Periksa apakah proses memiliki koneksi pada port yang kita cari
                for conn in proc.connections():
                    # Periksa jika ada koneksi dengan port yang sama, baik di localhost maupun 0.0.0.0
                    if conn.status == 'LISTEN' and conn.laddr.port == FLASK_PORT:
                        # Periksa apakah ini proses Python dan kemungkinan aplikasi Flask
                        if proc.name().lower() in ['python', 'python.exe', 'python3', 'python3.exe']:
                            cmdline = proc.cmdline()
                            # Verifikasi lebih lanjut ini adalah aplikasi Flask kita
                            if len(cmdline) > 1 and any('app.py' in arg for arg in cmdline):
                                flask_processes.append(proc)
                                break
            except (psutil.AccessDenied, psutil.NoSuchProcess, AttributeError):
                continue
    
    except Exception as e:
        print(f"Error saat mencari proses Flask: {e}")
    
    return flask_processes

def kill_flask_processes():
    """Menghentikan semua proses Flask yang ditemukan"""
    flask_processes = find_flask_processes()
    killed = False
    
    for proc in flask_processes:
        try:
            proc.terminate()
            print(f"Proses Flask dengan PID {proc.pid} berhasil dimatikan.")
            killed = True
        except Exception as e:
            print(f"Gagal mematikan proses dengan PID {proc.pid}: {e}")
    
    # Jika terminasi dengan graceful tidak berfungsi, coba paksa kill
    if not killed:
        try:
            # Metode alternatif untuk Windows
            if os.name == 'nt':
                # Gunakan netstat untuk cari proses di port yang ditentukan
                command = f'netstat -ano | findstr :{FLASK_PORT}'
                result = subprocess.check_output(command, shell=True).decode()
                if result:
                    lines = result.strip().split('\n')
                    for line in lines:
                        if 'LISTENING' in line:
                            pid = line.strip().split()[-1]
                            try:
                                # Matikan proses dengan PID yang ditemukan
                                subprocess.call(['taskkill', '/F', '/PID', pid])
                                print(f"Proses dengan PID {pid} berhasil dimatikan (metode alternatif).")
                                killed = True
                            except Exception as e:
                                print(f"Gagal mematikan proses (metode alternatif): {e}")
            
            # Metode alternatif untuk Linux/macOS
            else:
                try:
                    # Gunakan lsof untuk cari proses di port yang ditentukan
                    command = f"lsof -i :{FLASK_PORT} -t"
                    result = subprocess.check_output(command, shell=True).decode()
                    if result:
                        pids = result.strip().split('\n')
                        for pid in pids:
                            try:
                                # Kill -9 untuk paksa mematikan
                                os.kill(int(pid), signal.SIGKILL)
                                print(f"Proses dengan PID {pid} berhasil dimatikan paksa (metode alternatif).")
                                killed = True
                            except Exception as e:
                                print(f"Gagal mematikan proses (metode alternatif): {e}")
                except subprocess.CalledProcessError:
                    pass  # Tidak ada proses yang ditemukan
        except Exception as e:
            print(f"Error pada metode alternatif: {e}")
    
    return killed

def restart_flask_app():
    """Fungsi untuk merestart aplikasi Flask klasifikasi wajah"""
    print_header()
    print("RESTART APLIKASI KLASIFIKASI WAJAH")
    print("=" * 60)
    print("Proses ini akan mematikan aplikasi yang sedang berjalan dan menjalankannya kembali.")
    print(f"Aplikasi berjalan di port {FLASK_PORT} (http://0.0.0.0:{FLASK_PORT})\n")
    
    # Konfirmasi restart
    confirm = input("Apakah Anda yakin ingin merestart aplikasi? (y/n): ").lower()
    if confirm != 'y' and confirm != 'yes':
        print("\nRestart aplikasi dibatalkan.")
        input("\nTekan Enter untuk kembali ke menu utama...")
        return
    
    # Cek apakah file aplikasi Flask ada
    if not os.path.exists(FLASK_APP_PATH):
        print(f"\nERROR: File aplikasi Flask tidak ditemukan di {FLASK_APP_PATH}!")
        print("Pastikan path aplikasi Flask sudah benar.")
        input("\nTekan Enter untuk kembali ke menu utama...")
        return
    
    print(f"\nMematikan proses Flask yang sedang berjalan di port {FLASK_PORT}...")
    
    # Cari dan matikan proses Flask
    killed = kill_flask_processes()
    
    if not killed:
        print(f"Tidak ditemukan proses Flask aktif di port {FLASK_PORT}.")
    
    print("\nMenjalankan aplikasi Flask...")
    
    # Jalankan aplikasi Flask dalam proses baru
    try:
        flask_app_process = subprocess.Popen(
            [sys.executable, FLASK_APP_PATH],
            cwd=os.path.dirname(FLASK_APP_PATH)
        )
        print(f"Aplikasi Flask berhasil dijalankan dengan PID {flask_app_process.pid}")
        print(f"Server berjalan di http://0.0.0.0:{FLASK_PORT}")
    except Exception as e:
        print(f"Error saat menjalankan aplikasi Flask: {e}")
    
    input("\nTekan Enter untuk kembali ke menu utama...")

def stop_flask_app():
    """Fungsi untuk menghentikan aplikasi Flask yang sedang berjalan"""
    print_header()
    print("HENTIKAN APLIKASI KLASIFIKASI WAJAH")
    print("=" * 60)
    print(f"Proses ini akan mematikan aplikasi Flask yang sedang berjalan di port {FLASK_PORT}.")
    print("Aplikasi tidak akan dimulai ulang sampai Anda memilih opsi restart.\n")
    
    # Konfirmasi untuk menghentikan aplikasi
    confirm = input("Apakah Anda yakin ingin menghentikan aplikasi? (y/n): ").lower()
    if confirm != 'y' and confirm != 'yes':
        print("\nPenghentian aplikasi dibatalkan.")
        input("\nTekan Enter untuk kembali ke menu utama...")
        return
    
    print(f"\nMematikan proses Flask yang sedang berjalan di port {FLASK_PORT}...")
    
    # Cari dan matikan proses Flask
    killed = kill_flask_processes()
    
    if not killed:
        print(f"Tidak ditemukan proses Flask aktif di port {FLASK_PORT}.")
    else:
        print("\nAplikasi Flask berhasil dihentikan.")
    
    input("\nTekan Enter untuk kembali ke menu utama...")

def main():
    """Fungsi utama program"""
    while True:
        print_header()
        
        # Memeriksa folder utils
        if not os.path.exists(UTILS_FOLDER) or not os.path.isdir(UTILS_FOLDER):
            print(f"KESALAHAN KRITIS: Folder '{UTILS_FOLDER}' tidak ditemukan!")
            print(f"Pastikan folder '{UTILS_FOLDER}' ada di direktori yang sama dengan script ini.")
            print(f"Script Dir: {SCRIPT_DIR}")
            print(f"Utils Dir: {UTILS_FOLDER}")
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
        choice = input("Pilih menu [0-5]: ")
        
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
            elif option == 4:
                # Panggil fungsi restart aplikasi Flask
                restart_flask_app()
            elif option == 5:
                # Panggil fungsi stop aplikasi Flask
                stop_flask_app()
            else:
                print("\nPilihan tidak valid. Silakan pilih menu 0-5.")
                input("Tekan Enter untuk melanjutkan...")
        except ValueError:
            print("\nInput tidak valid. Masukkan angka 0-5.")
            input("Tekan Enter untuk melanjutkan...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        clear_screen()
        print("\nProgram dihentikan oleh pengguna.")
        sys.exit(0)
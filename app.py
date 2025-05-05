import os
import random
import shutil
import glob
import time
import zipfile
import json
from flask import Flask, jsonify, render_template, request, redirect, url_for, send_from_directory, session
from flask_session import Session
from werkzeug.utils import secure_filename
from classify_faces import classify_faces

# --- Utility Function ---
def clear_folder(folder_path):
    """Menghapus folder lama dan membuat folder kosong baru."""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)

# --- Membersihkan file dan folder saat aplikasi mulai ---
# Membersihkan file session
session_files = glob.glob('flask_session/*')
for file in session_files:
    os.remove(file)

# Membersihkan folder uploads dan zip
clear_folder('uploads')
clear_folder('zip')

# Membersihkan folder (Classified) sebelumnya
classified_folders = glob.glob('(Classified)*')
for folder in classified_folders:
    if os.path.isdir(folder):
        shutil.rmtree(folder)

# --- Konfigurasi Flask App ---
app = Flask(__name__)
app.secret_key = "your_secret_key"
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = 'flask_session'
app.config['SESSION_PERMANENT'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ZIP_FOLDER'] = 'zip'
Session(app)

# --- Routes ---
@app.route('/')
def index():
    """Menampilkan halaman utama."""
    output_path = session.get('output_path')
    output_folder_empty = True

    if output_path and os.path.exists(output_path):
        output_folder_empty = not os.listdir(output_path)

    return render_template(
        'index.html',
        output_path=output_path,
        original_folder_name=session.get('original_folder_name'),
        is_output_folder_empty=output_folder_empty,
        zip_available=session.get('zip_path') is not None
    )

processing_status = {}

@app.route('/status', methods=['GET'])
def check_status():
    """Endpoint untuk memeriksa status pemrosesan"""
    session_id = request.args.get('session_id')
    if not session_id or session_id not in processing_status:
        return jsonify({
            'status': 'unknown',
            'message': 'No processing information available',
            'progress': 0
        })
    
    return jsonify(processing_status[session_id])

# Fungsi helper untuk mengupdate status pemrosesan
def update_processing_status(session_id, status, message, progress):
    """
    Update status pemrosesan global
    
    Args:
        session_id: ID sesi unik
        status: Status pemrosesan (preparing, extracting, processing, complete, error)
        message: Pesan detail untuk ditampilkan ke pengguna
        progress: Persentase kemajuan (0-100)
    """
    processing_status[session_id] = {
        'status': status,
        'message': message,
        'progress': progress
    }

@app.route('/upload', methods=['POST'])
def upload_file():
    """Menerima upload file ZIP, mengekstrak, dan memulai proses klasifikasi wajah."""
    if 'zipfile' not in request.files:
        return redirect(request.url)
    file = request.files['zipfile']
    if file.filename == '':
        return redirect(request.url)
    if not file.filename.lower().endswith('.zip'):
        return "Hanya file .zip yang diperbolehkan", 400
    
    # Reset sesi dan folder upload
    session.clear()
    session.modified = True
    clear_folder(app.config['UPLOAD_FOLDER'])
    
    # Simpan file ZIP
    zip_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(zip_path)
    
    # Ekstrak ZIP
    extracted_folder_name = os.path.splitext(secure_filename(file.filename))[0]
    extracted_folder_path = os.path.join(app.config['UPLOAD_FOLDER'], extracted_folder_name)
    os.makedirs(extracted_folder_path, exist_ok=True)
    
    # Import shutil di awal untuk menghindari import berulang
    import shutil
    
    # Ekstrak ZIP ke folder sementara
    temp_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_extract')
    os.makedirs(temp_folder, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_folder)
    
    # Fungsi rekursif untuk memindahkan semua file foto ke folder target
    def move_all_images(source_dir, target_dir):
        """
        Memindahkan semua file gambar dari source_dir dan subdirektorinya ke target_dir.
        Jika file bukan gambar, lewatkan.
        """
        for item in os.listdir(source_dir):
            source_path = os.path.join(source_dir, item)
            
            # Jika direktori, cari file gambar di dalamnya secara rekursif
            if os.path.isdir(source_path):
                move_all_images(source_path, target_dir)
            else:
                # Pemeriksaan ekstensi file untuk gambar
                image_extensions = ['.jpg', '.jpeg', '.png']
                if any(item.lower().endswith(ext) for ext in image_extensions):
                    target_path = os.path.join(target_dir, item)
                    
                    # Jika file dengan nama yang sama sudah ada, tambahkan penanda unik
                    if os.path.exists(target_path):
                        base_name, ext = os.path.splitext(item)
                        target_path = os.path.join(target_dir, f"{base_name}_{int(time.time())}_{random.randint(1000, 9999)}{ext}")
                    
                    # Pindahkan file gambar
                    shutil.copy2(source_path, target_path)  # copy2 mempertahankan metadata
    
    # Periksa isi folder temp
    contents = os.listdir(temp_folder)
    
    # Jika hanya ada satu item dan itu adalah folder (kasus namafolder/foto-foto)
    if len(contents) == 1 and os.path.isdir(os.path.join(temp_folder, contents[0])):
        top_level_folder = os.path.join(temp_folder, contents[0])
        move_all_images(top_level_folder, extracted_folder_path)
    else:
        # Jika struktur campuran atau flat, pindahkan semua gambar dari semua folder/file
        move_all_images(temp_folder, extracted_folder_path)
    
    # Hapus folder temporary
    shutil.rmtree(temp_folder)
    
    # Hapus file ZIP setelah ekstraksi
    os.remove(zip_path)
    
    # Periksa apakah ada file yang berhasil dipindahkan
    if not os.listdir(extracted_folder_path):
        # Jika tidak ada file gambar yang ditemukan, beri pesan error
        shutil.rmtree(extracted_folder_path)  # Hapus folder kosong
        return "Tidak ada file gambar yang ditemukan dalam ZIP", 400
    
    # Simpan info di sesi
    session['original_folder_name'] = extracted_folder_name
    session['folder_name'] = extracted_folder_name
    
    # Mulai klasifikasi wajah
    output_folder_name = "(Classified) " + extracted_folder_name
    output_folder, output_zip_path = classify_faces(extracted_folder_path, output_folder=output_folder_name)
    session['output_path'] = output_folder
    session['zip_path'] = output_zip_path
    session['upload_complete'] = True
    
    return redirect(url_for('index'))

@app.route('/cancel', methods=['POST'])
def cancel_processing():
    """Endpoint untuk membatalkan proses klasifikasi yang sedang berjalan"""
    from classify_faces import cancel_processing
    
    if cancel_processing():
        return jsonify({'status': 'success', 'message': 'Processing cancelled'})
    return jsonify({'status': 'error', 'message': 'Failed to cancel processing'})


@app.route('/preview_folders')
def preview_folders():
    """Menampilkan daftar folder hasil klasifikasi."""
    output_path = session.get('output_path')
    if not output_path or not os.path.exists(output_path):
        return redirect(url_for('index'))

    folders = [
        name for name in os.listdir(output_path)
        if os.path.isdir(os.path.join(output_path, name))
        and name != 'labels'
        and len(os.listdir(os.path.join(output_path, name))) > 0
    ]

    return render_template('preview_folders.html', folders=folders)

@app.route('/preview/<folder_name>')
def preview_images(folder_name):
    """Menampilkan gambar-gambar dari folder tertentu hasil klasifikasi."""
    output_path = session.get('output_path')
    if not output_path:
        return redirect(url_for('index'))

    folder_path = os.path.join(output_path, folder_name)
    if not os.path.exists(folder_path):
        return redirect(url_for('preview_folders'))

    image_filenames = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('jpg', 'jpeg', 'png'))
    ]

    images = [
        url_for('serve_output', filename=os.path.join(folder_name, filename))
        for filename in image_filenames
    ]

    return render_template('preview_images.html', images=images, folder_name=folder_name)

@app.route('/output/<path:filename>')
def serve_output(filename):
    """Melayani file gambar untuk ditampilkan pada halaman preview."""
    output_path = session.get('output_path')
    if not output_path:
        return "No output directory set", 404

    filename = filename.replace("\\", "/")  # Normalisasi path
    filepath = os.path.join(output_path, filename)
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return "File not found", 404

    return send_from_directory(output_path, filename)

@app.route('/download_zip')
def download_zip():
    """Mengunduh hasil klasifikasi dalam bentuk file ZIP."""
    zip_path = session.get('zip_path')
    if zip_path and os.path.exists(zip_path):
        zip_dir, zip_filename = os.path.split(zip_path)
        return send_from_directory(zip_dir, zip_filename, as_attachment=True)
    return "ZIP file not found", 404

@app.route('/open_output')
def open_output():
    """Membuka folder hasil klasifikasi secara lokal."""
    output_path = session.get('output_path')
    if output_path and os.path.exists(output_path):
        try:
            os.startfile(output_path)  # Untuk Windows
        except Exception:
            os.system(f'xdg-open "{output_path}"')  # Untuk Linux/macOS
    return '', 204

@app.route('/reset', methods=['POST'])
def reset():
    """Mereset aplikasi: menghapus folder upload dan output terkait sesi saat ini saja."""
    # Clear uploads folder for current session
    folder_name = session.get('folder_name')
    if folder_name:
        folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder_name)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)

    # Delete only the specific classified ZIP file related to current session
    zip_path = session.get('zip_path')
    if zip_path and os.path.exists(zip_path):
        try:
            os.remove(zip_path)
        except Exception as e:
            print(f"Error removing zip file: {e}")

    # Delete the specific output folder for current session
    output_path = session.get('output_path')
    if output_path and os.path.exists(output_path):
        try:
            shutil.rmtree(output_path)
        except Exception as e:
            print(f"Error removing output folder: {e}")

    # Clear session data
    session.clear()
    return redirect(url_for('index'))

# --- Main Program ---
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)
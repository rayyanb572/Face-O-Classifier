from datetime import timedelta, datetime
import io
import os
import random
import shutil
import glob
import threading
import time
import zipfile
import json
import hashlib
import pickle
from PIL import Image
from collections import defaultdict
from flask import Flask, jsonify, render_template, request, redirect, url_for, send_from_directory, session, flash
from flask_session import Session
from werkzeug.utils import secure_filename
from classify_faces import classify_faces
from admin import admin_bp
import config

# --- Global Variables for Background Processing ---
background_processes = {}  # Store background process status

# --- Utility Function ---
def clear_folder(folder_path):
    """Menghapus folder lama dan membuat folder kosong baru."""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)

# --- Background Processing Function ---
def background_classify_faces(session_id, extracted_folder_path, output_folder_name):
    """Menjalankan klasifikasi wajah di background thread"""
    try:
        # Update status
        background_processes[session_id] = {
            'status': 'processing',
            'message': 'Processing...',
            'progress': 0,
            'start_time': time.time()
        }
        
        # Jalankan klasifikasi
        output_folder, output_zip_path, processing_time = classify_faces(
            extracted_folder_path, 
            output_folder=output_folder_name
        )
        
        # Update status selesai
        background_processes[session_id] = {
            'status': 'complete',
            'message': 'Classification completed successfully',
            'progress': 100,
            'output_folder': output_folder,
            'zip_path': output_zip_path,
            'processing_time': processing_time,
            'completed_at': time.time()
        }
        
    except Exception as e:
        # Update status error
        background_processes[session_id] = {
            'status': 'error',
            'message': f'Error during classification: {str(e)}',
            'progress': 0,
            'error': str(e)
        }

def create_thumbnail(image_path, max_size=(300, 300), quality=85):
    """
    Membuat thumbnail dari gambar dengan mempertahankan rasio aspek
    
    Args:
        image_path: Path ke file gambar
        max_size: Tuple (width, height) maksimum untuk thumbnail
        quality: Kualitas JPEG (1-100)
    
    Returns:
        Tuple (thumbnail_data, format) atau None jika gagal
    """
    try:
        with Image.open(image_path) as img:
            # Konversi ke RGB jika diperlukan (untuk PNG dengan transparency)
            if img.mode in ('RGBA', 'LA', 'P'):
                # Buat background putih untuk transparency
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Buat thumbnail dengan mempertahankan rasio aspek
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Simpan ke BytesIO
            img_io = io.BytesIO()
            img.save(img_io, format='JPEG', quality=quality, optimize=True)
            img_io.seek(0)
            
            return img_io.getvalue(), 'JPEG'
    except Exception as e:
        print(f"Error creating thumbnail for {image_path}: {e}")
        return None, None

# --- Membersihkan file dan folder saat aplikasi mulai ---
# Membersihkan file session
session_files = glob.glob('flask_session/*')
for file in session_files:
    os.remove(file)

# Membersihkan uploads dan zip folder saat aplikasi start
clear_folder('uploads')
clear_folder('zip')

# Membersihkan folder (Classified) sebelumnya yang mungkin tersisa dari crash
classified_folders = glob.glob('(Classified)*')
for folder in classified_folders:
    if os.path.isdir(folder):
        shutil.rmtree(folder)

# Ensure database folders exist
os.makedirs('database', exist_ok=True)

# --- Konfigurasi Flask App ---
app = Flask(__name__)
app.secret_key = config.SECRET_KEY
app.config['SESSION_TYPE'] = config.SESSION_TYPE
app.config['SESSION_FILE_DIR'] = config.SESSION_FILE_DIR
app.config['SESSION_PERMANENT'] = config.SESSION_PERMANENT
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['ZIP_FOLDER'] = config.ZIP_FOLDER
app.config['DATABASE_FOLDER'] = config.DATABASE_FOLDER
Session(app)

# Register Admin Blueprint
app.register_blueprint(admin_bp, url_prefix='/admin')

# --- Main Routes ---
@app.route('/')
def index():
    """Menampilkan halaman utama."""
    session_id = session.get('device_session_id')
    
    # Cek status background process
    if session_id and session_id in background_processes:
        process_status = background_processes[session_id]
        
        if process_status['status'] == 'complete':
            # Pindahkan hasil ke session
            session['output_path'] = process_status.get('output_folder')
            session['zip_path'] = process_status.get('zip_path')
            session['processing_time'] = process_status.get('processing_time')
            session['upload_complete'] = True
            
            # Hapus dari background processes
            del background_processes[session_id]
    
    output_path = session.get('output_path')
    output_folder_empty = True

    if output_path and os.path.exists(output_path):
        output_folder_empty = not os.listdir(output_path)

    return render_template(
        'index.html',
        output_path=output_path,
        original_folder_name=session.get('original_folder_name'),
        is_output_folder_empty=output_folder_empty,
        zip_available=session.get('zip_path') is not None,
        processing_time=session.get('processing_time')
    )

@app.route('/processing_status', methods=['GET'])
def check_processing_status():
    """Endpoint untuk memeriksa status pemrosesan background"""
    session_id = session.get('device_session_id')
    
    if not session_id:
        return jsonify({"status": "unknown", "message": "No session found"})
    
    if session_id in background_processes:
        return jsonify(background_processes[session_id])
    elif session.get('upload_complete'):
        return jsonify({"status": "complete"})
    else:
        return jsonify({"status": "unknown", "message": "No processing information available"})

@app.route('/upload', methods=['POST'])
def upload_file():
    """Menerima upload file ZIP, mengekstrak, dan memulai proses klasifikasi wajah di background."""
    if 'zipfile' not in request.files:
        return redirect(request.url)
    file = request.files['zipfile']
    if file.filename == '':
        return redirect(request.url)
    if not file.filename.lower().endswith('.zip'):
        return "Hanya file .zip yang diperbolehkan", 400
    
    # Reset sesi
    session.clear()
    session.modified = True
    
    # Buat session ID unik untuk device ini
    device_session_id = str(int(time.time())) + "_" + str(random.randint(1000, 9999))
    session['device_session_id'] = device_session_id
    
    # Buat folder khusus untuk session ini di uploads
    device_upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], device_session_id)
    os.makedirs(device_upload_folder, exist_ok=True)
    
    # Simpan file ZIP
    zip_path = os.path.join(device_upload_folder, secure_filename(file.filename))
    file.save(zip_path)
    
    # Ekstrak ZIP
    extracted_folder_name = os.path.splitext(secure_filename(file.filename))[0]
    extracted_folder_path = os.path.join(device_upload_folder, extracted_folder_name)
    os.makedirs(extracted_folder_path, exist_ok=True)
    
    # Import shutil di awal untuk menghindari import berulang
    import shutil
    
    # Ekstrak ZIP ke folder sementara
    temp_folder = os.path.join(device_upload_folder, 'temp_extract')
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
        shutil.rmtree(device_upload_folder)   # Hapus folder device juga
        return "Tidak ada file gambar yang ditemukan dalam ZIP", 400
    
    session['original_folder_name'] = extracted_folder_name
    session['folder_name'] = extracted_folder_name
    session['device_folder_path'] = device_upload_folder
    session['extracted_folder_path'] = extracted_folder_path
    
    # Mulai klasifikasi wajah di background
    output_folder_name = "(Classified) " + extracted_folder_name
    
    # Jalankan background thread
    thread = threading.Thread(
        target=background_classify_faces,
        args=(device_session_id, extracted_folder_path, output_folder_name)
    )
    thread.daemon = True
    thread.start()
    
    # Langsung redirect tanpa menunggu hasil
    return redirect(url_for('index'))

@app.route('/cancel', methods=['POST'])
def cancel_processing():
    """Endpoint untuk membatalkan proses klasifikasi yang sedang berjalan"""
    from classify_faces import cancel_processing
    
    session_id = session.get('device_session_id')
    
    if cancel_processing():
        # Update status background process jika ada
        if session_id and session_id in background_processes:
            background_processes[session_id] = {
                'status': 'cancelled',
                'message': 'Processing cancelled by user',
                'progress': 0
            }
        
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
    
    # Custom sorting: VISUALIZED pertama, UNKNOWN kedua, sisanya alfabetis
    def custom_sort_key(folder_name):
        folder_upper = folder_name.upper()
        if folder_upper == 'VISUALIZED':
            return (1, folder_name.lower())  # Pertama
        elif folder_upper == 'UNKNOWN':
            return (2, folder_name.lower())  # Kedua
        else:
            return (3, folder_name.lower())  # Folder lainnya setelahnya
    
    # Urutkan folder dengan custom key
    folders.sort(key=custom_sort_key)
    
    return render_template('preview_folders.html', folders=folders)

@app.route('/thumbnail/<path:filename>')
def serve_thumbnail(filename):
    """Melayani thumbnail gambar dengan resolusi rendah untuk preview cepat."""
    output_path = session.get('output_path')
    if not output_path:
        return "No output directory set", 404

    filename = filename.replace("\\", "/")  # Normalisasi path
    filepath = os.path.join(output_path, filename)
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return "File not found", 404

    # Cek apakah file adalah gambar
    if not filename.lower().endswith(('jpg', 'jpeg', 'png')):
        return "Not an image file", 400

    # Buat thumbnail
    thumbnail_data, format_type = create_thumbnail(
        filepath, 
        max_size=(300, 300),  # Ukuran maksimum thumbnail
        quality=75  # Kualitas kompresi
    )
    
    if thumbnail_data is None:
        # Fallback ke gambar asli jika thumbnail gagal dibuat
        return send_from_directory(output_path, filename)
    
    # Return thumbnail sebagai response
    from flask import Response
    return Response(
        thumbnail_data,
        mimetype=f'image/{format_type.lower()}'
    )

# Update route preview_images untuk menggunakan thumbnail
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

    # Urutkan nama file secara alfanumerik
    image_filenames.sort()

    # Buat data gambar dengan thumbnail dan full size URLs
    images = []
    for filename in image_filenames:
        images.append({
            'thumbnail': url_for('serve_thumbnail', filename=os.path.join(folder_name, filename)),
            'full': url_for('serve_output', filename=os.path.join(folder_name, filename)),
            'filename': filename
        })

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
            # Cek OS
            if os.name == 'nt':  # Windows
                os.startfile(output_path)
            else:  # Linux/macOS
                import subprocess
                subprocess.Popen(['xdg-open', output_path])
        except Exception as e:
            print(f"Error opening output folder: {e}")
    return '', 204

@app.route('/reset', methods=['POST'])
def reset():
    """Mereset aplikasi: menghapus folder upload dan output terkait sesi perangkat saat ini saja."""
    # Hapus folder uploads khusus untuk device ini
    device_folder_path = session.get('device_folder_path')
    if device_folder_path and os.path.exists(device_folder_path):
        try:
            shutil.rmtree(device_folder_path)
        except Exception as e:
            print(f"Error removing device upload folder: {e}")

    # Hapus file ZIP khusus untuk device ini
    zip_path = session.get('zip_path')
    if zip_path and os.path.exists(zip_path):
        try:
            os.remove(zip_path)
        except Exception as e:
            print(f"Error removing zip file: {e}")

    # Hapus folder output khusus untuk device ini
    output_path = session.get('output_path')
    if output_path and os.path.exists(output_path):
        try:
            shutil.rmtree(output_path)
        except Exception as e:
            print(f"Error removing output folder: {e}")

    # Bersihkan data sesi
    session.clear()
    return redirect(url_for('index'))

# --- Main Program ---
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)
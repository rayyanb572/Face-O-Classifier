from datetime import timedelta, datetime
import os
import random
import shutil
import glob
import time
import zipfile
import json
import hashlib
import pickle
from collections import defaultdict
from flask import Flask, jsonify, render_template, request, redirect, url_for, send_from_directory, session, flash
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

# Membersihkan uploads dan zip folder saat aplikasi start
clear_folder('uploads')
clear_folder('zip')

# Membersihkan folder (Classified) sebelumnya yang mungkin tersisa dari crash
classified_folders = glob.glob('(Classified)*')
for folder in classified_folders:
    if os.path.isdir(folder):
        shutil.rmtree(folder)

# --- Admin Configuration ---
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = hashlib.sha256("admin123".encode()).hexdigest()  # Default password: admin123

# Ensure database folders exist
os.makedirs('database', exist_ok=True)

# --- Konfigurasi Flask App ---
app = Flask(__name__)
app.secret_key = "your_secret_key"
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = 'flask_session'
app.config['SESSION_PERMANENT'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ZIP_FOLDER'] = 'zip'
app.config['DATABASE_FOLDER'] = 'database'
Session(app)

# --- Main Routes ---
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

@app.route('/processing_status', methods=['GET'])
def check_processing_status():
    """Endpoint untuk memeriksa status pemrosesan"""
    status = "processing"
    
    # Check if processing is complete based on output_path in session
    if session.get('upload_complete'):
        status = "complete"
    
    return jsonify({"status": status})

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
    
    # Simpan info di sesi
    session['original_folder_name'] = extracted_folder_name
    session['folder_name'] = extracted_folder_name
    session['device_folder_path'] = device_upload_folder
    session['extracted_folder_path'] = extracted_folder_path
    
    # Mulai klasifikasi wajah
    # Ubah: Hapus session ID dari nama folder output
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

# --- Admin Authentication Routes ---
@app.route('/admin/login', methods=['POST'])
def admin_login():
    """Handle admin login."""
    username = request.form.get('username')
    password = request.form.get('password')
    
    # Hash the password
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    
    if username == ADMIN_USERNAME and hashed_password == ADMIN_PASSWORD:
        session['admin_logged_in'] = True
        return redirect(url_for('admin_panel'))
    else:
        flash('Invalid username or password', 'danger')
        # Redirect back to index with a login_error parameter
        return redirect(url_for('index', login_error='true'))

def admin_required(func):
    """Decorator to require admin login."""
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return redirect(url_for('index'))
        return func(*args, **kwargs)
    decorated_function.__name__ = func.__name__
    return decorated_function

# Tambahkan fungsi untuk audit database
def audit_face_database():
    """Mengaudit database wajah untuk mendeteksi perbedaan antara file foto dan embeddings."""
    
    # Path ke file embeddings dan direktori database
    pickle_path = 'face_embeddings.pkl'  # File berada di root direktori
    metadata_path = 'face_embeddings_metadata.pkl'  # File berada di root direktori
    database_dir = app.config['DATABASE_FOLDER']
    
    # Cek keberadaan file pickle
    if not os.path.exists(pickle_path):
        return {
            'error': True,
            'message': f"File embeddings tidak ditemukan di {pickle_path}"
        }
    
    # Cek keberadaan file metadata
    has_metadata = os.path.exists(metadata_path)
    
    # Load embeddings
    with open(pickle_path, "rb") as f:
        embeddings = pickle.load(f)
    
    # Load metadata jika tersedia
    if has_metadata:
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
    else:
        metadata = {}
    
    # Hitung statistik embeddings
    total_embeddings = 0
    embeddings_per_person = {}
    
    for person, vectors in embeddings.items():
        embeddings_per_person[person] = len(vectors)
        total_embeddings += len(vectors)
    
    # Scan direktori database untuk menghitung file
    file_counts = defaultdict(int)
    file_list = defaultdict(list)
    valid_extensions = ('.jpg', '.jpeg', '.png')
    
    total_files = 0
    for person_name in os.listdir(database_dir):
        person_dir = os.path.join(database_dir, person_name)
        if os.path.isdir(person_dir) and person_name != 'embeddings':
            for file_name in os.listdir(person_dir):
                if file_name.lower().endswith(valid_extensions):
                    file_counts[person_name] += 1
                    file_list[person_name].append(file_name)
                    total_files += 1
    
    # Persiapkan data hasil audit
    persons_data = []
    
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
        
        status = "ok" if diff == 0 else "warning"
        
        persons_data.append({
            'name': person,
            'file_count': file_count,
            'embedding_count': emb_count,
            'difference': diff,
            'status': status,
            'problem_files': problem_files[:5]  # Batasi jumlah problem files yang ditampilkan
        })
    
    # Ringkasan Statistik
    summary = {
        'total_people': len(embeddings),
        'total_embeddings': total_embeddings,
        'total_files': total_files,
        'difference': total_embeddings - total_files,
        'has_metadata': has_metadata
    }
    
    return {
        'error': False,
        'summary': summary,
        'persons': persons_data
    }

# Tambahkan route untuk halaman audit database
@app.route('/admin/audit')
@admin_required
def admin_audit():
    """Halaman untuk mengaudit database wajah."""
    # Jalankan audit
    audit_result = audit_face_database()
    
    if audit_result.get('error'):
        session['admin_message'] = audit_result.get('message')
        session['admin_message_type'] = 'danger'
        return redirect(url_for('admin_panel'))
    
    return render_template(
        'admin_audit.html', 
        audit_result=audit_result
    )

# Modifikasi admin_panel untuk menambahkan link ke halaman audit
@app.route('/admin')
@admin_required
def admin_panel():
    """Admin panel main page."""
    # Get database statistics
    db_stats = get_database_stats()
    
    # Get list of people in the database
    people_list = get_people_list()
    
    message = session.pop('admin_message', None)
    message_type = session.pop('admin_message_type', 'info')
    
    return render_template(
        'admin.html', 
        db_stats=db_stats, 
        people_list=people_list,
        message=message,
        message_type=message_type
    )

def get_database_stats():
    """Get statistics about the face database."""
    db_folder = app.config['DATABASE_FOLDER']
    
    # Count people (folders in database)
    total_people = 0
    total_images = 0
    
    if os.path.exists(db_folder):
        for person_folder in os.listdir(db_folder):
            person_path = os.path.join(db_folder, person_folder)
            if os.path.isdir(person_path) and person_folder != 'embeddings':
                total_people += 1
                
                # Count images for this person
                for file in os.listdir(person_path):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        total_images += 1
    
    # Get last updated time from embeddings file if it exists
    last_updated = "Never"
    embeddings_path = 'face_embeddings.pkl'
    if os.path.exists(embeddings_path):
        mod_time = os.path.getmtime(embeddings_path)
        last_updated = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M')
    
    return {
        'total_people': total_people,
        'total_images': total_images,
        'last_updated': last_updated
    }

def get_people_list():
    """Get list of people in the database."""
    db_folder = app.config['DATABASE_FOLDER']
    people = []
    
    if os.path.exists(db_folder):
        for person_name in os.listdir(db_folder):
            person_path = os.path.join(db_folder, person_name)
            if os.path.isdir(person_path) and person_name != 'embeddings':
                # Count images
                image_count = len([
                    f for f in os.listdir(person_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ])
                
                people.append({
                    'id': person_name,  # Use folder name as ID
                    'name': person_name,
                    'image_count': image_count
                })
    
    # Sort by name
    people.sort(key=lambda x: x['name'])
    return people

@app.route('/admin/update_embeddings', methods=['POST'])
@admin_required
def admin_update_embeddings():
    """Rebuild face embeddings from the database."""
    try:
        # Use fixed confidence threshold of 0.6
        confidence_threshold = 0.6
            
        # Import face embedding update function from the correct location
        from embedding_manager_utils.update_face_embeddings import update_face_embeddings
        
        # Call the update function with database path and other parameters
        result = update_face_embeddings(
            database_dir=app.config['DATABASE_FOLDER'],
            output_path="face_embeddings.pkl",
            metadata_path="face_embeddings_metadata.pkl",
            confidence_threshold=confidence_threshold
        )
        
        # Create success message with statistics
        message = (
            f"Face embeddings telah diupdate! "
            f"Telah diproses {result['total_persons']} orang, "
            f"{result['new_embeddings']} data wajah baru, "
            f"{result['removed_embeddings']} data wajah yang dihapus"
        )
        
        session['admin_message'] = message
        session['admin_message_type'] = 'success'
    except Exception as e:
        session['admin_message'] = f'Error rebuilding embeddings: {str(e)}'
        session['admin_message_type'] = 'danger'
    
    return redirect(url_for('admin_panel'))

@app.route('/admin/add_person', methods=['POST'])
@admin_required
def admin_add_person():
    """Add a new person to the face database."""
    if 'person_images' not in request.files:
        session['admin_message'] = 'No file uploaded'
        session['admin_message_type'] = 'danger'
        return redirect(url_for('admin_panel'))
    
    file = request.files['person_images']
    person_name = request.form.get('person_name', '').strip()
    
    if not person_name:
        session['admin_message'] = 'Person name is required'
        session['admin_message_type'] = 'danger'
        return redirect(url_for('admin_panel'))
    
    if file.filename == '':
        session['admin_message'] = 'No file selected'
        session['admin_message_type'] = 'danger'
        return redirect(url_for('admin_panel'))
    
    if not file.filename.lower().endswith('.zip'):
        session['admin_message'] = 'Only ZIP files are allowed'
        session['admin_message_type'] = 'danger'
        return redirect(url_for('admin_panel'))
    
    try:
        # Prepare folders
        temp_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_person')
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
        os.makedirs(temp_folder, exist_ok=True)
        
        # Save and extract ZIP
        zip_path = os.path.join(temp_folder, file.filename)
        file.save(zip_path)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_folder)
        
        # Find images in extracted zip
        image_files = []
        for root, _, files in os.walk(temp_folder):
            for filename in files:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(root, filename))
        
        if not image_files:
            session['admin_message'] = 'No image files found in the ZIP'
            session['admin_message_type'] = 'danger'
            shutil.rmtree(temp_folder)
            return redirect(url_for('admin_panel'))
        
        # Create person folder
        person_folder = os.path.join(app.config['DATABASE_FOLDER'], person_name)
        if os.path.exists(person_folder):
            # Append timestamp if folder already exists
            person_name = f"{person_name}_{int(time.time())}"
            person_folder = os.path.join(app.config['DATABASE_FOLDER'], person_name)
        
        os.makedirs(person_folder, exist_ok=True)
        
        # Copy images to person folder
        for img_path in image_files:
            img_filename = os.path.basename(img_path)
            target_path = os.path.join(person_folder, img_filename)
            
            # Ensure unique filename
            if os.path.exists(target_path):
                base_name, ext = os.path.splitext(img_filename)
                target_path = os.path.join(person_folder, f"{base_name}_{int(time.time())}_{random.randint(1000, 9999)}{ext}")
            
            shutil.copy2(img_path, target_path)
        
        # Clean up temp folder
        shutil.rmtree(temp_folder)
        
        session['admin_message'] = f'Successfully added {person_name} with {len(image_files)} images'
        session['admin_message_type'] = 'success'
    except Exception as e:
        session['admin_message'] = f'Error adding person: {str(e)}'
        session['admin_message_type'] = 'danger'
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
    
    return redirect(url_for('admin_panel'))

# Additional routes and functions for person view and image management

@app.route('/admin/view_person/<person_id>')
@admin_required
def admin_view_person(person_id):
    """View images for a specific person."""
    person_folder = os.path.join(app.config['DATABASE_FOLDER'], person_id)
    
    if not os.path.exists(person_folder):
        session['admin_message'] = f'Person {person_id} not found'
        session['admin_message_type'] = 'danger'
        return redirect(url_for('admin_panel'))
    
    # Get all image files
    image_files = []
    for filename in os.listdir(person_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_files.append({
                'filename': filename,
                'url': url_for('admin_serve_image', person_id=person_id, filename=filename)
            })
    
    # Get message from session if exists
    message = session.pop('admin_message', None)
    message_type = session.pop('admin_message_type', 'info')
    
    return render_template(
        'admin_view_person.html', 
        person={"id": person_id, "name": person_id}, 
        images=image_files,
        message=message,
        message_type=message_type
    )

@app.route('/admin/image/<person_id>/<filename>')
@admin_required
def admin_serve_image(person_id, filename):
    """Serve image files for a person."""
    person_folder = os.path.join(app.config['DATABASE_FOLDER'], person_id)
    return send_from_directory(person_folder, filename)

@app.route('/admin/delete_image', methods=['POST'])
@admin_required
def admin_delete_image():
    """Delete a specific image for a person."""
    person_id = request.form.get('person_id')
    filename = request.form.get('filename')
    
    if not person_id or not filename:
        session['admin_message'] = 'Missing required parameters'
        session['admin_message_type'] = 'danger'
        return redirect(url_for('admin_panel'))
    
    image_path = os.path.join(app.config['DATABASE_FOLDER'], person_id, filename)
    
    if not os.path.exists(image_path):
        session['admin_message'] = f'Image not found: {filename}'
        session['admin_message_type'] = 'danger'
        return redirect(url_for('admin_view_person', person_id=person_id))
    
    try:
        # Delete the image file
        os.remove(image_path)
        
        session['admin_message'] = f'Successfully deleted image: {filename}'
        session['admin_message_type'] = 'success'
        
        # Check if this was the last image, if so, maybe prompt to delete the person
        person_folder = os.path.join(app.config['DATABASE_FOLDER'], person_id)
        remaining_images = [f for f in os.listdir(person_folder) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not remaining_images:
            session['admin_message'] = f'Deleted the last image for {person_id}. Consider removing this person.'
            session['admin_message_type'] = 'warning'
            
    except Exception as e:
        session['admin_message'] = f'Error deleting image: {str(e)}'
        session['admin_message_type'] = 'danger'
    
    return redirect(url_for('admin_view_person', person_id=person_id))

@app.route('/admin/delete_person', methods=['POST'])
@admin_required
def admin_delete_person():
    """Delete a person from the database."""
    person_id = request.form.get('person_id')
    
    if not person_id:
        session['admin_message'] = 'Person ID is required'
        session['admin_message_type'] = 'danger'
        return redirect(url_for('admin_panel'))
    
    person_folder = os.path.join(app.config['DATABASE_FOLDER'], person_id)
    
    if not os.path.exists(person_folder):
        session['admin_message'] = f'Person {person_id} not found'
        session['admin_message_type'] = 'danger'
        return redirect(url_for('admin_panel'))
    
    try:
        # Delete person folder
        shutil.rmtree(person_folder)
        
        session['admin_message'] = f'Successfully deleted {person_id}'
        session['admin_message_type'] = 'success'
    except Exception as e:
        session['admin_message'] = f'Error deleting person: {str(e)}'
        session['admin_message_type'] = 'danger'
    
    return redirect(url_for('admin_panel'))

@app.route('/admin/add_photos', methods=['POST'])
@admin_required
def admin_add_photos():
    """Add additional photos to an existing person."""
    person_id = request.form.get('person_id')
    
    if not person_id:
        session['admin_message'] = 'Person ID is required'
        session['admin_message_type'] = 'danger'
        return redirect(url_for('admin_panel'))
    
    if 'new_photos' not in request.files:
        session['admin_message'] = 'No files uploaded'
        session['admin_message_type'] = 'danger'
        return redirect(url_for('admin_view_person', person_id=person_id))
    
    # Get all files (multiple file upload)
    files = request.files.getlist('new_photos')
    
    if not files or files[0].filename == '':
        session['admin_message'] = 'No files selected'
        session['admin_message_type'] = 'danger'
        return redirect(url_for('admin_view_person', person_id=person_id))
    
    # Validate each file is an allowed image type
    valid_extensions = ['.jpg', '.jpeg', '.png']
    for file in files:
        if not any(file.filename.lower().endswith(ext) for ext in valid_extensions):
            session['admin_message'] = 'Only JPG, JPEG, and PNG files are allowed'
            session['admin_message_type'] = 'danger'
            return redirect(url_for('admin_view_person', person_id=person_id))
    
    # Save files
    person_folder = os.path.join(app.config['DATABASE_FOLDER'], person_id)
    
    if not os.path.exists(person_folder):
        os.makedirs(person_folder, exist_ok=True)
    
    files_saved = 0
    for file in files:
        if file and file.filename.strip():  # Additional check to avoid empty filenames
            try:
                # Secure the filename
                filename = secure_filename(file.filename)
                
                # Ensure unique filename
                base_name, ext = os.path.splitext(filename)
                final_filename = filename
                counter = 1
                
                # If file exists, add a counter to make it unique
                while os.path.exists(os.path.join(person_folder, final_filename)):
                    final_filename = f"{base_name}_{counter}{ext}"
                    counter += 1
                
                # Save the file
                file.save(os.path.join(person_folder, final_filename))
                files_saved += 1
                
            except Exception as e:
                print(f"Error saving file {file.filename}: {e}")
    
    if files_saved > 0:
        session['admin_message'] = f'Successfully added {files_saved} new photos for {person_id}'
        session['admin_message_type'] = 'success'
    else:
        session['admin_message'] = 'No files were saved due to errors'
        session['admin_message_type'] = 'warning'
    
    return redirect(url_for('admin_view_person', person_id=person_id))


@app.route('/admin/edit_person_name', methods=['POST'])
@admin_required
def admin_edit_person_name():
    """Edit the name of a person (rename folder)."""
    person_id = request.form.get('person_id')
    new_name = request.form.get('new_name', '').strip()
    
    if not person_id:
        session['admin_message'] = 'Person ID is required'
        session['admin_message_type'] = 'danger'
        return redirect(url_for('admin_panel'))
    
    if not new_name:
        session['admin_message'] = 'New name is required'
        session['admin_message_type'] = 'danger'
        return redirect(url_for('admin_panel'))
    
    old_folder_path = os.path.join(app.config['DATABASE_FOLDER'], person_id)
    new_folder_path = os.path.join(app.config['DATABASE_FOLDER'], new_name)
    
    # Check if old folder exists
    if not os.path.exists(old_folder_path):
        session['admin_message'] = f'Person {person_id} not found in database'
        session['admin_message_type'] = 'danger'
        return redirect(url_for('admin_panel'))
    
    # Check if new folder name already exists
    if os.path.exists(new_folder_path) and person_id != new_name:
        session['admin_message'] = f'A person with name {new_name} already exists in database'
        session['admin_message_type'] = 'danger'
        return redirect(url_for('admin_panel'))
    
    try:
        # If the name hasn't changed, don't do anything
        if person_id != new_name:
            # Rename folder
            os.rename(old_folder_path, new_folder_path)
            
            session['admin_message'] = f'Successfully renamed {person_id} to {new_name}'
            session['admin_message_type'] = 'success'
        else:
            session['admin_message'] = 'No changes made to name'
            session['admin_message_type'] = 'info'
    except Exception as e:
        session['admin_message'] = f'Error renaming person: {str(e)}'
        session['admin_message_type'] = 'danger'
    
    return redirect(url_for('admin_panel'))

@app.route('/admin/reprocess_embeddings', methods=['POST'])
@admin_required
def admin_reprocess_embeddings():
    """Reprocess face embeddings with specific confidence threshold to improve detection on problematic images."""
    try:
        # Get confidence threshold from form
        confidence_threshold = float(request.form.get('confidence_threshold', 0.5))
        
        # Validate range
        if not 0.3 <= confidence_threshold <= 0.7:
            session['admin_message'] = 'Error: Confidence threshold must be between 0.3 and 0.7'
            session['admin_message_type'] = 'danger'
            return redirect(url_for('admin_panel'))
        
        # Import the correct module
        from embedding_manager_utils.reprocess_face_embeddings import reprocess_problem_faces
        
        # Call the reprocess function with confidence threshold
        result = reprocess_problem_faces(
            database_dir=app.config['DATABASE_FOLDER'],
            embeddings_path="face_embeddings.pkl",
            metadata_path="face_embeddings_metadata.pkl",
            confidence_threshold=confidence_threshold
        )
        
        # Check if operation was successful
        if result['status'] == 'success':
            # Create success message with statistics
            message = (
                f"Face embeddings telah diproses ulang! "
                f"{result.get('total_problematic_files', 0)} Total file foto bermasalah telah di proses ulang"
            )
            session['admin_message'] = message
            session['admin_message_type'] = 'success'
        else:
            # Handle error case
            session['admin_message'] = f"Error: {result.get('message', 'Unknown error')}"
            session['admin_message_type'] = 'danger'
            
    except Exception as e:
        session['admin_message'] = f'Error reprocessing embeddings: {str(e)}'
        session['admin_message_type'] = 'danger'
    
    return redirect(url_for('admin_panel'))

# --- Main Program ---
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)
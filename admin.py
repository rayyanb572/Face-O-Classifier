import os
import shutil
import zipfile
import pickle
import hashlib
import time
import random
from datetime import datetime
from collections import defaultdict
from functools import wraps
from flask import Blueprint, render_template, request, redirect, url_for, session, flash, send_from_directory, current_app
from werkzeug.utils import secure_filename
import config

# Create Blueprint
admin_bp = Blueprint('admin', __name__, template_folder='templates')

# --- Helper Functions ---
def auto_update_embeddings():
    """Automatically update face embeddings after database changes."""
    try:
        # Use fixed confidence threshold of 0.6
        confidence_threshold = 0.6
           
        # Import face embedding update function from the correct location
        from embedding_manager_utils.update_face_embeddings import update_face_embeddings
       
        # Call the update function with database path and other parameters
        result = update_face_embeddings(
            database_dir=current_app.config['DATABASE_FOLDER'],
            output_path="face_embeddings.pkl",
            metadata_path="face_embeddings_metadata.pkl",
            confidence_threshold=confidence_threshold
        )
       
        return {
            'success': True,
            'message': ""
        }
    except Exception as e:
        return {
            'success': False,
            'message': f'Error saat updating embeddings: {str(e)}'
        }

def update_session_message(base_message, auto_update_result, message_type='success'):
    """Update session message with auto-update result."""
    if auto_update_result['success']:
        full_message = f"{base_message}. {auto_update_result['message']}"
        session['admin_message'] = full_message
        session['admin_message_type'] = message_type
    else:
        full_message = f"{base_message}. WARNING: {auto_update_result['message']}"
        session['admin_message'] = full_message
        session['admin_message_type'] = 'warning'

# --- Admin Authentication Routes ---
@admin_bp.route('/login', methods=['POST'])
def admin_login():
    """Handle admin login."""
    username = request.form.get('username')
    password = request.form.get('password')
    
    # Hash the password
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    
    if username == config.ADMIN_USERNAME and hashed_password == config.ADMIN_PASSWORD:
        session['admin_logged_in'] = True
        return redirect(url_for('admin.admin_panel'))
    else:
        flash('Invalid username or password', 'danger')
        # Redirect back to index with a login_error parameter
        return redirect(url_for('index', login_error='true'))

def admin_required(func):
    """Decorator to require admin login."""
    @wraps(func)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return redirect(url_for('index'))
        return func(*args, **kwargs)
    return decorated_function

# Tambahkan fungsi untuk audit database
def audit_face_database():
    """Mengaudit database wajah untuk mendeteksi perbedaan antara file foto dan embeddings."""
    
    # Path ke file embeddings dan direktori database
    pickle_path = 'face_embeddings.pkl'  # File berada di root direktori
    metadata_path = 'face_embeddings_metadata.pkl'  # File berada di root direktori
    database_dir = current_app.config['DATABASE_FOLDER']
    
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
@admin_bp.route('/audit')
@admin_required
def admin_audit():
    """Halaman untuk mengaudit database wajah."""
    # Jalankan audit
    audit_result = audit_face_database()
    
    if audit_result.get('error'):
        session['admin_message'] = audit_result.get('message')
        session['admin_message_type'] = 'danger'
        return redirect(url_for('admin.admin_panel'))
    
    return render_template(
        'admin_audit.html', 
        audit_result=audit_result
    )

# Modifikasi admin_panel untuk menambahkan link ke halaman audit
@admin_bp.route('/')
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
    db_folder = current_app.config['DATABASE_FOLDER']
    
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
    db_folder = current_app.config['DATABASE_FOLDER']
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

@admin_bp.route('/update_embeddings', methods=['POST'])
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
            database_dir=current_app.config['DATABASE_FOLDER'],
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
    
    return redirect(url_for('admin.admin_panel'))

@admin_bp.route('/add_person', methods=['POST'])
@admin_required
def admin_add_person():
    """Add a new person to the face database."""
    if 'person_images' not in request.files:
        session['admin_message'] = 'No file uploaded'
        session['admin_message_type'] = 'danger'
        return redirect(url_for('admin.admin_panel'))
    
    file = request.files['person_images']
    person_name = request.form.get('person_name', '').strip()
    
    if not person_name:
        session['admin_message'] = 'Person name is required'
        session['admin_message_type'] = 'danger'
        return redirect(url_for('admin.admin_panel'))
    
    if file.filename == '':
        session['admin_message'] = 'No file selected'
        session['admin_message_type'] = 'danger'
        return redirect(url_for('admin.admin_panel'))
    
    if not file.filename.lower().endswith('.zip'):
        session['admin_message'] = 'Only ZIP files are allowed'
        session['admin_message_type'] = 'danger'
        return redirect(url_for('admin.admin_panel'))
    
    try:
        # Prepare folders
        temp_folder = os.path.join(current_app.config['UPLOAD_FOLDER'], 'temp_person')
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
            return redirect(url_for('admin.admin_panel'))
        
        # Create person folder
        person_folder = os.path.join(current_app.config['DATABASE_FOLDER'], person_name)
        if os.path.exists(person_folder):
            # Append timestamp if folder already exists
            person_name = f"{person_name}_{int(time.time())}"
            person_folder = os.path.join(current_app.config['DATABASE_FOLDER'], person_name)
        
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
        
        # AUTO-UPDATE EMBEDDINGS
        base_message = f'Berhasil menambahkan {person_name} dengan {len(image_files)} foto'
        auto_update_result = auto_update_embeddings()
        update_session_message(base_message, auto_update_result, 'success')
        
    except Exception as e:
        session['admin_message'] = f'Error adding person: {str(e)}'
        session['admin_message_type'] = 'danger'
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
    
    return redirect(url_for('admin.admin_panel'))

# Additional routes and functions for person view and image management

@admin_bp.route('/view_person/<person_id>')
@admin_required
def admin_view_person(person_id):
    """View images for a specific person."""
    person_folder = os.path.join(current_app.config['DATABASE_FOLDER'], person_id)
    
    if not os.path.exists(person_folder):
        session['admin_message'] = f'Person {person_id} not found'
        session['admin_message_type'] = 'danger'
        return redirect(url_for('admin.admin_panel'))
    
    # Get all image files
    image_files = []
    for filename in os.listdir(person_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_files.append({
                'filename': filename,
                'url': url_for('admin.admin_serve_image', person_id=person_id, filename=filename)
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

@admin_bp.route('/image/<person_id>/<filename>')
@admin_required
def admin_serve_image(person_id, filename):
    """Serve image files for a person."""
    person_folder = os.path.join(current_app.config['DATABASE_FOLDER'], person_id)
    return send_from_directory(person_folder, filename)

@admin_bp.route('/delete_image', methods=['POST'])
@admin_required
def admin_delete_image():
    """Delete a specific image for a person."""
    person_id = request.form.get('person_id')
    filename = request.form.get('filename')
    
    if not person_id or not filename:
        session['admin_message'] = 'Missing required parameters'
        session['admin_message_type'] = 'danger'
        return redirect(url_for('admin.admin_panel'))
    
    image_path = os.path.join(current_app.config['DATABASE_FOLDER'], person_id, filename)
    
    if not os.path.exists(image_path):
        session['admin_message'] = f'Image not found: {filename}'
        session['admin_message_type'] = 'danger'
        return redirect(url_for('admin.admin_view_person', person_id=person_id))
    
    try:
        # Delete the image file
        os.remove(image_path)
        
        # Check if this was the last image
        person_folder = os.path.join(current_app.config['DATABASE_FOLDER'], person_id)
        remaining_images = [f for f in os.listdir(person_folder) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not remaining_images:
            base_message = f'Deleted the last image for {person_id}. Consider removing this person.'
            # Don't auto-update embeddings if no images remain
            session['admin_message'] = base_message
            session['admin_message_type'] = 'warning'
        else:
            # AUTO-UPDATE EMBEDDINGS
            base_message = f'Berhasil menghapus gambar: {filename}'
            auto_update_result = auto_update_embeddings()
            update_session_message(base_message, auto_update_result, 'success')
            
    except Exception as e:
        session['admin_message'] = f'Error deleting image: {str(e)}'
        session['admin_message_type'] = 'danger'
    
    return redirect(url_for('admin.admin_view_person', person_id=person_id))

@admin_bp.route('/delete_person', methods=['POST'])
@admin_required
def admin_delete_person():
    """Delete a person from the database."""
    person_id = request.form.get('person_id')
    
    if not person_id:
        session['admin_message'] = 'Person ID is required'
        session['admin_message_type'] = 'danger'
        return redirect(url_for('admin.admin_panel'))
    
    person_folder = os.path.join(current_app.config['DATABASE_FOLDER'], person_id)
    
    if not os.path.exists(person_folder):
        session['admin_message'] = f'Person {person_id} not found'
        session['admin_message_type'] = 'danger'
        return redirect(url_for('admin.admin_panel'))
    
    try:
        # Delete person folder
        shutil.rmtree(person_folder)
        
        # AUTO-UPDATE EMBEDDINGS
        base_message = f'Berhasil menghapus data {person_id}'
        auto_update_result = auto_update_embeddings()
        update_session_message(base_message, auto_update_result, 'success')
        
    except Exception as e:
        session['admin_message'] = f'Error deleting person: {str(e)}'
        session['admin_message_type'] = 'danger'
    
    return redirect(url_for('admin.admin_panel'))

@admin_bp.route('/add_photos', methods=['POST'])
@admin_required
def admin_add_photos():
    """Add additional photos to an existing person."""
    person_id = request.form.get('person_id')
    
    if not person_id:
        session['admin_message'] = 'Person ID is required'
        session['admin_message_type'] = 'danger'
        return redirect(url_for('admin.admin_panel'))
    
    if 'new_photos' not in request.files:
        session['admin_message'] = 'Tidak ada file yang diupload'
        session['admin_message_type'] = 'danger'
        return redirect(url_for('admin.admin_view_person', person_id=person_id))
    
    # Get all files (multiple file upload)
    files = request.files.getlist('new_photos')
    
    if not files or files[0].filename == '':
        session['admin_message'] = 'Tidak ada file yang dipilih'
        session['admin_message_type'] = 'danger'
        return redirect(url_for('admin.admin_view_person', person_id=person_id))
    
    # Validate each file is an allowed image type
    valid_extensions = ['.jpg', '.jpeg', '.png']
    for file in files:
        if not any(file.filename.lower().endswith(ext) for ext in valid_extensions):
            session['admin_message'] = 'Only JPG, JPEG, and PNG files are allowed'
            session['admin_message_type'] = 'danger'
            return redirect(url_for('admin.admin_view_person', person_id=person_id))
    
    # Save files
    person_folder = os.path.join(current_app.config['DATABASE_FOLDER'], person_id)
    
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
        # AUTO-UPDATE EMBEDDINGS
        base_message = f'Berhasil menambahkan {files_saved} foto baru untuk {person_id}'
        auto_update_result = auto_update_embeddings()
        update_session_message(base_message, auto_update_result, 'success')
    else:
        session['admin_message'] = 'No files were saved due to errors'
        session['admin_message_type'] = 'warning'
    
    return redirect(url_for('admin.admin_view_person', person_id=person_id))


@admin_bp.route('/edit_person_name', methods=['POST'])
@admin_required
def admin_edit_person_name():
    """Edit the name of a person (rename folder)."""
    person_id = request.form.get('person_id')
    new_name = request.form.get('new_name', '').strip()
    
    if not person_id:
        session['admin_message'] = 'Person ID is required'
        session['admin_message_type'] = 'danger'
        return redirect(url_for('admin.admin_panel'))
    
    if not new_name:
        session['admin_message'] = 'New name is required'
        session['admin_message_type'] = 'danger'
        return redirect(url_for('admin.admin_panel'))
    
    old_folder_path = os.path.join(current_app.config['DATABASE_FOLDER'], person_id)
    new_folder_path = os.path.join(current_app.config['DATABASE_FOLDER'], new_name)
    
    # Check if old folder exists
    if not os.path.exists(old_folder_path):
        session['admin_message'] = f'Person {person_id} not found in database'
        session['admin_message_type'] = 'danger'
        return redirect(url_for('admin.admin_panel'))
    
    # Check if new folder name already exists
    if os.path.exists(new_folder_path) and person_id != new_name:
        session['admin_message'] = f'A person with name {new_name} already exists in database'
        session['admin_message_type'] = 'danger'
        return redirect(url_for('admin.admin_panel'))
    
    try:
        # If the name hasn't changed, don't do anything
        if person_id != new_name:
            # Rename folder
            os.rename(old_folder_path, new_folder_path)
            
            # AUTO-UPDATE EMBEDDINGS
            base_message = f'Berhasil rename {person_id} ke {new_name}'
            auto_update_result = auto_update_embeddings()
            update_session_message(base_message, auto_update_result, 'success')
        else:
            session['admin_message'] = 'Tidak ada perubahan yang dilakukan pada nama'
            session['admin_message_type'] = 'info'
    except Exception as e:
        session['admin_message'] = f'Error renaming person: {str(e)}'
        session['admin_message_type'] = 'danger'
    
    return redirect(url_for('admin.admin_panel'))

@admin_bp.route('/reprocess_embeddings', methods=['POST'])
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
            return redirect(url_for('admin.admin_panel'))
        
        # Import the correct module
        from embedding_manager_utils.reprocess_face_embeddings import reprocess_problem_faces
        
        # Call the reprocess function with confidence threshold
        result = reprocess_problem_faces(
            database_dir=current_app.config['DATABASE_FOLDER'],
            embeddings_path="face_embeddings.pkl",
            metadata_path="face_embeddings_metadata.pkl",
            confidence_threshold=confidence_threshold
        )
        
        # Check if operation was successful
        if result['status'] == 'success':
            # Create success message with statistics
            message = (
                f"Face embeddings telah diproses ulang! "
                f"{result.get('total_problematic_files', 0)} total file foto bermasalah telah di proses ulang"
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
    
    return redirect(url_for('admin.admin_panel'))
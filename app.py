from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session, jsonify
import os
import shutil
import glob
import requests
from werkzeug.utils import secure_filename
from classify_faces import classify_faces
from flask_session import Session
from google_drive_downloader import GoogleDriveDownloader as gdd

# Hapus semua session saat aplikasi dimulai
session_files = glob.glob('flask_session/*')
for file in session_files:
    os.remove(file)

app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = 'flask_session'
app.config['SESSION_PERMANENT'] = False
Session(app)
app.secret_key = "your_secret_key"

UPLOAD_FOLDER = '/tmp/uploads'
OUTPUT_FOLDER = '/tmp/output_test'
DOWNLOAD_FOLDER = '/tmp/downloads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)

@app.route('/')
def index():
    if not session.get('upload_complete'):
        session.clear()
    return render_template('index.html', 
                           output_path=session.get('output_path'), 
                           original_folder_name=session.get('original_folder_name'))

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'folder' not in request.files:
        return redirect(request.url)
    
    folder = request.files.getlist('folder')
    if not folder:
        return redirect(request.url)
    
    session.clear()
    session.modified = True
    clear_folder(app.config['UPLOAD_FOLDER'])
    
    first_file_path = folder[0].filename
    uploaded_folder_name = os.path.dirname(first_file_path) or 'uploaded_folder'
    secure_folder_name = secure_filename(uploaded_folder_name)
    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    for file in folder:
        if file.filename:
            filename = secure_filename(os.path.basename(file.filename))
            file.save(os.path.join(folder_path, filename))
    
    session['original_folder_name'] = uploaded_folder_name
    session['folder_name'] = secure_folder_name
    
    output_path = classify_faces(folder_path)
    session['output_path'] = output_path
    session['upload_complete'] = True
    
    return redirect(url_for('index'))

@app.route('/download_gdrive', methods=['POST'])
def download_gdrive():
    data = request.json
    gdrive_link = data.get('gdrive_link')
    
    if not gdrive_link:
        return jsonify({'error': 'Google Drive link is required'}), 400
    
    file_id = gdrive_link.split('/d/')[1].split('/')[0] if '/d/' in gdrive_link else None
    if not file_id:
        return jsonify({'error': 'Invalid Google Drive link'}), 400
    
    clear_folder(app.config['DOWNLOAD_FOLDER'])
    download_path = os.path.join(app.config['DOWNLOAD_FOLDER'], 'downloaded_folder.zip')
    gdd.download_file_from_google_drive(file_id=file_id, dest_path=download_path, unzip=True)
    
    extracted_folder = os.path.join(app.config['DOWNLOAD_FOLDER'], 'extracted')
    os.makedirs(extracted_folder, exist_ok=True)
    
    output_path = classify_faces(extracted_folder)
    session['output_path'] = output_path
    session['upload_complete'] = True
    
    return jsonify({'message': 'Download and processing complete', 'output_path': output_path})

@app.route('/preview')
def preview():
    visualized_folder = os.path.join(app.config['OUTPUT_FOLDER'], 'visualized')
    images = os.listdir(visualized_folder)
    images = [url_for('send_image', filename=f'visualized/{img}') for img in images]
    return render_template('preview.html', images=images)

@app.route('/send_image/<path:filename>')
def send_image(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/download_result')
def download_result():
    output_path = session.get('output_path')
    if output_path and os.path.exists(output_path):
        shutil.make_archive(output_path, 'zip', output_path)
        return send_from_directory(os.path.dirname(output_path), os.path.basename(output_path) + '.zip', as_attachment=True)
    return jsonify({'error': 'No output available'}), 404

@app.route('/open_output')
def open_output():
    output_path = session.get('output_path')
    if output_path and os.path.exists(output_path):
        try:
            os.startfile(output_path)
        except Exception:
            os.system(f'xdg-open "{output_path}"')
    return '', 204

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)

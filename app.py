from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
import os
import shutil
from werkzeug.utils import secure_filename
from classify_faces import classify_faces
from flask_session import Session
import glob

# Hapus semua session yang tersimpan saat aplikasi baru dijalankan
session_files = glob.glob('flask_session/*')
for file in session_files:
    os.remove(file)

app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'  # Simpan session di file
app.config['SESSION_FILE_DIR'] = 'flask_session'  # Folder penyimpanan session
app.config['SESSION_PERMANENT'] = False  # Non-permanen agar mudah dihapus
Session(app)
app.secret_key = "your_secret_key"  # Kunci untuk sesi

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output_test'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

def clear_folder(folder_path):
    """Menghapus folder lama dan membuat ulang folder kosong"""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)

@app.route('/')
def index():
    if not session.get('upload_complete'):  # Jika belum ada proses upload
        session.clear()  # Reset session agar halaman awal bersih

    return render_template('index.html', 
                       output_path=session.get('output_path'), 
                       original_folder_name=session.get('original_folder_name'))  # ✅ Kirim variabel ke template

@app.route('/upload', methods=['POST'])
def upload_file():
    """Menghandle upload folder dan melakukan klasifikasi wajah"""
    if 'folder' not in request.files:
        return redirect(request.url)

    folder = request.files.getlist('folder')
    if not folder:
        return redirect(request.url)

    # 🔄 Reset session sebelum memproses folder baru
    session.clear()
    session.modified = True

    # Menghapus folder sebelumnya
    clear_folder(app.config['UPLOAD_FOLDER'])

    first_file_path = folder[0].filename
    uploaded_folder_name = os.path.dirname(first_file_path) or 'uploaded_folder'
    secure_folder_name = secure_filename(uploaded_folder_name)  # Nama aman untuk penyimpanan
    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_folder_name)
    os.makedirs(folder_path, exist_ok=True)

    for file in folder:
        if file.filename:
            filename = secure_filename(os.path.basename(file.filename))
            file.save(os.path.join(folder_path, filename))

    # ✅ Simpan nama asli dan nama aman di session
    session['original_folder_name'] = uploaded_folder_name  # Menyimpan nama asli
    session['folder_name'] = secure_folder_name  # Nama aman untuk penyimpanan

    # Proses klasifikasi wajah
    output_path = classify_faces(folder_path)

    # ✅ Simpan hasil proses di session setelah klasifikasi selesai
    session['output_path'] = output_path
    session['upload_complete'] = True

    return redirect(url_for('index'))

@app.route('/preview')
def preview():
    visualized_folder = os.path.join(app.config['OUTPUT_FOLDER'], 'visualized')
    images = os.listdir(visualized_folder)
    images = [url_for('send_image', filename=f'visualized/{img}') for img in images]
    return render_template('preview.html', images=images)

@app.route('/send_image/<path:filename>')
def send_image(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/open_output')
def open_output():
    output_path = session.get('output_path')
    if output_path and os.path.exists(output_path):
        try:
            os.startfile(output_path)  # Windows
        except Exception:
            os.system(f'xdg-open "{output_path}"')  # Linux/macOS
    return '', 204  # ✅ Tidak mereset halaman (No Content)

if __name__ == "__main__":
    app.run(debug=False, host='127.0.0.1', port=5000)

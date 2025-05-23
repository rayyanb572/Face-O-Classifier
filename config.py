import hashlib

# --- Admin Configuration ---
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = hashlib.sha256("admin123".encode()).hexdigest()  # Default password: admin123

# --- Flask Configuration ---
SECRET_KEY = "your_secret_key"
SESSION_TYPE = 'filesystem'
SESSION_FILE_DIR = 'flask_session'
SESSION_PERMANENT = False

# --- Folder Configuration ---
UPLOAD_FOLDER = 'uploads'
ZIP_FOLDER = 'zip'
DATABASE_FOLDER = 'database'

# --- File Extensions ---
ALLOWED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']
ALLOWED_ZIP_EXTENSIONS = ['.zip']
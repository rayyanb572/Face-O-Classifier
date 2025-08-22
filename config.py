import hashlib
import os
import sys
from dotenv import load_dotenv

# Load env
load_dotenv()

# --- Admin Configuration ---
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_PASSWORD_RAW = os.getenv("ADMIN_PASSWORD")
SECRET_KEY = os.getenv("SECRET_KEY")

# Hash password
ADMIN_PASSWORD = hashlib.sha256(ADMIN_PASSWORD_RAW.encode()).hexdigest()

# --- Flask Configuration ---
SESSION_TYPE = os.getenv("SESSION_TYPE", "filesystem")
SESSION_FILE_DIR = os.getenv("SESSION_FILE_DIR", "flask_session")
SESSION_PERMANENT = os.getenv("SESSION_PERMANENT", "False").lower() == "true"

# --- Folder Configuration ---
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
ZIP_FOLDER = os.getenv("ZIP_FOLDER", "zip")
DATABASE_FOLDER = os.getenv("DATABASE_FOLDER", "database")

# --- File Extensions ---
ALLOWED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']
ALLOWED_ZIP_EXTENSIONS = ['.zip']
"""
Configuration and environment settings
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Model paths
DEFAULT_MODEL_PATH = 'best_medicine_model.pth'
DEFAULT_LABELS_PATH = 'label_encoder_classes.npy'

# Flask configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

# Server configuration
DEFAULT_PORT = 8080
DEFAULT_HOST = '0.0.0.0'

# Output paths
DEFAULT_OUTPUT_CSV = 'predictions.csv'
DEFAULT_PRESCRIPTION_OUTPUT = 'prescription_analysis.json'

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global label classes (will be set by model loader)
label_classes = []
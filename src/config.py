import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# App configuration
APP_NAME = "Object Detection API"
VERSION = "1.0.0"

# Security configuration
SECRET_KEY_TOKEN = os.getenv("SECRET_KEY_TOKEN", "default-secret-key")

# Model configuration
MODEL_PATH = os.getenv("MODEL_PATH", "model/best.pt")  # Update with your model path

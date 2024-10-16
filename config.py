import logging
import os

# Constants
MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Check for required environment variables
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
    raise ValueError("AWS credentials not set.")

# Logging configuration
logging.basicConfig(level=logging.INFO)

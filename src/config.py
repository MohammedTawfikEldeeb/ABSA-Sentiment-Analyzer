from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()
project_root = Path(__file__).parent.parent

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
EXTRACTED_DATA_DIR = project_root / 'data' / 'raw'
PROCESSED_DATA_DIR = project_root / 'data' / 'processed'
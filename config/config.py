import os
from dotenv import load_dotenv

# Load environment variables from .env file only if they are not already set
load_dotenv(override=False)

# Force fetch from system environment (ECS task definition)
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is missing. Check ECS task env vars or .env file.")

# Constants
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DB_FAISS_PATH = "vectorstore/db_faiss"
DATA_PATH = "data/"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

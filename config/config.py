import os

# Load from environment (ECS will inject at runtime)
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN is not set. Please check ECS Task definition environment variables.")

# Constants
HUGGINGFACE_REPO_ID = "google/flan-t5-small"
DB_FAISS_PATH = "vectorstore/db_faiss"
DATA_PATH = "data/"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

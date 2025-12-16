import os
import pickle
from langchain_core.documents import Document

def load_cached_chunks(path: str = "data/chunks.pkl"):
    # Resolve relative to this file's directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"No cached chunks at {full_path}. Run preprocess.py once.")

    with open(full_path, "rb") as f:
        chunks = pickle.load(f)
    return chunks

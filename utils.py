import os
import pickle
from langchain_core.documents import Document

def load_cached_chunks(path: str = "data/chunks.pkl"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No cached chunks at {path}. Run preprocess.py once.")
    with open(path, "rb") as f:
        chunks = pickle.load(f)
    # If you saved list[Document] directly, this is already fine.
    # If you saved dicts, convert to Document here.
    return chunks

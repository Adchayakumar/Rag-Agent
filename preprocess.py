
import os, pickle
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


CORPUS_DIR = "corpus"  # put your 4 PDFs here


def load_pdfs(corpus_dir: str = CORPUS_DIR):
    """Load all PDF files from the corpus directory into LangChain Documents."""
    docs = []
    for fname in os.listdir(corpus_dir):
        if fname.lower().endswith(".pdf"):
            path = os.path.join(corpus_dir, fname)
            loader = PyPDFLoader(path)
            pdf_docs = loader.load()
            # add basic metadata so we know source file
            for d in pdf_docs:
                d.metadata.setdefault("source", path)
            docs.extend(pdf_docs)
    return docs


def clean_text(text: str) -> str:
    """Basic text cleaning: strip whitespace and normalize spaces."""
    # strip leading/trailing whitespace
    text = text.strip()
    # collapse multiple newlines/spaces
    lines = [line.strip() for line in text.splitlines()]
    text = " ".join(line for line in lines if line)
    return text


def preprocess_documents(corpus_dir: str = CORPUS_DIR):
    """
    Load PDFs, clean their text, and split into chunks.
    Returns a list of cleaned, chunked LangChain Documents.
    """
    raw_docs = load_pdfs(corpus_dir)

    # Clean page_content
    for d in raw_docs:
        d.page_content = clean_text(d.page_content)

    # Remove empty docs
    raw_docs = [d for d in raw_docs if d.page_content]

    # Chunking for RAG
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = text_splitter.split_documents(raw_docs)
    return chunks


if __name__ == "__main__":
    chunks = preprocess_documents()
    print(f"Loaded and chunked {len(chunks)} chunks from PDFs in '{CORPUS_DIR}'.")
    # print a sample
    for i, c in enumerate(chunks[:3], start=1):
        print(f"\n--- Chunk {i} ---")
        print(c.page_content[:300], "...")
        print("Metadata:", c.metadata)
   

    os.makedirs("data", exist_ok=True)
    
    with open("data/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

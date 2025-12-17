# summarize.py

import os
from typing import List

import google.generativeai as genai
from langchain_core.documents import Document

from retrieval import HybridRetriever
from preprocess import preprocess_documents

# ---------- LLM CONFIG (Gemini) ----------


try:
    import streamlit as st
    _secret_key = st.secrets.get("GEMINI_API_KEY", None)
except Exception:
    _secret_key = None

GENAI_API_KEY = _secret_key or os.environ.get("GEMINI_API_KEY")

if not GENAI_API_KEY:
    raise ValueError("Please set GEMINI_API_KEY environment variable or Streamlit secret.")

genai.configure(api_key=GENAI_API_KEY)
MODEL_NAME = "gemini-2.0-flash"



def llm_summarize(prompt: str) -> str:
    """Call Gemini with a simple text prompt and return text."""
    model = genai.GenerativeModel(MODEL_NAME)
    resp = model.generate_content(prompt)
    return resp.text or ""


# ---------- SUMMARIZATION LOGIC ----------

def build_summary_prompt(docs: List[Document], length: str = "short") -> str:
    """
    Build a prompt to summarize the content of docs.
    length: 'short', 'medium', or 'long'
    """
    if length == "short":
        length_instr = "in 2–3 sentences"
    elif length == "medium":
        length_instr = "in one concise paragraph"
    else:
        length_instr = "in 2–3 short paragraphs"

    joined = ""
    for i, d in enumerate(docs, start=1):
        joined += f"\n[DOC {i}]\n{d.page_content}\n"

    prompt = f"""
You are a helpful assistant that summarizes documents.

You will be given several document chunks. Read them carefully
and produce a single coherent summary that captures the main ideas
and key details, {length_instr}.

Important:
- Base the summary ONLY on the provided text.
- Do NOT add information that is not present in the documents.
- Do NOT mention 'DOC 1' / 'DOC 2' or similar; just write the summary.

DOCUMENTS:
{joined}
"""
    return prompt.strip()


def summarize_docs(docs: List[Document], length: str = "short") -> str:
    """Summarize the given documents with the desired length."""
    if not docs:
        return "No documents were provided for summarization."

    prompt = build_summary_prompt(docs, length=length)
    summary = llm_summarize(prompt)
    return summary.strip()


# ---------- DEMO: SEARCH + SUMMARIZE PIPELINE ----------

if __name__ == "__main__":
    # 1. Preprocess and build retriever
    chunks = preprocess_documents()
    retriever = HybridRetriever(chunks)

    query = "What are the main topics discussed in these documents?"
    print(f"Query: {query}")

    # 2. Retrieve top-k relevant chunks (hybrid search)
    top_docs_with_scores = retriever.search_hybrid(query, k=5)
    top_docs = [d for d, _ in top_docs_with_scores]

    # 3. Summarize them
    for length in ["short", "medium", "long"]:
        print(f"\n--- {length.upper()} SUMMARY ---")
        summary = summarize_docs(top_docs, length=length)
        print(summary)

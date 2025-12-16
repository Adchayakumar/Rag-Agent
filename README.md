

# Document Search and Summarization Using LLMs

A small, end‑to‑end system for document search and summarization over a PDF corpus using:

- Traditional information retrieval (TF‑IDF)  
- Dense LLM embeddings (HuggingFace + Chroma)  
- LLM‑based summarization (Gemini) with controllable summary length  
- Offline search evaluation via `evaluate_search.py`  
- A simple Streamlit UI for interactive use  

***

## 1. Goals and High‑Level Design

### Objective

The system is designed to:

1. Ingest and preprocess a small corpus of PDFs.  
2. Search the corpus using both traditional IR (TF‑IDF) and embeddings.  
3. Summarize the most relevant chunks using an LLM.  
4. Evaluate search accuracy on a small, labeled test set.  
5. Optionally provide a Streamlit UI for querying and summarizing.

### Main Components

- Data preparation: `preprocess.py`  
- Retrieval (TF‑IDF + embeddings): `retrieval.py`  
- Summarization (Gemini): `summarize.py`  
- Search evaluation: `evaluate_search.py`  
- User interface (Streamlit): `app.py`  

***

## 2. Corpus and Data Preparation (`preprocess.py`)

### 2.1 Corpus

Place your four PDFs under the `corpus/` directory:

- `What-is-machine-learning.pdf`  
- `What-is-deep-learning.pdf`  
- `What-is-LLM-fine-tuning.pdf`  
- `15-Risks-and-Dangers-of-Ai.pdf`  

These documents cover different AI/ML topics and serve as the corpus for search and summarization.

### 2.2 Preprocessing

Module: `preprocess.py`

Responsibilities:

- **Loading**  
  - Iterate over all `.pdf` files in `corpus/`.  
  - Use `PyPDFLoader` to load each PDF page as a LangChain `Document`.  
  - Attach metadata:
    - `source`: full path to the PDF  
    - `page`: page number  

- **Cleaning**  
  - Strip leading/trailing whitespace.  
  - Collapse multiple newlines/spaces into a single space.  
  - Drop empty documents after cleaning.  

- **Chunking**  
  - Use `RecursiveCharacterTextSplitter` with:  
    - `chunk_size = 1000`  
    - `chunk_overlap = 150`  
  - Produce overlapping text chunks suitable for search and summarization.

**API**

```python
preprocess_documents(corpus_dir: str = "corpus") -> List[Document]
```

Returns a list of cleaned, chunked `Document` objects for all PDFs.

***

## 3. Retrieval: TF‑IDF + Embeddings (`retrieval.py`)

### 3.1 `HybridRetriever`

Class: `HybridRetriever`

**Input:**

- `chunks`: list of `Document` objects from `preprocess_documents()`.

**Internal indexes:**

- **TF‑IDF index (traditional IR)**  
  - Uses `TfidfVectorizer` (unigrams + bigrams, limited max features).  
  - Builds a sparse TF‑IDF matrix over all chunk texts.  
  - Captures exact/keyword‑based relevance.

- **Embedding index (semantic search)**  
  - Uses `HuggingFaceEmbeddings` (e.g., `sentence-transformers/all-mpnet-base-v2`).  
  - Stores embeddings in a `Chroma` vector store.  
  - Captures semantic similarity even when wording is different.

### 3.2 Retrieval Methods

```python
search_tfidf(query: str, k: int = 5) -> List[Tuple[Document, float]]
```

- Converts query to a TF‑IDF vector.  
- Computes cosine similarity with all chunk vectors.  
- Returns top‑k chunks ranked by TF‑IDF similarity.

```python
search_embeddings(query: str, k: int = 5) -> List[Tuple[Document, float]]
```

- Uses `Chroma.similarity_search_with_score`.  
- Converts distances to similarity scores (e.g., `1 / (1 + distance)`).  
- Returns top‑k chunks ranked by embedding similarity.

```python
search_hybrid(query: str, k: int = 5, alpha: float = 0.5) -> List[Tuple[Document, float]]
```

- Runs both TF‑IDF and embedding search.  
- Combines scores:

  ```python
  combined_score = alpha * tfidf_score + (1 - alpha) * embedding_score
  ```

- Returns top‑k chunks sorted by combined score.

This implements a hybrid retrieval approach as required (traditional IR + embeddings).

***

## 4. Summarization (`summarize.py`)

### 4.1 LLM Configuration

- Provider: Gemini via `google-generativeai`  
- Environment variable: `GEMINI_API_KEY`  
- Default model: `gemini-2.0-flash` (configurable)

Helper function:

```python
llm_summarize(prompt: str) -> str
```

Sends a prompt to Gemini and returns the response text.

### 4.2 Summarization Logic

Goal:  
Given retrieved chunks, generate a coherent summary with configurable length.

Functions:

```python
build_summary_prompt(docs: List[Document], length: str = "short") -> str
```

- Concatenates chunk texts with markers like `[DOC 1]`, `[DOC 2]`, …  
- Adds instructions:
  - Use only the provided text as evidence.  
  - Do not introduce external information.  
  - Respect `length`:
    - `"short"` → 2–3 sentences  
    - `"medium"` → one concise paragraph  
    - `"long"` → 2–3 short paragraphs  

```python
summarize_docs(docs: List[Document], length: str = "short") -> str
```

- Builds the prompt and calls `llm_summarize`.  
- Returns the cleaned summary string.

**Usage pattern:**

```python
results = retriever.search_hybrid(query, k)
docs = [doc for doc, _ in results]
summary = summarize_docs(docs, length="short")
```

***

## 5. Search Evaluation (`evaluate_search.py`)

File: `evaluate_search.py`

### Purpose

Evaluate how well the hybrid retriever returns the correct PDF for a set of known queries.

### Test setup

Define a small list of queries mapped to expected PDFs, for example:

- “What is machine learning?” → `What-is-machine-learning.pdf`  
- “Explain deep learning and CNNs/RNNs/transformers.” → `What-is-deep-learning.pdf`  
- “What is LLM fine‑tuning?” → `What-is-LLM-fine-tuning.pdf`  
- “What are the main risks of AI?” → `15-Risks-and-Dangers-of-Ai.pdf`  

### Procedure

For each query:

- Call `HybridRetriever.search_hybrid(query, k)`.  
- Inspect `doc.metadata["source"]` for top‑k results.  
- Mark `hit = True` if any source filename contains the expected PDF name (or keyword).

### Logging

For each query, print:

- Query  
- Expected source keyword  
- Top‑k source filenames  
- `Hit: True/False`  

### Metric

```text
accuracy@k = hits / total_queries
```

**Run:**

```bash
python evaluate_search.py
```

This gives a simple, interpretable measure of search relevance.

***

## 6. Streamlit User Interface (`app.py`)

File: `app.py` (optional but recommended for demo)

### Responsibilities

- Initialize and cache the retriever:

```python
@st.cache_resource
def get_retriever():
    chunks = preprocess_documents()
    return HybridRetriever(chunks)
```

- Inputs:
  - Query text (`st.text_input`)  
  - Summary length (`st.selectbox`: short / medium / long)  
  - Top‑k documents (`st.slider`)

- Search & summarize:
  - On button click:
    - Run `search_hybrid(query, k)` to get top‑k chunks.  
    - Call `summarize_docs(docs, length)` to generate a summary.

- Display:
  - The generated summary.  
  - For each result:
    - Source filename and page.  
    - Short snippet of the chunk text.

**Run:**

```bash
streamlit run app.py
```

This provides a simple UI that lets reviewers interact with your RAG system.

***

## 7. How to Run the System

### 7.1 Install Dependencies

Example:

```bash
pip install langchain-community langchain-text-splitters langchain-huggingface langchain-chroma
pip install scikit-learn
pip install google-generativeai
pip install rouge-score
pip install streamlit
```

### 7.2 Prepare the Corpus

Place PDFs in:

```text
corpus/
  ├── What-is-machine-learning.pdf
  ├── What-is-deep-learning.pdf
  ├── What-is-LLM-fine-tuning.pdf
  └── 15-Risks-and-Dangers-of-Ai.pdf
```

### 7.3 Configure Gemini

Set API key:

```bash
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

### 7.4 Quick CLI Demo

```bash
python retrieval.py      # optional retrieval demo
python summarize.py      # demo: search + summarize pipeline
```

### 7.5 Evaluate Search

```bash
python evaluate_search.py
```

Check console output for per‑query hits and overall accuracy@k.

### 7.6 Streamlit UI

```bash
streamlit run app.py
```

Open the URL shown in the terminal to access the web interface.

***

## 8. Limitations and Future Work

- Small corpus (4 PDFs) → metrics are illustrative, not statistically strong.  
- Simple hybrid retrieval (linear fusion of TF‑IDF and embeddings) → could be extended with BM25, rerankers, or learned ranking models.  
- Single‑shot summarization → for longer documents, a hierarchical summarization pipeline (chunk → section → full doc) would be more robust.  
- Evaluation coverage → based on a small manually defined test set; future work could add more queries and more advanced metrics (BERTScore, human ratings).  
- Model provider → currently uses Gemini; the LLM wrapper can be generalized to support other providers (OpenAI, Groq, etc.).

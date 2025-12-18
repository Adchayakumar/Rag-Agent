
# 📄 Document Search and Summarization with RAG

[▶ You can see a deployed version of this app here](https://rag-agent-for.streamlit.app/)

> *A Retrieval-Augmented Generation (RAG) system for searching and summarizing a small PDF corpus using classical IR, dense embeddings, and a Gemini LLM — with built-in evaluation and a Streamlit UI.*

---

## 1. Project Overview

**Objective**

Design and implement a system that can:

- **Search** a fixed corpus of PDFs and return the most relevant chunks for a user query  
- **Summarize** retrieved chunks using a Large Language Model (Gemini)  
- **Evaluate**  
  - Retrieval quality (does the correct document appear in top-k?)  
  - Summary quality using ROUGE metrics against short human-written references  
- **Expose a clean UI** with  
  - Query input (with example suggestions)  
  - Adjustable summary length (short / medium / long)  
  - Pagination over retrieved chunks  

> The system is built for a fixed, small corpus (4 PDFs), as required in the assignment, but the architecture is scalable to larger datasets.

---

## 2. Data and Pre-processing

### Corpus

The corpus consists of **4 PDF documents**:

- What is machine learning  
- What is deep learning  
- What is LLM fine-tuning  
- 15 Risks and Dangers of AI  

All PDFs are stored in the `corpus/` directory and loaded using **LangChain’s `PyPDFLoader`**.

### Pre-processing Steps (`preprocess.py`)

**1. PDF Loading**
- Uses `PyPDFLoader` to load each PDF as a list of `Document` objects (one per page)
- Sets `source` metadata to the PDF path so evaluation can map chunks back to original files

**2. Text Cleaning**
- Removes leading and trailing whitespace  
- Drops empty lines  
- Normalizes multiple spaces and newlines into a single space  

*Rationale:* Reduces formatting noise without altering semantic content, improving both TF-IDF and embedding quality.

**3. Chunking**
- Uses `RecursiveCharacterTextSplitter` with  
  - `chunk_size = 1000`  
  - `chunk_overlap = 150`  

*Rationale:*  
- 1000 characters typically capture a complete paragraph or section  
- Overlap preserves context across chunk boundaries while limiting duplication

**4. Caching**
- All chunks are cached to `data/chunks.pkl`
- Evaluation scripts and the UI reuse this cache, avoiding repeated PDF parsing

---

## 3. Retrieval Methodology

### HybridRetriever (`retrieval.py`)

A custom retriever that combines **lexical** and **semantic** search signals.

---

### TF-IDF (Traditional IR)

- Uses `TfidfVectorizer` with  
  - `max_features = 5000`  
  - `ngram_range = (1, 2)`  
- Builds a TF-IDF matrix over all chunks  
- Computes cosine similarity between query and chunk vectors  
- Returns top-k chunks by score

---

### Embedding Search (Chroma)

- Uses `HuggingFaceEmbeddings`  
  - Model: `sentence-transformers/all-mpnet-base-v2`  
- Indexes chunks in a **Chroma** vector store  
- Retrieves top-k chunks using `similarity_search_with_score`  
- Converts distances into a simple similarity score

---

### Hybrid Search Strategy

- Retrieves top-k candidates from both TF-IDF and embedding search  
- Uses **object identity (`id(doc)`)** to merge scores safely  
- Final score:

\[
\text{combined} = \alpha \cdot \text{tfidf\_score} + (1 - \alpha) \cdot \text{embedding\_score}
\]

- Default value: `α = 0.5`  
- Returns top-k chunks ranked by combined score

**Why hybrid retrieval?**
- TF-IDF excels at exact keyword matches  
- Embeddings capture semantic similarity and paraphrases  
- Combining both improves robustness, especially for small corpora

---

## 4. Summarization Methodology

Implemented in `summarize.py`.

### Model Configuration

- Uses `google.generativeai` with **Gemini (`gemini-2.0-flash`)**
- API key resolution order:
  - `st.secrets["GEMINI_API_KEY"]` (Streamlit Cloud)
  - `GEMINI_API_KEY` or `GOOGLE_API_KEY` environment variables (local / Colab)

---

### Document Selection

- For a user query, `HybridRetriever.search_hybrid(query, k)` retrieves top-k chunks  
- Retrieved chunks are concatenated into a single context string  
- This context is passed to the LLM for summarization

---

### Summary Length Control

The `summarize_docs(docs, length)` function supports:

- **Short** — 2–3 concise sentences  
- **Medium** — one paragraph capturing key ideas  
- **Long** — multi-paragraph, detailed explanation  

The Streamlit UI exposes this option, generating exactly one summary per request.

---

## 5. Evaluation Procedure and Results

### 5.1 Retrieval Evaluation (`evaluate_search.py`)

**Test Queries**
- Machine learning  
- Deep learning  
- LLM fine-tuning  
- AI risks  

Each query maps to one expected PDF file.

**Evaluation Steps**
1. Run `search_hybrid(query, k=5)`  
2. Extract `source` filenames from top-k chunks  
3. Check if the expected PDF appears in results  
4. Compute accuracy:

\[
\text{accuracy} = \frac{\text{\#queries with correct PDF in top-k}}{\text{total queries}}
\]

**Results**
- `k = 5` → **Accuracy = 1.00 (4/4)**  
- Logs saved in `logs/search_eval.txt`

---

### 5.2 Summarization Evaluation (`evaluate_summary.py`)

**Reference Summaries**
- One short, human-written reference summary per PDF

**Evaluation Steps**
1. Select chunks belonging to a specific PDF  
2. Generate a **short** system summary  
3. Compute **ROUGE-1 F** and **ROUGE-L F**

**Example Results**
- *Machine Learning PDF*  
  - ROUGE-1 F ≈ 0.425  
  - ROUGE-L F ≈ 0.319  
- *Deep Learning PDF*  
  - ROUGE-1 F ≈ 0.370  
  - ROUGE-L F ≈ 0.241  

**Interpretation**
- ROUGE scores in the 0.3–0.4 range indicate strong coverage of key ideas  
- Differences reflect abstractive phrasing rather than factual errors  
- Manual inspection confirms summary faithfulness

---

## 6. Challenges and Solutions

**1. Environment and dependency issues**
- Protobuf version conflicts and GPU warnings in Colab  
- Fixed by pinning compatible versions and treating GPU logs as non-critical

**2. Hybrid retriever bug**
- `self.chunks.index(doc)` caused `ValueError`  
- Solved using identity-based aggregation (`id(doc)`)

**3. Filename mismatches**
- Metadata filenames differed from evaluation keys  
- Normalized filenames (lowercase, replace `-` / `_` with spaces)

**4. Performance**
- Embedding construction is slow on first run  
- Cached chunks and used `@st.cache_resource` in Streamlit

**5. API key safety**
- Avoided hardcoding  
- Used environment variables and Streamlit secrets

---

## 7. Setup and Running the Solution

### 7.1 Local / Colab (CLI)

**Clone**
```bash
git clone https://github.com/Adchayakumar/Rag-Agent.git
cd Rag-Agent
````

**Install**

```bash
pip install -r requirements.txt
```

**Set API Key**

```bash
export GEMINI_API_KEY="YOUR_REAL_KEY"
```

**Preprocess**

```bash
python preprocess.py
```

**Run Components**

```bash
python retrieval.py
python summarize.py
```

**Evaluate**

```bash
python evaluate_search.py
python evaluate_summary.py
```

---

### 7.2 Streamlit UI

```bash
streamlit run app.py
```

---

## 8. Repository Structure

* `corpus/` – Input PDFs
* `data/` – Cached chunks
* `logs/` – Evaluation logs
* `preprocess.py` – PDF loading & chunking
* `retrieval.py` – HybridRetriever
* `summarize.py` – Gemini summarization
* `evaluate_search.py` – Retrieval evaluation
* `evaluate_summary.py` – ROUGE evaluation
* `utils.py` – Utilities
* `app.py` – Streamlit UI
* `requirements.txt` – Dependencies







# Document Search and Summarization with RAG

[▶ Live Demo on Streamlit](https://rag-agent-for.streamlit.app/)

This project implements a Retrieval‑Augmented Generation (RAG) system that can search and summarize a small corpus of PDF documents using a traditional IR method (TF‑IDF), dense embeddings (Chroma + sentence transformers), and a Gemini LLM for summarization. It includes automated evaluation for both retrieval and summarization, plus a Streamlit UI.

***

## 1. Project Overview

The goal is to design a system that can:

- Search a fixed corpus of PDFs and return the most relevant chunks for a user query.  
- Summarize those chunks using a Large Language Model (Gemini).  
- Evaluate:
  - Retrieval quality (does the correct document appear in top‑k?).  
  - Summary quality using ROUGE metrics against short human‑written reference summaries.  
- Provide a user‑friendly interface with:
  - Query input (with example suggestions).  
  - Adjustable summary length (short / medium / long).  
  - Pagination over retrieved chunks.

The system is designed for a fixed, small corpus (4 PDFs), as required in the assignment, but the architecture can scale to larger corpora.

***

## 2. Data and Pre‑processing

### Corpus

The corpus consists of 4 PDF documents:

- What is machine learning  
- What is deep learning  
- What is LLM fine‑tuning  
- 15 Risks and Dangers of AI  

These are placed in the `corpus/` directory and loaded using LangChain’s `PyPDFLoader`.

### Pre‑processing steps

Implemented in `preprocess.py`:

1. **PDF loading**  
   - Uses `PyPDFLoader` to read each PDF into a list of `Document` objects (one per page).  
   - The `source` metadata is set to the PDF path so later evaluation can map chunks back to files.

2. **Text cleaning**  
   - Strips leading/trailing whitespace.  
   - Removes empty lines and normalizes multiple spaces/newlines into a single space.  
   - Justification: this reduces noise (line breaks, spacing) without changing semantic content, helping both TF‑IDF and embeddings.

3. **Chunking**  
   - Uses `RecursiveCharacterTextSplitter` with:
     - `chunk_size=1000`  
     - `chunk_overlap=150`  
   - Justification:
     - 1000 characters is enough to capture a complete paragraph or section.  
     - 150 overlap preserves context across chunk boundaries while limiting duplication.

4. **Caching**  
   - All chunks are cached to `data/chunks.pkl` after preprocessing.  
   - Evaluation scripts and the UI reuse this cache instead of re‑parsing PDFs, improving efficiency and reproducibility.

***

## 3. Retrieval Methodology

### HybridRetriever

Implemented in `retrieval.py` as a `HybridRetriever` class.

#### TF‑IDF (traditional IR)

- Uses `TfidfVectorizer` with:
  - `max_features=5000`  
  - `ngram_range=(1, 2)`  
- Builds a TF‑IDF matrix over all chunks.  
- For a query, computes cosine similarity between query TF‑IDF vector and all chunk vectors, returning top‑k chunks.

#### Embedding search (Chroma)

- Uses `HuggingFaceEmbeddings` with `sentence-transformers/all-mpnet-base-v2`.  
- Indexes chunks in a Chroma vector store.  
- For a query, retrieves top‑k chunks by embedding similarity via `similarity_search_with_score`, converting distances to a simple similarity score.

#### Hybrid search

- For each query, retrieves top‑k candidates from both TF‑IDF and embedding search.  
- Uses Python object identity (`id(doc)`) to combine scores per chunk, avoiding `self.chunks.index(doc)` issues.  
- Combined score:

  \[
  \text{combined} = \alpha \cdot \text{tfidf\_score} + (1 - \alpha) \cdot \text{embedding\_score}
  \]

  with default \(\alpha = 0.5\).  
- Returns top‑k chunks with highest combined scores.

#### Why hybrid?

- TF‑IDF is strong on exact keyword overlap.  
- Embeddings capture semantic similarity and paraphrases.  
- Combining them provides more robust retrieval, especially in small corpora where both lexical and semantic signals matter.

***

## 4. Summarization Methodology

Implemented in `summarize.py`.

### Model configuration

- Uses `google.generativeai` with Gemini (`gemini-2.0-flash`).  
- API key is read from:
  - `st.secrets["GEMINI_API_KEY"]` (for Streamlit Cloud), if available, or  
  - `GEMINI_API_KEY` / `GOOGLE_API_KEY` environment variables for local/Colab runs.

### Document selection

- For a given user query, `HybridRetriever.search_hybrid(query, k)` returns top‑k chunks.  
- These chunks are concatenated into a single context string passed to the LLM.

### Summary length control

- A `summarize_docs(docs, length)` function creates a prompt that asks the model for:
  - `"short"` summary: 2–3 concise sentences.  
  - `"medium"` summary: a paragraph capturing key points.  
  - `"long"` summary: more detailed multi‑paragraph explanation.  
- The Streamlit UI lets the user choose the desired length; the system generates only one summary per request.

***

## 5. Evaluation Procedure and Results

### 5.1 Retrieval evaluation (search)

Implemented in `evaluate_search.py`.

#### Test set

- 4 queries, each targeted at one specific PDF:
  - Machine learning  
  - Deep learning  
  - LLM fine‑tuning  
  - AI risks  
- For each, an `expected_source_keyword` is defined (the real PDF filename).

#### Evaluation method

For each query:

1. Run `search_hybrid(query, k=5)`.  
2. Extract `source` filenames of the top‑k chunks.  
3. Check if any of these filenames match the expected PDF (using normalized filename comparison for robustness).  
4. Compute accuracy:

\[
\text{accuracy} = \frac{\text{\#queries where expected PDF appears in top‑k}}{\text{total queries}}
\]

#### Results

- For k=5, accuracy = **1.00** (4/4 queries hit the correct document).  
- Detailed results are saved in `logs/search_eval.txt` and show that all top‑k results for each query come from the correct PDF.  
- This satisfies the requirement to “measure the accuracy of the search mechanism based on the relevance of returned documents.”

### 5.2 Summarization evaluation

Implemented in `evaluate_summary.py`.

#### Reference summaries

- A short human‑written reference summary is created for each PDF and stored in `REFERENCE_SUMMARIES`.

#### Evaluation method

For each PDF:

1. Select a subset of chunks belonging to that PDF (by filename in `metadata["source"]`).  
2. Run `summarize_docs(docs, length="short")` to generate a short system summary.  
3. Compute ROUGE‑1 F and ROUGE‑L F between reference and generated summaries using the `rouge_score` package.

Results are saved in `logs/summary_eval.txt`.

#### Results (example)

- Machine learning PDF:
  - ROUGE‑1 F ≈ 0.425  
  - ROUGE‑L F ≈ 0.319  
- Deep learning PDF:
  - ROUGE‑1 F ≈ 0.370  
  - ROUGE‑L F ≈ 0.241  

#### Interpretation

- ROUGE scores range from 0 to 1; higher means more overlap with the reference.  
- Values around 0.3–0.4 for ROUGE‑1 with short abstractive summaries indicate that the model covers most key points but with different wording and structure.  
- Together with manual inspection, this shows that summaries are reasonably faithful to the source documents.

***

## 6. Challenges and Solutions

1. **Environment and dependency issues (Colab)**  
   - Problems with protobuf versions and GPU warnings (cuFFT/cuDNN registration).  
   - Solution: pin compatible versions where needed and treat GPU logs as warnings when they do not affect RAG logic.

2. **Hybrid retriever bug**  
   - Initial implementation used `self.chunks.index(doc)` to map documents back to indices, causing `ValueError` when Python objects differed.  
   - Solution: switch to identity‑based aggregation (`id(doc)`) and score dictionaries, avoiding list lookups by object equality.

3. **Filename mismatches in evaluation**  
   - Evaluation initially used dashed filenames (e.g., `What-is-machine-learning.pdf`), while metadata had filenames with spaces.  
   - Solution: normalize filenames (replace `-` / `_` with spaces, lowercase) for robust matching, and update test keys to real filenames.

4. **Performance concerns**  
   - Building embeddings and vector store is slow on first run.  
   - Solution: cache chunks (`data/chunks.pkl`) and use `@st.cache_resource` in Streamlit so retriever is built only once per process; document this explicitly as an efficiency measure.

5. **API key handling**  
   - Gemini key should not be hardcoded or committed.  
   - Solution: read from environment variables for local/Colab and from Streamlit secrets (`st.secrets["GEMINI_API_KEY"]`) on Streamlit Cloud, following best practices.

***

## 7. Setup and Running the Solution

### 7.1 Local / Colab (CLI scripts)

**Clone the repo**

```bash
git clone https://github.com/Adchayakumar/Rag-Agent.git
cd Rag-Agent
```

**Install dependencies**

```bash
pip install -r requirements.txt
```

**Set Gemini API key**

On your machine / Colab:

```bash
export GEMINI_API_KEY="YOUR_REAL_KEY"
# or in Python:
# import os; os.environ["GEMINI_API_KEY"] = "YOUR_REAL_KEY"
```

**Preprocess and cache chunks**

```bash
python preprocess.py
# creates data/chunks.pkl
```

**Test retrieval and summarization**

```bash
python retrieval.py
python summarize.py
```

**Run evaluations**

```bash
python evaluate_search.py     # logs/search_eval.txt
python evaluate_summary.py    # logs/summary_eval.txt
```

### 7.2 Streamlit UI (local)

```bash
streamlit run app.py
```


## 8. Repository Structure

- `corpus/` – Input PDFs (fixed corpus).  
- `data/` – Cached chunks (`chunks.pkl`).  
- `logs/` – Evaluation logs (`search_eval.txt`, `summary_eval.txt`).  
- `preprocess.py` – PDF loading, cleaning, chunking, and caching.  
- `retrieval.py` – `HybridRetriever` (TF‑IDF + embeddings + hybrid).  
- `summarize.py` – Gemini configuration and summarization logic.  
- `evaluate_search.py` – Retrieval evaluation (top‑k accuracy).  
- `evaluate_summary.py` – Summary evaluation (ROUGE).  
- `utils.py` – Helper functions (e.g., `load_cached_chunks`).  
- `app.py` – Streamlit web interface.  
- `requirements.txt` – Dependencies.  



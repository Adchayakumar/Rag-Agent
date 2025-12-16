

from typing import List, Tuple
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from preprocess import preprocess_documents  

class HybridRetriever:
    """
    Builds:
      - TF-IDF index over text chunks (traditional IR)
      - Embedding index (Chroma) over the same chunks
    Provides search_tfidf, search_embeddings, search_hybrid.
    """

    def __init__(self, chunks: List[Document]):
        self.chunks = chunks

        # ----- TF-IDF -----
        texts = [d.page_content for d in chunks]
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)

        # ----- Embeddings + Chroma -----
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )

    # ---------- Traditional IR search (TF-IDF) ----------

    def search_tfidf(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Return top-k chunks by TF-IDF cosine similarity."""
        q_vec = self.tfidf_vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.tfidf_matrix)[0]  # shape (n_chunks,)
        top_idx = np.argsort(sims)[::-1][:k]
        results = [(self.chunks[i], float(sims[i])) for i in top_idx]
        return results

    # ---------- Embedding search (Chroma) ----------

    def search_embeddings(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Return top-k chunks by embedding similarity via Chroma."""
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
        # docs_with_scores: List[(Document, score)]
        # Chroma scores are distance-like; convert to similarity (optional)
        results = []
        for doc, score in docs_with_scores:
            sim = 1.0 / (1.0 + score)  # simple transform: smaller distance -> higher sim
            results.append((doc, float(sim)))
        return results

    # ---------- Hybrid search ----------

    def search_hybrid(self, query: str, k: int = 5, alpha: float = 0.5) -> List[Tuple[Document, float]]:
        """
        Combine TF-IDF and embedding scores.
        alpha controls weight: 0.5 means equal weight.
        Returns top-k chunks with highest combined score.
        """
        tfidf_results = self.search_tfidf(query, k=max(k, 10))
        embed_results = self.search_embeddings(query, k=max(k, 10))

        # Collect scores per document id (we'll use index in self.chunks as ID)
        tfidf_scores = {}
        for doc, score in tfidf_results:
            idx = self.chunks.index(doc)
            tfidf_scores[idx] = max(tfidf_scores.get(idx, 0.0), score)

        embed_scores = {}
        for doc, score in embed_results:
            idx = self.chunks.index(doc)
            embed_scores[idx] = max(embed_scores.get(idx, 0.0), score)

        # Combine scores (0 if missing)
        combined = []
        for idx in range(len(self.chunks)):
            s_tfidf = tfidf_scores.get(idx, 0.0)
            s_emb = embed_scores.get(idx, 0.0)
            combined_score = alpha * s_tfidf + (1 - alpha) * s_emb
            if combined_score > 0:
                combined.append((idx, combined_score))

        # Sort and take top-k
        combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)[:k]
        results = [(self.chunks[idx], float(score)) for idx, score in combined_sorted]
        return results


if __name__ == "__main__":
    # Example usage
    print("Preprocessing documents...")
    chunks = preprocess_documents()
    print(f"Total chunks: {len(chunks)}")

    retriever = HybridRetriever(chunks)

    query = "What is the main idea of the first document?"
    print(f"\nQuery: {query}")

    print("\nTop 3 TF-IDF results:")
    for doc, score in retriever.search_tfidf(query, k=3):
        print(f"- Score={score:.3f}, Text={doc.page_content[:120]}...")

    print("\nTop 3 Embedding results:")
    for doc, score in retriever.search_embeddings(query, k=3):
        print(f"- Score={score:.3f}, Text={doc.page_content[:120]}...")

    print("\nTop 3 Hybrid results:")
    for doc, score in retriever.search_hybrid(query, k=3):
        print(f"- Score={score:.3f}, Text={doc.page_content[:120]}...")

import os
from typing import List, Dict, Any

from utils import load_cached_chunks
from retrieval import HybridRetriever

# Small test set: query + which PDF it should hit [file:155]
TEST_QUERIES = [
    {
        "query": "What is machine learning and what are its main types?",
        "expected_source_keyword": "What is machine learning.pdf",
    },
    {
        "query": "How does deep learning work and what are CNNs, RNNs, and transformers?",
        "expected_source_keyword": "What is deep learning.pdf",
    },
    {
        "query": "What is LLM fine-tuning and why is it important?",
        "expected_source_keyword": "What is LLM fine-tuning.pdf",
    },
    {
        "query": "What are the main risks and dangers of AI?",
        "expected_source_keyword": "15 Risks and Dangers of Ai.pdf",
    },
]


def normalize_name(s: str) -> str:
    """Normalize filenames for more robust matching."""
    return (
        s.lower()
        .replace("-", " ")
        .replace("_", " ")
        .strip()
    )


def source_matches(doc, keyword: str) -> bool:
    """Check if the document's source metadata or path contains the keyword."""
    src = doc.metadata.get("source", "") or ""
    base = os.path.basename(src)
    return normalize_name(keyword) in normalize_name(base)


def evaluate_search(retriever: HybridRetriever, k: int = 5) -> Dict[str, Any]:
    results = []
    hits = 0

    for test in TEST_QUERIES:
        query = test["query"]
        expected_kw = test["expected_source_keyword"]

        top_docs = retriever.search_hybrid(query, k=k)
        top_sources = [os.path.basename(d.metadata.get("source", "")) for d, _ in top_docs]

        hit = any(source_matches(d, expected_kw) for d, _ in top_docs)
        if hit:
            hits += 1

        results.append(
            {
                "query": query,
                "expected_source_keyword": expected_kw,
                "top_sources": top_sources,
                "hit": hit,
            }
        )

    accuracy = hits / len(TEST_QUERIES) if TEST_QUERIES else 0.0
    return {"results": results, "accuracy": accuracy, "k": k}


if __name__ == "__main__":
    print("loading cached chunks...")
    chunks = load_cached_chunks()
    print(f"Total chunks: {len(chunks)}")

    retriever = HybridRetriever(chunks)

    eval_result = evaluate_search(retriever, k=5)

    print(f"\nSearch evaluation (k={eval_result['k']}):")
    print(f"Accuracy: {eval_result['accuracy']:.2f}\n")

    for r in eval_result["results"]:
        print("Query:", r["query"])
        print("Expected source contains:", r["expected_source_keyword"])
        print("Top sources:", r["top_sources"])
        print("Hit:", r["hit"])
        print("-" * 60)

    # ---- Save to logs/search_eval.txt ----
    os.makedirs("logs", exist_ok=True)

    log_path = "logs/search_eval.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Search evaluation (k={eval_result['k']}):\n")
        f.write(f"Accuracy: {eval_result['accuracy']:.2f}\n\n")
        for r in eval_result["results"]:
            f.write(f"Query: {r['query']}\n")
            f.write(f"Expected source contains: {r['expected_source_keyword']}\n")
            f.write(f"Top sources: {', '.join(r['top_sources'])}\n")
            f.write(f"Hit: {r['hit']}\n")
            f.write("-" * 60 + "\n")


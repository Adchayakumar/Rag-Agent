# evaluation/evaluate_summary.py

import os
from typing import List, Dict

from rouge_score import rouge_scorer

from preprocess import preprocess_documents
from retrieval import HybridRetriever
from summarize import summarize_docs


# Reference summaries you write manually (very short)
REFERENCE_SUMMARIES = {
    "What-is-machine-learning.pdf": (
        "Defines machine learning as a field where algorithms learn from data without explicit "
        "programming, explains types like supervised, unsupervised, semi-supervised, and "
        "reinforcement learning, and shows applications such as recommendation systems, "
        "fraud detection, and image recognition."
    ),
    "What-is-deep-learning.pdf": (
        "Explains deep learning as neural network based learning with many hidden layers, "
        "describes CNNs, RNNs, and transformers, and compares deep learning with traditional "
        "machine learning in terms of accuracy, data needs, and feature engineering."
    ),
    "What-is-LLM-fine-tuning.pdf": (
        "Describes LLM fine-tuning as adapting a pre-trained model to a specific domain or task, "
        "covers motivations like customization, data compliance, and limited labeled data, and "
        "outlines approaches such as feature extraction, full fine-tuning, supervised fine-tuning, and RLHF."
    ),
    "15-Risks-and-Dangers-of-Ai.pdf": (
        "Lists major risks of AI such as lack of transparency, job losses from automation, "
        "deepfakes and social manipulation, privacy violations, algorithmic bias, and military use, "
        "arguing for responsible development and regulation."
    ),
}


def pick_docs_for_source(chunks, source_keyword: str, max_docs: int = 8):
    """Filter chunks by source filename keyword and return a subset for summarization."""
    selected = [
        d for d in chunks
        if source_keyword.lower() in os.path.basename(d.metadata.get("source", "")).lower()
    ]
    return selected[:max_docs]


def evaluate_summaries(chunks) -> List[Dict]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    results = []

    for filename, reference in REFERENCE_SUMMARIES.items():
        docs = pick_docs_for_source(chunks, filename, max_docs=8)
        if not docs:
            continue

        generated = summarize_docs(docs, length="short")

        scores = scorer.score(reference, generated)
        results.append(
            {
                "filename": filename,
                "reference": reference,
                "generated": generated,
                "rouge1_f": scores["rouge1"].fmeasure,
                "rougeL_f": scores["rougeL"].fmeasure,
            }
        )

    return results


if __name__ == "__main__":
    print("Preprocessing documents...")
    chunks = preprocess_documents()
    print(f"Total chunks: {len(chunks)}")

    print("\nEvaluating summaries...")
    results = evaluate_summaries(chunks)

    for r in results:
        print("\nFile:", r["filename"])
        print(f"ROUGE-1 F: {r['rouge1_f']:.3f}, ROUGE-L F: {r['rougeL_f']:.3f}")
        print("Reference summary:")
        print(r["reference"])
        print("\nGenerated summary:")
        print(r["generated"])
        print("-" * 80)

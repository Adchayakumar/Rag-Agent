import os
from typing import List, Dict

from rouge_score import rouge_scorer

from utils import load_cached_chunks
from summarize import summarize_docs

# Reference summaries you write manually (very short) [file:155]
REFERENCE_SUMMARIES = {
    # Use the actual filenames as they appear in metadata (with spaces)
    "What is machine learning.pdf": (
        "Defines machine learning as a field where algorithms learn from data without explicit "
        "programming, explains types like supervised, unsupervised, semi-supervised, and "
        "reinforcement learning, and shows applications such as recommendation systems, "
        "fraud detection, and image recognition."
    ),
    "What is deep learning.pdf": (
        "Explains deep learning as neural network based learning with many hidden layers, "
        "describes CNNs, RNNs, and transformers, and compares deep learning with traditional "
        "machine learning in terms of accuracy, data needs, and feature engineering."
    ),
    "What is LLM fine-tuning.pdf": (
        "Describes LLM fine-tuning as adapting a pre-trained model to a specific domain or task, "
        "covers motivations like customization, data compliance, and limited labeled data, and "
        "outlines approaches such as feature extraction, full fine-tuning, supervised fine-tuning, and RLHF."
    ),
    "15 Risks and Dangers of Ai.pdf": (
        "Lists major risks of AI such as lack of transparency, job losses from automation, "
        "deepfakes and social manipulation, privacy violations, algorithmic bias, and military use, "
        "arguing for responsible development and regulation."
    ),
}


def normalize_name(s: str) -> str:
    """Normalize filenames for more robust matching."""
    return (
        s.lower()
        .replace("-", " ")
        .replace("_", " ")
        .strip()
    )


def pick_docs_for_source(chunks, source_keyword: str, max_docs: int = 8):
    """
    Filter chunks by source filename keyword and return a subset for summarization.
    Uses normalized name comparison to tolerate spaces vs dashes.
    """
    norm_kw = normalize_name(source_keyword)
    selected = []
    for d in chunks:
        src = d.metadata.get("source", "") or ""
        base = os.path.basename(src)
        if norm_kw in normalize_name(base):
            selected.append(d)
    return selected[:max_docs]


def evaluate_summaries(chunks) -> List[Dict]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    results = []

    for filename, reference in REFERENCE_SUMMARIES.items():
        docs = pick_docs_for_source(chunks, filename, max_docs=8)
        if not docs:
            # Helpful debug print
            print(f"[WARN] No docs found for filename key: {filename}")
            continue

        # Generate a short summary from these docs
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
    print("loading cached chunks...")
    chunks = load_cached_chunks()
    print(f"Total chunks: {len(chunks)}")

    print("\nEvaluating summaries...")
    results = evaluate_summaries(chunks)

    if not results:
        print("\n[INFO] No evaluation results produced. Check filename keys and metadata.")
    else:
        for r in results:
            print("\nFile:", r["filename"])
            print(f"ROUGE-1 F: {r['rouge1_f']:.3f}, ROUGE-L F: {r['rougeL_f']:.3f}")
            print("Reference summary:")
            print(r["reference"])
            print("\nGenerated summary:")
            print(r["generated"])
            print("-" * 80)

    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)

    with open("logs/summary_eval.txt", "w", encoding="utf-8") as f:
        if not results:
            f.write("No evaluation results. Check filename keys and metadata.\n")
        else:
            for r in results:
                f.write(f"File: {r['filename']}\n")
                f.write(f"ROUGE-1 F: {r['rouge1_f']:.3f}, ROUGE-L F: {r['rougeL_f']:.3f}\n")
                f.write("Reference summary:\n")
                f.write(r["reference"] + "\n\n")
                f.write("Generated summary:\n")
                f.write(r["generated"] + "\n")
                f.write("-" * 80 + "\n\n")


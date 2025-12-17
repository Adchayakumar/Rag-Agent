import os
from typing import List

import streamlit as st

from utils import load_cached_chunks
from retrieval import HybridRetriever
from summarize import summarize_docs


# Example queries for simple "auto-suggestion" dropdown
EXAMPLE_QUERIES = [
    "What is machine learning and what are its main types?",
    "How does deep learning work and what are CNNs, RNNs, and transformers?",
    "What is LLM fine-tuning and why is it important?",
    "What are the main risks and dangers of AI?",
]


@st.cache_resource
def get_retriever() -> HybridRetriever:
    """Load cached chunks and build HybridRetriever once per session."""
    chunks = load_cached_chunks()
    return HybridRetriever(chunks)


def main():
    st.set_page_config(page_title="Document Search & Summarization", layout="wide")

    st.title("Document Search & Summarization")

    retriever = get_retriever()

    # --- Sidebar controls ---
    st.sidebar.header("Controls")

    # Simple auto-suggestion: user can pick an example or type custom text
    example = st.sidebar.selectbox(
        "Example queries (optional):",
        [""] + EXAMPLE_QUERIES,
        index=0,
    )

    summary_length = st.sidebar.selectbox(
        "Summary length:",
        ["short", "medium", "long"],
        index=1,
        help="Adjust how detailed the summary should be.",
    )

    k_results = st.sidebar.slider(
        "Number of search results (top-k):",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="How many chunks to retrieve for summarization.",
    )

    st.sidebar.markdown(
        """
        This interface lets you:
        - Enter a query (or pick an example)
        - See the top retrieved chunks
        - Generate a summary with adjustable length
        """
    )

    # --- Main query input ---
    st.subheader("Enter your query")

    default_query = example if example else ""
    user_query = st.text_input(
        "Query:",
        value=default_query,
        placeholder="Type your question about the documents...",
    )

    if st.button("Search & Summarize") and user_query.strip():
        with st.spinner("Searching and summarizing..."):
            # 1) Retrieve top-k chunks using hybrid search
            docs_with_scores = retriever.search_hybrid(user_query, k=k_results)
            docs = [doc for doc, score in docs_with_scores]

            # 2) Show search results with simple pagination (page by page)
            st.subheader("Search results")
            if docs_with_scores:
                # Basic pagination using an index
                page = st.number_input(
                    "Result page",
                    min_value=1,
                    max_value=len(docs_with_scores),
                    value=1,
                    step=1,
                )
                doc, score = docs_with_scores[page - 1]
                st.markdown(f"**Result {page}/{len(docs_with_scores)}**")
                st.markdown(f"**Score:** {score:.3f}")
                st.markdown(
                    f"**Source:** {os.path.basename(doc.metadata.get('source', 'Unknown'))}, "
                    f"Page: {doc.metadata.get('page_label', '?')}"
                )
                st.write(doc.page_content)

            else:
                st.info("No relevant chunks found for this query.")

            # 3) Summarize using retrieved docs
            if docs:
                st.subheader("Summary")
                summary = summarize_docs(docs, length=summary_length)
                st.write(summary)
            else:
                st.info("No documents available to summarize.")

    else:
        st.info("Type a query or select an example, then click 'Search & Summarize'.")


if __name__ == "__main__":
    main()


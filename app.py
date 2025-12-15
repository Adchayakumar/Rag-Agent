
import streamlit as st
from preprocess import preprocess_documents
from retrieval import HybridRetriever
from summarize import summarize_docs

st.set_page_config(page_title="RAG Search & Summary")

st.title("RAG Search & Summarization Demo")

# Load & cache retriever
@st.cache_resource
def get_retriever():
    chunks = preprocess_documents()
    return HybridRetriever(chunks)

retriever = get_retriever()

query = st.text_input("Enter your question:")
length = st.selectbox("Summary length", ["short", "medium", "long"])
k = st.slider("Top-k documents", 1, 10, 5)

if st.button("Search & Summarize") and query:
    # 1) Search
    results = retriever.search_hybrid(query, k=k)
    docs = [d for d, _ in results]

    # 2) Summarize
    summary = summarize_docs(docs, length=length)

    st.subheader("Summary")
    st.write(summary)

    st.subheader("Top documents")
    for i, (doc, score) in enumerate(results, start=1):
        st.markdown(f"**Result {i}** (score={score:.3f})")
        st.write(doc.page_content[:300] + "...")

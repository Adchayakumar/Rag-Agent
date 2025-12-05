import streamlit as st
import tempfile
import os
import time

# --- LangChain & RAG Imports ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

# --- Page Config ---
st.set_page_config(page_title="Agentic RAG Assistant", layout="wide")

# --- CSS for "Agentic" Look ---
st.markdown("""
<style>
    .stChatMessage {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
    }
    .resource-box {
        border-left: 3px solid #FF4B4B;
        background-color: #fafafa;
        padding: 10px;
        margin-top: 5px;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

st.title("📄 Agentic RAG: Transparent Reasoning")
st.markdown("Upload a PDF. I will show you my **Tool Calls** (searching) and the **Resources** (content) I find before answering.")

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "agent" not in st.session_state:
    st.session_state.agent = None

# --- Sidebar: Configuration & Upload ---
with st.sidebar:
    st.header("1. Configuration")
    api_key = st.text_input("Google API Key", type="password")
    
    st.header("2. Upload Document")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    # Only rebuild the agent if we have files but no agent yet
    if uploaded_file and api_key and not st.session_state.vectorstore:
        with st.status("🏗️ Building Knowledge Base...", expanded=True) as status:
            try:
                # 1. Save temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # 2. Load
                st.write("📄 Loading PDF...")
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                
                # 3. Split
                st.write(f"✂️ Splitting {len(docs)} pages...")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=140)
                splits = text_splitter.split_documents(docs)
                
                # 4. Embed
                st.write("🧠 Embedding vectors (HuggingFace)...")
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
                
                # 5. Store
                # Create the vectorstore object locally
                vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
                st.session_state.vectorstore = vectorstore
                
                # --- DEFINE TOOL ---
                # CRITICAL FIX: We use the local 'vectorstore' variable in the closure.
                # We do NOT use 'st.session_state.vectorstore' inside the tool because 
                # tools run in threads where session_state is empty.
                @tool(response_format="content_and_artifact")
                def retrieve_context(query: str):
                    """Retrieve information to help answer a query."""
                    
                    # Direct access to the vectorstore object captured by closure
                    retrieved_docs = vectorstore.similarity_search(query, k=3)

                    def format_doc_with_metadata(doc):
                        source = doc.metadata.get('source', 'Unknown Source')
                        page = doc.metadata.get('page', 'Unknown Page')
                        return (
                            f"--- START OF DOCUMENT ---\n"
                            f"Source File: {os.path.basename(source)} (Page {page})\n"
                            f"Content: {doc.page_content}\n"
                            f"--- END OF DOCUMENT ---"
                        )

                    serialized = "\n\n".join(format_doc_with_metadata(doc) for doc in retrieved_docs)
                    return serialized, retrieved_docs

                # --- BUILD AGENT ---
                st.write("🤖 Assembling Agent...")
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash-lite", 
                    google_api_key=api_key,
                    temperature=0
                )
                #add system prompt to llm
                system_prompt = "You are a helpful assistant that provides accurate information based on the provided PDF documents."
                
                # Create ReAct Agent (Graph)
                st.session_state.agent = create_react_agent(llm, tools=[retrieve_context],system_prompt=system_prompt)
                
                status.update(label="✅ System Ready!", state="complete", expanded=False)
                os.remove(tmp_path)

            except Exception as e:
                st.error(f"Error: {e}")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- Chat Interface ---

# 1. Display History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            st.write(msg["content"])
            # If there were resources saved with this message, display them
            if "resources" in msg:
                with st.expander("📚 Sources Referenced"):
                    for res in msg["resources"]:
                        st.markdown(f"**Page {res.metadata.get('page', '?')}**: {res.page_content[:200]}...")

# 2. Handle Input
if prompt := st.chat_input("Ask about your PDF..."):
    if not st.session_state.agent:
        st.error("Please upload a PDF and provide an API Key first.")
        st.stop()

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Process with Agent
    with st.chat_message("assistant"):
        
        # Container to hold the steps
        step_container = st.status("🕵️ Agent Working...", expanded=True)
        
        final_response = ""
        collected_artifacts = []
        
        # Stream the agent's events
        try:
            stream = st.session_state.agent.stream(
                {"messages": [HumanMessage(content=prompt)]},
                stream_mode="updates" 
            )

            for chunk in stream:
                # CHECK FOR TOOL CALLS (The Agent deciding to act)
                if "agent" in chunk:
                    agent_msg = chunk["agent"]["messages"][0]
                    if agent_msg.tool_calls:
                        for tool_call in agent_msg.tool_calls:
                            step_container.write(f"🛠️ **Plan:** Calling tool `{tool_call['name']}`")
                            step_container.write(f"❓ **Query:** {tool_call['args']}")

                # CHECK FOR TOOL OUTPUTS (The actual RAG retrieval)
                if "tools" in chunk:
                    tool_msg = chunk["tools"]["messages"][0]
                    step_container.write(f"✅ **Action:** Retrieved data.")
                    
                    # Extract Artifacts (The raw documents)
                    if hasattr(tool_msg, "artifact") and tool_msg.artifact:
                        docs = tool_msg.artifact
                        collected_artifacts.extend(docs)
                        
                        # Show the resources inside the status box immediately
                        for i, doc in enumerate(docs):
                            with step_container.container():
                                st.markdown(f"""
                                <div class='resource-box'>
                                    <b>📄 Page {doc.metadata.get('page', '?')}</b><br>
                                    <small>{doc.page_content[:150]}...</small>
                                </div>
                                """, unsafe_allow_html=True)
                                
            # Get final answer
            final_state = st.session_state.agent.invoke({"messages": [HumanMessage(content=prompt)]})
            final_response_text = final_state["messages"][-1].content
            
            step_container.update(label="✅ Answer Generated", state="complete", expanded=False)
            st.markdown(final_response_text)

            # Append to history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": final_response_text,
                "resources": collected_artifacts
            })

        except Exception as e:
            st.error(f"An error occurred: {e}")
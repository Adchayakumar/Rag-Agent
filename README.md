📄 Agentic RAG Assistant
A Transparent Agentic RAG (Retrieval-Augmented Generation) application built with Streamlit and LangChain.

Unlike traditional "Black Box" RAG apps, this assistant explicitly visualizes its reasoning process—showing you exactly when it searches, what tools it calls, and which specific document pages it uses to generate an answer.

🚀 Key Features
🧠 Transparent Reasoning: Real-time visualization of the Agent's thought process (Tool Calls, Searching, Retrieving) using Streamlit's status containers.

📂 Document Analysis: Upload PDF documents to create a searchable knowledge base on the fly.

🔍 Smart Retrieval: Uses ChromaDB and HuggingFace Embeddings (all-mpnet-base-v2) for high-accuracy semantic search.

🤖 ReAct Agent Architecture: Powered by LangGraph, enabling the model to dynamically decide when to search for information versus answering directly.

📚 Source Citations: Every answer comes with expandable source citations showing the exact page number and content used.

🛠️ Tech Stack
Frontend: Streamlit

LLM: Google Gemini 2.5 Flash Lite 

Orchestration: LangChain & LangGraph

Vector Database: Chroma 

Embeddings: HuggingFace (sentence-transformers)

PDF Processing: PyPDFLoader

🔮 Roadmap: Moving to Multimodal
We are actively upgrading this application from a standard text-based RAG to a Multimodal Agentic RAG (Image + Text). Upcoming features include:

Image Analysis: The agent will be able to "see" and interpret images, charts, and diagrams within uploaded PDFs.

Visual Retrieval: Querying based on visual content.

ℹ️ Requirement Update: Due to these advanced agentic capabilities and the transition to multimodal models, the application currently requires a Google API Key to function. This ensures access to the Gemini models (Flash/Pro) needed for high-level reasoning and future vision support


⚙️ Installation

1. Clone the Repository

git clone [https://github.com/Adchayakumar/Rag-Agent.git]
cd Rag-Agent

2. Create a Virtual Environment (Recommended)
python -m venv venv

3. Install Dependencies
pip install -r requirements.txt

4. Set Up Environment Variables

Create a .env file in the root directory and add your Google API Key:

GOOGLE_API_KEY=your_actual_api_key_here


🏃‍♂️ Usage

Run the Streamlit App:

streamlit run app.py


Upload a PDF:
Use the sidebar to upload your PDF document. The app will process, split, and embed the text automatically.


Ask Questions:
Chat with the agent. Watch the "🕵️ Agent Working..." expander to see the real-time tool usage and retrieval steps.

📂 Project Structure

├── main.py                  # Main application logic (UI + RAG Pipeline)
├── .env                    # API Keys (Not tracked by Git)
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation




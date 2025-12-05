

# 📄 Agentic RAG Assistant

A **Transparent Agentic RAG (Retrieval-Augmented Generation)** application built with **Streamlit** and **LangChain**.

Unlike traditional *black-box* RAG apps, this assistant explicitly **visualizes its reasoning process**—showing exactly when it searches, what tools it calls, and which document pages it uses to generate an answer.

---

## 🚀 Key Features

### 🧠 Transparent Reasoning

Real-time visualization of the agent’s thought process (Tool Calls, Searching, Retrieving) using Streamlit status containers.

### 📂 Document Analysis

Upload PDF documents to create a searchable knowledge base on the fly.

### 🔍 Smart Retrieval

Uses **ChromaDB** and **HuggingFace Embeddings (all-mpnet-base-v2)** for high-accuracy semantic search.

### 🤖 ReAct Agent Architecture

Powered by **LangGraph**, enabling the model to dynamically decide when to search for information vs. answering directly.

### 📚 Source Citations

Every answer includes expandable citations showing the exact page numbers and referenced content.

---

## 🛠️ Tech Stack

| Component          | Technology                        |
| ------------------ | --------------------------------- |
| **Frontend**       | Streamlit                         |
| **LLM**            | Google Gemini 2.5 Flash Lite      |
| **Orchestration**  | LangChain & LangGraph             |
| **Vector DB**      | Chroma                            |
| **Embeddings**     | HuggingFace Sentence Transformers |
| **PDF Processing** | PyPDFLoader                       |

---

## 🔮 Roadmap: Moving to Multimodal

The application is being upgraded from **text-based RAG** to **Multimodal Agentic RAG (Image + Text)**.

### Upcoming Features

* **Image Analysis:** Interpret images, charts, and diagrams inside PDFs
* **Visual Retrieval:** Query based on visual content

### ❗ Requirement Update

Due to advanced agentic capabilities and multimodal upgrades, the application **requires a Google API Key** to access Gemini (Flash/Pro) models for high-level reasoning and future vision capabilities.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Adchayakumar/Rag-Agent.git
cd Rag-Agent
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the root directory:

```
GOOGLE_API_KEY=your_actual_api_key_here
```

---

## 🏃‍♂️ Usage

### Run the Streamlit App

```bash
streamlit run app.py
```

### Upload a PDF

Use the sidebar to upload your PDF. The app will automatically process, split, and embed the content.

### Ask Questions

Chat with the agent and view the **“🕵️ Agent Working…”** expander to see real-time tool usage and retrieval reasoning.

---

## 📂 Project Structure

```
├── main.py                # Main application logic (UI + RAG Pipeline)
├── .env                   # API Keys (ignored by Git)
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
```

---



# 🚀 Document Intelligence: ML Classification & RAG System

This project is an intelligent document management platform that automates file organization and enables natural language interaction with business documents. It combines **Classical Machine Learning (ML)** for categorization with **Retrieval-Augmented Generation (RAG)** architectures.



## 💡 The Problem
In corporate environments (such as auditing or accounting), documents are often disorganized in generic folders. Extracting specific information from hundreds of PDFs requires manual effort and exhaustive searching.

## 🛠️ My Solution
I have built an "End-to-End" application that:
1.  **Automatic Classification:** Upon uploading a PDF, a ML model (LinearSVC) trained on the "Northwind Traders Business Documents" dataset identifies the category (*Invoice, Shipping Order, Purchase Order, Report*).
2.  **Physical Organization:** The system automatically creates a folder structure based on the model's classification.
3.  **Intelligent Auditing (RAG):** Uses **LangChain** and **OpenAI** to index content into a vector database (**ChromaDB**), allowing users to chat with each document in isolation with high precision.

## 📁 Project Structure

The project is organized as a monorepo to separate the AI logic from the user interface:

```text
document-classifier/
├── backend/                # FastAPI, ML Logic & RAG
│   ├── artifacts/          # Serialized ML models (.joblib)
│   ├── ml/                 # Classification & PDF processing logic
│   ├── storage/            # Classified documents (Auto-generated)
│   └── main.py             # API Entry point
├── frontend/               # Angular 21 Web App
│   ├── src/                # Components, Services & Signals logic
│   └── ...
└── test_documents/         # Sample PDFs for testing classification & RAG
```

## 🏗️ Tech Stack

### Frontend
* **Angular 18:** Intensive use of Signals for modern, high-performance reactivity.
* **PrimeNG:** Professional UI featuring `p-tree` for hierarchical navigation and `p-fileupload`.
* **Tailwind CSS:** Custom "Dark Mode" design and responsive layout.

### Backend (AI Core)
* **FastAPI:** High-performance Python framework for the API.
* **Scikit-Learn:** Classification pipeline (TF-IDF + LinearSVC) serialized with `joblib`.
* **LangChain & OpenAI:** Orchestration of the RAG chain and Embeddings.
* **SQLite (SQLAlchemy):** Metadata persistence and document traceability.
* **ChromaDB:** Vector store for document embeddings and semantic search.

---

## 📊 Dataset & Training
The project utilizes a processed version of the **Northwind Traders** business document dataset.

* **Categories:** Invoice, Report, PurchaseOrder, ShippingOrder.
* **Preprocessing:** Text extraction via `PyMuPDF`, cleaning, and tokenization.
* **Model:** Multi-class classifier optimized for structured business text.

---

## 🚀 Getting Started

### 1. Prerequisites
* Python 3.10+
* Node.js & Angular CLI
* OpenAI API Key

### 2. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
# Create a .env file and add your OPENAI_API_KEY
uvicorn main:app --reload

## 3. Frontend Setup

cd frontend
npm install
ng serve

Navigate to http://localhost:4200

🧪 Testing
I have included a directory named test_documents/ containing sample PDFs for each category. You can upload these files through the UI to verify:

Correct classification: The file will move to the corresponding folder in the tree automatically.

RAG performance: Ask questions about total amounts, addresses, or dates within those specific files to test the AI Audit Assistant.
# ⚖️ Policy Compliance Checker RAG System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-121212?style=for-the-badge&logo=chainlink&logoColor=white)
![Cohere](https://img.shields.io/badge/Cohere-FF6B6B?style=for-the-badge&logo=openai&logoColor=white)

**A Legal/Policy Compliance Checker using Retrieval-Augmented Generation (RAG) with Pre-labeled Data, Local Embeddings, and LLM Reranking**

[GitHub Repo](https://github.com/NadeemAhmad3/Policy_Compliance_Checker_RAG_System)

</div>

---

## 📖 Overview

**Policy Compliance Checker RAG System** is an advanced legal AI assistant for automated contract review and compliance analysis. It leverages the CUAD dataset, local HuggingFace embeddings, FAISS vector search, and Cohere LLMs to provide instant and custom compliance checks on legal documents.

### 🎯 Key Highlights

- **510+ Contracts, 41 Clause Categories** (CUAD dataset)
- **Instant Compliance Checks** using pre-labeled data
- **RAG Pipeline** for custom queries and clause search
- **LLM Reranking** for high-precision retrieval
- **Modern Streamlit UI** for interactive analysis
- **Source Attribution** and detailed compliance reports

---

## ✨ Features

- **Dual Mode:**
  - **INSTANT:** Uses pre-labeled compliance data for fast results
  - **RAG:** Retrieves relevant clauses using local embeddings + LLM reranking
- **Parent-Child Document Indexing** for optimal retrieval
- **Custom Rule Checking** and full contract compliance analysis
- **Clause Comparison** across contracts
- **Summarization and Explanation** of findings
- **Interactive, Responsive UI** with custom CSS

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                POLICY COMPLIANCE CHECKER RAG SYSTEM         │
├──────────────┬─────────────────────────────┬────────────────┤
│  User Query  │  Streamlit Web Interface    │  API/CLI       │
├──────────────┴─────────────┬───────────────┴────────────────┤
│   ComplianceAgent (ReAct)  │ ComplianceChecker (RAG/Instant)│
├────────────────────────────┴────────────────────────────────┤
│  FAISS Vectorstore  │  Cohere LLM/Rerank  │  Pre-labeled   │
│  (Local Embeddings) │  (Custom QA, Rules) │  CUAD Data     │
├─────────────────────────────────────────────────────────────┤
│  Contracts (TXT/PDF) │  master_clauses.csv │  CUAD_v1.json │
└─────────────────────────────────────────────────────────────┘
```

---

## 📚 Dataset: CUAD v1

- **510 contracts** (TXT, PDF)
- **41 clause categories** (e.g., Governing Law, Assignment, Confidentiality)
- **master_clauses.csv:** Pre-labeled Yes/No + extracted text for each contract/category
- **CUAD_v1.json:** SQuAD-style Q&A pairs for each contract
- **Full contract text and PDF files** for raw retrieval

---

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/NadeemAhmad3/Policy_Compliance_Checker_RAG_System.git
   cd Policy_Compliance_Checker_RAG_System/task2
   ```
2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On Linux/Mac
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up environment variables:**
   - Create a `.env` file in the root directory with your Cohere API key:
     ```env
     COHERE_API_KEY=your_cohere_api_key_here
     ```
5. **Prepare the dataset:**
   - Ensure the `dataset/` folder contains all CUAD files (see structure below).

---

## 🚀 Usage

### 1. **Ingest Contracts and Build Vectorstore**
   ```bash
   python src/ingest.py --force
   ```
   - This will process contracts, build local embeddings, and create a FAISS index.

### 2. **Run the Streamlit App**
   ```bash
   streamlit run app.py --server.port 8502
   ```
   - Open your browser at [http://localhost:8502](http://localhost:8502)

### 3. **Features in the UI**
   - **Home:** Project overview
   - **Features:** System highlights
   - **Compliance:** Upload/select a contract, run compliance check (all rules or specific)
   - **AI Agent:** Ask complex compliance questions (multi-step reasoning)
   - **About:** Dataset, rules, and credits

---

## 🗂️ Project Structure

```
task2/
├── app.py                # Streamlit web app
├── requirements.txt      # Python dependencies
├── style.css             # Custom CSS for UI
├── dataset/              # CUAD data (CSV, JSON, TXT, PDF)
│   ├── master_clauses.csv
│   ├── CUAD_v1.json
│   ├── full_contract_txt/
│   └── full_contract_pdf/
├── rules/                # compliance_rules.json
├── src/
│   ├── agent.py          # ComplianceAgent (multi-step LLM)
│   ├── compliance_checker.py # Core compliance logic (RAG/Instant)
│   ├── config.py         # Configuration and paths
│   ├── ingest.py         # Data ingestion and vectorstore builder
│   └── __init__.py
├── vectorstore/          # FAISS index files
└── ...
```

---

## ⚡ Configuration

- **config.py:** Set model names, paths, and parameters
- **.env:** Store your Cohere API key securely
- **requirements.txt:** All dependencies listed for reproducibility

---

## 📝 Credits & References

- **CUAD Dataset:** [The Atticus Project](https://www.atticusprojectai.org/cuad)
- **LangChain, Cohere, HuggingFace, FAISS**
- **Original Author:** [Nadeem Ahmad](https://github.com/NadeemAhmad3)

---

## 📢 License

This project is for research and educational purposes. Please check the CUAD dataset license for data usage terms.

---

## 💡 Contact

For questions or contributions, open an issue or contact via GitHub.

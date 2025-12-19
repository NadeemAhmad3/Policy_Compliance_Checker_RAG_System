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

**Policy Compliance Checker RAG System** is an advanced legal AI assistant for automated contract review and compliance analysis. It is designed to help lawyers, compliance teams, and analysts rapidly locate relevant clauses, determine compliance with specific rules, and explain findings using a combination of:

- Pre-labeled CUAD dataset (instant checks)
- Local HuggingFace embeddings & FAISS (fast, private retrieval)
- Cohere models for reranking and answer generation
- A ReAct-style multi-step agent for complex reasoning

This README documents how to set up, run, and extend the Task 2 (legal compliance) project.

---

## 🚀 Quick Start (TL;DR)

1. Clone and move to task2:
   ```bash
   git clone https://github.com/NadeemAhmad3/Policy_Compliance_Checker_RAG_System.git
   cd Policy_Compliance_Checker_RAG_System/task2
   ```
2. Create and activate a venv:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Add `COHERE_API_KEY` to `.env`
5. Build vectorstore (optional, RAG mode):
   ```bash
   python src/ingest.py --force
   ```
6. Run the app:
   ```bash
   streamlit run app.py --server.port 8502
   ```

---

## ✨ Features (detailed)

- INSTANT mode: Use `master_clauses.csv` for instant, deterministic compliance checks (no LLM calls required)
- RAG mode: Use local embeddings + FAISS to retrieve relevant contract chunks, with Cohere Rerank as a second-stage filter
- Multi-step ComplianceAgent: Decomposes complex queries into plan steps (search, check rule, compare, summarize)
- Parent-Child indexing: Maintains large parent context with small child chunks for precise retrieval
- Source citation & relevance scoring
- Streamlit UI for interactive exploration and reporting

---

## 🏗️ Architecture & Data Flow

High-level system diagram:

```
┌─────────────────────────────────────────────────────────────────────┐
│                          USER INTERFACE                             │
│  Streamlit App (app.py) / CLI (agent.py) / Scripts (ingest.py)      │
└─────────────────────────────────────────────────────────────────────┘
                │                         │                      
                ▼                         ▼                      
┌──────────────────────────┐   ┌──────────────────────────┐        
│ ComplianceAgent (ReAct)  │   │ ComplianceChecker (RAG)  │        
│  - Plans actions         │   │  - Retriever (FAISS)     │        
│  - Orchestrates checks   │   │  - Cohere rerank/LLM     │        
└─────────┬────────────────┘   └──────────┬───────────────┘        
          │                                 │                      
          ▼                                 ▼                      
   ┌──────────────┐                ┌─────────────────────────┐     
   │ master_clauses.csv (INSTANT) │  │ vectorstore (FAISS)    │     
   │ CUAD_v1.json                 │  │ - Child embeddings     │     
   └──────────────────────────────┘  └─────────────────────────┘     
```

Parent-Child retrieval (short recap):
- Parents: large chunks (e.g., 2000 chars) to preserve context
- Children: small chunks (e.g., 400 chars) embedded into FAISS for precise matching
- Query → search children → find parent(s) → rerank with Cohere → assemble context → LLM answer

---

## 🔧 Tech Stack

- Python 3.10+
- LangChain (core, community)
- Cohere (LLM + Rerank)
- HuggingFace local embeddings (sentence-transformers)
- FAISS (vector store)
- Streamlit UI
- Pandas for CSV processing

---

## 📁 Project Structure (task2)

```
task2/
├── README.md
├── app.py                    # Streamlit web app (UI)
├── requirements.txt
├── style.css
├── dataset/                  # CUAD dataset files
│   ├── master_clauses.csv
│   ├── CUAD_v1.json
│   ├── full_contract_txt/
│   └── full_contract_pdf/
├── rules/                    # compliance_rules.json
├── src/
│   ├── agent.py              # ComplianceAgent (ReAct planning + execution)
│   ├── compliance_checker.py # RAG + INSTANT compliance logic
│   ├── config.py             # Constants & paths
│   ├── ingest.py             # Ingestion, splitting, embeddings, FAISS builder
│   └── __init__.py
├── vectorstore/              # Generated FAISS index and docstore (created by ingest)
└── analyze_dataset.py        # Analysis utilities (task2)
```

---

## 🔬 Detailed Components

### src/config.py
Key variables you'll likely tweak:

```python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "rerank-english-v3.0"
CHAT_MODEL = "command-r-plus-08-2024"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
RETRIEVAL_K = 8
RERANK_TOP_N = 3
COMPLIANCE_THRESHOLD = 0.7
PARENT_CHUNK_SIZE = 2000
CHILD_CHUNK_SIZE = 400
BATCH_SIZE = 32
```

Changes to these affect speed vs. accuracy tradeoffs. Using a larger `RETRIEVAL_K` and `RERANK_TOP_N` improves recall at a cost of runtime.

### src/ingest.py (what it does)
- Reads `master_clauses.csv` and raw `full_contract_txt/`
- Splits documents into parents and children using `RecursiveCharacterTextSplitter`
- Embeds children with HuggingFace embeddings (local, CPU/GPU aware)
- Builds FAISS index and saves to `vectorstore/faiss_index`

Example usage:
```bash
python src/ingest.py --force
```
Sample output lines you should see:
```
Loading local embedding model: sentence-transformers/all-MiniLM-L6-v2
Building embeddings: batch 1/16
Saving FAISS index to vectorstore/faiss_index/
✓ Vectorstore saved (num_vectors=12345)
```

### src/compliance_checker.py (core logic)
- Uses `master_clauses.csv` for INSTANT checks (deterministic answers)
- Uses vectorstore + Cohere Rerank for RAG mode
- Provides methods to:
  - list rules
  - check a specific rule for a contract
  - run full compliance across all rules for a contract
  - search contracts by keyword/term

Example (pseudo-code):
```python
from src.compliance_checker import ComplianceChecker
checker = ComplianceChecker(use_vectorstore=True)
checker.check_rule(filename='SOME_CONTRACT.txt', rule_id='governing_law')
```
Return format (dict):
```json
{
  "filename": "SOME_CONTRACT.txt",
  "rule_id": "governing_law",
  "compliant": true,
  "evidence": "This Agreement shall be governed by the laws of California.",
  "confidence": 0.92
}
```

### src/agent.py (ComplianceAgent)
- A ReAct-style multi-step planner/executor that uses Cohere Chat to
  decompose complex queries into specific actions (search, check rule, summarize)
- Useful for multi-part user queries ("Compare confidential clauses across contracts and summarize risks")

Run locally for debugging:
```bash
python src/agent.py
```

---

## 🧑‍💻 Installation & Setup (detailed)

### 1) Clone repo & create venv
```bash
git clone https://github.com/NadeemAhmad3/Policy_Compliance_Checker_RAG_System.git
cd Policy_Compliance_Checker_RAG_System/task2
python -m venv .venv
.venv\Scripts\activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Add `.env` with your Cohere key
```.env
COHERE_API_KEY=your_cohere_api_key_here
```

### 4) (Optional but recommended) Build vectorstore for RAG mode
```bash
python src/ingest.py --force
```
- If you skip this, the system will operate in **INSTANT** mode using `master_clauses.csv`.

### 5) Run the Streamlit UI
```bash
streamlit run app.py --server.port 8502
```

---

## 💬 Example Workflows & Queries

### Instant compliance check (no LLM)
- Select a contract from the dropdown → Click **Run Full Compliance**
- Output: Table of 41 rules with compliant flag, evidence snippets, and confidence

### RAG-based query
- Ask: "Does contract X restrict assignment without consent?"
- Process: Agent searches relevant children, reranks, and returns evidence with a final answer
- Output: Answer + top 3 source snippets with relevance scores

### Agent multi-step task
- Query: "Compare confidentiality clauses across these 3 contracts and summarize risk areas"
- Agent plan: search_contracts → extract clauses → compare_contracts → summarize_findings
- Output: Structured comparison + short summary and recommended actions

---

## ✅ Testing & Validation

- Quick import test:
```bash
python -c "from src.compliance_checker import ComplianceChecker; print('Import OK')"
```
- Python syntax check:
```bash
python -m py_compile app.py
```
- Unit tests: (Not included by default) — you can add tests under `tests/` and run with `pytest`.

---

## 🛠 Troubleshooting

- Vectorstore not found / corrupted:
  - Re-run `python src/ingest.py --force` to rebuild
  - Ensure `vectorstore/faiss_index` is writable
- Cohere API issues:
  - Confirm `COHERE_API_KEY` in `.env`
  - Watch for rate limits or temporary outages
- Memory/Performance:
  - Use smaller embedding model or enable GPU if available
  - Reduce `RETRIEVAL_K` / `RERANK_TOP_N` to speed up queries

---

## 🤝 Contributing

Contributions are welcome. Suggested workflow:

1. Fork the repo
2. Create a branch: `git checkout -b feature/my-feature`
3. Add tests and update README when adding features
4. Open a pull request with a clear description

---

## 📝 License & Data

- Project code: **MIT License** (check LICENSE in repo)
- Data: **CUAD dataset terms** apply — verify dataset license before commercial use

---

## 👤 Author & Contact

**Nadeem Ahmad** — Author and maintainer
- GitHub: https://github.com/NadeemAhmad3
- Email: nadeemahmad2703@gmail.com

---

## 🙏 Acknowledgments

- The Atticus Project — CUAD dataset
- Cohere — LLM & reranking
- LangChain — RAG primitives
- HuggingFace — Local embeddings
- FAISS — Vector indexing

---

If you'd like, I can also:
- Add example screenshots of the Streamlit UI
- Generate a minimal `docker-compose` for local deployment
- Add a `Makefile` with common commands

Please tell me which additions you want and I'll update the file accordingly.

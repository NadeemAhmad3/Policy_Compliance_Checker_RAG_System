"""
Configuration for Task 2: Policy Compliance Checker RAG System
Uses LOCAL embeddings (HuggingFace) to avoid API limits
Uses Cohere for LLM and Reranking only
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from parent .env file
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

# ============================================
# API CONFIGURATION
# ============================================
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# ============================================
# MODEL CONFIGURATION
# ============================================
# LOCAL Embeddings - No API limits!
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dimensions, fast
# Alternative: "sentence-transformers/all-mpnet-base-v2"  # 768 dimensions, better quality

# Cohere Models (using same key as Task 1)
RERANK_MODEL = "rerank-english-v3.0"
CHAT_MODEL = "command-r-plus-08-2024"

# ============================================
# PATH CONFIGURATION
# ============================================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "dataset"
CONTRACTS_DIR = DATA_DIR / "full_contract_txt"
CONTRACTS_PDF_DIR = DATA_DIR / "full_contract_pdf"  # PDF files directory
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
RULES_DIR = BASE_DIR / "rules"

# Pre-labeled data files (CUAD dataset already has answers!)
CUAD_JSON_PATH = DATA_DIR / "CUAD_v1.json"
MASTER_CLAUSES_PATH = DATA_DIR / "master_clauses.csv"
LABEL_EXCEL_DIR = DATA_DIR / "label_group_xlsx"

# ============================================
# CHUNKING CONFIGURATION (Parent-Child Indexing)
# ============================================
PARENT_CHUNK_SIZE = 2500  # Larger for legal context
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE = 500   # Smaller for precise retrieval
CHILD_CHUNK_OVERLAP = 50

# ============================================
# RETRIEVAL CONFIGURATION
# ============================================
RETRIEVAL_K = 10         # Initial retrieval count
RERANK_TOP_N = 5         # After reranking
BATCH_SIZE = 20          # Documents per batch during ingestion

# ============================================
# COMPLIANCE CONFIGURATION
# ============================================
COMPLIANCE_THRESHOLD = 0.7  # Confidence threshold for compliance
MAX_RULES_PER_CHECK = 15    # Maximum rules to check at once

# ============================================
# VALIDATION
# ============================================
def validate_config():
    """Validate configuration settings."""
    errors = []
    
    if not COHERE_API_KEY:
        errors.append("COHERE_API_KEY not found in .env file")
    
    if not CONTRACTS_DIR.exists():
        errors.append(f"Contracts directory not found: {CONTRACTS_DIR}")
    
    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(errors))
    
    print("✓ Configuration validated successfully")
    print(f"  - Cohere API Key: {'*' * 10}...{COHERE_API_KEY[-4:] if COHERE_API_KEY else 'NOT SET'}")
    print(f"  - Local Embedding Model: {EMBEDDING_MODEL}")
    print(f"  - Contracts Directory: {CONTRACTS_DIR}")
    print(f"  - Vectorstore Directory: {VECTORSTORE_DIR}")

if __name__ == "__main__":
    validate_config()

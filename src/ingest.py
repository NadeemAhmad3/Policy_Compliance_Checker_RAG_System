"""
Ingest module for Task 2: Policy Compliance Checker RAG System
Uses LOCAL HuggingFace embeddings (no API limits!)
Implements Parent-Child Document Indexing for optimal retrieval

DATASET INSIGHT:
- CUAD_v1.json has SQuAD format with 510 contracts and 41 pre-labeled QA pairs each
- master_clauses.csv has 510 contracts × 83 columns (41 categories with Yes/No + Answer)
- We can use pre-labeled data directly OR do RAG retrieval on raw contracts
"""

import os
import json
import pickle
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.stores import InMemoryStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document

# Handle imports whether running directly or as package
try:
    from config import (
        EMBEDDING_MODEL,
        CONTRACTS_DIR,
        CONTRACTS_PDF_DIR,
        VECTORSTORE_DIR,
        CUAD_JSON_PATH,
        MASTER_CLAUSES_PATH,
        PARENT_CHUNK_SIZE,
        PARENT_CHUNK_OVERLAP,
        CHILD_CHUNK_SIZE,
        CHILD_CHUNK_OVERLAP,
        BATCH_SIZE
    )
except ModuleNotFoundError:
    from src.config import (
        EMBEDDING_MODEL,
        CONTRACTS_DIR,
        CONTRACTS_PDF_DIR,
        VECTORSTORE_DIR,
        CUAD_JSON_PATH,
        MASTER_CLAUSES_PATH,
        PARENT_CHUNK_SIZE,
        PARENT_CHUNK_OVERLAP,
        CHILD_CHUNK_SIZE,
        CHILD_CHUNK_OVERLAP,
        BATCH_SIZE
    )


def get_local_embeddings() -> HuggingFaceEmbeddings:
    """
    Initialize LOCAL HuggingFace embeddings - NO API LIMITS!
    Uses sentence-transformers model that runs entirely on your machine.
    """
    print(f"✓ Loading local embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU available
        encode_kwargs={'normalize_embeddings': True}
    )
    print("✓ Local embeddings loaded successfully (no API limits!)")
    return embeddings


# ============================================
# LOAD PRE-LABELED DATA (CUAD Dataset)
# ============================================

def load_cuad_json() -> Dict[str, Any]:
    """
    Load CUAD_v1.json - SQuAD format with pre-labeled Q&A pairs.
    Each contract has 41 questions with extracted answer spans.
    """
    print(f"Loading CUAD JSON from: {CUAD_JSON_PATH}")
    with open(CUAD_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data['data'])} contracts with pre-labeled QA pairs")
    return data


def load_master_clauses() -> pd.DataFrame:
    """
    Load master_clauses.csv - 510 contracts × 83 columns.
    Contains Yes/No answers + extracted clause text for all 41 categories.
    """
    print(f"Loading master clauses from: {MASTER_CLAUSES_PATH}")
    df = pd.read_csv(MASTER_CLAUSES_PATH)
    print(f"✓ Loaded {len(df)} contracts with {len(df.columns)} columns")
    return df


def get_contract_compliance_from_csv(filename: str, df: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Get pre-labeled compliance data for a specific contract from master_clauses.csv.
    This provides INSTANT compliance checking without LLM calls!
    
    Args:
        filename: Contract filename (with or without .pdf extension)
        df: Pre-loaded dataframe (optional, will load if not provided)
    
    Returns:
        Dictionary with compliance data for all 41 categories
    """
    if df is None:
        df = load_master_clauses()
    
    # Find the contract row using exact match
    # If user selects from dropdown, filename is already correct
    row = df[df['Filename'] == filename]
    if row.empty:
        # Try fallback: if .txt, convert to .pdf
        filename_pdf = filename.replace('.txt', '.pdf')
        row = df[df['Filename'] == filename_pdf]
    if row.empty:
            # Fallback: try partial match (ignore extension, match start of filename)
            base_name = filename.rsplit('.', 1)[0]
            found = False
            for key in df['Filename']:
                key_base = key.rsplit('.', 1)[0]
                if base_name in key_base or key_base in base_name:
                    row = df[df['Filename'] == key]
                    found = True
                    break
            if not found:
                return {"error": f"Contract not found: {filename}"}
    row = row.iloc[0]
    
    # Build compliance result
    result = {
        "filename": row['Filename'],
        "document_name": row.get('Document Name-Answer', ''),
        "categories": {}
    }
    
    # Categories mapping (column pairs: Category -> Category-Answer)
    category_columns = [
        ('Document Name', 'Document Name-Answer'),
        ('Parties', 'Parties-Answer'),
        ('Agreement Date', 'Agreement Date-Answer'),
        ('Effective Date', 'Effective Date-Answer'),
        ('Expiration Date', 'Expiration Date-Answer'),
        ('Renewal Term', 'Renewal Term-Answer'),
        ('Governing Law', 'Governing Law-Answer'),
        ('Most Favored Nation', 'Most Favored Nation-Answer'),
        ('Non-Compete', 'Non-Compete-Answer'),
        ('Exclusivity', 'Exclusivity-Answer'),
        ('No-Solicit Of Customers', 'No-Solicit Of Customers-Answer'),
        ('No-Solicit Of Employees', 'No-Solicit Of Employees-Answer'),
        ('Non-Disparagement', 'Non-Disparagement-Answer'),
        ('Termination For Convenience', 'Termination For Convenience-Answer'),
        ('Anti-Assignment', 'Anti-Assignment-Answer'),
        ('Ip Ownership Assignment', 'Ip Ownership Assignment-Answer'),
        ('License Grant', 'License Grant-Answer'),
        ('Audit Rights', 'Audit Rights-Answer'),
        ('Uncapped Liability', 'Uncapped Liability-Answer'),
        ('Cap On Liability', 'Cap On Liability-Answer'),
        ('Insurance', 'Insurance-Answer'),
    ]
    
    for cat, answer_col in category_columns:
        if cat in row.index and answer_col in row.index:
            clause_text = row[cat]
            answer = row[answer_col]
            
            # Determine if present (Yes/No)
            has_clause = not (pd.isna(clause_text) or str(clause_text).strip() in ['[]', '', 'nan'])
            
            result["categories"][cat] = {
                "present": "Yes" if has_clause else "No",
                "answer": str(answer) if not pd.isna(answer) else "",
                "clause_text": str(clause_text) if has_clause else ""
            }
    
    return result


def load_contracts(contracts_dir: Path = CONTRACTS_DIR, max_files: Optional[int] = None) -> List[Document]:
    """
    Load contract documents from text files.
    
    Args:
        contracts_dir: Directory containing contract TXT files
        max_files: Optional limit on number of files to load (for testing)
    
    Returns:
        List of Document objects with content and metadata
    """
    documents = []
    txt_files = list(contracts_dir.glob("*.txt"))
    
    if max_files:
        txt_files = txt_files[:max_files]
    
    print(f"Loading {len(txt_files)} contract files...")
    
    for idx, file_path in enumerate(txt_files):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract contract type from filename
            filename = file_path.stem
            contract_type = extract_contract_type(filename)
            
            doc = Document(
                page_content=content,
                metadata={
                    'source': str(file_path),
                    'filename': filename,
                    'contract_type': contract_type,
                    'doc_id': idx
                }
            )
            documents.append(doc)
            
            if (idx + 1) % 50 == 0:
                print(f"  Loaded {idx + 1}/{len(txt_files)} contracts...")
                
        except Exception as e:
            print(f"  Warning: Could not load {file_path.name}: {e}")
    
    print(f"✓ Loaded {len(documents)} contracts successfully")
    return documents


def load_pdfs(pdf_dir: Path = CONTRACTS_PDF_DIR, max_files: Optional[int] = None) -> List[Document]:
    """
    Load contract documents from PDF files.
    Supports the CUAD dataset's PDF structure (Part_I, Part_II, Part_III folders).
    
    Args:
        pdf_dir: Directory containing contract PDF files
        max_files: Optional limit on number of files to load (for testing)
    
    Returns:
        List of Document objects with content and metadata
    """
    try:
        from langchain_community.document_loaders import PyPDFLoader
    except ImportError:
        print("⚠ PyPDFLoader not available. Install pypdf: pip install pypdf")
        print("  Falling back to TXT files...")
        return load_contracts(max_files=max_files)
    
    documents = []
    pdf_files = []
    
    # Check for Part_I, Part_II, Part_III subdirectories (CUAD structure)
    for part_dir in ["Part_I", "Part_II", "Part_III"]:
        part_path = pdf_dir / part_dir
        if part_path.exists():
            pdf_files.extend(list(part_path.glob("*.pdf")))
    
    # Also check root directory
    pdf_files.extend(list(pdf_dir.glob("*.pdf")))
    
    if not pdf_files:
        print(f"⚠ No PDF files found in {pdf_dir}")
        print("  Falling back to TXT files...")
        return load_contracts(max_files=max_files)
    
    if max_files:
        pdf_files = pdf_files[:max_files]
    
    print(f"Loading {len(pdf_files)} PDF contract files...")
    
    for idx, file_path in enumerate(pdf_files):
        try:
            loader = PyPDFLoader(str(file_path))
            pages = loader.load()
            
            # Combine all pages into single document
            content = "\n\n".join([page.page_content for page in pages])
            
            # Extract contract type from filename
            filename = file_path.stem
            contract_type = extract_contract_type(filename)
            
            doc = Document(
                page_content=content,
                metadata={
                    'source': str(file_path),
                    'filename': filename,
                    'contract_type': contract_type,
                    'doc_id': idx,
                    'num_pages': len(pages),
                    'file_type': 'pdf'
                }
            )
            documents.append(doc)
            
            if (idx + 1) % 20 == 0:
                print(f"  Loaded {idx + 1}/{len(pdf_files)} PDFs...")
                
        except Exception as e:
            print(f"  Warning: Could not load {file_path.name}: {e}")
    
    print(f"✓ Loaded {len(documents)} PDF contracts successfully")
    return documents


def extract_contract_type(filename: str) -> str:
    """Extract contract type from filename."""
    # Common contract types from CUAD dataset
    contract_types = [
        "Agency Agreement", "Affiliate Agreement", "Co-Branding Agreement",
        "Collaboration Agreement", "Consulting Agreement", "Content License Agreement",
        "Cooperation Agreement", "Development Agreement", "Distributor Agreement",
        "Endorsement Agreement", "Franchise Agreement", "Hosting Agreement",
        "IP Agreement", "Joint Venture Agreement", "License Agreement",
        "Maintenance Agreement", "Manufacturing Agreement", "Marketing Agreement",
        "Non-Compete Agreement", "Outsourcing Agreement", "Promotion Agreement",
        "Reseller Agreement", "Service Agreement", "Services Agreement",
        "Sponsorship Agreement", "Strategic Alliance Agreement", "Supply Agreement",
        "Transportation Agreement"
    ]
    
    filename_lower = filename.lower()
    for ctype in contract_types:
        if ctype.lower().replace(" ", "") in filename_lower.replace(" ", "").replace("-", "").replace("_", ""):
            return ctype
    
    return "Unknown"


def create_parent_child_retriever(
    documents: List[Document],
    embeddings: HuggingFaceEmbeddings
) -> tuple:
    """
    Create Parent-Child Document Retriever for optimal retrieval.
    
    - Child chunks: Small (500 chars) - used for precise vector search
    - Parent chunks: Large (2500 chars) - returned for full context
    
    This technique finds relevant sections with small chunks but returns
    the larger parent context for better LLM comprehension.
    """
    print("\n=== Creating Parent-Child Index ===")
    
    # Parent splitter - larger chunks for context
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Child splitter - smaller chunks for precise retrieval
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # In-memory store for parent documents
    docstore = InMemoryStore()
    
    # Create FAISS vectorstore for child chunks
    # Initialize with a dummy document first
    dummy_doc = Document(page_content="initialization", metadata={})
    vectorstore = FAISS.from_documents([dummy_doc], embeddings)
    
    # Create Parent Document Retriever
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )
    
    # Add documents in batches
    print(f"Adding {len(documents)} documents in batches of {BATCH_SIZE}...")
    
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i:i + BATCH_SIZE]
        retriever.add_documents(batch)
        print(f"  Processed batch {i // BATCH_SIZE + 1}/{(len(documents) + BATCH_SIZE - 1) // BATCH_SIZE}")
    
    print(f"✓ Parent-Child index created")
    print(f"  - Parent chunk size: {PARENT_CHUNK_SIZE}")
    print(f"  - Child chunk size: {CHILD_CHUNK_SIZE}")
    
    return retriever, vectorstore, docstore


def save_vectorstore(vectorstore: FAISS, docstore: InMemoryStore, output_dir: Path = VECTORSTORE_DIR):
    """Save vectorstore and docstore to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save FAISS index
    faiss_path = output_dir / "faiss_index"
    vectorstore.save_local(str(faiss_path))
    print(f"✓ FAISS index saved to {faiss_path}")
    
    # Save docstore
    docstore_path = output_dir / "docstore.pkl"
    with open(docstore_path, 'wb') as f:
        pickle.dump(docstore, f)
    print(f"✓ Docstore saved to {docstore_path}")


def load_vectorstore(embeddings: HuggingFaceEmbeddings, input_dir: Path = VECTORSTORE_DIR) -> tuple:
    """Load vectorstore and docstore from disk."""
    # Load FAISS index
    faiss_path = input_dir / "faiss_index"
    vectorstore = FAISS.load_local(
        str(faiss_path), 
        embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"✓ FAISS index loaded from {faiss_path}")
    
    # Load docstore
    docstore_path = input_dir / "docstore.pkl"
    with open(docstore_path, 'rb') as f:
        docstore = pickle.load(f)
    print(f"✓ Docstore loaded from {docstore_path}")
    
    return vectorstore, docstore


def ingest_contracts(max_files: Optional[int] = None, force_rebuild: bool = False, use_pdf: bool = False):
    """
    Main ingestion pipeline.
    
    Args:
        max_files: Limit number of files to process (for testing)
        force_rebuild: If True, rebuild even if vectorstore exists
        use_pdf: If True, load from PDF files instead of TXT
    """
    print("\n" + "=" * 60)
    print("TASK 2: Policy Compliance Checker - Document Ingestion")
    print("Using LOCAL embeddings (no API limits!)")
    print("=" * 60 + "\n")
    
    # Check if vectorstore already exists
    faiss_path = VECTORSTORE_DIR / "faiss_index"
    if faiss_path.exists() and not force_rebuild:
        print("✓ Vectorstore already exists. Use force_rebuild=True to recreate.")
        return
    
    # Initialize local embeddings
    embeddings = get_local_embeddings()
    
    # Load contracts (PDF or TXT)
    if use_pdf:
        documents = load_pdfs(max_files=max_files)
    else:
        documents = load_contracts(max_files=max_files)
    
    if not documents:
        print("✗ No documents found to ingest!")
        return
    
    # Create parent-child retriever
    retriever, vectorstore, docstore = create_parent_child_retriever(documents, embeddings)
    
    # Save to disk
    save_vectorstore(vectorstore, docstore)
    
    print("\n" + "=" * 60)
    print("✓ INGESTION COMPLETE")
    print(f"  - Documents processed: {len(documents)}")
    print(f"  - Source format: {'PDF' if use_pdf else 'TXT'}")
    print(f"  - Vectorstore location: {VECTORSTORE_DIR}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest contracts into vectorstore")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of files (for testing)")
    parser.add_argument("--force", action="store_true", help="Force rebuild vectorstore")
    parser.add_argument("--pdf", action="store_true", help="Use PDF files instead of TXT")
    
    args = parser.parse_args()
    
    ingest_contracts(max_files=args.max_files, force_rebuild=args.force, use_pdf=args.pdf)

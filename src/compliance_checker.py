"""
Compliance Checker module for Task 2: Policy Compliance Checker RAG System
Implements rule-based compliance checking with Cohere LLM + Reranking

KEY INSIGHT: CUAD dataset has PRE-LABELED data!
- master_clauses.csv: 510 contracts with 41 categories already labeled (Yes/No + extracted text)
- CUAD_v1.json: SQuAD format with answer spans
- We can use this for INSTANT compliance checks OR RAG for custom queries
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_cohere import ChatCohere, CohereRerank
from langchain_classic.retrievers import ContextualCompressionRetriever, ParentDocumentRetriever
from langchain_core.stores import InMemoryStore
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# Handle imports whether running directly or as package
try:
    from config import (
        COHERE_API_KEY,
        EMBEDDING_MODEL,
        RERANK_MODEL,
        CHAT_MODEL,
        VECTORSTORE_DIR,
        RULES_DIR,
        MASTER_CLAUSES_PATH,
        CUAD_JSON_PATH,
        RETRIEVAL_K,
        RERANK_TOP_N,
        COMPLIANCE_THRESHOLD
    )
    from ingest import load_vectorstore, get_local_embeddings, load_master_clauses, get_contract_compliance_from_csv
except ModuleNotFoundError:
    from src.config import (
        COHERE_API_KEY,
        EMBEDDING_MODEL,
        RERANK_MODEL,
        CHAT_MODEL,
        VECTORSTORE_DIR,
        RULES_DIR,
        MASTER_CLAUSES_PATH,
        CUAD_JSON_PATH,
        RETRIEVAL_K,
        RERANK_TOP_N,
        COMPLIANCE_THRESHOLD
    )
    from src.ingest import load_vectorstore, get_local_embeddings, load_master_clauses, get_contract_compliance_from_csv


class ComplianceChecker:
    """
    Compliance Checker using RAG with Cohere LLM and Reranking.
    Uses local embeddings for retrieval, Cohere for reranking and generation.
    
    TWO MODES:
    1. INSTANT MODE: Use pre-labeled CUAD data (master_clauses.csv)
    2. RAG MODE: Use vector retrieval + LLM for custom queries
    """
    
    def __init__(self, use_vectorstore: bool = True):
        """Initialize the compliance checker with all components."""
        print("\n=== Initializing Compliance Checker ===")
        
        # Load pre-labeled data (INSTANT compliance checks!)
        print("Loading pre-labeled CUAD data...")
        self.master_clauses_df = load_master_clauses()
        
        # Load local embeddings (no API limits!)
        self.embeddings = get_local_embeddings()
        
        # Try to load vectorstore if it exists
        self.vectorstore = None
        self.retriever = None
        
        if use_vectorstore:
            try:
                self.vectorstore, self.docstore = load_vectorstore(self.embeddings)
                
                # Create base retriever from vectorstore
                self.base_retriever = self.vectorstore.as_retriever(
                    search_kwargs={"k": RETRIEVAL_K}
                )
                
                # Initialize Cohere Reranker
                self.reranker = CohereRerank(
                    cohere_api_key=COHERE_API_KEY,
                    model=RERANK_MODEL,
                    top_n=RERANK_TOP_N
                )
                
                # Create compression retriever with reranking
                self.retriever = ContextualCompressionRetriever(
                    base_compressor=self.reranker,
                    base_retriever=self.base_retriever
                )
                print("✓ Vectorstore loaded for RAG mode")
            except Exception as e:
                print(f"⚠ Vectorstore not available: {e}")
                print("  Using INSTANT mode only (pre-labeled data)")
        
        # Initialize Cohere LLM
        self.llm = ChatCohere(
            cohere_api_key=COHERE_API_KEY,
            model=CHAT_MODEL,
            temperature=0.1  # Low temperature for consistent compliance analysis
        )
        
        # Load compliance rules
        self.rules = self._load_rules()
        
        print(f"✓ Loaded {len(self.rules)} compliance rules")
        print(f"✓ Pre-labeled data: {len(self.master_clauses_df)} contracts")
        print("✓ Compliance Checker initialized successfully\n")
    
    def _load_rules(self) -> List[Dict]:
        """Load compliance rules from JSON file."""
        rules_file = RULES_DIR / "compliance_rules.json"
        with open(rules_file, 'r') as f:
            rules_data = json.load(f)
        return rules_data.get('rules', [])
    
    # ============================================
    # INSTANT MODE: Use Pre-Labeled CUAD Data
    # ============================================
    
    def instant_compliance_check(self, contract_name: str) -> Dict[str, Any]:
        """
        INSTANT compliance check using pre-labeled CUAD data.
        NO LLM CALLS NEEDED - uses master_clauses.csv directly!
        
        Args:
            contract_name: Contract filename (partial match supported)
        
        Returns:
            Complete compliance report with all 41 categories
        """
        result = get_contract_compliance_from_csv(contract_name, self.master_clauses_df)
        
        if "error" in result:
            return result
        
        # Calculate compliance score
        categories = result.get("categories", {})
        critical_present = 0
        critical_missing = 0
        
        # Critical categories that should be present
        critical_categories = [
            "Parties", "Agreement Date", "Governing Law", 
            "Cap On Liability", "Termination For Convenience"
        ]
        
        for cat in critical_categories:
            if cat in categories:
                if categories[cat]["present"] == "Yes":
                    critical_present += 1
                else:
                    critical_missing += 1
        
        result["summary"] = {
            "total_categories_checked": len(categories),
            "clauses_found": sum(1 for c in categories.values() if c["present"] == "Yes"),
            "clauses_missing": sum(1 for c in categories.values() if c["present"] == "No"),
            "critical_present": critical_present,
            "critical_missing": critical_missing,
            "compliance_score": round(critical_present / len(critical_categories) * 100, 1)
        }
        
        return result
    
    def list_contracts(self, contract_type: str = None) -> List[str]:
        """List all available contracts, optionally filtered by type."""
        filenames = self.master_clauses_df['Filename'].tolist()
        
        if contract_type:
            filenames = [f for f in filenames if contract_type.lower() in f.lower()]
        
        return filenames[:50]  # Limit to 50 results
    
    # ============================================
    # COMPARISON TABLE: Compliant vs Non-Compliant
    # ============================================
    
    def generate_comparison_table(self, contract_name: str, output_format: str = "dict") -> Any:
        """
        Generate a comparison table of COMPLIANT vs NON-COMPLIANT sections.
        This is a key deliverable for the compliance checker.
        
        Args:
            contract_name: Contract filename to analyze
            output_format: "dict", "dataframe", or "markdown"
        
        Returns:
            Comparison table in requested format
        """
        result = self.instant_compliance_check(contract_name)
        
        if "error" in result:
            return result
        
        compliant = []
        non_compliant = []
        
        for category, info in result.get("categories", {}).items():
            row = {
                "Category": category,
                "Status": info["present"],
                "Evidence": info["clause_text"][:200] + "..." if len(info.get("clause_text", "")) > 200 else info.get("clause_text", ""),
                "Remediation": self._get_remediation(category, info["present"])
            }
            
            if info["present"] == "Yes":
                compliant.append(row)
            else:
                non_compliant.append(row)
        
        comparison = {
            "contract": result.get("filename", contract_name),
            "document_name": result.get("document_name", ""),
            "summary": result.get("summary", {}),
            "compliant_sections": compliant,
            "non_compliant_sections": non_compliant,
            "total_compliant": len(compliant),
            "total_non_compliant": len(non_compliant)
        }
        
        if output_format == "dataframe":
            import pandas as pd
            df_compliant = pd.DataFrame(compliant)
            df_compliant["Compliance"] = "COMPLIANT"
            df_non_compliant = pd.DataFrame(non_compliant)
            df_non_compliant["Compliance"] = "NON-COMPLIANT"
            return pd.concat([df_compliant, df_non_compliant], ignore_index=True)
        
        elif output_format == "markdown":
            return self._format_comparison_markdown(comparison)
        
        return comparison
    
    def _get_remediation(self, category: str, status: str) -> str:
        """Get remediation suggestion for a category."""
        if status == "Yes":
            return "No action needed - clause is present"
        
        remediation_map = {
            "Parties": "Add clear identification of all contracting parties with full legal names",
            "Agreement Date": "Add the date when the agreement was signed",
            "Effective Date": "Specify when the contract becomes effective",
            "Expiration Date": "Add contract end date or term duration",
            "Renewal Term": "Consider adding automatic renewal provisions",
            "Governing Law": "CRITICAL: Add governing law/jurisdiction clause",
            "Non-Compete": "Consider adding non-compete restrictions if needed",
            "Exclusivity": "Consider whether exclusive rights are required",
            "Termination For Convenience": "Add provisions for early termination",
            "Anti-Assignment": "Add restrictions on contract assignment",
            "Ip Ownership Assignment": "Clarify IP ownership and assignment terms",
            "License Grant": "Define scope of any licenses granted",
            "Cap On Liability": "CRITICAL: Add liability cap to limit exposure",
            "Uncapped Liability": "Review if uncapped liability is intentional",
            "Insurance": "Consider requiring insurance coverage",
            "Audit Rights": "Add audit rights for compliance verification",
        }
        
        return remediation_map.get(category, f"Consider adding {category} clause")
    
    def _format_comparison_markdown(self, comparison: Dict) -> str:
        """Format comparison as markdown table."""
        md = f"""# Compliance Comparison Report

**Contract:** {comparison['contract']}
**Document Name:** {comparison['document_name']}

## Summary
- Total Compliant: {comparison['total_compliant']}
- Total Non-Compliant: {comparison['total_non_compliant']}
- Compliance Score: {comparison['summary'].get('compliance_score', 'N/A')}%

## ✅ COMPLIANT Sections ({comparison['total_compliant']})

| Category | Evidence (excerpt) |
|----------|-------------------|
"""
        for item in comparison['compliant_sections'][:10]:
            evidence = item['Evidence'][:100].replace('\n', ' ').replace('|', '/') if item['Evidence'] else 'Present'
            md += f"| {item['Category']} | {evidence}... |\n"
        
        md += f"""
## ❌ NON-COMPLIANT Sections ({comparison['total_non_compliant']})

| Category | Remediation |
|----------|-------------|
"""
        for item in comparison['non_compliant_sections']:
            md += f"| {item['Category']} | {item['Remediation']} |\n"
        
        return md
    
    def compare_multiple_contracts(self, contract_names: List[str]) -> Dict[str, Any]:
        """
        Compare compliance across multiple contracts.
        
        Args:
            contract_names: List of contract filenames to compare
        
        Returns:
            Comparison matrix across contracts
        """
        results = []
        
        for name in contract_names[:10]:  # Limit to 10 contracts
            check = self.instant_compliance_check(name)
            if "error" not in check:
                row = {
                    "contract": check.get("filename", name)[:50],
                    "compliance_score": check["summary"]["compliance_score"],
                    "clauses_found": check["summary"]["clauses_found"],
                    "clauses_missing": check["summary"]["clauses_missing"],
                    "critical_present": check["summary"]["critical_present"],
                    "critical_missing": check["summary"]["critical_missing"]
                }
                results.append(row)
        
        return {
            "contracts_analyzed": len(results),
            "comparison": results,
            "average_compliance": round(sum(r["compliance_score"] for r in results) / len(results), 1) if results else 0
        }
    
    def retrieve_context(self, query: str, contract_filter: Optional[str] = None) -> List[Document]:
        """
        Retrieve relevant context using Parent-Child retrieval + Reranking.
        
        Args:
            query: The compliance question or search query
            contract_filter: Optional filter to search within specific contract
        
        Returns:
            List of relevant document chunks
        """
        # Get relevant documents with reranking
        docs = self.retriever.invoke(query)
        
        # Filter by contract if specified
        if contract_filter:
            docs = [d for d in docs if contract_filter.lower() in d.metadata.get('filename', '').lower()]
        
        return docs
    
    def check_single_rule(
        self, 
        rule: Dict, 
        contract_content: str = None,
        contract_name: str = None
    ) -> Dict[str, Any]:
        """
        Check a single compliance rule against contract content.
        
        Args:
            rule: Rule definition dictionary
            contract_content: Full contract text (if available)
            contract_name: Contract filename for filtered retrieval
        
        Returns:
            Compliance check result dictionary
        """
        # Retrieve relevant context
        if contract_name:
            context_docs = self.retrieve_context(rule['question'], contract_filter=contract_name)
        else:
            context_docs = self.retrieve_context(rule['question'])
        
        # Build context string
        context = "\n\n---\n\n".join([doc.page_content for doc in context_docs])
        
        # Build prompt for compliance check
        prompt = self._build_compliance_prompt(rule, context)
        
        # Get LLM response
        response = self.llm.invoke(prompt)
        
        # Parse response
        result = self._parse_compliance_response(response.content, rule)
        
        # Add source citations
        result['sources'] = [
            {
                'filename': doc.metadata.get('filename', 'Unknown'),
                'excerpt': doc.page_content[:200] + "..."
            }
            for doc in context_docs[:3]
        ]
        
        return result
    
    def _build_compliance_prompt(self, rule: Dict, context: str) -> str:
        """Build the compliance check prompt."""
        prompt = f"""You are a legal compliance expert analyzing a contract. 

COMPLIANCE RULE:
- Rule ID: {rule['id']}
- Rule Name: {rule['name']}
- Category: {rule['category']}
- Description: {rule['description']}
- Severity: {rule['severity'].upper()}
- Required: {"Yes" if rule.get('required', False) else "No"}

QUESTION TO ANSWER:
{rule['question']}

RELEVANT CONTRACT SECTIONS:
{context}

INSTRUCTIONS:
1. Analyze the contract sections above to answer the compliance question.
2. Determine if the contract is COMPLIANT, NON-COMPLIANT, or PARTIALLY COMPLIANT with this rule.
3. Provide specific evidence from the contract text.
4. If information is missing or unclear, note what additional information would be needed.

Respond in the following JSON format:
{{
    "status": "COMPLIANT" | "NON-COMPLIANT" | "PARTIALLY_COMPLIANT" | "INSUFFICIENT_DATA",
    "confidence": 0.0-1.0,
    "finding": "Brief summary of what was found",
    "evidence": "Exact quote or paraphrase from the contract",
    "recommendation": "Suggested action if non-compliant or partially compliant"
}}

RESPONSE:"""
        return prompt
    
    def _parse_compliance_response(self, response: str, rule: Dict) -> Dict[str, Any]:
        """Parse LLM response into structured result."""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                result = json.loads(json_match.group())
                result['rule_id'] = rule['id']
                result['rule_name'] = rule['name']
                result['category'] = rule['category']
                result['severity'] = rule['severity']
                return result
        except json.JSONDecodeError:
            pass
        
        # Fallback parsing
        return {
            'rule_id': rule['id'],
            'rule_name': rule['name'],
            'category': rule['category'],
            'severity': rule['severity'],
            'status': 'PARSE_ERROR',
            'confidence': 0.0,
            'finding': response[:500],
            'evidence': '',
            'recommendation': 'Manual review required'
        }
    
    def check_all_rules(
        self, 
        contract_name: Optional[str] = None,
        severity_filter: Optional[List[str]] = None,
        required_only: bool = False
    ) -> Dict[str, Any]:
        """
        Check all compliance rules against a contract.
        
        Args:
            contract_name: Optional contract filename to filter retrieval
            severity_filter: Optional list of severities to check (e.g., ['critical', 'high'])
            required_only: If True, only check required rules
        
        Returns:
            Complete compliance report
        """
        rules_to_check = self.rules
        
        # Apply filters
        if severity_filter:
            rules_to_check = [r for r in rules_to_check if r['severity'] in severity_filter]
        
        if required_only:
            rules_to_check = [r for r in rules_to_check if r.get('required', False)]
        
        print(f"Checking {len(rules_to_check)} rules...")
        
        results = []
        for i, rule in enumerate(rules_to_check):
            print(f"  [{i+1}/{len(rules_to_check)}] Checking: {rule['name']}...")
            result = self.check_single_rule(rule, contract_name=contract_name)
            results.append(result)
        
        # Calculate summary statistics
        summary = self._calculate_summary(results)
        
        return {
            'contract': contract_name or 'All contracts',
            'rules_checked': len(results),
            'summary': summary,
            'results': results
        }
    
    def _calculate_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics from compliance results."""
        status_counts = {}
        severity_issues = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        total_confidence = 0
        
        for r in results:
            status = r.get('status', 'UNKNOWN')
            status_counts[status] = status_counts.get(status, 0) + 1
            total_confidence += r.get('confidence', 0)
            
            if status in ['NON-COMPLIANT', 'PARTIALLY_COMPLIANT']:
                severity_issues[r.get('severity', 'medium')] += 1
        
        compliant_count = status_counts.get('COMPLIANT', 0)
        total_rules = len(results)
        
        return {
            'compliance_score': round(compliant_count / total_rules * 100, 1) if total_rules > 0 else 0,
            'average_confidence': round(total_confidence / total_rules, 2) if total_rules > 0 else 0,
            'status_breakdown': status_counts,
            'issues_by_severity': severity_issues,
            'critical_issues': severity_issues['critical'],
            'high_issues': severity_issues['high']
        }
    
    def query(self, question: str) -> str:
        """
        Answer a free-form compliance question.
        
        Args:
            question: Natural language question about contracts
        
        Returns:
            LLM-generated answer with evidence
        """
        # Retrieve context
        docs = self.retriever.invoke(question)
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        
        # Build QA prompt
        prompt = f"""You are a legal contract analysis expert. Answer the following question based on the contract context provided.

QUESTION: {question}

CONTRACT CONTEXT:
{context}

INSTRUCTIONS:
- Provide a clear, accurate answer based on the contract text
- Quote specific sections when relevant
- If the information is not found, say so clearly
- Cite which contract(s) the information comes from

ANSWER:"""
        
        response = self.llm.invoke(prompt)
        
        # Add source info
        sources = "\n\nSources:\n" + "\n".join([
            f"- {doc.metadata.get('filename', 'Unknown')}"
            for doc in docs[:3]
        ])
        
        return response.content + sources


def main():
    """Test the compliance checker."""
    # Initialize without vectorstore first (just use pre-labeled data)
    checker = ComplianceChecker(use_vectorstore=False)
    
    # Test INSTANT compliance check (no LLM needed!)
    print("\n" + "="*60)
    print("INSTANT COMPLIANCE CHECK (Using Pre-Labeled CUAD Data)")
    print("="*60)
    
    # List some contracts
    print("\nAvailable contracts (sample):")
    contracts = checker.list_contracts("Distributor")
    for c in contracts[:5]:
        print(f"  - {c[:60]}...")
    
    # Check first contract
    if contracts:
        print(f"\n\nChecking: {contracts[0][:50]}...")
        result = checker.instant_compliance_check(contracts[0])
        
        print(f"\nCompliance Summary:")
        print(f"  - Document: {result.get('document_name', 'N/A')}")
        if 'summary' in result:
            s = result['summary']
            print(f"  - Clauses Found: {s['clauses_found']}/{s['total_categories_checked']}")
            print(f"  - Critical Present: {s['critical_present']}")
            print(f"  - Compliance Score: {s['compliance_score']}%")
        
        print("\nCategory Details:")
        for cat, info in list(result.get('categories', {}).items())[:10]:
            status = "✓" if info['present'] == "Yes" else "✗"
            print(f"  {status} {cat}: {info['present']}")
            if info['answer'] and info['answer'] != 'nan':
                print(f"      Answer: {info['answer'][:80]}...")
        
        # ============================================
        # COMPARISON TABLE - Key Deliverable!
        # ============================================
        print("\n" + "="*60)
        print("COMPARISON TABLE: Compliant vs Non-Compliant Sections")
        print("="*60)
        
        comparison = checker.generate_comparison_table(contracts[0], output_format="dict")
        
        print(f"\n✅ COMPLIANT SECTIONS ({comparison['total_compliant']}):")
        print("-" * 50)
        for item in comparison['compliant_sections'][:5]:
            print(f"  • {item['Category']}")
        if comparison['total_compliant'] > 5:
            print(f"  ... and {comparison['total_compliant'] - 5} more")
        
        print(f"\n❌ NON-COMPLIANT SECTIONS ({comparison['total_non_compliant']}):")
        print("-" * 50)
        for item in comparison['non_compliant_sections'][:5]:
            print(f"  • {item['Category']}")
            print(f"    Remediation: {item['Remediation']}")
        
        # Markdown format
        print("\n" + "="*60)
        print("MARKDOWN REPORT")
        print("="*60)
        md_report = checker.generate_comparison_table(contracts[0], output_format="markdown")
        print(md_report[:1500])  # Print first part
        
        # Multiple contract comparison
        print("\n" + "="*60)
        print("MULTI-CONTRACT COMPARISON")
        print("="*60)
        multi_comparison = checker.compare_multiple_contracts(contracts[:5])
        print(f"\nContracts Analyzed: {multi_comparison['contracts_analyzed']}")
        print(f"Average Compliance Score: {multi_comparison['average_compliance']}%")
        print("\nContract-by-Contract:")
        for c in multi_comparison['comparison']:
            print(f"  • {c['contract'][:40]}... | Score: {c['compliance_score']}% | Found: {c['clauses_found']}")


if __name__ == "__main__":
    main()

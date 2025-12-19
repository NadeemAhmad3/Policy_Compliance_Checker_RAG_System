"""
Agent module for Task 2: Policy Compliance Checker RAG System
Implements a multi-step reasoning agent for complex compliance queries
"""

import json
from typing import List, Dict, Any, Optional
from enum import Enum

from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# Handle imports whether running directly or as package
try:
    from config import COHERE_API_KEY, CHAT_MODEL
    from compliance_checker import ComplianceChecker
except ModuleNotFoundError:
    from src.config import COHERE_API_KEY, CHAT_MODEL
    from src.compliance_checker import ComplianceChecker


class AgentAction(Enum):
    """Actions the agent can take."""
    SEARCH_CONTRACTS = "search_contracts"
    CHECK_SPECIFIC_RULE = "check_specific_rule"
    CHECK_ALL_RULES = "check_all_rules"
    COMPARE_CONTRACTS = "compare_contracts"
    SUMMARIZE_FINDINGS = "summarize_findings"
    ANSWER_QUESTION = "answer_question"


class ComplianceAgent:
    """
    Multi-step reasoning agent for complex compliance analysis.
    Uses ReAct-style prompting for step-by-step reasoning.
    """
    
    def __init__(self, checker: ComplianceChecker = None):
        """Initialize the agent with a compliance checker."""
        print("\n=== Initializing Compliance Agent ===")
        
        self.checker = checker or ComplianceChecker()
        
        self.llm = ChatCohere(
            cohere_api_key=COHERE_API_KEY,
            model=CHAT_MODEL,
            temperature=0.2
        )
        
        self.conversation_history = []
        self.analysis_results = []
        
        print("✓ Compliance Agent initialized\n")
    
    def _plan_actions(self, query: str) -> List[Dict[str, Any]]:
        """
        Plan which actions to take based on the user query.
        Uses the LLM to decompose complex queries into steps.
        """
        planning_prompt = f"""You are a legal compliance analysis planner. Given a user query, decompose it into specific actions.

USER QUERY: {query}

AVAILABLE ACTIONS:
1. search_contracts: Search for specific clauses or terms in contracts
2. check_specific_rule: Check a specific compliance rule (provide rule_id or rule_name)
3. check_all_rules: Run full compliance check on a contract
4. compare_contracts: Compare compliance across multiple contracts
5. summarize_findings: Summarize analysis results
6. answer_question: Answer a general question about contracts

AVAILABLE RULES:
{json.dumps([{"id": r["id"], "name": r["name"], "category": r["category"]} for r in self.checker.rules], indent=2)}

Respond with a JSON array of actions in order:
[
    {{"action": "action_name", "params": {{"key": "value"}}, "reason": "why this action"}}
]

Only include actions that are necessary. Be efficient.

PLANNED ACTIONS:"""
        
        response = self.llm.invoke(planning_prompt)
        
        try:
            import re
            json_match = re.search(r'\[[\s\S]*\]', response.content)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
        
        # Default: just answer the question directly
        return [{"action": "answer_question", "params": {"query": query}, "reason": "Direct answer"}]
    
    def _execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single planned action."""
        action_type = action.get('action')
        params = action.get('params', {})
        
        result = {
            'action': action_type,
            'params': params,
            'status': 'success'
        }
        
        try:
            if action_type == 'search_contracts':
                query = params.get('query', params.get('search_term', ''))
                # Use INSTANT mode if vectorstore not available
                if self.checker.retriever is None:
                    # Search in pre-labeled data instead
                    contracts = self.checker.list_contracts(query)
                    result['output'] = {
                        'num_results': len(contracts),
                        'mode': 'INSTANT (pre-labeled data)',
                        'results': [
                            {'contract': c[:80], 'type': 'pre-labeled'}
                            for c in contracts[:5]
                        ]
                    }
                else:
                    docs = self.checker.retrieve_context(query)
                    result['output'] = {
                        'num_results': len(docs),
                        'mode': 'RAG',
                        'results': [
                            {
                                'contract': doc.metadata.get('filename', 'Unknown'),
                                'excerpt': doc.page_content[:300] + "..."
                            }
                            for doc in docs[:5]
                        ]
                    }
            
            elif action_type == 'check_specific_rule':
                rule_id = params.get('rule_id')
                rule_name = params.get('rule_name')
                contract = params.get('contract_name')
                
                # Use INSTANT mode check
                if contract:
                    check_result = self.checker.instant_compliance_check(contract)
                    if rule_name and 'categories' in check_result:
                        # Filter to specific category
                        for cat, info in check_result.get('categories', {}).items():
                            if rule_name.lower() in cat.lower():
                                result['output'] = {
                                    'category': cat,
                                    'status': info['present'],
                                    'evidence': info.get('clause_text', '')[:500]
                                }
                                break
                        else:
                            result['output'] = check_result
                    else:
                        result['output'] = check_result
                else:
                    # Find the rule and check
                    rule = None
                    for r in self.checker.rules:
                        if r['id'] == rule_id or r['name'].lower() == str(rule_name).lower():
                            rule = r
                            break
                    
                    if rule:
                        # Get sample contracts and check
                        contracts = self.checker.list_contracts()[:3]
                        checks = []
                        for c in contracts:
                            check = self.checker.instant_compliance_check(c)
                            if 'categories' in check:
                                for cat, info in check['categories'].items():
                                    if rule['category'].lower() in cat.lower():
                                        checks.append({
                                            'contract': c[:50],
                                            'status': info['present'],
                                            'answer': info.get('answer', '')[:100]
                                        })
                                        break
                        result['output'] = {'rule': rule['name'], 'checks': checks}
                    else:
                        result['status'] = 'error'
                        result['output'] = f"Rule not found: {rule_id or rule_name}"
            
            elif action_type == 'check_all_rules':
                contract = params.get('contract_name')
                severity = params.get('severity_filter')
                required = params.get('required_only', False)
                
                check_result = self.checker.check_all_rules(
                    contract_name=contract,
                    severity_filter=severity,
                    required_only=required
                )
                result['output'] = check_result
            
            elif action_type == 'compare_contracts':
                contracts = params.get('contracts', [])
                rule_id = params.get('rule_id')
                
                comparison = []
                for contract in contracts[:5]:  # Limit to 5 contracts
                    if rule_id:
                        rule = next((r for r in self.checker.rules if r['id'] == rule_id), None)
                        if rule:
                            check_result = self.checker.check_single_rule(rule, contract_name=contract)
                            comparison.append({
                                'contract': contract,
                                'result': check_result
                            })
                    else:
                        check_result = self.checker.check_all_rules(
                            contract_name=contract,
                            required_only=True
                        )
                        comparison.append({
                            'contract': contract,
                            'summary': check_result['summary']
                        })
                
                result['output'] = comparison
            
            elif action_type == 'answer_question':
                query = params.get('query', '')
                answer = self.checker.query(query)
                result['output'] = answer
            
            elif action_type == 'summarize_findings':
                # Summarize all previous results
                result['output'] = self._summarize_results()
            
            else:
                result['status'] = 'error'
                result['output'] = f"Unknown action: {action_type}"
        
        except Exception as e:
            result['status'] = 'error'
            result['output'] = str(e)
        
        return result
    
    def _summarize_results(self) -> str:
        """Summarize all analysis results collected so far."""
        if not self.analysis_results:
            return "No analysis results to summarize."
        
        summary_prompt = f"""Summarize the following compliance analysis results:

{json.dumps(self.analysis_results, indent=2, default=str)}

Provide a clear, concise summary including:
1. Overall compliance status
2. Key findings and issues
3. Recommended actions

SUMMARY:"""
        
        response = self.llm.invoke(summary_prompt)
        return response.content
    
    def _generate_final_response(self, query: str, results: List[Dict]) -> str:
        """Generate final response based on all action results."""
        response_prompt = f"""You are a legal compliance expert. Based on the analysis performed, provide a comprehensive response to the user's query.

USER QUERY: {query}

ANALYSIS RESULTS:
{json.dumps(results, indent=2, default=str)}

Provide a clear, professional response that:
1. Directly answers the user's question
2. Cites specific evidence from the contracts
3. Highlights any compliance concerns
4. Provides actionable recommendations if relevant

RESPONSE:"""
        
        response = self.llm.invoke(response_prompt)
        return response.content
    
    def run(self, query: str, verbose: bool = True) -> str:
        """
        Run the agent to answer a compliance query.
        
        Args:
            query: User's compliance-related question
            verbose: If True, print intermediate steps
        
        Returns:
            Final response string
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"QUERY: {query}")
            print('='*60)
        
        # Step 1: Plan actions
        if verbose:
            print("\n📋 Planning actions...")
        
        actions = self._plan_actions(query)
        
        if verbose:
            print(f"   Planned {len(actions)} actions")
        
        # Step 2: Execute actions
        results = []
        for i, action in enumerate(actions):
            if verbose:
                print(f"\n🔄 Executing [{i+1}/{len(actions)}]: {action['action']}")
                if action.get('reason'):
                    print(f"   Reason: {action['reason']}")
            
            result = self._execute_action(action)
            results.append(result)
            self.analysis_results.append(result)
            
            if verbose:
                if result['status'] == 'success':
                    print(f"   ✓ Completed successfully")
                else:
                    print(f"   ✗ Error: {result.get('output', 'Unknown error')}")
        
        # Step 3: Generate final response
        if verbose:
            print("\n📝 Generating response...")
        
        final_response = self._generate_final_response(query, results)
        
        # Add to conversation history
        self.conversation_history.append({
            'query': query,
            'actions': actions,
            'response': final_response
        })
        
        if verbose:
            print("\n" + "="*60)
            print("RESPONSE:")
            print("="*60)
        
        return final_response
    
    def reset(self):
        """Reset the agent's state."""
        self.conversation_history = []
        self.analysis_results = []
        print("✓ Agent state reset")


def main():
    """Test the compliance agent."""
    print("Starting agent test (this may take a moment to load models)...")
    
    # Initialize agent (this also initializes the checker)
    agent = ComplianceAgent()
    
    # Test single query to demonstrate multi-step reasoning
    test_query = "What governing laws are specified in the distribution agreements?"
    
    print(f"\n{'='*60}")
    print("TESTING AGENT WITH QUERY:")
    print(f"'{test_query}'")
    print('='*60)
    
    response = agent.run(test_query, verbose=True)
    print(response)


if __name__ == "__main__":
    main()

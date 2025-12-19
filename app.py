import streamlit as st
import pandas as pd
import sys
import os

# Add task2 directory to path for imports
task2_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, task2_dir)

from src.compliance_checker import ComplianceChecker
from src.agent import ComplianceAgent

# Page Config
st.set_page_config(
    page_title="LegalCheck - Policy Compliance AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load Custom CSS
def load_css(file_name):
    css_path = os.path.join(os.path.dirname(__file__), file_name)
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("style.css")

# Navigation Bar
st.markdown('''<style>
.nav-link{text-decoration:none;color:#1E3A5F;font-weight:500;font-size:0.9rem;padding:0.5rem 1rem;border-radius:8px;transition:all 0.3s ease;position:relative;}
.nav-link:hover{color:#D4AF37;background:rgba(212,175,55,0.08);}
.nav-link::after{content:'';position:absolute;bottom:0;left:50%;width:0;height:2px;background:#D4AF37;transition:all 0.3s ease;transform:translateX(-50%);}
.nav-link:hover::after{width:60%;}
.nav-btn{background:linear-gradient(135deg,#1E3A5F,#152A45);color:white;padding:0.5rem 1.25rem;border-radius:50px;font-size:0.85rem;font-weight:600;cursor:pointer;transition:all 0.3s ease;box-shadow:0 4px 15px rgba(30,58,95,0.3);}
.nav-btn:hover{transform:translateY(-2px);box-shadow:0 6px 20px rgba(30,58,95,0.4);}
</style>
<nav style="position:fixed;top:0;left:0;right:0;z-index:9999;background:rgba(255,255,255,0.95);backdrop-filter:blur(10px);padding:0.75rem 3rem;display:flex;justify-content:space-between;align-items:center;box-shadow:0 2px 20px rgba(0,0,0,0.08);border-bottom:1px solid #e2e8f0;">
<div style="display:flex;align-items:center;gap:0.5rem;">
<div style="background:linear-gradient(135deg,#1E3A5F,#152A45);padding:0.5rem;border-radius:10px;display:flex;align-items:center;justify-content:center;">
<svg width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" stroke="#D4AF37" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
</div>
<span style="font-size:1.5rem;font-weight:700;color:#1E3A5F;font-family:'Playfair Display',serif;">LegalCheck</span>
</div>
<div style="display:flex;gap:0.5rem;align-items:center;">
<a href="#home" class="nav-link">Home</a>
<a href="#features" class="nav-link">Features</a>
<a href="#checker" class="nav-link">Compliance</a>
<a href="#agent" class="nav-link">AI Agent</a>
<a href="#about" class="nav-link">About</a>
<div class="nav-btn" style="margin-left:1rem;">Get Started</div>
</div>
</nav>
<div style="height:70px;"></div>''', unsafe_allow_html=True)

# Hero Section
st.markdown('''<div id="home" style="min-height:90vh;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;background:linear-gradient(180deg,#F8FAFC 0%,#F1F5F9 50%,#F8FAFC 100%);padding:3rem 2rem;position:relative;overflow:hidden;">
<div style="position:absolute;top:10%;left:5%;width:300px;height:300px;background:radial-gradient(circle,rgba(30,58,95,0.08) 0%,transparent 70%);border-radius:50%;"></div>
<div style="position:absolute;bottom:15%;right:10%;width:400px;height:400px;background:radial-gradient(circle,rgba(212,175,55,0.08) 0%,transparent 70%);border-radius:50%;"></div>
<div style="position:absolute;top:15%;left:10%;font-size:2.5rem;opacity:0.15;">⚖️</div>
<div style="position:absolute;top:25%;right:15%;font-size:2rem;opacity:0.15;">📜</div>
<div style="position:absolute;bottom:25%;left:15%;font-size:2.2rem;opacity:0.15;">🔏</div>
<div style="position:absolute;bottom:20%;right:20%;font-size:1.8rem;opacity:0.15;">📋</div>
<div style="position:absolute;top:40%;left:8%;font-size:1.5rem;opacity:0.12;">🏛️</div>
<div style="position:absolute;top:60%;right:8%;font-size:1.5rem;opacity:0.12;">📑</div>
<div style="position:relative;z-index:10;max-width:900px;">
<div style="margin-bottom:1.5rem;">
<span style="background:linear-gradient(135deg,#1E3A5F,#152A45);color:#D4AF37;padding:0.5rem 1.5rem;border-radius:50px;font-size:0.85rem;font-weight:600;letter-spacing:1px;display:inline-block;">⚖️ AI-POWERED COMPLIANCE</span>
</div>
<h1 style="font-size:3.5rem;font-weight:700;color:#1E3A5F;line-height:1.2;margin-bottom:1.5rem;font-family:'Playfair Display',serif;">Policy Compliance<br><span style="background:linear-gradient(135deg,#D4AF37,#F4D03F);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">Intelligence Platform</span></h1>
<p style="font-size:1.2rem;color:#64748B;line-height:1.8;margin-bottom:2.5rem;max-width:700px;margin-left:auto;margin-right:auto;">Analyze contracts against 25+ compliance rules instantly. Powered by CUAD dataset with 510 pre-labeled legal contracts and advanced RAG technology for comprehensive policy verification.</p>
<div style="display:flex;gap:1rem;justify-content:center;flex-wrap:wrap;margin-bottom:3rem;">
<a href="#checker" style="text-decoration:none;background:linear-gradient(135deg,#1E3A5F,#152A45);color:#fff;padding:1rem 2.5rem;border-radius:50px;font-size:1.1rem;font-weight:600;box-shadow:0 8px 25px rgba(30,58,95,0.35);display:inline-flex;align-items:center;gap:0.5rem;">Start Analysis →</a>
<a href="#features" style="text-decoration:none;background:#fff;color:#1E3A5F;padding:1rem 2.5rem;border-radius:50px;font-size:1.1rem;font-weight:600;border:2px solid #E2E8F0;display:inline-flex;align-items:center;gap:0.5rem;">Learn More</a>
</div>
<div style="display:flex;gap:3rem;justify-content:center;flex-wrap:wrap;">
<div style="text-align:center;"><div style="font-size:2.5rem;font-weight:700;color:#1E3A5F;">510</div><div style="font-size:0.9rem;color:#64748B;">Legal Contracts</div></div>
<div style="text-align:center;"><div style="font-size:2.5rem;font-weight:700;color:#D4AF37;">25+</div><div style="font-size:0.9rem;color:#64748B;">Compliance Rules</div></div>
<div style="text-align:center;"><div style="font-size:2.5rem;font-weight:700;color:#10B981;">41</div><div style="font-size:0.9rem;color:#64748B;">Clause Categories</div></div>
</div>
</div>
</div>''', unsafe_allow_html=True)

# Features Section
st.markdown('''<div id="features" style="padding:5rem 2rem;background:#fff;">
<div style="text-align:center;max-width:1100px;margin:0 auto;">
<p style="font-size:1rem;color:#D4AF37;font-weight:600;letter-spacing:2px;text-transform:uppercase;margin-bottom:0.5rem;">POWERED BY CUAD DATASET</p>
<h2 style="font-size:2.5rem;font-weight:700;color:#1E3A5F;margin-bottom:1rem;font-family:'Playfair Display',serif;">Intelligent Contract Analysis</h2>
<p style="font-size:1.1rem;color:#64748B;margin-bottom:3rem;">Powered by <strong style="color:#1E3A5F;">RAG</strong> &amp; <strong style="color:#D4AF37;">Cohere AI</strong></p>
<div style="display:flex;gap:2rem;justify-content:center;flex-wrap:wrap;">
<div style="background:linear-gradient(135deg,#F0F9FF,#E0F2FE);padding:2rem;border-radius:20px;width:300px;text-align:center;border:1px solid #BAE6FD;box-shadow:0 4px 15px rgba(30,58,95,0.08);">
<div style="background:linear-gradient(135deg,#1E3A5F,#152A45);width:60px;height:60px;border-radius:15px;display:flex;align-items:center;justify-content:center;margin:0 auto 1rem;box-shadow:0 4px 15px rgba(30,58,95,0.3);"><span style="font-size:1.5rem;">📋</span></div>
<h3 style="font-size:1.25rem;font-weight:600;color:#1E3A5F;margin-bottom:0.5rem;">Instant Compliance</h3>
<p style="font-size:0.9rem;color:#64748B;">Pre-labeled dataset enables instant compliance checks without LLM delays.</p>
</div>
<div style="background:linear-gradient(135deg,#FFFBEB,#FEF3C7);padding:2rem;border-radius:20px;width:300px;text-align:center;border:1px solid #FDE68A;box-shadow:0 4px 15px rgba(212,175,55,0.08);">
<div style="background:linear-gradient(135deg,#D4AF37,#F4D03F);width:60px;height:60px;border-radius:15px;display:flex;align-items:center;justify-content:center;margin:0 auto 1rem;box-shadow:0 4px 15px rgba(212,175,55,0.3);"><span style="font-size:1.5rem;">🤖</span></div>
<h3 style="font-size:1.25rem;font-weight:600;color:#1E3A5F;margin-bottom:0.5rem;">AI Agent</h3>
<p style="font-size:0.9rem;color:#64748B;">Multi-step reasoning agent for complex compliance questions with ReAct workflow.</p>
</div>
<div style="background:linear-gradient(135deg,#ECFDF5,#D1FAE5);padding:2rem;border-radius:20px;width:300px;text-align:center;border:1px solid #A7F3D0;box-shadow:0 4px 15px rgba(16,185,129,0.08);">
<div style="background:linear-gradient(135deg,#10B981,#059669);width:60px;height:60px;border-radius:15px;display:flex;align-items:center;justify-content:center;margin:0 auto 1rem;box-shadow:0 4px 15px rgba(16,185,129,0.3);"><span style="font-size:1.5rem;">📊</span></div>
<h3 style="font-size:1.25rem;font-weight:600;color:#1E3A5F;margin-bottom:0.5rem;">Comparison Table</h3>
<p style="font-size:0.9rem;color:#64748B;">Side-by-side view of compliant vs non-compliant sections with evidence.</p>
</div>
</div>
</div>
</div>''', unsafe_allow_html=True)

# Initialize Checker (cached)
@st.cache_resource
def load_checker():
    return ComplianceChecker()

@st.cache_resource
def load_agent():
    return ComplianceAgent()

# Main Compliance Checker Section
st.markdown('''<div id="checker" style="padding:4rem 2rem;background:linear-gradient(180deg,#F8FAFC 0%,#F1F5F9 100%);">
<div style="max-width:1000px;margin:0 auto;text-align:center;">
<p style="font-size:0.9rem;color:#D4AF37;font-weight:600;letter-spacing:2px;text-transform:uppercase;margin-bottom:0.5rem;">📋 COMPLIANCE CHECKER</p>
<h2 style="font-size:2.2rem;font-weight:700;color:#1E3A5F;margin-bottom:0.75rem;font-family:'Playfair Display',serif;">Contract Analysis Tool</h2>
<p style="font-size:1rem;color:#64748B;margin-bottom:2rem;">Select a contract and check compliance against 25+ policy rules</p>
</div>
</div>''', unsafe_allow_html=True)

# Initialize checker
with st.spinner("⚖️ Initializing Compliance Engine..."):
    checker = load_checker()

# Contract Selection UI
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown("""
    <div style="background:white;padding:2rem;border-radius:16px;box-shadow:0 4px 20px rgba(30,58,95,0.1);border:1px solid #E2E8F0;margin-bottom:2rem;">
        <h3 style="color:#1E3A5F;margin-bottom:1rem;font-size:1.2rem;">🔍 Select Contract for Analysis</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Get contract list
    if checker.master_clauses_df is not None:
        contracts = checker.master_clauses_df['Filename'].tolist()
        
        # Contract type filter
        contract_types = ['All Types', 'AGREEMENT', 'LICENSE', 'AMENDMENT', 'CONTRACT', 'Other']
        selected_type = st.selectbox("📁 Filter by Contract Type:", contract_types)
        
        # Filter contracts
        if selected_type != 'All Types':
            filtered_contracts = [c for c in contracts if selected_type.upper() in c.upper()]
        else:
            filtered_contracts = contracts
        
        selected_contract = st.selectbox(
            "📜 Select Contract:",
            filtered_contracts,  # Show all available contracts for accuracy
            help="Choose a contract from the CUAD dataset to analyze"
        )
        
        # Show contract info
        if selected_contract:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#F0F9FF,#E0F2FE);padding:1rem;border-radius:12px;margin:1rem 0;border-left:4px solid #1E3A5F;">
                <strong style="color:#1E3A5F;">📄 Selected:</strong> {selected_contract[:80]}...
            </div>
            """, unsafe_allow_html=True)
        
        # Analysis buttons
        col_a, col_b = st.columns(2)
        with col_a:
            analyze_btn = st.button("⚖️ Run Compliance Check", use_container_width=True, type="primary")
        with col_b:
            compare_btn = st.button("📊 Compare Top 5 Contracts", use_container_width=True)
        
        # Run Analysis
        if analyze_btn and selected_contract:
            with st.spinner("🔍 Analyzing contract compliance..."):
                results = checker.instant_compliance_check(selected_contract)
                # Patch: Build 'compliant' and 'non_compliant' lists for UI
                categories = results.get('categories', {})
                compliant = []
                non_compliant = []
                for cat, info in categories.items():
                    if info.get('present') == 'Yes':
                        compliant.append({'category': cat, 'evidence': info.get('clause_text', '')})
                    else:
                        non_compliant.append({'category': cat, 'evidence': info.get('clause_text', '')})
                results['compliant'] = compliant
                results['non_compliant'] = non_compliant
                
                if results:
                    # Display score with gauge
                    score = results.get('summary', {}).get('compliance_score', 0)
                    score_color = "#22C55E" if score >= 70 else "#F59E0B" if score >= 50 else "#EF4444"
                    
                    st.markdown(f"""
                    <div style="background:white;padding:2rem;border-radius:16px;box-shadow:0 4px 20px rgba(30,58,95,0.1);margin:2rem 0;text-align:center;">
                        <h3 style="color:#1E3A5F;margin-bottom:1rem;">Compliance Score</h3>
                        <div style="font-size:4rem;font-weight:700;color:{score_color};">{score:.0f}%</div>
                        <div style="background:#E2E8F0;height:12px;border-radius:6px;margin:1rem 0;overflow:hidden;">
                            <div style="background:linear-gradient(90deg,{score_color},{score_color});width:{score}%;height:100%;border-radius:6px;"></div>
                        </div>
                        <p style="color:#64748B;">Based on {len(results.get('compliant', []))} compliant and {len(results.get('non_compliant', []))} non-compliant sections</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Comparison Table
                    st.markdown("""
                    <div style="background:white;padding:2rem;border-radius:16px;box-shadow:0 4px 20px rgba(30,58,95,0.1);margin:2rem 0;">
                        <h3 style="color:#1E3A5F;margin-bottom:1rem;">📊 Compliance Comparison Table</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create two columns for compliant vs non-compliant
                    col_comp, col_noncomp = st.columns(2)
                    
                    with col_comp:
                        st.markdown("""
                        <div style="background:#ECFDF5;padding:1rem;border-radius:12px;border-left:4px solid #22C55E;margin-bottom:1rem;">
                            <h4 style="color:#166534;margin:0;">✅ Compliant Sections</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        for item in results.get('compliant', [])[:10]:
                            category = item.get('category', 'Unknown')
                            evidence = item.get('evidence', 'Present in contract')[:150]
                            st.markdown(f"""
                            <div style="background:white;padding:1rem;border-radius:8px;margin:0.5rem 0;border:1px solid #D1FAE5;">
                                <strong style="color:#166534;">📋 {category}</strong>
                                <p style="font-size:0.85rem;color:#64748B;margin:0.5rem 0 0 0;font-style:italic;">"{evidence}..."</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col_noncomp:
                        st.markdown("""
                        <div style="background:#FEF2F2;padding:1rem;border-radius:12px;border-left:4px solid #EF4444;margin-bottom:1rem;">
                            <h4 style="color:#991B1B;margin:0;">❌ Non-Compliant Sections</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        for item in results.get('non_compliant', [])[:10]:
                            category = item.get('category', 'Unknown')
                            st.markdown(f"""
                            <div style="background:white;padding:1rem;border-radius:8px;margin:0.5rem 0;border:1px solid #FECACA;">
                                <strong style="color:#991B1B;">⚠️ {category}</strong>
                                <p style="font-size:0.85rem;color:#64748B;margin:0.5rem 0 0 0;">Missing or not specified in contract</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Download results
                    with st.expander("📥 Download Full Report"):
                        import json
                        report_json = json.dumps(results, indent=2)
                        st.download_button(
                            label="⬇️ Download JSON Report",
                            data=report_json,
                            file_name=f"compliance_report_{selected_contract[:30]}.json",
                            mime="application/json"
                        )
        
        # Compare multiple contracts
        if compare_btn:
            with st.spinner("📊 Comparing contracts..."):
                comparison = checker.compare_multiple_contracts(filtered_contracts[:5])
                if comparison:
                    # Build custom HTML table for top 5 contracts comparison
                    table_html = '''
                    <div style="background:white;padding:2rem;border-radius:16px;box-shadow:0 4px 20px rgba(30,58,95,0.1);margin:2rem 0;">
                        <h3 style="color:#1E3A5F;margin-bottom:1rem;">📊 Multi-Contract Comparison</h3>
                        <table class="top-contracts-table" style="width:100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden;">
                            <thead>
                                <tr style="background: #F1F5F9;">
                                    <th style="padding: 12px; text-align: left; color: #1E3A5F; font-weight: 600; border-bottom: 2px solid #E2E8F0;">Contract</th>
                                    <th style="padding: 12px; text-align: left; color: #1E3A5F; font-weight: 600; border-bottom: 2px solid #E2E8F0;">Score</th>
                                    <th style="padding: 12px; text-align: left; color: #1E3A5F; font-weight: 600; border-bottom: 2px solid #E2E8F0;">Compliant</th>
                                    <th style="padding: 12px; text-align: left; color: #1E3A5F; font-weight: 600; border-bottom: 2px solid #E2E8F0;">Non-Compliant</th>
                                </tr>
                            </thead>
                            <tbody>
                    '''
                    for contract_data in comparison.get("comparison", []):
                        contract = contract_data.get("contract", "")
                        score_val = contract_data.get("compliance_score", 0)
                        compliant = contract_data.get("clauses_found", 0)
                        non_compliant = contract_data.get("clauses_missing", 0)
                        table_html += f'''
                            <tr style="background: #FFFFFF;">
                                <td style="padding: 10px 12px; color: #152A45; border-bottom: 1px solid #E2E8F0;">{contract[:50]}...</td>
                                <td style="padding: 10px 12px; color: #152A45; border-bottom: 1px solid #E2E8F0;">{score_val:.0f}%</td>
                                <td style="padding: 10px 12px; color: #152A45; border-bottom: 1px solid #E2E8F0;">{compliant}</td>
                                <td style="padding: 10px 12px; color: #152A45; border-bottom: 1px solid #E2E8F0;">{non_compliant}</td>
                            </tr>
                        '''
                    table_html += '''
                            </tbody>
                        </table>
                    </div>
                    '''
                    import streamlit.components.v1 as components
                    components.html(table_html, height=400, scrolling=True)
    else:
        st.error("❌ Could not load contract database. Please check the dataset path.")

# AI Agent Section
st.markdown('''<div id="agent" style="padding:4rem 2rem;background:#fff;">
<div style="max-width:1000px;margin:0 auto;text-align:center;">
<p style="font-size:0.9rem;color:#D4AF37;font-weight:600;letter-spacing:2px;text-transform:uppercase;margin-bottom:0.5rem;">🤖 AI COMPLIANCE AGENT</p>
<h2 style="font-size:2.2rem;font-weight:700;color:#1E3A5F;margin-bottom:0.75rem;font-family:'Playfair Display',serif;">Multi-Step Reasoning</h2>
<p style="font-size:1rem;color:#64748B;margin-bottom:2rem;">Ask complex compliance questions and get AI-powered analysis with step-by-step reasoning</p>
</div>
</div>''', unsafe_allow_html=True)

# Agent Chat Interface
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#FFFBEB,#FEF3C7);padding:1.5rem;border-radius:12px;margin-bottom:1.5rem;border:1px solid #FDE68A;">
        <h4 style="color:#92400E;margin:0 0 0.5rem 0;">💡 Example Questions:</h4>
        <ul style="color:#78350F;margin:0;padding-left:1.5rem;font-size:0.9rem;">
            <li>What governing laws are specified in distribution agreements?</li>
            <li>Find contracts with non-compete clauses</li>
            <li>Which contracts have termination provisions?</li>
            <li>Compare liability limitations across license agreements</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize agent in session state
    if "agent" not in st.session_state:
        with st.spinner("🤖 Loading AI Agent..."):
            st.session_state.agent = load_agent()
    
    # Chat input
    st.markdown('''
    <style>
    /* Force the container to be white */
    .complaint-section-textbox [data-baseweb="textarea"] {
        background-color: white !important;
    }

    /* Force the actual text area to have visible text and cursor */
    .complaint-section-textbox textarea {
        background-color: white !important;
        color: #1E293B !important;
        -webkit-text-fill-color: #1E293B !important;
        caret-color: #1E3A5F !important; /* Ensure the blinking cursor is visible */
        opacity: 1 !important;
    }

    /* Target focus state to prevent it from turning dark when clicked */
    .complaint-section-textbox textarea:focus {
        background-color: white !important;
        color: #1E293B !important;
        -webkit-text-fill-color: #1E293B !important;
    }

    /* Ensure placeholder is visible but distinct */
    .complaint-section-textbox textarea::placeholder {
        color: #64748B !important;
        -webkit-text-fill-color: #64748B !important;
    }
    </style>
    ''', unsafe_allow_html=True)
    st.markdown('<div class="complaint-section-textbox">', unsafe_allow_html=True)
    agent_query = st.text_area(
        "Ask a compliance question:",
        placeholder="e.g., What governing laws are specified in the distribution agreements?",
        height=100,
        key="agent_query"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    run_agent = st.button("🚀 Run AI Analysis", use_container_width=True, type="primary")
    
    # Process agent query
    if run_agent and agent_query:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#F0F9FF,#E0F2FE);padding:1rem 1.5rem;border-radius:15px;margin:1rem 0;border-left:4px solid #1E3A5F;">
            <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem;">
                <span style="font-size:1.2rem;">👤</span>
                <span style="font-weight:600;color:#1E3A5F;font-size:0.85rem;">Your Question</span>
            </div>
            <p style="color:#1E293B;margin:0;font-size:1rem;">{agent_query}</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("🤖 AI Agent is analyzing..."):
            # Show progress
            progress_placeholder = st.empty()
            progress_placeholder.markdown("""
            <div style="background:#F1F5F9;padding:1rem;border-radius:8px;">
                <p style="color:#64748B;margin:0;">🔄 Planning actions...</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Run agent
            response = st.session_state.agent.run(agent_query)
            progress_placeholder.empty()
            
            # Display response
            st.markdown(f"""
            <div style="background:white;padding:1.5rem;border-radius:15px;margin:1rem 0;border:1px solid #E2E8F0;box-shadow:0 4px 15px rgba(30,58,95,0.08);">
                <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:1rem;">
                    <div style="background:linear-gradient(135deg,#1E3A5F,#152A45);padding:0.4rem;border-radius:8px;">
                        <span style="font-size:1rem;">⚖️</span>
                    </div>
                    <span style="font-weight:600;color:#1E3A5F;font-size:0.9rem;">LegalCheck AI Response</span>
                </div>
                <div style="color:#1E293B;line-height:1.7;font-size:0.95rem;">{response}</div>
            </div>
            """, unsafe_allow_html=True)

# Rules Reference Section
st.markdown('''<div style="padding:3rem 2rem;background:linear-gradient(180deg,#F8FAFC 0%,#F1F5F9 100%);">
<div style="max-width:1000px;margin:0 auto;">
<p style="font-size:0.9rem;color:#D4AF37;font-weight:600;letter-spacing:2px;text-transform:uppercase;margin-bottom:0.5rem;text-align:center;">📜 COMPLIANCE RULES</p>
<h2 style="font-size:2rem;font-weight:700;color:#1E3A5F;margin-bottom:2rem;text-align:center;font-family:'Playfair Display',serif;">25+ Policy Rules</h2>
</div>
</div>''', unsafe_allow_html=True)

# Display rules in expandable section
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    with st.expander("📋 View All Compliance Rules", expanded=False):
        if checker.rules:
            # Build HTML table with light theme
            table_html = '''
            <table class="top-contracts-table" style="width:100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden;">
                <thead>
                    <tr style="background: #F1F5F9;">
                        <th style="padding: 12px; text-align: left; color: #1E3A5F; font-weight: 600; border-bottom: 2px solid #E2E8F0;">Rule ID</th>
                        <th style="padding: 12px; text-align: left; color: #1E3A5F; font-weight: 600; border-bottom: 2px solid #E2E8F0;">Name</th>
                        <th style="padding: 12px; text-align: left; color: #1E3A5F; font-weight: 600; border-bottom: 2px solid #E2E8F0;">Category</th>
                        <th style="padding: 12px; text-align: left; color: #1E3A5F; font-weight: 600; border-bottom: 2px solid #E2E8F0;">Severity</th>
                    </tr>
                </thead>
                <tbody>
            '''
            for i, rule in enumerate(checker.rules):
                bg_color = "#FFFFFF" if i % 2 == 0 else "#F8FAFC"
                severity = rule.get('severity', 'Medium')
                severity_color = "#EF4444" if severity == "critical" else "#F59E0B" if severity == "high" else "#22C55E" if severity == "low" else "#64748B"
                table_html += f'''
                    <tr style="background: {bg_color};">
                        <td style="padding: 10px 12px; color: #1E293B; border-bottom: 1px solid #E2E8F0;">{rule.get('id', 'Unknown')}</td>
                        <td style="padding: 10px 12px; color: #1E293B; border-bottom: 1px solid #E2E8F0;">{rule.get('name', 'Unknown')}</td>
                        <td style="padding: 10px 12px; color: #1E293B; border-bottom: 1px solid #E2E8F0;">{rule.get('category', 'General')}</td>
                        <td style="padding: 10px 12px; border-bottom: 1px solid #E2E8F0;"><span style="background: {severity_color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; font-weight: 500;">{severity}</span></td>
                    </tr>
                '''
            table_html += '</tbody></table>'
            import streamlit.components.v1 as components
            components.html(table_html, height=400, scrolling=True)

# Footer
st.markdown('''<div id="about" style="background:linear-gradient(135deg,#1E3A5F 0%,#152A45 100%);padding:2rem 2rem;margin-top:3rem;">
<div style="max-width:1000px;margin:0 auto;text-align:center;">
<div style="display:flex;align-items:center;justify-content:center;gap:0.5rem;margin-bottom:1rem;">
<div style="background:rgba(212,175,55,0.2);padding:0.4rem;border-radius:8px;display:flex;align-items:center;justify-content:center;">
<svg width="20" height="20" viewBox="0 0 24 24" fill="none"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" stroke="#D4AF37" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
</div>
<span style="font-size:1.25rem;font-weight:700;color:white;font-family:'Playfair Display',serif;">LegalCheck</span>
</div>
<p style="color:rgba(255,255,255,0.9);font-size:0.95rem;margin-bottom:1rem;">AI-Powered Policy Compliance Intelligence Platform</p>
<div style="display:flex;gap:1.5rem;justify-content:center;margin-bottom:1rem;flex-wrap:wrap;">
<span style="color:rgba(255,255,255,0.7);font-size:0.85rem;">📜 510 Contracts</span>
<span style="color:rgba(255,255,255,0.7);font-size:0.85rem;">⚖️ 25+ Rules</span>
<span style="color:rgba(255,255,255,0.7);font-size:0.85rem;">🤖 AI-Powered</span>
<span style="color:rgba(255,255,255,0.7);font-size:0.85rem;">📊 CUAD Dataset</span>
</div>
<div style="margin-bottom:1rem;">
<a href="mailto:nadeemahmad2703@gmail.com" style="color:#D4AF37;text-decoration:none;font-size:0.9rem;display:inline-flex;align-items:center;gap:0.5rem;background:rgba(255,255,255,0.1);padding:0.5rem 1.25rem;border-radius:50px;">📧 nadeemahmad2703@gmail.com</a>
</div>
<div style="border-top:1px solid rgba(255,255,255,0.2);padding-top:1rem;">
<p style="color:rgba(255,255,255,0.6);font-size:0.75rem;margin:0;">© 2025 LegalCheck. Empowering Legal Compliance with Artificial Intelligence</p>
</div>
</div>
</div>''', unsafe_allow_html=True)

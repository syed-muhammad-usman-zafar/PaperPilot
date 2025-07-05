import streamlit as st
import os
import sys
from dotenv import load_dotenv
import pandas as pd
import json
import base64

# Add utils to sys.path for imports if running as a script
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from utils.llm_extraction_agent import extract_with_llm
from utils.research_agent import get_real_source_summaries
from utils.summarizer import generate_bullet_summaries
from utils.knowledge_graph import build_knowledge_graph, show_graph, get_papers_for_keyword, get_authors_for_paper, get_chain_prompt_to_draft
from utils.writing_agent import writing_agent
from utils.citation_agent import citation_agent
from utils.orchestrator import generate_full_paper
from utils.citation_agent import calculate_citation_plan

# Main Streamlit UI logic

def get_paperplane_icon_base64():
    with open("assets/paperplane.ico", "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

def main():
    # Use a custom favicon. Place your 'paperplane.ico' file in the 'assets' folder at the project root.
    st.set_page_config(
        page_title="PaperPilot",
        page_icon="assets/paperplane.ico",  
        layout="wide"
    )

    # Hero/header section
    icon_base64 = get_paperplane_icon_base64()
    st.markdown(
        f'''
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; margin-bottom: 2em;">
            <div style="display: flex; align-items: center; justify-content: center;">
                <h1 style="margin-bottom: 0.2em; margin-right: 0.3em;">PaperPilot</h1>
                <img src="data:image/x-icon;base64,{icon_base64}" alt="Paper Airplane" style="height: 2.2em; vertical-align: middle; margin-left: 0.2em;" />
            </div>
            <h3 style="margin-top: 0.2em; margin-bottom: 0.2em; font-weight: normal;">From Prompt to Paper — We Fly You There.</h3>
            <div style="font-size: 1.1em; color: gray; margin-top: 0.5em;">
                A Neuro-Symbolic Agentic Framework for Research Paper Co-Authoring
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )

    # Main app background wrapper
    st.markdown("<div class='main-app-bg'>", unsafe_allow_html=True)
    tabs = st.tabs(["Client View", "Backend View"])

    # Initialize session state
    if 'client_prompt' not in st.session_state:
        st.session_state.client_prompt = ''
    if 'client_output' not in st.session_state:
        st.session_state.client_output = ''
    if 'client_warning' not in st.session_state:
        st.session_state.client_warning = ''
    if 'backend' not in st.session_state:
        st.session_state.backend = None
    if 'full_paper' not in st.session_state:
        st.session_state.full_paper = None
    if 'citation_plan' not in st.session_state:
        st.session_state.citation_plan = None

    with tabs[0]:
        st.header("Hi there! What would you like help drafting today?")
        prompt = st.text_input("Enter your research prompt:", value=st.session_state.client_prompt, key="client_input")
        
        submit = st.button("📝 Generate Full Paper", key="client_submit")
        
        if submit and prompt.strip():
            with st.spinner("Processing your request. Please wait..."):
                st.session_state.client_prompt = prompt
                # LLM extraction
                llm_extracted = extract_with_llm(prompt)
                # Fallbacks for missing fields
                domain = llm_extracted.get('domain', '')
                keywords = llm_extracted.get('key concepts') or llm_extracted.get('key_concepts', [])
                # --- Robust extraction for method/objective ---
                def safe_extract_first(lst, default):
                    if isinstance(lst, list) and lst:
                        val = lst[0]
                        if isinstance(val, str) and len(val.strip()) > 2:
                            return val.strip()
                    return default
                method = safe_extract_first(llm_extracted.get('research methods', []), 'analysis')
                objective = safe_extract_first(llm_extracted.get('objectives', []), 'investigate')
                data_types = llm_extracted.get('data types', [])
                # --- FIX: Ensure keywords is always a list of strings ---
                if isinstance(keywords, str):
                    # Try splitting by comma, then by whitespace if needed
                    if ',' in keywords:
                        keywords = [k.strip() for k in keywords.split(',') if k.strip()]
                    else:
                        keywords = [k.strip() for k in keywords.split() if k.strip()]
                elif not isinstance(keywords, list):
                    keywords = list(keywords)
                # Debug output
                print(f"[DEBUG] Final keywords for search: {keywords} (type: {type(keywords)})")
                
                # Calculate citation plan for full paper
                citation_plan = calculate_citation_plan(keywords, domain)
                total_papers_needed = sum(citation_plan.values())
                st.session_state.citation_plan = citation_plan
                
                # Get real sources with dynamic max_results
                max_results = total_papers_needed
                
                summaries = get_real_source_summaries(keywords, max_results=max_results)
                if not summaries:
                    st.session_state.client_warning = 'No real sources found from Semantic Scholar. Please try different keywords or check your internet connection.'
                    summaries = [{
                        'title': 'Placeholder Paper',
                        'summary': 'No real literature found for these keywords.',
                        'citation': '[1] Placeholder Author, "Placeholder Paper", Placeholder Journal, 2024',
                        'ref': '[1] Placeholder Author, "Placeholder Paper", Placeholder Journal, 2024',
                        'author_names': 'Unknown Author'
                    }]
                else:
                    st.session_state.client_warning = ''
                
                # Generate full paper
                full_paper = generate_full_paper(prompt, llm_extracted, summaries)
                st.session_state.full_paper = full_paper
                
                st.session_state.backend = {
                    'prompt': prompt,
                    'llm_extracted': llm_extracted,
                    'final_domain': domain,
                    'final_keywords': keywords,
                    'final_method': method,
                    'final_objective': objective,
                    'summaries': summaries,
                    'citation_plan': citation_plan,
                    'total_papers_needed': total_papers_needed
                }
        
        # Display results based on what was generated
        if st.session_state.client_warning:
            st.warning(st.session_state.client_warning)
        
        # Show citation plan info if available
        if st.session_state.citation_plan:
            with st.expander("📊 Citation Plan", expanded=False):
                st.write("**Papers needed per section:**")
                for section, count in st.session_state.citation_plan.items():
                    st.write(f"- {section}: {count} papers")
                st.write(f"**Total papers needed:** {sum(st.session_state.citation_plan.values())}")
                summaries = st.session_state.backend.get('summaries', []) if st.session_state.backend else []
                st.write(f"**Papers found:** {len(summaries)}")
        

        
        # Display full paper output
        if st.session_state.full_paper:
            st.markdown('---')
            st.subheader("📝 Full Academic Paper")
            
            # Paper title
            st.markdown(f"## {st.session_state.full_paper['title']}")
            
            # Display each section in expandable containers
            for section_name, paragraphs in st.session_state.full_paper['sections'].items():
                with st.expander(f"📄 {section_name}", expanded=(section_name in ["Abstract", "Introduction"])):
                    for i, paragraph in enumerate(paragraphs):
                        st.markdown(paragraph)
                        if i < len(paragraphs) - 1:  # Add spacing between paragraphs
                            st.markdown("")
            
            # Full paper view
            with st.expander("📋 Complete Paper View", expanded=False):
                full_text = f"# {st.session_state.full_paper['title']}\n\n"
                for section_name, paragraphs in st.session_state.full_paper['sections'].items():
                    full_text += f"## {section_name}\n\n"
                    for paragraph in paragraphs:
                        full_text += f"{paragraph}\n\n"
                st.markdown(full_text)

    with tabs[1]:
        st.header("Backend View (Debugging)")
        backend = st.session_state.get('backend', None)
        if backend:
            st.subheader("Raw User Prompt")
            st.code(backend['prompt'])
            st.subheader("Extracted Research Elements (LLM)")
            st.json(backend['llm_extracted'])
            st.subheader("Final Parameters Used")
            st.write(f"**Domain:** {backend.get('final_domain', '')}")
            st.write(f"**Keywords:** {backend.get('final_keywords', '')}")
            st.write(f"**Method:** {backend.get('final_method', '')}")
            st.write(f"**Objective:** {backend.get('final_objective', '')}")
            st.subheader("Bullet-point Literature Summaries")
            for bullet in backend.get('bullets', []):
                st.markdown(bullet)
            st.subheader("Raw Citation Strings from Semantic Scholar")
            for s in backend.get('summaries', []):
                st.markdown(f"- {s['citation']}")
            
            # Show citation plan if available
            if backend.get('citation_plan'):
                st.subheader("Citation Plan")
                st.json(backend['citation_plan'])
                st.write(f"**Total papers needed:** {backend.get('total_papers_needed', 0)}")
            
            # Show quick draft results if available
            if backend.get('draft_paragraph'):
                st.subheader("Uncited Draft Paragraph (LLM)")
                st.markdown(backend.get('draft_paragraph', ''))
                st.subheader("Cited Draft Paragraph")
                st.markdown(backend.get('cited_paragraph', ''))
            
            # Show full paper results if available
            if st.session_state.full_paper:
                st.subheader("Full Paper Structure")
                st.json({
                    'title': st.session_state.full_paper['title'],
                    'sections': list(st.session_state.full_paper['sections'].keys()),
                    'papers_found': st.session_state.full_paper['papers_found'],
                    'total_papers_needed': st.session_state.full_paper['total_papers_needed']
                })
                
                st.subheader("Section Assignments")
                for section, papers in st.session_state.full_paper['section_assignments'].items():
                    st.write(f"**{section}:** {len(papers)} papers")
                    for paper in papers:
                        st.markdown(f"- {paper['title']}")
            st.subheader("Knowledge Graph Output (JSON)")
            try:
                G = build_knowledge_graph(
                    backend.get('final_domain', ''),
                    backend.get('final_keywords', []),
                    backend.get('final_method', ''),
                    backend.get('final_objective', ''),
                    backend.get('summaries', []),
                    backend.get('draft_paragraph', '')
                )
                import networkx as nx
                st.json(nx.node_link_data(G))
                st.subheader("Knowledge Graph Visualization")
                show_graph(G)
                st.subheader("Knowledge Graph Reasoning Examples")
                if backend.get('final_keywords', []):
                    kw = backend['final_keywords'][0]
                    papers = get_papers_for_keyword(G, kw)
                    st.write(f"Papers related to keyword '{kw}': {papers}")
                paper_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'paper']
                if paper_nodes:
                    authors = get_authors_for_paper(G, paper_nodes[0])
                    st.write(f"Authors for paper '{paper_nodes[0]}': {authors}")
                chain = get_chain_prompt_to_draft(G)
                st.write(f"Chain from Prompt to DraftParagraph: {chain}")
            except Exception as e:
                st.warning(f"Could not build or display knowledge graph: {e}") 
    st.markdown("</div>", unsafe_allow_html=True) 
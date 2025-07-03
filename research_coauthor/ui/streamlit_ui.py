import streamlit as st
import os
import sys
from dotenv import load_dotenv
import pandas as pd
import json

# Add utils to sys.path for imports if running as a script
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from utils.llm_extraction_agent import extract_with_llm
from utils.literature_retrieval_agent import get_real_source_summaries
from utils.summarizer import generate_bullet_summaries
from utils.knowledge_graph import build_knowledge_graph, show_graph, get_papers_for_keyword, get_authors_for_paper, get_chain_prompt_to_draft
from utils.writing_agent import writing_agent

# Main Streamlit UI logic

def main():
    st.set_page_config(
        page_title="AI Powered Research Paper Co-Author",
        page_icon="🔬",
        layout="wide"
    )
    st.title("AI Powered Research Paper Co-Author")
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

    with tabs[0]:
        st.header("Hi there! What would you like help drafting today?")
        prompt = st.text_input("Enter your research prompt:", value=st.session_state.client_prompt, key="client_input")
        submit = st.button("Submit", key="client_submit")
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
                # Get real sources
                summaries = get_real_source_summaries(keywords, max_results=2)
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
                # Generate content
                bullet_points = generate_bullet_summaries(summaries)
                context = f"Domain: {domain}\nMethods: {method}\nObjectives: {objective}\nData Types: {', '.join(data_types)}\nKey Concepts: {', '.join(keywords)}"
                # Build the knowledge graph before drafting
                G = build_knowledge_graph(domain, keywords, method, objective, summaries, "")
                # Extract paper nodes (with metadata) from the graph
                paper_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'paper']
                papers = []
                for n in paper_nodes:
                    d = G.nodes[n]
                    # Find the corresponding summary for this paper node
                    for s in summaries:
                        if s.get('title', '') == d.get('title', ''):
                            papers.append(s)
                            break
                # Extract keyword nodes from the graph
                keyword_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'keyword']
                # Check if all papers are placeholders
                all_placeholders = all(
                    p.get('author_names', '') == 'Unknown Author' or 'No relevant papers found' in p.get('title', '')
                    for p in papers
                )
                if all_placeholders:
                    papers = []
                    keyword_nodes = []
                    context += "\nNote: No real literature is available for citation. Write an exploratory paragraph without citing external sources."
                draft_paragraph = writing_agent(context, papers, keyword_nodes)
                st.session_state.client_output = draft_paragraph
                st.session_state.backend = {
                    'prompt': prompt,
                    'llm_extracted': llm_extracted,
                    'final_domain': domain,
                    'final_keywords': keywords,
                    'final_method': method,
                    'final_objective': objective,
                    'summaries': summaries,
                    'bullets': bullet_points,
                    'draft_paragraph': draft_paragraph,
                    'cited_paragraph': draft_paragraph
                }
        if st.session_state.client_warning:
            st.warning(st.session_state.client_warning)
        if st.session_state.client_output:
            st.markdown('---')
            st.markdown(st.session_state.client_output)

    with tabs[1]:
        st.header("Backend View (Debugging / Research Agent)")
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
            st.subheader("Uncited Draft Paragraph (LLM)")
            st.markdown(backend.get('draft_paragraph', ''))
            st.subheader("Cited Draft Paragraph")
            st.markdown(backend.get('cited_paragraph', ''))
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
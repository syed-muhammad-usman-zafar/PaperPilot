import streamlit as st
import base64

# Add utils to sys.path for imports if running as a script
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils')) # This line is removed as per the new_code

from utils.llm_extraction_agent import extract_with_llm
from utils.research_agent import get_real_source_summaries
from utils.knowledge_graph import build_knowledge_graph, show_graph, get_papers_for_keyword, get_authors_for_paper, get_chain_prompt_to_draft
from utils.citation_agent import citation_agent
from utils.orchestrator import generate_full_paper
from utils.citation_agent import calculate_citation_plan

# PDF extraction helper
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

# Main Streamlit UI logic

def get_paperplane_icon_base64():
    with open("assets/paperplane.ico", "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        if not pdfplumber:
            st.error("pdfplumber is required for PDF extraction. Please install it.")
            return ""
        with pdfplumber.open(uploaded_file) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    elif uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8", errors="ignore")
    else:
        st.error("Unsupported file type. Please upload a PDF or TXT file.")
        return ""

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

    st.markdown("<div class='main-app-bg'>", unsafe_allow_html=True)
    tabs = st.tabs(["Client View", "View Thought Process"])

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
    if 'user_research_metadata' not in st.session_state:
        st.session_state.user_research_metadata = None
    if 'user_research_text' not in st.session_state:
        st.session_state.user_research_text = None
    if 'user_research_added' not in st.session_state:
        st.session_state.user_research_added = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []


    with tabs[0]:
        st.header("Hi there! What would you like help drafting today?")
        # --- User Research Upload Section ---
        st.subheader("Upload Your Own Research (PDF or TXT)")
        uploaded_file = st.file_uploader("Upload a PDF or TXT file with your research/notes:", type=["pdf", "txt"])
        if uploaded_file:
            user_text = extract_text_from_file(uploaded_file)
            st.session_state.user_research_text = user_text
            if user_text.strip():
                st.markdown("**Extracted Text Preview:**")
                st.text_area("User Research Text", value=user_text[:2000], height=200, disabled=True)
                if st.button("Extract Metadata from My Research", key="extract_user_metadata"):
                    with st.spinner("Extracting metadata from your research (minimal tokens)..."):
                        metadata = extract_with_llm(user_text[:3000])  # Only use first 3000 chars for LLM
                        # Add user attribution
                        metadata['source'] = 'user_research'
                        metadata['reference'] = 'my findings'
                        st.session_state.user_research_metadata = metadata
                        st.session_state.user_research_added = False
        # --- Metadata Review/Edit Form ---
        if st.session_state.user_research_metadata and not st.session_state.user_research_added:
            st.markdown("**Review/Edit Extracted Metadata:**")
            meta = st.session_state.user_research_metadata
            with st.form("user_metadata_form"):
                domain = st.text_input("Domain", value=meta.get('domain', ''))
                methods = st.text_input("Research Methods", value=','.join(meta.get('research methods', [])))
                objectives = st.text_input("Objectives", value=','.join(meta.get('objectives', [])))
                data_types = st.text_input("Data Types", value=','.join(meta.get('data types', [])))
                key_concepts = st.text_input("Key Concepts", value=','.join(meta.get('key concepts', [])))
                method_type = st.text_input("Method Type", value=meta.get('method_type', ''))
                objective_scope = st.text_input("Objective Scope", value=meta.get('objective_scope', ''))
                submitted = st.form_submit_button("Add My Research to Knowledge Graph")
                if submitted:
                    # Build a dict for the user research node
                    user_node = {
                        'title': 'My Research',
                        'author_names': 'You',
                        'summary': st.session_state.user_research_text[:500],
                        'domain': domain,
                        'research methods': [m.strip() for m in methods.split(',') if m.strip()],
                        'objectives': [o.strip() for o in objectives.split(',') if o.strip()],
                        'data types': [d.strip() for d in data_types.split(',') if d.strip()],
                        'key concepts': [k.strip() for k in key_concepts.split(',') if k.strip()],
                        'method_type': method_type,
                        'objective_scope': objective_scope,
                        'source': 'user_research',
                        'reference': 'my findings',
                        'year': '2024',
                        'venue': 'User Submission'
                    }
                    # Add to KG (as a paper node)
                    if 'user_papers' not in st.session_state:
                        st.session_state.user_papers = []
                    st.session_state.user_papers.append(user_node)
                    st.session_state.user_research_added = True
                    st.success("Your research has been added to the knowledge graph and will be referenced as 'my findings'.")

        # --- Main Prompt Input ---
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
                def safe_extract_first(lst, default):
                    if isinstance(lst, list) and lst:
                        val = lst[0]
                        if isinstance(val, str) and len(val.strip()) > 2:
                            return val.strip()
                    return default
                method = safe_extract_first(llm_extracted.get('research methods', []), 'analysis')
                objective = safe_extract_first(llm_extracted.get('objectives', []), 'investigate')
                data_types = llm_extracted.get('data types', [])
                if isinstance(keywords, str):
                    if ',' in keywords:
                        keywords = [k.strip() for k in keywords.split(',') if k.strip()]
                    else:
                        keywords = [k.strip() for k in keywords.split() if k.strip()]
                elif not isinstance(keywords, list):
                    keywords = list(keywords)
                # Calculate citation plan for full paper
                method_type = llm_extracted.get('method_type', method)
                objective_scope = llm_extracted.get('objective_scope', objective)
                citation_plan = calculate_citation_plan(keywords, method_type, objective_scope)
                total_papers_needed = sum(citation_plan.values())
                st.session_state.citation_plan = citation_plan
                max_results = total_papers_needed
                summaries = get_real_source_summaries(keywords, max_results=max_results)
                # --- Prepare user research as context, not as a citable source ---
                user_research_context = None
                if 'user_papers' in st.session_state and st.session_state.user_papers:
                    # Use the first user paper as context (or merge if multiple)
                    user_research_context = st.session_state.user_papers[0]
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
                # Generate full paper, passing user research context separately
                full_paper = generate_full_paper(prompt, llm_extracted, summaries, user_research_context=user_research_context)
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
                    'total_papers_needed': total_papers_needed,
                    'user_research_context': user_research_context
                }

        # Display results based on what was generated
        if st.session_state.client_warning:
            st.warning(st.session_state.client_warning)
        if st.session_state.full_paper:
            st.markdown('---')
            st.subheader("📝 Full Academic Paper")
            st.markdown(f"## {st.session_state.full_paper['title']}")
            for section_name, paragraphs in st.session_state.full_paper['sections'].items():
                with st.expander(f"📄 {section_name}", expanded=(section_name in ["Abstract", "Introduction"])):
                    for i, paragraph in enumerate(paragraphs):
                        processed_paragraph = paragraph
                        if 'citation_map' in st.session_state.full_paper:
                            citation_map = st.session_state.full_paper['citation_map']
                            for paper in st.session_state.full_paper['section_assignments'].get(section_name, []):
                                key = (paper['title'], paper['author_names'])
                                citation_num = citation_map.get(key)
                                if citation_num:
                                    import re
                                    processed_paragraph = re.sub(rf"Paper_{citation_num}", f"[{citation_num}]", processed_paragraph)
                                    processed_paragraph = re.sub(rf"Paper {citation_num}", f"[{citation_num}]", processed_paragraph)
                        st.text(processed_paragraph)
                        if i < len(paragraphs) - 1:
                            st.markdown("")
            if 'references' in st.session_state.full_paper:
                with st.expander("📄 References", expanded=False):
                    refs_md = st.session_state.full_paper['references'].replace('\n', '\n\n')
                    st.markdown(refs_md)
            # In the full paper view, assemble the paper in the correct order as plain text:
            section_order = [
                "Abstract", "Introduction", "Literature Review", "Methodology", "Experiments / Results", "Conclusion"
            ]
            full_text = ""
            for section in section_order:
                full_text += f"{section}\n\n"
                paragraphs = st.session_state.full_paper['sections'].get(section, ["This section was not generated by the LLM."])
                for paragraph in paragraphs:
                    full_text += paragraph + "\n\n"
            st.text_area("Full Paper Text", value=full_text.strip(), height=600)

            # --- Follow-up Question Feature ---
            st.markdown('---')
            st.subheader("💬 Ask a Follow-up Question About Your Paper or Research")
            followup = st.text_input("Type your follow-up question here...", key="followup_input")
            if st.button("Ask", key="followup_send") and followup.strip():
                # Prepare context for LLM: use generated paper, user research, and metadata
                context_chunks = []
                if st.session_state.full_paper:
                    import re
                    section_match = re.search(r'(abstract|introduction|literature review|methodology|results|conclusion)', followup, re.I)
                    if section_match:
                        section = section_match.group(1).title()
                        section_text = '\n'.join(st.session_state.full_paper['sections'].get(section, []))
                        if section_text:
                            context_chunks.append(f"{section} Section:\n{section_text}")
                    else:
                        context_chunks.append(f"Title: {st.session_state.full_paper['title']}")
                        for sec in ['Abstract', 'Introduction']:
                            sec_text = '\n'.join(st.session_state.full_paper['sections'].get(sec, []))
                            if sec_text:
                                context_chunks.append(f"{sec}: {sec_text}")
                if st.session_state.user_research_text:
                    context_chunks.append(f"User Uploaded Research: {st.session_state.user_research_text[:1000]}")
                if st.session_state.user_research_metadata:
                    context_chunks.append(f"User Research Metadata: {st.session_state.user_research_metadata}")
                chat_context = '\n'.join(context_chunks)
                chat_prompt = f"You are an expert research assistant. Given the following context, answer the user's question as clearly and concisely as possible.\nContext:\n{chat_context}\n\nUser question: {followup}"
                from utils.llm_extraction_agent import model
                try:
                    response = model.generate_content(chat_prompt, generation_config={"max_output_tokens": 100})
                    answer = response.text.strip() if response.text else "[No answer generated.]"
                except Exception as e:
                    answer = f"[Error: {e}]"
                st.markdown(f"**PaperPilot:** {answer}")

    with tabs[1]:
        st.header("See how PaperPilot works behind the scenes")
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
                # Use the knowledge graph from the full paper result (neuro-symbolic integration)
                if st.session_state.full_paper and 'knowledge_graph' in st.session_state.full_paper:
                    G = st.session_state.full_paper['knowledge_graph']
                    st.success("✅ Using neuro-symbolic knowledge graph from paper generation!")
                else:
                    # Fallback to building knowledge graph for display
                    G = build_knowledge_graph(
                        backend.get('final_domain', ''),
                        backend.get('final_keywords', []),
                        backend.get('final_method', ''),
                        backend.get('final_objective', ''),
                        backend.get('summaries', []),
                        backend.get('draft_paragraph', '')
                    )
                    st.info("ℹ️ Using fallback knowledge graph for display only")
                
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
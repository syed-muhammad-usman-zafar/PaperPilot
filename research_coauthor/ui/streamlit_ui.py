import streamlit as st
import base64
from utils.llm_extraction_agent import extract_with_llm
from utils.research_agent import get_real_source_summaries
from utils.knowledge_graph import build_knowledge_graph, show_graph
from utils.citation_agent import calculate_citation_plan
from utils.orchestrator import generate_full_paper
from utils.docx_export import create_paper_docx

try:
    import pdfplumber
except ImportError:
    pdfplumber = None
def get_paperplane_icon_base64():
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    assets_path = os.path.join(current_dir, "..", "..", "assets", "paperplane.ico")
    try:
        with open(assets_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        fallback_path = os.path.join("..", "assets", "paperplane.ico")
        with open(fallback_path, "rb") as image_file:
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
    st.set_page_config(
        page_title="PaperPilot",
        page_icon="../assets/paperplane.ico",
        layout="wide"
    )
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
    if 'chat_cache' not in st.session_state:
        st.session_state.chat_cache = {}

    with tabs[0]:
        st.header("Hi there! What would you like help drafting today?")
        if not st.session_state.get('user_research_added', False):
            st.subheader("Upload Your Own Research (PDF or TXT)")
            uploaded_file = st.file_uploader(
                "Upload a PDF or TXT file with your research/notes:",
                type=["pdf", "txt"],
                key="research_file_uploader"
            )
            if uploaded_file:
                user_text = extract_text_from_file(uploaded_file)
                st.session_state.user_research_text = user_text

                if st.button("Add to Paper"):
                    lines = user_text.splitlines()
                    title = lines[0].strip() if lines and len(lines[0].strip()) > 3 else "User Uploaded Research"
                    summary = " ".join(lines[1:]).strip() if len(lines) > 1 else user_text.strip()
                    user_paper = {
                        'title': title,
                        'summary': summary,
                        'citation': '[User Uploaded]',
                        'ref': '[User Uploaded]',
                        'author_names': 'User',
                        'source': 'user_upload'
                    }
                    if 'user_papers' not in st.session_state:
                        st.session_state.user_papers = []
                    st.session_state.user_papers = [user_paper]
                    st.session_state.user_research_added = True
                    st.success("Research added! Continue with your prompt below.")

        prompt = st.text_input("Enter your research prompt:", value=st.session_state.client_prompt, key="client_input")
        submit = st.button("📝 Generate Full Paper", key="client_submit")
        
        if submit and prompt.strip():
            with st.spinner("Processing your request. Please wait..."):
                st.session_state.client_prompt = prompt
                
                llm_extracted = extract_with_llm(prompt)
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
                
                if isinstance(keywords, str):
                    if ',' in keywords:
                        keywords = [k.strip() for k in keywords.split(',') if k.strip()]
                    else:
                        keywords = [k.strip() for k in keywords.split() if k.strip()]
                elif not isinstance(keywords, list):
                    keywords = list(keywords)
                method_type = llm_extracted.get('method_type', method)
                objective_scope = llm_extracted.get('objective_scope', objective)
                citation_plan = calculate_citation_plan(keywords, method_type, objective_scope)
                total_papers_needed = sum(citation_plan.values())
                st.session_state.citation_plan = citation_plan
                max_results = total_papers_needed
                summaries = get_real_source_summaries(keywords, max_results=max_results)
                
                user_research_context = None
                if 'user_papers' in st.session_state and st.session_state.user_papers:
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
            
                full_paper = generate_full_paper(
                    prompt,
                    llm_extracted,
                    summaries,
                    user_research_context=user_research_context
                )
                
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

        
        if st.session_state.get('client_warning'):
            st.warning(st.session_state.client_warning)
            
        if st.session_state.get('full_paper'):
            st.markdown('---')
            st.subheader("📝 Full Academic Paper")
            st.markdown(f"## {st.session_state.full_paper['title']}")
            
            # Show the complete paper as one continuous document
            raw_output = st.session_state.full_paper.get('raw_output', 'No content available')
            
            if raw_output and raw_output != 'No content available' and raw_output != '[Error generating paper]':
                # Display the entire paper content at once
                st.markdown("### Complete Research Paper")
                
                # Clean up the raw output and remove references section if it exists
                # (we'll add it separately to avoid duplicates)
                import re
                cleaned_output = raw_output.replace('**', '**').strip()
                
                # Remove any references section from the raw output to avoid duplicates
                cleaned_output = re.sub(r'\*\*REFERENCES?\*\*.*', '', cleaned_output, flags=re.DOTALL | re.IGNORECASE)
                cleaned_output = re.sub(r'REFERENCES?\s*\n.*', '', cleaned_output, flags=re.DOTALL | re.IGNORECASE)
                cleaned_output = cleaned_output.strip()
                
                # Split into paragraphs for better readability
                paragraphs = [p.strip() for p in cleaned_output.split('\n\n') if p.strip()]
                
                for paragraph in paragraphs:
                    if paragraph.strip():
                        # Handle section headers differently
                        if paragraph.startswith('**') and paragraph.endswith('**'):
                            st.markdown(f"### {paragraph.replace('**', '')}")
                        else:
                            st.write(paragraph)
                        st.markdown("")  # Add space between paragraphs
                
                # Add references section directly to the main paper content (only once)
                if 'references' in st.session_state.full_paper:
                    st.markdown("### References")
                    refs_content = st.session_state.full_paper['references']
                    if refs_content:
                        # Clean up references formatting
                        if refs_content.startswith('References\n'):
                            refs_content = refs_content.replace('References\n', '', 1)
                        
                        # Split references into individual items and display them properly
                        ref_lines = [ref.strip() for ref in refs_content.split('\n') if ref.strip()]
                        for ref in ref_lines:
                            if ref.strip() and not ref.startswith('[Add ') and 'more relevant references' not in ref:
                                st.write(ref)
                                st.markdown("")
            
            else:
                st.error("❌ No paper content was generated. Please try again.")
                
                # Show debug info
                if st.checkbox("🔍 Show Debug Info"):
                    st.json({
                        "raw_output": raw_output,
                        "sections": st.session_state.full_paper.get('sections', {}),
                        "title": st.session_state.full_paper.get('title', 'No title')
                    })
            
            st.markdown('---')
            st.subheader("📥 Download Paper")
            if st.button("📄 Download as DOCX", key="download_docx"):
                try:
                    print(f"[DEBUG] DOCX Download - paper_data keys: {list(st.session_state.full_paper.keys())}")
                    if 'references' in st.session_state.full_paper:
                        print(f"[DEBUG] DOCX Download - references length: {len(st.session_state.full_paper['references'])}")
                        print(f"[DEBUG] DOCX Download - references preview: {st.session_state.full_paper['references'][:200]}...")
                    else:
                        print("[DEBUG] DOCX Download - No 'references' key found in paper_data")
                   
                    docx_bytes = create_paper_docx(
                        st.session_state.full_paper,
                        st.session_state.client_prompt
                    )
                    st.download_button(
                        label="💾 Click to Download DOCX",
                        data=docx_bytes,
                        file_name=f"PaperPilot_{st.session_state.full_paper['title'][:30].replace(' ', '_')}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
                    st.success("✅ DOCX file ready for download! You can add images manually in Word.")
                except Exception as e:
                    st.error(f"❌ Error creating DOCX: {e}")

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
            
            st.subheader("Raw Citation Strings from Semantic Scholar")
            for s in backend.get('summaries', []):
                st.markdown(f"- {s['citation']}")
            
       
            if backend.get('citation_plan'):
                st.subheader("Citation Plan")
                st.json(backend['citation_plan'])
                st.write(f"**Total papers needed:** {backend.get('total_papers_needed', 0)}")
            
            
            if st.session_state.full_paper:
                st.subheader("Full Paper Structure")
                st.json({
                    'title': st.session_state.full_paper['title'],
                    'sections': list(st.session_state.full_paper['sections'].keys()),
                    'papers_found': st.session_state.full_paper.get('papers_found', 0),
                    'total_papers_needed': st.session_state.full_paper.get('total_papers_needed', 0)
                })
            
            st.subheader("Knowledge Graph Output (JSON)")
            try:
                
                if st.session_state.full_paper and 'knowledge_graph' in st.session_state.full_paper:
                    G = st.session_state.full_paper['knowledge_graph']
                    st.success("✅ Using neuro-symbolic knowledge graph from paper generation!")
                else:
               
                    G = build_knowledge_graph(
                        backend.get('final_domain', ''),
                        backend.get('final_keywords', []),
                        backend.get('final_method', ''),
                        backend.get('final_objective', ''),
                        backend.get('summaries', []),
                        backend.get('draft_paragraph', '')
                    )
                    st.info("ℹ️ Using fallback knowledge graph for display only")
                
                try:
                    import networkx as nx
                    st.json(nx.node_link_data(G))
                    st.subheader("Knowledge Graph Visualization")
                    show_graph(G)
                except ImportError:
                    st.warning("NetworkX not available. Knowledge graph visualization disabled.")
                
            except Exception as e:
                st.warning(f"Could not build or display knowledge graph: {e}")
    
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

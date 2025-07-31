import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import base64
from utils.llm_extraction_agent import extract_with_llm
from utils.research_agent import get_real_source_summaries
from utils.knowledge_graph import build_knowledge_graph, show_graph
from utils.citation_agent import calculate_citation_plan
from utils.orchestrator import generate_full_paper
from utils.docx_export import create_paper_docx
from utils.validation_agent import validate_llm_extraction, validate_real_source_summaries, paper_score, rate_paper, val_score
from utils.chat_agent import PaperChatAgent


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
        page_icon="../assets/paper-plane.png",
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
                
                try:
                    vals = validate_llm_extraction(llm_extracted, prompt)
                except ValueError as e:
                    print(f"Extraction validation error: {e}")
                    return
                else:
                    print("\n\n After validation:\n")
                    domain, keywords, method, objective, validation = vals
                    llm_extracted = {
                        'domain': domain,
                        'key_concepts': keywords,
                        'methods': method,
                        'objectives': objective,
                        'validation_requirements': validation
                    }
                    print(vals)

                citation_plan = calculate_citation_plan(keywords, method, objective)
                total_papers_needed = sum(citation_plan.values())
                st.session_state.citation_plan = citation_plan
                max_results = total_papers_needed
                summaries = get_real_source_summaries(keywords, 50)
                
                st.session_state.validation = validation

                try:
                    vals = validate_real_source_summaries(prompt,max_results,summaries)
                except ValueError as e:
                    print(f"Extraction validation error: {e}")
                else:
                    summaries = vals
                    print("\n\nFinal Summaries:\n")
                    print(len(summaries))


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
                    domain, keywords, method, objective, validation,
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
          
            val_norm = val_score(
                st.session_state.validation,
                st.session_state.full_paper
            )
            
            report = rate_paper(
                final_paper=st.session_state.full_paper['raw_output'],
                prompt=st.session_state.client_prompt,
                context=st.session_state.user_research_text if st.session_state.user_research_text else "",
                val_norm=val_norm
            )

            score_value = report.pop("Score")
            
            tooltip_text = """**How We Grade Your Papers:**

**Quality Assessment (10-Point Scale)**

• **Content Relevance** - How well the paper addresses your research prompt and incorporates relevant literature
• **Writing Clarity** - Professional academic writing style, clear explanations, and logical flow
• **Structure & Organization** - Proper academic format with introduction, methodology, results, and conclusion
• **Research Depth** - Comprehensive coverage of the topic with appropriate citations and analysis

**Scoring Guide:**
• **9-10**: Exceptional quality, publication-ready
• **7-8**: High quality, minor revisions needed  
• **5-6**: Good foundation, moderate improvements required
• **3-4**: Needs significant enhancement
• **1-2**: Major revision required

Your paper is automatically evaluated using advanced AI to ensure consistent, objective grading across all submissions."""

        
            st.subheader("Score")
            st.metric(
                label="",
                value=f"{score_value}/10",
                help=tooltip_text
            )

            
            raw_output = st.session_state.full_paper.get('raw_output', 'No content available')
            
            if raw_output and raw_output != 'No content available' and raw_output != '[Error generating paper]':
                st.markdown("### Complete Research Paper")
                
                import re
                cleaned_output = raw_output.replace('**', '**').strip()

             
                cleaned_output = re.sub(r'\*\*REFERENCES?\*\*.*', '', cleaned_output, flags=re.DOTALL | re.IGNORECASE)
                cleaned_output = re.sub(r'REFERENCES?\s*\n.*', '', cleaned_output, flags=re.DOTALL | re.IGNORECASE)
                cleaned_output = cleaned_output.strip()
                
              
                paragraphs = [p.strip() for p in cleaned_output.split('\n\n') if p.strip()]
                
                for paragraph in paragraphs:
                    if paragraph.strip():
                       
                        if paragraph.startswith('**') and paragraph.endswith('**'):
                            st.markdown(f"### {paragraph.replace('**', '')}")
                        else:
                            st.write(paragraph)
                        st.markdown("")
                
            
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
                
                if st.checkbox("🔍 Show Debug Info"):
                    st.json({
                        "raw_output": raw_output,
                        "sections": st.session_state.full_paper.get('sections', {}),
                        "title": st.session_state.full_paper.get('title', 'No title')
                    })
            
          
            st.markdown('---')
            st.subheader("💬 Chat with Your Paper")
            st.markdown("Ask questions, request modifications, or refine your paper!")
        
            if 'chat_agent' not in st.session_state:
                st.session_state.chat_agent = PaperChatAgent()
            
            chat_container = st.container()
            
            
            if st.session_state.chat_history:
                with chat_container:
                    st.markdown("**💭 Conversation History:**")
                    
                
                    for i, message in enumerate(st.session_state.chat_history):
                        if message['role'] == 'user':
                           
                            st.info(f"**You:** {message['content']}")
                        else:
                           
                            st.success(f"**PaperPilot:**\n\n{message['content']}")
                        
                    
                        if i < len(st.session_state.chat_history) - 1:
                            st.markdown("")
                    
                    
                    st.markdown("---")
            
           
            st.markdown("**💭 Continue the conversation:**")
            
           
            col1, col2 = st.columns([5, 1])
            
            
            if 'chat_input_key' not in st.session_state:
                st.session_state.chat_input_key = 0
            
            with col1:
                user_message = st.text_area(
                    "Ask a question or request changes:",
                    placeholder="e.g., 'Explain the methodology in simpler terms' or 'Add more examples to the introduction'",
                    key=f"main_chat_input_{st.session_state.chat_input_key}",
                    height=80
                )
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True) 
                send_message = st.button("📤 Send Message", key="main_send_chat", use_container_width=True)
                clear_chat = st.button("🗑️ Clear Chat History", key="main_clear_chat", use_container_width=True)
            
           
            if clear_chat:
                st.session_state.chat_history = []
                st.session_state.chat_agent = PaperChatAgent()
                st.rerun()
            
            if send_message and user_message.strip():
              
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': user_message.strip()
                })
                
                
                with st.spinner("PaperPilot is thinking..."):
                    try:
                        paper_content = st.session_state.full_paper.get('raw_output', '')
                        paper_sections = st.session_state.full_paper.get('sections', {})
                        
                        result = st.session_state.chat_agent.process_user_input(
                            user_message.strip(), paper_content, paper_sections
                        )
                        
                        if result['type'] == 'question':
                        
                            bot_response = result['answer']
                            st.session_state.chat_history.append({
                                'role': 'bot',
                                'content': bot_response
                            })
                            
                        elif result['type'] == 'modification':
                           
                            mod_result = result['result']
                            if mod_result['success']:
                                bot_response = f"✅ **Paper Updated Successfully!**\n\n"
                                
                                if mod_result['modified_section'] and mod_result['modified_section'] in paper_sections:
                                   
                                    paper_sections[mod_result['modified_section']] = [mod_result['modified_content']]
                                    st.session_state.full_paper['sections'] = paper_sections
                                    bot_response += f"**Modified Section:** {mod_result['modified_section']}\n\n"
                                    bot_response += f"**Changes Made:**\n{mod_result['modified_content'][:400]}...\n\n"
                                else:
                                
                                    st.session_state.full_paper['raw_output'] = mod_result['modified_content']
                                    st.session_state.full_paper['sections'] = {}  
                                    bot_response += f"**Full Paper Updated**\n\n"
                                    bot_response += f"**New Content Preview:**\n{mod_result['modified_content'][:400]}...\n\n"
                                
                                bot_response += "🔄 *Your paper has been updated! The changes are reflected in the document above and will be included when you download.*"
                            else:
                                bot_response = f"❌ **Modification Failed:** {mod_result.get('error', 'Unknown error occurred')}"
                            
                            st.session_state.chat_history.append({
                                'role': 'bot',
                                'content': bot_response
                            })
                            
                    except Exception as e:
                        error_response = f"❌ **Error:** {str(e)}\n\nPlease try rephrasing your request or ask a different question."
                        st.session_state.chat_history.append({
                            'role': 'bot',
                            'content': error_response
                        })
            
                st.session_state.chat_input_key += 1
                
                st.rerun()
        
            if not st.session_state.chat_history:
                st.markdown("**✨ Quick Actions:**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("❓ Explain methodology", key="quick_method"):
                        st.session_state.chat_history.append({
                            'role': 'user', 'content': 'Explain the methodology section in simple terms'
                        })
                        st.session_state.chat_input_key += 1
                        st.rerun()
                
                with col2:
                    if st.button("📝 Improve writing", key="quick_improve"):
                        st.session_state.chat_history.append({
                            'role': 'user', 'content': 'Improve the writing quality and clarity of the paper'
                        })
                        st.session_state.chat_input_key += 1
                        st.rerun()
                
                with col3:
                    if st.button("📚 Add examples", key="quick_examples"):
                        st.session_state.chat_history.append({
                            'role': 'user', 'content': 'Add more practical examples to make the content clearer'
                        })
                        st.session_state.chat_input_key += 1
                        st.rerun()
            
        
            with st.expander("💡 Chat Tips & Examples"):
                st.markdown("""
                **Questions you can ask:**
                - "What are the key findings of this paper?"
                - "Explain the results in simpler terms"
                - "Summarize the main contributions"
                - "How does this methodology work?"
                
                **Modifications you can request:**
                - "Make the abstract more concise"
                - "Add more detail to the introduction"
                - "Improve the conclusion with stronger arguments"
                - "Fix any grammatical errors"
                - "Make the writing more engaging"
                
                **💡 Pro Tip:** Be specific in your requests for better results!
                """)
            
         
            st.markdown('---')
            st.subheader("📥 Download Paper")
            if st.button("📄 Download as DOCX", key="download_docx"):
                try:
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
                    st.success("✅ DOCX file ready for download!")
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

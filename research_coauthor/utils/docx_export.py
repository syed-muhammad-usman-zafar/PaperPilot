
import io
import base64
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
import streamlit as st

def create_paper_docx(paper_data, research_query):
   
    try:
        doc = Document()
        sections = doc.sections
        for section in sections:
            section.top_margin = Inches(1)
            section.bottom_margin = Inches(1)
            section.left_margin = Inches(1)
            section.right_margin = Inches(1)
    
        title = doc.add_heading(research_query, level=0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph()
        section_mapping = {
            "Abstract": "Abstract",
            "Introduction": "Introduction", 
            "Literature Review": "Literature Review",
            "Methodology": "Methodology",
            "Experiments / Results": "Experiments / Results",
            "Conclusion": "Conclusion"
        }
        sections_data = paper_data.get('sections', {})
        
        for section_title, section_key in section_mapping.items():
            heading = doc.add_heading(section_title, level=1)
            section_content = sections_data.get(section_key, [])
            
            if section_content and len(section_content) > 0:
                for paragraph_text in section_content:
                    if paragraph_text and paragraph_text.strip():
                        clean_text = paragraph_text.strip()
                        if not clean_text.startswith('[') or not clean_text.endswith(']'):
                            p = doc.add_paragraph(clean_text)
                            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            else:
                p = doc.add_paragraph(f"[{section_title} content to be added]")
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            # ...existing code...
            
            # Add space between sections
            doc.add_paragraph()
        
        # Add References section - check both structured and raw output
        references_added = False
        
        print(f"[DEBUG] DOCX Export - Checking for references in paper_data")
        print(f"[DEBUG] paper_data keys: {list(paper_data.keys())}")
        print(f"[DEBUG] 'references' in paper_data: {'references' in paper_data}")
        
        if 'references' in paper_data:
            print(f"[DEBUG] References content preview: {paper_data['references'][:200]}...")
        
        # First, try to use structured references if available
        if 'references' in paper_data and paper_data['references']:
            doc.add_heading('References', level=1)
            refs_content = paper_data['references']
            
            # Clean up references formatting
            if refs_content.startswith('References\n'):
                refs_content = refs_content.replace('References\n', '', 1)
            
            ref_lines = [line.strip() for line in refs_content.split('\n') if line.strip()]
            valid_refs = []
            
            for ref_line in ref_lines:
                # Filter out placeholder text and incomplete references
                if (ref_line and 
                    not ref_line.startswith('[Add ') and 
                    'more relevant references' not in ref_line and
                    not ref_line.startswith('[P') and 
                    len(ref_line) > 20):  # Ensure it's not just a fragment
                    p = doc.add_paragraph(ref_line)
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                    valid_refs.append(ref_line)
            
            references_added = True
            print(f"[DEBUG] Added {len(valid_refs)} valid references to DOCX")
            
        # If no structured references, try to extract from raw output
        elif 'raw_output' in paper_data:
            raw_output = paper_data.get('raw_output', '')
            print(f"[DEBUG] Checking raw_output for references (length: {len(raw_output)})")
            
            if 'References' in raw_output or 'REFERENCES' in raw_output:
                import re
                ref_match = re.search(r'(?:References|REFERENCES)\s*\n(.*?)(?:\n\n|\Z)', raw_output, re.DOTALL | re.IGNORECASE)
                if ref_match:
                    doc.add_heading('References', level=1)
                    ref_content = ref_match.group(1).strip()
                    
                    ref_lines = [line.strip() for line in ref_content.split('\n') if line.strip()]
                    valid_refs = []
                    
                    for ref_line in ref_lines:
                        if (ref_line and 
                            not ref_line.startswith('[Add ') and 
                            'more relevant references' not in ref_line and
                            not ref_line.startswith('[P') and
                            len(ref_line) > 20):
                            p = doc.add_paragraph(ref_line)
                            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                            valid_refs.append(ref_line)
                    
                    references_added = True
                    print(f"[DEBUG] Added {len(valid_refs)} references from raw output to DOCX")
        
        if not references_added:
            print("[DEBUG] No references were added to DOCX - references data may be missing or filtered out")
            # Add a placeholder if no references found
            doc.add_heading('References', level=1)
            p = doc.add_paragraph("References will be added here.")
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        
        # Save to bytes buffer
        docx_buffer = io.BytesIO()
        doc.save(docx_buffer)
        docx_buffer.seek(0)
        
        return docx_buffer.getvalue()
        
    except Exception as e:
        st.error(f"Error creating DOCX: {str(e)}")
        return None

def generate_filename(research_query):
    import re
    
    # Clean the research query for filename
    clean_name = re.sub(r'[^\w\s-]', '', research_query)
    clean_name = re.sub(r'[-\s]+', '_', clean_name)
    clean_name = clean_name.strip('_')
    
    # Limit length and add extension
    if len(clean_name) > 50:
        clean_name = clean_name[:50]
    
    return f"{clean_name}_research_paper.docx"

def create_download_button(paper_data, research_query, key="download_docx"):
    try:
        with st.spinner("📄 Generating editable DOCX file..."):
            docx_bytes = create_paper_docx(paper_data, research_query)
        
        if docx_bytes:
            filename = generate_filename(research_query)
            st.download_button(
                label="📥 Download as Editable DOCX",
                data=docx_bytes,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key=key,
                help="Download the research paper as an editable Microsoft Word document"
            )
            return True
        else:
            st.error("Failed to generate DOCX file")
            return False
            
    except Exception as e:
        st.error(f"Error creating download: {str(e)}")
        return False

def preview_docx_content(paper_data, research_query):
    st.markdown("### 📋 DOCX Export Preview:")
    sections_data = paper_data.get('sections', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Document Structure:**")
        st.write(f"📄 Title: {research_query}")
        
        section_count = 0
        for section_name, content in sections_data.items():
            if content and len(content) > 0:
                real_content = [p for p in content if p and not p.strip().startswith('[')]
                if real_content:
                    section_count += 1
                    word_count = sum(len(p.split()) for p in real_content)
                    st.write(f"📝 {section_name}: {len(real_content)} paragraphs (~{word_count} words)")
        
        st.write(f"✅ Total sections with content: {section_count}")
    
    with col2:
        st.markdown("**Features:**")
        st.write("✅ Formatted headings and sections")
        st.write("✅ Justified paragraph alignment") 
        st.write("✅ Proper margins and spacing")
        st.write("✅ References section")
        st.write("✅ Fully editable in Microsoft Word")
        total_chars = sum(len(str(content)) for content in sections_data.values())
        estimated_size_kb = max(20, total_chars // 50)  # Rough estimate
        st.write(f"📊 Estimated size: ~{estimated_size_kb} KB")

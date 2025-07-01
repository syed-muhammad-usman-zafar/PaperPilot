import streamlit as st
import re
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime
import json
import random
import requests

class ResearchPromptExtractor:
    
    
    def __init__(self):
        # Domain classification keywords
        self.domain_keywords = {
            'Computer Science': [
                'algorithm', 'machine learning', 'artificial intelligence', 'ai', 'ml', 'deep learning',
                'neural network', 'software', 'programming', 'data science', 'computer vision',
                'natural language processing', 'nlp', 'robotics', 'cybersecurity', 'blockchain',
                'database', 'web', 'mobile app', 'user interface', 'ux', 'ui'
            ],
            'Biology/Medicine': [
                'clinical', 'medical', 'healthcare', 'disease', 'treatment', 'patient', 'diagnosis',
                'therapeutic', 'pharmaceutical', 'drug', 'vaccine', 'genetic', 'dna', 'rna',
                'protein', 'cell', 'tissue', 'organ', 'biology', 'biomedical', 'epidemiology',
                'pathology', 'surgery', 'hospital', 'clinic'
            ],
            'Psychology': [
                'behavior', 'cognitive', 'mental health', 'psychology', 'psychological',
                'therapy', 'counseling', 'emotion', 'learning', 'memory', 'perception',
                'social psychology', 'developmental', 'personality', 'stress', 'anxiety',
                'depression', 'wellbeing', 'motivation'
            ],
            'Physics': [
                'quantum', 'mechanics', 'thermodynamics', 'electromagnetism', 'optics',
                'particle', 'relativity', 'energy', 'force', 'wave', 'frequency',
                'physics', 'physical', 'nuclear', 'atomic', 'molecular'
            ],
            'Chemistry': [
                'chemical', 'chemistry', 'reaction', 'compound', 'molecule', 'element',
                'catalyst', 'synthesis', 'organic', 'inorganic', 'analytical',
                'spectroscopy', 'chromatography', 'polymer', 'material science'
            ],
            'Economics/Business': [
                'economic', 'economics', 'financial', 'market', 'business', 'trade',
                'investment', 'profit', 'revenue', 'cost', 'pricing', 'consumer',
                'supply', 'demand', 'inflation', 'gdp', 'finance', 'banking'
            ],
            'Education': [
                'education', 'educational', 'learning', 'teaching', 'student', 'school',
                'curriculum', 'pedagogy', 'assessment', 'academic', 'classroom',
                'instruction', 'knowledge', 'skill'
            ],
            'Environmental Science': [
                'environment', 'environmental', 'climate', 'sustainability', 'carbon',
                'emission', 'renewable', 'energy', 'pollution', 'ecosystem',
                'biodiversity', 'conservation', 'green', 'ecological'
            ]
        }
        
        # Research methods keywords
        self.method_keywords = {
            'survey': ['survey', 'questionnaire', 'poll', 'interview', 'focus group'],
            'experiment': ['experiment', 'experimental', 'trial', 'test', 'controlled study'],
            'analysis': ['analysis', 'analyze', 'statistical analysis', 'data analysis'],
            'modeling': ['model', 'modeling', 'simulation', 'mathematical model'],
            'review': ['review', 'literature review', 'systematic review', 'meta-analysis'],
            'observation': ['observation', 'observational', 'ethnography', 'case study'],
            'comparison': ['compare', 'comparison', 'comparative', 'versus', 'vs'],
            'correlation': ['correlation', 'relationship', 'association', 'connection']
        }
        
        # Research objectives keywords
        self.objective_keywords = {
            'investigate': ['investigate', 'investigation', 'explore', 'exploration', 'examine'],
            'analyze': ['analyze', 'analysis', 'assess', 'assessment', 'evaluate', 'evaluation'],
            'compare': ['compare', 'comparison', 'contrast', 'differentiate'],
            'measure': ['measure', 'measurement', 'quantify', 'determine'],
            'predict': ['predict', 'prediction', 'forecast', 'estimate'],
            'understand': ['understand', 'comprehend', 'explain', 'clarify'],
            'improve': ['improve', 'optimize', 'enhance', 'better'],
            'develop': ['develop', 'create', 'design', 'build', 'construct'],
            'validate': ['validate', 'verify', 'confirm', 'test', 'prove']
        }
        
        # Data types keywords
        self.data_type_keywords = {
            'quantitative': ['quantitative', 'numerical', 'statistical', 'numeric', 'measurement'],
            'qualitative': ['qualitative', 'interview', 'narrative', 'textual', 'observational'],
            'survey data': ['survey', 'questionnaire', 'response', 'feedback'],
            'experimental data': ['experimental', 'trial', 'laboratory', 'controlled'],
            'observational data': ['observational', 'field study', 'natural setting'],
            'secondary data': ['existing data', 'published', 'archive', 'database'],
            'longitudinal': ['longitudinal', 'time series', 'over time', 'temporal'],
            'cross-sectional': ['cross-sectional', 'snapshot', 'point in time']
        }
    
    def extract_domain(self, prompt: str) -> Tuple[str, float]:
        """Extract the most likely research domain from the prompt"""
        prompt_lower = prompt.lower()
        domain_scores: dict[str, float] = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords:
                
                occurrences = len(re.findall(r'\b' + re.escape(keyword) + r'\b', prompt_lower))
                
                weight = len(keyword.split()) * 1.5 if len(keyword.split()) > 1 else 1
                score += occurrences * weight
            
            domain_scores[domain] = score
        
        if not domain_scores or max(domain_scores.values()) == 0:
            return "General/Interdisciplinary", 0.0
        
        best_domain = max(domain_scores, key=lambda k: domain_scores[k])
        confidence = domain_scores[best_domain] / sum(domain_scores.values()) if sum(domain_scores.values()) > 0 else 0
        
        return best_domain, confidence
    
    def extract_methods(self, prompt: str) -> List[Tuple[str, int]]:
        """Extract research methods mentioned in the prompt"""
        prompt_lower = prompt.lower()
        found_methods = []
        
        for method, keywords in self.method_keywords.items():
            count = 0
            for keyword in keywords:
                count += len(re.findall(r'\b' + re.escape(keyword) + r'\b', prompt_lower))
            
            if count > 0:
                found_methods.append((method, count))
        
        return sorted(found_methods, key=lambda x: x[1], reverse=True)
    
    def extract_objectives(self, prompt: str) -> List[Tuple[str, int]]:
        """Extract research objectives from the prompt"""
        prompt_lower = prompt.lower()
        found_objectives = []
        
        for objective, keywords in self.objective_keywords.items():
            count = 0
            for keyword in keywords:
                count += len(re.findall(r'\b' + re.escape(keyword) + r'\b', prompt_lower))
            
            if count > 0:
                found_objectives.append((objective, count))
        
        return sorted(found_objectives, key=lambda x: x[1], reverse=True)
    
    def extract_data_types(self, prompt: str) -> List[Tuple[str, int]]:
      
        prompt_lower = prompt.lower()
        found_data_types = []
        
        for data_type, keywords in self.data_type_keywords.items():
            count = 0
            for keyword in keywords:
                count += len(re.findall(r'\b' + re.escape(keyword) + r'\b', prompt_lower))
            
            if count > 0:
                found_data_types.append((data_type, count))
        
        return sorted(found_data_types, key=lambda x: x[1], reverse=True)
    
    def extract_key_concepts(self, prompt: str) -> List[str]:
        
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'want', 'study', 'research'}
        
        words = re.findall(r'\b\w+\b', prompt.lower())
        candidates = []
        for i in range(len(words)):
            if words[i] not in stop_words and len(words[i]) > 3:
               
                if i < len(words) - 1 and words[i+1] not in stop_words:
                    phrase = f"{words[i]} {words[i+1]}"
                    candidates.append(phrase.title())
                candidates.append(words[i].title())
       
        unique_concepts = []
        seen = set()
        for c in candidates:
            if c.lower() not in seen and len(c) > 3 and c.lower() not in stop_words:
                unique_concepts.append(c)
                seen.add(c.lower())
        # Prioritize multi-word concepts
        multi_word = [c for c in unique_concepts if ' ' in c]
        single_word = [c for c in unique_concepts if ' ' not in c]
        return multi_word[:5] + single_word[:3]
    
    def process_prompt(self, prompt: str) -> Dict:
       
        domain, domain_confidence = self.extract_domain(prompt)
        methods = self.extract_methods(prompt)
        objectives = self.extract_objectives(prompt)
        data_types = self.extract_data_types(prompt)
        key_concepts = self.extract_key_concepts(prompt)
        
        return {
            'domain': {
                'name': domain,
                'confidence': domain_confidence
            },
            'methods': methods,
            'objectives': objectives,
            'data_types': data_types,
            'key_concepts': key_concepts,
            'prompt_length': len(prompt),
            'word_count': len(prompt.split())
        }

def get_real_source_summaries(keywords, max_results=2):
   
    query = ' '.join(keywords[:4])
    url = f'https://api.semanticscholar.org/graph/v1/paper/search?query={query}&fields=title,authors,year,venue,abstract&limit={max_results}'
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
      
        print(f"[DEBUG] Querying Semantic Scholar with: {query}")
        print(f"[DEBUG] API URL: {url}")
        print(f"[DEBUG] API Response: {data}")
        papers = data.get('data', [])
        results = []
        for i, paper in enumerate(papers):
            title = paper.get('title', 'No Title')
            authors = ', '.join([a.get('name', '') for a in paper.get('authors', [])][:3])
            year = paper.get('year', 'n.d.')
            venue = paper.get('venue', 'Unknown Venue')
            abstract = paper.get('abstract', '')
            citation = f"[{i+1}] {authors}, '{title}', {venue}, {year}"
            summary = abstract if abstract else f"{title} discusses topics related to {query}."
            results.append({
                'title': title,
                'summary': summary,
                'citation': citation,
                'ref': citation
            })
        return results
    except Exception as e:
        print(f"[DEBUG] Exception in get_real_source_summaries: {e}")
        return []

def truncate_at_sentence(text, max_len=300):
    if len(text) <= max_len:
        return text
    sentences = re.split(r'(?<=[.!?]) +', text)
    out = ''
    for s in sentences:
        if len(out) + len(s) > max_len:
            break
        out += s + ' '
    return out.strip()

def generate_bullet_summaries(summaries):
    bullets = []
    for s in summaries:
        summary = s['summary']
        citation = s['citation']
        if summary:
            summary_trunc = truncate_at_sentence(summary, 300)
            bullets.append(f"- {summary_trunc} ({citation})")
        else:
            bullets.append(f"- {s['title']} ({citation})")
    return bullets

def get_year_from_citation(citation):
    match = re.search(r", (\d{4})[.,)]", citation)
    if match:
        return match.group(1)
    return "n.d."

def get_first_author(citation):
    match = re.match(r"\[\d+\] ([^,]+)", citation)
    if match:
        return match.group(1)
    return "Unknown"

def generate_intro_paragraph(domain, keywords, method, objective, summaries):
    if summaries:
        first = summaries[0]
        second = summaries[1] if len(summaries) > 1 else None
        para = f"Recent research in {domain.lower()} has leveraged {method} to {objective}. "
        if first['summary']:
            year1 = get_year_from_citation(first['citation'])
            author1 = get_first_author(first['citation'])
            summary1 = truncate_at_sentence(first['summary'], 180)
            para += f"For example, {author1} et al. ({year1}) found that {summary1} "
        if second and second['summary']:
            year2 = get_year_from_citation(second['citation'])
            author2 = get_first_author(second['citation'])
            summary2 = truncate_at_sentence(second['summary'], 180)
            para += f"Additionally, {author2} et al. ({year2}) reported that {summary2} "
        
        para = re.sub(rf"{method} to {objective}", f"{method} for the purpose of {objective}", para)
        para = re.sub(rf"{objective} using {method}", f"{objective} with the help of {method}", para)
        para += f"Building on these findings, our study will {objective} with the help of {method}, focusing on {', '.join(keywords[:2]) if keywords else 'key concepts'}."
        return para.strip()
    return "Insufficient source data to generate a contextual paragraph."

def citation_agent(paragraph, summaries):

    sentences = paragraph.split('.')
    cited = []
    for i, sent in enumerate(sentences):
        sent = sent.strip()
        if sent:
            if i < len(summaries):
                sent += f" {summaries[i]['citation']}"
            cited.append(sent)
    cited_paragraph = '. '.join(cited).strip()
    if not cited_paragraph.endswith('.'):
        cited_paragraph += '.'
    return cited_paragraph

# Knowledge Graph Data Structure 
def build_knowledge_graph(domain, keywords, method, objective, summaries, draft_paragraph):
    
    graph = {
        'Domain': domain,
        'Keywords': keywords,
        'Method': method,
        'Objective': objective,
        'Sources': [s['citation'] for s in summaries],
        'DraftParagraph': draft_paragraph
    }
    return graph

def main():
    st.set_page_config(
        page_title="Research Prompt NLP Extractor",
        page_icon="🔬",
        layout="wide"
    )
    st.title("🔬 AI Research Paper Co-Author System")

    extractor = ResearchPromptExtractor()

    tabs = st.tabs(["Prompt Extraction", "Drafting (AI Co-Author)"])


    with tabs[0]:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.header("📝 Research Prompt Input")
            if hasattr(st.session_state, 'sample_prompt'):
                default_text = st.session_state.sample_prompt
            else:
                default_text = ""
            user_prompt = st.text_area(
                "Describe your research idea:",
                value=default_text,
                height=200,
                placeholder="Example: I want to study the effects of social media on teenage mental health using survey data and statistical analysis to understand behavioral patterns...",
                help="Describe your research in natural language - mention your goals, methods, and what you want to study"
            )
            st.caption(f"📊 **{len(user_prompt)}** characters, **{len(user_prompt.split())}** words")
            if st.button("🧠 Extract Research Elements", type="primary", disabled=len(user_prompt.strip()) < 10, key="extract_btn"):
                if user_prompt.strip():
                    with st.spinner("Analyzing your research prompt..."):
                        results = extractor.process_prompt(user_prompt)
                        st.session_state.results = results
                        st.session_state.processed_prompt = user_prompt
                        st.success("✅ Analysis complete!")
                else:
                    st.warning("Please enter a research prompt (at least 10 characters)")
        with col2:
            st.header("🎯 Extracted Research Elements")
            if hasattr(st.session_state, 'results'):
                results = st.session_state.results
                st.subheader("🏷️ Research Domain")
                domain_name = results['domain']['name']
                confidence = results['domain']['confidence']
                if confidence > 0:
                    st.success(f"**{domain_name}**")
                    st.progress(confidence)
                    st.caption(f"Confidence: {confidence:.1%}")
                else:
                    st.info("**General/Interdisciplinary** - No specific domain detected")
                st.subheader("🔬 Research Methods")
                if results['methods']:
                    methods_df = pd.DataFrame(results['methods'], columns=['Method', 'Mentions'])
                    st.dataframe(methods_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No specific research methods detected")
                st.subheader("🎯 Research Objectives")
                if results['objectives']:
                    objectives_df = pd.DataFrame(results['objectives'], columns=['Objective', 'Mentions'])
                    st.dataframe(objectives_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No clear research objectives detected")
                st.subheader("📊 Data Types")
                if results['data_types']:
                    data_types_df = pd.DataFrame(results['data_types'], columns=['Data Type', 'Mentions'])
                    st.dataframe(data_types_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No specific data types mentioned")
            else:
                st.info("👈 Enter and process a research prompt to see extracted elements")
        if hasattr(st.session_state, 'results'):
            st.header("💡 Key Concepts & Summary")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("🔑 Extracted Key Concepts")
                if st.session_state.results['key_concepts']:
                    concepts_html = ""
                    for concept in st.session_state.results['key_concepts']:
                        concepts_html += f'<span style="background-color: #050505; padding: 4px 8px; margin: 2px; border-radius: 12px; display: inline-block; font-size: 14px;">{concept}</span> '
                    st.markdown(concepts_html, unsafe_allow_html=True)
                else:
                    st.info("No key concepts extracted")
            with col2:
                st.subheader("📈 Analysis Stats")
                st.metric("Domain Confidence", f"{st.session_state.results['domain']['confidence']:.1%}")
                st.metric("Methods Found", len(st.session_state.results['methods']))
                st.metric("Objectives Found", len(st.session_state.results['objectives']))
                st.metric("Data Types Found", len(st.session_state.results['data_types']))

    # --- Drafting Tab ---
    with tabs[1]:
        st.header("Drafting Assistant (AI Co-Author)")
        if not hasattr(st.session_state, 'results'):
            st.info("Please extract research elements in the first tab before drafting.")
        else:
            results = st.session_state.results
            with st.expander("🔍 Review & Edit Extracted Elements", expanded=True):
                domain = st.text_input("Domain", value=results['domain']['name'], key="draft_domain")
                keywords = st.text_input("Keywords (comma-separated)", value=", ".join(results['key_concepts']), key="draft_keywords")
                method = st.text_input("Method", value=results['methods'][0][0] if results['methods'] else '', key="draft_method")
                objective = st.text_input("Objective", value=results['objectives'][0][0] if results['objectives'] else '', key="draft_objective")
            if 'draft_history' not in st.session_state:
                st.session_state.draft_history = []
            st.markdown("---")
            st.subheader("Draft Conversation")
            for i, msg in enumerate(st.session_state.draft_history):
                with st.container():
                    st.markdown(f"<div style='background:#f5f5f5; border-radius:10px; padding:10px; margin-bottom:8px;'><b>{msg['role']}:</b> {msg['content']}</div>", unsafe_allow_html=True)
            if st.button("✍️ Generate Draft Paragraph", type="primary", key="generate_paragraph"):
                with st.spinner("Retrieving real sources from Semantic Scholar..."):
                    kw_list = [k.strip() for k in keywords.split(',') if k.strip()]
                    st.write(f"**[DEBUG] Using keywords for Semantic Scholar:** {kw_list}")
                    summaries = get_real_source_summaries(kw_list, max_results=2)
                    st.write(f"**[DEBUG] Summaries returned:** {summaries}")
                    if not summaries:
                        st.warning("No real sources found. Please try different keywords or check your internet connection.")
                        return
                    bullet_points = generate_bullet_summaries(summaries)
                    st.session_state.draft_history.append({
                        'role': 'Research Agent',
                        'content': 'Key findings from recent literature:'
                    })
                    for bullet in bullet_points:
                        st.session_state.draft_history.append({
                            'role': 'Source',
                            'content': bullet
                        })
                    draft_paragraph = generate_intro_paragraph(domain, kw_list, method, objective, summaries)
                    st.session_state.draft_history.append({
                        'role': 'Writing Agent',
                        'content': draft_paragraph
                    })
                    with st.spinner("Citation Agent: Attaching citations..."):
                        cited_paragraph = citation_agent(draft_paragraph, summaries)
                        st.session_state.draft_history.append({
                            'role': 'Citation Agent',
                            'content': cited_paragraph
                        })
                    st.session_state.knowledge_graph = build_knowledge_graph(domain, kw_list, method, objective, summaries, cited_paragraph)
                    st.session_state.reference_list = [s['ref'] for s in summaries]
            if 'knowledge_graph' in st.session_state:
                st.markdown('---')
                st.subheader(' Knowledge Graph (Preview)')
                kg = st.session_state.knowledge_graph
                st.json(kg)
            if 'reference_list' in st.session_state:
                st.markdown('---')
                st.subheader('📚 References')
                for ref in st.session_state.reference_list:
                    st.markdown(f"- {ref}")
           

if __name__ == "__main__":
    main()
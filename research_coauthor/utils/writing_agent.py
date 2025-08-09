import os
import json
from dotenv import load_dotenv
from .model_config import generate_with_optimal_model, TaskType

load_dotenv()


def analyze_knowledge_graph(G):
    """Analyze the knowledge graph efficiently without additional LLM calls."""
    insights = []
    
    # Basic graph structure
    node_count = len(G.nodes)
    edge_count = len(G.edges)
    insights.append(f"Graph: {node_count} nodes, {edge_count} edges")
    
    # Analyze paper nodes and their content
    paper_nodes = [n for n, d in G.nodes(data=True) if d.get('type') in ['paper', 'user_research']]
    insights.append(f"Papers: {len(paper_nodes)}")
    
    # Extract key themes from paper summaries (without LLM)
    all_summaries = []
    total_words = 0
    
    for node_id in paper_nodes:
        node_data = G.nodes[node_id]
        summary = node_data.get('summary', '')
        if summary and len(summary.strip()) > 10:
            all_summaries.append(summary)
            total_words += len(summary.split())
    
    if all_summaries:
        insights.append(f"Content: {total_words} words")
        
        # Simple keyword extraction (no LLM needed)
        common_words = []
        for summary in all_summaries:
            # Extract meaningful words (>4 chars, not common words)
            words = [w.lower().strip('.,!?()[]') for w in summary.split() 
                    if len(w) > 4 and w.lower() not in ['research', 'study', 'paper', 'analysis', 'method', 'results']]
            common_words.extend(words)
        
        if common_words:
            from collections import Counter
            word_freq = Counter(common_words)
            # Only include words that appear multiple times
            themes = [word for word, count in word_freq.most_common(5) if count > 1]
            if themes:
                insights.append(f"Themes: {', '.join(themes)}")
    
    # Count user research
    user_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'user_research']
    if user_nodes:
        insights.append(f"User papers: {len(user_nodes)}")
    
    # Count keywords
    keyword_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'keyword']
    if keyword_nodes:
        insights.append(f"Keywords: {len(keyword_nodes)}")
    
    return insights

def generate_full_paper_with_llm(context, papers, knowledge_graph_summary, user_research_context=None):
    """
    Generate the full research paper with proper in-text citations - Uses ALL available papers.
    """
    # Build concise paper summaries with content for writing - USE ALL PAPERS
    paper_summaries = []
    citation_mapping = {}
    
    # Process ALL papers, not just first 6
    for i, paper in enumerate(papers):  # Use ALL papers available
        ref_num = i + 1
        author_names = paper.get('author_names', 'Unknown Author')
        title = paper.get('title', 'No Title')
        year = paper.get('year', 'n.d.')
        summary = paper.get('summary', 'No content available')
        
        # Create citation mapping for in-text references
        citation_key = f"{{{ref_num}}}"
        citation_mapping[ref_num] = {
            'in_text': citation_key,
            'author': author_names.split(',')[0] if ',' in author_names else author_names,
            'year': year
        }
        
        # Add concise content for each paper (token-efficient but comprehensive)
        paper_summaries.append(
            f"{citation_key}: {author_names[:30]}({year})-{summary[:120]}..."
        )
    
    print(f"[DEBUG] Using ALL {len(papers)} papers for citations and content")
    
    # Create concise citation examples - show first 10, indicate more available
    citation_examples = "; ".join([
        f"{info['in_text']}={info['author']}({info['year']})"
        for info in list(citation_mapping.values())[:10]
    ])
    if len(papers) > 10:
        citation_examples += f"; ...and {len(papers)-10} more papers"
    
    
    papers_chunk1 = "\n".join(paper_summaries[:len(papers)//2])
    papers_chunk2 = "\n".join(paper_summaries[len(papers)//2:])
    
    # Minimal context to save tokens
    user_context_str = ""
    if user_research_context and user_research_context.get('summary'):
        user_context_str = f"\nUSER RESEARCH PRIORITY: {user_research_context['summary'][:200]}...\n(Integrate this user research content throughout the paper)\n"
    
    prompt = (
        f"Write academic paper: {context[:120]}\n\n"
        f"CITATIONS: Use {{1}}-{{{len(papers)}}} throughout ALL sections.\n"
        f"Examples: {citation_examples}\n\n"
        f"PAPERS (Part 1):\n{papers_chunk1}\n\n"
        f"PAPERS (Part 2):\n{papers_chunk2}\n\n"
        f"{user_context_str}\n\n"
        f"Write with citations from ALL {len(papers)} papers:\n"
        f"**ABSTRACT** (120w): Problem, method, findings. Cite key papers.\n"
        f"**INTRODUCTION** (180w): Background, gap, objectives. Use relevant citations.\n"
        f"**LITERATURE REVIEW** (300w): Analyze ALL papers, group themes, cite extensively.\n"
        f"**METHODOLOGY** (140w): Approach, cite methodological papers.\n"
        f"**EXPERIMENTS/RESULTS** (180w): Findings, compare with ALL relevant papers.\n"
        f"**CONCLUSION** (100w): Summary, future work, cite supporting papers.\n"
        f"**REFERENCES** (list [1]-[{len(papers)}])\n\n"
        f"CRITICAL: Reference ALL {len(papers)} papers contextually. Use actual content."
    )
    
    try:
        # Use Gemini 2.0 Flash with increased efficiency
        from .model_config import generate_with_optimal_model, TaskType
        response = generate_with_optimal_model(TaskType.GENERATION, prompt, max_output_tokens=1536)  # Original balanced setting
        raw_output = response.text
        
        # Section parsing - handle the exact format the LLM is generating
        import re
        
        sections = {}
        section_titles = ["Abstract", "Introduction", "Literature Review", "Methodology", "Experiments / Results", "Conclusion"]
        
        # Split by the **SECTION** pattern that the LLM is actually generating
        section_splits = re.split(r'\*\*([A-Z][^*]+)\*\*', raw_output, flags=re.IGNORECASE)
        
        if len(section_splits) > 1:
            for i in range(1, len(section_splits), 2):
                if i + 1 < len(section_splits):
                    section_name = section_splits[i].strip()
                    section_content = section_splits[i + 1].strip()
                    
                    # Map to standard section names
                    standard_name = None
                    for std_name in section_titles:
                        if section_name.upper() in std_name.upper() or std_name.upper() in section_name.upper():
                            standard_name = std_name
                            break
                        # Handle "Experiments / Results" variations
                        if "EXPERIMENT" in section_name.upper() or "RESULT" in section_name.upper():
                            standard_name = "Experiments / Results"
                            break
                    
                    if standard_name and section_content and len(section_content) > 20:
                        # Split into paragraphs
                        paragraphs = [p.strip() for p in section_content.split("\n\n") if len(p.strip()) > 20]
                        if not paragraphs and section_content:
                            paragraphs = [section_content]
                        sections[standard_name] = paragraphs
        
        # Fill in missing sections with placeholders
        for section_title in section_titles:
            if section_title not in sections:
                sections[section_title] = [f"[{section_title} section not found in output]"]
        
        return {"raw_output": raw_output, "sections": sections}
        
    except Exception as e:
        # Minimal fallback without additional LLM calls
        section_titles = ["Abstract", "Introduction", "Literature Review", "Methodology", "Results", "Conclusion"]
        return {
            "raw_output": "[Error generating paper]", 
            "sections": {title: [f"[{title} not generated due to error]"] for title in section_titles}
        }
import openai
import os
import networkx as nx
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def analyze_knowledge_graph(G):
    """Analyze the knowledge graph to extract meaningful relationships and insights."""
    insights = []
    
    # Get all paper nodes
    paper_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'paper']
    
    # Analyze relationships between keywords and papers
    keyword_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'keyword']
    for keyword in keyword_nodes:
        related_papers = list(G.successors(keyword))
        if related_papers:
            insights.append(f"Keyword '{keyword}' is supported by {len(related_papers)} papers")
    
    # Analyze author contributions
    for paper in paper_nodes:
        authors = [n for n in G.successors(paper) if G.nodes[n].get('type') == 'author']
        if authors:
            paper_title = G.nodes[paper].get('title', paper)
            insights.append(f"Paper '{paper_title[:50]}...' has {len(authors)} author(s)")
    
    # Find central concepts (nodes with most connections)
    centrality = nx.degree_centrality(G)
    central_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:3]
    insights.append(f"Most central concepts: {[node for node, _ in central_nodes]}")
    
    # Analyze research methodology connections
    method_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'method']
    for method in method_nodes:
        supporting_papers = list(G.predecessors(method))
        insights.append(f"Method '{method}' is supported by {len(supporting_papers)} papers")
    
    return insights

def get_contextual_papers(G, context_keywords):
    """Get papers that are most relevant to the current context."""
    relevant_papers = []
    
    for keyword in context_keywords:
        if keyword in G.nodes:
            # Get papers directly related to this keyword
            related_papers = list(G.successors(keyword))
            for paper in related_papers:
                if G.nodes[paper].get('type') == 'paper':
                    relevant_papers.append(paper)
    
    # Also get papers that are connected to multiple keywords (higher relevance)
    multi_connected_papers = []
    for paper in G.nodes():
        if G.nodes[paper].get('type') == 'paper':
            incoming_keywords = [n for n in G.predecessors(paper) if G.nodes[n].get('type') == 'keyword']
            if len(incoming_keywords) > 1:
                multi_connected_papers.append((paper, len(incoming_keywords)))
    
    # Sort by number of keyword connections
    multi_connected_papers.sort(key=lambda x: x[1], reverse=True)
    
    return relevant_papers, multi_connected_papers

def writing_agent(context, papers, key_concepts, knowledge_graph=None):
    """Generate a draft paragraph using OpenAI LLM with knowledge graph reasoning."""
    
    # Build a structured list of papers
    paper_list = "\n".join([
        f"[{i+1}] {p['author_names']}, '{p['title']}', {p.get('venue', 'Unknown Venue')}, {p.get('year', 'n.d.')}"
        for i, p in enumerate(papers)
    ])
    
    # Build a structured list of key concepts
    key_concepts_str = ", ".join(key_concepts)
    
    # Enhanced system prompt with knowledge graph insights
    system_prompt = (
        "You are an expert research co-author with access to a knowledge graph that connects research concepts, "
        "papers, authors, and methodologies. Write a short, human-like introduction paragraph for a research paper. "
        "You must cite ONLY from the following papers (use (Author, Year) style):\n"
        f"{paper_list}\n"
        "You must cover the following key concepts: "
        f"{key_concepts_str}. "
        "Do not invent citations, datasets, or references. Ensure at least one citation is included."
    )
    
    # Add knowledge graph insights if available
    if knowledge_graph:
        insights = analyze_knowledge_graph(knowledge_graph)
        relevant_papers, multi_connected = get_contextual_papers(knowledge_graph, key_concepts)
        
        graph_insights = "\n".join(insights[:5])  # Limit to top 5 insights
        
        # Add information about highly connected papers
        if multi_connected:
            central_papers = [f"- {p[0]}: connected to {p[1]} keywords" for p in multi_connected[:3]]
            graph_insights += f"\n\nMost relevant papers (by keyword connections):\n" + "\n".join(central_papers)
        
        system_prompt += f"\n\nKnowledge Graph Insights:\n{graph_insights}\n\nUse these insights to create more coherent and well-connected content."
    
    user_content = f"Context: {context}"
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        temperature=0.5,
        max_tokens=300
    )
    
    return response.choices[0].message.content

def group_papers_by_theme(papers: List[Dict], keywords: List[str]) -> Dict[str, List[Dict]]:
    """
    Group papers by thematic similarity for Literature Review section.
    
    Args:
        papers: List of paper summaries
        keywords: Research keywords for context
    
    Returns:
        Dictionary mapping themes to grouped papers
    """
    if not papers:
        return {}
    
    # Create themes based on keywords and paper content
    themes = {}
    
    for paper in papers:
        title = paper.get('title', '').lower()
        summary = paper.get('summary', '').lower()
        content = f"{title} {summary}"
        
        # Find the most relevant keyword for this paper
        best_theme = "General"
        max_relevance = 0
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            relevance = content.count(keyword_lower)
            if relevance > max_relevance:
                max_relevance = relevance
                best_theme = keyword
        
        if best_theme not in themes:
            themes[best_theme] = []
        themes[best_theme].append(paper)
    
    return themes

def generate_section_paragraphs(section_name: str, papers: List[Dict], 
                              context: str, n_paragraphs: int = 2,
                              section_type: str = "general") -> List[str]:
    """
    Generate multiple paragraphs for a paper section using LLM.
    
    Args:
        section_name: Name of the section (e.g., "Introduction")
        papers: List of papers assigned to this section
        context: Research context and keywords
        n_paragraphs: Number of paragraphs to generate
        section_type: Type of section for specialized generation
    
    Returns:
        List of generated paragraphs
    """
    if not papers:
        # Generate placeholder content
        return [f"This {section_name.lower()} section would typically discuss the research context and objectives."]
    
    # Build paper list for citation
    paper_list = "\n".join([
        f"[{i+1}] {p['author_names']}, '{p['title']}', {p.get('venue', 'Unknown Venue')}, {p.get('year', 'n.d.')}"
        for i, p in enumerate(papers)
    ])
    
    # Specialized prompts for different sections
    if section_type == "literature_review":
        system_prompt = (
            f"You are an expert academic writer. Write {n_paragraphs} well-structured paragraphs for the {section_name} section. "
            f"Focus on synthesizing and comparing the following papers:\n{paper_list}\n"
            f"Use academic transitions like 'Similarly,' 'In contrast,' 'Building on this work,' 'Furthermore,' etc. "
            f"Each paragraph should flow naturally to the next. Cite papers using (Author, Year) format. "
            f"Context: {context}"
        )
    elif section_type == "methodology":
        system_prompt = (
            f"You are an expert academic writer. Write {n_paragraphs} paragraphs for the {section_name} section. "
            f"Focus on research methods and approaches, citing relevant methodological papers:\n{paper_list}\n"
            f"Context: {context}"
        )
    else:
        system_prompt = (
            f"You are an expert academic writer. Write {n_paragraphs} paragraphs for the {section_name} section. "
            f"Use the following papers for citations:\n{paper_list}\n"
            f"Context: {context}"
        )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate {n_paragraphs} paragraphs for the {section_name} section."}
            ],
            temperature=0.7,
            max_tokens=800
        )
        
        content = response.choices[0].message.content
        if not content:
            return [f"Error generating {section_name} content. Please try again."]
        # Split into paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        # Ensure we have the requested number of paragraphs
        if len(paragraphs) < n_paragraphs:
            # Pad with additional content
            while len(paragraphs) < n_paragraphs:
                paragraphs.append(f"Additional {section_name.lower()} content would be developed here.")
        elif len(paragraphs) > n_paragraphs:
            # Truncate to requested number
            paragraphs = paragraphs[:n_paragraphs]
        
        return paragraphs
        
    except Exception as e:
        print(f"Error generating {section_name} paragraphs: {e}")
        return [f"Error generating {section_name} content. Please try again."]

def generate_literature_review_section(papers: List[Dict], context: str, keywords: List[str]) -> List[str]:
    """
    Generate a comprehensive Literature Review section with thematic grouping.
    
    Args:
        papers: List of papers for literature review
        context: Research context
        keywords: Research keywords
    
    Returns:
        List of paragraphs for the Literature Review section
    """
    if not papers:
        return ["The literature review would examine existing research in this domain."]
    
    # Group papers by theme
    themed_papers = group_papers_by_theme(papers, keywords)
    
    paragraphs = []
    
    # Generate introduction paragraph
    intro_prompt = (
        "Write an introductory paragraph for a literature review section that sets up the research context. "
        f"Context: {context}"
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert academic writer specializing in literature reviews."},
                {"role": "user", "content": intro_prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        content = response.choices[0].message.content
        if content:
            paragraphs.append(content.strip())
        else:
            paragraphs.append("This literature review examines the current state of research in this domain.")
    except:
        paragraphs.append("This literature review examines the current state of research in this domain.")
    
    # Generate thematic paragraphs
    for theme, theme_papers in themed_papers.items():
        if not theme_papers:
            continue
            
        paper_list = "\n".join([
            f"[{i+1}] {p['author_names']}, '{p['title']}', {p.get('venue', 'Unknown Venue')}, {p.get('year', 'n.d.')}"
            for i, p in enumerate(theme_papers)
        ])
        
        theme_prompt = (
            f"Write a paragraph discussing research related to '{theme}'. "
            f"Use these papers: {paper_list}\n"
            f"Use academic transitions and cite papers using (Author, Year) format. "
            f"Context: {context}"
        )
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert academic writer. Write coherent literature review paragraphs with smooth transitions."},
                    {"role": "user", "content": theme_prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            content = response.choices[0].message.content
            if content:
                paragraphs.append(content.strip())
            else:
                paragraphs.append(f"Research in {theme} has been explored by various authors in the field.")
        except:
            paragraphs.append(f"Research in {theme} has been explored by various authors in the field.")
    
    # Generate conclusion paragraph for literature review
    conclusion_prompt = (
        "Write a concluding paragraph for the literature review that summarizes key findings and identifies research gaps. "
        f"Context: {context}"
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert academic writer."},
                {"role": "user", "content": conclusion_prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        content = response.choices[0].message.content
        if content:
            paragraphs.append(content.strip())
        else:
            paragraphs.append("This review highlights the current state of research and identifies areas for future investigation.")
    except:
        paragraphs.append("This review highlights the current state of research and identifies areas for future investigation.")
    
    return paragraphs 
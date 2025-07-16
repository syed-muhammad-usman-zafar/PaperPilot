import os
import json
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("OPENAI_API_KEY"))
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

def analyze_knowledge_graph(G):
    # Example: return a list of insights from the knowledge graph
    return [f"Graph has {len(G.nodes)} nodes and {len(G.edges)} edges."]

def writing_agent(context, papers, key_concepts, knowledge_graph=None):
    paper_list = "\n".join([
        f"{p.get('author_names', 'Unknown Author')}, '{p.get('title', 'No Title')}', {p.get('venue', 'Unknown Venue')}, {p.get('year', 'n.d.')}"
        for p in papers
    ])
    key_concepts_str = ", ".join(key_concepts)
    system_prompt = (
        "You are an expert research co-author. Write a short, human-like introduction paragraph for a research paper. "
        "Cite ONLY from the following papers (use (Author, Year) style):\n"
        f"{paper_list}\n"
        "Cover the following key concepts: "
        f"{key_concepts_str}. "
        "Do not invent citations, datasets, or references. Ensure at least one citation is included."
    )
    user_content = f"Context: {context}"
    full_prompt = f"{system_prompt}\n{user_content}"
    try:
        response = model.generate_content(full_prompt, generation_config={"max_output_tokens": 100})
        return response.text
    except Exception as e:
        print(f"[DEBUG] Gemini writing_agent failed: {e}")
        return "[Error generating writing output.]"

def group_papers_by_theme(papers, keywords):
    if not papers:
        return {}
    themes = {}
    for paper in papers:
        title = paper.get('title', '').lower()
        summary = paper.get('summary', '').lower()
        content = f"{title} {summary}"
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

def generate_section_paragraphs(section_name, papers, context, n_paragraphs=2, section_type="general", knowledge_graph=None, citation_map=None, user_research_context=None):
    if not papers:
        return [f"This {section_name.lower()} section would typically discuss the research context and objectives."]
    paper_list = "\n".join([
        f"{p.get('author_names', 'Unknown Author')}, '{p.get('title', 'No Title')}', {p.get('venue', 'Unknown Venue')}, {p.get('year', 'n.d.')}"
        for p in papers
    ])
    # Blend user research context into the prompt if present
    user_context_str = ""
    if user_research_context and user_research_context.get('summary'):
        user_context_str = f"\nYou may also incorporate the following user-provided research findings into the narrative (do not cite as a reference, but blend naturally): {user_research_context['summary']}"
    system_prompt = (
        f"You are an expert academic writer. Write {n_paragraphs} paragraphs for the {section_name} section. "
        f"Use the following papers for citations:\n{paper_list}\n"
        f"Context: {context}{user_context_str}"
    )
    full_prompt = system_prompt
    try:
        response = model.generate_content(full_prompt, generation_config={"max_output_tokens": 100})
        content = response.text
        if not content:
            return [f"Error generating {section_name} content. Please try again."]
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        if len(paragraphs) < n_paragraphs:
            while len(paragraphs) < n_paragraphs:
                paragraphs.append(f"Additional {section_name.lower()} content would be developed here.")
        elif len(paragraphs) > n_paragraphs:
            paragraphs = paragraphs[:n_paragraphs]
        return paragraphs
    except Exception as e:
        print(f"Error generating {section_name} paragraphs: {e}")
        return [f"Error generating {section_name} content. Please try again."]

def generate_literature_review_section(papers, context, keywords, knowledge_graph=None, citation_map=None, user_research_context=None):
    if not papers:
        return ["The literature review would examine existing research in this domain."]
    themed_papers = group_papers_by_theme(papers, keywords)
    paragraphs = []
    # Blend user research context into the prompt if present
    user_context_str = ""
    if user_research_context and user_research_context.get('summary'):
        user_context_str = f"\nYou may also incorporate the following user-provided research findings into the narrative (do not cite as a reference, but blend naturally): {user_research_context['summary']}"
    intro_prompt = (
        "Write an introductory paragraph for a literature review section that sets up the research context. "
        f"Context: {context}{user_context_str}"
    )
    try:
        response = model.generate_content(intro_prompt, generation_config={"max_output_tokens": 100})
        content = response.text
        if content:
            paragraphs.append(content.strip())
        else:
            paragraphs.append("This literature review examines the current state of research in this domain.")
    except:
        paragraphs.append("This literature review examines the current state of research in this domain.")
    for theme, theme_papers in themed_papers.items():
        if not theme_papers:
            continue
        paper_list = "\n".join([
            f"{p.get('author_names', 'Unknown Author')}, '{p.get('title', 'No Title')}', {p.get('venue', 'Unknown Venue')}, {p.get('year', 'n.d.')}"
            for p in theme_papers
        ])
        theme_prompt = (
            f"Write a paragraph discussing research related to '{theme}'. "
            f"Use these papers: {paper_list}\n"
            f"Context: {context}{user_context_str}"
        )
        try:
            response = model.generate_content(theme_prompt, generation_config={"max_output_tokens": 100})
            content = response.text
            if content:
                paragraphs.append(content.strip())
            else:
                paragraphs.append(f"Research in {theme} has been explored by various authors in the field.")
        except:
            paragraphs.append(f"Research in {theme} has been explored by various authors in the field.")
    conclusion_prompt = (
        "Write a concluding paragraph for the literature review that summarizes key findings and identifies research gaps. "
        f"Context: {context}{user_context_str}"
    )
    try:
        response = model.generate_content(conclusion_prompt, generation_config={"max_output_tokens": 100})
        content = response.text
        if content:
            paragraphs.append(content.strip())
        else:
            paragraphs.append("This review highlights the current state of research and identifies areas for future investigation.")
    except:
        paragraphs.append("This review highlights the current state of research and identifies areas for future investigation.")
    return paragraphs
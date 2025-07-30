import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st

def build_knowledge_graph(domain, keywords, method, objective, summaries, draft_paragraph):
    #print(f"[DEBUG] Building knowledge graph with:\n  domain: {domain}\n  keywords: {keywords}\n  method: {method}\n  objective: {objective}\n  summaries: {summaries}\n  draft_paragraph: {draft_paragraph[:60]}...")
    G = nx.DiGraph()

    G.add_node('Prompt', type='prompt')
    if domain and len(domain.strip()) > 2:
        G.add_node(domain, type='domain')
        G.add_edge('Prompt', domain, relation='has_domain')
    if method and len(method.strip()) > 2:
        G.add_node(method, type='method')
        G.add_edge('Prompt', method, relation='has_method')
    if objective and len(objective.strip()) > 2:
        G.add_node(objective, type='objective')
        G.add_edge('Prompt', objective, relation='has_objective')
    G.add_node('DraftParagraph', type='draft')
    G.add_edge('Prompt', 'DraftParagraph', relation='generates')

    if keywords and isinstance(keywords, list):
        for kw in keywords[:5]:
            if isinstance(kw, str) and len(kw.strip()) > 2:
                G.add_node(kw, type='keyword')
                G.add_edge('Prompt', kw, relation='has_keyword')
    for i, s in enumerate(summaries):
        is_user = s.get('source') == 'user_research'
        paper_id = f"Paper_{i+1}"
        node_type = 'user_research' if is_user else 'paper'
        node_attrs = {
            'type': node_type,
            'title': s.get('title', ''),
            'citation': s.get('citation', ''),
            'summary': s.get('summary', ''),  
            'abstract': s.get('abstract', ''),
            'findings': s.get('findings', ''),
            'venue': s.get('venue', ''),
            'year': s.get('year', ''),
            'is_user_research': is_user
        }
        G.add_node(paper_id, **node_attrs)
        if is_user:
            G.add_edge('Prompt', paper_id, relation='contributed_by_user')
        else:
            G.add_edge('Prompt', paper_id, relation='cites')
        G.add_edge(paper_id, 'DraftParagraph', relation='supports')
        if keywords and isinstance(keywords, list):
            for kw in keywords[:5]:
                if isinstance(kw, str) and len(kw.strip()) > 2:
                    G.add_edge(kw, paper_id, relation='related_to')

        authors = s.get('author_names', 'Unknown Author')
        if authors and len(authors.strip()) > 2:
            G.add_node(authors, type='author')
            G.add_edge(paper_id, authors, relation='written_by')
    #print(f"[DEBUG] Knowledge graph nodes: {list(G.nodes(data=True))}")
    #print(f"[DEBUG] Knowledge graph edges: {list(G.edges(data=True))}")
    return G

def get_papers_for_keyword(G, keyword):
    """Return all papers related to a keyword node."""
    return list(G.successors(keyword))

def get_authors_for_paper(G, paper_id):
    """Return all authors for a paper node."""
    return [n for n in G.successors(paper_id) if G.nodes[n].get('type') == 'author']

def get_chain_prompt_to_draft(G):
    """Return the shortest path from Prompt to DraftParagraph."""
    try:
        return nx.shortest_path(G, 'Prompt', 'DraftParagraph')
    except nx.NetworkXNoPath:
        return []

def extract_paper_content(G):
    """Extract all paper content (summaries, abstracts) from the knowledge graph."""
    paper_content = []
    paper_nodes = [n for n, d in G.nodes(data=True) if d.get('type') in ['paper', 'user_research']]
    
    for node_id in paper_nodes:
        node_data = G.nodes[node_id]
        content_info = {
            'paper_id': node_id,
            'title': node_data.get('title', ''),
            'summary': node_data.get('summary', ''),
            'abstract': node_data.get('abstract', ''),
            'findings': node_data.get('findings', ''),
            'author': node_data.get('author_names', ''),
            'year': node_data.get('year', ''),
            'venue': node_data.get('venue', ''),
            'is_user_research': node_data.get('is_user_research', False)
        }
        paper_content.append(content_info)
    
    return paper_content

def get_research_themes_from_graph(G):
    """Extract research themes by analyzing paper content in the knowledge graph."""
    paper_content = extract_paper_content(G)
    all_text = []
    
    for paper in paper_content:
        text_parts = [
            paper.get('summary', ''),
            paper.get('abstract', ''),
            paper.get('findings', '')
        ]
        combined_text = ' '.join([t for t in text_parts if t])
        if combined_text.strip():
            all_text.append(combined_text)
    
    if not all_text:
        return []

    from collections import Counter
    all_words = []
    for text in all_text:
        words = [w.lower().strip('.,!?') for w in text.split() if len(w) > 4]
        all_words.extend(words)
    
    if all_words:
        word_freq = Counter(all_words)
        return [word for word, count in word_freq.most_common(10) if count > 1]
    
    return []

def show_graph(G):
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42, k=0.7)
    node_labels = {n: n if len(str(n)) < 25 else str(n)[:22] + '...' for n in G.nodes}
    nx.draw(G, pos, with_labels=True, labels=node_labels, node_color='lightblue', edge_color='gray', node_size=1600, font_size=10)
    edge_labels = nx.get_edge_attributes(G, 'relation')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=9)
    st.pyplot(plt.gcf())
    plt.clf()
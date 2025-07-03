import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st

def build_knowledge_graph(domain, keywords, method, objective, summaries, draft_paragraph):
    """Build a robust, clear knowledge graph using NetworkX."""
    print(f"[DEBUG] Building knowledge graph with:\n  domain: {domain}\n  keywords: {keywords}\n  method: {method}\n  objective: {objective}\n  summaries: {summaries}\n  draft_paragraph: {draft_paragraph[:60]}...")
    G = nx.DiGraph()
    # Add main nodes only if non-empty
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
    # Limit keywords to top 5 for clarity
    if keywords and isinstance(keywords, list):
        for kw in keywords[:5]:
            if isinstance(kw, str) and len(kw.strip()) > 2:
                G.add_node(kw, type='keyword')
                G.add_edge('Prompt', kw, relation='has_keyword')
    # Add papers and authors
    for i, s in enumerate(summaries):
        paper_id = f"Paper_{i+1}"
        G.add_node(paper_id, type='paper', title=s.get('title', ''), citation=s.get('citation', ''))
        G.add_edge('Prompt', paper_id, relation='cites')
        G.add_edge(paper_id, 'DraftParagraph', relation='supports')
        # Link keywords to papers
        if keywords and isinstance(keywords, list):
            for kw in keywords[:5]:
                if isinstance(kw, str) and len(kw.strip()) > 2:
                    G.add_edge(kw, paper_id, relation='related_to')
        # Use explicit author_names from summary dict, fallback to 'Unknown Author'
        authors = s.get('author_names', 'Unknown Author')
        if authors and len(authors.strip()) > 2:
            G.add_node(authors, type='author')
            G.add_edge(paper_id, authors, relation='written_by')
    print(f"[DEBUG] Knowledge graph nodes: {list(G.nodes(data=True))}")
    print(f"[DEBUG] Knowledge graph edges: {list(G.edges(data=True))}")
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

def show_graph(G):
    """Visualize the knowledge graph using matplotlib and Streamlit."""
    plt.figure(figsize=(14, 10))  # Larger figure for clarity
    pos = nx.spring_layout(G, seed=42, k=0.7)  # More spread out
    # Shorten long node labels for display
    node_labels = {n: n if len(str(n)) < 25 else str(n)[:22] + '...' for n in G.nodes}
    nx.draw(G, pos, with_labels=True, labels=node_labels, node_color='lightblue', edge_color='gray', node_size=1600, font_size=10)
    edge_labels = nx.get_edge_attributes(G, 'relation')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=9)
    st.pyplot(plt.gcf())
    plt.clf() 
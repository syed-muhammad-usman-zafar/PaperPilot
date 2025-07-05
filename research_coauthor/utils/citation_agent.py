import re
from typing import List, Dict

def get_year_from_citation(citation):
    """Extract the year from a citation string."""
    match = re.search(r", (\d{4})[.,)]", citation)
    if match:
        return match.group(1)
    return "n.d."

def get_first_author(citation):
    """Extract the first author from a citation string."""
    match = re.match(r"\[\d+\] ([^,]+)", citation)
    if match:
        return match.group(1)
    return "Unknown"

def citation_agent(paragraph, summaries):
    """Inject citations into a paragraph based on summaries."""
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

def calculate_citation_plan(keywords: List[str], domain: str) -> Dict[str, int]:
    """
    Dynamically calculate citation plan based on research complexity.
    
    Args:
        keywords: List of extracted keywords
        domain: Research domain (e.g., 'Computer Science', 'Biology')
    
    Returns:
        Dictionary mapping section names to number of citations needed
    """
    # Base plan
    base_plan = {
        "Abstract": 1,
        "Introduction": 2,
        "Literature Review": 5,
        "Methodology": 2,
        "Experiments / Results": 2,
        "Conclusion": 1
    }
    
    # Adjust based on number of keywords
    keyword_count = len(keywords)
    if keyword_count > 8:
        # Complex research - increase citations
        base_plan["Literature Review"] = min(8, base_plan["Literature Review"] + 2)
        base_plan["Introduction"] = min(4, base_plan["Introduction"] + 1)
    elif keyword_count < 4:
        # Simple research - decrease citations
        base_plan["Literature Review"] = max(3, base_plan["Literature Review"] - 1)
    
    # Adjust based on domain
    domain_lower = domain.lower()
    if any(term in domain_lower for term in ['computer science', 'cs', 'artificial intelligence', 'ai', 'machine learning']):
        # CS/AI domains typically need more citations
        base_plan["Literature Review"] = min(8, base_plan["Literature Review"] + 1)
        base_plan["Methodology"] = min(4, base_plan["Methodology"] + 1)
    elif any(term in domain_lower for term in ['biology', 'biomedical', 'medical', 'health']):
        # Biology/medical domains need extensive literature review
        base_plan["Literature Review"] = min(10, base_plan["Literature Review"] + 2)
        base_plan["Introduction"] = min(4, base_plan["Introduction"] + 1)
    
    return base_plan

def assign_papers_to_sections(summaries: List[Dict], plan: Dict[str, int]) -> Dict[str, List[Dict]]:
    """
    Evenly assign retrieved summaries to paper sections based on the citation plan.
    
    Args:
        summaries: List of paper summaries from literature retrieval
        plan: Citation plan dictionary
    
    Returns:
        Dictionary mapping section names to assigned paper summaries
    """
    total_needed = sum(plan.values())
    available_papers = len(summaries)
    
    if available_papers == 0:
        # Return empty assignments if no papers available
        return {section: [] for section in plan.keys()}
    
    if available_papers < total_needed:
        # Redistribute papers proportionally
        print(f"[WARNING] Only {available_papers} papers found, adjusting citations accordingly.")
        adjusted_plan = {}
        for section, count in plan.items():
            proportion = count / total_needed
            adjusted_plan[section] = max(1, int(proportion * available_papers))
        
        # Ensure we don't exceed available papers
        while sum(adjusted_plan.values()) > available_papers:
            # Reduce from sections with highest counts
            max_section = max(adjusted_plan.items(), key=lambda x: x[1])[0]
            adjusted_plan[max_section] = max(1, adjusted_plan[max_section] - 1)
        
        plan = adjusted_plan
    
    # Assign papers to sections
    section_assignments = {section: [] for section in plan.keys()}
    paper_index = 0
    
    for section, count in plan.items():
        for _ in range(count):
            if paper_index < len(summaries):
                section_assignments[section].append(summaries[paper_index])
                paper_index += 1
    
    return section_assignments 
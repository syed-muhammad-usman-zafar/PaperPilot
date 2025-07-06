import re
from typing import List, Dict

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

def calculate_citation_plan(keywords: List[str], method: str, objective: str) -> Dict[str, int]:
    """
    Use LLM-derived insights to dynamically calculate citation needs.
    
    Args:
        keywords: List of extracted keywords
        method: LLM-derived method type (empirical/theoretical/review/exploratory)
        objective: LLM-derived objective scope (exploratory/confirmatory/analytical/comparative)
    
    Returns:
        Dictionary mapping section names to number of citations needed
    """
    # Base citation plan
    base_plan = {
        "Abstract": 1,
        "Introduction": 3,
        "Literature Review": 6,
        "Methodology": 3,
        "Experiments / Results": 2,
        "Conclusion": 1
    }
    
    # Adjust based on number of keywords (complexity)
    keyword_count = len(keywords)
    if keyword_count > 8:
        # Complex research - increase citations
        base_plan["Literature Review"] = min(8, base_plan["Literature Review"] + 2)
        base_plan["Introduction"] = min(4, base_plan["Introduction"] + 1)
    elif keyword_count < 4:
        # Simple research - decrease citations
        base_plan["Literature Review"] = max(3, base_plan["Literature Review"] - 1)
    
    # Use LLM-derived method type
    method_lower = method.lower()
    if 'empirical' in method_lower or 'experiment' in method_lower or 'study' in method_lower:
        # Empirical work - needs more methodology citations
        base_plan["Methodology"] = min(4, base_plan["Methodology"] + 1)
        base_plan["Experiments / Results"] = min(3, base_plan["Experiments / Results"] + 1)
    elif 'review' in method_lower or 'survey' in method_lower or 'synthesis' in method_lower:
        # Review work - needs extensive literature review
        base_plan["Literature Review"] = min(9, base_plan["Literature Review"] + 2)
    elif 'theoretical' in method_lower or 'model' in method_lower or 'framework' in method_lower:
        # Theoretical work - needs more literature review
        base_plan["Literature Review"] = min(8, base_plan["Literature Review"] + 1)
    elif 'exploratory' in method_lower:
        # Exploratory work - needs broader literature review
        base_plan["Literature Review"] = min(8, base_plan["Literature Review"] + 1)
    
    # Use LLM-derived objective scope
    objective_lower = objective.lower()
    if 'exploratory' in objective_lower or 'explore' in objective_lower:
        # Exploratory research - needs broader literature review
        base_plan["Literature Review"] = min(8, base_plan["Literature Review"] + 1)
    elif 'confirmatory' in objective_lower or 'confirm' in objective_lower or 'test' in objective_lower:
        # Confirmatory research - needs more methodology citations
        base_plan["Methodology"] = min(4, base_plan["Methodology"] + 1)
    elif 'analytical' in objective_lower or 'analyze' in objective_lower:
        # Analytical research - balanced approach
        base_plan["Literature Review"] = min(8, base_plan["Literature Review"] + 1)
        base_plan["Methodology"] = min(4, base_plan["Methodology"] + 1)
    elif 'comparative' in objective_lower or 'compare' in objective_lower:
        # Comparative research - needs extensive literature review
        base_plan["Literature Review"] = min(9, base_plan["Literature Review"] + 2)
    
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
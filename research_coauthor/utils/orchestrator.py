from typing import List, Dict, Any
from .citation_agent import calculate_citation_plan, assign_papers_to_sections
from .writing_agent import generate_section_paragraphs, generate_literature_review_section

def generate_full_paper(prompt: str, llm_extracted: Dict[str, Any], 
                       summaries: List[Dict]) -> Dict[str, Any]:
    """
    Generate a complete academic paper with all sections.
    
    Args:
        prompt: Original user prompt
        llm_extracted: Extracted research elements from LLM
        summaries: Retrieved paper summaries
    
    Returns:
        Dictionary containing the complete paper structure
    """
    # Extract research elements
    domain = llm_extracted.get('domain', 'General Research')
    keywords = llm_extracted.get('key concepts') or llm_extracted.get('key_concepts', [])
    method = llm_extracted.get('research methods', ['analysis'])[0] if llm_extracted.get('research methods') else 'analysis'
    objective = llm_extracted.get('objectives', ['investigate'])[0] if llm_extracted.get('objectives') else 'investigate'
    
    # Ensure keywords is a list
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(',') if k.strip()]
    elif not isinstance(keywords, list):
        keywords = list(keywords)
    
    # Calculate citation plan using citation_agent
    citation_plan = calculate_citation_plan(keywords, domain)
    total_papers_needed = sum(citation_plan.values())
    
    # Assign papers to sections using citation_agent
    section_assignments = assign_papers_to_sections(summaries, citation_plan)
    
    # Build context string
    context = f"Domain: {domain}\nMethods: {method}\nObjectives: {objective}\nKey Concepts: {', '.join(keywords)}"
    
    # Generate sections using writing_agent
    paper_sections = {}
    
    # Abstract
    paper_sections["Abstract"] = generate_section_paragraphs(
        "Abstract", section_assignments["Abstract"], context, 1, "abstract"
    )
    
    # Introduction
    paper_sections["Introduction"] = generate_section_paragraphs(
        "Introduction", section_assignments["Introduction"], context, 2, "introduction"
    )
    
    # Literature Review
    paper_sections["Literature Review"] = generate_literature_review_section(
        section_assignments["Literature Review"], context, keywords
    )
    
    # Methodology
    paper_sections["Methodology"] = generate_section_paragraphs(
        "Methodology", section_assignments["Methodology"], context, 2, "methodology"
    )
    
    # Experiments / Results
    paper_sections["Experiments / Results"] = generate_section_paragraphs(
        "Experiments / Results", section_assignments["Experiments / Results"], context, 2, "results"
    )
    
    # Conclusion
    paper_sections["Conclusion"] = generate_section_paragraphs(
        "Conclusion", section_assignments["Conclusion"], context, 1, "conclusion"
    )
    
    return {
        "title": f"Research on {', '.join(keywords[:3])}",
        "sections": paper_sections,
        "citation_plan": citation_plan,
        "section_assignments": section_assignments,
        "total_papers_needed": total_papers_needed,
        "papers_found": len(summaries),
        "context": context
    } 
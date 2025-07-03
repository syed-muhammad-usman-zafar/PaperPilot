import re

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
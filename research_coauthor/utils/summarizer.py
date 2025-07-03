import re
from .citation_agent import get_first_author, get_year_from_citation

def truncate_at_sentence(text, max_len=300):
    """Truncate text at the nearest sentence boundary under max_len."""
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
    """Generate bullet-point summaries from Semantic Scholar results."""
    bullets = []
    for s in summaries:
        author = get_first_author(s['citation'])
        year = get_year_from_citation(s['citation'])
        summary = truncate_at_sentence(s['summary'], 300)
        bullets.append(f"- {author} et al. ({year}): {summary} [CITATION: {s['citation']}]")
    return bullets 
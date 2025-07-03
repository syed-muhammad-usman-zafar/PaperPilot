import openai
import os
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def writing_agent(context, papers, key_concepts):
    """Generate a draft paragraph using OpenAI LLM, given context, papers, and key concepts from the knowledge graph."""
    # Build a structured list of papers
    paper_list = "\n".join([
        f"[{i+1}] {p['author_names']}, '{p['title']}', {p.get('venue', 'Unknown Venue')}, {p.get('year', 'n.d.')}"
        for i, p in enumerate(papers)
    ])
    # Build a structured list of key concepts
    key_concepts_str = ", ".join(key_concepts)
    system_prompt = (
        "You are an expert research co-author. Write a short, human-like introduction paragraph for a research paper. "
        "You must cite ONLY from the following papers (use (Author, Year) style):\n"
        f"{paper_list}\n"
        "You must cover the following key concepts: "
        f"{key_concepts_str}. "
        "Do not invent citations, datasets, or references. Ensure at least one citation is included."
    )
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
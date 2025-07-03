import openai
import os
from dotenv import load_dotenv
import json

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_with_llm(prompt):
    """Extract research elements from a prompt using OpenAI LLM."""
    system_prompt = (
        "Extract the following as JSON: domain, research methods, objectives, data types, key concepts. "
        "Be concise and accurate."
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=300
    )
    content = response.choices[0].message.content
    if content is None:
        return {}
    try:
        return json.loads(content)
    except Exception:
        return {}

def draft_paragraph_with_llm(context, literature_bullets):
    """Generate a draft paragraph using OpenAI LLM, given context and literature bullets."""
    system_prompt = (
        "You are an expert research co-author. Write a short, human-like introduction paragraph for a research paper, "
        "synthesizing the following context and literature findings. "
        "Cite the provided literature in the paragraph using (Author, Year) style, and ensure at least one citation is included. "
        "Only use the provided literature bullets for citations and do not invent datasets, references, or sources."
    )
    user_content = f"Context: {context}\nKey findings:\n" + '\n'.join(literature_bullets)
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
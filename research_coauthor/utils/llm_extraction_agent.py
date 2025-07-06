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
        "Additionally, analyze the research context and derive: "
        "method_type (empirical/theoretical/review/exploratory) and "
        "objective_scope (exploratory/confirmatory/analytical/comparative). "
        "For vague prompts, infer these from keywords and context. "
        "Be concise and accurate."
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=400
    )
    content = response.choices[0].message.content
    print(f"[DEBUG] LLM extraction output: {content}")
    if content is None:
        return {}
    try:
        parsed = json.loads(content)
        print(f"[DEBUG] Parsed LLM extraction: {parsed}")
        return parsed
    except Exception as e:
        print(f"[DEBUG] LLM extraction JSON error: {e}")
        return {} 
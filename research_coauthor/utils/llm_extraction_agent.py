import os
import json
import re
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("OPENAI_API_KEY"))
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

def extract_with_llm(prompt):
    system_prompt = (
        "Extract the following as JSON: domain, research methods, objectives, data types, key concepts. "
        "Also infer: method_type (empirical/theoretical/review/exploratory), "
        "objective_scope (exploratory/confirmatory/analytical/comparative). "
        "Be concise."
    )
    full_prompt = f"{system_prompt}\nPrompt: {prompt}"
    try:
        response = model.generate_content(full_prompt, generation_config={"max_output_tokens": 200})
        content = response.text
        print(f"[DEBUG] Gemini extraction output: {content}")
        if not content:
            return {}
        # Remove markdown code block if present
        content = content.strip()
        if content.startswith("```json"):
            content = content.lstrip("`json").rstrip("`").strip()
        elif content.startswith("```"):
            content = content.lstrip("`").rstrip("`").strip()
        try:
            parsed = json.loads(content)
            print(f"[DEBUG] Parsed Gemini extraction: {parsed}")
            return parsed
        except Exception as e:
            print(f"[DEBUG] Gemini JSON error: {e}")
            # Fallback: try to extract keywords with regex
            keywords = []
            m = re.search(r'"key[_ ]?concepts"\s*:\s*\[(.*?)\]', content, re.DOTALL)
            if m:
                raw = m.group(1)
                keywords = [k.strip().strip('"\'') for k in raw.split(',') if k.strip()]
            print(f"[DEBUG] Fallback keywords: {keywords}")
            return {"key_concepts": keywords}
    except Exception as e:
        print(f"[DEBUG] Gemini failed: {e}")
        return {} 
import os
import json
import re
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("OPENAI_API_KEY"))
model = genai.GenerativeModel("models/gemini-2.5-flash")


def extract_with_llm(prompt):
    system_prompt = (
    "Extract the following structured JSON from the given prompt. The output must be a valid JSON object with these exact keys:\n"
    "- domain (string)\n"
    "- research methods (list of strings)\n"
    "- objectives (list of strings)\n"
    "- data types (list of strings)\n"
    "- key concepts (list of strings or comma-separated string)\n"
    "- method_type (string: one of 'empirical', 'theoretical', 'review', 'exploratory')\n"
    "- objective_scope (string: one of 'exploratory', 'confirmatory', 'analytical', 'comparative')\n"
    "- validation_requirements (list of strings. About requirements the resultant paper needs to have in accordance with the prompt and the extracted data)\n\n"
    "Output ONLY valid JSON. Do not include markdown formatting, explanations, or extra text."
    "Aim for the total Json tokens to be below 1000. and do not exceed this limit in any scenario"
    )

    full_prompt = f"{system_prompt}\nPrompt: {prompt}"
    try:
        response = model.generate_content(full_prompt, generation_config={"max_output_tokens": 1500, "temperature": 0.3})
        content = response.text
        print(f"[DEBUG] Gemini full response: {response}")
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
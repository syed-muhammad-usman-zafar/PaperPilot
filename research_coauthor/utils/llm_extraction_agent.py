import os
import json
import re
from dotenv import load_dotenv
from .model_config import generate_with_optimal_model, TaskType

load_dotenv()


def extract_with_llm(prompt):
    # More concise prompt to avoid token limits
    extraction_prompt = f"""Extract JSON from this research prompt:
"{prompt[:800]}"

Return only valid JSON:
{{"domain": "field", "key_concepts": ["term1", "term2"], "methods": ["method1"], "objectives": ["goal1"], "validation_requirements": ["requirement1","requirement2"]}}

Validation requirements is a list of strings about requirements the resultant paper needs to have in accordance with the prompt and the extracted data
Output ONLY valid JSON. Do not include markdown formatting, explanations, or extra text."""

    try:
        from .model_config import model_manager, TaskType, ModelType
        model = model_manager.models[ModelType.FAST]
        config = model_manager.get_config_for_task(TaskType.EXTRACTION)
     
        response = model.generate_content(extraction_prompt, generation_config=config)
        
        if not response or not response.text:
            return {"_error": "No response from LLM"}
            
        content = response.text.strip()
        print(f"[DEBUG] Extraction output: {content}")
        
        # Try to parse JSON
        try:
            # Clean the response - remove markdown formatting
            json_text = content
            if "```json" in json_text:
                json_text = json_text.split("```json")[1].split("```")[0]
            elif "```" in json_text:
                json_text = json_text.split("```")[1].split("```")[0]
            
            result = json.loads(json_text)
            
            # Ensure key_concepts exists and has valid keywords
            if 'key_concepts' not in result:
                result['key_concepts'] = result.get('key concepts', [])
            
            # If still no keywords, extract from domain and objectives
            if not result.get('key_concepts'):
                keywords = []
                domain = result.get('domain', '')
                objectives = result.get('objectives', [])
                
                # Extract keywords from domain
                if domain:
                    domain_words = re.findall(r'\b[A-Za-z]{4,}\b', domain)
                    keywords.extend(domain_words[:2])
                
                # Extract keywords from objectives
                for obj in objectives[:2]:
                    if isinstance(obj, str):
                        obj_words = re.findall(r'\b[A-Za-z]{4,}\b', obj)
                        keywords.extend(obj_words[:2])
                
                result['key_concepts'] = keywords[:5] if keywords else ['technology', 'analysis']
            
            print(f"[DEBUG] Parsed extraction: {result}")
            return result
            
        except json.JSONDecodeError as e:
            print(f"[DEBUG] JSON error: {e}")
            # Robust regex fallback
            fallback = {
                'domain': 'General Research',
                'key_concepts': [],
                'methods': ['analysis'],
                'objectives': ['investigate'],
                'validation_requirements': ['peer review', 'reproducibility']
            }
            
            # Extract domain
            domain_match = re.search(r'"domain":\s*"([^"]+)"', content)
            if domain_match:
                fallback['domain'] = domain_match.group(1)
            
            # Extract key concepts/keywords
            concepts = []
            concept_patterns = [
                r'"key_concepts":\s*\[(.*?)\]',
                r'"key concepts":\s*\[(.*?)\]',
                r'"keywords":\s*\[(.*?)\]'
            ]
            
            for pattern in concept_patterns:
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    concept_text = match.group(1)
                    # Extract quoted strings
                    concepts = re.findall(r'"([^"]+)"', concept_text)
                    break
            
            # If no concepts found, extract from visible text
            if not concepts:
                # Look for meaningful terms in the response
                words = re.findall(r'\b[A-Za-z]{4,}\b', content)
                # Filter out common words
                meaningful_words = [w for w in words if w.lower() not in ['analysis', 'research', 'study', 'data', 'method', 'objective']]
                concepts = meaningful_words[:5]
            
            fallback['key_concepts'] = concepts[:5] if concepts else ['technology', 'economics', 'business']
            
            print(f"[DEBUG] Fallback extraction: {fallback}")
            fallback['_error'] = "LLM output was not valid JSON. Used regex fallback."
            return fallback
            
        return {"_error": "Could not extract research elements from LLM output."}
    except Exception as e:
        print(f"[DEBUG] Extraction failed: {e}")
        # If quota or API error, return error info
        if "quota" in str(e).lower() or "429" in str(e):
            return {"_error": "The AI extraction service is temporarily unavailable due to usage limits. Please try again later."}
        return {"_error": f"Extraction failed: {e}"}
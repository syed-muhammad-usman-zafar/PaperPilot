"""
Multi-model configuration for optimal efficiency across different tasks
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai
from enum import Enum

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class ModelType(Enum):
    """Different model types optimized for different tasks"""
    FAST = "gemini-2.5-flash"           
    BALANCED = "gemini-1.5-flash-latest"
    POWERFUL = "gemini-1.5-pro-latest" 

class TaskType(Enum):
    """Different task types in our architecture"""
    EXTRACTION = "extraction"          
    ANALYSIS = "analysis"             
    GENERATION = "generation"         
    CHAT = "chat"                   


TASK_MODEL_MAP = {
    TaskType.EXTRACTION: ModelType.FAST,
    TaskType.ANALYSIS: ModelType.FAST,
    TaskType.GENERATION: ModelType.FAST,
    TaskType.CHAT: ModelType.FAST,
}

MODEL_CONFIGS = {
    ModelType.FAST: {
        "max_output_tokens": 1536,       
        "temperature": 0.1,              
        "top_p": 0.7,                   
    },
    ModelType.BALANCED: {
        "max_output_tokens": 2048,     
        "temperature": 0.3,              
        "top_p": 0.8,
    },
    ModelType.POWERFUL: {
        "max_output_tokens": 3072,      
        "temperature": 0.5,              
        "top_p": 0.9,
    }
}

class ModelManager:
  
    
    def __init__(self):
        self.models = {}
        self._initialize_models()
    
    def _initialize_models(self):
       
        for model_type in ModelType:
            self.models[model_type] = genai.GenerativeModel(model_type.value)
    
    def get_model_for_task(self, task_type: TaskType):
     
        model_type = TASK_MODEL_MAP.get(task_type, ModelType.BALANCED)
        return self.models[model_type]
    
    def get_config_for_task(self, task_type: TaskType):
        
        model_type = TASK_MODEL_MAP.get(task_type, ModelType.BALANCED)
        return MODEL_CONFIGS[model_type]
    
    def generate_content(self, task_type: TaskType, prompt: str, **kwargs):
      
        
        optimized_prompt = optimize_prompt_for_tokens(prompt, task_type)
        estimated_input_tokens = estimate_tokens(optimized_prompt)
        config = self.get_config_for_task(task_type)
        max_output = config.get('max_output_tokens', 512)
        
        print(f"[DEBUG] Using {TASK_MODEL_MAP[task_type].value} for {task_type.value}")
        print(f"[DEBUG] Estimated input tokens: {estimated_input_tokens}, max output: {max_output}")
        
        max_input_tokens = 32000 - max_output - 500 
        if estimated_input_tokens > max_input_tokens:
            optimized_prompt = truncate_content_smartly(optimized_prompt, max_input_tokens)
            print(f"[DEBUG] Truncated input to fit context window")
        
        model = self.get_model_for_task(task_type)
        
        
        generation_config = {**config, **kwargs}
        
        try:
            response = model.generate_content(optimized_prompt, generation_config=generation_config)
            
           
            if hasattr(response, 'usage_metadata'):
                print(f"[DEBUG] Actual tokens used: {response.usage_metadata}")
            
            return response
        except Exception as e:
            print(f"[DEBUG] {TASK_MODEL_MAP[task_type].value} failed for {task_type.value}: {e}")
           
            if task_type == TaskType.GENERATION:
                print("[DEBUG] Falling back to fast model for generation")
                fallback_model = self.models[ModelType.FAST]
                fallback_config = MODEL_CONFIGS[ModelType.FAST]
                fallback_prompt = truncate_content_smartly(optimized_prompt, 8000)
                return fallback_model.generate_content(fallback_prompt, generation_config=fallback_config)
            raise e


model_manager = ModelManager()

def get_model_for_task(task_type: TaskType):
    return model_manager.get_model_for_task(task_type)

def generate_with_optimal_model(task_type: TaskType, prompt: str, **kwargs):
    return model_manager.generate_content(task_type, prompt, **kwargs)

def optimize_prompt_for_tokens(prompt: str, task_type: TaskType) -> str:
    optimizations = {
        "please ": "",
        "could you ": "",
        "I would like you to ": "",
        "can you ": "",
        "research paper": "paper",
        "academic writing": "writing",
        "literature review": "lit review",
        "methodology section": "methods",
        "experimental results": "results",
        "in conclusion": "finally",
        " very ": " ",
        " really ": " ",
        " quite ": " ",
        " rather ": " ",
        "  ": " ",
        "\n\n\n": "\n\n",
    }
    
    optimized = prompt
    for old, new in optimizations.items():
        optimized = optimized.replace(old, new)

    if task_type == TaskType.EXTRACTION:
        if "Extract the following as JSON:" in optimized:
            optimized = optimized.replace(
                "Extract the following as JSON: domain, research methods, objectives, data types, key concepts. "
                "Also infer: method_type (empirical/theoretical/review/exploratory), "
                "objective_scope (exploratory/confirmatory/analytical/comparative). "
                "Be concise.",
                "JSON extract: domain, methods, objectives, key_concepts, method_type, objective_scope."
            )
    
    elif task_type == TaskType.GENERATION:
        optimized = optimized.replace(
            "You are an expert academic writer. Write a full research paper in plain text (not markdown, no # or ## headers). ",
            "Write research paper (plain text, no markdown). "
        )
        optimized = optimized.replace(
            "Do not skip any section. Do not use markdown or # headers. Separate each section with two newlines. Use double newlines between paragraphs. ",
            "Include all sections. No markdown. Double newlines between sections/paragraphs. "
        )
    
    return optimized.strip()

def estimate_tokens(text: str) -> int:
    return len(text) // 4

def truncate_content_smartly(content: str, max_tokens: int) -> str:
    estimated_tokens = estimate_tokens(content)
    
    if estimated_tokens <= max_tokens:
        return content
 
    target_chars = max_tokens * 4 * 0.9 
    
    if len(content) <= target_chars:
        return content
    
    sentences = content.split('. ')
    if len(sentences) > 1:
        truncated = ""
        for sentence in sentences:
            if len(truncated + sentence + '. ') <= target_chars:
                truncated += sentence + '. '
            else:
                break
        return truncated.strip()
    else:
        return content[:int(target_chars)-3] + "..."

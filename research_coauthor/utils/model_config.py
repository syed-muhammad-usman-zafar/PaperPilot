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
    FAST = "gemini-2.5-flash"           # Latest 2.5 flash - most advanced and efficient
    BALANCED = "gemini-1.5-flash-latest" # Flash latest for moderate tasks
    POWERFUL = "gemini-1.5-pro-latest"   # Latest pro version for complex tasks

class TaskType(Enum):
    """Different task types in our architecture"""
    EXTRACTION = "extraction"           # LLM extraction from prompts
    ANALYSIS = "analysis"              # Knowledge graph analysis
    GENERATION = "generation"          # Full paper generation
    CHAT = "chat"                      # Follow-up Q&A

# Optimal model mapping for each task
TASK_MODEL_MAP = {
    TaskType.EXTRACTION: ModelType.FAST,      # Simple JSON extraction
    TaskType.ANALYSIS: ModelType.FAST,        # Quick content analysis
    TaskType.GENERATION: ModelType.FAST,      # Full paper generation (free-tier only)
    TaskType.CHAT: ModelType.FAST,            # Quick responses
}

# Model configuration settings - Optimized for Gemini 2.0 Flash efficiency
MODEL_CONFIGS = {
    ModelType.FAST: {
        "max_output_tokens": 1536,       # Increased for 2.0 Flash better efficiency
        "temperature": 0.1,              # Lower for more focused, concise outputs
        "top_p": 0.7,                   # Reduced for more deterministic responses
    },
    ModelType.BALANCED: {
        "max_output_tokens": 2048,       # Moderate increase
        "temperature": 0.3,              # Lower for efficiency
        "top_p": 0.8,
    },
    ModelType.POWERFUL: {
        "max_output_tokens": 3072,       # Higher for complex tasks
        "temperature": 0.5,              # Lower for more focused generation
        "top_p": 0.9,
    }
}

class ModelManager:
    """Manages multiple Gemini models for different tasks"""
    
    def __init__(self):
        self.models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all model instances"""
        for model_type in ModelType:
            self.models[model_type] = genai.GenerativeModel(model_type.value)
    
    def get_model_for_task(self, task_type: TaskType):
        """Get the optimal model for a specific task"""
        model_type = TASK_MODEL_MAP.get(task_type, ModelType.BALANCED)
        return self.models[model_type]
    
    def get_config_for_task(self, task_type: TaskType):
        """Get the optimal configuration for a specific task"""
        model_type = TASK_MODEL_MAP.get(task_type, ModelType.BALANCED)
        return MODEL_CONFIGS[model_type]
    
    def generate_content(self, task_type: TaskType, prompt: str, **kwargs):
        """Generate content using the optimal model for the task with token optimization"""
        
        # Optimize prompt for token efficiency
        optimized_prompt = optimize_prompt_for_tokens(prompt, task_type)
        
        # Estimate and log token usage
        estimated_input_tokens = estimate_tokens(optimized_prompt)
        config = self.get_config_for_task(task_type)
        max_output = config.get('max_output_tokens', 512)
        
        print(f"[DEBUG] Using {TASK_MODEL_MAP[task_type].value} for {task_type.value}")
        print(f"[DEBUG] Estimated input tokens: {estimated_input_tokens}, max output: {max_output}")
        
        # Truncate input if too long (keep safety margin)
        max_input_tokens = 32000 - max_output - 500  # Safety margin
        if estimated_input_tokens > max_input_tokens:
            optimized_prompt = truncate_content_smartly(optimized_prompt, max_input_tokens)
            print(f"[DEBUG] Truncated input to fit context window")
        
        model = self.get_model_for_task(task_type)
        
        # Override config with any provided kwargs
        generation_config = {**config, **kwargs}
        
        try:
            response = model.generate_content(optimized_prompt, generation_config=generation_config)
            
            # Log actual token usage if available
            if hasattr(response, 'usage_metadata'):
                print(f"[DEBUG] Actual tokens used: {response.usage_metadata}")
            
            return response
        except Exception as e:
            print(f"[DEBUG] {TASK_MODEL_MAP[task_type].value} failed for {task_type.value}: {e}")
            # Fallback to fast model if powerful model fails
            if task_type == TaskType.GENERATION:
                print("[DEBUG] Falling back to fast model for generation")
                fallback_model = self.models[ModelType.FAST]
                fallback_config = MODEL_CONFIGS[ModelType.FAST]
                # Further reduce tokens for fallback
                fallback_prompt = truncate_content_smartly(optimized_prompt, 8000)
                return fallback_model.generate_content(fallback_prompt, generation_config=fallback_config)
            raise e

# Global model manager instance
model_manager = ModelManager()

def get_model_for_task(task_type: TaskType):
    """Convenience function to get model for task"""
    return model_manager.get_model_for_task(task_type)

def generate_with_optimal_model(task_type: TaskType, prompt: str, **kwargs):
    """Convenience function to generate content with optimal model"""
    return model_manager.generate_content(task_type, prompt, **kwargs)

def optimize_prompt_for_tokens(prompt: str, task_type: TaskType) -> str:
    """Optimize prompts to reduce token usage while maintaining effectiveness"""
    
    # Token-saving replacements
    optimizations = {
        # Remove redundant phrases
        "please ": "",
        "could you ": "",
        "I would like you to ": "",
        "can you ": "",
        
        # Shorten common academic phrases
        "research paper": "paper",
        "academic writing": "writing",
        "literature review": "lit review",
        "methodology section": "methods",
        "experimental results": "results",
        "in conclusion": "finally",
        
        # Remove filler words
        " very ": " ",
        " really ": " ",
        " quite ": " ",
        " rather ": " ",
        
        # Compress spacing
        "  ": " ",
        "\n\n\n": "\n\n",
    }
    
    optimized = prompt
    for old, new in optimizations.items():
        optimized = optimized.replace(old, new)
    
    # Task-specific optimizations
    if task_type == TaskType.EXTRACTION:
        # Ultra-concise extraction prompts
        if "Extract the following as JSON:" in optimized:
            optimized = optimized.replace(
                "Extract the following as JSON: domain, research methods, objectives, data types, key concepts. "
                "Also infer: method_type (empirical/theoretical/review/exploratory), "
                "objective_scope (exploratory/confirmatory/analytical/comparative). "
                "Be concise.",
                "JSON extract: domain, methods, objectives, key_concepts, method_type, objective_scope."
            )
    
    elif task_type == TaskType.GENERATION:
        # Remove verbose instructions for generation
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
    """Rough estimation of token count (1 token ≈ 4 characters for English)"""
    return len(text) // 4

def truncate_content_smartly(content: str, max_tokens: int) -> str:
    """Intelligently truncate content to fit token limits"""
    estimated_tokens = estimate_tokens(content)
    
    if estimated_tokens <= max_tokens:
        return content
    
    # Calculate target length (with safety margin)
    target_chars = max_tokens * 4 * 0.9  # 90% to be safe
    
    if len(content) <= target_chars:
        return content
    
    # Smart truncation strategies
    sentences = content.split('. ')
    if len(sentences) > 1:
        # Truncate by sentences to maintain coherence
        truncated = ""
        for sentence in sentences:
            if len(truncated + sentence + '. ') <= target_chars:
                truncated += sentence + '. '
            else:
                break
        return truncated.strip()
    else:
        # Simple truncation with ellipsis
        return content[:int(target_chars)-3] + "..."

"""
Token optimization configuration file
Easily adjust token limits and optimization settings
"""

# Token limits for different tasks (optimized for efficiency)
TOKEN_LIMITS = {
    "extraction": {
        "input_max": 600,      # ~150 tokens for user prompt
        "output_max": 150,     # Minimal JSON output
        "truncate_at": 150     # Truncate long prompts
    },
    
    "generation": {
        "input_max": 8000,     # ~2000 tokens for context + papers
        "output_max": 1536,    # Sufficient for full paper
        "papers_max": 8,       # Limit number of papers to include
        "paper_content_max": 200  # Max chars per paper summary
    },
    
    "chat": {
        "input_max": 500,      # ~125 tokens for context
        "output_max": 50,      # Short answers
        "context_max": 300,    # Max context chars
        "question_max": 100    # Max question length
    },
    
    "analysis": {
        "no_llm": True,        # Pure algorithmic analysis
        "max_themes": 5,       # Max themes to extract
        "min_word_length": 4   # Min word length for theme extraction
    }
}

# Cost optimization settings
COST_OPTIMIZATION = {
    "enable_caching": True,           # Cache repeated requests
    "smart_truncation": True,         # Intelligent content truncation
    "compress_prompts": True,         # Automatic prompt compression
    "fallback_on_error": True,        # Use cheaper models on fallback
    "limit_paper_count": True,        # Limit papers to reduce context
    "minimal_context": True           # Use minimal context for tasks
}

# Prompt compression rules
PROMPT_OPTIMIZATIONS = {
    # Remove redundant phrases
    "remove_phrases": [
        "please ", "could you ", "I would like you to ", "can you "
    ],
    
    # Shorten academic terms
    "academic_shortcuts": {
        "research paper": "paper",
        "academic writing": "writing", 
        "literature review": "lit review",
        "methodology section": "methods",
        "experimental results": "results",
        "in conclusion": "finally"
    },
    
    # Remove filler words
    "remove_fillers": [
        " very ", " really ", " quite ", " rather "
    ],
    
    # Task-specific optimizations
    "task_specific": {
        "extraction": {
            "before": "Extract the following as JSON: domain, research methods, objectives, data types, key concepts. Also infer: method_type (empirical/theoretical/review/exploratory), objective_scope (exploratory/confirmatory/analytical/comparative). Be concise.",
            "after": "JSON extract: domain, methods, objectives, key_concepts, method_type, objective_scope."
        },
        "generation": {
            "before": "You are an expert academic writer. Write a full research paper in plain text (not markdown, no # or ## headers).",
            "after": "Write research paper (plain text, no markdown)."
        }
    }
}

# Model efficiency settings
MODEL_EFFICIENCY = {
    "fast_model": {
        "temperature": 0.1,      # More focused outputs
        "top_p": 0.7,           # Reduced randomness
        "max_tokens": 256       # Conservative limit
    },
    
    "powerful_model": {
        "temperature": 0.5,      # Balanced creativity
        "top_p": 0.9,           # Good diversity
        "max_tokens": 1536      # Sufficient for complex tasks
    }
}

# Monitoring and alerting
MONITORING = {
    "log_token_usage": True,         # Track token consumption
    "warn_on_high_usage": True,      # Alert when usage is high
    "token_budget_daily": 50000,     # Daily token budget
    "alert_threshold": 0.8           # Alert at 80% of budget
}

def get_token_limit(task_type: str, limit_type: str) -> int:
    """Get token limit for specific task and limit type"""
    return TOKEN_LIMITS.get(task_type, {}).get(limit_type, 512)

def should_optimize(optimization_type: str) -> bool:
    """Check if specific optimization should be applied"""
    return COST_OPTIMIZATION.get(optimization_type, True)

def get_model_config(model_type: str) -> dict:
    """Get optimized configuration for model type"""
    return MODEL_EFFICIENCY.get(model_type, MODEL_EFFICIENCY["fast_model"])

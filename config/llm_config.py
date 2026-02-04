"""
LLM API Configuration
Loads API keys and configuration for various LLM providers.
"""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# Google Gemini API
# =============================================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", None)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# =============================================================================
# Groq Cloud API
# =============================================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)
GROQ_API_BASE_URL = os.getenv("GROQ_API_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# =============================================================================
# Together AI API
# =============================================================================
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", None)
TOGETHER_API_BASE_URL = os.getenv("TOGETHER_API_BASE_URL", "https://api.together.xyz/v1")
TOGETHER_MODEL = os.getenv("TOGETHER_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")

# =============================================================================
# Hugging Face API
# =============================================================================
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", None)
HUGGINGFACE_API_BASE_URL = os.getenv("HUGGINGFACE_API_BASE_URL", "https://api-inference.huggingface.co")
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct")

# =============================================================================
# WaveSpeed API
# =============================================================================
WAVESPEED_API_KEY = os.getenv("WAVESPEED_API_KEY", None)
WAVESPEED_API_BASE_URL = os.getenv("WAVESPEED_API_BASE_URL", "https://api.wavespeed.ai/v1")
WAVESPEED_MODEL = os.getenv("WAVESPEED_MODEL", "default")

# =============================================================================
# Default Provider Selection
# =============================================================================
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "gemini")
DEFAULT_EMBEDDING_PROVIDER = os.getenv("DEFAULT_EMBEDDING_PROVIDER", "nomic")

# =============================================================================
# Rate Limiting & Retry Configuration
# =============================================================================
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY_SECONDS = int(os.getenv("RETRY_DELAY_SECONDS", "20"))
RATE_LIMIT_REQUESTS_PER_MINUTE = int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "6"))

# =============================================================================
# Helper Functions
# =============================================================================

def get_api_key(provider: str) -> Optional[str]:
    """
    Get API key for a specific provider.
    
    Args:
        provider: One of 'gemini', 'groq', 'together', 'huggingface', 'wavespeed'
    
    Returns:
        API key string or None if not set
    """
    provider_map = {
        'gemini': GOOGLE_API_KEY,
        'groq': GROQ_API_KEY,
        'together': TOGETHER_API_KEY,
        'huggingface': HUGGINGFACE_API_KEY,
        'wavespeed': WAVESPEED_API_KEY,
    }
    return provider_map.get(provider.lower())

def is_provider_available(provider: str) -> bool:
    """
    Check if a provider is configured and available.
    
    Args:
        provider: One of 'gemini', 'groq', 'together', 'huggingface', 'wavespeed'
    
    Returns:
        True if API key is set, False otherwise
    """
    api_key = get_api_key(provider)
    return api_key is not None and api_key != "your_gemini_api_key_here"

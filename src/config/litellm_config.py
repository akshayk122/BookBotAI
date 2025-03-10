"""
Gemini AI configuration for the application.
"""
import os
import logging
from typing import Dict, Any
import google.generativeai as genai

# Configure Gemini
def configure_gemini():
    """
    Configure Gemini AI with the necessary settings.
    """
    # Set API key from environment variable or use the provided one
    api_key = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY")
    genai.configure(api_key=api_key)
    
    # Configure logging to minimize error messages
    logging.basicConfig(level=logging.ERROR)
    
    # Set default generation parameters for better results
    generation_config = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
    }
    
    # Return configuration for reference
    return {
        "api_key": "[REDACTED]",  # Don't expose the actual key in logs
        "model": "gemini-2.0-flash",
        "generation_config": generation_config
    }

# Alias for backward compatibility
def configure_litellm():
    """
    Alias for configure_gemini for backward compatibility.
    """
    return configure_gemini()

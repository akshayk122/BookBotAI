"""
LLM wrapper for Gemini AI integration.
"""
import os
from typing import Dict, Any, Optional, List
import google.generativeai as genai
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

class GeminiLLM(LLM):
    """
    LangChain wrapper for Google's Gemini Pro model.
    """
    model_name: str = "gemini-2.0-flash"
    temperature: float = 1.0
    max_output_tokens: int = 8192
    top_p: float = 0.95
    top_k: int = 40
    
    def __init__(self, **kwargs):
        """Initialize the Gemini LLM wrapper."""
        super().__init__(**kwargs)
        
        # Ensure Gemini API key is available
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        # Configure the Gemini API
        genai.configure(api_key=api_key)
    
    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "gemini"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[list] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        """
        Call the Gemini model with the given prompt.
        
        Args:
            prompt: The prompt to send to the model
            stop: Optional list of stop sequences
            run_manager: Optional callback manager
            
        Returns:
            The generated text
        """
        # Set up generation config
        generation_config = {
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "top_k": kwargs.get("top_k", self.top_k),
            "max_output_tokens": kwargs.get("max_output_tokens", self.max_output_tokens),
        }
        
        try:
            # Get the model
            model = genai.GenerativeModel(model_name=self.model_name,
                                         generation_config=generation_config)
            
            # Generate content
            response = model.generate_content(prompt)
            
            # Return the text
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'parts'):
                return ''.join(part.text for part in response.parts if hasattr(part, 'text'))
            else:
                return str(response)
        except Exception as e:
            # Handle errors
            error_msg = f"Error calling Gemini API: {str(e)}"
            print(error_msg)
            return error_msg

def get_gemini_llm(**kwargs) -> GeminiLLM:
    """
    Get a Gemini LLM instance with the specified parameters.
    
    Args:
        **kwargs: Additional parameters to pass to the LLM
        
    Returns:
        A configured GeminiLLM instance
    """
    return GeminiLLM(**kwargs)

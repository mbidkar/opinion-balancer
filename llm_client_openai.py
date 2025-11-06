"""
OpenAI LLM Client for OpinionBalancer
Uses OpenAI's API with GPT models for text generation
"""

import os
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class OpenAILLMClient:
    """
    OpenAI API client for OpinionBalancer
    Supports GPT-4, GPT-3.5-turbo, and other OpenAI models
    """
    
    def __init__(
        self, 
        model: str = "gpt-5",
        api_key: Optional[str] = None,
        max_completion_tokens: int = 1000,
        temperature: float = 0.7
    ):
        """
        Initialize OpenAI client
        
        Args:
            model: OpenAI model name (gpt-5, gpt-4, gpt-3.5-turbo, etc.)
            api_key: OpenAI API key (reads from OPENAI_API_KEY env var if None)
            max_completion_tokens: Maximum tokens to generate (GPT-5 uses max_completion_tokens)
            temperature: Sampling temperature (0.0 to 1.0)
        """
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        print(f"✅ OpenAI client initialized with model: {self.model}")
    
    def generate(self, prompt: str, system_message: str = None) -> str:
        """
        Generate text using OpenAI API
        
        Args:
            prompt: Input text prompt
            system_message: Optional system message for context
            
        Returns:
            Generated text response
        """
        try:
            # Prepare messages
            messages = []
            
            if system_message:
                messages.append({"role": "system", "content": system_message})
            
            messages.append({"role": "user", "content": prompt})
            
            # Make API call with GPT-5 compatible parameters
            api_params = {
                "model": self.model,
                "messages": messages,
                "max_completion_tokens": self.max_completion_tokens,
            }
            
            # Only add temperature for models that support it (not GPT-5)
            if not self.model.startswith("gpt-5"):
                api_params["temperature"] = self.temperature
            
            response = self.client.chat.completions.create(**api_params)
            
            # Extract generated text
            generated_text = response.choices[0].message.content
            
            return generated_text.strip()
            
        except Exception as e:
            print(f"❌ OpenAI API error: {e}")
            return f"Error generating response: {e}"
    
    def test_connection(self) -> bool:
        """
        Test OpenAI API connectivity
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            test_response = self.generate("Say 'Hello, OpinionBalancer!' to confirm the connection.")
            return "OpinionBalancer" in test_response or "Hello" in test_response
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False
    
    def get_model_info(self) -> dict:
        """
        Get information about the current model
        
        Returns:
            Dictionary with model information
        """
        return {
            "model": self.model,
            "max_completion_tokens": self.max_completion_tokens,
            "temperature": self.temperature,
            "api_key_set": bool(self.api_key)
        }

# Convenience function for backward compatibility
def create_llm_client(model: str = "gpt-5") -> OpenAILLMClient:
    """Create and return an OpenAI LLM client"""
    return OpenAILLMClient(model=model)
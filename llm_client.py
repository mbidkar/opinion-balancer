"""
LLM Client for OpinionBalancer
Supports both Ollama and GPT-2 backends
"""

import yaml
from typing import Dict, Any, Optional


def make_llm_client(config_path: str = "config.yaml"):
    """
    Create LLM client based on configuration
    
    Args:
        config_path: Path to config file
        
    Returns:
        LLMClient instance (Ollama or GPT-2)
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    provider = config.get('models', {}).get('provider', 'ollama')
    
    if provider == 'gpt2':
        from llm_client_gpt2 import GPT2LLMClient
        model_path = config['models'].get('model_path')
        device = config['models'].get('device', 'auto')
        return GPT2LLMClient(model_path=model_path, device=device)
    else:
        # Fallback to Ollama client
        return OllamaLLMClient(config_path)


class OllamaLLMClient:
    """Ollama client using OpenAI-compatible interface"""
    
    def __init__(self, config_path: str = "config.yaml"):
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
        
        self.ChatOpenAI = ChatOpenAI
        self.HumanMessage = HumanMessage
        self.SystemMessage = SystemMessage
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.models_config = self.config['models']
        self._clients = {}
    
    def _get_client(self, role: str, temperature: Optional[float] = None):
        """Get or create LLM client for specific role"""
        if temperature is None:
            temperature = self.models_config['temperatures'].get(role, 0.5)
        
        client_key = f"{role}_{temperature}"
        
        if client_key not in self._clients:
            self._clients[client_key] = self.ChatOpenAI(
                base_url=self.models_config['base_url'],
                model=self.models_config['model'],
                temperature=temperature,
                max_tokens=self.models_config['max_tokens']
            )
        
        return self._clients[client_key]
    
    def generate(self, 
                prompt: str, 
                role: str = "writer",
                system_message: Optional[str] = None,
                temperature: Optional[float] = None) -> str:
        """Generate text using Ollama"""
        client = self._get_client(role, temperature)
        
        messages = []
        if system_message:
            messages.append(self.SystemMessage(content=system_message))
        messages.append(self.HumanMessage(content=prompt))
        
        try:
            response = client.invoke(messages)
            content = response.content
            
            # Clean up the response - remove end marker if present
            if "### END" in content:
                content = content.split("### END")[0].strip()
            
            return content
            
        except Exception as e:
            print(f"Error generating with {role} client: {e}")
            return f"Error: Could not generate response. Check Ollama connection."
    
    def test_connection(self) -> bool:
        """Test if Ollama is responding"""
        try:
            response = self.generate(
                "Hello! This is a connection test.", 
                role="writer"
            )
            return len(response) > 0 and "Error:" not in response
        except:
            return False


# Legacy functions - now handled by the factory function above

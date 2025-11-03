"""
LLM Client for OpinionBalancer
Ollama integration using OpenAI-compatible interface
"""

import yaml
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


class LLMClient:
    """Ollama client using OpenAI-compatible interface"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.models_config = self.config['models']
        self._clients = {}
    
    def _get_client(self, role: str, temperature: Optional[float] = None) -> ChatOpenAI:
        """Get or create LLM client for specific role"""
        if temperature is None:
            temperature = self.models_config['temperatures'].get(role, 0.5)
        
        client_key = f"{role}_{temperature}"
        
        if client_key not in self._clients:
            self._clients[client_key] = ChatOpenAI(
                base_url=self.models_config['base_url'],
                api_key="ollama",  # dummy key for local Ollama
                model=self.models_config['model'],
                temperature=temperature,
                max_tokens=self.models_config.get('max_tokens', 2048),
                stop=["### END"]  # Stop at end marker
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
            messages.append(SystemMessage(content=system_message))
        messages.append(HumanMessage(content=prompt))
        
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


def make_llm_client(config_path: str = "config.yaml") -> LLMClient:
    """Factory function to create LLM client"""
    return LLMClient(config_path)


# For backward compatibility with blueprint examples
def make_llm(role: str, temperature: float, config_path: str = "config.yaml") -> ChatOpenAI:
    """Create individual LLM client (legacy interface)"""
    client = LLMClient(config_path)
    return client._get_client(role, temperature)

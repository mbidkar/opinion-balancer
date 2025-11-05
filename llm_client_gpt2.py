"""
GPT-2 LLM Client for PACE-ICE
Local Hugging Face Transformers implementation
"""

import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Optional, Dict, Any
import warnings

# Suppress some transformer warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


class GPT2LLMClient:
    """Local GPT-2 client using Hugging Face Transformers"""
    
    def __init__(self, model_path: str = None, device: str = "auto"):
        """
        Initialize GPT-2 client
        
        Args:
            model_path: Path to local GPT-2 model on PACE-ICE
            device: Device to run on ('cuda', 'cpu', or 'auto')
        """
        self.model_path = model_path or "/storage/data/mod-huggingface-0/openai-community__gpt2-medium/models--openai-community--gpt2-medium/snapshots/6dcaa7a952f72f9298047fd5137cd6e4f05f41da"
        
        # Auto-detect device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the GPT-2 model and tokenizer"""
        try:
            print(f"🔄 Loading GPT-2 model from: {self.model_path}")
            print(f"🖥️  Device: {self.device}")
            
            # Load tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)
            
            # Add pad token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = GPT2LMHeadModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
                
            self.model.eval()  # Set to evaluation mode
            
            print("✅ GPT-2 model loaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to load GPT-2 model: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test if the model is loaded and working"""
        try:
            if self.model is None or self.tokenizer is None:
                return False
            
            # Simple test generation
            test_response = self.generate("Hello", max_length=20, temperature=0.7)
            return len(test_response.strip()) > 0
            
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False
    
    def generate(
        self, 
        prompt: str, 
        role: str = "assistant",
        system_message: str = None,
        max_length: int = 500,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True
    ) -> str:
        """
        Generate text using GPT-2
        
        Args:
            prompt: Input prompt
            role: Role for generation (for compatibility, not used in GPT-2)
            system_message: System message (for compatibility, not used in GPT-2) 
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling
            do_sample: Whether to use sampling
            
        Returns:
            Generated text
        """
        try:
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("Model not loaded")
            
            # Combine system message and prompt if provided
            full_prompt = prompt
            if system_message:
                full_prompt = f"{system_message}\n\n{prompt}"
            
            # Tokenize input
            inputs = self.tokenizer.encode(
                full_prompt, 
                return_tensors="pt", 
                max_length=1024, 
                truncation=True
            )
            inputs = inputs.to(self.device)
            
            # Calculate max new tokens (total length - input length)
            input_length = inputs.shape[1]
            max_new_tokens = min(max_length, 1024 - input_length)
            
            if max_new_tokens <= 0:
                return "Error: Input too long"
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode only the generated part (exclude input)
            generated_tokens = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return generated_text.strip()
            
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return f"Generation failed: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "model_type": "GPT-2 Medium",
            "parameters": "355M",
            "loaded": self.model is not None and self.tokenizer is not None
        }


# Global client instance
_gpt2_client = None


def make_llm_client(model_path: str = None) -> GPT2LLMClient:
    """
    Create or get global GPT-2 LLM client instance
    
    Args:
        model_path: Optional custom model path
        
    Returns:
        GPT2LLMClient instance
    """
    global _gpt2_client
    
    if _gpt2_client is None:
        _gpt2_client = GPT2LLMClient(model_path=model_path)
    
    return _gpt2_client


def test_gpt2_client():
    """Test the GPT-2 client"""
    print("🧪 Testing GPT-2 Client")
    print("=" * 40)
    
    try:
        client = make_llm_client()
        
        # Test connection
        if not client.test_connection():
            print("❌ Connection test failed")
            return False
        
        print("✅ Connection test passed")
        
        # Test generation
        test_prompt = "The benefits of remote work include"
        response = client.generate(test_prompt, max_length=100)
        
        print(f"\n📝 Test Generation:")
        print(f"Prompt: {test_prompt}")
        print(f"Response: {response}")
        
        # Model info
        info = client.get_model_info()
        print(f"\n🤖 Model Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    test_gpt2_client()

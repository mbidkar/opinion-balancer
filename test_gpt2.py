#!/usr/bin/env python3
"""
Test script for GPT-2 LLM Client
Tests basic functionality of the GPT2LLMClient
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_client_gpt2 import GPT2LLMClient

def test_gpt2_client():
    """Test the GPT-2 client with a simple prompt"""
    print("🔧 Testing GPT-2 LLM Client...")
    
    try:
        # Initialize the client
        print("📦 Initializing GPT-2 client...")
        client = GPT2LLMClient()
        print("✅ Client initialized successfully")
        
        # Test prompt
        test_prompt = "The future of artificial intelligence is"
        
        print(f"\n📝 Testing with prompt: '{test_prompt}'")
        print("⏳ Generating response...")
        
        # Generate response
        response = client.generate(test_prompt)
        
        print(f"\n🤖 GPT-2 Response:")
        print("-" * 50)
        print(response)
        print("-" * 50)
        
        # Verify response
        if response and len(response.strip()) > len(test_prompt):
            print("\n✅ Test PASSED: GPT-2 client is working correctly!")
            return True
        else:
            print("\n❌ Test FAILED: Response is empty or too short")
            return False
            
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure transformers and torch are installed:")
        print("   pip install transformers torch")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_device_detection():
    """Test device detection (CPU/GPU)"""
    print("\n🖥️  Testing device detection...")
    
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔍 Detected device: {device}")
        
        if torch.cuda.is_available():
            print(f"🎮 GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print("💻 Running on CPU")
            
    except ImportError:
        print("❌ PyTorch not available")

if __name__ == "__main__":
    print("🧪 GPT-2 Client Test Suite")
    print("=" * 50)
    
    # Test device detection
    test_device_detection()
    
    # Test GPT-2 client
    success = test_gpt2_client()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! GPT-2 client is ready to use.")
    else:
        print("❌ Tests failed. Check the error messages above.")
    
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Test script for OpenAI integration in OpinionBalancer
Tests the new OpenAI client and verifies API connectivity
"""

import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_client_openai import OpenAILLMClient

def test_openai_connection():
    """Test basic OpenAI API connectivity"""
    print("🔧 Testing OpenAI API Connection...")
    print("=" * 50)
    
    try:
        # Check if API key is set
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ OPENAI_API_KEY not found in environment")
            return False
        
        print(f"✅ API key found: {api_key[:10]}...{api_key[-6:]}")
        
        # Initialize client
        print("\n📦 Initializing OpenAI client...")
        client = OpenAILLMClient(model="gpt-5", temperature=0.7, max_completion_tokens=1000)
        
        # Test connection
        print("🔗 Testing API connection...")
        if client.test_connection():
            print("✅ OpenAI API connection successful!")
        else:
            print("❌ OpenAI API connection failed")
            return False
        
        # Test basic generation
        print("\n📝 Testing text generation...")
        test_prompt = "Write a brief, balanced opinion about renewable energy in exactly 2 sentences."
        
        response = client.generate(
            prompt=test_prompt,
            system_message="You are a balanced opinion writer."
        )
        
        print(f"\n🤖 OpenAI Response:")
        print("-" * 40)
        print(response)
        print("-" * 40)
        
        # Verify response quality
        if response and len(response.strip()) > 50:
            print("\n✅ Text generation test PASSED!")
            return True
        else:
            print("\n❌ Text generation test FAILED - response too short")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def test_model_variants():
    """Test different OpenAI model variants"""
    print("\n🎛️ Testing Model Variants...")
    print("=" * 50)
    
    models_to_test = [
        ("gpt-5", "Latest GPT-5 model"),
        ("gpt-4", "GPT-4 fallback model"),
        ("gpt-3.5-turbo", "GPT-3.5 fallback model"),
    ]
    
    for model, description in models_to_test:
        print(f"\n🧪 Testing {model} - {description}")
        try:
            client = OpenAILLMClient(model=model, temperature=0.5, max_completion_tokens=100)
            
            response = client.generate(
                "What is artificial intelligence?",
                system_message="Answer briefly and clearly."
            )
            
            if response:
                print(f"✅ {model}: Working")
                print(f"   Response length: {len(response)} chars")
            else:
                print(f"❌ {model}: No response")
                
        except Exception as e:
            print(f"❌ {model}: Error - {e}")

def test_opinion_balancer_integration():
    """Test integration with OpinionBalancer workflow"""
    print("\n🏗️ Testing OpinionBalancer Integration...")
    print("=" * 50)
    
    try:
        # Test draft writing simulation
        print("✍️ Testing draft writing simulation...")
        
        client = OpenAILLMClient(model="gpt-5", temperature=0.8, max_completion_tokens=800)
        
        draft_prompt = """Write a balanced 150-word opinion piece on the topic: "Remote work policies"

Requirements:
- Present both benefits and drawbacks
- Target audience: general readers
- Professional tone
- Include multiple perspectives"""
        
        draft = client.generate(
            prompt=draft_prompt,
            system_message="You are an expert opinion writer focused on balanced, fair reporting."
        )
        
        print(f"\n📄 Generated Draft ({len(draft)} chars):")
        print("-" * 40)
        print(draft)
        print("-" * 40)
        
        # Test editing simulation
        print("\n✏️ Testing editing simulation...")
        
        edit_prompt = f"""Improve this draft by making it more concise and adding a stronger conclusion:

DRAFT:
{draft}

INSTRUCTIONS:
- Reduce length to ~100 words
- Strengthen the conclusion
- Maintain balance"""
        
        edited = client.generate(
            prompt=edit_prompt,
            system_message="You are an expert editor focused on applying specific edits while maintaining quality and balance."
        )
        
        print(f"\n📝 Edited Version ({len(edited)} chars):")
        print("-" * 40)
        print(edited)
        print("-" * 40)
        
        print("\n✅ OpinionBalancer integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 OpenAI Integration Test Suite")
    print("=" * 60)
    
    # Run all tests
    connection_test = test_openai_connection()
    
    if connection_test:
        test_model_variants()
        integration_test = test_opinion_balancer_integration()
        
        print("\n" + "=" * 60)
        if connection_test and integration_test:
            print("🎉 All tests passed! OpenAI integration is ready.")
            sys.exit(0)
        else:
            print("❌ Some tests failed. Check the errors above.")
            sys.exit(1)
    else:
        print("\n❌ Basic connection test failed. Fix API connectivity first.")
        sys.exit(1)
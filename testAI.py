import requests
import json

def test_ollama():
    """Test if Ollama is running and can respond to requests"""
    try:
        # Test if Ollama is running
        print("🤖 Testing Ollama AI")
        print("=" * 40)
        
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json()
            print("✅ Ollama is running!")
            print("Available models:")
            for model in models.get('models', []):
                print(f"  - {model['name']}")
            
            # Try to generate text with the first available model
            if models.get('models'):
                model_name = models['models'][0]['name']
                print(f"\n🧠 Testing with model: {model_name}")
                
                payload = {
                    "model": model_name,
                    "prompt": "Hello from my LangGraph project! Please respond in a friendly way.",
                    "stream": False
                }
                
                print("Generating response...")
                response = requests.post("http://localhost:11434/api/generate", 
                                       json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ Response: {result['response']}")
                    print("\n🎉 Success! Your AI setup is working perfectly!")
                else:
                    print(f"❌ Error generating text: {response.status_code}")
            else:
                print("❌ No models available. Install a model with: ollama pull llama3.2:1b")
        else:
            print("❌ Ollama is not responding properly.")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama.")
        print("Make sure Ollama is running with: ollama serve")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_ollama()
import requests
import json

def test_huggingface_api():
    """Test using Hugging Face's free inference API"""
    
    # Using a free model from Hugging Face
    API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
    
    headers = {
        "Content-Type": "application/json",
    }
    
    payload = {
        "inputs": "Hello from my LangGraph project!"
    }
    
    try:
        print("Testing Hugging Face Inference API...")
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                print(f"Response: {result[0].get('generated_text', 'No response generated')}")
            else:
                print(f"Unexpected response format: {result}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Error connecting to Hugging Face API: {e}")

if __name__ == "__main__":
    test_huggingface_api()

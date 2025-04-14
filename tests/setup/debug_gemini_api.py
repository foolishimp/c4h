#!/usr/bin/env python3
# debug_gemini_api.py - Direct test of Gemini API without LiteLLM

import os
import json
import requests
import sys
from urllib.parse import urljoin

def test_gemini_direct():
    """Test Gemini API access directly without LiteLLM"""
    # Get API key from environment
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set")
        return False
    
    # Base URL for Gemini API
    base_urls = [
        "https://generativelanguage.googleapis.com/v1beta",
        "https://generativelanguage.googleapis.com/v1beta/models",
        "https://generativelanguage.googleapis.com/v1",
        "https://generativelanguage.googleapis.com/v1/models",
    ]
    
    # Model names to try
    model_names = [
        "gemini-pro",
        "gemini-1.5-pro-latest",
        "gemini-pro-2.5",
        "gemini-2.5-pro",
        "gemini-1.5-flash"
    ]
    
    # Content for testing
    content = "Summarize the key features of Gemini 2.5 in 2-3 bullet points."
    
    # Try each base URL and model combination
    success = False
    for base_url in base_urls:
        for model_name in model_names:
            try:
                # Print attempt info
                print(f"\nTrying: {base_url} with model {model_name}")
                
                # Model URL formats to try
                model_urls = [
                    # Direct generation endpoint with model parameter
                    f"{base_url}/generateContent?key={api_key}",
                    # Model-specific endpoint
                    f"{base_url}/{model_name}:generateContent?key={api_key}",
                ]
                
                for model_url in model_urls:
                    try:
                        print(f"  URL: {model_url.replace(api_key, 'API_KEY')}")
                        
                        # Request payload
                        payload = {
                            "contents": [
                                {
                                    "role": "user",
                                    "parts": [{"text": content}]
                                }
                            ],
                            "generationConfig": {
                                "temperature": 0.4,
                                "topK": 32,
                                "topP": 0.95,
                                "maxOutputTokens": 2048,
                            }
                        }
                        
                        # If first URL format, include model in payload
                        if "generateContent?key=" in model_url and not "/" + model_name + ":" in model_url:
                            payload["model"] = model_name
                        
                        # Make the API request
                        headers = {
                            "Content-Type": "application/json"
                        }
                        
                        response = requests.post(
                            model_url,
                            headers=headers,
                            json=payload,
                            timeout=30
                        )
                        
                        # Check response
                        if response.status_code == 200:
                            result = response.json()
                            content = result.get("candidates", [{}])[0].get("content", {})
                            parts = content.get("parts", [])
                            text = parts[0].get("text", "") if parts else ""
                            
                            print(f"✅ SUCCESS with {model_name} at {base_url}")
                            print(f"Response: {text[:200]}...")
                            print(f"Full response: {json.dumps(result, indent=2)[:500]}...")
                            
                            success = True
                            print("\n==== WORKING CONFIGURATION ====")
                            print(f"Base URL: {base_url}")
                            print(f"Model name: {model_name}")
                            print(f"Endpoint: {'model-specific' if '/' + model_name + ':' in model_url else 'generateContent with model param'}")
                            print("===============================")
                            return True
                        else:
                            print(f"❌ FAILED with status {response.status_code}")
                            print(f"Error: {response.text}")
                    except Exception as url_err:
                        print(f"❌ ERROR with URL {model_url.replace(api_key, 'API_KEY')}: {str(url_err)}")
            except Exception as model_err:
                print(f"❌ ERROR with model {model_name}: {str(model_err)}")
    
    if not success:
        print("\n❌ All attempts failed. Check your API key and model access.")
    return success

if __name__ == "__main__":
    success = test_gemini_direct()
    sys.exit(0 if success else 1)